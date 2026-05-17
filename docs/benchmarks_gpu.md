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
Exhaustive (query)                                         3.10     1_490.34     1_493.44       1.0000          1.0000        18.31
Exhaustive (self)                                          3.10    15_875.79    15_878.89       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.07       652.35       657.42       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.07     5_432.47     5_437.54       1.0000          1.0000        18.31
IVF-GPU-nl273-np13 (query)                               406.16       296.61       702.77       0.9875          1.0010         1.15
IVF-GPU-nl273-np16 (query)                               406.16       344.77       750.93       0.9975          1.0002         1.15
IVF-GPU-nl273-np23 (query)                               406.16       437.02       843.19       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     406.16     1_633.01     2_039.17       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               788.33       332.11     1_120.45       0.9925          1.0006         1.15
IVF-GPU-nl387-np27 (query)                               788.33       408.94     1_197.28       0.9997          1.0000         1.15
IVF-GPU-nl387 (self)                                     788.33     1_406.01     2_194.34       0.9998          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             1_548.49       318.06     1_866.54       0.9888          1.0008         1.15
IVF-GPU-nl547-np27 (query)                             1_548.49       334.73     1_883.21       0.9969          1.0002         1.15
IVF-GPU-nl547-np33 (query)                             1_548.49       351.53     1_900.02       0.9999          1.0000         1.15
IVF-GPU-nl547 (self)                                   1_548.49     1_354.76     2_903.24       0.9998          1.0000         1.15
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
Exhaustive (query)                                         4.06     1_743.55     1_747.61       1.0000          1.0000        18.88
Exhaustive (self)                                          4.06    18_163.96    18_168.02       1.0000          1.0000        18.88
GPU-Exhaustive (query)                                     6.34       665.10       671.43       0.9999          1.0000        18.88
GPU-Exhaustive (self)                                      6.34     5_640.00     5_646.33       1.0000          1.0000        18.88
IVF-GPU-nl273-np13 (query)                               388.22       304.04       692.26       0.9881          1.0009         1.15
IVF-GPU-nl273-np16 (query)                               388.22       344.74       732.96       0.9975          1.0002         1.15
IVF-GPU-nl273-np23 (query)                               388.22       435.38       823.60       0.9999          1.0000         1.15
IVF-GPU-nl273 (self)                                     388.22     1_579.94     1_968.16       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               741.39       336.60     1_077.99       0.9930          1.0005         1.15
IVF-GPU-nl387-np27 (query)                               741.39       422.67     1_164.07       0.9997          1.0000         1.15
IVF-GPU-nl387 (self)                                     741.39     1_435.79     2_177.18       0.9997          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             1_466.82       310.45     1_777.27       0.9899          1.0007         1.15
IVF-GPU-nl547-np27 (query)                             1_466.82       342.87     1_809.69       0.9972          1.0002         1.15
IVF-GPU-nl547-np33 (query)                             1_466.82       352.92     1_819.75       0.9998          1.0000         1.15
IVF-GPU-nl547 (self)                                   1_466.82     1_355.48     2_822.30       0.9998          1.0000         1.15
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
Exhaustive (query)                                         3.19     1_586.37     1_589.56       1.0000          1.0000        18.31
Exhaustive (self)                                          3.19    16_533.57    16_536.76       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.67       647.38       653.05       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.67     5_433.44     5_439.10       1.0000          1.0000        18.31
IVF-GPU-nl273-np13 (query)                               402.58       290.35       692.93       0.9999          1.0000         1.15
IVF-GPU-nl273-np16 (query)                               402.58       239.80       642.38       1.0000          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               402.58       391.97       794.55       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     402.58     1_509.64     1_912.22       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               792.20       342.84     1_135.04       0.9999          1.0000         1.15
IVF-GPU-nl387-np27 (query)                               792.20       391.54     1_183.75       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                     792.20     1_307.41     2_099.62       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             1_506.70       294.54     1_801.24       1.0000          1.0000         1.15
IVF-GPU-nl547-np27 (query)                             1_506.70       334.02     1_840.72       1.0000          1.0000         1.15
IVF-GPU-nl547-np33 (query)                             1_506.70       342.98     1_849.68       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                   1_506.70     1_285.33     2_792.03       1.0000          1.0000         1.15
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
Exhaustive (query)                                         3.12     1_571.93     1_575.04       1.0000          1.0000        18.31
Exhaustive (self)                                          3.12    16_357.10    16_360.22       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.32       646.13       651.45       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.32     5_429.71     5_435.03       1.0000          1.0000        18.31
IVF-GPU-nl273-np13 (query)                               433.30       280.61       713.91       1.0000          1.0000         1.15
IVF-GPU-nl273-np16 (query)                               433.30       337.33       770.63       1.0000          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               433.30       404.55       837.85       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     433.30     1_494.02     1_927.33       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               780.37       328.30     1_108.67       1.0000          1.0000         1.15
IVF-GPU-nl387-np27 (query)                               780.37       396.13     1_176.50       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                     780.37     1_292.16     2_072.53       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             1_519.46       160.78     1_680.24       1.0000          1.0000         1.15
IVF-GPU-nl547-np27 (query)                             1_519.46       300.62     1_820.08       1.0000          1.0000         1.15
IVF-GPU-nl547-np33 (query)                             1_519.46       352.19     1_871.65       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                   1_519.46     1_232.05     2_751.51       1.0000          1.0000         1.15
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
Exhaustive (query)                                        14.53     6_031.04     6_045.56       1.0000          1.0000        73.24
Exhaustive (self)                                         14.53    63_238.51    63_253.03       1.0000          1.0000        73.24
GPU-Exhaustive (query)                                    21.87     1_358.40     1_380.27       1.0000          1.0000        73.24
GPU-Exhaustive (self)                                     21.87    12_524.28    12_546.15       1.0000          1.0000        73.24
IVF-GPU-nl273-np13 (query)                               513.86       449.54       963.40       0.9998          1.0000         1.15
IVF-GPU-nl273-np16 (query)                               513.86       518.44     1_032.30       1.0000          1.0000         1.15
IVF-GPU-nl273-np23 (query)                               513.86       632.98     1_146.84       1.0000          1.0000         1.15
IVF-GPU-nl273 (self)                                     513.86     3_949.50     4_463.36       1.0000          1.0000         1.15
IVF-GPU-nl387-np19 (query)                               981.26       456.50     1_437.77       1.0000          1.0000         1.15
IVF-GPU-nl387-np27 (query)                               981.26       577.29     1_558.56       1.0000          1.0000         1.15
IVF-GPU-nl387 (self)                                     981.26     3_403.70     4_384.96       1.0000          1.0000         1.15
IVF-GPU-nl547-np23 (query)                             1_932.84       331.27     2_264.11       1.0000          1.0000         1.15
IVF-GPU-nl547-np27 (query)                             1_932.84       457.87     2_390.71       1.0000          1.0000         1.15
IVF-GPU-nl547-np33 (query)                             1_932.84       559.38     2_492.22       1.0000          1.0000         1.15
IVF-GPU-nl547 (self)                                   1_932.84     3_138.13     5_070.96       1.0000          1.0000         1.15
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
Exhaustive (query)                                        11.61     5_031.20     5_042.81       1.0000          1.0000        61.04
Exhaustive (self)                                         11.61    87_295.80    87_307.41       1.0000          1.0000        61.04
IVF-nl353-np17 (query)                                   857.97       338.62     1_196.59       1.0000          1.0000        61.12
IVF-nl353-np18 (query)                                   857.97       358.24     1_216.21       1.0000          1.0000        61.12
IVF-nl353-np26 (query)                                   857.97       511.16     1_369.14       1.0000          1.0000        61.12
IVF-nl353 (self)                                         857.97     7_735.96     8_593.93       1.0000          1.0000        61.12
IVF-nl500-np22 (query)                                 1_429.95       322.88     1_752.84       1.0000          1.0000        61.16
IVF-nl500-np25 (query)                                 1_429.95       359.46     1_789.41       1.0000          1.0000        61.16
IVF-nl500-np31 (query)                                 1_429.95       449.69     1_879.64       1.0000          1.0000        61.16
IVF-nl500 (self)                                       1_429.95     6_405.49     7_835.44       1.0000          1.0000        61.16
IVF-nl707-np26 (query)                                 2_800.76       282.68     3_083.43       1.0000          1.0000        61.21
IVF-nl707-np35 (query)                                 2_800.76       368.62     3_169.38       1.0000          1.0000        61.21
IVF-nl707-np37 (query)                                 2_800.76       389.31     3_190.07       1.0000          1.0000        61.21
IVF-nl707 (self)                                       2_800.76     5_084.10     7_884.86       1.0000          1.0000        61.21
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
Exhaustive (query)                                        11.05     5_212.15     5_223.20       1.0000          1.0000        61.04
Exhaustive (self)                                         11.05    85_791.13    85_802.18       1.0000          1.0000        61.04
GPU-Exhaustive (query)                                    17.41     1_418.34     1_435.76       1.0000          1.0000        61.04
GPU-Exhaustive (self)                                     17.41    21_507.17    21_524.58       1.0000          1.0000        61.04
IVF-GPU-nl353-np17 (query)                               743.25       429.35     1_172.60       1.0000          1.0000         1.91
IVF-GPU-nl353-np18 (query)                               743.25       507.44     1_250.69       1.0000          1.0000         1.91
IVF-GPU-nl353-np26 (query)                               743.25       616.45     1_359.70       1.0000          1.0000         1.91
IVF-GPU-nl353 (self)                                     743.25     5_209.60     5_952.85       1.0000          1.0000         1.91
IVF-GPU-nl500-np22 (query)                             1_436.82       480.03     1_916.84       1.0000          1.0000         1.91
IVF-GPU-nl500-np25 (query)                             1_436.82       532.54     1_969.35       1.0000          1.0000         1.91
IVF-GPU-nl500-np31 (query)                             1_436.82       573.41     2_010.22       1.0000          1.0000         1.91
IVF-GPU-nl500 (self)                                   1_436.82     4_651.91     6_088.73       1.0000          1.0000         1.91
IVF-GPU-nl707-np26 (query)                             2_802.45       422.05     3_224.49       1.0000          1.0000         1.91
IVF-GPU-nl707-np35 (query)                             2_802.45       483.38     3_285.82       1.0000          1.0000         1.91
IVF-GPU-nl707-np37 (query)                             2_802.45       531.11     3_333.55       1.0000          1.0000         1.91
IVF-GPU-nl707 (self)                                   2_802.45     4_022.08     6_824.52       1.0000          1.0000         1.91
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
Exhaustive (query)                                        24.53    10_861.13    10_885.67       1.0000          1.0000       122.07
Exhaustive (self)                                         24.53   182_410.13   182_434.66       1.0000          1.0000       122.07
IVF-nl353-np17 (query)                                   836.27       700.85     1_537.12       1.0000          1.0000       122.25
IVF-nl353-np18 (query)                                   836.27       741.24     1_577.51       1.0000          1.0000       122.25
IVF-nl353-np26 (query)                                   836.27     1_063.05     1_899.32       1.0000          1.0000       122.25
IVF-nl353 (self)                                         836.27    16_994.03    17_830.30       1.0000          1.0000       122.25
IVF-nl500-np22 (query)                                 1_608.55       657.14     2_265.68       1.0000          1.0000       122.32
IVF-nl500-np25 (query)                                 1_608.55       740.87     2_349.41       1.0000          1.0000       122.32
IVF-nl500-np31 (query)                                 1_608.55       914.09     2_522.64       1.0000          1.0000       122.32
IVF-nl500 (self)                                       1_608.55    14_903.59    16_512.13       1.0000          1.0000       122.32
IVF-nl707-np26 (query)                                 3_318.37       568.89     3_887.25       0.9999          1.0000       122.42
IVF-nl707-np35 (query)                                 3_318.37       748.13     4_066.50       1.0000          1.0000       122.42
IVF-nl707-np37 (query)                                 3_318.37       798.04     4_116.41       1.0000          1.0000       122.42
IVF-nl707 (self)                                       3_318.37    12_711.35    16_029.72       1.0000          1.0000       122.42
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
Exhaustive (query)                                        24.41    10_722.73    10_747.14       1.0000          1.0000       122.07
Exhaustive (self)                                         24.41   186_327.15   186_351.55       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                    41.63     2_223.07     2_264.70       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                     41.63    34_757.09    34_798.71       1.0000          1.0000       122.07
IVF-GPU-nl353-np17 (query)                               867.85       606.47     1_474.33       1.0000          1.0000         1.91
IVF-GPU-nl353-np18 (query)                               867.85       631.85     1_499.70       1.0000          1.0000         1.91
IVF-GPU-nl353-np26 (query)                               867.85       819.56     1_687.41       1.0000          1.0000         1.91
IVF-GPU-nl353 (self)                                     867.85     9_186.25    10_054.10       1.0000          1.0000         1.91
IVF-GPU-nl500-np22 (query)                             1_683.96       644.35     2_328.31       1.0000          1.0000         1.91
IVF-GPU-nl500-np25 (query)                             1_683.96       655.03     2_338.99       1.0000          1.0000         1.91
IVF-GPU-nl500-np31 (query)                             1_683.96       788.81     2_472.77       1.0000          1.0000         1.91
IVF-GPU-nl500 (self)                                   1_683.96     8_079.64     9_763.60       1.0000          1.0000         1.91
IVF-GPU-nl707-np26 (query)                             3_400.09       591.04     3_991.13       0.9999          1.0000         1.91
IVF-GPU-nl707-np35 (query)                             3_400.09       642.32     4_042.42       1.0000          1.0000         1.91
IVF-GPU-nl707-np37 (query)                             3_400.09       672.59     4_072.68       1.0000          1.0000         1.91
IVF-GPU-nl707 (self)                                   3_400.09     6_960.78    10_360.87       1.0000          1.0000         1.91
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
Exhaustive (query)                                        20.25    12_004.03    12_024.28       1.0000          1.0000       122.07
Exhaustive (self)                                         20.25   377_461.32   377_481.56       1.0000          1.0000       122.07
IVF-nl500-np22 (query)                                 1_638.10       651.13     2_289.22       1.0000          1.0000       122.20
IVF-nl500-np25 (query)                                 1_638.10       729.15     2_367.25       1.0000          1.0000       122.20
IVF-nl500-np31 (query)                                 1_638.10       883.97     2_522.07       1.0000          1.0000       122.20
IVF-nl500 (self)                                       1_638.10    28_048.93    29_687.03       1.0000          1.0000       122.20
IVF-nl707-np26 (query)                                 3_029.62       575.98     3_605.60       1.0000          1.0000       122.25
IVF-nl707-np35 (query)                                 3_029.62       752.80     3_782.41       1.0000          1.0000       122.25
IVF-nl707-np37 (query)                                 3_029.62       803.02     3_832.64       1.0000          1.0000       122.25
IVF-nl707 (self)                                       3_029.62    25_004.18    28_033.80       1.0000          1.0000       122.25
IVF-nl1000-np31 (query)                                5_690.08       501.64     6_191.71       0.9999          1.0000       122.32
IVF-nl1000-np44 (query)                                5_690.08       686.79     6_376.86       1.0000          1.0000       122.32
IVF-nl1000-np50 (query)                                5_690.08       775.74     6_465.81       1.0000          1.0000       122.32
IVF-nl1000 (self)                                      5_690.08    21_261.99    26_952.07       1.0000          1.0000       122.32
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
Exhaustive (query)                                        20.73    11_962.77    11_983.50       1.0000          1.0000       122.07
Exhaustive (self)                                         20.73   380_898.51   380_919.25       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                    32.90     2_702.32     2_735.22       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                     32.90    85_719.37    85_752.26       1.0000          1.0000       122.07
IVF-GPU-nl500-np22 (query)                             1_754.94       624.08     2_379.03       1.0000          1.0000         3.82
IVF-GPU-nl500-np25 (query)                             1_754.94       658.67     2_413.62       1.0000          1.0000         3.82
IVF-GPU-nl500-np31 (query)                             1_754.94       745.75     2_500.69       1.0000          1.0000         3.82
IVF-GPU-nl500 (self)                                   1_754.94    16_003.21    17_758.16       1.0000          1.0000         3.82
IVF-GPU-nl707-np26 (query)                             3_456.22       586.18     4_042.40       1.0000          1.0000         3.82
IVF-GPU-nl707-np35 (query)                             3_456.22       663.04     4_119.26       1.0000          1.0000         3.82
IVF-GPU-nl707-np37 (query)                             3_456.22       666.17     4_122.40       1.0000          1.0000         3.82
IVF-GPU-nl707 (self)                                   3_456.22    14_162.98    17_619.21       1.0000          1.0000         3.82
IVF-GPU-nl1000-np31 (query)                            6_608.93       547.05     7_155.98       0.9999          1.0000         3.82
IVF-GPU-nl1000-np44 (query)                            6_608.93       612.69     7_221.62       1.0000          1.0000         3.82
IVF-GPU-nl1000-np50 (query)                            6_608.93       651.56     7_260.49       1.0000          1.0000         3.82
IVF-GPU-nl1000 (self)                                  6_608.93    12_156.71    18_765.64       1.0000          1.0000         3.82
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
Exhaustive (query)                                        46.48    25_019.49    25_065.97       1.0000          1.0000       244.14
Exhaustive (self)                                         46.48   844_703.70   844_750.18       1.0000          1.0000       244.14
IVF-nl500-np22 (query)                                 1_803.79     1_331.16     3_134.96       1.0000          1.0000       244.39
IVF-nl500-np25 (query)                                 1_803.79     1_490.06     3_293.86       1.0000          1.0000       244.39
IVF-nl500-np31 (query)                                 1_803.79     1_820.19     3_623.98       1.0000          1.0000       244.39
IVF-nl500 (self)                                       1_803.79    59_396.55    61_200.34       1.0000          1.0000       244.39
IVF-nl707-np26 (query)                                 3_510.81     1_156.75     4_667.56       0.9999          1.0000       244.49
IVF-nl707-np35 (query)                                 3_510.81     1_515.91     5_026.72       1.0000          1.0000       244.49
IVF-nl707-np37 (query)                                 3_510.81     1_593.02     5_103.83       1.0000          1.0000       244.49
IVF-nl707 (self)                                       3_510.81    51_831.41    55_342.22       1.0000          1.0000       244.49
IVF-nl1000-np31 (query)                                8_041.71       994.47     9_036.17       0.9998          1.0000       244.64
IVF-nl1000-np44 (query)                                8_041.71     1_373.45     9_415.15       1.0000          1.0000       244.64
IVF-nl1000-np50 (query)                                8_041.71     1_538.21     9_579.92       1.0000          1.0000       244.64
IVF-nl1000 (self)                                      8_041.71    44_286.38    52_328.08       1.0000          1.0000       244.64
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
Exhaustive (query)                                        46.12    25_995.19    26_041.30       1.0000          1.0000       244.14
Exhaustive (self)                                         46.12   845_022.81   845_068.92       1.0000          1.0000       244.14
GPU-Exhaustive (query)                                    71.53     4_338.34     4_409.86       1.0000          1.0000       244.14
GPU-Exhaustive (self)                                     71.53   138_443.01   138_514.54       1.0000          1.0000       244.14
IVF-GPU-nl500-np22 (query)                             1_873.79       991.28     2_865.07       1.0000          1.0000         3.82
IVF-GPU-nl500-np25 (query)                             1_873.79       986.67     2_860.46       1.0000          1.0000         3.82
IVF-GPU-nl500-np31 (query)                             1_873.79     1_199.58     3_073.37       1.0000          1.0000         3.82
IVF-GPU-nl500 (self)                                   1_873.79    29_442.34    31_316.13       1.0000          1.0000         3.82
IVF-GPU-nl707-np26 (query)                             3_611.91       910.61     4_522.53       0.9999          1.0000         3.82
IVF-GPU-nl707-np35 (query)                             3_611.91     1_008.25     4_620.17       1.0000          1.0000         3.82
IVF-GPU-nl707-np37 (query)                             3_611.91     1_027.99     4_639.90       1.0000          1.0000         3.82
IVF-GPU-nl707 (self)                                   3_611.91    25_670.11    29_282.02       1.0000          1.0000         3.82
IVF-GPU-nl1000-np31 (query)                            8_093.65       790.33     8_883.98       0.9998          1.0000         3.82
IVF-GPU-nl1000-np44 (query)                            8_093.65       983.06     9_076.71       1.0000          1.0000         3.82
IVF-GPU-nl1000-np50 (query)                            8_093.65     1_016.74     9_110.39       1.0000          1.0000         3.82
IVF-GPU-nl1000 (self)                                  8_093.65    22_158.08    30_251.73       1.0000          1.0000         3.82
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
CPU-Exhaustive (query)                                     3.34     1_747.87     1_751.21       1.0000          1.0000        18.31
CPU-Exhaustive (self)                                      3.34    17_760.31    17_763.65       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.49       645.19       650.68       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.49     5_431.75     5_437.24       1.0000          1.0000        18.31
CAGRA-auto (query)                                       600.09       129.16       729.25       0.9411          1.0034        86.98
CAGRA-auto (self)                                        600.09       713.67     1_313.76       0.9377          1.0048        86.98
CAGRA-bw16 (query)                                       600.09        85.92       686.01       0.9194          1.0046        86.98
CAGRA-bw16 (self)                                        600.09       328.91       929.00       0.9149          1.0068        86.98
CAGRA-bw30 (query)                                       600.09       114.38       714.48       0.9389          1.0035        86.98
CAGRA-bw30 (self)                                        600.09       656.04     1_256.13       0.9356          1.0050        86.98
CAGRA-bw48 (query)                                       600.09       318.00       918.09       0.9559          1.0025        86.98
CAGRA-bw48 (self)                                        600.09     1_240.44     1_840.53       0.9530          1.0037        86.98
CAGRA-bw64 (query)                                       600.09       270.85       870.94       0.9650          1.0020        86.98
CAGRA-bw64 (self)                                        600.09     1_913.07     2_513.16       0.9626          1.0030        86.98
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
CPU-Exhaustive (query)                                     3.87     1_644.54     1_648.40       1.0000          1.0000        18.88
CPU-Exhaustive (self)                                      3.87    17_243.40    17_247.27       1.0000          1.0000        18.88
GPU-Exhaustive (query)                                     6.37       662.76       669.13       0.9999          1.0000        18.88
GPU-Exhaustive (self)                                      6.37     5_635.88     5_642.25       1.0000          1.0000        18.88
CAGRA-auto (query)                                       674.25       198.20       872.46       0.9419          1.0035        87.55
CAGRA-auto (self)                                        674.25       720.22     1_394.48       0.9395          1.0046        87.55
CAGRA-bw16 (query)                                       674.25       157.31       831.57       0.9202          1.0047        87.55
CAGRA-bw16 (self)                                        674.25       338.40     1_012.66       0.9170          1.0065        87.55
CAGRA-bw30 (query)                                       674.25       167.18       841.44       0.9399          1.0036        87.55
CAGRA-bw30 (self)                                        674.25       662.56     1_336.82       0.9373          1.0048        87.55
CAGRA-bw48 (query)                                       674.25       194.32       868.57       0.9563          1.0026        87.55
CAGRA-bw48 (self)                                        674.25     1_245.11     1_919.36       0.9544          1.0035        87.55
CAGRA-bw64 (query)                                       674.25       314.50       988.76       0.9655          1.0020        87.55
CAGRA-bw64 (self)                                        674.25     1_915.05     2_589.31       0.9639          1.0028        87.55
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
CPU-Exhaustive (query)                                     3.07     1_668.55     1_671.62       1.0000          1.0000        18.31
CPU-Exhaustive (self)                                      3.07    17_490.05    17_493.13       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     5.13       645.60       650.72       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      5.13     5_430.92     5_436.05       1.0000          1.0000        18.31
CAGRA-auto (query)                                       548.76       113.88       662.64       0.9952          1.0002        86.98
CAGRA-auto (self)                                        548.76       646.14     1_194.90       0.9970          1.0002        86.98
CAGRA-bw16 (query)                                       548.76        83.22       631.97       0.9869          1.0006        86.98
CAGRA-bw16 (self)                                        548.76       314.51       863.27       0.9925          1.0004        86.98
CAGRA-bw30 (query)                                       548.76       116.02       664.78       0.9946          1.0003        86.98
CAGRA-bw30 (self)                                        548.76       596.52     1_145.27       0.9967          1.0002        86.98
CAGRA-bw48 (query)                                       548.76       212.75       761.50       0.9976          1.0001        86.98
CAGRA-bw48 (self)                                        548.76     1_072.55     1_621.31       0.9985          1.0001        86.98
CAGRA-bw64 (query)                                       548.76       214.55       763.30       0.9987          1.0001        86.98
CAGRA-bw64 (self)                                        548.76     1_607.50     2_156.26       0.9991          1.0001        86.98
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
CPU-Exhaustive (query)                                     3.38     1_649.55     1_652.92       1.0000          1.0000        18.31
CPU-Exhaustive (self)                                      3.38    17_281.09    17_284.47       1.0000          1.0000        18.31
GPU-Exhaustive (query)                                     4.97       638.32       643.29       1.0000          1.0000        18.31
GPU-Exhaustive (self)                                      4.97     5_428.44     5_433.41       1.0000          1.0000        18.31
CAGRA-auto (query)                                       555.07       113.05       668.12       0.9985          1.0001        86.98
CAGRA-auto (self)                                        555.07       639.72     1_194.80       0.9986          1.0001        86.98
CAGRA-bw16 (query)                                       555.07        83.67       638.74       0.9953          1.0002        86.98
CAGRA-bw16 (self)                                        555.07       314.78       869.85       0.9957          1.0003        86.98
CAGRA-bw30 (query)                                       555.07       108.06       663.13       0.9984          1.0001        86.98
CAGRA-bw30 (self)                                        555.07       588.16     1_143.23       0.9984          1.0001        86.98
CAGRA-bw48 (query)                                       555.07       154.66       709.74       0.9994          1.0000        86.98
CAGRA-bw48 (self)                                        555.07     1_050.80     1_605.88       0.9994          1.0001        86.98
CAGRA-bw64 (query)                                       555.07       268.96       824.04       0.9997          1.0000        86.98
CAGRA-bw64 (self)                                        555.07     1_569.79     2_124.86       0.9997          1.0000        86.98
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
CPU-Exhaustive (query)                                    14.42     6_233.70     6_248.11       1.0000          1.0000        73.24
CPU-Exhaustive (self)                                     14.42    63_595.59    63_610.00       1.0000          1.0000        73.24
GPU-Exhaustive (query)                                    21.72     1_362.12     1_383.84       1.0000          1.0000        73.24
GPU-Exhaustive (self)                                     21.72    12_536.26    12_557.98       1.0000          1.0000        73.24
CAGRA-auto (query)                                     3_091.03       278.81     3_369.83       0.9940          1.0003       141.91
CAGRA-auto (self)                                      3_091.03       803.05     3_894.08       0.9933          1.0005       141.91
CAGRA-bw16 (query)                                     3_091.03       191.81     3_282.84       0.9875          1.0006       141.91
CAGRA-bw16 (self)                                      3_091.03       416.75     3_507.77       0.9856          1.0009       141.91
CAGRA-bw30 (query)                                     3_091.03       227.43     3_318.46       0.9935          1.0004       141.91
CAGRA-bw30 (self)                                      3_091.03       749.17     3_840.20       0.9927          1.0005       141.91
CAGRA-bw48 (query)                                     3_091.03       328.87     3_419.90       0.9969          1.0002       141.91
CAGRA-bw48 (self)                                      3_091.03     1_307.26     4_398.28       0.9964          1.0003       141.91
CAGRA-bw64 (query)                                     3_091.03       345.06     3_436.09       0.9983          1.0001       141.91
CAGRA-bw64 (self)                                      3_091.03     1_916.87     5_007.90       0.9978          1.0002       141.91
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
CPU-Exhaustive (query)                                    11.53     8_382.56     8_394.08       1.0000          1.0000        61.04
CPU-Exhaustive (self)                                     11.53    85_987.30    85_998.83       1.0000          1.0000        61.04
GPU-Exhaustive (query)                                    17.65     2_323.15     2_340.80       1.0000          1.0000        61.04
GPU-Exhaustive (self)                                     17.65    21_505.58    21_523.23       1.0000          1.0000        61.04
CAGRA-auto (query)                                     2_499.17       393.74     2_892.90       0.9949          1.0002       175.48
CAGRA-auto (self)                                      2_499.17     1_203.85     3_703.02       0.9939          1.0004       175.48
CAGRA-bw16 (query)                                     2_499.17       278.91     2_778.08       0.9885          1.0005       175.48
CAGRA-bw16 (self)                                      2_499.17       596.05     3_095.22       0.9865          1.0008       175.48
CAGRA-bw30 (query)                                     2_499.17       331.20     2_830.37       0.9944          1.0003       175.48
CAGRA-bw30 (self)                                      2_499.17     1_109.15     3_608.31       0.9933          1.0004       175.48
CAGRA-bw48 (query)                                     2_499.17       474.51     2_973.67       0.9974          1.0001       175.48
CAGRA-bw48 (self)                                      2_499.17     1_977.83     4_476.99       0.9969          1.0002       175.48
CAGRA-bw64 (query)                                     2_499.17       539.34     3_038.50       0.9984          1.0001       175.48
CAGRA-bw64 (self)                                      2_499.17     2_963.72     5_462.88       0.9981          1.0001       175.48
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
CPU-Exhaustive (query)                                    24.74    17_970.77    17_995.51       1.0000          1.0000       122.07
CPU-Exhaustive (self)                                     24.74   184_524.86   184_549.60       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                    37.14     3_641.99     3_679.13       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                     37.14    34_726.36    34_763.50       1.0000          1.0000       122.07
CAGRA-auto (query)                                     6_760.15       549.89     7_310.04       0.9927          1.0004       236.51
CAGRA-auto (self)                                      6_760.15     1_367.11     8_127.26       0.9910          1.0006       236.51
CAGRA-bw16 (query)                                     6_760.15       488.00     7_248.14       0.9851          1.0008       236.51
CAGRA-bw16 (self)                                      6_760.15       710.19     7_470.33       0.9818          1.0011       236.51
CAGRA-bw30 (query)                                     6_760.15       534.46     7_294.61       0.9921          1.0004       236.51
CAGRA-bw30 (self)                                      6_760.15     1_274.41     8_034.55       0.9903          1.0006       236.51
CAGRA-bw48 (query)                                     6_760.15       682.34     7_442.49       0.9960          1.0002       236.51
CAGRA-bw48 (self)                                      6_760.15     2_215.79     8_975.94       0.9951          1.0003       236.51
CAGRA-bw64 (query)                                     6_760.15       766.41     7_526.56       0.9975          1.0001       236.51
CAGRA-bw64 (self)                                      6_760.15     3_266.78    10_026.92       0.9969          1.0002       236.51
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
CPU-Exhaustive (query)                                    20.71    39_426.47    39_447.18       1.0000          1.0000       122.07
CPU-Exhaustive (self)                                     20.71   379_581.81   379_602.51       1.0000          1.0000       122.07
GPU-Exhaustive (query)                                    35.30     8_722.21     8_757.51       1.0000          1.0000       122.07
GPU-Exhaustive (self)                                     35.30    85_683.36    85_718.65       1.0000          1.0000       122.07
CAGRA-auto (query)                                     6_350.42       707.04     7_057.46       0.9912          1.0005       350.95
CAGRA-auto (self)                                      6_350.42     2_449.18     8_799.60       0.9894          1.0007       350.95
CAGRA-bw16 (query)                                     6_350.42       637.67     6_988.09       0.9826          1.0009       350.95
CAGRA-bw16 (self)                                      6_350.42     1_214.88     7_565.30       0.9796          1.0013       350.95
CAGRA-bw30 (query)                                     6_350.42       735.75     7_086.17       0.9905          1.0005       350.95
CAGRA-bw30 (self)                                      6_350.42     2_265.66     8_616.08       0.9886          1.0007       350.95
CAGRA-bw48 (query)                                     6_350.42       878.03     7_228.45       0.9950          1.0003       350.95
CAGRA-bw48 (self)                                      6_350.42     4_054.90    10_405.32       0.9939          1.0004       350.95
CAGRA-bw64 (query)                                     6_350.42     1_074.88     7_425.30       0.9969          1.0002       350.95
CAGRA-bw64 (self)                                      6_350.42     6_080.96    12_431.38       0.9961          1.0003       350.95
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
CPU-Exhaustive (query)                                    46.46    84_421.08    84_467.54       1.0000          1.0000       244.14
CPU-Exhaustive (self)                                     46.46   842_065.57   842_112.03       1.0000          1.0000       244.14
GPU-Exhaustive (query)                                    75.06    14_070.72    14_145.78       1.0000          1.0000       244.14
GPU-Exhaustive (self)                                     75.06   138_281.98   138_357.04       1.0000          1.0000       244.14
CAGRA-auto (query)                                    16_892.52     1_180.45    18_072.97       0.9878          1.0007       473.02
CAGRA-auto (self)                                     16_892.52     2_773.19    19_665.71       0.9851          1.0010       473.02
CAGRA-bw16 (query)                                    16_892.52     1_043.82    17_936.34       0.9777          1.0012       473.02
CAGRA-bw16 (self)                                     16_892.52     1_426.15    18_318.67       0.9733          1.0018       473.02
CAGRA-bw30 (query)                                    16_892.52     1_175.57    18_068.09       0.9869          1.0008       473.02
CAGRA-bw30 (self)                                     16_892.52     2_576.87    19_469.39       0.9841          1.0011       473.02
CAGRA-bw48 (query)                                    16_892.52     1_346.24    18_238.76       0.9925          1.0004       473.02
CAGRA-bw48 (self)                                     16_892.52     4_507.87    21_400.39       0.9910          1.0006       473.02
CAGRA-bw64 (query)                                    16_892.52     1_526.07    18_418.59       0.9950          1.0003       473.02
CAGRA-bw64 (self)                                     16_892.52     6_657.65    23_550.17       0.9940          1.0004       473.02
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
GPU-Exhaustive (ground truth)                             10.12    15_011.15    15_021.27       1.0000          1.0000        30.52
CPU-NNDescent (k=15)                                   4_798.12     1_126.21     5_924.33       1.0000          1.0000       279.88
GPU-NND bk=1x refine=0 (extract)                         821.78        41.94       863.73       0.8934          1.0889       144.96
GPU-NND bk=1x refine=0 (self-beam)                       821.78     1_130.89     1_952.67       0.9954          1.0004       144.96
GPU-NND bk=1x refine=1 (extract)                         867.69        41.94       909.63       0.9227          1.0848       144.96
GPU-NND bk=1x refine=1 (self-beam)                       867.69     1_076.07     1_943.76       0.9960          1.0003       144.96
GPU-NND bk=1x refine=2 (extract)                         909.39        42.90       952.30       0.9242          1.0847       144.96
GPU-NND bk=1x refine=2 (self-beam)                       909.39     1_076.65     1_986.05       0.9961          1.0003       144.96
GPU-NND bk=2x refine=0 (extract)                       1_191.47        41.91     1_233.38       0.9267          1.0844       144.96
GPU-NND bk=2x refine=0 (self-beam)                     1_191.47     1_079.95     2_271.42       0.9980          1.0001       144.96
GPU-NND bk=2x refine=1 (extract)                       1_499.71        42.10     1_541.81       0.9329          1.0837       144.96
GPU-NND bk=2x refine=1 (self-beam)                     1_499.71     1_083.01     2_582.72       0.9984          1.0001       144.96
GPU-NND bk=2x refine=2 (extract)                       1_805.75        41.87     1_847.62       0.9330          1.0837       144.96
GPU-NND bk=2x refine=2 (self-beam)                     1_805.75     1_080.40     2_886.15       0.9984          1.0001       144.96
GPU-NND bk=3x refine=0 (extract)                       2_155.73        41.47     2_197.20       0.9305          1.0840       144.96
GPU-NND bk=3x refine=0 (self-beam)                     2_155.73     1_082.03     3_237.76       0.9983          1.0001       144.96
GPU-NND bk=3x refine=1 (extract)                       2_840.53        42.10     2_882.63       0.9333          1.0837       144.96
GPU-NND bk=3x refine=1 (self-beam)                     2_840.53     1_086.19     3_926.72       0.9986          1.0001       144.96
GPU-NND bk=3x refine=2 (extract)                       3_546.53        42.10     3_588.63       0.9333          1.0837       144.96
GPU-NND bk=3x refine=2 (self-beam)                     3_546.53     1_084.78     4_631.31       0.9985          1.0001       144.96
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
GPU-Exhaustive (ground truth)                             21.18    21_451.10    21_472.29       1.0000          1.0000        61.04
CPU-NNDescent (k=15)                                   6_108.77     1_594.76     7_703.53       1.0000          1.0000       377.92
GPU-NND bk=1x refine=0 (extract)                       1_199.15        42.27     1_241.42       0.8651          1.0910       175.48
GPU-NND bk=1x refine=0 (self-beam)                     1_199.15     1_237.50     2_436.65       0.9882          1.0010       175.48
GPU-NND bk=1x refine=1 (extract)                       1_645.54        41.73     1_687.27       0.9105          1.0848       175.48
GPU-NND bk=1x refine=1 (self-beam)                     1_645.54     1_190.72     2_836.26       0.9896          1.0009       175.48
GPU-NND bk=1x refine=2 (extract)                       2_115.99        42.16     2_158.15       0.9138          1.0844       175.48
GPU-NND bk=1x refine=2 (self-beam)                     2_115.99     1_192.13     3_308.12       0.9898          1.0008       175.48
GPU-NND bk=2x refine=0 (extract)                       1_995.01        42.50     2_037.52       0.9188          1.0838       175.48
GPU-NND bk=2x refine=0 (self-beam)                     1_995.01     1_190.95     3_185.96       0.9941          1.0003       175.48
GPU-NND bk=2x refine=1 (extract)                       3_590.57        41.32     3_631.89       0.9318          1.0825       175.48
GPU-NND bk=2x refine=1 (self-beam)                     3_590.57     1_193.76     4_784.33       0.9954          1.0002       175.48
GPU-NND bk=2x refine=2 (extract)                       5_203.18        42.06     5_245.24       0.9320          1.0824       175.48
GPU-NND bk=2x refine=2 (self-beam)                     5_203.18     1_194.93     6_398.11       0.9955          1.0002       175.48
GPU-NND bk=3x refine=0 (extract)                       4_622.47        41.49     4_663.96       0.9280          1.0828       175.48
GPU-NND bk=3x refine=0 (self-beam)                     4_622.47     1_199.56     5_822.04       0.9952          1.0002       175.48
GPU-NND bk=3x refine=1 (extract)                       7_489.01        42.18     7_531.19       0.9332          1.0823       175.48
GPU-NND bk=3x refine=1 (self-beam)                     7_489.01     1_198.01     8_687.02       0.9958          1.0002       175.48
GPU-NND bk=3x refine=2 (extract)                      10_364.82        41.91    10_406.72       0.9332          1.0823       175.48
GPU-NND bk=3x refine=2 (self-beam)                    10_364.82     1_195.62    11_560.43       0.9958          1.0002       175.48
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
GPU-Exhaustive (ground truth)                             19.21    59_446.35    59_465.56       1.0000          1.0000        61.04
CPU-NNDescent (k=15)                                  10_557.82     2_611.16    13_168.98       1.0000          1.0000       627.78
GPU-NND bk=1x refine=0 (extract)                       1_571.97        85.03     1_657.00       0.8738          1.0914       289.92
GPU-NND bk=1x refine=0 (self-beam)                     1_571.97     2_186.54     3_758.50       0.9916          1.0007       289.92
GPU-NND bk=1x refine=1 (extract)                       1_867.66        83.62     1_951.28       0.9149          1.0854       289.92
GPU-NND bk=1x refine=1 (self-beam)                     1_867.66     2_180.06     4_047.72       0.9928          1.0006       289.92
GPU-NND bk=1x refine=2 (extract)                       2_222.15        84.22     2_306.37       0.9178          1.0850       289.92
GPU-NND bk=1x refine=2 (self-beam)                     2_222.15     2_177.38     4_399.53       0.9930          1.0006       289.92
GPU-NND bk=2x refine=0 (extract)                       2_441.75        83.97     2_525.72       0.9229          1.0845       289.92
GPU-NND bk=2x refine=0 (self-beam)                     2_441.75     2_190.51     4_632.26       0.9966          1.0002       289.92
GPU-NND bk=2x refine=1 (extract)                       3_499.98        84.11     3_584.10       0.9324          1.0834       289.92
GPU-NND bk=2x refine=1 (self-beam)                     3_499.98     2_190.73     5_690.71       0.9974          1.0001       289.92
GPU-NND bk=2x refine=2 (extract)                       4_516.07        83.32     4_599.39       0.9326          1.0834       289.92
GPU-NND bk=2x refine=2 (self-beam)                     4_516.07     2_191.52     6_707.59       0.9974          1.0001       289.92
GPU-NND bk=3x refine=0 (extract)                       4_522.16        84.00     4_606.16       0.9294          1.0838       289.92
GPU-NND bk=3x refine=0 (self-beam)                     4_522.16     2_195.17     6_717.33       0.9972          1.0001       289.92
GPU-NND bk=3x refine=1 (extract)                       6_443.52        84.14     6_527.66       0.9332          1.0834       289.92
GPU-NND bk=3x refine=1 (self-beam)                     6_443.52     2_193.75     8_637.28       0.9976          1.0001       289.92
GPU-NND bk=3x refine=2 (extract)                       8_315.14        84.23     8_399.37       0.9333          1.0834       289.92
GPU-NND bk=3x refine=2 (self-beam)                     8_315.14     2_194.50    10_509.64       0.9976          1.0001       289.92
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
GPU-Exhaustive (ground truth)                             37.10    85_100.12    85_137.22       1.0000          1.0000       122.07
CPU-NNDescent (k=15)                                  14_367.49     4_096.71    18_464.20       1.0000          1.0000       793.86
GPU-NND bk=1x refine=0 (extract)                       2_486.94        85.37     2_572.31       0.8322          1.0957       350.95
GPU-NND bk=1x refine=0 (self-beam)                     2_486.94     2_424.22     4_911.17       0.9786          1.0019       350.95
GPU-NND bk=1x refine=1 (extract)                       3_822.38        84.30     3_906.68       0.8939          1.0864       350.95
GPU-NND bk=1x refine=1 (self-beam)                     3_822.38     2_415.06     6_237.43       0.9814          1.0016       350.95
GPU-NND bk=1x refine=2 (extract)                       5_207.09        84.40     5_291.48       0.9005          1.0856       350.95
GPU-NND bk=1x refine=2 (self-beam)                     5_207.09     2_412.32     7_619.41       0.9819          1.0015       350.95
GPU-NND bk=2x refine=0 (extract)                       4_219.73        84.32     4_304.05       0.9101          1.0845       350.95
GPU-NND bk=2x refine=0 (self-beam)                     4_219.73     2_413.76     6_633.49       0.9901          1.0006       350.95
GPU-NND bk=2x refine=1 (extract)                       9_164.64        84.58     9_249.22       0.9301          1.0823       350.95
GPU-NND bk=2x refine=1 (self-beam)                     9_164.64     2_410.06    11_574.70       0.9924          1.0004       350.95
GPU-NND bk=2x refine=2 (extract)                      14_257.30        84.50    14_341.80       0.9308          1.0823       350.95
GPU-NND bk=2x refine=2 (self-beam)                    14_257.30     2_412.26    16_669.55       0.9927          1.0004       350.95
GPU-NND bk=3x refine=0 (extract)                       9_852.96        84.22     9_937.18       0.9254          1.0828       350.95
GPU-NND bk=3x refine=0 (self-beam)                     9_852.96     2_419.07    12_272.03       0.9924          1.0004       350.95
GPU-NND bk=3x refine=1 (extract)                      18_738.78        84.56    18_823.34       0.9329          1.0821       350.95
GPU-NND bk=3x refine=1 (self-beam)                    18_738.78     2_423.55    21_162.32       0.9936          1.0003       350.95
GPU-NND bk=3x refine=2 (extract)                      27_899.47        84.47    27_983.94       0.9330          1.0821       350.95
GPU-NND bk=3x refine=2 (self-beam)                    27_899.47     2_413.57    30_313.04       0.9937          1.0003       350.95
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
GPU-Exhaustive (ground truth)                             40.52   236_835.93   236_876.45       1.0000          1.0000       122.07
CPU-NNDescent (k=15)                                  23_723.92     6_230.81    29_954.72       1.0000          1.0000      1143.55
GPU-NND bk=1x refine=0 (extract)                       3_334.02       171.02     3_505.04       0.8561          1.0938       579.83
GPU-NND bk=1x refine=0 (self-beam)                     3_334.02     4_488.35     7_822.37       0.9870          1.0011       579.83
GPU-NND bk=1x refine=1 (extract)                       4_278.77       169.36     4_448.13       0.9061          1.0862       579.83
GPU-NND bk=1x refine=1 (self-beam)                     4_278.77     4_462.98     8_741.75       0.9888          1.0009       579.83
GPU-NND bk=1x refine=2 (extract)                       5_347.54       169.30     5_516.84       0.9108          1.0856       579.83
GPU-NND bk=1x refine=2 (self-beam)                     5_347.54     4_415.60     9_763.14       0.9892          1.0009       579.83
GPU-NND bk=2x refine=0 (extract)                       5_193.33       167.42     5_360.75       0.9186          1.0847       579.83
GPU-NND bk=2x refine=0 (self-beam)                     5_193.33     4_442.66     9_635.99       0.9947          1.0003       579.83
GPU-NND bk=2x refine=1 (extract)                       8_395.73       167.98     8_563.71       0.9317          1.0832       579.83
GPU-NND bk=2x refine=1 (self-beam)                     8_395.73     4_429.86    12_825.59       0.9960          1.0002       579.83
GPU-NND bk=2x refine=2 (extract)                      11_659.66       169.52    11_829.17       0.9321          1.0832       579.83
GPU-NND bk=2x refine=2 (self-beam)                    11_659.66     4_435.41    16_095.07       0.9961          1.0002       579.83
GPU-NND bk=3x refine=0 (extract)                       9_290.83       167.77     9_458.60       0.9280          1.0836       579.83
GPU-NND bk=3x refine=0 (self-beam)                     9_290.83     4_443.88    13_734.71       0.9960          1.0002       579.83
GPU-NND bk=3x refine=1 (extract)                      15_313.22       166.47    15_479.69       0.9332          1.0831       579.83
GPU-NND bk=3x refine=1 (self-beam)                    15_313.22     4_460.90    19_774.13       0.9966          1.0001       579.83
GPU-NND bk=3x refine=2 (extract)                      21_027.00       167.05    21_194.05       0.9332          1.0831       579.83
GPU-NND bk=3x refine=2 (self-beam)                    21_027.00     4_451.00    25_478.00       0.9966          1.0001       579.83
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
GPU-Exhaustive (ground truth)                             69.76   338_893.41   338_963.17       1.0000          1.0000       244.14
CPU-NNDescent (k=15)                                  34_243.08     9_643.38    43_886.46       0.9999          1.0000      1659.75
GPU-NND bk=1x refine=0 (extract)                       4_995.61       169.52     5_165.12       0.8033          1.1002       701.90
GPU-NND bk=1x refine=0 (self-beam)                     4_995.61     4_995.08     9_990.69       0.9687          1.0028       701.90
GPU-NND bk=1x refine=1 (extract)                       8_977.04       165.65     9_142.68       0.8772          1.0884       701.90
GPU-NND bk=1x refine=1 (self-beam)                     8_977.04     4_924.70    13_901.74       0.9728          1.0024       701.90
GPU-NND bk=1x refine=2 (extract)                      13_014.04       170.08    13_184.12       0.8870          1.0871       701.90
GPU-NND bk=1x refine=2 (self-beam)                    13_014.04     4_886.09    17_900.14       0.9738          1.0022       701.90
GPU-NND bk=2x refine=0 (extract)                       8_866.41       165.99     9_032.40       0.9020          1.0853       701.90
GPU-NND bk=2x refine=0 (self-beam)                     8_866.41     4_957.87    13_824.28       0.9861          1.0009       701.90
GPU-NND bk=2x refine=1 (extract)                      23_061.75       165.91    23_227.66       0.9281          1.0824       701.90
GPU-NND bk=2x refine=1 (self-beam)                    23_061.75     4_938.20    27_999.96       0.9896          1.0006       701.90
GPU-NND bk=2x refine=2 (extract)                      37_447.52       165.78    37_613.30       0.9293          1.0822       701.90
GPU-NND bk=2x refine=2 (self-beam)                    37_447.52     4_891.03    42_338.56       0.9900          1.0005       701.90
GPU-NND bk=3x refine=0 (extract)                      21_116.55       181.45    21_297.99       0.9230          1.0829       701.90
GPU-NND bk=3x refine=0 (self-beam)                    21_116.55     4_932.12    26_048.66       0.9899          1.0005       701.90
GPU-NND bk=3x refine=1 (extract)                      46_768.52       166.33    46_934.84       0.9326          1.0819       701.90
GPU-NND bk=3x refine=1 (self-beam)                    46_768.52     4_957.94    51_726.45       0.9916          1.0004       701.90
GPU-NND bk=3x refine=2 (extract)                      72_403.27       167.74    72_571.02       0.9328          1.0819       701.90
GPU-NND bk=3x refine=2 (self-beam)                    72_403.27     4_923.57    77_326.84       0.9917          1.0003       701.90
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
GPU-Exhaustive (ground truth)                             84.96 1_471_849.52 1_471_934.48       1.0000          1.0000       305.18
CPU-NNDescent (k=15)                                  68_899.69    19_373.79    88_273.49       0.9999          1.0000      3254.84
GPU-NND bk=1x refine=0 (extract)                       7_300.28       427.11     7_727.39       0.7873          1.1048      1449.59
GPU-NND bk=1x refine=0 (self-beam)                     7_300.28    11_485.08    18_785.36       0.9671          1.0032      1449.59
GPU-NND bk=1x refine=1 (extract)                      10_857.48       440.14    11_297.62       0.8696          1.0903      1449.59
GPU-NND bk=1x refine=1 (self-beam)                    10_857.48    11_421.65    22_279.13       0.9724          1.0026      1449.59
GPU-NND bk=1x refine=2 (extract)                      14_236.39       451.38    14_687.77       0.8825          1.0884      1449.59
GPU-NND bk=1x refine=2 (self-beam)                    14_236.39    11_454.01    25_690.39       0.9738          1.0024      1449.59
GPU-NND bk=2x refine=0 (extract)                      12_630.28       432.52    13_062.80       0.9034          1.0858      1449.59
GPU-NND bk=2x refine=0 (self-beam)                    12_630.28    11_483.88    24_114.16       0.9884          1.0008      1449.59
GPU-NND bk=2x refine=1 (extract)                      24_035.25       429.25    24_464.50       0.9285          1.0828      1449.59
GPU-NND bk=2x refine=1 (self-beam)                    24_035.25    11_449.30    35_484.55       0.9918          1.0004      1449.59
GPU-NND bk=2x refine=2 (extract)                      35_405.03       430.25    35_835.28       0.9297          1.0827      1449.59
GPU-NND bk=2x refine=2 (self-beam)                    35_405.03    11_468.63    46_873.66       0.9922          1.0004      1449.59
GPU-NND bk=3x refine=0 (extract)                      24_194.99       428.77    24_623.76       0.9236          1.0833      1449.59
GPU-NND bk=3x refine=0 (self-beam)                    24_194.99    11_463.30    35_658.29       0.9921          1.0004      1449.59
GPU-NND bk=3x refine=1 (extract)                      44_798.88       429.78    45_228.66       0.9327          1.0824      1449.59
GPU-NND bk=3x refine=1 (self-beam)                    44_798.88    11_486.58    56_285.45       0.9936          1.0003      1449.59
GPU-NND bk=3x refine=2 (extract)                      65_242.86       424.09    65_666.94       0.9329          1.0824      1449.59
GPU-NND bk=3x refine=2 (self-beam)                    65_242.86    11_475.35    76_718.21       0.9937          1.0003      1449.59
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
