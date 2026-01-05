## GPU-accelerated indices benchmarks and parameter gridsearch

Below are benchmarks shown for the GPU-accelerated code. If you wish to run
the example script, please use:

```bash
cargo run --example gridsearch_gpu --features gpu --release
```

To run all of the benchmarks, you can just run:

```bash
bash examples/run_benchmarks.sh --gpu
```

### Comparison against CPU exhaustive

The GPU acceleration is notable already in the exhaustive index. The IVF-GPU
reaches very fast speeds here, but not much faster actually than the IVF-CPU
version. Due to the current implementation, there is quite a bit of overhead
in copying data from CPU to GPU during individual kernel launches. The 
advantages of the GPU versions are stronger when querying more samples at
higher dimensionality and more samples, see next section.

**Euclidean:**

```
==============================================================================================================
Benchmark: 150k cells, 32D (CPU vs GPU Exhaustive vs IVF-GPU)
==============================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
--------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    3.33      2314.06      2317.39       1.0000     0.000000
Exhaustive (self)                                     3.33     23161.76     23165.09       1.0000     0.000000
GPU-Exhaustive (query)                                3.61      1623.18      1626.79       1.0000     0.000001
GPU-Exhaustive (self)                                 3.61     14302.28     14305.89       1.0000     0.000001
IVF-GPU-nl273-np13 (query)                         1330.98       422.18      1753.17       0.9889     0.034524
IVF-GPU-nl273-np16 (query)                         1330.98       467.60      1798.58       0.9967     0.006610
IVF-GPU-nl273-np23 (query)                         1330.98       595.75      1926.74       1.0000     0.000001
IVF-GPU-nl273 (self)                               1330.98      4723.60      6054.58       1.0000     0.000001
IVF-GPU-nl387-np19 (query)                         1858.13       522.43      2380.56       0.9926     0.031090
IVF-GPU-nl387-np27 (query)                         1858.13       590.31      2448.45       0.9998     0.000654
IVF-GPU-nl387 (self)                               1858.13      4283.18      6141.32       0.9998     0.000692
IVF-GPU-nl547-np23 (query)                         2646.63       559.33      3205.96       0.9882     0.041945
IVF-GPU-nl547-np27 (query)                         2646.63       558.41      3205.05       0.9958     0.014726
IVF-GPU-nl547-np33 (query)                         2646.63       618.91      3265.55       0.9997     0.001037
IVF-GPU-nl547 (self)                               2646.63      4137.90      6784.53       0.9997     0.001004
--------------------------------------------------------------------------------------------------------------
```

**Cosine:**

```
==============================================================================================================
Benchmark: 150k cells, 32D (CPU vs GPU Exhaustive vs IVF-GPU)
==============================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
--------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                    4.30      2401.05      2405.35       1.0000     0.000000
Exhaustive (self)                                     4.30     25200.76     25205.06       1.0000     0.000000
GPU-Exhaustive (query)                                4.90      1884.93      1889.83       1.0000     0.000000
GPU-Exhaustive (self)                                 4.90     15899.72     15904.61       1.0000     0.000000
IVF-GPU-nl273-np13 (query)                         1405.84       421.80      1827.64       0.9894     0.000025
IVF-GPU-nl273-np16 (query)                         1405.84       459.87      1865.72       0.9968     0.000005
IVF-GPU-nl273-np23 (query)                         1405.84       595.19      2001.03       1.0000     0.000000
IVF-GPU-nl273 (self)                               1405.84      4745.21      6151.05       1.0000     0.000000
IVF-GPU-nl387-np19 (query)                         1934.81       504.04      2438.85       0.9930     0.000019
IVF-GPU-nl387-np27 (query)                         1934.81       582.69      2517.50       0.9998     0.000000
IVF-GPU-nl387 (self)                               1934.81      4195.22      6130.03       0.9998     0.000000
IVF-GPU-nl547-np23 (query)                         2713.71       525.51      3239.22       0.9891     0.000025
IVF-GPU-nl547-np27 (query)                         2713.71       537.67      3251.38       0.9962     0.000009
IVF-GPU-nl547-np33 (query)                         2713.71       602.22      3315.93       0.9996     0.000001
IVF-GPU-nl547 (self)                               2713.71      3903.81      6617.52       0.9997     0.000001
--------------------------------------------------------------------------------------------------------------
```

### Comparison against IVF CPU

In this case, the IVF CPU implementation is being compared against the GPU 
version. GPU acceleration shines with larger data sets and larger dimensions, 
hence, the number of samples was increased to 250_000 and dimensions to 64 for 
these benchmarks.

**IVF CPU:**

```
==============================================================================================================
Benchmark: 250k cells, 64D
==============================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
--------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                   11.71     10555.55     10567.27       1.0000     0.000000
Exhaustive (self)                                    11.71    164411.49    164423.20       1.0000     0.000000
IVF-nl353-np17 (query)                             7218.70      1378.19      8596.89       0.9864     0.155917
IVF-nl353-np18 (query)                             7218.70      1543.47      8762.17       0.9916     0.094062
IVF-nl353-np26 (query)                             7218.70      2101.89      9320.59       1.0000     0.000000
IVF-nl353 (self)                                   7218.70     34138.42     41357.11       1.0000     0.000000
IVF-nl500-np22 (query)                            10201.08      1392.97     11594.05       0.9848     0.145250
IVF-nl500-np25 (query)                            10201.08      1438.85     11639.93       0.9962     0.036920
IVF-nl500-np31 (query)                            10201.08      1765.53     11966.61       1.0000     0.000000
IVF-nl500 (self)                                  10201.08     29752.22     39953.30       1.0000     0.000000
IVF-nl707-np26 (query)                            14343.36      1090.68     15434.03       0.9588     0.408459
IVF-nl707-np35 (query)                            14343.36      1459.89     15803.25       0.9956     0.040789
IVF-nl707-np37 (query)                            14343.36      1512.90     15856.25       0.9981     0.017126
IVF-nl707 (self)                                  14343.36     25460.63     39803.99       0.9983     0.016073
--------------------------------------------------------------------------------------------------------------
```

**IVF GPU:**

```
==============================================================================================================
Benchmark: 250k cells, 64D (CPU vs GPU Exhaustive vs IVF-GPU)
==============================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
--------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                   11.17      9834.11      9845.29       1.0000     0.000000
Exhaustive (self)                                    11.17    168466.33    168477.51       1.0000     0.000000
GPU-Exhaustive (query)                               11.24      3007.39      3018.63       1.0000     0.000005
GPU-Exhaustive (self)                                11.24     46702.88     46714.12       1.0000     0.000005
IVF-GPU-nl353-np17 (query)                         7235.14       698.72      7933.86       0.9864     0.155944
IVF-GPU-nl353-np18 (query)                         7235.14       670.30      7905.44       0.9916     0.094067
IVF-GPU-nl353-np26 (query)                         7235.14       880.22      8115.37       1.0000     0.000005
IVF-GPU-nl353 (self)                               7235.14     11086.66     18321.80       1.0000     0.000005
IVF-GPU-nl500-np22 (query)                        10297.05       783.21     11080.26       0.9848     0.145255
IVF-GPU-nl500-np25 (query)                        10297.05       826.27     11123.32       0.9962     0.036926
IVF-GPU-nl500-np31 (query)                        10297.05       952.98     11250.03       1.0000     0.000005
IVF-GPU-nl500 (self)                              10297.05     10278.52     20575.56       1.0000     0.000005
IVF-GPU-nl707-np26 (query)                        14444.37       722.41     15166.78       0.9588     0.408607
IVF-GPU-nl707-np35 (query)                        14444.37       827.04     15271.41       0.9956     0.040794
IVF-GPU-nl707-np37 (query)                        14444.37       847.41     15291.79       0.9981     0.017132
IVF-GPU-nl707 (self)                              14444.37     10047.64     24492.01       0.9983     0.016078
--------------------------------------------------------------------------------------------------------------
```

The results here are more favourable of the GPU acceleration. We go from ~170
seconds with exhaustive search on CPU to ~50 seconds on GPU; if using IVF, we
can reduce at the fastest settings the time from ~40 seconds on CPU to ~20 
seconds on GPU. This gives us a total acceleration of 
**170 seconds *(exhaustive on CPU)* to 20 seconds *(IVF on GPU)***, a `8.5x` 
acceleration while having a Recall 1.00. Results are becoming even more 
pronounced with more cells.

**IVF CPU (more cells):**

We can appreciate that we are reaching scales in which exhaustive searches
are becoming prohibitively expensive. 

```
==============================================================================================================
Benchmark: 500k cells, 64D
==============================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
--------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                   21.96     19581.88     19603.84       1.0000     0.000000
Exhaustive (self)                                    21.96    668740.44    668762.40       1.0000     0.000000
IVF-nl500-np22 (query)                            20290.17      3944.69     24234.85       0.9864     0.122958
IVF-nl500-np25 (query)                            20290.17      4502.30     24792.47       0.9956     0.039509
IVF-nl500-np31 (query)                            20290.17      5112.93     25403.10       0.9997     0.003572
IVF-nl500 (self)                                  20290.17    173014.83    193305.00       0.9996     0.003517
IVF-nl707-np26 (query)                            29235.74      2923.84     32159.58       0.9666     0.281916
IVF-nl707-np35 (query)                            29235.74      3889.28     33125.02       0.9962     0.020123
IVF-nl707-np37 (query)                            29235.74      4097.72     33333.46       0.9981     0.007499
IVF-nl707 (self)                                  29235.74    147602.31    176838.05       0.9981     0.007531
IVF-nl1000-np31 (query)                           42866.80      2725.61     45592.42       0.9390     0.545610
IVF-nl1000-np44 (query)                           42866.80      3862.53     46729.34       0.9898     0.070742
IVF-nl1000-np50 (query)                           42866.80      4185.87     47052.67       0.9968     0.017600
IVF-nl1000 (self)                                 42866.80    121734.54    164601.35       0.9896     0.073495
--------------------------------------------------------------------------------------------------------------
```

**IVF GPU (more cells)**

```
==============================================================================================================
Benchmark: 500k cells, 64D (CPU vs GPU Exhaustive vs IVF-GPU)
==============================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
--------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                   21.42     20476.94     20498.36       1.0000     0.000000
Exhaustive (self)                                    21.42    673356.63    673378.05       1.0000     0.000000
GPU-Exhaustive (query)                               21.52      5844.45      5865.97       1.0000     0.000005
GPU-Exhaustive (self)                                21.52    186939.90    186961.41       1.0000     0.000005
IVF-GPU-nl500-np22 (query)                        20926.97      1127.02     22053.98       0.9862     0.123332
IVF-GPU-nl500-np25 (query)                        20926.97      1201.08     22128.05       0.9956     0.039545
IVF-GPU-nl500-np31 (query)                        20926.97      1433.27     22360.23       0.9997     0.003578
IVF-GPU-nl500 (self)                              20926.97     33455.74     54382.71       0.9996     0.003523
IVF-GPU-nl707-np26 (query)                        28800.60      1226.19     30026.79       0.9666     0.281920
IVF-GPU-nl707-np35 (query)                        28800.60      1479.87     30280.47       0.9962     0.020128
IVF-GPU-nl707-np37 (query)                        28800.60      1534.85     30335.45       0.9981     0.007504
IVF-GPU-nl707 (self)                              28800.60     31263.51     60064.11       0.9981     0.007536
IVF-GPU-nl1000-np31 (query)                       40795.10      1145.53     41940.63       0.9390     0.544478
IVF-GPU-nl1000-np44 (query)                       40795.10      1439.01     42234.11       0.9898     0.070914
IVF-GPU-nl1000-np50 (query)                       40795.10      1562.75     42357.84       0.9969     0.016519
IVF-GPU-nl1000 (self)                             40795.10     31678.21     72473.31       0.9896     0.073186
--------------------------------------------------------------------------------------------------------------
```

In this configuration, the exhaustive CPU to GPU already causes a massive
reduction in the generation of the full kNN. From ~11 minutes to ~3 minutes.
However, the true acceleleration comes with the IVF-GPU versions that reduces
the full kNN generation to a minute.

### Higher dimensionality

With even higher dimensionality, we can observe the advantage of the 
GPU-accelerated versions. Particularly, the difference in the exhaustive search 
is already very impressive with a 7x increase in querying speed. Basically, the 
more dimensions you are dealing with, the better the GPU acceleration works.

```
==============================================================================================================
Benchmark: 150k cells, 128D (CPU vs GPU Exhaustive vs IVF-GPU)
==============================================================================================================
Method                                          Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
--------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                   14.83     17216.10     17230.94       1.0000     0.000000
Exhaustive (self)                                    14.83    184119.86    184134.70       1.0000     0.000000
GPU-Exhaustive (query)                               15.84      2745.96      2761.80       1.0000     0.000003
GPU-Exhaustive (self)                                15.84     26674.69     26690.53       1.0000     0.000003
IVF-GPU-nl273-np13 (query)                         9818.04       657.51     10475.55       0.9934     0.025305
IVF-GPU-nl273-np16 (query)                         9818.04       784.92     10602.96       0.9984     0.006803
IVF-GPU-nl273-np23 (query)                         9818.04       980.51     10798.55       1.0000     0.000003
IVF-GPU-nl273 (self)                               9818.04      6906.68     16724.72       1.0000     0.000003
IVF-GPU-nl387-np19 (query)                        14258.34       766.71     15025.05       0.9957     0.015730
IVF-GPU-nl387-np27 (query)                        14258.34      1026.94     15285.29       0.9999     0.000534
IVF-GPU-nl387 (self)                              14258.34      6906.35     21164.70       0.9999     0.000532
IVF-GPU-nl547-np23 (query)                        19279.11       776.69     20055.80       0.9925     0.024024
IVF-GPU-nl547-np27 (query)                        19279.11       826.67     20105.78       0.9971     0.008552
IVF-GPU-nl547-np33 (query)                        19279.11       933.68     20212.79       0.9996     0.001042
IVF-GPU-nl547 (self)                              19279.11      6307.14     25586.25       0.9996     0.001054
--------------------------------------------------------------------------------------------------------------
```

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*