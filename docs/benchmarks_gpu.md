## GPU-accelerated indices benchmarks and parameter gridsearch

Below are benchmarks shown for the GPU-accelerated code. If you wish to run
the example script, please use

```bash
cargo run --example gridsearch_gpu --features gpu
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
====================================================================================================
Benchmark: 150k cells, 32D (CPU vs GPU Exhaustive vs IVF-GPU)
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
CPU-Exhaustive                              3.48     24372.91     24376.38       1.0000     0.000000
GPU-Exhaustive                              3.27     15536.64     15539.92       1.0000     0.000003
IVF-GPU-kNN-nl273-np13                   1470.27      3163.37      4633.64       0.9954     0.036068
IVF-GPU-kNN-nl273-np16                   1470.27      3668.76      5139.03       0.9998     0.001564
IVF-GPU-kNN-nl273-np23                   1470.27      4714.30      6184.57       1.0000     0.000003
IVF-GPU-kNN-nl387-np19                   2216.07      3473.84      5689.91       0.9962     0.020323
IVF-GPU-kNN-nl387-np27                   2216.07      4536.40      6752.46       1.0000     0.000003
IVF-GPU-kNN-nl547-np23                   2926.25      3489.63      6415.87       0.9905     0.043297
IVF-GPU-kNN-nl547-np27                   2926.25      3636.37      6562.61       0.9971     0.009404
IVF-GPU-kNN-nl547-np33                   2926.25      4077.09      7003.34       0.9997     0.000619
----------------------------------------------------------------------------------------------------
```

**Cosine:**

```
====================================================================================================
Benchmark: 150k cells, 32D (CPU vs GPU Exhaustive vs IVF-GPU)
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
CPU-Exhaustive                              5.38     23923.51     23928.89       1.0000     0.000000
GPU-Exhaustive                              4.59     14661.81     14666.40       1.0000     0.000000
IVF-GPU-kNN-nl273-np13                   3911.37      3105.11      7016.47       0.9956     0.000021
IVF-GPU-kNN-nl273-np16                   3911.37      3485.04      7396.40       0.9998     0.000001
IVF-GPU-kNN-nl273-np23                   3911.37      4644.56      8555.93       1.0000     0.000000
IVF-GPU-kNN-nl273-np27                   3911.37      5304.09      9215.46       1.0000     0.000000
IVF-GPU-kNN-nl387-np19                   5494.76      3276.61      8771.36       0.9965     0.000012
IVF-GPU-kNN-nl387-np27                   5494.76      4256.59      9751.34       1.0000     0.000000
IVF-GPU-kNN-nl387-np38                   5494.76      5410.46     10905.22       1.0000     0.000000
IVF-GPU-kNN-nl547-np23                   7799.18      3458.07     11257.26       0.9913     0.000026
IVF-GPU-kNN-nl547-np27                   7799.18      3670.37     11469.55       0.9973     0.000006
IVF-GPU-kNN-nl547-np33                   7799.18      4144.41     11943.59       0.9997     0.000000
IVF-GPU-kNN-nl547-np54                   7799.18      5778.40     13577.59       1.0000     0.000000
----------------------------------------------------------------------------------------------------
```

### Comparison against IVF CPU

In this case, the IVF CPU implementation is being compared against the GPU 
version. GPU acceleration shines with larger data sets and larger dimensions, 
hence, the number of samples was increased to 250_000 and dimensions to 64 for 
these benchmarks.

**IVF CPU:**

```
====================================================================================================
Benchmark: 250k cells, 64D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                 11.49    174098.40    174109.89       1.0000     0.000000
IVF-nl353-np17                           7965.47     26373.37     34338.84       0.9731     0.215766
IVF-nl353-np18                           7965.47     28007.54     35973.01       0.9794     0.161612
IVF-nl353-np26                           7965.47     42428.59     50394.06       0.9998     0.000764
IVF-nl500-np22                          11198.37     26345.49     37543.86       0.9692     0.228525
IVF-nl500-np25                          11198.37     29088.33     40286.71       0.9828     0.114928
IVF-nl500-np31                          11198.37     34172.25     45370.62       0.9965     0.020768
IVF-nl707-np26                          14991.48     20896.10     35887.57       0.9496     0.423474
IVF-nl707-np35                          14991.48     27335.69     42327.17       0.9857     0.099977
IVF-nl707-np37                          14991.48     30396.85     45388.33       0.9900     0.067311
----------------------------------------------------------------------------------------------------
```

**IVF GPU:**

```
====================================================================================================
Benchmark: 250k cells, 64D (CPU vs GPU Exhaustive vs IVF-GPU)
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
CPU-Exhaustive                             11.81    172074.12    172085.93       1.0000     0.000000
GPU-Exhaustive                             11.40     48502.88     48514.28       1.0000     0.000005
IVF-GPU-kNN-nl353-np17                   7450.28      8518.02     15968.30       0.9731     0.216414
IVF-GPU-kNN-nl353-np18                   7450.28      8906.56     16356.84       0.9793     0.161999
IVF-GPU-kNN-nl353-np26                   7450.28     12083.07     19533.35       0.9998     0.000769
IVF-GPU-kNN-nl500-np22                  10970.30      8547.57     19517.87       0.9692     0.228530
IVF-GPU-kNN-nl500-np25                  10970.30      9625.96     20596.26       0.9828     0.114932
IVF-GPU-kNN-nl500-np31                  10970.30     11960.46     22930.76       0.9965     0.020773
IVF-GPU-kNN-nl707-np26                  15065.30      8704.64     23769.94       0.9496     0.423478
IVF-GPU-kNN-nl707-np35                  15065.30     10054.25     25119.56       0.9857     0.099982
IVF-GPU-kNN-nl707-np37                  15065.30     10617.65     25682.96       0.9900     0.067315
----------------------------------------------------------------------------------------------------
```

The results here are more favourable of the GPU acceleration. We go from 170
seconds with exhaustive search on CPU to ~48 seconds on GPU; if using IVF, we
can reduce at the fastest settings the time from ~35 seconds on CPU to ~16 
seconds on GPU. This gives us a total acceleration of 
**170 seconds *(exhaustive on CPU)* to 16 seconds *(IVF on GPU)***, a 10x 
acceleration while having a Recall ≥ 0.97. Results are becoming even more 
pronounced with more cells.

**IVF CPU (more cells):**

```
====================================================================================================
Benchmark: 500k cells, 64D
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
Exhaustive                                 22.97    676239.91    676262.88       1.0000     0.000000
IVF-nl500-np22                          20942.65    120145.16    141087.81       0.9747     0.211297
IVF-nl500-np25                          20942.65    135705.63    156648.28       0.9885     0.091280
IVF-nl500-np31                          20942.65    169026.13    189968.78       0.9994     0.005102
IVF-nl707-np26                          29563.90    102623.66    132187.57       0.9549     0.364671
IVF-nl707-np35                          29563.90    138084.40    167648.30       0.9902     0.069567
IVF-nl707-np37                          29563.90    144369.77    173933.67       0.9938     0.042671
IVF-nl1000-np31                         41588.25     87036.56    128624.81       0.9276     0.600609
IVF-nl1000-np44                         41588.25    121261.59    162849.84       0.9811     0.135394
IVF-nl1000-np50                         41588.25    138448.53    180036.78       0.9910     0.060453
----------------------------------------------------------------------------------------------------
```

**IVF GPU (more cells)**

```
====================================================================================================
Benchmark: 500k cells, 64D (CPU vs GPU Exhaustive vs IVF-GPU)
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
CPU-Exhaustive                             23.38    675656.14    675679.52       1.0000     0.000000
GPU-Exhaustive                             21.19    191814.09    191835.27       1.0000     0.000005
IVF-GPU-kNN-nl500-np22                  20887.08     26795.66     47682.74       0.9747     0.211300
IVF-GPU-kNN-nl500-np25                  20887.08     29818.20     50705.29       0.9885     0.091284
IVF-GPU-kNN-nl500-np31                  20887.08     35783.17     56670.26       0.9994     0.005107
IVF-GPU-kNN-nl707-np26                  29621.14     23818.10     53439.24       0.9549     0.364655
IVF-GPU-kNN-nl707-np35                  29621.14     31119.45     60740.59       0.9902     0.069590
IVF-GPU-kNN-nl707-np37                  29621.14     32673.51     62294.66       0.9938     0.042676
IVF-GPU-kNN-nl1000-np31                 41618.89     22620.53     64239.42       0.9276     0.600433
IVF-GPU-kNN-nl1000-np44                 41618.89     28410.96     70029.85       0.9811     0.135380
IVF-GPU-kNN-nl1000-np50                 41618.89     31406.06     73024.95       0.9910     0.060457
----------------------------------------------------------------------------------------------------
```

### Higher dimensionality

With even higher dimensionality, we can observe the advantage of the GPU-accelerated
versions. Particularly, the difference in the exhaustive search is already very
impressive with a 7x increase in querying speed. Basically, the more dimensions
you are dealing with, the better the GPU acceleration works.

```
====================================================================================================
Benchmark: 150k cells, 128D (CPU vs GPU Exhaustive vs IVF-GPU)
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
CPU-Exhaustive                             16.35    176066.56    176082.90       1.0000     0.000000
GPU-Exhaustive                             15.10     25935.82     25950.93       1.0000     0.000003
IVF-GPU-kNN-nl273-np13                   9980.38      4755.05     14735.43       0.9957     0.018309
IVF-GPU-kNN-nl273-np16                   9980.38      5383.63     15364.00       0.9993     0.002087
IVF-GPU-kNN-nl273-np23                   9980.38      7338.52     17318.90       1.0000     0.000003
IVF-GPU-kNN-nl387-np19                  14234.64      5642.01     19876.65       0.9981     0.009823
IVF-GPU-kNN-nl387-np27                  14234.64      6955.58     21190.22       1.0000     0.000003
IVF-GPU-kNN-nl547-np23                  20161.21      5964.91     26126.12       0.9961     0.019722
IVF-GPU-kNN-nl547-np27                  20161.21      6421.80     26583.01       0.9989     0.004981
IVF-GPU-kNN-nl547-np33                  20161.21      7179.69     27340.90       0.9999     0.000131
----------------------------------------------------------------------------------------------------
```

All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.