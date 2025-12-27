## GPU-accelerated indices benchmarks and parameter gridsearch

Below are benchmarks shown the 

### Comparison against CPU exhaustive

The GPU acceleration is notable already in the exhaustive index. The IVF-GPU
reaches very fast speeds here, but not much faster actually than the IVF-CPU
version. Due to the current implementation, there is quite a bit of overhead
in copying data from CPU to GPU during individual kernel launches. The 
advantages of the GPU versions are stronger when querying more samples at
higher dimensionality, see next section.

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
hence, the number of samples was increased to 250_000 and dimensions to 48 for 
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
Benchmark: 500k cells, 32D (CPU vs GPU Exhaustive vs IVF-GPU)
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
CPU-Exhaustive                             11.21    261010.65    261021.87       1.0000     0.000000
GPU-Exhaustive                             20.81    175500.07    175520.88       1.0000     0.000002
IVF-GPU-kNN-nl500-np22                   7979.49     24180.63     32160.12       0.9938     0.017569
IVF-GPU-kNN-nl500-np25                   7979.49     26730.53     34710.02       0.9978     0.004182
IVF-GPU-kNN-nl500-np31                   7979.49     30677.90     38657.39       0.9998     0.000206
IVF-GPU-kNN-nl500-np50                   7979.49     47844.94     55824.44       1.0000     0.000002
IVF-GPU-kNN-nl707-np26                  11900.13     22641.16     34541.29       0.9861     0.055257
IVF-GPU-kNN-nl707-np35                  11900.13     27722.34     39622.47       0.9981     0.005752
IVF-GPU-kNN-nl707-np37                  11900.13     29293.64     41193.77       0.9989     0.002852
IVF-GPU-kNN-nl707-np70                  11900.13     51075.16     62975.29       1.0000     0.000002
IVF-GPU-kNN-nl1000-np31                 16655.31     20986.76     37642.07       0.9787     0.093678
IVF-GPU-kNN-nl1000-np44                 16655.31     26519.50     43174.81       0.9972     0.008676
IVF-GPU-kNN-nl1000-np50                 16655.31     30444.66     47099.97       0.9992     0.001973
IVF-GPU-kNN-nl1000-np100                16655.31     52896.22     69551.53       1.0000     0.000002
```

### Higher dimensionality

With even higher dimensionality, we can observe the advantage of the GPU-accelerated
versions. Particularly, the difference in the exhaustive search is already very
impressive with a 7x increase in querying speed.

```
====================================================================================================
Benchmark: 150k cells, 128D (CPU vs GPU Exhaustive vs IVF-GPU)
====================================================================================================
Method                                Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error
----------------------------------------------------------------------------------------------------
CPU-Exhaustive                             15.25    194745.13    194760.37       1.0000     0.000000
GPU-Exhaustive                             17.07     26365.08     26382.14       1.0000     0.000003
IVF-GPU-kNN-nl273-np13                   9825.64      4803.13     14628.77       0.9957     0.018317
IVF-GPU-kNN-nl273-np16                   9825.64      5401.57     15227.21       0.9993     0.002087
IVF-GPU-kNN-nl273-np23                   9825.64      7152.16     16977.80       1.0000     0.000003
IVF-GPU-kNN-nl273-np27                   9825.64      7927.21     17752.85       1.0000     0.000003
IVF-GPU-kNN-nl387-np19                  14274.27      5790.95     20065.22       0.9981     0.009822
IVF-GPU-kNN-nl387-np27                  14274.27      7187.35     21461.62       1.0000     0.000003
IVF-GPU-kNN-nl387-np38                  14274.27      9105.42     23379.69       1.0000     0.000003
IVF-GPU-kNN-nl547-np23                  19754.76      5949.04     25703.80       0.9961     0.019664
IVF-GPU-kNN-nl547-np27                  19754.76      6208.95     25963.71       0.9989     0.004980
IVF-GPU-kNN-nl547-np33                  19754.76      7175.23     26929.99       0.9999     0.000131
IVF-GPU-kNN-nl547-np54                  19754.76     10194.08     29948.84       1.0000     0.000003
----------------------------------------------------------------------------------------------------
```

All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.