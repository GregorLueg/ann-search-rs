## Binarised indices benchmarks and parameter

Binarised indices compress the data stored in the index structure itself via
very aggressive quantisation to (basically) only bits. This has two impacts:

1. Drastic reduction in memory fingerprint of the index itself.
2. Increased query speed in most cases as the bit-wise operations are very
fast on modern CPUs.
3. However, when not using any re-ranking of the top candidates, dramatically
lower recall (less so for RaBitQ, an excellent way of compressing vectors).

The benchmarks below show scenarios with and without re-ranking.

```bash
cargo run --example gridsearch_binary --release --features binary
```

If you wish to run all of the benchmarks, below, you can just run:

```bash
bash ./examples/run_benchmarks.sh --binary
```

Similar to the other benchmarks, index building, query against 10% slightly
different data based on the trainings data and full kNN generation is being
benchmarked. Index size in memory is also provided. Compared to other
benchmarks, we will use the `"correlated"`, `"lowrank"` and `"quantisation"`
with higher dimensionality, but reduced samples (for the sake of fast'ish
benchmarking). The different synthetic data types pose different challenges
for the quantisation methods.

## Table of Contents

- [Binarisation](#binary-ivf-and-exhaustive)
- [RaBitQ](#rabitq-ivf-and-exhaustive)

### <u>Binary (IVF and exhaustive)</u>

Three binarisations are offered in this crate:

- **SimHash**: Projects vectors onto random hyperplanes and encodes the sign of
  each projection as a bit. The random planes are orthogonalised to improve
  coverage of the vector space.
- **PCA Hashing**: Uses PCA to find the axes of maximum variance in the data,
  then binarises by taking the sign of each data point's projection onto the
  top principal components. More expensive to build than SimHash but tends to
  yield better recall as the projections are data-adapted rather than random.
  If the number of requested bits exceeds the dimensionality, the excess bits
  are filled with random orthogonal projections.
- **Signed**: Simply encodes the sign of each embedding dimension directly as
  a bit, meaning n_bits is fixed to the number of dimensions. Straightforward
  but only sensible for high-dimensional data; at low dimensionality the recall
  degrades quickly.

These indices have the option to use a VecStore that saves the original data on
disk for fast retrieval and re-ranking. This is recommended if you wish to
maintain reasonable recall. Generally speaking, these indices shine in very
high-dimensional data where memory requirements become constraining.

**Key parameters *(general)*:**

- *n_bits*: Into how many bits to encode the data. The binariser has two
  different options here to generate the bits (more on that later). As one
  can appreciate the higher the number, the better the Recall.
- *binarisation_init*: Three options are provided in the crate. `"random"` that
  generates random planes that are subsequently orthogonalised, `"pca"` that
  leverages PCA to identify axis of maximum variation or `"signed"` that just
  uses the sign of the respective embedding dimensions. In this case, `n_bits`
  is set automatically to `n_dim`. Signed only really makes sense if you have
  a lot of dimensions; otherwise, the performance is not good (at all).
- *reranking*: The Binary indices have the option to store the original vectors
  on disk. Once Hamming distance has been leveraged to identify the most
  interesting potential neighbours, the on-disk vectors are loaded in and the
  results are re-ranked. A key parameter here is the reranking_factor, i.e.,
  how many more vectors are reranked than the desired k. For example 10 means
  that `10 * k vectors` are scored and then re-ranked. The more candidates you
  allow here, the better the Recall. The default is `20`. In the benchmarks, we
  will show lower versions to explore the impact here.

**Key parameters *(IVF-specific)*:**

- *Number of lists (nl)*: The number of independent k-means cluster to generate.
  If the structure of the data is unknown, people use `sqrt(n)` as a heuristic.
- *Number of points (np)*: The number of clusters to probe during search.
  Numbers here tend to be `sqrt(nlist)` or up to 5% of the nlist.

The self queries (i.e., kNN generation ) are done with `reranking_factor = 10`.
The performance of the binarisation is very dependent on the underlying
data. For some of the datasets we still reach decent Recalls of ≥0.8 in some
configurations; for others not at all and the Recall rapidly drops to ~0.5
and worse.

#### Correlated data

<details>
<summary><b>Correlated data - 256 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 256D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        10.55     4_766.13     4_776.68       1.0000     0.000000        48.83
Exhaustive (self)                                         10.55    16_121.75    16_132.30       1.0000     0.000000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_605.01       261.13     2_866.14       0.1028          NaN         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_605.01       382.21     2_987.22       0.3108    43.300474         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_605.01       505.38     3_110.39       0.4592    25.328172         1.78
ExhaustiveBinary-256-random (self)                     2_605.01     1_243.08     3_848.09       0.3108    42.875985         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_946.32       270.98     3_217.30       0.2435          NaN         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_946.32       374.27     3_320.58       0.6570    11.967453         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_946.32       509.93     3_456.25       0.7864     5.976325         1.78
ExhaustiveBinary-256-pca (self)                        2_946.32     1_268.07     4_214.39       0.6576    11.860151         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_204.78       481.24     5_686.03       0.1292          NaN         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_204.78       576.79     5_781.57       0.3988    30.860166         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_204.78       672.98     5_877.76       0.5650    16.596919         3.55
ExhaustiveBinary-512-random (self)                     5_204.78     1_992.60     7_197.38       0.3982    30.569280         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_605.00       483.70     6_088.70       0.2613          NaN         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_605.00       617.84     6_222.83       0.7226     8.224501         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_605.00       757.69     6_362.69       0.8654     3.067534         3.55
ExhaustiveBinary-512-pca (self)                        5_605.00     1_948.57     7_553.57       0.7232     8.171647         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_288.95       822.37    11_111.32       0.1731          NaN         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_288.95       960.81    11_249.75       0.5267    18.264305         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_288.95     1_116.36    11_405.30       0.6959     8.715766         7.10
ExhaustiveBinary-1024-random (self)                   10_288.95     3_114.70    13_403.65       0.5248    18.184021         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               11_143.49       835.63    11_979.12       0.2885          NaN         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                11_143.49     1_005.36    12_148.85       0.7675     5.999434         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                11_143.49     1_159.34    12_302.84       0.8973     2.018226         7.10
ExhaustiveBinary-1024-pca (self)                      11_143.49     3_110.78    14_254.27       0.7680     5.949327         7.10
ExhaustiveBinary-256-signed_no_rr (query)              2_583.37       248.20     2_831.57       0.1028          NaN         1.78
ExhaustiveBinary-256-signed-rf10 (query)               2_583.37       353.25     2_936.62       0.3108    43.300474         1.78
ExhaustiveBinary-256-signed-rf20 (query)               2_583.37       454.96     3_038.33       0.4592    25.328172         1.78
ExhaustiveBinary-256-signed (self)                     2_583.37     1_159.65     3_743.02       0.3108    42.875985         1.78
IVF-Binary-256-nl158-np7-rf0-random (query)            4_052.96       113.24     4_166.20       0.1083          NaN         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           4_052.96       119.36     4_172.32       0.1058          NaN         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           4_052.96       123.25     4_176.21       0.1039          NaN         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           4_052.96       173.33     4_226.29       0.3319    38.821454         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           4_052.96       224.38     4_277.34       0.4851    22.106414         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          4_052.96       181.73     4_234.69       0.3204    41.267767         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          4_052.96       237.64     4_290.60       0.4705    23.851119         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          4_052.96       193.86     4_246.83       0.3134    42.767082         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          4_052.96       253.65     4_306.61       0.4620    24.928273         1.93
IVF-Binary-256-nl158-random (self)                     4_052.96       554.36     4_607.32       0.3197    40.913889         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           3_036.88       118.06     3_154.94       0.1086          NaN         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           3_036.88       119.80     3_156.67       0.1068          NaN         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           3_036.88       125.45     3_162.33       0.1038          NaN         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          3_036.88       176.04     3_212.92       0.3245    41.356252         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          3_036.88       227.98     3_264.85       0.4754    24.091784         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          3_036.88       179.56     3_216.44       0.3186    42.214971         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          3_036.88       234.56     3_271.44       0.4684    24.649239         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          3_036.88       189.73     3_226.61       0.3121    43.028382         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          3_036.88       248.92     3_285.80       0.4615    25.146775         2.00
IVF-Binary-256-nl223-random (self)                     3_036.88       549.66     3_586.54       0.3187    41.815356         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           3_254.35       124.00     3_378.35       0.1062          NaN         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           3_254.35       125.60     3_379.96       0.1052          NaN         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           3_254.35       129.47     3_383.82       0.1036          NaN         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          3_254.35       184.12     3_438.47       0.3216    41.840948         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          3_254.35       236.26     3_490.62       0.4723    24.352976         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          3_254.35       188.64     3_443.00       0.3186    42.204456         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          3_254.35       241.38     3_495.73       0.4682    24.633897         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          3_254.35       191.49     3_445.84       0.3137    42.843288         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          3_254.35       249.26     3_503.62       0.4628    25.014669         2.09
IVF-Binary-256-nl316-random (self)                     3_254.35       564.52     3_818.88       0.3184    41.790454         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               4_381.35       118.60     4_499.96       0.2480          NaN         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              4_381.35       124.25     4_505.60       0.2466          NaN         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              4_381.35       132.69     4_514.04       0.2458          NaN         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              4_381.35       185.73     4_567.08       0.6898    10.239882         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              4_381.35       236.54     4_617.89       0.8312     4.750092         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             4_381.35       192.37     4_573.72       0.6913     9.937400         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             4_381.35       253.55     4_634.91       0.8392     4.008755         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             4_381.35       203.33     4_584.69       0.6889    10.134793         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             4_381.35       277.02     4_658.38       0.8364     4.129932         1.93
IVF-Binary-256-nl158-pca (self)                        4_381.35       595.09     4_976.45       0.6921     9.838518         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              3_368.37       121.89     3_490.26       0.2469          NaN         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              3_368.37       124.76     3_493.13       0.2462          NaN         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              3_368.37       130.89     3_499.26       0.2457          NaN         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             3_368.37       188.78     3_557.16       0.6922    10.008007         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             3_368.37       240.98     3_609.36       0.8410     3.993291         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             3_368.37       193.46     3_561.84       0.6905    10.075332         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             3_368.37       248.86     3_617.23       0.8391     4.052990         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             3_368.37       206.69     3_575.07       0.6887    10.183600         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             3_368.37       265.15     3_633.52       0.8361     4.151937         2.00
IVF-Binary-256-nl223-pca (self)                        3_368.37       612.58     3_980.95       0.6910     9.997322         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              3_593.36       132.02     3_725.38       0.2469          NaN         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              3_593.36       130.14     3_723.51       0.2464          NaN         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              3_593.36       134.87     3_728.23       0.2458          NaN         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             3_593.36       196.96     3_790.32       0.6928     9.992318         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             3_593.36       250.18     3_843.54       0.8413     3.983978         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             3_593.36       196.53     3_789.90       0.6915    10.045751         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             3_593.36       254.92     3_848.28       0.8404     4.014304         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             3_593.36       203.89     3_797.25       0.6897    10.130937         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             3_593.36       266.63     3_859.99       0.8376     4.108078         2.09
IVF-Binary-256-nl316-pca (self)                        3_593.36       651.78     4_245.15       0.6922     9.942858         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_571.30       208.85     6_780.15       0.1337          NaN         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_571.30       217.04     6_788.34       0.1314          NaN         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_571.30       229.24     6_800.53       0.1300          NaN         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_571.30       271.47     6_842.77       0.4144    28.473873         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_571.30       323.71     6_895.01       0.5795    15.215400         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_571.30       283.15     6_854.44       0.4062    29.754981         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_571.30       342.17     6_913.46       0.5725    15.854750         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_571.30       296.71     6_868.00       0.4011    30.534510         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_571.30       358.59     6_929.88       0.5670    16.375622         3.71
IVF-Binary-512-nl158-random (self)                     6_571.30       893.69     7_464.99       0.4047    29.534453         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_563.58       211.29     5_774.87       0.1320          NaN         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_563.58       216.04     5_779.61       0.1307          NaN         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_563.58       224.36     5_787.94       0.1293          NaN         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_563.58       275.44     5_839.01       0.4075    29.996061         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_563.58       326.27     5_889.84       0.5735    16.072763         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_563.58       279.82     5_843.40       0.4035    30.405525         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_563.58       334.33     5_897.90       0.5688    16.341726         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_563.58       292.55     5_856.13       0.4000    30.729230         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_563.58       352.97     5_916.54       0.5653    16.533698         3.77
IVF-Binary-512-nl223-random (self)                     5_563.58       878.93     6_442.50       0.4029    30.118263         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           5_737.48       218.20     5_955.68       0.1312          NaN         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           5_737.48       220.89     5_958.37       0.1305          NaN         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           5_737.48       226.81     5_964.30       0.1296          NaN         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          5_737.48       282.21     6_019.69       0.4064    30.161109         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          5_737.48       333.57     6_071.05       0.5728    16.131551         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          5_737.48       282.45     6_019.93       0.4039    30.386090         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          5_737.48       338.97     6_076.45       0.5701    16.277028         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          5_737.48       292.34     6_029.82       0.4011    30.652841         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          5_737.48       371.62     6_109.10       0.5663    16.463769         3.86
IVF-Binary-512-nl316-random (self)                     5_737.48       894.61     6_632.10       0.4031    30.098181         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               6_990.48       215.13     7_205.60       0.2630          NaN         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              6_990.48       223.95     7_214.43       0.2623          NaN         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              6_990.48       232.47     7_222.95       0.2616          NaN         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              6_990.48       283.91     7_274.39       0.7182     8.713693         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              6_990.48       339.27     7_329.75       0.8521     4.076349         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             6_990.48       296.27     7_286.74       0.7236     8.138391         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             6_990.48       358.97     7_349.45       0.8650     3.108818         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             6_990.48       309.63     7_300.11       0.7230     8.193878         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             6_990.48       374.53     7_365.01       0.8657     3.057313         3.71
IVF-Binary-512-nl158-pca (self)                        6_990.48       937.44     7_927.92       0.7241     8.102071         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              5_977.00       218.73     6_195.73       0.2624          NaN         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              5_977.00       228.98     6_205.98       0.2618          NaN         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              5_977.00       232.46     6_209.46       0.2615          NaN         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             5_977.00       286.59     6_263.59       0.7241     8.168147         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             5_977.00       339.55     6_316.55       0.8663     3.043275         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             5_977.00       293.08     6_270.08       0.7231     8.196721         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             5_977.00       349.05     6_326.05       0.8657     3.051725         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             5_977.00       308.28     6_285.28       0.7228     8.210064         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             5_977.00       369.31     6_346.31       0.8654     3.061065         3.77
IVF-Binary-512-nl223-pca (self)                        5_977.00       921.60     6_898.61       0.7234     8.153282         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              6_194.39       225.37     6_419.76       0.2623          NaN         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              6_194.39       228.63     6_423.02       0.2619          NaN         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              6_194.39       237.32     6_431.71       0.2616          NaN         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             6_194.39       295.43     6_489.82       0.7242     8.166510         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             6_194.39       347.14     6_541.53       0.8661     3.046978         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             6_194.39       311.36     6_505.75       0.7234     8.193121         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             6_194.39       353.35     6_547.74       0.8659     3.053080         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             6_194.39       326.06     6_520.45       0.7227     8.211498         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             6_194.39       365.90     6_560.29       0.8654     3.060741         3.86
IVF-Binary-512-nl316-pca (self)                        6_194.39       933.22     7_127.61       0.7239     8.139230         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_431.63       396.10    11_827.73       0.1770          NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_431.63       406.59    11_838.23       0.1752          NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_431.63       418.27    11_849.90       0.1740          NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_431.63       461.40    11_893.04       0.5317    17.771500         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_431.63       518.13    11_949.77       0.6953     8.882184         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_431.63       478.53    11_910.16       0.5304    17.856724         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_431.63       539.33    11_970.97       0.6984     8.545597         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_431.63       497.38    11_929.02       0.5276    18.145809         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_431.63       562.37    11_994.01       0.6966     8.669237         7.26
IVF-Binary-1024-nl158-random (self)                   11_431.63     1_545.92    12_977.56       0.5284    17.804914         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_409.08       397.91    10_806.98       0.1759          NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_409.08       403.60    10_812.68       0.1749          NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_409.08       419.57    10_828.65       0.1739          NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_409.08       469.03    10_878.11       0.5316    17.934552         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_409.08       520.12    10_929.20       0.7012     8.517556         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_409.08       474.40    10_883.48       0.5290    18.107284         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_409.08       533.78    10_942.86       0.6981     8.633901         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_409.08       498.66    10_907.74       0.5268    18.236031         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_409.08       574.24    10_983.31       0.6964     8.692795         7.32
IVF-Binary-1024-nl223-random (self)                   10_409.08     1_538.53    11_947.61       0.5274    18.012960         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         10_650.73       401.67    11_052.41       0.1753          NaN         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)         10_650.73       405.72    11_056.45       0.1748          NaN         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)         10_650.73       418.57    11_069.31       0.1741          NaN         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)        10_650.73       471.57    11_122.30       0.5315    17.960579         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)        10_650.73       526.91    11_177.64       0.7010     8.551004         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)        10_650.73       474.01    11_124.74       0.5296    18.070954         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)        10_650.73       532.50    11_183.23       0.6987     8.627558         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)        10_650.73       490.83    11_141.56       0.5276    18.180669         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)        10_650.73       552.11    11_202.85       0.6963     8.701409         7.41
IVF-Binary-1024-nl316-random (self)                   10_650.73     1_529.07    12_179.80       0.5281    17.982320         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)             12_112.79       408.68    12_521.48       0.2900          NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            12_112.79       421.01    12_533.80       0.2897          NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            12_112.79       434.60    12_547.39       0.2889          NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            12_112.79       482.35    12_595.15       0.7586     6.829705         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            12_112.79       536.16    12_648.95       0.8788     3.280646         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           12_112.79       500.75    12_613.54       0.7677     6.000876         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           12_112.79       559.78    12_672.57       0.8958     2.117318         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           12_112.79       517.04    12_629.83       0.7679     5.991751         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           12_112.79       583.47    12_696.27       0.8974     2.022817         7.26
IVF-Binary-1024-nl158-pca (self)                      12_112.79     1_612.91    13_725.70       0.7678     5.964602         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            11_082.20       410.62    11_492.82       0.2897          NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            11_082.20       419.87    11_502.07       0.2892          NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            11_082.20       431.72    11_513.92       0.2888          NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           11_082.20       482.94    11_565.15       0.7686     5.985640         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           11_082.20       538.89    11_621.10       0.8979     2.012913         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           11_082.20       490.50    11_572.70       0.7678     6.005988         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           11_082.20       556.56    11_638.77       0.8976     2.015277         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           11_082.20       512.43    11_594.63       0.7675     6.017011         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           11_082.20       575.51    11_657.71       0.8973     2.019636         7.32
IVF-Binary-1024-nl223-pca (self)                      11_082.20     1_586.28    12_668.48       0.7684     5.933963         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            11_286.57       416.98    11_703.55       0.2896          NaN         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            11_286.57       420.44    11_707.01       0.2893          NaN         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            11_286.57       432.18    11_718.75       0.2889          NaN         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           11_286.57       491.66    11_778.23       0.7684     5.985534         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           11_286.57       548.00    11_834.57       0.8980     2.010364         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           11_286.57       492.86    11_779.43       0.7680     5.997559         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           11_286.57       563.24    11_849.81       0.8979     2.010035         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           11_286.57       509.72    11_796.29       0.7675     6.011111         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           11_286.57       571.11    11_857.68       0.8976     2.014855         7.42
IVF-Binary-1024-nl316-pca (self)                      11_286.57     1_594.04    12_880.61       0.7687     5.924525         7.42
IVF-Binary-256-nl158-np7-rf0-signed (query)            4_019.95       113.26     4_133.22       0.1083          NaN         1.93
IVF-Binary-256-nl158-np12-rf0-signed (query)           4_019.95       119.43     4_139.38       0.1058          NaN         1.93
IVF-Binary-256-nl158-np17-rf0-signed (query)           4_019.95       126.36     4_146.32       0.1039          NaN         1.93
IVF-Binary-256-nl158-np7-rf10-signed (query)           4_019.95       178.04     4_198.00       0.3319    38.821454         1.93
IVF-Binary-256-nl158-np7-rf20-signed (query)           4_019.95       224.19     4_244.14       0.4851    22.106414         1.93
IVF-Binary-256-nl158-np12-rf10-signed (query)          4_019.95       183.84     4_203.79       0.3204    41.267767         1.93
IVF-Binary-256-nl158-np12-rf20-signed (query)          4_019.95       239.55     4_259.51       0.4705    23.851119         1.93
IVF-Binary-256-nl158-np17-rf10-signed (query)          4_019.95       194.48     4_214.44       0.3134    42.767082         1.93
IVF-Binary-256-nl158-np17-rf20-signed (query)          4_019.95       253.73     4_273.68       0.4620    24.928273         1.93
IVF-Binary-256-nl158-signed (self)                     4_019.95       566.19     4_586.15       0.3197    40.913889         1.93
IVF-Binary-256-nl223-np11-rf0-signed (query)           3_011.70       122.01     3_133.71       0.1086          NaN         2.00
IVF-Binary-256-nl223-np14-rf0-signed (query)           3_011.70       121.29     3_132.99       0.1068          NaN         2.00
IVF-Binary-256-nl223-np21-rf0-signed (query)           3_011.70       125.92     3_137.62       0.1038          NaN         2.00
IVF-Binary-256-nl223-np11-rf10-signed (query)          3_011.70       181.96     3_193.66       0.3245    41.356252         2.00
IVF-Binary-256-nl223-np11-rf20-signed (query)          3_011.70       228.97     3_240.67       0.4754    24.091784         2.00
IVF-Binary-256-nl223-np14-rf10-signed (query)          3_011.70       180.55     3_192.25       0.3186    42.214971         2.00
IVF-Binary-256-nl223-np14-rf20-signed (query)          3_011.70       235.78     3_247.48       0.4684    24.649239         2.00
IVF-Binary-256-nl223-np21-rf10-signed (query)          3_011.70       189.74     3_201.44       0.3121    43.028382         2.00
IVF-Binary-256-nl223-np21-rf20-signed (query)          3_011.70       249.68     3_261.38       0.4615    25.146775         2.00
IVF-Binary-256-nl223-signed (self)                     3_011.70       551.58     3_563.28       0.3187    41.815356         2.00
IVF-Binary-256-nl316-np15-rf0-signed (query)           3_204.68       124.40     3_329.08       0.1062          NaN         2.09
IVF-Binary-256-nl316-np17-rf0-signed (query)           3_204.68       125.46     3_330.14       0.1052          NaN         2.09
IVF-Binary-256-nl316-np25-rf0-signed (query)           3_204.68       131.00     3_335.68       0.1036          NaN         2.09
IVF-Binary-256-nl316-np15-rf10-signed (query)          3_204.68       184.38     3_389.06       0.3216    41.840948         2.09
IVF-Binary-256-nl316-np15-rf20-signed (query)          3_204.68       237.38     3_442.06       0.4723    24.352976         2.09
IVF-Binary-256-nl316-np17-rf10-signed (query)          3_204.68       184.92     3_389.60       0.3186    42.204456         2.09
IVF-Binary-256-nl316-np17-rf20-signed (query)          3_204.68       239.69     3_444.37       0.4682    24.633897         2.09
IVF-Binary-256-nl316-np25-rf10-signed (query)          3_204.68       192.42     3_397.10       0.3137    42.843288         2.09
IVF-Binary-256-nl316-np25-rf20-signed (query)          3_204.68       249.60     3_454.28       0.4628    25.014669         2.09
IVF-Binary-256-nl316-signed (self)                     3_204.68       566.17     3_770.85       0.3184    41.790454         2.09
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 512D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        19.95     9_596.74     9_616.69       1.0000     0.000000        97.66
Exhaustive (self)                                         19.95    32_493.22    32_513.17       1.0000     0.000000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_667.03       354.91     6_021.94       0.0922          NaN         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_667.03       483.89     6_150.92       0.2633   169.301831         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_667.03       615.39     6_282.42       0.3984   101.532086         2.03
ExhaustiveBinary-256-random (self)                     5_667.03     1_591.30     7_258.33       0.2634   168.374114         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_327.72       361.67     6_689.39       0.1932          NaN         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_327.72       513.32     6_841.04       0.5408    65.556763         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_327.72       629.25     6_956.98       0.6776    36.729938         2.03
ExhaustiveBinary-256-pca (self)                        6_327.72     1_644.00     7_971.73       0.5402    65.374917         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_200.64       666.66    11_867.30       0.1108          NaN         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_200.64       808.97    12_009.61       0.3300   130.093417         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_200.64       924.35    12_124.99       0.4812    74.252695         4.05
ExhaustiveBinary-512-random (self)                    11_200.64     2_606.45    13_807.09       0.3293   129.846684         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_883.26       670.41    12_553.68       0.2354          NaN         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_883.26       804.72    12_687.99       0.5811    56.845265         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_883.26       938.71    12_821.97       0.6958    33.780093         4.05
ExhaustiveBinary-512-pca (self)                       11_883.26     2_701.45    14_584.71       0.5800    56.874208         4.05
ExhaustiveBinary-1024-random_no_rr (query)            22_053.19     1_196.10    23_249.29       0.1367          NaN         8.10
ExhaustiveBinary-1024-random-rf10 (query)             22_053.19     1_323.05    23_376.24       0.4187    93.031962         8.10
ExhaustiveBinary-1024-random-rf20 (query)             22_053.19     1_478.80    23_531.98       0.5846    48.774664         8.10
ExhaustiveBinary-1024-random (self)                   22_053.19     4_404.43    26_457.61       0.4189    92.578758         8.10
ExhaustiveBinary-1024-pca_no_rr (query)               23_215.34     1_207.98    24_423.32       0.2643          NaN         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                23_215.34     1_355.13    24_570.48       0.7223    27.045523         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                23_215.34     1_508.59    24_723.94       0.8665     9.915273         8.11
ExhaustiveBinary-1024-pca (self)                      23_215.34     4_550.25    27_765.60       0.7210    27.032852         8.11
ExhaustiveBinary-512-signed_no_rr (query)             11_066.56       662.58    11_729.14       0.1108          NaN         4.05
ExhaustiveBinary-512-signed-rf10 (query)              11_066.56       785.86    11_852.42       0.3300   130.093417         4.05
ExhaustiveBinary-512-signed-rf20 (query)              11_066.56       919.04    11_985.59       0.4812    74.252695         4.05
ExhaustiveBinary-512-signed (self)                    11_066.56     2_599.88    13_666.44       0.3293   129.846684         4.05
IVF-Binary-256-nl158-np7-rf0-random (query)            8_777.60       228.09     9_005.68       0.0956          NaN         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           8_777.60       236.36     9_013.96       0.0938          NaN         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           8_777.60       236.79     9_014.38       0.0928          NaN         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           8_777.60       315.28     9_092.88       0.2769   158.070384         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           8_777.60       394.68     9_172.28       0.4159    93.352404         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          8_777.60       324.21     9_101.81       0.2684   165.235944         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          8_777.60       407.89     9_185.49       0.4036    99.044818         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          8_777.60       332.85     9_110.45       0.2655   167.598087         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          8_777.60       419.91     9_197.51       0.4002   100.639478         2.34
IVF-Binary-256-nl158-random (self)                     8_777.60       991.82     9_769.42       0.2679   164.490522         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           6_680.65       238.20     6_918.85       0.0959          NaN         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)           6_680.65       241.14     6_921.79       0.0948          NaN         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)           6_680.65       247.69     6_928.34       0.0932          NaN         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)          6_680.65       323.84     7_004.49       0.2728   163.979892         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)          6_680.65       436.51     7_117.16       0.4095    98.018473         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)          6_680.65       329.70     7_010.35       0.2689   166.302984         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)          6_680.65       413.90     7_094.55       0.4043    99.819643         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)          6_680.65       337.98     7_018.63       0.2652   168.089696         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)          6_680.65       426.44     7_107.09       0.4003   100.880474         2.46
IVF-Binary-256-nl223-random (self)                     6_680.65     1_014.92     7_695.57       0.2680   165.706416         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)           6_966.43       253.30     7_219.73       0.0948          NaN         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           6_966.43       260.68     7_227.10       0.0943          NaN         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           6_966.43       257.69     7_224.11       0.0934          NaN         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          6_966.43       338.64     7_305.06       0.2696   165.755796         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          6_966.43       420.63     7_387.05       0.4062    99.294646         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          6_966.43       337.94     7_304.36       0.2676   166.853264         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          6_966.43       422.29     7_388.72       0.4039   100.013910         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          6_966.43       345.78     7_312.21       0.2648   168.226438         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          6_966.43       434.84     7_401.27       0.4006   100.966764         2.65
IVF-Binary-256-nl316-random (self)                     6_966.43     1_053.81     8_020.24       0.2670   166.225560         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               9_431.87       238.82     9_670.69       0.1970          NaN         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              9_431.87       241.28     9_673.16       0.1959          NaN         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              9_431.87       249.11     9_680.98       0.1954          NaN         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              9_431.87       333.82     9_765.69       0.5806    55.069767         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              9_431.87       414.49     9_846.36       0.7427    26.640402         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             9_431.87       345.00     9_776.88       0.5784    55.465176         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             9_431.87       429.90     9_861.77       0.7431    25.831263         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             9_431.87       350.14     9_782.01       0.5759    56.235871         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             9_431.87       449.07     9_880.94       0.7395    26.454083         2.34
IVF-Binary-256-nl158-pca (self)                        9_431.87     1_055.61    10_487.49       0.5776    55.307736         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              7_386.18       244.87     7_631.05       0.1963          NaN         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              7_386.18       249.00     7_635.18       0.1957          NaN         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              7_386.18       264.13     7_650.31       0.1953          NaN         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             7_386.18       347.11     7_733.30       0.5809    55.037205         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             7_386.18       430.27     7_816.45       0.7464    25.412896         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             7_386.18       343.66     7_729.84       0.5782    55.695026         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             7_386.18       435.06     7_821.24       0.7428    25.978082         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             7_386.18       353.80     7_739.98       0.5752    56.490542         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             7_386.18       448.97     7_835.15       0.7377    26.823021         2.47
IVF-Binary-256-nl223-pca (self)                        7_386.18     1_073.40     8_459.59       0.5773    55.528900         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              7_621.72       259.62     7_881.34       0.1960          NaN         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              7_621.72       262.24     7_883.96       0.1958          NaN         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              7_621.72       266.15     7_887.87       0.1954          NaN         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             7_621.72       358.27     7_979.98       0.5806    55.169913         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             7_621.72       442.57     8_064.29       0.7465    25.391085         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             7_621.72       358.19     7_979.90       0.5795    55.405946         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             7_621.72       446.99     8_068.71       0.7452    25.591676         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             7_621.72       364.97     7_986.68       0.5763    56.244572         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             7_621.72       459.55     8_081.27       0.7400    26.435566         2.65
IVF-Binary-256-nl316-pca (self)                        7_621.72     1_113.00     8_734.72       0.5787    55.203425         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           14_228.37       429.13    14_657.50       0.1137          NaN         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          14_228.37       434.57    14_662.94       0.1118          NaN         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          14_228.37       440.42    14_668.79       0.1112          NaN         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          14_228.37       515.16    14_743.53       0.3396   123.984956         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          14_228.37       593.63    14_822.00       0.4926    70.020940         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         14_228.37       522.75    14_751.12       0.3326   128.439904         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         14_228.37       607.83    14_836.20       0.4845    72.926803         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         14_228.37       533.24    14_761.61       0.3308   129.615004         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         14_228.37       626.50    14_854.87       0.4821    73.824280         4.36
IVF-Binary-512-nl158-random (self)                    14_228.37     1_673.18    15_901.56       0.3324   127.925257         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          12_204.85       438.55    12_643.40       0.1125          NaN         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          12_204.85       441.90    12_646.75       0.1117          NaN         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          12_204.85       462.39    12_667.24       0.1110          NaN         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         12_204.85       526.51    12_731.36       0.3361   127.562557         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         12_204.85       611.01    12_815.86       0.4885    72.435013         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         12_204.85       528.26    12_733.11       0.3331   128.980824         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         12_204.85       613.78    12_818.63       0.4845    73.481797         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         12_204.85       544.97    12_749.82       0.3308   129.767854         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         12_204.85       646.61    12_851.46       0.4819    74.011904         4.49
IVF-Binary-512-nl223-random (self)                    12_204.85     1_694.79    13_899.64       0.3325   128.605521         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          12_460.70       452.14    12_912.84       0.1120          NaN         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          12_460.70       454.39    12_915.09       0.1118          NaN         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          12_460.70       460.72    12_921.42       0.1114          NaN         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         12_460.70       538.44    12_999.13       0.3337   128.602015         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         12_460.70       618.65    13_079.34       0.4861    73.096204         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         12_460.70       542.38    13_003.08       0.3326   129.155439         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         12_460.70       624.03    13_084.73       0.4845    73.516318         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         12_460.70       653.85    13_114.54       0.3311   129.728289         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         12_460.70       642.97    13_103.67       0.4825    73.945967         4.67
IVF-Binary-512-nl316-random (self)                    12_460.70     1_780.98    14_241.68       0.3318   128.875996         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              15_072.64       438.92    15_511.56       0.2492          NaN         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             15_072.64       443.51    15_516.15       0.2482          NaN         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             15_072.64       453.12    15_525.76       0.2476          NaN         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             15_072.64       526.93    15_599.57       0.6820    35.354500         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             15_072.64       610.51    15_683.16       0.8252    16.264309         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            15_072.64       538.38    15_611.02       0.6811    34.585986         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            15_072.64       645.16    15_717.80       0.8272    14.547321         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            15_072.64       565.84    15_638.48       0.6760    35.607839         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            15_072.64       656.21    15_728.85       0.8203    15.358502         4.36
IVF-Binary-512-nl158-pca (self)                       15_072.64     1_734.79    16_807.43       0.6805    34.555851         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             12_987.21       446.14    13_433.34       0.2488          NaN         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             12_987.21       454.13    13_441.34       0.2479          NaN         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             12_987.21       462.06    13_449.27       0.2470          NaN         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            12_987.21       539.86    13_527.07       0.6862    33.820600         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            12_987.21       623.26    13_610.47       0.8341    13.838383         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            12_987.21       556.48    13_543.69       0.6821    34.665570         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            12_987.21       649.35    13_636.56       0.8288    14.446211         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            12_987.21       559.16    13_546.36       0.6743    36.189548         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            12_987.21       657.61    13_644.82       0.8173    15.935167         4.49
IVF-Binary-512-nl223-pca (self)                       12_987.21     1_776.55    14_763.76       0.6808    34.669079         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             13_761.70       460.33    14_222.04       0.2484          NaN         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             13_761.70       460.81    14_222.52       0.2481          NaN         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             13_761.70       475.63    14_237.34       0.2473          NaN         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            13_761.70       563.76    14_325.47       0.6871    33.676595         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            13_761.70       639.78    14_401.48       0.8358    13.651413         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            13_761.70       554.17    14_315.88       0.6852    34.025497         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            13_761.70       641.94    14_403.64       0.8332    13.943271         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            13_761.70       567.48    14_329.18       0.6776    35.478741         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            13_761.70       665.68    14_427.38       0.8224    15.255941         4.67
IVF-Binary-512-nl316-pca (self)                       13_761.70     1_821.20    15_582.90       0.6840    34.073426         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          26_035.36       834.31    26_869.67       0.1389          NaN         8.41
IVF-Binary-1024-nl158-np12-rf0-random (query)         26_035.36       860.66    26_896.02       0.1372          NaN         8.41
IVF-Binary-1024-nl158-np17-rf0-random (query)         26_035.36       858.61    26_893.97       0.1369          NaN         8.41
IVF-Binary-1024-nl158-np7-rf10-random (query)         26_035.36       948.31    26_983.67       0.4242    90.411129         8.41
IVF-Binary-1024-nl158-np7-rf20-random (query)         26_035.36     1_013.68    27_049.04       0.5881    47.998983         8.41
IVF-Binary-1024-nl158-np12-rf10-random (query)        26_035.36       947.35    26_982.71       0.4203    92.212994         8.41
IVF-Binary-1024-nl158-np12-rf20-random (query)        26_035.36     1_065.90    27_101.26       0.5862    48.272945         8.41
IVF-Binary-1024-nl158-np17-rf10-random (query)        26_035.36     1_050.77    27_086.13       0.4188    92.864832         8.41
IVF-Binary-1024-nl158-np17-rf20-random (query)        26_035.36     1_103.33    27_138.69       0.5849    48.616056         8.41
IVF-Binary-1024-nl158-random (self)                   26_035.36     3_058.80    29_094.16       0.4206    91.701181         8.41
IVF-Binary-1024-nl223-np11-rf0-random (query)         23_097.38       830.98    23_928.36       0.1383          NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         23_097.38       837.87    23_935.26       0.1375          NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         23_097.38       854.24    23_951.63       0.1371          NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        23_097.38       922.59    24_019.98       0.4230    91.745460         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        23_097.38     1_003.53    24_100.92       0.5891    47.936995         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        23_097.38       932.08    24_029.47       0.4204    92.538890         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        23_097.38     1_011.47    24_108.85       0.5863    48.483196         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        23_097.38       946.98    24_044.36       0.4189    92.914436         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        23_097.38     1_041.11    24_138.50       0.5852    48.679308         8.54
IVF-Binary-1024-nl223-random (self)                   23_097.38     3_027.57    26_124.95       0.4206    92.095325         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         23_314.56       865.11    24_179.66       0.1378          NaN         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)         23_314.56       852.55    24_167.11       0.1375          NaN         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)         23_314.56       863.88    24_178.43       0.1371          NaN         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)        23_314.56       940.12    24_254.68       0.4213    92.228460         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)        23_314.56     1_045.06    24_359.61       0.5878    48.287709         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)        23_314.56       938.38    24_252.94       0.4203    92.575491         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)        23_314.56     1_023.48    24_338.04       0.5866    48.495089         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)        23_314.56       956.39    24_270.95       0.4191    92.885206         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)        23_314.56     1_047.53    24_362.09       0.5854    48.686731         8.72
IVF-Binary-1024-nl316-random (self)                   23_314.56     3_109.58    26_424.14       0.4205    92.189863         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)             26_328.39       858.24    27_186.63       0.2645          NaN         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            26_328.39       858.28    27_186.67       0.2645          NaN         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            26_328.39       866.96    27_195.36       0.2644          NaN         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            26_328.39       946.28    27_274.67       0.7154    29.452034         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            26_328.39     1_029.60    27_357.99       0.8525    13.316467         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           26_328.39       955.37    27_283.76       0.7224    27.029224         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           26_328.39     1_045.89    27_374.28       0.8660    10.070513         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           26_328.39       969.43    27_297.82       0.7226    26.992115         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           26_328.39     1_061.32    27_389.71       0.8665     9.941726         8.42
IVF-Binary-1024-nl158-pca (self)                      26_328.39     3_113.48    29_441.87       0.7211    27.033690         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            24_314.77       914.81    25_229.58       0.2649          NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            24_314.77       864.60    25_179.37       0.2646          NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            24_314.77       879.23    25_194.00       0.2645          NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           24_314.77       951.07    25_265.84       0.7227    27.025420         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           24_314.77     1_034.65    25_349.42       0.8657    10.034123         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           24_314.77       959.40    25_274.17       0.7224    27.052724         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           24_314.77     1_048.82    25_363.59       0.8665     9.899019         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           24_314.77       981.50    25_296.27       0.7222    27.067324         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           24_314.77     1_075.96    25_390.73       0.8665     9.901282         8.54
IVF-Binary-1024-nl223-pca (self)                      24_314.77     3_134.00    27_448.77       0.7212    26.975540         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            24_602.19       876.52    25_478.71       0.2646          NaN         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            24_602.19       873.88    25_476.07       0.2644          NaN         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            24_602.19       886.10    25_488.29       0.2644          NaN         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           24_602.19       966.03    25_568.23       0.7227    26.985380         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           24_602.19     1_053.56    25_655.76       0.8664     9.929616         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           24_602.19       967.98    25_570.17       0.7224    27.012015         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           24_602.19     1_058.37    25_660.57       0.8664     9.924556         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           24_602.19       985.11    25_587.31       0.7223    27.036512         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           24_602.19     1_080.14    25_682.33       0.8663     9.934700         8.73
IVF-Binary-1024-nl316-pca (self)                      24_602.19     3_158.68    27_760.87       0.7212    26.992697         8.73
IVF-Binary-512-nl158-np7-rf0-signed (query)           14_241.40       430.86    14_672.26       0.1137          NaN         4.36
IVF-Binary-512-nl158-np12-rf0-signed (query)          14_241.40       439.75    14_681.15       0.1118          NaN         4.36
IVF-Binary-512-nl158-np17-rf0-signed (query)          14_241.40       445.90    14_687.29       0.1112          NaN         4.36
IVF-Binary-512-nl158-np7-rf10-signed (query)          14_241.40       520.75    14_762.14       0.3396   123.984956         4.36
IVF-Binary-512-nl158-np7-rf20-signed (query)          14_241.40       593.73    14_835.13       0.4926    70.020940         4.36
IVF-Binary-512-nl158-np12-rf10-signed (query)         14_241.40       523.63    14_765.02       0.3326   128.439904         4.36
IVF-Binary-512-nl158-np12-rf20-signed (query)         14_241.40       607.68    14_849.07       0.4845    72.926803         4.36
IVF-Binary-512-nl158-np17-rf10-signed (query)         14_241.40       561.20    14_802.60       0.3308   129.615004         4.36
IVF-Binary-512-nl158-np17-rf20-signed (query)         14_241.40       619.74    14_861.14       0.4821    73.824280         4.36
IVF-Binary-512-nl158-signed (self)                    14_241.40     1_670.65    15_912.05       0.3324   127.925257         4.36
IVF-Binary-512-nl223-np11-rf0-signed (query)          12_215.14       436.42    12_651.57       0.1125          NaN         4.49
IVF-Binary-512-nl223-np14-rf0-signed (query)          12_215.14       442.00    12_657.14       0.1117          NaN         4.49
IVF-Binary-512-nl223-np21-rf0-signed (query)          12_215.14       451.65    12_666.79       0.1110          NaN         4.49
IVF-Binary-512-nl223-np11-rf10-signed (query)         12_215.14       523.72    12_738.86       0.3361   127.562557         4.49
IVF-Binary-512-nl223-np11-rf20-signed (query)         12_215.14       606.25    12_821.39       0.4885    72.435013         4.49
IVF-Binary-512-nl223-np14-rf10-signed (query)         12_215.14       532.72    12_747.87       0.3331   128.980824         4.49
IVF-Binary-512-nl223-np14-rf20-signed (query)         12_215.14       615.50    12_830.64       0.4845    73.481797         4.49
IVF-Binary-512-nl223-np21-rf10-signed (query)         12_215.14       541.68    12_756.82       0.3308   129.767854         4.49
IVF-Binary-512-nl223-np21-rf20-signed (query)         12_215.14       632.62    12_847.76       0.4819    74.011904         4.49
IVF-Binary-512-nl223-signed (self)                    12_215.14     1_697.15    13_912.30       0.3325   128.605521         4.49
IVF-Binary-512-nl316-np15-rf0-signed (query)          12_461.49       451.41    12_912.90       0.1120          NaN         4.67
IVF-Binary-512-nl316-np17-rf0-signed (query)          12_461.49       452.90    12_914.39       0.1118          NaN         4.67
IVF-Binary-512-nl316-np25-rf0-signed (query)          12_461.49       460.53    12_922.03       0.1114          NaN         4.67
IVF-Binary-512-nl316-np15-rf10-signed (query)         12_461.49       538.37    12_999.87       0.3337   128.602015         4.67
IVF-Binary-512-nl316-np15-rf20-signed (query)         12_461.49       619.89    13_081.38       0.4861    73.096204         4.67
IVF-Binary-512-nl316-np17-rf10-signed (query)         12_461.49       538.94    13_000.43       0.3326   129.155439         4.67
IVF-Binary-512-nl316-np17-rf20-signed (query)         12_461.49       622.43    13_083.92       0.4845    73.516318         4.67
IVF-Binary-512-nl316-np25-rf10-signed (query)         12_461.49       550.47    13_011.97       0.3311   129.728289         4.67
IVF-Binary-512-nl316-np25-rf20-signed (query)         12_461.49       640.22    13_101.71       0.4825    73.945967         4.67
IVF-Binary-512-nl316-signed (self)                    12_461.49     2_001.05    14_462.55       0.3318   128.875996         4.67
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 1024 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 1024D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        40.67    23_855.23    23_895.89       1.0000     0.000000       195.31
Exhaustive (self)                                         40.67    75_222.71    75_263.37       1.0000     0.000000       195.31
ExhaustiveBinary-256-random_no_rr (query)             11_921.78       601.17    12_522.95       0.0863          NaN         2.53
ExhaustiveBinary-256-random-rf10 (query)              11_921.78       759.65    12_681.43       0.2291   759.619395         2.53
ExhaustiveBinary-256-random-rf20 (query)              11_921.78       935.30    12_857.09       0.3517   476.974594         2.53
ExhaustiveBinary-256-random (self)                    11_921.78     2_515.38    14_437.16       0.2292   758.888518         2.53
ExhaustiveBinary-256-pca_no_rr (query)                13_549.93       597.48    14_147.41       0.1779          NaN         2.53
ExhaustiveBinary-256-pca-rf10 (query)                 13_549.93       779.26    14_329.19       0.4999   292.911580         2.53
ExhaustiveBinary-256-pca-rf20 (query)                 13_549.93       954.70    14_504.63       0.6332   172.489148         2.53
ExhaustiveBinary-256-pca (self)                       13_549.93     2_571.80    16_121.73       0.5005   292.462909         2.53
ExhaustiveBinary-512-random_no_rr (query)             23_491.36     1_131.99    24_623.35       0.0971          NaN         5.05
ExhaustiveBinary-512-random-rf10 (query)              23_491.36     1_288.15    24_779.52       0.2718   640.934141         5.05
ExhaustiveBinary-512-random-rf20 (query)              23_491.36     1_472.20    24_963.56       0.4094   390.089149         5.05
ExhaustiveBinary-512-random (self)                    23_491.36     4_253.98    27_745.34       0.2721   640.227048         5.05
ExhaustiveBinary-512-pca_no_rr (query)                25_049.12     1_144.21    26_193.33       0.1925          NaN         5.06
ExhaustiveBinary-512-pca-rf10 (query)                 25_049.12     1_307.14    26_356.26       0.4876   312.871467         5.06
ExhaustiveBinary-512-pca-rf20 (query)                 25_049.12     1_508.80    26_557.92       0.6023   197.320835         5.06
ExhaustiveBinary-512-pca (self)                       25_049.12     4_324.11    29_373.24       0.4889   311.325942         5.06
ExhaustiveBinary-1024-random_no_rr (query)            47_018.79     2_107.74    49_126.53       0.1135          NaN        10.10
ExhaustiveBinary-1024-random-rf10 (query)             47_018.79     2_276.77    49_295.56       0.3401   496.311094        10.10
ExhaustiveBinary-1024-random-rf20 (query)             47_018.79     2_470.51    49_489.30       0.4953   285.246962        10.10
ExhaustiveBinary-1024-random (self)                   47_018.79     7_547.46    54_566.25       0.3415   492.986787        10.10
ExhaustiveBinary-1024-pca_no_rr (query)               48_608.27     2_155.72    50_763.99       0.1995          NaN        10.11
ExhaustiveBinary-1024-pca-rf10 (query)                48_608.27     2_309.01    50_917.29       0.4720   341.746775        10.11
ExhaustiveBinary-1024-pca-rf20 (query)                48_608.27     2_516.12    51_124.39       0.5759   222.235689        10.11
ExhaustiveBinary-1024-pca (self)                      48_608.27     7_692.29    56_300.56       0.4729   340.398858        10.11
ExhaustiveBinary-1024-signed_no_rr (query)            47_011.51     2_109.71    49_121.22       0.1135          NaN        10.10
ExhaustiveBinary-1024-signed-rf10 (query)             47_011.51     2_282.09    49_293.60       0.3401   496.311094        10.10
ExhaustiveBinary-1024-signed-rf20 (query)             47_011.51     2_476.91    49_488.42       0.4953   285.246962        10.10
ExhaustiveBinary-1024-signed (self)                   47_011.51     7_561.23    54_572.75       0.3415   492.986787        10.10
IVF-Binary-256-nl158-np7-rf0-random (query)           19_009.24       488.32    19_497.56       0.0888          NaN         3.14
IVF-Binary-256-nl158-np12-rf0-random (query)          19_009.24       494.84    19_504.08       0.0876          NaN         3.14
IVF-Binary-256-nl158-np17-rf0-random (query)          19_009.24       502.08    19_511.32       0.0870          NaN         3.14
IVF-Binary-256-nl158-np7-rf10-random (query)          19_009.24       623.25    19_632.49       0.2366   732.722507         3.14
IVF-Binary-256-nl158-np7-rf20-random (query)          19_009.24       744.85    19_754.08       0.3616   456.807346         3.14
IVF-Binary-256-nl158-np12-rf10-random (query)         19_009.24       627.22    19_636.45       0.2320   750.312886         3.14
IVF-Binary-256-nl158-np12-rf20-random (query)         19_009.24       759.93    19_769.17       0.3546   471.377427         3.14
IVF-Binary-256-nl158-np17-rf10-random (query)         19_009.24       641.24    19_650.48       0.2302   756.469366         3.14
IVF-Binary-256-nl158-np17-rf20-random (query)         19_009.24       778.72    19_787.96       0.3525   475.508600         3.14
IVF-Binary-256-nl158-random (self)                    19_009.24     1_998.54    21_007.77       0.2318   749.728177         3.14
IVF-Binary-256-nl223-np11-rf0-random (query)          13_866.75       520.50    14_387.25       0.0882          NaN         3.40
IVF-Binary-256-nl223-np14-rf0-random (query)          13_866.75       521.48    14_388.23       0.0876          NaN         3.40
IVF-Binary-256-nl223-np21-rf0-random (query)          13_866.75       528.98    14_395.73       0.0870          NaN         3.40
IVF-Binary-256-nl223-np11-rf10-random (query)         13_866.75       648.71    14_515.46       0.2356   740.137410         3.40
IVF-Binary-256-nl223-np11-rf20-random (query)         13_866.75       773.51    14_640.26       0.3603   463.208068         3.40
IVF-Binary-256-nl223-np14-rf10-random (query)         13_866.75       646.72    14_513.47       0.2331   748.286506         3.40
IVF-Binary-256-nl223-np14-rf20-random (query)         13_866.75       785.83    14_652.58       0.3569   469.066922         3.40
IVF-Binary-256-nl223-np21-rf10-random (query)         13_866.75       678.98    14_545.73       0.2309   754.484059         3.40
IVF-Binary-256-nl223-np21-rf20-random (query)         13_866.75       797.23    14_663.98       0.3543   473.058786         3.40
IVF-Binary-256-nl223-random (self)                    13_866.75     2_068.32    15_935.07       0.2331   747.959196         3.40
IVF-Binary-256-nl316-np15-rf0-random (query)          14_569.36       552.29    15_121.65       0.0875          NaN         3.76
IVF-Binary-256-nl316-np17-rf0-random (query)          14_569.36       552.20    15_121.55       0.0873          NaN         3.76
IVF-Binary-256-nl316-np25-rf0-random (query)          14_569.36       587.72    15_157.07       0.0867          NaN         3.76
IVF-Binary-256-nl316-np15-rf10-random (query)         14_569.36       683.84    15_253.19       0.2342   744.331605         3.76
IVF-Binary-256-nl316-np15-rf20-random (query)         14_569.36       821.75    15_391.11       0.3579   466.432204         3.76
IVF-Binary-256-nl316-np17-rf10-random (query)         14_569.36       677.88    15_247.23       0.2328   748.318536         3.76
IVF-Binary-256-nl316-np17-rf20-random (query)         14_569.36       808.36    15_377.71       0.3560   469.701940         3.76
IVF-Binary-256-nl316-np25-rf10-random (query)         14_569.36       684.89    15_254.25       0.2307   755.703778         3.76
IVF-Binary-256-nl316-np25-rf20-random (query)         14_569.36       818.91    15_388.27       0.3532   475.141394         3.76
IVF-Binary-256-nl316-random (self)                    14_569.36     2_171.78    16_741.14       0.2322   749.111081         3.76
IVF-Binary-256-nl158-np7-rf0-pca (query)              20_698.56       495.87    21_194.43       0.1812          NaN         3.15
IVF-Binary-256-nl158-np12-rf0-pca (query)             20_698.56       501.17    21_199.73       0.1807          NaN         3.15
IVF-Binary-256-nl158-np17-rf0-pca (query)             20_698.56       506.76    21_205.32       0.1801          NaN         3.15
IVF-Binary-256-nl158-np7-rf10-pca (query)             20_698.56       632.33    21_330.89       0.5398   251.373249         3.15
IVF-Binary-256-nl158-np7-rf20-pca (query)             20_698.56       758.76    21_457.32       0.7043   126.633866         3.15
IVF-Binary-256-nl158-np12-rf10-pca (query)            20_698.56       653.49    21_352.05       0.5392   250.889585         3.15
IVF-Binary-256-nl158-np12-rf20-pca (query)            20_698.56       775.92    21_474.47       0.7038   124.883884         3.15
IVF-Binary-256-nl158-np17-rf10-pca (query)            20_698.56       654.64    21_353.19       0.5365   253.456650         3.15
IVF-Binary-256-nl158-np17-rf20-pca (query)            20_698.56       805.18    21_503.74       0.6991   127.815218         3.15
IVF-Binary-256-nl158-pca (self)                       20_698.56     2_061.04    22_759.60       0.5397   250.334596         3.15
IVF-Binary-256-nl223-np11-rf0-pca (query)             15_632.61       579.16    16_211.77       0.1812          NaN         3.40
IVF-Binary-256-nl223-np14-rf0-pca (query)             15_632.61       551.67    16_184.28       0.1807          NaN         3.40
IVF-Binary-256-nl223-np21-rf0-pca (query)             15_632.61       553.29    16_185.90       0.1803          NaN         3.40
IVF-Binary-256-nl223-np11-rf10-pca (query)            15_632.61       690.84    16_323.45       0.5414   249.058560         3.40
IVF-Binary-256-nl223-np11-rf20-pca (query)            15_632.61       833.22    16_465.83       0.7078   123.155814         3.40
IVF-Binary-256-nl223-np14-rf10-pca (query)            15_632.61       693.48    16_326.09       0.5405   249.820721         3.40
IVF-Binary-256-nl223-np14-rf20-pca (query)            15_632.61       844.35    16_476.97       0.7063   123.697363         3.40
IVF-Binary-256-nl223-np21-rf10-pca (query)            15_632.61       697.99    16_330.60       0.5376   252.590480         3.40
IVF-Binary-256-nl223-np21-rf20-pca (query)            15_632.61       854.34    16_486.95       0.7012   126.654817         3.40
IVF-Binary-256-nl223-pca (self)                       15_632.61     2_524.59    18_157.20       0.5415   248.915107         3.40
IVF-Binary-256-nl316-np15-rf0-pca (query)             16_388.44       556.96    16_945.40       0.1811          NaN         3.77
IVF-Binary-256-nl316-np17-rf0-pca (query)             16_388.44       558.67    16_947.12       0.1809          NaN         3.77
IVF-Binary-256-nl316-np25-rf0-pca (query)             16_388.44       565.62    16_954.06       0.1804          NaN         3.77
IVF-Binary-256-nl316-np15-rf10-pca (query)            16_388.44       695.09    17_083.54       0.5415   248.805319         3.77
IVF-Binary-256-nl316-np15-rf20-pca (query)            16_388.44       831.02    17_219.46       0.7085   122.774182         3.77
IVF-Binary-256-nl316-np17-rf10-pca (query)            16_388.44       697.64    17_086.09       0.5411   249.208950         3.77
IVF-Binary-256-nl316-np17-rf20-pca (query)            16_388.44       827.91    17_216.35       0.7081   122.856965         3.77
IVF-Binary-256-nl316-np25-rf10-pca (query)            16_388.44       705.42    17_093.87       0.5385   251.557470         3.77
IVF-Binary-256-nl316-np25-rf20-pca (query)            16_388.44       846.81    17_235.25       0.7035   125.315100         3.77
IVF-Binary-256-nl316-pca (self)                       16_388.44     2_238.01    18_626.46       0.5422   248.195205         3.77
IVF-Binary-512-nl158-np7-rf0-random (query)           30_858.60       925.52    31_784.12       0.0982          NaN         5.67
IVF-Binary-512-nl158-np12-rf0-random (query)          30_858.60       931.74    31_790.34       0.0974          NaN         5.67
IVF-Binary-512-nl158-np17-rf0-random (query)          30_858.60       945.79    31_804.39       0.0970          NaN         5.67
IVF-Binary-512-nl158-np7-rf10-random (query)          30_858.60     1_049.59    31_908.19       0.2773   624.029620         5.67
IVF-Binary-512-nl158-np7-rf20-random (query)          30_858.60     1_172.95    32_031.55       0.4162   378.351283         5.67
IVF-Binary-512-nl158-np12-rf10-random (query)         30_858.60     1_060.41    31_919.01       0.2731   636.515426         5.67
IVF-Binary-512-nl158-np12-rf20-random (query)         30_858.60     1_193.62    32_052.22       0.4113   387.137457         5.67
IVF-Binary-512-nl158-np17-rf10-random (query)         30_858.60     1_065.71    31_924.31       0.2721   639.488030         5.67
IVF-Binary-512-nl158-np17-rf20-random (query)         30_858.60     1_206.47    32_065.07       0.4101   389.028420         5.67
IVF-Binary-512-nl158-random (self)                    30_858.60     3_431.79    34_290.39       0.2735   635.828630         5.67
IVF-Binary-512-nl223-np11-rf0-random (query)          25_680.88       943.92    26_624.80       0.0982          NaN         5.92
IVF-Binary-512-nl223-np14-rf0-random (query)          25_680.88       949.31    26_630.19       0.0977          NaN         5.92
IVF-Binary-512-nl223-np21-rf0-random (query)          25_680.88       961.91    26_642.78       0.0973          NaN         5.92
IVF-Binary-512-nl223-np11-rf10-random (query)         25_680.88     1_070.50    26_751.37       0.2762   630.318794         5.92
IVF-Binary-512-nl223-np11-rf20-random (query)         25_680.88     1_195.29    26_876.17       0.4155   381.626294         5.92
IVF-Binary-512-nl223-np14-rf10-random (query)         25_680.88     1_070.41    26_751.29       0.2747   635.018110         5.92
IVF-Binary-512-nl223-np14-rf20-random (query)         25_680.88     1_201.27    26_882.15       0.4129   385.392835         5.92
IVF-Binary-512-nl223-np21-rf10-random (query)         25_680.88     1_083.80    26_764.67       0.2731   638.223116         5.92
IVF-Binary-512-nl223-np21-rf20-random (query)         25_680.88     1_219.18    26_900.05       0.4110   387.907665         5.92
IVF-Binary-512-nl223-random (self)                    25_680.88     3_523.34    29_204.21       0.2746   634.150983         5.92
IVF-Binary-512-nl316-np15-rf0-random (query)          26_393.41       983.91    27_377.32       0.0980          NaN         6.29
IVF-Binary-512-nl316-np17-rf0-random (query)          26_393.41       982.46    27_375.87       0.0977          NaN         6.29
IVF-Binary-512-nl316-np25-rf0-random (query)          26_393.41       998.11    27_391.52       0.0973          NaN         6.29
IVF-Binary-512-nl316-np15-rf10-random (query)         26_393.41     1_112.27    27_505.68       0.2746   632.970108         6.29
IVF-Binary-512-nl316-np15-rf20-random (query)         26_393.41     1_234.87    27_628.28       0.4130   384.249081         6.29
IVF-Binary-512-nl316-np17-rf10-random (query)         26_393.41     1_109.47    27_502.88       0.2736   635.378984         6.29
IVF-Binary-512-nl316-np17-rf20-random (query)         26_393.41     1_239.21    27_632.62       0.4116   386.282888         6.29
IVF-Binary-512-nl316-np25-rf10-random (query)         26_393.41     1_120.27    27_513.68       0.2721   639.758354         6.29
IVF-Binary-512-nl316-np25-rf20-random (query)         26_393.41     1_256.91    27_650.32       0.4096   389.506176         6.29
IVF-Binary-512-nl316-random (self)                    26_393.41     3_613.25    30_006.66       0.2741   634.027808         6.29
IVF-Binary-512-nl158-np7-rf0-pca (query)              32_573.97       987.33    33_561.30       0.2054          NaN         5.67
IVF-Binary-512-nl158-np12-rf0-pca (query)             32_573.97       946.97    33_520.93       0.2044          NaN         5.67
IVF-Binary-512-nl158-np17-rf0-pca (query)             32_573.97       958.82    33_532.79       0.2036          NaN         5.67
IVF-Binary-512-nl158-np7-rf10-pca (query)             32_573.97     1_063.99    33_637.96       0.5922   204.435433         5.67
IVF-Binary-512-nl158-np7-rf20-pca (query)             32_573.97     1_198.59    33_772.56       0.7493   100.086773         5.67
IVF-Binary-512-nl158-np12-rf10-pca (query)            32_573.97     1_074.41    33_648.37       0.5875   207.134808         5.67
IVF-Binary-512-nl158-np12-rf20-pca (query)            32_573.97     1_211.42    33_785.39       0.7447    99.908972         5.67
IVF-Binary-512-nl158-np17-rf10-pca (query)            32_573.97     1_089.68    33_663.64       0.5803   213.737524         5.67
IVF-Binary-512-nl158-np17-rf20-pca (query)            32_573.97     1_239.32    33_813.29       0.7334   106.370859         5.67
IVF-Binary-512-nl158-pca (self)                       32_573.97     3_518.83    36_092.79       0.5881   206.168604         5.67
IVF-Binary-512-nl223-np11-rf0-pca (query)             27_368.53       960.41    28_328.94       0.2055          NaN         5.93
IVF-Binary-512-nl223-np14-rf0-pca (query)             27_368.53       966.22    28_334.75       0.2048          NaN         5.93
IVF-Binary-512-nl223-np21-rf0-pca (query)             27_368.53       991.55    28_360.08       0.2039          NaN         5.93
IVF-Binary-512-nl223-np11-rf10-pca (query)            27_368.53     1_085.64    28_454.17       0.5948   201.582794         5.93
IVF-Binary-512-nl223-np11-rf20-pca (query)            27_368.53     1_221.21    28_589.74       0.7555    94.992844         5.93
IVF-Binary-512-nl223-np14-rf10-pca (query)            27_368.53     1_088.10    28_456.62       0.5919   203.790325         5.93
IVF-Binary-512-nl223-np14-rf20-pca (query)            27_368.53     1_225.42    28_593.95       0.7516    96.626563         5.93
IVF-Binary-512-nl223-np21-rf10-pca (query)            27_368.53     1_104.33    28_472.86       0.5837   210.922207         5.93
IVF-Binary-512-nl223-np21-rf20-pca (query)            27_368.53     1_253.42    28_621.95       0.7392   103.811896         5.93
IVF-Binary-512-nl223-pca (self)                       27_368.53     3_574.30    30_942.83       0.5927   202.864594         5.93
IVF-Binary-512-nl316-np15-rf0-pca (query)             28_164.32     1_004.60    29_168.93       0.2055          NaN         6.29
IVF-Binary-512-nl316-np17-rf0-pca (query)             28_164.32       997.55    29_161.87       0.2053          NaN         6.29
IVF-Binary-512-nl316-np25-rf0-pca (query)             28_164.32     1_011.46    29_175.78       0.2045          NaN         6.29
IVF-Binary-512-nl316-np15-rf10-pca (query)            28_164.32     1_125.04    29_289.36       0.5954   201.060118         6.29
IVF-Binary-512-nl316-np15-rf20-pca (query)            28_164.32     1_269.92    29_434.25       0.7574    93.885052         6.29
IVF-Binary-512-nl316-np17-rf10-pca (query)            28_164.32     1_118.50    29_282.82       0.5943   201.938907         6.29
IVF-Binary-512-nl316-np17-rf20-pca (query)            28_164.32     1_265.96    29_430.28       0.7558    94.462146         6.29
IVF-Binary-512-nl316-np25-rf10-pca (query)            28_164.32     1_130.48    29_294.80       0.5873   207.893232         6.29
IVF-Binary-512-nl316-np25-rf20-pca (query)            28_164.32     1_281.77    29_446.10       0.7451   100.220611         6.29
IVF-Binary-512-nl316-pca (self)                       28_164.32     3_658.22    31_822.55       0.5954   200.590079         6.29
IVF-Binary-1024-nl158-np7-rf0-random (query)          54_473.11     1_822.26    56_295.37       0.1150          NaN        10.72
IVF-Binary-1024-nl158-np12-rf0-random (query)         54_473.11     1_805.50    56_278.61       0.1141          NaN        10.72
IVF-Binary-1024-nl158-np17-rf0-random (query)         54_473.11     1_820.53    56_293.64       0.1138          NaN        10.72
IVF-Binary-1024-nl158-np7-rf10-random (query)         54_473.11     1_896.58    56_369.69       0.3437   488.216685        10.72
IVF-Binary-1024-nl158-np7-rf20-random (query)         54_473.11     2_019.61    56_492.72       0.4987   280.616590        10.72
IVF-Binary-1024-nl158-np12-rf10-random (query)        54_473.11     1_934.83    56_407.94       0.3407   495.004071        10.72
IVF-Binary-1024-nl158-np12-rf20-random (query)        54_473.11     2_042.90    56_516.01       0.4958   284.471007        10.72
IVF-Binary-1024-nl158-np17-rf10-random (query)        54_473.11     1_962.09    56_435.20       0.3403   496.161364        10.72
IVF-Binary-1024-nl158-np17-rf20-random (query)        54_473.11     2_069.15    56_542.25       0.4951   285.270903        10.72
IVF-Binary-1024-nl158-random (self)                   54_473.11     6_287.87    60_760.98       0.3426   490.883418        10.72
IVF-Binary-1024-nl223-np11-rf0-random (query)         49_472.35     1_812.26    51_284.61       0.1146          NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-random (query)         49_472.35     1_820.45    51_292.81       0.1144          NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-random (query)         49_472.35     1_831.91    51_304.26       0.1142          NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-random (query)        49_472.35     1_915.91    51_388.26       0.3435   490.494880        10.98
IVF-Binary-1024-nl223-np11-rf20-random (query)        49_472.35     2_046.92    51_519.27       0.4983   281.598746        10.98
IVF-Binary-1024-nl223-np14-rf10-random (query)        49_472.35     1_923.62    51_395.97       0.3421   493.150925        10.98
IVF-Binary-1024-nl223-np14-rf20-random (query)        49_472.35     2_055.43    51_527.78       0.4972   283.029055        10.98
IVF-Binary-1024-nl223-np21-rf10-random (query)        49_472.35     1_943.80    51_416.15       0.3410   495.035352        10.98
IVF-Binary-1024-nl223-np21-rf20-random (query)        49_472.35     2_100.08    51_572.43       0.4959   284.383033        10.98
IVF-Binary-1024-nl223-random (self)                   49_472.35     6_325.57    55_797.92       0.3434   489.503266        10.98
IVF-Binary-1024-nl316-np15-rf0-random (query)         49_882.71     1_843.49    51_726.21       0.1142          NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-random (query)         49_882.71     1_862.42    51_745.13       0.1140          NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-random (query)         49_882.71     1_856.03    51_738.74       0.1136          NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-random (query)        49_882.71     1_953.60    51_836.32       0.3433   490.876614        11.34
IVF-Binary-1024-nl316-np15-rf20-random (query)        49_882.71     2_080.57    51_963.28       0.4977   282.352173        11.34
IVF-Binary-1024-nl316-np17-rf10-random (query)        49_882.71     1_951.80    51_834.52       0.3425   492.360543        11.34
IVF-Binary-1024-nl316-np17-rf20-random (query)        49_882.71     2_155.02    52_037.74       0.4969   283.321269        11.34
IVF-Binary-1024-nl316-np25-rf10-random (query)        49_882.71     1_962.81    51_845.53       0.3413   494.738410        11.34
IVF-Binary-1024-nl316-np25-rf20-random (query)        49_882.71     2_104.21    51_986.93       0.4955   285.035035        11.34
IVF-Binary-1024-nl316-random (self)                   49_882.71     6_415.99    56_298.70       0.3432   489.252071        11.34
IVF-Binary-1024-nl158-np7-rf0-pca (query)             56_317.46     1_809.10    58_126.56       0.2286          NaN        10.73
IVF-Binary-1024-nl158-np12-rf0-pca (query)            56_317.46     1_828.49    58_145.95       0.2269          NaN        10.73
IVF-Binary-1024-nl158-np17-rf0-pca (query)            56_317.46     1_984.89    58_302.35       0.2252          NaN        10.73
IVF-Binary-1024-nl158-np7-rf10-pca (query)            56_317.46     1_918.90    58_236.36       0.6376   168.479023        10.73
IVF-Binary-1024-nl158-np7-rf20-pca (query)            56_317.46     2_049.58    58_367.04       0.7858    80.537891        10.73
IVF-Binary-1024-nl158-np12-rf10-pca (query)           56_317.46     1_938.67    58_256.13       0.6275   174.238618        10.73
IVF-Binary-1024-nl158-np12-rf20-pca (query)           56_317.46     2_092.00    58_409.46       0.7753    82.241313        10.73
IVF-Binary-1024-nl158-np17-rf10-pca (query)           56_317.46     1_988.76    58_306.22       0.6147   184.908878        10.73
IVF-Binary-1024-nl158-np17-rf20-pca (query)           56_317.46     2_117.84    58_435.30       0.7569    92.295688        10.73
IVF-Binary-1024-nl158-pca (self)                      56_317.46     6_399.11    62_716.57       0.6296   172.171405        10.73
IVF-Binary-1024-nl223-np11-rf0-pca (query)            51_035.61     1_830.67    52_866.28       0.2286          NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-pca (query)            51_035.61     1_842.28    52_877.89       0.2277          NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-pca (query)            51_035.61     1_860.78    52_896.39       0.2260          NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-pca (query)           51_035.61     1_940.18    52_975.79       0.6423   163.996219        10.98
IVF-Binary-1024-nl223-np11-rf20-pca (query)           51_035.61     2_079.05    53_114.66       0.7947    74.032733        10.98
IVF-Binary-1024-nl223-np14-rf10-pca (query)           51_035.61     1_980.94    53_016.55       0.6366   168.001461        10.98
IVF-Binary-1024-nl223-np14-rf20-pca (query)           51_035.61     2_084.70    53_120.31       0.7873    77.080297        10.98
IVF-Binary-1024-nl223-np21-rf10-pca (query)           51_035.61     2_098.58    53_134.19       0.6210   180.418392        10.98
IVF-Binary-1024-nl223-np21-rf20-pca (query)           51_035.61     2_116.13    53_151.74       0.7667    87.990148        10.98
IVF-Binary-1024-nl223-pca (self)                      51_035.61     6_425.08    57_460.69       0.6384   166.337776        10.98
IVF-Binary-1024-nl316-np15-rf0-pca (query)            51_567.55     1_866.36    53_433.91       0.2288          NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-pca (query)            51_567.55     1_871.64    53_439.18       0.2284          NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-pca (query)            51_567.55     1_885.66    53_453.20       0.2269          NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-pca (query)           51_567.55     1_975.99    53_543.54       0.6440   162.709068        11.34
IVF-Binary-1024-nl316-np15-rf20-pca (query)           51_567.55     2_108.48    53_676.03       0.7981    72.383640        11.34
IVF-Binary-1024-nl316-np17-rf10-pca (query)           51_567.55     1_985.97    53_553.52       0.6418   164.051537        11.34
IVF-Binary-1024-nl316-np17-rf20-pca (query)           51_567.55     2_111.75    53_679.29       0.7951    73.578108        11.34
IVF-Binary-1024-nl316-np25-rf10-pca (query)           51_567.55     1_995.59    53_563.14       0.6286   174.270035        11.34
IVF-Binary-1024-nl316-np25-rf20-pca (query)           51_567.55     2_140.33    53_707.88       0.7771    82.518195        11.34
IVF-Binary-1024-nl316-pca (self)                      51_567.55     6_545.75    58_113.30       0.6436   162.455399        11.34
IVF-Binary-1024-nl158-np7-rf0-signed (query)          54_671.06     1_783.77    56_454.83       0.1150          NaN        10.72
IVF-Binary-1024-nl158-np12-rf0-signed (query)         54_671.06     1_798.72    56_469.77       0.1141          NaN        10.72
IVF-Binary-1024-nl158-np17-rf0-signed (query)         54_671.06     1_823.32    56_494.37       0.1138          NaN        10.72
IVF-Binary-1024-nl158-np7-rf10-signed (query)         54_671.06     1_888.12    56_559.17       0.3437   488.216685        10.72
IVF-Binary-1024-nl158-np7-rf20-signed (query)         54_671.06     2_016.04    56_687.10       0.4987   280.616590        10.72
IVF-Binary-1024-nl158-np12-rf10-signed (query)        54_671.06     1_901.33    56_572.39       0.3407   495.004071        10.72
IVF-Binary-1024-nl158-np12-rf20-signed (query)        54_671.06     2_058.31    56_729.37       0.4958   284.471007        10.72
IVF-Binary-1024-nl158-np17-rf10-signed (query)        54_671.06     1_937.56    56_608.62       0.3403   496.161364        10.72
IVF-Binary-1024-nl158-np17-rf20-signed (query)        54_671.06     2_072.71    56_743.77       0.4951   285.270903        10.72
IVF-Binary-1024-nl158-signed (self)                   54_671.06     6_266.74    60_937.80       0.3426   490.883418        10.72
IVF-Binary-1024-nl223-np11-rf0-signed (query)         49_738.44     1_819.23    51_557.67       0.1146          NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-signed (query)         49_738.44     1_845.26    51_583.70       0.1144          NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-signed (query)         49_738.44     1_832.11    51_570.55       0.1142          NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-signed (query)        49_738.44     1_925.97    51_664.40       0.3435   490.494880        10.98
IVF-Binary-1024-nl223-np11-rf20-signed (query)        49_738.44     2_059.29    51_797.73       0.4983   281.598746        10.98
IVF-Binary-1024-nl223-np14-rf10-signed (query)        49_738.44     1_943.99    51_682.42       0.3421   493.150925        10.98
IVF-Binary-1024-nl223-np14-rf20-signed (query)        49_738.44     2_071.56    51_810.00       0.4972   283.029055        10.98
IVF-Binary-1024-nl223-np21-rf10-signed (query)        49_738.44     1_950.80    51_689.24       0.3410   495.035352        10.98
IVF-Binary-1024-nl223-np21-rf20-signed (query)        49_738.44     2_121.85    51_860.29       0.4959   284.383033        10.98
IVF-Binary-1024-nl223-signed (self)                   49_738.44     6_384.07    56_122.51       0.3434   489.503266        10.98
IVF-Binary-1024-nl316-np15-rf0-signed (query)         50_335.17     1_849.29    52_184.46       0.1142          NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-signed (query)         50_335.17     1_849.34    52_184.51       0.1140          NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-signed (query)         50_335.17     1_870.10    52_205.27       0.1136          NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-signed (query)        50_335.17     1_956.99    52_292.16       0.3433   490.876614        11.34
IVF-Binary-1024-nl316-np15-rf20-signed (query)        50_335.17     2_083.85    52_419.02       0.4977   282.352173        11.34
IVF-Binary-1024-nl316-np17-rf10-signed (query)        50_335.17     1_949.37    52_284.54       0.3425   492.360543        11.34
IVF-Binary-1024-nl316-np17-rf20-signed (query)        50_335.17     2_076.27    52_411.44       0.4969   283.321269        11.34
IVF-Binary-1024-nl316-np25-rf10-signed (query)        50_335.17     1_992.93    52_328.10       0.3413   494.738410        11.34
IVF-Binary-1024-nl316-np25-rf20-signed (query)        50_335.17     2_107.84    52_443.01       0.4955   285.035035        11.34
IVF-Binary-1024-nl316-signed (self)                   50_335.17     6_408.33    56_743.50       0.3432   489.252071        11.34
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 256D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         9.84     4_167.22     4_177.06       1.0000     0.000000        48.83
Exhaustive (self)                                          9.84    14_041.79    14_051.63       1.0000     0.000000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_566.75       247.68     2_814.44       0.0452          NaN         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_566.75       342.09     2_908.84       0.2166   205.987976         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_566.75       443.44     3_010.19       0.3508   122.431890         1.78
ExhaustiveBinary-256-random (self)                     2_566.75     1_146.88     3_713.64       0.2176   204.182890         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_773.44       254.17     3_027.62       0.1506          NaN         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_773.44       366.33     3_139.78       0.3709   121.391667         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_773.44       469.86     3_243.30       0.4756    78.206768         1.78
ExhaustiveBinary-256-pca (self)                        2_773.44     1_200.92     3_974.36       0.3713   121.705212         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_087.85       445.62     5_533.47       0.0909          NaN         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_087.85       548.37     5_636.23       0.2883   148.489435         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_087.85       654.33     5_742.18       0.4362    86.638439         3.55
ExhaustiveBinary-512-random (self)                     5_087.85     1_816.92     6_904.77       0.2882   148.203149         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_393.95       456.93     5_850.88       0.1952          NaN         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_393.95       566.89     5_960.84       0.5963    47.580116         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_393.95       675.48     6_069.43       0.7690    20.556593         3.55
ExhaustiveBinary-512-pca (self)                        5_393.95     1_879.66     7_273.60       0.5967    47.561589         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_089.55       764.53    10_854.08       0.1215          NaN         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_089.55       875.55    10_965.10       0.3776   108.177948         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_089.55       996.57    11_086.12       0.5431    59.455892         7.10
ExhaustiveBinary-1024-random (self)                   10_089.55     2_899.62    12_989.16       0.3784   108.128717         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_570.64       776.36    11_347.00       0.2039          NaN         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_570.64       895.75    11_466.39       0.6175    43.357743         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_570.64     1_021.61    11_592.25       0.7866    18.300056         7.10
ExhaustiveBinary-1024-pca (self)                      10_570.64     3_080.97    13_651.60       0.6179    43.368804         7.10
ExhaustiveBinary-256-signed_no_rr (query)              2_569.79       256.79     2_826.58       0.0452          NaN         1.78
ExhaustiveBinary-256-signed-rf10 (query)               2_569.79       361.07     2_930.86       0.2166   205.987976         1.78
ExhaustiveBinary-256-signed-rf20 (query)               2_569.79       451.05     3_020.84       0.3508   122.431890         1.78
ExhaustiveBinary-256-signed (self)                     2_569.79     1_156.49     3_726.28       0.2176   204.182890         1.78
IVF-Binary-256-nl158-np7-rf0-random (query)            4_125.59       112.95     4_238.54       0.0662          NaN         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           4_125.59       127.43     4_253.02       0.0556          NaN         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           4_125.59       124.89     4_250.48       0.0468          NaN         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           4_125.59       168.92     4_294.52       0.2635   169.087863         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           4_125.59       217.48     4_343.08       0.4000   100.994009         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          4_125.59       180.44     4_306.03       0.2383   192.222496         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          4_125.59       233.02     4_358.61       0.3719   115.013704         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          4_125.59       187.03     4_312.62       0.2214   205.864740         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          4_125.59       247.05     4_372.64       0.3563   121.711576         1.93
IVF-Binary-256-nl158-random (self)                     4_125.59       510.43     4_636.02       0.2390   190.283771         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           3_079.59       118.19     3_197.77       0.0609          NaN         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           3_079.59       121.40     3_200.98       0.0534          NaN         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           3_079.59       126.07     3_205.66       0.0480          NaN         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          3_079.59       174.29     3_253.88       0.2517   178.786345         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          3_079.59       222.84     3_302.43       0.3905   106.337417         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          3_079.59       173.95     3_253.53       0.2362   192.807913         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          3_079.59       227.42     3_307.00       0.3749   114.115562         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          3_079.59       186.72     3_266.30       0.2225   204.877325         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          3_079.59       246.50     3_326.09       0.3606   120.584444         2.00
IVF-Binary-256-nl223-random (self)                     3_079.59       512.83     3_592.41       0.2373   190.908868         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           3_314.58       155.92     3_470.51       0.0544          NaN         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           3_314.58       155.70     3_470.28       0.0510          NaN         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           3_314.58       140.65     3_455.24       0.0479          NaN         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          3_314.58       182.87     3_497.46       0.2417   188.172407         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          3_314.58       230.04     3_544.62       0.3824   110.268860         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          3_314.58       180.72     3_495.30       0.2352   193.477570         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          3_314.58       235.12     3_549.70       0.3752   113.301632         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          3_314.58       185.98     3_500.56       0.2242   202.664173         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          3_314.58       249.18     3_563.76       0.3607   119.902317         2.09
IVF-Binary-256-nl316-random (self)                     3_314.58       535.66     3_850.25       0.2360   191.766575         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               4_297.10       118.63     4_415.73       0.1910          NaN         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              4_297.10       124.11     4_421.21       0.1880          NaN         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              4_297.10       130.10     4_427.20       0.1857          NaN         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              4_297.10       188.07     4_485.17       0.5792    51.154231         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              4_297.10       244.18     4_541.28       0.7491    23.096768         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             4_297.10       196.81     4_493.91       0.5597    55.443907         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             4_297.10       260.09     4_557.19       0.7203    26.973621         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             4_297.10       209.78     4_506.88       0.5437    59.244054         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             4_297.10       277.96     4_575.06       0.6982    30.189041         1.93
IVF-Binary-256-nl158-pca (self)                        4_297.10       609.78     4_906.88       0.5602    55.447702         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              3_284.82       124.22     3_409.04       0.1904          NaN         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              3_284.82       135.00     3_419.82       0.1892          NaN         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              3_284.82       134.17     3_418.99       0.1866          NaN         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             3_284.82       194.67     3_479.49       0.5759    51.898385         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             3_284.82       262.40     3_547.22       0.7430    23.932507         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             3_284.82       198.54     3_483.36       0.5677    53.680257         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             3_284.82       260.05     3_544.87       0.7313    25.489634         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             3_284.82       210.41     3_495.23       0.5492    58.009133         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             3_284.82       280.22     3_565.04       0.7057    29.105095         2.00
IVF-Binary-256-nl223-pca (self)                        3_284.82       613.95     3_898.77       0.5678    53.723730         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              3_507.73       131.48     3_639.21       0.1904          NaN         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              3_507.73       134.36     3_642.08       0.1898          NaN         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              3_507.73       140.11     3_647.84       0.1877          NaN         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             3_507.73       203.73     3_711.46       0.5768    51.700603         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             3_507.73       265.02     3_772.74       0.7451    23.638011         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             3_507.73       203.63     3_711.36       0.5725    52.601198         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             3_507.73       266.45     3_774.18       0.7391    24.435369         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             3_507.73       212.91     3_720.63       0.5576    55.962929         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             3_507.73       279.57     3_787.30       0.7175    27.416049         2.09
IVF-Binary-256-nl316-pca (self)                        3_507.73       638.75     4_146.48       0.5728    52.611173         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_609.62       213.17     6_822.79       0.1005          NaN         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_609.62       220.49     6_830.11       0.0953          NaN         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_609.62       226.06     6_835.68       0.0917          NaN         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_609.62       271.20     6_880.82       0.3084   137.968385         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_609.62       319.95     6_929.57       0.4586    80.392889         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_609.62       281.79     6_891.41       0.2982   143.773270         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_609.62       337.32     6_946.95       0.4474    83.782076         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_609.62       298.24     6_907.86       0.2904   147.643335         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_609.62       384.97     6_994.59       0.4400    85.856107         3.71
IVF-Binary-512-nl158-random (self)                     6_609.62       870.64     7_480.26       0.2981   143.634415         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_573.78       213.91     5_787.70       0.0983          NaN         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_573.78       219.90     5_793.68       0.0950          NaN         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_573.78       228.81     5_802.60       0.0918          NaN         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_573.78       277.77     5_851.56       0.3061   139.118196         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_573.78       326.15     5_899.94       0.4558    81.158160         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_573.78       280.38     5_854.16       0.2997   142.536240         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_573.78       334.84     5_908.63       0.4489    83.148885         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_573.78       306.46     5_880.24       0.2927   146.561216         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_573.78       355.22     5_929.00       0.4409    85.487616         3.77
IVF-Binary-512-nl223-random (self)                     5_573.78       871.39     6_445.17       0.2997   142.434872         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           5_789.93       221.70     6_011.63       0.0959          NaN         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           5_789.93       224.25     6_014.19       0.0943          NaN         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           5_789.93       230.73     6_020.67       0.0922          NaN         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          5_789.93       282.64     6_072.58       0.3031   140.754656         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          5_789.93       338.79     6_128.72       0.4538    81.920226         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          5_789.93       285.43     6_075.37       0.2990   142.847798         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          5_789.93       350.16     6_140.10       0.4493    83.158852         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          5_789.93       295.25     6_085.18       0.2932   146.118013         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          5_789.93       355.53     6_145.46       0.4418    85.252773         3.86
IVF-Binary-512-nl316-random (self)                     5_789.93       886.32     6_676.26       0.2992   142.673085         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               7_006.98       216.88     7_223.86       0.1961          NaN         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              7_006.98       237.73     7_244.71       0.1957          NaN         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              7_006.98       238.56     7_245.54       0.1955          NaN         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              7_006.98       287.11     7_294.09       0.5971    47.379960         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              7_006.98       344.03     7_351.00       0.7699    20.454958         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             7_006.98       314.14     7_321.12       0.5962    47.558431         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             7_006.98       367.54     7_374.52       0.7692    20.535237         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             7_006.98       319.98     7_326.96       0.5961    47.585555         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             7_006.98       384.69     7_391.67       0.7691    20.550685         3.71
IVF-Binary-512-nl158-pca (self)                        7_006.98       945.77     7_952.75       0.5969    47.517749         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              5_944.27       223.20     6_167.47       0.1958          NaN         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              5_944.27       229.95     6_174.22       0.1956          NaN         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              5_944.27       239.59     6_183.86       0.1953          NaN         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             5_944.27       300.28     6_244.55       0.5979    47.263353         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             5_944.27       353.91     6_298.18       0.7698    20.452679         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             5_944.27       297.34     6_241.61       0.5973    47.365307         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             5_944.27       358.02     6_302.29       0.7698    20.457916         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             5_944.27       315.90     6_260.17       0.5965    47.523414         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             5_944.27       379.59     6_323.86       0.7691    20.542325         3.77
IVF-Binary-512-nl223-pca (self)                        5_944.27       955.20     6_899.46       0.5976    47.376655         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              6_072.35       229.43     6_301.78       0.1959          NaN         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              6_072.35       232.45     6_304.79       0.1956          NaN         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              6_072.35       240.95     6_313.29       0.1954          NaN         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             6_072.35       305.27     6_377.62       0.5973    47.344398         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             6_072.35       364.99     6_437.33       0.7700    20.447206         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             6_072.35       306.43     6_378.78       0.5969    47.444321         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             6_072.35       365.00     6_437.35       0.7696    20.491324         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             6_072.35       316.28     6_388.62       0.5962    47.593749         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             6_072.35       378.87     6_451.21       0.7690    20.583655         3.86
IVF-Binary-512-nl316-pca (self)                        6_072.35       953.85     7_026.19       0.5976    47.381001         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_545.02       399.17    11_944.19       0.1266          NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_545.02       414.27    11_959.29       0.1243          NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_545.02       431.06    11_976.07       0.1224          NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_545.02       463.83    12_008.85       0.3899   103.659598         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_545.02       522.83    12_067.84       0.5551    56.926553         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_545.02       482.79    12_027.80       0.3839   106.017893         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_545.02       543.01    12_088.03       0.5487    58.380620         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_545.02       505.41    12_050.43       0.3796   107.638093         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_545.02       571.32    12_116.33       0.5450    59.204414         7.26
IVF-Binary-1024-nl158-random (self)                   11_545.02     1_540.56    13_085.58       0.3839   106.247673         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_476.05       407.05    10_883.09       0.1256          NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_476.05       412.30    10_888.34       0.1242          NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_476.05       432.66    10_908.70       0.1226          NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_476.05       472.99    10_949.03       0.3876   104.409175         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_476.05       538.68    11_014.72       0.5548    56.969571         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_476.05       479.49    10_955.53       0.3837   105.851906         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_476.05       542.64    11_018.68       0.5508    57.823451         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_476.05       500.81    10_976.86       0.3793   107.618742         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_476.05       565.55    11_041.60       0.5460    58.950479         7.32
IVF-Binary-1024-nl223-random (self)                   10_476.05     1_550.00    12_026.04       0.3848   105.699938         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         10_733.18       413.40    11_146.59       0.1247          NaN         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)         10_733.18       416.55    11_149.73       0.1239          NaN         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)         10_733.18       428.36    11_161.54       0.1228          NaN         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)        10_733.18       481.32    11_214.50       0.3872   104.698794         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)        10_733.18       537.08    11_270.26       0.5533    57.315277         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)        10_733.18       482.59    11_215.77       0.3847   105.620055         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)        10_733.18       542.96    11_276.14       0.5507    57.885432         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)        10_733.18       497.75    11_230.94       0.3803   107.285753         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)        10_733.18       564.56    11_297.75       0.5460    58.922896         7.41
IVF-Binary-1024-nl316-random (self)                   10_733.18     1_543.66    12_276.84       0.3852   105.585304         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)             11_989.52       415.54    12_405.07       0.2048          NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            11_989.52       435.33    12_424.86       0.2043          NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            11_989.52       446.41    12_435.93       0.2042          NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            11_989.52       487.98    12_477.50       0.6183    43.205649         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            11_989.52       545.88    12_535.40       0.7874    18.200739         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           11_989.52       509.49    12_499.01       0.6178    43.329266         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           11_989.52       569.48    12_559.00       0.7868    18.277544         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           11_989.52       531.11    12_520.64       0.6177    43.352025         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           11_989.52       594.86    12_584.38       0.7867    18.290436         7.26
IVF-Binary-1024-nl158-pca (self)                      11_989.52     1_622.86    13_612.38       0.6181    43.343560         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            10_970.04       424.65    11_394.69       0.2047          NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            10_970.04       435.87    11_405.91       0.2045          NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            10_970.04       448.24    11_418.28       0.2041          NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           10_970.04       501.60    11_471.64       0.6185    43.173007         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           10_970.04       556.66    11_526.70       0.7871    18.282360         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           10_970.04       500.41    11_470.45       0.6181    43.244707         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           10_970.04       571.41    11_541.45       0.7872    18.265315         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           10_970.04       527.44    11_497.48       0.6175    43.378396         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           10_970.04       590.57    11_560.61       0.7867    18.315876         7.32
IVF-Binary-1024-nl223-pca (self)                      10_970.04     1_608.32    12_578.36       0.6186    43.228058         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            11_194.91       428.90    11_623.81       0.2046          NaN         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            11_194.91       434.17    11_629.08       0.2044          NaN         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            11_194.91       445.01    11_639.92       0.2041          NaN         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           11_194.91       503.85    11_698.76       0.6186    43.140928         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           11_194.91       564.05    11_758.95       0.7875    18.201091         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           11_194.91       505.30    11_700.21       0.6181    43.234358         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           11_194.91       572.85    11_767.76       0.7874    18.237721         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           11_194.91       521.22    11_716.13       0.6174    43.372254         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           11_194.91       589.86    11_784.77       0.7870    18.296891         7.42
IVF-Binary-1024-nl316-pca (self)                      11_194.91     1_627.78    12_822.69       0.6186    43.235336         7.42
IVF-Binary-256-nl158-np7-rf0-signed (query)            4_072.85       112.95     4_185.80       0.0662          NaN         1.93
IVF-Binary-256-nl158-np12-rf0-signed (query)           4_072.85       127.26     4_200.11       0.0556          NaN         1.93
IVF-Binary-256-nl158-np17-rf0-signed (query)           4_072.85       125.01     4_197.86       0.0468          NaN         1.93
IVF-Binary-256-nl158-np7-rf10-signed (query)           4_072.85       172.86     4_245.71       0.2635   169.087863         1.93
IVF-Binary-256-nl158-np7-rf20-signed (query)           4_072.85       216.84     4_289.69       0.4000   100.994009         1.93
IVF-Binary-256-nl158-np12-rf10-signed (query)          4_072.85       174.90     4_247.75       0.2383   192.222496         1.93
IVF-Binary-256-nl158-np12-rf20-signed (query)          4_072.85       228.33     4_301.18       0.3719   115.013704         1.93
IVF-Binary-256-nl158-np17-rf10-signed (query)          4_072.85       186.21     4_259.06       0.2214   205.864740         1.93
IVF-Binary-256-nl158-np17-rf20-signed (query)          4_072.85       243.76     4_316.61       0.3563   121.711576         1.93
IVF-Binary-256-nl158-signed (self)                     4_072.85       511.51     4_584.35       0.2390   190.283771         1.93
IVF-Binary-256-nl223-np11-rf0-signed (query)           3_035.07       122.73     3_157.80       0.0609          NaN         2.00
IVF-Binary-256-nl223-np14-rf0-signed (query)           3_035.07       120.39     3_155.46       0.0534          NaN         2.00
IVF-Binary-256-nl223-np21-rf0-signed (query)           3_035.07       130.36     3_165.43       0.0480          NaN         2.00
IVF-Binary-256-nl223-np11-rf10-signed (query)          3_035.07       196.74     3_231.81       0.2517   178.786345         2.00
IVF-Binary-256-nl223-np11-rf20-signed (query)          3_035.07       227.39     3_262.47       0.3905   106.337417         2.00
IVF-Binary-256-nl223-np14-rf10-signed (query)          3_035.07       190.79     3_225.86       0.2362   192.807913         2.00
IVF-Binary-256-nl223-np14-rf20-signed (query)          3_035.07       237.66     3_272.73       0.3749   114.115562         2.00
IVF-Binary-256-nl223-np21-rf10-signed (query)          3_035.07       188.49     3_223.56       0.2225   204.877325         2.00
IVF-Binary-256-nl223-np21-rf20-signed (query)          3_035.07       241.42     3_276.49       0.3606   120.584444         2.00
IVF-Binary-256-nl223-signed (self)                     3_035.07       511.86     3_546.93       0.2373   190.908868         2.00
IVF-Binary-256-nl316-np15-rf0-signed (query)           3_261.12       125.38     3_386.50       0.0544          NaN         2.09
IVF-Binary-256-nl316-np17-rf0-signed (query)           3_261.12       126.00     3_387.12       0.0510          NaN         2.09
IVF-Binary-256-nl316-np25-rf0-signed (query)           3_261.12       130.33     3_391.45       0.0479          NaN         2.09
IVF-Binary-256-nl316-np15-rf10-signed (query)          3_261.12       179.86     3_440.98       0.2417   188.172407         2.09
IVF-Binary-256-nl316-np15-rf20-signed (query)          3_261.12       231.12     3_492.23       0.3824   110.268860         2.09
IVF-Binary-256-nl316-np17-rf10-signed (query)          3_261.12       180.48     3_441.59       0.2352   193.477570         2.09
IVF-Binary-256-nl316-np17-rf20-signed (query)          3_261.12       234.42     3_495.54       0.3752   113.301632         2.09
IVF-Binary-256-nl316-np25-rf10-signed (query)          3_261.12       186.88     3_447.99       0.2242   202.664173         2.09
IVF-Binary-256-nl316-np25-rf20-signed (query)          3_261.12       249.80     3_510.92       0.3607   119.902317         2.09
IVF-Binary-256-nl316-signed (self)                     3_261.12       545.61     3_806.72       0.2360   191.766575         2.09
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 512D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        19.39     9_810.90     9_830.30       1.0000     0.000000        97.66
Exhaustive (self)                                         19.39    32_555.18    32_574.58       1.0000     0.000000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_656.73       352.81     6_009.54       0.0301          NaN         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_656.73       464.06     6_120.78       0.1667   846.330982         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_656.73       585.73     6_242.46       0.2843   530.669319         2.03
ExhaustiveBinary-256-random (self)                     5_656.73     1_530.67     7_187.40       0.1675   839.210644         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_107.50       374.35     6_481.84       0.1949          NaN         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_107.50       559.51     6_667.01       0.5110   212.085163         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_107.50       655.83     6_763.33       0.6361   125.154332         2.03
ExhaustiveBinary-256-pca (self)                        6_107.50     1_689.35     7_796.85       0.5101   212.330485         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_149.92       675.24    11_825.15       0.0615          NaN         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_149.92       795.61    11_945.52       0.2173   624.008278         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_149.92       969.86    12_119.78       0.3430   379.162923         4.05
ExhaustiveBinary-512-random (self)                    11_149.92     2_628.53    13_778.45       0.2175   620.171217         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_716.09       720.59    12_436.67       0.1629          NaN         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_716.09       828.26    12_544.35       0.4096   343.851433         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_716.09       966.73    12_682.81       0.5240   205.227569         4.05
ExhaustiveBinary-512-pca (self)                       11_716.09     2_735.93    14_452.02       0.4100   415.184108         4.05
ExhaustiveBinary-1024-random_no_rr (query)            22_017.75     1_215.78    23_233.53       0.0926          NaN         8.10
ExhaustiveBinary-1024-random-rf10 (query)             22_017.75     1_355.18    23_372.93       0.2661   498.520954         8.10
ExhaustiveBinary-1024-random-rf20 (query)             22_017.75     1_498.81    23_516.56       0.4064   299.575988         8.10
ExhaustiveBinary-1024-random (self)                   22_017.75     4_549.45    26_567.20       0.2658   498.605058         8.10
ExhaustiveBinary-1024-pca_no_rr (query)               23_051.01     1_267.24    24_318.26       0.2653          NaN         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                23_051.01     1_389.96    24_440.98       0.7317    76.940529         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                23_051.01     1_649.14    24_700.16       0.8761    27.420317         8.11
ExhaustiveBinary-1024-pca (self)                      23_051.01     4_608.70    27_659.72       0.7322    76.251329         8.11
ExhaustiveBinary-512-signed_no_rr (query)             11_141.15       678.15    11_819.30       0.0615          NaN         4.05
ExhaustiveBinary-512-signed-rf10 (query)              11_141.15       796.12    11_937.26       0.2173   624.008278         4.05
ExhaustiveBinary-512-signed-rf20 (query)              11_141.15       933.87    12_075.02       0.3430   379.162923         4.05
ExhaustiveBinary-512-signed (self)                    11_141.15     2_631.26    13_772.41       0.2175   620.171217         4.05
IVF-Binary-256-nl158-np7-rf0-random (query)            8_936.98       235.50     9_172.49       0.0472          NaN         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           8_936.98       243.03     9_180.01       0.0382          NaN         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           8_936.98       242.41     9_179.39       0.0307          NaN         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           8_936.98       311.29     9_248.27       0.2149   657.362367         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           8_936.98       391.64     9_328.62       0.3407   400.819184         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          8_936.98       314.89     9_251.87       0.1920   746.751582         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          8_936.98       401.41     9_338.39       0.3136   462.733927         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          8_936.98       320.14     9_257.12       0.1679   848.955367         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          8_936.98       408.13     9_345.11       0.2873   531.988518         2.34
IVF-Binary-256-nl158-random (self)                     8_936.98       950.26     9_887.24       0.1935   741.450269         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           6_605.54       242.79     6_848.33       0.0490          NaN         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)           6_605.54       244.61     6_850.15       0.0358          NaN         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)           6_605.54       249.54     6_855.08       0.0309          NaN         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)          6_605.54       323.39     6_928.94       0.2145   670.027037         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)          6_605.54       401.23     7_006.77       0.3388   413.521252         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)          6_605.54       321.00     6_926.55       0.1840   786.813887         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)          6_605.54       407.50     7_013.05       0.3093   478.245738         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)          6_605.54       330.54     6_936.08       0.1694   842.300268         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)          6_605.54       416.04     7_021.59       0.2924   518.970099         2.46
IVF-Binary-256-nl223-random (self)                     6_605.54       969.51     7_575.06       0.1846   781.373279         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)           7_078.82       255.98     7_334.79       0.0378          NaN         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           7_078.82       259.68     7_338.49       0.0348          NaN         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           7_078.82       262.81     7_341.63       0.0337          NaN         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          7_078.82       335.64     7_414.46       0.1931   758.216871         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          7_078.82       419.25     7_498.07       0.3198   459.985252         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          7_078.82       335.29     7_414.11       0.1829   797.876734         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          7_078.82       423.46     7_502.28       0.3085   487.133184         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          7_078.82       342.13     7_420.94       0.1778   817.466171         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          7_078.82       425.91     7_504.73       0.3015   500.912356         2.65
IVF-Binary-256-nl316-random (self)                     7_078.82     1_031.08     8_109.90       0.1829   794.601711         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               9_350.73       251.70     9_602.44       0.2130          NaN         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              9_350.73       245.29     9_596.03       0.2116          NaN         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              9_350.73       251.65     9_602.38       0.2106          NaN         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              9_350.73       337.70     9_688.44       0.6243   131.867137         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              9_350.73       428.20     9_778.94       0.7868    57.581118         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             9_350.73       344.84     9_695.57       0.6186   135.158996         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             9_350.73       438.39     9_789.13       0.7797    59.952946         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             9_350.73       356.17     9_706.90       0.6111   139.609277         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             9_350.73       455.99     9_806.73       0.7687    64.083177         2.34
IVF-Binary-256-nl158-pca (self)                        9_350.73     1_073.63    10_424.36       0.6181   135.250627         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              7_049.85       254.98     7_304.83       0.2128          NaN         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              7_049.85       257.75     7_307.60       0.2117          NaN         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              7_049.85       258.88     7_308.73       0.2107          NaN         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             7_049.85       348.93     7_398.78       0.6236   132.118425         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             7_049.85       438.46     7_488.31       0.7874    57.197673         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             7_049.85       352.35     7_402.20       0.6200   134.224822         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             7_049.85       449.14     7_499.00       0.7824    58.989236         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             7_049.85       363.15     7_413.00       0.6130   138.462978         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             7_049.85       463.88     7_513.74       0.7722    62.821064         2.47
IVF-Binary-256-nl223-pca (self)                        7_049.85     1_099.92     8_149.78       0.6199   134.182038         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              7_612.61       265.01     7_877.62       0.2128          NaN         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              7_612.61       270.64     7_883.25       0.2124          NaN         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              7_612.61       272.00     7_884.61       0.2115          NaN         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             7_612.61       365.27     7_977.87       0.6243   131.739695         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             7_612.61       453.47     8_066.08       0.7882    56.871810         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             7_612.61       365.95     7_978.56       0.6225   132.848126         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             7_612.61       459.98     8_072.59       0.7855    57.862793         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             7_612.61       373.75     7_986.35       0.6164   136.456816         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             7_612.61       474.58     8_087.19       0.7765    61.135787         2.65
IVF-Binary-256-nl316-pca (self)                        7_612.61     1_143.49     8_756.10       0.6221   132.910104         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           14_507.31       442.95    14_950.26       0.0757          NaN         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          14_507.31       447.75    14_955.06       0.0695          NaN         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          14_507.31       456.12    14_963.42       0.0619          NaN         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          14_507.31       522.48    15_029.79       0.2416   555.252678         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          14_507.31       599.85    15_107.16       0.3713   338.891854         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         14_507.31       532.03    15_039.33       0.2290   590.717676         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         14_507.31       634.43    15_141.74       0.3559   361.899684         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         14_507.31       546.04    15_053.35       0.2186   624.054862         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         14_507.31       632.94    15_140.25       0.3453   378.950941         4.36
IVF-Binary-512-nl158-random (self)                    14_507.31     1_683.38    16_190.69       0.2296   587.029234         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          12_203.74       449.41    12_653.16       0.0769          NaN         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          12_203.74       459.83    12_663.58       0.0673          NaN         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          12_203.74       461.99    12_665.73       0.0627          NaN         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         12_203.74       532.18    12_735.92       0.2389   566.708737         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         12_203.74       612.20    12_815.94       0.3689   348.116971         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         12_203.74       538.90    12_742.65       0.2253   603.381054         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         12_203.74       626.53    12_830.28       0.3536   367.691128         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         12_203.74       554.28    12_758.03       0.2190   620.681331         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         12_203.74       636.20    12_839.94       0.3476   375.031644         4.49
IVF-Binary-512-nl223-random (self)                    12_203.74     1_697.64    13_901.38       0.2254   601.858320         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          12_599.52       472.97    13_072.48       0.0682          NaN         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          12_599.52       470.52    13_070.04       0.0656          NaN         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          12_599.52       476.39    13_075.90       0.0647          NaN         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         12_599.52       552.39    13_151.91       0.2320   585.225999         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         12_599.52       628.32    13_227.84       0.3611   356.312394         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         12_599.52       545.50    13_145.01       0.2286   594.217807         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         12_599.52       632.15    13_231.67       0.3568   361.146342         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         12_599.52       570.18    13_169.70       0.2249   603.241949         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         12_599.52       779.69    13_379.21       0.3529   366.348422         4.67
IVF-Binary-512-nl316-random (self)                    12_599.52     2_054.61    14_654.13       0.2280   593.371378         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              15_264.31       458.03    15_722.34       0.2583          NaN         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             15_264.31       464.15    15_728.47       0.2504          NaN         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             15_264.31       469.56    15_733.87       0.2420          NaN         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             15_264.31       563.00    15_827.32       0.7094    86.877683         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             15_264.31       639.70    15_904.01       0.8517    35.080538         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            15_264.31       564.24    15_828.56       0.6823    99.503250         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            15_264.31       655.63    15_919.94       0.8266    42.211664         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            15_264.31       590.74    15_855.05       0.6544   114.019612         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            15_264.31       741.23    16_005.54       0.7982    51.886519         4.36
IVF-Binary-512-nl158-pca (self)                       15_264.31     1_828.54    17_092.86       0.6840    98.411011         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             12_763.91       466.25    13_230.15       0.2563          NaN         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             12_763.91       477.04    13_240.94       0.2522          NaN         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             12_763.91       478.97    13_242.87       0.2442          NaN         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            12_763.91       594.15    13_358.06       0.7034    89.449241         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            12_763.91       653.04    13_416.94       0.8482    35.623604         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            12_763.91       567.81    13_331.71       0.6899    95.700913         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            12_763.91       662.75    13_426.66       0.8345    39.739986         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            12_763.91       592.00    13_355.91       0.6626   109.498429         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            12_763.91       682.92    13_446.83       0.8061    49.049948         4.49
IVF-Binary-512-nl223-pca (self)                       12_763.91     1_826.89    14_590.79       0.6914    94.894325         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             13_308.41       478.04    13_786.44       0.2569          NaN         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             13_308.41       492.35    13_800.76       0.2548          NaN         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             13_308.41       487.45    13_795.86       0.2477          NaN         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            13_308.41       576.94    13_885.35       0.7059    88.384892         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            13_308.41       674.35    13_982.76       0.8505    34.853125         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            13_308.41       576.05    13_884.46       0.6991    91.549174         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            13_308.41       684.46    13_992.87       0.8434    37.008237         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            13_308.41       596.14    13_904.55       0.6745   103.357370         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            13_308.41       690.94    13_999.35       0.8190    44.728231         4.67
IVF-Binary-512-nl316-pca (self)                       13_308.41     1_883.76    15_192.17       0.6999    90.839997         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          25_465.33       844.25    26_309.58       0.0973          NaN         8.41
IVF-Binary-1024-nl158-np12-rf0-random (query)         25_465.33       863.27    26_328.61       0.0953          NaN         8.41
IVF-Binary-1024-nl158-np17-rf0-random (query)         25_465.33       874.84    26_340.17       0.0929          NaN         8.41
IVF-Binary-1024-nl158-np7-rf10-random (query)         25_465.33       930.88    26_396.21       0.2786   477.035041         8.41
IVF-Binary-1024-nl158-np7-rf20-random (query)         25_465.33     1_005.70    26_471.03       0.4204   287.003771         8.41
IVF-Binary-1024-nl158-np12-rf10-random (query)        25_465.33       938.31    26_403.64       0.2713   489.178423         8.41
IVF-Binary-1024-nl158-np12-rf20-random (query)        25_465.33     1_022.99    26_488.33       0.4120   294.860497         8.41
IVF-Binary-1024-nl158-np17-rf10-random (query)        25_465.33       955.79    26_421.12       0.2663   499.060952         8.41
IVF-Binary-1024-nl158-np17-rf20-random (query)        25_465.33     1_046.72    26_512.06       0.4068   299.884730         8.41
IVF-Binary-1024-nl158-random (self)                   25_465.33     3_037.86    28_503.20       0.2717   489.009328         8.41
IVF-Binary-1024-nl223-np11-rf0-random (query)         23_022.23       857.68    23_879.92       0.0968          NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         23_022.23       869.79    23_892.02       0.0944          NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         23_022.23       882.42    23_904.65       0.0930          NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        23_022.23       938.63    23_960.87       0.2785   477.606498         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        23_022.23     1_038.75    24_060.98       0.4196   287.127240         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        23_022.23       949.07    23_971.31       0.2714   489.607184         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        23_022.23     1_036.22    24_058.45       0.4118   294.721274         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        23_022.23       971.27    23_993.50       0.2684   495.503396         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        23_022.23     1_054.71    24_076.95       0.4078   298.572155         8.54
IVF-Binary-1024-nl223-random (self)                   23_022.23     3_067.57    26_089.80       0.2712   489.989697         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         23_447.99       877.96    24_325.95       0.0956          NaN         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)         23_447.99       878.19    24_326.18       0.0948          NaN         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)         23_447.99       895.98    24_343.97       0.0941          NaN         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)        23_447.99       953.55    24_401.54       0.2747   484.020708         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)        23_447.99     1_123.29    24_571.28       0.4162   290.424189         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)        23_447.99       955.71    24_403.70       0.2724   488.257631         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)        23_447.99     1_042.18    24_490.17       0.4131   293.356372         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)        23_447.99       978.32    24_426.32       0.2703   492.068087         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)        23_447.99     1_060.24    24_508.23       0.4108   295.554918         8.72
IVF-Binary-1024-nl316-random (self)                   23_447.99     3_108.47    26_556.47       0.2725   488.008450         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)             26_315.75       877.09    27_192.84       0.2656          NaN         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            26_315.75       891.33    27_207.09       0.2654          NaN         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            26_315.75       903.16    27_218.92       0.2653          NaN         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            26_315.75       959.69    27_275.45       0.7289    78.397498         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            26_315.75     1_047.84    27_363.60       0.8706    29.635272         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           26_315.75       977.50    27_293.25       0.7314    77.015011         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           26_315.75     1_075.78    27_391.53       0.8758    27.491818         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           26_315.75       997.02    27_312.78       0.7314    77.030688         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           26_315.75     1_161.13    27_476.88       0.8758    27.507367         8.42
IVF-Binary-1024-nl158-pca (self)                      26_315.75     3_454.56    29_770.31       0.7322    76.270579         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            24_101.09       858.40    24_959.49       0.2655          NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            24_101.09       863.90    24_964.98       0.2654          NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            24_101.09       874.45    24_975.53       0.2653          NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           24_101.09       950.76    25_051.85       0.7313    77.190593         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           24_101.09     1_034.90    25_135.99       0.8755    27.681635         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           24_101.09       970.21    25_071.29       0.7315    77.060386         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           24_101.09     1_048.29    25_149.38       0.8762    27.408699         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           24_101.09       977.79    25_078.88       0.7315    77.097993         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           24_101.09     1_077.20    25_178.28       0.8762    27.410547         8.54
IVF-Binary-1024-nl223-pca (self)                      24_101.09     3_131.53    27_232.61       0.7323    76.188370         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            24_441.85       871.76    25_313.60       0.2656          NaN         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            24_441.85       873.48    25_315.33       0.2655          NaN         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            24_441.85       899.63    25_341.48       0.2654          NaN         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           24_441.85       969.38    25_411.22       0.7317    76.948456         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           24_441.85     1_061.76    25_503.61       0.8756    27.624517         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           24_441.85       969.45    25_411.29       0.7318    76.886620         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           24_441.85     1_056.92    25_498.77       0.8760    27.481555         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           24_441.85       983.79    25_425.63       0.7317    76.941404         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           24_441.85     1_075.52    25_517.36       0.8761    27.460127         8.73
IVF-Binary-1024-nl316-pca (self)                      24_441.85     3_156.13    27_597.97       0.7325    76.128971         8.73
IVF-Binary-512-nl158-np7-rf0-signed (query)           14_284.08       423.14    14_707.22       0.0757          NaN         4.36
IVF-Binary-512-nl158-np12-rf0-signed (query)          14_284.08       430.95    14_715.03       0.0695          NaN         4.36
IVF-Binary-512-nl158-np17-rf0-signed (query)          14_284.08       440.93    14_725.01       0.0619          NaN         4.36
IVF-Binary-512-nl158-np7-rf10-signed (query)          14_284.08       503.36    14_787.44       0.2416   555.252678         4.36
IVF-Binary-512-nl158-np7-rf20-signed (query)          14_284.08       580.00    14_864.08       0.3713   338.891854         4.36
IVF-Binary-512-nl158-np12-rf10-signed (query)         14_284.08       510.79    14_794.87       0.2290   590.717676         4.36
IVF-Binary-512-nl158-np12-rf20-signed (query)         14_284.08       592.33    14_876.41       0.3559   361.899684         4.36
IVF-Binary-512-nl158-np17-rf10-signed (query)         14_284.08       523.79    14_807.86       0.2186   624.054862         4.36
IVF-Binary-512-nl158-np17-rf20-signed (query)         14_284.08       616.89    14_900.96       0.3453   378.950941         4.36
IVF-Binary-512-nl158-signed (self)                    14_284.08     1_630.81    15_914.89       0.2296   587.029234         4.36
IVF-Binary-512-nl223-np11-rf0-signed (query)          12_077.84       433.82    12_511.65       0.0769          NaN         4.49
IVF-Binary-512-nl223-np14-rf0-signed (query)          12_077.84       438.54    12_516.38       0.0673          NaN         4.49
IVF-Binary-512-nl223-np21-rf0-signed (query)          12_077.84       448.82    12_526.66       0.0627          NaN         4.49
IVF-Binary-512-nl223-np11-rf10-signed (query)         12_077.84       515.41    12_593.25       0.2389   566.708737         4.49
IVF-Binary-512-nl223-np11-rf20-signed (query)         12_077.84       594.50    12_672.34       0.3689   348.116971         4.49
IVF-Binary-512-nl223-np14-rf10-signed (query)         12_077.84       520.12    12_597.96       0.2253   603.381054         4.49
IVF-Binary-512-nl223-np14-rf20-signed (query)         12_077.84       599.20    12_677.04       0.3536   367.691128         4.49
IVF-Binary-512-nl223-np21-rf10-signed (query)         12_077.84       532.84    12_610.68       0.2190   620.681331         4.49
IVF-Binary-512-nl223-np21-rf20-signed (query)         12_077.84       615.50    12_693.34       0.3476   375.031644         4.49
IVF-Binary-512-nl223-signed (self)                    12_077.84     1_657.25    13_735.08       0.2254   601.858320         4.49
IVF-Binary-512-nl316-np15-rf0-signed (query)          12_545.76       447.30    12_993.06       0.0682          NaN         4.67
IVF-Binary-512-nl316-np17-rf0-signed (query)          12_545.76       450.20    12_995.96       0.0656          NaN         4.67
IVF-Binary-512-nl316-np25-rf0-signed (query)          12_545.76       457.39    13_003.15       0.0647          NaN         4.67
IVF-Binary-512-nl316-np15-rf10-signed (query)         12_545.76       530.86    13_076.62       0.2320   585.225999         4.67
IVF-Binary-512-nl316-np15-rf20-signed (query)         12_545.76       605.14    13_150.90       0.3611   356.312394         4.67
IVF-Binary-512-nl316-np17-rf10-signed (query)         12_545.76       531.49    13_077.25       0.2286   594.217807         4.67
IVF-Binary-512-nl316-np17-rf20-signed (query)         12_545.76       613.52    13_159.27       0.3568   361.146342         4.67
IVF-Binary-512-nl316-np25-rf10-signed (query)         12_545.76       540.20    13_085.96       0.2249   603.241949         4.67
IVF-Binary-512-nl316-np25-rf20-signed (query)         12_545.76       624.93    13_170.69       0.3529   366.348422         4.67
IVF-Binary-512-nl316-signed (self)                    12_545.76     1_709.37    14_255.13       0.2280   593.371378         4.67
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 1024 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 1024D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        40.83    22_751.91    22_792.74       1.0000     0.000000       195.31
Exhaustive (self)                                         40.83    75_281.03    75_321.86       1.0000     0.000000       195.31
ExhaustiveBinary-256-random_no_rr (query)             12_009.02       588.59    12_597.61       0.0200          NaN         2.53
ExhaustiveBinary-256-random-rf10 (query)              12_009.02       780.68    12_789.70       0.1403  3161.218785         2.53
ExhaustiveBinary-256-random-rf20 (query)              12_009.02       950.54    12_959.56       0.2487  2005.173102         2.53
ExhaustiveBinary-256-random (self)                    12_009.02     2_438.64    14_447.66       0.1399  3156.210757         2.53
ExhaustiveBinary-256-pca_no_rr (query)                12_844.51       594.74    13_439.25       0.1640          NaN         2.53
ExhaustiveBinary-256-pca-rf10 (query)                 12_844.51       773.41    13_617.92       0.4530   796.709044         2.53
ExhaustiveBinary-256-pca-rf20 (query)                 12_844.51       951.79    13_796.30       0.5817   486.052218         2.53
ExhaustiveBinary-256-pca (self)                       12_844.51     2_553.59    15_398.10       0.4537   795.735831         2.53
ExhaustiveBinary-512-random_no_rr (query)             23_504.60     1_133.41    24_638.00       0.0498          NaN         5.05
ExhaustiveBinary-512-random-rf10 (query)              23_504.60     1_284.06    24_788.66       0.1854  2209.362983         5.05
ExhaustiveBinary-512-random-rf20 (query)              23_504.60     1_463.41    24_968.00       0.2990  1340.512945         5.05
ExhaustiveBinary-512-random (self)                    23_504.60     4_243.11    27_747.71       0.1862  2178.200805         5.05
ExhaustiveBinary-512-pca_no_rr (query)                24_621.05     1_141.42    25_762.47       0.1760          NaN         5.06
ExhaustiveBinary-512-pca-rf10 (query)                 24_621.05     1_316.11    25_937.16       0.4358   893.251520         5.06
ExhaustiveBinary-512-pca-rf20 (query)                 24_621.05     1_493.12    26_114.17       0.5492   553.977157         5.06
ExhaustiveBinary-512-pca (self)                       24_621.05     4_336.25    28_957.30       0.4361   891.981957         5.06
ExhaustiveBinary-1024-random_no_rr (query)            47_096.48     2_107.66    49_204.14       0.0806          NaN        10.10
ExhaustiveBinary-1024-random-rf10 (query)             47_096.48     2_286.08    49_382.57       0.2166  1814.307607        10.10
ExhaustiveBinary-1024-random-rf20 (query)             47_096.48     2_466.51    49_563.00       0.3365  1140.227104        10.10
ExhaustiveBinary-1024-random (self)                   47_096.48     7_559.29    54_655.77       0.2173  1809.604655        10.10
ExhaustiveBinary-1024-pca_no_rr (query)               48_428.21     2_132.13    50_560.34       0.1454          NaN        10.11
ExhaustiveBinary-1024-pca-rf10 (query)                48_428.21     2_317.22    50_745.43       0.3760 430233.613490        10.11
ExhaustiveBinary-1024-pca-rf20 (query)                48_428.21     2_517.83    50_946.04       0.4902   735.530419        10.11
ExhaustiveBinary-1024-pca (self)                      48_428.21     7_680.06    56_108.27       0.3763 475351.202512        10.11
ExhaustiveBinary-1024-signed_no_rr (query)            46_992.24     2_149.09    49_141.33       0.0806          NaN        10.10
ExhaustiveBinary-1024-signed-rf10 (query)             46_992.24     2_272.56    49_264.80       0.2166  1814.307607        10.10
ExhaustiveBinary-1024-signed-rf20 (query)             46_992.24     2_469.41    49_461.64       0.3365  1140.227104        10.10
ExhaustiveBinary-1024-signed (self)                   46_992.24     7_534.10    54_526.34       0.2173  1809.604655        10.10
IVF-Binary-256-nl158-np7-rf0-random (query)           19_353.11       488.00    19_841.12       0.0352          NaN         3.14
IVF-Binary-256-nl158-np12-rf0-random (query)          19_353.11       496.78    19_849.89       0.0271          NaN         3.14
IVF-Binary-256-nl158-np17-rf0-random (query)          19_353.11       498.45    19_851.57       0.0204          NaN         3.14
IVF-Binary-256-nl158-np7-rf10-random (query)          19_353.11       603.61    19_956.73       0.1913  2315.091828         3.14
IVF-Binary-256-nl158-np7-rf20-random (query)          19_353.11       733.64    20_086.76       0.3042  1443.107131         3.14
IVF-Binary-256-nl158-np12-rf10-random (query)         19_353.11       608.93    19_962.04       0.1691  2704.137484         3.14
IVF-Binary-256-nl158-np12-rf20-random (query)         19_353.11       741.35    20_094.46       0.2795  1699.727660         3.14
IVF-Binary-256-nl158-np17-rf10-random (query)         19_353.11       613.38    19_966.49       0.1452  3114.423144         3.14
IVF-Binary-256-nl158-np17-rf20-random (query)         19_353.11       745.02    20_098.13       0.2546  1962.516874         3.14
IVF-Binary-256-nl158-random (self)                    19_353.11     1_906.05    21_259.17       0.1681  2709.166990         3.14
IVF-Binary-256-nl223-np11-rf0-random (query)          14_069.98       513.61    14_583.59       0.0342          NaN         3.40
IVF-Binary-256-nl223-np14-rf0-random (query)          14_069.98       514.48    14_584.45       0.0269          NaN         3.40
IVF-Binary-256-nl223-np21-rf0-random (query)          14_069.98       522.07    14_592.05       0.0203          NaN         3.40
IVF-Binary-256-nl223-np11-rf10-random (query)         14_069.98       631.09    14_701.06       0.1748  2605.848286         3.40
IVF-Binary-256-nl223-np11-rf20-random (query)         14_069.98       752.88    14_822.85       0.2878  1645.332524         3.40
IVF-Binary-256-nl223-np14-rf10-random (query)         14_069.98       626.62    14_696.59       0.1589  2887.347437         3.40
IVF-Binary-256-nl223-np14-rf20-random (query)         14_069.98       762.82    14_832.79       0.2721  1807.780735         3.40
IVF-Binary-256-nl223-np21-rf10-random (query)         14_069.98       668.50    14_738.47       0.1414  3175.990961         3.40
IVF-Binary-256-nl223-np21-rf20-random (query)         14_069.98       766.56    14_836.53       0.2541  1986.051368         3.40
IVF-Binary-256-nl223-random (self)                    14_069.98     1_974.67    16_044.65       0.1587  2879.618086         3.40
IVF-Binary-256-nl316-np15-rf0-random (query)          14_657.42       547.91    15_205.32       0.0257          NaN         3.76
IVF-Binary-256-nl316-np17-rf0-random (query)          14_657.42       548.81    15_206.23       0.0232          NaN         3.76
IVF-Binary-256-nl316-np25-rf0-random (query)          14_657.42       552.55    15_209.96       0.0220          NaN         3.76
IVF-Binary-256-nl316-np15-rf10-random (query)         14_657.42       690.38    15_347.80       0.1623  2839.231047         3.76
IVF-Binary-256-nl316-np15-rf20-random (query)         14_657.42       781.12    15_438.53       0.2783  1753.089653         3.76
IVF-Binary-256-nl316-np17-rf10-random (query)         14_657.42       659.53    15_316.95       0.1523  3011.893565         3.76
IVF-Binary-256-nl316-np17-rf20-random (query)         14_657.42       786.01    15_443.43       0.2641  1897.869832         3.76
IVF-Binary-256-nl316-np25-rf10-random (query)         14_657.42       665.74    15_323.15       0.1473  3080.237141         3.76
IVF-Binary-256-nl316-np25-rf20-random (query)         14_657.42       797.29    15_454.71       0.2583  1945.645836         3.76
IVF-Binary-256-nl316-random (self)                    14_657.42     2_096.34    16_753.76       0.1524  3003.698715         3.76
IVF-Binary-256-nl158-np7-rf0-pca (query)              20_372.91       497.48    20_870.39       0.1705          NaN         3.15
IVF-Binary-256-nl158-np12-rf0-pca (query)             20_372.91       509.80    20_882.71       0.1702          NaN         3.15
IVF-Binary-256-nl158-np17-rf0-pca (query)             20_372.91       507.92    20_880.83       0.1698          NaN         3.15
IVF-Binary-256-nl158-np7-rf10-pca (query)             20_372.91       639.14    21_012.05       0.5173   623.470661         3.15
IVF-Binary-256-nl158-np7-rf20-pca (query)             20_372.91       766.18    21_139.09       0.6884   308.901555         3.15
IVF-Binary-256-nl158-np12-rf10-pca (query)            20_372.91       638.33    21_011.23       0.5146   629.676170         3.15
IVF-Binary-256-nl158-np12-rf20-pca (query)            20_372.91       775.71    21_148.62       0.6832   316.028145         3.15
IVF-Binary-256-nl158-np17-rf10-pca (query)            20_372.91       653.27    21_026.17       0.5110   638.232371         3.15
IVF-Binary-256-nl158-np17-rf20-pca (query)            20_372.91       795.56    21_168.46       0.6769   325.297761         3.15
IVF-Binary-256-nl158-pca (self)                       20_372.91     2_060.43    22_433.34       0.5154   628.212265         3.15
IVF-Binary-256-nl223-np11-rf0-pca (query)             15_060.23       520.75    15_580.98       0.1706          NaN         3.40
IVF-Binary-256-nl223-np14-rf0-pca (query)             15_060.23       523.30    15_583.52       0.1703          NaN         3.40
IVF-Binary-256-nl223-np21-rf0-pca (query)             15_060.23       530.89    15_591.11       0.1699          NaN         3.40
IVF-Binary-256-nl223-np11-rf10-pca (query)            15_060.23       661.49    15_721.72       0.5174   623.335441         3.40
IVF-Binary-256-nl223-np11-rf20-pca (query)            15_060.23       791.94    15_852.17       0.6889   308.175201         3.40
IVF-Binary-256-nl223-np14-rf10-pca (query)            15_060.23       660.24    15_720.46       0.5159   626.887411         3.40
IVF-Binary-256-nl223-np14-rf20-pca (query)            15_060.23       796.80    15_857.03       0.6862   311.803702         3.40
IVF-Binary-256-nl223-np21-rf10-pca (query)            15_060.23       681.18    15_741.40       0.5125   634.868159         3.40
IVF-Binary-256-nl223-np21-rf20-pca (query)            15_060.23       817.54    15_877.77       0.6800   320.865385         3.40
IVF-Binary-256-nl223-pca (self)                       15_060.23     2_129.52    17_189.75       0.5168   624.950419         3.40
IVF-Binary-256-nl316-np15-rf0-pca (query)             15_735.68       559.64    16_295.32       0.1705          NaN         3.77
IVF-Binary-256-nl316-np17-rf0-pca (query)             15_735.68       559.10    16_294.78       0.1703          NaN         3.77
IVF-Binary-256-nl316-np25-rf0-pca (query)             15_735.68       582.37    16_318.05       0.1702          NaN         3.77
IVF-Binary-256-nl316-np15-rf10-pca (query)            15_735.68       714.72    16_450.39       0.5176   622.368454         3.77
IVF-Binary-256-nl316-np15-rf20-pca (query)            15_735.68       837.10    16_572.78       0.6896   307.086923         3.77
IVF-Binary-256-nl316-np17-rf10-pca (query)            15_735.68       695.70    16_431.38       0.5166   624.603414         3.77
IVF-Binary-256-nl316-np17-rf20-pca (query)            15_735.68       828.88    16_564.55       0.6882   309.108270         3.77
IVF-Binary-256-nl316-np25-rf10-pca (query)            15_735.68       707.77    16_443.45       0.5144   630.079142         3.77
IVF-Binary-256-nl316-np25-rf20-pca (query)            15_735.68       844.81    16_580.49       0.6836   315.686189         3.77
IVF-Binary-256-nl316-pca (self)                       15_735.68     2_246.65    17_982.32       0.5178   622.605608         3.77
IVF-Binary-512-nl158-np7-rf0-random (query)           31_142.48       925.56    32_068.04       0.0663          NaN         5.67
IVF-Binary-512-nl158-np12-rf0-random (query)          31_142.48       931.71    32_074.19       0.0585          NaN         5.67
IVF-Binary-512-nl158-np17-rf0-random (query)          31_142.48       938.90    32_081.38       0.0509          NaN         5.67
IVF-Binary-512-nl158-np7-rf10-random (query)          31_142.48     1_032.79    32_175.27       0.2077  1924.042316         5.67
IVF-Binary-512-nl158-np7-rf20-random (query)          31_142.48     1_151.30    32_293.78       0.3200  1216.130795         5.67
IVF-Binary-512-nl158-np12-rf10-random (query)         31_142.48     1_038.78    32_181.26       0.1981  2028.825123         5.67
IVF-Binary-512-nl158-np12-rf20-random (query)         31_142.48     1_169.77    32_312.25       0.3100  1270.514737         5.67
IVF-Binary-512-nl158-np17-rf10-random (query)         31_142.48     1_052.90    32_195.38       0.1878  2181.292675         5.67
IVF-Binary-512-nl158-np17-rf20-random (query)         31_142.48     1_204.59    32_347.07       0.3011  1333.103447         5.67
IVF-Binary-512-nl158-random (self)                    31_142.48     3_389.84    34_532.32       0.1981  2013.922997         5.67
IVF-Binary-512-nl223-np11-rf0-random (query)          25_864.49       942.81    26_807.30       0.0618          NaN         5.92
IVF-Binary-512-nl223-np14-rf0-random (query)          25_864.49       947.56    26_812.04       0.0551          NaN         5.92
IVF-Binary-512-nl223-np21-rf0-random (query)          25_864.49       969.67    26_834.16       0.0503          NaN         5.92
IVF-Binary-512-nl223-np11-rf10-random (query)         25_864.49     1_057.70    26_922.19       0.2005  2039.644311         5.92
IVF-Binary-512-nl223-np11-rf20-random (query)         25_864.49     1_181.93    27_046.42       0.3140  1261.363599         5.92
IVF-Binary-512-nl223-np14-rf10-random (query)         25_864.49     1_062.21    26_926.69       0.1949  2094.587153         5.92
IVF-Binary-512-nl223-np14-rf20-random (query)         25_864.49     1_191.28    27_055.77       0.3077  1287.620212         5.92
IVF-Binary-512-nl223-np21-rf10-random (query)         25_864.49     1_155.26    27_019.75       0.1866  2208.394321         5.92
IVF-Binary-512-nl223-np21-rf20-random (query)         25_864.49     1_216.85    27_081.33       0.2996  1343.288342         5.92
IVF-Binary-512-nl223-random (self)                    25_864.49     3_443.95    29_308.44       0.1959  2069.164724         5.92
IVF-Binary-512-nl316-np15-rf0-random (query)          26_490.42       984.17    27_474.59       0.0556          NaN         6.29
IVF-Binary-512-nl316-np17-rf0-random (query)          26_490.42       985.20    27_475.62       0.0529          NaN         6.29
IVF-Binary-512-nl316-np25-rf0-random (query)          26_490.42       997.77    27_488.19       0.0515          NaN         6.29
IVF-Binary-512-nl316-np15-rf10-random (query)         26_490.42     1_092.84    27_583.26       0.1962  2080.979487         6.29
IVF-Binary-512-nl316-np15-rf20-random (query)         26_490.42     1_296.97    27_787.39       0.3124  1272.831907         6.29
IVF-Binary-512-nl316-np17-rf10-random (query)         26_490.42     1_100.25    27_590.66       0.1919  2134.733498         6.29
IVF-Binary-512-nl316-np17-rf20-random (query)         26_490.42     1_227.49    27_717.91       0.3082  1293.741148         6.29
IVF-Binary-512-nl316-np25-rf10-random (query)         26_490.42     1_102.97    27_593.39       0.1890  2172.917293         6.29
IVF-Binary-512-nl316-np25-rf20-random (query)         26_490.42     1_235.94    27_726.35       0.3044  1321.251000         6.29
IVF-Binary-512-nl316-random (self)                    26_490.42     3_558.74    30_049.16       0.1930  2108.547671         6.29
IVF-Binary-512-nl158-np7-rf0-pca (query)              32_077.76       950.85    33_028.61       0.2349          NaN         5.67
IVF-Binary-512-nl158-np12-rf0-pca (query)             32_077.76       957.06    33_034.82       0.2313          NaN         5.67
IVF-Binary-512-nl158-np17-rf0-pca (query)             32_077.76       958.28    33_036.04       0.2275          NaN         5.67
IVF-Binary-512-nl158-np7-rf10-pca (query)             32_077.76     1_062.31    33_140.07       0.6583   343.753610         5.67
IVF-Binary-512-nl158-np7-rf20-pca (query)             32_077.76     1_192.69    33_270.45       0.8130   147.587200         5.67
IVF-Binary-512-nl158-np12-rf10-pca (query)            32_077.76     1_072.18    33_149.94       0.6408   372.107321         5.67
IVF-Binary-512-nl158-np12-rf20-pca (query)            32_077.76     1_212.10    33_289.86       0.7937   166.655726         5.67
IVF-Binary-512-nl158-np17-rf10-pca (query)            32_077.76     1_092.85    33_170.61       0.6231   402.488792         5.67
IVF-Binary-512-nl158-np17-rf20-pca (query)            32_077.76     1_236.05    33_313.81       0.7712   191.346405         5.67
IVF-Binary-512-nl158-pca (self)                       32_077.76     3_521.78    35_599.54       0.6422   370.162293         5.67
IVF-Binary-512-nl223-np11-rf0-pca (query)             26_921.28       960.66    27_881.94       0.2345          NaN         5.93
IVF-Binary-512-nl223-np14-rf0-pca (query)             26_921.28       965.65    27_886.93       0.2329          NaN         5.93
IVF-Binary-512-nl223-np21-rf0-pca (query)             26_921.28       976.38    27_897.66       0.2290          NaN         5.93
IVF-Binary-512-nl223-np11-rf10-pca (query)            26_921.28     1_089.62    28_010.90       0.6586   343.255071         5.93
IVF-Binary-512-nl223-np11-rf20-pca (query)            26_921.28     1_221.18    28_142.46       0.8142   145.916178         5.93
IVF-Binary-512-nl223-np14-rf10-pca (query)            26_921.28     1_092.30    28_013.58       0.6505   356.130964         5.93
IVF-Binary-512-nl223-np14-rf20-pca (query)            26_921.28     1_235.26    28_156.54       0.8044   155.463974         5.93
IVF-Binary-512-nl223-np21-rf10-pca (query)            26_921.28     1_116.24    28_037.52       0.6314   388.296245         5.93
IVF-Binary-512-nl223-np21-rf20-pca (query)            26_921.28     1_252.22    28_173.50       0.7818   180.016810         5.93
IVF-Binary-512-nl223-pca (self)                       26_921.28     3_578.26    30_499.54       0.6508   355.941403         5.93
IVF-Binary-512-nl316-np15-rf0-pca (query)             27_330.90       999.35    28_330.26       0.2351          NaN         6.29
IVF-Binary-512-nl316-np17-rf0-pca (query)             27_330.90     1_015.75    28_346.65       0.2343          NaN         6.29
IVF-Binary-512-nl316-np25-rf0-pca (query)             27_330.90     1_018.94    28_349.84       0.2314          NaN         6.29
IVF-Binary-512-nl316-np15-rf10-pca (query)            27_330.90     1_133.93    28_464.83       0.6608   339.636141         6.29
IVF-Binary-512-nl316-np15-rf20-pca (query)            27_330.90     1_283.69    28_614.59       0.8171   142.604945         6.29
IVF-Binary-512-nl316-np17-rf10-pca (query)            27_330.90     1_125.14    28_456.04       0.6567   346.057150         6.29
IVF-Binary-512-nl316-np17-rf20-pca (query)            27_330.90     1_280.33    28_611.23       0.8123   147.452660         6.29
IVF-Binary-512-nl316-np25-rf10-pca (query)            27_330.90     1_140.08    28_470.99       0.6415   370.884108         6.29
IVF-Binary-512-nl316-np25-rf20-pca (query)            27_330.90     1_295.30    28_626.21       0.7938   166.656581         6.29
IVF-Binary-512-nl316-pca (self)                       27_330.90     3_690.53    31_021.43       0.6571   345.555930         6.29
IVF-Binary-1024-nl158-np7-rf0-random (query)          54_737.83     1_789.66    56_527.49       0.0862          NaN        10.72
IVF-Binary-1024-nl158-np12-rf0-random (query)         54_737.83     1_807.13    56_544.96       0.0842          NaN        10.72
IVF-Binary-1024-nl158-np17-rf0-random (query)         54_737.83     1_819.29    56_557.12       0.0812          NaN        10.72
IVF-Binary-1024-nl158-np7-rf10-random (query)         54_737.83     2_023.23    56_761.07       0.2259  1748.056943        10.72
IVF-Binary-1024-nl158-np7-rf20-random (query)         54_737.83     2_018.94    56_756.78       0.3483  1097.112458        10.72
IVF-Binary-1024-nl158-np12-rf10-random (query)        54_737.83     1_905.91    56_643.74       0.2210  1783.173302        10.72
IVF-Binary-1024-nl158-np12-rf20-random (query)        54_737.83     2_061.69    56_799.52       0.3414  1121.336215        10.72
IVF-Binary-1024-nl158-np17-rf10-random (query)        54_737.83     1_928.09    56_665.92       0.2171  1812.761678        10.72
IVF-Binary-1024-nl158-np17-rf20-random (query)        54_737.83     2_062.67    56_800.50       0.3369  1138.708496        10.72
IVF-Binary-1024-nl158-random (self)                   54_737.83     6_342.23    61_080.06       0.2219  1779.835846        10.72
IVF-Binary-1024-nl223-np11-rf0-random (query)         49_246.96     1_811.70    51_058.66       0.0841          NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-random (query)         49_246.96     1_821.21    51_068.17       0.0827          NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-random (query)         49_246.96     1_839.84    51_086.80       0.0812          NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-random (query)        49_246.96     1_935.79    51_182.75       0.2236  1769.689171        10.98
IVF-Binary-1024-nl223-np11-rf20-random (query)        49_246.96     2_075.55    51_322.51       0.3453  1109.377052        10.98
IVF-Binary-1024-nl223-np14-rf10-random (query)        49_246.96     1_947.82    51_194.78       0.2205  1791.340136        10.98
IVF-Binary-1024-nl223-np14-rf20-random (query)        49_246.96     2_062.92    51_309.87       0.3414  1123.811156        10.98
IVF-Binary-1024-nl223-np21-rf10-random (query)        49_246.96     1_952.17    51_199.12       0.2171  1815.059428        10.98
IVF-Binary-1024-nl223-np21-rf20-random (query)        49_246.96     2_096.84    51_343.79       0.3372  1139.307983        10.98
IVF-Binary-1024-nl223-random (self)                   49_246.96     6_360.66    55_607.61       0.2214  1784.941531        10.98
IVF-Binary-1024-nl316-np15-rf0-random (query)         49_840.17     1_854.71    51_694.87       0.0827          NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-random (query)         49_840.17     1_860.71    51_700.88       0.0818          NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-random (query)         49_840.17     1_866.31    51_706.48       0.0813          NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-random (query)        49_840.17     1_949.22    51_789.39       0.2228  1775.377330        11.34
IVF-Binary-1024-nl316-np15-rf20-random (query)        49_840.17     2_082.48    51_922.64       0.3447  1111.475917        11.34
IVF-Binary-1024-nl316-np17-rf10-random (query)        49_840.17     1_992.36    51_832.53       0.2208  1790.434505        11.34
IVF-Binary-1024-nl316-np17-rf20-random (query)        49_840.17     2_111.61    51_951.77       0.3420  1121.450614        11.34
IVF-Binary-1024-nl316-np25-rf10-random (query)        49_840.17     1_968.36    51_808.53       0.2193  1800.845154        11.34
IVF-Binary-1024-nl316-np25-rf20-random (query)        49_840.17     2_118.59    51_958.76       0.3398  1129.697361        11.34
IVF-Binary-1024-nl316-random (self)                   49_840.17     6_437.12    56_277.28       0.2209  1786.577159        11.34
IVF-Binary-1024-nl158-np7-rf0-pca (query)             55_585.74     1_823.64    57_409.37       0.2857          NaN        10.73
IVF-Binary-1024-nl158-np12-rf0-pca (query)            55_585.74     1_844.69    57_430.43       0.2683          NaN        10.73
IVF-Binary-1024-nl158-np17-rf0-pca (query)            55_585.74     1_852.87    57_438.60       0.2534          NaN        10.73
IVF-Binary-1024-nl158-np7-rf10-pca (query)            55_585.74     1_922.88    57_508.62       0.7448   216.672541        10.73
IVF-Binary-1024-nl158-np7-rf20-pca (query)            55_585.74     2_053.66    57_639.39       0.8755    85.254261        10.73
IVF-Binary-1024-nl158-np12-rf10-pca (query)           55_585.74     1_938.24    57_523.98       0.7064   266.296081        10.73
IVF-Binary-1024-nl158-np12-rf20-pca (query)           55_585.74     2_088.77    57_674.50       0.8422   112.165315        10.73
IVF-Binary-1024-nl158-np17-rf10-pca (query)           55_585.74     1_967.13    57_552.87       0.6684   322.407633        10.73
IVF-Binary-1024-nl158-np17-rf20-pca (query)           55_585.74     2_111.39    57_697.13       0.8066   146.674035        10.73
IVF-Binary-1024-nl158-pca (self)                      55_585.74     6_423.17    62_008.91       0.7068   265.529244        10.73
IVF-Binary-1024-nl223-np11-rf0-pca (query)            50_515.95     1_844.34    52_360.29       0.2842          NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-pca (query)            50_515.95     1_862.79    52_378.73       0.2764          NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-pca (query)            50_515.95     1_903.49    52_419.44       0.2600          NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-pca (query)           50_515.95     1_992.46    52_508.41       0.7446   217.334583        10.98
IVF-Binary-1024-nl223-np11-rf20-pca (query)           50_515.95     2_128.79    52_644.74       0.8770    82.795061        10.98
IVF-Binary-1024-nl223-np14-rf10-pca (query)           50_515.95     1_956.44    52_472.38       0.7255   240.704196        10.98
IVF-Binary-1024-nl223-np14-rf20-pca (query)           50_515.95     2_123.49    52_639.44       0.8601    96.380300        10.98
IVF-Binary-1024-nl223-np21-rf10-pca (query)           50_515.95     1_979.16    52_495.11       0.6844   298.381251        10.98
IVF-Binary-1024-nl223-np21-rf20-pca (query)           50_515.95     2_143.37    52_659.32       0.8217   131.638639        10.98
IVF-Binary-1024-nl223-pca (self)                      50_515.95     6_465.68    56_981.63       0.7259   240.296884        10.98
IVF-Binary-1024-nl316-np15-rf0-pca (query)            50_929.54     1_908.56    52_838.10       0.2857          NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-pca (query)            50_929.54     1_906.08    52_835.61       0.2818          NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-pca (query)            50_929.54     1_879.81    52_809.34       0.2684          NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-pca (query)           50_929.54     2_006.65    52_936.18       0.7493   210.561652        11.34
IVF-Binary-1024-nl316-np15-rf20-pca (query)           50_929.54     2_122.44    53_051.98       0.8819    78.222503        11.34
IVF-Binary-1024-nl316-np17-rf10-pca (query)           50_929.54     1_984.82    52_914.36       0.7401   221.750546        11.34
IVF-Binary-1024-nl316-np17-rf20-pca (query)           50_929.54     2_129.29    53_058.83       0.8736    84.839687        11.34
IVF-Binary-1024-nl316-np25-rf10-pca (query)           50_929.54     2_003.57    52_933.11       0.7058   266.885649        11.34
IVF-Binary-1024-nl316-np25-rf20-pca (query)           50_929.54     2_153.07    53_082.61       0.8420   112.146077        11.34
IVF-Binary-1024-nl316-pca (self)                      50_929.54     6_562.84    57_492.37       0.7408   220.784940        11.34
IVF-Binary-1024-nl158-np7-rf0-signed (query)          54_601.15     1_783.55    56_384.70       0.0862          NaN        10.72
IVF-Binary-1024-nl158-np12-rf0-signed (query)         54_601.15     1_830.52    56_431.67       0.0842          NaN        10.72
IVF-Binary-1024-nl158-np17-rf0-signed (query)         54_601.15     1_822.86    56_424.01       0.0812          NaN        10.72
IVF-Binary-1024-nl158-np7-rf10-signed (query)         54_601.15     1_883.59    56_484.74       0.2259  1748.056943        10.72
IVF-Binary-1024-nl158-np7-rf20-signed (query)         54_601.15     2_007.55    56_608.71       0.3483  1097.112458        10.72
IVF-Binary-1024-nl158-np12-rf10-signed (query)        54_601.15     1_894.41    56_495.57       0.2210  1783.173302        10.72
IVF-Binary-1024-nl158-np12-rf20-signed (query)        54_601.15     2_038.53    56_639.69       0.3414  1121.336215        10.72
IVF-Binary-1024-nl158-np17-rf10-signed (query)        54_601.15     1_925.85    56_527.00       0.2171  1812.761678        10.72
IVF-Binary-1024-nl158-np17-rf20-signed (query)        54_601.15     2_072.14    56_673.30       0.3369  1138.708496        10.72
IVF-Binary-1024-nl158-signed (self)                   54_601.15     6_316.19    60_917.34       0.2219  1779.835846        10.72
IVF-Binary-1024-nl223-np11-rf0-signed (query)         49_529.59     1_809.98    51_339.57       0.0841          NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-signed (query)         49_529.59     1_832.12    51_361.71       0.0827          NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-signed (query)         49_529.59     1_843.53    51_373.12       0.0812          NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-signed (query)        49_529.59     1_919.86    51_449.45       0.2236  1769.689171        10.98
IVF-Binary-1024-nl223-np11-rf20-signed (query)        49_529.59     2_033.59    51_563.18       0.3453  1109.377052        10.98
IVF-Binary-1024-nl223-np14-rf10-signed (query)        49_529.59     1_916.30    51_445.90       0.2205  1791.340136        10.98
IVF-Binary-1024-nl223-np14-rf20-signed (query)        49_529.59     2_047.80    51_577.39       0.3414  1123.811156        10.98
IVF-Binary-1024-nl223-np21-rf10-signed (query)        49_529.59     1_934.50    51_464.09       0.2171  1815.059428        10.98
IVF-Binary-1024-nl223-np21-rf20-signed (query)        49_529.59     2_075.06    51_604.66       0.3372  1139.307983        10.98
IVF-Binary-1024-nl223-signed (self)                   49_529.59     6_289.12    55_818.72       0.2214  1784.941531        10.98
IVF-Binary-1024-nl316-np15-rf0-signed (query)         49_972.81     1_845.48    51_818.29       0.0827          NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-signed (query)         49_972.81     1_848.49    51_821.30       0.0818          NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-signed (query)         49_972.81     1_852.33    51_825.13       0.0813          NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-signed (query)        49_972.81     1_947.05    51_919.86       0.2228  1775.377330        11.34
IVF-Binary-1024-nl316-np15-rf20-signed (query)        49_972.81     2_069.97    52_042.77       0.3447  1111.475917        11.34
IVF-Binary-1024-nl316-np17-rf10-signed (query)        49_972.81     1_951.61    51_924.42       0.2208  1790.434505        11.34
IVF-Binary-1024-nl316-np17-rf20-signed (query)        49_972.81     2_076.93    52_049.74       0.3420  1121.450614        11.34
IVF-Binary-1024-nl316-np25-rf10-signed (query)        49_972.81     1_964.87    51_937.67       0.2193  1800.845154        11.34
IVF-Binary-1024-nl316-np25-rf20-signed (query)        49_972.81     2_110.51    52_083.32       0.3398  1129.697361        11.34
IVF-Binary-1024-nl316-signed (self)                   49_972.81     6_457.01    56_429.81       0.2209  1786.577159        11.34
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

#### Quantisation (stress) data

<details>
<summary><b>Quantisation stress data - 256 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 256D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         6.79     4_222.57     4_229.36       1.0000     0.000000        48.83
Exhaustive (self)                                          6.79    14_506.07    14_512.87       1.0000     0.000000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_577.58       253.26     2_830.84       0.0387          NaN         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_577.58       366.62     2_944.19       0.2254     0.011114         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_577.58       482.06     3_059.64       0.3590     0.006290         1.78
ExhaustiveBinary-256-random (self)                     2_577.58     1_239.10     3_816.67       0.5886     0.060711         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_837.81       261.28     3_099.09       0.0284          NaN         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_837.81       368.92     3_206.72       0.1316     0.028474         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_837.81       477.43     3_315.23       0.1964     0.012549         1.78
ExhaustiveBinary-256-pca (self)                        2_837.81     1_222.24     4_060.05       0.2462     0.006202         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_101.20       459.75     5_560.95       0.0605          NaN         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_101.20       568.27     5_669.47       0.3044     0.008791         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_101.20       678.84     5_780.04       0.4560     0.005019         3.55
ExhaustiveBinary-512-random (self)                     5_101.20     2_003.88     7_105.08       0.6749     0.090442         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_528.80       470.77     5_999.58       0.0806          NaN         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_528.80       592.02     6_120.82       0.3799     0.005279         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_528.80       697.85     6_226.65       0.5477     0.002506         3.55
ExhaustiveBinary-512-pca (self)                        5_528.80     1_932.71     7_461.51       0.6562     0.000615         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_114.59       789.31    10_903.90       0.0996          NaN         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_114.59       900.09    11_014.69       0.4213     0.006014         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_114.59     1_029.41    11_144.00       0.5847     0.003378         7.10
ExhaustiveBinary-1024-random (self)                   10_114.59     2_993.65    13_108.24       0.7571     0.111193         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_630.29       828.96    11_459.25       0.1211          NaN         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_630.29       933.03    11_563.32       0.5030     0.005818         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_630.29     1_044.42    11_674.71       0.6838     0.002670         7.10
ExhaustiveBinary-1024-pca (self)                      10_630.29     3_042.89    13_673.18       0.8235     0.000272         7.10
ExhaustiveBinary-256-signed_no_rr (query)              2_580.95       254.12     2_835.07       0.0387          NaN         1.78
ExhaustiveBinary-256-signed-rf10 (query)               2_580.95       377.92     2_958.87       0.2254     0.011114         1.78
ExhaustiveBinary-256-signed-rf20 (query)               2_580.95       472.32     3_053.27       0.3590     0.006290         1.78
ExhaustiveBinary-256-signed (self)                     2_580.95     1_204.82     3_785.77       0.5886     0.060711         1.78
IVF-Binary-256-nl158-np7-rf0-random (query)            3_822.56       124.16     3_946.72       0.0509          NaN         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           3_822.56       135.20     3_957.75       0.0465          NaN         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           3_822.56       148.67     3_971.22       0.0446          NaN         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           3_822.56       197.59     4_020.15       0.3052     0.005764         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           3_822.56       243.11     4_065.67       0.4816     0.003002         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          3_822.56       209.32     4_031.88       0.2841     0.006651         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          3_822.56       271.68     4_094.24       0.4542     0.003510         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          3_822.56       230.77     4_053.33       0.2744     0.007096         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          3_822.56       308.74     4_131.30       0.4401     0.003772         1.93
IVF-Binary-256-nl158-random (self)                     3_822.56       647.77     4_470.33       0.7484     0.000430         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           2_682.16       120.33     2_802.49       0.0512          NaN         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           2_682.16       123.29     2_805.45       0.0481          NaN         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           2_682.16       131.12     2_813.28       0.0452          NaN         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          2_682.16       212.76     2_894.92       0.3029     0.005893         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          2_682.16       258.55     2_940.71       0.4808     0.003183         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          2_682.16       199.65     2_881.82       0.2901     0.006353         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          2_682.16       248.88     2_931.04       0.4641     0.003455         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          2_682.16       207.11     2_889.27       0.2785     0.006784         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          2_682.16       268.53     2_950.69       0.4478     0.003655         2.00
IVF-Binary-256-nl223-random (self)                     2_682.16       567.85     3_250.02       0.7426     0.000488         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           2_718.32       135.85     2_854.17       0.0506          NaN         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           2_718.32       137.96     2_856.28       0.0491          NaN         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           2_718.32       137.35     2_855.67       0.0461          NaN         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          2_718.32       196.67     2_914.99       0.3030     0.005872         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          2_718.32       251.29     2_969.61       0.4824     0.003181         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          2_718.32       199.25     2_917.57       0.2971     0.006058         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          2_718.32       264.97     2_983.29       0.4742     0.003294         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          2_718.32       202.57     2_920.89       0.2832     0.006557         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          2_718.32       278.65     2_996.97       0.4549     0.003561         2.09
IVF-Binary-256-nl316-random (self)                     2_718.32       579.65     3_297.97       0.7435     0.000482         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               4_113.37       137.63     4_251.00       0.0612          NaN         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              4_113.37       140.31     4_253.68       0.0515          NaN         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              4_113.37       152.42     4_265.79       0.0461          NaN         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              4_113.37       199.20     4_312.57       0.3050     0.005489         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              4_113.37       262.93     4_376.30       0.4482     0.003105         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             4_113.37       218.93     4_332.30       0.2538     0.007314         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             4_113.37       297.22     4_410.59       0.3768     0.004210         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             4_113.37       235.27     4_348.64       0.2255     0.008867         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             4_113.37       314.71     4_428.08       0.3355     0.005090         1.93
IVF-Binary-256-nl158-pca (self)                        4_113.37       710.55     4_823.92       0.3531     0.001863         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              2_927.50       127.35     3_054.85       0.0657          NaN         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              2_927.50       130.79     3_058.29       0.0603          NaN         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              2_927.50       134.08     3_061.58       0.0540          NaN         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             2_927.50       195.67     3_123.17       0.3321     0.004831         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             2_927.50       257.78     3_185.28       0.4963     0.002617         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             2_927.50       209.90     3_137.40       0.3066     0.005413         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             2_927.50       271.41     3_198.91       0.4589     0.003031         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             2_927.50       211.09     3_138.59       0.2715     0.006379         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             2_927.50       282.08     3_209.58       0.4064     0.003686         2.00
IVF-Binary-256-nl223-pca (self)                        2_927.50       621.98     3_549.48       0.3965     0.001514         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              2_954.71       132.04     3_086.75       0.0667          NaN         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              2_954.71       133.77     3_088.48       0.0641          NaN         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              2_954.71       140.51     3_095.22       0.0571          NaN         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             2_954.71       202.66     3_157.37       0.3384     0.004722         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             2_954.71       304.30     3_259.02       0.5061     0.002535         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             2_954.71       203.13     3_157.84       0.3251     0.004995         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             2_954.71       295.52     3_250.23       0.4865     0.002731         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             2_954.71       216.16     3_170.87       0.2885     0.005877         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             2_954.71       297.91     3_252.62       0.4321     0.003349         2.09
IVF-Binary-256-nl316-pca (self)                        2_954.71       631.01     3_585.72       0.4100     0.001426         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_385.00       221.57     6_606.57       0.0736          NaN         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_385.00       242.31     6_627.31       0.0692          NaN         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_385.00       270.71     6_655.71       0.0671          NaN         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_385.00       294.46     6_679.46       0.3873     0.004295         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_385.00       347.67     6_732.67       0.5762     0.002149         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_385.00       329.68     6_714.68       0.3697     0.004805         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_385.00       383.66     6_768.66       0.5551     0.002436         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_385.00       348.84     6_733.84       0.3601     0.005108         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_385.00       427.61     6_812.61       0.5426     0.002629         3.71
IVF-Binary-512-nl158-random (self)                     6_385.00     1_018.00     7_402.99       0.8442     0.000221         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_237.34       223.48     5_460.82       0.0735          NaN         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_237.34       246.42     5_483.76       0.0705          NaN         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_237.34       258.00     5_495.34       0.0678          NaN         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_237.34       320.90     5_558.24       0.3851     0.004489         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_237.34       359.10     5_596.44       0.5741     0.002312         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_237.34       298.18     5_535.52       0.3734     0.004801         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_237.34       359.10     5_596.44       0.5598     0.002493         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_237.34       317.43     5_554.77       0.3645     0.004992         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_237.34       382.47     5_619.81       0.5480     0.002598         3.77
IVF-Binary-512-nl223-random (self)                     5_237.34       918.00     6_155.34       0.8385     0.000264         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           5_175.04       220.29     5_395.33       0.0731          NaN         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           5_175.04       221.74     5_396.77       0.0717          NaN         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           5_175.04       230.47     5_405.51       0.0685          NaN         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          5_175.04       287.37     5_462.40       0.3853     0.004492         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          5_175.04       337.12     5_512.16       0.5760     0.002303         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          5_175.04       287.28     5_462.32       0.3798     0.004621         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          5_175.04       342.27     5_517.31       0.5692     0.002380         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          5_175.04       300.34     5_475.38       0.3680     0.004901         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          5_175.04       355.99     5_531.03       0.5534     0.002543         3.86
IVF-Binary-512-nl316-random (self)                     5_175.04       878.10     6_053.13       0.8398     0.000259         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               6_581.70       224.77     6_806.47       0.0969          NaN         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              6_581.70       244.58     6_826.27       0.0918          NaN         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              6_581.70       263.82     6_845.52       0.0894          NaN         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              6_581.70       297.13     6_878.83       0.4469     0.003124         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              6_581.70       350.26     6_931.96       0.6308     0.001535         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             6_581.70       338.38     6_920.08       0.4262     0.003500         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             6_581.70       380.97     6_962.67       0.6079     0.001758         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             6_581.70       349.34     6_931.04       0.4173     0.003705         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             6_581.70       422.17     7_003.87       0.5961     0.001897         3.71
IVF-Binary-512-nl158-pca (self)                        6_581.70     1_006.24     7_587.94       0.6714     0.000512         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              5_448.06       222.10     5_670.15       0.0964          NaN         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              5_448.06       224.08     5_672.14       0.0937          NaN         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              5_448.06       239.45     5_687.51       0.0910          NaN         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             5_448.06       288.65     5_736.71       0.4490     0.003228         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             5_448.06       340.12     5_788.18       0.6410     0.001542         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             5_448.06       292.94     5_741.00       0.4379     0.003415         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             5_448.06       350.48     5_798.53       0.6260     0.001676         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             5_448.06       307.01     5_755.06       0.4265     0.003567         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             5_448.06       372.04     5_820.09       0.6109     0.001784         3.77
IVF-Binary-512-nl223-pca (self)                        5_448.06       910.66     6_358.72       0.6652     0.000599         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              5_479.49       225.77     5_705.26       0.0968          NaN         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              5_479.49       227.34     5_706.84       0.0954          NaN         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              5_479.49       236.70     5_716.19       0.0922          NaN         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             5_479.49       294.90     5_774.39       0.4524     0.003206         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             5_479.49       348.44     5_827.93       0.6451     0.001527         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             5_479.49       294.98     5_774.47       0.4465     0.003284         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             5_479.49       351.21     5_830.70       0.6375     0.001584         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             5_479.49       305.21     5_784.70       0.4325     0.003480         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             5_479.49       367.65     5_847.14       0.6188     0.001726         3.86
IVF-Binary-512-nl316-pca (self)                        5_479.49       917.32     6_396.81       0.6661     0.000593         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_172.76       405.44    11_578.19       0.1145          NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_172.76       434.99    11_607.75       0.1107          NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_172.76       465.93    11_638.68       0.1084          NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_172.76       481.13    11_653.89       0.5052     0.002837         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_172.76       530.83    11_703.59       0.6944     0.001316         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_172.76       515.22    11_687.98       0.4910     0.003136         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_172.76       583.00    11_755.76       0.6794     0.001486         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_172.76       557.34    11_730.09       0.4821     0.003335         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_172.76       630.61    11_803.37       0.6691     0.001614         7.26
IVF-Binary-1024-nl158-random (self)                   11_172.76     1_657.66    12_830.42       0.9241     0.000089         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_039.20       397.95    10_437.15       0.1144          NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_039.20       404.59    10_443.79       0.1111          NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_039.20       419.32    10_458.52       0.1086          NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_039.20       466.53    10_505.73       0.5009     0.003071         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_039.20       516.79    10_555.99       0.6893     0.001456         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_039.20       473.74    10_512.94       0.4913     0.003253         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_039.20       530.93    10_570.13       0.6783     0.001572         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_039.20       494.80    10_534.00       0.4848     0.003329         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_039.20       558.19    10_597.39       0.6715     0.001615         7.32
IVF-Binary-1024-nl223-random (self)                   10_039.20     1_510.68    11_549.88       0.9201     0.000112         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         10_069.33       402.83    10_472.16       0.1143          NaN         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)         10_069.33       407.42    10_476.74       0.1127          NaN         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)         10_069.33       427.60    10_496.92       0.1094          NaN         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)        10_069.33       477.30    10_546.63       0.5017     0.003068         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)        10_069.33       525.33    10_594.66       0.6914     0.001450         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)        10_069.33       476.77    10_546.10       0.4967     0.003150         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)        10_069.33       536.22    10_605.55       0.6857     0.001501         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)        10_069.33       490.02    10_559.35       0.4874     0.003295         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)        10_069.33       551.47    10_620.80       0.6748     0.001590         7.41
IVF-Binary-1024-nl316-random (self)                   10_069.33     1_538.35    11_607.68       0.9211     0.000109         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)             11_678.84       423.01    12_101.84       0.1358          NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            11_678.84       451.24    12_130.07       0.1313          NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            11_678.84       483.81    12_162.64       0.1292          NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            11_678.84       492.94    12_171.77       0.5564     0.002128         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            11_678.84       547.54    12_226.38       0.7414     0.000910         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           11_678.84       531.15    12_209.99       0.5426     0.002343         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           11_678.84       597.17    12_276.01       0.7291     0.001003         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           11_678.84       572.08    12_250.92       0.5360     0.002489         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           11_678.84       647.23    12_326.07       0.7218     0.001083         7.26
IVF-Binary-1024-nl158-pca (self)                      11_678.84     1_719.32    13_398.16       0.8299     0.000217         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            10_566.21       411.52    10_977.73       0.1344          NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            10_566.21       425.22    10_991.42       0.1323          NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            10_566.21       433.30    10_999.50       0.1304          NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           10_566.21       486.79    11_053.00       0.5541     0.002242         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           10_566.21       537.81    11_104.02       0.7434     0.000957         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           10_566.21       491.23    11_057.44       0.5473     0.002340         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           10_566.21       548.35    11_114.55       0.7354     0.001010         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           10_566.21       510.93    11_077.14       0.5408     0.002409         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           10_566.21       581.17    11_147.37       0.7276     0.001051         7.32
IVF-Binary-1024-nl223-pca (self)                      10_566.21     1_561.88    12_128.09       0.8252     0.000265         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            10_582.46       434.61    11_017.07       0.1346          NaN         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            10_582.46       471.51    11_053.97       0.1335          NaN         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            10_582.46       434.40    11_016.87       0.1312          NaN         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           10_582.46       489.45    11_071.92       0.5554     0.002240         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           10_582.46       584.64    11_167.11       0.7452     0.000954         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           10_582.46       490.67    11_073.13       0.5522     0.002271         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           10_582.46       547.56    11_130.02       0.7413     0.000975         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           10_582.46       506.24    11_088.71       0.5438     0.002369         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           10_582.46       568.57    11_151.04       0.7316     0.001027         7.42
IVF-Binary-1024-nl316-pca (self)                      10_582.46     1_564.47    12_146.93       0.8253     0.000265         7.42
IVF-Binary-256-nl158-np7-rf0-signed (query)            3_710.43       122.33     3_832.76       0.0509          NaN         1.93
IVF-Binary-256-nl158-np12-rf0-signed (query)           3_710.43       131.26     3_841.69       0.0465          NaN         1.93
IVF-Binary-256-nl158-np17-rf0-signed (query)           3_710.43       144.14     3_854.57       0.0446          NaN         1.93
IVF-Binary-256-nl158-np7-rf10-signed (query)           3_710.43       190.23     3_900.66       0.3052     0.005764         1.93
IVF-Binary-256-nl158-np7-rf20-signed (query)           3_710.43       235.10     3_945.53       0.4816     0.003002         1.93
IVF-Binary-256-nl158-np12-rf10-signed (query)          3_710.43       205.20     3_915.63       0.2841     0.006651         1.93
IVF-Binary-256-nl158-np12-rf20-signed (query)          3_710.43       263.15     3_973.58       0.4542     0.003510         1.93
IVF-Binary-256-nl158-np17-rf10-signed (query)          3_710.43       225.17     3_935.60       0.2744     0.007096         1.93
IVF-Binary-256-nl158-np17-rf20-signed (query)          3_710.43       291.80     4_002.23       0.4401     0.003772         1.93
IVF-Binary-256-nl158-signed (self)                     3_710.43       618.51     4_328.94       0.7484     0.000430         1.93
IVF-Binary-256-nl223-np11-rf0-signed (query)           2_613.28       117.94     2_731.22       0.0512          NaN         2.00
IVF-Binary-256-nl223-np14-rf0-signed (query)           2_613.28       120.67     2_733.95       0.0481          NaN         2.00
IVF-Binary-256-nl223-np21-rf0-signed (query)           2_613.28       127.90     2_741.18       0.0452          NaN         2.00
IVF-Binary-256-nl223-np11-rf10-signed (query)          2_613.28       181.60     2_794.88       0.3029     0.005893         2.00
IVF-Binary-256-nl223-np11-rf20-signed (query)          2_613.28       231.06     2_844.34       0.4808     0.003183         2.00
IVF-Binary-256-nl223-np14-rf10-signed (query)          2_613.28       186.47     2_799.74       0.2901     0.006353         2.00
IVF-Binary-256-nl223-np14-rf20-signed (query)          2_613.28       239.03     2_852.31       0.4641     0.003455         2.00
IVF-Binary-256-nl223-np21-rf10-signed (query)          2_613.28       198.25     2_811.53       0.2785     0.006784         2.00
IVF-Binary-256-nl223-np21-rf20-signed (query)          2_613.28       255.21     2_868.48       0.4478     0.003655         2.00
IVF-Binary-256-nl223-signed (self)                     2_613.28       555.48     3_168.76       0.7426     0.000488         2.00
IVF-Binary-256-nl316-np15-rf0-signed (query)           2_644.43       128.10     2_772.53       0.0506          NaN         2.09
IVF-Binary-256-nl316-np17-rf0-signed (query)           2_644.43       128.73     2_773.16       0.0491          NaN         2.09
IVF-Binary-256-nl316-np25-rf0-signed (query)           2_644.43       132.07     2_776.50       0.0461          NaN         2.09
IVF-Binary-256-nl316-np15-rf10-signed (query)          2_644.43       205.38     2_849.81       0.3030     0.005872         2.09
IVF-Binary-256-nl316-np15-rf20-signed (query)          2_644.43       240.16     2_884.59       0.4824     0.003181         2.09
IVF-Binary-256-nl316-np17-rf10-signed (query)          2_644.43       188.44     2_832.87       0.2971     0.006058         2.09
IVF-Binary-256-nl316-np17-rf20-signed (query)          2_644.43       243.64     2_888.07       0.4742     0.003294         2.09
IVF-Binary-256-nl316-np25-rf10-signed (query)          2_644.43       196.52     2_840.95       0.2832     0.006557         2.09
IVF-Binary-256-nl316-np25-rf20-signed (query)          2_644.43       256.38     2_900.81       0.4549     0.003561         2.09
IVF-Binary-256-nl316-signed (self)                     2_644.43       558.52     3_202.95       0.7435     0.000482         2.09
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Quantisation stress data - 512 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 512D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        19.07     9_715.79     9_734.86       1.0000     0.000000        97.66
Exhaustive (self)                                         19.07    32_320.44    32_339.51       1.0000     0.000000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_667.56       357.30     6_024.86       0.0197          NaN         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_667.56       488.18     6_155.74       0.1454     0.013787         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_667.56       615.19     6_282.75       0.2497     0.008656         2.03
ExhaustiveBinary-256-random (self)                     5_667.56     1_601.35     7_268.91       0.6222     0.030427         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_184.71       363.32     6_548.03       0.0177          NaN         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_184.71       495.82     6_680.53       0.0885     0.031766         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_184.71       668.03     6_852.74       0.1361     0.014944         2.03
ExhaustiveBinary-256-pca (self)                        6_184.71     1_634.18     7_818.89       0.2209     0.004851         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_101.75       670.34    11_772.09       0.0425          NaN         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_101.75       791.12    11_892.87       0.2372     0.008527         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_101.75       926.29    12_028.04       0.3726     0.004997         4.05
ExhaustiveBinary-512-random (self)                    11_101.75     2_615.99    13_717.74       0.7112     0.040314         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_752.99       678.60    12_431.59       0.0149          NaN         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_752.99       827.80    12_580.78       0.0785     0.060317         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_752.99       973.30    12_726.29       0.1227     0.022920         4.05
ExhaustiveBinary-512-pca (self)                       11_752.99     2_741.35    14_494.34       0.1561     0.023357         4.05
ExhaustiveBinary-1024-random_no_rr (query)            22_248.08     1_177.60    23_425.68       0.0662          NaN         8.10
ExhaustiveBinary-1024-random-rf10 (query)             22_248.08     1_330.91    23_578.99       0.3226     0.006213         8.10
ExhaustiveBinary-1024-random-rf20 (query)             22_248.08     1_497.22    23_745.30       0.4753     0.003526         8.10
ExhaustiveBinary-1024-random (self)                   22_248.08     4_444.78    26_692.86       0.7729     0.047906         8.10
ExhaustiveBinary-1024-pca_no_rr (query)               23_132.52     1_204.14    24_336.67       0.0749          NaN         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                23_132.52     1_357.34    24_489.86       0.3494     0.004665         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                23_132.52     1_510.95    24_643.47       0.5064     0.002500         8.11
ExhaustiveBinary-1024-pca (self)                      23_132.52     4_599.26    27_731.79       0.6913     0.000423         8.11
ExhaustiveBinary-512-signed_no_rr (query)             11_087.39       681.43    11_768.83       0.0425          NaN         4.05
ExhaustiveBinary-512-signed-rf10 (query)              11_087.39       806.91    11_894.30       0.2372     0.008527         4.05
ExhaustiveBinary-512-signed-rf20 (query)              11_087.39       946.23    12_033.62       0.3726     0.004997         4.05
ExhaustiveBinary-512-signed (self)                    11_087.39     2_728.11    13_815.51       0.7112     0.040314         4.05
IVF-Binary-256-nl158-np7-rf0-random (query)            8_185.39       235.73     8_421.12       0.0301          NaN         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           8_185.39       247.32     8_432.71       0.0255          NaN         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           8_185.39       253.37     8_438.76       0.0237          NaN         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           8_185.39       333.56     8_518.95       0.2211     0.007204         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           8_185.39       421.24     8_606.64       0.3773     0.003833         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          8_185.39       343.21     8_528.61       0.1951     0.008457         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          8_185.39       445.06     8_630.46       0.3354     0.004779         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          8_185.39       364.26     8_549.65       0.1845     0.009133         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          8_185.39       457.66     8_643.06       0.3185     0.005265         2.34
IVF-Binary-256-nl158-random (self)                     8_185.39     1_035.83     9_221.22       0.7814     0.000302         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           5_780.26       246.32     6_026.57       0.0303          NaN         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)           5_780.26       248.15     6_028.41       0.0277          NaN         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)           5_780.26       252.22     6_032.48       0.0252          NaN         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)          5_780.26       340.38     6_120.64       0.2240     0.006948         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)          5_780.26       437.02     6_217.27       0.3838     0.003791         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)          5_780.26       342.29     6_122.54       0.2095     0.007638         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)          5_780.26       432.49     6_212.74       0.3612     0.004167         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)          5_780.26       354.60     6_134.86       0.1942     0.008424         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)          5_780.26       446.64     6_226.89       0.3374     0.004657         2.46
IVF-Binary-256-nl223-random (self)                     5_780.26     1_024.59     6_804.85       0.7767     0.000338         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)           5_792.86       258.48     6_051.35       0.0302          NaN         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           5_792.86       260.49     6_053.35       0.0287          NaN         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           5_792.86       270.86     6_063.73       0.0259          NaN         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          5_792.86       353.95     6_146.82       0.2253     0.006978         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          5_792.86       442.70     6_235.56       0.3874     0.003781         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          5_792.86       354.99     6_147.85       0.2169     0.007274         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          5_792.86       444.79     6_237.65       0.3744     0.003960         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          5_792.86       362.11     6_154.98       0.2002     0.008059         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          5_792.86       457.80     6_250.66       0.3475     0.004413         2.65
IVF-Binary-256-nl316-random (self)                     5_792.86     1_058.93     6_851.79       0.7765     0.000336         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               8_783.87       252.50     9_036.37       0.0474          NaN         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              8_783.87       252.08     9_035.95       0.0383          NaN         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              8_783.87       264.14     9_048.01       0.0338          NaN         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              8_783.87       343.85     9_127.71       0.2568     0.005229         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              8_783.87       433.52     9_217.39       0.3997     0.003068         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             8_783.87       378.90     9_162.77       0.2058     0.006692         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             8_783.87       472.65     9_256.52       0.3228     0.004105         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             8_783.87       372.03     9_155.90       0.1801     0.007917         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             8_783.87       481.11     9_264.98       0.2819     0.004873         2.34
IVF-Binary-256-nl158-pca (self)                        8_783.87     1_144.30     9_928.17       0.3375     0.001467         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              6_372.64       266.15     6_638.79       0.0492          NaN         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              6_372.64       255.02     6_627.66       0.0450          NaN         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              6_372.64       272.85     6_645.49       0.0395          NaN         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             6_372.64       395.22     6_767.86       0.2682     0.005104         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             6_372.64       464.55     6_837.19       0.4193     0.002958         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             6_372.64       356.35     6_728.99       0.2454     0.005612         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             6_372.64       460.73     6_833.37       0.3842     0.003344         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             6_372.64       365.78     6_738.42       0.2134     0.006525         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             6_372.64       471.69     6_844.33       0.3346     0.003991         2.47
IVF-Binary-256-nl223-pca (self)                        6_372.64     1_108.22     7_480.86       0.3700     0.001279         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              6_352.98       267.74     6_620.72       0.0499          NaN         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              6_352.98       272.53     6_625.51       0.0477          NaN         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              6_352.98       277.73     6_630.71       0.0420          NaN         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             6_352.98       368.35     6_721.33       0.2726     0.005077         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             6_352.98       474.70     6_827.68       0.4269     0.002905         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             6_352.98       431.47     6_784.45       0.2605     0.005327         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             6_352.98       548.48     6_901.46       0.4081     0.003097         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             6_352.98       450.76     6_803.74       0.2280     0.006161         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             6_352.98       534.63     6_887.61       0.3570     0.003696         2.65
IVF-Binary-256-nl316-pca (self)                        6_352.98     1_261.21     7_614.19       0.3813     0.001216         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           14_020.36       450.89    14_471.24       0.0550          NaN         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          14_020.36       459.73    14_480.08       0.0503          NaN         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          14_020.36       474.81    14_495.17       0.0485          NaN         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          14_020.36       539.77    14_560.13       0.3197     0.004673         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          14_020.36       623.98    14_644.34       0.5000     0.002437         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         14_020.36       556.51    14_576.86       0.2982     0.005190         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         14_020.36       650.31    14_670.67       0.4687     0.002816         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         14_020.36       574.78    14_595.14       0.2893     0.005465         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         14_020.36       676.46    14_696.82       0.4553     0.003028         4.36
IVF-Binary-512-nl158-random (self)                    14_020.36     1_819.23    15_839.58       0.8734     0.000157         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          11_286.21       450.36    11_736.56       0.0551          NaN         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          11_286.21       464.38    11_750.59       0.0524          NaN         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          11_286.21       465.10    11_751.30       0.0500          NaN         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         11_286.21       554.68    11_840.89       0.3198     0.004909         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         11_286.21       628.00    11_914.21       0.4995     0.002623         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         11_286.21       545.91    11_832.11       0.3078     0.005183         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         11_286.21       643.26    11_929.47       0.4831     0.002807         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         11_286.21       565.75    11_851.96       0.2965     0.005384         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         11_286.21       655.31    11_941.52       0.4669     0.002944         4.49
IVF-Binary-512-nl223-random (self)                    11_286.21     1_712.27    12_998.48       0.8701     0.000177         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          11_332.31       464.31    11_796.62       0.0552          NaN         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          11_332.31       478.31    11_810.62       0.0536          NaN         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          11_332.31       474.99    11_807.30       0.0508          NaN         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         11_332.31       557.52    11_889.83       0.3216     0.004922         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         11_332.31       644.76    11_977.07       0.5021     0.002628         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         11_332.31       567.43    11_899.73       0.3146     0.005061         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         11_332.31       649.34    11_981.65       0.4930     0.002718         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         11_332.31       569.05    11_901.36       0.3015     0.005286         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         11_332.31       663.24    11_995.55       0.4742     0.002874         4.67
IVF-Binary-512-nl316-random (self)                    11_332.31     1_761.92    13_094.23       0.8706     0.000175         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              14_476.90       455.69    14_932.59       0.0554          NaN         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             14_476.90       471.44    14_948.34       0.0429          NaN         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             14_476.90       488.61    14_965.51       0.0366          NaN         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             14_476.90       560.38    15_037.28       0.2851     0.004810         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             14_476.90       646.05    15_122.94       0.4336     0.002762         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            14_476.90       572.63    15_049.53       0.2231     0.006584         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            14_476.90       675.68    15_152.58       0.3413     0.003912         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            14_476.90       592.20    15_069.10       0.1894     0.008386         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            14_476.90       767.33    15_244.23       0.2911     0.004829         4.36
IVF-Binary-512-nl158-pca (self)                       14_476.90     1_861.31    16_338.21       0.2813     0.001975         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             12_015.91       462.98    12_478.89       0.0585          NaN         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             12_015.91       467.26    12_483.17       0.0528          NaN         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             12_015.91       479.20    12_495.11       0.0450          NaN         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            12_015.91       561.05    12_576.95       0.3017     0.004674         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            12_015.91       656.97    12_672.87       0.4606     0.002631         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            12_015.91       568.08    12_583.98       0.2734     0.005272         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            12_015.91       663.91    12_679.81       0.4189     0.003047         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            12_015.91       580.43    12_596.33       0.2319     0.006419         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            12_015.91       679.93    12_695.83       0.3575     0.003786         4.49
IVF-Binary-512-nl223-pca (self)                       12_015.91     1_876.89    13_892.80       0.3205     0.001624         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             12_093.18       533.35    12_626.53       0.0596          NaN         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             12_093.18       486.79    12_579.97       0.0564          NaN         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             12_093.18       490.16    12_583.35       0.0482          NaN         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            12_093.18       577.02    12_670.21       0.3073     0.004642         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            12_093.18       668.10    12_761.29       0.4686     0.002586         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            12_093.18       589.21    12_682.39       0.2925     0.004923         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            12_093.18       697.58    12_790.76       0.4463     0.002792         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            12_093.18       602.78    12_695.97       0.2500     0.005916         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            12_093.18       687.78    12_780.97       0.3851     0.003457         4.67
IVF-Binary-512-nl316-pca (self)                       12_093.18     1_859.55    13_952.74       0.3346     0.001520         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          25_015.72       861.13    25_876.85       0.0801          NaN         8.41
IVF-Binary-1024-nl158-np12-rf0-random (query)         25_015.72       886.14    25_901.86       0.0753          NaN         8.41
IVF-Binary-1024-nl158-np17-rf0-random (query)         25_015.72       909.41    25_925.13       0.0734          NaN         8.41
IVF-Binary-1024-nl158-np7-rf10-random (query)         25_015.72       956.87    25_972.59       0.4066     0.003471         8.41
IVF-Binary-1024-nl158-np7-rf20-random (query)         25_015.72     1_036.48    26_052.19       0.5989     0.001714         8.41
IVF-Binary-1024-nl158-np12-rf10-random (query)        25_015.72       973.31    25_989.03       0.3864     0.003842         8.41
IVF-Binary-1024-nl158-np12-rf20-random (query)        25_015.72     1_066.78    26_082.50       0.5728     0.001968         8.41
IVF-Binary-1024-nl158-np17-rf10-random (query)        25_015.72     1_000.20    26_015.92       0.3781     0.004043         8.41
IVF-Binary-1024-nl158-np17-rf20-random (query)        25_015.72     1_108.59    26_124.30       0.5612     0.002121         8.41
IVF-Binary-1024-nl158-random (self)                   25_015.72     3_150.63    28_166.35       0.9356     0.000065         8.41
IVF-Binary-1024-nl223-np11-rf0-random (query)         22_247.96       863.54    23_111.51       0.0801          NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         22_247.96       869.49    23_117.46       0.0771          NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         22_247.96       887.56    23_135.53       0.0744          NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        22_247.96       951.95    23_199.92       0.4065     0.003684         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        22_247.96     1_036.00    23_283.96       0.5973     0.001863         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        22_247.96       955.88    23_203.85       0.3955     0.003876         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        22_247.96     1_048.38    23_296.35       0.5846     0.001990         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        22_247.96       985.12    23_233.09       0.3850     0.003990         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        22_247.96     1_071.31    23_319.28       0.5716     0.002067         8.54
IVF-Binary-1024-nl223-random (self)                   22_247.96     3_084.63    25_332.59       0.9333     0.000075         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         22_219.03       878.51    23_097.54       0.0802          NaN         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)         22_219.03       884.92    23_103.95       0.0783          NaN         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)         22_219.03       895.35    23_114.38       0.0752          NaN         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)        22_219.03       973.69    23_192.72       0.4081     0.003693         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)        22_219.03     1_055.84    23_274.87       0.5998     0.001867         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)        22_219.03       967.92    23_186.95       0.4016     0.003790         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)        22_219.03     1_057.52    23_276.55       0.5922     0.001931         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)        22_219.03       991.96    23_210.99       0.3899     0.003927         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)        22_219.03     1_086.74    23_305.77       0.5780     0.002024         8.72
IVF-Binary-1024-nl316-random (self)                   22_219.03     3_104.80    25_323.83       0.9338     0.000073         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)             25_825.78       882.15    26_707.93       0.0964          NaN         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            25_825.78       907.33    26_733.12       0.0896          NaN         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            25_825.78       928.12    26_753.90       0.0865          NaN         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            25_825.78       973.13    26_798.91       0.4410     0.002767         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            25_825.78     1_059.83    26_885.61       0.6270     0.001368         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           25_825.78     1_003.27    26_829.05       0.4120     0.003173         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           25_825.78     1_096.45    26_922.23       0.5901     0.001652         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           25_825.78     1_027.40    26_853.19       0.3992     0.003409         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           25_825.78     1_131.67    26_957.46       0.5720     0.001826         8.42
IVF-Binary-1024-nl158-pca (self)                      25_825.78     3_248.16    29_073.94       0.7126     0.000367         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            23_421.28       886.26    24_307.54       0.0975          NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            23_421.28       894.98    24_316.26       0.0942          NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            23_421.28       909.27    24_330.55       0.0902          NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           23_421.28       978.70    24_399.98       0.4463     0.002830         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           23_421.28     1_066.83    24_488.11       0.6354     0.001396         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           23_421.28       982.82    24_404.10       0.4329     0.002995         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           23_421.28     1_080.83    24_502.11       0.6184     0.001512         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           23_421.28     1_004.67    24_425.95       0.4167     0.003172         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           23_421.28     1_098.60    24_519.89       0.5972     0.001641         8.54
IVF-Binary-1024-nl223-pca (self)                      23_421.28     3_174.35    26_595.63       0.7111     0.000387         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            23_312.76       902.49    24_215.25       0.0984          NaN         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            23_312.76       906.44    24_219.20       0.0964          NaN         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            23_312.76       917.50    24_230.25       0.0920          NaN         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           23_312.76       995.52    24_308.27       0.4496     0.002816         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           23_312.76     1_083.92    24_396.68       0.6394     0.001382         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           23_312.76       992.96    24_305.71       0.4419     0.002901         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           23_312.76     1_126.67    24_439.43       0.6303     0.001441         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           23_312.76     1_007.77    24_320.53       0.4245     0.003087         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           23_312.76     1_120.99    24_433.75       0.6075     0.001575         8.73
IVF-Binary-1024-nl316-pca (self)                      23_312.76     3_225.81    26_538.57       0.7118     0.000383         8.73
IVF-Binary-512-nl158-np7-rf0-signed (query)           13_749.61       442.30    14_191.91       0.0550          NaN         4.36
IVF-Binary-512-nl158-np12-rf0-signed (query)          13_749.61       458.56    14_208.17       0.0503          NaN         4.36
IVF-Binary-512-nl158-np17-rf0-signed (query)          13_749.61       495.91    14_245.52       0.0485          NaN         4.36
IVF-Binary-512-nl158-np7-rf10-signed (query)          13_749.61       537.65    14_287.27       0.3197     0.004673         4.36
IVF-Binary-512-nl158-np7-rf20-signed (query)          13_749.61       620.73    14_370.34       0.5000     0.002437         4.36
IVF-Binary-512-nl158-np12-rf10-signed (query)         13_749.61       553.29    14_302.91       0.2982     0.005190         4.36
IVF-Binary-512-nl158-np12-rf20-signed (query)         13_749.61       649.74    14_399.35       0.4687     0.002816         4.36
IVF-Binary-512-nl158-np17-rf10-signed (query)         13_749.61       575.52    14_325.14       0.2893     0.005465         4.36
IVF-Binary-512-nl158-np17-rf20-signed (query)         13_749.61       671.10    14_420.71       0.4553     0.003028         4.36
IVF-Binary-512-nl158-signed (self)                    13_749.61     1_737.79    15_487.40       0.8734     0.000157         4.36
IVF-Binary-512-nl223-np11-rf0-signed (query)          11_404.68       458.39    11_863.07       0.0551          NaN         4.49
IVF-Binary-512-nl223-np14-rf0-signed (query)          11_404.68       455.66    11_860.34       0.0524          NaN         4.49
IVF-Binary-512-nl223-np21-rf0-signed (query)          11_404.68       464.41    11_869.09       0.0500          NaN         4.49
IVF-Binary-512-nl223-np11-rf10-signed (query)         11_404.68       545.88    11_950.56       0.3198     0.004909         4.49
IVF-Binary-512-nl223-np11-rf20-signed (query)         11_404.68       629.21    12_033.88       0.4995     0.002623         4.49
IVF-Binary-512-nl223-np14-rf10-signed (query)         11_404.68       555.36    11_960.04       0.3078     0.005183         4.49
IVF-Binary-512-nl223-np14-rf20-signed (query)         11_404.68       639.84    12_044.52       0.4831     0.002807         4.49
IVF-Binary-512-nl223-np21-rf10-signed (query)         11_404.68       574.98    11_979.66       0.2965     0.005384         4.49
IVF-Binary-512-nl223-np21-rf20-signed (query)         11_404.68       653.42    12_058.09       0.4669     0.002944         4.49
IVF-Binary-512-nl223-signed (self)                    11_404.68     1_727.61    13_132.28       0.8701     0.000177         4.49
IVF-Binary-512-nl316-np15-rf0-signed (query)          11_315.15       464.05    11_779.20       0.0552          NaN         4.67
IVF-Binary-512-nl316-np17-rf0-signed (query)          11_315.15       468.77    11_783.92       0.0536          NaN         4.67
IVF-Binary-512-nl316-np25-rf0-signed (query)          11_315.15       484.98    11_800.13       0.0508          NaN         4.67
IVF-Binary-512-nl316-np15-rf10-signed (query)         11_315.15       557.17    11_872.32       0.3216     0.004922         4.67
IVF-Binary-512-nl316-np15-rf20-signed (query)         11_315.15       651.80    11_966.95       0.5021     0.002628         4.67
IVF-Binary-512-nl316-np17-rf10-signed (query)         11_315.15       558.01    11_873.16       0.3146     0.005061         4.67
IVF-Binary-512-nl316-np17-rf20-signed (query)         11_315.15       666.28    11_981.43       0.4930     0.002718         4.67
IVF-Binary-512-nl316-np25-rf10-signed (query)         11_315.15       585.68    11_900.83       0.3015     0.005286         4.67
IVF-Binary-512-nl316-np25-rf20-signed (query)         11_315.15       671.45    11_986.60       0.4742     0.002874         4.67
IVF-Binary-512-nl316-signed (self)                    11_315.15     1_745.51    13_060.66       0.8706     0.000175         4.67
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Quantisation stress data - 1024 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 1024D - Binary Quantisation
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        37.69    22_597.43    22_635.11       1.0000     0.000000       195.31
Exhaustive (self)                                         37.69    75_352.16    75_389.84       1.0000     0.000000       195.31
ExhaustiveBinary-256-random_no_rr (query)             12_021.34       600.75    12_622.09       0.0086          NaN         2.53
ExhaustiveBinary-256-random-rf10 (query)              12_021.34       767.35    12_788.69       0.0837     0.025782         2.53
ExhaustiveBinary-256-random-rf20 (query)              12_021.34       953.81    12_975.15       0.1557     0.016412         2.53
ExhaustiveBinary-256-random (self)                    12_021.34     2_521.69    14_543.03       0.6166     0.023897         2.53
ExhaustiveBinary-256-pca_no_rr (query)                13_179.58       599.81    13_779.39       0.0142          NaN         2.53
ExhaustiveBinary-256-pca-rf10 (query)                 13_179.58       771.40    13_950.99       0.0739     0.025420         2.53
ExhaustiveBinary-256-pca-rf20 (query)                 13_179.58       947.53    14_127.11       0.1169     0.012192         2.53
ExhaustiveBinary-256-pca (self)                       13_179.58     2_549.00    15_728.59       0.1825     0.003568         2.53
ExhaustiveBinary-512-random_no_rr (query)             23_526.95     1_127.64    24_654.59       0.0192          NaN         5.05
ExhaustiveBinary-512-random-rf10 (query)              23_526.95     1_291.16    24_818.11       0.1384     0.013214         5.05
ExhaustiveBinary-512-random-rf20 (query)              23_526.95     1_474.30    25_001.25       0.2398     0.008082         5.05
ExhaustiveBinary-512-random (self)                    23_526.95     4_273.42    27_800.37       0.7044     0.032217         5.05
ExhaustiveBinary-512-pca_no_rr (query)                24_835.00     1_149.55    25_984.54       0.0097          NaN         5.06
ExhaustiveBinary-512-pca-rf10 (query)                 24_835.00     1_308.75    26_143.75       0.0527     0.047667         5.06
ExhaustiveBinary-512-pca-rf20 (query)                 24_835.00     1_515.42    26_350.42       0.0858     0.017508         5.06
ExhaustiveBinary-512-pca (self)                       24_835.00     4_359.67    29_194.67       0.1295     0.011583         5.06
ExhaustiveBinary-1024-random_no_rr (query)            47_173.94     2_144.80    49_318.74       0.0426          NaN        10.10
ExhaustiveBinary-1024-random-rf10 (query)             47_173.94     2_291.03    49_464.97       0.2373     0.007598        10.10
ExhaustiveBinary-1024-random-rf20 (query)             47_173.94     2_487.83    49_661.76       0.3715     0.004379        10.10
ExhaustiveBinary-1024-random (self)                   47_173.94     7_595.20    54_769.13       0.7686     0.037832        10.10
ExhaustiveBinary-1024-pca_no_rr (query)               48_304.68     2_139.03    50_443.71       0.0086          NaN        10.11
ExhaustiveBinary-1024-pca-rf10 (query)                48_304.68     2_325.59    50_630.26       0.0490     0.084671        10.11
ExhaustiveBinary-1024-pca-rf20 (query)                48_304.68     2_514.63    50_819.31       0.0797     0.028511        10.11
ExhaustiveBinary-1024-pca (self)                      48_304.68     7_778.28    56_082.96       0.1033     0.055137        10.11
ExhaustiveBinary-1024-signed_no_rr (query)            46_969.60     2_115.35    49_084.95       0.0426          NaN        10.10
ExhaustiveBinary-1024-signed-rf10 (query)             46_969.60     2_286.21    49_255.81       0.2373     0.007598        10.10
ExhaustiveBinary-1024-signed-rf20 (query)             46_969.60     2_484.02    49_453.62       0.3715     0.004379        10.10
ExhaustiveBinary-1024-signed (self)                   46_969.60     7_597.96    54_567.56       0.7686     0.037832        10.10
IVF-Binary-256-nl158-np7-rf0-random (query)           17_586.79       491.88    18_078.67       0.0178          NaN         3.14
IVF-Binary-256-nl158-np12-rf0-random (query)          17_586.79       502.67    18_089.46       0.0138          NaN         3.14
IVF-Binary-256-nl158-np17-rf0-random (query)          17_586.79       543.77    18_130.56       0.0122          NaN         3.14
IVF-Binary-256-nl158-np7-rf10-random (query)          17_586.79       628.57    18_215.36       0.1535     0.010625         3.14
IVF-Binary-256-nl158-np7-rf20-random (query)          17_586.79       755.75    18_342.54       0.2799     0.005478         3.14
IVF-Binary-256-nl158-np12-rf10-random (query)         17_586.79       651.48    18_238.27       0.1278     0.013663         3.14
IVF-Binary-256-nl158-np12-rf20-random (query)         17_586.79       779.87    18_366.67       0.2376     0.007464         3.14
IVF-Binary-256-nl158-np17-rf10-random (query)         17_586.79       651.53    18_238.32       0.1162     0.015824         3.14
IVF-Binary-256-nl158-np17-rf20-random (query)         17_586.79       795.34    18_382.13       0.2180     0.008833         3.14
IVF-Binary-256-nl158-random (self)                    17_586.79     2_033.71    19_620.50       0.7825     0.000181         3.14
IVF-Binary-256-nl223-np11-rf0-random (query)          11_949.75       520.60    12_470.35       0.0184          NaN         3.40
IVF-Binary-256-nl223-np14-rf0-random (query)          11_949.75       519.03    12_468.77       0.0156          NaN         3.40
IVF-Binary-256-nl223-np21-rf0-random (query)          11_949.75       528.48    12_478.22       0.0133          NaN         3.40
IVF-Binary-256-nl223-np11-rf10-random (query)         11_949.75       662.92    12_612.66       0.1594     0.009780         3.40
IVF-Binary-256-nl223-np11-rf20-random (query)         11_949.75       780.98    12_730.72       0.2906     0.005270         3.40
IVF-Binary-256-nl223-np14-rf10-random (query)         11_949.75       655.98    12_605.72       0.1416     0.011640         3.40
IVF-Binary-256-nl223-np14-rf20-random (query)         11_949.75       790.69    12_740.44       0.2623     0.006218         3.40
IVF-Binary-256-nl223-np21-rf10-random (query)         11_949.75       665.84    12_615.59       0.1240     0.014156         3.40
IVF-Binary-256-nl223-np21-rf20-random (query)         11_949.75       803.07    12_752.81       0.2335     0.007721         3.40
IVF-Binary-256-nl223-random (self)                    11_949.75     2_054.74    14_004.48       0.7845     0.000179         3.40
IVF-Binary-256-nl316-np15-rf0-random (query)          12_057.89       549.89    12_607.79       0.0178          NaN         3.76
IVF-Binary-256-nl316-np17-rf0-random (query)          12_057.89       556.00    12_613.89       0.0165          NaN         3.76
IVF-Binary-256-nl316-np25-rf0-random (query)          12_057.89       558.93    12_616.82       0.0140          NaN         3.76
IVF-Binary-256-nl316-np15-rf10-random (query)         12_057.89       687.94    12_745.84       0.1595     0.009827         3.76
IVF-Binary-256-nl316-np15-rf20-random (query)         12_057.89       820.47    12_878.36       0.2939     0.005127         3.76
IVF-Binary-256-nl316-np17-rf10-random (query)         12_057.89       691.60    12_749.49       0.1506     0.010701         3.76
IVF-Binary-256-nl316-np17-rf20-random (query)         12_057.89       819.66    12_877.56       0.2786     0.005588         3.76
IVF-Binary-256-nl316-np25-rf10-random (query)         12_057.89       693.99    12_751.89       0.1321     0.013102         3.76
IVF-Binary-256-nl316-np25-rf20-random (query)         12_057.89       839.99    12_897.88       0.2470     0.006949         3.76
IVF-Binary-256-nl316-random (self)                    12_057.89     2_162.13    14_220.02       0.7846     0.000179         3.76
IVF-Binary-256-nl158-np7-rf0-pca (query)              18_825.92       499.30    19_325.21       0.0446          NaN         3.15
IVF-Binary-256-nl158-np12-rf0-pca (query)             18_825.92       517.32    19_343.24       0.0369          NaN         3.15
IVF-Binary-256-nl158-np17-rf0-pca (query)             18_825.92       521.46    19_347.38       0.0328          NaN         3.15
IVF-Binary-256-nl158-np7-rf10-pca (query)             18_825.92       646.21    19_472.13       0.2517     0.004757         3.15
IVF-Binary-256-nl158-np7-rf20-pca (query)             18_825.92       782.78    19_608.69       0.3958     0.002838         3.15
IVF-Binary-256-nl158-np12-rf10-pca (query)            18_825.92       647.39    19_473.31       0.2065     0.005811         3.15
IVF-Binary-256-nl158-np12-rf20-pca (query)            18_825.92       793.13    19_619.04       0.3249     0.003651         3.15
IVF-Binary-256-nl158-np17-rf10-pca (query)            18_825.92       660.44    19_486.35       0.1808     0.006669         3.15
IVF-Binary-256-nl158-np17-rf20-pca (query)            18_825.92       809.77    19_635.68       0.2839     0.004282         3.15
IVF-Binary-256-nl158-pca (self)                       18_825.92     2_103.37    20_929.28       0.2954     0.001423         3.15
IVF-Binary-256-nl223-np11-rf0-pca (query)             13_208.82       528.57    13_737.39       0.0455          NaN         3.40
IVF-Binary-256-nl223-np14-rf0-pca (query)             13_208.82       567.81    13_776.63       0.0412          NaN         3.40
IVF-Binary-256-nl223-np21-rf0-pca (query)             13_208.82       538.88    13_747.70       0.0359          NaN         3.40
IVF-Binary-256-nl223-np11-rf10-pca (query)            13_208.82       663.80    13_872.62       0.2576     0.004614         3.40
IVF-Binary-256-nl223-np11-rf20-pca (query)            13_208.82       794.52    14_003.34       0.4039     0.002738         3.40
IVF-Binary-256-nl223-np14-rf10-pca (query)            13_208.82       666.51    13_875.34       0.2331     0.005119         3.40
IVF-Binary-256-nl223-np14-rf20-pca (query)            13_208.82       802.90    14_011.73       0.3664     0.003119         3.40
IVF-Binary-256-nl223-np21-rf10-pca (query)            13_208.82       678.14    13_886.96       0.2008     0.005951         3.40
IVF-Binary-256-nl223-np21-rf20-pca (query)            13_208.82       822.15    14_030.97       0.3160     0.003751         3.40
IVF-Binary-256-nl223-pca (self)                       13_208.82     2_171.53    15_380.35       0.3168     0.001255         3.40
IVF-Binary-256-nl316-np15-rf0-pca (query)             13_292.19       554.18    13_846.37       0.0461          NaN         3.77
IVF-Binary-256-nl316-np17-rf0-pca (query)             13_292.19       555.87    13_848.05       0.0441          NaN         3.77
IVF-Binary-256-nl316-np25-rf0-pca (query)             13_292.19       558.92    13_851.11       0.0387          NaN         3.77
IVF-Binary-256-nl316-np15-rf10-pca (query)            13_292.19       697.61    13_989.79       0.2622     0.004546         3.77
IVF-Binary-256-nl316-np15-rf20-pca (query)            13_292.19       829.27    14_121.46       0.4106     0.002678         3.77
IVF-Binary-256-nl316-np17-rf10-pca (query)            13_292.19       694.00    13_986.19       0.2490     0.004791         3.77
IVF-Binary-256-nl316-np17-rf20-pca (query)            13_292.19       834.50    14_126.69       0.3908     0.002867         3.77
IVF-Binary-256-nl316-np25-rf10-pca (query)            13_292.19       705.05    13_997.24       0.2162     0.005517         3.77
IVF-Binary-256-nl316-np25-rf20-pca (query)            13_292.19       847.29    14_139.48       0.3408     0.003418         3.77
IVF-Binary-256-nl316-pca (self)                       13_292.19     2_246.87    15_539.06       0.3283     0.001182         3.77
IVF-Binary-512-nl158-np7-rf0-random (query)           29_379.21       930.76    30_309.97       0.0294          NaN         5.67
IVF-Binary-512-nl158-np12-rf0-random (query)          29_379.21       939.40    30_318.62       0.0255          NaN         5.67
IVF-Binary-512-nl158-np17-rf0-random (query)          29_379.21       956.43    30_335.64       0.0236          NaN         5.67
IVF-Binary-512-nl158-np7-rf10-random (query)          29_379.21     1_050.14    30_429.35       0.2105     0.007154         5.67
IVF-Binary-512-nl158-np7-rf20-random (query)          29_379.21     1_180.02    30_559.24       0.3621     0.003751         5.67
IVF-Binary-512-nl158-np12-rf10-random (query)         29_379.21     1_067.85    30_447.07       0.1874     0.008181         5.67
IVF-Binary-512-nl158-np12-rf20-random (query)         29_379.21     1_204.02    30_583.24       0.3266     0.004468         5.67
IVF-Binary-512-nl158-np17-rf10-random (query)         29_379.21     1_082.16    30_461.37       0.1767     0.008840         5.67
IVF-Binary-512-nl158-np17-rf20-random (query)         29_379.21     1_226.69    30_605.91       0.3087     0.004892         5.67
IVF-Binary-512-nl158-random (self)                    29_379.21     3_436.20    32_815.41       0.8789     0.000082         5.67
IVF-Binary-512-nl223-np11-rf0-random (query)          23_654.45       948.82    24_603.27       0.0304          NaN         5.92
IVF-Binary-512-nl223-np14-rf0-random (query)          23_654.45       956.70    24_611.15       0.0272          NaN         5.92
IVF-Binary-512-nl223-np21-rf0-random (query)          23_654.45       964.80    24_619.25       0.0245          NaN         5.92
IVF-Binary-512-nl223-np11-rf10-random (query)         23_654.45     1_077.58    24_732.03       0.2167     0.006646         5.92
IVF-Binary-512-nl223-np11-rf20-random (query)         23_654.45     1_212.88    24_867.33       0.3718     0.003560         5.92
IVF-Binary-512-nl223-np14-rf10-random (query)         23_654.45     1_076.40    24_730.85       0.2004     0.007437         5.92
IVF-Binary-512-nl223-np14-rf20-random (query)         23_654.45     1_305.42    24_959.86       0.3481     0.004002         5.92
IVF-Binary-512-nl223-np21-rf10-random (query)         23_654.45     1_092.45    24_746.90       0.1847     0.008241         5.92
IVF-Binary-512-nl223-np21-rf20-random (query)         23_654.45     1_231.91    24_886.35       0.3242     0.004514         5.92
IVF-Binary-512-nl223-random (self)                    23_654.45     3_467.81    27_122.26       0.8799     0.000080         5.92
IVF-Binary-512-nl316-np15-rf0-random (query)          23_662.17       983.52    24_645.69       0.0300          NaN         6.29
IVF-Binary-512-nl316-np17-rf0-random (query)          23_662.17       985.04    24_647.21       0.0284          NaN         6.29
IVF-Binary-512-nl316-np25-rf0-random (query)          23_662.17       995.07    24_657.24       0.0258          NaN         6.29
IVF-Binary-512-nl316-np15-rf10-random (query)         23_662.17     1_103.11    24_765.28       0.2175     0.006697         6.29
IVF-Binary-512-nl316-np15-rf20-random (query)         23_662.17     1_230.73    24_892.90       0.3751     0.003510         6.29
IVF-Binary-512-nl316-np17-rf10-random (query)         23_662.17     1_109.94    24_772.11       0.2089     0.007081         6.29
IVF-Binary-512-nl316-np17-rf20-random (query)         23_662.17     1_240.37    24_902.54       0.3623     0.003727         6.29
IVF-Binary-512-nl316-np25-rf10-random (query)         23_662.17     1_125.31    24_787.48       0.1922     0.007904         6.29
IVF-Binary-512-nl316-np25-rf20-random (query)         23_662.17     1_254.20    24_916.37       0.3357     0.004247         6.29
IVF-Binary-512-nl316-random (self)                    23_662.17     3_558.65    27_220.82       0.8804     0.000080         6.29
IVF-Binary-512-nl158-np7-rf0-pca (query)              30_679.55       939.77    31_619.32       0.0425          NaN         5.67
IVF-Binary-512-nl158-np12-rf0-pca (query)             30_679.55       949.99    31_629.54       0.0335          NaN         5.67
IVF-Binary-512-nl158-np17-rf0-pca (query)             30_679.55       972.54    31_652.09       0.0284          NaN         5.67
IVF-Binary-512-nl158-np7-rf10-pca (query)             30_679.55     1_074.14    31_753.69       0.2373     0.005024         5.67
IVF-Binary-512-nl158-np7-rf20-pca (query)             30_679.55     1_196.70    31_876.25       0.3764     0.003026         5.67
IVF-Binary-512-nl158-np12-rf10-pca (query)            30_679.55     1_079.89    31_759.44       0.1861     0.006335         5.67
IVF-Binary-512-nl158-np12-rf20-pca (query)            30_679.55     1_234.63    31_914.18       0.2964     0.004015         5.67
IVF-Binary-512-nl158-np17-rf10-pca (query)            30_679.55     1_102.27    31_781.82       0.1576     0.007460         5.67
IVF-Binary-512-nl158-np17-rf20-pca (query)            30_679.55     1_243.64    31_923.19       0.2504     0.004808         5.67
IVF-Binary-512-nl158-pca (self)                       30_679.55     3_550.44    34_229.99       0.2400     0.001855         5.67
IVF-Binary-512-nl223-np11-rf0-pca (query)             25_014.36       960.37    25_974.73       0.0430          NaN         5.93
IVF-Binary-512-nl223-np14-rf0-pca (query)             25_014.36       964.63    25_978.99       0.0383          NaN         5.93
IVF-Binary-512-nl223-np21-rf0-pca (query)             25_014.36       976.06    25_990.42       0.0321          NaN         5.93
IVF-Binary-512-nl223-np11-rf10-pca (query)            25_014.36     1_091.50    26_105.86       0.2425     0.004898         5.93
IVF-Binary-512-nl223-np11-rf20-pca (query)            25_014.36     1_230.00    26_244.36       0.3842     0.002926         5.93
IVF-Binary-512-nl223-np14-rf10-pca (query)            25_014.36     1_090.40    26_104.76       0.2157     0.005491         5.93
IVF-Binary-512-nl223-np14-rf20-pca (query)            25_014.36     1_230.87    26_245.23       0.3424     0.003374         5.93
IVF-Binary-512-nl223-np21-rf10-pca (query)            25_014.36     1_174.90    26_189.26       0.1799     0.006528         5.93
IVF-Binary-512-nl223-np21-rf20-pca (query)            25_014.36     1_276.20    26_290.55       0.2853     0.004166         5.93
IVF-Binary-512-nl223-pca (self)                       25_014.36     3_591.82    28_606.18       0.2639     0.001594         5.93
IVF-Binary-512-nl316-np15-rf0-pca (query)             25_055.22       995.68    26_050.90       0.0441          NaN         6.29
IVF-Binary-512-nl316-np17-rf0-pca (query)             25_055.22     1_001.23    26_056.46       0.0415          NaN         6.29
IVF-Binary-512-nl316-np25-rf0-pca (query)             25_055.22     1_022.58    26_077.80       0.0352          NaN         6.29
IVF-Binary-512-nl316-np15-rf10-pca (query)            25_055.22     1_133.73    26_188.96       0.2473     0.004813         6.29
IVF-Binary-512-nl316-np15-rf20-pca (query)            25_055.22     1_339.36    26_394.58       0.3903     0.002866         6.29
IVF-Binary-512-nl316-np17-rf10-pca (query)            25_055.22     1_159.47    26_214.69       0.2335     0.005100         6.29
IVF-Binary-512-nl316-np17-rf20-pca (query)            25_055.22     1_261.55    26_316.77       0.3690     0.003080         6.29
IVF-Binary-512-nl316-np25-rf10-pca (query)            25_055.22     1_135.56    26_190.78       0.1974     0.005985         6.29
IVF-Binary-512-nl316-np25-rf20-pca (query)            25_055.22     1_275.24    26_330.47       0.3133     0.003753         6.29
IVF-Binary-512-nl316-pca (self)                       25_055.22     3_684.70    28_739.92       0.2766     0.001484         6.29
IVF-Binary-1024-nl158-np7-rf0-random (query)          52_919.07     1_786.25    54_705.32       0.0554          NaN        10.72
IVF-Binary-1024-nl158-np12-rf0-random (query)         52_919.07     1_805.71    54_724.78       0.0516          NaN        10.72
IVF-Binary-1024-nl158-np17-rf0-random (query)         52_919.07     1_825.84    54_744.91       0.0495          NaN        10.72
IVF-Binary-1024-nl158-np7-rf10-random (query)         52_919.07     1_889.99    54_809.06       0.3144     0.004566        10.72
IVF-Binary-1024-nl158-np7-rf20-random (query)         52_919.07     2_018.14    54_937.21       0.4901     0.002316        10.72
IVF-Binary-1024-nl158-np12-rf10-random (query)        52_919.07     1_914.01    54_833.09       0.2972     0.004891        10.72
IVF-Binary-1024-nl158-np12-rf20-random (query)        52_919.07     2_053.40    54_972.48       0.4668     0.002520        10.72
IVF-Binary-1024-nl158-np17-rf10-random (query)        52_919.07     1_944.24    54_863.31       0.2883     0.005082        10.72
IVF-Binary-1024-nl158-np17-rf20-random (query)        52_919.07     2_083.52    55_002.59       0.4539     0.002643        10.72
IVF-Binary-1024-nl158-random (self)                   52_919.07     6_265.29    59_184.36       0.9463     0.000028        10.72
IVF-Binary-1024-nl223-np11-rf0-random (query)         47_185.32     1_816.05    49_001.37       0.0569          NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-random (query)         47_185.32     1_821.64    49_006.95       0.0538          NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-random (query)         47_185.32     1_838.88    49_024.20       0.0509          NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-random (query)        47_185.32     1_920.71    49_106.03       0.3214     0.004287        10.98
IVF-Binary-1024-nl223-np11-rf20-random (query)        47_185.32     2_051.32    49_236.64       0.4986     0.002213        10.98
IVF-Binary-1024-nl223-np14-rf10-random (query)        47_185.32     1_945.00    49_130.31       0.3081     0.004618        10.98
IVF-Binary-1024-nl223-np14-rf20-random (query)        47_185.32     2_065.45    49_250.77       0.4819     0.002391        10.98
IVF-Binary-1024-nl223-np21-rf10-random (query)        47_185.32     1_949.13    49_134.45       0.2960     0.004847        10.98
IVF-Binary-1024-nl223-np21-rf20-random (query)        47_185.32     2_099.22    49_284.53       0.4651     0.002538        10.98
IVF-Binary-1024-nl223-random (self)                   47_185.32     6_308.65    53_493.96       0.9468     0.000028        10.98
IVF-Binary-1024-nl316-np15-rf0-random (query)         47_161.83     1_844.55    49_006.39       0.0566          NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-random (query)         47_161.83     1_862.01    49_023.85       0.0550          NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-random (query)         47_161.83     1_864.80    49_026.63       0.0519          NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-random (query)        47_161.83     1_991.36    49_153.20       0.3220     0.004319        11.34
IVF-Binary-1024-nl316-np15-rf20-random (query)        47_161.83     2_083.49    49_245.32       0.5007     0.002209        11.34
IVF-Binary-1024-nl316-np17-rf10-random (query)        47_161.83     1_953.26    49_115.09       0.3151     0.004482        11.34
IVF-Binary-1024-nl316-np17-rf20-random (query)        47_161.83     2_089.17    49_251.01       0.4918     0.002302        11.34
IVF-Binary-1024-nl316-np25-rf10-random (query)        47_161.83     1_968.41    49_130.24       0.3018     0.004752        11.34
IVF-Binary-1024-nl316-np25-rf20-random (query)        47_161.83     2_160.90    49_322.73       0.4739     0.002463        11.34
IVF-Binary-1024-nl316-random (self)                   47_161.83     6_382.42    53_544.25       0.9472     0.000027        11.34
IVF-Binary-1024-nl158-np7-rf0-pca (query)             54_131.03     1_828.02    55_959.05       0.0476          NaN        10.73
IVF-Binary-1024-nl158-np12-rf0-pca (query)            54_131.03     1_834.71    55_965.74       0.0364          NaN        10.73
IVF-Binary-1024-nl158-np17-rf0-pca (query)            54_131.03     1_879.37    56_010.40       0.0304          NaN        10.73
IVF-Binary-1024-nl158-np7-rf10-pca (query)            54_131.03     1_986.09    56_117.12       0.2582     0.004760        10.73
IVF-Binary-1024-nl158-np7-rf20-pca (query)            54_131.03     2_112.06    56_243.08       0.4018     0.002814        10.73
IVF-Binary-1024-nl158-np12-rf10-pca (query)           54_131.03     1_981.19    56_112.21       0.1973     0.006272        10.73
IVF-Binary-1024-nl158-np12-rf20-pca (query)           54_131.03     2_123.39    56_254.41       0.3113     0.003882        10.73
IVF-Binary-1024-nl158-np17-rf10-pca (query)           54_131.03     2_002.05    56_133.08       0.1626     0.007640        10.73
IVF-Binary-1024-nl158-np17-rf20-pca (query)           54_131.03     2_161.02    56_292.05       0.2580     0.004786        10.73
IVF-Binary-1024-nl158-pca (self)                      54_131.03     6_541.31    60_672.34       0.2039     0.002293        10.73
IVF-Binary-1024-nl223-np11-rf0-pca (query)            48_706.95     1_856.94    50_563.89       0.0490          NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-pca (query)            48_706.95     1_865.74    50_572.69       0.0430          NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-pca (query)            48_706.95     1_882.32    50_589.27       0.0349          NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-pca (query)           48_706.95     1_981.61    50_688.57       0.2631     0.004657        10.98
IVF-Binary-1024-nl223-np11-rf20-pca (query)           48_706.95     2_084.53    50_791.48       0.4107     0.002723        10.98
IVF-Binary-1024-nl223-np14-rf10-pca (query)           48_706.95     1_955.45    50_662.41       0.2322     0.005304        10.98
IVF-Binary-1024-nl223-np14-rf20-pca (query)           48_706.95     2_102.59    50_809.54       0.3647     0.003196        10.98
IVF-Binary-1024-nl223-np21-rf10-pca (query)           48_706.95     1_982.94    50_689.89       0.1896     0.006518        10.98
IVF-Binary-1024-nl223-np21-rf20-pca (query)           48_706.95     2_128.65    50_835.61       0.2994     0.004060        10.98
IVF-Binary-1024-nl223-pca (self)                      48_706.95     6_550.58    55_257.53       0.2292     0.001920        10.98
IVF-Binary-1024-nl316-np15-rf0-pca (query)            48_578.52     1_882.28    50_460.81       0.0502          NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-pca (query)            48_578.52     1_889.07    50_467.60       0.0470          NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-pca (query)            48_578.52     1_889.77    50_468.30       0.0390          NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-pca (query)           48_578.52     1_999.61    50_578.14       0.2684     0.004567        11.34
IVF-Binary-1024-nl316-np15-rf20-pca (query)           48_578.52     2_142.99    50_721.51       0.4188     0.002658        11.34
IVF-Binary-1024-nl316-np17-rf10-pca (query)           48_578.52     2_019.64    50_598.17       0.2524     0.004877        11.34
IVF-Binary-1024-nl316-np17-rf20-pca (query)           48_578.52     2_147.28    50_725.80       0.3947     0.002885        11.34
IVF-Binary-1024-nl316-np25-rf10-pca (query)           48_578.52     2_004.62    50_583.14       0.2101     0.005891        11.34
IVF-Binary-1024-nl316-np25-rf20-pca (query)           48_578.52     2_159.97    50_738.50       0.3316     0.003611        11.34
IVF-Binary-1024-nl316-pca (self)                      48_578.52     6_591.56    55_170.08       0.2428     0.001767        11.34
IVF-Binary-1024-nl158-np7-rf0-signed (query)          52_923.52     1_790.34    54_713.86       0.0554          NaN        10.72
IVF-Binary-1024-nl158-np12-rf0-signed (query)         52_923.52     1_849.35    54_772.87       0.0516          NaN        10.72
IVF-Binary-1024-nl158-np17-rf0-signed (query)         52_923.52     1_852.06    54_775.58       0.0495          NaN        10.72
IVF-Binary-1024-nl158-np7-rf10-signed (query)         52_923.52     1_890.28    54_813.80       0.3144     0.004566        10.72
IVF-Binary-1024-nl158-np7-rf20-signed (query)         52_923.52     2_172.67    55_096.19       0.4901     0.002316        10.72
IVF-Binary-1024-nl158-np12-rf10-signed (query)        52_923.52     1_912.22    54_835.75       0.2972     0.004891        10.72
IVF-Binary-1024-nl158-np12-rf20-signed (query)        52_923.52     2_049.74    54_973.26       0.4668     0.002520        10.72
IVF-Binary-1024-nl158-np17-rf10-signed (query)        52_923.52     1_944.67    54_868.19       0.2883     0.005082        10.72
IVF-Binary-1024-nl158-np17-rf20-signed (query)        52_923.52     2_082.42    55_005.94       0.4539     0.002643        10.72
IVF-Binary-1024-nl158-signed (self)                   52_923.52     6_297.28    59_220.80       0.9463     0.000028        10.72
IVF-Binary-1024-nl223-np11-rf0-signed (query)         47_110.19     1_826.37    48_936.56       0.0569          NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-signed (query)         47_110.19     1_832.17    48_942.36       0.0538          NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-signed (query)         47_110.19     1_844.19    48_954.38       0.0509          NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-signed (query)        47_110.19     1_916.17    49_026.36       0.3214     0.004287        10.98
IVF-Binary-1024-nl223-np11-rf20-signed (query)        47_110.19     2_048.32    49_158.51       0.4986     0.002213        10.98
IVF-Binary-1024-nl223-np14-rf10-signed (query)        47_110.19     1_930.94    49_041.13       0.3081     0.004618        10.98
IVF-Binary-1024-nl223-np14-rf20-signed (query)        47_110.19     2_068.55    49_178.74       0.4819     0.002391        10.98
IVF-Binary-1024-nl223-np21-rf10-signed (query)        47_110.19     1_954.73    49_064.92       0.2960     0.004847        10.98
IVF-Binary-1024-nl223-np21-rf20-signed (query)        47_110.19     2_093.84    49_204.03       0.4651     0.002538        10.98
IVF-Binary-1024-nl223-signed (self)                   47_110.19     6_312.31    53_422.50       0.9468     0.000028        10.98
IVF-Binary-1024-nl316-np15-rf0-signed (query)         47_210.65     1_852.27    49_062.91       0.0566          NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-signed (query)         47_210.65     1_865.46    49_076.10       0.0550          NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-signed (query)         47_210.65     1_861.24    49_071.89       0.0519          NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-signed (query)        47_210.65     1_957.40    49_168.05       0.3220     0.004319        11.34
IVF-Binary-1024-nl316-np15-rf20-signed (query)        47_210.65     2_116.66    49_327.30       0.5007     0.002209        11.34
IVF-Binary-1024-nl316-np17-rf10-signed (query)        47_210.65     1_963.72    49_174.36       0.3151     0.004482        11.34
IVF-Binary-1024-nl316-np17-rf20-signed (query)        47_210.65     2_109.14    49_319.78       0.4918     0.002302        11.34
IVF-Binary-1024-nl316-np25-rf10-signed (query)        47_210.65     1_974.92    49_185.57       0.3018     0.004752        11.34
IVF-Binary-1024-nl316-np25-rf20-signed (query)        47_210.65     2_120.51    49_331.15       0.4739     0.002463        11.34
IVF-Binary-1024-nl316-signed (self)                   47_210.65     6_400.75    53_611.40       0.9472     0.000027        11.34
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

### <u>RaBitQ (IVF and exhaustive)</u>

[RaBitQ](https://arxiv.org/abs/2405.12497) is an very powerful
quantisation that combines strong compression with excellent Recalls (even
without re-ranking!). It works better on higher dimensions. In the case of the
`ExhaustiveRaBitQ`, the quantiser itself generates a smaller number of centroids
for quantisation purposes (`sqrt(n)` centroids in this case). On the other
hand for the `IVF-RaBitQ` index, the IVF centroids are directly used for
centroid calculations in the quantiser. The only disadvantage over the binary
quantiser is the reduced query speed due to the more complex approximate
distance calculation.

**Key parameters *(RaBitQ)*:**

- *reranking*: The RaBitQ indices have the option to store the original vectors
  on disk. Once the RaBitQ-specific approximated distance has been leveraged to
  identify the most interesting potential neighbours, the on-disk vectors are
  loaded in and the results are re-ranked. A key parameter here is the
  reranking_factor, i.e., how many more vectors are reranked than the desired k.
  For example 10 means that `10 * k vectors` are scored and then re-ranked. The
  more candidates you allow here, the better the Recall.

**Key parameters *(IVF-specific)*:**

- *Number of lists (nl)*: The number of independent k-means cluster to generate.
  If the structure of the data is unknown, people use `sqrt(n)` as a heuristic.
- *Number of points (np)*: The number of clusters to probe during search.
  Numbers here tend to be `sqrt(nlist)` or up to 5% of the nlist.

#### Correlated data

<details>
<summary><b>Correlated data - 256 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 256D - IVF-RaBitQ
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         9.79     4_241.92     4_251.71       1.0000     0.000000        48.83
Exhaustive (self)                                          9.79    14_418.32    14_428.12       1.0000     0.000000        48.83
ExhaustiveRaBitQ-rf0 (query)                           1_412.16       804.66     2_216.82       0.6238   509.198241         2.84
ExhaustiveRaBitQ-rf5 (query)                           1_412.16       875.72     2_287.88       0.9762     0.273432         2.84
ExhaustiveRaBitQ-rf10 (query)                          1_412.16       928.38     2_340.54       0.9980     0.018895         2.84
ExhaustiveRaBitQ-rf20 (query)                          1_412.16     1_027.41     2_439.57       1.0000     0.000246         2.84
ExhaustiveRaBitQ (self)                                1_412.16     3_108.06     4_520.22       0.9980     0.016800         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       2_130.95       259.61     2_390.55       0.6169          NaN         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      2_130.95       404.82     2_535.77       0.6231          NaN         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      2_130.95       551.22     2_682.17       0.6238          NaN         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      2_130.95       355.11     2_486.06       0.9658     1.738589         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      2_130.95       431.03     2_561.98       0.9673     1.727326         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     2_130.95       508.69     2_639.63       0.9932     0.241006         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     2_130.95       592.39     2_723.34       0.9949     0.228246         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     2_130.95       664.80     2_795.74       0.9979     0.029603         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     2_130.95       757.18     2_888.13       0.9996     0.015543         2.89
IVF-RaBitQ-nl158 (self)                                2_130.95     2_517.23     4_648.17       0.9996     0.015816         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_064.15       326.67     1_390.81       0.6270          NaN         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_064.15       414.10     1_478.25       0.6271          NaN         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_064.15       600.65     1_664.80       0.6271          NaN         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_064.15       420.33     1_484.48       0.9967     0.036592         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_064.15       498.99     1_563.14       0.9983     0.022044         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_064.15       510.56     1_574.70       0.9983     0.015069         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_064.15       589.30     1_653.44       1.0000     0.000213         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_064.15       706.43     1_770.58       0.9983     0.015069         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_064.15       797.56     1_861.70       1.0000     0.000213         2.95
IVF-RaBitQ-nl223 (self)                                1_064.15     2_673.37     3_737.52       1.0000     0.000127         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_277.04       404.04     1_681.08       0.6282          NaN         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_277.04       456.47     1_733.50       0.6284          NaN         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_277.04       658.98     1_936.01       0.6284          NaN         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_277.04       501.53     1_778.56       0.9961     0.037378         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_277.04       579.43     1_856.47       0.9978     0.022430         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_277.04       553.65     1_830.68       0.9979     0.017901         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_277.04       631.35     1_908.39       0.9996     0.002966         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_277.04       756.09     2_033.12       0.9983     0.015147         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_277.04       841.95     2_118.99       1.0000     0.000187         3.04
IVF-RaBitQ-nl316 (self)                                1_277.04     2_801.78     4_078.82       1.0000     0.000231         3.04
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 512D - IVF-RaBitQ
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        20.25     9_880.15     9_900.40       1.0000     0.000000        97.66
Exhaustive (self)                                         20.25    33_301.69    33_321.94       1.0000     0.000000        97.66
ExhaustiveRaBitQ-rf0 (query)                           3_997.89     2_334.44     6_332.33       0.6393  2557.546029         5.23
ExhaustiveRaBitQ-rf5 (query)                           3_997.89     2_554.10     6_552.00       0.9805     0.692624         5.23
ExhaustiveRaBitQ-rf10 (query)                          3_997.89     2_491.92     6_489.81       0.9986     0.064651         5.23
ExhaustiveRaBitQ-rf20 (query)                          3_997.89     2_570.77     6_568.66       0.9999     0.019619         5.23
ExhaustiveRaBitQ (self)                                3_997.89     8_161.43    12_159.32       0.9986     0.051589         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_541.80       739.86     6_281.66       0.6324          NaN         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_541.80     1_208.74     6_750.54       0.6390          NaN         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_541.80     1_669.60     7_211.40       0.6393          NaN         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_541.80       856.21     6_398.01       0.9695     5.111295         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_541.80       955.97     6_497.77       0.9706     5.086144         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_541.80     1_326.18     6_867.97       0.9968     0.376475         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_541.80     1_434.96     6_976.76       0.9981     0.344071         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_541.80     1_785.68     7_327.48       0.9986     0.061060         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_541.80     1_901.93     7_443.73       0.9999     0.025746         5.32
IVF-RaBitQ-nl158 (self)                                5_541.80     6_332.40    11_874.20       0.9999     0.021321         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_402.16     1_070.89     4_473.05       0.6415          NaN         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_402.16     1_355.81     4_757.96       0.6420          NaN         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_402.16     2_001.39     5_403.55       0.6420          NaN         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_402.16     1_181.82     4_583.98       0.9951     0.378923         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_402.16     1_286.69     4_688.85       0.9964     0.341769         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_402.16     1_464.07     4_866.23       0.9986     0.045527         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_402.16     1_572.77     4_974.93       0.9999     0.005683         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_402.16     2_111.30     5_513.46       0.9986     0.045527         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_402.16     2_251.63     5_653.78       0.9999     0.005683         5.44
IVF-RaBitQ-nl223 (self)                                3_402.16     7_433.60    10_835.76       0.9999     0.013992         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      3_665.45     1_385.61     5_051.06       0.6421          NaN         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      3_665.45     1_571.18     5_236.62       0.6423          NaN         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      3_665.45     2_271.31     5_936.76       0.6423          NaN         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     3_665.45     1_493.51     5_158.96       0.9976     0.114281         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     3_665.45     1_650.62     5_316.07       0.9988     0.082729         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     3_665.45     1_670.23     5_335.67       0.9987     0.039297         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     3_665.45     1_778.68     5_444.13       0.9999     0.007779         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     3_665.45     2_376.23     6_041.68       0.9987     0.038766         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     3_665.45     2_491.85     6_157.29       1.0000     0.007248         5.63
IVF-RaBitQ-nl316 (self)                                3_665.45     8_301.86    11_967.31       0.9999     0.006527         5.63
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 1024 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 1024D - IVF-RaBitQ
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        40.74    22_604.53    22_645.27       1.0000     0.000000       195.31
Exhaustive (self)                                         40.74    75_238.86    75_279.60       1.0000     0.000000       195.31
ExhaustiveRaBitQ-rf0 (query)                          14_466.79     9_539.10    24_005.89       0.0429 14396.745902        11.50
ExhaustiveRaBitQ-rf5 (query)                          14_466.79     9_358.16    23_824.95       0.1234  2350.683225        11.50
ExhaustiveRaBitQ-rf10 (query)                         14_466.79     9_318.21    23_785.00       0.2113  1921.675110        11.50
ExhaustiveRaBitQ-rf20 (query)                         14_466.79     9_465.99    23_932.78       0.3595  1503.516734        11.50
ExhaustiveRaBitQ (self)                               14_466.79    31_078.42    45_545.20       0.2087  1922.869256        11.50
IVF-RaBitQ-nl158-np7-rf0 (query)                      17_956.01     3_032.49    20_988.50       0.0414          NaN        11.68
IVF-RaBitQ-nl158-np12-rf0 (query)                     17_956.01     5_159.97    23_115.98       0.0413          NaN        11.68
IVF-RaBitQ-nl158-np17-rf0 (query)                     17_956.01     7_232.97    25_188.98       0.0412          NaN        11.68
IVF-RaBitQ-nl158-np7-rf10 (query)                     17_956.01     3_089.25    21_045.26       0.2113  1896.666023        11.68
IVF-RaBitQ-nl158-np7-rf20 (query)                     17_956.01     3_224.93    21_180.93       0.3639  1467.572904        11.68
IVF-RaBitQ-nl158-np12-rf10 (query)                    17_956.01     5_097.91    23_053.92       0.2074  1926.811490        11.68
IVF-RaBitQ-nl158-np12-rf20 (query)                    17_956.01     5_250.05    23_206.06       0.3575  1500.383598        11.68
IVF-RaBitQ-nl158-np17-rf10 (query)                    17_956.01     7_247.43    25_203.44       0.2074  1927.032482        11.68
IVF-RaBitQ-nl158-np17-rf20 (query)                    17_956.01     7_262.30    25_218.31       0.3574  1500.796501        11.68
IVF-RaBitQ-nl158 (self)                               17_956.01    24_238.00    42_194.01       0.3563  1497.879216        11.68
IVF-RaBitQ-nl223-np11-rf0 (query)                     12_584.21     4_616.14    17_200.34       0.0400          NaN        11.93
IVF-RaBitQ-nl223-np14-rf0 (query)                     12_584.21     5_793.53    18_377.74       0.0399          NaN        11.93
IVF-RaBitQ-nl223-np21-rf0 (query)                     12_584.21     8_648.69    21_232.90       0.0399          NaN        11.93
IVF-RaBitQ-nl223-np11-rf10 (query)                    12_584.21     4_552.31    17_136.52       0.2056  1918.207480        11.93
IVF-RaBitQ-nl223-np11-rf20 (query)                    12_584.21     4_699.19    17_283.40       0.3580  1480.670314        11.93
IVF-RaBitQ-nl223-np14-rf10 (query)                    12_584.21     5_734.49    18_318.69       0.2046  1924.355065        11.93
IVF-RaBitQ-nl223-np14-rf20 (query)                    12_584.21     5_863.48    18_447.69       0.3560  1488.040082        11.93
IVF-RaBitQ-nl223-np21-rf10 (query)                    12_584.21     8_483.26    21_067.46       0.2042  1926.423341        11.93
IVF-RaBitQ-nl223-np21-rf20 (query)                    12_584.21     8_629.78    21_213.98       0.3551  1490.448247        11.93
IVF-RaBitQ-nl223 (self)                               12_584.21    28_670.19    41_254.40       0.3531  1493.472165        11.93
IVF-RaBitQ-nl316-np15-rf0 (query)                     13_308.31     6_106.59    19_414.90       0.0387          NaN        12.30
IVF-RaBitQ-nl316-np17-rf0 (query)                     13_308.31     6_893.21    20_201.52       0.0386          NaN        12.30
IVF-RaBitQ-nl316-np25-rf0 (query)                     13_308.31    10_069.58    23_377.89       0.0386          NaN        12.30
IVF-RaBitQ-nl316-np15-rf10 (query)                    13_308.31     6_042.53    19_350.84       0.2025  1916.980021        12.30
IVF-RaBitQ-nl316-np15-rf20 (query)                    13_308.31     6_175.97    19_484.28       0.3563  1470.114712        12.30
IVF-RaBitQ-nl316-np17-rf10 (query)                    13_308.31     6_811.26    20_119.57       0.2018  1919.979585        12.30
IVF-RaBitQ-nl316-np17-rf20 (query)                    13_308.31     6_922.60    20_230.91       0.3552  1473.830685        12.30
IVF-RaBitQ-nl316-np25-rf10 (query)                    13_308.31     9_835.56    23_143.88       0.2008  1924.427166        12.30
IVF-RaBitQ-nl316-np25-rf20 (query)                    13_308.31    10_024.43    23_332.74       0.3537  1478.603995        12.30
IVF-RaBitQ-nl316 (self)                               13_308.31    33_311.47    46_619.78       0.3508  1482.272644        12.30
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 256D - IVF-RaBitQ
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        10.05     4_248.77     4_258.82       1.0000     0.000000        48.83
Exhaustive (self)                                         10.05    14_798.40    14_808.45       1.0000     0.000000        48.83
ExhaustiveRaBitQ-rf0 (query)                           1_454.72       825.06     2_279.78       0.6862  1193.634759         2.84
ExhaustiveRaBitQ-rf5 (query)                           1_454.72       883.38     2_338.10       0.9929     0.200102         2.84
ExhaustiveRaBitQ-rf10 (query)                          1_454.72       940.13     2_394.85       0.9999     0.002866         2.84
ExhaustiveRaBitQ-rf20 (query)                          1_454.72     1_030.60     2_485.32       1.0000     0.000051         2.84
ExhaustiveRaBitQ (self)                                1_454.72     3_127.34     4_582.06       0.9998     0.003581         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       2_133.59       244.53     2_378.12       0.6859          NaN         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      2_133.59       404.46     2_538.05       0.6863          NaN         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      2_133.59       559.30     2_692.89       0.6863          NaN         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      2_133.59       334.64     2_468.23       0.9979     0.157239         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      2_133.59       410.26     2_543.85       0.9980     0.154229         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     2_133.59       505.00     2_638.59       0.9998     0.002991         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     2_133.59       584.91     2_718.50       1.0000     0.000000         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     2_133.59       674.98     2_808.57       0.9998     0.002991         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     2_133.59       753.65     2_887.24       1.0000     0.000000         2.89
IVF-RaBitQ-nl158 (self)                                2_133.59     2_516.31     4_649.90       1.0000     0.000000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_096.41       327.10     1_423.51       0.6865          NaN         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_096.41       415.01     1_511.42       0.6870          NaN         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_096.41       612.40     1_708.81       0.6872          NaN         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_096.41       422.52     1_518.93       0.9969     0.239423         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_096.41       499.71     1_596.12       0.9970     0.237272         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_096.41       507.11     1_603.52       0.9989     0.070782         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_096.41       587.50     1_683.91       0.9991     0.068244         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_096.41       711.25     1_807.67       0.9999     0.002538         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_096.41       797.21     1_893.62       1.0000     0.000000         2.95
IVF-RaBitQ-nl223 (self)                                1_096.41     2_845.92     3_942.33       1.0000     0.000000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_321.34       408.65     1_729.99       0.6884          NaN         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_321.34       461.72     1_783.06       0.6886          NaN         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_321.34       661.97     1_983.31       0.6887          NaN         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_321.34       509.47     1_830.80       0.9978     0.150660         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_321.34       585.42     1_906.76       0.9980     0.147613         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_321.34       559.26     1_880.60       0.9990     0.061890         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_321.34       641.58     1_962.91       0.9991     0.058739         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_321.34       769.75     2_091.09       0.9999     0.003138         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_321.34       852.34     2_173.68       1.0000     0.000000         3.04
IVF-RaBitQ-nl316 (self)                                1_321.34     2_838.45     4_159.79       1.0000     0.000000         3.04
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 512D - IVF-RaBitQ
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        20.00     9_848.26     9_868.26       1.0000     0.000000        97.66
Exhaustive (self)                                         20.00    33_237.78    33_257.78       1.0000     0.000000        97.66
ExhaustiveRaBitQ-rf0 (query)                           4_069.82     2_376.33     6_446.15       0.7001  5670.051223         5.23
ExhaustiveRaBitQ-rf5 (query)                           4_069.82     2_438.26     6_508.09       0.9944     0.479492         5.23
ExhaustiveRaBitQ-rf10 (query)                          4_069.82     2_499.57     6_569.39       0.9998     0.067485         5.23
ExhaustiveRaBitQ-rf20 (query)                          4_069.82     2_622.85     6_692.68       0.9999     0.061372         5.23
ExhaustiveRaBitQ (self)                                4_069.82     8_336.19    12_406.01       0.9998     0.086293         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_572.71       721.47     6_294.18       0.6970          NaN         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_572.71     1_209.10     6_781.81       0.7002          NaN         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_572.71     1_694.56     7_267.27       0.7002          NaN         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_572.71       837.93     6_410.64       0.9888     3.129748         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_572.71       935.05     6_507.77       0.9889     3.124182         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_572.71     1_319.48     6_892.19       0.9998     0.062482         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_572.71     1_427.06     6_999.77       0.9999     0.056845         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_572.71     1_808.29     7_381.00       0.9998     0.062482         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_572.71     1_920.41     7_493.12       0.9999     0.056845         5.32
IVF-RaBitQ-nl158 (self)                                5_572.71     6_387.50    11_960.21       0.9999     0.054815         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_294.91     1_078.90     4_373.81       0.7002          NaN         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_294.91     1_354.32     4_649.23       0.7008          NaN         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_294.91     1_982.25     5_277.16       0.7008          NaN         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_294.91     1_183.34     4_478.25       0.9975     0.628189         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_294.91     1_298.62     4_593.53       0.9977     0.622248         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_294.91     1_464.72     4_759.63       0.9996     0.092321         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_294.91     1_578.50     4_873.41       0.9998     0.085764         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_294.91     2_095.19     5_390.10       0.9997     0.076300         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_294.91     2_210.69     5_505.60       0.9999     0.069743         5.44
IVF-RaBitQ-nl223 (self)                                3_294.91     7_337.00    10_631.91       0.9999     0.061579         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      3_753.52     1_392.58     5_146.10       0.7016          NaN         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      3_753.52     1_558.55     5_312.07       0.7020          NaN         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      3_753.52     2_259.31     6_012.83       0.7021          NaN         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     3_753.52     1_534.23     5_287.75       0.9982     0.459604         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     3_753.52     1_594.22     5_347.73       0.9982     0.453734         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     3_753.52     1_663.70     5_417.22       0.9994     0.140853         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     3_753.52     1_768.43     5_521.95       0.9995     0.134984         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     3_753.52     2_356.60     6_110.12       0.9998     0.058040         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     3_753.52     2_469.38     6_222.90       0.9999     0.052092         5.63
IVF-RaBitQ-nl316 (self)                                3_753.52     8_221.35    11_974.87       0.9999     0.068818         5.63
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 1024 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 1024D - IVF-RaBitQ
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        40.41    22_685.16    22_725.57       1.0000     0.000000       195.31
Exhaustive (self)                                         40.41    75_546.56    75_586.97       1.0000     0.000000       195.31
ExhaustiveRaBitQ-rf0 (query)                          14_619.28     9_794.95    24_414.23       0.0399 25574.201622        11.50
ExhaustiveRaBitQ-rf5 (query)                          14_619.28     9_499.47    24_118.75       0.1235  4781.029744        11.50
ExhaustiveRaBitQ-rf10 (query)                         14_619.28     9_510.96    24_130.24       0.2143  3800.599789        11.50
ExhaustiveRaBitQ-rf20 (query)                         14_619.28     9_649.98    24_269.26       0.3671  2881.100363        11.50
ExhaustiveRaBitQ (self)                               14_619.28    31_991.73    46_611.01       0.2161  3787.679792        11.50
IVF-RaBitQ-nl158-np7-rf0 (query)                      18_027.42     3_022.41    21_049.83       0.0392          NaN        11.68
IVF-RaBitQ-nl158-np12-rf0 (query)                     18_027.42     5_111.29    23_138.72       0.0392          NaN        11.68
IVF-RaBitQ-nl158-np17-rf0 (query)                     18_027.42     7_239.01    25_266.43       0.0392          NaN        11.68
IVF-RaBitQ-nl158-np7-rf10 (query)                     18_027.42     3_105.66    21_133.08       0.2120  3787.418526        11.68
IVF-RaBitQ-nl158-np7-rf20 (query)                     18_027.42     3_246.86    21_274.28       0.3689  2844.041245        11.68
IVF-RaBitQ-nl158-np12-rf10 (query)                    18_027.42     5_129.75    23_157.17       0.2103  3807.178034        11.68
IVF-RaBitQ-nl158-np12-rf20 (query)                    18_027.42     5_316.92    23_344.34       0.3658  2866.235568        11.68
IVF-RaBitQ-nl158-np17-rf10 (query)                    18_027.42     7_198.11    25_225.53       0.2103  3807.178034        11.68
IVF-RaBitQ-nl158-np17-rf20 (query)                    18_027.42     7_325.45    25_352.87       0.3658  2866.235568        11.68
IVF-RaBitQ-nl158 (self)                               18_027.42    24_305.25    42_332.68       0.3694  2855.515816        11.68
IVF-RaBitQ-nl223-np11-rf0 (query)                     12_707.91     4_579.96    17_287.87       0.0382          NaN        11.93
IVF-RaBitQ-nl223-np14-rf0 (query)                     12_707.91     5_836.47    18_544.38       0.0382          NaN        11.93
IVF-RaBitQ-nl223-np21-rf0 (query)                     12_707.91     8_666.33    21_374.24       0.0382          NaN        11.93
IVF-RaBitQ-nl223-np11-rf10 (query)                    12_707.91     4_557.03    17_264.94       0.2092  3786.075187        11.93
IVF-RaBitQ-nl223-np11-rf20 (query)                    12_707.91     4_709.88    17_417.79       0.3659  2831.164823        11.93
IVF-RaBitQ-nl223-np14-rf10 (query)                    12_707.91     5_743.05    18_450.96       0.2086  3792.203830        11.93
IVF-RaBitQ-nl223-np14-rf20 (query)                    12_707.91     6_118.39    18_826.30       0.3646  2838.864334        11.93
IVF-RaBitQ-nl223-np21-rf10 (query)                    12_707.91     9_092.28    21_800.19       0.2086  3792.203830        11.93
IVF-RaBitQ-nl223-np21-rf20 (query)                    12_707.91     9_324.24    22_032.15       0.3646  2838.864334        11.93
IVF-RaBitQ-nl223 (self)                               12_707.91    29_113.15    41_821.06       0.3680  2829.759729        11.93
IVF-RaBitQ-nl316-np15-rf0 (query)                     13_369.97     6_143.04    19_513.01       0.0355          NaN        12.30
IVF-RaBitQ-nl316-np17-rf0 (query)                     13_369.97     6_983.34    20_353.31       0.0354          NaN        12.30
IVF-RaBitQ-nl316-np25-rf0 (query)                     13_369.97    10_166.89    23_536.86       0.0354          NaN        12.30
IVF-RaBitQ-nl316-np15-rf10 (query)                    13_369.97     6_112.85    19_482.82       0.2014  3804.522962        12.30
IVF-RaBitQ-nl316-np15-rf20 (query)                    13_369.97     6_248.51    19_618.49       0.3612  2815.103523        12.30
IVF-RaBitQ-nl316-np17-rf10 (query)                    13_369.97     6_892.30    20_262.28       0.2008  3809.855539        12.30
IVF-RaBitQ-nl316-np17-rf20 (query)                    13_369.97     7_059.78    20_429.76       0.3604  2820.290564        12.30
IVF-RaBitQ-nl316-np25-rf10 (query)                    13_369.97     9_987.51    23_357.48       0.2008  3809.855539        12.30
IVF-RaBitQ-nl316-np25-rf20 (query)                    13_369.97    10_126.52    23_496.49       0.3604  2820.290564        12.30
IVF-RaBitQ-nl316 (self)                               13_369.97    33_672.15    47_042.12       0.3644  2809.947648        12.30
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

#### Quantisation (stress) data

<details>
<summary><b>Quantisation stress data - 256 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 256D - IVF-RaBitQ
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         6.79     4_268.26     4_275.05       1.0000     0.000000        48.83
Exhaustive (self)                                          6.79    14_351.78    14_358.56       1.0000     0.000000        48.83
ExhaustiveRaBitQ-rf0 (query)                           1_202.56     1_092.13     2_294.69       0.3458     0.155870         2.84
ExhaustiveRaBitQ-rf5 (query)                           1_202.56     1_161.90     2_364.46       0.7547     0.000682         2.84
ExhaustiveRaBitQ-rf10 (query)                          1_202.56     1_228.87     2_431.43       0.8884     0.000242         2.84
ExhaustiveRaBitQ-rf20 (query)                          1_202.56     1_333.72     2_536.28       0.9630     0.000068         2.84
ExhaustiveRaBitQ (self)                                1_202.56     4_071.69     5_274.24       1.0000     0.000000         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_800.94       315.89     2_116.83       0.3617          NaN         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_800.94       547.31     2_348.25       0.3614          NaN         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_800.94       769.52     2_570.45       0.3612          NaN         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_800.94       433.22     2_234.15       0.9035     0.000209         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_800.94       479.15     2_280.08       0.9688     0.000072         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_800.94       647.51     2_448.45       0.9056     0.000190         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_800.94       794.10     2_595.04       0.9718     0.000050         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_800.94       896.00     2_696.94       0.9059     0.000187         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_800.94       991.58     2_792.51       0.9722     0.000047         2.89
IVF-RaBitQ-nl158 (self)                                1_800.94     3_279.58     5_080.52       1.0000     0.000000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                        706.16       329.94     1_036.10       0.4010          NaN         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                        706.16       413.74     1_119.90       0.4008          NaN         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                        706.16       620.97     1_327.13       0.4004          NaN         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                       706.16       416.26     1_122.41       0.9356     0.000118         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                       706.16       482.83     1_188.99       0.9845     0.000026         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                       706.16       506.92     1_213.08       0.9359     0.000116         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                       706.16       581.43     1_287.58       0.9850     0.000023         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                       706.16       719.81     1_425.97       0.9360     0.000115         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                       706.16       808.38     1_514.53       0.9854     0.000022         2.95
IVF-RaBitQ-nl223 (self)                                  706.16     2_614.31     3_320.46       1.0000     0.000000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                        700.34       405.15     1_105.48       0.4148          NaN         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                        700.34       454.44     1_154.78       0.4147          NaN         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                        700.34       708.07     1_408.41       0.4144          NaN         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                       700.34       496.70     1_197.03       0.9437     0.000097         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                       700.34       564.50     1_264.84       0.9877     0.000019         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                       700.34       551.26     1_251.59       0.9436     0.000097         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                       700.34       620.35     1_320.68       0.9878     0.000018         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                       700.34       755.98     1_456.31       0.9435     0.000097         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                       700.34       834.50     1_534.84       0.9881     0.000017         3.04
IVF-RaBitQ-nl316 (self)                                  700.34     2_765.95     3_466.29       1.0000     0.000000         3.04
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Quantisation stress data - 512 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 512D - IVF-RaBitQ
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        19.90     9_813.34     9_833.24       1.0000     0.000000        97.66
Exhaustive (self)                                         19.90    33_283.41    33_303.31       1.0000     0.000000        97.66
ExhaustiveRaBitQ-rf0 (query)                           3_564.91     3_020.80     6_585.71       0.3332     0.148992         5.23
ExhaustiveRaBitQ-rf5 (query)                           3_564.91     3_105.55     6_670.46       0.7285     0.000732         5.23
ExhaustiveRaBitQ-rf10 (query)                          3_564.91     3_178.72     6_743.62       0.8641     0.000288         5.23
ExhaustiveRaBitQ-rf20 (query)                          3_564.91     3_321.99     6_886.90       0.9459     0.000098         5.23
ExhaustiveRaBitQ (self)                                3_564.91    10_680.14    14_245.05       0.9996     0.000001         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       4_863.70       809.13     5_672.83       0.3606          NaN         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      4_863.70     1_602.73     6_466.43       0.3591          NaN         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      4_863.70     2_083.05     6_946.74       0.3586          NaN         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      4_863.70       990.22     5_853.92       0.8886     0.000222         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      4_863.70     1_049.86     5_913.56       0.9556     0.000085         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     4_863.70     1_541.18     6_404.88       0.8908     0.000210         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     4_863.70     1_782.63     6_646.33       0.9606     0.000066         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     4_863.70     2_204.55     7_068.25       0.8904     0.000210         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     4_863.70     2_432.84     7_296.54       0.9609     0.000065         5.32
IVF-RaBitQ-nl158 (self)                                4_863.70     7_410.23    12_273.92       0.9998     0.000000         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      2_456.46     1_057.71     3_514.17       0.3848          NaN         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      2_456.46     1_333.48     3_789.94       0.3842          NaN         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      2_456.46     1_972.47     4_428.93       0.3833          NaN         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     2_456.46     1_168.60     3_625.06       0.9139     0.000156         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     2_456.46     1_264.58     3_721.04       0.9706     0.000050         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     2_456.46     1_437.60     3_894.06       0.9148     0.000152         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     2_456.46     1_542.10     3_998.56       0.9729     0.000044         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     2_456.46     2_215.53     4_671.99       0.9148     0.000150         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     2_456.46     2_196.77     4_653.23       0.9743     0.000039         5.44
IVF-RaBitQ-nl223 (self)                                2_456.46     7_253.11     9_709.57       0.9999     0.000000         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      2_422.14     1_379.35     3_801.49       0.3961          NaN         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      2_422.14     1_556.06     3_978.21       0.3958          NaN         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      2_422.14     2_249.72     4_671.86       0.3948          NaN         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     2_422.14     1_483.53     3_905.67       0.9223     0.000138         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     2_422.14     1_597.96     4_020.10       0.9745     0.000043         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     2_422.14     1_651.25     4_073.39       0.9227     0.000137         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     2_422.14     1_757.30     4_179.44       0.9756     0.000040         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     2_422.14     2_345.42     4_767.56       0.9232     0.000134         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     2_422.14     2_473.98     4_896.12       0.9775     0.000035         5.63
IVF-RaBitQ-nl316 (self)                                2_422.14     8_142.94    10_565.08       0.9999     0.000000         5.63
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Quantisation stress data - 1024 dimensions</b>:</summary>
</br>
<pre><code>
================================================================================================================================
Benchmark: 50k samples, 1024D - IVF-RaBitQ
================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k   Dist Error    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        38.05    22_900.58    22_938.63       1.0000     0.000000       195.31
Exhaustive (self)                                         38.05    75_290.88    75_328.93       1.0000     0.000000       195.31
ExhaustiveRaBitQ-rf0 (query)                          13_311.73    10_512.04    23_823.77       0.0237     0.927991        11.50
ExhaustiveRaBitQ-rf5 (query)                          13_311.73    10_236.90    23_548.62       0.0587     0.013868        11.50
ExhaustiveRaBitQ-rf10 (query)                         13_311.73    10_268.61    23_580.34       0.0961     0.010703        11.50
ExhaustiveRaBitQ-rf20 (query)                         13_311.73    10_437.42    23_749.15       0.1705     0.008001        11.50
ExhaustiveRaBitQ (self)                               13_311.73    34_461.02    47_772.75       0.2337     0.003281        11.50
IVF-RaBitQ-nl158-np7-rf0 (query)                      16_324.66     3_052.28    19_376.95       0.0263          NaN        11.68
IVF-RaBitQ-nl158-np12-rf0 (query)                     16_324.66     5_193.28    21_517.95       0.0260          NaN        11.68
IVF-RaBitQ-nl158-np17-rf0 (query)                     16_324.66     7_332.07    23_656.73       0.0259          NaN        11.68
IVF-RaBitQ-nl158-np7-rf10 (query)                     16_324.66     3_094.32    19_418.98       0.1106     0.007922        11.68
IVF-RaBitQ-nl158-np7-rf20 (query)                     16_324.66     3_239.25    19_563.92       0.2013     0.005594        11.68
IVF-RaBitQ-nl158-np12-rf10 (query)                    16_324.66     5_138.45    21_463.11       0.1051     0.008611        11.68
IVF-RaBitQ-nl158-np12-rf20 (query)                    16_324.66     5_361.40    21_686.06       0.1874     0.006218        11.68
IVF-RaBitQ-nl158-np17-rf10 (query)                    16_324.66     7_226.91    23_551.57       0.1030     0.009041        11.68
IVF-RaBitQ-nl158-np17-rf20 (query)                    16_324.66     7_364.42    23_689.08       0.1822     0.006605        11.68
IVF-RaBitQ-nl158 (self)                               16_324.66    24_551.37    40_876.03       0.4799     0.001492        11.68
IVF-RaBitQ-nl223-np11-rf0 (query)                     10_714.72     4_555.20    15_269.92       0.0277          NaN        11.93
IVF-RaBitQ-nl223-np14-rf0 (query)                     10_714.72     5_779.47    16_494.19       0.0275          NaN        11.93
IVF-RaBitQ-nl223-np21-rf0 (query)                     10_714.72     8_668.05    19_382.77       0.0272          NaN        11.93
IVF-RaBitQ-nl223-np11-rf10 (query)                    10_714.72     4_584.44    15_299.16       0.1148     0.007517        11.93
IVF-RaBitQ-nl223-np11-rf20 (query)                    10_714.72     4_730.83    15_445.54       0.2075     0.005254        11.93
IVF-RaBitQ-nl223-np14-rf10 (query)                    10_714.72     5_734.72    16_449.44       0.1120     0.007794        11.93
IVF-RaBitQ-nl223-np14-rf20 (query)                    10_714.72     5_868.64    16_583.35       0.2007     0.005499        11.93
IVF-RaBitQ-nl223-np21-rf10 (query)                    10_714.72     8_487.53    19_202.25       0.1083     0.008250        11.93
IVF-RaBitQ-nl223-np21-rf20 (query)                    10_714.72     8_596.16    19_310.88       0.1922     0.005920        11.93
IVF-RaBitQ-nl223 (self)                               10_714.72    28_567.65    39_282.37       0.5665     0.001065        11.93
IVF-RaBitQ-nl316-np15-rf0 (query)                     10_753.92     6_126.49    16_880.42       0.0289          NaN        12.30
IVF-RaBitQ-nl316-np17-rf0 (query)                     10_753.92     6_941.56    17_695.48       0.0287          NaN        12.30
IVF-RaBitQ-nl316-np25-rf0 (query)                     10_753.92    10_117.45    20_871.38       0.0284          NaN        12.30
IVF-RaBitQ-nl316-np15-rf10 (query)                    10_753.92     6_064.31    16_818.23       0.1208     0.007211        12.30
IVF-RaBitQ-nl316-np15-rf20 (query)                    10_753.92     6_188.63    16_942.56       0.2187     0.004982        12.30
IVF-RaBitQ-nl316-np17-rf10 (query)                    10_753.92     6_863.28    17_617.20       0.1191     0.007354        12.30
IVF-RaBitQ-nl316-np17-rf20 (query)                    10_753.92     6_963.61    17_717.53       0.2145     0.005105        12.30
IVF-RaBitQ-nl316-np25-rf10 (query)                    10_753.92     9_890.18    20_644.10       0.1148     0.007773        12.30
IVF-RaBitQ-nl316-np25-rf20 (query)                    10_753.92    10_026.55    20_780.48       0.2047     0.005487        12.30
IVF-RaBitQ-nl316 (self)                               10_753.92    33_358.06    44_111.98       0.6266     0.000824        12.30
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

Overall, this is a fantastic binary index that massively compresses the data,
while still allowing for great Recalls. If you need to compress your data
and reduce memory fingerprint, please, use RaBitQ!

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
*Last update: 2026/04/05 with version **0.2.11***
