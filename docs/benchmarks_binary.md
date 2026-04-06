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
benchmarking).

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
Exhaustive (query)                                        10.47     4_583.11     4_593.58       1.0000     0.000000        48.83
Exhaustive (self)                                         10.47    15_197.76    15_208.23       1.0000     0.000000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_588.05       261.18     2_849.24       0.1028          NaN         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_588.05       367.58     2_955.63       0.3108    43.300474         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_588.05       473.69     3_061.75       0.4592    25.328172         1.78
ExhaustiveBinary-256-random (self)                     2_588.05     1_216.42     3_804.48       0.3108    42.875985         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_917.06       265.95     3_183.01       0.2435          NaN         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_917.06       371.52     3_288.58       0.6570    11.967453         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_917.06       504.30     3_421.36       0.7864     5.976325         1.78
ExhaustiveBinary-256-pca (self)                        2_917.06     1_243.50     4_160.56       0.6576    11.860151         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_051.20       467.51     5_518.71       0.1292          NaN         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_051.20       572.91     5_624.12       0.3988    30.860166         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_051.20       700.38     5_751.58       0.5650    16.596919         3.55
ExhaustiveBinary-512-random (self)                     5_051.20     1_917.95     6_969.15       0.3982    30.569280         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_504.28       479.64     5_983.93       0.2613          NaN         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_504.28       587.73     6_092.02       0.7226     8.224501         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_504.28       729.47     6_233.75       0.8654     3.067534         3.55
ExhaustiveBinary-512-pca (self)                        5_504.28     1_981.87     7_486.15       0.7232     8.171647         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_040.26       802.67    10_842.93       0.1731          NaN         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_040.26       914.95    10_955.21       0.5267    18.264305         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_040.26     1_040.87    11_081.13       0.6959     8.715766         7.10
ExhaustiveBinary-1024-random (self)                   10_040.26     3_020.96    13_061.22       0.5248    18.184021         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_721.68       815.28    11_536.97       0.2885          NaN         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_721.68       930.87    11_652.55       0.7675     5.999434         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_721.68     1_066.86    11_788.54       0.8973     2.018226         7.10
ExhaustiveBinary-1024-pca (self)                      10_721.68     3_194.98    13_916.67       0.7680     5.949327         7.10
ExhaustiveBinary-256-signed_no_rr (query)              2_587.90       269.38     2_857.28       0.1028          NaN         1.78
ExhaustiveBinary-256-signed-rf10 (query)               2_587.90       367.03     2_954.93       0.3108    43.300474         1.78
ExhaustiveBinary-256-signed-rf20 (query)               2_587.90       479.31     3_067.21       0.4592    25.328172         1.78
ExhaustiveBinary-256-signed (self)                     2_587.90     1_219.16     3_807.06       0.3108    42.875985         1.78
IVF-Binary-256-nl158-np7-rf0-random (query)            4_119.12       117.86     4_236.97       0.1083          NaN         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           4_119.12       120.30     4_239.41       0.1058          NaN         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           4_119.12       130.80     4_249.91       0.1039          NaN         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           4_119.12       184.45     4_303.56       0.3319    38.821454         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           4_119.12       239.60     4_358.71       0.4851    22.106414         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          4_119.12       193.55     4_312.67       0.3204    41.267767         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          4_119.12       243.37     4_362.48       0.4705    23.851119         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          4_119.12       205.84     4_324.96       0.3134    42.767082         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          4_119.12       271.82     4_390.94       0.4620    24.928273         1.93
IVF-Binary-256-nl158-random (self)                     4_119.12       573.69     4_692.80       0.3197    40.913889         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           3_080.72       123.52     3_204.24       0.1086          NaN         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           3_080.72       129.50     3_210.22       0.1068          NaN         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           3_080.72       131.79     3_212.51       0.1038          NaN         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          3_080.72       189.21     3_269.92       0.3245    41.356252         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          3_080.72       232.31     3_313.02       0.4754    24.091784         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          3_080.72       191.27     3_271.98       0.3186    42.214971         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          3_080.72       250.80     3_331.52       0.4684    24.649239         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          3_080.72       202.92     3_283.64       0.3121    43.028382         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          3_080.72       255.57     3_336.29       0.4615    25.146775         2.00
IVF-Binary-256-nl223-random (self)                     3_080.72       576.50     3_657.21       0.3187    41.815356         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           3_264.76       128.77     3_393.53       0.1062          NaN         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           3_264.76       131.88     3_396.64       0.1052          NaN         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           3_264.76       136.76     3_401.52       0.1036          NaN         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          3_264.76       197.69     3_462.45       0.3216    41.840948         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          3_264.76       254.50     3_519.26       0.4723    24.352976         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          3_264.76       189.94     3_454.70       0.3186    42.204456         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          3_264.76       256.76     3_521.52       0.4682    24.633897         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          3_264.76       210.98     3_475.74       0.3137    42.843288         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          3_264.76       271.58     3_536.34       0.4628    25.014669         2.09
IVF-Binary-256-nl316-random (self)                     3_264.76       587.07     3_851.83       0.3184    41.790454         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               4_558.33       119.81     4_678.14       0.2480          NaN         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              4_558.33       139.34     4_697.67       0.2466          NaN         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              4_558.33       137.00     4_695.32       0.2458          NaN         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              4_558.33       200.29     4_758.61       0.6898    10.239882         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              4_558.33       256.61     4_814.94       0.8312     4.750092         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             4_558.33       196.38     4_754.70       0.6913     9.937400         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             4_558.33       271.14     4_829.47       0.8392     4.008755         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             4_558.33       241.06     4_799.38       0.6889    10.134793         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             4_558.33       287.43     4_845.75       0.8364     4.129932         1.93
IVF-Binary-256-nl158-pca (self)                        4_558.33       620.95     5_179.27       0.6921     9.838518         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              3_381.42       123.99     3_505.41       0.2469          NaN         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              3_381.42       128.66     3_510.08       0.2462          NaN         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              3_381.42       142.44     3_523.86       0.2457          NaN         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             3_381.42       199.69     3_581.12       0.6922    10.008007         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             3_381.42       260.46     3_641.89       0.8410     3.993291         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             3_381.42       197.16     3_578.59       0.6905    10.075332         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             3_381.42       263.14     3_644.56       0.8391     4.052990         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             3_381.42       218.13     3_599.55       0.6887    10.183600         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             3_381.42       299.89     3_681.31       0.8361     4.151937         2.00
IVF-Binary-256-nl223-pca (self)                        3_381.42       614.22     3_995.64       0.6910     9.997322         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              3_608.03       133.54     3_741.57       0.2469          NaN         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              3_608.03       135.76     3_743.79       0.2464          NaN         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              3_608.03       141.54     3_749.57       0.2458          NaN         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             3_608.03       209.24     3_817.27       0.6928     9.992318         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             3_608.03       262.96     3_870.99       0.8413     3.983978         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             3_608.03       203.53     3_811.56       0.6915    10.045751         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             3_608.03       271.38     3_879.41       0.8404     4.014304         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             3_608.03       219.40     3_827.43       0.6897    10.130937         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             3_608.03       277.53     3_885.56       0.8376     4.108078         2.09
IVF-Binary-256-nl316-pca (self)                        3_608.03       634.96     4_242.99       0.6922     9.942858         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_650.88       218.67     6_869.55       0.1337          NaN         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_650.88       228.25     6_879.13       0.1314          NaN         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_650.88       242.51     6_893.39       0.1300          NaN         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_650.88       285.98     6_936.86       0.4144    28.473873         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_650.88       346.60     6_997.48       0.5795    15.215400         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_650.88       296.61     6_947.49       0.4062    29.754981         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_650.88       358.42     7_009.30       0.5725    15.854750         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_650.88       317.48     6_968.36       0.4011    30.534510         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_650.88       373.61     7_024.49       0.5670    16.375622         3.71
IVF-Binary-512-nl158-random (self)                     6_650.88       939.09     7_589.98       0.4047    29.534453         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_597.77       221.85     5_819.62       0.1320          NaN         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_597.77       219.46     5_817.23       0.1307          NaN         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_597.77       234.69     5_832.46       0.1293          NaN         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_597.77       312.92     5_910.69       0.4075    29.996061         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_597.77       342.57     5_940.34       0.5735    16.072763         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_597.77       290.34     5_888.11       0.4035    30.405525         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_597.77       356.70     5_954.47       0.5688    16.341726         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_597.77       306.65     5_904.42       0.4000    30.729230         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_597.77       368.00     5_965.77       0.5653    16.533698         3.77
IVF-Binary-512-nl223-random (self)                     5_597.77       915.18     6_512.95       0.4029    30.118263         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           5_787.96       226.56     6_014.51       0.1312          NaN         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           5_787.96       231.31     6_019.26       0.1305          NaN         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           5_787.96       239.88     6_027.83       0.1296          NaN         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          5_787.96       289.03     6_076.99       0.4064    30.161109         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          5_787.96       356.28     6_144.23       0.5728    16.131551         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          5_787.96       300.54     6_088.49       0.4039    30.386090         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          5_787.96       348.80     6_136.75       0.5701    16.277028         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          5_787.96       309.84     6_097.79       0.4011    30.652841         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          5_787.96       377.57     6_165.52       0.5663    16.463769         3.86
IVF-Binary-512-nl316-random (self)                     5_787.96       949.46     6_737.42       0.4031    30.098181         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               7_100.06       218.99     7_319.06       0.2630          NaN         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              7_100.06       235.53     7_335.59       0.2623          NaN         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              7_100.06       246.75     7_346.81       0.2616          NaN         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              7_100.06       298.93     7_399.00       0.7182     8.713693         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              7_100.06       350.08     7_450.14       0.8521     4.076349         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             7_100.06       316.34     7_416.41       0.7236     8.138391         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             7_100.06       373.69     7_473.76       0.8650     3.108818         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             7_100.06       325.48     7_425.54       0.7230     8.193878         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             7_100.06       401.14     7_501.21       0.8657     3.057313         3.71
IVF-Binary-512-nl158-pca (self)                        7_100.06       989.60     8_089.66       0.7241     8.102071         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              6_023.42       225.21     6_248.62       0.2624          NaN         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              6_023.42       232.00     6_255.41       0.2618          NaN         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              6_023.42       252.12     6_275.54       0.2615          NaN         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             6_023.42       310.12     6_333.54       0.7241     8.168147         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             6_023.42       351.40     6_374.82       0.8663     3.043275         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             6_023.42       314.09     6_337.50       0.7231     8.196721         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             6_023.42       397.66     6_421.08       0.8657     3.051725         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             6_023.42       333.07     6_356.49       0.7228     8.210064         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             6_023.42       395.55     6_418.96       0.8654     3.061065         3.77
IVF-Binary-512-nl223-pca (self)                        6_023.42       962.07     6_985.49       0.7234     8.153282         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              6_219.89       229.60     6_449.49       0.2623          NaN         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              6_219.89       239.51     6_459.40       0.2619          NaN         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              6_219.89       252.29     6_472.17       0.2616          NaN         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             6_219.89       314.16     6_534.05       0.7242     8.166510         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             6_219.89       362.66     6_582.54       0.8661     3.046978         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             6_219.89       318.54     6_538.43       0.7234     8.193121         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             6_219.89       371.70     6_591.59       0.8659     3.053080         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             6_219.89       319.32     6_539.21       0.7227     8.211498         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             6_219.89       391.87     6_611.76       0.8654     3.060741         3.86
IVF-Binary-512-nl316-pca (self)                        6_219.89       976.97     7_196.86       0.7239     8.139230         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_646.92       414.71    12_061.63       0.1770          NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_646.92       423.26    12_070.18       0.1752          NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_646.92       442.66    12_089.59       0.1740          NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_646.92       487.13    12_134.06       0.5317    17.771500         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_646.92       550.78    12_197.70       0.6953     8.882184         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_646.92       496.74    12_143.66       0.5304    17.856724         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_646.92       584.29    12_231.22       0.6984     8.545597         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_646.92       515.47    12_162.39       0.5276    18.145809         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_646.92       597.29    12_244.22       0.6966     8.669237         7.26
IVF-Binary-1024-nl158-random (self)                   11_646.92     1_631.08    13_278.00       0.5284    17.804914         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_506.00       417.20    10_923.19       0.1759          NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_506.00       423.05    10_929.05       0.1749          NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_506.00       436.99    10_942.98       0.1739          NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_506.00       490.76    10_996.75       0.5316    17.934552         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_506.00       543.54    11_049.54       0.7012     8.517556         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_506.00       496.25    11_002.24       0.5290    18.107284         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_506.00       558.46    11_064.45       0.6981     8.633901         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_506.00       536.31    11_042.30       0.5268    18.236031         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_506.00       591.70    11_097.70       0.6964     8.692795         7.32
IVF-Binary-1024-nl223-random (self)                   10_506.00     1_606.86    12_112.86       0.5274    18.012960         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         10_679.08       417.79    11_096.86       0.1753          NaN         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)         10_679.08       428.66    11_107.74       0.1748          NaN         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)         10_679.08       448.60    11_127.68       0.1741          NaN         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)        10_679.08       525.01    11_204.08       0.5315    17.960579         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)        10_679.08       572.18    11_251.25       0.7010     8.551004         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)        10_679.08       501.77    11_180.85       0.5296    18.070954         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)        10_679.08       559.07    11_238.14       0.6987     8.627558         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)        10_679.08       512.68    11_191.75       0.5276    18.180669         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)        10_679.08       583.89    11_262.97       0.6963     8.701409         7.41
IVF-Binary-1024-nl316-random (self)                   10_679.08     1_588.27    12_267.35       0.5281    17.982320         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)             12_221.75       430.40    12_652.14       0.2900          NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            12_221.75       440.14    12_661.89       0.2897          NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            12_221.75       457.71    12_679.45       0.2889          NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            12_221.75       497.05    12_718.79       0.7586     6.829705         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            12_221.75       571.56    12_793.31       0.8788     3.280646         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           12_221.75       518.09    12_739.83       0.7677     6.000876         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           12_221.75       591.83    12_813.58       0.8958     2.117318         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           12_221.75       544.08    12_765.82       0.7679     5.991751         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           12_221.75       612.73    12_834.47       0.8974     2.022817         7.26
IVF-Binary-1024-nl158-pca (self)                      12_221.75     1_690.14    13_911.88       0.7678     5.964602         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            11_154.42       431.73    11_586.15       0.2897          NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            11_154.42       443.12    11_597.53       0.2892          NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            11_154.42       448.39    11_602.81       0.2888          NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           11_154.42       511.72    11_666.13       0.7686     5.985640         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           11_154.42       564.82    11_719.24       0.8979     2.012913         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           11_154.42       518.44    11_672.85       0.7678     6.005988         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           11_154.42       601.06    11_755.48       0.8976     2.015277         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           11_154.42       531.55    11_685.97       0.7675     6.017011         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           11_154.42       615.55    11_769.97       0.8973     2.019636         7.32
IVF-Binary-1024-nl223-pca (self)                      11_154.42     1_650.57    12_804.98       0.7684     5.933963         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            11_372.39       429.09    11_801.48       0.2896          NaN         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            11_372.39       451.00    11_823.39       0.2893          NaN         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            11_372.39       447.55    11_819.94       0.2889          NaN         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           11_372.39       520.01    11_892.41       0.7684     5.985534         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           11_372.39       571.71    11_944.10       0.8980     2.010364         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           11_372.39       525.74    11_898.13       0.7680     5.997559         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           11_372.39       578.00    11_950.39       0.8979     2.010035         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           11_372.39       536.30    11_908.69       0.7675     6.011111         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           11_372.39       611.53    11_983.92       0.8976     2.014855         7.42
IVF-Binary-1024-nl316-pca (self)                      11_372.39     1_667.99    13_040.38       0.7687     5.924525         7.42
IVF-Binary-256-nl158-np7-rf0-signed (query)            4_126.85       119.12     4_245.96       0.1083          NaN         1.93
IVF-Binary-256-nl158-np12-rf0-signed (query)           4_126.85       123.82     4_250.67       0.1058          NaN         1.93
IVF-Binary-256-nl158-np17-rf0-signed (query)           4_126.85       131.12     4_257.97       0.1039          NaN         1.93
IVF-Binary-256-nl158-np7-rf10-signed (query)           4_126.85       182.88     4_309.72       0.3319    38.821454         1.93
IVF-Binary-256-nl158-np7-rf20-signed (query)           4_126.85       231.78     4_358.63       0.4851    22.106414         1.93
IVF-Binary-256-nl158-np12-rf10-signed (query)          4_126.85       199.67     4_326.52       0.3204    41.267767         1.93
IVF-Binary-256-nl158-np12-rf20-signed (query)          4_126.85       258.00     4_384.85       0.4705    23.851119         1.93
IVF-Binary-256-nl158-np17-rf10-signed (query)          4_126.85       205.13     4_331.98       0.3134    42.767082         1.93
IVF-Binary-256-nl158-np17-rf20-signed (query)          4_126.85       277.49     4_404.33       0.4620    24.928273         1.93
IVF-Binary-256-nl158-signed (self)                     4_126.85       582.90     4_709.75       0.3197    40.913889         1.93
IVF-Binary-256-nl223-np11-rf0-signed (query)           3_030.23       119.84     3_150.06       0.1086          NaN         2.00
IVF-Binary-256-nl223-np14-rf0-signed (query)           3_030.23       121.66     3_151.88       0.1068          NaN         2.00
IVF-Binary-256-nl223-np21-rf0-signed (query)           3_030.23       130.70     3_160.93       0.1038          NaN         2.00
IVF-Binary-256-nl223-np11-rf10-signed (query)          3_030.23       188.85     3_219.07       0.3245    41.356252         2.00
IVF-Binary-256-nl223-np11-rf20-signed (query)          3_030.23       244.04     3_274.27       0.4754    24.091784         2.00
IVF-Binary-256-nl223-np14-rf10-signed (query)          3_030.23       252.74     3_282.96       0.3186    42.214971         2.00
IVF-Binary-256-nl223-np14-rf20-signed (query)          3_030.23       298.14     3_328.37       0.4684    24.649239         2.00
IVF-Binary-256-nl223-np21-rf10-signed (query)          3_030.23       218.81     3_249.03       0.3121    43.028382         2.00
IVF-Binary-256-nl223-np21-rf20-signed (query)          3_030.23       268.38     3_298.61       0.4615    25.146775         2.00
IVF-Binary-256-nl223-signed (self)                     3_030.23       590.68     3_620.90       0.3187    41.815356         2.00
IVF-Binary-256-nl316-np15-rf0-signed (query)           3_258.17       129.63     3_387.80       0.1062          NaN         2.09
IVF-Binary-256-nl316-np17-rf0-signed (query)           3_258.17       129.71     3_387.88       0.1052          NaN         2.09
IVF-Binary-256-nl316-np25-rf0-signed (query)           3_258.17       135.12     3_393.29       0.1036          NaN         2.09
IVF-Binary-256-nl316-np15-rf10-signed (query)          3_258.17       215.79     3_473.96       0.3216    41.840948         2.09
IVF-Binary-256-nl316-np15-rf20-signed (query)          3_258.17       252.40     3_510.57       0.4723    24.352976         2.09
IVF-Binary-256-nl316-np17-rf10-signed (query)          3_258.17       190.12     3_448.29       0.3186    42.204456         2.09
IVF-Binary-256-nl316-np17-rf20-signed (query)          3_258.17       247.02     3_505.18       0.4682    24.633897         2.09
IVF-Binary-256-nl316-np25-rf10-signed (query)          3_258.17       209.99     3_468.15       0.3137    42.843288         2.09
IVF-Binary-256-nl316-np25-rf20-signed (query)          3_258.17       266.51     3_524.68       0.4628    25.014669         2.09
IVF-Binary-256-nl316-signed (self)                     3_258.17       602.24     3_860.41       0.3184    41.790454         2.09
--------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:binary:euclidean:correlated:512:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 1024 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:binary:euclidean:correlated:1024:50000 -->
</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:binary:euclidean:lowrank:256:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:binary:euclidean:lowrank:512:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 1024 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:binary:euclidean:lowrank:1024:50000 -->
</code></pre>
</details>

#### Quantisation (stress) data

<details>
<summary><b>Quantisation stress data - 256 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:binary:euclidean:quantisation:256:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Quantisation stress data - 512 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:binary:euclidean:quantisation:512:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Quantisation stress data - 1024 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:binary:euclidean:quantisation:1024:50000 -->
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
<!-- BENCH:rabitq:euclidean:correlated:256:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:rabitq:euclidean:correlated:512:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 1024 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:rabitq:euclidean:correlated:1024:50000 -->
</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:rabitq:euclidean:lowrank:256:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:rabitq:euclidean:lowrank:512:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 1024 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:rabitq:euclidean:lowrank:1024:50000 -->
</code></pre>
</details>

#### Quantisation (stress) data

<details>
<summary><b>Quantisation stress data - 256 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:rabitq:euclidean:quantisation:256:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Quantisation stress data - 512 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:rabitq:euclidean:quantisation:512:50000 -->
</code></pre>
</details>

---

<details>
<summary><b>Quantisation stress data - 1024 dimensions</b>:</summary>
</br>
<pre><code>
<!-- BENCH:rabitq:euclidean:quantisation:1024:50000 -->
</code></pre>
</details>

Overall, this is a fantastic binary index that massively compresses the data,
while still allowing for great Recalls. If you need to compress your data
and reduce memory fingerprint, please, use RaBitQ!

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
*Last update: 2026/04/05 with version **0.2.11***
