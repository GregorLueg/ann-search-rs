## Binarised indices benchmarks and parameter

Binarised indices compress the data stored in the index structure itself via
very aggressive quantisation to (basically) only bits. This has two impacts:

1. Drastic reduction in memory fingerprint of the index itself.
2. Increased query speed in most cases as the bit-wise operations are very
fast on modern CPUs.
3. However, when not using any re-ranking of the top candidates, dramatically
lower recall (less so for RaBitQ -- an excellent way of compressing vectors --
and pending the data type TurboQuant).

The benchmarks below show scenarios with and without re-ranking. For the simple
binary versions use:

```bash
cargo run --example gridsearch_binary --release --features binary -- --dim 512 --n-samples 50000 --data embedding
```

For RaBitQ:

```bash
cargo run --example gridsearch_rabitq --release --features binary -- --dim 512 --n-samples 50000 --data embedding
```

For TurboQuantisation

```bash
cargo run --example gridsearch_tq --release --features binary -- --dim 512 --n-samples 50000 --data embedding
```

Similar to the other benchmarks, index building, query against 10% slightly
different data based on the trainings data and full kNN generation is being
benchmarked. Index size in memory is also provided. Compared to other
benchmarks, we will use the `"correlated"`, `"lowrank"` and `"embedding"`
with higher dimensionality, but reduced samples (for the sake of fast'ish
benchmarking).

## Table of Contents

- [Binarisation](#binary-ivf-and-exhaustive)
- [RaBitQ](#rabitq-ivf-and-exhaustive)
- [TurboQuant](#turboquant-ivf-and-exhaustive)

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
  degrades dramatically. In the IVF version, the signed version is based on
  the sign of the residual of the centroid to increase performance.

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
  uses the sign of the respective embedding dimensions (or residual for the
  IVF). In this case, `n_bits` is set automatically to `n_dim`. Signed only
  really makes sense if you have a lot of dimensions; otherwise, the performance
  is not great (at all).
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
===================================================================================================================================
Benchmark: 50k samples, 256D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.72     3_962.45     3_995.17       1.0000          1.0000        48.83
Exhaustive (self)                                         32.72    13_221.25    13_253.97       1.0000          1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_714.32       278.84     2_993.16       0.0377             NaN         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_714.32       372.93     3_087.24       0.1695          1.1187         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_714.32       475.50     3_189.82       0.2766          1.0759         1.78
ExhaustiveBinary-256-random (self)                     2_714.32     1_240.36     3_954.68       0.1761          1.1125         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_750.13       281.16     3_031.30       0.1873             NaN         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_750.13       394.56     3_144.69       0.5340          1.0265         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_750.13       508.18     3_258.32       0.6684          1.0147         1.78
ExhaustiveBinary-256-pca (self)                        2_750.13     1_296.94     4_047.08       0.5319          1.0270         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_259.97       465.12     5_725.09       0.0697             NaN         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_259.97       562.81     5_822.78       0.2156          1.0884         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_259.97       664.20     5_924.17       0.3321          1.0553         3.55
ExhaustiveBinary-512-random (self)                     5_259.97     1_868.37     7_128.34       0.2185          1.0859         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_301.34       466.01     5_767.35       0.2013             NaN         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_301.34       573.28     5_874.63       0.6344          1.0175         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_301.34       679.70     5_981.04       0.8009          1.0073         3.55
ExhaustiveBinary-512-pca (self)                        5_301.34     1_908.35     7_209.70       0.6350          1.0176         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_418.48       776.17    11_194.65       0.0965             NaN         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_418.48       880.81    11_299.29       0.2686          1.0681         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_418.48       993.03    11_411.51       0.4004          1.0419         7.10
ExhaustiveBinary-1024-random (self)                   10_418.48     2_926.88    13_345.36       0.2691          1.0684         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_469.61       770.93    11_240.55       0.2080             NaN         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_469.61       887.66    11_357.28       0.6484          1.0164         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_469.61     1_003.54    11_473.15       0.8112          1.0067         7.10
ExhaustiveBinary-1024-pca (self)                      10_469.61     2_944.99    13_414.60       0.6483          1.0165         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   66.25       477.61       543.86       0.0290             NaN         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    66.25       507.60       573.85       0.1603          1.1277         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    66.25       835.33       901.58       0.2712          1.0802         1.53
ExhaustiveBinary-256-sign (self)                          66.25     1_668.11     1_734.36       0.1661          1.1226         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)            4_108.59       121.05     4_229.64       0.0689             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           4_108.59       130.03     4_238.62       0.0687             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           4_108.59       132.61     4_241.21       0.0687             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           4_108.59       175.47     4_284.06       0.2600          1.0745         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           4_108.59       221.74     4_330.33       0.3790          1.0478         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          4_108.59       180.62     4_289.21       0.2557          1.0763         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          4_108.59       234.68     4_343.27       0.3698          1.0498         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          4_108.59       187.53     4_296.12       0.2557          1.0763         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          4_108.59       235.88     4_344.47       0.3698          1.0498         1.93
IVF-Binary-256-nl158-random (self)                     4_108.59       536.14     4_644.73       0.2624          1.0706         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           3_197.30       128.53     3_325.83       0.0774             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           3_197.30       129.38     3_326.68       0.0773             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           3_197.30       135.45     3_332.75       0.0773             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          3_197.30       181.80     3_379.10       0.2678          1.0725         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          3_197.30       228.51     3_425.82       0.3845          1.0474         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          3_197.30       182.92     3_380.22       0.2663          1.0731         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          3_197.30       230.94     3_428.24       0.3805          1.0482         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          3_197.30       200.72     3_398.02       0.2649          1.0737         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          3_197.30       237.61     3_434.91       0.3778          1.0488         2.00
IVF-Binary-256-nl223-random (self)                     3_197.30       549.14     3_746.44       0.2732          1.0676         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           3_368.74       139.96     3_508.69       0.0840             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           3_368.74       143.17     3_511.90       0.0839             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           3_368.74       145.25     3_513.99       0.0838             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          3_368.74       200.61     3_569.35       0.2722          1.0714         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          3_368.74       240.37     3_609.10       0.3861          1.0472         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          3_368.74       193.67     3_562.41       0.2711          1.0719         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          3_368.74       255.93     3_624.66       0.3835          1.0477         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          3_368.74       201.43     3_570.16       0.2693          1.0726         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          3_368.74       250.89     3_619.62       0.3798          1.0486         2.09
IVF-Binary-256-nl316-random (self)                     3_368.74       582.92     3_951.66       0.2779          1.0664         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               4_212.08       121.38     4_333.46       0.1991             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              4_212.08       127.30     4_339.38       0.1976             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              4_212.08       133.76     4_345.84       0.1971             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              4_212.08       196.75     4_408.82       0.6303          1.0178         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              4_212.08       230.44     4_442.52       0.7969          1.0074         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             4_212.08       188.72     4_400.80       0.6202          1.0186         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             4_212.08       240.84     4_452.92       0.7843          1.0080         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             4_212.08       207.99     4_420.06       0.6135          1.0191         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             4_212.08       249.08     4_461.15       0.7752          1.0085         1.93
IVF-Binary-256-nl158-pca (self)                        4_212.08       575.41     4_787.49       0.6195          1.0188         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              3_274.04       129.06     3_403.09       0.1989             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              3_274.04       133.49     3_407.53       0.1979             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              3_274.04       137.17     3_411.21       0.1971             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             3_274.04       188.82     3_462.85       0.6283          1.0179         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             3_274.04       240.11     3_514.15       0.7948          1.0075         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             3_274.04       190.79     3_464.83       0.6225          1.0184         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             3_274.04       242.93     3_516.97       0.7875          1.0078         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             3_274.04       200.51     3_474.55       0.6154          1.0189         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             3_274.04       250.65     3_524.69       0.7781          1.0083         2.00
IVF-Binary-256-nl223-pca (self)                        3_274.04       585.43     3_859.47       0.6218          1.0186         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              3_434.10       136.50     3_570.60       0.1997             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              3_434.10       137.78     3_571.87       0.1991             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              3_434.10       145.42     3_579.51       0.1982             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             3_434.10       199.75     3_633.85       0.6296          1.0178         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             3_434.10       247.63     3_681.72       0.7961          1.0074         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             3_434.10       197.68     3_631.78       0.6260          1.0181         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             3_434.10       247.54     3_681.64       0.7918          1.0077         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             3_434.10       203.69     3_637.79       0.6186          1.0187         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             3_434.10       255.14     3_689.24       0.7824          1.0081         2.09
IVF-Binary-256-nl316-pca (self)                        3_434.10       603.18     4_037.27       0.6255          1.0183         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_771.88       219.55     6_991.43       0.0876             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_771.88       228.69     7_000.57       0.0869             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_771.88       238.00     7_009.88       0.0869             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_771.88       279.56     7_051.44       0.2614          1.0705         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_771.88       326.50     7_098.38       0.3839          1.0449         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_771.88       289.06     7_060.94       0.2557          1.0727         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_771.88       339.29     7_111.17       0.3737          1.0468         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_771.88       295.49     7_067.37       0.2557          1.0727         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_771.88       348.34     7_120.22       0.3737          1.0468         3.71
IVF-Binary-512-nl158-random (self)                     6_771.88       903.90     7_675.78       0.2577          1.0711         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_842.68       226.05     6_068.72       0.0908             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_842.68       230.44     6_073.12       0.0906             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_842.68       240.91     6_083.58       0.0903             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_842.68       289.13     6_131.81       0.2650          1.0697         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_842.68       336.47     6_179.14       0.3865          1.0445         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_842.68       290.32     6_132.99       0.2621          1.0708         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_842.68       338.45     6_181.12       0.3814          1.0454         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_842.68       299.72     6_142.40       0.2599          1.0718         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_842.68       354.53     6_197.21       0.3776          1.0463         3.77
IVF-Binary-512-nl223-random (self)                     5_842.68       920.32     6_763.00       0.2638          1.0693         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           6_010.27       233.99     6_244.26       0.0930             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           6_010.27       237.39     6_247.66       0.0928             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           6_010.27       246.07     6_256.34       0.0925             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          6_010.27       296.92     6_307.19       0.2643          1.0698         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          6_010.27       342.44     6_352.71       0.3843          1.0447         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          6_010.27       294.50     6_304.77       0.2625          1.0705         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          6_010.27       345.19     6_355.46       0.3812          1.0454         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          6_010.27       306.92     6_317.19       0.2598          1.0717         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          6_010.27       353.34     6_363.61       0.3767          1.0464         3.86
IVF-Binary-512-nl316-random (self)                     6_010.27       930.39     6_940.66       0.2645          1.0690         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               6_813.84       219.69     7_033.53       0.2033             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              6_813.84       229.57     7_043.40       0.2023             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              6_813.84       240.90     7_054.74       0.2023             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              6_813.84       285.47     7_099.30       0.6398          1.0170         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              6_813.84       330.28     7_144.12       0.8065          1.0070         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             6_813.84       289.40     7_103.24       0.6359          1.0174         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             6_813.84       340.31     7_154.15       0.8019          1.0072         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             6_813.84       298.75     7_112.59       0.6359          1.0174         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             6_813.84       371.42     7_185.25       0.8019          1.0072         3.71
IVF-Binary-512-nl158-pca (self)                        6_813.84       914.43     7_728.27       0.6363          1.0175         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              5_893.27       227.64     6_120.90       0.2037             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              5_893.27       239.82     6_133.09       0.2030             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              5_893.27       239.87     6_133.14       0.2027             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             5_893.27       289.56     6_182.83       0.6399          1.0170         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             5_893.27       337.52     6_230.78       0.8072          1.0069         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             5_893.27       289.87     6_183.14       0.6376          1.0172         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             5_893.27       359.47     6_252.74       0.8040          1.0071         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             5_893.27       298.73     6_192.00       0.6360          1.0174         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             5_893.27       351.11     6_244.37       0.8021          1.0072         3.77
IVF-Binary-512-nl223-pca (self)                        5_893.27       913.83     6_807.09       0.6378          1.0174         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              6_093.74       235.09     6_328.83       0.2040             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              6_093.74       235.98     6_329.72       0.2036             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              6_093.74       243.99     6_337.73       0.2031             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             6_093.74       321.26     6_415.00       0.6401          1.0170         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             6_093.74       348.86     6_442.60       0.8068          1.0070         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             6_093.74       298.84     6_392.58       0.6388          1.0171         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             6_093.74       345.89     6_439.62       0.8050          1.0071         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             6_093.74       303.05     6_396.79       0.6367          1.0173         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             6_093.74       824.02     6_917.75       0.8025          1.0072         3.86
IVF-Binary-512-nl316-pca (self)                        6_093.74     1_346.32     7_440.06       0.6388          1.0173         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          12_168.83       413.33    12_582.16       0.1032             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         12_168.83       422.52    12_591.35       0.1026             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         12_168.83       436.10    12_604.93       0.1026             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         12_168.83       472.16    12_640.99       0.2930          1.0615         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         12_168.83       517.83    12_686.66       0.4304          1.0376         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        12_168.83       534.11    12_702.94       0.2871          1.0632         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        12_168.83       607.83    12_776.66       0.4205          1.0391         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        12_168.83       574.80    12_743.63       0.2871          1.0632         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        12_168.83       588.25    12_757.09       0.4205          1.0391         7.26
IVF-Binary-1024-nl158-random (self)                   12_168.83     1_565.41    13_734.24       0.2876          1.0635         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_922.44       414.32    11_336.77       0.1042             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_922.44       423.09    11_345.54       0.1038             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_922.44       433.56    11_356.01       0.1035             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_922.44       475.56    11_398.01       0.2945          1.0612         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_922.44       531.14    11_453.59       0.4321          1.0374         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_922.44       478.40    11_400.85       0.2914          1.0621         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_922.44       530.95    11_453.39       0.4268          1.0382         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_922.44       494.67    11_417.11       0.2891          1.0628         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_922.44       544.89    11_467.34       0.4229          1.0388         7.32
IVF-Binary-1024-nl223-random (self)                   10_922.44     1_547.74    12_470.19       0.2915          1.0624         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         11_069.26       421.85    11_491.11       0.1043             NaN         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)         11_069.26       426.57    11_495.83       0.1041             NaN         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)         11_069.26       450.17    11_519.44       0.1037             NaN         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)        11_069.26       482.63    11_551.90       0.2933          1.0613         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)        11_069.26       531.21    11_600.47       0.4297          1.0376         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)        11_069.26       483.03    11_552.29       0.2914          1.0619         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)        11_069.26       534.83    11_604.10       0.4262          1.0382         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)        11_069.26       498.07    11_567.33       0.2888          1.0628         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)        11_069.26       556.28    11_625.55       0.4213          1.0390         7.42
IVF-Binary-1024-nl316-random (self)                   11_069.26     1_562.90    12_632.17       0.2917          1.0622         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)             11_864.09       407.40    12_271.49       0.2102             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            11_864.09       423.16    12_287.25       0.2094             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            11_864.09       433.77    12_297.86       0.2094             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            11_864.09       466.98    12_331.07       0.6533          1.0159         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            11_864.09       513.62    12_377.71       0.8163          1.0065         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           11_864.09       478.16    12_342.25       0.6497          1.0163         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           11_864.09       527.39    12_391.48       0.8122          1.0067         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           11_864.09       501.20    12_365.29       0.6497          1.0163         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           11_864.09       547.64    12_411.73       0.8122          1.0067         7.26
IVF-Binary-1024-nl158-pca (self)                      11_864.09     1_549.22    13_413.31       0.6497          1.0164         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            10_946.26       414.62    11_360.87       0.2104             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            10_946.26       420.15    11_366.41       0.2098             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            10_946.26       431.87    11_378.13       0.2094             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           10_946.26       473.14    11_419.40       0.6530          1.0159         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           10_946.26       522.42    11_468.67       0.8167          1.0065         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           10_946.26       481.85    11_428.10       0.6506          1.0162         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           10_946.26       528.72    11_474.98       0.8140          1.0066         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           10_946.26       530.18    11_476.43       0.6493          1.0163         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           10_946.26       544.10    11_490.36       0.8121          1.0067         7.32
IVF-Binary-1024-nl223-pca (self)                      10_946.26     1_534.20    12_480.46       0.6510          1.0163         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            11_156.00       428.46    11_584.46       0.2105             NaN         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            11_156.00       426.86    11_582.85       0.2101             NaN         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            11_156.00       438.76    11_594.76       0.2097             NaN         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           11_156.00       494.16    11_650.15       0.6530          1.0160         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           11_156.00       532.64    11_688.63       0.8162          1.0065         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           11_156.00       485.04    11_641.04       0.6517          1.0161         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           11_156.00       532.89    11_688.88       0.8146          1.0066         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           11_156.00       493.79    11_649.79       0.6499          1.0163         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           11_156.00       545.21    11_701.21       0.8122          1.0067         7.42
IVF-Binary-1024-nl316-pca (self)                      11_156.00     1_552.87    12_708.87       0.6519          1.0162         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)              1_433.97       293.87     1_727.84       0.1892             NaN         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)             1_433.97       322.35     1_756.32       0.1690             NaN         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)             1_433.97       344.90     1_778.87       0.1645             NaN         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)             1_433.97       325.07     1_759.04       0.5693          1.0245         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)             1_433.97       561.01     1_994.98       0.7520          1.0100         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)            1_433.97       359.44     1_793.42       0.4918          1.0331         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)            1_433.97       592.69     2_026.66       0.6582          1.0160         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)            1_433.97       366.11     1_800.08       0.4528          1.0386         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)            1_433.97       632.16     2_066.13       0.6049          1.0202         1.68
IVF-Binary-256-nl158-sign (self)                       1_433.97     1_111.96     2_545.93       0.4908          1.0335         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               538.42       314.28       852.70       0.1796             NaN         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               538.42       329.00       867.42       0.1700             NaN         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               538.42       353.78       892.19       0.1582             NaN         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              538.42       346.55       884.97       0.5356          1.0276         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              538.42       585.39     1_123.80       0.7156          1.0121         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              538.42       357.15       895.57       0.4999          1.0316         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              538.42       606.25     1_144.67       0.6716          1.0149         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              538.42       380.81       919.23       0.4453          1.0391         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              538.42       644.11     1_182.53       0.6029          1.0201         1.75
IVF-Binary-256-nl223-sign (self)                         538.42     1_138.21     1_676.63       0.4991          1.0320         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               717.45       334.36     1_051.81       0.1745             NaN         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               717.45       349.69     1_067.14       0.1679             NaN         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               717.45       368.64     1_086.09       0.1559             NaN         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              717.45       365.15     1_082.60       0.5249          1.0284         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              717.45       605.44     1_322.89       0.7080          1.0124         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              717.45       370.63     1_088.08       0.5025          1.0310         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              717.45       652.37     1_369.82       0.6814          1.0142         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              717.45       395.10     1_112.54       0.4420          1.0392         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              717.45       655.98     1_373.43       0.6043          1.0199         1.84
IVF-Binary-256-nl316-sign (self)                         717.45     1_192.04     1_909.49       0.5001          1.0315         1.84
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        69.55     9_632.18     9_701.73       1.0000          1.0000        97.66
Exhaustive (self)                                         69.55    32_713.32    32_782.87       1.0000          1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_889.76       439.17     6_328.93       0.0309             NaN         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_889.76       549.36     6_439.12       0.1453          1.0932         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_889.76       680.80     6_570.56       0.2424          1.0615         2.03
ExhaustiveBinary-256-random (self)                     5_889.76     1_787.30     7_677.06       0.1492          1.0890         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_140.58       422.71     6_563.29       0.1394             NaN         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_140.58       571.97     6_712.55       0.3908          1.0290         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_140.58       723.95     6_864.53       0.5142          1.0182         2.03
ExhaustiveBinary-256-pca (self)                        6_140.58     1_842.96     7_983.54       0.3906          1.0291         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_596.02       702.36    12_298.39       0.0645             NaN         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_596.02       826.02    12_422.04       0.1842          1.0654         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_596.02     1_003.17    12_599.19       0.2847          1.0420         4.05
ExhaustiveBinary-512-random (self)                    11_596.02     2_732.74    14_328.77       0.1862          1.0633         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_752.12       705.60    12_457.72       0.1676             NaN         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_752.12       838.63    12_590.75       0.4490          1.2486         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_752.12     1_006.22    12_758.34       0.5696          1.0152         4.05
ExhaustiveBinary-512-pca (self)                       11_752.12     2_783.43    14_535.55       0.4499          1.3208         4.05
ExhaustiveBinary-1024-random_no_rr (query)            22_970.78     1_259.75    24_230.53       0.0840             NaN         8.11
ExhaustiveBinary-1024-random-rf10 (query)             22_970.78     1_405.09    24_375.88       0.2133          1.0553         8.11
ExhaustiveBinary-1024-random-rf20 (query)             22_970.78     1_552.43    24_523.21       0.3231          1.0357         8.11
ExhaustiveBinary-1024-random (self)                   22_970.78     4_669.69    27_640.47       0.2125          1.0553         8.11
ExhaustiveBinary-1024-pca_no_rr (query)               23_165.49     1_260.45    24_425.94       0.2037             NaN         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                23_165.49     1_433.58    24_599.07       0.6282          1.0116         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                23_165.49     1_589.29    24_754.78       0.7931          1.0049         8.11
ExhaustiveBinary-1024-pca (self)                      23_165.49     4_824.96    27_990.45       0.6281          1.0116         8.11
ExhaustiveBinary-512-sign_no_rr (query)                  110.00       706.17       816.17       0.0497             NaN         3.05
ExhaustiveBinary-512-sign-rf10 (query)                   110.00       763.48       873.47       0.1718          1.0749         3.05
ExhaustiveBinary-512-sign-rf20 (query)                   110.00     1_272.81     1_382.80       0.2766          1.0469         3.05
ExhaustiveBinary-512-sign (self)                         110.00     2_518.94     2_628.94       0.1738          1.0727         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)            9_127.62       247.59     9_375.21       0.0627             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           9_127.62       255.07     9_382.69       0.0626             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           9_127.62       261.14     9_388.75       0.0626             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           9_127.62       335.77     9_463.39       0.2391          1.0535         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           9_127.62       420.97     9_548.59       0.3512          1.0345         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          9_127.62       349.93     9_477.55       0.2369          1.0541         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          9_127.62       426.68     9_554.30       0.3472          1.0350         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          9_127.62       369.21     9_496.83       0.2369          1.0541         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          9_127.62       438.42     9_566.04       0.3472          1.0350         2.34
IVF-Binary-256-nl158-random (self)                     9_127.62     1_027.24    10_154.86       0.2401          1.0503         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           6_883.93       261.18     7_145.11       0.0683             NaN         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)           6_883.93       263.99     7_147.91       0.0682             NaN         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)           6_883.93       271.29     7_155.22       0.0682             NaN         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)          6_883.93       355.15     7_239.08       0.2457          1.0514         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)          6_883.93       438.89     7_322.81       0.3549          1.0341         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)          6_883.93       350.52     7_234.45       0.2443          1.0518         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)          6_883.93       444.20     7_328.13       0.3519          1.0345         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)          6_883.93       361.34     7_245.26       0.2436          1.0521         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)          6_883.93       453.47     7_337.40       0.3505          1.0348         2.47
IVF-Binary-256-nl223-random (self)                     6_883.93     1_054.71     7_938.63       0.2481          1.0478         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)           7_148.61       291.99     7_440.60       0.0747             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           7_148.61       279.99     7_428.60       0.0747             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           7_148.61       282.65     7_431.26       0.0746             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          7_148.61       366.51     7_515.12       0.2505          1.0503         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          7_148.61       465.78     7_614.39       0.3595          1.0334         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          7_148.61       362.01     7_510.62       0.2495          1.0506         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          7_148.61       444.97     7_593.58       0.3577          1.0337         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          7_148.61       367.53     7_516.14       0.2484          1.0510         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          7_148.61       455.49     7_604.10       0.3549          1.0342         2.65
IVF-Binary-256-nl316-random (self)                     7_148.61     1_103.07     8_251.69       0.2535          1.0466         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               9_314.27       250.85     9_565.12       0.1474             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              9_314.27       257.01     9_571.28       0.1464             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              9_314.27       264.22     9_578.49       0.1462             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              9_314.27       354.09     9_668.36       0.4622          1.0222         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              9_314.27       441.25     9_755.52       0.6310          1.0117         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             9_314.27       358.20     9_672.47       0.4545          1.0229         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             9_314.27       484.02     9_798.29       0.6177          1.0123         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             9_314.27       361.98     9_676.24       0.4509          1.0232         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             9_314.27       452.06     9_766.33       0.6112          1.0126         2.34
IVF-Binary-256-nl158-pca (self)                        9_314.27     1_094.09    10_408.36       0.4544          1.0230         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              7_124.80       263.15     7_387.94       0.1473             NaN         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              7_124.80       265.89     7_390.69       0.1466             NaN         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              7_124.80       272.18     7_396.98       0.1461             NaN         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             7_124.80       364.20     7_489.00       0.4625          1.0222         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             7_124.80       449.81     7_574.61       0.6300          1.0117         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             7_124.80       361.14     7_485.94       0.4575          1.0226         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             7_124.80       452.20     7_577.00       0.6220          1.0121         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             7_124.80       368.24     7_493.04       0.4528          1.0230         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             7_124.80       463.99     7_588.78       0.6141          1.0125         2.47
IVF-Binary-256-nl223-pca (self)                        7_124.80     1_121.45     8_246.25       0.4569          1.0228         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              7_422.89       277.86     7_700.75       0.1473             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              7_422.89       282.60     7_705.49       0.1469             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              7_422.89       289.45     7_712.34       0.1463             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             7_422.89       380.86     7_803.75       0.4626          1.0222         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             7_422.89       461.84     7_884.73       0.6303          1.0117         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             7_422.89       374.33     7_797.23       0.4600          1.0224         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             7_422.89       468.01     7_890.90       0.6262          1.0119         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             7_422.89       381.17     7_804.06       0.4541          1.0230         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             7_422.89       476.07     7_898.97       0.6166          1.0124         2.65
IVF-Binary-256-nl316-pca (self)                        7_422.89     1_237.06     8_659.95       0.4595          1.0225         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           14_901.30       464.49    15_365.79       0.0821             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          14_901.30       475.89    15_377.20       0.0818             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          14_901.30       484.30    15_385.61       0.0818             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          14_901.30       551.95    15_453.25       0.2219          1.0530         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          14_901.30       685.86    15_587.16       0.3325          1.0345         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         14_901.30       564.20    15_465.50       0.2176          1.0544         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         14_901.30       654.78    15_556.09       0.3243          1.0358         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         14_901.30       574.31    15_475.61       0.2176          1.0544         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         14_901.30       659.11    15_560.42       0.3243          1.0358         4.36
IVF-Binary-512-nl158-random (self)                    14_901.30     1_789.91    16_691.21       0.2193          1.0530         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          12_660.14       476.90    13_137.04       0.0837             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          12_660.14       487.38    13_147.52       0.0834             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          12_660.14       494.93    13_155.07       0.0833             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         12_660.14       568.60    13_228.73       0.2218          1.0531         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         12_660.14       646.67    13_306.81       0.3316          1.0346         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         12_660.14       584.63    13_244.77       0.2186          1.0541         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         12_660.14       694.85    13_354.98       0.3256          1.0355         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         12_660.14       607.74    13_267.88       0.2174          1.0545         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         12_660.14       700.94    13_361.08       0.3233          1.0359         4.49
IVF-Binary-512-nl223-random (self)                    12_660.14     1_809.09    14_469.22       0.2208          1.0527         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          12_928.17       495.57    13_423.74       0.0853             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          12_928.17       496.24    13_424.42       0.0851             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          12_928.17       508.16    13_436.33       0.0849             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         12_928.17       589.59    13_517.76       0.2241          1.0525         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         12_928.17       666.63    13_594.80       0.3326          1.0344         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         12_928.17       589.56    13_517.73       0.2221          1.0532         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         12_928.17       668.11    13_596.28       0.3292          1.0349         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         12_928.17       598.19    13_526.37       0.2195          1.0540         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         12_928.17       680.97    13_609.14       0.3244          1.0357         4.67
IVF-Binary-512-nl316-random (self)                    12_928.17     1_865.16    14_793.33       0.2239          1.0519         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              15_080.96       466.25    15_547.21       0.2022             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             15_080.96       478.33    15_559.29       0.1994             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             15_080.96       491.91    15_572.86       0.1973             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             15_080.96       561.06    15_642.02       0.6221          1.0119         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             15_080.96       641.54    15_722.50       0.7874          1.0051         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            15_080.96       568.14    15_649.10       0.6058          1.0127         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            15_080.96       668.16    15_749.12       0.7663          1.0058         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            15_080.96       579.10    15_660.06       0.5927          1.0134         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            15_080.96       679.74    15_760.70       0.7505          1.0063         4.36
IVF-Binary-512-nl158-pca (self)                       15_080.96     1_817.02    16_897.98       0.6061          1.0127         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             12_788.15       482.14    13_270.28       0.2023             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             12_788.15       491.52    13_279.67       0.2008             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             12_788.15       507.85    13_296.00       0.1987             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            12_788.15       601.22    13_389.37       0.6194          1.0120         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            12_788.15       693.26    13_481.40       0.7842          1.0052         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            12_788.15       586.39    13_374.54       0.6115          1.0124         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            12_788.15       669.94    13_458.09       0.7748          1.0055         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            12_788.15       595.54    13_383.69       0.5998          1.0130         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            12_788.15       682.15    13_470.30       0.7593          1.0060         4.49
IVF-Binary-512-nl223-pca (self)                       12_788.15     1_837.36    14_625.51       0.6121          1.0124         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             12_982.27       500.94    13_483.21       0.2024             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             12_982.27       498.81    13_481.08       0.2016             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             12_982.27       510.36    13_492.63       0.1993             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            12_982.27       593.62    13_575.89       0.6203          1.0119         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            12_982.27       679.99    13_662.27       0.7853          1.0052         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            12_982.27       594.59    13_576.86       0.6163          1.0121         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            12_982.27       684.87    13_667.14       0.7804          1.0053         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            12_982.27       605.40    13_587.68       0.6041          1.0128         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            12_982.27       695.95    13_678.22       0.7652          1.0058         4.67
IVF-Binary-512-nl316-pca (self)                       12_982.27     1_868.76    14_851.03       0.6166          1.0122         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          26_192.97       899.78    27_092.75       0.0908             NaN         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)         26_192.97       912.10    27_105.07       0.0903             NaN         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)         26_192.97       927.73    27_120.69       0.0903             NaN         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)         26_192.97       981.55    27_174.52       0.2370          1.0493         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)         26_192.97     1_199.00    27_391.97       0.3544          1.0317         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)        26_192.97     1_044.40    27_237.37       0.2323          1.0507         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)        26_192.97     1_085.46    27_278.43       0.3452          1.0329         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)        26_192.97     1_119.16    27_312.13       0.2323          1.0507         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)        26_192.97     1_232.57    27_425.54       0.3452          1.0329         8.42
IVF-Binary-1024-nl158-random (self)                   26_192.97     3_247.83    29_440.80       0.2311          1.0508         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)         23_888.06       907.18    24_795.24       0.0910             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         23_888.06       913.38    24_801.44       0.0906             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         23_888.06       926.83    24_814.89       0.0904             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        23_888.06     1_014.06    24_902.12       0.2359          1.0495         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        23_888.06     1_089.62    24_977.68       0.3537          1.0318         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        23_888.06       999.69    24_887.74       0.2323          1.0505         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        23_888.06     1_111.80    24_999.86       0.3470          1.0327         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        23_888.06     1_026.56    24_914.62       0.2309          1.0508         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        23_888.06     1_119.37    25_007.42       0.3447          1.0330         8.54
IVF-Binary-1024-nl223-random (self)                   23_888.06     3_239.61    27_127.67       0.2322          1.0505         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         24_434.85       929.46    25_364.31       0.0913             NaN         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)         24_434.85       935.60    25_370.44       0.0911             NaN         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)         24_434.85       940.34    25_375.19       0.0908             NaN         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)        24_434.85     1_019.59    25_454.44       0.2370          1.0492         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)        24_434.85     1_094.86    25_529.70       0.3543          1.0316         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)        24_434.85     1_003.93    25_438.77       0.2351          1.0498         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)        24_434.85     1_149.82    25_584.67       0.3505          1.0321         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)        24_434.85     1_021.18    25_456.03       0.2324          1.0505         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)        24_434.85     1_117.75    25_552.59       0.3457          1.0328         8.73
IVF-Binary-1024-nl316-random (self)                   24_434.85     3_283.78    27_718.62       0.2348          1.0498         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)             26_447.66       893.58    27_341.24       0.2052             NaN         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            26_447.66       917.25    27_364.91       0.2041             NaN         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            26_447.66       949.99    27_397.65       0.2041             NaN         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            26_447.66       989.04    27_436.69       0.6328          1.0113         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            26_447.66     1_073.31    27_520.97       0.7987          1.0048         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           26_447.66       999.33    27_446.99       0.6289          1.0116         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           26_447.66     1_089.48    27_537.13       0.7939          1.0049         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           26_447.66     1_010.53    27_458.19       0.6289          1.0116         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           26_447.66     1_103.87    27_551.53       0.7939          1.0049         8.42
IVF-Binary-1024-nl158-pca (self)                      26_447.66     3_265.28    29_712.94       0.6289          1.0116         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            24_058.90       911.65    24_970.55       0.2054             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            24_058.90       912.77    24_971.67       0.2046             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            24_058.90       929.38    24_988.28       0.2042             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           24_058.90       987.86    25_046.76       0.6337          1.0113         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           24_058.90     1_084.86    25_143.76       0.7996          1.0047         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           24_058.90     1_002.80    25_061.70       0.6305          1.0115         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           24_058.90     1_092.83    25_151.73       0.7960          1.0048         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           24_058.90     1_018.33    25_077.23       0.6288          1.0116         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           24_058.90     1_107.93    25_166.83       0.7938          1.0049         8.54
IVF-Binary-1024-nl223-pca (self)                      24_058.90     3_257.82    27_316.72       0.6310          1.0115         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            24_465.23       930.16    25_395.39       0.2057             NaN         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            24_465.23       943.43    25_408.66       0.2052             NaN         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            24_465.23       977.82    25_443.05       0.2044             NaN         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           24_465.23     1_057.59    25_522.82       0.6341          1.0113         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           24_465.23     1_093.47    25_558.70       0.7994          1.0047         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           24_465.23     1_010.28    25_475.51       0.6325          1.0114         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           24_465.23     1_101.49    25_566.72       0.7976          1.0048         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           24_465.23     1_023.46    25_488.69       0.6294          1.0115         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           24_465.23     1_115.16    25_580.39       0.7938          1.0049         8.73
IVF-Binary-1024-nl316-pca (self)                      24_465.23     3_266.28    27_731.52       0.6326          1.0114         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)              3_316.95       465.70     3_782.65       0.1903             NaN         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)             3_316.95       498.10     3_815.05       0.1666             NaN         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)             3_316.95       532.46     3_849.41       0.1605             NaN         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)             3_316.95       525.24     3_842.19       0.5641          1.0159         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)             3_316.95       914.89     4_231.84       0.7467          1.0067         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)            3_316.95       555.35     3_872.30       0.4736          1.0227         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)            3_316.95       974.44     4_291.39       0.6386          1.0113         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)            3_316.95       589.02     3_905.97       0.4276          1.0273         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)            3_316.95     1_008.80     4_325.75       0.5778          1.0147         3.36
IVF-Binary-512-nl158-sign (self)                       3_316.95     1_750.69     5_067.64       0.4720          1.0229         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)             1_065.96       504.28     1_570.24       0.1852             NaN         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)             1_065.96       519.76     1_585.72       0.1731             NaN         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)             1_065.96       566.70     1_632.65       0.1616             NaN         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)            1_065.96       554.94     1_620.90       0.5408          1.0176         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)            1_065.96       951.08     2_017.04       0.7201          1.0078         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)            1_065.96       569.06     1_635.02       0.4996          1.0208         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)            1_065.96       970.88     2_036.84       0.6711          1.0099         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)            1_065.96       612.41     1_678.37       0.4440          1.0723         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)            1_065.96     1_025.38     2_091.34       0.5990          1.0136         3.49
IVF-Binary-512-nl223-sign (self)                       1_065.96     1_799.95     2_865.91       0.4991          1.0209         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)             1_376.59       530.43     1_907.02       0.1808             NaN         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)             1_376.59       548.80     1_925.39       0.1734             NaN         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)             1_376.59       582.65     1_959.24       0.1577             NaN         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)            1_376.59       586.49     1_963.08       0.5320          1.0180         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)            1_376.59       985.57     2_362.16       0.7118          1.0080         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)            1_376.59       595.35     1_971.94       0.5094          1.0197         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)            1_376.59       997.81     2_374.41       0.6862          1.0091         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)            1_376.59       632.35     2_008.94       0.4500          1.0556         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)            1_376.59     1_055.61     2_432.21       0.6108          1.0128         3.67
IVF-Binary-512-nl316-sign (self)                       1_376.59     1_903.36     3_279.95       0.5089          1.0199         3.67
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       102.39    15_940.38    16_042.78       1.0000          1.0000       146.48
Exhaustive (self)                                        102.39    54_168.01    54_270.40       1.0000          1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)              9_079.86       531.65     9_611.51       0.0332             NaN         2.28
ExhaustiveBinary-256-random-rf10 (query)               9_079.86       677.76     9_757.62       0.1442          1.0758         2.28
ExhaustiveBinary-256-random-rf20 (query)               9_079.86       834.18     9_914.03       0.2373          1.0487         2.28
ExhaustiveBinary-256-random (self)                     9_079.86     2_248.20    11_328.06       0.1474          1.0721         2.28
ExhaustiveBinary-256-pca_no_rr (query)                 9_472.70       532.06    10_004.76       0.1247             NaN         2.28
ExhaustiveBinary-256-pca-rf10 (query)                  9_472.70       707.88    10_180.59       0.3430          1.0271         2.28
ExhaustiveBinary-256-pca-rf20 (query)                  9_472.70       882.94    10_355.64       0.4582          1.0177         2.28
ExhaustiveBinary-256-pca (self)                        9_472.70     2_326.07    11_798.78       0.3433          1.0271         2.28
ExhaustiveBinary-512-random_no_rr (query)             17_747.75       911.19    18_658.94       0.0614             NaN         4.55
ExhaustiveBinary-512-random-rf10 (query)              17_747.75     1_066.37    18_814.12       0.1711          1.0556         4.55
ExhaustiveBinary-512-random-rf20 (query)              17_747.75     1_232.98    18_980.73       0.2648          1.0362         4.55
ExhaustiveBinary-512-random (self)                    17_747.75     3_535.99    21_283.74       0.1730          1.0537         4.55
ExhaustiveBinary-512-pca_no_rr (query)                18_075.87       918.34    18_994.21       0.1450             NaN         4.55
ExhaustiveBinary-512-pca-rf10 (query)                 18_075.87     1_079.55    19_155.41       0.3791          1.0668         4.55
ExhaustiveBinary-512-pca-rf20 (query)                 18_075.87     1_251.39    19_327.25       0.4884          1.0163         4.55
ExhaustiveBinary-512-pca (self)                       18_075.87     3_635.44    21_711.30       0.3800          1.0959         4.55
ExhaustiveBinary-1024-random_no_rr (query)            35_357.56     1_703.80    37_061.36       0.0802             NaN         9.11
ExhaustiveBinary-1024-random-rf10 (query)             35_357.56     1_876.30    37_233.86       0.1928          1.0470         9.11
ExhaustiveBinary-1024-random-rf20 (query)             35_357.56     2_052.61    37_410.16       0.2946          1.0307         9.11
ExhaustiveBinary-1024-random (self)                   35_357.56     6_213.08    41_570.64       0.1937          1.0468         9.11
ExhaustiveBinary-1024-pca_no_rr (query)               35_605.79     1_713.58    37_319.37       0.2012             NaN         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                35_605.79     1_975.87    37_581.66       0.6300          1.0091         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                35_605.79     2_095.51    37_701.30       0.7963          1.0039         9.11
ExhaustiveBinary-1024-pca (self)                      35_605.79     6_293.21    41_899.01       0.6289          1.0092         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  191.25       941.48     1_132.73       0.0656             NaN         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   191.25     1_031.70     1_222.96       0.1789          1.0520         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   191.25     1_688.34     1_879.59       0.2785          1.0331         4.58
ExhaustiveBinary-768-sign (self)                         191.25     3_396.54     3_587.79       0.1795          1.0512         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)           13_852.27       374.37    14_226.64       0.0632             NaN         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)          13_852.27       379.04    14_231.32       0.0630             NaN         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)          13_852.27       386.97    14_239.24       0.0630             NaN         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)          13_852.27       488.25    14_340.52       0.2166          1.0479         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)          13_852.27       613.42    14_465.69       0.3248          1.0312         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)         13_852.27       520.10    14_372.37       0.2127          1.0489         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)         13_852.27       611.32    14_463.59       0.3153          1.0323         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)         13_852.27       601.58    14_453.85       0.2127          1.0489         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)         13_852.27       647.05    14_499.32       0.3153          1.0323         2.74
IVF-Binary-256-nl158-random (self)                    13_852.27     1_674.71    15_526.98       0.2156          1.0455         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)          10_365.68       396.02    10_761.70       0.0693             NaN         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)          10_365.68       400.64    10_766.32       0.0691             NaN         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)          10_365.68       405.20    10_770.88       0.0690             NaN         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)         10_365.68       515.68    10_881.36       0.2307          1.0439         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)         10_365.68       615.50    10_981.18       0.3376          1.0286         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)         10_365.68       514.71    10_880.39       0.2280          1.0446         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)         10_365.68       628.75    10_994.43       0.3321          1.0294         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)         10_365.68       527.35    10_893.03       0.2271          1.0449         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)         10_365.68       637.21    11_002.89       0.3300          1.0297         2.93
IVF-Binary-256-nl223-random (self)                    10_365.68     1_591.05    11_956.73       0.2307          1.0413         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)          10_911.23       420.07    11_331.30       0.0744             NaN         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)          10_911.23       429.45    11_340.68       0.0744             NaN         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)          10_911.23       427.77    11_339.01       0.0743             NaN         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)         10_911.23       550.83    11_462.07       0.2356          1.0424         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)         10_911.23       643.85    11_555.08       0.3409          1.0283         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)         10_911.23       540.72    11_451.96       0.2343          1.0428         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)         10_911.23       639.74    11_550.98       0.3380          1.0287         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)         10_911.23       583.90    11_495.14       0.2327          1.0433         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)         10_911.23       659.86    11_571.10       0.3346          1.0292         3.21
IVF-Binary-256-nl316-random (self)                    10_911.23     1_665.25    12_576.48       0.2366          1.0396         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)              14_318.50       382.49    14_700.99       0.1303             NaN         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)             14_318.50       382.94    14_701.43       0.1294             NaN         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)             14_318.50       389.81    14_708.31       0.1293             NaN         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)             14_318.50       512.50    14_830.99       0.4036          1.0217         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)             14_318.50       621.36    14_939.85       0.5656          1.0121         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)            14_318.50       509.62    14_828.11       0.3955          1.0224         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)            14_318.50       626.22    14_944.71       0.5518          1.0127         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)            14_318.50       517.79    14_836.29       0.3932          1.0225         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)            14_318.50       639.89    14_958.38       0.5468          1.0129         2.74
IVF-Binary-256-nl158-pca (self)                       14_318.50     1_604.29    15_922.79       0.3958          1.0224         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)             10_840.85       397.23    11_238.08       0.1302             NaN         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)             10_840.85       401.47    11_242.32       0.1295             NaN         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)             10_840.85       407.64    11_248.48       0.1293             NaN         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)            10_840.85       523.75    11_364.60       0.4015          1.0218         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)            10_840.85       639.32    11_480.16       0.5610          1.0122         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)            10_840.85       525.73    11_366.57       0.3965          1.0222         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)            10_840.85       644.90    11_485.74       0.5531          1.0126         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)            10_840.85       537.24    11_378.08       0.3928          1.0225         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)            10_840.85       652.82    11_493.67       0.5463          1.0129         2.93
IVF-Binary-256-nl223-pca (self)                       10_840.85     1_652.39    12_493.24       0.3964          1.0223         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)             11_385.37       423.02    11_808.39       0.1299             NaN         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)             11_385.37       429.61    11_814.98       0.1297             NaN         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)             11_385.37       437.34    11_822.72       0.1293             NaN         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)            11_385.37       556.09    11_941.46       0.4009          1.0218         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)            11_385.37       678.85    12_064.22       0.5608          1.0122         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)            11_385.37       552.52    11_937.89       0.3985          1.0221         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)            11_385.37       673.62    12_058.99       0.5567          1.0124         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)            11_385.37       566.10    11_951.47       0.3942          1.0224         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)            11_385.37       680.14    12_065.52       0.5490          1.0128         3.21
IVF-Binary-256-nl316-pca (self)                       11_385.37     1_739.31    13_124.68       0.3985          1.0221         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)           22_801.66       821.60    23_623.25       0.0781             NaN         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)          22_801.66       846.48    23_648.14       0.0776             NaN         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)          22_801.66       815.13    23_616.78       0.0776             NaN         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)          22_801.66       909.31    23_710.96       0.2086          1.0447         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)          22_801.66     1_035.58    23_837.23       0.3146          1.0290         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)         22_801.66       952.13    23_753.79       0.2030          1.0459         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)         22_801.66     1_009.09    23_810.75       0.3037          1.0302         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)         22_801.66       877.45    23_679.10       0.2030          1.0459         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)         22_801.66       990.68    23_792.33       0.3037          1.0302         5.02
IVF-Binary-512-nl158-random (self)                    22_801.66     2_935.44    25_737.10       0.2040          1.0447         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)          19_462.51       722.31    20_184.82       0.0816             NaN         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)          19_462.51       726.56    20_189.07       0.0812             NaN         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)          19_462.51       743.06    20_205.57       0.0811             NaN         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)         19_462.51       838.00    20_300.52       0.2116          1.0436         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)         19_462.51       944.19    20_406.70       0.3155          1.0287         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)         19_462.51       850.30    20_312.81       0.2083          1.0445         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)         19_462.51       964.82    20_427.33       0.3091          1.0295         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)         19_462.51       894.18    20_356.69       0.2072          1.0448         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)         19_462.51       962.34    20_424.85       0.3071          1.0298         5.21
IVF-Binary-512-nl223-random (self)                    19_462.51     2_705.49    22_168.00       0.2097          1.0432         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)          19_784.68       744.29    20_528.97       0.0832             NaN         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)          19_784.68       760.89    20_545.57       0.0830             NaN         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)          19_784.68       751.80    20_536.49       0.0828             NaN         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)         19_784.68       875.01    20_659.69       0.2134          1.0433         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)         19_784.68     1_048.78    20_833.47       0.3161          1.0287         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)         19_784.68       879.59    20_664.27       0.2114          1.0438         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)         19_784.68       975.55    20_760.23       0.3126          1.0291         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)         19_784.68       874.54    20_659.23       0.2093          1.0444         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)         19_784.68       985.62    20_770.30       0.3087          1.0296         5.48
IVF-Binary-512-nl316-random (self)                    19_784.68     2_781.08    22_565.76       0.2126          1.0426         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)              23_182.77       698.79    23_881.57       0.1702             NaN         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)             23_182.77       721.12    23_903.89       0.1680             NaN         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)             23_182.77       745.14    23_927.91       0.1671             NaN         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)             23_182.77       852.07    24_034.84       0.5301          1.0137         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)             23_182.77       946.41    24_129.18       0.7011          1.0067         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)            23_182.77       844.03    24_026.80       0.5154          1.0145         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)            23_182.77       946.18    24_128.96       0.6805          1.0074         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)            23_182.77       845.71    24_028.48       0.5065          1.0150         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)            23_182.77       955.98    24_138.75       0.6668          1.0078         5.02
IVF-Binary-512-nl158-pca (self)                       23_182.77     2_675.37    25_858.14       0.5168          1.0145         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)             19_665.29       754.64    20_419.93       0.1695             NaN         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)             19_665.29       743.01    20_408.31       0.1680             NaN         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)             19_665.29       737.05    20_402.34       0.1666             NaN         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)            19_665.29       845.31    20_510.60       0.5243          1.0140         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)            19_665.29       953.86    20_619.15       0.6927          1.0069         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)            19_665.29       840.55    20_505.85       0.5163          1.0144         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)            19_665.29       962.44    20_627.73       0.6820          1.0073         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)            19_665.29       868.85    20_534.15       0.5060          1.0150         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)            19_665.29       972.93    20_638.22       0.6662          1.0078         5.21
IVF-Binary-512-nl223-pca (self)                       19_665.29     2_714.38    22_379.67       0.5174          1.0145         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)             20_124.19       752.37    20_876.56       0.1694             NaN         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)             20_124.19       751.40    20_875.59       0.1688             NaN         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)             20_124.19       771.40    20_895.59       0.1674             NaN         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)            20_124.19       894.80    21_018.99       0.5247          1.0140         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)            20_124.19     1_024.78    21_148.97       0.6935          1.0069         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)            20_124.19       872.24    20_996.43       0.5208          1.0142         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)            20_124.19       986.59    21_110.78       0.6884          1.0071         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)            20_124.19       876.36    21_000.55       0.5109          1.0147         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)            20_124.19     1_004.61    21_128.80       0.6739          1.0076         5.48
IVF-Binary-512-nl316-pca (self)                       20_124.19     2_807.06    22_931.25       0.5219          1.0142         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)          40_452.20     1_350.51    41_802.71       0.0862             NaN         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)         40_452.20     1_356.98    41_809.18       0.0855             NaN         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)         40_452.20     1_373.33    41_825.53       0.0855             NaN         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)         40_452.20     1_465.97    41_918.17       0.2152          1.0422         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)         40_452.20     1_571.85    42_024.05       0.3256          1.0274         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)        40_452.20     1_466.04    41_918.24       0.2091          1.0434         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)        40_452.20     1_587.67    42_039.87       0.3142          1.0285         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)        40_452.20     1_489.43    41_941.63       0.2091          1.0434         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)        40_452.20     1_619.43    42_071.63       0.3142          1.0285         9.57
IVF-Binary-1024-nl158-random (self)                   40_452.20     4_808.27    45_260.48       0.2102          1.0432         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)         36_777.90     1_374.14    38_152.05       0.0871             NaN         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)         36_777.90     1_375.84    38_153.75       0.0867             NaN         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)         36_777.90     1_484.56    38_262.47       0.0865             NaN         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)        36_777.90     1_488.84    38_266.75       0.2153          1.0419         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)        36_777.90     1_593.97    38_371.87       0.3256          1.0272         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)        36_777.90     1_494.44    38_272.34       0.2113          1.0429         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)        36_777.90     1_606.66    38_384.57       0.3177          1.0281         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)        36_777.90     1_497.30    38_275.20       0.2103          1.0432         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)        36_777.90     1_619.46    38_397.37       0.3155          1.0284         9.76
IVF-Binary-1024-nl223-random (self)                   36_777.90     4_903.86    41_681.76       0.2120          1.0427         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)         37_376.43     1_400.70    38_777.13       0.0872             NaN        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)         37_376.43     1_407.62    38_784.05       0.0869             NaN        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)         37_376.43     1_413.62    38_790.05       0.0866             NaN        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)        37_376.43     1_519.57    38_896.01       0.2154          1.0419        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)        37_376.43     1_631.95    39_008.38       0.3254          1.0273        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)        37_376.43     1_506.50    38_882.94       0.2132          1.0425        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)        37_376.43     1_627.94    39_004.37       0.3213          1.0278        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)        37_376.43     1_529.75    38_906.19       0.2110          1.0431        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)        37_376.43     1_688.40    39_064.83       0.3168          1.0284        10.04
IVF-Binary-1024-nl316-random (self)                   37_376.43     5_017.02    42_393.46       0.2138          1.0423        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)             40_670.92     1_362.45    42_033.37       0.2023             NaN         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)            40_670.92     1_365.37    42_036.29       0.2015             NaN         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)            40_670.92     1_397.81    42_068.73       0.2015             NaN         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)            40_670.92     1_468.51    42_139.43       0.6338          1.0090         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)            40_670.92     1_575.24    42_246.16       0.8011          1.0037         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)           40_670.92     1_476.69    42_147.60       0.6303          1.0091         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)           40_670.92     1_609.18    42_280.10       0.7968          1.0039         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)           40_670.92     1_527.60    42_198.52       0.6303          1.0091         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)           40_670.92     1_607.18    42_278.10       0.7968          1.0039         9.57
IVF-Binary-1024-nl158-pca (self)                      40_670.92     4_839.15    45_510.07       0.6292          1.0092         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)            37_077.09     1_370.88    38_447.97       0.2026             NaN         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)            37_077.09     1_386.02    38_463.11       0.2018             NaN         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)            37_077.09     1_394.46    38_471.56       0.2016             NaN         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)           37_077.09     1_490.71    38_567.81       0.6344          1.0089         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)           37_077.09     1_605.06    38_682.15       0.8014          1.0037         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)           37_077.09     1_488.61    38_565.70       0.6315          1.0091         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)           37_077.09     1_603.65    38_680.75       0.7980          1.0038         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)           37_077.09     1_509.88    38_586.98       0.6308          1.0091         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)           37_077.09     1_620.22    38_697.31       0.7970          1.0039         9.76
IVF-Binary-1024-nl223-pca (self)                      37_077.09     4_871.09    41_948.18       0.6302          1.0092         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)            37_666.78     1_397.09    39_063.86       0.2024             NaN        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)            37_666.78     1_393.36    39_060.14       0.2020             NaN        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)            37_666.78     1_403.49    39_070.27       0.2017             NaN        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)           37_666.78     1_502.71    39_169.49       0.6335          1.0090        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)           37_666.78     1_619.33    39_286.11       0.8005          1.0037        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)           37_666.78     1_514.51    39_181.29       0.6319          1.0090        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)           37_666.78     1_651.75    39_318.53       0.7988          1.0038        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)           37_666.78     1_599.57    39_266.34       0.6304          1.0091        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)           37_666.78     1_638.63    39_305.40       0.7968          1.0039        10.04
IVF-Binary-1024-nl316-pca (self)                      37_666.78     4_959.18    42_625.96       0.6310          1.0091        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)              5_056.93       643.71     5_700.64       0.1938             NaN         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)             5_056.93       695.04     5_751.98       0.1714             NaN         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)             5_056.93       737.84     5_794.77       0.1647             NaN         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)             5_056.93       736.48     5_793.41       0.5714          1.0122         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)             5_056.93     1_293.10     6_350.03       0.7530          1.0051         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)            5_056.93       777.89     5_834.83       0.4916          1.0166         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)            5_056.93     1_340.98     6_397.91       0.6605          1.0080         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)            5_056.93       807.12     5_864.06       0.4448          1.0200         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)            5_056.93     1_383.05     6_439.98       0.5966          1.0106         5.04
IVF-Binary-768-nl158-sign (self)                       5_056.93     2_449.74     7_506.67       0.4918          1.0166         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)             1_542.37       687.92     2_230.29       0.1859             NaN         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)             1_542.37       724.67     2_267.04       0.1725             NaN         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)             1_542.37       776.10     2_318.46       0.1584             NaN         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)            1_542.37       807.89     2_350.26       0.5332          1.0142         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)            1_542.37     1_371.76     2_914.13       0.7094          1.0065         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)            1_542.37       816.66     2_359.02       0.4918          1.0167         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)            1_542.37     1_379.05     2_921.42       0.6619          1.0081         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)            1_542.37       852.70     2_395.07       0.4268          1.0214         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)            1_542.37     1_446.24     2_988.61       0.5778          1.0116         5.23
IVF-Binary-768-nl223-sign (self)                       1_542.37     2_542.92     4_085.29       0.4919          1.0168         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)             2_116.14       747.16     2_863.29       0.1807             NaN         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)             2_116.14       770.96     2_887.10       0.1735             NaN         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)             2_116.14       819.70     2_935.83       0.1570             NaN         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)            2_116.14       839.57     2_955.71       0.5243          1.0145         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)            2_116.14     1_400.19     3_516.32       0.7041          1.0066         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)            2_116.14       868.30     2_984.43       0.5025          1.0158         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)            2_116.14     1_528.18     3_644.32       0.6789          1.0074         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)            2_116.14       925.01     3_041.14       0.4367          1.0204         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)            2_116.14     1_490.91     3_607.04       0.5979          1.0105         5.51
IVF-Binary-768-nl316-sign (self)                       2_116.14     2_729.77     4_845.90       0.5043          1.0159         5.51
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.74     4_051.81     4_084.55       1.0000          1.0000        48.83
Exhaustive (self)                                         32.74    13_175.70    13_208.44       1.0000          1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_697.68       279.30     2_976.97       0.0883             NaN         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_697.68       383.55     3_081.22       0.3317          1.1509         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_697.68       482.89     3_180.57       0.4734          1.0879         1.78
ExhaustiveBinary-256-random (self)                     2_697.68     1_265.88     3_963.56       0.3570          1.1574         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_770.55       280.96     3_051.51       0.1109             NaN         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_770.55       401.71     3_172.26       0.3158          1.5913         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_770.55       499.68     3_270.23       0.4283          1.3071         1.78
ExhaustiveBinary-256-pca (self)                        2_770.55     1_302.14     4_072.68       0.2950          2.2349         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_313.38       464.22     5_777.60       0.1384             NaN         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_313.38       562.97     5_876.35       0.4321          1.0993         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_313.38       668.23     5_981.61       0.5844          1.0539         3.55
ExhaustiveBinary-512-random (self)                     5_313.38     1_870.11     7_183.49       0.4548          1.1058         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_389.82       467.08     5_856.90       0.1283             NaN         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_389.82       572.88     5_962.70       0.4093          1.1233         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_389.82       679.41     6_069.24       0.5766          1.0656         3.55
ExhaustiveBinary-512-pca (self)                        5_389.82     1_910.26     7_300.08       0.4109          1.1443         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_469.82       778.07    11_247.89       0.1996             NaN         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_469.82       881.93    11_351.75       0.5516          1.0599         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_469.82       994.05    11_463.87       0.7090          1.0295         7.10
ExhaustiveBinary-1024-random (self)                   10_469.82     2_923.11    13_392.93       0.5787          1.0643         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_584.31       778.35    11_362.67       0.1535             NaN         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_584.31       886.18    11_470.49       0.4703          1.0875         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_584.31       997.84    11_582.16       0.6408          1.0449         7.10
ExhaustiveBinary-1024-pca (self)                      10_584.31     3_016.80    13_601.11       0.4638          1.1066         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   55.41       533.74       589.15       0.0928             NaN         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    55.41       545.43       600.84       0.3447          1.1453         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    55.41       890.86       946.27       0.5011          1.0792         1.53
ExhaustiveBinary-256-sign (self)                          55.41     1_761.25     1_816.66       0.3670          1.1539         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)            4_374.26       129.34     4_503.60       0.0926             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           4_374.26       121.62     4_495.88       0.0925             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           4_374.26       124.92     4_499.18       0.0925             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           4_374.26       183.51     4_557.77       0.3384          1.1478         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           4_374.26       232.78     4_607.04       0.4782          1.0862         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          4_374.26       180.01     4_554.27       0.3375          1.1479         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          4_374.26       234.71     4_608.97       0.4779          1.0863         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          4_374.26       181.55     4_555.81       0.3375          1.1479         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          4_374.26       235.98     4_610.24       0.4779          1.0863         1.93
IVF-Binary-256-nl158-random (self)                     4_374.26       547.40     4_921.66       0.3624          1.1542         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           3_309.17       126.97     3_436.14       0.1058             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           3_309.17       128.77     3_437.94       0.1058             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           3_309.17       135.70     3_444.87       0.1058             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          3_309.17       185.20     3_494.37       0.3691          1.1290         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          3_309.17       234.22     3_543.38       0.5078          1.0755         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          3_309.17       184.65     3_493.82       0.3690          1.1291         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          3_309.17       236.40     3_545.57       0.5077          1.0755         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          3_309.17       189.03     3_498.20       0.3690          1.1291         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          3_309.17       240.93     3_550.10       0.5077          1.0755         2.00
IVF-Binary-256-nl223-random (self)                     3_309.17       552.08     3_861.25       0.3937          1.1340         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           3_507.17       133.02     3_640.19       0.1121             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           3_507.17       137.90     3_645.08       0.1121             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           3_507.17       137.03     3_644.20       0.1121             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          3_507.17       192.20     3_699.37       0.3795          1.1237         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          3_507.17       236.17     3_743.34       0.5175          1.0726         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          3_507.17       189.68     3_696.85       0.3794          1.1238         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          3_507.17       238.59     3_745.76       0.5174          1.0726         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          3_507.17       196.30     3_703.47       0.3794          1.1238         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          3_507.17       241.70     3_748.87       0.5174          1.0726         2.09
IVF-Binary-256-nl316-random (self)                     3_507.17       569.42     4_076.59       0.4031          1.1281         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               4_105.13       131.52     4_236.66       0.1196             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              4_105.13       133.61     4_238.75       0.1159             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              4_105.13       124.05     4_229.18       0.1149             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              4_105.13       191.50     4_296.63       0.3989          1.1281         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              4_105.13       243.47     4_348.61       0.5641          1.0697         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             4_105.13       195.30     4_300.43       0.3773          1.1546         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             4_105.13       249.71     4_354.85       0.5416          1.0820         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             4_105.13       200.67     4_305.80       0.3706          1.1749         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             4_105.13       253.78     4_358.91       0.5311          1.0926         1.93
IVF-Binary-256-nl158-pca (self)                        4_105.13       599.70     4_704.83       0.3840          1.1735         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              3_558.51       264.09     3_822.60       0.1170             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              3_558.51       184.79     3_743.30       0.1162             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              3_558.51       162.72     3_721.23       0.1153             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             3_558.51       285.03     3_843.54       0.3795          1.1454         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             3_558.51       348.88     3_907.40       0.5458          1.0767         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             3_558.51       280.78     3_839.29       0.3747          1.1573         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             3_558.51       355.85     3_914.36       0.5385          1.0832         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             3_558.51       250.78     3_809.29       0.3683          1.1782         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             3_558.51       307.28     3_865.79       0.5272          1.0947         2.00
IVF-Binary-256-nl223-pca (self)                        3_558.51       632.21     4_190.72       0.3821          1.1750         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              3_608.33       151.28     3_759.61       0.1170             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              3_608.33       150.51     3_758.84       0.1165             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              3_608.33       144.64     3_752.97       0.1155             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             3_608.33       218.35     3_826.68       0.3788          1.1469         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             3_608.33       297.15     3_905.48       0.5450          1.0773         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             3_608.33       245.02     3_853.35       0.3765          1.1517         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             3_608.33       284.02     3_892.35       0.5412          1.0802         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             3_608.33       220.60     3_828.93       0.3700          1.1708         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             3_608.33       280.29     3_888.62       0.5302          1.0907         2.09
IVF-Binary-256-nl316-pca (self)                        3_608.33       668.80     4_277.13       0.3839          1.1696         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_768.01       226.97     6_994.98       0.1407             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_768.01       232.61     7_000.62       0.1407             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_768.01       236.64     7_004.66       0.1407             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_768.01       313.66     7_081.67       0.4351          1.0983         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_768.01       364.92     7_132.93       0.5866          1.0533         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_768.01       302.18     7_070.19       0.4348          1.0983         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_768.01       385.38     7_153.39       0.5865          1.0534         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_768.01       317.08     7_085.09       0.4348          1.0983         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_768.01       370.82     7_138.83       0.5865          1.0534         3.71
IVF-Binary-512-nl158-random (self)                     6_768.01       934.15     7_702.17       0.4572          1.1049         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_933.78       231.02     6_164.80       0.1487             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_933.78       236.52     6_170.30       0.1487             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_933.78       241.91     6_175.70       0.1487             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_933.78       299.63     6_233.41       0.4489          1.0924         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_933.78       352.83     6_286.62       0.5987          1.0505         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_933.78       300.93     6_234.71       0.4489          1.0924         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_933.78       351.68     6_285.46       0.5986          1.0505         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_933.78       300.59     6_234.37       0.4489          1.0924         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_933.78       348.78     6_282.56       0.5986          1.0505         3.77
IVF-Binary-512-nl223-random (self)                     5_933.78       913.85     6_847.63       0.4711          1.0994         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           6_335.52       239.54     6_575.06       0.1514             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           6_335.52       243.82     6_579.34       0.1514             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           6_335.52       275.99     6_611.51       0.1514             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          6_335.52       330.81     6_666.33       0.4541          1.0907         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          6_335.52       350.74     6_686.26       0.6030          1.0496         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          6_335.52       295.10     6_630.62       0.4541          1.0907         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          6_335.52       345.58     6_681.10       0.6029          1.0496         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          6_335.52       324.26     6_659.78       0.4541          1.0907         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          6_335.52       376.47     6_711.99       0.6029          1.0496         3.86
IVF-Binary-512-nl316-random (self)                     6_335.52       982.66     7_318.18       0.4751          1.0976         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               6_866.64       222.62     7_089.26       0.1322             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              6_866.64       225.82     7_092.46       0.1293             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              6_866.64       227.34     7_093.98       0.1287             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              6_866.64       292.99     7_159.63       0.4307          1.1079         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              6_866.64       346.45     7_213.10       0.5985          1.0571         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             6_866.64       293.57     7_160.21       0.4145          1.1177         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             6_866.64       354.08     7_220.72       0.5839          1.0618         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             6_866.64       317.71     7_184.35       0.4109          1.1209         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             6_866.64       418.72     7_285.36       0.5793          1.0638         3.71
IVF-Binary-512-nl158-pca (self)                        6_866.64     1_072.78     7_939.42       0.4164          1.1371         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              6_136.99       263.71     6_400.70       0.1303             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              6_136.99       264.84     6_401.83       0.1299             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              6_136.99       277.32     6_414.31       0.1295             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             6_136.99       374.02     6_511.01       0.4159          1.1153         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             6_136.99       417.56     6_554.55       0.5852          1.0604         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             6_136.99       357.82     6_494.80       0.4134          1.1179         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             6_136.99       412.18     6_549.17       0.5819          1.0620         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             6_136.99       358.94     6_495.93       0.4111          1.1205         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             6_136.99       421.50     6_558.48       0.5789          1.0636         3.77
IVF-Binary-512-nl223-pca (self)                        6_136.99     1_032.89     7_169.88       0.4152          1.1371         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              6_388.02       243.84     6_631.86       0.1302             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              6_388.02       245.18     6_633.20       0.1300             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              6_388.02       246.57     6_634.59       0.1295             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             6_388.02       309.11     6_697.13       0.4152          1.1160         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             6_388.02       356.47     6_744.49       0.5845          1.0607         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             6_388.02       303.29     6_691.31       0.4140          1.1172         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             6_388.02       355.60     6_743.62       0.5828          1.0615         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             6_388.02       307.88     6_695.90       0.4113          1.1200         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             6_388.02       361.89     6_749.90       0.5793          1.0632         3.86
IVF-Binary-512-nl316-pca (self)                        6_388.02       964.42     7_352.44       0.4159          1.1364         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_941.11       451.16    12_392.27       0.2008             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_941.11       436.66    12_377.77       0.2008             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_941.11       492.14    12_433.25       0.2008             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_941.11       505.68    12_446.79       0.5526          1.0596         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_941.11       562.74    12_503.85       0.7099          1.0293         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_941.11       553.23    12_494.35       0.5525          1.0596         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_941.11       557.49    12_498.60       0.7099          1.0293         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_941.11       515.24    12_456.35       0.5525          1.0596         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_941.11       648.14    12_589.25       0.7099          1.0293         7.26
IVF-Binary-1024-nl158-random (self)                   11_941.11     1_637.33    13_578.44       0.5797          1.0640         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         11_264.97       454.65    11_719.61       0.2057             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         11_264.97       436.14    11_701.10       0.2057             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         11_264.97       454.82    11_719.78       0.2057             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        11_264.97       561.44    11_826.41       0.5596          1.0579         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        11_264.97       557.26    11_822.22       0.7145          1.0286         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        11_264.97       489.33    11_754.29       0.5596          1.0579         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        11_264.97       604.08    11_869.04       0.7145          1.0286         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        11_264.97       542.02    11_806.98       0.5596          1.0579         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        11_264.97       601.78    11_866.74       0.7145          1.0286         7.32
IVF-Binary-1024-nl223-random (self)                   11_264.97     1_573.68    12_838.64       0.5860          1.0623         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         11_270.80       449.64    11_720.44       0.2073             NaN         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)         11_270.80       425.99    11_696.79       0.2073             NaN         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)         11_270.80       430.98    11_701.78       0.2073             NaN         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)        11_270.80       484.76    11_755.56       0.5622          1.0572         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)        11_270.80       538.08    11_808.88       0.7168          1.0282         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)        11_270.80       493.84    11_764.64       0.5622          1.0572         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)        11_270.80       543.56    11_814.36       0.7168          1.0282         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)        11_270.80       490.47    11_761.27       0.5622          1.0572         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)        11_270.80       542.17    11_812.97       0.7168          1.0282         7.42
IVF-Binary-1024-nl316-random (self)                   11_270.80     1_552.26    12_823.06       0.5884          1.0616         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)             11_854.58       404.94    12_259.52       0.1551             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            11_854.58       409.20    12_263.78       0.1538             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            11_854.58       415.18    12_269.76       0.1538             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            11_854.58       480.65    12_335.24       0.4829          1.0838         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            11_854.58       527.21    12_381.79       0.6510          1.0428         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           11_854.58       479.38    12_333.96       0.4724          1.0867         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           11_854.58       549.59    12_404.17       0.6435          1.0443         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           11_854.58       486.00    12_340.59       0.4710          1.0871         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           11_854.58       541.08    12_395.66       0.6415          1.0446         7.26
IVF-Binary-1024-nl158-pca (self)                      11_854.58     1_551.86    13_406.44       0.4658          1.1058         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            11_134.55       410.16    11_544.71       0.1547             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            11_134.55       412.98    11_547.54       0.1546             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            11_134.55       424.51    11_559.06       0.1545             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           11_134.55       479.46    11_614.01       0.4737          1.0859         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           11_134.55       531.54    11_666.09       0.6443          1.0439         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           11_134.55       480.33    11_614.88       0.4725          1.0864         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           11_134.55       535.67    11_670.23       0.6429          1.0443         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           11_134.55       488.25    11_622.81       0.4717          1.0866         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           11_134.55       555.77    11_690.33       0.6418          1.0445         7.32
IVF-Binary-1024-nl223-pca (self)                      11_134.55     1_563.88    12_698.43       0.4656          1.1055         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            11_330.31       419.61    11_749.92       0.1548             NaN         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            11_330.31       419.88    11_750.19       0.1547             NaN         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            11_330.31       432.77    11_763.08       0.1546             NaN         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           11_330.31       488.70    11_819.01       0.4733          1.0861         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           11_330.31       552.79    11_883.10       0.6441          1.0440         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           11_330.31       486.13    11_816.44       0.4727          1.0864         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           11_330.31       540.39    11_870.70       0.6432          1.0442         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           11_330.31       509.16    11_839.46       0.4716          1.0867         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           11_330.31       549.15    11_879.46       0.6420          1.0444         7.42
IVF-Binary-1024-nl316-pca (self)                      11_330.31     1_576.41    12_906.72       0.4658          1.1055         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)              1_390.32       302.46     1_692.78       0.1181             NaN         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)             1_390.32       322.09     1_712.41       0.1029             NaN         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)             1_390.32       344.28     1_734.61       0.0901             NaN         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)             1_390.32       335.37     1_725.69       0.8722          1.0202         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)             1_390.32       582.03     1_972.35       0.9499          1.0093         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)            1_390.32       348.09     1_738.41       0.7961          1.0555         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)            1_390.32       636.50     2_026.82       0.9269          1.0138         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)            1_390.32       372.32     1_762.65       0.7261          1.1071         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)            1_390.32       620.11     2_010.43       0.8938          1.0226         1.68
IVF-Binary-256-nl158-sign (self)                       1_390.32     1_072.77     2_463.10       0.8251          1.0405         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               700.24       314.13     1_014.38       0.1160             NaN         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               700.24       328.96     1_029.20       0.1082             NaN         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               700.24       362.42     1_062.66       0.0929             NaN         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              700.24       344.57     1_044.82       0.8015          1.0303         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              700.24       588.10     1_288.34       0.9325          1.0098         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              700.24       349.60     1_049.85       0.7466          1.0466         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              700.24       593.76     1_294.01       0.9102          1.0137         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              700.24       386.71     1_086.95       0.6394          1.1165         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              700.24       643.01     1_343.26       0.8489          1.0248         1.75
IVF-Binary-256-nl223-sign (self)                         700.24     1_137.41     1_837.65       0.7838          1.0413         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               858.81       337.13     1_195.93       0.1437             NaN         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               858.81       345.42     1_204.22       0.1319             NaN         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               858.81       376.09     1_234.90       0.1081             NaN         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              858.81       366.69     1_225.49       0.7867          1.0345         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              858.81       614.12     1_472.93       0.9223          1.0108         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              858.81       371.25     1_230.06       0.7581          1.0413         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              858.81       615.59     1_474.40       0.9104          1.0129         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              858.81       397.08     1_255.89       0.6533          1.0804         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              858.81       652.34     1_511.14       0.8522          1.0232         1.84
IVF-Binary-256-nl316-sign (self)                         858.81     1_171.55     2_030.35       0.7916          1.0394         1.84
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        70.19     9_730.07     9_800.26       1.0000          1.0000        97.66
Exhaustive (self)                                         70.19    33_150.78    33_220.97       1.0000          1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_879.33       421.78     6_301.11       0.0656             NaN         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_879.33       554.27     6_433.60       0.2690          1.1363         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_879.33       694.57     6_573.90       0.3926          1.0839         2.03
ExhaustiveBinary-256-random (self)                     5_879.33     1_807.86     7_687.19       0.2893          1.1351         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_162.78       428.52     6_591.30       0.1754             NaN         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_162.78       572.02     6_734.80       0.4664          1.2428         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_162.78       723.92     6_886.70       0.5850          1.1744         2.03
ExhaustiveBinary-256-pca (self)                        6_162.78     1_879.65     8_042.42       0.4693          1.2512         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_657.94       701.08    12_359.02       0.1027             NaN         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_657.94       836.76    12_494.70       0.3402          1.0982         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_657.94     1_008.70    12_666.64       0.4711          1.0574         4.05
ExhaustiveBinary-512-random (self)                    11_657.94     2_883.07    14_541.01       0.3572          1.1005         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_869.07       703.71    12_572.78       0.1672             NaN         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_869.07       859.53    12_728.60       0.4213          1.4308         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_869.07     1_001.85    12_870.91       0.5297          1.2408         4.05
ExhaustiveBinary-512-pca (self)                       11_869.07     2_833.73    14_702.80       0.4104          1.6976         4.05
ExhaustiveBinary-1024-random_no_rr (query)            23_041.45     1_260.28    24_301.73       0.1484             NaN         8.11
ExhaustiveBinary-1024-random-rf10 (query)             23_041.45     1_407.17    24_448.63       0.4138          1.0699         8.11
ExhaustiveBinary-1024-random-rf20 (query)             23_041.45     1_555.54    24_596.99       0.5532          1.0397         8.11
ExhaustiveBinary-1024-random (self)                   23_041.45     4_675.08    27_716.53       0.4277          1.0760         8.11
ExhaustiveBinary-1024-pca_no_rr (query)               23_282.89     1_278.00    24_560.89       0.2385             NaN         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                23_282.89     1_426.33    24_709.22       0.6735          1.0459         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                23_282.89     1_636.29    24_919.18       0.8214          1.0198         8.11
ExhaustiveBinary-1024-pca (self)                      23_282.89     4_731.16    28_014.05       0.6814          1.0492         8.11
ExhaustiveBinary-512-sign_no_rr (query)                  119.29       715.19       834.48       0.1121             NaN         3.05
ExhaustiveBinary-512-sign-rf10 (query)                   119.29       796.63       915.92       0.3436          1.0978         3.05
ExhaustiveBinary-512-sign-rf20 (query)                   119.29     1_360.48     1_479.77       0.4811          1.0554         3.05
ExhaustiveBinary-512-sign (self)                         119.29     2_559.79     2_679.08       0.3589          1.1018         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)            8_740.43       249.55     8_989.98       0.0681             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           8_740.43       257.85     8_998.28       0.0681             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           8_740.43       254.11     8_994.54       0.0681             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           8_740.43       336.26     9_076.69       0.2720          1.1354         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           8_740.43       422.47     9_162.90       0.3943          1.0835         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          8_740.43       334.28     9_074.71       0.2710          1.1355         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          8_740.43       425.41     9_165.85       0.3937          1.0835         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          8_740.43       335.82     9_076.26       0.2709          1.1356         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          8_740.43       435.60     9_176.03       0.3936          1.0836         2.34
IVF-Binary-256-nl158-random (self)                     8_740.43     1_062.68     9_803.11       0.2912          1.1343         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           6_949.22       257.03     7_206.24       0.0821             NaN         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)           6_949.22       261.90     7_211.11       0.0821             NaN         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)           6_949.22       274.20     7_223.42       0.0821             NaN         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)          6_949.22       352.50     7_301.72       0.3011          1.1172         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)          6_949.22       438.05     7_387.26       0.4248          1.0723         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)          6_949.22       348.19     7_297.41       0.3009          1.1174         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)          6_949.22       439.41     7_388.63       0.4241          1.0725         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)          6_949.22       351.38     7_300.60       0.3008          1.1174         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)          6_949.22       440.52     7_389.73       0.4240          1.0725         2.47
IVF-Binary-256-nl223-random (self)                     6_949.22     1_045.48     7_994.70       0.3205          1.1158         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)           7_381.51       280.98     7_662.49       0.0875             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           7_381.51       277.69     7_659.20       0.0875             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           7_381.51       278.63     7_660.15       0.0875             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          7_381.51       364.97     7_746.48       0.3116          1.1116         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          7_381.51       451.25     7_832.76       0.4341          1.0692         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          7_381.51       360.31     7_741.83       0.3114          1.1116         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          7_381.51       449.62     7_831.13       0.4336          1.0694         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          7_381.51       363.47     7_744.98       0.3113          1.1117         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          7_381.51       452.87     7_834.38       0.4333          1.0694         2.65
IVF-Binary-256-nl316-random (self)                     7_381.51     1_096.10     8_477.61       0.3310          1.1095         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               8_925.51       255.18     9_180.69       0.1938             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              8_925.51       251.02     9_176.53       0.1883             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              8_925.51       251.54     9_177.06       0.1872             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              8_925.51       353.95     9_279.47       0.5815          1.0747         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              8_925.51       446.63     9_372.15       0.7399          1.0348         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             8_925.51       352.87     9_278.38       0.5580          1.0958         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             8_925.51       453.40     9_378.92       0.7183          1.0418         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             8_925.51       355.15     9_280.67       0.5521          1.1115         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             8_925.51       454.34     9_379.85       0.7103          1.0476         2.34
IVF-Binary-256-nl158-pca (self)                        8_925.51     1_095.15    10_020.66       0.5790          1.0992         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              7_209.02       261.82     7_470.84       0.1891             NaN         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              7_209.02       262.87     7_471.89       0.1882             NaN         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              7_209.02       266.56     7_475.58       0.1874             NaN         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             7_209.02       366.84     7_575.86       0.5608          1.0858         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             7_209.02       462.29     7_671.31       0.7220          1.0379         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             7_209.02       365.70     7_574.72       0.5561          1.0947         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             7_209.02       465.61     7_674.63       0.7153          1.0415         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             7_209.02       370.79     7_579.81       0.5505          1.1114         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             7_209.02       469.76     7_678.78       0.7070          1.0482         2.47
IVF-Binary-256-nl223-pca (self)                        7_209.02     1_144.16     8_353.18       0.5765          1.0984         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              7_599.15       278.14     7_877.29       0.1890             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              7_599.15       280.78     7_879.93       0.1886             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              7_599.15       283.01     7_882.16       0.1876             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             7_599.15       382.85     7_982.00       0.5596          1.0869         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             7_599.15       492.09     8_091.24       0.7207          1.0383         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             7_599.15       376.16     7_975.31       0.5573          1.0906         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             7_599.15       475.57     8_074.72       0.7176          1.0399         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             7_599.15       383.13     7_982.28       0.5514          1.1071         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             7_599.15       480.11     8_079.26       0.7090          1.0462         2.65
IVF-Binary-256-nl316-pca (self)                        7_599.15     1_184.81     8_783.96       0.5781          1.0942         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           14_503.40       470.02    14_973.42       0.1037             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          14_503.40       499.14    15_002.54       0.1037             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          14_503.40       560.12    15_063.52       0.1037             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          14_503.40       597.42    15_100.81       0.3407          1.0981         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          14_503.40       675.46    15_178.86       0.4714          1.0574         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         14_503.40       578.06    15_081.46       0.3406          1.0981         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         14_503.40       656.88    15_160.27       0.4713          1.0574         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         14_503.40       563.56    15_066.96       0.3406          1.0981         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         14_503.40       650.57    15_153.97       0.4713          1.0574         4.36
IVF-Binary-512-nl158-random (self)                    14_503.40     1_745.94    16_249.34       0.3576          1.1003         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          12_786.85       475.49    13_262.35       0.1132             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          12_786.85       478.66    13_265.51       0.1132             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          12_786.85       509.13    13_295.98       0.1132             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         12_786.85       582.63    13_369.48       0.3571          1.0910         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         12_786.85       652.50    13_439.36       0.4875          1.0534         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         12_786.85       590.36    13_377.22       0.3566          1.0912         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         12_786.85       653.49    13_440.34       0.4867          1.0535         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         12_786.85       571.99    13_358.84       0.3566          1.0912         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         12_786.85       658.40    13_445.25       0.4867          1.0535         4.49
IVF-Binary-512-nl223-random (self)                    12_786.85     1_786.88    14_573.73       0.3716          1.0945         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          13_117.15       490.92    13_608.07       0.1163             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          13_117.15       494.33    13_611.49       0.1163             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          13_117.15       496.56    13_613.71       0.1163             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         13_117.15       578.40    13_695.55       0.3621          1.0889         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         13_117.15       668.51    13_785.67       0.4922          1.0523         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         13_117.15       582.44    13_699.59       0.3619          1.0890         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         13_117.15       668.71    13_785.86       0.4918          1.0524         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         13_117.15       586.64    13_703.80       0.3618          1.0891         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         13_117.15       685.69    13_802.84       0.4916          1.0525         4.67
IVF-Binary-512-nl316-random (self)                    13_117.15     1_839.83    14_956.98       0.3770          1.0924         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              14_772.35       482.48    15_254.83       0.2262             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             14_772.35       479.30    15_251.64       0.2196             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             14_772.35       485.36    15_257.71       0.2177             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             14_772.35       579.24    15_351.59       0.6546          1.0666         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             14_772.35       656.61    15_428.96       0.8024          1.0301         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            14_772.35       568.91    15_341.26       0.6281          1.0924         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            14_772.35       667.64    15_439.99       0.7783          1.0388         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            14_772.35       573.36    15_345.71       0.6191          1.1147         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            14_772.35       670.17    15_442.51       0.7661          1.0484         4.36
IVF-Binary-512-nl158-pca (self)                       14_772.35     1_811.83    16_584.17       0.6371          1.0961         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             12_937.93       478.37    13_416.31       0.2207             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             12_937.93       501.36    13_439.29       0.2190             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             12_937.93       485.55    13_423.48       0.2165             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            12_937.93       604.36    13_542.29       0.6314          1.0820         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            12_937.93       701.03    13_638.96       0.7819          1.0347         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            12_937.93       580.37    13_518.30       0.6237          1.0951         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            12_937.93       680.88    13_618.81       0.7726          1.0399         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            12_937.93       587.44    13_525.37       0.6123          1.1199         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            12_937.93       684.99    13_622.92       0.7580          1.0518         4.49
IVF-Binary-512-nl223-pca (self)                       12_937.93     2_011.99    14_949.93       0.6321          1.0984         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             13_319.79       494.20    13_813.99       0.2204             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             13_319.79       497.78    13_817.57       0.2194             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             13_319.79       511.60    13_831.39       0.2169             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            13_319.79       593.26    13_913.05       0.6306          1.0827         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            13_319.79       690.11    14_009.90       0.7810          1.0349         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            13_319.79       599.01    13_918.80       0.6269          1.0886         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            13_319.79       691.54    14_011.33       0.7763          1.0372         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            13_319.79       598.83    13_918.61       0.6155          1.1131         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            13_319.79       699.48    14_019.27       0.7620          1.0479         4.67
IVF-Binary-512-nl316-pca (self)                       13_319.79     1_909.32    15_229.11       0.6352          1.0922         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          25_911.53       905.48    26_817.01       0.1488             NaN         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)         25_911.53       890.69    26_802.22       0.1488             NaN         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)         25_911.53       898.96    26_810.49       0.1488             NaN         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)         25_911.53       977.50    26_889.03       0.4139          1.0699         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)         25_911.53     1_059.31    26_970.84       0.5533          1.0396         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)        25_911.53       969.79    26_881.32       0.4139          1.0699         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)        25_911.53     1_064.82    26_976.35       0.5533          1.0396         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)        25_911.53       977.42    26_888.96       0.4139          1.0699         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)        25_911.53     1_075.97    26_987.50       0.5533          1.0396         8.42
IVF-Binary-1024-nl158-random (self)                   25_911.53     3_159.89    29_071.42       0.4278          1.0760         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)         24_096.35       900.31    24_996.66       0.1538             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         24_096.35       923.64    25_019.99       0.1537             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         24_096.35       917.47    25_013.82       0.1537             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        24_096.35       992.54    25_088.89       0.4216          1.0676         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        24_096.35     1_076.28    25_172.63       0.5608          1.0383         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        24_096.35       991.10    25_087.45       0.4212          1.0677         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        24_096.35     1_078.18    25_174.53       0.5603          1.0384         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        24_096.35     1_014.00    25_110.35       0.4212          1.0677         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        24_096.35     1_112.07    25_208.42       0.5603          1.0384         8.54
IVF-Binary-1024-nl223-random (self)                   24_096.35     3_315.88    27_412.23       0.4359          1.0736         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         24_468.83       921.10    25_389.94       0.1556             NaN         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)         24_468.83       930.33    25_399.16       0.1556             NaN         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)         24_468.83       930.07    25_398.90       0.1556             NaN         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)        24_468.83     1_012.65    25_481.48       0.4245          1.0669         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)        24_468.83     1_104.01    25_572.84       0.5636          1.0379         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)        24_468.83     1_014.11    25_482.95       0.4243          1.0669         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)        24_468.83     1_110.18    25_579.01       0.5634          1.0380         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)        24_468.83     1_019.13    25_487.96       0.4241          1.0670         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)        24_468.83     1_145.04    25_613.87       0.5632          1.0380         8.73
IVF-Binary-1024-nl316-random (self)                   24_468.83     3_262.37    27_731.20       0.4383          1.0729         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)             26_356.51       911.20    27_267.71       0.2433             NaN         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            26_356.51       913.41    27_269.93       0.2394             NaN         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            26_356.51       933.56    27_290.08       0.2388             NaN         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            26_356.51     1_020.36    27_376.87       0.6896          1.0422         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            26_356.51     1_084.32    27_440.84       0.8328          1.0183         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           26_356.51       998.46    27_354.98       0.6768          1.0450         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           26_356.51     1_097.22    27_453.74       0.8247          1.0193         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           26_356.51     1_000.14    27_356.65       0.6744          1.0455         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           26_356.51     1_104.27    27_460.78       0.8224          1.0196         8.42
IVF-Binary-1024-nl158-pca (self)                      26_356.51     3_235.98    29_592.49       0.6844          1.0483         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            24_445.56       906.48    25_352.04       0.2400             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            24_445.56       909.82    25_355.38       0.2394             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            24_445.56       945.07    25_390.63       0.2390             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           24_445.56     1_003.95    25_449.51       0.6780          1.0436         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           24_445.56     1_106.08    25_551.63       0.8260          1.0186         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           24_445.56     1_002.41    25_447.97       0.6761          1.0445         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           24_445.56     1_092.02    25_537.58       0.8243          1.0190         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           24_445.56     1_010.33    25_455.89       0.6744          1.0450         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           24_445.56     1_101.27    25_546.83       0.8225          1.0193         8.54
IVF-Binary-1024-nl223-pca (self)                      24_445.56     3_297.72    27_743.28       0.6841          1.0477         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            24_755.11       918.51    25_673.62       0.2398             NaN         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            24_755.11       922.23    25_677.34       0.2395             NaN         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            24_755.11       939.50    25_694.60       0.2389             NaN         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           24_755.11     1_016.68    25_771.79       0.6773          1.0437         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           24_755.11     1_105.92    25_861.03       0.8256          1.0186         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           24_755.11     1_010.21    25_765.32       0.6766          1.0441         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           24_755.11     1_107.89    25_863.00       0.8250          1.0188         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           24_755.11     1_046.86    25_801.97       0.6744          1.0449         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           24_755.11     1_181.42    25_936.53       0.8228          1.0192         8.73
IVF-Binary-1024-nl316-pca (self)                      24_755.11     3_336.97    28_092.08       0.6846          1.0474         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)              2_935.12       459.20     3_394.33       0.0796             NaN         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)             2_935.12       487.12     3_422.25       0.0771             NaN         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)             2_935.12       520.54     3_455.66       0.0767             NaN         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)             2_935.12       510.55     3_445.68       0.8037          1.0347         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)             2_935.12       909.90     3_845.03       0.9345          1.0088         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)            2_935.12       541.30     3_476.43       0.6754          1.3610         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)            2_935.12       927.47     3_862.60       0.9105          1.0120         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)            2_935.12       560.78     3_495.90       0.5847          1.4920         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)            2_935.12       963.44     3_898.56       0.8762          1.0176         3.36
IVF-Binary-512-nl158-sign (self)                       2_935.12     1_659.12     4_594.24       0.7095          1.2592         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)             1_195.82       498.12     1_693.95       0.0995             NaN         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)             1_195.82       517.54     1_713.36       0.0808             NaN         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)             1_195.82       566.58     1_762.40       0.0726             NaN         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)            1_195.82       544.96     1_740.79       0.6676          1.1939         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)            1_195.82       956.35     2_152.18       0.9123          1.0093         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)            1_195.82       562.55     1_758.38       0.5712          1.3939         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)            1_195.82       960.12     2_155.94       0.8816          1.0126         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)            1_195.82       603.32     1_799.14       0.4422          2.2451         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)            1_195.82     1_006.95     2_202.77       0.8070          1.0209         3.49
IVF-Binary-512-nl223-sign (self)                       1_195.82     1_756.00     2_951.82       0.6193          1.1591         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)             1_643.92       537.93     2_181.85       0.0939             NaN         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)             1_643.92       566.05     2_209.97       0.0896             NaN         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)             1_643.92       620.36     2_264.28       0.0742             NaN         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)            1_643.92       587.49     2_231.41       0.6831          1.0516         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)            1_643.92       986.32     2_630.24       0.9004          1.0114         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)            1_643.92       598.53     2_242.45       0.6316          1.0912         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)            1_643.92     1_001.71     2_645.63       0.8838          1.0132         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)            1_643.92       640.87     2_284.79       0.5121          1.4460         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)            1_643.92     1_128.12     2_772.04       0.8252          1.0212         3.67
IVF-Binary-512-nl316-sign (self)                       1_643.92     1_936.68     3_580.60       0.6725          1.0535         3.67
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       102.56    16_082.56    16_185.12       1.0000          1.0000       146.48
Exhaustive (self)                                        102.56    54_027.53    54_130.09       1.0000          1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)              9_078.00       540.67     9_618.67       0.0560             NaN         2.28
ExhaustiveBinary-256-random-rf10 (query)               9_078.00       690.90     9_768.90       0.2347          1.1233         2.28
ExhaustiveBinary-256-random-rf20 (query)               9_078.00       841.91     9_919.91       0.3417          1.0805         2.28
ExhaustiveBinary-256-random (self)                     9_078.00     2_223.04    11_301.04       0.2470          1.1218         2.28
ExhaustiveBinary-256-pca_no_rr (query)                 9_546.17       536.29    10_082.46       0.1692             NaN         2.28
ExhaustiveBinary-256-pca-rf10 (query)                  9_546.17       708.09    10_254.26       0.4636          1.1896         2.28
ExhaustiveBinary-256-pca-rf20 (query)                  9_546.17       875.62    10_421.78       0.5918          1.1225         2.28
ExhaustiveBinary-256-pca (self)                        9_546.17     2_325.34    11_871.51       0.4806          1.1914         2.28
ExhaustiveBinary-512-random_no_rr (query)             17_845.06       913.25    18_758.31       0.0796             NaN         4.55
ExhaustiveBinary-512-random-rf10 (query)              17_845.06     1_063.78    18_908.83       0.2823          1.0957         4.55
ExhaustiveBinary-512-random-rf20 (query)              17_845.06     1_226.39    19_071.44       0.3932          1.0588         4.55
ExhaustiveBinary-512-random (self)                    17_845.06     3_526.61    21_371.67       0.2970          1.0894         4.55
ExhaustiveBinary-512-pca_no_rr (query)                18_116.65       914.34    19_030.98       0.1883             NaN         4.55
ExhaustiveBinary-512-pca-rf10 (query)                 18_116.65     1_088.36    19_205.00       0.4652          1.2549         4.55
ExhaustiveBinary-512-pca-rf20 (query)                 18_116.65     1_261.52    19_378.16       0.5760          1.1920         4.55
ExhaustiveBinary-512-pca (self)                       18_116.65     3_593.67    21_710.31       0.4711          1.2568         4.55
ExhaustiveBinary-1024-random_no_rr (query)            35_414.48     1_705.51    37_120.00       0.1220             NaN         9.11
ExhaustiveBinary-1024-random-rf10 (query)             35_414.48     1_906.71    37_321.19       0.3369          1.0686         9.11
ExhaustiveBinary-1024-random-rf20 (query)             35_414.48     2_049.57    37_464.05       0.4565          1.0414         9.11
ExhaustiveBinary-1024-random (self)                   35_414.48     6_229.78    41_644.27       0.3446          1.0715         9.11
ExhaustiveBinary-1024-pca_no_rr (query)               35_539.71     1_714.44    37_254.14       0.2682             NaN         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                35_539.71     1_898.44    37_438.15       0.7078          1.0551         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                35_539.71     2_069.91    37_609.62       0.8379          1.0248         9.11
ExhaustiveBinary-1024-pca (self)                      35_539.71     6_262.05    41_801.76       0.7159          1.0572         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  194.98       942.94     1_137.92       0.1170             NaN         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   194.98     1_035.02     1_230.01       0.3259          1.0737         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   194.98     1_684.77     1_879.75       0.4466          1.0437         4.58
ExhaustiveBinary-768-sign (self)                         194.98     3_458.60     3_653.58       0.3337          1.0756         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)           13_452.00       371.40    13_823.39       0.0585             NaN         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)          13_452.00       375.51    13_827.51       0.0585             NaN         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)          13_452.00       376.64    13_828.64       0.0585             NaN         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)          13_452.00       494.43    13_946.42       0.2380          1.1221         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)          13_452.00       589.61    14_041.60       0.3441          1.0797         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)         13_452.00       481.67    13_933.67       0.2373          1.1222         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)         13_452.00       591.72    14_043.71       0.3436          1.0798         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)         13_452.00       487.65    13_939.65       0.2373          1.1222         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)         13_452.00       591.60    14_043.60       0.3436          1.0798         2.74
IVF-Binary-256-nl158-random (self)                    13_452.00     1_469.98    14_921.97       0.2497          1.1205         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)          10_469.06       395.46    10_864.51       0.0688             NaN         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)          10_469.06       399.49    10_868.55       0.0688             NaN         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)          10_469.06       399.93    10_868.98       0.0688             NaN         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)         10_469.06       505.38    10_974.44       0.2672          1.1034         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)         10_469.06       611.36    11_080.41       0.3745          1.0667         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)         10_469.06       502.88    10_971.94       0.2671          1.1034         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)         10_469.06       621.57    11_090.63       0.3743          1.0667         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)         10_469.06       505.11    10_974.16       0.2671          1.1034         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)         10_469.06       616.54    11_085.60       0.3743          1.0667         2.93
IVF-Binary-256-nl223-random (self)                    10_469.06     1_549.02    12_018.08       0.2801          1.0992         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)          11_144.31       420.27    11_564.58       0.0754             NaN         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)          11_144.31       425.46    11_569.77       0.0754             NaN         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)          11_144.31       458.81    11_603.12       0.0753             NaN         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)         11_144.31       534.13    11_678.44       0.2807          1.0951         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)         11_144.31       635.13    11_779.43       0.3902          1.0611         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)         11_144.31       532.72    11_677.03       0.2805          1.0952         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)         11_144.31       644.01    11_788.32       0.3898          1.0611         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)         11_144.31       544.13    11_688.44       0.2803          1.0952         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)         11_144.31       645.73    11_790.03       0.3897          1.0612         3.21
IVF-Binary-256-nl316-random (self)                    11_144.31     1_648.81    12_793.12       0.2939          1.0898         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)              13_869.85       378.10    14_247.95       0.1803             NaN         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)             13_869.85       386.33    14_256.18       0.1753             NaN         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)             13_869.85       382.91    14_252.76       0.1744             NaN         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)             13_869.85       514.76    14_384.61       0.5385          1.0692         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)             13_869.85       627.97    14_497.82       0.6977          1.0352         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)            13_869.85       512.23    14_382.08       0.5175          1.0847         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)            13_869.85       659.76    14_529.61       0.6790          1.0396         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)            13_869.85       517.71    14_387.57       0.5130          1.0938         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)            13_869.85       644.81    14_514.66       0.6727          1.0428         2.74
IVF-Binary-256-nl158-pca (self)                       13_869.85     1_716.14    15_585.99       0.5384          1.0860         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)             11_043.92       395.89    11_439.81       0.1765             NaN         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)             11_043.92       398.50    11_442.41       0.1754             NaN         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)             11_043.92       398.26    11_442.17       0.1746             NaN         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)            11_043.92       536.39    11_580.31       0.5229          1.0734         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)            11_043.92       646.82    11_690.74       0.6854          1.0351         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)            11_043.92       528.02    11_571.94       0.5180          1.0797         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)            11_043.92       646.66    11_690.57       0.6790          1.0374         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)            11_043.92       529.28    11_573.19       0.5136          1.0881         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)            11_043.92       647.37    11_691.29       0.6725          1.0408         2.93
IVF-Binary-256-nl223-pca (self)                       11_043.92     1_679.14    12_723.05       0.5387          1.0806         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)             11_759.71       429.16    12_188.87       0.1765             NaN         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)             11_759.71       427.89    12_187.60       0.1760             NaN         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)             11_759.71       423.88    12_183.59       0.1750             NaN         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)            11_759.71       558.28    12_318.00       0.5213          1.0747         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)            11_759.71       667.90    12_427.62       0.6833          1.0358         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)            11_759.71       555.42    12_315.13       0.5190          1.0772         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)            11_759.71       675.74    12_435.45       0.6804          1.0369         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)            11_759.71       559.47    12_319.18       0.5142          1.0870         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)            11_759.71       671.41    12_431.12       0.6734          1.0404         3.21
IVF-Binary-256-nl316-pca (self)                       11_759.71     1_745.20    13_504.91       0.5398          1.0785         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)           22_302.71       700.49    23_003.20       0.0808             NaN         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)          22_302.71       703.17    23_005.89       0.0808             NaN         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)          22_302.71       721.83    23_024.54       0.0808             NaN         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)          22_302.71       810.05    23_112.76       0.2830          1.0955         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)          22_302.71       914.11    23_216.83       0.3936          1.0587         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)         22_302.71       812.59    23_115.30       0.2829          1.0955         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)         22_302.71       926.96    23_229.67       0.3936          1.0587         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)         22_302.71       817.71    23_120.43       0.2829          1.0955         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)         22_302.71       929.00    23_231.72       0.3936          1.0587         5.02
IVF-Binary-512-nl158-random (self)                    22_302.71     2_582.21    24_884.92       0.2976          1.0892         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)          19_564.69       720.82    20_285.52       0.0910             NaN         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)          19_564.69       719.91    20_284.61       0.0910             NaN         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)          19_564.69       725.67    20_290.37       0.0910             NaN         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)         19_564.69       835.00    20_399.69       0.2987          1.0874         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)         19_564.69       939.07    20_503.76       0.4078          1.0544         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)         19_564.69       827.08    20_391.77       0.2987          1.0874         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)         19_564.69       945.50    20_510.19       0.4077          1.0544         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)         19_564.69       859.19    20_423.88       0.2987          1.0874         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)         19_564.69       946.68    20_511.37       0.4077          1.0544         5.21
IVF-Binary-512-nl223-random (self)                    19_564.69     2_650.66    22_215.35       0.3110          1.0828         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)          20_049.37       749.73    20_799.10       0.0960             NaN         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)          20_049.37       748.51    20_797.88       0.0959             NaN         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)          20_049.37       747.89    20_797.26       0.0959             NaN         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)         20_049.37       853.18    20_902.55       0.3053          1.0838         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)         20_049.37       959.40    21_008.77       0.4150          1.0524         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)         20_049.37       856.07    20_905.44       0.3051          1.0839         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)         20_049.37       971.61    21_020.98       0.4147          1.0524         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)         20_049.37       862.39    20_911.76       0.3050          1.0839         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)         20_049.37       966.31    21_015.68       0.4145          1.0525         5.48
IVF-Binary-512-nl316-random (self)                    20_049.37     2_749.43    22_798.80       0.3169          1.0799         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)              22_717.08       714.26    23_431.34       0.2345             NaN         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)             22_717.08       711.74    23_428.83       0.2283             NaN         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)             22_717.08       710.84    23_427.93       0.2269             NaN         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)             22_717.08       828.44    23_545.52       0.6556          1.0604         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)             22_717.08       950.89    23_667.97       0.7994          1.0291         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)            22_717.08       826.63    23_543.72       0.6312          1.0810         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)            22_717.08       939.51    23_656.60       0.7780          1.0353         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)            22_717.08       839.55    23_556.63       0.6244          1.0965         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)            22_717.08       953.22    23_670.31       0.7691          1.0413         5.02
IVF-Binary-512-nl158-pca (self)                       22_717.08     2_682.64    25_399.73       0.6462          1.0829         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)             19_786.06       722.70    20_508.76       0.2298             NaN         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)             19_786.06       725.20    20_511.26       0.2282             NaN         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)             19_786.06       733.53    20_519.59       0.2266             NaN         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)            19_786.06       849.62    20_635.68       0.6370          1.0674         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)            19_786.06       961.81    20_747.87       0.7861          1.0302         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)            19_786.06       847.81    20_633.87       0.6303          1.0771         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)            19_786.06       961.49    20_747.56       0.7773          1.0337         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)            19_786.06       855.07    20_641.13       0.6219          1.0941         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)            19_786.06       973.41    20_759.47       0.7663          1.0404         5.21
IVF-Binary-512-nl223-pca (self)                       19_786.06     2_748.84    22_534.90       0.6449          1.0785         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)             20_429.89       748.42    21_178.31       0.2293             NaN         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)             20_429.89       759.86    21_189.74       0.2286             NaN         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)             20_429.89       748.50    21_178.39       0.2268             NaN         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)            20_429.89       876.50    21_306.39       0.6351          1.0692         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)            20_429.89       990.80    21_420.69       0.7833          1.0307         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)            20_429.89       879.78    21_309.66       0.6320          1.0731         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)            20_429.89       995.05    21_424.93       0.7795          1.0322         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)            20_429.89       877.86    21_307.74       0.6233          1.0907         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)            20_429.89       997.06    21_426.94       0.7683          1.0390         5.48
IVF-Binary-512-nl316-pca (self)                       20_429.89     2_830.47    23_260.35       0.6466          1.0745         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)          39_832.79     1_372.55    41_205.34       0.1224             NaN         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)         39_832.79     1_403.34    41_236.13       0.1224             NaN         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)         39_832.79     1_389.00    41_221.79       0.1224             NaN         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)         39_832.79     1_461.39    41_294.19       0.3372          1.0685         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)         39_832.79     1_579.45    41_412.25       0.4567          1.0414         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)        39_832.79     1_478.16    41_310.95       0.3372          1.0685         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)        39_832.79     1_586.74    41_419.53       0.4567          1.0414         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)        39_832.79     1_463.25    41_296.04       0.3372          1.0685         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)        39_832.79     1_573.56    41_406.35       0.4567          1.0414         9.57
IVF-Binary-1024-nl158-random (self)                   39_832.79     4_752.81    44_585.60       0.3448          1.0714         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)         36_943.39     1_446.35    38_389.73       0.1271             NaN         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)         36_943.39     1_374.48    38_317.86       0.1271             NaN         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)         36_943.39     1_378.33    38_321.71       0.1271             NaN         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)        36_943.39     1_486.32    38_429.71       0.3436          1.0664         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)        36_943.39     1_657.83    38_601.22       0.4624          1.0401         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)        36_943.39     1_540.45    38_483.84       0.3435          1.0664         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)        36_943.39     1_613.07    38_556.46       0.4623          1.0401         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)        36_943.39     1_498.41    38_441.80       0.3435          1.0664         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)        36_943.39     1_597.17    38_540.56       0.4623          1.0401         9.76
IVF-Binary-1024-nl223-random (self)                   36_943.39     4_817.49    41_760.87       0.3508          1.0696         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)         37_509.33     1_392.24    38_901.58       0.1287             NaN        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)         37_509.33     1_395.90    38_905.23       0.1287             NaN        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)         37_509.33     1_396.26    38_905.60       0.1287             NaN        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)        37_509.33     1_504.40    39_013.73       0.3467          1.0653        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)        37_509.33     1_617.44    39_126.77       0.4660          1.0395        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)        37_509.33     1_523.30    39_032.64       0.3465          1.0653        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)        37_509.33     1_639.75    39_149.08       0.4657          1.0396        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)        37_509.33     1_595.07    39_104.41       0.3464          1.0653        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)        37_509.33     1_650.12    39_159.45       0.4656          1.0396        10.04
IVF-Binary-1024-nl316-random (self)                   37_509.33     4_899.54    42_408.88       0.3541          1.0686        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)             40_101.42     1_355.85    41_457.27       0.2750             NaN         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)            40_101.42     1_359.73    41_461.15       0.2697             NaN         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)            40_101.42     1_361.73    41_463.14       0.2687             NaN         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)            40_101.42     1_477.07    41_578.48       0.7313          1.0439         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)            40_101.42     1_581.88    41_683.29       0.8585          1.0204         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)           40_101.42     1_501.13    41_602.54       0.7143          1.0492         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)           40_101.42     1_611.22    41_712.64       0.8465          1.0219         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)           40_101.42     1_504.59    41_606.01       0.7107          1.0512         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)           40_101.42     1_662.31    41_763.72       0.8423          1.0228         9.57
IVF-Binary-1024-nl158-pca (self)                      40_101.42     4_856.30    44_957.71       0.7224          1.0510         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)            37_133.73     1_374.36    38_508.09       0.2708             NaN         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)            37_133.73     1_387.72    38_521.45       0.2697             NaN         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)            37_133.73     1_376.96    38_510.69       0.2690             NaN         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)           37_133.73     1_507.07    38_640.80       0.7179          1.0461         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)           37_133.73     1_618.40    38_752.13       0.8511          1.0205         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)           37_133.73     1_493.56    38_627.29       0.7138          1.0483         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)           37_133.73     1_613.36    38_747.09       0.8465          1.0214         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)           37_133.73     1_519.53    38_653.26       0.7108          1.0503         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)           37_133.73     1_621.17    38_754.90       0.8428          1.0223         9.76
IVF-Binary-1024-nl223-pca (self)                      37_133.73     5_052.98    42_186.71       0.7221          1.0498         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)            37_808.65     1_398.57    39_207.22       0.2703             NaN        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)            37_808.65     1_400.05    39_208.70       0.2699             NaN        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)            37_808.65     1_401.92    39_210.57       0.2689             NaN        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)           37_808.65     1_511.69    39_320.35       0.7160          1.0467        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)           37_808.65     1_623.33    39_431.98       0.8490          1.0207        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)           37_808.65     1_546.06    39_354.71       0.7146          1.0477        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)           37_808.65     1_634.99    39_443.65       0.8473          1.0211        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)           37_808.65     1_529.87    39_338.52       0.7113          1.0501        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)           37_808.65     1_636.40    39_445.05       0.8431          1.0222        10.04
IVF-Binary-1024-nl316-pca (self)                      37_808.65     4_971.82    42_780.47       0.7230          1.0491        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)              4_596.67       636.13     5_232.79       0.0781             NaN         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)             4_596.67       681.47     5_278.14       0.0770             NaN         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)             4_596.67       727.04     5_323.70       0.0768             NaN         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)             4_596.67       715.73     5_312.39       0.7120          1.1279         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)             4_596.67     1_283.52     5_880.19       0.9139          1.0072         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)            4_596.67       751.27     5_347.94       0.5244          3.5419         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)            4_596.67     1_403.94     6_000.60       0.8718          1.0107         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)            4_596.67       790.63     5_387.30       0.4117          4.8459         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)            4_596.67     1_360.00     5_956.67       0.8250          1.0154         5.04
IVF-Binary-768-nl158-sign (self)                       4_596.67     2_323.09     6_919.75       0.5477          3.5673         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)             1_716.94       688.29     2_405.23       0.0864             NaN         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)             1_716.94       716.31     2_433.25       0.0741             NaN         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)             1_716.94       783.18     2_500.12       0.0728             NaN         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)            1_716.94       764.56     2_481.50       0.5428          2.2671         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)            1_716.94     1_333.40     3_050.35       0.8752          1.0102         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)            1_716.94       784.70     2_501.64       0.4389          2.7508         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)            1_716.94     1_355.87     3_072.82       0.8397          1.0132         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)            1_716.94       845.51     2_562.45       0.3225          3.5334         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)            1_716.94     1_406.07     3_123.01       0.7498          1.0214         5.23
IVF-Binary-768-nl223-sign (self)                       1_716.94     2_471.94     4_188.88       0.4691          2.6701         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)             2_277.43       755.83     3_033.26       0.0847             NaN         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)             2_277.43       773.12     3_050.55       0.0841             NaN         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)             2_277.43       842.11     3_119.55       0.0714             NaN         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)            2_277.43       816.96     3_094.39       0.5462          1.6004         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)            2_277.43     1_368.97     3_646.41       0.8628          1.0120         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)            2_277.43       829.57     3_107.00       0.4894          1.9189         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)            2_277.43     1_382.73     3_660.17       0.8424          1.0138         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)            2_277.43       889.20     3_166.63       0.3652          2.4917         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)            2_277.43     1_452.63     3_730.06       0.7692          1.0209         5.51
IVF-Binary-768-nl316-sign (self)                       2_277.43     2_628.43     4_905.87       0.5190          1.7958         5.51
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

#### Cell embeddings

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        33.06     4_065.76     4_098.82       1.0000          1.0000        48.83
Exhaustive (self)                                         33.06    13_483.16    13_516.23       1.0000          1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_692.32       281.83     2_974.15       0.5519             NaN         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_692.32       398.88     3_091.20       0.9881          1.0022         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_692.32       505.25     3_197.56       0.9980          1.0003         1.78
ExhaustiveBinary-256-random (self)                     2_692.32     1_309.05     4_001.37       0.9881          1.0022         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_806.92       284.01     3_090.94       0.1183             NaN         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_806.92       391.80     3_198.73       0.3215          1.9397         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_806.92       492.97     3_299.89       0.4181          1.5883         1.78
ExhaustiveBinary-256-pca (self)                        2_806.92     1_286.70     4_093.62       0.3192          1.9523         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_316.96       446.71     5_763.67       0.6305             NaN         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_316.96       563.68     5_880.64       0.9975          1.0004         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_316.96       673.22     5_990.18       0.9998          1.0000         3.55
ExhaustiveBinary-512-random (self)                     5_316.96     1_855.52     7_172.48       0.9973          1.0004         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_444.75       445.61     5_890.36       0.3665             NaN         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_444.75       561.92     6_006.67       0.8453          1.0657         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_444.75       669.03     6_113.78       0.9396          1.0206         3.55
ExhaustiveBinary-512-pca (self)                        5_444.75     1_854.83     7_299.58       0.8325          1.0737         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_529.34       780.91    11_310.25       0.6758             NaN         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_529.34       892.85    11_422.19       0.9995          1.0001         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_529.34     1_009.94    11_539.28       0.9999          1.0000         7.10
ExhaustiveBinary-1024-random (self)                   10_529.34     2_960.84    13_490.17       0.9993          1.0001         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_571.04       780.01    11_351.05       0.5577             NaN         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_571.04       898.66    11_469.70       0.9880          1.0028         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_571.04     1_019.35    11_590.39       0.9987          1.0003         7.10
ExhaustiveBinary-1024-pca (self)                      10_571.04     2_956.17    13_527.21       0.9861          1.0033         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   54.84       475.49       530.33       0.0376             NaN         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    54.84       501.24       556.08       0.1617          2.7567         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    54.84       808.11       862.95       0.2739          1.9837         1.53
ExhaustiveBinary-256-sign (self)                          54.84     1_654.38     1_709.22       0.1691          2.7353         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)            4_127.88       127.92     4_255.80       0.5643             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           4_127.88       134.27     4_262.15       0.5580             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           4_127.88       143.98     4_271.86       0.5561             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           4_127.88       192.05     4_319.93       0.9901          1.0018         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           4_127.88       244.14     4_372.02       0.9966          1.0007         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          4_127.88       199.75     4_327.64       0.9903          1.0017         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          4_127.88       256.58     4_384.46       0.9985          1.0002         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          4_127.88       206.75     4_334.64       0.9895          1.0019         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          4_127.88       273.23     4_401.11       0.9984          1.0002         1.93
IVF-Binary-256-nl158-random (self)                     4_127.88       620.32     4_748.21       0.9902          1.0017         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           3_172.23       130.16     3_302.39       0.5623             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           3_172.23       134.05     3_306.28       0.5598             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           3_172.23       141.20     3_313.43       0.5573             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          3_172.23       193.02     3_365.25       0.9910          1.0015         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          3_172.23       245.43     3_417.66       0.9983          1.0002         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          3_172.23       195.10     3_367.33       0.9906          1.0016         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          3_172.23       249.14     3_421.37       0.9986          1.0002         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          3_172.23       201.81     3_374.04       0.9896          1.0018         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          3_172.23       258.66     3_430.89       0.9984          1.0002         2.00
IVF-Binary-256-nl223-random (self)                     3_172.23       653.48     3_825.71       0.9904          1.0017         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           3_315.91       147.38     3_463.29       0.5621             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           3_315.91       140.62     3_456.53       0.5610             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           3_315.91       146.02     3_461.93       0.5582             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          3_315.91       197.59     3_513.50       0.9911          1.0016         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          3_315.91       265.49     3_581.40       0.9986          1.0002         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          3_315.91       198.96     3_514.88       0.9908          1.0016         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          3_315.91       252.40     3_568.31       0.9986          1.0002         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          3_315.91       203.99     3_519.90       0.9899          1.0018         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          3_315.91       259.88     3_575.79       0.9985          1.0002         2.09
IVF-Binary-256-nl316-random (self)                     3_315.91       604.61     3_920.52       0.9908          1.0016         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               4_227.73       125.79     4_353.51       0.1526             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              4_227.73       134.71     4_362.44       0.1376             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              4_227.73       143.14     4_370.87       0.1312             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              4_227.73       208.24     4_435.96       0.4905          1.4114         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              4_227.73       256.45     4_484.18       0.6393          1.2170         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             4_227.73       207.22     4_434.95       0.4273          1.5364         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             4_227.73       274.10     4_501.83       0.5676          1.2972         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             4_227.73       227.89     4_455.62       0.3950          1.6190         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             4_227.73       292.48     4_520.21       0.5274          1.3518         1.93
IVF-Binary-256-nl158-pca (self)                        4_227.73       661.03     4_888.76       0.4249          1.5443         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              3_257.26       131.53     3_388.80       0.1485             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              3_257.26       133.90     3_391.16       0.1425             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              3_257.26       142.04     3_399.31       0.1343             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             3_257.26       197.39     3_454.65       0.4802          1.4241         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             3_257.26       257.89     3_515.16       0.6314          1.2221         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             3_257.26       202.38     3_459.65       0.4522          1.4777         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             3_257.26       265.07     3_522.34       0.6001          1.2555         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             3_257.26       213.66     3_470.93       0.4119          1.5715         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             3_257.26       281.67     3_538.94       0.5507          1.3172         2.00
IVF-Binary-256-nl223-pca (self)                        3_257.26       638.86     3_896.12       0.4498          1.4839         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              3_405.04       139.43     3_544.47       0.1484             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              3_405.04       141.64     3_546.68       0.1449             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              3_405.04       148.21     3_553.25       0.1367             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             3_405.04       205.36     3_610.40       0.4799          1.4229         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             3_405.04       263.72     3_668.76       0.6330          1.2199         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             3_405.04       206.41     3_611.46       0.4655          1.4491         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             3_405.04       267.55     3_672.59       0.6164          1.2366         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             3_405.04       214.64     3_619.68       0.4256          1.5342         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             3_405.04       280.88     3_685.92       0.5689          1.2914         2.09
IVF-Binary-256-nl316-pca (self)                        3_405.04       647.13     4_052.18       0.4630          1.4549         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_729.57       221.17     6_950.74       0.6403             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_729.57       235.22     6_964.79       0.6347             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_729.57       244.91     6_974.48       0.6329             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_729.57       290.49     7_020.06       0.9962          1.0007         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_729.57       343.66     7_073.22       0.9975          1.0005         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_729.57       295.46     7_025.03       0.9980          1.0003         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_729.57       355.47     7_085.04       0.9998          1.0000         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_729.57       307.96     7_037.52       0.9978          1.0003         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_729.57       377.26     7_106.83       0.9998          1.0000         3.71
IVF-Binary-512-nl158-random (self)                     6_729.57       951.05     7_680.62       0.9979          1.0003         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_753.54       231.72     5_985.27       0.6379             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_753.54       250.64     6_004.19       0.6357             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_753.54       242.35     5_995.89       0.6336             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_753.54       286.95     6_040.50       0.9976          1.0003         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_753.54       338.92     6_092.47       0.9993          1.0001         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_753.54       293.12     6_046.66       0.9979          1.0003         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_753.54       345.50     6_099.04       0.9997          1.0000         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_753.54       299.36     6_052.91       0.9978          1.0003         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_753.54       358.19     6_111.74       0.9998          1.0000         3.77
IVF-Binary-512-nl223-random (self)                     5_753.54       915.35     6_668.89       0.9978          1.0003         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           5_864.14       232.91     6_097.06       0.6377             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           5_864.14       239.95     6_104.09       0.6368             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           5_864.14       244.61     6_108.76       0.6344             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          5_864.14       293.31     6_157.45       0.9979          1.0003         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          5_864.14       348.01     6_212.16       0.9996          1.0001         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          5_864.14       293.75     6_157.89       0.9980          1.0003         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          5_864.14       351.77     6_215.91       0.9997          1.0000         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          5_864.14       301.02     6_165.16       0.9978          1.0003         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          5_864.14       359.38     6_223.52       0.9998          1.0000         3.86
IVF-Binary-512-nl316-random (self)                     5_864.14       924.04     6_788.18       0.9979          1.0003         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               6_785.99       222.30     7_008.29       0.3819             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              6_785.99       234.77     7_020.76       0.3733             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              6_785.99       249.57     7_035.56       0.3705             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              6_785.99       286.23     7_072.23       0.8864          1.0420         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              6_785.99       343.45     7_129.45       0.9653          1.0103         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             6_785.99       301.64     7_087.64       0.8657          1.0534         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             6_785.99       361.08     7_147.07       0.9543          1.0146         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             6_785.99       309.28     7_095.27       0.8565          1.0588         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             6_785.99       374.22     7_160.21       0.9480          1.0171         3.71
IVF-Binary-512-nl158-pca (self)                        6_785.99       984.44     7_770.44       0.8541          1.0600         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              5_857.21       225.66     6_082.87       0.3781             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              5_857.21       239.28     6_096.49       0.3748             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              5_857.21       246.08     6_103.29       0.3714             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             5_857.21       290.68     6_147.89       0.8810          1.0451         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             5_857.21       344.92     6_202.14       0.9631          1.0112         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             5_857.21       291.64     6_148.86       0.8711          1.0504         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             5_857.21       351.78     6_208.99       0.9578          1.0132         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             5_857.21       306.20     6_163.42       0.8591          1.0574         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             5_857.21       364.32     6_221.53       0.9499          1.0164         3.77
IVF-Binary-512-nl223-pca (self)                        5_857.21       931.79     6_789.00       0.8602          1.0566         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              5_968.03       234.44     6_202.47       0.3778             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              5_968.03       236.88     6_204.91       0.3763             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              5_968.03       248.24     6_216.27       0.3724             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             5_968.03       296.87     6_264.90       0.8796          1.0460         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             5_968.03       353.45     6_321.49       0.9626          1.0115         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             5_968.03       303.44     6_271.47       0.8749          1.0483         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             5_968.03       353.53     6_321.57       0.9602          1.0124         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             5_968.03       309.61     6_277.65       0.8626          1.0551         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             5_968.03       368.11     6_336.14       0.9525          1.0154         3.86
IVF-Binary-512-nl316-pca (self)                        5_968.03       940.47     6_908.50       0.8641          1.0545         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_812.56       417.47    12_230.03       0.6842             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_812.56       438.69    12_251.25       0.6791             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_812.56       454.77    12_267.33       0.6774             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_812.56       498.15    12_310.71       0.9974          1.0006         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_812.56       537.34    12_349.90       0.9976          1.0005         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_812.56       496.80    12_309.36       0.9996          1.0000         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_812.56       557.77    12_370.33       0.9999          1.0000         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_812.56       515.46    12_328.03       0.9996          1.0000         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_812.56       580.86    12_393.42       1.0000          1.0000         7.26
IVF-Binary-1024-nl158-random (self)                   11_812.56     1_626.34    13_438.90       0.9995          1.0001         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_875.02       473.62    11_348.64       0.6819             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_875.02       445.72    11_320.75       0.6799             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_875.02       471.53    11_346.55       0.6781             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_875.02       503.24    11_378.27       0.9990          1.0001         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_875.02       574.83    11_449.85       0.9993          1.0001         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_875.02       582.89    11_457.91       0.9995          1.0001         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_875.02       579.30    11_454.32       0.9998          1.0000         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_875.02       530.47    11_405.49       0.9995          1.0001         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_875.02       598.91    11_473.93       0.9999          1.0000         7.32
IVF-Binary-1024-nl223-random (self)                   10_875.02     1_642.75    12_517.77       0.9993          1.0001         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         11_092.50       465.34    11_557.85       0.6813             NaN         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)         11_092.50       466.00    11_558.51       0.6804             NaN         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)         11_092.50       463.73    11_556.23       0.6785             NaN         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)        11_092.50       513.20    11_605.71       0.9993          1.0001         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)        11_092.50       591.88    11_684.38       0.9996          1.0001         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)        11_092.50       578.26    11_670.76       0.9995          1.0001         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)        11_092.50       580.25    11_672.75       0.9998          1.0000         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)        11_092.50       527.49    11_619.99       0.9996          1.0001         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)        11_092.50       594.10    11_686.61       0.9999          1.0000         7.42
IVF-Binary-1024-nl316-random (self)                   11_092.50     1_783.43    12_875.94       0.9994          1.0001         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)             12_383.77       439.52    12_823.29       0.5685             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            12_383.77       456.21    12_839.98       0.5623             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            12_383.77       473.52    12_857.29       0.5605             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            12_383.77       538.58    12_922.36       0.9912          1.0019         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            12_383.77       566.90    12_950.67       0.9972          1.0006         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           12_383.77       524.85    12_908.63       0.9907          1.0020         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           12_383.77       594.02    12_977.79       0.9992          1.0002         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           12_383.77       549.89    12_933.66       0.9896          1.0023         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           12_383.77       618.09    13_001.87       0.9990          1.0002         7.26
IVF-Binary-1024-nl158-pca (self)                      12_383.77     1_699.01    14_082.79       0.9890          1.0025         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            11_039.92       441.70    11_481.62       0.5654             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            11_039.92       451.40    11_491.32       0.5630             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            11_039.92       464.90    11_504.82       0.5608             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           11_039.92       513.51    11_553.43       0.9916          1.0017         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           11_039.92       567.81    11_607.73       0.9987          1.0002         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           11_039.92       511.93    11_551.85       0.9909          1.0019         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           11_039.92       576.76    11_616.68       0.9991          1.0002         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           11_039.92       532.70    11_572.62       0.9897          1.0023         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           11_039.92       603.77    11_643.69       0.9990          1.0002         7.32
IVF-Binary-1024-nl223-pca (self)                      11_039.92     1_662.73    12_702.65       0.9892          1.0024         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            11_209.34       492.15    11_701.48       0.5650             NaN         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            11_209.34       449.99    11_659.32       0.5640             NaN         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            11_209.34       463.81    11_673.14       0.5616             NaN         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           11_209.34       516.16    11_725.50       0.9917          1.0018         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           11_209.34       575.09    11_784.43       0.9990          1.0002         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           11_209.34       514.47    11_723.80       0.9914          1.0018         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           11_209.34       581.11    11_790.44       0.9991          1.0002         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           11_209.34       530.35    11_739.69       0.9902          1.0022         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           11_209.34       620.56    11_829.89       0.9991          1.0002         7.42
IVF-Binary-1024-nl316-pca (self)                      11_209.34     1_714.27    12_923.61       0.9897          1.0023         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)              1_531.01       323.06     1_854.07       0.3698             NaN         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)             1_531.01       357.67     1_888.68       0.3462             NaN         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)             1_531.01       373.04     1_904.05       0.3311             NaN         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)             1_531.01       351.95     1_882.96       0.7366          1.1543         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)             1_531.01       609.25     2_140.27       0.9128          1.0435         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)            1_531.01       383.53     1_914.55       0.6107          1.2802         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)            1_531.01       659.85     2_190.86       0.8374          1.0873         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)            1_531.01       414.19     1_945.21       0.5500          1.3853         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)            1_531.01       703.90     2_234.91       0.7858          1.1266         1.68
IVF-Binary-256-nl158-sign (self)                       1_531.01     1_218.05     2_749.06       0.6111          1.2816         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               539.36       327.11       866.46       0.3263             NaN         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               539.36       341.79       881.14       0.3166             NaN         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               539.36       373.72       913.08       0.2970             NaN         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              539.36       368.84       908.19       0.6563          1.2511         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              539.36       628.83     1_168.18       0.8241          1.1195         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              539.36       379.01       918.37       0.6036          1.3096         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              539.36       646.52     1_185.87       0.7915          1.1454         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              539.36       413.71       953.06       0.5241          1.4387         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              539.36       694.85     1_234.20       0.7304          1.2030         1.75
IVF-Binary-256-nl223-sign (self)                         539.36     1_181.05     1_720.41       0.6050          1.3105         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               687.60       343.64     1_031.24       0.2935             NaN         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               687.60       352.49     1_040.09       0.2884             NaN         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               687.60       389.73     1_077.33       0.2715             NaN         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              687.60       383.85     1_071.46       0.6138          1.3201         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              687.60       645.94     1_333.54       0.7603          1.1809         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              687.60       393.33     1_080.93       0.5883          1.3573         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              687.60       656.80     1_344.40       0.7441          1.1984         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              687.60       427.73     1_115.33       0.5138          1.4884         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              687.60       725.71     1_413.31       0.6932          1.2611         1.84
IVF-Binary-256-nl316-sign (self)                         687.60     1_215.51     1_903.11       0.5887          1.3544         1.84
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        71.04     9_635.32     9_706.36       1.0000          1.0000        97.66
Exhaustive (self)                                         71.04    33_250.99    33_322.03       1.0000          1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_891.21       426.77     6_317.98       0.5547             NaN         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_891.21       575.88     6_467.09       0.9898          1.0017         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_891.21       730.56     6_621.77       0.9985          1.0002         2.03
ExhaustiveBinary-256-random (self)                     5_891.21     1_879.34     7_770.55       0.9900          1.0016         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_268.84       439.15     6_707.99       0.1212             NaN         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_268.84       586.54     6_855.38       0.3407          1.8751         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_268.84       741.99     7_010.83       0.4406          1.5429         2.03
ExhaustiveBinary-256-pca (self)                        6_268.84     1_959.99     8_228.83       0.3366          1.8907         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_941.69       757.91    12_699.61       0.6013             NaN         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_941.69       896.49    12_838.19       0.9977          1.0003         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_941.69     1_050.64    12_992.33       0.9998          1.0000         4.05
ExhaustiveBinary-512-random (self)                    11_941.69     2_969.95    14_911.64       0.9975          1.0003         4.05
ExhaustiveBinary-512-pca_no_rr (query)                12_211.76       736.46    12_948.22       0.1147             NaN         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 12_211.76       884.52    13_096.28       0.2782          2.2254         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 12_211.76     1_005.81    13_217.57       0.3475          1.8252         4.05
ExhaustiveBinary-512-pca (self)                       12_211.76     3_043.17    15_254.93       0.2742          2.2528         4.05
ExhaustiveBinary-1024-random_no_rr (query)            23_877.27     1_275.03    25_152.30       0.6624             NaN         8.11
ExhaustiveBinary-1024-random-rf10 (query)             23_877.27     1_433.55    25_310.82       0.9995          1.0001         8.11
ExhaustiveBinary-1024-random-rf20 (query)             23_877.27     1_603.28    25_480.55       1.0000          1.0000         8.11
ExhaustiveBinary-1024-random (self)                   23_877.27     4_740.44    28_617.71       0.9994          1.0001         8.11
ExhaustiveBinary-1024-pca_no_rr (query)               23_902.64     1_290.78    25_193.42       0.3939             NaN         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                23_902.64     1_433.15    25_335.78       0.8323          1.0743         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                23_902.64     1_607.60    25_510.24       0.9198          1.0285         8.11
ExhaustiveBinary-1024-pca (self)                      23_902.64     4_742.81    28_645.45       0.8160          1.0854         8.11
ExhaustiveBinary-512-sign_no_rr (query)                  142.92       715.23       858.15       0.0400             NaN         3.05
ExhaustiveBinary-512-sign-rf10 (query)                   142.92       774.22       917.14       0.1821          2.5571         3.05
ExhaustiveBinary-512-sign-rf20 (query)                   142.92     1_253.56     1_396.48       0.3140          1.8428         3.05
ExhaustiveBinary-512-sign (self)                         142.92     2_509.76     2_652.68       0.1897          2.5286         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)            9_312.06       261.67     9_573.72       0.5636             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           9_312.06       277.69     9_589.75       0.5602             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           9_312.06       283.81     9_595.87       0.5587             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           9_312.06       360.53     9_672.59       0.9918          1.0013         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           9_312.06       454.18     9_766.24       0.9978          1.0004         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          9_312.06       364.09     9_676.14       0.9916          1.0013         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          9_312.06       464.50     9_776.55       0.9988          1.0001         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          9_312.06       376.33     9_688.38       0.9909          1.0014         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          9_312.06       471.20     9_783.25       0.9988          1.0001         2.34
IVF-Binary-256-nl158-random (self)                     9_312.06     1_128.08    10_440.13       0.9919          1.0012         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           6_923.22       270.34     7_193.55       0.5617             NaN         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)           6_923.22       286.40     7_209.62       0.5605             NaN         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)           6_923.22       282.20     7_205.41       0.5589             NaN         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)          6_923.22       378.67     7_301.89       0.9923          1.0012         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)          6_923.22       461.12     7_384.34       0.9989          1.0001         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)          6_923.22       368.01     7_291.23       0.9919          1.0012         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)          6_923.22       462.44     7_385.66       0.9990          1.0001         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)          6_923.22       379.27     7_302.49       0.9911          1.0014         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)          6_923.22       483.38     7_406.60       0.9988          1.0001         2.47
IVF-Binary-256-nl223-random (self)                     6_923.22     1_126.71     8_049.93       0.9921          1.0012         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)           7_121.73       285.40     7_407.13       0.5613             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           7_121.73       292.54     7_414.27       0.5607             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           7_121.73       302.30     7_424.03       0.5595             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          7_121.73       386.21     7_507.94       0.9922          1.0012         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          7_121.73       476.20     7_597.92       0.9990          1.0001         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          7_121.73       392.42     7_514.15       0.9920          1.0012         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          7_121.73       492.56     7_614.29       0.9990          1.0001         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          7_121.73       398.64     7_520.37       0.9913          1.0014         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          7_121.73       480.16     7_601.88       0.9989          1.0001         2.65
IVF-Binary-256-nl316-random (self)                     7_121.73     1_155.86     8_277.58       0.9922          1.0012         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               9_612.05       262.47     9_874.52       0.1449             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              9_612.05       271.81     9_883.86       0.1357             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              9_612.05       283.31     9_895.35       0.1312             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              9_612.05       367.12     9_979.17       0.4662          1.4653         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              9_612.05       470.81    10_082.86       0.6170          1.2407         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             9_612.05       369.26     9_981.31       0.4254          1.5532         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             9_612.05       474.66    10_086.71       0.5649          1.3025         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             9_612.05       378.95     9_991.00       0.4036          1.6095         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             9_612.05       483.31    10_095.36       0.5374          1.3411         2.34
IVF-Binary-256-nl158-pca (self)                        9_612.05     1_163.96    10_776.00       0.4224          1.5627         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              7_063.50       269.29     7_332.79       0.1420             NaN         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              7_063.50       272.06     7_335.56       0.1380             NaN         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              7_063.50       291.45     7_354.95       0.1324             NaN         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             7_063.50       377.18     7_440.68       0.4569          1.4774         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             7_063.50       470.83     7_534.33       0.6070          1.2481         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             7_063.50       373.11     7_436.61       0.4391          1.5147         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             7_063.50       472.38     7_535.88       0.5851          1.2734         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             7_063.50       380.88     7_444.38       0.4120          1.5814         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             7_063.50       487.99     7_551.49       0.5505          1.3189         2.47
IVF-Binary-256-nl223-pca (self)                        7_063.50     1_154.62     8_218.12       0.4362          1.5225         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              7_370.50       289.29     7_659.79       0.1410             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              7_370.50       295.23     7_665.72       0.1390             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              7_370.50       302.55     7_673.04       0.1341             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             7_370.50       390.12     7_760.62       0.4555          1.4768         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             7_370.50       484.23     7_854.73       0.6070          1.2468         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             7_370.50       403.80     7_774.30       0.4462          1.4963         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             7_370.50       486.74     7_857.23       0.5954          1.2599         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             7_370.50       394.83     7_765.32       0.4208          1.5556         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             7_370.50       499.92     7_870.42       0.5625          1.3010         2.65
IVF-Binary-256-nl316-pca (self)                        7_370.50     1_201.47     8_571.97       0.4438          1.5029         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           15_012.64       478.79    15_491.44       0.6099             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          15_012.64       490.72    15_503.37       0.6061             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          15_012.64       501.62    15_514.26       0.6042             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          15_012.64       566.74    15_579.39       0.9972          1.0004         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          15_012.64       660.09    15_672.74       0.9985          1.0003         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         15_012.64       576.55    15_589.19       0.9983          1.0002         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         15_012.64       670.53    15_683.18       0.9998          1.0000         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         15_012.64       614.81    15_627.45       0.9981          1.0002         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         15_012.64       685.92    15_698.57       0.9998          1.0000         4.36
IVF-Binary-512-nl158-random (self)                    15_012.64     1_828.08    16_840.72       0.9982          1.0002         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          12_525.31       489.41    13_014.71       0.6073             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          12_525.31       506.32    13_031.63       0.6060             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          12_525.31       502.80    13_028.11       0.6042             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         12_525.31       573.96    13_099.27       0.9982          1.0002         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         12_525.31       669.13    13_194.43       0.9996          1.0001         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         12_525.31       575.58    13_100.88       0.9983          1.0002         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         12_525.31       667.15    13_192.46       0.9998          1.0000         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         12_525.31       587.78    13_113.09       0.9980          1.0002         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         12_525.31       684.04    13_209.35       0.9998          1.0000         4.49
IVF-Binary-512-nl223-random (self)                    12_525.31     1_819.47    14_344.78       0.9982          1.0002         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          12_808.60       497.78    13_306.38       0.6071             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          12_808.60       523.54    13_332.14       0.6063             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          12_808.60       511.59    13_320.19       0.6048             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         12_808.60       586.04    13_394.64       0.9984          1.0002         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         12_808.60       673.41    13_482.00       0.9998          1.0000         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         12_808.60       587.11    13_395.71       0.9983          1.0002         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         12_808.60       677.47    13_486.07       0.9998          1.0000         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         12_808.60       595.34    13_403.93       0.9981          1.0002         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         12_808.60       692.78    13_501.38       0.9998          1.0000         4.67
IVF-Binary-512-nl316-random (self)                    12_808.60     1_847.73    14_656.33       0.9982          1.0002         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              15_159.25       487.80    15_647.05       0.1403             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             15_159.25       491.26    15_650.51       0.1308             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             15_159.25       505.57    15_664.82       0.1260             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             15_159.25       569.98    15_729.23       0.4249          1.5449         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             15_159.25       662.33    15_821.59       0.5633          1.3025         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            15_159.25       587.99    15_747.24       0.3820          1.6565         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            15_159.25       690.84    15_850.10       0.5038          1.3864         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            15_159.25       601.61    15_760.86       0.3597          1.7319         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            15_159.25       706.19    15_865.44       0.4726          1.4417         4.36
IVF-Binary-512-nl158-pca (self)                       15_159.25     1_888.37    17_047.62       0.3782          1.6706         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             12_773.33       487.97    13_261.30       0.1373             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             12_773.33       492.44    13_265.77       0.1333             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             12_773.33       512.82    13_286.15       0.1274             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            12_773.33       583.92    13_357.25       0.4162          1.5552         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            12_773.33       673.20    13_446.53       0.5534          1.3090         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            12_773.33       586.75    13_360.08       0.3973          1.6031         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            12_773.33       680.25    13_453.58       0.5275          1.3449         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            12_773.33       594.86    13_368.19       0.3699          1.6876         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            12_773.33       702.54    13_475.87       0.4885          1.4075         4.49
IVF-Binary-512-nl223-pca (self)                       12_773.33     1_882.29    14_655.62       0.3937          1.6148         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             13_056.68       500.16    13_556.85       0.1362             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             13_056.68       501.80    13_558.48       0.1342             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             13_056.68       515.31    13_571.99       0.1289             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            13_056.68       592.63    13_649.31       0.4152          1.5547         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            13_056.68       694.57    13_751.26       0.5522          1.3079         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            13_056.68       598.60    13_655.28       0.4053          1.5792         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            13_056.68       692.17    13_748.86       0.5384          1.3264         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            13_056.68       608.53    13_665.22       0.3791          1.6548         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            13_056.68       706.22    13_762.90       0.5016          1.3833         4.67
IVF-Binary-512-nl316-pca (self)                       13_056.68     1_916.56    14_973.24       0.4012          1.5900         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          26_202.70       902.80    27_105.50       0.6688             NaN         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)         26_202.70       951.51    27_154.21       0.6655             NaN         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)         26_202.70       952.66    27_155.36       0.6640             NaN         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)         26_202.70       987.10    27_189.79       0.9983          1.0003         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)         26_202.70     1_071.71    27_274.41       0.9985          1.0003         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)        26_202.70     1_004.91    27_207.61       0.9996          1.0001         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)        26_202.70     1_106.94    27_309.64       0.9999          1.0000         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)        26_202.70     1_024.76    27_227.46       0.9996          1.0001         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)        26_202.70     1_137.47    27_340.17       1.0000          1.0000         8.42
IVF-Binary-1024-nl158-random (self)                   26_202.70     3_269.07    29_471.77       0.9996          1.0001         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)         23_963.89       910.47    24_874.35       0.6669             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         23_963.89       916.49    24_880.38       0.6657             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         23_963.89       937.53    24_901.42       0.6644             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        23_963.89     1_000.02    24_963.91       0.9994          1.0001         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        23_963.89     1_083.58    25_047.47       0.9997          1.0000         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        23_963.89     1_021.48    24_985.37       0.9996          1.0001         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        23_963.89     1_108.23    25_072.11       0.9999          1.0000         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        23_963.89     1_016.32    24_980.20       0.9996          1.0001         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        23_963.89     1_154.22    25_118.11       1.0000          1.0000         8.54
IVF-Binary-1024-nl223-random (self)                   23_963.89     3_229.32    27_193.20       0.9996          1.0001         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         24_118.04       932.82    25_050.86       0.6663             NaN         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)         24_118.04       929.00    25_047.05       0.6658             NaN         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)         24_118.04       940.50    25_058.54       0.6646             NaN         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)        24_118.04     1_004.38    25_122.42       0.9996          1.0001         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)        24_118.04     1_099.88    25_217.92       0.9999          1.0000         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)        24_118.04     1_035.70    25_153.74       0.9996          1.0001         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)        24_118.04     1_169.81    25_287.85       0.9999          1.0000         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)        24_118.04     1_081.79    25_199.83       0.9996          1.0001         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)        24_118.04     1_124.78    25_242.82       1.0000          1.0000         8.73
IVF-Binary-1024-nl316-random (self)                   24_118.04     3_239.16    27_357.20       0.9995          1.0001         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)             26_447.27       906.46    27_353.72       0.4022             NaN         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            26_447.27       925.89    27_373.16       0.3980             NaN         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            26_447.27       937.85    27_385.11       0.3963             NaN         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            26_447.27       994.03    27_441.30       0.8540          1.0604         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            26_447.27     1_081.29    27_528.56       0.9425          1.0187         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           26_447.27     1_008.23    27_455.50       0.8431          1.0671         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           26_447.27     1_103.48    27_550.75       0.9317          1.0230         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           26_447.27     1_020.66    27_467.93       0.8385          1.0701         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           26_447.27     1_124.39    27_571.65       0.9264          1.0254         8.42
IVF-Binary-1024-nl158-pca (self)                      26_447.27     3_269.45    29_716.71       0.8277          1.0772         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            24_206.63       921.00    25_127.62       0.3995             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            24_206.63       920.78    25_127.40       0.3982             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            24_206.63       934.82    25_141.45       0.3964             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           24_206.63     1_005.04    25_211.67       0.8498          1.0631         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           24_206.63     1_093.84    25_300.46       0.9391          1.0200         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           24_206.63     1_007.26    25_213.89       0.8455          1.0657         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           24_206.63     1_137.88    25_344.51       0.9342          1.0220         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           24_206.63     1_027.34    25_233.96       0.8394          1.0696         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           24_206.63     1_169.30    25_375.93       0.9272          1.0251         8.54
IVF-Binary-1024-nl223-pca (self)                      24_206.63     3_247.78    27_454.41       0.8300          1.0757         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            24_409.37       924.25    25_333.62       0.3992             NaN         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            24_409.37       926.62    25_335.99       0.3985             NaN         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            24_409.37       941.97    25_351.34       0.3969             NaN         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           24_409.37     1_010.15    25_419.52       0.8494          1.0634         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           24_409.37     1_107.52    25_516.89       0.9381          1.0205         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           24_409.37     1_065.54    25_474.91       0.8471          1.0648         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           24_409.37     1_108.88    25_518.25       0.9355          1.0215         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           24_409.37     1_029.97    25_439.34       0.8409          1.0685         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           24_409.37     1_172.16    25_581.53       0.9293          1.0242         8.73
IVF-Binary-1024-nl316-pca (self)                      24_409.37     3_280.06    27_689.43       0.8315          1.0747         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)              3_400.16       485.50     3_885.66       0.1510             NaN         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)             3_400.16       539.12     3_939.28       0.1357             NaN         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)             3_400.16       582.98     3_983.14       0.1275             NaN         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)             3_400.16       547.25     3_947.41       0.4693          1.6408         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)             3_400.16       970.23     4_370.39       0.5937          1.4489         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)            3_400.16       588.96     3_989.12       0.3934          1.8970         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)            3_400.16     1_035.92     4_436.08       0.5132          1.6505         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)            3_400.16       639.26     4_039.42       0.3560          2.0579         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)            3_400.16     1_096.02     4_496.18       0.4688          1.7920         3.36
IVF-Binary-512-nl158-sign (self)                       3_400.16     1_825.34     5_225.50       0.3949          1.8842         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               918.90       516.16     1_435.06       0.1212             NaN         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               918.90       538.37     1_457.27       0.1175             NaN         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               918.90       594.40     1_513.30       0.1107             NaN         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              918.90       571.78     1_490.68       0.4200          1.7154         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              918.90       999.12     1_918.02       0.5243          1.5142         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              918.90       597.55     1_516.45       0.3913          1.7978         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              918.90     1_030.62     1_949.52       0.4905          1.5868         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              918.90       649.86     1_568.76       0.3460          1.9611         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              918.90     1_107.27     2_026.17       0.4404          1.7068         3.49
IVF-Binary-512-nl223-sign (self)                         918.90     1_809.44     2_728.34       0.3897          1.7976         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)             1_169.24       544.34     1_713.59       0.1151             NaN         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)             1_169.24       561.52     1_730.76       0.1136             NaN         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)             1_169.24       622.37     1_791.61       0.1106             NaN         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)            1_169.24       611.10     1_780.34       0.4127          1.7028         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)            1_169.24     1_045.77     2_215.01       0.5130          1.5028         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)            1_169.24       639.86     1_809.10       0.3986          1.7407         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)            1_169.24     1_062.07     2_231.31       0.4956          1.5385         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)            1_169.24       674.08     1_843.32       0.3574          1.8706         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)            1_169.24     1_133.28     2_302.52       0.4496          1.6513         3.67
IVF-Binary-512-nl316-sign (self)                       1_169.24     1_887.75     3_056.99       0.3990          1.7352         3.67
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       102.93    16_267.94    16_370.87       1.0000          1.0000       146.48
Exhaustive (self)                                        102.93    53_999.52    54_102.45       1.0000          1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)              9_051.76       551.18     9_602.93       0.5361             NaN         2.28
ExhaustiveBinary-256-random-rf10 (query)               9_051.76       708.54     9_760.29       0.9868          1.0022         2.28
ExhaustiveBinary-256-random-rf20 (query)               9_051.76       882.87     9_934.63       0.9980          1.0003         2.28
ExhaustiveBinary-256-random (self)                     9_051.76     2_326.56    11_378.32       0.9876          1.0021         2.28
ExhaustiveBinary-256-pca_no_rr (query)                 9_496.13       539.34    10_035.46       0.1281             NaN         2.28
ExhaustiveBinary-256-pca-rf10 (query)                  9_496.13       699.73    10_195.86       0.3750          1.7664         2.28
ExhaustiveBinary-256-pca-rf20 (query)                  9_496.13       876.18    10_372.30       0.5003          1.4421         2.28
ExhaustiveBinary-256-pca (self)                        9_496.13     2_320.30    11_816.43       0.3725          1.7770         2.28
ExhaustiveBinary-512-random_no_rr (query)             17_677.15       934.24    18_611.39       0.5866             NaN         4.55
ExhaustiveBinary-512-random-rf10 (query)              17_677.15     1_101.51    18_778.66       0.9966          1.0005         4.55
ExhaustiveBinary-512-random-rf20 (query)              17_677.15     1_285.56    18_962.70       0.9997          1.0001         4.55
ExhaustiveBinary-512-random (self)                    17_677.15     3_639.12    21_316.27       0.9969          1.0004         4.55
ExhaustiveBinary-512-pca_no_rr (query)                18_138.87       930.96    19_069.82       0.1131             NaN         4.55
ExhaustiveBinary-512-pca-rf10 (query)                 18_138.87     1_087.98    19_226.84       0.3166          2.0536         4.55
ExhaustiveBinary-512-pca-rf20 (query)                 18_138.87     1_255.96    19_394.82       0.4179          1.6492         4.55
ExhaustiveBinary-512-pca (self)                       18_138.87     3_609.00    21_747.86       0.3146          2.0599         4.55
ExhaustiveBinary-1024-random_no_rr (query)            35_203.88     1_711.79    36_915.67       0.6446             NaN         9.11
ExhaustiveBinary-1024-random-rf10 (query)             35_203.88     1_895.42    37_099.31       0.9993          1.0001         9.11
ExhaustiveBinary-1024-random-rf20 (query)             35_203.88     2_068.95    37_272.83       0.9999          1.0000         9.11
ExhaustiveBinary-1024-random (self)                   35_203.88     6_293.09    41_496.97       0.9994          1.0001         9.11
ExhaustiveBinary-1024-pca_no_rr (query)               35_679.04     1_707.27    37_386.31       0.2379             NaN         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                35_679.04     1_879.55    37_558.59       0.6180          1.2507         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                35_679.04     2_062.43    37_741.47       0.7452          1.1303         9.11
ExhaustiveBinary-1024-pca (self)                      35_679.04     6_253.33    41_932.37       0.6017          1.2739         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  194.16       923.37     1_117.53       0.0421             NaN         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   194.16     1_010.67     1_204.83       0.1896          2.5241         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   194.16     1_697.53     1_891.69       0.3229          1.8300         4.58
ExhaustiveBinary-768-sign (self)                         194.16     3_344.67     3_538.84       0.1997          2.4832         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)           14_058.67       384.46    14_443.13       0.5432             NaN         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)          14_058.67       394.47    14_453.14       0.5409             NaN         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)          14_058.67       402.96    14_461.63       0.5399             NaN         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)          14_058.67       521.00    14_579.67       0.9892          1.0018         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)          14_058.67       656.84    14_715.50       0.9985          1.0002         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)         14_058.67       516.44    14_575.10       0.9883          1.0019         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)         14_058.67       635.39    14_694.06       0.9985          1.0002         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)         14_058.67       518.71    14_577.38       0.9877          1.0021         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)         14_058.67       636.20    14_694.87       0.9984          1.0002         2.74
IVF-Binary-256-nl158-random (self)                    14_058.67     1_641.33    15_700.00       0.9890          1.0018         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)          10_174.21       399.80    10_574.01       0.5424             NaN         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)          10_174.21       404.88    10_579.09       0.5416             NaN         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)          10_174.21       413.37    10_587.58       0.5406             NaN         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)         10_174.21       518.09    10_692.30       0.9890          1.0018         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)         10_174.21       637.24    10_811.45       0.9987          1.0002         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)         10_174.21       520.20    10_694.41       0.9885          1.0019         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)         10_174.21       651.25    10_825.46       0.9986          1.0002         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)         10_174.21       531.35    10_705.56       0.9879          1.0021         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)         10_174.21       641.98    10_816.19       0.9984          1.0002         2.93
IVF-Binary-256-nl223-random (self)                    10_174.21     1_615.98    11_790.19       0.9893          1.0018         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)          10_519.47       425.00    10_944.46       0.5428             NaN         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)          10_519.47       426.86    10_946.33       0.5424             NaN         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)          10_519.47       436.45    10_955.92       0.5415             NaN         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)         10_519.47       541.00    11_060.46       0.9891          1.0018         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)         10_519.47       651.19    11_170.65       0.9987          1.0002         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)         10_519.47       544.08    11_063.54       0.9889          1.0019         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)         10_519.47       650.32    11_169.78       0.9987          1.0002         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)         10_519.47       546.75    11_066.22       0.9882          1.0020         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)         10_519.47       671.95    11_191.42       0.9984          1.0002         3.21
IVF-Binary-256-nl316-random (self)                    10_519.47     1_678.43    12_197.90       0.9896          1.0017         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)              14_648.63       387.33    15_035.96       0.1456             NaN         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)             14_648.63       396.25    15_044.88       0.1387             NaN         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)             14_648.63       405.84    15_054.47       0.1353             NaN         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)             14_648.63       538.80    15_187.43       0.4663          1.4675         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)             14_648.63       618.88    15_267.51       0.6301          1.2293         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)            14_648.63       517.91    15_166.54       0.4330          1.5385         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)            14_648.63       639.66    15_288.29       0.5879          1.2763         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)            14_648.63       534.66    15_183.29       0.4154          1.5816         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)            14_648.63       662.71    15_311.34       0.5658          1.3044         2.74
IVF-Binary-256-nl158-pca (self)                       14_648.63     1_641.75    16_290.38       0.4312          1.5430         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)             10_745.42       399.58    11_145.00       0.1439             NaN         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)             10_745.42       404.26    11_149.69       0.1410             NaN         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)             10_745.42       417.57    11_162.99       0.1368             NaN         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)            10_745.42       522.35    11_267.78       0.4605          1.4711         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)            10_745.42       641.63    11_387.06       0.6246          1.2311         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)            10_745.42       536.30    11_281.72       0.4464          1.5013         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)            10_745.42       643.34    11_388.76       0.6063          1.2513         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)            10_745.42       538.41    11_283.83       0.4240          1.5557         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)            10_745.42       658.88    11_404.30       0.5775          1.2871         2.93
IVF-Binary-256-nl223-pca (self)                       10_745.42     1_655.89    12_401.31       0.4444          1.5069         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)             11_118.16       430.63    11_548.79       0.1440             NaN         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)             11_118.16       425.22    11_543.38       0.1425             NaN         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)             11_118.16       431.89    11_550.05       0.1384             NaN         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)            11_118.16       553.35    11_671.51       0.4618          1.4673         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)            11_118.16       665.96    11_784.12       0.6271          1.2275         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)            11_118.16       559.12    11_677.28       0.4542          1.4835         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)            11_118.16       664.19    11_782.36       0.6169          1.2384         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)            11_118.16       563.17    11_681.34       0.4328          1.5314         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)            11_118.16       682.14    11_800.31       0.5894          1.2709         3.21
IVF-Binary-256-nl316-pca (self)                       11_118.16     1_745.94    12_864.10       0.4522          1.4878         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)           22_988.30       710.94    23_699.24       0.5926             NaN         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)          22_988.30       721.40    23_709.70       0.5900             NaN         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)          22_988.30       738.24    23_726.54       0.5888             NaN         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)          22_988.30       827.32    23_815.62       0.9972          1.0004         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)          22_988.30       933.76    23_922.06       0.9994          1.0001         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)         22_988.30       834.05    23_822.35       0.9972          1.0004         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)         22_988.30       949.66    23_937.96       0.9998          1.0000         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)         22_988.30       847.26    23_835.56       0.9969          1.0004         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)         22_988.30       971.46    23_959.76       0.9997          1.0000         5.02
IVF-Binary-512-nl158-random (self)                    22_988.30     2_678.73    25_667.03       0.9974          1.0003         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)          19_147.54       722.55    19_870.09       0.5912             NaN         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)          19_147.54       730.28    19_877.82       0.5904             NaN         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)          19_147.54       747.28    19_894.82       0.5893             NaN         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)         19_147.54       845.35    19_992.89       0.9973          1.0004         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)         19_147.54       950.49    20_098.03       0.9997          1.0000         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)         19_147.54       848.48    19_996.02       0.9972          1.0004         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)         19_147.54       960.55    20_108.09       0.9998          1.0000         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)         19_147.54       852.02    19_999.56       0.9970          1.0004         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)         19_147.54       968.43    20_115.98       0.9997          1.0000         5.21
IVF-Binary-512-nl223-random (self)                    19_147.54     2_731.62    21_879.16       0.9975          1.0003         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)          19_359.58       752.72    20_112.30       0.5914             NaN         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)          19_359.58       759.66    20_119.24       0.5910             NaN         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)          19_359.58       769.16    20_128.74       0.5899             NaN         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)         19_359.58       866.93    20_226.51       0.9975          1.0003         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)         19_359.58       980.85    20_340.43       0.9998          1.0000         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)         19_359.58       870.34    20_229.92       0.9974          1.0004         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)         19_359.58       977.66    20_337.25       0.9998          1.0000         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)         19_359.58       909.11    20_268.69       0.9972          1.0004         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)         19_359.58       994.63    20_354.21       0.9998          1.0000         5.48
IVF-Binary-512-nl316-random (self)                    19_359.58     2_761.42    22_121.00       0.9976          1.0003         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)              23_602.98       714.16    24_317.13       0.1309             NaN         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)             23_602.98       732.55    24_335.53       0.1241             NaN         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)             23_602.98       743.03    24_346.00       0.1206             NaN         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)             23_602.98       840.48    24_443.46       0.4224          1.5614         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)             23_602.98       952.40    24_555.37       0.5793          1.2883         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)            23_602.98       851.37    24_454.35       0.3880          1.6528         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)            23_602.98       972.36    24_575.34       0.5317          1.3526         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)            23_602.98       862.41    24_465.38       0.3696          1.7106         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)            23_602.98       987.48    24_590.46       0.5068          1.3927         5.02
IVF-Binary-512-nl158-pca (self)                       23_602.98     2_849.37    26_452.35       0.3863          1.6578         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)             20_063.45       733.60    20_797.04       0.1290             NaN         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)             20_063.45       744.56    20_808.01       0.1260             NaN         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)             20_063.45       748.85    20_812.30       0.1218             NaN         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)            20_063.45       857.39    20_920.84       0.4165          1.5662         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)            20_063.45       968.83    21_032.28       0.5723          1.2915         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)            20_063.45       856.48    20_919.93       0.4017          1.6051         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)            20_063.45       974.76    21_038.21       0.5516          1.3189         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)            20_063.45       865.65    20_929.10       0.3784          1.6758         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)            20_063.45       996.47    21_059.92       0.5198          1.3682         5.21
IVF-Binary-512-nl223-pca (self)                       20_063.45     2_823.77    22_887.22       0.4000          1.6099         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)             20_376.86       771.71    21_148.57       0.1288             NaN         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)             20_376.86       758.93    21_135.79       0.1273             NaN         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)             20_376.86       766.17    21_143.02       0.1233             NaN         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)            20_376.86       882.44    21_259.29       0.4180          1.5602         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)            20_376.86       992.42    21_369.28       0.5746          1.2867         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)            20_376.86       873.54    21_250.39       0.4100          1.5803         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)            20_376.86     1_005.51    21_382.37       0.5630          1.3017         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)            20_376.86       911.96    21_288.82       0.3876          1.6436         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)            20_376.86     1_018.25    21_395.11       0.5324          1.3461         5.48
IVF-Binary-512-nl316-pca (self)                       20_376.86     2_832.83    23_209.68       0.4084          1.5848         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)          41_380.32     1_376.77    42_757.10       0.6496             NaN         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)         41_380.32     1_418.46    42_798.79       0.6472             NaN         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)         41_380.32     1_397.27    42_777.59       0.6461             NaN         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)         41_380.32     1_491.43    42_871.76       0.9992          1.0001         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)         41_380.32     1_588.68    42_969.01       0.9995          1.0001         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)        41_380.32     1_496.27    42_876.59       0.9995          1.0001         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)        41_380.32     1_624.73    43_005.05       0.9999          1.0000         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)        41_380.32     1_520.61    42_900.93       0.9994          1.0001         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)        41_380.32     1_648.31    43_028.64       0.9999          1.0000         9.57
IVF-Binary-1024-nl158-random (self)                   41_380.32     4_921.09    46_301.41       0.9995          1.0001         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)         37_386.84     1_375.46    38_762.31       0.6478             NaN         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)         37_386.84     1_405.94    38_792.78       0.6470             NaN         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)         37_386.84     1_402.62    38_789.47       0.6459             NaN         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)        37_386.84     1_498.43    38_885.28       0.9994          1.0001         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)        37_386.84     1_589.14    38_975.99       0.9998          1.0000         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)        37_386.84     1_481.33    38_868.17       0.9994          1.0001         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)        37_386.84     1_601.76    38_988.60       0.9999          1.0000         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)        37_386.84     1_505.41    38_892.25       0.9994          1.0001         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)        37_386.84     1_618.86    39_005.71       0.9999          1.0000         9.76
IVF-Binary-1024-nl223-random (self)                   37_386.84     4_842.55    42_229.40       0.9995          1.0001         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)         37_274.91     1_401.76    38_676.68       0.6478             NaN        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)         37_274.91     1_402.33    38_677.24       0.6474             NaN        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)         37_274.91     1_417.69    38_692.61       0.6465             NaN        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)        37_274.91     1_513.75    38_788.67       0.9995          1.0001        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)        37_274.91     1_628.26    38_903.17       0.9999          1.0000        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)        37_274.91     1_522.00    38_796.92       0.9995          1.0001        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)        37_274.91     1_631.31    38_906.22       0.9999          1.0000        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)        37_274.91     1_533.20    38_808.12       0.9994          1.0001        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)        37_274.91     1_645.73    38_920.65       0.9999          1.0000        10.04
IVF-Binary-1024-nl316-random (self)                   37_274.91     4_900.53    42_175.45       0.9995          1.0001        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)             41_114.01     1_363.50    42_477.51       0.2464             NaN         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)            41_114.01     1_382.99    42_497.00       0.2423             NaN         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)            41_114.01     1_397.37    42_511.38       0.2402             NaN         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)            41_114.01     1_521.38    42_635.39       0.6566          1.2083         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)            41_114.01     1_635.33    42_749.33       0.7952          1.0964         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)           41_114.01     1_497.70    42_611.71       0.6378          1.2270         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)           41_114.01     1_660.98    42_774.99       0.7705          1.1117         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)           41_114.01     1_517.83    42_631.84       0.6290          1.2372         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)           41_114.01     1_641.19    42_755.20       0.7594          1.1193         9.57
IVF-Binary-1024-nl158-pca (self)                      41_114.01     4_909.10    46_023.11       0.6232          1.2468         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)            37_057.62     1_369.03    38_426.64       0.2449             NaN         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)            37_057.62     1_390.05    38_447.66       0.2431             NaN         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)            37_057.62     1_393.53    38_451.15       0.2408             NaN         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)           37_057.62     1_491.77    38_549.39       0.6523          1.2109         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)           37_057.62     1_609.27    38_666.88       0.7909          1.0980         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)           37_057.62     1_513.82    38_571.43       0.6443          1.2192         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)           37_057.62     1_670.50    38_728.12       0.7798          1.1049         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)           37_057.62     1_554.56    38_612.17       0.6327          1.2324         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)           37_057.62     1_650.45    38_708.07       0.7644          1.1156         9.76
IVF-Binary-1024-nl223-pca (self)                      37_057.62     4_900.26    41_957.87       0.6304          1.2374         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)            37_293.64     1_443.54    38_737.18       0.2449             NaN        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)            37_293.64     1_391.28    38_684.92       0.2440             NaN        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)            37_293.64     1_407.76    38_701.40       0.2417             NaN        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)           37_293.64     1_523.95    38_817.59       0.6517          1.2115        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)           37_293.64     1_621.48    38_915.13       0.7905          1.0983        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)           37_293.64     1_512.02    38_805.66       0.6474          1.2159        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)           37_293.64     1_636.46    38_930.11       0.7846          1.1020        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)           37_293.64     1_524.32    38_817.96       0.6363          1.2280        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)           37_293.64     1_657.97    38_951.62       0.7696          1.1118        10.04
IVF-Binary-1024-nl316-pca (self)                      37_293.64     5_029.68    42_323.32       0.6343          1.2333        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)              5_047.40       657.27     5_704.67       0.1093             NaN         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)             5_047.40       723.51     5_770.91       0.0938             NaN         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)             5_047.40       781.63     5_829.02       0.0874             NaN         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)             5_047.40       740.22     5_787.62       0.3886          1.9236         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)             5_047.40     1_317.63     6_365.03       0.4895          1.6962         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)            5_047.40       798.60     5_846.00       0.3238          2.2334         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)            5_047.40     1_396.82     6_444.22       0.4024          1.9696         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)            5_047.40       857.82     5_905.22       0.2977          2.3673         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)            5_047.40     1_469.45     6_516.84       0.3638          2.1015         5.04
IVF-Binary-768-nl158-sign (self)                       5_047.40     2_490.29     7_537.68       0.3272          2.2215         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)             1_351.89       707.58     2_059.47       0.1024             NaN         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)             1_351.89       740.23     2_092.12       0.0957             NaN         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)             1_351.89       812.85     2_164.74       0.0885             NaN         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)            1_351.89       788.35     2_140.24       0.3639          1.9436         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)            1_351.89     1_364.15     2_716.04       0.4694          1.6783         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)            1_351.89       825.95     2_177.85       0.3384          2.0467         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)            1_351.89     1_477.11     2_829.00       0.4320          1.7713         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)            1_351.89       951.62     2_303.52       0.3021          2.2128         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)            1_351.89     1_515.93     2_867.83       0.3778          1.9468         5.23
IVF-Binary-768-nl223-sign (self)                       1_351.89     2_558.80     3_910.69       0.3378          2.0489         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)             1_700.15       769.44     2_469.59       0.0933             NaN         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)             1_700.15       773.30     2_473.45       0.0905             NaN         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)             1_700.15       854.04     2_554.19       0.0847             NaN         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)            1_700.15       848.21     2_548.36       0.3579          1.8927         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)            1_700.15     1_417.41     3_117.56       0.4584          1.6470         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)            1_700.15       860.77     2_560.92       0.3461          1.9341         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)            1_700.15     1_439.96     3_140.11       0.4395          1.6900         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)            1_700.15       939.35     2_639.50       0.3116          2.0760         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)            1_700.15     1_533.69     3_233.84       0.3881          1.8300         5.51
IVF-Binary-768-nl316-sign (self)                       1_700.15     2_653.51     4_353.66       0.3468          1.9292         5.51
-----------------------------------------------------------------------------------------------------------------------------------

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
===================================================================================================================================
Benchmark: 50k samples, 256D - IVF-RaBitQ
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        35.41     4_125.88     4_161.29       1.0000          1.0000        48.83
Exhaustive (self)                                         35.41    14_367.54    14_402.95       1.0000          1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                           1_539.24     1_315.71     2_854.95       0.5172             NaN         2.84
ExhaustiveRaBitQ-rf5 (query)                           1_539.24     1_371.59     2_910.83       0.9146          1.0018         2.84
ExhaustiveRaBitQ-rf10 (query)                          1_539.24     1_376.61     2_915.85       0.9813          1.0003         2.84
ExhaustiveRaBitQ-rf20 (query)                          1_539.24     1_475.94     3_015.18       0.9982          1.0000         2.84
ExhaustiveRaBitQ (self)                                1_539.24     4_588.22     6_127.46       0.9819          1.0003         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       2_138.47       341.51     2_479.97       0.5206             NaN         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      2_138.47       559.04     2_697.50       0.5206             NaN         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      2_138.47       775.83     2_914.30       0.5206             NaN         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      2_138.47       436.07     2_574.53       0.9810          1.0003         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      2_138.47       508.77     2_647.24       0.9970          1.0001         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     2_138.47       644.87     2_783.33       0.9818          1.0003         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     2_138.47       715.13     2_853.60       0.9982          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     2_138.47       859.65     2_998.12       0.9818          1.0003         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     2_138.47       937.36     3_075.83       0.9982          1.0000         2.89
IVF-RaBitQ-nl158 (self)                                2_138.47     3_143.50     5_281.97       0.9983          1.0000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_166.54       450.19     1_616.72       0.5225             NaN         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_166.54       551.64     1_718.18       0.5224             NaN         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_166.54       810.27     1_976.80       0.5223             NaN         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_166.54       536.22     1_702.76       0.9817          1.0003         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_166.54       612.13     1_778.66       0.9976          1.0000         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_166.54       702.35     1_868.88       0.9820          1.0003         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_166.54       710.73     1_877.27       0.9982          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_166.54       882.74     2_049.28       0.9821          1.0003         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_166.54       963.59     2_130.12       0.9983          1.0000         2.95
IVF-RaBitQ-nl223 (self)                                1_166.54     3_212.68     4_379.22       0.9984          1.0000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_299.16       518.27     1_817.43       0.5259             NaN         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_299.16       580.95     1_880.11       0.5258             NaN         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_299.16       831.36     2_130.52       0.5257             NaN         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_299.16       611.15     1_910.31       0.9824          1.0003         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_299.16       685.48     1_984.65       0.9981          1.0000         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_299.16       665.59     1_964.75       0.9826          1.0003         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_299.16       750.01     2_049.17       0.9983          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_299.16       932.35     2_231.52       0.9826          1.0003         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_299.16       989.52     2_288.68       0.9984          1.0000         3.04
IVF-RaBitQ-nl316 (self)                                1_299.16     3_283.55     4_582.72       0.9985          1.0000         3.04
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D - IVF-RaBitQ
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        71.57    10_110.37    10_181.94       1.0000          1.0000        97.66
Exhaustive (self)                                         71.57    32_216.98    32_288.56       1.0000          1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           4_076.10     3_196.45     7_272.54       0.5146             NaN         5.23
ExhaustiveRaBitQ-rf5 (query)                           4_076.10     3_235.25     7_311.34       0.9105          1.0013         5.23
ExhaustiveRaBitQ-rf10 (query)                          4_076.10     3_294.22     7_370.32       0.9792          1.0002         5.23
ExhaustiveRaBitQ-rf20 (query)                          4_076.10     3_413.92     7_490.02       0.9979          1.0000         5.23
ExhaustiveRaBitQ (self)                                4_076.10    12_186.40    16_262.49       0.9793          1.0002         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_740.48       933.04     6_673.52       0.5155             NaN         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_740.48     1_569.85     7_310.33       0.5154             NaN         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_740.48     2_186.56     7_927.04       0.5154             NaN         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_740.48     1_057.18     6_797.67       0.9794          1.0002         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_740.48     1_130.99     6_871.47       0.9975          1.0000         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_740.48     1_649.70     7_390.18       0.9797          1.0002         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_740.48     1_760.49     7_500.98       0.9980          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_740.48     2_247.80     7_988.28       0.9797          1.0002         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_740.48     2_348.29     8_088.77       0.9980          1.0000         5.32
IVF-RaBitQ-nl158 (self)                                5_740.48     7_847.97    13_588.45       0.9979          1.0000         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_340.15     1_294.12     4_634.26       0.5171             NaN         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_340.15     1_627.20     4_967.34       0.5170             NaN         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_340.15     2_522.51     5_862.65       0.5170             NaN         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_340.15     1_393.90     4_734.04       0.9789          1.0002         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_340.15     1_489.80     4_829.94       0.9966          1.0000         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_340.15     1_717.54     5_057.69       0.9799          1.0002         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_340.15     1_817.83     5_157.97       0.9978          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_340.15     2_470.62     5_810.76       0.9799          1.0002         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_340.15     2_566.87     5_907.01       0.9979          1.0000         5.44
IVF-RaBitQ-nl223 (self)                                3_340.15     8_659.99    12_000.13       0.9979          1.0000         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      3_568.36     1_598.97     5_167.33       0.5189             NaN         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      3_568.36     1_786.12     5_354.47       0.5189             NaN         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      3_568.36     2_590.06     6_158.42       0.5189             NaN         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     3_568.36     1_700.80     5_269.16       0.9795          1.0002         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     3_568.36     1_790.58     5_358.93       0.9970          1.0000         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     3_568.36     1_873.59     5_441.95       0.9800          1.0002         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     3_568.36     1_971.47     5_539.83       0.9977          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     3_568.36     2_673.94     6_242.29       0.9803          1.0002         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     3_568.36     2_769.95     6_338.31       0.9981          1.0000         5.63
IVF-RaBitQ-nl316 (self)                                3_568.36     9_208.85    12_777.21       0.9980          1.0000         5.63
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D - IVF-RaBitQ
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       102.15    16_245.56    16_347.70       1.0000          1.0000       146.48
Exhaustive (self)                                        102.15    53_663.12    53_765.26       1.0000          1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           8_238.54     6_414.53    14_653.07       0.5107             NaN         8.11
ExhaustiveRaBitQ-rf5 (query)                           8_238.54     7_038.16    15_276.70       0.9042          1.0011         8.11
ExhaustiveRaBitQ-rf10 (query)                          8_238.54     6_520.22    14_758.76       0.9776          1.0002         8.11
ExhaustiveRaBitQ-rf20 (query)                          8_238.54     6_657.23    14_895.77       0.9975          1.0000         8.11
ExhaustiveRaBitQ (self)                                8_238.54    22_111.22    30_349.76       0.9776          1.0002         8.11
IVF-RaBitQ-nl158-np7-rf0 (query)                      11_034.90     1_902.26    12_937.17       0.5134             NaN         8.25
IVF-RaBitQ-nl158-np12-rf0 (query)                     11_034.90     3_142.94    14_177.84       0.5133             NaN         8.25
IVF-RaBitQ-nl158-np17-rf0 (query)                     11_034.90     4_465.21    15_500.11       0.5133             NaN         8.25
IVF-RaBitQ-nl158-np7-rf10 (query)                     11_034.90     2_025.41    13_060.31       0.9761          1.0002         8.25
IVF-RaBitQ-nl158-np7-rf20 (query)                     11_034.90     2_151.57    13_186.47       0.9960          1.0000         8.25
IVF-RaBitQ-nl158-np12-rf10 (query)                    11_034.90     3_248.52    14_283.42       0.9773          1.0002         8.25
IVF-RaBitQ-nl158-np12-rf20 (query)                    11_034.90     3_364.03    14_398.94       0.9976          1.0000         8.25
IVF-RaBitQ-nl158-np17-rf10 (query)                    11_034.90     4_505.90    15_540.80       0.9773          1.0002         8.25
IVF-RaBitQ-nl158-np17-rf20 (query)                    11_034.90     4_642.34    15_677.24       0.9976          1.0000         8.25
IVF-RaBitQ-nl158 (self)                               11_034.90    15_523.14    26_558.04       0.9975          1.0000         8.25
IVF-RaBitQ-nl223-np11-rf0 (query)                      7_205.35     2_809.47    10_014.82       0.5147             NaN         8.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      7_205.35     3_533.35    10_738.70       0.5146             NaN         8.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      7_205.35     5_263.67    12_469.02       0.5145             NaN         8.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     7_205.35     2_918.08    10_123.43       0.9782          1.0002         8.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     7_205.35     3_042.17    10_247.52       0.9970          1.0000         8.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     7_205.35     3_590.12    10_795.47       0.9787          1.0002         8.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     7_205.35     3_750.86    10_956.21       0.9977          1.0000         8.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     7_205.35     5_300.46    12_505.81       0.9786          1.0002         8.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     7_205.35     5_423.16    12_628.51       0.9977          1.0000         8.44
IVF-RaBitQ-nl223 (self)                                7_205.35    18_398.20    25_603.55       0.9978          1.0000         8.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      7_708.97     3_603.68    11_312.65       0.5160             NaN         8.71
IVF-RaBitQ-nl316-np17-rf0 (query)                      7_708.97     4_070.44    11_779.41       0.5160             NaN         8.71
IVF-RaBitQ-nl316-np25-rf0 (query)                      7_708.97     5_952.86    13_661.83       0.5160             NaN         8.71
IVF-RaBitQ-nl316-np15-rf10 (query)                     7_708.97     3_689.58    11_398.55       0.9784          1.0002         8.71
IVF-RaBitQ-nl316-np15-rf20 (query)                     7_708.97     3_847.85    11_556.82       0.9973          1.0000         8.71
IVF-RaBitQ-nl316-np17-rf10 (query)                     7_708.97     4_143.72    11_852.69       0.9787          1.0002         8.71
IVF-RaBitQ-nl316-np17-rf20 (query)                     7_708.97     4_252.98    11_961.95       0.9977          1.0000         8.71
IVF-RaBitQ-nl316-np25-rf10 (query)                     7_708.97     5_926.65    13_635.62       0.9788          1.0002         8.71
IVF-RaBitQ-nl316-np25-rf20 (query)                     7_708.97     6_047.16    13_756.13       0.9978          1.0000         8.71
IVF-RaBitQ-nl316 (self)                                7_708.97    20_357.62    28_066.59       0.9978          1.0000         8.71
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D - IVF-RaBitQ
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        33.59     4_217.22     4_250.81       1.0000          1.0000        48.83
Exhaustive (self)                                         33.59    14_504.47    14_538.06       1.0000          1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                           1_376.65     1_104.80     2_481.44       0.7288             NaN         2.84
ExhaustiveRaBitQ-rf5 (query)                           1_376.65     1_176.19     2_552.84       0.9969          1.0001         2.84
ExhaustiveRaBitQ-rf10 (query)                          1_376.65     1_222.45     2_599.10       0.9999          1.0000         2.84
ExhaustiveRaBitQ-rf20 (query)                          1_376.65     1_326.57     2_703.21       1.0000          1.0000         2.84
ExhaustiveRaBitQ (self)                                1_376.65     4_048.48     5_425.12       1.0000          1.0000         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       2_015.52       318.61     2_334.14       0.7296             NaN         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      2_015.52       441.09     2_456.61       0.7296             NaN         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      2_015.52       592.65     2_608.17       0.7296             NaN         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      2_015.52       419.07     2_434.59       0.9999          1.0000         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      2_015.52       507.69     2_523.22       1.0000          1.0000         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     2_015.52       539.12     2_554.65       0.9999          1.0000         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     2_015.52       632.32     2_647.84       1.0000          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     2_015.52       683.51     2_699.03       0.9999          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     2_015.52       765.81     2_781.33       1.0000          1.0000         2.89
IVF-RaBitQ-nl158 (self)                                2_015.52     2_546.46     4_561.98       1.0000          1.0000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_272.78       417.95     1_690.73       0.7351             NaN         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_272.78       498.69     1_771.47       0.7351             NaN         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_272.78       717.70     1_990.48       0.7351             NaN         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_272.78       504.05     1_776.83       1.0000          1.0000         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_272.78       598.57     1_871.35       1.0000          1.0000         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_272.78       587.20     1_859.98       1.0000          1.0000         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_272.78       668.61     1_941.40       1.0000          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_272.78       800.71     2_073.49       1.0000          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_272.78       880.16     2_152.95       1.0000          1.0000         2.95
IVF-RaBitQ-nl223 (self)                                1_272.78     2_970.11     4_242.90       1.0000          1.0000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_447.87       504.99     1_952.86       0.7372             NaN         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_447.87       559.15     2_007.03       0.7372             NaN         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_447.87       772.73     2_220.60       0.7372             NaN         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_447.87       595.06     2_042.93       1.0000          1.0000         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_447.87       670.97     2_118.84       1.0000          1.0000         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_447.87       645.07     2_092.94       1.0000          1.0000         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_447.87       726.56     2_174.43       1.0000          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_447.87       863.68     2_311.55       1.0000          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_447.87       945.02     2_392.89       1.0000          1.0000         3.04
IVF-RaBitQ-nl316 (self)                                1_447.87     3_136.37     4_584.24       1.0000          1.0000         3.04
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D - IVF-RaBitQ
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        69.72     9_544.57     9_614.30       1.0000          1.0000        97.66
Exhaustive (self)                                         69.72    32_485.54    32_555.26       1.0000          1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           3_759.67     3_043.64     6_803.31       0.7431             NaN         5.23
ExhaustiveRaBitQ-rf5 (query)                           3_759.67     3_079.20     6_838.87       0.9978          1.0000         5.23
ExhaustiveRaBitQ-rf10 (query)                          3_759.67     3_136.08     6_895.76       1.0000          1.0000         5.23
ExhaustiveRaBitQ-rf20 (query)                          3_759.67     3_262.21     7_021.88       1.0000          1.0000         5.23
ExhaustiveRaBitQ (self)                                3_759.67    10_379.89    14_139.57       1.0000          1.0000         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_073.93       862.86     5_936.79       0.7438             NaN         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_073.93     1_299.03     6_372.96       0.7438             NaN         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_073.93     1_737.00     6_810.94       0.7438             NaN         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_073.93       980.56     6_054.50       1.0000          1.0000         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_073.93     1_093.63     6_167.56       1.0000          1.0000         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_073.93     1_417.33     6_491.27       1.0000          1.0000         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_073.93     1_565.64     6_639.57       1.0000          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_073.93     1_905.82     6_979.75       1.0000          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_073.93     1_983.83     7_057.76       1.0000          1.0000         5.32
IVF-RaBitQ-nl158 (self)                                5_073.93     6_567.81    11_641.75       1.0000          1.0000         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_359.34     1_265.78     4_625.12       0.7471             NaN         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_359.34     1_503.10     4_862.44       0.7475             NaN         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_359.34     2_172.76     5_532.10       0.7475             NaN         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_359.34     1_326.25     4_685.59       0.9987          1.0001         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_359.34     1_438.16     4_797.50       0.9988          1.0001         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_359.34     1_595.27     4_954.61       1.0000          1.0000         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_359.34     1_702.63     5_061.96       1.0000          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_359.34     2_248.24     5_607.57       1.0000          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_359.34     2_358.63     5_717.97       1.0000          1.0000         5.44
IVF-RaBitQ-nl223 (self)                                3_359.34     7_850.31    11_209.65       1.0000          1.0000         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      3_773.43     1_531.69     5_305.12       0.7478             NaN         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      3_773.43     1_710.21     5_483.64       0.7480             NaN         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      3_773.43     2_422.36     6_195.79       0.7481             NaN         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     3_773.43     1_661.39     5_434.82       0.9989          1.0001         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     3_773.43     1_769.22     5_542.65       0.9989          1.0000         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     3_773.43     1_833.83     5_607.26       0.9998          1.0000         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     3_773.43     1_940.83     5_714.26       0.9998          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     3_773.43     2_545.55     6_318.98       1.0000          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     3_773.43     2_673.60     6_447.03       1.0000          1.0000         5.63
IVF-RaBitQ-nl316 (self)                                3_773.43     8_952.61    12_726.04       1.0000          1.0000         5.63
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D - IVF-RaBitQ
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       103.72    15_972.11    16_075.83       1.0000          1.0000       146.48
Exhaustive (self)                                        103.72    54_086.01    54_189.73       1.0000          1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           7_942.11     6_029.34    13_971.45       0.7244             NaN         8.11
ExhaustiveRaBitQ-rf5 (query)                           7_942.11     6_029.89    13_972.01       0.9954          1.0000         8.11
ExhaustiveRaBitQ-rf10 (query)                          7_942.11     6_113.46    14_055.57       0.9999          1.0000         8.11
ExhaustiveRaBitQ-rf20 (query)                          7_942.11     6_317.43    14_259.54       1.0000          1.0000         8.11
ExhaustiveRaBitQ (self)                                7_942.11    21_160.54    29_102.66       1.0000          1.0000         8.11
IVF-RaBitQ-nl158-np7-rf0 (query)                      10_071.79     1_828.33    11_900.12       0.7260             NaN         8.25
IVF-RaBitQ-nl158-np12-rf0 (query)                     10_071.79     2_884.45    12_956.24       0.7260             NaN         8.25
IVF-RaBitQ-nl158-np17-rf0 (query)                     10_071.79     3_956.51    14_028.31       0.7260             NaN         8.25
IVF-RaBitQ-nl158-np7-rf10 (query)                     10_071.79     1_960.33    12_032.12       0.9999          1.0000         8.25
IVF-RaBitQ-nl158-np7-rf20 (query)                     10_071.79     2_088.98    12_160.77       1.0000          1.0000         8.25
IVF-RaBitQ-nl158-np12-rf10 (query)                    10_071.79     2_974.21    13_046.00       0.9999          1.0000         8.25
IVF-RaBitQ-nl158-np12-rf20 (query)                    10_071.79     3_138.84    13_210.63       1.0000          1.0000         8.25
IVF-RaBitQ-nl158-np17-rf10 (query)                    10_071.79     4_029.76    14_101.55       0.9999          1.0000         8.25
IVF-RaBitQ-nl158-np17-rf20 (query)                    10_071.79     4_673.55    14_745.34       1.0000          1.0000         8.25
IVF-RaBitQ-nl158 (self)                               10_071.79    14_930.77    25_002.57       1.0000          1.0000         8.25
IVF-RaBitQ-nl223-np11-rf0 (query)                      7_249.64     2_691.69     9_941.32       0.7271             NaN         8.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      7_249.64     3_341.69    10_591.33       0.7272             NaN         8.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      7_249.64     4_898.38    12_148.01       0.7272             NaN         8.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     7_249.64     2_803.50    10_053.14       0.9997          1.0000         8.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     7_249.64     2_950.16    10_199.80       0.9998          1.0000         8.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     7_249.64     3_471.67    10_721.31       0.9999          1.0000         8.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     7_249.64     3_547.31    10_796.95       1.0000          1.0000         8.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     7_249.64     4_936.47    12_186.11       0.9999          1.0000         8.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     7_249.64     5_007.88    12_257.52       1.0000          1.0000         8.44
IVF-RaBitQ-nl223 (self)                                7_249.64    16_826.35    24_075.99       1.0000          1.0000         8.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      7_701.63     3_502.76    11_204.39       0.7283             NaN         8.71
IVF-RaBitQ-nl316-np17-rf0 (query)                      7_701.63     3_929.63    11_631.25       0.7286             NaN         8.71
IVF-RaBitQ-nl316-np25-rf0 (query)                      7_701.63     5_645.33    13_346.96       0.7286             NaN         8.71
IVF-RaBitQ-nl316-np15-rf10 (query)                     7_701.63     3_614.14    11_315.77       0.9987          1.0000         8.71
IVF-RaBitQ-nl316-np15-rf20 (query)                     7_701.63     3_728.90    11_430.52       0.9988          1.0000         8.71
IVF-RaBitQ-nl316-np17-rf10 (query)                     7_701.63     4_022.11    11_723.74       0.9996          1.0000         8.71
IVF-RaBitQ-nl316-np17-rf20 (query)                     7_701.63     4_151.84    11_853.47       0.9997          1.0000         8.71
IVF-RaBitQ-nl316-np25-rf10 (query)                     7_701.63     5_670.62    13_372.25       0.9999          1.0000         8.71
IVF-RaBitQ-nl316-np25-rf20 (query)                     7_701.63     5_910.59    13_612.22       1.0000          1.0000         8.71
IVF-RaBitQ-nl316 (self)                                7_701.63    19_444.88    27_146.51       1.0000          1.0000         8.71
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

#### Cell embeddings

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D - IVF-RaBitQ
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        34.00     4_193.64     4_227.64       1.0000          1.0000        48.83
Exhaustive (self)                                         34.00    14_247.64    14_281.63       1.0000          1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                           1_498.00     1_488.78     2_986.78       0.8680             NaN         2.84
ExhaustiveRaBitQ-rf5 (query)                           1_498.00     1_540.99     3_038.99       1.0000          1.0000         2.84
ExhaustiveRaBitQ-rf10 (query)                          1_498.00     1_614.11     3_112.11       1.0000          1.0000         2.84
ExhaustiveRaBitQ-rf20 (query)                          1_498.00     1_742.29     3_240.29       1.0000          1.0000         2.84
ExhaustiveRaBitQ (self)                                1_498.00     5_344.15     6_842.15       1.0000          1.0000         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       2_134.69       406.35     2_541.04       0.8725             NaN         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      2_134.69       664.44     2_799.13       0.8730             NaN         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      2_134.69       923.14     3_057.83       0.8730             NaN         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      2_134.69       510.46     2_645.16       0.9976          1.0005         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      2_134.69       593.47     2_728.17       0.9976          1.0005         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     2_134.69       764.34     2_899.03       0.9999          1.0000         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     2_134.69       853.55     2_988.24       0.9999          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     2_134.69     1_020.72     3_155.41       1.0000          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     2_134.69     1_112.77     3_247.46       1.0000          1.0000         2.89
IVF-RaBitQ-nl158 (self)                                2_134.69     3_720.27     5_854.97       1.0000          1.0000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_101.89       482.38     1_584.27       0.8832             NaN         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_101.89       599.29     1_701.18       0.8833             NaN         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_101.89       880.59     1_982.48       0.8832             NaN         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_101.89       573.75     1_675.64       0.9994          1.0001         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_101.89       665.25     1_767.14       0.9994          1.0001         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_101.89       687.15     1_789.04       0.9999          1.0000         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_101.89       781.02     1_882.91       0.9999          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_101.89       959.73     2_061.63       1.0000          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_101.89     1_085.90     2_187.79       1.0000          1.0000         2.95
IVF-RaBitQ-nl223 (self)                                1_101.89     3_546.89     4_648.78       1.0000          1.0000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_282.80       548.17     1_830.98       0.8894             NaN         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_282.80       640.79     1_923.59       0.8894             NaN         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_282.80       883.85     2_166.66       0.8894             NaN         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_282.80       643.62     1_926.42       0.9997          1.0001         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_282.80       728.38     2_011.18       0.9997          1.0001         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_282.80       726.76     2_009.56       0.9998          1.0000         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_282.80       806.76     2_089.56       0.9998          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_282.80       968.50     2_251.30       1.0000          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_282.80     1_057.23     2_340.03       1.0000          1.0000         3.04
IVF-RaBitQ-nl316 (self)                                1_282.80     3_517.61     4_800.42       1.0000          1.0000         3.04
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D - IVF-RaBitQ
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        72.30     9_551.16     9_623.46       1.0000          1.0000        97.66
Exhaustive (self)                                         72.30    32_089.27    32_161.57       1.0000          1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           4_072.68     3_647.67     7_720.36       0.9025             NaN         5.23
ExhaustiveRaBitQ-rf5 (query)                           4_072.68     3_719.98     7_792.66       1.0000          1.0000         5.23
ExhaustiveRaBitQ-rf10 (query)                          4_072.68     3_730.42     7_803.10       1.0000          1.0000         5.23
ExhaustiveRaBitQ-rf20 (query)                          4_072.68     3_864.62     7_937.30       1.0000          1.0000         5.23
ExhaustiveRaBitQ (self)                                4_072.68    12_333.25    16_405.94       1.0000          1.0000         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_629.05     1_026.56     6_655.61       0.9066             NaN         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_629.05     1_816.03     7_445.08       0.9071             NaN         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_629.05     2_392.27     8_021.32       0.9071             NaN         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_629.05     1_127.15     6_756.20       0.9985          1.0003         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_629.05     1_243.21     6_872.26       0.9985          1.0003         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_629.05     1_797.98     7_427.03       0.9999          1.0000         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_629.05     1_915.78     7_544.83       0.9999          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_629.05     2_466.50     8_095.55       1.0000          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_629.05     2_570.84     8_199.89       1.0000          1.0000         5.32
IVF-RaBitQ-nl158 (self)                                5_629.05     8_568.65    14_197.71       1.0000          1.0000         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_125.21     1_349.48     4_474.68       0.9151             NaN         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_125.21     1_688.18     4_813.38       0.9151             NaN         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_125.21     2_545.68     5_670.89       0.9151             NaN         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_125.21     1_421.66     4_546.87       0.9997          1.0000         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_125.21     1_544.04     4_669.25       0.9997          1.0000         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_125.21     1_768.11     4_893.32       1.0000          1.0000         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_125.21     1_866.39     4_991.60       1.0000          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_125.21     2_541.67     5_666.88       1.0000          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_125.21     2_656.15     5_781.36       1.0000          1.0000         5.44
IVF-RaBitQ-nl223 (self)                                3_125.21     8_793.93    11_919.13       1.0000          1.0000         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      3_340.55     1_617.27     4_957.81       0.9190             NaN         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      3_340.55     1_815.44     5_155.98       0.9190             NaN         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      3_340.55     2_649.54     5_990.08       0.9190             NaN         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     3_340.55     1_722.41     5_062.95       0.9999          1.0000         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     3_340.55     1_830.33     5_170.88       0.9999          1.0000         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     3_340.55     1_986.68     5_327.23       0.9999          1.0000         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     3_340.55     2_032.56     5_373.11       0.9999          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     3_340.55     2_748.82     6_089.37       1.0000          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     3_340.55     2_867.08     6_207.63       1.0000          1.0000         5.63
IVF-RaBitQ-nl316 (self)                                3_340.55     9_504.36    12_844.91       1.0000          1.0000         5.63
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D - IVF-RaBitQ
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       101.75    15_991.15    16_092.89       1.0000          1.0000       146.48
Exhaustive (self)                                        101.75    53_976.10    54_077.85       1.0000          1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           8_311.89     6_951.90    15_263.80       0.9249             NaN         8.11
ExhaustiveRaBitQ-rf5 (query)                           8_311.89     6_928.49    15_240.38       1.0000          1.0000         8.11
ExhaustiveRaBitQ-rf10 (query)                          8_311.89     7_103.78    15_415.67       1.0000          1.0000         8.11
ExhaustiveRaBitQ-rf20 (query)                          8_311.89     7_189.79    15_501.68       1.0000          1.0000         8.11
ExhaustiveRaBitQ (self)                                8_311.89    23_375.84    31_687.74       1.0000          1.0000         8.11
IVF-RaBitQ-nl158-np7-rf0 (query)                      10_560.87     2_086.59    12_647.47       0.9272             NaN         8.25
IVF-RaBitQ-nl158-np12-rf0 (query)                     10_560.87     3_549.78    14_110.65       0.9274             NaN         8.25
IVF-RaBitQ-nl158-np17-rf0 (query)                     10_560.87     4_846.86    15_407.74       0.9274             NaN         8.25
IVF-RaBitQ-nl158-np7-rf10 (query)                     10_560.87     2_186.71    12_747.59       0.9996          1.0001         8.25
IVF-RaBitQ-nl158-np7-rf20 (query)                     10_560.87     2_315.51    12_876.38       0.9996          1.0001         8.25
IVF-RaBitQ-nl158-np12-rf10 (query)                    10_560.87     3_667.42    14_228.30       1.0000          1.0000         8.25
IVF-RaBitQ-nl158-np12-rf20 (query)                    10_560.87     3_731.19    14_292.06       1.0000          1.0000         8.25
IVF-RaBitQ-nl158-np17-rf10 (query)                    10_560.87     4_905.28    15_466.15       1.0000          1.0000         8.25
IVF-RaBitQ-nl158-np17-rf20 (query)                    10_560.87     5_017.12    15_578.00       1.0000          1.0000         8.25
IVF-RaBitQ-nl158 (self)                               10_560.87    16_772.80    27_333.67       1.0000          1.0000         8.25
IVF-RaBitQ-nl223-np11-rf0 (query)                      6_792.01     2_902.73     9_694.74       0.9324             NaN         8.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      6_792.01     3_630.26    10_422.27       0.9324             NaN         8.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      6_792.01     5_428.54    12_220.55       0.9324             NaN         8.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     6_792.01     2_994.94     9_786.95       0.9999          1.0000         8.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     6_792.01     3_087.39     9_879.39       0.9999          1.0000         8.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     6_792.01     3_700.90    10_492.91       1.0000          1.0000         8.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     6_792.01     3_938.13    10_730.14       1.0000          1.0000         8.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     6_792.01     5_447.56    12_239.57       1.0000          1.0000         8.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     6_792.01     5_559.42    12_351.43       1.0000          1.0000         8.44
IVF-RaBitQ-nl223 (self)                                6_792.01    18_566.51    25_358.51       1.0000          1.0000         8.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      7_155.03     3_677.26    10_832.29       0.9360             NaN         8.71
IVF-RaBitQ-nl316-np17-rf0 (query)                      7_155.03     4_133.08    11_288.11       0.9360             NaN         8.71
IVF-RaBitQ-nl316-np25-rf0 (query)                      7_155.03     6_045.92    13_200.95       0.9360             NaN         8.71
IVF-RaBitQ-nl316-np15-rf10 (query)                     7_155.03     3_730.03    10_885.06       1.0000          1.0000         8.71
IVF-RaBitQ-nl316-np15-rf20 (query)                     7_155.03     3_938.41    11_093.44       1.0000          1.0000         8.71
IVF-RaBitQ-nl316-np17-rf10 (query)                     7_155.03     4_207.08    11_362.11       1.0000          1.0000         8.71
IVF-RaBitQ-nl316-np17-rf20 (query)                     7_155.03     4_328.37    11_483.40       1.0000          1.0000         8.71
IVF-RaBitQ-nl316-np25-rf10 (query)                     7_155.03     6_060.06    13_215.09       1.0000          1.0000         8.71
IVF-RaBitQ-nl316-np25-rf20 (query)                     7_155.03     6_180.94    13_335.97       1.0000          1.0000         8.71
IVF-RaBitQ-nl316 (self)                                7_155.03    20_557.14    27_712.17       1.0000          1.0000         8.71
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

Overall, this is a fantastic binary index that massively compresses the data,
while still allowing for great Recalls. If you need to compress your data
and reduce memory fingerprint, please, use RaBitQ!

### <u>TurboQuant (IVF and exhaustive)</u>

[TurboQuant](https://arxiv.org/abs/2504.19874) is a scalar quantisation scheme.
It applies a fixed random orthogonal rotation to each unit-normalised vector,
which drives every coordinate towards the same Beta distribution, and then
quantises each rotated coordinate against a Lloyd-Max codebook that is optimal
for that distribution. Codes are stored in bit-plane format and scored with a
FAISS PQ4-style fast-scan lookup table, so distance estimation is SIMD-friendly
and fast.

Encoding is data-oblivious: a single shared rotation and a single shared
codebook are used for every vector, with no per-cluster residuals. For the
`ExhaustiveTurboQuant` index every query scans the whole set via the block-fused
SIMD kernel. For the `IVF-TurboQuant` index the clustering is routing only — the
same global encoding is reused and vectors are merely bucketed into cells, so
the IVF centroids do not feed the quantiser (unlike IVF-RaBitQ). As with the
other indices, the original vectors can be stored on disk for exact re-ranking.

**Key parameters *(TurboQuant)*:**

- *bits*: Bits per coordinate, either 2, 3, or 4. The higher the number, the
  better the Recall, at the cost of memory. Note that 3-bit has no SIMD kernel
  and falls back to the scalar scorer, which is markedly slower; prefer 4-bit
  unless memory forces otherwise. In the benchmarks we show 2-bit and 4-bit.
- *reranking*: As with the other indices, the original vectors can be stored on
  disk. Once the TurboQuant approximated distance has identified the most
  interesting candidates, the on-disk vectors are loaded and the results are
  re-ranked. The reranking_factor controls how many more vectors are rescored
  than the desired k, e.g., 10 means `10 * k vectors` are scored and then
  re-ranked. The more candidates, the better the Recall. The default is `20`.

**Key parameters *(IVF-specific)*:**

- *Number of lists (nl)*: The number of independent k-means clusters to
  generate. If the structure of the data is unknown, people use `sqrt(n)` as a
  heuristic.
- *Number of points (np)*: The number of clusters to probe during search.
  Numbers here tend to be `sqrt(nlist)` or up to 5% of the nlist.

The self queries (i.e., kNN generation) are done with `reranking_factor = 20`.
You will quickly appreciate that the performance on strongly clustered data is
very poor. This index shines especially for large dimensional data that is
generated by neural networks: the area this index shines in and was designed
for.

#### Correlated data

<details>
<summary><b>Correlated data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D - TurboQuant + IVF
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        33.10     4_315.19     4_348.29       1.0000          1.0000        48.83
Exhaustive (self)                                         33.10    13_879.91    13_913.00       1.0000          1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              191.01       365.50       556.50       0.0109             NaN         7.12
ExhaustiveTQ-b2-rf5 (query)                              191.01       445.17       636.18       0.0526          1.2562         7.12
ExhaustiveTQ-b2-rf10 (query)                             191.01       580.82       771.83       0.1030          1.1894         7.12
ExhaustiveTQ-b2-rf20 (query)                             191.01       972.18     1_163.19       0.2003          1.1318         7.12
ExhaustiveTQ-b2 (self)                                   191.01     3_277.07     3_468.08       0.1995          1.1335         7.12
ExhaustiveTQ-b4-rf0 (query)                              268.35       582.09       850.44       0.0132             NaN        13.22
ExhaustiveTQ-b4-rf5 (query)                              268.35       674.40       942.75       0.0576          1.2376        13.22
ExhaustiveTQ-b4-rf10 (query)                             268.35       812.05     1_080.40       0.1079          1.1773        13.22
ExhaustiveTQ-b4-rf20 (query)                             268.35     1_207.29     1_475.65       0.2030          1.1256        13.22
ExhaustiveTQ-b4 (self)                                   268.35     4_030.77     4_299.12       0.2033          1.1266        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_414.17       117.47     1_531.64       0.0116             NaN         7.81
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_414.17       151.89     1_566.06       0.0109             NaN         7.81
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_414.17       184.22     1_598.39       0.0109             NaN         7.81
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_414.17       288.30     1_702.47       0.1105          1.1790         7.81
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_414.17       527.57     1_941.73       0.2158          1.1228         7.81
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_414.17       337.40     1_751.56       0.1035          1.1886         7.81
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_414.17       617.94     2_032.11       0.2012          1.1311         7.81
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_414.17       376.63     1_790.79       0.1030          1.1894         7.81
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_414.17       685.78     2_099.95       0.2003          1.1318         7.81
IVF-TQ-b2-nl158 (self)                                 1_414.17     1_313.57     2_727.73       0.1995          1.1335         7.81
IVF-TQ-b2-nl223-np11-rf0 (query)                         749.25       132.72       881.97       0.0113             NaN         7.94
IVF-TQ-b2-nl223-np14-rf0 (query)                         749.25       156.80       906.05       0.0109             NaN         7.94
IVF-TQ-b2-nl223-np21-rf0 (query)                         749.25       184.15       933.40       0.0109             NaN         7.94
IVF-TQ-b2-nl223-np11-rf10 (query)                        749.25       323.59     1_072.84       0.1067          1.1837         7.94
IVF-TQ-b2-nl223-np11-rf20 (query)                        749.25       591.18     1_340.43       0.2082          1.1267         7.94
IVF-TQ-b2-nl223-np14-rf10 (query)                        749.25       341.70     1_090.95       0.1035          1.1885         7.94
IVF-TQ-b2-nl223-np14-rf20 (query)                        749.25       651.45     1_400.70       0.2014          1.1310         7.94
IVF-TQ-b2-nl223-np21-rf10 (query)                        749.25       392.61     1_141.86       0.1030          1.1894         7.94
IVF-TQ-b2-nl223-np21-rf20 (query)                        749.25       705.81     1_455.06       0.2003          1.1318         7.94
IVF-TQ-b2-nl223 (self)                                   749.25     1_326.76     2_076.01       0.1995          1.1335         7.94
IVF-TQ-b2-nl316-np15-rf0 (query)                         970.27       149.96     1_120.23       0.0113             NaN         8.13
IVF-TQ-b2-nl316-np17-rf0 (query)                         970.27       145.89     1_116.17       0.0111             NaN         8.13
IVF-TQ-b2-nl316-np25-rf0 (query)                         970.27       181.34     1_151.62       0.0109             NaN         8.13
IVF-TQ-b2-nl316-np15-rf10 (query)                        970.27       322.35     1_292.63       0.1074          1.1831         8.13
IVF-TQ-b2-nl316-np15-rf20 (query)                        970.27       576.57     1_546.84       0.2091          1.1263         8.13
IVF-TQ-b2-nl316-np17-rf10 (query)                        970.27       319.04     1_289.31       0.1049          1.1866         8.13
IVF-TQ-b2-nl316-np17-rf20 (query)                        970.27       587.26     1_557.54       0.2041          1.1294         8.13
IVF-TQ-b2-nl316-np25-rf10 (query)                        970.27       398.82     1_369.10       0.1030          1.1894         8.13
IVF-TQ-b2-nl316-np25-rf20 (query)                        970.27       681.32     1_651.60       0.2003          1.1318         8.13
IVF-TQ-b2-nl316 (self)                                   970.27     1_339.72     2_309.99       0.1995          1.1335         8.13
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_629.52       160.92     1_790.44       0.0140             NaN        14.07
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_629.52       216.19     1_845.72       0.0132             NaN        14.07
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_629.52       269.48     1_899.00       0.0132             NaN        14.07
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_629.52       342.58     1_972.10       0.1158          1.1694        14.07
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_629.52       591.69     2_221.21       0.2185          1.1185        14.07
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_629.52       414.14     2_043.66       0.1084          1.1766        14.07
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_629.52       703.45     2_332.98       0.2040          1.1250        14.07
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_629.52       481.66     2_111.18       0.1079          1.1773        14.07
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_629.52       799.45     2_428.97       0.2030          1.1256        14.07
IVF-TQ-b4-nl158 (self)                                 1_629.52     1_543.34     3_172.86       0.2033          1.1266        14.07
IVF-TQ-b4-nl223-np11-rf0 (query)                         818.85       184.39     1_003.24       0.0137             NaN        14.26
IVF-TQ-b4-nl223-np14-rf0 (query)                         818.85       211.26     1_030.11       0.0133             NaN        14.26
IVF-TQ-b4-nl223-np21-rf0 (query)                         818.85       268.82     1_087.67       0.0132             NaN        14.26
IVF-TQ-b4-nl223-np11-rf10 (query)                        818.85       377.23     1_196.07       0.1124          1.1723        14.26
IVF-TQ-b4-nl223-np11-rf20 (query)                        818.85       649.39     1_468.24       0.2117          1.1211        14.26
IVF-TQ-b4-nl223-np14-rf10 (query)                        818.85       408.37     1_227.22       0.1086          1.1765        14.26
IVF-TQ-b4-nl223-np14-rf20 (query)                        818.85       699.53     1_518.37       0.2040          1.1251        14.26
IVF-TQ-b4-nl223-np21-rf10 (query)                        818.85       484.18     1_303.02       0.1079          1.1773        14.26
IVF-TQ-b4-nl223-np21-rf20 (query)                        818.85       800.31     1_619.16       0.2030          1.1256        14.26
IVF-TQ-b4-nl223 (self)                                   818.85     1_495.80     2_314.65       0.2033          1.1266        14.26
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_044.88       188.07     1_232.95       0.0137             NaN        14.56
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_044.88       196.66     1_241.55       0.0134             NaN        14.56
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_044.88       249.54     1_294.42       0.0132             NaN        14.56
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_044.88       399.06     1_443.94       0.1130          1.1713        14.56
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_044.88       651.03     1_695.91       0.2124          1.1205        14.56
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_044.88       392.09     1_436.97       0.1103          1.1744        14.56
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_044.88       660.27     1_705.15       0.2074          1.1232        14.56
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_044.88       454.89     1_499.77       0.1079          1.1773        14.56
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_044.88       767.61     1_812.49       0.2030          1.1256        14.56
IVF-TQ-b4-nl316 (self)                                 1_044.88     1_494.91     2_539.79       0.2033          1.1266        14.56
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D - TurboQuant + IVF
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        69.83     9_654.21     9_724.04       1.0000          1.0000        97.66
Exhaustive (self)                                         69.83    32_663.05    32_732.88       1.0000          1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              479.65       665.63     1_145.28       0.0120             NaN        13.97
ExhaustiveTQ-b2-rf5 (query)                              479.65       764.22     1_243.87       0.0561          1.1729        13.97
ExhaustiveTQ-b2-rf10 (query)                             479.65       906.46     1_386.10       0.1081          1.1302        13.97
ExhaustiveTQ-b2-rf20 (query)                             479.65     1_328.40     1_808.05       0.2057          1.0911        13.97
ExhaustiveTQ-b2 (self)                                   479.65     4_395.05     4_874.70       0.2055          1.0916        13.97
ExhaustiveTQ-b4-rf0 (query)                              607.34     1_131.59     1_738.93       0.0183             NaN        26.18
ExhaustiveTQ-b4-rf5 (query)                              607.34     1_243.89     1_851.23       0.0633          1.1621        26.18
ExhaustiveTQ-b4-rf10 (query)                             607.34     1_384.30     1_991.64       0.1141          1.1229        26.18
ExhaustiveTQ-b4-rf20 (query)                             607.34     1_805.47     2_412.81       0.2061          1.0883        26.18
ExhaustiveTQ-b4 (self)                                   607.34     5_993.68     6_601.02       0.2069          1.0881        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        3_193.85       234.57     3_428.41       0.0125             NaN        14.98
IVF-TQ-b2-nl158-np12-rf0 (query)                       3_193.85       287.57     3_481.42       0.0119             NaN        14.98
IVF-TQ-b2-nl158-np17-rf0 (query)                       3_193.85       335.21     3_529.06       0.0119             NaN        14.98
IVF-TQ-b2-nl158-np7-rf10 (query)                       3_193.85       436.05     3_629.90       0.1140          1.1257        14.98
IVF-TQ-b2-nl158-np7-rf20 (query)                       3_193.85       704.16     3_898.01       0.2176          1.0868        14.98
IVF-TQ-b2-nl158-np12-rf10 (query)                      3_193.85       499.00     3_692.85       0.1081          1.1302        14.98
IVF-TQ-b2-nl158-np12-rf20 (query)                      3_193.85       796.25     3_990.10       0.2057          1.0911        14.98
IVF-TQ-b2-nl158-np17-rf10 (query)                      3_193.85       557.48     3_751.32       0.1081          1.1302        14.98
IVF-TQ-b2-nl158-np17-rf20 (query)                      3_193.85       876.69     4_070.54       0.2057          1.0911        14.98
IVF-TQ-b2-nl158 (self)                                 3_193.85     1_836.00     5_029.85       0.2055          1.0916        14.98
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_532.53       256.87     1_789.40       0.0123             NaN        15.21
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_532.53       282.06     1_814.59       0.0120             NaN        15.21
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_532.53       336.02     1_868.55       0.0119             NaN        15.21
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_532.53       460.00     1_992.53       0.1112          1.1282        15.21
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_532.53       806.24     2_338.77       0.2117          1.0891        15.21
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_532.53       496.84     2_029.37       0.1086          1.1299        15.21
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_532.53       793.13     2_325.66       0.2066          1.0908        15.21
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_532.53       558.09     2_090.62       0.1081          1.1302        15.21
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_532.53       887.76     2_420.29       0.2057          1.0911        15.21
IVF-TQ-b2-nl223 (self)                                 1_532.53     1_964.48     3_497.01       0.2055          1.0916        15.21
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_861.09       273.43     2_134.52       0.0123             NaN        15.54
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_861.09       300.50     2_161.60       0.0121             NaN        15.54
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_861.09       358.56     2_219.65       0.0119             NaN        15.54
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_861.09       505.51     2_366.60       0.1120          1.1274        15.54
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_861.09       812.41     2_673.50       0.2135          1.0883        15.54
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_861.09       485.18     2_346.27       0.1095          1.1293        15.54
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_861.09       790.71     2_651.80       0.2085          1.0902        15.54
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_861.09       540.82     2_401.92       0.1081          1.1302        15.54
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_861.09       872.78     2_733.87       0.2057          1.0911        15.54
IVF-TQ-b2-nl316 (self)                                 1_861.09     1_901.03     3_762.12       0.2055          1.0916        15.54
IVF-TQ-b4-nl158-np7-rf0 (query)                        3_275.27       351.15     3_626.42       0.0191             NaN        27.50
IVF-TQ-b4-nl158-np12-rf0 (query)                       3_275.27       416.62     3_691.89       0.0183             NaN        27.50
IVF-TQ-b4-nl158-np17-rf0 (query)                       3_275.27       495.94     3_771.21       0.0183             NaN        27.50
IVF-TQ-b4-nl158-np7-rf10 (query)                       3_275.27       540.76     3_816.03       0.1206          1.1186        27.50
IVF-TQ-b4-nl158-np7-rf20 (query)                       3_275.27       822.21     4_097.48       0.2184          1.0844        27.50
IVF-TQ-b4-nl158-np12-rf10 (query)                      3_275.27       647.13     3_922.40       0.1141          1.1229        27.50
IVF-TQ-b4-nl158-np12-rf20 (query)                      3_275.27       959.75     4_235.02       0.2061          1.0883        27.50
IVF-TQ-b4-nl158-np17-rf10 (query)                      3_275.27       746.31     4_021.58       0.1140          1.1229        27.50
IVF-TQ-b4-nl158-np17-rf20 (query)                      3_275.27     1_072.47     4_347.74       0.2061          1.0883        27.50
IVF-TQ-b4-nl158 (self)                                 3_275.27     2_132.72     5_407.99       0.2069          1.0881        27.50
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_668.12       350.56     2_018.68       0.0186             NaN        27.83
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_668.12       398.36     2_066.48       0.0183             NaN        27.83
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_668.12       491.50     2_159.62       0.0183             NaN        27.83
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_668.12       565.59     2_233.70       0.1171          1.1210        27.83
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_668.12       863.24     2_531.36       0.2121          1.0864        27.83
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_668.12       629.93     2_298.05       0.1144          1.1227        27.83
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_668.12       930.07     2_598.18       0.2070          1.0880        27.83
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_668.12       730.77     2_398.89       0.1141          1.1229        27.83
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_668.12     1_070.02     2_738.14       0.2061          1.0883        27.83
IVF-TQ-b4-nl223 (self)                                 1_668.12     2_146.34     3_814.46       0.2069          1.0881        27.83
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_986.77       384.07     2_370.84       0.0187             NaN        28.31
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_986.77       387.93     2_374.71       0.0184             NaN        28.31
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_986.77       513.24     2_500.01       0.0183             NaN        28.31
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_986.77       584.62     2_571.39       0.1179          1.1204        28.31
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_986.77       904.55     2_891.32       0.2141          1.0857        28.31
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_986.77       607.03     2_593.80       0.1154          1.1220        28.31
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_986.77       921.40     2_908.17       0.2091          1.0874        28.31
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_986.77       703.95     2_690.72       0.1141          1.1229        28.31
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_986.77     1_044.55     3_031.32       0.2061          1.0883        28.31
IVF-TQ-b4-nl316 (self)                                 1_986.77     2_172.53     4_159.30       0.2069          1.0881        28.31
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D - TurboQuant + IVF
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       102.64    16_058.05    16_160.69       1.0000          1.0000       146.48
Exhaustive (self)                                        102.64    53_929.48    54_032.12       1.0000          1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              978.43     1_012.54     1_990.97       0.0154             NaN        21.33
ExhaustiveTQ-b2-rf5 (query)                              978.43     1_184.80     2_163.23       0.0627          1.1385        21.33
ExhaustiveTQ-b2-rf10 (query)                             978.43     1_322.10     2_300.53       0.1152          1.1036        21.33
ExhaustiveTQ-b2-rf20 (query)                             978.43     1_738.25     2_716.68       0.2128          1.0710        21.33
ExhaustiveTQ-b2 (self)                                   978.43     5_761.20     6_739.63       0.2134          1.0712        21.33
ExhaustiveTQ-b4-rf0 (query)                            1_161.70     1_826.43     2_988.14       0.0148             NaN        39.64
ExhaustiveTQ-b4-rf5 (query)                            1_161.70     1_865.02     3_026.72       0.0558          1.1453        39.64
ExhaustiveTQ-b4-rf10 (query)                           1_161.70     2_015.53     3_177.23       0.1025          1.1154        39.64
ExhaustiveTQ-b4-rf20 (query)                           1_161.70     2_493.33     3_655.04       0.1923          1.0877        39.64
ExhaustiveTQ-b4 (self)                                 1_161.70     8_235.66     9_397.36       0.1918          1.0881        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        5_238.38       416.90     5_655.28       0.0162             NaN        22.62
IVF-TQ-b2-nl158-np12-rf0 (query)                       5_238.38       486.11     5_724.49       0.0154             NaN        22.62
IVF-TQ-b2-nl158-np17-rf0 (query)                       5_238.38       555.55     5_793.93       0.0154             NaN        22.62
IVF-TQ-b2-nl158-np7-rf10 (query)                       5_238.38       652.02     5_890.40       0.1215          1.1004        22.62
IVF-TQ-b2-nl158-np7-rf20 (query)                       5_238.38       946.32     6_184.70       0.2243          1.0677        22.62
IVF-TQ-b2-nl158-np12-rf10 (query)                      5_238.38       743.17     5_981.55       0.1152          1.1036        22.62
IVF-TQ-b2-nl158-np12-rf20 (query)                      5_238.38     1_064.86     6_303.24       0.2128          1.0710        22.62
IVF-TQ-b2-nl158-np17-rf10 (query)                      5_238.38       830.92     6_069.29       0.1152          1.1036        22.62
IVF-TQ-b2-nl158-np17-rf20 (query)                      5_238.38     1_245.34     6_483.72       0.2128          1.0710        22.62
IVF-TQ-b2-nl158 (self)                                 5_238.38     2_603.35     7_841.73       0.2134          1.0712        22.62
IVF-TQ-b2-nl223-np11-rf0 (query)                       2_711.68       446.20     3_157.87       0.0160             NaN        23.00
IVF-TQ-b2-nl223-np14-rf0 (query)                       2_711.68       472.34     3_184.02       0.0155             NaN        23.00
IVF-TQ-b2-nl223-np21-rf0 (query)                       2_711.68       538.32     3_250.00       0.0154             NaN        23.00
IVF-TQ-b2-nl223-np11-rf10 (query)                      2_711.68       695.70     3_407.38       0.1209          1.1003        23.00
IVF-TQ-b2-nl223-np11-rf20 (query)                      2_711.68     1_010.43     3_722.11       0.2233          1.0678        23.00
IVF-TQ-b2-nl223-np14-rf10 (query)                      2_711.68       708.63     3_420.31       0.1162          1.1029        23.00
IVF-TQ-b2-nl223-np14-rf20 (query)                      2_711.68     1_049.14     3_760.81       0.2147          1.0703        23.00
IVF-TQ-b2-nl223-np21-rf10 (query)                      2_711.68       799.55     3_511.23       0.1152          1.1036        23.00
IVF-TQ-b2-nl223-np21-rf20 (query)                      2_711.68     1_152.32     3_864.00       0.2128          1.0710        23.00
IVF-TQ-b2-nl223 (self)                                 2_711.68     2_604.97     5_316.65       0.2134          1.0712        23.00
IVF-TQ-b2-nl316-np15-rf0 (query)                       3_151.49       470.85     3_622.34       0.0159             NaN        23.53
IVF-TQ-b2-nl316-np17-rf0 (query)                       3_151.49       488.47     3_639.96       0.0156             NaN        23.53
IVF-TQ-b2-nl316-np25-rf0 (query)                       3_151.49       538.09     3_689.58       0.0154             NaN        23.53
IVF-TQ-b2-nl316-np15-rf10 (query)                      3_151.49       734.43     3_885.92       0.1204          1.1007        23.53
IVF-TQ-b2-nl316-np15-rf20 (query)                      3_151.49     1_100.41     4_251.90       0.2221          1.0682        23.53
IVF-TQ-b2-nl316-np17-rf10 (query)                      3_151.49       748.20     3_899.69       0.1177          1.1021        23.53
IVF-TQ-b2-nl316-np17-rf20 (query)                      3_151.49     1_110.27     4_261.76       0.2175          1.0695        23.53
IVF-TQ-b2-nl316-np25-rf10 (query)                      3_151.49       787.49     3_938.98       0.1152          1.1036        23.53
IVF-TQ-b2-nl316-np25-rf20 (query)                      3_151.49     1_135.19     4_286.68       0.2128          1.0710        23.53
IVF-TQ-b2-nl316 (self)                                 3_151.49     2_615.18     5_766.67       0.2134          1.0712        23.53
IVF-TQ-b4-nl158-np7-rf0 (query)                        5_404.03       543.92     5_947.94       0.0155             NaN        41.38
IVF-TQ-b4-nl158-np12-rf0 (query)                       5_404.03       687.87     6_091.89       0.0148             NaN        41.38
IVF-TQ-b4-nl158-np17-rf0 (query)                       5_404.03       806.49     6_210.51       0.0148             NaN        41.38
IVF-TQ-b4-nl158-np7-rf10 (query)                       5_404.03       797.16     6_201.18       0.1084          1.1125        41.38
IVF-TQ-b4-nl158-np7-rf20 (query)                       5_404.03     1_094.17     6_498.20       0.2038          1.0851        41.38
IVF-TQ-b4-nl158-np12-rf10 (query)                      5_404.03       948.68     6_352.71       0.1025          1.1154        41.38
IVF-TQ-b4-nl158-np12-rf20 (query)                      5_404.03     1_284.78     6_688.81       0.1923          1.0877        41.38
IVF-TQ-b4-nl158-np17-rf10 (query)                      5_404.03     1_132.76     6_536.79       0.1025          1.1154        41.38
IVF-TQ-b4-nl158-np17-rf20 (query)                      5_404.03     1_463.80     6_867.83       0.1923          1.0877        41.38
IVF-TQ-b4-nl158 (self)                                 5_404.03     3_013.59     8_417.61       0.1918          1.0881        41.38
IVF-TQ-b4-nl223-np11-rf0 (query)                       2_837.75       589.36     3_427.11       0.0156             NaN        41.96
IVF-TQ-b4-nl223-np14-rf0 (query)                       2_837.75       656.99     3_494.74       0.0150             NaN        41.96
IVF-TQ-b4-nl223-np21-rf0 (query)                       2_837.75       790.28     3_628.03       0.0148             NaN        41.96
IVF-TQ-b4-nl223-np11-rf10 (query)                      2_837.75       843.55     3_681.29       0.1075          1.1123        41.96
IVF-TQ-b4-nl223-np11-rf20 (query)                      2_837.75     1_195.95     4_033.69       0.2025          1.0849        41.96
IVF-TQ-b4-nl223-np14-rf10 (query)                      2_837.75       970.10     3_807.85       0.1035          1.1147        41.96
IVF-TQ-b4-nl223-np14-rf20 (query)                      2_837.75     1_245.92     4_083.67       0.1941          1.0871        41.96
IVF-TQ-b4-nl223-np21-rf10 (query)                      2_837.75     1_042.84     3_880.59       0.1025          1.1154        41.96
IVF-TQ-b4-nl223-np21-rf20 (query)                      2_837.75     1_407.05     4_244.79       0.1923          1.0877        41.96
IVF-TQ-b4-nl223 (self)                                 2_837.75     3_022.90     5_860.65       0.1918          1.0881        41.96
IVF-TQ-b4-nl316-np15-rf0 (query)                       3_257.19       619.99     3_877.18       0.0155             NaN        42.73
IVF-TQ-b4-nl316-np17-rf0 (query)                       3_257.19       638.95     3_896.14       0.0152             NaN        42.73
IVF-TQ-b4-nl316-np25-rf0 (query)                       3_257.19       742.59     3_999.78       0.0148             NaN        42.73
IVF-TQ-b4-nl316-np15-rf10 (query)                      3_257.19       857.53     4_114.72       0.1073          1.1126        42.73
IVF-TQ-b4-nl316-np15-rf20 (query)                      3_257.19     1_189.89     4_447.08       0.2016          1.0851        42.73
IVF-TQ-b4-nl316-np17-rf10 (query)                      3_257.19       886.49     4_143.68       0.1049          1.1138        42.73
IVF-TQ-b4-nl316-np17-rf20 (query)                      3_257.19     1_226.91     4_484.10       0.1968          1.0863        42.73
IVF-TQ-b4-nl316-np25-rf10 (query)                      3_257.19       989.73     4_246.92       0.1025          1.1154        42.73
IVF-TQ-b4-nl316-np25-rf20 (query)                      3_257.19     1_353.60     4_610.79       0.1923          1.0877        42.73
IVF-TQ-b4-nl316 (self)                                 3_257.19     3_031.77     6_288.96       0.1918          1.0881        42.73
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D - TurboQuant + IVF
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        34.09     4_195.16     4_229.25       1.0000          1.0000        48.83
Exhaustive (self)                                         34.09    14_122.57    14_156.65       1.0000          1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              179.47       376.62       556.09       0.0662             NaN         7.12
ExhaustiveTQ-b2-rf5 (query)                              179.47       460.62       640.09       0.1862          1.3185         7.12
ExhaustiveTQ-b2-rf10 (query)                             179.47       583.82       763.29       0.2699          1.2136         7.12
ExhaustiveTQ-b2-rf20 (query)                             179.47       967.45     1_146.92       0.4056          1.1279         7.12
ExhaustiveTQ-b2 (self)                                   179.47     3_215.32     3_394.79       0.4070          1.1561         7.12
ExhaustiveTQ-b4-rf0 (query)                              268.12       592.60       860.72       0.0871             NaN        13.22
ExhaustiveTQ-b4-rf5 (query)                              268.12       693.29       961.41       0.2059          1.2890        13.22
ExhaustiveTQ-b4-rf10 (query)                             268.12       810.96     1_079.09       0.2865          1.1965        13.22
ExhaustiveTQ-b4-rf20 (query)                             268.12     1_203.66     1_471.78       0.4170          1.1210        13.22
ExhaustiveTQ-b4 (self)                                   268.12     3_999.94     4_268.06       0.4165          1.1485        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_324.69       111.95     1_436.64       0.0664             NaN         7.81
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_324.69       125.61     1_450.30       0.0662             NaN         7.81
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_324.69       140.99     1_465.68       0.0662             NaN         7.81
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_324.69       318.22     1_642.91       0.2700          1.2136         7.81
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_324.69       640.22     1_964.92       0.4056          1.1279         7.81
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_324.69       334.24     1_658.94       0.2699          1.2136         7.81
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_324.69       687.17     2_011.86       0.4056          1.1279         7.81
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_324.69       399.70     1_724.39       0.2699          1.2136         7.81
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_324.69       791.70     2_116.39       0.4056          1.1279         7.81
IVF-TQ-b2-nl158 (self)                                 1_324.69     1_206.54     2_531.24       0.4070          1.1561         7.81
IVF-TQ-b2-nl223-np11-rf0 (query)                         861.93       124.60       986.53       0.0664             NaN         7.93
IVF-TQ-b2-nl223-np14-rf0 (query)                         861.93       129.34       991.27       0.0662             NaN         7.93
IVF-TQ-b2-nl223-np21-rf0 (query)                         861.93       153.87     1_015.80       0.0662             NaN         7.93
IVF-TQ-b2-nl223-np11-rf10 (query)                        861.93       308.04     1_169.96       0.2711          1.2125         7.93
IVF-TQ-b2-nl223-np11-rf20 (query)                        861.93       581.50     1_443.42       0.4078          1.1269         7.93
IVF-TQ-b2-nl223-np14-rf10 (query)                        861.93       332.95     1_194.88       0.2699          1.2136         7.93
IVF-TQ-b2-nl223-np14-rf20 (query)                        861.93       608.88     1_470.80       0.4056          1.1279         7.93
IVF-TQ-b2-nl223-np21-rf10 (query)                        861.93       352.78     1_214.70       0.2699          1.2136         7.93
IVF-TQ-b2-nl223-np21-rf20 (query)                        861.93       669.44     1_531.37       0.4056          1.1279         7.93
IVF-TQ-b2-nl223 (self)                                   861.93     1_190.18     2_052.11       0.4070          1.1561         7.93
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_043.99       134.49     1_178.48       0.0663             NaN         8.11
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_043.99       136.44     1_180.43       0.0663             NaN         8.11
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_043.99       167.40     1_211.39       0.0662             NaN         8.11
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_043.99       330.28     1_374.27       0.2707          1.2128         8.11
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_043.99       601.02     1_645.01       0.4072          1.1271         8.11
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_043.99       345.08     1_389.07       0.2702          1.2133         8.11
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_043.99       604.50     1_648.48       0.4061          1.1277         8.11
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_043.99       353.92     1_397.91       0.2699          1.2136         8.11
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_043.99       609.36     1_653.35       0.4056          1.1279         8.11
IVF-TQ-b2-nl316 (self)                                 1_043.99     1_190.12     2_234.11       0.4070          1.1561         8.11
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_388.83       150.31     1_539.14       0.0872             NaN        14.07
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_388.83       171.02     1_559.85       0.0871             NaN        14.07
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_388.83       199.69     1_588.52       0.0871             NaN        14.07
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_388.83       367.06     1_755.89       0.2865          1.1965        14.07
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_388.83       694.50     2_083.32       0.4170          1.1210        14.07
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_388.83       391.49     1_780.32       0.2865          1.1965        14.07
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_388.83       747.28     2_136.11       0.4170          1.1210        14.07
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_388.83       428.92     1_817.75       0.2865          1.1965        14.07
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_388.83       823.87     2_212.69       0.4170          1.1210        14.07
IVF-TQ-b4-nl158 (self)                                 1_388.83     1_224.88     2_613.71       0.4165          1.1485        14.07
IVF-TQ-b4-nl223-np11-rf0 (query)                         924.31       162.81     1_087.12       0.0873             NaN        14.24
IVF-TQ-b4-nl223-np14-rf0 (query)                         924.31       173.87     1_098.18       0.0871             NaN        14.24
IVF-TQ-b4-nl223-np21-rf0 (query)                         924.31       215.33     1_139.64       0.0871             NaN        14.24
IVF-TQ-b4-nl223-np11-rf10 (query)                        924.31       353.90     1_278.21       0.2876          1.1957        14.24
IVF-TQ-b4-nl223-np11-rf20 (query)                        924.31       636.33     1_560.64       0.4188          1.1202        14.24
IVF-TQ-b4-nl223-np14-rf10 (query)                        924.31       368.25     1_292.56       0.2866          1.1964        14.24
IVF-TQ-b4-nl223-np14-rf20 (query)                        924.31       658.09     1_582.40       0.4171          1.1210        14.24
IVF-TQ-b4-nl223-np21-rf10 (query)                        924.31       421.34     1_345.65       0.2865          1.1965        14.24
IVF-TQ-b4-nl223-np21-rf20 (query)                        924.31       730.85     1_655.15       0.4170          1.1210        14.24
IVF-TQ-b4-nl223 (self)                                   924.31     1_230.51     2_154.82       0.4166          1.1485        14.24
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_119.04       173.88     1_292.92       0.0872             NaN        14.51
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_119.04       183.31     1_302.35       0.0872             NaN        14.51
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_119.04       211.07     1_330.11       0.0871             NaN        14.51
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_119.04       361.85     1_480.89       0.2873          1.1959        14.51
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_119.04       615.74     1_734.78       0.4183          1.1204        14.51
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_119.04       367.71     1_486.75       0.2868          1.1962        14.51
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_119.04       628.82     1_747.86       0.4174          1.1208        14.51
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_119.04       404.32     1_523.36       0.2865          1.1965        14.51
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_119.04       676.51     1_795.55       0.4170          1.1210        14.51
IVF-TQ-b4-nl316 (self)                                 1_119.04     1_243.47     2_362.51       0.4165          1.1485        14.51
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D - TurboQuant + IVF
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        69.60     9_529.81     9_599.41       1.0000          1.0000        97.66
Exhaustive (self)                                         69.60    32_869.07    32_938.67       1.0000          1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              487.11       663.20     1_150.31       0.0709             NaN        13.97
ExhaustiveTQ-b2-rf5 (query)                              487.11       758.64     1_245.74       0.1815          1.2341        13.97
ExhaustiveTQ-b2-rf10 (query)                             487.11       901.53     1_388.63       0.2475          1.1648        13.97
ExhaustiveTQ-b2-rf20 (query)                             487.11     1_318.02     1_805.13       0.3619          1.1046        13.97
ExhaustiveTQ-b2 (self)                                   487.11     4_355.61     4_842.71       0.3623          1.1225        13.97
ExhaustiveTQ-b4-rf0 (query)                              601.65     1_136.65     1_738.30       0.0862             NaN        26.18
ExhaustiveTQ-b4-rf5 (query)                              601.65     1_240.97     1_842.62       0.1892          1.2262        26.18
ExhaustiveTQ-b4-rf10 (query)                             601.65     1_381.48     1_983.12       0.2498          1.1620        26.18
ExhaustiveTQ-b4-rf20 (query)                             601.65     1_793.58     2_395.22       0.3584          1.1058        26.18
ExhaustiveTQ-b4 (self)                                   601.65     5_960.42     6_562.06       0.3580          1.1245        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        2_878.71       224.79     3_103.50       0.0709             NaN        14.98
IVF-TQ-b2-nl158-np12-rf0 (query)                       2_878.71       243.72     3_122.43       0.0709             NaN        14.98
IVF-TQ-b2-nl158-np17-rf0 (query)                       2_878.71       251.62     3_130.33       0.0709             NaN        14.98
IVF-TQ-b2-nl158-np7-rf10 (query)                       2_878.71       488.71     3_367.43       0.2475          1.1648        14.98
IVF-TQ-b2-nl158-np7-rf20 (query)                       2_878.71       805.78     3_684.49       0.3619          1.1046        14.98
IVF-TQ-b2-nl158-np12-rf10 (query)                      2_878.71       465.68     3_344.39       0.2475          1.1648        14.98
IVF-TQ-b2-nl158-np12-rf20 (query)                      2_878.71       828.94     3_707.65       0.3619          1.1046        14.98
IVF-TQ-b2-nl158-np17-rf10 (query)                      2_878.71       494.91     3_373.62       0.2475          1.1648        14.98
IVF-TQ-b2-nl158-np17-rf20 (query)                      2_878.71       877.68     3_756.39       0.3619          1.1046        14.98
IVF-TQ-b2-nl158 (self)                                 2_878.71     1_626.76     4_505.47       0.3623          1.1225        14.98
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_689.92       243.51     1_933.42       0.0709             NaN        15.19
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_689.92       255.93     1_945.84       0.0709             NaN        15.19
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_689.92       278.37     1_968.29       0.0709             NaN        15.19
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_689.92       461.89     2_151.81       0.2475          1.1648        15.19
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_689.92       906.32     2_596.23       0.3619          1.1046        15.19
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_689.92       680.26     2_370.18       0.2475          1.1648        15.19
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_689.92       908.81     2_598.73       0.3619          1.1046        15.19
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_689.92       544.97     2_234.89       0.2475          1.1648        15.19
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_689.92       896.29     2_586.21       0.3619          1.1046        15.19
IVF-TQ-b2-nl223 (self)                                 1_689.92     1_877.12     3_567.03       0.3623          1.1225        15.19
IVF-TQ-b2-nl316-np15-rf0 (query)                       2_304.87       271.29     2_576.17       0.0709             NaN        15.55
IVF-TQ-b2-nl316-np17-rf0 (query)                       2_304.87       274.96     2_579.84       0.0709             NaN        15.55
IVF-TQ-b2-nl316-np25-rf0 (query)                       2_304.87       286.53     2_591.40       0.0709             NaN        15.55
IVF-TQ-b2-nl316-np15-rf10 (query)                      2_304.87       482.42     2_787.30       0.2475          1.1648        15.55
IVF-TQ-b2-nl316-np15-rf20 (query)                      2_304.87       752.95     3_057.82       0.3619          1.1046        15.55
IVF-TQ-b2-nl316-np17-rf10 (query)                      2_304.87       474.66     2_779.54       0.2475          1.1648        15.55
IVF-TQ-b2-nl316-np17-rf20 (query)                      2_304.87       770.05     3_074.92       0.3619          1.1046        15.55
IVF-TQ-b2-nl316-np25-rf10 (query)                      2_304.87       496.27     2_801.15       0.2475          1.1648        15.55
IVF-TQ-b2-nl316-np25-rf20 (query)                      2_304.87       801.66     3_106.53       0.3619          1.1046        15.55
IVF-TQ-b2-nl316 (self)                                 2_304.87     1_743.80     4_048.68       0.3623          1.1225        15.55
IVF-TQ-b4-nl158-np7-rf0 (query)                        2_990.86       296.75     3_287.61       0.0862             NaN        27.51
IVF-TQ-b4-nl158-np12-rf0 (query)                       2_990.86       320.19     3_311.05       0.0862             NaN        27.51
IVF-TQ-b4-nl158-np17-rf0 (query)                       2_990.86       348.58     3_339.44       0.0862             NaN        27.51
IVF-TQ-b4-nl158-np7-rf10 (query)                       2_990.86       553.18     3_544.04       0.2498          1.1620        27.51
IVF-TQ-b4-nl158-np7-rf20 (query)                       2_990.86       905.58     3_896.44       0.3584          1.1058        27.51
IVF-TQ-b4-nl158-np12-rf10 (query)                      2_990.86       578.01     3_568.87       0.2498          1.1620        27.51
IVF-TQ-b4-nl158-np12-rf20 (query)                      2_990.86       943.97     3_934.83       0.3584          1.1058        27.51
IVF-TQ-b4-nl158-np17-rf10 (query)                      2_990.86       614.56     3_605.42       0.2498          1.1620        27.51
IVF-TQ-b4-nl158-np17-rf20 (query)                      2_990.86       994.64     3_985.50       0.3584          1.1058        27.51
IVF-TQ-b4-nl158 (self)                                 2_990.86     1_834.24     4_825.10       0.3580          1.1245        27.51
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_823.18       322.23     2_145.41       0.0862             NaN        27.81
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_823.18       349.61     2_172.80       0.0862             NaN        27.81
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_823.18       387.81     2_210.99       0.0862             NaN        27.81
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_823.18       555.19     2_378.37       0.2498          1.1620        27.81
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_823.18       859.25     2_682.43       0.3584          1.1058        27.81
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_823.18       579.71     2_402.89       0.2498          1.1620        27.81
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_823.18       884.06     2_707.24       0.3584          1.1058        27.81
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_823.18       626.85     2_450.03       0.2498          1.1620        27.81
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_823.18       953.53     2_776.71       0.3584          1.1058        27.81
IVF-TQ-b4-nl223 (self)                                 1_823.18     1_876.73     3_699.91       0.3580          1.1245        27.81
IVF-TQ-b4-nl316-np15-rf0 (query)                       2_147.26       347.91     2_495.17       0.0862             NaN        28.33
IVF-TQ-b4-nl316-np17-rf0 (query)                       2_147.26       351.77     2_499.03       0.0862             NaN        28.33
IVF-TQ-b4-nl316-np25-rf0 (query)                       2_147.26       390.75     2_538.01       0.0862             NaN        28.33
IVF-TQ-b4-nl316-np15-rf10 (query)                      2_147.26       579.46     2_726.71       0.2498          1.1620        28.33
IVF-TQ-b4-nl316-np15-rf20 (query)                      2_147.26       845.19     2_992.45       0.3584          1.1058        28.33
IVF-TQ-b4-nl316-np17-rf10 (query)                      2_147.26       592.43     2_739.68       0.2498          1.1620        28.33
IVF-TQ-b4-nl316-np17-rf20 (query)                      2_147.26       872.41     3_019.67       0.3584          1.1058        28.33
IVF-TQ-b4-nl316-np25-rf10 (query)                      2_147.26       624.71     2_771.96       0.2498          1.1620        28.33
IVF-TQ-b4-nl316-np25-rf20 (query)                      2_147.26       918.01     3_065.27       0.3584          1.1058        28.33
IVF-TQ-b4-nl316 (self)                                 2_147.26     1_935.88     4_083.14       0.3580          1.1245        28.33
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D - TurboQuant + IVF
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       109.13    16_439.97    16_549.10       1.0000          1.0000       146.48
Exhaustive (self)                                        109.13    54_686.10    54_795.23       1.0000          1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              987.76     1_006.33     1_994.08       0.0719             NaN        21.33
ExhaustiveTQ-b2-rf5 (query)                              987.76     1_124.67     2_112.43       0.1764          1.1855        21.33
ExhaustiveTQ-b2-rf10 (query)                             987.76     1_280.35     2_268.10       0.2312          1.1365        21.33
ExhaustiveTQ-b2-rf20 (query)                             987.76     1_713.10     2_700.85       0.3300          1.0920        21.33
ExhaustiveTQ-b2 (self)                                   987.76     5_711.47     6_699.23       0.3296          1.1027        21.33
ExhaustiveTQ-b4-rf0 (query)                            1_138.14     1_820.91     2_959.05       0.0844             NaN        39.64
ExhaustiveTQ-b4-rf5 (query)                            1_138.14     1_917.51     3_055.66       0.1812          1.1813        39.64
ExhaustiveTQ-b4-rf10 (query)                           1_138.14     2_040.81     3_178.95       0.2330          1.1352        39.64
ExhaustiveTQ-b4-rf20 (query)                           1_138.14     2_477.29     3_615.43       0.3263          1.0942        39.64
ExhaustiveTQ-b4 (self)                                 1_138.14     8_231.09     9_369.23       0.3287          1.1030        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        4_564.82       399.19     4_964.01       0.0719             NaN        22.63
IVF-TQ-b2-nl158-np12-rf0 (query)                       4_564.82       422.40     4_987.22       0.0719             NaN        22.63
IVF-TQ-b2-nl158-np17-rf0 (query)                       4_564.82       442.27     5_007.09       0.0719             NaN        22.63
IVF-TQ-b2-nl158-np7-rf10 (query)                       4_564.82       671.37     5_236.19       0.2313          1.1365        22.63
IVF-TQ-b2-nl158-np7-rf20 (query)                       4_564.82     1_066.71     5_631.53       0.3300          1.0920        22.63
IVF-TQ-b2-nl158-np12-rf10 (query)                      4_564.82       695.65     5_260.47       0.2313          1.1365        22.63
IVF-TQ-b2-nl158-np12-rf20 (query)                      4_564.82     1_101.91     5_666.73       0.3300          1.0920        22.63
IVF-TQ-b2-nl158-np17-rf10 (query)                      4_564.82       716.77     5_281.59       0.2313          1.1365        22.63
IVF-TQ-b2-nl158-np17-rf20 (query)                      4_564.82     1_125.43     5_690.25       0.3300          1.0920        22.63
IVF-TQ-b2-nl158 (self)                                 4_564.82     2_345.20     6_910.02       0.3296          1.1027        22.63
IVF-TQ-b2-nl223-np11-rf0 (query)                       2_618.93       428.32     3_047.25       0.0719             NaN        22.99
IVF-TQ-b2-nl223-np14-rf0 (query)                       2_618.93       441.97     3_060.90       0.0719             NaN        22.99
IVF-TQ-b2-nl223-np21-rf0 (query)                       2_618.93       466.03     3_084.97       0.0719             NaN        22.99
IVF-TQ-b2-nl223-np11-rf10 (query)                      2_618.93       669.88     3_288.81       0.2312          1.1365        22.99
IVF-TQ-b2-nl223-np11-rf20 (query)                      2_618.93       999.39     3_618.33       0.3300          1.0920        22.99
IVF-TQ-b2-nl223-np14-rf10 (query)                      2_618.93       696.63     3_315.56       0.2313          1.1365        22.99
IVF-TQ-b2-nl223-np14-rf20 (query)                      2_618.93     1_064.28     3_683.21       0.3300          1.0920        22.99
IVF-TQ-b2-nl223-np21-rf10 (query)                      2_618.93       735.79     3_354.72       0.2313          1.1365        22.99
IVF-TQ-b2-nl223-np21-rf20 (query)                      2_618.93     1_068.56     3_687.49       0.3300          1.0920        22.99
IVF-TQ-b2-nl223 (self)                                 2_618.93     2_455.51     5_074.45       0.3296          1.1027        22.99
IVF-TQ-b2-nl316-np15-rf0 (query)                       3_155.87       457.75     3_613.62       0.0719             NaN        23.51
IVF-TQ-b2-nl316-np17-rf0 (query)                       3_155.87       462.11     3_617.99       0.0719             NaN        23.51
IVF-TQ-b2-nl316-np25-rf0 (query)                       3_155.87       489.23     3_645.10       0.0719             NaN        23.51
IVF-TQ-b2-nl316-np15-rf10 (query)                      3_155.87       694.73     3_850.60       0.2313          1.1365        23.51
IVF-TQ-b2-nl316-np15-rf20 (query)                      3_155.87     1_014.38     4_170.26       0.3300          1.0920        23.51
IVF-TQ-b2-nl316-np17-rf10 (query)                      3_155.87       696.44     3_852.31       0.2313          1.1365        23.51
IVF-TQ-b2-nl316-np17-rf20 (query)                      3_155.87     1_117.36     4_273.23       0.3300          1.0920        23.51
IVF-TQ-b2-nl316-np25-rf10 (query)                      3_155.87       735.18     3_891.06       0.2313          1.1365        23.51
IVF-TQ-b2-nl316-np25-rf20 (query)                      3_155.87     1_062.22     4_218.10       0.3300          1.0920        23.51
IVF-TQ-b2-nl316 (self)                                 3_155.87     2_543.08     5_698.95       0.3296          1.1027        23.51
IVF-TQ-b4-nl158-np7-rf0 (query)                        4_778.24       523.39     5_301.63       0.0844             NaN        41.40
IVF-TQ-b4-nl158-np12-rf0 (query)                       4_778.24       556.85     5_335.08       0.0844             NaN        41.40
IVF-TQ-b4-nl158-np17-rf0 (query)                       4_778.24       587.95     5_366.18       0.0844             NaN        41.40
IVF-TQ-b4-nl158-np7-rf10 (query)                       4_778.24       827.08     5_605.31       0.2329          1.1352        41.40
IVF-TQ-b4-nl158-np7-rf20 (query)                       4_778.24     1_212.08     5_990.32       0.3263          1.0942        41.40
IVF-TQ-b4-nl158-np12-rf10 (query)                      4_778.24       872.19     5_650.42       0.2330          1.1352        41.40
IVF-TQ-b4-nl158-np12-rf20 (query)                      4_778.24     1_273.44     6_051.67       0.3263          1.0942        41.40
IVF-TQ-b4-nl158-np17-rf10 (query)                      4_778.24       884.08     5_662.31       0.2329          1.1352        41.40
IVF-TQ-b4-nl158-np17-rf20 (query)                      4_778.24     1_336.48     6_114.72       0.3263          1.0942        41.40
IVF-TQ-b4-nl158 (self)                                 4_778.24     2_647.50     7_425.74       0.3287          1.1030        41.40
IVF-TQ-b4-nl223-np11-rf0 (query)                       2_812.85       563.65     3_376.50       0.0844             NaN        41.92
IVF-TQ-b4-nl223-np14-rf0 (query)                       2_812.85       582.06     3_394.91       0.0844             NaN        41.92
IVF-TQ-b4-nl223-np21-rf0 (query)                       2_812.85       628.86     3_441.71       0.0844             NaN        41.92
IVF-TQ-b4-nl223-np11-rf10 (query)                      2_812.85       816.06     3_628.91       0.2330          1.1352        41.92
IVF-TQ-b4-nl223-np11-rf20 (query)                      2_812.85     1_150.74     3_963.59       0.3263          1.0942        41.92
IVF-TQ-b4-nl223-np14-rf10 (query)                      2_812.85       843.06     3_655.91       0.2329          1.1352        41.92
IVF-TQ-b4-nl223-np14-rf20 (query)                      2_812.85     1_179.90     3_992.75       0.3264          1.0942        41.92
IVF-TQ-b4-nl223-np21-rf10 (query)                      2_812.85       906.42     3_719.27       0.2329          1.1352        41.92
IVF-TQ-b4-nl223-np21-rf20 (query)                      2_812.85     1_263.93     4_076.78       0.3264          1.0942        41.92
IVF-TQ-b4-nl223 (self)                                 2_812.85     2_740.87     5_553.72       0.3287          1.1030        41.92
IVF-TQ-b4-nl316-np15-rf0 (query)                       3_422.07       605.41     4_027.49       0.0844             NaN        42.69
IVF-TQ-b4-nl316-np17-rf0 (query)                       3_422.07       606.12     4_028.19       0.0844             NaN        42.69
IVF-TQ-b4-nl316-np25-rf0 (query)                       3_422.07       670.79     4_092.87       0.0844             NaN        42.69
IVF-TQ-b4-nl316-np15-rf10 (query)                      3_422.07       841.54     4_263.62       0.2330          1.1352        42.69
IVF-TQ-b4-nl316-np15-rf20 (query)                      3_422.07     1_163.33     4_585.41       0.3264          1.0942        42.69
IVF-TQ-b4-nl316-np17-rf10 (query)                      3_422.07       854.10     4_276.17       0.2330          1.1352        42.69
IVF-TQ-b4-nl316-np17-rf20 (query)                      3_422.07     1_175.81     4_597.89       0.3263          1.0942        42.69
IVF-TQ-b4-nl316-np25-rf10 (query)                      3_422.07       914.12     4_336.19       0.2329          1.1352        42.69
IVF-TQ-b4-nl316-np25-rf20 (query)                      3_422.07     1_251.64     4_673.72       0.3263          1.0942        42.69
IVF-TQ-b4-nl316 (self)                                 3_422.07     2_980.04     6_402.12       0.3287          1.1030        42.69
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

#### Cell embeddings data

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D - TurboQuant + IVF
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        33.37     4_181.68     4_215.05       1.0000          1.0000        48.83
Exhaustive (self)                                         33.37    14_136.82    14_170.19       1.0000          1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              184.90       372.55       557.45       0.7919             NaN         7.12
ExhaustiveTQ-b2-rf5 (query)                              184.90       459.79       644.69       0.9995          1.0000         7.12
ExhaustiveTQ-b2-rf10 (query)                             184.90       592.99       777.89       1.0000          1.0000         7.12
ExhaustiveTQ-b2-rf20 (query)                             184.90       992.81     1_177.71       1.0000          1.0000         7.12
ExhaustiveTQ-b2 (self)                                   184.90     3_282.07     3_466.97       1.0000          1.0000         7.12
ExhaustiveTQ-b4-rf0 (query)                              262.77       590.66       853.43       0.8727             NaN        13.22
ExhaustiveTQ-b4-rf5 (query)                              262.77       684.84       947.61       1.0000          1.0000        13.22
ExhaustiveTQ-b4-rf10 (query)                             262.77       814.70     1_077.46       1.0000          1.0000        13.22
ExhaustiveTQ-b4-rf20 (query)                             262.77     1_214.76     1_477.53       1.0000          1.0000        13.22
ExhaustiveTQ-b4 (self)                                   262.77     4_031.10     4_293.87       1.0000          1.0000        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_460.87       137.60     1_598.47       0.7916             NaN         7.78
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_460.87       181.98     1_642.85       0.7918             NaN         7.78
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_460.87       220.97     1_681.84       0.7919             NaN         7.78
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_460.87       348.43     1_809.31       0.9982          1.0004         7.78
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_460.87       634.46     2_095.34       0.9982          1.0004         7.78
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_460.87       411.37     1_872.24       1.0000          1.0000         7.78
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_460.87       739.18     2_200.05       1.0000          1.0000         7.78
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_460.87       493.57     1_954.44       1.0000          1.0000         7.78
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_460.87       877.11     2_337.98       1.0000          1.0000         7.78
IVF-TQ-b2-nl158 (self)                                 1_460.87     1_303.91     2_764.78       1.0000          1.0000         7.78
IVF-TQ-b2-nl223-np11-rf0 (query)                         731.15       138.24       869.39       0.7919             NaN         7.93
IVF-TQ-b2-nl223-np14-rf0 (query)                         731.15       155.66       886.82       0.7919             NaN         7.93
IVF-TQ-b2-nl223-np21-rf0 (query)                         731.15       194.11       925.27       0.7919             NaN         7.93
IVF-TQ-b2-nl223-np11-rf10 (query)                        731.15       339.20     1_070.35       0.9995          1.0001         7.93
IVF-TQ-b2-nl223-np11-rf20 (query)                        731.15       606.57     1_337.73       0.9995          1.0001         7.93
IVF-TQ-b2-nl223-np14-rf10 (query)                        731.15       363.76     1_094.92       0.9999          1.0000         7.93
IVF-TQ-b2-nl223-np14-rf20 (query)                        731.15       655.69     1_386.84       0.9999          1.0000         7.93
IVF-TQ-b2-nl223-np21-rf10 (query)                        731.15       418.25     1_149.41       1.0000          1.0000         7.93
IVF-TQ-b2-nl223-np21-rf20 (query)                        731.15       746.03     1_477.18       1.0000          1.0000         7.93
IVF-TQ-b2-nl223 (self)                                   731.15     1_133.71     1_864.86       1.0000          1.0000         7.93
IVF-TQ-b2-nl316-np15-rf0 (query)                         930.01       146.11     1_076.12       0.7919             NaN         8.12
IVF-TQ-b2-nl316-np17-rf0 (query)                         930.01       154.33     1_084.34       0.7918             NaN         8.12
IVF-TQ-b2-nl316-np25-rf0 (query)                         930.01       187.59     1_117.59       0.7919             NaN         8.12
IVF-TQ-b2-nl316-np15-rf10 (query)                        930.01       348.27     1_278.27       0.9998          1.0000         8.12
IVF-TQ-b2-nl316-np15-rf20 (query)                        930.01       616.05     1_546.06       0.9998          1.0000         8.12
IVF-TQ-b2-nl316-np17-rf10 (query)                        930.01       343.29     1_273.30       0.9999          1.0000         8.12
IVF-TQ-b2-nl316-np17-rf20 (query)                        930.01       621.27     1_551.28       0.9999          1.0000         8.12
IVF-TQ-b2-nl316-np25-rf10 (query)                        930.01       391.31     1_321.32       1.0000          1.0000         8.12
IVF-TQ-b2-nl316-np25-rf20 (query)                        930.01       692.43     1_622.44       1.0000          1.0000         8.12
IVF-TQ-b2-nl316 (self)                                   930.01     1_108.38     2_038.39       1.0000          1.0000         8.12
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_516.19       190.87     1_707.07       0.8721             NaN        14.02
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_516.19       266.86     1_783.05       0.8727             NaN        14.02
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_516.19       326.49     1_842.68       0.8727             NaN        14.02
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_516.19       407.13     1_923.32       0.9982          1.0004        14.02
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_516.19       703.18     2_219.38       0.9982          1.0004        14.02
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_516.19       504.43     2_020.62       1.0000          1.0000        14.02
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_516.19       894.13     2_410.32       1.0000          1.0000        14.02
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_516.19       576.44     2_092.63       1.0000          1.0000        14.02
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_516.19       923.78     2_439.97       1.0000          1.0000        14.02
IVF-TQ-b4-nl158 (self)                                 1_516.19     1_396.21     2_912.40       1.0000          1.0000        14.02
IVF-TQ-b4-nl223-np11-rf0 (query)                         801.54       191.65       993.19       0.8726             NaN        14.24
IVF-TQ-b4-nl223-np14-rf0 (query)                         801.54       220.34     1_021.88       0.8727             NaN        14.24
IVF-TQ-b4-nl223-np21-rf0 (query)                         801.54       283.41     1_084.95       0.8727             NaN        14.24
IVF-TQ-b4-nl223-np11-rf10 (query)                        801.54       390.83     1_192.37       0.9995          1.0001        14.24
IVF-TQ-b4-nl223-np11-rf20 (query)                        801.54       665.07     1_466.61       0.9995          1.0001        14.24
IVF-TQ-b4-nl223-np14-rf10 (query)                        801.54       431.41     1_232.96       0.9999          1.0000        14.24
IVF-TQ-b4-nl223-np14-rf20 (query)                        801.54       724.27     1_525.81       0.9999          1.0000        14.24
IVF-TQ-b4-nl223-np21-rf10 (query)                        801.54       507.97     1_309.51       1.0000          1.0000        14.24
IVF-TQ-b4-nl223-np21-rf20 (query)                        801.54       838.63     1_640.17       1.0000          1.0000        14.24
IVF-TQ-b4-nl223 (self)                                   801.54     1_247.34     2_048.88       1.0000          1.0000        14.24
IVF-TQ-b4-nl316-np15-rf0 (query)                         992.63       197.25     1_189.88       0.8727             NaN        14.53
IVF-TQ-b4-nl316-np17-rf0 (query)                         992.63       217.58     1_210.21       0.8727             NaN        14.53
IVF-TQ-b4-nl316-np25-rf0 (query)                         992.63       264.10     1_256.73       0.8727             NaN        14.53
IVF-TQ-b4-nl316-np15-rf10 (query)                        992.63       388.75     1_381.38       0.9998          1.0000        14.53
IVF-TQ-b4-nl316-np15-rf20 (query)                        992.63       651.25     1_643.88       0.9998          1.0000        14.53
IVF-TQ-b4-nl316-np17-rf10 (query)                        992.63       405.47     1_398.10       0.9999          1.0000        14.53
IVF-TQ-b4-nl316-np17-rf20 (query)                        992.63       737.94     1_730.57       0.9999          1.0000        14.53
IVF-TQ-b4-nl316-np25-rf10 (query)                        992.63       480.39     1_473.02       1.0000          1.0000        14.53
IVF-TQ-b4-nl316-np25-rf20 (query)                        992.63       791.20     1_783.83       1.0000          1.0000        14.53
IVF-TQ-b4-nl316 (self)                                   992.63     1_187.48     2_180.10       1.0000          1.0000        14.53
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D - TurboQuant + IVF
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        70.92     9_853.42     9_924.33       1.0000          1.0000        97.66
Exhaustive (self)                                         70.92    32_808.78    32_879.70       1.0000          1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              498.80       644.94     1_143.73       0.8424             NaN        13.97
ExhaustiveTQ-b2-rf5 (query)                              498.80       746.96     1_245.76       1.0000          1.0000        13.97
ExhaustiveTQ-b2-rf10 (query)                             498.80       902.57     1_401.36       1.0000          1.0000        13.97
ExhaustiveTQ-b2-rf20 (query)                             498.80     1_332.16     1_830.96       1.0000          1.0000        13.97
ExhaustiveTQ-b2 (self)                                   498.80     4_461.17     4_959.96       1.0000          1.0000        13.97
ExhaustiveTQ-b4-rf0 (query)                              603.92     1_123.69     1_727.61       0.8985             NaN        26.18
ExhaustiveTQ-b4-rf5 (query)                              603.92     1_226.03     1_829.95       1.0000          1.0000        26.18
ExhaustiveTQ-b4-rf10 (query)                             603.92     1_369.81     1_973.74       1.0000          1.0000        26.18
ExhaustiveTQ-b4-rf20 (query)                             603.92     1_813.06     2_416.99       1.0000          1.0000        26.18
ExhaustiveTQ-b4 (self)                                   603.92     6_024.79     6_628.71       1.0000          1.0000        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        3_268.01       262.81     3_530.82       0.8420             NaN        14.96
IVF-TQ-b2-nl158-np12-rf0 (query)                       3_268.01       332.59     3_600.60       0.8424             NaN        14.96
IVF-TQ-b2-nl158-np17-rf0 (query)                       3_268.01       392.20     3_660.21       0.8424             NaN        14.96
IVF-TQ-b2-nl158-np7-rf10 (query)                       3_268.01       506.30     3_774.31       0.9986          1.0003        14.96
IVF-TQ-b2-nl158-np7-rf20 (query)                       3_268.01       823.96     4_091.97       0.9986          1.0003        14.96
IVF-TQ-b2-nl158-np12-rf10 (query)                      3_268.01       591.50     3_859.51       0.9999          1.0000        14.96
IVF-TQ-b2-nl158-np12-rf20 (query)                      3_268.01       941.89     4_209.90       0.9999          1.0000        14.96
IVF-TQ-b2-nl158-np17-rf10 (query)                      3_268.01       661.95     3_929.96       1.0000          1.0000        14.96
IVF-TQ-b2-nl158-np17-rf20 (query)                      3_268.01     1_033.74     4_301.75       1.0000          1.0000        14.96
IVF-TQ-b2-nl158 (self)                                 3_268.01     1_746.42     5_014.43       1.0000          1.0000        14.96
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_393.67       276.55     1_670.22       0.8424             NaN        15.24
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_393.67       307.70     1_701.37       0.8424             NaN        15.24
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_393.67       376.42     1_770.09       0.8424             NaN        15.24
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_393.67       500.37     1_894.04       0.9997          1.0000        15.24
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_393.67       799.37     2_193.04       0.9997          1.0000        15.24
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_393.67       538.81     1_932.48       0.9999          1.0000        15.24
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_393.67       899.28     2_292.96       0.9999          1.0000        15.24
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_393.67       642.81     2_036.48       1.0000          1.0000        15.24
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_393.67       972.15     2_365.82       1.0000          1.0000        15.24
IVF-TQ-b2-nl223 (self)                                 1_393.67     1_693.90     3_087.58       1.0000          1.0000        15.24
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_624.96       286.44     1_911.40       0.8424             NaN        15.57
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_624.96       301.56     1_926.52       0.8424             NaN        15.57
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_624.96       353.91     1_978.87       0.8424             NaN        15.57
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_624.96       504.40     2_129.36       0.9999          1.0000        15.57
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_624.96       806.06     2_431.02       0.9999          1.0000        15.57
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_624.96       519.66     2_144.62       1.0000          1.0000        15.57
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_624.96       833.46     2_458.42       1.0000          1.0000        15.57
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_624.96       584.86     2_209.83       1.0000          1.0000        15.57
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_624.96       934.31     2_559.27       1.0000          1.0000        15.57
IVF-TQ-b2-nl316 (self)                                 1_624.96     1_674.48     3_299.44       1.0000          1.0000        15.57
IVF-TQ-b4-nl158-np7-rf0 (query)                        3_340.15       372.91     3_713.06       0.8977             NaN        27.46
IVF-TQ-b4-nl158-np12-rf0 (query)                       3_340.15       493.37     3_833.53       0.8985             NaN        27.46
IVF-TQ-b4-nl158-np17-rf0 (query)                       3_340.15       603.68     3_943.84       0.8985             NaN        27.46
IVF-TQ-b4-nl158-np7-rf10 (query)                       3_340.15       614.80     3_954.95       0.9986          1.0003        27.46
IVF-TQ-b4-nl158-np7-rf20 (query)                       3_340.15       938.07     4_278.22       0.9986          1.0003        27.46
IVF-TQ-b4-nl158-np12-rf10 (query)                      3_340.15       753.59     4_093.74       0.9999          1.0000        27.46
IVF-TQ-b4-nl158-np12-rf20 (query)                      3_340.15     1_113.45     4_453.60       0.9999          1.0000        27.46
IVF-TQ-b4-nl158-np17-rf10 (query)                      3_340.15       871.68     4_211.83       1.0000          1.0000        27.46
IVF-TQ-b4-nl158-np17-rf20 (query)                      3_340.15     1_241.48     4_581.64       1.0000          1.0000        27.46
IVF-TQ-b4-nl158 (self)                                 3_340.15     2_063.53     5_403.68       1.0000          1.0000        27.46
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_490.83       382.22     1_873.05       0.8984             NaN        27.90
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_490.83       434.18     1_925.01       0.8984             NaN        27.90
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_490.83       548.39     2_039.22       0.8985             NaN        27.90
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_490.83       609.74     2_100.57       0.9997          1.0000        27.90
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_490.83       911.67     2_402.50       0.9997          1.0000        27.90
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_490.83       668.58     2_159.41       0.9999          1.0000        27.90
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_490.83       979.13     2_469.95       0.9999          1.0000        27.90
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_490.83       783.66     2_274.49       1.0000          1.0000        27.90
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_490.83     1_142.35     2_633.17       1.0000          1.0000        27.90
IVF-TQ-b4-nl223 (self)                                 1_490.83     1_958.87     3_449.70       1.0000          1.0000        27.90
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_730.43       390.36     2_120.78       0.8984             NaN        28.38
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_730.43       419.70     2_150.13       0.8984             NaN        28.38
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_730.43       513.15     2_243.58       0.8985             NaN        28.38
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_730.43       615.17     2_345.59       0.9999          1.0000        28.38
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_730.43       920.88     2_651.31       0.9999          1.0000        28.38
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_730.43       644.94     2_375.36       1.0000          1.0000        28.38
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_730.43       958.67     2_689.09       1.0000          1.0000        28.38
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_730.43       749.89     2_480.31       1.0000          1.0000        28.38
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_730.43     1_092.87     2_823.30       1.0000          1.0000        28.38
IVF-TQ-b4-nl316 (self)                                 1_730.43     1_902.39     3_632.81       1.0000          1.0000        28.38
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D - TurboQuant + IVF
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       108.89    15_849.15    15_958.04       1.0000          1.0000       146.48
Exhaustive (self)                                        108.89    53_975.36    54_084.25       1.0000          1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                            1_005.52     1_013.99     2_019.51       0.8736             NaN        21.33
ExhaustiveTQ-b2-rf5 (query)                            1_005.52     1_139.65     2_145.17       1.0000          1.0000        21.33
ExhaustiveTQ-b2-rf10 (query)                           1_005.52     1_338.52     2_344.04       1.0000          1.0000        21.33
ExhaustiveTQ-b2-rf20 (query)                           1_005.52     1_757.77     2_763.29       1.0000          1.0000        21.33
ExhaustiveTQ-b2 (self)                                 1_005.52     5_836.71     6_842.23       1.0000          1.0000        21.33
ExhaustiveTQ-b4-rf0 (query)                            1_137.05     1_838.78     2_975.83       0.9097             NaN        39.64
ExhaustiveTQ-b4-rf5 (query)                            1_137.05     1_892.95     3_030.00       1.0000          1.0000        39.64
ExhaustiveTQ-b4-rf10 (query)                           1_137.05     2_045.42     3_182.47       1.0000          1.0000        39.64
ExhaustiveTQ-b4-rf20 (query)                           1_137.05     2_514.72     3_651.77       1.0000          1.0000        39.64
ExhaustiveTQ-b4 (self)                                 1_137.05     8_305.73     9_442.78       1.0000          1.0000        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        5_220.05       464.21     5_684.26       0.8735             NaN        22.61
IVF-TQ-b2-nl158-np12-rf0 (query)                       5_220.05       557.83     5_777.88       0.8736             NaN        22.61
IVF-TQ-b2-nl158-np17-rf0 (query)                       5_220.05       648.06     5_868.11       0.8736             NaN        22.61
IVF-TQ-b2-nl158-np7-rf10 (query)                       5_220.05       749.52     5_969.57       0.9995          1.0001        22.61
IVF-TQ-b2-nl158-np7-rf20 (query)                       5_220.05     1_081.60     6_301.65       0.9995          1.0001        22.61
IVF-TQ-b2-nl158-np12-rf10 (query)                      5_220.05       848.62     6_068.67       1.0000          1.0000        22.61
IVF-TQ-b2-nl158-np12-rf20 (query)                      5_220.05     1_246.32     6_466.37       1.0000          1.0000        22.61
IVF-TQ-b2-nl158-np17-rf10 (query)                      5_220.05       947.53     6_167.58       1.0000          1.0000        22.61
IVF-TQ-b2-nl158-np17-rf20 (query)                      5_220.05     1_332.01     6_552.06       1.0000          1.0000        22.61
IVF-TQ-b2-nl158 (self)                                 5_220.05     2_518.40     7_738.45       1.0000          1.0000        22.61
IVF-TQ-b2-nl223-np11-rf0 (query)                       2_425.99       477.86     2_903.86       0.8736             NaN        23.00
IVF-TQ-b2-nl223-np14-rf0 (query)                       2_425.99       542.99     2_968.98       0.8736             NaN        23.00
IVF-TQ-b2-nl223-np21-rf0 (query)                       2_425.99       593.37     3_019.36       0.8736             NaN        23.00
IVF-TQ-b2-nl223-np11-rf10 (query)                      2_425.99       735.07     3_161.06       0.9998          1.0000        23.00
IVF-TQ-b2-nl223-np11-rf20 (query)                      2_425.99     1_073.86     3_499.86       0.9998          1.0000        23.00
IVF-TQ-b2-nl223-np14-rf10 (query)                      2_425.99       777.78     3_203.77       1.0000          1.0000        23.00
IVF-TQ-b2-nl223-np14-rf20 (query)                      2_425.99     1_131.86     3_557.86       1.0000          1.0000        23.00
IVF-TQ-b2-nl223-np21-rf10 (query)                      2_425.99       868.07     3_294.06       1.0000          1.0000        23.00
IVF-TQ-b2-nl223-np21-rf20 (query)                      2_425.99     1_280.09     3_706.09       1.0000          1.0000        23.00
IVF-TQ-b2-nl223 (self)                                 2_425.99     2_513.19     4_939.18       1.0000          1.0000        23.00
IVF-TQ-b2-nl316-np15-rf0 (query)                       2_680.00       510.89     3_190.89       0.8736             NaN        23.53
IVF-TQ-b2-nl316-np17-rf0 (query)                       2_680.00       524.57     3_204.57       0.8736             NaN        23.53
IVF-TQ-b2-nl316-np25-rf0 (query)                       2_680.00       597.51     3_277.51       0.8736             NaN        23.53
IVF-TQ-b2-nl316-np15-rf10 (query)                      2_680.00       746.27     3_426.27       1.0000          1.0000        23.53
IVF-TQ-b2-nl316-np15-rf20 (query)                      2_680.00     1_083.04     3_763.04       1.0000          1.0000        23.53
IVF-TQ-b2-nl316-np17-rf10 (query)                      2_680.00       756.39     3_436.39       1.0000          1.0000        23.53
IVF-TQ-b2-nl316-np17-rf20 (query)                      2_680.00     1_111.45     3_791.45       1.0000          1.0000        23.53
IVF-TQ-b2-nl316-np25-rf10 (query)                      2_680.00       849.46     3_529.46       1.0000          1.0000        23.53
IVF-TQ-b2-nl316-np25-rf20 (query)                      2_680.00     1_219.01     3_899.01       1.0000          1.0000        23.53
IVF-TQ-b2-nl316 (self)                                 2_680.00     2_413.57     5_093.57       1.0000          1.0000        23.53
IVF-TQ-b4-nl158-np7-rf0 (query)                        5_324.74       636.94     5_961.68       0.9094             NaN        41.37
IVF-TQ-b4-nl158-np12-rf0 (query)                       5_324.74       818.88     6_143.62       0.9097             NaN        41.37
IVF-TQ-b4-nl158-np17-rf0 (query)                       5_324.74       975.44     6_300.17       0.9097             NaN        41.37
IVF-TQ-b4-nl158-np7-rf10 (query)                       5_324.74       921.17     6_245.91       0.9995          1.0001        41.37
IVF-TQ-b4-nl158-np7-rf20 (query)                       5_324.74     1_275.34     6_600.07       0.9995          1.0001        41.37
IVF-TQ-b4-nl158-np12-rf10 (query)                      5_324.74     1_104.81     6_429.55       1.0000          1.0000        41.37
IVF-TQ-b4-nl158-np12-rf20 (query)                      5_324.74     1_471.42     6_796.16       1.0000          1.0000        41.37
IVF-TQ-b4-nl158-np17-rf10 (query)                      5_324.74     1_254.54     6_579.28       1.0000          1.0000        41.37
IVF-TQ-b4-nl158-np17-rf20 (query)                      5_324.74     1_656.51     6_981.25       1.0000          1.0000        41.37
IVF-TQ-b4-nl158 (self)                                 5_324.74     3_015.25     8_339.99       1.0000          1.0000        41.37
IVF-TQ-b4-nl223-np11-rf0 (query)                       2_468.52       653.21     3_121.73       0.9096             NaN        41.96
IVF-TQ-b4-nl223-np14-rf0 (query)                       2_468.52       715.73     3_184.25       0.9097             NaN        41.96
IVF-TQ-b4-nl223-np21-rf0 (query)                       2_468.52       887.88     3_356.40       0.9097             NaN        41.96
IVF-TQ-b4-nl223-np11-rf10 (query)                      2_468.52       904.52     3_373.05       0.9998          1.0000        41.96
IVF-TQ-b4-nl223-np11-rf20 (query)                      2_468.52     1_267.74     3_736.26       0.9998          1.0000        41.96
IVF-TQ-b4-nl223-np14-rf10 (query)                      2_468.52     1_004.07     3_472.59       1.0000          1.0000        41.96
IVF-TQ-b4-nl223-np14-rf20 (query)                      2_468.52     1_325.92     3_794.45       1.0000          1.0000        41.96
IVF-TQ-b4-nl223-np21-rf10 (query)                      2_468.52     1_141.80     3_610.33       1.0000          1.0000        41.96
IVF-TQ-b4-nl223-np21-rf20 (query)                      2_468.52     1_528.01     3_996.53       1.0000          1.0000        41.96
IVF-TQ-b4-nl223 (self)                                 2_468.52     2_886.06     5_354.58       1.0000          1.0000        41.96
IVF-TQ-b4-nl316-np15-rf0 (query)                       2_828.41       712.89     3_541.30       0.9097             NaN        42.73
IVF-TQ-b4-nl316-np17-rf0 (query)                       2_828.41       720.50     3_548.91       0.9097             NaN        42.73
IVF-TQ-b4-nl316-np25-rf0 (query)                       2_828.41       838.55     3_666.97       0.9097             NaN        42.73
IVF-TQ-b4-nl316-np15-rf10 (query)                      2_828.41       910.15     3_738.56       1.0000          1.0000        42.73
IVF-TQ-b4-nl316-np15-rf20 (query)                      2_828.41     1_248.34     4_076.76       1.0000          1.0000        42.73
IVF-TQ-b4-nl316-np17-rf10 (query)                      2_828.41       945.20     3_773.61       1.0000          1.0000        42.73
IVF-TQ-b4-nl316-np17-rf20 (query)                      2_828.41     1_306.19     4_134.60       1.0000          1.0000        42.73
IVF-TQ-b4-nl316-np25-rf10 (query)                      2_828.41     1_100.12     3_928.53       1.0000          1.0000        42.73
IVF-TQ-b4-nl316-np25-rf20 (query)                      2_828.41     1_476.31     4_304.73       1.0000          1.0000        42.73
IVF-TQ-b4-nl316 (self)                                 2_828.41     2_846.59     5_675.01       1.0000          1.0000        42.73
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
