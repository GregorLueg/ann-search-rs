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
===================================================================================================================================
Benchmark: 50k samples, 256D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         9.90     4_308.28     4_318.18       1.0000          1.0000        48.83
Exhaustive (self)                                          9.90    14_756.02    14_765.92       1.0000          1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_578.93       287.72     2_866.65       0.0311             NaN         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_578.93       371.13     2_950.07       0.1623          1.1262         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_578.93       473.33     3_052.26       0.2695          1.0812         1.78
ExhaustiveBinary-256-random (self)                     2_578.93     1_234.52     3_813.46       0.1681          1.1202         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_793.16       292.26     3_085.41       0.1873             NaN         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_793.16       403.04     3_196.20       0.5340          1.0265         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_793.16       508.90     3_302.05       0.6684          1.0147         1.78
ExhaustiveBinary-256-pca (self)                        2_793.16     1_338.36     4_131.52       0.5319          1.0270         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_146.26       475.74     5_622.00       0.0628             NaN         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_146.26       604.40     5_750.65       0.2086          1.0932         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_146.26       713.03     5_859.29       0.3233          1.0582         3.55
ExhaustiveBinary-512-random (self)                     5_146.26     1_906.09     7_052.34       0.2130          1.0890         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_481.29       461.57     5_942.87       0.2013             NaN         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_481.29       578.98     6_060.28       0.6344          1.0175         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_481.29       693.23     6_174.52       0.8009          1.0073         3.55
ExhaustiveBinary-512-pca (self)                        5_481.29     1_912.38     7_393.68       0.6350          1.0176         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_106.40       791.61    10_898.01       0.0901             NaN         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_106.40       900.49    11_006.90       0.2539          1.0733         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_106.40     1_020.54    11_126.94       0.3803          1.0456         7.10
ExhaustiveBinary-1024-random (self)                   10_106.40     2_982.20    13_088.60       0.2560          1.0731         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_692.38       808.88    11_501.27       0.2080             NaN         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_692.38       940.68    11_633.06       0.6484          1.0164         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_692.38     1_049.29    11_741.68       0.8112          1.0067         7.10
ExhaustiveBinary-1024-pca (self)                      10_692.38     3_093.98    13_786.37       0.6483          1.0165         7.10
ExhaustiveBinary-256-signed_no_rr (query)              2_584.53       292.27     2_876.79       0.0311             NaN         1.78
ExhaustiveBinary-256-signed-rf10 (query)               2_584.53       376.55     2_961.07       0.1623          1.1262         1.78
ExhaustiveBinary-256-signed-rf20 (query)               2_584.53       483.93     3_068.46       0.2695          1.0812         1.78
ExhaustiveBinary-256-signed (self)                     2_584.53     1_238.37     3_822.90       0.1681          1.1202         1.78
IVF-Binary-256-nl158-np7-rf0-random (query)            4_050.95       115.79     4_166.74       0.0553             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           4_050.95       123.02     4_173.97       0.0419             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           4_050.95       129.82     4_180.77       0.0336             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           4_050.95       171.26     4_222.21       0.2371          1.0915         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           4_050.95       220.43     4_271.38       0.3629          1.0576         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          4_050.95       176.23     4_227.18       0.1990          1.1180         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          4_050.95       233.09     4_284.04       0.3175          1.0724         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          4_050.95       186.56     4_237.51       0.1717          1.1366         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          4_050.95       242.96     4_293.91       0.2820          1.0860         1.93
IVF-Binary-256-nl158-random (self)                     4_050.95       522.90     4_573.85       0.2048          1.1126         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           3_088.80       122.00     3_210.80       0.0524             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           3_088.80       125.78     3_214.58       0.0407             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           3_088.80       129.97     3_218.77       0.0324             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          3_088.80       176.09     3_264.89       0.2308          1.0969         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          3_088.80       231.22     3_320.03       0.3557          1.0603         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          3_088.80       181.71     3_270.51       0.1978          1.1184         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          3_088.80       233.25     3_322.05       0.3168          1.0721         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          3_088.80       201.92     3_290.72       0.1716          1.1362         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          3_088.80       245.37     3_334.17       0.2840          1.0839         2.00
IVF-Binary-256-nl223-random (self)                     3_088.80       532.04     3_620.84       0.2034          1.1133         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           3_319.28       127.56     3_446.84       0.0421             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           3_319.28       130.19     3_449.46       0.0388             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           3_319.28       140.50     3_459.78       0.0349             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          3_319.28       183.71     3_502.99       0.2022          1.1114         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          3_319.28       232.19     3_551.47       0.3241          1.0690         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          3_319.28       181.57     3_500.85       0.1921          1.1168         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          3_319.28       239.73     3_559.01       0.3111          1.0727         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          3_319.28       188.60     3_507.87       0.1776          1.1270         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          3_319.28       247.16     3_566.44       0.2917          1.0798         2.09
IVF-Binary-256-nl316-random (self)                     3_319.28       550.89     3_870.16       0.1966          1.1118         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               4_293.56       128.63     4_422.19       0.1990             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              4_293.56       129.69     4_423.26       0.1973             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              4_293.56       136.85     4_430.41       0.1966             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              4_293.56       190.15     4_483.71       0.6300          1.0178         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              4_293.56       243.80     4_537.37       0.7968          1.0074         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             4_293.56       200.04     4_493.61       0.6193          1.0187         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             4_293.56       260.58     4_554.14       0.7838          1.0081         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             4_293.56       212.82     4_506.38       0.6122          1.0192         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             4_293.56       278.87     4_572.44       0.7743          1.0085         1.93
IVF-Binary-256-nl158-pca (self)                        4_293.56       629.69     4_923.25       0.6187          1.0189         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              3_307.32       129.05     3_436.37       0.1984             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              3_307.32       131.70     3_439.02       0.1972             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              3_307.32       139.16     3_446.48       0.1961             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             3_307.32       198.80     3_506.12       0.6277          1.0179         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             3_307.32       249.25     3_556.57       0.7944          1.0075         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             3_307.32       202.94     3_510.26       0.6216          1.0184         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             3_307.32       262.29     3_569.61       0.7868          1.0079         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             3_307.32       224.78     3_532.10       0.6139          1.0191         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             3_307.32       279.13     3_586.45       0.7769          1.0084         2.00
IVF-Binary-256-nl223-pca (self)                        3_307.32       614.41     3_921.73       0.6208          1.0187         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              3_499.65       135.84     3_635.49       0.1988             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              3_499.65       137.58     3_637.23       0.1982             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              3_499.65       143.59     3_643.24       0.1970             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             3_499.65       208.79     3_708.44       0.6287          1.0179         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             3_499.65       263.63     3_763.28       0.7957          1.0075         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             3_499.65       206.34     3_705.99       0.6250          1.0182         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             3_499.65       265.10     3_764.75       0.7913          1.0077         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             3_499.65       224.34     3_723.99       0.6174          1.0188         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             3_499.65       279.40     3_779.05       0.7816          1.0082         2.09
IVF-Binary-256-nl316-pca (self)                        3_499.65       626.07     4_125.72       0.6245          1.0184         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_572.15       211.93     6_784.08       0.0790             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_572.15       230.01     6_802.16       0.0704             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_572.15       231.38     6_803.53       0.0642             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_572.15       277.64     6_849.79       0.2477          1.0769         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_572.15       324.70     6_896.85       0.3704          1.0480         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_572.15       286.76     6_858.91       0.2271          1.0864         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_572.15       345.95     6_918.10       0.3453          1.0537         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_572.15       303.61     6_875.76       0.2144          1.0929         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_572.15       358.27     6_930.42       0.3315          1.0572         3.71
IVF-Binary-512-nl158-random (self)                     6_572.15       883.30     7_455.44       0.2315          1.0829         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_606.12       218.30     5_824.42       0.0777             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_606.12       224.52     5_830.65       0.0708             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_606.12       231.90     5_838.03       0.0644             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_606.12       279.75     5_885.87       0.2450          1.0773         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_606.12       333.81     5_939.93       0.3685          1.0481         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_606.12       290.98     5_897.10       0.2293          1.0845         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_606.12       343.45     5_949.57       0.3486          1.0527         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_606.12       301.19     5_907.31       0.2146          1.0925         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_606.12       363.08     5_969.20       0.3307          1.0575         3.77
IVF-Binary-512-nl223-random (self)                     5_606.12       895.27     6_501.39       0.2330          1.0812         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           5_824.85       225.07     6_049.92       0.0705             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           5_824.85       230.64     6_055.49       0.0686             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           5_824.85       235.82     6_060.67       0.0659             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          5_824.85       285.04     6_109.89       0.2320          1.0829         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          5_824.85       349.43     6_174.28       0.3543          1.0514         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          5_824.85       288.72     6_113.57       0.2271          1.0849         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          5_824.85       344.40     6_169.25       0.3475          1.0529         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          5_824.85       306.66     6_131.51       0.2175          1.0903         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          5_824.85       357.21     6_182.06       0.3343          1.0564         3.86
IVF-Binary-512-nl316-random (self)                     5_824.85       901.53     6_726.38       0.2310          1.0817         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               6_870.67       223.72     7_094.39       0.2032             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              6_870.67       231.42     7_102.09       0.2018             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              6_870.67       241.21     7_111.88       0.2016             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              6_870.67       289.98     7_160.66       0.6396          1.0170         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              6_870.67       341.56     7_212.23       0.8063          1.0070         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             6_870.67       304.98     7_175.65       0.6351          1.0174         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             6_870.67       361.83     7_232.50       0.8013          1.0072         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             6_870.67       319.40     7_190.07       0.6347          1.0175         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             6_870.67       380.91     7_251.59       0.8011          1.0073         3.71
IVF-Binary-512-nl158-pca (self)                        6_870.67       949.13     7_819.81       0.6356          1.0176         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              5_965.03       228.18     6_193.21       0.2032             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              5_965.03       245.82     6_210.85       0.2023             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              5_965.03       243.92     6_208.95       0.2015             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             5_965.03       295.51     6_260.54       0.6394          1.0170         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             5_965.03       350.14     6_315.17       0.8068          1.0070         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             5_965.03       304.78     6_269.81       0.6367          1.0173         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             5_965.03       364.83     6_329.86       0.8034          1.0071         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             5_965.03       320.57     6_285.60       0.6344          1.0175         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             5_965.03       381.26     6_346.29       0.8010          1.0073         3.77
IVF-Binary-512-nl223-pca (self)                        5_965.03       961.11     6_926.13       0.6369          1.0174         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              6_141.95       234.35     6_376.30       0.2030             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              6_141.95       239.48     6_381.43       0.2026             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              6_141.95       245.52     6_387.46       0.2017             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             6_141.95       308.05     6_449.99       0.6392          1.0171         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             6_141.95       362.88     6_504.83       0.8063          1.0070         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             6_141.95       309.22     6_451.17       0.6377          1.0172         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             6_141.95       364.50     6_506.44       0.8044          1.0071         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             6_141.95       317.34     6_459.29       0.6353          1.0174         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             6_141.95       382.26     6_524.21       0.8016          1.0072         3.86
IVF-Binary-512-nl316-pca (self)                        6_141.95       957.84     7_099.78       0.6378          1.0173         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_484.48       407.63    11_892.11       0.0964             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_484.48       425.30    11_909.78       0.0933             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_484.48       444.62    11_929.10       0.0909             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_484.48       471.30    11_955.78       0.2765          1.0662         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_484.48       525.33    12_009.81       0.4097          1.0410         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_484.48       507.65    11_992.13       0.2643          1.0701         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_484.48       550.73    12_035.21       0.3922          1.0437         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_484.48       505.53    11_990.01       0.2577          1.0723         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_484.48       571.45    12_055.93       0.3852          1.0449         7.26
IVF-Binary-1024-nl158-random (self)                   11_484.48     1_559.12    13_043.60       0.2668          1.0699         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_570.56       415.39    10_985.96       0.0967             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_570.56       435.67    11_006.23       0.0936             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_570.56       440.88    11_011.44       0.0909             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_570.56       488.34    11_058.90       0.2759          1.0662         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_570.56       533.83    11_104.39       0.4080          1.0410         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_570.56       488.30    11_058.86       0.2664          1.0694         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_570.56       554.34    11_124.90       0.3953          1.0431         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_570.56       509.78    11_080.34       0.2579          1.0726         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_570.56       572.25    11_142.82       0.3844          1.0451         7.32
IVF-Binary-1024-nl223-random (self)                   10_570.56     1_555.48    12_126.04       0.2684          1.0693         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         10_764.21       419.97    11_184.18       0.0936             NaN         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)         10_764.21       426.87    11_191.08       0.0928             NaN         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)         10_764.21       433.53    11_197.74       0.0914             NaN         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)        10_764.21       485.21    11_249.42       0.2690          1.0685         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)        10_764.21       539.77    11_303.98       0.4008          1.0422         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)        10_764.21       483.45    11_247.66       0.2655          1.0696         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)        10_764.21       547.00    11_311.20       0.3954          1.0431         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)        10_764.21       504.97    11_269.17       0.2593          1.0721         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)        10_764.21       562.57    11_326.78       0.3856          1.0450         7.41
IVF-Binary-1024-nl316-random (self)                   10_764.21     1_566.19    12_330.39       0.2673          1.0695         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)             12_083.78       429.48    12_513.26       0.2100             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            12_083.78       448.85    12_532.62       0.2088             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            12_083.78       460.13    12_543.90       0.2085             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            12_083.78       502.29    12_586.07       0.6531          1.0160         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            12_083.78       543.78    12_627.56       0.8162          1.0065         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           12_083.78       507.94    12_591.72       0.6490          1.0163         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           12_083.78       615.16    12_698.94       0.8117          1.0067         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           12_083.78       531.65    12_615.43       0.6488          1.0163         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           12_083.78       592.33    12_676.10       0.8114          1.0067         7.26
IVF-Binary-1024-nl158-pca (self)                      12_083.78     1_624.80    13_708.57       0.6490          1.0165         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            11_131.17       430.07    11_561.24       0.2099             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            11_131.17       439.49    11_570.66       0.2092             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            11_131.17       464.34    11_595.51       0.2084             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           11_131.17       497.34    11_628.51       0.6525          1.0160         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           11_131.17       552.14    11_683.31       0.8164          1.0065         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           11_131.17       507.28    11_638.45       0.6497          1.0162         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           11_131.17       571.84    11_703.01       0.8135          1.0066         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           11_131.17       531.14    11_662.32       0.6480          1.0164         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           11_131.17       592.66    11_723.83       0.8112          1.0067         7.32
IVF-Binary-1024-nl223-pca (self)                      11_131.17     1_617.22    12_748.39       0.6502          1.0163         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            11_334.93       442.34    11_777.27       0.2098             NaN         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            11_334.93       450.98    11_785.91       0.2093             NaN         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            11_334.93       456.10    11_791.03       0.2086             NaN         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           11_334.93       511.42    11_846.35       0.6523          1.0160         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           11_334.93       568.43    11_903.36       0.8158          1.0065         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           11_334.93       508.15    11_843.08       0.6508          1.0161         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           11_334.93       582.41    11_917.34       0.8141          1.0066         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           11_334.93       524.36    11_859.29       0.6487          1.0164         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           11_334.93       612.80    11_947.73       0.8115          1.0067         7.42
IVF-Binary-1024-nl316-pca (self)                      11_334.93     1_622.70    12_957.63       0.6510          1.0163         7.42
IVF-Binary-256-nl158-np7-rf0-signed (query)            4_043.62       119.52     4_163.14       0.0553             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-signed (query)           4_043.62       122.05     4_165.68       0.0419             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-signed (query)           4_043.62       130.06     4_173.68       0.0336             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-signed (query)           4_043.62       169.52     4_213.14       0.2371          1.0915         1.93
IVF-Binary-256-nl158-np7-rf20-signed (query)           4_043.62       219.48     4_263.10       0.3629          1.0576         1.93
IVF-Binary-256-nl158-np12-rf10-signed (query)          4_043.62       190.22     4_233.84       0.1990          1.1180         1.93
IVF-Binary-256-nl158-np12-rf20-signed (query)          4_043.62       230.08     4_273.71       0.3175          1.0724         1.93
IVF-Binary-256-nl158-np17-rf10-signed (query)          4_043.62       188.15     4_231.77       0.1717          1.1366         1.93
IVF-Binary-256-nl158-np17-rf20-signed (query)          4_043.62       241.36     4_284.98       0.2820          1.0860         1.93
IVF-Binary-256-nl158-signed (self)                     4_043.62       519.36     4_562.98       0.2048          1.1126         1.93
IVF-Binary-256-nl223-np11-rf0-signed (query)           3_098.38       122.01     3_220.40       0.0524             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-signed (query)           3_098.38       124.07     3_222.46       0.0407             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-signed (query)           3_098.38       130.33     3_228.71       0.0324             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-signed (query)          3_098.38       177.29     3_275.67       0.2308          1.0969         2.00
IVF-Binary-256-nl223-np11-rf20-signed (query)          3_098.38       227.75     3_326.13       0.3557          1.0603         2.00
IVF-Binary-256-nl223-np14-rf10-signed (query)          3_098.38       178.80     3_277.18       0.1978          1.1184         2.00
IVF-Binary-256-nl223-np14-rf20-signed (query)          3_098.38       235.72     3_334.10       0.3168          1.0721         2.00
IVF-Binary-256-nl223-np21-rf10-signed (query)          3_098.38       191.94     3_290.32       0.1716          1.1362         2.00
IVF-Binary-256-nl223-np21-rf20-signed (query)          3_098.38       244.74     3_343.13       0.2840          1.0839         2.00
IVF-Binary-256-nl223-signed (self)                     3_098.38       532.82     3_631.21       0.2034          1.1133         2.00
IVF-Binary-256-nl316-np15-rf0-signed (query)           3_302.25       127.03     3_429.28       0.0421             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-signed (query)           3_302.25       128.70     3_430.95       0.0388             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-signed (query)           3_302.25       136.07     3_438.32       0.0349             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-signed (query)          3_302.25       185.51     3_487.76       0.2022          1.1114         2.09
IVF-Binary-256-nl316-np15-rf20-signed (query)          3_302.25       232.82     3_535.07       0.3241          1.0690         2.09
IVF-Binary-256-nl316-np17-rf10-signed (query)          3_302.25       186.53     3_488.77       0.1921          1.1168         2.09
IVF-Binary-256-nl316-np17-rf20-signed (query)          3_302.25       234.71     3_536.96       0.3111          1.0727         2.09
IVF-Binary-256-nl316-np25-rf10-signed (query)          3_302.25       188.39     3_490.64       0.1776          1.1270         2.09
IVF-Binary-256-nl316-np25-rf20-signed (query)          3_302.25       244.41     3_546.65       0.2917          1.0798         2.09
IVF-Binary-256-nl316-signed (self)                     3_302.25       546.02     3_848.27       0.1966          1.1118         2.09
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
Exhaustive (query)                                        19.93     9_458.86     9_478.79       1.0000          1.0000        97.66
Exhaustive (self)                                         19.93    32_504.85    32_524.78       1.0000          1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_696.24       377.33     6_073.57       0.0292             NaN         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_696.24       489.46     6_185.70       0.1469          1.0921         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_696.24       617.61     6_313.84       0.2447          1.0585         2.03
ExhaustiveBinary-256-random (self)                     5_696.24     1_615.33     7_311.57       0.1509          1.0875         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_108.24       387.48     6_495.72       0.1394             NaN         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_108.24       521.86     6_630.10       0.3908          1.0290         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_108.24       647.55     6_755.78       0.5142          1.0182         2.03
ExhaustiveBinary-256-pca (self)                        6_108.24     1_706.67     7_814.91       0.3906          1.0291         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_127.77       646.33    11_774.10       0.0610             NaN         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_127.77       765.14    11_892.90       0.1812          1.0668         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_127.77       892.88    12_020.64       0.2805          1.0428         4.05
ExhaustiveBinary-512-random (self)                    11_127.77     2_525.09    13_652.86       0.1844          1.0639         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_715.47       657.93    12_373.40       0.1676             NaN         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_715.47       789.59    12_505.07       0.4490          1.2486         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_715.47       921.15    12_636.62       0.5696          1.0152         4.05
ExhaustiveBinary-512-pca (self)                       11_715.47     2_616.12    14_331.60       0.4499          1.3208         4.05
ExhaustiveBinary-1024-random_no_rr (query)            22_036.60     1_174.05    23_210.65       0.0826             NaN         8.10
ExhaustiveBinary-1024-random-rf10 (query)             22_036.60     1_309.96    23_346.56       0.2064          1.0563         8.10
ExhaustiveBinary-1024-random-rf20 (query)             22_036.60     1_444.60    23_481.20       0.3140          1.0365         8.10
ExhaustiveBinary-1024-random (self)                   22_036.60     4_667.96    26_704.55       0.2075          1.0564         8.10
ExhaustiveBinary-1024-pca_no_rr (query)               22_937.20     1_202.02    24_139.22       0.2037             NaN         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                22_937.20     1_347.80    24_285.00       0.6282          1.0116         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                22_937.20     1_490.76    24_427.96       0.7931          1.0049         8.11
ExhaustiveBinary-1024-pca (self)                      22_937.20     4_481.34    27_418.54       0.6281          1.0116         8.11
ExhaustiveBinary-512-signed_no_rr (query)             11_036.43       642.75    11_679.18       0.0610             NaN         4.05
ExhaustiveBinary-512-signed-rf10 (query)              11_036.43       771.84    11_808.26       0.1812          1.0668         4.05
ExhaustiveBinary-512-signed-rf20 (query)              11_036.43       888.71    11_925.14       0.2805          1.0428         4.05
ExhaustiveBinary-512-signed (self)                    11_036.43     2_529.34    13_565.76       0.1844          1.0639         4.05
IVF-Binary-256-nl158-np7-rf0-random (query)            8_652.79       226.01     8_878.80       0.0552             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           8_652.79       232.40     8_885.19       0.0437             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           8_652.79       240.74     8_893.54       0.0311             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           8_652.79       304.53     8_957.32       0.2177          1.0627         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           8_652.79       380.19     9_032.98       0.3277          1.0402         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          8_652.79       309.18     8_961.97       0.1815          1.0819         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          8_652.79       390.66     9_043.45       0.2846          1.0520         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          8_652.79       319.79     8_972.58       0.1523          1.0975         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          8_652.79       403.52     9_056.32       0.2512          1.0606         2.34
IVF-Binary-256-nl158-random (self)                     8_652.79       952.92     9_605.71       0.1855          1.0777         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           6_516.37       237.08     6_753.45       0.0484             NaN         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)           6_516.37       243.45     6_759.82       0.0398             NaN         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)           6_516.37       249.16     6_765.54       0.0318             NaN         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)          6_516.37       317.47     6_833.84       0.2010          1.0697         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)          6_516.37       393.43     6_909.81       0.3126          1.0429         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)          6_516.37       317.96     6_834.33       0.1812          1.0789         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)          6_516.37       400.41     6_916.79       0.2892          1.0486         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)          6_516.37       325.61     6_841.98       0.1563          1.0924         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)          6_516.37       414.76     6_931.13       0.2570          1.0578         2.46
IVF-Binary-256-nl223-random (self)                     6_516.37       982.26     7_498.63       0.1845          1.0746         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)           6_821.84       250.88     7_072.72       0.0387             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           6_821.84       254.48     7_076.32       0.0357             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           6_821.84       257.54     7_079.38       0.0324             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          6_821.84       330.69     7_152.53       0.1803          1.0784         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          6_821.84       410.28     7_232.12       0.2936          1.0472         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          6_821.84       331.48     7_153.32       0.1748          1.0808         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          6_821.84       414.80     7_236.64       0.2853          1.0488         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          6_821.84       334.08     7_155.92       0.1598          1.0890         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          6_821.84       420.20     7_242.04       0.2631          1.0553         2.65
IVF-Binary-256-nl316-random (self)                     6_821.84     1_023.24     7_845.08       0.1791          1.0763         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               9_083.24       235.57     9_318.81       0.1473             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              9_083.24       242.22     9_325.45       0.1458             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              9_083.24       261.51     9_344.75       0.1452             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              9_083.24       325.58     9_408.81       0.4621          1.0223         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              9_083.24       405.63     9_488.87       0.6308          1.0117         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             9_083.24       337.68     9_420.92       0.4534          1.0230         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             9_083.24       419.62     9_502.85       0.6169          1.0124         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             9_083.24       348.73     9_431.97       0.4489          1.0234         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             9_083.24       443.06     9_526.30       0.6094          1.0127         2.34
IVF-Binary-256-nl158-pca (self)                        9_083.24     1_050.68    10_133.92       0.4533          1.0231         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              6_944.32       246.92     7_191.24       0.1470             NaN         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              6_944.32       249.39     7_193.71       0.1460             NaN         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              6_944.32       256.15     7_200.47       0.1451             NaN         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             6_944.32       339.28     7_283.60       0.4617          1.0222         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             6_944.32       421.10     7_365.42       0.6294          1.0117         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             6_944.32       341.66     7_285.98       0.4562          1.0227         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             6_944.32       431.24     7_375.55       0.6209          1.0121         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             6_944.32       352.53     7_296.85       0.4508          1.0232         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             6_944.32       444.10     7_388.42       0.6125          1.0126         2.47
IVF-Binary-256-nl223-pca (self)                        6_944.32     1_075.16     8_019.48       0.4557          1.0229         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              7_266.77       259.64     7_526.42       0.1465             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              7_266.77       263.29     7_530.06       0.1461             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              7_266.77       267.07     7_533.85       0.1452             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             7_266.77       353.52     7_620.30       0.4614          1.0223         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             7_266.77       437.20     7_703.98       0.6294          1.0117         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             7_266.77       353.51     7_620.28       0.4586          1.0225         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             7_266.77       441.46     7_708.23       0.6252          1.0119         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             7_266.77       362.63     7_629.40       0.4523          1.0231         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             7_266.77       453.91     7_720.68       0.6150          1.0124         2.65
IVF-Binary-256-nl316-pca (self)                        7_266.77     1_164.68     8_431.45       0.4582          1.0226         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           14_163.83       423.51    14_587.34       0.0772             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          14_163.83       431.98    14_595.81       0.0686             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          14_163.83       444.51    14_608.34       0.0614             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          14_163.83       503.05    14_666.87       0.2152          1.0555         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          14_163.83       581.59    14_745.42       0.3254          1.0358         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         14_163.83       511.27    14_675.09       0.1976          1.0622         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         14_163.83       621.33    14_785.16       0.3011          1.0401         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         14_163.83       531.87    14_695.70       0.1846          1.0669         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         14_163.83       613.48    14_777.30       0.2853          1.0428         4.36
IVF-Binary-512-nl158-random (self)                    14_163.83     1_640.57    15_804.40       0.2006          1.0599         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          12_015.47       443.83    12_459.30       0.0732             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          12_015.47       436.71    12_452.17       0.0682             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          12_015.47       448.13    12_463.60       0.0626             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         12_015.47       515.54    12_531.01       0.2087          1.0573         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         12_015.47       592.90    12_608.37       0.3171          1.0368         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         12_015.47       518.57    12_534.04       0.1981          1.0608         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         12_015.47       600.53    12_616.00       0.3029          1.0390         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         12_015.47       530.10    12_545.57       0.1855          1.0660         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         12_015.47       623.20    12_638.67       0.2868          1.0422         4.49
IVF-Binary-512-nl223-random (self)                    12_015.47     1_660.84    13_676.31       0.2014          1.0584         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          12_292.86       448.57    12_741.43       0.0678             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          12_292.86       454.86    12_747.72       0.0662             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          12_292.86       458.19    12_751.05       0.0631             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         12_292.86       529.83    12_822.69       0.2012          1.0595         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         12_292.86       608.66    12_901.52       0.3103          1.0377         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         12_292.86       529.50    12_822.36       0.1971          1.0608         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         12_292.86       608.92    12_901.78       0.3043          1.0387         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         12_292.86       539.50    12_832.36       0.1878          1.0648         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         12_292.86       624.40    12_917.26       0.2905          1.0414         4.67
IVF-Binary-512-nl316-random (self)                    12_292.86     1_706.66    13_999.52       0.2005          1.0584         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              14_709.62       441.62    15_151.24       0.2021             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             14_709.62       445.46    15_155.09       0.1991             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             14_709.62       456.46    15_166.09       0.1965             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             14_709.62       567.23    15_276.85       0.6220          1.0119         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             14_709.62       601.83    15_311.46       0.7874          1.0051         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            14_709.62       536.17    15_245.79       0.6055          1.0127         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            14_709.62       620.57    15_330.19       0.7662          1.0058         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            14_709.62       556.83    15_266.46       0.5920          1.0134         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            14_709.62       648.62    15_358.24       0.7501          1.0063         4.36
IVF-Binary-512-nl158-pca (self)                       14_709.62     1_724.22    16_433.85       0.6057          1.0128         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             12_605.94       461.57    13_067.52       0.2020             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             12_605.94       464.49    13_070.43       0.2002             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             12_605.94       461.36    13_067.30       0.1979             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            12_605.94       544.06    13_150.01       0.6192          1.0120         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            12_605.94       622.09    13_228.04       0.7840          1.0052         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            12_605.94       541.73    13_147.68       0.6110          1.0124         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            12_605.94       627.73    13_233.67       0.7745          1.0055         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            12_605.94       556.17    13_162.11       0.5990          1.0131         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            12_605.94       647.16    13_253.10       0.7589          1.0060         4.49
IVF-Binary-512-nl223-pca (self)                       12_605.94     1_742.53    14_348.47       0.6116          1.0124         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             12_872.23       463.55    13_335.78       0.2017             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             12_872.23       488.63    13_360.86       0.2008             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             12_872.23       471.78    13_344.00       0.1984             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            12_872.23       553.26    13_425.49       0.6197          1.0120         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            12_872.23       635.47    13_507.69       0.7851          1.0052         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            12_872.23       555.44    13_427.66       0.6158          1.0122         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            12_872.23       639.95    13_512.18       0.7800          1.0053         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            12_872.23       564.50    13_436.72       0.6033          1.0128         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            12_872.23       655.45    13_527.67       0.7648          1.0058         4.67
IVF-Binary-512-nl316-pca (self)                       12_872.23     1_776.03    14_648.25       0.6161          1.0122         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          25_059.94       813.43    25_873.37       0.0880             NaN         8.41
IVF-Binary-1024-nl158-np12-rf0-random (query)         25_059.94       828.60    25_888.54       0.0852             NaN         8.41
IVF-Binary-1024-nl158-np17-rf0-random (query)         25_059.94       846.04    25_905.99       0.0831             NaN         8.41
IVF-Binary-1024-nl158-np7-rf10-random (query)         25_059.94       892.99    25_952.93       0.2267          1.0511         8.41
IVF-Binary-1024-nl158-np7-rf20-random (query)         25_059.94       972.51    26_032.45       0.3427          1.0328         8.41
IVF-Binary-1024-nl158-np12-rf10-random (query)        25_059.94       909.97    25_969.92       0.2152          1.0544         8.41
IVF-Binary-1024-nl158-np12-rf20-random (query)        25_059.94       989.30    26_049.24       0.3265          1.0352         8.41
IVF-Binary-1024-nl158-np17-rf10-random (query)        25_059.94       932.16    25_992.11       0.2080          1.0564         8.41
IVF-Binary-1024-nl158-np17-rf20-random (query)        25_059.94     1_022.20    26_082.14       0.3178          1.0364         8.41
IVF-Binary-1024-nl158-random (self)                   25_059.94     2_973.50    28_033.44       0.2175          1.0543         8.41
IVF-Binary-1024-nl223-np11-rf0-random (query)         22_831.41       829.58    23_660.99       0.0867             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         22_831.41       836.63    23_668.04       0.0851             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         22_831.41       848.97    23_680.38       0.0832             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        22_831.41       908.14    23_739.55       0.2225          1.0520         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        22_831.41       985.97    23_817.38       0.3379          1.0333         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        22_831.41       915.92    23_747.33       0.2162          1.0539         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        22_831.41       996.10    23_827.51       0.3270          1.0348         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        22_831.41       933.17    23_764.58       0.2090          1.0561         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        22_831.41     1_020.47    23_851.88       0.3172          1.0363         8.54
IVF-Binary-1024-nl223-random (self)                   22_831.41     2_996.33    25_827.74       0.2178          1.0538         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         23_192.26       840.91    24_033.18       0.0853             NaN         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)         23_192.26       845.10    24_037.36       0.0848             NaN         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)         23_192.26       855.98    24_048.24       0.0834             NaN         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)        23_192.26       945.09    24_137.35       0.2202          1.0526         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)        23_192.26     1_009.62    24_201.88       0.3355          1.0336         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)        23_192.26       924.65    24_116.91       0.2166          1.0536         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)        23_192.26     1_014.71    24_206.97       0.3301          1.0343         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)        23_192.26       948.68    24_140.95       0.2101          1.0556         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)        23_192.26     1_031.85    24_224.11       0.3202          1.0359         8.72
IVF-Binary-1024-nl316-random (self)                   23_192.26     3_031.78    26_224.04       0.2183          1.0535         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)             26_016.36       837.74    26_854.10       0.2050             NaN         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            26_016.36       854.91    26_871.27       0.2037             NaN         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            26_016.36       872.44    26_888.79       0.2034             NaN         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            26_016.36       925.85    26_942.20       0.6327          1.0113         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            26_016.36     1_000.52    27_016.88       0.7986          1.0048         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           26_016.36       942.95    26_959.31       0.6284          1.0116         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           26_016.36     1_026.07    27_042.43       0.7935          1.0049         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           26_016.36       976.94    26_993.30       0.6279          1.0116         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           26_016.36     1_066.23    27_082.59       0.7932          1.0049         8.42
IVF-Binary-1024-nl158-pca (self)                      26_016.36     3_066.99    29_083.35       0.6284          1.0116         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            23_897.65       853.12    24_750.77       0.2051             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            23_897.65       861.65    24_759.30       0.2041             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            23_897.65       897.34    24_794.99       0.2036             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           23_897.65       940.58    24_838.24       0.6334          1.0113         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           23_897.65     1_016.72    24_914.37       0.7994          1.0047         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           23_897.65       946.46    24_844.12       0.6300          1.0115         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           23_897.65     1_029.37    24_927.03       0.7956          1.0049         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           23_897.65     1_046.34    24_944.00       0.6278          1.0116         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           23_897.65     1_056.04    24_953.69       0.7931          1.0049         8.54
IVF-Binary-1024-nl223-pca (self)                      23_897.65     3_078.29    26_975.95       0.6304          1.0115         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            24_253.90       865.47    25_119.37       0.2051             NaN         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            24_253.90       871.25    25_125.15       0.2046             NaN         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            24_253.90       883.89    25_137.79       0.2037             NaN         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           24_253.90       954.19    25_208.09       0.6336          1.0113         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           24_253.90     1_035.40    25_289.30       0.7992          1.0047         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           24_253.90       956.53    25_210.43       0.6319          1.0114         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           24_253.90     1_043.25    25_297.15       0.7973          1.0048         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           24_253.90       972.67    25_226.57       0.6285          1.0116         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           24_253.90     1_128.07    25_381.97       0.7934          1.0049         8.73
IVF-Binary-1024-nl316-pca (self)                      24_253.90     3_111.42    27_365.31       0.6320          1.0114         8.73
IVF-Binary-512-nl158-np7-rf0-signed (query)           14_094.82       432.51    14_527.33       0.0772             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-signed (query)          14_094.82       436.24    14_531.06       0.0686             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-signed (query)          14_094.82       439.69    14_534.51       0.0614             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-signed (query)          14_094.82       504.31    14_599.13       0.2152          1.0555         4.36
IVF-Binary-512-nl158-np7-rf20-signed (query)          14_094.82       579.16    14_673.98       0.3254          1.0358         4.36
IVF-Binary-512-nl158-np12-rf10-signed (query)         14_094.82       509.24    14_604.06       0.1976          1.0622         4.36
IVF-Binary-512-nl158-np12-rf20-signed (query)         14_094.82       591.21    14_686.03       0.3011          1.0401         4.36
IVF-Binary-512-nl158-np17-rf10-signed (query)         14_094.82       526.36    14_621.18       0.1846          1.0669         4.36
IVF-Binary-512-nl158-np17-rf20-signed (query)         14_094.82       609.78    14_704.60       0.2853          1.0428         4.36
IVF-Binary-512-nl158-signed (self)                    14_094.82     1_637.46    15_732.28       0.2006          1.0599         4.36
IVF-Binary-512-nl223-np11-rf0-signed (query)          11_978.43       435.61    12_414.04       0.0732             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-signed (query)          11_978.43       438.13    12_416.57       0.0682             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-signed (query)          11_978.43       446.28    12_424.71       0.0626             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-signed (query)         11_978.43       514.24    12_492.68       0.2087          1.0573         4.49
IVF-Binary-512-nl223-np11-rf20-signed (query)         11_978.43       590.69    12_569.13       0.3171          1.0368         4.49
IVF-Binary-512-nl223-np14-rf10-signed (query)         11_978.43       519.99    12_498.43       0.1981          1.0608         4.49
IVF-Binary-512-nl223-np14-rf20-signed (query)         11_978.43       597.51    12_575.94       0.3029          1.0390         4.49
IVF-Binary-512-nl223-np21-rf10-signed (query)         11_978.43       531.68    12_510.12       0.1855          1.0660         4.49
IVF-Binary-512-nl223-np21-rf20-signed (query)         11_978.43       612.65    12_591.08       0.2868          1.0422         4.49
IVF-Binary-512-nl223-signed (self)                    11_978.43     1_657.40    13_635.84       0.2014          1.0584         4.49
IVF-Binary-512-nl316-np15-rf0-signed (query)          12_300.01       448.31    12_748.32       0.0678             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-signed (query)          12_300.01       453.33    12_753.34       0.0662             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-signed (query)          12_300.01       457.14    12_757.16       0.0631             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-signed (query)         12_300.01       541.05    12_841.06       0.2012          1.0595         4.67
IVF-Binary-512-nl316-np15-rf20-signed (query)         12_300.01       636.45    12_936.47       0.3103          1.0377         4.67
IVF-Binary-512-nl316-np17-rf10-signed (query)         12_300.01       541.86    12_841.88       0.1971          1.0608         4.67
IVF-Binary-512-nl316-np17-rf20-signed (query)         12_300.01       614.84    12_914.85       0.3043          1.0387         4.67
IVF-Binary-512-nl316-np25-rf10-signed (query)         12_300.01       541.71    12_841.72       0.1878          1.0648         4.67
IVF-Binary-512-nl316-np25-rf20-signed (query)         12_300.01       626.78    12_926.79       0.2905          1.0414         4.67
IVF-Binary-512-nl316-signed (self)                    12_300.01     1_703.46    14_003.48       0.2005          1.0584         4.67
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
Exhaustive (query)                                        31.50    15_803.78    15_835.27       1.0000          1.0000       146.48
Exhaustive (self)                                         31.50    55_623.63    55_655.13       1.0000          1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)              8_830.42       487.11     9_317.53       0.0337             NaN         2.28
ExhaustiveBinary-256-random-rf10 (query)               8_830.42       623.49     9_453.91       0.1500          1.0692         2.28
ExhaustiveBinary-256-random-rf20 (query)               8_830.42       769.39     9_599.81       0.2428          1.0431         2.28
ExhaustiveBinary-256-random (self)                     8_830.42     2_070.08    10_900.49       0.1545          1.0644         2.28
ExhaustiveBinary-256-pca_no_rr (query)                 9_332.94       493.09     9_826.04       0.1247             NaN         2.28
ExhaustiveBinary-256-pca-rf10 (query)                  9_332.94       646.66     9_979.60       0.3430          1.0271         2.28
ExhaustiveBinary-256-pca-rf20 (query)                  9_332.94       796.61    10_129.56       0.4582          1.0177         2.28
ExhaustiveBinary-256-pca (self)                        9_332.94     2_142.74    11_475.69       0.3433          1.0271         2.28
ExhaustiveBinary-512-random_no_rr (query)             17_225.07       855.86    18_080.94       0.0628             NaN         4.55
ExhaustiveBinary-512-random-rf10 (query)              17_225.07     1_002.69    18_227.77       0.1726          1.0535         4.55
ExhaustiveBinary-512-random-rf20 (query)              17_225.07     1_150.83    18_375.90       0.2658          1.0348         4.55
ExhaustiveBinary-512-random (self)                    17_225.07     3_332.44    20_557.51       0.1745          1.0515         4.55
ExhaustiveBinary-512-pca_no_rr (query)                17_985.66       872.63    18_858.29       0.1450             NaN         4.55
ExhaustiveBinary-512-pca-rf10 (query)                 17_985.66     1_027.48    19_013.14       0.3791          1.0668         4.55
ExhaustiveBinary-512-pca-rf20 (query)                 17_985.66     1_198.17    19_183.82       0.4884          1.0163         4.55
ExhaustiveBinary-512-pca (self)                       17_985.66     3_391.97    21_377.62       0.3800          1.0959         4.55
ExhaustiveBinary-1024-random_no_rr (query)            34_317.79     1_605.43    35_923.22       0.0818             NaN         9.10
ExhaustiveBinary-1024-random-rf10 (query)             34_317.79     1_754.28    36_072.07       0.1931          1.0465         9.10
ExhaustiveBinary-1024-random-rf20 (query)             34_317.79     1_929.44    36_247.23       0.2959          1.0303         9.10
ExhaustiveBinary-1024-random (self)                   34_317.79     5_838.36    40_156.14       0.1941          1.0466         9.10
ExhaustiveBinary-1024-pca_no_rr (query)               35_379.80     1_624.00    37_003.80       0.2012             NaN         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                35_379.80     1_792.07    37_171.87       0.6300          1.0091         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                35_379.80     1_972.90    37_352.71       0.7963          1.0039         9.11
ExhaustiveBinary-1024-pca (self)                      35_379.80     5_948.62    41_328.42       0.6289          1.0092         9.11
ExhaustiveBinary-768-signed_no_rr (query)             25_833.46     1_243.37    27_076.83       0.0764             NaN         6.83
ExhaustiveBinary-768-signed-rf10 (query)              25_833.46     1_396.16    27_229.61       0.1848          1.0490         6.83
ExhaustiveBinary-768-signed-rf20 (query)              25_833.46     1_551.31    27_384.77       0.2828          1.0320         6.83
ExhaustiveBinary-768-signed (self)                    25_833.46     4_618.90    30_452.36       0.1853          1.0486         6.83
IVF-Binary-256-nl158-np7-rf0-random (query)           13_364.14       350.55    13_714.69       0.0556             NaN         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)          13_364.14       352.42    13_716.55       0.0441             NaN         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)          13_364.14       359.51    13_723.65       0.0368             NaN         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)          13_364.14       449.93    13_814.07       0.1932          1.0533         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)          13_364.14       547.34    13_911.47       0.2968          1.0337         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)         13_364.14       455.54    13_819.68       0.1705          1.0619         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)         13_364.14       554.42    13_918.56       0.2657          1.0390         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)         13_364.14       460.90    13_825.04       0.1566          1.0677         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)         13_364.14       565.18    13_929.32       0.2518          1.0419         2.74
IVF-Binary-256-nl158-random (self)                    13_364.14     1_425.37    14_789.51       0.1751          1.0575         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)          10_006.18       364.94    10_371.11       0.0536             NaN         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)          10_006.18       366.32    10_372.50       0.0445             NaN         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)          10_006.18       374.04    10_380.22       0.0375             NaN         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)         10_006.18       472.62    10_478.80       0.1878          1.0561         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)         10_006.18       566.38    10_572.56       0.2911          1.0348         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)         10_006.18       476.95    10_483.12       0.1715          1.0617         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)         10_006.18       574.98    10_581.16       0.2707          1.0382         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)         10_006.18       477.41    10_483.59       0.1577          1.0671         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)         10_006.18       585.52    10_591.70       0.2539          1.0410         2.93
IVF-Binary-256-nl223-random (self)                    10_006.18     1_497.62    11_503.80       0.1760          1.0574         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)          10_618.34       387.44    11_005.79       0.0461             NaN         3.20
IVF-Binary-256-nl316-np17-rf0-random (query)          10_618.34       404.28    11_022.62       0.0420             NaN         3.20
IVF-Binary-256-nl316-np25-rf0-random (query)          10_618.34       399.11    11_017.46       0.0386             NaN         3.20
IVF-Binary-256-nl316-np15-rf10-random (query)         10_618.34       493.25    11_111.59       0.1750          1.0599         3.20
IVF-Binary-256-nl316-np15-rf20-random (query)         10_618.34       598.24    11_216.58       0.2772          1.0369         3.20
IVF-Binary-256-nl316-np17-rf10-random (query)         10_618.34       489.97    11_108.32       0.1675          1.0628         3.20
IVF-Binary-256-nl316-np17-rf20-random (query)         10_618.34       598.96    11_217.30       0.2674          1.0386         3.20
IVF-Binary-256-nl316-np25-rf10-random (query)         10_618.34       499.58    11_117.92       0.1595          1.0663         3.20
IVF-Binary-256-nl316-np25-rf20-random (query)         10_618.34       605.33    11_223.67       0.2556          1.0411         3.20
IVF-Binary-256-nl316-random (self)                    10_618.34     1_552.49    12_170.83       0.1713          1.0586         3.20
IVF-Binary-256-nl158-np7-rf0-pca (query)              14_019.06       354.97    14_374.03       0.1301             NaN         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)             14_019.06       361.16    14_380.23       0.1289             NaN         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)             14_019.06       366.34    14_385.40       0.1285             NaN         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)             14_019.06       475.04    14_494.11       0.4031          1.0217         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)             14_019.06       571.33    14_590.39       0.5652          1.0121         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)            14_019.06       475.16    14_494.22       0.3946          1.0224         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)            14_019.06       583.90    14_602.96       0.5509          1.0127         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)            14_019.06       524.58    14_543.64       0.3918          1.0226         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)            14_019.06       597.58    14_616.64       0.5457          1.0130         2.74
IVF-Binary-256-nl158-pca (self)                       14_019.06     1_513.41    15_532.47       0.3948          1.0225         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)             10_640.04       372.96    11_013.00       0.1298             NaN         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)             10_640.04       376.03    11_016.06       0.1290             NaN         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)             10_640.04       383.31    11_023.34       0.1284             NaN         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)            10_640.04       486.65    11_126.68       0.4010          1.0218         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)            10_640.04       593.07    11_233.11       0.5605          1.0122         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)            10_640.04       490.47    11_130.51       0.3955          1.0223         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)            10_640.04       603.94    11_243.98       0.5523          1.0126         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)            10_640.04       503.78    11_143.82       0.3911          1.0227         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)            10_640.04       617.88    11_257.91       0.5450          1.0130         2.93
IVF-Binary-256-nl223-pca (self)                       10_640.04     1_549.02    12_189.06       0.3954          1.0224         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)             11_293.76       395.23    11_688.99       0.1294             NaN         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)             11_293.76       398.98    11_692.74       0.1290             NaN         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)             11_293.76       406.05    11_699.81       0.1284             NaN         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)            11_293.76       555.27    11_849.03       0.4000          1.0219         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)            11_293.76       617.37    11_911.13       0.5599          1.0123         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)            11_293.76       511.54    11_805.29       0.3974          1.0222         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)            11_293.76       623.91    11_917.66       0.5557          1.0125         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)            11_293.76       519.93    11_813.69       0.3927          1.0226         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)            11_293.76       634.53    11_928.29       0.5475          1.0129         3.21
IVF-Binary-256-nl316-pca (self)                       11_293.76     1_627.69    12_921.45       0.3974          1.0222         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)           21_912.00       646.36    22_558.37       0.0741             NaN         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)          21_912.00       653.24    22_565.25       0.0687             NaN         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)          21_912.00       662.92    22_574.92       0.0646             NaN         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)          21_912.00       751.57    22_663.58       0.1979          1.0471         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)          21_912.00       844.74    22_756.74       0.3014          1.0307         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)         21_912.00       757.75    22_669.76       0.1826          1.0511         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)         21_912.00       857.04    22_769.04       0.2793          1.0333         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)         21_912.00       767.44    22_679.44       0.1755          1.0530         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)         21_912.00       872.29    22_784.29       0.2711          1.0344         5.02
IVF-Binary-512-nl158-random (self)                    21_912.00     2_447.01    24_359.02       0.1849          1.0493         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)          18_539.99       665.62    19_205.61       0.0731             NaN         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)          18_539.99       669.90    19_209.89       0.0683             NaN         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)          18_539.99       680.63    19_220.62       0.0649             NaN         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)         18_539.99       767.91    19_307.90       0.1960          1.0473         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)         18_539.99       870.74    19_410.73       0.2978          1.0306         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)         18_539.99       769.77    19_309.76       0.1858          1.0500         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)         18_539.99       926.73    19_466.72       0.2836          1.0324         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)         18_539.99       784.10    19_324.09       0.1772          1.0524         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)         18_539.99       892.50    19_432.49       0.2725          1.0340         5.21
IVF-Binary-512-nl223-random (self)                    18_539.99     2_493.95    21_033.94       0.1878          1.0482         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)          19_172.84       687.84    19_860.68       0.0696             NaN         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)          19_172.84       693.89    19_866.73       0.0676             NaN         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)          19_172.84       697.05    19_869.89       0.0660             NaN         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)         19_172.84       797.11    19_969.95       0.1897          1.0484         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)         19_172.84       894.11    20_066.96       0.2906          1.0313         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)         19_172.84       793.70    19_966.54       0.1846          1.0498         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)         19_172.84       896.60    20_069.44       0.2838          1.0322         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)         19_172.84       804.15    19_976.99       0.1785          1.0519         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)         19_172.84       908.56    20_081.40       0.2740          1.0337         5.48
IVF-Binary-512-nl316-random (self)                    19_172.84     2_570.94    21_743.78       0.1870          1.0482         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)              22_706.32       663.90    23_370.22       0.1701             NaN         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)             22_706.32       687.18    23_393.49       0.1677             NaN         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)             22_706.32       677.85    23_384.17       0.1666             NaN         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)             22_706.32       771.83    23_478.14       0.5298          1.0137         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)             22_706.32       874.88    23_581.20       0.7010          1.0067         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)            22_706.32       781.38    23_487.70       0.5150          1.0145         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)            22_706.32       891.50    23_597.82       0.6802          1.0074         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)            22_706.32       797.11    23_503.42       0.5059          1.0150         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)            22_706.32       908.24    23_614.56       0.6665          1.0078         5.02
IVF-Binary-512-nl158-pca (self)                       22_706.32     2_535.58    25_241.90       0.5164          1.0146         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)             19_333.44       680.55    20_013.99       0.1692             NaN         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)             19_333.44       683.81    20_017.25       0.1676             NaN         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)             19_333.44       696.94    20_030.38       0.1660             NaN         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)            19_333.44       792.18    20_125.62       0.5239          1.0140         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)            19_333.44       896.18    20_229.62       0.6925          1.0069         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)            19_333.44       795.57    20_129.01       0.5159          1.0144         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)            19_333.44       903.76    20_237.19       0.6817          1.0073         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)            19_333.44       811.92    20_145.36       0.5053          1.0150         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)            19_333.44       926.53    20_259.97       0.6656          1.0078         5.21
IVF-Binary-512-nl223-pca (self)                       19_333.44     2_576.10    21_909.54       0.5170          1.0145         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)             19_976.01       704.96    20_680.97       0.1691             NaN         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)             19_976.01       705.15    20_681.16       0.1683             NaN         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)             19_976.01       713.19    20_689.20       0.1668             NaN         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)            19_976.01       817.67    20_793.69       0.5242          1.0140         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)            19_976.01       920.90    20_896.91       0.6932          1.0069         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)            19_976.01       818.65    20_794.66       0.5203          1.0142         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)            19_976.01       925.72    20_901.73       0.6881          1.0071         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)            19_976.01       828.61    20_804.62       0.5102          1.0148         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)            19_976.01       940.56    20_916.57       0.6733          1.0076         5.48
IVF-Binary-512-nl316-pca (self)                       19_976.01     2_660.86    22_636.87       0.5214          1.0142         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)          39_056.93     1_255.44    40_312.37       0.0854             NaN         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)         39_056.93     1_270.67    40_327.60       0.0832             NaN         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)         39_056.93     1_282.59    40_339.51       0.0820             NaN         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)         39_056.93     1_354.42    40_411.34       0.2106          1.0431         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)         39_056.93     1_449.96    40_506.89       0.3205          1.0279         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)        39_056.93     1_381.78    40_438.71       0.1993          1.0454         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)        39_056.93     1_480.16    40_537.09       0.3029          1.0297         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)        39_056.93     1_380.41    40_437.33       0.1956          1.0461         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)        39_056.93     1_487.96    40_544.89       0.2986          1.0301         9.57
IVF-Binary-1024-nl158-random (self)                   39_056.93     4_474.53    43_531.46       0.2004          1.0454         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)         35_671.44     1_276.62    36_948.05       0.0848             NaN         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)         35_671.44     1_282.86    36_954.29       0.0834             NaN         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)         35_671.44     1_298.05    36_969.48       0.0820             NaN         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)        35_671.44     1_376.17    37_047.61       0.2091          1.0431         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)        35_671.44     1_477.08    37_148.52       0.3186          1.0278         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)        35_671.44     1_382.62    37_054.05       0.2021          1.0446         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)        35_671.44     1_486.56    37_158.00       0.3071          1.0291         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)        35_671.44     1_403.96    37_075.39       0.1966          1.0459         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)        35_671.44     1_513.33    37_184.77       0.3000          1.0299         9.76
IVF-Binary-1024-nl223-random (self)                   35_671.44     4_608.59    40_280.02       0.2028          1.0447         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)         36_295.79     1_294.69    37_590.48       0.0838             NaN        10.03
IVF-Binary-1024-nl316-np17-rf0-random (query)         36_295.79     1_303.65    37_599.44       0.0830             NaN        10.03
IVF-Binary-1024-nl316-np25-rf0-random (query)         36_295.79     1_307.83    37_603.63       0.0823             NaN        10.03
IVF-Binary-1024-nl316-np15-rf10-random (query)        36_295.79     1_398.06    37_693.85       0.2051          1.0437        10.03
IVF-Binary-1024-nl316-np15-rf20-random (query)        36_295.79     1_496.84    37_792.63       0.3145          1.0282        10.03
IVF-Binary-1024-nl316-np17-rf10-random (query)        36_295.79     1_397.78    37_693.57       0.2016          1.0445        10.03
IVF-Binary-1024-nl316-np17-rf20-random (query)        36_295.79     1_502.44    37_798.23       0.3083          1.0289        10.03
IVF-Binary-1024-nl316-np25-rf10-random (query)        36_295.79     1_413.07    37_708.86       0.1970          1.0458        10.03
IVF-Binary-1024-nl316-np25-rf20-random (query)        36_295.79     1_520.94    37_816.73       0.3009          1.0299        10.03
IVF-Binary-1024-nl316-random (self)                   36_295.79     4_598.28    40_894.07       0.2020          1.0447        10.03
IVF-Binary-1024-nl158-np7-rf0-pca (query)             40_074.63     1_279.18    41_353.82       0.2021             NaN         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)            40_074.63     1_288.26    41_362.89       0.2012             NaN         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)            40_074.63     1_310.31    41_384.94       0.2012             NaN         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)            40_074.63     1_383.69    41_458.32       0.6336          1.0090         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)            40_074.63     1_482.38    41_557.01       0.8010          1.0037         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)           40_074.63     1_421.37    41_496.00       0.6299          1.0091         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)           40_074.63     1_513.53    41_588.16       0.7966          1.0039         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)           40_074.63     1_414.97    41_489.61       0.6298          1.0091         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)           40_074.63     1_522.49    41_597.12       0.7965          1.0039         9.57
IVF-Binary-1024-nl158-pca (self)                      40_074.63     4_572.38    44_647.02       0.6289          1.0092         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)            36_738.67     1_321.36    38_060.03       0.2023             NaN         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)            36_738.67     1_315.01    38_053.68       0.2015             NaN         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)            36_738.67     1_335.33    38_074.00       0.2012             NaN         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)           36_738.67     1_417.93    38_156.60       0.6342          1.0089         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)           36_738.67     1_505.59    38_244.26       0.8013          1.0037         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)           36_738.67     1_414.07    38_152.74       0.6311          1.0091         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)           36_738.67     1_517.85    38_256.51       0.7977          1.0038         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)           36_738.67     1_438.79    38_177.46       0.6302          1.0091         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)           36_738.67     1_547.40    38_286.07       0.7967          1.0039         9.76
IVF-Binary-1024-nl223-pca (self)                      36_738.67     5_108.11    41_846.77       0.6299          1.0092         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)            37_532.94     1_333.10    38_866.04       0.2021             NaN        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)            37_532.94     1_337.48    38_870.41       0.2016             NaN        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)            37_532.94     1_351.40    38_884.34       0.2012             NaN        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)           37_532.94     1_445.48    38_978.42       0.6333          1.0090        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)           37_532.94     1_545.70    39_078.64       0.8004          1.0037        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)           37_532.94     1_444.38    38_977.32       0.6316          1.0091        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)           37_532.94     1_549.49    39_082.43       0.7987          1.0038        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)           37_532.94     1_457.30    38_990.23       0.6298          1.0091        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)           37_532.94     1_571.97    39_104.91       0.7965          1.0039        10.04
IVF-Binary-1024-nl316-pca (self)                      37_532.94     4_715.80    42_248.74       0.6306          1.0091        10.04
IVF-Binary-768-nl158-np7-rf0-signed (query)           30_538.21       952.38    31_490.59       0.0817             NaN         7.29
IVF-Binary-768-nl158-np12-rf0-signed (query)          30_538.21       961.16    31_499.38       0.0786             NaN         7.29
IVF-Binary-768-nl158-np17-rf0-signed (query)          30_538.21       973.82    31_512.03       0.0768             NaN         7.29
IVF-Binary-768-nl158-np7-rf10-signed (query)          30_538.21     1_058.09    31_596.30       0.2051          1.0446         7.29
IVF-Binary-768-nl158-np7-rf20-signed (query)          30_538.21     1_151.09    31_689.30       0.3112          1.0290         7.29
IVF-Binary-768-nl158-np12-rf10-signed (query)         30_538.21     1_071.63    31_609.84       0.1925          1.0474         7.29
IVF-Binary-768-nl158-np12-rf20-signed (query)         30_538.21     1_166.28    31_704.50       0.2920          1.0311         7.29
IVF-Binary-768-nl158-np17-rf10-signed (query)         30_538.21     1_076.63    31_614.84       0.1875          1.0486         7.29
IVF-Binary-768-nl158-np17-rf20-signed (query)         30_538.21     1_181.11    31_719.32       0.2861          1.0318         7.29
IVF-Binary-768-nl158-signed (self)                    30_538.21     3_469.78    34_008.00       0.1931          1.0471         7.29
IVF-Binary-768-nl223-np11-rf0-signed (query)          27_161.88       974.90    28_136.78       0.0815             NaN         7.48
IVF-Binary-768-nl223-np14-rf0-signed (query)          27_161.88       977.04    28_138.92       0.0795             NaN         7.48
IVF-Binary-768-nl223-np21-rf0-signed (query)          27_161.88     1_004.13    28_166.00       0.0776             NaN         7.48
IVF-Binary-768-nl223-np11-rf10-signed (query)         27_161.88     1_071.87    28_233.74       0.2028          1.0447         7.48
IVF-Binary-768-nl223-np11-rf20-signed (query)         27_161.88     1_175.72    28_337.60       0.3086          1.0290         7.48
IVF-Binary-768-nl223-np14-rf10-signed (query)         27_161.88     1_094.46    28_256.33       0.1947          1.0467         7.48
IVF-Binary-768-nl223-np14-rf20-signed (query)         27_161.88     1_188.39    28_350.27       0.2966          1.0305         7.48
IVF-Binary-768-nl223-np21-rf10-signed (query)         27_161.88     1_095.74    28_257.62       0.1879          1.0484         7.48
IVF-Binary-768-nl223-np21-rf20-signed (query)         27_161.88     1_206.10    28_367.98       0.2878          1.0316         7.48
IVF-Binary-768-nl223-signed (self)                    27_161.88     3_521.64    30_683.52       0.1953          1.0464         7.48
IVF-Binary-768-nl316-np15-rf0-signed (query)          27_776.07       995.79    28_771.86       0.0801             NaN         7.76
IVF-Binary-768-nl316-np17-rf0-signed (query)          27_776.07     1_001.82    28_777.89       0.0790             NaN         7.76
IVF-Binary-768-nl316-np25-rf0-signed (query)          27_776.07     1_007.71    28_783.79       0.0778             NaN         7.76
IVF-Binary-768-nl316-np15-rf10-signed (query)         27_776.07     1_100.90    28_876.97       0.1985          1.0455         7.76
IVF-Binary-768-nl316-np15-rf20-signed (query)         27_776.07     1_204.78    28_980.85       0.3033          1.0295         7.76
IVF-Binary-768-nl316-np17-rf10-signed (query)         27_776.07     1_101.41    28_877.48       0.1943          1.0465         7.76
IVF-Binary-768-nl316-np17-rf20-signed (query)         27_776.07     1_214.61    28_990.68       0.2969          1.0303         7.76
IVF-Binary-768-nl316-np25-rf10-signed (query)         27_776.07     1_114.11    28_890.18       0.1889          1.0481         7.76
IVF-Binary-768-nl316-np25-rf20-signed (query)         27_776.07     1_225.24    29_001.31       0.2887          1.0315         7.76
IVF-Binary-768-nl316-signed (self)                    27_776.07     3_590.10    31_366.17       0.1949          1.0463         7.76
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
Exhaustive (query)                                         9.95     4_444.84     4_454.80       1.0000          1.0000        48.83
Exhaustive (self)                                          9.95    14_653.80    14_663.75       1.0000          1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_603.80       303.91     2_907.71       0.0875             NaN         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_603.80       389.49     2_993.29       0.3366          1.1470         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_603.80       494.65     3_098.45       0.4794          1.0850         1.78
ExhaustiveBinary-256-random (self)                     2_603.80     1_281.13     3_884.93       0.3636          1.1515         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_795.35       289.58     3_084.93       0.1109             NaN         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_795.35       409.47     3_204.81       0.3158          1.5913         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_795.35       518.30     3_313.65       0.4283          1.3071         1.78
ExhaustiveBinary-256-pca (self)                        2_795.35     1_352.32     4_147.67       0.2950          2.2349         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_138.85       434.68     5_573.53       0.1342             NaN         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_138.85       570.04     5_708.89       0.4316          1.0992         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_138.85       638.88     5_777.73       0.5828          1.0538         3.55
ExhaustiveBinary-512-random (self)                     5_138.85     1_786.87     6_925.72       0.4551          1.1054         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_419.51       446.14     5_865.65       0.1283             NaN         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_419.51       553.92     5_973.43       0.4093          1.1233         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_419.51       663.60     6_083.11       0.5766          1.0656         3.55
ExhaustiveBinary-512-pca (self)                        5_419.51     1_831.77     7_251.28       0.4109          1.1443         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_136.89       764.65    10_901.54       0.1930             NaN         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_136.89       873.27    11_010.16       0.5442          1.0617         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_136.89       976.21    11_113.10       0.7015          1.0305         7.10
ExhaustiveBinary-1024-random (self)                   10_136.89     2_917.93    13_054.82       0.5708          1.0663         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_640.48       769.59    11_410.07       0.1535             NaN         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_640.48       882.36    11_522.84       0.4703          1.0875         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_640.48       996.46    11_636.94       0.6408          1.0449         7.10
ExhaustiveBinary-1024-pca (self)                      10_640.48     2_940.09    13_580.57       0.4638          1.1066         7.10
ExhaustiveBinary-256-signed_no_rr (query)              2_592.62       274.47     2_867.09       0.0875             NaN         1.78
ExhaustiveBinary-256-signed-rf10 (query)               2_592.62       376.71     2_969.33       0.3366          1.1470         1.78
ExhaustiveBinary-256-signed-rf20 (query)               2_592.62       472.19     3_064.81       0.4794          1.0850         1.78
ExhaustiveBinary-256-signed (self)                     2_592.62     1_232.64     3_825.26       0.3636          1.1515         1.78
IVF-Binary-256-nl158-np7-rf0-random (query)            3_941.50       112.77     4_054.26       0.0905             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           3_941.50       113.71     4_055.21       0.0889             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           3_941.50       116.78     4_058.27       0.0879             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           3_941.50       168.35     4_109.84       0.3414          1.1450         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           3_941.50       216.19     4_157.69       0.4835          1.0837         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          3_941.50       169.70     4_111.19       0.3391          1.1459         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          3_941.50       222.31     4_163.80       0.4816          1.0842         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          3_941.50       176.78     4_118.27       0.3378          1.1466         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          3_941.50       230.90     4_172.40       0.4805          1.0846         1.93
IVF-Binary-256-nl158-random (self)                     3_941.50       511.90     4_453.40       0.3662          1.1503         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           3_217.30       116.70     3_334.00       0.0949             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           3_217.30       120.46     3_337.76       0.0906             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           3_217.30       124.12     3_341.42       0.0885             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          3_217.30       173.84     3_391.14       0.3513          1.1397         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          3_217.30       223.69     3_441.00       0.4946          1.0802         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          3_217.30       175.18     3_392.49       0.3420          1.1459         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          3_217.30       228.41     3_445.71       0.4848          1.0840         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          3_217.30       184.07     3_401.37       0.3385          1.1480         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          3_217.30       239.12     3_456.42       0.4816          1.0850         2.00
IVF-Binary-256-nl223-random (self)                     3_217.30       529.12     3_746.43       0.3692          1.1507         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           3_443.18       123.26     3_566.44       0.0923             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           3_443.18       124.15     3_567.33       0.0905             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           3_443.18       128.53     3_571.71       0.0892             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          3_443.18       181.09     3_624.27       0.3480          1.1422         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          3_443.18       234.22     3_677.39       0.4900          1.0824         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          3_443.18       179.20     3_622.38       0.3442          1.1442         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          3_443.18       243.78     3_686.96       0.4865          1.0835         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          3_443.18       185.83     3_629.01       0.3411          1.1460         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          3_443.18       239.59     3_682.77       0.4830          1.0845         2.09
IVF-Binary-256-nl316-random (self)                     3_443.18       544.07     3_987.25       0.3712          1.1489         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               4_115.09       116.62     4_231.71       0.1195             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              4_115.09       118.08     4_233.16       0.1157             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              4_115.09       121.16     4_236.25       0.1148             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              4_115.09       184.55     4_299.64       0.3989          1.1281         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              4_115.09       236.83     4_351.91       0.5690          1.0681         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             4_115.09       185.91     4_301.00       0.3772          1.1547         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             4_115.09       257.77     4_372.86       0.5415          1.0820         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             4_115.09       192.24     4_307.33       0.3703          1.1750         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             4_115.09       252.19     4_367.28       0.5309          1.0927         1.93
IVF-Binary-256-nl158-pca (self)                        4_115.09       577.85     4_692.93       0.3839          1.1736         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              3_405.76       122.37     3_528.13       0.1167             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              3_405.76       123.56     3_529.32       0.1158             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              3_405.76       128.07     3_533.83       0.1147             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             3_405.76       191.12     3_596.88       0.3791          1.1456         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             3_405.76       244.68     3_650.44       0.5456          1.0767         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             3_405.76       192.21     3_597.97       0.3742          1.1576         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             3_405.76       251.93     3_657.69       0.5380          1.0833         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             3_405.76       201.33     3_607.09       0.3676          1.1786         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             3_405.76       263.25     3_669.01       0.5266          1.0949         2.00
IVF-Binary-256-nl223-pca (self)                        3_405.76       598.06     4_003.82       0.3819          1.1751         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              3_624.72       128.39     3_753.11       0.1164             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              3_624.72       129.27     3_753.99       0.1159             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              3_624.72       133.38     3_758.10       0.1148             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             3_624.72       196.23     3_820.95       0.3784          1.1471         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             3_624.72       251.93     3_876.65       0.5447          1.0773         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             3_624.72       196.25     3_820.97       0.3760          1.1519         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             3_624.72       254.11     3_878.84       0.5408          1.0803         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             3_624.72       202.88     3_827.60       0.3694          1.1712         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             3_624.72       264.18     3_888.90       0.5297          1.0908         2.09
IVF-Binary-256-nl316-pca (self)                        3_624.72       610.87     4_235.59       0.3836          1.1697         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_404.03       200.70     6_604.73       0.1359             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_404.03       205.15     6_609.18       0.1353             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_404.03       208.60     6_612.62       0.1348             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_404.03       261.32     6_665.35       0.4333          1.0987         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_404.03       314.75     6_718.77       0.5845          1.0534         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_404.03       264.70     6_668.72       0.4325          1.0989         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_404.03       317.47     6_721.50       0.5836          1.0536         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_404.03       271.58     6_675.61       0.4321          1.0990         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_404.03       327.61     6_731.64       0.5833          1.0536         3.71
IVF-Binary-512-nl158-random (self)                     6_404.03       827.40     7_231.43       0.4561          1.1050         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_723.80       207.07     5_930.88       0.1394             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_723.80       210.93     5_934.73       0.1362             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_723.80       215.52     5_939.32       0.1350             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_723.80       267.62     5_991.42       0.4391          1.0962         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_723.80       317.02     6_040.83       0.5887          1.0525         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_723.80       269.05     5_992.85       0.4342          1.0985         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_723.80       330.16     6_053.96       0.5848          1.0535         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_723.80       279.62     6_003.43       0.4321          1.0993         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_723.80       340.52     6_064.32       0.5829          1.0539         3.77
IVF-Binary-512-nl223-random (self)                     5_723.80       860.18     6_583.98       0.4579          1.1048         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           5_935.29       213.67     6_148.96       0.1373             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           5_935.29       216.56     6_151.85       0.1363             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           5_935.29       222.92     6_158.21       0.1354             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          5_935.29       277.45     6_212.75       0.4377          1.0972         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          5_935.29       323.54     6_258.83       0.5881          1.0527         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          5_935.29       273.30     6_208.59       0.4355          1.0980         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          5_935.29       345.23     6_280.52       0.5863          1.0531         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          5_935.29       281.60     6_216.89       0.4336          1.0987         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          5_935.29       335.86     6_271.15       0.5845          1.0535         3.86
IVF-Binary-512-nl316-random (self)                     5_935.29       861.47     6_796.76       0.4598          1.1039         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               6_733.73       210.14     6_943.87       0.1322             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              6_733.73       213.57     6_947.31       0.1292             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              6_733.73       217.77     6_951.51       0.1286             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              6_733.73       277.29     7_011.02       0.4307          1.1079         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              6_733.73       330.11     7_063.85       0.6023          1.0562         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             6_733.73       280.51     7_014.24       0.4144          1.1177         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             6_733.73       336.25     7_069.99       0.5839          1.0618         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             6_733.73       286.91     7_020.64       0.4108          1.1209         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             6_733.73       346.84     7_080.57       0.5792          1.0638         3.71
IVF-Binary-512-nl158-pca (self)                        6_733.73       888.68     7_622.41       0.4163          1.1372         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              6_055.24       215.73     6_270.97       0.1299             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              6_055.24       218.92     6_274.16       0.1293             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              6_055.24       228.20     6_283.44       0.1288             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             6_055.24       299.02     6_354.26       0.4156          1.1155         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             6_055.24       339.37     6_394.61       0.5849          1.0605         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             6_055.24       286.12     6_341.37       0.4129          1.1181         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             6_055.24       343.69     6_398.93       0.5816          1.0621         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             6_055.24       297.02     6_352.26       0.4106          1.1207         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             6_055.24       368.50     6_423.74       0.5785          1.0637         3.77
IVF-Binary-512-nl223-pca (self)                        6_055.24       911.74     6_966.98       0.4149          1.1373         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              6_224.93       223.56     6_448.49       0.1297             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              6_224.93       223.33     6_448.25       0.1294             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              6_224.93       231.68     6_456.61       0.1288             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             6_224.93       292.67     6_517.60       0.4147          1.1162         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             6_224.93       344.93     6_569.86       0.5843          1.0608         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             6_224.93       291.57     6_516.49       0.4135          1.1174         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             6_224.93       346.85     6_571.78       0.5825          1.0616         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             6_224.93       298.53     6_523.46       0.4107          1.1202         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             6_224.93       355.62     6_580.55       0.5789          1.0633         3.86
IVF-Binary-512-nl316-pca (self)                        6_224.93       922.31     7_147.24       0.4156          1.1365         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_417.52       383.21    11_800.73       0.1940             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_417.52       387.11    11_804.63       0.1935             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_417.52       395.84    11_813.35       0.1933             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_417.52       445.27    11_862.79       0.5448          1.0615         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_417.52       496.14    11_913.66       0.7021          1.0305         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_417.52       453.99    11_871.51       0.5445          1.0616         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_417.52       539.97    11_957.49       0.7018          1.0305         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_417.52       461.46    11_878.97       0.5444          1.0617         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_417.52       515.95    11_933.47       0.7017          1.0305         7.26
IVF-Binary-1024-nl158-random (self)                   11_417.52     1_479.26    12_896.77       0.5712          1.0662         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_584.30       393.08    10_977.38       0.1959             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_584.30       393.95    10_978.24       0.1943             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_584.30       405.88    10_990.18       0.1937             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_584.30       451.66    11_035.96       0.5475          1.0609         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_584.30       503.19    11_087.49       0.7042          1.0301         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_584.30       456.87    11_041.17       0.5456          1.0614         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_584.30       509.62    11_093.92       0.7024          1.0304         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_584.30       469.43    11_053.72       0.5447          1.0616         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_584.30       526.97    11_111.26       0.7016          1.0305         7.32
IVF-Binary-1024-nl223-random (self)                   10_584.30     1_464.31    12_048.60       0.5725          1.0659         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         10_795.19       397.46    11_192.65       0.1948             NaN         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)         10_795.19       398.56    11_193.75       0.1943             NaN         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)         10_795.19       409.58    11_204.77       0.1936             NaN         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)        10_795.19       458.38    11_253.57       0.5477          1.0609         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)        10_795.19       509.54    11_304.73       0.7042          1.0302         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)        10_795.19       459.68    11_254.87       0.5466          1.0611         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)        10_795.19       512.24    11_307.43       0.7032          1.0303         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)        10_795.19       469.22    11_264.41       0.5454          1.0615         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)        10_795.19       528.69    11_323.88       0.7022          1.0305         7.41
IVF-Binary-1024-nl316-random (self)                   10_795.19     1_484.13    12_279.32       0.5729          1.0658         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)             11_808.27       400.59    12_208.86       0.1551             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            11_808.27       402.85    12_211.12       0.1537             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            11_808.27       410.57    12_218.84       0.1536             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            11_808.27       464.69    12_272.96       0.4828          1.0838         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            11_808.27       523.20    12_331.47       0.6531          1.0425         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           11_808.27       469.90    12_278.17       0.4723          1.0867         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           11_808.27       525.92    12_334.19       0.6435          1.0443         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           11_808.27       480.10    12_288.37       0.4709          1.0871         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           11_808.27       537.20    12_345.47       0.6414          1.0447         7.26
IVF-Binary-1024-nl158-pca (self)                      11_808.27     1_523.56    13_331.83       0.4658          1.1059         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            11_101.55       407.40    11_508.94       0.1544             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            11_101.55       418.45    11_520.00       0.1541             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            11_101.55       422.56    11_524.11       0.1540             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           11_101.55       478.72    11_580.27       0.4734          1.0860         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           11_101.55       526.76    11_628.31       0.6441          1.0440         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           11_101.55       476.49    11_578.04       0.4721          1.0865         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           11_101.55       534.17    11_635.72       0.6425          1.0443         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           11_101.55       490.83    11_592.38       0.4712          1.0868         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           11_101.55       562.37    11_663.92       0.6414          1.0446         7.32
IVF-Binary-1024-nl223-pca (self)                      11_101.55     1_545.58    12_647.13       0.4653          1.1056         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            11_425.41       413.81    11_839.22       0.1543             NaN         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            11_425.41       414.32    11_839.73       0.1542             NaN         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            11_425.41       428.89    11_854.31       0.1539             NaN         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           11_425.41       480.71    11_906.12       0.4729          1.0862         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           11_425.41       533.75    11_959.17       0.6437          1.0440         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           11_425.41       479.36    11_904.78       0.4722          1.0865         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           11_425.41       538.69    11_964.10       0.6428          1.0443         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           11_425.41       490.27    11_915.68       0.4710          1.0869         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           11_425.41       551.67    11_977.08       0.6415          1.0445         7.42
IVF-Binary-1024-nl316-pca (self)                      11_425.41     1_553.55    12_978.97       0.4656          1.1056         7.42
IVF-Binary-256-nl158-np7-rf0-signed (query)            3_903.83       111.36     4_015.20       0.0905             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-signed (query)           3_903.83       113.42     4_017.25       0.0889             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-signed (query)           3_903.83       116.49     4_020.33       0.0879             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-signed (query)           3_903.83       167.22     4_071.05       0.3414          1.1450         1.93
IVF-Binary-256-nl158-np7-rf20-signed (query)           3_903.83       220.16     4_123.99       0.4835          1.0837         1.93
IVF-Binary-256-nl158-np12-rf10-signed (query)          3_903.83       170.00     4_073.83       0.3391          1.1459         1.93
IVF-Binary-256-nl158-np12-rf20-signed (query)          3_903.83       221.88     4_125.72       0.4816          1.0842         1.93
IVF-Binary-256-nl158-np17-rf10-signed (query)          3_903.83       176.50     4_080.33       0.3378          1.1466         1.93
IVF-Binary-256-nl158-np17-rf20-signed (query)          3_903.83       231.03     4_134.86       0.4805          1.0846         1.93
IVF-Binary-256-nl158-signed (self)                     3_903.83       509.23     4_413.07       0.3662          1.1503         1.93
IVF-Binary-256-nl223-np11-rf0-signed (query)           3_199.87       117.60     3_317.47       0.0949             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-signed (query)           3_199.87       119.06     3_318.94       0.0906             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-signed (query)           3_199.87       123.26     3_323.13       0.0885             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-signed (query)          3_199.87       174.63     3_374.51       0.3513          1.1397         2.00
IVF-Binary-256-nl223-np11-rf20-signed (query)          3_199.87       222.40     3_422.27       0.4946          1.0802         2.00
IVF-Binary-256-nl223-np14-rf10-signed (query)          3_199.87       174.95     3_374.82       0.3420          1.1459         2.00
IVF-Binary-256-nl223-np14-rf20-signed (query)          3_199.87       227.28     3_427.15       0.4848          1.0840         2.00
IVF-Binary-256-nl223-np21-rf10-signed (query)          3_199.87       183.80     3_383.67       0.3385          1.1480         2.00
IVF-Binary-256-nl223-np21-rf20-signed (query)          3_199.87       239.44     3_439.32       0.4816          1.0850         2.00
IVF-Binary-256-nl223-signed (self)                     3_199.87       529.64     3_729.51       0.3692          1.1507         2.00
IVF-Binary-256-nl316-np15-rf0-signed (query)           3_424.21       123.41     3_547.61       0.0923             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-signed (query)           3_424.21       124.25     3_548.45       0.0905             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-signed (query)           3_424.21       127.85     3_552.05       0.0892             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-signed (query)          3_424.21       179.38     3_603.58       0.3480          1.1422         2.09
IVF-Binary-256-nl316-np15-rf20-signed (query)          3_424.21       230.72     3_654.93       0.4900          1.0824         2.09
IVF-Binary-256-nl316-np17-rf10-signed (query)          3_424.21       179.30     3_603.50       0.3442          1.1442         2.09
IVF-Binary-256-nl316-np17-rf20-signed (query)          3_424.21       242.00     3_666.21       0.4865          1.0835         2.09
IVF-Binary-256-nl316-np25-rf10-signed (query)          3_424.21       185.41     3_609.62       0.3411          1.1460         2.09
IVF-Binary-256-nl316-np25-rf20-signed (query)          3_424.21       239.79     3_663.99       0.4830          1.0845         2.09
IVF-Binary-256-nl316-signed (self)                     3_424.21       543.84     3_968.05       0.3712          1.1489         2.09
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
Exhaustive (query)                                        21.17     9_518.38     9_539.55       1.0000          1.0000        97.66
Exhaustive (self)                                         21.17    31_855.06    31_876.23       1.0000          1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_641.74       375.30     6_017.04       0.0610             NaN         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_641.74       497.91     6_139.64       0.2580          1.1443         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_641.74       616.56     6_258.30       0.3781          1.0904         2.03
ExhaustiveBinary-256-random (self)                     5_641.74     1_625.37     7_267.11       0.2746          1.1471         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_117.00       392.52     6_509.52       0.1754             NaN         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_117.00       518.42     6_635.42       0.4664          1.2428         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_117.00       645.78     6_762.78       0.5850          1.1744         2.03
ExhaustiveBinary-256-pca (self)                        6_117.00     1_713.76     7_830.76       0.4693          1.2512         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_220.54       642.92    11_863.46       0.0950             NaN         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_220.54       765.63    11_986.18       0.3259          1.1053         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_220.54       897.21    12_117.75       0.4522          1.0627         4.05
ExhaustiveBinary-512-random (self)                    11_220.54     2_539.33    13_759.87       0.3417          1.1072         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_778.15       668.01    12_446.16       0.1672             NaN         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_778.15       791.36    12_569.51       0.4213          1.4308         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_778.15       928.83    12_706.98       0.5297          1.2408         4.05
ExhaustiveBinary-512-pca (self)                       11_778.15     2_631.85    14_410.00       0.4104          1.6976         4.05
ExhaustiveBinary-1024-random_no_rr (query)            21_914.98     1_171.52    23_086.50       0.1421             NaN         8.10
ExhaustiveBinary-1024-random-rf10 (query)             21_914.98     1_311.07    23_226.05       0.4020          1.0734         8.10
ExhaustiveBinary-1024-random-rf20 (query)             21_914.98     1_446.13    23_361.11       0.5400          1.0418         8.10
ExhaustiveBinary-1024-random (self)                   21_914.98     4_343.55    26_258.53       0.4166          1.0795         8.10
ExhaustiveBinary-1024-pca_no_rr (query)               22_957.81     1_200.63    24_158.44       0.2385             NaN         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                22_957.81     1_343.98    24_301.78       0.6735          1.0459         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                22_957.81     1_483.34    24_441.14       0.8214          1.0198         8.11
ExhaustiveBinary-1024-pca (self)                      22_957.81     4_473.84    27_431.64       0.6814          1.0492         8.11
ExhaustiveBinary-512-signed_no_rr (query)             11_073.31       661.83    11_735.14       0.0950             NaN         4.05
ExhaustiveBinary-512-signed-rf10 (query)              11_073.31       767.81    11_841.12       0.3259          1.1053         4.05
ExhaustiveBinary-512-signed-rf20 (query)              11_073.31       895.29    11_968.60       0.4522          1.0627         4.05
ExhaustiveBinary-512-signed (self)                    11_073.31     2_577.13    13_650.44       0.3417          1.1072         4.05
IVF-Binary-256-nl158-np7-rf0-random (query)            8_187.60       224.41     8_412.01       0.0636             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           8_187.60       227.31     8_414.91       0.0625             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           8_187.60       229.27     8_416.87       0.0616             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           8_187.60       301.65     8_489.25       0.2619          1.1425         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           8_187.60       378.49     8_566.08       0.3812          1.0895         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          8_187.60       301.54     8_489.14       0.2599          1.1431         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          8_187.60       383.82     8_571.41       0.3795          1.0898         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          8_187.60       305.95     8_493.55       0.2591          1.1436         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          8_187.60       390.40     8_578.00       0.3791          1.0899         2.34
IVF-Binary-256-nl158-random (self)                     8_187.60       923.94     9_111.54       0.2768          1.1457         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           6_548.14       235.28     6_783.42       0.0686             NaN         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)           6_548.14       237.43     6_785.57       0.0660             NaN         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)           6_548.14       242.04     6_790.18       0.0628             NaN         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)          6_548.14       331.36     6_879.50       0.2760          1.1336         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)          6_548.14       413.44     6_961.58       0.3976          1.0826         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)          6_548.14       316.84     6_864.98       0.2702          1.1376         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)          6_548.14       394.82     6_942.95       0.3920          1.0851         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)          6_548.14       321.51     6_869.65       0.2626          1.1421         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)          6_548.14       408.67     6_956.80       0.3839          1.0881         2.46
IVF-Binary-256-nl223-random (self)                     6_548.14       966.95     7_515.08       0.2870          1.1394         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)           7_012.77       249.55     7_262.32       0.0668             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           7_012.77       250.59     7_263.36       0.0643             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           7_012.77       253.51     7_266.28       0.0631             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          7_012.77       329.28     7_342.05       0.2736          1.1351         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          7_012.77       407.06     7_419.84       0.3953          1.0838         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          7_012.77       328.00     7_340.78       0.2680          1.1386         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          7_012.77       408.65     7_421.43       0.3895          1.0861         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          7_012.77       332.11     7_344.88       0.2644          1.1407         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          7_012.77       416.10     7_428.87       0.3855          1.0874         2.65
IVF-Binary-256-nl316-random (self)                     7_012.77     1_006.71     8_019.49       0.2843          1.1412         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               8_673.97       237.22     8_911.19       0.1938             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              8_673.97       241.59     8_915.57       0.1882             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              8_673.97       237.56     8_911.53       0.1872             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              8_673.97       327.43     9_001.41       0.5815          1.0747         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              8_673.97       414.50     9_088.47       0.7420          1.0341         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             8_673.97       356.81     9_030.79       0.5579          1.0959         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             8_673.97       416.71     9_090.68       0.7182          1.0418         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             8_673.97       332.07     9_006.04       0.5519          1.1117         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             8_673.97       418.97     9_092.95       0.7102          1.0477         2.34
IVF-Binary-256-nl158-pca (self)                        8_673.97     1_030.63     9_704.60       0.5788          1.0992         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              7_151.07       244.97     7_396.04       0.1888             NaN         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              7_151.07       246.97     7_398.04       0.1878             NaN         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              7_151.07       258.82     7_409.89       0.1869             NaN         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             7_151.07       339.77     7_490.84       0.5604          1.0859         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             7_151.07       424.59     7_575.66       0.7217          1.0380         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             7_151.07       341.21     7_492.28       0.5555          1.0948         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             7_151.07       427.78     7_578.85       0.7149          1.0416         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             7_151.07       346.83     7_497.90       0.5496          1.1117         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             7_151.07       436.14     7_587.21       0.7064          1.0483         2.47
IVF-Binary-256-nl223-pca (self)                        7_151.07     1_069.48     8_220.55       0.5761          1.0985         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              7_533.10       258.90     7_792.00       0.1885             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              7_533.10       270.50     7_803.60       0.1879             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              7_533.10       264.02     7_797.12       0.1867             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             7_533.10       353.17     7_886.27       0.5593          1.0870         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             7_533.10       438.62     7_971.73       0.7204          1.0383         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             7_533.10       352.97     7_886.08       0.5569          1.0907         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             7_533.10       440.63     7_973.74       0.7171          1.0400         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             7_533.10       356.90     7_890.00       0.5507          1.1073         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             7_533.10       445.53     7_978.63       0.7084          1.0463         2.65
IVF-Binary-256-nl316-pca (self)                        7_533.10     1_112.59     8_645.69       0.5776          1.0944         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           13_677.34       421.21    14_098.55       0.0961             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          13_677.34       432.52    14_109.87       0.0956             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          13_677.34       430.59    14_107.93       0.0952             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          13_677.34       502.55    14_179.89       0.3265          1.1051         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          13_677.34       578.93    14_256.28       0.4526          1.0625         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         13_677.34       505.31    14_182.65       0.3262          1.1052         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         13_677.34       582.84    14_260.18       0.4524          1.0626         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         13_677.34       507.43    14_184.78       0.3262          1.1052         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         13_677.34       589.86    14_267.21       0.4524          1.0626         4.36
IVF-Binary-512-nl158-random (self)                    13_677.34     1_595.47    15_272.81       0.3421          1.1071         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          12_061.46       430.42    12_491.88       0.1002             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          12_061.46       434.18    12_495.64       0.0985             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          12_061.46       438.76    12_500.22       0.0961             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         12_061.46       515.04    12_576.50       0.3350          1.1015         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         12_061.46       591.78    12_653.24       0.4609          1.0605         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         12_061.46       514.45    12_575.91       0.3327          1.1027         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         12_061.46       594.79    12_656.25       0.4583          1.0612         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         12_061.46       523.76    12_585.22       0.3291          1.1045         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         12_061.46       605.63    12_667.09       0.4548          1.0621         4.49
IVF-Binary-512-nl223-random (self)                    12_061.46     1_642.34    13_703.79       0.3479          1.1048         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          12_506.89       457.27    12_964.16       0.0994             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          12_506.89       445.22    12_952.11       0.0977             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          12_506.89       456.89    12_963.78       0.0967             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         12_506.89       527.56    13_034.45       0.3339          1.1019         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         12_506.89       609.84    13_116.73       0.4598          1.0607         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         12_506.89       526.68    13_033.57       0.3308          1.1034         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         12_506.89       612.43    13_119.32       0.4569          1.0615         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         12_506.89       535.58    13_042.47       0.3288          1.1044         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         12_506.89       614.70    13_121.59       0.4546          1.0621         4.67
IVF-Binary-512-nl316-random (self)                    12_506.89     1_689.64    14_196.53       0.3472          1.1051         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              14_820.90       437.42    15_258.33       0.2262             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             14_820.90       436.20    15_257.10       0.2196             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             14_820.90       440.24    15_261.14       0.2176             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             14_820.90       528.49    15_349.40       0.6546          1.0666         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             14_820.90       621.02    15_441.92       0.8045          1.0295         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            14_820.90       527.66    15_348.56       0.6281          1.0925         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            14_820.90       615.08    15_435.99       0.7783          1.0388         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            14_820.90       535.46    15_356.37       0.6191          1.1147         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            14_820.90       620.33    15_441.23       0.7661          1.0484         4.36
IVF-Binary-512-nl158-pca (self)                       14_820.90     1_702.30    16_523.20       0.6370          1.0961         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             12_683.76       448.24    13_131.99       0.2206             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             12_683.76       452.23    13_135.98       0.2187             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             12_683.76       458.43    13_142.18       0.2163             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            12_683.76       542.52    13_226.28       0.6313          1.0820         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            12_683.76       625.28    13_309.04       0.7819          1.0348         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            12_683.76       545.10    13_228.85       0.6236          1.0951         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            12_683.76       629.01    13_312.77       0.7725          1.0399         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            12_683.76       550.10    13_233.86       0.6121          1.1200         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            12_683.76       642.62    13_326.37       0.7578          1.0519         4.49
IVF-Binary-512-nl223-pca (self)                       12_683.76     1_742.78    14_426.54       0.6320          1.0984         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             13_134.68       464.12    13_598.80       0.2202             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             13_134.68       465.76    13_600.44       0.2192             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             13_134.68       464.40    13_599.08       0.2166             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            13_134.68       556.89    13_691.58       0.6305          1.0828         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            13_134.68       639.63    13_774.31       0.7810          1.0349         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            13_134.68       555.58    13_690.27       0.6268          1.0887         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            13_134.68       655.41    13_790.10       0.7762          1.0372         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            13_134.68       561.04    13_695.73       0.6154          1.1131         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            13_134.68       662.13    13_796.81       0.7619          1.0479         4.67
IVF-Binary-512-nl316-pca (self)                       13_134.68     1_786.75    14_921.44       0.6351          1.0922         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          24_540.67       810.71    25_351.38       0.1426             NaN         8.41
IVF-Binary-1024-nl158-np12-rf0-random (query)         24_540.67       815.23    25_355.90       0.1424             NaN         8.41
IVF-Binary-1024-nl158-np17-rf0-random (query)         24_540.67       817.65    25_358.32       0.1422             NaN         8.41
IVF-Binary-1024-nl158-np7-rf10-random (query)         24_540.67       893.98    25_434.65       0.4022          1.0734         8.41
IVF-Binary-1024-nl158-np7-rf20-random (query)         24_540.67       969.19    25_509.87       0.5402          1.0418         8.41
IVF-Binary-1024-nl158-np12-rf10-random (query)        24_540.67       895.74    25_436.41       0.4021          1.0734         8.41
IVF-Binary-1024-nl158-np12-rf20-random (query)        24_540.67       977.81    25_518.49       0.5401          1.0418         8.41
IVF-Binary-1024-nl158-np17-rf10-random (query)        24_540.67       909.77    25_450.44       0.4021          1.0734         8.41
IVF-Binary-1024-nl158-np17-rf20-random (query)        24_540.67     1_058.41    25_599.09       0.5401          1.0418         8.41
IVF-Binary-1024-nl158-random (self)                   24_540.67     2_915.42    27_456.09       0.4167          1.0794         8.41
IVF-Binary-1024-nl223-np11-rf0-random (query)         22_921.45       824.15    23_745.61       0.1449             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         22_921.45       828.57    23_750.02       0.1443             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         22_921.45       836.32    23_757.77       0.1431             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        22_921.45       906.49    23_827.94       0.4070          1.0720         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        22_921.45       985.51    23_906.96       0.5438          1.0411         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        22_921.45       926.07    23_847.53       0.4058          1.0724         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        22_921.45       986.22    23_907.67       0.5425          1.0413         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        22_921.45       921.92    23_843.37       0.4036          1.0731         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        22_921.45     1_004.53    23_925.98       0.5407          1.0417         8.54
IVF-Binary-1024-nl223-random (self)                   22_921.45     2_950.54    25_871.99       0.4198          1.0786         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         23_371.46       863.35    24_234.82       0.1442             NaN         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)         23_371.46       840.23    24_211.69       0.1435             NaN         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)         23_371.46       848.45    24_219.91       0.1431             NaN         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)        23_371.46       918.89    24_290.35       0.4064          1.0722         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)        23_371.46       998.49    24_369.95       0.5438          1.0412         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)        23_371.46       921.13    24_292.59       0.4049          1.0727         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)        23_371.46     1_018.74    24_390.20       0.5421          1.0415         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)        23_371.46       929.88    24_301.34       0.4037          1.0731         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)        23_371.46     1_010.86    24_382.32       0.5409          1.0417         8.72
IVF-Binary-1024-nl316-random (self)                   23_371.46     3_021.16    26_392.62       0.4193          1.0788         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)             25_634.64       839.29    26_473.93       0.2433             NaN         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            25_634.64       841.24    26_475.88       0.2393             NaN         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            25_634.64       898.41    26_533.05       0.2387             NaN         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            25_634.64       950.08    26_584.72       0.6896          1.0422         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            25_634.64     1_008.14    26_642.78       0.8340          1.0181         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           25_634.64       930.34    26_564.98       0.6768          1.0450         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           25_634.64     1_023.15    26_657.79       0.8247          1.0193         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           25_634.64       945.15    26_579.79       0.6743          1.0455         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           25_634.64     1_019.33    26_653.97       0.8223          1.0196         8.42
IVF-Binary-1024-nl158-pca (self)                      25_634.64     3_033.54    28_668.18       0.6844          1.0483         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            24_005.96       876.43    24_882.39       0.2399             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            24_005.96       855.18    24_861.14       0.2393             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            24_005.96       866.88    24_872.84       0.2388             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           24_005.96       939.66    24_945.61       0.6779          1.0436         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           24_005.96     1_033.43    25_039.39       0.8260          1.0186         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           24_005.96       949.96    24_955.92       0.6760          1.0445         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           24_005.96     1_026.20    25_032.16       0.8243          1.0190         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           24_005.96       954.60    24_960.56       0.6743          1.0450         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           24_005.96     1_044.92    25_050.88       0.8224          1.0193         8.54
IVF-Binary-1024-nl223-pca (self)                      24_005.96     3_099.45    27_105.41       0.6839          1.0478         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            24_426.87       864.76    25_291.64       0.2395             NaN         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            24_426.87       869.01    25_295.89       0.2392             NaN         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            24_426.87       875.85    25_302.73       0.2386             NaN         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           24_426.87       953.94    25_380.82       0.6771          1.0438         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           24_426.87     1_034.98    25_461.86       0.8256          1.0186         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           24_426.87       953.65    25_380.52       0.6764          1.0442         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           24_426.87     1_040.70    25_467.58       0.8249          1.0188         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           24_426.87       962.60    25_389.48       0.6741          1.0449         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           24_426.87     1_048.62    25_475.49       0.8227          1.0192         8.73
IVF-Binary-1024-nl316-pca (self)                      24_426.87     3_120.17    27_547.04       0.6845          1.0475         8.73
IVF-Binary-512-nl158-np7-rf0-signed (query)           13_692.08       427.65    14_119.73       0.0961             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-signed (query)          13_692.08       437.68    14_129.76       0.0956             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-signed (query)          13_692.08       431.30    14_123.38       0.0952             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-signed (query)          13_692.08       502.52    14_194.60       0.3265          1.1051         4.36
IVF-Binary-512-nl158-np7-rf20-signed (query)          13_692.08       578.47    14_270.55       0.4526          1.0625         4.36
IVF-Binary-512-nl158-np12-rf10-signed (query)         13_692.08       504.41    14_196.49       0.3262          1.1052         4.36
IVF-Binary-512-nl158-np12-rf20-signed (query)         13_692.08       582.82    14_274.90       0.4524          1.0626         4.36
IVF-Binary-512-nl158-np17-rf10-signed (query)         13_692.08       506.67    14_198.75       0.3262          1.1052         4.36
IVF-Binary-512-nl158-np17-rf20-signed (query)         13_692.08       596.23    14_288.31       0.4524          1.0626         4.36
IVF-Binary-512-nl158-signed (self)                    13_692.08     1_601.44    15_293.52       0.3421          1.1071         4.36
IVF-Binary-512-nl223-np11-rf0-signed (query)          12_072.22       433.71    12_505.93       0.1002             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-signed (query)          12_072.22       434.14    12_506.36       0.0985             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-signed (query)          12_072.22       443.08    12_515.31       0.0961             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-signed (query)         12_072.22       513.27    12_585.49       0.3350          1.1015         4.49
IVF-Binary-512-nl223-np11-rf20-signed (query)         12_072.22       590.19    12_662.42       0.4609          1.0605         4.49
IVF-Binary-512-nl223-np14-rf10-signed (query)         12_072.22       515.43    12_587.66       0.3327          1.1027         4.49
IVF-Binary-512-nl223-np14-rf20-signed (query)         12_072.22       594.07    12_666.29       0.4583          1.0612         4.49
IVF-Binary-512-nl223-np21-rf10-signed (query)         12_072.22       523.29    12_595.52       0.3291          1.1045         4.49
IVF-Binary-512-nl223-np21-rf20-signed (query)         12_072.22       605.92    12_678.15       0.4548          1.0621         4.49
IVF-Binary-512-nl223-signed (self)                    12_072.22     1_639.45    13_711.67       0.3479          1.1048         4.49
IVF-Binary-512-nl316-np15-rf0-signed (query)          12_540.06       446.46    12_986.52       0.0994             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-signed (query)          12_540.06       448.31    12_988.37       0.0977             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-signed (query)          12_540.06       454.55    12_994.61       0.0967             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-signed (query)         12_540.06       527.74    13_067.81       0.3339          1.1019         4.67
IVF-Binary-512-nl316-np15-rf20-signed (query)         12_540.06       605.63    13_145.70       0.4598          1.0607         4.67
IVF-Binary-512-nl316-np17-rf10-signed (query)         12_540.06       528.17    13_068.23       0.3308          1.1034         4.67
IVF-Binary-512-nl316-np17-rf20-signed (query)         12_540.06       611.74    13_151.80       0.4569          1.0615         4.67
IVF-Binary-512-nl316-np25-rf10-signed (query)         12_540.06       542.74    13_082.80       0.3288          1.1044         4.67
IVF-Binary-512-nl316-np25-rf20-signed (query)         12_540.06       665.07    13_205.14       0.4546          1.0621         4.67
IVF-Binary-512-nl316-signed (self)                    12_540.06     1_689.83    14_229.89       0.3472          1.1051         4.67
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
Exhaustive (query)                                        29.67    15_711.18    15_740.85       1.0000          1.0000       146.48
Exhaustive (self)                                         29.67    52_088.30    52_117.97       1.0000          1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)              8_820.53       488.31     9_308.84       0.0519             NaN         2.28
ExhaustiveBinary-256-random-rf10 (query)               8_820.53       615.78     9_436.31       0.2246          1.1305         2.28
ExhaustiveBinary-256-random-rf20 (query)               8_820.53       757.70     9_578.23       0.3310          1.0856         2.28
ExhaustiveBinary-256-random (self)                     8_820.53     2_017.76    10_838.29       0.2354          1.1317         2.28
ExhaustiveBinary-256-pca_no_rr (query)                 9_352.13       495.04     9_847.17       0.1692             NaN         2.28
ExhaustiveBinary-256-pca-rf10 (query)                  9_352.13       647.32     9_999.45       0.4636          1.1896         2.28
ExhaustiveBinary-256-pca-rf20 (query)                  9_352.13       794.26    10_146.39       0.5918          1.1225         2.28
ExhaustiveBinary-256-pca (self)                        9_352.13     2_135.50    11_487.63       0.4806          1.1914         2.28
ExhaustiveBinary-512-random_no_rr (query)             17_246.79       859.11    18_105.90       0.0750             NaN         4.55
ExhaustiveBinary-512-random-rf10 (query)              17_246.79     1_000.64    18_247.43       0.2723          1.1002         4.55
ExhaustiveBinary-512-random-rf20 (query)              17_246.79     1_147.09    18_393.88       0.3825          1.0620         4.55
ExhaustiveBinary-512-random (self)                    17_246.79     3_304.67    20_551.46       0.2876          1.0943         4.55
ExhaustiveBinary-512-pca_no_rr (query)                18_031.02       878.77    18_909.79       0.1883             NaN         4.55
ExhaustiveBinary-512-pca-rf10 (query)                 18_031.02     1_031.05    19_062.07       0.4652          1.2549         4.55
ExhaustiveBinary-512-pca-rf20 (query)                 18_031.02     1_182.54    19_213.56       0.5760          1.1920         4.55
ExhaustiveBinary-512-pca (self)                       18_031.02     3_463.29    21_494.31       0.4711          1.2568         4.55
ExhaustiveBinary-1024-random_no_rr (query)            34_336.24     1_599.35    35_935.59       0.1134             NaN         9.10
ExhaustiveBinary-1024-random-rf10 (query)             34_336.24     1_758.23    36_094.47       0.3258          1.0732         9.10
ExhaustiveBinary-1024-random-rf20 (query)             34_336.24     1_913.69    36_249.93       0.4396          1.0445         9.10
ExhaustiveBinary-1024-random (self)                   34_336.24     5_858.74    40_194.98       0.3340          1.0747         9.10
ExhaustiveBinary-1024-pca_no_rr (query)               35_378.00     1_623.40    37_001.40       0.2682             NaN         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                35_378.00     1_793.91    37_171.91       0.7078          1.0551         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                35_378.00     1_962.89    37_340.89       0.8379          1.0248         9.11
ExhaustiveBinary-1024-pca (self)                      35_378.00     5_954.70    41_332.70       0.7159          1.0572         9.11
ExhaustiveBinary-768-signed_no_rr (query)             25_839.77     1_243.63    27_083.40       0.0960             NaN         6.83
ExhaustiveBinary-768-signed-rf10 (query)              25_839.77     1_389.98    27_229.75       0.3033          1.0835         6.83
ExhaustiveBinary-768-signed-rf20 (query)              25_839.77     1_543.43    27_383.20       0.4156          1.0509         6.83
ExhaustiveBinary-768-signed (self)                    25_839.77     4_603.41    30_443.18       0.3145          1.0814         6.83
IVF-Binary-256-nl158-np7-rf0-random (query)           12_770.18       345.16    13_115.35       0.0543             NaN         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)          12_770.18       346.58    13_116.76       0.0533             NaN         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)          12_770.18       349.15    13_119.34       0.0525             NaN         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)          12_770.18       438.81    13_208.99       0.2287          1.1286         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)          12_770.18       530.81    13_300.99       0.3341          1.0845         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)         12_770.18       438.27    13_208.46       0.2268          1.1291         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)         12_770.18       537.26    13_307.45       0.3326          1.0848         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)         12_770.18       441.17    13_211.35       0.2261          1.1295         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)         12_770.18       542.45    13_312.63       0.3324          1.0848         2.74
IVF-Binary-256-nl158-random (self)                    12_770.18     1_359.57    14_129.75       0.2377          1.1303         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)          10_157.32       361.55    10_518.87       0.0576             NaN         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)          10_157.32       363.83    10_521.15       0.0554             NaN         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)          10_157.32       365.75    10_523.07       0.0536             NaN         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)         10_157.32       461.50    10_618.83       0.2443          1.1175         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)         10_157.32       557.70    10_715.02       0.3525          1.0758         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)         10_157.32       458.14    10_615.46       0.2358          1.1233         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)         10_157.32       560.86    10_718.18       0.3421          1.0805         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)         10_157.32       467.63    10_624.95       0.2314          1.1266         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)         10_157.32       564.59    10_721.92       0.3371          1.0830         2.93
IVF-Binary-256-nl223-random (self)                    10_157.32     1_423.16    11_580.48       0.2478          1.1231         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)          10_787.25       387.00    11_174.25       0.0562             NaN         3.20
IVF-Binary-256-nl316-np17-rf0-random (query)          10_787.25       387.46    11_174.70       0.0546             NaN         3.20
IVF-Binary-256-nl316-np25-rf0-random (query)          10_787.25       388.60    11_175.84       0.0542             NaN         3.20
IVF-Binary-256-nl316-np15-rf10-random (query)         10_787.25       483.27    11_270.52       0.2380          1.1223         3.20
IVF-Binary-256-nl316-np15-rf20-random (query)         10_787.25       584.93    11_372.18       0.3451          1.0793         3.20
IVF-Binary-256-nl316-np17-rf10-random (query)         10_787.25       494.57    11_281.82       0.2342          1.1248         3.20
IVF-Binary-256-nl316-np17-rf20-random (query)         10_787.25       579.88    11_367.13       0.3405          1.0812         3.20
IVF-Binary-256-nl316-np25-rf10-random (query)         10_787.25       482.60    11_269.85       0.2334          1.1257         3.20
IVF-Binary-256-nl316-np25-rf20-random (query)         10_787.25       585.24    11_372.48       0.3378          1.0825         3.20
IVF-Binary-256-nl316-random (self)                    10_787.25     1_499.55    12_286.79       0.2452          1.1255         3.20
IVF-Binary-256-nl158-np7-rf0-pca (query)              13_477.26       353.81    13_831.07       0.1802             NaN         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)             13_477.26       356.66    13_833.92       0.1751             NaN         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)             13_477.26       361.48    13_838.74       0.1742             NaN         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)             13_477.26       470.25    13_947.51       0.5384          1.0693         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)             13_477.26       575.73    14_052.98       0.7007          1.0345         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)            13_477.26       473.06    13_950.32       0.5170          1.0848         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)            13_477.26       582.18    14_059.44       0.6786          1.0397         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)            13_477.26       485.66    13_962.91       0.5125          1.0940         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)            13_477.26       585.98    14_063.24       0.6722          1.0429         2.74
IVF-Binary-256-nl158-pca (self)                       13_477.26     1_609.99    15_087.25       0.5381          1.0862         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)             10_881.06       372.87    11_253.93       0.1762             NaN         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)             10_881.06       374.00    11_255.07       0.1749             NaN         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)             10_881.06       378.04    11_259.11       0.1739             NaN         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)            10_881.06       488.81    11_369.88       0.5223          1.0736         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)            10_881.06       595.92    11_476.98       0.6847          1.0352         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)            10_881.06       490.86    11_371.92       0.5169          1.0800         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)            10_881.06       602.42    11_483.48       0.6780          1.0376         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)            10_881.06       495.61    11_376.67       0.5122          1.0884         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)            10_881.06       605.03    11_486.09       0.6712          1.0410         2.93
IVF-Binary-256-nl223-pca (self)                       10_881.06     1_556.75    12_437.82       0.5378          1.0810         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)             11_509.70       398.94    11_908.64       0.1758             NaN         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)             11_509.70       418.24    11_927.94       0.1752             NaN         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)             11_509.70       395.86    11_905.56       0.1742             NaN         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)            11_509.70       512.32    12_022.03       0.5205          1.0749         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)            11_509.70       619.14    12_128.84       0.6828          1.0359         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)            11_509.70       509.42    12_019.12       0.5180          1.0775         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)            11_509.70       619.46    12_129.16       0.6798          1.0369         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)            11_509.70       514.16    12_023.86       0.5131          1.0873         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)            11_509.70       624.31    12_134.01       0.6725          1.0406         3.21
IVF-Binary-256-nl316-pca (self)                       11_509.70     1_627.50    13_137.20       0.5390          1.0787         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)           21_345.04       646.70    21_991.75       0.0763             NaN         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)          21_345.04       648.20    21_993.24       0.0758             NaN         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)          21_345.04       671.27    22_016.31       0.0753             NaN         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)          21_345.04       745.68    22_090.73       0.2733          1.1001         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)          21_345.04       843.19    22_188.23       0.3831          1.0619         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)         21_345.04       755.94    22_100.99       0.2730          1.1001         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)         21_345.04       853.71    22_198.76       0.3829          1.0619         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)         21_345.04       753.64    22_098.69       0.2729          1.1001         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)         21_345.04       859.38    22_204.42       0.3829          1.0619         5.02
IVF-Binary-512-nl158-random (self)                    21_345.04     2_411.17    23_756.21       0.2882          1.0941         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)          18_697.80       672.64    19_370.44       0.0810             NaN         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)          18_697.80       674.60    19_372.40       0.0783             NaN         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)          18_697.80       684.10    19_381.91       0.0769             NaN         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)         18_697.80       774.84    19_472.64       0.2824          1.0950         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)         18_697.80       896.64    19_594.44       0.3900          1.0593         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)         18_697.80       779.09    19_476.89       0.2781          1.0974         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)         18_697.80       917.81    19_615.62       0.3866          1.0605         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)         18_697.80       782.19    19_479.99       0.2763          1.0986         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)         18_697.80       874.70    19_572.50       0.3846          1.0613         5.21
IVF-Binary-512-nl223-random (self)                    18_697.80     2_494.33    21_192.13       0.2924          1.0922         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)          19_328.21       707.04    20_035.25       0.0790             NaN         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)          19_328.21       690.11    20_018.32       0.0778             NaN         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)          19_328.21       701.99    20_030.21       0.0774             NaN         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)         19_328.21       790.55    20_118.76       0.2797          1.0968         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)         19_328.21       885.75    20_213.96       0.3911          1.0591         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)         19_328.21       799.39    20_127.60       0.2779          1.0977         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)         19_328.21       891.49    20_219.70       0.3886          1.0598         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)         19_328.21       795.49    20_123.71       0.2768          1.0983         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)         19_328.21       897.50    20_225.71       0.3867          1.0604         5.48
IVF-Binary-512-nl316-random (self)                    19_328.21     2_533.41    21_861.62       0.2929          1.0919         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)              22_170.08       658.35    22_828.44       0.2344             NaN         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)             22_170.08       666.25    22_836.34       0.2281             NaN         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)             22_170.08       662.63    22_832.71       0.2267             NaN         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)             22_170.08       788.33    22_958.42       0.6555          1.0605         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)             22_170.08       886.98    23_057.07       0.8024          1.0284         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)            22_170.08       781.36    22_951.44       0.6310          1.0810         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)            22_170.08       883.40    23_053.48       0.7779          1.0353         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)            22_170.08       782.81    22_952.90       0.6242          1.0966         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)            22_170.08       888.04    23_058.13       0.7689          1.0413         5.02
IVF-Binary-512-nl158-pca (self)                       22_170.08     2_524.22    24_694.30       0.6460          1.0830         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)             19_484.07       677.12    20_161.19       0.2295             NaN         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)             19_484.07       704.15    20_188.22       0.2278             NaN         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)             19_484.07       682.64    20_166.71       0.2260             NaN         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)            19_484.07       798.22    20_282.29       0.6368          1.0675         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)            19_484.07       909.13    20_393.20       0.7859          1.0303         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)            19_484.07       795.43    20_279.50       0.6298          1.0772         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)            19_484.07       908.25    20_392.32       0.7771          1.0337         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)            19_484.07       802.42    20_286.49       0.6213          1.0942         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)            19_484.07       909.34    20_393.40       0.7660          1.0404         5.21
IVF-Binary-512-nl223-pca (self)                       19_484.07     2_593.67    22_077.73       0.6445          1.0786         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)             20_120.93       700.95    20_821.88       0.2290             NaN         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)             20_120.93       700.31    20_821.24       0.2282             NaN         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)             20_120.93       705.21    20_826.14       0.2264             NaN         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)            20_120.93       816.64    20_937.57       0.6348          1.0692         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)            20_120.93       944.08    21_065.01       0.7832          1.0307         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)            20_120.93       819.99    20_940.92       0.6316          1.0732         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)            20_120.93       922.40    21_043.33       0.7793          1.0322         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)            20_120.93       820.80    20_941.73       0.6230          1.0908         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)            20_120.93       929.95    21_050.87       0.7681          1.0390         5.48
IVF-Binary-512-nl316-pca (self)                       20_120.93     2_682.26    22_803.18       0.6463          1.0746         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)          38_446.90     1_250.81    39_697.72       0.1139             NaN         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)         38_446.90     1_253.67    39_700.58       0.1137             NaN         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)         38_446.90     1_256.48    39_703.38       0.1135             NaN         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)         38_446.90     1_348.64    39_795.54       0.3260          1.0732         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)         38_446.90     1_446.88    39_893.79       0.4398          1.0445         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)        38_446.90     1_352.29    39_799.20       0.3260          1.0732         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)        38_446.90     1_473.46    39_920.36       0.4398          1.0445         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)        38_446.90     1_356.79    39_803.70       0.3260          1.0732         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)        38_446.90     1_459.51    39_906.41       0.4398          1.0445         9.57
IVF-Binary-1024-nl158-random (self)                   38_446.90     4_440.41    42_887.31       0.3341          1.0746         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)         35_755.59     1_269.50    37_025.09       0.1166             NaN         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)         35_755.59     1_270.99    37_026.58       0.1152             NaN         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)         35_755.59     1_277.44    37_033.03       0.1145             NaN         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)        35_755.59     1_369.14    37_124.73       0.3293          1.0717         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)        35_755.59     1_465.51    37_221.09       0.4439          1.0435         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)        35_755.59     1_370.16    37_125.74       0.3277          1.0725         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)        35_755.59     1_471.35    37_226.94       0.4424          1.0439         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)        35_755.59     1_378.73    37_134.32       0.3268          1.0729         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)        35_755.59     1_480.32    37_235.90       0.4415          1.0441         9.76
IVF-Binary-1024-nl223-random (self)                   35_755.59     4_476.35    40_231.94       0.3366          1.0738         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)         36_436.79     1_292.11    37_728.90       0.1161             NaN        10.03
IVF-Binary-1024-nl316-np17-rf0-random (query)         36_436.79     1_292.84    37_729.63       0.1154             NaN        10.03
IVF-Binary-1024-nl316-np25-rf0-random (query)         36_436.79     1_299.14    37_735.93       0.1150             NaN        10.03
IVF-Binary-1024-nl316-np15-rf10-random (query)        36_436.79     1_393.37    37_830.15       0.3292          1.0719        10.03
IVF-Binary-1024-nl316-np15-rf20-random (query)        36_436.79     1_490.40    37_927.19       0.4445          1.0434        10.03
IVF-Binary-1024-nl316-np17-rf10-random (query)        36_436.79     1_389.58    37_826.37       0.3280          1.0723        10.03
IVF-Binary-1024-nl316-np17-rf20-random (query)        36_436.79     1_501.06    37_937.85       0.4430          1.0437        10.03
IVF-Binary-1024-nl316-np25-rf10-random (query)        36_436.79     1_400.51    37_837.30       0.3278          1.0724        10.03
IVF-Binary-1024-nl316-np25-rf20-random (query)        36_436.79     1_566.52    38_003.31       0.4424          1.0438        10.03
IVF-Binary-1024-nl316-random (self)                   36_436.79     5_137.22    41_574.01       0.3370          1.0737        10.03
IVF-Binary-1024-nl158-np7-rf0-pca (query)             39_832.29     1_292.42    41_124.71       0.2750             NaN         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)            39_832.29     1_296.67    41_128.95       0.2697             NaN         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)            39_832.29     1_291.60    41_123.89       0.2686             NaN         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)            39_832.29     1_394.24    41_226.53       0.7313          1.0439         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)            39_832.29     1_495.84    41_328.13       0.8606          1.0201         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)           39_832.29     1_395.89    41_228.17       0.7142          1.0493         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)           39_832.29     1_501.83    41_334.12       0.8465          1.0219         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)           39_832.29     1_400.11    41_232.40       0.7106          1.0512         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)           39_832.29     1_540.33    41_372.62       0.8422          1.0228         9.57
IVF-Binary-1024-nl158-pca (self)                      39_832.29     4_590.66    44_422.94       0.7224          1.0510         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)            36_881.48     1_306.51    38_187.99       0.2707             NaN         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)            36_881.48     1_300.30    38_181.79       0.2695             NaN         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)            36_881.48     1_307.79    38_189.27       0.2688             NaN         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)           36_881.48     1_411.22    38_292.70       0.7177          1.0461         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)           36_881.48     1_513.33    38_394.81       0.8510          1.0205         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)           36_881.48     1_495.09    38_376.57       0.7136          1.0484         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)           36_881.48     1_537.58    38_419.06       0.8464          1.0214         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)           36_881.48     1_424.31    38_305.80       0.7105          1.0504         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)           36_881.48     1_540.51    38_421.99       0.8427          1.0223         9.76
IVF-Binary-1024-nl223-pca (self)                      36_881.48     4_633.86    41_515.34       0.7219          1.0498         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)            37_516.85     1_318.60    38_835.45       0.2701             NaN        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)            37_516.85     1_321.86    38_838.71       0.2697             NaN        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)            37_516.85     1_323.11    38_839.95       0.2687             NaN        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)           37_516.85     1_431.11    38_947.95       0.7158          1.0468        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)           37_516.85     1_534.61    39_051.46       0.8489          1.0207        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)           37_516.85     1_429.07    38_945.91       0.7144          1.0477        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)           37_516.85     1_539.49    39_056.33       0.8472          1.0212        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)           37_516.85     1_436.28    38_953.13       0.7110          1.0501        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)           37_516.85     1_559.32    39_076.16       0.8430          1.0222        10.04
IVF-Binary-1024-nl316-pca (self)                      37_516.85     4_708.08    42_224.92       0.7228          1.0491        10.04
IVF-Binary-768-nl158-np7-rf0-signed (query)           29_957.71       949.06    30_906.77       0.0969             NaN         7.29
IVF-Binary-768-nl158-np12-rf0-signed (query)          29_957.71       952.41    30_910.12       0.0966             NaN         7.29
IVF-Binary-768-nl158-np17-rf0-signed (query)          29_957.71       954.10    30_911.81       0.0963             NaN         7.29
IVF-Binary-768-nl158-np7-rf10-signed (query)          29_957.71     1_051.04    31_008.75       0.3036          1.0834         7.29
IVF-Binary-768-nl158-np7-rf20-signed (query)          29_957.71     1_146.55    31_104.26       0.4159          1.0509         7.29
IVF-Binary-768-nl158-np12-rf10-signed (query)         29_957.71     1_067.17    31_024.88       0.3036          1.0834         7.29
IVF-Binary-768-nl158-np12-rf20-signed (query)         29_957.71     1_155.87    31_113.58       0.4158          1.0509         7.29
IVF-Binary-768-nl158-np17-rf10-signed (query)         29_957.71     1_063.16    31_020.87       0.3036          1.0835         7.29
IVF-Binary-768-nl158-np17-rf20-signed (query)         29_957.71     1_161.76    31_119.47       0.4158          1.0509         7.29
IVF-Binary-768-nl158-signed (self)                    29_957.71     3_419.14    33_376.85       0.3148          1.0813         7.29
IVF-Binary-768-nl223-np11-rf0-signed (query)          27_291.13       967.24    28_258.37       0.1002             NaN         7.48
IVF-Binary-768-nl223-np14-rf0-signed (query)          27_291.13       989.35    28_280.48       0.0982             NaN         7.48
IVF-Binary-768-nl223-np21-rf0-signed (query)          27_291.13       974.21    28_265.34       0.0973             NaN         7.48
IVF-Binary-768-nl223-np11-rf10-signed (query)         27_291.13     1_073.26    28_364.39       0.3090          1.0811         7.48
IVF-Binary-768-nl223-np11-rf20-signed (query)         27_291.13     1_169.93    28_461.06       0.4206          1.0496         7.48
IVF-Binary-768-nl223-np14-rf10-signed (query)         27_291.13     1_073.09    28_364.22       0.3067          1.0822         7.48
IVF-Binary-768-nl223-np14-rf20-signed (query)         27_291.13     1_177.59    28_468.72       0.4185          1.0501         7.48
IVF-Binary-768-nl223-np21-rf10-signed (query)         27_291.13     1_085.11    28_376.24       0.3055          1.0829         7.48
IVF-Binary-768-nl223-np21-rf20-signed (query)         27_291.13     1_202.91    28_494.04       0.4174          1.0503         7.48
IVF-Binary-768-nl223-signed (self)                    27_291.13     3_482.98    30_774.11       0.3179          1.0801         7.48
IVF-Binary-768-nl316-np15-rf0-signed (query)          27_941.85     1_003.63    28_945.48       0.0986             NaN         7.76
IVF-Binary-768-nl316-np17-rf0-signed (query)          27_941.85     1_001.09    28_942.93       0.0978             NaN         7.76
IVF-Binary-768-nl316-np25-rf0-signed (query)          27_941.85     1_013.34    28_955.19       0.0974             NaN         7.76
IVF-Binary-768-nl316-np15-rf10-signed (query)         27_941.85     1_098.24    29_040.08       0.3090          1.0810         7.76
IVF-Binary-768-nl316-np15-rf20-signed (query)         27_941.85     1_205.19    29_147.04       0.4203          1.0495         7.76
IVF-Binary-768-nl316-np17-rf10-signed (query)         27_941.85     1_092.80    29_034.64       0.3076          1.0816         7.76
IVF-Binary-768-nl316-np17-rf20-signed (query)         27_941.85     1_197.20    29_139.05       0.4186          1.0499         7.76
IVF-Binary-768-nl316-np25-rf10-signed (query)         27_941.85     1_107.83    29_049.68       0.3066          1.0822         7.76
IVF-Binary-768-nl316-np25-rf20-signed (query)         27_941.85     1_203.16    29_145.01       0.4179          1.0501         7.76
IVF-Binary-768-nl316-signed (self)                    27_941.85     3_581.74    31_523.59       0.3184          1.0798         7.76
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

#### Quantisation (stress) data

<details>
<summary><b>Quantisation stress data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        10.44     4_113.93     4_124.36       1.0000          1.0000        48.83
Exhaustive (self)                                         10.44    13_800.48    13_810.91       1.0000          1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_578.17       277.60     2_855.78       0.2840             NaN         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_578.17       382.25     2_960.43       0.8630          1.0794         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_578.17       485.31     3_063.49       0.9501          1.0210         1.78
ExhaustiveBinary-256-random (self)                     2_578.17     1_279.44     3_857.61       0.8606          1.0824         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_798.65       278.23     3_076.88       0.1183             NaN         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_798.65       382.54     3_181.18       0.3215          1.9397         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_798.65       481.27     3_279.91       0.4181          1.5883         1.78
ExhaustiveBinary-256-pca (self)                        2_798.65     1_265.01     4_063.65       0.3192          1.9523         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_082.46       435.93     5_518.39       0.3036             NaN         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_082.46       545.55     5_628.00       0.9167          1.0510         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_082.46       651.63     5_734.08       0.9760          1.0116         3.55
ExhaustiveBinary-512-random (self)                     5_082.46     1_801.38     6_883.83       0.9147          1.0543         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_429.74       441.15     5_870.89       0.3665             NaN         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_429.74       562.61     5_992.34       0.8453          1.0657         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_429.74       657.16     6_086.89       0.9396          1.0206         3.55
ExhaustiveBinary-512-pca (self)                        5_429.74     1_860.32     7_290.06       0.8325          1.0737         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_082.09       755.52    10_837.61       0.3110             NaN         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_082.09       867.65    10_949.74       0.9411          1.0418         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_082.09       984.01    11_066.10       0.9840          1.0092         7.10
ExhaustiveBinary-1024-random (self)                   10_082.09     2_886.33    12_968.42       0.9397          1.0445         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_640.01       793.71    11_433.71       0.5577             NaN         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_640.01       887.15    11_527.16       0.9880          1.0028         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_640.01     1_009.51    11_649.51       0.9987          1.0003         7.10
ExhaustiveBinary-1024-pca (self)                      10_640.01     2_953.63    13_593.63       0.9861          1.0033         7.10
ExhaustiveBinary-256-signed_no_rr (query)              2_581.57       278.39     2_859.96       0.2840             NaN         1.78
ExhaustiveBinary-256-signed-rf10 (query)               2_581.57       384.11     2_965.69       0.8630          1.0794         1.78
ExhaustiveBinary-256-signed-rf20 (query)               2_581.57       484.39     3_065.96       0.9501          1.0210         1.78
ExhaustiveBinary-256-signed (self)                     2_581.57     1_262.07     3_843.65       0.8606          1.0824         1.78
IVF-Binary-256-nl158-np7-rf0-random (query)            4_031.39       117.21     4_148.60       0.3903             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           4_031.39       126.66     4_158.05       0.3491             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           4_031.39       135.94     4_167.33       0.3283             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           4_031.39       201.36     4_232.75       0.9569          1.0109         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           4_031.39       239.40     4_270.80       0.9891          1.0022         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          4_031.39       199.53     4_230.93       0.9376          1.0176         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          4_031.39       270.67     4_302.06       0.9853          1.0031         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          4_031.39       214.82     4_246.21       0.9218          1.0242         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          4_031.39       282.25     4_313.64       0.9801          1.0045         1.93
IVF-Binary-256-nl158-random (self)                     4_031.39       629.63     4_661.02       0.9363          1.0184         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           3_062.06       123.63     3_185.69       0.3792             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           3_062.06       125.42     3_187.48       0.3618             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           3_062.06       132.53     3_194.59       0.3375             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          3_062.06       186.75     3_248.81       0.9546          1.0115         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          3_062.06       245.04     3_307.10       0.9902          1.0019         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          3_062.06       192.91     3_254.97       0.9451          1.0152         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          3_062.06       252.06     3_314.12       0.9874          1.0026         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          3_062.06       207.12     3_269.18       0.9283          1.0221         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          3_062.06       278.05     3_340.11       0.9818          1.0041         2.00
IVF-Binary-256-nl223-random (self)                     3_062.06       603.10     3_665.16       0.9439          1.0157         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           3_252.90       128.36     3_381.25       0.3752             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           3_252.90       131.04     3_383.94       0.3667             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           3_252.90       136.93     3_389.83       0.3436             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          3_252.90       196.38     3_449.28       0.9542          1.0117         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          3_252.90       248.95     3_501.85       0.9906          1.0018         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          3_252.90       196.74     3_449.64       0.9494          1.0133         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          3_252.90       255.71     3_508.60       0.9893          1.0021         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          3_252.90       206.29     3_459.19       0.9345          1.0191         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          3_252.90       271.40     3_524.30       0.9845          1.0033         2.09
IVF-Binary-256-nl316-random (self)                     3_252.90       613.11     3_866.01       0.9485          1.0136         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               4_252.25       121.08     4_373.33       0.1525             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              4_252.25       129.87     4_382.12       0.1374             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              4_252.25       138.51     4_390.76       0.1309             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              4_252.25       189.08     4_441.33       0.4905          1.4115         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              4_252.25       247.48     4_499.73       0.6392          1.2170         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             4_252.25       203.40     4_455.64       0.4272          1.5368         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             4_252.25       271.63     4_523.87       0.5675          1.2973         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             4_252.25       218.58     4_470.83       0.3949          1.6196         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             4_252.25       290.95     4_543.20       0.5273          1.3520         1.93
IVF-Binary-256-nl158-pca (self)                        4_252.25       653.32     4_905.56       0.4247          1.5447         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              3_282.09       124.88     3_406.97       0.1483             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              3_282.09       129.09     3_411.18       0.1422             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              3_282.09       136.23     3_418.32       0.1340             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             3_282.09       193.09     3_475.18       0.4801          1.4243         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             3_282.09       251.07     3_533.16       0.6313          1.2223         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             3_282.09       198.02     3_480.11       0.4520          1.4780         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             3_282.09       262.46     3_544.55       0.6000          1.2556         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             3_282.09       213.75     3_495.84       0.4117          1.5721         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             3_282.09       281.23     3_563.32       0.5506          1.3174         2.00
IVF-Binary-256-nl223-pca (self)                        3_282.09       629.12     3_911.21       0.4496          1.4842         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              3_459.99       131.23     3_591.22       0.1481             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              3_459.99       133.45     3_593.44       0.1446             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              3_459.99       140.22     3_600.21       0.1363             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             3_459.99       203.08     3_663.07       0.4798          1.4233         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             3_459.99       258.82     3_718.81       0.6328          1.2201         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             3_459.99       202.74     3_662.73       0.4654          1.4494         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             3_459.99       262.95     3_722.94       0.6164          1.2368         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             3_459.99       211.98     3_671.98       0.4254          1.5346         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             3_459.99       278.95     3_738.95       0.5687          1.2916         2.09
IVF-Binary-256-nl316-pca (self)                        3_459.99       637.45     4_097.44       0.4628          1.4553         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_531.94       209.06     6_741.00       0.4173             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_531.94       223.98     6_755.92       0.3712             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_531.94       235.96     6_767.90       0.3492             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_531.94       278.44     6_810.38       0.9829          1.0037         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_531.94       332.54     6_864.48       0.9955          1.0009         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_531.94       294.87     6_826.81       0.9744          1.0059         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_531.94       360.80     6_892.74       0.9960          1.0007         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_531.94       316.23     6_848.17       0.9646          1.0091         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_531.94       387.84     6_919.78       0.9940          1.0011         3.71
IVF-Binary-512-nl158-random (self)                     6_531.94       969.11     7_501.05       0.9732          1.0067         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_562.99       217.32     5_780.31       0.4051             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_562.99       218.67     5_781.66       0.3861             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_562.99       230.69     5_793.68       0.3594             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_562.99       281.72     5_844.71       0.9827          1.0036         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_562.99       334.96     5_897.95       0.9971          1.0005         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_562.99       287.94     5_850.93       0.9777          1.0052         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_562.99       351.83     5_914.82       0.9964          1.0006         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_562.99       303.92     5_866.91       0.9678          1.0084         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_562.99       369.68     5_932.67       0.9944          1.0011         3.77
IVF-Binary-512-nl223-random (self)                     5_562.99       978.50     6_541.49       0.9768          1.0057         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           5_688.27       219.84     5_908.11       0.4019             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           5_688.27       221.51     5_909.79       0.3927             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           5_688.27       230.75     5_919.02       0.3669             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          5_688.27       287.65     5_975.93       0.9827          1.0036         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          5_688.27       348.12     6_036.40       0.9976          1.0004         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          5_688.27       288.93     5_977.20       0.9803          1.0043         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          5_688.27       347.30     6_035.57       0.9971          1.0005         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          5_688.27       301.01     5_989.29       0.9718          1.0070         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          5_688.27       365.40     6_053.67       0.9955          1.0008         3.86
IVF-Binary-512-nl316-random (self)                     5_688.27       918.34     6_606.62       0.9795          1.0048         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               6_857.89       216.83     7_074.72       0.3817             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              6_857.89       229.40     7_087.29       0.3729             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              6_857.89       241.75     7_099.65       0.3697             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              6_857.89       286.08     7_143.97       0.8864          1.0420         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              6_857.89       338.90     7_196.79       0.9653          1.0104         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             6_857.89       300.21     7_158.11       0.8656          1.0534         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             6_857.89       376.33     7_234.22       0.9543          1.0147         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             6_857.89       319.62     7_177.51       0.8563          1.0589         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             6_857.89       390.06     7_247.95       0.9479          1.0171         3.71
IVF-Binary-512-nl158-pca (self)                        6_857.89       969.08     7_826.97       0.8541          1.0600         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              5_836.00       220.96     6_056.96       0.3777             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              5_836.00       225.71     6_061.71       0.3741             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              5_836.00       236.22     6_072.22       0.3706             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             5_836.00       287.51     6_123.50       0.8810          1.0451         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             5_836.00       342.85     6_178.85       0.9631          1.0112         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             5_836.00       294.86     6_130.86       0.8710          1.0504         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             5_836.00       357.45     6_193.45       0.9578          1.0132         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             5_836.00       310.20     6_146.20       0.8589          1.0574         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             5_836.00       384.36     6_220.36       0.9499          1.0164         3.77
IVF-Binary-512-nl223-pca (self)                        5_836.00       942.14     6_778.14       0.8601          1.0566         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              6_006.49       226.51     6_233.00       0.3771             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              6_006.49       229.13     6_235.61       0.3755             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              6_006.49       238.91     6_245.39       0.3714             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             6_006.49       295.39     6_301.87       0.8796          1.0460         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             6_006.49       351.89     6_358.38       0.9626          1.0115         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             6_006.49       296.84     6_303.33       0.8748          1.0483         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             6_006.49       355.91     6_362.39       0.9602          1.0124         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             6_006.49       310.77     6_317.26       0.8625          1.0551         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             6_006.49       374.77     6_381.26       0.9524          1.0154         3.86
IVF-Binary-512-nl316-pca (self)                        6_006.49       946.39     6_952.88       0.8640          1.0545         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_354.14       395.22    11_749.36       0.4316             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_354.14       415.05    11_769.19       0.3816             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_354.14       435.32    11_789.46       0.3586             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_354.14       464.65    11_818.79       0.9908          1.0021         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_354.14       521.31    11_875.45       0.9967          1.0007         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_354.14       489.09    11_843.23       0.9868          1.0032         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_354.14       556.73    11_910.87       0.9983          1.0003         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_354.14       515.74    11_869.88       0.9808          1.0052         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_354.14       588.77    11_942.91       0.9973          1.0006         7.26
IVF-Binary-1024-nl158-random (self)                   11_354.14     1_621.38    12_975.52       0.9863          1.0039         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_431.50       401.18    10_832.68       0.4179             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_431.50       407.79    10_839.29       0.3976             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_431.50       423.42    10_854.92       0.3693             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_431.50       467.15    10_898.65       0.9911          1.0019         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_431.50       523.67    10_955.16       0.9984          1.0003         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_431.50       476.66    10_908.15       0.9885          1.0029         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_431.50       539.94    10_971.43       0.9984          1.0003         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_431.50       499.94    10_931.44       0.9821          1.0050         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_431.50       586.01    11_017.51       0.9974          1.0006         7.32
IVF-Binary-1024-nl223-random (self)                   10_431.50     1_583.23    12_014.72       0.9881          1.0033         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         10_556.62       406.13    10_962.75       0.4144             NaN         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)         10_556.62       410.00    10_966.62       0.4047             NaN         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)         10_556.62       434.50    10_991.12       0.3768             NaN         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)        10_556.62       472.80    11_029.42       0.9914          1.0019         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)        10_556.62       530.24    11_086.86       0.9988          1.0002         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)        10_556.62       478.68    11_035.30       0.9901          1.0024         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)        10_556.62       537.40    11_094.02       0.9987          1.0003         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)        10_556.62       495.89    11_052.51       0.9848          1.0041         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)        10_556.62       559.58    11_116.20       0.9980          1.0004         7.41
IVF-Binary-1024-nl316-random (self)                   10_556.62     1_568.12    12_124.74       0.9897          1.0027         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)             12_025.06       408.89    12_433.96       0.5685             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            12_025.06       430.21    12_455.28       0.5620             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            12_025.06       452.69    12_477.76       0.5600             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            12_025.06       477.36    12_502.42       0.9912          1.0019         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            12_025.06       532.67    12_557.74       0.9972          1.0006         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           12_025.06       501.01    12_526.07       0.9907          1.0020         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           12_025.06       578.84    12_603.90       0.9992          1.0002         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           12_025.06       529.62    12_554.68       0.9895          1.0023         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           12_025.06       599.43    12_624.49       0.9990          1.0002         7.26
IVF-Binary-1024-nl158-pca (self)                      12_025.06     1_635.97    13_661.03       0.9890          1.0025         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            10_990.14       413.91    11_404.06       0.5652             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            10_990.14       421.32    11_411.46       0.5627             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            10_990.14       442.64    11_432.78       0.5604             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           10_990.14       492.68    11_482.82       0.9916          1.0017         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           10_990.14       538.83    11_528.97       0.9987          1.0002         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           10_990.14       491.46    11_481.60       0.9909          1.0019         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           10_990.14       580.89    11_571.03       0.9991          1.0002         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           10_990.14       512.55    11_502.69       0.9896          1.0023         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           10_990.14       579.81    11_569.95       0.9990          1.0002         7.32
IVF-Binary-1024-nl223-pca (self)                      10_990.14     1_594.63    12_584.77       0.9892          1.0024         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            11_149.48       418.40    11_567.88       0.5647             NaN         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            11_149.48       423.37    11_572.85       0.5636             NaN         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            11_149.48       439.35    11_588.83       0.5612             NaN         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           11_149.48       487.94    11_637.41       0.9917          1.0018         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           11_149.48       544.16    11_693.63       0.9990          1.0002         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           11_149.48       489.61    11_639.09       0.9914          1.0018         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           11_149.48       549.36    11_698.84       0.9991          1.0002         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           11_149.48       507.36    11_656.84       0.9902          1.0022         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           11_149.48       572.91    11_722.38       0.9991          1.0002         7.42
IVF-Binary-1024-nl316-pca (self)                      11_149.48     1_587.97    12_737.45       0.9897          1.0023         7.42
IVF-Binary-256-nl158-np7-rf0-signed (query)            3_983.76       117.14     4_100.90       0.3903             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-signed (query)           3_983.76       126.31     4_110.07       0.3491             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-signed (query)           3_983.76       136.47     4_120.23       0.3283             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-signed (query)           3_983.76       185.75     4_169.51       0.9569          1.0109         1.93
IVF-Binary-256-nl158-np7-rf20-signed (query)           3_983.76       243.34     4_227.10       0.9891          1.0022         1.93
IVF-Binary-256-nl158-np12-rf10-signed (query)          3_983.76       198.34     4_182.10       0.9376          1.0176         1.93
IVF-Binary-256-nl158-np12-rf20-signed (query)          3_983.76       261.08     4_244.84       0.9853          1.0031         1.93
IVF-Binary-256-nl158-np17-rf10-signed (query)          3_983.76       213.40     4_197.16       0.9218          1.0242         1.93
IVF-Binary-256-nl158-np17-rf20-signed (query)          3_983.76       281.59     4_265.35       0.9801          1.0045         1.93
IVF-Binary-256-nl158-signed (self)                     3_983.76       626.76     4_610.52       0.9363          1.0184         1.93
IVF-Binary-256-nl223-np11-rf0-signed (query)           3_034.03       121.29     3_155.32       0.3792             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-signed (query)           3_034.03       125.86     3_159.89       0.3618             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-signed (query)           3_034.03       133.21     3_167.24       0.3375             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-signed (query)          3_034.03       186.74     3_220.76       0.9546          1.0115         2.00
IVF-Binary-256-nl223-np11-rf20-signed (query)          3_034.03       250.07     3_284.10       0.9902          1.0019         2.00
IVF-Binary-256-nl223-np14-rf10-signed (query)          3_034.03       193.16     3_227.18       0.9451          1.0152         2.00
IVF-Binary-256-nl223-np14-rf20-signed (query)          3_034.03       251.91     3_285.94       0.9874          1.0026         2.00
IVF-Binary-256-nl223-np21-rf10-signed (query)          3_034.03       206.40     3_240.43       0.9283          1.0221         2.00
IVF-Binary-256-nl223-np21-rf20-signed (query)          3_034.03       269.79     3_303.82       0.9818          1.0041         2.00
IVF-Binary-256-nl223-signed (self)                     3_034.03       602.51     3_636.54       0.9439          1.0157         2.00
IVF-Binary-256-nl316-np15-rf0-signed (query)           3_194.09       128.21     3_322.31       0.3752             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-signed (query)           3_194.09       129.68     3_323.77       0.3667             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-signed (query)           3_194.09       136.04     3_330.13       0.3436             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-signed (query)          3_194.09       195.75     3_389.84       0.9542          1.0117         2.09
IVF-Binary-256-nl316-np15-rf20-signed (query)          3_194.09       249.47     3_443.56       0.9906          1.0018         2.09
IVF-Binary-256-nl316-np17-rf10-signed (query)          3_194.09       196.12     3_390.22       0.9494          1.0133         2.09
IVF-Binary-256-nl316-np17-rf20-signed (query)          3_194.09       254.41     3_448.51       0.9893          1.0021         2.09
IVF-Binary-256-nl316-np25-rf10-signed (query)          3_194.09       206.58     3_400.67       0.9345          1.0191         2.09
IVF-Binary-256-nl316-np25-rf20-signed (query)          3_194.09       269.22     3_463.31       0.9845          1.0033         2.09
IVF-Binary-256-nl316-signed (self)                     3_194.09       609.50     3_803.59       0.9485          1.0136         2.09
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Quantisation stress data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        21.21     9_535.19     9_556.41       1.0000          1.0000        97.66
Exhaustive (self)                                         21.21    32_061.85    32_083.06       1.0000          1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_640.13       378.55     6_018.68       0.2776             NaN         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_640.13       518.61     6_158.74       0.8648          1.0698         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_640.13       640.75     6_280.89       0.9562          1.0171         2.03
ExhaustiveBinary-256-random (self)                     5_640.13     1_686.35     7_326.48       0.8664          1.0698         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_166.29       385.62     6_551.91       0.1212             NaN         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_166.29       546.73     6_713.01       0.3407          1.8751         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_166.29       641.91     6_808.20       0.4406          1.5429         2.03
ExhaustiveBinary-256-pca (self)                        6_166.29     1_694.93     7_861.22       0.3366          1.8907         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_229.87       644.06    11_873.93       0.2965             NaN         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_229.87       776.80    12_006.67       0.9188          1.0453         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_229.87       918.92    12_148.79       0.9788          1.0097         4.05
ExhaustiveBinary-512-random (self)                    11_229.87     2_570.79    13_800.66       0.9199          1.0452         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_763.37       658.07    12_421.44       0.1147             NaN         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_763.37       783.50    12_546.87       0.2782          2.2254         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_763.37       909.87    12_673.24       0.3475          1.8252         4.05
ExhaustiveBinary-512-pca (self)                       11_763.37     2_591.69    14_355.06       0.2742          2.2528         4.05
ExhaustiveBinary-1024-random_no_rr (query)            21_855.77     1_174.43    23_030.21       0.3064             NaN         8.10
ExhaustiveBinary-1024-random-rf10 (query)             21_855.77     1_320.02    23_175.79       0.9430          1.0366         8.10
ExhaustiveBinary-1024-random-rf20 (query)             21_855.77     1_472.34    23_328.11       0.9861          1.0075         8.10
ExhaustiveBinary-1024-random (self)                   21_855.77     4_382.62    26_238.39       0.9439          1.0367         8.10
ExhaustiveBinary-1024-pca_no_rr (query)               23_007.58     1_200.23    24_207.81       0.3939             NaN         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                23_007.58     1_347.20    24_354.78       0.8323          1.0743         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                23_007.58     1_474.59    24_482.17       0.9198          1.0285         8.11
ExhaustiveBinary-1024-pca (self)                      23_007.58     4_458.59    27_466.17       0.8160          1.0854         8.11
ExhaustiveBinary-512-signed_no_rr (query)             11_034.48       643.94    11_678.43       0.2965             NaN         4.05
ExhaustiveBinary-512-signed-rf10 (query)              11_034.48       775.37    11_809.85       0.9188          1.0453         4.05
ExhaustiveBinary-512-signed-rf20 (query)              11_034.48       909.86    11_944.35       0.9788          1.0097         4.05
ExhaustiveBinary-512-signed (self)                    11_034.48     2_569.90    13_604.39       0.9199          1.0452         4.05
IVF-Binary-256-nl158-np7-rf0-random (query)            8_616.56       232.15     8_848.71       0.3651             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           8_616.56       241.55     8_858.12       0.3326             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           8_616.56       252.54     8_869.10       0.3138             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           8_616.56       321.24     8_937.80       0.9498          1.0139         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           8_616.56       401.69     9_018.25       0.9902          1.0021         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          8_616.56       335.71     8_952.27       0.9293          1.0223         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          8_616.56       426.52     9_043.08       0.9852          1.0034         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          8_616.56       352.05     8_968.61       0.9138          1.0290         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          8_616.56       446.72     9_063.28       0.9797          1.0049         2.34
IVF-Binary-256-nl158-random (self)                     8_616.56     1_053.93     9_670.49       0.9307          1.0217         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           6_336.92       243.35     6_580.28       0.3536             NaN         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)           6_336.92       245.51     6_582.44       0.3401             NaN         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)           6_336.92       253.34     6_590.26       0.3194             NaN         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)          6_336.92       332.49     6_669.41       0.9466          1.0146         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)          6_336.92       412.39     6_749.31       0.9909          1.0019         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)          6_336.92       340.07     6_676.99       0.9380          1.0180         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)          6_336.92       425.56     6_762.49       0.9884          1.0025         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)          6_336.92       352.15     6_689.07       0.9208          1.0251         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)          6_336.92       443.47     6_780.39       0.9826          1.0040         2.46
IVF-Binary-256-nl223-random (self)                     6_336.92     1_056.61     7_393.54       0.9387          1.0178         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)           6_624.88       254.87     6_879.76       0.3480             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           6_624.88       259.33     6_884.22       0.3420             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           6_624.88       263.37     6_888.25       0.3246             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          6_624.88       349.20     6_974.08       0.9452          1.0151         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          6_624.88       430.62     7_055.50       0.9909          1.0019         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          6_624.88       349.75     6_974.64       0.9406          1.0168         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          6_624.88       433.80     7_058.69       0.9896          1.0022         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          6_624.88       364.29     6_989.18       0.9256          1.0228         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          6_624.88       450.91     7_075.80       0.9852          1.0033         2.65
IVF-Binary-256-nl316-random (self)                     6_624.88     1_086.99     7_711.88       0.9420          1.0162         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               9_141.41       238.27     9_379.68       0.1446             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              9_141.41       248.79     9_390.20       0.1353             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              9_141.41       257.26     9_398.68       0.1305             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              9_141.41       327.28     9_468.69       0.4660          1.4656         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              9_141.41       412.44     9_553.85       0.6170          1.2408         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             9_141.41       340.17     9_481.58       0.4250          1.5540         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             9_141.41       433.94     9_575.35       0.5648          1.3027         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             9_141.41       355.36     9_496.77       0.4032          1.6107         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             9_141.41       453.35     9_594.76       0.5371          1.3415         2.34
IVF-Binary-256-nl158-pca (self)                        9_141.41     1_076.87    10_218.28       0.4221          1.5634         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              6_858.93       255.98     7_114.91       0.1417             NaN         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              6_858.93       252.21     7_111.13       0.1375             NaN         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              6_858.93       259.92     7_118.85       0.1316             NaN         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             6_858.93       338.28     7_197.21       0.4567          1.4780         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             6_858.93       419.72     7_278.65       0.6069          1.2482         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             6_858.93       342.78     7_201.71       0.4388          1.5156         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             6_858.93       430.81     7_289.73       0.5849          1.2736         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             6_858.93       356.51     7_215.44       0.4115          1.5826         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             6_858.93       468.14     7_327.07       0.5502          1.3193         2.47
IVF-Binary-256-nl223-pca (self)                        6_858.93     1_077.46     7_936.39       0.4359          1.5231         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              7_132.17       267.40     7_399.57       0.1403             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              7_132.17       263.44     7_395.61       0.1382             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              7_132.17       281.53     7_413.70       0.1334             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             7_132.17       351.71     7_483.88       0.4552          1.4775         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             7_132.17       435.06     7_567.24       0.6068          1.2470         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             7_132.17       352.70     7_484.87       0.4459          1.4972         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             7_132.17       440.91     7_573.08       0.5952          1.2602         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             7_132.17       362.94     7_495.11       0.4205          1.5566         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             7_132.17       455.47     7_587.64       0.5622          1.3014         2.65
IVF-Binary-256-nl316-pca (self)                        7_132.17     1_109.45     8_241.62       0.4434          1.5036         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           14_150.17       426.77    14_576.95       0.3916             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          14_150.17       441.71    14_591.88       0.3554             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          14_150.17       454.32    14_604.49       0.3353             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          14_150.17       517.25    14_667.42       0.9791          1.0056         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          14_150.17       596.40    14_746.57       0.9960          1.0009         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         14_150.17       553.91    14_704.09       0.9680          1.0098         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         14_150.17       625.45    14_775.62       0.9953          1.0013         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         14_150.17       567.53    14_717.70       0.9578          1.0135         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         14_150.17       655.65    14_805.83       0.9933          1.0018         4.36
IVF-Binary-512-nl158-random (self)                    14_150.17     1_726.04    15_876.21       0.9688          1.0094         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          11_833.25       435.67    12_268.92       0.3780             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          11_833.25       443.21    12_276.46       0.3640             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          11_833.25       457.13    12_290.38       0.3413             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         11_833.25       527.96    12_361.22       0.9782          1.0057         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         11_833.25       606.63    12_439.89       0.9972          1.0006         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         11_833.25       534.96    12_368.21       0.9733          1.0074         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         11_833.25       619.83    12_453.08       0.9965          1.0008         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         11_833.25       555.39    12_388.64       0.9621          1.0114         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         11_833.25       653.79    12_487.04       0.9943          1.0014         4.49
IVF-Binary-512-nl223-random (self)                    11_833.25     1_725.94    13_559.20       0.9737          1.0073         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          12_098.94       451.44    12_550.37       0.3736             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          12_098.94       482.94    12_581.88       0.3669             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          12_098.94       468.03    12_566.97       0.3475             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         12_098.94       543.62    12_642.55       0.9781          1.0056         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         12_098.94       623.26    12_722.19       0.9976          1.0005         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         12_098.94       543.08    12_642.01       0.9756          1.0066         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         12_098.94       628.33    12_727.27       0.9971          1.0006         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         12_098.94       557.64    12_656.57       0.9663          1.0098         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         12_098.94       649.18    12_748.12       0.9953          1.0011         4.67
IVF-Binary-512-nl316-random (self)                    12_098.94     1_735.40    13_834.33       0.9761          1.0062         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              14_770.18       438.78    15_208.96       0.1402             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             14_770.18       451.53    15_221.71       0.1307             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             14_770.18       464.05    15_234.23       0.1257             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             14_770.18       532.03    15_302.21       0.4248          1.5450         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             14_770.18       613.08    15_383.26       0.5632          1.3025         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            14_770.18       543.96    15_314.14       0.3820          1.6569         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            14_770.18       640.48    15_410.66       0.5038          1.3865         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            14_770.18       564.77    15_334.95       0.3596          1.7323         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            14_770.18       660.01    15_430.19       0.4725          1.4419         4.36
IVF-Binary-512-nl158-pca (self)                       14_770.18     1_770.20    16_540.38       0.3781          1.6708         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             12_487.82       449.20    12_937.02       0.1371             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             12_487.82       455.09    12_942.91       0.1332             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             12_487.82       464.44    12_952.26       0.1272             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            12_487.82       538.30    13_026.12       0.4162          1.5553         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            12_487.82       622.25    13_110.07       0.5534          1.3091         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            12_487.82       543.37    13_031.19       0.3973          1.6033         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            12_487.82       633.65    13_121.46       0.5274          1.3450         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            12_487.82       561.15    13_048.97       0.3699          1.6879         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            12_487.82       654.32    13_142.14       0.4885          1.4076         4.49
IVF-Binary-512-nl223-pca (self)                       12_487.82     1_779.60    14_267.42       0.3936          1.6150         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             12_792.94       463.38    13_256.32       0.1360             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             12_792.94       465.94    13_258.89       0.1339             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             12_792.94       474.67    13_267.61       0.1287             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            12_792.94       552.96    13_345.90       0.4151          1.5550         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            12_792.94       635.91    13_428.85       0.5522          1.3080         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            12_792.94       554.76    13_347.71       0.4052          1.5795         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            12_792.94       640.36    13_433.30       0.5383          1.3265         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            12_792.94       571.77    13_364.71       0.3790          1.6551         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            12_792.94       659.80    13_452.74       0.5015          1.3834         4.67
IVF-Binary-512-nl316-pca (self)                       12_792.94     1_790.01    14_582.95       0.4011          1.5903         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          24_979.56       822.92    25_802.48       0.4047             NaN         8.41
IVF-Binary-1024-nl158-np12-rf0-random (query)         24_979.56       867.30    25_846.85       0.3672             NaN         8.41
IVF-Binary-1024-nl158-np17-rf0-random (query)         24_979.56       862.79    25_842.35       0.3461             NaN         8.41
IVF-Binary-1024-nl158-np7-rf10-random (query)         24_979.56       932.67    25_912.23       0.9890          1.0035         8.41
IVF-Binary-1024-nl158-np7-rf20-random (query)         24_979.56       989.69    25_969.24       0.9971          1.0007         8.41
IVF-Binary-1024-nl158-np12-rf10-random (query)        24_979.56       935.90    25_915.46       0.9827          1.0060         8.41
IVF-Binary-1024-nl158-np12-rf20-random (query)        24_979.56     1_024.54    26_004.10       0.9973          1.0009         8.41
IVF-Binary-1024-nl158-np17-rf10-random (query)        24_979.56       963.98    25_943.54       0.9758          1.0087         8.41
IVF-Binary-1024-nl158-np17-rf20-random (query)        24_979.56     1_059.95    26_039.51       0.9962          1.0013         8.41
IVF-Binary-1024-nl158-random (self)                   24_979.56     3_059.30    28_038.86       0.9830          1.0059         8.41
IVF-Binary-1024-nl223-np11-rf0-random (query)         22_672.10       836.60    23_508.69       0.3911             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         22_672.10       841.77    23_513.87       0.3763             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         22_672.10       863.04    23_535.14       0.3520             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        22_672.10       923.80    23_595.89       0.9890          1.0033         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        22_672.10     1_004.98    23_677.07       0.9984          1.0004         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        22_672.10       932.63    23_604.73       0.9860          1.0044         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        22_672.10     1_019.62    23_691.71       0.9980          1.0005         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        22_672.10       962.39    23_634.49       0.9785          1.0073         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        22_672.10     1_054.70    23_726.79       0.9969          1.0010         8.54
IVF-Binary-1024-nl223-random (self)                   22_672.10     3_049.47    25_721.57       0.9861          1.0045         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         22_984.45       849.36    23_833.82       0.3862             NaN         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)         22_984.45       861.01    23_845.46       0.3791             NaN         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)         22_984.45       869.59    23_854.04       0.3587             NaN         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)        22_984.45       934.23    23_918.69       0.9890          1.0033         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)        22_984.45     1_018.14    24_002.60       0.9986          1.0004         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)        22_984.45       942.30    23_926.76       0.9874          1.0039         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)        22_984.45     1_026.99    24_011.44       0.9983          1.0005         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)        22_984.45       972.16    23_956.62       0.9815          1.0062         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)        22_984.45     1_052.77    24_037.22       0.9974          1.0008         8.72
IVF-Binary-1024-nl316-random (self)                   22_984.45     3_079.20    26_063.65       0.9877          1.0038         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)             26_103.34       848.08    26_951.43       0.4021             NaN         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            26_103.34       879.07    26_982.41       0.3977             NaN         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            26_103.34       910.61    27_013.96       0.3959             NaN         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            26_103.34       929.15    27_032.49       0.8540          1.0604         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            26_103.34     1_012.08    27_115.42       0.9425          1.0187         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           26_103.34       954.17    27_057.52       0.8431          1.0671         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           26_103.34     1_049.17    27_152.52       0.9317          1.0230         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           26_103.34       981.90    27_085.25       0.8385          1.0701         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           26_103.34     1_084.24    27_187.58       0.9264          1.0254         8.42
IVF-Binary-1024-nl158-pca (self)                      26_103.34     3_123.68    29_227.02       0.8277          1.0772         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            23_804.18       856.58    24_660.76       0.3994             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            23_804.18       866.19    24_670.36       0.3980             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            23_804.18       883.86    24_688.04       0.3960             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           23_804.18       939.91    24_744.09       0.8497          1.0632         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           23_804.18     1_022.56    24_826.73       0.9391          1.0200         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           23_804.18       952.11    24_756.29       0.8454          1.0657         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           23_804.18     1_037.66    24_841.84       0.9342          1.0220         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           23_804.18       974.06    24_778.24       0.8393          1.0697         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           23_804.18     1_067.94    24_872.12       0.9272          1.0251         8.54
IVF-Binary-1024-nl223-pca (self)                      23_804.18     3_105.96    26_910.14       0.8300          1.0757         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            24_058.79       872.72    24_931.50       0.3988             NaN         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            24_058.79       874.14    24_932.93       0.3979             NaN         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            24_058.79       888.16    24_946.94       0.3963             NaN         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           24_058.79       955.37    25_014.16       0.8493          1.0634         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           24_058.79     1_231.07    25_289.86       0.9381          1.0205         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           24_058.79     1_049.48    25_108.27       0.8470          1.0648         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           24_058.79     1_149.85    25_208.64       0.9355          1.0215         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           24_058.79     1_121.75    25_180.54       0.8409          1.0685         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           24_058.79     1_158.84    25_217.63       0.9293          1.0242         8.73
IVF-Binary-1024-nl316-pca (self)                      24_058.79     3_469.96    27_528.74       0.8315          1.0747         8.73
IVF-Binary-512-nl158-np7-rf0-signed (query)           14_510.35       429.11    14_939.46       0.3916             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-signed (query)          14_510.35       442.84    14_953.19       0.3554             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-signed (query)          14_510.35       456.83    14_967.18       0.3353             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-signed (query)          14_510.35       519.62    15_029.97       0.9791          1.0056         4.36
IVF-Binary-512-nl158-np7-rf20-signed (query)          14_510.35       620.83    15_131.18       0.9960          1.0009         4.36
IVF-Binary-512-nl158-np12-rf10-signed (query)         14_510.35       537.46    15_047.81       0.9680          1.0098         4.36
IVF-Binary-512-nl158-np12-rf20-signed (query)         14_510.35       626.63    15_136.98       0.9953          1.0013         4.36
IVF-Binary-512-nl158-np17-rf10-signed (query)         14_510.35       559.11    15_069.46       0.9578          1.0135         4.36
IVF-Binary-512-nl158-np17-rf20-signed (query)         14_510.35       650.08    15_160.42       0.9933          1.0018         4.36
IVF-Binary-512-nl158-signed (self)                    14_510.35     1_730.86    16_241.20       0.9688          1.0094         4.36
IVF-Binary-512-nl223-np11-rf0-signed (query)          11_838.38       437.40    12_275.77       0.3780             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-signed (query)          11_838.38       443.68    12_282.06       0.3640             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-signed (query)          11_838.38       456.20    12_294.58       0.3413             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-signed (query)         11_838.38       525.30    12_363.68       0.9782          1.0057         4.49
IVF-Binary-512-nl223-np11-rf20-signed (query)         11_838.38       607.01    12_445.39       0.9972          1.0006         4.49
IVF-Binary-512-nl223-np14-rf10-signed (query)         11_838.38       533.72    12_372.10       0.9733          1.0074         4.49
IVF-Binary-512-nl223-np14-rf20-signed (query)         11_838.38       619.24    12_457.62       0.9965          1.0008         4.49
IVF-Binary-512-nl223-np21-rf10-signed (query)         11_838.38       551.98    12_390.36       0.9621          1.0114         4.49
IVF-Binary-512-nl223-np21-rf20-signed (query)         11_838.38       652.45    12_490.82       0.9943          1.0014         4.49
IVF-Binary-512-nl223-signed (self)                    11_838.38     1_706.10    13_544.47       0.9737          1.0073         4.49
IVF-Binary-512-nl316-np15-rf0-signed (query)          12_087.93       450.92    12_538.84       0.3736             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-signed (query)          12_087.93       474.55    12_562.48       0.3669             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-signed (query)          12_087.93       462.77    12_550.69       0.3475             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-signed (query)         12_087.93       542.00    12_629.93       0.9781          1.0056         4.67
IVF-Binary-512-nl316-np15-rf20-signed (query)         12_087.93       625.52    12_713.45       0.9976          1.0005         4.67
IVF-Binary-512-nl316-np17-rf10-signed (query)         12_087.93       545.41    12_633.34       0.9756          1.0066         4.67
IVF-Binary-512-nl316-np17-rf20-signed (query)         12_087.93       630.79    12_718.72       0.9971          1.0006         4.67
IVF-Binary-512-nl316-np25-rf10-signed (query)         12_087.93       562.95    12_650.88       0.9663          1.0098         4.67
IVF-Binary-512-nl316-np25-rf20-signed (query)         12_087.93       662.64    12_750.57       0.9953          1.0011         4.67
IVF-Binary-512-nl316-signed (self)                    12_087.93     1_744.63    13_832.55       0.9761          1.0062         4.67
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Quantisation stress data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        31.20    15_664.29    15_695.49       1.0000          1.0000       146.48
Exhaustive (self)                                         31.20    52_229.24    52_260.44       1.0000          1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)              8_777.63       487.57     9_265.20       0.2669             NaN         2.28
ExhaustiveBinary-256-random-rf10 (query)               8_777.63       648.19     9_425.81       0.8498          1.0802         2.28
ExhaustiveBinary-256-random-rf20 (query)               8_777.63       811.09     9_588.72       0.9492          1.0191         2.28
ExhaustiveBinary-256-random (self)                     8_777.63     2_110.04    10_887.67       0.8514          1.0784         2.28
ExhaustiveBinary-256-pca_no_rr (query)                 9_424.47       496.48     9_920.95       0.1281             NaN         2.28
ExhaustiveBinary-256-pca-rf10 (query)                  9_424.47       641.77    10_066.24       0.3750          1.7664         2.28
ExhaustiveBinary-256-pca-rf20 (query)                  9_424.47       789.57    10_214.04       0.5003          1.4421         2.28
ExhaustiveBinary-256-pca (self)                        9_424.47     2_125.78    11_550.25       0.3725          1.7770         2.28
ExhaustiveBinary-512-random_no_rr (query)             17_222.74       857.88    18_080.62       0.2888             NaN         4.55
ExhaustiveBinary-512-random-rf10 (query)              17_222.74     1_013.87    18_236.61       0.9099          1.0520         4.55
ExhaustiveBinary-512-random-rf20 (query)              17_222.74     1_166.21    18_388.94       0.9755          1.0108         4.55
ExhaustiveBinary-512-random (self)                    17_222.74     3_365.20    20_587.94       0.9111          1.0500         4.55
ExhaustiveBinary-512-pca_no_rr (query)                18_072.42       870.51    18_942.93       0.1131             NaN         4.55
ExhaustiveBinary-512-pca-rf10 (query)                 18_072.42     1_027.47    19_099.89       0.3166          2.0536         4.55
ExhaustiveBinary-512-pca-rf20 (query)                 18_072.42     1_181.96    19_254.38       0.4179          1.6492         4.55
ExhaustiveBinary-512-pca (self)                       18_072.42     3_457.32    21_529.74       0.3146          2.0599         4.55
ExhaustiveBinary-1024-random_no_rr (query)            34_343.93     1_619.04    35_962.97       0.2977             NaN         9.10
ExhaustiveBinary-1024-random-rf10 (query)             34_343.93     1_774.37    36_118.30       0.9384          1.0420         9.10
ExhaustiveBinary-1024-random-rf20 (query)             34_343.93     1_934.38    36_278.31       0.9843          1.0081         9.10
ExhaustiveBinary-1024-random (self)                   34_343.93     5_865.56    40_209.49       0.9399          1.0400         9.10
ExhaustiveBinary-1024-pca_no_rr (query)               35_508.69     1_635.49    37_144.18       0.2379             NaN         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                35_508.69     1_787.45    37_296.15       0.6180          1.2507         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                35_508.69     1_958.21    37_466.90       0.7452          1.1303         9.11
ExhaustiveBinary-1024-pca (self)                      35_508.69     5_955.87    41_464.56       0.6017          1.2739         9.11
ExhaustiveBinary-768-signed_no_rr (query)             25_839.60     1_246.95    27_086.55       0.2944             NaN         6.83
ExhaustiveBinary-768-signed-rf10 (query)              25_839.60     1_400.88    27_240.48       0.9283          1.0452         6.83
ExhaustiveBinary-768-signed-rf20 (query)              25_839.60     1_565.08    27_404.68       0.9818          1.0090         6.83
ExhaustiveBinary-768-signed (self)                    25_839.60     4_645.85    30_485.45       0.9296          1.0433         6.83
IVF-Binary-256-nl158-np7-rf0-random (query)           13_340.72       351.39    13_692.11       0.3403             NaN         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)          13_340.72       376.79    13_717.51       0.3122             NaN         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)          13_340.72       370.01    13_710.73       0.2964             NaN         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)          13_340.72       464.18    13_804.90       0.9318          1.0212         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)          13_340.72       564.00    13_904.72       0.9864          1.0032         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)         13_340.72       478.27    13_818.99       0.9084          1.0317         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)         13_340.72       587.60    13_928.32       0.9790          1.0053         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)         13_340.72       492.89    13_833.61       0.8929          1.0399         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)         13_340.72       617.59    13_958.31       0.9728          1.0072         2.74
IVF-Binary-256-nl158-random (self)                    13_340.72     1_512.30    14_853.02       0.9092          1.0309         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)           9_789.03       366.45    10_155.48       0.3314             NaN         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)           9_789.03       371.52    10_160.55       0.3198             NaN         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)           9_789.03       379.60    10_168.63       0.3010             NaN         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)          9_789.03       479.81    10_268.84       0.9276          1.0225         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)          9_789.03       578.22    10_367.26       0.9861          1.0033         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)          9_789.03       485.21    10_274.24       0.9184          1.0266         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)          9_789.03       593.43    10_382.46       0.9826          1.0042         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)          9_789.03       510.20    10_299.23       0.8996          1.0358         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)          9_789.03       612.61    10_401.65       0.9757          1.0061         2.93
IVF-Binary-256-nl223-random (self)                     9_789.03     1_532.04    11_321.07       0.9190          1.0260         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)          10_146.45       389.96    10_536.41       0.3286             NaN         3.20
IVF-Binary-256-nl316-np17-rf0-random (query)          10_146.45       394.02    10_540.47       0.3234             NaN         3.20
IVF-Binary-256-nl316-np25-rf0-random (query)          10_146.45       399.09    10_545.54       0.3077             NaN         3.20
IVF-Binary-256-nl316-np15-rf10-random (query)         10_146.45       504.25    10_650.70       0.9284          1.0213         3.20
IVF-Binary-256-nl316-np15-rf20-random (query)         10_146.45       616.34    10_762.79       0.9866          1.0029         3.20
IVF-Binary-256-nl316-np17-rf10-random (query)         10_146.45       509.70    10_656.15       0.9235          1.0235         3.20
IVF-Binary-256-nl316-np17-rf20-random (query)         10_146.45       620.62    10_767.07       0.9849          1.0034         3.20
IVF-Binary-256-nl316-np25-rf10-random (query)         10_146.45       527.66    10_674.11       0.9073          1.0311         3.20
IVF-Binary-256-nl316-np25-rf20-random (query)         10_146.45       626.28    10_772.73       0.9792          1.0050         3.20
IVF-Binary-256-nl316-random (self)                    10_146.45     1_613.71    11_760.16       0.9242          1.0230         3.20
IVF-Binary-256-nl158-np7-rf0-pca (query)              14_146.99       358.48    14_505.47       0.1454             NaN         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)             14_146.99       368.89    14_515.89       0.1382             NaN         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)             14_146.99       376.53    14_523.53       0.1346             NaN         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)             14_146.99       469.57    14_616.57       0.4662          1.4677         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)             14_146.99       570.48    14_717.47       0.6300          1.2294         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)            14_146.99       482.79    14_629.78       0.4327          1.5393         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)            14_146.99       593.85    14_740.84       0.5877          1.2766         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)            14_146.99       498.62    14_645.62       0.4150          1.5829         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)            14_146.99       615.35    14_762.34       0.5655          1.3049         2.74
IVF-Binary-256-nl158-pca (self)                       14_146.99     1_539.86    15_686.86       0.4310          1.5438         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)             10_517.85       373.52    10_891.37       0.1435             NaN         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)             10_517.85       377.52    10_895.37       0.1404             NaN         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)             10_517.85       387.09    10_904.94       0.1360             NaN         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)            10_517.85       487.92    11_005.77       0.4603          1.4717         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)            10_517.85       589.76    11_107.61       0.6244          1.2313         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)            10_517.85       490.78    11_008.63       0.4461          1.5021         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)            10_517.85       598.27    11_116.12       0.6061          1.2515         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)            10_517.85       509.79    11_027.64       0.4237          1.5567         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)            10_517.85       618.13    11_135.98       0.5772          1.2875         2.93
IVF-Binary-256-nl223-pca (self)                       10_517.85     1_559.14    12_076.98       0.4440          1.5077         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)             10_914.68       395.93    11_310.62       0.1432             NaN         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)             10_914.68       400.36    11_315.04       0.1417             NaN         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)             10_914.68       415.01    11_329.70       0.1377             NaN         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)            10_914.68       510.33    11_425.01       0.4614          1.4682         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)            10_914.68       614.68    11_529.36       0.6269          1.2277         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)            10_914.68       510.27    11_424.96       0.4537          1.4846         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)            10_914.68       618.47    11_533.15       0.6166          1.2387         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)            10_914.68       521.68    11_436.36       0.4324          1.5326         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)            10_914.68       634.09    11_548.78       0.5891          1.2713         3.21
IVF-Binary-256-nl316-pca (self)                       10_914.68     1_618.62    12_533.31       0.4518          1.4886         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)           21_922.67       653.12    22_575.80       0.3662             NaN         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)          21_922.67       668.49    22_591.16       0.3358             NaN         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)          21_922.67       676.33    22_599.01       0.3196             NaN         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)          21_922.67       766.13    22_688.80       0.9712          1.0087         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)          21_922.67       864.69    22_787.37       0.9956          1.0012         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)         21_922.67       783.98    22_706.66       0.9565          1.0148         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)         21_922.67       914.55    22_837.23       0.9931          1.0019         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)         21_922.67       807.16    22_729.83       0.9452          1.0202         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)         21_922.67       919.67    22_842.35       0.9902          1.0028         5.02
IVF-Binary-512-nl158-random (self)                    21_922.67     2_544.91    24_467.59       0.9570          1.0139         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)          18_384.45       669.91    19_054.36       0.3580             NaN         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)          18_384.45       676.40    19_060.85       0.3461             NaN         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)          18_384.45       686.48    19_070.93       0.3254             NaN         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)         18_384.45       780.45    19_164.90       0.9695          1.0093         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)         18_384.45       886.19    19_270.64       0.9958          1.0011         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)         18_384.45       786.99    19_171.44       0.9635          1.0117         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)         18_384.45       892.23    19_276.67       0.9947          1.0015         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)         18_384.45       805.73    19_190.18       0.9506          1.0173         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)         18_384.45       937.68    19_322.13       0.9918          1.0022         5.21
IVF-Binary-512-nl223-random (self)                    18_384.45     2_549.07    20_933.52       0.9638          1.0111         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)          18_723.99       691.54    19_415.53       0.3548             NaN         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)          18_723.99       696.68    19_420.67       0.3490             NaN         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)          18_723.99       704.04    19_428.03       0.3318             NaN         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)         18_723.99       809.57    19_533.56       0.9699          1.0086         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)         18_723.99       903.16    19_627.15       0.9963          1.0008         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)         18_723.99       805.52    19_529.51       0.9667          1.0099         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)         18_723.99       914.81    19_638.79       0.9956          1.0010         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)         18_723.99       823.72    19_547.71       0.9558          1.0145         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)         18_723.99       954.92    19_678.91       0.9933          1.0017         5.48
IVF-Binary-512-nl316-random (self)                    18_723.99     2_617.13    21_341.12       0.9674          1.0093         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)              22_862.92       671.48    23_534.40       0.1309             NaN         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)             22_862.92       680.50    23_543.42       0.1239             NaN         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)             22_862.92       692.55    23_555.47       0.1204             NaN         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)             22_862.92       779.12    23_642.04       0.4224          1.5615         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)             22_862.92       878.13    23_741.05       0.5793          1.2883         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)            22_862.92       792.11    23_655.03       0.3879          1.6531         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)            22_862.92       905.53    23_768.44       0.5316          1.3527         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)            22_862.92       811.24    23_674.16       0.3695          1.7111         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)            22_862.92       927.40    23_790.31       0.5067          1.3929         5.02
IVF-Binary-512-nl158-pca (self)                       22_862.92     2_576.24    25_439.15       0.3862          1.6582         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)             19_226.48       683.39    19_909.87       0.1289             NaN         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)             19_226.48       736.43    19_962.91       0.1258             NaN         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)             19_226.48       702.88    19_929.36       0.1216             NaN         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)            19_226.48       808.40    20_034.88       0.4165          1.5664         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)            19_226.48       901.04    20_127.52       0.5722          1.2915         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)            19_226.48       807.63    20_034.11       0.4017          1.6053         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)            19_226.48       979.83    20_206.31       0.5516          1.3190         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)            19_226.48       818.13    20_044.61       0.3783          1.6763         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)            19_226.48       933.43    20_159.91       0.5197          1.3683         5.21
IVF-Binary-512-nl223-pca (self)                       19_226.48     2_598.95    21_825.42       0.4000          1.6102         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)             19_602.76       701.92    20_304.68       0.1285             NaN         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)             19_602.76       706.21    20_308.98       0.1271             NaN         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)             19_602.76       716.24    20_319.00       0.1231             NaN         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)            19_602.76       819.79    20_422.56       0.4178          1.5606         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)            19_602.76       925.01    20_527.78       0.5745          1.2869         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)            19_602.76       819.44    20_422.20       0.4099          1.5808         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)            19_602.76       927.01    20_529.77       0.5630          1.3018         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)            19_602.76       831.78    20_434.54       0.3875          1.6442         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)            19_602.76       943.58    20_546.34       0.5323          1.3462         5.48
IVF-Binary-512-nl316-pca (self)                       19_602.76     2_669.04    22_271.81       0.4083          1.5851         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)          39_018.84     1_265.97    40_284.82       0.3791             NaN         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)         39_018.84     1_280.94    40_299.78       0.3465             NaN         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)         39_018.84     1_298.32    40_317.17       0.3295             NaN         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)         39_018.84     1_366.85    40_385.69       0.9854          1.0053         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)         39_018.84     1_597.14    40_615.98       0.9977          1.0008         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)        39_018.84     1_685.02    40_703.86       0.9758          1.0097         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)        39_018.84     1_671.17    40_690.01       0.9966          1.0012         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)        39_018.84     1_630.36    40_649.21       0.9677          1.0136         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)        39_018.84     1_741.83    40_760.68       0.9949          1.0018         9.57
IVF-Binary-1024-nl158-random (self)                   39_018.84     5_208.15    44_226.99       0.9767          1.0087         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)         35_513.16     1_284.33    36_797.49       0.3703             NaN         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)         35_513.16     1_291.11    36_804.27       0.3574             NaN         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)         35_513.16     1_316.36    36_829.52       0.3359             NaN         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)        35_513.16     1_385.68    36_898.83       0.9846          1.0057         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)        35_513.16     1_490.43    37_003.59       0.9980          1.0008         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)        35_513.16     1_435.29    36_948.44       0.9805          1.0074         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)        35_513.16     1_513.80    37_026.96       0.9974          1.0010         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)        35_513.16     1_470.61    36_983.77       0.9715          1.0113         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)        35_513.16     1_535.52    37_048.68       0.9956          1.0015         9.76
IVF-Binary-1024-nl223-random (self)                   35_513.16     4_582.28    40_095.44       0.9812          1.0069         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)         35_848.11     1_339.25    37_187.36       0.3678             NaN        10.03
IVF-Binary-1024-nl316-np17-rf0-random (query)         35_848.11     1_308.42    37_156.52       0.3616             NaN        10.03
IVF-Binary-1024-nl316-np25-rf0-random (query)         35_848.11     1_322.70    37_170.81       0.3429             NaN        10.03
IVF-Binary-1024-nl316-np15-rf10-random (query)        35_848.11     1_418.67    37_266.77       0.9849          1.0051        10.03
IVF-Binary-1024-nl316-np15-rf20-random (query)        35_848.11     1_515.27    37_363.38       0.9982          1.0005        10.03
IVF-Binary-1024-nl316-np17-rf10-random (query)        35_848.11     1_414.02    37_262.13       0.9830          1.0060        10.03
IVF-Binary-1024-nl316-np17-rf20-random (query)        35_848.11     1_520.07    37_368.18       0.9980          1.0006        10.03
IVF-Binary-1024-nl316-np25-rf10-random (query)        35_848.11     1_439.08    37_287.19       0.9753          1.0093        10.03
IVF-Binary-1024-nl316-np25-rf20-random (query)        35_848.11     1_544.37    37_392.48       0.9966          1.0011        10.03
IVF-Binary-1024-nl316-random (self)                   35_848.11     4_622.25    40_470.36       0.9838          1.0054        10.03
IVF-Binary-1024-nl158-np7-rf0-pca (query)             40_181.30     1_286.89    41_468.19       0.2464             NaN         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)            40_181.30     1_318.21    41_499.50       0.2420             NaN         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)            40_181.30     1_332.08    41_513.38       0.2398             NaN         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)            40_181.30     1_391.13    41_572.42       0.6566          1.2083         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)            40_181.30     1_496.09    41_677.39       0.7952          1.0964         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)           40_181.30     1_423.97    41_605.27       0.6378          1.2271         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)           40_181.30     1_527.64    41_708.94       0.7704          1.1117         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)           40_181.30     1_440.92    41_622.21       0.6289          1.2373         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)           40_181.30     1_561.34    41_742.64       0.7594          1.1193         9.57
IVF-Binary-1024-nl158-pca (self)                      40_181.30     4_646.55    44_827.85       0.6231          1.2468         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)            36_611.00     1_302.37    37_913.37       0.2448             NaN         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)            36_611.00     1_308.71    37_919.71       0.2429             NaN         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)            36_611.00     1_325.36    37_936.36       0.2405             NaN         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)           36_611.00     1_430.70    38_041.71       0.6523          1.2110         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)           36_611.00     1_510.38    38_121.39       0.7909          1.0980         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)           36_611.00     1_419.92    38_030.92       0.6442          1.2192         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)           36_611.00     1_539.76    38_150.76       0.7798          1.1049         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)           36_611.00     1_441.58    38_052.58       0.6326          1.2325         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)           36_611.00     1_562.93    38_173.93       0.7644          1.1156         9.76
IVF-Binary-1024-nl223-pca (self)                      36_611.00     4_659.44    41_270.44       0.6303          1.2375         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)            36_973.42     1_326.51    38_299.92       0.2446             NaN        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)            36_973.42     1_332.97    38_306.38       0.2437             NaN        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)            36_973.42     1_339.36    38_312.78       0.2413             NaN        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)           36_973.42     1_436.21    38_409.63       0.6517          1.2115        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)           36_973.42     1_544.26    38_517.68       0.7904          1.0983        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)           36_973.42     1_433.95    38_407.36       0.6474          1.2159        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)           36_973.42     1_541.74    38_515.16       0.7846          1.1020        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)           36_973.42     1_453.45    38_426.87       0.6362          1.2280        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)           36_973.42     1_568.32    38_541.73       0.7696          1.1118        10.04
IVF-Binary-1024-nl316-pca (self)                      36_973.42     4_721.16    41_694.57       0.6342          1.2334        10.04
IVF-Binary-768-nl158-np7-rf0-signed (query)           30_578.82       981.54    31_560.36       0.3741             NaN         7.29
IVF-Binary-768-nl158-np12-rf0-signed (query)          30_578.82       981.29    31_560.10       0.3421             NaN         7.29
IVF-Binary-768-nl158-np17-rf0-signed (query)          30_578.82     1_051.73    31_630.55       0.3255             NaN         7.29
IVF-Binary-768-nl158-np7-rf10-signed (query)          30_578.82     1_079.89    31_658.71       0.9809          1.0063         7.29
IVF-Binary-768-nl158-np7-rf20-signed (query)          30_578.82     1_178.50    31_757.31       0.9971          1.0010         7.29
IVF-Binary-768-nl158-np12-rf10-signed (query)         30_578.82     1_094.21    31_673.03       0.9694          1.0113         7.29
IVF-Binary-768-nl158-np12-rf20-signed (query)         30_578.82     1_320.08    31_898.90       0.9956          1.0015         7.29
IVF-Binary-768-nl158-np17-rf10-signed (query)         30_578.82     1_129.51    31_708.33       0.9601          1.0157         7.29
IVF-Binary-768-nl158-np17-rf20-signed (query)         30_578.82     1_232.64    31_811.45       0.9935          1.0021         7.29
IVF-Binary-768-nl158-signed (self)                    30_578.82     3_676.70    34_255.52       0.9702          1.0104         7.29
IVF-Binary-768-nl223-np11-rf0-signed (query)          26_996.64       977.45    27_974.09       0.3653             NaN         7.48
IVF-Binary-768-nl223-np14-rf0-signed (query)          26_996.64       982.99    27_979.63       0.3528             NaN         7.48
IVF-Binary-768-nl223-np21-rf0-signed (query)          26_996.64     1_000.94    27_997.58       0.3315             NaN         7.48
IVF-Binary-768-nl223-np11-rf10-signed (query)         26_996.64     1_115.85    28_112.49       0.9798          1.0066         7.48
IVF-Binary-768-nl223-np11-rf20-signed (query)         26_996.64     1_185.68    28_182.32       0.9974          1.0009         7.48
IVF-Binary-768-nl223-np14-rf10-signed (query)         26_996.64     1_101.47    28_098.12       0.9750          1.0086         7.48
IVF-Binary-768-nl223-np14-rf20-signed (query)         26_996.64     1_201.51    28_198.15       0.9966          1.0011         7.48
IVF-Binary-768-nl223-np21-rf10-signed (query)         26_996.64     1_118.66    28_115.30       0.9645          1.0131         7.48
IVF-Binary-768-nl223-np21-rf20-signed (query)         26_996.64     1_231.35    28_227.99       0.9946          1.0017         7.48
IVF-Binary-768-nl223-signed (self)                    26_996.64     3_560.29    30_556.93       0.9756          1.0081         7.48
IVF-Binary-768-nl316-np15-rf0-signed (query)          27_350.68       999.47    28_350.14       0.3629             NaN         7.76
IVF-Binary-768-nl316-np17-rf0-signed (query)          27_350.68     1_001.61    28_352.28       0.3568             NaN         7.76
IVF-Binary-768-nl316-np25-rf0-signed (query)          27_350.68     1_014.53    28_365.21       0.3386             NaN         7.76
IVF-Binary-768-nl316-np15-rf10-signed (query)         27_350.68     1_108.42    28_459.10       0.9799          1.0062         7.76
IVF-Binary-768-nl316-np15-rf20-signed (query)         27_350.68     1_221.31    28_571.99       0.9978          1.0006         7.76
IVF-Binary-768-nl316-np17-rf10-signed (query)         27_350.68     1_116.56    28_467.23       0.9775          1.0072         7.76
IVF-Binary-768-nl316-np17-rf20-signed (query)         27_350.68     1_218.74    28_569.42       0.9974          1.0007         7.76
IVF-Binary-768-nl316-np25-rf10-signed (query)         27_350.68     1_137.20    28_487.87       0.9685          1.0109         7.76
IVF-Binary-768-nl316-np25-rf20-signed (query)         27_350.68     1_238.54    28_589.21       0.9957          1.0013         7.76
IVF-Binary-768-nl316-signed (self)                    27_350.68     3_607.19    30_957.87       0.9786          1.0065         7.76
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
Exhaustive (query)                                         9.72     4_071.87     4_081.59       1.0000          1.0000        48.83
Exhaustive (self)                                          9.72    13_781.00    13_790.72       1.0000          1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                           1_386.76       779.87     2_166.63       0.5172             NaN         2.84
ExhaustiveRaBitQ-rf5 (query)                           1_386.76       844.21     2_230.96       0.9146          1.0018         2.84
ExhaustiveRaBitQ-rf10 (query)                          1_386.76       905.22     2_291.97       0.9813          1.0003         2.84
ExhaustiveRaBitQ-rf20 (query)                          1_386.76       991.10     2_377.86       0.9982          1.0000         2.84
ExhaustiveRaBitQ (self)                                1_386.76     3_009.07     4_395.83       0.9819          1.0003         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       2_020.72       230.12     2_250.84       0.5206             NaN         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      2_020.72       375.44     2_396.16       0.5206             NaN         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      2_020.72       522.85     2_543.57       0.5206             NaN         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      2_020.72       314.56     2_335.28       0.9810          1.0003         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      2_020.72       379.86     2_400.58       0.9970          1.0001         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     2_020.72       468.39     2_489.11       0.9818          1.0003         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     2_020.72       542.78     2_563.51       0.9982          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     2_020.72       625.36     2_646.08       0.9818          1.0003         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     2_020.72       703.27     2_723.99       0.9982          1.0000         2.89
IVF-RaBitQ-nl158 (self)                                2_020.72     2_343.70     4_364.43       0.9983          1.0000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_085.88       323.38     1_409.27       0.5225             NaN         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_085.88       401.46     1_487.34       0.5224             NaN         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_085.88       575.24     1_661.13       0.5223             NaN         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_085.88       407.19     1_493.07       0.9817          1.0003         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_085.88       471.35     1_557.23       0.9976          1.0000         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_085.88       495.75     1_581.63       0.9820          1.0003         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_085.88       562.24     1_648.13       0.9982          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_085.88       680.02     1_765.91       0.9821          1.0003         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_085.88       759.18     1_845.06       0.9983          1.0000         2.95
IVF-RaBitQ-nl223 (self)                                1_085.88     2_527.01     3_612.90       0.9984          1.0000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_279.97       392.31     1_672.29       0.5259             NaN         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_279.97       441.33     1_721.30       0.5258             NaN         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_279.97       630.91     1_910.88       0.5257             NaN         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_279.97       489.58     1_769.55       0.9824          1.0003         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_279.97       560.86     1_840.83       0.9981          1.0000         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_279.97       538.56     1_818.53       0.9826          1.0003         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_279.97       614.83     1_894.81       0.9983          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_279.97       736.46     2_016.43       0.9826          1.0003         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_279.97       813.22     2_093.19       0.9984          1.0000         3.04
IVF-RaBitQ-nl316 (self)                                1_279.97     2_700.21     3_980.18       0.9985          1.0000         3.04
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
Exhaustive (query)                                        20.29     9_490.34     9_510.63       1.0000          1.0000        97.66
Exhaustive (self)                                         20.29    33_494.95    33_515.24       1.0000          1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           3_937.67     2_235.34     6_173.01       0.5146             NaN         5.23
ExhaustiveRaBitQ-rf5 (query)                           3_937.67     2_299.90     6_237.57       0.9104          1.0013         5.23
ExhaustiveRaBitQ-rf10 (query)                          3_937.67     2_368.95     6_306.62       0.9791          1.0002         5.23
ExhaustiveRaBitQ-rf20 (query)                          3_937.67     2_497.62     6_435.29       0.9978          1.0000         5.23
ExhaustiveRaBitQ (self)                                3_937.67     7_916.30    11_853.97       0.9792          1.0002         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_341.25       682.80     6_024.06       0.5155             NaN         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_341.25     1_149.97     6_491.23       0.5154             NaN         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_341.25     1_616.26     6_957.51       0.5154             NaN         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_341.25       804.33     6_145.59       0.9793          1.0002         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_341.25       879.95     6_221.21       0.9975          1.0000         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_341.25     1_258.61     6_599.86       0.9797          1.0002         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_341.25     1_354.52     6_695.77       0.9980          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_341.25     1_742.06     7_083.32       0.9797          1.0002         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_341.25     1_849.30     7_190.55       0.9980          1.0000         5.32
IVF-RaBitQ-nl158 (self)                                5_341.25     6_154.69    11_495.94       0.9979          1.0000         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_237.42     1_007.42     4_244.84       0.5171             NaN         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_237.42     1_269.41     4_506.83       0.5170             NaN         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_237.42     1_883.06     5_120.49       0.5170             NaN         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_237.42     1_114.09     4_351.51       0.9788          1.0003         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_237.42     1_206.23     4_443.66       0.9965          1.0001         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_237.42     1_380.30     4_617.73       0.9798          1.0002         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_237.42     1_475.58     4_713.00       0.9978          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_237.42     1_995.28     5_232.70       0.9799          1.0002         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_237.42     2_099.47     5_336.89       0.9978          1.0000         5.44
IVF-RaBitQ-nl223 (self)                                3_237.42     6_983.55    10_220.97       0.9979          1.0000         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      3_555.06     1_333.88     4_888.93       0.5189             NaN         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      3_555.06     1_475.92     5_030.97       0.5189             NaN         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      3_555.06     2_142.98     5_698.04       0.5189             NaN         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     3_555.06     1_426.93     4_981.99       0.9795          1.0002         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     3_555.06     1_523.26     5_078.32       0.9970          1.0000         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     3_555.06     1_589.01     5_144.06       0.9800          1.0002         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     3_555.06     1_688.68     5_243.73       0.9977          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     3_555.06     2_258.75     5_813.81       0.9803          1.0002         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     3_555.06     2_364.12     5_919.18       0.9980          1.0000         5.63
IVF-RaBitQ-nl316 (self)                                3_555.06     7_861.74    11_416.80       0.9980          1.0000         5.63
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
Exhaustive (query)                                        30.29    15_600.29    15_630.58       1.0000          1.0000       146.48
Exhaustive (self)                                         30.29    52_328.15    52_358.45       1.0000          1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           8_061.85     4_984.97    13_046.82       0.4232             NaN         8.11
ExhaustiveRaBitQ-rf5 (query)                           8_061.85     5_073.35    13_135.20       0.7121          1.0471         8.11
ExhaustiveRaBitQ-rf10 (query)                          8_061.85     5_150.37    13_212.22       0.7599          1.0457         8.11
ExhaustiveRaBitQ-rf20 (query)                          8_061.85     5_292.65    13_354.50       0.7730          1.0443         8.11
ExhaustiveRaBitQ (self)                                8_061.85    17_072.96    25_134.81       0.7622          1.0457         8.11
IVF-RaBitQ-nl158-np7-rf0 (query)                      10_262.23     1_539.04    11_801.27       0.4277             NaN         8.25
IVF-RaBitQ-nl158-np12-rf0 (query)                     10_262.23     2_578.64    12_840.87       0.4276             NaN         8.25
IVF-RaBitQ-nl158-np17-rf0 (query)                     10_262.23     3_658.04    13_920.26       0.4276             NaN         8.25
IVF-RaBitQ-nl158-np7-rf10 (query)                     10_262.23     1_677.19    11_939.41       0.7620          1.0454         8.25
IVF-RaBitQ-nl158-np7-rf20 (query)                     10_262.23     1_794.23    12_056.45       0.7748          1.0438         8.25
IVF-RaBitQ-nl158-np12-rf10 (query)                    10_262.23     2_742.56    13_004.78       0.7628          1.0454         8.25
IVF-RaBitQ-nl158-np12-rf20 (query)                    10_262.23     2_855.70    13_117.92       0.7759          1.0439         8.25
IVF-RaBitQ-nl158-np17-rf10 (query)                    10_262.23     3_789.29    14_051.51       0.7628          1.0454         8.25
IVF-RaBitQ-nl158-np17-rf20 (query)                    10_262.23     3_937.34    14_199.56       0.7759          1.0439         8.25
IVF-RaBitQ-nl158 (self)                               10_262.23    12_941.74    23_203.97       0.7786          1.0435         8.25
IVF-RaBitQ-nl223-np11-rf0 (query)                      6_889.39     2_341.85     9_231.24       0.4307             NaN         8.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      6_889.39     2_949.46     9_838.85       0.4306             NaN         8.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      6_889.39     4_430.54    11_319.93       0.4306             NaN         8.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     6_889.39     2_474.21     9_363.61       0.7653          1.0448         8.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     6_889.39     2_590.51     9_479.91       0.7778          1.0432         8.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     6_889.39     3_088.52     9_977.92       0.7657          1.0448         8.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     6_889.39     3_200.87    10_090.26       0.7783          1.0432         8.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     6_889.39     4_548.49    11_437.89       0.7657          1.0448         8.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     6_889.39     4_687.47    11_576.86       0.7783          1.0432         8.44
IVF-RaBitQ-nl223 (self)                                6_889.39    15_713.06    22_602.45       0.7819          1.0431         8.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      7_523.21     3_103.39    10_626.60       0.4314             NaN         8.71
IVF-RaBitQ-nl316-np17-rf0 (query)                      7_523.21     3_478.04    11_001.26       0.4315             NaN         8.71
IVF-RaBitQ-nl316-np25-rf0 (query)                      7_523.21     5_067.62    12_590.84       0.4314             NaN         8.71
IVF-RaBitQ-nl316-np15-rf10 (query)                     7_523.21     3_240.00    10_763.22       0.7677          1.0443         8.71
IVF-RaBitQ-nl316-np15-rf20 (query)                     7_523.21     3_350.08    10_873.29       0.7806          1.0426         8.71
IVF-RaBitQ-nl316-np17-rf10 (query)                     7_523.21     3_655.42    11_178.63       0.7679          1.0443         8.71
IVF-RaBitQ-nl316-np17-rf20 (query)                     7_523.21     3_776.78    11_299.99       0.7808          1.0426         8.71
IVF-RaBitQ-nl316-np25-rf10 (query)                     7_523.21     5_256.68    12_779.89       0.7680          1.0443         8.71
IVF-RaBitQ-nl316-np25-rf20 (query)                     7_523.21     5_347.51    12_870.73       0.7809          1.0426         8.71
IVF-RaBitQ-nl316 (self)                                7_523.21    17_833.69    25_356.90       0.7845          1.0426         8.71
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
Exhaustive (query)                                         9.89     4_119.19     4_129.09       1.0000          1.0000        48.83
Exhaustive (self)                                          9.89    14_241.56    14_251.45       1.0000          1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                           1_334.18       709.46     2_043.63       0.7288             NaN         2.84
ExhaustiveRaBitQ-rf5 (query)                           1_334.18       766.62     2_100.79       0.9969          1.0001         2.84
ExhaustiveRaBitQ-rf10 (query)                          1_334.18       825.53     2_159.70       0.9999          1.0000         2.84
ExhaustiveRaBitQ-rf20 (query)                          1_334.18       925.59     2_259.76       1.0000          1.0000         2.84
ExhaustiveRaBitQ (self)                                1_334.18     2_754.14     4_088.32       1.0000          1.0000         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_978.67       218.71     2_197.39       0.7296             NaN         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_978.67       328.48     2_307.15       0.7296             NaN         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_978.67       440.92     2_419.59       0.7296             NaN         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_978.67       306.52     2_285.19       0.9999          1.0000         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_978.67       379.55     2_358.22       1.0000          1.0000         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_978.67       419.71     2_398.38       0.9999          1.0000         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_978.67       500.30     2_478.97       1.0000          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_978.67       541.91     2_520.59       0.9999          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_978.67       623.47     2_602.14       1.0000          1.0000         2.89
IVF-RaBitQ-nl158 (self)                                1_978.67     2_054.58     4_033.26       1.0000          1.0000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_267.90       301.27     1_569.16       0.7351             NaN         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_267.90       365.60     1_633.50       0.7351             NaN         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_267.90       530.49     1_798.38       0.7351             NaN         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_267.90       390.36     1_658.25       1.0000          1.0000         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_267.90       461.97     1_729.87       1.0000          1.0000         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_267.90       458.61     1_726.50       1.0000          1.0000         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_267.90       543.46     1_811.35       1.0000          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_267.90       632.53     1_900.43       1.0000          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_267.90       713.96     1_981.86       1.0000          1.0000         2.95
IVF-RaBitQ-nl223 (self)                                1_267.90     2_371.36     3_639.26       1.0000          1.0000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_504.29       404.83     1_909.12       0.7372             NaN         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_504.29       429.38     1_933.66       0.7372             NaN         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_504.29       601.80     2_106.08       0.7372             NaN         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_504.29       486.84     1_991.12       1.0000          1.0000         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_504.29       552.45     2_056.73       1.0000          1.0000         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_504.29       520.77     2_025.06       1.0000          1.0000         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_504.29       590.14     2_094.43       1.0000          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_504.29       703.75     2_208.04       1.0000          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_504.29       779.70     2_283.99       1.0000          1.0000         3.04
IVF-RaBitQ-nl316 (self)                                1_504.29     2_619.47     4_123.75       1.0000          1.0000         3.04
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
Exhaustive (query)                                        20.72     9_627.45     9_648.17       1.0000          1.0000        97.66
Exhaustive (self)                                         20.72    31_807.06    31_827.78       1.0000          1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           3_655.96     2_147.89     5_803.85       0.7430             NaN         5.23
ExhaustiveRaBitQ-rf5 (query)                           3_655.96     2_214.84     5_870.80       0.9976          1.0001         5.23
ExhaustiveRaBitQ-rf10 (query)                          3_655.96     2_293.43     5_949.39       0.9998          1.0000         5.23
ExhaustiveRaBitQ-rf20 (query)                          3_655.96     2_390.63     6_046.58       0.9999          1.0000         5.23
ExhaustiveRaBitQ (self)                                3_655.96     7_577.74    11_233.70       0.9999          1.0000         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       4_905.34       663.32     5_568.65       0.7437             NaN         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      4_905.34     1_065.97     5_971.30       0.7437             NaN         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      4_905.34     1_445.21     6_350.55       0.7437             NaN         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      4_905.34       775.80     5_681.14       0.9998          1.0000         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      4_905.34       871.46     5_776.80       0.9999          1.0000         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     4_905.34     1_162.92     6_068.26       0.9998          1.0000         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     4_905.34     1_273.64     6_178.98       0.9999          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     4_905.34     1_560.34     6_465.68       0.9998          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     4_905.34     1_656.48     6_561.82       0.9999          1.0000         5.32
IVF-RaBitQ-nl158 (self)                                4_905.34     5_511.59    10_416.93       0.9999          1.0000         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_321.27       984.05     4_305.32       0.7471             NaN         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_321.27     1_229.65     4_550.92       0.7475             NaN         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_321.27     1_786.92     5_108.19       0.7475             NaN         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_321.27     1_093.21     4_414.48       0.9986          1.0001         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_321.27     1_190.33     4_511.60       0.9986          1.0001         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_321.27     1_363.90     4_685.17       0.9998          1.0000         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_321.27     1_440.69     4_761.96       0.9999          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_321.27     1_903.48     5_224.75       0.9998          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_321.27     2_010.04     5_331.31       0.9999          1.0000         5.44
IVF-RaBitQ-nl223 (self)                                3_321.27     6_701.10    10_022.37       0.9999          1.0000         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      3_724.01     1_292.40     5_016.42       0.7477             NaN         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      3_724.01     1_447.62     5_171.64       0.7480             NaN         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      3_724.01     2_093.86     5_817.87       0.7480             NaN         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     3_724.01     1_401.46     5_125.47       0.9988          1.0001         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     3_724.01     1_496.86     5_220.87       0.9988          1.0001         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     3_724.01     1_559.31     5_283.32       0.9997          1.0000         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     3_724.01     1_656.60     5_380.62       0.9997          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     3_724.01     2_196.41     5_920.42       0.9999          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     3_724.01     2_295.98     6_019.99       0.9999          1.0000         5.63
IVF-RaBitQ-nl316 (self)                                3_724.01     7_615.38    11_339.40       0.9999          1.0000         5.63
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
Exhaustive (query)                                        30.42    15_669.96    15_700.38       1.0000          1.0000       146.48
Exhaustive (self)                                         30.42    52_609.26    52_639.68       1.0000          1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           7_622.63     4_815.82    12_438.45       0.5491             NaN         8.11
ExhaustiveRaBitQ-rf5 (query)                           7_622.63     4_874.04    12_496.67       0.7158          1.0583         8.11
ExhaustiveRaBitQ-rf10 (query)                          7_622.63     4_973.52    12_596.14       0.7301          1.0543         8.11
ExhaustiveRaBitQ-rf20 (query)                          7_622.63     5_061.24    12_683.86       0.7373          1.0514         8.11
ExhaustiveRaBitQ (self)                                7_622.63    16_458.24    24_080.87       0.7190          1.0641         8.11
IVF-RaBitQ-nl158-np7-rf0 (query)                       9_652.10     1_509.56    11_161.66       0.5591             NaN         8.25
IVF-RaBitQ-nl158-np12-rf0 (query)                      9_652.10     2_493.29    12_145.38       0.5591             NaN         8.25
IVF-RaBitQ-nl158-np17-rf0 (query)                      9_652.10     3_433.67    13_085.77       0.5591             NaN         8.25
IVF-RaBitQ-nl158-np7-rf10 (query)                      9_652.10     1_651.22    11_303.32       0.7414          1.0499         8.25
IVF-RaBitQ-nl158-np7-rf20 (query)                      9_652.10     1_759.87    11_411.97       0.7476          1.0473         8.25
IVF-RaBitQ-nl158-np12-rf10 (query)                     9_652.10     2_619.85    12_271.94       0.7414          1.0499         8.25
IVF-RaBitQ-nl158-np12-rf20 (query)                     9_652.10     2_754.32    12_406.42       0.7476          1.0474         8.25
IVF-RaBitQ-nl158-np17-rf10 (query)                     9_652.10     3_584.83    13_236.93       0.7414          1.0499         8.25
IVF-RaBitQ-nl158-np17-rf20 (query)                     9_652.10     3_690.01    13_342.11       0.7476          1.0474         8.25
IVF-RaBitQ-nl158 (self)                                9_652.10    12_224.18    21_876.28       0.7378          1.0570         8.25
IVF-RaBitQ-nl223-np11-rf0 (query)                      7_023.38     2_304.86     9_328.23       0.5676             NaN         8.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      7_023.38     2_894.86     9_918.24       0.5676             NaN         8.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      7_023.38     4_267.38    11_290.76       0.5676             NaN         8.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     7_023.38     2_417.95     9_441.33       0.7482          1.0473         8.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     7_023.38     2_524.32     9_547.69       0.7541          1.0448         8.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     7_023.38     2_998.31    10_021.69       0.7482          1.0473         8.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     7_023.38     3_101.23    10_124.61       0.7542          1.0447         8.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     7_023.38     4_344.37    11_367.75       0.7482          1.0473         8.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     7_023.38     4_450.15    11_473.53       0.7542          1.0447         8.44
IVF-RaBitQ-nl223 (self)                                7_023.38    14_857.77    21_881.15       0.7454          1.0536         8.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      7_677.39     3_071.90    10_749.29       0.5720             NaN         8.71
IVF-RaBitQ-nl316-np17-rf0 (query)                      7_677.39     3_453.51    11_130.90       0.5723             NaN         8.71
IVF-RaBitQ-nl316-np25-rf0 (query)                      7_677.39     4_974.95    12_652.34       0.5724             NaN         8.71
IVF-RaBitQ-nl316-np15-rf10 (query)                     7_677.39     3_211.60    10_888.99       0.7503          1.0459         8.71
IVF-RaBitQ-nl316-np15-rf20 (query)                     7_677.39     3_307.51    10_984.90       0.7561          1.0434         8.71
IVF-RaBitQ-nl316-np17-rf10 (query)                     7_677.39     3_586.41    11_263.81       0.7509          1.0458         8.71
IVF-RaBitQ-nl316-np17-rf20 (query)                     7_677.39     3_677.78    11_355.17       0.7567          1.0434         8.71
IVF-RaBitQ-nl316-np25-rf10 (query)                     7_677.39     5_110.46    12_787.85       0.7511          1.0458         8.71
IVF-RaBitQ-nl316-np25-rf20 (query)                     7_677.39     5_225.91    12_903.30       0.7569          1.0434         8.71
IVF-RaBitQ-nl316 (self)                                7_677.39    17_350.16    25_027.55       0.7496          1.0519         8.71
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

#### Quantisation (stress) data

<details>
<summary><b>Quantisation stress data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D - IVF-RaBitQ
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         9.88     4_101.71     4_111.59       1.0000          1.0000        48.83
Exhaustive (self)                                          9.88    13_710.38    13_720.27       1.0000          1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                           1_411.67       875.88     2_287.55       0.8680             NaN         2.84
ExhaustiveRaBitQ-rf5 (query)                           1_411.67       940.40     2_352.07       1.0000          1.0000         2.84
ExhaustiveRaBitQ-rf10 (query)                          1_411.67     1_016.25     2_427.92       1.0000          1.0000         2.84
ExhaustiveRaBitQ-rf20 (query)                          1_411.67     1_152.27     2_563.94       1.0000          1.0000         2.84
ExhaustiveRaBitQ (self)                                1_411.67     3_367.04     4_778.71       1.0000          1.0000         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       2_050.09       255.74     2_305.83       0.8725             NaN         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      2_050.09       423.21     2_473.30       0.8730             NaN         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      2_050.09       587.34     2_637.43       0.8730             NaN         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      2_050.09       349.23     2_399.32       0.9976          1.0005         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      2_050.09       423.76     2_473.85       0.9976          1.0005         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     2_050.09       528.16     2_578.24       0.9999          1.0000         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     2_050.09       613.54     2_663.63       0.9999          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     2_050.09       706.94     2_757.02       1.0000          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     2_050.09       805.85     2_855.94       1.0000          1.0000         2.89
IVF-RaBitQ-nl158 (self)                                2_050.09     2_678.75     4_728.84       1.0000          1.0000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_073.24       327.22     1_400.47       0.8832             NaN         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_073.24       410.48     1_483.72       0.8833             NaN         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_073.24       599.99     1_673.23       0.8832             NaN         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_073.24       420.46     1_493.70       0.9994          1.0001         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_073.24       493.87     1_567.11       0.9994          1.0001         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_073.24       509.05     1_582.29       0.9999          1.0000         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_073.24       594.61     1_667.85       0.9999          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_073.24       711.07     1_784.32       1.0000          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_073.24       804.09     1_877.33       1.0000          1.0000         2.95
IVF-RaBitQ-nl223 (self)                                1_073.24     2_690.96     3_764.20       1.0000          1.0000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_230.38       404.41     1_634.79       0.8894             NaN         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_230.38       453.95     1_684.33       0.8894             NaN         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_230.38       651.18     1_881.56       0.8894             NaN         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_230.38       498.48     1_728.86       0.9997          1.0001         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_230.38       574.45     1_804.83       0.9997          1.0001         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_230.38       556.59     1_786.96       0.9998          1.0000         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_230.38       636.14     1_866.51       0.9998          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_230.38       755.81     1_986.18       1.0000          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_230.38       850.57     2_080.95       1.0000          1.0000         3.04
IVF-RaBitQ-nl316 (self)                                1_230.38     2_818.14     4_048.52       1.0000          1.0000         3.04
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Quantisation stress data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D - IVF-RaBitQ
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        20.17     9_494.89     9_515.06       1.0000          1.0000        97.66
Exhaustive (self)                                         20.17    32_515.34    32_535.50       1.0000          1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           4_327.02     2_387.10     6_714.13       0.8989             NaN         5.23
ExhaustiveRaBitQ-rf5 (query)                           4_327.02     2_471.35     6_798.38       0.9947          1.0032         5.23
ExhaustiveRaBitQ-rf10 (query)                          4_327.02     2_656.72     6_983.74       0.9955          1.0025         5.23
ExhaustiveRaBitQ-rf20 (query)                          4_327.02     2_676.30     7_003.32       0.9973          1.0014         5.23
ExhaustiveRaBitQ (self)                                4_327.02     8_441.58    12_768.60       0.9957          1.0023         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_349.67       726.99     6_076.66       0.9027             NaN         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_349.67     1_218.30     6_567.97       0.9032             NaN         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_349.67     1_708.40     7_058.07       0.9032             NaN         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_349.67       839.96     6_189.63       0.9944          1.0026         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_349.67       931.53     6_281.20       0.9964          1.0013         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_349.67     1_336.92     6_686.59       0.9958          1.0024         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_349.67     1_445.68     6_795.36       0.9977          1.0011         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_349.67     1_842.35     7_192.02       0.9958          1.0024         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_349.67     1_966.29     7_315.97       0.9978          1.0011         5.32
IVF-RaBitQ-nl158 (self)                                5_349.67     6_526.86    11_876.53       0.9981          1.0009         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_064.73     1_037.54     4_102.27       0.9115             NaN         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_064.73     1_333.45     4_398.18       0.9116             NaN         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_064.73     1_932.91     4_997.64       0.9115             NaN         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_064.73     1_137.59     4_202.32       0.9967          1.0015         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_064.73     1_275.09     4_339.82       0.9985          1.0005         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_064.73     1_428.96     4_493.69       0.9969          1.0015         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_064.73     1_523.45     4_588.17       0.9987          1.0005         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_064.73     2_063.23     5_127.96       0.9969          1.0015         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_064.73     2_181.65     5_246.38       0.9988          1.0005         5.44
IVF-RaBitQ-nl223 (self)                                3_064.73     7_217.96    10_282.69       0.9988          1.0005         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      3_351.69     1_353.82     4_705.51       0.9161             NaN         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      3_351.69     1_508.51     4_860.20       0.9161             NaN         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      3_351.69     2_206.08     5_557.78       0.9162             NaN         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     3_351.69     1_443.83     4_795.53       0.9979          1.0009         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     3_351.69     1_546.73     4_898.43       0.9992          1.0003         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     3_351.69     1_621.62     4_973.31       0.9980          1.0009         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     3_351.69     1_719.14     5_070.84       0.9992          1.0003         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     3_351.69     2_304.51     5_656.20       0.9980          1.0009         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     3_351.69     2_411.01     5_762.70       0.9992          1.0003         5.63
IVF-RaBitQ-nl316 (self)                                3_351.69     8_040.12    11_391.81       0.9993          1.0003         5.63
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Quantisation stress data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D - IVF-RaBitQ
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        31.56    15_577.25    15_608.80       1.0000          1.0000       146.48
Exhaustive (self)                                         31.56    52_003.14    52_034.70       1.0000          1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           8_099.22     5_175.00    13_274.22       0.2320             NaN         8.11
ExhaustiveRaBitQ-rf5 (query)                           8_099.22     5_251.78    13_351.00       0.3931          1.7431         8.11
ExhaustiveRaBitQ-rf10 (query)                          8_099.22     5_324.29    13_423.52       0.4910          1.5609         8.11
ExhaustiveRaBitQ-rf20 (query)                          8_099.22     5_470.85    13_570.08       0.5926          1.4067         8.11
ExhaustiveRaBitQ (self)                                8_099.22    17_800.69    25_899.91       0.4907          1.5597         8.11
IVF-RaBitQ-nl158-np7-rf0 (query)                      10_239.89     1_610.73    11_850.61       0.2546             NaN         8.25
IVF-RaBitQ-nl158-np12-rf0 (query)                     10_239.89     2_710.10    12_949.99       0.2546             NaN         8.25
IVF-RaBitQ-nl158-np17-rf0 (query)                     10_239.89     3_806.09    14_045.97       0.2546             NaN         8.25
IVF-RaBitQ-nl158-np7-rf10 (query)                     10_239.89     1_752.95    11_992.84       0.5309          1.4777         8.25
IVF-RaBitQ-nl158-np7-rf20 (query)                     10_239.89     1_845.61    12_085.49       0.6338          1.3364         8.25
IVF-RaBitQ-nl158-np12-rf10 (query)                    10_239.89     2_856.85    13_096.74       0.5301          1.4787         8.25
IVF-RaBitQ-nl158-np12-rf20 (query)                    10_239.89     2_998.12    13_238.01       0.6322          1.3382         8.25
IVF-RaBitQ-nl158-np17-rf10 (query)                    10_239.89     3_930.97    14_170.86       0.5300          1.4788         8.25
IVF-RaBitQ-nl158-np17-rf20 (query)                    10_239.89     4_063.92    14_303.81       0.6319          1.3384         8.25
IVF-RaBitQ-nl158 (self)                               10_239.89    13_532.61    23_772.50       0.6326          1.3376         8.25
IVF-RaBitQ-nl223-np11-rf0 (query)                      6_687.49     2_370.77     9_058.25       0.2866             NaN         8.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      6_687.49     2_991.07     9_678.56       0.2866             NaN         8.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      6_687.49     4_526.53    11_214.02       0.2866             NaN         8.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     6_687.49     2_483.79     9_171.27       0.5900          1.3674         8.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     6_687.49     2_595.05     9_282.54       0.6957          1.2465         8.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     6_687.49     3_125.74     9_813.22       0.5898          1.3676         8.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     6_687.49     3_227.06     9_914.54       0.6950          1.2470         8.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     6_687.49     4_620.91    11_308.40       0.5896          1.3677         8.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     6_687.49     4_722.81    11_410.29       0.6947          1.2473         8.44
IVF-RaBitQ-nl223 (self)                                6_687.49    15_687.36    22_374.85       0.6959          1.2447         8.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      7_082.07     3_122.06    10_204.13       0.3188             NaN         8.71
IVF-RaBitQ-nl316-np17-rf0 (query)                      7_082.07     3_515.29    10_597.36       0.3188             NaN         8.71
IVF-RaBitQ-nl316-np25-rf0 (query)                      7_082.07     5_133.07    12_215.14       0.3188             NaN         8.71
IVF-RaBitQ-nl316-np15-rf10 (query)                     7_082.07     3_553.52    10_635.59       0.6419          1.2868         8.71
IVF-RaBitQ-nl316-np15-rf20 (query)                     7_082.07     3_630.80    10_712.88       0.7470          1.1826         8.71
IVF-RaBitQ-nl316-np17-rf10 (query)                     7_082.07     3_822.99    10_905.07       0.6419          1.2869         8.71
IVF-RaBitQ-nl316-np17-rf20 (query)                     7_082.07     4_116.02    11_198.09       0.7467          1.1828         8.71
IVF-RaBitQ-nl316-np25-rf10 (query)                     7_082.07     5_657.13    12_739.20       0.6417          1.2870         8.71
IVF-RaBitQ-nl316-np25-rf20 (query)                     7_082.07     5_577.32    12_659.39       0.7464          1.1831         8.71
IVF-RaBitQ-nl316 (self)                                7_082.07    17_963.68    25_045.76       0.7431          1.1852         8.71
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
Exhaustive (query)                                         9.86     4_093.13     4_102.99       1.0000          1.0000        48.83
Exhaustive (self)                                          9.86    13_733.27    13_743.13       1.0000          1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              151.54       355.14       506.69       0.0109             NaN         7.12
ExhaustiveTQ-b2-rf5 (query)                              151.54       439.12       590.67       0.0526          1.2562         7.12
ExhaustiveTQ-b2-rf10 (query)                             151.54       556.84       708.39       0.1030          1.1894         7.12
ExhaustiveTQ-b2-rf20 (query)                             151.54       926.69     1_078.23       0.2003          1.1318         7.12
ExhaustiveTQ-b2 (self)                                   151.54     3_086.86     3_238.40       0.1995          1.1335         7.12
ExhaustiveTQ-b4-rf0 (query)                              233.16       566.24       799.40       0.0132             NaN        13.22
ExhaustiveTQ-b4-rf5 (query)                              233.16       647.98       881.13       0.0576          1.2376        13.22
ExhaustiveTQ-b4-rf10 (query)                             233.16       773.02     1_006.17       0.1079          1.1773        13.22
ExhaustiveTQ-b4-rf20 (query)                             233.16     1_144.90     1_378.06       0.2030          1.1256        13.22
ExhaustiveTQ-b4 (self)                                   233.16     3_808.03     4_041.19       0.2033          1.1266        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_374.42       107.02     1_481.44       0.0116             NaN         7.81
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_374.42       140.95     1_515.37       0.0109             NaN         7.81
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_374.42       172.44     1_546.86       0.0109             NaN         7.81
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_374.42       268.53     1_642.94       0.1105          1.1790         7.81
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_374.42       495.86     1_870.28       0.2158          1.1228         7.81
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_374.42       314.58     1_689.00       0.1035          1.1886         7.81
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_374.42       583.70     1_958.12       0.2012          1.1311         7.81
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_374.42       355.55     1_729.96       0.1030          1.1894         7.81
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_374.42       647.41     2_021.82       0.2003          1.1318         7.81
IVF-TQ-b2-nl158 (self)                                 1_374.42     1_317.04     2_691.45       0.1995          1.1335         7.81
IVF-TQ-b2-nl223-np11-rf0 (query)                         706.74       120.70       827.44       0.0113             NaN         7.94
IVF-TQ-b2-nl223-np14-rf0 (query)                         706.74       135.57       842.31       0.0109             NaN         7.94
IVF-TQ-b2-nl223-np21-rf0 (query)                         706.74       170.58       877.32       0.0109             NaN         7.94
IVF-TQ-b2-nl223-np11-rf10 (query)                        706.74       296.07     1_002.80       0.1067          1.1837         7.94
IVF-TQ-b2-nl223-np11-rf20 (query)                        706.74       578.83     1_285.56       0.2082          1.1267         7.94
IVF-TQ-b2-nl223-np14-rf10 (query)                        706.74       319.42     1_026.16       0.1035          1.1885         7.94
IVF-TQ-b2-nl223-np14-rf20 (query)                        706.74       593.73     1_300.47       0.2014          1.1310         7.94
IVF-TQ-b2-nl223-np21-rf10 (query)                        706.74       366.79     1_073.53       0.1030          1.1894         7.94
IVF-TQ-b2-nl223-np21-rf20 (query)                        706.74       666.69     1_373.43       0.2003          1.1318         7.94
IVF-TQ-b2-nl223 (self)                                   706.74     1_335.44     2_042.18       0.1995          1.1335         7.94
IVF-TQ-b2-nl316-np15-rf0 (query)                         969.37       125.48     1_094.84       0.0113             NaN         8.13
IVF-TQ-b2-nl316-np17-rf0 (query)                         969.37       132.99     1_102.36       0.0111             NaN         8.13
IVF-TQ-b2-nl316-np25-rf0 (query)                         969.37       164.75     1_134.11       0.0109             NaN         8.13
IVF-TQ-b2-nl316-np15-rf10 (query)                        969.37       288.14     1_257.50       0.1074          1.1831         8.13
IVF-TQ-b2-nl316-np15-rf20 (query)                        969.37       544.21     1_513.58       0.2091          1.1263         8.13
IVF-TQ-b2-nl316-np17-rf10 (query)                        969.37       296.43     1_265.80       0.1049          1.1866         8.13
IVF-TQ-b2-nl316-np17-rf20 (query)                        969.37       551.85     1_521.22       0.2041          1.1294         8.13
IVF-TQ-b2-nl316-np25-rf10 (query)                        969.37       344.67     1_314.03       0.1030          1.1894         8.13
IVF-TQ-b2-nl316-np25-rf20 (query)                        969.37       635.40     1_604.77       0.2003          1.1318         8.13
IVF-TQ-b2-nl316 (self)                                   969.37     1_242.55     2_211.92       0.1995          1.1335         8.13
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_457.39       148.70     1_606.09       0.0140             NaN        14.07
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_457.39       202.36     1_659.75       0.0132             NaN        14.07
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_457.39       254.16     1_711.55       0.0132             NaN        14.07
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_457.39       316.75     1_774.14       0.1158          1.1694        14.07
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_457.39       551.88     2_009.27       0.2185          1.1185        14.07
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_457.39       386.94     1_844.33       0.1084          1.1766        14.07
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_457.39       657.92     2_115.31       0.2040          1.1250        14.07
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_457.39       449.55     1_906.94       0.1079          1.1773        14.07
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_457.39       751.09     2_208.48       0.2030          1.1256        14.07
IVF-TQ-b4-nl158 (self)                                 1_457.39     1_446.69     2_904.08       0.2033          1.1266        14.07
IVF-TQ-b4-nl223-np11-rf0 (query)                         809.52       166.07       975.59       0.0137             NaN        14.26
IVF-TQ-b4-nl223-np14-rf0 (query)                         809.52       189.91       999.43       0.0133             NaN        14.26
IVF-TQ-b4-nl223-np21-rf0 (query)                         809.52       249.67     1_059.19       0.0132             NaN        14.26
IVF-TQ-b4-nl223-np11-rf10 (query)                        809.52       349.65     1_159.18       0.1124          1.1723        14.26
IVF-TQ-b4-nl223-np11-rf20 (query)                        809.52       608.71     1_418.23       0.2117          1.1211        14.26
IVF-TQ-b4-nl223-np14-rf10 (query)                        809.52       379.35     1_188.88       0.1086          1.1765        14.26
IVF-TQ-b4-nl223-np14-rf20 (query)                        809.52       654.19     1_463.72       0.2040          1.1251        14.26
IVF-TQ-b4-nl223-np21-rf10 (query)                        809.52       455.79     1_265.31       0.1079          1.1773        14.26
IVF-TQ-b4-nl223-np21-rf20 (query)                        809.52       745.93     1_555.45       0.2030          1.1256        14.26
IVF-TQ-b4-nl223 (self)                                   809.52     1_463.11     2_272.64       0.2033          1.1266        14.26
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_060.40       170.11     1_230.52       0.0137             NaN        14.56
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_060.40       182.23     1_242.63       0.0134             NaN        14.56
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_060.40       234.09     1_294.49       0.0132             NaN        14.56
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_060.40       341.09     1_401.49       0.1130          1.1713        14.56
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_060.40       593.68     1_654.08       0.2124          1.1205        14.56
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_060.40       355.70     1_416.10       0.1103          1.1744        14.56
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_060.40       616.69     1_677.09       0.2074          1.1232        14.56
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_060.40       422.66     1_483.06       0.1079          1.1773        14.56
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_060.40       714.31     1_774.71       0.2030          1.1256        14.56
IVF-TQ-b4-nl316 (self)                                 1_060.40     1_371.70     2_432.10       0.2033          1.1266        14.56
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
Exhaustive (query)                                        20.34     9_382.26     9_402.60       1.0000          1.0000        97.66
Exhaustive (self)                                         20.34    32_381.85    32_402.19       1.0000          1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              422.58       658.40     1_080.98       0.0120             NaN        13.97
ExhaustiveTQ-b2-rf5 (query)                              422.58       726.83     1_149.41       0.0561          1.1729        13.97
ExhaustiveTQ-b2-rf10 (query)                             422.58       864.48     1_287.06       0.1081          1.1302        13.97
ExhaustiveTQ-b2-rf20 (query)                             422.58     1_250.66     1_673.24       0.2057          1.0911        13.97
ExhaustiveTQ-b2 (self)                                   422.58     4_165.12     4_587.70       0.2055          1.0916        13.97
ExhaustiveTQ-b4-rf0 (query)                              539.91     1_112.66     1_652.57       0.0183             NaN        26.18
ExhaustiveTQ-b4-rf5 (query)                              539.91     1_205.23     1_745.13       0.0633          1.1621        26.18
ExhaustiveTQ-b4-rf10 (query)                             539.91     1_334.56     1_874.47       0.1141          1.1229        26.18
ExhaustiveTQ-b4-rf20 (query)                             539.91     1_743.60     2_283.50       0.2061          1.0883        26.18
ExhaustiveTQ-b4 (self)                                   539.91     5_741.20     6_281.11       0.2069          1.0881        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        3_064.68       222.49     3_287.16       0.0125             NaN        14.98
IVF-TQ-b2-nl158-np12-rf0 (query)                       3_064.68       274.78     3_339.46       0.0119             NaN        14.98
IVF-TQ-b2-nl158-np17-rf0 (query)                       3_064.68       318.58     3_383.26       0.0119             NaN        14.98
IVF-TQ-b2-nl158-np7-rf10 (query)                       3_064.68       433.33     3_498.00       0.1140          1.1257        14.98
IVF-TQ-b2-nl158-np7-rf20 (query)                       3_064.68       661.92     3_726.60       0.2176          1.0868        14.98
IVF-TQ-b2-nl158-np12-rf10 (query)                      3_064.68       472.19     3_536.86       0.1081          1.1302        14.98
IVF-TQ-b2-nl158-np12-rf20 (query)                      3_064.68       754.14     3_818.82       0.2057          1.0911        14.98
IVF-TQ-b2-nl158-np17-rf10 (query)                      3_064.68       527.69     3_592.37       0.1081          1.1302        14.98
IVF-TQ-b2-nl158-np17-rf20 (query)                      3_064.68       833.30     3_897.97       0.2057          1.0911        14.98
IVF-TQ-b2-nl158 (self)                                 3_064.68     1_783.89     4_848.57       0.2055          1.0916        14.98
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_440.20       241.16     1_681.36       0.0123             NaN        15.21
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_440.20       266.22     1_706.43       0.0120             NaN        15.21
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_440.20       317.92     1_758.12       0.0119             NaN        15.21
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_440.20       427.62     1_867.82       0.1112          1.1282        15.21
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_440.20       700.56     2_140.76       0.2117          1.0891        15.21
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_440.20       463.42     1_903.63       0.1086          1.1299        15.21
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_440.20       748.29     2_188.49       0.2066          1.0908        15.21
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_440.20       533.80     1_974.00       0.1081          1.1302        15.21
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_440.20       849.01     2_289.21       0.2057          1.0911        15.21
IVF-TQ-b2-nl223 (self)                                 1_440.20     1_815.94     3_256.14       0.2055          1.0916        15.21
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_799.53       255.04     2_054.57       0.0123             NaN        15.54
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_799.53       268.21     2_067.74       0.0121             NaN        15.54
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_799.53       323.31     2_122.84       0.0119             NaN        15.54
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_799.53       437.12     2_236.65       0.1120          1.1274        15.54
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_799.53       716.44     2_515.97       0.2135          1.0883        15.54
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_799.53       449.10     2_248.63       0.1095          1.1293        15.54
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_799.53       749.31     2_548.85       0.2085          1.0902        15.54
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_799.53       512.40     2_311.93       0.1081          1.1302        15.54
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_799.53       829.54     2_629.08       0.2057          1.0911        15.54
IVF-TQ-b2-nl316 (self)                                 1_799.53     1_848.72     3_648.25       0.2055          1.0916        15.54
IVF-TQ-b4-nl158-np7-rf0 (query)                        3_166.08       306.93     3_473.01       0.0191             NaN        27.50
IVF-TQ-b4-nl158-np12-rf0 (query)                       3_166.08       398.43     3_564.51       0.0183             NaN        27.50
IVF-TQ-b4-nl158-np17-rf0 (query)                       3_166.08       477.76     3_643.84       0.0183             NaN        27.50
IVF-TQ-b4-nl158-np7-rf10 (query)                       3_166.08       508.40     3_674.48       0.1206          1.1186        27.50
IVF-TQ-b4-nl158-np7-rf20 (query)                       3_166.08       775.23     3_941.31       0.2184          1.0844        27.50
IVF-TQ-b4-nl158-np12-rf10 (query)                      3_166.08       613.96     3_780.03       0.1141          1.1229        27.50
IVF-TQ-b4-nl158-np12-rf20 (query)                      3_166.08       910.28     4_076.36       0.2061          1.0883        27.50
IVF-TQ-b4-nl158-np17-rf10 (query)                      3_166.08       703.42     3_869.50       0.1140          1.1229        27.50
IVF-TQ-b4-nl158-np17-rf20 (query)                      3_166.08     1_018.62     4_184.70       0.2061          1.0883        27.50
IVF-TQ-b4-nl158 (self)                                 3_166.08     2_093.46     5_259.54       0.2069          1.0881        27.50
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_572.76       335.27     1_908.03       0.0186             NaN        27.83
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_572.76       375.41     1_948.17       0.0183             NaN        27.83
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_572.76       469.94     2_042.70       0.0183             NaN        27.83
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_572.76       558.15     2_130.91       0.1171          1.1210        27.83
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_572.76       818.56     2_391.31       0.2121          1.0864        27.83
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_572.76       590.66     2_163.42       0.1144          1.1227        27.83
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_572.76       902.97     2_475.73       0.2070          1.0880        27.83
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_572.76       701.69     2_274.45       0.1141          1.1229        27.83
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_572.76     1_014.82     2_587.57       0.2061          1.0883        27.83
IVF-TQ-b4-nl223 (self)                                 1_572.76     2_109.45     3_682.21       0.2069          1.0881        27.83
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_901.29       347.12     2_248.40       0.0187             NaN        28.31
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_901.29       366.98     2_268.26       0.0184             NaN        28.31
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_901.29       449.83     2_351.12       0.0183             NaN        28.31
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_901.29       545.61     2_446.90       0.1179          1.1204        28.31
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_901.29       834.27     2_735.56       0.2141          1.0857        28.31
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_901.29       568.44     2_469.72       0.1154          1.1220        28.31
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_901.29       866.15     2_767.44       0.2091          1.0874        28.31
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_901.29       681.98     2_583.27       0.1141          1.1229        28.31
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_901.29       989.26     2_890.55       0.2061          1.0883        28.31
IVF-TQ-b4-nl316 (self)                                 1_901.29     2_121.04     4_022.32       0.2069          1.0881        28.31
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
Exhaustive (query)                                        29.71    15_451.93    15_481.64       1.0000          1.0000       146.48
Exhaustive (self)                                         29.71    52_208.88    52_238.60       1.0000          1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              871.35       968.73     1_840.08       0.0154             NaN        21.33
ExhaustiveTQ-b2-rf5 (query)                              871.35     1_072.06     1_943.41       0.0627          1.1385        21.33
ExhaustiveTQ-b2-rf10 (query)                             871.35     1_221.11     2_092.46       0.1152          1.1036        21.33
ExhaustiveTQ-b2-rf20 (query)                             871.35     1_643.30     2_514.65       0.2128          1.0710        21.33
ExhaustiveTQ-b2 (self)                                   871.35     5_455.99     6_327.34       0.2134          1.0712        21.33
ExhaustiveTQ-b4-rf0 (query)                            1_020.34     1_742.61     2_762.95       0.0148             NaN        39.64
ExhaustiveTQ-b4-rf5 (query)                            1_020.34     1_802.32     2_822.66       0.0558          1.1453        39.64
ExhaustiveTQ-b4-rf10 (query)                           1_020.34     1_949.43     2_969.78       0.1025          1.1154        39.64
ExhaustiveTQ-b4-rf20 (query)                           1_020.34     2_379.17     3_399.51       0.1923          1.0877        39.64
ExhaustiveTQ-b4 (self)                                 1_020.34     7_927.32     8_947.67       0.1918          1.0881        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        4_845.61       393.04     5_238.64       0.0162             NaN        22.62
IVF-TQ-b2-nl158-np12-rf0 (query)                       4_845.61       464.07     5_309.68       0.0154             NaN        22.62
IVF-TQ-b2-nl158-np17-rf0 (query)                       4_845.61       528.77     5_374.38       0.0154             NaN        22.62
IVF-TQ-b2-nl158-np7-rf10 (query)                       4_845.61       608.87     5_454.48       0.1215          1.1004        22.62
IVF-TQ-b2-nl158-np7-rf20 (query)                       4_845.61       883.82     5_729.43       0.2243          1.0677        22.62
IVF-TQ-b2-nl158-np12-rf10 (query)                      4_845.61       696.76     5_542.36       0.1152          1.1036        22.62
IVF-TQ-b2-nl158-np12-rf20 (query)                      4_845.61     1_000.82     5_846.43       0.2128          1.0710        22.62
IVF-TQ-b2-nl158-np17-rf10 (query)                      4_845.61       781.12     5_626.73       0.1152          1.1036        22.62
IVF-TQ-b2-nl158-np17-rf20 (query)                      4_845.61     1_125.17     5_970.78       0.2128          1.0710        22.62
IVF-TQ-b2-nl158 (self)                                 4_845.61     2_616.99     7_462.60       0.2134          1.0712        22.62
IVF-TQ-b2-nl223-np11-rf0 (query)                       2_485.43       423.71     2_909.14       0.0160             NaN        23.00
IVF-TQ-b2-nl223-np14-rf0 (query)                       2_485.43       455.65     2_941.08       0.0155             NaN        23.00
IVF-TQ-b2-nl223-np21-rf0 (query)                       2_485.43       517.04     3_002.47       0.0154             NaN        23.00
IVF-TQ-b2-nl223-np11-rf10 (query)                      2_485.43       643.25     3_128.68       0.1209          1.1003        23.00
IVF-TQ-b2-nl223-np11-rf20 (query)                      2_485.43       956.73     3_442.15       0.2233          1.0678        23.00
IVF-TQ-b2-nl223-np14-rf10 (query)                      2_485.43       669.46     3_154.89       0.1162          1.1029        23.00
IVF-TQ-b2-nl223-np14-rf20 (query)                      2_485.43     1_004.85     3_490.27       0.2147          1.0703        23.00
IVF-TQ-b2-nl223-np21-rf10 (query)                      2_485.43       745.75     3_231.17       0.1152          1.1036        23.00
IVF-TQ-b2-nl223-np21-rf20 (query)                      2_485.43     1_088.79     3_574.22       0.2128          1.0710        23.00
IVF-TQ-b2-nl223 (self)                                 2_485.43     2_517.32     5_002.75       0.2134          1.0712        23.00
IVF-TQ-b2-nl316-np15-rf0 (query)                       2_953.70       444.67     3_398.37       0.0159             NaN        23.53
IVF-TQ-b2-nl316-np17-rf0 (query)                       2_953.70       457.65     3_411.35       0.0156             NaN        23.53
IVF-TQ-b2-nl316-np25-rf0 (query)                       2_953.70       510.87     3_464.57       0.0154             NaN        23.53
IVF-TQ-b2-nl316-np15-rf10 (query)                      2_953.70       659.80     3_613.50       0.1204          1.1007        23.53
IVF-TQ-b2-nl316-np15-rf20 (query)                      2_953.70       964.25     3_917.95       0.2221          1.0682        23.53
IVF-TQ-b2-nl316-np17-rf10 (query)                      2_953.70       673.15     3_626.85       0.1177          1.1021        23.53
IVF-TQ-b2-nl316-np17-rf20 (query)                      2_953.70       995.25     3_948.95       0.2175          1.0695        23.53
IVF-TQ-b2-nl316-np25-rf10 (query)                      2_953.70       736.82     3_690.52       0.1152          1.1036        23.53
IVF-TQ-b2-nl316-np25-rf20 (query)                      2_953.70     1_064.10     4_017.80       0.2128          1.0710        23.53
IVF-TQ-b2-nl316 (self)                                 2_953.70     2_532.32     5_486.02       0.2134          1.0712        23.53
IVF-TQ-b4-nl158-np7-rf0 (query)                        5_036.07       518.46     5_554.53       0.0155             NaN        41.38
IVF-TQ-b4-nl158-np12-rf0 (query)                       5_036.07       650.64     5_686.71       0.0148             NaN        41.38
IVF-TQ-b4-nl158-np17-rf0 (query)                       5_036.07       774.60     5_810.67       0.0148             NaN        41.38
IVF-TQ-b4-nl158-np7-rf10 (query)                       5_036.07       753.80     5_789.87       0.1084          1.1125        41.38
IVF-TQ-b4-nl158-np7-rf20 (query)                       5_036.07     1_051.35     6_087.42       0.2038          1.0851        41.38
IVF-TQ-b4-nl158-np12-rf10 (query)                      5_036.07       896.17     5_932.24       0.1025          1.1154        41.38
IVF-TQ-b4-nl158-np12-rf20 (query)                      5_036.07     1_225.99     6_262.06       0.1923          1.0877        41.38
IVF-TQ-b4-nl158-np17-rf10 (query)                      5_036.07     1_042.81     6_078.88       0.1025          1.1154        41.38
IVF-TQ-b4-nl158-np17-rf20 (query)                      5_036.07     1_388.21     6_424.28       0.1923          1.0877        41.38
IVF-TQ-b4-nl158 (self)                                 5_036.07     3_054.30     8_090.37       0.1918          1.0881        41.38
IVF-TQ-b4-nl223-np11-rf0 (query)                       2_636.91       558.85     3_195.75       0.0156             NaN        41.96
IVF-TQ-b4-nl223-np14-rf0 (query)                       2_636.91       610.49     3_247.40       0.0150             NaN        41.96
IVF-TQ-b4-nl223-np21-rf0 (query)                       2_636.91       749.47     3_386.38       0.0148             NaN        41.96
IVF-TQ-b4-nl223-np11-rf10 (query)                      2_636.91       793.78     3_430.68       0.1075          1.1123        41.96
IVF-TQ-b4-nl223-np11-rf20 (query)                      2_636.91     1_104.98     3_741.89       0.2025          1.0849        41.96
IVF-TQ-b4-nl223-np14-rf10 (query)                      2_636.91       847.21     3_484.12       0.1035          1.1147        41.96
IVF-TQ-b4-nl223-np14-rf20 (query)                      2_636.91     1_168.70     3_805.60       0.1941          1.0871        41.96
IVF-TQ-b4-nl223-np21-rf10 (query)                      2_636.91       974.48     3_611.39       0.1025          1.1154        41.96
IVF-TQ-b4-nl223-np21-rf20 (query)                      2_636.91     1_320.82     3_957.72       0.1923          1.0877        41.96
IVF-TQ-b4-nl223 (self)                                 2_636.91     2_875.34     5_512.25       0.1918          1.0881        41.96
IVF-TQ-b4-nl316-np15-rf0 (query)                       3_113.57       582.01     3_695.58       0.0155             NaN        42.73
IVF-TQ-b4-nl316-np17-rf0 (query)                       3_113.57       610.74     3_724.30       0.0152             NaN        42.73
IVF-TQ-b4-nl316-np25-rf0 (query)                       3_113.57       713.91     3_827.48       0.0148             NaN        42.73
IVF-TQ-b4-nl316-np15-rf10 (query)                      3_113.57       812.33     3_925.89       0.1073          1.1126        42.73
IVF-TQ-b4-nl316-np15-rf20 (query)                      3_113.57     1_129.12     4_242.69       0.2016          1.0851        42.73
IVF-TQ-b4-nl316-np17-rf10 (query)                      3_113.57       849.30     3_962.86       0.1049          1.1138        42.73
IVF-TQ-b4-nl316-np17-rf20 (query)                      3_113.57     1_155.32     4_268.89       0.1968          1.0863        42.73
IVF-TQ-b4-nl316-np25-rf10 (query)                      3_113.57       962.96     4_076.53       0.1025          1.1154        42.73
IVF-TQ-b4-nl316-np25-rf20 (query)                      3_113.57     1_276.35     4_389.92       0.1923          1.0877        42.73
IVF-TQ-b4-nl316 (self)                                 3_113.57     2_900.74     6_014.31       0.1918          1.0881        42.73
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
Exhaustive (query)                                        10.38     4_272.24     4_282.62       1.0000          1.0000        48.83
Exhaustive (self)                                         10.38    14_561.53    14_571.91       1.0000          1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              154.24       372.85       527.09       0.0662             NaN         7.12
ExhaustiveTQ-b2-rf5 (query)                              154.24       447.32       601.55       0.1862          1.3185         7.12
ExhaustiveTQ-b2-rf10 (query)                             154.24       568.11       722.35       0.2699          1.2136         7.12
ExhaustiveTQ-b2-rf20 (query)                             154.24       928.44     1_082.68       0.4056          1.1279         7.12
ExhaustiveTQ-b2 (self)                                   154.24     3_063.80     3_218.04       0.4070          1.1561         7.12
ExhaustiveTQ-b4-rf0 (query)                              233.35       582.76       816.10       0.0871             NaN        13.22
ExhaustiveTQ-b4-rf5 (query)                              233.35       664.98       898.32       0.2059          1.2890        13.22
ExhaustiveTQ-b4-rf10 (query)                             233.35       787.06     1_020.41       0.2865          1.1965        13.22
ExhaustiveTQ-b4-rf20 (query)                             233.35     1_141.05     1_374.40       0.4170          1.1210        13.22
ExhaustiveTQ-b4 (self)                                   233.35     3_794.05     4_027.40       0.4165          1.1485        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_268.95       102.07     1_371.03       0.0664             NaN         7.81
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_268.95       114.30     1_383.26       0.0662             NaN         7.81
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_268.95       129.86     1_398.82       0.0662             NaN         7.81
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_268.95       295.99     1_564.95       0.2699          1.2136         7.81
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_268.95       602.92     1_871.88       0.4055          1.1279         7.81
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_268.95       318.50     1_587.46       0.2699          1.2136         7.81
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_268.95       665.89     1_934.85       0.4056          1.1279         7.81
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_268.95       340.19     1_609.14       0.2699          1.2136         7.81
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_268.95       710.12     1_979.08       0.4056          1.1279         7.81
IVF-TQ-b2-nl158 (self)                                 1_268.95     1_178.74     2_447.69       0.4070          1.1561         7.81
IVF-TQ-b2-nl223-np11-rf0 (query)                         835.35       110.92       946.26       0.0664             NaN         7.93
IVF-TQ-b2-nl223-np14-rf0 (query)                         835.35       117.96       953.31       0.0662             NaN         7.93
IVF-TQ-b2-nl223-np21-rf0 (query)                         835.35       143.75       979.10       0.0662             NaN         7.93
IVF-TQ-b2-nl223-np11-rf10 (query)                        835.35       283.40     1_118.74       0.2711          1.2125         7.93
IVF-TQ-b2-nl223-np11-rf20 (query)                        835.35       546.14     1_381.48       0.4078          1.1269         7.93
IVF-TQ-b2-nl223-np14-rf10 (query)                        835.35       292.42     1_127.77       0.2699          1.2136         7.93
IVF-TQ-b2-nl223-np14-rf20 (query)                        835.35       564.53     1_399.87       0.4056          1.1279         7.93
IVF-TQ-b2-nl223-np21-rf10 (query)                        835.35       329.09     1_164.44       0.2699          1.2136         7.93
IVF-TQ-b2-nl223-np21-rf20 (query)                        835.35       618.88     1_454.23       0.4056          1.1279         7.93
IVF-TQ-b2-nl223 (self)                                   835.35     1_183.17     2_018.52       0.4070          1.1561         7.93
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_063.64       119.25     1_182.89       0.0663             NaN         8.11
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_063.64       121.62     1_185.26       0.0663             NaN         8.11
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_063.64       143.01     1_206.65       0.0662             NaN         8.11
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_063.64       282.65     1_346.29       0.2707          1.2128         8.11
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_063.64       518.67     1_582.31       0.4072          1.1271         8.11
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_063.64       286.40     1_350.04       0.2702          1.2133         8.11
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_063.64       530.25     1_593.89       0.4061          1.1277         8.11
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_063.64       313.19     1_376.83       0.2699          1.2136         8.11
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_063.64       603.31     1_666.95       0.4056          1.1279         8.11
IVF-TQ-b2-nl316 (self)                                 1_063.64     1_152.66     2_216.30       0.4070          1.1561         8.11
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_342.77       140.32     1_483.09       0.0872             NaN        14.06
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_342.77       161.38     1_504.15       0.0871             NaN        14.06
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_342.77       184.65     1_527.42       0.0871             NaN        14.06
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_342.77       342.19     1_684.96       0.2865          1.1965        14.06
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_342.77       652.77     1_995.54       0.4169          1.1210        14.06
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_342.77       367.18     1_709.95       0.2865          1.1965        14.06
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_342.77       703.16     2_045.93       0.4170          1.1210        14.06
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_342.77       409.67     1_752.45       0.2865          1.1965        14.06
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_342.77       770.00     2_112.77       0.4170          1.1210        14.06
IVF-TQ-b4-nl158 (self)                                 1_342.77     1_269.47     2_612.24       0.4165          1.1485        14.06
IVF-TQ-b4-nl223-np11-rf0 (query)                         925.93       151.25     1_077.17       0.0873             NaN        14.24
IVF-TQ-b4-nl223-np14-rf0 (query)                         925.93       162.60     1_088.53       0.0871             NaN        14.24
IVF-TQ-b4-nl223-np21-rf0 (query)                         925.93       203.04     1_128.97       0.0871             NaN        14.24
IVF-TQ-b4-nl223-np11-rf10 (query)                        925.93       330.37     1_256.30       0.2876          1.1957        14.24
IVF-TQ-b4-nl223-np11-rf20 (query)                        925.93       592.85     1_518.78       0.4188          1.1202        14.24
IVF-TQ-b4-nl223-np14-rf10 (query)                        925.93       342.04     1_267.97       0.2866          1.1964        14.24
IVF-TQ-b4-nl223-np14-rf20 (query)                        925.93       626.87     1_552.80       0.4171          1.1210        14.24
IVF-TQ-b4-nl223-np21-rf10 (query)                        925.93       393.96     1_319.89       0.2865          1.1965        14.24
IVF-TQ-b4-nl223-np21-rf20 (query)                        925.93       686.97     1_612.90       0.4170          1.1210        14.24
IVF-TQ-b4-nl223 (self)                                   925.93     1_294.56     2_220.48       0.4165          1.1485        14.24
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_146.65       159.79     1_306.44       0.0872             NaN        14.51
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_146.65       166.06     1_312.71       0.0872             NaN        14.51
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_146.65       197.50     1_344.14       0.0871             NaN        14.51
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_146.65       331.38     1_478.03       0.2873          1.1959        14.51
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_146.65       572.98     1_719.62       0.4183          1.1204        14.51
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_146.65       355.35     1_501.99       0.2868          1.1962        14.51
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_146.65       596.39     1_743.04       0.4174          1.1208        14.51
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_146.65       374.65     1_521.30       0.2865          1.1965        14.51
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_146.65       636.42     1_783.07       0.4170          1.1210        14.51
IVF-TQ-b4-nl316 (self)                                 1_146.65     1_254.66     2_401.30       0.4165          1.1485        14.51
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
Exhaustive (query)                                        20.06     9_461.85     9_481.91       1.0000          1.0000        97.66
Exhaustive (self)                                         20.06    32_350.80    32_370.86       1.0000          1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              421.40       645.52     1_066.92       0.0709             NaN        13.97
ExhaustiveTQ-b2-rf5 (query)                              421.40       744.96     1_166.36       0.1815          1.2341        13.97
ExhaustiveTQ-b2-rf10 (query)                             421.40       869.71     1_291.11       0.2475          1.1648        13.97
ExhaustiveTQ-b2-rf20 (query)                             421.40     1_248.19     1_669.59       0.3619          1.1046        13.97
ExhaustiveTQ-b2 (self)                                   421.40     4_149.55     4_570.95       0.3623          1.1225        13.97
ExhaustiveTQ-b4-rf0 (query)                              543.00     1_106.49     1_649.49       0.0862             NaN        26.18
ExhaustiveTQ-b4-rf5 (query)                              543.00     1_200.16     1_743.15       0.1892          1.2262        26.18
ExhaustiveTQ-b4-rf10 (query)                             543.00     1_333.82     1_876.82       0.2498          1.1620        26.18
ExhaustiveTQ-b4-rf20 (query)                             543.00     1_728.77     2_271.77       0.3584          1.1058        26.18
ExhaustiveTQ-b4 (self)                                   543.00     5_701.08     6_244.08       0.3580          1.1245        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        2_714.53       208.39     2_922.92       0.0709             NaN        14.98
IVF-TQ-b2-nl158-np12-rf0 (query)                       2_714.53       221.38     2_935.90       0.0709             NaN        14.98
IVF-TQ-b2-nl158-np17-rf0 (query)                       2_714.53       236.31     2_950.84       0.0709             NaN        14.98
IVF-TQ-b2-nl158-np7-rf10 (query)                       2_714.53       428.67     3_143.20       0.2475          1.1648        14.98
IVF-TQ-b2-nl158-np7-rf20 (query)                       2_714.53       758.96     3_473.49       0.3619          1.1046        14.98
IVF-TQ-b2-nl158-np12-rf10 (query)                      2_714.53       439.63     3_154.16       0.2475          1.1648        14.98
IVF-TQ-b2-nl158-np12-rf20 (query)                      2_714.53       834.18     3_548.70       0.3619          1.1046        14.98
IVF-TQ-b2-nl158-np17-rf10 (query)                      2_714.53       462.04     3_176.57       0.2475          1.1648        14.98
IVF-TQ-b2-nl158-np17-rf20 (query)                      2_714.53       813.95     3_528.48       0.3619          1.1046        14.98
IVF-TQ-b2-nl158 (self)                                 2_714.53     1_660.08     4_374.61       0.3623          1.1225        14.98
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_596.81       225.72     1_822.52       0.0709             NaN        15.19
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_596.81       235.06     1_831.87       0.0709             NaN        15.19
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_596.81       260.51     1_857.31       0.0709             NaN        15.19
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_596.81       440.46     2_037.27       0.2475          1.1648        15.19
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_596.81       711.22     2_308.03       0.3619          1.1046        15.19
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_596.81       462.70     2_059.50       0.2475          1.1648        15.19
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_596.81       730.30     2_327.10       0.3619          1.1046        15.19
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_596.81       469.32     2_066.13       0.2475          1.1648        15.19
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_596.81       772.08     2_368.88       0.3619          1.1046        15.19
IVF-TQ-b2-nl223 (self)                                 1_596.81     1_664.88     3_261.68       0.3623          1.1225        15.19
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_982.52       243.04     2_225.56       0.0709             NaN        15.55
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_982.52       246.54     2_229.06       0.0709             NaN        15.55
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_982.52       266.91     2_249.43       0.0709             NaN        15.55
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_982.52       435.87     2_418.39       0.2475          1.1648        15.55
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_982.52       702.01     2_684.54       0.3619          1.1046        15.55
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_982.52       438.08     2_420.60       0.2475          1.1648        15.55
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_982.52       709.93     2_692.46       0.3619          1.1046        15.55
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_982.52       463.68     2_446.20       0.2475          1.1648        15.55
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_982.52       749.29     2_731.81       0.3619          1.1046        15.55
IVF-TQ-b2-nl316 (self)                                 1_982.52     1_670.83     3_653.35       0.3623          1.1225        15.55
IVF-TQ-b4-nl158-np7-rf0 (query)                        2_813.92       283.36     3_097.27       0.0862             NaN        27.51
IVF-TQ-b4-nl158-np12-rf0 (query)                       2_813.92       310.09     3_124.01       0.0862             NaN        27.51
IVF-TQ-b4-nl158-np17-rf0 (query)                       2_813.92       332.81     3_146.72       0.0862             NaN        27.51
IVF-TQ-b4-nl158-np7-rf10 (query)                       2_813.92       516.28     3_330.20       0.2498          1.1620        27.51
IVF-TQ-b4-nl158-np7-rf20 (query)                       2_813.92       852.95     3_666.86       0.3584          1.1058        27.51
IVF-TQ-b4-nl158-np12-rf10 (query)                      2_813.92       537.54     3_351.46       0.2498          1.1620        27.51
IVF-TQ-b4-nl158-np12-rf20 (query)                      2_813.92       888.67     3_702.59       0.3584          1.1058        27.51
IVF-TQ-b4-nl158-np17-rf10 (query)                      2_813.92       567.46     3_381.37       0.2498          1.1620        27.51
IVF-TQ-b4-nl158-np17-rf20 (query)                      2_813.92       929.62     3_743.54       0.3584          1.1058        27.51
IVF-TQ-b4-nl158 (self)                                 2_813.92     1_885.47     4_699.38       0.3580          1.1245        27.51
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_731.99       305.70     2_037.69       0.0862             NaN        27.81
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_731.99       320.18     2_052.17       0.0862             NaN        27.81
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_731.99       367.47     2_099.46       0.0862             NaN        27.81
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_731.99       517.46     2_249.45       0.2498          1.1620        27.81
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_731.99       803.46     2_535.45       0.3584          1.1058        27.81
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_731.99       534.36     2_266.35       0.2498          1.1620        27.81
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_731.99       829.48     2_561.47       0.3584          1.1058        27.81
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_731.99       584.51     2_316.50       0.2498          1.1620        27.81
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_731.99       893.45     2_625.44       0.3584          1.1058        27.81
IVF-TQ-b4-nl223 (self)                                 1_731.99     1_862.52     3_594.51       0.3580          1.1245        27.81
IVF-TQ-b4-nl316-np15-rf0 (query)                       2_104.75       323.00     2_427.75       0.0862             NaN        28.33
IVF-TQ-b4-nl316-np17-rf0 (query)                       2_104.75       331.12     2_435.88       0.0862             NaN        28.33
IVF-TQ-b4-nl316-np25-rf0 (query)                       2_104.75       368.63     2_473.39       0.0862             NaN        28.33
IVF-TQ-b4-nl316-np15-rf10 (query)                      2_104.75       525.44     2_630.19       0.2498          1.1620        28.33
IVF-TQ-b4-nl316-np15-rf20 (query)                      2_104.75       803.63     2_908.38       0.3584          1.1058        28.33
IVF-TQ-b4-nl316-np17-rf10 (query)                      2_104.75       548.68     2_653.43       0.2498          1.1620        28.33
IVF-TQ-b4-nl316-np17-rf20 (query)                      2_104.75       809.43     2_914.18       0.3584          1.1058        28.33
IVF-TQ-b4-nl316-np25-rf10 (query)                      2_104.75       573.97     2_678.72       0.2498          1.1620        28.33
IVF-TQ-b4-nl316-np25-rf20 (query)                      2_104.75       865.52     2_970.27       0.3584          1.1058        28.33
IVF-TQ-b4-nl316 (self)                                 2_104.75     1_865.90     3_970.66       0.3580          1.1245        28.33
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
Exhaustive (query)                                        30.99    15_508.01    15_539.00       1.0000          1.0000       146.48
Exhaustive (self)                                         30.99    52_751.86    52_782.85       1.0000          1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              877.75       970.79     1_848.55       0.0719             NaN        21.33
ExhaustiveTQ-b2-rf5 (query)                              877.75     1_070.92     1_948.67       0.1764          1.1855        21.33
ExhaustiveTQ-b2-rf10 (query)                             877.75     1_222.35     2_100.10       0.2312          1.1365        21.33
ExhaustiveTQ-b2-rf20 (query)                             877.75     1_646.40     2_524.15       0.3300          1.0920        21.33
ExhaustiveTQ-b2 (self)                                   877.75     5_460.95     6_338.70       0.3296          1.1027        21.33
ExhaustiveTQ-b4-rf0 (query)                            1_020.11     1_721.03     2_741.14       0.0844             NaN        39.64
ExhaustiveTQ-b4-rf5 (query)                            1_020.11     1_817.35     2_837.47       0.1812          1.1813        39.64
ExhaustiveTQ-b4-rf10 (query)                           1_020.11     1_965.45     2_985.56       0.2330          1.1352        39.64
ExhaustiveTQ-b4-rf20 (query)                           1_020.11     2_377.39     3_397.50       0.3263          1.0942        39.64
ExhaustiveTQ-b4 (self)                                 1_020.11     8_647.20     9_667.31       0.3287          1.1030        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        4_593.29       381.20     4_974.49       0.0719             NaN        22.63
IVF-TQ-b2-nl158-np12-rf0 (query)                       4_593.29       401.89     4_995.18       0.0719             NaN        22.63
IVF-TQ-b2-nl158-np17-rf0 (query)                       4_593.29       417.57     5_010.86       0.0719             NaN        22.63
IVF-TQ-b2-nl158-np7-rf10 (query)                       4_593.29       633.71     5_227.00       0.2312          1.1365        22.63
IVF-TQ-b2-nl158-np7-rf20 (query)                       4_593.29     1_003.12     5_596.41       0.3300          1.0920        22.63
IVF-TQ-b2-nl158-np12-rf10 (query)                      4_593.29       659.99     5_253.28       0.2313          1.1365        22.63
IVF-TQ-b2-nl158-np12-rf20 (query)                      4_593.29     1_037.92     5_631.21       0.3300          1.0920        22.63
IVF-TQ-b2-nl158-np17-rf10 (query)                      4_593.29       675.30     5_268.59       0.2313          1.1365        22.63
IVF-TQ-b2-nl158-np17-rf20 (query)                      4_593.29     1_072.59     5_665.88       0.3300          1.0920        22.63
IVF-TQ-b2-nl158 (self)                                 4_593.29     2_277.06     6_870.35       0.3296          1.1027        22.63
IVF-TQ-b2-nl223-np11-rf0 (query)                       2_397.28       401.77     2_799.06       0.0719             NaN        22.99
IVF-TQ-b2-nl223-np14-rf0 (query)                       2_397.28       414.28     2_811.56       0.0719             NaN        22.99
IVF-TQ-b2-nl223-np21-rf0 (query)                       2_397.28       455.65     2_852.93       0.0719             NaN        22.99
IVF-TQ-b2-nl223-np11-rf10 (query)                      2_397.28       628.62     3_025.91       0.2312          1.1365        22.99
IVF-TQ-b2-nl223-np11-rf20 (query)                      2_397.28       935.08     3_332.36       0.3300          1.0920        22.99
IVF-TQ-b2-nl223-np14-rf10 (query)                      2_397.28       644.99     3_042.27       0.2313          1.1365        22.99
IVF-TQ-b2-nl223-np14-rf20 (query)                      2_397.28       953.64     3_350.93       0.3300          1.0920        22.99
IVF-TQ-b2-nl223-np21-rf10 (query)                      2_397.28       679.65     3_076.93       0.2313          1.1365        22.99
IVF-TQ-b2-nl223-np21-rf20 (query)                      2_397.28     1_021.72     3_419.00       0.3300          1.0920        22.99
IVF-TQ-b2-nl223 (self)                                 2_397.28     2_346.76     4_744.04       0.3296          1.1027        22.99
IVF-TQ-b2-nl316-np15-rf0 (query)                       3_078.19       430.66     3_508.85       0.0719             NaN        23.51
IVF-TQ-b2-nl316-np17-rf0 (query)                       3_078.19       436.04     3_514.23       0.0719             NaN        23.51
IVF-TQ-b2-nl316-np25-rf0 (query)                       3_078.19       461.55     3_539.74       0.0719             NaN        23.51
IVF-TQ-b2-nl316-np15-rf10 (query)                      3_078.19       648.06     3_726.26       0.2313          1.1365        23.51
IVF-TQ-b2-nl316-np15-rf20 (query)                      3_078.19       941.35     4_019.54       0.3300          1.0920        23.51
IVF-TQ-b2-nl316-np17-rf10 (query)                      3_078.19       663.31     3_741.51       0.2313          1.1365        23.51
IVF-TQ-b2-nl316-np17-rf20 (query)                      3_078.19       950.26     4_028.45       0.3300          1.0920        23.51
IVF-TQ-b2-nl316-np25-rf10 (query)                      3_078.19       683.11     3_761.30       0.2313          1.1365        23.51
IVF-TQ-b2-nl316-np25-rf20 (query)                      3_078.19       998.20     4_076.39       0.3300          1.0920        23.51
IVF-TQ-b2-nl316 (self)                                 3_078.19     2_386.80     5_464.99       0.3296          1.1027        23.51
IVF-TQ-b4-nl158-np7-rf0 (query)                        4_485.11       493.17     4_978.28       0.0844             NaN        41.40
IVF-TQ-b4-nl158-np12-rf0 (query)                       4_485.11       526.86     5_011.98       0.0844             NaN        41.40
IVF-TQ-b4-nl158-np17-rf0 (query)                       4_485.11       589.44     5_074.56       0.0844             NaN        41.40
IVF-TQ-b4-nl158-np7-rf10 (query)                       4_485.11       758.11     5_243.22       0.2329          1.1352        41.40
IVF-TQ-b4-nl158-np7-rf20 (query)                       4_485.11     1_144.59     5_629.70       0.3263          1.0942        41.40
IVF-TQ-b4-nl158-np12-rf10 (query)                      4_485.11       797.39     5_282.51       0.2330          1.1352        41.40
IVF-TQ-b4-nl158-np12-rf20 (query)                      4_485.11     1_189.67     5_674.78       0.3263          1.0942        41.40
IVF-TQ-b4-nl158-np17-rf10 (query)                      4_485.11       849.89     5_335.01       0.2329          1.1352        41.40
IVF-TQ-b4-nl158-np17-rf20 (query)                      4_485.11     1_230.80     5_715.91       0.3263          1.0942        41.40
IVF-TQ-b4-nl158 (self)                                 4_485.11     2_551.63     7_036.74       0.3287          1.1030        41.40
IVF-TQ-b4-nl223-np11-rf0 (query)                       2_556.33       528.09     3_084.42       0.0844             NaN        41.92
IVF-TQ-b4-nl223-np14-rf0 (query)                       2_556.33       543.14     3_099.47       0.0844             NaN        41.92
IVF-TQ-b4-nl223-np21-rf0 (query)                       2_556.33       601.42     3_157.76       0.0844             NaN        41.92
IVF-TQ-b4-nl223-np11-rf10 (query)                      2_556.33       765.88     3_322.21       0.2330          1.1352        41.92
IVF-TQ-b4-nl223-np11-rf20 (query)                      2_556.33     1_076.98     3_633.31       0.3263          1.0942        41.92
IVF-TQ-b4-nl223-np14-rf10 (query)                      2_556.33       788.01     3_344.34       0.2329          1.1352        41.92
IVF-TQ-b4-nl223-np14-rf20 (query)                      2_556.33     1_108.29     3_664.62       0.3264          1.0942        41.92
IVF-TQ-b4-nl223-np21-rf10 (query)                      2_556.33       852.43     3_408.77       0.2329          1.1352        41.92
IVF-TQ-b4-nl223-np21-rf20 (query)                      2_556.33     1_188.92     3_745.25       0.3264          1.0942        41.92
IVF-TQ-b4-nl223 (self)                                 2_556.33     2_667.50     5_223.84       0.3287          1.1030        41.92
IVF-TQ-b4-nl316-np15-rf0 (query)                       3_259.64       553.70     3_813.34       0.0844             NaN        42.69
IVF-TQ-b4-nl316-np17-rf0 (query)                       3_259.64       567.66     3_827.30       0.0844             NaN        42.69
IVF-TQ-b4-nl316-np25-rf0 (query)                       3_259.64       613.91     3_873.55       0.0844             NaN        42.69
IVF-TQ-b4-nl316-np15-rf10 (query)                      3_259.64       779.96     4_039.60       0.2330          1.1352        42.69
IVF-TQ-b4-nl316-np15-rf20 (query)                      3_259.64     1_085.45     4_345.09       0.3264          1.0942        42.69
IVF-TQ-b4-nl316-np17-rf10 (query)                      3_259.64       798.02     4_057.66       0.2330          1.1352        42.69
IVF-TQ-b4-nl316-np17-rf20 (query)                      3_259.64     1_102.57     4_362.21       0.3263          1.0942        42.69
IVF-TQ-b4-nl316-np25-rf10 (query)                      3_259.64       861.75     4_121.39       0.2329          1.1352        42.69
IVF-TQ-b4-nl316-np25-rf20 (query)                      3_259.64     1_178.99     4_438.63       0.3263          1.0942        42.69
IVF-TQ-b4-nl316 (self)                                 3_259.64     2_697.27     5_956.91       0.3287          1.1030        42.69
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

#### Quantisation (stress) data

<details>
<summary><b>Quantisation stress data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D - TurboQuant + IVF
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         9.89     4_165.19     4_175.08       1.0000          1.0000        48.83
Exhaustive (self)                                          9.89    14_019.74    14_029.63       1.0000          1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              153.69       363.58       517.27       0.7919             NaN         7.12
ExhaustiveTQ-b2-rf5 (query)                              153.69       443.24       596.94       0.9995          1.0000         7.12
ExhaustiveTQ-b2-rf10 (query)                             153.69       570.06       723.75       1.0000          1.0000         7.12
ExhaustiveTQ-b2-rf20 (query)                             153.69       939.81     1_093.50       1.0000          1.0000         7.12
ExhaustiveTQ-b2 (self)                                   153.69     3_119.39     3_273.08       1.0000          1.0000         7.12
ExhaustiveTQ-b4-rf0 (query)                              236.15       575.09       811.24       0.8727             NaN        13.22
ExhaustiveTQ-b4-rf5 (query)                              236.15       681.64       917.79       1.0000          1.0000        13.22
ExhaustiveTQ-b4-rf10 (query)                             236.15       777.04     1_013.18       1.0000          1.0000        13.22
ExhaustiveTQ-b4-rf20 (query)                             236.15     1_150.60     1_386.75       1.0000          1.0000        13.22
ExhaustiveTQ-b4 (self)                                   236.15     3_837.76     4_073.91       1.0000          1.0000        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_380.24       128.07     1_508.31       0.7916             NaN         7.78
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_380.24       170.56     1_550.80       0.7918             NaN         7.78
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_380.24       207.40     1_587.64       0.7919             NaN         7.78
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_380.24       345.56     1_725.80       0.9982          1.0004         7.78
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_380.24       599.15     1_979.39       0.9982          1.0004         7.78
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_380.24       385.98     1_766.22       1.0000          1.0000         7.78
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_380.24       695.08     2_075.32       1.0000          1.0000         7.78
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_380.24       447.25     1_827.49       1.0000          1.0000         7.78
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_380.24       764.86     2_145.10       1.0000          1.0000         7.78
IVF-TQ-b2-nl158 (self)                                 1_380.24     1_494.58     2_874.82       1.0000          1.0000         7.78
IVF-TQ-b2-nl223-np11-rf0 (query)                         699.38       134.36       833.74       0.7919             NaN         7.93
IVF-TQ-b2-nl223-np14-rf0 (query)                         699.38       145.10       844.49       0.7919             NaN         7.93
IVF-TQ-b2-nl223-np21-rf0 (query)                         699.38       182.18       881.57       0.7919             NaN         7.93
IVF-TQ-b2-nl223-np11-rf10 (query)                        699.38       312.43     1_011.81       0.9995          1.0001         7.93
IVF-TQ-b2-nl223-np11-rf20 (query)                        699.38       625.03     1_324.42       0.9995          1.0001         7.93
IVF-TQ-b2-nl223-np14-rf10 (query)                        699.38       336.19     1_035.58       0.9999          1.0000         7.93
IVF-TQ-b2-nl223-np14-rf20 (query)                        699.38       614.40     1_313.79       0.9999          1.0000         7.93
IVF-TQ-b2-nl223-np21-rf10 (query)                        699.38       390.48     1_089.86       1.0000          1.0000         7.93
IVF-TQ-b2-nl223-np21-rf20 (query)                        699.38       697.58     1_396.96       1.0000          1.0000         7.93
IVF-TQ-b2-nl223 (self)                                   699.38     1_349.02     2_048.40       1.0000          1.0000         7.93
IVF-TQ-b2-nl316-np15-rf0 (query)                         911.61       140.99     1_052.60       0.7919             NaN         8.12
IVF-TQ-b2-nl316-np17-rf0 (query)                         911.61       140.98     1_052.59       0.7918             NaN         8.12
IVF-TQ-b2-nl316-np25-rf0 (query)                         911.61       172.82     1_084.43       0.7919             NaN         8.12
IVF-TQ-b2-nl316-np15-rf10 (query)                        911.61       304.85     1_216.47       0.9998          1.0000         8.12
IVF-TQ-b2-nl316-np15-rf20 (query)                        911.61       557.52     1_469.14       0.9998          1.0000         8.12
IVF-TQ-b2-nl316-np17-rf10 (query)                        911.61       316.71     1_228.32       0.9999          1.0000         8.12
IVF-TQ-b2-nl316-np17-rf20 (query)                        911.61       578.08     1_489.70       0.9999          1.0000         8.12
IVF-TQ-b2-nl316-np25-rf10 (query)                        911.61       361.56     1_273.17       1.0000          1.0000         8.12
IVF-TQ-b2-nl316-np25-rf20 (query)                        911.61       647.26     1_558.87       1.0000          1.0000         8.12
IVF-TQ-b2-nl316 (self)                                   911.61     1_290.63     2_202.24       1.0000          1.0000         8.12
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_455.79       180.64     1_636.43       0.8721             NaN        14.02
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_455.79       251.41     1_707.20       0.8727             NaN        14.02
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_455.79       312.06     1_767.85       0.8727             NaN        14.02
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_455.79       381.07     1_836.86       0.9982          1.0004        14.02
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_455.79       654.03     2_109.83       0.9982          1.0004        14.02
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_455.79       468.69     1_924.48       1.0000          1.0000        14.02
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_455.79       780.44     2_236.23       1.0000          1.0000        14.02
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_455.79       545.90     2_001.70       1.0000          1.0000        14.02
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_455.79       872.61     2_328.40       1.0000          1.0000        14.02
IVF-TQ-b4-nl158 (self)                                 1_455.79     1_632.82     3_088.61       1.0000          1.0000        14.02
IVF-TQ-b4-nl223-np11-rf0 (query)                         789.68       179.77       969.45       0.8726             NaN        14.24
IVF-TQ-b4-nl223-np14-rf0 (query)                         789.68       207.13       996.80       0.8727             NaN        14.24
IVF-TQ-b4-nl223-np21-rf0 (query)                         789.68       267.92     1_057.60       0.8727             NaN        14.24
IVF-TQ-b4-nl223-np11-rf10 (query)                        789.68       371.54     1_161.21       0.9995          1.0001        14.24
IVF-TQ-b4-nl223-np11-rf20 (query)                        789.68       626.59     1_416.27       0.9995          1.0001        14.24
IVF-TQ-b4-nl223-np14-rf10 (query)                        789.68       401.78     1_191.45       0.9999          1.0000        14.24
IVF-TQ-b4-nl223-np14-rf20 (query)                        789.68       691.59     1_481.26       0.9999          1.0000        14.24
IVF-TQ-b4-nl223-np21-rf10 (query)                        789.68       479.54     1_269.21       1.0000          1.0000        14.24
IVF-TQ-b4-nl223-np21-rf20 (query)                        789.68       788.72     1_578.40       1.0000          1.0000        14.24
IVF-TQ-b4-nl223 (self)                                   789.68     1_463.24     2_252.92       1.0000          1.0000        14.24
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_009.68       183.21     1_192.89       0.8727             NaN        14.53
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_009.68       197.42     1_207.10       0.8727             NaN        14.53
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_009.68       249.08     1_258.76       0.8727             NaN        14.53
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_009.68       357.03     1_366.71       0.9998          1.0000        14.53
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_009.68       626.42     1_636.10       0.9998          1.0000        14.53
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_009.68       375.64     1_385.31       0.9999          1.0000        14.53
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_009.68       643.44     1_653.12       0.9999          1.0000        14.53
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_009.68       439.52     1_449.20       1.0000          1.0000        14.53
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_009.68       730.51     1_740.18       1.0000          1.0000        14.53
IVF-TQ-b4-nl316 (self)                                 1_009.68     1_413.57     2_423.25       1.0000          1.0000        14.53
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Quantisation stress data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D - TurboQuant + IVF
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        19.99     9_453.72     9_473.71       1.0000          1.0000        97.66
Exhaustive (self)                                         19.99    32_395.11    32_415.10       1.0000          1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              447.56       668.99     1_116.55       0.8424             NaN        13.97
ExhaustiveTQ-b2-rf5 (query)                              447.56       740.07     1_187.64       1.0000          1.0000        13.97
ExhaustiveTQ-b2-rf10 (query)                             447.56       876.25     1_323.82       1.0000          1.0000        13.97
ExhaustiveTQ-b2-rf20 (query)                             447.56     1_268.39     1_715.95       1.0000          1.0000        13.97
ExhaustiveTQ-b2 (self)                                   447.56     4_229.19     4_676.75       1.0000          1.0000        13.97
ExhaustiveTQ-b4-rf0 (query)                              540.43     1_102.19     1_642.62       0.8985             NaN        26.18
ExhaustiveTQ-b4-rf5 (query)                              540.43     1_204.77     1_745.21       1.0000          1.0000        26.18
ExhaustiveTQ-b4-rf10 (query)                             540.43     1_341.77     1_882.20       1.0000          1.0000        26.18
ExhaustiveTQ-b4-rf20 (query)                             540.43     1_729.32     2_269.76       1.0000          1.0000        26.18
ExhaustiveTQ-b4 (self)                                   540.43     5_752.88     6_293.32       1.0000          1.0000        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        3_022.69       248.19     3_270.88       0.8420             NaN        14.96
IVF-TQ-b2-nl158-np12-rf0 (query)                       3_022.69       315.91     3_338.60       0.8424             NaN        14.96
IVF-TQ-b2-nl158-np17-rf0 (query)                       3_022.69       375.03     3_397.72       0.8424             NaN        14.96
IVF-TQ-b2-nl158-np7-rf10 (query)                       3_022.69       474.13     3_496.81       0.9986          1.0003        14.96
IVF-TQ-b2-nl158-np7-rf20 (query)                       3_022.69       772.55     3_795.24       0.9986          1.0003        14.96
IVF-TQ-b2-nl158-np12-rf10 (query)                      3_022.69       555.37     3_578.06       0.9999          1.0000        14.96
IVF-TQ-b2-nl158-np12-rf20 (query)                      3_022.69       889.43     3_912.12       0.9999          1.0000        14.96
IVF-TQ-b2-nl158-np17-rf10 (query)                      3_022.69       621.52     3_644.21       1.0000          1.0000        14.96
IVF-TQ-b2-nl158-np17-rf20 (query)                      3_022.69       968.71     3_991.39       1.0000          1.0000        14.96
IVF-TQ-b2-nl158 (self)                                 3_022.69     2_025.13     5_047.81       1.0000          1.0000        14.96
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_321.70       259.74     1_581.44       0.8424             NaN        15.24
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_321.70       306.41     1_628.11       0.8424             NaN        15.24
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_321.70       348.84     1_670.54       0.8424             NaN        15.24
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_321.70       468.06     1_789.77       0.9997          1.0000        15.24
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_321.70       750.47     2_072.17       0.9997          1.0000        15.24
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_321.70       518.49     1_840.19       0.9999          1.0000        15.24
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_321.70       794.34     2_116.05       0.9999          1.0000        15.24
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_321.70       575.76     1_897.46       1.0000          1.0000        15.24
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_321.70       888.29     2_209.99       1.0000          1.0000        15.24
IVF-TQ-b2-nl223 (self)                                 1_321.70     1_945.27     3_266.97       1.0000          1.0000        15.24
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_573.73       270.17     1_843.91       0.8424             NaN        15.57
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_573.73       284.90     1_858.64       0.8424             NaN        15.57
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_573.73       356.92     1_930.65       0.8424             NaN        15.57
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_573.73       476.78     2_050.51       0.9999          1.0000        15.57
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_573.73       764.75     2_338.49       0.9999          1.0000        15.57
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_573.73       490.00     2_063.73       1.0000          1.0000        15.57
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_573.73       783.67     2_357.40       1.0000          1.0000        15.57
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_573.73       549.61     2_123.34       1.0000          1.0000        15.57
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_573.73       867.53     2_441.26       1.0000          1.0000        15.57
IVF-TQ-b2-nl316 (self)                                 1_573.73     1_879.16     3_452.89       1.0000          1.0000        15.57
IVF-TQ-b4-nl158-np7-rf0 (query)                        3_173.48       352.46     3_525.93       0.8977             NaN        27.46
IVF-TQ-b4-nl158-np12-rf0 (query)                       3_173.48       473.77     3_647.25       0.8985             NaN        27.46
IVF-TQ-b4-nl158-np17-rf0 (query)                       3_173.48       576.72     3_750.20       0.8985             NaN        27.46
IVF-TQ-b4-nl158-np7-rf10 (query)                       3_173.48       586.70     3_760.17       0.9986          1.0003        27.46
IVF-TQ-b4-nl158-np7-rf20 (query)                       3_173.48       884.87     4_058.34       0.9986          1.0003        27.46
IVF-TQ-b4-nl158-np12-rf10 (query)                      3_173.48       716.10     3_889.57       0.9999          1.0000        27.46
IVF-TQ-b4-nl158-np12-rf20 (query)                      3_173.48     1_051.12     4_224.60       0.9999          1.0000        27.46
IVF-TQ-b4-nl158-np17-rf10 (query)                      3_173.48       827.44     4_000.92       1.0000          1.0000        27.46
IVF-TQ-b4-nl158-np17-rf20 (query)                      3_173.48     1_177.36     4_350.84       1.0000          1.0000        27.46
IVF-TQ-b4-nl158 (self)                                 3_173.48     2_318.70     5_492.18       1.0000          1.0000        27.46
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_444.11       382.16     1_826.27       0.8984             NaN        27.90
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_444.11       421.11     1_865.21       0.8984             NaN        27.90
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_444.11       527.14     1_971.25       0.8985             NaN        27.90
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_444.11       574.68     2_018.79       0.9997          1.0000        27.90
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_444.11       859.77     2_303.88       0.9997          1.0000        27.90
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_444.11       628.63     2_072.74       0.9999          1.0000        27.90
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_444.11       930.89     2_375.00       0.9999          1.0000        27.90
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_444.11       752.61     2_196.71       1.0000          1.0000        27.90
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_444.11     1_068.81     2_512.92       1.0000          1.0000        27.90
IVF-TQ-b4-nl223 (self)                                 1_444.11     2_176.42     3_620.53       1.0000          1.0000        27.90
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_667.26       372.58     2_039.83       0.8984             NaN        28.38
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_667.26       396.06     2_063.32       0.8984             NaN        28.38
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_667.26       500.52     2_167.77       0.8985             NaN        28.38
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_667.26       581.62     2_248.87       0.9999          1.0000        28.38
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_667.26       872.84     2_540.10       0.9999          1.0000        28.38
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_667.26       609.08     2_276.33       1.0000          1.0000        28.38
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_667.26       902.46     2_569.71       1.0000          1.0000        28.38
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_667.26       706.18     2_373.44       1.0000          1.0000        28.38
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_667.26     1_022.52     2_689.78       1.0000          1.0000        28.38
IVF-TQ-b4-nl316 (self)                                 1_667.26     2_124.49     3_791.75       1.0000          1.0000        28.38
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Quantisation stress data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D - TurboQuant + IVF
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        31.74    15_533.30    15_565.04       1.0000          1.0000       146.48
Exhaustive (self)                                         31.74    52_222.13    52_253.86       1.0000          1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              874.38       972.06     1_846.45       0.8736             NaN        21.33
ExhaustiveTQ-b2-rf5 (query)                              874.38     1_084.00     1_958.39       1.0000          1.0000        21.33
ExhaustiveTQ-b2-rf10 (query)                             874.38     1_237.65     2_112.04       1.0000          1.0000        21.33
ExhaustiveTQ-b2-rf20 (query)                             874.38     1_660.74     2_535.12       1.0000          1.0000        21.33
ExhaustiveTQ-b2 (self)                                   874.38     5_511.66     6_386.05       1.0000          1.0000        21.33
ExhaustiveTQ-b4-rf0 (query)                            1_025.96     1_738.12     2_764.09       0.9097             NaN        39.64
ExhaustiveTQ-b4-rf5 (query)                            1_025.96     1_805.48     2_831.44       1.0000          1.0000        39.64
ExhaustiveTQ-b4-rf10 (query)                           1_025.96     1_957.09     2_983.05       1.0000          1.0000        39.64
ExhaustiveTQ-b4-rf20 (query)                           1_025.96     2_371.85     3_397.81       1.0000          1.0000        39.64
ExhaustiveTQ-b4 (self)                                 1_025.96     7_909.61     8_935.58       1.0000          1.0000        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        4_768.02       437.68     5_205.70       0.8735             NaN        22.61
IVF-TQ-b2-nl158-np12-rf0 (query)                       4_768.02       533.76     5_301.78       0.8736             NaN        22.61
IVF-TQ-b2-nl158-np17-rf0 (query)                       4_768.02       608.62     5_376.64       0.8736             NaN        22.61
IVF-TQ-b2-nl158-np7-rf10 (query)                       4_768.02       688.55     5_456.57       0.9995          1.0001        22.61
IVF-TQ-b2-nl158-np7-rf20 (query)                       4_768.02     1_004.88     5_772.90       0.9995          1.0001        22.61
IVF-TQ-b2-nl158-np12-rf10 (query)                      4_768.02       799.44     5_567.46       1.0000          1.0000        22.61
IVF-TQ-b2-nl158-np12-rf20 (query)                      4_768.02     1_150.04     5_918.06       1.0000          1.0000        22.61
IVF-TQ-b2-nl158-np17-rf10 (query)                      4_768.02       885.00     5_653.02       1.0000          1.0000        22.61
IVF-TQ-b2-nl158-np17-rf20 (query)                      4_768.02     1_249.07     6_017.09       1.0000          1.0000        22.61
IVF-TQ-b2-nl158 (self)                                 4_768.02     2_782.51     7_550.53       1.0000          1.0000        22.61
IVF-TQ-b2-nl223-np11-rf0 (query)                       2_130.70       447.45     2_578.16       0.8736             NaN        23.00
IVF-TQ-b2-nl223-np14-rf0 (query)                       2_130.70       487.17     2_617.87       0.8736             NaN        23.00
IVF-TQ-b2-nl223-np21-rf0 (query)                       2_130.70       569.13     2_699.83       0.8736             NaN        23.00
IVF-TQ-b2-nl223-np11-rf10 (query)                      2_130.70       679.18     2_809.89       0.9998          1.0000        23.00
IVF-TQ-b2-nl223-np11-rf20 (query)                      2_130.70       996.79     3_127.49       0.9998          1.0000        23.00
IVF-TQ-b2-nl223-np14-rf10 (query)                      2_130.70       723.34     2_854.05       1.0000          1.0000        23.00
IVF-TQ-b2-nl223-np14-rf20 (query)                      2_130.70     1_066.02     3_196.72       1.0000          1.0000        23.00
IVF-TQ-b2-nl223-np21-rf10 (query)                      2_130.70       817.73     2_948.43       1.0000          1.0000        23.00
IVF-TQ-b2-nl223-np21-rf20 (query)                      2_130.70     1_168.92     3_299.62       1.0000          1.0000        23.00
IVF-TQ-b2-nl223 (self)                                 2_130.70     2_646.41     4_777.11       1.0000          1.0000        23.00
IVF-TQ-b2-nl316-np15-rf0 (query)                       2_504.84       470.21     2_975.05       0.8736             NaN        23.53
IVF-TQ-b2-nl316-np17-rf0 (query)                       2_504.84       487.92     2_992.76       0.8736             NaN        23.53
IVF-TQ-b2-nl316-np25-rf0 (query)                       2_504.84       559.75     3_064.59       0.8736             NaN        23.53
IVF-TQ-b2-nl316-np15-rf10 (query)                      2_504.84       694.77     3_199.61       1.0000          1.0000        23.53
IVF-TQ-b2-nl316-np15-rf20 (query)                      2_504.84     1_009.85     3_514.69       1.0000          1.0000        23.53
IVF-TQ-b2-nl316-np17-rf10 (query)                      2_504.84       715.32     3_220.16       1.0000          1.0000        23.53
IVF-TQ-b2-nl316-np17-rf20 (query)                      2_504.84     1_039.39     3_544.23       1.0000          1.0000        23.53
IVF-TQ-b2-nl316-np25-rf10 (query)                      2_504.84       797.75     3_302.58       1.0000          1.0000        23.53
IVF-TQ-b2-nl316-np25-rf20 (query)                      2_504.84     1_154.86     3_659.69       1.0000          1.0000        23.53
IVF-TQ-b2-nl316 (self)                                 2_504.84     2_649.34     5_154.17       1.0000          1.0000        23.53
IVF-TQ-b4-nl158-np7-rf0 (query)                        4_945.22       602.21     5_547.43       0.9094             NaN        41.37
IVF-TQ-b4-nl158-np12-rf0 (query)                       4_945.22       777.54     5_722.76       0.9097             NaN        41.37
IVF-TQ-b4-nl158-np17-rf0 (query)                       4_945.22       924.09     5_869.32       0.9097             NaN        41.37
IVF-TQ-b4-nl158-np7-rf10 (query)                       4_945.22       856.04     5_801.26       0.9995          1.0001        41.37
IVF-TQ-b4-nl158-np7-rf20 (query)                       4_945.22     1_173.91     6_119.13       0.9995          1.0001        41.37
IVF-TQ-b4-nl158-np12-rf10 (query)                      4_945.22     1_046.62     5_991.84       1.0000          1.0000        41.37
IVF-TQ-b4-nl158-np12-rf20 (query)                      4_945.22     1_399.06     6_344.28       1.0000          1.0000        41.37
IVF-TQ-b4-nl158-np17-rf10 (query)                      4_945.22     1_204.05     6_149.27       1.0000          1.0000        41.37
IVF-TQ-b4-nl158-np17-rf20 (query)                      4_945.22     1_564.70     6_509.92       1.0000          1.0000        41.37
IVF-TQ-b4-nl158 (self)                                 4_945.22     3_266.19     8_211.42       1.0000          1.0000        41.37
IVF-TQ-b4-nl223-np11-rf0 (query)                       2_284.85       649.20     2_934.05       0.9096             NaN        41.96
IVF-TQ-b4-nl223-np14-rf0 (query)                       2_284.85       683.70     2_968.55       0.9097             NaN        41.96
IVF-TQ-b4-nl223-np21-rf0 (query)                       2_284.85       837.70     3_122.56       0.9097             NaN        41.96
IVF-TQ-b4-nl223-np11-rf10 (query)                      2_284.85       842.41     3_127.26       0.9998          1.0000        41.96
IVF-TQ-b4-nl223-np11-rf20 (query)                      2_284.85     1_156.83     3_441.68       0.9998          1.0000        41.96
IVF-TQ-b4-nl223-np14-rf10 (query)                      2_284.85       919.81     3_204.67       1.0000          1.0000        41.96
IVF-TQ-b4-nl223-np14-rf20 (query)                      2_284.85     1_250.56     3_535.41       1.0000          1.0000        41.96
IVF-TQ-b4-nl223-np21-rf10 (query)                      2_284.85     1_084.61     3_369.46       1.0000          1.0000        41.96
IVF-TQ-b4-nl223-np21-rf20 (query)                      2_284.85     1_434.98     3_719.84       1.0000          1.0000        41.96
IVF-TQ-b4-nl223 (self)                                 2_284.85     3_047.04     5_331.90       1.0000          1.0000        41.96
IVF-TQ-b4-nl316-np15-rf0 (query)                       2_615.77       625.64     3_241.41       0.9097             NaN        42.73
IVF-TQ-b4-nl316-np17-rf0 (query)                       2_615.77       667.76     3_283.53       0.9097             NaN        42.73
IVF-TQ-b4-nl316-np25-rf0 (query)                       2_615.77       817.47     3_433.24       0.9097             NaN        42.73
IVF-TQ-b4-nl316-np15-rf10 (query)                      2_615.77       852.22     3_467.99       1.0000          1.0000        42.73
IVF-TQ-b4-nl316-np15-rf20 (query)                      2_615.77     1_169.95     3_785.72       1.0000          1.0000        42.73
IVF-TQ-b4-nl316-np17-rf10 (query)                      2_615.77       889.89     3_505.67       1.0000          1.0000        42.73
IVF-TQ-b4-nl316-np17-rf20 (query)                      2_615.77     1_224.12     3_839.89       1.0000          1.0000        42.73
IVF-TQ-b4-nl316-np25-rf10 (query)                      2_615.77     1_036.68     3_652.45       1.0000          1.0000        42.73
IVF-TQ-b4-nl316-np25-rf20 (query)                      2_615.77     1_411.45     4_027.22       1.0000          1.0000        42.73
IVF-TQ-b4-nl316 (self)                                 2_615.77     2_987.42     5_603.19       1.0000          1.0000        42.73
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
