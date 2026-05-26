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
cargo run --example gridsearch_binary --release --features binary -- --n-dim 512 --n-samples 50000 --data embedding
```

For RaBitQ:

```bash
cargo run --example gridsearch_rabitq --release --features binary -- --n-dim 512 --n-samples 50000 --data embedding
```

For TurboQuantisation

```bash
cargo run --example gridsearch_tq --release --features binary -- --n-dim 512 --n-samples 50000 --data embedding
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
Exhaustive (query)                                         9.84     4_094.53     4_104.37       1.0000          1.0000        48.83
Exhaustive (self)                                          9.84    13_875.05    13_884.89       1.0000          1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_567.08       271.01     2_838.10       0.0311             NaN         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_567.08       358.52     2_925.60       0.1623          1.1262         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_567.08       452.35     3_019.44       0.2695          1.0812         1.78
ExhaustiveBinary-256-random (self)                     2_567.08     1_183.68     3_750.76       0.1681          1.1202         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_766.21       278.96     3_045.17       0.1873             NaN         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_766.21       384.53     3_150.74       0.5340          1.0265         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_766.21       484.78     3_250.99       0.6684          1.0147         1.78
ExhaustiveBinary-256-pca (self)                        2_766.21     1_269.74     4_035.95       0.5319          1.0270         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_070.36       429.75     5_500.12       0.0628             NaN         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_070.36       528.15     5_598.52       0.2086          1.0932         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_070.36       632.91     5_703.28       0.3233          1.0582         3.55
ExhaustiveBinary-512-random (self)                     5_070.36     1_769.64     6_840.00       0.2130          1.0890         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_393.66       440.84     5_834.50       0.2013             NaN         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_393.66       550.45     5_944.10       0.6344          1.0175         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_393.66       658.53     6_052.19       0.8009          1.0073         3.55
ExhaustiveBinary-512-pca (self)                        5_393.66     1_822.78     7_216.44       0.6350          1.0176         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_052.85       801.14    10_853.99       0.0901             NaN         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_052.85       865.18    10_918.03       0.2539          1.0733         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_052.85       971.19    11_024.03       0.3803          1.0456         7.10
ExhaustiveBinary-1024-random (self)                   10_052.85     2_853.60    12_906.45       0.2560          1.0731         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_599.11       774.89    11_374.00       0.2080             NaN         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_599.11       892.96    11_492.07       0.6484          1.0164         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_599.11       993.26    11_592.36       0.8112          1.0067         7.10
ExhaustiveBinary-1024-pca (self)                      10_599.11     2_957.28    13_556.38       0.6483          1.0165         7.10
ExhaustiveBinary-256-signed_no_rr (query)              2_569.25       271.03     2_840.29       0.0311             NaN         1.78
ExhaustiveBinary-256-signed-rf10 (query)               2_569.25       357.94     2_927.19       0.1623          1.1262         1.78
ExhaustiveBinary-256-signed-rf20 (query)               2_569.25       452.18     3_021.43       0.2695          1.0812         1.78
ExhaustiveBinary-256-signed (self)                     2_569.25     1_182.49     3_751.74       0.1681          1.1202         1.78
IVF-Binary-256-nl158-np7-rf0-random (query)            3_970.92       111.30     4_082.22       0.0553             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           3_970.92       117.24     4_088.16       0.0419             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           3_970.92       125.03     4_095.95       0.0336             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           3_970.92       161.51     4_132.43       0.2371          1.0915         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           3_970.92       206.70     4_177.62       0.3629          1.0576         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          3_970.92       167.38     4_138.30       0.1990          1.1180         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          3_970.92       217.52     4_188.44       0.3175          1.0724         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          3_970.92       175.55     4_146.47       0.1717          1.1366         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          3_970.92       233.30     4_204.22       0.2820          1.0860         1.93
IVF-Binary-256-nl158-random (self)                     3_970.92       502.05     4_472.97       0.2048          1.1126         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           3_078.89       117.61     3_196.50       0.0524             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           3_078.89       119.05     3_197.94       0.0407             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           3_078.89       129.55     3_208.44       0.0324             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          3_078.89       172.70     3_251.58       0.2308          1.0969         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          3_078.89       215.42     3_294.31       0.3557          1.0603         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          3_078.89       170.62     3_249.50       0.1978          1.1184         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          3_078.89       219.93     3_298.82       0.3168          1.0721         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          3_078.89       178.44     3_257.33       0.1716          1.1362         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          3_078.89       231.22     3_310.10       0.2840          1.0839         2.00
IVF-Binary-256-nl223-random (self)                     3_078.89       508.37     3_587.26       0.2034          1.1133         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           3_274.42       123.30     3_397.72       0.0421             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           3_274.42       125.95     3_400.37       0.0388             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           3_274.42       131.28     3_405.70       0.0349             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          3_274.42       172.73     3_447.15       0.2022          1.1114         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          3_274.42       238.19     3_512.61       0.3241          1.0690         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          3_274.42       173.10     3_447.52       0.1921          1.1168         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          3_274.42       221.90     3_496.32       0.3111          1.0727         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          3_274.42       180.36     3_454.78       0.1776          1.1270         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          3_274.42       230.88     3_505.30       0.2917          1.0798         2.09
IVF-Binary-256-nl316-random (self)                     3_274.42       522.50     3_796.92       0.1966          1.1118         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               4_201.81       118.09     4_319.91       0.1990             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              4_201.81       124.39     4_326.21       0.1973             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              4_201.81       133.20     4_335.02       0.1966             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              4_201.81       180.53     4_382.35       0.6300          1.0178         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              4_201.81       228.30     4_430.11       0.7968          1.0074         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             4_201.81       189.03     4_390.85       0.6193          1.0187         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             4_201.81       246.37     4_448.18       0.7838          1.0081         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             4_201.81       201.09     4_402.90       0.6122          1.0192         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             4_201.81       262.62     4_464.43       0.7743          1.0085         1.93
IVF-Binary-256-nl158-pca (self)                        4_201.81       580.52     4_782.34       0.6187          1.0189         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              3_291.71       123.79     3_415.50       0.1984             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              3_291.71       127.36     3_419.07       0.1972             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              3_291.71       135.22     3_426.93       0.1961             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             3_291.71       185.07     3_476.79       0.6277          1.0179         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             3_291.71       235.76     3_527.47       0.7944          1.0075         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             3_291.71       190.51     3_482.22       0.6216          1.0184         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             3_291.71       246.47     3_538.18       0.7868          1.0079         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             3_291.71       203.81     3_495.52       0.6139          1.0191         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             3_291.71       264.00     3_555.72       0.7769          1.0084         2.00
IVF-Binary-256-nl223-pca (self)                        3_291.71       583.24     3_874.95       0.6208          1.0187         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              3_492.94       129.82     3_622.77       0.1988             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              3_492.94       131.49     3_624.43       0.1982             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              3_492.94       144.33     3_637.27       0.1970             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             3_492.94       192.34     3_685.28       0.6287          1.0179         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             3_492.94       245.74     3_738.68       0.7957          1.0075         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             3_492.94       194.04     3_686.99       0.6250          1.0182         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             3_492.94       249.59     3_742.54       0.7913          1.0077         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             3_492.94       202.81     3_695.76       0.6174          1.0188         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             3_492.94       262.69     3_755.63       0.7816          1.0082         2.09
IVF-Binary-256-nl316-pca (self)                        3_492.94       594.20     4_087.14       0.6245          1.0184         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_398.47       202.90     6_601.37       0.0790             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_398.47       212.99     6_611.45       0.0704             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_398.47       225.14     6_623.60       0.0642             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_398.47       258.88     6_657.34       0.2477          1.0769         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_398.47       305.97     6_704.44       0.3704          1.0480         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_398.47       268.04     6_666.51       0.2271          1.0864         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_398.47       321.25     6_719.72       0.3453          1.0537         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_398.47       279.82     6_678.29       0.2144          1.0929         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_398.47       336.25     6_734.72       0.3315          1.0572         3.71
IVF-Binary-512-nl158-random (self)                     6_398.47       843.00     7_241.47       0.2315          1.0829         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_507.55       208.07     5_715.62       0.0777             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_507.55       213.07     5_720.62       0.0708             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_507.55       220.42     5_727.97       0.0644             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_507.55       266.92     5_774.47       0.2450          1.0773         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_507.55       313.24     5_820.79       0.3685          1.0481         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_507.55       269.04     5_776.59       0.2293          1.0845         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_507.55       324.83     5_832.38       0.3486          1.0527         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_507.55       281.20     5_788.75       0.2146          1.0925         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_507.55       339.32     5_846.87       0.3307          1.0575         3.77
IVF-Binary-512-nl223-random (self)                     5_507.55       845.25     6_352.80       0.2330          1.0812         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           5_689.41       214.23     5_903.65       0.0705             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           5_689.41       216.39     5_905.81       0.0686             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           5_689.41       223.82     5_913.23       0.0659             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          5_689.41       270.31     5_959.72       0.2320          1.0829         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          5_689.41       345.63     6_035.05       0.3543          1.0514         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          5_689.41       271.39     5_960.80       0.2271          1.0849         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          5_689.41       323.88     6_013.30       0.3475          1.0529         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          5_689.41       281.15     5_970.56       0.2175          1.0903         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          5_689.41       339.14     6_028.56       0.3343          1.0564         3.86
IVF-Binary-512-nl316-random (self)                     5_689.41       867.46     6_556.88       0.2310          1.0817         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               6_761.72       218.73     6_980.46       0.2032             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              6_761.72       222.67     6_984.40       0.2018             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              6_761.72       233.30     6_995.02       0.2016             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              6_761.72       278.49     7_040.21       0.6396          1.0170         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              6_761.72       322.31     7_084.04       0.8063          1.0070         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             6_761.72       285.30     7_047.02       0.6351          1.0174         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             6_761.72       340.60     7_102.32       0.8013          1.0072         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             6_761.72       299.46     7_061.18       0.6347          1.0175         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             6_761.72       357.94     7_119.66       0.8011          1.0073         3.71
IVF-Binary-512-nl158-pca (self)                        6_761.72       900.19     7_661.91       0.6356          1.0176         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              5_831.75       218.05     6_049.80       0.2032             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              5_831.75       223.08     6_054.83       0.2023             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              5_831.75       236.36     6_068.11       0.2015             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             5_831.75       278.67     6_110.42       0.6394          1.0170         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             5_831.75       329.41     6_161.16       0.8068          1.0070         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             5_831.75       288.32     6_120.07       0.6367          1.0173         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             5_831.75       346.54     6_178.29       0.8034          1.0071         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             5_831.75       303.45     6_135.20       0.6344          1.0175         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             5_831.75       361.51     6_193.26       0.8010          1.0073         3.77
IVF-Binary-512-nl223-pca (self)                        5_831.75       904.74     6_736.49       0.6369          1.0174         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              6_014.39       224.74     6_239.12       0.2030             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              6_014.39       226.88     6_241.26       0.2026             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              6_014.39       233.72     6_248.11       0.2017             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             6_014.39       288.25     6_302.63       0.6392          1.0171         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             6_014.39       341.17     6_355.56       0.8063          1.0070         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             6_014.39       288.08     6_302.46       0.6377          1.0172         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             6_014.39       343.39     6_357.78       0.8044          1.0071         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             6_014.39       299.26     6_313.65       0.6353          1.0174         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             6_014.39       358.12     6_372.51       0.8016          1.0072         3.86
IVF-Binary-512-nl316-pca (self)                        6_014.39       910.88     6_925.27       0.6378          1.0173         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_281.38       385.14    11_666.52       0.0964             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_281.38       407.06    11_688.44       0.0933             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_281.38       416.13    11_697.52       0.0909             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_281.38       445.00    11_726.38       0.2765          1.0662         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_281.38       501.61    11_782.99       0.4097          1.0410         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_281.38       460.96    11_742.34       0.2643          1.0701         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_281.38       518.31    11_799.69       0.3922          1.0437         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_281.38       477.89    11_759.28       0.2577          1.0723         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_281.38       540.71    11_822.10       0.3852          1.0449         7.26
IVF-Binary-1024-nl158-random (self)                   11_281.38     1_487.84    12_769.22       0.2668          1.0699         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_403.02       394.79    10_797.82       0.0967             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_403.02       399.65    10_802.67       0.0936             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_403.02       413.68    10_816.70       0.0909             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_403.02       452.44    10_855.47       0.2759          1.0662         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_403.02       507.32    10_910.34       0.4080          1.0410         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_403.02       460.32    10_863.34       0.2664          1.0694         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_403.02       515.18    10_918.20       0.3953          1.0431         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_403.02       479.44    10_882.46       0.2579          1.0726         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_403.02       538.22    10_941.25       0.3844          1.0451         7.32
IVF-Binary-1024-nl223-random (self)                   10_403.02     1_483.34    11_886.36       0.2684          1.0693         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         10_592.36       409.37    11_001.74       0.0936             NaN         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)         10_592.36       402.29    10_994.66       0.0928             NaN         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)         10_592.36       412.34    11_004.70       0.0914             NaN         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)        10_592.36       457.50    11_049.87       0.2690          1.0685         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)        10_592.36       510.56    11_102.93       0.4008          1.0422         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)        10_592.36       460.01    11_052.37       0.2655          1.0696         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)        10_592.36       515.17    11_107.53       0.3954          1.0431         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)        10_592.36       474.22    11_066.59       0.2593          1.0721         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)        10_592.36       539.06    11_131.42       0.3856          1.0450         7.41
IVF-Binary-1024-nl316-random (self)                   10_592.36     1_484.82    12_077.19       0.2673          1.0695         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)             11_854.22       402.48    12_256.69       0.2100             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            11_854.22       416.64    12_270.86       0.2088             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            11_854.22       433.81    12_288.02       0.2085             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            11_854.22       463.81    12_318.02       0.6531          1.0160         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            11_854.22       511.15    12_365.37       0.8162          1.0065         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           11_854.22       481.55    12_335.77       0.6490          1.0163         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           11_854.22       535.78    12_389.99       0.8117          1.0067         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           11_854.22       499.91    12_354.12       0.6488          1.0163         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           11_854.22       559.71    12_413.93       0.8114          1.0067         7.26
IVF-Binary-1024-nl158-pca (self)                      11_854.22     1_551.63    13_405.85       0.6490          1.0165         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            10_952.17       411.98    11_364.15       0.2099             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            10_952.17       418.04    11_370.20       0.2092             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            10_952.17       431.30    11_383.47       0.2084             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           10_952.17       472.10    11_424.27       0.6525          1.0160         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           10_952.17       521.51    11_473.68       0.8164          1.0065         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           10_952.17       481.92    11_434.09       0.6497          1.0162         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           10_952.17       535.57    11_487.74       0.8135          1.0066         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           10_952.17       499.74    11_451.91       0.6480          1.0164         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           10_952.17       559.57    11_511.74       0.8112          1.0067         7.32
IVF-Binary-1024-nl223-pca (self)                      10_952.17     1_549.86    12_502.03       0.6502          1.0163         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            11_152.06       418.92    11_570.98       0.2098             NaN         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            11_152.06       423.96    11_576.03       0.2093             NaN         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            11_152.06       430.44    11_582.50       0.2086             NaN         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           11_152.06       478.41    11_630.47       0.6523          1.0160         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           11_152.06       546.85    11_698.91       0.8158          1.0065         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           11_152.06       481.00    11_633.06       0.6508          1.0161         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           11_152.06       536.45    11_688.51       0.8141          1.0066         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           11_152.06       495.90    11_647.96       0.6487          1.0164         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           11_152.06       555.92    11_707.98       0.8115          1.0067         7.42
IVF-Binary-1024-nl316-pca (self)                      11_152.06     1_548.15    12_700.21       0.6510          1.0163         7.42
IVF-Binary-256-nl158-np7-rf0-signed (query)            3_940.04       111.17     4_051.21       0.0553             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-signed (query)           3_940.04       117.03     4_057.07       0.0419             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-signed (query)           3_940.04       124.16     4_064.20       0.0336             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-signed (query)           3_940.04       160.81     4_100.85       0.2371          1.0915         1.93
IVF-Binary-256-nl158-np7-rf20-signed (query)           3_940.04       206.55     4_146.59       0.3629          1.0576         1.93
IVF-Binary-256-nl158-np12-rf10-signed (query)          3_940.04       168.38     4_108.42       0.1990          1.1180         1.93
IVF-Binary-256-nl158-np12-rf20-signed (query)          3_940.04       217.60     4_157.64       0.3175          1.0724         1.93
IVF-Binary-256-nl158-np17-rf10-signed (query)          3_940.04       175.74     4_115.78       0.1717          1.1366         1.93
IVF-Binary-256-nl158-np17-rf20-signed (query)          3_940.04       227.71     4_167.75       0.2820          1.0860         1.93
IVF-Binary-256-nl158-signed (self)                     3_940.04       499.92     4_439.96       0.2048          1.1126         1.93
IVF-Binary-256-nl223-np11-rf0-signed (query)           3_026.29       124.16     3_150.45       0.0524             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-signed (query)           3_026.29       119.42     3_145.71       0.0407             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-signed (query)           3_026.29       125.52     3_151.81       0.0324             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-signed (query)          3_026.29       167.25     3_193.54       0.2308          1.0969         2.00
IVF-Binary-256-nl223-np11-rf20-signed (query)          3_026.29       213.81     3_240.10       0.3557          1.0603         2.00
IVF-Binary-256-nl223-np14-rf10-signed (query)          3_026.29       170.31     3_196.60       0.1978          1.1184         2.00
IVF-Binary-256-nl223-np14-rf20-signed (query)          3_026.29       219.56     3_245.85       0.3168          1.0721         2.00
IVF-Binary-256-nl223-np21-rf10-signed (query)          3_026.29       177.95     3_204.24       0.1716          1.1362         2.00
IVF-Binary-256-nl223-np21-rf20-signed (query)          3_026.29       231.14     3_257.43       0.2840          1.0839         2.00
IVF-Binary-256-nl223-signed (self)                     3_026.29       514.65     3_540.94       0.2034          1.1133         2.00
IVF-Binary-256-nl316-np15-rf0-signed (query)           3_239.40       122.81     3_362.21       0.0421             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-signed (query)           3_239.40       123.94     3_363.34       0.0388             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-signed (query)           3_239.40       129.50     3_368.89       0.0349             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-signed (query)          3_239.40       174.49     3_413.88       0.2022          1.1114         2.09
IVF-Binary-256-nl316-np15-rf20-signed (query)          3_239.40       219.94     3_459.34       0.3241          1.0690         2.09
IVF-Binary-256-nl316-np17-rf10-signed (query)          3_239.40       172.74     3_412.14       0.1921          1.1168         2.09
IVF-Binary-256-nl316-np17-rf20-signed (query)          3_239.40       221.50     3_460.89       0.3111          1.0727         2.09
IVF-Binary-256-nl316-np25-rf10-signed (query)          3_239.40       179.62     3_419.02       0.1776          1.1270         2.09
IVF-Binary-256-nl316-np25-rf20-signed (query)          3_239.40       235.72     3_475.12       0.2917          1.0798         2.09
IVF-Binary-256-nl316-signed (self)                     3_239.40       523.17     3_762.57       0.1966          1.1118         2.09
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
Exhaustive (query)                                        20.20     9_556.94     9_577.14       1.0000          1.0000        97.66
Exhaustive (self)                                         20.20    32_435.03    32_455.23       1.0000          1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_642.16       375.45     6_017.61       0.0292             NaN         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_642.16       487.82     6_129.98       0.1469          1.0921         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_642.16       606.23     6_248.39       0.2447          1.0585         2.03
ExhaustiveBinary-256-random (self)                     5_642.16     1_605.84     7_248.00       0.1509          1.0875         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_113.48       386.36     6_499.84       0.1394             NaN         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_113.48       516.25     6_629.74       0.3908          1.0290         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_113.48       692.90     6_806.39       0.5142          1.0182         2.03
ExhaustiveBinary-256-pca (self)                        6_113.48     1_709.05     7_822.54       0.3906          1.0291         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_217.54       645.90    11_863.44       0.0610             NaN         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_217.54       769.84    11_987.38       0.1812          1.0668         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_217.54       891.59    12_109.13       0.2805          1.0428         4.05
ExhaustiveBinary-512-random (self)                    11_217.54     2_524.53    13_742.07       0.1844          1.0639         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_739.56       656.91    12_396.46       0.1676             NaN         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_739.56       785.53    12_525.08       0.4490          1.2486         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_739.56       919.01    12_658.56       0.5696          1.0152         4.05
ExhaustiveBinary-512-pca (self)                       11_739.56     2_634.28    14_373.84       0.4499          1.3208         4.05
ExhaustiveBinary-1024-random_no_rr (query)            21_954.66     1_172.99    23_127.64       0.0826             NaN         8.10
ExhaustiveBinary-1024-random-rf10 (query)             21_954.66     1_305.52    23_260.18       0.2064          1.0563         8.10
ExhaustiveBinary-1024-random-rf20 (query)             21_954.66     1_470.21    23_424.86       0.3140          1.0365         8.10
ExhaustiveBinary-1024-random (self)                   21_954.66     4_330.55    26_285.20       0.2075          1.0564         8.10
ExhaustiveBinary-1024-pca_no_rr (query)               22_990.14     1_198.31    24_188.46       0.2037             NaN         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                22_990.14     1_340.70    24_330.84       0.6282          1.0116         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                22_990.14     1_482.79    24_472.93       0.7931          1.0049         8.11
ExhaustiveBinary-1024-pca (self)                      22_990.14     4_467.68    27_457.83       0.6281          1.0116         8.11
ExhaustiveBinary-512-signed_no_rr (query)             11_028.73       642.96    11_671.68       0.0610             NaN         4.05
ExhaustiveBinary-512-signed-rf10 (query)              11_028.73       765.23    11_793.95       0.1812          1.0668         4.05
ExhaustiveBinary-512-signed-rf20 (query)              11_028.73       886.13    11_914.85       0.2805          1.0428         4.05
ExhaustiveBinary-512-signed (self)                    11_028.73     2_519.80    13_548.53       0.1844          1.0639         4.05
IVF-Binary-256-nl158-np7-rf0-random (query)            8_616.94       226.19     8_843.12       0.0552             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           8_616.94       232.87     8_849.80       0.0437             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           8_616.94       240.63     8_857.56       0.0311             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           8_616.94       304.36     8_921.30       0.2177          1.0627         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           8_616.94       381.57     8_998.50       0.3277          1.0402         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          8_616.94       307.58     8_924.51       0.1815          1.0819         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          8_616.94       390.95     9_007.89       0.2846          1.0520         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          8_616.94       334.84     8_951.78       0.1523          1.0975         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          8_616.94       408.74     9_025.68       0.2512          1.0606         2.34
IVF-Binary-256-nl158-random (self)                     8_616.94       949.05     9_565.98       0.1855          1.0777         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           6_533.26       237.78     6_771.04       0.0484             NaN         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)           6_533.26       241.04     6_774.29       0.0398             NaN         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)           6_533.26       245.92     6_779.18       0.0318             NaN         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)          6_533.26       314.52     6_847.78       0.2010          1.0697         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)          6_533.26       391.35     6_924.61       0.3126          1.0429         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)          6_533.26       317.28     6_850.54       0.1812          1.0789         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)          6_533.26       397.41     6_930.66       0.2892          1.0486         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)          6_533.26       333.15     6_866.41       0.1563          1.0924         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)          6_533.26       410.56     6_943.82       0.2570          1.0578         2.46
IVF-Binary-256-nl223-random (self)                     6_533.26       975.82     7_509.08       0.1845          1.0746         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)           6_797.33       251.73     7_049.06       0.0387             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           6_797.33       251.62     7_048.95       0.0357             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           6_797.33       257.49     7_054.82       0.0324             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          6_797.33       328.98     7_126.31       0.1803          1.0784         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          6_797.33       409.43     7_206.76       0.2936          1.0472         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          6_797.33       327.40     7_124.73       0.1748          1.0808         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          6_797.33       411.76     7_209.09       0.2853          1.0488         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          6_797.33       335.33     7_132.65       0.1598          1.0890         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          6_797.33       418.12     7_215.45       0.2631          1.0553         2.65
IVF-Binary-256-nl316-random (self)                     6_797.33     1_016.96     7_814.29       0.1791          1.0763         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               9_095.27       235.53     9_330.80       0.1473             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              9_095.27       242.07     9_337.34       0.1458             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              9_095.27       249.08     9_344.35       0.1452             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              9_095.27       325.36     9_420.63       0.4621          1.0223         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              9_095.27       405.90     9_501.17       0.6308          1.0117         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             9_095.27       333.50     9_428.77       0.4534          1.0230         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             9_095.27       421.12     9_516.39       0.6169          1.0124         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             9_095.27       347.68     9_442.95       0.4489          1.0234         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             9_095.27       442.97     9_538.24       0.6094          1.0127         2.34
IVF-Binary-256-nl158-pca (self)                        9_095.27     1_040.50    10_135.77       0.4533          1.0231         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              6_996.07       245.69     7_241.77       0.1470             NaN         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              6_996.07       251.32     7_247.39       0.1460             NaN         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              6_996.07       255.34     7_251.41       0.1451             NaN         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             6_996.07       336.97     7_333.04       0.4617          1.0222         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             6_996.07       418.96     7_415.03       0.6294          1.0117         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             6_996.07       339.81     7_335.89       0.4562          1.0227         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             6_996.07       428.10     7_424.18       0.6209          1.0121         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             6_996.07       352.86     7_348.93       0.4508          1.0232         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             6_996.07       443.04     7_439.11       0.6125          1.0126         2.47
IVF-Binary-256-nl223-pca (self)                        6_996.07     1_064.37     8_060.44       0.4557          1.0229         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              7_258.14       259.65     7_517.79       0.1465             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              7_258.14       262.40     7_520.54       0.1461             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              7_258.14       266.80     7_524.94       0.1452             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             7_258.14       357.86     7_616.00       0.4614          1.0223         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             7_258.14       435.96     7_694.10       0.6294          1.0117         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             7_258.14       355.37     7_613.51       0.4586          1.0225         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             7_258.14       453.41     7_711.54       0.6252          1.0119         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             7_258.14       362.09     7_620.23       0.4523          1.0231         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             7_258.14       452.09     7_710.22       0.6150          1.0124         2.65
IVF-Binary-256-nl316-pca (self)                        7_258.14     1_107.97     8_366.11       0.4582          1.0226         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           14_095.53       421.36    14_516.89       0.0772             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          14_095.53       429.98    14_525.51       0.0686             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          14_095.53       440.28    14_535.81       0.0614             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          14_095.53       506.65    14_602.19       0.2152          1.0555         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          14_095.53       575.55    14_671.09       0.3254          1.0358         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         14_095.53       509.46    14_604.99       0.1976          1.0622         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         14_095.53       639.15    14_734.69       0.3011          1.0401         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         14_095.53       526.20    14_621.74       0.1846          1.0669         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         14_095.53       608.26    14_703.79       0.2853          1.0428         4.36
IVF-Binary-512-nl158-random (self)                    14_095.53     1_633.13    15_728.67       0.2006          1.0599         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          11_979.79       435.32    12_415.11       0.0732             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          11_979.79       435.53    12_415.32       0.0682             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          11_979.79       446.43    12_426.22       0.0626             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         11_979.79       513.34    12_493.13       0.2087          1.0573         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         11_979.79       590.09    12_569.88       0.3171          1.0368         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         11_979.79       517.62    12_497.41       0.1981          1.0608         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         11_979.79       597.75    12_577.55       0.3029          1.0390         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         11_979.79       528.21    12_508.00       0.1855          1.0660         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         11_979.79       613.77    12_593.56       0.2868          1.0422         4.49
IVF-Binary-512-nl223-random (self)                    11_979.79     1_656.67    13_636.46       0.2014          1.0584         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          12_288.91       448.14    12_737.06       0.0678             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          12_288.91       447.82    12_736.73       0.0662             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          12_288.91       458.71    12_747.63       0.0631             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         12_288.91       527.49    12_816.40       0.2012          1.0595         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         12_288.91       607.13    12_896.04       0.3103          1.0377         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         12_288.91       527.91    12_816.83       0.1971          1.0608         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         12_288.91       621.93    12_910.84       0.3043          1.0387         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         12_288.91       537.37    12_826.29       0.1878          1.0648         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         12_288.91       623.21    12_912.13       0.2905          1.0414         4.67
IVF-Binary-512-nl316-random (self)                    12_288.91     1_702.63    13_991.55       0.2005          1.0584         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              14_745.08       434.65    15_179.73       0.2021             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             14_745.08       445.65    15_190.73       0.1991             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             14_745.08       455.48    15_200.57       0.1965             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             14_745.08       529.93    15_275.02       0.6220          1.0119         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             14_745.08       599.98    15_345.07       0.7874          1.0051         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            14_745.08       534.61    15_279.69       0.6055          1.0127         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            14_745.08       620.60    15_365.68       0.7662          1.0058         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            14_745.08       553.45    15_298.53       0.5920          1.0134         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            14_745.08       645.61    15_390.70       0.7501          1.0063         4.36
IVF-Binary-512-nl158-pca (self)                       14_745.08     1_713.28    16_458.36       0.6057          1.0128         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             12_585.74       447.97    13_033.71       0.2020             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             12_585.74       451.48    13_037.22       0.2002             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             12_585.74       460.66    13_046.39       0.1979             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            12_585.74       536.65    13_122.39       0.6192          1.0120         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            12_585.74       615.00    13_200.74       0.7840          1.0052         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            12_585.74       540.89    13_126.63       0.6110          1.0124         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            12_585.74       629.23    13_214.97       0.7745          1.0055         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            12_585.74       560.73    13_146.47       0.5990          1.0131         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            12_585.74       649.96    13_235.70       0.7589          1.0060         4.49
IVF-Binary-512-nl223-pca (self)                       12_585.74     1_732.11    14_317.85       0.6116          1.0124         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             12_905.06       460.80    13_365.85       0.2017             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             12_905.06       466.93    13_371.99       0.2008             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             12_905.06       470.91    13_375.96       0.1984             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            12_905.06       551.32    13_456.38       0.6197          1.0120         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            12_905.06       634.13    13_539.19       0.7851          1.0052         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            12_905.06       552.35    13_457.40       0.6158          1.0122         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            12_905.06       636.82    13_541.88       0.7800          1.0053         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            12_905.06       564.08    13_469.14       0.6033          1.0128         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            12_905.06       652.90    13_557.96       0.7648          1.0058         4.67
IVF-Binary-512-nl316-pca (self)                       12_905.06     1_791.69    14_696.75       0.6161          1.0122         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          25_027.72       820.01    25_847.73       0.0880             NaN         8.41
IVF-Binary-1024-nl158-np12-rf0-random (query)         25_027.72       830.22    25_857.93       0.0852             NaN         8.41
IVF-Binary-1024-nl158-np17-rf0-random (query)         25_027.72       843.77    25_871.49       0.0831             NaN         8.41
IVF-Binary-1024-nl158-np7-rf10-random (query)         25_027.72       899.77    25_927.49       0.2267          1.0511         8.41
IVF-Binary-1024-nl158-np7-rf20-random (query)         25_027.72       967.83    25_995.55       0.3427          1.0328         8.41
IVF-Binary-1024-nl158-np12-rf10-random (query)        25_027.72       908.25    25_935.97       0.2152          1.0544         8.41
IVF-Binary-1024-nl158-np12-rf20-random (query)        25_027.72       989.41    26_017.12       0.3265          1.0352         8.41
IVF-Binary-1024-nl158-np17-rf10-random (query)        25_027.72       928.49    25_956.21       0.2080          1.0564         8.41
IVF-Binary-1024-nl158-np17-rf20-random (query)        25_027.72     1_015.70    26_043.42       0.3178          1.0364         8.41
IVF-Binary-1024-nl158-random (self)                   25_027.72     2_982.74    28_010.46       0.2175          1.0543         8.41
IVF-Binary-1024-nl223-np11-rf0-random (query)         22_862.85       833.94    23_696.79       0.0867             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         22_862.85       839.71    23_702.56       0.0851             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         22_862.85       862.68    23_725.53       0.0832             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        22_862.85       919.32    23_782.17       0.2225          1.0520         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        22_862.85       987.07    23_849.92       0.3379          1.0333         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        22_862.85       916.25    23_779.10       0.2162          1.0539         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        22_862.85       994.43    23_857.28       0.3270          1.0348         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        22_862.85       933.04    23_795.89       0.2090          1.0561         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        22_862.85     1_017.86    23_880.71       0.3172          1.0363         8.54
IVF-Binary-1024-nl223-random (self)                   22_862.85     2_984.16    25_847.01       0.2178          1.0538         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         23_164.77       841.26    24_006.02       0.0853             NaN         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)         23_164.77       845.02    24_009.78       0.0848             NaN         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)         23_164.77       873.62    24_038.38       0.0834             NaN         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)        23_164.77       933.46    24_098.22       0.2202          1.0526         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)        23_164.77     1_010.55    24_175.31       0.3355          1.0336         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)        23_164.77       927.64    24_092.41       0.2166          1.0536         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)        23_164.77     1_009.60    24_174.36       0.3301          1.0343         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)        23_164.77       940.95    24_105.71       0.2101          1.0556         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)        23_164.77     1_029.27    24_194.04       0.3202          1.0359         8.72
IVF-Binary-1024-nl316-random (self)                   23_164.77     3_079.77    26_244.54       0.2183          1.0535         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)             26_028.68       838.87    26_867.56       0.2050             NaN         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            26_028.68       863.39    26_892.08       0.2037             NaN         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            26_028.68       878.48    26_907.17       0.2034             NaN         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            26_028.68       921.30    26_949.98       0.6327          1.0113         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            26_028.68       999.01    27_027.70       0.7986          1.0048         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           26_028.68       951.27    26_979.95       0.6284          1.0116         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           26_028.68     1_029.26    27_057.94       0.7935          1.0049         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           26_028.68       965.95    26_994.64       0.6279          1.0116         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           26_028.68     1_055.40    27_084.08       0.7932          1.0049         8.42
IVF-Binary-1024-nl158-pca (self)                      26_028.68     3_063.42    29_092.11       0.6284          1.0116         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            23_866.76       850.84    24_717.59       0.2051             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            23_866.76       857.81    24_724.57       0.2041             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            23_866.76       872.42    24_739.17       0.2036             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           23_866.76       936.81    24_803.57       0.6334          1.0113         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           23_866.76     1_018.22    24_884.98       0.7994          1.0047         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           23_866.76       952.74    24_819.49       0.6300          1.0115         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           23_866.76     1_028.91    24_895.66       0.7956          1.0049         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           23_866.76       966.14    24_832.90       0.6278          1.0116         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           23_866.76     1_054.74    24_921.49       0.7931          1.0049         8.54
IVF-Binary-1024-nl223-pca (self)                      23_866.76     3_078.79    26_945.54       0.6304          1.0115         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            24_183.22       866.21    25_049.43       0.2051             NaN         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            24_183.22       868.70    25_051.92       0.2046             NaN         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            24_183.22       880.70    25_063.92       0.2037             NaN         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           24_183.22       952.47    25_135.69       0.6336          1.0113         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           24_183.22     1_034.02    25_217.24       0.7992          1.0047         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           24_183.22       959.54    25_142.76       0.6319          1.0114         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           24_183.22     1_041.45    25_224.67       0.7973          1.0048         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           24_183.22       977.10    25_160.32       0.6285          1.0116         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           24_183.22     1_058.13    25_241.35       0.7934          1.0049         8.73
IVF-Binary-1024-nl316-pca (self)                      24_183.22     3_112.87    27_296.09       0.6320          1.0114         8.73
IVF-Binary-512-nl158-np7-rf0-signed (query)           14_111.06       420.28    14_531.34       0.0772             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-signed (query)          14_111.06       430.08    14_541.13       0.0686             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-signed (query)          14_111.06       441.05    14_552.11       0.0614             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-signed (query)          14_111.06       500.55    14_611.61       0.2152          1.0555         4.36
IVF-Binary-512-nl158-np7-rf20-signed (query)          14_111.06       574.30    14_685.36       0.3254          1.0358         4.36
IVF-Binary-512-nl158-np12-rf10-signed (query)         14_111.06       509.10    14_620.16       0.1976          1.0622         4.36
IVF-Binary-512-nl158-np12-rf20-signed (query)         14_111.06       611.89    14_722.95       0.3011          1.0401         4.36
IVF-Binary-512-nl158-np17-rf10-signed (query)         14_111.06       529.41    14_640.46       0.1846          1.0669         4.36
IVF-Binary-512-nl158-np17-rf20-signed (query)         14_111.06       610.77    14_721.82       0.2853          1.0428         4.36
IVF-Binary-512-nl158-signed (self)                    14_111.06     1_639.49    15_750.55       0.2006          1.0599         4.36
IVF-Binary-512-nl223-np11-rf0-signed (query)          12_192.59       432.78    12_625.37       0.0732             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-signed (query)          12_192.59       438.15    12_630.74       0.0682             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-signed (query)          12_192.59       447.26    12_639.85       0.0626             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-signed (query)         12_192.59       518.50    12_711.09       0.2087          1.0573         4.49
IVF-Binary-512-nl223-np11-rf20-signed (query)         12_192.59       589.25    12_781.84       0.3171          1.0368         4.49
IVF-Binary-512-nl223-np14-rf10-signed (query)         12_192.59       518.06    12_710.65       0.1981          1.0608         4.49
IVF-Binary-512-nl223-np14-rf20-signed (query)         12_192.59       597.78    12_790.37       0.3029          1.0390         4.49
IVF-Binary-512-nl223-np21-rf10-signed (query)         12_192.59       528.74    12_721.33       0.1855          1.0660         4.49
IVF-Binary-512-nl223-np21-rf20-signed (query)         12_192.59       613.81    12_806.40       0.2868          1.0422         4.49
IVF-Binary-512-nl223-signed (self)                    12_192.59     1_666.87    13_859.46       0.2014          1.0584         4.49
IVF-Binary-512-nl316-np15-rf0-signed (query)          12_301.13       446.77    12_747.90       0.0678             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-signed (query)          12_301.13       453.03    12_754.16       0.0662             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-signed (query)          12_301.13       456.40    12_757.53       0.0631             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-signed (query)         12_301.13       528.06    12_829.19       0.2012          1.0595         4.67
IVF-Binary-512-nl316-np15-rf20-signed (query)         12_301.13       606.03    12_907.16       0.3103          1.0377         4.67
IVF-Binary-512-nl316-np17-rf10-signed (query)         12_301.13       529.01    12_830.14       0.1971          1.0608         4.67
IVF-Binary-512-nl316-np17-rf20-signed (query)         12_301.13       612.72    12_913.85       0.3043          1.0387         4.67
IVF-Binary-512-nl316-np25-rf10-signed (query)         12_301.13       544.34    12_845.47       0.1878          1.0648         4.67
IVF-Binary-512-nl316-np25-rf20-signed (query)         12_301.13       622.87    12_924.00       0.2905          1.0414         4.67
IVF-Binary-512-nl316-signed (self)                    12_301.13     1_702.68    14_003.81       0.2005          1.0584         4.67
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
Exhaustive (query)                                        29.41    15_708.38    15_737.79       1.0000          1.0000       146.48
Exhaustive (self)                                         29.41    52_228.14    52_257.54       1.0000          1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)              8_804.14       486.06     9_290.20       0.0337             NaN         2.28
ExhaustiveBinary-256-random-rf10 (query)               8_804.14       623.36     9_427.50       0.1500          1.0692         2.28
ExhaustiveBinary-256-random-rf20 (query)               8_804.14       765.59     9_569.73       0.2428          1.0431         2.28
ExhaustiveBinary-256-random (self)                     8_804.14     2_080.91    10_885.05       0.1545          1.0644         2.28
ExhaustiveBinary-256-pca_no_rr (query)                 9_359.98       492.89     9_852.87       0.1247             NaN         2.28
ExhaustiveBinary-256-pca-rf10 (query)                  9_359.98       644.84    10_004.82       0.3430          1.0271         2.28
ExhaustiveBinary-256-pca-rf20 (query)                  9_359.98       808.83    10_168.82       0.4582          1.0177         2.28
ExhaustiveBinary-256-pca (self)                        9_359.98     2_138.68    11_498.66       0.3433          1.0271         2.28
ExhaustiveBinary-512-random_no_rr (query)             17_208.18       864.11    18_072.28       0.0628             NaN         4.55
ExhaustiveBinary-512-random-rf10 (query)              17_208.18       997.93    18_206.11       0.1726          1.0535         4.55
ExhaustiveBinary-512-random-rf20 (query)              17_208.18     1_147.31    18_355.49       0.2658          1.0348         4.55
ExhaustiveBinary-512-random (self)                    17_208.18     3_312.27    20_520.45       0.1745          1.0515         4.55
ExhaustiveBinary-512-pca_no_rr (query)                18_091.43       870.60    18_962.03       0.1450             NaN         4.55
ExhaustiveBinary-512-pca-rf10 (query)                 18_091.43     1_022.82    19_114.25       0.3791          1.0668         4.55
ExhaustiveBinary-512-pca-rf20 (query)                 18_091.43     1_175.12    19_266.55       0.4884          1.0163         4.55
ExhaustiveBinary-512-pca (self)                       18_091.43     3_398.89    21_490.31       0.3800          1.0959         4.55
ExhaustiveBinary-1024-random_no_rr (query)            34_285.25     1_596.10    35_881.34       0.0818             NaN         9.10
ExhaustiveBinary-1024-random-rf10 (query)             34_285.25     1_773.67    36_058.92       0.1931          1.0465         9.10
ExhaustiveBinary-1024-random-rf20 (query)             34_285.25     1_926.94    36_212.19       0.2959          1.0303         9.10
ExhaustiveBinary-1024-random (self)                   34_285.25     5_822.09    40_107.34       0.1941          1.0466         9.10
ExhaustiveBinary-1024-pca_no_rr (query)               35_337.29     1_622.09    36_959.38       0.2012             NaN         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                35_337.29     1_790.39    37_127.68       0.6300          1.0091         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                35_337.29     1_953.65    37_290.94       0.7963          1.0039         9.11
ExhaustiveBinary-1024-pca (self)                      35_337.29     5_949.85    41_287.14       0.6289          1.0092         9.11
ExhaustiveBinary-768-signed_no_rr (query)             25_795.01     1_274.72    27_069.72       0.0764             NaN         6.83
ExhaustiveBinary-768-signed-rf10 (query)              25_795.01     1_389.44    27_184.45       0.1848          1.0490         6.83
ExhaustiveBinary-768-signed-rf20 (query)              25_795.01     1_556.91    27_351.91       0.2828          1.0320         6.83
ExhaustiveBinary-768-signed (self)                    25_795.01     4_614.50    30_409.51       0.1853          1.0486         6.83
IVF-Binary-256-nl158-np7-rf0-random (query)           13_436.17       345.82    13_781.98       0.0556             NaN         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)          13_436.17       351.22    13_787.38       0.0441             NaN         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)          13_436.17       357.44    13_793.60       0.0368             NaN         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)          13_436.17       448.74    13_884.91       0.1932          1.0533         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)          13_436.17       545.07    13_981.24       0.2968          1.0337         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)         13_436.17       451.40    13_887.57       0.1705          1.0619         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)         13_436.17       552.29    13_988.46       0.2657          1.0390         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)         13_436.17       458.60    13_894.76       0.1566          1.0677         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)         13_436.17       567.82    14_003.99       0.2518          1.0419         2.74
IVF-Binary-256-nl158-random (self)                    13_436.17     1_436.34    14_872.51       0.1751          1.0575         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)           9_971.27       363.74    10_335.01       0.0536             NaN         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)           9_971.27       366.26    10_337.53       0.0445             NaN         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)           9_971.27       375.89    10_347.16       0.0375             NaN         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)          9_971.27       466.15    10_437.42       0.1878          1.0561         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)          9_971.27       564.10    10_535.37       0.2911          1.0348         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)          9_971.27       468.60    10_439.87       0.1715          1.0617         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)          9_971.27       578.64    10_549.91       0.2707          1.0382         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)          9_971.27       476.33    10_447.60       0.1577          1.0671         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)          9_971.27       589.99    10_561.26       0.2539          1.0410         2.93
IVF-Binary-256-nl223-random (self)                     9_971.27     1_470.07    11_441.34       0.1760          1.0574         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)          10_620.95       387.82    11_008.77       0.0461             NaN         3.20
IVF-Binary-256-nl316-np17-rf0-random (query)          10_620.95       387.06    11_008.02       0.0420             NaN         3.20
IVF-Binary-256-nl316-np25-rf0-random (query)          10_620.95       392.67    11_013.63       0.0386             NaN         3.20
IVF-Binary-256-nl316-np15-rf10-random (query)         10_620.95       490.26    11_111.21       0.1750          1.0599         3.20
IVF-Binary-256-nl316-np15-rf20-random (query)         10_620.95       589.66    11_210.62       0.2772          1.0369         3.20
IVF-Binary-256-nl316-np17-rf10-random (query)         10_620.95       488.79    11_109.74       0.1675          1.0628         3.20
IVF-Binary-256-nl316-np17-rf20-random (query)         10_620.95       599.08    11_220.04       0.2674          1.0386         3.20
IVF-Binary-256-nl316-np25-rf10-random (query)         10_620.95       495.44    11_116.39       0.1595          1.0663         3.20
IVF-Binary-256-nl316-np25-rf20-random (query)         10_620.95       608.70    11_229.66       0.2556          1.0411         3.20
IVF-Binary-256-nl316-random (self)                    10_620.95     1_573.22    12_194.17       0.1713          1.0586         3.20
IVF-Binary-256-nl158-np7-rf0-pca (query)              14_014.51       354.93    14_369.45       0.1301             NaN         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)             14_014.51       359.73    14_374.24       0.1289             NaN         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)             14_014.51       366.28    14_380.79       0.1285             NaN         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)             14_014.51       468.88    14_483.39       0.4031          1.0217         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)             14_014.51       571.41    14_585.92       0.5652          1.0121         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)            14_014.51       474.85    14_489.36       0.3946          1.0224         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)            14_014.51       583.75    14_598.26       0.5509          1.0127         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)            14_014.51       485.68    14_500.19       0.3918          1.0226         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)            14_014.51       602.48    14_616.99       0.5457          1.0130         2.74
IVF-Binary-256-nl158-pca (self)                       14_014.51     1_501.10    15_515.61       0.3948          1.0225         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)             10_632.35       372.18    11_004.53       0.1298             NaN         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)             10_632.35       375.54    11_007.90       0.1290             NaN         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)             10_632.35       382.58    11_014.93       0.1284             NaN         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)            10_632.35       486.56    11_118.91       0.4010          1.0218         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)            10_632.35       593.51    11_225.86       0.5605          1.0122         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)            10_632.35       489.27    11_121.62       0.3955          1.0223         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)            10_632.35       600.25    11_232.60       0.5523          1.0126         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)            10_632.35       503.75    11_136.10       0.3911          1.0227         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)            10_632.35       646.64    11_278.99       0.5450          1.0130         2.93
IVF-Binary-256-nl223-pca (self)                       10_632.35     1_567.48    12_199.84       0.3954          1.0224         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)             11_275.92       394.59    11_670.51       0.1294             NaN         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)             11_275.92       396.85    11_672.77       0.1290             NaN         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)             11_275.92       401.39    11_677.32       0.1284             NaN         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)            11_275.92       511.34    11_787.27       0.4000          1.0219         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)            11_275.92       617.65    11_893.57       0.5599          1.0123         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)            11_275.92       512.03    11_787.95       0.3974          1.0222         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)            11_275.92       624.87    11_900.80       0.5557          1.0125         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)            11_275.92       519.89    11_795.81       0.3927          1.0226         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)            11_275.92       631.24    11_907.16       0.5475          1.0129         3.21
IVF-Binary-256-nl316-pca (self)                       11_275.92     1_625.86    12_901.78       0.3974          1.0222         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)           21_911.39       646.63    22_558.02       0.0741             NaN         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)          21_911.39       655.86    22_567.25       0.0687             NaN         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)          21_911.39       661.63    22_573.01       0.0646             NaN         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)          21_911.39       749.20    22_660.59       0.1979          1.0471         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)          21_911.39       844.60    22_755.99       0.3014          1.0307         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)         21_911.39       756.17    22_667.56       0.1826          1.0511         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)         21_911.39       857.75    22_769.14       0.2793          1.0333         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)         21_911.39       767.31    22_678.70       0.1755          1.0530         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)         21_911.39       869.58    22_780.97       0.2711          1.0344         5.02
IVF-Binary-512-nl158-random (self)                    21_911.39     2_442.22    24_353.61       0.1849          1.0493         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)          18_537.05       664.43    19_201.49       0.0731             NaN         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)          18_537.05       668.29    19_205.34       0.0683             NaN         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)          18_537.05       688.54    19_225.59       0.0649             NaN         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)         18_537.05       769.11    19_306.16       0.1960          1.0473         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)         18_537.05       874.25    19_411.31       0.2978          1.0306         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)         18_537.05       832.39    19_369.44       0.1858          1.0500         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)         18_537.05       871.73    19_408.79       0.2836          1.0324         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)         18_537.05       786.37    19_323.43       0.1772          1.0524         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)         18_537.05       891.93    19_428.99       0.2725          1.0340         5.21
IVF-Binary-512-nl223-random (self)                    18_537.05     2_495.41    21_032.47       0.1878          1.0482         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)          19_167.93       688.70    19_856.63       0.0696             NaN         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)          19_167.93       697.19    19_865.11       0.0676             NaN         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)          19_167.93       697.39    19_865.31       0.0660             NaN         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)         19_167.93       793.37    19_961.29       0.1897          1.0484         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)         19_167.93       892.58    20_060.51       0.2906          1.0313         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)         19_167.93       793.46    19_961.38       0.1846          1.0498         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)         19_167.93       899.38    20_067.31       0.2838          1.0322         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)         19_167.93       803.70    19_971.63       0.1785          1.0519         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)         19_167.93       909.84    20_077.76       0.2740          1.0337         5.48
IVF-Binary-512-nl316-random (self)                    19_167.93     2_567.92    21_735.84       0.1870          1.0482         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)              22_678.40       660.31    23_338.70       0.1701             NaN         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)             22_678.40       670.28    23_348.68       0.1677             NaN         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)             22_678.40       677.51    23_355.91       0.1666             NaN         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)             22_678.40       772.04    23_450.44       0.5298          1.0137         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)             22_678.40       870.58    23_548.98       0.7010          1.0067         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)            22_678.40       780.46    23_458.86       0.5150          1.0145         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)            22_678.40       886.72    23_565.11       0.6802          1.0074         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)            22_678.40       793.54    23_471.93       0.5059          1.0150         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)            22_678.40       903.07    23_581.47       0.6665          1.0078         5.02
IVF-Binary-512-nl158-pca (self)                       22_678.40     2_693.36    25_371.75       0.5164          1.0146         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)             19_574.79       681.36    20_256.15       0.1692             NaN         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)             19_574.79       684.19    20_258.98       0.1676             NaN         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)             19_574.79       694.60    20_269.39       0.1660             NaN         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)            19_574.79       797.30    20_372.09       0.5239          1.0140         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)            19_574.79       895.43    20_470.22       0.6925          1.0069         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)            19_574.79       803.12    20_377.91       0.5159          1.0144         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)            19_574.79       904.76    20_479.55       0.6817          1.0073         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)            19_574.79       812.28    20_387.07       0.5053          1.0150         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)            19_574.79       930.16    20_504.94       0.6656          1.0078         5.21
IVF-Binary-512-nl223-pca (self)                       19_574.79     2_587.68    22_162.47       0.5170          1.0145         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)             19_946.90       702.65    20_649.56       0.1691             NaN         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)             19_946.90       715.07    20_661.97       0.1683             NaN         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)             19_946.90       774.87    20_721.78       0.1668             NaN         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)            19_946.90       829.96    20_776.86       0.5242          1.0140         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)            19_946.90       925.39    20_872.30       0.6932          1.0069         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)            19_946.90       829.78    20_776.69       0.5203          1.0142         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)            19_946.90       930.19    20_877.09       0.6881          1.0071         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)            19_946.90       829.30    20_776.20       0.5102          1.0148         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)            19_946.90       940.65    20_887.56       0.6733          1.0076         5.48
IVF-Binary-512-nl316-pca (self)                       19_946.90     2_652.26    22_599.17       0.5214          1.0142         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)          39_012.37     1_255.49    40_267.86       0.0854             NaN         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)         39_012.37     1_276.33    40_288.70       0.0832             NaN         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)         39_012.37     1_290.06    40_302.43       0.0820             NaN         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)         39_012.37     1_353.84    40_366.21       0.2106          1.0431         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)         39_012.37     1_463.53    40_475.90       0.3205          1.0279         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)        39_012.37     1_372.55    40_384.92       0.1993          1.0454         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)        39_012.37     1_470.44    40_482.81       0.3029          1.0297         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)        39_012.37     1_381.73    40_394.10       0.1956          1.0461         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)        39_012.37     1_485.15    40_497.52       0.2986          1.0301         9.57
IVF-Binary-1024-nl158-random (self)                   39_012.37     4_521.68    43_534.05       0.2004          1.0454         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)         35_618.93     1_275.67    36_894.60       0.0848             NaN         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)         35_618.93     1_294.03    36_912.96       0.0834             NaN         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)         35_618.93     1_293.77    36_912.70       0.0820             NaN         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)        35_618.93     1_373.77    36_992.70       0.2091          1.0431         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)        35_618.93     1_474.79    37_093.72       0.3186          1.0278         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)        35_618.93     1_378.66    36_997.59       0.2021          1.0446         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)        35_618.93     1_480.94    37_099.87       0.3071          1.0291         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)        35_618.93     1_405.22    37_024.15       0.1966          1.0459         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)        35_618.93     1_523.88    37_142.81       0.3000          1.0299         9.76
IVF-Binary-1024-nl223-random (self)                   35_618.93     4_560.74    40_179.67       0.2028          1.0447         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)         36_270.73     1_296.58    37_567.32       0.0838             NaN        10.03
IVF-Binary-1024-nl316-np17-rf0-random (query)         36_270.73     1_299.35    37_570.08       0.0830             NaN        10.03
IVF-Binary-1024-nl316-np25-rf0-random (query)         36_270.73     1_307.85    37_578.58       0.0823             NaN        10.03
IVF-Binary-1024-nl316-np15-rf10-random (query)        36_270.73     1_397.73    37_668.46       0.2051          1.0437        10.03
IVF-Binary-1024-nl316-np15-rf20-random (query)        36_270.73     1_500.09    37_770.83       0.3145          1.0282        10.03
IVF-Binary-1024-nl316-np17-rf10-random (query)        36_270.73     1_409.94    37_680.67       0.2016          1.0445        10.03
IVF-Binary-1024-nl316-np17-rf20-random (query)        36_270.73     1_503.37    37_774.11       0.3083          1.0289        10.03
IVF-Binary-1024-nl316-np25-rf10-random (query)        36_270.73     1_421.05    37_691.79       0.1970          1.0458        10.03
IVF-Binary-1024-nl316-np25-rf20-random (query)        36_270.73     1_524.31    37_795.04       0.3009          1.0299        10.03
IVF-Binary-1024-nl316-random (self)                   36_270.73     4_588.07    40_858.81       0.2020          1.0447        10.03
IVF-Binary-1024-nl158-np7-rf0-pca (query)             40_124.00     1_286.19    41_410.19       0.2021             NaN         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)            40_124.00     1_294.26    41_418.26       0.2012             NaN         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)            40_124.00     1_300.94    41_424.94       0.2012             NaN         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)            40_124.00     1_398.07    41_522.07       0.6336          1.0090         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)            40_124.00     1_481.68    41_605.68       0.8010          1.0037         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)           40_124.00     1_398.81    41_522.81       0.6299          1.0091         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)           40_124.00     1_501.50    41_625.50       0.7966          1.0039         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)           40_124.00     1_423.38    41_547.38       0.6298          1.0091         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)           40_124.00     1_523.75    41_647.75       0.7965          1.0039         9.57
IVF-Binary-1024-nl158-pca (self)                      40_124.00     4_591.27    44_715.27       0.6289          1.0092         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)            36_687.27     1_301.63    37_988.90       0.2023             NaN         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)            36_687.27     1_318.31    38_005.58       0.2015             NaN         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)            36_687.27     1_320.00    38_007.27       0.2012             NaN         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)           36_687.27     1_417.55    38_104.82       0.6342          1.0089         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)           36_687.27     1_521.14    38_208.41       0.8013          1.0037         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)           36_687.27     1_415.55    38_102.82       0.6311          1.0091         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)           36_687.27     1_517.51    38_204.78       0.7977          1.0038         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)           36_687.27     1_450.22    38_137.49       0.6302          1.0091         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)           36_687.27     1_582.03    38_269.30       0.7967          1.0039         9.76
IVF-Binary-1024-nl223-pca (self)                      36_687.27     4_665.87    41_353.14       0.6299          1.0092         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)            37_295.32     1_322.50    38_617.81       0.2021             NaN        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)            37_295.32     1_323.82    38_619.14       0.2016             NaN        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)            37_295.32     1_334.05    38_629.36       0.2012             NaN        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)           37_295.32     1_440.54    38_735.86       0.6333          1.0090        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)           37_295.32     1_530.53    38_825.85       0.8004          1.0037        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)           37_295.32     1_429.66    38_724.97       0.6316          1.0091        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)           37_295.32     1_536.59    38_831.91       0.7987          1.0038        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)           37_295.32     1_445.96    38_741.27       0.6298          1.0091        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)           37_295.32     1_578.29    38_873.61       0.7965          1.0039        10.04
IVF-Binary-1024-nl316-pca (self)                      37_295.32     4_695.48    41_990.80       0.6306          1.0091        10.04
IVF-Binary-768-nl158-np7-rf0-signed (query)           30_601.64       949.95    31_551.60       0.0817             NaN         7.29
IVF-Binary-768-nl158-np12-rf0-signed (query)          30_601.64       960.90    31_562.55       0.0786             NaN         7.29
IVF-Binary-768-nl158-np17-rf0-signed (query)          30_601.64       972.12    31_573.77       0.0768             NaN         7.29
IVF-Binary-768-nl158-np7-rf10-signed (query)          30_601.64     1_051.88    31_653.52       0.2051          1.0446         7.29
IVF-Binary-768-nl158-np7-rf20-signed (query)          30_601.64     1_150.60    31_752.24       0.3112          1.0290         7.29
IVF-Binary-768-nl158-np12-rf10-signed (query)         30_601.64     1_062.90    31_664.54       0.1925          1.0474         7.29
IVF-Binary-768-nl158-np12-rf20-signed (query)         30_601.64     1_182.26    31_783.90       0.2920          1.0311         7.29
IVF-Binary-768-nl158-np17-rf10-signed (query)         30_601.64     1_076.42    31_678.07       0.1875          1.0486         7.29
IVF-Binary-768-nl158-np17-rf20-signed (query)         30_601.64     1_181.04    31_782.69       0.2861          1.0318         7.29
IVF-Binary-768-nl158-signed (self)                    30_601.64     3_461.41    34_063.06       0.1931          1.0471         7.29
IVF-Binary-768-nl223-np11-rf0-signed (query)          27_130.16       969.09    28_099.25       0.0815             NaN         7.48
IVF-Binary-768-nl223-np14-rf0-signed (query)          27_130.16       981.30    28_111.47       0.0795             NaN         7.48
IVF-Binary-768-nl223-np21-rf0-signed (query)          27_130.16       990.34    28_120.50       0.0776             NaN         7.48
IVF-Binary-768-nl223-np11-rf10-signed (query)         27_130.16     1_071.85    28_202.01       0.2028          1.0447         7.48
IVF-Binary-768-nl223-np11-rf20-signed (query)         27_130.16     1_171.07    28_301.23       0.3086          1.0290         7.48
IVF-Binary-768-nl223-np14-rf10-signed (query)         27_130.16     1_105.06    28_235.22       0.1947          1.0467         7.48
IVF-Binary-768-nl223-np14-rf20-signed (query)         27_130.16     1_193.12    28_323.28       0.2966          1.0305         7.48
IVF-Binary-768-nl223-np21-rf10-signed (query)         27_130.16     1_095.08    28_225.24       0.1879          1.0484         7.48
IVF-Binary-768-nl223-np21-rf20-signed (query)         27_130.16     1_231.24    28_361.40       0.2878          1.0316         7.48
IVF-Binary-768-nl223-signed (self)                    27_130.16     3_518.56    30_648.72       0.1953          1.0464         7.48
IVF-Binary-768-nl316-np15-rf0-signed (query)          27_763.98       992.87    28_756.85       0.0801             NaN         7.76
IVF-Binary-768-nl316-np17-rf0-signed (query)          27_763.98     1_001.42    28_765.40       0.0790             NaN         7.76
IVF-Binary-768-nl316-np25-rf0-signed (query)          27_763.98     1_005.02    28_768.99       0.0778             NaN         7.76
IVF-Binary-768-nl316-np15-rf10-signed (query)         27_763.98     1_097.10    28_861.07       0.1985          1.0455         7.76
IVF-Binary-768-nl316-np15-rf20-signed (query)         27_763.98     1_195.44    28_959.42       0.3033          1.0295         7.76
IVF-Binary-768-nl316-np17-rf10-signed (query)         27_763.98     1_096.63    28_860.61       0.1943          1.0465         7.76
IVF-Binary-768-nl316-np17-rf20-signed (query)         27_763.98     1_201.72    28_965.70       0.2969          1.0303         7.76
IVF-Binary-768-nl316-np25-rf10-signed (query)         27_763.98     1_120.40    28_884.38       0.1889          1.0481         7.76
IVF-Binary-768-nl316-np25-rf20-signed (query)         27_763.98     1_216.71    28_980.69       0.2887          1.0315         7.76
IVF-Binary-768-nl316-signed (self)                    27_763.98     3_578.73    31_342.71       0.1949          1.0463         7.76
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
Exhaustive (query)                                         9.86     4_152.55     4_162.41       1.0000          1.0000        48.83
Exhaustive (self)                                          9.86    13_754.54    13_764.41       1.0000          1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_589.19       272.69     2_861.88       0.0875             NaN         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_589.19       379.66     2_968.85       0.3366          1.1470         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_589.19       469.96     3_059.15       0.4794          1.0850         1.78
ExhaustiveBinary-256-random (self)                     2_589.19     1_225.81     3_814.99       0.3636          1.1515         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_777.59       278.95     3_056.53       0.1109             NaN         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_777.59       386.97     3_164.56       0.3158          1.5913         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_777.59       489.14     3_266.73       0.4283          1.3071         1.78
ExhaustiveBinary-256-pca (self)                        2_777.59     1_280.57     4_058.15       0.2950          2.2349         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_074.87       437.42     5_512.29       0.1342             NaN         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_074.87       536.49     5_611.35       0.4316          1.0992         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_074.87       637.42     5_712.29       0.5828          1.0538         3.55
ExhaustiveBinary-512-random (self)                     5_074.87     1_769.73     6_844.60       0.4551          1.1054         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_408.19       441.23     5_849.43       0.1283             NaN         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_408.19       550.75     5_958.94       0.4093          1.1233         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_408.19       656.48     6_064.67       0.5766          1.0656         3.55
ExhaustiveBinary-512-pca (self)                        5_408.19     1_827.59     7_235.78       0.4109          1.1443         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_027.38       757.14    10_784.52       0.1930             NaN         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_027.38       861.05    10_888.43       0.5442          1.0617         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_027.38       968.64    10_996.02       0.7015          1.0305         7.10
ExhaustiveBinary-1024-random (self)                   10_027.38     3_063.44    13_090.82       0.5708          1.0663         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_526.04       769.63    11_295.67       0.1535             NaN         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_526.04       884.57    11_410.61       0.4703          1.0875         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_526.04       998.50    11_524.55       0.6408          1.0449         7.10
ExhaustiveBinary-1024-pca (self)                      10_526.04     2_943.13    13_469.17       0.4638          1.1066         7.10
ExhaustiveBinary-256-signed_no_rr (query)              2_589.38       272.42     2_861.81       0.0875             NaN         1.78
ExhaustiveBinary-256-signed-rf10 (query)               2_589.38       372.16     2_961.55       0.3366          1.1470         1.78
ExhaustiveBinary-256-signed-rf20 (query)               2_589.38       469.77     3_059.15       0.4794          1.0850         1.78
ExhaustiveBinary-256-signed (self)                     2_589.38     1_228.22     3_817.61       0.3636          1.1515         1.78
IVF-Binary-256-nl158-np7-rf0-random (query)            3_929.80       111.82     4_041.62       0.0905             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           3_929.80       113.20     4_043.00       0.0889             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           3_929.80       116.29     4_046.09       0.0879             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           3_929.80       166.95     4_096.75       0.3414          1.1450         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           3_929.80       217.75     4_147.55       0.4835          1.0837         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          3_929.80       170.63     4_100.43       0.3391          1.1459         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          3_929.80       221.82     4_151.62       0.4816          1.0842         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          3_929.80       176.42     4_106.22       0.3378          1.1466         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          3_929.80       230.12     4_159.92       0.4805          1.0846         1.93
IVF-Binary-256-nl158-random (self)                     3_929.80       509.54     4_439.34       0.3662          1.1503         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           3_184.93       116.44     3_301.37       0.0949             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           3_184.93       120.52     3_305.45       0.0906             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           3_184.93       123.41     3_308.34       0.0885             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          3_184.93       173.86     3_358.79       0.3513          1.1397         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          3_184.93       221.90     3_406.83       0.4946          1.0802         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          3_184.93       175.02     3_359.95       0.3420          1.1459         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          3_184.93       227.93     3_412.86       0.4848          1.0840         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          3_184.93       183.73     3_368.66       0.3385          1.1480         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          3_184.93       238.60     3_423.52       0.4816          1.0850         2.00
IVF-Binary-256-nl223-random (self)                     3_184.93       529.30     3_714.23       0.3692          1.1507         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           3_412.67       123.19     3_535.86       0.0923             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           3_412.67       123.92     3_536.59       0.0905             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           3_412.67       129.27     3_541.93       0.0892             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          3_412.67       179.66     3_592.33       0.3480          1.1422         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          3_412.67       229.71     3_642.38       0.4900          1.0824         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          3_412.67       179.13     3_591.80       0.3442          1.1442         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          3_412.67       236.84     3_649.50       0.4865          1.0835         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          3_412.67       186.92     3_599.58       0.3411          1.1460         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          3_412.67       238.06     3_650.73       0.4830          1.0845         2.09
IVF-Binary-256-nl316-random (self)                     3_412.67       544.08     3_956.74       0.3712          1.1489         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               4_091.18       116.03     4_207.21       0.1195             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              4_091.18       118.22     4_209.40       0.1157             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              4_091.18       121.02     4_212.20       0.1148             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              4_091.18       194.62     4_285.80       0.3989          1.1281         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              4_091.18       237.66     4_328.84       0.5690          1.0681         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             4_091.18       186.53     4_277.71       0.3772          1.1547         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             4_091.18       244.01     4_335.20       0.5415          1.0820         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             4_091.18       192.77     4_283.95       0.3703          1.1750         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             4_091.18       250.66     4_341.84       0.5309          1.0927         1.93
IVF-Binary-256-nl158-pca (self)                        4_091.18       578.04     4_669.22       0.3839          1.1736         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              3_380.75       124.09     3_504.84       0.1167             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              3_380.75       124.84     3_505.59       0.1158             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              3_380.75       128.45     3_509.20       0.1147             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             3_380.75       190.02     3_570.77       0.3791          1.1456         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             3_380.75       248.53     3_629.27       0.5456          1.0767         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             3_380.75       209.21     3_589.96       0.3742          1.1576         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             3_380.75       249.66     3_630.41       0.5380          1.0833         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             3_380.75       205.11     3_585.85       0.3676          1.1786         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             3_380.75       262.69     3_643.44       0.5266          1.0949         2.00
IVF-Binary-256-nl223-pca (self)                        3_380.75       599.33     3_980.08       0.3819          1.1751         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              3_604.96       128.79     3_733.75       0.1164             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              3_604.96       130.33     3_735.29       0.1159             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              3_604.96       134.68     3_739.64       0.1148             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             3_604.96       195.98     3_800.94       0.3784          1.1471         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             3_604.96       250.65     3_855.61       0.5447          1.0773         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             3_604.96       196.10     3_801.06       0.3760          1.1519         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             3_604.96       252.38     3_857.34       0.5408          1.0803         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             3_604.96       206.11     3_811.07       0.3694          1.1712         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             3_604.96       261.85     3_866.81       0.5297          1.0908         2.09
IVF-Binary-256-nl316-pca (self)                        3_604.96       611.25     4_216.21       0.3836          1.1697         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_361.97       201.56     6_563.53       0.1359             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_361.97       204.69     6_566.66       0.1353             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_361.97       209.82     6_571.79       0.1348             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_361.97       259.96     6_621.93       0.4333          1.0987         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_361.97       310.34     6_672.31       0.5845          1.0534         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_361.97       263.74     6_625.71       0.4325          1.0989         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_361.97       317.03     6_679.00       0.5836          1.0536         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_361.97       272.20     6_634.17       0.4321          1.0990         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_361.97       325.91     6_687.89       0.5833          1.0536         3.71
IVF-Binary-512-nl158-random (self)                     6_361.97       827.36     7_189.33       0.4561          1.1050         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_636.18       208.77     5_844.95       0.1394             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_636.18       209.80     5_845.98       0.1362             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_636.18       216.78     5_852.96       0.1350             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_636.18       267.84     5_904.02       0.4391          1.0962         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_636.18       316.38     5_952.56       0.5887          1.0525         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_636.18       271.02     5_907.20       0.4342          1.0985         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_636.18       321.85     5_958.03       0.5848          1.0535         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_636.18       279.12     5_915.31       0.4321          1.0993         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_636.18       337.03     5_973.21       0.5829          1.0539         3.77
IVF-Binary-512-nl223-random (self)                     5_636.18       845.05     6_481.24       0.4579          1.1048         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           5_854.68       214.30     6_068.98       0.1373             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           5_854.68       214.46     6_069.14       0.1363             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           5_854.68       220.98     6_075.66       0.1354             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          5_854.68       274.33     6_129.01       0.4377          1.0972         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          5_854.68       322.99     6_177.67       0.5881          1.0527         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          5_854.68       275.91     6_130.60       0.4355          1.0980         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          5_854.68       326.04     6_180.73       0.5863          1.0531         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          5_854.68       282.51     6_137.20       0.4336          1.0987         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          5_854.68       335.43     6_190.12       0.5845          1.0535         3.86
IVF-Binary-512-nl316-random (self)                     5_854.68       859.83     6_714.51       0.4598          1.1039         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               6_678.72       209.96     6_888.68       0.1322             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              6_678.72       212.86     6_891.57       0.1292             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              6_678.72       217.09     6_895.81       0.1286             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              6_678.72       297.42     6_976.14       0.4307          1.1079         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              6_678.72       331.03     7_009.75       0.6023          1.0562         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             6_678.72       282.72     6_961.44       0.4144          1.1177         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             6_678.72       336.74     7_015.46       0.5839          1.0618         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             6_678.72       289.60     6_968.32       0.4108          1.1209         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             6_678.72       344.92     7_023.64       0.5792          1.0638         3.71
IVF-Binary-512-nl158-pca (self)                        6_678.72       897.42     7_576.14       0.4163          1.1372         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              5_965.75       216.75     6_182.51       0.1299             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              5_965.75       221.21     6_186.96       0.1293             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              5_965.75       240.32     6_206.07       0.1288             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             5_965.75       284.28     6_250.03       0.4156          1.1155         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             5_965.75       337.97     6_303.72       0.5849          1.0605         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             5_965.75       287.50     6_253.25       0.4129          1.1181         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             5_965.75       344.09     6_309.84       0.5816          1.0621         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             5_965.75       296.78     6_262.53       0.4106          1.1207         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             5_965.75       362.65     6_328.40       0.5785          1.0637         3.77
IVF-Binary-512-nl223-pca (self)                        5_965.75       912.52     6_878.27       0.4149          1.1373         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              6_172.05       222.81     6_394.87       0.1297             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              6_172.05       224.21     6_396.26       0.1294             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              6_172.05       229.35     6_401.40       0.1288             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             6_172.05       290.39     6_462.44       0.4147          1.1162         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             6_172.05       344.64     6_516.69       0.5843          1.0608         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             6_172.05       291.91     6_463.96       0.4135          1.1174         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             6_172.05       345.96     6_518.02       0.5825          1.0616         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             6_172.05       297.53     6_469.58       0.4107          1.1202         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             6_172.05       369.99     6_542.05       0.5789          1.0633         3.86
IVF-Binary-512-nl316-pca (self)                        6_172.05       922.45     7_094.50       0.4156          1.1365         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_274.06       383.86    11_657.92       0.1940             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_274.06       388.44    11_662.50       0.1935             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_274.06       393.40    11_667.46       0.1933             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_274.06       445.51    11_719.57       0.5448          1.0615         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_274.06       504.91    11_778.98       0.7021          1.0305         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_274.06       449.96    11_724.02       0.5445          1.0616         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_274.06       511.19    11_785.26       0.7018          1.0305         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_274.06       459.31    11_733.37       0.5444          1.0617         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_274.06       516.40    11_790.47       0.7017          1.0305         7.26
IVF-Binary-1024-nl158-random (self)                   11_274.06     1_445.79    12_719.85       0.5712          1.0662         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_564.01       390.59    10_954.60       0.1959             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_564.01       393.73    10_957.74       0.1943             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_564.01       404.24    10_968.25       0.1937             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_564.01       453.41    11_017.42       0.5475          1.0609         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_564.01       502.95    11_066.96       0.7042          1.0301         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_564.01       456.18    11_020.19       0.5456          1.0614         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_564.01       509.98    11_073.99       0.7024          1.0304         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_564.01       473.60    11_037.60       0.5447          1.0616         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_564.01       529.43    11_093.44       0.7016          1.0305         7.32
IVF-Binary-1024-nl223-random (self)                   10_564.01     1_464.33    12_028.34       0.5725          1.0659         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         10_764.01       397.46    11_161.47       0.1948             NaN         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)         10_764.01       400.56    11_164.56       0.1943             NaN         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)         10_764.01       408.68    11_172.69       0.1936             NaN         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)        10_764.01       459.50    11_223.51       0.5477          1.0609         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)        10_764.01       509.54    11_273.55       0.7042          1.0302         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)        10_764.01       457.86    11_221.87       0.5466          1.0611         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)        10_764.01       512.65    11_276.66       0.7032          1.0303         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)        10_764.01       468.73    11_232.74       0.5454          1.0615         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)        10_764.01       530.20    11_294.21       0.7022          1.0305         7.41
IVF-Binary-1024-nl316-random (self)                   10_764.01     1_475.49    12_239.49       0.5729          1.0658         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)             11_782.98       398.39    12_181.37       0.1551             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            11_782.98       403.79    12_186.77       0.1537             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            11_782.98       410.30    12_193.28       0.1536             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            11_782.98       465.72    12_248.70       0.4828          1.0838         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            11_782.98       519.35    12_302.33       0.6531          1.0425         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           11_782.98       470.33    12_253.31       0.4723          1.0867         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           11_782.98       526.34    12_309.32       0.6435          1.0443         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           11_782.98       478.96    12_261.94       0.4709          1.0871         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           11_782.98       538.80    12_321.78       0.6414          1.0447         7.26
IVF-Binary-1024-nl158-pca (self)                      11_782.98     1_627.10    13_410.08       0.4658          1.1059         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            11_097.42       424.25    11_521.67       0.1544             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            11_097.42       408.32    11_505.74       0.1541             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            11_097.42       422.62    11_520.05       0.1540             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           11_097.42       477.68    11_575.10       0.4734          1.0860         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           11_097.42       526.16    11_623.59       0.6441          1.0440         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           11_097.42       476.99    11_574.41       0.4721          1.0865         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           11_097.42       533.74    11_631.17       0.6425          1.0443         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           11_097.42       492.50    11_589.92       0.4712          1.0868         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           11_097.42       551.19    11_648.61       0.6414          1.0446         7.32
IVF-Binary-1024-nl223-pca (self)                      11_097.42     1_549.14    12_646.56       0.4653          1.1056         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            11_304.25       412.14    11_716.39       0.1543             NaN         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            11_304.25       420.31    11_724.56       0.1542             NaN         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            11_304.25       424.30    11_728.55       0.1539             NaN         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           11_304.25       483.66    11_787.91       0.4729          1.0862         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           11_304.25       539.83    11_844.08       0.6437          1.0440         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           11_304.25       479.33    11_783.58       0.4722          1.0865         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           11_304.25       538.33    11_842.58       0.6428          1.0443         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           11_304.25       493.02    11_797.27       0.4710          1.0869         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           11_304.25       550.08    11_854.33       0.6415          1.0445         7.42
IVF-Binary-1024-nl316-pca (self)                      11_304.25     1_558.51    12_862.76       0.4656          1.1056         7.42
IVF-Binary-256-nl158-np7-rf0-signed (query)            3_879.04       114.20     3_993.25       0.0905             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-signed (query)           3_879.04       114.39     3_993.43       0.0889             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-signed (query)           3_879.04       116.67     3_995.71       0.0879             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-signed (query)           3_879.04       168.53     4_047.57       0.3414          1.1450         1.93
IVF-Binary-256-nl158-np7-rf20-signed (query)           3_879.04       218.31     4_097.36       0.4835          1.0837         1.93
IVF-Binary-256-nl158-np12-rf10-signed (query)          3_879.04       170.39     4_049.44       0.3391          1.1459         1.93
IVF-Binary-256-nl158-np12-rf20-signed (query)          3_879.04       221.91     4_100.95       0.4816          1.0842         1.93
IVF-Binary-256-nl158-np17-rf10-signed (query)          3_879.04       176.77     4_055.81       0.3378          1.1466         1.93
IVF-Binary-256-nl158-np17-rf20-signed (query)          3_879.04       232.06     4_111.10       0.4805          1.0846         1.93
IVF-Binary-256-nl158-signed (self)                     3_879.04       513.25     4_392.30       0.3662          1.1503         1.93
IVF-Binary-256-nl223-np11-rf0-signed (query)           3_184.94       118.02     3_302.96       0.0949             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-signed (query)           3_184.94       118.48     3_303.42       0.0906             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-signed (query)           3_184.94       124.53     3_309.48       0.0885             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-signed (query)          3_184.94       173.60     3_358.54       0.3513          1.1397         2.00
IVF-Binary-256-nl223-np11-rf20-signed (query)          3_184.94       221.93     3_406.87       0.4946          1.0802         2.00
IVF-Binary-256-nl223-np14-rf10-signed (query)          3_184.94       175.04     3_359.98       0.3420          1.1459         2.00
IVF-Binary-256-nl223-np14-rf20-signed (query)          3_184.94       227.56     3_412.51       0.4848          1.0840         2.00
IVF-Binary-256-nl223-np21-rf10-signed (query)          3_184.94       183.53     3_368.48       0.3385          1.1480         2.00
IVF-Binary-256-nl223-np21-rf20-signed (query)          3_184.94       239.02     3_423.97       0.4816          1.0850         2.00
IVF-Binary-256-nl223-signed (self)                     3_184.94       528.13     3_713.07       0.3692          1.1507         2.00
IVF-Binary-256-nl316-np15-rf0-signed (query)           3_408.40       134.42     3_542.82       0.0923             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-signed (query)           3_408.40       138.32     3_546.72       0.0905             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-signed (query)           3_408.40       154.16     3_562.57       0.0892             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-signed (query)          3_408.40       218.46     3_626.86       0.3480          1.1422         2.09
IVF-Binary-256-nl316-np15-rf20-signed (query)          3_408.40       265.45     3_673.85       0.4900          1.0824         2.09
IVF-Binary-256-nl316-np17-rf10-signed (query)          3_408.40       231.07     3_639.47       0.3442          1.1442         2.09
IVF-Binary-256-nl316-np17-rf20-signed (query)          3_408.40       297.39     3_705.79       0.4865          1.0835         2.09
IVF-Binary-256-nl316-np25-rf10-signed (query)          3_408.40       208.05     3_616.45       0.3411          1.1460         2.09
IVF-Binary-256-nl316-np25-rf20-signed (query)          3_408.40       267.09     3_675.50       0.4830          1.0845         2.09
IVF-Binary-256-nl316-signed (self)                     3_408.40       615.97     4_024.38       0.3712          1.1489         2.09
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
Exhaustive (query)                                        20.23     9_594.75     9_614.98       1.0000          1.0000        97.66
Exhaustive (self)                                         20.23    32_389.12    32_409.35       1.0000          1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_635.89       376.12     6_012.01       0.0610             NaN         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_635.89       493.74     6_129.63       0.2580          1.1443         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_635.89       616.35     6_252.24       0.3781          1.0904         2.03
ExhaustiveBinary-256-random (self)                     5_635.89     1_621.21     7_257.10       0.2746          1.1471         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_101.43       386.61     6_488.04       0.1754             NaN         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_101.43       518.87     6_620.30       0.4664          1.2428         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_101.43       648.67     6_750.10       0.5850          1.1744         2.03
ExhaustiveBinary-256-pca (self)                        6_101.43     1_738.42     7_839.85       0.4693          1.2512         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_170.52       653.81    11_824.33       0.0950             NaN         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_170.52       772.08    11_942.59       0.3259          1.1053         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_170.52       896.73    12_067.25       0.4522          1.0627         4.05
ExhaustiveBinary-512-random (self)                    11_170.52     2_535.47    13_705.99       0.3417          1.1072         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_834.41       658.16    12_492.57       0.1672             NaN         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_834.41       793.53    12_627.94       0.4213          1.4308         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_834.41       930.37    12_764.78       0.5297          1.2408         4.05
ExhaustiveBinary-512-pca (self)                       11_834.41     2_622.83    14_457.25       0.4104          1.6976         4.05
ExhaustiveBinary-1024-random_no_rr (query)            21_889.48     1_177.55    23_067.03       0.1421             NaN         8.10
ExhaustiveBinary-1024-random-rf10 (query)             21_889.48     1_307.23    23_196.70       0.4020          1.0734         8.10
ExhaustiveBinary-1024-random-rf20 (query)             21_889.48     1_447.27    23_336.74       0.5400          1.0418         8.10
ExhaustiveBinary-1024-random (self)                   21_889.48     4_337.23    26_226.70       0.4166          1.0795         8.10
ExhaustiveBinary-1024-pca_no_rr (query)               22_952.65     1_200.67    24_153.33       0.2385             NaN         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                22_952.65     1_341.59    24_294.24       0.6735          1.0459         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                22_952.65     1_485.22    24_437.87       0.8214          1.0198         8.11
ExhaustiveBinary-1024-pca (self)                      22_952.65     4_453.09    27_405.74       0.6814          1.0492         8.11
ExhaustiveBinary-512-signed_no_rr (query)             11_036.13       649.78    11_685.91       0.0950             NaN         4.05
ExhaustiveBinary-512-signed-rf10 (query)              11_036.13       767.20    11_803.33       0.3259          1.1053         4.05
ExhaustiveBinary-512-signed-rf20 (query)              11_036.13       894.00    11_930.12       0.4522          1.0627         4.05
ExhaustiveBinary-512-signed (self)                    11_036.13     2_536.08    13_572.20       0.3417          1.1072         4.05
IVF-Binary-256-nl158-np7-rf0-random (query)            8_179.07       225.05     8_404.11       0.0636             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           8_179.07       228.16     8_407.23       0.0625             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           8_179.07       228.50     8_407.56       0.0616             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           8_179.07       307.22     8_486.29       0.2619          1.1425         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           8_179.07       379.72     8_558.79       0.3812          1.0895         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          8_179.07       301.61     8_480.68       0.2599          1.1431         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          8_179.07       390.39     8_569.45       0.3795          1.0898         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          8_179.07       305.68     8_484.74       0.2591          1.1436         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          8_179.07       403.77     8_582.83       0.3791          1.0899         2.34
IVF-Binary-256-nl158-random (self)                     8_179.07       922.35     9_101.41       0.2768          1.1457         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           6_549.73       236.30     6_786.03       0.0686             NaN         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)           6_549.73       238.24     6_787.97       0.0660             NaN         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)           6_549.73       240.82     6_790.55       0.0628             NaN         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)          6_549.73       317.64     6_867.37       0.2760          1.1336         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)          6_549.73       393.25     6_942.98       0.3976          1.0826         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)          6_549.73       315.73     6_865.45       0.2702          1.1376         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)          6_549.73       395.83     6_945.56       0.3920          1.0851         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)          6_549.73       321.33     6_871.06       0.2626          1.1421         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)          6_549.73       405.52     6_955.25       0.3839          1.0881         2.46
IVF-Binary-256-nl223-random (self)                     6_549.73       965.26     7_514.99       0.2870          1.1394         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)           7_005.66       249.91     7_255.57       0.0668             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           7_005.66       251.30     7_256.96       0.0643             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           7_005.66       253.20     7_258.86       0.0631             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          7_005.66       329.24     7_334.90       0.2736          1.1351         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          7_005.66       405.65     7_411.31       0.3953          1.0838         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          7_005.66       326.74     7_332.40       0.2680          1.1386         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          7_005.66       407.39     7_413.05       0.3895          1.0861         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          7_005.66       331.57     7_337.23       0.2644          1.1407         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          7_005.66       412.76     7_418.42       0.3855          1.0874         2.65
IVF-Binary-256-nl316-random (self)                     7_005.66     1_008.56     8_014.22       0.2843          1.1412         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               8_657.60       234.18     8_891.78       0.1938             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              8_657.60       235.34     8_892.94       0.1882             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              8_657.60       239.98     8_897.58       0.1872             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              8_657.60       330.07     8_987.66       0.5815          1.0747         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              8_657.60       416.54     9_074.14       0.7420          1.0341         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             8_657.60       328.59     8_986.19       0.5579          1.0959         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             8_657.60       414.60     9_072.20       0.7182          1.0418         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             8_657.60       333.90     8_991.50       0.5519          1.1117         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             8_657.60       418.47     9_076.06       0.7102          1.0477         2.34
IVF-Binary-256-nl158-pca (self)                        8_657.60     1_029.85     9_687.45       0.5788          1.0992         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              7_024.74       245.45     7_270.19       0.1888             NaN         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              7_024.74       250.43     7_275.17       0.1878             NaN         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              7_024.74       249.75     7_274.48       0.1869             NaN         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             7_024.74       341.30     7_366.04       0.5604          1.0859         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             7_024.74       427.17     7_451.91       0.7217          1.0380         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             7_024.74       365.40     7_390.14       0.5555          1.0948         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             7_024.74       427.07     7_451.80       0.7149          1.0416         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             7_024.74       347.01     7_371.75       0.5496          1.1117         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             7_024.74       438.62     7_463.36       0.7064          1.0483         2.47
IVF-Binary-256-nl223-pca (self)                        7_024.74     1_070.77     8_095.51       0.5761          1.0985         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              7_478.85       260.35     7_739.19       0.1885             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              7_478.85       259.49     7_738.34       0.1879             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              7_478.85       262.50     7_741.35       0.1867             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             7_478.85       358.04     7_836.89       0.5593          1.0870         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             7_478.85       441.29     7_920.14       0.7204          1.0383         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             7_478.85       353.23     7_832.08       0.5569          1.0907         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             7_478.85       440.00     7_918.85       0.7171          1.0400         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             7_478.85       357.99     7_836.84       0.5507          1.1073         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             7_478.85       444.71     7_923.56       0.7084          1.0463         2.65
IVF-Binary-256-nl316-pca (self)                        7_478.85     1_112.42     8_591.27       0.5776          1.0944         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           13_679.28       421.33    14_100.60       0.0961             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          13_679.28       423.59    14_102.87       0.0956             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          13_679.28       424.62    14_103.90       0.0952             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          13_679.28       501.11    14_180.38       0.3265          1.1051         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          13_679.28       583.13    14_262.41       0.4526          1.0625         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         13_679.28       502.32    14_181.60       0.3262          1.1052         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         13_679.28       626.72    14_306.00       0.4524          1.0626         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         13_679.28       506.77    14_186.05       0.3262          1.1052         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         13_679.28       588.13    14_267.40       0.4524          1.0626         4.36
IVF-Binary-512-nl158-random (self)                    13_679.28     1_598.10    15_277.38       0.3421          1.1071         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          12_055.52       431.22    12_486.75       0.1002             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          12_055.52       442.33    12_497.86       0.0985             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          12_055.52       441.38    12_496.90       0.0961             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         12_055.52       513.86    12_569.38       0.3350          1.1015         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         12_055.52       590.29    12_645.82       0.4609          1.0605         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         12_055.52       514.98    12_570.50       0.3327          1.1027         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         12_055.52       594.57    12_650.09       0.4583          1.0612         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         12_055.52       526.80    12_582.32       0.3291          1.1045         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         12_055.52       605.44    12_660.96       0.4548          1.0621         4.49
IVF-Binary-512-nl223-random (self)                    12_055.52     1_647.51    13_703.03       0.3479          1.1048         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          12_522.58       447.17    12_969.75       0.0994             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          12_522.58       487.16    13_009.74       0.0977             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          12_522.58       487.53    13_010.12       0.0967             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         12_522.58       561.38    13_083.96       0.3339          1.1019         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         12_522.58       610.95    13_133.54       0.4598          1.0607         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         12_522.58       561.19    13_083.77       0.3308          1.1034         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         12_522.58       629.58    13_152.17       0.4569          1.0615         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         12_522.58       541.58    13_064.17       0.3288          1.1044         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         12_522.58       622.37    13_144.95       0.4546          1.0621         4.67
IVF-Binary-512-nl316-random (self)                    12_522.58     1_708.55    14_231.14       0.3472          1.1051         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              14_293.95       434.19    14_728.14       0.2262             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             14_293.95       436.32    14_730.27       0.2196             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             14_293.95       438.68    14_732.64       0.2176             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             14_293.95       526.72    14_820.68       0.6546          1.0666         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             14_293.95       614.55    14_908.50       0.8045          1.0295         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            14_293.95       527.89    14_821.84       0.6281          1.0925         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            14_293.95       614.51    14_908.46       0.7783          1.0388         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            14_293.95       532.09    14_826.04       0.6191          1.1147         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            14_293.95       625.35    14_919.30       0.7661          1.0484         4.36
IVF-Binary-512-nl158-pca (self)                       14_293.95     1_700.56    15_994.52       0.6370          1.0961         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             12_639.28       446.40    13_085.68       0.2206             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             12_639.28       446.35    13_085.63       0.2187             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             12_639.28       460.22    13_099.49       0.2163             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            12_639.28       538.07    13_177.35       0.6313          1.0820         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            12_639.28       633.93    13_273.20       0.7819          1.0348         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            12_639.28       539.31    13_178.59       0.6236          1.0951         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            12_639.28       625.99    13_265.27       0.7725          1.0399         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            12_639.28       547.91    13_187.19       0.6121          1.1200         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            12_639.28       637.27    13_276.55       0.7578          1.0519         4.49
IVF-Binary-512-nl223-pca (self)                       12_639.28     1_744.82    14_384.10       0.6320          1.0984         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             13_101.38       458.79    13_560.17       0.2202             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             13_101.38       460.81    13_562.19       0.2192             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             13_101.38       463.45    13_564.84       0.2166             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            13_101.38       554.55    13_655.93       0.6305          1.0828         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            13_101.38       634.83    13_736.21       0.7810          1.0349         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            13_101.38       552.89    13_654.27       0.6268          1.0887         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            13_101.38       643.10    13_744.48       0.7762          1.0372         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            13_101.38       557.65    13_659.03       0.6154          1.1131         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            13_101.38       646.89    13_748.28       0.7619          1.0479         4.67
IVF-Binary-512-nl316-pca (self)                       13_101.38     1_779.44    14_880.83       0.6351          1.0922         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          24_555.88       808.42    25_364.30       0.1426             NaN         8.41
IVF-Binary-1024-nl158-np12-rf0-random (query)         24_555.88       814.04    25_369.91       0.1424             NaN         8.41
IVF-Binary-1024-nl158-np17-rf0-random (query)         24_555.88       822.80    25_378.68       0.1422             NaN         8.41
IVF-Binary-1024-nl158-np7-rf10-random (query)         24_555.88       898.99    25_454.87       0.4022          1.0734         8.41
IVF-Binary-1024-nl158-np7-rf20-random (query)         24_555.88       984.53    25_540.40       0.5402          1.0418         8.41
IVF-Binary-1024-nl158-np12-rf10-random (query)        24_555.88       899.61    25_455.49       0.4021          1.0734         8.41
IVF-Binary-1024-nl158-np12-rf20-random (query)        24_555.88       983.35    25_539.22       0.5401          1.0418         8.41
IVF-Binary-1024-nl158-np17-rf10-random (query)        24_555.88       903.82    25_459.70       0.4021          1.0734         8.41
IVF-Binary-1024-nl158-np17-rf20-random (query)        24_555.88       989.51    25_545.39       0.5401          1.0418         8.41
IVF-Binary-1024-nl158-random (self)                   24_555.88     2_924.14    27_480.02       0.4167          1.0794         8.41
IVF-Binary-1024-nl223-np11-rf0-random (query)         22_937.89       823.32    23_761.21       0.1449             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         22_937.89       825.51    23_763.40       0.1443             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         22_937.89       833.03    23_770.92       0.1431             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        22_937.89       905.68    23_843.56       0.4070          1.0720         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        22_937.89       984.59    23_922.48       0.5438          1.0411         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        22_937.89       908.43    23_846.31       0.4058          1.0724         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        22_937.89       988.05    23_925.93       0.5425          1.0413         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        22_937.89       919.21    23_857.10       0.4036          1.0731         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        22_937.89     1_005.16    23_943.05       0.5407          1.0417         8.54
IVF-Binary-1024-nl223-random (self)                   22_937.89     2_951.49    25_889.38       0.4198          1.0786         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         23_359.35       838.86    24_198.21       0.1442             NaN         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)         23_359.35       839.58    24_198.94       0.1435             NaN         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)         23_359.35       844.95    24_204.31       0.1431             NaN         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)        23_359.35       921.38    24_280.74       0.4064          1.0722         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)        23_359.35       996.12    24_355.47       0.5438          1.0412         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)        23_359.35       921.21    24_280.57       0.4049          1.0727         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)        23_359.35       999.33    24_358.68       0.5421          1.0415         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)        23_359.35       937.14    24_296.49       0.4037          1.0731         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)        23_359.35     1_029.18    24_388.53       0.5409          1.0417         8.72
IVF-Binary-1024-nl316-random (self)                   23_359.35     2_989.39    26_348.75       0.4193          1.0788         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)             25_602.39       845.26    26_447.66       0.2433             NaN         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            25_602.39       904.08    26_506.47       0.2393             NaN         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            25_602.39       844.48    26_446.88       0.2387             NaN         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            25_602.39       926.61    26_529.01       0.6896          1.0422         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            25_602.39     1_008.64    26_611.04       0.8340          1.0181         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           25_602.39       931.55    26_533.94       0.6768          1.0450         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           25_602.39     1_013.27    26_615.66       0.8247          1.0193         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           25_602.39       933.52    26_535.91       0.6743          1.0455         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           25_602.39     1_017.27    26_619.67       0.8223          1.0196         8.42
IVF-Binary-1024-nl158-pca (self)                      25_602.39     3_027.34    28_629.73       0.6844          1.0483         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            23_954.18       847.47    24_801.65       0.2399             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            23_954.18       861.21    24_815.39       0.2393             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            23_954.18       859.77    24_813.94       0.2388             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           23_954.18       950.43    24_904.61       0.6779          1.0436         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           23_954.18     1_018.85    24_973.03       0.8260          1.0186         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           23_954.18       940.54    24_894.71       0.6760          1.0445         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           23_954.18     1_024.61    24_978.79       0.8243          1.0190         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           23_954.18       953.31    24_907.49       0.6743          1.0450         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           23_954.18     1_037.64    24_991.82       0.8224          1.0193         8.54
IVF-Binary-1024-nl223-pca (self)                      23_954.18     3_072.39    27_026.57       0.6839          1.0478         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            24_425.83       863.96    25_289.79       0.2395             NaN         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            24_425.83       864.47    25_290.30       0.2392             NaN         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            24_425.83       870.98    25_296.81       0.2386             NaN         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           24_425.83       952.56    25_378.39       0.6771          1.0438         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           24_425.83     1_034.19    25_460.02       0.8256          1.0186         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           24_425.83       953.71    25_379.54       0.6764          1.0442         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           24_425.83     1_036.96    25_462.79       0.8249          1.0188         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           24_425.83       976.47    25_402.30       0.6741          1.0449         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           24_425.83     1_046.03    25_471.86       0.8227          1.0192         8.73
IVF-Binary-1024-nl316-pca (self)                      24_425.83     3_120.70    27_546.53       0.6845          1.0475         8.73
IVF-Binary-512-nl158-np7-rf0-signed (query)           13_728.86       419.85    14_148.70       0.0961             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-signed (query)          13_728.86       421.91    14_150.77       0.0956             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-signed (query)          13_728.86       429.04    14_157.89       0.0952             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-signed (query)          13_728.86       499.89    14_228.75       0.3265          1.1051         4.36
IVF-Binary-512-nl158-np7-rf20-signed (query)          13_728.86       577.21    14_306.06       0.4526          1.0625         4.36
IVF-Binary-512-nl158-np12-rf10-signed (query)         13_728.86       501.41    14_230.26       0.3262          1.1052         4.36
IVF-Binary-512-nl158-np12-rf20-signed (query)         13_728.86       587.98    14_316.84       0.4524          1.0626         4.36
IVF-Binary-512-nl158-np17-rf10-signed (query)         13_728.86       506.16    14_235.02       0.3262          1.1052         4.36
IVF-Binary-512-nl158-np17-rf20-signed (query)         13_728.86       586.91    14_315.77       0.4524          1.0626         4.36
IVF-Binary-512-nl158-signed (self)                    13_728.86     1_596.96    15_325.81       0.3421          1.1071         4.36
IVF-Binary-512-nl223-np11-rf0-signed (query)          12_039.44       430.00    12_469.44       0.1002             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-signed (query)          12_039.44       440.94    12_480.38       0.0985             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-signed (query)          12_039.44       440.91    12_480.35       0.0961             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-signed (query)         12_039.44       514.68    12_554.12       0.3350          1.1015         4.49
IVF-Binary-512-nl223-np11-rf20-signed (query)         12_039.44       594.47    12_633.91       0.4609          1.0605         4.49
IVF-Binary-512-nl223-np14-rf10-signed (query)         12_039.44       514.37    12_553.81       0.3327          1.1027         4.49
IVF-Binary-512-nl223-np14-rf20-signed (query)         12_039.44       591.10    12_630.54       0.4583          1.0612         4.49
IVF-Binary-512-nl223-np21-rf10-signed (query)         12_039.44       523.45    12_562.89       0.3291          1.1045         4.49
IVF-Binary-512-nl223-np21-rf20-signed (query)         12_039.44       611.08    12_650.52       0.4548          1.0621         4.49
IVF-Binary-512-nl223-signed (self)                    12_039.44     1_775.85    13_815.29       0.3479          1.1048         4.49
IVF-Binary-512-nl316-np15-rf0-signed (query)          12_519.53       463.31    12_982.84       0.0994             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-signed (query)          12_519.53       450.44    12_969.97       0.0977             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-signed (query)          12_519.53       449.71    12_969.24       0.0967             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-signed (query)         12_519.53       527.70    13_047.22       0.3339          1.1019         4.67
IVF-Binary-512-nl316-np15-rf20-signed (query)         12_519.53       606.27    13_125.80       0.4598          1.0607         4.67
IVF-Binary-512-nl316-np17-rf10-signed (query)         12_519.53       533.97    13_053.49       0.3308          1.1034         4.67
IVF-Binary-512-nl316-np17-rf20-signed (query)         12_519.53       607.48    13_127.01       0.4569          1.0615         4.67
IVF-Binary-512-nl316-np25-rf10-signed (query)         12_519.53       547.48    13_067.00       0.3288          1.1044         4.67
IVF-Binary-512-nl316-np25-rf20-signed (query)         12_519.53       622.65    13_142.17       0.4546          1.0621         4.67
IVF-Binary-512-nl316-signed (self)                    12_519.53     1_680.32    14_199.85       0.3472          1.1051         4.67
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
Exhaustive (query)                                        30.07    15_644.63    15_674.70       1.0000          1.0000       146.48
Exhaustive (self)                                         30.07    52_223.33    52_253.40       1.0000          1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)              8_770.89       482.58     9_253.47       0.0519             NaN         2.28
ExhaustiveBinary-256-random-rf10 (query)               8_770.89       613.54     9_384.43       0.2246          1.1305         2.28
ExhaustiveBinary-256-random-rf20 (query)               8_770.89       753.63     9_524.52       0.3310          1.0856         2.28
ExhaustiveBinary-256-random (self)                     8_770.89     2_014.23    10_785.12       0.2354          1.1317         2.28
ExhaustiveBinary-256-pca_no_rr (query)                 9_379.68       492.57     9_872.26       0.1692             NaN         2.28
ExhaustiveBinary-256-pca-rf10 (query)                  9_379.68       646.06    10_025.74       0.4636          1.1896         2.28
ExhaustiveBinary-256-pca-rf20 (query)                  9_379.68       797.52    10_177.21       0.5918          1.1225         2.28
ExhaustiveBinary-256-pca (self)                        9_379.68     2_129.70    11_509.38       0.4806          1.1914         2.28
ExhaustiveBinary-512-random_no_rr (query)             17_203.06       857.08    18_060.13       0.0750             NaN         4.55
ExhaustiveBinary-512-random-rf10 (query)              17_203.06       997.98    18_201.04       0.2723          1.1002         4.55
ExhaustiveBinary-512-random-rf20 (query)              17_203.06     1_146.82    18_349.88       0.3825          1.0620         4.55
ExhaustiveBinary-512-random (self)                    17_203.06     3_296.03    20_499.09       0.2876          1.0943         4.55
ExhaustiveBinary-512-pca_no_rr (query)                18_021.02       870.72    18_891.74       0.1883             NaN         4.55
ExhaustiveBinary-512-pca-rf10 (query)                 18_021.02     1_024.72    19_045.74       0.4652          1.2549         4.55
ExhaustiveBinary-512-pca-rf20 (query)                 18_021.02     1_181.32    19_202.34       0.5760          1.1920         4.55
ExhaustiveBinary-512-pca (self)                       18_021.02     3_405.41    21_426.43       0.4711          1.2568         4.55
ExhaustiveBinary-1024-random_no_rr (query)            34_293.10     1_598.25    35_891.35       0.1134             NaN         9.10
ExhaustiveBinary-1024-random-rf10 (query)             34_293.10     1_753.48    36_046.57       0.3258          1.0732         9.10
ExhaustiveBinary-1024-random-rf20 (query)             34_293.10     1_914.05    36_207.15       0.4396          1.0445         9.10
ExhaustiveBinary-1024-random (self)                   34_293.10     5_819.90    40_113.00       0.3340          1.0747         9.10
ExhaustiveBinary-1024-pca_no_rr (query)               35_386.32     1_622.61    37_008.94       0.2682             NaN         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                35_386.32     1_789.70    37_176.03       0.7078          1.0551         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                35_386.32     1_948.51    37_334.83       0.8379          1.0248         9.11
ExhaustiveBinary-1024-pca (self)                      35_386.32     5_935.52    41_321.84       0.7159          1.0572         9.11
ExhaustiveBinary-768-signed_no_rr (query)             25_838.88     1_237.77    27_076.65       0.0960             NaN         6.83
ExhaustiveBinary-768-signed-rf10 (query)              25_838.88     1_389.14    27_228.02       0.3033          1.0835         6.83
ExhaustiveBinary-768-signed-rf20 (query)              25_838.88     1_544.70    27_383.58       0.4156          1.0509         6.83
ExhaustiveBinary-768-signed (self)                    25_838.88     4_616.16    30_455.04       0.3145          1.0814         6.83
IVF-Binary-256-nl158-np7-rf0-random (query)           12_757.90       344.99    13_102.89       0.0543             NaN         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)          12_757.90       346.79    13_104.69       0.0533             NaN         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)          12_757.90       348.30    13_106.19       0.0525             NaN         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)          12_757.90       437.37    13_195.27       0.2287          1.1286         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)          12_757.90       529.92    13_287.82       0.3341          1.0845         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)         12_757.90       437.68    13_195.57       0.2268          1.1291         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)         12_757.90       536.64    13_294.54       0.3326          1.0848         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)         12_757.90       441.21    13_199.11       0.2261          1.1295         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)         12_757.90       540.09    13_297.99       0.3324          1.0848         2.74
IVF-Binary-256-nl158-random (self)                    12_757.90     1_358.25    14_116.14       0.2377          1.1303         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)          10_108.95       361.22    10_470.18       0.0576             NaN         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)          10_108.95       364.38    10_473.34       0.0554             NaN         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)          10_108.95       367.03    10_475.98       0.0536             NaN         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)         10_108.95       460.40    10_569.36       0.2443          1.1175         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)         10_108.95       554.85    10_663.81       0.3525          1.0758         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)         10_108.95       459.35    10_568.30       0.2358          1.1233         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)         10_108.95       557.38    10_666.33       0.3421          1.0805         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)         10_108.95       463.38    10_572.33       0.2314          1.1266         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)         10_108.95       567.24    10_676.20       0.3371          1.0830         2.93
IVF-Binary-256-nl223-random (self)                    10_108.95     1_425.50    11_534.46       0.2478          1.1231         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)          10_775.86       384.29    11_160.15       0.0562             NaN         3.20
IVF-Binary-256-nl316-np17-rf0-random (query)          10_775.86       385.41    11_161.27       0.0546             NaN         3.20
IVF-Binary-256-nl316-np25-rf0-random (query)          10_775.86       400.78    11_176.64       0.0542             NaN         3.20
IVF-Binary-256-nl316-np15-rf10-random (query)         10_775.86       485.99    11_261.85       0.2380          1.1223         3.20
IVF-Binary-256-nl316-np15-rf20-random (query)         10_775.86       577.12    11_352.98       0.3451          1.0793         3.20
IVF-Binary-256-nl316-np17-rf10-random (query)         10_775.86       485.21    11_261.08       0.2342          1.1248         3.20
IVF-Binary-256-nl316-np17-rf20-random (query)         10_775.86       576.90    11_352.77       0.3405          1.0812         3.20
IVF-Binary-256-nl316-np25-rf10-random (query)         10_775.86       482.88    11_258.75       0.2334          1.1257         3.20
IVF-Binary-256-nl316-np25-rf20-random (query)         10_775.86       585.84    11_361.70       0.3378          1.0825         3.20
IVF-Binary-256-nl316-random (self)                    10_775.86     1_495.86    12_271.72       0.2452          1.1255         3.20
IVF-Binary-256-nl158-np7-rf0-pca (query)              13_440.02       353.39    13_793.41       0.1802             NaN         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)             13_440.02       355.05    13_795.07       0.1751             NaN         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)             13_440.02       356.67    13_796.69       0.1742             NaN         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)             13_440.02       477.72    13_917.74       0.5384          1.0693         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)             13_440.02       571.58    14_011.60       0.7007          1.0345         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)            13_440.02       470.04    13_910.06       0.5170          1.0848         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)            13_440.02       589.79    14_029.81       0.6786          1.0397         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)            13_440.02       473.37    13_913.39       0.5125          1.0940         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)            13_440.02       589.11    14_029.13       0.6722          1.0429         2.74
IVF-Binary-256-nl158-pca (self)                       13_440.02     1_500.78    14_940.80       0.5381          1.0862         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)             10_804.11       370.05    11_174.16       0.1762             NaN         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)             10_804.11       371.05    11_175.17       0.1749             NaN         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)             10_804.11       382.48    11_186.59       0.1739             NaN         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)            10_804.11       487.76    11_291.87       0.5223          1.0736         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)            10_804.11       591.25    11_395.36       0.6847          1.0352         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)            10_804.11       492.71    11_296.82       0.5169          1.0800         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)            10_804.11       608.03    11_412.14       0.6780          1.0376         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)            10_804.11       503.58    11_307.69       0.5122          1.0884         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)            10_804.11       608.51    11_412.62       0.6712          1.0410         2.93
IVF-Binary-256-nl223-pca (self)                       10_804.11     1_557.76    12_361.88       0.5378          1.0810         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)             11_456.08       409.40    11_865.48       0.1758             NaN         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)             11_456.08       394.38    11_850.46       0.1752             NaN         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)             11_456.08       397.56    11_853.64       0.1742             NaN         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)            11_456.08       512.33    11_968.42       0.5205          1.0749         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)            11_456.08       613.47    12_069.55       0.6828          1.0359         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)            11_456.08       509.12    11_965.20       0.5180          1.0775         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)            11_456.08       618.63    12_074.71       0.6798          1.0369         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)            11_456.08       513.89    11_969.97       0.5131          1.0873         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)            11_456.08       632.31    12_088.39       0.6725          1.0406         3.21
IVF-Binary-256-nl316-pca (self)                       11_456.08     1_626.25    13_082.33       0.5390          1.0787         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)           21_516.47       645.29    22_161.76       0.0763             NaN         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)          21_516.47       666.86    22_183.32       0.0758             NaN         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)          21_516.47       649.80    22_166.27       0.0753             NaN         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)          21_516.47       746.00    22_262.47       0.2733          1.1001         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)          21_516.47       840.93    22_357.39       0.3831          1.0619         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)         21_516.47       749.36    22_265.83       0.2730          1.1001         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)         21_516.47       849.96    22_366.42       0.3829          1.0619         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)         21_516.47       752.78    22_269.24       0.2729          1.1001         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)         21_516.47       852.89    22_369.36       0.3829          1.0619         5.02
IVF-Binary-512-nl158-random (self)                    21_516.47     2_424.06    23_940.53       0.2882          1.0941         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)          18_662.18       662.02    19_324.20       0.0810             NaN         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)          18_662.18       667.96    19_330.14       0.0783             NaN         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)          18_662.18       668.24    19_330.42       0.0769             NaN         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)         18_662.18       768.34    19_430.52       0.2824          1.0950         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)         18_662.18       872.39    19_534.57       0.3900          1.0593         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)         18_662.18       765.97    19_428.15       0.2781          1.0974         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)         18_662.18       867.12    19_529.30       0.3866          1.0605         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)         18_662.18       775.16    19_437.33       0.2763          1.0986         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)         18_662.18       885.56    19_547.74       0.3846          1.0613         5.21
IVF-Binary-512-nl223-random (self)                    18_662.18     2_464.67    21_126.85       0.2924          1.0922         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)          19_340.43       689.69    20_030.13       0.0790             NaN         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)          19_340.43       687.27    20_027.70       0.0778             NaN         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)          19_340.43       691.21    20_031.64       0.0774             NaN         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)         19_340.43       797.85    20_138.29       0.2797          1.0968         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)         19_340.43       885.27    20_225.71       0.3911          1.0591         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)         19_340.43       799.64    20_140.08       0.2779          1.0977         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)         19_340.43       891.01    20_231.45       0.3886          1.0598         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)         19_340.43       797.99    20_138.42       0.2768          1.0983         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)         19_340.43       897.28    20_237.71       0.3867          1.0604         5.48
IVF-Binary-512-nl316-random (self)                    19_340.43     2_532.03    21_872.46       0.2929          1.0919         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)              22_122.51       658.83    22_781.34       0.2344             NaN         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)             22_122.51       662.24    22_784.76       0.2281             NaN         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)             22_122.51       671.85    22_794.36       0.2267             NaN         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)             22_122.51       788.41    22_910.92       0.6555          1.0605         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)             22_122.51       877.53    23_000.04       0.8024          1.0284         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)            22_122.51       778.23    22_900.74       0.6310          1.0810         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)            22_122.51       959.22    23_081.74       0.7779          1.0353         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)            22_122.51       781.56    22_904.07       0.6242          1.0966         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)            22_122.51       888.23    23_010.75       0.7689          1.0413         5.02
IVF-Binary-512-nl158-pca (self)                       22_122.51     2_527.94    24_650.46       0.6460          1.0830         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)             19_482.02       679.65    20_161.68       0.2295             NaN         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)             19_482.02       681.30    20_163.32       0.2278             NaN         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)             19_482.02       682.81    20_164.83       0.2260             NaN         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)            19_482.02       799.33    20_281.35       0.6368          1.0675         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)            19_482.02       909.36    20_391.38       0.7859          1.0303         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)            19_482.02       798.48    20_280.51       0.6298          1.0772         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)            19_482.02       904.69    20_386.71       0.7771          1.0337         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)            19_482.02       803.16    20_285.19       0.6213          1.0942         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)            19_482.02       915.90    20_397.92       0.7660          1.0404         5.21
IVF-Binary-512-nl223-pca (self)                       19_482.02     2_591.83    22_073.85       0.6445          1.0786         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)             20_127.04       710.63    20_837.67       0.2290             NaN         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)             20_127.04       702.74    20_829.78       0.2282             NaN         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)             20_127.04       714.64    20_841.68       0.2264             NaN         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)            20_127.04       822.27    20_949.30       0.6348          1.0692         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)            20_127.04       936.93    21_063.97       0.7832          1.0307         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)            20_127.04       821.21    20_948.25       0.6316          1.0732         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)            20_127.04       924.53    21_051.57       0.7793          1.0322         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)            20_127.04       820.92    20_947.96       0.6230          1.0908         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)            20_127.04       929.82    21_056.85       0.7681          1.0390         5.48
IVF-Binary-512-nl316-pca (self)                       20_127.04     2_703.53    22_830.57       0.6463          1.0746         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)          38_407.94     1_252.85    39_660.79       0.1139             NaN         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)         38_407.94     1_260.74    39_668.68       0.1137             NaN         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)         38_407.94     1_259.14    39_667.08       0.1135             NaN         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)         38_407.94     1_350.07    39_758.01       0.3260          1.0732         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)         38_407.94     1_461.60    39_869.54       0.4398          1.0445         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)        38_407.94     1_353.46    39_761.40       0.3260          1.0732         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)        38_407.94     1_454.87    39_862.81       0.4398          1.0445         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)        38_407.94     1_357.90    39_765.84       0.3260          1.0732         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)        38_407.94     1_476.15    39_884.09       0.4398          1.0445         9.57
IVF-Binary-1024-nl158-random (self)                   38_407.94     4_456.01    42_863.95       0.3341          1.0746         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)         35_754.54     1_284.32    37_038.86       0.1166             NaN         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)         35_754.54     1_285.61    37_040.15       0.1152             NaN         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)         35_754.54     1_278.38    37_032.92       0.1145             NaN         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)        35_754.54     1_380.93    37_135.48       0.3293          1.0717         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)        35_754.54     1_465.66    37_220.20       0.4439          1.0435         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)        35_754.54     1_387.15    37_141.69       0.3277          1.0725         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)        35_754.54     1_476.10    37_230.64       0.4424          1.0439         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)        35_754.54     1_397.50    37_152.04       0.3268          1.0729         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)        35_754.54     1_499.00    37_253.54       0.4415          1.0441         9.76
IVF-Binary-1024-nl223-random (self)                   35_754.54     4_486.03    40_240.57       0.3366          1.0738         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)         36_395.95     1_293.75    37_689.70       0.1161             NaN        10.03
IVF-Binary-1024-nl316-np17-rf0-random (query)         36_395.95     1_295.01    37_690.97       0.1154             NaN        10.03
IVF-Binary-1024-nl316-np25-rf0-random (query)         36_395.95     1_299.84    37_695.80       0.1150             NaN        10.03
IVF-Binary-1024-nl316-np15-rf10-random (query)        36_395.95     1_413.09    37_809.04       0.3292          1.0719        10.03
IVF-Binary-1024-nl316-np15-rf20-random (query)        36_395.95     1_495.58    37_891.54       0.4445          1.0434        10.03
IVF-Binary-1024-nl316-np17-rf10-random (query)        36_395.95     1_405.11    37_801.07       0.3280          1.0723        10.03
IVF-Binary-1024-nl316-np17-rf20-random (query)        36_395.95     1_496.65    37_892.60       0.4430          1.0437        10.03
IVF-Binary-1024-nl316-np25-rf10-random (query)        36_395.95     1_400.45    37_796.40       0.3278          1.0724        10.03
IVF-Binary-1024-nl316-np25-rf20-random (query)        36_395.95     1_548.61    37_944.56       0.4424          1.0438        10.03
IVF-Binary-1024-nl316-random (self)                   36_395.95     4_580.20    40_976.15       0.3370          1.0737        10.03
IVF-Binary-1024-nl158-np7-rf0-pca (query)             39_497.26     1_273.25    40_770.51       0.2750             NaN         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)            39_497.26     1_299.62    40_796.87       0.2697             NaN         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)            39_497.26     1_285.70    40_782.96       0.2686             NaN         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)            39_497.26     1_385.83    40_883.08       0.7313          1.0439         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)            39_497.26     1_485.49    40_982.74       0.8606          1.0201         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)           39_497.26     1_388.41    40_885.67       0.7142          1.0493         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)           39_497.26     1_496.14    40_993.40       0.8465          1.0219         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)           39_497.26     1_392.28    40_889.54       0.7106          1.0512         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)           39_497.26     1_500.98    40_998.24       0.8422          1.0228         9.57
IVF-Binary-1024-nl158-pca (self)                      39_497.26     4_550.21    44_047.46       0.7224          1.0510         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)            36_866.73     1_298.28    38_165.01       0.2707             NaN         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)            36_866.73     1_305.14    38_171.87       0.2695             NaN         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)            36_866.73     1_300.46    38_167.18       0.2688             NaN         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)           36_866.73     1_406.88    38_273.61       0.7177          1.0461         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)           36_866.73     1_508.31    38_375.04       0.8510          1.0205         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)           36_866.73     1_416.32    38_283.05       0.7136          1.0484         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)           36_866.73     1_513.34    38_380.07       0.8464          1.0214         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)           36_866.73     1_415.98    38_282.71       0.7105          1.0504         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)           36_866.73     1_523.93    38_390.66       0.8427          1.0223         9.76
IVF-Binary-1024-nl223-pca (self)                      36_866.73     4_613.41    41_480.14       0.7219          1.0498         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)            37_519.17     1_316.58    38_835.75       0.2701             NaN        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)            37_519.17     1_318.52    38_837.69       0.2697             NaN        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)            37_519.17     1_326.73    38_845.90       0.2687             NaN        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)           37_519.17     1_441.98    38_961.14       0.7158          1.0468        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)           37_519.17     1_532.14    39_051.31       0.8489          1.0207        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)           37_519.17     1_428.41    38_947.57       0.7144          1.0477        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)           37_519.17     1_536.39    39_055.56       0.8472          1.0212        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)           37_519.17     1_435.28    38_954.45       0.7110          1.0501        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)           37_519.17     1_555.23    39_074.40       0.8430          1.0222        10.04
IVF-Binary-1024-nl316-pca (self)                      37_519.17     4_696.70    42_215.86       0.7228          1.0491        10.04
IVF-Binary-768-nl158-np7-rf0-signed (query)           29_925.23       947.51    30_872.74       0.0969             NaN         7.29
IVF-Binary-768-nl158-np12-rf0-signed (query)          29_925.23       950.35    30_875.57       0.0966             NaN         7.29
IVF-Binary-768-nl158-np17-rf0-signed (query)          29_925.23       955.21    30_880.44       0.0963             NaN         7.29
IVF-Binary-768-nl158-np7-rf10-signed (query)          29_925.23     1_047.77    30_973.00       0.3036          1.0834         7.29
IVF-Binary-768-nl158-np7-rf20-signed (query)          29_925.23     1_144.59    31_069.81       0.4159          1.0509         7.29
IVF-Binary-768-nl158-np12-rf10-signed (query)         29_925.23     1_053.64    30_978.87       0.3036          1.0834         7.29
IVF-Binary-768-nl158-np12-rf20-signed (query)         29_925.23     1_152.79    31_078.01       0.4158          1.0509         7.29
IVF-Binary-768-nl158-np17-rf10-signed (query)         29_925.23     1_069.07    30_994.29       0.3036          1.0835         7.29
IVF-Binary-768-nl158-np17-rf20-signed (query)         29_925.23     1_157.84    31_083.06       0.4158          1.0509         7.29
IVF-Binary-768-nl158-signed (self)                    29_925.23     3_433.65    33_358.88       0.3148          1.0813         7.29
IVF-Binary-768-nl223-np11-rf0-signed (query)          27_289.93       967.72    28_257.65       0.1002             NaN         7.48
IVF-Binary-768-nl223-np14-rf0-signed (query)          27_289.93       969.21    28_259.15       0.0982             NaN         7.48
IVF-Binary-768-nl223-np21-rf0-signed (query)          27_289.93       975.02    28_264.95       0.0973             NaN         7.48
IVF-Binary-768-nl223-np11-rf10-signed (query)         27_289.93     1_115.67    28_405.60       0.3090          1.0811         7.48
IVF-Binary-768-nl223-np11-rf20-signed (query)         27_289.93     1_179.58    28_469.51       0.4206          1.0496         7.48
IVF-Binary-768-nl223-np14-rf10-signed (query)         27_289.93     1_071.02    28_360.95       0.3067          1.0822         7.48
IVF-Binary-768-nl223-np14-rf20-signed (query)         27_289.93     1_171.04    28_460.98       0.4185          1.0501         7.48
IVF-Binary-768-nl223-np21-rf10-signed (query)         27_289.93     1_078.51    28_368.45       0.3055          1.0829         7.48
IVF-Binary-768-nl223-np21-rf20-signed (query)         27_289.93     1_181.61    28_471.54       0.4174          1.0503         7.48
IVF-Binary-768-nl223-signed (self)                    27_289.93     3_477.88    30_767.81       0.3179          1.0801         7.48
IVF-Binary-768-nl316-np15-rf0-signed (query)          27_934.68       990.10    28_924.77       0.0986             NaN         7.76
IVF-Binary-768-nl316-np17-rf0-signed (query)          27_934.68       992.65    28_927.33       0.0978             NaN         7.76
IVF-Binary-768-nl316-np25-rf0-signed (query)          27_934.68     1_008.77    28_943.45       0.0974             NaN         7.76
IVF-Binary-768-nl316-np15-rf10-signed (query)         27_934.68     1_101.35    29_036.03       0.3090          1.0810         7.76
IVF-Binary-768-nl316-np15-rf20-signed (query)         27_934.68     1_190.99    29_125.67       0.4203          1.0495         7.76
IVF-Binary-768-nl316-np17-rf10-signed (query)         27_934.68     1_091.24    29_025.92       0.3076          1.0816         7.76
IVF-Binary-768-nl316-np17-rf20-signed (query)         27_934.68     1_192.91    29_127.58       0.4186          1.0499         7.76
IVF-Binary-768-nl316-np25-rf10-signed (query)         27_934.68     1_097.22    29_031.90       0.3066          1.0822         7.76
IVF-Binary-768-nl316-np25-rf20-signed (query)         27_934.68     1_201.00    29_135.68       0.4179          1.0501         7.76
IVF-Binary-768-nl316-signed (self)                    27_934.68     3_545.33    31_480.00       0.3184          1.0798         7.76
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
Exhaustive (query)                                        10.22     4_556.45     4_566.67       1.0000          1.0000        48.83
Exhaustive (self)                                         10.22    14_256.76    14_266.99       1.0000          1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_581.31       275.32     2_856.63       0.2840             NaN         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_581.31       383.43     2_964.74       0.8630          1.0794         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_581.31       484.58     3_065.89       0.9501          1.0210         1.78
ExhaustiveBinary-256-random (self)                     2_581.31     1_263.13     3_844.44       0.8606          1.0824         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_797.55       284.63     3_082.18       0.1183             NaN         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_797.55       382.77     3_180.31       0.3215          1.9397         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_797.55       480.39     3_277.93       0.4181          1.5883         1.78
ExhaustiveBinary-256-pca (self)                        2_797.55     1_265.04     4_062.58       0.3192          1.9523         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_076.03       440.08     5_516.11       0.3036             NaN         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_076.03       545.70     5_621.72       0.9167          1.0510         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_076.03       656.34     5_732.37       0.9760          1.0116         3.55
ExhaustiveBinary-512-random (self)                     5_076.03     1_801.55     6_877.58       0.9147          1.0543         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_443.39       444.26     5_887.64       0.3665             NaN         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_443.39       551.40     5_994.79       0.8453          1.0657         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_443.39       655.85     6_099.24       0.9396          1.0206         3.55
ExhaustiveBinary-512-pca (self)                        5_443.39     1_822.86     7_266.24       0.8325          1.0737         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_019.51       756.07    10_775.57       0.3110             NaN         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_019.51       868.39    10_887.90       0.9411          1.0418         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_019.51       996.60    11_016.10       0.9840          1.0092         7.10
ExhaustiveBinary-1024-random (self)                   10_019.51     2_903.27    12_922.78       0.9397          1.0445         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_516.92       788.24    11_305.16       0.5577             NaN         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_516.92       889.12    11_406.04       0.9880          1.0028         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_516.92       999.22    11_516.14       0.9987          1.0003         7.10
ExhaustiveBinary-1024-pca (self)                      10_516.92     2_937.17    13_454.09       0.9861          1.0033         7.10
ExhaustiveBinary-256-signed_no_rr (query)              2_578.69       275.17     2_853.86       0.2840             NaN         1.78
ExhaustiveBinary-256-signed-rf10 (query)               2_578.69       382.35     2_961.04       0.8630          1.0794         1.78
ExhaustiveBinary-256-signed-rf20 (query)               2_578.69       485.15     3_063.84       0.9501          1.0210         1.78
ExhaustiveBinary-256-signed (self)                     2_578.69     1_261.82     3_840.51       0.8606          1.0824         1.78
IVF-Binary-256-nl158-np7-rf0-random (query)            3_994.72       117.37     4_112.09       0.3903             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           3_994.72       139.07     4_133.79       0.3491             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           3_994.72       140.75     4_135.47       0.3283             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           3_994.72       187.54     4_182.26       0.9569          1.0109         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           3_994.72       241.90     4_236.62       0.9891          1.0022         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          3_994.72       202.28     4_197.00       0.9376          1.0176         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          3_994.72       265.84     4_260.57       0.9853          1.0031         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          3_994.72       228.79     4_223.51       0.9218          1.0242         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          3_994.72       284.28     4_279.00       0.9801          1.0045         1.93
IVF-Binary-256-nl158-random (self)                     3_994.72       628.58     4_623.30       0.9363          1.0184         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           3_079.92       122.92     3_202.84       0.3792             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           3_079.92       126.34     3_206.27       0.3618             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           3_079.92       136.06     3_215.99       0.3375             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          3_079.92       190.64     3_270.56       0.9546          1.0115         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          3_079.92       244.66     3_324.58       0.9902          1.0019         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          3_079.92       196.66     3_276.58       0.9451          1.0152         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          3_079.92       255.58     3_335.50       0.9874          1.0026         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          3_079.92       208.76     3_288.68       0.9283          1.0221         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          3_079.92       274.46     3_354.39       0.9818          1.0041         2.00
IVF-Binary-256-nl223-random (self)                     3_079.92       609.39     3_689.31       0.9439          1.0157         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           3_190.36       129.91     3_320.27       0.3752             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           3_190.36       136.91     3_327.27       0.3667             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           3_190.36       138.66     3_329.02       0.3436             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          3_190.36       201.95     3_392.30       0.9542          1.0117         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          3_190.36       253.52     3_443.88       0.9906          1.0018         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          3_190.36       197.50     3_387.86       0.9494          1.0133         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          3_190.36       256.05     3_446.41       0.9893          1.0021         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          3_190.36       209.75     3_400.11       0.9345          1.0191         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          3_190.36       274.00     3_464.36       0.9845          1.0033         2.09
IVF-Binary-256-nl316-random (self)                     3_190.36       619.28     3_809.64       0.9485          1.0136         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               4_266.81       121.30     4_388.11       0.1525             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              4_266.81       131.39     4_398.20       0.1374             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              4_266.81       148.55     4_415.36       0.1309             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              4_266.81       190.06     4_456.87       0.4905          1.4115         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              4_266.81       249.99     4_516.80       0.6392          1.2170         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             4_266.81       203.88     4_470.69       0.4272          1.5368         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             4_266.81       273.97     4_540.78       0.5675          1.2973         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             4_266.81       224.19     4_491.00       0.3949          1.6196         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             4_266.81       291.44     4_558.25       0.5273          1.3520         1.93
IVF-Binary-256-nl158-pca (self)                        4_266.81       659.27     4_926.08       0.4247          1.5447         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              3_258.74       124.94     3_383.68       0.1483             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              3_258.74       128.79     3_387.52       0.1422             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              3_258.74       136.19     3_394.93       0.1340             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             3_258.74       195.57     3_454.30       0.4801          1.4243         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             3_258.74       252.05     3_510.79       0.6313          1.2223         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             3_258.74       198.97     3_457.71       0.4520          1.4780         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             3_258.74       263.43     3_522.17       0.6000          1.2556         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             3_258.74       215.07     3_473.80       0.4117          1.5721         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             3_258.74       283.03     3_541.76       0.5506          1.3174         2.00
IVF-Binary-256-nl223-pca (self)                        3_258.74       662.72     3_921.45       0.4496          1.4842         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              3_497.30       133.44     3_630.74       0.1481             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              3_497.30       133.26     3_630.56       0.1446             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              3_497.30       139.88     3_637.18       0.1363             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             3_497.30       201.14     3_698.44       0.4798          1.4233         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             3_497.30       258.60     3_755.90       0.6328          1.2201         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             3_497.30       202.04     3_699.34       0.4654          1.4494         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             3_497.30       263.77     3_761.07       0.6164          1.2368         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             3_497.30       213.23     3_710.53       0.4254          1.5346         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             3_497.30       292.34     3_789.64       0.5687          1.2916         2.09
IVF-Binary-256-nl316-pca (self)                        3_497.30       638.34     4_135.63       0.4628          1.4553         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_523.65       210.04     6_733.69       0.4173             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_523.65       222.20     6_745.85       0.3712             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_523.65       235.23     6_758.88       0.3492             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_523.65       281.93     6_805.58       0.9829          1.0037         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_523.65       333.14     6_856.79       0.9955          1.0009         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_523.65       294.70     6_818.35       0.9744          1.0059         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_523.65       358.75     6_882.40       0.9960          1.0007         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_523.65       313.30     6_836.95       0.9646          1.0091         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_523.65       386.35     6_910.00       0.9940          1.0011         3.71
IVF-Binary-512-nl158-random (self)                     6_523.65       952.22     7_475.87       0.9732          1.0067         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_570.68       212.51     5_783.19       0.4051             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_570.68       218.32     5_789.00       0.3861             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_570.68       229.11     5_799.79       0.3594             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_570.68       280.58     5_851.26       0.9827          1.0036         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_570.68       337.61     5_908.29       0.9971          1.0005         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_570.68       286.63     5_857.31       0.9777          1.0052         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_570.68       347.94     5_918.62       0.9964          1.0006         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_570.68       302.73     5_873.41       0.9678          1.0084         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_570.68       368.50     5_939.18       0.9944          1.0011         3.77
IVF-Binary-512-nl223-random (self)                     5_570.68       930.43     6_501.12       0.9768          1.0057         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           5_654.13       219.24     5_873.37       0.4019             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           5_654.13       221.87     5_876.00       0.3927             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           5_654.13       231.90     5_886.03       0.3669             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          5_654.13       287.10     5_941.23       0.9827          1.0036         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          5_654.13       346.17     6_000.30       0.9976          1.0004         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          5_654.13       290.40     5_944.53       0.9803          1.0043         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          5_654.13       346.94     6_001.07       0.9971          1.0005         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          5_654.13       302.78     5_956.91       0.9718          1.0070         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          5_654.13       364.95     6_019.08       0.9955          1.0008         3.86
IVF-Binary-512-nl316-random (self)                     5_654.13       920.03     6_574.16       0.9795          1.0048         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               6_835.14       216.66     7_051.80       0.3817             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              6_835.14       229.40     7_064.54       0.3729             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              6_835.14       244.44     7_079.58       0.3697             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              6_835.14       283.15     7_118.29       0.8864          1.0420         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              6_835.14       339.85     7_174.99       0.9653          1.0104         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             6_835.14       324.71     7_159.85       0.8656          1.0534         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             6_835.14       365.70     7_200.84       0.9543          1.0147         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             6_835.14       324.97     7_160.11       0.8563          1.0589         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             6_835.14       390.84     7_225.98       0.9479          1.0171         3.71
IVF-Binary-512-nl158-pca (self)                        6_835.14       970.75     7_805.88       0.8541          1.0600         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              5_854.71       220.51     6_075.23       0.3777             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              5_854.71       225.75     6_080.47       0.3741             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              5_854.71       235.79     6_090.50       0.3706             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             5_854.71       286.01     6_140.72       0.8810          1.0451         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             5_854.71       347.42     6_202.13       0.9631          1.0112         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             5_854.71       294.67     6_149.39       0.8710          1.0504         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             5_854.71       355.35     6_210.07       0.9578          1.0132         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             5_854.71       310.10     6_164.81       0.8589          1.0574         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             5_854.71       376.52     6_231.23       0.9499          1.0164         3.77
IVF-Binary-512-nl223-pca (self)                        5_854.71       941.04     6_795.75       0.8601          1.0566         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              6_004.22       226.63     6_230.85       0.3771             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              6_004.22       228.78     6_233.00       0.3755             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              6_004.22       239.31     6_243.53       0.3714             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             6_004.22       294.56     6_298.78       0.8796          1.0460         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             6_004.22       350.02     6_354.24       0.9626          1.0115         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             6_004.22       322.09     6_326.31       0.8748          1.0483         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             6_004.22       360.95     6_365.17       0.9602          1.0124         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             6_004.22       309.16     6_313.38       0.8625          1.0551         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             6_004.22       374.91     6_379.13       0.9524          1.0154         3.86
IVF-Binary-512-nl316-pca (self)                        6_004.22       946.32     6_950.54       0.8640          1.0545         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_394.43       394.68    11_789.10       0.4316             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_394.43       415.72    11_810.14       0.3816             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_394.43       437.83    11_832.26       0.3586             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_394.43       463.40    11_857.82       0.9908          1.0021         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_394.43       521.13    11_915.56       0.9967          1.0007         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_394.43       492.55    11_886.97       0.9868          1.0032         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_394.43       565.65    11_960.07       0.9983          1.0003         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_394.43       515.21    11_909.63       0.9808          1.0052         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_394.43       587.33    11_981.76       0.9973          1.0006         7.26
IVF-Binary-1024-nl158-random (self)                   11_394.43     1_599.54    12_993.96       0.9863          1.0039         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_464.36       400.08    10_864.44       0.4179             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_464.36       409.85    10_874.20       0.3976             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_464.36       423.86    10_888.21       0.3693             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_464.36       469.19    10_933.55       0.9911          1.0019         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_464.36       522.92    10_987.28       0.9984          1.0003         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_464.36       477.72    10_942.07       0.9885          1.0029         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_464.36       538.75    11_003.11       0.9984          1.0003         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_464.36       499.01    10_963.37       0.9821          1.0050         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_464.36       569.57    11_033.92       0.9974          1.0006         7.32
IVF-Binary-1024-nl223-random (self)                   10_464.36     1_552.21    12_016.57       0.9881          1.0033         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         10_586.23       420.77    11_007.00       0.4144             NaN         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)         10_586.23       412.14    10_998.36       0.4047             NaN         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)         10_586.23       426.38    11_012.60       0.3768             NaN         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)        10_586.23       474.17    11_060.39       0.9914          1.0019         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)        10_586.23       528.32    11_114.55       0.9988          1.0002         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)        10_586.23       477.70    11_063.93       0.9901          1.0024         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)        10_586.23       537.47    11_123.69       0.9987          1.0003         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)        10_586.23       495.17    11_081.40       0.9848          1.0041         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)        10_586.23       562.67    11_148.89       0.9980          1.0004         7.41
IVF-Binary-1024-nl316-random (self)                   10_586.23     1_563.77    12_150.00       0.9897          1.0027         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)             11_958.87       413.38    12_372.25       0.5685             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            11_958.87       433.64    12_392.51       0.5620             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            11_958.87       452.01    12_410.88       0.5600             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            11_958.87       475.61    12_434.48       0.9912          1.0019         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            11_958.87       543.90    12_502.76       0.9972          1.0006         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           11_958.87       505.11    12_463.98       0.9907          1.0020         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           11_958.87       567.00    12_525.87       0.9992          1.0002         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           11_958.87       528.30    12_487.17       0.9895          1.0023         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           11_958.87       600.07    12_558.93       0.9990          1.0002         7.26
IVF-Binary-1024-nl158-pca (self)                      11_958.87     1_636.33    13_595.20       0.9890          1.0025         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            10_976.73       412.37    11_389.10       0.5652             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            10_976.73       421.67    11_398.40       0.5627             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            10_976.73       439.37    11_416.10       0.5604             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           10_976.73       478.92    11_455.65       0.9916          1.0017         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           10_976.73       534.48    11_511.21       0.9987          1.0002         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           10_976.73       492.63    11_469.36       0.9909          1.0019         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           10_976.73       554.51    11_531.24       0.9991          1.0002         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           10_976.73       511.48    11_488.21       0.9896          1.0023         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           10_976.73       578.85    11_555.58       0.9990          1.0002         7.32
IVF-Binary-1024-nl223-pca (self)                      10_976.73     1_608.63    12_585.36       0.9892          1.0024         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            11_137.02       419.69    11_556.71       0.5647             NaN         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            11_137.02       425.38    11_562.40       0.5636             NaN         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            11_137.02       436.95    11_573.97       0.5612             NaN         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           11_137.02       487.60    11_624.63       0.9917          1.0018         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           11_137.02       542.47    11_679.49       0.9990          1.0002         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           11_137.02       492.53    11_629.55       0.9914          1.0018         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           11_137.02       552.65    11_689.67       0.9991          1.0002         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           11_137.02       508.57    11_645.59       0.9902          1.0022         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           11_137.02       572.74    11_709.77       0.9991          1.0002         7.42
IVF-Binary-1024-nl316-pca (self)                      11_137.02     1_735.19    12_872.21       0.9897          1.0023         7.42
IVF-Binary-256-nl158-np7-rf0-signed (query)            3_993.11       117.12     4_110.24       0.3903             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-signed (query)           3_993.11       126.62     4_119.73       0.3491             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-signed (query)           3_993.11       135.78     4_128.89       0.3283             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-signed (query)           3_993.11       185.01     4_178.12       0.9569          1.0109         1.93
IVF-Binary-256-nl158-np7-rf20-signed (query)           3_993.11       238.28     4_231.39       0.9891          1.0022         1.93
IVF-Binary-256-nl158-np12-rf10-signed (query)          3_993.11       198.50     4_191.61       0.9376          1.0176         1.93
IVF-Binary-256-nl158-np12-rf20-signed (query)          3_993.11       262.18     4_255.29       0.9853          1.0031         1.93
IVF-Binary-256-nl158-np17-rf10-signed (query)          3_993.11       214.03     4_207.14       0.9218          1.0242         1.93
IVF-Binary-256-nl158-np17-rf20-signed (query)          3_993.11       281.44     4_274.55       0.9801          1.0045         1.93
IVF-Binary-256-nl158-signed (self)                     3_993.11       627.10     4_620.21       0.9363          1.0184         1.93
IVF-Binary-256-nl223-np11-rf0-signed (query)           3_044.07       121.39     3_165.46       0.3792             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-signed (query)           3_044.07       127.62     3_171.69       0.3618             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-signed (query)           3_044.07       133.42     3_177.49       0.3375             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-signed (query)          3_044.07       187.25     3_231.32       0.9546          1.0115         2.00
IVF-Binary-256-nl223-np11-rf20-signed (query)          3_044.07       241.80     3_285.87       0.9902          1.0019         2.00
IVF-Binary-256-nl223-np14-rf10-signed (query)          3_044.07       194.32     3_238.39       0.9451          1.0152         2.00
IVF-Binary-256-nl223-np14-rf20-signed (query)          3_044.07       256.19     3_300.26       0.9874          1.0026         2.00
IVF-Binary-256-nl223-np21-rf10-signed (query)          3_044.07       206.04     3_250.11       0.9283          1.0221         2.00
IVF-Binary-256-nl223-np21-rf20-signed (query)          3_044.07       270.15     3_314.22       0.9818          1.0041         2.00
IVF-Binary-256-nl223-signed (self)                     3_044.07       602.93     3_647.00       0.9439          1.0157         2.00
IVF-Binary-256-nl316-np15-rf0-signed (query)           3_209.31       128.70     3_338.00       0.3752             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-signed (query)           3_209.31       132.73     3_342.03       0.3667             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-signed (query)           3_209.31       135.69     3_345.00       0.3436             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-signed (query)          3_209.31       194.18     3_403.48       0.9542          1.0117         2.09
IVF-Binary-256-nl316-np15-rf20-signed (query)          3_209.31       248.25     3_457.56       0.9906          1.0018         2.09
IVF-Binary-256-nl316-np17-rf10-signed (query)          3_209.31       196.20     3_405.51       0.9494          1.0133         2.09
IVF-Binary-256-nl316-np17-rf20-signed (query)          3_209.31       253.83     3_463.13       0.9893          1.0021         2.09
IVF-Binary-256-nl316-np25-rf10-signed (query)          3_209.31       206.97     3_416.27       0.9345          1.0191         2.09
IVF-Binary-256-nl316-np25-rf20-signed (query)          3_209.31       267.74     3_477.05       0.9845          1.0033         2.09
IVF-Binary-256-nl316-signed (self)                     3_209.31       610.27     3_819.58       0.9485          1.0136         2.09
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
Exhaustive (query)                                        19.88    10_143.04    10_162.92       1.0000          1.0000        97.66
Exhaustive (self)                                         19.88    32_047.54    32_067.42       1.0000          1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_634.62       378.30     6_012.92       0.2776             NaN         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_634.62       511.59     6_146.20       0.8648          1.0698         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_634.62       637.05     6_271.67       0.9562          1.0171         2.03
ExhaustiveBinary-256-random (self)                     5_634.62     1_682.56     7_317.17       0.8664          1.0698         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_159.27       384.79     6_544.06       0.1212             NaN         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_159.27       512.85     6_672.12       0.3407          1.8751         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_159.27       635.60     6_794.87       0.4406          1.5429         2.03
ExhaustiveBinary-256-pca (self)                        6_159.27     1_692.26     7_851.53       0.3366          1.8907         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_165.84       646.33    11_812.17       0.2965             NaN         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_165.84       775.79    11_941.63       0.9188          1.0453         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_165.84       918.54    12_084.38       0.9788          1.0097         4.05
ExhaustiveBinary-512-random (self)                    11_165.84     2_569.99    13_735.83       0.9199          1.0452         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_712.24       657.81    12_370.05       0.1147             NaN         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_712.24       781.88    12_494.11       0.2782          2.2254         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_712.24       908.54    12_620.78       0.3475          1.8252         4.05
ExhaustiveBinary-512-pca (self)                       11_712.24     2_591.38    14_303.61       0.2742          2.2528         4.05
ExhaustiveBinary-1024-random_no_rr (query)            21_915.06     1_193.59    23_108.65       0.3064             NaN         8.10
ExhaustiveBinary-1024-random-rf10 (query)             21_915.06     1_312.71    23_227.77       0.9430          1.0366         8.10
ExhaustiveBinary-1024-random-rf20 (query)             21_915.06     1_454.60    23_369.66       0.9861          1.0075         8.10
ExhaustiveBinary-1024-random (self)                   21_915.06     4_360.35    26_275.41       0.9439          1.0367         8.10
ExhaustiveBinary-1024-pca_no_rr (query)               22_985.45     1_200.20    24_185.65       0.3939             NaN         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                22_985.45     1_344.79    24_330.24       0.8323          1.0743         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                22_985.45     1_476.71    24_462.16       0.9198          1.0285         8.11
ExhaustiveBinary-1024-pca (self)                      22_985.45     4_445.37    27_430.82       0.8160          1.0854         8.11
ExhaustiveBinary-512-signed_no_rr (query)             11_045.33       646.02    11_691.35       0.2965             NaN         4.05
ExhaustiveBinary-512-signed-rf10 (query)              11_045.33       777.55    11_822.87       0.9188          1.0453         4.05
ExhaustiveBinary-512-signed-rf20 (query)              11_045.33       911.35    11_956.68       0.9788          1.0097         4.05
ExhaustiveBinary-512-signed (self)                    11_045.33     2_570.32    13_615.65       0.9199          1.0452         4.05
IVF-Binary-256-nl158-np7-rf0-random (query)            8_632.72       231.66     8_864.37       0.3651             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           8_632.72       242.48     8_875.19       0.3326             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           8_632.72       249.95     8_882.66       0.3138             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           8_632.72       322.58     8_955.29       0.9498          1.0139         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           8_632.72       402.77     9_035.48       0.9902          1.0021         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          8_632.72       335.75     8_968.47       0.9293          1.0223         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          8_632.72       425.19     9_057.91       0.9852          1.0034         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          8_632.72       351.97     8_984.68       0.9138          1.0290         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          8_632.72       448.50     9_081.21       0.9797          1.0049         2.34
IVF-Binary-256-nl158-random (self)                     8_632.72     1_055.51     9_688.22       0.9307          1.0217         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           6_362.11       258.00     6_620.11       0.3536             NaN         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)           6_362.11       245.25     6_607.36       0.3401             NaN         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)           6_362.11       253.45     6_615.56       0.3194             NaN         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)          6_362.11       333.50     6_695.61       0.9466          1.0146         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)          6_362.11       418.38     6_780.49       0.9909          1.0019         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)          6_362.11       337.12     6_699.23       0.9380          1.0180         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)          6_362.11       422.16     6_784.27       0.9884          1.0025         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)          6_362.11       351.45     6_713.56       0.9208          1.0251         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)          6_362.11       444.18     6_806.29       0.9826          1.0040         2.46
IVF-Binary-256-nl223-random (self)                     6_362.11     1_144.82     7_506.93       0.9387          1.0178         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)           6_575.18       254.84     6_830.01       0.3480             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           6_575.18       256.77     6_831.95       0.3420             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           6_575.18       265.07     6_840.24       0.3246             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          6_575.18       348.13     6_923.31       0.9452          1.0151         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          6_575.18       429.21     7_004.39       0.9909          1.0019         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          6_575.18       352.16     6_927.34       0.9406          1.0168         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          6_575.18       433.27     7_008.44       0.9896          1.0022         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          6_575.18       361.05     6_936.23       0.9256          1.0228         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          6_575.18       451.36     7_026.53       0.9852          1.0033         2.65
IVF-Binary-256-nl316-random (self)                     6_575.18     1_080.26     7_655.44       0.9420          1.0162         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               9_136.44       237.74     9_374.18       0.1446             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              9_136.44       247.00     9_383.45       0.1353             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              9_136.44       256.06     9_392.51       0.1305             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              9_136.44       327.25     9_463.69       0.4660          1.4656         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              9_136.44       412.37     9_548.81       0.6170          1.2408         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             9_136.44       341.18     9_477.62       0.4250          1.5540         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             9_136.44       432.95     9_569.40       0.5648          1.3027         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             9_136.44       355.71     9_492.15       0.4032          1.6107         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             9_136.44       462.22     9_598.66       0.5371          1.3415         2.34
IVF-Binary-256-nl158-pca (self)                        9_136.44     1_086.31    10_222.76       0.4221          1.5634         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              6_884.36       247.48     7_131.84       0.1417             NaN         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              6_884.36       252.19     7_136.55       0.1375             NaN         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              6_884.36       263.75     7_148.11       0.1316             NaN         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             6_884.36       337.13     7_221.49       0.4567          1.4780         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             6_884.36       420.88     7_305.24       0.6069          1.2482         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             6_884.36       341.26     7_225.62       0.4388          1.5156         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             6_884.36       433.39     7_317.75       0.5849          1.2736         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             6_884.36       354.90     7_239.26       0.4115          1.5826         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             6_884.36       450.08     7_334.44       0.5502          1.3193         2.47
IVF-Binary-256-nl223-pca (self)                        6_884.36     1_078.51     7_962.87       0.4359          1.5231         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              7_288.16       274.10     7_562.26       0.1403             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              7_288.16       273.98     7_562.14       0.1382             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              7_288.16       270.53     7_558.69       0.1334             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             7_288.16       354.15     7_642.31       0.4552          1.4775         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             7_288.16       443.93     7_732.10       0.6068          1.2470         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             7_288.16       363.91     7_652.07       0.4459          1.4972         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             7_288.16       451.04     7_739.20       0.5952          1.2602         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             7_288.16       364.99     7_653.15       0.4205          1.5566         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             7_288.16       454.38     7_742.54       0.5622          1.3014         2.65
IVF-Binary-256-nl316-pca (self)                        7_288.16     1_120.79     8_408.95       0.4434          1.5036         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           14_154.70       429.14    14_583.84       0.3916             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          14_154.70       440.89    14_595.59       0.3554             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          14_154.70       452.42    14_607.12       0.3353             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          14_154.70       517.64    14_672.34       0.9791          1.0056         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          14_154.70       596.89    14_751.59       0.9960          1.0009         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         14_154.70       534.10    14_688.80       0.9680          1.0098         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         14_154.70       622.73    14_777.43       0.9953          1.0013         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         14_154.70       554.61    14_709.31       0.9578          1.0135         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         14_154.70       650.92    14_805.62       0.9933          1.0018         4.36
IVF-Binary-512-nl158-random (self)                    14_154.70     1_723.55    15_878.25       0.9688          1.0094         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          11_825.63       437.78    12_263.41       0.3780             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          11_825.63       445.47    12_271.10       0.3640             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          11_825.63       462.89    12_288.51       0.3413             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         11_825.63       527.02    12_352.65       0.9782          1.0057         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         11_825.63       612.46    12_438.08       0.9972          1.0006         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         11_825.63       534.01    12_359.64       0.9733          1.0074         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         11_825.63       630.48    12_456.11       0.9965          1.0008         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         11_825.63       556.13    12_381.76       0.9621          1.0114         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         11_825.63       653.63    12_479.26       0.9943          1.0014         4.49
IVF-Binary-512-nl223-random (self)                    11_825.63     1_708.44    13_534.07       0.9737          1.0073         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          12_084.67       472.93    12_557.60       0.3736             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          12_084.67       461.93    12_546.60       0.3669             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          12_084.67       477.68    12_562.35       0.3475             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         12_084.67       554.17    12_638.85       0.9781          1.0056         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         12_084.67       624.11    12_708.78       0.9976          1.0005         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         12_084.67       543.78    12_628.45       0.9756          1.0066         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         12_084.67       646.83    12_731.50       0.9971          1.0006         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         12_084.67       556.94    12_641.62       0.9663          1.0098         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         12_084.67       647.25    12_731.92       0.9953          1.0011         4.67
IVF-Binary-512-nl316-random (self)                    12_084.67     1_744.38    13_829.05       0.9761          1.0062         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              14_786.11       441.49    15_227.59       0.1402             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             14_786.11       452.83    15_238.94       0.1307             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             14_786.11       465.71    15_251.82       0.1257             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             14_786.11       529.66    15_315.77       0.4248          1.5450         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             14_786.11       612.51    15_398.62       0.5632          1.3025         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            14_786.11       546.90    15_333.00       0.3820          1.6569         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            14_786.11       639.78    15_425.88       0.5038          1.3865         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            14_786.11       564.88    15_350.99       0.3596          1.7323         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            14_786.11       662.08    15_448.19       0.4725          1.4419         4.36
IVF-Binary-512-nl158-pca (self)                       14_786.11     1_770.60    16_556.70       0.3781          1.6708         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             12_454.77       453.33    12_908.10       0.1371             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             12_454.77       456.50    12_911.28       0.1332             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             12_454.77       465.90    12_920.67       0.1272             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            12_454.77       542.44    12_997.21       0.4162          1.5553         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            12_454.77       622.60    13_077.37       0.5534          1.3091         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            12_454.77       548.48    13_003.26       0.3973          1.6033         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            12_454.77       643.89    13_098.66       0.5274          1.3450         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            12_454.77       560.72    13_015.49       0.3699          1.6879         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            12_454.77       655.12    13_109.90       0.4885          1.4076         4.49
IVF-Binary-512-nl223-pca (self)                       12_454.77     1_788.73    14_243.51       0.3936          1.6150         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             12_730.30       470.69    13_200.99       0.1360             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             12_730.30       474.70    13_205.00       0.1339             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             12_730.30       478.34    13_208.64       0.1287             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            12_730.30       556.02    13_286.33       0.4151          1.5550         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            12_730.30       636.18    13_366.48       0.5522          1.3080         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            12_730.30       573.60    13_303.90       0.4052          1.5795         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            12_730.30       641.86    13_372.16       0.5383          1.3265         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            12_730.30       567.66    13_297.97       0.3790          1.6551         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            12_730.30       660.83    13_391.14       0.5015          1.3834         4.67
IVF-Binary-512-nl316-pca (self)                       12_730.30     1_792.77    14_523.07       0.4011          1.5903         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          24_953.35       828.94    25_782.28       0.4047             NaN         8.41
IVF-Binary-1024-nl158-np12-rf0-random (query)         24_953.35       879.31    25_832.66       0.3672             NaN         8.41
IVF-Binary-1024-nl158-np17-rf0-random (query)         24_953.35       883.79    25_837.13       0.3461             NaN         8.41
IVF-Binary-1024-nl158-np7-rf10-random (query)         24_953.35       918.29    25_871.64       0.9890          1.0035         8.41
IVF-Binary-1024-nl158-np7-rf20-random (query)         24_953.35     1_024.75    25_978.10       0.9971          1.0007         8.41
IVF-Binary-1024-nl158-np12-rf10-random (query)        24_953.35       956.88    25_910.23       0.9827          1.0060         8.41
IVF-Binary-1024-nl158-np12-rf20-random (query)        24_953.35     1_036.40    25_989.75       0.9973          1.0009         8.41
IVF-Binary-1024-nl158-np17-rf10-random (query)        24_953.35       961.39    25_914.74       0.9758          1.0087         8.41
IVF-Binary-1024-nl158-np17-rf20-random (query)        24_953.35     1_059.63    26_012.98       0.9962          1.0013         8.41
IVF-Binary-1024-nl158-random (self)                   24_953.35     3_069.72    28_023.07       0.9830          1.0059         8.41
IVF-Binary-1024-nl223-np11-rf0-random (query)         22_655.55       830.32    23_485.87       0.3911             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         22_655.55       840.23    23_495.78       0.3763             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         22_655.55       859.71    23_515.26       0.3520             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        22_655.55       918.82    23_574.37       0.9890          1.0033         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        22_655.55       998.64    23_654.18       0.9984          1.0004         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        22_655.55       932.88    23_588.43       0.9860          1.0044         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        22_655.55     1_017.58    23_673.13       0.9980          1.0005         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        22_655.55       953.35    23_608.90       0.9785          1.0073         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        22_655.55     1_055.26    23_710.81       0.9969          1.0010         8.54
IVF-Binary-1024-nl223-random (self)                   22_655.55     3_028.04    25_683.59       0.9861          1.0045         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         22_929.62       855.42    23_785.03       0.3862             NaN         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)         22_929.62       860.77    23_790.39       0.3791             NaN         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)         22_929.62       931.04    23_860.65       0.3587             NaN         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)        22_929.62       943.54    23_873.15       0.9890          1.0033         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)        22_929.62     1_094.32    24_023.94       0.9986          1.0004         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)        22_929.62       938.18    23_867.80       0.9874          1.0039         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)        22_929.62     1_023.78    23_953.40       0.9983          1.0005         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)        22_929.62       959.00    23_888.62       0.9815          1.0062         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)        22_929.62     1_066.74    23_996.35       0.9974          1.0008         8.72
IVF-Binary-1024-nl316-random (self)                   22_929.62     3_057.02    25_986.64       0.9877          1.0038         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)             26_049.92       849.93    26_899.85       0.4021             NaN         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            26_049.92       872.68    26_922.59       0.3977             NaN         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            26_049.92       886.48    26_936.39       0.3959             NaN         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            26_049.92       934.17    26_984.08       0.8540          1.0604         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            26_049.92     1_012.41    27_062.33       0.9425          1.0187         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           26_049.92       960.70    27_010.62       0.8431          1.0671         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           26_049.92     1_048.52    27_098.43       0.9317          1.0230         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           26_049.92       985.51    27_035.43       0.8385          1.0701         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           26_049.92     1_078.86    27_128.78       0.9264          1.0254         8.42
IVF-Binary-1024-nl158-pca (self)                      26_049.92     3_122.74    29_172.66       0.8277          1.0772         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            23_820.37       868.22    24_688.59       0.3994             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            23_820.37       865.47    24_685.84       0.3980             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            23_820.37       883.19    24_703.56       0.3960             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           23_820.37       941.38    24_761.75       0.8497          1.0632         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           23_820.37     1_025.24    24_845.60       0.9391          1.0200         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           23_820.37       953.40    24_773.76       0.8454          1.0657         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           23_820.37     1_050.75    24_871.12       0.9342          1.0220         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           23_820.37       982.44    24_802.80       0.8393          1.0697         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           23_820.37     1_068.42    24_888.79       0.9272          1.0251         8.54
IVF-Binary-1024-nl223-pca (self)                      23_820.37     3_160.15    26_980.51       0.8300          1.0757         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            24_046.11       870.55    24_916.66       0.3988             NaN         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            24_046.11       873.70    24_919.81       0.3979             NaN         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            24_046.11       893.84    24_939.95       0.3963             NaN         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           24_046.11       954.71    25_000.83       0.8493          1.0634         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           24_046.11     1_037.82    25_083.93       0.9381          1.0205         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           24_046.11       958.58    25_004.69       0.8470          1.0648         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           24_046.11     1_044.29    25_090.40       0.9355          1.0215         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           24_046.11       987.81    25_033.92       0.8409          1.0685         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           24_046.11     1_067.73    25_113.84       0.9293          1.0242         8.73
IVF-Binary-1024-nl316-pca (self)                      24_046.11     3_122.73    27_168.84       0.8315          1.0747         8.73
IVF-Binary-512-nl158-np7-rf0-signed (query)           14_105.26       428.35    14_533.61       0.3916             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-signed (query)          14_105.26       439.51    14_544.77       0.3554             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-signed (query)          14_105.26       452.26    14_557.51       0.3353             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-signed (query)          14_105.26       516.80    14_622.06       0.9791          1.0056         4.36
IVF-Binary-512-nl158-np7-rf20-signed (query)          14_105.26       594.89    14_700.14       0.9960          1.0009         4.36
IVF-Binary-512-nl158-np12-rf10-signed (query)         14_105.26       540.74    14_646.00       0.9680          1.0098         4.36
IVF-Binary-512-nl158-np12-rf20-signed (query)         14_105.26       622.35    14_727.61       0.9953          1.0013         4.36
IVF-Binary-512-nl158-np17-rf10-signed (query)         14_105.26       554.57    14_659.82       0.9578          1.0135         4.36
IVF-Binary-512-nl158-np17-rf20-signed (query)         14_105.26       648.47    14_753.73       0.9933          1.0018         4.36
IVF-Binary-512-nl158-signed (self)                    14_105.26     1_717.32    15_822.58       0.9688          1.0094         4.36
IVF-Binary-512-nl223-np11-rf0-signed (query)          11_849.00       436.37    12_285.37       0.3780             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-signed (query)          11_849.00       444.16    12_293.15       0.3640             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-signed (query)          11_849.00       452.13    12_301.13       0.3413             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-signed (query)         11_849.00       524.83    12_373.82       0.9782          1.0057         4.49
IVF-Binary-512-nl223-np11-rf20-signed (query)         11_849.00       607.15    12_456.15       0.9972          1.0006         4.49
IVF-Binary-512-nl223-np14-rf10-signed (query)         11_849.00       533.13    12_382.12       0.9733          1.0074         4.49
IVF-Binary-512-nl223-np14-rf20-signed (query)         11_849.00       619.55    12_468.54       0.9965          1.0008         4.49
IVF-Binary-512-nl223-np21-rf10-signed (query)         11_849.00       552.84    12_401.83       0.9621          1.0114         4.49
IVF-Binary-512-nl223-np21-rf20-signed (query)         11_849.00       641.91    12_490.90       0.9943          1.0014         4.49
IVF-Binary-512-nl223-signed (self)                    11_849.00     1_707.06    13_556.06       0.9737          1.0073         4.49
IVF-Binary-512-nl316-np15-rf0-signed (query)          12_089.32       452.41    12_541.73       0.3736             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-signed (query)          12_089.32       454.66    12_543.98       0.3669             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-signed (query)          12_089.32       462.17    12_551.49       0.3475             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-signed (query)         12_089.32       546.08    12_635.40       0.9781          1.0056         4.67
IVF-Binary-512-nl316-np15-rf20-signed (query)         12_089.32       624.70    12_714.02       0.9976          1.0005         4.67
IVF-Binary-512-nl316-np17-rf10-signed (query)         12_089.32       544.85    12_634.17       0.9756          1.0066         4.67
IVF-Binary-512-nl316-np17-rf20-signed (query)         12_089.32       629.40    12_718.72       0.9971          1.0006         4.67
IVF-Binary-512-nl316-np25-rf10-signed (query)         12_089.32       557.95    12_647.27       0.9663          1.0098         4.67
IVF-Binary-512-nl316-np25-rf20-signed (query)         12_089.32       653.26    12_742.58       0.9953          1.0011         4.67
IVF-Binary-512-nl316-signed (self)                    12_089.32     1_787.22    13_876.54       0.9761          1.0062         4.67
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
Exhaustive (query)                                        31.09    15_613.05    15_644.13       1.0000          1.0000       146.48
Exhaustive (self)                                         31.09    52_535.28    52_566.37       1.0000          1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)              8_771.68       484.56     9_256.24       0.2669             NaN         2.28
ExhaustiveBinary-256-random-rf10 (query)               8_771.68       640.46     9_412.13       0.8498          1.0802         2.28
ExhaustiveBinary-256-random-rf20 (query)               8_771.68       798.40     9_570.07       0.9492          1.0191         2.28
ExhaustiveBinary-256-random (self)                     8_771.68     2_107.54    10_879.22       0.8514          1.0784         2.28
ExhaustiveBinary-256-pca_no_rr (query)                 9_376.77       491.50     9_868.27       0.1281             NaN         2.28
ExhaustiveBinary-256-pca-rf10 (query)                  9_376.77       642.10    10_018.87       0.3750          1.7664         2.28
ExhaustiveBinary-256-pca-rf20 (query)                  9_376.77       786.29    10_163.06       0.5003          1.4421         2.28
ExhaustiveBinary-256-pca (self)                        9_376.77     2_114.21    11_490.98       0.3725          1.7770         2.28
ExhaustiveBinary-512-random_no_rr (query)             17_215.97       858.08    18_074.05       0.2888             NaN         4.55
ExhaustiveBinary-512-random-rf10 (query)              17_215.97     1_017.84    18_233.81       0.9099          1.0520         4.55
ExhaustiveBinary-512-random-rf20 (query)              17_215.97     1_170.47    18_386.44       0.9755          1.0108         4.55
ExhaustiveBinary-512-random (self)                    17_215.97     3_350.60    20_566.57       0.9111          1.0500         4.55
ExhaustiveBinary-512-pca_no_rr (query)                18_049.23       871.78    18_921.01       0.1131             NaN         4.55
ExhaustiveBinary-512-pca-rf10 (query)                 18_049.23     1_018.66    19_067.89       0.3166          2.0536         4.55
ExhaustiveBinary-512-pca-rf20 (query)                 18_049.23     1_178.36    19_227.59       0.4179          1.6492         4.55
ExhaustiveBinary-512-pca (self)                       18_049.23     3_380.63    21_429.87       0.3146          2.0599         4.55
ExhaustiveBinary-1024-random_no_rr (query)            34_312.87     1_596.63    35_909.50       0.2977             NaN         9.10
ExhaustiveBinary-1024-random-rf10 (query)             34_312.87     1_761.87    36_074.74       0.9384          1.0420         9.10
ExhaustiveBinary-1024-random-rf20 (query)             34_312.87     1_925.82    36_238.69       0.9843          1.0081         9.10
ExhaustiveBinary-1024-random (self)                   34_312.87     5_871.59    40_184.46       0.9399          1.0400         9.10
ExhaustiveBinary-1024-pca_no_rr (query)               35_409.98     1_622.48    37_032.46       0.2379             NaN         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                35_409.98     1_784.11    37_194.09       0.6180          1.2507         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                35_409.98     1_948.87    37_358.84       0.7452          1.1303         9.11
ExhaustiveBinary-1024-pca (self)                      35_409.98     5_924.73    41_334.71       0.6017          1.2739         9.11
ExhaustiveBinary-768-signed_no_rr (query)             25_828.57     1_245.64    27_074.21       0.2944             NaN         6.83
ExhaustiveBinary-768-signed-rf10 (query)              25_828.57     1_399.73    27_228.30       0.9283          1.0452         6.83
ExhaustiveBinary-768-signed-rf20 (query)              25_828.57     1_559.50    27_388.07       0.9818          1.0090         6.83
ExhaustiveBinary-768-signed (self)                    25_828.57     4_645.68    30_474.25       0.9296          1.0433         6.83
IVF-Binary-256-nl158-np7-rf0-random (query)           13_712.89       351.08    14_063.97       0.3403             NaN         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)          13_712.89       362.20    14_075.09       0.3122             NaN         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)          13_712.89       369.17    14_082.06       0.2964             NaN         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)          13_712.89       465.19    14_178.09       0.9318          1.0212         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)          13_712.89       567.95    14_280.85       0.9864          1.0032         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)         13_712.89       479.62    14_192.52       0.9084          1.0317         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)         13_712.89       605.31    14_318.20       0.9790          1.0053         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)         13_712.89       495.12    14_208.01       0.8929          1.0399         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)         13_712.89       611.19    14_324.08       0.9728          1.0072         2.74
IVF-Binary-256-nl158-random (self)                    13_712.89     1_524.30    15_237.20       0.9092          1.0309         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)           9_786.62       367.63    10_154.25       0.3314             NaN         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)           9_786.62       398.66    10_185.28       0.3198             NaN         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)           9_786.62       380.66    10_167.28       0.3010             NaN         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)          9_786.62       479.33    10_265.95       0.9276          1.0225         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)          9_786.62       582.74    10_369.36       0.9861          1.0033         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)          9_786.62       483.88    10_270.50       0.9184          1.0266         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)          9_786.62       591.77    10_378.39       0.9826          1.0042         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)          9_786.62       508.62    10_295.24       0.8996          1.0358         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)          9_786.62       610.94    10_397.56       0.9757          1.0061         2.93
IVF-Binary-256-nl223-random (self)                     9_786.62     1_545.06    11_331.69       0.9190          1.0260         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)          10_128.53       393.93    10_522.46       0.3286             NaN         3.20
IVF-Binary-256-nl316-np17-rf0-random (query)          10_128.53       393.40    10_521.94       0.3234             NaN         3.20
IVF-Binary-256-nl316-np25-rf0-random (query)          10_128.53       399.89    10_528.43       0.3077             NaN         3.20
IVF-Binary-256-nl316-np15-rf10-random (query)         10_128.53       503.23    10_631.76       0.9284          1.0213         3.20
IVF-Binary-256-nl316-np15-rf20-random (query)         10_128.53       605.47    10_734.00       0.9866          1.0029         3.20
IVF-Binary-256-nl316-np17-rf10-random (query)         10_128.53       504.41    10_632.94       0.9235          1.0235         3.20
IVF-Binary-256-nl316-np17-rf20-random (query)         10_128.53       622.00    10_750.54       0.9849          1.0034         3.20
IVF-Binary-256-nl316-np25-rf10-random (query)         10_128.53       514.76    10_643.29       0.9073          1.0311         3.20
IVF-Binary-256-nl316-np25-rf20-random (query)         10_128.53       628.58    10_757.12       0.9792          1.0050         3.20
IVF-Binary-256-nl316-random (self)                    10_128.53     1_590.39    11_718.92       0.9242          1.0230         3.20
IVF-Binary-256-nl158-np7-rf0-pca (query)              14_110.63       367.80    14_478.42       0.1454             NaN         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)             14_110.63       379.26    14_489.89       0.1382             NaN         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)             14_110.63       374.86    14_485.48       0.1346             NaN         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)             14_110.63       470.10    14_580.72       0.4662          1.4677         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)             14_110.63       572.01    14_682.63       0.6300          1.2294         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)            14_110.63       483.10    14_593.73       0.4327          1.5393         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)            14_110.63       594.56    14_705.19       0.5877          1.2766         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)            14_110.63       498.83    14_609.45       0.4150          1.5829         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)            14_110.63       615.80    14_726.43       0.5655          1.3049         2.74
IVF-Binary-256-nl158-pca (self)                       14_110.63     1_557.14    15_667.77       0.4310          1.5438         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)             10_531.79       374.34    10_906.13       0.1435             NaN         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)             10_531.79       379.15    10_910.94       0.1404             NaN         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)             10_531.79       386.22    10_918.01       0.1360             NaN         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)            10_531.79       493.03    11_024.82       0.4603          1.4717         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)            10_531.79       589.44    11_121.23       0.6244          1.2313         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)            10_531.79       490.66    11_022.45       0.4461          1.5021         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)            10_531.79       607.76    11_139.55       0.6061          1.2515         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)            10_531.79       508.81    11_040.60       0.4237          1.5567         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)            10_531.79       619.86    11_151.65       0.5772          1.2875         2.93
IVF-Binary-256-nl223-pca (self)                       10_531.79     1_558.09    12_089.88       0.4440          1.5077         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)             10_900.32       398.08    11_298.41       0.1432             NaN         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)             10_900.32       398.71    11_299.04       0.1417             NaN         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)             10_900.32       404.72    11_305.04       0.1377             NaN         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)            10_900.32       509.82    11_410.15       0.4614          1.4682         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)            10_900.32       616.35    11_516.67       0.6269          1.2277         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)            10_900.32       514.25    11_414.58       0.4537          1.4846         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)            10_900.32       619.36    11_519.68       0.6166          1.2387         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)            10_900.32       520.77    11_421.09       0.4324          1.5326         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)            10_900.32       637.96    11_538.28       0.5891          1.2713         3.21
IVF-Binary-256-nl316-pca (self)                       10_900.32     1_621.74    12_522.07       0.4518          1.4886         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)           21_923.52       653.64    22_577.16       0.3662             NaN         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)          21_923.52       667.96    22_591.48       0.3358             NaN         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)          21_923.52       677.99    22_601.52       0.3196             NaN         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)          21_923.52       763.88    22_687.40       0.9712          1.0087         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)          21_923.52       873.14    22_796.66       0.9956          1.0012         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)         21_923.52       799.49    22_723.01       0.9565          1.0148         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)         21_923.52       890.30    22_813.82       0.9931          1.0019         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)         21_923.52       802.68    22_726.20       0.9452          1.0202         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)         21_923.52       991.15    22_914.67       0.9902          1.0028         5.02
IVF-Binary-512-nl158-random (self)                    21_923.52     2_533.23    24_456.75       0.9570          1.0139         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)          18_338.04       676.57    19_014.61       0.3580             NaN         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)          18_338.04       676.48    19_014.52       0.3461             NaN         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)          18_338.04       691.76    19_029.80       0.3254             NaN         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)         18_338.04       780.87    19_118.91       0.9695          1.0093         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)         18_338.04       889.98    19_228.02       0.9958          1.0011         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)         18_338.04       789.49    19_127.53       0.9635          1.0117         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)         18_338.04       892.85    19_230.89       0.9947          1.0015         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)         18_338.04       810.02    19_148.06       0.9506          1.0173         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)         18_338.04       919.55    19_257.59       0.9918          1.0022         5.21
IVF-Binary-512-nl223-random (self)                    18_338.04     2_550.61    20_888.65       0.9638          1.0111         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)          18_732.90       693.01    19_425.91       0.3548             NaN         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)          18_732.90       694.25    19_427.15       0.3490             NaN         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)          18_732.90       712.03    19_444.93       0.3318             NaN         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)         18_732.90       807.81    19_540.71       0.9699          1.0086         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)         18_732.90       904.82    19_637.72       0.9963          1.0008         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)         18_732.90       807.94    19_540.84       0.9667          1.0099         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)         18_732.90       911.43    19_644.33       0.9956          1.0010         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)         18_732.90       818.91    19_551.81       0.9558          1.0145         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)         18_732.90       941.59    19_674.49       0.9933          1.0017         5.48
IVF-Binary-512-nl316-random (self)                    18_732.90     2_619.78    21_352.69       0.9674          1.0093         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)              22_757.67       666.22    23_423.89       0.1309             NaN         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)             22_757.67       679.70    23_437.37       0.1239             NaN         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)             22_757.67       690.00    23_447.67       0.1204             NaN         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)             22_757.67       778.21    23_535.88       0.4224          1.5615         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)             22_757.67       877.49    23_635.16       0.5793          1.2883         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)            22_757.67       792.75    23_550.42       0.3879          1.6531         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)            22_757.67       906.87    23_664.54       0.5316          1.3527         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)            22_757.67       811.99    23_569.66       0.3695          1.7111         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)            22_757.67       930.20    23_687.87       0.5067          1.3929         5.02
IVF-Binary-512-nl158-pca (self)                       22_757.67     2_582.48    25_340.15       0.3862          1.6582         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)             19_185.48       686.79    19_872.27       0.1289             NaN         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)             19_185.48       688.52    19_874.00       0.1258             NaN         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)             19_185.48       698.07    19_883.55       0.1216             NaN         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)            19_185.48       806.56    19_992.03       0.4165          1.5664         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)            19_185.48       966.39    20_151.86       0.5722          1.2915         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)            19_185.48       800.22    19_985.69       0.4017          1.6053         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)            19_185.48       908.51    20_093.99       0.5516          1.3190         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)            19_185.48       831.13    20_016.61       0.3783          1.6763         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)            19_185.48       940.04    20_125.52       0.5197          1.3683         5.21
IVF-Binary-512-nl223-pca (self)                       19_185.48     2_594.07    21_779.55       0.4000          1.6102         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)             19_557.01       714.53    20_271.54       0.1285             NaN         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)             19_557.01       714.15    20_271.16       0.1271             NaN         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)             19_557.01       736.06    20_293.08       0.1231             NaN         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)            19_557.01       816.86    20_373.87       0.4178          1.5606         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)            19_557.01       921.63    20_478.64       0.5745          1.2869         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)            19_557.01       817.49    20_374.50       0.4099          1.5808         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)            19_557.01       925.56    20_482.58       0.5630          1.3018         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)            19_557.01       839.47    20_396.49       0.3875          1.6442         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)            19_557.01       947.48    20_504.49       0.5323          1.3462         5.48
IVF-Binary-512-nl316-pca (self)                       19_557.01     2_655.55    22_212.56       0.4083          1.5851         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)          39_011.19     1_264.99    40_276.18       0.3791             NaN         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)         39_011.19     1_302.05    40_313.24       0.3465             NaN         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)         39_011.19     1_300.97    40_312.16       0.3295             NaN         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)         39_011.19     1_370.83    40_382.02       0.9854          1.0053         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)         39_011.19     1_472.68    40_483.87       0.9977          1.0008         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)        39_011.19     1_443.60    40_454.79       0.9758          1.0097         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)        39_011.19     1_552.81    40_564.00       0.9966          1.0012         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)        39_011.19     1_441.83    40_453.02       0.9677          1.0136         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)        39_011.19     1_559.17    40_570.37       0.9949          1.0018         9.57
IVF-Binary-1024-nl158-random (self)                   39_011.19     4_608.01    43_619.20       0.9767          1.0087         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)         35_449.03     1_319.97    36_769.00       0.3703             NaN         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)         35_449.03     1_287.41    36_736.45       0.3574             NaN         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)         35_449.03     1_303.42    36_752.45       0.3359             NaN         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)        35_449.03     1_384.46    36_833.49       0.9846          1.0057         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)        35_449.03     1_485.32    36_934.35       0.9980          1.0008         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)        35_449.03     1_395.16    36_844.19       0.9805          1.0074         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)        35_449.03     1_502.93    36_951.96       0.9974          1.0010         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)        35_449.03     1_421.09    36_870.12       0.9715          1.0113         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)        35_449.03     1_536.60    36_985.63       0.9956          1.0015         9.76
IVF-Binary-1024-nl223-random (self)                   35_449.03     4_574.27    40_023.31       0.9812          1.0069         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)         35_851.09     1_300.60    37_151.70       0.3678             NaN        10.03
IVF-Binary-1024-nl316-np17-rf0-random (query)         35_851.09     1_324.82    37_175.92       0.3616             NaN        10.03
IVF-Binary-1024-nl316-np25-rf0-random (query)         35_851.09     1_319.81    37_170.90       0.3429             NaN        10.03
IVF-Binary-1024-nl316-np15-rf10-random (query)        35_851.09     1_408.36    37_259.46       0.9849          1.0051        10.03
IVF-Binary-1024-nl316-np15-rf20-random (query)        35_851.09     1_508.90    37_360.00       0.9982          1.0005        10.03
IVF-Binary-1024-nl316-np17-rf10-random (query)        35_851.09     1_420.80    37_271.89       0.9830          1.0060        10.03
IVF-Binary-1024-nl316-np17-rf20-random (query)        35_851.09     1_530.30    37_381.39       0.9980          1.0006        10.03
IVF-Binary-1024-nl316-np25-rf10-random (query)        35_851.09     1_459.08    37_310.18       0.9753          1.0093        10.03
IVF-Binary-1024-nl316-np25-rf20-random (query)        35_851.09     1_541.38    37_392.48       0.9966          1.0011        10.03
IVF-Binary-1024-nl316-random (self)                   35_851.09     4_613.49    40_464.59       0.9838          1.0054        10.03
IVF-Binary-1024-nl158-np7-rf0-pca (query)             40_114.46     1_287.83    41_402.29       0.2464             NaN         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)            40_114.46     1_303.32    41_417.78       0.2420             NaN         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)            40_114.46     1_337.97    41_452.43       0.2398             NaN         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)            40_114.46     1_390.57    41_505.03       0.6566          1.2083         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)            40_114.46     1_492.45    41_606.91       0.7952          1.0964         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)           40_114.46     1_413.85    41_528.31       0.6378          1.2271         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)           40_114.46     1_524.56    41_639.02       0.7704          1.1117         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)           40_114.46     1_465.71    41_580.18       0.6289          1.2373         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)           40_114.46     1_679.33    41_793.79       0.7594          1.1193         9.57
IVF-Binary-1024-nl158-pca (self)                      40_114.46     4_639.24    44_753.70       0.6231          1.2468         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)            36_556.54     1_301.42    37_857.97       0.2448             NaN         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)            36_556.54     1_309.79    37_866.33       0.2429             NaN         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)            36_556.54     1_325.28    37_881.83       0.2405             NaN         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)           36_556.54     1_406.69    37_963.24       0.6523          1.2110         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)           36_556.54     1_515.07    38_071.61       0.7909          1.0980         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)           36_556.54     1_453.62    38_010.16       0.6442          1.2192         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)           36_556.54     1_543.35    38_099.89       0.7798          1.1049         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)           36_556.54     1_443.41    37_999.95       0.6326          1.2325         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)           36_556.54     1_561.15    38_117.70       0.7644          1.1156         9.76
IVF-Binary-1024-nl223-pca (self)                      36_556.54     4_665.84    41_222.39       0.6303          1.2375         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)            36_960.98     1_325.17    38_286.15       0.2446             NaN        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)            36_960.98     1_324.30    38_285.28       0.2437             NaN        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)            36_960.98     1_338.87    38_299.85       0.2413             NaN        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)           36_960.98     1_428.47    38_389.45       0.6517          1.2115        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)           36_960.98     1_532.38    38_493.36       0.7904          1.0983        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)           36_960.98     1_432.24    38_393.22       0.6474          1.2159        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)           36_960.98     1_558.86    38_519.83       0.7846          1.1020        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)           36_960.98     1_451.14    38_412.12       0.6362          1.2280        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)           36_960.98     1_564.76    38_525.74       0.7696          1.1118        10.04
IVF-Binary-1024-nl316-pca (self)                      36_960.98     4_719.55    41_680.53       0.6342          1.2334        10.04
IVF-Binary-768-nl158-np7-rf0-signed (query)           30_548.25       962.37    31_510.63       0.3741             NaN         7.29
IVF-Binary-768-nl158-np12-rf0-signed (query)          30_548.25       978.05    31_526.31       0.3421             NaN         7.29
IVF-Binary-768-nl158-np17-rf0-signed (query)          30_548.25       999.49    31_547.74       0.3255             NaN         7.29
IVF-Binary-768-nl158-np7-rf10-signed (query)          30_548.25     1_076.84    31_625.09       0.9809          1.0063         7.29
IVF-Binary-768-nl158-np7-rf20-signed (query)          30_548.25     1_171.01    31_719.26       0.9971          1.0010         7.29
IVF-Binary-768-nl158-np12-rf10-signed (query)         30_548.25     1_091.81    31_640.07       0.9694          1.0113         7.29
IVF-Binary-768-nl158-np12-rf20-signed (query)         30_548.25     1_202.27    31_750.52       0.9956          1.0015         7.29
IVF-Binary-768-nl158-np17-rf10-signed (query)         30_548.25     1_130.67    31_678.92       0.9601          1.0157         7.29
IVF-Binary-768-nl158-np17-rf20-signed (query)         30_548.25     1_232.77    31_781.03       0.9935          1.0021         7.29
IVF-Binary-768-nl158-signed (self)                    30_548.25     3_563.10    34_111.36       0.9702          1.0104         7.29
IVF-Binary-768-nl223-np11-rf0-signed (query)          27_040.62       976.03    28_016.64       0.3653             NaN         7.48
IVF-Binary-768-nl223-np14-rf0-signed (query)          27_040.62       983.15    28_023.77       0.3528             NaN         7.48
IVF-Binary-768-nl223-np21-rf0-signed (query)          27_040.62     1_008.97    28_049.59       0.3315             NaN         7.48
IVF-Binary-768-nl223-np11-rf10-signed (query)         27_040.62     1_085.96    28_126.58       0.9798          1.0066         7.48
IVF-Binary-768-nl223-np11-rf20-signed (query)         27_040.62     1_185.72    28_226.34       0.9974          1.0009         7.48
IVF-Binary-768-nl223-np14-rf10-signed (query)         27_040.62     1_095.18    28_135.79       0.9750          1.0086         7.48
IVF-Binary-768-nl223-np14-rf20-signed (query)         27_040.62     1_200.93    28_241.55       0.9966          1.0011         7.48
IVF-Binary-768-nl223-np21-rf10-signed (query)         27_040.62     1_117.30    28_157.92       0.9645          1.0131         7.48
IVF-Binary-768-nl223-np21-rf20-signed (query)         27_040.62     1_250.23    28_290.84       0.9946          1.0017         7.48
IVF-Binary-768-nl223-signed (self)                    27_040.62     3_569.60    30_610.22       0.9756          1.0081         7.48
IVF-Binary-768-nl316-np15-rf0-signed (query)          27_358.44       996.48    28_354.93       0.3629             NaN         7.76
IVF-Binary-768-nl316-np17-rf0-signed (query)          27_358.44       998.71    28_357.16       0.3568             NaN         7.76
IVF-Binary-768-nl316-np25-rf0-signed (query)          27_358.44     1_016.09    28_374.54       0.3386             NaN         7.76
IVF-Binary-768-nl316-np15-rf10-signed (query)         27_358.44     1_106.61    28_465.05       0.9799          1.0062         7.76
IVF-Binary-768-nl316-np15-rf20-signed (query)         27_358.44     1_207.76    28_566.20       0.9978          1.0006         7.76
IVF-Binary-768-nl316-np17-rf10-signed (query)         27_358.44     1_199.69    28_558.13       0.9775          1.0072         7.76
IVF-Binary-768-nl316-np17-rf20-signed (query)         27_358.44     1_214.42    28_572.86       0.9974          1.0007         7.76
IVF-Binary-768-nl316-np25-rf10-signed (query)         27_358.44     1_126.58    28_485.02       0.9685          1.0109         7.76
IVF-Binary-768-nl316-np25-rf20-signed (query)         27_358.44     1_252.93    28_611.37       0.9957          1.0013         7.76
IVF-Binary-768-nl316-signed (self)                    27_358.44     3_628.16    30_986.60       0.9786          1.0065         7.76
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
Exhaustive (query)                                         9.81     4_133.54     4_143.35       1.0000          1.0000        48.83
Exhaustive (self)                                          9.81    13_815.79    13_825.60       1.0000          1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                           1_390.90     1_188.03     2_578.93       0.5172             NaN         2.84
ExhaustiveRaBitQ-rf5 (query)                           1_390.90     1_252.64     2_643.54       0.9146          1.0018         2.84
ExhaustiveRaBitQ-rf10 (query)                          1_390.90     1_311.16     2_702.06       0.9813          1.0003         2.84
ExhaustiveRaBitQ-rf20 (query)                          1_390.90     1_398.83     2_789.73       0.9982          1.0000         2.84
ExhaustiveRaBitQ (self)                                1_390.90     4_365.06     5_755.96       0.9819          1.0003         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       2_009.00       324.48     2_333.48       0.5206             NaN         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      2_009.00       528.69     2_537.69       0.5206             NaN         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      2_009.00       739.82     2_748.83       0.5206             NaN         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      2_009.00       409.98     2_418.98       0.9810          1.0003         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      2_009.00       488.96     2_497.97       0.9970          1.0001         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     2_009.00       623.95     2_632.95       0.9818          1.0003         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     2_009.00       698.12     2_707.12       0.9982          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     2_009.00       847.49     2_856.49       0.9818          1.0003         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     2_009.00       923.99     2_932.99       0.9982          1.0000         2.89
IVF-RaBitQ-nl158 (self)                                2_009.00     3_082.23     5_091.24       0.9983          1.0000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_089.77       421.62     1_511.40       0.5225             NaN         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_089.77       524.42     1_614.20       0.5224             NaN         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_089.77       763.56     1_853.34       0.5223             NaN         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_089.77       506.83     1_596.60       0.9817          1.0003         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_089.77       576.79     1_666.57       0.9976          1.0000         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_089.77       618.01     1_707.78       0.9820          1.0003         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_089.77       693.15     1_782.93       0.9982          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_089.77       866.83     1_956.60       0.9821          1.0003         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_089.77       948.16     2_037.93       0.9983          1.0000         2.95
IVF-RaBitQ-nl223 (self)                                1_089.77     3_151.79     4_241.56       0.9984          1.0000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_280.56       491.05     1_771.61       0.5259             NaN         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_280.56       557.17     1_837.73       0.5258             NaN         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_280.56       791.29     2_071.85       0.5257             NaN         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_280.56       582.01     1_862.57       0.9824          1.0003         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_280.56       653.31     1_933.86       0.9981          1.0000         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_280.56       643.45     1_924.00       0.9826          1.0003         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_280.56       746.87     2_027.42       0.9983          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_280.56       898.99     2_179.55       0.9826          1.0003         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_280.56       976.34     2_256.89       0.9984          1.0000         3.04
IVF-RaBitQ-nl316 (self)                                1_280.56     3_324.60     4_605.15       0.9985          1.0000         3.04
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
Exhaustive (query)                                        21.02     9_689.21     9_710.23       1.0000          1.0000        97.66
Exhaustive (self)                                         21.02    32_463.83    32_484.85       1.0000          1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           3_928.63     2_964.70     6_893.33       0.5146             NaN         5.23
ExhaustiveRaBitQ-rf5 (query)                           3_928.63     3_056.87     6_985.50       0.9105          1.0013         5.23
ExhaustiveRaBitQ-rf10 (query)                          3_928.63     3_152.06     7_080.69       0.9792          1.0002         5.23
ExhaustiveRaBitQ-rf20 (query)                          3_928.63     3_232.09     7_160.73       0.9979          1.0000         5.23
ExhaustiveRaBitQ (self)                                3_928.63    10_401.53    14_330.17       0.9793          1.0002         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_327.79       863.07     6_190.85       0.5155             NaN         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_327.79     1_447.18     6_774.96       0.5154             NaN         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_327.79     2_242.58     7_570.36       0.5154             NaN         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_327.79       984.57     6_312.35       0.9794          1.0002         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_327.79     1_066.84     6_394.63       0.9975          1.0000         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_327.79     1_573.34     6_901.12       0.9797          1.0002         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_327.79     1_668.64     6_996.42       0.9980          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_327.79     2_202.63     7_530.41       0.9797          1.0002         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_327.79     2_292.32     7_620.10       0.9980          1.0000         5.32
IVF-RaBitQ-nl158 (self)                                5_327.79     7_673.88    13_001.67       0.9979          1.0000         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_218.69     1_221.58     4_440.28       0.5171             NaN         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_218.69     1_550.34     4_769.03       0.5170             NaN         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_218.69     2_280.03     5_498.72       0.5170             NaN         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_218.69     1_323.85     4_542.55       0.9789          1.0002         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_218.69     1_407.55     4_626.24       0.9966          1.0000         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_218.69     1_630.17     4_848.87       0.9799          1.0002         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_218.69     1_744.03     4_962.72       0.9978          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_218.69     2_363.47     5_582.17       0.9799          1.0002         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_218.69     2_472.26     5_690.96       0.9979          1.0000         5.44
IVF-RaBitQ-nl223 (self)                                3_218.69     8_370.39    11_589.08       0.9979          1.0000         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      3_539.72     1_498.47     5_038.19       0.5189             NaN         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      3_539.72     1_685.82     5_225.54       0.5189             NaN         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      3_539.72     2_447.73     5_987.45       0.5189             NaN         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     3_539.72     1_614.11     5_153.83       0.9795          1.0002         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     3_539.72     1_710.57     5_250.29       0.9970          1.0000         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     3_539.72     1_799.99     5_339.72       0.9800          1.0002         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     3_539.72     1_902.28     5_442.00       0.9977          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     3_539.72     2_565.54     6_105.26       0.9803          1.0002         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     3_539.72     2_686.45     6_226.17       0.9981          1.0000         5.63
IVF-RaBitQ-nl316 (self)                                3_539.72     8_888.65    12_428.37       0.9980          1.0000         5.63
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
Exhaustive (query)                                        30.27    16_454.14    16_484.40       1.0000          1.0000       146.48
Exhaustive (self)                                         30.27    52_468.00    52_498.26       1.0000          1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           8_126.39     6_105.91    14_232.30       0.5107             NaN         8.11
ExhaustiveRaBitQ-rf5 (query)                           8_126.39     6_134.50    14_260.89       0.9042          1.0011         8.11
ExhaustiveRaBitQ-rf10 (query)                          8_126.39     6_211.70    14_338.09       0.9776          1.0002         8.11
ExhaustiveRaBitQ-rf20 (query)                          8_126.39     6_338.20    14_464.59       0.9975          1.0000         8.11
ExhaustiveRaBitQ (self)                                8_126.39    20_690.74    28_817.13       0.9776          1.0002         8.11
IVF-RaBitQ-nl158-np7-rf0 (query)                      10_281.99     1_809.15    12_091.14       0.5134             NaN         8.25
IVF-RaBitQ-nl158-np12-rf0 (query)                     10_281.99     2_955.59    13_237.58       0.5133             NaN         8.25
IVF-RaBitQ-nl158-np17-rf0 (query)                     10_281.99     4_165.21    14_447.20       0.5133             NaN         8.25
IVF-RaBitQ-nl158-np7-rf10 (query)                     10_281.99     1_914.17    12_196.16       0.9761          1.0002         8.25
IVF-RaBitQ-nl158-np7-rf20 (query)                     10_281.99     2_037.57    12_319.56       0.9960          1.0000         8.25
IVF-RaBitQ-nl158-np12-rf10 (query)                    10_281.99     3_078.65    13_360.64       0.9773          1.0002         8.25
IVF-RaBitQ-nl158-np12-rf20 (query)                    10_281.99     3_195.98    13_477.97       0.9976          1.0000         8.25
IVF-RaBitQ-nl158-np17-rf10 (query)                    10_281.99     4_294.34    14_576.33       0.9773          1.0002         8.25
IVF-RaBitQ-nl158-np17-rf20 (query)                    10_281.99     4_432.21    14_714.20       0.9976          1.0000         8.25
IVF-RaBitQ-nl158 (self)                               10_281.99    14_709.33    24_991.32       0.9975          1.0000         8.25
IVF-RaBitQ-nl223-np11-rf0 (query)                      6_932.77     2_627.68     9_560.44       0.5147             NaN         8.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      6_932.77     3_304.97    10_237.73       0.5146             NaN         8.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      6_932.77     4_924.23    11_857.00       0.5145             NaN         8.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     6_932.77     2_763.03     9_695.79       0.9782          1.0002         8.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     6_932.77     2_868.67     9_801.44       0.9970          1.0000         8.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     6_932.77     3_421.65    10_354.42       0.9787          1.0002         8.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     6_932.77     3_541.87    10_474.64       0.9977          1.0000         8.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     6_932.77     5_049.82    11_982.58       0.9786          1.0002         8.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     6_932.77     5_170.68    12_103.45       0.9977          1.0000         8.44
IVF-RaBitQ-nl223 (self)                                6_932.77    17_252.14    24_184.91       0.9978          1.0000         8.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      7_528.11     3_369.38    10_897.49       0.5160             NaN         8.71
IVF-RaBitQ-nl316-np17-rf0 (query)                      7_528.11     3_801.21    11_329.32       0.5160             NaN         8.71
IVF-RaBitQ-nl316-np25-rf0 (query)                      7_528.11     5_525.78    13_053.88       0.5160             NaN         8.71
IVF-RaBitQ-nl316-np15-rf10 (query)                     7_528.11     3_572.38    11_100.48       0.9784          1.0002         8.71
IVF-RaBitQ-nl316-np15-rf20 (query)                     7_528.11     3_688.08    11_216.18       0.9973          1.0000         8.71
IVF-RaBitQ-nl316-np17-rf10 (query)                     7_528.11     3_987.32    11_515.43       0.9787          1.0002         8.71
IVF-RaBitQ-nl316-np17-rf20 (query)                     7_528.11     4_143.57    11_671.68       0.9977          1.0000         8.71
IVF-RaBitQ-nl316-np25-rf10 (query)                     7_528.11     5_804.57    13_332.68       0.9788          1.0002         8.71
IVF-RaBitQ-nl316-np25-rf20 (query)                     7_528.11     5_901.36    13_429.46       0.9978          1.0000         8.71
IVF-RaBitQ-nl316 (self)                                7_528.11    19_346.03    26_874.14       0.9978          1.0000         8.71
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
Exhaustive (query)                                         9.99     4_154.53     4_164.52       1.0000          1.0000        48.83
Exhaustive (self)                                          9.99    14_301.17    14_311.16       1.0000          1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                           1_327.34     1_040.58     2_367.92       0.7288             NaN         2.84
ExhaustiveRaBitQ-rf5 (query)                           1_327.34     1_098.21     2_425.55       0.9969          1.0001         2.84
ExhaustiveRaBitQ-rf10 (query)                          1_327.34     1_159.94     2_487.28       0.9999          1.0000         2.84
ExhaustiveRaBitQ-rf20 (query)                          1_327.34     1_262.72     2_590.06       1.0000          1.0000         2.84
ExhaustiveRaBitQ (self)                                1_327.34     3_855.12     5_182.46       1.0000          1.0000         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_939.90       288.95     2_228.85       0.7296             NaN         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_939.90       413.01     2_352.91       0.7296             NaN         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_939.90       570.95     2_510.86       0.7296             NaN         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_939.90       381.89     2_321.79       0.9999          1.0000         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_939.90       463.74     2_403.65       1.0000          1.0000         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_939.90       507.49     2_447.39       0.9999          1.0000         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_939.90       584.44     2_524.34       1.0000          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_939.90       656.49     2_596.40       0.9999          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_939.90       729.64     2_669.55       1.0000          1.0000         2.89
IVF-RaBitQ-nl158 (self)                                1_939.90     2_434.68     4_374.58       1.0000          1.0000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_268.59       381.55     1_650.14       0.7351             NaN         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_268.59       465.65     1_734.25       0.7351             NaN         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_268.59       664.53     1_933.12       0.7351             NaN         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_268.59       472.83     1_741.42       1.0000          1.0000         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_268.59       545.74     1_814.33       1.0000          1.0000         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_268.59       565.28     1_833.87       1.0000          1.0000         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_268.59       636.29     1_904.88       1.0000          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_268.59       766.82     2_035.41       1.0000          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_268.59       850.94     2_119.53       1.0000          1.0000         2.95
IVF-RaBitQ-nl223 (self)                                1_268.59     2_825.04     4_093.63       1.0000          1.0000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_493.91       467.17     1_961.09       0.7372             NaN         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_493.91       509.98     2_003.89       0.7372             NaN         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_493.91       720.70     2_214.62       0.7372             NaN         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_493.91       555.02     2_048.94       1.0000          1.0000         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_493.91       628.23     2_122.14       1.0000          1.0000         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_493.91       600.62     2_094.54       1.0000          1.0000         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_493.91       677.39     2_171.31       1.0000          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_493.91       815.65     2_309.56       1.0000          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_493.91       894.17     2_388.09       1.0000          1.0000         3.04
IVF-RaBitQ-nl316 (self)                                1_493.91     2_978.31     4_472.22       1.0000          1.0000         3.04
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
Exhaustive (query)                                        20.02     9_605.51     9_625.53       1.0000          1.0000        97.66
Exhaustive (self)                                         20.02    31_988.27    32_008.29       1.0000          1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           3_649.81     2_810.08     6_459.90       0.7431             NaN         5.23
ExhaustiveRaBitQ-rf5 (query)                           3_649.81     2_889.28     6_539.09       0.9978          1.0000         5.23
ExhaustiveRaBitQ-rf10 (query)                          3_649.81     2_943.33     6_593.14       1.0000          1.0000         5.23
ExhaustiveRaBitQ-rf20 (query)                          3_649.81     3_048.56     6_698.37       1.0000          1.0000         5.23
ExhaustiveRaBitQ (self)                                3_649.81     9_778.04    13_427.85       1.0000          1.0000         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       4_922.55       806.80     5_729.34       0.7438             NaN         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      4_922.55     1_218.48     6_141.02       0.7438             NaN         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      4_922.55     1_646.76     6_569.31       0.7438             NaN         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      4_922.55       923.84     5_846.39       1.0000          1.0000         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      4_922.55     1_023.94     5_946.49       1.0000          1.0000         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     4_922.55     1_335.06     6_257.61       1.0000          1.0000         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     4_922.55     1_437.14     6_359.69       1.0000          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     4_922.55     1_765.82     6_688.36       1.0000          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     4_922.55     1_870.13     6_792.68       1.0000          1.0000         5.32
IVF-RaBitQ-nl158 (self)                                4_922.55     6_222.43    11_144.98       1.0000          1.0000         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_265.25     1_160.77     4_426.03       0.7471             NaN         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_265.25     1_409.45     4_674.71       0.7475             NaN         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_265.25     2_064.36     5_329.61       0.7475             NaN         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_265.25     1_282.69     4_547.94       0.9987          1.0001         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_265.25     1_368.52     4_633.78       0.9988          1.0001         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_265.25     1_526.98     4_792.23       1.0000          1.0000         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_265.25     1_625.40     4_890.65       1.0000          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_265.25     2_175.49     5_440.75       1.0000          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_265.25     2_263.08     5_528.34       1.0000          1.0000         5.44
IVF-RaBitQ-nl223 (self)                                3_265.25     7_516.98    10_782.23       1.0000          1.0000         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      3_730.27     1_448.73     5_179.00       0.7478             NaN         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      3_730.27     1_618.06     5_348.33       0.7480             NaN         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      3_730.27     2_289.71     6_019.98       0.7481             NaN         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     3_730.27     1_561.73     5_292.00       0.9989          1.0001         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     3_730.27     1_697.31     5_427.58       0.9989          1.0000         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     3_730.27     1_748.96     5_479.23       0.9998          1.0000         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     3_730.27     1_835.42     5_565.69       0.9998          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     3_730.27     2_411.11     6_141.38       1.0000          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     3_730.27     2_502.75     6_233.02       1.0000          1.0000         5.63
IVF-RaBitQ-nl316 (self)                                3_730.27     8_341.47    12_071.74       1.0000          1.0000         5.63
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
Exhaustive (query)                                        30.36    15_648.79    15_679.15       1.0000          1.0000       146.48
Exhaustive (self)                                         30.36    52_848.28    52_878.64       1.0000          1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           7_617.20     5_681.35    13_298.56       0.7244             NaN         8.11
ExhaustiveRaBitQ-rf5 (query)                           7_617.20     6_239.07    13_856.27       0.9954          1.0000         8.11
ExhaustiveRaBitQ-rf10 (query)                          7_617.20     6_171.00    13_788.20       0.9999          1.0000         8.11
ExhaustiveRaBitQ-rf20 (query)                          7_617.20     6_812.32    14_429.53       1.0000          1.0000         8.11
ExhaustiveRaBitQ (self)                                7_617.20    19_587.95    27_205.16       1.0000          1.0000         8.11
IVF-RaBitQ-nl158-np7-rf0 (query)                       9_662.91     1_719.84    11_382.75       0.7260             NaN         8.25
IVF-RaBitQ-nl158-np12-rf0 (query)                      9_662.91     2_710.17    12_373.09       0.7260             NaN         8.25
IVF-RaBitQ-nl158-np17-rf0 (query)                      9_662.91     3_714.12    13_377.03       0.7260             NaN         8.25
IVF-RaBitQ-nl158-np7-rf10 (query)                      9_662.91     1_864.74    11_527.66       0.9999          1.0000         8.25
IVF-RaBitQ-nl158-np7-rf20 (query)                      9_662.91     1_974.62    11_637.53       1.0000          1.0000         8.25
IVF-RaBitQ-nl158-np12-rf10 (query)                     9_662.91     2_836.87    12_499.78       0.9999          1.0000         8.25
IVF-RaBitQ-nl158-np12-rf20 (query)                     9_662.91     2_982.59    12_645.50       1.0000          1.0000         8.25
IVF-RaBitQ-nl158-np17-rf10 (query)                     9_662.91     3_830.81    13_493.72       0.9999          1.0000         8.25
IVF-RaBitQ-nl158-np17-rf20 (query)                     9_662.91     3_973.62    13_636.53       1.0000          1.0000         8.25
IVF-RaBitQ-nl158 (self)                                9_662.91    13_137.27    22_800.18       1.0000          1.0000         8.25
IVF-RaBitQ-nl223-np11-rf0 (query)                      7_038.94     2_511.41     9_550.35       0.7271             NaN         8.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      7_038.94     3_116.69    10_155.63       0.7272             NaN         8.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      7_038.94     4_516.21    11_555.15       0.7272             NaN         8.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     7_038.94     2_644.36     9_683.30       0.9997          1.0000         8.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     7_038.94     2_759.34     9_798.28       0.9998          1.0000         8.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     7_038.94     3_227.91    10_266.85       0.9999          1.0000         8.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     7_038.94     3_351.00    10_389.94       1.0000          1.0000         8.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     7_038.94     4_639.58    11_678.52       0.9999          1.0000         8.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     7_038.94     4_774.24    11_813.18       1.0000          1.0000         8.44
IVF-RaBitQ-nl223 (self)                                7_038.94    15_849.44    22_888.38       1.0000          1.0000         8.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      7_670.38     3_276.83    10_947.21       0.7283             NaN         8.71
IVF-RaBitQ-nl316-np17-rf0 (query)                      7_670.38     3_680.78    11_351.15       0.7286             NaN         8.71
IVF-RaBitQ-nl316-np25-rf0 (query)                      7_670.38     5_255.14    12_925.52       0.7286             NaN         8.71
IVF-RaBitQ-nl316-np15-rf10 (query)                     7_670.38     3_439.97    11_110.35       0.9987          1.0000         8.71
IVF-RaBitQ-nl316-np15-rf20 (query)                     7_670.38     3_533.66    11_204.04       0.9988          1.0000         8.71
IVF-RaBitQ-nl316-np17-rf10 (query)                     7_670.38     3_807.13    11_477.51       0.9996          1.0000         8.71
IVF-RaBitQ-nl316-np17-rf20 (query)                     7_670.38     3_959.04    11_629.42       0.9997          1.0000         8.71
IVF-RaBitQ-nl316-np25-rf10 (query)                     7_670.38     5_374.70    13_045.08       0.9999          1.0000         8.71
IVF-RaBitQ-nl316-np25-rf20 (query)                     7_670.38     5_484.73    13_155.11       1.0000          1.0000         8.71
IVF-RaBitQ-nl316 (self)                                7_670.38    18_325.79    25_996.17       1.0000          1.0000         8.71
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
Exhaustive (query)                                         9.75     4_112.38     4_122.13       1.0000          1.0000        48.83
Exhaustive (self)                                          9.75    13_857.16    13_866.91       1.0000          1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                           1_429.12     1_396.69     2_825.82       0.8680             NaN         2.84
ExhaustiveRaBitQ-rf5 (query)                           1_429.12     1_478.91     2_908.03       1.0000          1.0000         2.84
ExhaustiveRaBitQ-rf10 (query)                          1_429.12     1_721.58     3_150.70       1.0000          1.0000         2.84
ExhaustiveRaBitQ-rf20 (query)                          1_429.12     1_810.15     3_239.27       1.0000          1.0000         2.84
ExhaustiveRaBitQ (self)                                1_429.12     5_341.65     6_770.77       1.0000          1.0000         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       2_209.30       475.39     2_684.68       0.8725             NaN         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      2_209.30       689.53     2_898.83       0.8730             NaN         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      2_209.30       961.88     3_171.18       0.8730             NaN         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      2_209.30       498.31     2_707.61       0.9976          1.0005         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      2_209.30       576.89     2_786.18       0.9976          1.0005         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     2_209.30       743.70     2_952.99       0.9999          1.0000         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     2_209.30       832.82     3_042.11       0.9999          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     2_209.30     1_022.39     3_231.69       1.0000          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     2_209.30     1_108.11     3_317.40       1.0000          1.0000         2.89
IVF-RaBitQ-nl158 (self)                                2_209.30     3_685.05     5_894.34       1.0000          1.0000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_086.15       447.70     1_533.85       0.8832             NaN         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_086.15       560.15     1_646.30       0.8833             NaN         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_086.15       821.47     1_907.62       0.8832             NaN         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_086.15       540.67     1_626.82       0.9994          1.0001         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_086.15       618.99     1_705.15       0.9994          1.0001         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_086.15       667.03     1_753.18       0.9999          1.0000         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_086.15       789.18     1_875.33       0.9999          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_086.15       941.53     2_027.68       1.0000          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_086.15     1_029.52     2_115.68       1.0000          1.0000         2.95
IVF-RaBitQ-nl223 (self)                                1_086.15     3_475.68     4_561.83       1.0000          1.0000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_236.94       516.80     1_753.74       0.8894             NaN         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_236.94       583.67     1_820.60       0.8894             NaN         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_236.94       932.02     2_168.96       0.8894             NaN         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_236.94       629.76     1_866.70       0.9997          1.0001         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_236.94       717.89     1_954.83       0.9997          1.0001         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_236.94       682.92     1_919.86       0.9998          1.0000         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_236.94       765.23     2_002.17       0.9998          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_236.94       947.40     2_184.34       1.0000          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_236.94     1_040.47     2_277.40       1.0000          1.0000         3.04
IVF-RaBitQ-nl316 (self)                                1_236.94     3_475.98     4_712.92       1.0000          1.0000         3.04
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
Exhaustive (query)                                        20.23     9_574.90     9_595.13       1.0000          1.0000        97.66
Exhaustive (self)                                         20.23    32_206.91    32_227.14       1.0000          1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           3_942.23     3_362.32     7_304.56       0.9025             NaN         5.23
ExhaustiveRaBitQ-rf5 (query)                           3_942.23     3_436.92     7_379.16       1.0000          1.0000         5.23
ExhaustiveRaBitQ-rf10 (query)                          3_942.23     3_559.45     7_501.68       1.0000          1.0000         5.23
ExhaustiveRaBitQ-rf20 (query)                          3_942.23     3_697.96     7_640.19       1.0000          1.0000         5.23
ExhaustiveRaBitQ (self)                                3_942.23    11_759.91    15_702.15       1.0000          1.0000         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_339.05       961.87     6_300.92       0.9066             NaN         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_339.05     1_609.00     6_948.05       0.9071             NaN         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_339.05     2_269.64     7_608.69       0.9071             NaN         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_339.05     1_092.16     6_431.21       0.9985          1.0003         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_339.05     1_168.84     6_507.89       0.9985          1.0003         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_339.05     1_731.05     7_070.10       0.9999          1.0000         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_339.05     1_847.34     7_186.39       0.9999          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_339.05     2_386.47     7_725.52       1.0000          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_339.05     2_513.69     7_852.74       1.0000          1.0000         5.32
IVF-RaBitQ-nl158 (self)                                5_339.05     8_357.65    13_696.70       1.0000          1.0000         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_058.77     1_253.99     4_312.76       0.9151             NaN         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_058.77     1_582.81     4_641.58       0.9151             NaN         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_058.77     2_362.23     5_421.00       0.9151             NaN         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_058.77     1_377.48     4_436.25       0.9997          1.0000         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_058.77     1_458.44     4_517.21       0.9997          1.0000         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_058.77     1_697.01     4_755.77       1.0000          1.0000         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_058.77     1_804.36     4_863.13       1.0000          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_058.77     2_476.48     5_535.25       1.0000          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_058.77     2_576.61     5_635.38       1.0000          1.0000         5.44
IVF-RaBitQ-nl223 (self)                                3_058.77     8_588.97    11_647.74       1.0000          1.0000         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      3_298.67     1_544.26     4_842.93       0.9190             NaN         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      3_298.67     1_753.11     5_051.78       0.9190             NaN         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      3_298.67     2_528.34     5_827.02       0.9190             NaN         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     3_298.67     1_683.69     4_982.36       0.9999          1.0000         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     3_298.67     1_755.94     5_054.61       0.9999          1.0000         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     3_298.67     1_867.97     5_166.64       0.9999          1.0000         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     3_298.67     1_960.47     5_259.14       0.9999          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     3_298.67     2_684.56     5_983.23       1.0000          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     3_298.67     2_772.19     6_070.86       1.0000          1.0000         5.63
IVF-RaBitQ-nl316 (self)                                3_298.67     9_235.80    12_534.47       1.0000          1.0000         5.63
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
Exhaustive (query)                                        30.19    15_706.94    15_737.12       1.0000          1.0000       146.48
Exhaustive (self)                                         30.19    52_281.71    52_311.89       1.0000          1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           8_104.96     6_541.45    14_646.42       0.9249             NaN         8.11
ExhaustiveRaBitQ-rf5 (query)                           8_104.96     6_600.15    14_705.12       1.0000          1.0000         8.11
ExhaustiveRaBitQ-rf10 (query)                          8_104.96     6_685.59    14_790.56       1.0000          1.0000         8.11
ExhaustiveRaBitQ-rf20 (query)                          8_104.96     6_825.17    14_930.13       1.0000          1.0000         8.11
ExhaustiveRaBitQ (self)                                8_104.96    22_393.16    30_498.13       1.0000          1.0000         8.11
IVF-RaBitQ-nl158-np7-rf0 (query)                      10_230.20     1_960.62    12_190.82       0.9272             NaN         8.25
IVF-RaBitQ-nl158-np12-rf0 (query)                     10_230.20     3_274.82    13_505.02       0.9274             NaN         8.25
IVF-RaBitQ-nl158-np17-rf0 (query)                     10_230.20     4_559.83    14_790.03       0.9274             NaN         8.25
IVF-RaBitQ-nl158-np7-rf10 (query)                     10_230.20     2_084.91    12_315.11       0.9996          1.0001         8.25
IVF-RaBitQ-nl158-np7-rf20 (query)                     10_230.20     2_193.36    12_423.56       0.9996          1.0001         8.25
IVF-RaBitQ-nl158-np12-rf10 (query)                    10_230.20     3_402.82    13_633.02       1.0000          1.0000         8.25
IVF-RaBitQ-nl158-np12-rf20 (query)                    10_230.20     3_524.43    13_754.63       1.0000          1.0000         8.25
IVF-RaBitQ-nl158-np17-rf10 (query)                    10_230.20     4_724.70    14_954.90       1.0000          1.0000         8.25
IVF-RaBitQ-nl158-np17-rf20 (query)                    10_230.20     4_866.25    15_096.45       1.0000          1.0000         8.25
IVF-RaBitQ-nl158 (self)                               10_230.20    16_140.02    26_370.22       1.0000          1.0000         8.25
IVF-RaBitQ-nl223-np11-rf0 (query)                      6_686.42     2_706.34     9_392.76       0.9324             NaN         8.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      6_686.42     3_437.84    10_124.26       0.9324             NaN         8.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      6_686.42     5_152.49    11_838.91       0.9324             NaN         8.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     6_686.42     2_869.98     9_556.40       0.9999          1.0000         8.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     6_686.42     2_948.61     9_635.03       0.9999          1.0000         8.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     6_686.42     3_553.60    10_240.02       1.0000          1.0000         8.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     6_686.42     3_653.25    10_339.67       1.0000          1.0000         8.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     6_686.42     5_208.16    11_894.58       1.0000          1.0000         8.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     6_686.42     5_320.94    12_007.36       1.0000          1.0000         8.44
IVF-RaBitQ-nl223 (self)                                6_686.42    17_785.13    24_471.56       1.0000          1.0000         8.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      7_056.95     3_428.98    10_485.92       0.9360             NaN         8.71
IVF-RaBitQ-nl316-np17-rf0 (query)                      7_056.95     3_870.97    10_927.91       0.9360             NaN         8.71
IVF-RaBitQ-nl316-np25-rf0 (query)                      7_056.95     5_662.52    12_719.47       0.9360             NaN         8.71
IVF-RaBitQ-nl316-np15-rf10 (query)                     7_056.95     3_551.25    10_608.19       1.0000          1.0000         8.71
IVF-RaBitQ-nl316-np15-rf20 (query)                     7_056.95     3_708.51    10_765.46       1.0000          1.0000         8.71
IVF-RaBitQ-nl316-np17-rf10 (query)                     7_056.95     3_987.94    11_044.88       1.0000          1.0000         8.71
IVF-RaBitQ-nl316-np17-rf20 (query)                     7_056.95     4_109.47    11_166.42       1.0000          1.0000         8.71
IVF-RaBitQ-nl316-np25-rf10 (query)                     7_056.95     5_768.72    12_825.66       1.0000          1.0000         8.71
IVF-RaBitQ-nl316-np25-rf20 (query)                     7_056.95     5_903.94    12_960.89       1.0000          1.0000         8.71
IVF-RaBitQ-nl316 (self)                                7_056.95    19_630.80    26_687.74       1.0000          1.0000         8.71
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
generated by neural networks -- the area this index shines in.

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
Exhaustive (query)                                         9.78     4_102.02     4_111.80       1.0000          1.0000        48.83
Exhaustive (self)                                          9.78    13_902.69    13_912.47       1.0000          1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              157.97       391.10       549.07       0.0109             NaN         7.12
ExhaustiveTQ-b2-rf5 (query)                              157.97       433.66       591.64       0.0526          1.2562         7.12
ExhaustiveTQ-b2-rf10 (query)                             157.97       556.86       714.83       0.1030          1.1894         7.12
ExhaustiveTQ-b2-rf20 (query)                             157.97       926.45     1_084.43       0.2003          1.1318         7.12
ExhaustiveTQ-b2 (self)                                   157.97     3_088.97     3_246.94       0.1995          1.1335         7.12
ExhaustiveTQ-b4-rf0 (query)                              234.34       569.52       803.86       0.0132             NaN        13.22
ExhaustiveTQ-b4-rf5 (query)                              234.34       652.97       887.31       0.0576          1.2376        13.22
ExhaustiveTQ-b4-rf10 (query)                             234.34       775.70     1_010.04       0.1079          1.1773        13.22
ExhaustiveTQ-b4-rf20 (query)                             234.34     1_146.08     1_380.42       0.2030          1.1256        13.22
ExhaustiveTQ-b4 (self)                                   234.34     3_812.85     4_047.19       0.2033          1.1266        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_362.98       106.67     1_469.65       0.0116             NaN         7.81
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_362.98       140.73     1_503.70       0.0109             NaN         7.81
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_362.98       171.92     1_534.90       0.0109             NaN         7.81
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_362.98       265.93     1_628.91       0.1105          1.1790         7.81
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_362.98       495.06     1_858.04       0.2158          1.1228         7.81
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_362.98       317.71     1_680.68       0.1035          1.1886         7.81
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_362.98       582.68     1_945.66       0.2012          1.1311         7.81
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_362.98       353.49     1_716.47       0.1030          1.1894         7.81
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_362.98       647.50     2_010.48       0.2003          1.1318         7.81
IVF-TQ-b2-nl158 (self)                                 1_362.98     1_303.13     2_666.10       0.1995          1.1335         7.81
IVF-TQ-b2-nl223-np11-rf0 (query)                         698.76       120.24       819.00       0.0113             NaN         7.94
IVF-TQ-b2-nl223-np14-rf0 (query)                         698.76       136.02       834.79       0.0109             NaN         7.94
IVF-TQ-b2-nl223-np21-rf0 (query)                         698.76       170.43       869.19       0.0109             NaN         7.94
IVF-TQ-b2-nl223-np11-rf10 (query)                        698.76       295.69       994.45       0.1067          1.1837         7.94
IVF-TQ-b2-nl223-np11-rf20 (query)                        698.76       554.26     1_253.02       0.2082          1.1267         7.94
IVF-TQ-b2-nl223-np14-rf10 (query)                        698.76       318.44     1_017.20       0.1035          1.1885         7.94
IVF-TQ-b2-nl223-np14-rf20 (query)                        698.76       604.46     1_303.22       0.2014          1.1310         7.94
IVF-TQ-b2-nl223-np21-rf10 (query)                        698.76       367.59     1_066.35       0.1030          1.1894         7.94
IVF-TQ-b2-nl223-np21-rf20 (query)                        698.76       664.58     1_363.34       0.2003          1.1318         7.94
IVF-TQ-b2-nl223 (self)                                   698.76     1_338.60     2_037.36       0.1995          1.1335         7.94
IVF-TQ-b2-nl316-np15-rf0 (query)                         981.08       125.20     1_106.28       0.0113             NaN         8.13
IVF-TQ-b2-nl316-np17-rf0 (query)                         981.08       134.41     1_115.49       0.0111             NaN         8.13
IVF-TQ-b2-nl316-np25-rf0 (query)                         981.08       163.40     1_144.48       0.0109             NaN         8.13
IVF-TQ-b2-nl316-np15-rf10 (query)                        981.08       286.70     1_267.78       0.1074          1.1831         8.13
IVF-TQ-b2-nl316-np15-rf20 (query)                        981.08       534.98     1_516.06       0.2091          1.1263         8.13
IVF-TQ-b2-nl316-np17-rf10 (query)                        981.08       295.53     1_276.61       0.1049          1.1866         8.13
IVF-TQ-b2-nl316-np17-rf20 (query)                        981.08       586.66     1_567.74       0.2041          1.1294         8.13
IVF-TQ-b2-nl316-np25-rf10 (query)                        981.08       343.90     1_324.98       0.1030          1.1894         8.13
IVF-TQ-b2-nl316-np25-rf20 (query)                        981.08       634.19     1_615.26       0.2003          1.1318         8.13
IVF-TQ-b2-nl316 (self)                                   981.08     1_245.13     2_226.21       0.1995          1.1335         8.13
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_454.54       147.88     1_602.41       0.0140             NaN        14.07
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_454.54       202.17     1_656.70       0.0132             NaN        14.07
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_454.54       253.59     1_708.13       0.0132             NaN        14.07
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_454.54       315.98     1_770.52       0.1158          1.1694        14.07
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_454.54       549.53     2_004.07       0.2185          1.1185        14.07
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_454.54       385.64     1_840.18       0.1084          1.1766        14.07
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_454.54       664.92     2_119.46       0.2040          1.1250        14.07
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_454.54       467.41     1_921.95       0.1079          1.1773        14.07
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_454.54       752.83     2_207.37       0.2030          1.1256        14.07
IVF-TQ-b4-nl158 (self)                                 1_454.54     1_443.39     2_897.93       0.2033          1.1266        14.07
IVF-TQ-b4-nl223-np11-rf0 (query)                         817.51       168.25       985.76       0.0137             NaN        14.26
IVF-TQ-b4-nl223-np14-rf0 (query)                         817.51       190.69     1_008.20       0.0133             NaN        14.26
IVF-TQ-b4-nl223-np21-rf0 (query)                         817.51       247.92     1_065.43       0.0132             NaN        14.26
IVF-TQ-b4-nl223-np11-rf10 (query)                        817.51       346.84     1_164.35       0.1124          1.1723        14.26
IVF-TQ-b4-nl223-np11-rf20 (query)                        817.51       614.55     1_432.06       0.2117          1.1211        14.26
IVF-TQ-b4-nl223-np14-rf10 (query)                        817.51       380.15     1_197.66       0.1086          1.1765        14.26
IVF-TQ-b4-nl223-np14-rf20 (query)                        817.51       665.06     1_482.57       0.2040          1.1251        14.26
IVF-TQ-b4-nl223-np21-rf10 (query)                        817.51       452.89     1_270.40       0.1079          1.1773        14.26
IVF-TQ-b4-nl223-np21-rf20 (query)                        817.51       745.75     1_563.26       0.2030          1.1256        14.26
IVF-TQ-b4-nl223 (self)                                   817.51     1_462.37     2_279.88       0.2033          1.1266        14.26
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_051.79       172.80     1_224.59       0.0137             NaN        14.56
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_051.79       181.58     1_233.37       0.0134             NaN        14.56
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_051.79       232.64     1_284.43       0.0132             NaN        14.56
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_051.79       339.54     1_391.33       0.1130          1.1713        14.56
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_051.79       592.37     1_644.17       0.2124          1.1205        14.56
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_051.79       381.27     1_433.06       0.1103          1.1744        14.56
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_051.79       610.39     1_662.18       0.2074          1.1232        14.56
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_051.79       421.35     1_473.14       0.1079          1.1773        14.56
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_051.79       722.69     1_774.49       0.2030          1.1256        14.56
IVF-TQ-b4-nl316 (self)                                 1_051.79     1_370.11     2_421.91       0.2033          1.1266        14.56
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
Exhaustive (query)                                        20.77     9_530.28     9_551.04       1.0000          1.0000        97.66
Exhaustive (self)                                         20.77    32_295.37    32_316.14       1.0000          1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              424.15       642.01     1_066.16       0.0120             NaN        13.97
ExhaustiveTQ-b2-rf5 (query)                              424.15       735.66     1_159.82       0.0561          1.1729        13.97
ExhaustiveTQ-b2-rf10 (query)                             424.15       867.67     1_291.83       0.1081          1.1302        13.97
ExhaustiveTQ-b2-rf20 (query)                             424.15     1_252.65     1_676.80       0.2057          1.0911        13.97
ExhaustiveTQ-b2 (self)                                   424.15     4_165.58     4_589.73       0.2055          1.0916        13.97
ExhaustiveTQ-b4-rf0 (query)                              540.15     1_111.86     1_652.02       0.0183             NaN        26.18
ExhaustiveTQ-b4-rf5 (query)                              540.15     1_211.92     1_752.07       0.0633          1.1621        26.18
ExhaustiveTQ-b4-rf10 (query)                             540.15     1_350.75     1_890.90       0.1141          1.1229        26.18
ExhaustiveTQ-b4-rf20 (query)                             540.15     1_728.80     2_268.95       0.2061          1.0883        26.18
ExhaustiveTQ-b4 (self)                                   540.15     5_754.72     6_294.87       0.2069          1.0881        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        2_994.29       222.02     3_216.31       0.0125             NaN        14.98
IVF-TQ-b2-nl158-np12-rf0 (query)                       2_994.29       273.63     3_267.92       0.0119             NaN        14.98
IVF-TQ-b2-nl158-np17-rf0 (query)                       2_994.29       323.16     3_317.45       0.0119             NaN        14.98
IVF-TQ-b2-nl158-np7-rf10 (query)                       2_994.29       408.96     3_403.25       0.1140          1.1257        14.98
IVF-TQ-b2-nl158-np7-rf20 (query)                       2_994.29       663.41     3_657.71       0.2176          1.0868        14.98
IVF-TQ-b2-nl158-np12-rf10 (query)                      2_994.29       471.83     3_466.12       0.1081          1.1302        14.98
IVF-TQ-b2-nl158-np12-rf20 (query)                      2_994.29       756.28     3_750.58       0.2057          1.0911        14.98
IVF-TQ-b2-nl158-np17-rf10 (query)                      2_994.29       527.89     3_522.18       0.1081          1.1302        14.98
IVF-TQ-b2-nl158-np17-rf20 (query)                      2_994.29       832.84     3_827.13       0.2057          1.0911        14.98
IVF-TQ-b2-nl158 (self)                                 2_994.29     1_788.32     4_782.61       0.2055          1.0916        14.98
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_453.07       240.52     1_693.59       0.0123             NaN        15.21
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_453.07       263.94     1_717.01       0.0120             NaN        15.21
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_453.07       317.90     1_770.98       0.0119             NaN        15.21
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_453.07       426.82     1_879.89       0.1112          1.1282        15.21
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_453.07       700.60     2_153.67       0.2117          1.0891        15.21
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_453.07       462.86     1_915.93       0.1086          1.1299        15.21
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_453.07       747.03     2_200.10       0.2066          1.0908        15.21
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_453.07       526.02     1_979.09       0.1081          1.1302        15.21
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_453.07       835.58     2_288.65       0.2057          1.0911        15.21
IVF-TQ-b2-nl223 (self)                                 1_453.07     1_811.17     3_264.24       0.2055          1.0916        15.21
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_802.70       254.77     2_057.47       0.0123             NaN        15.54
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_802.70       265.37     2_068.07       0.0121             NaN        15.54
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_802.70       312.63     2_115.33       0.0119             NaN        15.54
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_802.70       436.78     2_239.47       0.1120          1.1274        15.54
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_802.70       718.63     2_521.32       0.2135          1.0883        15.54
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_802.70       447.93     2_250.63       0.1095          1.1293        15.54
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_802.70       737.66     2_540.36       0.2085          1.0902        15.54
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_802.70       510.30     2_313.00       0.1081          1.1302        15.54
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_802.70       822.54     2_625.24       0.2057          1.0911        15.54
IVF-TQ-b2-nl316 (self)                                 1_802.70     1_854.85     3_657.55       0.2055          1.0916        15.54
IVF-TQ-b4-nl158-np7-rf0 (query)                        3_151.20       306.45     3_457.65       0.0191             NaN        27.50
IVF-TQ-b4-nl158-np12-rf0 (query)                       3_151.20       399.50     3_550.70       0.0183             NaN        27.50
IVF-TQ-b4-nl158-np17-rf0 (query)                       3_151.20       476.59     3_627.79       0.0183             NaN        27.50
IVF-TQ-b4-nl158-np7-rf10 (query)                       3_151.20       507.67     3_658.87       0.1206          1.1186        27.50
IVF-TQ-b4-nl158-np7-rf20 (query)                       3_151.20       771.37     3_922.57       0.2184          1.0844        27.50
IVF-TQ-b4-nl158-np12-rf10 (query)                      3_151.20       610.53     3_761.73       0.1141          1.1229        27.50
IVF-TQ-b4-nl158-np12-rf20 (query)                      3_151.20       910.66     4_061.86       0.2061          1.0883        27.50
IVF-TQ-b4-nl158-np17-rf10 (query)                      3_151.20       712.25     3_863.46       0.1140          1.1229        27.50
IVF-TQ-b4-nl158-np17-rf20 (query)                      3_151.20     1_017.52     4_168.72       0.2061          1.0883        27.50
IVF-TQ-b4-nl158 (self)                                 3_151.20     2_074.47     5_225.67       0.2069          1.0881        27.50
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_557.63       333.03     1_890.66       0.0186             NaN        27.83
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_557.63       374.77     1_932.40       0.0183             NaN        27.83
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_557.63       469.76     2_027.40       0.0183             NaN        27.83
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_557.63       535.39     2_093.02       0.1171          1.1210        27.83
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_557.63       817.24     2_374.87       0.2121          1.0864        27.83
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_557.63       590.11     2_147.74       0.1144          1.1227        27.83
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_557.63       880.68     2_438.31       0.2070          1.0880        27.83
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_557.63       696.03     2_253.67       0.1141          1.1229        27.83
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_557.63     1_011.82     2_569.45       0.2061          1.0883        27.83
IVF-TQ-b4-nl223 (self)                                 1_557.63     2_133.56     3_691.19       0.2069          1.0881        27.83
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_897.59       347.41     2_245.00       0.0187             NaN        28.31
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_897.59       367.40     2_264.99       0.0184             NaN        28.31
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_897.59       449.78     2_347.37       0.0183             NaN        28.31
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_897.59       544.99     2_442.58       0.1179          1.1204        28.31
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_897.59       835.96     2_733.55       0.2141          1.0857        28.31
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_897.59       568.79     2_466.38       0.1154          1.1220        28.31
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_897.59       868.21     2_765.80       0.2091          1.0874        28.31
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_897.59       679.53     2_577.12       0.1141          1.1229        28.31
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_897.59       991.61     2_889.20       0.2061          1.0883        28.31
IVF-TQ-b4-nl316 (self)                                 1_897.59     2_132.12     4_029.71       0.2069          1.0881        28.31
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
Exhaustive (query)                                        30.40    15_488.45    15_518.85       1.0000          1.0000       146.48
Exhaustive (self)                                         30.40    52_242.79    52_273.18       1.0000          1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              867.27       966.81     1_834.08       0.0154             NaN        21.33
ExhaustiveTQ-b2-rf5 (query)                              867.27     1_064.53     1_931.80       0.0627          1.1385        21.33
ExhaustiveTQ-b2-rf10 (query)                             867.27     1_218.52     2_085.79       0.1152          1.1036        21.33
ExhaustiveTQ-b2-rf20 (query)                             867.27     1_647.08     2_514.35       0.2128          1.0710        21.33
ExhaustiveTQ-b2 (self)                                   867.27     5_452.92     6_320.19       0.2134          1.0712        21.33
ExhaustiveTQ-b4-rf0 (query)                            1_022.84     1_735.30     2_758.14       0.0148             NaN        39.64
ExhaustiveTQ-b4-rf5 (query)                            1_022.84     1_815.04     2_837.88       0.0558          1.1453        39.64
ExhaustiveTQ-b4-rf10 (query)                           1_022.84     1_964.05     2_986.89       0.1025          1.1154        39.64
ExhaustiveTQ-b4-rf20 (query)                           1_022.84     2_568.00     3_590.85       0.1923          1.0877        39.64
ExhaustiveTQ-b4 (self)                                 1_022.84     7_937.86     8_960.71       0.1918          1.0881        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        4_851.37       392.19     5_243.56       0.0162             NaN        22.62
IVF-TQ-b2-nl158-np12-rf0 (query)                       4_851.37       496.87     5_348.24       0.0154             NaN        22.62
IVF-TQ-b2-nl158-np17-rf0 (query)                       4_851.37       531.95     5_383.32       0.0154             NaN        22.62
IVF-TQ-b2-nl158-np7-rf10 (query)                       4_851.37       610.76     5_462.13       0.1215          1.1004        22.62
IVF-TQ-b2-nl158-np7-rf20 (query)                       4_851.37       883.05     5_734.42       0.2243          1.0677        22.62
IVF-TQ-b2-nl158-np12-rf10 (query)                      4_851.37       693.03     5_544.40       0.1152          1.1036        22.62
IVF-TQ-b2-nl158-np12-rf20 (query)                      4_851.37     1_006.34     5_857.71       0.2128          1.0710        22.62
IVF-TQ-b2-nl158-np17-rf10 (query)                      4_851.37       780.96     5_632.33       0.1152          1.1036        22.62
IVF-TQ-b2-nl158-np17-rf20 (query)                      4_851.37     1_124.33     5_975.70       0.2128          1.0710        22.62
IVF-TQ-b2-nl158 (self)                                 4_851.37     2_570.24     7_421.61       0.2134          1.0712        22.62
IVF-TQ-b2-nl223-np11-rf0 (query)                       2_483.83       424.66     2_908.49       0.0160             NaN        23.00
IVF-TQ-b2-nl223-np14-rf0 (query)                       2_483.83       450.74     2_934.57       0.0155             NaN        23.00
IVF-TQ-b2-nl223-np21-rf0 (query)                       2_483.83       527.63     3_011.46       0.0154             NaN        23.00
IVF-TQ-b2-nl223-np11-rf10 (query)                      2_483.83       642.02     3_125.85       0.1209          1.1003        23.00
IVF-TQ-b2-nl223-np11-rf20 (query)                      2_483.83       958.48     3_442.31       0.2233          1.0678        23.00
IVF-TQ-b2-nl223-np14-rf10 (query)                      2_483.83       675.25     3_159.08       0.1162          1.1029        23.00
IVF-TQ-b2-nl223-np14-rf20 (query)                      2_483.83       992.39     3_476.21       0.2147          1.0703        23.00
IVF-TQ-b2-nl223-np21-rf10 (query)                      2_483.83       745.36     3_229.19       0.1152          1.1036        23.00
IVF-TQ-b2-nl223-np21-rf20 (query)                      2_483.83     1_095.32     3_579.15       0.2128          1.0710        23.00
IVF-TQ-b2-nl223 (self)                                 2_483.83     2_525.32     5_009.14       0.2134          1.0712        23.00
IVF-TQ-b2-nl316-np15-rf0 (query)                       2_978.84       444.54     3_423.38       0.0159             NaN        23.53
IVF-TQ-b2-nl316-np17-rf0 (query)                       2_978.84       457.03     3_435.86       0.0156             NaN        23.53
IVF-TQ-b2-nl316-np25-rf0 (query)                       2_978.84       512.26     3_491.10       0.0154             NaN        23.53
IVF-TQ-b2-nl316-np15-rf10 (query)                      2_978.84       661.68     3_640.52       0.1204          1.1007        23.53
IVF-TQ-b2-nl316-np15-rf20 (query)                      2_978.84       966.21     3_945.05       0.2221          1.0682        23.53
IVF-TQ-b2-nl316-np17-rf10 (query)                      2_978.84       672.54     3_651.38       0.1177          1.1021        23.53
IVF-TQ-b2-nl316-np17-rf20 (query)                      2_978.84       990.95     3_969.79       0.2175          1.0695        23.53
IVF-TQ-b2-nl316-np25-rf10 (query)                      2_978.84       766.42     3_745.26       0.1152          1.1036        23.53
IVF-TQ-b2-nl316-np25-rf20 (query)                      2_978.84     1_094.86     4_073.70       0.2128          1.0710        23.53
IVF-TQ-b2-nl316 (self)                                 2_978.84     2_554.77     5_533.61       0.2134          1.0712        23.53
IVF-TQ-b4-nl158-np7-rf0 (query)                        5_009.12       519.88     5_529.00       0.0155             NaN        41.38
IVF-TQ-b4-nl158-np12-rf0 (query)                       5_009.12       649.88     5_659.00       0.0148             NaN        41.38
IVF-TQ-b4-nl158-np17-rf0 (query)                       5_009.12       769.24     5_778.36       0.0148             NaN        41.38
IVF-TQ-b4-nl158-np7-rf10 (query)                       5_009.12       751.71     5_760.83       0.1084          1.1125        41.38
IVF-TQ-b4-nl158-np7-rf20 (query)                       5_009.12     1_032.41     6_041.52       0.2038          1.0851        41.38
IVF-TQ-b4-nl158-np12-rf10 (query)                      5_009.12       893.72     5_902.83       0.1025          1.1154        41.38
IVF-TQ-b4-nl158-np12-rf20 (query)                      5_009.12     1_209.17     6_218.28       0.1923          1.0877        41.38
IVF-TQ-b4-nl158-np17-rf10 (query)                      5_009.12     1_036.92     6_046.04       0.1025          1.1154        41.38
IVF-TQ-b4-nl158-np17-rf20 (query)                      5_009.12     1_396.22     6_405.34       0.1923          1.0877        41.38
IVF-TQ-b4-nl158 (self)                                 5_009.12     3_004.51     8_013.62       0.1918          1.0881        41.38
IVF-TQ-b4-nl223-np11-rf0 (query)                       2_642.01       560.28     3_202.29       0.0156             NaN        41.96
IVF-TQ-b4-nl223-np14-rf0 (query)                       2_642.01       610.57     3_252.58       0.0150             NaN        41.96
IVF-TQ-b4-nl223-np21-rf0 (query)                       2_642.01       733.34     3_375.35       0.0148             NaN        41.96
IVF-TQ-b4-nl223-np11-rf10 (query)                      2_642.01       797.39     3_439.39       0.1075          1.1123        41.96
IVF-TQ-b4-nl223-np11-rf20 (query)                      2_642.01     1_113.03     3_755.04       0.2025          1.0849        41.96
IVF-TQ-b4-nl223-np14-rf10 (query)                      2_642.01       842.46     3_484.47       0.1035          1.1147        41.96
IVF-TQ-b4-nl223-np14-rf20 (query)                      2_642.01     1_167.79     3_809.80       0.1941          1.0871        41.96
IVF-TQ-b4-nl223-np21-rf10 (query)                      2_642.01       973.41     3_615.42       0.1025          1.1154        41.96
IVF-TQ-b4-nl223-np21-rf20 (query)                      2_642.01     1_320.76     3_962.76       0.1923          1.0877        41.96
IVF-TQ-b4-nl223 (self)                                 2_642.01     2_867.59     5_509.60       0.1918          1.0881        41.96
IVF-TQ-b4-nl316-np15-rf0 (query)                       3_124.90       584.35     3_709.25       0.0155             NaN        42.73
IVF-TQ-b4-nl316-np17-rf0 (query)                       3_124.90       606.81     3_731.71       0.0152             NaN        42.73
IVF-TQ-b4-nl316-np25-rf0 (query)                       3_124.90       712.57     3_837.47       0.0148             NaN        42.73
IVF-TQ-b4-nl316-np15-rf10 (query)                      3_124.90       808.42     3_933.33       0.1073          1.1126        42.73
IVF-TQ-b4-nl316-np15-rf20 (query)                      3_124.90     1_118.65     4_243.56       0.2016          1.0851        42.73
IVF-TQ-b4-nl316-np17-rf10 (query)                      3_124.90       833.72     3_958.62       0.1049          1.1138        42.73
IVF-TQ-b4-nl316-np17-rf20 (query)                      3_124.90     1_164.17     4_289.07       0.1968          1.0863        42.73
IVF-TQ-b4-nl316-np25-rf10 (query)                      3_124.90       941.12     4_066.03       0.1025          1.1154        42.73
IVF-TQ-b4-nl316-np25-rf20 (query)                      3_124.90     1_286.73     4_411.63       0.1923          1.0877        42.73
IVF-TQ-b4-nl316 (self)                                 3_124.90     2_937.34     6_062.24       0.1918          1.0881        42.73
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
Exhaustive (query)                                        10.00     4_161.75     4_171.75       1.0000          1.0000        48.83
Exhaustive (self)                                         10.00    14_582.17    14_592.17       1.0000          1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              151.28       370.26       521.54       0.0662             NaN         7.12
ExhaustiveTQ-b2-rf5 (query)                              151.28       445.33       596.60       0.1862          1.3185         7.12
ExhaustiveTQ-b2-rf10 (query)                             151.28       565.63       716.91       0.2699          1.2136         7.12
ExhaustiveTQ-b2-rf20 (query)                             151.28       922.78     1_074.06       0.4056          1.1279         7.12
ExhaustiveTQ-b2 (self)                                   151.28     3_061.54     3_212.82       0.4070          1.1561         7.12
ExhaustiveTQ-b4-rf0 (query)                              232.79       577.55       810.34       0.0871             NaN        13.22
ExhaustiveTQ-b4-rf5 (query)                              232.79       661.84       894.63       0.2059          1.2890        13.22
ExhaustiveTQ-b4-rf10 (query)                             232.79       783.83     1_016.62       0.2865          1.1965        13.22
ExhaustiveTQ-b4-rf20 (query)                             232.79     1_139.84     1_372.63       0.4170          1.1210        13.22
ExhaustiveTQ-b4 (self)                                   232.79     3_792.04     4_024.82       0.4165          1.1485        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_275.28       101.86     1_377.14       0.0664             NaN         7.81
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_275.28       114.77     1_390.05       0.0662             NaN         7.81
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_275.28       129.48     1_404.75       0.0662             NaN         7.81
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_275.28       295.10     1_570.38       0.2699          1.2136         7.81
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_275.28       602.22     1_877.50       0.4055          1.1279         7.81
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_275.28       312.29     1_587.57       0.2699          1.2136         7.81
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_275.28       647.53     1_922.81       0.4056          1.1279         7.81
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_275.28       338.16     1_613.44       0.2699          1.2136         7.81
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_275.28       702.31     1_977.59       0.4056          1.1279         7.81
IVF-TQ-b2-nl158 (self)                                 1_275.28     1_171.44     2_446.72       0.4070          1.1561         7.81
IVF-TQ-b2-nl223-np11-rf0 (query)                         837.37       110.59       947.96       0.0664             NaN         7.93
IVF-TQ-b2-nl223-np14-rf0 (query)                         837.37       117.72       955.09       0.0662             NaN         7.93
IVF-TQ-b2-nl223-np21-rf0 (query)                         837.37       142.56       979.93       0.0662             NaN         7.93
IVF-TQ-b2-nl223-np11-rf10 (query)                        837.37       283.07     1_120.44       0.2711          1.2125         7.93
IVF-TQ-b2-nl223-np11-rf20 (query)                        837.37       545.49     1_382.86       0.4078          1.1269         7.93
IVF-TQ-b2-nl223-np14-rf10 (query)                        837.37       303.42     1_140.79       0.2699          1.2136         7.93
IVF-TQ-b2-nl223-np14-rf20 (query)                        837.37       563.90     1_401.28       0.4056          1.1279         7.93
IVF-TQ-b2-nl223-np21-rf10 (query)                        837.37       328.67     1_166.04       0.2699          1.2136         7.93
IVF-TQ-b2-nl223-np21-rf20 (query)                        837.37       618.82     1_456.20       0.4056          1.1279         7.93
IVF-TQ-b2-nl223 (self)                                   837.37     1_182.57     2_019.94       0.4070          1.1561         7.93
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_064.96       124.66     1_189.61       0.0663             NaN         8.11
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_064.96       126.83     1_191.79       0.0663             NaN         8.11
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_064.96       140.28     1_205.23       0.0662             NaN         8.11
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_064.96       281.10     1_346.05       0.2707          1.2128         8.11
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_064.96       517.29     1_582.25       0.4072          1.1271         8.11
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_064.96       288.91     1_353.86       0.2702          1.2133         8.11
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_064.96       529.54     1_594.49       0.4061          1.1277         8.11
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_064.96       310.50     1_375.46       0.2699          1.2136         8.11
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_064.96       563.69     1_628.65       0.4056          1.1279         8.11
IVF-TQ-b2-nl316 (self)                                 1_064.96     1_150.01     2_214.97       0.4070          1.1561         8.11
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_332.32       139.15     1_471.47       0.0872             NaN        14.06
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_332.32       159.86     1_492.18       0.0871             NaN        14.06
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_332.32       183.79     1_516.12       0.0871             NaN        14.06
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_332.32       342.32     1_674.64       0.2865          1.1965        14.06
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_332.32       652.81     1_985.13       0.4169          1.1210        14.06
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_332.32       365.58     1_697.90       0.2865          1.1965        14.06
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_332.32       701.32     2_033.65       0.4170          1.1210        14.06
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_332.32       399.84     1_732.16       0.2865          1.1965        14.06
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_332.32       766.87     2_099.20       0.4170          1.1210        14.06
IVF-TQ-b4-nl158 (self)                                 1_332.32     1_266.55     2_598.87       0.4165          1.1485        14.06
IVF-TQ-b4-nl223-np11-rf0 (query)                         921.10       151.17     1_072.27       0.0873             NaN        14.24
IVF-TQ-b4-nl223-np14-rf0 (query)                         921.10       161.73     1_082.82       0.0871             NaN        14.24
IVF-TQ-b4-nl223-np21-rf0 (query)                         921.10       202.01     1_123.11       0.0871             NaN        14.24
IVF-TQ-b4-nl223-np11-rf10 (query)                        921.10       328.38     1_249.48       0.2876          1.1957        14.24
IVF-TQ-b4-nl223-np11-rf20 (query)                        921.10       591.22     1_512.32       0.4188          1.1202        14.24
IVF-TQ-b4-nl223-np14-rf10 (query)                        921.10       341.33     1_262.43       0.2866          1.1964        14.24
IVF-TQ-b4-nl223-np14-rf20 (query)                        921.10       615.41     1_536.51       0.4171          1.1210        14.24
IVF-TQ-b4-nl223-np21-rf10 (query)                        921.10       394.77     1_315.87       0.2865          1.1965        14.24
IVF-TQ-b4-nl223-np21-rf20 (query)                        921.10       687.86     1_608.96       0.4170          1.1210        14.24
IVF-TQ-b4-nl223 (self)                                   921.10     1_289.90     2_210.99       0.4165          1.1485        14.24
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_146.91       160.14     1_307.05       0.0872             NaN        14.51
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_146.91       164.78     1_311.69       0.0872             NaN        14.51
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_146.91       196.38     1_343.29       0.0871             NaN        14.51
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_146.91       331.77     1_478.69       0.2873          1.1959        14.51
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_146.91       572.72     1_719.63       0.4183          1.1204        14.51
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_146.91       340.29     1_487.20       0.2868          1.1962        14.51
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_146.91       586.85     1_733.76       0.4174          1.1208        14.51
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_146.91       375.26     1_522.17       0.2865          1.1965        14.51
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_146.91       633.50     1_780.41       0.4170          1.1210        14.51
IVF-TQ-b4-nl316 (self)                                 1_146.91     1_251.20     2_398.11       0.4165          1.1485        14.51
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
Exhaustive (query)                                        20.07     9_605.41     9_625.48       1.0000          1.0000        97.66
Exhaustive (self)                                         20.07    32_846.16    32_866.23       1.0000          1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              459.38       675.16     1_134.54       0.0709             NaN        13.97
ExhaustiveTQ-b2-rf5 (query)                              459.38       812.65     1_272.03       0.1815          1.2341        13.97
ExhaustiveTQ-b2-rf10 (query)                             459.38       909.33     1_368.71       0.2475          1.1648        13.97
ExhaustiveTQ-b2-rf20 (query)                             459.38     1_324.19     1_783.57       0.3619          1.1046        13.97
ExhaustiveTQ-b2 (self)                                   459.38     4_373.11     4_832.49       0.3623          1.1225        13.97
ExhaustiveTQ-b4-rf0 (query)                              561.56     1_132.05     1_693.61       0.0862             NaN        26.18
ExhaustiveTQ-b4-rf5 (query)                              561.56     1_355.21     1_916.78       0.1892          1.2262        26.18
ExhaustiveTQ-b4-rf10 (query)                             561.56     1_434.11     1_995.68       0.2498          1.1620        26.18
ExhaustiveTQ-b4-rf20 (query)                             561.56     1_888.53     2_450.09       0.3584          1.1058        26.18
ExhaustiveTQ-b4 (self)                                   561.56     5_833.64     6_395.20       0.3580          1.1245        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        2_708.97       208.93     2_917.90       0.0709             NaN        14.98
IVF-TQ-b2-nl158-np12-rf0 (query)                       2_708.97       222.04     2_931.01       0.0709             NaN        14.98
IVF-TQ-b2-nl158-np17-rf0 (query)                       2_708.97       236.83     2_945.80       0.0709             NaN        14.98
IVF-TQ-b2-nl158-np7-rf10 (query)                       2_708.97       426.22     3_135.20       0.2475          1.1648        14.98
IVF-TQ-b2-nl158-np7-rf20 (query)                       2_708.97       809.15     3_518.12       0.3619          1.1046        14.98
IVF-TQ-b2-nl158-np12-rf10 (query)                      2_708.97       438.71     3_147.68       0.2475          1.1648        14.98
IVF-TQ-b2-nl158-np12-rf20 (query)                      2_708.97       777.86     3_486.84       0.3619          1.1046        14.98
IVF-TQ-b2-nl158-np17-rf10 (query)                      2_708.97       460.62     3_169.59       0.2475          1.1648        14.98
IVF-TQ-b2-nl158-np17-rf20 (query)                      2_708.97       812.86     3_521.83       0.3619          1.1046        14.98
IVF-TQ-b2-nl158 (self)                                 2_708.97     1_664.88     4_373.86       0.3623          1.1225        14.98
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_597.05       227.17     1_824.22       0.0709             NaN        15.19
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_597.05       236.92     1_833.97       0.0709             NaN        15.19
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_597.05       261.13     1_858.18       0.0709             NaN        15.19
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_597.05       427.75     2_024.80       0.2475          1.1648        15.19
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_597.05       714.50     2_311.55       0.3619          1.1046        15.19
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_597.05       438.93     2_035.98       0.2475          1.1648        15.19
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_597.05       729.14     2_326.19       0.3619          1.1046        15.19
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_597.05       466.22     2_063.27       0.2475          1.1648        15.19
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_597.05       776.19     2_373.24       0.3619          1.1046        15.19
IVF-TQ-b2-nl223 (self)                                 1_597.05     1_664.51     3_261.56       0.3623          1.1225        15.19
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_990.04       243.46     2_233.50       0.0709             NaN        15.55
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_990.04       245.99     2_236.03       0.0709             NaN        15.55
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_990.04       266.88     2_256.92       0.0709             NaN        15.55
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_990.04       467.14     2_457.18       0.2475          1.1648        15.55
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_990.04       700.62     2_690.66       0.3619          1.1046        15.55
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_990.04       439.27     2_429.31       0.2475          1.1648        15.55
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_990.04       720.67     2_710.71       0.3619          1.1046        15.55
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_990.04       463.70     2_453.74       0.2475          1.1648        15.55
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_990.04       749.85     2_739.89       0.3619          1.1046        15.55
IVF-TQ-b2-nl316 (self)                                 1_990.04     1_713.47     3_703.51       0.3623          1.1225        15.55
IVF-TQ-b4-nl158-np7-rf0 (query)                        2_810.17       281.99     3_092.16       0.0862             NaN        27.51
IVF-TQ-b4-nl158-np12-rf0 (query)                       2_810.17       304.72     3_114.89       0.0862             NaN        27.51
IVF-TQ-b4-nl158-np17-rf0 (query)                       2_810.17       332.27     3_142.43       0.0862             NaN        27.51
IVF-TQ-b4-nl158-np7-rf10 (query)                       2_810.17       515.99     3_326.16       0.2498          1.1620        27.51
IVF-TQ-b4-nl158-np7-rf20 (query)                       2_810.17       864.62     3_674.79       0.3584          1.1058        27.51
IVF-TQ-b4-nl158-np12-rf10 (query)                      2_810.17       535.74     3_345.90       0.2498          1.1620        27.51
IVF-TQ-b4-nl158-np12-rf20 (query)                      2_810.17       883.83     3_694.00       0.3584          1.1058        27.51
IVF-TQ-b4-nl158-np17-rf10 (query)                      2_810.17       567.94     3_378.10       0.2498          1.1620        27.51
IVF-TQ-b4-nl158-np17-rf20 (query)                      2_810.17       936.95     3_747.12       0.3584          1.1058        27.51
IVF-TQ-b4-nl158 (self)                                 2_810.17     1_855.50     4_665.67       0.3580          1.1245        27.51
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_729.06       306.48     2_035.54       0.0862             NaN        27.81
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_729.06       320.60     2_049.66       0.0862             NaN        27.81
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_729.06       366.52     2_095.58       0.0862             NaN        27.81
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_729.06       519.49     2_248.55       0.2498          1.1620        27.81
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_729.06       808.47     2_537.53       0.3584          1.1058        27.81
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_729.06       533.63     2_262.69       0.2498          1.1620        27.81
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_729.06       830.32     2_559.38       0.3584          1.1058        27.81
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_729.06       589.49     2_318.55       0.2498          1.1620        27.81
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_729.06       898.08     2_627.14       0.3584          1.1058        27.81
IVF-TQ-b4-nl223 (self)                                 1_729.06     1_868.77     3_597.83       0.3580          1.1245        27.81
IVF-TQ-b4-nl316-np15-rf0 (query)                       2_106.73       323.69     2_430.42       0.0862             NaN        28.33
IVF-TQ-b4-nl316-np17-rf0 (query)                       2_106.73       339.30     2_446.03       0.0862             NaN        28.33
IVF-TQ-b4-nl316-np25-rf0 (query)                       2_106.73       367.84     2_474.57       0.0862             NaN        28.33
IVF-TQ-b4-nl316-np15-rf10 (query)                      2_106.73       526.74     2_633.47       0.2498          1.1620        28.33
IVF-TQ-b4-nl316-np15-rf20 (query)                      2_106.73       807.60     2_914.33       0.3584          1.1058        28.33
IVF-TQ-b4-nl316-np17-rf10 (query)                      2_106.73       543.75     2_650.48       0.2498          1.1620        28.33
IVF-TQ-b4-nl316-np17-rf20 (query)                      2_106.73       817.74     2_924.47       0.3584          1.1058        28.33
IVF-TQ-b4-nl316-np25-rf10 (query)                      2_106.73       577.63     2_684.36       0.2498          1.1620        28.33
IVF-TQ-b4-nl316-np25-rf20 (query)                      2_106.73       866.90     2_973.63       0.3584          1.1058        28.33
IVF-TQ-b4-nl316 (self)                                 2_106.73     1_898.45     4_005.18       0.3580          1.1245        28.33
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
Exhaustive (query)                                        32.08    15_760.15    15_792.23       1.0000          1.0000       146.48
Exhaustive (self)                                         32.08    52_305.41    52_337.49       1.0000          1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              867.29       975.48     1_842.77       0.0719             NaN        21.33
ExhaustiveTQ-b2-rf5 (query)                              867.29     1_071.96     1_939.25       0.1764          1.1855        21.33
ExhaustiveTQ-b2-rf10 (query)                             867.29     1_224.54     2_091.83       0.2312          1.1365        21.33
ExhaustiveTQ-b2-rf20 (query)                             867.29     1_644.23     2_511.52       0.3300          1.0920        21.33
ExhaustiveTQ-b2 (self)                                   867.29     5_449.01     6_316.30       0.3296          1.1027        21.33
ExhaustiveTQ-b4-rf0 (query)                            1_028.03     1_757.77     2_785.80       0.0844             NaN        39.64
ExhaustiveTQ-b4-rf5 (query)                            1_028.03     1_809.23     2_837.26       0.1812          1.1813        39.64
ExhaustiveTQ-b4-rf10 (query)                           1_028.03     2_007.75     3_035.77       0.2330          1.1352        39.64
ExhaustiveTQ-b4-rf20 (query)                           1_028.03     2_404.68     3_432.70       0.3263          1.0942        39.64
ExhaustiveTQ-b4 (self)                                 1_028.03     8_024.08     9_052.11       0.3287          1.1030        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        4_276.24       377.49     4_653.73       0.0719             NaN        22.63
IVF-TQ-b2-nl158-np12-rf0 (query)                       4_276.24       396.88     4_673.12       0.0719             NaN        22.63
IVF-TQ-b2-nl158-np17-rf0 (query)                       4_276.24       415.60     4_691.84       0.0719             NaN        22.63
IVF-TQ-b2-nl158-np7-rf10 (query)                       4_276.24       631.04     4_907.28       0.2312          1.1365        22.63
IVF-TQ-b2-nl158-np7-rf20 (query)                       4_276.24       996.84     5_273.08       0.3300          1.0920        22.63
IVF-TQ-b2-nl158-np12-rf10 (query)                      4_276.24       653.28     4_929.52       0.2313          1.1365        22.63
IVF-TQ-b2-nl158-np12-rf20 (query)                      4_276.24     1_033.41     5_309.65       0.3300          1.0920        22.63
IVF-TQ-b2-nl158-np17-rf10 (query)                      4_276.24       675.40     4_951.64       0.2313          1.1365        22.63
IVF-TQ-b2-nl158-np17-rf20 (query)                      4_276.24     1_063.08     5_339.32       0.3300          1.0920        22.63
IVF-TQ-b2-nl158 (self)                                 4_276.24     2_284.00     6_560.24       0.3296          1.1027        22.63
IVF-TQ-b2-nl223-np11-rf0 (query)                       2_362.50       401.19     2_763.69       0.0719             NaN        22.99
IVF-TQ-b2-nl223-np14-rf0 (query)                       2_362.50       412.78     2_775.28       0.0719             NaN        22.99
IVF-TQ-b2-nl223-np21-rf0 (query)                       2_362.50       449.29     2_811.79       0.0719             NaN        22.99
IVF-TQ-b2-nl223-np11-rf10 (query)                      2_362.50       625.60     2_988.10       0.2312          1.1365        22.99
IVF-TQ-b2-nl223-np11-rf20 (query)                      2_362.50       937.94     3_300.44       0.3300          1.0920        22.99
IVF-TQ-b2-nl223-np14-rf10 (query)                      2_362.50       684.28     3_046.78       0.2313          1.1365        22.99
IVF-TQ-b2-nl223-np14-rf20 (query)                      2_362.50       961.37     3_323.87       0.3300          1.0920        22.99
IVF-TQ-b2-nl223-np21-rf10 (query)                      2_362.50       678.10     3_040.61       0.2313          1.1365        22.99
IVF-TQ-b2-nl223-np21-rf20 (query)                      2_362.50     1_006.44     3_368.94       0.3300          1.0920        22.99
IVF-TQ-b2-nl223 (self)                                 2_362.50     2_336.16     4_698.66       0.3296          1.1027        22.99
IVF-TQ-b2-nl316-np15-rf0 (query)                       3_065.82       428.54     3_494.35       0.0719             NaN        23.51
IVF-TQ-b2-nl316-np17-rf0 (query)                       3_065.82       435.23     3_501.04       0.0719             NaN        23.51
IVF-TQ-b2-nl316-np25-rf0 (query)                       3_065.82       461.18     3_526.99       0.0719             NaN        23.51
IVF-TQ-b2-nl316-np15-rf10 (query)                      3_065.82       641.97     3_707.79       0.2313          1.1365        23.51
IVF-TQ-b2-nl316-np15-rf20 (query)                      3_065.82       934.61     4_000.43       0.3300          1.0920        23.51
IVF-TQ-b2-nl316-np17-rf10 (query)                      3_065.82       647.36     3_713.17       0.2313          1.1365        23.51
IVF-TQ-b2-nl316-np17-rf20 (query)                      3_065.82       952.17     4_017.99       0.3300          1.0920        23.51
IVF-TQ-b2-nl316-np25-rf10 (query)                      3_065.82       681.81     3_747.63       0.2313          1.1365        23.51
IVF-TQ-b2-nl316-np25-rf20 (query)                      3_065.82       992.39     4_058.20       0.3300          1.0920        23.51
IVF-TQ-b2-nl316 (self)                                 3_065.82     2_384.52     5_450.34       0.3296          1.1027        23.51
IVF-TQ-b4-nl158-np7-rf0 (query)                        4_404.80       490.39     4_895.19       0.0844             NaN        41.40
IVF-TQ-b4-nl158-np12-rf0 (query)                       4_404.80       526.76     4_931.56       0.0844             NaN        41.40
IVF-TQ-b4-nl158-np17-rf0 (query)                       4_404.80       559.66     4_964.46       0.0844             NaN        41.40
IVF-TQ-b4-nl158-np7-rf10 (query)                       4_404.80       759.49     5_164.29       0.2329          1.1352        41.40
IVF-TQ-b4-nl158-np7-rf20 (query)                       4_404.80     1_132.45     5_537.25       0.3263          1.0942        41.40
IVF-TQ-b4-nl158-np12-rf10 (query)                      4_404.80       797.11     5_201.91       0.2330          1.1352        41.40
IVF-TQ-b4-nl158-np12-rf20 (query)                      4_404.80     1_185.03     5_589.83       0.3263          1.0942        41.40
IVF-TQ-b4-nl158-np17-rf10 (query)                      4_404.80       835.19     5_239.98       0.2329          1.1352        41.40
IVF-TQ-b4-nl158-np17-rf20 (query)                      4_404.80     1_236.31     5_641.11       0.3263          1.0942        41.40
IVF-TQ-b4-nl158 (self)                                 4_404.80     2_563.53     6_968.32       0.3287          1.1030        41.40
IVF-TQ-b4-nl223-np11-rf0 (query)                       2_511.73       521.65     3_033.37       0.0844             NaN        41.92
IVF-TQ-b4-nl223-np14-rf0 (query)                       2_511.73       543.05     3_054.77       0.0844             NaN        41.92
IVF-TQ-b4-nl223-np21-rf0 (query)                       2_511.73       598.23     3_109.96       0.0844             NaN        41.92
IVF-TQ-b4-nl223-np11-rf10 (query)                      2_511.73       760.95     3_272.68       0.2330          1.1352        41.92
IVF-TQ-b4-nl223-np11-rf20 (query)                      2_511.73     1_089.84     3_601.57       0.3263          1.0942        41.92
IVF-TQ-b4-nl223-np14-rf10 (query)                      2_511.73       784.32     3_296.05       0.2329          1.1352        41.92
IVF-TQ-b4-nl223-np14-rf20 (query)                      2_511.73     1_103.48     3_615.20       0.3264          1.0942        41.92
IVF-TQ-b4-nl223-np21-rf10 (query)                      2_511.73       848.51     3_360.24       0.2329          1.1352        41.92
IVF-TQ-b4-nl223-np21-rf20 (query)                      2_511.73     1_182.19     3_693.91       0.3264          1.0942        41.92
IVF-TQ-b4-nl223 (self)                                 2_511.73     2_638.70     5_150.43       0.3287          1.1030        41.92
IVF-TQ-b4-nl316-np15-rf0 (query)                       3_232.44       553.40     3_785.84       0.0844             NaN        42.69
IVF-TQ-b4-nl316-np17-rf0 (query)                       3_232.44       564.56     3_797.01       0.0844             NaN        42.69
IVF-TQ-b4-nl316-np25-rf0 (query)                       3_232.44       617.78     3_850.22       0.0844             NaN        42.69
IVF-TQ-b4-nl316-np15-rf10 (query)                      3_232.44       779.46     4_011.91       0.2330          1.1352        42.69
IVF-TQ-b4-nl316-np15-rf20 (query)                      3_232.44     1_078.83     4_311.27       0.3264          1.0942        42.69
IVF-TQ-b4-nl316-np17-rf10 (query)                      3_232.44       791.87     4_024.32       0.2330          1.1352        42.69
IVF-TQ-b4-nl316-np17-rf20 (query)                      3_232.44     1_098.50     4_330.95       0.3263          1.0942        42.69
IVF-TQ-b4-nl316-np25-rf10 (query)                      3_232.44       847.53     4_079.98       0.2329          1.1352        42.69
IVF-TQ-b4-nl316-np25-rf20 (query)                      3_232.44     1_166.57     4_399.01       0.3263          1.0942        42.69
IVF-TQ-b4-nl316 (self)                                 3_232.44     2_686.13     5_918.57       0.3287          1.1030        42.69
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

#### Quantisation (stress) data

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D - TurboQuant + IVF
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         9.76     4_141.64     4_151.40       1.0000          1.0000        48.83
Exhaustive (self)                                          9.76    14_200.06    14_209.82       1.0000          1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              151.23       363.00       514.23       0.7919             NaN         7.12
ExhaustiveTQ-b2-rf5 (query)                              151.23       442.53       593.77       0.9995          1.0000         7.12
ExhaustiveTQ-b2-rf10 (query)                             151.23       566.55       717.78       1.0000          1.0000         7.12
ExhaustiveTQ-b2-rf20 (query)                             151.23       936.89     1_088.12       1.0000          1.0000         7.12
ExhaustiveTQ-b2 (self)                                   151.23     3_116.96     3_268.19       1.0000          1.0000         7.12
ExhaustiveTQ-b4-rf0 (query)                              233.01       573.84       806.85       0.8727             NaN        13.22
ExhaustiveTQ-b4-rf5 (query)                              233.01       652.69       885.71       1.0000          1.0000        13.22
ExhaustiveTQ-b4-rf10 (query)                             233.01       779.35     1_012.37       1.0000          1.0000        13.22
ExhaustiveTQ-b4-rf20 (query)                             233.01     1_157.17     1_390.18       1.0000          1.0000        13.22
ExhaustiveTQ-b4 (self)                                   233.01     3_821.68     4_054.69       1.0000          1.0000        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_401.89       126.95     1_528.84       0.7916             NaN         7.78
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_401.89       170.33     1_572.22       0.7918             NaN         7.78
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_401.89       206.95     1_608.84       0.7919             NaN         7.78
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_401.89       324.45     1_726.34       0.9982          1.0004         7.78
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_401.89       597.83     1_999.72       0.9982          1.0004         7.78
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_401.89       385.22     1_787.12       1.0000          1.0000         7.78
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_401.89       699.21     2_101.11       1.0000          1.0000         7.78
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_401.89       434.91     1_836.80       1.0000          1.0000         7.78
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_401.89       763.94     2_165.83       1.0000          1.0000         7.78
IVF-TQ-b2-nl158 (self)                                 1_401.89     1_503.08     2_904.97       1.0000          1.0000         7.78
IVF-TQ-b2-nl223-np11-rf0 (query)                         699.60       127.93       827.53       0.7919             NaN         7.93
IVF-TQ-b2-nl223-np14-rf0 (query)                         699.60       144.98       844.58       0.7919             NaN         7.93
IVF-TQ-b2-nl223-np21-rf0 (query)                         699.60       181.76       881.36       0.7919             NaN         7.93
IVF-TQ-b2-nl223-np11-rf10 (query)                        699.60       311.66     1_011.26       0.9995          1.0001         7.93
IVF-TQ-b2-nl223-np11-rf20 (query)                        699.60       571.17     1_270.77       0.9995          1.0001         7.93
IVF-TQ-b2-nl223-np14-rf10 (query)                        699.60       335.77     1_035.37       0.9999          1.0000         7.93
IVF-TQ-b2-nl223-np14-rf20 (query)                        699.60       613.92     1_313.52       0.9999          1.0000         7.93
IVF-TQ-b2-nl223-np21-rf10 (query)                        699.60       388.69     1_088.29       1.0000          1.0000         7.93
IVF-TQ-b2-nl223-np21-rf20 (query)                        699.60       696.96     1_396.56       1.0000          1.0000         7.93
IVF-TQ-b2-nl223 (self)                                   699.60     1_350.65     2_050.24       1.0000          1.0000         7.93
IVF-TQ-b2-nl316-np15-rf0 (query)                         911.46       132.49     1_043.95       0.7919             NaN         8.12
IVF-TQ-b2-nl316-np17-rf0 (query)                         911.46       141.29     1_052.74       0.7918             NaN         8.12
IVF-TQ-b2-nl316-np25-rf0 (query)                         911.46       172.55     1_084.01       0.7919             NaN         8.12
IVF-TQ-b2-nl316-np15-rf10 (query)                        911.46       306.19     1_217.65       0.9998          1.0000         8.12
IVF-TQ-b2-nl316-np15-rf20 (query)                        911.46       559.04     1_470.50       0.9998          1.0000         8.12
IVF-TQ-b2-nl316-np17-rf10 (query)                        911.46       317.05     1_228.51       0.9999          1.0000         8.12
IVF-TQ-b2-nl316-np17-rf20 (query)                        911.46       579.66     1_491.12       0.9999          1.0000         8.12
IVF-TQ-b2-nl316-np25-rf10 (query)                        911.46       361.18     1_272.63       1.0000          1.0000         8.12
IVF-TQ-b2-nl316-np25-rf20 (query)                        911.46       647.63     1_559.09       1.0000          1.0000         8.12
IVF-TQ-b2-nl316 (self)                                   911.46     1_289.73     2_201.19       1.0000          1.0000         8.12
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_464.13       182.58     1_646.71       0.8721             NaN        14.02
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_464.13       251.66     1_715.79       0.8727             NaN        14.02
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_464.13       310.96     1_775.09       0.8727             NaN        14.02
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_464.13       379.71     1_843.84       0.9982          1.0004        14.02
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_464.13       656.26     2_120.39       0.9982          1.0004        14.02
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_464.13       468.89     1_933.02       1.0000          1.0000        14.02
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_464.13       784.33     2_248.45       1.0000          1.0000        14.02
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_464.13       540.42     2_004.55       1.0000          1.0000        14.02
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_464.13       872.22     2_336.35       1.0000          1.0000        14.02
IVF-TQ-b4-nl158 (self)                                 1_464.13     1_646.17     3_110.29       1.0000          1.0000        14.02
IVF-TQ-b4-nl223-np11-rf0 (query)                         787.67       179.13       966.81       0.8726             NaN        14.24
IVF-TQ-b4-nl223-np14-rf0 (query)                         787.67       206.88       994.55       0.8727             NaN        14.24
IVF-TQ-b4-nl223-np21-rf0 (query)                         787.67       266.86     1_054.53       0.8727             NaN        14.24
IVF-TQ-b4-nl223-np11-rf10 (query)                        787.67       365.12     1_152.79       0.9995          1.0001        14.24
IVF-TQ-b4-nl223-np11-rf20 (query)                        787.67       624.71     1_412.38       0.9995          1.0001        14.24
IVF-TQ-b4-nl223-np14-rf10 (query)                        787.67       403.81     1_191.49       0.9999          1.0000        14.24
IVF-TQ-b4-nl223-np14-rf20 (query)                        787.67       680.05     1_467.72       0.9999          1.0000        14.24
IVF-TQ-b4-nl223-np21-rf10 (query)                        787.67       477.09     1_264.77       1.0000          1.0000        14.24
IVF-TQ-b4-nl223-np21-rf20 (query)                        787.67       785.75     1_573.43       1.0000          1.0000        14.24
IVF-TQ-b4-nl223 (self)                                   787.67     1_461.25     2_248.92       1.0000          1.0000        14.24
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_006.75       185.24     1_191.99       0.8727             NaN        14.53
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_006.75       204.38     1_211.13       0.8727             NaN        14.53
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_006.75       260.59     1_267.34       0.8727             NaN        14.53
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_006.75       357.13     1_363.88       0.9998          1.0000        14.53
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_006.75       612.92     1_619.67       0.9998          1.0000        14.53
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_006.75       374.47     1_381.22       0.9999          1.0000        14.53
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_006.75       638.85     1_645.60       0.9999          1.0000        14.53
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_006.75       440.77     1_447.52       1.0000          1.0000        14.53
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_006.75       735.19     1_741.94       1.0000          1.0000        14.53
IVF-TQ-b4-nl316 (self)                                 1_006.75     1_393.31     2_400.06       1.0000          1.0000        14.53
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
Exhaustive (query)                                        20.72     9_682.09     9_702.81       1.0000          1.0000        97.66
Exhaustive (self)                                         20.72    32_197.00    32_217.72       1.0000          1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              425.11       641.90     1_067.01       0.8424             NaN        13.97
ExhaustiveTQ-b2-rf5 (query)                              425.11       734.76     1_159.87       1.0000          1.0000        13.97
ExhaustiveTQ-b2-rf10 (query)                             425.11       873.19     1_298.30       1.0000          1.0000        13.97
ExhaustiveTQ-b2-rf20 (query)                             425.11     1_270.01     1_695.13       1.0000          1.0000        13.97
ExhaustiveTQ-b2 (self)                                   425.11     4_207.52     4_632.63       1.0000          1.0000        13.97
ExhaustiveTQ-b4-rf0 (query)                              545.09     1_108.98     1_654.06       0.8985             NaN        26.18
ExhaustiveTQ-b4-rf5 (query)                              545.09     1_198.51     1_743.60       1.0000          1.0000        26.18
ExhaustiveTQ-b4-rf10 (query)                             545.09     1_329.79     1_874.88       1.0000          1.0000        26.18
ExhaustiveTQ-b4-rf20 (query)                             545.09     1_733.98     2_279.07       1.0000          1.0000        26.18
ExhaustiveTQ-b4 (self)                                   545.09     5_823.08     6_368.16       1.0000          1.0000        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        3_024.74       253.74     3_278.48       0.8420             NaN        14.96
IVF-TQ-b2-nl158-np12-rf0 (query)                       3_024.74       326.61     3_351.35       0.8424             NaN        14.96
IVF-TQ-b2-nl158-np17-rf0 (query)                       3_024.74       381.28     3_406.02       0.8424             NaN        14.96
IVF-TQ-b2-nl158-np7-rf10 (query)                       3_024.74       475.49     3_500.23       0.9986          1.0003        14.96
IVF-TQ-b2-nl158-np7-rf20 (query)                       3_024.74       776.64     3_801.38       0.9986          1.0003        14.96
IVF-TQ-b2-nl158-np12-rf10 (query)                      3_024.74       556.62     3_581.36       0.9999          1.0000        14.96
IVF-TQ-b2-nl158-np12-rf20 (query)                      3_024.74       890.84     3_915.58       0.9999          1.0000        14.96
IVF-TQ-b2-nl158-np17-rf10 (query)                      3_024.74       625.48     3_650.22       1.0000          1.0000        14.96
IVF-TQ-b2-nl158-np17-rf20 (query)                      3_024.74       971.94     3_996.68       1.0000          1.0000        14.96
IVF-TQ-b2-nl158 (self)                                 3_024.74     2_033.93     5_058.67       1.0000          1.0000        14.96
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_311.03       260.34     1_571.37       0.8424             NaN        15.24
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_311.03       287.58     1_598.61       0.8424             NaN        15.24
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_311.03       346.85     1_657.88       0.8424             NaN        15.24
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_311.03       469.42     1_780.45       0.9997          1.0000        15.24
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_311.03       753.05     2_064.09       0.9997          1.0000        15.24
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_311.03       499.61     1_810.65       0.9999          1.0000        15.24
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_311.03       797.09     2_108.13       0.9999          1.0000        15.24
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_311.03       570.39     1_881.43       1.0000          1.0000        15.24
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_311.03       885.75     2_196.78       1.0000          1.0000        15.24
IVF-TQ-b2-nl223 (self)                                 1_311.03     1_950.47     3_261.51       1.0000          1.0000        15.24
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_543.35       268.39     1_811.75       0.8424             NaN        15.57
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_543.35       282.41     1_825.77       0.8424             NaN        15.57
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_543.35       334.45     1_877.80       0.8424             NaN        15.57
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_543.35       472.75     2_016.10       0.9999          1.0000        15.57
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_543.35       757.67     2_301.02       0.9999          1.0000        15.57
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_543.35       487.93     2_031.28       1.0000          1.0000        15.57
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_543.35       784.18     2_327.54       1.0000          1.0000        15.57
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_543.35       547.86     2_091.21       1.0000          1.0000        15.57
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_543.35       863.06     2_406.41       1.0000          1.0000        15.57
IVF-TQ-b2-nl316 (self)                                 1_543.35     1_875.94     3_419.29       1.0000          1.0000        15.57
IVF-TQ-b4-nl158-np7-rf0 (query)                        3_189.29       352.33     3_541.62       0.8977             NaN        27.46
IVF-TQ-b4-nl158-np12-rf0 (query)                       3_189.29       475.65     3_664.95       0.8985             NaN        27.46
IVF-TQ-b4-nl158-np17-rf0 (query)                       3_189.29       582.30     3_771.60       0.8985             NaN        27.46
IVF-TQ-b4-nl158-np7-rf10 (query)                       3_189.29       581.03     3_770.33       0.9986          1.0003        27.46
IVF-TQ-b4-nl158-np7-rf20 (query)                       3_189.29       886.00     4_075.29       0.9986          1.0003        27.46
IVF-TQ-b4-nl158-np12-rf10 (query)                      3_189.29       721.78     3_911.07       0.9999          1.0000        27.46
IVF-TQ-b4-nl158-np12-rf20 (query)                      3_189.29     1_054.44     4_243.74       0.9999          1.0000        27.46
IVF-TQ-b4-nl158-np17-rf10 (query)                      3_189.29       829.44     4_018.74       1.0000          1.0000        27.46
IVF-TQ-b4-nl158-np17-rf20 (query)                      3_189.29     1_180.98     4_370.28       1.0000          1.0000        27.46
IVF-TQ-b4-nl158 (self)                                 3_189.29     2_329.99     5_519.28       1.0000          1.0000        27.46
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_423.23       364.11     1_787.34       0.8984             NaN        27.90
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_423.23       415.09     1_838.32       0.8984             NaN        27.90
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_423.23       562.01     1_985.24       0.8985             NaN        27.90
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_423.23       573.35     1_996.58       0.9997          1.0000        27.90
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_423.23       864.28     2_287.51       0.9997          1.0000        27.90
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_423.23       628.53     2_051.76       0.9999          1.0000        27.90
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_423.23       925.13     2_348.36       0.9999          1.0000        27.90
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_423.23       755.97     2_179.19       1.0000          1.0000        27.90
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_423.23     1_063.37     2_486.60       1.0000          1.0000        27.90
IVF-TQ-b4-nl223 (self)                                 1_423.23     2_175.07     3_598.30       1.0000          1.0000        27.90
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_659.24       375.25     2_034.49       0.8984             NaN        28.38
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_659.24       396.97     2_056.21       0.8984             NaN        28.38
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_659.24       488.88     2_148.12       0.8985             NaN        28.38
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_659.24       578.78     2_238.02       0.9999          1.0000        28.38
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_659.24       863.85     2_523.09       0.9999          1.0000        28.38
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_659.24       607.71     2_266.95       1.0000          1.0000        28.38
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_659.24       899.96     2_559.20       1.0000          1.0000        28.38
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_659.24       719.56     2_378.80       1.0000          1.0000        28.38
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_659.24     1_022.09     2_681.33       1.0000          1.0000        28.38
IVF-TQ-b4-nl316 (self)                                 1_659.24     2_120.09     3_779.33       1.0000          1.0000        28.38
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
Exhaustive (query)                                        30.84    15_607.65    15_638.48       1.0000          1.0000       146.48
Exhaustive (self)                                         30.84    52_248.93    52_279.77       1.0000          1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              869.33       969.94     1_839.27       0.8736             NaN        21.33
ExhaustiveTQ-b2-rf5 (query)                              869.33     1_075.66     1_944.99       1.0000          1.0000        21.33
ExhaustiveTQ-b2-rf10 (query)                             869.33     1_228.16     2_097.49       1.0000          1.0000        21.33
ExhaustiveTQ-b2-rf20 (query)                             869.33     1_656.22     2_525.55       1.0000          1.0000        21.33
ExhaustiveTQ-b2 (self)                                   869.33     5_489.68     6_359.00       1.0000          1.0000        21.33
ExhaustiveTQ-b4-rf0 (query)                            1_026.31     1_739.51     2_765.82       0.9097             NaN        39.64
ExhaustiveTQ-b4-rf5 (query)                            1_026.31     1_818.19     2_844.50       1.0000          1.0000        39.64
ExhaustiveTQ-b4-rf10 (query)                           1_026.31     1_960.62     2_986.93       1.0000          1.0000        39.64
ExhaustiveTQ-b4-rf20 (query)                           1_026.31     2_413.33     3_439.64       1.0000          1.0000        39.64
ExhaustiveTQ-b4 (self)                                 1_026.31     7_933.62     8_959.93       1.0000          1.0000        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        4_784.19       437.16     5_221.34       0.8735             NaN        22.61
IVF-TQ-b2-nl158-np12-rf0 (query)                       4_784.19       531.70     5_315.89       0.8736             NaN        22.61
IVF-TQ-b2-nl158-np17-rf0 (query)                       4_784.19       608.42     5_392.61       0.8736             NaN        22.61
IVF-TQ-b2-nl158-np7-rf10 (query)                       4_784.19       687.71     5_471.90       0.9995          1.0001        22.61
IVF-TQ-b2-nl158-np7-rf20 (query)                       4_784.19     1_006.98     5_791.17       0.9995          1.0001        22.61
IVF-TQ-b2-nl158-np12-rf10 (query)                      4_784.19       802.52     5_586.71       1.0000          1.0000        22.61
IVF-TQ-b2-nl158-np12-rf20 (query)                      4_784.19     1_146.07     5_930.26       1.0000          1.0000        22.61
IVF-TQ-b2-nl158-np17-rf10 (query)                      4_784.19       884.45     5_668.64       1.0000          1.0000        22.61
IVF-TQ-b2-nl158-np17-rf20 (query)                      4_784.19     1_248.38     6_032.56       1.0000          1.0000        22.61
IVF-TQ-b2-nl158 (self)                                 4_784.19     2_804.36     7_588.55       1.0000          1.0000        22.61
IVF-TQ-b2-nl223-np11-rf0 (query)                       2_126.57       447.99     2_574.56       0.8736             NaN        23.00
IVF-TQ-b2-nl223-np14-rf0 (query)                       2_126.57       485.74     2_612.32       0.8736             NaN        23.00
IVF-TQ-b2-nl223-np21-rf0 (query)                       2_126.57       569.13     2_695.70       0.8736             NaN        23.00
IVF-TQ-b2-nl223-np11-rf10 (query)                      2_126.57       685.82     2_812.40       0.9998          1.0000        23.00
IVF-TQ-b2-nl223-np11-rf20 (query)                      2_126.57     1_007.13     3_133.70       0.9998          1.0000        23.00
IVF-TQ-b2-nl223-np14-rf10 (query)                      2_126.57       738.34     2_864.91       1.0000          1.0000        23.00
IVF-TQ-b2-nl223-np14-rf20 (query)                      2_126.57     1_060.29     3_186.86       1.0000          1.0000        23.00
IVF-TQ-b2-nl223-np21-rf10 (query)                      2_126.57       817.06     2_943.63       1.0000          1.0000        23.00
IVF-TQ-b2-nl223-np21-rf20 (query)                      2_126.57     1_166.80     3_293.37       1.0000          1.0000        23.00
IVF-TQ-b2-nl223 (self)                                 2_126.57     2_662.50     4_789.07       1.0000          1.0000        23.00
IVF-TQ-b2-nl316-np15-rf0 (query)                       2_483.09       468.81     2_951.90       0.8736             NaN        23.53
IVF-TQ-b2-nl316-np17-rf0 (query)                       2_483.09       492.25     2_975.33       0.8736             NaN        23.53
IVF-TQ-b2-nl316-np25-rf0 (query)                       2_483.09       559.43     3_042.52       0.8736             NaN        23.53
IVF-TQ-b2-nl316-np15-rf10 (query)                      2_483.09       694.19     3_177.27       1.0000          1.0000        23.53
IVF-TQ-b2-nl316-np15-rf20 (query)                      2_483.09     1_007.79     3_490.88       1.0000          1.0000        23.53
IVF-TQ-b2-nl316-np17-rf10 (query)                      2_483.09       713.17     3_196.26       1.0000          1.0000        23.53
IVF-TQ-b2-nl316-np17-rf20 (query)                      2_483.09     1_047.38     3_530.47       1.0000          1.0000        23.53
IVF-TQ-b2-nl316-np25-rf10 (query)                      2_483.09       799.74     3_282.83       1.0000          1.0000        23.53
IVF-TQ-b2-nl316-np25-rf20 (query)                      2_483.09     1_143.81     3_626.89       1.0000          1.0000        23.53
IVF-TQ-b2-nl316 (self)                                 2_483.09     2_628.10     5_111.19       1.0000          1.0000        23.53
IVF-TQ-b4-nl158-np7-rf0 (query)                        4_944.06       600.16     5_544.23       0.9094             NaN        41.37
IVF-TQ-b4-nl158-np12-rf0 (query)                       4_944.06       775.05     5_719.12       0.9097             NaN        41.37
IVF-TQ-b4-nl158-np17-rf0 (query)                       4_944.06       919.86     5_863.93       0.9097             NaN        41.37
IVF-TQ-b4-nl158-np7-rf10 (query)                       4_944.06       855.68     5_799.74       0.9995          1.0001        41.37
IVF-TQ-b4-nl158-np7-rf20 (query)                       4_944.06     1_174.53     6_118.59       0.9995          1.0001        41.37
IVF-TQ-b4-nl158-np12-rf10 (query)                      4_944.06     1_046.30     5_990.36       1.0000          1.0000        41.37
IVF-TQ-b4-nl158-np12-rf20 (query)                      4_944.06     1_396.74     6_340.80       1.0000          1.0000        41.37
IVF-TQ-b4-nl158-np17-rf10 (query)                      4_944.06     1_200.27     6_144.33       1.0000          1.0000        41.37
IVF-TQ-b4-nl158-np17-rf20 (query)                      4_944.06     1_573.69     6_517.75       1.0000          1.0000        41.37
IVF-TQ-b4-nl158 (self)                                 4_944.06     3_267.52     8_211.59       1.0000          1.0000        41.37
IVF-TQ-b4-nl223-np11-rf0 (query)                       2_288.76       606.17     2_894.92       0.9096             NaN        41.96
IVF-TQ-b4-nl223-np14-rf0 (query)                       2_288.76       678.83     2_967.58       0.9097             NaN        41.96
IVF-TQ-b4-nl223-np21-rf0 (query)                       2_288.76       844.88     3_133.64       0.9097             NaN        41.96
IVF-TQ-b4-nl223-np11-rf10 (query)                      2_288.76       842.14     3_130.90       0.9998          1.0000        41.96
IVF-TQ-b4-nl223-np11-rf20 (query)                      2_288.76     1_164.64     3_453.40       0.9998          1.0000        41.96
IVF-TQ-b4-nl223-np14-rf10 (query)                      2_288.76       919.38     3_208.14       1.0000          1.0000        41.96
IVF-TQ-b4-nl223-np14-rf20 (query)                      2_288.76     1_250.96     3_539.71       1.0000          1.0000        41.96
IVF-TQ-b4-nl223-np21-rf10 (query)                      2_288.76     1_084.44     3_373.20       1.0000          1.0000        41.96
IVF-TQ-b4-nl223-np21-rf20 (query)                      2_288.76     1_454.79     3_743.55       1.0000          1.0000        41.96
IVF-TQ-b4-nl223 (self)                                 2_288.76     3_055.80     5_344.56       1.0000          1.0000        41.96
IVF-TQ-b4-nl316-np15-rf0 (query)                       2_613.69       624.90     3_238.59       0.9097             NaN        42.73
IVF-TQ-b4-nl316-np17-rf0 (query)                       2_613.69       664.60     3_278.29       0.9097             NaN        42.73
IVF-TQ-b4-nl316-np25-rf0 (query)                       2_613.69       807.44     3_421.13       0.9097             NaN        42.73
IVF-TQ-b4-nl316-np15-rf10 (query)                      2_613.69       852.72     3_466.41       1.0000          1.0000        42.73
IVF-TQ-b4-nl316-np15-rf20 (query)                      2_613.69     1_168.28     3_781.97       1.0000          1.0000        42.73
IVF-TQ-b4-nl316-np17-rf10 (query)                      2_613.69       889.92     3_503.61       1.0000          1.0000        42.73
IVF-TQ-b4-nl316-np17-rf20 (query)                      2_613.69     1_217.33     3_831.02       1.0000          1.0000        42.73
IVF-TQ-b4-nl316-np25-rf10 (query)                      2_613.69     1_046.95     3_660.63       1.0000          1.0000        42.73
IVF-TQ-b4-nl316-np25-rf20 (query)                      2_613.69     1_390.15     4_003.84       1.0000          1.0000        42.73
IVF-TQ-b4-nl316 (self)                                 2_613.69     2_994.16     5_607.85       1.0000          1.0000        42.73
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
