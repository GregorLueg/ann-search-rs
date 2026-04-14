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
===================================================================================================================================
Benchmark: 50k samples, 256D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        10.24     4_418.44     4_428.68       1.0000          1.0000        48.83
Exhaustive (self)                                         10.24    14_639.15    14_649.39       1.0000          1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_594.28       263.21     2_857.49       0.1006             NaN         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_594.28       361.78     2_956.06       0.2890          1.0907         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_594.28       474.08     3_068.36       0.4246          1.0554         1.78
ExhaustiveBinary-256-random (self)                     2_594.28     1_200.47     3_794.75       0.2886          1.0907         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_887.23       263.07     3_150.31       0.2261             NaN         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_887.23       378.82     3_266.05       0.5963          1.0281         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_887.23       484.78     3_372.02       0.7219          1.0155         1.78
ExhaustiveBinary-256-pca (self)                        2_887.23     1_241.21     4_128.45       0.5956          1.0283         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_050.85       464.19     5_515.05       0.1249             NaN         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_050.85       570.46     5_621.32       0.3718          1.0678         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_050.85       684.19     5_735.04       0.5258          1.0390         3.55
ExhaustiveBinary-512-random (self)                     5_050.85     1_921.60     6_972.46       0.3724          1.0679         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_440.02       473.69     5_913.72       0.2472             NaN         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_440.02       589.88     6_029.90       0.6835          1.0189         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_440.02       704.07     6_144.09       0.8321          1.0077         3.55
ExhaustiveBinary-512-pca (self)                        5_440.02     1_944.81     7_384.83       0.6838          1.0189         3.55
ExhaustiveBinary-1024-random_no_rr (query)             9_968.28       788.51    10_756.79       0.1692             NaN         7.10
ExhaustiveBinary-1024-random-rf10 (query)              9_968.28       908.01    10_876.29       0.5028          1.0425         7.10
ExhaustiveBinary-1024-random-rf20 (query)              9_968.28     1_037.05    11_005.33       0.6675          1.0220         7.10
ExhaustiveBinary-1024-random (self)                    9_968.28     3_006.20    12_974.48       0.5038          1.0426         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_617.09       808.08    11_425.18       0.2755             NaN         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_617.09       924.43    11_541.52       0.7370          1.0141         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_617.09     1_060.90    11_678.00       0.8733          1.0053         7.10
ExhaustiveBinary-1024-pca (self)                      10_617.09     3_084.24    13_701.33       0.7371          1.0141         7.10
ExhaustiveBinary-256-signed_no_rr (query)              2_566.07       256.67     2_822.74       0.1006             NaN         1.78
ExhaustiveBinary-256-signed-rf10 (query)               2_566.07       364.50     2_930.57       0.2890          1.0907         1.78
ExhaustiveBinary-256-signed-rf20 (query)               2_566.07       473.79     3_039.86       0.4246          1.0554         1.78
ExhaustiveBinary-256-signed (self)                     2_566.07     1_201.66     3_767.73       0.2886          1.0907         1.78
IVF-Binary-256-nl158-np7-rf0-random (query)            4_121.97       137.25     4_259.21       0.1063             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           4_121.97       151.70     4_273.67       0.1034             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           4_121.97       127.84     4_249.81       0.1017             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           4_121.97       182.26     4_304.23       0.3128          1.0834         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           4_121.97       235.12     4_357.09       0.4564          1.0499         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          4_121.97       197.08     4_319.05       0.2990          1.0878         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          4_121.97       281.58     4_403.54       0.4375          1.0533         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          4_121.97       206.84     4_328.81       0.2914          1.0901         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          4_121.97       264.15     4_386.11       0.4284          1.0549         1.93
IVF-Binary-256-nl158-random (self)                     4_121.97       579.57     4_701.53       0.2986          1.0878         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           3_171.08       125.01     3_296.09       0.1054             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           3_171.08       130.55     3_301.63       0.1027             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           3_171.08       131.08     3_302.17       0.1013             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          3_171.08       191.55     3_362.63       0.3066          1.0843         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          3_171.08       247.32     3_418.41       0.4485          1.0506         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          3_171.08       189.89     3_360.97       0.2976          1.0874         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          3_171.08       252.11     3_423.20       0.4366          1.0529         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          3_171.08       204.05     3_375.13       0.2925          1.0893         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          3_171.08       263.36     3_434.44       0.4296          1.0544         2.00
IVF-Binary-256-nl223-random (self)                     3_171.08       578.02     3_749.11       0.2974          1.0875         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           3_255.02       128.74     3_383.76       0.1048             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           3_255.02       136.68     3_391.71       0.1036             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           3_255.02       135.64     3_390.66       0.1019             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          3_255.02       195.66     3_450.69       0.3062          1.0846         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          3_255.02       249.14     3_504.16       0.4494          1.0505         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          3_255.02       193.92     3_448.95       0.3014          1.0861         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          3_255.02       257.08     3_512.10       0.4427          1.0519         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          3_255.02       204.67     3_459.69       0.2928          1.0893         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          3_255.02       261.78     3_516.80       0.4300          1.0544         2.09
IVF-Binary-256-nl316-random (self)                     3_255.02       595.35     3_850.37       0.3014          1.0861         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               4_436.03       121.46     4_557.49       0.2338             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              4_436.03       126.53     4_562.56       0.2321             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              4_436.03       134.87     4_570.90       0.2311             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              4_436.03       193.92     4_629.95       0.6495          1.0223         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              4_436.03       253.61     4_689.65       0.7928          1.0106         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             4_436.03       203.95     4_639.98       0.6501          1.0222         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             4_436.03       266.57     4_702.60       0.7994          1.0099         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             4_436.03       228.14     4_664.17       0.6462          1.0226         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             4_436.03       281.85     4_717.88       0.7949          1.0102         1.93
IVF-Binary-256-nl158-pca (self)                        4_436.03       624.30     5_060.34       0.6500          1.0223         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              3_416.22       125.97     3_542.19       0.2328             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              3_416.22       129.83     3_546.05       0.2314             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              3_416.22       135.10     3_551.32       0.2305             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             3_416.22       197.75     3_613.97       0.6533          1.0219         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             3_416.22       255.12     3_671.35       0.8035          1.0097         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             3_416.22       201.49     3_617.71       0.6507          1.0221         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             3_416.22       266.54     3_682.76       0.8016          1.0098         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             3_416.22       212.70     3_628.92       0.6465          1.0225         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             3_416.22       279.99     3_696.21       0.7959          1.0101         2.00
IVF-Binary-256-nl223-pca (self)                        3_416.22       633.90     4_050.12       0.6509          1.0222         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              3_565.98       132.74     3_698.71       0.2329             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              3_565.98       134.72     3_700.69       0.2323             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              3_565.98       138.91     3_704.88       0.2309             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             3_565.98       205.48     3_771.45       0.6559          1.0216         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             3_565.98       263.93     3_829.91       0.8063          1.0095         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             3_565.98       206.70     3_772.68       0.6542          1.0218         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             3_565.98       267.06     3_833.03       0.8054          1.0095         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             3_565.98       213.91     3_779.88       0.6484          1.0224         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             3_565.98       278.48     3_844.45       0.7987          1.0100         2.09
IVF-Binary-256-nl316-pca (self)                        3_565.98       638.23     4_204.21       0.6541          1.0219         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_628.72       268.15     6_896.87       0.1303             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_628.72       246.72     6_875.44       0.1275             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_628.72       232.02     6_860.74       0.1259             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_628.72       298.46     6_927.18       0.3902          1.0636         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_628.72       337.09     6_965.81       0.5457          1.0363         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_628.72       305.25     6_933.98       0.3795          1.0661         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_628.72       365.75     6_994.48       0.5338          1.0380         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_628.72       315.79     6_944.51       0.3742          1.0674         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_628.72       374.42     7_003.15       0.5273          1.0389         3.71
IVF-Binary-512-nl158-random (self)                     6_628.72       922.44     7_551.17       0.3794          1.0663         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_600.45       220.37     5_820.82       0.1288             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_600.45       227.72     5_828.18       0.1268             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_600.45       235.30     5_835.76       0.1257             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_600.45       287.93     5_888.38       0.3852          1.0641         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_600.45       347.29     5_947.74       0.5432          1.0363         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_600.45       294.60     5_895.05       0.3785          1.0659         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_600.45       354.62     5_955.08       0.5340          1.0377         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_600.45       308.34     5_908.80       0.3743          1.0671         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_600.45       372.39     5_972.84       0.5286          1.0386         3.77
IVF-Binary-512-nl223-random (self)                     5_600.45       920.49     6_520.95       0.3792          1.0660         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           5_732.45       226.58     5_959.04       0.1284             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           5_732.45       235.35     5_967.80       0.1276             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           5_732.45       235.07     5_967.53       0.1259             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          5_732.45       293.31     6_025.76       0.3866          1.0640         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          5_732.45       351.10     6_083.55       0.5453          1.0361         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          5_732.45       294.34     6_026.79       0.3826          1.0649         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          5_732.45       359.51     6_091.97       0.5398          1.0369         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          5_732.45       304.74     6_037.20       0.3750          1.0670         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          5_732.45       370.03     6_102.48       0.5290          1.0386         3.86
IVF-Binary-512-nl316-random (self)                     5_732.45       932.15     6_664.60       0.3829          1.0651         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               7_054.63       226.40     7_281.03       0.2494             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              7_054.63       235.31     7_289.94       0.2482             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              7_054.63       242.02     7_296.65       0.2474             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              7_054.63       296.67     7_351.30       0.6798          1.0193         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              7_054.63       354.81     7_409.44       0.8187          1.0089         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             7_054.63       308.71     7_363.34       0.6847          1.0187         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             7_054.63       373.25     7_427.88       0.8321          1.0078         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             7_054.63       330.87     7_385.49       0.6837          1.0188         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             7_054.63       409.88     7_464.51       0.8324          1.0077         3.71
IVF-Binary-512-nl158-pca (self)                        7_054.63       998.74     8_053.36       0.6856          1.0188         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              6_013.86       236.62     6_250.48       0.2491             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              6_013.86       240.00     6_253.86       0.2480             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              6_013.86       242.95     6_256.81       0.2473             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             6_013.86       303.57     6_317.43       0.6861          1.0186         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             6_013.86       360.96     6_374.82       0.8324          1.0078         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             6_013.86       306.89     6_320.75       0.6850          1.0187         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             6_013.86       370.10     6_383.95       0.8333          1.0077         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             6_013.86       321.48     6_335.34       0.6838          1.0188         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             6_013.86       390.05     6_403.91       0.8325          1.0077         3.77
IVF-Binary-512-nl223-pca (self)                        6_013.86       973.24     6_987.10       0.6855          1.0188         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              6_149.19       241.26     6_390.45       0.2494             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              6_149.19       237.18     6_386.37       0.2488             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              6_149.19       243.98     6_393.17       0.2477             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             6_149.19       313.16     6_462.35       0.6879          1.0184         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             6_149.19       374.53     6_523.72       0.8338          1.0077         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             6_149.19       311.30     6_460.49       0.6873          1.0185         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             6_149.19       373.23     6_522.42       0.8347          1.0076         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             6_149.19       320.19     6_469.38       0.6839          1.0188         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             6_149.19       387.55     6_536.74       0.8323          1.0078         3.86
IVF-Binary-512-nl316-pca (self)                        6_149.19       980.78     7_129.97       0.6875          1.0186         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_516.93       411.34    11_928.27       0.1737             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_516.93       423.82    11_940.75       0.1713             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_516.93       438.05    11_954.98       0.1698             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_516.93       484.38    12_001.30       0.5112          1.0412         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_516.93       537.12    12_054.05       0.6693          1.0219         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_516.93       495.59    12_012.52       0.5072          1.0419         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_516.93       562.24    12_079.16       0.6704          1.0217         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_516.93       515.98    12_032.91       0.5038          1.0424         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_516.93       587.67    12_104.60       0.6679          1.0220         7.26
IVF-Binary-1024-nl158-random (self)                   11_516.93     1_603.00    13_119.93       0.5082          1.0419         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_513.05       413.57    10_926.62       0.1725             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_513.05       422.19    10_935.24       0.1708             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_513.05       434.65    10_947.70       0.1699             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_513.05       485.61    10_998.66       0.5122          1.0407         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_513.05       549.92    11_062.97       0.6774          1.0208         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_513.05       492.17    11_005.22       0.5077          1.0416         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_513.05       559.39    11_072.44       0.6724          1.0214         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_513.05       513.23    11_026.28       0.5043          1.0422         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_513.05       580.74    11_093.79       0.6686          1.0219         7.32
IVF-Binary-1024-nl223-random (self)                   10_513.05     1_587.71    12_100.76       0.5085          1.0417         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         10_704.51       423.73    11_128.24       0.1721             NaN         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)         10_704.51       425.14    11_129.64       0.1715             NaN         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)         10_704.51       438.13    11_142.63       0.1701             NaN         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)        10_704.51       504.39    11_208.90       0.5134          1.0407         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)        10_704.51       548.41    11_252.92       0.6788          1.0207         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)        10_704.51       493.82    11_198.33       0.5106          1.0411         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)        10_704.51       562.95    11_267.46       0.6760          1.0211         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)        10_704.51       507.79    11_212.30       0.5048          1.0422         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)        10_704.51       578.88    11_283.39       0.6689          1.0218         7.41
IVF-Binary-1024-nl316-random (self)                   10_704.51     1_586.99    12_291.50       0.5114          1.0412         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)             12_161.27       428.30    12_589.57       0.2769             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            12_161.27       442.06    12_603.33       0.2764             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            12_161.27       451.82    12_613.09       0.2757             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            12_161.27       499.88    12_661.15       0.7271          1.0150         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            12_161.27       561.86    12_723.14       0.8507          1.0069         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           12_161.27       518.83    12_680.10       0.7373          1.0140         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           12_161.27       587.89    12_749.17       0.8718          1.0054         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           12_161.27       537.89    12_699.17       0.7374          1.0140         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           12_161.27       621.33    12_782.60       0.8736          1.0053         7.26
IVF-Binary-1024-nl158-pca (self)                      12_161.27     1_679.18    13_840.45       0.7376          1.0141         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            11_165.79       433.32    11_599.11       0.2772             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            11_165.79       444.66    11_610.45       0.2763             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            11_165.79       449.39    11_615.18       0.2756             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           11_165.79       506.48    11_672.27       0.7381          1.0140         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           11_165.79       571.98    11_737.77       0.8706          1.0055         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           11_165.79       526.66    11_692.45       0.7380          1.0140         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           11_165.79       580.14    11_745.93       0.8736          1.0053         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           11_165.79       535.35    11_701.14       0.7371          1.0141         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           11_165.79       606.68    11_772.47       0.8733          1.0053         7.32
IVF-Binary-1024-nl223-pca (self)                      11_165.79     1_653.96    12_819.75       0.7385          1.0140         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            11_323.43       453.83    11_777.26       0.2775             NaN         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            11_323.43       439.91    11_763.33       0.2770             NaN         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            11_323.43       453.48    11_776.90       0.2759             NaN         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           11_323.43       522.29    11_845.71       0.7391          1.0139         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           11_323.43       575.39    11_898.82       0.8717          1.0054         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           11_323.43       519.65    11_843.07       0.7394          1.0139         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           11_323.43       588.67    11_912.10       0.8740          1.0053         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           11_323.43       531.31    11_854.73       0.7375          1.0140         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           11_323.43       603.99    11_927.41       0.8737          1.0053         7.42
IVF-Binary-1024-nl316-pca (self)                      11_323.43     1_680.41    13_003.84       0.7395          1.0139         7.42
IVF-Binary-256-nl158-np7-rf0-signed (query)            4_122.39       117.84     4_240.23       0.1063             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-signed (query)           4_122.39       123.01     4_245.40       0.1034             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-signed (query)           4_122.39       128.92     4_251.31       0.1017             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-signed (query)           4_122.39       181.20     4_303.59       0.3128          1.0834         1.93
IVF-Binary-256-nl158-np7-rf20-signed (query)           4_122.39       234.97     4_357.36       0.4564          1.0499         1.93
IVF-Binary-256-nl158-np12-rf10-signed (query)          4_122.39       192.29     4_314.69       0.2990          1.0878         1.93
IVF-Binary-256-nl158-np12-rf20-signed (query)          4_122.39       251.87     4_374.26       0.4375          1.0533         1.93
IVF-Binary-256-nl158-np17-rf10-signed (query)          4_122.39       201.77     4_324.16       0.2914          1.0901         1.93
IVF-Binary-256-nl158-np17-rf20-signed (query)          4_122.39       266.06     4_388.45       0.4284          1.0549         1.93
IVF-Binary-256-nl158-signed (self)                     4_122.39       580.79     4_703.18       0.2986          1.0878         1.93
IVF-Binary-256-nl223-np11-rf0-signed (query)           3_099.56       123.35     3_222.91       0.1054             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-signed (query)           3_099.56       125.42     3_224.97       0.1027             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-signed (query)           3_099.56       132.07     3_231.63       0.1013             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-signed (query)          3_099.56       188.69     3_288.24       0.3066          1.0843         2.00
IVF-Binary-256-nl223-np11-rf20-signed (query)          3_099.56       240.94     3_340.49       0.4485          1.0506         2.00
IVF-Binary-256-nl223-np14-rf10-signed (query)          3_099.56       189.54     3_289.10       0.2976          1.0874         2.00
IVF-Binary-256-nl223-np14-rf20-signed (query)          3_099.56       250.85     3_350.40       0.4366          1.0529         2.00
IVF-Binary-256-nl223-np21-rf10-signed (query)          3_099.56       200.94     3_300.49       0.2925          1.0893         2.00
IVF-Binary-256-nl223-np21-rf20-signed (query)          3_099.56       268.09     3_367.65       0.4296          1.0544         2.00
IVF-Binary-256-nl223-signed (self)                     3_099.56       589.09     3_688.64       0.2974          1.0875         2.00
IVF-Binary-256-nl316-np15-rf0-signed (query)           3_256.38       135.59     3_391.97       0.1048             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-signed (query)           3_256.38       129.40     3_385.78       0.1036             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-signed (query)           3_256.38       135.05     3_391.43       0.1019             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-signed (query)          3_256.38       200.28     3_456.66       0.3062          1.0846         2.09
IVF-Binary-256-nl316-np15-rf20-signed (query)          3_256.38       248.21     3_504.59       0.4494          1.0505         2.09
IVF-Binary-256-nl316-np17-rf10-signed (query)          3_256.38       196.00     3_452.38       0.3014          1.0861         2.09
IVF-Binary-256-nl316-np17-rf20-signed (query)          3_256.38       254.25     3_510.63       0.4427          1.0519         2.09
IVF-Binary-256-nl316-np25-rf10-signed (query)          3_256.38       200.39     3_456.77       0.2928          1.0893         2.09
IVF-Binary-256-nl316-np25-rf20-signed (query)          3_256.38       266.99     3_523.37       0.4300          1.0544         2.09
IVF-Binary-256-nl316-signed (self)                     3_256.38       593.65     3_850.03       0.3014          1.0861         2.09
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
Exhaustive (query)                                        20.36    10_073.29    10_093.66       1.0000          1.0000        97.66
Exhaustive (self)                                         20.36    33_724.82    33_745.19       1.0000          1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_679.57       370.06     6_049.63       0.0893             NaN         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_679.57       502.59     6_182.16       0.2397          1.0749         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_679.57       637.68     6_317.25       0.3605          1.0474         2.03
ExhaustiveBinary-256-random (self)                     5_679.57     1_659.00     7_338.57       0.2400          1.0748         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_307.89       377.37     6_685.26       0.1907             NaN         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_307.89       521.90     6_829.79       0.5143          1.0274         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_307.89       654.94     6_962.83       0.6417          1.0162         2.03
ExhaustiveBinary-256-pca (self)                        6_307.89     1_703.67     8_011.56       0.5163          1.0273         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_177.16       684.53    11_861.69       0.1039             NaN         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_177.16       817.88    11_995.04       0.2971          1.0611         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_177.16       966.45    12_143.61       0.4344          1.0373         4.05
ExhaustiveBinary-512-random (self)                    11_177.16     2_712.87    13_890.02       0.2968          1.0610         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_958.71       707.53    12_666.24       0.2182             NaN         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_958.71       837.29    12_796.00       0.5365          1.0253         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_958.71       982.06    12_940.76       0.6508          1.0154         4.05
ExhaustiveBinary-512-pca (self)                       11_958.71     2_777.64    14_736.35       0.5379          1.0252         4.05
ExhaustiveBinary-1024-random_no_rr (query)            22_142.56     1_231.18    23_373.74       0.1273             NaN         8.10
ExhaustiveBinary-1024-random-rf10 (query)             22_142.56     1_376.09    23_518.65       0.3824          1.0455         8.10
ExhaustiveBinary-1024-random-rf20 (query)             22_142.56     1_559.55    23_702.10       0.5379          1.0261         8.10
ExhaustiveBinary-1024-random (self)                   22_142.56     4_591.56    26_734.12       0.3828          1.0454         8.10
ExhaustiveBinary-1024-pca_no_rr (query)               23_340.79     1_254.65    24_595.43       0.2506             NaN         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                23_340.79     1_421.18    24_761.97       0.6932          1.0125         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                23_340.79     1_582.62    24_923.41       0.8406          1.0051         8.11
ExhaustiveBinary-1024-pca (self)                      23_340.79     4_751.07    28_091.86       0.6937          1.0125         8.11
ExhaustiveBinary-512-signed_no_rr (query)             11_088.66       693.84    11_782.50       0.1039             NaN         4.05
ExhaustiveBinary-512-signed-rf10 (query)              11_088.66       818.80    11_907.46       0.2971          1.0611         4.05
ExhaustiveBinary-512-signed-rf20 (query)              11_088.66       966.40    12_055.06       0.4344          1.0373         4.05
ExhaustiveBinary-512-signed (self)                    11_088.66     2_700.35    13_789.01       0.2968          1.0610         4.05
IVF-Binary-256-nl158-np7-rf0-random (query)            8_924.78       238.31     9_163.08       0.0930             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           8_924.78       248.90     9_173.67       0.0910             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           8_924.78       248.74     9_173.51       0.0899             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           8_924.78       338.08     9_262.86       0.2562          1.0706         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           8_924.78       422.04     9_346.82       0.3841          1.0441         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          8_924.78       341.28     9_266.05       0.2455          1.0732         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          8_924.78       430.07     9_354.84       0.3695          1.0461         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          8_924.78       349.41     9_274.19       0.2406          1.0746         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          8_924.78       443.17     9_367.94       0.3628          1.0472         2.34
IVF-Binary-256-nl158-random (self)                     8_924.78     1_053.05     9_977.83       0.2463          1.0731         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           6_624.51       246.92     6_871.44       0.0926             NaN         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)           6_624.51       252.94     6_877.45       0.0914             NaN         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)           6_624.51       256.37     6_880.89       0.0898             NaN         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)          6_624.51       344.87     6_969.38       0.2507          1.0715         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)          6_624.51       427.94     7_052.45       0.3754          1.0449         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)          6_624.51       346.60     6_971.11       0.2453          1.0732         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)          6_624.51       431.41     7_055.93       0.3675          1.0462         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)          6_624.51       361.72     6_986.23       0.2406          1.0748         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)          6_624.51       455.03     7_079.54       0.3612          1.0474         2.46
IVF-Binary-256-nl223-random (self)                     6_624.51     1_064.90     7_689.42       0.2459          1.0730         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)           6_958.22       262.50     7_220.73       0.0916             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           6_958.22       263.97     7_222.19       0.0908             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           6_958.22       267.91     7_226.13       0.0900             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          6_958.22       355.51     7_313.73       0.2484          1.0722         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          6_958.22       446.89     7_405.11       0.3725          1.0453         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          6_958.22       356.90     7_315.12       0.2446          1.0733         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          6_958.22       448.06     7_406.28       0.3671          1.0463         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          6_958.22       367.86     7_326.08       0.2412          1.0744         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          6_958.22       468.49     7_426.71       0.3624          1.0471         2.65
IVF-Binary-256-nl316-random (self)                     6_958.22     1_121.05     8_079.27       0.2449          1.0733         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               9_649.37       250.21     9_899.58       0.1962             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              9_649.37       257.50     9_906.87       0.1955             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              9_649.37       259.77     9_909.14       0.1948             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              9_649.37       348.21     9_997.58       0.5664          1.0224         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              9_649.37       438.23    10_087.60       0.7221          1.0113         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             9_649.37       355.26    10_004.63       0.5686          1.0221         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             9_649.37       454.45    10_103.82       0.7288          1.0108         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             9_649.37       376.66    10_026.03       0.5653          1.0224         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             9_649.37       470.15    10_119.52       0.7241          1.0110         2.34
IVF-Binary-256-nl158-pca (self)                        9_649.37     1_119.98    10_769.36       0.5708          1.0220         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              7_286.22       255.55     7_541.77       0.1962             NaN         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              7_286.22       259.99     7_546.21       0.1954             NaN         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              7_286.22       274.21     7_560.43       0.1948             NaN         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             7_286.22       355.57     7_641.79       0.5703          1.0220         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             7_286.22       453.40     7_739.62       0.7308          1.0106         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             7_286.22       364.51     7_650.74       0.5680          1.0222         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             7_286.22       462.77     7_748.99       0.7292          1.0107         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             7_286.22       369.17     7_655.39       0.5655          1.0224         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             7_286.22       472.57     7_758.79       0.7249          1.0110         2.47
IVF-Binary-256-nl223-pca (self)                        7_286.22     1_125.51     8_411.73       0.5703          1.0221         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              7_629.80       267.17     7_896.97       0.1960             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              7_629.80       275.72     7_905.52       0.1954             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              7_629.80       275.08     7_904.88       0.1949             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             7_629.80       371.36     8_001.16       0.5718          1.0218         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             7_629.80       466.51     8_096.31       0.7343          1.0105         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             7_629.80       371.39     8_001.19       0.5698          1.0220         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             7_629.80       492.96     8_122.76       0.7321          1.0106         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             7_629.80       381.29     8_011.09       0.5670          1.0222         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             7_629.80       486.33     8_116.12       0.7274          1.0108         2.65
IVF-Binary-256-nl316-pca (self)                        7_629.80     1_184.24     8_814.04       0.5719          1.0219         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           14_452.41       451.44    14_903.85       0.1075             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          14_452.41       464.13    14_916.53       0.1057             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          14_452.41       473.01    14_925.42       0.1046             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          14_452.41       543.47    14_995.88       0.3090          1.0587         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          14_452.41       628.08    15_080.49       0.4503          1.0355         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         14_452.41       558.14    15_010.55       0.3015          1.0601         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         14_452.41       649.17    15_101.58       0.4408          1.0365         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         14_452.41       568.50    15_020.91       0.2976          1.0610         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         14_452.41       665.15    15_117.56       0.4354          1.0372         4.36
IVF-Binary-512-nl158-random (self)                    14_452.41     1_782.89    16_235.30       0.3015          1.0601         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          12_231.02       471.02    12_702.04       0.1064             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          12_231.02       471.92    12_702.94       0.1054             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          12_231.02       476.70    12_707.72       0.1044             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         12_231.02       552.94    12_783.96       0.3056          1.0589         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         12_231.02       640.75    12_871.77       0.4458          1.0357         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         12_231.02       555.29    12_786.31       0.3007          1.0601         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         12_231.02       652.40    12_883.42       0.4393          1.0366         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         12_231.02       590.84    12_821.86       0.2972          1.0611         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         12_231.02       664.64    12_895.67       0.4349          1.0373         4.49
IVF-Binary-512-nl223-random (self)                    12_231.02     1_781.04    14_012.07       0.3008          1.0601         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          12_461.35       476.01    12_937.37       0.1060             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          12_461.35       475.87    12_937.22       0.1053             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          12_461.35       486.19    12_947.54       0.1047             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         12_461.35       563.47    13_024.82       0.3039          1.0594         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         12_461.35       649.97    13_111.32       0.4442          1.0360         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         12_461.35       569.30    13_030.65       0.3008          1.0602         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         12_461.35       662.74    13_124.09       0.4397          1.0366         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         12_461.35       577.39    13_038.74       0.2978          1.0609         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         12_461.35       669.82    13_131.17       0.4360          1.0371         4.67
IVF-Binary-512-nl316-random (self)                    12_461.35     1_825.41    14_286.77       0.3008          1.0601         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              15_270.53       489.67    15_760.20       0.2351             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             15_270.53       484.95    15_755.49       0.2346             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             15_270.53       489.47    15_760.01       0.2334             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             15_270.53       570.83    15_841.36       0.6487          1.0157         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             15_270.53       652.77    15_923.30       0.7884          1.0077         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            15_270.53       573.77    15_844.30       0.6521          1.0153         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            15_270.53       713.38    15_983.91       0.7985          1.0069         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            15_270.53       610.19    15_880.72       0.6451          1.0157         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            15_270.53       697.45    15_967.98       0.7900          1.0073         4.36
IVF-Binary-512-nl158-pca (self)                       15_270.53     1_854.37    17_124.90       0.6524          1.0153         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             13_020.67       484.00    13_504.67       0.2347             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             13_020.67       477.34    13_498.01       0.2340             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             13_020.67       485.67    13_506.34       0.2332             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            13_020.67       572.38    13_593.05       0.6559          1.0150         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            13_020.67       666.79    13_687.46       0.8048          1.0067         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            13_020.67       576.53    13_597.20       0.6536          1.0151         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            13_020.67       678.50    13_699.17       0.8031          1.0067         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            13_020.67       597.69    13_618.36       0.6471          1.0156         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            13_020.67       702.58    13_723.25       0.7931          1.0072         4.49
IVF-Binary-512-nl223-pca (self)                       13_020.67     1_862.95    14_883.62       0.6538          1.0152         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             13_275.92       488.54    13_764.46       0.2352             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             13_275.92       488.36    13_764.28       0.2347             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             13_275.92       497.47    13_773.39       0.2339             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            13_275.92       586.71    13_862.63       0.6589          1.0148         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            13_275.92       679.62    13_955.54       0.8096          1.0064         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            13_275.92       586.69    13_862.62       0.6568          1.0149         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            13_275.92       720.18    13_996.11       0.8076          1.0065         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            13_275.92       597.43    13_873.36       0.6502          1.0154         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            13_275.92       704.90    13_980.82       0.7981          1.0069         4.67
IVF-Binary-512-nl316-pca (self)                       13_275.92     2_017.84    15_293.76       0.6571          1.0149         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          25_329.32       868.46    26_197.78       0.1306             NaN         8.41
IVF-Binary-1024-nl158-np12-rf0-random (query)         25_329.32       886.64    26_215.96       0.1289             NaN         8.41
IVF-Binary-1024-nl158-np17-rf0-random (query)         25_329.32       901.28    26_230.60       0.1279             NaN         8.41
IVF-Binary-1024-nl158-np7-rf10-random (query)         25_329.32       956.05    26_285.38       0.3897          1.0444         8.41
IVF-Binary-1024-nl158-np7-rf20-random (query)         25_329.32     1_047.00    26_376.32       0.5436          1.0256         8.41
IVF-Binary-1024-nl158-np12-rf10-random (query)        25_329.32       986.28    26_315.60       0.3858          1.0449         8.41
IVF-Binary-1024-nl158-np12-rf20-random (query)        25_329.32     1_069.43    26_398.75       0.5426          1.0257         8.41
IVF-Binary-1024-nl158-np17-rf10-random (query)        25_329.32     1_000.66    26_329.98       0.3828          1.0454         8.41
IVF-Binary-1024-nl158-np17-rf20-random (query)        25_329.32     1_181.37    26_510.69       0.5386          1.0260         8.41
IVF-Binary-1024-nl158-random (self)                   25_329.32     3_229.81    28_559.13       0.3865          1.0449         8.41
IVF-Binary-1024-nl223-np11-rf0-random (query)         23_112.52       880.56    23_993.08       0.1299             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         23_112.52       881.06    23_993.59       0.1289             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         23_112.52       897.53    24_010.05       0.1280             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        23_112.52       977.45    24_089.98       0.3892          1.0442         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        23_112.52     1_055.90    24_168.43       0.5456          1.0253         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        23_112.52       973.75    24_086.27       0.3854          1.0449         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        23_112.52     1_067.20    24_179.72       0.5398          1.0259         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        23_112.52       995.82    24_108.34       0.3829          1.0454         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        23_112.52     1_107.89    24_220.41       0.5372          1.0262         8.54
IVF-Binary-1024-nl223-random (self)                   23_112.52     3_154.43    26_266.95       0.3854          1.0450         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         23_516.83       889.78    24_406.61       0.1291             NaN         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)         23_516.83       901.78    24_418.62       0.1284             NaN         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)         23_516.83       905.15    24_421.98       0.1278             NaN         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)        23_516.83       980.03    24_496.87       0.3879          1.0445         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)        23_516.83     1_245.45    24_762.28       0.5448          1.0254         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)        23_516.83     1_088.38    24_605.21       0.3851          1.0450         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)        23_516.83     1_169.09    24_685.92       0.5409          1.0258         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)        23_516.83     1_086.35    24_603.18       0.3830          1.0454         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)        23_516.83     1_163.30    24_680.13       0.5381          1.0261         8.72
IVF-Binary-1024-nl316-random (self)                   23_516.83     3_314.23    26_831.06       0.3856          1.0449         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)             26_860.89       897.57    27_758.46       0.2510             NaN         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            26_860.89       910.47    27_771.36       0.2512             NaN         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            26_860.89       925.86    27_786.75       0.2507             NaN         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            26_860.89       991.62    27_852.51       0.6813          1.0134         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            26_860.89     1_075.46    27_936.35       0.8163          1.0064         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           26_860.89     1_017.32    27_878.21       0.6931          1.0125         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           26_860.89     1_101.68    27_962.57       0.8385          1.0052         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           26_860.89     1_033.82    27_894.71       0.6931          1.0125         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           26_860.89     1_126.76    27_987.65       0.8405          1.0051         8.42
IVF-Binary-1024-nl158-pca (self)                      26_860.89     3_292.69    30_153.58       0.6933          1.0125         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            24_321.40       904.66    25_226.06       0.2512             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            24_321.40       918.87    25_240.28       0.2509             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            24_321.40       926.09    25_247.50       0.2507             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           24_321.40       993.74    25_315.15       0.6923          1.0125         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           24_321.40     1_086.65    25_408.06       0.8371          1.0053         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           24_321.40     1_002.60    25_324.01       0.6936          1.0125         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           24_321.40     1_109.25    25_430.66       0.8405          1.0051         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           24_321.40     1_040.78    25_362.18       0.6932          1.0125         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           24_321.40     1_163.05    25_484.45       0.8405          1.0051         8.54
IVF-Binary-1024-nl223-pca (self)                      24_321.40     3_278.55    27_599.96       0.6941          1.0125         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            24_976.75     1_002.21    25_978.96       0.2512             NaN         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            24_976.75     1_038.55    26_015.30       0.2509             NaN         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            24_976.75     1_088.86    26_065.60       0.2506             NaN         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           24_976.75     1_132.97    26_109.72       0.6943          1.0124         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           24_976.75     1_187.00    26_163.74       0.8399          1.0051         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           24_976.75     1_041.40    26_018.15       0.6939          1.0124         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           24_976.75     1_121.59    26_098.34       0.8408          1.0051         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           24_976.75     1_030.40    26_007.15       0.6932          1.0125         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           24_976.75     1_135.04    26_111.79       0.8405          1.0051         8.73
IVF-Binary-1024-nl316-pca (self)                      24_976.75     3_308.74    28_285.48       0.6943          1.0125         8.73
IVF-Binary-512-nl158-np7-rf0-signed (query)           14_431.36       449.16    14_880.52       0.1075             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-signed (query)          14_431.36       460.81    14_892.17       0.1057             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-signed (query)          14_431.36       470.51    14_901.87       0.1046             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-signed (query)          14_431.36       539.52    14_970.88       0.3090          1.0587         4.36
IVF-Binary-512-nl158-np7-rf20-signed (query)          14_431.36       626.62    15_057.98       0.4503          1.0355         4.36
IVF-Binary-512-nl158-np12-rf10-signed (query)         14_431.36       563.47    14_994.83       0.3015          1.0601         4.36
IVF-Binary-512-nl158-np12-rf20-signed (query)         14_431.36       642.72    15_074.07       0.4408          1.0365         4.36
IVF-Binary-512-nl158-np17-rf10-signed (query)         14_431.36       569.17    15_000.52       0.2976          1.0610         4.36
IVF-Binary-512-nl158-np17-rf20-signed (query)         14_431.36       659.06    15_090.42       0.4354          1.0372         4.36
IVF-Binary-512-nl158-signed (self)                    14_431.36     1_764.29    16_195.64       0.3015          1.0601         4.36
IVF-Binary-512-nl223-np11-rf0-signed (query)          12_162.06       457.90    12_619.95       0.1064             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-signed (query)          12_162.06       463.76    12_625.81       0.1054             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-signed (query)          12_162.06       475.58    12_637.63       0.1044             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-signed (query)         12_162.06       549.28    12_711.34       0.3056          1.0589         4.49
IVF-Binary-512-nl223-np11-rf20-signed (query)         12_162.06       638.38    12_800.44       0.4458          1.0357         4.49
IVF-Binary-512-nl223-np14-rf10-signed (query)         12_162.06       553.30    12_715.35       0.3007          1.0601         4.49
IVF-Binary-512-nl223-np14-rf20-signed (query)         12_162.06       643.92    12_805.97       0.4393          1.0366         4.49
IVF-Binary-512-nl223-np21-rf10-signed (query)         12_162.06       565.87    12_727.93       0.2972          1.0611         4.49
IVF-Binary-512-nl223-np21-rf20-signed (query)         12_162.06       667.24    12_829.29       0.4349          1.0373         4.49
IVF-Binary-512-nl223-signed (self)                    12_162.06     1_779.61    13_941.67       0.3008          1.0601         4.49
IVF-Binary-512-nl316-np15-rf0-signed (query)          12_453.10       475.66    12_928.76       0.1060             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-signed (query)          12_453.10       475.27    12_928.37       0.1053             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-signed (query)          12_453.10       485.02    12_938.12       0.1047             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-signed (query)         12_453.10       563.39    13_016.49       0.3039          1.0594         4.67
IVF-Binary-512-nl316-np15-rf20-signed (query)         12_453.10       651.75    13_104.85       0.4442          1.0360         4.67
IVF-Binary-512-nl316-np17-rf10-signed (query)         12_453.10       565.39    13_018.49       0.3008          1.0602         4.67
IVF-Binary-512-nl316-np17-rf20-signed (query)         12_453.10       653.10    13_106.20       0.4397          1.0366         4.67
IVF-Binary-512-nl316-np25-rf10-signed (query)         12_453.10       576.02    13_029.12       0.2978          1.0609         4.67
IVF-Binary-512-nl316-np25-rf20-signed (query)         12_453.10       669.49    13_122.60       0.4360          1.0371         4.67
IVF-Binary-512-nl316-signed (self)                    12_453.10     1_811.10    14_264.20       0.3008          1.0601         4.67
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 1024 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 1024D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        41.74    23_078.98    23_120.71       1.0000          1.0000       195.31
Exhaustive (self)                                         41.74    76_502.45    76_544.18       1.0000          1.0000       195.31
ExhaustiveBinary-256-random_no_rr (query)             11_982.00       595.68    12_577.68       0.0840             NaN         2.53
ExhaustiveBinary-256-random-rf10 (query)              11_982.00       769.16    12_751.16       0.2060          1.0589         2.53
ExhaustiveBinary-256-random-rf20 (query)              11_982.00       951.30    12_933.31       0.3135          1.0384         2.53
ExhaustiveBinary-256-random (self)                    11_982.00     2_526.78    14_508.78       0.2052          1.0589         2.53
ExhaustiveBinary-256-pca_no_rr (query)                13_609.10       603.58    14_212.67       0.1619             NaN         2.53
ExhaustiveBinary-256-pca-rf10 (query)                 13_609.10       789.48    14_398.57       0.4353          1.0256         2.53
ExhaustiveBinary-256-pca-rf20 (query)                 13_609.10       991.05    14_600.14       0.5548          1.0162         2.53
ExhaustiveBinary-256-pca (self)                       13_609.10     2_578.33    16_187.43       0.4349          1.0256         2.53
ExhaustiveBinary-512-random_no_rr (query)             24_021.89     1_165.17    25_187.06       0.0922             NaN         5.05
ExhaustiveBinary-512-random-rf10 (query)              24_021.89     1_310.34    25_332.23       0.2430          1.0508         5.05
ExhaustiveBinary-512-random-rf20 (query)              24_021.89     1_535.58    25_557.47       0.3648          1.0323         5.05
ExhaustiveBinary-512-random (self)                    24_021.89     4_362.65    28_384.54       0.2441          1.0506         5.05
ExhaustiveBinary-512-pca_no_rr (query)                25_448.32     1_162.41    26_610.73       0.1735             NaN         5.06
ExhaustiveBinary-512-pca-rf10 (query)                 25_448.32     1_336.10    26_784.42       0.4277          1.0267         5.06
ExhaustiveBinary-512-pca-rf20 (query)                 25_448.32     1_530.42    26_978.75       0.5354          1.0175         5.06
ExhaustiveBinary-512-pca (self)                       25_448.32     4_407.39    29_855.71       0.4286          1.0266         5.06
ExhaustiveBinary-1024-random_no_rr (query)            47_213.38     2_137.18    49_350.56       0.1077             NaN        10.10
ExhaustiveBinary-1024-random-rf10 (query)             47_213.38     2_316.12    49_529.50       0.3093          1.0402        10.10
ExhaustiveBinary-1024-random-rf20 (query)             47_213.38     2_528.60    49_741.98       0.4493          1.0244        10.10
ExhaustiveBinary-1024-random (self)                   47_213.38     7_704.51    54_917.89       0.3094          1.0401        10.10
ExhaustiveBinary-1024-pca_no_rr (query)               48_770.06     2_478.86    51_248.92       0.2002             NaN        10.11
ExhaustiveBinary-1024-pca-rf10 (query)                48_770.06     2_601.31    51_371.37       0.4677          1.0246        10.11
ExhaustiveBinary-1024-pca-rf20 (query)                48_770.06     2_661.56    51_431.62       0.5715          1.0152        10.11
ExhaustiveBinary-1024-pca (self)                      48_770.06     8_461.54    57_231.60       0.4688          1.0240        10.11
ExhaustiveBinary-1024-signed_no_rr (query)            47_274.80     2_144.77    49_419.57       0.1077             NaN        10.10
ExhaustiveBinary-1024-signed-rf10 (query)             47_274.80     2_343.71    49_618.51       0.3093          1.0402        10.10
ExhaustiveBinary-1024-signed-rf20 (query)             47_274.80     2_516.31    49_791.11       0.4493          1.0244        10.10
ExhaustiveBinary-1024-signed (self)                   47_274.80     7_679.06    54_953.85       0.3094          1.0401        10.10
IVF-Binary-256-nl158-np7-rf0-random (query)           19_474.18       517.23    19_991.41       0.0856             NaN         3.14
IVF-Binary-256-nl158-np12-rf0-random (query)          19_474.18       524.07    19_998.26       0.0848             NaN         3.14
IVF-Binary-256-nl158-np17-rf0-random (query)          19_474.18       519.34    19_993.52       0.0843             NaN         3.14
IVF-Binary-256-nl158-np7-rf10-random (query)          19_474.18       633.85    20_108.03       0.2155          1.0567         3.14
IVF-Binary-256-nl158-np7-rf20-random (query)          19_474.18       781.87    20_256.06       0.3261          1.0368         3.14
IVF-Binary-256-nl158-np12-rf10-random (query)         19_474.18       667.18    20_141.37       0.2106          1.0579         3.14
IVF-Binary-256-nl158-np12-rf20-random (query)         19_474.18       781.23    20_255.41       0.3185          1.0377         3.14
IVF-Binary-256-nl158-np17-rf10-random (query)         19_474.18       656.07    20_130.25       0.2076          1.0586         3.14
IVF-Binary-256-nl158-np17-rf20-random (query)         19_474.18       795.36    20_269.54       0.3141          1.0383         3.14
IVF-Binary-256-nl158-random (self)                    19_474.18     2_092.34    21_566.53       0.2095          1.0579         3.14
IVF-Binary-256-nl223-np11-rf0-random (query)          13_883.57       524.52    14_408.09       0.0860             NaN         3.40
IVF-Binary-256-nl223-np14-rf0-random (query)          13_883.57       527.72    14_411.30       0.0851             NaN         3.40
IVF-Binary-256-nl223-np21-rf0-random (query)          13_883.57       532.41    14_415.99       0.0843             NaN         3.40
IVF-Binary-256-nl223-np11-rf10-random (query)         13_883.57       659.26    14_542.83       0.2135          1.0570         3.40
IVF-Binary-256-nl223-np11-rf20-random (query)         13_883.57       797.76    14_681.34       0.3241          1.0370         3.40
IVF-Binary-256-nl223-np14-rf10-random (query)         13_883.57       667.82    14_551.40       0.2100          1.0579         3.40
IVF-Binary-256-nl223-np14-rf20-random (query)         13_883.57       797.29    14_680.87       0.3192          1.0377         3.40
IVF-Binary-256-nl223-np21-rf10-random (query)         13_883.57       668.91    14_552.48       0.2075          1.0586         3.40
IVF-Binary-256-nl223-np21-rf20-random (query)         13_883.57       827.64    14_711.21       0.3160          1.0381         3.40
IVF-Binary-256-nl223-random (self)                    13_883.57     2_118.33    16_001.90       0.2092          1.0578         3.40
IVF-Binary-256-nl316-np15-rf0-random (query)          14_566.48       555.88    15_122.36       0.0853             NaN         3.76
IVF-Binary-256-nl316-np17-rf0-random (query)          14_566.48       562.84    15_129.32       0.0849             NaN         3.76
IVF-Binary-256-nl316-np25-rf0-random (query)          14_566.48       566.21    15_132.69       0.0845             NaN         3.76
IVF-Binary-256-nl316-np15-rf10-random (query)         14_566.48       703.64    15_270.12       0.2120          1.0574         3.76
IVF-Binary-256-nl316-np15-rf20-random (query)         14_566.48       841.27    15_407.75       0.3218          1.0373         3.76
IVF-Binary-256-nl316-np17-rf10-random (query)         14_566.48       695.92    15_262.40       0.2103          1.0579         3.76
IVF-Binary-256-nl316-np17-rf20-random (query)         14_566.48       826.31    15_392.79       0.3193          1.0376         3.76
IVF-Binary-256-nl316-np25-rf10-random (query)         14_566.48       698.88    15_265.36       0.2083          1.0584         3.76
IVF-Binary-256-nl316-np25-rf20-random (query)         14_566.48       859.41    15_425.89       0.3164          1.0380         3.76
IVF-Binary-256-nl316-random (self)                    14_566.48     2_241.91    16_808.39       0.2093          1.0578         3.76
IVF-Binary-256-nl158-np7-rf0-pca (query)              20_915.97       512.42    21_428.40       0.1677             NaN         3.15
IVF-Binary-256-nl158-np12-rf0-pca (query)             20_915.97       514.92    21_430.89       0.1675             NaN         3.15
IVF-Binary-256-nl158-np17-rf0-pca (query)             20_915.97       522.16    21_438.13       0.1670             NaN         3.15
IVF-Binary-256-nl158-np7-rf10-pca (query)             20_915.97       651.58    21_567.56       0.4931          1.0207         3.15
IVF-Binary-256-nl158-np7-rf20-pca (query)             20_915.97       784.72    21_700.69       0.6468          1.0113         3.15
IVF-Binary-256-nl158-np12-rf10-pca (query)            20_915.97       657.39    21_573.37       0.4949          1.0205         3.15
IVF-Binary-256-nl158-np12-rf20-pca (query)            20_915.97       840.84    21_756.81       0.6508          1.0111         3.15
IVF-Binary-256-nl158-np17-rf10-pca (query)            20_915.97       667.88    21_583.85       0.4906          1.0208         3.15
IVF-Binary-256-nl158-np17-rf20-pca (query)            20_915.97       889.89    21_805.87       0.6442          1.0113         3.15
IVF-Binary-256-nl158-pca (self)                       20_915.97     2_138.10    23_054.07       0.4943          1.0205         3.15
IVF-Binary-256-nl223-np11-rf0-pca (query)             15_820.73       532.33    16_353.06       0.1679             NaN         3.40
IVF-Binary-256-nl223-np14-rf0-pca (query)             15_820.73       534.45    16_355.18       0.1674             NaN         3.40
IVF-Binary-256-nl223-np21-rf0-pca (query)             15_820.73       542.28    16_363.01       0.1669             NaN         3.40
IVF-Binary-256-nl223-np11-rf10-pca (query)            15_820.73       678.20    16_498.92       0.4988          1.0202         3.40
IVF-Binary-256-nl223-np11-rf20-pca (query)            15_820.73       811.75    16_632.48       0.6593          1.0107         3.40
IVF-Binary-256-nl223-np14-rf10-pca (query)            15_820.73       675.67    16_496.39       0.4968          1.0204         3.40
IVF-Binary-256-nl223-np14-rf20-pca (query)            15_820.73       831.07    16_651.80       0.6564          1.0108         3.40
IVF-Binary-256-nl223-np21-rf10-pca (query)            15_820.73       712.89    16_533.62       0.4926          1.0207         3.40
IVF-Binary-256-nl223-np21-rf20-pca (query)            15_820.73       840.44    16_661.17       0.6488          1.0111         3.40
IVF-Binary-256-nl223-pca (self)                       15_820.73     2_186.47    18_007.20       0.4967          1.0203         3.40
IVF-Binary-256-nl316-np15-rf0-pca (query)             16_457.35       598.07    17_055.42       0.1681             NaN         3.77
IVF-Binary-256-nl316-np17-rf0-pca (query)             16_457.35       595.91    17_053.27       0.1678             NaN         3.77
IVF-Binary-256-nl316-np25-rf0-pca (query)             16_457.35       599.14    17_056.50       0.1673             NaN         3.77
IVF-Binary-256-nl316-np15-rf10-pca (query)            16_457.35       708.59    17_165.94       0.5001          1.0201         3.77
IVF-Binary-256-nl316-np15-rf20-pca (query)            16_457.35       876.57    17_333.92       0.6619          1.0106         3.77
IVF-Binary-256-nl316-np17-rf10-pca (query)            16_457.35       706.46    17_163.81       0.4989          1.0202         3.77
IVF-Binary-256-nl316-np17-rf20-pca (query)            16_457.35       882.14    17_339.50       0.6601          1.0106         3.77
IVF-Binary-256-nl316-np25-rf10-pca (query)            16_457.35       722.20    17_179.56       0.4942          1.0205         3.77
IVF-Binary-256-nl316-np25-rf20-pca (query)            16_457.35       862.36    17_319.71       0.6525          1.0110         3.77
IVF-Binary-256-nl316-pca (self)                       16_457.35     2_299.24    18_756.59       0.4987          1.0202         3.77
IVF-Binary-512-nl158-np7-rf0-random (query)           31_381.45       940.60    32_322.05       0.0937             NaN         5.67
IVF-Binary-512-nl158-np12-rf0-random (query)          31_381.45       949.19    32_330.64       0.0930             NaN         5.67
IVF-Binary-512-nl158-np17-rf0-random (query)          31_381.45       970.65    32_352.10       0.0924             NaN         5.67
IVF-Binary-512-nl158-np7-rf10-random (query)          31_381.45     1_065.28    32_446.73       0.2508          1.0494         5.67
IVF-Binary-512-nl158-np7-rf20-random (query)          31_381.45     1_200.72    32_582.17       0.3737          1.0313         5.67
IVF-Binary-512-nl158-np12-rf10-random (query)         31_381.45     1_099.19    32_480.64       0.2470          1.0501         5.67
IVF-Binary-512-nl158-np12-rf20-random (query)         31_381.45     1_217.57    32_599.02       0.3689          1.0318         5.67
IVF-Binary-512-nl158-np17-rf10-random (query)         31_381.45     1_100.08    32_481.52       0.2448          1.0505         5.67
IVF-Binary-512-nl158-np17-rf20-random (query)         31_381.45     1_247.53    32_628.98       0.3654          1.0322         5.67
IVF-Binary-512-nl158-random (self)                    31_381.45     3_515.54    34_896.99       0.2475          1.0499         5.67
IVF-Binary-512-nl223-np11-rf0-random (query)          25_904.22       975.81    26_880.03       0.0937             NaN         5.92
IVF-Binary-512-nl223-np14-rf0-random (query)          25_904.22       966.36    26_870.57       0.0931             NaN         5.92
IVF-Binary-512-nl223-np21-rf0-random (query)          25_904.22       975.28    26_879.50       0.0927             NaN         5.92
IVF-Binary-512-nl223-np11-rf10-random (query)         25_904.22     1_094.12    26_998.33       0.2494          1.0496         5.92
IVF-Binary-512-nl223-np11-rf20-random (query)         25_904.22     1_232.62    27_136.84       0.3728          1.0313         5.92
IVF-Binary-512-nl223-np14-rf10-random (query)         25_904.22     1_102.45    27_006.67       0.2462          1.0502         5.92
IVF-Binary-512-nl223-np14-rf20-random (query)         25_904.22     1_230.59    27_134.81       0.3684          1.0318         5.92
IVF-Binary-512-nl223-np21-rf10-random (query)         25_904.22     1_149.97    27_054.19       0.2446          1.0505         5.92
IVF-Binary-512-nl223-np21-rf20-random (query)         25_904.22     1_253.07    27_157.29       0.3662          1.0321         5.92
IVF-Binary-512-nl223-random (self)                    25_904.22     3_597.57    29_501.78       0.2469          1.0500         5.92
IVF-Binary-512-nl316-np15-rf0-random (query)          26_433.65     1_008.70    27_442.34       0.0933             NaN         6.29
IVF-Binary-512-nl316-np17-rf0-random (query)          26_433.65     1_005.06    27_438.70       0.0930             NaN         6.29
IVF-Binary-512-nl316-np25-rf0-random (query)          26_433.65     1_013.76    27_447.40       0.0927             NaN         6.29
IVF-Binary-512-nl316-np15-rf10-random (query)         26_433.65     1_140.17    27_573.82       0.2484          1.0497         6.29
IVF-Binary-512-nl316-np15-rf20-random (query)         26_433.65     1_269.03    27_702.68       0.3714          1.0315         6.29
IVF-Binary-512-nl316-np17-rf10-random (query)         26_433.65     1_130.83    27_564.48       0.2466          1.0501         6.29
IVF-Binary-512-nl316-np17-rf20-random (query)         26_433.65     1_269.90    27_703.55       0.3692          1.0318         6.29
IVF-Binary-512-nl316-np25-rf10-random (query)         26_433.65     1_140.71    27_574.35       0.2450          1.0504         6.29
IVF-Binary-512-nl316-np25-rf20-random (query)         26_433.65     1_302.53    27_736.17       0.3672          1.0321         6.29
IVF-Binary-512-nl316-random (self)                    26_433.65     3_688.43    30_122.07       0.2475          1.0499         6.29
IVF-Binary-512-nl158-np7-rf0-pca (query)              32_634.15       949.99    33_584.14       0.1894             NaN         5.67
IVF-Binary-512-nl158-np12-rf0-pca (query)             32_634.15       960.93    33_595.08       0.1890             NaN         5.67
IVF-Binary-512-nl158-np17-rf0-pca (query)             32_634.15       974.37    33_608.53       0.1879             NaN         5.67
IVF-Binary-512-nl158-np7-rf10-pca (query)             32_634.15     1_089.71    33_723.86       0.5413          1.0172         5.67
IVF-Binary-512-nl158-np7-rf20-pca (query)             32_634.15     1_213.64    33_847.79       0.6899          1.0093         5.67
IVF-Binary-512-nl158-np12-rf10-pca (query)            32_634.15     1_095.00    33_729.15       0.5401          1.0172         5.67
IVF-Binary-512-nl158-np12-rf20-pca (query)            32_634.15     1_236.59    33_870.74       0.6907          1.0091         5.67
IVF-Binary-512-nl158-np17-rf10-pca (query)            32_634.15     1_107.53    33_741.68       0.5311          1.0178         5.67
IVF-Binary-512-nl158-np17-rf20-pca (query)            32_634.15     1_258.67    33_892.83       0.6781          1.0096         5.67
IVF-Binary-512-nl158-pca (self)                       32_634.15     3_590.51    36_224.66       0.5401          1.0171         5.67
IVF-Binary-512-nl223-np11-rf0-pca (query)             27_418.53       977.46    28_395.99       0.1897             NaN         5.93
IVF-Binary-512-nl223-np14-rf0-pca (query)             27_418.53       984.96    28_403.49       0.1891             NaN         5.93
IVF-Binary-512-nl223-np21-rf0-pca (query)             27_418.53       996.29    28_414.82       0.1881             NaN         5.93
IVF-Binary-512-nl223-np11-rf10-pca (query)            27_418.53     1_116.94    28_535.46       0.5512          1.0165         5.93
IVF-Binary-512-nl223-np11-rf20-pca (query)            27_418.53     1_252.03    28_670.56       0.7062          1.0084         5.93
IVF-Binary-512-nl223-np14-rf10-pca (query)            27_418.53     1_148.43    28_566.96       0.5470          1.0167         5.93
IVF-Binary-512-nl223-np14-rf20-pca (query)            27_418.53     1_281.55    28_700.08       0.7024          1.0086         5.93
IVF-Binary-512-nl223-np21-rf10-pca (query)            27_418.53     1_130.08    28_548.61       0.5373          1.0173         5.93
IVF-Binary-512-nl223-np21-rf20-pca (query)            27_418.53     1_269.81    28_688.34       0.6885          1.0091         5.93
IVF-Binary-512-nl223-pca (self)                       27_418.53     3_648.51    31_067.04       0.5468          1.0167         5.93
IVF-Binary-512-nl316-np15-rf0-pca (query)             28_263.34     1_015.40    29_278.73       0.1901             NaN         6.29
IVF-Binary-512-nl316-np17-rf0-pca (query)             28_263.34     1_021.68    29_285.02       0.1896             NaN         6.29
IVF-Binary-512-nl316-np25-rf0-pca (query)             28_263.34     1_022.94    29_286.27       0.1885             NaN         6.29
IVF-Binary-512-nl316-np15-rf10-pca (query)            28_263.34     1_153.36    29_416.70       0.5534          1.0163         6.29
IVF-Binary-512-nl316-np15-rf20-pca (query)            28_263.34     1_275.40    29_538.73       0.7106          1.0083         6.29
IVF-Binary-512-nl316-np17-rf10-pca (query)            28_263.34     1_153.14    29_416.47       0.5511          1.0165         6.29
IVF-Binary-512-nl316-np17-rf20-pca (query)            28_263.34     1_308.74    29_572.07       0.7081          1.0083         6.29
IVF-Binary-512-nl316-np25-rf10-pca (query)            28_263.34     1_155.45    29_418.79       0.5411          1.0171         6.29
IVF-Binary-512-nl316-np25-rf20-pca (query)            28_263.34     1_308.36    29_571.69       0.6938          1.0089         6.29
IVF-Binary-512-nl316-pca (self)                       28_263.34     3_761.65    32_024.99       0.5508          1.0164         6.29
IVF-Binary-1024-nl158-np7-rf0-random (query)          54_782.48     1_818.51    56_600.99       0.1092             NaN        10.72
IVF-Binary-1024-nl158-np12-rf0-random (query)         54_782.48     1_832.07    56_614.55       0.1083             NaN        10.72
IVF-Binary-1024-nl158-np17-rf0-random (query)         54_782.48     1_870.82    56_653.30       0.1079             NaN        10.72
IVF-Binary-1024-nl158-np7-rf10-random (query)         54_782.48     1_932.32    56_714.80       0.3132          1.0396        10.72
IVF-Binary-1024-nl158-np7-rf20-random (query)         54_782.48     2_092.81    56_875.29       0.4522          1.0242        10.72
IVF-Binary-1024-nl158-np12-rf10-random (query)        54_782.48     1_985.04    56_767.53       0.3114          1.0399        10.72
IVF-Binary-1024-nl158-np12-rf20-random (query)        54_782.48     2_086.76    56_869.25       0.4521          1.0242        10.72
IVF-Binary-1024-nl158-np17-rf10-random (query)        54_782.48     1_982.94    56_765.43       0.3099          1.0401        10.72
IVF-Binary-1024-nl158-np17-rf20-random (query)        54_782.48     2_141.66    56_924.14       0.4503          1.0243        10.72
IVF-Binary-1024-nl158-random (self)                   54_782.48     6_446.37    61_228.85       0.3115          1.0398        10.72
IVF-Binary-1024-nl223-np11-rf0-random (query)         49_581.12     1_839.83    51_420.95       0.1092             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-random (query)         49_581.12     1_834.74    51_415.86       0.1085             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-random (query)         49_581.12     1_853.47    51_434.59       0.1081             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-random (query)        49_581.12     1_966.09    51_547.21       0.3144          1.0394        10.98
IVF-Binary-1024-nl223-np11-rf20-random (query)        49_581.12     2_092.61    51_673.73       0.4556          1.0238        10.98
IVF-Binary-1024-nl223-np14-rf10-random (query)        49_581.12     1_952.15    51_533.27       0.3119          1.0398        10.98
IVF-Binary-1024-nl223-np14-rf20-random (query)        49_581.12     2_093.26    51_674.38       0.4520          1.0242        10.98
IVF-Binary-1024-nl223-np21-rf10-random (query)        49_581.12     1_969.70    51_550.82       0.3107          1.0400        10.98
IVF-Binary-1024-nl223-np21-rf20-random (query)        49_581.12     2_121.76    51_702.88       0.4505          1.0243        10.98
IVF-Binary-1024-nl223-random (self)                   49_581.12     6_447.47    56_028.59       0.3117          1.0397        10.98
IVF-Binary-1024-nl316-np15-rf0-random (query)         50_001.63     1_881.38    51_883.01       0.1089             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-random (query)         50_001.63     1_877.12    51_878.75       0.1086             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-random (query)         50_001.63     1_888.78    51_890.42       0.1082             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-random (query)        50_001.63     1_985.52    51_987.16       0.3131          1.0396        11.34
IVF-Binary-1024-nl316-np15-rf20-random (query)        50_001.63     2_120.77    52_122.40       0.4544          1.0239        11.34
IVF-Binary-1024-nl316-np17-rf10-random (query)        50_001.63     1_987.11    51_988.74       0.3118          1.0398        11.34
IVF-Binary-1024-nl316-np17-rf20-random (query)        50_001.63     2_127.32    52_128.95       0.4525          1.0241        11.34
IVF-Binary-1024-nl316-np25-rf10-random (query)        50_001.63     2_002.27    52_003.91       0.3105          1.0400        11.34
IVF-Binary-1024-nl316-np25-rf20-random (query)        50_001.63     2_148.68    52_150.31       0.4510          1.0243        11.34
IVF-Binary-1024-nl316-random (self)                   50_001.63     6_546.18    56_547.82       0.3120          1.0397        11.34
IVF-Binary-1024-nl158-np7-rf0-pca (query)             56_362.33     1_842.75    58_205.08       0.2305             NaN        10.73
IVF-Binary-1024-nl158-np12-rf0-pca (query)            56_362.33     1_868.81    58_231.15       0.2299             NaN        10.73
IVF-Binary-1024-nl158-np17-rf0-pca (query)            56_362.33     1_906.93    58_269.26       0.2279             NaN        10.73
IVF-Binary-1024-nl158-np7-rf10-pca (query)            56_362.33     1_951.31    58_313.64       0.6227          1.0123        10.73
IVF-Binary-1024-nl158-np7-rf20-pca (query)            56_362.33     2_100.48    58_462.81       0.7603          1.0064        10.73
IVF-Binary-1024-nl158-np12-rf10-pca (query)           56_362.33     1_992.15    58_354.48       0.6215          1.0122        10.73
IVF-Binary-1024-nl158-np12-rf20-pca (query)           56_362.33     2_247.06    58_609.39       0.7634          1.0061        10.73
IVF-Binary-1024-nl158-np17-rf10-pca (query)           56_362.33     2_033.66    58_396.00       0.6081          1.0129        10.73
IVF-Binary-1024-nl158-np17-rf20-pca (query)           56_362.33     2_169.11    58_531.44       0.7480          1.0066        10.73
IVF-Binary-1024-nl158-pca (self)                      56_362.33     6_522.54    62_884.88       0.6213          1.0122        10.73
IVF-Binary-1024-nl223-np11-rf0-pca (query)            51_043.88     1_879.95    52_923.83       0.2323             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-pca (query)            51_043.88     1_874.62    52_918.50       0.2311             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-pca (query)            51_043.88     1_900.64    52_944.52       0.2290             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-pca (query)           51_043.88     1_967.45    53_011.33       0.6380          1.0114        10.98
IVF-Binary-1024-nl223-np11-rf20-pca (query)           51_043.88     2_133.61    53_177.49       0.7854          1.0053        10.98
IVF-Binary-1024-nl223-np14-rf10-pca (query)           51_043.88     1_987.19    53_031.07       0.6338          1.0115        10.98
IVF-Binary-1024-nl223-np14-rf20-pca (query)           51_043.88     2_166.04    53_209.92       0.7819          1.0054        10.98
IVF-Binary-1024-nl223-np21-rf10-pca (query)           51_043.88     2_027.45    53_071.33       0.6192          1.0123        10.98
IVF-Binary-1024-nl223-np21-rf20-pca (query)           51_043.88     2_179.12    53_223.00       0.7626          1.0060        10.98
IVF-Binary-1024-nl223-pca (self)                      51_043.88     6_574.76    57_618.64       0.6337          1.0115        10.98
IVF-Binary-1024-nl316-np15-rf0-pca (query)            51_922.87     1_914.27    53_837.14       0.2324             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-pca (query)            51_922.87     2_075.99    53_998.86       0.2320             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-pca (query)            51_922.87     1_919.98    53_842.85       0.2301             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-pca (query)           51_922.87     2_023.35    53_946.23       0.6421          1.0112        11.34
IVF-Binary-1024-nl316-np15-rf20-pca (query)           51_922.87     2_161.56    54_084.43       0.7914          1.0051        11.34
IVF-Binary-1024-nl316-np17-rf10-pca (query)           51_922.87     2_029.14    53_952.01       0.6396          1.0113        11.34
IVF-Binary-1024-nl316-np17-rf20-pca (query)           51_922.87     2_174.69    54_097.56       0.7890          1.0051        11.34
IVF-Binary-1024-nl316-np25-rf10-pca (query)           51_922.87     2_045.42    53_968.29       0.6251          1.0120        11.34
IVF-Binary-1024-nl316-np25-rf20-pca (query)           51_922.87     2_211.56    54_134.43       0.7701          1.0058        11.34
IVF-Binary-1024-nl316-pca (self)                      51_922.87     6_724.41    58_647.28       0.6390          1.0112        11.34
IVF-Binary-1024-nl158-np7-rf0-signed (query)          55_160.40     1_820.62    56_981.02       0.1092             NaN        10.72
IVF-Binary-1024-nl158-np12-rf0-signed (query)         55_160.40     1_843.59    57_003.99       0.1083             NaN        10.72
IVF-Binary-1024-nl158-np17-rf0-signed (query)         55_160.40     1_859.96    57_020.36       0.1079             NaN        10.72
IVF-Binary-1024-nl158-np7-rf10-signed (query)         55_160.40     1_929.24    57_089.64       0.3132          1.0396        10.72
IVF-Binary-1024-nl158-np7-rf20-signed (query)         55_160.40     2_065.06    57_225.46       0.4522          1.0242        10.72
IVF-Binary-1024-nl158-np12-rf10-signed (query)        55_160.40     1_947.01    57_107.41       0.3114          1.0399        10.72
IVF-Binary-1024-nl158-np12-rf20-signed (query)        55_160.40     2_086.65    57_247.05       0.4521          1.0242        10.72
IVF-Binary-1024-nl158-np17-rf10-signed (query)        55_160.40     1_964.64    57_125.04       0.3099          1.0401        10.72
IVF-Binary-1024-nl158-np17-rf20-signed (query)        55_160.40     2_130.49    57_290.89       0.4503          1.0243        10.72
IVF-Binary-1024-nl158-signed (self)                   55_160.40     6_441.76    61_602.16       0.3115          1.0398        10.72
IVF-Binary-1024-nl223-np11-rf0-signed (query)         49_597.16     1_836.63    51_433.78       0.1092             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-signed (query)         49_597.16     1_853.98    51_451.14       0.1085             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-signed (query)         49_597.16     1_859.90    51_457.06       0.1081             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-signed (query)        49_597.16     1_965.13    51_562.29       0.3144          1.0394        10.98
IVF-Binary-1024-nl223-np11-rf20-signed (query)        49_597.16     2_096.74    51_693.90       0.4556          1.0238        10.98
IVF-Binary-1024-nl223-np14-rf10-signed (query)        49_597.16     1_950.01    51_547.17       0.3119          1.0398        10.98
IVF-Binary-1024-nl223-np14-rf20-signed (query)        49_597.16     2_097.36    51_694.52       0.4520          1.0242        10.98
IVF-Binary-1024-nl223-np21-rf10-signed (query)        49_597.16     1_985.39    51_582.55       0.3107          1.0400        10.98
IVF-Binary-1024-nl223-np21-rf20-signed (query)        49_597.16     2_116.98    51_714.13       0.4505          1.0243        10.98
IVF-Binary-1024-nl223-signed (self)                   49_597.16     6_426.33    56_023.48       0.3117          1.0397        10.98
IVF-Binary-1024-nl316-np15-rf0-signed (query)         50_102.69     1_879.66    51_982.35       0.1089             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-signed (query)         50_102.69     1_886.83    51_989.53       0.1086             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-signed (query)         50_102.69     1_901.46    52_004.16       0.1082             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-signed (query)        50_102.69     2_012.92    52_115.61       0.3131          1.0396        11.34
IVF-Binary-1024-nl316-np15-rf20-signed (query)        50_102.69     2_141.22    52_243.91       0.4544          1.0239        11.34
IVF-Binary-1024-nl316-np17-rf10-signed (query)        50_102.69     2_007.39    52_110.08       0.3118          1.0398        11.34
IVF-Binary-1024-nl316-np17-rf20-signed (query)        50_102.69     2_157.19    52_259.89       0.4525          1.0241        11.34
IVF-Binary-1024-nl316-np25-rf10-signed (query)        50_102.69     2_020.95    52_123.65       0.3105          1.0400        11.34
IVF-Binary-1024-nl316-np25-rf20-signed (query)        50_102.69     2_167.92    52_270.61       0.4510          1.0243        11.34
IVF-Binary-1024-nl316-signed (self)                   50_102.69     6_617.39    56_720.09       0.3120          1.0397        11.34
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
Exhaustive (query)                                        10.11     4_461.25     4_471.35       1.0000          1.0000        48.83
Exhaustive (self)                                         10.11    14_711.69    14_721.79       1.0000          1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_587.32       259.68     2_847.00       0.0796             NaN         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_587.32       359.92     2_947.24       0.2931          1.1776         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_587.32       466.83     3_054.15       0.4374          1.1037         1.78
ExhaustiveBinary-256-random (self)                     2_587.32     1_252.01     3_839.34       0.2929          1.1769         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_772.38       262.45     3_034.82       0.0827             NaN         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_772.38       378.74     3_151.12       0.1182          3.9214         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_772.38       494.46     3_266.84       0.1432          1.4338         1.78
ExhaustiveBinary-256-pca (self)                        2_772.38     1_247.47     4_019.85       0.1157          6.7403         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_096.74       461.41     5_558.15       0.1270             NaN         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_096.74       568.31     5_665.05       0.3949          1.1228         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_096.74       681.98     5_778.72       0.5575          1.0675         3.55
ExhaustiveBinary-512-random (self)                     5_096.74     1_893.64     6_990.38       0.3929          1.1237         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_394.54       470.28     5_864.82       0.0965             NaN         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_394.54       588.32     5_982.86       0.2973          1.1772         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_394.54       707.40     6_101.94       0.4534          1.0999         3.55
ExhaustiveBinary-512-pca (self)                        5_394.54     1_995.39     7_389.93       0.2988          1.1773         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_073.19       798.64    10_871.83       0.1773             NaN         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_073.19       902.57    10_975.76       0.5381          1.0724         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_073.19     1_031.54    11_104.73       0.7086          1.0349         7.10
ExhaustiveBinary-1024-random (self)                   10_073.19     2_991.90    13_065.08       0.5368          1.0726         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_552.30       815.72    11_368.02       0.1035             NaN         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_552.30       930.31    11_482.62       0.3158          1.1656         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_552.30     1_064.62    11_616.93       0.4741          1.0929         7.10
ExhaustiveBinary-1024-pca (self)                      10_552.30     3_096.40    13_648.71       0.3173          1.1657         7.10
ExhaustiveBinary-256-signed_no_rr (query)              2_578.07       259.45     2_837.52       0.0796             NaN         1.78
ExhaustiveBinary-256-signed-rf10 (query)               2_578.07       371.16     2_949.23       0.2931          1.1776         1.78
ExhaustiveBinary-256-signed-rf20 (query)               2_578.07       469.55     3_047.62       0.4374          1.1037         1.78
ExhaustiveBinary-256-signed (self)                     2_578.07     1_197.67     3_775.74       0.2929          1.1769         1.78
IVF-Binary-256-nl158-np7-rf0-random (query)            4_191.37       115.32     4_306.69       0.1023             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           4_191.37       122.86     4_314.23       0.0908             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           4_191.37       127.30     4_318.67       0.0812             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           4_191.37       174.73     4_366.10       0.3326          1.1542         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           4_191.37       226.86     4_418.23       0.4815          1.0896         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          4_191.37       184.11     4_375.48       0.3113          1.1675         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          4_191.37       236.05     4_427.42       0.4578          1.0975         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          4_191.37       191.82     4_383.19       0.2976          1.1763         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          4_191.37       252.87     4_444.24       0.4434          1.1027         1.93
IVF-Binary-256-nl158-random (self)                     4_191.37       543.09     4_734.47       0.3113          1.1669         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           3_066.97       124.27     3_191.24       0.0981             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           3_066.97       124.28     3_191.25       0.0871             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           3_066.97       128.33     3_195.30       0.0818             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          3_066.97       181.71     3_248.68       0.3258          1.1579         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          3_066.97       236.29     3_303.26       0.4750          1.0915         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          3_066.97       185.08     3_252.05       0.3093          1.1684         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          3_066.97       243.48     3_310.45       0.4555          1.0983         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          3_066.97       198.26     3_265.23       0.2997          1.1749         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          3_066.97       258.05     3_325.02       0.4446          1.1021         2.00
IVF-Binary-256-nl223-random (self)                     3_066.97       556.87     3_623.84       0.3094          1.1678         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           3_271.83       129.28     3_401.10       0.0902             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           3_271.83       127.86     3_399.69       0.0873             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           3_271.83       133.27     3_405.10       0.0830             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          3_271.83       196.99     3_468.82       0.3190          1.1618         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          3_271.83       242.65     3_514.47       0.4684          1.0938         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          3_271.83       188.09     3_459.92       0.3139          1.1649         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          3_271.83       244.30     3_516.12       0.4615          1.0961         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          3_271.83       201.89     3_473.71       0.3025          1.1726         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          3_271.83       256.23     3_528.06       0.4481          1.1006         2.09
IVF-Binary-256-nl316-random (self)                     3_271.83       570.65     3_842.47       0.3137          1.1645         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               4_338.16       123.44     4_461.59       0.0939             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              4_338.16       128.54     4_466.69       0.0914             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              4_338.16       134.18     4_472.33       0.0903             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              4_338.16       194.23     4_532.39       0.2815          1.1901         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              4_338.16       254.24     4_592.39       0.4291          1.1101         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             4_338.16       204.10     4_542.26       0.2368          1.2309         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             4_338.16       272.47     4_610.63       0.3498          1.1457         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             4_338.16       214.57     4_552.72       0.2144          1.2578         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             4_338.16       291.31     4_629.47       0.3095          1.1698         1.93
IVF-Binary-256-nl158-pca (self)                        4_338.16       644.85     4_983.00       0.2370          1.2324         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              3_264.90       125.69     3_390.58       0.0937             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              3_264.90       130.11     3_395.01       0.0921             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              3_264.90       135.75     3_400.65       0.0906             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             3_264.90       201.85     3_466.75       0.2741          1.1960         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             3_264.90       266.10     3_531.00       0.4164          1.1152         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             3_264.90       207.20     3_472.09       0.2535          1.2142         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             3_264.90       273.73     3_538.63       0.3798          1.1311         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             3_264.90       226.24     3_491.13       0.2220          1.2482         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             3_264.90       292.50     3_557.39       0.3231          1.1612         2.00
IVF-Binary-256-nl223-pca (self)                        3_264.90       640.54     3_905.44       0.2542          1.2150         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              3_468.03       136.60     3_604.63       0.0934             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              3_468.03       135.01     3_603.03       0.0927             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              3_468.03       139.92     3_607.94       0.0910             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             3_468.03       208.43     3_676.46       0.2767          1.1930         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             3_468.03       269.84     3_737.87       0.4210          1.1124         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             3_468.03       229.39     3_697.42       0.2653          1.2024         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             3_468.03       279.06     3_747.09       0.4009          1.1206         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             3_468.03       221.44     3_689.47       0.2333          1.2342         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             3_468.03       291.03     3_759.06       0.3440          1.1485         2.09
IVF-Binary-256-nl316-pca (self)                        3_468.03       672.73     4_140.75       0.2664          1.2029         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_637.35       214.09     6_851.44       0.1365             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_637.35       222.51     6_859.86       0.1314             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_637.35       233.17     6_870.52       0.1280             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_637.35       274.64     6_911.99       0.4194          1.1128         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_637.35       329.07     6_966.42       0.5824          1.0615         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_637.35       289.84     6_927.19       0.4052          1.1187         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_637.35       344.43     6_981.78       0.5669          1.0653         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_637.35       301.19     6_938.54       0.3969          1.1224         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_637.35       365.36     7_002.72       0.5596          1.0672         3.71
IVF-Binary-512-nl158-random (self)                     6_637.35       892.78     7_530.13       0.4039          1.1194         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_542.56       217.61     5_760.17       0.1345             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_542.56       231.59     5_774.15       0.1304             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_542.56       231.80     5_774.35       0.1281             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_542.56       283.14     5_825.70       0.4149          1.1145         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_542.56       336.65     5_879.21       0.5792          1.0622         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_542.56       288.40     5_830.96       0.4037          1.1192         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_542.56       350.83     5_893.39       0.5672          1.0653         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_542.56       302.39     5_844.95       0.3977          1.1218         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_542.56       367.68     5_910.23       0.5610          1.0669         3.77
IVF-Binary-512-nl223-random (self)                     5_542.56       909.33     6_451.89       0.4027          1.1198         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           5_735.20       226.23     5_961.43       0.1326             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           5_735.20       225.98     5_961.18       0.1312             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           5_735.20       233.64     5_968.84       0.1288             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          5_735.20       289.94     6_025.14       0.4112          1.1160         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          5_735.20       345.89     6_081.09       0.5748          1.0633         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          5_735.20       289.70     6_024.90       0.4073          1.1177         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          5_735.20       348.01     6_083.21       0.5696          1.0646         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          5_735.20       302.15     6_037.35       0.4001          1.1208         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          5_735.20       365.68     6_100.87       0.5618          1.0666         3.86
IVF-Binary-512-nl316-random (self)                     5_735.20       906.13     6_641.33       0.4059          1.1185         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               6_890.01       223.56     7_113.58       0.0978             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              6_890.01       231.30     7_121.31       0.0966             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              6_890.01       238.87     7_128.88       0.0965             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              6_890.01       294.52     7_184.53       0.3062          1.1716         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              6_890.01       353.20     7_243.21       0.4660          1.0957         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             6_890.01       304.88     7_194.89       0.2974          1.1771         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             6_890.01       384.74     7_274.76       0.4534          1.0999         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             6_890.01       326.77     7_216.78       0.2974          1.1771         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             6_890.01       388.60     7_278.61       0.4534          1.0999         3.71
IVF-Binary-512-nl158-pca (self)                        6_890.01       958.60     7_848.61       0.2989          1.1772         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              5_841.04       225.87     6_066.91       0.0973             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              5_841.04       231.27     6_072.32       0.0966             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              5_841.04       247.37     6_088.41       0.0965             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             5_841.04       300.24     6_141.29       0.3028          1.1736         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             5_841.04       361.67     6_202.71       0.4616          1.0971         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             5_841.04       307.48     6_148.52       0.2978          1.1769         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             5_841.04       371.50     6_212.54       0.4540          1.0997         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             5_841.04       322.85     6_163.90       0.2974          1.1771         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             5_841.04       389.79     6_230.83       0.4535          1.0999         3.77
IVF-Binary-512-nl223-pca (self)                        5_841.04       978.26     6_819.31       0.2991          1.1770         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              6_030.88       233.67     6_264.55       0.0971             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              6_030.88       244.16     6_275.04       0.0967             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              6_030.88       247.40     6_278.27       0.0965             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             6_030.88       307.61     6_338.49       0.3015          1.1745         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             6_030.88       369.35     6_400.22       0.4592          1.0979         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             6_030.88       312.02     6_342.89       0.2987          1.1763         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             6_030.88       371.92     6_402.80       0.4552          1.0993         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             6_030.88       321.07     6_351.95       0.2975          1.1771         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             6_030.88       403.19     6_434.07       0.4535          1.0999         3.86
IVF-Binary-512-nl316-pca (self)                        6_030.88     1_224.89     7_255.76       0.3000          1.1764         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_803.42       410.47    12_213.89       0.1846             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_803.42       420.45    12_223.87       0.1806             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_803.42       436.72    12_240.14       0.1781             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_803.42       483.69    12_287.11       0.5530          1.0684         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_803.42       542.38    12_345.79       0.7207          1.0329         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_803.42       492.76    12_296.17       0.5441          1.0708         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_803.42       561.84    12_365.26       0.7125          1.0343         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_803.42       515.01    12_318.43       0.5400          1.0719         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_803.42       583.26    12_386.68       0.7094          1.0348         7.26
IVF-Binary-1024-nl158-random (self)                   11_803.42     1_570.48    13_373.89       0.5426          1.0711         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_499.94       412.41    10_912.36       0.1829             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_499.94       420.79    10_920.74       0.1800             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_499.94       444.02    10_943.97       0.1782             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_499.94       482.14    10_982.08       0.5508          1.0688         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_499.94       539.62    11_039.57       0.7205          1.0329         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_499.94       494.83    10_994.77       0.5438          1.0709         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_499.94       564.27    11_064.21       0.7139          1.0340         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_499.94       513.39    11_013.34       0.5403          1.0719         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_499.94       581.86    11_081.80       0.7110          1.0345         7.32
IVF-Binary-1024-nl223-random (self)                   10_499.94     1_675.44    12_175.38       0.5425          1.0711         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         10_681.30       432.31    11_113.61       0.1812             NaN         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)         10_681.30       422.17    11_103.47       0.1802             NaN         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)         10_681.30       434.84    11_116.14       0.1784             NaN         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)        10_681.30       489.92    11_171.22       0.5480          1.0697         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)        10_681.30       568.25    11_249.55       0.7182          1.0333         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)        10_681.30       514.51    11_195.81       0.5453          1.0705         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)        10_681.30       560.69    11_241.99       0.7151          1.0338         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)        10_681.30       514.28    11_195.58       0.5407          1.0718         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)        10_681.30       574.64    11_255.94       0.7111          1.0345         7.41
IVF-Binary-1024-nl316-random (self)                   10_681.30     1_572.95    12_254.25       0.5443          1.0706         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)             12_067.19       422.51    12_489.70       0.1049             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            12_067.19       442.96    12_510.16       0.1036             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            12_067.19       455.02    12_522.22       0.1035             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            12_067.19       506.86    12_574.06       0.3243          1.1605         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            12_067.19       567.40    12_634.60       0.4867          1.0889         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           12_067.19       515.21    12_582.41       0.3156          1.1656         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           12_067.19       598.75    12_665.94       0.4742          1.0929         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           12_067.19       538.58    12_605.78       0.3156          1.1656         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           12_067.19       607.40    12_674.60       0.4742          1.0929         7.26
IVF-Binary-1024-nl158-pca (self)                      12_067.19     1_675.32    13_742.51       0.3173          1.1657         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            11_015.22       432.76    11_447.98       0.1044             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            11_015.22       446.36    11_461.58       0.1036             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            11_015.22       453.01    11_468.22       0.1035             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           11_015.22       515.87    11_531.09       0.3211          1.1623         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           11_015.22       570.21    11_585.43       0.4821          1.0904         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           11_015.22       515.54    11_530.76       0.3161          1.1653         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           11_015.22       599.62    11_614.84       0.4747          1.0927         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           11_015.22       552.38    11_567.60       0.3158          1.1656         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           11_015.22       606.53    11_621.75       0.4741          1.0929         7.32
IVF-Binary-1024-nl223-pca (self)                      11_015.22     1_661.30    12_676.52       0.3176          1.1655         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            11_271.22       439.82    11_711.05       0.1041             NaN         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            11_271.22       448.53    11_719.75       0.1037             NaN         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            11_271.22       457.99    11_729.21       0.1035             NaN         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           11_271.22       513.13    11_784.35       0.3198          1.1632         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           11_271.22       582.33    11_853.55       0.4799          1.0911         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           11_271.22       514.13    11_785.35       0.3170          1.1648         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           11_271.22       583.96    11_855.19       0.4759          1.0923         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           11_271.22       542.18    11_813.41       0.3159          1.1656         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           11_271.22       607.28    11_878.50       0.4741          1.0929         7.42
IVF-Binary-1024-nl316-pca (self)                      11_271.22     1_663.24    12_934.47       0.3185          1.1649         7.42
IVF-Binary-256-nl158-np7-rf0-signed (query)            4_121.40       118.26     4_239.66       0.1023             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-signed (query)           4_121.40       120.48     4_241.88       0.0908             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-signed (query)           4_121.40       125.06     4_246.46       0.0812             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-signed (query)           4_121.40       186.09     4_307.49       0.3326          1.1542         1.93
IVF-Binary-256-nl158-np7-rf20-signed (query)           4_121.40       225.42     4_346.82       0.4815          1.0896         1.93
IVF-Binary-256-nl158-np12-rf10-signed (query)          4_121.40       185.39     4_306.79       0.3113          1.1675         1.93
IVF-Binary-256-nl158-np12-rf20-signed (query)          4_121.40       235.55     4_356.95       0.4578          1.0975         1.93
IVF-Binary-256-nl158-np17-rf10-signed (query)          4_121.40       192.06     4_313.46       0.2976          1.1763         1.93
IVF-Binary-256-nl158-np17-rf20-signed (query)          4_121.40       250.11     4_371.51       0.4434          1.1027         1.93
IVF-Binary-256-nl158-signed (self)                     4_121.40       542.06     4_663.46       0.3113          1.1669         1.93
IVF-Binary-256-nl223-np11-rf0-signed (query)           3_060.05       120.71     3_180.76       0.0981             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-signed (query)           3_060.05       123.24     3_183.29       0.0871             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-signed (query)           3_060.05       129.90     3_189.95       0.0818             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-signed (query)          3_060.05       180.38     3_240.44       0.3258          1.1579         2.00
IVF-Binary-256-nl223-np11-rf20-signed (query)          3_060.05       236.30     3_296.35       0.4750          1.0915         2.00
IVF-Binary-256-nl223-np14-rf10-signed (query)          3_060.05       187.08     3_247.13       0.3093          1.1684         2.00
IVF-Binary-256-nl223-np14-rf20-signed (query)          3_060.05       247.77     3_307.83       0.4555          1.0983         2.00
IVF-Binary-256-nl223-np21-rf10-signed (query)          3_060.05       194.61     3_254.67       0.2997          1.1749         2.00
IVF-Binary-256-nl223-np21-rf20-signed (query)          3_060.05       258.70     3_318.76       0.4446          1.1021         2.00
IVF-Binary-256-nl223-signed (self)                     3_060.05       558.55     3_618.60       0.3094          1.1678         2.00
IVF-Binary-256-nl316-np15-rf0-signed (query)           3_252.93       126.58     3_379.51       0.0902             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-signed (query)           3_252.93       127.47     3_380.40       0.0873             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-signed (query)           3_252.93       134.67     3_387.60       0.0830             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-signed (query)          3_252.93       188.06     3_440.99       0.3190          1.1618         2.09
IVF-Binary-256-nl316-np15-rf20-signed (query)          3_252.93       240.51     3_493.44       0.4684          1.0938         2.09
IVF-Binary-256-nl316-np17-rf10-signed (query)          3_252.93       188.16     3_441.08       0.3139          1.1649         2.09
IVF-Binary-256-nl316-np17-rf20-signed (query)          3_252.93       243.99     3_496.92       0.4615          1.0961         2.09
IVF-Binary-256-nl316-np25-rf10-signed (query)          3_252.93       208.34     3_461.27       0.3025          1.1726         2.09
IVF-Binary-256-nl316-np25-rf20-signed (query)          3_252.93       266.28     3_519.21       0.4481          1.1006         2.09
IVF-Binary-256-nl316-signed (self)                     3_252.93       590.28     3_843.21       0.3137          1.1645         2.09
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
Exhaustive (query)                                        20.38    10_042.14    10_062.52       1.0000          1.0000        97.66
Exhaustive (self)                                         20.38    33_803.57    33_823.95       1.0000          1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_682.60       378.44     6_061.04       0.0472             NaN         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_682.60       487.62     6_170.22       0.2012          1.1663         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_682.60       629.64     6_312.24       0.3224          1.1017         2.03
ExhaustiveBinary-256-random (self)                     5_682.60     1_637.14     7_319.74       0.2020          1.1654         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_124.31       376.81     6_501.11       0.1539             NaN         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_124.31       525.74     6_650.05       0.3795          1.0909         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_124.31       664.17     6_788.47       0.4855          1.0578         2.03
ExhaustiveBinary-256-pca (self)                        6_124.31     1_708.90     7_833.21       0.3796          1.0913         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_122.79       687.86    11_810.65       0.0860             NaN         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_122.79       817.80    11_940.59       0.2648          1.1242         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_122.79       965.59    12_088.38       0.3980          1.0752         4.05
ExhaustiveBinary-512-random (self)                    11_122.79     2_696.21    13_819.00       0.2643          1.1240         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_689.92       698.43    12_388.35       0.1171             NaN         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_689.92       843.10    12_533.03       0.2880         21.4934         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_689.92       992.88    12_682.81       0.3835          1.0941         4.05
ExhaustiveBinary-512-pca (self)                       11_689.92     2_788.83    14_478.75       0.2876         22.0894         4.05
ExhaustiveBinary-1024-random_no_rr (query)            22_032.36     1_226.83    23_259.19       0.1148             NaN         8.10
ExhaustiveBinary-1024-random-rf10 (query)             22_032.36     1_377.40    23_409.76       0.3455          1.0925         8.10
ExhaustiveBinary-1024-random-rf20 (query)             22_032.36     1_529.04    23_561.40       0.4981          1.0533         8.10
ExhaustiveBinary-1024-random (self)                   22_032.36     4_845.05    26_877.41       0.3445          1.0928         8.10
ExhaustiveBinary-1024-pca_no_rr (query)               23_044.03     1_253.53    24_297.56       0.2416             NaN         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                23_044.03     1_414.35    24_458.39       0.6860          1.0240         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                23_044.03     1_579.99    24_624.02       0.8390          1.0095         8.11
ExhaustiveBinary-1024-pca (self)                      23_044.03     4_672.35    27_716.38       0.6864          1.0239         8.11
ExhaustiveBinary-512-signed_no_rr (query)             11_105.14       683.34    11_788.48       0.0860             NaN         4.05
ExhaustiveBinary-512-signed-rf10 (query)              11_105.14       809.33    11_914.47       0.2648          1.1242         4.05
ExhaustiveBinary-512-signed-rf20 (query)              11_105.14       948.57    12_053.71       0.3980          1.0752         4.05
ExhaustiveBinary-512-signed (self)                    11_105.14     2_677.46    13_782.60       0.2643          1.1240         4.05
IVF-Binary-256-nl158-np7-rf0-random (query)            8_934.06       234.08     9_168.14       0.0675             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           8_934.06       244.84     9_178.90       0.0586             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           8_934.06       246.07     9_180.14       0.0493             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           8_934.06       326.80     9_260.87       0.2432          1.1377         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           8_934.06       406.49     9_340.55       0.3698          1.0844         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          8_934.06       330.39     9_264.46       0.2255          1.1497         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          8_934.06       420.65     9_354.71       0.3483          1.0921         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          8_934.06       336.64     9_270.70       0.2074          1.1635         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          8_934.06       431.43     9_365.49       0.3302          1.0996         2.34
IVF-Binary-256-nl158-random (self)                     8_934.06       996.51     9_930.57       0.2261          1.1490         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           6_530.50       252.43     6_782.93       0.0642             NaN         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)           6_530.50       249.21     6_779.70       0.0545             NaN         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)           6_530.50       254.10     6_784.60       0.0498             NaN         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)          6_530.50       341.46     6_871.95       0.2334          1.1444         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)          6_530.50       416.97     6_947.47       0.3573          1.0890         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)          6_530.50       335.94     6_866.43       0.2172          1.1566         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)          6_530.50       425.71     6_956.21       0.3403          1.0960         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)          6_530.50       350.40     6_880.90       0.2092          1.1621         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)          6_530.50       435.00     6_965.50       0.3311          1.0992         2.46
IVF-Binary-256-nl223-random (self)                     6_530.50     1_036.53     7_567.02       0.2181          1.1556         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)           6_873.49       261.45     7_134.93       0.0584             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           6_873.49       262.73     7_136.22       0.0547             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           6_873.49       276.23     7_149.72       0.0517             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          6_873.49       346.81     7_220.30       0.2294          1.1477         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          6_873.49       432.02     7_305.51       0.3567          1.0896         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          6_873.49       348.97     7_222.46       0.2238          1.1515         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          6_873.49       435.35     7_308.83       0.3497          1.0919         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          6_873.49       352.04     7_225.53       0.2156          1.1575         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          6_873.49       446.08     7_319.56       0.3381          1.0961         2.65
IVF-Binary-256-nl316-random (self)                     6_873.49     1_076.55     7_950.03       0.2241          1.1510         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               9_265.11       247.59     9_512.69       0.2085             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              9_265.11       257.05     9_522.15       0.2050             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              9_265.11       255.15     9_520.26       0.2025             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              9_265.11       347.57     9_612.68       0.6076          1.0348         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              9_265.11       435.42     9_700.53       0.7667          1.0158         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             9_265.11       356.21     9_621.32       0.5909          1.0374         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             9_265.11       452.29     9_717.40       0.7466          1.0178         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             9_265.11       363.15     9_628.25       0.5763          1.0398         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             9_265.11       467.75     9_732.85       0.7278          1.0197         2.34
IVF-Binary-256-nl158-pca (self)                        9_265.11     1_168.16    10_433.27       0.5917          1.0371         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              6_925.42       255.36     7_180.77       0.2082             NaN         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              6_925.42       259.27     7_184.69       0.2065             NaN         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              6_925.42       263.79     7_189.21       0.2030             NaN         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             6_925.42       363.80     7_289.21       0.6061          1.0350         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             6_925.42       454.99     7_380.41       0.7661          1.0159         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             6_925.42       360.44     7_285.86       0.5976          1.0363         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             6_925.42       461.23     7_386.65       0.7555          1.0169         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             6_925.42       373.87     7_299.29       0.5790          1.0393         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             6_925.42       472.16     7_397.58       0.7325          1.0192         2.47
IVF-Binary-256-nl223-pca (self)                        6_925.42     1_137.11     8_062.52       0.5986          1.0361         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              7_347.77       273.53     7_621.30       0.2082             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              7_347.77       271.91     7_619.68       0.2074             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              7_347.77       280.15     7_627.93       0.2045             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             7_347.77       372.83     7_720.61       0.6077          1.0348         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             7_347.77       464.24     7_812.02       0.7686          1.0157         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             7_347.77       373.02     7_720.79       0.6033          1.0355         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             7_347.77       468.49     7_816.27       0.7629          1.0162         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             7_347.77       381.21     7_728.98       0.5873          1.0380         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             7_347.77       481.12     7_828.90       0.7424          1.0182         2.65
IVF-Binary-256-nl316-pca (self)                        7_347.77     1_192.92     8_540.70       0.6043          1.0352         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           14_390.93       451.64    14_842.57       0.0977             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          14_390.93       464.27    14_855.20       0.0934             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          14_390.93       463.63    14_854.56       0.0872             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          14_390.93       533.36    14_924.29       0.2879          1.1143         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          14_390.93       615.06    15_005.99       0.4248          1.0690         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         14_390.93       542.79    14_933.73       0.2772          1.1189         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         14_390.93       630.15    15_021.09       0.4113          1.0721         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         14_390.93       556.98    14_947.91       0.2686          1.1230         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         14_390.93       648.35    15_039.29       0.4024          1.0744         4.36
IVF-Binary-512-nl158-random (self)                    14_390.93     1_729.06    16_120.00       0.2766          1.1188         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          12_049.90       458.04    12_507.94       0.0946             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          12_049.90       466.26    12_516.16       0.0906             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          12_049.90       475.79    12_525.69       0.0882             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         12_049.90       550.12    12_600.02       0.2815          1.1166         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         12_049.90       631.16    12_681.06       0.4178          1.0706         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         12_049.90       554.68    12_604.58       0.2738          1.1202         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         12_049.90       638.04    12_687.94       0.4088          1.0728         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         12_049.90       573.99    12_623.89       0.2690          1.1224         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         12_049.90       657.63    12_707.53       0.4029          1.0743         4.49
IVF-Binary-512-nl223-random (self)                    12_049.90     1_785.24    13_835.14       0.2736          1.1201         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          12_682.20       475.38    13_157.58       0.0928             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          12_682.20       473.65    13_155.85       0.0914             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          12_682.20       480.93    13_163.13       0.0894             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         12_682.20       565.86    13_248.06       0.2818          1.1164         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         12_682.20       645.31    13_327.51       0.4179          1.0705         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         12_682.20       565.17    13_247.37       0.2784          1.1179         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         12_682.20       724.34    13_406.54       0.4133          1.0716         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         12_682.20       579.34    13_261.54       0.2728          1.1206         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         12_682.20       660.75    13_342.95       0.4065          1.0733         4.67
IVF-Binary-512-nl316-random (self)                    12_682.20     1_784.23    14_466.43       0.2777          1.1180         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              15_259.38       482.31    15_741.69       0.2324             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             15_259.38       480.51    15_739.88       0.2185             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             15_259.38       501.06    15_760.44       0.2074             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             15_259.38       561.52    15_820.89       0.6593          1.0274         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             15_259.38       652.27    15_911.64       0.8106          1.0119         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            15_259.38       572.11    15_831.48       0.6165          1.0336         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            15_259.38       667.52    15_926.90       0.7678          1.0156         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            15_259.38       600.25    15_859.62       0.5819          1.0391         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            15_259.38       706.69    15_966.06       0.7304          1.0193         4.36
IVF-Binary-512-nl158-pca (self)                       15_259.38     1_855.31    17_114.69       0.6164          1.0336         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             12_857.75       471.04    13_328.79       0.2304             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             12_857.75       477.68    13_335.43       0.2235             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             12_857.75       485.20    13_342.95       0.2103             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            12_857.75       576.45    13_434.20       0.6552          1.0280         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            12_857.75       682.85    13_540.60       0.8087          1.0120         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            12_857.75       592.07    13_449.82       0.6324          1.0311         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            12_857.75       694.98    13_552.73       0.7856          1.0140         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            12_857.75       602.62    13_460.37       0.5919          1.0374         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            12_857.75       704.07    13_561.82       0.7416          1.0181         4.49
IVF-Binary-512-nl223-pca (self)                       12_857.75     1_870.37    14_728.12       0.6334          1.0310         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             13_059.54       488.20    13_547.74       0.2313             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             13_059.54       489.35    13_548.89       0.2274             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             13_059.54       496.30    13_555.84       0.2157             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            13_059.54       587.26    13_646.80       0.6597          1.0273         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            13_059.54       678.39    13_737.93       0.8134          1.0116         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            13_059.54       610.23    13_669.77       0.6481          1.0289         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            13_059.54       714.02    13_773.56       0.8017          1.0126         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            13_059.54       600.51    13_660.04       0.6084          1.0347         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            13_059.54       720.79    13_780.33       0.7600          1.0163         4.67
IVF-Binary-512-nl316-pca (self)                       13_059.54     1_892.60    14_952.14       0.6482          1.0289         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          25_318.36       865.88    26_184.24       0.1197             NaN         8.41
IVF-Binary-1024-nl158-np12-rf0-random (query)         25_318.36       891.22    26_209.58       0.1173             NaN         8.41
IVF-Binary-1024-nl158-np17-rf0-random (query)         25_318.36       889.60    26_207.96       0.1153             NaN         8.41
IVF-Binary-1024-nl158-np7-rf10-random (query)         25_318.36     1_008.23    26_326.58       0.3607          1.0878         8.41
IVF-Binary-1024-nl158-np7-rf20-random (query)         25_318.36     1_034.60    26_352.96       0.5146          1.0503         8.41
IVF-Binary-1024-nl158-np12-rf10-random (query)        25_318.36       962.38    26_280.73       0.3529          1.0902         8.41
IVF-Binary-1024-nl158-np12-rf20-random (query)        25_318.36     1_062.33    26_380.69       0.5055          1.0519         8.41
IVF-Binary-1024-nl158-np17-rf10-random (query)        25_318.36       984.18    26_302.54       0.3485          1.0917         8.41
IVF-Binary-1024-nl158-np17-rf20-random (query)        25_318.36     1_080.86    26_399.22       0.5006          1.0529         8.41
IVF-Binary-1024-nl158-random (self)                   25_318.36     3_153.59    28_471.95       0.3520          1.0905         8.41
IVF-Binary-1024-nl223-np11-rf0-random (query)         23_086.84       874.25    23_961.09       0.1184             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         23_086.84       885.61    23_972.44       0.1166             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         23_086.84       898.33    23_985.17       0.1155             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        23_086.84       970.53    24_057.37       0.3560          1.0892         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        23_086.84     1_071.08    24_157.92       0.5090          1.0513         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        23_086.84       969.20    24_056.04       0.3513          1.0908         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        23_086.84     1_060.86    24_147.69       0.5032          1.0523         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        23_086.84       985.04    24_071.88       0.3480          1.0919         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        23_086.84     1_083.29    24_170.13       0.4992          1.0531         8.54
IVF-Binary-1024-nl223-random (self)                   23_086.84     3_143.60    26_230.44       0.3504          1.0910         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         23_335.82       889.30    24_225.13       0.1184             NaN         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)         23_335.82       891.48    24_227.31       0.1177             NaN         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)         23_335.82       913.71    24_249.53       0.1167             NaN         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)        23_335.82       974.13    24_309.95       0.3561          1.0891         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)        23_335.82     1_064.56    24_400.38       0.5101          1.0511         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)        23_335.82       978.58    24_314.40       0.3535          1.0900         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)        23_335.82     1_086.97    24_422.80       0.5064          1.0517         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)        23_335.82     1_017.14    24_352.96       0.3495          1.0913         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)        23_335.82     1_146.40    24_482.23       0.5022          1.0525         8.72
IVF-Binary-1024-nl316-random (self)                   23_335.82     3_163.38    26_499.21       0.3524          1.0903         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)             26_307.55       892.13    27_199.68       0.2423             NaN         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            26_307.55       926.08    27_233.63       0.2418             NaN         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            26_307.55       937.76    27_245.31       0.2417             NaN         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            26_307.55       978.68    27_286.23       0.6848          1.0241         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            26_307.55     1_070.24    27_377.78       0.8347          1.0099         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           26_307.55       995.40    27_302.95       0.6863          1.0240         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           26_307.55     1_096.35    27_403.90       0.8391          1.0095         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           26_307.55     1_017.55    27_325.10       0.6863          1.0240         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           26_307.55     1_125.27    27_432.81       0.8391          1.0095         8.42
IVF-Binary-1024-nl158-pca (self)                      26_307.55     3_258.53    29_566.07       0.6865          1.0239         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            24_011.06       930.31    24_941.38       0.2423             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            24_011.06       910.21    24_921.28       0.2420             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            24_011.06       932.10    24_943.17       0.2419             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           24_011.06     1_005.05    25_016.11       0.6864          1.0239         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           24_011.06     1_085.89    25_096.95       0.8383          1.0096         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           24_011.06     1_013.36    25_024.43       0.6864          1.0240         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           24_011.06     1_109.96    25_121.03       0.8391          1.0095         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           24_011.06     1_019.74    25_030.81       0.6862          1.0240         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           24_011.06     1_156.14    25_167.20       0.8390          1.0095         8.54
IVF-Binary-1024-nl223-pca (self)                      24_011.06     3_272.45    27_283.52       0.6866          1.0239         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            24_301.54       918.13    25_219.67       0.2422             NaN         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            24_301.54       920.81    25_222.35       0.2419             NaN         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            24_301.54       930.25    25_231.79       0.2417             NaN         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           24_301.54     1_011.77    25_313.31       0.6866          1.0239         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           24_301.54     1_111.47    25_413.01       0.8392          1.0095         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           24_301.54     1_024.51    25_326.05       0.6864          1.0239         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           24_301.54     1_110.87    25_412.42       0.8392          1.0095         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           24_301.54     1_052.74    25_354.28       0.6861          1.0240         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           24_301.54     1_128.43    25_429.98       0.8390          1.0095         8.73
IVF-Binary-1024-nl316-pca (self)                      24_301.54     3_294.98    27_596.53       0.6867          1.0239         8.73
IVF-Binary-512-nl158-np7-rf0-signed (query)           14_349.62       450.62    14_800.24       0.0977             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-signed (query)          14_349.62       456.56    14_806.18       0.0934             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-signed (query)          14_349.62       467.05    14_816.67       0.0872             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-signed (query)          14_349.62       535.25    14_884.87       0.2879          1.1143         4.36
IVF-Binary-512-nl158-np7-rf20-signed (query)          14_349.62       618.21    14_967.83       0.4248          1.0690         4.36
IVF-Binary-512-nl158-np12-rf10-signed (query)         14_349.62       544.30    14_893.92       0.2772          1.1189         4.36
IVF-Binary-512-nl158-np12-rf20-signed (query)         14_349.62       631.58    14_981.20       0.4113          1.0721         4.36
IVF-Binary-512-nl158-np17-rf10-signed (query)         14_349.62       560.34    14_909.96       0.2686          1.1230         4.36
IVF-Binary-512-nl158-np17-rf20-signed (query)         14_349.62       648.23    14_997.85       0.4024          1.0744         4.36
IVF-Binary-512-nl158-signed (self)                    14_349.62     1_733.86    16_083.48       0.2766          1.1188         4.36
IVF-Binary-512-nl223-np11-rf0-signed (query)          12_053.95       460.30    12_514.25       0.0946             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-signed (query)          12_053.95       464.22    12_518.16       0.0906             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-signed (query)          12_053.95       472.50    12_526.45       0.0882             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-signed (query)         12_053.95       550.52    12_604.47       0.2815          1.1166         4.49
IVF-Binary-512-nl223-np11-rf20-signed (query)         12_053.95       632.71    12_686.66       0.4178          1.0706         4.49
IVF-Binary-512-nl223-np14-rf10-signed (query)         12_053.95       550.96    12_604.91       0.2738          1.1202         4.49
IVF-Binary-512-nl223-np14-rf20-signed (query)         12_053.95       643.98    12_697.93       0.4088          1.0728         4.49
IVF-Binary-512-nl223-np21-rf10-signed (query)         12_053.95       564.27    12_618.22       0.2690          1.1224         4.49
IVF-Binary-512-nl223-np21-rf20-signed (query)         12_053.95       655.53    12_709.48       0.4029          1.0743         4.49
IVF-Binary-512-nl223-signed (self)                    12_053.95     1_756.16    13_810.11       0.2736          1.1201         4.49
IVF-Binary-512-nl316-np15-rf0-signed (query)          12_436.21       472.98    12_909.19       0.0928             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-signed (query)          12_436.21       473.08    12_909.29       0.0914             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-signed (query)          12_436.21       481.36    12_917.57       0.0894             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-signed (query)         12_436.21       560.90    12_997.11       0.2818          1.1164         4.67
IVF-Binary-512-nl316-np15-rf20-signed (query)         12_436.21       658.33    13_094.54       0.4179          1.0705         4.67
IVF-Binary-512-nl316-np17-rf10-signed (query)         12_436.21       563.75    12_999.96       0.2784          1.1179         4.67
IVF-Binary-512-nl316-np17-rf20-signed (query)         12_436.21       648.59    13_084.80       0.4133          1.0716         4.67
IVF-Binary-512-nl316-np25-rf10-signed (query)         12_436.21       570.09    13_006.30       0.2728          1.1206         4.67
IVF-Binary-512-nl316-np25-rf20-signed (query)         12_436.21       663.27    13_099.48       0.4065          1.0733         4.67
IVF-Binary-512-nl316-signed (self)                    12_436.21     1_794.12    14_230.33       0.2777          1.1180         4.67
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 1024 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 1024D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        39.90    23_098.35    23_138.26       1.0000          1.0000       195.31
Exhaustive (self)                                         39.90    76_838.51    76_878.41       1.0000          1.0000       195.31
ExhaustiveBinary-256-random_no_rr (query)             12_010.93       596.26    12_607.19       0.0290             NaN         2.53
ExhaustiveBinary-256-random-rf10 (query)              12_010.93       752.73    12_763.66       0.1514          1.1441         2.53
ExhaustiveBinary-256-random-rf20 (query)              12_010.93       929.71    12_940.64       0.2535          1.0923         2.53
ExhaustiveBinary-256-random (self)                    12_010.93     2_492.10    14_503.03       0.1523          1.1428         2.53
ExhaustiveBinary-256-pca_no_rr (query)                12_858.39       606.86    13_465.25       0.1730             NaN         2.53
ExhaustiveBinary-256-pca-rf10 (query)                 12_858.39       784.54    13_642.94       0.4383          1.0451         2.53
ExhaustiveBinary-256-pca-rf20 (query)                 12_858.39       977.83    13_836.22       0.5559          1.0281         2.53
ExhaustiveBinary-256-pca (self)                       12_858.39     2_587.49    15_445.89       0.4397          1.0450         2.53
ExhaustiveBinary-512-random_no_rr (query)             23_518.16     1_154.81    24_672.97       0.0628             NaN         5.05
ExhaustiveBinary-512-random-rf10 (query)              23_518.16     1_296.02    24_814.18       0.1942          1.1100         5.05
ExhaustiveBinary-512-random-rf20 (query)              23_518.16     1_478.83    24_996.99       0.3026          1.0701         5.05
ExhaustiveBinary-512-random (self)                    23_518.16     4_283.09    27_801.25       0.1954          1.1092         5.05
ExhaustiveBinary-512-pca_no_rr (query)                24_719.87     1_156.39    25_876.26       0.1519             NaN         5.06
ExhaustiveBinary-512-pca-rf10 (query)                 24_719.87     1_329.79    26_049.66       0.3855          1.1037         5.06
ExhaustiveBinary-512-pca-rf20 (query)                 24_719.87     1_515.90    26_235.77       0.4977          1.0366         5.06
ExhaustiveBinary-512-pca (self)                       24_719.87     4_390.29    29_110.16       0.3861          1.1000         5.06
ExhaustiveBinary-1024-random_no_rr (query)            47_148.53     2_225.58    49_374.11       0.0893             NaN        10.10
ExhaustiveBinary-1024-random-rf10 (query)             47_148.53     2_307.07    49_455.60       0.2382          1.0892        10.10
ExhaustiveBinary-1024-random-rf20 (query)             47_148.53     2_511.38    49_659.91       0.3610          1.0558        10.10
ExhaustiveBinary-1024-random (self)                   47_148.53     7_658.76    54_807.29       0.2382          1.0889        10.10
ExhaustiveBinary-1024-pca_no_rr (query)               48_549.20     2_170.21    50_719.41       0.1240             NaN        10.11
ExhaustiveBinary-1024-pca-rf10 (query)                48_549.20     2_350.00    50_899.20       0.3310        229.0833        10.11
ExhaustiveBinary-1024-pca-rf20 (query)                48_549.20     2_557.53    51_106.72       0.4429          1.2254        10.11
ExhaustiveBinary-1024-pca (self)                      48_549.20     7_807.95    56_357.15       0.3309        235.7787        10.11
ExhaustiveBinary-1024-signed_no_rr (query)            47_230.84     2_163.92    49_394.76       0.0893             NaN        10.10
ExhaustiveBinary-1024-signed-rf10 (query)             47_230.84     2_313.76    49_544.59       0.2382          1.0892        10.10
ExhaustiveBinary-1024-signed-rf20 (query)             47_230.84     2_520.17    49_751.01       0.3610          1.0558        10.10
ExhaustiveBinary-1024-signed (self)                   47_230.84     7_687.09    54_917.93       0.2382          1.0889        10.10
IVF-Binary-256-nl158-np7-rf0-random (query)           19_429.93       500.52    19_930.45       0.0466             NaN         3.14
IVF-Binary-256-nl158-np12-rf0-random (query)          19_429.93       504.09    19_934.02       0.0401             NaN         3.14
IVF-Binary-256-nl158-np17-rf0-random (query)          19_429.93       516.33    19_946.26       0.0318             NaN         3.14
IVF-Binary-256-nl158-np7-rf10-random (query)          19_429.93       621.46    20_051.39       0.1855          1.1211         3.14
IVF-Binary-256-nl158-np7-rf20-random (query)          19_429.93       748.36    20_178.30       0.2921          1.0777         3.14
IVF-Binary-256-nl158-np12-rf10-random (query)         19_429.93       634.89    20_064.82       0.1726          1.1303         3.14
IVF-Binary-256-nl158-np12-rf20-random (query)         19_429.93       752.48    20_182.41       0.2786          1.0828         3.14
IVF-Binary-256-nl158-np17-rf10-random (query)         19_429.93       632.55    20_062.48       0.1597          1.1394         3.14
IVF-Binary-256-nl158-np17-rf20-random (query)         19_429.93       761.97    20_191.90       0.2658          1.0879         3.14
IVF-Binary-256-nl158-random (self)                    19_429.93     1_974.36    21_404.30       0.1733          1.1290         3.14
IVF-Binary-256-nl223-np11-rf0-random (query)          14_001.50       518.26    14_519.76       0.0446             NaN         3.40
IVF-Binary-256-nl223-np14-rf0-random (query)          14_001.50       530.87    14_532.38       0.0349             NaN         3.40
IVF-Binary-256-nl223-np21-rf0-random (query)          14_001.50       529.66    14_531.16       0.0313             NaN         3.40
IVF-Binary-256-nl223-np11-rf10-random (query)         14_001.50       651.93    14_653.43       0.1883          1.1195         3.40
IVF-Binary-256-nl223-np11-rf20-random (query)         14_001.50       770.63    14_772.13       0.2980          1.0751         3.40
IVF-Binary-256-nl223-np14-rf10-random (query)         14_001.50       644.82    14_646.32       0.1646          1.1365         3.40
IVF-Binary-256-nl223-np14-rf20-random (query)         14_001.50       835.47    14_836.97       0.2716          1.0860         3.40
IVF-Binary-256-nl223-np21-rf10-random (query)         14_001.50       668.42    14_669.92       0.1549          1.1434         3.40
IVF-Binary-256-nl223-np21-rf20-random (query)         14_001.50       782.72    14_784.23       0.2595          1.0910         3.40
IVF-Binary-256-nl223-random (self)                    14_001.50     2_050.15    16_051.65       0.1655          1.1353         3.40
IVF-Binary-256-nl316-np15-rf0-random (query)          14_655.15       559.69    15_214.83       0.0364             NaN         3.76
IVF-Binary-256-nl316-np17-rf0-random (query)          14_655.15       565.35    15_220.50       0.0348             NaN         3.76
IVF-Binary-256-nl316-np25-rf0-random (query)          14_655.15       566.76    15_221.91       0.0330             NaN         3.76
IVF-Binary-256-nl316-np15-rf10-random (query)         14_655.15       681.00    15_336.15       0.1705          1.1324         3.76
IVF-Binary-256-nl316-np15-rf20-random (query)         14_655.15       807.54    15_462.69       0.2805          1.0829         3.76
IVF-Binary-256-nl316-np17-rf10-random (query)         14_655.15       680.53    15_335.68       0.1660          1.1354         3.76
IVF-Binary-256-nl316-np17-rf20-random (query)         14_655.15       825.22    15_480.37       0.2747          1.0849         3.76
IVF-Binary-256-nl316-np25-rf10-random (query)         14_655.15       689.28    15_344.43       0.1607          1.1388         3.76
IVF-Binary-256-nl316-np25-rf20-random (query)         14_655.15       814.93    15_470.08       0.2667          1.0878         3.76
IVF-Binary-256-nl316-random (self)                    14_655.15     2_169.54    16_824.69       0.1671          1.1342         3.76
IVF-Binary-256-nl158-np7-rf0-pca (query)              20_365.39       506.60    20_871.98       0.2011             NaN         3.15
IVF-Binary-256-nl158-np12-rf0-pca (query)             20_365.39       514.52    20_879.91       0.1995             NaN         3.15
IVF-Binary-256-nl158-np17-rf0-pca (query)             20_365.39       523.63    20_889.01       0.1980             NaN         3.15
IVF-Binary-256-nl158-np7-rf10-pca (query)             20_365.39       643.75    21_009.13       0.5842          1.0250         3.15
IVF-Binary-256-nl158-np7-rf20-pca (query)             20_365.39       785.69    21_151.07       0.7446          1.0119         3.15
IVF-Binary-256-nl158-np12-rf10-pca (query)            20_365.39       652.48    21_017.87       0.5775          1.0257         3.15
IVF-Binary-256-nl158-np12-rf20-pca (query)            20_365.39       788.63    21_154.02       0.7361          1.0124         3.15
IVF-Binary-256-nl158-np17-rf10-pca (query)            20_365.39       690.24    21_055.62       0.5672          1.0268         3.15
IVF-Binary-256-nl158-np17-rf20-pca (query)            20_365.39       807.23    21_172.61       0.7214          1.0134         3.15
IVF-Binary-256-nl158-pca (self)                       20_365.39     2_111.00    22_476.39       0.5783          1.0257         3.15
IVF-Binary-256-nl223-np11-rf0-pca (query)             15_119.55       536.09    15_655.64       0.2009             NaN         3.40
IVF-Binary-256-nl223-np14-rf0-pca (query)             15_119.55       539.58    15_659.13       0.2000             NaN         3.40
IVF-Binary-256-nl223-np21-rf0-pca (query)             15_119.55       539.25    15_658.79       0.1986             NaN         3.40
IVF-Binary-256-nl223-np11-rf10-pca (query)            15_119.55       669.63    15_789.18       0.5840          1.0250         3.40
IVF-Binary-256-nl223-np11-rf20-pca (query)            15_119.55       813.06    15_932.60       0.7445          1.0119         3.40
IVF-Binary-256-nl223-np14-rf10-pca (query)            15_119.55       688.70    15_808.24       0.5802          1.0254         3.40
IVF-Binary-256-nl223-np14-rf20-pca (query)            15_119.55       811.60    15_931.15       0.7408          1.0121         3.40
IVF-Binary-256-nl223-np21-rf10-pca (query)            15_119.55       693.07    15_812.62       0.5710          1.0264         3.40
IVF-Binary-256-nl223-np21-rf20-pca (query)            15_119.55       831.94    15_951.49       0.7267          1.0131         3.40
IVF-Binary-256-nl223-pca (self)                       15_119.55     2_191.11    17_310.65       0.5809          1.0254         3.40
IVF-Binary-256-nl316-np15-rf0-pca (query)             15_760.32       565.09    16_325.41       0.2008             NaN         3.77
IVF-Binary-256-nl316-np17-rf0-pca (query)             15_760.32       565.46    16_325.78       0.2005             NaN         3.77
IVF-Binary-256-nl316-np25-rf0-pca (query)             15_760.32       585.14    16_345.46       0.1995             NaN         3.77
IVF-Binary-256-nl316-np15-rf10-pca (query)            15_760.32       706.79    16_467.10       0.5848          1.0249         3.77
IVF-Binary-256-nl316-np15-rf20-pca (query)            15_760.32       843.13    16_603.45       0.7463          1.0117         3.77
IVF-Binary-256-nl316-np17-rf10-pca (query)            15_760.32       704.19    16_464.51       0.5831          1.0251         3.77
IVF-Binary-256-nl316-np17-rf20-pca (query)            15_760.32       846.04    16_606.36       0.7445          1.0119         3.77
IVF-Binary-256-nl316-np25-rf10-pca (query)            15_760.32       714.98    16_475.30       0.5760          1.0258         3.77
IVF-Binary-256-nl316-np25-rf20-pca (query)            15_760.32       859.02    16_619.34       0.7341          1.0126         3.77
IVF-Binary-256-nl316-pca (self)                       15_760.32     2_282.18    18_042.50       0.5837          1.0251         3.77
IVF-Binary-512-nl158-np7-rf0-random (query)           31_229.51       929.37    32_158.88       0.0738             NaN         5.67
IVF-Binary-512-nl158-np12-rf0-random (query)          31_229.51       958.14    32_187.65       0.0699             NaN         5.67
IVF-Binary-512-nl158-np17-rf0-random (query)          31_229.51       952.33    32_181.84       0.0656             NaN         5.67
IVF-Binary-512-nl158-np7-rf10-random (query)          31_229.51     1_050.76    32_280.27       0.2129          1.1011         5.67
IVF-Binary-512-nl158-np7-rf20-random (query)          31_229.51     1_194.34    32_423.86       0.3277          1.0638         5.67
IVF-Binary-512-nl158-np12-rf10-random (query)         31_229.51     1_093.79    32_323.31       0.2065          1.1043         5.67
IVF-Binary-512-nl158-np12-rf20-random (query)         31_229.51     1_210.52    32_440.04       0.3187          1.0660         5.67
IVF-Binary-512-nl158-np17-rf10-random (query)         31_229.51     1_075.50    32_305.01       0.2008          1.1069         5.67
IVF-Binary-512-nl158-np17-rf20-random (query)         31_229.51     1_205.00    32_434.52       0.3121          1.0675         5.67
IVF-Binary-512-nl158-random (self)                    31_229.51     3_470.49    34_700.00       0.2075          1.1033         5.67
IVF-Binary-512-nl223-np11-rf0-random (query)          26_373.00       968.84    27_341.84       0.0738             NaN         5.92
IVF-Binary-512-nl223-np14-rf0-random (query)          26_373.00       971.76    27_344.76       0.0664             NaN         5.92
IVF-Binary-512-nl223-np21-rf0-random (query)          26_373.00       974.29    27_347.28       0.0641             NaN         5.92
IVF-Binary-512-nl223-np11-rf10-random (query)         26_373.00     1_109.58    27_482.58       0.2149          1.0991         5.92
IVF-Binary-512-nl223-np11-rf20-random (query)         26_373.00     1_215.45    27_588.45       0.3290          1.0627         5.92
IVF-Binary-512-nl223-np14-rf10-random (query)         26_373.00     1_090.58    27_463.57       0.2038          1.1055         5.92
IVF-Binary-512-nl223-np14-rf20-random (query)         26_373.00     1_240.94    27_613.94       0.3156          1.0666         5.92
IVF-Binary-512-nl223-np21-rf10-random (query)         26_373.00     1_096.34    27_469.34       0.1975          1.1089         5.92
IVF-Binary-512-nl223-np21-rf20-random (query)         26_373.00     1_268.87    27_641.86       0.3090          1.0686         5.92
IVF-Binary-512-nl223-random (self)                    26_373.00     4_061.76    30_434.76       0.2037          1.1051         5.92
IVF-Binary-512-nl316-np15-rf0-random (query)          26_794.13       997.48    27_791.61       0.0683             NaN         6.29
IVF-Binary-512-nl316-np17-rf0-random (query)          26_794.13       998.68    27_792.81       0.0673             NaN         6.29
IVF-Binary-512-nl316-np25-rf0-random (query)          26_794.13     1_010.46    27_804.59       0.0659             NaN         6.29
IVF-Binary-512-nl316-np15-rf10-random (query)         26_794.13     1_114.50    27_908.63       0.2080          1.1036         6.29
IVF-Binary-512-nl316-np15-rf20-random (query)         26_794.13     1_245.43    28_039.56       0.3214          1.0652         6.29
IVF-Binary-512-nl316-np17-rf10-random (query)         26_794.13     1_114.19    27_908.32       0.2053          1.1050         6.29
IVF-Binary-512-nl316-np17-rf20-random (query)         26_794.13     1_249.65    28_043.78       0.3171          1.0663         6.29
IVF-Binary-512-nl316-np25-rf10-random (query)         26_794.13     1_122.53    27_916.66       0.2013          1.1069         6.29
IVF-Binary-512-nl316-np25-rf20-random (query)         26_794.13     1_289.25    28_083.38       0.3119          1.0675         6.29
IVF-Binary-512-nl316-random (self)                    26_794.13     3_618.54    30_412.67       0.2056          1.1044         6.29
IVF-Binary-512-nl158-np7-rf0-pca (query)              32_328.85       953.37    33_282.23       0.2590             NaN         5.67
IVF-Binary-512-nl158-np12-rf0-pca (query)             32_328.85       958.83    33_287.69       0.2497             NaN         5.67
IVF-Binary-512-nl158-np17-rf0-pca (query)             32_328.85       973.22    33_302.08       0.2402             NaN         5.67
IVF-Binary-512-nl158-np7-rf10-pca (query)             32_328.85     1_103.61    33_432.46       0.7002          1.0145         5.67
IVF-Binary-512-nl158-np7-rf20-pca (query)             32_328.85     1_228.30    33_557.15       0.8406          1.0062         5.67
IVF-Binary-512-nl158-np12-rf10-pca (query)            32_328.85     1_095.42    33_424.27       0.6745          1.0165         5.67
IVF-Binary-512-nl158-np12-rf20-pca (query)            32_328.85     1_236.31    33_565.16       0.8168          1.0072         5.67
IVF-Binary-512-nl158-np17-rf10-pca (query)            32_328.85     1_111.95    33_440.80       0.6442          1.0191         5.67
IVF-Binary-512-nl158-np17-rf20-pca (query)            32_328.85     1_263.22    33_592.07       0.7855          1.0089         5.67
IVF-Binary-512-nl158-pca (self)                       32_328.85     3_599.76    35_928.61       0.6752          1.0165         5.67
IVF-Binary-512-nl223-np11-rf0-pca (query)             27_267.30       981.20    28_248.51       0.2574             NaN         5.93
IVF-Binary-512-nl223-np14-rf0-pca (query)             27_267.30       984.74    28_252.05       0.2535             NaN         5.93
IVF-Binary-512-nl223-np21-rf0-pca (query)             27_267.30       991.38    28_258.69       0.2437             NaN         5.93
IVF-Binary-512-nl223-np11-rf10-pca (query)            27_267.30     1_118.93    28_386.23       0.6972          1.0147         5.93
IVF-Binary-512-nl223-np11-rf20-pca (query)            27_267.30     1_251.90    28_519.21       0.8391          1.0062         5.93
IVF-Binary-512-nl223-np14-rf10-pca (query)            27_267.30     1_130.40    28_397.70       0.6866          1.0155         5.93
IVF-Binary-512-nl223-np14-rf20-pca (query)            27_267.30     1_262.85    28_530.15       0.8296          1.0066         5.93
IVF-Binary-512-nl223-np21-rf10-pca (query)            27_267.30     1_143.99    28_411.29       0.6566          1.0180         5.93
IVF-Binary-512-nl223-np21-rf20-pca (query)            27_267.30     1_276.05    28_543.35       0.7980          1.0083         5.93
IVF-Binary-512-nl223-pca (self)                       27_267.30     3_665.06    30_932.36       0.6870          1.0156         5.93
IVF-Binary-512-nl316-np15-rf0-pca (query)             27_612.49     1_011.57    28_624.06       0.2583             NaN         6.29
IVF-Binary-512-nl316-np17-rf0-pca (query)             27_612.49     1_019.10    28_631.60       0.2565             NaN         6.29
IVF-Binary-512-nl316-np25-rf0-pca (query)             27_612.49     1_022.95    28_635.44       0.2487             NaN         6.29
IVF-Binary-512-nl316-np15-rf10-pca (query)            27_612.49     1_140.31    28_752.81       0.7009          1.0144         6.29
IVF-Binary-512-nl316-np15-rf20-pca (query)            27_612.49     1_275.79    28_888.28       0.8436          1.0059         6.29
IVF-Binary-512-nl316-np17-rf10-pca (query)            27_612.49     1_140.64    28_753.13       0.6959          1.0148         6.29
IVF-Binary-512-nl316-np17-rf20-pca (query)            27_612.49     1_283.78    28_896.27       0.8394          1.0061         6.29
IVF-Binary-512-nl316-np25-rf10-pca (query)            27_612.49     1_157.10    28_769.59       0.6714          1.0168         6.29
IVF-Binary-512-nl316-np25-rf20-pca (query)            27_612.49     1_295.48    28_907.97       0.8138          1.0074         6.29
IVF-Binary-512-nl316-pca (self)                       27_612.49     3_753.18    31_365.67       0.6967          1.0148         6.29
IVF-Binary-1024-nl158-np7-rf0-random (query)          54_626.04     1_807.47    56_433.51       0.0929             NaN        10.72
IVF-Binary-1024-nl158-np12-rf0-random (query)         54_626.04     1_841.94    56_467.98       0.0915             NaN        10.72
IVF-Binary-1024-nl158-np17-rf0-random (query)         54_626.04     1_848.53    56_474.57       0.0906             NaN        10.72
IVF-Binary-1024-nl158-np7-rf10-random (query)         54_626.04     2_055.81    56_681.85       0.2511          1.0849        10.72
IVF-Binary-1024-nl158-np7-rf20-random (query)         54_626.04     2_045.76    56_671.80       0.3779          1.0527        10.72
IVF-Binary-1024-nl158-np12-rf10-random (query)        54_626.04     1_932.32    56_558.36       0.2460          1.0867        10.72
IVF-Binary-1024-nl158-np12-rf20-random (query)        54_626.04     2_059.44    56_685.47       0.3703          1.0541        10.72
IVF-Binary-1024-nl158-np17-rf10-random (query)        54_626.04     1_942.73    56_568.77       0.2426          1.0878        10.72
IVF-Binary-1024-nl158-np17-rf20-random (query)        54_626.04     2_104.00    56_730.04       0.3665          1.0548        10.72
IVF-Binary-1024-nl158-random (self)                   54_626.04     6_334.59    60_960.63       0.2458          1.0866        10.72
IVF-Binary-1024-nl223-np11-rf0-random (query)         49_544.23     1_840.75    51_384.99       0.0929             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-random (query)         49_544.23     1_852.46    51_396.69       0.0909             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-random (query)         49_544.23     1_878.31    51_422.54       0.0900             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-random (query)        49_544.23     1_947.06    51_491.30       0.2495          1.0856        10.98
IVF-Binary-1024-nl223-np11-rf20-random (query)        49_544.23     2_074.04    51_618.28       0.3749          1.0534        10.98
IVF-Binary-1024-nl223-np14-rf10-random (query)        49_544.23     1_974.04    51_518.27       0.2430          1.0876        10.98
IVF-Binary-1024-nl223-np14-rf20-random (query)        49_544.23     2_082.57    51_626.80       0.3670          1.0547        10.98
IVF-Binary-1024-nl223-np21-rf10-random (query)        49_544.23     1_977.29    51_521.52       0.2402          1.0885        10.98
IVF-Binary-1024-nl223-np21-rf20-random (query)        49_544.23     2_103.77    51_648.00       0.3641          1.0552        10.98
IVF-Binary-1024-nl223-random (self)                   49_544.23     6_415.64    55_959.87       0.2435          1.0873        10.98
IVF-Binary-1024-nl316-np15-rf0-random (query)         50_036.76     1_877.94    51_914.71       0.0918             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-random (query)         50_036.76     1_874.90    51_911.66       0.0913             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-random (query)         50_036.76     1_898.67    51_935.44       0.0906             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-random (query)        50_036.76     1_992.56    52_029.33       0.2465          1.0864        11.34
IVF-Binary-1024-nl316-np15-rf20-random (query)        50_036.76     2_100.56    52_137.32       0.3722          1.0538        11.34
IVF-Binary-1024-nl316-np17-rf10-random (query)        50_036.76     1_974.58    52_011.34       0.2444          1.0871        11.34
IVF-Binary-1024-nl316-np17-rf20-random (query)        50_036.76     2_122.84    52_159.61       0.3692          1.0544        11.34
IVF-Binary-1024-nl316-np25-rf10-random (query)        50_036.76     1_995.94    52_032.70       0.2419          1.0880        11.34
IVF-Binary-1024-nl316-np25-rf20-random (query)        50_036.76     2_129.13    52_165.90       0.3655          1.0550        11.34
IVF-Binary-1024-nl316-random (self)                   50_036.76     6_504.31    56_541.08       0.2448          1.0869        11.34
IVF-Binary-1024-nl158-np7-rf0-pca (query)             55_746.42     1_836.44    57_582.86       0.3012             NaN        10.73
IVF-Binary-1024-nl158-np12-rf0-pca (query)            55_746.42     1_858.46    57_604.88       0.2743             NaN        10.73
IVF-Binary-1024-nl158-np17-rf0-pca (query)            55_746.42     1_886.32    57_632.74       0.2512             NaN        10.73
IVF-Binary-1024-nl158-np7-rf10-pca (query)            55_746.42     1_952.57    57_698.99       0.7699          1.0097        10.73
IVF-Binary-1024-nl158-np7-rf20-pca (query)            55_746.42     2_275.03    58_021.45       0.8902          1.0038        10.73
IVF-Binary-1024-nl158-np12-rf10-pca (query)           55_746.42     1_959.75    57_706.17       0.7208          1.0128        10.73
IVF-Binary-1024-nl158-np12-rf20-pca (query)           55_746.42     2_114.23    57_860.64       0.8547          1.0052        10.73
IVF-Binary-1024-nl158-np17-rf10-pca (query)           55_746.42     1_993.74    57_740.16       0.6729          1.0166        10.73
IVF-Binary-1024-nl158-np17-rf20-pca (query)           55_746.42     2_138.31    57_884.73       0.8118          1.0073        10.73
IVF-Binary-1024-nl158-pca (self)                      55_746.42     6_492.05    62_238.47       0.7229          1.0127        10.73
IVF-Binary-1024-nl223-np11-rf0-pca (query)            50_667.66     1_873.23    52_540.89       0.2963             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-pca (query)            50_667.66     1_867.00    52_534.66       0.2853             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-pca (query)            50_667.66     1_894.86    52_562.52       0.2607             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-pca (query)           50_667.66     1_977.87    52_645.53       0.7635          1.0101        10.98
IVF-Binary-1024-nl223-np11-rf20-pca (query)           50_667.66     2_119.21    52_786.87       0.8870          1.0039        10.98
IVF-Binary-1024-nl223-np14-rf10-pca (query)           50_667.66     1_983.27    52_650.93       0.7442          1.0113        10.98
IVF-Binary-1024-nl223-np14-rf20-pca (query)           50_667.66     2_137.73    52_805.40       0.8738          1.0044        10.98
IVF-Binary-1024-nl223-np21-rf10-pca (query)           50_667.66     2_009.09    52_676.75       0.6935          1.0150        10.98
IVF-Binary-1024-nl223-np21-rf20-pca (query)           50_667.66     2_160.51    52_828.17       0.8303          1.0064        10.98
IVF-Binary-1024-nl223-pca (self)                      50_667.66     6_574.51    57_242.17       0.7453          1.0112        10.98
IVF-Binary-1024-nl316-np15-rf0-pca (query)            51_295.64     1_899.81    53_195.45       0.2992             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-pca (query)            51_295.64     1_897.78    53_193.42       0.2939             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-pca (query)            51_295.64     1_916.82    53_212.46       0.2724             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-pca (query)           51_295.64     2_010.53    53_306.17       0.7698          1.0097        11.34
IVF-Binary-1024-nl316-np15-rf20-pca (query)           51_295.64     2_157.19    53_452.82       0.8934          1.0036        11.34
IVF-Binary-1024-nl316-np17-rf10-pca (query)           51_295.64     2_040.55    53_336.18       0.7606          1.0102        11.34
IVF-Binary-1024-nl316-np17-rf20-pca (query)           51_295.64     2_161.29    53_456.93       0.8877          1.0038        11.34
IVF-Binary-1024-nl316-np25-rf10-pca (query)           51_295.64     2_043.20    53_338.84       0.7184          1.0131        11.34
IVF-Binary-1024-nl316-np25-rf20-pca (query)           51_295.64     2_186.20    53_481.84       0.8524          1.0053        11.34
IVF-Binary-1024-nl316-pca (self)                      51_295.64     6_722.85    58_018.49       0.7626          1.0101        11.34
IVF-Binary-1024-nl158-np7-rf0-signed (query)          54_816.11     1_801.52    56_617.63       0.0929             NaN        10.72
IVF-Binary-1024-nl158-np12-rf0-signed (query)         54_816.11     1_822.07    56_638.18       0.0915             NaN        10.72
IVF-Binary-1024-nl158-np17-rf0-signed (query)         54_816.11     1_843.91    56_660.02       0.0906             NaN        10.72
IVF-Binary-1024-nl158-np7-rf10-signed (query)         54_816.11     1_909.45    56_725.56       0.2511          1.0849        10.72
IVF-Binary-1024-nl158-np7-rf20-signed (query)         54_816.11     2_064.51    56_880.62       0.3779          1.0527        10.72
IVF-Binary-1024-nl158-np12-rf10-signed (query)        54_816.11     1_947.15    56_763.26       0.2460          1.0867        10.72
IVF-Binary-1024-nl158-np12-rf20-signed (query)        54_816.11     2_093.84    56_909.95       0.3703          1.0541        10.72
IVF-Binary-1024-nl158-np17-rf10-signed (query)        54_816.11     1_977.66    56_793.77       0.2426          1.0878        10.72
IVF-Binary-1024-nl158-np17-rf20-signed (query)        54_816.11     2_087.06    56_903.17       0.3665          1.0548        10.72
IVF-Binary-1024-nl158-signed (self)                   54_816.11     6_355.30    61_171.42       0.2458          1.0866        10.72
IVF-Binary-1024-nl223-np11-rf0-signed (query)         49_759.63     1_839.67    51_599.29       0.0929             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-signed (query)         49_759.63     1_838.02    51_597.64       0.0909             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-signed (query)         49_759.63     1_851.42    51_611.05       0.0900             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-signed (query)        49_759.63     1_941.73    51_701.35       0.2495          1.0856        10.98
IVF-Binary-1024-nl223-np11-rf20-signed (query)        49_759.63     2_073.00    51_832.62       0.3749          1.0534        10.98
IVF-Binary-1024-nl223-np14-rf10-signed (query)        49_759.63     1_955.83    51_715.46       0.2430          1.0876        10.98
IVF-Binary-1024-nl223-np14-rf20-signed (query)        49_759.63     2_085.53    51_845.16       0.3670          1.0547        10.98
IVF-Binary-1024-nl223-np21-rf10-signed (query)        49_759.63     1_979.77    51_739.39       0.2402          1.0885        10.98
IVF-Binary-1024-nl223-np21-rf20-signed (query)        49_759.63     2_124.82    51_884.44       0.3641          1.0552        10.98
IVF-Binary-1024-nl223-signed (self)                   49_759.63     6_384.32    56_143.95       0.2435          1.0873        10.98
IVF-Binary-1024-nl316-np15-rf0-signed (query)         50_220.02     2_015.63    52_235.66       0.0918             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-signed (query)         50_220.02     1_868.90    52_088.93       0.0913             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-signed (query)         50_220.02     2_004.87    52_224.89       0.0906             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-signed (query)        50_220.02     1_968.16    52_188.19       0.2465          1.0864        11.34
IVF-Binary-1024-nl316-np15-rf20-signed (query)        50_220.02     2_130.09    52_350.12       0.3722          1.0538        11.34
IVF-Binary-1024-nl316-np17-rf10-signed (query)        50_220.02     1_976.68    52_196.70       0.2444          1.0871        11.34
IVF-Binary-1024-nl316-np17-rf20-signed (query)        50_220.02     2_116.01    52_336.03       0.3692          1.0544        11.34
IVF-Binary-1024-nl316-np25-rf10-signed (query)        50_220.02     1_991.53    52_211.55       0.2419          1.0880        11.34
IVF-Binary-1024-nl316-np25-rf20-signed (query)        50_220.02     2_123.10    52_343.12       0.3655          1.0550        11.34
IVF-Binary-1024-nl316-signed (self)                   50_220.02     6_492.86    56_712.89       0.2448          1.0869        11.34
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
Exhaustive (query)                                         6.81     4_386.03     4_392.84       1.0000          1.0000        48.83
Exhaustive (self)                                          6.81    14_726.28    14_733.09       1.0000          1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_583.28       256.80     2_840.09       0.0387             NaN         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_583.28       371.45     2_954.73       0.2254          1.0175         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_583.28       478.40     3_061.69       0.3590          1.0099         1.78
ExhaustiveBinary-256-random (self)                     2_583.28     1_223.76     3_807.04       0.5886         13.9170         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_804.00       262.54     3_066.54       0.0284             NaN         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_804.00       376.78     3_180.79       0.1316          1.0445         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_804.00       488.20     3_292.21       0.1964          1.0197         1.78
ExhaustiveBinary-256-pca (self)                        2_804.00     1_240.29     4_044.30       0.2462          1.6481         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_094.53       466.25     5_560.78       0.0605             NaN         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_094.53       580.89     5_675.42       0.3044          1.0140         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_094.53       691.45     5_785.99       0.4560          1.0080         3.55
ExhaustiveBinary-512-random (self)                     5_094.53     1_937.12     7_031.65       0.6749         20.2700         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_423.23       484.11     5_907.34       0.0806             NaN         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_423.23       585.63     6_008.86       0.3799          1.0083         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_423.23       700.70     6_123.92       0.5477          1.0039         3.55
ExhaustiveBinary-512-pca (self)                        5_423.23     1_958.89     7_382.11       0.6562          1.0566         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_065.85       793.08    10_858.93       0.0996             NaN         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_065.85       913.09    10_978.95       0.4213          1.0096         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_065.85     1_056.40    11_122.25       0.5847          1.0054         7.10
ExhaustiveBinary-1024-random (self)                   10_065.85     3_064.11    13_129.96       0.7571         24.6771         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_804.82       809.54    11_614.36       0.1211             NaN         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_804.82       953.32    11_758.14       0.5030          1.0092         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_804.82     1_070.00    11_874.82       0.6838          1.0042         7.10
ExhaustiveBinary-1024-pca (self)                      10_804.82     3_089.58    13_894.39       0.8235          1.0221         7.10
ExhaustiveBinary-256-signed_no_rr (query)              2_619.06       258.67     2_877.72       0.0387             NaN         1.78
ExhaustiveBinary-256-signed-rf10 (query)               2_619.06       370.69     2_989.75       0.2254          1.0175         1.78
ExhaustiveBinary-256-signed-rf20 (query)               2_619.06       479.30     3_098.36       0.3590          1.0099         1.78
ExhaustiveBinary-256-signed (self)                     2_619.06     1_219.97     3_839.02       0.5886         13.9170         1.78
IVF-Binary-256-nl158-np7-rf0-random (query)            3_907.64       124.99     4_032.63       0.0509             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           3_907.64       138.18     4_045.82       0.0465             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           3_907.64       152.80     4_060.44       0.0446             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           3_907.64       193.70     4_101.34       0.3052          1.0091         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           3_907.64       256.54     4_164.18       0.4816          1.0047         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          3_907.64       213.64     4_121.28       0.2841          1.0105         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          3_907.64       284.02     4_191.66       0.4542          1.0055         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          3_907.64       235.81     4_143.45       0.2744          1.0112         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          3_907.64       306.08     4_213.73       0.4401          1.0059         1.93
IVF-Binary-256-nl158-random (self)                     3_907.64       643.61     4_551.25       0.7484          1.0402         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           2_698.62       121.40     2_820.02       0.0512             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           2_698.62       125.56     2_824.18       0.0481             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           2_698.62       132.85     2_831.47       0.0452             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          2_698.62       194.11     2_892.73       0.3029          1.0093         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          2_698.62       248.48     2_947.10       0.4808          1.0050         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          2_698.62       193.23     2_891.84       0.2901          1.0100         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          2_698.62       261.59     2_960.21       0.4641          1.0054         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          2_698.62       206.23     2_904.85       0.2785          1.0106         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          2_698.62       271.92     2_970.54       0.4478          1.0057         2.00
IVF-Binary-256-nl223-random (self)                     2_698.62       577.29     3_275.90       0.7426          1.0418         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           2_724.97       130.08     2_855.06       0.0506             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           2_724.97       130.04     2_855.02       0.0491             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           2_724.97       136.87     2_861.84       0.0461             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          2_724.97       202.07     2_927.04       0.3030          1.0092         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          2_724.97       251.06     2_976.03       0.4824          1.0050         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          2_724.97       198.67     2_923.64       0.2971          1.0095         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          2_724.97       260.02     2_984.99       0.4742          1.0052         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          2_724.97       207.42     2_932.39       0.2832          1.0103         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          2_724.97       270.31     2_995.29       0.4549          1.0056         2.09
IVF-Binary-256-nl316-random (self)                     2_724.97       585.56     3_310.53       0.7435          1.0416         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               4_092.43       128.71     4_221.15       0.0612             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              4_092.43       144.07     4_236.50       0.0515             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              4_092.43       158.66     4_251.09       0.0461             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              4_092.43       209.33     4_301.76       0.3050          1.0087         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              4_092.43       273.45     4_365.89       0.4482          1.0049         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             4_092.43       224.31     4_316.74       0.2538          1.0115         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             4_092.43       297.26     4_389.69       0.3768          1.0067         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             4_092.43       246.18     4_338.61       0.2255          1.0140         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             4_092.43       320.96     4_413.39       0.3355          1.0080         1.93
IVF-Binary-256-nl158-pca (self)                        4_092.43       701.14     4_793.57       0.3531          1.2483         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              2_910.89       125.63     3_036.52       0.0657             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              2_910.89       132.64     3_043.53       0.0603             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              2_910.89       156.91     3_067.80       0.0540             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             2_910.89       201.28     3_112.18       0.3321          1.0076         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             2_910.89       259.59     3_170.48       0.4963          1.0041         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             2_910.89       203.20     3_114.10       0.3066          1.0085         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             2_910.89       264.98     3_175.87       0.4589          1.0048         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             2_910.89       214.78     3_125.68       0.2715          1.0100         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             2_910.89       292.88     3_203.77       0.4064          1.0058         2.00
IVF-Binary-256-nl223-pca (self)                        2_910.89       620.95     3_531.84       0.3965          1.1690         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              2_977.19       137.69     3_114.88       0.0667             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              2_977.19       135.67     3_112.86       0.0641             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              2_977.19       139.14     3_116.32       0.0571             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             2_977.19       205.71     3_182.90       0.3384          1.0074         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             2_977.19       263.90     3_241.09       0.5061          1.0040         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             2_977.19       215.26     3_192.45       0.3251          1.0078         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             2_977.19       269.10     3_246.28       0.4865          1.0043         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             2_977.19       229.11     3_206.30       0.2885          1.0092         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             2_977.19       282.30     3_259.49       0.4321          1.0053         2.09
IVF-Binary-256-nl316-pca (self)                        2_977.19       644.60     3_621.79       0.4100          1.1566         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_377.72       229.87     6_607.59       0.0736             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_377.72       253.61     6_631.32       0.0692             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_377.72       265.03     6_642.74       0.0671             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_377.72       300.29     6_678.01       0.3873          1.0068         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_377.72       350.50     6_728.21       0.5762          1.0034         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_377.72       326.92     6_704.64       0.3697          1.0076         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_377.72       388.40     6_766.12       0.5551          1.0038         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_377.72       359.91     6_737.63       0.3601          1.0080         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_377.72       426.43     6_804.14       0.5426          1.0041         3.71
IVF-Binary-512-nl158-random (self)                     6_377.72     1_007.06     7_384.77       0.8442          1.0200         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_116.88       219.29     5_336.17       0.0735             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_116.88       225.57     5_342.45       0.0705             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_116.88       236.47     5_353.35       0.0678             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_116.88       287.73     5_404.61       0.3851          1.0070         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_116.88       343.06     5_459.94       0.5741          1.0036         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_116.88       293.09     5_409.97       0.3734          1.0075         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_116.88       379.16     5_496.04       0.5598          1.0039         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_116.88       313.82     5_430.70       0.3645          1.0078         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_116.88       378.05     5_494.93       0.5480          1.0041         3.77
IVF-Binary-512-nl223-random (self)                     5_116.88       917.98     6_034.86       0.8385          1.0211         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           5_143.43       225.33     5_368.76       0.0731             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           5_143.43       228.96     5_372.39       0.0717             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           5_143.43       240.85     5_384.28       0.0685             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          5_143.43       295.56     5_438.99       0.3853          1.0070         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          5_143.43       349.62     5_493.05       0.5760          1.0036         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          5_143.43       297.79     5_441.22       0.3798          1.0073         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          5_143.43       372.03     5_515.46       0.5692          1.0037         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          5_143.43       312.03     5_455.46       0.3680          1.0077         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          5_143.43       372.85     5_516.28       0.5534          1.0040         3.86
IVF-Binary-512-nl316-random (self)                     5_143.43       910.10     6_053.53       0.8398          1.0208         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               6_595.90       231.52     6_827.42       0.0969             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              6_595.90       258.87     6_854.77       0.0918             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              6_595.90       277.36     6_873.26       0.0894             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              6_595.90       305.42     6_901.32       0.4469          1.0049         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              6_595.90       364.79     6_960.68       0.6308          1.0024         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             6_595.90       335.83     6_931.73       0.4262          1.0055         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             6_595.90       400.11     6_996.00       0.6079          1.0028         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             6_595.90       364.18     6_960.08       0.4173          1.0058         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             6_595.90       445.59     7_041.48       0.5961          1.0030         3.71
IVF-Binary-512-nl158-pca (self)                        6_595.90     1_045.43     7_641.33       0.6714          1.0520         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              5_434.23       230.51     5_664.74       0.0964             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              5_434.23       243.49     5_677.73       0.0937             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              5_434.23       243.39     5_677.62       0.0910             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             5_434.23       300.96     5_735.19       0.4490          1.0051         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             5_434.23       358.42     5_792.65       0.6410          1.0024         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             5_434.23       304.75     5_738.99       0.4379          1.0054         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             5_434.23       367.40     5_801.63       0.6260          1.0026         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             5_434.23       326.39     5_760.62       0.4265          1.0056         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             5_434.23       391.05     5_825.28       0.6109          1.0028         3.77
IVF-Binary-512-nl223-pca (self)                        5_434.23       943.70     6_377.94       0.6652          1.0540         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              5_462.95       237.16     5_700.11       0.0968             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              5_462.95       243.20     5_706.15       0.0954             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              5_462.95       244.14     5_707.10       0.0922             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             5_462.95       309.63     5_772.58       0.4524          1.0050         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             5_462.95       362.13     5_825.09       0.6451          1.0024         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             5_462.95       308.24     5_771.19       0.4465          1.0052         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             5_462.95       389.16     5_852.12       0.6375          1.0025         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             5_462.95       321.97     5_784.93       0.4325          1.0055         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             5_462.95       385.64     5_848.59       0.6188          1.0027         3.86
IVF-Binary-512-nl316-pca (self)                        5_462.95       950.13     6_413.08       0.6661          1.0537         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_283.42       425.09    11_708.51       0.1145             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_283.42       459.42    11_742.84       0.1107             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_283.42       500.08    11_783.50       0.1084             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_283.42       506.94    11_790.36       0.5052          1.0045         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_283.42       584.65    11_868.07       0.6944          1.0021         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_283.42       545.66    11_829.08       0.4910          1.0049         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_283.42       622.11    11_905.53       0.6794          1.0023         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_283.42       601.17    11_884.59       0.4821          1.0052         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_283.42       660.85    11_944.27       0.6691          1.0025         7.26
IVF-Binary-1024-nl158-random (self)                   11_283.42     1_754.58    13_037.99       0.9241          1.0076         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_173.97       419.83    10_593.79       0.1144             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_173.97       424.57    10_598.53       0.1111             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_173.97       444.90    10_618.86       0.1086             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_173.97       483.08    10_657.04       0.5009          1.0048         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_173.97       542.10    10_716.06       0.6893          1.0023         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_173.97       501.47    10_675.44       0.4913          1.0051         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_173.97       556.32    10_730.29       0.6783          1.0025         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_173.97       515.76    10_689.73       0.4848          1.0052         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_173.97       584.60    10_758.57       0.6715          1.0025         7.32
IVF-Binary-1024-nl223-random (self)                   10_173.97     1_568.90    11_742.86       0.9201          1.0082         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         10_173.35       426.51    10_599.85       0.1143             NaN         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)         10_173.35       434.24    10_607.59       0.1127             NaN         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)         10_173.35       447.13    10_620.48       0.1094             NaN         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)        10_173.35       507.63    10_680.97       0.5017          1.0048         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)        10_173.35       559.50    10_732.85       0.6914          1.0023         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)        10_173.35       493.71    10_667.05       0.4967          1.0049         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)        10_173.35       555.13    10_728.48       0.6857          1.0024         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)        10_173.35       513.44    10_686.79       0.4874          1.0052         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)        10_173.35       577.49    10_750.83       0.6748          1.0025         7.41
IVF-Binary-1024-nl316-random (self)                   10_173.35     1_566.58    11_739.93       0.9211          1.0080         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)             11_829.99       458.21    12_288.20       0.1358             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            11_829.99       470.54    12_300.53       0.1313             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            11_829.99       504.46    12_334.45       0.1292             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            11_829.99       516.19    12_346.18       0.5564          1.0034         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            11_829.99       576.72    12_406.71       0.7414          1.0014         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           11_829.99       555.22    12_385.21       0.5426          1.0037         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           11_829.99       621.89    12_451.88       0.7291          1.0016         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           11_829.99       617.70    12_447.69       0.5360          1.0039         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           11_829.99       671.77    12_501.76       0.7218          1.0017         7.26
IVF-Binary-1024-nl158-pca (self)                      11_829.99     1_768.00    13_597.99       0.8299          1.0204         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            10_722.71       429.03    11_151.74       0.1344             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            10_722.71       438.58    11_161.30       0.1323             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            10_722.71       452.00    11_174.71       0.1304             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           10_722.71       500.65    11_223.37       0.5541          1.0035         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           10_722.71       569.44    11_292.16       0.7434          1.0015         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           10_722.71       513.92    11_236.64       0.5473          1.0037         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           10_722.71       574.52    11_297.23       0.7354          1.0016         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           10_722.71       530.93    11_253.64       0.5408          1.0038         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           10_722.71       599.68    11_322.39       0.7276          1.0017         7.32
IVF-Binary-1024-nl223-pca (self)                      10_722.71     1_622.59    12_345.31       0.8252          1.0214         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            10_618.23       443.15    11_061.38       0.1346             NaN         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            10_618.23       439.48    11_057.71       0.1335             NaN         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            10_618.23       451.30    11_069.53       0.1312             NaN         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           10_618.23       507.92    11_126.15       0.5554          1.0035         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           10_618.23       571.34    11_189.58       0.7452          1.0015         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           10_618.23       512.94    11_131.17       0.5522          1.0036         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           10_618.23       579.25    11_197.48       0.7413          1.0015         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           10_618.23       530.18    11_148.41       0.5438          1.0037         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           10_618.23       598.04    11_216.27       0.7316          1.0016         7.42
IVF-Binary-1024-nl316-pca (self)                      10_618.23     1_615.15    12_233.38       0.8253          1.0214         7.42
IVF-Binary-256-nl158-np7-rf0-signed (query)            3_784.91       124.54     3_909.45       0.0509             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-signed (query)           3_784.91       135.40     3_920.31       0.0465             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-signed (query)           3_784.91       156.55     3_941.46       0.0446             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-signed (query)           3_784.91       196.33     3_981.24       0.3052          1.0091         1.93
IVF-Binary-256-nl158-np7-rf20-signed (query)           3_784.91       246.43     4_031.34       0.4816          1.0047         1.93
IVF-Binary-256-nl158-np12-rf10-signed (query)          3_784.91       210.36     3_995.27       0.2841          1.0105         1.93
IVF-Binary-256-nl158-np12-rf20-signed (query)          3_784.91       280.11     4_065.02       0.4542          1.0055         1.93
IVF-Binary-256-nl158-np17-rf10-signed (query)          3_784.91       234.73     4_019.64       0.2744          1.0112         1.93
IVF-Binary-256-nl158-np17-rf20-signed (query)          3_784.91       306.28     4_091.19       0.4401          1.0059         1.93
IVF-Binary-256-nl158-signed (self)                     3_784.91       644.61     4_429.52       0.7484          1.0402         1.93
IVF-Binary-256-nl223-np11-rf0-signed (query)           2_618.34       123.14     2_741.48       0.0512             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-signed (query)           2_618.34       126.50     2_744.84       0.0481             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-signed (query)           2_618.34       133.46     2_751.80       0.0452             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-signed (query)          2_618.34       188.35     2_806.69       0.3029          1.0093         2.00
IVF-Binary-256-nl223-np11-rf20-signed (query)          2_618.34       245.11     2_863.45       0.4808          1.0050         2.00
IVF-Binary-256-nl223-np14-rf10-signed (query)          2_618.34       193.31     2_811.65       0.2901          1.0100         2.00
IVF-Binary-256-nl223-np14-rf20-signed (query)          2_618.34       251.99     2_870.33       0.4641          1.0054         2.00
IVF-Binary-256-nl223-np21-rf10-signed (query)          2_618.34       204.64     2_822.98       0.2785          1.0106         2.00
IVF-Binary-256-nl223-np21-rf20-signed (query)          2_618.34       268.17     2_886.51       0.4478          1.0057         2.00
IVF-Binary-256-nl223-signed (self)                     2_618.34       571.12     3_189.46       0.7426          1.0418         2.00
IVF-Binary-256-nl316-np15-rf0-signed (query)           2_661.75       128.84     2_790.59       0.0506             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-signed (query)           2_661.75       140.38     2_802.13       0.0491             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-signed (query)           2_661.75       135.30     2_797.05       0.0461             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-signed (query)          2_661.75       198.32     2_860.07       0.3030          1.0092         2.09
IVF-Binary-256-nl316-np15-rf20-signed (query)          2_661.75       250.70     2_912.45       0.4824          1.0050         2.09
IVF-Binary-256-nl316-np17-rf10-signed (query)          2_661.75       197.14     2_858.89       0.2971          1.0095         2.09
IVF-Binary-256-nl316-np17-rf20-signed (query)          2_661.75       253.27     2_915.02       0.4742          1.0052         2.09
IVF-Binary-256-nl316-np25-rf10-signed (query)          2_661.75       206.32     2_868.07       0.2832          1.0103         2.09
IVF-Binary-256-nl316-np25-rf20-signed (query)          2_661.75       266.94     2_928.69       0.4549          1.0056         2.09
IVF-Binary-256-nl316-signed (self)                     2_661.75       581.89     3_243.64       0.7435          1.0416         2.09
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
Exhaustive (query)                                        20.43     9_995.87    10_016.30       1.0000          1.0000        97.66
Exhaustive (self)                                         20.43    33_584.74    33_605.18       1.0000          1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_683.08       369.95     6_053.04       0.0197             NaN         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_683.08       507.10     6_190.19       0.1454          1.0108         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_683.08       645.40     6_328.48       0.2497          1.0068         2.03
ExhaustiveBinary-256-random (self)                     5_683.08     1_967.98     7_651.06       0.6222          7.2120         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_315.53       377.00     6_692.53       0.0177             NaN         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_315.53       524.03     6_839.57       0.0885          1.0249         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_315.53       655.36     6_970.90       0.1361          1.0117         2.03
ExhaustiveBinary-256-pca (self)                        6_315.53     1_701.75     8_017.28       0.2209          1.7046         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_126.98       692.76    11_819.74       0.0425             NaN         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_126.98       828.21    11_955.19       0.2372          1.0067         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_126.98       969.26    12_096.25       0.3726          1.0039         4.05
ExhaustiveBinary-512-random (self)                    11_126.98     2_717.60    13_844.59       0.7112          9.2589         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_819.19       699.21    12_518.40       0.0149             NaN         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_819.19       839.61    12_658.81       0.0785          1.0474         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_819.19       980.21    12_799.40       0.1227          1.0180         4.05
ExhaustiveBinary-512-pca (self)                       11_819.19     2_795.76    14_614.96       0.1561          3.9887         4.05
ExhaustiveBinary-1024-random_no_rr (query)            22_049.53     1_240.13    23_289.66       0.0662             NaN         8.10
ExhaustiveBinary-1024-random-rf10 (query)             22_049.53     1_381.14    23_430.67       0.3226          1.0049         8.10
ExhaustiveBinary-1024-random-rf20 (query)             22_049.53     1_538.98    23_588.51       0.4753          1.0028         8.10
ExhaustiveBinary-1024-random (self)                   22_049.53     4_570.49    26_620.02       0.7729         10.8323         8.10
ExhaustiveBinary-1024-pca_no_rr (query)               23_261.63     1_257.44    24_519.07       0.0749             NaN         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                23_261.63     1_406.91    24_668.55       0.3494          1.0037         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                23_261.63     1_575.29    24_836.92       0.5064          1.0020         8.11
ExhaustiveBinary-1024-pca (self)                      23_261.63     4_740.01    28_001.64       0.6913          1.0488         8.11
ExhaustiveBinary-512-signed_no_rr (query)             11_107.61       687.93    11_795.54       0.0425             NaN         4.05
ExhaustiveBinary-512-signed-rf10 (query)              11_107.61       825.16    11_932.77       0.2372          1.0067         4.05
ExhaustiveBinary-512-signed-rf20 (query)              11_107.61       963.82    12_071.43       0.3726          1.0039         4.05
ExhaustiveBinary-512-signed (self)                    11_107.61     2_730.36    13_837.97       0.7112          9.2589         4.05
IVF-Binary-256-nl158-np7-rf0-random (query)            8_214.41       239.09     8_453.50       0.0301             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           8_214.41       251.13     8_465.54       0.0255             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           8_214.41       258.32     8_472.73       0.0237             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           8_214.41       340.74     8_555.15       0.2211          1.0057         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           8_214.41       430.74     8_645.15       0.3773          1.0030         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          8_214.41       352.30     8_566.71       0.1951          1.0066         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          8_214.41       445.69     8_660.10       0.3354          1.0037         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          8_214.41       364.93     8_579.34       0.1845          1.0072         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          8_214.41       469.54     8_683.95       0.3185          1.0041         2.34
IVF-Binary-256-nl158-random (self)                     8_214.41     1_055.02     9_269.43       0.7814          1.0320         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           5_772.04       249.52     6_021.56       0.0303             NaN         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)           5_772.04       252.66     6_024.69       0.0277             NaN         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)           5_772.04       262.11     6_034.15       0.0252             NaN         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)          5_772.04       345.80     6_117.83       0.2240          1.0055         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)          5_772.04       443.55     6_215.59       0.3838          1.0030         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)          5_772.04       349.16     6_121.19       0.2095          1.0060         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)          5_772.04       441.31     6_213.35       0.3612          1.0033         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)          5_772.04       360.54     6_132.57       0.1942          1.0066         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)          5_772.04       454.19     6_226.23       0.3374          1.0037         2.46
IVF-Binary-256-nl223-random (self)                     5_772.04     1_042.58     6_814.61       0.7767          1.0338         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)           5_765.88       263.40     6_029.28       0.0302             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           5_765.88       267.52     6_033.40       0.0287             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           5_765.88       271.27     6_037.14       0.0259             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          5_765.88       365.62     6_131.49       0.2253          1.0055         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          5_765.88       458.25     6_224.13       0.3874          1.0030         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          5_765.88       362.92     6_128.80       0.2169          1.0057         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          5_765.88       452.73     6_218.61       0.3744          1.0031         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          5_765.88       375.71     6_141.59       0.2002          1.0063         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          5_765.88       467.15     6_233.02       0.3475          1.0035         2.65
IVF-Binary-256-nl316-random (self)                     5_765.88     1_081.73     6_847.61       0.7765          1.0339         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               8_768.39       247.26     9_015.65       0.0474             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              8_768.39       257.01     9_025.40       0.0383             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              8_768.39       266.67     9_035.06       0.0338             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              8_768.39       360.41     9_128.81       0.2568          1.0041         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              8_768.39       451.27     9_219.66       0.3997          1.0024         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             8_768.39       361.43     9_129.82       0.2058          1.0053         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             8_768.39       464.78     9_233.17       0.3228          1.0032         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             8_768.39       378.03     9_146.42       0.1801          1.0062         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             8_768.39       482.30     9_250.70       0.2819          1.0038         2.34
IVF-Binary-256-nl158-pca (self)                        8_768.39     1_215.25     9_983.64       0.3375          1.2161         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              6_271.15       256.15     6_527.30       0.0492             NaN         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              6_271.15       260.93     6_532.08       0.0450             NaN         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              6_271.15       273.88     6_545.02       0.0395             NaN         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             6_271.15       357.61     6_628.75       0.2682          1.0040         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             6_271.15       449.36     6_720.51       0.4193          1.0023         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             6_271.15       357.92     6_629.07       0.2454          1.0044         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             6_271.15       473.81     6_744.95       0.3842          1.0026         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             6_271.15       372.11     6_643.26       0.2134          1.0051         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             6_271.15       471.12     6_742.26       0.3346          1.0031         2.47
IVF-Binary-256-nl223-pca (self)                        6_271.15     1_129.07     7_400.22       0.3700          1.1736         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              6_279.63       270.72     6_550.35       0.0499             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              6_279.63       272.64     6_552.27       0.0477             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              6_279.63       276.04     6_555.67       0.0420             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             6_279.63       375.20     6_654.82       0.2726          1.0040         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             6_279.63       467.39     6_747.02       0.4269          1.0023         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             6_279.63       374.71     6_654.34       0.2605          1.0042         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             6_279.63       469.79     6_749.41       0.4081          1.0024         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             6_279.63       384.32     6_663.95       0.2280          1.0048         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             6_279.63       477.24     6_756.87       0.3570          1.0029         2.65
IVF-Binary-256-nl316-pca (self)                        6_279.63     1_169.36     7_448.99       0.3813          1.1642         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           13_773.37       452.03    14_225.40       0.0550             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          13_773.37       467.66    14_241.03       0.0503             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          13_773.37       483.24    14_256.62       0.0485             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          13_773.37       546.96    14_320.34       0.3197          1.0037         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          13_773.37       629.96    14_403.33       0.5000          1.0019         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         13_773.37       562.00    14_335.37       0.2982          1.0041         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         13_773.37       658.97    14_432.34       0.4687          1.0022         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         13_773.37       583.67    14_357.04       0.2893          1.0043         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         13_773.37       687.47    14_460.84       0.4553          1.0024         4.36
IVF-Binary-512-nl158-random (self)                    13_773.37     1_770.76    15_544.13       0.8734          1.0152         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          11_275.62       469.33    11_744.94       0.0551             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          11_275.62       477.50    11_753.12       0.0524             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          11_275.62       477.78    11_753.39       0.0500             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         11_275.62       552.53    11_828.15       0.3198          1.0039         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         11_275.62       640.09    11_915.71       0.4995          1.0021         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         11_275.62       560.42    11_836.03       0.3078          1.0041         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         11_275.62       649.48    11_925.09       0.4831          1.0022         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         11_275.62       578.88    11_854.50       0.2965          1.0042         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         11_275.62       667.51    11_943.13       0.4669          1.0023         4.49
IVF-Binary-512-nl223-random (self)                    11_275.62     1_744.45    13_020.07       0.8701          1.0161         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          11_368.95       484.49    11_853.44       0.0552             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          11_368.95       477.50    11_846.46       0.0536             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          11_368.95       484.82    11_853.77       0.0508             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         11_368.95       573.81    11_942.77       0.3216          1.0039         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         11_368.95       665.23    12_034.19       0.5021          1.0021         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         11_368.95       569.25    11_938.21       0.3146          1.0040         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         11_368.95       658.87    12_027.83       0.4930          1.0021         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         11_368.95       585.61    11_954.56       0.3015          1.0041         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         11_368.95       677.84    12_046.80       0.4742          1.0023         4.67
IVF-Binary-512-nl316-random (self)                    11_368.95     1_778.58    13_147.54       0.8706          1.0160         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              14_388.82       466.48    14_855.30       0.0554             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             14_388.82       483.93    14_872.75       0.0429             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             14_388.82       497.86    14_886.68       0.0366             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             14_388.82       574.56    14_963.38       0.2851          1.0038         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             14_388.82       658.52    15_047.33       0.4336          1.0022         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            14_388.82       588.92    14_977.74       0.2231          1.0052         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            14_388.82       688.45    15_077.27       0.3413          1.0031         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            14_388.82       604.41    14_993.22       0.1894          1.0066         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            14_388.82       712.79    15_101.60       0.2911          1.0038         4.36
IVF-Binary-512-nl158-pca (self)                       14_388.82     1_902.51    16_291.33       0.2813          1.2960         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             11_924.24       470.02    12_394.26       0.0585             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             11_924.24       477.80    12_402.04       0.0528             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             11_924.24       485.55    12_409.79       0.0450             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            11_924.24       580.27    12_504.50       0.3017          1.0037         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            11_924.24       662.73    12_586.96       0.4606          1.0021         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            11_924.24       577.84    12_502.07       0.2734          1.0041         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            11_924.24       672.54    12_596.78       0.4189          1.0024         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            11_924.24       590.16    12_514.40       0.2319          1.0050         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            11_924.24       690.94    12_615.18       0.3575          1.0030         4.49
IVF-Binary-512-nl223-pca (self)                       11_924.24     1_863.51    13_787.75       0.3205          1.2194         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             12_053.32       490.93    12_544.25       0.0596             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             12_053.32       491.07    12_544.39       0.0564             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             12_053.32       497.05    12_550.37       0.0482             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            12_053.32       589.65    12_642.97       0.3073          1.0036         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            12_053.32       693.96    12_747.28       0.4686          1.0020         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            12_053.32       595.86    12_649.18       0.2925          1.0039         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            12_053.32       684.57    12_737.89       0.4463          1.0022         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            12_053.32       601.88    12_655.20       0.2500          1.0046         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            12_053.32       700.73    12_754.05       0.3851          1.0027         4.67
IVF-Binary-512-nl316-pca (self)                       12_053.32     1_898.76    13_952.08       0.3346          1.2038         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          24_696.52       870.44    25_566.96       0.0801             NaN         8.41
IVF-Binary-1024-nl158-np12-rf0-random (query)         24_696.52       895.12    25_591.64       0.0753             NaN         8.41
IVF-Binary-1024-nl158-np17-rf0-random (query)         24_696.52       913.30    25_609.82       0.0734             NaN         8.41
IVF-Binary-1024-nl158-np7-rf10-random (query)         24_696.52       976.49    25_673.01       0.4066          1.0027         8.41
IVF-Binary-1024-nl158-np7-rf20-random (query)         24_696.52     1_046.69    25_743.21       0.5989          1.0013         8.41
IVF-Binary-1024-nl158-np12-rf10-random (query)        24_696.52       991.88    25_688.40       0.3864          1.0030         8.41
IVF-Binary-1024-nl158-np12-rf20-random (query)        24_696.52     1_161.82    25_858.34       0.5728          1.0015         8.41
IVF-Binary-1024-nl158-np17-rf10-random (query)        24_696.52     1_012.10    25_708.62       0.3781          1.0032         8.41
IVF-Binary-1024-nl158-np17-rf20-random (query)        24_696.52     1_114.48    25_811.00       0.5612          1.0017         8.41
IVF-Binary-1024-nl158-random (self)                   24_696.52     3_183.65    27_880.18       0.9356          1.0062         8.41
IVF-Binary-1024-nl223-np11-rf0-random (query)         22_202.95       875.79    23_078.74       0.0801             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         22_202.95       892.86    23_095.81       0.0771             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         22_202.95       896.15    23_099.10       0.0744             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        22_202.95       971.98    23_174.93       0.4065          1.0029         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        22_202.95     1_048.26    23_251.21       0.5973          1.0015         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        22_202.95       970.41    23_173.36       0.3955          1.0030         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        22_202.95     1_078.84    23_281.79       0.5846          1.0016         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        22_202.95       988.16    23_191.11       0.3850          1.0031         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        22_202.95     1_084.48    23_287.43       0.5716          1.0016         8.54
IVF-Binary-1024-nl223-random (self)                   22_202.95     3_118.77    25_321.72       0.9333          1.0066         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         22_261.33       889.24    23_150.58       0.0802             NaN         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)         22_261.33       893.13    23_154.47       0.0783             NaN         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)         22_261.33       910.90    23_172.23       0.0752             NaN         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)        22_261.33       978.16    23_239.49       0.4081          1.0029         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)        22_261.33     1_067.55    23_328.88       0.5998          1.0015         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)        22_261.33       979.24    23_240.57       0.4016          1.0030         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)        22_261.33     1_071.79    23_333.12       0.5922          1.0015         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)        22_261.33     1_009.25    23_270.59       0.3899          1.0031         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)        22_261.33     1_093.71    23_355.04       0.5780          1.0016         8.72
IVF-Binary-1024-nl316-random (self)                   22_261.33     3_147.21    25_408.54       0.9338          1.0065         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)             25_775.56       895.97    26_671.54       0.0964             NaN         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            25_775.56       922.16    26_697.73       0.0896             NaN         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            25_775.56       950.29    26_725.86       0.0865             NaN         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            25_775.56       987.56    26_763.12       0.4410          1.0022         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            25_775.56     1_085.42    26_860.98       0.6270          1.0011         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           25_775.56     1_099.68    26_875.24       0.4120          1.0025         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           25_775.56     1_112.48    26_888.04       0.5901          1.0013         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           25_775.56     1_043.27    26_818.83       0.3992          1.0027         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           25_775.56     1_148.74    26_924.31       0.5720          1.0014         8.42
IVF-Binary-1024-nl158-pca (self)                      25_775.56     3_298.11    29_073.67       0.7126          1.0421         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            23_297.31       902.71    24_200.02       0.0975             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            23_297.31       907.17    24_204.48       0.0942             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            23_297.31       931.53    24_228.84       0.0902             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           23_297.31       991.04    24_288.36       0.4463          1.0022         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           23_297.31     1_079.23    24_376.55       0.6354          1.0011         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           23_297.31     1_008.86    24_306.17       0.4329          1.0024         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           23_297.31     1_096.25    24_393.57       0.6184          1.0012         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           23_297.31     1_016.38    24_313.70       0.4167          1.0025         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           23_297.31     1_129.55    24_426.87       0.5972          1.0013         8.54
IVF-Binary-1024-nl223-pca (self)                      23_297.31     3_222.88    26_520.20       0.7111          1.0429         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            23_396.43       918.47    24_314.91       0.0984             NaN         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            23_396.43       917.47    24_313.91       0.0964             NaN         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            23_396.43       938.38    24_334.81       0.0920             NaN         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           23_396.43     1_007.81    24_404.25       0.4496          1.0022         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           23_396.43     1_099.74    24_496.17       0.6394          1.0011         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           23_396.43     1_012.58    24_409.02       0.4419          1.0023         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           23_396.43     1_117.76    24_514.20       0.6303          1.0011         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           23_396.43     1_033.73    24_430.17       0.4245          1.0024         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           23_396.43     1_133.82    24_530.25       0.6075          1.0012         8.73
IVF-Binary-1024-nl316-pca (self)                      23_396.43     3_261.62    26_658.05       0.7118          1.0427         8.73
IVF-Binary-512-nl158-np7-rf0-signed (query)           13_733.76       450.93    14_184.69       0.0550             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-signed (query)          13_733.76       486.64    14_220.40       0.0503             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-signed (query)          13_733.76       480.60    14_214.36       0.0485             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-signed (query)          13_733.76       546.72    14_280.47       0.3197          1.0037         4.36
IVF-Binary-512-nl158-np7-rf20-signed (query)          13_733.76       642.01    14_375.77       0.5000          1.0019         4.36
IVF-Binary-512-nl158-np12-rf10-signed (query)         13_733.76       571.09    14_304.85       0.2982          1.0041         4.36
IVF-Binary-512-nl158-np12-rf20-signed (query)         13_733.76       662.77    14_396.52       0.4687          1.0022         4.36
IVF-Binary-512-nl158-np17-rf10-signed (query)         13_733.76       585.55    14_319.31       0.2893          1.0043         4.36
IVF-Binary-512-nl158-np17-rf20-signed (query)         13_733.76       686.59    14_420.35       0.4553          1.0024         4.36
IVF-Binary-512-nl158-signed (self)                    13_733.76     1_783.12    15_516.88       0.8734          1.0152         4.36
IVF-Binary-512-nl223-np11-rf0-signed (query)          11_310.29       460.63    11_770.93       0.0551             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-signed (query)          11_310.29       463.73    11_774.03       0.0524             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-signed (query)          11_310.29       477.69    11_787.98       0.0500             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-signed (query)         11_310.29       552.71    11_863.01       0.3198          1.0039         4.49
IVF-Binary-512-nl223-np11-rf20-signed (query)         11_310.29       638.75    11_949.04       0.4995          1.0021         4.49
IVF-Binary-512-nl223-np14-rf10-signed (query)         11_310.29       556.77    11_867.06       0.3078          1.0041         4.49
IVF-Binary-512-nl223-np14-rf20-signed (query)         11_310.29       657.37    11_967.66       0.4831          1.0022         4.49
IVF-Binary-512-nl223-np21-rf10-signed (query)         11_310.29       585.38    11_895.67       0.2965          1.0042         4.49
IVF-Binary-512-nl223-np21-rf20-signed (query)         11_310.29       666.91    11_977.20       0.4669          1.0023         4.49
IVF-Binary-512-nl223-signed (self)                    11_310.29     1_743.08    13_053.38       0.8701          1.0161         4.49
IVF-Binary-512-nl316-np15-rf0-signed (query)          11_295.80       473.00    11_768.80       0.0552             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-signed (query)          11_295.80       487.22    11_783.02       0.0536             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-signed (query)          11_295.80       482.80    11_778.60       0.0508             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-signed (query)         11_295.80       566.93    11_862.73       0.3216          1.0039         4.67
IVF-Binary-512-nl316-np15-rf20-signed (query)         11_295.80       657.00    11_952.80       0.5021          1.0021         4.67
IVF-Binary-512-nl316-np17-rf10-signed (query)         11_295.80       571.78    11_867.58       0.3146          1.0040         4.67
IVF-Binary-512-nl316-np17-rf20-signed (query)         11_295.80       664.35    11_960.15       0.4930          1.0021         4.67
IVF-Binary-512-nl316-np25-rf10-signed (query)         11_295.80       580.05    11_875.85       0.3015          1.0041         4.67
IVF-Binary-512-nl316-np25-rf20-signed (query)         11_295.80       673.76    11_969.57       0.4742          1.0023         4.67
IVF-Binary-512-nl316-signed (self)                    11_295.80     1_776.75    13_072.55       0.8706          1.0160         4.67
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Quantisation stress data - 1024 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 1024D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        39.14    23_047.09    23_086.23       1.0000          1.0000       195.31
Exhaustive (self)                                         39.14    76_633.43    76_672.57       1.0000          1.0000       195.31
ExhaustiveBinary-256-random_no_rr (query)             12_046.43       608.34    12_654.77       0.0086             NaN         2.53
ExhaustiveBinary-256-random-rf10 (query)              12_046.43       777.10    12_823.53       0.0837          1.0101         2.53
ExhaustiveBinary-256-random-rf20 (query)              12_046.43       960.48    13_006.91       0.1557          1.0064         2.53
ExhaustiveBinary-256-random (self)                    12_046.43     2_561.38    14_607.81       0.6166          5.6565         2.53
ExhaustiveBinary-256-pca_no_rr (query)                13_240.05       607.35    13_847.40       0.0142             NaN         2.53
ExhaustiveBinary-256-pca-rf10 (query)                 13_240.05       792.22    14_032.27       0.0739          1.0100         2.53
ExhaustiveBinary-256-pca-rf20 (query)                 13_240.05       975.51    14_215.56       0.1169          1.0048         2.53
ExhaustiveBinary-256-pca (self)                       13_240.05     2_594.08    15_834.13       0.1825          1.5785         2.53
ExhaustiveBinary-512-random_no_rr (query)             23_555.93     1_161.20    24_717.13       0.0192             NaN         5.05
ExhaustiveBinary-512-random-rf10 (query)              23_555.93     1_348.50    24_904.43       0.1384          1.0052         5.05
ExhaustiveBinary-512-random-rf20 (query)              23_555.93     1_522.38    25_078.31       0.2398          1.0032         5.05
ExhaustiveBinary-512-random (self)                    23_555.93     4_373.66    27_929.59       0.7044          7.2752         5.05
ExhaustiveBinary-512-pca_no_rr (query)                24_875.66     1_158.29    26_033.95       0.0097             NaN         5.06
ExhaustiveBinary-512-pca-rf10 (query)                 24_875.66     1_324.43    26_200.09       0.0527          1.0187         5.06
ExhaustiveBinary-512-pca-rf20 (query)                 24_875.66     1_511.26    26_386.92       0.0858          1.0069         5.06
ExhaustiveBinary-512-pca (self)                       24_875.66     4_385.42    29_261.07       0.1295          2.6401         5.06
ExhaustiveBinary-1024-random_no_rr (query)            47_288.35     2_173.00    49_461.35       0.0426             NaN        10.10
ExhaustiveBinary-1024-random-rf10 (query)             47_288.35     2_326.43    49_614.78       0.2373          1.0030        10.10
ExhaustiveBinary-1024-random-rf20 (query)             47_288.35     2_550.89    49_839.24       0.3715          1.0017        10.10
ExhaustiveBinary-1024-random (self)                   47_288.35     7_690.02    54_978.37       0.7686          8.3967        10.10
ExhaustiveBinary-1024-pca_no_rr (query)               48_710.63     2_174.87    50_885.49       0.0086             NaN        10.11
ExhaustiveBinary-1024-pca-rf10 (query)                48_710.63     2_345.13    51_055.76       0.0490          1.0332        10.11
ExhaustiveBinary-1024-pca-rf20 (query)                48_710.63     2_550.65    51_261.28       0.0797          1.0112        10.11
ExhaustiveBinary-1024-pca (self)                      48_710.63     7_867.36    56_577.99       0.1033          8.4885        10.11
ExhaustiveBinary-1024-signed_no_rr (query)            47_228.55     2_164.03    49_392.58       0.0426             NaN        10.10
ExhaustiveBinary-1024-signed-rf10 (query)             47_228.55     2_318.47    49_547.01       0.2373          1.0030        10.10
ExhaustiveBinary-1024-signed-rf20 (query)             47_228.55     2_522.03    49_750.58       0.3715          1.0017        10.10
ExhaustiveBinary-1024-signed (self)                   47_228.55     7_734.91    54_963.46       0.7686          8.3967        10.10
IVF-Binary-256-nl158-np7-rf0-random (query)           17_746.98       530.96    18_277.94       0.0178             NaN         3.14
IVF-Binary-256-nl158-np12-rf0-random (query)          17_746.98       510.24    18_257.22       0.0138             NaN         3.14
IVF-Binary-256-nl158-np17-rf0-random (query)          17_746.98       513.58    18_260.56       0.0122             NaN         3.14
IVF-Binary-256-nl158-np7-rf10-random (query)          17_746.98       644.50    18_391.48       0.1535          1.0042         3.14
IVF-Binary-256-nl158-np7-rf20-random (query)          17_746.98       773.01    18_519.99       0.2799          1.0021         3.14
IVF-Binary-256-nl158-np12-rf10-random (query)         17_746.98       654.41    18_401.39       0.1278          1.0053         3.14
IVF-Binary-256-nl158-np12-rf20-random (query)         17_746.98       809.55    18_556.53       0.2376          1.0029         3.14
IVF-Binary-256-nl158-np17-rf10-random (query)         17_746.98       661.90    18_408.88       0.1162          1.0062         3.14
IVF-Binary-256-nl158-np17-rf20-random (query)         17_746.98       809.32    18_556.30       0.2180          1.0035         3.14
IVF-Binary-256-nl158-random (self)                    17_746.98     2_061.02    19_808.00       0.7825          1.0294         3.14
IVF-Binary-256-nl223-np11-rf0-random (query)          12_002.51       528.72    12_531.23       0.0184             NaN         3.40
IVF-Binary-256-nl223-np14-rf0-random (query)          12_002.51       530.24    12_532.75       0.0156             NaN         3.40
IVF-Binary-256-nl223-np21-rf0-random (query)          12_002.51       531.94    12_534.45       0.0133             NaN         3.40
IVF-Binary-256-nl223-np11-rf10-random (query)         12_002.51       711.55    12_714.06       0.1594          1.0038         3.40
IVF-Binary-256-nl223-np11-rf20-random (query)         12_002.51       819.10    12_821.61       0.2906          1.0021         3.40
IVF-Binary-256-nl223-np14-rf10-random (query)         12_002.51       665.86    12_668.37       0.1416          1.0046         3.40
IVF-Binary-256-nl223-np14-rf20-random (query)         12_002.51       804.73    12_807.24       0.2623          1.0024         3.40
IVF-Binary-256-nl223-np21-rf10-random (query)         12_002.51       678.18    12_680.70       0.1240          1.0055         3.40
IVF-Binary-256-nl223-np21-rf20-random (query)         12_002.51       821.04    12_823.56       0.2335          1.0030         3.40
IVF-Binary-256-nl223-random (self)                    12_002.51     2_086.10    14_088.61       0.7845          1.0290         3.40
IVF-Binary-256-nl316-np15-rf0-random (query)          12_060.84       563.70    12_624.54       0.0178             NaN         3.76
IVF-Binary-256-nl316-np17-rf0-random (query)          12_060.84       563.12    12_623.96       0.0165             NaN         3.76
IVF-Binary-256-nl316-np25-rf0-random (query)          12_060.84       566.46    12_627.30       0.0140             NaN         3.76
IVF-Binary-256-nl316-np15-rf10-random (query)         12_060.84       697.85    12_758.69       0.1595          1.0038         3.76
IVF-Binary-256-nl316-np15-rf20-random (query)         12_060.84       836.09    12_896.93       0.2939          1.0020         3.76
IVF-Binary-256-nl316-np17-rf10-random (query)         12_060.84       702.33    12_763.17       0.1506          1.0042         3.76
IVF-Binary-256-nl316-np17-rf20-random (query)         12_060.84       834.96    12_895.80       0.2786          1.0022         3.76
IVF-Binary-256-nl316-np25-rf10-random (query)         12_060.84       707.23    12_768.07       0.1321          1.0051         3.76
IVF-Binary-256-nl316-np25-rf20-random (query)         12_060.84       851.89    12_912.73       0.2470          1.0027         3.76
IVF-Binary-256-nl316-random (self)                    12_060.84     2_233.02    14_293.86       0.7846          1.0290         3.76
IVF-Binary-256-nl158-np7-rf0-pca (query)              19_067.02       508.61    19_575.63       0.0446             NaN         3.15
IVF-Binary-256-nl158-np12-rf0-pca (query)             19_067.02       515.87    19_582.89       0.0369             NaN         3.15
IVF-Binary-256-nl158-np17-rf0-pca (query)             19_067.02       522.32    19_589.34       0.0328             NaN         3.15
IVF-Binary-256-nl158-np7-rf10-pca (query)             19_067.02       661.77    19_728.78       0.2517          1.0019         3.15
IVF-Binary-256-nl158-np7-rf20-pca (query)             19_067.02       806.47    19_873.49       0.3958          1.0011         3.15
IVF-Binary-256-nl158-np12-rf10-pca (query)            19_067.02       663.35    19_730.37       0.2065          1.0023         3.15
IVF-Binary-256-nl158-np12-rf20-pca (query)            19_067.02       825.72    19_892.74       0.3249          1.0014         3.15
IVF-Binary-256-nl158-np17-rf10-pca (query)            19_067.02       691.68    19_758.70       0.1808          1.0026         3.15
IVF-Binary-256-nl158-np17-rf20-pca (query)            19_067.02       827.55    19_894.57       0.2839          1.0017         3.15
IVF-Binary-256-nl158-pca (self)                       19_067.02     2_143.67    21_210.68       0.2954          1.2366         3.15
IVF-Binary-256-nl223-np11-rf0-pca (query)             13_366.18       533.51    13_899.69       0.0455             NaN         3.40
IVF-Binary-256-nl223-np14-rf0-pca (query)             13_366.18       538.04    13_904.21       0.0412             NaN         3.40
IVF-Binary-256-nl223-np21-rf0-pca (query)             13_366.18       540.40    13_906.58       0.0359             NaN         3.40
IVF-Binary-256-nl223-np11-rf10-pca (query)            13_366.18       708.28    14_074.46       0.2576          1.0018         3.40
IVF-Binary-256-nl223-np11-rf20-pca (query)            13_366.18       813.84    14_180.02       0.4039          1.0011         3.40
IVF-Binary-256-nl223-np14-rf10-pca (query)            13_366.18       675.39    14_041.57       0.2331          1.0020         3.40
IVF-Binary-256-nl223-np14-rf20-pca (query)            13_366.18       826.19    14_192.37       0.3664          1.0012         3.40
IVF-Binary-256-nl223-np21-rf10-pca (query)            13_366.18       694.32    14_060.50       0.2008          1.0023         3.40
IVF-Binary-256-nl223-np21-rf20-pca (query)            13_366.18       847.84    14_214.02       0.3160          1.0015         3.40
IVF-Binary-256-nl223-pca (self)                       13_366.18     2_182.04    15_548.22       0.3168          1.2064         3.40
IVF-Binary-256-nl316-np15-rf0-pca (query)             13_436.02       564.60    14_000.61       0.0461             NaN         3.77
IVF-Binary-256-nl316-np17-rf0-pca (query)             13_436.02       574.79    14_010.81       0.0441             NaN         3.77
IVF-Binary-256-nl316-np25-rf0-pca (query)             13_436.02       583.58    14_019.60       0.0387             NaN         3.77
IVF-Binary-256-nl316-np15-rf10-pca (query)            13_436.02       713.67    14_149.69       0.2622          1.0018         3.77
IVF-Binary-256-nl316-np15-rf20-pca (query)            13_436.02       849.70    14_285.71       0.4106          1.0010         3.77
IVF-Binary-256-nl316-np17-rf10-pca (query)            13_436.02       706.48    14_142.50       0.2490          1.0019         3.77
IVF-Binary-256-nl316-np17-rf20-pca (query)            13_436.02       845.30    14_281.32       0.3908          1.0011         3.77
IVF-Binary-256-nl316-np25-rf10-pca (query)            13_436.02       721.87    14_157.89       0.2162          1.0022         3.77
IVF-Binary-256-nl316-np25-rf20-pca (query)            13_436.02       864.76    14_300.78       0.3408          1.0013         3.77
IVF-Binary-256-nl316-pca (self)                       13_436.02     2_288.87    15_724.89       0.3283          1.1939         3.77
IVF-Binary-512-nl158-np7-rf0-random (query)           29_712.29       938.95    30_651.24       0.0294             NaN         5.67
IVF-Binary-512-nl158-np12-rf0-random (query)          29_712.29       956.49    30_668.79       0.0255             NaN         5.67
IVF-Binary-512-nl158-np17-rf0-random (query)          29_712.29       962.52    30_674.82       0.0236             NaN         5.67
IVF-Binary-512-nl158-np7-rf10-random (query)          29_712.29     1_064.37    30_776.66       0.2105          1.0028         5.67
IVF-Binary-512-nl158-np7-rf20-random (query)          29_712.29     1_192.57    30_904.86       0.3621          1.0015         5.67
IVF-Binary-512-nl158-np12-rf10-random (query)         29_712.29     1_081.97    30_794.26       0.1874          1.0032         5.67
IVF-Binary-512-nl158-np12-rf20-random (query)         29_712.29     1_217.73    30_930.02       0.3266          1.0018         5.67
IVF-Binary-512-nl158-np17-rf10-random (query)         29_712.29     1_100.58    30_812.87       0.1767          1.0035         5.67
IVF-Binary-512-nl158-np17-rf20-random (query)         29_712.29     1_244.59    30_956.88       0.3087          1.0019         5.67
IVF-Binary-512-nl158-random (self)                    29_712.29     3_485.19    33_197.48       0.8789          1.0130         5.67
IVF-Binary-512-nl223-np11-rf0-random (query)          23_875.74       967.00    24_842.74       0.0304             NaN         5.92
IVF-Binary-512-nl223-np14-rf0-random (query)          23_875.74       969.92    24_845.66       0.0272             NaN         5.92
IVF-Binary-512-nl223-np21-rf0-random (query)          23_875.74       980.18    24_855.91       0.0245             NaN         5.92
IVF-Binary-512-nl223-np11-rf10-random (query)         23_875.74     1_094.14    24_969.88       0.2167          1.0026         5.92
IVF-Binary-512-nl223-np11-rf20-random (query)         23_875.74     1_217.39    25_093.13       0.3718          1.0014         5.92
IVF-Binary-512-nl223-np14-rf10-random (query)         23_875.74     1_094.76    24_970.50       0.2004          1.0029         5.92
IVF-Binary-512-nl223-np14-rf20-random (query)         23_875.74     1_230.06    25_105.80       0.3481          1.0016         5.92
IVF-Binary-512-nl223-np21-rf10-random (query)         23_875.74     1_106.84    24_982.58       0.1847          1.0032         5.92
IVF-Binary-512-nl223-np21-rf20-random (query)         23_875.74     1_254.47    25_130.21       0.3242          1.0018         5.92
IVF-Binary-512-nl223-random (self)                    23_875.74     3_516.91    27_392.65       0.8799          1.0128         5.92
IVF-Binary-512-nl316-np15-rf0-random (query)          23_946.81     1_019.37    24_966.19       0.0300             NaN         6.29
IVF-Binary-512-nl316-np17-rf0-random (query)          23_946.81     1_003.86    24_950.67       0.0284             NaN         6.29
IVF-Binary-512-nl316-np25-rf0-random (query)          23_946.81     1_011.05    24_957.86       0.0258             NaN         6.29
IVF-Binary-512-nl316-np15-rf10-random (query)         23_946.81     1_126.64    25_073.45       0.2175          1.0026         6.29
IVF-Binary-512-nl316-np15-rf20-random (query)         23_946.81     1_257.98    25_204.79       0.3751          1.0014         6.29
IVF-Binary-512-nl316-np17-rf10-random (query)         23_946.81     1_126.56    25_073.37       0.2089          1.0028         6.29
IVF-Binary-512-nl316-np17-rf20-random (query)         23_946.81     1_262.43    25_209.24       0.3623          1.0015         6.29
IVF-Binary-512-nl316-np25-rf10-random (query)         23_946.81     1_141.79    25_088.60       0.1922          1.0031         6.29
IVF-Binary-512-nl316-np25-rf20-random (query)         23_946.81     1_282.82    25_229.63       0.3357          1.0017         6.29
IVF-Binary-512-nl316-random (self)                    23_946.81     3_624.26    27_571.07       0.8804          1.0128         6.29
IVF-Binary-512-nl158-np7-rf0-pca (query)              30_924.03       951.97    31_876.00       0.0425             NaN         5.67
IVF-Binary-512-nl158-np12-rf0-pca (query)             30_924.03       976.09    31_900.12       0.0335             NaN         5.67
IVF-Binary-512-nl158-np17-rf0-pca (query)             30_924.03       981.60    31_905.64       0.0284             NaN         5.67
IVF-Binary-512-nl158-np7-rf10-pca (query)             30_924.03     1_090.42    32_014.45       0.2373          1.0020         5.67
IVF-Binary-512-nl158-np7-rf20-pca (query)             30_924.03     1_217.71    32_141.74       0.3764          1.0012         5.67
IVF-Binary-512-nl158-np12-rf10-pca (query)            30_924.03     1_100.39    32_024.42       0.1861          1.0025         5.67
IVF-Binary-512-nl158-np12-rf20-pca (query)            30_924.03     1_268.36    32_192.39       0.2964          1.0016         5.67
IVF-Binary-512-nl158-np17-rf10-pca (query)            30_924.03     1_233.55    32_157.58       0.1576          1.0029         5.67
IVF-Binary-512-nl158-np17-rf20-pca (query)            30_924.03     1_270.81    32_194.85       0.2504          1.0019         5.67
IVF-Binary-512-nl158-pca (self)                       30_924.03     3_626.06    34_550.09       0.2400          1.3093         5.67
IVF-Binary-512-nl223-np11-rf0-pca (query)             25_538.64       989.86    26_528.49       0.0430             NaN         5.93
IVF-Binary-512-nl223-np14-rf0-pca (query)             25_538.64       995.62    26_534.25       0.0383             NaN         5.93
IVF-Binary-512-nl223-np21-rf0-pca (query)             25_538.64     1_018.65    26_557.28       0.0321             NaN         5.93
IVF-Binary-512-nl223-np11-rf10-pca (query)            25_538.64     1_112.60    26_651.24       0.2425          1.0019         5.93
IVF-Binary-512-nl223-np11-rf20-pca (query)            25_538.64     1_278.08    26_816.72       0.3842          1.0011         5.93
IVF-Binary-512-nl223-np14-rf10-pca (query)            25_538.64     1_113.01    26_651.65       0.2157          1.0022         5.93
IVF-Binary-512-nl223-np14-rf20-pca (query)            25_538.64     1_264.57    26_803.21       0.3424          1.0013         5.93
IVF-Binary-512-nl223-np21-rf10-pca (query)            25_538.64     1_130.91    26_669.55       0.1799          1.0026         5.93
IVF-Binary-512-nl223-np21-rf20-pca (query)            25_538.64     1_290.43    26_829.07       0.2853          1.0016         5.93
IVF-Binary-512-nl223-pca (self)                       25_538.64     3_653.79    29_192.43       0.2639          1.2612         5.93
IVF-Binary-512-nl316-np15-rf0-pca (query)             25_602.85     1_016.73    26_619.58       0.0441             NaN         6.29
IVF-Binary-512-nl316-np17-rf0-pca (query)             25_602.85     1_010.37    26_613.23       0.0415             NaN         6.29
IVF-Binary-512-nl316-np25-rf0-pca (query)             25_602.85     1_023.71    26_626.57       0.0352             NaN         6.29
IVF-Binary-512-nl316-np15-rf10-pca (query)            25_602.85     1_144.66    26_747.52       0.2473          1.0019         6.29
IVF-Binary-512-nl316-np15-rf20-pca (query)            25_602.85     1_288.37    26_891.22       0.3903          1.0011         6.29
IVF-Binary-512-nl316-np17-rf10-pca (query)            25_602.85     1_144.31    26_747.17       0.2335          1.0020         6.29
IVF-Binary-512-nl316-np17-rf20-pca (query)            25_602.85     1_287.61    26_890.47       0.3690          1.0012         6.29
IVF-Binary-512-nl316-np25-rf10-pca (query)            25_602.85     1_186.52    26_789.38       0.1974          1.0023         6.29
IVF-Binary-512-nl316-np25-rf20-pca (query)            25_602.85     1_329.12    26_931.97       0.3133          1.0015         6.29
IVF-Binary-512-nl316-pca (self)                       25_602.85     3_775.28    29_378.13       0.2766          1.2422         6.29
IVF-Binary-1024-nl158-np7-rf0-random (query)          53_142.72     1_817.78    54_960.50       0.0554             NaN        10.72
IVF-Binary-1024-nl158-np12-rf0-random (query)         53_142.72     1_836.32    54_979.04       0.0516             NaN        10.72
IVF-Binary-1024-nl158-np17-rf0-random (query)         53_142.72     1_851.97    54_994.69       0.0495             NaN        10.72
IVF-Binary-1024-nl158-np7-rf10-random (query)         53_142.72     1_915.67    55_058.39       0.3144          1.0018        10.72
IVF-Binary-1024-nl158-np7-rf20-random (query)         53_142.72     2_054.51    55_197.23       0.4901          1.0009        10.72
IVF-Binary-1024-nl158-np12-rf10-random (query)        53_142.72     2_077.32    55_220.04       0.2972          1.0019        10.72
IVF-Binary-1024-nl158-np12-rf20-random (query)        53_142.72     2_081.47    55_224.19       0.4668          1.0010        10.72
IVF-Binary-1024-nl158-np17-rf10-random (query)        53_142.72     1_967.50    55_110.22       0.2883          1.0020        10.72
IVF-Binary-1024-nl158-np17-rf20-random (query)        53_142.72     2_112.88    55_255.60       0.4539          1.0010        10.72
IVF-Binary-1024-nl158-random (self)                   53_142.72     6_348.85    59_491.57       0.9463          1.0044        10.72
IVF-Binary-1024-nl223-np11-rf0-random (query)         47_542.87     1_865.32    49_408.19       0.0569             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-random (query)         47_542.87     1_864.21    49_407.09       0.0538             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-random (query)         47_542.87     1_860.85    49_403.73       0.0509             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-random (query)        47_542.87     1_950.30    49_493.18       0.3214          1.0017        10.98
IVF-Binary-1024-nl223-np11-rf20-random (query)        47_542.87     2_075.58    49_618.45       0.4986          1.0009        10.98
IVF-Binary-1024-nl223-np14-rf10-random (query)        47_542.87     1_952.05    49_494.93       0.3081          1.0018        10.98
IVF-Binary-1024-nl223-np14-rf20-random (query)        47_542.87     2_109.64    49_652.52       0.4819          1.0009        10.98
IVF-Binary-1024-nl223-np21-rf10-random (query)        47_542.87     2_026.85    49_569.72       0.2960          1.0019        10.98
IVF-Binary-1024-nl223-np21-rf20-random (query)        47_542.87     2_150.78    49_693.65       0.4651          1.0010        10.98
IVF-Binary-1024-nl223-random (self)                   47_542.87     6_449.80    53_992.67       0.9468          1.0044        10.98
IVF-Binary-1024-nl316-np15-rf0-random (query)         47_439.02     1_874.59    49_313.61       0.0566             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-random (query)         47_439.02     1_891.77    49_330.79       0.0550             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-random (query)         47_439.02     1_889.17    49_328.20       0.0519             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-random (query)        47_439.02     2_000.16    49_439.18       0.3220          1.0017        11.34
IVF-Binary-1024-nl316-np15-rf20-random (query)        47_439.02     2_120.99    49_560.01       0.5007          1.0009        11.34
IVF-Binary-1024-nl316-np17-rf10-random (query)        47_439.02     2_005.82    49_444.84       0.3151          1.0018        11.34
IVF-Binary-1024-nl316-np17-rf20-random (query)        47_439.02     2_130.09    49_569.12       0.4918          1.0009        11.34
IVF-Binary-1024-nl316-np25-rf10-random (query)        47_439.02     2_015.88    49_454.90       0.3018          1.0019        11.34
IVF-Binary-1024-nl316-np25-rf20-random (query)        47_439.02     2_152.69    49_591.71       0.4739          1.0010        11.34
IVF-Binary-1024-nl316-random (self)                   47_439.02     6_516.54    53_955.56       0.9472          1.0043        11.34
IVF-Binary-1024-nl158-np7-rf0-pca (query)             54_640.46     1_848.71    56_489.17       0.0476             NaN        10.73
IVF-Binary-1024-nl158-np12-rf0-pca (query)            54_640.46     1_871.61    56_512.07       0.0364             NaN        10.73
IVF-Binary-1024-nl158-np17-rf0-pca (query)            54_640.46     1_882.86    56_523.32       0.0304             NaN        10.73
IVF-Binary-1024-nl158-np7-rf10-pca (query)            54_640.46     1_964.96    56_605.42       0.2582          1.0019        10.73
IVF-Binary-1024-nl158-np7-rf20-pca (query)            54_640.46     2_100.25    56_740.71       0.4018          1.0011        10.73
IVF-Binary-1024-nl158-np12-rf10-pca (query)           54_640.46     1_987.22    56_627.69       0.1973          1.0025        10.73
IVF-Binary-1024-nl158-np12-rf20-pca (query)           54_640.46     2_131.31    56_771.78       0.3113          1.0015        10.73
IVF-Binary-1024-nl158-np17-rf10-pca (query)           54_640.46     2_027.19    56_667.65       0.1626          1.0030        10.73
IVF-Binary-1024-nl158-np17-rf20-pca (query)           54_640.46     2_331.34    56_971.80       0.2580          1.0019        10.73
IVF-Binary-1024-nl158-pca (self)                      54_640.46     6_576.07    61_216.53       0.2039          1.3825        10.73
IVF-Binary-1024-nl223-np11-rf0-pca (query)            48_727.16     1_876.34    50_603.51       0.0490             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-pca (query)            48_727.16     1_877.97    50_605.13       0.0430             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-pca (query)            48_727.16     1_895.50    50_622.67       0.0349             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-pca (query)           48_727.16     2_014.86    50_742.02       0.2631          1.0018        10.98
IVF-Binary-1024-nl223-np11-rf20-pca (query)           48_727.16     2_136.64    50_863.80       0.4107          1.0011        10.98
IVF-Binary-1024-nl223-np14-rf10-pca (query)           48_727.16     2_056.92    50_784.08       0.2322          1.0021        10.98
IVF-Binary-1024-nl223-np14-rf20-pca (query)           48_727.16     2_141.95    50_869.11       0.3647          1.0013        10.98
IVF-Binary-1024-nl223-np21-rf10-pca (query)           48_727.16     2_014.29    50_741.46       0.1896          1.0026        10.98
IVF-Binary-1024-nl223-np21-rf20-pca (query)           48_727.16     2_164.81    50_891.97       0.2994          1.0016        10.98
IVF-Binary-1024-nl223-pca (self)                      48_727.16     6_591.03    55_318.19       0.2292          1.3129        10.98
IVF-Binary-1024-nl316-np15-rf0-pca (query)            48_673.14     1_910.27    50_583.41       0.0502             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-pca (query)            48_673.14     1_935.92    50_609.05       0.0470             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-pca (query)            48_673.14     1_925.97    50_599.11       0.0390             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-pca (query)           48_673.14     2_009.01    50_682.15       0.2684          1.0018        11.34
IVF-Binary-1024-nl316-np15-rf20-pca (query)           48_673.14     2_173.51    50_846.65       0.4188          1.0010        11.34
IVF-Binary-1024-nl316-np17-rf10-pca (query)           48_673.14     2_033.74    50_706.88       0.2524          1.0019        11.34
IVF-Binary-1024-nl316-np17-rf20-pca (query)           48_673.14     2_169.20    50_842.33       0.3947          1.0011        11.34
IVF-Binary-1024-nl316-np25-rf10-pca (query)           48_673.14     2_033.76    50_706.90       0.2101          1.0023        11.34
IVF-Binary-1024-nl316-np25-rf20-pca (query)           48_673.14     2_193.93    50_867.06       0.3316          1.0014        11.34
IVF-Binary-1024-nl316-pca (self)                      48_673.14     6_669.06    55_342.19       0.2428          1.2866        11.34
IVF-Binary-1024-nl158-np7-rf0-signed (query)          53_265.87     1_845.26    55_111.13       0.0554             NaN        10.72
IVF-Binary-1024-nl158-np12-rf0-signed (query)         53_265.87     1_837.41    55_103.28       0.0516             NaN        10.72
IVF-Binary-1024-nl158-np17-rf0-signed (query)         53_265.87     1_853.01    55_118.88       0.0495             NaN        10.72
IVF-Binary-1024-nl158-np7-rf10-signed (query)         53_265.87     1_923.95    55_189.82       0.3144          1.0018        10.72
IVF-Binary-1024-nl158-np7-rf20-signed (query)         53_265.87     2_075.73    55_341.60       0.4901          1.0009        10.72
IVF-Binary-1024-nl158-np12-rf10-signed (query)        53_265.87     1_972.76    55_238.63       0.2972          1.0019        10.72
IVF-Binary-1024-nl158-np12-rf20-signed (query)        53_265.87     2_082.91    55_348.78       0.4668          1.0010        10.72
IVF-Binary-1024-nl158-np17-rf10-signed (query)        53_265.87     1_979.69    55_245.56       0.2883          1.0020        10.72
IVF-Binary-1024-nl158-np17-rf20-signed (query)        53_265.87     2_121.17    55_387.04       0.4539          1.0010        10.72
IVF-Binary-1024-nl158-signed (self)                   53_265.87     6_377.62    59_643.49       0.9463          1.0044        10.72
IVF-Binary-1024-nl223-np11-rf0-signed (query)         47_333.46     1_866.13    49_199.59       0.0569             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-signed (query)         47_333.46     1_847.43    49_180.90       0.0538             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-signed (query)         47_333.46     1_856.41    49_189.88       0.0509             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-signed (query)        47_333.46     1_968.52    49_301.98       0.3214          1.0017        10.98
IVF-Binary-1024-nl223-np11-rf20-signed (query)        47_333.46     2_106.63    49_440.09       0.4986          1.0009        10.98
IVF-Binary-1024-nl223-np14-rf10-signed (query)        47_333.46     1_954.11    49_287.57       0.3081          1.0018        10.98
IVF-Binary-1024-nl223-np14-rf20-signed (query)        47_333.46     2_128.91    49_462.37       0.4819          1.0009        10.98
IVF-Binary-1024-nl223-np21-rf10-signed (query)        47_333.46     1_986.77    49_320.23       0.2960          1.0019        10.98
IVF-Binary-1024-nl223-np21-rf20-signed (query)        47_333.46     2_132.63    49_466.10       0.4651          1.0010        10.98
IVF-Binary-1024-nl223-signed (self)                   47_333.46     6_869.21    54_202.68       0.9468          1.0044        10.98
IVF-Binary-1024-nl316-np15-rf0-signed (query)         47_513.89     1_884.24    49_398.13       0.0566             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-signed (query)         47_513.89     1_873.30    49_387.19       0.0550             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-signed (query)         47_513.89     1_900.58    49_414.46       0.0519             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-signed (query)        47_513.89     2_001.67    49_515.55       0.3220          1.0017        11.34
IVF-Binary-1024-nl316-np15-rf20-signed (query)        47_513.89     2_119.73    49_633.62       0.5007          1.0009        11.34
IVF-Binary-1024-nl316-np17-rf10-signed (query)        47_513.89     1_989.48    49_503.37       0.3151          1.0018        11.34
IVF-Binary-1024-nl316-np17-rf20-signed (query)        47_513.89     2_136.77    49_650.66       0.4918          1.0009        11.34
IVF-Binary-1024-nl316-np25-rf10-signed (query)        47_513.89     2_011.88    49_525.77       0.3018          1.0019        11.34
IVF-Binary-1024-nl316-np25-rf20-signed (query)        47_513.89     2_150.63    49_664.52       0.4739          1.0010        11.34
IVF-Binary-1024-nl316-signed (self)                   47_513.89     6_493.56    54_007.45       0.9472          1.0043        11.34
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
Exhaustive (query)                                        10.29     4_362.68     4_372.97       1.0000          1.0000        48.83
Exhaustive (self)                                         10.29    14_702.93    14_713.22       1.0000          1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                           1_474.05       823.76     2_297.81       0.6105             NaN         2.84
ExhaustiveRaBitQ-rf5 (query)                           1_474.05       869.92     2_343.97       0.9692          1.0007         2.84
ExhaustiveRaBitQ-rf10 (query)                          1_474.05       928.79     2_402.84       0.9968          1.0001         2.84
ExhaustiveRaBitQ-rf20 (query)                          1_474.05     1_029.21     2_503.26       0.9999          1.0000         2.84
ExhaustiveRaBitQ (self)                                1_474.05     3_101.69     4_575.74       0.9967          1.0001         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       2_158.28       260.52     2_418.80       0.6023             NaN         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      2_158.28       415.17     2_573.45       0.6110             NaN         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      2_158.28       557.71     2_715.99       0.6120             NaN         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      2_158.28       357.80     2_516.09       0.9495          1.0025         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      2_158.28       433.99     2_592.27       0.9517          1.0025         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     2_158.28       516.95     2_675.24       0.9905          1.0003         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     2_158.28       601.22     2_759.50       0.9935          1.0003         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     2_158.28       673.06     2_831.34       0.9965          1.0001         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     2_158.28       764.48     2_922.77       0.9997          1.0000         2.89
IVF-RaBitQ-nl158 (self)                                2_158.28     2_553.01     4_711.29       0.9997          1.0000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_144.21       337.31     1_481.52       0.6111             NaN         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_144.21       422.77     1_566.98       0.6128             NaN         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_144.21       620.19     1_764.40       0.6133             NaN         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_144.21       437.56     1_581.77       0.9830          1.0007         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_144.21       520.18     1_664.39       0.9856          1.0007         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_144.21       525.83     1_670.04       0.9938          1.0002         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_144.21       624.47     1_768.68       0.9967          1.0001         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_144.21       731.79     1_876.00       0.9968          1.0001         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_144.21       828.19     1_972.40       0.9999          1.0000         2.95
IVF-RaBitQ-nl223 (self)                                1_144.21     2_767.18     3_911.39       0.9999          1.0000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_307.64       414.28     1_721.92       0.6130             NaN         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_307.64       462.15     1_769.79       0.6141             NaN         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_307.64       669.50     1_977.14       0.6147             NaN         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_307.64       519.72     1_827.35       0.9826          1.0007         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_307.64       602.46     1_910.09       0.9850          1.0006         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_307.64       569.23     1_876.87       0.9903          1.0003         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_307.64       664.20     1_971.84       0.9930          1.0003         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_307.64       799.13     2_106.76       0.9970          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_307.64       876.42     2_184.06       0.9999          1.0000         3.04
IVF-RaBitQ-nl316 (self)                                1_307.64     2_902.21     4_209.85       0.9999          1.0000         3.04
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
Exhaustive (query)                                        20.18    10_126.88    10_147.05       1.0000          1.0000        97.66
Exhaustive (self)                                         20.18    33_790.23    33_810.41       1.0000          1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           4_162.04     2_401.33     6_563.38       0.6297             NaN         5.23
ExhaustiveRaBitQ-rf5 (query)                           4_162.04     2_459.28     6_621.32       0.9763          1.0003         5.23
ExhaustiveRaBitQ-rf10 (query)                          4_162.04     2_531.44     6_693.49       0.9979          1.0000         5.23
ExhaustiveRaBitQ-rf20 (query)                          4_162.04     2_661.70     6_823.74       0.9999          1.0000         5.23
ExhaustiveRaBitQ (self)                                4_162.04     8_506.29    12_668.34       0.9979          1.0000         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_592.47       784.96     6_377.43       0.6167             NaN         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_592.47     1_308.08     6_900.55       0.6289             NaN         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_592.47     1_767.09     7_359.56       0.6304             NaN         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_592.47       921.25     6_513.72       0.9462          1.0021         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_592.47     1_001.88     6_594.35       0.9476          1.0021         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_592.47     1_416.99     7_009.46       0.9901          1.0003         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_592.47     1_523.69     7_116.16       0.9919          1.0003         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_592.47     1_926.89     7_519.36       0.9980          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_592.47     2_113.99     7_706.46       0.9999          1.0000         5.32
IVF-RaBitQ-nl158 (self)                                5_592.47     6_809.45    12_401.92       0.9999          1.0000         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_315.44     1_100.93     4_416.37       0.6279             NaN         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_315.44     1_365.87     4_681.31       0.6306             NaN         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_315.44     2_013.76     5_329.20       0.6308             NaN         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_315.44     1_201.13     4_516.57       0.9852          1.0005         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_315.44     1_317.11     4_632.55       0.9869          1.0004         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_315.44     1_537.87     4_853.31       0.9970          1.0001         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_315.44     1_588.08     4_903.52       0.9989          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_315.44     2_285.48     5_600.92       0.9979          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_315.44     2_245.91     5_561.35       0.9999          1.0000         5.44
IVF-RaBitQ-nl223 (self)                                3_315.44     7_458.98    10_774.42       0.9999          1.0000         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      3_664.77     1_408.40     5_073.17       0.6304             NaN         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      3_664.77     1_594.62     5_259.39       0.6314             NaN         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      3_664.77     2_286.08     5_950.84       0.6316             NaN         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     3_664.77     1_552.35     5_217.11       0.9911          1.0003         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     3_664.77     1_740.51     5_405.28       0.9926          1.0002         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     3_664.77     1_724.10     5_388.86       0.9966          1.0001         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     3_664.77     1_832.83     5_497.60       0.9982          1.0001         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     3_664.77     2_445.33     6_110.10       0.9982          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     3_664.77     2_540.50     6_205.27       0.9999          1.0000         5.63
IVF-RaBitQ-nl316 (self)                                3_664.77     8_664.76    12_329.53       0.9999          1.0000         5.63
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 1024 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 1024D - IVF-RaBitQ
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        41.88    22_916.42    22_958.30       1.0000          1.0000       195.31
Exhaustive (self)                                         41.88    76_490.75    76_532.63       1.0000          1.0000       195.31
ExhaustiveRaBitQ-rf0 (query)                          14_573.00     9_850.23    24_423.23       0.0387             NaN        11.50
ExhaustiveRaBitQ-rf5 (query)                          14_573.00    10_405.32    24_978.33       0.1120          1.1672        11.50
ExhaustiveRaBitQ-rf10 (query)                         14_573.00     9_747.90    24_320.91       0.1905          1.1377        11.50
ExhaustiveRaBitQ-rf20 (query)                         14_573.00     9_909.52    24_482.52       0.3240          1.1093        11.50
ExhaustiveRaBitQ (self)                               14_573.00    32_488.55    47_061.56       0.1897          1.1378        11.50
IVF-RaBitQ-nl158-np7-rf0 (query)                      18_040.70     3_121.10    21_161.80       0.0382             NaN        11.68
IVF-RaBitQ-nl158-np12-rf0 (query)                     18_040.70     5_221.01    23_261.71       0.0380             NaN        11.68
IVF-RaBitQ-nl158-np17-rf0 (query)                     18_040.70     7_341.33    25_382.03       0.0378             NaN        11.68
IVF-RaBitQ-nl158-np7-rf10 (query)                     18_040.70     3_193.99    21_234.69       0.1949          1.1347        11.68
IVF-RaBitQ-nl158-np7-rf20 (query)                     18_040.70     3_341.74    21_382.44       0.3334          1.1053        11.68
IVF-RaBitQ-nl158-np12-rf10 (query)                    18_040.70     5_237.41    23_278.11       0.1894          1.1370        11.68
IVF-RaBitQ-nl158-np12-rf20 (query)                    18_040.70     5_425.91    23_466.61       0.3252          1.1080        11.68
IVF-RaBitQ-nl158-np17-rf10 (query)                    18_040.70     7_335.22    25_375.92       0.1872          1.1378        11.68
IVF-RaBitQ-nl158-np17-rf20 (query)                    18_040.70     7_510.29    25_550.99       0.3222          1.1088        11.68
IVF-RaBitQ-nl158 (self)                               18_040.70    24_989.62    43_030.32       0.3198          1.1090        11.68
IVF-RaBitQ-nl223-np11-rf0 (query)                     12_636.07     4_648.21    17_284.28       0.0372             NaN        11.93
IVF-RaBitQ-nl223-np14-rf0 (query)                     12_636.07     5_850.71    18_486.79       0.0372             NaN        11.93
IVF-RaBitQ-nl223-np21-rf0 (query)                     12_636.07     8_713.99    21_350.07       0.0371             NaN        11.93
IVF-RaBitQ-nl223-np11-rf10 (query)                    12_636.07     4_638.38    17_274.45       0.1864          1.1366        11.93
IVF-RaBitQ-nl223-np11-rf20 (query)                    12_636.07     4_802.85    17_438.93       0.3251          1.1068        11.93
IVF-RaBitQ-nl223-np14-rf10 (query)                    12_636.07     5_809.67    18_445.74       0.1836          1.1377        11.93
IVF-RaBitQ-nl223-np14-rf20 (query)                    12_636.07     5_957.63    18_593.71       0.3207          1.1080        11.93
IVF-RaBitQ-nl223-np21-rf10 (query)                    12_636.07     8_616.45    21_252.52       0.1831          1.1380        11.93
IVF-RaBitQ-nl223-np21-rf20 (query)                    12_636.07     8_731.74    21_367.82       0.3198          1.1081        11.93
IVF-RaBitQ-nl223 (self)                               12_636.07    29_115.58    41_751.65       0.3170          1.1084        11.93
IVF-RaBitQ-nl316-np15-rf0 (query)                     13_305.59     6_172.11    19_477.70       0.0360             NaN        12.30
IVF-RaBitQ-nl316-np17-rf0 (query)                     13_305.59     6_989.90    20_295.49       0.0359             NaN        12.30
IVF-RaBitQ-nl316-np25-rf0 (query)                     13_305.59    10_217.40    23_522.99       0.0357             NaN        12.30
IVF-RaBitQ-nl316-np15-rf10 (query)                    13_305.59     6_185.75    19_491.33       0.1813          1.1373        12.30
IVF-RaBitQ-nl316-np15-rf20 (query)                    13_305.59     6_300.24    19_605.82       0.3201          1.1069        12.30
IVF-RaBitQ-nl316-np17-rf10 (query)                    13_305.59     6_936.21    20_241.79       0.1800          1.1378        12.30
IVF-RaBitQ-nl316-np17-rf20 (query)                    13_305.59     7_071.96    20_377.55       0.3176          1.1075        12.30
IVF-RaBitQ-nl316-np25-rf10 (query)                    13_305.59    10_037.58    23_343.17       0.1789          1.1382        12.30
IVF-RaBitQ-nl316-np25-rf20 (query)                    13_305.59    10_164.99    23_470.58       0.3161          1.1078        12.30
IVF-RaBitQ-nl316 (self)                               13_305.59    33_950.35    47_255.93       0.3142          1.1079        12.30
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
Exhaustive (query)                                         9.90     4_361.47     4_371.36       1.0000          1.0000        48.83
Exhaustive (self)                                          9.90    14_892.93    14_902.83       1.0000          1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                           1_452.17       818.40     2_270.57       0.7462             NaN         2.84
ExhaustiveRaBitQ-rf5 (query)                           1_452.17       872.30     2_324.48       0.9986          1.0000         2.84
ExhaustiveRaBitQ-rf10 (query)                          1_452.17       928.08     2_380.25       1.0000          1.0000         2.84
ExhaustiveRaBitQ-rf20 (query)                          1_452.17     1_030.86     2_483.04       1.0000          1.0000         2.84
ExhaustiveRaBitQ (self)                                1_452.17     3_108.07     4_560.24       1.0000          1.0000         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       2_167.90       239.13     2_407.02       0.7478             NaN         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      2_167.90       399.34     2_567.24       0.7495             NaN         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      2_167.90       546.62     2_714.51       0.7495             NaN         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      2_167.90       333.28     2_501.18       0.9930          1.0006         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      2_167.90       410.36     2_578.26       0.9930          1.0006         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     2_167.90       493.40     2_661.29       1.0000          1.0000         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     2_167.90       579.93     2_747.82       1.0000          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     2_167.90       656.19     2_824.08       1.0000          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     2_167.90       767.47     2_935.36       1.0000          1.0000         2.89
IVF-RaBitQ-nl158 (self)                                2_167.90     2_504.14     4_672.04       1.0000          1.0000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_082.73       346.01     1_428.74       0.7499             NaN         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_082.73       424.73     1_507.46       0.7507             NaN         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_082.73       627.83     1_710.57       0.7507             NaN         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_082.73       428.53     1_511.27       0.9969          1.0002         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_082.73       515.19     1_597.92       0.9969          1.0002         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_082.73       517.06     1_599.79       0.9999          1.0000         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_082.73       599.09     1_681.82       0.9999          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_082.73       715.19     1_797.92       1.0000          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_082.73       812.91     1_895.64       1.0000          1.0000         2.95
IVF-RaBitQ-nl223 (self)                                1_082.73     2_681.93     3_764.66       1.0000          1.0000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_268.41       404.90     1_673.31       0.7539             NaN         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_268.41       458.21     1_726.62       0.7542             NaN         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_268.41       663.23     1_931.64       0.7542             NaN         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_268.41       503.64     1_772.06       0.9985          1.0001         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_268.41       586.12     1_854.54       0.9985          1.0001         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_268.41       553.88     1_822.29       0.9997          1.0000         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_268.41       639.05     1_907.46       0.9997          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_268.41       764.13     2_032.54       1.0000          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_268.41       867.31     2_135.72       1.0000          1.0000         3.04
IVF-RaBitQ-nl316 (self)                                1_268.41     2_847.82     4_116.23       1.0000          1.0000         3.04
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
Exhaustive (query)                                        20.61    10_701.78    10_722.39       1.0000          1.0000        97.66
Exhaustive (self)                                         20.61    34_647.56    34_668.16       1.0000          1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           4_039.63     2_446.62     6_486.25       0.7550             NaN         5.23
ExhaustiveRaBitQ-rf5 (query)                           4_039.63     2_503.19     6_542.81       0.9989          1.0000         5.23
ExhaustiveRaBitQ-rf10 (query)                          4_039.63     2_568.46     6_608.09       0.9998          1.0000         5.23
ExhaustiveRaBitQ-rf20 (query)                          4_039.63     2_699.75     6_739.38       0.9998          1.0000         5.23
ExhaustiveRaBitQ (self)                                4_039.63     8_568.17    12_607.80       0.9998          1.0000         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_538.58       740.34     6_278.92       0.7525             NaN         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_538.58     1_237.32     6_775.90       0.7571             NaN         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_538.58     1_725.94     7_264.52       0.7571             NaN         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_538.58       853.68     6_392.26       0.9861          1.0009         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_538.58       966.04     6_504.63       0.9861          1.0009         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_538.58     1_345.30     6_883.89       0.9998          1.0000         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_538.58     1_473.59     7_012.18       0.9998          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_538.58     1_841.18     7_379.76       0.9998          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_538.58     1_962.52     7_501.10       0.9998          1.0000         5.32
IVF-RaBitQ-nl158 (self)                                5_538.58     6_524.33    12_062.92       0.9998          1.0000         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_260.62     1_082.76     4_343.38       0.7545             NaN         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_260.62     1_363.25     4_623.86       0.7558             NaN         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_260.62     2_009.99     5_270.60       0.7560             NaN         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_260.62     1_200.15     4_460.77       0.9956          1.0003         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_260.62     1_302.23     4_562.85       0.9956          1.0003         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_260.62     1_485.95     4_746.57       0.9992          1.0000         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_260.62     1_592.71     4_853.33       0.9992          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_260.62     2_122.57     5_383.18       0.9999          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_260.62     2_240.55     5_501.16       0.9999          1.0000         5.44
IVF-RaBitQ-nl223 (self)                                3_260.62     7_496.77    10_757.39       0.9998          1.0000         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      3_562.08     1_402.80     4_964.88       0.7579             NaN         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      3_562.08     1_580.26     5_142.34       0.7586             NaN         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      3_562.08     2_293.98     5_856.07       0.7588             NaN         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     3_562.08     1_525.18     5_087.26       0.9969          1.0002         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     3_562.08     1_619.23     5_181.32       0.9969          1.0002         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     3_562.08     1_718.07     5_280.15       0.9993          1.0000         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     3_562.08     1_798.56     5_360.65       0.9993          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     3_562.08     2_398.92     5_961.00       0.9999          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     3_562.08     2_524.76     6_086.84       0.9999          1.0000         5.63
IVF-RaBitQ-nl316 (self)                                3_562.08     8_388.96    11_951.04       0.9998          1.0000         5.63
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 1024 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 1024D - IVF-RaBitQ
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        41.74    23_008.71    23_050.45       1.0000          1.0000       195.31
Exhaustive (self)                                         41.74    77_480.14    77_521.88       1.0000          1.0000       195.31
ExhaustiveRaBitQ-rf0 (query)                          15_012.93     9_655.73    24_668.66       0.0314             NaN        11.50
ExhaustiveRaBitQ-rf5 (query)                          15_012.93     9_464.29    24_477.21       0.0996          1.2429        11.50
ExhaustiveRaBitQ-rf10 (query)                         15_012.93     9_536.70    24_549.62       0.1751          1.1900        11.50
ExhaustiveRaBitQ-rf20 (query)                         15_012.93     9_663.04    24_675.97       0.3082          1.1415        11.50
ExhaustiveRaBitQ (self)                               15_012.93    31_762.95    46_775.88       0.1750          1.1903        11.50
IVF-RaBitQ-nl158-np7-rf0 (query)                      18_111.42     3_057.51    21_168.93       0.0306             NaN        11.68
IVF-RaBitQ-nl158-np12-rf0 (query)                     18_111.42     5_151.65    23_263.06       0.0304             NaN        11.68
IVF-RaBitQ-nl158-np17-rf0 (query)                     18_111.42     7_224.58    25_336.00       0.0304             NaN        11.68
IVF-RaBitQ-nl158-np7-rf10 (query)                     18_111.42     3_091.65    21_203.07       0.1738          1.1891        11.68
IVF-RaBitQ-nl158-np7-rf20 (query)                     18_111.42     3_269.67    21_381.08       0.3121          1.1391        11.68
IVF-RaBitQ-nl158-np12-rf10 (query)                    18_111.42     5_119.08    23_230.50       0.1705          1.1912        11.68
IVF-RaBitQ-nl158-np12-rf20 (query)                    18_111.42     5_289.94    23_401.36       0.3060          1.1411        11.68
IVF-RaBitQ-nl158-np17-rf10 (query)                    18_111.42     7_174.47    25_285.89       0.1705          1.1912        11.68
IVF-RaBitQ-nl158-np17-rf20 (query)                    18_111.42     7_307.54    25_418.96       0.3060          1.1411        11.68
IVF-RaBitQ-nl158 (self)                               18_111.42    24_348.64    42_460.06       0.3062          1.1412        11.68
IVF-RaBitQ-nl223-np11-rf0 (query)                     12_789.46     4_624.40    17_413.86       0.0300             NaN        11.93
IVF-RaBitQ-nl223-np14-rf0 (query)                     12_789.46     5_834.25    18_623.70       0.0298             NaN        11.93
IVF-RaBitQ-nl223-np21-rf0 (query)                     12_789.46     8_872.87    21_662.32       0.0298             NaN        11.93
IVF-RaBitQ-nl223-np11-rf10 (query)                    12_789.46     4_655.99    17_445.44       0.1703          1.1895        11.93
IVF-RaBitQ-nl223-np11-rf20 (query)                    12_789.46     4_820.03    17_609.49       0.3092          1.1387        11.93
IVF-RaBitQ-nl223-np14-rf10 (query)                    12_789.46     5_832.99    18_622.44       0.1682          1.1907        11.93
IVF-RaBitQ-nl223-np14-rf20 (query)                    12_789.46     5_989.86    18_779.31       0.3052          1.1400        11.93
IVF-RaBitQ-nl223-np21-rf10 (query)                    12_789.46     8_645.37    21_434.82       0.1681          1.1908        11.93
IVF-RaBitQ-nl223-np21-rf20 (query)                    12_789.46     8_744.24    21_533.70       0.3049          1.1401        11.93
IVF-RaBitQ-nl223 (self)                               12_789.46    29_114.70    41_904.16       0.3044          1.1403        11.93
IVF-RaBitQ-nl316-np15-rf0 (query)                     13_444.28     6_150.49    19_594.77       0.0281             NaN        12.30
IVF-RaBitQ-nl316-np17-rf0 (query)                     13_444.28     6_988.87    20_433.16       0.0280             NaN        12.30
IVF-RaBitQ-nl316-np25-rf0 (query)                     13_444.28    10_178.15    23_622.43       0.0279             NaN        12.30
IVF-RaBitQ-nl316-np15-rf10 (query)                    13_444.28     6_151.82    19_596.10       0.1656          1.1901        12.30
IVF-RaBitQ-nl316-np15-rf20 (query)                    13_444.28     6_318.15    19_762.43       0.3074          1.1378        12.30
IVF-RaBitQ-nl316-np17-rf10 (query)                    13_444.28     6_941.33    20_385.61       0.1642          1.1908        12.30
IVF-RaBitQ-nl316-np17-rf20 (query)                    13_444.28     7_046.85    20_491.13       0.3048          1.1386        12.30
IVF-RaBitQ-nl316-np25-rf10 (query)                    13_444.28    10_007.43    23_451.71       0.1634          1.1912        12.30
IVF-RaBitQ-nl316-np25-rf20 (query)                    13_444.28    10_129.79    23_574.07       0.3034          1.1390        12.30
IVF-RaBitQ-nl316 (self)                               13_444.28    34_095.19    47_539.47       0.3025          1.1393        12.30
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
Exhaustive (query)                                         6.71     4_425.42     4_432.13       1.0000          1.0000        48.83
Exhaustive (self)                                          6.71    14_769.17    14_775.87       1.0000          1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                           1_214.89     1_113.15     2_328.05       0.3458             NaN         2.84
ExhaustiveRaBitQ-rf5 (query)                           1_214.89     1_182.89     2_397.79       0.7547          1.0011         2.84
ExhaustiveRaBitQ-rf10 (query)                          1_214.89     1_241.55     2_456.44       0.8884          1.0004         2.84
ExhaustiveRaBitQ-rf20 (query)                          1_214.89     1_371.98     2_586.88       0.9630          1.0001         2.84
ExhaustiveRaBitQ (self)                                1_214.89     4_140.73     5_355.63       1.0000          1.0000         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_883.79       323.55     2_207.34       0.3617             NaN         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_883.79       551.62     2_435.41       0.3614             NaN         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_883.79       777.25     2_661.03       0.3612             NaN         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_883.79       421.05     2_304.83       0.9035          1.0003         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_883.79       488.93     2_372.72       0.9688          1.0001         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_883.79       655.14     2_538.93       0.9056          1.0003         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_883.79       739.48     2_623.26       0.9718          1.0001         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_883.79       911.22     2_795.01       0.9059          1.0003         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_883.79     1_004.74     2_888.52       0.9722          1.0001         2.89
IVF-RaBitQ-nl158 (self)                                1_883.79     3_377.84     5_261.63       1.0000          1.0000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                        715.11       361.55     1_076.66       0.4010             NaN         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                        715.11       427.28     1_142.39       0.4008             NaN         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                        715.11       619.12     1_334.23       0.4004             NaN         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                       715.11       425.57     1_140.68       0.9356          1.0002         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                       715.11       497.64     1_212.75       0.9845          1.0000         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                       715.11       514.67     1_229.78       0.9359          1.0002         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                       715.11       587.49     1_302.60       0.9850          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                       715.11       722.62     1_437.73       0.9360          1.0002         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                       715.11       808.40     1_523.50       0.9854          1.0000         2.95
IVF-RaBitQ-nl223 (self)                                  715.11     2_645.09     3_360.20       1.0000          1.0000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                        701.48       408.54     1_110.02       0.4148             NaN         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                        701.48       460.16     1_161.65       0.4147             NaN         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                        701.48       668.18     1_369.66       0.4144             NaN         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                       701.48       504.46     1_205.94       0.9437          1.0002         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                       701.48       580.45     1_281.93       0.9877          1.0000         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                       701.48       554.10     1_255.59       0.9436          1.0002         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                       701.48       649.87     1_351.35       0.9878          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                       701.48       762.25     1_463.74       0.9435          1.0002         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                       701.48       849.79     1_551.27       0.9881          1.0000         3.04
IVF-RaBitQ-nl316 (self)                                  701.48     2_796.12     3_497.60       1.0000          1.0000         3.04
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
Exhaustive (query)                                        20.32    10_095.38    10_115.70       1.0000          1.0000        97.66
Exhaustive (self)                                         20.32    34_076.21    34_096.53       1.0000          1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           3_606.82     3_031.49     6_638.31       0.3332             NaN         5.23
ExhaustiveRaBitQ-rf5 (query)                           3_606.82     3_089.66     6_696.49       0.7285          1.0006         5.23
ExhaustiveRaBitQ-rf10 (query)                          3_606.82     3_159.91     6_766.73       0.8641          1.0002         5.23
ExhaustiveRaBitQ-rf20 (query)                          3_606.82     3_306.12     6_912.94       0.9459          1.0001         5.23
ExhaustiveRaBitQ (self)                                3_606.82    10_587.05    14_193.87       0.9996          1.0001         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       4_891.74       800.26     5_692.00       0.3606             NaN         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      4_891.74     1_355.77     6_247.51       0.3591             NaN         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      4_891.74     1_910.82     6_802.56       0.3586             NaN         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      4_891.74       913.36     5_805.10       0.8886          1.0002         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      4_891.74     1_006.16     5_897.90       0.9556          1.0001         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     4_891.74     1_465.00     6_356.73       0.8908          1.0002         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     4_891.74     1_582.13     6_473.87       0.9606          1.0001         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     4_891.74     2_036.10     6_927.84       0.8904          1.0002         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     4_891.74     2_149.09     7_040.83       0.9609          1.0001         5.32
IVF-RaBitQ-nl158 (self)                                4_891.74     7_167.28    12_059.01       0.9998          1.0000         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      2_442.99     1_126.74     3_569.74       0.3848             NaN         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      2_442.99     1_354.06     3_797.05       0.3842             NaN         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      2_442.99     2_002.61     4_445.60       0.3833             NaN         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     2_442.99     1_185.86     3_628.85       0.9139          1.0001         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     2_442.99     1_301.27     3_744.26       0.9706          1.0000         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     2_442.99     1_457.94     3_900.93       0.9148          1.0001         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     2_442.99     1_563.78     4_006.77       0.9729          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     2_442.99     2_103.36     4_546.36       0.9148          1.0001         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     2_442.99     2_224.11     4_667.11       0.9743          1.0000         5.44
IVF-RaBitQ-nl223 (self)                                2_442.99     7_353.38     9_796.37       0.9999          1.0000         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      2_435.76     1_398.57     3_834.33       0.3961             NaN         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      2_435.76     1_577.91     4_013.67       0.3958             NaN         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      2_435.76     2_292.38     4_728.14       0.3948             NaN         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     2_435.76     1_513.25     3_949.01       0.9223          1.0001         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     2_435.76     1_611.52     4_047.28       0.9745          1.0000         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     2_435.76     1_679.23     4_114.99       0.9227          1.0001         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     2_435.76     1_784.31     4_220.07       0.9756          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     2_435.76     2_368.69     4_804.45       0.9232          1.0001         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     2_435.76     2_487.76     4_923.52       0.9775          1.0000         5.63
IVF-RaBitQ-nl316 (self)                                2_435.76     8_264.94    10_700.70       0.9999          1.0000         5.63
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

---

<details>
<summary><b>Quantisation stress data - 1024 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 1024D - IVF-RaBitQ
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        38.60    23_359.06    23_397.66       1.0000          1.0000       195.31
Exhaustive (self)                                         38.60    76_526.43    76_565.04       1.0000          1.0000       195.31
ExhaustiveRaBitQ-rf0 (query)                          13_403.14    10_499.03    23_902.17       0.0237             NaN        11.50
ExhaustiveRaBitQ-rf5 (query)                          13_403.14    10_384.02    23_787.16       0.0587          1.0054        11.50
ExhaustiveRaBitQ-rf10 (query)                         13_403.14    10_453.64    23_856.78       0.0961          1.0042        11.50
ExhaustiveRaBitQ-rf20 (query)                         13_403.14    10_621.75    24_024.89       0.1705          1.0031        11.50
ExhaustiveRaBitQ (self)                               13_403.14    34_930.58    48_333.72       0.2337          1.5506        11.50
IVF-RaBitQ-nl158-np7-rf0 (query)                      16_543.52     3_070.84    19_614.35       0.0263             NaN        11.68
IVF-RaBitQ-nl158-np12-rf0 (query)                     16_543.52     5_218.64    21_762.15       0.0260             NaN        11.68
IVF-RaBitQ-nl158-np17-rf0 (query)                     16_543.52     7_383.01    23_926.53       0.0259             NaN        11.68
IVF-RaBitQ-nl158-np7-rf10 (query)                     16_543.52     3_165.31    19_708.83       0.1106          1.0031        11.68
IVF-RaBitQ-nl158-np7-rf20 (query)                     16_543.52     3_286.01    19_829.52       0.2013          1.0022        11.68
IVF-RaBitQ-nl158-np12-rf10 (query)                    16_543.52     5_213.08    21_756.60       0.1051          1.0034        11.68
IVF-RaBitQ-nl158-np12-rf20 (query)                    16_543.52     5_386.16    21_929.67       0.1874          1.0024        11.68
IVF-RaBitQ-nl158-np17-rf10 (query)                    16_543.52     7_331.42    23_874.94       0.1030          1.0035        11.68
IVF-RaBitQ-nl158-np17-rf20 (query)                    16_543.52     7_457.36    24_000.88       0.1822          1.0026        11.68
IVF-RaBitQ-nl158 (self)                               16_543.52    24_916.23    41_459.74       0.4799          1.2475        11.68
IVF-RaBitQ-nl223-np11-rf0 (query)                     10_736.65     4_576.68    15_313.33       0.0277             NaN        11.93
IVF-RaBitQ-nl223-np14-rf0 (query)                     10_736.65     5_827.27    16_563.92       0.0275             NaN        11.93
IVF-RaBitQ-nl223-np21-rf0 (query)                     10_736.65     8_706.87    19_443.52       0.0272             NaN        11.93
IVF-RaBitQ-nl223-np11-rf10 (query)                    10_736.65     4_631.09    15_367.74       0.1148          1.0029        11.93
IVF-RaBitQ-nl223-np11-rf20 (query)                    10_736.65     4_763.83    15_500.48       0.2075          1.0021        11.93
IVF-RaBitQ-nl223-np14-rf10 (query)                    10_736.65     5_795.10    16_531.75       0.1120          1.0031        11.93
IVF-RaBitQ-nl223-np14-rf20 (query)                    10_736.65     5_960.84    16_697.49       0.2007          1.0022        11.93
IVF-RaBitQ-nl223-np21-rf10 (query)                    10_736.65     8_619.73    19_356.37       0.1083          1.0032        11.93
IVF-RaBitQ-nl223-np21-rf20 (query)                    10_736.65     8_752.20    19_488.85       0.1922          1.0023        11.93
IVF-RaBitQ-nl223 (self)                               10_736.65    29_315.69    40_052.34       0.5665          1.1700        11.93
IVF-RaBitQ-nl316-np15-rf0 (query)                     10_795.98     6_154.61    16_950.59       0.0289             NaN        12.30
IVF-RaBitQ-nl316-np17-rf0 (query)                     10_795.98     6_957.11    17_753.09       0.0287             NaN        12.30
IVF-RaBitQ-nl316-np25-rf0 (query)                     10_795.98    10_168.52    20_964.49       0.0284             NaN        12.30
IVF-RaBitQ-nl316-np15-rf10 (query)                    10_795.98     6_117.33    16_913.31       0.1208          1.0028        12.30
IVF-RaBitQ-nl316-np15-rf20 (query)                    10_795.98     6_702.32    17_498.30       0.2187          1.0020        12.30
IVF-RaBitQ-nl316-np17-rf10 (query)                    10_795.98     6_908.07    17_704.05       0.1191          1.0029        12.30
IVF-RaBitQ-nl316-np17-rf20 (query)                    10_795.98     7_166.19    17_962.17       0.2145          1.0020        12.30
IVF-RaBitQ-nl316-np25-rf10 (query)                    10_795.98    10_597.35    21_393.33       0.1148          1.0030        12.30
IVF-RaBitQ-nl316-np25-rf20 (query)                    10_795.98    11_435.69    22_231.67       0.2047          1.0022        12.30
IVF-RaBitQ-nl316 (self)                               10_795.98    33_928.40    44_724.38       0.6266          1.1310        12.30
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

Overall, this is a fantastic binary index that massively compresses the data,
while still allowing for great Recalls. If you need to compress your data
and reduce memory fingerprint, please, use RaBitQ!

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
