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
===================================================================================================================================
Benchmark: 50k samples, 256D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         9.83     4_173.12     4_182.95       1.0000          1.0000        48.83
Exhaustive (self)                                          9.83    14_201.87    14_211.70       1.0000          1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_580.26       247.53     2_827.79       0.1006             NaN         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_580.26       350.17     2_930.43       0.2890          1.0907         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_580.26       449.33     3_029.59       0.4246          1.0554         1.78
ExhaustiveBinary-256-random (self)                     2_580.26     1_149.52     3_729.77       0.2886          1.0907         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_884.55       252.86     3_137.41       0.2261             NaN         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_884.55       359.76     3_244.32       0.5963          1.0281         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_884.55       460.99     3_345.55       0.7219          1.0155         1.78
ExhaustiveBinary-256-pca (self)                        2_884.55     1_183.18     4_067.74       0.5956          1.0283         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_104.43       444.63     5_549.05       0.1249             NaN         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_104.43       549.83     5_654.26       0.3718          1.0678         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_104.43       651.61     5_756.04       0.5258          1.0390         3.55
ExhaustiveBinary-512-random (self)                     5_104.43     1_811.06     6_915.48       0.3724          1.0679         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_505.22       451.86     5_957.08       0.2472             NaN         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_505.22       573.29     6_078.51       0.6835          1.0189         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_505.22       668.70     6_173.92       0.8321          1.0077         3.55
ExhaustiveBinary-512-pca (self)                        5_505.22     1_852.65     7_357.87       0.6838          1.0189         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_063.86       761.82    10_825.68       0.1692             NaN         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_063.86       871.80    10_935.66       0.5028          1.0425         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_063.86       989.85    11_053.70       0.6675          1.0220         7.10
ExhaustiveBinary-1024-random (self)                   10_063.86     2_880.82    12_944.68       0.5038          1.0426         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_690.85       782.69    11_473.54       0.2755             NaN         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_690.85       886.83    11_577.68       0.7370          1.0141         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_690.85     1_027.55    11_718.40       0.8733          1.0053         7.10
ExhaustiveBinary-1024-pca (self)                      10_690.85     2_932.91    13_623.76       0.7371          1.0141         7.10
ExhaustiveBinary-256-signed_no_rr (query)              2_563.54       248.14     2_811.68       0.1006             NaN         1.78
ExhaustiveBinary-256-signed-rf10 (query)               2_563.54       350.07     2_913.61       0.2890          1.0907         1.78
ExhaustiveBinary-256-signed-rf20 (query)               2_563.54       449.67     3_013.21       0.4246          1.0554         1.78
ExhaustiveBinary-256-signed (self)                     2_563.54     1_152.38     3_715.92       0.2886          1.0907         1.78
IVF-Binary-256-nl158-np7-rf0-random (query)            4_083.02       115.20     4_198.22       0.1063             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           4_083.02       121.04     4_204.06       0.1034             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           4_083.02       126.59     4_209.60       0.1017             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           4_083.02       174.86     4_257.87       0.3128          1.0834         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           4_083.02       234.14     4_317.15       0.4564          1.0499         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          4_083.02       185.22     4_268.24       0.2990          1.0878         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          4_083.02       241.72     4_324.74       0.4375          1.0533         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          4_083.02       194.36     4_277.38       0.2914          1.0901         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          4_083.02       251.90     4_334.92       0.4284          1.0549         1.93
IVF-Binary-256-nl158-random (self)                     4_083.02       561.68     4_644.70       0.2986          1.0878         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           3_078.56       120.99     3_199.56       0.1054             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           3_078.56       125.52     3_204.09       0.1027             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           3_078.56       128.09     3_206.66       0.1013             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          3_078.56       180.50     3_259.06       0.3066          1.0843         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          3_078.56       238.16     3_316.73       0.4485          1.0506         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          3_078.56       184.59     3_263.15       0.2976          1.0874         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          3_078.56       241.24     3_319.81       0.4366          1.0529         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          3_078.56       192.79     3_271.35       0.2925          1.0893         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          3_078.56       252.81     3_331.37       0.4296          1.0544         2.00
IVF-Binary-256-nl223-random (self)                     3_078.56       568.59     3_647.15       0.2974          1.0875         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           3_257.34       126.51     3_383.86       0.1048             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           3_257.34       125.65     3_383.00       0.1036             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           3_257.34       129.83     3_387.17       0.1019             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          3_257.34       185.92     3_443.27       0.3062          1.0846         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          3_257.34       238.55     3_495.89       0.4494          1.0505         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          3_257.34       186.91     3_444.25       0.3014          1.0861         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          3_257.34       240.96     3_498.30       0.4427          1.0519         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          3_257.34       191.67     3_449.01       0.2928          1.0893         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          3_257.34       251.18     3_508.53       0.4300          1.0544         2.09
IVF-Binary-256-nl316-random (self)                     3_257.34       573.02     3_830.37       0.3014          1.0861         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               4_407.18       117.64     4_524.82       0.2338             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              4_407.18       122.70     4_529.88       0.2321             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              4_407.18       128.13     4_535.32       0.2311             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              4_407.18       184.27     4_591.45       0.6495          1.0223         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              4_407.18       242.21     4_649.39       0.7928          1.0106         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             4_407.18       191.21     4_598.40       0.6501          1.0222         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             4_407.18       256.11     4_663.29       0.7994          1.0099         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             4_407.18       202.94     4_610.13       0.6462          1.0226         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             4_407.18       267.49     4_674.68       0.7949          1.0102         1.93
IVF-Binary-256-nl158-pca (self)                        4_407.18       596.61     5_003.80       0.6500          1.0223         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              3_426.23       150.09     3_576.33       0.2328             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              3_426.23       150.90     3_577.14       0.2314             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              3_426.23       160.57     3_586.81       0.2305             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             3_426.23       196.38     3_622.61       0.6533          1.0219         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             3_426.23       253.47     3_679.71       0.8035          1.0097         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             3_426.23       193.49     3_619.72       0.6507          1.0221         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             3_426.23       254.73     3_680.96       0.8016          1.0098         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             3_426.23       202.25     3_628.49       0.6465          1.0225         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             3_426.23       266.63     3_692.87       0.7959          1.0101         2.00
IVF-Binary-256-nl223-pca (self)                        3_426.23       598.84     4_025.08       0.6509          1.0222         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              3_578.95       130.22     3_709.17       0.2329             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              3_578.95       130.04     3_708.99       0.2323             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              3_578.95       135.11     3_714.07       0.2309             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             3_578.95       195.67     3_774.62       0.6559          1.0216         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             3_578.95       250.32     3_829.28       0.8063          1.0095         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             3_578.95       195.71     3_774.67       0.6542          1.0218         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             3_578.95       253.98     3_832.93       0.8054          1.0095         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             3_578.95       204.10     3_783.05       0.6484          1.0224         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             3_578.95       264.15     3_843.10       0.7987          1.0100         2.09
IVF-Binary-256-nl316-pca (self)                        3_578.95       612.00     4_190.95       0.6541          1.0219         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_586.91       211.74     6_798.65       0.1303             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_586.91       215.03     6_801.93       0.1275             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_586.91       229.10     6_816.01       0.1259             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_586.91       291.23     6_878.13       0.3902          1.0636         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_586.91       322.58     6_909.48       0.5457          1.0363         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_586.91       279.68     6_866.59       0.3795          1.0661         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_586.91       337.47     6_924.37       0.5338          1.0380         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_586.91       292.97     6_879.87       0.3742          1.0674         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_586.91       377.27     6_964.17       0.5273          1.0389         3.71
IVF-Binary-512-nl158-random (self)                     6_586.91       889.24     7_476.15       0.3794          1.0663         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_537.88       211.05     5_748.93       0.1288             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_537.88       217.10     5_754.98       0.1268             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_537.88       225.80     5_763.68       0.1257             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_537.88       275.29     5_813.16       0.3852          1.0641         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_537.88       326.31     5_864.18       0.5432          1.0363         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_537.88       278.65     5_816.53       0.3785          1.0659         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_537.88       336.02     5_873.90       0.5340          1.0377         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_537.88       291.33     5_829.21       0.3743          1.0671         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_537.88       363.47     5_901.35       0.5286          1.0386         3.77
IVF-Binary-512-nl223-random (self)                     5_537.88       882.65     6_420.53       0.3792          1.0660         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           5_739.60       217.09     5_956.70       0.1284             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           5_739.60       219.13     5_958.73       0.1276             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           5_739.60       227.98     5_967.58       0.1259             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          5_739.60       283.74     6_023.35       0.3866          1.0640         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          5_739.60       351.15     6_090.76       0.5453          1.0361         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          5_739.60       282.89     6_022.49       0.3826          1.0649         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          5_739.60       345.13     6_084.73       0.5398          1.0369         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          5_739.60       290.00     6_029.60       0.3750          1.0670         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          5_739.60       348.75     6_088.35       0.5290          1.0386         3.86
IVF-Binary-512-nl316-random (self)                     5_739.60       891.95     6_631.55       0.3829          1.0651         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               7_081.44       223.60     7_305.05       0.2494             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              7_081.44       243.50     7_324.95       0.2482             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              7_081.44       250.12     7_331.56       0.2474             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              7_081.44       292.49     7_373.94       0.6798          1.0193         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              7_081.44       353.45     7_434.90       0.8187          1.0089         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             7_081.44       366.06     7_447.50       0.6847          1.0187         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             7_081.44       391.45     7_472.90       0.8321          1.0078         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             7_081.44       368.86     7_450.31       0.6837          1.0188         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             7_081.44       447.67     7_529.11       0.8324          1.0077         3.71
IVF-Binary-512-nl158-pca (self)                        7_081.44     1_014.95     8_096.40       0.6856          1.0188         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              6_038.46       221.09     6_259.55       0.2491             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              6_038.46       228.47     6_266.94       0.2480             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              6_038.46       237.95     6_276.42       0.2473             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             6_038.46       296.72     6_335.19       0.6861          1.0186         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             6_038.46       342.18     6_380.65       0.8324          1.0078         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             6_038.46       292.54     6_331.00       0.6850          1.0187         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             6_038.46       357.05     6_395.52       0.8333          1.0077         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             6_038.46       306.95     6_345.42       0.6838          1.0188         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             6_038.46       371.80     6_410.26       0.8325          1.0077         3.77
IVF-Binary-512-nl223-pca (self)                        6_038.46       937.76     6_976.23       0.6855          1.0188         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              6_141.56       226.94     6_368.50       0.2494             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              6_141.56       227.21     6_368.77       0.2488             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              6_141.56       238.31     6_379.87       0.2477             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             6_141.56       292.96     6_434.52       0.6879          1.0184         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             6_141.56       372.24     6_513.80       0.8338          1.0077         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             6_141.56       297.46     6_439.01       0.6873          1.0185         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             6_141.56       351.78     6_493.34       0.8347          1.0076         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             6_141.56       304.80     6_446.36       0.6839          1.0188         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             6_141.56       365.53     6_507.09       0.8323          1.0078         3.86
IVF-Binary-512-nl316-pca (self)                        6_141.56       955.52     7_097.08       0.6875          1.0186         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_449.30       390.75    11_840.05       0.1737             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_449.30       403.91    11_853.21       0.1713             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_449.30       415.57    11_864.88       0.1698             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_449.30       456.94    11_906.25       0.5112          1.0412         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_449.30       511.46    11_960.76       0.6693          1.0219         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_449.30       473.20    11_922.51       0.5072          1.0419         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_449.30       576.82    12_026.13       0.6704          1.0217         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_449.30       495.51    11_944.82       0.5038          1.0424         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_449.30       555.68    12_004.98       0.6679          1.0220         7.26
IVF-Binary-1024-nl158-random (self)                   11_449.30     1_543.50    12_992.81       0.5082          1.0419         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_513.19       396.96    10_910.16       0.1725             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_513.19       406.57    10_919.76       0.1708             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_513.19       415.48    10_928.67       0.1699             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_513.19       463.18    10_976.37       0.5122          1.0407         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_513.19       518.83    11_032.03       0.6774          1.0208         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_513.19       469.80    10_983.00       0.5077          1.0416         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_513.19       530.12    11_043.31       0.6724          1.0214         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_513.19       489.04    11_002.23       0.5043          1.0422         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_513.19       552.58    11_065.77       0.6686          1.0219         7.32
IVF-Binary-1024-nl223-random (self)                   10_513.19     1_520.33    12_033.52       0.5085          1.0417         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         10_651.00       419.26    11_070.26       0.1721             NaN         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)         10_651.00       408.87    11_059.88       0.1715             NaN         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)         10_651.00       415.57    11_066.57       0.1701             NaN         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)        10_651.00       469.46    11_120.46       0.5134          1.0407         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)        10_651.00       526.09    11_177.09       0.6788          1.0207         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)        10_651.00       471.26    11_122.26       0.5106          1.0411         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)        10_651.00       532.17    11_183.17       0.6760          1.0211         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)        10_651.00       483.45    11_134.45       0.5048          1.0422         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)        10_651.00       545.51    11_196.51       0.6689          1.0218         7.41
IVF-Binary-1024-nl316-random (self)                   10_651.00     1_544.44    12_195.45       0.5114          1.0412         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)             12_095.03       405.21    12_500.24       0.2769             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            12_095.03       418.37    12_513.40       0.2764             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            12_095.03       431.90    12_526.92       0.2757             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            12_095.03       475.27    12_570.30       0.7271          1.0150         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            12_095.03       530.54    12_625.57       0.8507          1.0069         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           12_095.03       492.06    12_587.08       0.7373          1.0140         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           12_095.03       553.61    12_648.64       0.8718          1.0054         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           12_095.03       512.07    12_607.09       0.7374          1.0140         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           12_095.03       578.15    12_673.18       0.8736          1.0053         7.26
IVF-Binary-1024-nl158-pca (self)                      12_095.03     1_603.99    13_699.02       0.7376          1.0141         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            11_200.66       413.52    11_614.17       0.2772             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            11_200.66       416.74    11_617.39       0.2763             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            11_200.66       439.58    11_640.23       0.2756             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           11_200.66       486.27    11_686.93       0.7381          1.0140         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           11_200.66       541.14    11_741.80       0.8706          1.0055         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           11_200.66       494.43    11_695.09       0.7380          1.0140         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           11_200.66       563.01    11_763.67       0.8736          1.0053         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           11_200.66       509.29    11_709.94       0.7371          1.0141         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           11_200.66       571.94    11_772.60       0.8733          1.0053         7.32
IVF-Binary-1024-nl223-pca (self)                      11_200.66     1_584.93    12_785.59       0.7385          1.0140         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            11_396.06       423.48    11_819.54       0.2775             NaN         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            11_396.06       424.90    11_820.96       0.2770             NaN         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            11_396.06       449.75    11_845.80       0.2759             NaN         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           11_396.06       499.31    11_895.36       0.7391          1.0139         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           11_396.06       543.71    11_939.76       0.8717          1.0054         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           11_396.06       490.57    11_886.63       0.7394          1.0139         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           11_396.06       551.42    11_947.48       0.8740          1.0053         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           11_396.06       539.54    11_935.59       0.7375          1.0140         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           11_396.06       595.15    11_991.21       0.8737          1.0053         7.42
IVF-Binary-1024-nl316-pca (self)                      11_396.06     1_588.29    12_984.35       0.7395          1.0139         7.42
IVF-Binary-256-nl158-np7-rf0-signed (query)            4_061.45       116.30     4_177.74       0.1063             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-signed (query)           4_061.45       118.80     4_180.25       0.1034             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-signed (query)           4_061.45       123.86     4_185.30       0.1017             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-signed (query)           4_061.45       172.51     4_233.96       0.3128          1.0834         1.93
IVF-Binary-256-nl158-np7-rf20-signed (query)           4_061.45       223.27     4_284.72       0.4564          1.0499         1.93
IVF-Binary-256-nl158-np12-rf10-signed (query)          4_061.45       181.11     4_242.56       0.2990          1.0878         1.93
IVF-Binary-256-nl158-np12-rf20-signed (query)          4_061.45       236.71     4_298.16       0.4375          1.0533         1.93
IVF-Binary-256-nl158-np17-rf10-signed (query)          4_061.45       191.80     4_253.25       0.2914          1.0901         1.93
IVF-Binary-256-nl158-np17-rf20-signed (query)          4_061.45       251.10     4_312.55       0.4284          1.0549         1.93
IVF-Binary-256-nl158-signed (self)                     4_061.45       564.89     4_626.34       0.2986          1.0878         1.93
IVF-Binary-256-nl223-np11-rf0-signed (query)           3_096.49       117.63     3_214.12       0.1054             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-signed (query)           3_096.49       119.81     3_216.30       0.1027             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-signed (query)           3_096.49       125.82     3_222.31       0.1013             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-signed (query)          3_096.49       177.58     3_274.07       0.3066          1.0843         2.00
IVF-Binary-256-nl223-np11-rf20-signed (query)          3_096.49       228.89     3_325.38       0.4485          1.0506         2.00
IVF-Binary-256-nl223-np14-rf10-signed (query)          3_096.49       180.30     3_276.80       0.2976          1.0874         2.00
IVF-Binary-256-nl223-np14-rf20-signed (query)          3_096.49       235.29     3_331.78       0.4366          1.0529         2.00
IVF-Binary-256-nl223-np21-rf10-signed (query)          3_096.49       191.25     3_287.75       0.2925          1.0893         2.00
IVF-Binary-256-nl223-np21-rf20-signed (query)          3_096.49       248.89     3_345.38       0.4296          1.0544         2.00
IVF-Binary-256-nl223-signed (self)                     3_096.49       554.18     3_650.67       0.2974          1.0875         2.00
IVF-Binary-256-nl316-np15-rf0-signed (query)           3_259.26       125.16     3_384.43       0.1048             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-signed (query)           3_259.26       125.31     3_384.58       0.1036             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-signed (query)           3_259.26       131.59     3_390.85       0.1019             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-signed (query)          3_259.26       184.32     3_443.59       0.3062          1.0846         2.09
IVF-Binary-256-nl316-np15-rf20-signed (query)          3_259.26       235.88     3_495.15       0.4494          1.0505         2.09
IVF-Binary-256-nl316-np17-rf10-signed (query)          3_259.26       184.57     3_443.84       0.3014          1.0861         2.09
IVF-Binary-256-nl316-np17-rf20-signed (query)          3_259.26       239.49     3_498.76       0.4427          1.0519         2.09
IVF-Binary-256-nl316-np25-rf10-signed (query)          3_259.26       190.82     3_450.09       0.2928          1.0893         2.09
IVF-Binary-256-nl316-np25-rf20-signed (query)          3_259.26       247.84     3_507.10       0.4300          1.0544         2.09
IVF-Binary-256-nl316-signed (self)                     3_259.26       568.27     3_827.53       0.3014          1.0861         2.09
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
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        20.05     9_583.62     9_603.67       1.0000          1.0000        97.66
Exhaustive (self)                                         20.05    32_213.74    32_233.79       1.0000          1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_648.38       353.65     6_002.02       0.0893             NaN         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_648.38       478.10     6_126.48       0.2397          1.0749         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_648.38       603.67     6_252.05       0.3605          1.0474         2.03
ExhaustiveBinary-256-random (self)                     5_648.38     1_574.19     7_222.57       0.2400          1.0748         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_271.59       364.89     6_636.47       0.1907             NaN         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_271.59       492.88     6_764.47       0.5143          1.0274         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_271.59       625.10     6_896.68       0.6417          1.0162         2.03
ExhaustiveBinary-256-pca (self)                        6_271.59     1_624.58     7_896.17       0.5163          1.0273         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_043.99       658.72    11_702.71       0.1039             NaN         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_043.99       777.77    11_821.76       0.2971          1.0611         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_043.99       908.22    11_952.21       0.4344          1.0373         4.05
ExhaustiveBinary-512-random (self)                    11_043.99     2_578.09    13_622.07       0.2968          1.0610         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_809.12       670.55    12_479.66       0.2182             NaN         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_809.12       797.75    12_606.87       0.5365          1.0253         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_809.12       943.99    12_753.10       0.6508          1.0154         4.05
ExhaustiveBinary-512-pca (self)                       11_809.12     2_652.52    14_461.64       0.5379          1.0252         4.05
ExhaustiveBinary-1024-random_no_rr (query)            21_913.38     1_185.58    23_098.96       0.1273             NaN         8.10
ExhaustiveBinary-1024-random-rf10 (query)             21_913.38     1_314.50    23_227.88       0.3824          1.0455         8.10
ExhaustiveBinary-1024-random-rf20 (query)             21_913.38     1_466.36    23_379.74       0.5379          1.0261         8.10
ExhaustiveBinary-1024-random (self)                   21_913.38     4_361.09    26_274.47       0.3828          1.0454         8.10
ExhaustiveBinary-1024-pca_no_rr (query)               23_116.36     1_193.99    24_310.36       0.2506             NaN         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                23_116.36     1_344.56    24_460.93       0.6932          1.0125         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                23_116.36     1_490.90    24_607.27       0.8406          1.0051         8.11
ExhaustiveBinary-1024-pca (self)                      23_116.36     4_462.27    27_578.64       0.6937          1.0125         8.11
ExhaustiveBinary-512-signed_no_rr (query)             11_232.03       664.30    11_896.33       0.1039             NaN         4.05
ExhaustiveBinary-512-signed-rf10 (query)              11_232.03       809.27    12_041.30       0.2971          1.0611         4.05
ExhaustiveBinary-512-signed-rf20 (query)              11_232.03       925.53    12_157.56       0.4344          1.0373         4.05
ExhaustiveBinary-512-signed (self)                    11_232.03     2_599.59    13_831.62       0.2968          1.0610         4.05
IVF-Binary-256-nl158-np7-rf0-random (query)            8_774.09       233.14     9_007.23       0.0930             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           8_774.09       238.60     9_012.69       0.0910             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           8_774.09       242.53     9_016.62       0.0899             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           8_774.09       317.33     9_091.43       0.2562          1.0706         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           8_774.09       399.96     9_174.05       0.3841          1.0441         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          8_774.09       326.98     9_101.07       0.2455          1.0732         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          8_774.09       412.94     9_187.03       0.3695          1.0461         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          8_774.09       336.74     9_110.83       0.2406          1.0746         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          8_774.09       426.67     9_200.77       0.3628          1.0472         2.34
IVF-Binary-256-nl158-random (self)                     8_774.09     1_019.39     9_793.49       0.2463          1.0731         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           6_588.95       241.75     6_830.70       0.0926             NaN         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)           6_588.95       243.81     6_832.75       0.0914             NaN         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)           6_588.95       248.54     6_837.48       0.0898             NaN         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)          6_588.95       330.82     6_919.77       0.2507          1.0715         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)          6_588.95       417.16     7_006.11       0.3754          1.0449         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)          6_588.95       345.11     6_934.06       0.2453          1.0732         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)          6_588.95       415.68     7_004.62       0.3675          1.0462         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)          6_588.95       336.82     6_925.76       0.2406          1.0748         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)          6_588.95       437.99     7_026.94       0.3612          1.0474         2.46
IVF-Binary-256-nl223-random (self)                     6_588.95     1_031.39     7_620.34       0.2459          1.0730         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)           6_941.43       255.49     7_196.92       0.0916             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           6_941.43       265.72     7_207.15       0.0908             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           6_941.43       260.78     7_202.21       0.0900             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          6_941.43       344.48     7_285.90       0.2484          1.0722         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          6_941.43       425.11     7_366.54       0.3725          1.0453         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          6_941.43       343.08     7_284.51       0.2446          1.0733         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          6_941.43       441.51     7_382.94       0.3671          1.0463         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          6_941.43       369.38     7_310.81       0.2412          1.0744         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          6_941.43       437.35     7_378.78       0.3624          1.0471         2.65
IVF-Binary-256-nl316-random (self)                     6_941.43     1_070.00     8_011.43       0.2449          1.0733         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               9_427.91       235.64     9_663.56       0.1962             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              9_427.91       242.03     9_669.94       0.1955             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              9_427.91       264.22     9_692.13       0.1948             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              9_427.91       329.00     9_756.91       0.5664          1.0224         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              9_427.91       411.31     9_839.23       0.7221          1.0113         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             9_427.91       337.14     9_765.05       0.5686          1.0221         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             9_427.91       426.83     9_854.74       0.7288          1.0108         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             9_427.91       346.71     9_774.63       0.5653          1.0224         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             9_427.91       443.22     9_871.13       0.7241          1.0110         2.34
IVF-Binary-256-nl158-pca (self)                        9_427.91     1_061.14    10_489.05       0.5708          1.0220         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              7_263.52       244.87     7_508.38       0.1962             NaN         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              7_263.52       247.23     7_510.74       0.1954             NaN         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              7_263.52       255.64     7_519.16       0.1948             NaN         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             7_263.52       340.83     7_604.35       0.5703          1.0220         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             7_263.52       425.07     7_688.59       0.7308          1.0106         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             7_263.52       340.52     7_604.03       0.5680          1.0222         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             7_263.52       430.31     7_693.83       0.7292          1.0107         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             7_263.52       349.99     7_613.51       0.5655          1.0224         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             7_263.52       446.66     7_710.18       0.7249          1.0110         2.47
IVF-Binary-256-nl223-pca (self)                        7_263.52     1_072.25     8_335.76       0.5703          1.0221         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              7_583.06       259.10     7_842.16       0.1960             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              7_583.06       260.74     7_843.80       0.1954             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              7_583.06       269.34     7_852.40       0.1949             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             7_583.06       354.57     7_937.63       0.5718          1.0218         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             7_583.06       437.91     8_020.98       0.7343          1.0105         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             7_583.06       354.02     7_937.08       0.5698          1.0220         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             7_583.06       441.85     8_024.91       0.7321          1.0106         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             7_583.06       360.79     7_943.85       0.5670          1.0222         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             7_583.06       454.28     8_037.34       0.7274          1.0108         2.65
IVF-Binary-256-nl316-pca (self)                        7_583.06     1_113.58     8_696.64       0.5719          1.0219         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           14_201.82       428.72    14_630.54       0.1075             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          14_201.82       437.89    14_639.71       0.1057             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          14_201.82       443.95    14_645.78       0.1046             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          14_201.82       515.52    14_717.34       0.3090          1.0587         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          14_201.82       589.61    14_791.43       0.4503          1.0355         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         14_201.82       523.15    14_724.98       0.3015          1.0601         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         14_201.82       606.71    14_808.53       0.4408          1.0365         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         14_201.82       535.74    14_737.56       0.2976          1.0610         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         14_201.82       652.64    14_854.46       0.4354          1.0372         4.36
IVF-Binary-512-nl158-random (self)                    14_201.82     1_683.07    15_884.90       0.3015          1.0601         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          12_066.28       434.96    12_501.24       0.1064             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          12_066.28       438.31    12_504.58       0.1054             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          12_066.28       447.49    12_513.77       0.1044             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         12_066.28       521.75    12_588.03       0.3056          1.0589         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         12_066.28       600.13    12_666.40       0.4458          1.0357         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         12_066.28       523.89    12_590.16       0.3007          1.0601         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         12_066.28       606.89    12_673.16       0.4393          1.0366         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         12_066.28       545.14    12_611.42       0.2972          1.0611         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         12_066.28       625.18    12_691.46       0.4349          1.0373         4.49
IVF-Binary-512-nl223-random (self)                    12_066.28     1_682.62    13_748.90       0.3008          1.0601         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          12_409.92       448.85    12_858.77       0.1060             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          12_409.92       451.25    12_861.17       0.1053             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          12_409.92       464.09    12_874.01       0.1047             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         12_409.92       535.22    12_945.14       0.3039          1.0594         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         12_409.92       618.57    13_028.49       0.4442          1.0360         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         12_409.92       542.97    12_952.89       0.3008          1.0602         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         12_409.92       618.61    13_028.53       0.4397          1.0366         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         12_409.92       545.62    12_955.54       0.2978          1.0609         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         12_409.92       634.39    13_044.31       0.4360          1.0371         4.67
IVF-Binary-512-nl316-random (self)                    12_409.92     1_720.39    14_130.31       0.3008          1.0601         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              15_038.26       439.31    15_477.57       0.2351             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             15_038.26       448.93    15_487.19       0.2346             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             15_038.26       456.58    15_494.84       0.2334             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             15_038.26       532.21    15_570.48       0.6487          1.0157         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             15_038.26       624.26    15_662.53       0.7884          1.0077         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            15_038.26       542.52    15_580.78       0.6521          1.0153         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            15_038.26       631.55    15_669.82       0.7985          1.0069         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            15_038.26       556.19    15_594.45       0.6451          1.0157         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            15_038.26       650.90    15_689.16       0.7900          1.0073         4.36
IVF-Binary-512-nl158-pca (self)                       15_038.26     1_763.66    16_801.93       0.6524          1.0153         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             12_857.65       449.90    13_307.54       0.2347             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             12_857.65       451.91    13_309.56       0.2340             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             12_857.65       459.72    13_317.37       0.2332             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            12_857.65       542.29    13_399.94       0.6559          1.0150         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            12_857.65       625.92    13_483.57       0.8048          1.0067         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            12_857.65       544.42    13_402.06       0.6536          1.0151         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            12_857.65       635.69    13_493.34       0.8031          1.0067         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            12_857.65       560.49    13_418.14       0.6471          1.0156         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            12_857.65       654.42    13_512.07       0.7931          1.0072         4.49
IVF-Binary-512-nl223-pca (self)                       12_857.65     1_759.67    14_617.32       0.6538          1.0152         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             13_227.30       462.91    13_690.22       0.2352             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             13_227.30       469.98    13_697.28       0.2347             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             13_227.30       470.86    13_698.16       0.2339             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            13_227.30       557.21    13_784.51       0.6589          1.0148         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            13_227.30       641.07    13_868.37       0.8096          1.0064         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            13_227.30       564.37    13_791.67       0.6568          1.0149         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            13_227.30       646.84    13_874.14       0.8076          1.0065         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            13_227.30       571.35    13_798.65       0.6502          1.0154         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            13_227.30       660.87    13_888.17       0.7981          1.0069         4.67
IVF-Binary-512-nl316-pca (self)                       13_227.30     1_855.21    15_082.51       0.6571          1.0149         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          25_130.37       834.58    25_964.95       0.1306             NaN         8.41
IVF-Binary-1024-nl158-np12-rf0-random (query)         25_130.37       849.82    25_980.19       0.1289             NaN         8.41
IVF-Binary-1024-nl158-np17-rf0-random (query)         25_130.37       861.55    25_991.92       0.1279             NaN         8.41
IVF-Binary-1024-nl158-np7-rf10-random (query)         25_130.37       923.07    26_053.45       0.3897          1.0444         8.41
IVF-Binary-1024-nl158-np7-rf20-random (query)         25_130.37     1_004.66    26_135.03       0.5436          1.0256         8.41
IVF-Binary-1024-nl158-np12-rf10-random (query)        25_130.37       950.50    26_080.87       0.3858          1.0449         8.41
IVF-Binary-1024-nl158-np12-rf20-random (query)        25_130.37     1_031.18    26_161.55       0.5426          1.0257         8.41
IVF-Binary-1024-nl158-np17-rf10-random (query)        25_130.37       964.40    26_094.78       0.3828          1.0454         8.41
IVF-Binary-1024-nl158-np17-rf20-random (query)        25_130.37     1_059.48    26_189.86       0.5386          1.0260         8.41
IVF-Binary-1024-nl158-random (self)                   25_130.37     3_088.88    28_219.26       0.3865          1.0449         8.41
IVF-Binary-1024-nl223-np11-rf0-random (query)         22_930.80       853.39    23_784.19       0.1299             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         22_930.80       858.73    23_789.53       0.1289             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         22_930.80       880.00    23_810.81       0.1280             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        22_930.80       961.56    23_892.36       0.3892          1.0442         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        22_930.80     1_029.36    23_960.16       0.5456          1.0253         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        22_930.80       953.82    23_884.62       0.3854          1.0449         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        22_930.80     1_037.99    23_968.80       0.5398          1.0259         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        22_930.80       972.54    23_903.34       0.3829          1.0454         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        22_930.80     1_064.28    23_995.09       0.5372          1.0262         8.54
IVF-Binary-1024-nl223-random (self)                   22_930.80     3_114.13    26_044.93       0.3854          1.0450         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         23_268.35       857.82    24_126.16       0.1291             NaN         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)         23_268.35       860.63    24_128.97       0.1284             NaN         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)         23_268.35       873.26    24_141.60       0.1278             NaN         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)        23_268.35       952.63    24_220.98       0.3879          1.0445         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)        23_268.35     1_037.62    24_305.96       0.5448          1.0254         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)        23_268.35       951.95    24_220.29       0.3851          1.0450         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)        23_268.35     1_042.22    24_310.57       0.5409          1.0258         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)        23_268.35       971.43    24_239.77       0.3830          1.0454         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)        23_268.35     1_059.87    24_328.22       0.5381          1.0261         8.72
IVF-Binary-1024-nl316-random (self)                   23_268.35     3_099.31    26_367.66       0.3856          1.0449         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)             26_362.44       843.44    27_205.88       0.2510             NaN         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            26_362.44       856.68    27_219.13       0.2512             NaN         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            26_362.44       878.68    27_241.12       0.2507             NaN         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            26_362.44       936.19    27_298.63       0.6813          1.0134         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            26_362.44     1_018.71    27_381.15       0.8163          1.0064         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           26_362.44       954.40    27_316.84       0.6931          1.0125         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           26_362.44     1_052.82    27_415.26       0.8385          1.0052         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           26_362.44       985.67    27_348.11       0.6931          1.0125         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           26_362.44     1_067.72    27_430.16       0.8405          1.0051         8.42
IVF-Binary-1024-nl158-pca (self)                      26_362.44     3_121.97    29_484.42       0.6933          1.0125         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            24_133.51       865.22    24_998.74       0.2512             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            24_133.51       856.46    24_989.98       0.2509             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            24_133.51       883.35    25_016.86       0.2507             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           24_133.51       948.34    25_081.85       0.6923          1.0125         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           24_133.51     1_033.34    25_166.85       0.8371          1.0053         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           24_133.51       972.76    25_106.27       0.6936          1.0125         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           24_133.51     1_041.25    25_174.77       0.8405          1.0051         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           24_133.51       974.30    25_107.82       0.6932          1.0125         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           24_133.51     1_074.81    25_208.32       0.8405          1.0051         8.54
IVF-Binary-1024-nl223-pca (self)                      24_133.51     3_164.67    27_298.18       0.6941          1.0125         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            24_484.53       865.18    25_349.71       0.2512             NaN         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            24_484.53       867.74    25_352.26       0.2509             NaN         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            24_484.53       886.86    25_371.38       0.2506             NaN         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           24_484.53       960.53    25_445.06       0.6943          1.0124         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           24_484.53     1_044.21    25_528.74       0.8399          1.0051         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           24_484.53       961.25    25_445.78       0.6939          1.0124         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           24_484.53     1_050.60    25_535.13       0.8408          1.0051         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           24_484.53       979.32    25_463.85       0.6932          1.0125         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           24_484.53     1_069.60    25_554.13       0.8405          1.0051         8.73
IVF-Binary-1024-nl316-pca (self)                      24_484.53     3_142.57    27_627.10       0.6943          1.0125         8.73
IVF-Binary-512-nl158-np7-rf0-signed (query)           14_198.73       425.02    14_623.75       0.1075             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-signed (query)          14_198.73       433.99    14_632.72       0.1057             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-signed (query)          14_198.73       443.51    14_642.24       0.1046             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-signed (query)          14_198.73       512.37    14_711.10       0.3090          1.0587         4.36
IVF-Binary-512-nl158-np7-rf20-signed (query)          14_198.73       589.02    14_787.75       0.4503          1.0355         4.36
IVF-Binary-512-nl158-np12-rf10-signed (query)         14_198.73       521.45    14_720.18       0.3015          1.0601         4.36
IVF-Binary-512-nl158-np12-rf20-signed (query)         14_198.73       606.16    14_804.89       0.4408          1.0365         4.36
IVF-Binary-512-nl158-np17-rf10-signed (query)         14_198.73       535.53    14_734.26       0.2976          1.0610         4.36
IVF-Binary-512-nl158-np17-rf20-signed (query)         14_198.73       624.73    14_823.46       0.4354          1.0372         4.36
IVF-Binary-512-nl158-signed (self)                    14_198.73     1_709.42    15_908.15       0.3015          1.0601         4.36
IVF-Binary-512-nl223-np11-rf0-signed (query)          12_072.72       434.44    12_507.16       0.1064             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-signed (query)          12_072.72       438.28    12_511.00       0.1054             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-signed (query)          12_072.72       446.84    12_519.56       0.1044             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-signed (query)         12_072.72       522.02    12_594.73       0.3056          1.0589         4.49
IVF-Binary-512-nl223-np11-rf20-signed (query)         12_072.72       599.90    12_672.61       0.4458          1.0357         4.49
IVF-Binary-512-nl223-np14-rf10-signed (query)         12_072.72       523.36    12_596.07       0.3007          1.0601         4.49
IVF-Binary-512-nl223-np14-rf20-signed (query)         12_072.72       606.88    12_679.60       0.4393          1.0366         4.49
IVF-Binary-512-nl223-np21-rf10-signed (query)         12_072.72       536.25    12_608.96       0.2972          1.0611         4.49
IVF-Binary-512-nl223-np21-rf20-signed (query)         12_072.72       623.82    12_696.53       0.4349          1.0373         4.49
IVF-Binary-512-nl223-signed (self)                    12_072.72     1_681.67    13_754.39       0.3008          1.0601         4.49
IVF-Binary-512-nl316-np15-rf0-signed (query)          12_408.22       450.61    12_858.83       0.1060             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-signed (query)          12_408.22       450.51    12_858.72       0.1053             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-signed (query)          12_408.22       457.62    12_865.84       0.1047             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-signed (query)         12_408.22       535.70    12_943.92       0.3039          1.0594         4.67
IVF-Binary-512-nl316-np15-rf20-signed (query)         12_408.22       612.42    13_020.63       0.4442          1.0360         4.67
IVF-Binary-512-nl316-np17-rf10-signed (query)         12_408.22       534.34    12_942.56       0.3008          1.0602         4.67
IVF-Binary-512-nl316-np17-rf20-signed (query)         12_408.22       617.80    13_026.02       0.4397          1.0366         4.67
IVF-Binary-512-nl316-np25-rf10-signed (query)         12_408.22       545.10    12_953.32       0.2978          1.0609         4.67
IVF-Binary-512-nl316-np25-rf20-signed (query)         12_408.22       634.27    13_042.49       0.4360          1.0371         4.67
IVF-Binary-512-nl316-signed (self)                    12_408.22     1_717.33    14_125.54       0.3008          1.0601         4.67
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
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        39.83    21_930.41    21_970.25       1.0000          1.0000       195.31
Exhaustive (self)                                         39.83    74_552.43    74_592.26       1.0000          1.0000       195.31
ExhaustiveBinary-256-random_no_rr (query)             11_975.18       581.00    12_556.18       0.0840             NaN         2.53
ExhaustiveBinary-256-random-rf10 (query)              11_975.18       735.22    12_710.40       0.2060          1.0589         2.53
ExhaustiveBinary-256-random-rf20 (query)              11_975.18       900.57    12_875.76       0.3135          1.0384         2.53
ExhaustiveBinary-256-random (self)                    11_975.18     2_420.02    14_395.20       0.2052          1.0589         2.53
ExhaustiveBinary-256-pca_no_rr (query)                13_300.72       575.98    13_876.70       0.1619             NaN         2.53
ExhaustiveBinary-256-pca-rf10 (query)                 13_300.72       752.33    14_053.05       0.4353          1.0256         2.53
ExhaustiveBinary-256-pca-rf20 (query)                 13_300.72       919.73    14_220.44       0.5548          1.0162         2.53
ExhaustiveBinary-256-pca (self)                       13_300.72     2_460.85    15_761.57       0.4349          1.0256         2.53
ExhaustiveBinary-512-random_no_rr (query)             23_408.06     1_084.54    24_492.60       0.0922             NaN         5.05
ExhaustiveBinary-512-random-rf10 (query)              23_408.06     1_247.85    24_655.91       0.2430          1.0508         5.05
ExhaustiveBinary-512-random-rf20 (query)              23_408.06     1_419.43    24_827.48       0.3648          1.0323         5.05
ExhaustiveBinary-512-random (self)                    23_408.06     4_127.70    27_535.76       0.2441          1.0506         5.05
ExhaustiveBinary-512-pca_no_rr (query)                24_981.11     1_101.04    26_082.15       0.1735             NaN         5.06
ExhaustiveBinary-512-pca-rf10 (query)                 24_981.11     1_272.85    26_253.96       0.4277          1.0267         5.06
ExhaustiveBinary-512-pca-rf20 (query)                 24_981.11     1_441.21    26_422.32       0.5354          1.0175         5.06
ExhaustiveBinary-512-pca (self)                       24_981.11     4_189.02    29_170.12       0.4286          1.0266         5.06
ExhaustiveBinary-1024-random_no_rr (query)            46_877.12     2_031.31    48_908.43       0.1077             NaN        10.10
ExhaustiveBinary-1024-random-rf10 (query)             46_877.12     2_210.22    49_087.35       0.3093          1.0402        10.10
ExhaustiveBinary-1024-random-rf20 (query)             46_877.12     2_425.10    49_302.22       0.4493          1.0244        10.10
ExhaustiveBinary-1024-random (self)                   46_877.12     7_369.64    54_246.76       0.3094          1.0401        10.10
ExhaustiveBinary-1024-pca_no_rr (query)               48_371.90     2_057.69    50_429.59       0.2002             NaN        10.11
ExhaustiveBinary-1024-pca-rf10 (query)                48_371.90     2_240.76    50_612.67       0.4677          1.0246        10.11
ExhaustiveBinary-1024-pca-rf20 (query)                48_371.90     2_429.63    50_801.53       0.5715          1.0152        10.11
ExhaustiveBinary-1024-pca (self)                      48_371.90     7_426.61    55_798.51       0.4688          1.0240        10.11
ExhaustiveBinary-1024-signed_no_rr (query)            46_878.95     2_032.42    48_911.37       0.1077             NaN        10.10
ExhaustiveBinary-1024-signed-rf10 (query)             46_878.95     2_214.58    49_093.53       0.3093          1.0402        10.10
ExhaustiveBinary-1024-signed-rf20 (query)             46_878.95     2_407.09    49_286.04       0.4493          1.0244        10.10
ExhaustiveBinary-1024-signed (self)                   46_878.95     7_336.97    54_215.91       0.3094          1.0401        10.10
IVF-Binary-256-nl158-np7-rf0-random (query)           18_938.63       473.71    19_412.34       0.0856             NaN         3.14
IVF-Binary-256-nl158-np12-rf0-random (query)          18_938.63       486.17    19_424.80       0.0848             NaN         3.14
IVF-Binary-256-nl158-np17-rf0-random (query)          18_938.63       489.86    19_428.49       0.0843             NaN         3.14
IVF-Binary-256-nl158-np7-rf10-random (query)          18_938.63       599.97    19_538.60       0.2155          1.0567         3.14
IVF-Binary-256-nl158-np7-rf20-random (query)          18_938.63       721.15    19_659.78       0.3261          1.0368         3.14
IVF-Binary-256-nl158-np12-rf10-random (query)         18_938.63       609.26    19_547.88       0.2106          1.0579         3.14
IVF-Binary-256-nl158-np12-rf20-random (query)         18_938.63       734.22    19_672.85       0.3185          1.0377         3.14
IVF-Binary-256-nl158-np17-rf10-random (query)         18_938.63       616.86    19_555.49       0.2076          1.0586         3.14
IVF-Binary-256-nl158-np17-rf20-random (query)         18_938.63       746.18    19_684.81       0.3141          1.0383         3.14
IVF-Binary-256-nl158-random (self)                    18_938.63     1_948.36    20_886.99       0.2095          1.0579         3.14
IVF-Binary-256-nl223-np11-rf0-random (query)          13_733.50       504.66    14_238.15       0.0860             NaN         3.40
IVF-Binary-256-nl223-np14-rf0-random (query)          13_733.50       495.55    14_229.05       0.0851             NaN         3.40
IVF-Binary-256-nl223-np21-rf0-random (query)          13_733.50       502.95    14_236.45       0.0843             NaN         3.40
IVF-Binary-256-nl223-np11-rf10-random (query)         13_733.50       623.63    14_357.13       0.2135          1.0570         3.40
IVF-Binary-256-nl223-np11-rf20-random (query)         13_733.50       741.84    14_475.34       0.3241          1.0370         3.40
IVF-Binary-256-nl223-np14-rf10-random (query)         13_733.50       624.71    14_358.20       0.2100          1.0579         3.40
IVF-Binary-256-nl223-np14-rf20-random (query)         13_733.50       754.68    14_488.17       0.3192          1.0377         3.40
IVF-Binary-256-nl223-np21-rf10-random (query)         13_733.50       638.72    14_372.22       0.2075          1.0586         3.40
IVF-Binary-256-nl223-np21-rf20-random (query)         13_733.50       760.65    14_494.15       0.3160          1.0381         3.40
IVF-Binary-256-nl223-random (self)                    13_733.50     2_010.52    15_744.02       0.2092          1.0578         3.40
IVF-Binary-256-nl316-np15-rf0-random (query)          14_428.67       527.53    14_956.20       0.0853             NaN         3.76
IVF-Binary-256-nl316-np17-rf0-random (query)          14_428.67       527.44    14_956.10       0.0849             NaN         3.76
IVF-Binary-256-nl316-np25-rf0-random (query)          14_428.67       531.87    14_960.54       0.0845             NaN         3.76
IVF-Binary-256-nl316-np15-rf10-random (query)         14_428.67       653.90    15_082.56       0.2120          1.0574         3.76
IVF-Binary-256-nl316-np15-rf20-random (query)         14_428.67       775.54    15_204.21       0.3218          1.0373         3.76
IVF-Binary-256-nl316-np17-rf10-random (query)         14_428.67       652.10    15_080.77       0.2103          1.0579         3.76
IVF-Binary-256-nl316-np17-rf20-random (query)         14_428.67       787.43    15_216.10       0.3193          1.0376         3.76
IVF-Binary-256-nl316-np25-rf10-random (query)         14_428.67       658.15    15_086.82       0.2083          1.0584         3.76
IVF-Binary-256-nl316-np25-rf20-random (query)         14_428.67       796.68    15_225.34       0.3164          1.0380         3.76
IVF-Binary-256-nl316-random (self)                    14_428.67     2_098.98    16_527.65       0.2093          1.0578         3.76
IVF-Binary-256-nl158-np7-rf0-pca (query)              20_425.01       480.58    20_905.59       0.1677             NaN         3.15
IVF-Binary-256-nl158-np12-rf0-pca (query)             20_425.01       490.22    20_915.23       0.1675             NaN         3.15
IVF-Binary-256-nl158-np17-rf0-pca (query)             20_425.01       492.35    20_917.36       0.1670             NaN         3.15
IVF-Binary-256-nl158-np7-rf10-pca (query)             20_425.01       614.36    21_039.38       0.4931          1.0207         3.15
IVF-Binary-256-nl158-np7-rf20-pca (query)             20_425.01       745.35    21_170.36       0.6468          1.0113         3.15
IVF-Binary-256-nl158-np12-rf10-pca (query)            20_425.01       621.70    21_046.71       0.4949          1.0205         3.15
IVF-Binary-256-nl158-np12-rf20-pca (query)            20_425.01       755.77    21_180.79       0.6508          1.0111         3.15
IVF-Binary-256-nl158-np17-rf10-pca (query)            20_425.01       631.93    21_056.94       0.4906          1.0208         3.15
IVF-Binary-256-nl158-np17-rf20-pca (query)            20_425.01       766.17    21_191.18       0.6442          1.0113         3.15
IVF-Binary-256-nl158-pca (self)                       20_425.01     2_015.28    22_440.30       0.4943          1.0205         3.15
IVF-Binary-256-nl223-np11-rf0-pca (query)             15_208.01       504.64    15_712.66       0.1679             NaN         3.40
IVF-Binary-256-nl223-np14-rf0-pca (query)             15_208.01       504.95    15_712.96       0.1674             NaN         3.40
IVF-Binary-256-nl223-np21-rf0-pca (query)             15_208.01       509.03    15_717.04       0.1669             NaN         3.40
IVF-Binary-256-nl223-np11-rf10-pca (query)            15_208.01       636.71    15_844.72       0.4988          1.0202         3.40
IVF-Binary-256-nl223-np11-rf20-pca (query)            15_208.01       765.78    15_973.80       0.6593          1.0107         3.40
IVF-Binary-256-nl223-np14-rf10-pca (query)            15_208.01       636.63    15_844.64       0.4968          1.0204         3.40
IVF-Binary-256-nl223-np14-rf20-pca (query)            15_208.01       767.26    15_975.27       0.6564          1.0108         3.40
IVF-Binary-256-nl223-np21-rf10-pca (query)            15_208.01       645.98    15_853.99       0.4926          1.0207         3.40
IVF-Binary-256-nl223-np21-rf20-pca (query)            15_208.01       787.47    15_995.48       0.6488          1.0111         3.40
IVF-Binary-256-nl223-pca (self)                       15_208.01     2_069.24    17_277.25       0.4967          1.0203         3.40
IVF-Binary-256-nl316-np15-rf0-pca (query)             15_910.79       531.61    16_442.41       0.1681             NaN         3.77
IVF-Binary-256-nl316-np17-rf0-pca (query)             15_910.79       533.32    16_444.11       0.1678             NaN         3.77
IVF-Binary-256-nl316-np25-rf0-pca (query)             15_910.79       537.58    16_448.37       0.1673             NaN         3.77
IVF-Binary-256-nl316-np15-rf10-pca (query)            15_910.79       679.29    16_590.08       0.5001          1.0201         3.77
IVF-Binary-256-nl316-np15-rf20-pca (query)            15_910.79       795.36    16_706.15       0.6619          1.0106         3.77
IVF-Binary-256-nl316-np17-rf10-pca (query)            15_910.79       668.89    16_579.68       0.4989          1.0202         3.77
IVF-Binary-256-nl316-np17-rf20-pca (query)            15_910.79       796.87    16_707.67       0.6601          1.0106         3.77
IVF-Binary-256-nl316-np25-rf10-pca (query)            15_910.79       676.40    16_587.20       0.4942          1.0205         3.77
IVF-Binary-256-nl316-np25-rf20-pca (query)            15_910.79       809.38    16_720.18       0.6525          1.0110         3.77
IVF-Binary-256-nl316-pca (self)                       15_910.79     2_160.15    18_070.94       0.4987          1.0202         3.77
IVF-Binary-512-nl158-np7-rf0-random (query)           30_715.02       881.98    31_597.01       0.0937             NaN         5.67
IVF-Binary-512-nl158-np12-rf0-random (query)          30_715.02       892.82    31_607.85       0.0930             NaN         5.67
IVF-Binary-512-nl158-np17-rf0-random (query)          30_715.02       904.62    31_619.64       0.0924             NaN         5.67
IVF-Binary-512-nl158-np7-rf10-random (query)          30_715.02     1_011.24    31_726.27       0.2508          1.0494         5.67
IVF-Binary-512-nl158-np7-rf20-random (query)          30_715.02     1_124.83    31_839.86       0.3737          1.0313         5.67
IVF-Binary-512-nl158-np12-rf10-random (query)         30_715.02     1_019.06    31_734.08       0.2470          1.0501         5.67
IVF-Binary-512-nl158-np12-rf20-random (query)         30_715.02     1_144.05    31_859.07       0.3689          1.0318         5.67
IVF-Binary-512-nl158-np17-rf10-random (query)         30_715.02     1_036.11    31_751.14       0.2448          1.0505         5.67
IVF-Binary-512-nl158-np17-rf20-random (query)         30_715.02     1_164.10    31_879.13       0.3654          1.0322         5.67
IVF-Binary-512-nl158-random (self)                    30_715.02     3_331.42    34_046.44       0.2475          1.0499         5.67
IVF-Binary-512-nl223-np11-rf0-random (query)          25_416.81       926.74    26_343.55       0.0937             NaN         5.92
IVF-Binary-512-nl223-np14-rf0-random (query)          25_416.81       920.81    26_337.62       0.0931             NaN         5.92
IVF-Binary-512-nl223-np21-rf0-random (query)          25_416.81       928.20    26_345.01       0.0927             NaN         5.92
IVF-Binary-512-nl223-np11-rf10-random (query)         25_416.81     1_052.65    26_469.46       0.2494          1.0496         5.92
IVF-Binary-512-nl223-np11-rf20-random (query)         25_416.81     1_165.68    26_582.49       0.3728          1.0313         5.92
IVF-Binary-512-nl223-np14-rf10-random (query)         25_416.81     1_049.72    26_466.53       0.2462          1.0502         5.92
IVF-Binary-512-nl223-np14-rf20-random (query)         25_416.81     1_176.57    26_593.38       0.3684          1.0318         5.92
IVF-Binary-512-nl223-np21-rf10-random (query)         25_416.81     1_059.52    26_476.33       0.2446          1.0505         5.92
IVF-Binary-512-nl223-np21-rf20-random (query)         25_416.81     1_191.41    26_608.22       0.3662          1.0321         5.92
IVF-Binary-512-nl223-random (self)                    25_416.81     3_409.14    28_825.95       0.2469          1.0500         5.92
IVF-Binary-512-nl316-np15-rf0-random (query)          26_080.09       937.89    27_017.98       0.0933             NaN         6.29
IVF-Binary-512-nl316-np17-rf0-random (query)          26_080.09       948.68    27_028.78       0.0930             NaN         6.29
IVF-Binary-512-nl316-np25-rf0-random (query)          26_080.09       947.80    27_027.89       0.0927             NaN         6.29
IVF-Binary-512-nl316-np15-rf10-random (query)         26_080.09     1_062.18    27_142.27       0.2484          1.0497         6.29
IVF-Binary-512-nl316-np15-rf20-random (query)         26_080.09     1_185.41    27_265.51       0.3714          1.0315         6.29
IVF-Binary-512-nl316-np17-rf10-random (query)         26_080.09     1_065.02    27_145.12       0.2466          1.0501         6.29
IVF-Binary-512-nl316-np17-rf20-random (query)         26_080.09     1_185.85    27_265.94       0.3692          1.0318         6.29
IVF-Binary-512-nl316-np25-rf10-random (query)         26_080.09     1_072.90    27_152.99       0.2450          1.0504         6.29
IVF-Binary-512-nl316-np25-rf20-random (query)         26_080.09     1_200.79    27_280.88       0.3672          1.0321         6.29
IVF-Binary-512-nl316-random (self)                    26_080.09     3_473.23    29_553.32       0.2475          1.0499         6.29
IVF-Binary-512-nl158-np7-rf0-pca (query)              32_233.73       897.96    33_131.68       0.1894             NaN         5.67
IVF-Binary-512-nl158-np12-rf0-pca (query)             32_233.73       907.13    33_140.85       0.1890             NaN         5.67
IVF-Binary-512-nl158-np17-rf0-pca (query)             32_233.73       918.09    33_151.82       0.1879             NaN         5.67
IVF-Binary-512-nl158-np7-rf10-pca (query)             32_233.73     1_027.61    33_261.34       0.5413          1.0172         5.67
IVF-Binary-512-nl158-np7-rf20-pca (query)             32_233.73     1_149.24    33_382.97       0.6899          1.0093         5.67
IVF-Binary-512-nl158-np12-rf10-pca (query)            32_233.73     1_038.71    33_272.44       0.5401          1.0172         5.67
IVF-Binary-512-nl158-np12-rf20-pca (query)            32_233.73     1_171.91    33_405.64       0.6907          1.0091         5.67
IVF-Binary-512-nl158-np17-rf10-pca (query)            32_233.73     1_053.66    33_287.39       0.5311          1.0178         5.67
IVF-Binary-512-nl158-np17-rf20-pca (query)            32_233.73     1_201.01    33_434.73       0.6781          1.0096         5.67
IVF-Binary-512-nl158-pca (self)                       32_233.73     3_404.36    35_638.09       0.5401          1.0171         5.67
IVF-Binary-512-nl223-np11-rf0-pca (query)             26_955.88       918.79    27_874.66       0.1897             NaN         5.93
IVF-Binary-512-nl223-np14-rf0-pca (query)             26_955.88       921.69    27_877.57       0.1891             NaN         5.93
IVF-Binary-512-nl223-np21-rf0-pca (query)             26_955.88       933.06    27_888.93       0.1881             NaN         5.93
IVF-Binary-512-nl223-np11-rf10-pca (query)            26_955.88     1_049.07    28_004.95       0.5512          1.0165         5.93
IVF-Binary-512-nl223-np11-rf20-pca (query)            26_955.88     1_174.32    28_130.20       0.7062          1.0084         5.93
IVF-Binary-512-nl223-np14-rf10-pca (query)            26_955.88     1_052.02    28_007.90       0.5470          1.0167         5.93
IVF-Binary-512-nl223-np14-rf20-pca (query)            26_955.88     1_182.84    28_138.72       0.7024          1.0086         5.93
IVF-Binary-512-nl223-np21-rf10-pca (query)            26_955.88     1_064.22    28_020.09       0.5373          1.0173         5.93
IVF-Binary-512-nl223-np21-rf20-pca (query)            26_955.88     1_200.41    28_156.29       0.6885          1.0091         5.93
IVF-Binary-512-nl223-pca (self)                       26_955.88     3_448.02    30_403.89       0.5468          1.0167         5.93
IVF-Binary-512-nl316-np15-rf0-pca (query)             27_770.60       963.34    28_733.94       0.1901             NaN         6.29
IVF-Binary-512-nl316-np17-rf0-pca (query)             27_770.60       953.41    28_724.01       0.1896             NaN         6.29
IVF-Binary-512-nl316-np25-rf0-pca (query)             27_770.60       962.68    28_733.28       0.1885             NaN         6.29
IVF-Binary-512-nl316-np15-rf10-pca (query)            27_770.60     1_089.84    28_860.44       0.5534          1.0163         6.29
IVF-Binary-512-nl316-np15-rf20-pca (query)            27_770.60     1_205.97    28_976.57       0.7106          1.0083         6.29
IVF-Binary-512-nl316-np17-rf10-pca (query)            27_770.60     1_083.59    28_854.19       0.5511          1.0165         6.29
IVF-Binary-512-nl316-np17-rf20-pca (query)            27_770.60     1_211.50    28_982.10       0.7081          1.0083         6.29
IVF-Binary-512-nl316-np25-rf10-pca (query)            27_770.60     1_092.58    28_863.19       0.5411          1.0171         6.29
IVF-Binary-512-nl316-np25-rf20-pca (query)            27_770.60     1_227.76    28_998.36       0.6938          1.0089         6.29
IVF-Binary-512-nl316-pca (self)                       27_770.60     3_542.98    31_313.58       0.5508          1.0164         6.29
IVF-Binary-1024-nl158-np7-rf0-random (query)          54_117.43     1_701.26    55_818.69       0.1092             NaN        10.72
IVF-Binary-1024-nl158-np12-rf0-random (query)         54_117.43     1_725.15    55_842.58       0.1083             NaN        10.72
IVF-Binary-1024-nl158-np17-rf0-random (query)         54_117.43     1_740.00    55_857.43       0.1079             NaN        10.72
IVF-Binary-1024-nl158-np7-rf10-random (query)         54_117.43     1_829.95    55_947.39       0.3132          1.0396        10.72
IVF-Binary-1024-nl158-np7-rf20-random (query)         54_117.43     1_943.45    56_060.89       0.4522          1.0242        10.72
IVF-Binary-1024-nl158-np12-rf10-random (query)        54_117.43     1_850.10    55_967.54       0.3114          1.0399        10.72
IVF-Binary-1024-nl158-np12-rf20-random (query)        54_117.43     1_970.93    56_088.37       0.4521          1.0242        10.72
IVF-Binary-1024-nl158-np17-rf10-random (query)        54_117.43     1_869.63    55_987.06       0.3099          1.0401        10.72
IVF-Binary-1024-nl158-np17-rf20-random (query)        54_117.43     1_995.59    56_113.02       0.4503          1.0243        10.72
IVF-Binary-1024-nl158-random (self)                   54_117.43     6_066.87    60_184.30       0.3115          1.0398        10.72
IVF-Binary-1024-nl223-np11-rf0-random (query)         48_874.46     1_724.45    50_598.90       0.1092             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-random (query)         48_874.46     1_760.69    50_635.15       0.1085             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-random (query)         48_874.46     1_746.01    50_620.47       0.1081             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-random (query)        48_874.46     1_848.26    50_722.72       0.3144          1.0394        10.98
IVF-Binary-1024-nl223-np11-rf20-random (query)        48_874.46     1_971.45    50_845.91       0.4556          1.0238        10.98
IVF-Binary-1024-nl223-np14-rf10-random (query)        48_874.46     1_851.72    50_726.18       0.3119          1.0398        10.98
IVF-Binary-1024-nl223-np14-rf20-random (query)        48_874.46     1_982.20    50_856.66       0.4520          1.0242        10.98
IVF-Binary-1024-nl223-np21-rf10-random (query)        48_874.46     1_870.05    50_744.50       0.3107          1.0400        10.98
IVF-Binary-1024-nl223-np21-rf20-random (query)        48_874.46     2_001.88    50_876.33       0.4505          1.0243        10.98
IVF-Binary-1024-nl223-random (self)                   48_874.46     6_103.82    54_978.27       0.3117          1.0397        10.98
IVF-Binary-1024-nl316-np15-rf0-random (query)         49_774.02     1_797.08    51_571.11       0.1089             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-random (query)         49_774.02     1_784.70    51_558.72       0.1086             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-random (query)         49_774.02     1_799.63    51_573.66       0.1082             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-random (query)        49_774.02     1_910.76    51_684.79       0.3131          1.0396        11.34
IVF-Binary-1024-nl316-np15-rf20-random (query)        49_774.02     2_037.16    51_811.19       0.4544          1.0239        11.34
IVF-Binary-1024-nl316-np17-rf10-random (query)        49_774.02     1_905.78    51_679.80       0.3118          1.0398        11.34
IVF-Binary-1024-nl316-np17-rf20-random (query)        49_774.02     2_036.05    51_810.07       0.4525          1.0241        11.34
IVF-Binary-1024-nl316-np25-rf10-random (query)        49_774.02     1_927.03    51_701.05       0.3105          1.0400        11.34
IVF-Binary-1024-nl316-np25-rf20-random (query)        49_774.02     2_060.92    51_834.95       0.4510          1.0243        11.34
IVF-Binary-1024-nl316-random (self)                   49_774.02     6_278.98    56_053.00       0.3120          1.0397        11.34
IVF-Binary-1024-nl158-np7-rf0-pca (query)             55_617.82     1_734.88    57_352.70       0.2305             NaN        10.73
IVF-Binary-1024-nl158-np12-rf0-pca (query)            55_617.82     1_751.40    57_369.23       0.2299             NaN        10.73
IVF-Binary-1024-nl158-np17-rf0-pca (query)            55_617.82     1_768.72    57_386.55       0.2279             NaN        10.73
IVF-Binary-1024-nl158-np7-rf10-pca (query)            55_617.82     1_857.94    57_475.76       0.6227          1.0123        10.73
IVF-Binary-1024-nl158-np7-rf20-pca (query)            55_617.82     1_985.96    57_603.79       0.7603          1.0064        10.73
IVF-Binary-1024-nl158-np12-rf10-pca (query)           55_617.82     1_881.55    57_499.37       0.6215          1.0122        10.73
IVF-Binary-1024-nl158-np12-rf20-pca (query)           55_617.82     2_011.12    57_628.94       0.7634          1.0061        10.73
IVF-Binary-1024-nl158-np17-rf10-pca (query)           55_617.82     1_899.91    57_517.73       0.6081          1.0129        10.73
IVF-Binary-1024-nl158-np17-rf20-pca (query)           55_617.82     2_209.99    57_827.81       0.7480          1.0066        10.73
IVF-Binary-1024-nl158-pca (self)                      55_617.82     6_211.51    61_829.34       0.6213          1.0122        10.73
IVF-Binary-1024-nl223-np11-rf0-pca (query)            50_396.88     1_766.62    52_163.49       0.2323             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-pca (query)            50_396.88     1_785.15    52_182.03       0.2311             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-pca (query)            50_396.88     1_783.42    52_180.29       0.2290             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-pca (query)           50_396.88     1_889.01    52_285.89       0.6380          1.0114        10.98
IVF-Binary-1024-nl223-np11-rf20-pca (query)           50_396.88     2_006.96    52_403.84       0.7854          1.0053        10.98
IVF-Binary-1024-nl223-np14-rf10-pca (query)           50_396.88     1_889.66    52_286.53       0.6338          1.0115        10.98
IVF-Binary-1024-nl223-np14-rf20-pca (query)           50_396.88     2_022.74    52_419.62       0.7819          1.0054        10.98
IVF-Binary-1024-nl223-np21-rf10-pca (query)           50_396.88     1_910.94    52_307.81       0.6192          1.0123        10.98
IVF-Binary-1024-nl223-np21-rf20-pca (query)           50_396.88     2_051.10    52_447.98       0.7626          1.0060        10.98
IVF-Binary-1024-nl223-pca (self)                      50_396.88     6_322.48    56_719.36       0.6337          1.0115        10.98
IVF-Binary-1024-nl316-np15-rf0-pca (query)            51_080.07     1_799.02    52_879.09       0.2324             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-pca (query)            51_080.07     1_813.85    52_893.92       0.2320             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-pca (query)            51_080.07     1_809.00    52_889.06       0.2301             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-pca (query)           51_080.07     1_913.01    52_993.07       0.6421          1.0112        11.34
IVF-Binary-1024-nl316-np15-rf20-pca (query)           51_080.07     2_049.72    53_129.79       0.7914          1.0051        11.34
IVF-Binary-1024-nl316-np17-rf10-pca (query)           51_080.07     1_915.11    52_995.17       0.6396          1.0113        11.34
IVF-Binary-1024-nl316-np17-rf20-pca (query)           51_080.07     2_052.75    53_132.82       0.7890          1.0051        11.34
IVF-Binary-1024-nl316-np25-rf10-pca (query)           51_080.07     1_928.98    53_009.05       0.6251          1.0120        11.34
IVF-Binary-1024-nl316-np25-rf20-pca (query)           51_080.07     2_068.81    53_148.88       0.7701          1.0058        11.34
IVF-Binary-1024-nl316-pca (self)                      51_080.07     6_324.68    57_404.75       0.6390          1.0112        11.34
IVF-Binary-1024-nl158-np7-rf0-signed (query)          54_075.95     1_726.03    55_801.97       0.1092             NaN        10.72
IVF-Binary-1024-nl158-np12-rf0-signed (query)         54_075.95     1_746.42    55_822.37       0.1083             NaN        10.72
IVF-Binary-1024-nl158-np17-rf0-signed (query)         54_075.95     1_746.96    55_822.91       0.1079             NaN        10.72
IVF-Binary-1024-nl158-np7-rf10-signed (query)         54_075.95     1_872.48    55_948.43       0.3132          1.0396        10.72
IVF-Binary-1024-nl158-np7-rf20-signed (query)         54_075.95     1_981.79    56_057.73       0.4522          1.0242        10.72
IVF-Binary-1024-nl158-np12-rf10-signed (query)        54_075.95     1_882.50    55_958.45       0.3114          1.0399        10.72
IVF-Binary-1024-nl158-np12-rf20-signed (query)        54_075.95     2_005.25    56_081.20       0.4521          1.0242        10.72
IVF-Binary-1024-nl158-np17-rf10-signed (query)        54_075.95     1_893.26    55_969.20       0.3099          1.0401        10.72
IVF-Binary-1024-nl158-np17-rf20-signed (query)        54_075.95     2_040.24    56_116.19       0.4503          1.0243        10.72
IVF-Binary-1024-nl158-signed (self)                   54_075.95     6_159.55    60_235.50       0.3115          1.0398        10.72
IVF-Binary-1024-nl223-np11-rf0-signed (query)         49_614.24     1_736.81    51_351.04       0.1092             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-signed (query)         49_614.24     1_754.28    51_368.52       0.1085             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-signed (query)         49_614.24     1_762.06    51_376.30       0.1081             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-signed (query)        49_614.24     1_854.08    51_468.32       0.3144          1.0394        10.98
IVF-Binary-1024-nl223-np11-rf20-signed (query)        49_614.24     2_016.69    51_630.93       0.4556          1.0238        10.98
IVF-Binary-1024-nl223-np14-rf10-signed (query)        49_614.24     1_866.70    51_480.94       0.3119          1.0398        10.98
IVF-Binary-1024-nl223-np14-rf20-signed (query)        49_614.24     1_993.59    51_607.83       0.4520          1.0242        10.98
IVF-Binary-1024-nl223-np21-rf10-signed (query)        49_614.24     1_878.59    51_492.82       0.3107          1.0400        10.98
IVF-Binary-1024-nl223-np21-rf20-signed (query)        49_614.24     2_014.60    51_628.84       0.4505          1.0243        10.98
IVF-Binary-1024-nl223-signed (self)                   49_614.24     6_133.19    55_747.43       0.3117          1.0397        10.98
IVF-Binary-1024-nl316-np15-rf0-signed (query)         49_562.20     1_779.22    51_341.43       0.1089             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-signed (query)         49_562.20     1_764.00    51_326.20       0.1086             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-signed (query)         49_562.20     1_777.45    51_339.65       0.1082             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-signed (query)        49_562.20     1_882.89    51_445.09       0.3131          1.0396        11.34
IVF-Binary-1024-nl316-np15-rf20-signed (query)        49_562.20     1_997.91    51_560.12       0.4544          1.0239        11.34
IVF-Binary-1024-nl316-np17-rf10-signed (query)        49_562.20     1_876.98    51_439.19       0.3118          1.0398        11.34
IVF-Binary-1024-nl316-np17-rf20-signed (query)        49_562.20     2_005.51    51_567.71       0.4525          1.0241        11.34
IVF-Binary-1024-nl316-np25-rf10-signed (query)        49_562.20     1_892.50    51_454.71       0.3105          1.0400        11.34
IVF-Binary-1024-nl316-np25-rf20-signed (query)        49_562.20     2_036.74    51_598.95       0.4510          1.0243        11.34
IVF-Binary-1024-nl316-signed (self)                   49_562.20     6_216.37    55_778.58       0.3120          1.0397        11.34
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
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         9.64     4_151.11     4_160.75       1.0000          1.0000        48.83
Exhaustive (self)                                          9.64    13_986.17    13_995.81       1.0000          1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_580.84       250.66     2_831.50       0.0796             NaN         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_580.84       347.27     2_928.11       0.2931          1.1776         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_580.84       451.20     3_032.04       0.4374          1.1037         1.78
ExhaustiveBinary-256-random (self)                     2_580.84     1_161.17     3_742.01       0.2929          1.1769         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_766.87       255.58     3_022.44       0.0827             NaN         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_766.87       363.39     3_130.26       0.1182          3.9214         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_766.87       467.75     3_234.62       0.1432          1.4338         1.78
ExhaustiveBinary-256-pca (self)                        2_766.87     1_198.84     3_965.71       0.1157          6.7403         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_085.52       446.80     5_532.32       0.1270             NaN         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_085.52       559.15     5_644.67       0.3949          1.1228         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_085.52       671.52     5_757.05       0.5575          1.0675         3.55
ExhaustiveBinary-512-random (self)                     5_085.52     1_825.83     6_911.35       0.3929          1.1237         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_462.75       479.94     5_942.69       0.0965             NaN         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_462.75       572.49     6_035.25       0.2973          1.1772         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_462.75       693.83     6_156.58       0.4534          1.0999         3.55
ExhaustiveBinary-512-pca (self)                        5_462.75     1_890.97     7_353.73       0.2988          1.1773         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_173.86       764.87    10_938.73       0.1773             NaN         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_173.86       879.77    11_053.64       0.5381          1.0724         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_173.86       995.75    11_169.61       0.7086          1.0349         7.10
ExhaustiveBinary-1024-random (self)                   10_173.86     2_892.46    13_066.33       0.5368          1.0726         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_649.37       804.86    11_454.23       0.1035             NaN         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_649.37       889.92    11_539.29       0.3158          1.1656         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_649.37     1_011.08    11_660.45       0.4741          1.0929         7.10
ExhaustiveBinary-1024-pca (self)                      10_649.37     2_953.07    13_602.44       0.3173          1.1657         7.10
ExhaustiveBinary-256-signed_no_rr (query)              2_541.34       247.24     2_788.58       0.0796             NaN         1.78
ExhaustiveBinary-256-signed-rf10 (query)               2_541.34       346.29     2_887.63       0.2931          1.1776         1.78
ExhaustiveBinary-256-signed-rf20 (query)               2_541.34       445.86     2_987.19       0.4374          1.1037         1.78
ExhaustiveBinary-256-signed (self)                     2_541.34     1_139.80     3_681.13       0.2929          1.1769         1.78
IVF-Binary-256-nl158-np7-rf0-random (query)            4_038.03       111.30     4_149.33       0.1023             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           4_038.03       115.83     4_153.86       0.0908             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           4_038.03       121.34     4_159.38       0.0812             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           4_038.03       167.34     4_205.37       0.3326          1.1542         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           4_038.03       214.01     4_252.05       0.4815          1.0896         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          4_038.03       180.97     4_219.00       0.3113          1.1675         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          4_038.03       223.59     4_261.63       0.4578          1.0975         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          4_038.03       182.53     4_220.57       0.2976          1.1763         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          4_038.03       237.54     4_275.57       0.4434          1.1027         1.93
IVF-Binary-256-nl158-random (self)                     4_038.03       520.21     4_558.24       0.3113          1.1669         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           3_013.34       116.04     3_129.38       0.0981             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           3_013.34       119.84     3_133.18       0.0871             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           3_013.34       125.59     3_138.92       0.0818             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          3_013.34       172.37     3_185.71       0.3258          1.1579         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          3_013.34       222.00     3_235.34       0.4750          1.0915         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          3_013.34       175.89     3_189.23       0.3093          1.1684         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          3_013.34       229.84     3_243.18       0.4555          1.0983         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          3_013.34       185.09     3_198.43       0.2997          1.1749         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          3_013.34       242.51     3_255.85       0.4446          1.1021         2.00
IVF-Binary-256-nl223-random (self)                     3_013.34       533.79     3_547.13       0.3094          1.1678         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           3_244.57       129.74     3_374.31       0.0902             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           3_244.57       124.92     3_369.48       0.0873             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           3_244.57       128.04     3_372.60       0.0830             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          3_244.57       178.33     3_422.90       0.3190          1.1618         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          3_244.57       229.13     3_473.69       0.4684          1.0938         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          3_244.57       178.64     3_423.21       0.3139          1.1649         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          3_244.57       232.96     3_477.53       0.4615          1.0961         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          3_244.57       187.98     3_432.55       0.3025          1.1726         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          3_244.57       243.94     3_488.50       0.4481          1.1006         2.09
IVF-Binary-256-nl316-random (self)                     3_244.57       546.20     3_790.76       0.3137          1.1645         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               4_230.62       118.70     4_349.32       0.0939             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              4_230.62       123.97     4_354.59       0.0914             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              4_230.62       128.71     4_359.33       0.0903             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              4_230.62       184.22     4_414.84       0.2815          1.1901         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              4_230.62       242.94     4_473.56       0.4291          1.1101         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             4_230.62       193.87     4_424.49       0.2368          1.2309         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             4_230.62       259.76     4_490.38       0.3498          1.1457         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             4_230.62       204.68     4_435.30       0.2144          1.2578         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             4_230.62       276.78     4_507.40       0.3095          1.1698         1.93
IVF-Binary-256-nl158-pca (self)                        4_230.62       617.39     4_848.01       0.2370          1.2324         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              3_279.76       122.99     3_402.75       0.0937             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              3_279.76       126.98     3_406.75       0.0921             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              3_279.76       140.62     3_420.38       0.0906             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             3_279.76       196.63     3_476.39       0.2741          1.1960         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             3_279.76       255.14     3_534.91       0.4164          1.1152         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             3_279.76       199.17     3_478.93       0.2535          1.2142         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             3_279.76       264.66     3_544.42       0.3798          1.1311         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             3_279.76       212.68     3_492.44       0.2220          1.2482         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             3_279.76       281.69     3_561.45       0.3231          1.1612         2.00
IVF-Binary-256-nl223-pca (self)                        3_279.76       627.49     3_907.26       0.2542          1.2150         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              3_426.79       131.59     3_558.38       0.0934             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              3_426.79       130.18     3_556.97       0.0927             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              3_426.79       137.66     3_564.46       0.0910             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             3_426.79       205.15     3_631.94       0.2767          1.1930         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             3_426.79       261.03     3_687.83       0.4210          1.1124         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             3_426.79       203.57     3_630.36       0.2653          1.2024         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             3_426.79       264.69     3_691.49       0.4009          1.1206         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             3_426.79       211.60     3_638.40       0.2333          1.2342         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             3_426.79       280.41     3_707.20       0.3440          1.1485         2.09
IVF-Binary-256-nl316-pca (self)                        3_426.79       636.79     4_063.58       0.2664          1.2029         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_526.73       206.36     6_733.09       0.1365             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_526.73       212.31     6_739.04       0.1314             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_526.73       222.77     6_749.50       0.1280             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_526.73       263.20     6_789.94       0.4194          1.1128         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_526.73       312.42     6_839.15       0.5824          1.0615         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_526.73       274.13     6_800.86       0.4052          1.1187         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_526.73       329.49     6_856.22       0.5669          1.0653         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_526.73       286.85     6_813.58       0.3969          1.1224         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_526.73       345.42     6_872.15       0.5596          1.0672         3.71
IVF-Binary-512-nl158-random (self)                     6_526.73       861.09     7_387.82       0.4039          1.1194         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_477.40       213.16     5_690.56       0.1345             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_477.40       213.33     5_690.73       0.1304             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_477.40       224.56     5_701.96       0.1281             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_477.40       268.47     5_745.87       0.4149          1.1145         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_477.40       319.11     5_796.51       0.5792          1.0622         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_477.40       274.45     5_751.85       0.4037          1.1192         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_477.40       329.93     5_807.32       0.5672          1.0653         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_477.40       287.71     5_765.11       0.3977          1.1218         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_477.40       346.82     5_824.22       0.5610          1.0669         3.77
IVF-Binary-512-nl223-random (self)                     5_477.40       867.78     6_345.18       0.4027          1.1198         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           5_659.32       215.52     5_874.84       0.1326             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           5_659.32       217.90     5_877.22       0.1312             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           5_659.32       226.08     5_885.40       0.1288             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          5_659.32       277.25     5_936.57       0.4112          1.1160         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          5_659.32       329.16     5_988.48       0.5748          1.0633         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          5_659.32       275.96     5_935.28       0.4073          1.1177         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          5_659.32       330.03     5_989.35       0.5696          1.0646         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          5_659.32       287.59     5_946.91       0.4001          1.1208         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          5_659.32       345.90     6_005.22       0.5618          1.0666         3.86
IVF-Binary-512-nl316-random (self)                     5_659.32       881.69     6_541.01       0.4059          1.1185         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               6_800.88       215.90     7_016.79       0.0978             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              6_800.88       222.15     7_023.03       0.0966             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              6_800.88       230.31     7_031.19       0.0965             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              6_800.88       283.03     7_083.91       0.3062          1.1716         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              6_800.88       337.25     7_138.13       0.4660          1.0957         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             6_800.88       290.46     7_091.34       0.2974          1.1771         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             6_800.88       348.81     7_149.69       0.4534          1.0999         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             6_800.88       304.71     7_105.59       0.2974          1.1771         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             6_800.88       369.42     7_170.31       0.4534          1.0999         3.71
IVF-Binary-512-nl158-pca (self)                        6_800.88       926.10     7_726.98       0.2989          1.1772         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              5_785.77       218.63     6_004.40       0.0973             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              5_785.77       230.26     6_016.03       0.0966             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              5_785.77       236.08     6_021.85       0.0965             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             5_785.77       288.08     6_073.85       0.3028          1.1736         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             5_785.77       342.44     6_128.21       0.4616          1.0971         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             5_785.77       293.24     6_079.01       0.2978          1.1769         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             5_785.77       353.07     6_138.85       0.4540          1.0997         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             5_785.77       305.45     6_091.22       0.2974          1.1771         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             5_785.77       370.98     6_156.75       0.4535          1.0999         3.77
IVF-Binary-512-nl223-pca (self)                        5_785.77       933.66     6_719.43       0.2991          1.1770         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              5_977.46       225.77     6_203.23       0.0971             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              5_977.46       229.53     6_206.99       0.0967             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              5_977.46       236.15     6_213.61       0.0965             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             5_977.46       296.86     6_274.32       0.3015          1.1745         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             5_977.46       366.36     6_343.82       0.4592          1.0979         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             5_977.46       310.60     6_288.06       0.2987          1.1763         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             5_977.46       359.51     6_336.97       0.4552          1.0993         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             5_977.46       312.36     6_289.83       0.2975          1.1771         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             5_977.46       373.38     6_350.84       0.4535          1.0999         3.86
IVF-Binary-512-nl316-pca (self)                        5_977.46       948.68     6_926.14       0.3000          1.1764         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_443.97       391.43    11_835.40       0.1846             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_443.97       404.56    11_848.52       0.1806             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_443.97       415.65    11_859.62       0.1781             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_443.97       453.66    11_897.63       0.5530          1.0684         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_443.97       506.80    11_950.76       0.7207          1.0329         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_443.97       468.83    11_912.80       0.5441          1.0708         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_443.97       528.67    11_972.64       0.7125          1.0343         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_443.97       491.31    11_935.28       0.5400          1.0719         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_443.97       552.70    11_996.67       0.7094          1.0348         7.26
IVF-Binary-1024-nl158-random (self)                   11_443.97     1_527.09    12_971.06       0.5426          1.0711         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_423.69       393.57    10_817.27       0.1829             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_423.69       401.58    10_825.27       0.1800             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_423.69       414.66    10_838.35       0.1782             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_423.69       459.68    10_883.38       0.5508          1.0688         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_423.69       511.12    10_934.81       0.7205          1.0329         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_423.69       466.62    10_890.32       0.5438          1.0709         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_423.69       525.86    10_949.56       0.7139          1.0340         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_423.69       485.18    10_908.87       0.5403          1.0719         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_423.69       550.66    10_974.35       0.7110          1.0345         7.32
IVF-Binary-1024-nl223-random (self)                   10_423.69     1_508.23    11_931.92       0.5425          1.0711         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         10_635.17       400.13    11_035.31       0.1812             NaN         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)         10_635.17       401.25    11_036.43       0.1802             NaN         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)         10_635.17       415.89    11_051.06       0.1784             NaN         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)        10_635.17       465.06    11_100.24       0.5480          1.0697         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)        10_635.17       518.25    11_153.42       0.7182          1.0333         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)        10_635.17       466.03    11_101.20       0.5453          1.0705         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)        10_635.17       533.51    11_168.69       0.7151          1.0338         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)        10_635.17       482.58    11_117.76       0.5407          1.0718         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)        10_635.17       544.83    11_180.00       0.7111          1.0345         7.41
IVF-Binary-1024-nl316-random (self)                   10_635.17     1_508.02    12_143.19       0.5443          1.0706         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)             11_949.58       405.50    12_355.08       0.1049             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            11_949.58       418.33    12_367.92       0.1036             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            11_949.58       431.56    12_381.14       0.1035             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            11_949.58       473.41    12_423.00       0.3243          1.1605         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            11_949.58       545.61    12_495.19       0.4867          1.0889         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           11_949.58       497.61    12_447.19       0.3156          1.1656         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           11_949.58       551.23    12_500.81       0.4742          1.0929         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           11_949.58       507.26    12_456.84       0.3156          1.1656         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           11_949.58       582.06    12_531.65       0.4742          1.0929         7.26
IVF-Binary-1024-nl158-pca (self)                      11_949.58     1_584.00    13_533.58       0.3173          1.1657         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            10_919.42       408.78    11_328.20       0.1044             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            10_919.42       415.16    11_334.58       0.1036             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            10_919.42       431.59    11_351.01       0.1035             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           10_919.42       481.39    11_400.81       0.3211          1.1623         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           10_919.42       539.10    11_458.52       0.4821          1.0904         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           10_919.42       488.73    11_408.15       0.3161          1.1653         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           10_919.42       552.64    11_472.05       0.4747          1.0927         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           10_919.42       507.01    11_426.42       0.3158          1.1656         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           10_919.42       572.33    11_491.75       0.4741          1.0929         7.32
IVF-Binary-1024-nl223-pca (self)                      10_919.42     1_586.32    12_505.73       0.3176          1.1655         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            11_115.89       415.83    11_531.72       0.1041             NaN         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            11_115.89       416.73    11_532.62       0.1037             NaN         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            11_115.89       430.20    11_546.09       0.1035             NaN         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           11_115.89       485.93    11_601.82       0.3198          1.1632         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           11_115.89       545.08    11_660.97       0.4799          1.0911         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           11_115.89       488.00    11_603.89       0.3170          1.1648         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           11_115.89       550.25    11_666.14       0.4759          1.0923         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           11_115.89       503.62    11_619.51       0.3159          1.1656         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           11_115.89       568.30    11_684.19       0.4741          1.0929         7.42
IVF-Binary-1024-nl316-pca (self)                      11_115.89     1_597.68    12_713.57       0.3185          1.1649         7.42
IVF-Binary-256-nl158-np7-rf0-signed (query)            4_054.18       110.79     4_164.97       0.1023             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-signed (query)           4_054.18       115.66     4_169.84       0.0908             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-signed (query)           4_054.18       121.02     4_175.20       0.0812             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-signed (query)           4_054.18       165.83     4_220.01       0.3326          1.1542         1.93
IVF-Binary-256-nl158-np7-rf20-signed (query)           4_054.18       219.29     4_273.47       0.4815          1.0896         1.93
IVF-Binary-256-nl158-np12-rf10-signed (query)          4_054.18       175.19     4_229.37       0.3113          1.1675         1.93
IVF-Binary-256-nl158-np12-rf20-signed (query)          4_054.18       224.73     4_278.91       0.4578          1.0975         1.93
IVF-Binary-256-nl158-np17-rf10-signed (query)          4_054.18       182.28     4_236.46       0.2976          1.1763         1.93
IVF-Binary-256-nl158-np17-rf20-signed (query)          4_054.18       239.89     4_294.07       0.4434          1.1027         1.93
IVF-Binary-256-nl158-signed (self)                     4_054.18       520.52     4_574.70       0.3113          1.1669         1.93
IVF-Binary-256-nl223-np11-rf0-signed (query)           3_012.12       118.20     3_130.31       0.0981             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-signed (query)           3_012.12       123.16     3_135.27       0.0871             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-signed (query)           3_012.12       126.77     3_138.88       0.0818             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-signed (query)          3_012.12       174.57     3_186.69       0.3258          1.1579         2.00
IVF-Binary-256-nl223-np11-rf20-signed (query)          3_012.12       223.50     3_235.61       0.4750          1.0915         2.00
IVF-Binary-256-nl223-np14-rf10-signed (query)          3_012.12       178.31     3_190.43       0.3093          1.1684         2.00
IVF-Binary-256-nl223-np14-rf20-signed (query)          3_012.12       233.48     3_245.60       0.4555          1.0983         2.00
IVF-Binary-256-nl223-np21-rf10-signed (query)          3_012.12       188.66     3_200.78       0.2997          1.1749         2.00
IVF-Binary-256-nl223-np21-rf20-signed (query)          3_012.12       245.72     3_257.84       0.4446          1.1021         2.00
IVF-Binary-256-nl223-signed (self)                     3_012.12       541.03     3_553.15       0.3094          1.1678         2.00
IVF-Binary-256-nl316-np15-rf0-signed (query)           3_210.78       122.14     3_332.92       0.0902             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-signed (query)           3_210.78       123.29     3_334.07       0.0873             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-signed (query)           3_210.78       130.59     3_341.37       0.0830             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-signed (query)          3_210.78       182.21     3_392.99       0.3190          1.1618         2.09
IVF-Binary-256-nl316-np15-rf20-signed (query)          3_210.78       230.23     3_441.01       0.4684          1.0938         2.09
IVF-Binary-256-nl316-np17-rf10-signed (query)          3_210.78       179.35     3_390.13       0.3139          1.1649         2.09
IVF-Binary-256-nl316-np17-rf20-signed (query)          3_210.78       231.75     3_442.53       0.4615          1.0961         2.09
IVF-Binary-256-nl316-np25-rf10-signed (query)          3_210.78       187.14     3_397.92       0.3025          1.1726         2.09
IVF-Binary-256-nl316-np25-rf20-signed (query)          3_210.78       243.80     3_454.57       0.4481          1.1006         2.09
IVF-Binary-256-nl316-signed (self)                     3_210.78       545.73     3_756.51       0.3137          1.1645         2.09
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
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        20.48     9_683.96     9_704.45       1.0000          1.0000        97.66
Exhaustive (self)                                         20.48    32_152.62    32_173.11       1.0000          1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_657.24       350.39     6_007.63       0.0472             NaN         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_657.24       467.54     6_124.78       0.2012          1.1663         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_657.24       592.17     6_249.41       0.3224          1.1017         2.03
ExhaustiveBinary-256-random (self)                     5_657.24     1_537.93     7_195.17       0.2020          1.1654         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_067.86       362.04     6_429.90       0.1539             NaN         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_067.86       530.16     6_598.03       0.3795          1.0909         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_067.86       627.64     6_695.51       0.4855          1.0578         2.03
ExhaustiveBinary-256-pca (self)                        6_067.86     1_626.44     7_694.30       0.3796          1.0913         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_079.47       659.31    11_738.78       0.0860             NaN         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_079.47       776.30    11_855.76       0.2648          1.1242         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_079.47       907.89    11_987.36       0.3980          1.0752         4.05
ExhaustiveBinary-512-random (self)                    11_079.47     2_613.53    13_693.00       0.2643          1.1240         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_621.82       667.63    12_289.45       0.1171             NaN         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_621.82       798.94    12_420.76       0.2880         21.4934         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_621.82       935.04    12_556.85       0.3835          1.0941         4.05
ExhaustiveBinary-512-pca (self)                       11_621.82     2_853.08    14_474.90       0.2876         22.0894         4.05
ExhaustiveBinary-1024-random_no_rr (query)            21_911.47     1_169.40    23_080.88       0.1148             NaN         8.10
ExhaustiveBinary-1024-random-rf10 (query)             21_911.47     1_330.09    23_241.57       0.3455          1.0925         8.10
ExhaustiveBinary-1024-random-rf20 (query)             21_911.47     1_454.48    23_365.95       0.4981          1.0533         8.10
ExhaustiveBinary-1024-random (self)                   21_911.47     4_344.10    26_255.57       0.3445          1.0928         8.10
ExhaustiveBinary-1024-pca_no_rr (query)               22_916.32     1_193.19    24_109.51       0.2416             NaN         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                22_916.32     1_345.19    24_261.51       0.6860          1.0240         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                22_916.32     1_493.70    24_410.02       0.8390          1.0095         8.11
ExhaustiveBinary-1024-pca (self)                      22_916.32     4_466.23    27_382.55       0.6864          1.0239         8.11
ExhaustiveBinary-512-signed_no_rr (query)             11_093.72       653.13    11_746.85       0.0860             NaN         4.05
ExhaustiveBinary-512-signed-rf10 (query)              11_093.72       829.31    11_923.03       0.2648          1.1242         4.05
ExhaustiveBinary-512-signed-rf20 (query)              11_093.72       901.33    11_995.04       0.3980          1.0752         4.05
ExhaustiveBinary-512-signed (self)                    11_093.72     2_557.63    13_651.35       0.2643          1.1240         4.05
IVF-Binary-256-nl158-np7-rf0-random (query)            8_700.46       232.42     8_932.87       0.0675             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           8_700.46       235.08     8_935.54       0.0586             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           8_700.46       241.88     8_942.34       0.0493             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           8_700.46       311.93     9_012.39       0.2432          1.1377         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           8_700.46       404.15     9_104.60       0.3698          1.0844         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          8_700.46       314.83     9_015.29       0.2255          1.1497         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          8_700.46       402.11     9_102.57       0.3483          1.0921         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          8_700.46       325.44     9_025.90       0.2074          1.1635         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          8_700.46       412.28     9_112.73       0.3302          1.0996         2.34
IVF-Binary-256-nl158-random (self)                     8_700.46       971.61     9_672.07       0.2261          1.1490         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           6_490.70       236.06     6_726.75       0.0642             NaN         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)           6_490.70       238.52     6_729.22       0.0545             NaN         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)           6_490.70       243.25     6_733.95       0.0498             NaN         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)          6_490.70       324.35     6_815.05       0.2334          1.1444         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)          6_490.70       393.32     6_884.02       0.3573          1.0890         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)          6_490.70       317.85     6_808.55       0.2172          1.1566         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)          6_490.70       397.86     6_888.56       0.3403          1.0960         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)          6_490.70       323.96     6_814.66       0.2092          1.1621         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)          6_490.70       412.66     6_903.36       0.3311          1.0992         2.46
IVF-Binary-256-nl223-random (self)                     6_490.70       980.53     7_471.23       0.2181          1.1556         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)           6_848.43       252.60     7_101.03       0.0584             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           6_848.43       254.93     7_103.36       0.0547             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           6_848.43       260.26     7_108.69       0.0517             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          6_848.43       334.52     7_182.95       0.2294          1.1477         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          6_848.43       412.62     7_261.05       0.3567          1.0896         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          6_848.43       331.83     7_180.25       0.2238          1.1515         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          6_848.43       414.99     7_263.42       0.3497          1.0919         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          6_848.43       339.46     7_187.89       0.2156          1.1575         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          6_848.43       424.56     7_272.98       0.3381          1.0961         2.65
IVF-Binary-256-nl316-random (self)                     6_848.43     1_031.97     7_880.39       0.2241          1.1510         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               9_247.35       233.81     9_481.16       0.2085             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              9_247.35       239.51     9_486.86       0.2050             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              9_247.35       245.39     9_492.75       0.2025             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              9_247.35       327.59     9_574.94       0.6076          1.0348         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              9_247.35       410.73     9_658.09       0.7667          1.0158         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             9_247.35       358.66     9_606.01       0.5909          1.0374         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             9_247.35       425.93     9_673.28       0.7466          1.0178         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             9_247.35       345.40     9_592.75       0.5763          1.0398         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             9_247.35       441.16     9_688.51       0.7278          1.0197         2.34
IVF-Binary-256-nl158-pca (self)                        9_247.35     1_054.20    10_301.55       0.5917          1.0371         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              7_027.74       246.57     7_274.31       0.2082             NaN         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              7_027.74       247.13     7_274.87       0.2065             NaN         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              7_027.74       252.35     7_280.09       0.2030             NaN         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             7_027.74       337.61     7_365.36       0.6061          1.0350         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             7_027.74       423.86     7_451.60       0.7661          1.0159         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             7_027.74       342.12     7_369.87       0.5976          1.0363         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             7_027.74       432.52     7_460.26       0.7555          1.0169         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             7_027.74       355.55     7_383.29       0.5790          1.0393         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             7_027.74       444.62     7_472.36       0.7325          1.0192         2.47
IVF-Binary-256-nl223-pca (self)                        7_027.74     1_079.22     8_106.96       0.5986          1.0361         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              7_344.61       258.52     7_603.13       0.2082             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              7_344.61       262.78     7_607.40       0.2074             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              7_344.61       265.09     7_609.70       0.2045             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             7_344.61       354.06     7_698.67       0.6077          1.0348         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             7_344.61       439.01     7_783.63       0.7686          1.0157         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             7_344.61       352.15     7_696.77       0.6033          1.0355         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             7_344.61       441.03     7_785.64       0.7629          1.0162         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             7_344.61       359.71     7_704.32       0.5873          1.0380         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             7_344.61       453.63     7_798.24       0.7424          1.0182         2.65
IVF-Binary-256-nl316-pca (self)                        7_344.61     1_112.84     8_457.45       0.6043          1.0352         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           14_303.32       427.74    14_731.06       0.0977             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          14_303.32       436.18    14_739.50       0.0934             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          14_303.32       443.50    14_746.82       0.0872             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          14_303.32       510.37    14_813.69       0.2879          1.1143         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          14_303.32       586.08    14_889.40       0.4248          1.0690         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         14_303.32       522.74    14_826.06       0.2772          1.1189         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         14_303.32       600.97    14_904.29       0.4113          1.0721         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         14_303.32       533.66    14_836.98       0.2686          1.1230         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         14_303.32       619.04    14_922.36       0.4024          1.0744         4.36
IVF-Binary-512-nl158-random (self)                    14_303.32     1_658.20    15_961.52       0.2766          1.1188         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          12_014.22       443.52    12_457.74       0.0946             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          12_014.22       446.72    12_460.94       0.0906             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          12_014.22       455.73    12_469.95       0.0882             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         12_014.22       528.09    12_542.30       0.2815          1.1166         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         12_014.22       605.14    12_619.36       0.4178          1.0706         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         12_014.22       531.64    12_545.85       0.2738          1.1202         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         12_014.22       612.41    12_626.62       0.4088          1.0728         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         12_014.22       558.08    12_572.29       0.2690          1.1224         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         12_014.22       635.96    12_650.18       0.4029          1.0743         4.49
IVF-Binary-512-nl223-random (self)                    12_014.22     1_698.41    13_712.63       0.2736          1.1201         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          12_327.87       457.34    12_785.21       0.0928             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          12_327.87       461.47    12_789.34       0.0914             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          12_327.87       466.91    12_794.78       0.0894             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         12_327.87       539.38    12_867.25       0.2818          1.1164         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         12_327.87       621.24    12_949.11       0.4179          1.0705         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         12_327.87       539.80    12_867.67       0.2784          1.1179         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         12_327.87       625.07    12_952.94       0.4133          1.0716         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         12_327.87       550.68    12_878.56       0.2728          1.1206         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         12_327.87       635.35    12_963.22       0.4065          1.0733         4.67
IVF-Binary-512-nl316-random (self)                    12_327.87     1_737.39    14_065.26       0.2777          1.1180         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              14_904.69       436.46    15_341.16       0.2324             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             14_904.69       464.94    15_369.63       0.2185             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             14_904.69       482.92    15_387.61       0.2074             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             14_904.69       528.92    15_433.62       0.6593          1.0274         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             14_904.69       612.97    15_517.66       0.8106          1.0119         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            14_904.69       539.86    15_444.55       0.6165          1.0336         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            14_904.69       630.92    15_535.61       0.7678          1.0156         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            14_904.69       561.73    15_466.42       0.5819          1.0391         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            14_904.69       652.48    15_557.17       0.7304          1.0193         4.36
IVF-Binary-512-nl158-pca (self)                       14_904.69     1_752.91    16_657.60       0.6164          1.0336         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             12_613.48       446.63    13_060.11       0.2304             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             12_613.48       451.32    13_064.80       0.2235             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             12_613.48       459.35    13_072.83       0.2103             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            12_613.48       543.93    13_157.41       0.6552          1.0280         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            12_613.48       628.08    13_241.55       0.8087          1.0120         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            12_613.48       546.58    13_160.06       0.6324          1.0311         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            12_613.48       637.26    13_250.74       0.7856          1.0140         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            12_613.48       559.38    13_172.85       0.5919          1.0374         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            12_613.48       654.93    13_268.40       0.7416          1.0181         4.49
IVF-Binary-512-nl223-pca (self)                       12_613.48     1_773.30    14_386.78       0.6334          1.0310         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             12_968.12       466.73    13_434.85       0.2313             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             12_968.12       468.30    13_436.43       0.2274             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             12_968.12       474.50    13_442.63       0.2157             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            12_968.12       571.17    13_539.30       0.6597          1.0273         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            12_968.12       650.59    13_618.71       0.8134          1.0116         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            12_968.12       563.42    13_531.54       0.6481          1.0289         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            12_968.12       652.36    13_620.49       0.8017          1.0126         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            12_968.12       573.22    13_541.34       0.6084          1.0347         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            12_968.12       671.59    13_639.72       0.7600          1.0163         4.67
IVF-Binary-512-nl316-pca (self)                       12_968.12     1_816.96    14_785.08       0.6482          1.0289         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          25_168.59       823.52    25_992.12       0.1197             NaN         8.41
IVF-Binary-1024-nl158-np12-rf0-random (query)         25_168.59       835.94    26_004.54       0.1173             NaN         8.41
IVF-Binary-1024-nl158-np17-rf0-random (query)         25_168.59       862.98    26_031.58       0.1153             NaN         8.41
IVF-Binary-1024-nl158-np7-rf10-random (query)         25_168.59       909.33    26_077.93       0.3607          1.0878         8.41
IVF-Binary-1024-nl158-np7-rf20-random (query)         25_168.59       987.41    26_156.00       0.5146          1.0503         8.41
IVF-Binary-1024-nl158-np12-rf10-random (query)        25_168.59       926.36    26_094.96       0.3529          1.0902         8.41
IVF-Binary-1024-nl158-np12-rf20-random (query)        25_168.59     1_010.24    26_178.84       0.5055          1.0519         8.41
IVF-Binary-1024-nl158-np17-rf10-random (query)        25_168.59       947.07    26_115.66       0.3485          1.0917         8.41
IVF-Binary-1024-nl158-np17-rf20-random (query)        25_168.59     1_037.53    26_206.13       0.5006          1.0529         8.41
IVF-Binary-1024-nl158-random (self)                   25_168.59     3_017.37    28_185.96       0.3520          1.0905         8.41
IVF-Binary-1024-nl223-np11-rf0-random (query)         22_868.52       987.13    23_855.66       0.1184             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         22_868.52       934.00    23_802.52       0.1166             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         22_868.52       997.26    23_865.78       0.1155             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        22_868.52     1_006.01    23_874.53       0.3560          1.0892         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        22_868.52     1_098.47    23_966.99       0.5090          1.0513         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        22_868.52       945.48    23_814.01       0.3513          1.0908         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        22_868.52     1_061.12    23_929.64       0.5032          1.0523         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        22_868.52       993.79    23_862.32       0.3480          1.0919         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        22_868.52     1_066.78    23_935.31       0.4992          1.0531         8.54
IVF-Binary-1024-nl223-random (self)                   22_868.52     3_204.00    26_072.53       0.3504          1.0910         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         23_322.54       849.51    24_172.05       0.1184             NaN         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)         23_322.54       850.48    24_173.02       0.1177             NaN         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)         23_322.54       864.86    24_187.40       0.1167             NaN         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)        23_322.54       935.06    24_257.60       0.3561          1.0891         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)        23_322.54     1_020.71    24_343.26       0.5101          1.0511         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)        23_322.54       945.91    24_268.46       0.3535          1.0900         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)        23_322.54     1_026.70    24_349.25       0.5064          1.0517         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)        23_322.54       959.82    24_282.36       0.3495          1.0913         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)        23_322.54     1_041.98    24_364.52       0.5022          1.0525         8.72
IVF-Binary-1024-nl316-random (self)                   23_322.54     3_065.09    26_387.64       0.3524          1.0903         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)             26_072.64       839.74    26_912.39       0.2423             NaN         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            26_072.64       920.06    26_992.70       0.2418             NaN         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            26_072.64       867.68    26_940.32       0.2417             NaN         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            26_072.64       932.99    27_005.63       0.6848          1.0241         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            26_072.64     1_020.15    27_092.79       0.8347          1.0099         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           26_072.64       948.91    27_021.55       0.6863          1.0240         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           26_072.64     1_036.52    27_109.17       0.8391          1.0095         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           26_072.64     1_050.75    27_123.40       0.6863          1.0240         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           26_072.64     1_063.75    27_136.39       0.8391          1.0095         8.42
IVF-Binary-1024-nl158-pca (self)                      26_072.64     3_135.32    29_207.97       0.6865          1.0239         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            23_849.04       860.49    24_709.52       0.2423             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            23_849.04       867.96    24_717.00       0.2420             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            23_849.04       885.34    24_734.38       0.2419             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           23_849.04       958.97    24_808.01       0.6864          1.0239         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           23_849.04     1_050.62    24_899.65       0.8383          1.0096         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           23_849.04       971.28    24_820.31       0.6864          1.0240         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           23_849.04     1_057.25    24_906.28       0.8391          1.0095         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           23_849.04       989.37    24_838.41       0.6862          1.0240         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           23_849.04     1_080.67    24_929.71       0.8390          1.0095         8.54
IVF-Binary-1024-nl223-pca (self)                      23_849.04     3_154.30    27_003.34       0.6866          1.0239         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            24_172.28       873.47    25_045.75       0.2422             NaN         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            24_172.28       884.58    25_056.86       0.2419             NaN         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            24_172.28       911.62    25_083.90       0.2417             NaN         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           24_172.28       970.71    25_142.99       0.6866          1.0239         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           24_172.28     1_061.64    25_233.92       0.8392          1.0095         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           24_172.28       974.66    25_146.94       0.6864          1.0239         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           24_172.28     1_062.76    25_235.04       0.8392          1.0095         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           24_172.28       991.84    25_164.12       0.6861          1.0240         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           24_172.28     1_084.05    25_256.33       0.8390          1.0095         8.73
IVF-Binary-1024-nl316-pca (self)                      24_172.28     3_182.23    27_354.51       0.6867          1.0239         8.73
IVF-Binary-512-nl158-np7-rf0-signed (query)           14_174.26       426.76    14_601.02       0.0977             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-signed (query)          14_174.26       438.49    14_612.75       0.0934             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-signed (query)          14_174.26       451.36    14_625.63       0.0872             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-signed (query)          14_174.26       519.17    14_693.44       0.2879          1.1143         4.36
IVF-Binary-512-nl158-np7-rf20-signed (query)          14_174.26       595.22    14_769.49       0.4248          1.0690         4.36
IVF-Binary-512-nl158-np12-rf10-signed (query)         14_174.26       531.81    14_706.07       0.2772          1.1189         4.36
IVF-Binary-512-nl158-np12-rf20-signed (query)         14_174.26       611.74    14_786.00       0.4113          1.0721         4.36
IVF-Binary-512-nl158-np17-rf10-signed (query)         14_174.26       545.85    14_720.11       0.2686          1.1230         4.36
IVF-Binary-512-nl158-np17-rf20-signed (query)         14_174.26       631.46    14_805.73       0.4024          1.0744         4.36
IVF-Binary-512-nl158-signed (self)                    14_174.26     1_688.42    15_862.68       0.2766          1.1188         4.36
IVF-Binary-512-nl223-np11-rf0-signed (query)          11_958.75       439.81    12_398.56       0.0946             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-signed (query)          11_958.75       445.28    12_404.03       0.0906             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-signed (query)          11_958.75       444.74    12_403.49       0.0882             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-signed (query)         11_958.75       515.59    12_474.34       0.2815          1.1166         4.49
IVF-Binary-512-nl223-np11-rf20-signed (query)         11_958.75       592.77    12_551.52       0.4178          1.0706         4.49
IVF-Binary-512-nl223-np14-rf10-signed (query)         11_958.75       522.56    12_481.31       0.2738          1.1202         4.49
IVF-Binary-512-nl223-np14-rf20-signed (query)         11_958.75       607.73    12_566.48       0.4088          1.0728         4.49
IVF-Binary-512-nl223-np21-rf10-signed (query)         11_958.75       531.92    12_490.67       0.2690          1.1224         4.49
IVF-Binary-512-nl223-np21-rf20-signed (query)         11_958.75       615.54    12_574.29       0.4029          1.0743         4.49
IVF-Binary-512-nl223-signed (self)                    11_958.75     1_666.99    13_625.74       0.2736          1.1201         4.49
IVF-Binary-512-nl316-np15-rf0-signed (query)          12_300.64       448.31    12_748.94       0.0928             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-signed (query)          12_300.64       447.74    12_748.37       0.0914             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-signed (query)          12_300.64       457.57    12_758.21       0.0894             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-signed (query)         12_300.64       530.47    12_831.10       0.2818          1.1164         4.67
IVF-Binary-512-nl316-np15-rf20-signed (query)         12_300.64       608.42    12_909.06       0.4179          1.0705         4.67
IVF-Binary-512-nl316-np17-rf10-signed (query)         12_300.64       538.78    12_839.41       0.2784          1.1179         4.67
IVF-Binary-512-nl316-np17-rf20-signed (query)         12_300.64       613.73    12_914.37       0.4133          1.0716         4.67
IVF-Binary-512-nl316-np25-rf10-signed (query)         12_300.64       547.89    12_848.53       0.2728          1.1206         4.67
IVF-Binary-512-nl316-np25-rf20-signed (query)         12_300.64       627.66    12_928.29       0.4065          1.0733         4.67
IVF-Binary-512-nl316-signed (self)                    12_300.64     1_829.23    14_129.87       0.2777          1.1180         4.67
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
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        40.15    21_899.14    21_939.29       1.0000          1.0000       195.31
Exhaustive (self)                                         40.15    72_906.05    72_946.20       1.0000          1.0000       195.31
ExhaustiveBinary-256-random_no_rr (query)             11_945.08       563.80    12_508.88       0.0290             NaN         2.53
ExhaustiveBinary-256-random-rf10 (query)              11_945.08       713.44    12_658.52       0.1514          1.1441         2.53
ExhaustiveBinary-256-random-rf20 (query)              11_945.08       876.67    12_821.75       0.2535          1.0923         2.53
ExhaustiveBinary-256-random (self)                    11_945.08     2_350.22    14_295.29       0.1523          1.1428         2.53
ExhaustiveBinary-256-pca_no_rr (query)                12_852.39       575.03    13_427.42       0.1730             NaN         2.53
ExhaustiveBinary-256-pca-rf10 (query)                 12_852.39       747.89    13_600.27       0.4383          1.0451         2.53
ExhaustiveBinary-256-pca-rf20 (query)                 12_852.39       919.36    13_771.75       0.5559          1.0281         2.53
ExhaustiveBinary-256-pca (self)                       12_852.39     2_467.49    15_319.88       0.4397          1.0450         2.53
ExhaustiveBinary-512-random_no_rr (query)             23_414.75     1_081.96    24_496.71       0.0628             NaN         5.05
ExhaustiveBinary-512-random-rf10 (query)              23_414.75     1_237.12    24_651.87       0.1942          1.1100         5.05
ExhaustiveBinary-512-random-rf20 (query)              23_414.75     1_414.64    24_829.40       0.3026          1.0701         5.05
ExhaustiveBinary-512-random (self)                    23_414.75     4_092.85    27_507.60       0.1954          1.1092         5.05
ExhaustiveBinary-512-pca_no_rr (query)                24_487.38     1_097.96    25_585.34       0.1519             NaN         5.06
ExhaustiveBinary-512-pca-rf10 (query)                 24_487.38     1_267.13    25_754.51       0.3855          1.1037         5.06
ExhaustiveBinary-512-pca-rf20 (query)                 24_487.38     1_439.92    25_927.30       0.4977          1.0366         5.06
ExhaustiveBinary-512-pca (self)                       24_487.38     4_381.93    28_869.31       0.3861          1.1000         5.06
ExhaustiveBinary-1024-random_no_rr (query)            46_933.80     2_025.90    48_959.70       0.0893             NaN        10.10
ExhaustiveBinary-1024-random-rf10 (query)             46_933.80     2_225.52    49_159.32       0.2382          1.0892        10.10
ExhaustiveBinary-1024-random-rf20 (query)             46_933.80     2_387.40    49_321.20       0.3610          1.0558        10.10
ExhaustiveBinary-1024-random (self)                   46_933.80     7_323.69    54_257.49       0.2382          1.0889        10.10
ExhaustiveBinary-1024-pca_no_rr (query)               47_936.63     2_055.32    49_991.95       0.1240             NaN        10.11
ExhaustiveBinary-1024-pca-rf10 (query)                47_936.63     2_240.82    50_177.44       0.3310        229.0833        10.11
ExhaustiveBinary-1024-pca-rf20 (query)                47_936.63     2_426.40    50_363.02       0.4429          1.2254        10.11
ExhaustiveBinary-1024-pca (self)                      47_936.63     7_443.46    55_380.09       0.3309        235.7787        10.11
ExhaustiveBinary-1024-signed_no_rr (query)            47_425.50     2_042.05    49_467.55       0.0893             NaN        10.10
ExhaustiveBinary-1024-signed-rf10 (query)             47_425.50     2_230.29    49_655.78       0.2382          1.0892        10.10
ExhaustiveBinary-1024-signed-rf20 (query)             47_425.50     2_414.04    49_839.53       0.3610          1.0558        10.10
ExhaustiveBinary-1024-signed (self)                   47_425.50     7_395.29    54_820.78       0.2382          1.0889        10.10
IVF-Binary-256-nl158-np7-rf0-random (query)           19_470.31       534.71    20_005.02       0.0466             NaN         3.14
IVF-Binary-256-nl158-np12-rf0-random (query)          19_470.31       532.10    20_002.42       0.0401             NaN         3.14
IVF-Binary-256-nl158-np17-rf0-random (query)          19_470.31       570.09    20_040.41       0.0318             NaN         3.14
IVF-Binary-256-nl158-np7-rf10-random (query)          19_470.31       668.69    20_139.00       0.1855          1.1211         3.14
IVF-Binary-256-nl158-np7-rf20-random (query)          19_470.31       751.05    20_221.36       0.2921          1.0777         3.14
IVF-Binary-256-nl158-np12-rf10-random (query)         19_470.31       623.06    20_093.37       0.1726          1.1303         3.14
IVF-Binary-256-nl158-np12-rf20-random (query)         19_470.31       729.00    20_199.31       0.2786          1.0828         3.14
IVF-Binary-256-nl158-np17-rf10-random (query)         19_470.31       611.59    20_081.91       0.1597          1.1394         3.14
IVF-Binary-256-nl158-np17-rf20-random (query)         19_470.31       737.21    20_207.53       0.2658          1.0879         3.14
IVF-Binary-256-nl158-random (self)                    19_470.31     1_884.17    21_354.49       0.1733          1.1290         3.14
IVF-Binary-256-nl223-np11-rf0-random (query)          14_048.30       491.44    14_539.74       0.0446             NaN         3.40
IVF-Binary-256-nl223-np14-rf0-random (query)          14_048.30       493.84    14_542.13       0.0349             NaN         3.40
IVF-Binary-256-nl223-np21-rf0-random (query)          14_048.30       499.52    14_547.82       0.0313             NaN         3.40
IVF-Binary-256-nl223-np11-rf10-random (query)         14_048.30       608.31    14_656.60       0.1883          1.1195         3.40
IVF-Binary-256-nl223-np11-rf20-random (query)         14_048.30       722.84    14_771.14       0.2980          1.0751         3.40
IVF-Binary-256-nl223-np14-rf10-random (query)         14_048.30       605.71    14_654.01       0.1646          1.1365         3.40
IVF-Binary-256-nl223-np14-rf20-random (query)         14_048.30       726.17    14_774.46       0.2716          1.0860         3.40
IVF-Binary-256-nl223-np21-rf10-random (query)         14_048.30       612.43    14_660.72       0.1549          1.1434         3.40
IVF-Binary-256-nl223-np21-rf20-random (query)         14_048.30       741.93    14_790.22       0.2595          1.0910         3.40
IVF-Binary-256-nl223-random (self)                    14_048.30     1_942.22    15_990.51       0.1655          1.1353         3.40
IVF-Binary-256-nl316-np15-rf0-random (query)          14_600.72       533.87    15_134.59       0.0364             NaN         3.76
IVF-Binary-256-nl316-np17-rf0-random (query)          14_600.72       536.55    15_137.27       0.0348             NaN         3.76
IVF-Binary-256-nl316-np25-rf0-random (query)          14_600.72       537.61    15_138.34       0.0330             NaN         3.76
IVF-Binary-256-nl316-np15-rf10-random (query)         14_600.72       647.14    15_247.86       0.1705          1.1324         3.76
IVF-Binary-256-nl316-np15-rf20-random (query)         14_600.72       770.18    15_370.90       0.2805          1.0829         3.76
IVF-Binary-256-nl316-np17-rf10-random (query)         14_600.72       649.80    15_250.53       0.1660          1.1354         3.76
IVF-Binary-256-nl316-np17-rf20-random (query)         14_600.72       773.10    15_373.82       0.2747          1.0849         3.76
IVF-Binary-256-nl316-np25-rf10-random (query)         14_600.72       657.17    15_257.89       0.1607          1.1388         3.76
IVF-Binary-256-nl316-np25-rf20-random (query)         14_600.72       782.00    15_382.72       0.2667          1.0878         3.76
IVF-Binary-256-nl316-random (self)                    14_600.72     2_077.98    16_678.70       0.1671          1.1342         3.76
IVF-Binary-256-nl158-np7-rf0-pca (query)              20_124.34       477.23    20_601.57       0.2011             NaN         3.15
IVF-Binary-256-nl158-np12-rf0-pca (query)             20_124.34       483.27    20_607.61       0.1995             NaN         3.15
IVF-Binary-256-nl158-np17-rf0-pca (query)             20_124.34       489.49    20_613.83       0.1980             NaN         3.15
IVF-Binary-256-nl158-np7-rf10-pca (query)             20_124.34       612.40    20_736.74       0.5842          1.0250         3.15
IVF-Binary-256-nl158-np7-rf20-pca (query)             20_124.34       734.74    20_859.07       0.7446          1.0119         3.15
IVF-Binary-256-nl158-np12-rf10-pca (query)            20_124.34       615.61    20_739.95       0.5775          1.0257         3.15
IVF-Binary-256-nl158-np12-rf20-pca (query)            20_124.34       744.76    20_869.10       0.7361          1.0124         3.15
IVF-Binary-256-nl158-np17-rf10-pca (query)            20_124.34       626.73    20_751.07       0.5672          1.0268         3.15
IVF-Binary-256-nl158-np17-rf20-pca (query)            20_124.34       762.46    20_886.79       0.7214          1.0134         3.15
IVF-Binary-256-nl158-pca (self)                       20_124.34     1_987.82    22_112.15       0.5783          1.0257         3.15
IVF-Binary-256-nl223-np11-rf0-pca (query)             14_961.16       501.33    15_462.49       0.2009             NaN         3.40
IVF-Binary-256-nl223-np14-rf0-pca (query)             14_961.16       504.17    15_465.32       0.2000             NaN         3.40
IVF-Binary-256-nl223-np21-rf0-pca (query)             14_961.16       510.38    15_471.54       0.1986             NaN         3.40
IVF-Binary-256-nl223-np11-rf10-pca (query)            14_961.16       637.82    15_598.98       0.5840          1.0250         3.40
IVF-Binary-256-nl223-np11-rf20-pca (query)            14_961.16       772.18    15_733.33       0.7445          1.0119         3.40
IVF-Binary-256-nl223-np14-rf10-pca (query)            14_961.16       639.55    15_600.71       0.5802          1.0254         3.40
IVF-Binary-256-nl223-np14-rf20-pca (query)            14_961.16       768.12    15_729.28       0.7408          1.0121         3.40
IVF-Binary-256-nl223-np21-rf10-pca (query)            14_961.16       648.81    15_609.97       0.5710          1.0264         3.40
IVF-Binary-256-nl223-np21-rf20-pca (query)            14_961.16       786.78    15_747.94       0.7267          1.0131         3.40
IVF-Binary-256-nl223-pca (self)                       14_961.16     2_059.85    17_021.01       0.5809          1.0254         3.40
IVF-Binary-256-nl316-np15-rf0-pca (query)             15_688.76       535.25    16_224.01       0.2008             NaN         3.77
IVF-Binary-256-nl316-np17-rf0-pca (query)             15_688.76       537.28    16_226.04       0.2005             NaN         3.77
IVF-Binary-256-nl316-np25-rf0-pca (query)             15_688.76       558.43    16_247.19       0.1995             NaN         3.77
IVF-Binary-256-nl316-np15-rf10-pca (query)            15_688.76       674.00    16_362.76       0.5848          1.0249         3.77
IVF-Binary-256-nl316-np15-rf20-pca (query)            15_688.76       799.02    16_487.78       0.7463          1.0117         3.77
IVF-Binary-256-nl316-np17-rf10-pca (query)            15_688.76       672.05    16_360.82       0.5831          1.0251         3.77
IVF-Binary-256-nl316-np17-rf20-pca (query)            15_688.76       818.44    16_507.20       0.7445          1.0119         3.77
IVF-Binary-256-nl316-np25-rf10-pca (query)            15_688.76       679.86    16_368.62       0.5760          1.0258         3.77
IVF-Binary-256-nl316-np25-rf20-pca (query)            15_688.76       821.36    16_510.12       0.7341          1.0126         3.77
IVF-Binary-256-nl316-pca (self)                       15_688.76     2_176.81    17_865.57       0.5837          1.0251         3.77
IVF-Binary-512-nl158-np7-rf0-random (query)           30_878.31       875.93    31_754.23       0.0738             NaN         5.67
IVF-Binary-512-nl158-np12-rf0-random (query)          30_878.31       886.55    31_764.86       0.0699             NaN         5.67
IVF-Binary-512-nl158-np17-rf0-random (query)          30_878.31       901.96    31_780.27       0.0656             NaN         5.67
IVF-Binary-512-nl158-np7-rf10-random (query)          30_878.31       995.36    31_873.67       0.2129          1.1011         5.67
IVF-Binary-512-nl158-np7-rf20-random (query)          30_878.31     1_108.71    31_987.01       0.3277          1.0638         5.67
IVF-Binary-512-nl158-np12-rf10-random (query)         30_878.31     1_003.05    31_881.36       0.2065          1.1043         5.67
IVF-Binary-512-nl158-np12-rf20-random (query)         30_878.31     1_125.51    32_003.82       0.3187          1.0660         5.67
IVF-Binary-512-nl158-np17-rf10-random (query)         30_878.31     1_016.06    31_894.36       0.2008          1.1069         5.67
IVF-Binary-512-nl158-np17-rf20-random (query)         30_878.31     1_154.26    32_032.57       0.3121          1.0675         5.67
IVF-Binary-512-nl158-random (self)                    30_878.31     3_278.66    34_156.97       0.2075          1.1033         5.67
IVF-Binary-512-nl223-np11-rf0-random (query)          25_608.42       903.34    26_511.76       0.0738             NaN         5.92
IVF-Binary-512-nl223-np14-rf0-random (query)          25_608.42       916.17    26_524.59       0.0664             NaN         5.92
IVF-Binary-512-nl223-np21-rf0-random (query)          25_608.42       919.21    26_527.63       0.0641             NaN         5.92
IVF-Binary-512-nl223-np11-rf10-random (query)         25_608.42     1_030.61    26_639.03       0.2149          1.0991         5.92
IVF-Binary-512-nl223-np11-rf20-random (query)         25_608.42     1_145.93    26_754.35       0.3290          1.0627         5.92
IVF-Binary-512-nl223-np14-rf10-random (query)         25_608.42     1_023.26    26_631.68       0.2038          1.1055         5.92
IVF-Binary-512-nl223-np14-rf20-random (query)         25_608.42     1_149.79    26_758.21       0.3156          1.0666         5.92
IVF-Binary-512-nl223-np21-rf10-random (query)         25_608.42     1_125.35    26_733.77       0.1975          1.1089         5.92
IVF-Binary-512-nl223-np21-rf20-random (query)         25_608.42     1_158.04    26_766.46       0.3090          1.0686         5.92
IVF-Binary-512-nl223-random (self)                    25_608.42     3_348.24    28_956.66       0.2037          1.1051         5.92
IVF-Binary-512-nl316-np15-rf0-random (query)          26_339.73     1_011.43    27_351.16       0.0683             NaN         6.29
IVF-Binary-512-nl316-np17-rf0-random (query)          26_339.73       938.07    27_277.80       0.0673             NaN         6.29
IVF-Binary-512-nl316-np25-rf0-random (query)          26_339.73       946.79    27_286.52       0.0659             NaN         6.29
IVF-Binary-512-nl316-np15-rf10-random (query)         26_339.73     1_053.76    27_393.49       0.2080          1.1036         6.29
IVF-Binary-512-nl316-np15-rf20-random (query)         26_339.73     1_173.17    27_512.90       0.3214          1.0652         6.29
IVF-Binary-512-nl316-np17-rf10-random (query)         26_339.73     1_056.98    27_396.71       0.2053          1.1050         6.29
IVF-Binary-512-nl316-np17-rf20-random (query)         26_339.73     1_173.66    27_513.39       0.3171          1.0663         6.29
IVF-Binary-512-nl316-np25-rf10-random (query)         26_339.73     1_116.94    27_456.67       0.2013          1.1069         6.29
IVF-Binary-512-nl316-np25-rf20-random (query)         26_339.73     1_189.45    27_529.18       0.3119          1.0675         6.29
IVF-Binary-512-nl316-random (self)                    26_339.73     3_460.08    29_799.81       0.2056          1.1044         6.29
IVF-Binary-512-nl158-np7-rf0-pca (query)              31_929.38       901.34    32_830.72       0.2590             NaN         5.67
IVF-Binary-512-nl158-np12-rf0-pca (query)             31_929.38       906.47    32_835.85       0.2497             NaN         5.67
IVF-Binary-512-nl158-np17-rf0-pca (query)             31_929.38       916.35    32_845.73       0.2402             NaN         5.67
IVF-Binary-512-nl158-np7-rf10-pca (query)             31_929.38     1_026.61    32_955.99       0.7002          1.0145         5.67
IVF-Binary-512-nl158-np7-rf20-pca (query)             31_929.38     1_144.64    33_074.02       0.8406          1.0062         5.67
IVF-Binary-512-nl158-np12-rf10-pca (query)            31_929.38     1_047.49    32_976.87       0.6745          1.0165         5.67
IVF-Binary-512-nl158-np12-rf20-pca (query)            31_929.38     1_166.01    33_095.39       0.8168          1.0072         5.67
IVF-Binary-512-nl158-np17-rf10-pca (query)            31_929.38     1_048.23    32_977.61       0.6442          1.0191         5.67
IVF-Binary-512-nl158-np17-rf20-pca (query)            31_929.38     1_187.22    33_116.60       0.7855          1.0089         5.67
IVF-Binary-512-nl158-pca (self)                       31_929.38     3_393.51    35_322.88       0.6752          1.0165         5.67
IVF-Binary-512-nl223-np11-rf0-pca (query)             26_724.61       919.19    27_643.80       0.2574             NaN         5.93
IVF-Binary-512-nl223-np14-rf0-pca (query)             26_724.61       930.15    27_654.75       0.2535             NaN         5.93
IVF-Binary-512-nl223-np21-rf0-pca (query)             26_724.61       932.60    27_657.21       0.2437             NaN         5.93
IVF-Binary-512-nl223-np11-rf10-pca (query)            26_724.61     1_049.15    27_773.76       0.6972          1.0147         5.93
IVF-Binary-512-nl223-np11-rf20-pca (query)            26_724.61     1_171.89    27_896.50       0.8391          1.0062         5.93
IVF-Binary-512-nl223-np14-rf10-pca (query)            26_724.61     1_053.61    27_778.22       0.6866          1.0155         5.93
IVF-Binary-512-nl223-np14-rf20-pca (query)            26_724.61     1_219.69    27_944.29       0.8296          1.0066         5.93
IVF-Binary-512-nl223-np21-rf10-pca (query)            26_724.61     1_072.14    27_796.75       0.6566          1.0180         5.93
IVF-Binary-512-nl223-np21-rf20-pca (query)            26_724.61     1_215.79    27_940.40       0.7980          1.0083         5.93
IVF-Binary-512-nl223-pca (self)                       26_724.61     3_470.36    30_194.97       0.6870          1.0156         5.93
IVF-Binary-512-nl316-np15-rf0-pca (query)             27_414.51       954.57    28_369.08       0.2583             NaN         6.29
IVF-Binary-512-nl316-np17-rf0-pca (query)             27_414.51       962.48    28_376.98       0.2565             NaN         6.29
IVF-Binary-512-nl316-np25-rf0-pca (query)             27_414.51       960.25    28_374.76       0.2487             NaN         6.29
IVF-Binary-512-nl316-np15-rf10-pca (query)            27_414.51     1_095.94    28_510.44       0.7009          1.0144         6.29
IVF-Binary-512-nl316-np15-rf20-pca (query)            27_414.51     1_229.04    28_643.55       0.8436          1.0059         6.29
IVF-Binary-512-nl316-np17-rf10-pca (query)            27_414.51     1_101.20    28_515.71       0.6959          1.0148         6.29
IVF-Binary-512-nl316-np17-rf20-pca (query)            27_414.51     1_240.95    28_655.45       0.8394          1.0061         6.29
IVF-Binary-512-nl316-np25-rf10-pca (query)            27_414.51     1_114.35    28_528.85       0.6714          1.0168         6.29
IVF-Binary-512-nl316-np25-rf20-pca (query)            27_414.51     1_249.92    28_664.42       0.8138          1.0074         6.29
IVF-Binary-512-nl316-pca (self)                       27_414.51     3_627.71    31_042.22       0.6967          1.0148         6.29
IVF-Binary-1024-nl158-np7-rf0-random (query)          54_346.59     1_768.43    56_115.02       0.0929             NaN        10.72
IVF-Binary-1024-nl158-np12-rf0-random (query)         54_346.59     1_743.03    56_089.61       0.0915             NaN        10.72
IVF-Binary-1024-nl158-np17-rf0-random (query)         54_346.59     1_755.46    56_102.04       0.0906             NaN        10.72
IVF-Binary-1024-nl158-np7-rf10-random (query)         54_346.59     1_843.45    56_190.03       0.2511          1.0849        10.72
IVF-Binary-1024-nl158-np7-rf20-random (query)         54_346.59     1_988.15    56_334.74       0.3779          1.0527        10.72
IVF-Binary-1024-nl158-np12-rf10-random (query)        54_346.59     1_857.96    56_204.55       0.2460          1.0867        10.72
IVF-Binary-1024-nl158-np12-rf20-random (query)        54_346.59     1_997.44    56_344.02       0.3703          1.0541        10.72
IVF-Binary-1024-nl158-np17-rf10-random (query)        54_346.59     1_888.23    56_234.82       0.2426          1.0878        10.72
IVF-Binary-1024-nl158-np17-rf20-random (query)        54_346.59     2_011.37    56_357.96       0.3665          1.0548        10.72
IVF-Binary-1024-nl158-random (self)                   54_346.59     6_129.68    60_476.27       0.2458          1.0866        10.72
IVF-Binary-1024-nl223-np11-rf0-random (query)         49_265.89     1_754.19    51_020.08       0.0929             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-random (query)         49_265.89     1_802.41    51_068.30       0.0909             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-random (query)         49_265.89     1_775.96    51_041.85       0.0900             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-random (query)        49_265.89     2_009.36    51_275.25       0.2495          1.0856        10.98
IVF-Binary-1024-nl223-np11-rf20-random (query)        49_265.89     2_036.36    51_302.25       0.3749          1.0534        10.98
IVF-Binary-1024-nl223-np14-rf10-random (query)        49_265.89     1_883.79    51_149.68       0.2430          1.0876        10.98
IVF-Binary-1024-nl223-np14-rf20-random (query)        49_265.89     2_010.13    51_276.02       0.3670          1.0547        10.98
IVF-Binary-1024-nl223-np21-rf10-random (query)        49_265.89     1_898.72    51_164.61       0.2402          1.0885        10.98
IVF-Binary-1024-nl223-np21-rf20-random (query)        49_265.89     2_040.35    51_306.24       0.3641          1.0552        10.98
IVF-Binary-1024-nl223-random (self)                   49_265.89     6_171.69    55_437.58       0.2435          1.0873        10.98
IVF-Binary-1024-nl316-np15-rf0-random (query)         49_817.89     1_762.79    51_580.68       0.0918             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-random (query)         49_817.89     1_760.74    51_578.63       0.0913             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-random (query)         49_817.89     1_801.85    51_619.74       0.0906             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-random (query)        49_817.89     1_907.63    51_725.53       0.2465          1.0864        11.34
IVF-Binary-1024-nl316-np15-rf20-random (query)        49_817.89     1_993.44    51_811.33       0.3722          1.0538        11.34
IVF-Binary-1024-nl316-np17-rf10-random (query)        49_817.89     1_869.97    51_687.86       0.2444          1.0871        11.34
IVF-Binary-1024-nl316-np17-rf20-random (query)        49_817.89     1_996.67    51_814.56       0.3692          1.0544        11.34
IVF-Binary-1024-nl316-np25-rf10-random (query)        49_817.89     1_884.79    51_702.68       0.2419          1.0880        11.34
IVF-Binary-1024-nl316-np25-rf20-random (query)        49_817.89     2_013.96    51_831.85       0.3655          1.0550        11.34
IVF-Binary-1024-nl316-random (self)                   49_817.89     6_173.99    55_991.88       0.2448          1.0869        11.34
IVF-Binary-1024-nl158-np7-rf0-pca (query)             55_432.77     1_722.43    57_155.20       0.3012             NaN        10.73
IVF-Binary-1024-nl158-np12-rf0-pca (query)            55_432.77     1_739.30    57_172.07       0.2743             NaN        10.73
IVF-Binary-1024-nl158-np17-rf0-pca (query)            55_432.77     1_756.93    57_189.70       0.2512             NaN        10.73
IVF-Binary-1024-nl158-np7-rf10-pca (query)            55_432.77     1_848.59    57_281.36       0.7699          1.0097        10.73
IVF-Binary-1024-nl158-np7-rf20-pca (query)            55_432.77     1_972.11    57_404.88       0.8902          1.0038        10.73
IVF-Binary-1024-nl158-np12-rf10-pca (query)           55_432.77     1_898.47    57_331.24       0.7208          1.0128        10.73
IVF-Binary-1024-nl158-np12-rf20-pca (query)           55_432.77     2_002.99    57_435.75       0.8547          1.0052        10.73
IVF-Binary-1024-nl158-np17-rf10-pca (query)           55_432.77     1_912.81    57_345.58       0.6729          1.0166        10.73
IVF-Binary-1024-nl158-np17-rf20-pca (query)           55_432.77     2_042.22    57_474.99       0.8118          1.0073        10.73
IVF-Binary-1024-nl158-pca (self)                      55_432.77     6_168.85    61_601.62       0.7229          1.0127        10.73
IVF-Binary-1024-nl223-np11-rf0-pca (query)            50_204.72     1_762.57    51_967.29       0.2963             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-pca (query)            50_204.72     1_762.97    51_967.70       0.2853             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-pca (query)            50_204.72     1_774.77    51_979.50       0.2607             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-pca (query)           50_204.72     1_883.65    52_088.38       0.7635          1.0101        10.98
IVF-Binary-1024-nl223-np11-rf20-pca (query)           50_204.72     2_001.55    52_206.28       0.8870          1.0039        10.98
IVF-Binary-1024-nl223-np14-rf10-pca (query)           50_204.72     1_884.50    52_089.23       0.7442          1.0113        10.98
IVF-Binary-1024-nl223-np14-rf20-pca (query)           50_204.72     2_020.84    52_225.56       0.8738          1.0044        10.98
IVF-Binary-1024-nl223-np21-rf10-pca (query)           50_204.72     1_905.17    52_109.89       0.6935          1.0150        10.98
IVF-Binary-1024-nl223-np21-rf20-pca (query)           50_204.72     2_045.13    52_249.86       0.8303          1.0064        10.98
IVF-Binary-1024-nl223-pca (self)                      50_204.72     6_227.23    56_431.95       0.7453          1.0112        10.98
IVF-Binary-1024-nl316-np15-rf0-pca (query)            50_853.08     1_787.25    52_640.33       0.2992             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-pca (query)            50_853.08     1_807.20    52_660.28       0.2939             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-pca (query)            50_853.08     1_803.17    52_656.25       0.2724             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-pca (query)           50_853.08     1_917.09    52_770.17       0.7698          1.0097        11.34
IVF-Binary-1024-nl316-np15-rf20-pca (query)           50_853.08     2_045.89    52_898.97       0.8934          1.0036        11.34
IVF-Binary-1024-nl316-np17-rf10-pca (query)           50_853.08     1_924.02    52_777.10       0.7606          1.0102        11.34
IVF-Binary-1024-nl316-np17-rf20-pca (query)           50_853.08     2_046.47    52_899.55       0.8877          1.0038        11.34
IVF-Binary-1024-nl316-np25-rf10-pca (query)           50_853.08     1_944.20    52_797.28       0.7184          1.0131        11.34
IVF-Binary-1024-nl316-np25-rf20-pca (query)           50_853.08     2_125.95    52_979.03       0.8524          1.0053        11.34
IVF-Binary-1024-nl316-pca (self)                      50_853.08     6_318.39    57_171.46       0.7626          1.0101        11.34
IVF-Binary-1024-nl158-np7-rf0-signed (query)          54_257.11     1_692.25    55_949.36       0.0929             NaN        10.72
IVF-Binary-1024-nl158-np12-rf0-signed (query)         54_257.11     1_707.94    55_965.05       0.0915             NaN        10.72
IVF-Binary-1024-nl158-np17-rf0-signed (query)         54_257.11     1_725.37    55_982.48       0.0906             NaN        10.72
IVF-Binary-1024-nl158-np7-rf10-signed (query)         54_257.11     1_809.26    56_066.37       0.2511          1.0849        10.72
IVF-Binary-1024-nl158-np7-rf20-signed (query)         54_257.11     1_953.74    56_210.85       0.3779          1.0527        10.72
IVF-Binary-1024-nl158-np12-rf10-signed (query)        54_257.11     1_823.64    56_080.75       0.2460          1.0867        10.72
IVF-Binary-1024-nl158-np12-rf20-signed (query)        54_257.11     1_959.97    56_217.08       0.3703          1.0541        10.72
IVF-Binary-1024-nl158-np17-rf10-signed (query)        54_257.11     1_851.06    56_108.17       0.2426          1.0878        10.72
IVF-Binary-1024-nl158-np17-rf20-signed (query)        54_257.11     1_974.88    56_231.99       0.3665          1.0548        10.72
IVF-Binary-1024-nl158-signed (self)                   54_257.11     6_481.78    60_738.89       0.2458          1.0866        10.72
IVF-Binary-1024-nl223-np11-rf0-signed (query)         49_268.16     1_754.56    51_022.72       0.0929             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-signed (query)         49_268.16     1_770.52    51_038.68       0.0909             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-signed (query)         49_268.16     1_780.21    51_048.38       0.0900             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-signed (query)        49_268.16     1_873.73    51_141.89       0.2495          1.0856        10.98
IVF-Binary-1024-nl223-np11-rf20-signed (query)        49_268.16     1_995.63    51_263.80       0.3749          1.0534        10.98
IVF-Binary-1024-nl223-np14-rf10-signed (query)        49_268.16     1_881.33    51_149.50       0.2430          1.0876        10.98
IVF-Binary-1024-nl223-np14-rf20-signed (query)        49_268.16     2_021.97    51_290.14       0.3670          1.0547        10.98
IVF-Binary-1024-nl223-np21-rf10-signed (query)        49_268.16     1_897.67    51_165.83       0.2402          1.0885        10.98
IVF-Binary-1024-nl223-np21-rf20-signed (query)        49_268.16     2_034.40    51_302.56       0.3641          1.0552        10.98
IVF-Binary-1024-nl223-signed (self)                   49_268.16     6_201.01    55_469.17       0.2435          1.0873        10.98
IVF-Binary-1024-nl316-np15-rf0-signed (query)         49_834.00     1_807.59    51_641.59       0.0918             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-signed (query)         49_834.00     1_802.38    51_636.38       0.0913             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-signed (query)         49_834.00     1_834.15    51_668.16       0.0906             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-signed (query)        49_834.00     1_917.22    51_751.23       0.2465          1.0864        11.34
IVF-Binary-1024-nl316-np15-rf20-signed (query)        49_834.00     2_050.25    51_884.26       0.3722          1.0538        11.34
IVF-Binary-1024-nl316-np17-rf10-signed (query)        49_834.00     1_916.70    51_750.71       0.2444          1.0871        11.34
IVF-Binary-1024-nl316-np17-rf20-signed (query)        49_834.00     2_060.62    51_894.62       0.3692          1.0544        11.34
IVF-Binary-1024-nl316-np25-rf10-signed (query)        49_834.00     1_935.46    51_769.47       0.2419          1.0880        11.34
IVF-Binary-1024-nl316-np25-rf20-signed (query)        49_834.00     2_072.62    51_906.62       0.3655          1.0550        11.34
IVF-Binary-1024-nl316-signed (self)                   49_834.00     6_325.84    56_159.84       0.2448          1.0869        11.34
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
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         6.64     4_230.88     4_237.52       1.0000          1.0000        48.83
Exhaustive (self)                                          6.64    14_082.46    14_089.10       1.0000          1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_578.31       248.79     2_827.10       0.0387             NaN         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_578.31       354.45     2_932.75       0.2254          1.0175         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_578.31       456.27     3_034.58       0.3590          1.0099         1.78
ExhaustiveBinary-256-random (self)                     2_578.31     1_175.22     3_753.53       0.5886         13.9170         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_802.33       252.39     3_054.72       0.0284             NaN         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_802.33       358.39     3_160.72       0.1316          1.0445         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_802.33       460.16     3_262.49       0.1964          1.0197         1.78
ExhaustiveBinary-256-pca (self)                        2_802.33     1_181.99     3_984.32       0.2462          1.6481         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_088.78       444.03     5_532.81       0.0605             NaN         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_088.78       551.57     5_640.35       0.3044          1.0140         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_088.78       656.36     5_745.14       0.4560          1.0080         3.55
ExhaustiveBinary-512-random (self)                     5_088.78     1_853.04     6_941.82       0.6749         20.2700         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_408.25       453.59     5_861.84       0.0806             NaN         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_408.25       560.51     5_968.76       0.3799          1.0083         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_408.25       665.36     6_073.60       0.5477          1.0039         3.55
ExhaustiveBinary-512-pca (self)                        5_408.25     1_867.92     7_276.17       0.6562          1.0566         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_115.51       763.59    10_879.09       0.0996             NaN         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_115.51       867.90    10_983.41       0.4213          1.0096         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_115.51       989.38    11_104.88       0.5847          1.0054         7.10
ExhaustiveBinary-1024-random (self)                   10_115.51     2_884.24    12_999.75       0.7571         24.6771         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_550.27       775.34    11_325.61       0.1211             NaN         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_550.27       884.76    11_435.03       0.5030          1.0092         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_550.27     1_013.90    11_564.17       0.6838          1.0042         7.10
ExhaustiveBinary-1024-pca (self)                      10_550.27     2_943.23    13_493.50       0.8235          1.0221         7.10
ExhaustiveBinary-256-signed_no_rr (query)              2_582.05       247.87     2_829.92       0.0387             NaN         1.78
ExhaustiveBinary-256-signed-rf10 (query)               2_582.05       354.47     2_936.52       0.2254          1.0175         1.78
ExhaustiveBinary-256-signed-rf20 (query)               2_582.05       462.86     3_044.90       0.3590          1.0099         1.78
ExhaustiveBinary-256-signed (self)                     2_582.05     1_170.49     3_752.54       0.5886         13.9170         1.78
IVF-Binary-256-nl158-np7-rf0-random (query)            3_767.96       129.25     3_897.22       0.0509             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           3_767.96       131.73     3_899.69       0.0465             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           3_767.96       144.51     3_912.47       0.0446             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           3_767.96       182.75     3_950.71       0.3052          1.0091         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           3_767.96       232.90     4_000.87       0.4816          1.0047         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          3_767.96       201.13     3_969.09       0.2841          1.0105         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          3_767.96       260.55     4_028.51       0.4542          1.0055         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          3_767.96       231.66     3_999.63       0.2744          1.0112         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          3_767.96       289.80     4_057.76       0.4401          1.0059         1.93
IVF-Binary-256-nl158-random (self)                     3_767.96       613.41     4_381.38       0.7484          1.0402         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           2_646.23       118.45     2_764.68       0.0512             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           2_646.23       121.05     2_767.28       0.0481             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           2_646.23       126.85     2_773.08       0.0452             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          2_646.23       179.10     2_825.33       0.3029          1.0093         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          2_646.23       228.87     2_875.10       0.4808          1.0050         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          2_646.23       184.26     2_830.49       0.2901          1.0100         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          2_646.23       238.22     2_884.45       0.4641          1.0054         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          2_646.23       194.02     2_840.25       0.2785          1.0106         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          2_646.23       252.51     2_898.74       0.4478          1.0057         2.00
IVF-Binary-256-nl223-random (self)                     2_646.23       543.46     3_189.69       0.7426          1.0418         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           2_696.92       126.27     2_823.19       0.0506             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           2_696.92       126.72     2_823.64       0.0491             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           2_696.92       130.76     2_827.69       0.0461             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          2_696.92       187.63     2_884.55       0.3030          1.0092         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          2_696.92       236.61     2_933.53       0.4824          1.0050         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          2_696.92       187.40     2_884.32       0.2971          1.0095         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          2_696.92       240.19     2_937.11       0.4742          1.0052         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          2_696.92       194.70     2_891.62       0.2832          1.0103         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          2_696.92       255.76     2_952.68       0.4549          1.0056         2.09
IVF-Binary-256-nl316-random (self)                     2_696.92       554.05     3_250.97       0.7435          1.0416         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               3_990.38       123.00     4_113.39       0.0612             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              3_990.38       134.63     4_125.01       0.0515             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              3_990.38       146.16     4_136.54       0.0461             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              3_990.38       191.70     4_182.08       0.3050          1.0087         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              3_990.38       253.50     4_243.88       0.4482          1.0049         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             3_990.38       207.85     4_198.23       0.2538          1.0115         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             3_990.38       287.27     4_277.65       0.3768          1.0067         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             3_990.38       226.98     4_217.36       0.2255          1.0140         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             3_990.38       306.07     4_296.45       0.3355          1.0080         1.93
IVF-Binary-256-nl158-pca (self)                        3_990.38       684.52     4_674.91       0.3531          1.2483         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              2_864.19       122.10     2_986.29       0.0657             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              2_864.19       123.90     2_988.09       0.0603             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              2_864.19       130.85     2_995.04       0.0540             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             2_864.19       188.71     3_052.90       0.3321          1.0076         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             2_864.19       245.84     3_110.04       0.4963          1.0041         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             2_864.19       191.78     3_055.97       0.3066          1.0085         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             2_864.19       251.99     3_116.18       0.4589          1.0048         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             2_864.19       202.55     3_066.75       0.2715          1.0100         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             2_864.19       268.65     3_132.84       0.4064          1.0058         2.00
IVF-Binary-256-nl223-pca (self)                        2_864.19       597.10     3_461.30       0.3965          1.1690         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              2_904.23       127.85     3_032.09       0.0667             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              2_904.23       129.44     3_033.67       0.0641             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              2_904.23       133.88     3_038.11       0.0571             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             2_904.23       194.76     3_098.99       0.3384          1.0074         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             2_904.23       249.92     3_154.15       0.5061          1.0040         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             2_904.23       198.67     3_102.90       0.3251          1.0078         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             2_904.23       258.18     3_162.42       0.4865          1.0043         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             2_904.23       206.69     3_110.92       0.2885          1.0092         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             2_904.23       269.30     3_173.53       0.4321          1.0053         2.09
IVF-Binary-256-nl316-pca (self)                        2_904.23       610.07     3_514.30       0.4100          1.1566         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_233.62       223.60     6_457.23       0.0736             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_233.62       237.63     6_471.26       0.0692             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_233.62       257.20     6_490.82       0.0671             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_233.62       281.53     6_515.15       0.3873          1.0068         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_233.62       331.26     6_564.88       0.5762          1.0034         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_233.62       307.69     6_541.31       0.3697          1.0076         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_233.62       367.68     6_601.30       0.5551          1.0038         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_233.62       342.63     6_576.25       0.3601          1.0080         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_233.62       404.80     6_638.42       0.5426          1.0041         3.71
IVF-Binary-512-nl158-random (self)                     6_233.62       972.45     7_206.07       0.8442          1.0200         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_155.90       211.16     5_367.07       0.0735             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_155.90       215.96     5_371.87       0.0705             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_155.90       225.99     5_381.89       0.0678             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_155.90       273.07     5_428.97       0.3851          1.0070         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_155.90       323.30     5_479.21       0.5741          1.0036         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_155.90       280.89     5_436.79       0.3734          1.0075         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_155.90       333.40     5_489.30       0.5598          1.0039         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_155.90       294.22     5_450.13       0.3645          1.0078         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_155.90       353.66     5_509.56       0.5480          1.0041         3.77
IVF-Binary-512-nl223-random (self)                     5_155.90       872.85     6_028.75       0.8385          1.0211         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           5_188.76       217.00     5_405.76       0.0731             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           5_188.76       221.75     5_410.51       0.0717             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           5_188.76       227.68     5_416.44       0.0685             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          5_188.76       281.55     5_470.31       0.3853          1.0070         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          5_188.76       331.77     5_520.53       0.5760          1.0036         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          5_188.76       286.74     5_475.51       0.3798          1.0073         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          5_188.76       336.74     5_525.51       0.5692          1.0037         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          5_188.76       293.49     5_482.25       0.3680          1.0077         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          5_188.76       351.63     5_540.40       0.5534          1.0040         3.86
IVF-Binary-512-nl316-random (self)                     5_188.76       871.25     6_060.01       0.8398          1.0208         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               6_601.26       222.93     6_824.19       0.0969             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              6_601.26       242.14     6_843.40       0.0918             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              6_601.26       264.23     6_865.49       0.0894             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              6_601.26       290.91     6_892.18       0.4469          1.0049         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              6_601.26       343.05     6_944.31       0.6308          1.0024         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             6_601.26       315.26     6_916.52       0.4262          1.0055         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             6_601.26       378.28     6_979.55       0.6079          1.0028         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             6_601.26       345.20     6_946.46       0.4173          1.0058         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             6_601.26       414.11     7_015.38       0.5961          1.0030         3.71
IVF-Binary-512-nl158-pca (self)                        6_601.26       997.06     7_598.32       0.6714          1.0520         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              5_503.45       218.28     5_721.73       0.0964             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              5_503.45       224.96     5_728.41       0.0937             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              5_503.45       233.63     5_737.08       0.0910             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             5_503.45       286.87     5_790.32       0.4490          1.0051         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             5_503.45       338.39     5_841.84       0.6410          1.0024         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             5_503.45       291.55     5_795.00       0.4379          1.0054         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             5_503.45       346.63     5_850.08       0.6260          1.0026         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             5_503.45       306.21     5_809.66       0.4265          1.0056         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             5_503.45       366.94     5_870.39       0.6109          1.0028         3.77
IVF-Binary-512-nl223-pca (self)                        5_503.45       906.97     6_410.42       0.6652          1.0540         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              5_532.34       226.17     5_758.51       0.0968             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              5_532.34       246.17     5_778.51       0.0954             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              5_532.34       235.59     5_767.93       0.0922             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             5_532.34       290.63     5_822.97       0.4524          1.0050         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             5_532.34       344.07     5_876.42       0.6451          1.0024         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             5_532.34       292.82     5_825.16       0.4465          1.0052         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             5_532.34       348.62     5_880.97       0.6375          1.0025         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             5_532.34       302.98     5_835.32       0.4325          1.0055         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             5_532.34       363.51     5_895.85       0.6188          1.0027         3.86
IVF-Binary-512-nl316-pca (self)                        5_532.34       909.60     6_441.94       0.6661          1.0537         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_296.97       403.80    11_700.77       0.1145             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_296.97       438.06    11_735.03       0.1107             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_296.97       463.57    11_760.54       0.1084             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_296.97       473.79    11_770.76       0.5052          1.0045         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_296.97       525.80    11_822.77       0.6944          1.0021         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_296.97       512.11    11_809.08       0.4910          1.0049         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_296.97       573.84    11_870.81       0.6794          1.0023         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_296.97       552.40    11_849.37       0.4821          1.0052         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_296.97       620.64    11_917.61       0.6691          1.0025         7.26
IVF-Binary-1024-nl158-random (self)                   11_296.97     1_650.36    12_947.33       0.9241          1.0076         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_082.48       394.98    10_477.47       0.1144             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_082.48       401.77    10_484.25       0.1111             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_082.48       418.99    10_501.48       0.1086             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_082.48       459.66    10_542.15       0.5009          1.0048         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_082.48       551.98    10_634.46       0.6893          1.0023         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_082.48       470.28    10_552.76       0.4913          1.0051         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_082.48       525.03    10_607.51       0.6783          1.0025         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_082.48       491.02    10_573.51       0.4848          1.0052         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_082.48       550.50    10_632.98       0.6715          1.0025         7.32
IVF-Binary-1024-nl223-random (self)                   10_082.48     1_497.95    11_580.43       0.9201          1.0082         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         10_224.86       400.72    10_625.58       0.1143             NaN         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)         10_224.86       409.18    10_634.04       0.1127             NaN         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)         10_224.86       416.32    10_641.18       0.1094             NaN         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)        10_224.86       467.45    10_692.31       0.5017          1.0048         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)        10_224.86       519.41    10_744.27       0.6914          1.0023         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)        10_224.86       469.48    10_694.34       0.4967          1.0049         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)        10_224.86       525.32    10_750.18       0.6857          1.0024         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)        10_224.86       484.97    10_709.83       0.4874          1.0052         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)        10_224.86       546.96    10_771.82       0.6748          1.0025         7.41
IVF-Binary-1024-nl316-random (self)                   10_224.86     1_499.98    11_724.84       0.9211          1.0080         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)             11_751.66       425.49    12_177.14       0.1358             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            11_751.66       456.82    12_208.48       0.1313             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            11_751.66       485.15    12_236.81       0.1292             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            11_751.66       498.76    12_250.42       0.5564          1.0034         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            11_751.66       551.73    12_303.39       0.7414          1.0014         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           11_751.66       535.89    12_287.55       0.5426          1.0037         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           11_751.66       602.04    12_353.69       0.7291          1.0016         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           11_751.66       579.83    12_331.49       0.5360          1.0039         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           11_751.66       649.58    12_401.23       0.7218          1.0017         7.26
IVF-Binary-1024-nl158-pca (self)                      11_751.66     1_728.26    13_479.92       0.8299          1.0204         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            10_631.60       408.06    11_039.66       0.1344             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            10_631.60       416.18    11_047.78       0.1323             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            10_631.60       433.52    11_065.12       0.1304             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           10_631.60       483.38    11_114.98       0.5541          1.0035         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           10_631.60       528.78    11_160.38       0.7434          1.0015         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           10_631.60       484.68    11_116.28       0.5473          1.0037         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           10_631.60       549.77    11_181.37       0.7354          1.0016         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           10_631.60       505.33    11_136.93       0.5408          1.0038         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           10_631.60       567.40    11_199.00       0.7276          1.0017         7.32
IVF-Binary-1024-nl223-pca (self)                      10_631.60     1_547.78    12_179.38       0.8252          1.0214         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            10_667.86       414.46    11_082.31       0.1346             NaN         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            10_667.86       418.03    11_085.89       0.1335             NaN         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            10_667.86       431.54    11_099.40       0.1312             NaN         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           10_667.86       487.42    11_155.28       0.5554          1.0035         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           10_667.86       538.17    11_206.03       0.7452          1.0015         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           10_667.86       485.57    11_153.43       0.5522          1.0036         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           10_667.86       543.00    11_210.86       0.7413          1.0015         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           10_667.86       501.31    11_169.17       0.5438          1.0037         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           10_667.86       562.07    11_229.93       0.7316          1.0016         7.42
IVF-Binary-1024-nl316-pca (self)                      10_667.86     1_549.96    12_217.82       0.8253          1.0214         7.42
IVF-Binary-256-nl158-np7-rf0-signed (query)            3_749.80       121.00     3_870.80       0.0509             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-signed (query)           3_749.80       131.03     3_880.83       0.0465             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-signed (query)           3_749.80       143.77     3_893.57       0.0446             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-signed (query)           3_749.80       181.89     3_931.69       0.3052          1.0091         1.93
IVF-Binary-256-nl158-np7-rf20-signed (query)           3_749.80       233.05     3_982.86       0.4816          1.0047         1.93
IVF-Binary-256-nl158-np12-rf10-signed (query)          3_749.80       200.98     3_950.78       0.2841          1.0105         1.93
IVF-Binary-256-nl158-np12-rf20-signed (query)          3_749.80       261.02     4_010.82       0.4542          1.0055         1.93
IVF-Binary-256-nl158-np17-rf10-signed (query)          3_749.80       222.38     3_972.18       0.2744          1.0112         1.93
IVF-Binary-256-nl158-np17-rf20-signed (query)          3_749.80       290.02     4_039.82       0.4401          1.0059         1.93
IVF-Binary-256-nl158-signed (self)                     3_749.80       614.47     4_364.27       0.7484          1.0402         1.93
IVF-Binary-256-nl223-np11-rf0-signed (query)           2_646.87       118.06     2_764.93       0.0512             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-signed (query)           2_646.87       123.43     2_770.30       0.0481             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-signed (query)           2_646.87       127.31     2_774.18       0.0452             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-signed (query)          2_646.87       179.79     2_826.66       0.3029          1.0093         2.00
IVF-Binary-256-nl223-np11-rf20-signed (query)          2_646.87       229.34     2_876.21       0.4808          1.0050         2.00
IVF-Binary-256-nl223-np14-rf10-signed (query)          2_646.87       182.82     2_829.69       0.2901          1.0100         2.00
IVF-Binary-256-nl223-np14-rf20-signed (query)          2_646.87       237.62     2_884.48       0.4641          1.0054         2.00
IVF-Binary-256-nl223-np21-rf10-signed (query)          2_646.87       196.72     2_843.59       0.2785          1.0106         2.00
IVF-Binary-256-nl223-np21-rf20-signed (query)          2_646.87       262.26     2_909.13       0.4478          1.0057         2.00
IVF-Binary-256-nl223-signed (self)                     2_646.87       543.45     3_190.32       0.7426          1.0418         2.00
IVF-Binary-256-nl316-np15-rf0-signed (query)           2_685.22       124.53     2_809.76       0.0506             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-signed (query)           2_685.22       125.50     2_810.72       0.0491             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-signed (query)           2_685.22       131.27     2_816.50       0.0461             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-signed (query)          2_685.22       185.71     2_870.93       0.3030          1.0092         2.09
IVF-Binary-256-nl316-np15-rf20-signed (query)          2_685.22       236.14     2_921.36       0.4824          1.0050         2.09
IVF-Binary-256-nl316-np17-rf10-signed (query)          2_685.22       187.05     2_872.27       0.2971          1.0095         2.09
IVF-Binary-256-nl316-np17-rf20-signed (query)          2_685.22       241.20     2_926.42       0.4742          1.0052         2.09
IVF-Binary-256-nl316-np25-rf10-signed (query)          2_685.22       194.48     2_879.71       0.2832          1.0103         2.09
IVF-Binary-256-nl316-np25-rf20-signed (query)          2_685.22       254.89     2_940.11       0.4549          1.0056         2.09
IVF-Binary-256-nl316-signed (self)                     2_685.22       554.38     3_239.61       0.7435          1.0416         2.09
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
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        18.88     9_645.79     9_664.67       1.0000          1.0000        97.66
Exhaustive (self)                                         18.88    32_317.73    32_336.61       1.0000          1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_662.40       353.76     6_016.17       0.0197             NaN         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_662.40       484.01     6_146.41       0.1454          1.0108         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_662.40       616.57     6_278.98       0.2497          1.0068         2.03
ExhaustiveBinary-256-random (self)                     5_662.40     1_613.93     7_276.34       0.6222          7.2120         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_171.57       360.24     6_531.81       0.0177             NaN         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_171.57       490.66     6_662.23       0.0885          1.0249         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_171.57       621.94     6_793.51       0.1361          1.0117         2.03
ExhaustiveBinary-256-pca (self)                        6_171.57     1_622.14     7_793.71       0.2209          1.7046         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_078.08       654.47    11_732.55       0.0425             NaN         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_078.08       785.74    11_863.82       0.2372          1.0067         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_078.08       916.19    11_994.27       0.3726          1.0039         4.05
ExhaustiveBinary-512-random (self)                    11_078.08     2_597.90    13_675.98       0.7112          9.2589         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_887.80       668.24    12_556.04       0.0149             NaN         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_887.80       797.49    12_685.29       0.0785          1.0474         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_887.80       927.36    12_815.16       0.1227          1.0180         4.05
ExhaustiveBinary-512-pca (self)                       11_887.80     2_654.40    14_542.20       0.1561          3.9887         4.05
ExhaustiveBinary-1024-random_no_rr (query)            21_950.93     1_171.37    23_122.31       0.0662             NaN         8.10
ExhaustiveBinary-1024-random-rf10 (query)             21_950.93     1_317.21    23_268.14       0.3226          1.0049         8.10
ExhaustiveBinary-1024-random-rf20 (query)             21_950.93     1_469.08    23_420.01       0.4753          1.0028         8.10
ExhaustiveBinary-1024-random (self)                   21_950.93     4_369.45    26_320.38       0.7729         10.8323         8.10
ExhaustiveBinary-1024-pca_no_rr (query)               23_007.79     1_210.69    24_218.48       0.0749             NaN         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                23_007.79     1_348.19    24_355.98       0.3494          1.0037         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                23_007.79     1_497.29    24_505.08       0.5064          1.0020         8.11
ExhaustiveBinary-1024-pca (self)                      23_007.79     4_477.09    27_484.88       0.6913          1.0488         8.11
ExhaustiveBinary-512-signed_no_rr (query)             11_055.53       655.68    11_711.22       0.0425             NaN         4.05
ExhaustiveBinary-512-signed-rf10 (query)              11_055.53       784.64    11_840.17       0.2372          1.0067         4.05
ExhaustiveBinary-512-signed-rf20 (query)              11_055.53       930.78    11_986.31       0.3726          1.0039         4.05
ExhaustiveBinary-512-signed (self)                    11_055.53     2_593.77    13_649.31       0.7112          9.2589         4.05
IVF-Binary-256-nl158-np7-rf0-random (query)            8_088.00       232.50     8_320.50       0.0301             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           8_088.00       241.50     8_329.50       0.0255             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           8_088.00       262.46     8_350.46       0.0237             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           8_088.00       321.08     8_409.08       0.2211          1.0057         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           8_088.00       401.01     8_489.01       0.3773          1.0030         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          8_088.00       331.23     8_419.23       0.1951          1.0066         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          8_088.00       419.97     8_507.97       0.3354          1.0037         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          8_088.00       345.00     8_433.01       0.1845          1.0072         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          8_088.00       437.96     8_525.96       0.3185          1.0041         2.34
IVF-Binary-256-nl158-random (self)                     8_088.00     1_006.84     9_094.85       0.7814          1.0320         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           5_738.99       237.16     5_976.15       0.0303             NaN         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)           5_738.99       242.58     5_981.58       0.0277             NaN         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)           5_738.99       246.21     5_985.21       0.0252             NaN         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)          5_738.99       327.28     6_066.28       0.2240          1.0055         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)          5_738.99       408.17     6_147.16       0.3838          1.0030         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)          5_738.99       329.29     6_068.29       0.2095          1.0060         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)          5_738.99       416.69     6_155.69       0.3612          1.0033         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)          5_738.99       339.54     6_078.53       0.1942          1.0066         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)          5_738.99       429.90     6_168.90       0.3374          1.0037         2.46
IVF-Binary-256-nl223-random (self)                     5_738.99       993.20     6_732.20       0.7767          1.0338         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)           5_730.48       251.01     5_981.49       0.0302             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           5_730.48       255.86     5_986.35       0.0287             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           5_730.48       276.34     6_006.83       0.0259             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          5_730.48       343.12     6_073.61       0.2253          1.0055         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          5_730.48       423.17     6_153.65       0.3874          1.0030         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          5_730.48       340.87     6_071.36       0.2169          1.0057         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          5_730.48       425.80     6_156.28       0.3744          1.0031         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          5_730.48       353.02     6_083.50       0.2002          1.0063         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          5_730.48       446.16     6_176.64       0.3475          1.0035         2.65
IVF-Binary-256-nl316-random (self)                     5_730.48     1_031.12     6_761.61       0.7765          1.0339         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               8_663.05       236.71     8_899.76       0.0474             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              8_663.05       245.85     8_908.90       0.0383             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              8_663.05       267.56     8_930.61       0.0338             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              8_663.05       341.21     9_004.26       0.2568          1.0041         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              8_663.05       418.70     9_081.75       0.3997          1.0024         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             8_663.05       342.23     9_005.28       0.2058          1.0053         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             8_663.05       437.41     9_100.46       0.3228          1.0032         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             8_663.05       354.55     9_017.60       0.1801          1.0062         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             8_663.05       454.79     9_117.84       0.2819          1.0038         2.34
IVF-Binary-256-nl158-pca (self)                        8_663.05     1_086.20     9_749.25       0.3375          1.2161         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              6_282.41       243.66     6_526.07       0.0492             NaN         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              6_282.41       246.34     6_528.76       0.0450             NaN         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              6_282.41       252.80     6_535.21       0.0395             NaN         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             6_282.41       340.47     6_622.88       0.2682          1.0040         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             6_282.41       421.68     6_704.09       0.4193          1.0023         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             6_282.41       339.58     6_622.00       0.2454          1.0044         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             6_282.41       430.60     6_713.02       0.3842          1.0026         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             6_282.41       349.03     6_631.45       0.2134          1.0051         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             6_282.41       442.37     6_724.78       0.3346          1.0031         2.47
IVF-Binary-256-nl223-pca (self)                        6_282.41     1_070.17     7_352.58       0.3700          1.1736         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              6_304.51       259.05     6_563.55       0.0499             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              6_304.51       258.68     6_563.19       0.0477             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              6_304.51       262.91     6_567.42       0.0420             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             6_304.51       352.39     6_656.90       0.2726          1.0040         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             6_304.51       443.66     6_748.17       0.4269          1.0023         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             6_304.51       356.70     6_661.21       0.2605          1.0042         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             6_304.51       441.05     6_745.56       0.4081          1.0024         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             6_304.51       362.93     6_667.44       0.2280          1.0048         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             6_304.51       453.57     6_758.08       0.3570          1.0029         2.65
IVF-Binary-256-nl316-pca (self)                        6_304.51     1_123.37     7_427.88       0.3813          1.1642         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           13_652.61       427.13    14_079.74       0.0550             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          13_652.61       441.59    14_094.20       0.0503             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          13_652.61       457.02    14_109.63       0.0485             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          13_652.61       517.82    14_170.42       0.3197          1.0037         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          13_652.61       600.54    14_253.15       0.5000          1.0019         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         13_652.61       534.25    14_186.86       0.2982          1.0041         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         13_652.61       620.17    14_272.78       0.4687          1.0022         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         13_652.61       557.62    14_210.23       0.2893          1.0043         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         13_652.61       647.84    14_300.45       0.4553          1.0024         4.36
IVF-Binary-512-nl158-random (self)                    13_652.61     1_693.95    15_346.56       0.8734          1.0152         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          11_407.78       433.34    11_841.12       0.0551             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          11_407.78       439.91    11_847.69       0.0524             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          11_407.78       446.72    11_854.50       0.0500             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         11_407.78       523.33    11_931.10       0.3198          1.0039         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         11_407.78       600.99    12_008.77       0.4995          1.0021         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         11_407.78       527.61    11_935.39       0.3078          1.0041         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         11_407.78       610.96    12_018.74       0.4831          1.0022         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         11_407.78       541.64    11_949.42       0.2965          1.0042         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         11_407.78       629.02    12_036.80       0.4669          1.0023         4.49
IVF-Binary-512-nl223-random (self)                    11_407.78     1_656.61    13_064.38       0.8701          1.0161         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          11_207.65       447.86    11_655.52       0.0552             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          11_207.65       449.46    11_657.12       0.0536             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          11_207.65       458.23    11_665.89       0.0508             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         11_207.65       539.01    11_746.66       0.3216          1.0039         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         11_207.65       619.87    11_827.52       0.5021          1.0021         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         11_207.65       538.28    11_745.94       0.3146          1.0040         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         11_207.65       621.51    11_829.16       0.4930          1.0021         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         11_207.65       549.00    11_756.65       0.3015          1.0041         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         11_207.65       642.61    11_850.26       0.4742          1.0023         4.67
IVF-Binary-512-nl316-random (self)                    11_207.65     1_704.97    12_912.62       0.8706          1.0160         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              14_253.97       440.02    14_693.99       0.0554             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             14_253.97       453.53    14_707.50       0.0429             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             14_253.97       467.26    14_721.23       0.0366             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             14_253.97       536.23    14_790.20       0.2851          1.0038         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             14_253.97       620.99    14_874.95       0.4336          1.0022         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            14_253.97       557.97    14_811.93       0.2231          1.0052         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            14_253.97       647.14    14_901.11       0.3413          1.0031         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            14_253.97       571.58    14_825.55       0.1894          1.0066         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            14_253.97       672.49    14_926.46       0.2911          1.0038         4.36
IVF-Binary-512-nl158-pca (self)                       14_253.97     1_796.15    16_050.12       0.2813          1.2960         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             11_906.23       446.10    12_352.33       0.0585             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             11_906.23       449.68    12_355.91       0.0528             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             11_906.23       461.89    12_368.12       0.0450             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            11_906.23       541.05    12_447.28       0.3017          1.0037         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            11_906.23       623.94    12_530.16       0.4606          1.0021         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            11_906.23       543.30    12_449.52       0.2734          1.0041         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            11_906.23       638.90    12_545.13       0.4189          1.0024         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            11_906.23       556.70    12_462.93       0.2319          1.0050         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            11_906.23       650.61    12_556.84       0.3575          1.0030         4.49
IVF-Binary-512-nl223-pca (self)                       11_906.23     1_760.95    13_667.17       0.3205          1.2194         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             11_896.84       459.83    12_356.68       0.0596             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             11_896.84       461.78    12_358.62       0.0564             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             11_896.84       469.28    12_366.12       0.0482             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            11_896.84       555.90    12_452.74       0.3073          1.0036         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            11_896.84       638.15    12_534.99       0.4686          1.0020         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            11_896.84       555.24    12_452.09       0.2925          1.0039         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            11_896.84       641.54    12_538.38       0.4463          1.0022         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            11_896.84       566.69    12_463.53       0.2500          1.0046         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            11_896.84       658.62    12_555.46       0.3851          1.0027         4.67
IVF-Binary-512-nl316-pca (self)                       11_896.84     1_800.38    13_697.22       0.3346          1.2038         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          24_437.33       820.30    25_257.63       0.0801             NaN         8.41
IVF-Binary-1024-nl158-np12-rf0-random (query)         24_437.33       844.18    25_281.51       0.0753             NaN         8.41
IVF-Binary-1024-nl158-np17-rf0-random (query)         24_437.33       864.08    25_301.41       0.0734             NaN         8.41
IVF-Binary-1024-nl158-np7-rf10-random (query)         24_437.33       911.37    25_348.70       0.4066          1.0027         8.41
IVF-Binary-1024-nl158-np7-rf20-random (query)         24_437.33     1_003.47    25_440.80       0.5989          1.0013         8.41
IVF-Binary-1024-nl158-np12-rf10-random (query)        24_437.33       939.78    25_377.11       0.3864          1.0030         8.41
IVF-Binary-1024-nl158-np12-rf20-random (query)        24_437.33     1_030.60    25_467.93       0.5728          1.0015         8.41
IVF-Binary-1024-nl158-np17-rf10-random (query)        24_437.33       964.91    25_402.24       0.3781          1.0032         8.41
IVF-Binary-1024-nl158-np17-rf20-random (query)        24_437.33     1_059.87    25_497.20       0.5612          1.0017         8.41
IVF-Binary-1024-nl158-random (self)                   24_437.33     3_035.41    27_472.74       0.9356          1.0062         8.41
IVF-Binary-1024-nl223-np11-rf0-random (query)         22_189.71       825.79    23_015.49       0.0801             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         22_189.71       830.73    23_020.43       0.0771             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         22_189.71       855.98    23_045.69       0.0744             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        22_189.71       915.32    23_105.03       0.4065          1.0029         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        22_189.71       992.80    23_182.50       0.5973          1.0015         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        22_189.71       924.04    23_113.75       0.3955          1.0030         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        22_189.71     1_006.12    23_195.82       0.5846          1.0016         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        22_189.71       943.66    23_133.36       0.3850          1.0031         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        22_189.71     1_031.23    23_220.93       0.5716          1.0016         8.54
IVF-Binary-1024-nl223-random (self)                   22_189.71     2_981.65    25_171.36       0.9333          1.0066         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         22_128.36       840.30    22_968.66       0.0802             NaN         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)         22_128.36       849.50    22_977.86       0.0783             NaN         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)         22_128.36       853.50    22_981.86       0.0752             NaN         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)        22_128.36       930.07    23_058.42       0.4081          1.0029         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)        22_128.36     1_008.96    23_137.31       0.5998          1.0015         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)        22_128.36       931.50    23_059.85       0.4016          1.0030         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)        22_128.36     1_018.69    23_147.05       0.5922          1.0015         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)        22_128.36       948.58    23_076.94       0.3899          1.0031         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)        22_128.36     1_045.64    23_174.00       0.5780          1.0016         8.72
IVF-Binary-1024-nl316-random (self)                   22_128.36     3_004.22    25_132.58       0.9338          1.0065         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)             25_507.03       865.71    26_372.74       0.0964             NaN         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            25_507.03       867.42    26_374.45       0.0896             NaN         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            25_507.03       890.46    26_397.50       0.0865             NaN         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            25_507.03     1_015.40    26_522.43       0.4410          1.0022         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            25_507.03     1_025.44    26_532.48       0.6270          1.0011         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           25_507.03       967.57    26_474.60       0.4120          1.0025         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           25_507.03     1_057.87    26_564.90       0.5901          1.0013         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           25_507.03     1_003.16    26_510.19       0.3992          1.0027         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           25_507.03     1_097.73    26_604.76       0.5720          1.0014         8.42
IVF-Binary-1024-nl158-pca (self)                      25_507.03     3_156.34    28_663.38       0.7126          1.0421         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            23_164.71       850.45    24_015.16       0.0975             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            23_164.71       855.26    24_019.98       0.0942             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            23_164.71       868.75    24_033.46       0.0902             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           23_164.71       944.00    24_108.71       0.4463          1.0022         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           23_164.71     1_023.07    24_187.78       0.6354          1.0011         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           23_164.71       948.85    24_113.57       0.4329          1.0024         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           23_164.71     1_036.03    24_200.74       0.6184          1.0012         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           23_164.71       972.18    24_136.89       0.4167          1.0025         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           23_164.71     1_060.36    24_225.07       0.5972          1.0013         8.54
IVF-Binary-1024-nl223-pca (self)                      23_164.71     3_077.56    26_242.27       0.7111          1.0429         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            23_158.00       867.06    24_025.06       0.0984             NaN         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            23_158.00       867.23    24_025.23       0.0964             NaN         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            23_158.00       878.32    24_036.32       0.0920             NaN         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           23_158.00       965.52    24_123.52       0.4496          1.0022         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           23_158.00     1_040.11    24_198.11       0.6394          1.0011         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           23_158.00       959.18    24_117.18       0.4419          1.0023         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           23_158.00     1_046.44    24_204.44       0.6303          1.0011         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           23_158.00       975.07    24_133.07       0.4245          1.0024         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           23_158.00     1_077.19    24_235.19       0.6075          1.0012         8.73
IVF-Binary-1024-nl316-pca (self)                      23_158.00     3_115.84    26_273.84       0.7118          1.0427         8.73
IVF-Binary-512-nl158-np7-rf0-signed (query)           13_534.45       433.69    13_968.14       0.0550             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-signed (query)          13_534.45       446.28    13_980.73       0.0503             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-signed (query)          13_534.45       460.87    13_995.32       0.0485             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-signed (query)          13_534.45       522.55    14_057.00       0.3197          1.0037         4.36
IVF-Binary-512-nl158-np7-rf20-signed (query)          13_534.45       604.67    14_139.12       0.5000          1.0019         4.36
IVF-Binary-512-nl158-np12-rf10-signed (query)         13_534.45       539.70    14_074.15       0.2982          1.0041         4.36
IVF-Binary-512-nl158-np12-rf20-signed (query)         13_534.45       626.40    14_160.85       0.4687          1.0022         4.36
IVF-Binary-512-nl158-np17-rf10-signed (query)         13_534.45       560.21    14_094.66       0.2893          1.0043         4.36
IVF-Binary-512-nl158-np17-rf20-signed (query)         13_534.45       651.69    14_186.14       0.4553          1.0024         4.36
IVF-Binary-512-nl158-signed (self)                    13_534.45     1_718.50    15_252.95       0.8734          1.0152         4.36
IVF-Binary-512-nl223-np11-rf0-signed (query)          11_217.67       434.19    11_651.86       0.0551             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-signed (query)          11_217.67       437.99    11_655.65       0.0524             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-signed (query)          11_217.67       448.01    11_665.67       0.0500             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-signed (query)         11_217.67       523.39    11_741.05       0.3198          1.0039         4.49
IVF-Binary-512-nl223-np11-rf20-signed (query)         11_217.67       604.01    11_821.68       0.4995          1.0021         4.49
IVF-Binary-512-nl223-np14-rf10-signed (query)         11_217.67       536.25    11_753.92       0.3078          1.0041         4.49
IVF-Binary-512-nl223-np14-rf20-signed (query)         11_217.67       614.51    11_832.18       0.4831          1.0022         4.49
IVF-Binary-512-nl223-np21-rf10-signed (query)         11_217.67       541.77    11_759.44       0.2965          1.0042         4.49
IVF-Binary-512-nl223-np21-rf20-signed (query)         11_217.67       629.23    11_846.90       0.4669          1.0023         4.49
IVF-Binary-512-nl223-signed (self)                    11_217.67     1_655.18    12_872.85       0.8701          1.0161         4.49
IVF-Binary-512-nl316-np15-rf0-signed (query)          11_199.17       479.43    11_678.59       0.0552             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-signed (query)          11_199.17       448.95    11_648.12       0.0536             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-signed (query)          11_199.17       457.76    11_656.92       0.0508             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-signed (query)         11_199.17       540.16    11_739.33       0.3216          1.0039         4.67
IVF-Binary-512-nl316-np15-rf20-signed (query)         11_199.17       618.85    11_818.02       0.5021          1.0021         4.67
IVF-Binary-512-nl316-np17-rf10-signed (query)         11_199.17       543.32    11_742.48       0.3146          1.0040         4.67
IVF-Binary-512-nl316-np17-rf20-signed (query)         11_199.17       621.31    11_820.48       0.4930          1.0021         4.67
IVF-Binary-512-nl316-np25-rf10-signed (query)         11_199.17       548.49    11_747.65       0.3015          1.0041         4.67
IVF-Binary-512-nl316-np25-rf20-signed (query)         11_199.17       636.87    11_836.03       0.4742          1.0023         4.67
IVF-Binary-512-nl316-signed (self)                    11_199.17     1_698.02    12_897.18       0.8706          1.0160         4.67
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
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        38.01    21_950.17    21_988.18       1.0000          1.0000       195.31
Exhaustive (self)                                         38.01    73_317.47    73_355.48       1.0000          1.0000       195.31
ExhaustiveBinary-256-random_no_rr (query)             11_964.67       571.31    12_535.99       0.0086             NaN         2.53
ExhaustiveBinary-256-random-rf10 (query)              11_964.67       744.21    12_708.88       0.0837          1.0101         2.53
ExhaustiveBinary-256-random-rf20 (query)              11_964.67       921.86    12_886.53       0.1557          1.0064         2.53
ExhaustiveBinary-256-random (self)                    11_964.67     2_431.10    14_395.78       0.6166          5.6565         2.53
ExhaustiveBinary-256-pca_no_rr (query)                13_210.10       577.07    13_787.17       0.0142             NaN         2.53
ExhaustiveBinary-256-pca-rf10 (query)                 13_210.10       746.40    13_956.50       0.0739          1.0100         2.53
ExhaustiveBinary-256-pca-rf20 (query)                 13_210.10       948.93    14_159.02       0.1169          1.0048         2.53
ExhaustiveBinary-256-pca (self)                       13_210.10     2_496.79    15_706.89       0.1825          1.5785         2.53
ExhaustiveBinary-512-random_no_rr (query)             23_494.36     1_085.50    24_579.86       0.0192             NaN         5.05
ExhaustiveBinary-512-random-rf10 (query)              23_494.36     1_248.90    24_743.27       0.1384          1.0052         5.05
ExhaustiveBinary-512-random-rf20 (query)              23_494.36     1_423.59    24_917.95       0.2398          1.0032         5.05
ExhaustiveBinary-512-random (self)                    23_494.36     4_469.07    27_963.43       0.7044          7.2752         5.05
ExhaustiveBinary-512-pca_no_rr (query)                24_833.61     1_096.19    25_929.80       0.0097             NaN         5.06
ExhaustiveBinary-512-pca-rf10 (query)                 24_833.61     1_357.90    26_191.51       0.0527          1.0187         5.06
ExhaustiveBinary-512-pca-rf20 (query)                 24_833.61     1_439.60    26_273.20       0.0858          1.0069         5.06
ExhaustiveBinary-512-pca (self)                       24_833.61     4_182.64    29_016.25       0.1295          2.6401         5.06
ExhaustiveBinary-1024-random_no_rr (query)            46_987.94     2_027.62    49_015.55       0.0426             NaN        10.10
ExhaustiveBinary-1024-random-rf10 (query)             46_987.94     2_228.17    49_216.11       0.2373          1.0030        10.10
ExhaustiveBinary-1024-random-rf20 (query)             46_987.94     2_434.32    49_422.26       0.3715          1.0017        10.10
ExhaustiveBinary-1024-random (self)                   46_987.94     7_338.94    54_326.88       0.7686          8.3967        10.10
ExhaustiveBinary-1024-pca_no_rr (query)               48_447.06     2_055.20    50_502.26       0.0086             NaN        10.11
ExhaustiveBinary-1024-pca-rf10 (query)                48_447.06     2_237.86    50_684.92       0.0490          1.0332        10.11
ExhaustiveBinary-1024-pca-rf20 (query)                48_447.06     2_425.06    50_872.12       0.0797          1.0112        10.11
ExhaustiveBinary-1024-pca (self)                      48_447.06     7_442.53    55_889.59       0.1033          8.4885        10.11
ExhaustiveBinary-1024-signed_no_rr (query)            47_026.61     2_034.21    49_060.82       0.0426             NaN        10.10
ExhaustiveBinary-1024-signed-rf10 (query)             47_026.61     2_214.26    49_240.87       0.2373          1.0030        10.10
ExhaustiveBinary-1024-signed-rf20 (query)             47_026.61     2_399.79    49_426.40       0.3715          1.0017        10.10
ExhaustiveBinary-1024-signed (self)                   47_026.61     7_436.20    54_462.81       0.7686          8.3967        10.10
IVF-Binary-256-nl158-np7-rf0-random (query)           17_417.44       472.03    17_889.48       0.0178             NaN         3.14
IVF-Binary-256-nl158-np12-rf0-random (query)          17_417.44       479.84    17_897.29       0.0138             NaN         3.14
IVF-Binary-256-nl158-np17-rf0-random (query)          17_417.44       486.56    17_904.00       0.0122             NaN         3.14
IVF-Binary-256-nl158-np7-rf10-random (query)          17_417.44       602.43    18_019.87       0.1535          1.0042         3.14
IVF-Binary-256-nl158-np7-rf20-random (query)          17_417.44       724.14    18_141.58       0.2799          1.0021         3.14
IVF-Binary-256-nl158-np12-rf10-random (query)         17_417.44       613.16    18_030.60       0.1278          1.0053         3.14
IVF-Binary-256-nl158-np12-rf20-random (query)         17_417.44       742.30    18_159.74       0.2376          1.0029         3.14
IVF-Binary-256-nl158-np17-rf10-random (query)         17_417.44       625.35    18_042.79       0.1162          1.0062         3.14
IVF-Binary-256-nl158-np17-rf20-random (query)         17_417.44       760.17    18_177.61       0.2180          1.0035         3.14
IVF-Binary-256-nl158-random (self)                    17_417.44     1_931.79    19_349.23       0.7825          1.0294         3.14
IVF-Binary-256-nl223-np11-rf0-random (query)          11_986.67       502.75    12_489.42       0.0184             NaN         3.40
IVF-Binary-256-nl223-np14-rf0-random (query)          11_986.67       509.08    12_495.75       0.0156             NaN         3.40
IVF-Binary-256-nl223-np21-rf0-random (query)          11_986.67       523.04    12_509.71       0.0133             NaN         3.40
IVF-Binary-256-nl223-np11-rf10-random (query)         11_986.67       636.48    12_623.15       0.1594          1.0038         3.40
IVF-Binary-256-nl223-np11-rf20-random (query)         11_986.67       761.28    12_747.94       0.2906          1.0021         3.40
IVF-Binary-256-nl223-np14-rf10-random (query)         11_986.67       640.60    12_627.27       0.1416          1.0046         3.40
IVF-Binary-256-nl223-np14-rf20-random (query)         11_986.67       767.48    12_754.15       0.2623          1.0024         3.40
IVF-Binary-256-nl223-np21-rf10-random (query)         11_986.67       663.86    12_650.52       0.1240          1.0055         3.40
IVF-Binary-256-nl223-np21-rf20-random (query)         11_986.67       783.17    12_769.84       0.2335          1.0030         3.40
IVF-Binary-256-nl223-random (self)                    11_986.67     2_012.22    13_998.88       0.7845          1.0290         3.40
IVF-Binary-256-nl316-np15-rf0-random (query)          11_997.51       533.40    12_530.92       0.0178             NaN         3.76
IVF-Binary-256-nl316-np17-rf0-random (query)          11_997.51       535.45    12_532.97       0.0165             NaN         3.76
IVF-Binary-256-nl316-np25-rf0-random (query)          11_997.51       559.12    12_556.64       0.0140             NaN         3.76
IVF-Binary-256-nl316-np15-rf10-random (query)         11_997.51       673.25    12_670.76       0.1595          1.0038         3.76
IVF-Binary-256-nl316-np15-rf20-random (query)         11_997.51       795.98    12_793.49       0.2939          1.0020         3.76
IVF-Binary-256-nl316-np17-rf10-random (query)         11_997.51       668.14    12_665.65       0.1506          1.0042         3.76
IVF-Binary-256-nl316-np17-rf20-random (query)         11_997.51       792.95    12_790.46       0.2786          1.0022         3.76
IVF-Binary-256-nl316-np25-rf10-random (query)         11_997.51       683.29    12_680.80       0.1321          1.0051         3.76
IVF-Binary-256-nl316-np25-rf20-random (query)         11_997.51       809.07    12_806.58       0.2470          1.0027         3.76
IVF-Binary-256-nl316-random (self)                    11_997.51     2_101.27    14_098.78       0.7846          1.0290         3.76
IVF-Binary-256-nl158-np7-rf0-pca (query)              18_712.03       485.40    19_197.43       0.0446             NaN         3.15
IVF-Binary-256-nl158-np12-rf0-pca (query)             18_712.03       489.85    19_201.88       0.0369             NaN         3.15
IVF-Binary-256-nl158-np17-rf0-pca (query)             18_712.03       495.95    19_207.98       0.0328             NaN         3.15
IVF-Binary-256-nl158-np7-rf10-pca (query)             18_712.03       616.58    19_328.61       0.2517          1.0019         3.15
IVF-Binary-256-nl158-np7-rf20-pca (query)             18_712.03       737.99    19_450.02       0.3958          1.0011         3.15
IVF-Binary-256-nl158-np12-rf10-pca (query)            18_712.03       623.62    19_335.65       0.2065          1.0023         3.15
IVF-Binary-256-nl158-np12-rf20-pca (query)            18_712.03       757.28    19_469.31       0.3249          1.0014         3.15
IVF-Binary-256-nl158-np17-rf10-pca (query)            18_712.03       645.42    19_357.45       0.1808          1.0026         3.15
IVF-Binary-256-nl158-np17-rf20-pca (query)            18_712.03       771.15    19_483.18       0.2839          1.0017         3.15
IVF-Binary-256-nl158-pca (self)                       18_712.03     2_014.13    20_726.16       0.2954          1.2366         3.15
IVF-Binary-256-nl223-np11-rf0-pca (query)             13_192.96       501.17    13_694.13       0.0455             NaN         3.40
IVF-Binary-256-nl223-np14-rf0-pca (query)             13_192.96       503.54    13_696.50       0.0412             NaN         3.40
IVF-Binary-256-nl223-np21-rf0-pca (query)             13_192.96       509.05    13_702.01       0.0359             NaN         3.40
IVF-Binary-256-nl223-np11-rf10-pca (query)            13_192.96       635.38    13_828.34       0.2576          1.0018         3.40
IVF-Binary-256-nl223-np11-rf20-pca (query)            13_192.96       759.14    13_952.10       0.4039          1.0011         3.40
IVF-Binary-256-nl223-np14-rf10-pca (query)            13_192.96       636.10    13_829.06       0.2331          1.0020         3.40
IVF-Binary-256-nl223-np14-rf20-pca (query)            13_192.96       769.55    13_962.51       0.3664          1.0012         3.40
IVF-Binary-256-nl223-np21-rf10-pca (query)            13_192.96       647.00    13_839.96       0.2008          1.0023         3.40
IVF-Binary-256-nl223-np21-rf20-pca (query)            13_192.96       785.46    13_978.42       0.3160          1.0015         3.40
IVF-Binary-256-nl223-pca (self)                       13_192.96     2_086.46    15_279.42       0.3168          1.2064         3.40
IVF-Binary-256-nl316-np15-rf0-pca (query)             13_344.10       538.43    13_882.53       0.0461             NaN         3.77
IVF-Binary-256-nl316-np17-rf0-pca (query)             13_344.10       534.12    13_878.22       0.0441             NaN         3.77
IVF-Binary-256-nl316-np25-rf0-pca (query)             13_344.10       540.15    13_884.25       0.0387             NaN         3.77
IVF-Binary-256-nl316-np15-rf10-pca (query)            13_344.10       667.62    14_011.71       0.2622          1.0018         3.77
IVF-Binary-256-nl316-np15-rf20-pca (query)            13_344.10       791.68    14_135.78       0.4106          1.0010         3.77
IVF-Binary-256-nl316-np17-rf10-pca (query)            13_344.10       666.46    14_010.56       0.2490          1.0019         3.77
IVF-Binary-256-nl316-np17-rf20-pca (query)            13_344.10       797.29    14_141.39       0.3908          1.0011         3.77
IVF-Binary-256-nl316-np25-rf10-pca (query)            13_344.10       675.21    14_019.30       0.2162          1.0022         3.77
IVF-Binary-256-nl316-np25-rf20-pca (query)            13_344.10       811.76    14_155.86       0.3408          1.0013         3.77
IVF-Binary-256-nl316-pca (self)                       13_344.10     2_153.65    15_497.74       0.3283          1.1939         3.77
IVF-Binary-512-nl158-np7-rf0-random (query)           29_104.17       880.47    29_984.65       0.0294             NaN         5.67
IVF-Binary-512-nl158-np12-rf0-random (query)          29_104.17       892.95    29_997.12       0.0255             NaN         5.67
IVF-Binary-512-nl158-np17-rf0-random (query)          29_104.17       907.79    30_011.96       0.0236             NaN         5.67
IVF-Binary-512-nl158-np7-rf10-random (query)          29_104.17     1_008.37    30_112.54       0.2105          1.0028         5.67
IVF-Binary-512-nl158-np7-rf20-random (query)          29_104.17     1_127.16    30_231.33       0.3621          1.0015         5.67
IVF-Binary-512-nl158-np12-rf10-random (query)         29_104.17     1_037.06    30_141.24       0.1874          1.0032         5.67
IVF-Binary-512-nl158-np12-rf20-random (query)         29_104.17     1_153.10    30_257.27       0.3266          1.0018         5.67
IVF-Binary-512-nl158-np17-rf10-random (query)         29_104.17     1_040.89    30_145.07       0.1767          1.0035         5.67
IVF-Binary-512-nl158-np17-rf20-random (query)         29_104.17     1_176.83    30_281.00       0.3087          1.0019         5.67
IVF-Binary-512-nl158-random (self)                    29_104.17     3_308.02    32_412.19       0.8789          1.0130         5.67
IVF-Binary-512-nl223-np11-rf0-random (query)          23_788.12       904.84    24_692.96       0.0304             NaN         5.92
IVF-Binary-512-nl223-np14-rf0-random (query)          23_788.12       908.65    24_696.77       0.0272             NaN         5.92
IVF-Binary-512-nl223-np21-rf0-random (query)          23_788.12       918.74    24_706.86       0.0245             NaN         5.92
IVF-Binary-512-nl223-np11-rf10-random (query)         23_788.12     1_041.44    24_829.56       0.2167          1.0026         5.92
IVF-Binary-512-nl223-np11-rf20-random (query)         23_788.12     1_150.66    24_938.77       0.3718          1.0014         5.92
IVF-Binary-512-nl223-np14-rf10-random (query)         23_788.12     1_039.82    24_827.94       0.2004          1.0029         5.92
IVF-Binary-512-nl223-np14-rf20-random (query)         23_788.12     1_161.17    24_949.29       0.3481          1.0016         5.92
IVF-Binary-512-nl223-np21-rf10-random (query)         23_788.12     1_051.52    24_839.63       0.1847          1.0032         5.92
IVF-Binary-512-nl223-np21-rf20-random (query)         23_788.12     1_191.08    24_979.19       0.3242          1.0018         5.92
IVF-Binary-512-nl223-random (self)                    23_788.12     3_355.82    27_143.94       0.8799          1.0128         5.92
IVF-Binary-512-nl316-np15-rf0-random (query)          23_677.61       954.11    24_631.73       0.0300             NaN         6.29
IVF-Binary-512-nl316-np17-rf0-random (query)          23_677.61       940.87    24_618.48       0.0284             NaN         6.29
IVF-Binary-512-nl316-np25-rf0-random (query)          23_677.61       957.73    24_635.34       0.0258             NaN         6.29
IVF-Binary-512-nl316-np15-rf10-random (query)         23_677.61     1_068.60    24_746.21       0.2175          1.0026         6.29
IVF-Binary-512-nl316-np15-rf20-random (query)         23_677.61     1_185.41    24_863.02       0.3751          1.0014         6.29
IVF-Binary-512-nl316-np17-rf10-random (query)         23_677.61     1_076.03    24_753.64       0.2089          1.0028         6.29
IVF-Binary-512-nl316-np17-rf20-random (query)         23_677.61     1_189.21    24_866.83       0.3623          1.0015         6.29
IVF-Binary-512-nl316-np25-rf10-random (query)         23_677.61     1_089.37    24_766.99       0.1922          1.0031         6.29
IVF-Binary-512-nl316-np25-rf20-random (query)         23_677.61     1_208.13    24_885.75       0.3357          1.0017         6.29
IVF-Binary-512-nl316-random (self)                    23_677.61     3_436.17    27_113.79       0.8804          1.0128         6.29
IVF-Binary-512-nl158-np7-rf0-pca (query)              30_630.72       896.44    31_527.16       0.0425             NaN         5.67
IVF-Binary-512-nl158-np12-rf0-pca (query)             30_630.72       912.12    31_542.84       0.0335             NaN         5.67
IVF-Binary-512-nl158-np17-rf0-pca (query)             30_630.72       927.62    31_558.34       0.0284             NaN         5.67
IVF-Binary-512-nl158-np7-rf10-pca (query)             30_630.72     1_035.84    31_666.56       0.2373          1.0020         5.67
IVF-Binary-512-nl158-np7-rf20-pca (query)             30_630.72     1_154.07    31_784.79       0.3764          1.0012         5.67
IVF-Binary-512-nl158-np12-rf10-pca (query)            30_630.72     1_042.75    31_673.47       0.1861          1.0025         5.67
IVF-Binary-512-nl158-np12-rf20-pca (query)            30_630.72     1_187.99    31_818.71       0.2964          1.0016         5.67
IVF-Binary-512-nl158-np17-rf10-pca (query)            30_630.72     1_063.09    31_693.81       0.1576          1.0029         5.67
IVF-Binary-512-nl158-np17-rf20-pca (query)            30_630.72     1_202.05    31_832.77       0.2504          1.0019         5.67
IVF-Binary-512-nl158-pca (self)                       30_630.72     3_446.31    34_077.03       0.2400          1.3093         5.67
IVF-Binary-512-nl223-np11-rf0-pca (query)             25_538.23     1_017.43    26_555.66       0.0430             NaN         5.93
IVF-Binary-512-nl223-np14-rf0-pca (query)             25_538.23       983.36    26_521.59       0.0383             NaN         5.93
IVF-Binary-512-nl223-np21-rf0-pca (query)             25_538.23     1_040.17    26_578.40       0.0321             NaN         5.93
IVF-Binary-512-nl223-np11-rf10-pca (query)            25_538.23     1_181.67    26_719.90       0.2425          1.0019         5.93
IVF-Binary-512-nl223-np11-rf20-pca (query)            25_538.23     1_242.15    26_780.37       0.3842          1.0011         5.93
IVF-Binary-512-nl223-np14-rf10-pca (query)            25_538.23     1_126.27    26_664.49       0.2157          1.0022         5.93
IVF-Binary-512-nl223-np14-rf20-pca (query)            25_538.23     1_250.90    26_789.13       0.3424          1.0013         5.93
IVF-Binary-512-nl223-np21-rf10-pca (query)            25_538.23     1_129.11    26_667.34       0.1799          1.0026         5.93
IVF-Binary-512-nl223-np21-rf20-pca (query)            25_538.23     1_317.20    26_855.42       0.2853          1.0016         5.93
IVF-Binary-512-nl223-pca (self)                       25_538.23     4_005.00    29_543.23       0.2639          1.2612         5.93
IVF-Binary-512-nl316-np15-rf0-pca (query)             25_132.00       969.11    26_101.10       0.0441             NaN         6.29
IVF-Binary-512-nl316-np17-rf0-pca (query)             25_132.00       953.61    26_085.61       0.0415             NaN         6.29
IVF-Binary-512-nl316-np25-rf0-pca (query)             25_132.00       962.35    26_094.35       0.0352             NaN         6.29
IVF-Binary-512-nl316-np15-rf10-pca (query)            25_132.00     1_087.34    26_219.34       0.2473          1.0019         6.29
IVF-Binary-512-nl316-np15-rf20-pca (query)            25_132.00     1_206.36    26_338.36       0.3903          1.0011         6.29
IVF-Binary-512-nl316-np17-rf10-pca (query)            25_132.00     1_080.64    26_212.64       0.2335          1.0020         6.29
IVF-Binary-512-nl316-np17-rf20-pca (query)            25_132.00     1_209.05    26_341.04       0.3690          1.0012         6.29
IVF-Binary-512-nl316-np25-rf10-pca (query)            25_132.00     1_092.81    26_224.81       0.1974          1.0023         6.29
IVF-Binary-512-nl316-np25-rf20-pca (query)            25_132.00     1_227.91    26_359.91       0.3133          1.0015         6.29
IVF-Binary-512-nl316-pca (self)                       25_132.00     3_549.72    28_681.72       0.2766          1.2422         6.29
IVF-Binary-1024-nl158-np7-rf0-random (query)          52_767.19     1_703.59    54_470.79       0.0554             NaN        10.72
IVF-Binary-1024-nl158-np12-rf0-random (query)         52_767.19     1_724.08    54_491.27       0.0516             NaN        10.72
IVF-Binary-1024-nl158-np17-rf0-random (query)         52_767.19     1_744.83    54_512.02       0.0495             NaN        10.72
IVF-Binary-1024-nl158-np7-rf10-random (query)         52_767.19     1_823.44    54_590.63       0.3144          1.0018        10.72
IVF-Binary-1024-nl158-np7-rf20-random (query)         52_767.19     1_948.31    54_715.50       0.4901          1.0009        10.72
IVF-Binary-1024-nl158-np12-rf10-random (query)        52_767.19     1_846.93    54_614.13       0.2972          1.0019        10.72
IVF-Binary-1024-nl158-np12-rf20-random (query)        52_767.19     1_979.28    54_746.48       0.4668          1.0010        10.72
IVF-Binary-1024-nl158-np17-rf10-random (query)        52_767.19     1_873.24    54_640.43       0.2883          1.0020        10.72
IVF-Binary-1024-nl158-np17-rf20-random (query)        52_767.19     2_009.65    54_776.85       0.4539          1.0010        10.72
IVF-Binary-1024-nl158-random (self)                   52_767.19     6_064.13    58_831.33       0.9463          1.0044        10.72
IVF-Binary-1024-nl223-np11-rf0-random (query)         47_494.59     1_723.72    49_218.31       0.0569             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-random (query)         47_494.59     1_733.32    49_227.91       0.0538             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-random (query)         47_494.59     1_749.63    49_244.22       0.0509             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-random (query)        47_494.59     1_846.16    49_340.76       0.3214          1.0017        10.98
IVF-Binary-1024-nl223-np11-rf20-random (query)        47_494.59     1_997.68    49_492.28       0.4986          1.0009        10.98
IVF-Binary-1024-nl223-np14-rf10-random (query)        47_494.59     1_853.60    49_348.19       0.3081          1.0018        10.98
IVF-Binary-1024-nl223-np14-rf20-random (query)        47_494.59     1_982.31    49_476.91       0.4819          1.0009        10.98
IVF-Binary-1024-nl223-np21-rf10-random (query)        47_494.59     1_886.20    49_380.79       0.2960          1.0019        10.98
IVF-Binary-1024-nl223-np21-rf20-random (query)        47_494.59     2_009.13    49_503.72       0.4651          1.0010        10.98
IVF-Binary-1024-nl223-random (self)                   47_494.59     6_086.85    53_581.44       0.9468          1.0044        10.98
IVF-Binary-1024-nl316-np15-rf0-random (query)         47_245.73     1_762.24    49_007.97       0.0566             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-random (query)         47_245.73     1_809.81    49_055.54       0.0550             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-random (query)         47_245.73     1_782.38    49_028.11       0.0519             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-random (query)        47_245.73     1_879.56    49_125.29       0.3220          1.0017        11.34
IVF-Binary-1024-nl316-np15-rf20-random (query)        47_245.73     2_006.00    49_251.73       0.5007          1.0009        11.34
IVF-Binary-1024-nl316-np17-rf10-random (query)        47_245.73     1_883.19    49_128.92       0.3151          1.0018        11.34
IVF-Binary-1024-nl316-np17-rf20-random (query)        47_245.73     2_008.17    49_253.90       0.4918          1.0009        11.34
IVF-Binary-1024-nl316-np25-rf10-random (query)        47_245.73     1_900.39    49_146.13       0.3018          1.0019        11.34
IVF-Binary-1024-nl316-np25-rf20-random (query)        47_245.73     2_029.45    49_275.18       0.4739          1.0010        11.34
IVF-Binary-1024-nl316-random (self)                   47_245.73     6_162.67    53_408.40       0.9472          1.0043        11.34
IVF-Binary-1024-nl158-np7-rf0-pca (query)             54_126.59     1_733.34    55_859.93       0.0476             NaN        10.73
IVF-Binary-1024-nl158-np12-rf0-pca (query)            54_126.59     1_774.14    55_900.73       0.0364             NaN        10.73
IVF-Binary-1024-nl158-np17-rf0-pca (query)            54_126.59     1_822.30    55_948.89       0.0304             NaN        10.73
IVF-Binary-1024-nl158-np7-rf10-pca (query)            54_126.59     1_892.33    56_018.91       0.2582          1.0019        10.73
IVF-Binary-1024-nl158-np7-rf20-pca (query)            54_126.59     1_993.97    56_120.56       0.4018          1.0011        10.73
IVF-Binary-1024-nl158-np12-rf10-pca (query)           54_126.59     1_899.79    56_026.37       0.1973          1.0025        10.73
IVF-Binary-1024-nl158-np12-rf20-pca (query)           54_126.59     2_029.05    56_155.64       0.3113          1.0015        10.73
IVF-Binary-1024-nl158-np17-rf10-pca (query)           54_126.59     1_900.66    56_027.24       0.1626          1.0030        10.73
IVF-Binary-1024-nl158-np17-rf20-pca (query)           54_126.59     2_077.69    56_204.28       0.2580          1.0019        10.73
IVF-Binary-1024-nl158-pca (self)                      54_126.59     6_284.66    60_411.25       0.2039          1.3825        10.73
IVF-Binary-1024-nl223-np11-rf0-pca (query)            48_517.70     1_778.34    50_296.04       0.0490             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-pca (query)            48_517.70     1_776.05    50_293.75       0.0430             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-pca (query)            48_517.70     1_836.28    50_353.98       0.0349             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-pca (query)           48_517.70     1_899.43    50_417.14       0.2631          1.0018        10.98
IVF-Binary-1024-nl223-np11-rf20-pca (query)           48_517.70     2_214.83    50_732.54       0.4107          1.0011        10.98
IVF-Binary-1024-nl223-np14-rf10-pca (query)           48_517.70     1_907.93    50_425.64       0.2322          1.0021        10.98
IVF-Binary-1024-nl223-np14-rf20-pca (query)           48_517.70     2_039.80    50_557.51       0.3647          1.0013        10.98
IVF-Binary-1024-nl223-np21-rf10-pca (query)           48_517.70     1_908.39    50_426.10       0.1896          1.0026        10.98
IVF-Binary-1024-nl223-np21-rf20-pca (query)           48_517.70     2_045.72    50_563.42       0.2994          1.0016        10.98
IVF-Binary-1024-nl223-pca (self)                      48_517.70     6_240.92    54_758.63       0.2292          1.3129        10.98
IVF-Binary-1024-nl316-np15-rf0-pca (query)            48_664.97     1_783.98    50_448.96       0.0502             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-pca (query)            48_664.97     1_789.80    50_454.77       0.0470             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-pca (query)            48_664.97     1_802.39    50_467.36       0.0390             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-pca (query)           48_664.97     1_913.72    50_578.69       0.2684          1.0018        11.34
IVF-Binary-1024-nl316-np15-rf20-pca (query)           48_664.97     2_064.45    50_729.43       0.4188          1.0010        11.34
IVF-Binary-1024-nl316-np17-rf10-pca (query)           48_664.97     2_062.43    50_727.41       0.2524          1.0019        11.34
IVF-Binary-1024-nl316-np17-rf20-pca (query)           48_664.97     2_049.32    50_714.29       0.3947          1.0011        11.34
IVF-Binary-1024-nl316-np25-rf10-pca (query)           48_664.97     1_951.25    50_616.23       0.2101          1.0023        11.34
IVF-Binary-1024-nl316-np25-rf20-pca (query)           48_664.97     2_067.11    50_732.08       0.3316          1.0014        11.34
IVF-Binary-1024-nl316-pca (self)                      48_664.97     6_400.55    55_065.53       0.2428          1.2866        11.34
IVF-Binary-1024-nl158-np7-rf0-signed (query)          53_402.48     1_833.60    55_236.08       0.0554             NaN        10.72
IVF-Binary-1024-nl158-np12-rf0-signed (query)         53_402.48     1_748.36    55_150.84       0.0516             NaN        10.72
IVF-Binary-1024-nl158-np17-rf0-signed (query)         53_402.48     1_800.07    55_202.55       0.0495             NaN        10.72
IVF-Binary-1024-nl158-np7-rf10-signed (query)         53_402.48     1_831.78    55_234.26       0.3144          1.0018        10.72
IVF-Binary-1024-nl158-np7-rf20-signed (query)         53_402.48     1_951.04    55_353.52       0.4901          1.0009        10.72
IVF-Binary-1024-nl158-np12-rf10-signed (query)        53_402.48     1_878.31    55_280.79       0.2972          1.0019        10.72
IVF-Binary-1024-nl158-np12-rf20-signed (query)        53_402.48     1_978.29    55_380.77       0.4668          1.0010        10.72
IVF-Binary-1024-nl158-np17-rf10-signed (query)        53_402.48     1_881.09    55_283.57       0.2883          1.0020        10.72
IVF-Binary-1024-nl158-np17-rf20-signed (query)        53_402.48     2_021.64    55_424.11       0.4539          1.0010        10.72
IVF-Binary-1024-nl158-signed (self)                   53_402.48     6_092.19    59_494.67       0.9463          1.0044        10.72
IVF-Binary-1024-nl223-np11-rf0-signed (query)         47_343.94     1_724.04    49_067.98       0.0569             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-signed (query)         47_343.94     1_731.95    49_075.89       0.0538             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-signed (query)         47_343.94     1_749.51    49_093.45       0.0509             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-signed (query)        47_343.94     1_869.43    49_213.37       0.3214          1.0017        10.98
IVF-Binary-1024-nl223-np11-rf20-signed (query)        47_343.94     1_967.74    49_311.68       0.4986          1.0009        10.98
IVF-Binary-1024-nl223-np14-rf10-signed (query)        47_343.94     1_856.90    49_200.84       0.3081          1.0018        10.98
IVF-Binary-1024-nl223-np14-rf20-signed (query)        47_343.94     1_980.19    49_324.13       0.4819          1.0009        10.98
IVF-Binary-1024-nl223-np21-rf10-signed (query)        47_343.94     1_879.68    49_223.62       0.2960          1.0019        10.98
IVF-Binary-1024-nl223-np21-rf20-signed (query)        47_343.94     2_018.84    49_362.77       0.4651          1.0010        10.98
IVF-Binary-1024-nl223-signed (self)                   47_343.94     6_071.32    53_415.26       0.9468          1.0044        10.98
IVF-Binary-1024-nl316-np15-rf0-signed (query)         47_235.50     1_760.34    48_995.85       0.0566             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-signed (query)         47_235.50     1_764.22    48_999.72       0.0550             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-signed (query)         47_235.50     1_779.73    49_015.23       0.0519             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-signed (query)        47_235.50     1_878.80    49_114.30       0.3220          1.0017        11.34
IVF-Binary-1024-nl316-np15-rf20-signed (query)        47_235.50     2_001.08    49_236.58       0.5007          1.0009        11.34
IVF-Binary-1024-nl316-np17-rf10-signed (query)        47_235.50     1_880.59    49_116.10       0.3151          1.0018        11.34
IVF-Binary-1024-nl316-np17-rf20-signed (query)        47_235.50     2_013.17    49_248.67       0.4918          1.0009        11.34
IVF-Binary-1024-nl316-np25-rf10-signed (query)        47_235.50     1_907.04    49_142.54       0.3018          1.0019        11.34
IVF-Binary-1024-nl316-np25-rf20-signed (query)        47_235.50     2_028.00    49_263.50       0.4739          1.0010        11.34
IVF-Binary-1024-nl316-signed (self)                   47_235.50     6_156.51    53_392.01       0.9472          1.0043        11.34
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
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         9.55     4_212.44     4_221.99       1.0000          1.0000        48.83
Exhaustive (self)                                          9.55    14_167.45    14_177.00       1.0000          1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                           1_434.14       776.46     2_210.60       0.6105             NaN         2.84
ExhaustiveRaBitQ-rf5 (query)                           1_434.14       842.64     2_276.78       0.9692          1.0007         2.84
ExhaustiveRaBitQ-rf10 (query)                          1_434.14       897.53     2_331.67       0.9968          1.0001         2.84
ExhaustiveRaBitQ-rf20 (query)                          1_434.14       991.60     2_425.74       0.9999          1.0000         2.84
ExhaustiveRaBitQ (self)                                1_434.14     2_983.18     4_417.32       0.9967          1.0001         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       2_103.09       264.51     2_367.60       0.6023             NaN         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      2_103.09       398.71     2_501.80       0.6110             NaN         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      2_103.09       542.16     2_645.25       0.6120             NaN         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      2_103.09       347.24     2_450.33       0.9495          1.0025         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      2_103.09       421.64     2_524.73       0.9517          1.0025         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     2_103.09       502.60     2_605.69       0.9905          1.0003         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     2_103.09       591.13     2_694.22       0.9935          1.0003         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     2_103.09       659.41     2_762.50       0.9965          1.0001         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     2_103.09       745.74     2_848.83       0.9997          1.0000         2.89
IVF-RaBitQ-nl158 (self)                                2_103.09     2_478.79     4_581.88       0.9997          1.0000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_134.43       330.52     1_464.95       0.6111             NaN         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_134.43       412.05     1_546.48       0.6128             NaN         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_134.43       602.06     1_736.49       0.6133             NaN         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_134.43       425.90     1_560.33       0.9830          1.0007         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_134.43       501.06     1_635.49       0.9856          1.0007         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_134.43       507.69     1_642.12       0.9938          1.0002         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_134.43       593.12     1_727.55       0.9967          1.0001         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_134.43       706.03     1_840.46       0.9968          1.0001         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_134.43       794.80     1_929.23       0.9999          1.0000         2.95
IVF-RaBitQ-nl223 (self)                                1_134.43     2_655.79     3_790.21       0.9999          1.0000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_296.03       408.13     1_704.16       0.6130             NaN         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_296.03       455.64     1_751.67       0.6141             NaN         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_296.03       650.17     1_946.20       0.6147             NaN         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_296.03       529.37     1_825.40       0.9826          1.0007         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_296.03       572.22     1_868.25       0.9850          1.0006         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_296.03       580.37     1_876.40       0.9903          1.0003         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_296.03       622.45     1_918.48       0.9930          1.0003         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_296.03       796.52     2_092.55       0.9970          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_296.03       825.30     2_121.33       0.9999          1.0000         3.04
IVF-RaBitQ-nl316 (self)                                1_296.03     2_739.40     4_035.43       0.9999          1.0000         3.04
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
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        19.75    10_237.93    10_257.68       1.0000          1.0000        97.66
Exhaustive (self)                                         19.75    33_456.79    33_476.55       1.0000          1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           4_054.97     2_329.69     6_384.66       0.6297             NaN         5.23
ExhaustiveRaBitQ-rf5 (query)                           4_054.97     2_398.18     6_453.15       0.9763          1.0003         5.23
ExhaustiveRaBitQ-rf10 (query)                          4_054.97     2_463.20     6_518.17       0.9979          1.0000         5.23
ExhaustiveRaBitQ-rf20 (query)                          4_054.97     2_577.09     6_632.07       0.9999          1.0000         5.23
ExhaustiveRaBitQ (self)                                4_054.97     8_231.55    12_286.52       0.9979          1.0000         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_472.25       759.73     6_231.99       0.6167             NaN         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_472.25     1_239.12     6_711.37       0.6289             NaN         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_472.25     1_707.20     7_179.46       0.6304             NaN         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_472.25       871.40     6_343.66       0.9462          1.0021         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_472.25       960.04     6_432.30       0.9476          1.0021         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_472.25     1_350.80     6_823.05       0.9901          1.0003         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_472.25     1_455.64     6_927.90       0.9919          1.0003         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_472.25     1_833.08     7_305.33       0.9980          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_472.25     1_961.31     7_433.56       0.9999          1.0000         5.32
IVF-RaBitQ-nl158 (self)                                5_472.25     6_470.43    11_942.69       0.9999          1.0000         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_327.88     1_051.25     4_379.13       0.6279             NaN         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_327.88     1_314.72     4_642.61       0.6306             NaN         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_327.88     1_937.92     5_265.81       0.6308             NaN         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_327.88     1_162.92     4_490.80       0.9852          1.0005         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_327.88     1_264.47     4_592.36       0.9869          1.0004         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_327.88     1_431.46     4_759.35       0.9970          1.0001         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_327.88     1_543.67     4_871.56       0.9989          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_327.88     2_066.16     5_394.05       0.9979          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_327.88     2_172.07     5_499.96       0.9999          1.0000         5.44
IVF-RaBitQ-nl223 (self)                                3_327.88     7_210.28    10_538.17       0.9999          1.0000         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      3_642.73     1_355.62     4_998.34       0.6304             NaN         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      3_642.73     1_522.45     5_165.18       0.6314             NaN         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      3_642.73     2_213.94     5_856.67       0.6316             NaN         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     3_642.73     1_472.14     5_114.87       0.9911          1.0003         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     3_642.73     1_570.07     5_212.80       0.9926          1.0002         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     3_642.73     1_639.02     5_281.75       0.9966          1.0001         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     3_642.73     1_742.99     5_385.72       0.9982          1.0001         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     3_642.73     2_340.35     5_983.08       0.9982          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     3_642.73     2_448.16     6_090.89       0.9999          1.0000         5.63
IVF-RaBitQ-nl316 (self)                                3_642.73     8_132.27    11_775.00       0.9999          1.0000         5.63
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
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        40.29    22_074.92    22_115.21       1.0000          1.0000       195.31
Exhaustive (self)                                         40.29    73_326.30    73_366.59       1.0000          1.0000       195.31
ExhaustiveRaBitQ-rf0 (query)                          14_332.60     9_511.74    23_844.34       0.0387             NaN        11.50
ExhaustiveRaBitQ-rf5 (query)                          14_332.60     9_493.82    23_826.42       0.1120          1.1672        11.50
ExhaustiveRaBitQ-rf10 (query)                         14_332.60     9_364.17    23_696.77       0.1905          1.1377        11.50
ExhaustiveRaBitQ-rf20 (query)                         14_332.60     9_499.77    23_832.37       0.3240          1.1093        11.50
ExhaustiveRaBitQ (self)                               14_332.60    31_062.63    45_395.23       0.1897          1.1378        11.50
IVF-RaBitQ-nl158-np7-rf0 (query)                      17_733.53     2_992.25    20_725.78       0.0382             NaN        11.68
IVF-RaBitQ-nl158-np12-rf0 (query)                     17_733.53     5_027.27    22_760.79       0.0380             NaN        11.68
IVF-RaBitQ-nl158-np17-rf0 (query)                     17_733.53     7_078.24    24_811.76       0.0378             NaN        11.68
IVF-RaBitQ-nl158-np7-rf10 (query)                     17_733.53     3_058.82    20_792.34       0.1949          1.1347        11.68
IVF-RaBitQ-nl158-np7-rf20 (query)                     17_733.53     3_208.96    20_942.49       0.3334          1.1053        11.68
IVF-RaBitQ-nl158-np12-rf10 (query)                    17_733.53     5_225.65    22_959.17       0.1894          1.1370        11.68
IVF-RaBitQ-nl158-np12-rf20 (query)                    17_733.53     5_134.29    22_867.81       0.3252          1.1080        11.68
IVF-RaBitQ-nl158-np17-rf10 (query)                    17_733.53     6_987.07    24_720.60       0.1872          1.1378        11.68
IVF-RaBitQ-nl158-np17-rf20 (query)                    17_733.53     7_112.65    24_846.18       0.3222          1.1088        11.68
IVF-RaBitQ-nl158 (self)                               17_733.53    23_673.60    41_407.13       0.3198          1.1090        11.68
IVF-RaBitQ-nl223-np11-rf0 (query)                     12_514.38     4_408.59    16_922.98       0.0372             NaN        11.93
IVF-RaBitQ-nl223-np14-rf0 (query)                     12_514.38     5_583.49    18_097.87       0.0372             NaN        11.93
IVF-RaBitQ-nl223-np21-rf0 (query)                     12_514.38     8_274.40    20_788.78       0.0371             NaN        11.93
IVF-RaBitQ-nl223-np11-rf10 (query)                    12_514.38     4_431.14    16_945.52       0.1864          1.1366        11.93
IVF-RaBitQ-nl223-np11-rf20 (query)                    12_514.38     4_569.79    17_084.17       0.3251          1.1068        11.93
IVF-RaBitQ-nl223-np14-rf10 (query)                    12_514.38     5_549.21    18_063.59       0.1836          1.1377        11.93
IVF-RaBitQ-nl223-np14-rf20 (query)                    12_514.38     5_706.50    18_220.89       0.3207          1.1080        11.93
IVF-RaBitQ-nl223-np21-rf10 (query)                    12_514.38     8_200.99    20_715.38       0.1831          1.1380        11.93
IVF-RaBitQ-nl223-np21-rf20 (query)                    12_514.38     8_344.98    20_859.36       0.3198          1.1081        11.93
IVF-RaBitQ-nl223 (self)                               12_514.38    27_849.51    40_363.89       0.3170          1.1084        11.93
IVF-RaBitQ-nl316-np15-rf0 (query)                     13_192.84     5_970.48    19_163.32       0.0360             NaN        12.30
IVF-RaBitQ-nl316-np17-rf0 (query)                     13_192.84     6_736.87    19_929.71       0.0359             NaN        12.30
IVF-RaBitQ-nl316-np25-rf0 (query)                     13_192.84     9_836.83    23_029.67       0.0357             NaN        12.30
IVF-RaBitQ-nl316-np15-rf10 (query)                    13_192.84     5_991.69    19_184.52       0.1813          1.1373        12.30
IVF-RaBitQ-nl316-np15-rf20 (query)                    13_192.84     6_138.08    19_330.92       0.3201          1.1069        12.30
IVF-RaBitQ-nl316-np17-rf10 (query)                    13_192.84     6_737.90    19_930.74       0.1800          1.1378        12.30
IVF-RaBitQ-nl316-np17-rf20 (query)                    13_192.84     6_918.42    20_111.25       0.3176          1.1075        12.30
IVF-RaBitQ-nl316-np25-rf10 (query)                    13_192.84     9_738.62    22_931.46       0.1789          1.1382        12.30
IVF-RaBitQ-nl316-np25-rf20 (query)                    13_192.84     9_908.14    23_100.98       0.3161          1.1078        12.30
IVF-RaBitQ-nl316 (self)                               13_192.84    33_827.41    47_020.25       0.3142          1.1079        12.30
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
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        10.07     4_251.64     4_261.72       1.0000          1.0000        48.83
Exhaustive (self)                                         10.07    14_331.38    14_341.45       1.0000          1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                           1_424.08       780.14     2_204.22       0.7462             NaN         2.84
ExhaustiveRaBitQ-rf5 (query)                           1_424.08       849.38     2_273.45       0.9986          1.0000         2.84
ExhaustiveRaBitQ-rf10 (query)                          1_424.08       899.53     2_323.61       1.0000          1.0000         2.84
ExhaustiveRaBitQ-rf20 (query)                          1_424.08       987.08     2_411.16       1.0000          1.0000         2.84
ExhaustiveRaBitQ (self)                                1_424.08     2_993.57     4_417.64       1.0000          1.0000         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       2_102.45       235.34     2_337.79       0.7478             NaN         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      2_102.45       382.88     2_485.33       0.7495             NaN         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      2_102.45       529.07     2_631.51       0.7495             NaN         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      2_102.45       320.67     2_423.12       0.9930          1.0006         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      2_102.45       390.78     2_493.22       0.9930          1.0006         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     2_102.45       474.60     2_577.05       1.0000          1.0000         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     2_102.45       550.96     2_653.41       1.0000          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     2_102.45       632.28     2_734.72       1.0000          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     2_102.45       714.97     2_817.41       1.0000          1.0000         2.89
IVF-RaBitQ-nl158 (self)                                2_102.45     2_397.97     4_500.42       1.0000          1.0000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_076.95       326.53     1_403.47       0.7499             NaN         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_076.95       399.64     1_476.59       0.7507             NaN         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_076.95       596.75     1_673.69       0.7507             NaN         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_076.95       409.79     1_486.74       0.9969          1.0002         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_076.95       482.88     1_559.83       0.9969          1.0002         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_076.95       498.65     1_575.59       0.9999          1.0000         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_076.95       577.17     1_654.11       0.9999          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_076.95       688.64     1_765.58       1.0000          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_076.95       775.47     1_852.42       1.0000          1.0000         2.95
IVF-RaBitQ-nl223 (self)                                1_076.95     2_579.39     3_656.34       1.0000          1.0000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_255.90       397.13     1_653.04       0.7539             NaN         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_255.90       446.82     1_702.72       0.7542             NaN         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_255.90       645.98     1_901.88       0.7542             NaN         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_255.90       490.57     1_746.48       0.9985          1.0001         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_255.90       566.19     1_822.10       0.9985          1.0001         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_255.90       540.04     1_795.94       0.9997          1.0000         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_255.90       618.96     1_874.86       0.9997          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_255.90       742.03     1_997.94       1.0000          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_255.90       823.06     2_078.96       1.0000          1.0000         3.04
IVF-RaBitQ-nl316 (self)                                1_255.90     2_757.72     4_013.63       1.0000          1.0000         3.04
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
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        20.19     9_574.45     9_594.64       1.0000          1.0000        97.66
Exhaustive (self)                                         20.19    32_269.38    32_289.57       1.0000          1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           4_010.85     2_374.57     6_385.42       0.7550             NaN         5.23
ExhaustiveRaBitQ-rf5 (query)                           4_010.85     2_435.59     6_446.44       0.9989          1.0000         5.23
ExhaustiveRaBitQ-rf10 (query)                          4_010.85     2_542.10     6_552.95       0.9998          1.0000         5.23
ExhaustiveRaBitQ-rf20 (query)                          4_010.85     2_610.91     6_621.76       0.9998          1.0000         5.23
ExhaustiveRaBitQ (self)                                4_010.85     8_489.25    12_500.10       0.9998          1.0000         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_451.94       710.95     6_162.89       0.7525             NaN         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_451.94     1_197.22     6_649.16       0.7571             NaN         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_451.94     1_656.51     7_108.45       0.7571             NaN         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_451.94       822.51     6_274.45       0.9861          1.0009         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_451.94       987.47     6_439.41       0.9861          1.0009         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_451.94     1_348.04     6_799.98       0.9998          1.0000         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_451.94     1_407.13     6_859.07       0.9998          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_451.94     1_780.78     7_232.72       0.9998          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_451.94     1_883.03     7_334.97       0.9998          1.0000         5.32
IVF-RaBitQ-nl158 (self)                                5_451.94     6_279.28    11_731.22       0.9998          1.0000         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_202.02     1_062.82     4_264.84       0.7545             NaN         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_202.02     1_346.28     4_548.30       0.7558             NaN         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_202.02     1_988.49     5_190.52       0.7560             NaN         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_202.02     1_183.73     4_385.75       0.9956          1.0003         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_202.02     1_265.52     4_467.54       0.9956          1.0003         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_202.02     1_467.84     4_669.86       0.9992          1.0000         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_202.02     1_555.73     4_757.76       0.9992          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_202.02     2_111.39     5_313.42       0.9999          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_202.02     2_199.62     5_401.65       0.9999          1.0000         5.44
IVF-RaBitQ-nl223 (self)                                3_202.02     7_307.13    10_509.15       0.9998          1.0000         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      3_587.89     1_437.95     5_025.85       0.7579             NaN         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      3_587.89     1_638.35     5_226.25       0.7586             NaN         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      3_587.89     2_341.00     5_928.89       0.7588             NaN         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     3_587.89     1_536.38     5_124.27       0.9969          1.0002         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     3_587.89     1_618.46     5_206.36       0.9969          1.0002         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     3_587.89     1_717.60     5_305.49       0.9993          1.0000         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     3_587.89     1_796.34     5_384.24       0.9993          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     3_587.89     2_440.53     6_028.42       0.9999          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     3_587.89     2_537.23     6_125.12       0.9999          1.0000         5.63
IVF-RaBitQ-nl316 (self)                                3_587.89     8_551.44    12_139.34       0.9998          1.0000         5.63
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
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        39.89    22_052.92    22_092.81       1.0000          1.0000       195.31
Exhaustive (self)                                         39.89    73_913.05    73_952.94       1.0000          1.0000       195.31
ExhaustiveRaBitQ-rf0 (query)                          14_498.31     9_211.19    23_709.50       0.0314             NaN        11.50
ExhaustiveRaBitQ-rf5 (query)                          14_498.31     9_094.68    23_592.99       0.0996          1.2429        11.50
ExhaustiveRaBitQ-rf10 (query)                         14_498.31     9_140.81    23_639.12       0.1751          1.1900        11.50
ExhaustiveRaBitQ-rf20 (query)                         14_498.31     9_298.76    23_797.07       0.3082          1.1415        11.50
ExhaustiveRaBitQ (self)                               14_498.31    30_393.82    44_892.13       0.1750          1.1903        11.50
IVF-RaBitQ-nl158-np7-rf0 (query)                      17_964.52     2_890.00    20_854.52       0.0306             NaN        11.68
IVF-RaBitQ-nl158-np12-rf0 (query)                     17_964.52     4_908.67    22_873.20       0.0304             NaN        11.68
IVF-RaBitQ-nl158-np17-rf0 (query)                     17_964.52     6_890.42    24_854.94       0.0304             NaN        11.68
IVF-RaBitQ-nl158-np7-rf10 (query)                     17_964.52     2_961.93    20_926.45       0.1738          1.1891        11.68
IVF-RaBitQ-nl158-np7-rf20 (query)                     17_964.52     3_116.95    21_081.47       0.3121          1.1391        11.68
IVF-RaBitQ-nl158-np12-rf10 (query)                    17_964.52     4_902.49    22_867.02       0.1705          1.1912        11.68
IVF-RaBitQ-nl158-np12-rf20 (query)                    17_964.52     5_052.03    23_016.55       0.3060          1.1411        11.68
IVF-RaBitQ-nl158-np17-rf10 (query)                    17_964.52     6_860.03    24_824.55       0.1705          1.1912        11.68
IVF-RaBitQ-nl158-np17-rf20 (query)                    17_964.52     6_979.00    24_943.52       0.3060          1.1411        11.68
IVF-RaBitQ-nl158 (self)                               17_964.52    23_240.47    41_204.99       0.3062          1.1412        11.68
IVF-RaBitQ-nl223-np11-rf0 (query)                     12_675.87     4_433.07    17_108.94       0.0300             NaN        11.93
IVF-RaBitQ-nl223-np14-rf0 (query)                     12_675.87     5_577.16    18_253.03       0.0298             NaN        11.93
IVF-RaBitQ-nl223-np21-rf0 (query)                     12_675.87     8_301.18    20_977.05       0.0298             NaN        11.93
IVF-RaBitQ-nl223-np11-rf10 (query)                    12_675.87     4_450.73    17_126.60       0.1703          1.1895        11.93
IVF-RaBitQ-nl223-np11-rf20 (query)                    12_675.87     4_594.91    17_270.78       0.3092          1.1387        11.93
IVF-RaBitQ-nl223-np14-rf10 (query)                    12_675.87     5_551.02    18_226.89       0.1682          1.1907        11.93
IVF-RaBitQ-nl223-np14-rf20 (query)                    12_675.87     5_699.15    18_375.02       0.3052          1.1400        11.93
IVF-RaBitQ-nl223-np21-rf10 (query)                    12_675.87     8_297.39    20_973.26       0.1681          1.1908        11.93
IVF-RaBitQ-nl223-np21-rf20 (query)                    12_675.87     8_371.06    21_046.93       0.3049          1.1401        11.93
IVF-RaBitQ-nl223 (self)                               12_675.87    27_894.58    40_570.45       0.3044          1.1403        11.93
IVF-RaBitQ-nl316-np15-rf0 (query)                     13_494.99     5_854.82    19_349.81       0.0281             NaN        12.30
IVF-RaBitQ-nl316-np17-rf0 (query)                     13_494.99     6_693.18    20_188.17       0.0280             NaN        12.30
IVF-RaBitQ-nl316-np25-rf0 (query)                     13_494.99     9_713.08    23_208.07       0.0279             NaN        12.30
IVF-RaBitQ-nl316-np15-rf10 (query)                    13_494.99     5_979.96    19_474.95       0.1656          1.1901        12.30
IVF-RaBitQ-nl316-np15-rf20 (query)                    13_494.99     6_090.38    19_585.37       0.3074          1.1378        12.30
IVF-RaBitQ-nl316-np17-rf10 (query)                    13_494.99     6_693.35    20_188.34       0.1642          1.1908        12.30
IVF-RaBitQ-nl316-np17-rf20 (query)                    13_494.99     6_821.52    20_316.51       0.3048          1.1386        12.30
IVF-RaBitQ-nl316-np25-rf10 (query)                    13_494.99     9_717.81    23_212.80       0.1634          1.1912        12.30
IVF-RaBitQ-nl316-np25-rf20 (query)                    13_494.99     9_847.44    23_342.43       0.3034          1.1390        12.30
IVF-RaBitQ-nl316 (self)                               13_494.99    32_793.00    46_287.99       0.3025          1.1393        12.30
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
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                         6.62     4_203.90     4_210.52       1.0000          1.0000        48.83
Exhaustive (self)                                          6.62    14_079.50    14_086.12       1.0000          1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                           1_183.71     1_067.55     2_251.27       0.3458             NaN         2.84
ExhaustiveRaBitQ-rf5 (query)                           1_183.71     1_137.85     2_321.57       0.7547          1.0011         2.84
ExhaustiveRaBitQ-rf10 (query)                          1_183.71     1_197.18     2_380.90       0.8884          1.0004         2.84
ExhaustiveRaBitQ-rf20 (query)                          1_183.71     1_294.73     2_478.44       0.9630          1.0001         2.84
ExhaustiveRaBitQ (self)                                1_183.71     3_986.93     5_170.64       1.0000          1.0000         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_792.99       309.48     2_102.47       0.3617             NaN         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_792.99       526.17     2_319.16       0.3614             NaN         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_792.99       752.21     2_545.19       0.3612             NaN         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_792.99       398.90     2_191.89       0.9035          1.0003         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_792.99       463.27     2_256.26       0.9688          1.0001         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_792.99       631.96     2_424.94       0.9056          1.0003         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_792.99       709.81     2_502.80       0.9718          1.0001         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_792.99       873.33     2_666.32       0.9059          1.0003         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_792.99       965.05     2_758.04       0.9722          1.0001         2.89
IVF-RaBitQ-nl158 (self)                                1_792.99     3_197.69     4_990.67       1.0000          1.0000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                        664.33       335.17       999.50       0.4010             NaN         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                        664.33       404.17     1_068.50       0.4008             NaN         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                        664.33       604.87     1_269.20       0.4004             NaN         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                       664.33       405.15     1_069.48       0.9356          1.0002         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                       664.33       467.61     1_131.93       0.9845          1.0000         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                       664.33       491.89     1_156.21       0.9359          1.0002         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                       664.33       560.80     1_225.13       0.9850          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                       664.33       695.51     1_359.84       0.9360          1.0002         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                       664.33       771.52     1_435.85       0.9854          1.0000         2.95
IVF-RaBitQ-nl223 (self)                                  664.33     2_535.74     3_200.07       1.0000          1.0000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                        694.17       399.61     1_093.78       0.4148             NaN         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                        694.17       446.08     1_140.25       0.4147             NaN         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                        694.17       645.51     1_339.68       0.4144             NaN         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                       694.17       482.67     1_176.84       0.9437          1.0002         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                       694.17       546.17     1_240.34       0.9877          1.0000         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                       694.17       531.90     1_226.07       0.9436          1.0002         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                       694.17       601.42     1_295.59       0.9878          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                       694.17       736.86     1_431.04       0.9435          1.0002         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                       694.17       810.37     1_504.55       0.9881          1.0000         3.04
IVF-RaBitQ-nl316 (self)                                  694.17     2_665.70     3_359.87       1.0000          1.0000         3.04
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
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        19.17     9_556.00     9_575.18       1.0000          1.0000        97.66
Exhaustive (self)                                         19.17    32_159.52    32_178.70       1.0000          1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           3_497.77     2_928.01     6_425.78       0.3332             NaN         5.23
ExhaustiveRaBitQ-rf5 (query)                           3_497.77     3_007.88     6_505.65       0.7285          1.0006         5.23
ExhaustiveRaBitQ-rf10 (query)                          3_497.77     3_128.84     6_626.60       0.8641          1.0002         5.23
ExhaustiveRaBitQ-rf20 (query)                          3_497.77     3_229.35     6_727.12       0.9459          1.0001         5.23
ExhaustiveRaBitQ (self)                                3_497.77    10_472.51    13_970.28       0.9996          1.0001         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       4_779.36       762.79     5_542.15       0.3606             NaN         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      4_779.36     1_298.33     6_077.69       0.3591             NaN         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      4_779.36     1_850.07     6_629.42       0.3586             NaN         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      4_779.36       887.22     5_666.57       0.8886          1.0002         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      4_779.36       974.77     5_754.13       0.9556          1.0001         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     4_779.36     1_417.91     6_197.26       0.8908          1.0002         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     4_779.36     1_516.09     6_295.45       0.9606          1.0001         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     4_779.36     1_962.50     6_741.85       0.8904          1.0002         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     4_779.36     2_093.52     6_872.88       0.9609          1.0001         5.32
IVF-RaBitQ-nl158 (self)                                4_779.36     6_886.53    11_665.88       0.9998          1.0000         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      2_447.40     1_024.92     3_472.32       0.3848             NaN         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      2_447.40     1_286.76     3_734.16       0.3842             NaN         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      2_447.40     1_904.32     4_351.72       0.3833             NaN         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     2_447.40     1_131.14     3_578.54       0.9139          1.0001         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     2_447.40     1_229.84     3_677.24       0.9706          1.0000         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     2_447.40     1_414.12     3_861.52       0.9148          1.0001         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     2_447.40     1_500.44     3_947.84       0.9729          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     2_447.40     2_026.96     4_474.36       0.9148          1.0001         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     2_447.40     2_127.73     4_575.13       0.9743          1.0000         5.44
IVF-RaBitQ-nl223 (self)                                2_447.40     7_043.26     9_490.66       0.9999          1.0000         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      2_428.87     1_333.87     3_762.74       0.3961             NaN         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      2_428.87     1_505.33     3_934.20       0.3958             NaN         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      2_428.87     2_179.90     4_608.77       0.3948             NaN         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     2_428.87     1_445.62     3_874.49       0.9223          1.0001         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     2_428.87     1_541.27     3_970.14       0.9745          1.0000         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     2_428.87     1_613.23     4_042.10       0.9227          1.0001         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     2_428.87     1_718.03     4_146.90       0.9756          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     2_428.87     2_332.03     4_760.90       0.9232          1.0001         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     2_428.87     2_414.17     4_843.04       0.9775          1.0000         5.63
IVF-RaBitQ-nl316 (self)                                2_428.87     8_118.17    10_547.04       0.9999          1.0000         5.63
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
--------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        39.55    22_044.03    22_083.58       1.0000          1.0000       195.31
Exhaustive (self)                                         39.55    73_213.48    73_253.02       1.0000          1.0000       195.31
ExhaustiveRaBitQ-rf0 (query)                          13_228.05    10_197.63    23_425.68       0.0237             NaN        11.50
ExhaustiveRaBitQ-rf5 (query)                          13_228.05     9_978.89    23_206.94       0.0587          1.0054        11.50
ExhaustiveRaBitQ-rf10 (query)                         13_228.05     9_985.18    23_213.23       0.0961          1.0042        11.50
ExhaustiveRaBitQ-rf20 (query)                         13_228.05    10_159.77    23_387.82       0.1705          1.0031        11.50
ExhaustiveRaBitQ (self)                               13_228.05    33_394.30    46_622.35       0.2337          1.5506        11.50
IVF-RaBitQ-nl158-np7-rf0 (query)                      16_306.07     2_959.82    19_265.89       0.0263             NaN        11.68
IVF-RaBitQ-nl158-np12-rf0 (query)                     16_306.07     5_048.79    21_354.86       0.0260             NaN        11.68
IVF-RaBitQ-nl158-np17-rf0 (query)                     16_306.07     7_108.24    23_414.31       0.0259             NaN        11.68
IVF-RaBitQ-nl158-np7-rf10 (query)                     16_306.07     3_033.37    19_339.44       0.1106          1.0031        11.68
IVF-RaBitQ-nl158-np7-rf20 (query)                     16_306.07     3_173.74    19_479.82       0.2013          1.0022        11.68
IVF-RaBitQ-nl158-np12-rf10 (query)                    16_306.07     5_065.63    21_371.70       0.1051          1.0034        11.68
IVF-RaBitQ-nl158-np12-rf20 (query)                    16_306.07     5_212.78    21_518.85       0.1874          1.0024        11.68
IVF-RaBitQ-nl158-np17-rf10 (query)                    16_306.07     7_111.58    23_417.65       0.1030          1.0035        11.68
IVF-RaBitQ-nl158-np17-rf20 (query)                    16_306.07     7_254.53    23_560.60       0.1822          1.0026        11.68
IVF-RaBitQ-nl158 (self)                               16_306.07    24_137.69    40_443.76       0.4799          1.2475        11.68
IVF-RaBitQ-nl223-np11-rf0 (query)                     10_713.09     4_429.14    15_142.23       0.0277             NaN        11.93
IVF-RaBitQ-nl223-np14-rf0 (query)                     10_713.09     5_598.02    16_311.12       0.0275             NaN        11.93
IVF-RaBitQ-nl223-np21-rf0 (query)                     10_713.09     8_377.73    19_090.82       0.0272             NaN        11.93
IVF-RaBitQ-nl223-np11-rf10 (query)                    10_713.09     4_462.80    15_175.89       0.1148          1.0029        11.93
IVF-RaBitQ-nl223-np11-rf20 (query)                    10_713.09     4_635.85    15_348.94       0.2075          1.0021        11.93
IVF-RaBitQ-nl223-np14-rf10 (query)                    10_713.09     5_612.78    16_325.87       0.1120          1.0031        11.93
IVF-RaBitQ-nl223-np14-rf20 (query)                    10_713.09     5_758.50    16_471.59       0.2007          1.0022        11.93
IVF-RaBitQ-nl223-np21-rf10 (query)                    10_713.09     8_333.68    19_046.77       0.1083          1.0032        11.93
IVF-RaBitQ-nl223-np21-rf20 (query)                    10_713.09     8_531.06    19_244.15       0.1922          1.0023        11.93
IVF-RaBitQ-nl223 (self)                               10_713.09    29_342.37    40_055.47       0.5665          1.1700        11.93
IVF-RaBitQ-nl316-np15-rf0 (query)                     10_776.76     5_945.23    16_721.99       0.0289             NaN        12.30
IVF-RaBitQ-nl316-np17-rf0 (query)                     10_776.76     6_709.11    17_485.87       0.0287             NaN        12.30
IVF-RaBitQ-nl316-np25-rf0 (query)                     10_776.76     9_792.50    20_569.26       0.0284             NaN        12.30
IVF-RaBitQ-nl316-np15-rf10 (query)                    10_776.76     6_022.18    16_798.94       0.1208          1.0028        12.30
IVF-RaBitQ-nl316-np15-rf20 (query)                    10_776.76     5_996.27    16_773.03       0.2187          1.0020        12.30
IVF-RaBitQ-nl316-np17-rf10 (query)                    10_776.76     6_585.21    17_361.97       0.1191          1.0029        12.30
IVF-RaBitQ-nl316-np17-rf20 (query)                    10_776.76     6_727.31    17_504.07       0.2145          1.0020        12.30
IVF-RaBitQ-nl316-np25-rf10 (query)                    10_776.76    10_302.03    21_078.79       0.1148          1.0030        12.30
IVF-RaBitQ-nl316-np25-rf20 (query)                    10_776.76     9_706.30    20_483.06       0.2047          1.0022        12.30
IVF-RaBitQ-nl316 (self)                               10_776.76    32_220.00    42_996.76       0.6266          1.1310        12.30
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

Overall, this is a fantastic binary index that massively compresses the data,
while still allowing for great Recalls. If you need to compress your data
and reduce memory fingerprint, please, use RaBitQ!

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
