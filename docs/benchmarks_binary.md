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
- [TurboQuantisation](#)

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
Exhaustive (query)                                         9.85     4_126.17     4_136.03       1.0000          1.0000        48.83
Exhaustive (self)                                          9.85    13_889.72    13_899.58       1.0000          1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_569.68       248.97     2_818.65       0.1006             NaN         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_569.68       357.19     2_926.87       0.2890          1.0907         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_569.68       447.41     3_017.09       0.4246          1.0554         1.78
ExhaustiveBinary-256-random (self)                     2_569.68     1_156.59     3_726.27       0.2886          1.0907         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_903.20       254.86     3_158.06       0.2261             NaN         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_903.20       362.57     3_265.77       0.5963          1.0281         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_903.20       462.05     3_365.25       0.7219          1.0155         1.78
ExhaustiveBinary-256-pca (self)                        2_903.20     1_209.57     4_112.77       0.5956          1.0283         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_060.28       444.14     5_504.42       0.1249             NaN         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_060.28       546.48     5_606.76       0.3718          1.0678         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_060.28       649.94     5_710.22       0.5258          1.0390         3.55
ExhaustiveBinary-512-random (self)                     5_060.28     1_818.41     6_878.69       0.3724          1.0679         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_529.82       456.08     5_985.90       0.2472             NaN         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_529.82       566.18     6_096.00       0.6835          1.0189         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_529.82       668.44     6_198.26       0.8321          1.0077         3.55
ExhaustiveBinary-512-pca (self)                        5_529.82     1_872.18     7_402.00       0.6838          1.0189         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_045.77       764.70    10_810.47       0.1692             NaN         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_045.77       876.97    10_922.75       0.5028          1.0425         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_045.77       984.10    11_029.87       0.6675          1.0220         7.10
ExhaustiveBinary-1024-random (self)                   10_045.77     2_868.16    12_913.93       0.5038          1.0426         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_679.74       780.60    11_460.34       0.2755             NaN         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_679.74       882.26    11_562.00       0.7370          1.0141         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_679.74     1_005.59    11_685.34       0.8733          1.0053         7.10
ExhaustiveBinary-1024-pca (self)                      10_679.74     2_946.51    13_626.25       0.7371          1.0141         7.10
ExhaustiveBinary-256-signed_no_rr (query)              2_550.42       247.73     2_798.15       0.1006             NaN         1.78
ExhaustiveBinary-256-signed-rf10 (query)               2_550.42       351.29     2_901.71       0.2890          1.0907         1.78
ExhaustiveBinary-256-signed-rf20 (query)               2_550.42       456.72     3_007.15       0.4246          1.0554         1.78
ExhaustiveBinary-256-signed (self)                     2_550.42     1_163.23     3_713.65       0.2886          1.0907         1.78
IVF-Binary-256-nl158-np7-rf0-random (query)            4_121.14       113.87     4_235.01       0.1063             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           4_121.14       119.93     4_241.06       0.1034             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           4_121.14       124.50     4_245.64       0.1017             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           4_121.14       175.31     4_296.44       0.3128          1.0834         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           4_121.14       223.69     4_344.83       0.4564          1.0499         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          4_121.14       181.72     4_302.85       0.2990          1.0878         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          4_121.14       237.91     4_359.04       0.4375          1.0533         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          4_121.14       192.28     4_313.42       0.2914          1.0901         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          4_121.14       254.23     4_375.37       0.4284          1.0549         1.93
IVF-Binary-256-nl158-random (self)                     4_121.14       563.98     4_685.11       0.2986          1.0878         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           3_117.33       117.93     3_235.27       0.1054             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           3_117.33       120.77     3_238.10       0.1027             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           3_117.33       126.77     3_244.11       0.1013             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          3_117.33       181.42     3_298.76       0.3066          1.0843         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          3_117.33       230.66     3_347.99       0.4485          1.0506         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          3_117.33       183.04     3_300.37       0.2976          1.0874         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          3_117.33       238.05     3_355.38       0.4366          1.0529         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          3_117.33       192.60     3_309.93       0.2925          1.0893         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          3_117.33       251.98     3_369.31       0.4296          1.0544         2.00
IVF-Binary-256-nl223-random (self)                     3_117.33       560.76     3_678.10       0.2974          1.0875         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           3_284.94       125.71     3_410.65       0.1048             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           3_284.94       128.01     3_412.95       0.1036             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           3_284.94       131.16     3_416.10       0.1019             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          3_284.94       184.49     3_469.43       0.3062          1.0846         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          3_284.94       238.85     3_523.79       0.4494          1.0505         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          3_284.94       186.55     3_471.49       0.3014          1.0861         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          3_284.94       240.49     3_525.42       0.4427          1.0519         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          3_284.94       193.12     3_478.05       0.2928          1.0893         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          3_284.94       253.53     3_538.47       0.4300          1.0544         2.09
IVF-Binary-256-nl316-random (self)                     3_284.94       577.58     3_862.51       0.3014          1.0861         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               4_404.06       124.79     4_528.86       0.2338             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              4_404.06       127.32     4_531.38       0.2321             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              4_404.06       128.97     4_533.03       0.2311             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              4_404.06       184.50     4_588.56       0.6495          1.0223         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              4_404.06       238.42     4_642.48       0.7928          1.0106         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             4_404.06       191.31     4_595.38       0.6501          1.0222         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             4_404.06       252.12     4_656.18       0.7994          1.0099         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             4_404.06       201.53     4_605.59       0.6462          1.0226         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             4_404.06       267.53     4_671.60       0.7949          1.0102         1.93
IVF-Binary-256-nl158-pca (self)                        4_404.06       599.54     5_003.61       0.6500          1.0223         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              3_472.00       121.96     3_593.96       0.2328             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              3_472.00       124.47     3_596.47       0.2314             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              3_472.00       133.64     3_605.64       0.2305             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             3_472.00       189.02     3_661.02       0.6533          1.0219         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             3_472.00       261.58     3_733.58       0.8035          1.0097         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             3_472.00       197.02     3_669.02       0.6507          1.0221         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             3_472.00       254.51     3_726.51       0.8016          1.0098         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             3_472.00       202.68     3_674.67       0.6465          1.0225         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             3_472.00       266.95     3_738.95       0.7959          1.0101         2.00
IVF-Binary-256-nl223-pca (self)                        3_472.00       604.19     4_076.18       0.6509          1.0222         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              3_618.17       130.11     3_748.28       0.2329             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              3_618.17       129.44     3_747.61       0.2323             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              3_618.17       134.51     3_752.67       0.2309             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             3_618.17       195.00     3_813.16       0.6559          1.0216         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             3_618.17       251.85     3_870.02       0.8063          1.0095         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             3_618.17       197.13     3_815.30       0.6542          1.0218         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             3_618.17       256.44     3_874.61       0.8054          1.0095         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             3_618.17       202.70     3_820.87       0.6484          1.0224         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             3_618.17       264.29     3_882.46       0.7987          1.0100         2.09
IVF-Binary-256-nl316-pca (self)                        3_618.17       613.76     4_231.93       0.6541          1.0219         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_550.07       209.58     6_759.65       0.1303             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_550.07       213.77     6_763.84       0.1275             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_550.07       222.21     6_772.28       0.1259             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_550.07       267.56     6_817.63       0.3902          1.0636         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_550.07       323.43     6_873.50       0.5457          1.0363         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_550.07       278.36     6_828.43       0.3795          1.0661         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_550.07       336.59     6_886.66       0.5338          1.0380         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_550.07       292.96     6_843.03       0.3742          1.0674         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_550.07       352.42     6_902.49       0.5273          1.0389         3.71
IVF-Binary-512-nl158-random (self)                     6_550.07       895.91     7_445.98       0.3794          1.0663         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_578.11       211.14     5_789.25       0.1288             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_578.11       215.69     5_793.80       0.1268             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_578.11       226.98     5_805.09       0.1257             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_578.11       274.89     5_853.00       0.3852          1.0641         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_578.11       326.84     5_904.94       0.5432          1.0363         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_578.11       277.72     5_855.83       0.3785          1.0659         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_578.11       333.56     5_911.67       0.5340          1.0377         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_578.11       309.63     5_887.74       0.3743          1.0671         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_578.11       357.81     5_935.92       0.5286          1.0386         3.77
IVF-Binary-512-nl223-random (self)                     5_578.11       880.82     6_458.93       0.3792          1.0660         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           5_718.24       228.95     5_947.19       0.1284             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           5_718.24       222.03     5_940.28       0.1276             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           5_718.24       229.27     5_947.51       0.1259             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          5_718.24       282.67     6_000.91       0.3866          1.0640         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          5_718.24       341.19     6_059.43       0.5453          1.0361         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          5_718.24       323.18     6_041.43       0.3826          1.0649         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          5_718.24       348.35     6_066.59       0.5398          1.0369         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          5_718.24       292.85     6_011.09       0.3750          1.0670         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          5_718.24       372.71     6_090.95       0.5290          1.0386         3.86
IVF-Binary-512-nl316-random (self)                     5_718.24       897.45     6_615.69       0.3829          1.0651         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               7_019.58       216.35     7_235.93       0.2494             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              7_019.58       224.78     7_244.37       0.2482             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              7_019.58       234.29     7_253.87       0.2474             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              7_019.58       282.55     7_302.13       0.6798          1.0193         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              7_019.58       336.93     7_356.51       0.8187          1.0089         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             7_019.58       293.13     7_312.71       0.6847          1.0187         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             7_019.58       354.90     7_374.48       0.8321          1.0078         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             7_019.58       327.41     7_346.99       0.6837          1.0188         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             7_019.58       373.74     7_393.33       0.8324          1.0077         3.71
IVF-Binary-512-nl158-pca (self)                        7_019.58       939.61     7_959.19       0.6856          1.0188         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              6_125.92       219.84     6_345.76       0.2491             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              6_125.92       223.14     6_349.06       0.2480             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              6_125.92       233.83     6_359.74       0.2473             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             6_125.92       289.90     6_415.82       0.6861          1.0186         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             6_125.92       342.64     6_468.56       0.8324          1.0078         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             6_125.92       295.97     6_421.89       0.6850          1.0187         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             6_125.92       360.43     6_486.34       0.8333          1.0077         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             6_125.92       315.24     6_441.16       0.6838          1.0188         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             6_125.92       370.55     6_496.47       0.8325          1.0077         3.77
IVF-Binary-512-nl223-pca (self)                        6_125.92       942.39     7_068.31       0.6855          1.0188         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              6_164.91       228.48     6_393.39       0.2494             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              6_164.91       228.80     6_393.71       0.2488             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              6_164.91       238.79     6_403.70       0.2477             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             6_164.91       298.12     6_463.03       0.6879          1.0184         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             6_164.91       350.67     6_515.58       0.8338          1.0077         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             6_164.91       296.57     6_461.48       0.6873          1.0185         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             6_164.91       366.38     6_531.29       0.8347          1.0076         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             6_164.91       303.69     6_468.61       0.6839          1.0188         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             6_164.91       367.97     6_532.88       0.8323          1.0078         3.86
IVF-Binary-512-nl316-pca (self)                        6_164.91       948.53     7_113.44       0.6875          1.0186         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_473.13       392.25    11_865.38       0.1737             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_473.13       411.71    11_884.85       0.1713             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_473.13       421.24    11_894.38       0.1698             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_473.13       461.05    11_934.18       0.5112          1.0412         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_473.13       513.62    11_986.75       0.6693          1.0219         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_473.13       476.98    11_950.11       0.5072          1.0419         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_473.13       538.70    12_011.83       0.6704          1.0217         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_473.13       495.34    11_968.47       0.5038          1.0424         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_473.13       561.44    12_034.58       0.6679          1.0220         7.26
IVF-Binary-1024-nl158-random (self)                   11_473.13     1_547.85    13_020.98       0.5082          1.0419         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_474.50       400.12    10_874.62       0.1725             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_474.50       404.46    10_878.97       0.1708             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_474.50       418.69    10_893.19       0.1699             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_474.50       493.12    10_967.62       0.5122          1.0407         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_474.50       523.10    10_997.60       0.6774          1.0208         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_474.50       474.13    10_948.63       0.5077          1.0416         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_474.50       529.30    11_003.81       0.6724          1.0214         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_474.50       492.39    10_966.89       0.5043          1.0422         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_474.50       559.20    11_033.70       0.6686          1.0219         7.32
IVF-Binary-1024-nl223-random (self)                   10_474.50     1_531.60    12_006.10       0.5085          1.0417         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         10_637.47       406.39    11_043.86       0.1721             NaN         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)         10_637.47       408.23    11_045.70       0.1715             NaN         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)         10_637.47       418.78    11_056.25       0.1701             NaN         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)        10_637.47       476.05    11_113.53       0.5134          1.0407         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)        10_637.47       530.97    11_168.44       0.6788          1.0207         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)        10_637.47       474.20    11_111.68       0.5106          1.0411         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)        10_637.47       537.96    11_175.44       0.6760          1.0211         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)        10_637.47       492.29    11_129.76       0.5048          1.0422         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)        10_637.47       553.07    11_190.54       0.6689          1.0218         7.41
IVF-Binary-1024-nl316-random (self)                   10_637.47     1_549.84    12_187.31       0.5114          1.0412         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)             12_149.48       410.40    12_559.88       0.2769             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            12_149.48       424.20    12_573.67       0.2764             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            12_149.48       436.67    12_586.15       0.2757             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            12_149.48       477.61    12_627.09       0.7271          1.0150         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            12_149.48       532.92    12_682.40       0.8507          1.0069         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           12_149.48       495.39    12_644.87       0.7373          1.0140         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           12_149.48       561.53    12_711.01       0.8718          1.0054         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           12_149.48       551.75    12_701.23       0.7374          1.0140         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           12_149.48       582.48    12_731.96       0.8736          1.0053         7.26
IVF-Binary-1024-nl158-pca (self)                      12_149.48     1_617.22    13_766.70       0.7376          1.0141         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            11_164.99       410.11    11_575.10       0.2772             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            11_164.99       415.60    11_580.59       0.2763             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            11_164.99       432.61    11_597.60       0.2756             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           11_164.99       480.99    11_645.98       0.7381          1.0140         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           11_164.99       535.78    11_700.77       0.8706          1.0055         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           11_164.99       486.39    11_651.38       0.7380          1.0140         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           11_164.99       551.91    11_716.90       0.8736          1.0053         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           11_164.99       515.48    11_680.47       0.7371          1.0141         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           11_164.99       571.77    11_736.76       0.8733          1.0053         7.32
IVF-Binary-1024-nl223-pca (self)                      11_164.99     1_586.83    12_751.82       0.7385          1.0140         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            11_409.41       417.42    11_826.83       0.2775             NaN         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            11_409.41       422.09    11_831.50       0.2770             NaN         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            11_409.41       431.20    11_840.61       0.2759             NaN         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           11_409.41       488.07    11_897.48       0.7391          1.0139         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           11_409.41       545.48    11_954.89       0.8717          1.0054         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           11_409.41       488.61    11_898.02       0.7394          1.0139         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           11_409.41       549.05    11_958.46       0.8740          1.0053         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           11_409.41       507.93    11_917.34       0.7375          1.0140         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           11_409.41       566.28    11_975.69       0.8737          1.0053         7.42
IVF-Binary-1024-nl316-pca (self)                      11_409.41     1_587.04    12_996.45       0.7395          1.0139         7.42
IVF-Binary-256-nl158-np7-rf0-signed (query)            4_051.19       112.80     4_163.98       0.1063             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-signed (query)           4_051.19       117.72     4_168.90       0.1034             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-signed (query)           4_051.19       124.11     4_175.29       0.1017             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-signed (query)           4_051.19       172.76     4_223.94       0.3128          1.0834         1.93
IVF-Binary-256-nl158-np7-rf20-signed (query)           4_051.19       223.31     4_274.49       0.4564          1.0499         1.93
IVF-Binary-256-nl158-np12-rf10-signed (query)          4_051.19       183.77     4_234.96       0.2990          1.0878         1.93
IVF-Binary-256-nl158-np12-rf20-signed (query)          4_051.19       236.30     4_287.48       0.4375          1.0533         1.93
IVF-Binary-256-nl158-np17-rf10-signed (query)          4_051.19       196.71     4_247.89       0.2914          1.0901         1.93
IVF-Binary-256-nl158-np17-rf20-signed (query)          4_051.19       251.82     4_303.00       0.4284          1.0549         1.93
IVF-Binary-256-nl158-signed (self)                     4_051.19       556.16     4_607.34       0.2986          1.0878         1.93
IVF-Binary-256-nl223-np11-rf0-signed (query)           3_093.51       117.19     3_210.70       0.1054             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-signed (query)           3_093.51       119.91     3_213.42       0.1027             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-signed (query)           3_093.51       125.24     3_218.74       0.1013             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-signed (query)          3_093.51       176.72     3_270.23       0.3066          1.0843         2.00
IVF-Binary-256-nl223-np11-rf20-signed (query)          3_093.51       237.87     3_331.38       0.4485          1.0506         2.00
IVF-Binary-256-nl223-np14-rf10-signed (query)          3_093.51       181.51     3_275.01       0.2976          1.0874         2.00
IVF-Binary-256-nl223-np14-rf20-signed (query)          3_093.51       235.37     3_328.88       0.4366          1.0529         2.00
IVF-Binary-256-nl223-np21-rf10-signed (query)          3_093.51       192.21     3_285.72       0.2925          1.0893         2.00
IVF-Binary-256-nl223-np21-rf20-signed (query)          3_093.51       251.49     3_344.99       0.4296          1.0544         2.00
IVF-Binary-256-nl223-signed (self)                     3_093.51       557.13     3_650.64       0.2974          1.0875         2.00
IVF-Binary-256-nl316-np15-rf0-signed (query)           3_254.73       123.73     3_378.46       0.1048             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-signed (query)           3_254.73       124.99     3_379.72       0.1036             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-signed (query)           3_254.73       130.59     3_385.32       0.1019             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-signed (query)          3_254.73       185.40     3_440.12       0.3062          1.0846         2.09
IVF-Binary-256-nl316-np15-rf20-signed (query)          3_254.73       237.08     3_491.81       0.4494          1.0505         2.09
IVF-Binary-256-nl316-np17-rf10-signed (query)          3_254.73       186.90     3_441.62       0.3014          1.0861         2.09
IVF-Binary-256-nl316-np17-rf20-signed (query)          3_254.73       240.04     3_494.76       0.4427          1.0519         2.09
IVF-Binary-256-nl316-np25-rf10-signed (query)          3_254.73       191.89     3_446.61       0.2928          1.0893         2.09
IVF-Binary-256-nl316-np25-rf20-signed (query)          3_254.73       247.89     3_502.62       0.4300          1.0544         2.09
IVF-Binary-256-nl316-signed (self)                     3_254.73       599.60     3_854.33       0.3014          1.0861         2.09
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
Exhaustive (query)                                        19.87     9_723.34     9_743.21       1.0000          1.0000        97.66
Exhaustive (self)                                         19.87    32_221.35    32_241.23       1.0000          1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_672.84       353.20     6_026.04       0.0893             NaN         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_672.84       478.12     6_150.96       0.2397          1.0749         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_672.84       610.17     6_283.01       0.3605          1.0474         2.03
ExhaustiveBinary-256-random (self)                     5_672.84     1_584.59     7_257.43       0.2400          1.0748         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_295.67       362.50     6_658.17       0.1907             NaN         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_295.67       492.90     6_788.57       0.5143          1.0274         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_295.67       620.40     6_916.07       0.6417          1.0162         2.03
ExhaustiveBinary-256-pca (self)                        6_295.67     1_621.80     7_917.47       0.5163          1.0273         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_097.15       653.95    11_751.10       0.1039             NaN         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_097.15       780.60    11_877.75       0.2971          1.0611         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_097.15       908.80    12_005.95       0.4344          1.0373         4.05
ExhaustiveBinary-512-random (self)                    11_097.15     2_580.24    13_677.39       0.2968          1.0610         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_861.15       669.05    12_530.20       0.2182             NaN         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_861.15       814.88    12_676.03       0.5365          1.0253         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_861.15       927.96    12_789.11       0.6508          1.0154         4.05
ExhaustiveBinary-512-pca (self)                       11_861.15     2_638.95    14_500.10       0.5379          1.0252         4.05
ExhaustiveBinary-1024-random_no_rr (query)            21_884.33     1_173.03    23_057.36       0.1273             NaN         8.10
ExhaustiveBinary-1024-random-rf10 (query)             21_884.33     1_313.87    23_198.20       0.3824          1.0455         8.10
ExhaustiveBinary-1024-random-rf20 (query)             21_884.33     1_455.20    23_339.53       0.5379          1.0261         8.10
ExhaustiveBinary-1024-random (self)                   21_884.33     4_376.02    26_260.35       0.3828          1.0454         8.10
ExhaustiveBinary-1024-pca_no_rr (query)               23_217.82     1_197.97    24_415.78       0.2506             NaN         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                23_217.82     1_349.30    24_567.11       0.6932          1.0125         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                23_217.82     1_493.01    24_710.83       0.8406          1.0051         8.11
ExhaustiveBinary-1024-pca (self)                      23_217.82     4_467.82    27_685.63       0.6937          1.0125         8.11
ExhaustiveBinary-512-signed_no_rr (query)             11_051.86       655.43    11_707.29       0.1039             NaN         4.05
ExhaustiveBinary-512-signed-rf10 (query)              11_051.86       778.63    11_830.49       0.2971          1.0611         4.05
ExhaustiveBinary-512-signed-rf20 (query)              11_051.86       911.36    11_963.22       0.4344          1.0373         4.05
ExhaustiveBinary-512-signed (self)                    11_051.86     2_593.37    13_645.23       0.2968          1.0610         4.05
IVF-Binary-256-nl158-np7-rf0-random (query)            8_679.76       229.82     8_909.58       0.0930             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           8_679.76       235.19     8_914.96       0.0910             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           8_679.76       240.31     8_920.08       0.0899             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           8_679.76       313.60     8_993.36       0.2562          1.0706         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           8_679.76       394.61     9_074.37       0.3841          1.0441         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          8_679.76       323.06     9_002.82       0.2455          1.0732         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          8_679.76       417.34     9_097.10       0.3695          1.0461         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          8_679.76       336.31     9_016.08       0.2406          1.0746         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          8_679.76       426.35     9_106.11       0.3628          1.0472         2.34
IVF-Binary-256-nl158-random (self)                     8_679.76     1_030.18     9_709.94       0.2463          1.0731         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           6_612.50       249.37     6_861.87       0.0926             NaN         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)           6_612.50       242.13     6_854.63       0.0914             NaN         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)           6_612.50       245.65     6_858.15       0.0898             NaN         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)          6_612.50       327.54     6_940.04       0.2507          1.0715         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)          6_612.50       408.54     7_021.04       0.3754          1.0449         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)          6_612.50       325.03     6_937.53       0.2453          1.0732         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)          6_612.50       407.42     7_019.92       0.3675          1.0462         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)          6_612.50       334.42     6_946.91       0.2406          1.0748         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)          6_612.50       425.99     7_038.49       0.3612          1.0474         2.46
IVF-Binary-256-nl223-random (self)                     6_612.50     1_021.21     7_633.71       0.2459          1.0730         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)           6_944.62       250.93     7_195.55       0.0916             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           6_944.62       254.36     7_198.98       0.0908             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           6_944.62       258.75     7_203.37       0.0900             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          6_944.62       341.02     7_285.64       0.2484          1.0722         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          6_944.62       419.18     7_363.80       0.3725          1.0453         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          6_944.62       338.54     7_283.16       0.2446          1.0733         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          6_944.62       425.67     7_370.29       0.3671          1.0463         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          6_944.62       355.87     7_300.49       0.2412          1.0744         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          6_944.62       434.15     7_378.77       0.3624          1.0471         2.65
IVF-Binary-256-nl316-random (self)                     6_944.62     1_136.99     8_081.61       0.2449          1.0733         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               9_366.44       237.79     9_604.24       0.1962             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              9_366.44       242.95     9_609.40       0.1955             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              9_366.44       249.10     9_615.55       0.1948             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              9_366.44       329.57     9_696.01       0.5664          1.0224         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              9_366.44       413.69     9_780.13       0.7221          1.0113         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             9_366.44       337.44     9_703.89       0.5686          1.0221         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             9_366.44       426.06     9_792.51       0.7288          1.0108         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             9_366.44       347.51     9_713.96       0.5653          1.0224         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             9_366.44       442.24     9_808.68       0.7241          1.0110         2.34
IVF-Binary-256-nl158-pca (self)                        9_366.44     1_065.43    10_431.87       0.5708          1.0220         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              7_263.26       244.60     7_507.86       0.1962             NaN         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              7_263.26       247.76     7_511.02       0.1954             NaN         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              7_263.26       262.16     7_525.43       0.1948             NaN         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             7_263.26       340.09     7_603.36       0.5703          1.0220         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             7_263.26       422.12     7_685.38       0.7308          1.0106         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             7_263.26       341.75     7_605.01       0.5680          1.0222         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             7_263.26       430.84     7_694.10       0.7292          1.0107         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             7_263.26       350.60     7_613.86       0.5655          1.0224         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             7_263.26       452.05     7_715.31       0.7249          1.0110         2.47
IVF-Binary-256-nl223-pca (self)                        7_263.26     1_073.93     8_337.19       0.5703          1.0221         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              7_648.31       261.37     7_909.68       0.1960             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              7_648.31       260.52     7_908.83       0.1954             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              7_648.31       265.36     7_913.67       0.1949             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             7_648.31       355.15     8_003.46       0.5718          1.0218         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             7_648.31       441.85     8_090.16       0.7343          1.0105         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             7_648.31       352.10     8_000.41       0.5698          1.0220         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             7_648.31       440.66     8_088.97       0.7321          1.0106         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             7_648.31       360.83     8_009.14       0.5670          1.0222         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             7_648.31       453.81     8_102.12       0.7274          1.0108         2.65
IVF-Binary-256-nl316-pca (self)                        7_648.31     1_117.57     8_765.89       0.5719          1.0219         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           14_313.02       429.29    14_742.30       0.1075             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          14_313.02       437.13    14_750.14       0.1057             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          14_313.02       443.38    14_756.40       0.1046             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          14_313.02       515.00    14_828.02       0.3090          1.0587         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          14_313.02       595.35    14_908.36       0.4503          1.0355         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         14_313.02       532.57    14_845.58       0.3015          1.0601         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         14_313.02       607.80    14_920.82       0.4408          1.0365         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         14_313.02       543.25    14_856.27       0.2976          1.0610         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         14_313.02       625.01    14_938.03       0.4354          1.0372         4.36
IVF-Binary-512-nl158-random (self)                    14_313.02     1_686.10    15_999.11       0.3015          1.0601         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          12_133.99       437.91    12_571.90       0.1064             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          12_133.99       440.52    12_574.51       0.1054             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          12_133.99       453.83    12_587.83       0.1044             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         12_133.99       521.90    12_655.89       0.3056          1.0589         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         12_133.99       596.76    12_730.75       0.4458          1.0357         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         12_133.99       526.00    12_659.99       0.3007          1.0601         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         12_133.99       611.69    12_745.68       0.4393          1.0366         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         12_133.99       539.01    12_673.00       0.2972          1.0611         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         12_133.99       623.71    12_757.70       0.4349          1.0373         4.49
IVF-Binary-512-nl223-random (self)                    12_133.99     1_688.96    13_822.95       0.3008          1.0601         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          12_416.28       453.83    12_870.11       0.1060             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          12_416.28       452.78    12_869.06       0.1053             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          12_416.28       459.65    12_875.92       0.1047             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         12_416.28       536.60    12_952.87       0.3039          1.0594         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         12_416.28       622.23    13_038.51       0.4442          1.0360         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         12_416.28       541.44    12_957.72       0.3008          1.0602         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         12_416.28       621.85    13_038.13       0.4397          1.0366         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         12_416.28       548.89    12_965.17       0.2978          1.0609         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         12_416.28       630.42    13_046.69       0.4360          1.0371         4.67
IVF-Binary-512-nl316-random (self)                    12_416.28     1_728.63    14_144.91       0.3008          1.0601         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              15_077.75       443.71    15_521.47       0.2351             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             15_077.75       455.68    15_533.43       0.2346             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             15_077.75       461.41    15_539.16       0.2334             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             15_077.75       534.08    15_611.83       0.6487          1.0157         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             15_077.75       620.36    15_698.11       0.7884          1.0077         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            15_077.75       550.06    15_627.82       0.6521          1.0153         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            15_077.75       643.82    15_721.58       0.7985          1.0069         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            15_077.75       567.87    15_645.62       0.6451          1.0157         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            15_077.75       663.92    15_741.67       0.7900          1.0073         4.36
IVF-Binary-512-nl158-pca (self)                       15_077.75     1_773.97    16_851.72       0.6524          1.0153         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             12_894.99       448.12    13_343.11       0.2347             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             12_894.99       452.78    13_347.77       0.2340             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             12_894.99       460.75    13_355.74       0.2332             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            12_894.99       542.47    13_437.45       0.6559          1.0150         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            12_894.99       636.85    13_531.84       0.8048          1.0067         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            12_894.99       544.52    13_439.51       0.6536          1.0151         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            12_894.99       635.51    13_530.50       0.8031          1.0067         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            12_894.99       560.44    13_455.42       0.6471          1.0156         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            12_894.99       649.56    13_544.55       0.7931          1.0072         4.49
IVF-Binary-512-nl223-pca (self)                       12_894.99     1_760.61    14_655.60       0.6538          1.0152         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             13_225.61       460.16    13_685.77       0.2352             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             13_225.61       463.34    13_688.95       0.2347             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             13_225.61       472.91    13_698.52       0.2339             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            13_225.61       558.06    13_783.67       0.6589          1.0148         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            13_225.61       638.81    13_864.42       0.8096          1.0064         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            13_225.61       557.58    13_783.19       0.6568          1.0149         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            13_225.61       648.35    13_873.96       0.8076          1.0065         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            13_225.61       570.53    13_796.15       0.6502          1.0154         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            13_225.61       661.86    13_887.47       0.7981          1.0069         4.67
IVF-Binary-512-nl316-pca (self)                       13_225.61     1_802.46    15_028.07       0.6571          1.0149         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          25_134.60       841.04    25_975.64       0.1306             NaN         8.41
IVF-Binary-1024-nl158-np12-rf0-random (query)         25_134.60       878.71    26_013.31       0.1289             NaN         8.41
IVF-Binary-1024-nl158-np17-rf0-random (query)         25_134.60       872.26    26_006.86       0.1279             NaN         8.41
IVF-Binary-1024-nl158-np7-rf10-random (query)         25_134.60       936.00    26_070.60       0.3897          1.0444         8.41
IVF-Binary-1024-nl158-np7-rf20-random (query)         25_134.60     1_172.97    26_307.57       0.5436          1.0256         8.41
IVF-Binary-1024-nl158-np12-rf10-random (query)        25_134.60     1_108.70    26_243.30       0.3858          1.0449         8.41
IVF-Binary-1024-nl158-np12-rf20-random (query)        25_134.60     1_087.59    26_222.19       0.5426          1.0257         8.41
IVF-Binary-1024-nl158-np17-rf10-random (query)        25_134.60     1_045.63    26_180.23       0.3828          1.0454         8.41
IVF-Binary-1024-nl158-np17-rf20-random (query)        25_134.60     1_091.47    26_226.07       0.5386          1.0260         8.41
IVF-Binary-1024-nl158-random (self)                   25_134.60     3_397.02    28_531.62       0.3865          1.0449         8.41
IVF-Binary-1024-nl223-np11-rf0-random (query)         23_217.30       854.37    24_071.67       0.1299             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         23_217.30       847.51    24_064.81       0.1289             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         23_217.30       874.99    24_092.30       0.1280             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        23_217.30       940.49    24_157.79       0.3892          1.0442         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        23_217.30     1_031.55    24_248.85       0.5456          1.0253         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        23_217.30       949.77    24_167.08       0.3854          1.0449         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        23_217.30     1_031.76    24_249.07       0.5398          1.0259         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        23_217.30       963.17    24_180.47       0.3829          1.0454         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        23_217.30     1_060.90    24_278.21       0.5372          1.0262         8.54
IVF-Binary-1024-nl223-random (self)                   23_217.30     3_085.87    26_303.17       0.3854          1.0450         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         23_341.66       855.08    24_196.75       0.1291             NaN         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)         23_341.66       882.18    24_223.85       0.1284             NaN         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)         23_341.66       872.13    24_213.79       0.1278             NaN         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)        23_341.66     1_043.83    24_385.50       0.3879          1.0445         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)        23_341.66     1_026.13    24_367.79       0.5448          1.0254         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)        23_341.66       938.22    24_279.89       0.3851          1.0450         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)        23_341.66     1_021.07    24_362.74       0.5409          1.0258         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)        23_341.66       959.16    24_300.82       0.3830          1.0454         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)        23_341.66     1_046.30    24_387.96       0.5381          1.0261         8.72
IVF-Binary-1024-nl316-random (self)                   23_341.66     3_082.80    26_424.46       0.3856          1.0449         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)             26_441.04       843.41    27_284.45       0.2510             NaN         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            26_441.04       859.57    27_300.61       0.2512             NaN         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            26_441.04       893.19    27_334.23       0.2507             NaN         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            26_441.04       957.52    27_398.56       0.6813          1.0134         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            26_441.04     1_043.44    27_484.48       0.8163          1.0064         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           26_441.04       984.77    27_425.81       0.6931          1.0125         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           26_441.04     1_082.80    27_523.84       0.8385          1.0052         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           26_441.04     1_011.59    27_452.63       0.6931          1.0125         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           26_441.04     1_098.11    27_539.15       0.8405          1.0051         8.42
IVF-Binary-1024-nl158-pca (self)                      26_441.04     3_195.81    29_636.85       0.6933          1.0125         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            24_309.75       857.26    25_167.01       0.2512             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            24_309.75       859.21    25_168.96       0.2509             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            24_309.75       879.18    25_188.94       0.2507             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           24_309.75       947.02    25_256.77       0.6923          1.0125         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           24_309.75     1_035.78    25_345.53       0.8371          1.0053         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           24_309.75       952.75    25_262.50       0.6936          1.0125         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           24_309.75     1_049.28    25_359.03       0.8405          1.0051         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           24_309.75       971.51    25_281.26       0.6932          1.0125         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           24_309.75     1_076.72    25_386.47       0.8405          1.0051         8.54
IVF-Binary-1024-nl223-pca (self)                      24_309.75     3_126.43    27_436.18       0.6941          1.0125         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            24_675.43       866.73    25_542.15       0.2512             NaN         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            24_675.43       877.57    25_553.00       0.2509             NaN         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            24_675.43       884.87    25_560.29       0.2506             NaN         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           24_675.43       966.51    25_641.94       0.6943          1.0124         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           24_675.43     1_045.63    25_721.06       0.8399          1.0051         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           24_675.43       960.45    25_635.87       0.6939          1.0124         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           24_675.43     1_054.72    25_730.14       0.8408          1.0051         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           24_675.43       983.67    25_659.10       0.6932          1.0125         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           24_675.43     1_072.71    25_748.14       0.8405          1.0051         8.73
IVF-Binary-1024-nl316-pca (self)                      24_675.43     3_160.95    27_836.37       0.6943          1.0125         8.73
IVF-Binary-512-nl158-np7-rf0-signed (query)           14_233.01       442.18    14_675.19       0.1075             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-signed (query)          14_233.01       444.37    14_677.38       0.1057             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-signed (query)          14_233.01       450.41    14_683.42       0.1046             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-signed (query)          14_233.01       525.17    14_758.18       0.3090          1.0587         4.36
IVF-Binary-512-nl158-np7-rf20-signed (query)          14_233.01       609.52    14_842.53       0.4503          1.0355         4.36
IVF-Binary-512-nl158-np12-rf10-signed (query)         14_233.01       535.61    14_768.62       0.3015          1.0601         4.36
IVF-Binary-512-nl158-np12-rf20-signed (query)         14_233.01       619.40    14_852.41       0.4408          1.0365         4.36
IVF-Binary-512-nl158-np17-rf10-signed (query)         14_233.01       544.04    14_777.05       0.2976          1.0610         4.36
IVF-Binary-512-nl158-np17-rf20-signed (query)         14_233.01       642.23    14_875.25       0.4354          1.0372         4.36
IVF-Binary-512-nl158-signed (self)                    14_233.01     1_724.64    15_957.65       0.3015          1.0601         4.36
IVF-Binary-512-nl223-np11-rf0-signed (query)          12_139.80       439.21    12_579.00       0.1064             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-signed (query)          12_139.80       443.03    12_582.83       0.1054             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-signed (query)          12_139.80       457.76    12_597.56       0.1044             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-signed (query)         12_139.80       532.61    12_672.41       0.3056          1.0589         4.49
IVF-Binary-512-nl223-np11-rf20-signed (query)         12_139.80       606.82    12_746.62       0.4458          1.0357         4.49
IVF-Binary-512-nl223-np14-rf10-signed (query)         12_139.80       533.53    12_673.32       0.3007          1.0601         4.49
IVF-Binary-512-nl223-np14-rf20-signed (query)         12_139.80       625.15    12_764.94       0.4393          1.0366         4.49
IVF-Binary-512-nl223-np21-rf10-signed (query)         12_139.80       546.51    12_686.31       0.2972          1.0611         4.49
IVF-Binary-512-nl223-np21-rf20-signed (query)         12_139.80       640.19    12_779.99       0.4349          1.0373         4.49
IVF-Binary-512-nl223-signed (self)                    12_139.80     1_703.92    13_843.72       0.3008          1.0601         4.49
IVF-Binary-512-nl316-np15-rf0-signed (query)          12_429.35       455.51    12_884.85       0.1060             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-signed (query)          12_429.35       457.89    12_887.24       0.1053             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-signed (query)          12_429.35       462.66    12_892.01       0.1047             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-signed (query)         12_429.35       543.03    12_972.38       0.3039          1.0594         4.67
IVF-Binary-512-nl316-np15-rf20-signed (query)         12_429.35       635.80    13_065.14       0.4442          1.0360         4.67
IVF-Binary-512-nl316-np17-rf10-signed (query)         12_429.35       543.05    12_972.40       0.3008          1.0602         4.67
IVF-Binary-512-nl316-np17-rf20-signed (query)         12_429.35       625.71    13_055.05       0.4397          1.0366         4.67
IVF-Binary-512-nl316-np25-rf10-signed (query)         12_429.35       568.47    12_997.82       0.2978          1.0609         4.67
IVF-Binary-512-nl316-np25-rf20-signed (query)         12_429.35       640.51    13_069.85       0.4360          1.0371         4.67
IVF-Binary-512-nl316-signed (self)                    12_429.35     1_750.70    14_180.04       0.3008          1.0601         4.67
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
Exhaustive (query)                                        40.97    22_126.77    22_167.75       1.0000          1.0000       195.31
Exhaustive (self)                                         40.97    73_233.18    73_274.16       1.0000          1.0000       195.31
ExhaustiveBinary-256-random_no_rr (query)             11_982.50       570.80    12_553.30       0.0840             NaN         2.53
ExhaustiveBinary-256-random-rf10 (query)              11_982.50       733.23    12_715.73       0.2060          1.0589         2.53
ExhaustiveBinary-256-random-rf20 (query)              11_982.50       905.96    12_888.46       0.3135          1.0384         2.53
ExhaustiveBinary-256-random (self)                    11_982.50     2_426.77    14_409.27       0.2052          1.0589         2.53
ExhaustiveBinary-256-pca_no_rr (query)                13_502.04       596.38    14_098.41       0.1619             NaN         2.53
ExhaustiveBinary-256-pca-rf10 (query)                 13_502.04       762.14    14_264.17       0.4353          1.0256         2.53
ExhaustiveBinary-256-pca-rf20 (query)                 13_502.04       917.12    14_419.16       0.5548          1.0162         2.53
ExhaustiveBinary-256-pca (self)                       13_502.04     2_466.87    15_968.90       0.4349          1.0256         2.53
ExhaustiveBinary-512-random_no_rr (query)             23_493.95     1_094.93    24_588.89       0.0922             NaN         5.05
ExhaustiveBinary-512-random-rf10 (query)              23_493.95     1_255.98    24_749.94       0.2430          1.0508         5.05
ExhaustiveBinary-512-random-rf20 (query)              23_493.95     1_417.49    24_911.45       0.3648          1.0323         5.05
ExhaustiveBinary-512-random (self)                    23_493.95     4_148.76    27_642.71       0.2441          1.0506         5.05
ExhaustiveBinary-512-pca_no_rr (query)                25_010.68     1_108.98    26_119.66       0.1735             NaN         5.06
ExhaustiveBinary-512-pca-rf10 (query)                 25_010.68     1_285.74    26_296.41       0.4277          1.0267         5.06
ExhaustiveBinary-512-pca-rf20 (query)                 25_010.68     1_454.12    26_464.79       0.5354          1.0175         5.06
ExhaustiveBinary-512-pca (self)                       25_010.68     4_254.91    29_265.58       0.4286          1.0266         5.06
ExhaustiveBinary-1024-random_no_rr (query)            47_104.40     2_041.91    49_146.31       0.1077             NaN        10.10
ExhaustiveBinary-1024-random-rf10 (query)             47_104.40     2_224.27    49_328.67       0.3093          1.0402        10.10
ExhaustiveBinary-1024-random-rf20 (query)             47_104.40     2_418.66    49_523.05       0.4493          1.0244        10.10
ExhaustiveBinary-1024-random (self)                   47_104.40     7_379.54    54_483.93       0.3094          1.0401        10.10
ExhaustiveBinary-1024-pca_no_rr (query)               48_780.92     2_083.32    50_864.24       0.2002             NaN        10.11
ExhaustiveBinary-1024-pca-rf10 (query)                48_780.92     2_247.33    51_028.25       0.4677          1.0246        10.11
ExhaustiveBinary-1024-pca-rf20 (query)                48_780.92     2_441.88    51_222.81       0.5715          1.0152        10.11
ExhaustiveBinary-1024-pca (self)                      48_780.92     7_474.24    56_255.17       0.4688          1.0240        10.11
ExhaustiveBinary-1024-signed_no_rr (query)            47_004.16     2_051.74    49_055.90       0.1077             NaN        10.10
ExhaustiveBinary-1024-signed-rf10 (query)             47_004.16     2_233.73    49_237.90       0.3093          1.0402        10.10
ExhaustiveBinary-1024-signed-rf20 (query)             47_004.16     2_404.05    49_408.21       0.4493          1.0244        10.10
ExhaustiveBinary-1024-signed (self)                   47_004.16     7_383.06    54_387.22       0.3094          1.0401        10.10
IVF-Binary-256-nl158-np7-rf0-random (query)           18_935.12       476.93    19_412.05       0.0856             NaN         3.14
IVF-Binary-256-nl158-np12-rf0-random (query)          18_935.12       484.23    19_419.35       0.0848             NaN         3.14
IVF-Binary-256-nl158-np17-rf0-random (query)          18_935.12       488.39    19_423.51       0.0843             NaN         3.14
IVF-Binary-256-nl158-np7-rf10-random (query)          18_935.12       651.15    19_586.27       0.2155          1.0567         3.14
IVF-Binary-256-nl158-np7-rf20-random (query)          18_935.12       732.12    19_667.24       0.3261          1.0368         3.14
IVF-Binary-256-nl158-np12-rf10-random (query)         18_935.12       615.29    19_550.41       0.2106          1.0579         3.14
IVF-Binary-256-nl158-np12-rf20-random (query)         18_935.12       750.37    19_685.49       0.3185          1.0377         3.14
IVF-Binary-256-nl158-np17-rf10-random (query)         18_935.12       626.02    19_561.14       0.2076          1.0586         3.14
IVF-Binary-256-nl158-np17-rf20-random (query)         18_935.12       768.70    19_703.81       0.3141          1.0383         3.14
IVF-Binary-256-nl158-random (self)                    18_935.12     1_963.86    20_898.98       0.2095          1.0579         3.14
IVF-Binary-256-nl223-np11-rf0-random (query)          13_787.53       496.08    14_283.61       0.0860             NaN         3.40
IVF-Binary-256-nl223-np14-rf0-random (query)          13_787.53       507.47    14_295.00       0.0851             NaN         3.40
IVF-Binary-256-nl223-np21-rf0-random (query)          13_787.53       504.68    14_292.21       0.0843             NaN         3.40
IVF-Binary-256-nl223-np11-rf10-random (query)         13_787.53       627.35    14_414.88       0.2135          1.0570         3.40
IVF-Binary-256-nl223-np11-rf20-random (query)         13_787.53       748.19    14_535.72       0.3241          1.0370         3.40
IVF-Binary-256-nl223-np14-rf10-random (query)         13_787.53       631.72    14_419.25       0.2100          1.0579         3.40
IVF-Binary-256-nl223-np14-rf20-random (query)         13_787.53       754.86    14_542.39       0.3192          1.0377         3.40
IVF-Binary-256-nl223-np21-rf10-random (query)         13_787.53       636.46    14_423.99       0.2075          1.0586         3.40
IVF-Binary-256-nl223-np21-rf20-random (query)         13_787.53       780.12    14_567.65       0.3160          1.0381         3.40
IVF-Binary-256-nl223-random (self)                    13_787.53     2_014.90    15_802.43       0.2092          1.0578         3.40
IVF-Binary-256-nl316-np15-rf0-random (query)          14_451.18       536.28    14_987.46       0.0853             NaN         3.76
IVF-Binary-256-nl316-np17-rf0-random (query)          14_451.18       550.94    15_002.12       0.0849             NaN         3.76
IVF-Binary-256-nl316-np25-rf0-random (query)          14_451.18       543.33    14_994.51       0.0845             NaN         3.76
IVF-Binary-256-nl316-np15-rf10-random (query)         14_451.18       665.20    15_116.39       0.2120          1.0574         3.76
IVF-Binary-256-nl316-np15-rf20-random (query)         14_451.18       788.20    15_239.38       0.3218          1.0373         3.76
IVF-Binary-256-nl316-np17-rf10-random (query)         14_451.18       660.96    15_112.14       0.2103          1.0579         3.76
IVF-Binary-256-nl316-np17-rf20-random (query)         14_451.18       804.43    15_255.61       0.3193          1.0376         3.76
IVF-Binary-256-nl316-np25-rf10-random (query)         14_451.18       669.28    15_120.46       0.2083          1.0584         3.76
IVF-Binary-256-nl316-np25-rf20-random (query)         14_451.18       800.98    15_252.16       0.3164          1.0380         3.76
IVF-Binary-256-nl316-random (self)                    14_451.18     2_131.15    16_582.33       0.2093          1.0578         3.76
IVF-Binary-256-nl158-np7-rf0-pca (query)              20_630.88       493.06    21_123.94       0.1677             NaN         3.15
IVF-Binary-256-nl158-np12-rf0-pca (query)             20_630.88       489.73    21_120.61       0.1675             NaN         3.15
IVF-Binary-256-nl158-np17-rf0-pca (query)             20_630.88       495.54    21_126.43       0.1670             NaN         3.15
IVF-Binary-256-nl158-np7-rf10-pca (query)             20_630.88       615.68    21_246.56       0.4931          1.0207         3.15
IVF-Binary-256-nl158-np7-rf20-pca (query)             20_630.88       744.20    21_375.09       0.6468          1.0113         3.15
IVF-Binary-256-nl158-np12-rf10-pca (query)            20_630.88       620.37    21_251.26       0.4949          1.0205         3.15
IVF-Binary-256-nl158-np12-rf20-pca (query)            20_630.88       756.46    21_387.35       0.6508          1.0111         3.15
IVF-Binary-256-nl158-np17-rf10-pca (query)            20_630.88       635.55    21_266.44       0.4906          1.0208         3.15
IVF-Binary-256-nl158-np17-rf20-pca (query)            20_630.88       775.08    21_405.97       0.6442          1.0113         3.15
IVF-Binary-256-nl158-pca (self)                       20_630.88     2_021.20    22_652.09       0.4943          1.0205         3.15
IVF-Binary-256-nl223-np11-rf0-pca (query)             15_358.34       507.76    15_866.10       0.1679             NaN         3.40
IVF-Binary-256-nl223-np14-rf0-pca (query)             15_358.34       507.30    15_865.64       0.1674             NaN         3.40
IVF-Binary-256-nl223-np21-rf0-pca (query)             15_358.34       517.14    15_875.48       0.1669             NaN         3.40
IVF-Binary-256-nl223-np11-rf10-pca (query)            15_358.34       640.76    15_999.10       0.4988          1.0202         3.40
IVF-Binary-256-nl223-np11-rf20-pca (query)            15_358.34       762.49    16_120.83       0.6593          1.0107         3.40
IVF-Binary-256-nl223-np14-rf10-pca (query)            15_358.34       639.86    15_998.20       0.4968          1.0204         3.40
IVF-Binary-256-nl223-np14-rf20-pca (query)            15_358.34       774.70    16_133.04       0.6564          1.0108         3.40
IVF-Binary-256-nl223-np21-rf10-pca (query)            15_358.34       650.88    16_009.22       0.4926          1.0207         3.40
IVF-Binary-256-nl223-np21-rf20-pca (query)            15_358.34       788.63    16_146.97       0.6488          1.0111         3.40
IVF-Binary-256-nl223-pca (self)                       15_358.34     2_083.09    17_441.43       0.4967          1.0203         3.40
IVF-Binary-256-nl316-np15-rf0-pca (query)             16_051.86       535.17    16_587.03       0.1681             NaN         3.77
IVF-Binary-256-nl316-np17-rf0-pca (query)             16_051.86       541.50    16_593.36       0.1678             NaN         3.77
IVF-Binary-256-nl316-np25-rf0-pca (query)             16_051.86       542.71    16_594.56       0.1673             NaN         3.77
IVF-Binary-256-nl316-np15-rf10-pca (query)            16_051.86       670.81    16_722.67       0.5001          1.0201         3.77
IVF-Binary-256-nl316-np15-rf20-pca (query)            16_051.86       803.37    16_855.23       0.6619          1.0106         3.77
IVF-Binary-256-nl316-np17-rf10-pca (query)            16_051.86       675.53    16_727.38       0.4989          1.0202         3.77
IVF-Binary-256-nl316-np17-rf20-pca (query)            16_051.86       800.52    16_852.38       0.6601          1.0106         3.77
IVF-Binary-256-nl316-np25-rf10-pca (query)            16_051.86       677.45    16_729.31       0.4942          1.0205         3.77
IVF-Binary-256-nl316-np25-rf20-pca (query)            16_051.86       812.18    16_864.03       0.6525          1.0110         3.77
IVF-Binary-256-nl316-pca (self)                       16_051.86     2_179.33    18_231.18       0.4987          1.0202         3.77
IVF-Binary-512-nl158-np7-rf0-random (query)           30_824.32       898.24    31_722.56       0.0937             NaN         5.67
IVF-Binary-512-nl158-np12-rf0-random (query)          30_824.32       904.19    31_728.51       0.0930             NaN         5.67
IVF-Binary-512-nl158-np17-rf0-random (query)          30_824.32       916.25    31_740.57       0.0924             NaN         5.67
IVF-Binary-512-nl158-np7-rf10-random (query)          30_824.32     1_013.84    31_838.16       0.2508          1.0494         5.67
IVF-Binary-512-nl158-np7-rf20-random (query)          30_824.32     1_147.91    31_972.23       0.3737          1.0313         5.67
IVF-Binary-512-nl158-np12-rf10-random (query)         30_824.32     1_027.95    31_852.27       0.2470          1.0501         5.67
IVF-Binary-512-nl158-np12-rf20-random (query)         30_824.32     1_160.51    31_984.83       0.3689          1.0318         5.67
IVF-Binary-512-nl158-np17-rf10-random (query)         30_824.32     1_040.24    31_864.56       0.2448          1.0505         5.67
IVF-Binary-512-nl158-np17-rf20-random (query)         30_824.32     1_170.46    31_994.78       0.3654          1.0322         5.67
IVF-Binary-512-nl158-random (self)                    30_824.32     3_347.80    34_172.12       0.2475          1.0499         5.67
IVF-Binary-512-nl223-np11-rf0-random (query)          25_506.59       907.92    26_414.51       0.0937             NaN         5.92
IVF-Binary-512-nl223-np14-rf0-random (query)          25_506.59       915.15    26_421.75       0.0931             NaN         5.92
IVF-Binary-512-nl223-np21-rf0-random (query)          25_506.59       929.09    26_435.68       0.0927             NaN         5.92
IVF-Binary-512-nl223-np11-rf10-random (query)         25_506.59     1_034.17    26_540.77       0.2494          1.0496         5.92
IVF-Binary-512-nl223-np11-rf20-random (query)         25_506.59     1_155.36    26_661.95       0.3728          1.0313         5.92
IVF-Binary-512-nl223-np14-rf10-random (query)         25_506.59     1_062.56    26_569.16       0.2462          1.0502         5.92
IVF-Binary-512-nl223-np14-rf20-random (query)         25_506.59     1_154.37    26_660.97       0.3684          1.0318         5.92
IVF-Binary-512-nl223-np21-rf10-random (query)         25_506.59     1_045.95    26_552.54       0.2446          1.0505         5.92
IVF-Binary-512-nl223-np21-rf20-random (query)         25_506.59     1_175.84    26_682.44       0.3662          1.0321         5.92
IVF-Binary-512-nl223-random (self)                    25_506.59     3_386.33    28_892.92       0.2469          1.0500         5.92
IVF-Binary-512-nl316-np15-rf0-random (query)          26_121.72       964.14    27_085.86       0.0933             NaN         6.29
IVF-Binary-512-nl316-np17-rf0-random (query)          26_121.72       964.44    27_086.16       0.0930             NaN         6.29
IVF-Binary-512-nl316-np25-rf0-random (query)          26_121.72       971.82    27_093.54       0.0927             NaN         6.29
IVF-Binary-512-nl316-np15-rf10-random (query)         26_121.72     1_095.64    27_217.36       0.2484          1.0497         6.29
IVF-Binary-512-nl316-np15-rf20-random (query)         26_121.72     1_218.06    27_339.78       0.3714          1.0315         6.29
IVF-Binary-512-nl316-np17-rf10-random (query)         26_121.72     1_091.74    27_213.46       0.2466          1.0501         6.29
IVF-Binary-512-nl316-np17-rf20-random (query)         26_121.72     1_225.43    27_347.15       0.3692          1.0318         6.29
IVF-Binary-512-nl316-np25-rf10-random (query)         26_121.72     1_103.67    27_225.39       0.2450          1.0504         6.29
IVF-Binary-512-nl316-np25-rf20-random (query)         26_121.72     1_257.58    27_379.30       0.3672          1.0321         6.29
IVF-Binary-512-nl316-random (self)                    26_121.72     3_631.97    29_753.69       0.2475          1.0499         6.29
IVF-Binary-512-nl158-np7-rf0-pca (query)              32_422.31       900.69    33_322.99       0.1894             NaN         5.67
IVF-Binary-512-nl158-np12-rf0-pca (query)             32_422.31       907.04    33_329.35       0.1890             NaN         5.67
IVF-Binary-512-nl158-np17-rf0-pca (query)             32_422.31       937.32    33_359.63       0.1879             NaN         5.67
IVF-Binary-512-nl158-np7-rf10-pca (query)             32_422.31     1_121.57    33_543.88       0.5413          1.0172         5.67
IVF-Binary-512-nl158-np7-rf20-pca (query)             32_422.31     1_151.23    33_573.54       0.6899          1.0093         5.67
IVF-Binary-512-nl158-np12-rf10-pca (query)            32_422.31     1_044.37    33_466.67       0.5401          1.0172         5.67
IVF-Binary-512-nl158-np12-rf20-pca (query)            32_422.31     1_173.66    33_595.96       0.6907          1.0091         5.67
IVF-Binary-512-nl158-np17-rf10-pca (query)            32_422.31     1_059.39    33_481.70       0.5311          1.0178         5.67
IVF-Binary-512-nl158-np17-rf20-pca (query)            32_422.31     1_202.06    33_624.36       0.6781          1.0096         5.67
IVF-Binary-512-nl158-pca (self)                       32_422.31     3_418.40    35_840.71       0.5401          1.0171         5.67
IVF-Binary-512-nl223-np11-rf0-pca (query)             27_305.66       923.83    28_229.49       0.1897             NaN         5.93
IVF-Binary-512-nl223-np14-rf0-pca (query)             27_305.66       932.83    28_238.48       0.1891             NaN         5.93
IVF-Binary-512-nl223-np21-rf0-pca (query)             27_305.66       932.15    28_237.81       0.1881             NaN         5.93
IVF-Binary-512-nl223-np11-rf10-pca (query)            27_305.66     1_053.91    28_359.57       0.5512          1.0165         5.93
IVF-Binary-512-nl223-np11-rf20-pca (query)            27_305.66     1_188.35    28_494.01       0.7062          1.0084         5.93
IVF-Binary-512-nl223-np14-rf10-pca (query)            27_305.66     1_077.80    28_383.46       0.5470          1.0167         5.93
IVF-Binary-512-nl223-np14-rf20-pca (query)            27_305.66     1_186.93    28_492.58       0.7024          1.0086         5.93
IVF-Binary-512-nl223-np21-rf10-pca (query)            27_305.66     1_068.43    28_374.09       0.5373          1.0173         5.93
IVF-Binary-512-nl223-np21-rf20-pca (query)            27_305.66     1_212.17    28_517.83       0.6885          1.0091         5.93
IVF-Binary-512-nl223-pca (self)                       27_305.66     3_472.10    30_777.76       0.5468          1.0167         5.93
IVF-Binary-512-nl316-np15-rf0-pca (query)             27_869.51       964.32    28_833.83       0.1901             NaN         6.29
IVF-Binary-512-nl316-np17-rf0-pca (query)             27_869.51       956.88    28_826.39       0.1896             NaN         6.29
IVF-Binary-512-nl316-np25-rf0-pca (query)             27_869.51       962.92    28_832.43       0.1885             NaN         6.29
IVF-Binary-512-nl316-np15-rf10-pca (query)            27_869.51     1_090.24    28_959.75       0.5534          1.0163         6.29
IVF-Binary-512-nl316-np15-rf20-pca (query)            27_869.51     1_211.40    29_080.90       0.7106          1.0083         6.29
IVF-Binary-512-nl316-np17-rf10-pca (query)            27_869.51     1_084.80    28_954.31       0.5511          1.0165         6.29
IVF-Binary-512-nl316-np17-rf20-pca (query)            27_869.51     1_221.51    29_091.02       0.7081          1.0083         6.29
IVF-Binary-512-nl316-np25-rf10-pca (query)            27_869.51     1_103.83    28_973.33       0.5411          1.0171         6.29
IVF-Binary-512-nl316-np25-rf20-pca (query)            27_869.51     1_236.53    29_106.04       0.6938          1.0089         6.29
IVF-Binary-512-nl316-pca (self)                       27_869.51     3_563.95    31_433.45       0.5508          1.0164         6.29
IVF-Binary-1024-nl158-np7-rf0-random (query)          54_366.80     1_706.20    56_073.00       0.1092             NaN        10.72
IVF-Binary-1024-nl158-np12-rf0-random (query)         54_366.80     1_730.43    56_097.23       0.1083             NaN        10.72
IVF-Binary-1024-nl158-np17-rf0-random (query)         54_366.80     1_743.67    56_110.46       0.1079             NaN        10.72
IVF-Binary-1024-nl158-np7-rf10-random (query)         54_366.80     1_873.78    56_240.58       0.3132          1.0396        10.72
IVF-Binary-1024-nl158-np7-rf20-random (query)         54_366.80     1_956.53    56_323.33       0.4522          1.0242        10.72
IVF-Binary-1024-nl158-np12-rf10-random (query)        54_366.80     1_848.63    56_215.43       0.3114          1.0399        10.72
IVF-Binary-1024-nl158-np12-rf20-random (query)        54_366.80     1_973.19    56_339.99       0.4521          1.0242        10.72
IVF-Binary-1024-nl158-np17-rf10-random (query)        54_366.80     1_867.90    56_234.70       0.3099          1.0401        10.72
IVF-Binary-1024-nl158-np17-rf20-random (query)        54_366.80     2_014.18    56_380.98       0.4503          1.0243        10.72
IVF-Binary-1024-nl158-random (self)                   54_366.80     6_101.22    60_468.02       0.3115          1.0398        10.72
IVF-Binary-1024-nl223-np11-rf0-random (query)         49_032.64     1_734.93    50_767.57       0.1092             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-random (query)         49_032.64     1_750.27    50_782.91       0.1085             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-random (query)         49_032.64     1_780.65    50_813.30       0.1081             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-random (query)        49_032.64     1_860.51    50_893.15       0.3144          1.0394        10.98
IVF-Binary-1024-nl223-np11-rf20-random (query)        49_032.64     1_970.55    51_003.20       0.4556          1.0238        10.98
IVF-Binary-1024-nl223-np14-rf10-random (query)        49_032.64     1_877.89    50_910.53       0.3119          1.0398        10.98
IVF-Binary-1024-nl223-np14-rf20-random (query)        49_032.64     2_005.08    51_037.73       0.4520          1.0242        10.98
IVF-Binary-1024-nl223-np21-rf10-random (query)        49_032.64     1_874.81    50_907.46       0.3107          1.0400        10.98
IVF-Binary-1024-nl223-np21-rf20-random (query)        49_032.64     2_011.03    51_043.67       0.4505          1.0243        10.98
IVF-Binary-1024-nl223-random (self)                   49_032.64     6_135.97    55_168.61       0.3117          1.0397        10.98
IVF-Binary-1024-nl316-np15-rf0-random (query)         49_676.92     1_772.37    51_449.29       0.1089             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-random (query)         49_676.92     1_777.49    51_454.41       0.1086             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-random (query)         49_676.92     1_791.80    51_468.72       0.1082             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-random (query)        49_676.92     1_889.12    51_566.04       0.3131          1.0396        11.34
IVF-Binary-1024-nl316-np15-rf20-random (query)        49_676.92     2_032.11    51_709.03       0.4544          1.0239        11.34
IVF-Binary-1024-nl316-np17-rf10-random (query)        49_676.92     1_887.03    51_563.95       0.3118          1.0398        11.34
IVF-Binary-1024-nl316-np17-rf20-random (query)        49_676.92     2_018.02    51_694.94       0.4525          1.0241        11.34
IVF-Binary-1024-nl316-np25-rf10-random (query)        49_676.92     1_898.00    51_574.92       0.3105          1.0400        11.34
IVF-Binary-1024-nl316-np25-rf20-random (query)        49_676.92     2_042.58    51_719.51       0.4510          1.0243        11.34
IVF-Binary-1024-nl316-random (self)                   49_676.92     6_219.53    55_896.45       0.3120          1.0397        11.34
IVF-Binary-1024-nl158-np7-rf0-pca (query)             55_790.81     1_784.74    57_575.56       0.2305             NaN        10.73
IVF-Binary-1024-nl158-np12-rf0-pca (query)            55_790.81     1_779.01    57_569.83       0.2299             NaN        10.73
IVF-Binary-1024-nl158-np17-rf0-pca (query)            55_790.81     1_792.09    57_582.91       0.2279             NaN        10.73
IVF-Binary-1024-nl158-np7-rf10-pca (query)            55_790.81     1_883.50    57_674.31       0.6227          1.0123        10.73
IVF-Binary-1024-nl158-np7-rf20-pca (query)            55_790.81     1_996.50    57_787.32       0.7603          1.0064        10.73
IVF-Binary-1024-nl158-np12-rf10-pca (query)           55_790.81     1_903.02    57_693.83       0.6215          1.0122        10.73
IVF-Binary-1024-nl158-np12-rf20-pca (query)           55_790.81     2_047.22    57_838.03       0.7634          1.0061        10.73
IVF-Binary-1024-nl158-np17-rf10-pca (query)           55_790.81     1_925.66    57_716.48       0.6081          1.0129        10.73
IVF-Binary-1024-nl158-np17-rf20-pca (query)           55_790.81     2_076.58    57_867.40       0.7480          1.0066        10.73
IVF-Binary-1024-nl158-pca (self)                      55_790.81     6_293.04    62_083.85       0.6213          1.0122        10.73
IVF-Binary-1024-nl223-np11-rf0-pca (query)            50_625.37     1_758.16    52_383.53       0.2323             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-pca (query)            50_625.37     1_760.36    52_385.73       0.2311             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-pca (query)            50_625.37     1_780.13    52_405.49       0.2290             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-pca (query)           50_625.37     1_882.91    52_508.28       0.6380          1.0114        10.98
IVF-Binary-1024-nl223-np11-rf20-pca (query)           50_625.37     2_025.69    52_651.06       0.7854          1.0053        10.98
IVF-Binary-1024-nl223-np14-rf10-pca (query)           50_625.37     1_895.64    52_521.01       0.6338          1.0115        10.98
IVF-Binary-1024-nl223-np14-rf20-pca (query)           50_625.37     2_024.67    52_650.04       0.7819          1.0054        10.98
IVF-Binary-1024-nl223-np21-rf10-pca (query)           50_625.37     1_905.25    52_530.62       0.6192          1.0123        10.98
IVF-Binary-1024-nl223-np21-rf20-pca (query)           50_625.37     2_058.98    52_684.35       0.7626          1.0060        10.98
IVF-Binary-1024-nl223-pca (self)                      50_625.37     6_242.53    56_867.90       0.6337          1.0115        10.98
IVF-Binary-1024-nl316-np15-rf0-pca (query)            51_596.63     1_801.27    53_397.90       0.2324             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-pca (query)            51_596.63     1_841.49    53_438.12       0.2320             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-pca (query)            51_596.63     1_843.49    53_440.12       0.2301             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-pca (query)           51_596.63     1_931.95    53_528.58       0.6421          1.0112        11.34
IVF-Binary-1024-nl316-np15-rf20-pca (query)           51_596.63     2_047.01    53_643.64       0.7914          1.0051        11.34
IVF-Binary-1024-nl316-np17-rf10-pca (query)           51_596.63     1_934.87    53_531.51       0.6396          1.0113        11.34
IVF-Binary-1024-nl316-np17-rf20-pca (query)           51_596.63     2_058.01    53_654.65       0.7890          1.0051        11.34
IVF-Binary-1024-nl316-np25-rf10-pca (query)           51_596.63     1_945.64    53_542.28       0.6251          1.0120        11.34
IVF-Binary-1024-nl316-np25-rf20-pca (query)           51_596.63     2_090.68    53_687.31       0.7701          1.0058        11.34
IVF-Binary-1024-nl316-pca (self)                      51_596.63     6_356.82    57_953.45       0.6390          1.0112        11.34
IVF-Binary-1024-nl158-np7-rf0-signed (query)          54_248.94     1_720.06    55_969.00       0.1092             NaN        10.72
IVF-Binary-1024-nl158-np12-rf0-signed (query)         54_248.94     1_728.15    55_977.08       0.1083             NaN        10.72
IVF-Binary-1024-nl158-np17-rf0-signed (query)         54_248.94     1_792.68    56_041.62       0.1079             NaN        10.72
IVF-Binary-1024-nl158-np7-rf10-signed (query)         54_248.94     1_847.40    56_096.34       0.3132          1.0396        10.72
IVF-Binary-1024-nl158-np7-rf20-signed (query)         54_248.94     1_963.98    56_212.92       0.4522          1.0242        10.72
IVF-Binary-1024-nl158-np12-rf10-signed (query)        54_248.94     1_857.91    56_106.85       0.3114          1.0399        10.72
IVF-Binary-1024-nl158-np12-rf20-signed (query)        54_248.94     1_978.47    56_227.41       0.4521          1.0242        10.72
IVF-Binary-1024-nl158-np17-rf10-signed (query)        54_248.94     1_882.82    56_131.76       0.3099          1.0401        10.72
IVF-Binary-1024-nl158-np17-rf20-signed (query)        54_248.94     2_006.37    56_255.31       0.4503          1.0243        10.72
IVF-Binary-1024-nl158-signed (self)                   54_248.94     6_565.84    60_814.78       0.3115          1.0398        10.72
IVF-Binary-1024-nl223-np11-rf0-signed (query)         49_050.62     1_766.74    50_817.36       0.1092             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-signed (query)         49_050.62     1_764.45    50_815.07       0.1085             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-signed (query)         49_050.62     1_758.82    50_809.44       0.1081             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-signed (query)        49_050.62     1_855.27    50_905.89       0.3144          1.0394        10.98
IVF-Binary-1024-nl223-np11-rf20-signed (query)        49_050.62     1_996.79    51_047.41       0.4556          1.0238        10.98
IVF-Binary-1024-nl223-np14-rf10-signed (query)        49_050.62     1_863.93    50_914.54       0.3119          1.0398        10.98
IVF-Binary-1024-nl223-np14-rf20-signed (query)        49_050.62     1_997.67    51_048.29       0.4520          1.0242        10.98
IVF-Binary-1024-nl223-np21-rf10-signed (query)        49_050.62     1_878.17    50_928.78       0.3107          1.0400        10.98
IVF-Binary-1024-nl223-np21-rf20-signed (query)        49_050.62     2_013.07    51_063.68       0.4505          1.0243        10.98
IVF-Binary-1024-nl223-signed (self)                   49_050.62     6_138.03    55_188.65       0.3117          1.0397        10.98
IVF-Binary-1024-nl316-np15-rf0-signed (query)         49_675.32     1_771.50    51_446.82       0.1089             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-signed (query)         49_675.32     1_774.89    51_450.21       0.1086             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-signed (query)         49_675.32     1_801.08    51_476.40       0.1082             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-signed (query)        49_675.32     1_886.83    51_562.15       0.3131          1.0396        11.34
IVF-Binary-1024-nl316-np15-rf20-signed (query)        49_675.32     2_004.21    51_679.53       0.4544          1.0239        11.34
IVF-Binary-1024-nl316-np17-rf10-signed (query)        49_675.32     1_886.40    51_561.72       0.3118          1.0398        11.34
IVF-Binary-1024-nl316-np17-rf20-signed (query)        49_675.32     2_030.65    51_705.97       0.4525          1.0241        11.34
IVF-Binary-1024-nl316-np25-rf10-signed (query)        49_675.32     1_913.78    51_589.10       0.3105          1.0400        11.34
IVF-Binary-1024-nl316-np25-rf20-signed (query)        49_675.32     2_042.96    51_718.28       0.4510          1.0243        11.34
IVF-Binary-1024-nl316-signed (self)                   49_675.32     6_257.55    55_932.87       0.3120          1.0397        11.34
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
Exhaustive (query)                                         9.93     4_230.69     4_240.61       1.0000          1.0000        48.83
Exhaustive (self)                                          9.93    13_980.53    13_990.46       1.0000          1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_572.20       248.67     2_820.87       0.0796             NaN         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_572.20       347.56     2_919.75       0.2931          1.1776         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_572.20       452.64     3_024.84       0.4374          1.1037         1.78
ExhaustiveBinary-256-random (self)                     2_572.20     1_145.44     3_717.63       0.2929          1.1769         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_776.22       252.81     3_029.02       0.0827             NaN         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_776.22       367.43     3_143.65       0.1182          3.9214         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_776.22       471.19     3_247.40       0.1432          1.4338         1.78
ExhaustiveBinary-256-pca (self)                        2_776.22     1_203.24     3_979.46       0.1157          6.7403         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_045.58       445.17     5_490.75       0.1270             NaN         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_045.58       548.94     5_594.52       0.3949          1.1228         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_045.58       649.46     5_695.04       0.5575          1.0675         3.55
ExhaustiveBinary-512-random (self)                     5_045.58     1_808.36     6_853.94       0.3929          1.1237         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_354.83       462.56     5_817.39       0.0965             NaN         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_354.83       561.95     5_916.77       0.2973          1.1772         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_354.83       672.31     6_027.13       0.4534          1.0999         3.55
ExhaustiveBinary-512-pca (self)                        5_354.83     1_873.61     7_228.44       0.2988          1.1773         3.55
ExhaustiveBinary-1024-random_no_rr (query)             9_971.93       759.81    10_731.74       0.1773             NaN         7.10
ExhaustiveBinary-1024-random-rf10 (query)              9_971.93       867.41    10_839.34       0.5381          1.0724         7.10
ExhaustiveBinary-1024-random-rf20 (query)              9_971.93       990.67    10_962.60       0.7086          1.0349         7.10
ExhaustiveBinary-1024-random (self)                    9_971.93     2_872.08    12_844.01       0.5368          1.0726         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_468.94       769.00    11_237.94       0.1035             NaN         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_468.94       895.05    11_364.00       0.3158          1.1656         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_468.94     1_018.28    11_487.23       0.4741          1.0929         7.10
ExhaustiveBinary-1024-pca (self)                      10_468.94     2_947.22    13_416.16       0.3173          1.1657         7.10
ExhaustiveBinary-256-signed_no_rr (query)              2_538.22       246.79     2_785.01       0.0796             NaN         1.78
ExhaustiveBinary-256-signed-rf10 (query)               2_538.22       344.57     2_882.79       0.2931          1.1776         1.78
ExhaustiveBinary-256-signed-rf20 (query)               2_538.22       445.68     2_983.89       0.4374          1.1037         1.78
ExhaustiveBinary-256-signed (self)                     2_538.22     1_148.97     3_687.18       0.2929          1.1769         1.78
IVF-Binary-256-nl158-np7-rf0-random (query)            4_036.32       110.71     4_147.03       0.1023             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           4_036.32       115.81     4_152.13       0.0908             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           4_036.32       120.73     4_157.05       0.0812             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           4_036.32       167.26     4_203.58       0.3327          1.1541         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           4_036.32       214.80     4_251.13       0.4815          1.0896         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          4_036.32       171.62     4_207.95       0.3114          1.1674         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          4_036.32       223.63     4_259.95       0.4579          1.0975         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          4_036.32       181.98     4_218.30       0.2976          1.1762         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          4_036.32       237.36     4_273.68       0.4434          1.1027         1.93
IVF-Binary-256-nl158-random (self)                     4_036.32       521.80     4_558.12       0.3113          1.1669         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           3_010.36       125.90     3_136.26       0.0981             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           3_010.36       120.90     3_131.26       0.0871             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           3_010.36       124.20     3_134.57       0.0818             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          3_010.36       172.92     3_183.28       0.3258          1.1579         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          3_010.36       222.12     3_232.48       0.4750          1.0915         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          3_010.36       176.83     3_187.19       0.3093          1.1684         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          3_010.36       230.81     3_241.17       0.4555          1.0983         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          3_010.36       186.02     3_196.38       0.2997          1.1749         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          3_010.36       244.48     3_254.84       0.4446          1.1021         2.00
IVF-Binary-256-nl223-random (self)                     3_010.36       538.65     3_549.02       0.3094          1.1678         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           3_410.43       127.43     3_537.86       0.0902             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           3_410.43       127.89     3_538.32       0.0873             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           3_410.43       137.88     3_548.31       0.0830             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          3_410.43       253.77     3_664.20       0.3190          1.1618         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          3_410.43       273.79     3_684.22       0.4684          1.0938         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          3_410.43       201.35     3_611.79       0.3139          1.1649         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          3_410.43       253.82     3_664.25       0.4616          1.0960         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          3_410.43       217.44     3_627.87       0.3026          1.1726         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          3_410.43       280.90     3_691.33       0.4481          1.1006         2.09
IVF-Binary-256-nl316-random (self)                     3_410.43       645.62     4_056.05       0.3137          1.1645         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               4_420.91       121.02     4_541.92       0.0939             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              4_420.91       123.83     4_544.74       0.0914             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              4_420.91       133.13     4_554.03       0.0903             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              4_420.91       188.35     4_609.25       0.2815          1.1901         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              4_420.91       247.11     4_668.01       0.4291          1.1101         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             4_420.91       205.12     4_626.02       0.2368          1.2309         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             4_420.91       263.51     4_684.41       0.3498          1.1457         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             4_420.91       207.87     4_628.78       0.2144          1.2578         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             4_420.91       280.00     4_700.91       0.3095          1.1698         1.93
IVF-Binary-256-nl158-pca (self)                        4_420.91       643.02     5_063.93       0.2370          1.2324         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              3_198.95       123.02     3_321.96       0.0937             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              3_198.95       125.36     3_324.31       0.0921             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              3_198.95       130.09     3_329.04       0.0906             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             3_198.95       192.28     3_391.23       0.2741          1.1960         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             3_198.95       251.79     3_450.74       0.4164          1.1152         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             3_198.95       200.38     3_399.33       0.2535          1.2142         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             3_198.95       262.37     3_461.32       0.3798          1.1311         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             3_198.95       206.83     3_405.78       0.2220          1.2482         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             3_198.95       285.17     3_484.12       0.3231          1.1612         2.00
IVF-Binary-256-nl223-pca (self)                        3_198.95       624.60     3_823.54       0.2542          1.2150         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              3_424.09       145.25     3_569.35       0.0934             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              3_424.09       134.72     3_558.81       0.0927             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              3_424.09       136.02     3_560.11       0.0910             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             3_424.09       201.80     3_625.89       0.2767          1.1930         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             3_424.09       260.00     3_684.09       0.4210          1.1124         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             3_424.09       203.79     3_627.88       0.2653          1.2024         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             3_424.09       268.14     3_692.23       0.4009          1.1206         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             3_424.09       208.89     3_632.98       0.2333          1.2342         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             3_424.09       276.92     3_701.01       0.3440          1.1485         2.09
IVF-Binary-256-nl316-pca (self)                        3_424.09       629.02     4_053.11       0.2664          1.2029         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_593.98       207.41     6_801.39       0.1365             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_593.98       213.16     6_807.13       0.1314             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_593.98       231.30     6_825.28       0.1280             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_593.98       263.34     6_857.32       0.4195          1.1128         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_593.98       314.06     6_908.04       0.5824          1.0615         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_593.98       274.14     6_868.12       0.4052          1.1187         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_593.98       334.06     6_928.04       0.5669          1.0653         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_593.98       291.70     6_885.68       0.3969          1.1224         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_593.98       347.65     6_941.63       0.5596          1.0672         3.71
IVF-Binary-512-nl158-random (self)                     6_593.98       870.81     7_464.79       0.4039          1.1194         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_562.68       211.08     5_773.76       0.1345             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_562.68       220.95     5_783.63       0.1304             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_562.68       223.54     5_786.22       0.1281             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_562.68       272.88     5_835.56       0.4149          1.1145         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_562.68       322.08     5_884.76       0.5792          1.0622         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_562.68       277.33     5_840.01       0.4037          1.1192         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_562.68       332.92     5_895.60       0.5672          1.0653         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_562.68       294.67     5_857.35       0.3977          1.1218         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_562.68       371.90     5_934.58       0.5610          1.0669         3.77
IVF-Binary-512-nl223-random (self)                     5_562.68       889.27     6_451.95       0.4027          1.1198         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           5_753.44       217.87     5_971.31       0.1326             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           5_753.44       234.15     5_987.59       0.1312             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           5_753.44       228.60     5_982.04       0.1288             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          5_753.44       280.82     6_034.25       0.4113          1.1160         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          5_753.44       329.10     6_082.54       0.5748          1.0633         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          5_753.44       289.07     6_042.50       0.4073          1.1177         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          5_753.44       335.59     6_089.02       0.5696          1.0646         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          5_753.44       293.47     6_046.91       0.4001          1.1208         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          5_753.44       352.27     6_105.71       0.5618          1.0666         3.86
IVF-Binary-512-nl316-random (self)                     5_753.44       915.56     6_669.00       0.4059          1.1185         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               6_960.50       214.59     7_175.09       0.0978             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              6_960.50       224.37     7_184.87       0.0966             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              6_960.50       244.74     7_205.23       0.0965             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              6_960.50       284.64     7_245.13       0.3062          1.1716         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              6_960.50       337.07     7_297.56       0.4660          1.0957         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             6_960.50       292.51     7_253.01       0.2974          1.1771         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             6_960.50       351.88     7_312.38       0.4534          1.0999         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             6_960.50       311.40     7_271.89       0.2974          1.1771         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             6_960.50       372.59     7_333.08       0.4534          1.0999         3.71
IVF-Binary-512-nl158-pca (self)                        6_960.50       951.64     7_912.13       0.2989          1.1772         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              5_844.53       222.26     6_066.79       0.0973             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              5_844.53       226.73     6_071.26       0.0966             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              5_844.53       241.43     6_085.96       0.0965             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             5_844.53       291.59     6_136.13       0.3028          1.1736         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             5_844.53       356.28     6_200.81       0.4616          1.0971         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             5_844.53       297.33     6_141.86       0.2978          1.1769         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             5_844.53       359.16     6_203.70       0.4540          1.0997         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             5_844.53       312.91     6_157.45       0.2974          1.1771         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             5_844.53       376.52     6_221.06       0.4535          1.0999         3.77
IVF-Binary-512-nl223-pca (self)                        5_844.53       961.95     6_806.48       0.2991          1.1770         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              5_976.59       224.99     6_201.57       0.0971             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              5_976.59       236.41     6_212.99       0.0967             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              5_976.59       233.81     6_210.39       0.0965             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             5_976.59       294.74     6_271.33       0.3015          1.1745         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             5_976.59       350.83     6_327.42       0.4592          1.0979         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             5_976.59       293.74     6_270.33       0.2987          1.1763         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             5_976.59       355.11     6_331.70       0.4552          1.0993         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             5_976.59       308.58     6_285.17       0.2975          1.1771         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             5_976.59       371.29     6_347.87       0.4535          1.0999         3.86
IVF-Binary-512-nl316-pca (self)                        5_976.59       951.41     6_927.99       0.3000          1.1764         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_546.60       391.39    11_937.99       0.1846             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_546.60       402.53    11_949.14       0.1806             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_546.60       416.85    11_963.45       0.1781             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_546.60       457.10    12_003.70       0.5530          1.0684         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_546.60       511.03    12_057.63       0.7207          1.0329         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_546.60       468.69    12_015.29       0.5441          1.0708         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_546.60       527.83    12_074.43       0.7125          1.0343         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_546.60       493.85    12_040.45       0.5400          1.0719         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_546.60       551.84    12_098.44       0.7094          1.0348         7.26
IVF-Binary-1024-nl158-random (self)                   11_546.60     1_526.30    13_072.90       0.5426          1.0711         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_444.08       405.12    10_849.20       0.1829             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_444.08       403.78    10_847.86       0.1800             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_444.08       412.48    10_856.56       0.1782             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_444.08       457.51    10_901.59       0.5508          1.0688         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_444.08       512.93    10_957.01       0.7205          1.0329         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_444.08       470.85    10_914.92       0.5438          1.0709         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_444.08       523.90    10_967.98       0.7139          1.0340         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_444.08       488.15    10_932.23       0.5403          1.0719         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_444.08       548.74    10_992.82       0.7110          1.0345         7.32
IVF-Binary-1024-nl223-random (self)                   10_444.08     1_510.31    11_954.39       0.5425          1.0711         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         10_595.20       403.09    10_998.29       0.1813             NaN         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)         10_595.20       412.68    11_007.87       0.1803             NaN         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)         10_595.20       423.72    11_018.92       0.1784             NaN         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)        10_595.20       467.69    11_062.89       0.5480          1.0697         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)        10_595.20       523.68    11_118.87       0.7182          1.0333         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)        10_595.20       470.00    11_065.19       0.5453          1.0705         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)        10_595.20       530.22    11_125.41       0.7151          1.0338         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)        10_595.20       488.83    11_084.03       0.5407          1.0718         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)        10_595.20       547.91    11_143.11       0.7111          1.0345         7.41
IVF-Binary-1024-nl316-random (self)                   10_595.20     1_524.72    12_119.92       0.5443          1.0706         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)             11_984.49       402.65    12_387.14       0.1049             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            11_984.49       417.57    12_402.06       0.1036             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            11_984.49       427.86    12_412.35       0.1035             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            11_984.49       479.16    12_463.65       0.3243          1.1605         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            11_984.49       534.58    12_519.07       0.4867          1.0889         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           11_984.49       488.15    12_472.64       0.3156          1.1656         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           11_984.49       550.75    12_535.24       0.4742          1.0929         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           11_984.49       510.72    12_495.20       0.3156          1.1656         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           11_984.49       576.49    12_560.98       0.4742          1.0929         7.26
IVF-Binary-1024-nl158-pca (self)                      11_984.49     1_588.34    13_572.83       0.3173          1.1657         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            10_919.85       410.27    11_330.11       0.1044             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            10_919.85       417.36    11_337.21       0.1036             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            10_919.85       430.61    11_350.46       0.1035             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           10_919.85       497.66    11_417.51       0.3211          1.1623         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           10_919.85       538.52    11_458.37       0.4821          1.0904         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           10_919.85       491.27    11_411.11       0.3161          1.1653         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           10_919.85       553.49    11_473.33       0.4747          1.0927         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           10_919.85       510.69    11_430.54       0.3158          1.1656         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           10_919.85       573.99    11_493.83       0.4741          1.0929         7.32
IVF-Binary-1024-nl223-pca (self)                      10_919.85     1_600.08    12_519.92       0.3176          1.1655         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            11_145.20       422.27    11_567.47       0.1041             NaN         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            11_145.20       423.73    11_568.93       0.1037             NaN         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            11_145.20       445.24    11_590.43       0.1035             NaN         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           11_145.20       492.15    11_637.34       0.3198          1.1632         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           11_145.20       553.40    11_698.59       0.4799          1.0911         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           11_145.20       497.46    11_642.66       0.3170          1.1648         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           11_145.20       555.63    11_700.83       0.4759          1.0923         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           11_145.20       510.66    11_655.86       0.3159          1.1656         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           11_145.20       578.56    11_723.75       0.4741          1.0929         7.42
IVF-Binary-1024-nl316-pca (self)                      11_145.20     1_616.21    12_761.41       0.3185          1.1649         7.42
IVF-Binary-256-nl158-np7-rf0-signed (query)            4_076.48       111.65     4_188.13       0.1023             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-signed (query)           4_076.48       118.90     4_195.38       0.0908             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-signed (query)           4_076.48       122.72     4_199.20       0.0812             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-signed (query)           4_076.48       167.88     4_244.36       0.3327          1.1541         1.93
IVF-Binary-256-nl158-np7-rf20-signed (query)           4_076.48       214.22     4_290.70       0.4815          1.0896         1.93
IVF-Binary-256-nl158-np12-rf10-signed (query)          4_076.48       174.19     4_250.67       0.3114          1.1674         1.93
IVF-Binary-256-nl158-np12-rf20-signed (query)          4_076.48       226.28     4_302.76       0.4579          1.0975         1.93
IVF-Binary-256-nl158-np17-rf10-signed (query)          4_076.48       184.04     4_260.52       0.2976          1.1762         1.93
IVF-Binary-256-nl158-np17-rf20-signed (query)          4_076.48       239.30     4_315.78       0.4434          1.1027         1.93
IVF-Binary-256-nl158-signed (self)                     4_076.48       527.00     4_603.48       0.3113          1.1669         1.93
IVF-Binary-256-nl223-np11-rf0-signed (query)           3_000.94       118.37     3_119.31       0.0981             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-signed (query)           3_000.94       120.12     3_121.06       0.0871             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-signed (query)           3_000.94       126.77     3_127.72       0.0818             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-signed (query)          3_000.94       172.61     3_173.55       0.3258          1.1579         2.00
IVF-Binary-256-nl223-np11-rf20-signed (query)          3_000.94       222.87     3_223.81       0.4750          1.0915         2.00
IVF-Binary-256-nl223-np14-rf10-signed (query)          3_000.94       187.10     3_188.05       0.3093          1.1684         2.00
IVF-Binary-256-nl223-np14-rf20-signed (query)          3_000.94       231.83     3_232.78       0.4555          1.0983         2.00
IVF-Binary-256-nl223-np21-rf10-signed (query)          3_000.94       186.65     3_187.60       0.2997          1.1749         2.00
IVF-Binary-256-nl223-np21-rf20-signed (query)          3_000.94       243.94     3_244.88       0.4446          1.1021         2.00
IVF-Binary-256-nl223-signed (self)                     3_000.94       537.57     3_538.52       0.3094          1.1678         2.00
IVF-Binary-256-nl316-np15-rf0-signed (query)           3_221.41       124.83     3_346.24       0.0902             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-signed (query)           3_221.41       123.91     3_345.32       0.0873             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-signed (query)           3_221.41       129.43     3_350.84       0.0830             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-signed (query)          3_221.41       180.50     3_401.91       0.3190          1.1618         2.09
IVF-Binary-256-nl316-np15-rf20-signed (query)          3_221.41       231.05     3_452.46       0.4684          1.0938         2.09
IVF-Binary-256-nl316-np17-rf10-signed (query)          3_221.41       180.41     3_401.83       0.3139          1.1649         2.09
IVF-Binary-256-nl316-np17-rf20-signed (query)          3_221.41       233.20     3_454.62       0.4616          1.0960         2.09
IVF-Binary-256-nl316-np25-rf10-signed (query)          3_221.41       190.81     3_412.22       0.3026          1.1726         2.09
IVF-Binary-256-nl316-np25-rf20-signed (query)          3_221.41       245.97     3_467.38       0.4481          1.1006         2.09
IVF-Binary-256-nl316-signed (self)                     3_221.41       553.04     3_774.45       0.3137          1.1645         2.09
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
Exhaustive (query)                                        20.88     9_658.09     9_678.97       1.0000          1.0000        97.66
Exhaustive (self)                                         20.88    32_289.17    32_310.05       1.0000          1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_678.47       353.25     6_031.72       0.0472             NaN         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_678.47       483.64     6_162.11       0.2012          1.1663         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_678.47       633.48     6_311.96       0.3224          1.1017         2.03
ExhaustiveBinary-256-random (self)                     5_678.47     1_549.14     7_227.62       0.2020          1.1654         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_084.55       360.58     6_445.13       0.1539             NaN         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_084.55       494.49     6_579.05       0.3795          1.0909         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_084.55       626.87     6_711.42       0.4855          1.0578         2.03
ExhaustiveBinary-256-pca (self)                        6_084.55     1_631.11     7_715.66       0.3796          1.0913         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_085.76       675.37    11_761.13       0.0860             NaN         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_085.76       775.97    11_861.73       0.2648          1.1242         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_085.76       900.98    11_986.74       0.3980          1.0752         4.05
ExhaustiveBinary-512-random (self)                    11_085.76     2_570.54    13_656.30       0.2643          1.1240         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_736.06       669.85    12_405.91       0.1171             NaN         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_736.06       801.50    12_537.56       0.2880         21.4934         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_736.06       936.56    12_672.63       0.3835          1.0941         4.05
ExhaustiveBinary-512-pca (self)                       11_736.06     2_669.71    14_405.77       0.2876         22.0894         4.05
ExhaustiveBinary-1024-random_no_rr (query)            22_068.77     1_181.49    23_250.26       0.1148             NaN         8.10
ExhaustiveBinary-1024-random-rf10 (query)             22_068.77     1_317.93    23_386.71       0.3455          1.0925         8.10
ExhaustiveBinary-1024-random-rf20 (query)             22_068.77     1_470.57    23_539.34       0.4981          1.0533         8.10
ExhaustiveBinary-1024-random (self)                   22_068.77     4_410.99    26_479.76       0.3445          1.0928         8.10
ExhaustiveBinary-1024-pca_no_rr (query)               23_072.14     1_202.92    24_275.06       0.2416             NaN         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                23_072.14     1_352.63    24_424.77       0.6860          1.0240         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                23_072.14     1_502.17    24_574.31       0.8390          1.0095         8.11
ExhaustiveBinary-1024-pca (self)                      23_072.14     4_478.22    27_550.36       0.6864          1.0239         8.11
ExhaustiveBinary-512-signed_no_rr (query)             11_110.13       658.94    11_769.07       0.0860             NaN         4.05
ExhaustiveBinary-512-signed-rf10 (query)              11_110.13       796.97    11_907.10       0.2648          1.1242         4.05
ExhaustiveBinary-512-signed-rf20 (query)              11_110.13       907.42    12_017.55       0.3980          1.0752         4.05
ExhaustiveBinary-512-signed (self)                    11_110.13     2_609.57    13_719.70       0.2643          1.1240         4.05
IVF-Binary-256-nl158-np7-rf0-random (query)            8_775.42       228.93     9_004.35       0.0675             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           8_775.42       233.29     9_008.71       0.0586             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           8_775.42       241.27     9_016.70       0.0493             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           8_775.42       309.63     9_085.05       0.2432          1.1377         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           8_775.42       387.57     9_162.99       0.3698          1.0844         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          8_775.42       313.68     9_089.10       0.2255          1.1497         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          8_775.42       396.57     9_171.99       0.3483          1.0921         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          8_775.42       321.76     9_097.18       0.2074          1.1635         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          8_775.42       408.02     9_183.44       0.3302          1.0996         2.34
IVF-Binary-256-nl158-random (self)                     8_775.42       965.28     9_740.70       0.2261          1.1490         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           6_465.10       241.93     6_707.03       0.0642             NaN         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)           6_465.10       241.21     6_706.30       0.0545             NaN         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)           6_465.10       248.06     6_713.16       0.0498             NaN         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)          6_465.10       318.25     6_783.35       0.2334          1.1444         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)          6_465.10       399.54     6_864.64       0.3573          1.0890         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)          6_465.10       331.83     6_796.92       0.2172          1.1566         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)          6_465.10       405.49     6_870.59       0.3403          1.0960         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)          6_465.10       331.92     6_797.02       0.2092          1.1621         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)          6_465.10       415.68     6_880.78       0.3311          1.0992         2.46
IVF-Binary-256-nl223-random (self)                     6_465.10       994.41     7_459.51       0.2181          1.1556         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)           6_852.69       253.31     7_106.00       0.0584             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           6_852.69       259.36     7_112.05       0.0547             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           6_852.69       261.55     7_114.24       0.0517             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          6_852.69       336.69     7_189.38       0.2294          1.1477         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          6_852.69       415.64     7_268.34       0.3567          1.0896         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          6_852.69       329.85     7_182.54       0.2238          1.1515         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          6_852.69       419.26     7_271.95       0.3497          1.0919         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          6_852.69       343.28     7_195.98       0.2156          1.1575         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          6_852.69       425.64     7_278.33       0.3381          1.0961         2.65
IVF-Binary-256-nl316-random (self)                     6_852.69     1_040.36     7_893.06       0.2241          1.1510         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               9_267.89       249.37     9_517.25       0.2085             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              9_267.89       240.87     9_508.76       0.2050             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              9_267.89       246.31     9_514.20       0.2025             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              9_267.89       331.86     9_599.75       0.6076          1.0348         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              9_267.89       417.69     9_685.57       0.7667          1.0158         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             9_267.89       339.87     9_607.76       0.5909          1.0374         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             9_267.89       424.27     9_692.16       0.7466          1.0178         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             9_267.89       347.08     9_614.97       0.5763          1.0398         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             9_267.89       441.07     9_708.95       0.7278          1.0197         2.34
IVF-Binary-256-nl158-pca (self)                        9_267.89     1_063.53    10_331.42       0.5917          1.0371         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              7_007.85       245.90     7_253.75       0.2082             NaN         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              7_007.85       247.60     7_255.45       0.2065             NaN         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              7_007.85       253.29     7_261.14       0.2030             NaN         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             7_007.85       339.52     7_347.36       0.6061          1.0350         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             7_007.85       423.46     7_431.30       0.7661          1.0159         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             7_007.85       342.63     7_350.47       0.5976          1.0363         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             7_007.85       469.15     7_477.00       0.7555          1.0169         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             7_007.85       351.13     7_358.97       0.5790          1.0393         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             7_007.85       448.12     7_455.96       0.7325          1.0192         2.47
IVF-Binary-256-nl223-pca (self)                        7_007.85     1_082.18     8_090.03       0.5986          1.0361         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              7_339.09       258.28     7_597.37       0.2082             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              7_339.09       263.70     7_602.79       0.2074             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              7_339.09       267.28     7_606.37       0.2045             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             7_339.09       354.49     7_693.58       0.6077          1.0348         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             7_339.09       440.70     7_779.79       0.7686          1.0157         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             7_339.09       355.22     7_694.31       0.6033          1.0355         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             7_339.09       442.43     7_781.53       0.7629          1.0162         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             7_339.09       361.68     7_700.77       0.5873          1.0380         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             7_339.09       455.56     7_794.65       0.7424          1.0182         2.65
IVF-Binary-256-nl316-pca (self)                        7_339.09     1_120.77     8_459.86       0.6043          1.0352         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           14_378.70       430.18    14_808.88       0.0977             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          14_378.70       444.10    14_822.80       0.0934             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          14_378.70       446.06    14_824.75       0.0872             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          14_378.70       510.82    14_889.52       0.2879          1.1143         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          14_378.70       588.98    14_967.68       0.4248          1.0690         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         14_378.70       523.42    14_902.12       0.2772          1.1189         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         14_378.70       607.68    14_986.38       0.4113          1.0721         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         14_378.70       535.47    14_914.17       0.2686          1.1230         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         14_378.70       623.79    15_002.49       0.4024          1.0744         4.36
IVF-Binary-512-nl158-random (self)                    14_378.70     1_672.67    16_051.36       0.2766          1.1188         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          12_096.51       437.58    12_534.09       0.0946             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          12_096.51       445.63    12_542.13       0.0906             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          12_096.51       458.17    12_554.68       0.0882             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         12_096.51       520.29    12_616.80       0.2815          1.1166         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         12_096.51       595.96    12_692.47       0.4178          1.0706         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         12_096.51       527.47    12_623.98       0.2738          1.1202         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         12_096.51       613.99    12_710.50       0.4088          1.0728         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         12_096.51       535.98    12_632.48       0.2690          1.1224         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         12_096.51       622.97    12_719.48       0.4029          1.0743         4.49
IVF-Binary-512-nl223-random (self)                    12_096.51     1_698.17    13_794.68       0.2736          1.1201         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          12_300.35       451.89    12_752.23       0.0928             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          12_300.35       454.60    12_754.95       0.0914             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          12_300.35       471.12    12_771.47       0.0894             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         12_300.35       536.04    12_836.38       0.2818          1.1164         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         12_300.35       617.00    12_917.35       0.4179          1.0705         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         12_300.35       537.25    12_837.60       0.2784          1.1179         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         12_300.35       619.36    12_919.71       0.4133          1.0716         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         12_300.35       549.23    12_849.57       0.2728          1.1206         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         12_300.35       631.76    12_932.11       0.4065          1.0733         4.67
IVF-Binary-512-nl316-random (self)                    12_300.35     1_726.01    14_026.36       0.2777          1.1180         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              15_088.52       442.36    15_530.88       0.2324             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             15_088.52       451.36    15_539.88       0.2185             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             15_088.52       459.58    15_548.11       0.2074             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             15_088.52       533.76    15_622.28       0.6593          1.0274         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             15_088.52       619.28    15_707.81       0.8106          1.0119         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            15_088.52       547.14    15_635.67       0.6165          1.0336         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            15_088.52       646.39    15_734.91       0.7678          1.0156         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            15_088.52       567.51    15_656.04       0.5819          1.0391         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            15_088.52       670.81    15_759.33       0.7304          1.0193         4.36
IVF-Binary-512-nl158-pca (self)                       15_088.52     1_792.26    16_880.79       0.6164          1.0336         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             12_757.53       452.92    13_210.45       0.2304             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             12_757.53       473.07    13_230.60       0.2235             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             12_757.53       461.91    13_219.44       0.2103             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            12_757.53       543.62    13_301.15       0.6552          1.0280         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            12_757.53       626.61    13_384.14       0.8087          1.0120         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            12_757.53       553.08    13_310.61       0.6324          1.0311         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            12_757.53       646.38    13_403.91       0.7856          1.0140         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            12_757.53       561.94    13_319.47       0.5919          1.0374         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            12_757.53       666.85    13_424.38       0.7416          1.0181         4.49
IVF-Binary-512-nl223-pca (self)                       12_757.53     1_788.84    14_546.37       0.6334          1.0310         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             12_858.78       467.59    13_326.38       0.2313             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             12_858.78       466.67    13_325.45       0.2274             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             12_858.78       478.92    13_337.70       0.2157             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            12_858.78       572.18    13_430.96       0.6597          1.0273         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            12_858.78       654.21    13_512.99       0.8134          1.0116         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            12_858.78       561.87    13_420.65       0.6481          1.0289         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            12_858.78       656.81    13_515.59       0.8017          1.0126         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            12_858.78       576.59    13_435.37       0.6084          1.0347         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            12_858.78       667.38    13_526.16       0.7600          1.0163         4.67
IVF-Binary-512-nl316-pca (self)                       12_858.78     1_818.05    14_676.83       0.6482          1.0289         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          25_330.21       820.76    26_150.96       0.1197             NaN         8.41
IVF-Binary-1024-nl158-np12-rf0-random (query)         25_330.21       839.79    26_170.00       0.1173             NaN         8.41
IVF-Binary-1024-nl158-np17-rf0-random (query)         25_330.21       850.20    26_180.40       0.1153             NaN         8.41
IVF-Binary-1024-nl158-np7-rf10-random (query)         25_330.21       913.38    26_243.59       0.3607          1.0878         8.41
IVF-Binary-1024-nl158-np7-rf20-random (query)         25_330.21     1_003.34    26_333.55       0.5146          1.0503         8.41
IVF-Binary-1024-nl158-np12-rf10-random (query)        25_330.21       928.03    26_258.24       0.3529          1.0902         8.41
IVF-Binary-1024-nl158-np12-rf20-random (query)        25_330.21     1_005.13    26_335.33       0.5055          1.0519         8.41
IVF-Binary-1024-nl158-np17-rf10-random (query)        25_330.21       950.63    26_280.83       0.3485          1.0917         8.41
IVF-Binary-1024-nl158-np17-rf20-random (query)        25_330.21     1_031.18    26_361.38       0.5006          1.0529         8.41
IVF-Binary-1024-nl158-random (self)                   25_330.21     3_026.79    28_357.00       0.3520          1.0905         8.41
IVF-Binary-1024-nl223-np11-rf0-random (query)         22_938.26       836.74    23_775.00       0.1184             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         22_938.26       845.47    23_783.74       0.1166             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         22_938.26       862.43    23_800.70       0.1155             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        22_938.26       926.66    23_864.92       0.3560          1.0892         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        22_938.26     1_016.46    23_954.73       0.5090          1.0513         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        22_938.26       934.46    23_872.72       0.3513          1.0908         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        22_938.26     1_021.36    23_959.63       0.5032          1.0523         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        22_938.26       953.03    23_891.30       0.3480          1.0919         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        22_938.26     1_039.44    23_977.70       0.4992          1.0531         8.54
IVF-Binary-1024-nl223-random (self)                   22_938.26     3_039.32    25_977.58       0.3504          1.0910         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         23_349.40       881.72    24_231.12       0.1184             NaN         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)         23_349.40       866.44    24_215.84       0.1177             NaN         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)         23_349.40       867.12    24_216.52       0.1167             NaN         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)        23_349.40       944.16    24_293.56       0.3561          1.0891         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)        23_349.40     1_026.46    24_375.86       0.5101          1.0511         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)        23_349.40       955.43    24_304.83       0.3535          1.0900         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)        23_349.40     1_052.89    24_402.29       0.5064          1.0517         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)        23_349.40       962.84    24_312.23       0.3495          1.0913         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)        23_349.40     1_062.64    24_412.04       0.5022          1.0525         8.72
IVF-Binary-1024-nl316-random (self)                   23_349.40     3_076.21    26_425.61       0.3524          1.0903         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)             26_317.22       842.23    27_159.44       0.2423             NaN         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            26_317.22       855.25    27_172.46       0.2418             NaN         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            26_317.22       871.25    27_188.46       0.2417             NaN         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            26_317.22       937.59    27_254.81       0.6848          1.0241         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            26_317.22     1_019.42    27_336.63       0.8347          1.0099         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           26_317.22       949.49    27_266.71       0.6863          1.0240         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           26_317.22     1_046.29    27_363.51       0.8391          1.0095         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           26_317.22       974.30    27_291.52       0.6863          1.0240         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           26_317.22     1_065.19    27_382.40       0.8391          1.0095         8.42
IVF-Binary-1024-nl158-pca (self)                      26_317.22     3_140.13    29_457.34       0.6865          1.0239         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            23_948.88       855.35    24_804.23       0.2423             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            23_948.88       866.48    24_815.36       0.2420             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            23_948.88       874.57    24_823.45       0.2419             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           23_948.88       944.47    24_893.35       0.6864          1.0239         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           23_948.88     1_030.33    24_979.21       0.8383          1.0096         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           23_948.88       957.38    24_906.26       0.6864          1.0240         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           23_948.88     1_041.11    24_989.99       0.8391          1.0095         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           23_948.88       977.74    24_926.62       0.6862          1.0240         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           23_948.88     1_069.44    25_018.32       0.8390          1.0095         8.54
IVF-Binary-1024-nl223-pca (self)                      23_948.88     3_124.39    27_073.27       0.6866          1.0239         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            24_215.40       869.87    25_085.27       0.2422             NaN         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            24_215.40       872.98    25_088.38       0.2419             NaN         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            24_215.40       889.66    25_105.06       0.2417             NaN         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           24_215.40       960.79    25_176.19       0.6866          1.0239         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           24_215.40     1_043.72    25_259.12       0.8392          1.0095         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           24_215.40       965.23    25_180.63       0.6864          1.0239         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           24_215.40     1_055.13    25_270.53       0.8392          1.0095         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           24_215.40       978.56    25_193.96       0.6861          1.0240         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           24_215.40     1_068.05    25_283.45       0.8390          1.0095         8.73
IVF-Binary-1024-nl316-pca (self)                      24_215.40     3_171.52    27_386.92       0.6867          1.0239         8.73
IVF-Binary-512-nl158-np7-rf0-signed (query)           14_178.54       422.72    14_601.26       0.0977             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-signed (query)          14_178.54       433.64    14_612.18       0.0934             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-signed (query)          14_178.54       440.69    14_619.22       0.0872             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-signed (query)          14_178.54       506.07    14_684.61       0.2879          1.1143         4.36
IVF-Binary-512-nl158-np7-rf20-signed (query)          14_178.54       579.78    14_758.32       0.4248          1.0690         4.36
IVF-Binary-512-nl158-np12-rf10-signed (query)         14_178.54       517.31    14_695.85       0.2772          1.1189         4.36
IVF-Binary-512-nl158-np12-rf20-signed (query)         14_178.54       594.94    14_773.48       0.4113          1.0721         4.36
IVF-Binary-512-nl158-np17-rf10-signed (query)         14_178.54       532.91    14_711.45       0.2686          1.1230         4.36
IVF-Binary-512-nl158-np17-rf20-signed (query)         14_178.54       616.64    14_795.18       0.4024          1.0744         4.36
IVF-Binary-512-nl158-signed (self)                    14_178.54     1_646.62    15_825.16       0.2766          1.1188         4.36
IVF-Binary-512-nl223-np11-rf0-signed (query)          11_916.71       435.93    12_352.63       0.0946             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-signed (query)          11_916.71       440.44    12_357.15       0.0906             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-signed (query)          11_916.71       451.75    12_368.45       0.0882             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-signed (query)         11_916.71       519.15    12_435.85       0.2815          1.1166         4.49
IVF-Binary-512-nl223-np11-rf20-signed (query)         11_916.71       602.14    12_518.85       0.4178          1.0706         4.49
IVF-Binary-512-nl223-np14-rf10-signed (query)         11_916.71       528.50    12_445.21       0.2738          1.1202         4.49
IVF-Binary-512-nl223-np14-rf20-signed (query)         11_916.71       611.71    12_528.42       0.4088          1.0728         4.49
IVF-Binary-512-nl223-np21-rf10-signed (query)         11_916.71       537.57    12_454.28       0.2690          1.1224         4.49
IVF-Binary-512-nl223-np21-rf20-signed (query)         11_916.71       628.12    12_544.82       0.4029          1.0743         4.49
IVF-Binary-512-nl223-signed (self)                    11_916.71     1_688.76    13_605.46       0.2736          1.1201         4.49
IVF-Binary-512-nl316-np15-rf0-signed (query)          12_302.24       452.22    12_754.45       0.0928             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-signed (query)          12_302.24       454.42    12_756.65       0.0914             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-signed (query)          12_302.24       463.54    12_765.78       0.0894             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-signed (query)         12_302.24       535.97    12_838.21       0.2818          1.1164         4.67
IVF-Binary-512-nl316-np15-rf20-signed (query)         12_302.24       614.14    12_916.37       0.4179          1.0705         4.67
IVF-Binary-512-nl316-np17-rf10-signed (query)         12_302.24       537.43    12_839.67       0.2784          1.1179         4.67
IVF-Binary-512-nl316-np17-rf20-signed (query)         12_302.24       615.91    12_918.15       0.4133          1.0716         4.67
IVF-Binary-512-nl316-np25-rf10-signed (query)         12_302.24       547.34    12_849.57       0.2728          1.1206         4.67
IVF-Binary-512-nl316-np25-rf20-signed (query)         12_302.24       632.64    12_934.88       0.4065          1.0733         4.67
IVF-Binary-512-nl316-signed (self)                    12_302.24     1_727.34    14_029.58       0.2777          1.1180         4.67
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
Exhaustive (query)                                        42.83    22_058.20    22_101.02       1.0000          1.0000       195.31
Exhaustive (self)                                         42.83    73_294.03    73_336.86       1.0000          1.0000       195.31
ExhaustiveBinary-256-random_no_rr (query)             11_949.32       583.42    12_532.75       0.0290             NaN         2.53
ExhaustiveBinary-256-random-rf10 (query)              11_949.32       713.26    12_662.58       0.1514          1.1441         2.53
ExhaustiveBinary-256-random-rf20 (query)              11_949.32       877.54    12_826.86       0.2535          1.0923         2.53
ExhaustiveBinary-256-random (self)                    11_949.32     2_358.23    14_307.55       0.1523          1.1428         2.53
ExhaustiveBinary-256-pca_no_rr (query)                12_979.29       577.89    13_557.17       0.1730             NaN         2.53
ExhaustiveBinary-256-pca-rf10 (query)                 12_979.29       750.58    13_729.87       0.4383          1.0451         2.53
ExhaustiveBinary-256-pca-rf20 (query)                 12_979.29       930.81    13_910.09       0.5559          1.0281         2.53
ExhaustiveBinary-256-pca (self)                       12_979.29     2_463.25    15_442.53       0.4397          1.0450         2.53
ExhaustiveBinary-512-random_no_rr (query)             23_435.53     1_086.12    24_521.65       0.0628             NaN         5.05
ExhaustiveBinary-512-random-rf10 (query)              23_435.53     1_238.79    24_674.32       0.1942          1.1100         5.05
ExhaustiveBinary-512-random-rf20 (query)              23_435.53     1_409.03    24_844.56       0.3026          1.0701         5.05
ExhaustiveBinary-512-random (self)                    23_435.53     4_136.78    27_572.31       0.1954          1.1092         5.05
ExhaustiveBinary-512-pca_no_rr (query)                24_467.44     1_100.87    25_568.31       0.1519             NaN         5.06
ExhaustiveBinary-512-pca-rf10 (query)                 24_467.44     1_270.41    25_737.84       0.3855          1.1037         5.06
ExhaustiveBinary-512-pca-rf20 (query)                 24_467.44     1_444.90    25_912.33       0.4977          1.0366         5.06
ExhaustiveBinary-512-pca (self)                       24_467.44     4_198.97    28_666.41       0.3861          1.1000         5.06
ExhaustiveBinary-1024-random_no_rr (query)            46_945.74     2_028.12    48_973.86       0.0893             NaN        10.10
ExhaustiveBinary-1024-random-rf10 (query)             46_945.74     2_222.01    49_167.75       0.2382          1.0892        10.10
ExhaustiveBinary-1024-random-rf20 (query)             46_945.74     2_395.49    49_341.23       0.3610          1.0558        10.10
ExhaustiveBinary-1024-random (self)                   46_945.74     7_328.78    54_274.52       0.2382          1.0889        10.10
ExhaustiveBinary-1024-pca_no_rr (query)               47_984.41     2_059.75    50_044.17       0.1240             NaN        10.11
ExhaustiveBinary-1024-pca-rf10 (query)                47_984.41     2_245.80    50_230.22       0.3310        229.0833        10.11
ExhaustiveBinary-1024-pca-rf20 (query)                47_984.41     2_427.38    50_411.80       0.4429          1.2254        10.11
ExhaustiveBinary-1024-pca (self)                      47_984.41     7_443.48    55_427.90       0.3309        235.7787        10.11
ExhaustiveBinary-1024-signed_no_rr (query)            46_952.77     2_032.99    48_985.77       0.0893             NaN        10.10
ExhaustiveBinary-1024-signed-rf10 (query)             46_952.77     2_224.78    49_177.55       0.2382          1.0892        10.10
ExhaustiveBinary-1024-signed-rf20 (query)             46_952.77     2_582.99    49_535.76       0.3610          1.0558        10.10
ExhaustiveBinary-1024-signed (self)                   46_952.77     7_413.99    54_366.77       0.2382          1.0889        10.10
IVF-Binary-256-nl158-np7-rf0-random (query)           19_002.41       480.82    19_483.24       0.0466             NaN         3.14
IVF-Binary-256-nl158-np12-rf0-random (query)          19_002.41       486.77    19_489.19       0.0401             NaN         3.14
IVF-Binary-256-nl158-np17-rf0-random (query)          19_002.41       498.00    19_500.42       0.0318             NaN         3.14
IVF-Binary-256-nl158-np7-rf10-random (query)          19_002.41       595.01    19_597.43       0.1855          1.1211         3.14
IVF-Binary-256-nl158-np7-rf20-random (query)          19_002.41       705.13    19_707.55       0.2921          1.0777         3.14
IVF-Binary-256-nl158-np12-rf10-random (query)         19_002.41       592.65    19_595.07       0.1726          1.1303         3.14
IVF-Binary-256-nl158-np12-rf20-random (query)         19_002.41       712.91    19_715.32       0.2786          1.0828         3.14
IVF-Binary-256-nl158-np17-rf10-random (query)         19_002.41       599.40    19_601.81       0.1597          1.1394         3.14
IVF-Binary-256-nl158-np17-rf20-random (query)         19_002.41       723.16    19_725.58       0.2658          1.0879         3.14
IVF-Binary-256-nl158-random (self)                    19_002.41     1_889.18    20_891.60       0.1733          1.1290         3.14
IVF-Binary-256-nl223-np11-rf0-random (query)          13_840.71       492.81    14_333.52       0.0446             NaN         3.40
IVF-Binary-256-nl223-np14-rf0-random (query)          13_840.71       493.69    14_334.40       0.0349             NaN         3.40
IVF-Binary-256-nl223-np21-rf0-random (query)          13_840.71       499.09    14_339.80       0.0313             NaN         3.40
IVF-Binary-256-nl223-np11-rf10-random (query)         13_840.71       608.16    14_448.87       0.1883          1.1195         3.40
IVF-Binary-256-nl223-np11-rf20-random (query)         13_840.71       723.64    14_564.36       0.2980          1.0751         3.40
IVF-Binary-256-nl223-np14-rf10-random (query)         13_840.71       605.91    14_446.62       0.1646          1.1365         3.40
IVF-Binary-256-nl223-np14-rf20-random (query)         13_840.71       727.23    14_567.94       0.2716          1.0860         3.40
IVF-Binary-256-nl223-np21-rf10-random (query)         13_840.71       611.04    14_451.75       0.1549          1.1434         3.40
IVF-Binary-256-nl223-np21-rf20-random (query)         13_840.71       735.25    14_575.96       0.2595          1.0910         3.40
IVF-Binary-256-nl223-random (self)                    13_840.71     1_931.11    15_771.83       0.1655          1.1353         3.40
IVF-Binary-256-nl316-np15-rf0-random (query)          14_526.93       524.09    15_051.03       0.0364             NaN         3.76
IVF-Binary-256-nl316-np17-rf0-random (query)          14_526.93       525.16    15_052.10       0.0348             NaN         3.76
IVF-Binary-256-nl316-np25-rf0-random (query)          14_526.93       530.24    15_057.17       0.0330             NaN         3.76
IVF-Binary-256-nl316-np15-rf10-random (query)         14_526.93       640.11    15_167.04       0.1705          1.1324         3.76
IVF-Binary-256-nl316-np15-rf20-random (query)         14_526.93       754.80    15_281.74       0.2805          1.0829         3.76
IVF-Binary-256-nl316-np17-rf10-random (query)         14_526.93       636.38    15_163.31       0.1660          1.1354         3.76
IVF-Binary-256-nl316-np17-rf20-random (query)         14_526.93       758.26    15_285.19       0.2747          1.0849         3.76
IVF-Binary-256-nl316-np25-rf10-random (query)         14_526.93       643.91    15_170.84       0.1607          1.1388         3.76
IVF-Binary-256-nl316-np25-rf20-random (query)         14_526.93       766.86    15_293.79       0.2667          1.0878         3.76
IVF-Binary-256-nl316-random (self)                    14_526.93     2_042.65    16_569.58       0.1671          1.1342         3.76
IVF-Binary-256-nl158-np7-rf0-pca (query)              19_966.31       478.28    20_444.59       0.2011             NaN         3.15
IVF-Binary-256-nl158-np12-rf0-pca (query)             19_966.31       483.96    20_450.27       0.1995             NaN         3.15
IVF-Binary-256-nl158-np17-rf0-pca (query)             19_966.31       490.19    20_456.50       0.1980             NaN         3.15
IVF-Binary-256-nl158-np7-rf10-pca (query)             19_966.31       612.27    20_578.58       0.5842          1.0250         3.15
IVF-Binary-256-nl158-np7-rf20-pca (query)             19_966.31       733.11    20_699.42       0.7446          1.0119         3.15
IVF-Binary-256-nl158-np12-rf10-pca (query)            19_966.31       614.96    20_581.27       0.5775          1.0257         3.15
IVF-Binary-256-nl158-np12-rf20-pca (query)            19_966.31       750.96    20_717.27       0.7361          1.0124         3.15
IVF-Binary-256-nl158-np17-rf10-pca (query)            19_966.31       629.27    20_595.58       0.5672          1.0268         3.15
IVF-Binary-256-nl158-np17-rf20-pca (query)            19_966.31       762.70    20_729.01       0.7214          1.0134         3.15
IVF-Binary-256-nl158-pca (self)                       19_966.31     1_983.95    21_950.26       0.5783          1.0257         3.15
IVF-Binary-256-nl223-np11-rf0-pca (query)             14_776.92       501.80    15_278.72       0.2009             NaN         3.40
IVF-Binary-256-nl223-np14-rf0-pca (query)             14_776.92       503.90    15_280.82       0.2000             NaN         3.40
IVF-Binary-256-nl223-np21-rf0-pca (query)             14_776.92       510.09    15_287.01       0.1986             NaN         3.40
IVF-Binary-256-nl223-np11-rf10-pca (query)            14_776.92       635.93    15_412.85       0.5840          1.0250         3.40
IVF-Binary-256-nl223-np11-rf20-pca (query)            14_776.92       759.86    15_536.79       0.7445          1.0119         3.40
IVF-Binary-256-nl223-np14-rf10-pca (query)            14_776.92       636.06    15_412.98       0.5802          1.0254         3.40
IVF-Binary-256-nl223-np14-rf20-pca (query)            14_776.92       765.22    15_542.15       0.7408          1.0121         3.40
IVF-Binary-256-nl223-np21-rf10-pca (query)            14_776.92       646.81    15_423.73       0.5710          1.0264         3.40
IVF-Binary-256-nl223-np21-rf20-pca (query)            14_776.92       786.53    15_563.45       0.7267          1.0131         3.40
IVF-Binary-256-nl223-pca (self)                       14_776.92     2_066.45    16_843.38       0.5809          1.0254         3.40
IVF-Binary-256-nl316-np15-rf0-pca (query)             15_418.35       533.55    15_951.90       0.2008             NaN         3.77
IVF-Binary-256-nl316-np17-rf0-pca (query)             15_418.35       534.07    15_952.42       0.2005             NaN         3.77
IVF-Binary-256-nl316-np25-rf0-pca (query)             15_418.35       539.46    15_957.82       0.1995             NaN         3.77
IVF-Binary-256-nl316-np15-rf10-pca (query)            15_418.35       680.61    16_098.96       0.5848          1.0249         3.77
IVF-Binary-256-nl316-np15-rf20-pca (query)            15_418.35       799.15    16_217.51       0.7463          1.0117         3.77
IVF-Binary-256-nl316-np17-rf10-pca (query)            15_418.35       668.55    16_086.90       0.5831          1.0251         3.77
IVF-Binary-256-nl316-np17-rf20-pca (query)            15_418.35       800.76    16_219.11       0.7445          1.0119         3.77
IVF-Binary-256-nl316-np25-rf10-pca (query)            15_418.35       676.32    16_094.68       0.5760          1.0258         3.77
IVF-Binary-256-nl316-np25-rf20-pca (query)            15_418.35       811.41    16_229.76       0.7341          1.0126         3.77
IVF-Binary-256-nl316-pca (self)                       15_418.35     2_162.53    17_580.88       0.5837          1.0251         3.77
IVF-Binary-512-nl158-np7-rf0-random (query)           30_598.17       876.64    31_474.81       0.0738             NaN         5.67
IVF-Binary-512-nl158-np12-rf0-random (query)          30_598.17       887.27    31_485.44       0.0699             NaN         5.67
IVF-Binary-512-nl158-np17-rf0-random (query)          30_598.17       904.08    31_502.25       0.0656             NaN         5.67
IVF-Binary-512-nl158-np7-rf10-random (query)          30_598.17     1_000.79    31_598.97       0.2129          1.1011         5.67
IVF-Binary-512-nl158-np7-rf20-random (query)          30_598.17     1_122.41    31_720.58       0.3277          1.0638         5.67
IVF-Binary-512-nl158-np12-rf10-random (query)         30_598.17     1_011.98    31_610.16       0.2065          1.1043         5.67
IVF-Binary-512-nl158-np12-rf20-random (query)         30_598.17     1_129.73    31_727.90       0.3187          1.0660         5.67
IVF-Binary-512-nl158-np17-rf10-random (query)         30_598.17     1_020.22    31_618.39       0.2008          1.1069         5.67
IVF-Binary-512-nl158-np17-rf20-random (query)         30_598.17     1_137.08    31_735.26       0.3121          1.0675         5.67
IVF-Binary-512-nl158-random (self)                    30_598.17     3_266.84    33_865.02       0.2075          1.1033         5.67
IVF-Binary-512-nl223-np11-rf0-random (query)          25_377.86       906.50    26_284.36       0.0738             NaN         5.92
IVF-Binary-512-nl223-np14-rf0-random (query)          25_377.86       911.35    26_289.21       0.0664             NaN         5.92
IVF-Binary-512-nl223-np21-rf0-random (query)          25_377.86       936.37    26_314.23       0.0641             NaN         5.92
IVF-Binary-512-nl223-np11-rf10-random (query)         25_377.86     1_030.35    26_408.21       0.2149          1.0991         5.92
IVF-Binary-512-nl223-np11-rf20-random (query)         25_377.86     1_158.00    26_535.87       0.3290          1.0627         5.92
IVF-Binary-512-nl223-np14-rf10-random (query)         25_377.86     1_023.71    26_401.57       0.2038          1.1055         5.92
IVF-Binary-512-nl223-np14-rf20-random (query)         25_377.86     1_159.29    26_537.15       0.3156          1.0666         5.92
IVF-Binary-512-nl223-np21-rf10-random (query)         25_377.86     1_040.41    26_418.27       0.1975          1.1089         5.92
IVF-Binary-512-nl223-np21-rf20-random (query)         25_377.86     1_172.91    26_550.77       0.3090          1.0686         5.92
IVF-Binary-512-nl223-random (self)                    25_377.86     3_802.81    29_180.67       0.2037          1.1051         5.92
IVF-Binary-512-nl316-np15-rf0-random (query)          26_338.16       947.42    27_285.58       0.0683             NaN         6.29
IVF-Binary-512-nl316-np17-rf0-random (query)          26_338.16       952.95    27_291.11       0.0673             NaN         6.29
IVF-Binary-512-nl316-np25-rf0-random (query)          26_338.16       967.46    27_305.62       0.0659             NaN         6.29
IVF-Binary-512-nl316-np15-rf10-random (query)         26_338.16     1_067.96    27_406.12       0.2080          1.1036         6.29
IVF-Binary-512-nl316-np15-rf20-random (query)         26_338.16     1_195.46    27_533.63       0.3214          1.0652         6.29
IVF-Binary-512-nl316-np17-rf10-random (query)         26_338.16     1_085.87    27_424.03       0.2053          1.1050         6.29
IVF-Binary-512-nl316-np17-rf20-random (query)         26_338.16     1_205.90    27_544.06       0.3171          1.0663         6.29
IVF-Binary-512-nl316-np25-rf10-random (query)         26_338.16     1_078.52    27_416.69       0.2013          1.1069         6.29
IVF-Binary-512-nl316-np25-rf20-random (query)         26_338.16     1_214.47    27_552.64       0.3119          1.0675         6.29
IVF-Binary-512-nl316-random (self)                    26_338.16     3_494.36    29_832.53       0.2056          1.1044         6.29
IVF-Binary-512-nl158-np7-rf0-pca (query)              31_973.51       901.24    32_874.76       0.2590             NaN         5.67
IVF-Binary-512-nl158-np12-rf0-pca (query)             31_973.51       908.53    32_882.04       0.2497             NaN         5.67
IVF-Binary-512-nl158-np17-rf0-pca (query)             31_973.51       946.25    32_919.77       0.2402             NaN         5.67
IVF-Binary-512-nl158-np7-rf10-pca (query)             31_973.51     1_025.03    32_998.55       0.7002          1.0145         5.67
IVF-Binary-512-nl158-np7-rf20-pca (query)             31_973.51     1_149.27    33_122.78       0.8406          1.0062         5.67
IVF-Binary-512-nl158-np12-rf10-pca (query)            31_973.51     1_034.75    33_008.26       0.6745          1.0165         5.67
IVF-Binary-512-nl158-np12-rf20-pca (query)            31_973.51     1_183.86    33_157.38       0.8168          1.0072         5.67
IVF-Binary-512-nl158-np17-rf10-pca (query)            31_973.51     1_064.29    33_037.80       0.6442          1.0191         5.67
IVF-Binary-512-nl158-np17-rf20-pca (query)            31_973.51     1_187.06    33_160.57       0.7855          1.0089         5.67
IVF-Binary-512-nl158-pca (self)                       31_973.51     3_418.61    35_392.13       0.6752          1.0165         5.67
IVF-Binary-512-nl223-np11-rf0-pca (query)             26_604.36       930.50    27_534.86       0.2574             NaN         5.93
IVF-Binary-512-nl223-np14-rf0-pca (query)             26_604.36       977.49    27_581.86       0.2535             NaN         5.93
IVF-Binary-512-nl223-np21-rf0-pca (query)             26_604.36       960.19    27_564.56       0.2437             NaN         5.93
IVF-Binary-512-nl223-np11-rf10-pca (query)            26_604.36     1_056.73    27_661.10       0.6972          1.0147         5.93
IVF-Binary-512-nl223-np11-rf20-pca (query)            26_604.36     1_182.31    27_786.67       0.8391          1.0062         5.93
IVF-Binary-512-nl223-np14-rf10-pca (query)            26_604.36     1_054.49    27_658.85       0.6866          1.0155         5.93
IVF-Binary-512-nl223-np14-rf20-pca (query)            26_604.36     1_188.90    27_793.27       0.8296          1.0066         5.93
IVF-Binary-512-nl223-np21-rf10-pca (query)            26_604.36     1_075.13    27_679.50       0.6566          1.0180         5.93
IVF-Binary-512-nl223-np21-rf20-pca (query)            26_604.36     1_218.36    27_822.72       0.7980          1.0083         5.93
IVF-Binary-512-nl223-pca (self)                       26_604.36     3_479.50    30_083.86       0.6870          1.0156         5.93
IVF-Binary-512-nl316-np15-rf0-pca (query)             27_256.84       959.87    28_216.71       0.2583             NaN         6.29
IVF-Binary-512-nl316-np17-rf0-pca (query)             27_256.84       960.20    28_217.04       0.2565             NaN         6.29
IVF-Binary-512-nl316-np25-rf0-pca (query)             27_256.84       966.95    28_223.80       0.2487             NaN         6.29
IVF-Binary-512-nl316-np15-rf10-pca (query)            27_256.84     1_088.11    28_344.95       0.7009          1.0144         6.29
IVF-Binary-512-nl316-np15-rf20-pca (query)            27_256.84     1_211.59    28_468.43       0.8436          1.0059         6.29
IVF-Binary-512-nl316-np17-rf10-pca (query)            27_256.84     1_092.01    28_348.85       0.6959          1.0148         6.29
IVF-Binary-512-nl316-np17-rf20-pca (query)            27_256.84     1_218.46    28_475.30       0.8394          1.0061         6.29
IVF-Binary-512-nl316-np25-rf10-pca (query)            27_256.84     1_102.29    28_359.13       0.6714          1.0168         6.29
IVF-Binary-512-nl316-np25-rf20-pca (query)            27_256.84     1_242.86    28_499.71       0.8138          1.0074         6.29
IVF-Binary-512-nl316-pca (self)                       27_256.84     3_573.83    30_830.67       0.6967          1.0148         6.29
IVF-Binary-1024-nl158-np7-rf0-random (query)          54_297.68     1_712.88    56_010.56       0.0929             NaN        10.72
IVF-Binary-1024-nl158-np12-rf0-random (query)         54_297.68     1_726.47    56_024.15       0.0915             NaN        10.72
IVF-Binary-1024-nl158-np17-rf0-random (query)         54_297.68     1_813.95    56_111.63       0.0906             NaN        10.72
IVF-Binary-1024-nl158-np7-rf10-random (query)         54_297.68     1_819.39    56_117.06       0.2511          1.0849        10.72
IVF-Binary-1024-nl158-np7-rf20-random (query)         54_297.68     1_959.14    56_256.82       0.3779          1.0527        10.72
IVF-Binary-1024-nl158-np12-rf10-random (query)        54_297.68     1_839.39    56_137.07       0.2460          1.0867        10.72
IVF-Binary-1024-nl158-np12-rf20-random (query)        54_297.68     1_958.52    56_256.20       0.3703          1.0541        10.72
IVF-Binary-1024-nl158-np17-rf10-random (query)        54_297.68     1_863.67    56_161.35       0.2426          1.0878        10.72
IVF-Binary-1024-nl158-np17-rf20-random (query)        54_297.68     1_974.99    56_272.66       0.3665          1.0548        10.72
IVF-Binary-1024-nl158-random (self)                   54_297.68     6_046.97    60_344.65       0.2458          1.0866        10.72
IVF-Binary-1024-nl223-np11-rf0-random (query)         49_081.46     1_738.12    50_819.58       0.0929             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-random (query)         49_081.46     1_755.51    50_836.97       0.0909             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-random (query)         49_081.46     1_761.48    50_842.94       0.0900             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-random (query)        49_081.46     1_846.65    50_928.12       0.2495          1.0856        10.98
IVF-Binary-1024-nl223-np11-rf20-random (query)        49_081.46     1_958.89    51_040.35       0.3749          1.0534        10.98
IVF-Binary-1024-nl223-np14-rf10-random (query)        49_081.46     1_856.67    50_938.14       0.2430          1.0876        10.98
IVF-Binary-1024-nl223-np14-rf20-random (query)        49_081.46     1_988.98    51_070.44       0.3670          1.0547        10.98
IVF-Binary-1024-nl223-np21-rf10-random (query)        49_081.46     1_867.40    50_948.86       0.2402          1.0885        10.98
IVF-Binary-1024-nl223-np21-rf20-random (query)        49_081.46     2_008.26    51_089.72       0.3641          1.0552        10.98
IVF-Binary-1024-nl223-random (self)                   49_081.46     6_108.95    55_190.41       0.2435          1.0873        10.98
IVF-Binary-1024-nl316-np15-rf0-random (query)         49_773.15     1_764.72    51_537.87       0.0918             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-random (query)         49_773.15     1_768.41    51_541.56       0.0913             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-random (query)         49_773.15     1_781.92    51_555.07       0.0906             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-random (query)        49_773.15     1_874.88    51_648.03       0.2465          1.0864        11.34
IVF-Binary-1024-nl316-np15-rf20-random (query)        49_773.15     1_997.97    51_771.12       0.3722          1.0538        11.34
IVF-Binary-1024-nl316-np17-rf10-random (query)        49_773.15     1_885.43    51_658.58       0.2444          1.0871        11.34
IVF-Binary-1024-nl316-np17-rf20-random (query)        49_773.15     2_041.07    51_814.22       0.3692          1.0544        11.34
IVF-Binary-1024-nl316-np25-rf10-random (query)        49_773.15     1_890.96    51_664.11       0.2419          1.0880        11.34
IVF-Binary-1024-nl316-np25-rf20-random (query)        49_773.15     2_042.90    51_816.05       0.3655          1.0550        11.34
IVF-Binary-1024-nl316-random (self)                   49_773.15     6_189.16    55_962.31       0.2448          1.0869        11.34
IVF-Binary-1024-nl158-np7-rf0-pca (query)             55_259.00     1_729.10    56_988.10       0.3012             NaN        10.73
IVF-Binary-1024-nl158-np12-rf0-pca (query)            55_259.00     1_768.39    57_027.39       0.2743             NaN        10.73
IVF-Binary-1024-nl158-np17-rf0-pca (query)            55_259.00     1_782.62    57_041.62       0.2512             NaN        10.73
IVF-Binary-1024-nl158-np7-rf10-pca (query)            55_259.00     1_858.19    57_117.19       0.7699          1.0097        10.73
IVF-Binary-1024-nl158-np7-rf20-pca (query)            55_259.00     1_984.67    57_243.67       0.8902          1.0038        10.73
IVF-Binary-1024-nl158-np12-rf10-pca (query)           55_259.00     1_888.75    57_147.75       0.7208          1.0128        10.73
IVF-Binary-1024-nl158-np12-rf20-pca (query)           55_259.00     2_015.74    57_274.74       0.8547          1.0052        10.73
IVF-Binary-1024-nl158-np17-rf10-pca (query)           55_259.00     1_892.09    57_151.08       0.6729          1.0166        10.73
IVF-Binary-1024-nl158-np17-rf20-pca (query)           55_259.00     2_053.89    57_312.89       0.8118          1.0073        10.73
IVF-Binary-1024-nl158-pca (self)                      55_259.00     6_204.70    61_463.70       0.7229          1.0127        10.73
IVF-Binary-1024-nl223-np11-rf0-pca (query)            50_150.40     1_762.26    51_912.65       0.2963             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-pca (query)            50_150.40     1_760.96    51_911.35       0.2853             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-pca (query)            50_150.40     1_780.47    51_930.87       0.2607             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-pca (query)           50_150.40     1_881.50    52_031.89       0.7635          1.0101        10.98
IVF-Binary-1024-nl223-np11-rf20-pca (query)           50_150.40     2_014.13    52_164.53       0.8870          1.0039        10.98
IVF-Binary-1024-nl223-np14-rf10-pca (query)           50_150.40     1_885.75    52_036.14       0.7442          1.0113        10.98
IVF-Binary-1024-nl223-np14-rf20-pca (query)           50_150.40     2_020.33    52_170.73       0.8738          1.0044        10.98
IVF-Binary-1024-nl223-np21-rf10-pca (query)           50_150.40     1_912.93    52_063.33       0.6935          1.0150        10.98
IVF-Binary-1024-nl223-np21-rf20-pca (query)           50_150.40     2_065.31    52_215.70       0.8303          1.0064        10.98
IVF-Binary-1024-nl223-pca (self)                      50_150.40     6_343.11    56_493.51       0.7453          1.0112        10.98
IVF-Binary-1024-nl316-np15-rf0-pca (query)            51_177.38     1_805.61    52_982.99       0.2992             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-pca (query)            51_177.38     1_830.77    53_008.15       0.2939             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-pca (query)            51_177.38     1_825.90    53_003.28       0.2724             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-pca (query)           51_177.38     1_915.31    53_092.69       0.7698          1.0097        11.34
IVF-Binary-1024-nl316-np15-rf20-pca (query)           51_177.38     2_057.14    53_234.52       0.8934          1.0036        11.34
IVF-Binary-1024-nl316-np17-rf10-pca (query)           51_177.38     1_933.44    53_110.82       0.7606          1.0102        11.34
IVF-Binary-1024-nl316-np17-rf20-pca (query)           51_177.38     2_228.98    53_406.36       0.8877          1.0038        11.34
IVF-Binary-1024-nl316-np25-rf10-pca (query)           51_177.38     1_934.58    53_111.96       0.7184          1.0131        11.34
IVF-Binary-1024-nl316-np25-rf20-pca (query)           51_177.38     2_073.23    53_250.61       0.8524          1.0053        11.34
IVF-Binary-1024-nl316-pca (self)                      51_177.38     6_344.08    57_521.46       0.7626          1.0101        11.34
IVF-Binary-1024-nl158-np7-rf0-signed (query)          54_313.70     1_699.55    56_013.25       0.0929             NaN        10.72
IVF-Binary-1024-nl158-np12-rf0-signed (query)         54_313.70     1_709.74    56_023.44       0.0915             NaN        10.72
IVF-Binary-1024-nl158-np17-rf0-signed (query)         54_313.70     1_780.41    56_094.11       0.0906             NaN        10.72
IVF-Binary-1024-nl158-np7-rf10-signed (query)         54_313.70     1_829.65    56_143.35       0.2511          1.0849        10.72
IVF-Binary-1024-nl158-np7-rf20-signed (query)         54_313.70     1_929.10    56_242.80       0.3779          1.0527        10.72
IVF-Binary-1024-nl158-np12-rf10-signed (query)        54_313.70     1_829.16    56_142.86       0.2460          1.0867        10.72
IVF-Binary-1024-nl158-np12-rf20-signed (query)        54_313.70     1_964.89    56_278.60       0.3703          1.0541        10.72
IVF-Binary-1024-nl158-np17-rf10-signed (query)        54_313.70     1_854.29    56_167.99       0.2426          1.0878        10.72
IVF-Binary-1024-nl158-np17-rf20-signed (query)        54_313.70     1_986.13    56_299.84       0.3665          1.0548        10.72
IVF-Binary-1024-nl158-signed (self)                   54_313.70     6_032.52    60_346.23       0.2458          1.0866        10.72
IVF-Binary-1024-nl223-np11-rf0-signed (query)         49_085.51     1_733.61    50_819.12       0.0929             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-signed (query)         49_085.51     1_739.83    50_825.33       0.0909             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-signed (query)         49_085.51     1_754.47    50_839.98       0.0900             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-signed (query)        49_085.51     1_848.56    50_934.07       0.2495          1.0856        10.98
IVF-Binary-1024-nl223-np11-rf20-signed (query)        49_085.51     1_967.22    51_052.73       0.3749          1.0534        10.98
IVF-Binary-1024-nl223-np14-rf10-signed (query)        49_085.51     1_854.77    50_940.28       0.2430          1.0876        10.98
IVF-Binary-1024-nl223-np14-rf20-signed (query)        49_085.51     1_992.27    51_077.77       0.3670          1.0547        10.98
IVF-Binary-1024-nl223-np21-rf10-signed (query)        49_085.51     1_868.78    50_954.28       0.2402          1.0885        10.98
IVF-Binary-1024-nl223-np21-rf20-signed (query)        49_085.51     1_989.73    51_075.24       0.3641          1.0552        10.98
IVF-Binary-1024-nl223-signed (self)                   49_085.51     6_100.46    55_185.97       0.2435          1.0873        10.98
IVF-Binary-1024-nl316-np15-rf0-signed (query)         49_711.54     1_785.79    51_497.33       0.0918             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-signed (query)         49_711.54     1_794.49    51_506.03       0.0913             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-signed (query)         49_711.54     1_810.50    51_522.04       0.0906             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-signed (query)        49_711.54     1_903.85    51_615.39       0.2465          1.0864        11.34
IVF-Binary-1024-nl316-np15-rf20-signed (query)        49_711.54     2_058.01    51_769.55       0.3722          1.0538        11.34
IVF-Binary-1024-nl316-np17-rf10-signed (query)        49_711.54     1_917.71    51_629.25       0.2444          1.0871        11.34
IVF-Binary-1024-nl316-np17-rf20-signed (query)        49_711.54     2_048.24    51_759.78       0.3692          1.0544        11.34
IVF-Binary-1024-nl316-np25-rf10-signed (query)        49_711.54     1_924.66    51_636.20       0.2419          1.0880        11.34
IVF-Binary-1024-nl316-np25-rf20-signed (query)        49_711.54     2_058.83    51_770.37       0.3655          1.0550        11.34
IVF-Binary-1024-nl316-signed (self)                   49_711.54     6_335.39    56_046.93       0.2448          1.0869        11.34
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
Exhaustive (query)                                         6.75     4_216.65     4_223.40       1.0000          1.0000        48.83
Exhaustive (self)                                          6.75    14_034.56    14_041.32       1.0000          1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_611.88       249.11     2_860.99       0.0387             NaN         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_611.88       356.21     2_968.09       0.2254          1.0175         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_611.88       464.29     3_076.17       0.3590          1.0099         1.78
ExhaustiveBinary-256-random (self)                     2_611.88     1_207.72     3_819.60       0.5886         13.9170         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_829.13       272.36     3_101.50       0.0284             NaN         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_829.13       358.66     3_187.79       0.1316          1.0445         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_829.13       460.82     3_289.95       0.1964          1.0197         1.78
ExhaustiveBinary-256-pca (self)                        2_829.13     1_199.20     4_028.33       0.2462          1.6481         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_149.98       466.66     5_616.63       0.0605             NaN         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_149.98       554.09     5_704.07       0.3044          1.0140         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_149.98       661.24     5_811.21       0.4560          1.0080         3.55
ExhaustiveBinary-512-random (self)                     5_149.98     1_839.73     6_989.71       0.6749         20.2700         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_484.14       454.76     5_938.90       0.0806             NaN         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_484.14       564.13     6_048.28       0.3799          1.0083         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_484.14       669.06     6_153.20       0.5477          1.0039         3.55
ExhaustiveBinary-512-pca (self)                        5_484.14     1_870.05     7_354.19       0.6562          1.0566         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_177.72       765.15    10_942.87       0.0996             NaN         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_177.72       874.27    11_051.99       0.4213          1.0096         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_177.72       991.70    11_169.42       0.5847          1.0054         7.10
ExhaustiveBinary-1024-random (self)                   10_177.72     2_895.24    13_072.96       0.7571         24.6771         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_606.43       779.07    11_385.50       0.1211             NaN         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_606.43       887.88    11_494.31       0.5030          1.0092         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_606.43     1_024.16    11_630.59       0.6838          1.0042         7.10
ExhaustiveBinary-1024-pca (self)                      10_606.43     2_969.17    13_575.60       0.8235          1.0221         7.10
ExhaustiveBinary-256-signed_no_rr (query)              2_617.92       250.91     2_868.83       0.0387             NaN         1.78
ExhaustiveBinary-256-signed-rf10 (query)               2_617.92       357.87     2_975.79       0.2254          1.0175         1.78
ExhaustiveBinary-256-signed-rf20 (query)               2_617.92       465.11     3_083.03       0.3590          1.0099         1.78
ExhaustiveBinary-256-signed (self)                     2_617.92     1_176.25     3_794.17       0.5886         13.9170         1.78
IVF-Binary-256-nl158-np7-rf0-random (query)            4_035.19       119.17     4_154.35       0.0512             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           4_035.19       131.71     4_166.90       0.0468             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           4_035.19       146.81     4_182.00       0.0447             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           4_035.19       185.83     4_221.02       0.3058          1.0092         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           4_035.19       234.91     4_270.10       0.4820          1.0048         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          4_035.19       208.37     4_243.56       0.2844          1.0105         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          4_035.19       264.98     4_300.16       0.4542          1.0055         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          4_035.19       224.51     4_259.70       0.2747          1.0111         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          4_035.19       291.75     4_326.94       0.4399          1.0059         1.93
IVF-Binary-256-nl158-random (self)                     4_035.19       628.87     4_664.06       0.7492          1.0400         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           3_039.38       120.31     3_159.68       0.0509             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           3_039.38       122.89     3_162.27       0.0479             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           3_039.38       128.76     3_168.14       0.0452             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          3_039.38       180.61     3_219.99       0.3024          1.0093         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          3_039.38       232.41     3_271.79       0.4795          1.0050         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          3_039.38       185.71     3_225.08       0.2895          1.0100         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          3_039.38       242.20     3_281.58       0.4632          1.0054         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          3_039.38       200.95     3_240.33       0.2784          1.0107         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          3_039.38       256.99     3_296.37       0.4476          1.0058         2.00
IVF-Binary-256-nl223-random (self)                     3_039.38       551.50     3_590.88       0.7429          1.0417         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           3_231.64       125.64     3_357.28       0.0506             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           3_231.64       125.82     3_357.46       0.0492             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           3_231.64       132.79     3_364.43       0.0461             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          3_231.64       188.31     3_419.96       0.3031          1.0093         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          3_231.64       254.22     3_485.86       0.4821          1.0050         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          3_231.64       189.18     3_420.83       0.2972          1.0095         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          3_231.64       240.97     3_472.61       0.4738          1.0052         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          3_231.64       195.85     3_427.49       0.2830          1.0103         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          3_231.64       273.35     3_504.99       0.4542          1.0056         2.09
IVF-Binary-256-nl316-random (self)                     3_231.64       559.37     3_791.01       0.7435          1.0416         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               4_237.11       125.16     4_362.27       0.0614             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              4_237.11       135.95     4_373.06       0.0514             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              4_237.11       148.31     4_385.43       0.0460             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              4_237.11       195.00     4_432.11       0.3049          1.0087         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              4_237.11       253.27     4_490.39       0.4484          1.0049         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             4_237.11       211.11     4_448.22       0.2531          1.0116         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             4_237.11       282.27     4_519.38       0.3761          1.0067         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             4_237.11       232.58     4_469.69       0.2249          1.0141         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             4_237.11       307.65     4_544.76       0.3349          1.0081         1.93
IVF-Binary-256-nl158-pca (self)                        4_237.11       668.51     4_905.63       0.3527          1.2486         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              3_247.91       122.41     3_370.32       0.0655             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              3_247.91       124.78     3_372.69       0.0603             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              3_247.91       131.59     3_379.50       0.0538             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             3_247.91       188.65     3_436.57       0.3313          1.0076         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             3_247.91       250.55     3_498.46       0.4956          1.0041         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             3_247.91       194.43     3_442.34       0.3058          1.0085         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             3_247.91       252.35     3_500.26       0.4572          1.0048         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             3_247.91       204.74     3_452.65       0.2708          1.0101         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             3_247.91       269.41     3_517.33       0.4055          1.0058         2.00
IVF-Binary-256-nl223-pca (self)                        3_247.91       598.31     3_846.22       0.3955          1.1701         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              3_440.36       128.25     3_568.61       0.0664             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              3_440.36       130.36     3_570.72       0.0639             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              3_440.36       134.31     3_574.67       0.0569             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             3_440.36       198.25     3_638.61       0.3377          1.0074         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             3_440.36       250.73     3_691.09       0.5053          1.0040         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             3_440.36       198.15     3_638.52       0.3242          1.0079         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             3_440.36       255.28     3_695.64       0.4857          1.0043         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             3_440.36       208.66     3_649.02       0.2878          1.0093         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             3_440.36       268.31     3_708.67       0.4313          1.0053         2.09
IVF-Binary-256-nl316-pca (self)                        3_440.36       613.22     4_053.58       0.4097          1.1569         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_567.70       219.66     6_787.36       0.0738             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_567.70       244.18     6_811.87       0.0692             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_567.70       268.85     6_836.54       0.0672             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_567.70       285.51     6_853.21       0.3879          1.0068         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_567.70       335.96     6_903.66       0.5764          1.0034         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_567.70       312.28     6_879.98       0.3698          1.0076         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_567.70       376.64     6_944.34       0.5546          1.0038         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_567.70       344.99     6_912.69       0.3603          1.0080         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_567.70       410.25     6_977.94       0.5422          1.0041         3.71
IVF-Binary-512-nl158-random (self)                     6_567.70       996.12     7_563.82       0.8444          1.0199         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_535.17       213.40     5_748.58       0.0734             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_535.17       219.57     5_754.75       0.0704             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_535.17       229.64     5_764.81       0.0677             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_535.17       278.84     5_814.01       0.3844          1.0071         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_535.17       330.64     5_865.82       0.5732          1.0036         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_535.17       283.45     5_818.63       0.3729          1.0075         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_535.17       337.72     5_872.89       0.5589          1.0039         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_535.17       302.85     5_838.03       0.3642          1.0078         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_535.17       361.04     5_896.21       0.5474          1.0041         3.77
IVF-Binary-512-nl223-random (self)                     5_535.17       876.52     6_411.69       0.8385          1.0211         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           5_725.30       220.92     5_946.22       0.0731             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           5_725.30       221.71     5_947.01       0.0715             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           5_725.30       229.26     5_954.57       0.0683             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          5_725.30       293.91     6_019.21       0.3848          1.0071         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          5_725.30       335.34     6_060.64       0.5757          1.0036         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          5_725.30       304.79     6_030.10       0.3796          1.0072         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          5_725.30       340.40     6_065.70       0.5685          1.0037         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          5_725.30       298.81     6_024.11       0.3674          1.0077         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          5_725.30       353.57     6_078.87       0.5525          1.0040         3.86
IVF-Binary-512-nl316-random (self)                     5_725.30       883.30     6_608.60       0.8398          1.0208         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               6_877.93       224.54     7_102.47       0.0968             NaN         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              6_877.93       262.58     7_140.51       0.0917             NaN         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              6_877.93       263.32     7_141.26       0.0892             NaN         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              6_877.93       291.13     7_169.06       0.4470          1.0049         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              6_877.93       345.27     7_223.20       0.6312          1.0024         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             6_877.93       316.52     7_194.45       0.4265          1.0055         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             6_877.93       380.50     7_258.43       0.6078          1.0028         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             6_877.93       347.96     7_225.90       0.4175          1.0058         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             6_877.93       414.81     7_292.74       0.5960          1.0030         3.71
IVF-Binary-512-nl158-pca (self)                        6_877.93     1_005.78     7_883.71       0.6714          1.0520         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              5_860.49       222.20     6_082.69       0.0962             NaN         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              5_860.49       230.34     6_090.84       0.0935             NaN         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              5_860.49       238.95     6_099.44       0.0909             NaN         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             5_860.49       287.98     6_148.48       0.4481          1.0051         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             5_860.49       340.56     6_201.05       0.6398          1.0024         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             5_860.49       292.83     6_153.32       0.4371          1.0054         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             5_860.49       353.99     6_214.49       0.6251          1.0026         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             5_860.49       308.76     6_169.26       0.4261          1.0056         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             5_860.49       370.23     6_230.72       0.6103          1.0028         3.77
IVF-Binary-512-nl223-pca (self)                        5_860.49       917.39     6_777.88       0.6652          1.0541         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              6_040.71       228.66     6_269.37       0.0969             NaN         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              6_040.71       232.05     6_272.76       0.0955             NaN         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              6_040.71       238.60     6_279.31       0.0923             NaN         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             6_040.71       295.73     6_336.44       0.4522          1.0050         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             6_040.71       351.03     6_391.73       0.6446          1.0024         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             6_040.71       298.24     6_338.94       0.4460          1.0052         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             6_040.71       354.71     6_395.42       0.6368          1.0025         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             6_040.71       311.48     6_352.19       0.4323          1.0055         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             6_040.71       369.53     6_410.24       0.6184          1.0027         3.86
IVF-Binary-512-nl316-pca (self)                        6_040.71       925.45     6_966.15       0.6660          1.0538         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_566.90       406.57    11_973.46       0.1148             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_566.90       432.77    11_999.67       0.1107             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_566.90       464.80    12_031.69       0.1084             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_566.90       475.48    12_042.37       0.5057          1.0045         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_566.90       530.63    12_097.53       0.6946          1.0021         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_566.90       513.65    12_080.55       0.4909          1.0049         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_566.90       576.05    12_142.94       0.6793          1.0023         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_566.90       555.50    12_122.40       0.4823          1.0052         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_566.90       637.86    12_204.76       0.6691          1.0025         7.26
IVF-Binary-1024-nl158-random (self)                   11_566.90     1_658.32    13_225.21       0.9242          1.0076         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_576.59       394.90    10_971.48       0.1141             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_576.59       401.35    10_977.94       0.1109             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_576.59       417.94    10_994.52       0.1086             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_576.59       461.74    11_038.33       0.5006          1.0048         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_576.59       509.55    11_086.14       0.6887          1.0023         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_576.59       469.96    11_046.54       0.4911          1.0051         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_576.59       524.42    11_101.01       0.6782          1.0025         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_576.59       489.91    11_066.50       0.4848          1.0052         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_576.59       552.57    11_129.16       0.6715          1.0025         7.32
IVF-Binary-1024-nl223-random (self)                   10_576.59     1_501.71    12_078.29       0.9202          1.0082         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         10_702.47       404.46    11_106.92       0.1142             NaN         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)         10_702.47       412.12    11_114.58       0.1127             NaN         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)         10_702.47       422.45    11_124.92       0.1094             NaN         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)        10_702.47       476.13    11_178.59       0.5015          1.0048         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)        10_702.47       523.68    11_226.15       0.6908          1.0023         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)        10_702.47       483.03    11_185.50       0.4966          1.0049         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)        10_702.47       532.48    11_234.95       0.6852          1.0024         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)        10_702.47       489.94    11_192.40       0.4872          1.0052         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)        10_702.47       548.00    11_250.46       0.6743          1.0025         7.41
IVF-Binary-1024-nl316-random (self)                   10_702.47     1_515.62    12_218.08       0.9209          1.0080         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)             12_106.64       420.90    12_527.54       0.1357             NaN         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            12_106.64       447.16    12_553.80       0.1314             NaN         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            12_106.64       485.05    12_591.70       0.1292             NaN         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            12_106.64       492.48    12_599.12       0.5563          1.0034         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            12_106.64       542.33    12_648.97       0.7413          1.0014         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           12_106.64       530.63    12_637.27       0.5423          1.0037         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           12_106.64       591.61    12_698.25       0.7288          1.0016         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           12_106.64       579.51    12_686.15       0.5359          1.0039         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           12_106.64       640.77    12_747.41       0.7220          1.0017         7.26
IVF-Binary-1024-nl158-pca (self)                      12_106.64     1_743.17    13_849.81       0.8300          1.0204         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            11_106.06       410.50    11_516.56       0.1343             NaN         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            11_106.06       416.29    11_522.35       0.1323             NaN         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            11_106.06       436.12    11_542.18       0.1304             NaN         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           11_106.06       477.85    11_583.91       0.5533          1.0036         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           11_106.06       530.18    11_636.24       0.7427          1.0015         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           11_106.06       484.60    11_590.66       0.5466          1.0037         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           11_106.06       556.34    11_662.40       0.7348          1.0016         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           11_106.06       504.12    11_610.19       0.5403          1.0038         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           11_106.06       566.96    11_673.02       0.7271          1.0017         7.32
IVF-Binary-1024-nl223-pca (self)                      11_106.06     1_548.83    12_654.89       0.8253          1.0214         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            11_225.61       424.59    11_650.20       0.1346             NaN         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            11_225.61       423.90    11_649.51       0.1335             NaN         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            11_225.61       435.44    11_661.05       0.1313             NaN         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           11_225.61       488.21    11_713.82       0.5552          1.0035         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           11_225.61       539.38    11_764.99       0.7450          1.0015         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           11_225.61       496.08    11_721.69       0.5517          1.0036         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           11_225.61       548.09    11_773.70       0.7411          1.0015         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           11_225.61       510.01    11_735.62       0.5436          1.0037         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           11_225.61       571.64    11_797.25       0.7313          1.0016         7.42
IVF-Binary-1024-nl316-pca (self)                      11_225.61     1_571.90    12_797.51       0.8253          1.0214         7.42
IVF-Binary-256-nl158-np7-rf0-signed (query)            4_013.80       119.71     4_133.52       0.0512             NaN         1.93
IVF-Binary-256-nl158-np12-rf0-signed (query)           4_013.80       138.42     4_152.23       0.0468             NaN         1.93
IVF-Binary-256-nl158-np17-rf0-signed (query)           4_013.80       145.43     4_159.23       0.0447             NaN         1.93
IVF-Binary-256-nl158-np7-rf10-signed (query)           4_013.80       185.60     4_199.41       0.3058          1.0092         1.93
IVF-Binary-256-nl158-np7-rf20-signed (query)           4_013.80       238.44     4_252.24       0.4820          1.0048         1.93
IVF-Binary-256-nl158-np12-rf10-signed (query)          4_013.80       205.31     4_219.11       0.2844          1.0105         1.93
IVF-Binary-256-nl158-np12-rf20-signed (query)          4_013.80       264.94     4_278.74       0.4542          1.0055         1.93
IVF-Binary-256-nl158-np17-rf10-signed (query)          4_013.80       226.53     4_240.33       0.2747          1.0111         1.93
IVF-Binary-256-nl158-np17-rf20-signed (query)          4_013.80       294.46     4_308.26       0.4399          1.0059         1.93
IVF-Binary-256-nl158-signed (self)                     4_013.80       633.72     4_647.53       0.7492          1.0400         1.93
IVF-Binary-256-nl223-np11-rf0-signed (query)           3_050.72       123.32     3_174.03       0.0509             NaN         2.00
IVF-Binary-256-nl223-np14-rf0-signed (query)           3_050.72       124.22     3_174.94       0.0479             NaN         2.00
IVF-Binary-256-nl223-np21-rf0-signed (query)           3_050.72       127.52     3_178.23       0.0452             NaN         2.00
IVF-Binary-256-nl223-np11-rf10-signed (query)          3_050.72       181.31     3_232.02       0.3024          1.0093         2.00
IVF-Binary-256-nl223-np11-rf20-signed (query)          3_050.72       230.33     3_281.05       0.4795          1.0050         2.00
IVF-Binary-256-nl223-np14-rf10-signed (query)          3_050.72       186.99     3_237.71       0.2895          1.0100         2.00
IVF-Binary-256-nl223-np14-rf20-signed (query)          3_050.72       238.49     3_289.20       0.4632          1.0054         2.00
IVF-Binary-256-nl223-np21-rf10-signed (query)          3_050.72       194.77     3_245.48       0.2784          1.0107         2.00
IVF-Binary-256-nl223-np21-rf20-signed (query)          3_050.72       254.84     3_305.55       0.4476          1.0058         2.00
IVF-Binary-256-nl223-signed (self)                     3_050.72       547.22     3_597.93       0.7429          1.0417         2.00
IVF-Binary-256-nl316-np15-rf0-signed (query)           3_224.51       124.28     3_348.80       0.0506             NaN         2.09
IVF-Binary-256-nl316-np17-rf0-signed (query)           3_224.51       128.19     3_352.71       0.0492             NaN         2.09
IVF-Binary-256-nl316-np25-rf0-signed (query)           3_224.51       131.81     3_356.32       0.0461             NaN         2.09
IVF-Binary-256-nl316-np15-rf10-signed (query)          3_224.51       189.02     3_413.53       0.3031          1.0093         2.09
IVF-Binary-256-nl316-np15-rf20-signed (query)          3_224.51       236.96     3_461.48       0.4821          1.0050         2.09
IVF-Binary-256-nl316-np17-rf10-signed (query)          3_224.51       187.26     3_411.77       0.2972          1.0095         2.09
IVF-Binary-256-nl316-np17-rf20-signed (query)          3_224.51       240.15     3_464.66       0.4738          1.0052         2.09
IVF-Binary-256-nl316-np25-rf10-signed (query)          3_224.51       195.90     3_420.41       0.2830          1.0103         2.09
IVF-Binary-256-nl316-np25-rf20-signed (query)          3_224.51       252.97     3_477.48       0.4542          1.0056         2.09
IVF-Binary-256-nl316-signed (self)                     3_224.51       556.47     3_780.99       0.7435          1.0416         2.09
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
Exhaustive (query)                                        19.01    10_131.02    10_150.03       1.0000          1.0000        97.66
Exhaustive (self)                                         19.01    33_247.61    33_266.62       1.0000          1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_682.80       353.59     6_036.39       0.0197             NaN         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_682.80       486.57     6_169.37       0.1454          1.0108         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_682.80       612.14     6_294.94       0.2497          1.0068         2.03
ExhaustiveBinary-256-random (self)                     5_682.80     1_622.44     7_305.24       0.6222          7.2120         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_208.14       361.45     6_569.58       0.0177             NaN         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_208.14       494.26     6_702.39       0.0885          1.0249         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_208.14       623.51     6_831.65       0.1361          1.0117         2.03
ExhaustiveBinary-256-pca (self)                        6_208.14     1_622.99     7_831.12       0.2209          1.7046         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_093.55       656.42    11_749.97       0.0425             NaN         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_093.55       798.98    11_892.53       0.2372          1.0067         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_093.55       927.19    12_020.74       0.3726          1.0039         4.05
ExhaustiveBinary-512-random (self)                    11_093.55     2_610.42    13_703.96       0.7112          9.2589         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_757.25       671.48    12_428.73       0.0149             NaN         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_757.25       800.76    12_558.01       0.0785          1.0474         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_757.25       928.46    12_685.71       0.1227          1.0180         4.05
ExhaustiveBinary-512-pca (self)                       11_757.25     2_646.78    14_404.03       0.1561          3.9887         4.05
ExhaustiveBinary-1024-random_no_rr (query)            21_977.93     1_197.33    23_175.26       0.0662             NaN         8.10
ExhaustiveBinary-1024-random-rf10 (query)             21_977.93     1_321.08    23_299.01       0.3226          1.0049         8.10
ExhaustiveBinary-1024-random-rf20 (query)             21_977.93     1_472.84    23_450.77       0.4753          1.0028         8.10
ExhaustiveBinary-1024-random (self)                   21_977.93     4_386.18    26_364.12       0.7729         10.8323         8.10
ExhaustiveBinary-1024-pca_no_rr (query)               23_064.02     1_203.38    24_267.39       0.0749             NaN         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                23_064.02     1_355.41    24_419.42       0.3494          1.0037         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                23_064.02     1_500.33    24_564.34       0.5064          1.0020         8.11
ExhaustiveBinary-1024-pca (self)                      23_064.02     4_486.06    27_550.08       0.6913          1.0488         8.11
ExhaustiveBinary-512-signed_no_rr (query)             11_058.65       658.44    11_717.09       0.0425             NaN         4.05
ExhaustiveBinary-512-signed-rf10 (query)              11_058.65       785.85    11_844.50       0.2372          1.0067         4.05
ExhaustiveBinary-512-signed-rf20 (query)              11_058.65       930.45    11_989.10       0.3726          1.0039         4.05
ExhaustiveBinary-512-signed (self)                    11_058.65     2_617.61    13_676.26       0.7112          9.2589         4.05
IVF-Binary-256-nl158-np7-rf0-random (query)            8_604.48       230.81     8_835.29       0.0301             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           8_604.48       242.92     8_847.40       0.0256             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           8_604.48       249.22     8_853.70       0.0238             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           8_604.48       326.49     8_930.97       0.2209          1.0057         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           8_604.48       400.81     9_005.29       0.3776          1.0030         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          8_604.48       331.33     8_935.81       0.1942          1.0067         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          8_604.48       423.15     9_027.63       0.3353          1.0037         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          8_604.48       347.44     8_951.92       0.1837          1.0072         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          8_604.48       441.02     9_045.50       0.3184          1.0041         2.34
IVF-Binary-256-nl158-random (self)                     8_604.48     1_012.86     9_617.34       0.7818          1.0318         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           6_339.59       243.68     6_583.27       0.0300             NaN         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)           6_339.59       245.65     6_585.24       0.0275             NaN         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)           6_339.59       249.52     6_589.10       0.0250             NaN         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)          6_339.59       331.47     6_671.05       0.2234          1.0055         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)          6_339.59       406.62     6_746.20       0.3835          1.0030         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)          6_339.59       330.10     6_669.68       0.2086          1.0060         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)          6_339.59       422.59     6_762.18       0.3606          1.0033         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)          6_339.59       341.85     6_681.43       0.1936          1.0066         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)          6_339.59       431.89     6_771.48       0.3371          1.0037         2.46
IVF-Binary-256-nl223-random (self)                     6_339.59       996.16     7_335.75       0.7768          1.0338         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)           6_598.96       252.88     6_851.84       0.0300             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           6_598.96       254.06     6_853.03       0.0285             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           6_598.96       259.23     6_858.19       0.0258             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          6_598.96       351.84     6_950.80       0.2243          1.0055         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          6_598.96       430.48     7_029.44       0.3865          1.0030         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          6_598.96       347.11     6_946.07       0.2161          1.0058         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          6_598.96       443.90     7_042.86       0.3737          1.0031         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          6_598.96       354.89     6_953.86       0.1995          1.0064         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          6_598.96       444.48     7_043.44       0.3469          1.0035         2.65
IVF-Binary-256-nl316-random (self)                     6_598.96     1_045.52     7_644.48       0.7769          1.0338         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               9_121.64       237.15     9_358.79       0.0472             NaN         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              9_121.64       245.98     9_367.62       0.0381             NaN         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              9_121.64       255.63     9_377.27       0.0338             NaN         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              9_121.64       344.55     9_466.19       0.2566          1.0041         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              9_121.64       418.56     9_540.21       0.3991          1.0024         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             9_121.64       345.61     9_467.25       0.2052          1.0053         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             9_121.64       438.65     9_560.29       0.3218          1.0032         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             9_121.64       355.97     9_477.61       0.1794          1.0063         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             9_121.64       455.71     9_577.35       0.2808          1.0038         2.34
IVF-Binary-256-nl158-pca (self)                        9_121.64     1_090.96    10_212.61       0.3368          1.2171         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              6_870.08       256.01     7_126.09       0.0490             NaN         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              6_870.08       247.18     7_117.26       0.0447             NaN         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              6_870.08       252.71     7_122.79       0.0396             NaN         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             6_870.08       337.65     7_207.74       0.2671          1.0040         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             6_870.08       423.94     7_294.02       0.4181          1.0023         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             6_870.08       341.71     7_211.79       0.2440          1.0044         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             6_870.08       432.82     7_302.91       0.3825          1.0026         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             6_870.08       352.59     7_222.67       0.2126          1.0052         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             6_870.08       447.03     7_317.11       0.3335          1.0031         2.47
IVF-Binary-256-nl223-pca (self)                        6_870.08     1_109.87     7_979.95       0.3689          1.1743         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              7_176.99       257.81     7_434.79       0.0495             NaN         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              7_176.99       261.99     7_438.97       0.0473             NaN         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              7_176.99       265.68     7_442.67       0.0417             NaN         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             7_176.99       359.53     7_536.52       0.2722          1.0040         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             7_176.99       440.52     7_617.50       0.4260          1.0023         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             7_176.99       355.20     7_532.19       0.2600          1.0042         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             7_176.99       441.94     7_618.92       0.4074          1.0024         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             7_176.99       362.25     7_539.23       0.2272          1.0049         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             7_176.99       459.15     7_636.13       0.3563          1.0029         2.65
IVF-Binary-256-nl316-pca (self)                        7_176.99     1_120.06     8_297.05       0.3805          1.1646         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           14_140.62       431.25    14_571.87       0.0549             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          14_140.62       446.96    14_587.57       0.0502             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          14_140.62       464.69    14_605.31       0.0484             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          14_140.62       519.66    14_660.28       0.3198          1.0037         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          14_140.62       594.32    14_734.94       0.5005          1.0019         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         14_140.62       535.29    14_675.90       0.2973          1.0041         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         14_140.62       620.07    14_760.69       0.4682          1.0022         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         14_140.62       554.80    14_695.41       0.2885          1.0043         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         14_140.62       651.07    14_791.68       0.4548          1.0024         4.36
IVF-Binary-512-nl158-random (self)                    14_140.62     1_697.33    15_837.95       0.8737          1.0151         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          11_826.68       440.21    12_266.88       0.0548             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          11_826.68       440.02    12_266.69       0.0521             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          11_826.68       451.62    12_278.30       0.0498             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         11_826.68       525.56    12_352.24       0.3197          1.0039         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         11_826.68       601.08    12_427.76       0.4995          1.0021         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         11_826.68       526.34    12_353.02       0.3081          1.0041         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         11_826.68       613.46    12_440.14       0.4830          1.0022         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         11_826.68       546.42    12_373.09       0.2968          1.0042         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         11_826.68       638.51    12_465.19       0.4670          1.0023         4.49
IVF-Binary-512-nl223-random (self)                    11_826.68     1_656.45    13_483.12       0.8700          1.0162         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          12_192.25       454.88    12_647.13       0.0550             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          12_192.25       474.99    12_667.24       0.0534             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          12_192.25       465.14    12_657.39       0.0507             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         12_192.25       539.82    12_732.07       0.3208          1.0039         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         12_192.25       617.59    12_809.84       0.5016          1.0021         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         12_192.25       544.33    12_736.58       0.3142          1.0040         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         12_192.25       624.46    12_816.71       0.4927          1.0021         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         12_192.25       551.58    12_743.83       0.3011          1.0042         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         12_192.25       648.83    12_841.08       0.4734          1.0023         4.67
IVF-Binary-512-nl316-random (self)                    12_192.25     1_700.08    13_892.33       0.8709          1.0159         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              14_737.12       453.73    15_190.85       0.0556             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             14_737.12       466.17    15_203.29       0.0428             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             14_737.12       474.52    15_211.64       0.0364             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             14_737.12       535.84    15_272.96       0.2850          1.0038         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             14_737.12       624.50    15_361.61       0.4340          1.0022         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            14_737.12       555.24    15_292.36       0.2221          1.0052         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            14_737.12       651.20    15_388.32       0.3406          1.0031         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            14_737.12       572.80    15_309.92       0.1885          1.0067         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            14_737.12       678.89    15_416.01       0.2903          1.0038         4.36
IVF-Binary-512-nl158-pca (self)                       14_737.12     1_817.83    16_554.94       0.2809          1.2965         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             12_599.18       449.62    13_048.80       0.0584             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             12_599.18       452.40    13_051.59       0.0527             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             12_599.18       459.70    13_058.89       0.0447             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            12_599.18       541.88    13_141.07       0.3002          1.0037         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            12_599.18       623.10    13_222.29       0.4592          1.0021         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            12_599.18       546.88    13_146.06       0.2715          1.0042         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            12_599.18       635.13    13_234.31       0.4171          1.0024         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            12_599.18       559.82    13_159.00       0.2310          1.0051         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            12_599.18       659.45    13_258.64       0.3556          1.0030         4.49
IVF-Binary-512-nl223-pca (self)                       12_599.18     1_769.79    14_368.97       0.3193          1.2206         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             12_746.26       464.54    13_210.80       0.0593             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             12_746.26       486.73    13_232.99       0.0561             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             12_746.26       472.10    13_218.36       0.0480             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            12_746.26       559.00    13_305.25       0.3068          1.0036         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            12_746.26       637.58    13_383.84       0.4675          1.0020         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            12_746.26       561.99    13_308.25       0.2915          1.0039         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            12_746.26       644.37    13_390.62       0.4457          1.0022         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            12_746.26       577.26    13_323.52       0.2497          1.0047         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            12_746.26       662.99    13_409.25       0.3844          1.0027         4.67
IVF-Binary-512-nl316-pca (self)                       12_746.26     1_807.52    14_553.77       0.3336          1.2043         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          24_916.16       838.21    25_754.37       0.0800             NaN         8.41
IVF-Binary-1024-nl158-np12-rf0-random (query)         24_916.16       842.04    25_758.20       0.0751             NaN         8.41
IVF-Binary-1024-nl158-np17-rf0-random (query)         24_916.16       865.31    25_781.47       0.0734             NaN         8.41
IVF-Binary-1024-nl158-np7-rf10-random (query)         24_916.16       925.34    25_841.50       0.4064          1.0027         8.41
IVF-Binary-1024-nl158-np7-rf20-random (query)         24_916.16       993.29    25_909.46       0.5999          1.0013         8.41
IVF-Binary-1024-nl158-np12-rf10-random (query)        24_916.16       945.53    25_861.69       0.3858          1.0030         8.41
IVF-Binary-1024-nl158-np12-rf20-random (query)        24_916.16     1_026.93    25_943.09       0.5727          1.0015         8.41
IVF-Binary-1024-nl158-np17-rf10-random (query)        24_916.16       977.60    25_893.77       0.3774          1.0032         8.41
IVF-Binary-1024-nl158-np17-rf20-random (query)        24_916.16     1_069.04    25_985.21       0.5607          1.0017         8.41
IVF-Binary-1024-nl158-random (self)                   24_916.16     3_068.71    27_984.87       0.9359          1.0061         8.41
IVF-Binary-1024-nl223-np11-rf0-random (query)         22_732.51       835.72    23_568.23       0.0800             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         22_732.51       845.42    23_577.93       0.0769             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         22_732.51       862.83    23_595.34       0.0744             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        22_732.51       923.33    23_655.84       0.4059          1.0029         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        22_732.51     1_012.60    23_745.10       0.5973          1.0015         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        22_732.51       940.59    23_673.10       0.3953          1.0030         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        22_732.51     1_032.22    23_764.73       0.5842          1.0016         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        22_732.51       968.72    23_701.23       0.3850          1.0031         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        22_732.51     1_045.05    23_777.56       0.5712          1.0016         8.54
IVF-Binary-1024-nl223-random (self)                   22_732.51     3_005.56    25_738.06       0.9334          1.0065         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         22_937.29       859.20    23_796.49       0.0800             NaN         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)         22_937.29       856.39    23_793.67       0.0783             NaN         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)         22_937.29       866.27    23_803.56       0.0751             NaN         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)        22_937.29       942.66    23_879.94       0.4078          1.0029         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)        22_937.29     1_030.33    23_967.62       0.5992          1.0015         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)        22_937.29       942.56    23_879.85       0.4018          1.0030         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)        22_937.29     1_030.70    23_967.99       0.5924          1.0015         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)        22_937.29       965.79    23_903.08       0.3896          1.0031         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)        22_937.29     1_060.94    23_998.23       0.5772          1.0016         8.72
IVF-Binary-1024-nl316-random (self)                   22_937.29     3_053.05    25_990.34       0.9340          1.0064         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)             26_120.82       850.99    26_971.81       0.0966             NaN         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            26_120.82       872.95    26_993.77       0.0894             NaN         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            26_120.82       891.25    27_012.07       0.0865             NaN         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            26_120.82       951.18    27_072.00       0.4417          1.0022         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            26_120.82     1_026.86    27_147.67       0.6284          1.0011         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           26_120.82       963.56    27_084.38       0.4116          1.0025         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           26_120.82     1_056.60    27_177.42       0.5897          1.0013         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           26_120.82       996.94    27_117.76       0.3990          1.0027         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           26_120.82     1_093.18    27_214.00       0.5718          1.0014         8.42
IVF-Binary-1024-nl158-pca (self)                      26_120.82     3_155.57    29_276.39       0.7125          1.0421         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            23_818.40       854.00    24_672.40       0.0975             NaN         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            23_818.40       862.26    24_680.66       0.0941             NaN         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            23_818.40       875.54    24_693.94       0.0903             NaN         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           23_818.40       975.37    24_793.77       0.4463          1.0022         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           23_818.40     1_041.23    24_859.63       0.6359          1.0011         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           23_818.40       995.82    24_814.22       0.4331          1.0024         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           23_818.40     1_057.45    24_875.85       0.6184          1.0012         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           23_818.40       975.94    24_794.34       0.4166          1.0025         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           23_818.40     1_067.11    24_885.51       0.5971          1.0013         8.54
IVF-Binary-1024-nl223-pca (self)                      23_818.40     3_088.00    26_906.40       0.7111          1.0429         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            24_124.76       869.85    24_994.61       0.0980             NaN         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            24_124.76       879.78    25_004.54       0.0962             NaN         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            24_124.76       879.37    25_004.13       0.0919             NaN         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           24_124.76       962.74    25_087.50       0.4498          1.0022         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           24_124.76     1_049.01    25_173.77       0.6402          1.0011         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           24_124.76       966.36    25_091.12       0.4427          1.0023         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           24_124.76     1_048.24    25_173.00       0.6311          1.0011         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           24_124.76       973.55    25_098.31       0.4251          1.0024         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           24_124.76     1_069.46    25_194.23       0.6080          1.0012         8.73
IVF-Binary-1024-nl316-pca (self)                      24_124.76     3_112.66    27_237.42       0.7121          1.0426         8.73
IVF-Binary-512-nl158-np7-rf0-signed (query)           14_030.79       428.48    14_459.27       0.0549             NaN         4.36
IVF-Binary-512-nl158-np12-rf0-signed (query)          14_030.79       446.13    14_476.92       0.0502             NaN         4.36
IVF-Binary-512-nl158-np17-rf0-signed (query)          14_030.79       462.02    14_492.81       0.0484             NaN         4.36
IVF-Binary-512-nl158-np7-rf10-signed (query)          14_030.79       531.53    14_562.32       0.3198          1.0037         4.36
IVF-Binary-512-nl158-np7-rf20-signed (query)          14_030.79       595.39    14_626.18       0.5005          1.0019         4.36
IVF-Binary-512-nl158-np12-rf10-signed (query)         14_030.79       536.54    14_567.33       0.2973          1.0041         4.36
IVF-Binary-512-nl158-np12-rf20-signed (query)         14_030.79       671.28    14_702.07       0.4682          1.0022         4.36
IVF-Binary-512-nl158-np17-rf10-signed (query)         14_030.79       556.89    14_587.68       0.2885          1.0043         4.36
IVF-Binary-512-nl158-np17-rf20-signed (query)         14_030.79       647.27    14_678.06       0.4548          1.0024         4.36
IVF-Binary-512-nl158-signed (self)                    14_030.79     1_689.40    15_720.19       0.8737          1.0151         4.36
IVF-Binary-512-nl223-np11-rf0-signed (query)          11_821.83       437.46    12_259.29       0.0548             NaN         4.49
IVF-Binary-512-nl223-np14-rf0-signed (query)          11_821.83       441.83    12_263.67       0.0521             NaN         4.49
IVF-Binary-512-nl223-np21-rf0-signed (query)          11_821.83       446.69    12_268.53       0.0498             NaN         4.49
IVF-Binary-512-nl223-np11-rf10-signed (query)         11_821.83       525.60    12_347.43       0.3197          1.0039         4.49
IVF-Binary-512-nl223-np11-rf20-signed (query)         11_821.83       599.83    12_421.66       0.4995          1.0021         4.49
IVF-Binary-512-nl223-np14-rf10-signed (query)         11_821.83       528.77    12_350.61       0.3081          1.0041         4.49
IVF-Binary-512-nl223-np14-rf20-signed (query)         11_821.83       615.28    12_437.11       0.4830          1.0022         4.49
IVF-Binary-512-nl223-np21-rf10-signed (query)         11_821.83       540.55    12_362.38       0.2968          1.0042         4.49
IVF-Binary-512-nl223-np21-rf20-signed (query)         11_821.83       631.20    12_453.04       0.4670          1.0023         4.49
IVF-Binary-512-nl223-signed (self)                    11_821.83     1_666.05    13_487.89       0.8700          1.0162         4.49
IVF-Binary-512-nl316-np15-rf0-signed (query)          12_104.43       449.71    12_554.15       0.0550             NaN         4.67
IVF-Binary-512-nl316-np17-rf0-signed (query)          12_104.43       451.01    12_555.44       0.0534             NaN         4.67
IVF-Binary-512-nl316-np25-rf0-signed (query)          12_104.43       458.50    12_562.93       0.0507             NaN         4.67
IVF-Binary-512-nl316-np15-rf10-signed (query)         12_104.43       539.87    12_644.31       0.3208          1.0039         4.67
IVF-Binary-512-nl316-np15-rf20-signed (query)         12_104.43       621.82    12_726.25       0.5016          1.0021         4.67
IVF-Binary-512-nl316-np17-rf10-signed (query)         12_104.43       539.13    12_643.56       0.3142          1.0040         4.67
IVF-Binary-512-nl316-np17-rf20-signed (query)         12_104.43       624.60    12_729.04       0.4927          1.0021         4.67
IVF-Binary-512-nl316-np25-rf10-signed (query)         12_104.43       553.74    12_658.17       0.3011          1.0042         4.67
IVF-Binary-512-nl316-np25-rf20-signed (query)         12_104.43       638.55    12_742.98       0.4734          1.0023         4.67
IVF-Binary-512-nl316-signed (self)                    12_104.43     1_706.25    13_810.68       0.8709          1.0159         4.67
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
Exhaustive (query)                                        39.11    22_776.90    22_816.00       1.0000          1.0000       195.31
Exhaustive (self)                                         39.11    73_864.26    73_903.37       1.0000          1.0000       195.31
ExhaustiveBinary-256-random_no_rr (query)             11_963.03       575.16    12_538.19       0.0086             NaN         2.53
ExhaustiveBinary-256-random-rf10 (query)              11_963.03       741.70    12_704.73       0.0837          1.0101         2.53
ExhaustiveBinary-256-random-rf20 (query)              11_963.03       913.62    12_876.65       0.1557          1.0064         2.53
ExhaustiveBinary-256-random (self)                    11_963.03     2_436.06    14_399.09       0.6166          5.6565         2.53
ExhaustiveBinary-256-pca_no_rr (query)                13_074.01       594.57    13_668.58       0.0142             NaN         2.53
ExhaustiveBinary-256-pca-rf10 (query)                 13_074.01       751.22    13_825.23       0.0739          1.0100         2.53
ExhaustiveBinary-256-pca-rf20 (query)                 13_074.01       915.76    13_989.77       0.1169          1.0048         2.53
ExhaustiveBinary-256-pca (self)                       13_074.01     2_468.91    15_542.92       0.1825          1.5785         2.53
ExhaustiveBinary-512-random_no_rr (query)             23_454.38     1_094.82    24_549.20       0.0192             NaN         5.05
ExhaustiveBinary-512-random-rf10 (query)              23_454.38     1_256.20    24_710.58       0.1384          1.0052         5.05
ExhaustiveBinary-512-random-rf20 (query)              23_454.38     1_425.30    24_879.68       0.2398          1.0032         5.05
ExhaustiveBinary-512-random (self)                    23_454.38     4_166.49    27_620.87       0.7044          7.2752         5.05
ExhaustiveBinary-512-pca_no_rr (query)                24_799.67     1_109.89    25_909.56       0.0097             NaN         5.06
ExhaustiveBinary-512-pca-rf10 (query)                 24_799.67     1_272.87    26_072.54       0.0527          1.0187         5.06
ExhaustiveBinary-512-pca-rf20 (query)                 24_799.67     1_441.47    26_241.14       0.0858          1.0069         5.06
ExhaustiveBinary-512-pca (self)                       24_799.67     4_210.25    29_009.92       0.1295          2.6401         5.06
ExhaustiveBinary-1024-random_no_rr (query)            46_961.69     2_045.43    49_007.12       0.0426             NaN        10.10
ExhaustiveBinary-1024-random-rf10 (query)             46_961.69     2_229.54    49_191.23       0.2373          1.0030        10.10
ExhaustiveBinary-1024-random-rf20 (query)             46_961.69     2_431.23    49_392.92       0.3715          1.0017        10.10
ExhaustiveBinary-1024-random (self)                   46_961.69     7_385.84    54_347.53       0.7686          8.3967        10.10
ExhaustiveBinary-1024-pca_no_rr (query)               48_548.95     2_105.07    50_654.02       0.0086             NaN        10.11
ExhaustiveBinary-1024-pca-rf10 (query)                48_548.95     2_291.88    50_840.83       0.0490          1.0332        10.11
ExhaustiveBinary-1024-pca-rf20 (query)                48_548.95     2_491.09    51_040.05       0.0797          1.0112        10.11
ExhaustiveBinary-1024-pca (self)                      48_548.95     7_667.90    56_216.85       0.1033          8.4885        10.11
ExhaustiveBinary-1024-signed_no_rr (query)            47_078.28     2_069.22    49_147.50       0.0426             NaN        10.10
ExhaustiveBinary-1024-signed-rf10 (query)             47_078.28     2_245.91    49_324.19       0.2373          1.0030        10.10
ExhaustiveBinary-1024-signed-rf20 (query)             47_078.28     2_431.72    49_510.00       0.3715          1.0017        10.10
ExhaustiveBinary-1024-signed (self)                   47_078.28     7_445.78    54_524.06       0.7686          8.3967        10.10
IVF-Binary-256-nl158-np7-rf0-random (query)           18_516.19       479.61    18_995.80       0.0173             NaN         3.14
IVF-Binary-256-nl158-np12-rf0-random (query)          18_516.19       490.06    19_006.25       0.0139             NaN         3.14
IVF-Binary-256-nl158-np17-rf0-random (query)          18_516.19       498.60    19_014.79       0.0121             NaN         3.14
IVF-Binary-256-nl158-np7-rf10-random (query)          18_516.19       616.80    19_133.00       0.1524          1.0042         3.14
IVF-Binary-256-nl158-np7-rf20-random (query)          18_516.19       741.21    19_257.41       0.2792          1.0022         3.14
IVF-Binary-256-nl158-np12-rf10-random (query)         18_516.19       622.18    19_138.37       0.1270          1.0055         3.14
IVF-Binary-256-nl158-np12-rf20-random (query)         18_516.19       754.65    19_270.84       0.2368          1.0030         3.14
IVF-Binary-256-nl158-np17-rf10-random (query)         18_516.19       650.27    19_166.47       0.1153          1.0063         3.14
IVF-Binary-256-nl158-np17-rf20-random (query)         18_516.19       774.95    19_291.15       0.2169          1.0035         3.14
IVF-Binary-256-nl158-random (self)                    18_516.19     1_960.84    20_477.03       0.7825          1.0293         3.14
IVF-Binary-256-nl223-np11-rf0-random (query)          13_214.96       506.88    13_721.84       0.0184             NaN         3.40
IVF-Binary-256-nl223-np14-rf0-random (query)          13_214.96       507.17    13_722.13       0.0154             NaN         3.40
IVF-Binary-256-nl223-np21-rf0-random (query)          13_214.96       511.54    13_726.50       0.0131             NaN         3.40
IVF-Binary-256-nl223-np11-rf10-random (query)         13_214.96       635.82    13_850.78       0.1595          1.0038         3.40
IVF-Binary-256-nl223-np11-rf20-random (query)         13_214.96       764.01    13_978.97       0.2910          1.0020         3.40
IVF-Binary-256-nl223-np14-rf10-random (query)         13_214.96       639.48    13_854.45       0.1407          1.0045         3.40
IVF-Binary-256-nl223-np14-rf20-random (query)         13_214.96       772.87    13_987.84       0.2612          1.0024         3.40
IVF-Binary-256-nl223-np21-rf10-random (query)         13_214.96       648.13    13_863.09       0.1235          1.0055         3.40
IVF-Binary-256-nl223-np21-rf20-random (query)         13_214.96       785.61    14_000.57       0.2318          1.0030         3.40
IVF-Binary-256-nl223-random (self)                    13_214.96     2_018.58    15_233.54       0.7845          1.0289         3.40
IVF-Binary-256-nl316-np15-rf0-random (query)          13_679.68       535.00    14_214.68       0.0179             NaN         3.76
IVF-Binary-256-nl316-np17-rf0-random (query)          13_679.68       537.66    14_217.34       0.0164             NaN         3.76
IVF-Binary-256-nl316-np25-rf0-random (query)          13_679.68       547.61    14_227.29       0.0141             NaN         3.76
IVF-Binary-256-nl316-np15-rf10-random (query)         13_679.68       671.74    14_351.43       0.1596          1.0038         3.76
IVF-Binary-256-nl316-np15-rf20-random (query)         13_679.68       796.72    14_476.41       0.2931          1.0020         3.76
IVF-Binary-256-nl316-np17-rf10-random (query)         13_679.68       676.18    14_355.86       0.1503          1.0042         3.76
IVF-Binary-256-nl316-np17-rf20-random (query)         13_679.68       800.64    14_480.32       0.2782          1.0022         3.76
IVF-Binary-256-nl316-np25-rf10-random (query)         13_679.68       679.62    14_359.30       0.1309          1.0051         3.76
IVF-Binary-256-nl316-np25-rf20-random (query)         13_679.68       810.69    14_490.37       0.2449          1.0027         3.76
IVF-Binary-256-nl316-random (self)                    13_679.68     2_102.40    15_782.08       0.7842          1.0291         3.76
IVF-Binary-256-nl158-np7-rf0-pca (query)              19_793.97       488.30    20_282.27       0.0443             NaN         3.15
IVF-Binary-256-nl158-np12-rf0-pca (query)             19_793.97       494.64    20_288.61       0.0366             NaN         3.15
IVF-Binary-256-nl158-np17-rf0-pca (query)             19_793.97       532.25    20_326.22       0.0326             NaN         3.15
IVF-Binary-256-nl158-np7-rf10-pca (query)             19_793.97       637.20    20_431.17       0.2520          1.0019         3.15
IVF-Binary-256-nl158-np7-rf20-pca (query)             19_793.97       750.76    20_544.73       0.3962          1.0011         3.15
IVF-Binary-256-nl158-np12-rf10-pca (query)            19_793.97       634.62    20_428.59       0.2055          1.0023         3.15
IVF-Binary-256-nl158-np12-rf20-pca (query)            19_793.97       772.03    20_566.00       0.3237          1.0014         3.15
IVF-Binary-256-nl158-np17-rf10-pca (query)            19_793.97       645.32    20_439.30       0.1802          1.0026         3.15
IVF-Binary-256-nl158-np17-rf20-pca (query)            19_793.97       796.91    20_590.88       0.2829          1.0017         3.15
IVF-Binary-256-nl158-pca (self)                       19_793.97     2_065.42    21_859.39       0.2960          1.2346         3.15
IVF-Binary-256-nl223-np11-rf0-pca (query)             14_516.44       504.77    15_021.21       0.0455             NaN         3.40
IVF-Binary-256-nl223-np14-rf0-pca (query)             14_516.44       506.84    15_023.28       0.0413             NaN         3.40
IVF-Binary-256-nl223-np21-rf0-pca (query)             14_516.44       508.95    15_025.39       0.0357             NaN         3.40
IVF-Binary-256-nl223-np11-rf10-pca (query)            14_516.44       638.63    15_155.07       0.2590          1.0018         3.40
IVF-Binary-256-nl223-np11-rf20-pca (query)            14_516.44       758.72    15_275.16       0.4054          1.0011         3.40
IVF-Binary-256-nl223-np14-rf10-pca (query)            14_516.44       646.22    15_162.65       0.2334          1.0020         3.40
IVF-Binary-256-nl223-np14-rf20-pca (query)            14_516.44       777.32    15_293.76       0.3665          1.0012         3.40
IVF-Binary-256-nl223-np21-rf10-pca (query)            14_516.44       652.95    15_169.39       0.1996          1.0023         3.40
IVF-Binary-256-nl223-np21-rf20-pca (query)            14_516.44       787.38    15_303.82       0.3146          1.0015         3.40
IVF-Binary-256-nl223-pca (self)                       14_516.44     2_080.15    16_596.58       0.3159          1.2069         3.40
IVF-Binary-256-nl316-np15-rf0-pca (query)             14_938.34       535.20    15_473.54       0.0463             NaN         3.77
IVF-Binary-256-nl316-np17-rf0-pca (query)             14_938.34       542.53    15_480.87       0.0442             NaN         3.77
IVF-Binary-256-nl316-np25-rf0-pca (query)             14_938.34       549.58    15_487.91       0.0385             NaN         3.77
IVF-Binary-256-nl316-np15-rf10-pca (query)            14_938.34       669.05    15_607.38       0.2629          1.0018         3.77
IVF-Binary-256-nl316-np15-rf20-pca (query)            14_938.34       805.93    15_744.26       0.4118          1.0010         3.77
IVF-Binary-256-nl316-np17-rf10-pca (query)            14_938.34       673.20    15_611.53       0.2496          1.0019         3.77
IVF-Binary-256-nl316-np17-rf20-pca (query)            14_938.34       794.41    15_732.75       0.3914          1.0011         3.77
IVF-Binary-256-nl316-np25-rf10-pca (query)            14_938.34       673.02    15_611.36       0.2159          1.0022         3.77
IVF-Binary-256-nl316-np25-rf20-pca (query)            14_938.34       813.56    15_751.90       0.3396          1.0013         3.77
IVF-Binary-256-nl316-pca (self)                       14_938.34     2_160.41    17_098.74       0.3276          1.1941         3.77
IVF-Binary-512-nl158-np7-rf0-random (query)           29_999.88       890.95    30_890.83       0.0290             NaN         5.67
IVF-Binary-512-nl158-np12-rf0-random (query)          29_999.88       898.87    30_898.75       0.0250             NaN         5.67
IVF-Binary-512-nl158-np17-rf0-random (query)          29_999.88       908.30    30_908.18       0.0234             NaN         5.67
IVF-Binary-512-nl158-np7-rf10-random (query)          29_999.88     1_016.98    31_016.86       0.2095          1.0028         5.67
IVF-Binary-512-nl158-np7-rf20-random (query)          29_999.88     1_127.01    31_126.89       0.3608          1.0015         5.67
IVF-Binary-512-nl158-np12-rf10-random (query)         29_999.88     1_029.99    31_029.87       0.1863          1.0032         5.67
IVF-Binary-512-nl158-np12-rf20-random (query)         29_999.88     1_182.97    31_182.85       0.3245          1.0018         5.67
IVF-Binary-512-nl158-np17-rf10-random (query)         29_999.88     1_050.06    31_049.94       0.1757          1.0035         5.67
IVF-Binary-512-nl158-np17-rf20-random (query)         29_999.88     1_198.30    31_198.19       0.3078          1.0019         5.67
IVF-Binary-512-nl158-random (self)                    29_999.88     3_330.57    33_330.45       0.8789          1.0130         5.67
IVF-Binary-512-nl223-np11-rf0-random (query)          25_370.47       918.53    26_289.00       0.0303             NaN         5.92
IVF-Binary-512-nl223-np14-rf0-random (query)          25_370.47       919.51    26_289.97       0.0269             NaN         5.92
IVF-Binary-512-nl223-np21-rf0-random (query)          25_370.47       934.01    26_304.48       0.0244             NaN         5.92
IVF-Binary-512-nl223-np11-rf10-random (query)         25_370.47     1_057.52    26_427.98       0.2171          1.0026         5.92
IVF-Binary-512-nl223-np11-rf20-random (query)         25_370.47     1_207.54    26_578.01       0.3729          1.0014         5.92
IVF-Binary-512-nl223-np14-rf10-random (query)         25_370.47     1_054.31    26_424.77       0.1998          1.0029         5.92
IVF-Binary-512-nl223-np14-rf20-random (query)         25_370.47     1_183.25    26_553.72       0.3473          1.0016         5.92
IVF-Binary-512-nl223-np21-rf10-random (query)         25_370.47     1_065.77    26_436.24       0.1839          1.0032         5.92
IVF-Binary-512-nl223-np21-rf20-random (query)         25_370.47     1_198.66    26_569.13       0.3219          1.0018         5.92
IVF-Binary-512-nl223-random (self)                    25_370.47     3_381.65    28_752.12       0.8802          1.0128         5.92
IVF-Binary-512-nl316-np15-rf0-random (query)          25_739.50       951.30    26_690.80       0.0297             NaN         6.29
IVF-Binary-512-nl316-np17-rf0-random (query)          25_739.50       948.82    26_688.32       0.0283             NaN         6.29
IVF-Binary-512-nl316-np25-rf0-random (query)          25_739.50       958.16    26_697.67       0.0255             NaN         6.29
IVF-Binary-512-nl316-np15-rf10-random (query)         25_739.50     1_084.50    26_824.00       0.2169          1.0026         6.29
IVF-Binary-512-nl316-np15-rf20-random (query)         25_739.50     1_205.66    26_945.16       0.3751          1.0014         6.29
IVF-Binary-512-nl316-np17-rf10-random (query)         25_739.50     1_082.17    26_821.67       0.2084          1.0028         6.29
IVF-Binary-512-nl316-np17-rf20-random (query)         25_739.50     1_203.96    26_943.46       0.3621          1.0015         6.29
IVF-Binary-512-nl316-np25-rf10-random (query)         25_739.50     1_096.16    26_835.67       0.1909          1.0031         6.29
IVF-Binary-512-nl316-np25-rf20-random (query)         25_739.50     1_224.16    26_963.67       0.3340          1.0017         6.29
IVF-Binary-512-nl316-random (self)                    25_739.50     3_484.70    29_224.20       0.8803          1.0128         6.29
IVF-Binary-512-nl158-np7-rf0-pca (query)              31_579.73       895.55    32_475.28       0.0423             NaN         5.67
IVF-Binary-512-nl158-np12-rf0-pca (query)             31_579.73       915.10    32_494.83       0.0333             NaN         5.67
IVF-Binary-512-nl158-np17-rf0-pca (query)             31_579.73       925.22    32_504.95       0.0283             NaN         5.67
IVF-Binary-512-nl158-np7-rf10-pca (query)             31_579.73     1_029.05    32_608.77       0.2381          1.0020         5.67
IVF-Binary-512-nl158-np7-rf20-pca (query)             31_579.73     1_157.17    32_736.89       0.3779          1.0012         5.67
IVF-Binary-512-nl158-np12-rf10-pca (query)            31_579.73     1_046.98    32_626.70       0.1856          1.0025         5.67
IVF-Binary-512-nl158-np12-rf20-pca (query)            31_579.73     1_173.92    32_753.64       0.2955          1.0016         5.67
IVF-Binary-512-nl158-np17-rf10-pca (query)            31_579.73     1_060.99    32_640.71       0.1572          1.0029         5.67
IVF-Binary-512-nl158-np17-rf20-pca (query)            31_579.73     1_212.33    32_792.06       0.2502          1.0019         5.67
IVF-Binary-512-nl158-pca (self)                       31_579.73     3_439.96    35_019.68       0.2404          1.3063         5.67
IVF-Binary-512-nl223-np11-rf0-pca (query)             26_341.62       937.93    27_279.55       0.0435             NaN         5.93
IVF-Binary-512-nl223-np14-rf0-pca (query)             26_341.62       926.83    27_268.45       0.0384             NaN         5.93
IVF-Binary-512-nl223-np21-rf0-pca (query)             26_341.62       942.92    27_284.55       0.0320             NaN         5.93
IVF-Binary-512-nl223-np11-rf10-pca (query)            26_341.62     1_056.76    27_398.38       0.2439          1.0019         5.93
IVF-Binary-512-nl223-np11-rf20-pca (query)            26_341.62     1_174.00    27_515.62       0.3867          1.0011         5.93
IVF-Binary-512-nl223-np14-rf10-pca (query)            26_341.62     1_059.60    27_401.23       0.2158          1.0021         5.93
IVF-Binary-512-nl223-np14-rf20-pca (query)            26_341.62     1_181.40    27_523.02       0.3431          1.0013         5.93
IVF-Binary-512-nl223-np21-rf10-pca (query)            26_341.62     1_073.85    27_415.47       0.1789          1.0026         5.93
IVF-Binary-512-nl223-np21-rf20-pca (query)            26_341.62     1_206.49    27_548.11       0.2844          1.0016         5.93
IVF-Binary-512-nl223-pca (self)                       26_341.62     3_476.52    29_818.14       0.2632          1.2615         5.93
IVF-Binary-512-nl316-np15-rf0-pca (query)             26_859.94       952.43    27_812.37       0.0442             NaN         6.29
IVF-Binary-512-nl316-np17-rf0-pca (query)             26_859.94       955.34    27_815.29       0.0416             NaN         6.29
IVF-Binary-512-nl316-np25-rf0-pca (query)             26_859.94       969.64    27_829.58       0.0352             NaN         6.29
IVF-Binary-512-nl316-np15-rf10-pca (query)            26_859.94     1_083.18    27_943.13       0.2482          1.0019         6.29
IVF-Binary-512-nl316-np15-rf20-pca (query)            26_859.94     1_205.23    28_065.17       0.3931          1.0011         6.29
IVF-Binary-512-nl316-np17-rf10-pca (query)            26_859.94     1_080.31    27_940.25       0.2339          1.0020         6.29
IVF-Binary-512-nl316-np17-rf20-pca (query)            26_859.94     1_305.42    28_165.37       0.3706          1.0012         6.29
IVF-Binary-512-nl316-np25-rf10-pca (query)            26_859.94     1_094.86    27_954.81       0.1970          1.0023         6.29
IVF-Binary-512-nl316-np25-rf20-pca (query)            26_859.94     1_241.20    28_101.14       0.3128          1.0015         6.29
IVF-Binary-512-nl316-pca (self)                       26_859.94     3_568.24    30_428.18       0.2759          1.2425         6.29
IVF-Binary-1024-nl158-np7-rf0-random (query)          53_658.24     1_724.34    55_382.58       0.0551             NaN        10.72
IVF-Binary-1024-nl158-np12-rf0-random (query)         53_658.24     1_865.81    55_524.05       0.0511             NaN        10.72
IVF-Binary-1024-nl158-np17-rf0-random (query)         53_658.24     1_753.09    55_411.33       0.0493             NaN        10.72
IVF-Binary-1024-nl158-np7-rf10-random (query)         53_658.24     1_830.06    55_488.30       0.3140          1.0018        10.72
IVF-Binary-1024-nl158-np7-rf20-random (query)         53_658.24     1_949.06    55_607.30       0.4899          1.0009        10.72
IVF-Binary-1024-nl158-np12-rf10-random (query)        53_658.24     1_845.24    55_503.48       0.2962          1.0019        10.72
IVF-Binary-1024-nl158-np12-rf20-random (query)        53_658.24     1_979.38    55_637.62       0.4653          1.0010        10.72
IVF-Binary-1024-nl158-np17-rf10-random (query)        53_658.24     1_886.52    55_544.76       0.2875          1.0020        10.72
IVF-Binary-1024-nl158-np17-rf20-random (query)        53_658.24     2_011.60    55_669.84       0.4530          1.0010        10.72
IVF-Binary-1024-nl158-random (self)                   53_658.24     6_073.48    59_731.72       0.9463          1.0044        10.72
IVF-Binary-1024-nl223-np11-rf0-random (query)         48_436.82     1_794.74    50_231.57       0.0568             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-random (query)         48_436.82     1_787.70    50_224.53       0.0531             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-random (query)         48_436.82     1_786.63    50_223.45       0.0506             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-random (query)        48_436.82     1_899.63    50_336.45       0.3220          1.0017        10.98
IVF-Binary-1024-nl223-np11-rf20-random (query)        48_436.82     2_008.25    50_445.07       0.5002          1.0009        10.98
IVF-Binary-1024-nl223-np14-rf10-random (query)        48_436.82     1_897.30    50_334.13       0.3072          1.0018        10.98
IVF-Binary-1024-nl223-np14-rf20-random (query)        48_436.82     2_025.27    50_462.09       0.4808          1.0009        10.98
IVF-Binary-1024-nl223-np21-rf10-random (query)        48_436.82     2_107.02    50_543.85       0.2944          1.0019        10.98
IVF-Binary-1024-nl223-np21-rf20-random (query)        48_436.82     2_278.13    50_714.96       0.4632          1.0010        10.98
IVF-Binary-1024-nl223-random (self)                   48_436.82     6_466.83    54_903.65       0.9472          1.0043        10.98
IVF-Binary-1024-nl316-np15-rf0-random (query)         49_159.57     1_783.44    50_943.00       0.0561             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-random (query)         49_159.57     1_769.42    50_928.99       0.0545             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-random (query)         49_159.57     1_785.15    50_944.72       0.0516             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-random (query)        49_159.57     1_885.87    51_045.44       0.3220          1.0017        11.34
IVF-Binary-1024-nl316-np15-rf20-random (query)        49_159.57     2_001.37    51_160.93       0.5018          1.0009        11.34
IVF-Binary-1024-nl316-np17-rf10-random (query)        49_159.57     1_895.38    51_054.95       0.3150          1.0018        11.34
IVF-Binary-1024-nl316-np17-rf20-random (query)        49_159.57     2_035.06    51_194.63       0.4923          1.0009        11.34
IVF-Binary-1024-nl316-np25-rf10-random (query)        49_159.57     1_904.47    51_064.04       0.3007          1.0019        11.34
IVF-Binary-1024-nl316-np25-rf20-random (query)        49_159.57     2_039.15    51_198.72       0.4728          1.0010        11.34
IVF-Binary-1024-nl316-random (self)                   49_159.57     6_221.70    55_381.27       0.9474          1.0043        11.34
IVF-Binary-1024-nl158-np7-rf0-pca (query)             55_031.57     1_740.63    56_772.19       0.0479             NaN        10.73
IVF-Binary-1024-nl158-np12-rf0-pca (query)            55_031.57     1_763.63    56_795.20       0.0363             NaN        10.73
IVF-Binary-1024-nl158-np17-rf0-pca (query)            55_031.57     1_775.06    56_806.62       0.0302             NaN        10.73
IVF-Binary-1024-nl158-np7-rf10-pca (query)            55_031.57     1_861.51    56_893.08       0.2594          1.0019        10.73
IVF-Binary-1024-nl158-np7-rf20-pca (query)            55_031.57     1_990.93    57_022.50       0.4046          1.0011        10.73
IVF-Binary-1024-nl158-np12-rf10-pca (query)           55_031.57     1_882.26    56_913.82       0.1969          1.0025        10.73
IVF-Binary-1024-nl158-np12-rf20-pca (query)           55_031.57     2_020.28    57_051.85       0.3104          1.0015        10.73
IVF-Binary-1024-nl158-np17-rf10-pca (query)           55_031.57     1_920.43    56_951.99       0.1622          1.0030        10.73
IVF-Binary-1024-nl158-np17-rf20-pca (query)           55_031.57     2_055.50    57_087.07       0.2576          1.0019        10.73
IVF-Binary-1024-nl158-pca (self)                      55_031.57     6_327.23    61_358.80       0.2041          1.3780        10.73
IVF-Binary-1024-nl223-np11-rf0-pca (query)            49_729.38     1_764.64    51_494.02       0.0495             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-pca (query)            49_729.38     1_795.06    51_524.44       0.0431             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-pca (query)            49_729.38     1_776.78    51_506.16       0.0349             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-pca (query)           49_729.38     1_891.62    51_620.99       0.2653          1.0018        10.98
IVF-Binary-1024-nl223-np11-rf20-pca (query)           49_729.38     2_014.64    51_744.02       0.4138          1.0011        10.98
IVF-Binary-1024-nl223-np14-rf10-pca (query)           49_729.38     1_894.83    51_624.20       0.2330          1.0021        10.98
IVF-Binary-1024-nl223-np14-rf20-pca (query)           49_729.38     2_178.50    51_907.88       0.3658          1.0012        10.98
IVF-Binary-1024-nl223-np21-rf10-pca (query)           49_729.38     1_906.61    51_635.99       0.1889          1.0026        10.98
IVF-Binary-1024-nl223-np21-rf20-pca (query)           49_729.38     2_063.16    51_792.53       0.2988          1.0016        10.98
IVF-Binary-1024-nl223-pca (self)                      49_729.38     6_256.73    55_986.10       0.2289          1.3126        10.98
IVF-Binary-1024-nl316-np15-rf0-pca (query)            50_233.00     1_795.31    52_028.31       0.0506             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-pca (query)            50_233.00     1_827.13    52_060.13       0.0472             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-pca (query)            50_233.00     1_812.09    52_045.09       0.0389             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-pca (query)           50_233.00     1_921.21    52_154.21       0.2701          1.0018        11.34
IVF-Binary-1024-nl316-np15-rf20-pca (query)           50_233.00     2_072.57    52_305.57       0.4216          1.0010        11.34
IVF-Binary-1024-nl316-np17-rf10-pca (query)           50_233.00     1_923.13    52_156.12       0.2532          1.0019        11.34
IVF-Binary-1024-nl316-np17-rf20-pca (query)           50_233.00     2_071.96    52_304.95       0.3972          1.0011        11.34
IVF-Binary-1024-nl316-np25-rf10-pca (query)           50_233.00     1_945.27    52_178.26       0.2098          1.0023        11.34
IVF-Binary-1024-nl316-np25-rf20-pca (query)           50_233.00     2_093.62    52_326.61       0.3314          1.0014        11.34
IVF-Binary-1024-nl316-pca (self)                      50_233.00     6_346.13    56_579.12       0.2425          1.2865        11.34
IVF-Binary-1024-nl158-np7-rf0-signed (query)          53_626.07     1_711.98    55_338.05       0.0551             NaN        10.72
IVF-Binary-1024-nl158-np12-rf0-signed (query)         53_626.07     1_732.69    55_358.76       0.0511             NaN        10.72
IVF-Binary-1024-nl158-np17-rf0-signed (query)         53_626.07     1_745.56    55_371.64       0.0493             NaN        10.72
IVF-Binary-1024-nl158-np7-rf10-signed (query)         53_626.07     1_849.07    55_475.14       0.3140          1.0018        10.72
IVF-Binary-1024-nl158-np7-rf20-signed (query)         53_626.07     1_951.12    55_577.19       0.4899          1.0009        10.72
IVF-Binary-1024-nl158-np12-rf10-signed (query)        53_626.07     1_856.40    55_482.48       0.2962          1.0019        10.72
IVF-Binary-1024-nl158-np12-rf20-signed (query)        53_626.07     1_980.43    55_606.51       0.4653          1.0010        10.72
IVF-Binary-1024-nl158-np17-rf10-signed (query)        53_626.07     1_873.89    55_499.96       0.2875          1.0020        10.72
IVF-Binary-1024-nl158-np17-rf20-signed (query)        53_626.07     2_025.66    55_651.74       0.4530          1.0010        10.72
IVF-Binary-1024-nl158-signed (self)                   53_626.07     6_046.61    59_672.69       0.9463          1.0044        10.72
IVF-Binary-1024-nl223-np11-rf0-signed (query)         48_563.33     1_751.27    50_314.60       0.0568             NaN        10.98
IVF-Binary-1024-nl223-np14-rf0-signed (query)         48_563.33     1_767.45    50_330.78       0.0531             NaN        10.98
IVF-Binary-1024-nl223-np21-rf0-signed (query)         48_563.33     1_834.13    50_397.46       0.0506             NaN        10.98
IVF-Binary-1024-nl223-np11-rf10-signed (query)        48_563.33     1_873.16    50_436.48       0.3220          1.0017        10.98
IVF-Binary-1024-nl223-np11-rf20-signed (query)        48_563.33     2_035.68    50_599.00       0.5002          1.0009        10.98
IVF-Binary-1024-nl223-np14-rf10-signed (query)        48_563.33     1_880.51    50_443.83       0.3072          1.0018        10.98
IVF-Binary-1024-nl223-np14-rf20-signed (query)        48_563.33     2_016.20    50_579.52       0.4808          1.0009        10.98
IVF-Binary-1024-nl223-np21-rf10-signed (query)        48_563.33     1_904.71    50_468.04       0.2944          1.0019        10.98
IVF-Binary-1024-nl223-np21-rf20-signed (query)        48_563.33     2_039.83    50_603.16       0.4632          1.0010        10.98
IVF-Binary-1024-nl223-signed (self)                   48_563.33     6_145.87    54_709.20       0.9472          1.0043        10.98
IVF-Binary-1024-nl316-np15-rf0-signed (query)         48_932.56     1_759.40    50_691.96       0.0561             NaN        11.34
IVF-Binary-1024-nl316-np17-rf0-signed (query)         48_932.56     1_770.73    50_703.29       0.0545             NaN        11.34
IVF-Binary-1024-nl316-np25-rf0-signed (query)         48_932.56     1_779.13    50_711.69       0.0516             NaN        11.34
IVF-Binary-1024-nl316-np15-rf10-signed (query)        48_932.56     1_894.17    50_826.73       0.3220          1.0017        11.34
IVF-Binary-1024-nl316-np15-rf20-signed (query)        48_932.56     2_008.12    50_940.68       0.5018          1.0009        11.34
IVF-Binary-1024-nl316-np17-rf10-signed (query)        48_932.56     1_904.79    50_837.35       0.3150          1.0018        11.34
IVF-Binary-1024-nl316-np17-rf20-signed (query)        48_932.56     2_022.59    50_955.16       0.4923          1.0009        11.34
IVF-Binary-1024-nl316-np25-rf10-signed (query)        48_932.56     1_903.85    50_836.41       0.3007          1.0019        11.34
IVF-Binary-1024-nl316-np25-rf20-signed (query)        48_932.56     2_031.49    50_964.06       0.4728          1.0010        11.34
IVF-Binary-1024-nl316-signed (self)                   48_932.56     6_178.62    55_111.18       0.9474          1.0043        11.34
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
Exhaustive (query)                                        10.22     4_142.80     4_153.02       1.0000          1.0000        48.83
Exhaustive (self)                                         10.22    14_008.67    14_018.89       1.0000          1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                           1_463.65       788.22     2_251.87       0.6106             NaN         2.84
ExhaustiveRaBitQ-rf5 (query)                           1_463.65       849.52     2_313.16       0.9692          1.0007         2.84
ExhaustiveRaBitQ-rf10 (query)                          1_463.65       899.01     2_362.65       0.9967          1.0001         2.84
ExhaustiveRaBitQ-rf20 (query)                          1_463.65       996.15     2_459.80       0.9999          1.0000         2.84
ExhaustiveRaBitQ (self)                                1_463.65     3_000.94     4_464.58       0.9967          1.0001         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       2_117.35       251.91     2_369.26       0.6023             NaN         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      2_117.35       398.76     2_516.11       0.6110             NaN         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      2_117.35       541.04     2_658.39       0.6120             NaN         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      2_117.35       344.16     2_461.51       0.9495          1.0025         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      2_117.35       415.87     2_533.22       0.9517          1.0025         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     2_117.35       500.09     2_617.44       0.9905          1.0003         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     2_117.35       580.85     2_698.20       0.9935          1.0003         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     2_117.35       652.38     2_769.73       0.9965          1.0001         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     2_117.35       751.50     2_868.85       0.9997          1.0000         2.89
IVF-RaBitQ-nl158 (self)                                2_117.35     2_462.62     4_579.97       0.9997          1.0000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_172.28       348.57     1_520.85       0.6111             NaN         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_172.28       405.60     1_577.88       0.6128             NaN         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_172.28       600.53     1_772.82       0.6133             NaN         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_172.28       421.63     1_593.91       0.9830          1.0007         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_172.28       497.65     1_669.93       0.9856          1.0007         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_172.28       508.75     1_681.03       0.9938          1.0002         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_172.28       587.92     1_760.20       0.9967          1.0001         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_172.28       702.31     1_874.59       0.9968          1.0001         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_172.28       789.19     1_961.47       0.9999          1.0000         2.95
IVF-RaBitQ-nl223 (self)                                1_172.28     2_619.26     3_791.54       0.9999          1.0000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_315.37       401.62     1_716.98       0.6130             NaN         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_315.37       452.34     1_767.71       0.6141             NaN         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_315.37       646.62     1_961.98       0.6147             NaN         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_315.37       497.35     1_812.72       0.9826          1.0007         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_315.37       572.40     1_887.77       0.9850          1.0006         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_315.37       544.90     1_860.27       0.9903          1.0003         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_315.37       621.69     1_937.05       0.9930          1.0003         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_315.37       744.41     2_059.78       0.9970          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_315.37       829.36     2_144.73       0.9999          1.0000         3.04
IVF-RaBitQ-nl316 (self)                                1_315.37     2_752.87     4_068.24       0.9999          1.0000         3.04
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
Exhaustive (query)                                        20.01     9_733.13     9_753.14       1.0000          1.0000        97.66
Exhaustive (self)                                         20.01    32_401.49    32_421.50       1.0000          1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           4_046.81     2_345.83     6_392.64       0.6297             NaN         5.23
ExhaustiveRaBitQ-rf5 (query)                           4_046.81     2_385.75     6_432.56       0.9763          1.0003         5.23
ExhaustiveRaBitQ-rf10 (query)                          4_046.81     2_464.30     6_511.11       0.9979          1.0000         5.23
ExhaustiveRaBitQ-rf20 (query)                          4_046.81     2_578.08     6_624.89       0.9999          1.0000         5.23
ExhaustiveRaBitQ (self)                                4_046.81     8_255.55    12_302.35       0.9979          1.0000         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_436.81       752.48     6_189.28       0.6167             NaN         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_436.81     1_266.40     6_703.21       0.6289             NaN         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_436.81     1_698.78     7_135.59       0.6304             NaN         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_436.81       867.07     6_303.88       0.9462          1.0021         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_436.81       963.70     6_400.51       0.9476          1.0021         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_436.81     1_342.17     6_778.97       0.9901          1.0003         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_436.81     1_453.62     6_890.43       0.9919          1.0003         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_436.81     1_832.29     7_269.10       0.9980          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_436.81     1_940.13     7_376.94       0.9999          1.0000         5.32
IVF-RaBitQ-nl158 (self)                                5_436.81     6_456.67    11_893.47       0.9999          1.0000         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_319.39     1_035.08     4_354.48       0.6279             NaN         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_319.39     1_299.36     4_618.75       0.6306             NaN         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_319.39     1_917.92     5_237.31       0.6308             NaN         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_319.39     1_154.07     4_473.46       0.9852          1.0005         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_319.39     1_259.29     4_578.68       0.9869          1.0004         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_319.39     1_416.73     4_736.13       0.9970          1.0001         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_319.39     1_528.52     4_847.92       0.9989          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_319.39     2_053.54     5_372.93       0.9979          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_319.39     2_142.19     5_461.59       0.9999          1.0000         5.44
IVF-RaBitQ-nl223 (self)                                3_319.39     7_149.71    10_469.10       0.9999          1.0000         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      3_651.49     1_354.11     5_005.60       0.6304             NaN         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      3_651.49     1_529.98     5_181.46       0.6314             NaN         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      3_651.49     2_237.94     5_889.42       0.6316             NaN         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     3_651.49     1_484.41     5_135.90       0.9911          1.0003         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     3_651.49     1_602.09     5_253.58       0.9926          1.0002         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     3_651.49     1_664.55     5_316.04       0.9966          1.0001         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     3_651.49     1_765.78     5_417.27       0.9982          1.0001         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     3_651.49     2_394.08     6_045.57       0.9982          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     3_651.49     2_519.82     6_171.31       0.9999          1.0000         5.63
IVF-RaBitQ-nl316 (self)                                3_651.49     8_206.42    11_857.90       0.9999          1.0000         5.63
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
Exhaustive (query)                                        40.98    22_105.06    22_146.04       1.0000          1.0000       195.31
Exhaustive (self)                                         40.98    73_418.98    73_459.96       1.0000          1.0000       195.31
ExhaustiveRaBitQ-rf0 (query)                          14_282.31     9_526.45    23_808.76       0.0387             NaN        11.50
ExhaustiveRaBitQ-rf5 (query)                          14_282.31     9_445.61    23_727.93       0.1120          1.1672        11.50
ExhaustiveRaBitQ-rf10 (query)                         14_282.31     9_474.99    23_757.31       0.1905          1.1377        11.50
ExhaustiveRaBitQ-rf20 (query)                         14_282.31     9_629.88    23_912.20       0.3240          1.1093        11.50
ExhaustiveRaBitQ (self)                               14_282.31    31_599.06    45_881.38       0.1897          1.1378        11.50
IVF-RaBitQ-nl158-np7-rf0 (query)                      17_845.29     3_028.53    20_873.83       0.0382             NaN        11.68
IVF-RaBitQ-nl158-np12-rf0 (query)                     17_845.29     5_063.43    22_908.73       0.0380             NaN        11.68
IVF-RaBitQ-nl158-np17-rf0 (query)                     17_845.29     7_129.90    24_975.19       0.0378             NaN        11.68
IVF-RaBitQ-nl158-np7-rf10 (query)                     17_845.29     3_105.22    20_950.51       0.1949          1.1347        11.68
IVF-RaBitQ-nl158-np7-rf20 (query)                     17_845.29     3_232.25    21_077.55       0.3334          1.1053        11.68
IVF-RaBitQ-nl158-np12-rf10 (query)                    17_845.29     5_103.83    22_949.12       0.1894          1.1370        11.68
IVF-RaBitQ-nl158-np12-rf20 (query)                    17_845.29     5_256.41    23_101.70       0.3252          1.1080        11.68
IVF-RaBitQ-nl158-np17-rf10 (query)                    17_845.29     7_130.51    24_975.80       0.1872          1.1378        11.68
IVF-RaBitQ-nl158-np17-rf20 (query)                    17_845.29     7_258.46    25_103.75       0.3222          1.1088        11.68
IVF-RaBitQ-nl158 (self)                               17_845.29    24_260.61    42_105.90       0.3198          1.1090        11.68
IVF-RaBitQ-nl223-np11-rf0 (query)                     12_582.02     4_460.14    17_042.17       0.0372             NaN        11.93
IVF-RaBitQ-nl223-np14-rf0 (query)                     12_582.02     5_656.77    18_238.79       0.0372             NaN        11.93
IVF-RaBitQ-nl223-np21-rf0 (query)                     12_582.02     8_399.79    20_981.82       0.0371             NaN        11.93
IVF-RaBitQ-nl223-np11-rf10 (query)                    12_582.02     4_501.35    17_083.37       0.1864          1.1366        11.93
IVF-RaBitQ-nl223-np11-rf20 (query)                    12_582.02     4_645.59    17_227.61       0.3251          1.1068        11.93
IVF-RaBitQ-nl223-np14-rf10 (query)                    12_582.02     5_674.96    18_256.99       0.1836          1.1377        11.93
IVF-RaBitQ-nl223-np14-rf20 (query)                    12_582.02     5_810.28    18_392.30       0.3207          1.1080        11.93
IVF-RaBitQ-nl223-np21-rf10 (query)                    12_582.02     8_340.95    20_922.97       0.1831          1.1380        11.93
IVF-RaBitQ-nl223-np21-rf20 (query)                    12_582.02     8_493.16    21_075.18       0.3198          1.1081        11.93
IVF-RaBitQ-nl223 (self)                               12_582.02    28_257.57    40_839.59       0.3170          1.1084        11.93
IVF-RaBitQ-nl316-np15-rf0 (query)                     13_145.05     5_973.08    19_118.12       0.0360             NaN        12.30
IVF-RaBitQ-nl316-np17-rf0 (query)                     13_145.05     6_721.54    19_866.59       0.0359             NaN        12.30
IVF-RaBitQ-nl316-np25-rf0 (query)                     13_145.05     9_826.46    22_971.50       0.0357             NaN        12.30
IVF-RaBitQ-nl316-np15-rf10 (query)                    13_145.05     5_942.54    19_087.59       0.1813          1.1373        12.30
IVF-RaBitQ-nl316-np15-rf20 (query)                    13_145.05     6_093.81    19_238.85       0.3201          1.1069        12.30
IVF-RaBitQ-nl316-np17-rf10 (query)                    13_145.05     6_712.47    19_857.52       0.1800          1.1378        12.30
IVF-RaBitQ-nl316-np17-rf20 (query)                    13_145.05     6_823.59    19_968.64       0.3176          1.1075        12.30
IVF-RaBitQ-nl316-np25-rf10 (query)                    13_145.05     9_717.07    22_862.11       0.1789          1.1382        12.30
IVF-RaBitQ-nl316-np25-rf20 (query)                    13_145.05     9_867.35    23_012.40       0.3161          1.1078        12.30
IVF-RaBitQ-nl316 (self)                               13_145.05    32_775.40    45_920.45       0.3142          1.1079        12.30
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
Exhaustive (query)                                         9.68     4_190.18     4_199.86       1.0000          1.0000        48.83
Exhaustive (self)                                          9.68    14_072.09    14_081.77       1.0000          1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                           1_436.59       786.88     2_223.47       0.7462             NaN         2.84
ExhaustiveRaBitQ-rf5 (query)                           1_436.59       856.86     2_293.45       0.9986          1.0000         2.84
ExhaustiveRaBitQ-rf10 (query)                          1_436.59       920.29     2_356.88       1.0000          1.0000         2.84
ExhaustiveRaBitQ-rf20 (query)                          1_436.59       987.03     2_423.61       1.0000          1.0000         2.84
ExhaustiveRaBitQ (self)                                1_436.59     2_998.17     4_434.76       1.0000          1.0000         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       2_101.01       235.42     2_336.43       0.7478             NaN         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      2_101.01       387.88     2_488.89       0.7496             NaN         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      2_101.01       534.77     2_635.78       0.7496             NaN         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      2_101.01       326.35     2_427.36       0.9930          1.0006         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      2_101.01       401.73     2_502.74       0.9930          1.0006         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     2_101.01       479.69     2_580.70       1.0000          1.0000         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     2_101.01       562.66     2_663.68       1.0000          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     2_101.01       666.56     2_767.57       1.0000          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     2_101.01       743.02     2_844.04       1.0000          1.0000         2.89
IVF-RaBitQ-nl158 (self)                                2_101.01     2_413.93     4_514.95       1.0000          1.0000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_057.99       320.93     1_378.91       0.7499             NaN         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_057.99       400.45     1_458.44       0.7507             NaN         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_057.99       603.03     1_661.01       0.7507             NaN         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_057.99       412.81     1_470.79       0.9969          1.0002         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_057.99       484.41     1_542.40       0.9969          1.0002         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_057.99       496.13     1_554.11       0.9999          1.0000         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_057.99       579.47     1_637.45       0.9999          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_057.99       693.58     1_751.57       1.0000          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_057.99       776.51     1_834.50       1.0000          1.0000         2.95
IVF-RaBitQ-nl223 (self)                                1_057.99     2_595.08     3_653.06       1.0000          1.0000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_268.02       400.18     1_668.21       0.7539             NaN         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_268.02       448.11     1_716.13       0.7542             NaN         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_268.02       645.88     1_913.90       0.7542             NaN         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_268.02       491.54     1_759.57       0.9985          1.0001         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_268.02       568.52     1_836.54       0.9985          1.0001         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_268.02       541.22     1_809.25       0.9997          1.0000         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_268.02       618.97     1_887.00       0.9997          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_268.02       743.74     2_011.76       1.0000          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_268.02       829.57     2_097.59       1.0000          1.0000         3.04
IVF-RaBitQ-nl316 (self)                                1_268.02     2_756.49     4_024.51       1.0000          1.0000         3.04
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
Exhaustive (query)                                        20.65     9_597.71     9_618.36       1.0000          1.0000        97.66
Exhaustive (self)                                         20.65    33_176.23    33_196.87       1.0000          1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           4_288.27     2_355.34     6_643.60       0.7550             NaN         5.23
ExhaustiveRaBitQ-rf5 (query)                           4_288.27     2_441.10     6_729.36       0.9989          1.0000         5.23
ExhaustiveRaBitQ-rf10 (query)                          4_288.27     2_502.47     6_790.74       0.9998          1.0000         5.23
ExhaustiveRaBitQ-rf20 (query)                          4_288.27     2_606.53     6_894.80       0.9998          1.0000         5.23
ExhaustiveRaBitQ (self)                                4_288.27     8_387.90    12_676.16       0.9998          1.0000         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_469.52       738.03     6_207.55       0.7525             NaN         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_469.52     1_205.40     6_674.92       0.7571             NaN         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_469.52     1_682.89     7_152.42       0.7571             NaN         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_469.52       839.05     6_308.57       0.9861          1.0009         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_469.52       931.84     6_401.36       0.9861          1.0009         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_469.52     1_320.63     6_790.15       0.9998          1.0000         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_469.52     1_426.99     6_896.52       0.9998          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_469.52     1_803.55     7_273.07       0.9998          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_469.52     1_930.23     7_399.75       0.9998          1.0000         5.32
IVF-RaBitQ-nl158 (self)                                5_469.52     6_408.48    11_878.00       0.9998          1.0000         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_166.28     1_047.48     4_213.76       0.7545             NaN         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_166.28     1_327.96     4_494.24       0.7558             NaN         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_166.28     1_963.25     5_129.53       0.7560             NaN         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_166.28     1_168.01     4_334.29       0.9956          1.0003         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_166.28     1_276.88     4_443.17       0.9956          1.0003         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_166.28     1_441.76     4_608.04       0.9992          1.0000         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_166.28     1_550.86     4_717.14       0.9992          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_166.28     2_081.92     5_248.20       0.9999          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_166.28     2_187.67     5_353.95       0.9999          1.0000         5.44
IVF-RaBitQ-nl223 (self)                                3_166.28     7_345.02    10_511.30       0.9998          1.0000         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      3_465.04     1_351.07     4_816.11       0.7579             NaN         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      3_465.04     1_527.93     4_992.97       0.7586             NaN         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      3_465.04     2_223.30     5_688.34       0.7588             NaN         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     3_465.04     1_465.20     4_930.24       0.9969          1.0002         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     3_465.04     1_571.28     5_036.33       0.9969          1.0002         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     3_465.04     1_638.43     5_103.47       0.9993          1.0000         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     3_465.04     1_743.96     5_209.00       0.9993          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     3_465.04     2_336.84     5_801.88       0.9999          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     3_465.04     2_445.37     5_910.41       0.9999          1.0000         5.63
IVF-RaBitQ-nl316 (self)                                3_465.04     8_154.73    11_619.77       0.9998          1.0000         5.63
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
Exhaustive (query)                                        40.77    21_998.67    22_039.43       1.0000          1.0000       195.31
Exhaustive (self)                                         40.77    73_245.57    73_286.34       1.0000          1.0000       195.31
ExhaustiveRaBitQ-rf0 (query)                          14_515.92     9_103.51    23_619.42       0.0314             NaN        11.50
ExhaustiveRaBitQ-rf5 (query)                          14_515.92     9_117.85    23_633.76       0.0996          1.2429        11.50
ExhaustiveRaBitQ-rf10 (query)                         14_515.92     9_180.23    23_696.15       0.1751          1.1900        11.50
ExhaustiveRaBitQ-rf20 (query)                         14_515.92     9_569.56    24_085.48       0.3082          1.1415        11.50
ExhaustiveRaBitQ (self)                               14_515.92    32_272.56    46_788.47       0.1750          1.1903        11.50
IVF-RaBitQ-nl158-np7-rf0 (query)                      17_727.72     2_912.96    20_640.68       0.0306             NaN        11.68
IVF-RaBitQ-nl158-np12-rf0 (query)                     17_727.72     4_943.32    22_671.04       0.0304             NaN        11.68
IVF-RaBitQ-nl158-np17-rf0 (query)                     17_727.72     6_965.08    24_692.80       0.0304             NaN        11.68
IVF-RaBitQ-nl158-np7-rf10 (query)                     17_727.72     2_992.53    20_720.25       0.1738          1.1891        11.68
IVF-RaBitQ-nl158-np7-rf20 (query)                     17_727.72     3_135.24    20_862.96       0.3121          1.1391        11.68
IVF-RaBitQ-nl158-np12-rf10 (query)                    17_727.72     4_973.17    22_700.89       0.1705          1.1912        11.68
IVF-RaBitQ-nl158-np12-rf20 (query)                    17_727.72     5_145.88    22_873.60       0.3060          1.1411        11.68
IVF-RaBitQ-nl158-np17-rf10 (query)                    17_727.72     6_948.47    24_676.19       0.1705          1.1912        11.68
IVF-RaBitQ-nl158-np17-rf20 (query)                    17_727.72     7_079.12    24_806.84       0.3060          1.1411        11.68
IVF-RaBitQ-nl158 (self)                               17_727.72    23_575.12    41_302.84       0.3062          1.1412        11.68
IVF-RaBitQ-nl223-np11-rf0 (query)                     12_559.62     4_361.62    16_921.24       0.0300             NaN        11.93
IVF-RaBitQ-nl223-np14-rf0 (query)                     12_559.62     5_595.75    18_155.38       0.0298             NaN        11.93
IVF-RaBitQ-nl223-np21-rf0 (query)                     12_559.62     8_316.05    20_875.68       0.0298             NaN        11.93
IVF-RaBitQ-nl223-np11-rf10 (query)                    12_559.62     4_438.56    16_998.18       0.1703          1.1895        11.93
IVF-RaBitQ-nl223-np11-rf20 (query)                    12_559.62     4_574.55    17_134.18       0.3092          1.1387        11.93
IVF-RaBitQ-nl223-np14-rf10 (query)                    12_559.62     5_572.51    18_132.13       0.1682          1.1907        11.93
IVF-RaBitQ-nl223-np14-rf20 (query)                    12_559.62     5_731.24    18_290.86       0.3052          1.1400        11.93
IVF-RaBitQ-nl223-np21-rf10 (query)                    12_559.62     8_335.28    20_894.90       0.1681          1.1908        11.93
IVF-RaBitQ-nl223-np21-rf20 (query)                    12_559.62     8_394.54    20_954.16       0.3049          1.1401        11.93
IVF-RaBitQ-nl223 (self)                               12_559.62    27_978.42    40_538.04       0.3044          1.1403        11.93
IVF-RaBitQ-nl316-np15-rf0 (query)                     13_334.72     5_854.70    19_189.42       0.0281             NaN        12.30
IVF-RaBitQ-nl316-np17-rf0 (query)                     13_334.72     6_621.54    19_956.26       0.0280             NaN        12.30
IVF-RaBitQ-nl316-np25-rf0 (query)                     13_334.72     9_866.97    23_201.69       0.0279             NaN        12.30
IVF-RaBitQ-nl316-np15-rf10 (query)                    13_334.72     6_020.12    19_354.84       0.1656          1.1901        12.30
IVF-RaBitQ-nl316-np15-rf20 (query)                    13_334.72     6_060.77    19_395.49       0.3074          1.1378        12.30
IVF-RaBitQ-nl316-np17-rf10 (query)                    13_334.72     6_696.58    20_031.30       0.1642          1.1908        12.30
IVF-RaBitQ-nl316-np17-rf20 (query)                    13_334.72     6_956.18    20_290.90       0.3048          1.1386        12.30
IVF-RaBitQ-nl316-np25-rf10 (query)                    13_334.72     9_690.61    23_025.32       0.1634          1.1912        12.30
IVF-RaBitQ-nl316-np25-rf20 (query)                    13_334.72     9_845.43    23_180.15       0.3034          1.1390        12.30
IVF-RaBitQ-nl316 (self)                               13_334.72    32_787.48    46_122.20       0.3025          1.1393        12.30
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
Exhaustive (query)                                         6.69     4_160.93     4_167.62       1.0000          1.0000        48.83
Exhaustive (self)                                          6.69    14_052.56    14_059.25       1.0000          1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                           1_392.88     1_072.31     2_465.19       0.3521             NaN         2.84
ExhaustiveRaBitQ-rf5 (query)                           1_392.88     1_148.56     2_541.44       0.7628          1.0010         2.84
ExhaustiveRaBitQ-rf10 (query)                          1_392.88     1_201.32     2_594.19       0.8947          1.0004         2.84
ExhaustiveRaBitQ-rf20 (query)                          1_392.88     1_302.86     2_695.73       0.9663          1.0001         2.84
ExhaustiveRaBitQ (self)                                1_392.88     4_013.44     5_406.32       1.0000          1.0000         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       2_034.81       312.87     2_347.68       0.3694             NaN         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      2_034.81       530.90     2_565.71       0.3690             NaN         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      2_034.81       757.71     2_792.52       0.3688             NaN         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      2_034.81       400.65     2_435.46       0.9102          1.0003         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      2_034.81       469.48     2_504.29       0.9725          1.0001         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     2_034.81       638.25     2_673.06       0.9119          1.0003         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     2_034.81       729.51     2_764.32       0.9748          1.0001         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     2_034.81       884.75     2_919.56       0.9121          1.0003         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     2_034.81       967.73     3_002.55       0.9752          1.0001         2.89
IVF-RaBitQ-nl158 (self)                                2_034.81     3_206.73     5_241.54       1.0000          1.0000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_033.97       323.10     1_357.07       0.4046             NaN         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_033.97       405.32     1_439.29       0.4044             NaN         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_033.97       600.68     1_634.65       0.4041             NaN         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_033.97       409.20     1_443.17       0.9380          1.0002         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_033.97       470.76     1_504.73       0.9854          1.0000         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_033.97       497.49     1_531.46       0.9381          1.0002         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_033.97       561.31     1_595.28       0.9859          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_033.97       698.80     1_732.77       0.9382          1.0002         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_033.97       773.85     1_807.82       0.9861          1.0000         2.95
IVF-RaBitQ-nl223 (self)                                1_033.97     2_544.44     3_578.41       1.0000          1.0000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_201.42       399.48     1_600.90       0.4183             NaN         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_201.42       449.89     1_651.31       0.4181             NaN         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_201.42       652.10     1_853.52       0.4179             NaN         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_201.42       485.57     1_686.99       0.9454          1.0001         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_201.42       546.65     1_748.07       0.9885          1.0000         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_201.42       537.24     1_738.66       0.9454          1.0001         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_201.42       604.47     1_805.89       0.9886          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_201.42       736.35     1_937.76       0.9453          1.0001         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_201.42       814.26     2_015.68       0.9888          1.0000         3.04
IVF-RaBitQ-nl316 (self)                                1_201.42     2_673.16     3_874.58       1.0000          1.0000         3.04
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
Exhaustive (query)                                        19.13     9_684.50     9_703.63       1.0000          1.0000        97.66
Exhaustive (self)                                         19.13    32_375.08    32_394.20       1.0000          1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           3_862.11     2_918.53     6_780.65       0.3390             NaN         5.23
ExhaustiveRaBitQ-rf5 (query)                           3_862.11     3_035.55     6_897.66       0.7391          1.0005         5.23
ExhaustiveRaBitQ-rf10 (query)                          3_862.11     3_132.55     6_994.67       0.8736          1.0002         5.23
ExhaustiveRaBitQ-rf20 (query)                          3_862.11     3_234.02     7_096.14       0.9511          1.0001         5.23
ExhaustiveRaBitQ (self)                                3_862.11    10_298.45    14_160.56       0.9997          1.0001         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_303.35       764.50     6_067.84       0.3677             NaN         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_303.35     1_301.44     6_604.78       0.3663             NaN         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_303.35     1_838.24     7_141.59       0.3657             NaN         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_303.35       874.94     6_178.29       0.8982          1.0002         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_303.35       976.00     6_279.35       0.9620          1.0001         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_303.35     1_421.53     6_724.87       0.8997          1.0001         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_303.35     1_512.80     6_816.14       0.9654          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_303.35     2_001.66     7_305.00       0.8993          1.0001         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_303.35     2_072.92     7_376.27       0.9655          1.0000         5.32
IVF-RaBitQ-nl158 (self)                                5_303.35     6_903.85    12_207.20       0.9999          1.0000         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_059.37     1_065.01     4_124.38       0.3882             NaN         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_059.37     1_301.99     4_361.36       0.3876             NaN         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_059.37     1_933.15     4_992.53       0.3866             NaN         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_059.37     1_141.15     4_200.52       0.9178          1.0001         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_059.37     1_229.24     4_288.61       0.9738          1.0000         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_059.37     1_409.79     4_469.16       0.9185          1.0001         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_059.37     1_502.50     4_561.87       0.9754          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_059.37     2_042.64     5_102.01       0.9182          1.0001         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_059.37     2_152.37     5_211.74       0.9763          1.0000         5.44
IVF-RaBitQ-nl223 (self)                                3_059.37     7_079.58    10_138.95       0.9999          1.0000         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      3_306.28     1_333.83     4_640.12       0.4033             NaN         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      3_306.28     1_516.10     4_822.38       0.4029             NaN         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      3_306.28     2_184.27     5_490.55       0.4019             NaN         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     3_306.28     1_445.87     4_752.15       0.9274          1.0001         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     3_306.28     1_542.15     4_848.43       0.9774          1.0000         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     3_306.28     1_624.78     4_931.06       0.9277          1.0001         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     3_306.28     1_709.56     5_015.85       0.9783          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     3_306.28     2_302.83     5_609.11       0.9278          1.0001         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     3_306.28     2_393.71     5_699.99       0.9796          1.0000         5.63
IVF-RaBitQ-nl316 (self)                                3_306.28     7_959.64    11_265.92       0.9999          1.0000         5.63
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
Exhaustive (query)                                        39.24    22_151.70    22_190.94       1.0000          1.0000       195.31
Exhaustive (self)                                         39.24    73_255.15    73_294.39       1.0000          1.0000       195.31
ExhaustiveRaBitQ-rf0 (query)                          14_006.32    10_167.29    24_173.61       0.0250             NaN        11.50
ExhaustiveRaBitQ-rf5 (query)                          14_006.32    10_067.45    24_073.77       0.0597          1.0053        11.50
ExhaustiveRaBitQ-rf10 (query)                         14_006.32    10_115.00    24_121.32       0.0979          1.0041        11.50
ExhaustiveRaBitQ-rf20 (query)                         14_006.32    10_291.63    24_297.95       0.1724          1.0031        11.50
ExhaustiveRaBitQ (self)                               14_006.32    34_027.07    48_033.39       0.2491          1.4917        11.50
IVF-RaBitQ-nl158-np7-rf0 (query)                      17_166.72     2_890.60    20_057.32       0.0268             NaN        11.68
IVF-RaBitQ-nl158-np12-rf0 (query)                     17_166.72     4_936.75    22_103.47       0.0264             NaN        11.68
IVF-RaBitQ-nl158-np17-rf0 (query)                     17_166.72     7_017.55    24_184.28       0.0263             NaN        11.68
IVF-RaBitQ-nl158-np7-rf10 (query)                     17_166.72     3_017.75    20_184.48       0.1108          1.0030        11.68
IVF-RaBitQ-nl158-np7-rf20 (query)                     17_166.72     3_160.38    20_327.10       0.2020          1.0021        11.68
IVF-RaBitQ-nl158-np12-rf10 (query)                    17_166.72     5_051.02    22_217.74       0.1055          1.0033        11.68
IVF-RaBitQ-nl158-np12-rf20 (query)                    17_166.72     5_162.29    22_329.01       0.1888          1.0024        11.68
IVF-RaBitQ-nl158-np17-rf10 (query)                    17_166.72     7_068.49    24_235.21       0.1037          1.0034        11.68
IVF-RaBitQ-nl158-np17-rf20 (query)                    17_166.72     7_192.16    24_358.89       0.1839          1.0025        11.68
IVF-RaBitQ-nl158 (self)                               17_166.72    24_168.59    41_335.32       0.5467          1.1840        11.68
IVF-RaBitQ-nl223-np11-rf0 (query)                     11_926.81     4_398.24    16_325.05       0.0284             NaN        11.93
IVF-RaBitQ-nl223-np14-rf0 (query)                     11_926.81     5_661.18    17_587.99       0.0282             NaN        11.93
IVF-RaBitQ-nl223-np21-rf0 (query)                     11_926.81     9_047.49    20_974.29       0.0279             NaN        11.93
IVF-RaBitQ-nl223-np11-rf10 (query)                    11_926.81     4_483.35    16_410.16       0.1159          1.0029        11.93
IVF-RaBitQ-nl223-np11-rf20 (query)                    11_926.81     4_579.93    16_506.74       0.2094          1.0020        11.93
IVF-RaBitQ-nl223-np14-rf10 (query)                    11_926.81     5_565.94    17_492.75       0.1130          1.0030        11.93
IVF-RaBitQ-nl223-np14-rf20 (query)                    11_926.81     5_713.66    17_640.46       0.2024          1.0021        11.93
IVF-RaBitQ-nl223-np21-rf10 (query)                    11_926.81     8_260.82    20_187.62       0.1095          1.0032        11.93
IVF-RaBitQ-nl223-np21-rf20 (query)                    11_926.81     8_402.22    20_329.03       0.1943          1.0023        11.93
IVF-RaBitQ-nl223 (self)                               11_926.81    27_966.40    39_893.21       0.6328          1.1262        11.93
IVF-RaBitQ-nl316-np15-rf0 (query)                     12_493.98     5_916.07    18_410.05       0.0298             NaN        12.30
IVF-RaBitQ-nl316-np17-rf0 (query)                     12_493.98     6_708.72    19_202.71       0.0297             NaN        12.30
IVF-RaBitQ-nl316-np25-rf0 (query)                     12_493.98     9_774.98    22_268.97       0.0293             NaN        12.30
IVF-RaBitQ-nl316-np15-rf10 (query)                    12_493.98     5_926.07    18_420.05       0.1228          1.0028        12.30
IVF-RaBitQ-nl316-np15-rf20 (query)                    12_493.98     6_037.13    18_531.11       0.2214          1.0019        12.30
IVF-RaBitQ-nl316-np17-rf10 (query)                    12_493.98     6_675.70    19_169.68       0.1212          1.0028        12.30
IVF-RaBitQ-nl316-np17-rf20 (query)                    12_493.98     6_812.97    19_306.96       0.2173          1.0020        12.30
IVF-RaBitQ-nl316-np25-rf10 (query)                    12_493.98     9_689.84    22_183.83       0.1169          1.0030        12.30
IVF-RaBitQ-nl316-np25-rf20 (query)                    12_493.98     9_816.53    22_310.51       0.2071          1.0021        12.30
IVF-RaBitQ-nl316 (self)                               12_493.98    32_719.96    45_213.94       0.7009          1.0897        12.30
-----------------------------------------------------------------------------------------------------------------------------------

</code></pre>
</details>

Overall, this is a fantastic binary index that massively compresses the data,
while still allowing for great Recalls. If you need to compress your data
and reduce memory fingerprint, please, use RaBitQ!

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
