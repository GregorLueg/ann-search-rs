## Binarised indices benchmarks and parameter

Binarised indices push the compression to (roughly) bits. Three consequences:

1. The index footprint collapses.
2. Queries usually get faster, because bitwise operations are cheap on modern
   CPUs.
3. Without re-ranking the top candidates, recall drops hard. Less so for RaBitQ,
   and for TurboQuant it depends on the data.

The benchmarks below show both, with and without re-ranking. For the simple
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

As with the other benchmarks: index build, query against a 10% subsample with
noise added, and full self-kNN generation, plus the in-memory index size. These
runs use `"correlated"`, `"lowrank"` and `"embedding"` at higher dimensionality
with fewer samples, since that is where binarisation belongs.

**On the distance-ratio column.** A binarised index reports an approximate
distance, not the distance. Every ratio here is recomputed in `f32` from the
original vectors against the neighbours the index returned, so it measures
retrieval quality alone and the re-ranked and non-re-ranked rows sit on the same
footing.

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

These indices can keep the original vectors in a `VecStore` on disk for
re-ranking. Recommended if you want the recall to stay usable. Their home ground
is very high-dimensional data where memory is the binding constraint.

**Tunable parameters *(general)*:**

- *n_bits*: How many bits to encode each vector into. More bits, better recall,
  bigger index.
- *binarisation_init*: Three options are provided in the crate. `"random"` that
  generates random planes that are subsequently orthogonalised, `"pca"` that
  leverages PCA to identify axis of maximum variation or `"signed"` that just
  uses the sign of the respective embedding dimensions (or residual for the
  IVF). In this case, `n_bits` is set automatically to `n_dim`. Signed only
  really makes sense if you have a lot of dimensions; otherwise, the performance
  is not great (at all).
- *reranking_factor*: Hamming distance picks the candidates, then the on-disk
  vectors are loaded and the candidates re-scored exactly. The factor is how
  many more than `k` get re-scored, so `10` means `10 * k` vectors. More
  candidates, better recall. Default `20`; the grid runs lower values to show
  what that costs.

**Tunable parameters *(IVF-specific)*:**

- *Number of lists (nl)*: Number of k-means clusters, `sqrt(n)` as a default.
- *Number of probes (np)*: Typically `sqrt(nlist)` or up to 5% of `nlist`.

Self queries run with `reranking_factor = 10`.

#### Correlated data

<details>
<summary><b>Correlated data - 256 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 256D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.52       708.05       740.57       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.52     2_305.19     2_337.71       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_706.67       306.98     3_013.64       0.1199          1.4616            1.4199         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_706.67       437.59     3_144.26       0.3412          1.0941            1.0814         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_706.67       523.26     3_229.93       0.4468          1.0571            1.0475         1.78
ExhaustiveBinary-256-random (self)                     2_706.67     1_382.53     4_089.20       0.3455          1.0895            1.0798         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_794.06       317.11     3_111.17       0.1726        320.7544          238.0319         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_794.06       447.07     3_241.13       0.4714          6.2695            1.0281         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_794.06       570.48     3_364.54       0.5869          3.3746            1.0156         1.78
ExhaustiveBinary-256-pca (self)                        2_794.06     1_402.73     4_196.79       0.4710          6.5199            1.0286         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_256.61       456.76     5_713.37       0.1589          1.3546            1.3300         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_256.61       567.24     5_823.85       0.3786          1.0692            1.0677         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_256.61       663.59     5_920.20       0.4875          1.0424            1.0395         3.55
ExhaustiveBinary-512-random (self)                     5_256.61     1_817.70     7_074.31       0.3805          1.0675            1.0675         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_256.77       457.06     5_713.83       0.2202          2.2990            1.1931         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_256.77       563.35     5_820.11       0.6784          1.1118            1.0137         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_256.77       672.25     5_929.02       0.8250          1.0439            1.0038         3.55
ExhaustiveBinary-512-pca (self)                        5_256.77     1_862.21     7_118.97       0.6791          1.1157            1.0136         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_406.12       792.04    11_198.16       0.1929          1.2763            1.2696         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_406.12       892.02    11_298.14       0.4214          1.0550            1.0552         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_406.12     1_004.66    11_410.79       0.5434          1.0327            1.0308         7.10
ExhaustiveBinary-1024-random (self)                   10_406.12     2_942.56    13_348.69       0.4233          1.0547            1.0552         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_440.95       847.31    11_288.26       0.2566          1.3583            1.1747         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_440.95     1_015.80    11_456.75       0.7234          1.0350            1.0105         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_440.95     1_111.61    11_552.56       0.8564          1.0155            1.0023         7.10
ExhaustiveBinary-1024-pca (self)                      10_440.95     3_055.66    13_496.61       0.7248          1.0333            1.0103         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   61.89       492.97       554.87       0.1211          1.4988            1.4523         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    61.89       527.01       588.90       0.3286          1.1039            1.0884         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    61.89       790.16       852.05       0.4385          1.0623            1.0494         1.53
ExhaustiveBinary-256-sign (self)                          61.89     1_646.85     1_708.74       0.3332          1.0988            1.0859         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)            3_557.35       121.58     3_678.93       0.1239          1.4352            1.3995         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           3_557.35       124.21     3_681.57       0.1239          1.4352            1.3995         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           3_557.35       134.26     3_691.62       0.1239          1.4352            1.3995         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           3_557.35       176.14     3_733.49       0.3497          1.0878            1.0780         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           3_557.35       226.38     3_783.73       0.4589          1.0524            1.0449         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          3_557.35       181.34     3_738.69       0.3497          1.0878            1.0780         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          3_557.35       238.79     3_796.14       0.4589          1.0524            1.0449         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          3_557.35       178.50     3_735.85       0.3497          1.0878            1.0780         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          3_557.35       230.65     3_788.00       0.4589          1.0524            1.0449         1.93
IVF-Binary-256-nl158-random (self)                     3_557.35       499.82     4_057.18       0.3544          1.0828            1.0764         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           3_180.51       132.10     3_312.61       0.1409          1.3640            1.3205         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           3_180.51       131.20     3_311.70       0.1409          1.3646            1.3210         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           3_180.51       135.13     3_315.63       0.1409          1.3646            1.3210         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          3_180.51       182.61     3_363.11       0.3881          1.0691            1.0628         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          3_180.51       233.46     3_413.96       0.4965          1.0424            1.0371         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          3_180.51       188.64     3_369.14       0.3880          1.0691            1.0628         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          3_180.51       230.66     3_411.16       0.4956          1.0427            1.0372         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          3_180.51       187.58     3_368.09       0.3880          1.0691            1.0628         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          3_180.51       232.77     3_413.28       0.4956          1.0428            1.0372         2.00
IVF-Binary-256-nl223-random (self)                     3_180.51       499.11     3_679.61       0.3928          1.0648            1.0620         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           3_311.70       134.72     3_446.42       0.1513          1.3330            1.2879         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           3_311.70       134.52     3_446.21       0.1513          1.3333            1.2880         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           3_311.70       134.31     3_446.01       0.1513          1.3333            1.2881         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          3_311.70       185.82     3_497.52       0.4040          1.0639            1.0580         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          3_311.70       234.10     3_545.80       0.5072          1.0411            1.0355         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          3_311.70       187.43     3_499.13       0.4038          1.0640            1.0581         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          3_311.70       276.22     3_587.92       0.5061          1.0414            1.0357         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          3_311.70       211.61     3_523.31       0.4038          1.0639            1.0581         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          3_311.70       276.54     3_588.24       0.5062          1.0413            1.0357         2.09
IVF-Binary-256-nl316-random (self)                     3_311.70       582.55     3_894.24       0.4078          1.0601            1.0574         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               3_752.21       136.96     3_889.18       0.1869         14.2552            1.2671         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              3_752.21       136.75     3_888.96       0.1843         21.8993            1.2820         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              3_752.21       134.55     3_886.77       0.1838         25.1324            1.2871         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              3_752.21       205.94     3_958.16       0.5832          1.8925            1.0188         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              3_752.21       262.02     4_014.24       0.7452          1.2735            1.0068         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             3_752.21       214.56     3_966.77       0.5641          2.3912            1.0191         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             3_752.21       257.04     4_009.25       0.7154          1.3978            1.0073         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             3_752.21       200.56     3_952.78       0.5598          2.5788            1.0191         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             3_752.21       266.75     4_018.96       0.7091          1.4462            1.0075         1.93
IVF-Binary-256-nl158-pca (self)                        3_752.21       646.68     4_398.89       0.5637          2.4135            1.0193         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              3_275.01       146.89     3_421.90       0.1852         19.1910            1.2540         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              3_275.01       135.30     3_410.31       0.1843         22.5823            1.2737         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              3_275.01       139.05     3_414.06       0.1838         27.6893            1.2891         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             3_275.01       203.64     3_478.65       0.5692          2.0866            1.0187         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             3_275.01       268.35     3_543.36       0.7228          1.3200            1.0072         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             3_275.01       202.97     3_477.98       0.5634          2.2762            1.0190         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             3_275.01       286.32     3_561.32       0.7145          1.3747            1.0074         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             3_275.01       210.95     3_485.96       0.5585          2.5583            1.0192         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             3_275.01       266.08     3_541.09       0.7061          1.4591            1.0076         2.00
IVF-Binary-256-nl223-pca (self)                        3_275.01       575.54     3_850.54       0.5634          2.3254            1.0192         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              3_435.36       140.15     3_575.51       0.1853         19.4866            1.2481         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              3_435.36       144.66     3_580.02       0.1849         20.8112            1.2534         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              3_435.36       217.26     3_652.62       0.1842         26.9898            1.2888         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             3_435.36       234.57     3_669.93       0.5686          2.0714            1.0187         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             3_435.36       284.82     3_720.18       0.7218          1.3128            1.0071         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             3_435.36       225.56     3_660.91       0.5653          2.1695            1.0188         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             3_435.36       286.57     3_721.92       0.7174          1.3396            1.0073         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             3_435.36       224.38     3_659.74       0.5590          2.5037            1.0191         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             3_435.36       289.87     3_725.22       0.7072          1.4380            1.0076         2.09
IVF-Binary-256-nl316-pca (self)                        3_435.36       639.00     4_074.36       0.5656          2.2122            1.0190         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_243.23       235.70     6_478.93       0.1613          1.3437            1.3212         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_243.23       243.35     6_486.58       0.1613          1.3437            1.3212         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_243.23       245.20     6_488.43       0.1613          1.3437            1.3212         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_243.23       299.65     6_542.88       0.3829          1.0672            1.0659         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_243.23       347.10     6_590.33       0.4931          1.0411            1.0384         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_243.23       291.86     6_535.10       0.3829          1.0672            1.0659         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_243.23       334.05     6_577.28       0.4931          1.0411            1.0384         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_243.23       278.22     6_521.45       0.3829          1.0672            1.0659         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_243.23       335.59     6_578.82       0.4931          1.0411            1.0384         3.71
IVF-Binary-512-nl158-random (self)                     6_243.23       862.03     7_105.26       0.3848          1.0655            1.0659         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_854.63       242.10     6_096.73       0.1711          1.3022            1.2803         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_854.63       238.89     6_093.53       0.1711          1.3025            1.2807         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_854.63       251.54     6_106.18       0.1711          1.3025            1.2807         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_854.63       294.72     6_149.35       0.4021          1.0606            1.0594         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_854.63       346.87     6_201.50       0.5154          1.0371            1.0346         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_854.63       281.39     6_136.03       0.4017          1.0608            1.0595         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_854.63       374.48     6_229.12       0.5146          1.0372            1.0347         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_854.63       301.19     6_155.82       0.4017          1.0608            1.0595         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_854.63       363.29     6_217.92       0.5146          1.0372            1.0347         3.77
IVF-Binary-512-nl223-random (self)                     5_854.63       928.82     6_783.45       0.4033          1.0594            1.0594         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           6_025.18       248.40     6_273.58       0.1758          1.2873            1.2661         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           6_025.18       261.04     6_286.22       0.1758          1.2879            1.2667         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           6_025.18       256.56     6_281.74       0.1758          1.2879            1.2667         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          6_025.18       318.49     6_343.67       0.4078          1.0591            1.0574         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          6_025.18       358.42     6_383.60       0.5205          1.0365            1.0337         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          6_025.18       308.94     6_334.12       0.4072          1.0593            1.0576         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          6_025.18       367.10     6_392.28       0.5194          1.0367            1.0338         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          6_025.18       328.77     6_353.95       0.4073          1.0592            1.0575         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          6_025.18       368.67     6_393.85       0.5195          1.0367            1.0338         3.86
IVF-Binary-512-nl316-random (self)                     6_025.18       889.22     6_914.40       0.4096          1.0580            1.0577         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               6_326.89       235.70     6_562.59       0.2208          2.2207            1.1931         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              6_326.89       231.85     6_558.73       0.2207          2.2360            1.1931         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              6_326.89       222.90     6_549.79       0.2207          2.2360            1.1931         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              6_326.89       286.04     6_612.92       0.6801          1.1057            1.0135         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              6_326.89       336.74     6_663.63       0.8261          1.0417            1.0038         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             6_326.89       284.22     6_611.11       0.6794          1.1060            1.0136         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             6_326.89       384.30     6_711.19       0.8258          1.0417            1.0038         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             6_326.89       304.95     6_631.84       0.6794          1.1060            1.0136         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             6_326.89       358.69     6_685.58       0.8258          1.0417            1.0038         3.71
IVF-Binary-512-nl158-pca (self)                        6_326.89       900.74     7_227.62       0.6801          1.1092            1.0136         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              5_837.60       221.58     6_059.18       0.2215          2.2240            1.1912         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              5_837.60       225.04     6_062.64       0.2212          2.2316            1.1923         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              5_837.60       227.59     6_065.18       0.2212          2.2330            1.1923         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             5_837.60       284.87     6_122.47       0.6810          1.1042            1.0135         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             5_837.60       332.18     6_169.77       0.8274          1.0392            1.0038         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             5_837.60       283.47     6_121.07       0.6803          1.1050            1.0136         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             5_837.60       334.48     6_172.08       0.8266          1.0397            1.0038         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             5_837.60       295.55     6_133.14       0.6802          1.1053            1.0136         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             5_837.60       340.09     6_177.68       0.8265          1.0398            1.0038         3.77
IVF-Binary-512-nl223-pca (self)                        5_837.60       852.22     6_689.82       0.6811          1.1070            1.0135         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              6_002.53       227.40     6_229.93       0.2218          2.2048            1.1900         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              6_002.53       227.45     6_229.98       0.2217          2.2108            1.1911         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              6_002.53       231.19     6_233.72       0.2217          2.2203            1.1915         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             6_002.53       293.33     6_295.86       0.6815          1.1020            1.0135         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             6_002.53       344.44     6_346.97       0.8280          1.0386            1.0038         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             6_002.53       292.54     6_295.07       0.6808          1.1029            1.0135         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             6_002.53       335.05     6_337.59       0.8271          1.0393            1.0038         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             6_002.53       288.51     6_291.04       0.6806          1.1032            1.0135         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             6_002.53       345.87     6_348.40       0.8269          1.0394            1.0038         3.86
IVF-Binary-512-nl316-pca (self)                        6_002.53       861.86     6_864.40       0.6820          1.1048            1.0134         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_383.22       420.87    11_804.09       0.1940          1.2725            1.2657         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_383.22       414.12    11_797.34       0.1940          1.2725            1.2657         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_383.22       416.18    11_799.41       0.1940          1.2725            1.2657         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_383.22       470.38    11_853.60       0.4237          1.0543            1.0547         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_383.22       526.41    11_909.64       0.5462          1.0323            1.0303         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_383.22       472.69    11_855.91       0.4237          1.0543            1.0547         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_383.22       527.32    11_910.54       0.5462          1.0323            1.0303         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_383.22       473.28    11_856.50       0.4237          1.0543            1.0547         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_383.22       534.73    11_917.95       0.5462          1.0323            1.0303         7.26
IVF-Binary-1024-nl158-random (self)                   11_383.22     1_486.14    12_869.36       0.4255          1.0541            1.0546         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_879.60       415.47    11_295.07       0.1972          1.2567            1.2493         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_879.60       419.98    11_299.58       0.1971          1.2572            1.2499         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_879.60       424.70    11_304.30       0.1971          1.2572            1.2499         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_879.60       473.34    11_352.94       0.4345          1.0516            1.0515         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_879.60       520.94    11_400.54       0.5571          1.0307            1.0286         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_879.60       473.54    11_353.14       0.4341          1.0517            1.0517         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_879.60       526.21    11_405.81       0.5566          1.0308            1.0287         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_879.60       479.91    11_359.50       0.4341          1.0517            1.0517         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_879.60       530.41    11_410.01       0.5566          1.0308            1.0287         7.32
IVF-Binary-1024-nl223-random (self)                   10_879.60     1_486.64    12_366.23       0.4359          1.0514            1.0517         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         11_099.48       420.80    11_520.29       0.1989          1.2508            1.2435         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)         11_099.48       431.68    11_531.16       0.1988          1.2515            1.2441         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)         11_099.48       434.40    11_533.88       0.1988          1.2515            1.2441         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)        11_099.48       482.50    11_581.98       0.4379          1.0509            1.0505         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)        11_099.48       525.30    11_624.78       0.5598          1.0304            1.0283         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)        11_099.48       478.50    11_577.99       0.4373          1.0510            1.0506         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)        11_099.48       532.72    11_632.20       0.5589          1.0306            1.0284         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)        11_099.48       514.32    11_613.81       0.4374          1.0510            1.0506         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)        11_099.48       533.26    11_632.75       0.5590          1.0305            1.0284         7.42
IVF-Binary-1024-nl316-random (self)                   11_099.48     1_501.71    12_601.20       0.4390          1.0508            1.0508         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)             11_391.47       411.51    11_802.98       0.2570          1.3524            1.1746         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            11_391.47       418.13    11_809.60       0.2570          1.3524            1.1746         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            11_391.47       437.25    11_828.73       0.2570          1.3524            1.1746         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            11_391.47       472.52    11_864.00       0.7238          1.0345            1.0104         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            11_391.47       533.71    11_925.18       0.8568          1.0153            1.0023         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           11_391.47       485.28    11_876.76       0.7238          1.0345            1.0104         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           11_391.47       569.54    11_961.02       0.8568          1.0153            1.0023         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           11_391.47       506.30    11_897.77       0.7238          1.0345            1.0104         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           11_391.47       560.49    11_951.97       0.8568          1.0153            1.0023         7.26
IVF-Binary-1024-nl158-pca (self)                      11_391.47     1_542.42    12_933.90       0.7252          1.0329            1.0103         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            10_980.78       493.49    11_474.26       0.2576          1.3508            1.1735         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            10_980.78       440.61    11_421.39       0.2574          1.3518            1.1742         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            10_980.78       513.18    11_493.96       0.2574          1.3518            1.1742         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           10_980.78       518.64    11_499.42       0.7250          1.0335            1.0103         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           10_980.78       525.75    11_506.52       0.8581          1.0147            1.0023         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           10_980.78       477.01    11_457.79       0.7247          1.0337            1.0104         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           10_980.78       526.68    11_507.45       0.8575          1.0148            1.0023         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           10_980.78       483.00    11_463.77       0.7247          1.0336            1.0104         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           10_980.78       541.39    11_522.17       0.8575          1.0148            1.0023         7.32
IVF-Binary-1024-nl223-pca (self)                      10_980.78     1_495.58    12_476.36       0.7261          1.0322            1.0102         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            11_147.02       421.54    11_568.55       0.2580          1.3492            1.1731         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            11_147.02       444.59    11_591.61       0.2580          1.3497            1.1735         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            11_147.02       427.38    11_574.39       0.2579          1.3499            1.1736         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           11_147.02       481.44    11_628.46       0.7257          1.0335            1.0103         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           11_147.02       525.15    11_672.17       0.8587          1.0149            1.0023         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           11_147.02       475.65    11_622.66       0.7253          1.0336            1.0103         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           11_147.02       527.65    11_674.67       0.8581          1.0149            1.0023         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           11_147.02       486.25    11_633.27       0.7253          1.0335            1.0104         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           11_147.02       538.85    11_685.87       0.8581          1.0149            1.0023         7.42
IVF-Binary-1024-nl316-pca (self)                      11_147.02     1_528.86    12_675.88       0.7268          1.0319            1.0102         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)                982.65       271.06     1_253.71       0.0604        339.0708          301.2567         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               982.65       292.14     1_274.79       0.0584        646.7529          625.6251         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               982.65       309.73     1_292.38       0.0581        710.8123          692.7255         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               982.65       287.85     1_270.50       0.2402         12.7621           10.6530         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               982.65       500.58     1_483.23       0.7125          1.3805            1.0092         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              982.65       308.55     1_291.20       0.1305         18.9834           18.2392         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              982.65       513.72     1_496.37       0.5024          4.8334            1.0288         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              982.65       339.14     1_321.79       0.1206         19.2033           18.3189         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              982.65       538.16     1_520.81       0.4474          5.9673            1.0402         1.68
IVF-Binary-256-nl158-sign (self)                         982.65       947.49     1_930.14       0.1311         19.1169           18.3513         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               557.97       291.81       849.78       0.0745        330.3456          242.0893         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               557.97       303.60       861.57       0.0575        523.6539          462.5988         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               557.97       327.54       885.51       0.0543        646.9433          644.7309         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              557.97       313.71       871.68       0.2680         13.4476            5.8520         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              557.97       548.52     1_106.49       0.5451          5.9049            1.0202         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              557.97       321.36       879.34       0.1819         19.0778           17.8634         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              557.97       534.90     1_092.87       0.4633          6.4645            1.0375         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              557.97       343.39       901.36       0.1175         22.8228           21.9953         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              557.97       559.80     1_117.77       0.3815          7.4572            1.0641         1.75
IVF-Binary-256-nl223-sign (self)                         557.97       988.80     1_546.77       0.1853         19.1727           17.9324         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               725.26       313.78     1_039.04       0.0835        323.0485          202.1954         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               725.26       314.43     1_039.69       0.0756        395.7054          299.1095         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               725.26       340.87     1_066.13       0.0590        673.9990          711.5677         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              725.26       332.47     1_057.72       0.2813         15.5690            5.6256         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              725.26       535.76     1_261.02       0.5377          2.7530            1.0229         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              725.26       338.54     1_063.80       0.2461         19.0979           17.6045         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              725.26       543.76     1_269.02       0.4848          4.3947            1.0327         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              725.26       358.23     1_083.48       0.1196         25.9642           25.1980         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              725.26       576.98     1_302.24       0.3660          6.9475            1.0688         1.84
IVF-Binary-256-nl316-sign (self)                         725.26     1_026.87     1_752.13       0.2474         19.2637           17.1280         1.84
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 512D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.49     1_219.93     1_288.42       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.49     4_053.63     4_122.12       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_852.93       435.87     6_288.80       0.1109          1.3511            1.3090         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_852.93       552.41     6_405.34       0.3145          1.0825            1.0612         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_852.93       681.98     6_534.91       0.4104          1.0522            1.0368         2.03
ExhaustiveBinary-256-random (self)                     5_852.93     1_752.50     7_605.43       0.3162          1.0784            1.0600         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_048.80       442.02     6_490.82       0.1336        288.8429          235.6612         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_048.80       571.46     6_620.26       0.3611          4.5931            1.0304         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_048.80       720.41     6_769.21       0.4705          2.4399            1.0190         2.03
ExhaustiveBinary-256-pca (self)                        6_048.80     1_819.18     7_867.98       0.3591          4.5254            1.0304         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_487.42       693.62    12_181.03       0.1529          1.2600            1.2299         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_487.42       826.53    12_313.95       0.3465          1.0564            1.0513         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_487.42       948.07    12_435.48       0.4455          1.0358            1.0312         4.05
ExhaustiveBinary-512-random (self)                    11_487.42     2_645.42    14_132.84       0.3476          1.0547            1.0512         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_621.26       693.18    12_314.44       0.1525        897.5233          752.8813         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_621.26       825.57    12_446.83       0.3949          9.5587            1.0271         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_621.26       959.49    12_580.74       0.4963          5.7543            1.0164         4.05
ExhaustiveBinary-512-pca (self)                       11_621.26     2_701.25    14_322.50       0.3914          9.3357            1.0275         4.05
ExhaustiveBinary-1024-random_no_rr (query)            22_745.08     1_247.53    23_992.62       0.1816          1.2043            1.1936         8.11
ExhaustiveBinary-1024-random-rf10 (query)             22_745.08     1_386.13    24_131.22       0.3748          1.0447            1.0451         8.11
ExhaustiveBinary-1024-random-rf20 (query)             22_745.08     1_533.68    24_278.76       0.4789          1.0282            1.0270         8.11
ExhaustiveBinary-1024-random (self)                   22_745.08     4_540.33    27_285.42       0.3754          1.0447            1.0451         8.11
ExhaustiveBinary-1024-pca_no_rr (query)               22_987.63     1_266.90    24_254.53       0.2289          1.6157            1.1146         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                22_987.63     1_389.91    24_377.54       0.6908          1.0533            1.0085         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                22_987.63     1_534.38    24_522.00       0.8304          1.0213            1.0025         8.11
ExhaustiveBinary-1024-pca (self)                      22_987.63     4_565.40    27_553.03       0.6878          1.0558            1.0086         8.11
ExhaustiveBinary-512-sign_no_rr (query)                  145.20       689.77       834.96       0.1518          1.2700            1.2528         3.05
ExhaustiveBinary-512-sign-rf10 (query)                   145.20       731.56       876.75       0.3399          1.0607            1.0535         3.05
ExhaustiveBinary-512-sign-rf20 (query)                   145.20     1_170.83     1_316.02       0.4405          1.0369            1.0318         3.05
ExhaustiveBinary-512-sign (self)                         145.20     2_380.09     2_525.28       0.3407          1.0594            1.0531         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)            7_459.60       250.61     7_710.21       0.1150          1.3343            1.2989         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           7_459.60       252.25     7_711.85       0.1150          1.3343            1.2989         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           7_459.60       252.99     7_712.59       0.1150          1.3343            1.2989         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           7_459.60       331.42     7_791.02       0.3211          1.0782            1.0587         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           7_459.60       403.36     7_862.96       0.4184          1.0495            1.0354         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          7_459.60       326.88     7_786.48       0.3211          1.0782            1.0587         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          7_459.60       405.99     7_865.59       0.4184          1.0495            1.0354         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          7_459.60       325.24     7_784.84       0.3211          1.0782            1.0587         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          7_459.60       406.76     7_866.36       0.4184          1.0495            1.0354         2.34
IVF-Binary-256-nl158-random (self)                     7_459.60       907.53     8_367.13       0.3229          1.0742            1.0575         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           6_681.84       256.43     6_938.27       0.1336          1.2722            1.2321         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)           6_681.84       256.88     6_938.71       0.1336          1.2722            1.2321         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)           6_681.84       257.85     6_939.68       0.1336          1.2722            1.2321         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)          6_681.84       334.44     7_016.28       0.3734          1.0541            1.0444         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)          6_681.84       412.38     7_094.22       0.4736          1.0348            1.0271         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)          6_681.84       333.28     7_015.11       0.3734          1.0541            1.0444         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)          6_681.84       424.10     7_105.93       0.4736          1.0348            1.0271         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)          6_681.84       336.11     7_017.95       0.3734          1.0541            1.0444         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)          6_681.84       418.55     7_100.38       0.4736          1.0348            1.0271         2.47
IVF-Binary-256-nl223-random (self)                     6_681.84       934.20     7_616.04       0.3753          1.0508            1.0437         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)           6_934.17       275.81     7_209.98       0.1440          1.2536            1.2121         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           6_934.17       265.17     7_199.34       0.1440          1.2536            1.2121         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           6_934.17       266.10     7_200.27       0.1440          1.2536            1.2121         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          6_934.17       349.28     7_283.45       0.3845          1.0503            1.0417         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          6_934.17       424.86     7_359.03       0.4835          1.0332            1.0259         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          6_934.17       342.81     7_276.98       0.3845          1.0503            1.0417         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          6_934.17       430.38     7_364.56       0.4835          1.0332            1.0259         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          6_934.17       345.09     7_279.26       0.3845          1.0503            1.0417         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          6_934.17       426.80     7_360.97       0.4835          1.0332            1.0259         2.65
IVF-Binary-256-nl316-random (self)                     6_934.17       959.61     7_893.78       0.3860          1.0466            1.0412         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               7_689.13       250.20     7_939.34       0.1431         14.7258            1.3415         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              7_689.13       269.74     7_958.87       0.1410         19.4444            1.4544         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              7_689.13       253.34     7_942.48       0.1402         22.1170            1.4933         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              7_689.13       346.64     8_035.77       0.4473          1.6628            1.0237         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              7_689.13       432.93     8_122.07       0.6115          1.2161            1.0117         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             7_689.13       350.56     8_039.69       0.4308          1.9143            1.0240         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             7_689.13       435.63     8_124.76       0.5857          1.2813            1.0125         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             7_689.13       353.30     8_042.43       0.4250          2.0432            1.0241         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             7_689.13       446.03     8_135.16       0.5761          1.3191            1.0126         2.34
IVF-Binary-256-nl158-pca (self)                        7_689.13     1_015.19     8_704.32       0.4288          1.9033            1.0241         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              6_890.71       264.56     7_155.26       0.1418         16.9635            1.3027         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              6_890.71       259.69     7_150.40       0.1411         19.3935            1.4480         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              6_890.71       261.39     7_152.09       0.1405         23.0517            1.5097         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             6_890.71       349.80     7_240.51       0.4342          1.7147            1.0239         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             6_890.71       433.18     7_323.88       0.5911          1.2219            1.0125         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             6_890.71       354.38     7_245.09       0.4290          1.8228            1.0240         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             6_890.71       456.17     7_346.87       0.5822          1.2514            1.0126         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             6_890.71       351.62     7_242.33       0.4248          1.9641            1.0241         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             6_890.71       442.13     7_332.83       0.5747          1.2935            1.0127         2.47
IVF-Binary-256-nl223-pca (self)                        6_890.71     1_015.66     7_906.37       0.4270          1.8146            1.0242         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              7_138.88       265.71     7_404.59       0.1418         16.9963            1.2846         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              7_138.88       268.75     7_407.63       0.1414         18.3735            1.4060         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              7_138.88       275.64     7_414.52       0.1409         22.3424            1.5036         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             7_138.88       358.69     7_497.57       0.4337          1.7242            1.0239         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             7_138.88       443.71     7_582.59       0.5897          1.2244            1.0124         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             7_138.88       357.21     7_496.10       0.4305          1.7800            1.0240         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             7_138.88       443.95     7_582.83       0.5847          1.2399            1.0125         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             7_138.88       359.22     7_498.10       0.4260          1.9092            1.0241         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             7_138.88       457.94     7_596.82       0.5762          1.2801            1.0126         2.65
IVF-Binary-256-nl316-pca (self)                        7_138.88     1_033.18     8_172.06       0.4281          1.7771            1.0240         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           13_147.73       457.38    13_605.11       0.1550          1.2528            1.2250         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          13_147.73       459.68    13_607.41       0.1550          1.2528            1.2250         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          13_147.73       466.86    13_614.59       0.1550          1.2528            1.2250         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          13_147.73       540.85    13_688.57       0.3498          1.0552            1.0504         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          13_147.73       621.47    13_769.19       0.4497          1.0349            1.0306         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         13_147.73       534.47    13_682.20       0.3498          1.0552            1.0504         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         13_147.73       626.78    13_774.50       0.4497          1.0349            1.0306         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         13_147.73       544.00    13_691.72       0.3498          1.0552            1.0504         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         13_147.73       656.96    13_804.68       0.4497          1.0349            1.0306         4.36
IVF-Binary-512-nl158-random (self)                    13_147.73     1_621.99    14_769.71       0.3507          1.0536            1.0502         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          12_337.85       474.37    12_812.22       0.1644          1.2224            1.1966         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          12_337.85       471.49    12_809.34       0.1644          1.2224            1.1966         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          12_337.85       492.68    12_830.53       0.1644          1.2224            1.1966         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         12_337.85       544.09    12_881.93       0.3714          1.0473            1.0446         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         12_337.85       623.48    12_961.33       0.4711          1.0308            1.0275         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         12_337.85       544.54    12_882.39       0.3714          1.0473            1.0446         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         12_337.85       624.16    12_962.00       0.4711          1.0308            1.0275         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         12_337.85       560.57    12_898.42       0.3714          1.0473            1.0446         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         12_337.85       628.36    12_966.21       0.4711          1.0308            1.0275         4.49
IVF-Binary-512-nl223-random (self)                    12_337.85     1_646.89    13_984.74       0.3720          1.0462            1.0446         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          12_576.67       475.68    13_052.35       0.1675          1.2154            1.1888         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          12_576.67       478.23    13_054.90       0.1675          1.2154            1.1888         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          12_576.67       478.75    13_055.43       0.1675          1.2154            1.1888         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         12_576.67       550.90    13_127.58       0.3763          1.0461            1.0435         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         12_576.67       627.89    13_204.57       0.4754          1.0302            1.0270         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         12_576.67       576.52    13_153.20       0.3763          1.0461            1.0435         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         12_576.67       636.73    13_213.40       0.4754          1.0302            1.0270         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         12_576.67       555.20    13_131.87       0.3763          1.0461            1.0435         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         12_576.67       645.45    13_222.12       0.4754          1.0302            1.0270         4.67
IVF-Binary-512-nl316-random (self)                    12_576.67     1_674.00    14_250.67       0.3763          1.0452            1.0436         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              13_292.05       460.93    13_752.99       0.1861         17.3473            1.1607         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             13_292.05       459.96    13_752.02       0.1837         25.5859            1.1750         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             13_292.05       466.88    13_758.94       0.1828         31.0043            1.1845         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             13_292.05       556.25    13_848.30       0.5751          1.8379            1.0124         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             13_292.05       633.85    13_925.90       0.7310          1.2609            1.0049         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            13_292.05       552.77    13_844.82       0.5565          2.3045            1.0126         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            13_292.05       640.22    13_932.28       0.7021          1.3803            1.0053         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            13_292.05       554.68    13_846.74       0.5493          2.6504            1.0128         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            13_292.05       649.70    13_941.75       0.6910          1.4737            1.0054         4.36
IVF-Binary-512-nl158-pca (self)                       13_292.05     1_712.66    15_004.72       0.5535          2.3059            1.0128         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             12_522.11       465.99    12_988.10       0.1842         21.3504            1.1607         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             12_522.11       469.09    12_991.19       0.1834         25.5815            1.1729         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             12_522.11       481.03    13_003.14       0.1827         33.9565            1.1930         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            12_522.11       554.17    13_076.28       0.5593          2.0321            1.0126         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            12_522.11       638.49    13_160.60       0.7076          1.2987            1.0052         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            12_522.11       554.28    13_076.39       0.5532          2.2252            1.0127         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            12_522.11       637.35    13_159.46       0.6971          1.3561            1.0053         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            12_522.11       558.36    13_080.47       0.5470          2.6055            1.0129         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            12_522.11       644.28    13_166.39       0.6871          1.4687            1.0055         4.49
IVF-Binary-512-nl223-pca (self)                       12_522.11     1_697.85    14_219.96       0.5505          2.2265            1.0129         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             12_754.71       482.74    13_237.45       0.1843         21.5306            1.1594         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             12_754.71       476.25    13_230.96       0.1838         23.6640            1.1667         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             12_754.71       483.40    13_238.11       0.1832         31.8963            1.1888         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            12_754.71       565.80    13_320.51       0.5587          2.0592            1.0125         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            12_754.71       651.01    13_405.72       0.7059          1.3047            1.0052         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            12_754.71       561.12    13_315.83       0.5550          2.1573            1.0126         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            12_754.71       651.34    13_406.05       0.7000          1.3339            1.0053         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            12_754.71       566.98    13_321.69       0.5483          2.4688            1.0129         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            12_754.71       653.03    13_407.74       0.6892          1.4258            1.0055         4.67
IVF-Binary-512-nl316-pca (self)                       12_754.71     1_717.84    14_472.55       0.5522          2.1609            1.0128         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          24_569.48       898.63    25_468.11       0.1825          1.2017            1.1917         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)         24_569.48       939.82    25_509.30       0.1825          1.2017            1.1917         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)         24_569.48       874.19    25_443.67       0.1825          1.2017            1.1917         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)         24_569.48       977.32    25_546.79       0.3769          1.0441            1.0446         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)         24_569.48     1_057.75    25_627.22       0.4817          1.0278            1.0266         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)        24_569.48       948.36    25_517.84       0.3769          1.0441            1.0446         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)        24_569.48     1_032.67    25_602.15       0.4817          1.0278            1.0266         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)        24_569.48       956.71    25_526.19       0.3769          1.0441            1.0446         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)        24_569.48     1_125.90    25_695.38       0.4817          1.0278            1.0266         8.42
IVF-Binary-1024-nl158-random (self)                   24_569.48     3_330.04    27_899.52       0.3775          1.0441            1.0445         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)         23_796.43       865.13    24_661.56       0.1859          1.1883            1.1790         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         23_796.43       867.26    24_663.69       0.1859          1.1883            1.1790         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         23_796.43       888.88    24_685.31       0.1859          1.1883            1.1790         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        23_796.43       946.65    24_743.08       0.3873          1.0417            1.0419         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        23_796.43     1_014.15    24_810.57       0.4938          1.0263            1.0249         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        23_796.43       935.15    24_731.58       0.3873          1.0417            1.0419         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        23_796.43     1_018.93    24_815.36       0.4938          1.0263            1.0249         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        23_796.43       942.96    24_739.38       0.3873          1.0417            1.0419         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        23_796.43     1_029.62    24_826.05       0.4938          1.0263            1.0249         8.54
IVF-Binary-1024-nl223-random (self)                   23_796.43     2_972.35    26_768.78       0.3882          1.0416            1.0420         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         24_119.20       957.77    25_076.98       0.1871          1.1851            1.1757         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)         24_119.20       933.98    25_053.18       0.1871          1.1851            1.1757         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)         24_119.20       912.59    25_031.80       0.1871          1.1851            1.1757         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)        24_119.20       985.82    25_105.02       0.3898          1.0411            1.0414         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)        24_119.20     1_064.62    25_183.82       0.4958          1.0260            1.0247         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)        24_119.20       945.89    25_065.09       0.3898          1.0411            1.0414         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)        24_119.20     1_119.29    25_238.50       0.4958          1.0260            1.0247         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)        24_119.20     1_027.74    25_146.95       0.3898          1.0411            1.0414         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)        24_119.20     1_080.81    25_200.02       0.4958          1.0260            1.0247         8.73
IVF-Binary-1024-nl316-random (self)                   24_119.20     3_138.33    27_257.54       0.3905          1.0410            1.0415         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)             24_899.93       891.06    25_790.99       0.2293          1.6010            1.1148         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            24_899.93       870.21    25_770.14       0.2293          1.6037            1.1148         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            24_899.93       920.35    25_820.28       0.2293          1.6037            1.1148         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            24_899.93     1_048.07    25_948.01       0.6915          1.0527            1.0085         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            24_899.93     1_163.70    26_063.63       0.8309          1.0211            1.0025         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           24_899.93     1_039.75    25_939.69       0.6911          1.0528            1.0085         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           24_899.93     1_114.10    26_014.04       0.8308          1.0211            1.0025         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           24_899.93     1_028.63    25_928.57       0.6911          1.0528            1.0085         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           24_899.93     1_085.47    25_985.40       0.8308          1.0211            1.0025         8.42
IVF-Binary-1024-nl158-pca (self)                      24_899.93     3_087.33    27_987.27       0.6880          1.0553            1.0086         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            24_272.68     1_074.16    25_346.84       0.2296          1.5965            1.1145         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            24_272.68       953.20    25_225.88       0.2296          1.5966            1.1145         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            24_272.68       931.62    25_204.30       0.2296          1.5966            1.1145         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           24_272.68       974.96    25_247.64       0.6920          1.0499            1.0085         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           24_272.68     1_083.27    25_355.95       0.8315          1.0198            1.0025         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           24_272.68       992.92    25_265.60       0.6919          1.0499            1.0085         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           24_272.68     1_130.14    25_402.82       0.8314          1.0199            1.0025         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           24_272.68     1_020.03    25_292.71       0.6919          1.0499            1.0085         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           24_272.68     1_098.43    25_371.11       0.8314          1.0199            1.0025         8.54
IVF-Binary-1024-nl223-pca (self)                      24_272.68     3_011.36    27_284.04       0.6893          1.0521            1.0085         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            24_176.83       884.34    25_061.17       0.2301          1.5949            1.1143         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            24_176.83       882.04    25_058.87       0.2301          1.5949            1.1143         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            24_176.83       915.65    25_092.48       0.2301          1.5951            1.1143         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           24_176.83       964.99    25_141.82       0.6925          1.0494            1.0085         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           24_176.83     1_037.17    25_214.00       0.8317          1.0197            1.0025         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           24_176.83       963.78    25_140.61       0.6924          1.0494            1.0085         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           24_176.83     1_042.91    25_219.74       0.8317          1.0197            1.0025         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           24_176.83       977.29    25_154.12       0.6924          1.0494            1.0085         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           24_176.83     1_066.88    25_243.71       0.8317          1.0197            1.0025         8.73
IVF-Binary-1024-nl316-pca (self)                      24_176.83     3_057.03    27_233.86       0.6894          1.0519            1.0085         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)              1_816.58       437.01     2_253.59       0.0656        327.6909          208.3967         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)             1_816.58       467.35     2_283.92       0.0642        547.4464          360.0161         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)             1_816.58       485.12     2_301.70       0.0638        615.5449          479.5490         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)             1_816.58       465.58     2_282.16       0.2285         15.8942           16.1011         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)             1_816.58       820.00     2_636.57       0.7342          1.0280            1.0061         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)            1_816.58       495.62     2_312.20       0.1416         20.2673           21.4708         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)            1_816.58       840.04     2_656.62       0.5249          4.3538            1.0162         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)            1_816.58       509.79     2_326.36       0.1257         21.2309           22.1920         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)            1_816.58       876.63     2_693.20       0.4614          5.7308            1.0237         3.36
IVF-Binary-512-nl158-sign (self)                       1_816.58     1_437.79     3_254.37       0.1411         20.4652           21.6805         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               971.08       461.39     1_432.47       0.0732        401.4531          189.1446         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               971.08       478.42     1_449.50       0.0646        583.1073          476.3585         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               971.08       517.98     1_489.07       0.0642        732.8877          852.3300         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              971.08       515.50     1_486.58       0.2369         14.7391            9.3703         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              971.08       888.85     1_859.93       0.5809          2.3790            1.0130         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              971.08       557.89     1_528.97       0.1610         21.3471           22.9807         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              971.08       901.55     1_872.63       0.4856          4.2385            1.0213         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              971.08       556.95     1_528.03       0.1234         25.3001           25.3670         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              971.08       943.52     1_914.60       0.3760          8.6040            1.0429         3.49
IVF-Binary-512-nl223-sign (self)                         971.08     1_516.41     2_487.50       0.1578         21.4598           23.1244         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)             1_220.34       484.33     1_704.67       0.0683        381.9140          167.4888         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)             1_220.34       495.49     1_715.83       0.0670        488.5484          403.1886         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)             1_220.34       537.36     1_757.70       0.0663        746.8086          831.8734         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)            1_220.34       528.37     1_748.71       0.2539         11.2563            4.2341         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)            1_220.34       911.49     2_131.83       0.5646          2.6870            1.0142         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)            1_220.34       528.29     1_748.63       0.1958         14.5463            9.4992         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)            1_220.34       891.55     2_111.89       0.5034          3.2888            1.0197         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)            1_220.34       578.40     1_798.75       0.1199         25.9296           19.2330         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)            1_220.34       952.41     2_172.75       0.3402          5.3031            1.0540         3.67
IVF-Binary-512-nl316-sign (self)                       1_220.34     1_721.94     2_942.29       0.1934         14.6881            9.6092         3.67
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 768D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        99.54     1_761.25     1_860.79       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.54     5_816.62     5_916.16       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)              8_974.34       540.17     9_514.51       0.1141          1.2808            1.2432         2.28
ExhaustiveBinary-256-random-rf10 (query)               8_974.34       674.10     9_648.43       0.3149          1.0655            1.0475         2.28
ExhaustiveBinary-256-random-rf20 (query)               8_974.34       817.50     9_791.84       0.4080          1.0420            1.0293         2.28
ExhaustiveBinary-256-random (self)                     8_974.34     2_152.23    11_126.56       0.3170          1.0618            1.0471         2.28
ExhaustiveBinary-256-pca_no_rr (query)                 9_306.60       541.58     9_848.19       0.1206        288.1145          236.3901         2.28
ExhaustiveBinary-256-pca-rf10 (query)                  9_306.60       689.01     9_995.61       0.3155          3.9425            1.0290         2.28
ExhaustiveBinary-256-pca-rf20 (query)                  9_306.60       841.11    10_147.71       0.4166          2.2106            1.0189         2.28
ExhaustiveBinary-256-pca (self)                        9_306.60     2_211.07    11_517.68       0.3143          3.9926            1.0290         2.28
ExhaustiveBinary-512-random_no_rr (query)             17_658.25       927.05    18_585.30       0.1506          1.2093            1.1807         4.55
ExhaustiveBinary-512-random-rf10 (query)              17_658.25     1_054.45    18_712.69       0.3395          1.0453            1.0425         4.55
ExhaustiveBinary-512-random-rf20 (query)              17_658.25     1_203.97    18_862.21       0.4325          1.0293            1.0264         4.55
ExhaustiveBinary-512-random (self)                    17_658.25     3_653.32    21_311.56       0.3401          1.0435            1.0422         4.55
ExhaustiveBinary-512-pca_no_rr (query)                17_932.51       922.37    18_854.87       0.1352        868.1603          740.1379         4.55
ExhaustiveBinary-512-pca-rf10 (query)                 17_932.51     1_060.78    18_993.28       0.3351          6.8743            1.0277         4.55
ExhaustiveBinary-512-pca-rf20 (query)                 17_932.51     1_214.74    19_147.24       0.4290          4.3782            1.0179         4.55
ExhaustiveBinary-512-pca (self)                       17_932.51     3_456.20    21_388.70       0.3348          6.9733            1.0278         4.55
ExhaustiveBinary-1024-random_no_rr (query)            35_185.10     1_676.18    36_861.28       0.1762          1.1673            1.1570         9.11
ExhaustiveBinary-1024-random-rf10 (query)             35_185.10     1_820.16    37_005.26       0.3602          1.0383            1.0383         9.11
ExhaustiveBinary-1024-random-rf20 (query)             35_185.10     1_989.58    37_174.68       0.4619          1.0244            1.0230         9.11
ExhaustiveBinary-1024-random (self)                   35_185.10     5_992.85    41_177.95       0.3601          1.0377            1.0382         9.11
ExhaustiveBinary-1024-pca_no_rr (query)               35_782.77     1_780.25    37_563.03       0.2068          2.8363            1.0912         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                35_782.77     1_896.98    37_679.75       0.6440          1.1364            1.0083         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                35_782.77     2_101.51    37_884.28       0.7955          1.0504            1.0027         9.11
ExhaustiveBinary-1024-pca (self)                      35_782.77     6_625.82    42_408.59       0.6428          1.1368            1.0084         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  239.38       976.12     1_215.51       0.1691          1.1871            1.1718         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   239.38     1_064.29     1_303.67       0.3431          1.0433            1.0415         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   239.38     1_620.65     1_860.03       0.4437          1.0266            1.0249         4.58
ExhaustiveBinary-768-sign (self)                         239.38     3_096.00     3_335.38       0.3437          1.0423            1.0413         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)           11_508.80       365.95    11_874.76       0.1169          1.2711            1.2394         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)          11_508.80       375.41    11_884.22       0.1169          1.2711            1.2394         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)          11_508.80       375.74    11_884.55       0.1169          1.2711            1.2394         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)          11_508.80       457.66    11_966.47       0.3176          1.0634            1.0467         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)          11_508.80       554.66    12_063.47       0.4104          1.0411            1.0289         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)         11_508.80       459.60    11_968.41       0.3176          1.0634            1.0467         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)         11_508.80       555.86    12_064.67       0.4104          1.0411            1.0289         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)         11_508.80       457.65    11_966.45       0.3176          1.0634            1.0467         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)         11_508.80       561.30    12_070.11       0.4104          1.0411            1.0289         2.74
IVF-Binary-256-nl158-random (self)                    11_508.80     1_336.91    12_845.72       0.3197          1.0600            1.0463         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)          10_068.89       376.33    10_445.22       0.1316          1.2343            1.1950         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)          10_068.89       374.84    10_443.73       0.1316          1.2346            1.1952         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)          10_068.89       379.82    10_448.71       0.1316          1.2346            1.1952         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)         10_068.89       504.81    10_573.70       0.3458          1.0521            1.0395         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)         10_068.89       591.33    10_660.22       0.4475          1.0325            1.0241         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)         10_068.89       472.14    10_541.03       0.3455          1.0522            1.0396         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)         10_068.89       575.18    10_644.07       0.4470          1.0326            1.0242         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)         10_068.89       483.01    10_551.90       0.3456          1.0522            1.0396         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)         10_068.89       619.08    10_687.97       0.4470          1.0326            1.0242         2.93
IVF-Binary-256-nl223-random (self)                    10_068.89     1_455.96    11_524.85       0.3478          1.0487            1.0393         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)          10_474.90       412.45    10_887.35       0.1392          1.2180            1.1762         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)          10_474.90       389.33    10_864.23       0.1392          1.2187            1.1764         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)          10_474.90       390.13    10_865.03       0.1392          1.2187            1.1764         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)         10_474.90       482.03    10_956.94       0.3596          1.0465            1.0373         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)         10_474.90       581.09    11_056.00       0.4602          1.0294            1.0230         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)         10_474.90       487.56    10_962.46       0.3595          1.0465            1.0373         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)         10_474.90       607.64    11_082.54       0.4600          1.0295            1.0230         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)         10_474.90       493.28    10_968.18       0.3594          1.0465            1.0373         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)         10_474.90       606.24    11_081.14       0.4600          1.0295            1.0230         3.21
IVF-Binary-256-nl316-random (self)                    10_474.90     1_454.52    11_929.43       0.3615          1.0433            1.0370         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)              11_845.13       368.40    12_213.53       0.1289         12.7584            1.4470         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)             11_845.13       370.82    12_215.95       0.1268         18.0733            1.6895         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)             11_845.13       385.85    12_230.98       0.1262         20.8195            1.7937         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)             11_845.13       490.16    12_335.29       0.3981          1.5902            1.0231         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)             11_845.13       624.71    12_469.83       0.5591          1.1835            1.0123         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)            11_845.13       530.17    12_375.30       0.3805          1.8083            1.0233         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)            11_845.13       592.20    12_437.33       0.5312          1.2431            1.0129         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)            11_845.13       486.30    12_331.42       0.3760          1.9563            1.0234         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)            11_845.13       599.68    12_444.80       0.5237          1.2823            1.0130         2.74
IVF-Binary-256-nl158-pca (self)                       11_845.13     1_450.50    13_295.63       0.3805          1.7814            1.0233         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)             10_373.77       385.58    10_759.35       0.1278         16.7049            1.1940         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)             10_373.77       376.87    10_750.64       0.1272         19.3063            1.4996         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)             10_373.77       381.69    10_755.46       0.1266         22.8508            1.8404         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)            10_373.77       494.06    10_867.83       0.3847          1.6939            1.0231         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)            10_373.77       593.19    10_966.96       0.5376          1.2153            1.0128         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)            10_373.77       487.84    10_861.61       0.3793          1.8243            1.0233         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)            10_373.77       591.33    10_965.10       0.5283          1.2515            1.0129         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)            10_373.77       504.15    10_877.92       0.3751          1.9517            1.0234         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)            10_373.77       607.23    10_981.00       0.5208          1.2866            1.0131         2.93
IVF-Binary-256-nl223-pca (self)                       10_373.77     1_467.59    11_841.37       0.3790          1.7933            1.0233         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)             10_805.51       388.16    11_193.67       0.1278         16.4815            1.1707         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)             10_805.51       392.96    11_198.47       0.1274         17.7939            1.3294         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)             10_805.51       394.81    11_200.32       0.1267         21.7029            1.7965         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)            10_805.51       499.25    11_304.75       0.3844          1.6238            1.0231         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)            10_805.51       610.18    11_415.68       0.5371          1.1964            1.0128         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)            10_805.51       498.96    11_304.47       0.3813          1.6996            1.0233         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)            10_805.51       638.64    11_444.14       0.5321          1.2169            1.0129         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)            10_805.51       507.75    11_313.26       0.3760          1.8627            1.0235         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)            10_805.51       614.94    11_420.44       0.5225          1.2626            1.0131         3.21
IVF-Binary-256-nl316-pca (self)                       10_805.51     1_509.08    12_314.58       0.3811          1.6778            1.0233         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)           20_144.07       678.94    20_823.01       0.1522          1.2056            1.1784         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)          20_144.07       680.81    20_824.88       0.1522          1.2056            1.1784         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)          20_144.07       684.68    20_828.75       0.1522          1.2056            1.1784         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)          20_144.07       772.03    20_916.10       0.3406          1.0448            1.0422         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)          20_144.07       868.94    21_013.01       0.4343          1.0290            1.0261         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)         20_144.07       775.42    20_919.49       0.3406          1.0448            1.0422         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)         20_144.07       880.10    21_024.17       0.4343          1.0290            1.0261         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)         20_144.07       777.82    20_921.89       0.3406          1.0448            1.0422         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)         20_144.07       879.39    21_023.46       0.4343          1.0290            1.0261         5.02
IVF-Binary-512-nl158-random (self)                    20_144.07     2_399.03    22_543.10       0.3414          1.0431            1.0419         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)          18_778.60       695.25    19_473.85       0.1589          1.1868            1.1603         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)          18_778.60       688.46    19_467.06       0.1588          1.1870            1.1604         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)          18_778.60       699.65    19_478.25       0.1588          1.1870            1.1604         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)         18_778.60       787.46    19_566.06       0.3555          1.0408            1.0385         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)         18_778.60       881.77    19_660.37       0.4528          1.0262            1.0238         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)         18_778.60       787.39    19_565.99       0.3552          1.0408            1.0386         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)         18_778.60       888.70    19_667.30       0.4522          1.0262            1.0239         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)         18_778.60       794.89    19_573.49       0.3552          1.0408            1.0386         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)         18_778.60       887.29    19_665.89       0.4522          1.0262            1.0239         5.21
IVF-Binary-512-nl223-random (self)                    18_778.60     2_426.09    21_204.69       0.3555          1.0394            1.0385         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)          19_259.71       698.21    19_957.91       0.1619          1.1799            1.1536         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)          19_259.71       697.33    19_957.03       0.1618          1.1800            1.1538         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)          19_259.71       702.76    19_962.47       0.1618          1.1801            1.1538         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)         19_259.71       792.74    20_052.45       0.3616          1.0393            1.0373         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)         19_259.71       891.91    20_151.62       0.4576          1.0255            1.0234         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)         19_259.71       794.96    20_054.67       0.3614          1.0393            1.0374         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)         19_259.71       905.53    20_165.23       0.4573          1.0255            1.0235         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)         19_259.71       802.30    20_062.01       0.3614          1.0394            1.0374         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)         19_259.71       897.38    20_157.09       0.4572          1.0255            1.0235         5.48
IVF-Binary-512-nl316-random (self)                    19_259.71     2_464.64    21_724.35       0.3614          1.0379            1.0374         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)              20_821.65       694.16    21_515.81       0.1607         14.7248            1.1597         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)             20_821.65       680.50    21_502.15       0.1582         23.0357            1.4088         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)             20_821.65       695.39    21_517.04       0.1576         28.4656            1.6323         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)             20_821.65       849.69    21_671.34       0.4968          1.6945            1.0147         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)             20_821.65       934.06    21_755.71       0.6607          1.2136            1.0067         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)            20_821.65       813.30    21_634.95       0.4782          2.0527            1.0149         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)            20_821.65       957.19    21_778.84       0.6301          1.3106            1.0072         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)            20_821.65       851.64    21_673.29       0.4726          2.3311            1.0150         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)            20_821.65       969.74    21_791.40       0.6205          1.3855            1.0073         5.02
IVF-Binary-512-nl158-pca (self)                       20_821.65     2_668.59    23_490.24       0.4781          2.0337            1.0149         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)             19_190.40       709.48    19_899.88       0.1589         20.9282            1.1296         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)             19_190.40       684.67    19_875.06       0.1582         25.1702            1.2241         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)             19_190.40       697.94    19_888.34       0.1575         33.6171            1.7613         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)            19_190.40       879.33    20_069.73       0.4821          1.8819            1.0146         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)            19_190.40       959.53    20_149.93       0.6364          1.2668            1.0070         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)            19_190.40       859.03    20_049.43       0.4760          2.1089            1.0148         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)            19_190.40       994.59    20_184.99       0.6259          1.3278            1.0072         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)            19_190.40       840.89    20_031.29       0.4699          2.4197            1.0151         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)            19_190.40     1_032.55    20_222.95       0.6153          1.4150            1.0074         5.21
IVF-Binary-512-nl223-pca (self)                       19_190.40     2_792.61    21_983.01       0.4757          2.0743            1.0149         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)             19_841.72       697.79    20_539.51       0.1590         20.4279            1.1269         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)             19_841.72       710.34    20_552.06       0.1585         22.6091            1.1414         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)             19_841.72       717.72    20_559.44       0.1578         30.8068            1.6323         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)            19_841.72       887.28    20_729.00       0.4814          1.8151            1.0147         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)            19_841.72     1_020.36    20_862.08       0.6357          1.2483            1.0070         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)            19_841.72       902.77    20_744.49       0.4779          1.9322            1.0148         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)            19_841.72       949.50    20_791.22       0.6300          1.2798            1.0071         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)            19_841.72       825.71    20_667.43       0.4707          2.2746            1.0151         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)            19_841.72       964.47    20_806.19       0.6176          1.3719            1.0074         5.48
IVF-Binary-512-nl316-pca (self)                       19_841.72     2_876.67    22_718.39       0.4780          1.9133            1.0148         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)          38_231.14     1_307.96    39_539.11       0.1768          1.1660            1.1560         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)         38_231.14     1_317.42    39_548.57       0.1768          1.1660            1.1560         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)         38_231.14     1_317.83    39_548.98       0.1768          1.1660            1.1560         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)         38_231.14     1_402.79    39_633.93       0.3610          1.0381            1.0381         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)         38_231.14     1_484.62    39_715.77       0.4630          1.0242            1.0229         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)        38_231.14     1_392.71    39_623.85       0.3610          1.0381            1.0381         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)        38_231.14     1_484.87    39_716.02       0.4630          1.0242            1.0229         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)        38_231.14     1_396.03    39_627.18       0.3610          1.0381            1.0381         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)        38_231.14     1_504.86    39_736.01       0.4630          1.0242            1.0229         9.57
IVF-Binary-1024-nl158-random (self)                   38_231.14     4_700.38    42_931.52       0.3610          1.0375            1.0381         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)         36_423.32     1_359.04    37_782.35       0.1790          1.1581            1.1484         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)         36_423.32     1_335.23    37_758.55       0.1789          1.1584            1.1487         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)         36_423.32     1_319.27    37_742.59       0.1789          1.1584            1.1487         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)        36_423.32     1_411.27    37_834.58       0.3691          1.0364            1.0364         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)        36_423.32     1_507.97    37_931.28       0.4723          1.0232            1.0220         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)        36_423.32     1_434.85    37_858.16       0.3687          1.0364            1.0365         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)        36_423.32     1_522.54    37_945.86       0.4716          1.0233            1.0220         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)        36_423.32     1_420.30    37_843.62       0.3688          1.0364            1.0365         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)        36_423.32     1_517.51    37_940.83       0.4716          1.0233            1.0220         9.76
IVF-Binary-1024-nl223-random (self)                   36_423.32     4_616.20    41_039.51       0.3685          1.0359            1.0364         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)         36_768.40     1_320.87    38_089.27       0.1803          1.1548            1.1453        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)         36_768.40     1_319.91    38_088.31       0.1802          1.1550            1.1455        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)         36_768.40     1_341.68    38_110.09       0.1802          1.1551            1.1455        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)        36_768.40     1_409.56    38_177.97       0.3724          1.0358            1.0359        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)        36_768.40     1_509.95    38_278.35       0.4756          1.0228            1.0216        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)        36_768.40     1_416.74    38_185.14       0.3723          1.0358            1.0359        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)        36_768.40     1_507.14    38_275.55       0.4754          1.0228            1.0216        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)        36_768.40     1_513.39    38_281.80       0.3723          1.0358            1.0359        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)        36_768.40     1_517.83    38_286.23       0.4753          1.0228            1.0216        10.04
IVF-Binary-1024-nl316-random (self)                   36_768.40     4_540.64    41_309.04       0.3721          1.0352            1.0358        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)             37_878.03     1_297.01    39_175.04       0.2075          2.6256            1.0912         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)            37_878.03     1_304.69    39_182.72       0.2071          2.7616            1.0912         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)            37_878.03     1_330.09    39_208.12       0.2071          2.7766            1.0912         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)            37_878.03     1_399.12    39_277.15       0.6494          1.1311            1.0081         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)            37_878.03     1_497.12    39_375.15       0.7999          1.0485            1.0025         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)           37_878.03     1_402.07    39_280.10       0.6450          1.1325            1.0083         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)           37_878.03     1_509.59    39_387.62       0.7963          1.0492            1.0027         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)           37_878.03     1_423.25    39_301.28       0.6445          1.1326            1.0083         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)           37_878.03     1_521.28    39_399.31       0.7960          1.0492            1.0027         9.57
IVF-Binary-1024-nl158-pca (self)                      37_878.03     4_552.35    42_430.38       0.6440          1.1334            1.0083         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)            36_513.71     1_304.75    37_818.46       0.2077          2.7121            1.0906         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)            36_513.71     1_301.46    37_815.17       0.2075          2.7485            1.0909         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)            36_513.71     1_312.12    37_825.83       0.2075          2.7713            1.0909         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)           36_513.71     1_411.55    37_925.26       0.6477          1.1240            1.0082         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)           36_513.71     1_513.39    38_027.10       0.7989          1.0468            1.0027         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)           36_513.71     1_402.21    37_915.92       0.6462          1.1292            1.0083         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)           36_513.71     1_518.69    38_032.40       0.7973          1.0482            1.0027         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)           36_513.71     1_424.14    37_937.85       0.6457          1.1305            1.0083         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)           36_513.71     1_523.44    38_037.15       0.7968          1.0486            1.0027         9.76
IVF-Binary-1024-nl223-pca (self)                      36_513.71     4_534.23    41_047.94       0.6447          1.1296            1.0083         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)            36_935.51     1_327.15    38_262.66       0.2075          2.7124            1.0905        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)            36_935.51     1_317.88    38_253.38       0.2074          2.7332            1.0908        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)            36_935.51     1_324.02    38_259.52       0.2074          2.7605            1.0909        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)           36_935.51     1_416.29    38_351.80       0.6474          1.1223            1.0083        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)           36_935.51     1_512.94    38_448.44       0.7985          1.0457            1.0027        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)           36_935.51     1_435.47    38_370.97       0.6466          1.1248            1.0083        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)           36_935.51     1_533.48    38_468.98       0.7976          1.0465            1.0027        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)           36_935.51     1_425.75    38_361.26       0.6458          1.1262            1.0083        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)           36_935.51     1_533.81    38_469.31       0.7968          1.0471            1.0027        10.04
IVF-Binary-1024-nl316-pca (self)                      36_935.51     4_556.98    41_492.48       0.6451          1.1256            1.0083        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)              2_606.25       577.51     3_183.75       0.0683        392.3383          248.0775         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)             2_606.25       639.12     3_245.37       0.0671        599.1367          464.9128         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)             2_606.25       672.32     3_278.57       0.0669        664.1884          501.3283         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)             2_606.25       627.72     3_233.96       0.2247         16.2957           17.1916         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)             2_606.25     1_112.66     3_718.90       0.7264          1.0235            1.0049         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)            2_606.25       674.55     3_280.79       0.1381         20.2411           21.5074         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)            2_606.25     1_172.32     3_778.57       0.5311          4.8959            1.0116         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)            2_606.25       720.79     3_327.04       0.1232         22.6937           22.6334         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)            2_606.25     1_210.21     3_816.45       0.4592          5.6620            1.0180         5.04
IVF-Binary-768-nl158-sign (self)                       2_606.25     1_986.46     4_592.71       0.1381         20.2340           21.5947         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)             1_320.72       629.46     1_950.18       0.0838        322.6009          138.5984         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)             1_320.72       656.68     1_977.41       0.0705        422.9169          307.0583         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)             1_320.72       709.97     2_030.69       0.0626        721.8450          842.0128         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)            1_320.72       685.26     2_005.98       0.3018         17.9210            4.4404         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)            1_320.72     1_226.54     2_547.26       0.5599          6.1700            1.0085         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)            1_320.72       710.23     2_030.95       0.2273         20.4026           10.6272         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)            1_320.72     1_206.53     2_527.26       0.4968          7.9296            1.0133         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)            1_320.72       771.19     2_091.91       0.1226         30.9641           21.5813         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)            1_320.72     1_266.78     2_587.50       0.3579          9.4895            1.0431         5.23
IVF-Binary-768-nl223-sign (self)                       1_320.72     2_124.72     3_445.45       0.2258         20.7209           12.1499         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)             1_790.23       671.35     2_461.58       0.0902        339.0098          154.5000         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)             1_790.23       689.96     2_480.20       0.0758        428.5042          211.1469         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)             1_790.23       755.45     2_545.68       0.0675        757.1587          810.8178         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)            1_790.23       736.25     2_526.49       0.2936         12.5064            3.7891         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)            1_790.23     1_214.16     3_004.39       0.5371          4.4503            1.0117         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)            1_790.23       746.71     2_536.94       0.2574         13.8389            4.0363         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)            1_790.23     1_244.87     3_035.11       0.4890          5.8066            1.0154         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)            1_790.23       808.21     2_598.44       0.1284         23.6497           19.1251         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)            1_790.23     1_304.20     3_094.43       0.3559          7.2503            1.0411         5.51
IVF-Binary-768-nl316-sign (self)                       1_790.23     2_234.36     4_024.59       0.2564         14.0152            4.2503         5.51
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 256D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.77       670.60       703.37       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.77     2_272.93     2_305.70       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_690.10       289.19     2_979.29       0.0883          1.6472            1.6601         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_690.10       392.07     3_082.17       0.3318          1.1509            1.1429         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_690.10       494.97     3_185.07       0.4735          1.0879            1.0771         1.78
ExhaustiveBinary-256-random (self)                     2_690.10     1_272.64     3_962.74       0.3570          1.1574            1.1517         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_772.93       291.44     3_064.37       0.1109         80.1365            1.7598         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_772.93       414.13     3_187.06       0.3158          1.5913            1.1224         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_772.93       519.03     3_291.97       0.4282          1.3071            1.0760         1.78
ExhaustiveBinary-256-pca (self)                        2_772.93     1_330.75     4_103.68       0.2950          2.2349            1.1670         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_308.70       452.27     5_760.96       0.1384          1.5143            1.5158         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_308.70       563.84     5_872.53       0.4321          1.0993            1.0940         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_308.70       671.33     5_980.03       0.5844          1.0539            1.0489         3.55
ExhaustiveBinary-512-random (self)                     5_308.70     1_969.14     7_277.84       0.4549          1.1058            1.1042         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_375.44       457.56     5_833.00       0.1283          1.7193            1.5407         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_375.44       570.64     5_946.07       0.4093          1.1233            1.0895         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_375.44       673.34     6_048.77       0.5766          1.0656            1.0445         3.55
ExhaustiveBinary-512-pca (self)                        5_375.44     1_864.05     7_239.48       0.4109          1.1443            1.1073         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_494.69       795.79    11_290.48       0.1995          1.3854            1.3901         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_494.69       965.45    11_460.14       0.5516          1.0599            1.0567         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_494.69     1_007.19    11_501.87       0.7089          1.0295            1.0257         7.10
ExhaustiveBinary-1024-random (self)                   10_494.69     2_963.58    13_458.27       0.5787          1.0643            1.0611         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_591.44       792.94    11_384.39       0.1536          1.5190            1.4680         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_591.44       898.57    11_490.02       0.4703          1.0875            1.0721         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_591.44     1_015.09    11_606.54       0.6409          1.0449            1.0342         7.10
ExhaustiveBinary-1024-pca (self)                      10_591.44     2_974.30    13_565.74       0.4638          1.1066            1.0892         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   59.60       508.44       568.04       0.0927          1.6630            1.6696         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    59.60       538.30       597.89       0.3446          1.1453            1.1333         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    59.60       829.81       889.40       0.5011          1.0792            1.0682         1.53
ExhaustiveBinary-256-sign (self)                          59.60     1_643.09     1_702.69       0.3670          1.1539            1.1453         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)            3_644.15       121.68     3_765.83       0.0925          1.6368            1.6516         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           3_644.15       130.46     3_774.61       0.0925          1.6370            1.6516         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           3_644.15       126.50     3_770.65       0.0925          1.6370            1.6516         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           3_644.15       177.57     3_821.72       0.3386          1.1478            1.1396         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           3_644.15       227.70     3_871.86       0.4784          1.0862            1.0759         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          3_644.15       177.41     3_821.57       0.3378          1.1479            1.1396         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          3_644.15       238.36     3_882.51       0.4781          1.0862            1.0759         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          3_644.15       183.28     3_827.43       0.3378          1.1479            1.1396         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          3_644.15       233.80     3_877.96       0.4781          1.0863            1.0759         1.93
IVF-Binary-256-nl158-random (self)                     3_644.15       496.44     4_140.59       0.3625          1.1542            1.1493         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           3_252.81       130.59     3_383.40       0.1059          1.5891            1.5960         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           3_252.81       128.22     3_381.03       0.1059          1.5892            1.5961         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           3_252.81       131.84     3_384.65       0.1059          1.5892            1.5961         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          3_252.81       185.03     3_437.84       0.3701          1.1286            1.1211         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          3_252.81       230.91     3_483.71       0.5088          1.0753            1.0667         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          3_252.81       183.22     3_436.03       0.3700          1.1286            1.1211         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          3_252.81       233.59     3_486.39       0.5087          1.0753            1.0667         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          3_252.81       188.60     3_441.41       0.3700          1.1286            1.1211         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          3_252.81       241.80     3_494.61       0.5087          1.0753            1.0667         2.00
IVF-Binary-256-nl223-random (self)                     3_252.81       509.12     3_761.93       0.3938          1.1339            1.1328         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           3_468.23       132.91     3_601.14       0.1116          1.5721            1.5759         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           3_468.23       142.50     3_610.72       0.1116          1.5725            1.5762         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           3_468.23       149.37     3_617.59       0.1116          1.5725            1.5762         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          3_468.23       203.37     3_671.60       0.3793          1.1238            1.1169         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          3_468.23       260.36     3_728.59       0.5182          1.0725            1.0646         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          3_468.23       209.64     3_677.87       0.3791          1.1239            1.1169         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          3_468.23       266.52     3_734.74       0.5181          1.0725            1.0647         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          3_468.23       199.82     3_668.04       0.3791          1.1239            1.1169         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          3_468.23       243.82     3_712.05       0.5180          1.0725            1.0647         2.09
IVF-Binary-256-nl316-random (self)                     3_468.23       534.61     4_002.83       0.4033          1.1281            1.1281         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               3_697.83       124.33     3_822.16       0.1196          2.2348            1.6349         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              3_697.83       144.32     3_842.15       0.1159          2.7209            1.6456         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              3_697.83       127.86     3_825.69       0.1149          3.1662            1.6488         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              3_697.83       192.99     3_890.82       0.3989          1.1281            1.0946         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              3_697.83       250.17     3_948.00       0.5641          1.0697            1.0479         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             3_697.83       192.36     3_890.19       0.3773          1.1546            1.0996         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             3_697.83       256.68     3_954.51       0.5416          1.0820            1.0512         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             3_697.83       197.10     3_894.94       0.3706          1.1749            1.1008         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             3_697.83       263.43     3_961.26       0.5311          1.0926            1.0523         1.93
IVF-Binary-256-nl158-pca (self)                        3_697.83       576.43     4_274.26       0.3839          1.1735            1.1175         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              3_331.32       132.60     3_463.92       0.1170          2.4465            1.6381         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              3_331.32       130.64     3_461.96       0.1162          2.7282            1.6439         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              3_331.32       144.47     3_475.79       0.1153          3.4563            1.6485         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             3_331.32       199.23     3_530.55       0.3793          1.1454            1.0992         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             3_331.32       261.53     3_592.85       0.5457          1.0767            1.0509         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             3_331.32       199.20     3_530.52       0.3747          1.1574            1.1002         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             3_331.32       257.57     3_588.89       0.5384          1.0831            1.0517         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             3_331.32       202.59     3_533.91       0.3682          1.1785            1.1016         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             3_331.32       262.67     3_593.99       0.5270          1.0946            1.0530         2.00
IVF-Binary-256-nl223-pca (self)                        3_331.32       579.56     3_910.88       0.3821          1.1748            1.1184         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              3_505.87       134.50     3_640.37       0.1168          2.4348            1.6331         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              3_505.87       134.55     3_640.43       0.1163          2.6126            1.6415         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              3_505.87       138.64     3_644.52       0.1153          3.2327            1.6485         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             3_505.87       200.05     3_705.92       0.3788          1.1469            1.0992         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             3_505.87       254.60     3_760.47       0.5451          1.0773            1.0510         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             3_505.87       199.66     3_705.53       0.3765          1.1518            1.0998         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             3_505.87       256.38     3_762.25       0.5413          1.0802            1.0514         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             3_505.87       204.76     3_710.63       0.3699          1.1709            1.1012         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             3_505.87       263.72     3_769.59       0.5302          1.0908            1.0528         2.09
IVF-Binary-256-nl316-pca (self)                        3_505.87       586.16     4_092.03       0.3839          1.1696            1.1178         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_215.69       215.73     6_431.42       0.1407          1.5101            1.5120         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_215.69       218.04     6_433.73       0.1406          1.5102            1.5121         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_215.69       221.05     6_436.74       0.1406          1.5102            1.5121         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_215.69       277.21     6_492.90       0.4352          1.0982            1.0933         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_215.69       329.35     6_545.04       0.5866          1.0534            1.0485         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_215.69       279.16     6_494.85       0.4348          1.0983            1.0933         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_215.69       331.55     6_547.24       0.5864          1.0534            1.0485         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_215.69       283.22     6_498.91       0.4348          1.0983            1.0933         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_215.69       335.16     6_550.85       0.5864          1.0534            1.0485         3.71
IVF-Binary-512-nl158-random (self)                     6_215.69       840.22     7_055.91       0.4573          1.1049            1.1032         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_804.51       220.28     6_024.79       0.1491          1.4885            1.4902         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_804.51       225.30     6_029.81       0.1491          1.4885            1.4903         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_804.51       228.19     6_032.71       0.1491          1.4885            1.4903         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_804.51       280.75     6_085.26       0.4493          1.0924            1.0879         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_804.51       331.78     6_136.29       0.5986          1.0505            1.0455         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_804.51       280.53     6_085.04       0.4493          1.0924            1.0879         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_804.51       332.30     6_136.81       0.5986          1.0505            1.0455         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_804.51       287.27     6_091.78       0.4493          1.0924            1.0879         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_804.51       338.10     6_142.61       0.5986          1.0505            1.0455         3.77
IVF-Binary-512-nl223-random (self)                     5_804.51       847.22     6_651.73       0.4712          1.0994            1.0976         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           6_014.31       226.19     6_240.51       0.1514          1.4814            1.4830         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           6_014.31       228.20     6_242.51       0.1514          1.4815            1.4831         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           6_014.31       234.46     6_248.77       0.1514          1.4815            1.4831         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          6_014.31       283.46     6_297.77       0.4542          1.0908            1.0868         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          6_014.31       334.88     6_349.19       0.6033          1.0495            1.0446         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          6_014.31       284.37     6_298.68       0.4541          1.0908            1.0868         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          6_014.31       362.73     6_377.04       0.6032          1.0495            1.0446         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          6_014.31       359.27     6_373.58       0.4541          1.0908            1.0868         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          6_014.31       395.27     6_409.58       0.6032          1.0495            1.0446         3.86
IVF-Binary-512-nl316-random (self)                     6_014.31       985.59     6_999.90       0.4756          1.0976            1.0958         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               6_425.44       221.45     6_646.89       0.1322          1.6282            1.5374         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              6_425.44       223.57     6_649.02       0.1293          1.6816            1.5402         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              6_425.44       223.84     6_649.28       0.1287          1.7030            1.5403         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              6_425.44       291.99     6_717.43       0.4307          1.1079            1.0845         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              6_425.44       341.52     6_766.96       0.5984          1.0571            1.0416         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             6_425.44       291.31     6_716.75       0.4144          1.1177            1.0882         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             6_425.44       355.11     6_780.55       0.5839          1.0618            1.0435         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             6_425.44       292.18     6_717.62       0.4108          1.1209            1.0892         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             6_425.44       352.88     6_778.32       0.5792          1.0638            1.0442         3.71
IVF-Binary-512-nl158-pca (self)                        6_425.44       882.49     7_307.94       0.4164          1.1371            1.1056         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              5_876.09       222.68     6_098.77       0.1302          1.6704            1.5377         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              5_876.09       225.05     6_101.14       0.1298          1.6865            1.5379         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              5_876.09       230.26     6_106.35       0.1294          1.7053            1.5379         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             5_876.09       293.53     6_169.63       0.4158          1.1153            1.0881         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             5_876.09       347.64     6_223.73       0.5850          1.0604            1.0434         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             5_876.09       289.71     6_165.80       0.4133          1.1179            1.0887         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             5_876.09       347.00     6_223.09       0.5819          1.0620            1.0440         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             5_876.09       299.69     6_175.78       0.4110          1.1205            1.0893         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             5_876.09       355.47     6_231.56       0.5788          1.0636            1.0443         3.77
IVF-Binary-512-nl223-pca (self)                        5_876.09       891.72     6_767.81       0.4152          1.1371            1.1061         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              6_079.20       226.29     6_305.49       0.1303          1.6707            1.5366         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              6_079.20       231.89     6_311.09       0.1300          1.6804            1.5376         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              6_079.20       232.26     6_311.46       0.1296          1.7019            1.5378         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             6_079.20       294.21     6_373.41       0.4150          1.1160            1.0882         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             6_079.20       348.15     6_427.35       0.5846          1.0607            1.0435         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             6_079.20       292.99     6_372.19       0.4138          1.1173            1.0885         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             6_079.20       351.98     6_431.18       0.5829          1.0615            1.0438         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             6_079.20       302.06     6_381.26       0.4111          1.1200            1.0891         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             6_079.20       355.48     6_434.68       0.5794          1.0632            1.0443         3.86
IVF-Binary-512-nl316-pca (self)                        6_079.20       901.28     6_980.48       0.4158          1.1364            1.1060         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_340.22       407.68    11_747.90       0.2007          1.3838            1.3892         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_340.22       413.73    11_753.95       0.2007          1.3838            1.3892         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_340.22       417.74    11_757.96       0.2007          1.3838            1.3892         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_340.22       477.57    11_817.79       0.5525          1.0596            1.0565         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_340.22       524.91    11_865.13       0.7098          1.0293            1.0256         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_340.22       487.09    11_827.31       0.5524          1.0596            1.0565         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_340.22       531.96    11_872.18       0.7098          1.0293            1.0256         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_340.22       491.74    11_831.96       0.5524          1.0596            1.0565         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_340.22       533.00    11_873.22       0.7098          1.0293            1.0256         7.26
IVF-Binary-1024-nl158-random (self)                   11_340.22     1_491.50    12_831.72       0.5796          1.0641            1.0608         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_954.72       421.70    11_376.42       0.2055          1.3748            1.3786         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_954.72       416.37    11_371.09       0.2055          1.3748            1.3786         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_954.72       434.80    11_389.52       0.2055          1.3748            1.3786         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_954.72       484.65    11_439.37       0.5594          1.0579            1.0548         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_954.72       525.36    11_480.08       0.7146          1.0285            1.0249         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_954.72       474.70    11_429.42       0.5594          1.0579            1.0548         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_954.72       530.33    11_485.05       0.7146          1.0285            1.0249         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_954.72       482.78    11_437.50       0.5594          1.0579            1.0548         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_954.72       540.70    11_495.42       0.7146          1.0285            1.0249         7.32
IVF-Binary-1024-nl223-random (self)                   10_954.72     1_497.38    12_452.10       0.5861          1.0623            1.0588         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         11_155.66       422.86    11_578.52       0.2070          1.3717            1.3766         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)         11_155.66       433.01    11_588.67       0.2070          1.3717            1.3766         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)         11_155.66       428.95    11_584.61       0.2070          1.3717            1.3766         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)        11_155.66       481.36    11_637.02       0.5618          1.0573            1.0543         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)        11_155.66       533.27    11_688.92       0.7169          1.0282            1.0245         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)        11_155.66       502.92    11_658.57       0.5618          1.0573            1.0543         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)        11_155.66       560.89    11_716.55       0.7169          1.0282            1.0245         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)        11_155.66       487.88    11_643.54       0.5618          1.0573            1.0543         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)        11_155.66       543.86    11_699.52       0.7169          1.0282            1.0245         7.42
IVF-Binary-1024-nl316-random (self)                   11_155.66     1_505.10    12_660.76       0.5887          1.0616            1.0582         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)             11_393.77       410.30    11_804.07       0.1551          1.5010            1.4641         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            11_393.77       413.69    11_807.46       0.1538          1.5156            1.4673         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            11_393.77       418.66    11_812.44       0.1538          1.5171            1.4676         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            11_393.77       476.25    11_870.02       0.4829          1.0838            1.0700         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            11_393.77       530.47    11_924.24       0.6509          1.0428            1.0330         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           11_393.77       478.03    11_871.80       0.4724          1.0867            1.0717         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           11_393.77       561.96    11_955.73       0.6434          1.0443            1.0339         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           11_393.77       485.55    11_879.32       0.4710          1.0871            1.0720         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           11_393.77       544.49    11_938.27       0.6414          1.0447            1.0341         7.26
IVF-Binary-1024-nl158-pca (self)                      11_393.77     1_519.22    12_913.00       0.4658          1.1058            1.0888         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            10_999.56       414.10    11_413.66       0.1546          1.5129            1.4655         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            10_999.56       419.05    11_418.61       0.1545          1.5147            1.4658         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            10_999.56       428.49    11_428.06       0.1545          1.5156            1.4659         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           10_999.56       486.82    11_486.38       0.4737          1.0859            1.0714         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           10_999.56       546.56    11_546.12       0.6441          1.0439            1.0339         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           10_999.56       490.81    11_490.37       0.4725          1.0864            1.0717         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           10_999.56       542.49    11_542.05       0.6427          1.0443            1.0340         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           10_999.56       491.97    11_491.53       0.4717          1.0866            1.0719         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           10_999.56       551.68    11_551.24       0.6417          1.0445            1.0342         7.32
IVF-Binary-1024-nl223-pca (self)                      10_999.56     1_532.17    12_531.73       0.4656          1.1055            1.0888         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            11_229.85       420.20    11_650.05       0.1549          1.5124            1.4650         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            11_229.85       423.01    11_652.86       0.1548          1.5138            1.4653         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            11_229.85       429.03    11_658.89       0.1547          1.5151            1.4656         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           11_229.85       485.88    11_715.73       0.4732          1.0861            1.0715         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           11_229.85       540.04    11_769.90       0.6439          1.0440            1.0339         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           11_229.85       485.29    11_715.14       0.4726          1.0864            1.0717         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           11_229.85       541.98    11_771.84       0.6432          1.0442            1.0339         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           11_229.85       493.28    11_723.14       0.4716          1.0867            1.0718         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           11_229.85       560.28    11_790.14       0.6419          1.0444            1.0341         7.42
IVF-Binary-1024-nl316-pca (self)                      11_229.85     1_535.73    12_765.59       0.4659          1.1055            1.0889         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)                989.56       264.94     1_254.50       0.1181          8.9404            7.0474         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               989.56       288.19     1_277.75       0.1030         22.8355           10.6900         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               989.56       319.56     1_309.12       0.0901         39.0570           19.2855         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               989.56       296.64     1_286.20       0.8716          1.0202            1.0025         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               989.56       504.38     1_493.94       0.9479          1.0093            1.0000         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              989.56       316.16     1_305.72       0.7958          1.0555            1.0060         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              989.56       548.73     1_538.29       0.9253          1.0139            1.0000         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              989.56       339.98     1_329.54       0.7262          1.1070            1.0121         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              989.56       570.42     1_559.98       0.8927          1.0226            1.0000         1.68
IVF-Binary-256-nl158-sign (self)                         989.56       971.58     1_961.14       0.8248          1.0405            1.0050         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               626.99       300.60       927.60       0.1164         23.6305           14.0136         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               626.99       308.68       935.67       0.1087         33.0540           24.0646         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               626.99       337.56       964.55       0.0924         64.0223           47.8572         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              626.99       325.30       952.30       0.8017          1.0297            1.0074         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              626.99       542.51     1_169.50       0.9323          1.0095            1.0000         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              626.99       335.44       962.44       0.7470          1.0466            1.0125         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              626.99       558.92     1_185.91       0.9102          1.0133            1.0000         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              626.99       363.03       990.03       0.6397          1.1152            1.0288         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              626.99       591.68     1_218.67       0.8492          1.0243            1.0036         1.75
IVF-Binary-256-nl223-sign (self)                         626.99     1_026.05     1_653.04       0.7850          1.0405            1.0103         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               830.88       312.91     1_143.79       0.1424         25.1331           15.9617         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               830.88       329.78     1_160.66       0.1293         29.1378           20.5648         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               830.88       349.74     1_180.62       0.1047         67.1736           64.6710         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              830.88       342.88     1_173.77       0.7897          1.0337            1.0087         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              830.88       558.93     1_389.81       0.9218          1.0106            1.0000         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              830.88       352.66     1_183.54       0.7606          1.0405            1.0115         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              830.88       570.76     1_401.65       0.9097          1.0126            1.0002         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              830.88       395.08     1_225.97       0.6515          1.0819            1.0268         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              830.88       611.88     1_442.76       0.8496          1.0231            1.0038         1.84
IVF-Binary-256-nl316-sign (self)                         830.88     1_067.05     1_897.94       0.7929          1.0389            1.0098         1.84
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 512D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.70     1_669.95     1_737.65       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         67.70     4_980.38     5_048.08       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_863.93       446.93     6_310.86       0.0656          1.4921            1.4988         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_863.93       592.97     6_456.90       0.2689          1.1363            1.1297         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_863.93       690.56     6_554.49       0.3927          1.0839            1.0734         2.03
ExhaustiveBinary-256-random (self)                     5_863.93     1_785.51     7_649.44       0.2893          1.1351            1.1318         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_069.92       436.53     6_506.45       0.1753        309.4819          218.7149         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_069.92       571.76     6_641.68       0.4663          1.2428            1.0429         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_069.92       743.70     6_813.63       0.5849          1.1744            1.0245         2.03
ExhaustiveBinary-256-pca (self)                        6_069.92     1_835.90     7_905.82       0.4694          1.2512            1.0497         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_477.74       692.42    12_170.17       0.1027          1.4104            1.4180         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_477.74       822.58    12_300.33       0.3403          1.0982            1.0949         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_477.74       972.42    12_450.17       0.4712          1.0574            1.0539         4.05
ExhaustiveBinary-512-random (self)                    11_477.74     2_643.71    14_121.46       0.3573          1.1005            1.1032         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_749.37       693.93    12_443.31       0.1672       1095.7764         1104.6002         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_749.37       826.30    12_575.68       0.4213          1.4308            1.0563         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_749.37       968.31    12_717.69       0.5294          1.2408            1.0330         4.05
ExhaustiveBinary-512-pca (self)                       11_749.37     2_712.55    14_461.93       0.4105          1.6976            1.0716         4.05
ExhaustiveBinary-1024-random_no_rr (query)            22_978.01     1_244.99    24_223.00       0.1484          1.3311            1.3388         8.11
ExhaustiveBinary-1024-random-rf10 (query)             22_978.01     1_373.52    24_351.53       0.4138          1.0699            1.0704         8.11
ExhaustiveBinary-1024-random-rf20 (query)             22_978.01     1_675.86    24_653.87       0.5532          1.0397            1.0384         8.11
ExhaustiveBinary-1024-random (self)                   22_978.01     4_636.15    27_614.16       0.4277          1.0760            1.0781         8.11
ExhaustiveBinary-1024-pca_no_rr (query)               23_442.65     1_236.50    24_679.15       0.2384          1.4238            1.2002         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                23_442.65     1_376.34    24_818.99       0.6733          1.0459            1.0172         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                23_442.65     1_728.17    25_170.82       0.8205          1.0198            1.0059         8.11
ExhaustiveBinary-1024-pca (self)                      23_442.65     4_871.69    28_314.34       0.6814          1.0491            1.0189         8.11
ExhaustiveBinary-512-sign_no_rr (query)                  130.80       682.20       813.01       0.1122          1.4085            1.4174         3.05
ExhaustiveBinary-512-sign-rf10 (query)                   130.80       728.60       859.41       0.3436          1.0978            1.0934         3.05
ExhaustiveBinary-512-sign-rf20 (query)                   130.80     1_170.21     1_301.01       0.4809          1.0554            1.0512         3.05
ExhaustiveBinary-512-sign (self)                         130.80     2_343.09     2_473.89       0.3589          1.1017            1.1024         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)            7_312.84       245.34     7_558.18       0.0682          1.4854            1.4968         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           7_312.84       263.31     7_576.15       0.0682          1.4856            1.4970         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           7_312.84       264.78     7_577.62       0.0682          1.4856            1.4970         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           7_312.84       380.11     7_692.95       0.2719          1.1354            1.1295         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           7_312.84       449.37     7_762.21       0.3944          1.0835            1.0732         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          7_312.84       362.73     7_675.57       0.2709          1.1355            1.1295         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          7_312.84       422.94     7_735.78       0.3938          1.0835            1.0733         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          7_312.84       328.53     7_641.37       0.2709          1.1355            1.1295         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          7_312.84       459.45     7_772.29       0.3937          1.0836            1.0733         2.34
IVF-Binary-256-nl158-random (self)                     7_312.84       997.78     8_310.62       0.2912          1.1342            1.1316         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           6_759.96       254.99     7_014.95       0.0816          1.4541            1.4594         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)           6_759.96       256.16     7_016.13       0.0816          1.4542            1.4596         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)           6_759.96       258.51     7_018.47       0.0816          1.4542            1.4596         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)          6_759.96       353.73     7_113.70       0.3014          1.1172            1.1103         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)          6_759.96       411.88     7_171.85       0.4247          1.0722            1.0644         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)          6_759.96       336.04     7_096.01       0.3012          1.1174            1.1103         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)          6_759.96       412.55     7_172.51       0.4240          1.0724            1.0646         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)          6_759.96       368.71     7_128.67       0.3012          1.1174            1.1103         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)          6_759.96       420.50     7_180.47       0.4239          1.0724            1.0646         2.47
IVF-Binary-256-nl223-random (self)                     6_759.96       948.80     7_708.77       0.3207          1.1157            1.1163         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)           7_134.28       262.21     7_396.49       0.0882          1.4389            1.4411         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           7_134.28       267.34     7_401.62       0.0881          1.4390            1.4412         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           7_134.28       267.21     7_401.49       0.0881          1.4391            1.4413         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          7_134.28       353.40     7_487.68       0.3120          1.1115            1.1048         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          7_134.28       419.43     7_553.71       0.4343          1.0692            1.0621         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          7_134.28       340.92     7_475.20       0.3118          1.1116            1.1049         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          7_134.28       442.16     7_576.44       0.4337          1.0693            1.0623         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          7_134.28       346.21     7_480.49       0.3116          1.1117            1.1050         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          7_134.28       453.11     7_587.39       0.4333          1.0694            1.0624         2.65
IVF-Binary-256-nl316-random (self)                     7_134.28     1_002.99     8_137.27       0.3309          1.1095            1.1117         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               7_555.44       251.58     7_807.02       0.1937          2.5726            1.3030         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              7_555.44       258.60     7_814.05       0.1882          3.1866            1.3199         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              7_555.44       253.04     7_808.48       0.1871          3.7048            1.3282         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              7_555.44       353.04     7_908.48       0.5814          1.0746            1.0276         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              7_555.44       434.13     7_989.57       0.7398          1.0348            1.0116         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             7_555.44       360.23     7_915.67       0.5579          1.0958            1.0296         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             7_555.44       436.80     7_992.24       0.7182          1.0418            1.0127         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             7_555.44       351.45     7_906.89       0.5520          1.1115            1.0299         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             7_555.44       461.58     8_017.02       0.7102          1.0476            1.0130         2.34
IVF-Binary-256-nl158-pca (self)                        7_555.44     1_025.79     8_581.23       0.5791          1.0992            1.0307         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              6_948.05       263.01     7_211.06       0.1890          2.8892            1.2973         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              6_948.05       260.61     7_208.66       0.1881          3.4498            1.3186         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              6_948.05       262.42     7_210.47       0.1872          4.3649            1.3437         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             6_948.05       351.61     7_299.66       0.5606          1.0856            1.0296         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             6_948.05       470.96     7_419.01       0.7221          1.0378            1.0127         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             6_948.05       352.69     7_300.74       0.5559          1.0946            1.0299         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             6_948.05       451.84     7_399.89       0.7152          1.0414            1.0130         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             6_948.05       356.44     7_304.49       0.5504          1.1111            1.0303         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             6_948.05       465.97     7_414.02       0.7069          1.0481            1.0133         2.47
IVF-Binary-256-nl223-pca (self)                        6_948.05     1_038.43     7_986.48       0.5768          1.0982            1.0311         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              7_351.60       265.54     7_617.14       0.1889          2.9462            1.3005         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              7_351.60       267.06     7_618.66       0.1885          3.2387            1.3087         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              7_351.60       276.75     7_628.35       0.1875          4.1378            1.3387         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             7_351.60       368.16     7_719.76       0.5596          1.0869            1.0297         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             7_351.60       472.96     7_824.56       0.7207          1.0384            1.0128         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             7_351.60       361.51     7_713.11       0.5574          1.0905            1.0298         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             7_351.60       481.07     7_832.67       0.7175          1.0399            1.0129         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             7_351.60       364.65     7_716.25       0.5514          1.1070            1.0302         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             7_351.60       458.10     7_809.70       0.7091          1.0462            1.0132         2.65
IVF-Binary-256-nl316-pca (self)                        7_351.60     1_089.38     8_440.98       0.5783          1.0941            1.0309         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           13_022.88       448.08    13_470.96       0.1037          1.4086            1.4175         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          13_022.88       471.35    13_494.23       0.1037          1.4086            1.4175         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          13_022.88       453.47    13_476.35       0.1037          1.4086            1.4175         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          13_022.88       525.45    13_548.33       0.3407          1.0981            1.0949         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          13_022.88       633.56    13_656.44       0.4715          1.0573            1.0539         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         13_022.88       524.21    13_547.09       0.3407          1.0981            1.0949         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         13_022.88       607.46    13_630.34       0.4715          1.0573            1.0539         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         13_022.88       556.39    13_579.27       0.3407          1.0981            1.0949         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         13_022.88       607.42    13_630.30       0.4715          1.0573            1.0539         4.36
IVF-Binary-512-nl158-random (self)                    13_022.88     1_631.89    14_654.77       0.3577          1.1003            1.1032         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          12_397.71       455.67    12_853.38       0.1132          1.3888            1.3925         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          12_397.71       453.60    12_851.31       0.1131          1.3891            1.3927         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          12_397.71       484.20    12_881.91       0.1131          1.3891            1.3927         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         12_397.71       531.99    12_929.70       0.3571          1.0910            1.0888         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         12_397.71       606.62    13_004.33       0.4871          1.0533            1.0506         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         12_397.71       555.80    12_953.51       0.3566          1.0912            1.0890         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         12_397.71       665.25    13_062.96       0.4863          1.0535            1.0508         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         12_397.71       543.50    12_941.21       0.3566          1.0912            1.0890         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         12_397.71       619.54    13_017.25       0.4863          1.0535            1.0508         4.49
IVF-Binary-512-nl223-random (self)                    12_397.71     1_642.03    14_039.74       0.3716          1.0945            1.0976         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          12_816.28       464.96    13_281.24       0.1167          1.3816            1.3853         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          12_816.28       465.18    13_281.45       0.1167          1.3818            1.3854         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          12_816.28       489.09    13_305.37       0.1166          1.3819            1.3855         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         12_816.28       544.93    13_361.21       0.3624          1.0887            1.0870         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         12_816.28       621.16    13_437.43       0.4925          1.0522            1.0496         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         12_816.28       610.98    13_427.26       0.3621          1.0888            1.0871         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         12_816.28       684.86    13_501.13       0.4920          1.0524            1.0497         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         12_816.28       588.71    13_404.99       0.3620          1.0889            1.0871         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         12_816.28       710.97    13_527.25       0.4916          1.0524            1.0498         4.67
IVF-Binary-512-nl316-random (self)                    12_816.28     1_723.02    14_539.30       0.3767          1.0925            1.0959         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              13_293.37       470.95    13_764.32       0.2260          3.6107            1.4991         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             13_293.37       455.40    13_748.77       0.2194          5.7358            2.0263         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             13_293.37       473.38    13_766.75       0.2175          8.8217            2.6463         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             13_293.37       547.31    13_840.68       0.6545          1.0666            1.0183         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             13_293.37       633.37    13_926.74       0.8017          1.0301            1.0064         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            13_293.37       569.48    13_862.85       0.6280          1.0924            1.0199         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            13_293.37       643.76    13_937.13       0.7777          1.0388            1.0073         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            13_293.37       556.19    13_849.55       0.6190          1.1147            1.0204         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            13_293.37       631.10    13_924.47       0.7656          1.0484            1.0078         4.36
IVF-Binary-512-nl158-pca (self)                       13_293.37     1_695.13    14_988.50       0.6371          1.0961            1.0221         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             12_582.83       467.96    13_050.79       0.2206          5.2740            1.5975         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             12_582.83       459.44    13_042.27       0.2189          7.4234            2.1418         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             12_582.83       467.10    13_049.94       0.2164         13.2704            3.6901         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            12_582.83       557.18    13_140.01       0.6310          1.0817            1.0198         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            12_582.83       638.40    13_221.24       0.7810          1.0346            1.0074         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            12_582.83       556.68    13_139.52       0.6238          1.0950            1.0203         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            12_582.83       648.47    13_231.30       0.7721          1.0398            1.0077         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            12_582.83       558.78    13_141.61       0.6123          1.1198            1.0211         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            12_582.83       646.67    13_229.50       0.7574          1.0518            1.0082         4.49
IVF-Binary-512-nl223-pca (self)                       12_582.83     1_728.58    14_311.41       0.6323          1.0983            1.0225         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             13_022.21       474.29    13_496.49       0.2202          4.9794            1.5935         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             13_022.21       473.48    13_495.68       0.2193          6.4216            1.9114         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             13_022.21       474.62    13_496.83       0.2167         11.5553            3.2316         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            13_022.21       572.71    13_594.92       0.6305          1.0828            1.0198         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            13_022.21       663.60    13_685.80       0.7803          1.0349            1.0074         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            13_022.21       571.35    13_593.56       0.6268          1.0884            1.0200         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            13_022.21       645.03    13_667.24       0.7758          1.0371            1.0075         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            13_022.21       559.23    13_581.44       0.6155          1.1127            1.0209         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            13_022.21       646.49    13_668.70       0.7615          1.0479            1.0081         4.67
IVF-Binary-512-nl316-pca (self)                       13_022.21     1_758.46    14_780.66       0.6353          1.0920            1.0223         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          24_390.07       865.55    25_255.61       0.1488          1.3307            1.3387         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)         24_390.07       891.62    25_281.68       0.1488          1.3307            1.3387         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)         24_390.07       870.80    25_260.87       0.1488          1.3307            1.3387         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)         24_390.07       948.03    25_338.10       0.4138          1.0699            1.0704         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)         24_390.07     1_024.96    25_415.02       0.5533          1.0396            1.0384         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)        24_390.07       941.48    25_331.55       0.4138          1.0699            1.0704         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)        24_390.07     1_029.15    25_419.22       0.5533          1.0396            1.0384         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)        24_390.07     1_023.64    25_413.71       0.4138          1.0699            1.0704         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)        24_390.07     1_033.63    25_423.69       0.5533          1.0396            1.0384         8.42
IVF-Binary-1024-nl158-random (self)                   24_390.07     3_018.54    27_408.61       0.4278          1.0760            1.0780         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)         23_782.00       866.18    24_648.18       0.1538          1.3218            1.3305         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         23_782.00       886.01    24_668.01       0.1537          1.3219            1.3307         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         23_782.00       883.93    24_665.93       0.1537          1.3219            1.3307         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        23_782.00       953.38    24_735.39       0.4217          1.0675            1.0684         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        23_782.00     1_028.90    24_810.90       0.5606          1.0383            1.0373         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        23_782.00       956.00    24_738.01       0.4213          1.0676            1.0686         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        23_782.00     1_033.51    24_815.51       0.5600          1.0384            1.0374         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        23_782.00       976.90    24_758.90       0.4213          1.0676            1.0686         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        23_782.00     1_050.76    24_832.76       0.5600          1.0384            1.0374         8.54
IVF-Binary-1024-nl223-random (self)                   23_782.00     3_056.10    26_838.10       0.4358          1.0736            1.0757         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         24_188.46       874.92    25_063.37       0.1558          1.3182            1.3259         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)         24_188.46       880.76    25_069.22       0.1557          1.3183            1.3260         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)         24_188.46       895.06    25_083.52       0.1557          1.3184            1.3262         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)        24_188.46       960.65    25_149.11       0.4242          1.0669            1.0676         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)        24_188.46     1_045.38    25_233.83       0.5639          1.0378            1.0367         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)        24_188.46       965.98    25_154.44       0.4240          1.0669            1.0678         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)        24_188.46     1_039.67    25_228.13       0.5636          1.0379            1.0368         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)        24_188.46       975.23    25_163.68       0.4238          1.0670            1.0679         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)        24_188.46     1_061.52    25_249.97       0.5633          1.0380            1.0369         8.73
IVF-Binary-1024-nl316-random (self)                   24_188.46     3_077.57    27_266.03       0.4386          1.0728            1.0748         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)             24_678.12       886.35    25_564.47       0.2432          1.3581            1.1999         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            24_678.12       888.05    25_566.16       0.2392          1.4031            1.2001         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            24_678.12       892.12    25_570.24       0.2386          1.4170            1.2001         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            24_678.12       990.87    25_668.99       0.6894          1.0422            1.0158         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            24_678.12     1_050.74    25_728.86       0.8320          1.0183            1.0052         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           24_678.12       975.77    25_653.89       0.6766          1.0450            1.0169         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           24_678.12     1_046.13    25_724.25       0.8238          1.0193            1.0057         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           24_678.12       958.56    25_636.68       0.6742          1.0455            1.0171         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           24_678.12     1_046.16    25_724.27       0.8215          1.0195            1.0058         8.42
IVF-Binary-1024-nl158-pca (self)                      24_678.12     3_056.33    27_734.44       0.6844          1.0483            1.0186         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            24_053.93       871.10    24_925.03       0.2399          1.3930            1.1995         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            24_053.93       875.65    24_929.58       0.2392          1.4054            1.2000         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            24_053.93       876.48    24_930.41       0.2389          1.4186            1.2000         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           24_053.93       955.46    25_009.39       0.6778          1.0436            1.0169         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           24_053.93     1_039.78    25_093.71       0.8252          1.0185            1.0057         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           24_053.93       956.83    25_010.76       0.6758          1.0445            1.0170         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           24_053.93     1_040.50    25_094.43       0.8234          1.0190            1.0058         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           24_053.93       979.43    25_033.36       0.6741          1.0450            1.0171         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           24_053.93     1_058.81    25_112.74       0.8216          1.0193            1.0059         8.54
IVF-Binary-1024-nl223-pca (self)                      24_053.93     3_082.21    27_136.14       0.6842          1.0477            1.0187         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            24_374.93       872.01    25_246.93       0.2396          1.3949            1.1992         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            24_374.93       875.32    25_250.25       0.2394          1.3998            1.1993         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            24_374.93       881.67    25_256.60       0.2388          1.4160            1.1999         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           24_374.93       971.95    25_346.87       0.6773          1.0438            1.0169         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           24_374.93     1_045.27    25_420.19       0.8249          1.0186            1.0058         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           24_374.93       970.98    25_345.90       0.6766          1.0441            1.0169         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           24_374.93     1_044.05    25_418.97       0.8243          1.0188            1.0058         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           24_374.93       967.71    25_342.63       0.6744          1.0449            1.0171         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           24_374.93     1_061.06    25_435.98       0.8221          1.0192            1.0058         8.73
IVF-Binary-1024-nl316-pca (self)                      24_374.93     3_186.76    27_561.69       0.6847          1.0474            1.0187         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)              1_686.51       425.24     2_111.75       0.0796         12.0508           10.1033         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)             1_686.51       484.88     2_171.39       0.0771         17.7820           10.7946         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)             1_686.51       511.81     2_198.32       0.0767         20.0653           11.0347         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)             1_686.51       524.74     2_211.25       0.8031          1.0346            1.0054         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)             1_686.51       834.40     2_520.90       0.9295          1.0088            1.0000         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)            1_686.51       515.11     2_201.62       0.6753          1.3610            1.0128         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)            1_686.51       846.33     2_532.84       0.9067          1.0120            1.0000         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)            1_686.51       501.45     2_187.96       0.5847          1.4920            1.0298         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)            1_686.51       886.32     2_572.83       0.8736          1.0176            1.0013         3.36
IVF-Binary-512-nl158-sign (self)                       1_686.51     1_387.64     3_074.14       0.7091          1.2592            1.0118         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)             1_060.96       452.61     1_513.57       0.0999         20.0265           15.8728         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)             1_060.96       476.39     1_537.35       0.0807         33.3106           22.1997         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)             1_060.96       509.00     1_569.96       0.0726         66.1032           42.6949         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)            1_060.96       486.89     1_547.85       0.6638          1.1579            1.0171         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)            1_060.96       822.70     1_883.66       0.9054          1.0097            1.0004         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)            1_060.96       565.29     1_626.25       0.5689          1.3836            1.0409         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)            1_060.96       858.92     1_919.88       0.8783          1.0129            1.0015         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)            1_060.96       565.51     1_626.47       0.4446          2.2376            1.4453         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)            1_060.96       949.93     2_010.89       0.8040          1.0215            1.0051         3.49
IVF-Binary-512-nl223-sign (self)                       1_060.96     1_565.46     2_626.42       0.6163          1.1785            1.0310         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)             1_524.21       476.87     2_001.08       0.0942         20.0074           13.7648         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)             1_524.21       479.65     2_003.86       0.0897         29.5793           15.6739         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)             1_524.21       535.41     2_059.62       0.0746         49.6884           31.4256         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)            1_524.21       514.12     2_038.33       0.6848          1.0538            1.0141         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)            1_524.21       861.01     2_385.23       0.8963          1.0116            1.0006         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)            1_524.21       530.92     2_055.13       0.6344          1.1266            1.0213         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)            1_524.21       877.91     2_402.12       0.8807          1.0134            1.0013         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)            1_524.21       570.73     2_094.94       0.5206          1.4766            1.0521         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)            1_524.21       924.25     2_448.47       0.8242          1.0213            1.0039         3.67
IVF-Binary-512-nl316-sign (self)                       1_524.21     1_601.18     3_125.40       0.6741          1.0530            1.0184         3.67
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 768D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       101.15     1_802.86     1_904.00       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.15     5_896.66     5_997.81       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)              9_151.92       538.81     9_690.73       0.0560          1.3883            1.3893         2.28
ExhaustiveBinary-256-random-rf10 (query)               9_151.92       691.20     9_843.12       0.2348          1.1233            1.1184         2.28
ExhaustiveBinary-256-random-rf20 (query)               9_151.92       819.15     9_971.07       0.3419          1.0805            1.0657         2.28
ExhaustiveBinary-256-random (self)                     9_151.92     2_226.45    11_378.37       0.2472          1.1217            1.1092         2.28
ExhaustiveBinary-256-pca_no_rr (query)                 9_473.56       550.63    10_024.19       0.1693        237.6734           31.9251         2.28
ExhaustiveBinary-256-pca-rf10 (query)                  9_473.56       756.11    10_229.67       0.4634          1.1896            1.0299         2.28
ExhaustiveBinary-256-pca-rf20 (query)                  9_473.56       870.75    10_344.30       0.5917          1.1224            1.0168         2.28
ExhaustiveBinary-256-pca (self)                        9_473.56     2_288.59    11_762.15       0.4804          1.1913            1.0308         2.28
ExhaustiveBinary-512-random_no_rr (query)             17_927.93       917.26    18_845.19       0.0796          1.3400            1.3521         4.55
ExhaustiveBinary-512-random-rf10 (query)              17_927.93     1_068.38    18_996.31       0.2824          1.0956            1.0876         4.55
ExhaustiveBinary-512-random-rf20 (query)              17_927.93     1_228.51    19_156.44       0.3933          1.0588            1.0526         4.55
ExhaustiveBinary-512-random (self)                    17_927.93     3_439.39    21_367.32       0.2970          1.0893            1.0898         4.55
ExhaustiveBinary-512-pca_no_rr (query)                18_019.07       930.02    18_949.10       0.1882       1500.0959         1408.1786         4.55
ExhaustiveBinary-512-pca-rf10 (query)                 18_019.07     1_077.33    19_096.40       0.4651          1.2548            1.0301         4.55
ExhaustiveBinary-512-pca-rf20 (query)                 18_019.07     1_245.73    19_264.80       0.5757          1.1920            1.0173         4.55
ExhaustiveBinary-512-pca (self)                       18_019.07     3_520.13    21_539.21       0.4710          1.2567            1.0323         4.55
ExhaustiveBinary-1024-random_no_rr (query)            35_582.96     1_673.64    37_256.60       0.1221          1.2753            1.2758         9.11
ExhaustiveBinary-1024-random-rf10 (query)             35_582.96     1_851.08    37_434.04       0.3372          1.0685            1.0692         9.11
ExhaustiveBinary-1024-random-rf20 (query)             35_582.96     1_986.64    37_569.60       0.4564          1.0414            1.0411         9.11
ExhaustiveBinary-1024-random (self)                   35_582.96     5_959.74    41_542.70       0.3447          1.0714            1.0746         9.11
ExhaustiveBinary-1024-pca_no_rr (query)               35_697.18     1_744.35    37_441.54       0.2681          1.8439            1.1207         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                35_697.18     1_824.76    37_521.94       0.7069          1.0551            1.0084         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                35_697.18     2_009.23    37_706.41       0.8317          1.0248            1.0025         9.11
ExhaustiveBinary-1024-pca (self)                      35_697.18     6_367.55    42_064.73       0.7148          1.0572            1.0087         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  270.80       981.39     1_252.19       0.1171          1.2871            1.2900         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   270.80     1_080.85     1_351.64       0.3258          1.0737            1.0725         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   270.80     1_704.58     1_975.37       0.4465          1.0436            1.0424         4.58
ExhaustiveBinary-768-sign (self)                         270.80     3_209.86     3_480.66       0.3335          1.0755            1.0780         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)           11_251.75       363.02    11_614.78       0.0586          1.3825            1.3881         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)          11_251.75       365.01    11_616.76       0.0585          1.3826            1.3881         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)          11_251.75       369.51    11_621.27       0.0585          1.3826            1.3881         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)          11_251.75       449.86    11_701.62       0.2381          1.1220            1.1178         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)          11_251.75       542.59    11_794.35       0.3443          1.0797            1.0656         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)         11_251.75       449.43    11_701.18       0.2374          1.1221            1.1178         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)         11_251.75       546.39    11_798.14       0.3438          1.0797            1.0656         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)         11_251.75       450.58    11_702.34       0.2374          1.1221            1.1178         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)         11_251.75       559.30    11_811.05       0.3438          1.0797            1.0656         2.74
IVF-Binary-256-nl158-random (self)                    11_251.75     1_296.54    12_548.29       0.2499          1.1204            1.1089         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)          10_302.59       386.95    10_689.55       0.0684          1.3633            1.3685         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)          10_302.59       373.85    10_676.45       0.0684          1.3635            1.3685         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)          10_302.59       376.73    10_679.33       0.0684          1.3635            1.3685         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)         10_302.59       465.15    10_767.75       0.2654          1.1042            1.0935         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)         10_302.59       562.75    10_865.34       0.3732          1.0669            1.0566         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)         10_302.59       464.15    10_766.74       0.2653          1.1042            1.0935         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)         10_302.59       567.12    10_869.72       0.3731          1.0669            1.0566         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)         10_302.59       469.25    10_771.84       0.2653          1.1042            1.0935         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)         10_302.59       572.50    10_875.09       0.3731          1.0669            1.0566         2.93
IVF-Binary-256-nl223-random (self)                    10_302.59     1_337.48    11_640.08       0.2793          1.0999            1.0945         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)          10_852.97       389.04    11_242.01       0.0754          1.3475            1.3498         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)          10_852.97       387.35    11_240.32       0.0754          1.3476            1.3498         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)          10_852.97       388.51    11_241.48       0.0754          1.3478            1.3499         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)         10_852.97       496.85    11_349.82       0.2804          1.0950            1.0860         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)         10_852.97       574.53    11_427.50       0.3885          1.0613            1.0528         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)         10_852.97       482.19    11_335.16       0.2802          1.0950            1.0860         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)         10_852.97       577.46    11_430.43       0.3883          1.0613            1.0529         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)         10_852.97       484.64    11_337.61       0.2801          1.0951            1.0860         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)         10_852.97       608.85    11_461.82       0.3882          1.0613            1.0529         3.21
IVF-Binary-256-nl316-random (self)                    10_852.97     1_392.43    12_245.40       0.2938          1.0897            1.0887         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)              11_581.57       371.24    11_952.81       0.1804          2.4698            1.1927         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)             11_581.57       367.91    11_949.49       0.1754          2.9727            1.1948         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)             11_581.57       370.61    11_952.19       0.1745          3.1815            1.1953         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)             11_581.57       483.54    12_065.11       0.5384          1.0691            1.0238         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)             11_581.57       584.00    12_165.57       0.6973          1.0352            1.0110         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)            11_581.57       478.82    12_060.39       0.5174          1.0846            1.0249         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)            11_581.57       586.59    12_168.16       0.6786          1.0396            1.0118         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)            11_581.57       480.90    12_062.47       0.5129          1.0937            1.0251         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)            11_581.57       588.92    12_170.49       0.6722          1.0428            1.0119         2.74
IVF-Binary-256-nl158-pca (self)                       11_581.57     1_429.91    13_011.49       0.5382          1.0860            1.0250         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)             10_628.63       392.09    11_020.71       0.1767          2.6320            1.1905         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)             10_628.63       376.55    11_005.18       0.1755          2.9612            1.1941         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)             10_628.63       379.46    11_008.09       0.1748          3.3432            1.1958         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)            10_628.63       484.72    11_113.35       0.5229          1.0734            1.0248         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)            10_628.63       589.60    11_218.23       0.6851          1.0350            1.0116         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)            10_628.63       485.72    11_114.34       0.5180          1.0795            1.0251         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)            10_628.63       597.84    11_226.47       0.6788          1.0374            1.0118         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)            10_628.63       486.64    11_115.26       0.5135          1.0881            1.0252         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)            10_628.63       592.60    11_221.23       0.6722          1.0407            1.0119         2.93
IVF-Binary-256-nl223-pca (self)                       10_628.63     1_445.56    12_074.18       0.5385          1.0805            1.0252         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)             11_192.92       386.57    11_579.49       0.1766          2.6847            1.1900         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)             11_192.92       389.78    11_582.70       0.1761          2.8714            1.1913         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)             11_192.92       390.60    11_583.52       0.1752          3.2819            1.1959         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)            11_192.92       495.59    11_688.51       0.5213          1.0746            1.0249         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)            11_192.92       634.41    11_827.33       0.6831          1.0357            1.0117         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)            11_192.92       494.90    11_687.82       0.5189          1.0772            1.0251         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)            11_192.92       607.45    11_800.37       0.6802          1.0367            1.0118         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)            11_192.92       497.03    11_689.95       0.5141          1.0869            1.0252         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)            11_192.92       636.30    11_829.22       0.6729          1.0404            1.0119         3.21
IVF-Binary-256-nl316-pca (self)                       11_192.92     1_482.56    12_675.48       0.5394          1.0784            1.0251         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)           19_923.96       673.14    20_597.11       0.0808          1.3381            1.3517         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)          19_923.96       674.54    20_598.50       0.0808          1.3381            1.3517         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)          19_923.96       686.05    20_610.02       0.0808          1.3381            1.3517         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)          19_923.96       768.99    20_692.95       0.2831          1.0954            1.0875         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)          19_923.96       856.11    20_780.07       0.3938          1.0587            1.0526         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)         19_923.96       768.96    20_692.93       0.2830          1.0954            1.0875         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)         19_923.96       863.02    20_786.98       0.3937          1.0587            1.0526         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)         19_923.96       778.00    20_701.97       0.2830          1.0954            1.0875         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)         19_923.96       861.52    20_785.49       0.3937          1.0587            1.0526         5.02
IVF-Binary-512-nl158-random (self)                    19_923.96     2_365.78    22_289.75       0.2976          1.0891            1.0898         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)          18_991.74       682.66    19_674.40       0.0907          1.3213            1.3277         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)          18_991.74       683.88    19_675.63       0.0907          1.3213            1.3277         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)          18_991.74       689.69    19_681.43       0.0907          1.3213            1.3277         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)         18_991.74       778.14    19_769.88       0.2982          1.0874            1.0823         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)         18_991.74       876.92    19_868.66       0.4081          1.0543            1.0499         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)         18_991.74       778.53    19_770.27       0.2982          1.0874            1.0823         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)         18_991.74       879.75    19_871.49       0.4081          1.0543            1.0499         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)         18_991.74       782.49    19_774.23       0.2982          1.0874            1.0823         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)         18_991.74       874.14    19_865.88       0.4081          1.0543            1.0499         5.21
IVF-Binary-512-nl223-random (self)                    18_991.74     2_396.71    21_388.45       0.3107          1.0827            1.0855         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)          19_590.80       703.77    20_294.57       0.0961          1.3108            1.3132         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)          19_590.80       693.59    20_284.39       0.0961          1.3108            1.3133         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)          19_590.80       694.50    20_285.30       0.0961          1.3108            1.3133         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)         19_590.80       790.31    20_381.11       0.3052          1.0839            1.0792         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)         19_590.80       890.62    20_481.42       0.4151          1.0523            1.0483         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)         19_590.80       795.96    20_386.76       0.3051          1.0839            1.0792         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)         19_590.80       883.16    20_473.96       0.4149          1.0523            1.0483         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)         19_590.80       792.06    20_382.86       0.3051          1.0839            1.0792         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)         19_590.80       889.61    20_480.41       0.4148          1.0523            1.0484         5.48
IVF-Binary-512-nl316-random (self)                    19_590.80     2_720.55    22_311.35       0.3169          1.0798            1.0832         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)              20_500.07       689.22    21_189.29       0.2344          4.0218            1.2269         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)             20_500.07       691.17    21_191.24       0.2283          6.5456            1.4339         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)             20_500.07       680.09    21_180.16       0.2269          8.6596            1.6356         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)             20_500.07       784.12    21_284.19       0.6553          1.0603            1.0127         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)             20_500.07       878.97    21_379.04       0.7965          1.0290            1.0046         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)            20_500.07       781.32    21_281.40       0.6309          1.0809            1.0136         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)            20_500.07       895.87    21_395.94       0.7752          1.0352            1.0052         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)            20_500.07       784.02    21_284.09       0.6241          1.0965            1.0138         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)            20_500.07       899.00    21_399.07       0.7663          1.0412            1.0054         5.02
IVF-Binary-512-nl158-pca (self)                       20_500.07     2_459.93    22_960.00       0.6458          1.0828            1.0137         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)             19_292.70       689.08    19_981.78       0.2299          5.1521            1.2355         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)             19_292.70       686.25    19_978.95       0.2284          6.9168            1.4194         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)             19_292.70       687.37    19_980.07       0.2268         10.6677            1.8782         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)            19_292.70       797.38    20_090.09       0.6368          1.0674            1.0134         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)            19_292.70       892.24    20_184.94       0.7834          1.0302            1.0051         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)            19_292.70       812.26    20_104.97       0.6299          1.0769            1.0137         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)            19_292.70       902.89    20_195.59       0.7747          1.0336            1.0053         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)            19_292.70       797.83    20_090.54       0.6217          1.0942            1.0139         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)            19_292.70       899.63    20_192.33       0.7635          1.0405            1.0055         5.21
IVF-Binary-512-nl223-pca (self)                       19_292.70     2_521.77    21_814.47       0.6446          1.0783            1.0139         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)             19_896.19       788.63    20_684.82       0.2293          5.2151            1.2225         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)             19_896.19       715.20    20_611.38       0.2286          6.2822            1.3100         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)             19_896.19       716.82    20_613.01       0.2268          9.8368            1.7677         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)            19_896.19       860.35    20_756.54       0.6348          1.0690            1.0135         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)            19_896.19     1_007.03    20_903.22       0.7808          1.0307            1.0051         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)            19_896.19       836.28    20_732.47       0.6317          1.0729            1.0136         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)            19_896.19       971.77    20_867.96       0.7766          1.0321            1.0052         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)            19_896.19       832.86    20_729.04       0.6231          1.0907            1.0139         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)            19_896.19       940.69    20_836.88       0.7655          1.0390            1.0054         5.48
IVF-Binary-512-nl316-pca (self)                       19_896.19     2_594.24    22_490.42       0.6463          1.0743            1.0138         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)          37_977.27     1_333.36    39_310.63       0.1225          1.2749            1.2758         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)         37_977.27     1_334.43    39_311.70       0.1225          1.2749            1.2758         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)         37_977.27     1_345.15    39_322.42       0.1225          1.2749            1.2758         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)         37_977.27     1_466.20    39_443.47       0.3374          1.0684            1.0692         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)         37_977.27     1_527.50    39_504.77       0.4566          1.0413            1.0411         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)        37_977.27     1_435.45    39_412.72       0.3374          1.0685            1.0692         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)        37_977.27     1_524.38    39_501.65       0.4566          1.0413            1.0411         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)        37_977.27     1_432.99    39_410.26       0.3374          1.0685            1.0692         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)        37_977.27     1_542.13    39_519.40       0.4566          1.0413            1.0411         9.57
IVF-Binary-1024-nl158-random (self)                   37_977.27     4_581.46    42_558.73       0.3449          1.0714            1.0746         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)         37_040.50     1_357.04    38_397.54       0.1270          1.2675            1.2689         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)         37_040.50     1_358.45    38_398.95       0.1270          1.2675            1.2689         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)         37_040.50     1_364.56    38_405.06       0.1270          1.2675            1.2689         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)        37_040.50     1_494.09    38_534.60       0.3440          1.0663            1.0676         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)        37_040.50     1_557.29    38_597.79       0.4630          1.0400            1.0402         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)        37_040.50     1_451.95    38_492.46       0.3440          1.0663            1.0676         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)        37_040.50     1_541.62    38_582.12       0.4629          1.0400            1.0403         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)        37_040.50     1_435.95    38_476.45       0.3440          1.0663            1.0676         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)        37_040.50     1_558.12    38_598.62       0.4629          1.0400            1.0403         9.76
IVF-Binary-1024-nl223-random (self)                   37_040.50     4_591.10    41_631.60       0.3511          1.0694            1.0729         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)         37_597.00     1_331.33    38_928.34       0.1292          1.2641            1.2662        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)         37_597.00     1_318.23    38_915.24       0.1291          1.2641            1.2662        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)         37_597.00     1_319.57    38_916.58       0.1291          1.2641            1.2662        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)        37_597.00     1_430.36    39_027.36       0.3474          1.0652            1.0667        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)        37_597.00     1_547.95    39_144.96       0.4662          1.0394            1.0396        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)        37_597.00     1_490.32    39_087.32       0.3473          1.0652            1.0667        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)        37_597.00     1_579.13    39_176.13       0.4660          1.0395            1.0397        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)        37_597.00     1_425.85    39_022.85       0.3473          1.0652            1.0667        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)        37_597.00     1_512.19    39_109.19       0.4659          1.0395            1.0397        10.04
IVF-Binary-1024-nl316-random (self)                   37_597.00     4_534.23    42_131.24       0.3541          1.0686            1.0720        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)             37_661.92     1_294.93    38_956.85       0.2750          1.6270            1.1205         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)            37_661.92     1_292.69    38_954.61       0.2697          1.7363            1.1207         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)            37_661.92     1_301.59    38_963.51       0.2687          1.7745            1.1207         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)            37_661.92     1_396.98    39_058.90       0.7304          1.0438            1.0078         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)            37_661.92     1_497.47    39_159.39       0.8523          1.0203            1.0021         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)           37_661.92     1_399.53    39_061.45       0.7134          1.0492            1.0083         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)           37_661.92     1_543.60    39_205.52       0.8403          1.0219            1.0024         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)           37_661.92     1_413.46    39_075.38       0.7099          1.0511            1.0084         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)           37_661.92     1_512.96    39_174.88       0.8361          1.0227            1.0024         9.57
IVF-Binary-1024-nl158-pca (self)                      37_661.92     4_583.05    42_244.97       0.7213          1.0509            1.0087         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)            36_708.88     1_298.63    38_007.51       0.2707          1.6902            1.1200         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)            36_708.88     1_302.38    38_011.26       0.2697          1.7395            1.1208         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)            36_708.88     1_311.44    38_020.33       0.2689          1.7755            1.1208         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)           36_708.88     1_409.24    38_118.12       0.7170          1.0460            1.0083         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)           36_708.88     1_502.44    38_211.32       0.8449          1.0205            1.0024         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)           36_708.88     1_407.16    38_116.04       0.7130          1.0482            1.0084         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)           36_708.88     1_514.29    38_223.17       0.8402          1.0214            1.0024         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)           36_708.88     1_420.44    38_129.32       0.7099          1.0503            1.0084         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)           36_708.88     1_516.20    38_225.09       0.8365          1.0223            1.0025         9.76
IVF-Binary-1024-nl223-pca (self)                      36_708.88     4_600.02    41_308.90       0.7209          1.0497            1.0087         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)            37_336.72     1_310.38    38_647.10       0.2705          1.6860            1.1196        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)            37_336.72     1_315.46    38_652.18       0.2700          1.7156            1.1198        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)            37_336.72     1_318.59    38_655.32       0.2690          1.7717            1.1206        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)           37_336.72     1_421.30    38_758.02       0.7154          1.0466            1.0084        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)           37_336.72     1_521.25    38_857.97       0.8434          1.0206            1.0024        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)           37_336.72     1_419.46    38_756.19       0.7139          1.0476            1.0084        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)           37_336.72     1_522.84    38_859.57       0.8414          1.0211            1.0024        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)           37_336.72     1_423.89    38_760.61       0.7103          1.0500            1.0084        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)           37_336.72     1_528.69    38_865.42       0.8370          1.0221            1.0025        10.04
IVF-Binary-1024-nl316-pca (self)                      37_336.72     4_594.11    41_930.83       0.7221          1.0490            1.0087        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)              2_444.90       569.67     3_014.56       0.0781         16.7559           14.3851         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)             2_444.90       607.13     3_052.03       0.0770         18.7743           15.0229         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)             2_444.90       684.82     3_129.71       0.0768         21.4159           15.6548         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)             2_444.90       627.11     3_072.01       0.7106          1.1278            1.0089         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)             2_444.90     1_095.76     3_540.65       0.9017          1.0072            1.0002         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)            2_444.90       664.92     3_109.82       0.5239          3.5417            1.0202         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)            2_444.90     1_130.71     3_575.61       0.8635          1.0107            1.0014         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)            2_444.90       706.41     3_151.30       0.4113          4.8456            1.3984         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)            2_444.90     1_172.16     3_617.06       0.8197          1.0153            1.0029         5.04
IVF-Binary-768-nl158-sign (self)                       2_444.90     1_951.83     4_396.73       0.5474          3.5670            1.0194         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)             1_502.48       606.66     2_109.14       0.0862         49.5242           19.8729         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)             1_502.48       637.02     2_139.50       0.0739         58.1859           31.1503         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)             1_502.48       691.12     2_193.60       0.0726         74.5487           42.5126         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)            1_502.48       656.11     2_158.59       0.5502          2.1281            1.0199         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)            1_502.48     1_133.64     2_636.12       0.8658          1.0102            1.0013         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)            1_502.48       680.51     2_182.99       0.4427          2.8890            1.0679         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)            1_502.48     1_146.37     2_648.85       0.8333          1.0132            1.0025         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)            1_502.48       739.12     2_241.60       0.3235          3.9461            1.6294         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)            1_502.48     1_214.09     2_716.57       0.7485          1.0213            1.0062         5.23
IVF-Binary-768-nl223-sign (self)                       1_502.48     2_170.26     3_672.74       0.4734          2.7249            1.0529         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)             2_113.15       644.99     2_758.14       0.0846         42.9395           18.9362         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)             2_113.15       683.36     2_796.51       0.0839         55.0976           21.5085         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)             2_113.15       725.43     2_838.58       0.0715         89.4095           23.1457         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)            2_113.15       701.57     2_814.73       0.5454          1.5844            1.0241         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)            2_113.15     1_182.85     3_296.00       0.8546          1.0119            1.0017         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)            2_113.15       719.52     2_832.67       0.4891          1.9253            1.0393         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)            2_113.15     1_187.62     3_300.78       0.8352          1.0136            1.0024         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)            2_113.15       775.84     2_888.99       0.3684          2.5474            1.5637         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)            2_113.15     1_256.29     3_369.44       0.7643          1.0210            1.0055         5.51
IVF-Binary-768-nl316-sign (self)                       2_113.15     2_156.13     4_269.28       0.5195          1.7909            1.0353         5.51
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Cell embeddings

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 256D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.49       689.10       721.59       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.49     2_206.76     2_239.25       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_670.64       290.18     2_960.82       0.5519          1.8827            1.5884         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_670.64       408.98     3_079.62       0.9881          1.0022            1.0000         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_670.64       526.39     3_197.03       0.9980          1.0003            1.0000         1.78
ExhaustiveBinary-256-random (self)                     2_670.64     1_318.39     3_989.03       0.9881          1.0022            1.0000         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_771.49       296.44     3_067.93       0.1183         14.1069           10.4668         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_771.49       398.81     3_170.30       0.3215          1.9397            1.8015         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_771.49       499.32     3_270.81       0.4181          1.5883            1.4969         1.78
ExhaustiveBinary-256-pca (self)                        2_771.49     1_295.58     4_067.07       0.3192          1.9523            1.8114         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_221.42       456.97     5_678.39       0.6305          1.5768            1.3636         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_221.42       571.09     5_792.51       0.9975          1.0004            1.0000         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_221.42       692.02     5_913.44       0.9998          1.0000            1.0000         3.55
ExhaustiveBinary-512-random (self)                     5_221.42     1_871.97     7_093.39       0.9972          1.0004            1.0000         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_328.89       461.56     5_790.45       0.3665          2.7738            2.2420         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_328.89       568.70     5_897.59       0.8453          1.0657            1.0329         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_328.89       678.93     6_007.81       0.9396          1.0206            1.0000         3.55
ExhaustiveBinary-512-pca (self)                        5_328.89     1_858.12     7_187.01       0.8325          1.0737            1.0387         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_367.79       786.77    11_154.56       0.6758          1.4452            1.2803         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_367.79       907.63    11_275.41       0.9995          1.0001            1.0000         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_367.79     1_024.46    11_392.25       0.9999          1.0000            1.0000         7.10
ExhaustiveBinary-1024-random (self)                   10_367.79     2_992.36    13_360.15       0.9993          1.0001            1.0000         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_456.99       806.08    11_263.07       0.5577          1.7682            1.4986         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_456.99       908.54    11_365.53       0.9880          1.0028            1.0000         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_456.99     1_077.72    11_534.71       0.9987          1.0003            1.0000         7.10
ExhaustiveBinary-1024-pca (self)                      10_456.99     3_182.53    13_639.52       0.9861          1.0033            1.0000         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   59.82       476.32       536.15       0.0376         19.4742           14.8778         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    59.82       498.31       558.14       0.1617          2.7567            2.6547         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    59.82       804.32       864.15       0.2739          1.9837            1.9248         1.53
ExhaustiveBinary-256-sign (self)                          59.82     1_625.37     1_685.20       0.1691          2.7353            2.6299         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)            3_675.86       128.56     3_804.43       0.5644          1.6744            1.5171         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           3_675.86       138.10     3_813.96       0.5580          1.7409            1.5527         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           3_675.86       150.72     3_826.58       0.5562          1.7768            1.5630         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           3_675.86       194.63     3_870.50       0.9901          1.0018            1.0000         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           3_675.86       268.72     3_944.59       0.9967          1.0006            1.0000         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          3_675.86       203.49     3_879.36       0.9903          1.0017            1.0000         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          3_675.86       265.57     3_941.43       0.9985          1.0002            1.0000         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          3_675.86       212.70     3_888.57       0.9894          1.0019            1.0000         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          3_675.86       270.62     3_946.48       0.9983          1.0002            1.0000         1.93
IVF-Binary-256-nl158-random (self)                     3_675.86       604.53     4_280.39       0.9902          1.0017            1.0000         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           3_081.24       130.87     3_212.11       0.5623          1.6901            1.5271         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           3_081.24       135.77     3_217.01       0.5598          1.7203            1.5426         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           3_081.24       142.28     3_223.53       0.5573          1.7614            1.5583         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          3_081.24       195.25     3_276.49       0.9910          1.0015            1.0000         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          3_081.24       249.65     3_330.89       0.9983          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          3_081.24       198.21     3_279.45       0.9906          1.0016            1.0000         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          3_081.24       250.43     3_331.67       0.9986          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          3_081.24       205.35     3_286.60       0.9896          1.0018            1.0000         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          3_081.24       259.95     3_341.19       0.9984          1.0002            1.0000         2.00
IVF-Binary-256-nl223-random (self)                     3_081.24       573.23     3_654.47       0.9904          1.0017            1.0000         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           3_248.68       139.37     3_388.05       0.5622          1.6843            1.5282         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           3_248.68       141.15     3_389.84       0.5610          1.6983            1.5370         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           3_248.68       146.38     3_395.06       0.5582          1.7399            1.5536         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          3_248.68       200.51     3_449.19       0.9911          1.0015            1.0000         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          3_248.68       261.97     3_510.66       0.9986          1.0002            1.0000         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          3_248.68       201.51     3_450.19       0.9908          1.0016            1.0000         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          3_248.68       289.40     3_538.09       0.9986          1.0002            1.0000         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          3_248.68       216.32     3_465.00       0.9899          1.0018            1.0000         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          3_248.68       267.49     3_516.18       0.9985          1.0002            1.0000         2.09
IVF-Binary-256-nl316-random (self)                     3_248.68       605.88     3_854.57       0.9908          1.0016            1.0000         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               4_022.75       135.88     4_158.63       0.1527          5.1206            4.6838         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              4_022.75       169.60     4_192.34       0.1376          6.0959            5.4915         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              4_022.75       150.98     4_173.72       0.1312          6.8304            6.1460         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              4_022.75       206.66     4_229.41       0.4905          1.4114            1.3555         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              4_022.75       304.88     4_327.62       0.6392          1.2170            1.1784         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             4_022.75       246.63     4_269.38       0.4274          1.5364            1.4682         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             4_022.75       302.07     4_324.82       0.5675          1.2973            1.2498         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             4_022.75       235.24     4_257.99       0.3950          1.6191            1.5471         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             4_022.75       299.12     4_321.86       0.5273          1.3518            1.2985         1.93
IVF-Binary-256-nl158-pca (self)                        4_022.75       661.18     4_683.93       0.4248          1.5444            1.4781         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              3_534.58       154.90     3_689.48       0.1485          5.1295            4.7161         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              3_534.58       138.74     3_673.32       0.1425          5.5191            5.0362         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              3_534.58       154.77     3_689.35       0.1343          6.2989            5.7114         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             3_534.58       219.96     3_754.54       0.4802          1.4241            1.3692         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             3_534.58       274.05     3_808.63       0.6314          1.2222            1.1847         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             3_534.58       212.71     3_747.29       0.4522          1.4777            1.4213         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             3_534.58       281.04     3_815.62       0.6001          1.2555            1.2161         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             3_534.58       224.11     3_758.69       0.4119          1.5715            1.5054         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             3_534.58       294.73     3_829.31       0.5507          1.3172            1.2708         2.00
IVF-Binary-256-nl223-pca (self)                        3_534.58       653.22     4_187.80       0.4498          1.4839            1.4265         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              3_513.77       144.78     3_658.55       0.1484          5.0403            4.6575         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              3_513.77       142.67     3_656.44       0.1450          5.2295            4.8212         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              3_513.77       149.12     3_662.89       0.1368          5.9003            5.4040         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             3_513.77       218.13     3_731.90       0.4799          1.4228            1.3692         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             3_513.77       269.61     3_783.38       0.6328          1.2200            1.1835         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             3_513.77       209.24     3_723.01       0.4654          1.4492            1.3942         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             3_513.77       271.06     3_784.83       0.6163          1.2367            1.1985         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             3_513.77       228.30     3_742.07       0.4256          1.5342            1.4731         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             3_513.77       287.49     3_801.26       0.5689          1.2913            1.2486         2.09
IVF-Binary-256-nl316-pca (self)                        3_513.77       640.36     4_154.13       0.4629          1.4549            1.4001         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_274.95       228.16     6_503.11       0.6403          1.4493            1.3344         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_274.95       240.91     6_515.86       0.6346          1.4947            1.3510         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_274.95       255.75     6_530.70       0.6329          1.5212            1.3568         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_274.95       296.78     6_571.73       0.9963          1.0007            1.0000         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_274.95       352.99     6_627.94       0.9976          1.0005            1.0000         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_274.95       309.74     6_584.69       0.9980          1.0003            1.0000         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_274.95       370.07     6_645.02       0.9998          1.0000            1.0000         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_274.95       328.44     6_603.40       0.9978          1.0003            1.0000         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_274.95       382.44     6_657.39       0.9998          1.0000            1.0000         3.71
IVF-Binary-512-nl158-random (self)                     6_274.95       954.46     7_229.41       0.9978          1.0003            1.0000         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_692.66       229.16     5_921.83       0.6379          1.4649            1.3384         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_692.66       234.67     5_927.33       0.6357          1.4848            1.3474         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_692.66       247.30     5_939.97       0.6336          1.5131            1.3537         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_692.66       299.66     5_992.32       0.9976          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_692.66       350.51     6_043.17       0.9993          1.0001            1.0000         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_692.66       301.96     5_994.62       0.9979          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_692.66       360.92     6_053.58       0.9997          1.0000            1.0000         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_692.66       317.34     6_010.00       0.9978          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_692.66       402.21     6_094.87       0.9998          1.0000            1.0000         3.77
IVF-Binary-512-nl223-random (self)                     5_692.66       933.59     6_626.25       0.9977          1.0003            1.0000         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           5_835.68       234.36     6_070.04       0.6377          1.4619            1.3395         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           5_835.68       236.48     6_072.15       0.6368          1.4710            1.3426         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           5_835.68       245.55     6_081.23       0.6344          1.4973            1.3510         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          5_835.68       298.31     6_133.99       0.9979          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          5_835.68       356.87     6_192.55       0.9996          1.0001            1.0000         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          5_835.68       299.97     6_135.65       0.9980          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          5_835.68       355.63     6_191.31       0.9997          1.0000            1.0000         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          5_835.68       316.09     6_151.76       0.9978          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          5_835.68       367.56     6_203.24       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-random (self)                     5_835.68       928.63     6_764.31       0.9978          1.0003            1.0000         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               6_390.29       231.64     6_621.93       0.3819          2.2246            2.0447         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              6_390.29       246.27     6_636.56       0.3733          2.3651            2.1310         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              6_390.29       256.20     6_646.50       0.3705          2.4596            2.1717         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              6_390.29       314.02     6_704.31       0.8865          1.0420            1.0188         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              6_390.29       356.65     6_746.94       0.9653          1.0104            1.0000         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             6_390.29       315.02     6_705.31       0.8656          1.0534            1.0258         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             6_390.29       377.32     6_767.61       0.9542          1.0147            1.0000         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             6_390.29       326.65     6_716.94       0.8565          1.0588            1.0289         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             6_390.29       391.45     6_781.74       0.9480          1.0171            1.0000         3.71
IVF-Binary-512-nl158-pca (self)                        6_390.29       972.38     7_362.67       0.8541          1.0600            1.0304         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              5_802.52       230.26     6_032.78       0.3781          2.2518            2.0704         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              5_802.52       236.39     6_038.91       0.3748          2.3105            2.1054         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              5_802.52       251.40     6_053.92       0.3714          2.4105            2.1533         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             5_802.52       299.59     6_102.11       0.8810          1.0451            1.0204         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             5_802.52       362.55     6_165.07       0.9631          1.0112            1.0000         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             5_802.52       305.11     6_107.62       0.8711          1.0504            1.0239         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             5_802.52       364.06     6_166.58       0.9578          1.0132            1.0000         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             5_802.52       314.73     6_117.25       0.8591          1.0574            1.0280         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             5_802.52       375.60     6_178.12       0.9499          1.0164            1.0000         3.77
IVF-Binary-512-nl223-pca (self)                        5_802.52       953.86     6_756.38       0.8602          1.0566            1.0282         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              5_979.94       235.87     6_215.81       0.3779          2.2494            2.0657         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              5_979.94       240.42     6_220.36       0.3763          2.2755            2.0823         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              5_979.94       244.63     6_224.57       0.3724          2.3637            2.1355         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             5_979.94       301.78     6_281.72       0.8796          1.0459            1.0207         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             5_979.94       356.68     6_336.61       0.9626          1.0116            1.0000         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             5_979.94       301.24     6_281.17       0.8749          1.0483            1.0223         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             5_979.94       360.98     6_340.92       0.9602          1.0125            1.0000         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             5_979.94       310.95     6_290.89       0.8626          1.0551            1.0266         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             5_979.94       373.53     6_353.47       0.9524          1.0154            1.0000         3.86
IVF-Binary-512-nl316-pca (self)                        5_979.94       924.12     6_904.06       0.8641          1.0545            1.0266         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_435.74       425.73    11_861.47       0.6843          1.3554            1.2564         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_435.74       447.48    11_883.21       0.6791          1.3904            1.2727         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_435.74       464.47    11_900.21       0.6775          1.4122            1.2763         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_435.74       496.17    11_931.90       0.9974          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_435.74       552.98    11_988.71       0.9976          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_435.74       530.44    11_966.17       0.9996          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_435.74       577.68    12_013.42       0.9999          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_435.74       535.81    11_971.55       0.9996          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_435.74       599.44    12_035.18       0.9999          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-random (self)                   11_435.74     1_671.46    13_107.19       0.9994          1.0001            1.0000         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_821.73       429.71    11_251.43       0.6818          1.3682            1.2621         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_821.73       441.16    11_262.89       0.6799          1.3836            1.2679         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_821.73       460.36    11_282.08       0.6781          1.4037            1.2740         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_821.73       494.08    11_315.81       0.9990          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_821.73       574.98    11_396.71       0.9993          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_821.73       499.75    11_321.48       0.9995          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_821.73       562.33    11_384.06       0.9998          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_821.73       516.51    11_338.24       0.9995          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_821.73       580.01    11_401.74       0.9999          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-random (self)                   10_821.73     1_576.32    12_398.05       0.9993          1.0001            1.0000         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         10_972.63       436.38    11_409.01       0.6814          1.3673            1.2632         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)         10_972.63       440.18    11_412.81       0.6805          1.3746            1.2655         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)         10_972.63       452.08    11_424.70       0.6785          1.3952            1.2721         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)        10_972.63       496.83    11_469.46       0.9993          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)        10_972.63       560.98    11_533.60       0.9997          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)        10_972.63       498.32    11_470.94       0.9994          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)        10_972.63       557.43    11_530.05       0.9998          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)        10_972.63       518.00    11_490.63       0.9995          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)        10_972.63       575.11    11_547.73       0.9999          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-random (self)                   10_972.63     1_575.41    12_548.04       0.9993          1.0001            1.0000         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)             11_547.79       429.89    11_977.68       0.5686          1.5695            1.4513         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            11_547.79       446.80    11_994.59       0.5623          1.6313            1.4783         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            11_547.79       466.33    12_014.12       0.5605          1.6705            1.4883         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            11_547.79       496.79    12_044.58       0.9912          1.0019            1.0000         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            11_547.79       557.47    12_105.26       0.9972          1.0006            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           11_547.79       514.46    12_062.25       0.9907          1.0020            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           11_547.79       582.09    12_129.88       0.9992          1.0002            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           11_547.79       536.73    12_084.52       0.9895          1.0023            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           11_547.79       605.74    12_153.53       0.9990          1.0002            1.0000         7.26
IVF-Binary-1024-nl158-pca (self)                      11_547.79     1_655.42    13_203.21       0.9889          1.0025            1.0000         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            10_925.76       452.57    11_378.33       0.5654          1.5890            1.4586         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            10_925.76       435.63    11_361.39       0.5631          1.6138            1.4706         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            10_925.76       454.17    11_379.93       0.5608          1.6523            1.4825         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           10_925.76       499.10    11_424.86       0.9916          1.0017            1.0000         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           10_925.76       551.45    11_477.21       0.9987          1.0002            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           10_925.76       500.78    11_426.54       0.9909          1.0019            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           10_925.76       562.44    11_488.20       0.9990          1.0002            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           10_925.76       550.61    11_476.37       0.9897          1.0023            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           10_925.76       583.22    11_508.99       0.9990          1.0002            1.0000         7.32
IVF-Binary-1024-nl223-pca (self)                      10_925.76     1_612.31    12_538.07       0.9892          1.0024            1.0000         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            11_069.52       442.39    11_511.91       0.5650          1.5883            1.4586         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            11_069.52       442.92    11_512.44       0.5640          1.5992            1.4649         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            11_069.52       449.57    11_519.09       0.5616          1.6346            1.4791         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           11_069.52       502.49    11_572.01       0.9917          1.0017            1.0000         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           11_069.52       558.60    11_628.12       0.9990          1.0002            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           11_069.52       500.88    11_570.40       0.9914          1.0018            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           11_069.52       569.11    11_638.63       0.9991          1.0002            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           11_069.52       515.33    11_584.85       0.9902          1.0022            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           11_069.52       581.66    11_651.17       0.9991          1.0002            1.0000         7.42
IVF-Binary-1024-nl316-pca (self)                      11_069.52     1_569.54    12_639.06       0.9896          1.0023            1.0000         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)              1_096.10       297.43     1_393.53       0.3698          2.2923            2.0232         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)             1_096.10       332.81     1_428.91       0.3462          2.4709            2.1626         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)             1_096.10       357.60     1_453.70       0.3312          2.6495            2.2473         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)             1_096.10       335.48     1_431.58       0.7370          1.1542            1.0836         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)             1_096.10       586.04     1_682.14       0.9127          1.0437            1.0000         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)            1_096.10       369.65     1_465.75       0.6111          1.2797            1.1979         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)            1_096.10       637.63     1_733.73       0.8375          1.0872            1.0259         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)            1_096.10       393.32     1_489.42       0.5503          1.3843            1.2645         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)            1_096.10       662.31     1_758.40       0.7858          1.1267            1.0514         1.68
IVF-Binary-256-nl158-sign (self)                       1_096.10     1_137.64     2_233.74       0.6110          1.2817            1.1969         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               497.62       307.71       805.33       0.3263          2.5881            2.1617         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               497.62       326.99       824.61       0.3166          2.6847            2.2373         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               497.62       364.18       861.81       0.2971          2.9458            2.3783         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              497.62       348.46       846.08       0.6563          1.2510            1.1328         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              497.62       588.10     1_085.72       0.8243          1.1194            1.0197         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              497.62       377.10       874.73       0.6037          1.3094            1.1855         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              497.62       616.41     1_114.03       0.7917          1.1453            1.0344         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              497.62       395.45       893.07       0.5241          1.4384            1.2862         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              497.62       658.87     1_156.49       0.7306          1.2030            1.0694         1.75
IVF-Binary-256-nl223-sign (self)                         497.62     1_110.10     1_607.72       0.6051          1.3104            1.1855         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               660.34       323.44       983.78       0.2930          2.8391            2.3380         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               660.34       332.58       992.93       0.2880          2.9065            2.3795         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               660.34       369.62     1_029.96       0.2711          3.1733            2.5248         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              660.34       360.99     1_021.33       0.6133          1.3217            1.1663         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              660.34       605.54     1_265.88       0.7590          1.1822            1.0464         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              660.34       377.63     1_037.97       0.5879          1.3579            1.1935         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              660.34       625.51     1_285.85       0.7428          1.1998            1.0547         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              660.34       409.81     1_070.16       0.5132          1.4900            1.2890         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              660.34       680.23     1_340.57       0.6920          1.2620            1.0877         1.84
IVF-Binary-256-nl316-sign (self)                         660.34     1_132.11     1_792.46       0.5883          1.3553            1.1938         1.84
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 512D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.82     1_234.15     1_301.97       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         67.82     4_019.59     4_087.41       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_845.68       430.83     6_276.51       0.5547          1.7646            1.5364         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_845.68       561.91     6_407.59       0.9898          1.0017            1.0000         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_845.68       705.70     6_551.38       0.9984          1.0002            1.0000         2.03
ExhaustiveBinary-256-random (self)                     5_845.68     1_829.88     7_675.55       0.9899          1.0016            1.0000         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_092.60       429.67     6_522.27       0.1212         13.1779           10.1573         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_092.60       562.33     6_654.93       0.3406          1.8751            1.7379         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_092.60       686.48     6_779.08       0.4406          1.5429            1.4489         2.03
ExhaustiveBinary-256-pca (self)                        6_092.60     1_877.84     7_970.44       0.3366          1.8907            1.7569         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_568.84       685.00    12_253.84       0.6013          1.6760            1.4608         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_568.84       822.16    12_391.01       0.9977          1.0003            1.0000         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_568.84       959.03    12_527.87       0.9997          1.0000            1.0000         4.05
ExhaustiveBinary-512-random (self)                    11_568.84     2_664.16    14_233.00       0.9975          1.0003            1.0000         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_639.85       686.46    12_326.31       0.1147         15.9331           12.0721         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_639.85       808.82    12_448.67       0.2782          2.2254            2.0213         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_639.85       934.51    12_574.36       0.3475          1.8252            1.6935         4.05
ExhaustiveBinary-512-pca (self)                       11_639.85     2_770.77    14_410.62       0.2742          2.2528            2.0453         4.05
ExhaustiveBinary-1024-random_no_rr (query)            22_724.83     1_232.11    23_956.93       0.6624          1.4553            1.3048         8.11
ExhaustiveBinary-1024-random-rf10 (query)             22_724.83     1_380.73    24_105.55       0.9994          1.0001            1.0000         8.11
ExhaustiveBinary-1024-random-rf20 (query)             22_724.83     1_527.97    24_252.79       0.9999          1.0000            1.0000         8.11
ExhaustiveBinary-1024-random (self)                   22_724.83     4_519.00    27_243.82       0.9994          1.0001            1.0000         8.11
ExhaustiveBinary-1024-pca_no_rr (query)               23_098.53     1_323.31    24_421.84       0.3939          2.4382            2.0579         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                23_098.53     1_552.22    24_650.75       0.8322          1.0743            1.0411         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                23_098.53     1_516.93    24_615.46       0.9198          1.0285            1.0052         8.11
ExhaustiveBinary-1024-pca (self)                      23_098.53     4_505.61    27_604.15       0.8160          1.0854            1.0492         8.11
ExhaustiveBinary-512-sign_no_rr (query)                  132.42       663.60       796.02       0.0400         18.1509           13.6734         3.05
ExhaustiveBinary-512-sign-rf10 (query)                   132.42       708.03       840.45       0.1821          2.5572            2.4621         3.05
ExhaustiveBinary-512-sign-rf20 (query)                   132.42     1_135.81     1_268.24       0.3140          1.8428            1.7786         3.05
ExhaustiveBinary-512-sign (self)                         132.42     2_270.68     2_403.10       0.1897          2.5286            2.4283         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)            7_702.20       252.76     7_954.97       0.5635          1.6351            1.4890         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           7_702.20       264.23     7_966.44       0.5601          1.6665            1.5056         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           7_702.20       272.29     7_974.49       0.5586          1.6897            1.5137         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           7_702.20       339.37     8_041.57       0.9918          1.0013            1.0000         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           7_702.20       419.63     8_121.83       0.9978          1.0003            1.0000         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          7_702.20       343.78     8_045.98       0.9916          1.0013            1.0000         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          7_702.20       425.48     8_127.69       0.9988          1.0001            1.0000         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          7_702.20       350.95     8_053.16       0.9909          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          7_702.20       441.50     8_143.70       0.9987          1.0001            1.0000         2.34
IVF-Binary-256-nl158-random (self)                     7_702.20       992.60     8_694.80       0.9918          1.0012            1.0000         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           6_517.80       258.41     6_776.21       0.5616          1.6428            1.4934         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)           6_517.80       258.31     6_776.11       0.5604          1.6550            1.5016         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)           6_517.80       267.74     6_785.54       0.5589          1.6783            1.5113         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)          6_517.80       340.55     6_858.35       0.9922          1.0012            1.0000         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)          6_517.80       418.10     6_935.90       0.9989          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)          6_517.80       339.11     6_856.91       0.9919          1.0012            1.0000         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)          6_517.80       423.31     6_941.11       0.9989          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)          6_517.80       347.71     6_865.52       0.9911          1.0014            1.0000         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)          6_517.80       432.29     6_950.09       0.9988          1.0001            1.0000         2.47
IVF-Binary-256-nl223-random (self)                     6_517.80       973.72     7_491.53       0.9920          1.0012            1.0000         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)           6_756.64       264.83     7_021.47       0.5613          1.6459            1.4960         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           6_756.64       266.59     7_023.23       0.5607          1.6528            1.5007         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           6_756.64       272.10     7_028.74       0.5595          1.6692            1.5100         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          6_756.64       346.94     7_103.58       0.9922          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          6_756.64       426.26     7_182.90       0.9990          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          6_756.64       355.86     7_112.50       0.9919          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          6_756.64       443.61     7_200.25       0.9990          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          6_756.64       351.85     7_108.49       0.9913          1.0014            1.0000         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          6_756.64       433.57     7_190.20       0.9989          1.0001            1.0000         2.65
IVF-Binary-256-nl316-random (self)                     6_756.64       979.00     7_735.63       0.9922          1.0012            1.0000         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               7_948.74       252.63     8_201.37       0.1449          5.7762            5.1731         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              7_948.74       258.33     8_207.07       0.1357          6.6787            5.8453         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              7_948.74       269.84     8_218.58       0.1312          7.3557            6.3951         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              7_948.74       334.50     8_283.24       0.4661          1.4656            1.3972         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              7_948.74       419.80     8_368.54       0.6170          1.2408            1.1914         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             7_948.74       346.03     8_294.77       0.4251          1.5539            1.4789         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             7_948.74       435.53     8_384.27       0.5648          1.3027            1.2483         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             7_948.74       353.87     8_302.61       0.4036          1.6097            1.5333         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             7_948.74       454.30     8_403.04       0.5372          1.3413            1.2866         2.34
IVF-Binary-256-nl158-pca (self)                        7_948.74     1_025.53     8_974.27       0.4223          1.5629            1.4870         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              6_796.37       260.58     7_056.95       0.1421          5.7900            5.2134         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              6_796.37       277.75     7_074.12       0.1380          6.1428            5.5220         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              6_796.37       269.83     7_066.20       0.1324          6.8902            6.1135         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             6_796.37       344.92     7_141.29       0.4570          1.4773            1.4107         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             6_796.37       448.54     7_244.91       0.6071          1.2480            1.2022         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             6_796.37       352.67     7_149.04       0.4391          1.5146            1.4483         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             6_796.37       436.78     7_233.15       0.5852          1.2733            1.2262         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             6_796.37       358.03     7_154.40       0.4120          1.5813            1.5114         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             6_796.37       448.76     7_245.13       0.5505          1.3188            1.2679         2.47
IVF-Binary-256-nl223-pca (self)                        6_796.37     1_065.28     7_861.65       0.4362          1.5225            1.4536         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              7_035.15       264.91     7_300.07       0.1410          5.7370            5.2155         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              7_035.15       266.66     7_301.82       0.1390          5.8998            5.3439         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              7_035.15       270.65     7_305.81       0.1341          6.4951            5.8558         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             7_035.15       351.58     7_386.73       0.4555          1.4768            1.4133         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             7_035.15       437.28     7_472.44       0.6070          1.2468            1.2014         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             7_035.15       356.32     7_391.47       0.4462          1.4963            1.4325         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             7_035.15       435.21     7_470.37       0.5954          1.2599            1.2157         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             7_035.15       361.58     7_396.73       0.4208          1.5556            1.4872         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             7_035.15       447.78     7_482.94       0.5625          1.3010            1.2520         2.65
IVF-Binary-256-nl316-pca (self)                        7_035.15     1_021.44     8_056.59       0.4438          1.5029            1.4356         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           13_381.35       454.80    13_836.15       0.6099          1.5613            1.4188         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          13_381.35       476.21    13_857.56       0.6060          1.5937            1.4384         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          13_381.35       479.45    13_860.80       0.6043          1.6153            1.4467         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          13_381.35       538.03    13_919.38       0.9973          1.0004            1.0000         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          13_381.35       641.88    14_023.23       0.9985          1.0003            1.0000         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         13_381.35       550.24    13_931.59       0.9982          1.0002            1.0000         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         13_381.35       634.80    14_016.14       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         13_381.35       569.43    13_950.78       0.9981          1.0002            1.0000         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         13_381.35       648.51    14_029.86       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-random (self)                    13_381.35     1_697.69    15_079.04       0.9982          1.0002            1.0000         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          12_188.13       459.91    12_648.04       0.6073          1.5772            1.4297         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          12_188.13       476.26    12_664.39       0.6060          1.5877            1.4373         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          12_188.13       475.34    12_663.47       0.6042          1.6107            1.4479         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         12_188.13       540.38    12_728.51       0.9982          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         12_188.13       615.25    12_803.38       0.9996          1.0001            1.0000         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         12_188.13       539.69    12_727.82       0.9983          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         12_188.13       624.91    12_813.04       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         12_188.13       553.60    12_741.73       0.9980          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         12_188.13       660.18    12_848.31       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-random (self)                    12_188.13     1_661.98    13_850.11       0.9981          1.0002            1.0000         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          12_466.97       469.86    12_936.83       0.6071          1.5794            1.4300         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          12_466.97       484.25    12_951.22       0.6063          1.5867            1.4335         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          12_466.97       478.12    12_945.09       0.6048          1.6036            1.4412         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         12_466.97       546.75    13_013.71       0.9983          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         12_466.97       629.22    13_096.19       0.9997          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         12_466.97       545.37    13_012.34       0.9983          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         12_466.97       628.58    13_095.54       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         12_466.97       556.97    13_023.94       0.9981          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         12_466.97       642.92    13_109.88       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-random (self)                    12_466.97     1_669.75    14_136.72       0.9982          1.0002            1.0000         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              13_663.33       462.41    14_125.75       0.1402          6.0802            5.4372         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             13_663.33       470.69    14_134.03       0.1307          7.1069            6.2067         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             13_663.33       523.93    14_187.27       0.1259          7.9042            6.8252         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             13_663.33       543.11    14_206.44       0.4249          1.5448            1.4724         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             13_663.33       629.15    14_292.48       0.5632          1.3027            1.2514         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            13_663.33       565.80    14_229.14       0.3819          1.6569            1.5761         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            13_663.33       646.14    14_309.48       0.5036          1.3868            1.3297         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            13_663.33       578.61    14_241.94       0.3597          1.7319            1.6499         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            13_663.33       666.87    14_330.20       0.4726          1.4420            1.3820         4.36
IVF-Binary-512-nl158-pca (self)                       13_663.33     1_739.75    15_403.08       0.3781          1.6708            1.5912         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             12_504.91       472.22    12_977.14       0.1373          6.0883            5.4841         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             12_504.91       486.30    12_991.22       0.1333          6.4764            5.7742         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             12_504.91       495.40    13_000.31       0.1274          7.3349            6.4962         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            12_504.91       568.53    13_073.45       0.4163          1.5550            1.4908         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            12_504.91       686.14    13_191.05       0.5535          1.3089            1.2639         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            12_504.91       627.55    13_132.46       0.3974          1.6030            1.5342         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            12_504.91       712.01    13_216.92       0.5275          1.3449            1.2960         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            12_504.91       619.38    13_124.29       0.3699          1.6875            1.6156         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            12_504.91       713.51    13_218.42       0.4885          1.4075            1.3570         4.49
IVF-Binary-512-nl223-pca (self)                       12_504.91     1_887.10    14_392.01       0.3937          1.6147            1.5457         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             12_898.71       508.00    13_406.72       0.1362          6.0280            5.4414         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             12_898.71       519.09    13_417.80       0.1342          6.2163            5.5994         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             12_898.71       528.97    13_427.69       0.1289          6.8940            6.1917         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            12_898.71       622.31    13_521.02       0.4152          1.5547            1.4922         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            12_898.71       712.27    13_610.98       0.5522          1.3079            1.2642         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            12_898.71       597.64    13_496.35       0.4053          1.5792            1.5141         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            12_898.71       669.19    13_567.91       0.5384          1.3264            1.2812         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            12_898.71       565.19    13_463.90       0.3791          1.6548            1.5812         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            12_898.71       715.89    13_614.60       0.5016          1.3833            1.3353         4.67
IVF-Binary-512-nl316-pca (self)                       12_898.71     1_742.33    14_641.05       0.4012          1.5900            1.5243         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          24_807.88       899.16    25_707.04       0.6689          1.3867            1.2840         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)         24_807.88       907.51    25_715.38       0.6656          1.4072            1.2936         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)         24_807.88       909.96    25_717.84       0.6641          1.4214            1.2989         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)         24_807.88       952.47    25_760.34       0.9983          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)         24_807.88     1_030.46    25_838.34       0.9985          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)        24_807.88       973.27    25_781.14       0.9995          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)        24_807.88     1_052.41    25_860.29       0.9999          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)        24_807.88     1_009.99    25_817.87       0.9995          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)        24_807.88     1_076.04    25_883.91       0.9999          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-random (self)                   24_807.88     3_108.94    27_916.82       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)         23_577.98       878.78    24_456.75       0.6669          1.3969            1.2900         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         23_577.98       883.06    24_461.04       0.6657          1.4046            1.2935         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         23_577.98       906.74    24_484.71       0.6644          1.4194            1.2991         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        23_577.98       974.20    24_552.18       0.9994          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        23_577.98     1_041.90    24_619.88       0.9997          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        23_577.98       960.23    24_538.20       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        23_577.98     1_049.51    24_627.49       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        23_577.98     1_015.36    24_593.34       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        23_577.98     1_054.95    24_632.92       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-random (self)                   23_577.98     3_032.25    26_610.23       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         23_724.11       884.15    24_608.26       0.6663          1.4016            1.2907         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)         23_724.11       944.06    24_668.17       0.6658          1.4059            1.2929         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)         23_724.11       911.53    24_635.64       0.6646          1.4173            1.2981         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)        23_724.11       968.76    24_692.87       0.9995          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)        23_724.11     1_055.28    24_779.39       0.9998          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)        23_724.11       967.30    24_691.41       0.9995          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)        23_724.11     1_062.43    24_786.54       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)        23_724.11       977.25    24_701.36       0.9996          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)        23_724.11     1_081.44    24_805.55       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-random (self)                   23_724.11     3_043.38    26_767.49       0.9995          1.0001            1.0000         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)             24_899.28       864.77    25_764.05       0.4023          2.1616            1.9511         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            24_899.28       891.55    25_790.83       0.3979          2.2360            1.9958         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            24_899.28       908.47    25_807.74       0.3962          2.2803            2.0172         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            24_899.28       945.01    25_844.28       0.8540          1.0604            1.0318         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            24_899.28     1_039.14    25_938.42       0.9426          1.0186            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           24_899.28       982.12    25_881.39       0.8431          1.0671            1.0365         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           24_899.28     1_049.86    25_949.13       0.9316          1.0231            1.0011         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           24_899.28       989.86    25_889.14       0.8385          1.0701            1.0382         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           24_899.28     1_084.68    25_983.96       0.9264          1.0254            1.0030         8.42
IVF-Binary-1024-nl158-pca (self)                      24_899.28     3_107.47    28_006.75       0.8277          1.0772            1.0439         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            23_838.27       875.89    24_714.16       0.3995          2.1870            1.9698         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            23_838.27       894.83    24_733.10       0.3982          2.2111            1.9882         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            23_838.27       894.67    24_732.94       0.3964          2.2676            2.0165         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           23_838.27       951.37    24_789.64       0.8498          1.0631            1.0335         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           23_838.27     1_029.28    24_867.55       0.9391          1.0200            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           23_838.27       955.88    24_794.14       0.8455          1.0657            1.0354         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           23_838.27     1_037.49    24_875.76       0.9342          1.0220            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           23_838.27       973.08    24_811.35       0.8394          1.0696            1.0380         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           23_838.27     1_065.51    24_903.77       0.9272          1.0251            1.0024         8.54
IVF-Binary-1024-nl223-pca (self)                      23_838.27     3_047.40    26_885.67       0.8300          1.0757            1.0429         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            23_988.41       882.66    24_871.07       0.3992          2.1899            1.9743         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            23_988.41       924.22    24_912.63       0.3984          2.2034            1.9834         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            23_988.41       977.01    24_965.42       0.3969          2.2462            2.0083         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           23_988.41       985.20    24_973.60       0.8494          1.0634            1.0338         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           23_988.41     1_069.85    25_058.26       0.9381          1.0205            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           23_988.41       984.08    24_972.49       0.8471          1.0648            1.0349         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           23_988.41     1_073.29    25_061.70       0.9355          1.0215            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           23_988.41     1_006.75    24_995.16       0.8409          1.0685            1.0373         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           23_988.41     1_091.93    25_080.33       0.9293          1.0242            1.0017         8.73
IVF-Binary-1024-nl316-pca (self)                      23_988.41     3_121.34    27_109.75       0.8315          1.0747            1.0423         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)              2_042.45       417.79     2_460.23       0.1510          4.5058            3.8746         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)             2_042.45       468.72     2_511.17       0.1361          5.0356            4.2653         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)             2_042.45       520.91     2_563.36       0.1277          5.5367            4.5857         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)             2_042.45       488.52     2_530.96       0.4704          1.6381            1.3330         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)             2_042.45       878.99     2_921.44       0.5942          1.4484            1.1576         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)            2_042.45       533.62     2_576.07       0.3942          1.8935            1.4818         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)            2_042.45       926.95     2_969.40       0.5140          1.6480            1.2489         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)            2_042.45       574.61     2_617.06       0.3569          2.0547            1.5908         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)            2_042.45     1_003.50     3_045.95       0.4697          1.7895            1.3254         3.36
IVF-Binary-512-nl158-sign (self)                       2_042.45     1_625.39     3_667.83       0.3955          1.8825            1.4821         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               852.36       470.06     1_322.42       0.1213          4.6896            4.1250         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               852.36       493.56     1_345.92       0.1174          4.9249            4.2951         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               852.36       541.48     1_393.84       0.1106          5.4835            4.6867         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              852.36       532.85     1_385.21       0.4197          1.7170            1.4180         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              852.36       928.75     1_781.11       0.5241          1.5150            1.2411         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              852.36       559.52     1_411.88       0.3911          1.7989            1.4901         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              852.36       975.80     1_828.16       0.4903          1.5872            1.2903         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              852.36       596.20     1_448.57       0.3459          1.9626            1.6202         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              852.36     1_018.95     1_871.31       0.4403          1.7076            1.3831         3.49
IVF-Binary-512-nl223-sign (self)                         852.36     1_619.22     2_471.58       0.3897          1.7981            1.4918         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)             1_093.11       498.84     1_591.95       0.1151          4.6038            4.0973         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)             1_093.11       511.96     1_605.07       0.1136          4.7084            4.1602         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)             1_093.11       561.55     1_654.66       0.1106          5.1122            4.4320         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)            1_093.11       559.21     1_652.31       0.4126          1.7028            1.4384         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)            1_093.11       943.48     2_036.59       0.5130          1.5028            1.2665         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)            1_093.11       560.09     1_653.20       0.3986          1.7407            1.4735         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)            1_093.11       955.18     2_048.29       0.4955          1.5385            1.2897         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)            1_093.11       605.29     1_698.40       0.3574          1.8706            1.5859         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)            1_093.11     1_014.86     2_107.97       0.4496          1.6513            1.3709         3.67
IVF-Binary-512-nl316-sign (self)                       1_093.11     1_632.83     2_725.93       0.3990          1.7352            1.4743         3.67
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 768D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        99.20     1_790.88     1_890.08       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.20     5_833.97     5_933.17       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)              9_026.41       539.68     9_566.09       0.5361          1.8069            1.5908         2.28
ExhaustiveBinary-256-random-rf10 (query)               9_026.41       705.20     9_731.62       0.9868          1.0022            1.0000         2.28
ExhaustiveBinary-256-random-rf20 (query)               9_026.41       910.61     9_937.02       0.9980          1.0003            1.0000         2.28
ExhaustiveBinary-256-random (self)                     9_026.41     2_217.19    11_243.61       0.9875          1.0021            1.0000         2.28
ExhaustiveBinary-256-pca_no_rr (query)                 9_362.68       541.29     9_903.97       0.1281         12.2485            9.2701         2.28
ExhaustiveBinary-256-pca-rf10 (query)                  9_362.68       691.76    10_054.44       0.3750          1.7664            1.6288         2.28
ExhaustiveBinary-256-pca-rf20 (query)                  9_362.68       832.94    10_195.62       0.5003          1.4421            1.3341         2.28
ExhaustiveBinary-256-pca (self)                        9_362.68     2_194.21    11_556.89       0.3725          1.7770            1.6344         2.28
ExhaustiveBinary-512-random_no_rr (query)             17_631.84       899.25    18_531.09       0.5866          1.6777            1.4945         4.55
ExhaustiveBinary-512-random-rf10 (query)              17_631.84     1_052.96    18_684.80       0.9966          1.0005            1.0000         4.55
ExhaustiveBinary-512-random-rf20 (query)              17_631.84     1_210.13    18_841.98       0.9996          1.0001            1.0000         4.55
ExhaustiveBinary-512-random (self)                    17_631.84     3_426.18    21_058.03       0.9969          1.0004            1.0000         4.55
ExhaustiveBinary-512-pca_no_rr (query)                18_077.56       927.08    19_004.64       0.1131         15.1544           11.3937         4.55
ExhaustiveBinary-512-pca-rf10 (query)                 18_077.56     1_068.72    19_146.28       0.3166          2.0536            1.8315         4.55
ExhaustiveBinary-512-pca-rf20 (query)                 18_077.56     1_226.93    19_304.49       0.4179          1.6492            1.4934         4.55
ExhaustiveBinary-512-pca (self)                       18_077.56     3_486.97    21_564.53       0.3146          2.0599            1.8407         4.55
ExhaustiveBinary-1024-random_no_rr (query)            35_588.35     1_706.06    37_294.41       0.6446          1.4908            1.3512         9.11
ExhaustiveBinary-1024-random-rf10 (query)             35_588.35     1_891.34    37_479.70       0.9993          1.0001            1.0000         9.11
ExhaustiveBinary-1024-random-rf20 (query)             35_588.35     2_088.43    37_676.78       0.9999          1.0000            1.0000         9.11
ExhaustiveBinary-1024-random (self)                   35_588.35     6_158.25    41_746.61       0.9993          1.0001            1.0000         9.11
ExhaustiveBinary-1024-pca_no_rr (query)               35_903.30     1_756.43    37_659.73       0.2379          4.2518            3.5219         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                35_903.30     1_867.60    37_770.90       0.6180          1.2507            1.1871         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                35_903.30     2_041.90    37_945.19       0.7452          1.1303            1.0889         9.11
ExhaustiveBinary-1024-pca (self)                      35_903.30     6_128.58    42_031.87       0.6017          1.2739            1.2055         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  215.01       886.85     1_101.86       0.0420         17.7101           13.0961         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   215.01       953.22     1_168.23       0.1896          2.5240            2.4053         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   215.01     1_542.12     1_757.13       0.3228          1.8300            1.7351         4.58
ExhaustiveBinary-768-sign (self)                         215.01     3_109.16     3_324.17       0.1997          2.4832            2.3544         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)           11_924.01       376.12    12_300.13       0.5435          1.7067            1.5434         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)          11_924.01       387.24    12_311.24       0.5415          1.7256            1.5559         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)          11_924.01       405.58    12_329.58       0.5404          1.7418            1.5622         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)          11_924.01       490.83    12_414.84       0.9891          1.0017            1.0000         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)          11_924.01       594.83    12_518.84       0.9984          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)         11_924.01       489.60    12_413.61       0.9884          1.0019            1.0000         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)         11_924.01       594.52    12_518.53       0.9985          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)         11_924.01       532.77    12_456.77       0.9878          1.0020            1.0000         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)         11_924.01       609.55    12_533.55       0.9983          1.0002            1.0000         2.74
IVF-Binary-256-nl158-random (self)                    11_924.01     1_433.74    13_357.74       0.9891          1.0018            1.0000         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)          10_094.72       395.33    10_490.04       0.5424          1.7135            1.5511         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)          10_094.72       398.66    10_493.38       0.5416          1.7223            1.5568         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)          10_094.72       405.19    10_499.91       0.5406          1.7399            1.5637         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)         10_094.72       493.75    10_588.47       0.9890          1.0018            1.0000         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)         10_094.72       589.93    10_684.64       0.9986          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)         10_094.72       494.78    10_589.50       0.9885          1.0019            1.0000         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)         10_094.72       600.78    10_695.50       0.9986          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)         10_094.72       500.68    10_595.39       0.9879          1.0021            1.0000         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)         10_094.72       616.22    10_710.94       0.9983          1.0002            1.0000         2.93
IVF-Binary-256-nl223-random (self)                    10_094.72     1_448.58    11_543.30       0.9893          1.0017            1.0000         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)          10_450.52       397.54    10_848.06       0.5427          1.7075            1.5483         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)          10_450.52       396.26    10_846.77       0.5424          1.7119            1.5506         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)          10_450.52       408.17    10_858.68       0.5414          1.7263            1.5577         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)         10_450.52       508.73    10_959.24       0.9891          1.0018            1.0000         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)         10_450.52       605.82    11_056.33       0.9987          1.0002            1.0000         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)         10_450.52       502.63    10_953.15       0.9888          1.0019            1.0000         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)         10_450.52       604.46    11_054.98       0.9986          1.0002            1.0000         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)         10_450.52       507.52    10_958.04       0.9881          1.0020            1.0000         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)         10_450.52       644.59    11_095.10       0.9984          1.0002            1.0000         3.21
IVF-Binary-256-nl316-random (self)                    10_450.52     1_456.85    11_907.37       0.9896          1.0017            1.0000         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)              12_258.78       380.04    12_638.82       0.1457          5.8563            5.2304         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)             12_258.78       383.00    12_641.78       0.1391          6.6825            5.8419         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)             12_258.78       391.85    12_650.62       0.1357          7.3261            6.3027         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)             12_258.78       482.98    12_741.76       0.4666          1.4653            1.3962         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)             12_258.78       585.20    12_843.98       0.6312          1.2274            1.1753         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)            12_258.78       487.51    12_746.29       0.4346          1.5336            1.4630         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)            12_258.78       596.88    12_855.65       0.5900          1.2731            1.2181         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)            12_258.78       497.83    12_756.61       0.4168          1.5776            1.5029         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)            12_258.78       608.22    12_866.99       0.5672          1.3020            1.2431         2.74
IVF-Binary-256-nl158-pca (self)                       12_258.78     1_485.20    13_743.98       0.4323          1.5394            1.4692         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)             10_444.82       393.27    10_838.09       0.1439          5.8917            5.2757         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)             10_444.82       400.85    10_845.67       0.1410          6.1872            5.5139         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)             10_444.82       398.60    10_843.42       0.1368          6.8758            6.0590         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)            10_444.82       490.04    10_934.86       0.4605          1.4712            1.4044         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)            10_444.82       590.89    11_035.71       0.6245          1.2311            1.1812         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)            10_444.82       490.66    10_935.48       0.4464          1.5013            1.4342         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)            10_444.82       603.70    11_048.52       0.6063          1.2512            1.1995         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)            10_444.82       500.15    10_944.97       0.4240          1.5556            1.4847         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)            10_444.82       627.00    11_071.82       0.5775          1.2871            1.2318         2.93
IVF-Binary-256-nl223-pca (self)                       10_444.82     1_456.86    11_901.68       0.4444          1.5069            1.4395         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)             10_788.49       399.30    11_187.78       0.1440          5.7939            5.2401         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)             10_788.49       394.18    11_182.67       0.1425          5.9525            5.3554         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)             10_788.49       401.66    11_190.15       0.1385          6.5002            5.7837         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)            10_788.49       501.21    11_289.70       0.4618          1.4673            1.4012         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)            10_788.49       605.96    11_394.45       0.6271          1.2275            1.1795         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)            10_788.49       501.14    11_289.63       0.4542          1.4835            1.4175         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)            10_788.49       610.76    11_399.25       0.6168          1.2384            1.1896         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)            10_788.49       510.64    11_299.13       0.4328          1.5314            1.4638         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)            10_788.49       646.28    11_434.76       0.5893          1.2709            1.2196         3.21
IVF-Binary-256-nl316-pca (self)                       10_788.49     1_512.74    12_301.23       0.4521          1.4878            1.4231         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)           20_497.95       700.50    21_198.45       0.5928          1.5995            1.4605         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)          20_497.95       707.88    21_205.83       0.5905          1.6171            1.4731         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)          20_497.95       717.37    21_215.32       0.5893          1.6328            1.4788         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)          20_497.95       805.71    21_303.66       0.9972          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)          20_497.95       892.09    21_390.04       0.9994          1.0001            1.0000         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)         20_497.95       810.28    21_308.23       0.9972          1.0003            1.0000         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)         20_497.95       931.11    21_429.07       0.9998          1.0000            1.0000         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)         20_497.95       816.43    21_314.39       0.9969          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)         20_497.95       930.04    21_427.99       0.9997          1.0000            1.0000         5.02
IVF-Binary-512-nl158-random (self)                    20_497.95     2_513.90    23_011.86       0.9975          1.0003            1.0000         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)          18_701.54       712.23    19_413.77       0.5912          1.6081            1.4661         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)          18_701.54       711.14    19_412.68       0.5904          1.6164            1.4718         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)          18_701.54       722.19    19_423.74       0.5893          1.6325            1.4790         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)         18_701.54       801.48    19_503.02       0.9973          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)         18_701.54       901.88    19_603.42       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)         18_701.54       810.64    19_512.18       0.9971          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)         18_701.54       901.99    19_603.53       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)         18_701.54       821.90    19_523.45       0.9970          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)         18_701.54       920.10    19_621.64       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-random (self)                    18_701.54     2_478.47    21_180.01       0.9975          1.0003            1.0000         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)          19_095.87       716.15    19_812.02       0.5914          1.6049            1.4666         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)          19_095.87       712.21    19_808.08       0.5910          1.6095            1.4694         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)          19_095.87       724.76    19_820.62       0.5899          1.6239            1.4754         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)         19_095.87       818.10    19_913.97       0.9974          1.0003            1.0000         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)         19_095.87       909.48    20_005.35       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)         19_095.87       814.26    19_910.13       0.9974          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)         19_095.87       911.46    20_007.33       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)         19_095.87       836.25    19_932.12       0.9971          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)         19_095.87       980.99    20_076.86       0.9997          1.0000            1.0000         5.48
IVF-Binary-512-nl316-random (self)                    19_095.87     2_815.35    21_911.22       0.9976          1.0003            1.0000         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)              21_149.79       697.80    21_847.58       0.1305          6.4503            5.7489         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)             21_149.79       720.23    21_870.02       0.1240          7.4486            6.4833         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)             21_149.79       719.03    21_868.82       0.1206          8.2329            7.0306         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)             21_149.79       798.19    21_947.98       0.4226          1.5592            1.4869         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)             21_149.79       899.73    22_049.52       0.5795          1.2871            1.2326         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)            21_149.79       836.22    21_986.01       0.3886          1.6494            1.5744         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)            21_149.79       925.14    22_074.92       0.5326          1.3504            1.2916         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)            21_149.79       825.02    21_974.81       0.3705          1.7076            1.6261         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)            21_149.79       937.52    22_087.31       0.5073          1.3915            1.3280         5.02
IVF-Binary-512-nl158-pca (self)                       21_149.79     2_567.65    23_717.43       0.3870          1.6546            1.5787         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)             19_108.87       706.22    19_815.09       0.1290          6.4930            5.8028         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)             19_108.87       733.12    19_841.99       0.1260          6.8508            6.1013         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)             19_108.87       730.88    19_839.75       0.1218          7.6778            6.7186         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)            19_108.87       814.47    19_923.34       0.4165          1.5663            1.4969         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)            19_108.87       907.85    20_016.72       0.5722          1.2915            1.2393         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)            19_108.87       820.94    19_929.81       0.4017          1.6050            1.5371         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)            19_108.87       926.31    20_035.18       0.5516          1.3190            1.2656         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)            19_108.87       825.81    19_934.68       0.3784          1.6758            1.6008         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)            19_108.87       953.29    20_062.16       0.5198          1.3682            1.3096         5.21
IVF-Binary-512-nl223-pca (self)                       19_108.87     2_548.75    21_657.62       0.4000          1.6099            1.5385         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)             19_535.49       716.45    20_251.95       0.1288          6.3626            5.7453         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)             19_535.49       715.99    20_251.49       0.1273          6.5540            5.9125         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)             19_535.49       734.07    20_269.57       0.1233          7.2119            6.4371         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)            19_535.49       821.18    20_356.67       0.4179          1.5602            1.4926         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)            19_535.49       930.65    20_466.15       0.5746          1.2868            1.2367         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)            19_535.49       826.09    20_361.59       0.4100          1.5803            1.5143         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)            19_535.49       927.12    20_462.61       0.5630          1.3018            1.2517         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)            19_535.49       836.47    20_371.96       0.3876          1.6436            1.5764         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)            19_535.49       946.84    20_482.33       0.5324          1.3461            1.2910         5.48
IVF-Binary-512-nl316-pca (self)                       19_535.49     2_571.29    22_106.78       0.4084          1.5847            1.5167         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)          37_985.35     1_339.57    39_324.92       0.6492          1.4395            1.3309         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)         37_985.35     1_356.60    39_341.96       0.6471          1.4527            1.3401         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)         37_985.35     1_391.91    39_377.26       0.6459          1.4636            1.3437         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)         37_985.35     1_428.37    39_413.72       0.9991          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)         37_985.35     1_519.71    39_505.06       0.9995          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)        37_985.35     1_442.91    39_428.26       0.9995          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)        37_985.35     1_548.43    39_533.78       0.9999          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)        37_985.35     1_461.83    39_447.18       0.9994          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)        37_985.35     1_571.08    39_556.43       0.9999          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-random (self)                   37_985.35     4_694.70    42_680.05       0.9995          1.0001            1.0000         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)         36_256.35     1_353.38    37_609.73       0.6478          1.4482            1.3356         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)         36_256.35     1_348.04    37_604.40       0.6470          1.4545            1.3394         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)         36_256.35     1_369.87    37_626.22       0.6459          1.4660            1.3437         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)        36_256.35     1_436.36    37_692.72       0.9994          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)        36_256.35     1_541.16    37_797.51       0.9998          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)        36_256.35     1_436.09    37_692.44       0.9994          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)        36_256.35     1_536.69    37_793.04       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)        36_256.35     1_455.50    37_711.85       0.9994          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)        36_256.35     1_565.68    37_822.03       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-random (self)                   36_256.35     4_628.21    40_884.56       0.9995          1.0001            1.0000         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)         36_632.76     1_355.99    37_988.75       0.6478          1.4451            1.3360        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)         36_632.76     1_352.95    37_985.71       0.6474          1.4486            1.3381        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)         36_632.76     1_363.19    37_995.95       0.6465          1.4581            1.3419        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)        36_632.76     1_463.87    38_096.63       0.9995          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)        36_632.76     1_554.06    38_186.82       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)        36_632.76     1_526.92    38_159.68       0.9995          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)        36_632.76     1_552.26    38_185.02       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)        36_632.76     1_460.23    38_092.99       0.9994          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)        36_632.76     1_564.32    38_197.08       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-random (self)                   36_632.76     4_611.45    41_244.21       0.9995          1.0001            1.0000        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)             38_262.76     1_344.00    39_606.76       0.2469          3.4833            3.1162         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)            38_262.76     1_357.40    39_620.16       0.2424          3.6736            3.2451         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)            38_262.76     1_381.23    39_643.99       0.2408          3.7958            3.3267         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)            38_262.76     1_433.99    39_696.75       0.6578          1.2066            1.1496         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)            38_262.76     1_547.91    39_810.67       0.7968          1.0953            1.0612         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)           38_262.76     1_452.50    39_715.26       0.6394          1.2250            1.1657         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)           38_262.76     1_567.85    39_830.61       0.7721          1.1103            1.0735         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)           38_262.76     1_472.07    39_734.83       0.6306          1.2347            1.1737         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)           38_262.76     1_590.54    39_853.30       0.7608          1.1180            1.0798         9.57
IVF-Binary-1024-nl158-pca (self)                      38_262.76     4_771.25    43_034.01       0.6243          1.2452            1.1823         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)            36_590.06     1_358.97    37_949.03       0.2449          3.5197            3.1491         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)            36_590.06     1_368.33    37_958.39       0.2431          3.5920            3.2028         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)            36_590.06     1_366.86    37_956.92       0.2408          3.7558            3.3134         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)           36_590.06     1_439.49    38_029.55       0.6523          1.2109            1.1547         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)           36_590.06     1_587.19    38_177.24       0.7909          1.0980            1.0641         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)           36_590.06     1_458.23    38_048.28       0.6443          1.2192            1.1606         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)           36_590.06     1_553.75    38_143.81       0.7798          1.1049            1.0698         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)           36_590.06     1_469.07    38_059.12       0.6327          1.2324            1.1717         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)           36_590.06     1_566.55    38_156.61       0.7644          1.1156            1.0777         9.76
IVF-Binary-1024-nl223-pca (self)                      36_590.06     4_654.70    41_244.76       0.6304          1.2374            1.1766         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)            36_900.49     1_350.60    38_251.09       0.2449          3.5038            3.1446        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)            36_900.49     1_365.79    38_266.28       0.2440          3.5444            3.1721        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)            36_900.49     1_373.86    38_274.36       0.2417          3.6757            3.2627        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)           36_900.49     1_450.28    38_350.77       0.6517          1.2115            1.1547        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)           36_900.49     1_561.94    38_462.43       0.7905          1.0983            1.0643        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)           36_900.49     1_448.08    38_348.57       0.6474          1.2159            1.1581        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)           36_900.49     1_558.57    38_459.06       0.7846          1.1020            1.0674        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)           36_900.49     1_468.78    38_369.27       0.6363          1.2280            1.1679        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)           36_900.49     1_581.87    38_482.36       0.7696          1.1118            1.0752        10.04
IVF-Binary-1024-nl316-pca (self)                      36_900.49     4_665.70    41_566.19       0.6343          1.2333            1.1728        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)              2_990.16       596.18     3_586.34       0.1087          4.9638            4.2820         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)             2_990.16       668.40     3_658.56       0.0910          5.6532            4.8194         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)             2_990.16       725.61     3_715.77       0.0835          6.4783            5.2784         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)             2_990.16       678.06     3_668.22       0.3875          1.9344            1.4961         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)             2_990.16     1_246.06     4_236.22       0.4922          1.6906            1.2787         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)            2_990.16       774.16     3_764.32       0.3241          2.2103            1.7117         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)            2_990.16     1_297.12     4_287.28       0.4045          1.9478            1.4779         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)            2_990.16       780.89     3_771.05       0.2968          2.3464            1.8295         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)            2_990.16     1_312.28     4_302.44       0.3625          2.1099            1.6150         5.04
IVF-Binary-768-nl158-sign (self)                       2_990.16     2_122.61     5_112.77       0.3246          2.2070            1.7069         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)             1_237.97       622.01     1_859.98       0.1024          4.9876            4.3487         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)             1_237.97       655.43     1_893.40       0.0957          5.3044            4.5915         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)             1_237.97       731.60     1_969.57       0.0884          6.0740            5.0397         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)            1_237.97       704.85     1_942.82       0.3639          1.9436            1.5676         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)            1_237.97     1_211.75     2_449.72       0.4695          1.6779            1.3224         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)            1_237.97       735.64     1_973.61       0.3385          2.0460            1.6505         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)            1_237.97     1_253.52     2_491.49       0.4321          1.7706            1.4013         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)            1_237.97       838.33     2_076.30       0.3021          2.2125            1.7971         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)            1_237.97     1_425.87     2_663.84       0.3777          1.9467            1.5524         5.23
IVF-Binary-768-nl223-sign (self)                       1_237.97     2_240.96     3_478.93       0.3378          2.0482            1.6525         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)             1_619.72       711.46     2_331.17       0.0933          5.0942            4.4866         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)             1_619.72       731.18     2_350.89       0.0905          5.2553            4.5966         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)             1_619.72       826.13     2_445.84       0.0847          5.8642            4.9952         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)            1_619.72       794.26     2_413.98       0.3578          1.8929            1.5824         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)            1_619.72     1_355.91     2_975.63       0.4584          1.6471            1.3457         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)            1_619.72       810.47     2_430.19       0.3461          1.9342            1.6236         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)            1_619.72     1_359.06     2_978.77       0.4395          1.6901            1.3849         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)            1_619.72       840.03     2_459.75       0.3116          2.0759            1.7505         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)            1_619.72     1_368.77     2_988.49       0.3881          1.8299            1.5147         5.51
IVF-Binary-768-nl316-sign (self)                       1_619.72     2_338.26     3_957.98       0.3468          1.9292            1.6208         5.51
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### <u>RaBitQ (IVF and exhaustive)</u>

[RaBitQ](https://arxiv.org/abs/2405.12497) binarises against a centroid and
keeps enough side information to reconstruct an unbiased distance estimate, so
it holds up without re-ranking where plain sign bits do not. Better the higher
the dimensionality. `ExhaustiveRaBitQ` trains its own `sqrt(n)` centroids;
`IVF-RaBitQ` reuses the IVF centroids directly. The price against a plain binary
index is query speed, since the approximate distance is more work than a popcount.

**Tunable parameters *(RaBitQ)*:**

- *reranking_factor*: As for the binary indices. The RaBitQ estimate picks the
  candidates, then the on-disk vectors are loaded and re-scored exactly. `10`
  means `10 * k` vectors get re-scored.

**Tunable parameters *(IVF-specific)*:**

- *Number of lists (nl)*: Number of k-means clusters, `sqrt(n)` as a default.
- *Number of probes (np)*: Typically `sqrt(nlist)` or up to 5% of `nlist`.

#### Correlated data

<details>
<summary><b>Correlated data - 256 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 256D - IVF-RaBitQ
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.97       690.12       723.09       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.97     2_204.97     2_237.94       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             832.10       567.48     1_399.58       0.5641          1.0369            1.0367         2.84
ExhaustiveRaBitQ-rf5 (query)                             832.10       625.57     1_457.67       0.9225          1.0017            1.0006         2.84
ExhaustiveRaBitQ-rf10 (query)                            832.10       684.84     1_516.94       0.9776          1.0003            1.0000         2.84
ExhaustiveRaBitQ-rf20 (query)                            832.10       773.35     1_605.45       0.9903          1.0000            1.0000         2.84
ExhaustiveRaBitQ (self)                                  832.10     2_275.63     3_107.73       0.9780          1.0003            1.0000         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_336.57       262.23     1_598.80       0.5751          1.0343            1.0347         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_336.57       357.71     1_694.28       0.5751          1.0343            1.0347         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_336.57       430.39     1_766.96       0.5751          1.0343            1.0347         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_336.57       359.13     1_695.69       0.9785          1.0003            1.0000         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_336.57       437.84     1_774.41       0.9905          1.0000            1.0000         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_336.57       446.28     1_782.85       0.9785          1.0003            1.0000         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_336.57       524.06     1_860.63       0.9905          1.0000            1.0000         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_336.57       525.26     1_861.82       0.9785          1.0003            1.0000         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_336.57       606.24     1_942.81       0.9905          1.0000            1.0000         2.89
IVF-RaBitQ-nl158 (self)                                1_336.57     1_997.38     3_333.94       0.9904          1.0000            1.0000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                        898.64       318.57     1_217.21       0.5864          1.0326            1.0324         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                        898.64       373.29     1_271.93       0.5864          1.0326            1.0324         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                        898.64       508.08     1_406.72       0.5864          1.0326            1.0324         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                       898.64       407.43     1_306.07       0.9814          1.0002            1.0000         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                       898.64       480.40     1_379.03       0.9907          1.0000            1.0000         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                       898.64       461.43     1_360.07       0.9815          1.0002            1.0000         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                       898.64       535.52     1_434.16       0.9908          1.0000            1.0000         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                       898.64       601.09     1_499.72       0.9815          1.0002            1.0000         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                       898.64       673.12     1_571.76       0.9908          1.0000            1.0000         2.95
IVF-RaBitQ-nl223 (self)                                  898.64     2_121.43     3_020.06       0.9907          1.0000            1.0000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_082.48       370.58     1_453.07       0.5946          1.0309            1.0310         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_082.48       400.47     1_482.96       0.5947          1.0309            1.0310         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_082.48       541.33     1_623.81       0.5947          1.0308            1.0309         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_082.48       443.56     1_526.04       0.9823          1.0002            1.0000         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_082.48       511.53     1_594.02       0.9908          1.0000            1.0000         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_082.48       478.67     1_561.15       0.9823          1.0002            1.0000         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_082.48       550.18     1_632.66       0.9909          1.0000            1.0000         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_082.48       617.14     1_699.63       0.9824          1.0002            1.0000         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_082.48       683.64     1_766.12       0.9910          1.0000            1.0000         3.04
IVF-RaBitQ-nl316 (self)                                1_082.48     2_254.27     3_336.75       0.9908          1.0000            1.0000         3.04
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 512D - IVF-RaBitQ
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.47     1_292.22     1_360.69       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.47     4_363.79     4_432.27       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           2_052.29     1_495.12     3_547.41       0.5729          1.0233            1.0233         5.23
ExhaustiveRaBitQ-rf5 (query)                           2_052.29     1_554.03     3_606.32       0.9181          1.0011            1.0004         5.23
ExhaustiveRaBitQ-rf10 (query)                          2_052.29     1_617.94     3_670.23       0.9701          1.0002            1.0000         5.23
ExhaustiveRaBitQ-rf20 (query)                          2_052.29     1_729.16     3_781.45       0.9820          1.0000            1.0000         5.23
ExhaustiveRaBitQ (self)                                2_052.29     5_386.58     7_438.87       0.9701          1.0002            1.0000         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       3_143.98       601.46     3_745.44       0.5821          1.0218            1.0223         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      3_143.98       865.97     4_009.95       0.5821          1.0218            1.0223         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      3_143.98     1_132.52     4_276.49       0.5821          1.0218            1.0223         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      3_143.98       722.49     3_866.46       0.9707          1.0002            1.0000         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      3_143.98       821.96     3_965.93       0.9820          1.0000            1.0000         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     3_143.98       978.43     4_122.41       0.9707          1.0002            1.0000         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     3_143.98     1_075.39     4_219.36       0.9820          1.0000            1.0000         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     3_143.98     1_234.03     4_378.01       0.9707          1.0002            1.0000         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     3_143.98     1_334.76     4_478.73       0.9820          1.0000            1.0000         5.32
IVF-RaBitQ-nl158 (self)                                3_143.98     4_412.88     7_556.86       0.9817          1.0000            1.0000         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      2_189.23       791.62     2_980.85       0.5907          1.0210            1.0212         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      2_189.23       948.12     3_137.34       0.5907          1.0210            1.0212         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      2_189.23     1_335.82     3_525.04       0.5907          1.0210            1.0212         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     2_189.23       893.74     3_082.96       0.9728          1.0001            1.0000         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     2_189.23       979.96     3_169.18       0.9823          1.0000            1.0000         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     2_189.23     1_045.58     3_234.81       0.9729          1.0001            1.0000         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     2_189.23     1_139.70     3_328.93       0.9823          1.0000            1.0000         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     2_189.23     1_418.24     3_607.47       0.9729          1.0001            1.0000         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     2_189.23     1_509.85     3_699.08       0.9823          1.0000            1.0000         5.44
IVF-RaBitQ-nl223 (self)                                2_189.23     4_981.95     7_171.18       0.9819          1.0000            1.0000         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      2_404.37       955.98     3_360.35       0.5965          1.0201            1.0205         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      2_404.37     1_086.47     3_490.84       0.5965          1.0201            1.0205         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      2_404.37     1_492.43     3_896.80       0.5965          1.0201            1.0205         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     2_404.37     1_068.89     3_473.27       0.9732          1.0001            1.0000         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     2_404.37     1_146.56     3_550.93       0.9824          1.0000            1.0000         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     2_404.37     1_162.65     3_567.02       0.9732          1.0001            1.0000         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     2_404.37     1_267.43     3_671.80       0.9824          1.0000            1.0000         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     2_404.37     1_574.51     3_978.89       0.9732          1.0001            1.0000         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     2_404.37     1_667.14     4_071.52       0.9824          1.0000            1.0000         5.63
IVF-RaBitQ-nl316 (self)                                2_404.37     5_549.99     7_954.36       0.9819          1.0000            1.0000         5.63
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 768D - IVF-RaBitQ
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       102.45     1_812.27     1_914.73       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        102.45     5_973.77     6_076.22       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           3_789.76     2_810.56     6_600.32       0.5744          1.0183            1.0183         8.11
ExhaustiveRaBitQ-rf5 (query)                           3_789.76     2_891.13     6_680.88       0.9135          1.0009            1.0003         8.11
ExhaustiveRaBitQ-rf10 (query)                          3_789.76     2_951.39     6_741.15       0.9630          1.0001            1.0000         8.11
ExhaustiveRaBitQ-rf20 (query)                          3_789.76     3_081.70     6_871.46       0.9740          1.0000            1.0000         8.11
ExhaustiveRaBitQ (self)                                3_789.76    10_409.67    14_199.42       0.9631          1.0001            1.0000         8.11
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_276.49     1_085.55     6_362.04       0.5831          1.0170            1.0175         8.25
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_276.49     1_608.77     6_885.25       0.5831          1.0170            1.0175         8.25
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_276.49     2_138.34     7_414.83       0.5831          1.0170            1.0175         8.25
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_276.49     1_220.19     6_496.67       0.9643          1.0001            1.0000         8.25
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_276.49     1_384.39     6_660.88       0.9741          1.0000            1.0000         8.25
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_276.49     1_773.19     7_049.68       0.9643          1.0001            1.0000         8.25
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_276.49     1_992.31     7_268.79       0.9741          1.0000            1.0000         8.25
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_276.49     2_273.63     7_550.12       0.9643          1.0001            1.0000         8.25
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_276.49     2_417.59     7_694.08       0.9741          1.0000            1.0000         8.25
IVF-RaBitQ-nl158 (self)                                5_276.49     7_914.19    13_190.67       0.9738          1.0000            1.0000         8.25
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_896.02     1_488.88     5_384.90       0.5837          1.0174            1.0174         8.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_896.02     1_828.84     5_724.86       0.5837          1.0174            1.0174         8.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_896.02     2_597.40     6_493.42       0.5837          1.0174            1.0174         8.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_896.02     1_639.35     5_535.38       0.9635          1.0001            1.0000         8.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_896.02     1_739.44     5_635.46       0.9740          1.0000            1.0000         8.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_896.02     1_953.05     5_849.07       0.9635          1.0001            1.0000         8.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_896.02     2_057.69     5_953.71       0.9740          1.0000            1.0000         8.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_896.02     2_735.13     6_631.15       0.9636          1.0001            1.0000         8.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_896.02     2_856.80     6_752.82       0.9740          1.0000            1.0000         8.44
IVF-RaBitQ-nl223 (self)                                3_896.02     9_457.59    13_353.61       0.9738          1.0000            1.0000         8.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      4_260.91     1_868.13     6_129.04       0.5946          1.0161            1.0165         8.71
IVF-RaBitQ-nl316-np17-rf0 (query)                      4_260.91     2_102.93     6_363.84       0.5946          1.0161            1.0165         8.71
IVF-RaBitQ-nl316-np25-rf0 (query)                      4_260.91     2_961.21     7_222.12       0.5946          1.0161            1.0165         8.71
IVF-RaBitQ-nl316-np15-rf10 (query)                     4_260.91     2_006.77     6_267.68       0.9656          1.0001            1.0000         8.71
IVF-RaBitQ-nl316-np15-rf20 (query)                     4_260.91     2_109.11     6_370.02       0.9744          1.0000            1.0000         8.71
IVF-RaBitQ-nl316-np17-rf10 (query)                     4_260.91     2_226.32     6_487.23       0.9657          1.0001            1.0000         8.71
IVF-RaBitQ-nl316-np17-rf20 (query)                     4_260.91     2_323.39     6_584.30       0.9744          1.0000            1.0000         8.71
IVF-RaBitQ-nl316-np25-rf10 (query)                     4_260.91     3_085.81     7_346.72       0.9657          1.0001            1.0000         8.71
IVF-RaBitQ-nl316-np25-rf20 (query)                     4_260.91     3_231.28     7_492.19       0.9744          1.0000            1.0000         8.71
IVF-RaBitQ-nl316 (self)                                4_260.91    10_570.22    14_831.13       0.9740          1.0000            1.0000         8.71
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 256D - IVF-RaBitQ
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.49       698.95       731.45       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.49     2_353.50     2_385.99       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             927.36       903.24     1_830.59       0.7286          1.0245            1.0235         2.84
ExhaustiveRaBitQ-rf5 (query)                             927.36       958.88     1_886.24       0.9947          1.0001            1.0000         2.84
ExhaustiveRaBitQ-rf10 (query)                            927.36     1_080.53     2_007.89       0.9976          1.0000            1.0000         2.84
ExhaustiveRaBitQ-rf20 (query)                            927.36     1_130.94     2_058.30       0.9977          1.0000            1.0000         2.84
ExhaustiveRaBitQ (self)                                  927.36     3_377.93     4_305.29       0.9977          1.0000            1.0000         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_404.05       248.05     1_652.10       0.7297          1.0243            1.0234         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_404.05       346.68     1_750.72       0.7297          1.0243            1.0234         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_404.05       436.35     1_840.40       0.7297          1.0243            1.0234         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_404.05       345.55     1_749.59       0.9976          1.0000            1.0000         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_404.05       426.34     1_830.39       0.9977          1.0000            1.0000         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_404.05       436.99     1_841.03       0.9976          1.0000            1.0000         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_404.05       527.44     1_931.49       0.9977          1.0000            1.0000         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_404.05       532.63     1_936.68       0.9976          1.0000            1.0000         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_404.05       615.47     2_019.51       0.9977          1.0000            1.0000         2.89
IVF-RaBitQ-nl158 (self)                                1_404.05     2_005.15     3_409.19       0.9977          1.0000            1.0000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                        976.74       314.01     1_290.74       0.7341          1.0236            1.0227         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                        976.74       367.84     1_344.58       0.7341          1.0236            1.0227         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                        976.74       524.22     1_500.95       0.7341          1.0236            1.0227         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                       976.74       403.54     1_380.28       0.9976          1.0000            1.0000         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                       976.74       484.68     1_461.42       0.9977          1.0000            1.0000         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                       976.74       451.83     1_428.57       0.9976          1.0000            1.0000         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                       976.74       537.43     1_514.17       0.9977          1.0000            1.0000         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                       976.74       615.33     1_592.07       0.9976          1.0000            1.0000         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                       976.74       688.54     1_665.28       0.9977          1.0000            1.0000         2.95
IVF-RaBitQ-nl223 (self)                                  976.74     2_272.34     3_249.08       0.9977          1.0000            1.0000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_175.54       355.64     1_531.17       0.7371          1.0230            1.0219         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_175.54       393.71     1_569.25       0.7371          1.0230            1.0219         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_175.54       560.15     1_735.69       0.7371          1.0230            1.0219         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_175.54       451.58     1_627.12       0.9976          1.0000            1.0000         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_175.54       532.21     1_707.75       0.9977          1.0000            1.0000         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_175.54       482.08     1_657.62       0.9976          1.0000            1.0000         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_175.54       582.50     1_758.04       0.9977          1.0000            1.0000         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_175.54       638.62     1_814.16       0.9976          1.0000            1.0000         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_175.54       715.27     1_890.80       0.9977          1.0000            1.0000         3.04
IVF-RaBitQ-nl316 (self)                                1_175.54     2_373.51     3_549.05       0.9977          1.0000            1.0000         3.04
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 512D - IVF-RaBitQ
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.88     1_260.45     1_328.33       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         67.88     4_168.28     4_236.16       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           2_127.77     2_136.65     4_264.42       0.7429          1.0147            1.0141         5.23
ExhaustiveRaBitQ-rf5 (query)                           2_127.77     2_240.26     4_368.03       0.9905          1.0000            1.0000         5.23
ExhaustiveRaBitQ-rf10 (query)                          2_127.77     2_265.11     4_392.88       0.9923          1.0000            1.0000         5.23
ExhaustiveRaBitQ-rf20 (query)                          2_127.77     2_380.19     4_507.96       0.9923          1.0000            1.0000         5.23
ExhaustiveRaBitQ (self)                                2_127.77     7_502.11     9_629.88       0.9923          1.0000            1.0000         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       2_896.74       587.77     3_484.50       0.7437          1.0145            1.0140         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      2_896.74       818.80     3_715.54       0.7437          1.0145            1.0140         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      2_896.74     1_076.79     3_973.53       0.7437          1.0145            1.0140         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      2_896.74       699.77     3_596.51       0.9923          1.0000            1.0000         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      2_896.74       806.15     3_702.89       0.9923          1.0000            1.0000         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     2_896.74       929.28     3_826.02       0.9923          1.0000            1.0000         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     2_896.74     1_035.86     3_932.60       0.9923          1.0000            1.0000         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     2_896.74     1_184.87     4_081.61       0.9923          1.0000            1.0000         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     2_896.74     1_313.39     4_210.13       0.9923          1.0000            1.0000         5.32
IVF-RaBitQ-nl158 (self)                                2_896.74     4_242.19     7_138.93       0.9923          1.0000            1.0000         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      2_258.97       769.54     3_028.52       0.7464          1.0142            1.0135         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      2_258.97       923.48     3_182.45       0.7467          1.0142            1.0135         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      2_258.97     1_331.99     3_590.96       0.7467          1.0142            1.0135         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     2_258.97       887.08     3_146.05       0.9912          1.0000            1.0000         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     2_258.97       995.15     3_254.12       0.9912          1.0000            1.0000         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     2_258.97     1_036.04     3_295.02       0.9923          1.0000            1.0000         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     2_258.97     1_153.58     3_412.55       0.9923          1.0000            1.0000         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     2_258.97     1_470.82     3_729.79       0.9923          1.0000            1.0000         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     2_258.97     1_535.68     3_794.65       0.9923          1.0000            1.0000         5.44
IVF-RaBitQ-nl223 (self)                                2_258.97     5_103.24     7_362.21       0.9923          1.0000            1.0000         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      2_653.85       968.71     3_622.56       0.7473          1.0141            1.0135         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      2_653.85     1_041.93     3_695.78       0.7477          1.0141            1.0135         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      2_653.85     1_453.04     4_106.89       0.7478          1.0140            1.0135         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     2_653.85     1_054.94     3_708.79       0.9909          1.0001            1.0000         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     2_653.85     1_155.25     3_809.10       0.9910          1.0001            1.0000         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     2_653.85     1_155.17     3_809.03       0.9918          1.0000            1.0000         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     2_653.85     1_257.16     3_911.01       0.9919          1.0000            1.0000         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     2_653.85     1_569.13     4_222.98       0.9923          1.0000            1.0000         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     2_653.85     1_664.69     4_318.54       0.9923          1.0000            1.0000         5.63
IVF-RaBitQ-nl316 (self)                                2_653.85     5_992.06     8_645.91       0.9923          1.0000            1.0000         5.63
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 768D - IVF-RaBitQ
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       102.86     1_795.72     1_898.58       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        102.86     5_877.93     5_980.79       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           3_896.79     3_585.46     7_482.26       0.7241          1.0120            1.0117         8.11
ExhaustiveRaBitQ-rf5 (query)                           3_896.79     3_677.20     7_574.00       0.9743          1.0000            1.0000         8.11
ExhaustiveRaBitQ-rf10 (query)                          3_896.79     3_765.47     7_662.27       0.9771          0.9999            1.0000         8.11
ExhaustiveRaBitQ-rf20 (query)                          3_896.79     3_878.39     7_775.19       0.9772          0.9999            1.0000         8.11
ExhaustiveRaBitQ (self)                                3_896.79    12_402.26    16_299.05       0.9772          0.9999            1.0000         8.11
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_109.33     1_049.89     6_159.23       0.7258          1.0119            1.0114         8.25
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_109.33     1_575.11     6_684.44       0.7258          1.0119            1.0114         8.25
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_109.33     2_079.25     7_188.58       0.7258          1.0119            1.0114         8.25
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_109.33     1_196.31     6_305.64       0.9771          0.9999            1.0000         8.25
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_109.33     1_322.99     6_432.32       0.9772          0.9999            1.0000         8.25
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_109.33     1_726.85     6_836.19       0.9771          0.9999            1.0000         8.25
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_109.33     1_823.66     6_932.99       0.9772          0.9999            1.0000         8.25
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_109.33     2_224.81     7_334.14       0.9771          0.9999            1.0000         8.25
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_109.33     2_359.56     7_468.90       0.9772          0.9999            1.0000         8.25
IVF-RaBitQ-nl158 (self)                                5_109.33     7_779.96    12_889.29       0.9772          0.9999            1.0000         8.25
IVF-RaBitQ-nl223-np11-rf0 (query)                      4_058.73     1_477.14     5_535.87       0.7273          1.0117            1.0114         8.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      4_058.73     1_844.66     5_903.39       0.7273          1.0117            1.0113         8.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      4_058.73     2_522.20     6_580.93       0.7273          1.0117            1.0113         8.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     4_058.73     1_617.56     5_676.29       0.9770          0.9999            1.0000         8.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     4_058.73     1_717.42     5_776.15       0.9770          0.9999            1.0000         8.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     4_058.73     1_906.96     5_965.69       0.9771          0.9999            1.0000         8.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     4_058.73     2_027.67     6_086.40       0.9772          0.9999            1.0000         8.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     4_058.73     2_673.75     6_732.48       0.9771          0.9999            1.0000         8.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     4_058.73     2_802.26     6_860.99       0.9772          0.9999            1.0000         8.44
IVF-RaBitQ-nl223 (self)                                4_058.73     9_266.99    13_325.72       0.9772          0.9999            1.0000         8.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      4_642.31     1_839.75     6_482.06       0.7281          1.0116            1.0113         8.71
IVF-RaBitQ-nl316-np17-rf0 (query)                      4_642.31     2_051.78     6_694.08       0.7283          1.0116            1.0112         8.71
IVF-RaBitQ-nl316-np25-rf0 (query)                      4_642.31     2_883.13     7_525.43       0.7283          1.0116            1.0112         8.71
IVF-RaBitQ-nl316-np15-rf10 (query)                     4_642.31     1_985.17     6_627.48       0.9767          1.0000            1.0000         8.71
IVF-RaBitQ-nl316-np15-rf20 (query)                     4_642.31     2_094.84     6_737.15       0.9767          1.0000            1.0000         8.71
IVF-RaBitQ-nl316-np17-rf10 (query)                     4_642.31     2_187.38     6_829.69       0.9770          0.9999            1.0000         8.71
IVF-RaBitQ-nl316-np17-rf20 (query)                     4_642.31     2_304.09     6_946.39       0.9771          0.9999            1.0000         8.71
IVF-RaBitQ-nl316-np25-rf10 (query)                     4_642.31     3_042.36     7_684.67       0.9771          0.9999            1.0000         8.71
IVF-RaBitQ-nl316-np25-rf20 (query)                     4_642.31     3_143.86     7_786.17       0.9772          0.9999            1.0000         8.71
IVF-RaBitQ-nl316 (self)                                4_642.31    10_430.15    15_072.45       0.9772          0.9999            1.0000         8.71
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Cell embeddings

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 256D - IVF-RaBitQ
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.92       731.00       763.91       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.92     2_413.34     2_446.25       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                           1_021.42     1_290.85     2_312.28       0.8680          1.0296            1.0242         2.84
ExhaustiveRaBitQ-rf5 (query)                           1_021.42     1_344.38     2_365.80       1.0000          1.0000            1.0000         2.84
ExhaustiveRaBitQ-rf10 (query)                          1_021.42     1_420.64     2_442.06       1.0000          1.0000            1.0000         2.84
ExhaustiveRaBitQ-rf20 (query)                          1_021.42     1_541.64     2_563.06       1.0000          1.0000            1.0000         2.84
ExhaustiveRaBitQ (self)                                1_021.42     4_719.59     5_741.01       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_492.06       342.72     1_834.79       0.8728          1.0278            1.0225         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_492.06       560.55     2_052.61       0.8733          1.0275            1.0223         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_492.06       780.79     2_272.86       0.8733          1.0275            1.0223         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_492.06       440.55     1_932.62       0.9976          1.0005            1.0000         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_492.06       522.82     2_014.88       0.9976          1.0005            1.0000         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_492.06       655.16     2_147.22       0.9999          1.0000            1.0000         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_492.06       738.04     2_230.11       0.9999          1.0000            1.0000         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_492.06       865.17     2_357.23       1.0000          1.0000            1.0000         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_492.06       951.80     2_443.87       1.0000          1.0000            1.0000         2.89
IVF-RaBitQ-nl158 (self)                                1_492.06     3_137.35     4_629.41       1.0000          1.0000            1.0000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                        834.34       375.70     1_210.04       0.8833          1.0228            1.0186         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                        834.34       464.84     1_299.18       0.8834          1.0227            1.0186         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                        834.34       678.27     1_512.61       0.8833          1.0228            1.0186         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                       834.34       466.32     1_300.66       0.9994          1.0001            1.0000         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                       834.34       545.45     1_379.79       0.9994          1.0001            1.0000         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                       834.34       553.71     1_388.04       0.9999          1.0000            1.0000         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                       834.34       635.08     1_469.42       0.9999          1.0000            1.0000         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                       834.34       770.14     1_604.48       1.0000          1.0000            1.0000         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                       834.34       847.69     1_682.03       1.0000          1.0000            1.0000         2.95
IVF-RaBitQ-nl223 (self)                                  834.34     2_782.61     3_616.95       1.0000          1.0000            1.0000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                        970.35       412.24     1_382.58       0.8893          1.0202            1.0165         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                        970.35       458.51     1_428.86       0.8893          1.0202            1.0165         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                        970.35       665.88     1_636.22       0.8893          1.0202            1.0165         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                       970.35       494.83     1_465.18       0.9997          1.0000            1.0000         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                       970.35       574.20     1_544.55       0.9997          1.0000            1.0000         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                       970.35       551.02     1_521.36       0.9998          1.0000            1.0000         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                       970.35       635.05     1_605.39       0.9998          1.0000            1.0000         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                       970.35       740.72     1_711.07       1.0000          1.0000            1.0000         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                       970.35       820.86     1_791.21       1.0000          1.0000            1.0000         3.04
IVF-RaBitQ-nl316 (self)                                  970.35     2_717.18     3_687.53       1.0000          1.0000            1.0000         3.04
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 512D - IVF-RaBitQ
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        69.28     1_264.59     1_333.86       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.28     4_203.58     4_272.86       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           2_429.41     2_684.00     5_113.41       0.9024          1.0153            1.0116         5.23
ExhaustiveRaBitQ-rf5 (query)                           2_429.41     2_720.58     5_149.99       1.0000          1.0000            1.0000         5.23
ExhaustiveRaBitQ-rf10 (query)                          2_429.41     2_799.03     5_228.44       1.0000          1.0000            1.0000         5.23
ExhaustiveRaBitQ-rf20 (query)                          2_429.41     2_945.74     5_375.14       1.0000          1.0000            1.0000         5.23
ExhaustiveRaBitQ (self)                                2_429.41     9_275.14    11_704.55       1.0000          1.0000            1.0000         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       3_331.29       750.31     4_081.61       0.9068          1.0138            1.0103         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      3_331.29     1_216.49     4_547.78       0.9073          1.0135            1.0101         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      3_331.29     1_681.22     5_012.51       0.9073          1.0135            1.0101         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      3_331.29       847.54     4_178.83       0.9985          1.0003            1.0000         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      3_331.29       954.70     4_285.99       0.9985          1.0003            1.0000         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     3_331.29     1_321.64     4_652.93       0.9999          1.0000            1.0000         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     3_331.29     1_421.33     4_752.63       0.9999          1.0000            1.0000         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     3_331.29     1_781.13     5_112.42       1.0000          1.0000            1.0000         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     3_331.29     1_891.67     5_222.96       1.0000          1.0000            1.0000         5.32
IVF-RaBitQ-nl158 (self)                                3_331.29     6_233.01     9_564.30       1.0000          1.0000            1.0000         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      2_042.75       881.79     2_924.54       0.9151          1.0111            1.0083         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      2_042.75     1_156.98     3_199.73       0.9152          1.0111            1.0083         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      2_042.75     1_633.22     3_675.97       0.9152          1.0111            1.0083         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     2_042.75       988.92     3_031.67       0.9997          1.0000            1.0000         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     2_042.75     1_085.03     3_127.78       0.9997          1.0000            1.0000         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     2_042.75     1_205.97     3_248.72       0.9999          1.0000            1.0000         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     2_042.75     1_305.42     3_348.17       0.9999          1.0000            1.0000         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     2_042.75     1_744.65     3_787.40       1.0000          1.0000            1.0000         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     2_042.75     1_836.50     3_879.25       1.0000          1.0000            1.0000         5.44
IVF-RaBitQ-nl223 (self)                                2_042.75     6_049.46     8_092.21       1.0000          1.0000            1.0000         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      2_250.97     1_029.82     3_280.79       0.9189          1.0100            1.0073         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      2_250.97     1_168.05     3_419.02       0.9189          1.0100            1.0073         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      2_250.97     1_698.36     3_949.32       0.9190          1.0100            1.0073         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     2_250.97     1_130.81     3_381.78       0.9998          1.0000            1.0000         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     2_250.97     1_227.08     3_478.04       0.9998          1.0000            1.0000         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     2_250.97     1_257.94     3_508.90       0.9999          1.0000            1.0000         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     2_250.97     1_360.81     3_611.78       0.9999          1.0000            1.0000         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     2_250.97     1_780.95     4_031.91       1.0000          1.0000            1.0000         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     2_250.97     1_924.94     4_175.90       1.0000          1.0000            1.0000         5.63
IVF-RaBitQ-nl316 (self)                                2_250.97     6_822.34     9_073.30       0.9999          1.0000            1.0000         5.63
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 768D - IVF-RaBitQ
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       101.51     1_803.78     1_905.29       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.51     5_905.71     6_007.22       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           4_323.64     4_396.96     8_720.60       0.9249          1.0085            1.0061         8.11
ExhaustiveRaBitQ-rf5 (query)                           4_323.64     4_499.31     8_822.94       1.0000          1.0000            1.0000         8.11
ExhaustiveRaBitQ-rf10 (query)                          4_323.64     4_701.89     9_025.52       1.0000          1.0000            1.0000         8.11
ExhaustiveRaBitQ-rf20 (query)                          4_323.64     4_740.95     9_064.59       1.0000          1.0000            1.0000         8.11
ExhaustiveRaBitQ (self)                                4_323.64    15_194.35    19_517.99       0.9999          1.0000            1.0000         8.11
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_556.62     1_291.10     6_847.72       0.9274          1.0078            1.0055         8.25
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_556.62     2_123.55     7_680.17       0.9276          1.0078            1.0055         8.25
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_556.62     2_922.58     8_479.20       0.9276          1.0078            1.0055         8.25
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_556.62     1_419.33     6_975.95       0.9995          1.0001            1.0000         8.25
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_556.62     1_530.18     7_086.80       0.9995          1.0001            1.0000         8.25
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_556.62     2_239.29     7_795.91       1.0000          1.0000            1.0000         8.25
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_556.62     2_359.74     7_916.36       1.0000          1.0000            1.0000         8.25
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_556.62     3_064.77     8_621.39       1.0000          1.0000            1.0000         8.25
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_556.62     3_173.33     8_729.95       1.0000          1.0000            1.0000         8.25
IVF-RaBitQ-nl158 (self)                                5_556.62    10_669.41    16_226.03       0.9999          1.0000            1.0000         8.25
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_742.89     1_652.00     5_394.90       0.9323          1.0067            1.0046         8.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_742.89     2_080.94     5_823.83       0.9323          1.0067            1.0046         8.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_742.89     3_076.85     6_819.74       0.9323          1.0067            1.0046         8.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_742.89     1_801.57     5_544.46       0.9998          1.0000            1.0000         8.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_742.89     1_904.97     5_647.86       0.9998          1.0000            1.0000         8.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_742.89     2_234.91     5_977.80       0.9999          1.0000            1.0000         8.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_742.89     2_327.03     6_069.92       0.9999          1.0000            1.0000         8.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_742.89     3_245.11     6_988.00       1.0000          1.0000            1.0000         8.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_742.89     3_340.27     7_083.16       1.0000          1.0000            1.0000         8.44
IVF-RaBitQ-nl223 (self)                                3_742.89    11_049.42    14_792.31       0.9999          1.0000            1.0000         8.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      4_108.82     1_988.33     6_097.14       0.9360          1.0060            1.0038         8.71
IVF-RaBitQ-nl316-np17-rf0 (query)                      4_108.82     2_232.58     6_341.40       0.9360          1.0060            1.0038         8.71
IVF-RaBitQ-nl316-np25-rf0 (query)                      4_108.82     3_252.32     7_361.14       0.9360          1.0060            1.0038         8.71
IVF-RaBitQ-nl316-np15-rf10 (query)                     4_108.82     2_108.86     6_217.68       0.9999          1.0000            1.0000         8.71
IVF-RaBitQ-nl316-np15-rf20 (query)                     4_108.82     2_235.99     6_344.81       0.9999          1.0000            1.0000         8.71
IVF-RaBitQ-nl316-np17-rf10 (query)                     4_108.82     2_353.80     6_462.61       1.0000          1.0000            1.0000         8.71
IVF-RaBitQ-nl316-np17-rf20 (query)                     4_108.82     2_461.55     6_570.36       1.0000          1.0000            1.0000         8.71
IVF-RaBitQ-nl316-np25-rf10 (query)                     4_108.82     3_372.49     7_481.30       1.0000          1.0000            1.0000         8.71
IVF-RaBitQ-nl316-np25-rf20 (query)                     4_108.82     3_493.03     7_601.85       1.0000          1.0000            1.0000         8.71
IVF-RaBitQ-nl316 (self)                                4_108.82    11_490.86    15_599.68       0.9999          1.0000            1.0000         8.71
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

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

**Tunable parameters *(TurboQuant)*:**

- *bits*: Bits per coordinate, 2, 3 or 4. More bits, better recall, more memory.
  3-bit has no SIMD kernel and falls back to the scalar scorer, which is
  markedly slower, so prefer 4-bit unless memory forces otherwise. The grid runs
  2-bit and 4-bit.
- *reranking_factor*: As for the other indices. Default `20`.

**Tunable parameters *(IVF-specific)*:**

- *Number of lists (nl)*: Number of k-means clusters, `sqrt(n)` as a default.
- *Number of probes (np)*: Typically `sqrt(nlist)` or up to 5% of `nlist`.

Self queries run with `reranking_factor = 20`. The encoding is data-oblivious,
so this one was designed for high-dimensional neural-network output rather than
for strongly clustered data.

#### Correlated data

<details>
<summary><b>Correlated data - 256 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 256D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.87       678.37       711.24       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.87     2_209.86     2_242.73       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              145.98       360.81       506.79       0.0971          1.7176            1.5958         7.12
ExhaustiveTQ-b2-rf5 (query)                              145.98       441.85       587.82       0.2336          1.2025            1.2204         7.12
ExhaustiveTQ-b2-rf10 (query)                             145.98       576.37       722.35       0.2854          1.1453            1.1620         7.12
ExhaustiveTQ-b2-rf20 (query)                             145.98     1_029.54     1_175.52       0.3808          1.0970            1.0941         7.12
ExhaustiveTQ-b2 (self)                                   145.98     3_152.14     3_298.12       0.3816          1.0980            1.0957         7.12
ExhaustiveTQ-b4-rf0 (query)                              230.05       570.68       800.73       0.1094          1.5328            1.4996        13.22
ExhaustiveTQ-b4-rf5 (query)                              230.05       657.71       887.76       0.2368          1.1884            1.2090        13.22
ExhaustiveTQ-b4-rf10 (query)                             230.05       802.36     1_032.41       0.2885          1.1372            1.1543        13.22
ExhaustiveTQ-b4-rf20 (query)                             230.05     1_184.45     1_414.50       0.3822          1.0940            1.0970        13.22
ExhaustiveTQ-b4 (self)                                   230.05     3_910.54     4_140.59       0.3840          1.0938            1.0948        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                          867.26       109.42       976.69       0.0971          1.7176            1.5958         7.80
IVF-TQ-b2-nl158-np12-rf0 (query)                         867.26       121.34       988.60       0.0971          1.7176            1.5958         7.80
IVF-TQ-b2-nl158-np17-rf0 (query)                         867.26       128.92       996.18       0.0971          1.7176            1.5958         7.80
IVF-TQ-b2-nl158-np7-rf10 (query)                         867.26       306.60     1_173.86       0.2854          1.1453            1.1619         7.80
IVF-TQ-b2-nl158-np7-rf20 (query)                         867.26       626.07     1_493.33       0.3808          1.0970            1.0941         7.80
IVF-TQ-b2-nl158-np12-rf10 (query)                        867.26       329.45     1_196.72       0.2854          1.1453            1.1619         7.80
IVF-TQ-b2-nl158-np12-rf20 (query)                        867.26       658.96     1_526.23       0.3808          1.0970            1.0941         7.80
IVF-TQ-b2-nl158-np17-rf10 (query)                        867.26       330.49     1_197.75       0.2854          1.1453            1.1619         7.80
IVF-TQ-b2-nl158-np17-rf20 (query)                        867.26       672.47     1_539.73       0.3808          1.0970            1.0941         7.80
IVF-TQ-b2-nl158 (self)                                   867.26     1_054.41     1_921.68       0.3816          1.0980            1.0957         7.80
IVF-TQ-b2-nl223-np11-rf0 (query)                         672.61       116.57       789.18       0.0971          1.7164            1.5942         7.93
IVF-TQ-b2-nl223-np14-rf0 (query)                         672.61       123.39       796.00       0.0971          1.7176            1.5958         7.93
IVF-TQ-b2-nl223-np21-rf0 (query)                         672.61       136.85       809.46       0.0971          1.7176            1.5958         7.93
IVF-TQ-b2-nl223-np11-rf10 (query)                        672.61       286.27       958.88       0.2856          1.1450            1.1618         7.93
IVF-TQ-b2-nl223-np11-rf20 (query)                        672.61       552.39     1_225.00       0.3813          1.0967            1.0934         7.93
IVF-TQ-b2-nl223-np14-rf10 (query)                        672.61       294.22       966.83       0.2854          1.1453            1.1620         7.93
IVF-TQ-b2-nl223-np14-rf20 (query)                        672.61       572.96     1_245.57       0.3808          1.0970            1.0941         7.93
IVF-TQ-b2-nl223-np21-rf10 (query)                        672.61       310.93       983.54       0.2854          1.1453            1.1620         7.93
IVF-TQ-b2-nl223-np21-rf20 (query)                        672.61       600.90     1_273.52       0.3808          1.0970            1.0941         7.93
IVF-TQ-b2-nl223 (self)                                   672.61     1_062.01     1_734.62       0.3816          1.0980            1.0957         7.93
IVF-TQ-b2-nl316-np15-rf0 (query)                         864.28       120.99       985.28       0.0973          1.6435            1.5781         8.10
IVF-TQ-b2-nl316-np17-rf0 (query)                         864.28       125.99       990.28       0.0972          1.7163            1.5957         8.10
IVF-TQ-b2-nl316-np25-rf0 (query)                         864.28       138.97     1_003.25       0.0971          1.7176            1.5958         8.10
IVF-TQ-b2-nl316-np15-rf10 (query)                        864.28       294.83     1_159.11       0.2859          1.1447            1.1615         8.10
IVF-TQ-b2-nl316-np15-rf20 (query)                        864.28       544.93     1_409.21       0.3816          1.0965            1.0931         8.10
IVF-TQ-b2-nl316-np17-rf10 (query)                        864.28       293.67     1_157.96       0.2854          1.1453            1.1619         8.10
IVF-TQ-b2-nl316-np17-rf20 (query)                        864.28       553.31     1_417.59       0.3808          1.0970            1.0941         8.10
IVF-TQ-b2-nl316-np25-rf10 (query)                        864.28       319.09     1_183.37       0.2854          1.1453            1.1620         8.10
IVF-TQ-b2-nl316-np25-rf20 (query)                        864.28       585.46     1_449.75       0.3808          1.0970            1.0941         8.10
IVF-TQ-b2-nl316 (self)                                   864.28     1_064.13     1_928.41       0.3816          1.0980            1.0957         8.10
IVF-TQ-b4-nl158-np7-rf0 (query)                          947.41       147.60     1_095.01       0.1094          1.5328            1.4996        14.05
IVF-TQ-b4-nl158-np12-rf0 (query)                         947.41       165.61     1_113.02       0.1094          1.5328            1.4996        14.05
IVF-TQ-b4-nl158-np17-rf0 (query)                         947.41       179.75     1_127.16       0.1094          1.5328            1.4996        14.05
IVF-TQ-b4-nl158-np7-rf10 (query)                         947.41       358.84     1_306.26       0.2885          1.1372            1.1543        14.05
IVF-TQ-b4-nl158-np7-rf20 (query)                         947.41       681.29     1_628.70       0.3823          1.0940            1.0970        14.05
IVF-TQ-b4-nl158-np12-rf10 (query)                        947.41       388.98     1_336.39       0.2885          1.1372            1.1543        14.05
IVF-TQ-b4-nl158-np12-rf20 (query)                        947.41       723.17     1_670.59       0.3822          1.0940            1.0970        14.05
IVF-TQ-b4-nl158-np17-rf10 (query)                        947.41       393.38     1_340.79       0.2885          1.1372            1.1543        14.05
IVF-TQ-b4-nl158-np17-rf20 (query)                        947.41       750.30     1_697.71       0.3823          1.0940            1.0970        14.05
IVF-TQ-b4-nl158 (self)                                   947.41     1_091.82     2_039.23       0.3840          1.0938            1.0948        14.05
IVF-TQ-b4-nl223-np11-rf0 (query)                         767.59       157.91       925.49       0.1095          1.5317            1.4988        14.25
IVF-TQ-b4-nl223-np14-rf0 (query)                         767.59       171.54       939.12       0.1094          1.5328            1.4996        14.25
IVF-TQ-b4-nl223-np21-rf0 (query)                         767.59       192.96       960.55       0.1094          1.5328            1.4996        14.25
IVF-TQ-b4-nl223-np11-rf10 (query)                        767.59       337.81     1_105.40       0.2887          1.1370            1.1541        14.25
IVF-TQ-b4-nl223-np11-rf20 (query)                        767.59       621.77     1_389.36       0.3825          1.0938            1.0966        14.25
IVF-TQ-b4-nl223-np14-rf10 (query)                        767.59       356.75     1_124.34       0.2885          1.1372            1.1543        14.25
IVF-TQ-b4-nl223-np14-rf20 (query)                        767.59       644.21     1_411.79       0.3822          1.0940            1.0970        14.25
IVF-TQ-b4-nl223-np21-rf10 (query)                        767.59       380.99     1_148.58       0.2885          1.1372            1.1543        14.25
IVF-TQ-b4-nl223-np21-rf20 (query)                        767.59       667.75     1_435.33       0.3822          1.0940            1.0970        14.25
IVF-TQ-b4-nl223 (self)                                   767.59     1_100.41     1_867.99       0.3840          1.0938            1.0948        14.25
IVF-TQ-b4-nl316-np15-rf0 (query)                         953.09       166.00     1_119.09       0.1094          1.5304            1.4978        14.49
IVF-TQ-b4-nl316-np17-rf0 (query)                         953.09       174.61     1_127.70       0.1094          1.5328            1.4996        14.49
IVF-TQ-b4-nl316-np25-rf0 (query)                         953.09       193.21     1_146.30       0.1094          1.5328            1.4996        14.49
IVF-TQ-b4-nl316-np15-rf10 (query)                        953.09       342.79     1_295.88       0.2887          1.1369            1.1541        14.49
IVF-TQ-b4-nl316-np15-rf20 (query)                        953.09       603.31     1_556.40       0.3827          1.0937            1.0967        14.49
IVF-TQ-b4-nl316-np17-rf10 (query)                        953.09       363.35     1_316.44       0.2885          1.1372            1.1543        14.49
IVF-TQ-b4-nl316-np17-rf20 (query)                        953.09       619.30     1_572.39       0.3822          1.0940            1.0970        14.49
IVF-TQ-b4-nl316-np25-rf10 (query)                        953.09       374.27     1_327.36       0.2885          1.1372            1.1543        14.49
IVF-TQ-b4-nl316-np25-rf20 (query)                        953.09       658.15     1_611.24       0.3822          1.0940            1.0970        14.49
IVF-TQ-b4-nl316 (self)                                   953.09     1_110.67     2_063.76       0.3840          1.0938            1.0948        14.49
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 512D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.09     1_361.92     1_430.01       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.09     4_507.18     4_575.27       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              338.99       657.36       996.35       0.1208          1.3711            1.3320        13.97
ExhaustiveTQ-b2-rf5 (query)                              338.99       745.47     1_084.46       0.2421          1.1333            1.1573        13.97
ExhaustiveTQ-b2-rf10 (query)                             338.99       886.41     1_225.40       0.2934          1.0981            1.1177        13.97
ExhaustiveTQ-b2-rf20 (query)                             338.99     1_264.07     1_603.06       0.3881          1.0664            1.0469        13.97
ExhaustiveTQ-b2 (self)                                   338.99     4_156.38     4_495.37       0.3879          1.0667            1.0471        13.97
ExhaustiveTQ-b4-rf0 (query)                              471.24     1_147.10     1_618.34       0.1314          1.3172            1.3126        26.18
ExhaustiveTQ-b4-rf5 (query)                              471.24     1_257.73     1_728.97       0.2469          1.1254            1.1483        26.18
ExhaustiveTQ-b4-rf10 (query)                             471.24     1_389.57     1_860.81       0.2968          1.0928            1.0979        26.18
ExhaustiveTQ-b4-rf20 (query)                             471.24     1_771.09     2_242.33       0.3881          1.0643            1.0492        26.18
ExhaustiveTQ-b4 (self)                                   471.24     5_854.22     6_325.45       0.3881          1.0646            1.0495        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_921.98       196.06     2_118.03       0.1208          1.3711            1.3320        14.95
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_921.98       212.62     2_134.60       0.1208          1.3711            1.3320        14.95
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_921.98       227.50     2_149.47       0.1208          1.3711            1.3320        14.95
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_921.98       420.99     2_342.97       0.2934          1.0981            1.1177        14.95
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_921.98       754.02     2_676.00       0.3881          1.0664            1.0469        14.95
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_921.98       433.46     2_355.44       0.2934          1.0981            1.1177        14.95
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_921.98       789.84     2_711.82       0.3881          1.0664            1.0469        14.95
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_921.98       451.67     2_373.65       0.2934          1.0981            1.1177        14.95
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_921.98       809.32     2_731.30       0.3881          1.0664            1.0469        14.95
IVF-TQ-b2-nl158 (self)                                 1_921.98     1_413.05     3_335.03       0.3879          1.0667            1.0471        14.95
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_199.06       212.56     1_411.63       0.1208          1.3698            1.3299        15.19
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_199.06       220.94     1_420.00       0.1208          1.3711            1.3320        15.19
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_199.06       239.10     1_438.16       0.1208          1.3711            1.3320        15.19
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_199.06       407.69     1_606.75       0.2937          1.0979            1.1176        15.19
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_199.06       704.99     1_904.05       0.3887          1.0662            1.0466        15.19
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_199.06       419.06     1_618.12       0.2934          1.0981            1.1177        15.19
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_199.06       734.08     1_933.14       0.3881          1.0664            1.0469        15.19
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_199.06       442.54     1_641.60       0.2934          1.0981            1.1177        15.19
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_199.06       770.19     1_969.25       0.3881          1.0664            1.0469        15.19
IVF-TQ-b2-nl223 (self)                                 1_199.06     1_443.48     2_642.54       0.3879          1.0667            1.0471        15.19
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_537.23       218.38     1_755.61       0.1208          1.3689            1.3287        15.56
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_537.23       226.53     1_763.76       0.1208          1.3707            1.3311        15.56
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_537.23       246.42     1_783.64       0.1208          1.3711            1.3320        15.56
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_537.23       409.39     1_946.62       0.2938          1.0976            1.1175        15.56
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_537.23       683.63     2_220.86       0.3892          1.0660            1.0465        15.56
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_537.23       424.12     1_961.35       0.2934          1.0980            1.1176        15.56
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_537.23       728.78     2_266.01       0.3883          1.0663            1.0469        15.56
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_537.23       458.32     1_995.54       0.2934          1.0981            1.1177        15.56
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_537.23       758.50     2_295.73       0.3881          1.0664            1.0469        15.56
IVF-TQ-b2-nl316 (self)                                 1_537.23     1_480.22     3_017.45       0.3879          1.0667            1.0471        15.56
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_998.61       274.48     2_273.09       0.1314          1.3172            1.3127        27.44
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_998.61       300.68     2_299.29       0.1314          1.3172            1.3126        27.44
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_998.61       324.27     2_322.87       0.1314          1.3172            1.3127        27.44
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_998.61       506.97     2_505.58       0.2968          1.0928            1.0979        27.44
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_998.61       857.84     2_856.45       0.3880          1.0643            1.0492        27.44
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_998.61       538.27     2_536.87       0.2969          1.0928            1.0979        27.44
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_998.61       893.82     2_892.43       0.3880          1.0643            1.0492        27.44
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_998.61       562.15     2_560.76       0.2969          1.0928            1.0979        27.44
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_998.61       925.69     2_924.30       0.3880          1.0643            1.0492        27.44
IVF-TQ-b4-nl158 (self)                                 1_998.61     1_605.61     3_604.22       0.3881          1.0646            1.0495        27.44
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_323.55       314.81     1_638.36       0.1315          1.3158            1.3116        27.79
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_323.55       316.93     1_640.47       0.1314          1.3172            1.3126        27.79
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_323.55       348.39     1_671.94       0.1314          1.3172            1.3126        27.79
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_323.55       501.59     1_825.14       0.2972          1.0926            1.0973        27.79
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_323.55       804.68     2_128.23       0.3887          1.0641            1.0489        27.79
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_323.55       521.98     1_845.53       0.2968          1.0928            1.0979        27.79
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_323.55       838.26     2_161.81       0.3881          1.0643            1.0492        27.79
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_323.55       559.18     1_882.73       0.2968          1.0928            1.0979        27.79
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_323.55       890.85     2_214.40       0.3880          1.0643            1.0492        27.79
IVF-TQ-b4-nl223 (self)                                 1_323.55     1_637.63     2_961.18       0.3881          1.0646            1.0495        27.79
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_653.22       308.33     1_961.56       0.1315          1.3151            1.3108        28.35
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_653.22       320.43     1_973.65       0.1315          1.3164            1.3120        28.35
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_653.22       357.43     2_010.65       0.1314          1.3172            1.3126        28.35
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_653.22       506.23     2_159.46       0.2974          1.0925            1.0966        28.35
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_653.22       794.07     2_447.29       0.3891          1.0639            1.0484        28.35
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_653.22       520.50     2_173.72       0.2970          1.0928            1.0977        28.35
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_653.22       816.63     2_469.85       0.3883          1.0642            1.0491        28.35
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_653.22       565.79     2_219.01       0.2968          1.0928            1.0979        28.35
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_653.22       879.77     2_533.00       0.3881          1.0643            1.0492        28.35
IVF-TQ-b4-nl316 (self)                                 1_653.22     1_670.61     3_323.83       0.3881          1.0646            1.0495        28.35
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 768D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       110.27     1_981.23     2_091.50       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        110.27     6_641.39     6_751.65       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              643.07       983.12     1_626.19       0.1292          1.2709            1.2627        21.33
ExhaustiveTQ-b2-rf5 (query)                              643.07     1_090.91     1_733.98       0.2468          1.1062            1.1332        21.33
ExhaustiveTQ-b2-rf10 (query)                             643.07     1_223.66     1_866.73       0.3001          1.0773            1.0630        21.33
ExhaustiveTQ-b2-rf20 (query)                             643.07     1_635.93     2_279.00       0.3961          1.0509            1.0334        21.33
ExhaustiveTQ-b2 (self)                                   643.07     5_547.54     6_190.61       0.3973          1.0507            1.0331        21.33
ExhaustiveTQ-b4-rf0 (query)                              776.85     1_799.52     2_576.38       0.1340          1.2531            1.2591        39.64
ExhaustiveTQ-b4-rf5 (query)                              776.85     1_902.85     2_679.70       0.2401          1.1135            1.1401        39.64
ExhaustiveTQ-b4-rf10 (query)                             776.85     2_017.38     2_794.24       0.2871          1.0888            1.1142        39.64
ExhaustiveTQ-b4-rf20 (query)                             776.85     2_425.05     3_201.90       0.3752          1.0657            1.0812        39.64
ExhaustiveTQ-b4 (self)                                   776.85     7_982.70     8_759.56       0.3767          1.0653            1.0638        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        2_826.79       321.92     3_148.71       0.1292          1.2709            1.2627        22.66
IVF-TQ-b2-nl158-np12-rf0 (query)                       2_826.79       322.39     3_149.18       0.1292          1.2709            1.2627        22.66
IVF-TQ-b2-nl158-np17-rf0 (query)                       2_826.79       342.75     3_169.54       0.1292          1.2709            1.2627        22.66
IVF-TQ-b2-nl158-np7-rf10 (query)                       2_826.79       543.10     3_369.89       0.3001          1.0773            1.0630        22.66
IVF-TQ-b2-nl158-np7-rf20 (query)                       2_826.79       911.32     3_738.11       0.3961          1.0509            1.0334        22.66
IVF-TQ-b2-nl158-np12-rf10 (query)                      2_826.79       563.39     3_390.18       0.3001          1.0773            1.0630        22.66
IVF-TQ-b2-nl158-np12-rf20 (query)                      2_826.79       952.27     3_779.06       0.3961          1.0509            1.0334        22.66
IVF-TQ-b2-nl158-np17-rf10 (query)                      2_826.79       590.67     3_417.46       0.3001          1.0773            1.0630        22.66
IVF-TQ-b2-nl158-np17-rf20 (query)                      2_826.79       974.33     3_801.12       0.3961          1.0509            1.0334        22.66
IVF-TQ-b2-nl158 (self)                                 2_826.79     1_860.80     4_687.60       0.3973          1.0507            1.0331        22.66
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_925.60       316.63     2_242.23       0.1292          1.2709            1.2627        23.04
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_925.60       331.53     2_257.13       0.1292          1.2709            1.2627        23.04
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_925.60       361.44     2_287.05       0.1292          1.2709            1.2627        23.04
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_925.60       551.79     2_477.39       0.3001          1.0773            1.0630        23.04
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_925.60       859.62     2_785.22       0.3961          1.0509            1.0334        23.04
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_925.60       568.99     2_494.59       0.3001          1.0773            1.0630        23.04
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_925.60       882.34     2_807.94       0.3961          1.0509            1.0334        23.04
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_925.60       629.09     2_554.69       0.3001          1.0773            1.0630        23.04
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_925.60       938.80     2_864.41       0.3961          1.0509            1.0333        23.04
IVF-TQ-b2-nl223 (self)                                 1_925.60     1_912.40     3_838.01       0.3973          1.0507            1.0331        23.04
IVF-TQ-b2-nl316-np15-rf0 (query)                       2_273.83       329.05     2_602.87       0.1292          1.2709            1.2627        23.57
IVF-TQ-b2-nl316-np17-rf0 (query)                       2_273.83       340.21     2_614.03       0.1292          1.2709            1.2627        23.57
IVF-TQ-b2-nl316-np25-rf0 (query)                       2_273.83       372.23     2_646.05       0.1292          1.2709            1.2627        23.57
IVF-TQ-b2-nl316-np15-rf10 (query)                      2_273.83       559.42     2_833.25       0.3001          1.0773            1.0630        23.57
IVF-TQ-b2-nl316-np15-rf20 (query)                      2_273.83       870.17     3_143.99       0.3962          1.0508            1.0333        23.57
IVF-TQ-b2-nl316-np17-rf10 (query)                      2_273.83       568.28     2_842.10       0.3001          1.0773            1.0631        23.57
IVF-TQ-b2-nl316-np17-rf20 (query)                      2_273.83       886.33     3_160.16       0.3961          1.0509            1.0334        23.57
IVF-TQ-b2-nl316-np25-rf10 (query)                      2_273.83       601.56     2_875.38       0.3001          1.0773            1.0630        23.57
IVF-TQ-b2-nl316-np25-rf20 (query)                      2_273.83       923.14     3_196.97       0.3961          1.0509            1.0334        23.57
IVF-TQ-b2-nl316 (self)                                 2_273.83     2_073.89     4_347.72       0.3973          1.0507            1.0331        23.57
IVF-TQ-b4-nl158-np7-rf0 (query)                        2_906.63       431.28     3_337.91       0.1340          1.2531            1.2591        41.46
IVF-TQ-b4-nl158-np12-rf0 (query)                       2_906.63       479.21     3_385.84       0.1340          1.2531            1.2591        41.46
IVF-TQ-b4-nl158-np17-rf0 (query)                       2_906.63       508.89     3_415.53       0.1340          1.2531            1.2592        41.46
IVF-TQ-b4-nl158-np7-rf10 (query)                       2_906.63       693.84     3_600.47       0.2871          1.0888            1.1142        41.46
IVF-TQ-b4-nl158-np7-rf20 (query)                       2_906.63     1_041.17     3_947.81       0.3752          1.0657            1.0812        41.46
IVF-TQ-b4-nl158-np12-rf10 (query)                      2_906.63       737.42     3_644.06       0.2871          1.0888            1.1142        41.46
IVF-TQ-b4-nl158-np12-rf20 (query)                      2_906.63     1_108.66     4_015.29       0.3752          1.0657            1.0812        41.46
IVF-TQ-b4-nl158-np17-rf10 (query)                      2_906.63       761.20     3_667.83       0.2871          1.0888            1.1142        41.46
IVF-TQ-b4-nl158-np17-rf20 (query)                      2_906.63     1_172.40     4_079.03       0.3752          1.0657            1.0812        41.46
IVF-TQ-b4-nl158 (self)                                 2_906.63     2_151.70     5_058.33       0.3767          1.0653            1.0637        41.46
IVF-TQ-b4-nl223-np11-rf0 (query)                       2_091.45       462.60     2_554.05       0.1340          1.2531            1.2591        42.04
IVF-TQ-b4-nl223-np14-rf0 (query)                       2_091.45       491.96     2_583.41       0.1340          1.2531            1.2591        42.04
IVF-TQ-b4-nl223-np21-rf0 (query)                       2_091.45       557.54     2_648.99       0.1340          1.2531            1.2591        42.04
IVF-TQ-b4-nl223-np11-rf10 (query)                      2_091.45       698.63     2_790.08       0.2871          1.0888            1.1142        42.04
IVF-TQ-b4-nl223-np11-rf20 (query)                      2_091.45     1_027.11     3_118.56       0.3752          1.0657            1.0812        42.04
IVF-TQ-b4-nl223-np14-rf10 (query)                      2_091.45       726.28     2_817.73       0.2871          1.0888            1.1142        42.04
IVF-TQ-b4-nl223-np14-rf20 (query)                      2_091.45     1_085.00     3_176.45       0.3752          1.0657            1.0812        42.04
IVF-TQ-b4-nl223-np21-rf10 (query)                      2_091.45       789.26     2_880.71       0.2871          1.0888            1.1142        42.04
IVF-TQ-b4-nl223-np21-rf20 (query)                      2_091.45     1_137.29     3_228.74       0.3752          1.0657            1.0812        42.04
IVF-TQ-b4-nl223 (self)                                 2_091.45     2_207.60     4_299.05       0.3767          1.0653            1.0638        42.04
IVF-TQ-b4-nl316-np15-rf0 (query)                       2_477.69       497.12     2_974.81       0.1340          1.2530            1.2591        42.81
IVF-TQ-b4-nl316-np17-rf0 (query)                       2_477.69       506.93     2_984.62       0.1340          1.2531            1.2591        42.81
IVF-TQ-b4-nl316-np25-rf0 (query)                       2_477.69       567.72     3_045.41       0.1340          1.2531            1.2591        42.81
IVF-TQ-b4-nl316-np15-rf10 (query)                      2_477.69       725.27     3_202.96       0.2871          1.0887            1.1142        42.81
IVF-TQ-b4-nl316-np15-rf20 (query)                      2_477.69     1_029.55     3_507.24       0.3753          1.0657            1.0812        42.81
IVF-TQ-b4-nl316-np17-rf10 (query)                      2_477.69       726.50     3_204.19       0.2871          1.0888            1.1142        42.81
IVF-TQ-b4-nl316-np17-rf20 (query)                      2_477.69     1_046.56     3_524.26       0.3752          1.0657            1.0812        42.81
IVF-TQ-b4-nl316-np25-rf10 (query)                      2_477.69       800.36     3_278.05       0.2871          1.0888            1.1142        42.81
IVF-TQ-b4-nl316-np25-rf20 (query)                      2_477.69     1_140.93     3_618.62       0.3752          1.0657            1.0812        42.81
IVF-TQ-b4-nl316 (self)                                 2_477.69     2_274.61     4_752.30       0.3767          1.0653            1.0637        42.81
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 256D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.74       685.00       717.74       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.74     2_314.31     2_347.05       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              144.09       359.05       503.14       0.0662          1.9161            1.8570         7.12
ExhaustiveTQ-b2-rf5 (query)                              144.09       436.45       580.54       0.1862          1.3185            1.3388         7.12
ExhaustiveTQ-b2-rf10 (query)                             144.09       572.06       716.15       0.2699          1.2136            1.2178         7.12
ExhaustiveTQ-b2-rf20 (query)                             144.09       953.21     1_097.30       0.4056          1.1279            1.1077         7.12
ExhaustiveTQ-b2 (self)                                   144.09     3_120.13     3_264.22       0.4070          1.1561            1.1291         7.12
ExhaustiveTQ-b4-rf0 (query)                              230.81       574.05       804.86       0.0871          1.7209            1.7599        13.22
ExhaustiveTQ-b4-rf5 (query)                              230.81       658.11       888.92       0.2058          1.2890            1.3085        13.22
ExhaustiveTQ-b4-rf10 (query)                             230.81       795.64     1_026.45       0.2865          1.1965            1.2033        13.22
ExhaustiveTQ-b4-rf20 (query)                             230.81     1_175.26     1_406.08       0.4169          1.1210            1.1109        13.22
ExhaustiveTQ-b4 (self)                                   230.81     4_105.57     4_336.39       0.4165          1.1485            1.1373        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                          975.68       102.81     1_078.49       0.0662          1.9161            1.8570         7.80
IVF-TQ-b2-nl158-np12-rf0 (query)                         975.68       113.43     1_089.11       0.0662          1.9161            1.8570         7.80
IVF-TQ-b2-nl158-np17-rf0 (query)                         975.68       131.23     1_106.91       0.0662          1.9162            1.8569         7.80
IVF-TQ-b2-nl158-np7-rf10 (query)                         975.68       297.79     1_273.48       0.2699          1.2136            1.2178         7.80
IVF-TQ-b2-nl158-np7-rf20 (query)                         975.68       607.82     1_583.51       0.4056          1.1279            1.1078         7.80
IVF-TQ-b2-nl158-np12-rf10 (query)                        975.68       310.05     1_285.73       0.2699          1.2136            1.2178         7.80
IVF-TQ-b2-nl158-np12-rf20 (query)                        975.68       628.22     1_603.91       0.4056          1.1279            1.1077         7.80
IVF-TQ-b2-nl158-np17-rf10 (query)                        975.68       342.46     1_318.15       0.2699          1.2136            1.2178         7.80
IVF-TQ-b2-nl158-np17-rf20 (query)                        975.68       687.40     1_663.09       0.4056          1.1279            1.1077         7.80
IVF-TQ-b2-nl158 (self)                                   975.68     1_095.32     2_071.00       0.4070          1.1561            1.1291         7.80
IVF-TQ-b2-nl223-np11-rf0 (query)                         737.29       114.99       852.28       0.0665          1.9134            1.8546         7.94
IVF-TQ-b2-nl223-np14-rf0 (query)                         737.29       123.95       861.24       0.0662          1.9160            1.8569         7.94
IVF-TQ-b2-nl223-np21-rf0 (query)                         737.29       145.17       882.47       0.0662          1.9162            1.8570         7.94
IVF-TQ-b2-nl223-np11-rf10 (query)                        737.29       289.70     1_026.99       0.2711          1.2125            1.2167         7.94
IVF-TQ-b2-nl223-np11-rf20 (query)                        737.29       560.30     1_297.59       0.4078          1.1269            1.1065         7.94
IVF-TQ-b2-nl223-np14-rf10 (query)                        737.29       302.45     1_039.74       0.2699          1.2136            1.2178         7.94
IVF-TQ-b2-nl223-np14-rf20 (query)                        737.29       583.28     1_320.57       0.4056          1.1279            1.1077         7.94
IVF-TQ-b2-nl223-np21-rf10 (query)                        737.29       336.59     1_073.89       0.2699          1.2136            1.2178         7.94
IVF-TQ-b2-nl223-np21-rf20 (query)                        737.29       633.20     1_370.49       0.4056          1.1279            1.1077         7.94
IVF-TQ-b2-nl223 (self)                                   737.29     1_112.75     1_850.04       0.4070          1.1561            1.1291         7.94
IVF-TQ-b2-nl316-np15-rf0 (query)                         935.38       121.17     1_056.55       0.0664          1.9139            1.8545         8.12
IVF-TQ-b2-nl316-np17-rf0 (query)                         935.38       124.09     1_059.47       0.0663          1.9153            1.8562         8.12
IVF-TQ-b2-nl316-np25-rf0 (query)                         935.38       142.99     1_078.37       0.0662          1.9161            1.8570         8.12
IVF-TQ-b2-nl316-np15-rf10 (query)                        935.38       285.67     1_221.05       0.2708          1.2128            1.2171         8.12
IVF-TQ-b2-nl316-np15-rf20 (query)                        935.38       532.88     1_468.26       0.4072          1.1271            1.1068         8.12
IVF-TQ-b2-nl316-np17-rf10 (query)                        935.38       292.07     1_227.45       0.2702          1.2133            1.2175         8.12
IVF-TQ-b2-nl316-np17-rf20 (query)                        935.38       550.31     1_485.69       0.4061          1.1277            1.1075         8.12
IVF-TQ-b2-nl316-np25-rf10 (query)                        935.38       320.92     1_256.30       0.2699          1.2136            1.2178         8.12
IVF-TQ-b2-nl316-np25-rf20 (query)                        935.38       584.82     1_520.20       0.4056          1.1279            1.1077         8.12
IVF-TQ-b2-nl316 (self)                                   935.38     1_107.68     2_043.06       0.4070          1.1561            1.1291         8.12
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_065.05       139.46     1_204.51       0.0871          1.7208            1.7599        14.05
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_065.05       157.61     1_222.66       0.0871          1.7208            1.7599        14.05
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_065.05       183.31     1_248.35       0.0871          1.7209            1.7599        14.05
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_065.05       345.46     1_410.51       0.2865          1.1965            1.2033        14.05
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_065.05       653.47     1_718.52       0.4169          1.1210            1.1109        14.05
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_065.05       363.16     1_428.21       0.2865          1.1965            1.2033        14.05
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_065.05       682.87     1_747.92       0.4169          1.1210            1.1109        14.05
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_065.05       404.87     1_469.91       0.2865          1.1965            1.2033        14.05
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_065.05       747.21     1_812.25       0.4169          1.1210            1.1109        14.05
IVF-TQ-b4-nl158 (self)                                 1_065.05     1_193.15     2_258.19       0.4165          1.1485            1.1373        14.05
IVF-TQ-b4-nl223-np11-rf0 (query)                         833.10       153.45       986.54       0.0873          1.7186            1.7581        14.26
IVF-TQ-b4-nl223-np14-rf0 (query)                         833.10       176.58     1_009.68       0.0871          1.7199            1.7595        14.26
IVF-TQ-b4-nl223-np21-rf0 (query)                         833.10       206.55     1_039.64       0.0871          1.7209            1.7599        14.26
IVF-TQ-b4-nl223-np11-rf10 (query)                        833.10       349.60     1_182.70       0.2875          1.1957            1.2026        14.26
IVF-TQ-b4-nl223-np11-rf20 (query)                        833.10       607.85     1_440.95       0.4186          1.1202            1.1101        14.26
IVF-TQ-b4-nl223-np14-rf10 (query)                        833.10       353.76     1_186.86       0.2866          1.1964            1.2033        14.26
IVF-TQ-b4-nl223-np14-rf20 (query)                        833.10       640.45     1_473.55       0.4170          1.1210            1.1108        14.26
IVF-TQ-b4-nl223-np21-rf10 (query)                        833.10       408.30     1_241.39       0.2865          1.1965            1.2033        14.26
IVF-TQ-b4-nl223-np21-rf20 (query)                        833.10       709.97     1_543.07       0.4169          1.1210            1.1109        14.26
IVF-TQ-b4-nl223 (self)                                   833.10     1_225.22     2_058.32       0.4165          1.1485            1.1373        14.26
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_031.00       163.34     1_194.34       0.0872          1.7182            1.7581        14.53
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_031.00       187.06     1_218.06       0.0872          1.7195            1.7589        14.53
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_031.00       197.05     1_228.05       0.0871          1.7208            1.7599        14.53
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_031.00       339.06     1_370.06       0.2872          1.1958            1.2028        14.53
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_031.00       588.25     1_619.25       0.4182          1.1204            1.1102        14.53
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_031.00       346.00     1_377.00       0.2868          1.1962            1.2031        14.53
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_031.00       602.47     1_633.47       0.4173          1.1208            1.1106        14.53
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_031.00       382.48     1_413.47       0.2865          1.1965            1.2033        14.53
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_031.00       650.92     1_681.92       0.4169          1.1210            1.1109        14.53
IVF-TQ-b4-nl316 (self)                                 1_031.00     1_210.60     2_241.60       0.4165          1.1485            1.1373        14.53
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 512D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.40     1_347.08     1_415.48       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.40     4_524.09     4_592.48       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              345.01       655.72     1_000.73       0.0709          1.7142            1.5716        13.97
ExhaustiveTQ-b2-rf5 (query)                              345.01       767.94     1_112.95       0.1815          1.2341            1.2618        13.97
ExhaustiveTQ-b2-rf10 (query)                             345.01       876.19     1_221.20       0.2476          1.1648            1.1811        13.97
ExhaustiveTQ-b2-rf20 (query)                             345.01     1_252.94     1_597.95       0.3618          1.1046            1.0928        13.97
ExhaustiveTQ-b2 (self)                                   345.01     4_128.85     4_473.86       0.3623          1.1225            1.1079        13.97
ExhaustiveTQ-b4-rf0 (query)                              459.17     1_153.65     1_612.82       0.0861          1.5231            1.5504        26.18
ExhaustiveTQ-b4-rf5 (query)                              459.17     1_258.31     1_717.48       0.1891          1.2262            1.2508        26.18
ExhaustiveTQ-b4-rf10 (query)                             459.17     1_394.89     1_854.07       0.2497          1.1619            1.1790        26.18
ExhaustiveTQ-b4-rf20 (query)                             459.17     1_761.67     2_220.85       0.3582          1.1058            1.1035        26.18
ExhaustiveTQ-b4 (self)                                   459.17     5_866.34     6_325.52       0.3580          1.1245            1.1246        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_793.85       190.06     1_983.91       0.0709          1.7142            1.5716        14.98
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_793.85       202.22     1_996.07       0.0709          1.7142            1.5716        14.98
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_793.85       216.32     2_010.17       0.0709          1.7142            1.5716        14.98
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_793.85       414.55     2_208.40       0.2475          1.1648            1.1811        14.98
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_793.85       760.63     2_554.48       0.3618          1.1046            1.0928        14.98
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_793.85       425.03     2_218.88       0.2475          1.1648            1.1811        14.98
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_793.85       771.69     2_565.54       0.3618          1.1046            1.0928        14.98
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_793.85       453.19     2_247.04       0.2475          1.1648            1.1811        14.98
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_793.85       807.26     2_601.11       0.3618          1.1046            1.0928        14.98
IVF-TQ-b2-nl158 (self)                                 1_793.85     1_450.21     3_244.06       0.3623          1.1225            1.1079        14.98
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_381.59       205.28     1_586.87       0.0709          1.7142            1.5716        15.21
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_381.59       211.51     1_593.10       0.0709          1.7142            1.5716        15.21
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_381.59       237.25     1_618.84       0.0709          1.7142            1.5716        15.21
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_381.59       411.02     1_792.60       0.2475          1.1648            1.1811        15.21
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_381.59       699.75     2_081.34       0.3618          1.1046            1.0928        15.21
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_381.59       418.51     1_800.10       0.2476          1.1648            1.1811        15.21
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_381.59       719.03     2_100.61       0.3618          1.1046            1.0928        15.21
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_381.59       452.71     1_834.29       0.2476          1.1648            1.1811        15.21
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_381.59       759.22     2_140.80       0.3618          1.1046            1.0928        15.21
IVF-TQ-b2-nl223 (self)                                 1_381.59     1_482.45     2_864.03       0.3622          1.1225            1.1079        15.21
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_719.05       216.66     1_935.71       0.0709          1.7135            1.5714        15.55
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_719.05       218.02     1_937.07       0.0709          1.7142            1.5716        15.55
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_719.05       238.15     1_957.20       0.0709          1.7142            1.5716        15.55
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_719.05       410.45     2_129.50       0.2476          1.1648            1.1811        15.55
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_719.05       686.80     2_405.85       0.3618          1.1046            1.0928        15.55
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_719.05       428.15     2_147.20       0.2476          1.1648            1.1811        15.55
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_719.05       698.62     2_417.67       0.3618          1.1046            1.0928        15.55
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_719.05       448.57     2_167.62       0.2476          1.1648            1.1811        15.55
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_719.05       740.18     2_459.23       0.3618          1.1046            1.0928        15.55
IVF-TQ-b2-nl316 (self)                                 1_719.05     1_510.42     3_229.47       0.3623          1.1225            1.1079        15.55
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_897.81       263.00     2_160.81       0.0861          1.5231            1.5504        27.51
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_897.81       285.29     2_183.09       0.0861          1.5231            1.5504        27.51
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_897.81       311.03     2_208.83       0.0861          1.5231            1.5504        27.51
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_897.81       518.23     2_416.03       0.2497          1.1619            1.1790        27.51
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_897.81       850.69     2_748.49       0.3582          1.1058            1.1035        27.51
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_897.81       521.77     2_419.58       0.2497          1.1619            1.1790        27.51
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_897.81       897.45     2_795.25       0.3582          1.1058            1.1035        27.51
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_897.81       579.69     2_477.49       0.2497          1.1619            1.1790        27.51
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_897.81       925.02     2_822.83       0.3582          1.1058            1.1035        27.51
IVF-TQ-b4-nl158 (self)                                 1_897.81     1_648.92     3_546.73       0.3580          1.1245            1.1246        27.51
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_496.83       284.16     1_780.99       0.0861          1.5231            1.5504        27.85
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_496.83       298.62     1_795.45       0.0861          1.5231            1.5504        27.85
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_496.83       347.11     1_843.94       0.0861          1.5231            1.5504        27.85
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_496.83       501.93     1_998.76       0.2497          1.1619            1.1790        27.85
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_496.83       797.27     2_294.10       0.3582          1.1058            1.1035        27.85
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_496.83       516.58     2_013.41       0.2497          1.1619            1.1790        27.85
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_496.83       822.79     2_319.62       0.3582          1.1058            1.1035        27.85
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_496.83       574.76     2_071.59       0.2497          1.1619            1.1790        27.85
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_496.83       891.00     2_387.83       0.3582          1.1058            1.1035        27.85
IVF-TQ-b4-nl223 (self)                                 1_496.83     1_705.97     3_202.80       0.3580          1.1245            1.1246        27.85
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_840.75       299.58     2_140.33       0.0861          1.5231            1.5504        28.33
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_840.75       307.02     2_147.77       0.0861          1.5231            1.5504        28.33
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_840.75       345.69     2_186.44       0.0861          1.5231            1.5504        28.33
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_840.75       507.44     2_348.19       0.2497          1.1619            1.1790        28.33
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_840.75       788.63     2_629.38       0.3582          1.1058            1.1035        28.33
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_840.75       524.12     2_364.87       0.2497          1.1619            1.1790        28.33
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_840.75       811.66     2_652.41       0.3582          1.1058            1.1035        28.33
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_840.75       553.67     2_394.42       0.2497          1.1619            1.1790        28.33
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_840.75       858.16     2_698.91       0.3582          1.1058            1.1035        28.33
IVF-TQ-b4-nl316 (self)                                 1_840.75     1_731.70     3_572.45       0.3580          1.1245            1.1246        28.33
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 768D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       106.37     1_967.43     2_073.80       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        106.37     6_664.07     6_770.44       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              633.46       992.48     1_625.94       0.0719          1.4569            1.4224        21.33
ExhaustiveTQ-b2-rf5 (query)                              633.46     1_082.91     1_716.37       0.1764          1.1854            1.2126        21.33
ExhaustiveTQ-b2-rf10 (query)                             633.46     1_234.91     1_868.37       0.2313          1.1364            1.1602        21.33
ExhaustiveTQ-b2-rf20 (query)                             633.46     1_633.84     2_267.30       0.3303          1.0920            1.0865        21.33
ExhaustiveTQ-b2 (self)                                   633.46     5_366.85     6_000.31       0.3296          1.1026            1.0961        21.33
ExhaustiveTQ-b4-rf0 (query)                              759.73     1_816.49     2_576.22       0.0844          1.4135            1.4092        39.64
ExhaustiveTQ-b4-rf5 (query)                              759.73     1_903.22     2_662.95       0.1812          1.1812            1.2065        39.64
ExhaustiveTQ-b4-rf10 (query)                             759.73     2_044.01     2_803.74       0.2329          1.1351            1.1562        39.64
ExhaustiveTQ-b4-rf20 (query)                             759.73     2_401.83     3_161.56       0.3263          1.0942            1.1007        39.64
ExhaustiveTQ-b4 (self)                                   759.73     7_955.63     8_715.36       0.3285          1.1030            1.1020        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        2_705.93       294.90     3_000.83       0.0719          1.4569            1.4224        22.63
IVF-TQ-b2-nl158-np12-rf0 (query)                       2_705.93       314.01     3_019.94       0.0719          1.4569            1.4224        22.63
IVF-TQ-b2-nl158-np17-rf0 (query)                       2_705.93       330.81     3_036.74       0.0719          1.4569            1.4224        22.63
IVF-TQ-b2-nl158-np7-rf10 (query)                       2_705.93       562.55     3_268.48       0.2313          1.1364            1.1602        22.63
IVF-TQ-b2-nl158-np7-rf20 (query)                       2_705.93       935.78     3_641.71       0.3303          1.0920            1.0865        22.63
IVF-TQ-b2-nl158-np12-rf10 (query)                      2_705.93       572.52     3_278.45       0.2313          1.1364            1.1602        22.63
IVF-TQ-b2-nl158-np12-rf20 (query)                      2_705.93       987.26     3_693.19       0.3303          1.0920            1.0865        22.63
IVF-TQ-b2-nl158-np17-rf10 (query)                      2_705.93       595.39     3_301.32       0.2313          1.1364            1.1602        22.63
IVF-TQ-b2-nl158-np17-rf20 (query)                      2_705.93     1_058.61     3_764.55       0.3303          1.0920            1.0865        22.63
IVF-TQ-b2-nl158 (self)                                 2_705.93     1_904.53     4_610.46       0.3296          1.1026            1.0961        22.63
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_990.62       312.12     2_302.74       0.0719          1.4569            1.4223        22.96
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_990.62       325.17     2_315.79       0.0719          1.4569            1.4224        22.96
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_990.62       369.99     2_360.61       0.0719          1.4569            1.4224        22.96
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_990.62       565.42     2_556.05       0.2313          1.1364            1.1602        22.96
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_990.62       878.13     2_868.76       0.3303          1.0920            1.0865        22.96
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_990.62       574.33     2_564.95       0.2313          1.1364            1.1602        22.96
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_990.62       890.80     2_881.43       0.3303          1.0920            1.0865        22.96
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_990.62       604.57     2_595.19       0.2313          1.1364            1.1602        22.96
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_990.62       941.69     2_932.32       0.3303          1.0920            1.0865        22.96
IVF-TQ-b2-nl223 (self)                                 1_990.62     1_937.50     3_928.13       0.3296          1.1026            1.0961        22.96
IVF-TQ-b2-nl316-np15-rf0 (query)                       2_561.72       330.26     2_891.98       0.0719          1.4568            1.4224        23.53
IVF-TQ-b2-nl316-np17-rf0 (query)                       2_561.72       336.97     2_898.69       0.0719          1.4569            1.4224        23.53
IVF-TQ-b2-nl316-np25-rf0 (query)                       2_561.72       362.67     2_924.38       0.0719          1.4569            1.4224        23.53
IVF-TQ-b2-nl316-np15-rf10 (query)                      2_561.72       556.64     3_118.36       0.2313          1.1364            1.1602        23.53
IVF-TQ-b2-nl316-np15-rf20 (query)                      2_561.72       843.47     3_405.19       0.3303          1.0920            1.0865        23.53
IVF-TQ-b2-nl316-np17-rf10 (query)                      2_561.72       565.05     3_126.76       0.2313          1.1364            1.1602        23.53
IVF-TQ-b2-nl316-np17-rf20 (query)                      2_561.72       865.97     3_427.68       0.3303          1.0920            1.0865        23.53
IVF-TQ-b2-nl316-np25-rf10 (query)                      2_561.72       597.53     3_159.25       0.2313          1.1364            1.1602        23.53
IVF-TQ-b2-nl316-np25-rf20 (query)                      2_561.72       901.76     3_463.47       0.3303          1.0920            1.0865        23.53
IVF-TQ-b2-nl316 (self)                                 2_561.72     1_976.53     4_538.25       0.3296          1.1026            1.0961        23.53
IVF-TQ-b4-nl158-np7-rf0 (query)                        2_775.61       420.88     3_196.49       0.0844          1.4135            1.4092        41.40
IVF-TQ-b4-nl158-np12-rf0 (query)                       2_775.61       452.50     3_228.11       0.0844          1.4135            1.4092        41.40
IVF-TQ-b4-nl158-np17-rf0 (query)                       2_775.61       484.23     3_259.84       0.0844          1.4135            1.4092        41.40
IVF-TQ-b4-nl158-np7-rf10 (query)                       2_775.61       689.60     3_465.21       0.2329          1.1351            1.1562        41.40
IVF-TQ-b4-nl158-np7-rf20 (query)                       2_775.61     1_078.38     3_853.99       0.3263          1.0942            1.1007        41.40
IVF-TQ-b4-nl158-np12-rf10 (query)                      2_775.61       722.07     3_497.68       0.2329          1.1351            1.1562        41.40
IVF-TQ-b4-nl158-np12-rf20 (query)                      2_775.61     1_133.31     3_908.92       0.3263          1.0942            1.1007        41.40
IVF-TQ-b4-nl158-np17-rf10 (query)                      2_775.61       768.56     3_544.17       0.2329          1.1351            1.1562        41.40
IVF-TQ-b4-nl158-np17-rf20 (query)                      2_775.61     1_179.47     3_955.08       0.3263          1.0942            1.1007        41.40
IVF-TQ-b4-nl158 (self)                                 2_775.61     2_196.61     4_972.22       0.3285          1.1030            1.1019        41.40
IVF-TQ-b4-nl223-np11-rf0 (query)                       2_180.65       444.28     2_624.93       0.0844          1.4135            1.4092        41.87
IVF-TQ-b4-nl223-np14-rf0 (query)                       2_180.65       510.59     2_691.24       0.0844          1.4135            1.4092        41.87
IVF-TQ-b4-nl223-np21-rf0 (query)                       2_180.65       521.58     2_702.23       0.0844          1.4135            1.4092        41.87
IVF-TQ-b4-nl223-np11-rf10 (query)                      2_180.65       684.20     2_864.85       0.2329          1.1351            1.1562        41.87
IVF-TQ-b4-nl223-np11-rf20 (query)                      2_180.65     1_002.60     3_183.25       0.3263          1.0942            1.1007        41.87
IVF-TQ-b4-nl223-np14-rf10 (query)                      2_180.65       705.99     2_886.64       0.2329          1.1351            1.1562        41.87
IVF-TQ-b4-nl223-np14-rf20 (query)                      2_180.65     1_037.95     3_218.60       0.3263          1.0942            1.1007        41.87
IVF-TQ-b4-nl223-np21-rf10 (query)                      2_180.65       773.74     2_954.39       0.2329          1.1351            1.1562        41.87
IVF-TQ-b4-nl223-np21-rf20 (query)                      2_180.65     1_119.36     3_300.01       0.3263          1.0942            1.1007        41.87
IVF-TQ-b4-nl223 (self)                                 2_180.65     2_279.22     4_459.87       0.3285          1.1030            1.1020        41.87
IVF-TQ-b4-nl316-np15-rf0 (query)                       2_778.86       487.90     3_266.75       0.0844          1.4131            1.4090        42.73
IVF-TQ-b4-nl316-np17-rf0 (query)                       2_778.86       502.66     3_281.51       0.0844          1.4135            1.4092        42.73
IVF-TQ-b4-nl316-np25-rf0 (query)                       2_778.86       541.75     3_320.60       0.0844          1.4135            1.4092        42.73
IVF-TQ-b4-nl316-np15-rf10 (query)                      2_778.86       702.40     3_481.25       0.2329          1.1351            1.1562        42.73
IVF-TQ-b4-nl316-np15-rf20 (query)                      2_778.86       995.82     3_774.67       0.3263          1.0942            1.1006        42.73
IVF-TQ-b4-nl316-np17-rf10 (query)                      2_778.86       709.89     3_488.75       0.2329          1.1351            1.1562        42.73
IVF-TQ-b4-nl316-np17-rf20 (query)                      2_778.86     1_021.33     3_800.19       0.3263          1.0942            1.1007        42.73
IVF-TQ-b4-nl316-np25-rf10 (query)                      2_778.86       766.43     3_545.29       0.2329          1.1351            1.1562        42.73
IVF-TQ-b4-nl316-np25-rf20 (query)                      2_778.86     1_080.53     3_859.39       0.3263          1.0942            1.1007        42.73
IVF-TQ-b4-nl316 (self)                                 2_778.86     2_336.15     5_115.01       0.3285          1.1030            1.1019        42.73
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Cell embeddings data

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 256D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.81       714.52       747.33       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.81     2_415.36     2_448.17       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              144.87       366.20       511.07       0.7918          1.0898            1.0632         7.12
ExhaustiveTQ-b2-rf5 (query)                              144.87       446.43       591.30       0.9995          1.0000            1.0000         7.12
ExhaustiveTQ-b2-rf10 (query)                             144.87       586.47       731.34       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b2-rf20 (query)                             144.87       964.90     1_109.77       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b2 (self)                                   144.87     3_188.38     3_333.25       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b4-rf0 (query)                              230.04       586.92       816.96       0.8727          1.0322            1.0183        13.22
ExhaustiveTQ-b4-rf5 (query)                              230.04       669.81       899.85       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4-rf10 (query)                             230.04       804.25     1_034.28       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4-rf20 (query)                             230.04     1_184.64     1_414.68       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4 (self)                                   230.04     3_940.92     4_170.96       1.0000          1.0000            1.0000        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_070.23       130.00     1_200.23       0.7916          1.0897            1.0635         7.78
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_070.23       174.11     1_244.33       0.7918          1.0898            1.0632         7.78
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_070.23       213.12     1_283.35       0.7918          1.0898            1.0632         7.78
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_070.23       336.39     1_406.61       0.9981          1.0004            1.0000         7.78
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_070.23       621.26     1_691.49       0.9982          1.0004            1.0000         7.78
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_070.23       401.46     1_471.69       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_070.23       716.37     1_786.60       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_070.23       450.98     1_521.21       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_070.23       797.12     1_867.35       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158 (self)                                 1_070.23     1_206.07     2_276.30       0.9999          1.0000            1.0000         7.78
IVF-TQ-b2-nl223-np11-rf0 (query)                         619.15       131.23       750.38       0.7919          1.0897            1.0632         7.93
IVF-TQ-b2-nl223-np14-rf0 (query)                         619.15       147.16       766.31       0.7918          1.0897            1.0632         7.93
IVF-TQ-b2-nl223-np21-rf0 (query)                         619.15       184.32       803.46       0.7918          1.0898            1.0632         7.93
IVF-TQ-b2-nl223-np11-rf10 (query)                        619.15       326.89       946.04       0.9995          1.0001            1.0000         7.93
IVF-TQ-b2-nl223-np11-rf20 (query)                        619.15       595.99     1_215.14       0.9995          1.0001            1.0000         7.93
IVF-TQ-b2-nl223-np14-rf10 (query)                        619.15       350.59       969.74       0.9999          1.0000            1.0000         7.93
IVF-TQ-b2-nl223-np14-rf20 (query)                        619.15       634.02     1_253.16       0.9999          1.0000            1.0000         7.93
IVF-TQ-b2-nl223-np21-rf10 (query)                        619.15       402.46     1_021.60       1.0000          1.0000            1.0000         7.93
IVF-TQ-b2-nl223-np21-rf20 (query)                        619.15       719.43     1_338.58       1.0000          1.0000            1.0000         7.93
IVF-TQ-b2-nl223 (self)                                   619.15     1_062.64     1_681.79       1.0000          1.0000            1.0000         7.93
IVF-TQ-b2-nl316-np15-rf0 (query)                         811.83       133.57       945.40       0.7918          1.0897            1.0632         8.12
IVF-TQ-b2-nl316-np17-rf0 (query)                         811.83       142.62       954.45       0.7918          1.0898            1.0632         8.12
IVF-TQ-b2-nl316-np25-rf0 (query)                         811.83       173.89       985.72       0.7918          1.0898            1.0632         8.12
IVF-TQ-b2-nl316-np15-rf10 (query)                        811.83       313.84     1_125.67       0.9997          1.0000            1.0000         8.12
IVF-TQ-b2-nl316-np15-rf20 (query)                        811.83       574.42     1_386.25       0.9997          1.0000            1.0000         8.12
IVF-TQ-b2-nl316-np17-rf10 (query)                        811.83       327.44     1_139.27       0.9999          1.0000            1.0000         8.12
IVF-TQ-b2-nl316-np17-rf20 (query)                        811.83       598.52     1_410.35       0.9999          1.0000            1.0000         8.12
IVF-TQ-b2-nl316-np25-rf10 (query)                        811.83       388.58     1_200.41       1.0000          1.0000            1.0000         8.12
IVF-TQ-b2-nl316-np25-rf20 (query)                        811.83       667.21     1_479.04       1.0000          1.0000            1.0000         8.12
IVF-TQ-b2-nl316 (self)                                   811.83     1_018.74     1_830.57       1.0000          1.0000            1.0000         8.12
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_137.42       183.76     1_321.18       0.8721          1.0325            1.0187        14.02
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_137.42       258.34     1_395.76       0.8727          1.0322            1.0183        14.02
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_137.42       320.91     1_458.33       0.8727          1.0322            1.0183        14.02
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_137.42       412.78     1_550.20       0.9981          1.0004            1.0000        14.02
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_137.42       676.99     1_814.41       0.9982          1.0004            1.0000        14.02
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_137.42       486.26     1_623.68       1.0000          1.0000            1.0000        14.02
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_137.42       812.19     1_949.61       1.0000          1.0000            1.0000        14.02
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_137.42       559.45     1_696.87       1.0000          1.0000            1.0000        14.02
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_137.42       904.91     2_042.33       1.0000          1.0000            1.0000        14.02
IVF-TQ-b4-nl158 (self)                                 1_137.42     1_342.50     2_479.92       0.9999          1.0000            1.0000        14.02
IVF-TQ-b4-nl223-np11-rf0 (query)                         720.25       189.51       909.77       0.8726          1.0323            1.0184        14.23
IVF-TQ-b4-nl223-np14-rf0 (query)                         720.25       210.91       931.16       0.8727          1.0322            1.0183        14.23
IVF-TQ-b4-nl223-np21-rf0 (query)                         720.25       274.47       994.72       0.8727          1.0322            1.0183        14.23
IVF-TQ-b4-nl223-np11-rf10 (query)                        720.25       376.54     1_096.79       0.9995          1.0001            1.0000        14.23
IVF-TQ-b4-nl223-np11-rf20 (query)                        720.25       645.65     1_365.90       0.9995          1.0001            1.0000        14.23
IVF-TQ-b4-nl223-np14-rf10 (query)                        720.25       418.07     1_138.32       0.9999          1.0000            1.0000        14.23
IVF-TQ-b4-nl223-np14-rf20 (query)                        720.25       702.59     1_422.84       0.9999          1.0000            1.0000        14.23
IVF-TQ-b4-nl223-np21-rf10 (query)                        720.25       495.95     1_216.20       1.0000          1.0000            1.0000        14.23
IVF-TQ-b4-nl223-np21-rf20 (query)                        720.25       819.67     1_539.92       1.0000          1.0000            1.0000        14.23
IVF-TQ-b4-nl223 (self)                                   720.25     1_192.08     1_912.33       1.0000          1.0000            1.0000        14.23
IVF-TQ-b4-nl316-np15-rf0 (query)                         892.03       186.54     1_078.57       0.8727          1.0322            1.0184        14.54
IVF-TQ-b4-nl316-np17-rf0 (query)                         892.03       204.18     1_096.21       0.8727          1.0322            1.0183        14.54
IVF-TQ-b4-nl316-np25-rf0 (query)                         892.03       255.09     1_147.12       0.8727          1.0322            1.0183        14.54
IVF-TQ-b4-nl316-np15-rf10 (query)                        892.03       367.91     1_259.94       0.9997          1.0000            1.0000        14.54
IVF-TQ-b4-nl316-np15-rf20 (query)                        892.03       632.59     1_524.62       0.9997          1.0000            1.0000        14.54
IVF-TQ-b4-nl316-np17-rf10 (query)                        892.03       385.70     1_277.73       0.9999          1.0000            1.0000        14.54
IVF-TQ-b4-nl316-np17-rf20 (query)                        892.03       660.77     1_552.80       0.9999          1.0000            1.0000        14.54
IVF-TQ-b4-nl316-np25-rf10 (query)                        892.03       452.26     1_344.29       1.0000          1.0000            1.0000        14.54
IVF-TQ-b4-nl316-np25-rf20 (query)                        892.03       756.88     1_648.91       1.0000          1.0000            1.0000        14.54
IVF-TQ-b4-nl316 (self)                                   892.03     1_129.12     2_021.15       1.0000          1.0000            1.0000        14.54
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 512D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.76     1_359.28     1_428.05       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.76     4_524.56     4_593.32       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              352.08       670.05     1_022.13       0.8424          1.0447            1.0331        13.97
ExhaustiveTQ-b2-rf5 (query)                              352.08       745.63     1_097.71       0.9999          1.0000            1.0000        13.97
ExhaustiveTQ-b2-rf10 (query)                             352.08       900.38     1_252.45       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2-rf20 (query)                             352.08     1_291.74     1_643.82       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2 (self)                                   352.08     4_214.40     4_566.48       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b4-rf0 (query)                              458.18     1_151.40     1_609.58       0.8985          1.0191            1.0110        26.18
ExhaustiveTQ-b4-rf5 (query)                              458.18     1_247.28     1_705.45       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4-rf10 (query)                             458.18     1_387.37     1_845.55       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4-rf20 (query)                             458.18     1_769.17     2_227.35       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4 (self)                                   458.18     5_840.82     6_299.00       1.0000          1.0000            1.0000        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        2_147.34       236.96     2_384.30       0.8420          1.0449            1.0333        14.96
IVF-TQ-b2-nl158-np12-rf0 (query)                       2_147.34       301.73     2_449.07       0.8424          1.0447            1.0331        14.96
IVF-TQ-b2-nl158-np17-rf0 (query)                       2_147.34       372.34     2_519.68       0.8424          1.0447            1.0331        14.96
IVF-TQ-b2-nl158-np7-rf10 (query)                       2_147.34       474.29     2_621.63       0.9986          1.0003            1.0000        14.96
IVF-TQ-b2-nl158-np7-rf20 (query)                       2_147.34       770.91     2_918.25       0.9986          1.0003            1.0000        14.96
IVF-TQ-b2-nl158-np12-rf10 (query)                      2_147.34       549.61     2_696.95       0.9999          1.0000            1.0000        14.96
IVF-TQ-b2-nl158-np12-rf20 (query)                      2_147.34       890.92     3_038.26       0.9999          1.0000            1.0000        14.96
IVF-TQ-b2-nl158-np17-rf10 (query)                      2_147.34       624.01     2_771.35       0.9999          1.0000            1.0000        14.96
IVF-TQ-b2-nl158-np17-rf20 (query)                      2_147.34       990.24     3_137.59       0.9999          1.0000            1.0000        14.96
IVF-TQ-b2-nl158 (self)                                 2_147.34     1_570.35     3_717.69       1.0000          1.0000            1.0000        14.96
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_149.77       234.36     1_384.13       0.8423          1.0447            1.0331        15.25
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_149.77       263.82     1_413.59       0.8424          1.0447            1.0331        15.25
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_149.77       325.15     1_474.92       0.8424          1.0447            1.0331        15.25
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_149.77       467.63     1_617.40       0.9997          1.0000            1.0000        15.25
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_149.77       744.67     1_894.44       0.9997          1.0000            1.0000        15.25
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_149.77       481.48     1_631.26       0.9999          1.0000            1.0000        15.25
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_149.77       789.21     1_938.98       0.9999          1.0000            1.0000        15.25
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_149.77       552.13     1_701.91       1.0000          1.0000            1.0000        15.25
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_149.77       877.96     2_027.74       1.0000          1.0000            1.0000        15.25
IVF-TQ-b2-nl223 (self)                                 1_149.77     1_479.62     2_629.39       1.0000          1.0000            1.0000        15.25
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_395.93       247.01     1_642.94       0.8424          1.0447            1.0331        15.56
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_395.93       266.24     1_662.17       0.8424          1.0447            1.0331        15.56
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_395.93       314.93     1_710.86       0.8424          1.0447            1.0331        15.56
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_395.93       455.32     1_851.25       0.9999          1.0000            1.0000        15.56
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_395.93       764.72     2_160.65       0.9999          1.0000            1.0000        15.56
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_395.93       478.85     1_874.78       0.9999          1.0000            1.0000        15.56
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_395.93       870.23     2_266.16       0.9999          1.0000            1.0000        15.56
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_395.93       637.66     2_033.59       1.0000          1.0000            1.0000        15.56
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_395.93       954.19     2_350.12       1.0000          1.0000            1.0000        15.56
IVF-TQ-b2-nl316 (self)                                 1_395.93     1_578.81     2_974.74       1.0000          1.0000            1.0000        15.56
IVF-TQ-b4-nl158-np7-rf0 (query)                        2_478.19       337.86     2_816.04       0.8977          1.0194            1.0113        27.46
IVF-TQ-b4-nl158-np12-rf0 (query)                       2_478.19       472.31     2_950.50       0.8985          1.0191            1.0110        27.46
IVF-TQ-b4-nl158-np17-rf0 (query)                       2_478.19       573.46     3_051.65       0.8985          1.0191            1.0110        27.46
IVF-TQ-b4-nl158-np7-rf10 (query)                       2_478.19       572.76     3_050.94       0.9986          1.0003            1.0000        27.46
IVF-TQ-b4-nl158-np7-rf20 (query)                       2_478.19       884.04     3_362.23       0.9986          1.0003            1.0000        27.46
IVF-TQ-b4-nl158-np12-rf10 (query)                      2_478.19       711.47     3_189.65       0.9999          1.0000            1.0000        27.46
IVF-TQ-b4-nl158-np12-rf20 (query)                      2_478.19     1_056.24     3_534.43       0.9999          1.0000            1.0000        27.46
IVF-TQ-b4-nl158-np17-rf10 (query)                      2_478.19       827.40     3_305.59       0.9999          1.0000            1.0000        27.46
IVF-TQ-b4-nl158-np17-rf20 (query)                      2_478.19     1_190.69     3_668.88       0.9999          1.0000            1.0000        27.46
IVF-TQ-b4-nl158 (self)                                 2_478.19     1_871.64     4_349.82       1.0000          1.0000            1.0000        27.46
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_259.90       353.82     1_613.72       0.8984          1.0191            1.0111        27.91
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_259.90       401.73     1_661.63       0.8985          1.0191            1.0110        27.91
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_259.90       519.52     1_779.42       0.8985          1.0191            1.0110        27.91
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_259.90       567.37     1_827.27       0.9997          1.0000            1.0000        27.91
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_259.90       868.73     2_128.63       0.9997          1.0000            1.0000        27.91
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_259.90       621.27     1_881.17       0.9999          1.0000            1.0000        27.91
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_259.90       939.84     2_199.74       0.9999          1.0000            1.0000        27.91
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_259.90       741.82     2_001.72       1.0000          1.0000            1.0000        27.91
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_259.90     1_063.80     2_323.70       1.0000          1.0000            1.0000        27.91
IVF-TQ-b4-nl223 (self)                                 1_259.90     1_768.32     3_028.22       1.0000          1.0000            1.0000        27.91
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_502.03       356.74     1_858.77       0.8985          1.0191            1.0110        28.36
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_502.03       387.04     1_889.07       0.8985          1.0191            1.0110        28.36
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_502.03       480.06     1_982.09       0.8985          1.0191            1.0110        28.36
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_502.03       571.36     2_073.39       0.9999          1.0000            1.0000        28.36
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_502.03       872.36     2_374.39       0.9999          1.0000            1.0000        28.36
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_502.03       589.33     2_091.37       0.9999          1.0000            1.0000        28.36
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_502.03       916.11     2_418.14       0.9999          1.0000            1.0000        28.36
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_502.03       700.11     2_202.15       1.0000          1.0000            1.0000        28.36
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_502.03     1_034.71     2_536.74       1.0000          1.0000            1.0000        28.36
IVF-TQ-b4-nl316 (self)                                 1_502.03     1_717.20     3_219.23       1.0000          1.0000            1.0000        28.36
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 768D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       104.88     2_011.52     2_116.40       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        104.88     6_736.76     6_841.64       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              617.89     1_035.23     1_653.12       0.8736          1.0271            1.0199        21.33
ExhaustiveTQ-b2-rf5 (query)                              617.89     1_101.82     1_719.71       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2-rf10 (query)                             617.89     1_241.21     1_859.10       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2-rf20 (query)                             617.89     1_650.10     2_267.99       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2 (self)                                   617.89     5_406.82     6_024.71       0.9999          1.0000            1.0000        21.33
ExhaustiveTQ-b4-rf0 (query)                              762.60     1_817.26     2_579.86       0.9097          1.0146            1.0083        39.64
ExhaustiveTQ-b4-rf5 (query)                              762.60     1_916.94     2_679.54       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4-rf10 (query)                             762.60     2_029.22     2_791.82       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4-rf20 (query)                             762.60     2_412.89     3_175.49       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4 (self)                                   762.60     7_954.93     8_717.53       0.9999          1.0000            1.0000        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        3_145.40       360.05     3_505.45       0.8735          1.0272            1.0201        22.61
IVF-TQ-b2-nl158-np12-rf0 (query)                       3_145.40       463.87     3_609.27       0.8736          1.0271            1.0199        22.61
IVF-TQ-b2-nl158-np17-rf0 (query)                       3_145.40       542.55     3_687.96       0.8736          1.0271            1.0199        22.61
IVF-TQ-b2-nl158-np7-rf10 (query)                       3_145.40       626.16     3_771.57       0.9995          1.0001            1.0000        22.61
IVF-TQ-b2-nl158-np7-rf20 (query)                       3_145.40       933.93     4_079.33       0.9995          1.0001            1.0000        22.61
IVF-TQ-b2-nl158-np12-rf10 (query)                      3_145.40       719.96     3_865.36       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158-np12-rf20 (query)                      3_145.40     1_078.86     4_224.27       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158-np17-rf10 (query)                      3_145.40       812.73     3_958.13       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158-np17-rf20 (query)                      3_145.40     1_182.48     4_327.89       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158 (self)                                 3_145.40     2_076.66     5_222.06       0.9999          1.0000            1.0000        22.61
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_705.38       352.55     2_057.93       0.8736          1.0271            1.0200        23.01
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_705.38       392.33     2_097.72       0.8736          1.0271            1.0199        23.01
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_705.38       482.27     2_187.66       0.8736          1.0271            1.0199        23.01
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_705.38       593.57     2_298.95       0.9998          1.0000            1.0000        23.01
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_705.38       915.41     2_620.80       0.9998          1.0000            1.0000        23.01
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_705.38       636.17     2_341.55       0.9999          1.0000            1.0000        23.01
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_705.38       974.71     2_680.09       0.9999          1.0000            1.0000        23.01
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_705.38       736.36     2_441.75       1.0000          1.0000            1.0000        23.01
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_705.38     1_109.27     2_814.65       1.0000          1.0000            1.0000        23.01
IVF-TQ-b2-nl223 (self)                                 1_705.38     1_977.37     3_682.76       0.9999          1.0000            1.0000        23.01
IVF-TQ-b2-nl316-np15-rf0 (query)                       2_004.78       360.79     2_365.57       0.8736          1.0271            1.0200        23.53
IVF-TQ-b2-nl316-np17-rf0 (query)                       2_004.78       383.78     2_388.56       0.8736          1.0271            1.0200        23.53
IVF-TQ-b2-nl316-np25-rf0 (query)                       2_004.78       456.18     2_460.95       0.8736          1.0271            1.0199        23.53
IVF-TQ-b2-nl316-np15-rf10 (query)                      2_004.78       595.45     2_600.23       0.9999          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np15-rf20 (query)                      2_004.78       921.28     2_926.06       0.9999          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np17-rf10 (query)                      2_004.78       616.72     2_621.50       0.9999          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np17-rf20 (query)                      2_004.78       957.71     2_962.48       0.9999          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np25-rf10 (query)                      2_004.78       698.58     2_703.36       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np25-rf20 (query)                      2_004.78     1_064.85     3_069.62       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316 (self)                                 2_004.78     1_941.16     3_945.94       0.9999          1.0000            1.0000        23.53
IVF-TQ-b4-nl158-np7-rf0 (query)                        3_204.33       529.02     3_733.35       0.9094          1.0147            1.0084        41.37
IVF-TQ-b4-nl158-np12-rf0 (query)                       3_204.33       708.10     3_912.44       0.9097          1.0146            1.0083        41.37
IVF-TQ-b4-nl158-np17-rf0 (query)                       3_204.33       859.82     4_064.15       0.9097          1.0146            1.0083        41.37
IVF-TQ-b4-nl158-np7-rf10 (query)                       3_204.33       786.46     3_990.79       0.9995          1.0001            1.0000        41.37
IVF-TQ-b4-nl158-np7-rf20 (query)                       3_204.33     1_111.54     4_315.87       0.9995          1.0001            1.0000        41.37
IVF-TQ-b4-nl158-np12-rf10 (query)                      3_204.33       982.94     4_187.27       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158-np12-rf20 (query)                      3_204.33     1_343.20     4_547.53       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158-np17-rf10 (query)                      3_204.33     1_134.50     4_338.84       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158-np17-rf20 (query)                      3_204.33     1_506.73     4_711.06       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158 (self)                                 3_204.33     2_639.49     5_843.82       0.9999          1.0000            1.0000        41.37
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_906.07       533.92     2_439.99       0.9096          1.0146            1.0084        41.97
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_906.07       616.20     2_522.27       0.9097          1.0146            1.0083        41.97
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_906.07       784.98     2_691.05       0.9097          1.0146            1.0083        41.97
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_906.07       801.98     2_708.05       0.9998          1.0000            1.0000        41.97
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_906.07     1_085.35     2_991.42       0.9998          1.0000            1.0000        41.97
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_906.07       850.83     2_756.90       0.9999          1.0000            1.0000        41.97
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_906.07     1_192.75     3_098.83       0.9999          1.0000            1.0000        41.97
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_906.07     1_026.66     2_932.73       1.0000          1.0000            1.0000        41.97
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_906.07     1_371.09     3_277.16       1.0000          1.0000            1.0000        41.97
IVF-TQ-b4-nl223 (self)                                 1_906.07     2_478.47     4_384.54       0.9999          1.0000            1.0000        41.97
IVF-TQ-b4-nl316-np15-rf0 (query)                       2_200.24       560.37     2_760.61       0.9097          1.0146            1.0083        42.73
IVF-TQ-b4-nl316-np17-rf0 (query)                       2_200.24       596.71     2_796.95       0.9097          1.0146            1.0083        42.73
IVF-TQ-b4-nl316-np25-rf0 (query)                       2_200.24       745.11     2_945.35       0.9097          1.0146            1.0083        42.73
IVF-TQ-b4-nl316-np15-rf10 (query)                      2_200.24       781.05     2_981.29       0.9999          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np15-rf20 (query)                      2_200.24     1_093.39     3_293.63       0.9999          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np17-rf10 (query)                      2_200.24       814.47     3_014.71       0.9999          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np17-rf20 (query)                      2_200.24     1_136.44     3_336.68       0.9999          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np25-rf10 (query)                      2_200.24     1_009.51     3_209.75       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np25-rf20 (query)                      2_200.24     1_319.40     3_519.63       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316 (self)                                 2_200.24     2_444.16     4_644.40       0.9999          1.0000            1.0000        42.73
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
