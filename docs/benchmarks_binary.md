## Binarised indices benchmarks and parameter

Binarised indices push the compression to (roughly) bits. Three consequences:

1. The index footprint collapses.
2. Queries usually get faster, because bitwise operations are cheap on modern
   CPUs.
3. Without re-ranking the top candidates, recall drops hard. Less so for RaBitQ,
   and for TurboQuant it depends on the data.

The two graph indices split on that first point. The quantised graph (QG)
spends its bits on locality rather than on compression and comes out *larger*
than the raw vectors, so read it as a speed structure that happens to use
RaBitQ codes. HNSW-RaBitQ goes the other way and drops the vectors entirely.
Both live in their own section below.

The benchmarks below show both, with and without re-ranking. For the simple
binary versions use:

```bash
cargo run --example gridsearch_binary --release --features binary -- --dim 512 --n-samples 50000 --data embedding
```

For RaBitQ:

```bash
cargo run --example gridsearch_rabitq --release --features binary -- --dim 512 --n-samples 50000 --data embedding
```

For the two RaBitQ graph indices, which need both feature flags:

```bash
cargo run --example gridsearch_rabitq_graphs --release --features binary,quantised -- --dim 512 --n-samples 50000 --data embedding
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
- [RaBitQ graphs](#rabitq-graphs-qg-and-hnsw-rabitq)
  - [Quantised graph](#quantised-graph-qg)
  - [HNSW-RaBitQ](#hnsw-rabitq)
- [TurboQuant](#turboquant-ivf-and-exhaustive)

### <u>Binary (IVF and exhaustive)</u>

Three binarisations are offered in this crate:

- **SimHash**: Projects vectors onto random hyperplanes and encodes the sign of
  each projection as a bit. The random planes are orthogonalised to improve
  coverage of the vector space. The training data is only used to fit a
  per-feature mean: the hyperplanes pass through the origin, so on data sitting
  far from it every bit would otherwise land on the same side of every plane.
- **PCA Hashing**: Fits PCA on the (centred) training data and takes the sign of
  each point's score on a principal component as a bit. Only the leading
  components that cumulatively explain 90% of the variance are kept, and that
  count is capped at a sixteenth of the bit budget. The retained block is then
  rotated by ITQ (Gong and Lazebnik, "Iterative Quantization: A Procrustean
  Approach to Learning Binary Codes", CVPR 2011), which spreads variance evenly
  across those bits: raw PCA loadings pile nearly all of it into the first few
  components, leaving the trailing sign bits decided by rounding noise while
  they still count for a full unit of Hamming distance.

  Every bit past the retained block is a random orthogonal hyperplane, and that
  padding is the normal case rather than an edge case. At 512 bits at most 32
  are PCA bits, whatever the dimensionality. The cap is deliberate: past the
  genuinely structured directions a random hyperplane beats a PCA one, because
  it preserves angular distance by construction and a low-variance loading does
  not.

  More expensive to build than SimHash. Whether the data-adapted bits actually
  buy recall depends on the spectrum, so read it off the tables below rather
  than assuming they do.
- **Sign-based**: Simply encodes the sign of each embedding dimension directly
  as a bit, meaning `n_bits` is fixed to the number of dimensions.
  Straightforward but only sensible for high-dimensional data; at low
  dimensionality the recall degrades dramatically. Codes live in one global
  frame, on the IVF index too, so Hamming distances compare across Voronoi
  cells and widening `nprobe` can only add candidates.

These indices can keep the original vectors in a `VecStore` on disk for
re-ranking. Recommended if you want the recall to stay usable. Their home ground
is very high-dimensional data where memory is the binding constraint.

**Tunable parameters *(general)*:**

- *n_bits*: How many bits to encode each vector into. More bits, better recall,
  bigger index. For `"pca"` it also sets how many principal components can be
  spent, since the retained count is capped at `n_bits / 16`. The grid runs
  `256` and `512`, plus `1024` past 128 dimensions, each with `"random"` and
  `"pca"`. `"sign"` ignores the value and always emits `dim` bits, so it gets
  one row rather than a sweep.
- *binarisation_init*: Three options are provided in the crate. `"random"` for
  random planes that are subsequently orthogonalised, `"pca"` to identify axes
  of maximum variation, or `"sign"` to just use the sign of the respective
  embedding dimensions. In that last case
  `n_bits` is set automatically to `n_dim`. Sign-based only really makes sense
  if you have a lot of dimensions; otherwise the performance is not great (at
  all). Unrecognised strings print a warning and fall back to `"random"`, so
  watch the spelling: `"random_projections"`, `"pca_hashing"` and `"sign_based"`
  are the accepted long forms, and `"signed"` is not one of them.
- *reranking_factor*: Hamming distance picks the candidates, then the on-disk
  vectors are loaded and the candidates re-scored exactly. The factor is how
  many more than `k` get re-scored, so `10` means `10 * k` vectors. More
  candidates, better recall. Default `20`; the grid runs `10` and `20`, plus a
  `no_rr` row with no re-ranking at all.

**Tunable parameters *(IVF-specific)*:**

- *Number of lists (nl)*: Number of k-means clusters, `sqrt(n)` as a default.
  The grid runs `sqrt(n/2)`, `sqrt(n)` and `sqrt(2n)`.
- *Number of probes (np)*: `sqrt(nlist)`, `sqrt(2 * nlist)` and 5% of `nlist`,
  deduplicated.

Self queries run with `reranking_factor = 10`, the IVF ones at
`np = sqrt(nlist)`.

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
Exhaustive (query)                                        33.38       699.01       732.39       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.38     2_281.18     2_314.56       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                 82.04       240.20       322.24       0.1199          1.4617            1.4199         1.78
ExhaustiveBinary-256-random-rf10 (query)                  82.04       336.72       418.76       0.3411          1.0941            1.0814         1.78
ExhaustiveBinary-256-random-rf20 (query)                  82.04       435.49       517.53       0.4467          1.0571            1.0475         1.78
ExhaustiveBinary-256-random (self)                        82.04     1_120.38     1_202.41       0.3454          1.0895            1.0798         1.78
ExhaustiveBinary-256-pca_no_rr (query)                   111.02       239.78       350.79       0.1153          1.4748            1.4212         1.78
ExhaustiveBinary-256-pca-rf10 (query)                    111.02       358.56       469.57       0.3323          1.1029            1.0834         1.78
ExhaustiveBinary-256-pca-rf20 (query)                    111.02       438.98       550.00       0.4387          1.0631            1.0485         1.78
ExhaustiveBinary-256-pca (self)                          111.02     1_085.79     1_196.81       0.3391          1.0957            1.0813         1.78
ExhaustiveBinary-512-random_no_rr (query)                 90.34       357.82       448.16       0.1588          1.3547            1.3300         3.55
ExhaustiveBinary-512-random-rf10 (query)                  90.34       472.11       562.46       0.3786          1.0692            1.0677         3.55
ExhaustiveBinary-512-random-rf20 (query)                  90.34       576.59       666.93       0.4874          1.0424            1.0395         3.55
ExhaustiveBinary-512-random (self)                        90.34     1_527.84     1_618.18       0.3805          1.0675            1.0675         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   125.81       359.57       485.38       0.1564          1.3535            1.3265         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    125.81       463.98       589.79       0.3789          1.0710            1.0663         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    125.81       582.09       707.90       0.4903          1.0433            1.0387         3.55
ExhaustiveBinary-512-pca (self)                          125.81     1_537.93     1_663.74       0.3823          1.0678            1.0665         3.55
ExhaustiveBinary-1024-random_no_rr (query)               123.43       499.88       623.32       0.1929          1.2764            1.2696         7.10
ExhaustiveBinary-1024-random-rf10 (query)                123.43       615.72       739.15       0.4214          1.0550            1.0552         7.10
ExhaustiveBinary-1024-random-rf20 (query)                123.43       729.64       853.07       0.5434          1.0327            1.0308         7.10
ExhaustiveBinary-1024-random (self)                      123.43     2_015.28     2_138.71       0.4232          1.0547            1.0552         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  161.60       496.62       658.22       0.1921          1.2733            1.2652         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   161.60       612.53       774.13       0.4226          1.0546            1.0544         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   161.60       724.87       886.48       0.5443          1.0326            1.0305         7.10
ExhaustiveBinary-1024-pca (self)                         161.60     2_017.55     2_179.15       0.4236          1.0546            1.0548         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   47.35       442.64       490.00       0.1211          1.4987            1.4523         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    47.35       478.37       525.73       0.3284          1.1039            1.0884         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    47.35       713.07       760.42       0.4385          1.0624            1.0494         1.53
ExhaustiveBinary-256-sign (self)                          47.35     1_579.24     1_626.60       0.3334          1.0988            1.0859         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)              263.65        49.99       313.64       0.1245          1.4365            1.4004         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)             263.65        48.37       312.02       0.1245          1.4365            1.4004         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)             263.65        53.12       316.77       0.1245          1.4365            1.4004         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)             263.65        98.19       361.84       0.3492          1.0889            1.0776         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)             263.65       147.57       411.22       0.4555          1.0543            1.0456         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)            263.65        94.04       357.69       0.3492          1.0889            1.0776         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)            263.65       145.47       409.12       0.4555          1.0543            1.0456         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)            263.65       100.19       363.84       0.3492          1.0889            1.0776         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)            263.65       146.98       410.63       0.4555          1.0543            1.0456         1.93
IVF-Binary-256-nl158-random (self)                       263.65       208.26       471.91       0.3542          1.0841            1.0761         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             302.13        45.73       347.86       0.1414          1.3598            1.3149         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             302.13        48.44       350.57       0.1414          1.3603            1.3152         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             302.13        52.95       355.08       0.1414          1.3604            1.3152         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            302.13        96.21       398.33       0.3893          1.0691            1.0625         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            302.13       144.11       446.23       0.4974          1.0428            1.0371         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            302.13       101.11       403.24       0.3890          1.0692            1.0625         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            302.13       145.03       447.15       0.4969          1.0429            1.0372         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            302.13       100.01       402.14       0.3890          1.0692            1.0625         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            302.13       148.54       450.66       0.4969          1.0429            1.0372         2.00
IVF-Binary-256-nl223-random (self)                       302.13       212.68       514.81       0.3942          1.0643            1.0614         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             335.21        47.72       382.93       0.1498          1.3359            1.2910         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             335.21        48.06       383.27       0.1498          1.3363            1.2912         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             335.21        51.68       386.89       0.1498          1.3364            1.2912         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            335.21        99.50       434.71       0.4016          1.0646            1.0586         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            335.21       148.11       483.32       0.5055          1.0413            1.0359         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            335.21        96.97       432.18       0.4015          1.0647            1.0586         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            335.21       145.25       480.46       0.5053          1.0413            1.0360         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            335.21        99.76       434.97       0.4015          1.0647            1.0586         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            335.21       149.36       484.57       0.5053          1.0413            1.0360         2.09
IVF-Binary-256-nl316-random (self)                       335.21       219.48       554.70       0.4062          1.0604            1.0577         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)                 288.91        40.36       329.27       0.1201          1.4453            1.4029         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)                288.91        41.96       330.87       0.1201          1.4453            1.4029         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)                288.91        44.57       333.48       0.1201          1.4453            1.4029         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)                288.91        89.92       378.82       0.3431          1.0959            1.0788         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)                288.91       136.52       425.43       0.4521          1.0584            1.0457         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)               288.91        90.79       379.70       0.3431          1.0959            1.0788         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)               288.91       139.44       428.35       0.4521          1.0584            1.0457         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)               288.91        91.04       379.95       0.3431          1.0959            1.0788         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)               288.91       143.59       432.50       0.4521          1.0584            1.0457         1.93
IVF-Binary-256-nl158-pca (self)                          288.91       198.40       487.30       0.3495          1.0891            1.0771         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                293.16        44.76       337.92       0.1378          1.3708            1.3186         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                293.16        45.53       338.69       0.1377          1.3712            1.3187         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                293.16        48.42       341.58       0.1377          1.3712            1.3187         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               293.16        93.63       386.79       0.3827          1.0753            1.0638         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               293.16       153.53       446.69       0.4957          1.0457            1.0375         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               293.16        93.20       386.36       0.3826          1.0754            1.0638         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               293.16       141.81       434.97       0.4955          1.0457            1.0375         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               293.16       101.21       394.37       0.3826          1.0754            1.0638         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               293.16       145.16       438.32       0.4955          1.0457            1.0375         2.00
IVF-Binary-256-nl223-pca (self)                          293.16       208.20       501.36       0.3896          1.0691            1.0621         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                371.08        49.05       420.13       0.1466          1.3422            1.2923         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                371.08        48.94       420.02       0.1466          1.3426            1.2924         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                371.08        51.43       422.51       0.1466          1.3427            1.2924         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               371.08       100.21       471.30       0.3964          1.0705            1.0596         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               371.08       141.76       512.85       0.5067          1.0437            1.0356         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               371.08       106.14       477.22       0.3963          1.0705            1.0596         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               371.08       143.67       514.75       0.5066          1.0437            1.0356         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               371.08        99.16       470.24       0.3963          1.0705            1.0596         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               371.08       168.42       539.50       0.5066          1.0437            1.0356         2.09
IVF-Binary-256-nl316-pca (self)                          371.08       217.90       588.98       0.4023          1.0647            1.0583         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)              282.22        58.52       340.73       0.1614          1.3415            1.3204         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)             282.22        61.08       343.30       0.1614          1.3415            1.3204         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)             282.22        64.39       346.60       0.1614          1.3415            1.3204         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)             282.22       115.54       397.75       0.3837          1.0670            1.0659         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)             282.22       163.77       445.98       0.4941          1.0409            1.0384         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)            282.22       115.13       397.34       0.3837          1.0670            1.0659         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)            282.22       166.42       448.63       0.4941          1.0409            1.0384         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)            282.22       116.89       399.10       0.3837          1.0670            1.0659         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)            282.22       170.33       452.55       0.4941          1.0409            1.0384         3.71
IVF-Binary-512-nl158-random (self)                       282.22       286.67       568.89       0.3859          1.0652            1.0657         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             285.21        62.37       347.59       0.1711          1.2997            1.2781         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             285.21        70.17       355.38       0.1711          1.3000            1.2783         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             285.21        69.40       354.62       0.1711          1.3000            1.2783         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            285.21       115.85       401.06       0.4018          1.0605            1.0593         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            285.21       168.83       454.04       0.5143          1.0372            1.0348         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            285.21       115.49       400.70       0.4016          1.0605            1.0593         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            285.21       166.95       452.17       0.5140          1.0373            1.0348         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            285.21       121.27       406.49       0.4016          1.0605            1.0593         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            285.21       170.57       455.79       0.5140          1.0373            1.0348         3.77
IVF-Binary-512-nl223-random (self)                       285.21       291.70       576.91       0.4038          1.0592            1.0592         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             360.55        65.96       426.51       0.1755          1.2882            1.2668         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             360.55        66.77       427.32       0.1755          1.2884            1.2671         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             360.55        71.17       431.72       0.1755          1.2884            1.2671         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            360.55       119.75       480.29       0.4061          1.0594            1.0581         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            360.55       166.03       526.58       0.5177          1.0368            1.0341         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            360.55       120.21       480.76       0.4060          1.0595            1.0581         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            360.55       166.78       527.33       0.5174          1.0368            1.0341         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            360.55       121.64       482.18       0.4060          1.0595            1.0581         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            360.55       179.59       540.14       0.5174          1.0368            1.0341         3.86
IVF-Binary-512-nl316-random (self)                       360.55       296.06       656.61       0.4089          1.0580            1.0579         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)                 310.33        58.20       368.53       0.1596          1.3401            1.3152         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)                310.33        62.45       372.78       0.1596          1.3401            1.3152         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)                310.33        63.89       374.22       0.1596          1.3401            1.3152         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)                310.33       113.59       423.93       0.3846          1.0687            1.0645         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)                310.33       161.98       472.31       0.4979          1.0417            1.0374         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)               310.33       115.85       426.18       0.3846          1.0687            1.0645         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)               310.33       167.35       477.68       0.4979          1.0417            1.0374         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)               310.33       116.31       426.64       0.3846          1.0687            1.0645         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)               310.33       169.22       479.55       0.4979          1.0417            1.0374         3.71
IVF-Binary-512-nl158-pca (self)                          310.33       285.95       596.28       0.3875          1.0659            1.0647         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                318.66        62.22       380.88       0.1695          1.3031            1.2743         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                318.66        64.74       383.40       0.1695          1.3032            1.2744         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                318.66        68.78       387.45       0.1695          1.3032            1.2744         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               318.66       116.58       435.25       0.4038          1.0616            1.0582         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               318.66       168.80       487.46       0.5168          1.0382            1.0342         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               318.66       115.26       433.93       0.4037          1.0617            1.0583         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               318.66       165.53       484.20       0.5167          1.0382            1.0342         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               318.66       119.10       437.76       0.4037          1.0617            1.0583         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               318.66       170.63       489.29       0.5167          1.0382            1.0342         3.77
IVF-Binary-512-nl223-pca (self)                          318.66       288.64       607.31       0.4060          1.0597            1.0583         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                381.49        65.68       447.18       0.1734          1.2924            1.2651         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                381.49        67.22       448.71       0.1734          1.2925            1.2652         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                381.49        71.47       452.97       0.1734          1.2925            1.2652         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               381.49       118.58       500.07       0.4092          1.0604            1.0564         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               381.49       165.14       546.63       0.5223          1.0373            1.0334         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               381.49       121.75       503.24       0.4091          1.0604            1.0564         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               381.49       170.62       552.11       0.5223          1.0373            1.0334         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               381.49       121.83       503.32       0.4091          1.0604            1.0564         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               381.49       176.92       558.42       0.5223          1.0373            1.0334         3.86
IVF-Binary-512-nl316-pca (self)                          381.49       295.78       677.28       0.4117          1.0584            1.0567         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)             301.86        90.60       392.45       0.1942          1.2715            1.2650         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)            301.86        94.98       396.84       0.1942          1.2715            1.2650         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)            301.86       100.40       402.26       0.1942          1.2715            1.2650         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)            301.86       147.94       449.80       0.4245          1.0541            1.0543         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)            301.86       199.66       501.52       0.5468          1.0322            1.0302         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)           301.86       151.59       453.45       0.4245          1.0541            1.0543         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)           301.86       204.05       505.91       0.5468          1.0322            1.0302         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)           301.86       155.46       457.31       0.4245          1.0541            1.0543         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)           301.86       210.85       512.71       0.5468          1.0322            1.0302         7.26
IVF-Binary-1024-nl158-random (self)                      301.86       410.53       712.39       0.4263          1.0539            1.0544         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            308.13        93.65       401.78       0.1973          1.2556            1.2486         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            308.13        96.16       404.29       0.1972          1.2558            1.2487         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            308.13       103.67       411.80       0.1972          1.2558            1.2487         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           308.13       151.48       459.62       0.4343          1.0516            1.0515         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           308.13       200.78       508.91       0.5563          1.0309            1.0289         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           308.13       152.47       460.60       0.4341          1.0517            1.0516         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           308.13       203.09       511.22       0.5561          1.0309            1.0289         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           308.13       160.71       468.85       0.4341          1.0517            1.0516         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           308.13       210.32       518.46       0.5561          1.0309            1.0289         7.32
IVF-Binary-1024-nl223-random (self)                      308.13       409.67       717.81       0.4353          1.0515            1.0518         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            381.55        97.19       478.74       0.1988          1.2510            1.2444         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)            381.55        99.84       481.39       0.1988          1.2512            1.2445         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)            381.55       103.85       485.40       0.1988          1.2512            1.2445         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)           381.55       156.53       538.08       0.4364          1.0511            1.0510         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)           381.55       204.94       586.49       0.5577          1.0307            1.0286         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)           381.55       153.09       534.64       0.4363          1.0511            1.0510         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)           381.55       205.87       587.42       0.5576          1.0307            1.0287         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)           381.55       158.49       540.04       0.4363          1.0511            1.0510         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)           381.55       221.25       602.80       0.5576          1.0307            1.0287         7.42
IVF-Binary-1024-nl316-random (self)                      381.55       411.39       792.94       0.4380          1.0509            1.0513         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)                330.25        90.47       420.72       0.1934          1.2687            1.2608         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)               330.25        93.15       423.40       0.1934          1.2687            1.2608         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)               330.25        96.59       426.84       0.1934          1.2687            1.2608         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)               330.25       148.09       478.34       0.4258          1.0537            1.0535         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)               330.25       198.10       528.35       0.5482          1.0320            1.0299         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)              330.25       162.91       493.16       0.4258          1.0537            1.0535         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)              330.25       211.02       541.27       0.5482          1.0320            1.0299         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)              330.25       153.59       483.84       0.4258          1.0537            1.0535         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)              330.25       209.18       539.43       0.5482          1.0320            1.0299         7.26
IVF-Binary-1024-nl158-pca (self)                         330.25       407.28       737.53       0.4266          1.0537            1.0540         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               339.09        94.45       433.54       0.1974          1.2517            1.2440         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               339.09        96.59       435.68       0.1974          1.2518            1.2440         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               339.09       104.00       443.09       0.1974          1.2518            1.2440         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              339.09       153.42       492.51       0.4357          1.0510            1.0507         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              339.09       199.89       538.98       0.5589          1.0305            1.0281         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              339.09       151.57       490.66       0.4356          1.0510            1.0507         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              339.09       204.49       543.58       0.5587          1.0305            1.0282         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              339.09       157.01       496.10       0.4356          1.0510            1.0507         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              339.09       209.70       548.79       0.5587          1.0305            1.0282         7.32
IVF-Binary-1024-nl223-pca (self)                         339.09       408.58       747.67       0.4366          1.0510            1.0512         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               407.75        96.51       504.26       0.1988          1.2476            1.2400         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)               407.75        98.93       506.67       0.1988          1.2477            1.2401         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)               407.75       104.58       512.32       0.1988          1.2477            1.2401         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)              407.75       152.82       560.57       0.4389          1.0502            1.0501         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)              407.75       209.28       617.03       0.5614          1.0302            1.0279         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)              407.75       154.09       561.83       0.4389          1.0503            1.0501         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)              407.75       206.34       614.09       0.5613          1.0302            1.0279         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)              407.75       160.10       567.85       0.4389          1.0503            1.0501         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)              407.75       211.06       618.81       0.5613          1.0302            1.0279         7.42
IVF-Binary-1024-nl316-pca (self)                         407.75       411.11       818.86       0.4398          1.0503            1.0504         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)                228.61       145.76       374.37       0.1213          1.4946            1.4409         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               228.61       151.73       380.34       0.1213          1.4946            1.4409         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               228.61       148.71       377.32       0.1213          1.4946            1.4409         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               228.61       181.60       410.21       0.3330          1.1010            1.0848         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               228.61       325.94       554.55       0.4412          1.0614            1.0485         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              228.61       181.54       410.15       0.3330          1.1010            1.0848         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              228.61       327.54       556.15       0.4412          1.0614            1.0485         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              228.61       187.36       415.97       0.3330          1.1010            1.0848         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              228.61       336.68       565.29       0.4412          1.0614            1.0485         1.68
IVF-Binary-256-nl158-sign (self)                         228.61       492.60       721.21       0.3381          1.0958            1.0832         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               233.55       145.44       378.99       0.1226          1.4837            1.4298         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               233.55       149.21       382.77       0.1226          1.4850            1.4303         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               233.55       156.29       389.84       0.1225          1.4851            1.4303         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              233.55       192.08       425.63       0.3562          1.0888            1.0753         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              233.55       326.24       559.79       0.4587          1.0559            1.0444         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              233.55       187.31       420.86       0.3560          1.0890            1.0754         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              233.55       330.87       564.43       0.4584          1.0560            1.0444         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              233.55       189.74       423.30       0.3560          1.0890            1.0754         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              233.55       333.40       566.95       0.4584          1.0560            1.0444         1.75
IVF-Binary-256-nl223-sign (self)                         233.55       497.15       730.70       0.3612          1.0839            1.0738         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               298.71       148.79       447.50       0.1234          1.4702            1.4163         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               298.71       150.64       449.35       0.1234          1.4713            1.4163         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               298.71       151.83       450.54       0.1234          1.4715            1.4163         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              298.71       189.95       488.66       0.3598          1.0873            1.0740         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              298.71       329.47       628.18       0.4585          1.0560            1.0445         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              298.71       187.33       486.04       0.3597          1.0875            1.0741         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              298.71       333.60       632.31       0.4582          1.0561            1.0445         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              298.71       190.82       489.54       0.3597          1.0875            1.0741         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              298.71       340.50       639.21       0.4582          1.0561            1.0445         1.84
IVF-Binary-256-nl316-sign (self)                         298.71       534.32       833.03       0.3649          1.0824            1.0722         1.84
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
Exhaustive (query)                                        72.92     1_345.59     1_418.51       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         72.92     4_525.13     4_598.05       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                142.35       267.58       409.92       0.1109          1.3512            1.3092         2.03
ExhaustiveBinary-256-random-rf10 (query)                 142.35       398.99       541.33       0.3143          1.0825            1.0613         2.03
ExhaustiveBinary-256-random-rf20 (query)                 142.35       517.46       659.81       0.4100          1.0523            1.0368         2.03
ExhaustiveBinary-256-random (self)                       142.35     1_316.03     1_458.37       0.3161          1.0784            1.0600         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   235.24       267.82       503.06       0.1167          1.3480            1.2981         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    235.24       392.23       627.47       0.3159          1.0791            1.0596         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    235.24       523.54       758.78       0.4121          1.0505            1.0362         2.03
ExhaustiveBinary-256-pca (self)                          235.24     1_236.35     1_471.59       0.3171          1.0782            1.0588         2.03
ExhaustiveBinary-512-random_no_rr (query)                214.88       384.11       598.99       0.1528          1.2601            1.2299         4.05
ExhaustiveBinary-512-random-rf10 (query)                 214.88       527.58       742.46       0.3465          1.0565            1.0514         4.05
ExhaustiveBinary-512-random-rf20 (query)                 214.88       705.01       919.88       0.4452          1.0358            1.0312         4.05
ExhaustiveBinary-512-random (self)                       214.88     1_665.72     1_880.60       0.3476          1.0547            1.0512         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   316.88       394.18       711.06       0.1558          1.2535            1.2254         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    316.88       570.27       887.15       0.3512          1.0523            1.0507         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    316.88       666.92       983.80       0.4484          1.0329            1.0309         4.05
ExhaustiveBinary-512-pca (self)                          316.88     1_668.67     1_985.55       0.3515          1.0522            1.0505         4.05
ExhaustiveBinary-1024-random_no_rr (query)               259.20       590.27       849.47       0.1816          1.2043            1.1936         8.11
ExhaustiveBinary-1024-random-rf10 (query)                259.20       725.61       984.81       0.3747          1.0447            1.0452         8.11
ExhaustiveBinary-1024-random-rf20 (query)                259.20       869.87     1_129.07       0.4789          1.0282            1.0270         8.11
ExhaustiveBinary-1024-random (self)                      259.20     2_625.31     2_884.50       0.3754          1.0447            1.0451         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  368.87       588.32       957.19       0.1832          1.2013            1.1905         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   368.87       730.29     1_099.16       0.3798          1.0434            1.0443         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   368.87       879.28     1_248.15       0.4867          1.0272            1.0261         8.11
ExhaustiveBinary-1024-pca (self)                         368.87     2_437.62     2_806.48       0.3787          1.0436            1.0444         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   86.29       675.66       761.95       0.1518          1.2701            1.2528         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    86.29       743.63       829.92       0.3399          1.0607            1.0535         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    86.29     1_129.87     1_216.16       0.4406          1.0369            1.0319         3.05
ExhaustiveBinary-512-sign (self)                          86.29     2_484.55     2_570.84       0.3409          1.0595            1.0531         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)              403.68        76.63       480.31       0.1137          1.3389            1.3036         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)             403.68        77.78       481.46       0.1137          1.3389            1.3036         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)             403.68        82.32       486.00       0.1137          1.3389            1.3036         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)             403.68       159.00       562.68       0.3176          1.0801            1.0600         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)             403.68       269.40       673.08       0.4134          1.0507            1.0360         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)            403.68       164.85       568.53       0.3176          1.0801            1.0600         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)            403.68       245.03       648.71       0.4134          1.0507            1.0360         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)            403.68       152.02       555.70       0.3176          1.0801            1.0600         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)            403.68       254.61       658.29       0.4134          1.0507            1.0360         2.34
IVF-Binary-256-nl158-random (self)                       403.68       296.16       699.84       0.3195          1.0756            1.0588         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)             426.16        78.62       504.78       0.1323          1.2781            1.2367         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)             426.16        78.93       505.09       0.1322          1.2784            1.2369         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)             426.16        77.11       503.27       0.1322          1.2785            1.2369         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)            426.16       160.84       587.00       0.3685          1.0557            1.0457         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)            426.16       249.24       675.40       0.4682          1.0358            1.0279         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)            426.16       158.79       584.95       0.3683          1.0558            1.0457         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)            426.16       252.13       678.29       0.4678          1.0359            1.0280         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)            426.16       160.86       587.02       0.3683          1.0558            1.0457         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)            426.16       260.88       687.04       0.4678          1.0359            1.0280         2.47
IVF-Binary-256-nl223-random (self)                       426.16       352.20       778.36       0.3700          1.0518            1.0451         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)             502.26        80.03       582.29       0.1436          1.2537            1.2119         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)             502.26        80.23       582.49       0.1436          1.2539            1.2120         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)             502.26        82.98       585.23       0.1435          1.2545            1.2124         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)            502.26       166.86       669.12       0.3831          1.0510            1.0421         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)            502.26       257.12       759.38       0.4824          1.0341            1.0261         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)            502.26       164.08       666.34       0.3828          1.0511            1.0421         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)            502.26       255.17       757.43       0.4817          1.0343            1.0262         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)            502.26       168.76       671.02       0.3827          1.0511            1.0422         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)            502.26       259.64       761.90       0.4816          1.0343            1.0262         2.65
IVF-Binary-256-nl316-random (self)                       502.26       337.97       840.23       0.3847          1.0473            1.0415         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)                 533.48        69.70       603.18       0.1193          1.3358            1.2928         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)                533.48        69.09       602.57       0.1193          1.3358            1.2928         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)                533.48        69.54       603.01       0.1193          1.3358            1.2928         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)                533.48       153.79       687.27       0.3193          1.0758            1.0585         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)                533.48       244.14       777.62       0.4156          1.0484            1.0355         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)               533.48       151.05       684.53       0.3193          1.0758            1.0585         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)               533.48       241.30       774.78       0.4156          1.0484            1.0355         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)               533.48       162.04       695.52       0.3193          1.0758            1.0585         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)               533.48       244.10       777.58       0.4156          1.0484            1.0355         2.34
IVF-Binary-256-nl158-pca (self)                          533.48       293.60       827.08       0.3209          1.0742            1.0576         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)                554.90        82.68       637.58       0.1362          1.2797            1.2334         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)                554.90        79.45       634.35       0.1361          1.2801            1.2335         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)                554.90        81.36       636.25       0.1361          1.2801            1.2335         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)               554.90       175.97       730.87       0.3632          1.0554            1.0461         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)               554.90       268.14       823.03       0.4637          1.0358            1.0283         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)               554.90       189.39       744.29       0.3631          1.0555            1.0461         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)               554.90       282.52       837.42       0.4635          1.0358            1.0283         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)               554.90       178.60       733.50       0.3630          1.0555            1.0461         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)               554.90       268.65       823.54       0.4635          1.0358            1.0283         2.47
IVF-Binary-256-nl223-pca (self)                          554.90       321.23       876.13       0.3655          1.0537            1.0454         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)                606.62        78.64       685.26       0.1455          1.2599            1.2105         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)                606.62        80.58       687.20       0.1454          1.2600            1.2106         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)                606.62        82.94       689.56       0.1454          1.2604            1.2107         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)               606.62       170.81       777.44       0.3774          1.0508            1.0427         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)               606.62       258.82       865.44       0.4767          1.0333            1.0267         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)               606.62       165.48       772.11       0.3772          1.0509            1.0427         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)               606.62       279.59       886.22       0.4762          1.0334            1.0268         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)               606.62       168.77       775.39       0.3771          1.0509            1.0427         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)               606.62       260.87       867.49       0.4761          1.0334            1.0268         2.65
IVF-Binary-256-nl316-pca (self)                          606.62       344.72       951.34       0.3791          1.0493            1.0422         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)              492.91        93.31       586.22       0.1542          1.2560            1.2276         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)             492.91        96.80       589.71       0.1542          1.2560            1.2276         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)             492.91        99.86       592.77       0.1542          1.2560            1.2276         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)             492.91       185.53       678.44       0.3476          1.0561            1.0510         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)             492.91       276.33       769.24       0.4464          1.0356            1.0310         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)            492.91       185.33       678.23       0.3476          1.0561            1.0510         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)            492.91       280.53       773.44       0.4464          1.0356            1.0310         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)            492.91       188.05       680.96       0.3476          1.0561            1.0510         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)            492.91       284.47       777.38       0.4464          1.0356            1.0310         4.36
IVF-Binary-512-nl158-random (self)                       492.91       482.27       975.18       0.3486          1.0543            1.0508         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)             489.18       100.80       589.98       0.1643          1.2246            1.1980         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)             489.18       103.66       592.84       0.1642          1.2249            1.1983         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)             489.18       106.65       595.83       0.1642          1.2249            1.1983         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)            489.18       207.29       696.47       0.3692          1.0480            1.0450         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)            489.18       285.27       774.45       0.4686          1.0313            1.0278         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)            489.18       193.50       682.69       0.3689          1.0481            1.0451         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)            489.18       287.27       776.45       0.4680          1.0314            1.0279         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)            489.18       197.28       686.46       0.3689          1.0481            1.0451         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)            489.18       291.96       781.14       0.4680          1.0314            1.0279         4.49
IVF-Binary-512-nl223-random (self)                       489.18       541.57     1_030.75       0.3698          1.0470            1.0451         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)             608.40       112.49       720.90       0.1680          1.2145            1.1880         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)             608.40       116.00       724.40       0.1679          1.2148            1.1884         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)             608.40       119.51       727.92       0.1679          1.2152            1.1886         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)            608.40       198.18       806.59       0.3751          1.0467            1.0439         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)            608.40       292.02       900.42       0.4752          1.0304            1.0271         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)            608.40       196.49       804.90       0.3748          1.0468            1.0440         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)            608.40       289.70       898.11       0.4744          1.0305            1.0272         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)            608.40       199.26       807.66       0.3748          1.0468            1.0440         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)            608.40       296.05       904.45       0.4742          1.0305            1.0273         4.67
IVF-Binary-512-nl316-random (self)                       608.40       523.64     1_132.04       0.3755          1.0457            1.0439         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)                 600.89        93.80       694.70       0.1572          1.2489            1.2229         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)                600.89        96.33       697.22       0.1572          1.2489            1.2229         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)                600.89       103.07       703.97       0.1572          1.2489            1.2229         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)                600.89       188.44       789.33       0.3528          1.0515            1.0501         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)                600.89       277.30       878.19       0.4499          1.0326            1.0306         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)               600.89       184.86       785.75       0.3528          1.0515            1.0501         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)               600.89       290.67       891.56       0.4499          1.0326            1.0306         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)               600.89       187.41       788.30       0.3528          1.0515            1.0501         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)               600.89       283.20       884.10       0.4499          1.0326            1.0306         4.36
IVF-Binary-512-nl158-pca (self)                          600.89       543.34     1_144.24       0.3532          1.0513            1.0499         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)                613.44       107.15       720.59       0.1671          1.2191            1.1940         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)                613.44       109.17       722.61       0.1670          1.2194            1.1943         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)                613.44       109.32       722.76       0.1670          1.2194            1.1943         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)               613.44       194.46       807.90       0.3707          1.0459            1.0453         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)               613.44       283.08       896.52       0.4720          1.0291            1.0275         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)               613.44       245.51       858.95       0.3704          1.0460            1.0453         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)               613.44       325.42       938.87       0.4714          1.0292            1.0276         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)               613.44       212.86       826.30       0.3704          1.0460            1.0453         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)               613.44       313.60       927.04       0.4714          1.0292            1.0276         4.49
IVF-Binary-512-nl223-pca (self)                          613.44       499.37     1_112.82       0.3709          1.0458            1.0452         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)                704.93       106.94       811.86       0.1707          1.2099            1.1841         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)                704.93       107.98       812.91       0.1706          1.2101            1.1844         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)                704.93       112.45       817.38       0.1706          1.2102            1.1846         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)               704.93       199.69       904.62       0.3771          1.0444            1.0436         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)               704.93       292.02       996.95       0.4785          1.0284            1.0267         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)               704.93       196.26       901.18       0.3768          1.0445            1.0437         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)               704.93       295.88     1_000.80       0.4777          1.0285            1.0267         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)               704.93       200.56       905.49       0.3767          1.0445            1.0437         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)               704.93       295.62     1_000.54       0.4775          1.0286            1.0268         4.67
IVF-Binary-512-nl316-pca (self)                          704.93       468.80     1_173.73       0.3771          1.0445            1.0438         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)             532.36       146.21       678.58       0.1822          1.2030            1.1925         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)            532.36       151.55       683.92       0.1822          1.2030            1.1925         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)            532.36       154.00       686.37       0.1822          1.2030            1.1925         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)            532.36       246.34       778.70       0.3753          1.0446            1.0450         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)            532.36       336.31       868.67       0.4798          1.0281            1.0269         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)           532.36       246.91       779.28       0.3753          1.0446            1.0450         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)           532.36       381.32       913.68       0.4798          1.0281            1.0269         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)           532.36       264.36       796.73       0.3753          1.0446            1.0450         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)           532.36       351.66       884.03       0.4798          1.0281            1.0269         8.42
IVF-Binary-1024-nl158-random (self)                      532.36       635.68     1_168.04       0.3761          1.0445            1.0449         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)            529.49       152.32       681.81       0.1854          1.1897            1.1801         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)            529.49       157.64       687.13       0.1854          1.1899            1.1803         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)            529.49       160.54       690.02       0.1854          1.1899            1.1803         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)           529.49       244.87       774.36       0.3860          1.0419            1.0423         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)           529.49       355.30       884.78       0.4927          1.0265            1.0252         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)           529.49       273.19       802.67       0.3858          1.0419            1.0423         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)           529.49       352.29       881.77       0.4922          1.0265            1.0252         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)           529.49       257.59       787.08       0.3858          1.0419            1.0423         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)           529.49       356.48       885.96       0.4922          1.0265            1.0252         8.54
IVF-Binary-1024-nl223-random (self)                      529.49       646.50     1_175.99       0.3870          1.0419            1.0423         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)            627.17       162.96       790.13       0.1868          1.1852            1.1758         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)            627.17       160.63       787.80       0.1868          1.1855            1.1761         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)            627.17       165.72       792.90       0.1868          1.1856            1.1762         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)           627.17       284.61       911.79       0.3893          1.0414            1.0415         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)           627.17       390.69     1_017.87       0.4955          1.0262            1.0249         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)           627.17       255.61       882.79       0.3889          1.0415            1.0416         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)           627.17       364.21       991.38       0.4949          1.0263            1.0249         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)           627.17       260.85       888.03       0.3888          1.0415            1.0416         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)           627.17       362.58       989.75       0.4948          1.0263            1.0250         8.73
IVF-Binary-1024-nl316-random (self)                      627.17       669.69     1_296.86       0.3897          1.0414            1.0417         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)                624.15       146.46       770.61       0.1838          1.2000            1.1896         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)               624.15       147.61       771.76       0.1838          1.2000            1.1896         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)               624.15       152.51       776.66       0.1838          1.2000            1.1896         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)               624.15       240.64       864.79       0.3805          1.0433            1.0441         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)               624.15       333.46       957.61       0.4876          1.0270            1.0260         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)              624.15       239.93       864.08       0.3805          1.0433            1.0441         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)              624.15       346.51       970.66       0.4876          1.0270            1.0260         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)              624.15       244.21       868.36       0.3805          1.0433            1.0441         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)              624.15       356.88       981.03       0.4876          1.0270            1.0260         8.42
IVF-Binary-1024-nl158-pca (self)                         624.15       784.17     1_408.32       0.3794          1.0434            1.0442         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)               715.64       176.25       891.89       0.1870          1.1877            1.1785         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)               715.64       167.63       883.26       0.1870          1.1880            1.1787         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)               715.64       164.23       879.87       0.1870          1.1880            1.1787         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)              715.64       251.66       967.30       0.3905          1.0408            1.0416         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)              715.64       390.87     1_106.51       0.4989          1.0255            1.0246         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)              715.64       263.15       978.78       0.3902          1.0408            1.0417         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)              715.64       361.21     1_076.85       0.4983          1.0256            1.0247         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)              715.64       296.24     1_011.88       0.3902          1.0408            1.0417         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)              715.64       392.94     1_108.58       0.4983          1.0256            1.0247         8.54
IVF-Binary-1024-nl223-pca (self)                         715.64       736.82     1_452.45       0.3897          1.0410            1.0419         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)               732.78       168.85       901.62       0.1882          1.1834            1.1741         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)               732.78       168.60       901.37       0.1882          1.1836            1.1744         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)               732.78       173.64       906.42       0.1881          1.1837            1.1745         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)              732.78       260.68       993.46       0.3940          1.0401            1.0408         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)              732.78       367.85     1_100.63       0.5030          1.0252            1.0242         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)              732.78       265.42       998.20       0.3937          1.0402            1.0410         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)              732.78       365.70     1_098.48       0.5024          1.0252            1.0243         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)              732.78       268.18     1_000.96       0.3936          1.0403            1.0410         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)              732.78       371.14     1_103.92       0.5022          1.0253            1.0243         8.73
IVF-Binary-1024-nl316-pca (self)                         732.78       693.26     1_426.04       0.3929          1.0404            1.0411         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)                357.75       286.15       643.90       0.1519          1.2699            1.2515         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)               357.75       289.94       647.70       0.1519          1.2699            1.2515         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)               357.75       293.03       650.78       0.1519          1.2699            1.2515         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)               357.75       365.68       723.43       0.3414          1.0596            1.0531         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)               357.75       663.76     1_021.51       0.4416          1.0366            1.0317         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)              357.75       366.61       724.36       0.3414          1.0596            1.0531         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)              357.75       655.82     1_013.57       0.4416          1.0366            1.0317         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)              357.75       367.69       725.44       0.3414          1.0596            1.0531         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)              357.75       660.60     1_018.35       0.4416          1.0366            1.0317         3.36
IVF-Binary-512-nl158-sign (self)                         357.75     1_000.63     1_358.38       0.3425          1.0582            1.0527         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               359.97       290.26       650.23       0.1529          1.2635            1.2471         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               359.97       293.97       653.94       0.1528          1.2643            1.2484         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               359.97       298.74       658.71       0.1528          1.2643            1.2484         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              359.97       370.48       730.45       0.3504          1.0559            1.0502         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              359.97       656.12     1_016.09       0.4492          1.0348            1.0305         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              359.97       369.07       729.04       0.3502          1.0560            1.0502         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              359.97       685.65     1_045.61       0.4486          1.0350            1.0306         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              359.97       377.65       737.61       0.3502          1.0560            1.0502         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              359.97       667.48     1_027.45       0.4486          1.0350            1.0306         3.49
IVF-Binary-512-nl223-sign (self)                         359.97     1_009.96     1_369.92       0.3506          1.0551            1.0500         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)               456.86       302.58       759.44       0.1539          1.2662            1.2437         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)               456.86       297.47       754.33       0.1538          1.2675            1.2453         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)               456.86       300.95       757.81       0.1537          1.2681            1.2458         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)              456.86       374.77       831.63       0.3536          1.0549            1.0493         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)              456.86       667.84     1_124.70       0.4499          1.0345            1.0304         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)              456.86       373.57       830.43       0.3531          1.0551            1.0494         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)              456.86       783.59     1_240.45       0.4490          1.0347            1.0306         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)              456.86       415.93       872.79       0.3530          1.0551            1.0495         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)              456.86       745.44     1_202.30       0.4487          1.0348            1.0306         3.67
IVF-Binary-512-nl316-sign (self)                         456.86     1_168.65     1_625.51       0.3538          1.0540            1.0492         3.67
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
Exhaustive (query)                                       108.08     2_040.81     2_148.89       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        108.08     6_790.07     6_898.15       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                215.49       292.57       508.06       0.1140          1.2809            1.2433         2.28
ExhaustiveBinary-256-random-rf10 (query)                 215.49       443.62       659.11       0.3148          1.0656            1.0476         2.28
ExhaustiveBinary-256-random-rf20 (query)                 215.49       587.13       802.62       0.4075          1.0420            1.0293         2.28
ExhaustiveBinary-256-random (self)                       215.49     1_336.56     1_552.05       0.3168          1.0618            1.0471         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   408.89       291.38       700.27       0.1054          1.3026            1.2617         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    408.89       423.08       831.97       0.3012          1.0735            1.0517         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    408.89       575.02       983.91       0.3931          1.0471            1.0315         2.28
ExhaustiveBinary-256-pca (self)                          408.89     1_320.23     1_729.12       0.3048          1.0710            1.0504         2.28
ExhaustiveBinary-512-random_no_rr (query)                300.80       429.52       730.32       0.1506          1.2094            1.1809         4.55
ExhaustiveBinary-512-random-rf10 (query)                 300.80       594.27       895.08       0.3395          1.0453            1.0426         4.55
ExhaustiveBinary-512-random-rf20 (query)                 300.80       738.62     1_039.42       0.4326          1.0293            1.0264         4.55
ExhaustiveBinary-512-random (self)                       300.80     1_862.48     2_163.28       0.3401          1.0435            1.0423         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   550.12       424.12       974.24       0.1459          1.2162            1.1914         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    550.12       581.07     1_131.20       0.3341          1.0468            1.0435         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    550.12       766.99     1_317.11       0.4278          1.0295            1.0269         4.55
ExhaustiveBinary-512-pca (self)                          550.12     1_844.18     2_394.30       0.3355          1.0454            1.0433         4.55
ExhaustiveBinary-1024-random_no_rr (query)               509.01       656.40     1_165.41       0.1761          1.1673            1.1571         9.11
ExhaustiveBinary-1024-random-rf10 (query)                509.01       828.22     1_337.22       0.3603          1.0383            1.0383         9.11
ExhaustiveBinary-1024-random-rf20 (query)                509.01     1_037.11     1_546.12       0.4618          1.0244            1.0230         9.11
ExhaustiveBinary-1024-random (self)                      509.01     2_681.48     3_190.49       0.3602          1.0377            1.0383         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  712.41       658.41     1_370.82       0.1756          1.1686            1.1586         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   712.41       840.26     1_552.66       0.3594          1.0382            1.0385         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   712.41     1_047.81     1_760.21       0.4576          1.0246            1.0237         9.11
ExhaustiveBinary-1024-pca (self)                         712.41     2_710.18     3_422.59       0.3584          1.0383            1.0389         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  138.13       853.40       991.53       0.1691          1.1871            1.1718         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   138.13       953.46     1_091.59       0.3431          1.0433            1.0415         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   138.13     1_462.17     1_600.30       0.4437          1.0266            1.0250         4.58
ExhaustiveBinary-768-sign (self)                         138.13     3_073.32     3_211.44       0.3438          1.0424            1.0413         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)              705.16        95.04       800.20       0.1164          1.2725            1.2403         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)             705.16        96.48       801.64       0.1164          1.2725            1.2403         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)             705.16       102.75       807.91       0.1164          1.2725            1.2403         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)             705.16       211.39       916.55       0.3172          1.0636            1.0468         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)             705.16       314.95     1_020.11       0.4095          1.0414            1.0290         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)            705.16       201.43       906.59       0.3172          1.0636            1.0468         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)            705.16       329.56     1_034.72       0.4095          1.0414            1.0290         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)            705.16       211.45       916.61       0.3172          1.0636            1.0468         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)            705.16       326.59     1_031.74       0.4095          1.0414            1.0290         2.74
IVF-Binary-256-nl158-random (self)                       705.16       418.10     1_123.26       0.3191          1.0600            1.0465         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)             637.94        94.39       732.33       0.1331          1.2303            1.1909         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)             637.94       100.90       738.83       0.1331          1.2303            1.1909         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)             637.94        96.54       734.47       0.1331          1.2303            1.1909         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)            637.94       208.44       846.37       0.3574          1.0474            1.0376         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)            637.94       331.25       969.18       0.4568          1.0309            1.0232         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)            637.94       210.61       848.55       0.3573          1.0474            1.0376         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)            637.94       320.67       958.60       0.4568          1.0309            1.0232         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)            637.94       211.01       848.95       0.3573          1.0474            1.0376         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)            637.94       328.30       966.24       0.4568          1.0309            1.0232         2.93
IVF-Binary-256-nl223-random (self)                       637.94       453.18     1_091.12       0.3590          1.0439            1.0372         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)             730.48       103.01       833.49       0.1402          1.2168            1.1747         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)             730.48       104.18       834.66       0.1402          1.2169            1.1747         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)             730.48       107.59       838.07       0.1402          1.2169            1.1747         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)            730.48       230.49       960.97       0.3660          1.0441            1.0360         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)            730.48       349.87     1_080.35       0.4635          1.0293            1.0224         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)            730.48       222.50       952.98       0.3660          1.0441            1.0360         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)            730.48       332.34     1_062.82       0.4634          1.0293            1.0224         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)            730.48       215.52       946.01       0.3660          1.0441            1.0360         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)            730.48       346.79     1_077.27       0.4634          1.0293            1.0224         3.21
IVF-Binary-256-nl316-random (self)                       730.48       484.47     1_214.95       0.3675          1.0408            1.0357         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)                 852.91        89.64       942.56       0.1076          1.2918            1.2570         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)                852.91        86.68       939.59       0.1076          1.2918            1.2570         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)                852.91        93.00       945.92       0.1076          1.2918            1.2570         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)                852.91       197.43     1_050.35       0.3045          1.0705            1.0511         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)                852.91       307.92     1_160.83       0.3973          1.0449            1.0309         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)               852.91       189.85     1_042.76       0.3045          1.0705            1.0511         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)               852.91       305.48     1_158.39       0.3973          1.0449            1.0309         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)               852.91       192.39     1_045.31       0.3045          1.0705            1.0511         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)               852.91       311.59     1_164.51       0.3973          1.0449            1.0309         2.74
IVF-Binary-256-nl158-pca (self)                          852.91       388.32     1_241.24       0.3083          1.0672            1.0497         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)                797.71       103.66       901.38       0.1268          1.2395            1.2037         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)                797.71        94.68       892.39       0.1268          1.2395            1.2037         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)                797.71       102.41       900.12       0.1268          1.2395            1.2037         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)               797.71       214.85     1_012.56       0.3613          1.0476            1.0372         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)               797.71       343.02     1_140.73       0.4626          1.0301            1.0228         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)               797.71       200.92       998.64       0.3613          1.0476            1.0372         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)               797.71       321.65     1_119.37       0.4626          1.0301            1.0228         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)               797.71       208.91     1_006.63       0.3613          1.0476            1.0372         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)               797.71       334.63     1_132.34       0.4626          1.0301            1.0228         2.93
IVF-Binary-256-nl223-pca (self)                          797.71       421.77     1_219.48       0.3645          1.0451            1.0365         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)                935.96       102.16     1_038.12       0.1365          1.2209            1.1850         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)                935.96       102.57     1_038.53       0.1365          1.2210            1.1850         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)                935.96       112.81     1_048.77       0.1365          1.2210            1.1850         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)               935.96       219.04     1_154.99       0.3737          1.0433            1.0349         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)               935.96       343.08     1_279.04       0.4727          1.0280            1.0216         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)               935.96       218.19     1_154.15       0.3737          1.0433            1.0349         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)               935.96       331.93     1_267.89       0.4727          1.0280            1.0216         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)               935.96       214.91     1_150.87       0.3737          1.0433            1.0349         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)               935.96       334.04     1_270.00       0.4727          1.0280            1.0216         3.21
IVF-Binary-256-nl316-pca (self)                          935.96       484.66     1_420.62       0.3769          1.0410            1.0341         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)              718.75       128.88       847.62       0.1520          1.2064            1.1790         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)             718.75       133.25       852.00       0.1520          1.2064            1.1790         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)             718.75       132.60       851.35       0.1520          1.2064            1.1790         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)             718.75       238.83       957.58       0.3401          1.0451            1.0423         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)             718.75       350.73     1_069.48       0.4336          1.0292            1.0262         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)            718.75       236.64       955.39       0.3401          1.0451            1.0423         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)            718.75       353.97     1_072.72       0.4336          1.0292            1.0262         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)            718.75       235.29       954.04       0.3401          1.0451            1.0423         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)            718.75       354.93     1_073.68       0.4336          1.0292            1.0262         5.02
IVF-Binary-512-nl158-random (self)                       718.75       587.10     1_305.85       0.3407          1.0433            1.0421         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)             714.87       137.19       852.06       0.1603          1.1838            1.1572         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)             714.87       138.77       853.64       0.1603          1.1838            1.1573         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)             714.87       140.57       855.43       0.1603          1.1838            1.1573         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)            714.87       247.95       962.81       0.3579          1.0405            1.0379         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)            714.87       362.97     1_077.84       0.4565          1.0263            1.0235         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)            714.87       245.94       960.81       0.3579          1.0405            1.0379         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)            714.87       366.73     1_081.59       0.4565          1.0263            1.0235         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)            714.87       248.59       963.45       0.3579          1.0405            1.0379         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)            714.87       376.87     1_091.73       0.4565          1.0263            1.0235         5.21
IVF-Binary-512-nl223-random (self)                       714.87       597.25     1_312.11       0.3585          1.0390            1.0378         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)             796.24       140.82       937.06       0.1626          1.1785            1.1525         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)             796.24       142.81       939.05       0.1625          1.1787            1.1525         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)             796.24       143.77       940.02       0.1625          1.1787            1.1525         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)            796.24       258.04     1_054.28       0.3621          1.0391            1.0372         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)            796.24       374.90     1_171.14       0.4589          1.0256            1.0234         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)            796.24       249.91     1_046.16       0.3621          1.0391            1.0372         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)            796.24       374.81     1_171.05       0.4588          1.0256            1.0234         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)            796.24       267.81     1_064.05       0.3621          1.0391            1.0372         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)            796.24       386.42     1_182.66       0.4588          1.0256            1.0234         5.48
IVF-Binary-512-nl316-random (self)                       796.24       630.55     1_426.79       0.3624          1.0378            1.0372         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)                 893.50       121.18     1_014.69       0.1473          1.2122            1.1891         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)                893.50       123.53     1_017.03       0.1473          1.2122            1.1891         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)                893.50       125.03     1_018.53       0.1473          1.2122            1.1891         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)                893.50       234.38     1_127.88       0.3358          1.0460            1.0431         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)                893.50       356.73     1_250.23       0.4296          1.0292            1.0267         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)               893.50       233.28     1_126.78       0.3358          1.0460            1.0431         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)               893.50       362.75     1_256.26       0.4296          1.0292            1.0267         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)               893.50       236.61     1_130.11       0.3358          1.0460            1.0431         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)               893.50       362.76     1_256.26       0.4296          1.0292            1.0267         5.02
IVF-Binary-512-nl158-pca (self)                          893.50       561.86     1_455.36       0.3372          1.0445            1.0428         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)                881.51       134.13     1_015.64       0.1587          1.1837            1.1593         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)                881.51       139.52     1_021.03       0.1587          1.1837            1.1593         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)                881.51       136.22     1_017.73       0.1587          1.1837            1.1593         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)               881.51       250.29     1_131.80       0.3586          1.0397            1.0379         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)               881.51       364.59     1_246.10       0.4554          1.0259            1.0235         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)               881.51       245.01     1_126.51       0.3586          1.0397            1.0379         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)               881.51       377.63     1_259.14       0.4554          1.0259            1.0235         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)               881.51       248.27     1_129.78       0.3586          1.0397            1.0379         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)               881.51       398.96     1_280.47       0.4554          1.0259            1.0235         5.21
IVF-Binary-512-nl223-pca (self)                          881.51       592.79     1_474.30       0.3592          1.0390            1.0377         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)              1_019.81       139.47     1_159.28       0.1625          1.1766            1.1521         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)              1_019.81       143.64     1_163.45       0.1624          1.1767            1.1522         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)              1_019.81       147.68     1_167.49       0.1624          1.1767            1.1522         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)             1_019.81       265.41     1_285.23       0.3632          1.0386            1.0370         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)             1_019.81       392.02     1_411.83       0.4592          1.0254            1.0230         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)             1_019.81       249.45     1_269.27       0.3632          1.0386            1.0370         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)             1_019.81       383.18     1_402.99       0.4591          1.0254            1.0230         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)             1_019.81       256.48     1_276.30       0.3632          1.0386            1.0370         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)             1_019.81       383.42     1_403.23       0.4591          1.0254            1.0230         5.48
IVF-Binary-512-nl316-pca (self)                        1_019.81       626.25     1_646.06       0.3640          1.0380            1.0369         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)             914.49       197.16     1_111.65       0.1767          1.1663            1.1563         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)            914.49       200.89     1_115.38       0.1767          1.1663            1.1563         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)            914.49       205.40     1_119.89       0.1767          1.1663            1.1563         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)            914.49       320.37     1_234.86       0.3608          1.0382            1.0382         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)            914.49       453.18     1_367.67       0.4625          1.0243            1.0230         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)           914.49       336.42     1_250.91       0.3608          1.0382            1.0382         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)           914.49       460.10     1_374.59       0.4625          1.0243            1.0230         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)           914.49       332.25     1_246.74       0.3608          1.0382            1.0382         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)           914.49       464.24     1_378.73       0.4625          1.0243            1.0230         9.57
IVF-Binary-1024-nl158-random (self)                      914.49       876.50     1_790.99       0.3607          1.0376            1.0382         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)            870.70       213.46     1_084.15       0.1797          1.1568            1.1474         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)            870.70       207.64     1_078.34       0.1797          1.1568            1.1474         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)            870.70       216.97     1_087.66       0.1797          1.1568            1.1474         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)           870.70       335.94     1_206.64       0.3720          1.0360            1.0359         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)           870.70       464.04     1_334.73       0.4751          1.0229            1.0217         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)           870.70       337.60     1_208.30       0.3720          1.0360            1.0359         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)           870.70       467.00     1_337.69       0.4751          1.0229            1.0217         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)           870.70       347.53     1_218.23       0.3720          1.0360            1.0359         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)           870.70       476.98     1_347.68       0.4751          1.0229            1.0217         9.76
IVF-Binary-1024-nl223-random (self)                      870.70       900.72     1_771.41       0.3716          1.0354            1.0359         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)            987.12       223.53     1_210.65       0.1805          1.1543            1.1451        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)            987.12       228.03     1_215.14       0.1805          1.1544            1.1451        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)            987.12       224.28     1_211.39       0.1805          1.1544            1.1451        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)           987.12       359.29     1_346.41       0.3733          1.0356            1.0358        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)           987.12       482.78     1_469.90       0.4764          1.0227            1.0216        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)           987.12       353.06     1_340.17       0.3733          1.0356            1.0358        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)           987.12       485.39     1_472.51       0.4763          1.0227            1.0216        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)           987.12       364.13     1_351.24       0.3733          1.0356            1.0358        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)           987.12       495.49     1_482.61       0.4763          1.0227            1.0216        10.04
IVF-Binary-1024-nl316-random (self)                      987.12       961.54     1_948.66       0.3736          1.0350            1.0356        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)              1_108.89       201.03     1_309.92       0.1762          1.1673            1.1573         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)             1_108.89       227.19     1_336.08       0.1762          1.1673            1.1573         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)             1_108.89       212.57     1_321.46       0.1762          1.1673            1.1573         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)             1_108.89       318.13     1_427.02       0.3600          1.0380            1.0384         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)             1_108.89       456.36     1_565.25       0.4586          1.0245            1.0236         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)            1_108.89       322.18     1_431.07       0.3600          1.0380            1.0384         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)            1_108.89       459.12     1_568.01       0.4586          1.0245            1.0236         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)            1_108.89       335.62     1_444.51       0.3600          1.0380            1.0384         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)            1_108.89       461.82     1_570.71       0.4586          1.0245            1.0236         9.57
IVF-Binary-1024-nl158-pca (self)                       1_108.89       879.77     1_988.66       0.3591          1.0381            1.0388         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_073.66       210.17     1_283.83       0.1795          1.1565            1.1467         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_073.66       210.22     1_283.88       0.1795          1.1565            1.1467         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_073.66       214.94     1_288.60       0.1795          1.1565            1.1467         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_073.66       334.51     1_408.17       0.3711          1.0358            1.0360         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_073.66       465.64     1_539.30       0.4724          1.0231            1.0221         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_073.66       342.76     1_416.42       0.3711          1.0358            1.0360         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_073.66       473.74     1_547.40       0.4724          1.0231            1.0221         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_073.66       353.90     1_427.56       0.3711          1.0358            1.0360         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_073.66       475.51     1_549.17       0.4724          1.0231            1.0221         9.76
IVF-Binary-1024-nl223-pca (self)                       1_073.66       900.09     1_973.75       0.3704          1.0359            1.0363         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_200.75       221.62     1_422.37       0.1805          1.1540            1.1448        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_200.75       219.97     1_420.71       0.1805          1.1540            1.1448        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_200.75       226.76     1_427.51       0.1805          1.1540            1.1448        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_200.75       352.54     1_553.29       0.3737          1.0353            1.0357        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_200.75       490.12     1_690.87       0.4739          1.0228            1.0220        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_200.75       350.17     1_550.92       0.3737          1.0353            1.0357        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_200.75       483.44     1_684.19       0.4738          1.0228            1.0220        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_200.75       365.36     1_566.10       0.3737          1.0353            1.0357        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_200.75       517.17     1_717.92       0.4738          1.0228            1.0220        10.04
IVF-Binary-1024-nl316-pca (self)                       1_200.75       959.32     2_160.07       0.3726          1.0354            1.0359        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)                537.86       415.90       953.76       0.1693          1.1870            1.1720         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)               537.86       412.32       950.18       0.1693          1.1870            1.1720         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)               537.86       425.30       963.16       0.1693          1.1870            1.1720         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)               537.86       527.95     1_065.81       0.3434          1.0432            1.0414         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)               537.86       933.86     1_471.72       0.4438          1.0266            1.0249         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)              537.86       502.77     1_040.63       0.3434          1.0432            1.0414         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)              537.86       938.29     1_476.15       0.4438          1.0266            1.0249         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)              537.86       513.18     1_051.04       0.3434          1.0432            1.0414         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)              537.86       925.93     1_463.79       0.4438          1.0266            1.0249         5.04
IVF-Binary-768-nl158-sign (self)                         537.86     1_439.09     1_976.95       0.3440          1.0423            1.0413         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)               502.12       423.70       925.81       0.1692          1.1872            1.1717         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)               502.12       418.61       920.72       0.1692          1.1872            1.1717         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)               502.12       438.18       940.30       0.1692          1.1872            1.1717         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)              502.12       521.48     1_023.60       0.3492          1.0414            1.0401         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)              502.12       975.99     1_478.10       0.4483          1.0259            1.0245         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)              502.12       528.99     1_031.11       0.3492          1.0414            1.0401         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)              502.12       973.63     1_475.75       0.4483          1.0259            1.0245         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)              502.12       518.51     1_020.62       0.3492          1.0414            1.0401         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)              502.12       932.92     1_435.04       0.4483          1.0259            1.0245         5.23
IVF-Binary-768-nl223-sign (self)                         502.12     1_459.88     1_962.00       0.3496          1.0408            1.0400         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)               620.10       430.25     1_050.35       0.1694          1.1867            1.1714         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)               620.10       425.59     1_045.69       0.1694          1.1867            1.1714         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)               620.10       430.83     1_050.92       0.1694          1.1867            1.1714         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)              620.10       531.15     1_151.25       0.3501          1.0411            1.0399         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)              620.10       933.43     1_553.53       0.4490          1.0257            1.0244         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)              620.10       522.41     1_142.51       0.3501          1.0411            1.0399         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)              620.10       934.47     1_554.57       0.4489          1.0257            1.0244         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)              620.10       529.62     1_149.72       0.3501          1.0411            1.0399         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)              620.10       951.45     1_571.55       0.4489          1.0257            1.0244         5.51
IVF-Binary-768-nl316-sign (self)                         620.10     1_498.04     2_118.14       0.3504          1.0405            1.0398         5.51
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
Exhaustive (query)                                        33.04       678.17       711.22       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.04     2_353.47     2_386.51       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                 71.31       239.77       311.07       0.0970          1.6334            1.6378         1.78
ExhaustiveBinary-256-random-rf10 (query)                  71.31       344.35       415.66       0.3643          1.1391            1.1302         1.78
ExhaustiveBinary-256-random-rf20 (query)                  71.31       446.41       517.72       0.5087          1.0798            1.0701         1.78
ExhaustiveBinary-256-random (self)                        71.31     1_115.25     1_186.55       0.3862          1.1443            1.1409         1.78
ExhaustiveBinary-256-pca_no_rr (query)                    95.42       240.31       335.73       0.0922          1.6524            1.6606         1.78
ExhaustiveBinary-256-pca-rf10 (query)                     95.42       351.49       446.91       0.3517          1.1465            1.1384         1.78
ExhaustiveBinary-256-pca-rf20 (query)                     95.42       451.13       546.55       0.4943          1.0846            1.0744         1.78
ExhaustiveBinary-256-pca (self)                           95.42     1_123.48     1_218.90       0.3764          1.1502            1.1477         1.78
ExhaustiveBinary-512-random_no_rr (query)                 82.45       358.79       441.24       0.1464          1.5035            1.5085         3.55
ExhaustiveBinary-512-random-rf10 (query)                  82.45       479.52       561.97       0.4596          1.0936            1.0901         3.55
ExhaustiveBinary-512-random-rf20 (query)                  82.45       593.40       675.85       0.6085          1.0504            1.0459         3.55
ExhaustiveBinary-512-random (self)                        82.45     1_522.63     1_605.08       0.4800          1.0996            1.0995         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   107.75       358.33       466.08       0.1458          1.5041            1.5097         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    107.75       464.56       572.31       0.4543          1.0952            1.0911         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    107.75       569.46       677.21       0.6037          1.0513            1.0464         3.55
ExhaustiveBinary-512-pca (self)                          107.75     1_505.12     1_612.87       0.4777          1.1003            1.0997         3.55
ExhaustiveBinary-1024-random_no_rr (query)               114.15       497.09       611.24       0.2155          1.3655            1.3721         7.10
ExhaustiveBinary-1024-random-rf10 (query)                114.15       610.58       724.73       0.5869          1.0540            1.0515         7.10
ExhaustiveBinary-1024-random-rf20 (query)                114.15       721.50       835.65       0.7380          1.0260            1.0224         7.10
ExhaustiveBinary-1024-random (self)                      114.15     2_033.09     2_147.24       0.6118          1.0576            1.0546         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  138.38       506.10       644.48       0.2122          1.3735            1.3798         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   138.38       610.92       749.30       0.5776          1.0560            1.0532         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   138.38       723.63       862.01       0.7291          1.0273            1.0232         7.10
ExhaustiveBinary-1024-pca (self)                         138.38     2_067.45     2_205.83       0.6017          1.0602            1.0571         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   40.73       443.72       484.44       0.1044          1.6421            1.6499         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    40.73       479.11       519.83       0.3737          1.1368            1.1275         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    40.73       723.75       764.48       0.5265          1.0745            1.0646         1.53
ExhaustiveBinary-256-sign (self)                          40.73     1_551.42     1_592.15       0.3940          1.1439            1.1394         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)              226.81        49.24       276.05       0.1006          1.6235            1.6329         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)             226.81        51.75       278.56       0.1005          1.6238            1.6332         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)             226.81        55.49       282.30       0.1005          1.6238            1.6332         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)             226.81       101.34       328.15       0.3692          1.1376            1.1295         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)             226.81       147.80       374.62       0.5128          1.0789            1.0697         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)            226.81        98.58       325.40       0.3678          1.1379            1.1296         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)            226.81       150.38       377.19       0.5110          1.0792            1.0698         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)            226.81        99.76       326.58       0.3677          1.1379            1.1296         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)            226.81       159.26       386.07       0.5108          1.0792            1.0698         1.93
IVF-Binary-256-nl158-random (self)                       226.81       226.57       453.38       0.3897          1.1428            1.1402         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             265.30        45.50       310.79       0.1122          1.5873            1.5872         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             265.30        47.59       312.89       0.1121          1.5880            1.5877         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             265.30        52.47       317.77       0.1121          1.5881            1.5878         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            265.30        98.62       363.92       0.3938          1.1238            1.1155         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            265.30       152.66       417.96       0.5354          1.0713            1.0627         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            265.30        99.88       365.18       0.3933          1.1239            1.1157         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            265.30       151.91       417.21       0.5347          1.0714            1.0629         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            265.30       104.82       370.11       0.3931          1.1240            1.1157         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            265.30       154.99       420.29       0.5346          1.0714            1.0629         2.00
IVF-Binary-256-nl223-random (self)                       265.30       234.28       499.57       0.4141          1.1288            1.1266         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             356.60        48.02       404.62       0.1174          1.5678            1.5682         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             356.60        48.43       405.03       0.1173          1.5683            1.5687         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             356.60        52.45       409.05       0.1173          1.5685            1.5688         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            356.60       102.88       459.48       0.4041          1.1177            1.1109         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            356.60       150.68       507.27       0.5466          1.0675            1.0600         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            356.60       103.34       459.94       0.4034          1.1179            1.1111         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            356.60       150.78       507.38       0.5459          1.0677            1.0602         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            356.60       110.61       467.20       0.4033          1.1180            1.1112         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            356.60       161.03       517.63       0.5457          1.0677            1.0602         2.09
IVF-Binary-256-nl316-random (self)                       356.60       243.96       600.56       0.4252          1.1217            1.1220         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)                 250.00        40.26       290.26       0.0960          1.6427            1.6567         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)                250.00        42.93       292.92       0.0959          1.6429            1.6568         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)                250.00        46.19       296.18       0.0959          1.6429            1.6568         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)                250.00        96.86       346.85       0.3562          1.1449            1.1376         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)                250.00       144.38       394.38       0.4976          1.0836            1.0741         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)               250.00       102.19       352.18       0.3551          1.1451            1.1377         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)               250.00       146.82       396.82       0.4963          1.0838            1.0742         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)               250.00        98.99       348.99       0.3551          1.1451            1.1377         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)               250.00       149.33       399.33       0.4963          1.0838            1.0742         1.93
IVF-Binary-256-nl158-pca (self)                          250.00       225.91       475.91       0.3800          1.1487            1.1470         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                320.86        45.13       365.98       0.1095          1.5965            1.6007         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                320.86        47.25       368.11       0.1094          1.5975            1.6019         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                320.86        50.95       371.81       0.1094          1.5976            1.6020         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               320.86        99.26       420.12       0.3870          1.1269            1.1202         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               320.86       147.90       468.76       0.5263          1.0737            1.0657         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               320.86        99.18       420.04       0.3864          1.1272            1.1205         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               320.86       147.53       468.39       0.5254          1.0739            1.0660         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               320.86       105.57       426.43       0.3863          1.1272            1.1205         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               320.86       152.79       473.65       0.5253          1.0739            1.0660         2.00
IVF-Binary-256-nl223-pca (self)                          320.86       245.43       566.29       0.4099          1.1303            1.1302         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                363.16        47.73       410.89       0.1160          1.5759            1.5792         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                363.16        47.97       411.12       0.1159          1.5765            1.5798         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                363.16        51.98       415.14       0.1158          1.5771            1.5801         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               363.16       100.79       463.94       0.3965          1.1216            1.1148         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               363.16       149.52       512.68       0.5374          1.0701            1.0627         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               363.16       100.40       463.56       0.3959          1.1218            1.1151         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               363.16       158.04       521.20       0.5366          1.0703            1.0630         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               363.16       104.68       467.84       0.3957          1.1219            1.1152         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               363.16       154.23       517.39       0.5363          1.0703            1.0631         2.09
IVF-Binary-256-nl316-pca (self)                          363.16       246.66       609.82       0.4197          1.1247            1.1254         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)              245.17        60.14       305.31       0.1479          1.5006            1.5075         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)             245.17        60.69       305.85       0.1479          1.5007            1.5075         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)             245.17        65.69       310.86       0.1479          1.5007            1.5075         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)             245.17       118.79       363.96       0.4609          1.0933            1.0899         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)             245.17       167.09       412.26       0.6093          1.0503            1.0458         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)            245.17       118.52       363.69       0.4606          1.0933            1.0899         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)            245.17       170.74       415.91       0.6092          1.0503            1.0458         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)            245.17       123.31       368.48       0.4606          1.0933            1.0899         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)            245.17       177.32       422.49       0.6091          1.0503            1.0458         3.71
IVF-Binary-512-nl158-random (self)                       245.17       301.43       546.60       0.4809          1.0993            1.0994         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             288.19        61.75       349.94       0.1561          1.4799            1.4845         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             288.19        64.12       352.31       0.1561          1.4802            1.4846         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             288.19        74.77       362.96       0.1561          1.4802            1.4846         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            288.19       120.46       408.65       0.4754          1.0877            1.0848         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            288.19       168.94       457.13       0.6216          1.0474            1.0431         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            288.19       121.26       409.45       0.4750          1.0878            1.0849         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            288.19       171.78       459.97       0.6211          1.0475            1.0432         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            288.19       127.48       415.67       0.4750          1.0878            1.0849         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            288.19       178.68       466.87       0.6211          1.0475            1.0432         3.77
IVF-Binary-512-nl223-random (self)                       288.19       308.71       596.90       0.4946          1.0941            1.0938         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             350.67        65.52       416.19       0.1598          1.4705            1.4754         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             350.67        65.73       416.40       0.1597          1.4708            1.4759         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             350.67        72.12       422.79       0.1597          1.4708            1.4759         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            350.67       122.31       472.98       0.4803          1.0859            1.0830         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            350.67       170.31       520.98       0.6273          1.0463            1.0422         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            350.67       124.08       474.75       0.4799          1.0861            1.0831         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            350.67       173.28       523.95       0.6267          1.0464            1.0424         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            350.67       127.85       478.52       0.4798          1.0861            1.0831         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            350.67       179.22       529.89       0.6267          1.0464            1.0424         3.86
IVF-Binary-512-nl316-random (self)                       350.67       311.66       662.33       0.4995          1.0923            1.0917         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)                 271.29        57.80       329.08       0.1474          1.5015            1.5083         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)                271.29        60.49       331.78       0.1473          1.5015            1.5083         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)                271.29        65.43       336.72       0.1473          1.5015            1.5083         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)                271.29       119.53       390.82       0.4558          1.0948            1.0910         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)                271.29       169.23       440.52       0.6048          1.0510            1.0463         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)               271.29       121.99       393.28       0.4554          1.0948            1.0910         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)               271.29       171.04       442.32       0.6044          1.0511            1.0463         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)               271.29       122.99       394.27       0.4554          1.0948            1.0910         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)               271.29       177.44       448.73       0.6044          1.0511            1.0463         3.71
IVF-Binary-512-nl158-pca (self)                          271.29       307.89       579.18       0.4788          1.1000            1.0995         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                309.21        64.56       373.77       0.1557          1.4808            1.4847         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                309.21        64.17       373.38       0.1556          1.4812            1.4854         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                309.21        71.83       381.04       0.1556          1.4812            1.4854         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               309.21       120.08       429.29       0.4715          1.0888            1.0850         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               309.21       170.24       479.45       0.6191          1.0478            1.0432         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               309.21       120.06       429.27       0.4710          1.0889            1.0851         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               309.21       172.68       481.89       0.6185          1.0479            1.0434         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               309.21       126.92       436.13       0.4709          1.0889            1.0851         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               309.21       179.65       488.86       0.6184          1.0479            1.0434         3.77
IVF-Binary-512-nl223-pca (self)                          309.21       309.23       618.44       0.4931          1.0945            1.0940         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                376.88        65.32       442.20       0.1590          1.4717            1.4754         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                376.88        65.91       442.79       0.1589          1.4720            1.4760         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                376.88        73.42       450.30       0.1589          1.4721            1.4761         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               376.88       123.46       500.34       0.4766          1.0869            1.0833         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               376.88       174.98       551.86       0.6238          1.0467            1.0422         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               376.88       123.80       500.67       0.4761          1.0870            1.0835         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               376.88       171.97       548.85       0.6230          1.0469            1.0423         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               376.88       128.36       505.24       0.4761          1.0870            1.0835         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               376.88       179.20       556.08       0.6229          1.0469            1.0424         3.86
IVF-Binary-512-nl316-pca (self)                          376.88       313.40       690.28       0.4984          1.0925            1.0920         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)             273.85        89.89       363.74       0.2162          1.3647            1.3717         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)            273.85        93.90       367.75       0.2162          1.3647            1.3717         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)            273.85       100.87       374.72       0.2162          1.3647            1.3717         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)            273.85       152.83       426.69       0.5873          1.0539            1.0515         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)            273.85       202.79       476.64       0.7382          1.0260            1.0223         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)           273.85       154.59       428.44       0.5872          1.0539            1.0515         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)           273.85       209.22       483.07       0.7382          1.0260            1.0223         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)           273.85       160.46       434.31       0.5872          1.0539            1.0515         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)           273.85       220.02       493.87       0.7382          1.0260            1.0223         7.26
IVF-Binary-1024-nl158-random (self)                      273.85       423.24       697.09       0.6121          1.0575            1.0545         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            310.39        93.89       404.29       0.2204          1.3573            1.3648         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            310.39        95.91       406.30       0.2204          1.3574            1.3648         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            310.39       107.43       417.82       0.2204          1.3574            1.3648         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           310.39       159.28       469.67       0.5941          1.0523            1.0500         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           310.39       206.41       516.81       0.7440          1.0252            1.0214         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           310.39       156.69       467.09       0.5938          1.0524            1.0500         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           310.39       210.80       521.20       0.7437          1.0252            1.0215         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           310.39       167.71       478.11       0.5938          1.0524            1.0500         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           310.39       221.39       531.78       0.7436          1.0252            1.0215         7.32
IVF-Binary-1024-nl223-random (self)                      310.39       429.94       740.34       0.6183          1.0559            1.0530         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            381.11        96.08       477.19       0.2223          1.3533            1.3607         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)            381.11        97.44       478.56       0.2222          1.3534            1.3610         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)            381.11       106.53       487.64       0.2222          1.3534            1.3610         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)           381.11       156.56       537.67       0.5963          1.0517            1.0493         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)           381.11       210.04       591.16       0.7464          1.0248            1.0211         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)           381.11       158.14       539.25       0.5959          1.0518            1.0494         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)           381.11       210.91       592.02       0.7461          1.0248            1.0211         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)           381.11       164.87       545.98       0.5959          1.0518            1.0494         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)           381.11       220.97       602.08       0.7460          1.0248            1.0211         7.42
IVF-Binary-1024-nl316-random (self)                      381.11       576.47       957.58       0.6211          1.0552            1.0522         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)                318.40        90.10       408.50       0.2128          1.3727            1.3795         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)               318.40        99.95       418.35       0.2128          1.3727            1.3795         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)               318.40       104.51       422.91       0.2128          1.3727            1.3795         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)               318.40       197.01       515.41       0.5780          1.0559            1.0531         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)               318.40       233.28       551.68       0.7294          1.0272            1.0232         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)              318.40       164.49       482.89       0.5779          1.0559            1.0531         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)              318.40       237.06       555.46       0.7293          1.0272            1.0232         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)              318.40       162.25       480.65       0.5779          1.0559            1.0531         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)              318.40       239.08       557.48       0.7293          1.0272            1.0232         7.26
IVF-Binary-1024-nl158-pca (self)                         318.40       627.30       945.70       0.6020          1.0601            1.0571         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               470.03       103.45       573.48       0.2176          1.3640            1.3702         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               470.03       249.49       719.52       0.2175          1.3642            1.3705         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               470.03       167.58       637.61       0.2175          1.3642            1.3705         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              470.03       216.37       686.40       0.5853          1.0541            1.0511         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              470.03       260.94       730.97       0.7358          1.0263            1.0223         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              470.03       213.48       683.51       0.5850          1.0542            1.0512         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              470.03       305.28       775.31       0.7354          1.0264            1.0223         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              470.03       212.14       682.17       0.5849          1.0542            1.0512         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              470.03       288.97       759.01       0.7353          1.0264            1.0223         7.32
IVF-Binary-1024-nl223-pca (self)                         470.03       501.20       971.24       0.6094          1.0581            1.0550         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               496.54       105.35       601.89       0.2191          1.3601            1.3666         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)               496.54       102.34       598.89       0.2190          1.3603            1.3667         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)               496.54       108.14       604.68       0.2190          1.3603            1.3667         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)              496.54       166.73       663.27       0.5876          1.0535            1.0508         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)              496.54       226.47       723.01       0.7378          1.0260            1.0222         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)              496.54       164.68       661.22       0.5872          1.0536            1.0509         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)              496.54       226.06       722.60       0.7374          1.0261            1.0222         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)              496.54       176.51       673.05       0.5872          1.0536            1.0509         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)              496.54       248.09       744.64       0.7374          1.0261            1.0222         7.42
IVF-Binary-1024-nl316-pca (self)                         496.54       473.50       970.04       0.6114          1.0575            1.0543         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)                233.65       166.83       400.48       0.1042          1.6406            1.6481         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               233.65       169.10       402.75       0.1041          1.6411            1.6484         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               233.65       202.60       436.25       0.1041          1.6412            1.6485         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               233.65       234.99       468.63       0.3771          1.1355            1.1269         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               233.65       398.34       631.98       0.5286          1.0742            1.0643         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              233.65       216.64       450.29       0.3754          1.1359            1.1272         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              233.65       407.57       641.22       0.5283          1.0743            1.0643         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              233.65       246.80       480.45       0.3752          1.1359            1.1272         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              233.65       379.43       613.08       0.5281          1.0743            1.0644         1.68
IVF-Binary-256-nl158-sign (self)                         233.65       562.00       795.65       0.3957          1.1432            1.1389         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               248.34       159.60       407.94       0.1045          1.6391            1.6451         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               248.34       174.36       422.69       0.1044          1.6403            1.6460         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               248.34       165.59       413.93       0.1043          1.6406            1.6464         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              248.34       202.83       451.17       0.3865          1.1297            1.1209         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              248.34       387.79       636.13       0.5363          1.0715            1.0626         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              248.34       263.34       511.68       0.3859          1.1299            1.1214         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              248.34       428.80       677.13       0.5353          1.0718            1.0629         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              248.34       245.06       493.40       0.3858          1.1300            1.1214         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              248.34       390.71       639.04       0.5350          1.0719            1.0629         1.75
IVF-Binary-256-nl223-sign (self)                         248.34       567.77       816.11       0.4055          1.1379            1.1334         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               372.22       175.18       547.40       0.1050          1.6357            1.6455         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               372.22       175.23       547.45       0.1049          1.6364            1.6457         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               372.22       191.04       563.25       0.1048          1.6371            1.6464         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              372.22       228.77       600.99       0.3920          1.1266            1.1191         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              372.22       469.33       841.54       0.5390          1.0706            1.0621         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              372.22       244.52       616.74       0.3913          1.1269            1.1193         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              372.22       413.39       785.60       0.5381          1.0709            1.0622         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              372.22       224.55       596.76       0.3911          1.1270            1.1193         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              372.22       413.04       785.25       0.5378          1.0709            1.0623         1.84
IVF-Binary-256-nl316-sign (self)                         372.22       650.45     1_022.67       0.4107          1.1342            1.1317         1.84
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
Exhaustive (query)                                        70.82     1_321.27     1_392.09       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         70.82     4_561.92     4_632.74       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                137.41       266.39       403.80       0.0733          1.4884            1.4968         2.03
ExhaustiveBinary-256-random-rf10 (query)                 137.41       397.69       535.10       0.2947          1.1327            1.1260         2.03
ExhaustiveBinary-256-random-rf20 (query)                 137.41       521.38       658.80       0.4174          1.0830            1.0716         2.03
ExhaustiveBinary-256-random (self)                       137.41     1_290.12     1_427.53       0.3150          1.1321            1.1268         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   228.59       267.26       495.84       0.0722          1.4941            1.5001         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    228.59       397.44       626.03       0.2934          1.1324            1.1267         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    228.59       526.24       754.83       0.4181          1.0813            1.0715         2.03
ExhaustiveBinary-256-pca (self)                          228.59     1_300.56     1_529.14       0.3112          1.1338            1.1298         2.03
ExhaustiveBinary-512-random_no_rr (query)                207.58       397.34       604.92       0.1110          1.4064            1.4153         4.05
ExhaustiveBinary-512-random-rf10 (query)                 207.58       537.71       745.29       0.3695          1.0935            1.0919         4.05
ExhaustiveBinary-512-random-rf20 (query)                 207.58       671.07       878.65       0.4981          1.0544            1.0517         4.05
ExhaustiveBinary-512-random (self)                       207.58     1_702.57     1_910.15       0.3856          1.0953            1.0998         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   294.69       396.90       691.59       0.1063          1.4156            1.4272         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    294.69       530.08       824.77       0.3574          1.0990            1.0958         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    294.69       668.60       963.29       0.4853          1.0582            1.0544         4.05
ExhaustiveBinary-512-pca (self)                          294.69     1_699.75     1_994.44       0.3753          1.0995            1.1040         4.05
ExhaustiveBinary-1024-random_no_rr (query)               257.55       584.33       841.88       0.1593          1.3242            1.3318         8.11
ExhaustiveBinary-1024-random-rf10 (query)                257.55       745.55     1_003.10       0.4456          1.0660            1.0678         8.11
ExhaustiveBinary-1024-random-rf20 (query)                257.55       887.19     1_144.73       0.5824          1.0370            1.0362         8.11
ExhaustiveBinary-1024-random (self)                      257.55     2_675.50     2_933.05       0.4595          1.0714            1.0740         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  348.01       596.01       944.03       0.1599          1.3236            1.3333         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   348.01       747.52     1_095.53       0.4446          1.0658            1.0680         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   348.01       950.00     1_298.02       0.5812          1.0370            1.0362         8.11
ExhaustiveBinary-1024-pca (self)                         348.01     2_570.00     2_918.02       0.4582          1.0716            1.0746         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   85.60       689.09       774.69       0.1292          1.3815            1.3877         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    85.60       897.02       982.61       0.3927          1.0844            1.0833         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    85.60     1_159.01     1_244.60       0.5336          1.0464            1.0444         3.05
ExhaustiveBinary-512-sign (self)                          85.60     2_427.94     2_513.54       0.4063          1.0885            1.0916         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)              446.05        74.42       520.46       0.0763          1.4794            1.4941         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)             446.05        77.96       524.00       0.0762          1.4798            1.4942         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)             446.05        90.49       536.54       0.0762          1.4798            1.4942         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)             446.05       165.67       611.71       0.3000          1.1310            1.1254         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)             446.05       252.68       698.72       0.4213          1.0820            1.0713         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)            446.05       176.01       622.06       0.2985          1.1313            1.1255         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)            446.05       245.31       691.35       0.4202          1.0821            1.0714         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)            446.05       160.30       606.35       0.2985          1.1313            1.1255         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)            446.05       245.55       691.60       0.4201          1.0821            1.0714         2.34
IVF-Binary-256-nl158-random (self)                       446.05       311.38       757.42       0.3184          1.1308            1.1264         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)             471.86        73.88       545.74       0.0885          1.4508            1.4557         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)             471.86        96.35       568.21       0.0885          1.4511            1.4559         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)             471.86        77.30       549.17       0.0885          1.4511            1.4559         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)            471.86       161.20       633.06       0.3263          1.1151            1.1083         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)            471.86       251.12       722.98       0.4512          1.0704            1.0628         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)            471.86       159.95       631.81       0.3262          1.1151            1.1083         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)            471.86       253.60       725.46       0.4511          1.0704            1.0628         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)            471.86       164.99       636.85       0.3262          1.1151            1.1083         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)            471.86       271.29       743.15       0.4511          1.0704            1.0628         2.47
IVF-Binary-256-nl223-random (self)                       471.86       327.75       799.62       0.3439          1.1148            1.1139         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)             552.85        81.69       634.54       0.0950          1.4366            1.4392         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)             552.85        80.22       633.07       0.0950          1.4366            1.4392         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)             552.85        82.35       635.20       0.0950          1.4367            1.4392         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)            552.85       168.67       721.52       0.3368          1.1084            1.1032         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)            552.85       271.91       824.76       0.4647          1.0652            1.0593         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)            552.85       175.22       728.07       0.3367          1.1084            1.1032         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)            552.85       259.54       812.39       0.4646          1.0652            1.0593         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)            552.85       172.50       725.35       0.3366          1.1084            1.1032         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)            552.85       262.75       815.60       0.4646          1.0652            1.0593         2.65
IVF-Binary-256-nl316-random (self)                       552.85       350.10       902.95       0.3558          1.1069            1.1093         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)                 499.50        65.94       565.44       0.0755          1.4847            1.4975         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)                499.50        67.92       567.42       0.0754          1.4851            1.4976         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)                499.50        70.33       569.83       0.0754          1.4851            1.4976         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)                499.50       152.73       652.23       0.2991          1.1304            1.1259         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)                499.50       270.05       769.55       0.4222          1.0801            1.0710         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)               499.50       158.23       657.73       0.2978          1.1306            1.1260         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)               499.50       251.09       750.59       0.4210          1.0803            1.0711         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)               499.50       160.75       660.25       0.2978          1.1306            1.1260         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)               499.50       262.19       761.68       0.4209          1.0803            1.0711         2.34
IVF-Binary-256-nl158-pca (self)                          499.50       312.61       812.11       0.3154          1.1320            1.1293         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)                517.91        72.50       590.41       0.0881          1.4529            1.4576         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)                517.91        74.26       592.17       0.0880          1.4531            1.4578         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)                517.91        78.49       596.40       0.0880          1.4531            1.4578         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)               517.91       170.75       688.66       0.3292          1.1121            1.1063         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)               517.91       254.50       772.41       0.4516          1.0692            1.0621         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)               517.91       159.51       677.42       0.3291          1.1121            1.1063         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)               517.91       253.92       771.82       0.4515          1.0693            1.0621         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)               517.91       163.18       681.09       0.3291          1.1121            1.1063         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)               517.91       262.56       780.47       0.4515          1.0693            1.0621         2.47
IVF-Binary-256-nl223-pca (self)                          517.91       322.59       840.50       0.3476          1.1104            1.1128         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)                620.99        77.87       698.87       0.0950          1.4351            1.4365         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)                620.99        81.61       702.61       0.0950          1.4352            1.4366         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)                620.99        82.15       703.15       0.0950          1.4352            1.4366         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)               620.99       174.24       795.24       0.3402          1.1056            1.1005         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)               620.99       266.38       887.37       0.4617          1.0661            1.0595         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)               620.99       166.66       787.65       0.3401          1.1056            1.1005         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)               620.99       259.49       880.49       0.4616          1.0661            1.0595         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)               620.99       171.68       792.68       0.3401          1.1056            1.1005         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)               620.99       265.06       886.06       0.4616          1.0661            1.0595         2.65
IVF-Binary-256-nl316-pca (self)                          620.99       359.55       980.55       0.3590          1.1032            1.1073         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)              499.10        91.81       590.90       0.1124          1.4040            1.4144         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)             499.10        95.56       594.65       0.1124          1.4041            1.4144         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)             499.10        98.21       597.31       0.1124          1.4041            1.4144         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)             499.10       183.20       682.30       0.3712          1.0931            1.0918         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)             499.10       282.80       781.90       0.4994          1.0541            1.0516         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)            499.10       184.79       683.89       0.3709          1.0932            1.0918         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)            499.10       298.15       797.25       0.4992          1.0541            1.0516         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)            499.10       195.66       694.76       0.3709          1.0932            1.0918         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)            499.10       288.20       787.30       0.4992          1.0541            1.0516         4.36
IVF-Binary-512-nl158-random (self)                       499.10       446.23       945.33       0.3868          1.0949            1.0997         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)             509.18       101.50       610.68       0.1208          1.3872            1.3942         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)             509.18       100.92       610.10       0.1208          1.3873            1.3943         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)             509.18       106.44       615.62       0.1208          1.3873            1.3943         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)            509.18       197.56       706.73       0.3843          1.0875            1.0869         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)            509.18       287.50       796.67       0.5124          1.0510            1.0489         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)            509.18       190.15       699.33       0.3842          1.0875            1.0869         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)            509.18       292.71       801.88       0.5123          1.0510            1.0489         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)            509.18       194.01       703.18       0.3842          1.0875            1.0869         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)            509.18       294.64       803.82       0.5123          1.0510            1.0489         4.49
IVF-Binary-512-nl223-random (self)                       509.18       450.02       959.20       0.3990          1.0902            1.0949         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)             603.92       106.10       710.02       0.1243          1.3793            1.3847         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)             603.92       106.87       710.79       0.1243          1.3794            1.3847         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)             603.92       110.55       714.47       0.1243          1.3794            1.3847         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)            603.92       198.56       802.48       0.3889          1.0856            1.0850         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)            603.92       294.35       898.27       0.5162          1.0501            1.0479         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)            603.92       195.67       799.59       0.3888          1.0856            1.0850         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)            603.92       304.56       908.48       0.5162          1.0501            1.0479         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)            603.92       206.70       810.62       0.3888          1.0856            1.0850         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)            603.92       306.74       910.66       0.5162          1.0501            1.0479         4.67
IVF-Binary-512-nl316-random (self)                       603.92       475.01     1_078.93       0.4032          1.0885            1.0932         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)                 580.12        92.12       672.24       0.1079          1.4130            1.4261         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)                580.12        96.27       676.39       0.1079          1.4130            1.4261         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)                580.12        97.15       677.27       0.1079          1.4130            1.4261         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)                580.12       184.21       764.33       0.3595          1.0984            1.0956         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)                580.12       278.28       858.39       0.4868          1.0578            1.0543         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)               580.12       184.20       764.32       0.3591          1.0984            1.0956         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)               580.12       295.42       875.54       0.4865          1.0579            1.0543         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)               580.12       187.13       767.25       0.3591          1.0984            1.0956         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)               580.12       288.57       868.69       0.4865          1.0579            1.0543         4.36
IVF-Binary-512-nl158-pca (self)                          580.12       422.37     1_002.49       0.3769          1.0989            1.1039         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)                592.72        99.71       692.43       0.1169          1.3942            1.4010         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)                592.72       100.92       693.64       0.1169          1.3942            1.4010         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)                592.72       113.10       705.82       0.1169          1.3942            1.4010         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)               592.72       205.07       797.79       0.3738          1.0919            1.0898         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)               592.72       300.13       892.85       0.5015          1.0541            1.0513         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)               592.72       194.13       786.85       0.3738          1.0919            1.0898         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)               592.72       302.55       895.28       0.5015          1.0541            1.0513         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)               592.72       200.52       793.24       0.3738          1.0919            1.0898         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)               592.72       307.01       899.73       0.5015          1.0541            1.0513         4.49
IVF-Binary-512-nl223-pca (self)                          592.72       467.56     1_060.28       0.3906          1.0936            1.0982         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)                694.61       111.73       806.34       0.1208          1.3851            1.3903         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)                694.61       107.08       801.69       0.1208          1.3851            1.3903         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)                694.61       110.58       805.19       0.1208          1.3851            1.3903         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)               694.61       197.67       892.28       0.3799          1.0892            1.0879         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)               694.61       292.00       986.61       0.5061          1.0528            1.0504         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)               694.61       195.59       890.20       0.3799          1.0892            1.0879         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)               694.61       303.34       997.96       0.5060          1.0528            1.0505         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)               694.61       203.23       897.84       0.3799          1.0892            1.0879         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)               694.61       299.27       993.88       0.5060          1.0528            1.0505         4.67
IVF-Binary-512-nl316-pca (self)                          694.61       488.01     1_182.63       0.3958          1.0911            1.0960         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)             581.72       144.81       726.53       0.1598          1.3236            1.3316         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)            581.72       147.89       729.61       0.1598          1.3236            1.3316         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)            581.72       153.06       734.78       0.1598          1.3236            1.3316         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)            581.72       238.03       819.75       0.4461          1.0659            1.0678         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)            581.72       340.92       922.63       0.5827          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)           581.72       247.53       829.24       0.4461          1.0659            1.0678         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)           581.72       358.54       940.25       0.5827          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)           581.72       251.44       833.15       0.4461          1.0659            1.0678         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)           581.72       361.19       942.91       0.5827          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-random (self)                      581.72       646.74     1_228.45       0.4600          1.0713            1.0740         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)            561.18       154.02       715.19       0.1639          1.3163            1.3250         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)            561.18       154.63       715.81       0.1639          1.3163            1.3250         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)            561.18       164.30       725.48       0.1639          1.3163            1.3250         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)           561.18       248.25       809.42       0.4533          1.0639            1.0658         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)           561.18       348.13       909.31       0.5891          1.0358            1.0351         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)           561.18       251.99       813.17       0.4533          1.0639            1.0658         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)           561.18       359.38       920.56       0.5891          1.0358            1.0351         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)           561.18       267.12       828.29       0.4533          1.0639            1.0658         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)           561.18       391.99       953.17       0.5891          1.0358            1.0351         8.54
IVF-Binary-1024-nl223-random (self)                      561.18       686.57     1_247.75       0.4673          1.0692            1.0718         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)            752.50       176.10       928.60       0.1655          1.3134            1.3215         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)            752.50       163.81       916.30       0.1655          1.3134            1.3215         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)            752.50       170.59       923.08       0.1655          1.3134            1.3215         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)           752.50       281.38     1_033.88       0.4550          1.0634            1.0651         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)           752.50       374.14     1_126.64       0.5912          1.0356            1.0348         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)           752.50       268.53     1_021.02       0.4550          1.0634            1.0651         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)           752.50       381.46     1_133.96       0.5912          1.0356            1.0348         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)           752.50       272.06     1_024.55       0.4550          1.0634            1.0651         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)           752.50       371.07     1_123.57       0.5912          1.0356            1.0348         8.73
IVF-Binary-1024-nl316-random (self)                      752.50       689.27     1_441.77       0.4694          1.0686            1.0714         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)                638.21       147.18       785.39       0.1605          1.3229            1.3332         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)               638.21       153.21       791.41       0.1605          1.3229            1.3332         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)               638.21       155.42       793.62       0.1605          1.3229            1.3332         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)               638.21       245.52       883.72       0.4452          1.0657            1.0680         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)               638.21       341.75       979.95       0.5817          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)              638.21       241.95       880.16       0.4452          1.0657            1.0680         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)              638.21       352.83       991.03       0.5817          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)              638.21       247.49       885.70       0.4452          1.0657            1.0680         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)              638.21       352.23       990.44       0.5817          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-pca (self)                         638.21       653.94     1_292.15       0.4588          1.0715            1.0746         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)               677.31       155.24       832.55       0.1641          1.3158            1.3264         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)               677.31       160.63       837.94       0.1641          1.3158            1.3264         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)               677.31       165.20       842.51       0.1641          1.3158            1.3264         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)              677.31       273.01       950.33       0.4515          1.0638            1.0664         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)              677.31       365.18     1_042.49       0.5882          1.0359            1.0353         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)              677.31       248.67       925.99       0.4515          1.0638            1.0664         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)              677.31       358.33     1_035.64       0.5882          1.0359            1.0353         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)              677.31       255.62       932.93       0.4515          1.0638            1.0664         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)              677.31       365.09     1_042.40       0.5882          1.0359            1.0353         8.54
IVF-Binary-1024-nl223-pca (self)                         677.31       677.25     1_354.56       0.4657          1.0695            1.0725         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)               749.84       164.18       914.02       0.1656          1.3134            1.3240         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)               749.84       162.18       912.02       0.1656          1.3134            1.3240         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)               749.84       178.14       927.99       0.1656          1.3134            1.3240         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)              749.84       256.32     1_006.16       0.4539          1.0631            1.0657         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)              749.84       361.62     1_111.46       0.5899          1.0356            1.0350         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)              749.84       261.24     1_011.08       0.4539          1.0631            1.0657         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)              749.84       371.55     1_121.39       0.5899          1.0356            1.0350         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)              749.84       264.82     1_014.66       0.4539          1.0631            1.0657         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)              749.84       383.27     1_133.12       0.5899          1.0356            1.0350         8.73
IVF-Binary-1024-nl316-pca (self)                         749.84       701.33     1_451.17       0.4677          1.0689            1.0720         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)                378.18       295.51       673.69       0.1292          1.3813            1.3870         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)               378.18       303.28       681.46       0.1292          1.3813            1.3870         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)               378.18       302.48       680.66       0.1292          1.3813            1.3870         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)               378.18       363.75       741.93       0.3940          1.0840            1.0833         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)               378.18       674.06     1_052.24       0.5343          1.0463            1.0444         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)              378.18       373.18       751.36       0.3939          1.0840            1.0833         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)              378.18       663.36     1_041.55       0.5343          1.0463            1.0444         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)              378.18       382.95       761.14       0.3939          1.0840            1.0833         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)              378.18       660.53     1_038.72       0.5343          1.0463            1.0444         3.36
IVF-Binary-512-nl158-sign (self)                         378.18     1_010.83     1_389.01       0.4072          1.0883            1.0915         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               380.11       293.04       673.15       0.1295          1.3814            1.3865         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               380.11       301.39       681.50       0.1295          1.3814            1.3865         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               380.11       306.14       686.25       0.1295          1.3814            1.3865         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              380.11       392.78       772.89       0.3991          1.0819            1.0817         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              380.11       675.32     1_055.43       0.5376          1.0456            1.0440         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              380.11       392.19       772.30       0.3991          1.0819            1.0817         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              380.11       703.17     1_083.28       0.5375          1.0456            1.0440         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              380.11       375.09       755.20       0.3991          1.0819            1.0817         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              380.11       663.58     1_043.69       0.5375          1.0456            1.0440         3.49
IVF-Binary-512-nl223-sign (self)                         380.11     1_020.79     1_400.90       0.4125          1.0864            1.0898         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)               571.34       309.21       880.56       0.1294          1.3810            1.3867         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)               571.34       307.82       879.16       0.1294          1.3810            1.3867         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)               571.34       311.08       882.43       0.1294          1.3810            1.3867         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)              571.34       376.38       947.72       0.4000          1.0814            1.0813         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)              571.34       669.28     1_240.62       0.5386          1.0452            1.0437         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)              571.34       372.63       943.97       0.4000          1.0814            1.0813         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)              571.34       741.01     1_312.35       0.5385          1.0452            1.0437         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)              571.34       404.94       976.28       0.4000          1.0814            1.0813         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)              571.34       715.09     1_286.43       0.5385          1.0452            1.0437         3.67
IVF-Binary-512-nl316-sign (self)                         571.34     1_046.32     1_617.66       0.4132          1.0861            1.0894         3.67
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
Exhaustive (query)                                       102.74     1_927.76     2_030.50       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        102.74     6_577.35     6_680.09       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                196.68       286.83       483.52       0.0662          1.3769            1.3786         2.28
ExhaustiveBinary-256-random-rf10 (query)                 196.68       433.74       630.43       0.2741          1.1112            1.1017         2.28
ExhaustiveBinary-256-random-rf20 (query)                 196.68       576.39       773.07       0.3877          1.0706            1.0590         2.28
ExhaustiveBinary-256-random (self)                       196.68     1_303.84     1_500.53       0.2874          1.1077            1.0994         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   403.18       289.43       692.61       0.0653          1.3793            1.3770         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    403.18       431.08       834.26       0.2701          1.1138            1.1018         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    403.18       571.62       974.79       0.3852          1.0726            1.0585         2.28
ExhaustiveBinary-256-pca (self)                          403.18     1_309.77     1_712.95       0.2820          1.1100            1.0990         2.28
ExhaustiveBinary-512-random_no_rr (query)                300.58       422.70       723.28       0.0934          1.3249            1.3316         4.55
ExhaustiveBinary-512-random-rf10 (query)                 300.58       595.43       896.01       0.3220          1.0860            1.0806         4.55
ExhaustiveBinary-512-random-rf20 (query)                 300.58       778.26     1_078.83       0.4345          1.0527            1.0485         4.55
ExhaustiveBinary-512-random (self)                       300.58     1_844.64     2_145.22       0.3344          1.0826            1.0842         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   506.61       431.49       938.09       0.0951          1.3226            1.3263         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    506.61       578.72     1_085.33       0.3246          1.0847            1.0788         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    506.61       745.52     1_252.13       0.4400          1.0517            1.0468         4.55
ExhaustiveBinary-512-pca (self)                          506.61     1_816.86     2_323.47       0.3361          1.0819            1.0825         4.55
ExhaustiveBinary-1024-random_no_rr (query)               499.53       635.34     1_134.87       0.1319          1.2685            1.2723         9.11
ExhaustiveBinary-1024-random-rf10 (query)                499.53       808.54     1_308.07       0.3742          1.0644            1.0666         9.11
ExhaustiveBinary-1024-random-rf20 (query)                499.53       982.85     1_482.38       0.4906          1.0386            1.0390         9.11
ExhaustiveBinary-1024-random (self)                      499.53     2_661.69     3_161.22       0.3823          1.0666            1.0715         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  692.59       652.65     1_345.24       0.1355          1.2623            1.2651         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   692.59       827.88     1_520.46       0.3804          1.0622            1.0643         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   692.59       993.36     1_685.95       0.4993          1.0369            1.0374         9.11
ExhaustiveBinary-1024-pca (self)                         692.59     2_705.58     3_398.17       0.3870          1.0651            1.0695         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  128.56       856.80       985.36       0.1284          1.2822            1.2821         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   128.56       941.41     1_069.98       0.3618          1.0706            1.0699         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   128.56     1_460.98     1_589.54       0.4847          1.0407            1.0395         4.58
ExhaustiveBinary-768-sign (self)                         128.56     3_050.46     3_179.03       0.3694          1.0716            1.0747         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)              540.67        95.42       636.09       0.0686          1.3708            1.3769         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)             540.67        96.29       636.96       0.0685          1.3711            1.3770         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)             540.67       100.63       641.30       0.0685          1.3711            1.3770         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)             540.67       195.82       736.49       0.2782          1.1103            1.1014         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)             540.67       308.34       849.02       0.3901          1.0700            1.0589         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)            540.67       191.19       731.86       0.2770          1.1104            1.1014         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)            540.67       308.50       849.17       0.3894          1.0701            1.0589         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)            540.67       201.01       741.68       0.2769          1.1104            1.1014         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)            540.67       310.00       850.67       0.3893          1.0701            1.0589         2.74
IVF-Binary-256-nl158-random (self)                       540.67       389.23       929.90       0.2904          1.1068            1.0992         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)             574.52        94.95       669.47       0.0783          1.3507            1.3543         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)             574.52        94.73       669.25       0.0783          1.3508            1.3543         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)             574.52        97.00       671.52       0.0783          1.3508            1.3543         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)            574.52       205.03       779.55       0.3028          1.0946            1.0866         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)            574.52       318.58       893.10       0.4175          1.0588            1.0517         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)            574.52       202.82       777.34       0.3028          1.0946            1.0866         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)            574.52       322.97       897.49       0.4174          1.0588            1.0517         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)            574.52       207.80       782.32       0.3028          1.0946            1.0866         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)            574.52       328.56       903.08       0.4174          1.0588            1.0517         2.93
IVF-Binary-256-nl223-random (self)                       574.52       428.05     1_002.57       0.3172          1.0895            1.0882         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)             701.57       106.44       808.01       0.0851          1.3378            1.3389         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)             701.57       104.02       805.59       0.0851          1.3379            1.3390         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)             701.57       106.93       808.50       0.0851          1.3380            1.3390         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)            701.57       215.98       917.55       0.3176          1.0865            1.0809         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)            701.57       335.23     1_036.80       0.4309          1.0543            1.0490         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)            701.57       212.15       913.72       0.3175          1.0865            1.0809         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)            701.57       330.61     1_032.18       0.4308          1.0543            1.0490         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)            701.57       214.01       915.58       0.3174          1.0865            1.0809         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)            701.57       334.91     1_036.49       0.4307          1.0543            1.0490         3.21
IVF-Binary-256-nl316-random (self)                       701.57       467.75     1_169.32       0.3311          1.0807            1.0831         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)                 742.18        85.18       827.36       0.0678          1.3725            1.3752         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)                742.18        86.34       828.52       0.0677          1.3726            1.3752         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)                742.18        89.85       832.03       0.0677          1.3726            1.3752         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)                742.18       196.54       938.73       0.2747          1.1123            1.1014         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)                742.18       314.47     1_056.65       0.3883          1.0717            1.0584         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)               742.18       191.91       934.09       0.2735          1.1124            1.1014         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)               742.18       307.34     1_049.52       0.3874          1.0717            1.0585         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)               742.18       193.04       935.22       0.2735          1.1124            1.1014         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)               742.18       310.26     1_052.44       0.3873          1.0717            1.0585         2.74
IVF-Binary-256-nl158-pca (self)                          742.18       395.74     1_137.92       0.2853          1.1087            1.0988         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)                779.27        94.63       873.90       0.0776          1.3536            1.3551         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)                779.27        93.43       872.70       0.0776          1.3536            1.3551         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)                779.27        96.15       875.42       0.0776          1.3536            1.3551         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)               779.27       215.99       995.26       0.2987          1.0962            1.0873         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)               779.27       323.42     1_102.69       0.4151          1.0600            1.0520         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)               779.27       202.35       981.62       0.2986          1.0962            1.0873         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)               779.27       320.08     1_099.35       0.4150          1.0600            1.0520         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)               779.27       211.13       990.40       0.2986          1.0962            1.0873         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)               779.27       324.19     1_103.46       0.4149          1.0600            1.0520         2.93
IVF-Binary-256-nl223-pca (self)                          779.27       428.39     1_207.66       0.3109          1.0907            1.0885         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)                906.25       102.13     1_008.38       0.0848          1.3398            1.3382         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)                906.25       101.73     1_007.98       0.0848          1.3399            1.3382         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)                906.25       107.23     1_013.48       0.0848          1.3399            1.3382         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)               906.25       214.30     1_120.55       0.3102          1.0896            1.0821         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)               906.25       336.94     1_243.19       0.4249          1.0567            1.0497         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)               906.25       211.83     1_118.08       0.3101          1.0897            1.0821         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)               906.25       343.27     1_249.52       0.4248          1.0567            1.0497         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)               906.25       217.95     1_124.20       0.3101          1.0897            1.0821         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)               906.25       339.86     1_246.11       0.4247          1.0567            1.0497         3.21
IVF-Binary-256-nl316-pca (self)                          906.25       485.19     1_391.44       0.3223          1.0839            1.0844         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)              661.57       121.81       783.38       0.0948          1.3227            1.3308         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)             661.57       124.39       785.96       0.0947          1.3228            1.3308         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)             661.57       138.71       800.28       0.0947          1.3228            1.3308         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)             661.57       235.20       896.77       0.3234          1.0857            1.0806         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)             661.57       355.75     1_017.32       0.4355          1.0524            1.0484         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)            661.57       248.28       909.85       0.3230          1.0857            1.0806         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)            661.57       354.57     1_016.14       0.4354          1.0524            1.0484         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)            661.57       243.03       904.60       0.3230          1.0857            1.0806         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)            661.57       369.35     1_030.92       0.4354          1.0524            1.0484         5.02
IVF-Binary-512-nl158-random (self)                       661.57       565.32     1_226.89       0.3354          1.0823            1.0842         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)             685.94       130.69       816.62       0.1036          1.3077            1.3089         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)             685.94       138.09       824.03       0.1036          1.3077            1.3089         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)             685.94       135.49       821.43       0.1036          1.3077            1.3089         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)            685.94       248.87       934.81       0.3368          1.0788            1.0763         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)            685.94       364.41     1_050.34       0.4484          1.0488            1.0463         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)            685.94       246.56       932.50       0.3368          1.0788            1.0763         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)            685.94       366.55     1_052.49       0.4484          1.0488            1.0463         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)            685.94       248.30       934.24       0.3368          1.0788            1.0763         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)            685.94       389.25     1_075.19       0.4484          1.0488            1.0463         5.21
IVF-Binary-512-nl223-random (self)                       685.94       593.99     1_279.92       0.3474          1.0766            1.0803         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)             812.41       140.88       953.29       0.1079          1.3002            1.2990         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)             812.41       139.75       952.15       0.1079          1.3002            1.2990         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)             812.41       149.35       961.76       0.1079          1.3002            1.2990         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)            812.41       258.22     1_070.63       0.3426          1.0764            1.0745         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)            812.41       387.43     1_199.84       0.4546          1.0469            1.0449         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)            812.41       251.46     1_063.87       0.3426          1.0764            1.0745         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)            812.41       376.64     1_189.05       0.4545          1.0469            1.0449         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)            812.41       257.93     1_070.33       0.3426          1.0764            1.0745         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)            812.41       388.70     1_201.10       0.4545          1.0469            1.0449         5.48
IVF-Binary-512-nl316-random (self)                       812.41       646.14     1_458.55       0.3530          1.0743            1.0786         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)                 853.65       124.55       978.20       0.0962          1.3205            1.3253         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)                853.65       122.29       975.94       0.0961          1.3206            1.3253         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)                853.65       125.47       979.12       0.0961          1.3206            1.3253         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)                853.65       241.09     1_094.74       0.3262          1.0845            1.0787         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)                853.65       352.06     1_205.72       0.4407          1.0516            1.0467         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)               853.65       245.06     1_098.71       0.3256          1.0845            1.0787         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)               853.65       356.97     1_210.62       0.4406          1.0516            1.0467         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)               853.65       242.59     1_096.24       0.3256          1.0845            1.0787         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)               853.65       360.46     1_214.11       0.4406          1.0516            1.0467         5.02
IVF-Binary-512-nl158-pca (self)                          853.65       576.98     1_430.63       0.3369          1.0817            1.0825         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)                887.29       133.33     1_020.62       0.1043          1.3066            1.3057         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)                887.29       131.23     1_018.52       0.1043          1.3066            1.3057         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)                887.29       134.23     1_021.52       0.1043          1.3066            1.3057         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)               887.29       257.37     1_144.66       0.3375          1.0787            1.0753         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)               887.29       364.98     1_252.28       0.4530          1.0478            1.0448         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)               887.29       245.22     1_132.51       0.3375          1.0787            1.0753         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)               887.29       371.37     1_258.66       0.4530          1.0478            1.0448         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)               887.29       249.55     1_136.84       0.3375          1.0787            1.0753         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)               887.29       373.87     1_261.16       0.4530          1.0478            1.0448         5.21
IVF-Binary-512-nl223-pca (self)                          887.29       607.09     1_494.38       0.3473          1.0770            1.0796         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)              1_024.82       142.37     1_167.19       0.1081          1.2992            1.2963         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)              1_024.82       142.34     1_167.16       0.1081          1.2993            1.2963         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)              1_024.82       144.59     1_169.41       0.1081          1.2993            1.2963         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)             1_024.82       258.40     1_283.22       0.3443          1.0757            1.0732         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)             1_024.82       384.29     1_409.11       0.4586          1.0463            1.0439         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)             1_024.82       256.95     1_281.77       0.3442          1.0757            1.0732         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)             1_024.82       377.61     1_402.43       0.4585          1.0463            1.0439         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)             1_024.82       256.04     1_280.86       0.3442          1.0757            1.0732         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)             1_024.82       391.43     1_416.25       0.4585          1.0463            1.0439         5.48
IVF-Binary-512-nl316-pca (self)                        1_024.82       641.54     1_666.36       0.3530          1.0746            1.0780         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)             854.61       198.64     1_053.25       0.1327          1.2679            1.2722         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)            854.61       197.60     1_052.21       0.1327          1.2680            1.2722         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)            854.61       205.71     1_060.32       0.1327          1.2680            1.2722         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)            854.61       316.08     1_170.69       0.3746          1.0644            1.0666         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)            854.61       462.26     1_316.87       0.4908          1.0386            1.0390         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)           854.61       323.52     1_178.13       0.3745          1.0644            1.0666         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)           854.61       465.78     1_320.39       0.4907          1.0386            1.0390         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)           854.61       346.58     1_201.19       0.3745          1.0644            1.0666         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)           854.61       465.71     1_320.32       0.4907          1.0386            1.0390         9.57
IVF-Binary-1024-nl158-random (self)                      854.61       867.68     1_722.28       0.3825          1.0665            1.0715         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)            896.13       207.79     1_103.92       0.1370          1.2612            1.2658         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)            896.13       211.39     1_107.52       0.1370          1.2612            1.2658         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)            896.13       211.53     1_107.67       0.1370          1.2612            1.2658         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)           896.13       332.43     1_228.56       0.3801          1.0625            1.0654         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)           896.13       478.44     1_374.57       0.4966          1.0375            1.0380         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)           896.13       334.52     1_230.65       0.3801          1.0625            1.0654         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)           896.13       469.78     1_365.91       0.4966          1.0375            1.0380         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)           896.13       343.19     1_239.32       0.3801          1.0625            1.0654         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)           896.13       477.72     1_373.86       0.4966          1.0375            1.0380         9.76
IVF-Binary-1024-nl223-random (self)                      896.13       900.06     1_796.19       0.3880          1.0649            1.0699         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)          1_019.95       220.17     1_240.12       0.1388          1.2580            1.2631        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)          1_019.95       218.76     1_238.71       0.1388          1.2580            1.2631        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)          1_019.95       223.76     1_243.71       0.1388          1.2580            1.2631        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)         1_019.95       364.12     1_384.07       0.3834          1.0615            1.0643        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)         1_019.95       497.98     1_517.93       0.5004          1.0368            1.0374        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)         1_019.95       356.49     1_376.44       0.3834          1.0615            1.0643        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)         1_019.95       485.38     1_505.33       0.5004          1.0368            1.0374        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)         1_019.95       367.05     1_387.00       0.3834          1.0615            1.0643        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)         1_019.95       491.75     1_511.70       0.5004          1.0368            1.0374        10.04
IVF-Binary-1024-nl316-random (self)                    1_019.95       969.57     1_989.52       0.3911          1.0639            1.0689        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)              1_059.74       198.79     1_258.52       0.1360          1.2618            1.2650         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)             1_059.74       200.53     1_260.27       0.1360          1.2618            1.2650         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)             1_059.74       203.11     1_262.85       0.1360          1.2618            1.2650         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)             1_059.74       333.35     1_393.08       0.3808          1.0621            1.0643         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)             1_059.74       448.78     1_508.51       0.4995          1.0369            1.0374         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)            1_059.74       334.71     1_394.45       0.3807          1.0621            1.0643         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)            1_059.74       461.58     1_521.32       0.4995          1.0369            1.0374         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)            1_059.74       330.41     1_390.15       0.3807          1.0621            1.0643         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)            1_059.74       466.92     1_526.66       0.4995          1.0369            1.0374         9.57
IVF-Binary-1024-nl158-pca (self)                       1_059.74       884.80     1_944.54       0.3871          1.0651            1.0695         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_087.42       209.59     1_297.02       0.1396          1.2560            1.2601         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_087.42       208.90     1_296.32       0.1396          1.2560            1.2601         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_087.42       214.32     1_301.74       0.1396          1.2560            1.2601         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_087.42       339.00     1_426.42       0.3857          1.0604            1.0631         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_087.42       464.73     1_552.16       0.5050          1.0360            1.0365         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_087.42       338.01     1_425.43       0.3857          1.0604            1.0631         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_087.42       469.19     1_556.61       0.5050          1.0360            1.0365         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_087.42       346.61     1_434.03       0.3857          1.0604            1.0631         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_087.42       483.65     1_571.07       0.5050          1.0360            1.0365         9.76
IVF-Binary-1024-nl223-pca (self)                       1_087.42       906.67     1_994.10       0.3923          1.0636            1.0682         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_236.52       220.85     1_457.37       0.1414          1.2531            1.2577        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_236.52       220.26     1_456.78       0.1414          1.2531            1.2577        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_236.52       223.93     1_460.45       0.1414          1.2531            1.2577        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_236.52       355.41     1_591.93       0.3890          1.0595            1.0623        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_236.52       494.44     1_730.96       0.5081          1.0354            1.0361        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_236.52       352.86     1_589.38       0.3890          1.0595            1.0623        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_236.52       484.53     1_721.05       0.5081          1.0354            1.0361        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_236.52       364.32     1_600.84       0.3890          1.0595            1.0623        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_236.52       495.83     1_732.35       0.5081          1.0354            1.0361        10.04
IVF-Binary-1024-nl316-pca (self)                       1_236.52       956.17     2_192.69       0.3955          1.0627            1.0672        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)                484.19       407.78       891.97       0.1284          1.2834            1.2814         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)               484.19       411.41       895.59       0.1284          1.2834            1.2814         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)               484.19       413.88       898.07       0.1284          1.2834            1.2814         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)               484.19       503.40       987.59       0.3625          1.0704            1.0698         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)               484.19     1_064.12     1_548.31       0.4852          1.0406            1.0396         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)              484.19       558.72     1_042.90       0.3625          1.0704            1.0698         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)              484.19     1_005.93     1_490.12       0.4852          1.0406            1.0396         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)              484.19       529.13     1_013.32       0.3625          1.0704            1.0698         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)              484.19       991.45     1_475.63       0.4852          1.0406            1.0396         5.04
IVF-Binary-768-nl158-sign (self)                         484.19     1_551.67     2_035.86       0.3700          1.0715            1.0746         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)               779.12       420.72     1_199.84       0.1285          1.2800            1.2812         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)               779.12       441.53     1_220.65       0.1285          1.2800            1.2812         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)               779.12       444.98     1_224.10       0.1285          1.2800            1.2812         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)              779.12       520.67     1_299.79       0.3665          1.0688            1.0688         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)              779.12       939.78     1_718.90       0.4875          1.0400            1.0392         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)              779.12       519.64     1_298.76       0.3665          1.0688            1.0688         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)              779.12       928.43     1_707.55       0.4874          1.0400            1.0392         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)              779.12       521.36     1_300.48       0.3665          1.0688            1.0688         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)              779.12       931.92     1_711.04       0.4874          1.0400            1.0392         5.23
IVF-Binary-768-nl223-sign (self)                         779.12     1_465.54     2_244.66       0.3733          1.0702            1.0737         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)               656.46       423.45     1_079.91       0.1282          1.2804            1.2812         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)               656.46       433.89     1_090.35       0.1282          1.2804            1.2812         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)               656.46       427.24     1_083.70       0.1282          1.2804            1.2812         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)              656.46       523.82     1_180.28       0.3668          1.0684            1.0687         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)              656.46       932.72     1_589.18       0.4884          1.0398            1.0392         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)              656.46       519.06     1_175.52       0.3668          1.0684            1.0687         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)              656.46       941.22     1_597.68       0.4884          1.0398            1.0392         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)              656.46       526.98     1_183.44       0.3668          1.0684            1.0687         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)              656.46       938.25     1_594.71       0.4884          1.0398            1.0392         5.51
IVF-Binary-768-nl316-sign (self)                         656.46     1_502.10     2_158.56       0.3742          1.0700            1.0735         5.51
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
Exhaustive (query)                                        35.19       761.28       796.47       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         35.19     2_344.42     2_379.61       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                 76.73       248.79       325.51       0.5519          1.8826            1.5884         1.78
ExhaustiveBinary-256-random-rf10 (query)                  76.73       371.35       448.07       0.9881          1.0022            1.0000         1.78
ExhaustiveBinary-256-random-rf20 (query)                  76.73       600.73       677.46       0.9980          1.0003            1.0000         1.78
ExhaustiveBinary-256-random (self)                        76.73     1_317.40     1_394.13       0.9881          1.0022            1.0000         1.78
ExhaustiveBinary-256-pca_no_rr (query)                    97.28       247.98       345.25       0.5930          1.6081            1.4152         1.78
ExhaustiveBinary-256-pca-rf10 (query)                     97.28       364.75       462.03       0.9919          1.0013            1.0000         1.78
ExhaustiveBinary-256-pca-rf20 (query)                     97.28       478.37       575.65       0.9988          1.0001            1.0000         1.78
ExhaustiveBinary-256-pca (self)                           97.28     1_185.43     1_282.70       0.9915          1.0014            1.0000         1.78
ExhaustiveBinary-512-random_no_rr (query)                 83.83       379.12       462.96       0.6306          1.5767            1.3633         3.55
ExhaustiveBinary-512-random-rf10 (query)                  83.83       550.91       634.74       0.9975          1.0004            1.0000         3.55
ExhaustiveBinary-512-random-rf20 (query)                  83.83       628.43       712.27       0.9998          1.0000            1.0000         3.55
ExhaustiveBinary-512-random (self)                        83.83     1_706.60     1_790.43       0.9973          1.0004            1.0000         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   109.24       361.47       470.71       0.6479          1.4884            1.3147         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    109.24       492.64       601.88       0.9983          1.0002            1.0000         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    109.24       607.15       716.39       0.9998          1.0000            1.0000         3.55
ExhaustiveBinary-512-pca (self)                          109.24     1_686.37     1_795.61       0.9981          1.0002            1.0000         3.55
ExhaustiveBinary-1024-random_no_rr (query)               116.65       508.84       625.48       0.6758          1.4452            1.2804         7.10
ExhaustiveBinary-1024-random-rf10 (query)                116.65       651.64       768.29       0.9995          1.0001            1.0000         7.10
ExhaustiveBinary-1024-random-rf20 (query)                116.65       765.66       882.31       0.9999          1.0000            1.0000         7.10
ExhaustiveBinary-1024-random (self)                      116.65     2_362.56     2_479.21       0.9993          1.0001            1.0000         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  148.17       562.29       710.46       0.6838          1.4142            1.2651         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   148.17       654.81       802.98       0.9996          1.0000            1.0000         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   148.17       851.48       999.65       1.0000          1.0000            1.0000         7.10
ExhaustiveBinary-1024-pca (self)                         148.17     2_176.07     2_324.24       0.9995          1.0001            1.0000         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   43.30       473.97       517.27       0.0376         19.4734           14.8778         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    43.30       473.76       517.07       0.1617          2.7567            2.6548         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    43.30       722.92       766.23       0.2739          1.9837            1.9249         1.53
ExhaustiveBinary-256-sign (self)                          43.30     1_580.80     1_624.11       0.1691          2.7353            2.6299         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)              450.05        60.83       510.88       0.5656          1.6695            1.5131         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)             450.05        71.37       521.42       0.5589          1.7299            1.5498         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)             450.05        80.39       530.44       0.5568          1.7640            1.5630         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)             450.05       123.30       573.35       0.9903          1.0018            1.0000         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)             450.05       192.31       642.35       0.9968          1.0006            1.0000         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)            450.05       147.20       597.25       0.9907          1.0016            1.0000         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)            450.05       186.26       636.31       0.9986          1.0002            1.0000         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)            450.05       138.17       588.22       0.9898          1.0018            1.0000         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)            450.05       193.69       643.74       0.9984          1.0002            1.0000         1.93
IVF-Binary-256-nl158-random (self)                       450.05       337.57       787.62       0.9904          1.0017            1.0000         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             499.14        50.51       549.64       0.5629          1.6756            1.5245         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             499.14        54.98       554.12       0.5606          1.7006            1.5423         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             499.14        62.97       562.11       0.5578          1.7444            1.5592         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            499.14       113.51       612.64       0.9912          1.0015            1.0000         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            499.14       166.70       665.83       0.9984          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            499.14       114.85       613.99       0.9909          1.0015            1.0000         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            499.14       172.10       671.24       0.9987          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            499.14       124.18       623.31       0.9900          1.0017            1.0000         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            499.14       183.32       682.46       0.9985          1.0002            1.0000         2.00
IVF-Binary-256-nl223-random (self)                       499.14       290.84       789.98       0.9908          1.0016            1.0000         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             641.22        53.41       694.63       0.5619          1.6821            1.5289         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             641.22        54.75       695.97       0.5608          1.6947            1.5368         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             641.22        60.91       702.13       0.5581          1.7362            1.5552         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            641.22       113.56       754.78       0.9917          1.0014            1.0000         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            641.22       170.40       811.62       0.9987          1.0002            1.0000         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            641.22       113.57       754.78       0.9914          1.0014            1.0000         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            641.22       172.77       813.99       0.9988          1.0002            1.0000         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            641.22       120.00       761.21       0.9904          1.0017            1.0000         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            641.22       183.96       825.18       0.9986          1.0002            1.0000         2.09
IVF-Binary-256-nl316-random (self)                       641.22       277.91       919.13       0.9912          1.0015            1.0000         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)                 462.23        48.98       511.20       0.6039          1.4880            1.3749         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)                462.23        59.06       521.29       0.5989          1.5218            1.3915         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)                462.23        90.73       552.96       0.5975          1.5419            1.3962         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)                462.23       126.59       588.81       0.9926          1.0013            1.0000         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)                462.23       194.75       656.97       0.9972          1.0006            1.0000         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)               462.23       147.87       610.10       0.9933          1.0010            1.0000         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)               462.23       204.25       666.47       0.9991          1.0001            1.0000         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)               462.23       151.82       614.05       0.9927          1.0011            1.0000         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)               462.23       204.18       666.40       0.9990          1.0001            1.0000         1.93
IVF-Binary-256-nl158-pca (self)                          462.23       360.37       822.60       0.9929          1.0011            1.0000         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                572.97        57.34       630.31       0.6019          1.4944            1.3798         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                572.97        56.11       629.08       0.6000          1.5083            1.3871         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                572.97        62.26       635.23       0.5980          1.5316            1.3961         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               572.97       111.36       684.33       0.9937          1.0010            1.0000         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               572.97       167.80       740.77       0.9987          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               572.97       114.24       687.21       0.9935          1.0010            1.0000         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               572.97       168.75       741.72       0.9991          1.0001            1.0000         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               572.97       122.21       695.18       0.9929          1.0011            1.0000         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               572.97       177.59       750.56       0.9990          1.0001            1.0000         2.00
IVF-Binary-256-nl223-pca (self)                          572.97       292.18       865.14       0.9931          1.0011            1.0000         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                712.44        55.03       767.47       0.6011          1.4974            1.3831         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                712.44        55.49       767.94       0.6002          1.5051            1.3864         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                712.44        60.39       772.83       0.5984          1.5259            1.3938         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               712.44       114.77       827.21       0.9939          1.0009            1.0000         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               712.44       172.24       884.68       0.9991          1.0001            1.0000         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               712.44       113.79       826.23       0.9938          1.0009            1.0000         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               712.44       167.21       879.66       0.9992          1.0001            1.0000         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               712.44       124.96       837.41       0.9931          1.0011            1.0000         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               712.44       174.95       887.39       0.9991          1.0001            1.0000         2.09
IVF-Binary-256-nl316-pca (self)                          712.44       316.96     1_029.40       0.9934          1.0011            1.0000         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)              463.45        71.62       535.07       0.6411          1.4472            1.3312         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)             463.45        83.83       547.28       0.6352          1.4895            1.3489         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)             463.45        96.56       560.01       0.6333          1.5129            1.3546         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)             463.45       134.97       598.42       0.9965          1.0007            1.0000         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)             463.45       189.97       653.43       0.9978          1.0005            1.0000         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)            463.45       151.03       614.48       0.9982          1.0002            1.0000         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)            463.45       229.66       693.12       0.9999          1.0000            1.0000         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)            463.45       183.07       646.52       0.9979          1.0003            1.0000         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)            463.45       244.77       708.22       0.9998          1.0000            1.0000         3.71
IVF-Binary-512-nl158-random (self)                       463.45       426.80       890.25       0.9979          1.0003            1.0000         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             515.33        69.51       584.83       0.6386          1.4527            1.3378         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             515.33        74.38       589.70       0.6366          1.4693            1.3444         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             515.33        85.98       601.31       0.6340          1.5000            1.3529         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            515.33       132.95       648.28       0.9978          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            515.33       188.74       704.06       0.9993          1.0001            1.0000         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            515.33       136.33       651.66       0.9980          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            515.33       194.76       710.09       0.9998          1.0000            1.0000         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            515.33       148.68       664.01       0.9979          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            515.33       209.52       724.85       0.9998          1.0000            1.0000         3.77
IVF-Binary-512-nl223-random (self)                       515.33       364.46       879.79       0.9978          1.0003            1.0000         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             682.32        73.72       756.05       0.6377          1.4609            1.3400         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             682.32        73.67       756.00       0.6368          1.4691            1.3434         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             682.32        84.23       766.55       0.6346          1.4956            1.3511         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            682.32       134.72       817.05       0.9981          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            682.32       187.69       870.01       0.9996          1.0001            1.0000         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            682.32       137.29       819.61       0.9982          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            682.32       191.52       873.84       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            682.32       145.24       827.56       0.9980          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            682.32       201.69       884.01       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-random (self)                       682.32       361.44     1_043.76       0.9980          1.0003            1.0000         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)                 485.70        70.22       555.92       0.6577          1.3876            1.2898         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)                485.70        85.25       570.96       0.6524          1.4211            1.3034         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)                485.70       101.58       587.28       0.6509          1.4401            1.3080         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)                485.70       137.86       623.56       0.9969          1.0006            1.0000         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)                485.70       189.60       675.30       0.9978          1.0005            1.0000         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)               485.70       148.15       633.85       0.9987          1.0001            1.0000         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)               485.70       207.78       693.48       0.9999          1.0000            1.0000         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)               485.70       165.00       650.70       0.9985          1.0002            1.0000         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)               485.70       224.21       709.91       0.9999          1.0000            1.0000         3.71
IVF-Binary-512-nl158-pca (self)                          485.70       462.92       948.62       0.9986          1.0002            1.0000         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                546.55        69.33       615.88       0.6552          1.3961            1.2931         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                546.55        75.63       622.18       0.6533          1.4086            1.2988         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                546.55        86.68       633.23       0.6513          1.4315            1.3057         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               546.55       133.72       680.27       0.9983          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               546.55       187.61       734.16       0.9993          1.0001            1.0000         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               546.55       138.03       684.58       0.9986          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               546.55       192.66       739.21       0.9998          1.0000            1.0000         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               546.55       173.44       719.99       0.9985          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               546.55       213.11       759.66       0.9999          1.0000            1.0000         3.77
IVF-Binary-512-nl223-pca (self)                          546.55       398.90       945.45       0.9985          1.0002            1.0000         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                682.78        72.13       754.91       0.6544          1.4017            1.2942         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                682.78        74.31       757.09       0.6536          1.4083            1.2967         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                682.78        82.89       765.67       0.6516          1.4283            1.3033         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               682.78       132.51       815.30       0.9986          1.0002            1.0000         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               682.78       189.12       871.90       0.9996          1.0001            1.0000         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               682.78       133.95       816.73       0.9987          1.0001            1.0000         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               682.78       189.93       872.71       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               682.78       144.74       827.52       0.9986          1.0002            1.0000         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               682.78       203.39       886.17       0.9999          1.0000            1.0000         3.86
IVF-Binary-512-nl316-pca (self)                          682.78       352.07     1_034.85       0.9986          1.0002            1.0000         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)             475.57       102.41       577.98       0.6845          1.3518            1.2576         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)            475.57       128.09       603.66       0.6792          1.3861            1.2707         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)            475.57       158.02       633.58       0.6776          1.4035            1.2751         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)            475.57       245.57       721.13       0.9977          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)            475.57       261.21       736.77       0.9978          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)           475.57       227.24       702.81       0.9997          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)           475.57       288.78       764.34       0.9999          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)           475.57       223.67       699.23       0.9996          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)           475.57       295.00       770.56       1.0000          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-random (self)                      475.57       692.43     1_168.00       0.9995          1.0000            1.0000         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            576.04       113.52       689.56       0.6825          1.3585            1.2601         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            576.04       119.08       695.12       0.6806          1.3719            1.2667         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            576.04       142.05       718.09       0.6784          1.3948            1.2738         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           576.04       190.92       766.96       0.9991          1.0002            1.0000         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           576.04       275.31       851.35       0.9994          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           576.04       205.59       781.63       0.9995          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           576.04       255.98       832.02       0.9998          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           576.04       200.54       776.58       0.9996          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           576.04       257.34       833.37       0.9999          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-random (self)                      576.04       501.37     1_077.41       0.9994          1.0001            1.0000         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            718.18       130.61       848.80       0.6815          1.3670            1.2637         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)            718.18       132.36       850.55       0.6806          1.3736            1.2657         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)            718.18       151.06       869.24       0.6786          1.3933            1.2725         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)           718.18       203.44       921.63       0.9994          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)           718.18       261.03       979.22       0.9997          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)           718.18       191.21       909.39       0.9995          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)           718.18       264.06       982.24       0.9998          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)           718.18       228.74       946.92       0.9996          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)           718.18       276.40       994.58       1.0000          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-random (self)                      718.18       497.38     1_215.57       0.9994          1.0001            1.0000         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)                545.06       109.51       654.57       0.6928          1.3299            1.2427         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)               545.06       129.74       674.79       0.6877          1.3593            1.2559         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)               545.06       164.71       709.76       0.6860          1.3758            1.2595         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)               545.06       244.20       789.25       0.9977          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)               545.06       325.51       870.56       0.9978          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)              545.06       228.60       773.66       0.9998          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)              545.06       292.07       837.12       1.0000          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)              545.06       246.21       791.27       0.9997          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)              545.06       280.64       825.70       1.0000          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-pca (self)                         545.06       583.26     1_128.31       0.9997          1.0000            1.0000         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               623.95       104.84       728.79       0.6902          1.3389            1.2452         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               623.95       111.54       735.50       0.6884          1.3495            1.2507         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               623.95       127.82       751.77       0.6863          1.3695            1.2573         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              623.95       170.27       794.23       0.9992          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              623.95       229.33       853.29       0.9994          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              623.95       176.53       800.48       0.9996          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              623.95       238.73       862.68       0.9999          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              623.95       195.12       819.07       0.9996          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              623.95       257.81       881.76       1.0000          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-pca (self)                         623.95       505.03     1_128.98       0.9995          1.0001            1.0000         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               710.18       105.23       815.41       0.6892          1.3446            1.2492         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)               710.18       108.77       818.94       0.6884          1.3504            1.2513         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)               710.18       122.11       832.29       0.6867          1.3679            1.2566         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)              710.18       172.31       882.49       0.9995          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)              710.18       228.44       938.61       0.9997          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)              710.18       173.04       883.22       0.9996          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)              710.18       235.72       945.90       0.9998          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)              710.18       210.78       920.96       0.9997          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)              710.18       254.76       964.94       1.0000          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-pca (self)                         710.18       514.87     1_225.05       0.9995          1.0001            1.0000         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)                413.17       181.25       594.42       0.0687          6.6445            6.1205         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               413.17       196.21       609.38       0.0554          7.8737            7.1934         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               413.17       207.91       621.08       0.0506          8.7363            7.8879         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               413.17       222.26       635.42       0.3993          1.6143            1.5273         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               413.17       394.51       807.67       0.6370          1.2496            1.1927         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              413.17       239.79       652.96       0.3090          1.8465            1.7466         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              413.17       421.80       834.97       0.4807          1.4429            1.3742         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              413.17       250.99       664.15       0.2738          1.9854            1.8794         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              413.17       441.21       854.38       0.4129          1.5612            1.4833         1.68
IVF-Binary-256-nl158-sign (self)                         413.17       716.48     1_129.65       0.3147          1.8346            1.7365         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               473.13       170.50       643.64       0.0659          6.5535            6.0912         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               473.13       176.16       649.29       0.0606          6.9940            6.4957         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               473.13       193.29       666.42       0.0540          7.8607            7.2719         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              473.13       224.41       697.55       0.3659          1.6641            1.5816         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              473.13       382.06       855.20       0.5968          1.2835            1.2268         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              473.13       239.32       712.46       0.3334          1.7511            1.6659         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              473.13       415.45       888.58       0.5313          1.3595            1.2996         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              473.13       253.29       726.42       0.2921          1.9003            1.8127         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              473.13       421.26       894.39       0.4434          1.4929            1.4296         1.75
IVF-Binary-256-nl223-sign (self)                         473.13       645.27     1_118.40       0.3386          1.7410            1.6559         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               642.34       189.51       831.85       0.0661          6.4956            6.0418         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               642.34       235.21       877.55       0.0633          6.7181            6.2205         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               642.34       202.22       844.56       0.0564          7.4857            6.8850         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              642.34       242.11       884.45       0.3650          1.6701            1.5867         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              642.34       393.10     1_035.43       0.5869          1.2940            1.2404         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              642.34       245.58       887.92       0.3481          1.7146            1.6271         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              642.34       442.66     1_085.00       0.5539          1.3321            1.2772         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              642.34       246.81       889.15       0.3059          1.8520            1.7609         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              642.34       410.37     1_052.70       0.4670          1.4544            1.3942         1.84
IVF-Binary-256-nl316-sign (self)                         642.34       618.52     1_260.86       0.3541          1.7014            1.6169         1.84
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
Exhaustive (query)                                        70.99     1_397.90     1_468.90       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         70.99     4_724.46     4_795.46       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                163.17       275.47       438.64       0.5547          1.7646            1.5366         2.03
ExhaustiveBinary-256-random-rf10 (query)                 163.17       414.72       577.89       0.9898          1.0017            1.0000         2.03
ExhaustiveBinary-256-random-rf20 (query)                 163.17       565.30       728.47       0.9985          1.0002            1.0000         2.03
ExhaustiveBinary-256-random (self)                       163.17     1_316.84     1_480.02       0.9899          1.0016            1.0000         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   246.10       271.82       517.92       0.5767          1.6243            1.4311         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    246.10       414.12       660.22       0.9904          1.0016            1.0000         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    246.10       557.73       803.84       0.9984          1.0002            1.0000         2.03
ExhaustiveBinary-256-pca (self)                          246.10     1_315.17     1_561.27       0.9905          1.0016            1.0000         2.03
ExhaustiveBinary-512-random_no_rr (query)                225.46       412.85       638.31       0.6013          1.6760            1.4608         4.05
ExhaustiveBinary-512-random-rf10 (query)                 225.46       579.27       804.73       0.9977          1.0003            1.0000         4.05
ExhaustiveBinary-512-random-rf20 (query)                 225.46       700.30       925.76       0.9998          1.0000            1.0000         4.05
ExhaustiveBinary-512-random (self)                       225.46     1_788.46     2_013.92       0.9975          1.0003            1.0000         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   304.24       405.03       709.27       0.6443          1.4426            1.3064         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    304.24       549.15       853.38       0.9985          1.0002            1.0000         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    304.24       711.16     1_015.39       0.9999          1.0000            1.0000         4.05
ExhaustiveBinary-512-pca (self)                          304.24     1_762.22     2_066.45       0.9984          1.0002            1.0000         4.05
ExhaustiveBinary-1024-random_no_rr (query)               260.42       608.09       868.51       0.6624          1.4553            1.3048         8.11
ExhaustiveBinary-1024-random-rf10 (query)                260.42       769.49     1_029.91       0.9995          1.0001            1.0000         8.11
ExhaustiveBinary-1024-random-rf20 (query)                260.42       917.74     1_178.16       1.0000          1.0000            1.0000         8.11
ExhaustiveBinary-1024-random (self)                      260.42     2_669.95     2_930.37       0.9994          1.0001            1.0000         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  351.51       607.69       959.20       0.6865          1.3603            1.2383         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   351.51       781.04     1_132.54       0.9996          1.0001            1.0000         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   351.51       924.94     1_276.44       1.0000          1.0000            1.0000         8.11
ExhaustiveBinary-1024-pca (self)                         351.51     2_507.46     2_858.96       0.9996          1.0001            1.0000         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   88.46       687.04       775.50       0.0400         18.1511           13.6734         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    88.46       741.32       829.78       0.1821          2.5573            2.4620         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    88.46     1_118.24     1_206.70       0.3140          1.8429            1.7786         3.05
ExhaustiveBinary-512-sign (self)                          88.46     2_512.33     2_600.79       0.1897          2.5286            2.4283         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)              689.92        88.43       778.35       0.5627          1.6434            1.4885         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)             689.92        94.46       784.38       0.5594          1.6750            1.5064         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)             689.92       101.45       791.37       0.5580          1.7034            1.5141         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)             689.92       181.25       871.17       0.9915          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)             689.92       272.47       962.39       0.9978          1.0004            1.0000         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)            689.92       182.93       872.85       0.9913          1.0013            1.0000         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)            689.92       285.57       975.49       0.9988          1.0001            1.0000         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)            689.92       189.60       879.52       0.9907          1.0015            1.0000         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)            689.92       289.66       979.58       0.9987          1.0002            1.0000         2.34
IVF-Binary-256-nl158-random (self)                       689.92       418.44     1_108.36       0.9914          1.0014            1.0000         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)             725.69        76.91       802.60       0.5616          1.6456            1.4978         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)             725.69        83.82       809.51       0.5604          1.6579            1.5060         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)             725.69        91.40       817.08       0.5590          1.6819            1.5132         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)            725.69       188.60       914.28       0.9924          1.0011            1.0000         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)            725.69       268.00       993.68       0.9989          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)            725.69       177.22       902.91       0.9919          1.0012            1.0000         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)            725.69       274.14       999.82       0.9989          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)            725.69       184.20       909.88       0.9911          1.0014            1.0000         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)            725.69       284.28     1_009.97       0.9988          1.0001            1.0000         2.47
IVF-Binary-256-nl223-random (self)                       725.69       385.36     1_111.05       0.9920          1.0012            1.0000         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)             852.95        83.07       936.02       0.5614          1.6442            1.4960         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)             852.95        87.02       939.97       0.5608          1.6514            1.4997         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)             852.95        93.00       945.95       0.5594          1.6719            1.5089         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)            852.95       188.74     1_041.69       0.9924          1.0011            1.0000         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)            852.95       279.71     1_132.66       0.9990          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)            852.95       176.45     1_029.39       0.9921          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)            852.95       279.46     1_132.41       0.9990          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)            852.95       182.82     1_035.76       0.9913          1.0013            1.0000         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)            852.95       292.85     1_145.80       0.9988          1.0001            1.0000         2.65
IVF-Binary-256-nl316-random (self)                       852.95       378.99     1_231.94       0.9922          1.0012            1.0000         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)                 755.13        74.93       830.07       0.5835          1.5315            1.4056         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)                755.13        88.93       844.07       0.5809          1.5555            1.4149         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)                755.13        89.88       845.02       0.5799          1.5748            1.4194         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)                755.13       225.72       980.85       0.9916          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)                755.13       281.51     1_036.65       0.9978          1.0004            1.0000         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)               755.13       177.87       933.01       0.9916          1.0013            1.0000         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)               755.13       293.14     1_048.27       0.9988          1.0002            1.0000         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)               755.13       195.96       951.09       0.9910          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)               755.13       295.65     1_050.78       0.9987          1.0002            1.0000         2.34
IVF-Binary-256-nl158-pca (self)                          755.13       414.89     1_170.03       0.9916          1.0014            1.0000         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)                804.36        77.62       881.98       0.5830          1.5327            1.4106         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)                804.36        93.93       898.29       0.5818          1.5424            1.4155         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)                804.36        88.51       892.87       0.5807          1.5585            1.4200         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)               804.36       176.76       981.12       0.9924          1.0012            1.0000         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)               804.36       271.29     1_075.65       0.9988          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)               804.36       178.56       982.93       0.9921          1.0013            1.0000         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)               804.36       271.49     1_075.85       0.9989          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)               804.36       181.52       985.88       0.9914          1.0014            1.0000         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)               804.36       283.92     1_088.28       0.9987          1.0002            1.0000         2.47
IVF-Binary-256-nl223-pca (self)                          804.36       382.03     1_186.39       0.9921          1.0013            1.0000         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)                919.20        85.43     1_004.63       0.5827          1.5322            1.4087         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)                919.20        98.63     1_017.83       0.5823          1.5369            1.4103         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)                919.20        95.58     1_014.78       0.5810          1.5507            1.4178         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)               919.20       184.74     1_103.94       0.9923          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)               919.20       273.81     1_193.01       0.9990          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)               919.20       174.89     1_094.09       0.9920          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)               919.20       274.32     1_193.52       0.9989          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)               919.20       196.15     1_115.35       0.9914          1.0014            1.0000         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)               919.20       291.97     1_211.17       0.9988          1.0002            1.0000         2.65
IVF-Binary-256-nl316-pca (self)                          919.20       389.23     1_308.43       0.9922          1.0012            1.0000         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)              739.01       116.46       855.46       0.6084          1.5746            1.4241         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)             739.01       117.30       856.30       0.6049          1.6062            1.4408         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)             739.01       137.17       876.18       0.6033          1.6340            1.4480         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)             739.01       200.27       939.28       0.9972          1.0004            1.0000         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)             739.01       298.54     1_037.54       0.9985          1.0003            1.0000         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)            739.01       218.59       957.60       0.9981          1.0002            1.0000         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)            739.01       326.31     1_065.32       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)            739.01       222.69       961.69       0.9979          1.0003            1.0000         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)            739.01       330.49     1_069.50       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-random (self)                       739.01       546.44     1_285.45       0.9979          1.0003            1.0000         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)             777.03       109.52       886.55       0.6072          1.5782            1.4305         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)             777.03       117.65       894.68       0.6057          1.5924            1.4386         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)             777.03       126.36       903.38       0.6043          1.6132            1.4454         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)            777.03       203.53       980.55       0.9982          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)            777.03       301.33     1_078.36       0.9996          1.0000            1.0000         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)            777.03       227.47     1_004.50       0.9983          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)            777.03       322.29     1_099.32       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)            777.03       217.68       994.71       0.9981          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)            777.03       319.95     1_096.98       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-random (self)                       777.03       499.99     1_277.02       0.9982          1.0002            1.0000         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)             922.30       113.23     1_035.53       0.6067          1.5793            1.4339         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)             922.30       114.67     1_036.97       0.6061          1.5861            1.4377         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)             922.30       129.08     1_051.38       0.6044          1.6057            1.4454         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)            922.30       211.47     1_133.77       0.9985          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)            922.30       306.16     1_228.46       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)            922.30       206.77     1_129.07       0.9984          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)            922.30       308.34     1_230.64       0.9999          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)            922.30       216.23     1_138.53       0.9982          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)            922.30       331.79     1_254.09       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-random (self)                       922.30       498.56     1_420.86       0.9983          1.0002            1.0000         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)                 851.12       102.55       953.67       0.6496          1.3874            1.2888         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)                851.12       116.61       967.73       0.6470          1.4043            1.2978         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)                851.12       146.23       997.35       0.6459          1.4173            1.3008         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)                851.12       216.60     1_067.72       0.9975          1.0004            1.0000         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)                851.12       320.24     1_171.36       0.9985          1.0003            1.0000         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)               851.12       232.41     1_083.53       0.9986          1.0001            1.0000         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)               851.12       334.87     1_185.99       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)               851.12       230.89     1_082.01       0.9986          1.0002            1.0000         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)               851.12       324.03     1_175.15       0.9999          1.0000            1.0000         4.36
IVF-Binary-512-nl158-pca (self)                          851.12       533.96     1_385.08       0.9986          1.0002            1.0000         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)                872.68       108.16       980.84       0.6483          1.3915            1.2914         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)                872.68       110.78       983.46       0.6473          1.3993            1.2956         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)                872.68       131.16     1_003.84       0.6464          1.4090            1.2987         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)               872.68       208.75     1_081.42       0.9986          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)               872.68       311.44     1_184.12       0.9996          1.0001            1.0000         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)               872.68       206.82     1_079.50       0.9987          1.0001            1.0000         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)               872.68       306.86     1_179.54       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)               872.68       215.47     1_088.14       0.9986          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)               872.68       322.03     1_194.71       0.9999          1.0000            1.0000         4.49
IVF-Binary-512-nl223-pca (self)                          872.68       499.01     1_371.69       0.9987          1.0002            1.0000         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)                963.69       133.90     1_097.60       0.6480          1.3927            1.2940         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)                963.69       114.99     1_078.68       0.6475          1.3963            1.2954         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)                963.69       124.93     1_088.62       0.6465          1.4067            1.2990         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)               963.69       232.69     1_196.39       0.9988          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)               963.69       304.51     1_268.21       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)               963.69       207.23     1_170.92       0.9987          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)               963.69       315.63     1_279.33       0.9999          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)               963.69       221.91     1_185.60       0.9986          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)               963.69       323.01     1_286.70       0.9999          1.0000            1.0000         4.67
IVF-Binary-512-nl316-pca (self)                          963.69       533.05     1_496.74       0.9987          1.0001            1.0000         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)             843.96       162.65     1_006.61       0.6681          1.3933            1.2856         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)            843.96       180.34     1_024.30       0.6650          1.4133            1.2952         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)            843.96       205.74     1_049.70       0.6636          1.4310            1.3003         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)            843.96       274.39     1_118.35       0.9983          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)            843.96       359.59     1_203.55       0.9986          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)           843.96       276.72     1_120.68       0.9995          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)           843.96       393.11     1_237.07       0.9999          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)           843.96       294.54     1_138.50       0.9995          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)           843.96       410.22     1_254.18       1.0000          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-random (self)                      843.96       786.97     1_630.93       0.9995          1.0001            1.0000         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)            843.04       171.55     1_014.59       0.6666          1.3986            1.2895         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)            843.04       174.54     1_017.58       0.6654          1.4076            1.2936         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)            843.04       189.52     1_032.56       0.6643          1.4203            1.2986         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)           843.04       271.40     1_114.44       0.9994          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)           843.04       377.94     1_220.98       0.9997          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)           843.04       279.24     1_122.28       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)           843.04       378.59     1_221.63       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)           843.04       294.14     1_137.18       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)           843.04       416.20     1_259.24       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-random (self)                      843.04       743.07     1_586.11       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)            963.56       177.84     1_141.40       0.6664          1.3999            1.2907         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)            963.56       177.51     1_141.07       0.6659          1.4040            1.2926         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)            963.56       195.50     1_159.05       0.6646          1.4168            1.2972         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)           963.56       275.10     1_238.65       0.9996          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)           963.56       463.68     1_427.23       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)           963.56       301.91     1_265.47       0.9996          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)           963.56       394.09     1_357.65       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)           963.56       301.61     1_265.17       0.9996          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)           963.56       412.40     1_375.96       1.0000          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-random (self)                      963.56       750.69     1_714.24       0.9996          1.0001            1.0000         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)                899.99       161.02     1_061.01       0.6909          1.3175            1.2259         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)               899.99       186.89     1_086.88       0.6885          1.3324            1.2316         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)               899.99       202.21     1_102.19       0.6876          1.3436            1.2346         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)               899.99       276.35     1_176.34       0.9984          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)               899.99       364.54     1_264.53       0.9986          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)              899.99       280.83     1_180.82       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)              899.99       386.30     1_286.28       0.9999          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)              899.99       294.14     1_194.13       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)              899.99       410.49     1_310.48       1.0000          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-pca (self)                         899.99       791.26     1_691.25       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)               914.98       169.67     1_084.64       0.6897          1.3214            1.2293         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)               914.98       171.65     1_086.63       0.6887          1.3273            1.2309         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)               914.98       198.81     1_113.79       0.6878          1.3368            1.2338         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)              914.98       278.83     1_193.81       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)              914.98       363.11     1_278.08       0.9997          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)              914.98       267.90     1_182.88       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)              914.98       385.07     1_300.04       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)              914.98       293.21     1_208.18       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)              914.98       403.07     1_318.04       1.0000          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-pca (self)                         914.98       766.84     1_681.82       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_047.05       176.89     1_223.94       0.6897          1.3225            1.2300         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_047.05       182.01     1_229.05       0.6892          1.3257            1.2309         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_047.05       189.86     1_236.90       0.6881          1.3346            1.2341         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_047.05       294.87     1_341.92       0.9997          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_047.05       379.24     1_426.28       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_047.05       275.73     1_322.77       0.9997          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_047.05       380.25     1_427.29       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_047.05       291.22     1_338.26       0.9997          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_047.05       415.20     1_462.25       1.0000          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-pca (self)                       1_047.05       728.91     1_775.96       0.9996          1.0000            1.0000         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)                623.08       322.96       946.04       0.0587          7.9788            7.1475         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)               623.08       320.80       943.89       0.0517          9.1464            7.9827         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)               623.08       327.28       950.36       0.0486         10.0153            8.6481         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)               623.08       370.79       993.87       0.3222          1.8338            1.7364         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)               623.08       662.49     1_285.57       0.5149          1.4031            1.3349         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)              623.08       430.56     1_053.64       0.2801          1.9786            1.8635         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)              623.08       689.02     1_312.10       0.4400          1.5255            1.4474         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)              623.08       403.59     1_026.67       0.2590          2.0739            1.9494         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)              623.08       715.77     1_338.85       0.3996          1.6072            1.5243         3.36
IVF-Binary-512-nl158-sign (self)                         623.08     1_136.01     1_759.09       0.2870          1.9578            1.8406         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               693.52       315.37     1_008.89       0.0570          7.8806            7.1787         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               693.52       316.55     1_010.07       0.0540          8.2916            7.4994         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               693.52       352.35     1_045.87       0.0499          9.1818            8.1707         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              693.52       384.14     1_077.66       0.3139          1.8522            1.7521         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              693.52       671.83     1_365.35       0.5026          1.4175            1.3460         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              693.52       375.82     1_069.34       0.2945          1.9145            1.8086         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              693.52       669.02     1_362.54       0.4691          1.4689            1.3964         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              693.52       390.34     1_083.86       0.2686          2.0220            1.8987         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              693.52       690.73     1_384.25       0.4170          1.5688            1.4883         3.49
IVF-Binary-512-nl223-sign (self)                         693.52     1_045.35     1_738.87       0.3016          1.8962            1.7845         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)               755.35       306.26     1_061.61       0.0576          7.7316            7.1191         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)               755.35       299.53     1_054.88       0.0558          7.9302            7.2945         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)               755.35       332.59     1_087.94       0.0519          8.6360            7.8466         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)              755.35       384.06     1_139.41       0.3181          1.8372            1.7369         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)              755.35       686.86     1_442.21       0.5020          1.4131            1.3434         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)              755.35       395.75     1_151.10       0.3081          1.8711            1.7676         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)              755.35       689.19     1_444.55       0.4843          1.4392            1.3711         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)              755.35       411.60     1_166.95       0.2819          1.9709            1.8521         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)              755.35       699.59     1_454.95       0.4341          1.5294            1.4542         3.67
IVF-Binary-512-nl316-sign (self)                         755.35     1_083.15     1_838.50       0.3138          1.8534            1.7425         3.67
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
Exhaustive (query)                                       104.58     1_960.30     2_064.89       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        104.58     6_544.50     6_649.09       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                193.80       292.34       486.14       0.5361          1.8068            1.5908         2.28
ExhaustiveBinary-256-random-rf10 (query)                 193.80       454.04       647.84       0.9868          1.0022            1.0000         2.28
ExhaustiveBinary-256-random-rf20 (query)                 193.80       626.95       820.75       0.9980          1.0003            1.0000         2.28
ExhaustiveBinary-256-random (self)                       193.80     1_405.26     1_599.06       0.9876          1.0021            1.0000         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   403.24       297.41       700.65       0.5754          1.5495            1.4128         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    403.24       463.21       866.45       0.9895          1.0018            1.0000         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    403.24       601.43     1_004.67       0.9983          1.0002            1.0000         2.28
ExhaustiveBinary-256-pca (self)                          403.24     1_388.03     1_791.27       0.9897          1.0017            1.0000         2.28
ExhaustiveBinary-512-random_no_rr (query)                299.60       428.70       728.30       0.5866          1.6778            1.4946         4.55
ExhaustiveBinary-512-random-rf10 (query)                 299.60       591.15       890.75       0.9966          1.0005            1.0000         4.55
ExhaustiveBinary-512-random-rf20 (query)                 299.60       758.56     1_058.16       0.9997          1.0001            1.0000         4.55
ExhaustiveBinary-512-random (self)                       299.60     1_873.03     2_172.63       0.9969          1.0004            1.0000         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   499.47       420.62       920.09       0.6388          1.4217            1.3032         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    499.47       583.34     1_082.81       0.9979          1.0003            1.0000         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    499.47       750.44     1_249.91       0.9998          1.0000            1.0000         4.55
ExhaustiveBinary-512-pca (self)                          499.47     1_912.84     2_412.31       0.9981          1.0002            1.0000         4.55
ExhaustiveBinary-1024-random_no_rr (query)               504.10       641.97     1_146.08       0.6446          1.4909            1.3512         9.11
ExhaustiveBinary-1024-random-rf10 (query)                504.10       824.53     1_328.64       0.9993          1.0001            1.0000         9.11
ExhaustiveBinary-1024-random-rf20 (query)                504.10     1_006.56     1_510.66       0.9999          1.0000            1.0000         9.11
ExhaustiveBinary-1024-random (self)                      504.10     2_729.18     3_233.29       0.9994          1.0001            1.0000         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  699.58       660.57     1_360.15       0.6795          1.3452            1.2483         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   699.58       826.12     1_525.70       0.9996          1.0001            1.0000         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   699.58     1_012.96     1_712.54       1.0000          1.0000            1.0000         9.11
ExhaustiveBinary-1024-pca (self)                         699.58     2_842.91     3_542.49       0.9997          1.0000            1.0000         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  127.47       854.89       982.36       0.0420         17.7082           13.0970         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   127.47       918.93     1_046.40       0.1896          2.5240            2.4052         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   127.47     1_425.80     1_553.27       0.3229          1.8300            1.7348         4.58
ExhaustiveBinary-768-sign (self)                         127.47     2_957.10     3_084.57       0.1997          2.4832            2.3546         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)              981.24       100.53     1_081.77       0.5429          1.7099            1.5461         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)             981.24       111.93     1_093.17       0.5408          1.7331            1.5595         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)             981.24       120.82     1_102.06       0.5397          1.7545            1.5688         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)             981.24       212.01     1_193.24       0.9891          1.0018            1.0000         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)             981.24       336.80     1_318.03       0.9984          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)            981.24       221.65     1_202.88       0.9884          1.0019            1.0000         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)            981.24       344.67     1_325.91       0.9986          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)            981.24       226.16     1_207.40       0.9877          1.0021            1.0000         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)            981.24       348.52     1_329.76       0.9983          1.0002            1.0000         2.74
IVF-Binary-256-nl158-random (self)                       981.24       523.29     1_504.53       0.9891          1.0018            1.0000         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)           1_101.38       105.02     1_206.40       0.5419          1.7184            1.5524         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)           1_101.38       110.36     1_211.74       0.5412          1.7280            1.5571         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)           1_101.38       119.63     1_221.01       0.5401          1.7511            1.5659         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)          1_101.38       216.50     1_317.88       0.9888          1.0018            1.0000         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)          1_101.38       334.99     1_436.37       0.9985          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)          1_101.38       218.30     1_319.69       0.9885          1.0019            1.0000         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)          1_101.38       334.13     1_435.51       0.9985          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)          1_101.38       229.29     1_330.67       0.9877          1.0020            1.0000         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)          1_101.38       346.73     1_448.12       0.9983          1.0002            1.0000         2.93
IVF-Binary-256-nl223-random (self)                     1_101.38       474.02     1_575.41       0.9890          1.0018            1.0000         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)           1_382.48       111.45     1_493.94       0.5422          1.7108            1.5502         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)           1_382.48       112.41     1_494.89       0.5418          1.7157            1.5534         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)           1_382.48       114.11     1_496.59       0.5409          1.7314            1.5597         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)          1_382.48       219.98     1_602.46       0.9891          1.0018            1.0000         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)          1_382.48       341.48     1_723.96       0.9987          1.0002            1.0000         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)          1_382.48       224.09     1_606.57       0.9888          1.0018            1.0000         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)          1_382.48       343.33     1_725.81       0.9986          1.0002            1.0000         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)          1_382.48       227.83     1_610.31       0.9882          1.0020            1.0000         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)          1_382.48       352.97     1_735.46       0.9984          1.0002            1.0000         3.21
IVF-Binary-256-nl316-random (self)                     1_382.48       498.23     1_880.71       0.9894          1.0017            1.0000         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)               1_214.79        94.86     1_309.65       0.5812          1.4959            1.3935         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)              1_214.79       105.03     1_319.82       0.5794          1.5088            1.3993         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)              1_214.79       110.85     1_325.63       0.5787          1.5186            1.4018         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)              1_214.79       212.65     1_427.44       0.9913          1.0014            1.0000         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)              1_214.79       325.26     1_540.05       0.9984          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)             1_214.79       214.28     1_429.07       0.9907          1.0015            1.0000         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)             1_214.79       334.47     1_549.26       0.9987          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)             1_214.79       219.93     1_434.72       0.9902          1.0016            1.0000         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)             1_214.79       341.13     1_555.92       0.9985          1.0002            1.0000         2.74
IVF-Binary-256-nl158-pca (self)                        1_214.79       496.06     1_710.85       0.9910          1.0014            1.0000         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)              1_298.80        97.60     1_396.40       0.5800          1.5020            1.3965         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)              1_298.80       102.43     1_401.23       0.5795          1.5067            1.3989         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)              1_298.80       108.78     1_407.58       0.5787          1.5168            1.4019         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)             1_298.80       212.16     1_510.96       0.9909          1.0015            1.0000         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)             1_298.80       329.49     1_628.29       0.9987          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)             1_298.80       211.41     1_510.20       0.9906          1.0015            1.0000         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)             1_298.80       336.06     1_634.86       0.9987          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)             1_298.80       220.51     1_519.31       0.9901          1.0016            1.0000         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)             1_298.80       354.12     1_652.92       0.9985          1.0002            1.0000         2.93
IVF-Binary-256-nl223-pca (self)                        1_298.80       467.67     1_766.47       0.9909          1.0014            1.0000         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)              1_609.35       119.44     1_728.78       0.5807          1.4991            1.3929         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)              1_609.35       110.11     1_719.46       0.5804          1.5019            1.3943         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)              1_609.35       115.31     1_724.66       0.5799          1.5084            1.3972         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)             1_609.35       226.83     1_836.18       0.9912          1.0014            1.0000         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)             1_609.35       336.22     1_945.57       0.9989          1.0001            1.0000         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)             1_609.35       216.19     1_825.54       0.9909          1.0015            1.0000         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)             1_609.35       343.46     1_952.81       0.9988          1.0001            1.0000         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)             1_609.35       223.21     1_832.55       0.9903          1.0016            1.0000         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)             1_609.35       351.96     1_961.31       0.9986          1.0002            1.0000         3.21
IVF-Binary-256-nl316-pca (self)                        1_609.35       486.12     2_095.47       0.9913          1.0014            1.0000         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)            1_093.38       129.86     1_223.24       0.5925          1.6007            1.4618         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)           1_093.38       147.78     1_241.16       0.5899          1.6229            1.4748         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)           1_093.38       159.81     1_253.19       0.5889          1.6417            1.4805         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)           1_093.38       254.81     1_348.19       0.9972          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)           1_093.38       368.21     1_461.59       0.9993          1.0001            1.0000         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)          1_093.38       264.26     1_357.64       0.9972          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)          1_093.38       388.47     1_481.85       0.9998          1.0000            1.0000         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)          1_093.38       268.58     1_361.96       0.9970          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)          1_093.38       402.58     1_495.96       0.9997          1.0000            1.0000         5.02
IVF-Binary-512-nl158-random (self)                     1_093.38       652.76     1_746.14       0.9974          1.0003            1.0000         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)           1_202.88       141.03     1_343.92       0.5912          1.6104            1.4673         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)           1_202.88       146.62     1_349.50       0.5903          1.6195            1.4721         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)           1_202.88       156.37     1_359.25       0.5890          1.6407            1.4796         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)          1_202.88       254.97     1_457.85       0.9972          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)          1_202.88       376.01     1_578.90       0.9996          1.0001            1.0000         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)          1_202.88       253.05     1_455.93       0.9972          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)          1_202.88       386.43     1_589.31       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)          1_202.88       264.23     1_467.12       0.9969          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)          1_202.88       404.62     1_607.51       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-random (self)                     1_202.88       639.04     1_841.92       0.9974          1.0003            1.0000         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)           1_491.06       151.00     1_642.06       0.5911          1.6073            1.4674         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)           1_491.06       146.72     1_637.78       0.5906          1.6115            1.4698         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)           1_491.06       156.22     1_647.28       0.5897          1.6270            1.4771         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)          1_491.06       261.27     1_752.34       0.9974          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)          1_491.06       401.62     1_892.68       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)          1_491.06       259.93     1_750.99       0.9973          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)          1_491.06       399.06     1_890.12       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)          1_491.06       271.84     1_762.90       0.9971          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)          1_491.06       416.87     1_907.93       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-random (self)                     1_491.06       643.49     2_134.55       0.9976          1.0003            1.0000         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)               1_298.92       130.14     1_429.06       0.6432          1.3824            1.2890         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)              1_298.92       143.72     1_442.65       0.6414          1.3937            1.2943         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)              1_298.92       154.82     1_453.74       0.6407          1.4029            1.2975         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)              1_298.92       248.67     1_547.60       0.9980          1.0002            1.0000         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)              1_298.92       364.86     1_663.78       0.9994          1.0001            1.0000         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)             1_298.92       256.24     1_555.16       0.9982          1.0002            1.0000         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)             1_298.92       387.41     1_686.33       0.9999          1.0000            1.0000         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)             1_298.92       273.76     1_572.68       0.9981          1.0003            1.0000         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)             1_298.92       404.32     1_703.24       0.9998          1.0000            1.0000         5.02
IVF-Binary-512-nl158-pca (self)                        1_298.92       653.35     1_952.27       0.9983          1.0002            1.0000         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_417.95       136.34     1_554.30       0.6420          1.3886            1.2933         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_417.95       140.30     1_558.26       0.6414          1.3936            1.2957         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_417.95       152.12     1_570.08       0.6407          1.4030            1.2985         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_417.95       251.05     1_669.00       0.9982          1.0002            1.0000         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_417.95       376.95     1_794.90       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_417.95       256.80     1_674.76       0.9982          1.0002            1.0000         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_417.95       412.25     1_830.20       0.9998          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_417.95       266.42     1_684.37       0.9981          1.0003            1.0000         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_417.95       402.37     1_820.33       0.9998          1.0000            1.0000         5.21
IVF-Binary-512-nl223-pca (self)                        1_417.95       630.72     2_048.67       0.9983          1.0002            1.0000         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)              1_685.14       148.93     1_834.07       0.6422          1.3878            1.2924         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)              1_685.14       147.08     1_832.22       0.6419          1.3897            1.2933         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)              1_685.14       155.44     1_840.58       0.6413          1.3958            1.2957         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)             1_685.14       266.80     1_951.94       0.9984          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)             1_685.14       389.25     2_074.40       0.9999          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)             1_685.14       259.41     1_944.55       0.9983          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)             1_685.14       395.48     2_080.62       0.9999          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)             1_685.14       275.84     1_960.98       0.9981          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)             1_685.14       415.94     2_101.08       0.9999          1.0000            1.0000         5.48
IVF-Binary-512-nl316-pca (self)                        1_685.14       648.22     2_333.36       0.9984          1.0002            1.0000         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)           1_296.48       208.37     1_504.85       0.6492          1.4403            1.3298         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)          1_296.48       225.33     1_521.81       0.6468          1.4562            1.3410         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)          1_296.48       243.44     1_539.92       0.6457          1.4688            1.3455         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)          1_296.48       326.56     1_623.04       0.9990          1.0002            1.0000         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)          1_296.48       465.49     1_761.97       0.9994          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)         1_296.48       351.28     1_647.75       0.9995          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)         1_296.48       491.15     1_787.63       0.9999          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)         1_296.48       383.69     1_680.17       0.9994          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)         1_296.48       514.43     1_810.91       0.9999          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-random (self)                    1_296.48       983.37     2_279.85       0.9995          1.0001            1.0000         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_412.16       218.72     1_630.88       0.6479          1.4467            1.3358         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_412.16       219.63     1_631.79       0.6472          1.4539            1.3395         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_412.16       240.59     1_652.75       0.6461          1.4684            1.3443         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_412.16       348.32     1_760.48       0.9993          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_412.16       479.04     1_891.20       0.9997          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_412.16       356.07     1_768.23       0.9994          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_412.16       493.59     1_905.75       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_412.16       378.43     1_790.59       0.9994          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_412.16       519.15     1_931.31       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-random (self)                    1_412.16       946.79     2_358.95       0.9995          1.0001            1.0000         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)          1_703.41       231.66     1_935.07       0.6478          1.4461            1.3341        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)          1_703.41       228.87     1_932.28       0.6475          1.4494            1.3360        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)          1_703.41       250.01     1_953.43       0.6465          1.4599            1.3405        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)         1_703.41       365.35     2_068.76       0.9995          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)         1_703.41       491.88     2_195.29       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)         1_703.41       374.62     2_078.04       0.9995          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)         1_703.41       499.14     2_202.55       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)         1_703.41       383.29     2_086.70       0.9994          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)         1_703.41       522.98     2_226.39       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-random (self)                    1_703.41       971.72     2_675.13       0.9995          1.0001            1.0000        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)              1_513.15       209.09     1_722.25       0.6828          1.3187            1.2384         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)             1_513.15       225.46     1_738.61       0.6812          1.3271            1.2437         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)             1_513.15       246.43     1_759.58       0.6805          1.3344            1.2453         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)             1_513.15       341.65     1_854.81       0.9992          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)             1_513.15       463.59     1_976.75       0.9994          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)            1_513.15       353.23     1_866.38       0.9997          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)            1_513.15       490.60     2_003.76       1.0000          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)            1_513.15       377.61     1_890.77       0.9996          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)            1_513.15       514.08     2_027.24       1.0000          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-pca (self)                       1_513.15       998.55     2_511.70       0.9997          1.0000            1.0000         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_611.12       217.20     1_828.32       0.6817          1.3239            1.2402         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_611.12       226.69     1_837.82       0.6813          1.3271            1.2427         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_611.12       238.04     1_849.16       0.6805          1.3344            1.2451         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_611.12       345.02     1_956.14       0.9995          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_611.12       484.05     2_095.17       0.9998          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_611.12       354.37     1_965.50       0.9996          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_611.12       491.74     2_102.86       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_611.12       379.42     1_990.54       0.9996          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_611.12       514.60     2_125.72       1.0000          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-pca (self)                       1_611.12       966.49     2_577.61       0.9997          1.0000            1.0000         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_918.57       226.69     2_145.26       0.6817          1.3241            1.2409        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_918.57       232.71     2_151.28       0.6815          1.3255            1.2417        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_918.57       244.39     2_162.95       0.6809          1.3307            1.2441        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_918.57       368.56     2_287.13       0.9996          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_918.57       498.88     2_417.44       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_918.57       365.76     2_284.33       0.9996          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_918.57       499.95     2_418.52       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_918.57       382.98     2_301.55       0.9996          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_918.57       526.83     2_445.39       1.0000          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-pca (self)                       1_918.57       967.08     2_885.65       0.9997          1.0000            1.0000        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)                920.09       403.98     1_324.07       0.0572          8.2471            7.4925         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)               920.09       435.60     1_355.69       0.0520          9.4902            8.2157         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)               920.09       444.95     1_365.03       0.0494         10.3593            8.7490         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)               920.09       503.79     1_423.88       0.3101          1.8950            1.7877         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)               920.09       919.99     1_840.08       0.4773          1.4831            1.3817         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)              920.09       525.50     1_445.59       0.2784          2.0012            1.8790         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)              920.09       952.92     1_873.00       0.4277          1.5656            1.4630         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)              920.09       537.29     1_457.38       0.2619          2.0687            1.9351         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)              920.09       969.88     1_889.97       0.3996          1.6191            1.5099         5.04
IVF-Binary-768-nl158-sign (self)                         920.09     1_479.83     2_399.91       0.2911          1.9618            1.8389         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)             1_033.34       413.42     1_446.76       0.0575          8.1957            7.3503         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)             1_033.34       423.37     1_456.71       0.0550          8.6555            7.6759         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)             1_033.34       432.64     1_465.97       0.0517          9.6218            8.3391         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)            1_033.34       508.48     1_541.82       0.3112          1.8758            1.7623         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)            1_033.34       926.84     1_960.18       0.4738          1.4729            1.3874         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)            1_033.34       514.16     1_547.50       0.2974          1.9245            1.8040         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)            1_033.34       929.44     1_962.78       0.4500          1.5112            1.4210         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)            1_033.34       536.87     1_570.21       0.2763          2.0107            1.8820         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)            1_033.34       981.02     2_014.35       0.4154          1.5791            1.4831         5.23
IVF-Binary-768-nl223-sign (self)                       1_033.34     1_472.91     2_506.25       0.3097          1.8870            1.7670         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)             1_328.21       422.07     1_750.28       0.0579          7.9049            7.2012         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)             1_328.21       431.79     1_760.00       0.0566          8.1385            7.3655         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)             1_328.21       435.91     1_764.12       0.0534          8.9452            7.8698         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)            1_328.21       518.89     1_847.10       0.3161          1.8521            1.7485         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)            1_328.21       924.79     2_253.00       0.4802          1.4604            1.3754         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)            1_328.21       523.58     1_851.78       0.3080          1.8779            1.7716         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)            1_328.21       933.28     2_261.49       0.4678          1.4796            1.3939         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)            1_328.21       535.28     1_863.49       0.2866          1.9580            1.8379         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)            1_328.21       951.25     2_279.45       0.4331          1.5380            1.4459         5.51
IVF-Binary-768-nl316-sign (self)                       1_328.21     1_480.75     2_808.95       0.3207          1.8447            1.7287         5.51
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

- *n_probe*: `ExhaustiveRaBitQ` is exhaustive only in the sense that there is no
  IVF above it. The query still probes a share of its own `sqrt(n)` centroids,
  25% of them when left at `None`, which is what the grid runs. It is not swept.
- *reranking_factor (rf)*: As for the binary indices. The RaBitQ estimate picks
  the candidates, then the on-disk vectors are loaded and re-scored exactly.
  `10` means `10 * k` vectors get re-scored. The exhaustive grid runs `rf0` (the
  estimate alone, no re-ranking), `5`, `10` and `20`; the IVF grid runs `rf0`,
  `10` and `20`.

**Tunable parameters *(IVF-specific)*:**

- *Number of lists (nl)*: Number of k-means clusters, `sqrt(n)` as a default.
  The grid runs `sqrt(n/2)`, `sqrt(n)` and `sqrt(2n)`.
- *Number of probes (np)*: `sqrt(nlist)`, `sqrt(2 * nlist)` and 5% of `nlist`,
  deduplicated.

Self queries run at `reranking_factor = 10` for the exhaustive index and at the
default `20` for IVF, the latter at `np = sqrt(2 * nlist)`.

#### Correlated data

<details>
<summary><b>Correlated data - 256 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 256D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        34.11       732.47       766.57       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         34.11     2_355.18     2_389.29       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             505.55       184.76       690.31       0.5703          1.0361            1.0358         2.56
ExhaustiveRaBitQ-rf5 (query)                             505.55       231.65       737.20       0.9274          1.0016            1.0005         2.56
ExhaustiveRaBitQ-rf10 (query)                            505.55       275.87       781.42       0.9847          1.0003            1.0000         2.56
ExhaustiveRaBitQ-rf20 (query)                            505.55       353.84       859.39       0.9986          1.0000            1.0000         2.56
ExhaustiveRaBitQ (self)                                  505.55       894.64     1_400.18       0.9849          1.0003            1.0000         2.56
IVF-RaBitQ-nl158-np7-rf0 (query)                         548.68        83.28       631.96       0.5827          1.0331            1.0334         2.67
IVF-RaBitQ-nl158-np12-rf0 (query)                        548.68       117.93       666.61       0.5827          1.0331            1.0334         2.67
IVF-RaBitQ-nl158-np17-rf0 (query)                        548.68       145.88       694.56       0.5827          1.0331            1.0334         2.67
IVF-RaBitQ-nl158-np7-rf10 (query)                        548.68       159.22       707.90       0.9864          1.0002            1.0000         2.67
IVF-RaBitQ-nl158-np7-rf20 (query)                        548.68       232.56       781.24       0.9989          1.0000            1.0000         2.67
IVF-RaBitQ-nl158-np12-rf10 (query)                       548.68       197.42       746.10       0.9864          1.0002            1.0000         2.67
IVF-RaBitQ-nl158-np12-rf20 (query)                       548.68       251.37       800.05       0.9989          1.0000            1.0000         2.67
IVF-RaBitQ-nl158-np17-rf10 (query)                       548.68       217.24       765.92       0.9864          1.0002            1.0000         2.67
IVF-RaBitQ-nl158-np17-rf20 (query)                       548.68       283.63       832.31       0.9989          1.0000            1.0000         2.67
IVF-RaBitQ-nl158 (self)                                  548.68       908.95     1_457.63       0.9989          1.0000            1.0000         2.67
IVF-RaBitQ-nl223-np11-rf0 (query)                        523.52       115.09       638.60       0.5929          1.0314            1.0315         2.82
IVF-RaBitQ-nl223-np14-rf0 (query)                        523.52       130.24       653.76       0.5929          1.0314            1.0315         2.82
IVF-RaBitQ-nl223-np21-rf0 (query)                        523.52       176.13       699.64       0.5929          1.0314            1.0315         2.82
IVF-RaBitQ-nl223-np11-rf10 (query)                       523.52       184.05       707.56       0.9889          1.0002            1.0000         2.82
IVF-RaBitQ-nl223-np11-rf20 (query)                       523.52       240.39       763.90       0.9990          1.0000            1.0000         2.82
IVF-RaBitQ-nl223-np14-rf10 (query)                       523.52       198.01       721.53       0.9889          1.0002            1.0000         2.82
IVF-RaBitQ-nl223-np14-rf20 (query)                       523.52       271.42       794.93       0.9991          1.0000            1.0000         2.82
IVF-RaBitQ-nl223-np21-rf10 (query)                       523.52       244.01       767.53       0.9889          1.0002            1.0000         2.82
IVF-RaBitQ-nl223-np21-rf20 (query)                       523.52       300.74       824.26       0.9991          1.0000            1.0000         2.82
IVF-RaBitQ-nl223 (self)                                  523.52       956.22     1_479.73       0.9991          1.0000            1.0000         2.82
IVF-RaBitQ-nl316-np15-rf0 (query)                        605.15       137.81       742.96       0.6009          1.0300            1.0299         3.06
IVF-RaBitQ-nl316-np17-rf0 (query)                        605.15       151.96       757.11       0.6009          1.0299            1.0299         3.06
IVF-RaBitQ-nl316-np25-rf0 (query)                        605.15       208.80       813.95       0.6009          1.0299            1.0299         3.06
IVF-RaBitQ-nl316-np15-rf10 (query)                       605.15       210.06       815.21       0.9900          1.0002            1.0000         3.06
IVF-RaBitQ-nl316-np15-rf20 (query)                       605.15       266.09       871.23       0.9993          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf10 (query)                       605.15       221.51       826.66       0.9900          1.0002            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf20 (query)                       605.15       280.32       885.46       0.9994          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf10 (query)                       605.15       273.04       878.19       0.9900          1.0002            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf20 (query)                       605.15       330.25       935.39       0.9994          1.0000            1.0000         3.06
IVF-RaBitQ-nl316 (self)                                  605.15     1_056.79     1_661.94       0.9993          1.0000            1.0000         3.06
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 512D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        70.13     1_334.30     1_404.43       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         70.13     4_739.39     4_809.52       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                             869.99       327.22     1_197.21       0.5776          1.0229            1.0227         4.37
ExhaustiveRaBitQ-rf5 (query)                             869.99       387.96     1_257.95       0.9262          1.0011            1.0004         4.37
ExhaustiveRaBitQ-rf10 (query)                            869.99       441.44     1_311.43       0.9837          1.0002            1.0000         4.37
ExhaustiveRaBitQ-rf20 (query)                            869.99       541.70     1_411.69       0.9984          1.0000            1.0000         4.37
ExhaustiveRaBitQ (self)                                  869.99     1_384.82     2_254.81       0.9840          1.0002            1.0000         4.37
IVF-RaBitQ-nl158-np7-rf0 (query)                         982.46       152.47     1_134.93       0.5897          1.0210            1.0216         4.59
IVF-RaBitQ-nl158-np12-rf0 (query)                        982.46       204.01     1_186.48       0.5897          1.0210            1.0216         4.59
IVF-RaBitQ-nl158-np17-rf0 (query)                        982.46       263.76     1_246.23       0.5897          1.0210            1.0216         4.59
IVF-RaBitQ-nl158-np7-rf10 (query)                        982.46       261.55     1_244.02       0.9855          1.0002            1.0000         4.59
IVF-RaBitQ-nl158-np7-rf20 (query)                        982.46       350.24     1_332.70       0.9986          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np12-rf10 (query)                       982.46       304.28     1_286.75       0.9855          1.0002            1.0000         4.59
IVF-RaBitQ-nl158-np12-rf20 (query)                       982.46       404.13     1_386.59       0.9986          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np17-rf10 (query)                       982.46       359.83     1_342.29       0.9855          1.0002            1.0000         4.59
IVF-RaBitQ-nl158-np17-rf20 (query)                       982.46       461.15     1_443.61       0.9986          1.0000            1.0000         4.59
IVF-RaBitQ-nl158 (self)                                  982.46     1_439.29     2_421.76       0.9988          1.0000            1.0000         4.59
IVF-RaBitQ-nl223-np11-rf0 (query)                        925.77       193.54     1_119.31       0.5984          1.0202            1.0205         4.89
IVF-RaBitQ-nl223-np14-rf0 (query)                        925.77       227.43     1_153.20       0.5984          1.0202            1.0205         4.89
IVF-RaBitQ-nl223-np21-rf0 (query)                        925.77       310.30     1_236.07       0.5984          1.0202            1.0205         4.89
IVF-RaBitQ-nl223-np11-rf10 (query)                       925.77       293.50     1_219.27       0.9877          1.0001            1.0000         4.89
IVF-RaBitQ-nl223-np11-rf20 (query)                       925.77       384.90     1_310.67       0.9988          1.0000            1.0000         4.89
IVF-RaBitQ-nl223-np14-rf10 (query)                       925.77       327.28     1_253.05       0.9878          1.0001            1.0000         4.89
IVF-RaBitQ-nl223-np14-rf20 (query)                       925.77       419.84     1_345.61       0.9989          1.0000            1.0000         4.89
IVF-RaBitQ-nl223-np21-rf10 (query)                       925.77       416.60     1_342.37       0.9878          1.0001            1.0000         4.89
IVF-RaBitQ-nl223-np21-rf20 (query)                       925.77       489.75     1_415.52       0.9989          1.0000            1.0000         4.89
IVF-RaBitQ-nl223 (self)                                  925.77     1_567.68     2_493.45       0.9990          1.0000            1.0000         4.89
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_007.20       241.79     1_248.99       0.6051          1.0193            1.0197         5.35
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_007.20       265.13     1_272.33       0.6051          1.0193            1.0197         5.35
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_007.20       363.85     1_371.05       0.6051          1.0193            1.0197         5.35
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_007.20       347.36     1_354.56       0.9882          1.0001            1.0000         5.35
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_007.20       429.77     1_436.97       0.9990          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_007.20       360.50     1_367.70       0.9882          1.0001            1.0000         5.35
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_007.20       459.09     1_466.29       0.9990          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_007.20       455.12     1_462.32       0.9882          1.0001            1.0000         5.35
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_007.20       537.12     1_544.32       0.9990          1.0000            1.0000         5.35
IVF-RaBitQ-nl316 (self)                                1_007.20     1_719.48     2_726.67       0.9991          1.0000            1.0000         5.35
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 768D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       103.08     1_977.57     2_080.65       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        103.08     6_683.53     6_786.61       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           1_111.96       447.30     1_559.26       0.5778          1.0181            1.0180         6.16
ExhaustiveRaBitQ-rf5 (query)                           1_111.96       518.85     1_630.81       0.9239          1.0009            1.0003         6.16
ExhaustiveRaBitQ-rf10 (query)                          1_111.96       585.70     1_697.66       0.9829          1.0002            1.0000         6.16
ExhaustiveRaBitQ-rf20 (query)                          1_111.96       731.65     1_843.60       0.9983          1.0000            1.0000         6.16
ExhaustiveRaBitQ (self)                                1_111.96     1_901.40     3_013.36       0.9829          1.0002            1.0000         6.16
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_311.68       199.40     1_511.08       0.5908          1.0165            1.0169         6.50
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_311.68       281.83     1_593.51       0.5908          1.0165            1.0169         6.50
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_311.68       361.56     1_673.24       0.5908          1.0165            1.0169         6.50
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_311.68       327.63     1_639.30       0.9851          1.0001            1.0000         6.50
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_311.68       441.19     1_752.87       0.9986          1.0000            1.0000         6.50
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_311.68       398.25     1_709.93       0.9851          1.0001            1.0000         6.50
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_311.68       524.07     1_835.75       0.9986          1.0000            1.0000         6.50
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_311.68       482.39     1_794.07       0.9851          1.0001            1.0000         6.50
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_311.68       608.31     1_919.99       0.9986          1.0000            1.0000         6.50
IVF-RaBitQ-nl158 (self)                                1_311.68     1_909.66     3_221.34       0.9987          1.0000            1.0000         6.50
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_196.44       267.63     1_464.08       0.5898          1.0169            1.0169         6.96
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_196.44       313.43     1_509.88       0.5898          1.0169            1.0169         6.96
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_196.44       427.08     1_623.52       0.5898          1.0169            1.0169         6.96
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_196.44       393.78     1_590.23       0.9848          1.0001            1.0000         6.96
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_196.44       507.91     1_704.35       0.9985          1.0000            1.0000         6.96
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_196.44       434.55     1_631.00       0.9849          1.0001            1.0000         6.96
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_196.44       546.51     1_742.95       0.9985          1.0000            1.0000         6.96
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_196.44       545.43     1_741.87       0.9849          1.0001            1.0000         6.96
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_196.44       655.11     1_851.55       0.9985          1.0000            1.0000         6.96
IVF-RaBitQ-nl223 (self)                                1_196.44     2_119.93     3_316.37       0.9986          1.0000            1.0000         6.96
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_282.96       338.91     1_621.87       0.6025          1.0154            1.0159         7.66
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_282.96       377.37     1_660.33       0.6025          1.0154            1.0159         7.66
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_282.96       503.07     1_786.03       0.6025          1.0154            1.0159         7.66
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_282.96       459.72     1_742.68       0.9873          1.0001            1.0000         7.66
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_282.96       581.69     1_864.64       0.9988          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_282.96       484.38     1_767.34       0.9873          1.0001            1.0000         7.66
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_282.96       601.64     1_884.60       0.9989          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_282.96       616.63     1_899.58       0.9873          1.0001            1.0000         7.66
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_282.96       730.07     2_013.03       0.9989          1.0000            1.0000         7.66
IVF-RaBitQ-nl316 (self)                                1_282.96     2_335.56     3_618.51       0.9989          1.0000            1.0000         7.66
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 256D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        33.79       706.16       739.95       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.79     2_387.31     2_421.11       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             508.30       219.30       727.60       0.7383          1.0233            1.0223         2.57
ExhaustiveRaBitQ-rf5 (query)                             508.30       265.54       773.84       0.9978          1.0000            1.0000         2.57
ExhaustiveRaBitQ-rf10 (query)                            508.30       311.72       820.01       1.0000          1.0000            1.0000         2.57
ExhaustiveRaBitQ-rf20 (query)                            508.30       404.27       912.57       1.0000          1.0000            1.0000         2.57
ExhaustiveRaBitQ (self)                                  508.30     1_012.40     1_520.70       1.0000          1.0000            1.0000         2.57
IVF-RaBitQ-nl158-np7-rf0 (query)                         563.82        83.28       647.10       0.7411          1.0229            1.0219         2.68
IVF-RaBitQ-nl158-np12-rf0 (query)                        563.82       117.50       681.32       0.7411          1.0229            1.0219         2.68
IVF-RaBitQ-nl158-np17-rf0 (query)                        563.82       157.49       721.31       0.7411          1.0229            1.0219         2.68
IVF-RaBitQ-nl158-np7-rf10 (query)                        563.82       161.43       725.25       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np7-rf20 (query)                        563.82       224.51       788.33       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np12-rf10 (query)                       563.82       194.34       758.16       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np12-rf20 (query)                       563.82       259.57       823.39       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rf10 (query)                       563.82       222.86       786.68       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rf20 (query)                       563.82       295.81       859.63       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158 (self)                                  563.82       916.39     1_480.21       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl223-np11-rf0 (query)                        532.84       109.91       642.75       0.7444          1.0221            1.0210         2.84
IVF-RaBitQ-nl223-np14-rf0 (query)                        532.84       135.10       667.95       0.7444          1.0221            1.0210         2.84
IVF-RaBitQ-nl223-np21-rf0 (query)                        532.84       176.26       709.10       0.7444          1.0221            1.0210         2.84
IVF-RaBitQ-nl223-np11-rf10 (query)                       532.84       187.75       720.59       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np11-rf20 (query)                       532.84       252.54       785.38       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np14-rf10 (query)                       532.84       207.35       740.20       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np14-rf20 (query)                       532.84       266.29       799.13       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np21-rf10 (query)                       532.84       249.60       782.44       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np21-rf20 (query)                       532.84       312.01       844.85       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223 (self)                                  532.84     1_017.38     1_550.23       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl316-np15-rf0 (query)                        597.05       139.90       736.95       0.7482          1.0214            1.0202         3.06
IVF-RaBitQ-nl316-np17-rf0 (query)                        597.05       152.29       749.34       0.7482          1.0214            1.0202         3.06
IVF-RaBitQ-nl316-np25-rf0 (query)                        597.05       203.63       800.68       0.7482          1.0214            1.0202         3.06
IVF-RaBitQ-nl316-np15-rf10 (query)                       597.05       209.99       807.05       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np15-rf20 (query)                       597.05       273.32       870.38       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf10 (query)                       597.05       223.32       820.38       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf20 (query)                       597.05       289.32       886.37       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf10 (query)                       597.05       271.56       868.61       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf20 (query)                       597.05       346.30       943.35       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316 (self)                                  597.05     1_095.61     1_692.66       1.0000          1.0000            1.0000         3.06
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 512D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        71.64     1_373.56     1_445.19       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         71.64     4_601.70     4_673.34       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                             874.65       358.68     1_233.33       0.7526          1.0138            1.0132         4.36
ExhaustiveRaBitQ-rf5 (query)                             874.65       425.76     1_300.40       0.9982          1.0000            1.0000         4.36
ExhaustiveRaBitQ-rf10 (query)                            874.65       478.88     1_353.52       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ-rf20 (query)                            874.65       597.60     1_472.25       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ (self)                                  874.65     1_516.52     2_391.16       1.0000          1.0000            1.0000         4.36
IVF-RaBitQ-nl158-np7-rf0 (query)                         980.84       147.63     1_128.47       0.7545          1.0137            1.0130         4.59
IVF-RaBitQ-nl158-np12-rf0 (query)                        980.84       216.26     1_197.10       0.7545          1.0137            1.0130         4.59
IVF-RaBitQ-nl158-np17-rf0 (query)                        980.84       267.01     1_247.85       0.7545          1.0137            1.0130         4.59
IVF-RaBitQ-nl158-np7-rf10 (query)                        980.84       277.53     1_258.37       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np7-rf20 (query)                        980.84       358.40     1_339.24       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np12-rf10 (query)                       980.84       310.27     1_291.11       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np12-rf20 (query)                       980.84       406.33     1_387.17       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np17-rf10 (query)                       980.84       365.91     1_346.74       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np17-rf20 (query)                       980.84       456.24     1_437.08       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158 (self)                                  980.84     1_450.39     2_431.23       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl223-np11-rf0 (query)                        936.46       195.59     1_132.05       0.7568          1.0134            1.0128         4.89
IVF-RaBitQ-nl223-np14-rf0 (query)                        936.46       233.03     1_169.49       0.7570          1.0134            1.0128         4.89
IVF-RaBitQ-nl223-np21-rf0 (query)                        936.46       313.50     1_249.96       0.7570          1.0134            1.0128         4.89
IVF-RaBitQ-nl223-np11-rf10 (query)                       936.46       301.17     1_237.63       0.9994          1.0000            1.0000         4.89
IVF-RaBitQ-nl223-np11-rf20 (query)                       936.46       397.27     1_333.73       0.9995          1.0000            1.0000         4.89
IVF-RaBitQ-nl223-np14-rf10 (query)                       936.46       338.36     1_274.82       1.0000          1.0000            1.0000         4.89
IVF-RaBitQ-nl223-np14-rf20 (query)                       936.46       442.72     1_379.18       1.0000          1.0000            1.0000         4.89
IVF-RaBitQ-nl223-np21-rf10 (query)                       936.46       423.16     1_359.62       1.0000          1.0000            1.0000         4.89
IVF-RaBitQ-nl223-np21-rf20 (query)                       936.46       509.49     1_445.95       1.0000          1.0000            1.0000         4.89
IVF-RaBitQ-nl223 (self)                                  936.46     1_728.52     2_664.98       1.0000          1.0000            1.0000         4.89
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_029.53       242.19     1_271.72       0.7583          1.0132            1.0125         5.35
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_029.53       269.77     1_299.30       0.7583          1.0132            1.0125         5.35
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_029.53       354.90     1_384.43       0.7583          1.0132            1.0125         5.35
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_029.53       345.96     1_375.49       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_029.53       441.02     1_470.55       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_029.53       364.49     1_394.02       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_029.53       473.67     1_503.20       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_029.53       452.31     1_481.84       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_029.53       547.04     1_576.57       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316 (self)                                1_029.53     1_731.22     2_760.75       1.0000          1.0000            1.0000         5.35
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 768D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       103.19     1_936.60     2_039.79       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        103.19     6_575.02     6_678.21       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           1_122.88       503.82     1_626.70       0.7322          1.0116            1.0111         6.17
ExhaustiveRaBitQ-rf5 (query)                           1_122.88       575.69     1_698.57       0.9963          1.0000            1.0000         6.17
ExhaustiveRaBitQ-rf10 (query)                          1_122.88       640.41     1_763.29       0.9999          1.0000            1.0000         6.17
ExhaustiveRaBitQ-rf20 (query)                          1_122.88       768.70     1_891.58       1.0000          1.0000            1.0000         6.17
ExhaustiveRaBitQ (self)                                1_122.88     2_037.64     3_160.53       1.0000          1.0000            1.0000         6.17
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_227.57       201.60     1_429.17       0.7357          1.0112            1.0107         6.51
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_227.57       283.10     1_510.67       0.7357          1.0112            1.0107         6.51
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_227.57       374.99     1_602.56       0.7357          1.0112            1.0107         6.51
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_227.57       326.18     1_553.75       0.9999          1.0000            1.0000         6.51
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_227.57       441.22     1_668.79       1.0000          1.0000            1.0000         6.51
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_227.57       400.85     1_628.42       0.9999          1.0000            1.0000         6.51
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_227.57       518.98     1_746.55       1.0000          1.0000            1.0000         6.51
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_227.57       495.40     1_722.97       0.9999          1.0000            1.0000         6.51
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_227.57       611.74     1_839.31       1.0000          1.0000            1.0000         6.51
IVF-RaBitQ-nl158 (self)                                1_227.57     1_901.62     3_129.18       1.0000          1.0000            1.0000         6.51
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_178.79       271.19     1_449.98       0.7385          1.0110            1.0105         6.97
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_178.79       323.78     1_502.56       0.7385          1.0110            1.0105         6.97
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_178.79       440.93     1_619.72       0.7385          1.0110            1.0105         6.97
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_178.79       395.49     1_574.27       0.9999          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_178.79       510.59     1_689.37       1.0000          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_178.79       459.45     1_638.24       0.9999          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_178.79       560.25     1_739.04       1.0000          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_178.79       563.04     1_741.83       0.9999          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_178.79       674.65     1_853.44       1.0000          1.0000            1.0000         6.97
IVF-RaBitQ-nl223 (self)                                1_178.79     2_165.45     3_344.23       1.0000          1.0000            1.0000         6.97
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_299.71       345.52     1_645.23       0.7398          1.0109            1.0105         7.67
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_299.71       382.13     1_681.83       0.7398          1.0109            1.0105         7.67
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_299.71       532.14     1_831.85       0.7398          1.0109            1.0105         7.67
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_299.71       471.32     1_771.03       0.9999          1.0000            1.0000         7.67
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_299.71       585.61     1_885.32       1.0000          1.0000            1.0000         7.67
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_299.71       534.82     1_834.53       0.9999          1.0000            1.0000         7.67
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_299.71       627.93     1_927.63       1.0000          1.0000            1.0000         7.67
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_299.71       633.64     1_933.35       0.9999          1.0000            1.0000         7.67
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_299.71       748.79     2_048.49       1.0000          1.0000            1.0000         7.67
IVF-RaBitQ-nl316 (self)                                1_299.71     2_414.26     3_713.96       1.0000          1.0000            1.0000         7.67
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Cell embeddings

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 256D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        34.45       723.73       758.17       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         34.45     2_411.40     2_445.85       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             610.38       238.18       848.56       0.8706          1.0280            1.0231         2.57
ExhaustiveRaBitQ-rf5 (query)                             610.38       302.97       913.36       1.0000          1.0000            1.0000         2.57
ExhaustiveRaBitQ-rf10 (query)                            610.38       350.17       960.55       1.0000          1.0000            1.0000         2.57
ExhaustiveRaBitQ-rf20 (query)                            610.38       450.48     1_060.87       1.0000          1.0000            1.0000         2.57
ExhaustiveRaBitQ (self)                                  610.38     1_169.36     1_779.75       1.0000          1.0000            1.0000         2.57
IVF-RaBitQ-nl158-np7-rf0 (query)                         736.48        89.21       825.70       0.8738          1.0271            1.0220         2.68
IVF-RaBitQ-nl158-np12-rf0 (query)                        736.48       129.00       865.48       0.8745          1.0267            1.0217         2.68
IVF-RaBitQ-nl158-np17-rf0 (query)                        736.48       170.58       907.06       0.8745          1.0267            1.0217         2.68
IVF-RaBitQ-nl158-np7-rf10 (query)                        736.48       170.80       907.28       0.9976          1.0005            1.0000         2.68
IVF-RaBitQ-nl158-np7-rf20 (query)                        736.48       243.71       980.20       0.9976          1.0005            1.0000         2.68
IVF-RaBitQ-nl158-np12-rf10 (query)                       736.48       212.18       948.66       0.9999          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np12-rf20 (query)                       736.48       279.44     1_015.92       0.9999          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rf10 (query)                       736.48       255.97       992.45       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rf20 (query)                       736.48       320.00     1_056.48       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158 (self)                                  736.48     1_049.70     1_786.18       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl223-np11-rf0 (query)                        752.16       115.74       867.89       0.8845          1.0225            1.0183         2.83
IVF-RaBitQ-nl223-np14-rf0 (query)                        752.16       139.55       891.71       0.8846          1.0224            1.0182         2.83
IVF-RaBitQ-nl223-np21-rf0 (query)                        752.16       193.30       945.46       0.8846          1.0224            1.0182         2.83
IVF-RaBitQ-nl223-np11-rf10 (query)                       752.16       202.76       954.92       0.9993          1.0001            1.0000         2.83
IVF-RaBitQ-nl223-np11-rf20 (query)                       752.16       257.77     1_009.93       0.9993          1.0001            1.0000         2.83
IVF-RaBitQ-nl223-np14-rf10 (query)                       752.16       211.67       963.82       0.9998          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np14-rf20 (query)                       752.16       279.06     1_031.22       0.9998          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np21-rf10 (query)                       752.16       265.20     1_017.35       1.0000          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np21-rf20 (query)                       752.16       328.78     1_080.94       1.0000          1.0000            1.0000         2.83
IVF-RaBitQ-nl223 (self)                                  752.16     1_095.73     1_847.89       1.0000          1.0000            1.0000         2.83
IVF-RaBitQ-nl316-np15-rf0 (query)                        898.39       139.70     1_038.08       0.8907          1.0195            1.0158         3.06
IVF-RaBitQ-nl316-np17-rf0 (query)                        898.39       161.70     1_060.08       0.8907          1.0195            1.0158         3.06
IVF-RaBitQ-nl316-np25-rf0 (query)                        898.39       206.29     1_104.68       0.8908          1.0195            1.0157         3.06
IVF-RaBitQ-nl316-np15-rf10 (query)                       898.39       213.56     1_111.95       0.9997          1.0001            1.0000         3.06
IVF-RaBitQ-nl316-np15-rf20 (query)                       898.39       278.98     1_177.36       0.9997          1.0001            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf10 (query)                       898.39       223.26     1_121.65       0.9998          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf20 (query)                       898.39       291.41     1_189.80       0.9998          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf10 (query)                       898.39       286.86     1_185.24       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf20 (query)                       898.39       347.70     1_246.08       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316 (self)                                  898.39     1_121.61     2_019.99       1.0000          1.0000            1.0000         3.06
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 512D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        69.90     1_371.55     1_441.45       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.90     4_613.07     4_682.97       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           1_071.46       396.37     1_467.82       0.9095          1.0128            1.0096         4.36
ExhaustiveRaBitQ-rf5 (query)                           1_071.46       476.03     1_547.49       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ-rf10 (query)                          1_071.46       533.87     1_605.32       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ-rf20 (query)                          1_071.46       659.97     1_731.42       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ (self)                                1_071.46     1_706.52     2_777.98       1.0000          1.0000            1.0000         4.36
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_238.21       161.22     1_399.44       0.9150          1.0112            1.0081         4.59
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_238.21       223.27     1_461.48       0.9156          1.0109            1.0080         4.59
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_238.21       289.27     1_527.48       0.9157          1.0109            1.0080         4.59
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_238.21       261.22     1_499.43       0.9986          1.0003            1.0000         4.59
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_238.21       360.91     1_599.13       0.9986          1.0003            1.0000         4.59
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_238.21       323.92     1_562.13       0.9999          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_238.21       429.69     1_667.90       0.9999          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_238.21       392.27     1_630.48       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_238.21       492.63     1_730.84       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158 (self)                                1_238.21     1_589.71     2_827.92       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_147.00       199.47     1_346.47       0.9224          1.0091            1.0066         4.90
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_147.00       245.04     1_392.04       0.9224          1.0091            1.0066         4.90
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_147.00       328.04     1_475.04       0.9224          1.0091            1.0066         4.90
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_147.00       302.65     1_449.65       0.9997          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_147.00       395.67     1_542.67       0.9997          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_147.00       344.26     1_491.26       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_147.00       435.77     1_582.77       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_147.00       422.47     1_569.47       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_147.00       527.66     1_674.66       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223 (self)                                1_147.00     1_660.67     2_807.67       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_272.31       248.22     1_520.52       0.9274          1.0079            1.0055         5.36
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_272.31       271.79     1_544.10       0.9274          1.0079            1.0055         5.36
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_272.31       367.45     1_639.76       0.9274          1.0079            1.0055         5.36
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_272.31       352.92     1_625.23       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_272.31       450.11     1_722.42       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_272.31       365.53     1_637.83       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_272.31       475.51     1_747.81       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_272.31       463.28     1_735.59       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_272.31       560.08     1_832.39       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl316 (self)                                1_272.31     1_801.90     3_074.21       1.0000          1.0000            1.0000         5.36
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 768D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       103.43     1_968.25     2_071.68       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        103.43     6_847.37     6_950.79       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           1_431.68       543.83     1_975.51       0.9148          1.0115            1.0083         6.16
ExhaustiveRaBitQ-rf5 (query)                           1_431.68       627.43     2_059.11       1.0000          1.0000            1.0000         6.16
ExhaustiveRaBitQ-rf10 (query)                          1_431.68       695.14     2_126.82       1.0000          1.0000            1.0000         6.16
ExhaustiveRaBitQ-rf20 (query)                          1_431.68       837.48     2_269.16       1.0000          1.0000            1.0000         6.16
ExhaustiveRaBitQ (self)                                1_431.68     2_252.15     3_683.83       1.0000          1.0000            1.0000         6.16
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_690.12       206.36     1_896.48       0.9172          1.0107            1.0077         6.51
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_690.12       310.06     2_000.18       0.9174          1.0107            1.0076         6.51
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_690.12       402.24     2_092.36       0.9174          1.0107            1.0076         6.51
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_690.12       333.80     2_023.92       0.9995          1.0001            1.0000         6.51
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_690.12       450.30     2_140.42       0.9995          1.0001            1.0000         6.51
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_690.12       428.87     2_118.99       1.0000          1.0000            1.0000         6.51
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_690.12       555.14     2_245.25       1.0000          1.0000            1.0000         6.51
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_690.12       534.08     2_224.20       1.0000          1.0000            1.0000         6.51
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_690.12       637.97     2_328.09       1.0000          1.0000            1.0000         6.51
IVF-RaBitQ-nl158 (self)                                1_690.12     2_052.50     3_742.61       1.0000          1.0000            1.0000         6.51
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_676.19       275.68     1_951.87       0.9221          1.0094            1.0067         6.97
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_676.19       326.25     2_002.43       0.9221          1.0094            1.0067         6.97
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_676.19       454.83     2_131.02       0.9221          1.0094            1.0067         6.97
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_676.19       394.05     2_070.24       0.9999          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_676.19       525.88     2_202.06       0.9999          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_676.19       452.53     2_128.71       1.0000          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_676.19       567.83     2_244.02       1.0000          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_676.19       570.73     2_246.91       1.0000          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_676.19       688.51     2_364.69       1.0000          1.0000            1.0000         6.97
IVF-RaBitQ-nl223 (self)                                1_676.19     2_226.06     3_902.25       1.0000          1.0000            1.0000         6.97
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_874.24       341.81     2_216.05       0.9267          1.0082            1.0057         7.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_874.24       376.20     2_250.44       0.9267          1.0082            1.0057         7.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_874.24       512.25     2_386.49       0.9267          1.0082            1.0057         7.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_874.24       461.44     2_335.68       1.0000          1.0000            1.0000         7.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_874.24       580.51     2_454.75       1.0000          1.0000            1.0000         7.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_874.24       492.39     2_366.64       1.0000          1.0000            1.0000         7.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_874.24       613.81     2_488.05       1.0000          1.0000            1.0000         7.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_874.24       643.71     2_517.95       1.0000          1.0000            1.0000         7.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_874.24       759.76     2_634.00       1.0000          1.0000            1.0000         7.63
IVF-RaBitQ-nl316 (self)                                1_874.24     2_431.13     4_305.37       1.0000          1.0000            1.0000         7.63
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### <u>RaBitQ graphs (QG and HNSW-RaBitQ)</u>

Two graph indices built on RaBitQ codes, pulling in opposite directions. QG
keeps the float vectors and duplicates every code once per in-edge to make one
hop a single SIMD sweep: fast, but bigger than the data. HNSW-RaBitQ throws the
vectors away and keeps one multi-bit code per vertex: small, and the `ex_bits`
width is what buys the accuracy back. The column to read across the two is
`index_size_mb` at matched recall.

Both run from one example:

```bash
cargo run --example gridsearch_rabitq_graphs --release --features binary,quantised -- --data embedding --n-samples 50000 --dim 512
```

Euclidean and cosine only for both, no Manhattan.

#### Quantised graph (QG)

`QgIndex` is a SymphonyQG-style index (Gou et al., SIGMOD 2025): a Vamana graph
where every vertex additionally stores its own neighbours' one-bit RaBitQ codes,
quantised against that vertex and pre-transposed into the fast-scan block
layout. One hop is therefore a contiguous read plus one byte-shuffle sweep that
estimates all 32 neighbour distances at once, instead of one random memory
access and one distance kernel per neighbour. Exact distances are computed only
for the vertices the walk actually pops, which the estimator needs as its anchor
anyway, so there is no separate re-ranking stage and no `VecStore`.

**This one is not about memory.** Each vector's code is duplicated once per
in-edge, and the raw vectors have to stay resident for the exact distances, so
the index lands at roughly two to three times the size of the data. The point is
query speed at a given recall. If memory is the binding constraint, look at
HNSW-RaBitQ below or at `IVF-RaBitQ` above.

The graph is not from the paper. SymphonyQG builds its own topology with
random init, repeated search-prune-reverse rounds and a cosine-threshold refill
to force exact-degree regularity; `VamanaIndex` already yields a fixed-degree
graph, so it builds the topology here and what is kept from the paper is the
storage layout and the estimator.

**Tunable parameters *(QG)*:**

- *degree (d)*: Neighbour slots per vertex, a non-zero multiple of 32. Default
  `32`, which is exactly one fast-scan sweep per hop. Doubling it doubles both
  the code footprint and the per-hop work, so it wants a reason; the grid runs
  `32` and `64` to show whether the extra edges pay.
- *l_build (l)*: Beam width during construction, second Vamana pass. This is
  where the build time goes: the encoding is a flat few hundred milliseconds and
  everything else is Vamana. The first pass runs at the crate default, which is
  a small constant, because a wide first pass is both slower and worse. The grid
  runs `64` and `128`.
- *ef_search (ef)*: Beam width at query time, the usual recall/latency dial. The
  grid runs `k`, `2k`, `4k` and `8k`.

#### HNSW-RaBitQ

`HnswRaBitQIndex` links an ordinary HNSW graph on exact distances and then drops
the float vectors. What is left is the topology plus one RaBitQ+ code per
vertex, so the search answers from the codes alone and the footprint falls to a
fraction of the raw data. No `VecStore`, no re-ranking stage: if you need exact
distances back, re-rank against your own vectors.

Quantisation happens after the graph is built, so `ex_bits` moves accuracy and
size together without touching the topology. `ex_bits = 0` is plain one-bit
RaBitQ; the grid starts at `1`, so the codes are 2, 4, 6 and 9 bits wide per
coordinate. That is the dial worth sweeping first, and the tables are where you
read off what each step costs.

**Tunable parameters *(HNSW-RaBitQ)*:**

- *m*: Base connectivity. Layer 0 gets `2 * m` slots. The grid runs `16` and
  `32`.
- *ef_construction*: Beam width during construction, fixed at `200` in the grid.
- *ex_bits (ex)*: Magnitude bits per coordinate on top of the sign bit, so the
  code is `ex_bits + 1` bits wide. `0` is plain one-bit RaBitQ; the grid runs
  `1`, `3`, `5` and `8`.
- *ef_search (ef)*: Beam width at query time. Same `k`, `2k`, `4k`, `8k` grid as
  QG.

Self queries run at `ef_search = 4k` for both.

#### Correlated data

<details>
<summary><b>Correlated data - 256 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 256D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        33.53       695.94       729.46       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.53     2_266.46     2_299.99       1.0000          1.0000            1.0000        48.83
QG-d32-l64-ef15 (query)                                2_504.32       305.18     2_809.50       0.9910          1.0021            1.0000       116.35
QG-d32-l64-ef30 (query)                                2_504.32       427.45     2_931.77       0.9957          1.0017            1.0000       116.35
QG-d32-l64-ef60 (query)                                2_504.32       607.95     3_112.26       0.9973          1.0016            1.0000       116.35
QG-d32-l64-ef120 (query)                               2_504.32       892.29     3_396.61       0.9979          1.0015            1.0000       116.35
QG-d32-l64 (self)                                      2_504.32     2_006.91     4_511.23       0.9973          1.0016            1.0000       116.35
QG-d32-l128-ef15 (query)                               2_862.36       295.78     3_158.14       0.9914          1.0023            1.0000       116.35
QG-d32-l128-ef30 (query)                               2_862.36       428.43     3_290.79       0.9961          1.0017            1.0000       116.35
QG-d32-l128-ef60 (query)                               2_862.36       609.41     3_471.78       0.9977          1.0014            1.0000       116.35
QG-d32-l128-ef120 (query)                              2_862.36       892.60     3_754.96       0.9983          1.0013            1.0000       116.35
QG-d32-l128 (self)                                     2_862.36     1_994.02     4_856.38       0.9978          1.0014            1.0000       116.35
QG-d64-l64-ef15 (query)                                8_696.22       657.53     9_353.75       0.9990          1.0003            1.0000       183.49
QG-d64-l64-ef30 (query)                                8_696.22       886.22     9_582.44       0.9996          1.0003            1.0000       183.49
QG-d64-l64-ef60 (query)                                8_696.22     1_239.46     9_935.67       0.9997          1.0002            1.0000       183.49
QG-d64-l64-ef120 (query)                               8_696.22     1_775.72    10_471.93       0.9997          1.0002            1.0000       183.49
QG-d64-l64 (self)                                      8_696.22     4_069.89    12_766.10       0.9997          1.0002            1.0000       183.49
QG-d64-l128-ef15 (query)                               9_541.26       658.77    10_200.03       0.9992          1.0002            1.0000       183.49
QG-d64-l128-ef30 (query)                               9_541.26       898.20    10_439.46       0.9997          1.0002            1.0000       183.49
QG-d64-l128-ef60 (query)                               9_541.26     1_239.07    10_780.33       0.9998          1.0002            1.0000       183.49
QG-d64-l128-ef120 (query)                              9_541.26     1_779.09    11_320.35       0.9998          1.0001            1.0000       183.49
QG-d64-l128 (self)                                     9_541.26     4_134.59    13_675.85       0.9998          1.0002            1.0000       183.49
HnswRaBitQ-m16-ex1-ef15 (query)                        1_296.47       616.61     1_913.07       0.7355        124.8459            1.0085        10.85
HnswRaBitQ-m16-ex1-ef30 (query)                        1_296.47       895.98     2_192.44       0.7755          1.0122            1.0081        10.85
HnswRaBitQ-m16-ex1-ef60 (query)                        1_296.47     1_276.59     2_573.05       0.7770          1.0098            1.0080        10.85
HnswRaBitQ-m16-ex1-ef120 (query)                       1_296.47     1_752.93     3_049.39       0.7774          1.0090            1.0080        10.85
HnswRaBitQ-m16-ex1 (self)                              1_296.47     6_954.82     8_251.28       0.6964          1.0182            1.0162        10.85
HnswRaBitQ-m16-ex3-ef15 (query)                        1_434.38       613.01     2_047.39       0.9008          1.1333            1.0010        13.90
HnswRaBitQ-m16-ex3-ef30 (query)                        1_434.38       887.88     2_322.26       0.9206          1.0042            1.0006        13.90
HnswRaBitQ-m16-ex3-ef60 (query)                        1_434.38     1_263.80     2_698.18       0.9273          1.0014            1.0005        13.90
HnswRaBitQ-m16-ex3-ef120 (query)                       1_434.38     1_764.63     3_199.01       0.9287          1.0009            1.0005        13.90
HnswRaBitQ-m16-ex3 (self)                              1_434.38     6_856.24     8_290.62       0.9031          1.0021            1.0012        13.90
HnswRaBitQ-m16-ex5-ef15 (query)                        1_455.61       614.49     2_070.10       0.9395          1.0091            1.0001        16.95
HnswRaBitQ-m16-ex5-ef30 (query)                        1_455.61       897.73     2_353.34       0.9660          1.0025            1.0000        16.95
HnswRaBitQ-m16-ex5-ef60 (query)                        1_455.61     1_288.83     2_744.43       0.9759          1.0006            1.0000        16.95
HnswRaBitQ-m16-ex5-ef120 (query)                       1_455.61     1_762.34     3_217.95       0.9780          1.0002            1.0000        16.95
HnswRaBitQ-m16-ex5 (self)                              1_455.61     6_902.44     8_358.04       0.9695          1.0006            1.0000        16.95
HnswRaBitQ-m16-ex8-ef15 (query)                        2_174.54       623.42     2_797.97       0.9498          1.0516            1.0000        21.53
HnswRaBitQ-m16-ex8-ef30 (query)                        2_174.54       901.49     3_076.03       0.9809          1.0051            1.0000        21.53
HnswRaBitQ-m16-ex8-ef60 (query)                        2_174.54     1_304.15     3_478.70       0.9934          1.0008            1.0000        21.53
HnswRaBitQ-m16-ex8-ef120 (query)                       2_174.54     1_811.25     3_985.79       0.9963          1.0004            1.0000        21.53
HnswRaBitQ-m16-ex8 (self)                              2_174.54     6_937.42     9_111.96       0.9926          1.0010            1.0000        21.53
HnswRaBitQ-m32-ex1-ef15 (query)                        1_341.63       796.45     2_138.08       0.7740          1.0321            1.0080        16.95
HnswRaBitQ-m32-ex1-ef30 (query)                        1_341.63     1_128.99     2_470.62       0.7771          1.0108            1.0080        16.95
HnswRaBitQ-m32-ex1-ef60 (query)                        1_341.63     1_539.19     2_880.82       0.7775          1.0087            1.0080        16.95
HnswRaBitQ-m32-ex1-ef120 (query)                       1_341.63     1_992.57     3_334.20       0.7779          1.0084            1.0080        16.95
HnswRaBitQ-m32-ex1 (self)                              1_341.63     8_344.54     9_686.17       0.6962          1.0174            1.0162        16.95
HnswRaBitQ-m32-ex3-ef15 (query)                        1_396.70       807.67     2_204.37       0.9167          1.0042            1.0007        20.00
HnswRaBitQ-m32-ex3-ef30 (query)                        1_396.70     1_130.96     2_527.66       0.9258          1.0024            1.0006        20.00
HnswRaBitQ-m32-ex3-ef60 (query)                        1_396.70     1_546.71     2_943.41       0.9286          1.0010            1.0005        20.00
HnswRaBitQ-m32-ex3-ef120 (query)                       1_396.70     2_023.59     3_420.29       0.9290          1.0009            1.0005        20.00
HnswRaBitQ-m32-ex3 (self)                              1_396.70     8_427.17     9_823.87       0.9040          1.0016            1.0011        20.00
HnswRaBitQ-m32-ex5-ef15 (query)                        1_508.30       822.11     2_330.40       0.9603          1.0031            1.0000        23.06
HnswRaBitQ-m32-ex5-ef30 (query)                        1_508.30     1_157.51     2_665.81       0.9735          1.0011            1.0000        23.06
HnswRaBitQ-m32-ex5-ef60 (query)                        1_508.30     1_577.78     3_086.07       0.9774          1.0003            1.0000        23.06
HnswRaBitQ-m32-ex5-ef120 (query)                       1_508.30     2_085.08     3_593.38       0.9782          1.0002            1.0000        23.06
HnswRaBitQ-m32-ex5 (self)                              1_508.30     8_412.42     9_920.72       0.9709          1.0004            1.0000        23.06
HnswRaBitQ-m32-ex8-ef15 (query)                        2_231.73       825.05     3_056.78       0.9753          1.0030            1.0000        27.63
HnswRaBitQ-m32-ex8-ef30 (query)                        2_231.73     1_188.12     3_419.84       0.9915          1.0013            1.0000        27.63
HnswRaBitQ-m32-ex8-ef60 (query)                        2_231.73     1_618.96     3_850.69       0.9960          1.0004            1.0000        27.63
HnswRaBitQ-m32-ex8-ef120 (query)                       2_231.73     2_122.79     4_354.52       0.9971          1.0001            1.0000        27.63
HnswRaBitQ-m32-ex8 (self)                              2_231.73     9_127.21    11_358.93       0.9951          1.0002            1.0000        27.63
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 512D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        71.88     1_333.59     1_405.47       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         71.88     4_525.30     4_597.18       1.0000          1.0000            1.0000        97.66
QG-d32-l64-ef15 (query)                                4_940.69       515.79     5_456.48       0.9866          1.0050            1.0000       214.01
QG-d32-l64-ef30 (query)                                4_940.69       683.61     5_624.29       0.9920          1.0048            1.0000       214.01
QG-d32-l64-ef60 (query)                                4_940.69       920.12     5_860.81       0.9943          1.0046            1.0000       214.01
QG-d32-l64-ef120 (query)                               4_940.69     1_261.73     6_202.41       0.9952          1.0046            1.0000       214.01
QG-d32-l64 (self)                                      4_940.69     3_019.06     7_959.75       0.9941          1.0045            1.0000       214.01
QG-d32-l128-ef15 (query)                               5_628.06       505.42     6_133.48       0.9879          1.0035            1.0000       214.01
QG-d32-l128-ef30 (query)                               5_628.06       682.38     6_310.44       0.9931          1.0032            1.0000       214.01
QG-d32-l128-ef60 (query)                               5_628.06       929.98     6_558.04       0.9952          1.0031            1.0000       214.01
QG-d32-l128-ef120 (query)                              5_628.06     1_264.32     6_892.38       0.9962          1.0030            1.0000       214.01
QG-d32-l128 (self)                                     5_628.06     2_999.82     8_627.89       0.9950          1.0033            1.0000       214.01
QG-d64-l64-ef15 (query)                               18_261.31     1_032.93    19_294.24       0.9982          1.0009            1.0000       329.97
QG-d64-l64-ef30 (query)                               18_261.31     1_320.93    19_582.24       0.9988          1.0009            1.0000       329.97
QG-d64-l64-ef60 (query)                               18_261.31     1_743.74    20_005.05       0.9990          1.0009            1.0000       329.97
QG-d64-l64-ef120 (query)                              18_261.31     2_352.02    20_613.33       0.9991          1.0008            1.0000       329.97
QG-d64-l64 (self)                                     18_261.31     5_708.90    23_970.22       0.9990          1.0009            1.0000       329.97
QG-d64-l128-ef15 (query)                              18_959.69     1_021.94    19_981.63       0.9981          1.0010            1.0000       329.97
QG-d64-l128-ef30 (query)                              18_959.69     1_323.46    20_283.15       0.9987          1.0010            1.0000       329.97
QG-d64-l128-ef60 (query)                              18_959.69     1_734.58    20_694.27       0.9989          1.0009            1.0000       329.97
QG-d64-l128-ef120 (query)                             18_959.69     2_322.80    21_282.49       0.9992          1.0008            1.0000       329.97
QG-d64-l128 (self)                                    18_959.69     5_715.96    24_675.65       0.9988          1.0010            1.0000       329.97
HnswRaBitQ-m16-ex1-ef15 (query)                        2_344.14     1_331.14     3_675.28       0.7594          3.7486            1.0058        14.15
HnswRaBitQ-m16-ex1-ef30 (query)                        2_344.14     1_942.05     4_286.20       0.7732          1.0082            1.0055        14.15
HnswRaBitQ-m16-ex1-ef60 (query)                        2_344.14     2_720.87     5_065.02       0.7761          1.0062            1.0054        14.15
HnswRaBitQ-m16-ex1-ef120 (query)                       2_344.14     4_004.05     6_348.19       0.7767          1.0058            1.0054        14.15
HnswRaBitQ-m16-ex1 (self)                              2_344.14    14_565.16    16_909.30       0.6932          1.0126            1.0110        14.15
HnswRaBitQ-m16-ex3-ef15 (query)                        2_502.83     1_355.00     3_857.83       0.8910          1.0429            1.0008        20.25
HnswRaBitQ-m16-ex3-ef30 (query)                        2_502.83     1_923.90     4_426.73       0.9152          1.0066            1.0005        20.25
HnswRaBitQ-m16-ex3-ef60 (query)                        2_502.83     2_747.84     5_250.67       0.9248          1.0015            1.0004        20.25
HnswRaBitQ-m16-ex3-ef120 (query)                       2_502.83     3_769.30     6_272.13       0.9276          1.0007            1.0004        20.25
HnswRaBitQ-m16-ex3 (self)                              2_502.83    14_420.09    16_922.92       0.8997          1.0023            1.0008        20.25
HnswRaBitQ-m16-ex5-ef15 (query)                        2_700.26     1_325.65     4_025.91       0.9255          1.0186            1.0001        26.36
HnswRaBitQ-m16-ex5-ef30 (query)                        2_700.26     1_914.13     4_614.39       0.9587          1.0061            1.0000        26.36
HnswRaBitQ-m16-ex5-ef60 (query)                        2_700.26     2_756.88     5_457.14       0.9723          1.0017            1.0000        26.36
HnswRaBitQ-m16-ex5-ef120 (query)                       2_700.26     3_720.71     6_420.96       0.9764          1.0004            1.0000        26.36
HnswRaBitQ-m16-ex5 (self)                              2_700.26    14_501.26    17_201.51       0.9654          1.0019            1.0000        26.36
HnswRaBitQ-m16-ex8-ef15 (query)                        4_336.15     1_344.55     5_680.70       0.9351          1.0219            1.0000        35.51
HnswRaBitQ-m16-ex8-ef30 (query)                        4_336.15     1_944.80     6_280.95       0.9738          1.0039            1.0000        35.51
HnswRaBitQ-m16-ex8-ef60 (query)                        4_336.15     2_815.22     7_151.37       0.9908          1.0006            1.0000        35.51
HnswRaBitQ-m16-ex8-ef120 (query)                       4_336.15     3_760.69     8_096.84       0.9954          1.0003            1.0000        35.51
HnswRaBitQ-m16-ex8 (self)                              4_336.15    14_820.72    19_156.87       0.9899          1.0007            1.0000        35.51
HnswRaBitQ-m32-ex1-ef15 (query)                        2_415.71     1_802.80     4_218.51       0.7710          1.0692            1.0055        20.25
HnswRaBitQ-m32-ex1-ef30 (query)                        2_415.71     2_495.75     4_911.46       0.7753          1.0292            1.0054        20.25
HnswRaBitQ-m32-ex1-ef60 (query)                        2_415.71     3_368.54     5_784.25       0.7766          1.0059            1.0054        20.25
HnswRaBitQ-m32-ex1-ef120 (query)                       2_415.71     4_291.50     6_707.20       0.7767          1.0057            1.0054        20.25
HnswRaBitQ-m32-ex1 (self)                              2_415.71    18_042.81    20_458.52       0.6928          1.0124            1.0110        20.25
HnswRaBitQ-m32-ex3-ef15 (query)                        2_603.07     1_806.47     4_409.55       0.9121          1.0368            1.0005        26.36
HnswRaBitQ-m32-ex3-ef30 (query)                        2_603.07     2_525.69     5_128.76       0.9236          1.0019            1.0004        26.36
HnswRaBitQ-m32-ex3-ef60 (query)                        2_603.07     3_399.53     6_002.60       0.9272          1.0012            1.0004        26.36
HnswRaBitQ-m32-ex3-ef120 (query)                       2_603.07     4_379.58     6_982.66       0.9281          1.0006            1.0004        26.36
HnswRaBitQ-m32-ex3 (self)                              2_603.07    18_143.60    20_746.67       0.9016          1.0016            1.0008        26.36
HnswRaBitQ-m32-ex5-ef15 (query)                        2_810.45     1_821.66     4_632.11       0.9528          1.0770            1.0000        32.46
HnswRaBitQ-m32-ex5-ef30 (query)                        2_810.45     2_540.96     5_351.41       0.9705          1.0042            1.0000        32.46
HnswRaBitQ-m32-ex5-ef60 (query)                        2_810.45     3_423.15     6_233.60       0.9758          1.0004            1.0000        32.46
HnswRaBitQ-m32-ex5-ef120 (query)                       2_810.45     4_364.57     7_175.02       0.9771          1.0001            1.0000        32.46
HnswRaBitQ-m32-ex5 (self)                              2_810.45    18_295.40    21_105.86       0.9687          1.0004            1.0000        32.46
HnswRaBitQ-m32-ex8-ef15 (query)                        4_307.79     1_819.17     6_126.96       0.9678          1.0091            1.0000        41.62
HnswRaBitQ-m32-ex8-ef30 (query)                        4_307.79     2_544.57     6_852.37       0.9878          1.0030            1.0000        41.62
HnswRaBitQ-m32-ex8-ef60 (query)                        4_307.79     3_415.77     7_723.56       0.9947          1.0011            1.0000        41.62
HnswRaBitQ-m32-ex8-ef120 (query)                       4_307.79     4_702.76     9_010.55       0.9964          1.0001            1.0000        41.62
HnswRaBitQ-m32-ex8 (self)                              4_307.79    18_252.48    22_560.28       0.9938          1.0005            1.0000        41.62
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 768D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       107.44     1_985.99     2_093.43       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        107.44     6_658.72     6_766.16       1.0000          1.0000            1.0000       146.48
QG-d32-l64-ef15 (query)                                7_306.60       652.86     7_959.46       0.9853          1.0043            1.0000       311.66
QG-d32-l64-ef30 (query)                                7_306.60       860.00     8_166.60       0.9909          1.0041            1.0000       311.66
QG-d32-l64-ef60 (query)                                7_306.60     1_141.17     8_447.77       0.9935          1.0040            1.0000       311.66
QG-d32-l64-ef120 (query)                               7_306.60     1_518.95     8_825.55       0.9947          1.0039            1.0000       311.66
QG-d32-l64 (self)                                      7_306.60     3_733.03    11_039.63       0.9936          1.0039            1.0000       311.66
QG-d32-l128-ef15 (query)                               8_193.72       668.51     8_862.23       0.9863          1.0041            1.0000       311.66
QG-d32-l128-ef30 (query)                               8_193.72       878.05     9_071.77       0.9918          1.0038            1.0000       311.66
QG-d32-l128-ef60 (query)                               8_193.72     1_165.06     9_358.78       0.9944          1.0037            1.0000       311.66
QG-d32-l128-ef120 (query)                              8_193.72     1_573.10     9_766.82       0.9954          1.0036            1.0000       311.66
QG-d32-l128 (self)                                     8_193.72     3_739.58    11_933.29       0.9945          1.0035            1.0000       311.66
QG-d64-l64-ef15 (query)                               27_618.83     1_385.86    29_004.69       0.9963          1.0040            1.0000       476.46
QG-d64-l64-ef30 (query)                               27_618.83     1_731.35    29_350.19       0.9969          1.0039            1.0000       476.46
QG-d64-l64-ef60 (query)                               27_618.83     2_199.34    29_818.17       0.9972          1.0039            1.0000       476.46
QG-d64-l64-ef120 (query)                              27_618.83     3_059.87    30_678.71       0.9973          1.0039            1.0000       476.46
QG-d64-l64 (self)                                     27_618.83     7_229.31    34_848.15       0.9975          1.0035            1.0000       476.46
QG-d64-l128-ef15 (query)                              27_929.74     1_369.15    29_298.89       0.9978          1.0012            1.0000       476.46
QG-d64-l128-ef30 (query)                              27_929.74     2_026.16    29_955.90       0.9985          1.0012            1.0000       476.46
QG-d64-l128-ef60 (query)                              27_929.74     2_408.86    30_338.60       0.9987          1.0011            1.0000       476.46
QG-d64-l128-ef120 (query)                             27_929.74     3_079.46    31_009.20       0.9988          1.0011            1.0000       476.46
QG-d64-l128 (self)                                    27_929.74     7_500.76    35_430.50       0.9988          1.0011            1.0000       476.46
HnswRaBitQ-m16-ex1-ef15 (query)                        3_328.83     2_072.62     5_401.44       0.7576          1.8525            1.0046        17.45
HnswRaBitQ-m16-ex1-ef30 (query)                        3_328.83     2_951.63     6_280.45       0.7726          1.0469            1.0043        17.45
HnswRaBitQ-m16-ex1-ef60 (query)                        3_328.83     4_228.65     7_557.47       0.7768          1.0059            1.0042        17.45
HnswRaBitQ-m16-ex1-ef120 (query)                       3_328.83     5_688.20     9_017.03       0.7777          1.0048            1.0042        17.45
HnswRaBitQ-m16-ex1 (self)                              3_328.83    24_141.19    27_470.01       0.6919          1.0124            1.0088        17.45
HnswRaBitQ-m16-ex3-ef15 (query)                        3_641.34     2_078.47     5_719.81       0.8888          1.0148            1.0007        26.61
HnswRaBitQ-m16-ex3-ef30 (query)                        3_641.34     3_021.29     6_662.63       0.9134          1.0082            1.0004        26.61
HnswRaBitQ-m16-ex3-ef60 (query)                        3_641.34     4_269.81     7_911.15       0.9243          1.0021            1.0003        26.61
HnswRaBitQ-m16-ex3-ef120 (query)                       3_641.34     5_782.46     9_423.80       0.9276          1.0008            1.0003        26.61
HnswRaBitQ-m16-ex3 (self)                              3_641.34    22_609.16    26_250.50       0.8985          1.0021            1.0007        26.61
HnswRaBitQ-m16-ex5-ef15 (query)                        3_865.62     2_063.17     5_928.80       0.9221          1.0265            1.0001        35.76
HnswRaBitQ-m16-ex5-ef30 (query)                        3_865.62     3_012.81     6_878.44       0.9570          1.0035            1.0000        35.76
HnswRaBitQ-m16-ex5-ef60 (query)                        3_865.62     4_274.28     8_139.90       0.9728          1.0011            1.0000        35.76
HnswRaBitQ-m16-ex5-ef120 (query)                       3_865.62     5_758.74     9_624.37       0.9767          1.0004            1.0000        35.76
HnswRaBitQ-m16-ex5 (self)                              3_865.62    22_606.68    26_472.31       0.9643          1.0015            1.0000        35.76
HnswRaBitQ-m16-ex8-ef15 (query)                        6_154.96     2_044.09     8_199.05       0.9281          1.0746            1.0000        49.50
HnswRaBitQ-m16-ex8-ef30 (query)                        6_154.96     2_965.59     9_120.56       0.9698          1.0114            1.0000        49.50
HnswRaBitQ-m16-ex8-ef60 (query)                        6_154.96     4_258.44    10_413.40       0.9891          1.0029            1.0000        49.50
HnswRaBitQ-m16-ex8-ef120 (query)                       6_154.96     5_732.30    11_887.26       0.9950          1.0004            1.0000        49.50
HnswRaBitQ-m16-ex8 (self)                              6_154.96    24_155.92    30_310.88       0.9884          1.0020            1.0000        49.50
HnswRaBitQ-m32-ex1-ef15 (query)                        3_631.45     2_800.95     6_432.40       0.7717          1.1083            1.0043        23.56
HnswRaBitQ-m32-ex1-ef30 (query)                        3_631.45     3_909.14     7_540.58       0.7759          1.0412            1.0042        23.56
HnswRaBitQ-m32-ex1-ef60 (query)                        3_631.45     5_241.45     8_872.90       0.7775          1.0049            1.0042        23.56
HnswRaBitQ-m32-ex1-ef120 (query)                       3_631.45     6_647.77    10_279.22       0.7781          1.0044            1.0042        23.56
HnswRaBitQ-m32-ex1 (self)                              3_631.45    28_057.09    31_688.54       0.6921          1.0097            1.0088        23.56
HnswRaBitQ-m32-ex3-ef15 (query)                        3_665.80     2_858.03     6_523.82       0.9112          1.0038            1.0004        32.71
HnswRaBitQ-m32-ex3-ef30 (query)                        3_665.80     3_957.02     7_622.81       0.9230          1.0023            1.0003        32.71
HnswRaBitQ-m32-ex3-ef60 (query)                        3_665.80     5_288.90     8_954.69       0.9273          1.0008            1.0003        32.71
HnswRaBitQ-m32-ex3-ef120 (query)                       3_665.80     6_732.83    10_398.63       0.9284          1.0005            1.0003        32.71
HnswRaBitQ-m32-ex3 (self)                              3_665.80    28_478.30    32_144.10       0.9006          1.0012            1.0006        32.71
HnswRaBitQ-m32-ex5-ef15 (query)                        3_957.30     2_833.19     6_790.49       0.9531          1.0057            1.0000        41.87
HnswRaBitQ-m32-ex5-ef30 (query)                        3_957.30     4_023.33     7_980.63       0.9702          1.0006            1.0000        41.87
HnswRaBitQ-m32-ex5-ef60 (query)                        3_957.30     5_306.26     9_263.55       0.9761          1.0003            1.0000        41.87
HnswRaBitQ-m32-ex5-ef120 (query)                       3_957.30     6_777.11    10_734.41       0.9774          1.0001            1.0000        41.87
HnswRaBitQ-m32-ex5 (self)                              3_957.30    28_246.01    32_203.31       0.9679          1.0003            1.0000        41.87
HnswRaBitQ-m32-ex8-ef15 (query)                        6_311.19     2_867.46     9_178.65       0.9667          1.0075            1.0000        55.60
HnswRaBitQ-m32-ex8-ef30 (query)                        6_311.19     4_025.24    10_336.43       0.9869          1.0007            1.0000        55.60
HnswRaBitQ-m32-ex8-ef60 (query)                        6_311.19     5_357.69    11_668.88       0.9944          1.0003            1.0000        55.60
HnswRaBitQ-m32-ex8-ef120 (query)                       6_311.19     6_818.45    13_129.64       0.9960          1.0002            1.0000        55.60
HnswRaBitQ-m32-ex8 (self)                              6_311.19    28_562.56    34_873.75       0.9933          1.0003            1.0000        55.60
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 256D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        34.00       728.40       762.40       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         34.00     2_518.63     2_552.63       1.0000          1.0000            1.0000        48.83
QG-d32-l64-ef15 (query)                                3_023.33        82.54     3_105.88       0.9709          1.0009            1.0000       116.35
QG-d32-l64-ef30 (query)                                3_023.33       141.86     3_165.19       0.9983          1.0001            1.0000       116.35
QG-d32-l64-ef60 (query)                                3_023.33       263.76     3_287.10       0.9999          1.0000            1.0000       116.35
QG-d32-l64-ef120 (query)                               3_023.33       492.95     3_516.28       1.0000          1.0000            1.0000       116.35
QG-d32-l64 (self)                                      3_023.33       786.37     3_809.71       1.0000          1.0000            1.0000       116.35
QG-d32-l128-ef15 (query)                               3_442.63        84.26     3_526.89       0.9710          1.0009            1.0000       116.35
QG-d32-l128-ef30 (query)                               3_442.63       140.00     3_582.63       0.9985          1.0001            1.0000       116.35
QG-d32-l128-ef60 (query)                               3_442.63       257.21     3_699.85       1.0000          1.0000            1.0000       116.35
QG-d32-l128-ef120 (query)                              3_442.63       495.42     3_938.05       1.0000          1.0000            1.0000       116.35
QG-d32-l128 (self)                                     3_442.63       793.68     4_236.31       1.0000          1.0000            1.0000       116.35
QG-d64-l64-ef15 (query)                                5_555.58       117.44     5_673.01       0.9901          1.0002            1.0000       183.49
QG-d64-l64-ef30 (query)                                5_555.58       215.83     5_771.41       0.9999          1.0000            1.0000       183.49
QG-d64-l64-ef60 (query)                                5_555.58       415.48     5_971.05       1.0000          1.0000            1.0000       183.49
QG-d64-l64-ef120 (query)                               5_555.58       802.29     6_357.87       1.0000          1.0000            1.0000       183.49
QG-d64-l64 (self)                                      5_555.58     1_284.95     6_840.53       1.0000          1.0000            1.0000       183.49
QG-d64-l128-ef15 (query)                               6_154.51       116.68     6_271.19       0.9901          1.0002            1.0000       183.49
QG-d64-l128-ef30 (query)                               6_154.51       218.68     6_373.19       0.9999          1.0000            1.0000       183.49
QG-d64-l128-ef60 (query)                               6_154.51       417.27     6_571.78       1.0000          1.0000            1.0000       183.49
QG-d64-l128-ef120 (query)                              6_154.51       802.61     6_957.12       1.0000          1.0000            1.0000       183.49
QG-d64-l128 (self)                                     6_154.51     1_287.11     7_441.62       1.0000          1.0000            1.0000       183.49
HnswRaBitQ-m16-ex1-ef15 (query)                        1_355.34       661.95     2_017.30       0.8535          1.0073            1.0059        10.85
HnswRaBitQ-m16-ex1-ef30 (query)                        1_355.34       993.46     2_348.80       0.8665          1.0057            1.0049        10.85
HnswRaBitQ-m16-ex1-ef60 (query)                        1_355.34     1_475.52     2_830.87       0.8694          1.0053            1.0046        10.85
HnswRaBitQ-m16-ex1-ef120 (query)                       1_355.34     2_080.15     3_435.50       0.8697          1.0052            1.0046        10.85
HnswRaBitQ-m16-ex1 (self)                              1_355.34     8_134.07     9_489.41       0.8305          1.0109            1.0098        10.85
HnswRaBitQ-m16-ex3-ef15 (query)                        1_435.06       677.15     2_112.21       0.9278          1.0209            1.0008        13.90
HnswRaBitQ-m16-ex3-ef30 (query)                        1_435.06     1_016.44     2_451.50       0.9511          1.0010            1.0003        13.90
HnswRaBitQ-m16-ex3-ef60 (query)                        1_435.06     1_502.96     2_938.01       0.9574          1.0006            1.0001        13.90
HnswRaBitQ-m16-ex3-ef120 (query)                       1_435.06     2_104.35     3_539.40       0.9582          1.0005            1.0001        13.90
HnswRaBitQ-m16-ex3 (self)                              1_435.06     8_197.60     9_632.66       0.9469          1.0009            1.0005        13.90
HnswRaBitQ-m16-ex5-ef15 (query)                        1_519.85       671.27     2_191.12       0.9479          1.0025            1.0001        16.95
HnswRaBitQ-m16-ex5-ef30 (query)                        1_519.85     1_008.38     2_528.23       0.9780          1.0006            1.0000        16.95
HnswRaBitQ-m16-ex5-ef60 (query)                        1_519.85     1_506.00     3_025.85       0.9865          1.0001            1.0000        16.95
HnswRaBitQ-m16-ex5-ef120 (query)                       1_519.85     2_107.61     3_627.46       0.9877          1.0001            1.0000        16.95
HnswRaBitQ-m16-ex5 (self)                              1_519.85     8_133.10     9_652.95       0.9835          1.0002            1.0000        16.95
HnswRaBitQ-m16-ex8-ef15 (query)                        2_531.63       738.03     3_269.66       0.9533          1.0024            1.0000        21.53
HnswRaBitQ-m16-ex8-ef30 (query)                        2_531.63     1_077.71     3_609.34       0.9869          1.0006            1.0000        21.53
HnswRaBitQ-m16-ex8-ef60 (query)                        2_531.63     1_637.51     4_169.14       0.9966          1.0001            1.0000        21.53
HnswRaBitQ-m16-ex8-ef120 (query)                       2_531.63     2_269.12     4_800.75       0.9981          1.0001            1.0000        21.53
HnswRaBitQ-m16-ex8 (self)                              2_531.63     8_897.69    11_429.33       0.9965          1.0001            1.0000        21.53
HnswRaBitQ-m32-ex1-ef15 (query)                        1_525.71       841.39     2_367.10       0.8623          1.0061            1.0052        16.95
HnswRaBitQ-m32-ex1-ef30 (query)                        1_525.71     1_227.73     2_753.44       0.8686          1.0054            1.0047        16.95
HnswRaBitQ-m32-ex1-ef60 (query)                        1_525.71     1_746.93     3_272.64       0.8697          1.0052            1.0046        16.95
HnswRaBitQ-m32-ex1-ef120 (query)                       1_525.71     2_306.03     3_831.74       0.8698          1.0052            1.0046        16.95
HnswRaBitQ-m32-ex1 (self)                              1_525.71     9_613.10    11_138.81       0.8306          1.0108            1.0098        16.95
HnswRaBitQ-m32-ex3-ef15 (query)                        1_458.70       848.78     2_307.48       0.9431          1.0015            1.0004        20.00
HnswRaBitQ-m32-ex3-ef30 (query)                        1_458.70     1_237.18     2_695.89       0.9556          1.0006            1.0002        20.00
HnswRaBitQ-m32-ex3-ef60 (query)                        1_458.70     1_768.14     3_226.84       0.9580          1.0005            1.0001        20.00
HnswRaBitQ-m32-ex3-ef120 (query)                       1_458.70     2_336.29     3_794.99       0.9583          1.0004            1.0001        20.00
HnswRaBitQ-m32-ex3 (self)                              1_458.70     9_623.89    11_082.59       0.9475          1.0009            1.0005        20.00
HnswRaBitQ-m32-ex5-ef15 (query)                        1_649.45       857.02     2_506.46       0.9678          1.0011            1.0000        23.06
HnswRaBitQ-m32-ex5-ef30 (query)                        1_649.45     1_241.83     2_891.28       0.9840          1.0002            1.0000        23.06
HnswRaBitQ-m32-ex5-ef60 (query)                        1_649.45     1_757.53     3_406.98       0.9875          1.0001            1.0000        23.06
HnswRaBitQ-m32-ex5-ef120 (query)                       1_649.45     2_345.36     3_994.81       0.9878          1.0000            1.0000        23.06
HnswRaBitQ-m32-ex5 (self)                              1_649.45    10_149.90    11_799.35       0.9843          1.0001            1.0000        23.06
HnswRaBitQ-m32-ex8-ef15 (query)                        2_381.02       860.40     3_241.42       0.9750          1.0011            1.0000        27.63
HnswRaBitQ-m32-ex8-ef30 (query)                        2_381.02     1_268.68     3_649.71       0.9938          1.0002            1.0000        27.63
HnswRaBitQ-m32-ex8-ef60 (query)                        2_381.02     1_787.50     4_168.52       0.9980          1.0000            1.0000        27.63
HnswRaBitQ-m32-ex8-ef120 (query)                       2_381.02     2_373.74     4_754.76       0.9984          1.0000            1.0000        27.63
HnswRaBitQ-m32-ex8 (self)                              2_381.02     9_730.70    12_111.72       0.9975          1.0000            1.0000        27.63
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 512D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        70.32     1_349.23     1_419.54       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         70.32     4_537.17     4_607.49       1.0000          1.0000            1.0000        97.66
QG-d32-l64-ef15 (query)                                5_932.93       157.50     6_090.43       0.9593          1.0032            1.0000       214.01
QG-d32-l64-ef30 (query)                                5_932.93       242.32     6_175.25       0.9919          1.0013            1.0000       214.01
QG-d32-l64-ef60 (query)                                5_932.93       421.94     6_354.87       0.9979          1.0006            1.0000       214.01
QG-d32-l64-ef120 (query)                               5_932.93       736.42     6_669.35       0.9992          1.0004            1.0000       214.01
QG-d32-l64 (self)                                      5_932.93     1_307.67     7_240.61       0.9980          1.0007            1.0000       214.01
QG-d32-l128-ef15 (query)                               6_726.36       149.78     6_876.13       0.9610          1.0031            1.0000       214.01
QG-d32-l128-ef30 (query)                               6_726.36       244.86     6_971.22       0.9932          1.0012            1.0000       214.01
QG-d32-l128-ef60 (query)                               6_726.36       425.40     7_151.76       0.9987          1.0005            1.0000       214.01
QG-d32-l128-ef120 (query)                              6_726.36       750.02     7_476.38       0.9995          1.0003            1.0000       214.01
QG-d32-l128 (self)                                     6_726.36     1_262.12     7_988.48       0.9986          1.0006            1.0000       214.01
QG-d64-l64-ef15 (query)                               16_756.97       226.19    16_983.16       0.9920          1.0002            1.0000       329.97
QG-d64-l64-ef30 (query)                               16_756.97       401.40    17_158.37       0.9997          1.0000            1.0000       329.97
QG-d64-l64-ef60 (query)                               16_756.97       701.03    17_457.99       1.0000          1.0000            1.0000       329.97
QG-d64-l64-ef120 (query)                              16_756.97     1_260.15    18_017.12       1.0000          1.0000            1.0000       329.97
QG-d64-l64 (self)                                     16_756.97     2_168.35    18_925.32       1.0000          1.0000            1.0000       329.97
QG-d64-l128-ef15 (query)                              17_300.29       224.91    17_525.20       0.9923          1.0002            1.0000       329.97
QG-d64-l128-ef30 (query)                              17_300.29       394.22    17_694.51       0.9997          1.0000            1.0000       329.97
QG-d64-l128-ef60 (query)                              17_300.29       710.26    18_010.55       1.0000          1.0000            1.0000       329.97
QG-d64-l128-ef120 (query)                             17_300.29     1_256.88    18_557.17       1.0000          1.0000            1.0000       329.97
QG-d64-l128 (self)                                    17_300.29     2_193.33    19_493.62       1.0000          1.0000            1.0000       329.97
HnswRaBitQ-m16-ex1-ef15 (query)                        2_493.09     1_449.50     3_942.59       0.8456          1.0064            1.0044        14.15
HnswRaBitQ-m16-ex1-ef30 (query)                        2_493.09     2_163.09     4_656.18       0.8664          1.0042            1.0033        14.15
HnswRaBitQ-m16-ex1-ef60 (query)                        2_493.09     3_179.04     5_672.13       0.8737          1.0035            1.0029        14.15
HnswRaBitQ-m16-ex1-ef120 (query)                       2_493.09     4_352.95     6_846.05       0.8749          1.0034            1.0028        14.15
HnswRaBitQ-m16-ex1 (self)                              2_493.09    17_026.81    19_519.90       0.8358          1.0068            1.0060        14.15
HnswRaBitQ-m16-ex3-ef15 (query)                        2_648.50     1_450.63     4_099.13       0.9096          1.0037            1.0011        20.25
HnswRaBitQ-m16-ex3-ef30 (query)                        2_648.50     2_169.59     4_818.09       0.9451          1.0013            1.0003        20.25
HnswRaBitQ-m16-ex3-ef60 (query)                        2_648.50     3_198.36     5_846.85       0.9570          1.0006            1.0001        20.25
HnswRaBitQ-m16-ex3-ef120 (query)                       2_648.50     4_362.49     7_010.98       0.9592          1.0004            1.0001        20.25
HnswRaBitQ-m16-ex3 (self)                              2_648.50    17_070.19    19_718.68       0.9467          1.0009            1.0003        20.25
HnswRaBitQ-m16-ex5-ef15 (query)                        2_849.97     1_458.88     4_308.85       0.9241          1.0043            1.0007        26.36
HnswRaBitQ-m16-ex5-ef30 (query)                        2_849.97     2_180.42     5_030.39       0.9678          1.0010            1.0000        26.36
HnswRaBitQ-m16-ex5-ef60 (query)                        2_849.97     3_251.59     6_101.56       0.9837          1.0003            1.0000        26.36
HnswRaBitQ-m16-ex5-ef120 (query)                       2_849.97     4_398.04     7_248.01       0.9870          1.0002            1.0000        26.36
HnswRaBitQ-m16-ex5 (self)                              2_849.97    17_135.29    19_985.27       0.9813          1.0004            1.0000        26.36
HnswRaBitQ-m16-ex8-ef15 (query)                        4_355.74     1_471.03     5_826.77       0.9294          1.0036            1.0005        35.51
HnswRaBitQ-m16-ex8-ef30 (query)                        4_355.74     2_200.45     6_556.19       0.9755          1.0010            1.0000        35.51
HnswRaBitQ-m16-ex8-ef60 (query)                        4_355.74     3_239.94     7_595.67       0.9940          1.0003            1.0000        35.51
HnswRaBitQ-m16-ex8-ef120 (query)                       4_355.74     4_439.37     8_795.11       0.9976          1.0002            1.0000        35.51
HnswRaBitQ-m16-ex8 (self)                              4_355.74    18_094.94    22_450.68       0.9939          1.0003            1.0000        35.51
HnswRaBitQ-m32-ex1-ef15 (query)                        2_549.22     2_007.26     4_556.48       0.8622          1.0045            1.0034        20.25
HnswRaBitQ-m32-ex1-ef30 (query)                        2_549.22     2_856.56     5_405.77       0.8725          1.0035            1.0030        20.25
HnswRaBitQ-m32-ex1-ef60 (query)                        2_549.22     3_911.71     6_460.93       0.8748          1.0033            1.0029        20.25
HnswRaBitQ-m32-ex1-ef120 (query)                       2_549.22     4_911.24     7_460.45       0.8751          1.0032            1.0028        20.25
HnswRaBitQ-m32-ex1 (self)                              2_549.22    20_956.21    23_505.42       0.8365          1.0066            1.0060        20.25
HnswRaBitQ-m32-ex3-ef15 (query)                        2_700.40     1_982.83     4_683.23       0.9375          1.0016            1.0003        26.36
HnswRaBitQ-m32-ex3-ef30 (query)                        2_700.40     2_847.17     5_547.57       0.9549          1.0006            1.0001        26.36
HnswRaBitQ-m32-ex3-ef60 (query)                        2_700.40     3_943.32     6_643.72       0.9590          1.0003            1.0001        26.36
HnswRaBitQ-m32-ex3-ef120 (query)                       2_700.40     4_990.55     7_690.95       0.9596          1.0003            1.0001        26.36
HnswRaBitQ-m32-ex3 (self)                              2_700.40    21_203.04    23_903.44       0.9484          1.0006            1.0003        26.36
HnswRaBitQ-m32-ex5-ef15 (query)                        2_925.23     2_003.60     4_928.83       0.9588          1.0014            1.0000        32.46
HnswRaBitQ-m32-ex5-ef30 (query)                        2_925.23     2_964.53     5_889.76       0.9813          1.0003            1.0000        32.46
HnswRaBitQ-m32-ex5-ef60 (query)                        2_925.23     4_059.34     6_984.57       0.9869          1.0001            1.0000        32.46
HnswRaBitQ-m32-ex5-ef120 (query)                       2_925.23     5_033.23     7_958.46       0.9876          1.0000            1.0000        32.46
HnswRaBitQ-m32-ex5 (self)                              2_925.23    21_279.26    24_204.49       0.9839          1.0001            1.0000        32.46
HnswRaBitQ-m32-ex8-ef15 (query)                        4_421.94     2_018.36     6_440.29       0.9652          1.0013            1.0000        41.62
HnswRaBitQ-m32-ex8-ef30 (query)                        4_421.94     2_955.84     7_377.78       0.9905          1.0003            1.0000        41.62
HnswRaBitQ-m32-ex8-ef60 (query)                        4_421.94     3_990.22     8_412.15       0.9972          1.0001            1.0000        41.62
HnswRaBitQ-m32-ex8-ef120 (query)                       4_421.94     5_090.12     9_512.05       0.9982          1.0000            1.0000        41.62
HnswRaBitQ-m32-ex8 (self)                              4_421.94    21_477.11    25_899.05       0.9968          1.0001            1.0000        41.62
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 768D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       109.65     1_981.51     2_091.16       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        109.65     6_522.98     6_632.63       1.0000          1.0000            1.0000       146.48
QG-d32-l64-ef15 (query)                                8_064.08       259.96     8_324.04       0.9513          1.0068            1.0000       311.66
QG-d32-l64-ef30 (query)                                8_064.08       411.98     8_476.06       0.9810          1.0050            1.0000       311.66
QG-d32-l64-ef60 (query)                                8_064.08       669.79     8_733.87       0.9906          1.0042            1.0000       311.66
QG-d32-l64-ef120 (query)                               8_064.08     1_102.33     9_166.41       0.9937          1.0036            1.0000       311.66
QG-d32-l64 (self)                                      8_064.08     2_135.69    10_199.77       0.9907          1.0045            1.0000       311.66
QG-d32-l128-ef15 (query)                               9_057.34       256.06     9_313.40       0.9434          1.0469            1.0000       311.66
QG-d32-l128-ef30 (query)                               9_057.34       430.71     9_488.05       0.9723          1.0451            1.0000       311.66
QG-d32-l128-ef60 (query)                               9_057.34       694.62     9_751.96       0.9811          1.0437            1.0000       311.66
QG-d32-l128-ef120 (query)                              9_057.34     1_097.21    10_154.55       0.9858          1.0360            1.0000       311.66
QG-d32-l128 (self)                                     9_057.34     2_160.31    11_217.65       0.9812          1.0439            1.0000       311.66
QG-d64-l64-ef15 (query)                               30_485.77       510.27    30_996.04       0.9925          1.0010            1.0000       476.46
QG-d64-l64-ef30 (query)                               30_485.77       836.52    31_322.29       0.9982          1.0005            1.0000       476.46
QG-d64-l64-ef60 (query)                               30_485.77     1_320.71    31_806.48       0.9994          1.0002            1.0000       476.46
QG-d64-l64-ef120 (query)                              30_485.77     2_085.96    32_571.73       0.9997          1.0001            1.0000       476.46
QG-d64-l64 (self)                                     30_485.77     4_250.69    34_736.46       0.9995          1.0002            1.0000       476.46
QG-d64-l128-ef15 (query)                              30_664.55       500.44    31_164.99       0.9929          1.0008            1.0000       476.46
QG-d64-l128-ef30 (query)                              30_664.55       827.19    31_491.74       0.9985          1.0004            1.0000       476.46
QG-d64-l128-ef60 (query)                              30_664.55     1_312.99    31_977.54       0.9996          1.0002            1.0000       476.46
QG-d64-l128-ef120 (query)                             30_664.55     2_081.83    32_746.37       0.9999          1.0001            1.0000       476.46
QG-d64-l128 (self)                                    30_664.55     4_272.67    34_937.22       0.9997          1.0002            1.0000       476.46
HnswRaBitQ-m16-ex1-ef15 (query)                        3_526.99     2_263.74     5_790.74       0.8230          1.0094            1.0039        17.45
HnswRaBitQ-m16-ex1-ef30 (query)                        3_526.99     3_363.31     6_890.30       0.8514          1.0049            1.0029        17.45
HnswRaBitQ-m16-ex1-ef60 (query)                        3_526.99     4_910.03     8_437.02       0.8619          1.0034            1.0025        17.45
HnswRaBitQ-m16-ex1-ef120 (query)                       3_526.99     6_630.98    10_157.97       0.8643          1.0030            1.0024        17.45
HnswRaBitQ-m16-ex1 (self)                              3_526.99    26_235.54    29_762.53       0.8175          1.0061            1.0051        17.45
HnswRaBitQ-m16-ex3-ef15 (query)                        3_803.16     2_420.89     6_224.05       0.8573        143.4689            1.0014        26.61
HnswRaBitQ-m16-ex3-ef30 (query)                        3_803.16     3_401.24     7_204.40       0.8997        143.4165            1.0004        26.61
HnswRaBitQ-m16-ex3-ef60 (query)                        3_803.16     5_620.70     9_423.86       0.9495          1.1399            1.0001        26.61
HnswRaBitQ-m16-ex3-ef120 (query)                       3_803.16     7_014.25    10_817.41       0.9544          1.0005            1.0001        26.61
HnswRaBitQ-m16-ex3 (self)                              3_803.16    28_105.52    31_908.68       0.9374          1.1228            1.0003        26.61
HnswRaBitQ-m16-ex5-ef15 (query)                        4_005.98     2_290.81     6_296.79       0.9018          1.0064            1.0010        35.76
HnswRaBitQ-m16-ex5-ef30 (query)                        4_005.98     3_399.15     7_405.13       0.9537          1.0022            1.0000        35.76
HnswRaBitQ-m16-ex5-ef60 (query)                        4_005.98     4_971.91     8_977.90       0.9780          1.0008            1.0000        35.76
HnswRaBitQ-m16-ex5-ef120 (query)                       4_005.98     6_715.15    10_721.13       0.9846          1.0004            1.0000        35.76
HnswRaBitQ-m16-ex5 (self)                              4_005.98    26_417.30    30_423.29       0.9754          1.0008            1.0000        35.76
HnswRaBitQ-m16-ex8-ef15 (query)                        6_551.95     2_340.52     8_892.47       0.9067          1.0114            1.0010        49.50
HnswRaBitQ-m16-ex8-ef30 (query)                        6_551.95     3_437.42     9_989.37       0.9614          1.0022            1.0000        49.50
HnswRaBitQ-m16-ex8-ef60 (query)                        6_551.95     5_027.71    11_579.66       0.9886          1.0008            1.0000        49.50
HnswRaBitQ-m16-ex8-ef120 (query)                       6_551.95     6_879.14    13_431.09       0.9963          1.0003            1.0000        49.50
HnswRaBitQ-m16-ex8 (self)                              6_551.95    28_378.70    34_930.64       0.9892          1.0007            1.0000        49.50
HnswRaBitQ-m32-ex1-ef15 (query)                        3_576.35     3_193.40     6_769.75       0.8478          1.0187            1.0030        23.56
HnswRaBitQ-m32-ex1-ef30 (query)                        3_576.35     4_580.58     8_156.92       0.8608          1.0099            1.0026        23.56
HnswRaBitQ-m32-ex1-ef60 (query)                        3_576.35     6_165.94     9_742.29       0.8642          1.0028            1.0024        23.56
HnswRaBitQ-m32-ex1-ef120 (query)                       3_576.35     7_565.92    11_142.27       0.8650          1.0027            1.0024        23.56
HnswRaBitQ-m32-ex1 (self)                              3_576.35    32_884.05    36_460.40       0.8190          1.0056            1.0051        23.56
HnswRaBitQ-m32-ex3-ef15 (query)                        3_817.70     3_234.75     7_052.45       0.9263          1.0098            1.0004        32.71
HnswRaBitQ-m32-ex3-ef30 (query)                        3_817.70     4_684.08     8_501.78       0.9481          1.0008            1.0002        32.71
HnswRaBitQ-m32-ex3-ef60 (query)                        3_817.70     6_226.19    10_043.89       0.9544          1.0004            1.0001        32.71
HnswRaBitQ-m32-ex3-ef120 (query)                       3_817.70     7_724.86    11_542.56       0.9553          1.0003            1.0001        32.71
HnswRaBitQ-m32-ex3 (self)                              3_817.70    33_274.74    37_092.44       0.9416          1.0006            1.0003        32.71
HnswRaBitQ-m32-ex5-ef15 (query)                        4_169.92     3_228.54     7_398.46       0.9476          1.0023            1.0000        41.87
HnswRaBitQ-m32-ex5-ef30 (query)                        4_169.92     4_700.90     8_870.82       0.9756          1.0006            1.0000        41.87
HnswRaBitQ-m32-ex5-ef60 (query)                        4_169.92     6_276.78    10_446.70       0.9844          1.0002            1.0000        41.87
HnswRaBitQ-m32-ex5-ef120 (query)                       4_169.92     7_797.03    11_966.94       0.9860          1.0001            1.0000        41.87
HnswRaBitQ-m32-ex5 (self)                              4_169.92    35_834.16    40_004.07       0.9806          1.0002            1.0000        41.87
HnswRaBitQ-m32-ex8-ef15 (query)                        6_310.59     3_354.09     9_664.68       0.9547          1.0031            1.0000        55.60
HnswRaBitQ-m32-ex8-ef30 (query)                        6_310.59     4_623.31    10_933.91       0.9854          1.0006            1.0000        55.60
HnswRaBitQ-m32-ex8-ef60 (query)                        6_310.59     6_250.32    12_560.91       0.9956          1.0002            1.0000        55.60
HnswRaBitQ-m32-ex8-ef120 (query)                       6_310.59     7_774.45    14_085.04       0.9976          1.0001            1.0000        55.60
HnswRaBitQ-m32-ex8 (self)                              6_310.59    33_284.90    39_595.49       0.9952          1.0002            1.0000        55.60
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Cell embeddings data

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 256D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        34.19       725.41       759.60       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         34.19     2_447.31     2_481.50       1.0000          1.0000            1.0000        48.83
QG-d32-l64-ef15 (query)                                2_393.21        55.54     2_448.75       0.9890          1.0005            1.0000       116.35
QG-d32-l64-ef30 (query)                                2_393.21        87.70     2_480.91       0.9999          1.0000            1.0000       116.35
QG-d32-l64-ef60 (query)                                2_393.21       153.40     2_546.62       1.0000          1.0000            1.0000       116.35
QG-d32-l64-ef120 (query)                               2_393.21       282.95     2_676.16       1.0000          1.0000            1.0000       116.35
QG-d32-l64 (self)                                      2_393.21       461.27     2_854.48       1.0000          1.0000            1.0000       116.35
QG-d32-l128-ef15 (query)                               3_605.76        55.81     3_661.56       0.9886          1.0005            1.0000       116.35
QG-d32-l128-ef30 (query)                               3_605.76        86.76     3_692.52       0.9999          1.0000            1.0000       116.35
QG-d32-l128-ef60 (query)                               3_605.76       159.50     3_765.26       1.0000          1.0000            1.0000       116.35
QG-d32-l128-ef120 (query)                              3_605.76       292.51     3_898.26       1.0000          1.0000            1.0000       116.35
QG-d32-l128 (self)                                     3_605.76       460.34     4_066.10       1.0000          1.0000            1.0000       116.35
QG-d64-l64-ef15 (query)                                2_998.61        61.54     3_060.15       0.9905          1.0004            1.0000       183.49
QG-d64-l64-ef30 (query)                                2_998.61        95.31     3_093.92       0.9999          1.0000            1.0000       183.49
QG-d64-l64-ef60 (query)                                2_998.61       168.12     3_166.73       1.0000          1.0000            1.0000       183.49
QG-d64-l64-ef120 (query)                               2_998.61       324.29     3_322.90       1.0000          1.0000            1.0000       183.49
QG-d64-l64 (self)                                      2_998.61       536.79     3_535.40       1.0000          1.0000            1.0000       183.49
QG-d64-l128-ef15 (query)                               4_583.61        63.49     4_647.10       0.9907          1.0004            1.0000       183.49
QG-d64-l128-ef30 (query)                               4_583.61       104.66     4_688.27       1.0000          1.0000            1.0000       183.49
QG-d64-l128-ef60 (query)                               4_583.61       172.95     4_756.56       1.0000          1.0000            1.0000       183.49
QG-d64-l128-ef120 (query)                              4_583.61       335.44     4_919.05       1.0000          1.0000            1.0000       183.49
QG-d64-l128 (self)                                     4_583.61       543.86     5_127.47       1.0000          1.0000            1.0000       183.49
HnswRaBitQ-m16-ex1-ef15 (query)                        1_579.02       402.60     1_981.62       0.9348          1.0481            1.0033        10.85
HnswRaBitQ-m16-ex1-ef30 (query)                        1_579.02       576.63     2_155.64       0.9370          1.0364            1.0031        10.85
HnswRaBitQ-m16-ex1-ef60 (query)                        1_579.02       883.44     2_462.46       0.9386          1.0214            1.0031        10.85
HnswRaBitQ-m16-ex1-ef120 (query)                       1_579.02     1_401.92     2_980.94       0.9400          1.0100            1.0031        10.85
HnswRaBitQ-m16-ex1 (self)                              1_579.02     4_812.86     6_391.88       0.9110          1.0282            1.0088        10.85
HnswRaBitQ-m16-ex3-ef15 (query)                        1_665.82       406.11     2_071.93       0.9761          1.0209            1.0000        13.90
HnswRaBitQ-m16-ex3-ef30 (query)                        1_665.82       579.35     2_245.17       0.9787          1.0144            1.0000        13.90
HnswRaBitQ-m16-ex3-ef60 (query)                        1_665.82       884.23     2_550.05       0.9795          1.0094            1.0000        13.90
HnswRaBitQ-m16-ex3-ef120 (query)                       1_665.82     1_406.31     3_072.13       0.9805          1.0026            1.0000        13.90
HnswRaBitQ-m16-ex3 (self)                              1_665.82     4_820.45     6_486.27       0.9731          1.0085            1.0000        13.90
HnswRaBitQ-m16-ex5-ef15 (query)                        1_780.97       407.85     2_188.83       0.9887          1.0234            1.0000        16.95
HnswRaBitQ-m16-ex5-ef30 (query)                        1_780.97       587.02     2_367.99       0.9916          1.0196            1.0000        16.95
HnswRaBitQ-m16-ex5-ef60 (query)                        1_780.97       905.45     2_686.43       0.9927          1.0132            1.0000        16.95
HnswRaBitQ-m16-ex5-ef120 (query)                       1_780.97     1_447.85     3_228.83       0.9939          1.0049            1.0000        16.95
HnswRaBitQ-m16-ex5 (self)                              1_780.97     4_885.93     6_666.90       0.9910          1.0102            1.0000        16.95
HnswRaBitQ-m16-ex8-ef15 (query)                        2_505.97       413.49     2_919.46       0.9938          1.0185            1.0000        21.53
HnswRaBitQ-m16-ex8-ef30 (query)                        2_505.97       598.15     3_104.12       0.9967          1.0147            1.0000        21.53
HnswRaBitQ-m16-ex8-ef60 (query)                        2_505.97       919.86     3_425.83       0.9978          1.0093            1.0000        21.53
HnswRaBitQ-m16-ex8-ef120 (query)                       2_505.97     1_493.89     3_999.86       0.9984          1.0047            1.0000        21.53
HnswRaBitQ-m16-ex8 (self)                              2_505.97     4_974.95     7_480.92       0.9975          1.0094            1.0000        21.53
HnswRaBitQ-m32-ex1-ef15 (query)                        1_658.87       491.69     2_150.56       0.9403          1.0064            1.0031        16.95
HnswRaBitQ-m32-ex1-ef30 (query)                        1_658.87       667.60     2_326.47       0.9407          1.0059            1.0031        16.95
HnswRaBitQ-m32-ex1-ef60 (query)                        1_658.87       993.21     2_652.08       0.9407          1.0056            1.0031        16.95
HnswRaBitQ-m32-ex1-ef120 (query)                       1_658.87     1_540.50     3_199.38       0.9408          1.0050            1.0031        16.95
HnswRaBitQ-m32-ex1 (self)                              1_658.87     5_511.73     7_170.60       0.9131          1.0128            1.0087        16.95
HnswRaBitQ-m32-ex3-ef15 (query)                        1_736.07       501.65     2_237.72       0.9784          1.0067            1.0000        20.00
HnswRaBitQ-m32-ex3-ef30 (query)                        1_736.07       679.66     2_415.73       0.9802          1.0045            1.0000        20.00
HnswRaBitQ-m32-ex3-ef60 (query)                        1_736.07     1_005.00     2_741.08       0.9804          1.0036            1.0000        20.00
HnswRaBitQ-m32-ex3-ef120 (query)                       1_736.07     1_574.13     3_310.20       0.9808          1.0013            1.0000        20.00
HnswRaBitQ-m32-ex3 (self)                              1_736.07     5_637.46     7_373.54       0.9739          1.0033            1.0000        20.00
HnswRaBitQ-m32-ex5-ef15 (query)                        1_850.20       505.51     2_355.71       0.9927          1.0032            1.0000        23.06
HnswRaBitQ-m32-ex5-ef30 (query)                        1_850.20       681.88     2_532.08       0.9944          1.0021            1.0000        23.06
HnswRaBitQ-m32-ex5-ef60 (query)                        1_850.20     1_013.86     2_864.06       0.9946          1.0011            1.0000        23.06
HnswRaBitQ-m32-ex5-ef120 (query)                       1_850.20     1_581.67     3_431.87       0.9948          1.0000            1.0000        23.06
HnswRaBitQ-m32-ex5 (self)                              1_850.20     5_556.48     7_406.68       0.9926          1.0010            1.0000        23.06
HnswRaBitQ-m32-ex8-ef15 (query)                        2_597.43       500.63     3_098.06       0.9974          1.0013            1.0000        27.63
HnswRaBitQ-m32-ex8-ef30 (query)                        2_597.43       702.15     3_299.58       0.9993          1.0000            1.0000        27.63
HnswRaBitQ-m32-ex8-ef60 (query)                        2_597.43     1_047.29     3_644.73       0.9993          1.0000            1.0000        27.63
HnswRaBitQ-m32-ex8-ef120 (query)                       2_597.43     1_616.96     4_214.39       0.9993          1.0000            1.0000        27.63
HnswRaBitQ-m32-ex8 (self)                              2_597.43     5_666.48     8_263.91       0.9990          1.0002            1.0000        27.63
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 512D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        78.30     1_406.72     1_485.02       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         78.30     4_795.88     4_874.18       1.0000          1.0000            1.0000        97.66
QG-d32-l64-ef15 (query)                                4_485.44        87.90     4_573.34       0.9905          1.0005            1.0000       214.01
QG-d32-l64-ef30 (query)                                4_485.44       124.16     4_609.59       0.9999          1.0000            1.0000       214.01
QG-d32-l64-ef60 (query)                                4_485.44       195.44     4_680.88       1.0000          1.0000            1.0000       214.01
QG-d32-l64-ef120 (query)                               4_485.44       348.51     4_833.95       1.0000          1.0000            1.0000       214.01
QG-d32-l64 (self)                                      4_485.44       590.46     5_075.89       1.0000          1.0000            1.0000       214.01
QG-d32-l128-ef15 (query)                               6_988.23        91.35     7_079.58       0.9904          1.0005            1.0000       214.01
QG-d32-l128-ef30 (query)                               6_988.23       123.80     7_112.03       0.9999          1.0000            1.0000       214.01
QG-d32-l128-ef60 (query)                               6_988.23       199.41     7_187.64       1.0000          1.0000            1.0000       214.01
QG-d32-l128-ef120 (query)                              6_988.23       359.11     7_347.34       1.0000          1.0000            1.0000       214.01
QG-d32-l128 (self)                                     6_988.23       589.66     7_577.89       1.0000          1.0000            1.0000       214.01
QG-d64-l64-ef15 (query)                                5_570.49        99.83     5_670.31       0.9916          1.0004            1.0000       329.97
QG-d64-l64-ef30 (query)                                5_570.49       144.48     5_714.97       0.9999          1.0000            1.0000       329.97
QG-d64-l64-ef60 (query)                                5_570.49       222.84     5_793.33       1.0000          1.0000            1.0000       329.97
QG-d64-l64-ef120 (query)                               5_570.49       408.25     5_978.74       1.0000          1.0000            1.0000       329.97
QG-d64-l64 (self)                                      5_570.49       676.95     6_247.44       1.0000          1.0000            1.0000       329.97
QG-d64-l128-ef15 (query)                               8_865.21       105.02     8_970.22       0.9918          1.0004            1.0000       329.97
QG-d64-l128-ef30 (query)                               8_865.21       135.34     9_000.54       0.9999          1.0000            1.0000       329.97
QG-d64-l128-ef60 (query)                               8_865.21       219.64     9_084.85       1.0000          1.0000            1.0000       329.97
QG-d64-l128-ef120 (query)                              8_865.21       401.72     9_266.92       1.0000          1.0000            1.0000       329.97
QG-d64-l128 (self)                                     8_865.21       670.06     9_535.27       1.0000          1.0000            1.0000       329.97
HnswRaBitQ-m16-ex1-ef15 (query)                        2_638.81       759.58     3_398.39       0.9535          1.0315            1.0007        14.15
HnswRaBitQ-m16-ex1-ef30 (query)                        2_638.81     1_071.73     3_710.54       0.9555          1.0256            1.0006        14.15
HnswRaBitQ-m16-ex1-ef60 (query)                        2_638.81     1_589.73     4_228.54       0.9563          1.0169            1.0006        14.15
HnswRaBitQ-m16-ex1-ef120 (query)                       2_638.81     2_457.48     5_096.28       0.9570          1.0101            1.0006        14.15
HnswRaBitQ-m16-ex1 (self)                              2_638.81     8_472.34    11_111.14       0.9328          1.0224            1.0040        14.15
HnswRaBitQ-m16-ex3-ef15 (query)                        2_827.32       765.31     3_592.63       0.9813          1.0246            1.0000        20.25
HnswRaBitQ-m16-ex3-ef30 (query)                        2_827.32     1_073.18     3_900.50       0.9834          1.0231            1.0000        20.25
HnswRaBitQ-m16-ex3-ef60 (query)                        2_827.32     1_600.95     4_428.28       0.9841          1.0152            1.0000        20.25
HnswRaBitQ-m16-ex3-ef120 (query)                       2_827.32     2_477.26     5_304.58       0.9847          1.0101            1.0000        20.25
HnswRaBitQ-m16-ex3 (self)                              2_827.32     8_489.53    11_316.85       0.9796          1.0156            1.0000        20.25
HnswRaBitQ-m16-ex5-ef15 (query)                        3_016.64       778.10     3_794.75       0.9878          1.0706            1.0000        26.36
HnswRaBitQ-m16-ex5-ef30 (query)                        3_016.64     1_095.26     4_111.91       0.9920          1.0437            1.0000        26.36
HnswRaBitQ-m16-ex5-ef60 (query)                        3_016.64     1_620.41     4_637.06       0.9940          1.0220            1.0000        26.36
HnswRaBitQ-m16-ex5-ef120 (query)                       3_016.64     2_534.08     5_550.73       0.9949          1.0118            1.0000        26.36
HnswRaBitQ-m16-ex5 (self)                              3_016.64     8_636.43    11_653.07       0.9921          1.0243            1.0000        26.36
HnswRaBitQ-m16-ex8-ef15 (query)                        4_528.04       783.99     5_312.03       0.9908          1.0808            1.0000        35.51
HnswRaBitQ-m16-ex8-ef30 (query)                        4_528.04     1_108.33     5_636.37       0.9939          1.0698            1.0000        35.51
HnswRaBitQ-m16-ex8-ef60 (query)                        4_528.04     1_653.04     6_181.08       0.9963          1.0355            1.0000        35.51
HnswRaBitQ-m16-ex8-ef120 (query)                       4_528.04     2_541.60     7_069.64       0.9978          1.0178            1.0000        35.51
HnswRaBitQ-m16-ex8 (self)                              4_528.04     8_660.64    13_188.68       0.9960          1.0376            1.0000        35.51
HnswRaBitQ-m32-ex1-ef15 (query)                        2_757.73       939.70     3_697.43       0.9570          1.0027            1.0006        20.25
HnswRaBitQ-m32-ex1-ef30 (query)                        2_757.73     1_269.64     4_027.37       0.9578          1.0025            1.0006        20.25
HnswRaBitQ-m32-ex1-ef60 (query)                        2_757.73     1_815.01     4_572.74       0.9578          1.0025            1.0006        20.25
HnswRaBitQ-m32-ex1-ef120 (query)                       2_757.73     2_754.93     5_512.66       0.9578          1.0025            1.0006        20.25
HnswRaBitQ-m32-ex1 (self)                              2_757.73     9_691.24    12_448.97       0.9344          1.0068            1.0039        20.25
HnswRaBitQ-m32-ex3-ef15 (query)                        2_958.86       960.98     3_919.84       0.9846          1.0007            1.0000        26.36
HnswRaBitQ-m32-ex3-ef30 (query)                        2_958.86     1_280.39     4_239.24       0.9859          1.0005            1.0000        26.36
HnswRaBitQ-m32-ex3-ef60 (query)                        2_958.86     1_821.56     4_780.42       0.9860          1.0002            1.0000        26.36
HnswRaBitQ-m32-ex3-ef120 (query)                       2_958.86     2_755.08     5_713.94       0.9860          1.0002            1.0000        26.36
HnswRaBitQ-m32-ex3 (self)                              2_958.86     9_774.91    12_733.77       0.9815          1.0005            1.0000        26.36
HnswRaBitQ-m32-ex5-ef15 (query)                        3_130.97       963.13     4_094.10       0.9945          1.0002            1.0000        32.46
HnswRaBitQ-m32-ex5-ef30 (query)                        3_130.97     1_279.46     4_410.43       0.9962          1.0000            1.0000        32.46
HnswRaBitQ-m32-ex5-ef60 (query)                        3_130.97     1_850.66     4_981.64       0.9963          1.0000            1.0000        32.46
HnswRaBitQ-m32-ex5-ef120 (query)                       3_130.97     2_859.04     5_990.02       0.9963          1.0000            1.0000        32.46
HnswRaBitQ-m32-ex5 (self)                              3_130.97     9_801.61    12_932.58       0.9946          1.0001            1.0000        32.46
HnswRaBitQ-m32-ex8-ef15 (query)                        4_658.06       967.99     5_626.06       0.9975          1.0019            1.0000        41.62
HnswRaBitQ-m32-ex8-ef30 (query)                        4_658.06     1_318.16     5_976.22       0.9992          1.0018            1.0000        41.62
HnswRaBitQ-m32-ex8-ef60 (query)                        4_658.06     1_855.56     6_513.62       0.9995          1.0000            1.0000        41.62
HnswRaBitQ-m32-ex8-ef120 (query)                       4_658.06     2_831.97     7_490.04       0.9995          1.0000            1.0000        41.62
HnswRaBitQ-m32-ex8 (self)                              4_658.06     9_909.45    14_567.51       0.9992          1.0003            1.0000        41.62
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
=====================================================================================================================================================
Benchmark: 50k samples, 768D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       103.55     1_985.18     2_088.73       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        103.55     6_540.05     6_643.59       1.0000          1.0000            1.0000       146.48
QG-d32-l64-ef15 (query)                                6_294.76       105.00     6_399.76       0.9897          1.0005            1.0000       311.66
QG-d32-l64-ef30 (query)                                6_294.76       147.69     6_442.44       0.9999          1.0000            1.0000       311.66
QG-d32-l64-ef60 (query)                                6_294.76       228.35     6_523.10       1.0000          1.0000            1.0000       311.66
QG-d32-l64-ef120 (query)                               6_294.76       406.20     6_700.96       1.0000          1.0000            1.0000       311.66
QG-d32-l64 (self)                                      6_294.76       679.07     6_973.83       1.0000          1.0000            1.0000       311.66
QG-d32-l128-ef15 (query)                               9_737.75       108.60     9_846.35       0.9896          1.0005            1.0000       311.66
QG-d32-l128-ef30 (query)                               9_737.75       145.29     9_883.04       0.9998          1.0000            1.0000       311.66
QG-d32-l128-ef60 (query)                               9_737.75       229.15     9_966.90       1.0000          1.0000            1.0000       311.66
QG-d32-l128-ef120 (query)                              9_737.75       407.69    10_145.44       1.0000          1.0000            1.0000       311.66
QG-d32-l128 (self)                                     9_737.75       681.44    10_419.19       1.0000          1.0000            1.0000       311.66
QG-d64-l64-ef15 (query)                                7_899.53       121.46     8_021.00       0.9907          1.0004            1.0000       476.46
QG-d64-l64-ef30 (query)                                7_899.53       169.92     8_069.45       0.9999          1.0000            1.0000       476.46
QG-d64-l64-ef60 (query)                                7_899.53       266.42     8_165.95       1.0000          1.0000            1.0000       476.46
QG-d64-l64-ef120 (query)                               7_899.53       481.09     8_380.62       1.0000          1.0000            1.0000       476.46
QG-d64-l64 (self)                                      7_899.53       820.67     8_720.20       1.0000          1.0000            1.0000       476.46
QG-d64-l128-ef15 (query)                              12_633.46       115.80    12_749.26       0.9908          1.0004            1.0000       476.46
QG-d64-l128-ef30 (query)                              12_633.46       166.19    12_799.65       0.9999          1.0000            1.0000       476.46
QG-d64-l128-ef60 (query)                              12_633.46       272.23    12_905.69       1.0000          1.0000            1.0000       476.46
QG-d64-l128-ef120 (query)                             12_633.46       482.42    13_115.89       1.0000          1.0000            1.0000       476.46
QG-d64-l128 (self)                                    12_633.46       808.77    13_442.24       1.0000          1.0000            1.0000       476.46
HnswRaBitQ-m16-ex1-ef15 (query)                        3_638.00     1_119.06     4_757.06       0.9511          1.0039            1.0014        17.45
HnswRaBitQ-m16-ex1-ef30 (query)                        3_638.00     1_571.05     5_209.05       0.9521          1.0037            1.0013        17.45
HnswRaBitQ-m16-ex1-ef60 (query)                        3_638.00     2_245.15     5_883.15       0.9522          1.0037            1.0013        17.45
HnswRaBitQ-m16-ex1-ef120 (query)                       3_638.00     3_449.52     7_087.52       0.9522          1.0037            1.0013        17.45
HnswRaBitQ-m16-ex1 (self)                              3_638.00    11_972.95    15_610.95       0.9215          1.0118            1.0063        17.45
HnswRaBitQ-m16-ex3-ef15 (query)                        3_864.04     1_124.80     4_988.84       0.9766          1.0040            1.0000        26.61
HnswRaBitQ-m16-ex3-ef30 (query)                        3_864.04     1_552.50     5_416.55       0.9783          1.0023            1.0000        26.61
HnswRaBitQ-m16-ex3-ef60 (query)                        3_864.04     2_283.85     6_147.89       0.9784          1.0023            1.0000        26.61
HnswRaBitQ-m16-ex3-ef120 (query)                       3_864.04     3_457.50     7_321.54       0.9785          1.0016            1.0000        26.61
HnswRaBitQ-m16-ex3 (self)                              3_864.04    12_182.16    16_046.20       0.9709          1.0027            1.0000        26.61
HnswRaBitQ-m16-ex5-ef15 (query)                        4_125.11     1_109.70     5_234.81       0.9920          1.0072            1.0000        35.76
HnswRaBitQ-m16-ex5-ef30 (query)                        4_125.11     1_557.37     5_682.48       0.9943          1.0067            1.0000        35.76
HnswRaBitQ-m16-ex5-ef60 (query)                        4_125.11     2_285.10     6_410.20       0.9945          1.0053            1.0000        35.76
HnswRaBitQ-m16-ex5-ef120 (query)                       4_125.11     3_480.29     7_605.40       0.9946          1.0046            1.0000        35.76
HnswRaBitQ-m16-ex5 (self)                              4_125.11    12_117.86    16_242.97       0.9925          1.0060            1.0000        35.76
HnswRaBitQ-m16-ex8-ef15 (query)                        5_985.37     1_122.46     7_107.83       0.9965          1.0064            1.0000        49.50
HnswRaBitQ-m16-ex8-ef30 (query)                        5_985.37     1_550.07     7_535.44       0.9987          1.0062            1.0000        49.50
HnswRaBitQ-m16-ex8-ef60 (query)                        5_985.37     2_291.29     8_276.66       0.9990          1.0048            1.0000        49.50
HnswRaBitQ-m16-ex8-ef120 (query)                       5_985.37     3_499.75     9_485.12       0.9991          1.0036            1.0000        49.50
HnswRaBitQ-m16-ex8 (self)                              5_985.37    12_098.23    18_083.60       0.9990          1.0022            1.0000        49.50
HnswRaBitQ-m32-ex1-ef15 (query)                        3_767.47     1_389.14     5_156.61       0.9513          1.0039            1.0014        23.56
HnswRaBitQ-m32-ex1-ef30 (query)                        3_767.47     1_827.00     5_594.47       0.9521          1.0038            1.0013        23.56
HnswRaBitQ-m32-ex1-ef60 (query)                        3_767.47     2_581.36     6_348.83       0.9522          1.0038            1.0013        23.56
HnswRaBitQ-m32-ex1-ef120 (query)                       3_767.47     3_843.82     7_611.29       0.9522          1.0038            1.0013        23.56
HnswRaBitQ-m32-ex1 (self)                              3_767.47    13_828.42    17_595.89       0.9216          1.0107            1.0063        23.56
HnswRaBitQ-m32-ex3-ef15 (query)                        4_098.29     1_402.26     5_500.55       0.9772          1.0017            1.0000        32.71
HnswRaBitQ-m32-ex3-ef30 (query)                        4_098.29     1_833.19     5_931.48       0.9784          1.0015            1.0000        32.71
HnswRaBitQ-m32-ex3-ef60 (query)                        4_098.29     2_591.69     6_689.98       0.9785          1.0015            1.0000        32.71
HnswRaBitQ-m32-ex3-ef120 (query)                       4_098.29     3_845.56     7_943.85       0.9785          1.0015            1.0000        32.71
HnswRaBitQ-m32-ex3 (self)                              4_098.29    13_789.21    17_887.50       0.9710          1.0015            1.0000        32.71
HnswRaBitQ-m32-ex5-ef15 (query)                        4_236.51     1_419.23     5_655.74       0.9934          1.0011            1.0000        41.87
HnswRaBitQ-m32-ex5-ef30 (query)                        4_236.51     1_858.88     6_095.39       0.9950          1.0009            1.0000        41.87
HnswRaBitQ-m32-ex5-ef60 (query)                        4_236.51     2_610.11     6_846.62       0.9951          1.0009            1.0000        41.87
HnswRaBitQ-m32-ex5-ef120 (query)                       4_236.51     3_880.61     8_117.13       0.9951          1.0009            1.0000        41.87
HnswRaBitQ-m32-ex5 (self)                              4_236.51    13_970.92    18_207.43       0.9932          1.0005            1.0000        41.87
HnswRaBitQ-m32-ex8-ef15 (query)                        6_033.73     1_404.31     7_438.04       0.9976          1.0011            1.0000        55.60
HnswRaBitQ-m32-ex8-ef30 (query)                        6_033.73     1_847.90     7_881.64       0.9992          1.0009            1.0000        55.60
HnswRaBitQ-m32-ex8-ef60 (query)                        6_033.73     2_602.99     8_636.72       0.9993          1.0009            1.0000        55.60
HnswRaBitQ-m32-ex8-ef120 (query)                       6_033.73     3_877.38     9_911.12       0.9993          1.0009            1.0000        55.60
HnswRaBitQ-m32-ex8 (self)                              6_033.73    13_920.87    19_954.61       0.9991          1.0005            1.0000        55.60
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
- *reranking_factor (rf)*: As for the other indices. Default `20`. The
  exhaustive grid runs `rf0`, `5`, `10` and `20`; the IVF grid runs `rf0`, `10`
  and `20`.

**Tunable parameters *(IVF-specific)*:**

- *Number of lists (nl)*: Number of k-means clusters, `sqrt(n)` as a default.
  The grid runs `sqrt(n/2)`, `sqrt(n)` and `sqrt(2n)`.
- *Number of probes (np)*: `sqrt(nlist)`, `sqrt(2 * nlist)` and 5% of `nlist`,
  deduplicated.

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
Exhaustive (query)                                        33.76       695.71       729.47       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.76     2_261.29     2_295.05       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              160.23       372.66       532.89       0.0971          1.7176            1.5958         7.12
ExhaustiveTQ-b2-rf5 (query)                              160.23       448.87       609.10       0.2336          1.2025            1.2204         7.12
ExhaustiveTQ-b2-rf10 (query)                             160.23       587.11       747.34       0.2853          1.1453            1.1620         7.12
ExhaustiveTQ-b2-rf20 (query)                             160.23       969.58     1_129.81       0.3809          1.0970            1.0941         7.12
ExhaustiveTQ-b2 (self)                                   160.23     3_147.68     3_307.92       0.3814          1.0980            1.0957         7.12
ExhaustiveTQ-b4-rf0 (query)                              236.34       571.82       808.17       0.1094          1.5328            1.4997        13.22
ExhaustiveTQ-b4-rf5 (query)                              236.34       663.59       899.93       0.2368          1.1884            1.2090        13.22
ExhaustiveTQ-b4-rf10 (query)                             236.34       806.98     1_043.32       0.2884          1.1372            1.1543        13.22
ExhaustiveTQ-b4-rf20 (query)                             236.34     1_183.83     1_420.17       0.3823          1.0940            1.0970        13.22
ExhaustiveTQ-b4 (self)                                   236.34     3_923.69     4_160.04       0.3841          1.0938            1.0948        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                          349.21       107.42       456.63       0.0971          1.7176            1.5958         7.80
IVF-TQ-b2-nl158-np12-rf0 (query)                         349.21       115.02       464.23       0.0971          1.7176            1.5958         7.80
IVF-TQ-b2-nl158-np17-rf0 (query)                         349.21       125.95       475.16       0.0971          1.7176            1.5958         7.80
IVF-TQ-b2-nl158-np7-rf10 (query)                         349.21       295.45       644.66       0.2853          1.1453            1.1620         7.80
IVF-TQ-b2-nl158-np7-rf20 (query)                         349.21       609.78       958.99       0.3809          1.0970            1.0941         7.80
IVF-TQ-b2-nl158-np12-rf10 (query)                        349.21       305.84       655.05       0.2853          1.1453            1.1620         7.80
IVF-TQ-b2-nl158-np12-rf20 (query)                        349.21       632.92       982.13       0.3809          1.0970            1.0941         7.80
IVF-TQ-b2-nl158-np17-rf10 (query)                        349.21       319.27       668.48       0.2853          1.1453            1.1620         7.80
IVF-TQ-b2-nl158-np17-rf20 (query)                        349.21       665.64     1_014.85       0.3809          1.0970            1.0941         7.80
IVF-TQ-b2-nl158 (self)                                   349.21     1_048.92     1_398.13       0.3815          1.0980            1.0957         7.80
IVF-TQ-b2-nl223-np11-rf0 (query)                         349.66       113.24       462.90       0.0971          1.7165            1.5941         7.93
IVF-TQ-b2-nl223-np14-rf0 (query)                         349.66       121.20       470.86       0.0971          1.7176            1.5958         7.93
IVF-TQ-b2-nl223-np21-rf0 (query)                         349.66       142.42       492.08       0.0971          1.7176            1.5958         7.93
IVF-TQ-b2-nl223-np11-rf10 (query)                        349.66       288.47       638.13       0.2855          1.1450            1.1618         7.93
IVF-TQ-b2-nl223-np11-rf20 (query)                        349.66       557.18       906.84       0.3813          1.0967            1.0934         7.93
IVF-TQ-b2-nl223-np14-rf10 (query)                        349.66       294.67       644.33       0.2853          1.1453            1.1620         7.93
IVF-TQ-b2-nl223-np14-rf20 (query)                        349.66       577.63       927.30       0.3809          1.0970            1.0941         7.93
IVF-TQ-b2-nl223-np21-rf10 (query)                        349.66       311.16       660.82       0.2853          1.1453            1.1620         7.93
IVF-TQ-b2-nl223-np21-rf20 (query)                        349.66       604.37       954.03       0.3809          1.0970            1.0941         7.93
IVF-TQ-b2-nl223 (self)                                   349.66     1_024.55     1_374.21       0.3815          1.0980            1.0957         7.93
IVF-TQ-b2-nl316-np15-rf0 (query)                         431.84       117.54       549.38       0.0973          1.7112            1.5945         8.10
IVF-TQ-b2-nl316-np17-rf0 (query)                         431.84       121.85       553.68       0.0971          1.7175            1.5958         8.10
IVF-TQ-b2-nl316-np25-rf0 (query)                         431.84       133.83       565.66       0.0971          1.7176            1.5958         8.10
IVF-TQ-b2-nl316-np15-rf10 (query)                        431.84       285.77       717.61       0.2855          1.1451            1.1619         8.10
IVF-TQ-b2-nl316-np15-rf20 (query)                        431.84       540.43       972.27       0.3812          1.0968            1.0936         8.10
IVF-TQ-b2-nl316-np17-rf10 (query)                        431.84       288.47       720.31       0.2853          1.1453            1.1620         8.10
IVF-TQ-b2-nl316-np17-rf20 (query)                        431.84       573.14     1_004.98       0.3809          1.0970            1.0941         8.10
IVF-TQ-b2-nl316-np25-rf10 (query)                        431.84       305.09       736.92       0.2853          1.1453            1.1620         8.10
IVF-TQ-b2-nl316-np25-rf20 (query)                        431.84       578.47     1_010.31       0.3809          1.0970            1.0941         8.10
IVF-TQ-b2-nl316 (self)                                   431.84       972.42     1_404.26       0.3815          1.0980            1.0957         8.10
IVF-TQ-b4-nl158-np7-rf0 (query)                          436.10       144.46       580.56       0.1094          1.5328            1.4997        14.05
IVF-TQ-b4-nl158-np12-rf0 (query)                         436.10       164.31       600.41       0.1094          1.5328            1.4997        14.05
IVF-TQ-b4-nl158-np17-rf0 (query)                         436.10       187.36       623.46       0.1094          1.5328            1.4997        14.05
IVF-TQ-b4-nl158-np7-rf10 (query)                         436.10       354.60       790.70       0.2884          1.1372            1.1543        14.05
IVF-TQ-b4-nl158-np7-rf20 (query)                         436.10       687.22     1_123.32       0.3823          1.0940            1.0970        14.05
IVF-TQ-b4-nl158-np12-rf10 (query)                        436.10       376.50       812.60       0.2884          1.1372            1.1543        14.05
IVF-TQ-b4-nl158-np12-rf20 (query)                        436.10       730.99     1_167.09       0.3823          1.0940            1.0970        14.05
IVF-TQ-b4-nl158-np17-rf10 (query)                        436.10       381.96       818.06       0.2884          1.1372            1.1543        14.05
IVF-TQ-b4-nl158-np17-rf20 (query)                        436.10       726.71     1_162.81       0.3823          1.0940            1.0970        14.05
IVF-TQ-b4-nl158 (self)                                   436.10     1_093.49     1_529.59       0.3841          1.0938            1.0948        14.05
IVF-TQ-b4-nl223-np11-rf0 (query)                         446.59       156.42       603.01       0.1094          1.5315            1.4987        14.24
IVF-TQ-b4-nl223-np14-rf0 (query)                         446.59       167.06       613.65       0.1094          1.5328            1.4996        14.24
IVF-TQ-b4-nl223-np21-rf0 (query)                         446.59       188.65       635.24       0.1094          1.5328            1.4996        14.24
IVF-TQ-b4-nl223-np11-rf10 (query)                        446.59       340.36       786.95       0.2886          1.1370            1.1542        14.24
IVF-TQ-b4-nl223-np11-rf20 (query)                        446.59       622.66     1_069.24       0.3826          1.0939            1.0966        14.24
IVF-TQ-b4-nl223-np14-rf10 (query)                        446.59       351.55       798.14       0.2884          1.1372            1.1543        14.24
IVF-TQ-b4-nl223-np14-rf20 (query)                        446.59       640.96     1_087.55       0.3823          1.0940            1.0970        14.24
IVF-TQ-b4-nl223-np21-rf10 (query)                        446.59       375.09       821.68       0.2884          1.1372            1.1543        14.24
IVF-TQ-b4-nl223-np21-rf20 (query)                        446.59       674.38     1_120.96       0.3823          1.0940            1.0970        14.24
IVF-TQ-b4-nl223 (self)                                   446.59     1_085.96     1_532.55       0.3841          1.0938            1.0948        14.24
IVF-TQ-b4-nl316-np15-rf0 (query)                         520.72       162.68       683.39       0.1094          1.5320            1.4991        14.49
IVF-TQ-b4-nl316-np17-rf0 (query)                         520.72       168.43       689.14       0.1094          1.5328            1.4996        14.49
IVF-TQ-b4-nl316-np25-rf0 (query)                         520.72       191.08       711.80       0.1094          1.5328            1.4997        14.49
IVF-TQ-b4-nl316-np15-rf10 (query)                        520.72       354.82       875.54       0.2885          1.1371            1.1542        14.49
IVF-TQ-b4-nl316-np15-rf20 (query)                        520.72       600.54     1_121.25       0.3826          1.0939            1.0969        14.49
IVF-TQ-b4-nl316-np17-rf10 (query)                        520.72       347.46       868.17       0.2884          1.1372            1.1543        14.49
IVF-TQ-b4-nl316-np17-rf20 (query)                        520.72       620.90     1_141.62       0.3823          1.0940            1.0970        14.49
IVF-TQ-b4-nl316-np25-rf10 (query)                        520.72       373.09       893.80       0.2884          1.1372            1.1543        14.49
IVF-TQ-b4-nl316-np25-rf20 (query)                        520.72       652.30     1_173.02       0.3823          1.0940            1.0970        14.49
IVF-TQ-b4-nl316 (self)                                   520.72     1_108.82     1_629.53       0.3841          1.0938            1.0948        14.49
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
Exhaustive (query)                                        69.77     1_379.11     1_448.87       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.77     4_619.65     4_689.42       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              351.00       663.02     1_014.01       0.1207          1.3711            1.3320        13.97
ExhaustiveTQ-b2-rf5 (query)                              351.00       763.25     1_114.25       0.2421          1.1334            1.1574        13.97
ExhaustiveTQ-b2-rf10 (query)                             351.00       897.52     1_248.52       0.2934          1.0981            1.1177        13.97
ExhaustiveTQ-b2-rf20 (query)                             351.00     1_283.23     1_634.23       0.3880          1.0664            1.0469        13.97
ExhaustiveTQ-b2 (self)                                   351.00     4_185.01     4_536.01       0.3879          1.0667            1.0471        13.97
ExhaustiveTQ-b4-rf0 (query)                              471.24     1_177.58     1_648.82       0.1315          1.3172            1.3127        26.18
ExhaustiveTQ-b4-rf5 (query)                              471.24     1_304.09     1_775.33       0.2471          1.1254            1.1483        26.18
ExhaustiveTQ-b4-rf10 (query)                             471.24     1_412.95     1_884.19       0.2970          1.0929            1.0980        26.18
ExhaustiveTQ-b4-rf20 (query)                             471.24     1_791.60     2_262.85       0.3883          1.0643            1.0492        26.18
ExhaustiveTQ-b4 (self)                                   471.24     5_895.82     6_367.07       0.3881          1.0646            1.0495        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                          637.61       194.23       831.84       0.1207          1.3711            1.3320        14.96
IVF-TQ-b2-nl158-np12-rf0 (query)                         637.61       214.45       852.06       0.1207          1.3711            1.3320        14.96
IVF-TQ-b2-nl158-np17-rf0 (query)                         637.61       226.91       864.52       0.1207          1.3711            1.3320        14.96
IVF-TQ-b2-nl158-np7-rf10 (query)                         637.61       420.82     1_058.43       0.2934          1.0981            1.1177        14.96
IVF-TQ-b2-nl158-np7-rf20 (query)                         637.61       774.02     1_411.63       0.3880          1.0664            1.0469        14.96
IVF-TQ-b2-nl158-np12-rf10 (query)                        637.61       430.66     1_068.27       0.2934          1.0981            1.1177        14.96
IVF-TQ-b2-nl158-np12-rf20 (query)                        637.61       792.81     1_430.42       0.3880          1.0664            1.0469        14.96
IVF-TQ-b2-nl158-np17-rf10 (query)                        637.61       448.09     1_085.70       0.2934          1.0981            1.1177        14.96
IVF-TQ-b2-nl158-np17-rf20 (query)                        637.61       810.00     1_447.61       0.3880          1.0664            1.0469        14.96
IVF-TQ-b2-nl158 (self)                                   637.61     1_417.17     2_054.78       0.3879          1.0667            1.0471        14.96
IVF-TQ-b2-nl223-np11-rf0 (query)                         640.12       206.03       846.16       0.1208          1.3696            1.3298        15.18
IVF-TQ-b2-nl223-np14-rf0 (query)                         640.12       219.24       859.37       0.1207          1.3711            1.3320        15.18
IVF-TQ-b2-nl223-np21-rf0 (query)                         640.12       243.96       884.08       0.1207          1.3711            1.3320        15.18
IVF-TQ-b2-nl223-np11-rf10 (query)                        640.12       406.63     1_046.76       0.2937          1.0979            1.1176        15.18
IVF-TQ-b2-nl223-np11-rf20 (query)                        640.12       704.10     1_344.23       0.3887          1.0662            1.0467        15.18
IVF-TQ-b2-nl223-np14-rf10 (query)                        640.12       415.24     1_055.36       0.2934          1.0981            1.1178        15.18
IVF-TQ-b2-nl223-np14-rf20 (query)                        640.12       726.02     1_366.14       0.3880          1.0664            1.0469        15.18
IVF-TQ-b2-nl223-np21-rf10 (query)                        640.12       442.95     1_083.07       0.2934          1.0981            1.1177        15.18
IVF-TQ-b2-nl223-np21-rf20 (query)                        640.12       774.50     1_414.63       0.3880          1.0664            1.0469        15.18
IVF-TQ-b2-nl223 (self)                                   640.12     1_431.56     2_071.69       0.3879          1.0667            1.0471        15.18
IVF-TQ-b2-nl316-np15-rf0 (query)                         709.17       216.15       925.33       0.1208          1.3690            1.3288        15.56
IVF-TQ-b2-nl316-np17-rf0 (query)                         709.17       219.93       929.11       0.1208          1.3707            1.3313        15.56
IVF-TQ-b2-nl316-np25-rf0 (query)                         709.17       247.00       956.18       0.1207          1.3711            1.3320        15.56
IVF-TQ-b2-nl316-np15-rf10 (query)                        709.17       404.05     1_113.22       0.2939          1.0977            1.1176        15.56
IVF-TQ-b2-nl316-np15-rf20 (query)                        709.17       685.03     1_394.21       0.3891          1.0660            1.0466        15.56
IVF-TQ-b2-nl316-np17-rf10 (query)                        709.17       412.94     1_122.12       0.2935          1.0980            1.1177        15.56
IVF-TQ-b2-nl316-np17-rf20 (query)                        709.17       716.46     1_425.64       0.3882          1.0664            1.0469        15.56
IVF-TQ-b2-nl316-np25-rf10 (query)                        709.17       541.36     1_250.53       0.2934          1.0981            1.1178        15.56
IVF-TQ-b2-nl316-np25-rf20 (query)                        709.17       866.61     1_575.78       0.3880          1.0664            1.0469        15.56
IVF-TQ-b2-nl316 (self)                                   709.17     1_550.35     2_259.52       0.3879          1.0667            1.0471        15.56
IVF-TQ-b4-nl158-np7-rf0 (query)                          806.04       293.99     1_100.03       0.1315          1.3172            1.3127        27.46
IVF-TQ-b4-nl158-np12-rf0 (query)                         806.04       415.47     1_221.51       0.1315          1.3172            1.3127        27.46
IVF-TQ-b4-nl158-np17-rf0 (query)                         806.04       376.46     1_182.50       0.1315          1.3172            1.3127        27.46
IVF-TQ-b4-nl158-np7-rf10 (query)                         806.04       517.68     1_323.72       0.2970          1.0929            1.0979        27.46
IVF-TQ-b4-nl158-np7-rf20 (query)                         806.04       864.49     1_670.53       0.3883          1.0643            1.0492        27.46
IVF-TQ-b4-nl158-np12-rf10 (query)                        806.04       545.79     1_351.83       0.2970          1.0929            1.0980        27.46
IVF-TQ-b4-nl158-np12-rf20 (query)                        806.04       900.47     1_706.51       0.3883          1.0643            1.0492        27.46
IVF-TQ-b4-nl158-np17-rf10 (query)                        806.04       562.27     1_368.30       0.2970          1.0929            1.0980        27.46
IVF-TQ-b4-nl158-np17-rf20 (query)                        806.04       928.29     1_734.33       0.3883          1.0643            1.0492        27.46
IVF-TQ-b4-nl158 (self)                                   806.04     1_561.54     2_367.58       0.3881          1.0646            1.0495        27.46
IVF-TQ-b4-nl223-np11-rf0 (query)                         776.15       289.85     1_066.00       0.1316          1.3158            1.3117        27.77
IVF-TQ-b4-nl223-np14-rf0 (query)                         776.15       320.01     1_096.16       0.1315          1.3172            1.3127        27.77
IVF-TQ-b4-nl223-np21-rf0 (query)                         776.15       345.17     1_121.32       0.1315          1.3172            1.3127        27.77
IVF-TQ-b4-nl223-np11-rf10 (query)                        776.15       511.81     1_287.96       0.2974          1.0926            1.0973        27.77
IVF-TQ-b4-nl223-np11-rf20 (query)                        776.15       818.35     1_594.50       0.3889          1.0641            1.0488        27.77
IVF-TQ-b4-nl223-np14-rf10 (query)                        776.15       519.93     1_296.07       0.2970          1.0929            1.0980        27.77
IVF-TQ-b4-nl223-np14-rf20 (query)                        776.15       834.95     1_611.09       0.3883          1.0643            1.0492        27.77
IVF-TQ-b4-nl223-np21-rf10 (query)                        776.15       567.94     1_344.09       0.2970          1.0929            1.0980        27.77
IVF-TQ-b4-nl223-np21-rf20 (query)                        776.15       902.41     1_678.56       0.3882          1.0643            1.0492        27.77
IVF-TQ-b4-nl223 (self)                                   776.15     1_588.11     2_364.25       0.3881          1.0646            1.0495        27.77
IVF-TQ-b4-nl316-np15-rf0 (query)                         837.96       306.60     1_144.56       0.1316          1.3151            1.3108        28.36
IVF-TQ-b4-nl316-np17-rf0 (query)                         837.96       324.23     1_162.19       0.1315          1.3164            1.3121        28.36
IVF-TQ-b4-nl316-np25-rf0 (query)                         837.96       353.55     1_191.51       0.1315          1.3172            1.3127        28.36
IVF-TQ-b4-nl316-np15-rf10 (query)                        837.96       510.74     1_348.70       0.2976          1.0925            1.0968        28.36
IVF-TQ-b4-nl316-np15-rf20 (query)                        837.96       792.81     1_630.77       0.3893          1.0639            1.0484        28.36
IVF-TQ-b4-nl316-np17-rf10 (query)                        837.96       531.63     1_369.59       0.2971          1.0928            1.0978        28.36
IVF-TQ-b4-nl316-np17-rf20 (query)                        837.96       830.23     1_668.19       0.3885          1.0642            1.0491        28.36
IVF-TQ-b4-nl316-np25-rf10 (query)                        837.96       573.23     1_411.19       0.2970          1.0929            1.0980        28.36
IVF-TQ-b4-nl316-np25-rf20 (query)                        837.96       887.18     1_725.14       0.3883          1.0643            1.0492        28.36
IVF-TQ-b4-nl316 (self)                                   837.96     1_612.23     2_450.19       0.3881          1.0646            1.0495        28.36
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
Exhaustive (query)                                       102.54     2_010.90     2_113.44       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        102.54     6_658.29     6_760.83       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              615.96       987.29     1_603.25       0.1292          1.2710            1.2627        21.33
ExhaustiveTQ-b2-rf5 (query)                              615.96     1_136.41     1_752.37       0.2468          1.1062            1.1332        21.33
ExhaustiveTQ-b2-rf10 (query)                             615.96     1_221.13     1_837.08       0.3000          1.0773            1.0631        21.33
ExhaustiveTQ-b2-rf20 (query)                             615.96     1_632.00     2_247.96       0.3957          1.0509            1.0334        21.33
ExhaustiveTQ-b2 (self)                                   615.96     5_336.78     5_952.74       0.3973          1.0507            1.0331        21.33
ExhaustiveTQ-b4-rf0 (query)                              759.32     1_802.50     2_561.82       0.1340          1.2532            1.2592        39.64
ExhaustiveTQ-b4-rf5 (query)                              759.32     1_901.96     2_661.28       0.2401          1.1136            1.1402        39.64
ExhaustiveTQ-b4-rf10 (query)                             759.32     2_024.72     2_784.04       0.2870          1.0888            1.1143        39.64
ExhaustiveTQ-b4-rf20 (query)                             759.32     2_425.48     3_184.80       0.3752          1.0657            1.0812        39.64
ExhaustiveTQ-b4 (self)                                   759.32     8_088.58     8_847.90       0.3767          1.0654            1.0638        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_044.76       316.72     1_361.48       0.1292          1.2710            1.2627        22.63
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_044.76       322.50     1_367.26       0.1292          1.2710            1.2627        22.63
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_044.76       341.96     1_386.73       0.1292          1.2710            1.2627        22.63
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_044.76       536.67     1_581.43       0.3000          1.0774            1.0631        22.63
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_044.76       905.71     1_950.48       0.3957          1.0509            1.0334        22.63
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_044.76       563.14     1_607.90       0.3000          1.0774            1.0631        22.63
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_044.76       953.44     1_998.20       0.3957          1.0509            1.0334        22.63
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_044.76       583.22     1_627.98       0.3000          1.0773            1.0631        22.63
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_044.76       972.16     2_016.92       0.3957          1.0509            1.0334        22.63
IVF-TQ-b2-nl158 (self)                                 1_044.76     1_847.15     2_891.91       0.3973          1.0507            1.0331        22.63
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_002.94       312.20     1_315.15       0.1292          1.2710            1.2627        23.04
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_002.94       325.33     1_328.27       0.1292          1.2710            1.2627        23.04
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_002.94       376.66     1_379.61       0.1292          1.2710            1.2628        23.04
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_002.94       565.77     1_568.71       0.3000          1.0774            1.0632        23.04
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_002.94       866.42     1_869.36       0.3957          1.0509            1.0334        23.04
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_002.94       575.30     1_578.24       0.3000          1.0774            1.0632        23.04
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_002.94       894.56     1_897.50       0.3957          1.0509            1.0334        23.04
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_002.94       595.06     1_598.00       0.3000          1.0774            1.0631        23.04
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_002.94       936.77     1_939.71       0.3957          1.0509            1.0334        23.04
IVF-TQ-b2-nl223 (self)                                 1_002.94     1_860.42     2_863.37       0.3973          1.0507            1.0331        23.04
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_101.83       327.45     1_429.28       0.1292          1.2709            1.2627        23.59
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_101.83       342.26     1_444.09       0.1292          1.2710            1.2627        23.59
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_101.83       367.85     1_469.68       0.1292          1.2710            1.2627        23.59
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_101.83       551.84     1_653.68       0.3000          1.0773            1.0631        23.59
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_101.83       870.74     1_972.57       0.3957          1.0509            1.0334        23.59
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_101.83       554.02     1_655.85       0.3000          1.0773            1.0631        23.59
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_101.83       873.45     1_975.28       0.3957          1.0509            1.0334        23.59
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_101.83       602.18     1_704.02       0.3000          1.0774            1.0631        23.59
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_101.83       916.21     2_018.04       0.3956          1.0509            1.0334        23.59
IVF-TQ-b2-nl316 (self)                                 1_101.83     1_903.32     3_005.16       0.3973          1.0507            1.0331        23.59
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_154.15       429.71     1_583.86       0.1340          1.2532            1.2592        41.40
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_154.15       460.08     1_614.23       0.1340          1.2532            1.2592        41.40
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_154.15       498.33     1_652.48       0.1340          1.2532            1.2592        41.40
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_154.15       671.82     1_825.97       0.2870          1.0888            1.1143        41.40
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_154.15     1_042.53     2_196.68       0.3752          1.0657            1.0812        41.40
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_154.15       718.48     1_872.63       0.2870          1.0888            1.1143        41.40
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_154.15     1_113.57     2_267.72       0.3752          1.0657            1.0812        41.40
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_154.15       771.06     1_925.21       0.2870          1.0888            1.1143        41.40
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_154.15     1_147.07     2_301.22       0.3752          1.0657            1.0812        41.40
IVF-TQ-b4-nl158 (self)                                 1_154.15     2_148.34     3_302.49       0.3767          1.0654            1.0637        41.40
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_126.23       451.78     1_578.01       0.1340          1.2532            1.2592        42.04
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_126.23       480.44     1_606.67       0.1340          1.2532            1.2592        42.04
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_126.23       536.70     1_662.94       0.1340          1.2532            1.2592        42.04
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_126.23       690.11     1_816.35       0.2870          1.0888            1.1143        42.04
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_126.23     1_004.11     2_130.34       0.3752          1.0657            1.0812        42.04
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_126.23       749.85     1_876.08       0.2870          1.0888            1.1143        42.04
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_126.23     1_044.75     2_170.98       0.3752          1.0657            1.0812        42.04
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_126.23       789.71     1_915.95       0.2870          1.0888            1.1143        42.04
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_126.23     1_126.46     2_252.69       0.3752          1.0657            1.0812        42.04
IVF-TQ-b4-nl223 (self)                                 1_126.23     2_214.53     3_340.76       0.3766          1.0654            1.0638        42.04
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_312.91       474.89     1_787.80       0.1340          1.2531            1.2592        42.85
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_312.91       499.86     1_812.76       0.1340          1.2532            1.2592        42.85
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_312.91       547.91     1_860.82       0.1340          1.2532            1.2592        42.85
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_312.91       697.57     2_010.48       0.2870          1.0888            1.1143        42.85
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_312.91     1_015.43     2_328.34       0.3753          1.0657            1.0812        42.85
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_312.91       715.29     2_028.20       0.2870          1.0888            1.1143        42.85
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_312.91     1_030.85     2_343.75       0.3752          1.0657            1.0812        42.85
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_312.91       794.82     2_107.73       0.2870          1.0888            1.1143        42.85
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_312.91     1_105.41     2_418.31       0.3752          1.0657            1.0812        42.85
IVF-TQ-b4-nl316 (self)                                 1_312.91     2_265.46     3_578.37       0.3767          1.0654            1.0638        42.85
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
Exhaustive (query)                                        33.53       734.15       767.68       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.53     2_468.34     2_501.87       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              157.21       377.12       534.33       0.0756          2.3283            1.9295         7.12
ExhaustiveTQ-b2-rf5 (query)                              157.21       449.36       606.57       0.2072          1.3307            1.3578         7.12
ExhaustiveTQ-b2-rf10 (query)                             157.21       577.51       734.72       0.2886          1.2206            1.2322         7.12
ExhaustiveTQ-b2-rf20 (query)                             157.21       953.32     1_110.53       0.4151          1.1328            1.1147         7.12
ExhaustiveTQ-b2 (self)                                   157.21     3_157.68     3_314.89       0.4136          1.1619            1.1367         7.12
ExhaustiveTQ-b4-rf0 (query)                              232.07       591.86       823.93       0.1023          1.7129            1.7532        13.22
ExhaustiveTQ-b4-rf5 (query)                              232.07       686.48       918.55       0.2385          1.2770            1.3000        13.22
ExhaustiveTQ-b4-rf10 (query)                             232.07       826.80     1_058.87       0.3202          1.1874            1.1953        13.22
ExhaustiveTQ-b4-rf20 (query)                             232.07     1_220.45     1_452.52       0.4481          1.1142            1.1029        13.22
ExhaustiveTQ-b4 (self)                                   232.07     3_920.73     4_152.80       0.4463          1.1397            1.1286        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                          322.97       104.96       427.92       0.0756          2.3282            1.9295         7.82
IVF-TQ-b2-nl158-np12-rf0 (query)                         322.97       125.90       448.86       0.0756          2.3282            1.9295         7.82
IVF-TQ-b2-nl158-np17-rf0 (query)                         322.97       132.26       455.23       0.0756          2.3282            1.9295         7.82
IVF-TQ-b2-nl158-np7-rf10 (query)                         322.97       299.54       622.51       0.2887          1.2206            1.2322         7.82
IVF-TQ-b2-nl158-np7-rf20 (query)                         322.97       647.85       970.82       0.4151          1.1328            1.1147         7.82
IVF-TQ-b2-nl158-np12-rf10 (query)                        322.97       316.67       639.64       0.2886          1.2206            1.2322         7.82
IVF-TQ-b2-nl158-np12-rf20 (query)                        322.97       659.84       982.81       0.4151          1.1328            1.1147         7.82
IVF-TQ-b2-nl158-np17-rf10 (query)                        322.97       351.86       674.83       0.2886          1.2206            1.2322         7.82
IVF-TQ-b2-nl158-np17-rf20 (query)                        322.97       738.46     1_061.43       0.4151          1.1328            1.1147         7.82
IVF-TQ-b2-nl158 (self)                                   322.97     1_078.89     1_401.86       0.4136          1.1619            1.1367         7.82
IVF-TQ-b2-nl223-np11-rf0 (query)                         366.66       108.61       475.26       0.0756          2.3254            1.9253         7.93
IVF-TQ-b2-nl223-np14-rf0 (query)                         366.66       116.32       482.98       0.0756          2.3281            1.9295         7.93
IVF-TQ-b2-nl223-np21-rf0 (query)                         366.66       142.24       508.90       0.0756          2.3282            1.9295         7.93
IVF-TQ-b2-nl223-np11-rf10 (query)                        366.66       281.32       647.97       0.2891          1.2202            1.2316         7.93
IVF-TQ-b2-nl223-np11-rf20 (query)                        366.66       559.54       926.19       0.4159          1.1325            1.1141         7.93
IVF-TQ-b2-nl223-np14-rf10 (query)                        366.66       293.50       660.15       0.2886          1.2206            1.2322         7.93
IVF-TQ-b2-nl223-np14-rf20 (query)                        366.66       584.29       950.94       0.4151          1.1328            1.1147         7.93
IVF-TQ-b2-nl223-np21-rf10 (query)                        366.66       338.04       704.70       0.2886          1.2206            1.2322         7.93
IVF-TQ-b2-nl223-np21-rf20 (query)                        366.66       646.96     1_013.62       0.4151          1.1328            1.1147         7.93
IVF-TQ-b2-nl223 (self)                                   366.66     1_079.32     1_445.97       0.4136          1.1619            1.1367         7.93
IVF-TQ-b2-nl316-np15-rf0 (query)                         422.05       116.59       538.63       0.0757          2.3274            1.9285         8.11
IVF-TQ-b2-nl316-np17-rf0 (query)                         422.05       119.27       541.31       0.0756          2.3282            1.9294         8.11
IVF-TQ-b2-nl316-np25-rf0 (query)                         422.05       138.97       561.02       0.0756          2.3282            1.9295         8.11
IVF-TQ-b2-nl316-np15-rf10 (query)                        422.05       288.39       710.44       0.2892          1.2202            1.2316         8.11
IVF-TQ-b2-nl316-np15-rf20 (query)                        422.05       539.36       961.40       0.4159          1.1325            1.1141         8.11
IVF-TQ-b2-nl316-np17-rf10 (query)                        422.05       286.85       708.90       0.2887          1.2206            1.2322         8.11
IVF-TQ-b2-nl316-np17-rf20 (query)                        422.05       540.27       962.32       0.4151          1.1328            1.1147         8.11
IVF-TQ-b2-nl316-np25-rf10 (query)                        422.05       312.80       734.85       0.2886          1.2206            1.2322         8.11
IVF-TQ-b2-nl316-np25-rf20 (query)                        422.05       595.32     1_017.37       0.4151          1.1328            1.1147         8.11
IVF-TQ-b2-nl316 (self)                                   422.05     1_069.40     1_491.45       0.4136          1.1619            1.1367         8.11
IVF-TQ-b4-nl158-np7-rf0 (query)                          399.73       139.84       539.57       0.1023          1.7129            1.7532        14.09
IVF-TQ-b4-nl158-np12-rf0 (query)                         399.73       160.72       560.45       0.1023          1.7129            1.7532        14.09
IVF-TQ-b4-nl158-np17-rf0 (query)                         399.73       188.04       587.77       0.1023          1.7129            1.7532        14.09
IVF-TQ-b4-nl158-np7-rf10 (query)                         399.73       350.06       749.79       0.3202          1.1873            1.1953        14.09
IVF-TQ-b4-nl158-np7-rf20 (query)                         399.73       678.10     1_077.83       0.4481          1.1142            1.1029        14.09
IVF-TQ-b4-nl158-np12-rf10 (query)                        399.73       372.61       772.34       0.3202          1.1873            1.1953        14.09
IVF-TQ-b4-nl158-np12-rf20 (query)                        399.73       728.88     1_128.61       0.4481          1.1142            1.1029        14.09
IVF-TQ-b4-nl158-np17-rf10 (query)                        399.73       419.58       819.31       0.3202          1.1873            1.1953        14.09
IVF-TQ-b4-nl158-np17-rf20 (query)                        399.73       800.83     1_200.56       0.4481          1.1142            1.1029        14.09
IVF-TQ-b4-nl158 (self)                                   399.73     1_131.70     1_531.43       0.4463          1.1397            1.1286        14.09
IVF-TQ-b4-nl223-np11-rf0 (query)                         451.40       155.24       606.64       0.1024          1.7109            1.7518        14.23
IVF-TQ-b4-nl223-np14-rf0 (query)                         451.40       161.32       612.72       0.1023          1.7129            1.7532        14.23
IVF-TQ-b4-nl223-np21-rf0 (query)                         451.40       207.79       659.19       0.1023          1.7129            1.7532        14.23
IVF-TQ-b4-nl223-np11-rf10 (query)                        451.40       348.00       799.40       0.3206          1.1870            1.1948        14.23
IVF-TQ-b4-nl223-np11-rf20 (query)                        451.40       636.35     1_087.75       0.4489          1.1139            1.1025        14.23
IVF-TQ-b4-nl223-np14-rf10 (query)                        451.40       354.09       805.49       0.3202          1.1873            1.1953        14.23
IVF-TQ-b4-nl223-np14-rf20 (query)                        451.40       666.69     1_118.09       0.4481          1.1142            1.1029        14.23
IVF-TQ-b4-nl223-np21-rf10 (query)                        451.40       410.12       861.52       0.3201          1.1874            1.1954        14.23
IVF-TQ-b4-nl223-np21-rf20 (query)                        451.40       736.85     1_188.25       0.4481          1.1142            1.1029        14.23
IVF-TQ-b4-nl223 (self)                                   451.40     1_159.68     1_611.08       0.4463          1.1397            1.1286        14.23
IVF-TQ-b4-nl316-np15-rf0 (query)                         535.00       156.53       691.53       0.1024          1.7112            1.7520        14.51
IVF-TQ-b4-nl316-np17-rf0 (query)                         535.00       173.39       708.39       0.1023          1.7121            1.7528        14.51
IVF-TQ-b4-nl316-np25-rf0 (query)                         535.00       202.94       737.94       0.1023          1.7129            1.7532        14.51
IVF-TQ-b4-nl316-np15-rf10 (query)                        535.00       331.26       866.26       0.3207          1.1869            1.1949        14.51
IVF-TQ-b4-nl316-np15-rf20 (query)                        535.00       589.11     1_124.12       0.4491          1.1138            1.1024        14.51
IVF-TQ-b4-nl316-np17-rf10 (query)                        535.00       348.99       883.99       0.3202          1.1873            1.1953        14.51
IVF-TQ-b4-nl316-np17-rf20 (query)                        535.00       609.21     1_144.21       0.4482          1.1142            1.1029        14.51
IVF-TQ-b4-nl316-np25-rf10 (query)                        535.00       393.50       928.51       0.3201          1.1874            1.1953        14.51
IVF-TQ-b4-nl316-np25-rf20 (query)                        535.00       655.08     1_190.08       0.4481          1.1142            1.1029        14.51
IVF-TQ-b4-nl316 (self)                                   535.00     1_121.03     1_656.03       0.4463          1.1397            1.1286        14.51
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
Exhaustive (query)                                        69.54     1_381.70     1_451.25       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.54     4_552.54     4_622.09       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              341.03       661.02     1_002.05       0.0844          1.6539            1.5906        13.97
ExhaustiveTQ-b2-rf5 (query)                              341.03       744.54     1_085.57       0.2173          1.2230            1.2549        13.97
ExhaustiveTQ-b2-rf10 (query)                             341.03       883.77     1_224.80       0.2887          1.1550            1.1707        13.97
ExhaustiveTQ-b2-rf20 (query)                             341.03     1_283.40     1_624.43       0.4020          1.0974            1.0847        13.97
ExhaustiveTQ-b2 (self)                                   341.03     4_171.56     4_512.59       0.4025          1.1135            1.0971        13.97
ExhaustiveTQ-b4-rf0 (query)                              467.02     1_160.21     1_627.22       0.1044          1.5026            1.5346        26.18
ExhaustiveTQ-b4-rf5 (query)                              467.02     1_266.46     1_733.47       0.2294          1.2110            1.2410        26.18
ExhaustiveTQ-b4-rf10 (query)                             467.02     1_395.99     1_863.00       0.2943          1.1499            1.1675        26.18
ExhaustiveTQ-b4-rf20 (query)                             467.02     1_775.00     2_242.02       0.4029          1.0975            1.0929        26.18
ExhaustiveTQ-b4 (self)                                   467.02     5_868.26     6_335.28       0.4038          1.1130            1.1087        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                          659.09       194.45       853.54       0.0844          1.6539            1.5906        14.97
IVF-TQ-b2-nl158-np12-rf0 (query)                         659.09       203.41       862.50       0.0844          1.6539            1.5906        14.97
IVF-TQ-b2-nl158-np17-rf0 (query)                         659.09       235.58       894.67       0.0844          1.6539            1.5906        14.97
IVF-TQ-b2-nl158-np7-rf10 (query)                         659.09       418.51     1_077.60       0.2887          1.1550            1.1707        14.97
IVF-TQ-b2-nl158-np7-rf20 (query)                         659.09       760.28     1_419.37       0.4020          1.0974            1.0847        14.97
IVF-TQ-b2-nl158-np12-rf10 (query)                        659.09       422.03     1_081.12       0.2887          1.1550            1.1707        14.97
IVF-TQ-b2-nl158-np12-rf20 (query)                        659.09       788.13     1_447.22       0.4020          1.0974            1.0847        14.97
IVF-TQ-b2-nl158-np17-rf10 (query)                        659.09       443.94     1_103.02       0.2887          1.1550            1.1707        14.97
IVF-TQ-b2-nl158-np17-rf20 (query)                        659.09       816.39     1_475.48       0.4020          1.0974            1.0847        14.97
IVF-TQ-b2-nl158 (self)                                   659.09     1_431.32     2_090.41       0.4025          1.1135            1.0971        14.97
IVF-TQ-b2-nl223-np11-rf0 (query)                         638.78       201.35       840.13       0.0845          1.6537            1.5905        15.19
IVF-TQ-b2-nl223-np14-rf0 (query)                         638.78       213.78       852.55       0.0844          1.6539            1.5906        15.19
IVF-TQ-b2-nl223-np21-rf0 (query)                         638.78       246.39       885.17       0.0844          1.6539            1.5906        15.19
IVF-TQ-b2-nl223-np11-rf10 (query)                        638.78       410.54     1_049.31       0.2887          1.1550            1.1707        15.19
IVF-TQ-b2-nl223-np11-rf20 (query)                        638.78       717.19     1_355.97       0.4020          1.0974            1.0847        15.19
IVF-TQ-b2-nl223-np14-rf10 (query)                        638.78       431.71     1_070.49       0.2887          1.1550            1.1707        15.19
IVF-TQ-b2-nl223-np14-rf20 (query)                        638.78       754.38     1_393.15       0.4020          1.0974            1.0847        15.19
IVF-TQ-b2-nl223-np21-rf10 (query)                        638.78       456.51     1_095.28       0.2887          1.1550            1.1707        15.19
IVF-TQ-b2-nl223-np21-rf20 (query)                        638.78       786.91     1_425.69       0.4020          1.0974            1.0847        15.19
IVF-TQ-b2-nl223 (self)                                   638.78     1_459.50     2_098.27       0.4025          1.1135            1.0971        15.19
IVF-TQ-b2-nl316-np15-rf0 (query)                         741.02       210.23       951.25       0.0844          1.6539            1.5906        15.58
IVF-TQ-b2-nl316-np17-rf0 (query)                         741.02       216.55       957.57       0.0844          1.6539            1.5906        15.58
IVF-TQ-b2-nl316-np25-rf0 (query)                         741.02       243.62       984.64       0.0844          1.6539            1.5906        15.58
IVF-TQ-b2-nl316-np15-rf10 (query)                        741.02       405.92     1_146.93       0.2887          1.1550            1.1707        15.58
IVF-TQ-b2-nl316-np15-rf20 (query)                        741.02       721.76     1_462.78       0.4020          1.0974            1.0847        15.58
IVF-TQ-b2-nl316-np17-rf10 (query)                        741.02       411.20     1_152.22       0.2887          1.1550            1.1707        15.58
IVF-TQ-b2-nl316-np17-rf20 (query)                        741.02       706.37     1_447.39       0.4020          1.0974            1.0847        15.58
IVF-TQ-b2-nl316-np25-rf10 (query)                        741.02       453.49     1_194.50       0.2887          1.1550            1.1707        15.58
IVF-TQ-b2-nl316-np25-rf20 (query)                        741.02       796.69     1_537.71       0.4020          1.0974            1.0847        15.58
IVF-TQ-b2-nl316 (self)                                   741.02     1_483.57     2_224.58       0.4025          1.1135            1.0971        15.58
IVF-TQ-b4-nl158-np7-rf0 (query)                          763.78       260.47     1_024.25       0.1044          1.5026            1.5346        27.49
IVF-TQ-b4-nl158-np12-rf0 (query)                         763.78       288.00     1_051.78       0.1044          1.5026            1.5346        27.49
IVF-TQ-b4-nl158-np17-rf0 (query)                         763.78       322.61     1_086.39       0.1044          1.5026            1.5346        27.49
IVF-TQ-b4-nl158-np7-rf10 (query)                         763.78       498.59     1_262.37       0.2943          1.1499            1.1675        27.49
IVF-TQ-b4-nl158-np7-rf20 (query)                         763.78       861.20     1_624.98       0.4029          1.0975            1.0929        27.49
IVF-TQ-b4-nl158-np12-rf10 (query)                        763.78       521.21     1_284.99       0.2943          1.1499            1.1675        27.49
IVF-TQ-b4-nl158-np12-rf20 (query)                        763.78       892.88     1_656.66       0.4029          1.0975            1.0929        27.49
IVF-TQ-b4-nl158-np17-rf10 (query)                        763.78       554.90     1_318.68       0.2943          1.1499            1.1675        27.49
IVF-TQ-b4-nl158-np17-rf20 (query)                        763.78       939.71     1_703.49       0.4029          1.0975            1.0929        27.49
IVF-TQ-b4-nl158 (self)                                   763.78     1_626.35     2_390.13       0.4038          1.1130            1.1088        27.49
IVF-TQ-b4-nl223-np11-rf0 (query)                         782.64       284.05     1_066.69       0.1044          1.5026            1.5346        27.81
IVF-TQ-b4-nl223-np14-rf0 (query)                         782.64       303.87     1_086.51       0.1044          1.5026            1.5346        27.81
IVF-TQ-b4-nl223-np21-rf0 (query)                         782.64       356.92     1_139.56       0.1044          1.5026            1.5346        27.81
IVF-TQ-b4-nl223-np11-rf10 (query)                        782.64       511.89     1_294.53       0.2943          1.1499            1.1675        27.81
IVF-TQ-b4-nl223-np11-rf20 (query)                        782.64       811.93     1_594.57       0.4029          1.0975            1.0929        27.81
IVF-TQ-b4-nl223-np14-rf10 (query)                        782.64       538.21     1_320.85       0.2943          1.1499            1.1675        27.81
IVF-TQ-b4-nl223-np14-rf20 (query)                        782.64       844.39     1_627.03       0.4029          1.0975            1.0929        27.81
IVF-TQ-b4-nl223-np21-rf10 (query)                        782.64       594.48     1_377.12       0.2943          1.1499            1.1675        27.81
IVF-TQ-b4-nl223-np21-rf20 (query)                        782.64       931.41     1_714.05       0.4029          1.0975            1.0929        27.81
IVF-TQ-b4-nl223 (self)                                   782.64     1_675.37     2_458.01       0.4038          1.1130            1.1088        27.81
IVF-TQ-b4-nl316-np15-rf0 (query)                         872.23       296.80     1_169.03       0.1045          1.5026            1.5346        28.39
IVF-TQ-b4-nl316-np17-rf0 (query)                         872.23       312.62     1_184.85       0.1044          1.5026            1.5346        28.39
IVF-TQ-b4-nl316-np25-rf0 (query)                         872.23       365.92     1_238.15       0.1044          1.5026            1.5346        28.39
IVF-TQ-b4-nl316-np15-rf10 (query)                        872.23       504.75     1_376.98       0.2943          1.1499            1.1675        28.39
IVF-TQ-b4-nl316-np15-rf20 (query)                        872.23       810.02     1_682.25       0.4029          1.0975            1.0929        28.39
IVF-TQ-b4-nl316-np17-rf10 (query)                        872.23       514.89     1_387.12       0.2943          1.1499            1.1675        28.39
IVF-TQ-b4-nl316-np17-rf20 (query)                        872.23       803.90     1_676.13       0.4029          1.0975            1.0929        28.39
IVF-TQ-b4-nl316-np25-rf10 (query)                        872.23       577.25     1_449.48       0.2943          1.1499            1.1675        28.39
IVF-TQ-b4-nl316-np25-rf20 (query)                        872.23       878.23     1_750.46       0.4029          1.0975            1.0929        28.39
IVF-TQ-b4-nl316 (self)                                   872.23     1_671.76     2_543.99       0.4038          1.1130            1.1087        28.39
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
Exhaustive (query)                                       103.23     2_003.90     2_107.14       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        103.23     6_633.93     6_737.16       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              638.96       973.73     1_612.69       0.0841          1.5107            1.4226        21.33
ExhaustiveTQ-b2-rf5 (query)                              638.96     1_063.20     1_702.15       0.2144          1.1739            1.2056        21.33
ExhaustiveTQ-b2-rf10 (query)                             638.96     1_217.57     1_856.53       0.2770          1.1267            1.1512        21.33
ExhaustiveTQ-b2-rf20 (query)                             638.96     1_639.40     2_278.35       0.3770          1.0843            1.0724        21.33
ExhaustiveTQ-b2 (self)                                   638.96     5_366.33     6_005.29       0.3767          1.0935            1.0803        21.33
ExhaustiveTQ-b4-rf0 (query)                              764.60     1_814.51     2_579.11       0.0986          1.4231            1.4109        39.64
ExhaustiveTQ-b4-rf5 (query)                              764.60     1_907.95     2_672.55       0.2167          1.1746            1.2047        39.64
ExhaustiveTQ-b4-rf10 (query)                             764.60     2_023.04     2_787.64       0.2692          1.1311            1.1557        39.64
ExhaustiveTQ-b4-rf20 (query)                             764.60     2_438.05     3_202.65       0.3605          1.0923            1.1071        39.64
ExhaustiveTQ-b4 (self)                                   764.60     8_018.26     8_782.86       0.3609          1.1024            1.1182        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_009.62       294.24     1_303.86       0.0841          1.5107            1.4226        22.65
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_009.62       309.34     1_318.96       0.0841          1.5107            1.4226        22.65
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_009.62       342.46     1_352.08       0.0841          1.5107            1.4226        22.65
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_009.62       526.98     1_536.60       0.2771          1.1267            1.1512        22.65
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_009.62       886.21     1_895.83       0.3770          1.0843            1.0724        22.65
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_009.62       538.11     1_547.73       0.2771          1.1267            1.1512        22.65
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_009.62       933.65     1_943.27       0.3770          1.0843            1.0724        22.65
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_009.62       566.81     1_576.43       0.2771          1.1267            1.1512        22.65
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_009.62       944.47     1_954.09       0.3770          1.0843            1.0724        22.65
IVF-TQ-b2-nl158 (self)                                 1_009.62     1_870.17     2_879.79       0.3767          1.0935            1.0803        22.65
IVF-TQ-b2-nl223-np11-rf0 (query)                         987.59       302.38     1_289.97       0.0841          1.5107            1.4226        22.97
IVF-TQ-b2-nl223-np14-rf0 (query)                         987.59       319.98     1_307.57       0.0841          1.5107            1.4226        22.97
IVF-TQ-b2-nl223-np21-rf0 (query)                         987.59       351.56     1_339.15       0.0841          1.5107            1.4226        22.97
IVF-TQ-b2-nl223-np11-rf10 (query)                        987.59       554.45     1_542.04       0.2771          1.1267            1.1512        22.97
IVF-TQ-b2-nl223-np11-rf20 (query)                        987.59       868.80     1_856.39       0.3770          1.0843            1.0724        22.97
IVF-TQ-b2-nl223-np14-rf10 (query)                        987.59       560.65     1_548.25       0.2771          1.1267            1.1512        22.97
IVF-TQ-b2-nl223-np14-rf20 (query)                        987.59       899.89     1_887.48       0.3770          1.0843            1.0724        22.97
IVF-TQ-b2-nl223-np21-rf10 (query)                        987.59       600.57     1_588.16       0.2771          1.1267            1.1512        22.97
IVF-TQ-b2-nl223-np21-rf20 (query)                        987.59       967.20     1_954.79       0.3770          1.0843            1.0724        22.97
IVF-TQ-b2-nl223 (self)                                   987.59     1_903.23     2_890.82       0.3767          1.0935            1.0803        22.97
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_112.73       318.19     1_430.92       0.0841          1.5107            1.4226        23.53
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_112.73       327.64     1_440.37       0.0841          1.5107            1.4226        23.53
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_112.73       356.00     1_468.73       0.0841          1.5107            1.4226        23.53
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_112.73       547.23     1_659.96       0.2771          1.1267            1.1512        23.53
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_112.73       859.65     1_972.38       0.3770          1.0843            1.0724        23.53
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_112.73       553.00     1_665.73       0.2770          1.1267            1.1512        23.53
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_112.73       868.29     1_981.02       0.3770          1.0843            1.0724        23.53
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_112.73       593.40     1_706.13       0.2771          1.1267            1.1512        23.53
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_112.73       915.99     2_028.72       0.3770          1.0843            1.0724        23.53
IVF-TQ-b2-nl316 (self)                                 1_112.73     1_938.28     3_051.01       0.3767          1.0935            1.0803        23.53
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_102.63       400.09     1_502.72       0.0986          1.4231            1.4109        41.44
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_102.63       444.10     1_546.73       0.0986          1.4231            1.4109        41.44
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_102.63       490.76     1_593.39       0.0986          1.4231            1.4109        41.44
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_102.63       650.68     1_753.31       0.2692          1.1311            1.1557        41.44
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_102.63     1_023.59     2_126.22       0.3605          1.0923            1.1071        41.44
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_102.63       690.27     1_792.90       0.2692          1.1311            1.1557        41.44
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_102.63     1_071.55     2_174.18       0.3605          1.0923            1.1071        41.44
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_102.63       736.08     1_838.71       0.2692          1.1311            1.1557        41.44
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_102.63     1_123.34     2_225.97       0.3605          1.0923            1.1071        41.44
IVF-TQ-b4-nl158 (self)                                 1_102.63     2_201.61     3_304.24       0.3609          1.1024            1.1182        41.44
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_143.72       440.93     1_584.65       0.0986          1.4231            1.4109        41.90
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_143.72       474.25     1_617.97       0.0986          1.4231            1.4109        41.90
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_143.72       520.52     1_664.24       0.0986          1.4231            1.4109        41.90
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_143.72       692.87     1_836.59       0.2692          1.1311            1.1557        41.90
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_143.72     1_017.17     2_160.89       0.3605          1.0923            1.1071        41.90
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_143.72       712.64     1_856.36       0.2692          1.1311            1.1557        41.90
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_143.72     1_064.87     2_208.59       0.3605          1.0923            1.1071        41.90
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_143.72       776.40     1_920.12       0.2692          1.1311            1.1557        41.90
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_143.72     1_144.16     2_287.88       0.3605          1.0923            1.1071        41.90
IVF-TQ-b4-nl223 (self)                                 1_143.72     2_333.56     3_477.28       0.3609          1.1024            1.1182        41.90
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_282.01       472.18     1_754.19       0.0986          1.4231            1.4109        42.74
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_282.01       479.72     1_761.73       0.0986          1.4231            1.4109        42.74
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_282.01       540.02     1_822.03       0.0986          1.4231            1.4109        42.74
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_282.01       699.65     1_981.66       0.2692          1.1311            1.1557        42.74
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_282.01     1_004.65     2_286.66       0.3605          1.0923            1.1071        42.74
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_282.01       720.67     2_002.68       0.2692          1.1311            1.1557        42.74
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_282.01     1_030.51     2_312.52       0.3605          1.0923            1.1071        42.74
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_282.01       779.12     2_061.13       0.2692          1.1311            1.1558        42.74
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_282.01     1_101.68     2_383.69       0.3605          1.0923            1.1071        42.74
IVF-TQ-b4-nl316 (self)                                 1_282.01     2_332.43     3_614.44       0.3609          1.1024            1.1182        42.74
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
Exhaustive (query)                                        33.68       752.25       785.92       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.68     2_533.10     2_566.77       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              147.34       384.53       531.87       0.7918          1.0898            1.0632         7.12
ExhaustiveTQ-b2-rf5 (query)                              147.34       480.79       628.13       0.9995          1.0000            1.0000         7.12
ExhaustiveTQ-b2-rf10 (query)                             147.34       596.65       743.99       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b2-rf20 (query)                             147.34       979.09     1_126.43       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b2 (self)                                   147.34     3_217.52     3_364.86       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b4-rf0 (query)                              233.87       614.23       848.10       0.8728          1.0322            1.0183        13.22
ExhaustiveTQ-b4-rf5 (query)                              233.87       702.28       936.14       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4-rf10 (query)                             233.87       823.65     1_057.52       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4-rf20 (query)                             233.87     1_196.78     1_430.65       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4 (self)                                   233.87     3_997.26     4_231.13       1.0000          1.0000            1.0000        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                          513.43       131.44       644.87       0.7917          1.0897            1.0634         7.78
IVF-TQ-b2-nl158-np12-rf0 (query)                         513.43       173.54       686.97       0.7918          1.0898            1.0632         7.78
IVF-TQ-b2-nl158-np17-rf0 (query)                         513.43       211.23       724.66       0.7918          1.0898            1.0632         7.78
IVF-TQ-b2-nl158-np7-rf10 (query)                         513.43       335.89       849.32       0.9982          1.0004            1.0000         7.78
IVF-TQ-b2-nl158-np7-rf20 (query)                         513.43       619.40     1_132.83       0.9982          1.0004            1.0000         7.78
IVF-TQ-b2-nl158-np12-rf10 (query)                        513.43       402.68       916.11       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158-np12-rf20 (query)                        513.43       720.80     1_234.22       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158-np17-rf10 (query)                        513.43       455.43       968.85       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158-np17-rf20 (query)                        513.43       800.52     1_313.94       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158 (self)                                   513.43     1_191.99     1_705.42       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl223-np11-rf0 (query)                         592.91       128.46       721.37       0.7919          1.0897            1.0632         7.92
IVF-TQ-b2-nl223-np14-rf0 (query)                         592.91       145.73       738.64       0.7919          1.0897            1.0632         7.92
IVF-TQ-b2-nl223-np21-rf0 (query)                         592.91       181.91       774.82       0.7918          1.0898            1.0632         7.92
IVF-TQ-b2-nl223-np11-rf10 (query)                        592.91       318.29       911.20       0.9995          1.0001            1.0000         7.92
IVF-TQ-b2-nl223-np11-rf20 (query)                        592.91       602.02     1_194.93       0.9995          1.0001            1.0000         7.92
IVF-TQ-b2-nl223-np14-rf10 (query)                        592.91       358.08       950.98       0.9999          1.0000            1.0000         7.92
IVF-TQ-b2-nl223-np14-rf20 (query)                        592.91       640.87     1_233.78       0.9999          1.0000            1.0000         7.92
IVF-TQ-b2-nl223-np21-rf10 (query)                        592.91       403.19       996.10       1.0000          1.0000            1.0000         7.92
IVF-TQ-b2-nl223-np21-rf20 (query)                        592.91       720.58     1_313.49       1.0000          1.0000            1.0000         7.92
IVF-TQ-b2-nl223 (self)                                   592.91     1_058.73     1_651.64       1.0000          1.0000            1.0000         7.92
IVF-TQ-b2-nl316-np15-rf0 (query)                         712.42       129.78       842.21       0.7918          1.0897            1.0632         8.11
IVF-TQ-b2-nl316-np17-rf0 (query)                         712.42       157.76       870.18       0.7918          1.0898            1.0632         8.11
IVF-TQ-b2-nl316-np25-rf0 (query)                         712.42       176.47       888.89       0.7918          1.0898            1.0632         8.11
IVF-TQ-b2-nl316-np15-rf10 (query)                        712.42       309.73     1_022.15       0.9998          1.0000            1.0000         8.11
IVF-TQ-b2-nl316-np15-rf20 (query)                        712.42       583.25     1_295.67       0.9998          1.0000            1.0000         8.11
IVF-TQ-b2-nl316-np17-rf10 (query)                        712.42       321.44     1_033.86       0.9999          1.0000            1.0000         8.11
IVF-TQ-b2-nl316-np17-rf20 (query)                        712.42       603.96     1_316.38       0.9999          1.0000            1.0000         8.11
IVF-TQ-b2-nl316-np25-rf10 (query)                        712.42       366.48     1_078.90       1.0000          1.0000            1.0000         8.11
IVF-TQ-b2-nl316-np25-rf20 (query)                        712.42       672.21     1_384.64       1.0000          1.0000            1.0000         8.11
IVF-TQ-b2-nl316 (self)                                   712.42     1_017.41     1_729.84       1.0000          1.0000            1.0000         8.11
IVF-TQ-b4-nl158-np7-rf0 (query)                          582.85       181.52       764.38       0.8721          1.0325            1.0187        14.01
IVF-TQ-b4-nl158-np12-rf0 (query)                         582.85       256.16       839.01       0.8728          1.0322            1.0183        14.01
IVF-TQ-b4-nl158-np17-rf0 (query)                         582.85       318.55       901.40       0.8728          1.0322            1.0183        14.01
IVF-TQ-b4-nl158-np7-rf10 (query)                         582.85       390.22       973.07       0.9982          1.0004            1.0000        14.01
IVF-TQ-b4-nl158-np7-rf20 (query)                         582.85       679.44     1_262.30       0.9982          1.0004            1.0000        14.01
IVF-TQ-b4-nl158-np12-rf10 (query)                        582.85       485.21     1_068.06       1.0000          1.0000            1.0000        14.01
IVF-TQ-b4-nl158-np12-rf20 (query)                        582.85       814.46     1_397.31       1.0000          1.0000            1.0000        14.01
IVF-TQ-b4-nl158-np17-rf10 (query)                        582.85       559.71     1_142.56       1.0000          1.0000            1.0000        14.01
IVF-TQ-b4-nl158-np17-rf20 (query)                        582.85       922.73     1_505.58       1.0000          1.0000            1.0000        14.01
IVF-TQ-b4-nl158 (self)                                   582.85     1_263.31     1_846.16       1.0000          1.0000            1.0000        14.01
IVF-TQ-b4-nl223-np11-rf0 (query)                         677.88       181.26       859.14       0.8726          1.0323            1.0184        14.23
IVF-TQ-b4-nl223-np14-rf0 (query)                         677.88       206.89       884.78       0.8727          1.0322            1.0183        14.23
IVF-TQ-b4-nl223-np21-rf0 (query)                         677.88       270.73       948.62       0.8728          1.0322            1.0183        14.23
IVF-TQ-b4-nl223-np11-rf10 (query)                        677.88       388.30     1_066.18       0.9995          1.0001            1.0000        14.23
IVF-TQ-b4-nl223-np11-rf20 (query)                        677.88       643.40     1_321.28       0.9995          1.0001            1.0000        14.23
IVF-TQ-b4-nl223-np14-rf10 (query)                        677.88       419.98     1_097.87       0.9999          1.0000            1.0000        14.23
IVF-TQ-b4-nl223-np14-rf20 (query)                        677.88       711.90     1_389.78       0.9999          1.0000            1.0000        14.23
IVF-TQ-b4-nl223-np21-rf10 (query)                        677.88       489.50     1_167.39       1.0000          1.0000            1.0000        14.23
IVF-TQ-b4-nl223-np21-rf20 (query)                        677.88       825.14     1_503.02       1.0000          1.0000            1.0000        14.23
IVF-TQ-b4-nl223 (self)                                   677.88     1_118.27     1_796.15       1.0000          1.0000            1.0000        14.23
IVF-TQ-b4-nl316-np15-rf0 (query)                         803.94       188.42       992.36       0.8727          1.0322            1.0184        14.52
IVF-TQ-b4-nl316-np17-rf0 (query)                         803.94       193.11       997.05       0.8727          1.0322            1.0183        14.52
IVF-TQ-b4-nl316-np25-rf0 (query)                         803.94       252.04     1_055.97       0.8727          1.0322            1.0183        14.52
IVF-TQ-b4-nl316-np15-rf10 (query)                        803.94       363.10     1_167.03       0.9998          1.0000            1.0000        14.52
IVF-TQ-b4-nl316-np15-rf20 (query)                        803.94       636.41     1_440.34       0.9998          1.0000            1.0000        14.52
IVF-TQ-b4-nl316-np17-rf10 (query)                        803.94       381.73     1_185.66       0.9999          1.0000            1.0000        14.52
IVF-TQ-b4-nl316-np17-rf20 (query)                        803.94       653.24     1_457.18       0.9999          1.0000            1.0000        14.52
IVF-TQ-b4-nl316-np25-rf10 (query)                        803.94       445.96     1_249.90       1.0000          1.0000            1.0000        14.52
IVF-TQ-b4-nl316-np25-rf20 (query)                        803.94       751.59     1_555.53       1.0000          1.0000            1.0000        14.52
IVF-TQ-b4-nl316 (self)                                   803.94     1_049.57     1_853.51       1.0000          1.0000            1.0000        14.52
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
Exhaustive (query)                                        70.57     1_373.10     1_443.68       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         70.57     4_619.48     4_690.05       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              357.53       660.28     1_017.81       0.8424          1.0447            1.0331        13.97
ExhaustiveTQ-b2-rf5 (query)                              357.53       764.87     1_122.40       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2-rf10 (query)                             357.53       885.10     1_242.63       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2-rf20 (query)                             357.53     1_292.55     1_650.08       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2 (self)                                   357.53     4_230.85     4_588.38       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b4-rf0 (query)                              476.90     1_168.68     1_645.58       0.8985          1.0191            1.0110        26.18
ExhaustiveTQ-b4-rf5 (query)                              476.90     1_272.23     1_749.13       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4-rf10 (query)                             476.90     1_397.52     1_874.42       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4-rf20 (query)                             476.90     1_787.54     2_264.44       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4 (self)                                   476.90     5_880.17     6_357.07       1.0000          1.0000            1.0000        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                          896.51       232.37     1_128.88       0.8421          1.0449            1.0333        14.97
IVF-TQ-b2-nl158-np12-rf0 (query)                         896.51       304.58     1_201.09       0.8424          1.0447            1.0331        14.97
IVF-TQ-b2-nl158-np17-rf0 (query)                         896.51       360.52     1_257.03       0.8424          1.0447            1.0331        14.97
IVF-TQ-b2-nl158-np7-rf10 (query)                         896.51       474.08     1_370.59       0.9987          1.0003            1.0000        14.97
IVF-TQ-b2-nl158-np7-rf20 (query)                         896.51       758.91     1_655.42       0.9987          1.0003            1.0000        14.97
IVF-TQ-b2-nl158-np12-rf10 (query)                        896.51       543.83     1_440.34       0.9999          1.0000            1.0000        14.97
IVF-TQ-b2-nl158-np12-rf20 (query)                        896.51       882.48     1_778.99       0.9999          1.0000            1.0000        14.97
IVF-TQ-b2-nl158-np17-rf10 (query)                        896.51       612.51     1_509.03       1.0000          1.0000            1.0000        14.97
IVF-TQ-b2-nl158-np17-rf20 (query)                        896.51       964.15     1_860.66       1.0000          1.0000            1.0000        14.97
IVF-TQ-b2-nl158 (self)                                   896.51     1_584.39     2_480.90       1.0000          1.0000            1.0000        14.97
IVF-TQ-b2-nl223-np11-rf0 (query)                         887.31       232.54     1_119.85       0.8423          1.0447            1.0331        15.26
IVF-TQ-b2-nl223-np14-rf0 (query)                         887.31       261.85     1_149.16       0.8424          1.0447            1.0330        15.26
IVF-TQ-b2-nl223-np21-rf0 (query)                         887.31       322.69     1_209.99       0.8424          1.0447            1.0331        15.26
IVF-TQ-b2-nl223-np11-rf10 (query)                        887.31       448.42     1_335.72       0.9997          1.0000            1.0000        15.26
IVF-TQ-b2-nl223-np11-rf20 (query)                        887.31       738.14     1_625.44       0.9997          1.0000            1.0000        15.26
IVF-TQ-b2-nl223-np14-rf10 (query)                        887.31       480.22     1_367.53       0.9999          1.0000            1.0000        15.26
IVF-TQ-b2-nl223-np14-rf20 (query)                        887.31       795.69     1_682.99       0.9999          1.0000            1.0000        15.26
IVF-TQ-b2-nl223-np21-rf10 (query)                        887.31       553.55     1_440.86       1.0000          1.0000            1.0000        15.26
IVF-TQ-b2-nl223-np21-rf20 (query)                        887.31       888.33     1_775.64       1.0000          1.0000            1.0000        15.26
IVF-TQ-b2-nl223 (self)                                   887.31     1_477.96     2_365.26       1.0000          1.0000            1.0000        15.26
IVF-TQ-b2-nl316-np15-rf0 (query)                         977.22       238.32     1_215.54       0.8424          1.0447            1.0331        15.57
IVF-TQ-b2-nl316-np17-rf0 (query)                         977.22       249.93     1_227.14       0.8424          1.0447            1.0331        15.57
IVF-TQ-b2-nl316-np25-rf0 (query)                         977.22       312.91     1_290.13       0.8424          1.0447            1.0331        15.57
IVF-TQ-b2-nl316-np15-rf10 (query)                        977.22       451.22     1_428.43       0.9999          1.0000            1.0000        15.57
IVF-TQ-b2-nl316-np15-rf20 (query)                        977.22       748.23     1_725.44       0.9999          1.0000            1.0000        15.57
IVF-TQ-b2-nl316-np17-rf10 (query)                        977.22       466.36     1_443.58       1.0000          1.0000            1.0000        15.57
IVF-TQ-b2-nl316-np17-rf20 (query)                        977.22       778.29     1_755.50       1.0000          1.0000            1.0000        15.57
IVF-TQ-b2-nl316-np25-rf10 (query)                        977.22       526.04     1_503.26       1.0000          1.0000            1.0000        15.57
IVF-TQ-b2-nl316-np25-rf20 (query)                        977.22       859.37     1_836.58       1.0000          1.0000            1.0000        15.57
IVF-TQ-b2-nl316 (self)                                   977.22     1_433.95     2_411.16       1.0000          1.0000            1.0000        15.57
IVF-TQ-b4-nl158-np7-rf0 (query)                          971.47       339.90     1_311.37       0.8979          1.0193            1.0113        27.48
IVF-TQ-b4-nl158-np12-rf0 (query)                         971.47       474.17     1_445.65       0.8985          1.0191            1.0110        27.48
IVF-TQ-b4-nl158-np17-rf0 (query)                         971.47       577.02     1_548.49       0.8985          1.0191            1.0110        27.48
IVF-TQ-b4-nl158-np7-rf10 (query)                         971.47       573.83     1_545.30       0.9987          1.0003            1.0000        27.48
IVF-TQ-b4-nl158-np7-rf20 (query)                         971.47       884.81     1_856.28       0.9987          1.0003            1.0000        27.48
IVF-TQ-b4-nl158-np12-rf10 (query)                        971.47       719.62     1_691.10       0.9999          1.0000            1.0000        27.48
IVF-TQ-b4-nl158-np12-rf20 (query)                        971.47     1_046.20     2_017.67       0.9999          1.0000            1.0000        27.48
IVF-TQ-b4-nl158-np17-rf10 (query)                        971.47       826.64     1_798.12       1.0000          1.0000            1.0000        27.48
IVF-TQ-b4-nl158-np17-rf20 (query)                        971.47     1_190.00     2_161.48       1.0000          1.0000            1.0000        27.48
IVF-TQ-b4-nl158 (self)                                   971.47     1_916.99     2_888.46       1.0000          1.0000            1.0000        27.48
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_013.47       344.32     1_357.79       0.8984          1.0191            1.0111        27.93
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_013.47       401.48     1_414.95       0.8985          1.0191            1.0110        27.93
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_013.47       515.68     1_529.15       0.8985          1.0191            1.0110        27.93
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_013.47       564.63     1_578.10       0.9997          1.0000            1.0000        27.93
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_013.47       852.89     1_866.36       0.9997          1.0000            1.0000        27.93
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_013.47       615.07     1_628.54       0.9999          1.0000            1.0000        27.93
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_013.47       928.16     1_941.64       0.9999          1.0000            1.0000        27.93
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_013.47       739.32     1_752.80       1.0000          1.0000            1.0000        27.93
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_013.47     1_068.08     2_081.55       1.0000          1.0000            1.0000        27.93
IVF-TQ-b4-nl223 (self)                                 1_013.47     1_759.96     2_773.44       1.0000          1.0000            1.0000        27.93
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_107.68       349.63     1_457.31       0.8985          1.0191            1.0110        28.37
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_107.68       374.15     1_481.83       0.8985          1.0191            1.0110        28.37
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_107.68       476.00     1_583.68       0.8985          1.0191            1.0110        28.37
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_107.68       557.38     1_665.07       0.9999          1.0000            1.0000        28.37
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_107.68       855.19     1_962.87       0.9999          1.0000            1.0000        28.37
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_107.68       585.09     1_692.77       1.0000          1.0000            1.0000        28.37
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_107.68       891.41     1_999.09       1.0000          1.0000            1.0000        28.37
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_107.68       695.81     1_803.49       1.0000          1.0000            1.0000        28.37
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_107.68     1_032.46     2_140.14       1.0000          1.0000            1.0000        28.37
IVF-TQ-b4-nl316 (self)                                 1_107.68     1_690.93     2_798.62       1.0000          1.0000            1.0000        28.37
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
Exhaustive (query)                                       102.14     1_993.53     2_095.67       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        102.14     6_719.35     6_821.49       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              617.50       983.95     1_601.45       0.8736          1.0271            1.0199        21.33
ExhaustiveTQ-b2-rf5 (query)                              617.50     1_076.47     1_693.97       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2-rf10 (query)                             617.50     1_221.27     1_838.77       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2-rf20 (query)                             617.50     1_644.49     2_261.99       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2 (self)                                   617.50     5_386.10     6_003.60       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b4-rf0 (query)                              778.84     1_788.49     2_567.33       0.9097          1.0146            1.0083        39.64
ExhaustiveTQ-b4-rf5 (query)                              778.84     1_896.42     2_675.26       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4-rf10 (query)                             778.84     2_015.44     2_794.28       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4-rf20 (query)                             778.84     2_410.82     3_189.66       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4 (self)                                   778.84     7_965.25     8_744.08       1.0000          1.0000            1.0000        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_414.83       350.82     1_765.65       0.8735          1.0272            1.0201        22.61
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_414.83       451.64     1_866.47       0.8736          1.0271            1.0199        22.61
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_414.83       537.69     1_952.52       0.8736          1.0271            1.0199        22.61
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_414.83       611.88     2_026.71       0.9995          1.0001            1.0000        22.61
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_414.83       938.68     2_353.51       0.9995          1.0001            1.0000        22.61
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_414.83       724.04     2_138.87       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_414.83     1_080.17     2_495.00       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_414.83       814.16     2_228.99       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_414.83     1_183.92     2_598.75       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158 (self)                                 1_414.83     2_079.74     3_494.57       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_530.41       350.35     1_880.76       0.8736          1.0271            1.0200        23.00
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_530.41       387.32     1_917.73       0.8736          1.0271            1.0199        23.00
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_530.41       471.61     2_002.02       0.8736          1.0271            1.0199        23.00
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_530.41       592.85     2_123.26       0.9998          1.0000            1.0000        23.00
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_530.41       915.69     2_446.10       0.9998          1.0000            1.0000        23.00
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_530.41       680.69     2_211.10       1.0000          1.0000            1.0000        23.00
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_530.41       983.14     2_513.55       1.0000          1.0000            1.0000        23.00
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_530.41       735.44     2_265.86       1.0000          1.0000            1.0000        23.00
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_530.41     1_090.54     2_620.95       1.0000          1.0000            1.0000        23.00
IVF-TQ-b2-nl223 (self)                                 1_530.41     2_027.64     3_558.05       1.0000          1.0000            1.0000        23.00
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_799.12       355.09     2_154.21       0.8736          1.0271            1.0200        23.53
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_799.12       374.28     2_173.40       0.8736          1.0271            1.0200        23.53
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_799.12       448.08     2_247.20       0.8736          1.0271            1.0199        23.53
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_799.12       588.06     2_387.18       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_799.12       914.06     2_713.17       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_799.12       613.44     2_412.55       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_799.12       948.05     2_747.17       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_799.12       702.60     2_501.72       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_799.12     1_051.61     2_850.73       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316 (self)                                 1_799.12     1_973.20     3_772.32       1.0000          1.0000            1.0000        23.53
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_537.26       532.96     2_070.22       0.9094          1.0147            1.0084        41.37
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_537.26       712.23     2_249.49       0.9097          1.0146            1.0083        41.37
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_537.26       855.10     2_392.36       0.9097          1.0146            1.0083        41.37
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_537.26       787.03     2_324.29       0.9995          1.0001            1.0000        41.37
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_537.26     1_113.64     2_650.90       0.9995          1.0001            1.0000        41.37
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_537.26       973.33     2_510.59       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_537.26     1_330.53     2_867.79       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_537.26     1_135.06     2_672.32       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_537.26     1_503.11     3_040.37       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158 (self)                                 1_537.26     2_628.00     4_165.26       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_680.12       530.98     2_211.10       0.9096          1.0146            1.0084        41.94
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_680.12       606.49     2_286.61       0.9097          1.0146            1.0083        41.94
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_680.12       775.74     2_455.86       0.9097          1.0146            1.0083        41.94
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_680.12       772.34     2_452.45       0.9998          1.0000            1.0000        41.94
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_680.12     1_183.39     2_863.50       0.9998          1.0000            1.0000        41.94
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_680.12       851.96     2_532.07       1.0000          1.0000            1.0000        41.94
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_680.12     1_179.32     2_859.44       1.0000          1.0000            1.0000        41.94
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_680.12     1_021.12     2_701.24       1.0000          1.0000            1.0000        41.94
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_680.12     1_368.41     3_048.52       1.0000          1.0000            1.0000        41.94
IVF-TQ-b4-nl223 (self)                                 1_680.12     2_506.68     4_186.80       1.0000          1.0000            1.0000        41.94
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_974.63       551.71     2_526.34       0.9097          1.0146            1.0083        42.73
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_974.63       585.02     2_559.65       0.9097          1.0146            1.0083        42.73
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_974.63       734.17     2_708.80       0.9097          1.0146            1.0083        42.73
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_974.63       768.70     2_743.33       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_974.63     1_085.80     3_060.44       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_974.63       814.05     2_788.69       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_974.63     1_140.05     3_114.68       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_974.63       965.01     2_939.64       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_974.63     1_308.80     3_283.43       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316 (self)                                 1_974.63     2_501.35     4_475.99       1.0000          1.0000            1.0000        42.73
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
