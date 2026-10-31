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
Exhaustive (query)                                        32.96       707.15       740.11       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.96     2_303.14     2_336.10       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                 70.56       237.75       308.31       0.1199          1.4617            1.4199         1.78
ExhaustiveBinary-256-random-rf10 (query)                  70.56       359.76       430.32       0.3411          1.0941            1.0814         1.78
ExhaustiveBinary-256-random-rf20 (query)                  70.56       458.26       528.82       0.4467          1.0571            1.0475         1.78
ExhaustiveBinary-256-random (self)                        70.56     1_141.63     1_212.19       0.3454          1.0895            1.0798         1.78
ExhaustiveBinary-256-pca_no_rr (query)                   127.29       244.38       371.67       0.1153          1.4748            1.4212         1.78
ExhaustiveBinary-256-pca-rf10 (query)                    127.29       348.60       475.90       0.3323          1.1029            1.0834         1.78
ExhaustiveBinary-256-pca-rf20 (query)                    127.29       450.88       578.18       0.4387          1.0631            1.0485         1.78
ExhaustiveBinary-256-pca (self)                          127.29     1_120.84     1_248.13       0.3391          1.0957            1.0813         1.78
ExhaustiveBinary-512-random_no_rr (query)                103.27       361.64       464.91       0.1588          1.3547            1.3300         3.55
ExhaustiveBinary-512-random-rf10 (query)                 103.27       471.51       574.79       0.3786          1.0692            1.0677         3.55
ExhaustiveBinary-512-random-rf20 (query)                 103.27       590.44       693.72       0.4874          1.0424            1.0395         3.55
ExhaustiveBinary-512-random (self)                       103.27     1_538.45     1_641.73       0.3805          1.0675            1.0675         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   141.81       356.56       498.37       0.1564          1.3535            1.3265         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    141.81       472.32       614.12       0.3789          1.0710            1.0663         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    141.81       584.00       725.81       0.4903          1.0433            1.0387         3.55
ExhaustiveBinary-512-pca (self)                          141.81     1_613.78     1_755.59       0.3823          1.0678            1.0665         3.55
ExhaustiveBinary-1024-random_no_rr (query)               138.33       510.32       648.64       0.1929          1.2764            1.2696         7.10
ExhaustiveBinary-1024-random-rf10 (query)                138.33       635.80       774.13       0.4214          1.0550            1.0552         7.10
ExhaustiveBinary-1024-random-rf20 (query)                138.33       746.08       884.41       0.5434          1.0327            1.0308         7.10
ExhaustiveBinary-1024-random (self)                      138.33     2_096.14     2_234.46       0.4232          1.0547            1.0552         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  170.02       519.19       689.20       0.1921          1.2733            1.2652         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   170.02       646.39       816.41       0.4226          1.0546            1.0544         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   170.02       753.04       923.06       0.5443          1.0326            1.0305         7.10
ExhaustiveBinary-1024-pca (self)                         170.02     2_116.02     2_286.04       0.4236          1.0546            1.0548         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   69.69       427.76       497.45       0.1211          1.4987            1.4523         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    69.69       462.16       531.85       0.3284          1.1039            1.0884         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    69.69       696.37       766.06       0.4385          1.0624            1.0494         1.53
ExhaustiveBinary-256-sign (self)                          69.69     1_504.21     1_573.90       0.3334          1.0988            1.0859         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)              320.46        52.26       372.72       0.1245          1.4365            1.4004         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)             320.46        51.32       371.78       0.1245          1.4365            1.4004         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)             320.46        57.05       377.51       0.1245          1.4365            1.4004         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)             320.46        99.80       420.26       0.3492          1.0889            1.0776         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)             320.46       153.24       473.70       0.4555          1.0543            1.0456         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)            320.46       107.97       428.43       0.3492          1.0889            1.0776         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)            320.46       156.68       477.14       0.4555          1.0543            1.0456         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)            320.46        98.79       419.25       0.3492          1.0889            1.0776         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)            320.46       152.88       473.34       0.4555          1.0543            1.0456         1.93
IVF-Binary-256-nl158-random (self)                       320.46       212.05       532.51       0.3542          1.0841            1.0761         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             302.64        45.79       348.43       0.1414          1.3598            1.3149         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             302.64        47.63       350.27       0.1414          1.3603            1.3152         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             302.64        51.32       353.96       0.1414          1.3604            1.3152         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            302.64        98.61       401.24       0.3893          1.0691            1.0625         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            302.64       150.75       453.39       0.4974          1.0428            1.0371         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            302.64        98.99       401.62       0.3890          1.0692            1.0625         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            302.64       153.53       456.16       0.4969          1.0429            1.0372         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            302.64       101.70       404.33       0.3890          1.0692            1.0625         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            302.64       156.15       458.79       0.4969          1.0429            1.0372         2.00
IVF-Binary-256-nl223-random (self)                       302.64       220.57       523.21       0.3942          1.0643            1.0614         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             377.90        48.73       426.63       0.1498          1.3359            1.2910         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             377.90        49.97       427.87       0.1498          1.3363            1.2912         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             377.90        53.32       431.22       0.1498          1.3364            1.2912         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            377.90       104.92       482.82       0.4016          1.0646            1.0586         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            377.90       153.91       531.81       0.5055          1.0413            1.0359         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            377.90       100.90       478.80       0.4015          1.0647            1.0586         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            377.90       153.30       531.20       0.5053          1.0413            1.0360         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            377.90       109.15       487.06       0.4015          1.0647            1.0586         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            377.90       158.03       535.93       0.5053          1.0413            1.0360         2.09
IVF-Binary-256-nl316-random (self)                       377.90       228.04       605.94       0.4062          1.0604            1.0577         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)                 315.81        41.23       357.04       0.1201          1.4453            1.4029         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)                315.81        43.92       359.74       0.1201          1.4453            1.4029         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)                315.81        45.87       361.68       0.1201          1.4453            1.4029         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)                315.81        93.23       409.05       0.3431          1.0959            1.0788         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)                315.81       146.56       462.37       0.4521          1.0584            1.0457         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)               315.81        93.83       409.64       0.3431          1.0959            1.0788         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)               315.81       148.84       464.65       0.4521          1.0584            1.0457         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)               315.81        97.37       413.19       0.3431          1.0959            1.0788         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)               315.81       150.43       466.25       0.4521          1.0584            1.0457         1.93
IVF-Binary-256-nl158-pca (self)                          315.81       207.34       523.16       0.3495          1.0891            1.0771         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                303.68        45.41       349.09       0.1378          1.3708            1.3186         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                303.68        46.80       350.49       0.1377          1.3712            1.3187         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                303.68        49.92       353.60       0.1377          1.3712            1.3187         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               303.68        97.39       401.07       0.3827          1.0753            1.0638         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               303.68       150.77       454.45       0.4957          1.0457            1.0375         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               303.68        98.01       401.69       0.3826          1.0754            1.0638         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               303.68       151.30       454.98       0.4955          1.0457            1.0375         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               303.68       101.13       404.81       0.3826          1.0754            1.0638         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               303.68       152.95       456.63       0.4955          1.0457            1.0375         2.00
IVF-Binary-256-nl223-pca (self)                          303.68       216.11       519.79       0.3896          1.0691            1.0621         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                371.83        51.89       423.72       0.1466          1.3422            1.2923         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                371.83        48.65       420.47       0.1466          1.3426            1.2924         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                371.83        54.27       426.09       0.1466          1.3427            1.2924         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               371.83       104.66       476.49       0.3964          1.0705            1.0596         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               371.83       151.83       523.66       0.5067          1.0437            1.0356         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               371.83       106.89       478.71       0.3963          1.0705            1.0596         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               371.83       152.17       524.00       0.5066          1.0437            1.0356         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               371.83       102.80       474.62       0.3963          1.0705            1.0596         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               371.83       159.52       531.35       0.5066          1.0437            1.0356         2.09
IVF-Binary-256-nl316-pca (self)                          371.83       233.51       605.34       0.4023          1.0647            1.0583         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)              276.93        59.07       335.99       0.1614          1.3415            1.3204         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)             276.93        62.89       339.82       0.1614          1.3415            1.3204         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)             276.93        65.16       342.08       0.1614          1.3415            1.3204         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)             276.93       117.47       394.39       0.3837          1.0670            1.0659         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)             276.93       173.71       450.64       0.4941          1.0409            1.0384         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)            276.93       125.57       402.50       0.3837          1.0670            1.0659         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)            276.93       176.98       453.91       0.4941          1.0409            1.0384         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)            276.93       123.92       400.85       0.3837          1.0670            1.0659         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)            276.93       179.91       456.84       0.4941          1.0409            1.0384         3.71
IVF-Binary-512-nl158-random (self)                       276.93       297.30       574.23       0.3859          1.0652            1.0657         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             302.33        63.91       366.25       0.1711          1.2997            1.2781         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             302.33        66.23       368.56       0.1711          1.3000            1.2783         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             302.33        71.42       373.75       0.1711          1.3000            1.2783         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            302.33       119.57       421.90       0.4018          1.0605            1.0593         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            302.33       173.31       475.64       0.5143          1.0372            1.0348         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            302.33       122.87       425.20       0.4016          1.0605            1.0593         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            302.33       177.77       480.11       0.5140          1.0373            1.0348         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            302.33       125.66       427.99       0.4016          1.0605            1.0593         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            302.33       183.75       486.08       0.5140          1.0373            1.0348         3.77
IVF-Binary-512-nl223-random (self)                       302.33       301.94       604.27       0.4038          1.0592            1.0592         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             366.58        68.89       435.48       0.1755          1.2882            1.2668         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             366.58        71.29       437.87       0.1755          1.2884            1.2671         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             366.58        72.16       438.74       0.1755          1.2884            1.2671         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            366.58       124.05       490.63       0.4061          1.0594            1.0581         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            366.58       176.36       542.94       0.5177          1.0368            1.0341         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            366.58       124.12       490.70       0.4060          1.0595            1.0581         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            366.58       181.58       548.16       0.5174          1.0368            1.0341         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            366.58       127.14       493.72       0.4060          1.0595            1.0581         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            366.58       182.68       549.26       0.5174          1.0368            1.0341         3.86
IVF-Binary-512-nl316-random (self)                       366.58       317.09       683.68       0.4089          1.0580            1.0579         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)                 316.53        60.35       376.88       0.1596          1.3401            1.3152         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)                316.53        64.52       381.06       0.1596          1.3401            1.3152         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)                316.53        76.94       393.47       0.1596          1.3401            1.3152         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)                316.53       116.69       433.22       0.3846          1.0687            1.0645         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)                316.53       172.62       489.15       0.4979          1.0417            1.0374         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)               316.53       119.77       436.30       0.3846          1.0687            1.0645         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)               316.53       178.75       495.28       0.4979          1.0417            1.0374         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)               316.53       123.06       439.60       0.3846          1.0687            1.0645         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)               316.53       179.45       495.98       0.4979          1.0417            1.0374         3.71
IVF-Binary-512-nl158-pca (self)                          316.53       308.29       624.83       0.3875          1.0659            1.0647         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                345.42        66.03       411.44       0.1695          1.3031            1.2743         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                345.42        66.41       411.82       0.1695          1.3032            1.2744         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                345.42        70.66       416.07       0.1695          1.3032            1.2744         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               345.42       127.47       472.88       0.4038          1.0616            1.0582         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               345.42       178.54       523.95       0.5168          1.0382            1.0342         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               345.42       124.98       470.39       0.4037          1.0617            1.0583         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               345.42       185.42       530.84       0.5167          1.0382            1.0342         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               345.42       127.82       473.24       0.4037          1.0617            1.0583         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               345.42       180.53       525.95       0.5167          1.0382            1.0342         3.77
IVF-Binary-512-nl223-pca (self)                          345.42       301.23       646.65       0.4060          1.0597            1.0583         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                407.93        66.86       474.79       0.1734          1.2924            1.2651         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                407.93        67.68       475.61       0.1734          1.2925            1.2652         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                407.93        71.82       479.75       0.1734          1.2925            1.2652         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               407.93       122.61       530.54       0.4092          1.0604            1.0564         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               407.93       175.67       583.60       0.5223          1.0373            1.0334         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               407.93       122.39       530.32       0.4091          1.0604            1.0564         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               407.93       185.32       593.24       0.5223          1.0373            1.0334         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               407.93       128.81       536.74       0.4091          1.0604            1.0564         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               407.93       181.25       589.18       0.5223          1.0373            1.0334         3.86
IVF-Binary-512-nl316-pca (self)                          407.93       307.68       715.61       0.4117          1.0584            1.0567         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)             310.53        94.36       404.89       0.1942          1.2715            1.2650         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)            310.53        96.06       406.59       0.1942          1.2715            1.2650         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)            310.53       100.48       411.01       0.1942          1.2715            1.2650         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)            310.53       155.58       466.11       0.4245          1.0541            1.0543         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)            310.53       214.26       524.79       0.5468          1.0322            1.0302         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)           310.53       165.22       475.75       0.4245          1.0541            1.0543         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)           310.53       216.60       527.13       0.5468          1.0322            1.0302         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)           310.53       163.69       474.22       0.4245          1.0541            1.0543         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)           310.53       221.01       531.54       0.5468          1.0322            1.0302         7.26
IVF-Binary-1024-nl158-random (self)                      310.53       432.49       743.02       0.4263          1.0539            1.0544         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            326.90        99.03       425.93       0.1973          1.2556            1.2486         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            326.90        99.59       426.49       0.1972          1.2558            1.2487         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            326.90       105.62       432.52       0.1972          1.2558            1.2487         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           326.90       158.46       485.35       0.4343          1.0516            1.0515         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           326.90       212.77       539.66       0.5563          1.0309            1.0289         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           326.90       160.83       487.73       0.4341          1.0517            1.0516         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           326.90       216.15       543.04       0.5561          1.0309            1.0289         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           326.90       170.97       497.86       0.4341          1.0517            1.0516         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           326.90       222.53       549.42       0.5561          1.0309            1.0289         7.32
IVF-Binary-1024-nl223-random (self)                      326.90       436.38       763.27       0.4353          1.0515            1.0518         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            394.60       101.32       495.92       0.1988          1.2510            1.2444         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)            394.60       107.22       501.82       0.1988          1.2512            1.2445         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)            394.60       107.33       501.93       0.1988          1.2512            1.2445         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)           394.60       158.82       553.42       0.4364          1.0511            1.0510         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)           394.60       218.63       613.23       0.5577          1.0307            1.0286         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)           394.60       160.93       555.53       0.4363          1.0511            1.0510         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)           394.60       216.50       611.10       0.5576          1.0307            1.0287         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)           394.60       165.41       560.01       0.4363          1.0511            1.0510         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)           394.60       224.27       618.87       0.5576          1.0307            1.0287         7.41
IVF-Binary-1024-nl316-random (self)                      394.60       447.84       842.44       0.4380          1.0509            1.0513         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)                375.79        92.40       468.20       0.1934          1.2687            1.2608         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)               375.79        97.33       473.12       0.1934          1.2687            1.2608         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)               375.79       112.85       488.65       0.1934          1.2687            1.2608         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)               375.79       167.82       543.62       0.4258          1.0537            1.0535         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)               375.79       213.24       589.03       0.5482          1.0320            1.0299         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)              375.79       158.32       534.11       0.4258          1.0537            1.0535         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)              375.79       220.51       596.30       0.5482          1.0320            1.0299         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)              375.79       161.89       537.68       0.4258          1.0537            1.0535         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)              375.79       230.13       605.92       0.5482          1.0320            1.0299         7.26
IVF-Binary-1024-nl158-pca (self)                         375.79       438.22       814.01       0.4266          1.0537            1.0540         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               396.69        97.16       493.85       0.1974          1.2517            1.2440         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               396.69       100.54       497.23       0.1974          1.2518            1.2440         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               396.69       107.63       504.32       0.1974          1.2518            1.2440         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              396.69       158.68       555.37       0.4357          1.0510            1.0507         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              396.69       214.54       611.23       0.5589          1.0305            1.0281         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              396.69       160.55       557.24       0.4356          1.0510            1.0507         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              396.69       214.95       611.64       0.5587          1.0305            1.0282         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              396.69       166.19       562.88       0.4356          1.0510            1.0507         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              396.69       224.87       621.57       0.5587          1.0305            1.0282         7.32
IVF-Binary-1024-nl223-pca (self)                         396.69       446.09       842.78       0.4366          1.0510            1.0512         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               456.25       101.24       557.49       0.1988          1.2476            1.2400         7.41
IVF-Binary-1024-nl316-np17-rf0-pca (query)               456.25       101.98       558.23       0.1988          1.2477            1.2401         7.41
IVF-Binary-1024-nl316-np25-rf0-pca (query)               456.25       107.48       563.73       0.1988          1.2477            1.2401         7.41
IVF-Binary-1024-nl316-np15-rf10-pca (query)              456.25       158.89       615.14       0.4389          1.0502            1.0501         7.41
IVF-Binary-1024-nl316-np15-rf20-pca (query)              456.25       216.54       672.79       0.5614          1.0302            1.0279         7.41
IVF-Binary-1024-nl316-np17-rf10-pca (query)              456.25       159.54       615.79       0.4389          1.0503            1.0501         7.41
IVF-Binary-1024-nl316-np17-rf20-pca (query)              456.25       216.92       673.17       0.5613          1.0302            1.0279         7.41
IVF-Binary-1024-nl316-np25-rf10-pca (query)              456.25       165.52       621.76       0.4389          1.0503            1.0501         7.41
IVF-Binary-1024-nl316-np25-rf20-pca (query)              456.25       222.52       678.76       0.5613          1.0302            1.0279         7.41
IVF-Binary-1024-nl316-pca (self)                         456.25       474.22       930.47       0.4398          1.0503            1.0504         7.41
IVF-Binary-256-nl158-np7-rf0-sign (query)                271.47       131.09       402.56       0.1213          1.4946            1.4409         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               271.47       133.03       404.50       0.1213          1.4946            1.4409         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               271.47       136.20       407.68       0.1213          1.4946            1.4409         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               271.47       166.85       438.32       0.3330          1.1010            1.0848         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               271.47       302.33       573.80       0.4412          1.0614            1.0485         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              271.47       171.33       442.80       0.3330          1.1010            1.0848         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              271.47       302.57       574.04       0.4412          1.0614            1.0485         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              271.47       172.18       443.65       0.3330          1.1010            1.0848         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              271.47       300.11       571.58       0.4412          1.0614            1.0485         1.68
IVF-Binary-256-nl158-sign (self)                         271.47       439.05       710.52       0.3381          1.0958            1.0832         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               256.44       132.77       389.22       0.1226          1.4837            1.4298         1.74
IVF-Binary-256-nl223-np14-rf0-sign (query)               256.44       133.86       390.30       0.1226          1.4850            1.4303         1.74
IVF-Binary-256-nl223-np21-rf0-sign (query)               256.44       136.43       392.87       0.1225          1.4851            1.4303         1.74
IVF-Binary-256-nl223-np11-rf10-sign (query)              256.44       171.53       427.97       0.3562          1.0888            1.0753         1.74
IVF-Binary-256-nl223-np11-rf20-sign (query)              256.44       296.79       553.23       0.4587          1.0559            1.0444         1.74
IVF-Binary-256-nl223-np14-rf10-sign (query)              256.44       171.94       428.38       0.3560          1.0890            1.0754         1.74
IVF-Binary-256-nl223-np14-rf20-sign (query)              256.44       297.98       554.42       0.4584          1.0560            1.0444         1.74
IVF-Binary-256-nl223-np21-rf10-sign (query)              256.44       175.32       431.76       0.3560          1.0890            1.0754         1.74
IVF-Binary-256-nl223-np21-rf20-sign (query)              256.44       299.50       555.94       0.4584          1.0560            1.0444         1.74
IVF-Binary-256-nl223-sign (self)                         256.44       447.05       703.49       0.3612          1.0839            1.0738         1.74
IVF-Binary-256-nl316-np15-rf0-sign (query)               319.38       134.29       453.67       0.1234          1.4702            1.4163         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               319.38       135.06       454.44       0.1234          1.4713            1.4163         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               319.38       139.39       458.77       0.1234          1.4715            1.4163         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              319.38       174.61       493.99       0.3598          1.0873            1.0740         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              319.38       299.42       618.80       0.4585          1.0560            1.0445         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              319.38       174.30       493.68       0.3597          1.0875            1.0741         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              319.38       308.17       627.55       0.4582          1.0561            1.0445         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              319.38       179.93       499.31       0.3597          1.0875            1.0741         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              319.38       304.22       623.60       0.4582          1.0561            1.0445         1.84
IVF-Binary-256-nl316-sign (self)                         319.38       456.50       775.88       0.3649          1.0824            1.0722         1.84
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
Exhaustive (query)                                        68.04     1_391.70     1_459.75       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.04     4_746.25     4_814.30       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                134.48       272.72       407.19       0.1109          1.3512            1.3092         2.03
ExhaustiveBinary-256-random-rf10 (query)                 134.48       398.69       533.16       0.3143          1.0825            1.0613         2.03
ExhaustiveBinary-256-random-rf20 (query)                 134.48       535.46       669.93       0.4100          1.0523            1.0368         2.03
ExhaustiveBinary-256-random (self)                       134.48     1_247.59     1_382.07       0.3161          1.0784            1.0600         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   234.36       280.22       514.58       0.1167          1.3480            1.2981         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    234.36       394.06       628.42       0.3159          1.0791            1.0596         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    234.36       549.67       784.02       0.4121          1.0505            1.0362         2.03
ExhaustiveBinary-256-pca (self)                          234.36     1_252.63     1_486.99       0.3171          1.0782            1.0588         2.03
ExhaustiveBinary-512-random_no_rr (query)                234.50       397.40       631.90       0.1528          1.2601            1.2299         4.05
ExhaustiveBinary-512-random-rf10 (query)                 234.50       534.71       769.21       0.3465          1.0565            1.0514         4.05
ExhaustiveBinary-512-random-rf20 (query)                 234.50       672.82       907.32       0.4452          1.0358            1.0312         4.05
ExhaustiveBinary-512-random (self)                       234.50     1_734.30     1_968.80       0.3476          1.0547            1.0512         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   300.78       395.06       695.84       0.1558          1.2535            1.2254         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    300.78       528.81       829.60       0.3512          1.0523            1.0507         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    300.78       667.12       967.91       0.4484          1.0329            1.0309         4.05
ExhaustiveBinary-512-pca (self)                          300.78     1_742.53     2_043.31       0.3515          1.0522            1.0505         4.05
ExhaustiveBinary-1024-random_no_rr (query)               262.60       597.35       859.96       0.1816          1.2043            1.1936         8.11
ExhaustiveBinary-1024-random-rf10 (query)                262.60       747.74     1_010.34       0.3747          1.0447            1.0452         8.11
ExhaustiveBinary-1024-random-rf20 (query)                262.60       895.45     1_158.06       0.4789          1.0282            1.0270         8.11
ExhaustiveBinary-1024-random (self)                      262.60     2_446.77     2_709.37       0.3754          1.0447            1.0451         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  355.39       593.11       948.50       0.1832          1.2013            1.1905         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   355.39       742.90     1_098.29       0.3798          1.0434            1.0443         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   355.39       885.40     1_240.79       0.4867          1.0272            1.0261         8.11
ExhaustiveBinary-1024-pca (self)                         355.39     2_409.52     2_764.92       0.3787          1.0436            1.0444         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   85.04       716.31       801.35       0.1518          1.2701            1.2528         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    85.04       725.31       810.35       0.3399          1.0607            1.0535         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    85.04     1_091.07     1_176.11       0.4406          1.0369            1.0319         3.05
ExhaustiveBinary-512-sign (self)                          85.04     2_315.03     2_400.07       0.3409          1.0595            1.0531         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)              414.83        77.91       492.74       0.1137          1.3389            1.3036         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)             414.83        81.25       496.08       0.1137          1.3389            1.3036         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)             414.83        79.57       494.41       0.1137          1.3389            1.3036         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)             414.83       163.82       578.65       0.3176          1.0801            1.0600         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)             414.83       254.27       669.10       0.4134          1.0507            1.0360         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)            414.83       158.17       573.00       0.3176          1.0801            1.0600         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)            414.83       252.68       667.51       0.4134          1.0507            1.0360         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)            414.83       155.82       570.65       0.3176          1.0801            1.0600         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)            414.83       251.05       665.88       0.4134          1.0507            1.0360         2.34
IVF-Binary-256-nl158-random (self)                       414.83       294.78       709.61       0.3195          1.0756            1.0588         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)             414.02        72.57       486.59       0.1323          1.2781            1.2367         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)             414.02        77.03       491.04       0.1322          1.2784            1.2369         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)             414.02        77.52       491.53       0.1322          1.2785            1.2369         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)            414.02       163.26       577.27       0.3685          1.0557            1.0457         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)            414.02       256.37       670.39       0.4682          1.0358            1.0279         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)            414.02       161.90       575.92       0.3683          1.0558            1.0457         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)            414.02       257.17       671.19       0.4678          1.0359            1.0280         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)            414.02       165.23       579.24       0.3683          1.0558            1.0457         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)            414.02       258.21       672.22       0.4678          1.0359            1.0280         2.46
IVF-Binary-256-nl223-random (self)                       414.02       319.23       733.25       0.3700          1.0518            1.0451         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)             504.56        79.21       583.77       0.1436          1.2537            1.2119         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)             504.56        81.15       585.71       0.1436          1.2539            1.2120         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)             504.56        83.71       588.27       0.1435          1.2545            1.2124         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)            504.56       170.47       675.04       0.3831          1.0510            1.0421         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)            504.56       260.14       764.70       0.4824          1.0341            1.0261         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)            504.56       167.01       671.57       0.3828          1.0511            1.0421         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)            504.56       261.00       765.57       0.4817          1.0343            1.0262         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)            504.56       169.64       674.21       0.3827          1.0511            1.0422         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)            504.56       263.28       767.84       0.4816          1.0343            1.0262         2.65
IVF-Binary-256-nl316-random (self)                       504.56       341.79       846.35       0.3847          1.0473            1.0415         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)                 506.31        70.21       576.52       0.1193          1.3358            1.2928         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)                506.31        69.11       575.42       0.1193          1.3358            1.2928         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)                506.31        73.04       579.35       0.1193          1.3358            1.2928         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)                506.31       151.62       657.93       0.3193          1.0758            1.0585         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)                506.31       246.65       752.96       0.4156          1.0484            1.0355         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)               506.31       154.54       660.85       0.3193          1.0758            1.0585         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)               506.31       252.44       758.75       0.4156          1.0484            1.0355         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)               506.31       153.50       659.81       0.3193          1.0758            1.0585         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)               506.31       249.34       755.65       0.4156          1.0484            1.0355         2.34
IVF-Binary-256-nl158-pca (self)                          506.31       303.94       810.25       0.3209          1.0742            1.0576         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)                510.63        72.44       583.07       0.1362          1.2797            1.2334         2.46
IVF-Binary-256-nl223-np14-rf0-pca (query)                510.63        74.43       585.06       0.1361          1.2801            1.2335         2.46
IVF-Binary-256-nl223-np21-rf0-pca (query)                510.63        77.39       588.02       0.1361          1.2801            1.2335         2.46
IVF-Binary-256-nl223-np11-rf10-pca (query)               510.63       161.78       672.40       0.3632          1.0554            1.0461         2.46
IVF-Binary-256-nl223-np11-rf20-pca (query)               510.63       254.72       765.35       0.4637          1.0358            1.0283         2.46
IVF-Binary-256-nl223-np14-rf10-pca (query)               510.63       160.26       670.89       0.3631          1.0555            1.0461         2.46
IVF-Binary-256-nl223-np14-rf20-pca (query)               510.63       257.59       768.22       0.4635          1.0358            1.0283         2.46
IVF-Binary-256-nl223-np21-rf10-pca (query)               510.63       162.91       673.54       0.3630          1.0555            1.0461         2.46
IVF-Binary-256-nl223-np21-rf20-pca (query)               510.63       257.57       768.20       0.4635          1.0358            1.0283         2.46
IVF-Binary-256-nl223-pca (self)                          510.63       320.18       830.81       0.3655          1.0537            1.0454         2.46
IVF-Binary-256-nl316-np15-rf0-pca (query)                601.91        79.24       681.14       0.1455          1.2599            1.2105         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)                601.91        78.98       680.89       0.1454          1.2600            1.2106         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)                601.91        82.55       684.46       0.1454          1.2604            1.2107         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)               601.91       169.78       771.68       0.3774          1.0508            1.0427         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)               601.91       260.02       861.93       0.4767          1.0333            1.0267         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)               601.91       165.76       767.67       0.3772          1.0509            1.0427         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)               601.91       262.92       864.83       0.4762          1.0334            1.0268         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)               601.91       168.22       770.13       0.3771          1.0509            1.0427         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)               601.91       263.35       865.26       0.4761          1.0334            1.0268         2.65
IVF-Binary-256-nl316-pca (self)                          601.91       343.72       945.63       0.3791          1.0493            1.0422         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)              491.40        94.55       585.94       0.1542          1.2560            1.2276         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)             491.40        98.09       589.48       0.1542          1.2560            1.2276         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)             491.40       100.07       591.47       0.1542          1.2560            1.2276         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)             491.40       187.46       678.86       0.3476          1.0561            1.0510         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)             491.40       285.10       776.50       0.4464          1.0356            1.0310         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)            491.40       188.21       679.60       0.3476          1.0561            1.0510         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)            491.40       282.65       774.05       0.4464          1.0356            1.0310         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)            491.40       186.64       678.04       0.3476          1.0561            1.0510         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)            491.40       288.58       779.97       0.4464          1.0356            1.0310         4.36
IVF-Binary-512-nl158-random (self)                       491.40       427.78       919.17       0.3486          1.0543            1.0508         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)             513.43        99.64       613.07       0.1643          1.2246            1.1980         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)             513.43       101.39       614.82       0.1642          1.2249            1.1983         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)             513.43       105.77       619.20       0.1642          1.2249            1.1983         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)            513.43       191.45       704.87       0.3692          1.0480            1.0450         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)            513.43       288.78       802.20       0.4686          1.0313            1.0278         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)            513.43       190.50       703.93       0.3689          1.0481            1.0451         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)            513.43       289.70       803.13       0.4680          1.0314            1.0279         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)            513.43       196.11       709.53       0.3689          1.0481            1.0451         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)            513.43       292.75       806.18       0.4680          1.0314            1.0279         4.49
IVF-Binary-512-nl223-random (self)                       513.43       444.96       958.38       0.3698          1.0470            1.0451         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)             586.86       106.49       693.36       0.1680          1.2145            1.1880         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)             586.86       106.79       693.65       0.1679          1.2148            1.1884         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)             586.86       111.55       698.41       0.1679          1.2152            1.1886         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)            586.86       195.58       782.44       0.3751          1.0467            1.0439         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)            586.86       293.78       880.64       0.4752          1.0304            1.0271         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)            586.86       195.02       781.88       0.3748          1.0468            1.0440         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)            586.86       293.60       880.46       0.4744          1.0305            1.0272         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)            586.86       201.01       787.87       0.3748          1.0468            1.0440         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)            586.86       309.61       896.47       0.4742          1.0305            1.0273         4.67
IVF-Binary-512-nl316-random (self)                       586.86       466.16     1_053.03       0.3755          1.0457            1.0439         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)                 579.18        92.77       671.96       0.1572          1.2489            1.2229         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)                579.18       102.79       681.98       0.1572          1.2489            1.2229         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)                579.18        98.76       677.94       0.1572          1.2489            1.2229         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)                579.18       188.27       767.45       0.3528          1.0515            1.0501         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)                579.18       284.84       864.02       0.4499          1.0326            1.0306         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)               579.18       184.30       763.48       0.3528          1.0515            1.0501         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)               579.18       283.23       862.42       0.4499          1.0326            1.0306         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)               579.18       189.61       768.79       0.3528          1.0515            1.0501         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)               579.18       286.91       866.09       0.4499          1.0326            1.0306         4.36
IVF-Binary-512-nl158-pca (self)                          579.18       447.00     1_026.18       0.3532          1.0513            1.0499         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)                614.81        99.75       714.56       0.1671          1.2191            1.1940         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)                614.81       102.08       716.89       0.1670          1.2194            1.1943         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)                614.81       109.25       724.05       0.1670          1.2194            1.1943         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)               614.81       199.40       814.20       0.3707          1.0459            1.0453         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)               614.81       285.62       900.43       0.4720          1.0291            1.0275         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)               614.81       193.66       808.47       0.3704          1.0460            1.0453         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)               614.81       289.35       904.16       0.4714          1.0292            1.0276         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)               614.81       196.31       811.12       0.3704          1.0460            1.0453         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)               614.81       294.50       909.31       0.4714          1.0292            1.0276         4.49
IVF-Binary-512-nl223-pca (self)                          614.81       450.10     1_064.91       0.3709          1.0458            1.0452         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)                727.69       107.14       834.83       0.1707          1.2099            1.1841         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)                727.69       107.48       835.17       0.1706          1.2101            1.1844         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)                727.69       110.80       838.49       0.1706          1.2102            1.1846         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)               727.69       195.47       923.16       0.3771          1.0444            1.0436         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)               727.69       294.05     1_021.73       0.4785          1.0284            1.0267         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)               727.69       195.13       922.82       0.3768          1.0445            1.0437         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)               727.69       292.83     1_020.52       0.4777          1.0285            1.0267         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)               727.69       205.53       933.22       0.3767          1.0445            1.0437         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)               727.69       299.14     1_026.83       0.4775          1.0286            1.0268         4.67
IVF-Binary-512-nl316-pca (self)                          727.69       467.70     1_195.38       0.3771          1.0445            1.0438         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)             595.15       143.82       738.97       0.1822          1.2030            1.1925         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)            595.15       149.64       744.79       0.1822          1.2030            1.1925         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)            595.15       152.20       747.35       0.1822          1.2030            1.1925         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)            595.15       237.41       832.56       0.3753          1.0446            1.0450         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)            595.15       340.80       935.95       0.4798          1.0281            1.0269         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)           595.15       242.55       837.70       0.3753          1.0446            1.0450         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)           595.15       343.51       938.66       0.4798          1.0281            1.0269         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)           595.15       247.00       842.15       0.3753          1.0446            1.0450         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)           595.15       359.06       954.21       0.4798          1.0281            1.0269         8.42
IVF-Binary-1024-nl158-random (self)                      595.15       647.47     1_242.62       0.3761          1.0445            1.0449         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)            564.93       154.56       719.49       0.1854          1.1897            1.1801         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)            564.93       155.23       720.16       0.1854          1.1899            1.1803         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)            564.93       161.23       726.17       0.1854          1.1899            1.1803         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)           564.93       246.84       811.78       0.3860          1.0419            1.0423         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)           564.93       350.55       915.48       0.4927          1.0265            1.0252         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)           564.93       255.41       820.34       0.3858          1.0419            1.0423         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)           564.93       353.72       918.66       0.4922          1.0265            1.0252         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)           564.93       271.40       836.33       0.3858          1.0419            1.0423         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)           564.93       359.98       924.91       0.4922          1.0265            1.0252         8.54
IVF-Binary-1024-nl223-random (self)                      564.93       680.99     1_245.92       0.3870          1.0419            1.0423         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)            632.27       164.43       796.70       0.1868          1.1852            1.1758         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)            632.27       164.40       796.67       0.1868          1.1855            1.1761         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)            632.27       167.23       799.50       0.1868          1.1856            1.1762         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)           632.27       253.55       885.82       0.3893          1.0414            1.0415         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)           632.27       355.17       987.44       0.4955          1.0262            1.0249         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)           632.27       256.45       888.72       0.3889          1.0415            1.0416         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)           632.27       357.93       990.20       0.4949          1.0263            1.0249         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)           632.27       264.01       896.28       0.3888          1.0415            1.0416         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)           632.27       365.34       997.61       0.4948          1.0263            1.0250         8.72
IVF-Binary-1024-nl316-random (self)                      632.27       688.05     1_320.31       0.3897          1.0414            1.0417         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)                638.84       148.57       787.41       0.1838          1.2000            1.1896         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)               638.84       150.02       788.86       0.1838          1.2000            1.1896         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)               638.84       153.07       791.91       0.1838          1.2000            1.1896         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)               638.84       238.93       877.77       0.3805          1.0433            1.0441         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)               638.84       337.40       976.24       0.4876          1.0270            1.0260         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)              638.84       253.74       892.58       0.3805          1.0433            1.0441         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)              638.84       346.54       985.38       0.4876          1.0270            1.0260         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)              638.84       245.38       884.22       0.3805          1.0433            1.0441         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)              638.84       347.49       986.33       0.4876          1.0270            1.0260         8.42
IVF-Binary-1024-nl158-pca (self)                         638.84       657.84     1_296.68       0.3794          1.0434            1.0442         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)               643.08       156.83       799.91       0.1870          1.1877            1.1785         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)               643.08       164.27       807.35       0.1870          1.1880            1.1787         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)               643.08       161.59       804.67       0.1870          1.1880            1.1787         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)              643.08       248.55       891.63       0.3905          1.0408            1.0416         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)              643.08       344.62       987.70       0.4989          1.0255            1.0246         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)              643.08       252.04       895.12       0.3902          1.0408            1.0417         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)              643.08       351.24       994.32       0.4983          1.0256            1.0247         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)              643.08       254.67       897.75       0.3902          1.0408            1.0417         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)              643.08       357.26     1_000.34       0.4983          1.0256            1.0247         8.54
IVF-Binary-1024-nl223-pca (self)                         643.08       669.55     1_312.63       0.3897          1.0410            1.0419         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)               732.32       164.31       896.62       0.1882          1.1834            1.1741         8.72
IVF-Binary-1024-nl316-np17-rf0-pca (query)               732.32       162.46       894.78       0.1882          1.1836            1.1744         8.72
IVF-Binary-1024-nl316-np25-rf0-pca (query)               732.32       190.25       922.57       0.1881          1.1837            1.1745         8.72
IVF-Binary-1024-nl316-np15-rf10-pca (query)              732.32       262.96       995.27       0.3940          1.0401            1.0408         8.72
IVF-Binary-1024-nl316-np15-rf20-pca (query)              732.32       353.31     1_085.63       0.5030          1.0252            1.0242         8.72
IVF-Binary-1024-nl316-np17-rf10-pca (query)              732.32       256.95       989.27       0.3937          1.0402            1.0410         8.72
IVF-Binary-1024-nl316-np17-rf20-pca (query)              732.32       363.42     1_095.74       0.5024          1.0252            1.0243         8.72
IVF-Binary-1024-nl316-np25-rf10-pca (query)              732.32       273.45     1_005.77       0.3936          1.0403            1.0410         8.72
IVF-Binary-1024-nl316-np25-rf20-pca (query)              732.32       371.21     1_103.53       0.5022          1.0253            1.0243         8.72
IVF-Binary-1024-nl316-pca (self)                         732.32       699.08     1_431.40       0.3929          1.0404            1.0411         8.72
IVF-Binary-512-nl158-np7-rf0-sign (query)                359.51       257.69       617.20       0.1519          1.2699            1.2515         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)               359.51       261.09       620.60       0.1519          1.2699            1.2515         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)               359.51       264.47       623.98       0.1519          1.2699            1.2515         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)               359.51       330.89       690.40       0.3414          1.0596            1.0531         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)               359.51       579.45       938.96       0.4416          1.0366            1.0317         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)              359.51       328.13       687.64       0.3414          1.0596            1.0531         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)              359.51       583.10       942.61       0.4416          1.0366            1.0317         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)              359.51       331.28       690.79       0.3414          1.0596            1.0531         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)              359.51       589.25       948.76       0.4416          1.0366            1.0317         3.36
IVF-Binary-512-nl158-sign (self)                         359.51       886.46     1_245.97       0.3425          1.0582            1.0527         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               367.87       262.65       630.52       0.1529          1.2635            1.2471         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               367.87       268.42       636.29       0.1528          1.2643            1.2484         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               367.87       267.88       635.75       0.1528          1.2643            1.2484         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              367.87       330.72       698.59       0.3504          1.0559            1.0502         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              367.87       585.00       952.87       0.4492          1.0348            1.0305         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              367.87       331.87       699.73       0.3502          1.0560            1.0502         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              367.87       589.31       957.17       0.4486          1.0350            1.0306         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              367.87       341.08       708.95       0.3502          1.0560            1.0502         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              367.87       593.38       961.24       0.4486          1.0350            1.0306         3.49
IVF-Binary-512-nl223-sign (self)                         367.87       896.40     1_264.27       0.3506          1.0551            1.0500         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)               458.93       266.55       725.48       0.1539          1.2662            1.2437         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)               458.93       267.83       726.76       0.1538          1.2675            1.2453         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)               458.93       272.44       731.37       0.1537          1.2681            1.2458         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)              458.93       337.22       796.15       0.3536          1.0549            1.0493         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)              458.93       592.88     1_051.81       0.4499          1.0345            1.0304         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)              458.93       337.19       796.12       0.3531          1.0551            1.0494         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)              458.93       593.61     1_052.54       0.4490          1.0347            1.0306         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)              458.93       350.36       809.29       0.3530          1.0551            1.0495         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)              458.93       597.79     1_056.72       0.4487          1.0348            1.0306         3.67
IVF-Binary-512-nl316-sign (self)                         458.93       910.61     1_369.54       0.3538          1.0540            1.0492         3.67
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
Exhaustive (query)                                       103.53     1_968.06     2_071.59       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        103.53     6_650.19     6_753.72       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                197.17       289.69       486.86       0.1140          1.2809            1.2433         2.28
ExhaustiveBinary-256-random-rf10 (query)                 197.17       441.79       638.95       0.3148          1.0656            1.0476         2.28
ExhaustiveBinary-256-random-rf20 (query)                 197.17       586.68       783.84       0.4075          1.0420            1.0293         2.28
ExhaustiveBinary-256-random (self)                       197.17     1_339.82     1_536.98       0.3168          1.0618            1.0471         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   400.55       286.25       686.80       0.1054          1.3026            1.2617         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    400.55       426.80       827.35       0.3012          1.0735            1.0517         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    400.55       575.32       975.86       0.3931          1.0471            1.0315         2.28
ExhaustiveBinary-256-pca (self)                          400.55     1_321.88     1_722.43       0.3048          1.0710            1.0504         2.28
ExhaustiveBinary-512-random_no_rr (query)                300.77       410.65       711.42       0.1506          1.2094            1.1809         4.55
ExhaustiveBinary-512-random-rf10 (query)                 300.77       569.34       870.11       0.3395          1.0453            1.0426         4.55
ExhaustiveBinary-512-random-rf20 (query)                 300.77       732.79     1_033.56       0.4326          1.0293            1.0264         4.55
ExhaustiveBinary-512-random (self)                       300.77     1_864.64     2_165.41       0.3401          1.0435            1.0423         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   498.74       412.81       911.54       0.1459          1.2162            1.1914         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    498.74       567.38     1_066.11       0.3341          1.0468            1.0435         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    498.74       730.02     1_228.75       0.4278          1.0295            1.0269         4.55
ExhaustiveBinary-512-pca (self)                          498.74     1_817.68     2_316.42       0.3355          1.0454            1.0433         4.55
ExhaustiveBinary-1024-random_no_rr (query)               501.87       646.46     1_148.33       0.1761          1.1673            1.1571         9.11
ExhaustiveBinary-1024-random-rf10 (query)                501.87       819.43     1_321.29       0.3603          1.0383            1.0383         9.11
ExhaustiveBinary-1024-random-rf20 (query)                501.87       995.21     1_497.08       0.4618          1.0244            1.0230         9.11
ExhaustiveBinary-1024-random (self)                      501.87     2_669.11     3_170.98       0.3602          1.0377            1.0383         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  711.96       643.52     1_355.47       0.1756          1.1686            1.1586         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   711.96       814.53     1_526.49       0.3594          1.0382            1.0385         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   711.96       996.08     1_708.04       0.4576          1.0246            1.0237         9.11
ExhaustiveBinary-1024-pca (self)                         711.96     2_669.36     3_381.32       0.3584          1.0383            1.0389         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  132.66       837.89       970.54       0.1691          1.1871            1.1718         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   132.66       919.57     1_052.22       0.3431          1.0433            1.0415         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   132.66     1_420.87     1_553.52       0.4437          1.0266            1.0250         4.58
ExhaustiveBinary-768-sign (self)                         132.66     2_972.47     3_105.13       0.3438          1.0424            1.0413         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)              670.42        98.84       769.26       0.1164          1.2725            1.2403         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)             670.42        99.87       770.29       0.1164          1.2725            1.2403         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)             670.42       103.07       773.49       0.1164          1.2725            1.2403         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)             670.42       204.80       875.21       0.3172          1.0636            1.0468         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)             670.42       313.70       984.12       0.4095          1.0414            1.0290         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)            670.42       196.93       867.35       0.3172          1.0636            1.0468         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)            670.42       317.27       987.69       0.4095          1.0414            1.0290         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)            670.42       199.45       869.87       0.3172          1.0636            1.0468         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)            670.42       321.99       992.40       0.4095          1.0414            1.0290         2.74
IVF-Binary-256-nl158-random (self)                       670.42       429.28     1_099.69       0.3191          1.0600            1.0465         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)             567.25        94.89       662.13       0.1331          1.2303            1.1909         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)             567.25        96.37       663.62       0.1331          1.2303            1.1909         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)             567.25        98.92       666.16       0.1331          1.2303            1.1909         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)            567.25       207.73       774.98       0.3574          1.0474            1.0376         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)            567.25       325.94       893.18       0.4568          1.0309            1.0232         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)            567.25       207.44       774.68       0.3573          1.0474            1.0376         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)            567.25       325.24       892.49       0.4568          1.0309            1.0232         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)            567.25       208.95       776.20       0.3573          1.0474            1.0376         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)            567.25       326.76       894.01       0.4568          1.0309            1.0232         2.93
IVF-Binary-256-nl223-random (self)                       567.25       442.12     1_009.37       0.3590          1.0439            1.0372         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)             676.68       108.34       785.03       0.1402          1.2168            1.1747         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)             676.68       104.42       781.10       0.1402          1.2169            1.1747         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)             676.68       108.42       785.10       0.1402          1.2169            1.1747         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)            676.68       218.24       894.92       0.3660          1.0441            1.0360         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)            676.68       332.23     1_008.91       0.4635          1.0293            1.0224         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)            676.68       214.78       891.46       0.3660          1.0441            1.0360         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)            676.68       336.24     1_012.92       0.4634          1.0293            1.0224         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)            676.68       217.92       894.60       0.3660          1.0441            1.0360         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)            676.68       337.58     1_014.27       0.4634          1.0293            1.0224         3.21
IVF-Binary-256-nl316-random (self)                       676.68       470.72     1_147.40       0.3675          1.0408            1.0357         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)                 794.25        89.64       883.89       0.1076          1.2918            1.2570         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)                794.25        89.80       884.05       0.1076          1.2918            1.2570         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)                794.25        89.86       884.11       0.1076          1.2918            1.2570         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)                794.25       193.56       987.81       0.3045          1.0705            1.0511         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)                794.25       306.55     1_100.80       0.3973          1.0449            1.0309         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)               794.25       213.71     1_007.97       0.3045          1.0705            1.0511         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)               794.25       312.05     1_106.30       0.3973          1.0449            1.0309         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)               794.25       194.74       989.00       0.3045          1.0705            1.0511         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)               794.25       310.74     1_104.99       0.3973          1.0449            1.0309         2.74
IVF-Binary-256-nl158-pca (self)                          794.25       389.52     1_183.78       0.3083          1.0672            1.0497         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)                932.88        98.47     1_031.35       0.1268          1.2395            1.2037         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)                932.88        96.19     1_029.07       0.1268          1.2395            1.2037         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)                932.88        99.90     1_032.78       0.1268          1.2395            1.2037         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)               932.88       210.57     1_143.45       0.3613          1.0476            1.0372         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)               932.88       329.43     1_262.30       0.4626          1.0301            1.0228         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)               932.88       207.75     1_140.63       0.3613          1.0476            1.0372         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)               932.88       324.88     1_257.76       0.4626          1.0301            1.0228         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)               932.88       207.39     1_140.26       0.3613          1.0476            1.0372         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)               932.88       327.43     1_260.31       0.4626          1.0301            1.0228         2.93
IVF-Binary-256-nl223-pca (self)                          932.88       421.01     1_353.89       0.3645          1.0451            1.0365         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)                895.27       104.33       999.60       0.1365          1.2209            1.1850         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)                895.27       105.71     1_000.98       0.1365          1.2210            1.1850         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)                895.27       109.61     1_004.88       0.1365          1.2210            1.1850         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)               895.27       217.82     1_113.08       0.3737          1.0433            1.0349         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)               895.27       334.71     1_229.97       0.4727          1.0280            1.0216         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)               895.27       214.68     1_109.95       0.3737          1.0433            1.0349         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)               895.27       331.12     1_226.38       0.4727          1.0280            1.0216         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)               895.27       214.97     1_110.24       0.3737          1.0433            1.0349         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)               895.27       337.24     1_232.50       0.4727          1.0280            1.0216         3.21
IVF-Binary-256-nl316-pca (self)                          895.27       463.68     1_358.95       0.3769          1.0410            1.0341         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)              703.59       125.14       828.73       0.1520          1.2064            1.1790         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)             703.59       126.04       829.63       0.1520          1.2064            1.1790         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)             703.59       129.97       833.56       0.1520          1.2064            1.1790         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)             703.59       241.00       944.59       0.3401          1.0451            1.0423         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)             703.59       356.92     1_060.51       0.4336          1.0292            1.0262         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)            703.59       239.88       943.47       0.3401          1.0451            1.0423         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)            703.59       360.49     1_064.08       0.4336          1.0292            1.0262         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)            703.59       240.03       943.62       0.3401          1.0451            1.0423         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)            703.59       361.10     1_064.69       0.4336          1.0292            1.0262         5.02
IVF-Binary-512-nl158-random (self)                       703.59       574.98     1_278.57       0.3407          1.0433            1.0421         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)             670.12       132.77       802.89       0.1603          1.1838            1.1572         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)             670.12       133.42       803.53       0.1603          1.1838            1.1573         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)             670.12       137.44       807.56       0.1603          1.1838            1.1573         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)            670.12       246.14       916.26       0.3579          1.0405            1.0379         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)            670.12       367.84     1_037.96       0.4565          1.0263            1.0235         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)            670.12       244.38       914.50       0.3579          1.0405            1.0379         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)            670.12       366.91     1_037.03       0.4565          1.0263            1.0235         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)            670.12       250.24       920.36       0.3579          1.0405            1.0379         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)            670.12       373.22     1_043.34       0.4565          1.0263            1.0235         5.21
IVF-Binary-512-nl223-random (self)                       670.12       598.74     1_268.86       0.3585          1.0390            1.0378         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)             790.35       144.61       934.96       0.1626          1.1785            1.1525         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)             790.35       141.49       931.84       0.1625          1.1787            1.1525         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)             790.35       156.37       946.72       0.1625          1.1787            1.1525         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)            790.35       316.20     1_106.55       0.3621          1.0391            1.0372         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)            790.35       437.24     1_227.59       0.4589          1.0256            1.0234         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)            790.35       302.83     1_093.18       0.3621          1.0391            1.0372         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)            790.35       408.96     1_199.31       0.4588          1.0256            1.0234         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)            790.35       269.99     1_060.34       0.3621          1.0391            1.0372         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)            790.35       399.78     1_190.13       0.4588          1.0256            1.0234         5.48
IVF-Binary-512-nl316-random (self)                       790.35       683.45     1_473.80       0.3624          1.0378            1.0372         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)               1_002.44       129.22     1_131.66       0.1473          1.2122            1.1891         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)              1_002.44       146.76     1_149.20       0.1473          1.2122            1.1891         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)              1_002.44       128.50     1_130.94       0.1473          1.2122            1.1891         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)              1_002.44       253.09     1_255.52       0.3358          1.0460            1.0431         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)              1_002.44       357.12     1_359.56       0.4296          1.0292            1.0267         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)             1_002.44       237.73     1_240.17       0.3358          1.0460            1.0431         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)             1_002.44       357.73     1_360.17       0.4296          1.0292            1.0267         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)             1_002.44       243.15     1_245.59       0.3358          1.0460            1.0431         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)             1_002.44       361.07     1_363.51       0.4296          1.0292            1.0267         5.02
IVF-Binary-512-nl158-pca (self)                        1_002.44       579.02     1_581.46       0.3372          1.0445            1.0428         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)                876.24       133.64     1_009.88       0.1587          1.1837            1.1593         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)                876.24       134.94     1_011.18       0.1587          1.1837            1.1593         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)                876.24       137.82     1_014.07       0.1587          1.1837            1.1593         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)               876.24       245.81     1_122.05       0.3586          1.0397            1.0379         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)               876.24       365.96     1_242.20       0.4554          1.0259            1.0235         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)               876.24       250.99     1_127.23       0.3586          1.0397            1.0379         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)               876.24       369.31     1_245.55       0.4554          1.0259            1.0235         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)               876.24       249.50     1_125.74       0.3586          1.0397            1.0379         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)               876.24       375.29     1_251.53       0.4554          1.0259            1.0235         5.21
IVF-Binary-512-nl223-pca (self)                          876.24       599.74     1_475.98       0.3592          1.0390            1.0377         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)                996.63       147.46     1_144.09       0.1625          1.1766            1.1521         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)                996.63       144.86     1_141.49       0.1624          1.1767            1.1522         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)                996.63       147.33     1_143.96       0.1624          1.1767            1.1522         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)               996.63       254.45     1_251.08       0.3632          1.0386            1.0370         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)               996.63       375.43     1_372.06       0.4592          1.0254            1.0230         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)               996.63       256.53     1_253.16       0.3632          1.0386            1.0370         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)               996.63       379.87     1_376.50       0.4591          1.0254            1.0230         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)               996.63       258.66     1_255.30       0.3632          1.0386            1.0370         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)               996.63       392.49     1_389.12       0.4591          1.0254            1.0230         5.48
IVF-Binary-512-nl316-pca (self)                          996.63       634.33     1_630.96       0.3640          1.0380            1.0369         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)             905.70       200.09     1_105.79       0.1767          1.1663            1.1563         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)            905.70       202.01     1_107.71       0.1767          1.1663            1.1563         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)            905.70       207.10     1_112.80       0.1767          1.1663            1.1563         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)            905.70       317.43     1_223.13       0.3608          1.0382            1.0382         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)            905.70       449.28     1_354.98       0.4625          1.0243            1.0230         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)           905.70       321.84     1_227.54       0.3608          1.0382            1.0382         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)           905.70       459.32     1_365.01       0.4625          1.0243            1.0230         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)           905.70       342.90     1_248.59       0.3608          1.0382            1.0382         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)           905.70       461.99     1_367.68       0.4625          1.0243            1.0230         9.57
IVF-Binary-1024-nl158-random (self)                      905.70       881.37     1_787.06       0.3607          1.0376            1.0382         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)            874.53       213.25     1_087.78       0.1797          1.1568            1.1474         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)            874.53       214.83     1_089.35       0.1797          1.1568            1.1474         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)            874.53       217.24     1_091.77       0.1797          1.1568            1.1474         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)           874.53       333.15     1_207.67       0.3720          1.0360            1.0359         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)           874.53       465.19     1_339.71       0.4751          1.0229            1.0217         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)           874.53       337.07     1_211.60       0.3720          1.0360            1.0359         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)           874.53       467.82     1_342.34       0.4751          1.0229            1.0217         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)           874.53       345.80     1_220.33       0.3720          1.0360            1.0359         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)           874.53       479.11     1_353.64       0.4751          1.0229            1.0217         9.76
IVF-Binary-1024-nl223-random (self)                      874.53       913.97     1_788.49       0.3716          1.0354            1.0359         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)          1_002.05       225.48     1_227.52       0.1805          1.1543            1.1451        10.03
IVF-Binary-1024-nl316-np17-rf0-random (query)          1_002.05       220.61     1_222.66       0.1805          1.1544            1.1451        10.03
IVF-Binary-1024-nl316-np25-rf0-random (query)          1_002.05       235.93     1_237.98       0.1805          1.1544            1.1451        10.03
IVF-Binary-1024-nl316-np15-rf10-random (query)         1_002.05       349.78     1_351.83       0.3733          1.0356            1.0358        10.03
IVF-Binary-1024-nl316-np15-rf20-random (query)         1_002.05       478.70     1_480.75       0.4764          1.0227            1.0216        10.03
IVF-Binary-1024-nl316-np17-rf10-random (query)         1_002.05       349.83     1_351.88       0.3733          1.0356            1.0358        10.03
IVF-Binary-1024-nl316-np17-rf20-random (query)         1_002.05       482.23     1_484.27       0.4763          1.0227            1.0216        10.03
IVF-Binary-1024-nl316-np25-rf10-random (query)         1_002.05       361.86     1_363.90       0.3733          1.0356            1.0358        10.03
IVF-Binary-1024-nl316-np25-rf20-random (query)         1_002.05       492.81     1_494.86       0.4763          1.0227            1.0216        10.03
IVF-Binary-1024-nl316-random (self)                    1_002.05       961.60     1_963.65       0.3736          1.0350            1.0356        10.03
IVF-Binary-1024-nl158-np7-rf0-pca (query)              1_115.19       201.03     1_316.22       0.1762          1.1673            1.1573         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)             1_115.19       201.95     1_317.14       0.1762          1.1673            1.1573         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)             1_115.19       207.13     1_322.32       0.1762          1.1673            1.1573         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)             1_115.19       319.28     1_434.47       0.3600          1.0380            1.0384         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)             1_115.19       450.94     1_566.13       0.4586          1.0245            1.0236         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)            1_115.19       323.24     1_438.43       0.3600          1.0380            1.0384         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)            1_115.19       463.89     1_579.08       0.4586          1.0245            1.0236         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)            1_115.19       330.63     1_445.82       0.3600          1.0380            1.0384         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)            1_115.19       467.15     1_582.34       0.4586          1.0245            1.0236         9.57
IVF-Binary-1024-nl158-pca (self)                       1_115.19       882.90     1_998.09       0.3591          1.0381            1.0388         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_086.00       210.45     1_296.44       0.1795          1.1565            1.1467         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_086.00       210.20     1_296.19       0.1795          1.1565            1.1467         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_086.00       215.98     1_301.97       0.1795          1.1565            1.1467         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_086.00       336.07     1_422.07       0.3711          1.0358            1.0360         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_086.00       468.68     1_554.67       0.4724          1.0231            1.0221         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_086.00       337.19     1_423.19       0.3711          1.0358            1.0360         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_086.00       473.34     1_559.34       0.4724          1.0231            1.0221         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_086.00       351.80     1_437.79       0.3711          1.0358            1.0360         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_086.00       487.51     1_573.50       0.4724          1.0231            1.0221         9.76
IVF-Binary-1024-nl223-pca (self)                       1_086.00       909.21     1_995.20       0.3704          1.0359            1.0363         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_201.54       221.58     1_423.12       0.1805          1.1540            1.1448        10.03
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_201.54       220.13     1_421.67       0.1805          1.1540            1.1448        10.03
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_201.54       228.91     1_430.45       0.1805          1.1540            1.1448        10.03
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_201.54       348.50     1_550.04       0.3737          1.0353            1.0357        10.03
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_201.54       479.41     1_680.95       0.4739          1.0228            1.0220        10.03
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_201.54       362.20     1_563.75       0.3737          1.0353            1.0357        10.03
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_201.54       485.11     1_686.65       0.4738          1.0228            1.0220        10.03
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_201.54       361.38     1_562.92       0.3737          1.0353            1.0357        10.03
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_201.54       496.06     1_697.61       0.4738          1.0228            1.0220        10.03
IVF-Binary-1024-nl316-pca (self)                       1_201.54       954.40     2_155.95       0.3726          1.0354            1.0359        10.03
IVF-Binary-768-nl158-np7-rf0-sign (query)                532.22       389.37       921.58       0.1693          1.1870            1.1720         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)               532.22       396.80       929.01       0.1693          1.1870            1.1720         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)               532.22       415.74       947.95       0.1693          1.1870            1.1720         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)               532.22       471.62     1_003.84       0.3434          1.0432            1.0414         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)               532.22       855.06     1_387.28       0.4438          1.0266            1.0249         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)              532.22       475.62     1_007.83       0.3434          1.0432            1.0414         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)              532.22       859.85     1_392.06       0.4438          1.0266            1.0249         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)              532.22       480.78     1_013.00       0.3434          1.0432            1.0414         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)              532.22       865.43     1_397.65       0.4438          1.0266            1.0249         5.04
IVF-Binary-768-nl158-sign (self)                         532.22     1_338.08     1_870.30       0.3440          1.0423            1.0413         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)               510.33       391.29       901.63       0.1692          1.1872            1.1717         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)               510.33       391.54       901.87       0.1692          1.1872            1.1717         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)               510.33       398.33       908.66       0.1692          1.1872            1.1717         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)              510.33       478.96       989.29       0.3492          1.0414            1.0401         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)              510.33       863.54     1_373.87       0.4483          1.0259            1.0245         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)              510.33       479.20       989.54       0.3492          1.0414            1.0401         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)              510.33       871.90     1_382.23       0.4483          1.0259            1.0245         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)              510.33       494.79     1_005.13       0.3492          1.0414            1.0401         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)              510.33       875.60     1_385.93       0.4483          1.0259            1.0245         5.23
IVF-Binary-768-nl223-sign (self)                         510.33     1_348.43     1_858.76       0.3496          1.0408            1.0400         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)               618.92       401.79     1_020.71       0.1694          1.1867            1.1714         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)               618.92       405.00     1_023.91       0.1694          1.1867            1.1714         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)               618.92       405.54     1_024.46       0.1694          1.1867            1.1714         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)              618.92       490.41     1_109.32       0.3501          1.0411            1.0399         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)              618.92       870.59     1_489.51       0.4490          1.0257            1.0244         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)              618.92       491.09     1_110.00       0.3501          1.0411            1.0399         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)              618.92       879.79     1_498.70       0.4489          1.0257            1.0244         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)              618.92       501.18     1_120.10       0.3501          1.0411            1.0399         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)              618.92       900.53     1_519.45       0.4489          1.0257            1.0244         5.51
IVF-Binary-768-nl316-sign (self)                         618.92     1_390.04     2_008.96       0.3504          1.0405            1.0398         5.51
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
Exhaustive (query)                                        32.77       747.01       779.79       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.77     2_531.31     2_564.08       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                 71.22       239.81       311.03       0.0970          1.6334            1.6378         1.78
ExhaustiveBinary-256-random-rf10 (query)                  71.22       352.68       423.90       0.3643          1.1391            1.1302         1.78
ExhaustiveBinary-256-random-rf20 (query)                  71.22       462.45       533.68       0.5087          1.0798            1.0701         1.78
ExhaustiveBinary-256-random (self)                        71.22     1_143.19     1_214.42       0.3862          1.1443            1.1409         1.78
ExhaustiveBinary-256-pca_no_rr (query)                    96.62       254.59       351.21       0.0922          1.6524            1.6606         1.78
ExhaustiveBinary-256-pca-rf10 (query)                     96.62       354.14       450.76       0.3517          1.1465            1.1384         1.78
ExhaustiveBinary-256-pca-rf20 (query)                     96.62       460.67       557.29       0.4943          1.0846            1.0744         1.78
ExhaustiveBinary-256-pca (self)                           96.62     1_145.50     1_242.12       0.3764          1.1502            1.1477         1.78
ExhaustiveBinary-512-random_no_rr (query)                 82.71       359.05       441.76       0.1464          1.5035            1.5085         3.55
ExhaustiveBinary-512-random-rf10 (query)                  82.71       479.01       561.73       0.4596          1.0936            1.0901         3.55
ExhaustiveBinary-512-random-rf20 (query)                  82.71       590.08       672.80       0.6085          1.0504            1.0459         3.55
ExhaustiveBinary-512-random (self)                        82.71     1_587.42     1_670.13       0.4800          1.0996            1.0995         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   107.98       359.81       467.78       0.1458          1.5041            1.5097         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    107.98       483.68       591.65       0.4543          1.0952            1.0911         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    107.98       589.51       697.49       0.6037          1.0513            1.0464         3.55
ExhaustiveBinary-512-pca (self)                          107.98     1_575.81     1_683.79       0.4777          1.1003            1.0997         3.55
ExhaustiveBinary-1024-random_no_rr (query)               116.12       544.85       660.97       0.2155          1.3655            1.3721         7.10
ExhaustiveBinary-1024-random-rf10 (query)                116.12       663.75       779.87       0.5869          1.0540            1.0515         7.10
ExhaustiveBinary-1024-random-rf20 (query)                116.12       784.08       900.20       0.7380          1.0260            1.0224         7.10
ExhaustiveBinary-1024-random (self)                      116.12     2_210.93     2_327.04       0.6118          1.0576            1.0546         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  142.47       542.30       684.77       0.2122          1.3735            1.3798         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   142.47       658.94       801.41       0.5776          1.0560            1.0532         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   142.47       774.27       916.73       0.7291          1.0273            1.0232         7.10
ExhaustiveBinary-1024-pca (self)                         142.47     2_206.43     2_348.90       0.6017          1.0602            1.0571         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   44.74       436.89       481.63       0.1044          1.6421            1.6499         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    44.74       476.54       521.28       0.3737          1.1368            1.1275         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    44.74       716.51       761.25       0.5265          1.0745            1.0646         1.53
ExhaustiveBinary-256-sign (self)                          44.74     1_552.46     1_597.20       0.3940          1.1439            1.1394         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)              245.88        53.48       299.36       0.1006          1.6235            1.6329         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)             245.88        51.39       297.26       0.1005          1.6238            1.6332         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)             245.88        58.81       304.69       0.1005          1.6238            1.6332         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)             245.88       101.28       347.15       0.3692          1.1376            1.1295         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)             245.88       154.46       400.34       0.5128          1.0789            1.0697         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)            245.88       100.10       345.98       0.3678          1.1379            1.1296         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)            245.88       155.82       401.70       0.5110          1.0792            1.0698         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)            245.88       107.93       353.81       0.3677          1.1379            1.1296         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)            245.88       160.99       406.87       0.5108          1.0792            1.0698         1.93
IVF-Binary-256-nl158-random (self)                       245.88       228.50       474.38       0.3897          1.1428            1.1402         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             288.51        47.63       336.14       0.1122          1.5873            1.5872         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             288.51        47.91       336.41       0.1121          1.5880            1.5877         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             288.51        52.95       341.46       0.1121          1.5881            1.5878         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            288.51       102.33       390.83       0.3938          1.1238            1.1155         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            288.51       154.76       443.27       0.5354          1.0713            1.0627         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            288.51       101.76       390.27       0.3933          1.1239            1.1157         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            288.51       160.35       448.86       0.5347          1.0714            1.0629         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            288.51       108.98       397.49       0.3931          1.1240            1.1157         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            288.51       160.65       449.15       0.5346          1.0714            1.0629         2.00
IVF-Binary-256-nl223-random (self)                       288.51       235.18       523.69       0.4141          1.1288            1.1266         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             351.22        48.67       399.89       0.1174          1.5678            1.5682         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             351.22        49.51       400.73       0.1173          1.5683            1.5687         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             351.22        55.02       406.24       0.1173          1.5685            1.5688         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            351.22       103.16       454.39       0.4041          1.1177            1.1109         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            351.22       156.45       507.67       0.5466          1.0675            1.0600         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            351.22       103.12       454.34       0.4034          1.1179            1.1111         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            351.22       155.54       506.76       0.5459          1.0677            1.0602         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            351.22       107.95       459.17       0.4033          1.1180            1.1112         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            351.22       163.69       514.92       0.5457          1.0677            1.0602         2.09
IVF-Binary-256-nl316-random (self)                       351.22       241.71       592.93       0.4252          1.1217            1.1220         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)                 268.32        41.33       309.66       0.0960          1.6427            1.6567         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)                268.32        43.47       311.80       0.0959          1.6429            1.6568         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)                268.32        46.77       315.10       0.0959          1.6429            1.6568         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)                268.32        95.49       363.81       0.3562          1.1449            1.1376         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)                268.32       153.43       421.75       0.4976          1.0836            1.0741         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)               268.32        96.64       364.96       0.3551          1.1451            1.1377         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)               268.32       154.99       423.31       0.4963          1.0838            1.0742         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)               268.32        99.46       367.78       0.3551          1.1451            1.1377         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)               268.32       155.73       424.06       0.4963          1.0838            1.0742         1.93
IVF-Binary-256-nl158-pca (self)                          268.32       233.61       501.93       0.3800          1.1487            1.1470         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                316.99        45.31       362.30       0.1095          1.5965            1.6007         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                316.99        47.07       364.06       0.1094          1.5975            1.6019         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                316.99        52.17       369.17       0.1094          1.5976            1.6020         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               316.99       106.26       423.25       0.3870          1.1269            1.1202         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               316.99       156.43       473.42       0.5263          1.0737            1.0657         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               316.99       100.60       417.59       0.3864          1.1272            1.1205         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               316.99       156.25       473.24       0.5254          1.0739            1.0660         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               316.99       109.55       426.54       0.3863          1.1272            1.1205         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               316.99       161.80       478.79       0.5253          1.0739            1.0660         2.00
IVF-Binary-256-nl223-pca (self)                          316.99       232.57       549.56       0.4099          1.1303            1.1302         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                387.22        48.63       435.85       0.1160          1.5759            1.5792         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                387.22        48.78       436.00       0.1159          1.5765            1.5798         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                387.22        52.99       440.22       0.1158          1.5771            1.5801         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               387.22       106.06       493.29       0.3965          1.1216            1.1148         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               387.22       155.59       542.81       0.5374          1.0701            1.0627         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               387.22       104.86       492.09       0.3959          1.1218            1.1151         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               387.22       158.84       546.07       0.5366          1.0703            1.0630         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               387.22       106.64       493.86       0.3957          1.1219            1.1152         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               387.22       163.15       550.37       0.5363          1.0703            1.0631         2.09
IVF-Binary-256-nl316-pca (self)                          387.22       239.69       626.91       0.4197          1.1247            1.1254         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)              261.85        59.02       320.87       0.1479          1.5006            1.5075         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)             261.85        62.65       324.50       0.1479          1.5007            1.5075         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)             261.85        67.67       329.51       0.1479          1.5007            1.5075         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)             261.85       118.98       380.82       0.4609          1.0933            1.0899         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)             261.85       176.46       438.31       0.6093          1.0503            1.0458         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)            261.85       121.50       383.34       0.4606          1.0933            1.0899         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)            261.85       179.53       441.37       0.6092          1.0503            1.0458         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)            261.85       126.70       388.54       0.4606          1.0933            1.0899         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)            261.85       187.36       449.20       0.6091          1.0503            1.0458         3.71
IVF-Binary-512-nl158-random (self)                       261.85       304.76       566.60       0.4809          1.0993            1.0994         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             303.73        63.55       367.27       0.1561          1.4799            1.4845         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             303.73        66.29       370.01       0.1561          1.4802            1.4846         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             303.73        73.05       376.77       0.1561          1.4802            1.4846         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            303.73       122.81       426.53       0.4754          1.0877            1.0848         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            303.73       176.25       479.97       0.6216          1.0474            1.0431         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            303.73       124.21       427.93       0.4750          1.0878            1.0849         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            303.73       181.73       485.46       0.6211          1.0475            1.0432         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            303.73       130.12       433.84       0.4750          1.0878            1.0849         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            303.73       188.41       492.13       0.6211          1.0475            1.0432         3.77
IVF-Binary-512-nl223-random (self)                       303.73       310.49       614.21       0.4946          1.0941            1.0938         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             372.04        66.40       438.44       0.1598          1.4705            1.4754         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             372.04        67.60       439.64       0.1597          1.4708            1.4759         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             372.04        74.99       447.03       0.1597          1.4708            1.4759         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            372.04       125.41       497.44       0.4803          1.0859            1.0830         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            372.04       178.66       550.69       0.6273          1.0463            1.0422         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            372.04       127.57       499.61       0.4799          1.0861            1.0831         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            372.04       180.10       552.14       0.6267          1.0464            1.0424         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            372.04       131.22       503.26       0.4798          1.0861            1.0831         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            372.04       191.34       563.38       0.6267          1.0464            1.0424         3.86
IVF-Binary-512-nl316-random (self)                       372.04       315.76       687.79       0.4995          1.0923            1.0917         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)                 307.54        59.05       366.59       0.1474          1.5015            1.5083         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)                307.54        62.20       369.74       0.1473          1.5015            1.5083         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)                307.54        68.04       375.58       0.1473          1.5015            1.5083         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)                307.54       123.39       430.93       0.4558          1.0948            1.0910         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)                307.54       174.46       482.00       0.6048          1.0510            1.0463         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)               307.54       123.00       430.54       0.4554          1.0948            1.0910         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)               307.54       183.14       490.68       0.6044          1.0511            1.0463         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)               307.54       132.09       439.63       0.4554          1.0948            1.0910         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)               307.54       185.58       493.12       0.6044          1.0511            1.0463         3.71
IVF-Binary-512-nl158-pca (self)                          307.54       305.81       613.35       0.4788          1.1000            1.0995         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                356.69        63.40       420.09       0.1557          1.4808            1.4847         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                356.69        66.44       423.14       0.1556          1.4812            1.4854         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                356.69        72.70       429.39       0.1556          1.4812            1.4854         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               356.69       121.97       478.66       0.4715          1.0888            1.0850         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               356.69       176.69       533.38       0.6191          1.0478            1.0432         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               356.69       122.87       479.56       0.4710          1.0889            1.0851         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               356.69       178.76       535.45       0.6185          1.0479            1.0434         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               356.69       134.51       491.20       0.4709          1.0889            1.0851         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               356.69       187.36       544.05       0.6184          1.0479            1.0434         3.77
IVF-Binary-512-nl223-pca (self)                          356.69       318.13       674.82       0.4931          1.0945            1.0940         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                413.10        67.12       480.22       0.1590          1.4717            1.4754         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                413.10        68.98       482.08       0.1589          1.4720            1.4760         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                413.10        73.53       486.63       0.1589          1.4721            1.4761         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               413.10       124.78       537.88       0.4766          1.0869            1.0833         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               413.10       180.20       593.30       0.6238          1.0467            1.0422         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               413.10       128.23       541.33       0.4761          1.0870            1.0835         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               413.10       179.87       592.97       0.6230          1.0469            1.0423         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               413.10       132.09       545.19       0.4761          1.0870            1.0835         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               413.10       187.73       600.83       0.6229          1.0469            1.0424         3.86
IVF-Binary-512-nl316-pca (self)                          413.10       317.41       730.51       0.4984          1.0925            1.0920         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)             289.29        91.40       380.68       0.2162          1.3647            1.3717         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)            289.29        97.79       387.08       0.2162          1.3647            1.3717         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)            289.29       102.44       391.73       0.2162          1.3647            1.3717         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)            289.29       162.28       451.56       0.5873          1.0539            1.0515         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)            289.29       212.61       501.90       0.7382          1.0260            1.0223         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)           289.29       158.27       447.55       0.5872          1.0539            1.0515         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)           289.29       218.11       507.40       0.7382          1.0260            1.0223         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)           289.29       165.54       454.82       0.5872          1.0539            1.0515         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)           289.29       226.26       515.55       0.7382          1.0260            1.0223         7.26
IVF-Binary-1024-nl158-random (self)                      289.29       433.45       722.73       0.6121          1.0575            1.0545         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            330.19        95.88       426.07       0.2204          1.3573            1.3648         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            330.19        98.80       428.99       0.2204          1.3574            1.3648         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            330.19       108.64       438.83       0.2204          1.3574            1.3648         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           330.19       157.54       487.73       0.5941          1.0523            1.0500         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           330.19       215.96       546.15       0.7440          1.0252            1.0214         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           330.19       162.02       492.21       0.5938          1.0524            1.0500         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           330.19       223.92       554.11       0.7437          1.0252            1.0215         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           330.19       172.11       502.30       0.5938          1.0524            1.0500         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           330.19       229.82       560.01       0.7436          1.0252            1.0215         7.32
IVF-Binary-1024-nl223-random (self)                      330.19       439.86       770.05       0.6183          1.0559            1.0530         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            407.15       100.34       507.49       0.2223          1.3533            1.3607         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)            407.15       100.18       507.34       0.2222          1.3534            1.3610         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)            407.15       110.09       517.24       0.2222          1.3534            1.3610         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)           407.15       162.25       569.40       0.5963          1.0517            1.0493         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)           407.15       218.32       625.47       0.7464          1.0248            1.0211         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)           407.15       160.90       568.05       0.5959          1.0518            1.0494         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)           407.15       221.06       628.21       0.7461          1.0248            1.0211         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)           407.15       172.48       579.63       0.5959          1.0518            1.0494         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)           407.15       236.19       643.35       0.7460          1.0248            1.0211         7.41
IVF-Binary-1024-nl316-random (self)                      407.15       450.23       857.38       0.6211          1.0552            1.0522         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)                317.81        91.44       409.25       0.2128          1.3727            1.3795         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)               317.81        98.83       416.64       0.2128          1.3727            1.3795         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)               317.81       103.39       421.20       0.2128          1.3727            1.3795         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)               317.81       154.88       472.69       0.5780          1.0559            1.0531         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)               317.81       212.24       530.05       0.7294          1.0272            1.0232         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)              317.81       163.36       481.16       0.5779          1.0559            1.0531         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)              317.81       220.52       538.33       0.7293          1.0272            1.0232         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)              317.81       170.07       487.88       0.5779          1.0559            1.0531         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)              317.81       231.40       549.21       0.7293          1.0272            1.0232         7.26
IVF-Binary-1024-nl158-pca (self)                         317.81       443.46       761.26       0.6020          1.0601            1.0571         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               358.34        97.16       455.50       0.2176          1.3640            1.3702         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               358.34        98.05       456.39       0.2175          1.3642            1.3705         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               358.34       109.73       468.07       0.2175          1.3642            1.3705         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              358.34       157.97       516.31       0.5853          1.0541            1.0511         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              358.34       216.41       574.75       0.7358          1.0263            1.0223         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              358.34       161.54       519.88       0.5850          1.0542            1.0512         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              358.34       218.40       576.75       0.7354          1.0264            1.0223         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              358.34       170.15       528.49       0.5849          1.0542            1.0512         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              358.34       231.85       590.19       0.7353          1.0264            1.0223         7.32
IVF-Binary-1024-nl223-pca (self)                         358.34       446.29       804.63       0.6094          1.0581            1.0550         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               425.23       100.83       526.05       0.2191          1.3601            1.3666         7.41
IVF-Binary-1024-nl316-np17-rf0-pca (query)               425.23       101.43       526.66       0.2190          1.3603            1.3667         7.41
IVF-Binary-1024-nl316-np25-rf0-pca (query)               425.23       111.45       536.68       0.2190          1.3603            1.3667         7.41
IVF-Binary-1024-nl316-np15-rf10-pca (query)              425.23       160.46       585.68       0.5876          1.0535            1.0508         7.41
IVF-Binary-1024-nl316-np15-rf20-pca (query)              425.23       222.46       647.69       0.7378          1.0260            1.0222         7.41
IVF-Binary-1024-nl316-np17-rf10-pca (query)              425.23       161.23       586.46       0.5872          1.0536            1.0509         7.41
IVF-Binary-1024-nl316-np17-rf20-pca (query)              425.23       221.18       646.41       0.7374          1.0261            1.0222         7.41
IVF-Binary-1024-nl316-np25-rf10-pca (query)              425.23       169.71       594.94       0.5872          1.0536            1.0509         7.41
IVF-Binary-1024-nl316-np25-rf20-pca (query)              425.23       229.39       654.62       0.7374          1.0261            1.0222         7.41
IVF-Binary-1024-nl316-pca (self)                         425.23       461.22       886.45       0.6114          1.0575            1.0543         7.41
IVF-Binary-256-nl158-np7-rf0-sign (query)                212.93       137.35       350.28       0.1042          1.6406            1.6481         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               212.93       140.43       353.37       0.1041          1.6411            1.6484         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               212.93       147.79       360.72       0.1041          1.6412            1.6485         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               212.93       174.25       387.18       0.3771          1.1355            1.1269         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               212.93       304.34       517.27       0.5286          1.0742            1.0643         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              212.93       176.21       389.14       0.3754          1.1359            1.1272         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              212.93       307.75       520.68       0.5283          1.0743            1.0643         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              212.93       186.00       398.93       0.3752          1.1359            1.1272         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              212.93       309.40       522.33       0.5281          1.0743            1.0644         1.68
IVF-Binary-256-nl158-sign (self)                         212.93       481.51       694.44       0.3957          1.1432            1.1389         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               267.61       139.24       406.85       0.1045          1.6391            1.6451         1.74
IVF-Binary-256-nl223-np14-rf0-sign (query)               267.61       142.45       410.06       0.1044          1.6403            1.6460         1.74
IVF-Binary-256-nl223-np21-rf0-sign (query)               267.61       146.59       414.20       0.1043          1.6406            1.6464         1.74
IVF-Binary-256-nl223-np11-rf10-sign (query)              267.61       179.69       447.31       0.3865          1.1297            1.1209         1.74
IVF-Binary-256-nl223-np11-rf20-sign (query)              267.61       304.54       572.15       0.5363          1.0715            1.0626         1.74
IVF-Binary-256-nl223-np14-rf10-sign (query)              267.61       176.99       444.60       0.3859          1.1299            1.1214         1.74
IVF-Binary-256-nl223-np14-rf20-sign (query)              267.61       311.69       579.30       0.5353          1.0718            1.0629         1.74
IVF-Binary-256-nl223-np21-rf10-sign (query)              267.61       182.11       449.73       0.3858          1.1300            1.1214         1.74
IVF-Binary-256-nl223-np21-rf20-sign (query)              267.61       311.55       579.17       0.5350          1.0719            1.0629         1.74
IVF-Binary-256-nl223-sign (self)                         267.61       477.10       744.71       0.4055          1.1379            1.1334         1.74
IVF-Binary-256-nl316-np15-rf0-sign (query)               332.68       140.90       473.58       0.1050          1.6357            1.6455         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               332.68       141.36       474.04       0.1049          1.6364            1.6457         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               332.68       146.84       479.53       0.1048          1.6371            1.6464         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              332.68       180.01       512.70       0.3920          1.1266            1.1191         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              332.68       304.43       637.11       0.5390          1.0706            1.0621         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              332.68       178.93       511.62       0.3913          1.1269            1.1193         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              332.68       308.16       640.84       0.5381          1.0709            1.0622         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              332.68       184.45       517.13       0.3911          1.1270            1.1193         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              332.68       310.83       643.52       0.5378          1.0709            1.0623         1.84
IVF-Binary-256-nl316-sign (self)                         332.68       483.39       816.08       0.4107          1.1342            1.1317         1.84
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
Exhaustive (query)                                        68.08     1_376.48     1_444.56       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.08     4_670.09     4_738.17       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                132.21       267.68       399.89       0.0733          1.4884            1.4968         2.03
ExhaustiveBinary-256-random-rf10 (query)                 132.21       393.93       526.14       0.2947          1.1327            1.1260         2.03
ExhaustiveBinary-256-random-rf20 (query)                 132.21       526.57       658.77       0.4174          1.0830            1.0716         2.03
ExhaustiveBinary-256-random (self)                       132.21     1_254.70     1_386.90       0.3150          1.1321            1.1268         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   223.33       263.22       486.55       0.0722          1.4941            1.5001         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    223.33       395.72       619.05       0.2934          1.1324            1.1267         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    223.33       536.40       759.72       0.4181          1.0813            1.0715         2.03
ExhaustiveBinary-256-pca (self)                          223.33     1_244.39     1_467.71       0.3112          1.1338            1.1298         2.03
ExhaustiveBinary-512-random_no_rr (query)                204.50       386.82       591.33       0.1110          1.4064            1.4153         4.05
ExhaustiveBinary-512-random-rf10 (query)                 204.50       529.16       733.67       0.3695          1.0935            1.0919         4.05
ExhaustiveBinary-512-random-rf20 (query)                 204.50       665.88       870.38       0.4981          1.0544            1.0517         4.05
ExhaustiveBinary-512-random (self)                       204.50     1_705.81     1_910.32       0.3856          1.0953            1.0998         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   290.34       388.51       678.85       0.1063          1.4156            1.4272         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    290.34       528.00       818.34       0.3574          1.0990            1.0958         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    290.34       666.64       956.98       0.4853          1.0582            1.0544         4.05
ExhaustiveBinary-512-pca (self)                          290.34     1_703.97     1_994.32       0.3753          1.0995            1.1040         4.05
ExhaustiveBinary-1024-random_no_rr (query)               256.39       598.99       855.38       0.1593          1.3242            1.3318         8.11
ExhaustiveBinary-1024-random-rf10 (query)                256.39       747.24     1_003.64       0.4456          1.0660            1.0678         8.11
ExhaustiveBinary-1024-random-rf20 (query)                256.39       893.85     1_150.24       0.5824          1.0370            1.0362         8.11
ExhaustiveBinary-1024-random (self)                      256.39     2_467.90     2_724.29       0.4595          1.0714            1.0740         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  346.44       598.37       944.81       0.1599          1.3236            1.3333         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   346.44       746.23     1_092.67       0.4446          1.0658            1.0680         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   346.44       890.58     1_237.02       0.5812          1.0370            1.0362         8.11
ExhaustiveBinary-1024-pca (self)                         346.44     2_451.49     2_797.93       0.4582          1.0716            1.0746         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   84.51       663.09       747.60       0.1292          1.3815            1.3877         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    84.51       733.58       818.09       0.3927          1.0844            1.0833         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    84.51     1_100.11     1_184.62       0.5336          1.0464            1.0444         3.05
ExhaustiveBinary-512-sign (self)                          84.51     2_335.50     2_420.01       0.4063          1.0885            1.0916         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)              415.21        75.44       490.65       0.0763          1.4794            1.4941         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)             415.21        78.82       494.02       0.0762          1.4798            1.4942         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)             415.21        78.14       493.34       0.0762          1.4798            1.4942         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)             415.21       155.75       570.95       0.3000          1.1310            1.1254         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)             415.21       251.60       666.80       0.4213          1.0820            1.0713         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)            415.21       155.08       570.28       0.2985          1.1313            1.1255         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)            415.21       244.98       660.18       0.4202          1.0821            1.0714         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)            415.21       154.28       569.48       0.2985          1.1313            1.1255         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)            415.21       246.33       661.54       0.4201          1.0821            1.0714         2.34
IVF-Binary-256-nl158-random (self)                       415.21       301.36       716.56       0.3184          1.1308            1.1264         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)             425.80        72.31       498.12       0.0885          1.4508            1.4557         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)             425.80        74.21       500.02       0.0885          1.4511            1.4559         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)             425.80        77.55       503.35       0.0885          1.4511            1.4559         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)            425.80       159.68       585.48       0.3263          1.1151            1.1083         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)            425.80       254.28       680.09       0.4512          1.0704            1.0628         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)            425.80       159.44       585.25       0.3262          1.1151            1.1083         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)            425.80       253.50       679.30       0.4511          1.0704            1.0628         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)            425.80       162.11       587.91       0.3262          1.1151            1.1083         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)            425.80       260.03       685.83       0.4511          1.0704            1.0628         2.46
IVF-Binary-256-nl223-random (self)                       425.80       334.24       760.05       0.3439          1.1148            1.1139         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)             534.78        78.85       613.63       0.0950          1.4366            1.4392         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)             534.78        79.65       614.43       0.0950          1.4366            1.4392         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)             534.78        82.69       617.47       0.0950          1.4367            1.4392         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)            534.78       168.66       703.43       0.3368          1.1084            1.1032         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)            534.78       259.00       793.78       0.4647          1.0652            1.0593         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)            534.78       164.93       699.71       0.3367          1.1084            1.1032         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)            534.78       260.00       794.78       0.4646          1.0652            1.0593         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)            534.78       169.11       703.89       0.3366          1.1084            1.1032         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)            534.78       263.44       798.22       0.4646          1.0652            1.0593         2.65
IVF-Binary-256-nl316-random (self)                       534.78       350.74       885.52       0.3558          1.1069            1.1093         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)                 506.38        65.65       572.03       0.0755          1.4847            1.4975         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)                506.38        69.89       576.27       0.0754          1.4851            1.4976         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)                506.38        69.32       575.70       0.0754          1.4851            1.4976         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)                506.38       152.31       658.69       0.2991          1.1304            1.1259         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)                506.38       244.18       750.56       0.4222          1.0801            1.0710         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)               506.38       150.86       657.24       0.2978          1.1306            1.1260         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)               506.38       247.89       754.27       0.4210          1.0803            1.0711         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)               506.38       153.44       659.82       0.2978          1.1306            1.1260         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)               506.38       251.58       757.96       0.4209          1.0803            1.0711         2.34
IVF-Binary-256-nl158-pca (self)                          506.38       298.59       804.97       0.3154          1.1320            1.1293         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)                519.68        72.77       592.46       0.0881          1.4529            1.4576         2.46
IVF-Binary-256-nl223-np14-rf0-pca (query)                519.68        73.64       593.32       0.0880          1.4531            1.4578         2.46
IVF-Binary-256-nl223-np21-rf0-pca (query)                519.68        78.32       598.01       0.0880          1.4531            1.4578         2.46
IVF-Binary-256-nl223-np11-rf10-pca (query)               519.68       163.24       682.92       0.3292          1.1121            1.1063         2.46
IVF-Binary-256-nl223-np11-rf20-pca (query)               519.68       256.44       776.12       0.4516          1.0692            1.0621         2.46
IVF-Binary-256-nl223-np14-rf10-pca (query)               519.68       160.73       680.41       0.3291          1.1121            1.1063         2.46
IVF-Binary-256-nl223-np14-rf20-pca (query)               519.68       255.45       775.13       0.4515          1.0693            1.0621         2.46
IVF-Binary-256-nl223-np21-rf10-pca (query)               519.68       163.25       682.93       0.3291          1.1121            1.1063         2.46
IVF-Binary-256-nl223-np21-rf20-pca (query)               519.68       259.50       779.18       0.4515          1.0693            1.0621         2.46
IVF-Binary-256-nl223-pca (self)                          519.68       327.01       846.69       0.3476          1.1104            1.1128         2.46
IVF-Binary-256-nl316-np15-rf0-pca (query)                618.00        79.18       697.18       0.0950          1.4351            1.4365         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)                618.00        79.52       697.52       0.0950          1.4352            1.4366         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)                618.00        82.78       700.78       0.0950          1.4352            1.4366         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)               618.00       167.63       785.63       0.3402          1.1056            1.1005         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)               618.00       260.44       878.44       0.4617          1.0661            1.0595         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)               618.00       166.55       784.55       0.3401          1.1056            1.1005         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)               618.00       259.98       877.98       0.4616          1.0661            1.0595         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)               618.00       168.59       786.59       0.3401          1.1056            1.1005         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)               618.00       262.56       880.56       0.4616          1.0661            1.0595         2.65
IVF-Binary-256-nl316-pca (self)                          618.00       356.42       974.41       0.3590          1.1032            1.1073         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)              493.27        92.01       585.28       0.1124          1.4040            1.4144         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)             493.27        94.23       587.50       0.1124          1.4041            1.4144         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)             493.27        98.27       591.54       0.1124          1.4041            1.4144         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)             493.27       182.64       675.91       0.3712          1.0931            1.0918         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)             493.27       280.60       773.87       0.4994          1.0541            1.0516         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)            493.27       183.82       677.09       0.3709          1.0932            1.0918         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)            493.27       289.17       782.44       0.4992          1.0541            1.0516         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)            493.27       190.94       684.21       0.3709          1.0932            1.0918         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)            493.27       293.94       787.21       0.4992          1.0541            1.0516         4.36
IVF-Binary-512-nl158-random (self)                       493.27       431.50       924.77       0.3868          1.0949            1.0997         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)             507.25       100.43       607.69       0.1208          1.3872            1.3942         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)             507.25       100.62       607.87       0.1208          1.3873            1.3943         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)             507.25       106.29       613.55       0.1208          1.3873            1.3943         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)            507.25       191.91       699.17       0.3843          1.0875            1.0869         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)            507.25       287.75       795.01       0.5124          1.0510            1.0489         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)            507.25       191.77       699.03       0.3842          1.0875            1.0869         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)            507.25       293.11       800.36       0.5123          1.0510            1.0489         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)            507.25       201.50       708.75       0.3842          1.0875            1.0869         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)            507.25       313.91       821.16       0.5123          1.0510            1.0489         4.49
IVF-Binary-512-nl223-random (self)                       507.25       452.59       959.85       0.3990          1.0902            1.0949         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)             604.65       106.41       711.06       0.1243          1.3793            1.3847         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)             604.65       106.84       711.48       0.1243          1.3794            1.3847         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)             604.65       110.77       715.41       0.1243          1.3794            1.3847         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)            604.65       199.91       804.56       0.3889          1.0856            1.0850         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)            604.65       295.91       900.55       0.5162          1.0501            1.0479         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)            604.65       196.54       801.18       0.3888          1.0856            1.0850         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)            604.65       295.34       899.99       0.5162          1.0501            1.0479         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)            604.65       204.63       809.27       0.3888          1.0856            1.0850         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)            604.65       298.85       903.50       0.5162          1.0501            1.0479         4.67
IVF-Binary-512-nl316-random (self)                       604.65       474.88     1_079.53       0.4032          1.0885            1.0932         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)                 586.77        91.60       678.37       0.1079          1.4130            1.4261         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)                586.77        95.18       681.95       0.1079          1.4130            1.4261         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)                586.77        97.28       684.05       0.1079          1.4130            1.4261         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)                586.77       183.73       770.50       0.3595          1.0984            1.0956         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)                586.77       279.24       866.02       0.4868          1.0578            1.0543         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)               586.77       183.34       770.11       0.3591          1.0984            1.0956         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)               586.77       283.39       870.16       0.4865          1.0579            1.0543         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)               586.77       187.63       774.40       0.3591          1.0984            1.0956         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)               586.77       288.42       875.19       0.4865          1.0579            1.0543         4.36
IVF-Binary-512-nl158-pca (self)                          586.77       437.37     1_024.14       0.3769          1.0989            1.1039         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)                597.03       100.20       697.23       0.1169          1.3942            1.4010         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)                597.03       100.75       697.78       0.1169          1.3942            1.4010         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)                597.03       109.51       706.54       0.1169          1.3942            1.4010         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)               597.03       189.68       786.71       0.3738          1.0919            1.0898         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)               597.03       289.32       886.35       0.5015          1.0541            1.0513         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)               597.03       191.44       788.47       0.3738          1.0919            1.0898         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)               597.03       288.00       885.03       0.5015          1.0541            1.0513         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)               597.03       194.64       791.67       0.3738          1.0919            1.0898         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)               597.03       300.61       897.64       0.5015          1.0541            1.0513         4.49
IVF-Binary-512-nl223-pca (self)                          597.03       452.19     1_049.22       0.3906          1.0936            1.0982         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)                701.32       105.99       807.31       0.1208          1.3851            1.3903         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)                701.32       109.03       810.35       0.1208          1.3851            1.3903         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)                701.32       110.59       811.90       0.1208          1.3851            1.3903         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)               701.32       197.68       899.00       0.3799          1.0892            1.0879         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)               701.32       293.71       995.03       0.5061          1.0528            1.0504         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)               701.32       195.40       896.72       0.3799          1.0892            1.0879         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)               701.32       295.70       997.02       0.5060          1.0528            1.0505         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)               701.32       200.16       901.47       0.3799          1.0892            1.0879         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)               701.32       302.77     1_004.08       0.5060          1.0528            1.0505         4.67
IVF-Binary-512-nl316-pca (self)                          701.32       476.48     1_177.80       0.3958          1.0911            1.0960         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)             551.52       149.15       700.67       0.1598          1.3236            1.3316         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)            551.52       148.97       700.49       0.1598          1.3236            1.3316         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)            551.52       153.91       705.43       0.1598          1.3236            1.3316         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)            551.52       239.71       791.23       0.4461          1.0659            1.0678         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)            551.52       341.00       892.52       0.5827          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)           551.52       242.26       793.78       0.4461          1.0659            1.0678         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)           551.52       345.19       896.71       0.5827          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)           551.52       247.90       799.42       0.4461          1.0659            1.0678         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)           551.52       358.09       909.61       0.5827          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-random (self)                      551.52       660.12     1_211.64       0.4600          1.0713            1.0740         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)            561.09       155.66       716.75       0.1639          1.3163            1.3250         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)            561.09       158.07       719.16       0.1639          1.3163            1.3250         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)            561.09       162.15       723.24       0.1639          1.3163            1.3250         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)           561.09       249.66       810.75       0.4533          1.0639            1.0658         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)           561.09       348.21       909.31       0.5891          1.0358            1.0351         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)           561.09       250.92       812.01       0.4533          1.0639            1.0658         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)           561.09       356.46       917.55       0.5891          1.0358            1.0351         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)           561.09       262.13       823.23       0.4533          1.0639            1.0658         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)           561.09       364.96       926.05       0.5891          1.0358            1.0351         8.54
IVF-Binary-1024-nl223-random (self)                      561.09       671.39     1_232.48       0.4673          1.0692            1.0718         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)            660.39       163.36       823.75       0.1655          1.3134            1.3215         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)            660.39       160.83       821.22       0.1655          1.3134            1.3215         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)            660.39       168.76       829.15       0.1655          1.3134            1.3215         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)           660.39       256.18       916.57       0.4550          1.0634            1.0651         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)           660.39       357.97     1_018.35       0.5912          1.0356            1.0348         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)           660.39       255.84       916.22       0.4550          1.0634            1.0651         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)           660.39       362.04     1_022.42       0.5912          1.0356            1.0348         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)           660.39       265.26       925.64       0.4550          1.0634            1.0651         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)           660.39       370.48     1_030.86       0.5912          1.0356            1.0348         8.72
IVF-Binary-1024-nl316-random (self)                      660.39       693.23     1_353.61       0.4694          1.0686            1.0714         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)                637.75       147.26       785.01       0.1605          1.3229            1.3332         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)               637.75       148.96       786.71       0.1605          1.3229            1.3332         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)               637.75       154.16       791.91       0.1605          1.3229            1.3332         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)               637.75       239.78       877.53       0.4452          1.0657            1.0680         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)               637.75       340.22       977.97       0.5817          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)              637.75       242.61       880.36       0.4452          1.0657            1.0680         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)              637.75       347.46       985.21       0.5817          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)              637.75       248.21       885.96       0.4452          1.0657            1.0680         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)              637.75       353.36       991.11       0.5817          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-pca (self)                         637.75       652.05     1_289.80       0.4588          1.0715            1.0746         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)               649.73       155.80       805.53       0.1641          1.3158            1.3264         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)               649.73       155.08       804.81       0.1641          1.3158            1.3264         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)               649.73       166.16       815.89       0.1641          1.3158            1.3264         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)              649.73       252.19       901.92       0.4515          1.0638            1.0664         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)              649.73       356.95     1_006.68       0.5882          1.0359            1.0353         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)              649.73       252.93       902.66       0.4515          1.0638            1.0664         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)              649.73       355.39     1_005.12       0.5882          1.0359            1.0353         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)              649.73       260.39       910.12       0.4515          1.0638            1.0664         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)              649.73       363.65     1_013.38       0.5882          1.0359            1.0353         8.54
IVF-Binary-1024-nl223-pca (self)                         649.73       677.03     1_326.76       0.4657          1.0695            1.0725         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)               757.29       165.13       922.42       0.1656          1.3134            1.3240         8.72
IVF-Binary-1024-nl316-np17-rf0-pca (query)               757.29       161.86       919.15       0.1656          1.3134            1.3240         8.72
IVF-Binary-1024-nl316-np25-rf0-pca (query)               757.29       169.27       926.57       0.1656          1.3134            1.3240         8.72
IVF-Binary-1024-nl316-np15-rf10-pca (query)              757.29       258.36     1_015.65       0.4539          1.0631            1.0657         8.72
IVF-Binary-1024-nl316-np15-rf20-pca (query)              757.29       362.02     1_119.32       0.5899          1.0356            1.0350         8.72
IVF-Binary-1024-nl316-np17-rf10-pca (query)              757.29       256.56     1_013.86       0.4539          1.0631            1.0657         8.72
IVF-Binary-1024-nl316-np17-rf20-pca (query)              757.29       360.86     1_118.16       0.5899          1.0356            1.0350         8.72
IVF-Binary-1024-nl316-np25-rf10-pca (query)              757.29       267.26     1_024.56       0.4539          1.0631            1.0657         8.72
IVF-Binary-1024-nl316-np25-rf20-pca (query)              757.29       369.22     1_126.52       0.5899          1.0356            1.0350         8.72
IVF-Binary-1024-nl316-pca (self)                         757.29       698.70     1_456.00       0.4677          1.0689            1.0720         8.72
IVF-Binary-512-nl158-np7-rf0-sign (query)                375.33       259.99       635.32       0.1292          1.3813            1.3870         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)               375.33       264.79       640.12       0.1292          1.3813            1.3870         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)               375.33       265.56       640.89       0.1292          1.3813            1.3870         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)               375.33       327.55       702.89       0.3940          1.0840            1.0833         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)               375.33       584.98       960.32       0.5343          1.0463            1.0444         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)              375.33       331.99       707.32       0.3939          1.0840            1.0833         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)              375.33       608.42       983.75       0.5343          1.0463            1.0444         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)              375.33       333.91       709.24       0.3939          1.0840            1.0833         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)              375.33       602.82       978.16       0.5343          1.0463            1.0444         3.36
IVF-Binary-512-nl158-sign (self)                         375.33       892.69     1_268.03       0.4072          1.0883            1.0915         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               382.22       264.35       646.56       0.1295          1.3814            1.3865         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               382.22       265.68       647.90       0.1295          1.3814            1.3865         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               382.22       271.18       653.39       0.1295          1.3814            1.3865         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              382.22       333.73       715.94       0.3991          1.0819            1.0817         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              382.22       588.70       970.91       0.5376          1.0456            1.0440         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              382.22       335.86       718.08       0.3991          1.0819            1.0817         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              382.22       598.70       980.92       0.5375          1.0456            1.0440         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              382.22       340.63       722.85       0.3991          1.0819            1.0817         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              382.22       599.65       981.87       0.5375          1.0456            1.0440         3.49
IVF-Binary-512-nl223-sign (self)                         382.22       910.52     1_292.73       0.4125          1.0864            1.0898         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)               486.14       269.76       755.91       0.1294          1.3810            1.3867         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)               486.14       270.48       756.62       0.1294          1.3810            1.3867         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)               486.14       275.70       761.84       0.1294          1.3810            1.3867         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)              486.14       349.92       836.07       0.4000          1.0814            1.0813         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)              486.14       595.57     1_081.71       0.5386          1.0452            1.0437         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)              486.14       344.98       831.12       0.4000          1.0814            1.0813         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)              486.14       601.03     1_087.17       0.5385          1.0452            1.0437         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)              486.14       346.68       832.82       0.4000          1.0814            1.0813         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)              486.14       605.64     1_091.79       0.5385          1.0452            1.0437         3.67
IVF-Binary-512-nl316-sign (self)                         486.14       932.54     1_418.68       0.4132          1.0861            1.0894         3.67
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
Exhaustive (query)                                       100.47     1_958.35     2_058.82       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.47     6_498.06     6_598.54       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                191.61       286.84       478.45       0.0662          1.3769            1.3786         2.28
ExhaustiveBinary-256-random-rf10 (query)                 191.61       425.48       617.09       0.2741          1.1112            1.1017         2.28
ExhaustiveBinary-256-random-rf20 (query)                 191.61       579.17       770.79       0.3877          1.0706            1.0590         2.28
ExhaustiveBinary-256-random (self)                       191.61     1_328.74     1_520.36       0.2874          1.1077            1.0994         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   391.76       285.44       677.20       0.0653          1.3793            1.3770         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    391.76       428.48       820.24       0.2701          1.1138            1.1018         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    391.76       579.65       971.42       0.3852          1.0726            1.0585         2.28
ExhaustiveBinary-256-pca (self)                          391.76     1_334.33     1_726.09       0.2820          1.1100            1.0990         2.28
ExhaustiveBinary-512-random_no_rr (query)                299.89       410.53       710.42       0.0934          1.3249            1.3316         4.55
ExhaustiveBinary-512-random-rf10 (query)                 299.89       567.44       867.33       0.3220          1.0860            1.0806         4.55
ExhaustiveBinary-512-random-rf20 (query)                 299.89       756.72     1_056.62       0.4345          1.0527            1.0485         4.55
ExhaustiveBinary-512-random (self)                       299.89     1_896.23     2_196.12       0.3344          1.0826            1.0842         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   487.08       411.41       898.50       0.0951          1.3226            1.3263         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    487.08       567.78     1_054.86       0.3246          1.0847            1.0788         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    487.08       732.89     1_219.97       0.4400          1.0517            1.0468         4.55
ExhaustiveBinary-512-pca (self)                          487.08     1_826.33     2_313.41       0.3361          1.0819            1.0825         4.55
ExhaustiveBinary-1024-random_no_rr (query)               501.50       633.41     1_134.90       0.1319          1.2685            1.2723         9.11
ExhaustiveBinary-1024-random-rf10 (query)                501.50       813.17     1_314.67       0.3742          1.0644            1.0666         9.11
ExhaustiveBinary-1024-random-rf20 (query)                501.50       992.71     1_494.20       0.4906          1.0386            1.0390         9.11
ExhaustiveBinary-1024-random (self)                      501.50     2_656.85     3_158.35       0.3823          1.0666            1.0715         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  695.45       642.53     1_337.99       0.1355          1.2623            1.2651         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   695.45       812.84     1_508.29       0.3804          1.0622            1.0643         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   695.45       989.86     1_685.31       0.4993          1.0369            1.0374         9.11
ExhaustiveBinary-1024-pca (self)                         695.45     2_656.20     3_351.65       0.3870          1.0651            1.0695         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  129.06       837.85       966.91       0.1284          1.2822            1.2821         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   129.06       921.43     1_050.48       0.3618          1.0706            1.0699         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   129.06     1_417.16     1_546.22       0.4847          1.0407            1.0395         4.58
ExhaustiveBinary-768-sign (self)                         129.06     2_975.94     3_105.00       0.3694          1.0716            1.0747         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)              547.09        93.99       641.07       0.0686          1.3708            1.3769         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)             547.09        98.29       645.38       0.0685          1.3711            1.3770         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)             547.09       101.36       648.44       0.0685          1.3711            1.3770         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)             547.09       202.74       749.82       0.2782          1.1103            1.1014         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)             547.09       317.79       864.88       0.3901          1.0700            1.0589         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)            547.09       199.63       746.72       0.2770          1.1104            1.1014         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)            547.09       317.97       865.06       0.3894          1.0701            1.0589         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)            547.09       200.21       747.29       0.2769          1.1104            1.1014         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)            547.09       320.33       867.41       0.3893          1.0701            1.0589         2.74
IVF-Binary-256-nl158-random (self)                       547.09       388.14       935.22       0.2904          1.1068            1.0992         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)             580.47        95.16       675.63       0.0783          1.3507            1.3543         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)             580.47        94.53       675.00       0.0783          1.3508            1.3543         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)             580.47        96.79       677.26       0.0783          1.3508            1.3543         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)            580.47       202.30       782.77       0.3028          1.0946            1.0866         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)            580.47       316.59       897.06       0.4175          1.0588            1.0517         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)            580.47       207.34       787.81       0.3028          1.0946            1.0866         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)            580.47       318.87       899.34       0.4174          1.0588            1.0517         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)            580.47       203.54       784.01       0.3028          1.0946            1.0866         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)            580.47       324.43       904.90       0.4174          1.0588            1.0517         2.93
IVF-Binary-256-nl223-random (self)                       580.47       437.40     1_017.87       0.3172          1.0895            1.0882         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)             706.48       105.15       811.63       0.0851          1.3378            1.3389         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)             706.48       104.36       810.84       0.0851          1.3379            1.3390         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)             706.48       106.91       813.39       0.0851          1.3380            1.3390         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)            706.48       215.24       921.72       0.3176          1.0865            1.0809         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)            706.48       331.93     1_038.41       0.4309          1.0543            1.0490         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)            706.48       217.27       923.76       0.3175          1.0865            1.0809         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)            706.48       330.87     1_037.36       0.4308          1.0543            1.0490         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)            706.48       215.51       921.99       0.3174          1.0865            1.0809         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)            706.48       334.66     1_041.15       0.4307          1.0543            1.0490         3.21
IVF-Binary-256-nl316-random (self)                       706.48       474.59     1_181.07       0.3311          1.0807            1.0831         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)                 760.91        84.33       845.24       0.0678          1.3725            1.3752         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)                760.91        87.54       848.44       0.0677          1.3726            1.3752         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)                760.91        87.93       848.84       0.0677          1.3726            1.3752         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)                760.91       193.99       954.90       0.2747          1.1123            1.1014         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)                760.91       317.79     1_078.69       0.3883          1.0717            1.0584         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)               760.91       191.33       952.24       0.2735          1.1124            1.1014         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)               760.91       321.73     1_082.64       0.3874          1.0717            1.0585         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)               760.91       192.94       953.85       0.2735          1.1124            1.1014         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)               760.91       311.25     1_072.15       0.3873          1.0717            1.0585         2.74
IVF-Binary-256-nl158-pca (self)                          760.91       391.80     1_152.71       0.2853          1.1087            1.0988         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)                778.09        93.20       871.29       0.0776          1.3536            1.3551         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)                778.09        94.07       872.17       0.0776          1.3536            1.3551         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)                778.09        96.44       874.53       0.0776          1.3536            1.3551         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)               778.09       203.47       981.56       0.2987          1.0962            1.0873         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)               778.09       319.93     1_098.02       0.4151          1.0600            1.0520         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)               778.09       202.04       980.13       0.2986          1.0962            1.0873         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)               778.09       328.03     1_106.12       0.4150          1.0600            1.0520         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)               778.09       206.07       984.16       0.2986          1.0962            1.0873         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)               778.09       343.61     1_121.70       0.4149          1.0600            1.0520         2.93
IVF-Binary-256-nl223-pca (self)                          778.09       426.25     1_204.35       0.3109          1.0907            1.0885         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)                904.07       103.05     1_007.12       0.0848          1.3398            1.3382         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)                904.07       104.29     1_008.36       0.0848          1.3399            1.3382         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)                904.07       107.17     1_011.24       0.0848          1.3399            1.3382         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)               904.07       214.40     1_118.47       0.3102          1.0896            1.0821         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)               904.07       329.34     1_233.41       0.4249          1.0567            1.0497         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)               904.07       212.33     1_116.40       0.3101          1.0897            1.0821         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)               904.07       341.44     1_245.52       0.4248          1.0567            1.0497         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)               904.07       216.11     1_120.18       0.3101          1.0897            1.0821         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)               904.07       334.26     1_238.33       0.4247          1.0567            1.0497         3.21
IVF-Binary-256-nl316-pca (self)                          904.07       472.23     1_376.30       0.3223          1.0839            1.0844         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)              666.51       122.32       788.84       0.0948          1.3227            1.3308         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)             666.51       124.55       791.07       0.0947          1.3228            1.3308         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)             666.51       127.39       793.90       0.0947          1.3228            1.3308         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)             666.51       235.11       901.62       0.3234          1.0857            1.0806         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)             666.51       359.81     1_026.32       0.4355          1.0524            1.0484         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)            666.51       236.10       902.61       0.3230          1.0857            1.0806         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)            666.51       357.34     1_023.85       0.4354          1.0524            1.0484         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)            666.51       238.97       905.48       0.3230          1.0857            1.0806         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)            666.51       360.12     1_026.63       0.4354          1.0524            1.0484         5.02
IVF-Binary-512-nl158-random (self)                       666.51       586.35     1_252.86       0.3354          1.0823            1.0842         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)             696.54       130.82       827.36       0.1036          1.3077            1.3089         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)             696.54       133.09       829.63       0.1036          1.3077            1.3089         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)             696.54       137.10       833.64       0.1036          1.3077            1.3089         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)            696.54       250.32       946.86       0.3368          1.0788            1.0763         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)            696.54       371.03     1_067.57       0.4484          1.0488            1.0463         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)            696.54       248.21       944.75       0.3368          1.0788            1.0763         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)            696.54       388.68     1_085.22       0.4484          1.0488            1.0463         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)            696.54       253.34       949.88       0.3368          1.0788            1.0763         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)            696.54       386.54     1_083.08       0.4484          1.0488            1.0463         5.21
IVF-Binary-512-nl223-random (self)                       696.54       626.64     1_323.18       0.3474          1.0766            1.0803         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)             812.69       146.77       959.46       0.1079          1.3002            1.2990         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)             812.69       143.56       956.25       0.1079          1.3002            1.2990         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)             812.69       148.60       961.29       0.1079          1.3002            1.2990         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)            812.69       263.36     1_076.04       0.3426          1.0764            1.0745         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)            812.69       386.38     1_199.07       0.4546          1.0469            1.0449         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)            812.69       267.72     1_080.40       0.3426          1.0764            1.0745         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)            812.69       393.33     1_206.02       0.4545          1.0469            1.0449         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)            812.69       266.63     1_079.31       0.3426          1.0764            1.0745         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)            812.69       430.82     1_243.51       0.4545          1.0469            1.0449         5.48
IVF-Binary-512-nl316-random (self)                       812.69       683.08     1_495.76       0.3530          1.0743            1.0786         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)                 915.00       126.67     1_041.68       0.0962          1.3205            1.3253         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)                915.00       128.33     1_043.33       0.0961          1.3206            1.3253         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)                915.00       131.96     1_046.96       0.0961          1.3206            1.3253         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)                915.00       235.94     1_150.94       0.3262          1.0845            1.0787         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)                915.00       361.74     1_276.75       0.4407          1.0516            1.0467         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)               915.00       237.63     1_152.63       0.3256          1.0845            1.0787         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)               915.00       356.96     1_271.96       0.4406          1.0516            1.0467         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)               915.00       251.65     1_166.65       0.3256          1.0845            1.0787         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)               915.00       359.26     1_274.26       0.4406          1.0516            1.0467         5.02
IVF-Binary-512-nl158-pca (self)                          915.00       582.69     1_497.70       0.3369          1.0817            1.0825         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)                944.09       132.35     1_076.44       0.1043          1.3066            1.3057         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)                944.09       138.12     1_082.21       0.1043          1.3066            1.3057         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)                944.09       141.96     1_086.05       0.1043          1.3066            1.3057         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)               944.09       254.80     1_198.89       0.3375          1.0787            1.0753         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)               944.09       367.70     1_311.79       0.4530          1.0478            1.0448         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)               944.09       247.33     1_191.42       0.3375          1.0787            1.0753         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)               944.09       366.74     1_310.83       0.4530          1.0478            1.0448         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)               944.09       257.06     1_201.15       0.3375          1.0787            1.0753         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)               944.09       399.89     1_343.97       0.4530          1.0478            1.0448         5.21
IVF-Binary-512-nl223-pca (self)                          944.09       597.71     1_541.79       0.3473          1.0770            1.0796         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)              1_016.90       143.77     1_160.66       0.1081          1.2992            1.2963         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)              1_016.90       141.03     1_157.93       0.1081          1.2993            1.2963         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)              1_016.90       146.19     1_163.08       0.1081          1.2993            1.2963         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)             1_016.90       253.29     1_270.19       0.3443          1.0757            1.0732         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)             1_016.90       377.18     1_394.08       0.4586          1.0463            1.0439         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)             1_016.90       251.23     1_268.12       0.3442          1.0757            1.0732         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)             1_016.90       382.48     1_399.38       0.4585          1.0463            1.0439         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)             1_016.90       256.14     1_273.04       0.3442          1.0757            1.0732         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)             1_016.90       383.68     1_400.58       0.4585          1.0463            1.0439         5.48
IVF-Binary-512-nl316-pca (self)                        1_016.90       644.20     1_661.09       0.3530          1.0746            1.0780         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)             862.25       199.08     1_061.33       0.1327          1.2679            1.2722         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)            862.25       198.71     1_060.96       0.1327          1.2680            1.2722         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)            862.25       202.75     1_064.99       0.1327          1.2680            1.2722         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)            862.25       316.02     1_178.27       0.3746          1.0644            1.0666         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)            862.25       449.29     1_311.54       0.4908          1.0386            1.0390         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)           862.25       320.11     1_182.36       0.3745          1.0644            1.0666         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)           862.25       460.99     1_323.23       0.4907          1.0386            1.0390         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)           862.25       335.94     1_198.19       0.3745          1.0644            1.0666         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)           862.25       465.44     1_327.68       0.4907          1.0386            1.0390         9.57
IVF-Binary-1024-nl158-random (self)                      862.25       860.50     1_722.75       0.3825          1.0665            1.0715         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)            890.50       208.97     1_099.47       0.1370          1.2612            1.2658         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)            890.50       207.24     1_097.74       0.1370          1.2612            1.2658         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)            890.50       212.10     1_102.61       0.1370          1.2612            1.2658         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)           890.50       334.62     1_225.12       0.3801          1.0625            1.0654         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)           890.50       463.98     1_354.48       0.4966          1.0375            1.0380         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)           890.50       335.20     1_225.71       0.3801          1.0625            1.0654         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)           890.50       474.78     1_365.28       0.4966          1.0375            1.0380         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)           890.50       347.85     1_238.35       0.3801          1.0625            1.0654         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)           890.50       480.58     1_371.08       0.4966          1.0375            1.0380         9.76
IVF-Binary-1024-nl223-random (self)                      890.50       910.41     1_800.91       0.3880          1.0649            1.0699         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)          1_075.78       224.66     1_300.44       0.1388          1.2580            1.2631        10.03
IVF-Binary-1024-nl316-np17-rf0-random (query)          1_075.78       219.91     1_295.69       0.1388          1.2580            1.2631        10.03
IVF-Binary-1024-nl316-np25-rf0-random (query)          1_075.78       225.83     1_301.60       0.1388          1.2580            1.2631        10.03
IVF-Binary-1024-nl316-np15-rf10-random (query)         1_075.78       348.39     1_424.17       0.3834          1.0615            1.0643        10.03
IVF-Binary-1024-nl316-np15-rf20-random (query)         1_075.78       481.76     1_557.54       0.5004          1.0368            1.0374        10.03
IVF-Binary-1024-nl316-np17-rf10-random (query)         1_075.78       354.06     1_429.84       0.3834          1.0615            1.0643        10.03
IVF-Binary-1024-nl316-np17-rf20-random (query)         1_075.78       485.74     1_561.51       0.5004          1.0368            1.0374        10.03
IVF-Binary-1024-nl316-np25-rf10-random (query)         1_075.78       373.55     1_449.33       0.3834          1.0615            1.0643        10.03
IVF-Binary-1024-nl316-np25-rf20-random (query)         1_075.78       511.63     1_587.41       0.5004          1.0368            1.0374        10.03
IVF-Binary-1024-nl316-random (self)                    1_075.78       951.42     2_027.20       0.3911          1.0639            1.0689        10.03
IVF-Binary-1024-nl158-np7-rf0-pca (query)              1_094.70       201.90     1_296.59       0.1360          1.2618            1.2650         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)             1_094.70       200.26     1_294.96       0.1360          1.2618            1.2650         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)             1_094.70       205.42     1_300.12       0.1360          1.2618            1.2650         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)             1_094.70       321.72     1_416.42       0.3808          1.0621            1.0643         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)             1_094.70       453.14     1_547.83       0.4995          1.0369            1.0374         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)            1_094.70       333.61     1_428.30       0.3807          1.0621            1.0643         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)            1_094.70       460.30     1_555.00       0.4995          1.0369            1.0374         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)            1_094.70       332.05     1_426.75       0.3807          1.0621            1.0643         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)            1_094.70       472.46     1_567.16       0.4995          1.0369            1.0374         9.57
IVF-Binary-1024-nl158-pca (self)                       1_094.70       878.19     1_972.89       0.3871          1.0651            1.0695         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_096.48       213.88     1_310.36       0.1396          1.2560            1.2601         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_096.48       209.63     1_306.12       0.1396          1.2560            1.2601         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_096.48       213.55     1_310.03       0.1396          1.2560            1.2601         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_096.48       336.30     1_432.78       0.3857          1.0604            1.0631         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_096.48       465.53     1_562.01       0.5050          1.0360            1.0365         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_096.48       338.10     1_434.59       0.3857          1.0604            1.0631         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_096.48       473.12     1_569.60       0.5050          1.0360            1.0365         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_096.48       344.87     1_441.35       0.3857          1.0604            1.0631         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_096.48       483.35     1_579.84       0.5050          1.0360            1.0365         9.76
IVF-Binary-1024-nl223-pca (self)                       1_096.48       912.88     2_009.37       0.3923          1.0636            1.0682         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_219.27       221.65     1_440.92       0.1414          1.2531            1.2577        10.03
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_219.27       222.26     1_441.53       0.1414          1.2531            1.2577        10.03
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_219.27       226.45     1_445.72       0.1414          1.2531            1.2577        10.03
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_219.27       348.98     1_568.25       0.3890          1.0595            1.0623        10.03
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_219.27       480.83     1_700.10       0.5081          1.0354            1.0361        10.03
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_219.27       348.97     1_568.24       0.3890          1.0595            1.0623        10.03
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_219.27       483.48     1_702.75       0.5081          1.0354            1.0361        10.03
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_219.27       361.22     1_580.49       0.3890          1.0595            1.0623        10.03
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_219.27       492.88     1_712.15       0.5081          1.0354            1.0361        10.03
IVF-Binary-1024-nl316-pca (self)                       1_219.27       964.37     2_183.64       0.3955          1.0627            1.0672        10.03
IVF-Binary-768-nl158-np7-rf0-sign (query)                500.35       384.97       885.31       0.1284          1.2834            1.2814         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)               500.35       384.91       885.26       0.1284          1.2834            1.2814         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)               500.35       388.82       889.16       0.1284          1.2834            1.2814         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)               500.35       471.80       972.15       0.3625          1.0704            1.0698         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)               500.35       861.04     1_361.38       0.4852          1.0406            1.0396         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)              500.35       474.16       974.51       0.3625          1.0704            1.0698         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)              500.35       855.26     1_355.60       0.4852          1.0406            1.0396         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)              500.35       508.65     1_008.99       0.3625          1.0704            1.0698         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)              500.35       870.63     1_370.98       0.4852          1.0406            1.0396         5.04
IVF-Binary-768-nl158-sign (self)                         500.35     1_336.42     1_836.76       0.3700          1.0715            1.0746         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)               520.25       390.27       910.52       0.1285          1.2800            1.2812         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)               520.25       392.28       912.53       0.1285          1.2800            1.2812         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)               520.25       397.36       917.61       0.1285          1.2800            1.2812         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)              520.25       481.99     1_002.23       0.3665          1.0688            1.0688         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)              520.25       870.23     1_390.48       0.4875          1.0400            1.0392         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)              520.25       481.99     1_002.23       0.3665          1.0688            1.0688         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)              520.25       868.40     1_388.64       0.4874          1.0400            1.0392         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)              520.25       489.53     1_009.78       0.3665          1.0688            1.0688         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)              520.25       877.49     1_397.73       0.4874          1.0400            1.0392         5.23
IVF-Binary-768-nl223-sign (self)                         520.25     1_356.88     1_877.13       0.3733          1.0702            1.0737         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)               660.50       401.48     1_061.98       0.1282          1.2804            1.2812         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)               660.50       400.53     1_061.04       0.1282          1.2804            1.2812         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)               660.50       406.09     1_066.59       0.1282          1.2804            1.2812         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)              660.50       490.90     1_151.41       0.3668          1.0684            1.0687         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)              660.50       876.99     1_537.49       0.4884          1.0398            1.0392         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)              660.50       493.41     1_153.91       0.3668          1.0684            1.0687         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)              660.50       930.47     1_590.97       0.4884          1.0398            1.0392         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)              660.50       497.22     1_157.72       0.3668          1.0684            1.0687         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)              660.50       878.61     1_539.12       0.4884          1.0398            1.0392         5.51
IVF-Binary-768-nl316-sign (self)                         660.50     1_380.85     2_041.35       0.3742          1.0700            1.0735         5.51
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
Exhaustive (query)                                        33.16       762.04       795.19       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.16     2_539.90     2_573.06       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                 70.07       251.56       321.63       0.5519          1.8826            1.5884         1.78
ExhaustiveBinary-256-random-rf10 (query)                  70.07       371.70       441.76       0.9881          1.0022            1.0000         1.78
ExhaustiveBinary-256-random-rf20 (query)                  70.07       485.74       555.80       0.9980          1.0003            1.0000         1.78
ExhaustiveBinary-256-random (self)                        70.07     1_203.73     1_273.79       0.9881          1.0022            1.0000         1.78
ExhaustiveBinary-256-pca_no_rr (query)                    97.47       248.27       345.75       0.5930          1.6081            1.4152         1.78
ExhaustiveBinary-256-pca-rf10 (query)                     97.47       370.64       468.11       0.9919          1.0013            1.0000         1.78
ExhaustiveBinary-256-pca-rf20 (query)                     97.47       488.84       586.31       0.9988          1.0001            1.0000         1.78
ExhaustiveBinary-256-pca (self)                           97.47     1_214.86     1_312.33       0.9915          1.0014            1.0000         1.78
ExhaustiveBinary-512-random_no_rr (query)                 85.15       361.67       446.81       0.6306          1.5767            1.3633         3.55
ExhaustiveBinary-512-random-rf10 (query)                  85.15       491.83       576.98       0.9975          1.0004            1.0000         3.55
ExhaustiveBinary-512-random-rf20 (query)                  85.15       609.00       694.14       0.9998          1.0000            1.0000         3.55
ExhaustiveBinary-512-random (self)                        85.15     1_616.51     1_701.66       0.9973          1.0004            1.0000         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   108.67       365.29       473.96       0.6479          1.4884            1.3147         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    108.67       490.64       599.32       0.9983          1.0002            1.0000         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    108.67       612.98       721.65       0.9998          1.0000            1.0000         3.55
ExhaustiveBinary-512-pca (self)                          108.67     1_623.76     1_732.43       0.9981          1.0002            1.0000         3.55
ExhaustiveBinary-1024-random_no_rr (query)               116.69       550.44       667.12       0.6758          1.4452            1.2804         7.10
ExhaustiveBinary-1024-random-rf10 (query)                116.69       673.45       790.14       0.9995          1.0001            1.0000         7.10
ExhaustiveBinary-1024-random-rf20 (query)                116.69       799.49       916.17       0.9999          1.0000            1.0000         7.10
ExhaustiveBinary-1024-random (self)                      116.69     2_246.63     2_363.32       0.9993          1.0001            1.0000         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  141.60       543.18       684.78       0.6838          1.4142            1.2651         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   141.60       673.43       815.03       0.9996          1.0000            1.0000         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   141.60       830.64       972.24       1.0000          1.0000            1.0000         7.10
ExhaustiveBinary-1024-pca (self)                         141.60     2_337.59     2_479.19       0.9995          1.0001            1.0000         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   41.03       420.88       461.91       0.0376         19.4734           14.8778         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    41.03       455.86       496.89       0.1617          2.7567            2.6548         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    41.03       689.20       730.23       0.2739          1.9837            1.9249         1.53
ExhaustiveBinary-256-sign (self)                          41.03     1_480.04     1_521.07       0.1691          2.7353            2.6299         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)              431.81        58.12       489.92       0.5656          1.6695            1.5131         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)             431.81        67.31       499.12       0.5589          1.7299            1.5498         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)             431.81        77.78       509.59       0.5568          1.7640            1.5630         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)             431.81       120.62       552.42       0.9903          1.0018            1.0000         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)             431.81       176.96       608.76       0.9968          1.0006            1.0000         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)            431.81       130.27       562.08       0.9907          1.0016            1.0000         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)            431.81       188.60       620.40       0.9986          1.0002            1.0000         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)            431.81       134.74       566.54       0.9898          1.0018            1.0000         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)            431.81       197.37       629.18       0.9984          1.0002            1.0000         1.93
IVF-Binary-256-nl158-random (self)                       431.81       330.00       761.80       0.9904          1.0017            1.0000         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             505.17        51.98       557.15       0.5629          1.6756            1.5245         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             505.17        55.03       560.19       0.5606          1.7006            1.5423         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             505.17        63.20       568.37       0.5578          1.7444            1.5592         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            505.17       111.76       616.93       0.9912          1.0015            1.0000         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            505.17       169.88       675.05       0.9984          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            505.17       115.74       620.90       0.9909          1.0015            1.0000         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            505.17       177.39       682.55       0.9987          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            505.17       123.92       629.08       0.9900          1.0017            1.0000         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            505.17       186.72       691.88       0.9985          1.0002            1.0000         2.00
IVF-Binary-256-nl223-random (self)                       505.17       293.21       798.38       0.9908          1.0016            1.0000         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             651.46        52.88       704.35       0.5619          1.6821            1.5289         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             651.46        55.47       706.93       0.5608          1.6947            1.5368         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             651.46        62.43       713.90       0.5581          1.7362            1.5552         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            651.46       112.51       763.97       0.9917          1.0014            1.0000         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            651.46       171.82       823.29       0.9987          1.0002            1.0000         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            651.46       113.71       765.18       0.9914          1.0014            1.0000         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            651.46       172.58       824.04       0.9988          1.0002            1.0000         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            651.46       121.43       772.89       0.9904          1.0017            1.0000         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            651.46       180.98       832.45       0.9986          1.0002            1.0000         2.09
IVF-Binary-256-nl316-random (self)                       651.46       280.17       931.64       0.9912          1.0015            1.0000         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)                 455.98        49.89       505.87       0.6039          1.4880            1.3749         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)                455.98        62.00       517.98       0.5989          1.5218            1.3915         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)                455.98        70.05       526.03       0.5975          1.5419            1.3962         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)                455.98       114.21       570.19       0.9926          1.0013            1.0000         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)                455.98       171.05       627.03       0.9972          1.0006            1.0000         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)               455.98       123.39       579.37       0.9933          1.0010            1.0000         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)               455.98       183.98       639.96       0.9991          1.0001            1.0000         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)               455.98       133.71       589.69       0.9927          1.0011            1.0000         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)               455.98       195.67       651.65       0.9990          1.0001            1.0000         1.93
IVF-Binary-256-nl158-pca (self)                          455.98       332.18       788.16       0.9929          1.0011            1.0000         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                532.10        50.72       582.83       0.6019          1.4944            1.3798         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                532.10        55.72       587.82       0.6000          1.5083            1.3871         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                532.10        62.40       594.51       0.5980          1.5316            1.3961         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               532.10       111.67       643.77       0.9937          1.0010            1.0000         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               532.10       168.56       700.67       0.9987          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               532.10       116.05       648.16       0.9935          1.0010            1.0000         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               532.10       174.67       706.78       0.9991          1.0001            1.0000         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               532.10       122.60       654.70       0.9929          1.0011            1.0000         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               532.10       186.99       719.10       0.9990          1.0001            1.0000         2.00
IVF-Binary-256-nl223-pca (self)                          532.10       288.11       820.21       0.9931          1.0011            1.0000         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                663.40        54.66       718.07       0.6011          1.4974            1.3831         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                663.40        55.52       718.92       0.6002          1.5051            1.3864         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                663.40        60.48       723.88       0.5984          1.5259            1.3938         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               663.40       112.65       776.05       0.9939          1.0009            1.0000         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               663.40       170.18       833.58       0.9991          1.0001            1.0000         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               663.40       114.79       778.19       0.9938          1.0009            1.0000         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               663.40       173.40       836.80       0.9992          1.0001            1.0000         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               663.40       121.36       784.76       0.9931          1.0011            1.0000         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               663.40       182.97       846.38       0.9991          1.0001            1.0000         2.09
IVF-Binary-256-nl316-pca (self)                          663.40       308.55       971.95       0.9934          1.0011            1.0000         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)              446.58        70.29       516.87       0.6411          1.4472            1.3312         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)             446.58        83.07       529.65       0.6352          1.4895            1.3489         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)             446.58       101.18       547.76       0.6333          1.5129            1.3546         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)             446.58       137.19       583.77       0.9965          1.0007            1.0000         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)             446.58       195.49       642.07       0.9978          1.0005            1.0000         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)            446.58       150.00       596.58       0.9982          1.0002            1.0000         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)            446.58       216.07       662.65       0.9999          1.0000            1.0000         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)            446.58       166.55       613.13       0.9979          1.0003            1.0000         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)            446.58       228.08       674.65       0.9998          1.0000            1.0000         3.71
IVF-Binary-512-nl158-random (self)                       446.58       416.06       862.64       0.9979          1.0003            1.0000         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             526.45        69.63       596.07       0.6386          1.4527            1.3378         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             526.45        78.73       605.17       0.6366          1.4693            1.3444         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             526.45        87.93       614.37       0.6340          1.5000            1.3529         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            526.45       135.14       661.59       0.9978          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            526.45       194.97       721.41       0.9993          1.0001            1.0000         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            526.45       138.98       665.43       0.9980          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            526.45       202.21       728.66       0.9998          1.0000            1.0000         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            526.45       150.67       677.12       0.9979          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            526.45       215.94       742.39       0.9998          1.0000            1.0000         3.77
IVF-Binary-512-nl223-random (self)                       526.45       362.24       888.68       0.9978          1.0003            1.0000         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             652.99        71.19       724.17       0.6377          1.4609            1.3400         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             652.99        75.43       728.42       0.6368          1.4691            1.3434         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             652.99        85.67       738.66       0.6346          1.4956            1.3511         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            652.99       131.96       784.95       0.9981          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            652.99       192.50       845.48       0.9996          1.0001            1.0000         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            652.99       146.04       799.03       0.9982          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            652.99       195.96       848.95       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            652.99       144.28       797.27       0.9980          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            652.99       207.15       860.14       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-random (self)                       652.99       349.70     1_002.69       0.9980          1.0003            1.0000         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)                 468.83        68.64       537.47       0.6577          1.3876            1.2898         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)                468.83        84.11       552.94       0.6524          1.4211            1.3034         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)                468.83        98.27       567.10       0.6509          1.4401            1.3080         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)                468.83       134.69       603.51       0.9969          1.0006            1.0000         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)                468.83       196.09       664.91       0.9978          1.0005            1.0000         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)               468.83       149.36       618.19       0.9987          1.0001            1.0000         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)               468.83       212.70       681.52       0.9999          1.0000            1.0000         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)               468.83       163.62       632.45       0.9985          1.0002            1.0000         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)               468.83       235.51       704.33       0.9999          1.0000            1.0000         3.71
IVF-Binary-512-nl158-pca (self)                          468.83       415.04       883.87       0.9986          1.0002            1.0000         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                543.51        69.47       612.98       0.6552          1.3961            1.2931         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                543.51        75.50       619.01       0.6533          1.4086            1.2988         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                543.51        90.31       633.81       0.6513          1.4315            1.3057         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               543.51       132.25       675.76       0.9983          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               543.51       190.92       734.43       0.9993          1.0001            1.0000         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               543.51       139.44       682.95       0.9986          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               543.51       198.74       742.24       0.9998          1.0000            1.0000         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               543.51       149.14       692.64       0.9985          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               543.51       217.10       760.61       0.9999          1.0000            1.0000         3.77
IVF-Binary-512-nl223-pca (self)                          543.51       365.51       909.01       0.9985          1.0002            1.0000         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                677.50        73.81       751.31       0.6544          1.4017            1.2942         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                677.50        78.64       756.14       0.6536          1.4083            1.2967         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                677.50        83.10       760.60       0.6516          1.4283            1.3033         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               677.50       133.58       811.08       0.9986          1.0002            1.0000         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               677.50       195.07       872.57       0.9996          1.0001            1.0000         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               677.50       135.48       812.98       0.9987          1.0001            1.0000         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               677.50       195.23       872.73       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               677.50       145.39       822.89       0.9986          1.0002            1.0000         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               677.50       208.77       886.27       0.9999          1.0000            1.0000         3.86
IVF-Binary-512-nl316-pca (self)                          677.50       356.58     1_034.08       0.9986          1.0002            1.0000         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)             542.83       112.58       655.41       0.6845          1.3518            1.2576         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)            542.83       128.50       671.34       0.6792          1.3861            1.2707         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)            542.83       155.42       698.25       0.6776          1.4035            1.2751         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)            542.83       183.58       726.41       0.9977          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)            542.83       239.68       782.52       0.9978          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)           542.83       205.82       748.66       0.9997          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)           542.83       278.03       820.86       0.9999          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)           542.83       220.00       762.83       0.9996          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)           542.83       301.73       844.56       1.0000          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-random (self)                      542.83       583.36     1_126.19       0.9995          1.0000            1.0000         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            591.14       109.47       700.61       0.6825          1.3585            1.2601         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            591.14       111.44       702.58       0.6806          1.3719            1.2667         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            591.14       132.65       723.79       0.6784          1.3948            1.2738         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           591.14       176.17       767.32       0.9991          1.0002            1.0000         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           591.14       237.09       828.23       0.9994          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           591.14       183.18       774.32       0.9995          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           591.14       240.28       831.42       0.9998          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           591.14       193.97       785.11       0.9996          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           591.14       266.87       858.01       0.9999          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-random (self)                      591.14       510.13     1_101.27       0.9994          1.0001            1.0000         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            689.52       107.42       796.94       0.6815          1.3670            1.2637         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)            689.52       111.18       800.70       0.6806          1.3736            1.2657         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)            689.52       123.04       812.56       0.6786          1.3933            1.2725         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)           689.52       171.83       861.35       0.9994          1.0001            1.0000         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)           689.52       233.14       922.66       0.9997          1.0000            1.0000         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)           689.52       172.50       862.02       0.9995          1.0001            1.0000         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)           689.52       236.86       926.38       0.9998          1.0000            1.0000         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)           689.52       188.70       878.22       0.9996          1.0001            1.0000         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)           689.52       253.49       943.01       1.0000          1.0000            1.0000         7.41
IVF-Binary-1024-nl316-random (self)                      689.52       499.41     1_188.93       0.9994          1.0001            1.0000         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)                503.90       105.39       609.30       0.6928          1.3299            1.2427         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)               503.90       124.78       628.68       0.6877          1.3593            1.2559         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)               503.90       146.06       649.96       0.6860          1.3758            1.2595         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)               503.90       178.50       682.41       0.9977          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)               503.90       235.36       739.26       0.9978          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)              503.90       205.77       709.67       0.9998          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)              503.90       260.99       764.89       1.0000          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)              503.90       218.87       722.78       0.9997          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)              503.90       283.20       787.10       1.0000          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-pca (self)                         503.90       586.89     1_090.79       0.9997          1.0000            1.0000         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               578.32       106.63       684.94       0.6902          1.3389            1.2452         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               578.32       114.75       693.06       0.6884          1.3495            1.2507         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               578.32       129.62       707.94       0.6863          1.3695            1.2573         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              578.32       174.13       752.44       0.9992          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              578.32       232.50       810.82       0.9994          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              578.32       176.35       754.67       0.9996          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              578.32       244.30       822.62       0.9999          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              578.32       195.59       773.90       0.9996          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              578.32       262.45       840.76       1.0000          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-pca (self)                         578.32       547.75     1_126.07       0.9995          1.0001            1.0000         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               714.90       109.69       824.60       0.6892          1.3446            1.2492         7.41
IVF-Binary-1024-nl316-np17-rf0-pca (query)               714.90       110.47       825.37       0.6884          1.3504            1.2513         7.41
IVF-Binary-1024-nl316-np25-rf0-pca (query)               714.90       124.28       839.18       0.6867          1.3679            1.2566         7.41
IVF-Binary-1024-nl316-np15-rf10-pca (query)              714.90       169.25       884.16       0.9995          1.0001            1.0000         7.41
IVF-Binary-1024-nl316-np15-rf20-pca (query)              714.90       237.64       952.54       0.9997          1.0000            1.0000         7.41
IVF-Binary-1024-nl316-np17-rf10-pca (query)              714.90       172.73       887.64       0.9996          1.0001            1.0000         7.41
IVF-Binary-1024-nl316-np17-rf20-pca (query)              714.90       238.79       953.69       0.9998          1.0000            1.0000         7.41
IVF-Binary-1024-nl316-np25-rf10-pca (query)              714.90       217.81       932.71       0.9997          1.0000            1.0000         7.41
IVF-Binary-1024-nl316-np25-rf20-pca (query)              714.90       253.75       968.65       1.0000          1.0000            1.0000         7.41
IVF-Binary-1024-nl316-pca (self)                         714.90       505.09     1_219.99       0.9995          1.0001            1.0000         7.41
IVF-Binary-256-nl158-np7-rf0-sign (query)                419.31       164.31       583.62       0.0687          6.6445            6.1205         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               419.31       180.61       599.92       0.0554          7.8737            7.1934         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               419.31       196.58       615.89       0.0506          8.7363            7.8879         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               419.31       210.90       630.21       0.3993          1.6143            1.5273         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               419.31       348.64       767.95       0.6370          1.2496            1.1927         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              419.31       218.80       638.11       0.3090          1.8465            1.7466         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              419.31       372.90       792.21       0.4807          1.4429            1.3742         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              419.31       230.56       649.87       0.2738          1.9854            1.8794         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              419.31       391.47       810.78       0.4129          1.5612            1.4833         1.68
IVF-Binary-256-nl158-sign (self)                         419.31       643.61     1_062.92       0.3147          1.8346            1.7365         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               475.32       155.34       630.66       0.0659          6.5535            6.0912         1.74
IVF-Binary-256-nl223-np14-rf0-sign (query)               475.32       163.34       638.66       0.0606          6.9940            6.4957         1.74
IVF-Binary-256-nl223-np21-rf0-sign (query)               475.32       178.27       653.59       0.0540          7.8607            7.2719         1.74
IVF-Binary-256-nl223-np11-rf10-sign (query)              475.32       198.83       674.15       0.3659          1.6641            1.5816         1.74
IVF-Binary-256-nl223-np11-rf20-sign (query)              475.32       335.83       811.15       0.5968          1.2835            1.2268         1.74
IVF-Binary-256-nl223-np14-rf10-sign (query)              475.32       199.91       675.23       0.3334          1.7511            1.6659         1.74
IVF-Binary-256-nl223-np14-rf20-sign (query)              475.32       345.22       820.53       0.5313          1.3595            1.2996         1.74
IVF-Binary-256-nl223-np21-rf10-sign (query)              475.32       211.22       686.54       0.2921          1.9003            1.8127         1.74
IVF-Binary-256-nl223-np21-rf20-sign (query)              475.32       363.70       839.01       0.4434          1.4929            1.4296         1.74
IVF-Binary-256-nl223-sign (self)                         475.32       581.77     1_057.09       0.3386          1.7410            1.6559         1.74
IVF-Binary-256-nl316-np15-rf0-sign (query)               611.41       154.90       766.30       0.0661          6.4956            6.0418         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               611.41       155.38       766.79       0.0633          6.7181            6.2205         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               611.41       168.65       780.06       0.0564          7.4857            6.8850         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              611.41       196.38       807.79       0.3650          1.6701            1.5867         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              611.41       330.08       941.48       0.5869          1.2940            1.2404         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              611.41       202.59       814.00       0.3481          1.7146            1.6271         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              611.41       338.05       949.46       0.5539          1.3321            1.2772         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              611.41       205.29       816.70       0.3059          1.8520            1.7609         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              611.41       352.50       963.91       0.4670          1.4544            1.3942         1.84
IVF-Binary-256-nl316-sign (self)                         611.41       556.67     1_168.08       0.3541          1.7014            1.6169         1.84
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
Exhaustive (query)                                        68.42     1_404.29     1_472.71       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.42     4_627.67     4_696.09       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                132.43       268.85       401.27       0.5547          1.7646            1.5366         2.03
ExhaustiveBinary-256-random-rf10 (query)                 132.43       417.86       550.28       0.9898          1.0017            1.0000         2.03
ExhaustiveBinary-256-random-rf20 (query)                 132.43       571.21       703.64       0.9985          1.0002            1.0000         2.03
ExhaustiveBinary-256-random (self)                       132.43     1_331.67     1_464.10       0.9899          1.0016            1.0000         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   225.89       268.49       494.38       0.5767          1.6243            1.4311         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    225.89       420.37       646.25       0.9904          1.0016            1.0000         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    225.89       562.60       788.49       0.9984          1.0002            1.0000         2.03
ExhaustiveBinary-256-pca (self)                          225.89     1_334.61     1_560.49       0.9905          1.0016            1.0000         2.03
ExhaustiveBinary-512-random_no_rr (query)                205.09       391.45       596.54       0.6013          1.6760            1.4608         4.05
ExhaustiveBinary-512-random-rf10 (query)                 205.09       544.96       750.05       0.9977          1.0003            1.0000         4.05
ExhaustiveBinary-512-random-rf20 (query)                 205.09       688.45       893.54       0.9998          1.0000            1.0000         4.05
ExhaustiveBinary-512-random (self)                       205.09     1_788.31     1_993.40       0.9975          1.0003            1.0000         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   293.67       392.77       686.45       0.6443          1.4426            1.3064         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    293.67       545.54       839.21       0.9985          1.0002            1.0000         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    293.67       690.78       984.45       0.9999          1.0000            1.0000         4.05
ExhaustiveBinary-512-pca (self)                          293.67     1_765.00     2_058.67       0.9984          1.0002            1.0000         4.05
ExhaustiveBinary-1024-random_no_rr (query)               258.08       595.59       853.68       0.6624          1.4553            1.3048         8.11
ExhaustiveBinary-1024-random-rf10 (query)                258.08       762.64     1_020.72       0.9995          1.0001            1.0000         8.11
ExhaustiveBinary-1024-random-rf20 (query)                258.08       916.79     1_174.87       1.0000          1.0000            1.0000         8.11
ExhaustiveBinary-1024-random (self)                      258.08     2_509.13     2_767.22       0.9994          1.0001            1.0000         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  349.30       596.22       945.52       0.6865          1.3603            1.2383         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   349.30       762.71     1_112.01       0.9996          1.0001            1.0000         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   349.30       915.67     1_264.97       1.0000          1.0000            1.0000         8.11
ExhaustiveBinary-1024-pca (self)                         349.30     2_505.00     2_854.29       0.9996          1.0001            1.0000         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   84.16       640.63       724.79       0.0400         18.1511           13.6734         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    84.16       755.31       839.48       0.1821          2.5573            2.4620         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    84.16     1_064.29     1_148.45       0.3140          1.8429            1.7786         3.05
ExhaustiveBinary-512-sign (self)                          84.16     2_268.99     2_353.15       0.1897          2.5286            2.4283         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)              640.28        82.75       723.04       0.5627          1.6434            1.4885         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)             640.28        96.60       736.89       0.5594          1.6750            1.5064         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)             640.28       101.22       741.50       0.5580          1.7034            1.5141         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)             640.28       174.32       814.60       0.9915          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)             640.28       270.55       910.84       0.9978          1.0004            1.0000         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)            640.28       182.57       822.85       0.9913          1.0013            1.0000         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)            640.28       284.26       924.54       0.9988          1.0001            1.0000         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)            640.28       192.07       832.36       0.9907          1.0015            1.0000         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)            640.28       287.90       928.18       0.9987          1.0002            1.0000         2.34
IVF-Binary-256-nl158-random (self)                       640.28       409.76     1_050.04       0.9914          1.0014            1.0000         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)             676.42        77.76       754.18       0.5616          1.6456            1.4978         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)             676.42        82.11       758.53       0.5604          1.6579            1.5060         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)             676.42        89.92       766.34       0.5590          1.6819            1.5132         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)            676.42       172.41       848.84       0.9924          1.0011            1.0000         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)            676.42       268.34       944.77       0.9989          1.0001            1.0000         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)            676.42       173.96       850.38       0.9919          1.0012            1.0000         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)            676.42       274.29       950.71       0.9989          1.0001            1.0000         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)            676.42       181.24       857.67       0.9911          1.0014            1.0000         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)            676.42       281.59       958.02       0.9988          1.0001            1.0000         2.46
IVF-Binary-256-nl223-random (self)                       676.42       382.28     1_058.70       0.9920          1.0012            1.0000         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)             774.02        86.08       860.10       0.5614          1.6442            1.4960         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)             774.02        85.33       859.35       0.5608          1.6514            1.4997         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)             774.02        92.86       866.88       0.5594          1.6719            1.5089         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)            774.02       178.59       952.60       0.9924          1.0011            1.0000         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)            774.02       274.31     1_048.33       0.9990          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)            774.02       175.36       949.38       0.9921          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)            774.02       279.62     1_053.64       0.9990          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)            774.02       185.86       959.88       0.9913          1.0013            1.0000         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)            774.02       283.64     1_057.65       0.9988          1.0001            1.0000         2.65
IVF-Binary-256-nl316-random (self)                       774.02       384.97     1_158.99       0.9922          1.0012            1.0000         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)                 725.90        74.22       800.12       0.5835          1.5315            1.4056         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)                725.90        82.62       808.52       0.5809          1.5555            1.4149         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)                725.90        91.10       816.99       0.5799          1.5748            1.4194         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)                725.90       169.29       895.19       0.9916          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)                725.90       266.86       992.75       0.9978          1.0004            1.0000         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)               725.90       177.01       902.90       0.9916          1.0013            1.0000         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)               725.90       274.78     1_000.67       0.9988          1.0002            1.0000         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)               725.90       186.75       912.65       0.9910          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)               725.90       284.44     1_010.34       0.9987          1.0002            1.0000         2.34
IVF-Binary-256-nl158-pca (self)                          725.90       405.64     1_131.53       0.9916          1.0014            1.0000         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)                762.03        81.03       843.06       0.5830          1.5327            1.4106         2.46
IVF-Binary-256-nl223-np14-rf0-pca (query)                762.03        82.40       844.43       0.5818          1.5424            1.4155         2.46
IVF-Binary-256-nl223-np21-rf0-pca (query)                762.03        92.85       854.89       0.5807          1.5585            1.4200         2.46
IVF-Binary-256-nl223-np11-rf10-pca (query)               762.03       171.95       933.99       0.9924          1.0012            1.0000         2.46
IVF-Binary-256-nl223-np11-rf20-pca (query)               762.03       271.08     1_033.11       0.9988          1.0001            1.0000         2.46
IVF-Binary-256-nl223-np14-rf10-pca (query)               762.03       172.33       934.36       0.9921          1.0013            1.0000         2.46
IVF-Binary-256-nl223-np14-rf20-pca (query)               762.03       271.35     1_033.38       0.9989          1.0001            1.0000         2.46
IVF-Binary-256-nl223-np21-rf10-pca (query)               762.03       179.55       941.58       0.9914          1.0014            1.0000         2.46
IVF-Binary-256-nl223-np21-rf20-pca (query)               762.03       281.03     1_043.06       0.9987          1.0002            1.0000         2.46
IVF-Binary-256-nl223-pca (self)                          762.03       385.35     1_147.38       0.9921          1.0013            1.0000         2.46
IVF-Binary-256-nl316-np15-rf0-pca (query)                988.16        85.51     1_073.67       0.5827          1.5322            1.4087         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)                988.16       104.04     1_092.20       0.5823          1.5369            1.4103         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)                988.16        95.65     1_083.81       0.5810          1.5507            1.4178         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)               988.16       182.91     1_171.07       0.9923          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)               988.16       295.52     1_283.68       0.9990          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)               988.16       182.10     1_170.26       0.9920          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)               988.16       286.46     1_274.63       0.9989          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)               988.16       200.16     1_188.32       0.9914          1.0014            1.0000         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)               988.16       296.62     1_284.78       0.9988          1.0002            1.0000         2.65
IVF-Binary-256-nl316-pca (self)                          988.16       381.38     1_369.55       0.9922          1.0012            1.0000         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)              714.12       101.95       816.07       0.6084          1.5746            1.4241         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)             714.12       115.25       829.37       0.6049          1.6062            1.4408         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)             714.12       128.83       842.96       0.6033          1.6340            1.4480         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)             714.12       199.18       913.30       0.9972          1.0004            1.0000         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)             714.12       293.96     1_008.08       0.9985          1.0003            1.0000         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)            714.12       209.06       923.18       0.9981          1.0002            1.0000         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)            714.12       312.83     1_026.96       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)            714.12       224.43       938.55       0.9979          1.0003            1.0000         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)            714.12       324.74     1_038.86       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-random (self)                       714.12       530.20     1_244.32       0.9979          1.0003            1.0000         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)             764.01       105.92       869.93       0.6072          1.5782            1.4305         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)             764.01       111.37       875.38       0.6057          1.5924            1.4386         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)             764.01       123.58       887.59       0.6043          1.6132            1.4454         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)            764.01       200.87       964.88       0.9982          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)            764.01       302.32     1_066.32       0.9996          1.0000            1.0000         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)            764.01       207.53       971.54       0.9983          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)            764.01       306.00     1_070.01       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)            764.01       215.40       979.41       0.9981          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)            764.01       319.58     1_083.59       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-random (self)                       764.01       500.00     1_264.01       0.9982          1.0002            1.0000         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)             853.04       111.85       964.89       0.6067          1.5793            1.4339         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)             853.04       113.36       966.40       0.6061          1.5861            1.4377         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)             853.04       123.52       976.56       0.6044          1.6057            1.4454         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)            853.04       205.37     1_058.41       0.9985          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)            853.04       306.06     1_159.10       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)            853.04       204.32     1_057.36       0.9984          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)            853.04       305.75     1_158.79       0.9999          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)            853.04       214.68     1_067.71       0.9982          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)            853.04       317.47     1_170.51       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-random (self)                       853.04       492.22     1_345.25       0.9983          1.0002            1.0000         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)                 810.03       101.63       911.66       0.6496          1.3874            1.2888         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)                810.03       120.97       931.00       0.6470          1.4043            1.2978         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)                810.03       128.02       938.05       0.6459          1.4173            1.3008         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)                810.03       200.51     1_010.54       0.9975          1.0004            1.0000         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)                810.03       297.40     1_107.43       0.9985          1.0003            1.0000         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)               810.03       211.64     1_021.67       0.9986          1.0001            1.0000         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)               810.03       317.82     1_127.84       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)               810.03       233.93     1_043.96       0.9986          1.0002            1.0000         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)               810.03       327.17     1_137.20       0.9999          1.0000            1.0000         4.36
IVF-Binary-512-nl158-pca (self)                          810.03       532.17     1_342.19       0.9986          1.0002            1.0000         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)                827.53       106.62       934.15       0.6483          1.3915            1.2914         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)                827.53       110.65       938.18       0.6473          1.3993            1.2956         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)                827.53       133.67       961.19       0.6464          1.4090            1.2987         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)               827.53       200.15     1_027.68       0.9986          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)               827.53       300.66     1_128.18       0.9996          1.0001            1.0000         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)               827.53       203.67     1_031.19       0.9987          1.0001            1.0000         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)               827.53       304.03     1_131.55       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)               827.53       214.15     1_041.68       0.9986          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)               827.53       317.83     1_145.35       0.9999          1.0000            1.0000         4.49
IVF-Binary-512-nl223-pca (self)                          827.53       492.25     1_319.77       0.9987          1.0002            1.0000         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)                950.77       112.29     1_063.06       0.6480          1.3927            1.2940         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)                950.77       114.87     1_065.64       0.6475          1.3963            1.2954         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)                950.77       123.46     1_074.24       0.6465          1.4067            1.2990         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)               950.77       209.93     1_160.71       0.9988          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)               950.77       304.35     1_255.13       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)               950.77       212.15     1_162.93       0.9987          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)               950.77       313.96     1_264.74       0.9999          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)               950.77       213.47     1_164.25       0.9986          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)               950.77       317.39     1_268.16       0.9999          1.0000            1.0000         4.67
IVF-Binary-512-nl316-pca (self)                          950.77       500.77     1_451.54       0.9987          1.0001            1.0000         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)             768.73       158.57       927.30       0.6681          1.3933            1.2856         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)            768.73       175.34       944.07       0.6650          1.4133            1.2952         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)            768.73       193.66       962.40       0.6636          1.4310            1.3003         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)            768.73       255.72     1_024.46       0.9983          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)            768.73       356.52     1_125.25       0.9986          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)           768.73       273.45     1_042.18       0.9995          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)           768.73       384.50     1_153.24       0.9999          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)           768.73       291.81     1_060.54       0.9995          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)           768.73       402.58     1_171.31       1.0000          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-random (self)                      768.73       777.46     1_546.19       0.9995          1.0001            1.0000         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)            797.29       164.57       961.86       0.6666          1.3986            1.2895         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)            797.29       167.95       965.24       0.6654          1.4076            1.2936         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)            797.29       184.54       981.83       0.6643          1.4203            1.2986         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)           797.29       255.74     1_053.03       0.9994          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)           797.29       362.83     1_160.12       0.9997          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)           797.29       263.60     1_060.89       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)           797.29       374.80     1_172.09       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)           797.29       282.85     1_080.14       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)           797.29       391.56     1_188.85       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-random (self)                      797.29       723.89     1_521.18       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)            920.01       170.05     1_090.06       0.6664          1.3999            1.2907         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)            920.01       171.68     1_091.69       0.6659          1.4040            1.2926         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)            920.01       183.97     1_103.99       0.6646          1.4168            1.2972         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)           920.01       263.70     1_183.72       0.9996          1.0001            1.0000         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)           920.01       369.24     1_289.25       0.9999          1.0000            1.0000         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)           920.01       267.85     1_187.86       0.9996          1.0001            1.0000         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)           920.01       376.52     1_296.53       0.9999          1.0000            1.0000         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)           920.01       281.78     1_201.79       0.9996          1.0001            1.0000         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)           920.01       399.75     1_319.77       1.0000          1.0000            1.0000         8.72
IVF-Binary-1024-nl316-random (self)                      920.01       717.88     1_637.89       0.9996          1.0001            1.0000         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)                866.85       159.60     1_026.45       0.6909          1.3175            1.2259         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)               866.85       177.25     1_044.10       0.6885          1.3324            1.2316         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)               866.85       195.46     1_062.31       0.6876          1.3436            1.2346         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)               866.85       255.46     1_122.31       0.9984          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)               866.85       354.55     1_221.40       0.9986          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)              866.85       273.35     1_140.20       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)              866.85       378.31     1_245.16       0.9999          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)              866.85       299.43     1_166.28       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)              866.85       409.17     1_276.03       1.0000          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-pca (self)                         866.85       784.54     1_651.39       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)               886.45       165.51     1_051.96       0.6897          1.3214            1.2293         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)               886.45       168.81     1_055.26       0.6887          1.3273            1.2309         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)               886.45       188.35     1_074.80       0.6878          1.3368            1.2338         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)              886.45       264.30     1_150.75       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)              886.45       367.23     1_253.68       0.9997          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)              886.45       264.81     1_151.26       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)              886.45       373.00     1_259.45       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)              886.45       287.72     1_174.17       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)              886.45       394.51     1_280.96       1.0000          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-pca (self)                         886.45       728.47     1_614.92       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_010.44       170.54     1_180.98       0.6897          1.3225            1.2300         8.72
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_010.44       171.76     1_182.19       0.6892          1.3257            1.2309         8.72
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_010.44       183.69     1_194.12       0.6881          1.3346            1.2341         8.72
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_010.44       264.91     1_275.34       0.9997          1.0000            1.0000         8.72
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_010.44       366.60     1_377.03       0.9999          1.0000            1.0000         8.72
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_010.44       265.72     1_276.16       0.9997          1.0000            1.0000         8.72
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_010.44       375.00     1_385.44       0.9999          1.0000            1.0000         8.72
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_010.44       282.92     1_293.36       0.9997          1.0001            1.0000         8.72
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_010.44       394.98     1_405.42       1.0000          1.0000            1.0000         8.72
IVF-Binary-1024-nl316-pca (self)                       1_010.44       718.77     1_729.21       0.9996          1.0000            1.0000         8.72
IVF-Binary-512-nl158-np7-rf0-sign (query)                595.93       262.62       858.55       0.0587          7.9788            7.1475         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)               595.93       279.64       875.57       0.0517          9.1464            7.9827         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)               595.93       296.52       892.45       0.0486         10.0153            8.6481         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)               595.93       332.64       928.56       0.3222          1.8338            1.7364         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)               595.93       590.90     1_186.83       0.5149          1.4031            1.3349         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)              595.93       350.14       946.06       0.2801          1.9786            1.8635         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)              595.93       616.97     1_212.90       0.4400          1.5255            1.4474         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)              595.93       368.12       964.04       0.2590          2.0739            1.9494         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)              595.93       636.05     1_231.97       0.3996          1.6072            1.5243         3.36
IVF-Binary-512-nl158-sign (self)                         595.93       979.36     1_575.29       0.2870          1.9578            1.8406         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               622.05       262.06       884.12       0.0570          7.8806            7.1787         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               622.05       269.46       891.52       0.0540          8.2916            7.4994         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               622.05       283.38       905.43       0.0499          9.1818            8.1707         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              622.05       354.49       976.54       0.3139          1.8522            1.7521         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              622.05       592.48     1_214.53       0.5026          1.4175            1.3460         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              622.05       359.99       982.05       0.2945          1.9145            1.8086         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              622.05       603.50     1_225.55       0.4691          1.4689            1.3964         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              622.05       354.91       976.96       0.2686          2.0220            1.8987         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              622.05       621.17     1_243.22       0.4170          1.5688            1.4883         3.49
IVF-Binary-512-nl223-sign (self)                         622.05       935.18     1_557.23       0.3016          1.8962            1.7845         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)               739.78       265.31     1_005.10       0.0576          7.7316            7.1191         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)               739.78       271.24     1_011.02       0.0558          7.9302            7.2945         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)               739.78       279.85     1_019.63       0.0519          8.6360            7.8466         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)              739.78       337.03     1_076.81       0.3181          1.8372            1.7369         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)              739.78       595.23     1_335.02       0.5020          1.4131            1.3434         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)              739.78       340.31     1_080.09       0.3081          1.8711            1.7676         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)              739.78       601.91     1_341.70       0.4843          1.4392            1.3711         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)              739.78       360.54     1_100.32       0.2819          1.9709            1.8521         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)              739.78       616.52     1_356.30       0.4341          1.5294            1.4542         3.67
IVF-Binary-512-nl316-sign (self)                         739.78       943.58     1_683.36       0.3138          1.8534            1.7425         3.67
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
Exhaustive (query)                                       100.26     1_974.33     2_074.60       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.26     6_586.21     6_686.47       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                192.57       282.54       475.11       0.5361          1.8068            1.5908         2.28
ExhaustiveBinary-256-random-rf10 (query)                 192.57       453.83       646.40       0.9868          1.0022            1.0000         2.28
ExhaustiveBinary-256-random-rf20 (query)                 192.57       615.94       808.50       0.9980          1.0003            1.0000         2.28
ExhaustiveBinary-256-random (self)                       192.57     1_428.30     1_620.87       0.9876          1.0021            1.0000         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   389.73       288.51       678.24       0.5754          1.5495            1.4128         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    389.73       454.58       844.31       0.9895          1.0018            1.0000         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    389.73       630.47     1_020.21       0.9983          1.0002            1.0000         2.28
ExhaustiveBinary-256-pca (self)                          389.73     1_433.45     1_823.18       0.9897          1.0017            1.0000         2.28
ExhaustiveBinary-512-random_no_rr (query)                299.07       412.64       711.71       0.5866          1.6778            1.4946         4.55
ExhaustiveBinary-512-random-rf10 (query)                 299.07       588.20       887.27       0.9966          1.0005            1.0000         4.55
ExhaustiveBinary-512-random-rf20 (query)                 299.07       773.50     1_072.57       0.9997          1.0001            1.0000         4.55
ExhaustiveBinary-512-random (self)                       299.07     1_892.41     2_191.48       0.9969          1.0004            1.0000         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   497.10       413.75       910.86       0.6388          1.4217            1.3032         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    497.10       583.42     1_080.52       0.9979          1.0003            1.0000         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    497.10       751.64     1_248.75       0.9998          1.0000            1.0000         4.55
ExhaustiveBinary-512-pca (self)                          497.10     1_901.62     2_398.72       0.9981          1.0002            1.0000         4.55
ExhaustiveBinary-1024-random_no_rr (query)               499.46       640.54     1_139.99       0.6446          1.4909            1.3512         9.11
ExhaustiveBinary-1024-random-rf10 (query)                499.46       827.25     1_326.71       0.9993          1.0001            1.0000         9.11
ExhaustiveBinary-1024-random-rf20 (query)                499.46     1_014.63     1_514.09       0.9999          1.0000            1.0000         9.11
ExhaustiveBinary-1024-random (self)                      499.46     2_709.74     3_209.20       0.9994          1.0001            1.0000         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  703.10       652.37     1_355.47       0.6795          1.3452            1.2483         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   703.10       826.70     1_529.80       0.9996          1.0001            1.0000         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   703.10     1_057.12     1_760.22       1.0000          1.0000            1.0000         9.11
ExhaustiveBinary-1024-pca (self)                         703.10     2_722.34     3_425.43       0.9997          1.0000            1.0000         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  128.42       816.25       944.68       0.0420         17.7082           13.0970         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   128.42       897.60     1_026.03       0.1896          2.5240            2.4052         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   128.42     1_397.75     1_526.17       0.3229          1.8300            1.7348         4.58
ExhaustiveBinary-768-sign (self)                         128.42     2_881.81     3_010.23       0.1997          2.4832            2.3546         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)              989.74       103.10     1_092.84       0.5429          1.7099            1.5461         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)             989.74       113.91     1_103.65       0.5408          1.7331            1.5595         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)             989.74       124.40     1_114.14       0.5397          1.7545            1.5688         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)             989.74       218.62     1_208.37       0.9891          1.0018            1.0000         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)             989.74       346.56     1_336.30       0.9984          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)            989.74       225.54     1_215.29       0.9884          1.0019            1.0000         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)            989.74       344.94     1_334.68       0.9986          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)            989.74       254.31     1_244.05       0.9877          1.0021            1.0000         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)            989.74       349.26     1_339.00       0.9983          1.0002            1.0000         2.74
IVF-Binary-256-nl158-random (self)                       989.74       497.82     1_487.56       0.9891          1.0018            1.0000         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)           1_092.74        99.73     1_192.47       0.5419          1.7184            1.5524         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)           1_092.74       110.84     1_203.58       0.5412          1.7280            1.5571         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)           1_092.74       110.81     1_203.55       0.5401          1.7511            1.5659         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)          1_092.74       220.22     1_312.96       0.9888          1.0018            1.0000         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)          1_092.74       369.01     1_461.75       0.9985          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)          1_092.74       227.34     1_320.09       0.9885          1.0019            1.0000         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)          1_092.74       357.97     1_450.71       0.9985          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)          1_092.74       227.15     1_319.89       0.9877          1.0020            1.0000         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)          1_092.74       347.69     1_440.43       0.9983          1.0002            1.0000         2.93
IVF-Binary-256-nl223-random (self)                     1_092.74       486.69     1_579.43       0.9890          1.0018            1.0000         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)           1_381.20       111.44     1_492.64       0.5422          1.7108            1.5502         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)           1_381.20       110.97     1_492.17       0.5418          1.7157            1.5534         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)           1_381.20       117.82     1_499.02       0.5409          1.7314            1.5597         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)          1_381.20       227.51     1_608.71       0.9891          1.0018            1.0000         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)          1_381.20       349.51     1_730.71       0.9987          1.0002            1.0000         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)          1_381.20       222.07     1_603.27       0.9888          1.0018            1.0000         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)          1_381.20       346.00     1_727.20       0.9986          1.0002            1.0000         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)          1_381.20       228.55     1_609.75       0.9882          1.0020            1.0000         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)          1_381.20       355.11     1_736.31       0.9984          1.0002            1.0000         3.21
IVF-Binary-256-nl316-random (self)                     1_381.20       496.42     1_877.62       0.9894          1.0017            1.0000         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)               1_181.78        93.43     1_275.21       0.5812          1.4959            1.3935         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)              1_181.78       108.81     1_290.59       0.5794          1.5088            1.3993         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)              1_181.78       110.14     1_291.92       0.5787          1.5186            1.4018         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)              1_181.78       212.33     1_394.11       0.9913          1.0014            1.0000         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)              1_181.78       330.36     1_512.14       0.9984          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)             1_181.78       216.61     1_398.38       0.9907          1.0015            1.0000         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)             1_181.78       338.91     1_520.68       0.9987          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)             1_181.78       224.46     1_406.24       0.9902          1.0016            1.0000         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)             1_181.78       352.10     1_533.88       0.9985          1.0002            1.0000         2.74
IVF-Binary-256-nl158-pca (self)                        1_181.78       490.98     1_672.76       0.9910          1.0014            1.0000         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)              1_303.42       100.83     1_404.25       0.5800          1.5020            1.3965         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)              1_303.42       103.23     1_406.65       0.5795          1.5067            1.3989         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)              1_303.42       110.78     1_414.20       0.5787          1.5168            1.4019         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)             1_303.42       215.54     1_518.97       0.9909          1.0015            1.0000         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)             1_303.42       332.57     1_636.00       0.9987          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)             1_303.42       215.30     1_518.72       0.9906          1.0015            1.0000         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)             1_303.42       338.45     1_641.87       0.9987          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)             1_303.42       225.72     1_529.14       0.9901          1.0016            1.0000         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)             1_303.42       346.08     1_649.50       0.9985          1.0002            1.0000         2.93
IVF-Binary-256-nl223-pca (self)                        1_303.42       469.99     1_773.41       0.9909          1.0014            1.0000         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)              1_579.80       109.99     1_689.79       0.5807          1.4991            1.3929         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)              1_579.80       109.21     1_689.01       0.5804          1.5019            1.3943         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)              1_579.80       115.87     1_695.68       0.5799          1.5084            1.3972         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)             1_579.80       223.56     1_803.36       0.9912          1.0014            1.0000         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)             1_579.80       345.83     1_925.64       0.9989          1.0001            1.0000         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)             1_579.80       223.08     1_802.89       0.9909          1.0015            1.0000         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)             1_579.80       353.13     1_932.93       0.9988          1.0001            1.0000         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)             1_579.80       229.08     1_808.89       0.9903          1.0016            1.0000         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)             1_579.80       348.10     1_927.91       0.9986          1.0002            1.0000         3.21
IVF-Binary-256-nl316-pca (self)                        1_579.80       494.38     2_074.19       0.9913          1.0014            1.0000         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)            1_107.42       133.48     1_240.91       0.5925          1.6007            1.4618         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)           1_107.42       145.30     1_252.72       0.5899          1.6229            1.4748         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)           1_107.42       159.51     1_266.93       0.5889          1.6417            1.4805         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)           1_107.42       247.69     1_355.11       0.9972          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)           1_107.42       371.53     1_478.96       0.9993          1.0001            1.0000         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)          1_107.42       259.15     1_366.58       0.9972          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)          1_107.42       385.17     1_492.59       0.9998          1.0000            1.0000         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)          1_107.42       271.37     1_378.79       0.9970          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)          1_107.42       399.26     1_506.68       0.9997          1.0000            1.0000         5.02
IVF-Binary-512-nl158-random (self)                     1_107.42       655.68     1_763.10       0.9974          1.0003            1.0000         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)           1_207.50       137.13     1_344.63       0.5912          1.6104            1.4673         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)           1_207.50       141.63     1_349.13       0.5903          1.6195            1.4721         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)           1_207.50       155.67     1_363.17       0.5890          1.6407            1.4796         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)          1_207.50       253.57     1_461.06       0.9972          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)          1_207.50       376.61     1_584.11       0.9996          1.0001            1.0000         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)          1_207.50       256.16     1_463.66       0.9972          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)          1_207.50       387.23     1_594.73       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)          1_207.50       267.89     1_475.39       0.9969          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)          1_207.50       400.65     1_608.15       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-random (self)                     1_207.50       631.31     1_838.81       0.9974          1.0003            1.0000         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)           1_501.40       147.48     1_648.88       0.5911          1.6073            1.4674         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)           1_501.40       147.56     1_648.96       0.5906          1.6115            1.4698         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)           1_501.40       158.29     1_659.70       0.5897          1.6270            1.4771         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)          1_501.40       261.50     1_762.90       0.9974          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)          1_501.40       394.37     1_895.77       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)          1_501.40       260.82     1_762.23       0.9973          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)          1_501.40       393.99     1_895.40       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)          1_501.40       274.91     1_776.31       0.9971          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)          1_501.40       418.04     1_919.44       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-random (self)                     1_501.40       646.65     2_148.06       0.9976          1.0003            1.0000         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)               1_289.31       131.09     1_420.40       0.6432          1.3824            1.2890         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)              1_289.31       144.28     1_433.60       0.6414          1.3937            1.2943         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)              1_289.31       157.06     1_446.38       0.6407          1.4029            1.2975         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)              1_289.31       249.19     1_538.50       0.9980          1.0002            1.0000         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)              1_289.31       380.15     1_669.46       0.9994          1.0001            1.0000         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)             1_289.31       261.25     1_550.56       0.9982          1.0002            1.0000         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)             1_289.31       388.73     1_678.04       0.9999          1.0000            1.0000         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)             1_289.31       275.00     1_564.31       0.9981          1.0003            1.0000         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)             1_289.31       404.00     1_693.31       0.9998          1.0000            1.0000         5.02
IVF-Binary-512-nl158-pca (self)                        1_289.31       657.82     1_947.14       0.9983          1.0002            1.0000         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_409.09       142.07     1_551.16       0.6420          1.3886            1.2933         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_409.09       142.48     1_551.57       0.6414          1.3936            1.2957         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_409.09       153.74     1_562.83       0.6407          1.4030            1.2985         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_409.09       255.48     1_664.57       0.9982          1.0002            1.0000         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_409.09       383.91     1_793.00       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_409.09       258.35     1_667.44       0.9982          1.0002            1.0000         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_409.09       392.92     1_802.02       0.9998          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_409.09       269.60     1_678.70       0.9981          1.0003            1.0000         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_409.09       404.93     1_814.02       0.9998          1.0000            1.0000         5.21
IVF-Binary-512-nl223-pca (self)                        1_409.09       628.13     2_037.22       0.9983          1.0002            1.0000         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)              1_678.51       146.16     1_824.67       0.6422          1.3878            1.2924         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)              1_678.51       147.48     1_825.99       0.6419          1.3897            1.2933         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)              1_678.51       158.01     1_836.53       0.6413          1.3958            1.2957         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)             1_678.51       261.11     1_939.62       0.9984          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)             1_678.51       389.16     2_067.67       0.9999          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)             1_678.51       262.27     1_940.78       0.9983          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)             1_678.51       394.97     2_073.49       0.9999          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)             1_678.51       271.84     1_950.35       0.9981          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)             1_678.51       410.22     2_088.73       0.9999          1.0000            1.0000         5.48
IVF-Binary-512-nl316-pca (self)                        1_678.51       640.27     2_318.79       0.9984          1.0002            1.0000         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)           1_284.48       211.64     1_496.12       0.6492          1.4403            1.3298         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)          1_284.48       226.90     1_511.37       0.6468          1.4562            1.3410         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)          1_284.48       242.41     1_526.88       0.6457          1.4688            1.3455         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)          1_284.48       330.09     1_614.57       0.9990          1.0002            1.0000         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)          1_284.48       467.22     1_751.70       0.9994          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)         1_284.48       354.01     1_638.48       0.9995          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)         1_284.48       497.69     1_782.17       0.9999          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)         1_284.48       370.12     1_654.60       0.9994          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)         1_284.48       514.24     1_798.71       0.9999          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-random (self)                    1_284.48       973.51     2_257.98       0.9995          1.0001            1.0000         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_417.76       222.10     1_639.86       0.6479          1.4467            1.3358         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_417.76       221.30     1_639.05       0.6472          1.4539            1.3395         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_417.76       251.74     1_669.49       0.6461          1.4684            1.3443         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_417.76       344.64     1_762.40       0.9993          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_417.76       487.06     1_904.81       0.9997          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_417.76       355.98     1_773.74       0.9994          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_417.76       497.67     1_915.42       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_417.76       374.85     1_792.60       0.9994          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_417.76       519.48     1_937.23       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-random (self)                    1_417.76       972.44     2_390.19       0.9995          1.0001            1.0000         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)          1_685.63       232.22     1_917.85       0.6478          1.4461            1.3341        10.03
IVF-Binary-1024-nl316-np17-rf0-random (query)          1_685.63       233.14     1_918.77       0.6475          1.4494            1.3360        10.03
IVF-Binary-1024-nl316-np25-rf0-random (query)          1_685.63       240.65     1_926.28       0.6465          1.4599            1.3405        10.03
IVF-Binary-1024-nl316-np15-rf10-random (query)         1_685.63       362.25     2_047.88       0.9995          1.0001            1.0000        10.03
IVF-Binary-1024-nl316-np15-rf20-random (query)         1_685.63       497.33     2_182.96       0.9999          1.0000            1.0000        10.03
IVF-Binary-1024-nl316-np17-rf10-random (query)         1_685.63       366.74     2_052.36       0.9995          1.0001            1.0000        10.03
IVF-Binary-1024-nl316-np17-rf20-random (query)         1_685.63       503.65     2_189.27       0.9999          1.0000            1.0000        10.03
IVF-Binary-1024-nl316-np25-rf10-random (query)         1_685.63       384.52     2_070.15       0.9994          1.0001            1.0000        10.03
IVF-Binary-1024-nl316-np25-rf20-random (query)         1_685.63       526.93     2_212.56       0.9999          1.0000            1.0000        10.03
IVF-Binary-1024-nl316-random (self)                    1_685.63       958.98     2_644.61       0.9995          1.0001            1.0000        10.03
IVF-Binary-1024-nl158-np7-rf0-pca (query)              1_499.54       211.15     1_710.70       0.6828          1.3187            1.2384         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)             1_499.54       225.95     1_725.49       0.6812          1.3271            1.2437         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)             1_499.54       245.69     1_745.23       0.6805          1.3344            1.2453         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)             1_499.54       330.60     1_830.14       0.9992          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)             1_499.54       463.72     1_963.27       0.9994          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)            1_499.54       352.33     1_851.87       0.9997          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)            1_499.54       492.55     1_992.09       1.0000          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)            1_499.54       377.68     1_877.22       0.9996          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)            1_499.54       520.39     2_019.94       1.0000          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-pca (self)                       1_499.54       984.99     2_484.54       0.9997          1.0000            1.0000         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_608.10       220.36     1_828.46       0.6817          1.3239            1.2402         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_608.10       225.76     1_833.86       0.6813          1.3271            1.2427         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_608.10       243.34     1_851.44       0.6805          1.3344            1.2451         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_608.10       343.13     1_951.23       0.9995          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_608.10       480.19     2_088.29       0.9998          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_608.10       358.94     1_967.04       0.9996          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_608.10       510.83     2_118.93       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_608.10       374.28     1_982.38       0.9996          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_608.10       517.24     2_125.34       1.0000          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-pca (self)                       1_608.10       940.02     2_548.12       0.9997          1.0000            1.0000         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_897.48       227.48     2_124.97       0.6817          1.3241            1.2409        10.03
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_897.48       227.34     2_124.83       0.6815          1.3255            1.2417        10.03
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_897.48       247.62     2_145.11       0.6809          1.3307            1.2441        10.03
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_897.48       356.49     2_253.97       0.9996          1.0000            1.0000        10.03
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_897.48       498.24     2_395.73       0.9999          1.0000            1.0000        10.03
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_897.48       358.20     2_255.68       0.9996          1.0001            1.0000        10.03
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_897.48       501.11     2_398.59       0.9999          1.0000            1.0000        10.03
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_897.48       380.25     2_277.73       0.9996          1.0001            1.0000        10.03
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_897.48       520.09     2_417.58       1.0000          1.0000            1.0000        10.03
IVF-Binary-1024-nl316-pca (self)                       1_897.48       956.05     2_853.54       0.9997          1.0000            1.0000        10.03
IVF-Binary-768-nl158-np7-rf0-sign (query)                948.36       390.13     1_338.49       0.0572          8.2471            7.4925         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)               948.36       401.74     1_350.11       0.0520          9.4902            8.2157         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)               948.36       428.24     1_376.60       0.0494         10.3593            8.7490         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)               948.36       481.33     1_429.70       0.3101          1.8950            1.7877         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)               948.36       849.38     1_797.75       0.4773          1.4831            1.3817         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)              948.36       492.08     1_440.45       0.2784          2.0012            1.8790         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)              948.36       874.41     1_822.77       0.4277          1.5656            1.4630         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)              948.36       506.56     1_454.92       0.2619          2.0687            1.9351         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)              948.36       898.17     1_846.53       0.3996          1.6191            1.5099         5.04
IVF-Binary-768-nl158-sign (self)                         948.36     1_392.73     2_341.09       0.2911          1.9618            1.8389         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)             1_052.04       386.94     1_438.98       0.0575          8.1957            7.3503         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)             1_052.04       396.28     1_448.33       0.0550          8.6555            7.6759         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)             1_052.04       411.61     1_463.65       0.0517          9.6218            8.3391         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)            1_052.04       478.41     1_530.45       0.3112          1.8758            1.7623         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)            1_052.04       857.15     1_909.20       0.4738          1.4729            1.3874         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)            1_052.04       489.55     1_541.59       0.2974          1.9245            1.8040         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)            1_052.04       867.76     1_919.80       0.4500          1.5112            1.4210         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)            1_052.04       505.36     1_557.40       0.2763          2.0107            1.8820         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)            1_052.04       895.79     1_947.83       0.4154          1.5791            1.4831         5.23
IVF-Binary-768-nl223-sign (self)                       1_052.04     1_388.64     2_440.68       0.3097          1.8870            1.7670         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)             1_322.64       394.33     1_716.97       0.0579          7.9049            7.2012         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)             1_322.64       398.71     1_721.35       0.0566          8.1385            7.3655         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)             1_322.64       417.52     1_740.16       0.0534          8.9452            7.8698         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)            1_322.64       488.36     1_811.00       0.3161          1.8521            1.7485         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)            1_322.64       908.05     2_230.69       0.4802          1.4604            1.3754         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)            1_322.64       490.10     1_812.74       0.3080          1.8779            1.7716         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)            1_322.64       873.57     2_196.21       0.4678          1.4796            1.3939         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)            1_322.64       506.66     1_829.30       0.2866          1.9580            1.8379         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)            1_322.64       895.36     2_218.00       0.4331          1.5380            1.4459         5.51
IVF-Binary-768-nl316-sign (self)                       1_322.64     1_415.96     2_738.60       0.3207          1.8447            1.7287         5.51
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
Exhaustive (query)                                        33.52       711.34       744.86       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.52     2_325.04     2_358.57       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             275.29        61.81       337.10       0.4587          1.0613            1.0633         2.56
ExhaustiveRaBitQ-rf5 (query)                             275.29       116.73       392.02       0.8093          1.0067            1.0054         2.56
ExhaustiveRaBitQ-rf10 (query)                            275.29       169.33       444.62       0.9208          1.0021            1.0006         2.56
ExhaustiveRaBitQ-rf20 (query)                            275.29       251.95       527.24       0.9802          1.0004            1.0000         2.56
ExhaustiveRaBitQ-rfB (query)                             275.29       161.49       436.78       0.9787          1.0005            1.0000         2.56
ExhaustiveRaBitQ (self)                                  275.29       551.65       826.94       0.9199          1.0022            1.0007         2.56
IVF-RaBitQ-nl158-np7-rf0 (query)                         388.26        50.31       438.58       0.4700          1.0578            1.0606         2.67
IVF-RaBitQ-nl158-np12-rf0 (query)                        388.26        59.38       447.64       0.4700          1.0578            1.0606         2.67
IVF-RaBitQ-nl158-np17-rf0 (query)                        388.26        82.49       470.75       0.4700          1.0578            1.0606         2.67
IVF-RaBitQ-nl158-np7-rf10 (query)                        388.26       131.68       519.94       0.9256          1.0020            1.0005         2.67
IVF-RaBitQ-nl158-np7-rf20 (query)                        388.26       216.92       605.18       0.9819          1.0004            1.0000         2.67
IVF-RaBitQ-nl158-np7-rfB (query)                         388.26       117.82       506.08       0.9780          1.0005            1.0000         2.67
IVF-RaBitQ-nl158-np12-rf10 (query)                       388.26       153.46       541.72       0.9256          1.0020            1.0005         2.67
IVF-RaBitQ-nl158-np12-rf20 (query)                       388.26       220.21       608.48       0.9819          1.0004            1.0000         2.67
IVF-RaBitQ-nl158-np12-rfB (query)                        388.26       118.98       507.24       0.9780          1.0005            1.0000         2.67
IVF-RaBitQ-nl158-np17-rf10 (query)                       388.26       141.56       529.83       0.9256          1.0020            1.0005         2.67
IVF-RaBitQ-nl158-np17-rf20 (query)                       388.26       221.07       609.33       0.9819          1.0004            1.0000         2.67
IVF-RaBitQ-nl158-np17-rfB (query)                        388.26       122.63       510.89       0.9780          1.0005            1.0000         2.67
IVF-RaBitQ-nl158 (self)                                  388.26       367.04       755.30       0.9784          1.0005            1.0000         2.67
IVF-RaBitQ-nl223-np11-rf0 (query)                        319.47        51.94       371.41       0.4777          1.0559            1.0579         2.82
IVF-RaBitQ-nl223-np14-rf0 (query)                        319.47        53.17       372.64       0.4778          1.0559            1.0579         2.82
IVF-RaBitQ-nl223-np21-rf0 (query)                        319.47        58.28       377.75       0.4778          1.0559            1.0579         2.82
IVF-RaBitQ-nl223-np11-rf10 (query)                       319.47       131.83       451.31       0.9328          1.0017            1.0002         2.82
IVF-RaBitQ-nl223-np11-rf20 (query)                       319.47       199.13       518.60       0.9849          1.0003            1.0000         2.82
IVF-RaBitQ-nl223-np11-rfB (query)                        319.47       114.77       434.25       0.9786          1.0005            1.0000         2.82
IVF-RaBitQ-nl223-np14-rf10 (query)                       319.47       134.02       453.49       0.9329          1.0017            1.0002         2.82
IVF-RaBitQ-nl223-np14-rf20 (query)                       319.47       207.17       526.64       0.9850          1.0003            1.0000         2.82
IVF-RaBitQ-nl223-np14-rfB (query)                        319.47       114.11       433.58       0.9787          1.0005            1.0000         2.82
IVF-RaBitQ-nl223-np21-rf10 (query)                       319.47       138.88       458.35       0.9329          1.0017            1.0002         2.82
IVF-RaBitQ-nl223-np21-rf20 (query)                       319.47       210.25       529.73       0.9850          1.0003            1.0000         2.82
IVF-RaBitQ-nl223-np21-rfB (query)                        319.47       115.50       434.97       0.9787          1.0005            1.0000         2.82
IVF-RaBitQ-nl223 (self)                                  319.47       352.42       671.89       0.9782          1.0005            1.0000         2.82
IVF-RaBitQ-nl316-np15-rf0 (query)                        389.50        53.79       443.28       0.4845          1.0538            1.0560         3.05
IVF-RaBitQ-nl316-np17-rf0 (query)                        389.50        55.76       445.26       0.4845          1.0538            1.0560         3.05
IVF-RaBitQ-nl316-np25-rf0 (query)                        389.50        58.28       447.78       0.4845          1.0538            1.0560         3.05
IVF-RaBitQ-nl316-np15-rf10 (query)                       389.50       126.64       516.14       0.9360          1.0016            1.0000         3.05
IVF-RaBitQ-nl316-np15-rf20 (query)                       389.50       199.73       589.22       0.9855          1.0003            1.0000         3.05
IVF-RaBitQ-nl316-np15-rfB (query)                        389.50       110.07       499.56       0.9779          1.0005            1.0000         3.05
IVF-RaBitQ-nl316-np17-rf10 (query)                       389.50       132.00       521.50       0.9360          1.0016            1.0000         3.05
IVF-RaBitQ-nl316-np17-rf20 (query)                       389.50       202.98       592.48       0.9855          1.0003            1.0000         3.05
IVF-RaBitQ-nl316-np17-rfB (query)                        389.50       112.94       502.43       0.9780          1.0005            1.0000         3.05
IVF-RaBitQ-nl316-np25-rf10 (query)                       389.50       130.65       520.15       0.9360          1.0016            1.0000         3.05
IVF-RaBitQ-nl316-np25-rf20 (query)                       389.50       205.18       594.68       0.9855          1.0003            1.0000         3.05
IVF-RaBitQ-nl316-np25-rfB (query)                        389.50       117.75       507.25       0.9780          1.0005            1.0000         3.05
IVF-RaBitQ-nl316 (self)                                  389.50       345.82       735.31       0.9784          1.0005            1.0000         3.05
IVF-RaBitQ-nl223-ex2-np14 (query)                        331.84       120.34       452.18       0.8663          1.0030            1.0026         6.27
IVF-RaBitQ-nl223-ex2-np21 (query)                        331.84       125.93       457.77       0.8663          1.0030            1.0026         6.27
IVF-RaBitQ-nl223-ex4-np14 (query)                        331.24       136.21       467.45       0.9455          1.0007            1.0002         9.32
IVF-RaBitQ-nl223-ex4-np21 (query)                        331.24       138.37       469.61       0.9455          1.0007            1.0002         9.32
IVF-RaBitQ-nl223-ex8-np14 (query)                        366.96       128.09       495.05       0.9711          1.0005            1.0000        15.42
IVF-RaBitQ-nl223-ex8-np21 (query)                        366.96       128.30       495.26       0.9711          1.0005            1.0000        15.42
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
Exhaustive (query)                                        68.07     1_329.60     1_397.67       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.07     4_512.35     4_580.42       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                             389.74       101.01       490.75       0.4490          1.0427            1.0449         4.37
ExhaustiveRaBitQ-rf5 (query)                             389.74       170.95       560.69       0.7775          1.0056            1.0048         4.37
ExhaustiveRaBitQ-rf10 (query)                            389.74       235.32       625.06       0.8975          1.0019            1.0010         4.37
ExhaustiveRaBitQ-rf20 (query)                            389.74       357.39       747.14       0.9713          1.0005            1.0000         4.37
ExhaustiveRaBitQ-rfB (query)                             389.74       322.04       711.79       0.9714          1.0005            1.0000         4.37
ExhaustiveRaBitQ (self)                                  389.74       705.85     1_095.60       0.8982          1.0020            1.0010         4.37
IVF-RaBitQ-nl158-np7-rf0 (query)                         517.69        90.20       607.89       0.4583          1.0407            1.0431         4.59
IVF-RaBitQ-nl158-np12-rf0 (query)                        517.69        87.64       605.33       0.4583          1.0407            1.0431         4.59
IVF-RaBitQ-nl158-np17-rf0 (query)                        517.69        90.54       608.24       0.4583          1.0407            1.0431         4.59
IVF-RaBitQ-nl158-np7-rf10 (query)                        517.69       204.85       722.54       0.9031          1.0018            1.0009         4.59
IVF-RaBitQ-nl158-np7-rf20 (query)                        517.69       314.91       832.60       0.9733          1.0004            1.0000         4.59
IVF-RaBitQ-nl158-np7-rfB (query)                         517.69       236.42       754.12       0.9713          1.0005            1.0000         4.59
IVF-RaBitQ-nl158-np12-rf10 (query)                       517.69       205.29       722.99       0.9031          1.0018            1.0009         4.59
IVF-RaBitQ-nl158-np12-rf20 (query)                       517.69       319.70       837.40       0.9733          1.0004            1.0000         4.59
IVF-RaBitQ-nl158-np12-rfB (query)                        517.69       241.61       759.30       0.9713          1.0005            1.0000         4.59
IVF-RaBitQ-nl158-np17-rf10 (query)                       517.69       209.01       726.71       0.9031          1.0018            1.0009         4.59
IVF-RaBitQ-nl158-np17-rf20 (query)                       517.69       322.51       840.20       0.9733          1.0004            1.0000         4.59
IVF-RaBitQ-nl158-np17-rfB (query)                        517.69       245.75       763.44       0.9713          1.0005            1.0000         4.59
IVF-RaBitQ-nl158 (self)                                  517.69       768.66     1_286.35       0.9710          1.0005            1.0000         4.59
IVF-RaBitQ-nl223-np11-rf0 (query)                        459.88        89.46       549.34       0.4636          1.0397            1.0419         4.88
IVF-RaBitQ-nl223-np14-rf0 (query)                        459.88        90.70       550.58       0.4637          1.0397            1.0419         4.88
IVF-RaBitQ-nl223-np21-rf0 (query)                        459.88        95.09       554.97       0.4637          1.0397            1.0419         4.88
IVF-RaBitQ-nl223-np11-rf10 (query)                       459.88       204.39       664.27       0.9113          1.0016            1.0007         4.88
IVF-RaBitQ-nl223-np11-rf20 (query)                       459.88       310.98       770.86       0.9764          1.0004            1.0000         4.88
IVF-RaBitQ-nl223-np11-rfB (query)                        459.88       239.60       699.48       0.9720          1.0004            1.0000         4.88
IVF-RaBitQ-nl223-np14-rf10 (query)                       459.88       203.87       663.75       0.9114          1.0016            1.0007         4.88
IVF-RaBitQ-nl223-np14-rf20 (query)                       459.88       313.50       773.38       0.9764          1.0004            1.0000         4.88
IVF-RaBitQ-nl223-np14-rfB (query)                        459.88       227.83       687.70       0.9720          1.0004            1.0000         4.88
IVF-RaBitQ-nl223-np21-rf10 (query)                       459.88       208.90       668.77       0.9114          1.0016            1.0007         4.88
IVF-RaBitQ-nl223-np21-rf20 (query)                       459.88       332.18       792.06       0.9764          1.0004            1.0000         4.88
IVF-RaBitQ-nl223-np21-rfB (query)                        459.88       231.69       691.56       0.9720          1.0004            1.0000         4.88
IVF-RaBitQ-nl223 (self)                                  459.88       701.72     1_161.59       0.9716          1.0004            1.0000         4.88
IVF-RaBitQ-nl316-np15-rf0 (query)                        537.28        94.74       632.02       0.4692          1.0386            1.0410         5.35
IVF-RaBitQ-nl316-np17-rf0 (query)                        537.28        98.65       635.93       0.4692          1.0386            1.0410         5.35
IVF-RaBitQ-nl316-np25-rf0 (query)                        537.28       102.44       639.72       0.4692          1.0386            1.0410         5.35
IVF-RaBitQ-nl316-np15-rf10 (query)                       537.28       211.20       748.48       0.9133          1.0016            1.0006         5.35
IVF-RaBitQ-nl316-np15-rf20 (query)                       537.28       318.62       855.90       0.9770          1.0003            1.0000         5.35
IVF-RaBitQ-nl316-np15-rfB (query)                        537.28       225.20       762.48       0.9721          1.0004            1.0000         5.35
IVF-RaBitQ-nl316-np17-rf10 (query)                       537.28       211.39       748.67       0.9133          1.0016            1.0006         5.35
IVF-RaBitQ-nl316-np17-rf20 (query)                       537.28       322.97       860.26       0.9770          1.0003            1.0000         5.35
IVF-RaBitQ-nl316-np17-rfB (query)                        537.28       231.59       768.87       0.9721          1.0004            1.0000         5.35
IVF-RaBitQ-nl316-np25-rf10 (query)                       537.28       215.77       753.06       0.9133          1.0016            1.0006         5.35
IVF-RaBitQ-nl316-np25-rf20 (query)                       537.28       328.02       865.31       0.9770          1.0003            1.0000         5.35
IVF-RaBitQ-nl316-np25-rfB (query)                        537.28       232.39       769.67       0.9721          1.0004            1.0000         5.35
IVF-RaBitQ-nl316 (self)                                  537.28       708.83     1_246.12       0.9714          1.0004            1.0000         5.35
IVF-RaBitQ-nl223-ex2-np14 (query)                        516.33       242.91       759.24       0.8635          1.0021            1.0018        11.37
IVF-RaBitQ-nl223-ex2-np21 (query)                        516.33       243.89       760.23       0.8635          1.0021            1.0018        11.37
IVF-RaBitQ-nl223-ex4-np14 (query)                        528.28       269.45       797.72       0.9407          1.0006            1.0002        17.47
IVF-RaBitQ-nl223-ex4-np21 (query)                        528.28       273.27       801.54       0.9407          1.0006            1.0002        17.47
IVF-RaBitQ-nl223-ex8-np14 (query)                        573.34       243.79       817.13       0.9664          1.0005            1.0000        29.68
IVF-RaBitQ-nl223-ex8-np21 (query)                        573.34       247.65       820.99       0.9664          1.0005            1.0000        29.68
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
Exhaustive (query)                                       100.63     1_979.14     2_079.77       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.63     6_698.68     6_799.31       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                             545.20       125.37       670.57       0.4437          1.0346            1.0365         6.16
ExhaustiveRaBitQ-rf5 (query)                             545.20       205.14       750.34       0.7684          1.0047            1.0042         6.16
ExhaustiveRaBitQ-rf10 (query)                            545.20       288.24       833.44       0.8898          1.0017            1.0010         6.16
ExhaustiveRaBitQ-rf20 (query)                            545.20       427.16       972.36       0.9668          1.0004            1.0000         6.16
ExhaustiveRaBitQ-rfB (query)                             545.20       419.72       964.92       0.9693          1.0004            1.0000         6.16
ExhaustiveRaBitQ (self)                                  545.20       859.04     1_404.24       0.8897          1.0017            1.0010         6.16
IVF-RaBitQ-nl158-np7-rf0 (query)                         679.27       105.06       784.33       0.4536          1.0329            1.0354         6.50
IVF-RaBitQ-nl158-np12-rf0 (query)                        679.27       110.47       789.73       0.4536          1.0329            1.0354         6.50
IVF-RaBitQ-nl158-np17-rf0 (query)                        679.27       114.19       793.45       0.4536          1.0329            1.0354         6.50
IVF-RaBitQ-nl158-np7-rf10 (query)                        679.27       248.91       928.18       0.8961          1.0016            1.0008         6.50
IVF-RaBitQ-nl158-np7-rf20 (query)                        679.27       383.02     1_062.29       0.9693          1.0004            1.0000         6.50
IVF-RaBitQ-nl158-np7-rfB (query)                         679.27       319.27       998.53       0.9693          1.0004            1.0000         6.50
IVF-RaBitQ-nl158-np12-rf10 (query)                       679.27       251.68       930.95       0.8961          1.0016            1.0008         6.50
IVF-RaBitQ-nl158-np12-rf20 (query)                       679.27       391.82     1_071.09       0.9693          1.0004            1.0000         6.50
IVF-RaBitQ-nl158-np12-rfB (query)                        679.27       328.50     1_007.77       0.9693          1.0004            1.0000         6.50
IVF-RaBitQ-nl158-np17-rf10 (query)                       679.27       257.45       936.72       0.8961          1.0016            1.0008         6.50
IVF-RaBitQ-nl158-np17-rf20 (query)                       679.27       394.95     1_074.22       0.9693          1.0004            1.0000         6.50
IVF-RaBitQ-nl158-np17-rfB (query)                        679.27       333.28     1_012.54       0.9693          1.0004            1.0000         6.50
IVF-RaBitQ-nl158 (self)                                  679.27     1_019.30     1_698.56       0.9690          1.0004            1.0000         6.50
IVF-RaBitQ-nl223-np11-rf0 (query)                        620.96       116.98       737.94       0.4528          1.0331            1.0349         6.96
IVF-RaBitQ-nl223-np14-rf0 (query)                        620.96       116.91       737.87       0.4528          1.0331            1.0349         6.96
IVF-RaBitQ-nl223-np21-rf0 (query)                        620.96       124.05       745.01       0.4528          1.0331            1.0349         6.96
IVF-RaBitQ-nl223-np11-rf10 (query)                       620.96       261.84       882.80       0.8970          1.0016            1.0008         6.96
IVF-RaBitQ-nl223-np11-rf20 (query)                       620.96       387.09     1_008.06       0.9705          1.0004            1.0000         6.96
IVF-RaBitQ-nl223-np11-rfB (query)                        620.96       310.85       931.81       0.9698          1.0004            1.0000         6.96
IVF-RaBitQ-nl223-np14-rf10 (query)                       620.96       260.43       881.39       0.8970          1.0016            1.0008         6.96
IVF-RaBitQ-nl223-np14-rf20 (query)                       620.96       392.86     1_013.82       0.9705          1.0004            1.0000         6.96
IVF-RaBitQ-nl223-np14-rfB (query)                        620.96       312.77       933.73       0.9699          1.0004            1.0000         6.96
IVF-RaBitQ-nl223-np21-rf10 (query)                       620.96       266.65       887.62       0.8970          1.0016            1.0008         6.96
IVF-RaBitQ-nl223-np21-rf20 (query)                       620.96       401.59     1_022.55       0.9705          1.0004            1.0000         6.96
IVF-RaBitQ-nl223-np21-rfB (query)                        620.96       327.66       948.63       0.9699          1.0004            1.0000         6.96
IVF-RaBitQ-nl223 (self)                                  620.96       982.05     1_603.01       0.9697          1.0004            1.0000         6.96
IVF-RaBitQ-nl316-np15-rf0 (query)                        742.07       123.68       865.76       0.4633          1.0315            1.0338         7.66
IVF-RaBitQ-nl316-np17-rf0 (query)                        742.07       126.95       869.03       0.4633          1.0315            1.0338         7.66
IVF-RaBitQ-nl316-np25-rf0 (query)                        742.07       132.60       874.68       0.4633          1.0315            1.0338         7.66
IVF-RaBitQ-nl316-np15-rf10 (query)                       742.07       268.17     1_010.25       0.9041          1.0014            1.0006         7.66
IVF-RaBitQ-nl316-np15-rf20 (query)                       742.07       409.33     1_151.41       0.9731          1.0003            1.0000         7.66
IVF-RaBitQ-nl316-np15-rfB (query)                        742.07       304.07     1_046.15       0.9696          1.0004            1.0000         7.66
IVF-RaBitQ-nl316-np17-rf10 (query)                       742.07       268.95     1_011.02       0.9041          1.0014            1.0006         7.66
IVF-RaBitQ-nl316-np17-rf20 (query)                       742.07       401.78     1_143.85       0.9732          1.0003            1.0000         7.66
IVF-RaBitQ-nl316-np17-rfB (query)                        742.07       314.72     1_056.79       0.9696          1.0004            1.0000         7.66
IVF-RaBitQ-nl316-np25-rf10 (query)                       742.07       280.03     1_022.10       0.9041          1.0014            1.0006         7.66
IVF-RaBitQ-nl316-np25-rf20 (query)                       742.07       416.59     1_158.66       0.9732          1.0003            1.0000         7.66
IVF-RaBitQ-nl316-np25-rfB (query)                        742.07       317.98     1_060.05       0.9697          1.0004            1.0000         7.66
IVF-RaBitQ-nl316 (self)                                  742.07       963.79     1_705.86       0.9694          1.0004            1.0000         7.66
IVF-RaBitQ-nl223-ex2-np14 (query)                        660.30       356.18     1_016.49       0.8632          1.0017            1.0014        16.53
IVF-RaBitQ-nl223-ex2-np21 (query)                        660.30       347.60     1_007.91       0.8632          1.0017            1.0014        16.53
IVF-RaBitQ-nl223-ex4-np14 (query)                        686.37       391.54     1_077.92       0.9404          1.0005            1.0001        25.68
IVF-RaBitQ-nl223-ex4-np21 (query)                        686.37       395.05     1_081.42       0.9404          1.0005            1.0001        25.68
IVF-RaBitQ-nl223-ex8-np14 (query)                        783.17       350.26     1_133.43       0.9659          1.0004            1.0000        43.99
IVF-RaBitQ-nl223-ex8-np21 (query)                        783.17       354.03     1_137.20       0.9659          1.0004            1.0000        43.99
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
Exhaustive (query)                                        32.98       711.99       744.97       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.98     2_383.60     2_416.57       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             254.90        80.67       335.56       0.6872          1.0347            1.0333         2.57
ExhaustiveRaBitQ-rf5 (query)                             254.90       138.05       392.95       0.9895          1.0003            1.0000         2.57
ExhaustiveRaBitQ-rf10 (query)                            254.90       193.38       448.28       0.9995          1.0000            1.0000         2.57
ExhaustiveRaBitQ-rf20 (query)                            254.90       292.04       546.93       1.0000          1.0000            1.0000         2.57
ExhaustiveRaBitQ-rfB (query)                             254.90       133.04       387.94       0.9959          1.0001            1.0000         2.57
ExhaustiveRaBitQ (self)                                  254.90       610.04       864.93       0.9997          1.0000            1.0000         2.57
IVF-RaBitQ-nl158-np7-rf0 (query)                         300.52        47.97       348.49       0.6905          1.0340            1.0327         2.68
IVF-RaBitQ-nl158-np12-rf0 (query)                        300.52        49.57       350.09       0.6905          1.0340            1.0327         2.68
IVF-RaBitQ-nl158-np17-rf0 (query)                        300.52        53.55       354.07       0.6905          1.0340            1.0327         2.68
IVF-RaBitQ-nl158-np7-rf10 (query)                        300.52       133.03       433.55       0.9995          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np7-rf20 (query)                        300.52       211.12       511.64       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np7-rfB (query)                         300.52        83.75       384.27       0.9959          1.0001            1.0000         2.68
IVF-RaBitQ-nl158-np12-rf10 (query)                       300.52       137.63       438.15       0.9995          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np12-rf20 (query)                       300.52       212.17       512.69       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np12-rfB (query)                        300.52        88.33       388.85       0.9959          1.0001            1.0000         2.68
IVF-RaBitQ-nl158-np17-rf10 (query)                       300.52       141.05       441.58       0.9995          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rf20 (query)                       300.52       221.91       522.43       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rfB (query)                        300.52        96.26       396.78       0.9959          1.0001            1.0000         2.68
IVF-RaBitQ-nl158 (self)                                  300.52       256.17       556.69       0.9956          1.0001            1.0000         2.68
IVF-RaBitQ-nl223-np11-rf0 (query)                        332.42        50.82       383.23       0.6956          1.0328            1.0313         2.84
IVF-RaBitQ-nl223-np14-rf0 (query)                        332.42        52.59       385.01       0.6956          1.0328            1.0313         2.84
IVF-RaBitQ-nl223-np21-rf0 (query)                        332.42        63.39       395.80       0.6956          1.0328            1.0313         2.84
IVF-RaBitQ-nl223-np11-rf10 (query)                       332.42       131.98       464.40       0.9996          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np11-rf20 (query)                       332.42       206.11       538.53       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np11-rfB (query)                        332.42        85.59       418.00       0.9963          1.0001            1.0000         2.84
IVF-RaBitQ-nl223-np14-rf10 (query)                       332.42       133.22       465.63       0.9996          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np14-rf20 (query)                       332.42       209.21       541.63       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np14-rfB (query)                        332.42        87.81       420.23       0.9963          1.0001            1.0000         2.84
IVF-RaBitQ-nl223-np21-rf10 (query)                       332.42       138.81       471.23       0.9996          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np21-rf20 (query)                       332.42       215.66       548.08       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np21-rfB (query)                        332.42        92.32       424.73       0.9963          1.0001            1.0000         2.84
IVF-RaBitQ-nl223 (self)                                  332.42       271.81       604.22       0.9959          1.0001            1.0000         2.84
IVF-RaBitQ-nl316-np15-rf0 (query)                        366.29        53.96       420.24       0.6991          1.0318            1.0307         3.06
IVF-RaBitQ-nl316-np17-rf0 (query)                        366.29        54.18       420.46       0.6991          1.0318            1.0307         3.06
IVF-RaBitQ-nl316-np25-rf0 (query)                        366.29        59.40       425.69       0.6991          1.0318            1.0307         3.06
IVF-RaBitQ-nl316-np15-rf10 (query)                       366.29       135.06       501.35       0.9996          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np15-rf20 (query)                       366.29       209.44       575.72       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np15-rfB (query)                        366.29        87.14       453.42       0.9961          1.0001            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf10 (query)                       366.29       135.03       501.32       0.9996          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf20 (query)                       366.29       209.74       576.03       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rfB (query)                        366.29        89.15       455.44       0.9961          1.0001            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf10 (query)                       366.29       139.62       505.90       0.9996          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf20 (query)                       366.29       216.20       582.49       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rfB (query)                        366.29        95.09       461.37       0.9961          1.0001            1.0000         3.06
IVF-RaBitQ-nl316 (self)                                  366.29       266.20       632.48       0.9961          1.0001            1.0000         3.06
IVF-RaBitQ-nl223-ex2-np14 (query)                        323.01        95.59       418.59       0.9245          1.0016            1.0012         6.26
IVF-RaBitQ-nl223-ex2-np21 (query)                        323.01       101.20       424.21       0.9245          1.0016            1.0012         6.26
IVF-RaBitQ-nl223-ex4-np14 (query)                        330.96       103.43       434.39       0.9730          1.0002            1.0000         9.32
IVF-RaBitQ-nl223-ex4-np21 (query)                        330.96       109.08       440.04       0.9730          1.0002            1.0000         9.32
IVF-RaBitQ-nl223-ex8-np14 (query)                        352.99        97.72       450.70       0.9914          1.0001            1.0000        15.42
IVF-RaBitQ-nl223-ex8-np21 (query)                        352.99       102.34       455.33       0.9914          1.0001            1.0000        15.42
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
Exhaustive (query)                                        68.53     1_358.91     1_427.44       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.53     4_593.01     4_661.55       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                             415.71       127.97       543.67       0.6581          1.0282            1.0278         4.35
ExhaustiveRaBitQ-rf5 (query)                             415.71       195.19       610.90       0.9782          1.0005            1.0000         4.35
ExhaustiveRaBitQ-rf10 (query)                            415.71       262.25       677.95       0.9981          1.0000            1.0000         4.35
ExhaustiveRaBitQ-rf20 (query)                            415.71       392.27       807.98       0.9999          1.0000            1.0000         4.35
ExhaustiveRaBitQ-rfB (query)                             415.71       235.07       650.78       0.9902          1.0002            1.0000         4.35
ExhaustiveRaBitQ (self)                                  415.71       813.91     1_229.62       0.9985          1.0000            1.0000         4.35
IVF-RaBitQ-nl158-np7-rf0 (query)                         509.32        83.20       592.52       0.6593          1.0281            1.0279         4.59
IVF-RaBitQ-nl158-np12-rf0 (query)                        509.32        86.13       595.44       0.6593          1.0281            1.0279         4.59
IVF-RaBitQ-nl158-np17-rf0 (query)                        509.32        88.57       597.88       0.6593          1.0281            1.0279         4.59
IVF-RaBitQ-nl158-np7-rf10 (query)                        509.32       202.20       711.51       0.9981          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np7-rf20 (query)                        509.32       313.49       822.80       0.9999          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np7-rfB (query)                         509.32       156.50       665.82       0.9904          1.0002            1.0000         4.59
IVF-RaBitQ-nl158-np12-rf10 (query)                       509.32       203.92       713.23       0.9981          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np12-rf20 (query)                       509.32       317.23       826.54       0.9999          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np12-rfB (query)                        509.32       160.93       670.24       0.9904          1.0002            1.0000         4.59
IVF-RaBitQ-nl158-np17-rf10 (query)                       509.32       210.53       719.84       0.9981          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np17-rf20 (query)                       509.32       322.18       831.50       0.9999          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np17-rfB (query)                        509.32       162.21       671.52       0.9904          1.0002            1.0000         4.59
IVF-RaBitQ-nl158 (self)                                  509.32       454.26       963.57       0.9889          1.0003            1.0000         4.59
IVF-RaBitQ-nl223-np11-rf0 (query)                        479.22        89.34       568.56       0.6639          1.0274            1.0272         4.89
IVF-RaBitQ-nl223-np14-rf0 (query)                        479.22        91.08       570.30       0.6640          1.0274            1.0271         4.89
IVF-RaBitQ-nl223-np21-rf0 (query)                        479.22        96.68       575.90       0.6640          1.0274            1.0271         4.89
IVF-RaBitQ-nl223-np11-rf10 (query)                       479.22       207.83       687.05       0.9976          1.0001            1.0000         4.89
IVF-RaBitQ-nl223-np11-rf20 (query)                       479.22       316.94       796.16       0.9994          1.0000            1.0000         4.89
IVF-RaBitQ-nl223-np11-rfB (query)                        479.22       160.53       639.75       0.9899          1.0002            1.0000         4.89
IVF-RaBitQ-nl223-np14-rf10 (query)                       479.22       208.46       687.68       0.9982          1.0000            1.0000         4.89
IVF-RaBitQ-nl223-np14-rf20 (query)                       479.22       320.13       799.35       1.0000          1.0000            1.0000         4.89
IVF-RaBitQ-nl223-np14-rfB (query)                        479.22       161.80       641.02       0.9904          1.0002            1.0000         4.89
IVF-RaBitQ-nl223-np21-rf10 (query)                       479.22       215.32       694.54       0.9982          1.0000            1.0000         4.89
IVF-RaBitQ-nl223-np21-rf20 (query)                       479.22       330.60       809.82       1.0000          1.0000            1.0000         4.89
IVF-RaBitQ-nl223-np21-rfB (query)                        479.22       168.07       647.30       0.9904          1.0002            1.0000         4.89
IVF-RaBitQ-nl223 (self)                                  479.22       473.09       952.31       0.9892          1.0003            1.0000         4.89
IVF-RaBitQ-nl316-np15-rf0 (query)                        565.09        95.83       660.91       0.6654          1.0271            1.0269         5.35
IVF-RaBitQ-nl316-np17-rf0 (query)                        565.09        95.96       661.05       0.6654          1.0271            1.0269         5.35
IVF-RaBitQ-nl316-np25-rf0 (query)                        565.09       102.36       667.45       0.6654          1.0271            1.0269         5.35
IVF-RaBitQ-nl316-np15-rf10 (query)                       565.09       215.68       780.76       0.9983          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np15-rf20 (query)                       565.09       326.60       891.69       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np15-rfB (query)                        565.09       167.06       732.14       0.9905          1.0002            1.0000         5.35
IVF-RaBitQ-nl316-np17-rf10 (query)                       565.09       218.73       783.81       0.9983          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np17-rf20 (query)                       565.09       329.21       894.30       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np17-rfB (query)                        565.09       167.85       732.94       0.9905          1.0002            1.0000         5.35
IVF-RaBitQ-nl316-np25-rf10 (query)                       565.09       221.51       786.60       0.9983          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np25-rf20 (query)                       565.09       350.83       915.92       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np25-rfB (query)                        565.09       175.00       740.09       0.9905          1.0002            1.0000         5.35
IVF-RaBitQ-nl316 (self)                                  565.09       495.20     1_060.29       0.9892          1.0003            1.0000         5.35
IVF-RaBitQ-nl223-ex2-np14 (query)                        535.79       175.14       710.93       0.9255          1.0011            1.0008        11.38
IVF-RaBitQ-nl223-ex2-np21 (query)                        535.79       181.00       716.79       0.9255          1.0011            1.0008        11.38
IVF-RaBitQ-nl223-ex4-np14 (query)                        528.90       195.34       724.24       0.9701          1.0003            1.0000        17.48
IVF-RaBitQ-nl223-ex4-np21 (query)                        528.90       204.53       733.43       0.9701          1.0003            1.0000        17.48
IVF-RaBitQ-nl223-ex8-np14 (query)                        585.90       180.96       766.86       0.9872          1.0002            1.0000        29.69
IVF-RaBitQ-nl223-ex8-np21 (query)                        585.90       184.90       770.80       0.9872          1.0002            1.0000        29.69
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
Exhaustive (query)                                        99.98     1_973.27     2_073.25       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.98     6_668.56     6_768.54       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                             567.60       164.54       732.15       0.5777          1.0323            1.0331         6.17
ExhaustiveRaBitQ-rf5 (query)                             567.60       244.42       812.02       0.9286          1.0016            1.0006         6.17
ExhaustiveRaBitQ-rf10 (query)                            567.60       328.69       896.29       0.9841          1.0003            1.0000         6.17
ExhaustiveRaBitQ-rf20 (query)                            567.60       475.10     1_042.70       0.9987          1.0000            1.0000         6.17
ExhaustiveRaBitQ-rfB (query)                             567.60       338.83       906.44       0.9772          1.0004            1.0000         6.17
ExhaustiveRaBitQ (self)                                  567.60       988.01     1_555.61       0.9852          1.0003            1.0000         6.17
IVF-RaBitQ-nl158-np7-rf0 (query)                         678.97       104.69       783.66       0.5819          1.0317            1.0328         6.51
IVF-RaBitQ-nl158-np12-rf0 (query)                        678.97       108.97       787.94       0.5819          1.0317            1.0328         6.51
IVF-RaBitQ-nl158-np17-rf0 (query)                        678.97       115.62       794.59       0.5819          1.0317            1.0328         6.51
IVF-RaBitQ-nl158-np7-rf10 (query)                        678.97       247.33       926.30       0.9851          1.0003            1.0000         6.51
IVF-RaBitQ-nl158-np7-rf20 (query)                        678.97       380.01     1_058.98       0.9987          1.0000            1.0000         6.51
IVF-RaBitQ-nl158-np7-rfB (query)                         678.97       223.71       902.68       0.9776          1.0004            1.0000         6.51
IVF-RaBitQ-nl158-np12-rf10 (query)                       678.97       256.32       935.29       0.9851          1.0003            1.0000         6.51
IVF-RaBitQ-nl158-np12-rf20 (query)                       678.97       389.76     1_068.73       0.9987          1.0000            1.0000         6.51
IVF-RaBitQ-nl158-np12-rfB (query)                        678.97       236.46       915.43       0.9776          1.0004            1.0000         6.51
IVF-RaBitQ-nl158-np17-rf10 (query)                       678.97       259.79       938.76       0.9851          1.0003            1.0000         6.51
IVF-RaBitQ-nl158-np17-rf20 (query)                       678.97       397.88     1_076.85       0.9987          1.0000            1.0000         6.51
IVF-RaBitQ-nl158-np17-rfB (query)                        678.97       235.32       914.29       0.9776          1.0004            1.0000         6.51
IVF-RaBitQ-nl158 (self)                                  678.97       674.86     1_353.83       0.9755          1.0005            1.0000         6.51
IVF-RaBitQ-nl223-np11-rf0 (query)                        637.00       114.75       751.75       0.5848          1.0312            1.0323         6.97
IVF-RaBitQ-nl223-np14-rf0 (query)                        637.00       116.64       753.64       0.5848          1.0312            1.0323         6.97
IVF-RaBitQ-nl223-np21-rf0 (query)                        637.00       122.70       759.70       0.5848          1.0312            1.0323         6.97
IVF-RaBitQ-nl223-np11-rf10 (query)                       637.00       256.84       893.84       0.9857          1.0003            1.0000         6.97
IVF-RaBitQ-nl223-np11-rf20 (query)                       637.00       391.06     1_028.05       0.9988          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np11-rfB (query)                        637.00       228.56       865.56       0.9778          1.0004            1.0000         6.97
IVF-RaBitQ-nl223-np14-rf10 (query)                       637.00       258.40       895.40       0.9857          1.0003            1.0000         6.97
IVF-RaBitQ-nl223-np14-rf20 (query)                       637.00       402.25     1_039.25       0.9988          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np14-rfB (query)                        637.00       232.29       869.29       0.9778          1.0004            1.0000         6.97
IVF-RaBitQ-nl223-np21-rf10 (query)                       637.00       266.37       903.37       0.9857          1.0003            1.0000         6.97
IVF-RaBitQ-nl223-np21-rf20 (query)                       637.00       411.86     1_048.86       0.9988          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np21-rfB (query)                        637.00       238.92       875.92       0.9778          1.0004            1.0000         6.97
IVF-RaBitQ-nl223 (self)                                  637.00       695.27     1_332.26       0.9757          1.0005            1.0000         6.97
IVF-RaBitQ-nl316-np15-rf0 (query)                        739.65       129.19       868.84       0.5878          1.0308            1.0321         7.67
IVF-RaBitQ-nl316-np17-rf0 (query)                        739.65       127.21       866.87       0.5878          1.0308            1.0321         7.67
IVF-RaBitQ-nl316-np25-rf0 (query)                        739.65       134.27       873.93       0.5878          1.0308            1.0321         7.67
IVF-RaBitQ-nl316-np15-rf10 (query)                       739.65       272.89     1_012.55       0.9856          1.0003            1.0000         7.67
IVF-RaBitQ-nl316-np15-rf20 (query)                       739.65       408.02     1_147.67       0.9989          1.0000            1.0000         7.67
IVF-RaBitQ-nl316-np15-rfB (query)                        739.65       241.19       980.85       0.9774          1.0004            1.0000         7.67
IVF-RaBitQ-nl316-np17-rf10 (query)                       739.65       271.38     1_011.04       0.9856          1.0003            1.0000         7.67
IVF-RaBitQ-nl316-np17-rf20 (query)                       739.65       410.04     1_149.70       0.9989          1.0000            1.0000         7.67
IVF-RaBitQ-nl316-np17-rfB (query)                        739.65       240.92       980.57       0.9774          1.0004            1.0000         7.67
IVF-RaBitQ-nl316-np25-rf10 (query)                       739.65       280.87     1_020.52       0.9856          1.0003            1.0000         7.67
IVF-RaBitQ-nl316-np25-rf20 (query)                       739.65       425.32     1_164.97       0.9989          1.0000            1.0000         7.67
IVF-RaBitQ-nl316-np25-rfB (query)                        739.65       252.04       991.69       0.9774          1.0004            1.0000         7.67
IVF-RaBitQ-nl316 (self)                                  739.65       726.78     1_466.43       0.9763          1.0005            1.0000         7.67
IVF-RaBitQ-nl223-ex2-np14 (query)                        674.07       278.80       952.87       0.9140          1.0012            1.0008        16.49
IVF-RaBitQ-nl223-ex2-np21 (query)                        674.07       271.43       945.50       0.9140          1.0012            1.0008        16.49
IVF-RaBitQ-nl223-ex4-np14 (query)                        706.92       298.25     1_005.17       0.9588          1.0005            1.0000        25.65
IVF-RaBitQ-nl223-ex4-np21 (query)                        706.92       303.17     1_010.09       0.9588          1.0005            1.0000        25.65
IVF-RaBitQ-nl223-ex8-np14 (query)                        792.55       279.70     1_072.24       0.9748          1.0004            1.0000        43.96
IVF-RaBitQ-nl223-ex8-np21 (query)                        792.55       275.49     1_068.04       0.9748          1.0004            1.0000        43.96
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
Exhaustive (query)                                        33.07       729.32       762.39       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.07     2_419.32     2_452.39       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             385.86       100.28       486.14       0.8703          1.0284            1.0233         2.57
ExhaustiveRaBitQ-rf5 (query)                             385.86       167.45       553.31       1.0000          1.0000            1.0000         2.57
ExhaustiveRaBitQ-rf10 (query)                            385.86       232.50       618.36       1.0000          1.0000            1.0000         2.57
ExhaustiveRaBitQ-rf20 (query)                            385.86       340.82       726.68       1.0000          1.0000            1.0000         2.57
ExhaustiveRaBitQ-rfB (query)                             385.86       143.07       528.93       1.0000          1.0000            1.0000         2.57
ExhaustiveRaBitQ (self)                                  385.86       731.45     1_117.31       1.0000          1.0000            1.0000         2.57
IVF-RaBitQ-nl158-np7-rf0 (query)                         513.46        51.87       565.33       0.8729          1.0276            1.0225         2.68
IVF-RaBitQ-nl158-np12-rf0 (query)                        513.46        60.68       574.14       0.8735          1.0272            1.0223         2.68
IVF-RaBitQ-nl158-np17-rf0 (query)                        513.46        69.23       582.70       0.8735          1.0272            1.0223         2.68
IVF-RaBitQ-nl158-np7-rf10 (query)                        513.46       140.09       653.55       0.9976          1.0005            1.0000         2.68
IVF-RaBitQ-nl158-np7-rf20 (query)                        513.46       215.92       729.38       0.9976          1.0005            1.0000         2.68
IVF-RaBitQ-nl158-np7-rfB (query)                         513.46        77.01       590.48       0.9976          1.0005            1.0000         2.68
IVF-RaBitQ-nl158-np12-rf10 (query)                       513.46       149.77       663.23       0.9999          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np12-rf20 (query)                       513.46       229.64       743.10       0.9999          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np12-rfB (query)                        513.46        85.80       599.26       0.9999          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rf10 (query)                       513.46       159.96       673.42       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rf20 (query)                       513.46       240.70       754.16       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rfB (query)                        513.46        96.19       609.65       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158 (self)                                  513.46       284.38       797.84       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl223-np11-rf0 (query)                        530.71        53.42       584.14       0.8835          1.0228            1.0187         2.83
IVF-RaBitQ-nl223-np14-rf0 (query)                        530.71        57.40       588.11       0.8836          1.0228            1.0186         2.83
IVF-RaBitQ-nl223-np21-rf0 (query)                        530.71        64.18       594.90       0.8836          1.0228            1.0186         2.83
IVF-RaBitQ-nl223-np11-rf10 (query)                       530.71       142.42       673.14       0.9993          1.0001            1.0000         2.83
IVF-RaBitQ-nl223-np11-rf20 (query)                       530.71       214.50       745.22       0.9993          1.0001            1.0000         2.83
IVF-RaBitQ-nl223-np11-rfB (query)                        530.71        75.93       606.65       0.9993          1.0001            1.0000         2.83
IVF-RaBitQ-nl223-np14-rf10 (query)                       530.71       141.72       672.44       0.9998          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np14-rf20 (query)                       530.71       218.01       748.72       0.9998          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np14-rfB (query)                        530.71        79.03       609.74       0.9998          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np21-rf10 (query)                       530.71       150.75       681.46       1.0000          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np21-rf20 (query)                       530.71       227.50       758.21       1.0000          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np21-rfB (query)                        530.71        85.60       616.31       1.0000          1.0000            1.0000         2.83
IVF-RaBitQ-nl223 (self)                                  530.71       256.22       786.94       1.0000          1.0000            1.0000         2.83
IVF-RaBitQ-nl316-np15-rf0 (query)                        650.32        55.89       706.21       0.8898          1.0198            1.0159         3.06
IVF-RaBitQ-nl316-np17-rf0 (query)                        650.32        58.98       709.30       0.8899          1.0197            1.0159         3.06
IVF-RaBitQ-nl316-np25-rf0 (query)                        650.32        65.19       715.51       0.8900          1.0197            1.0159         3.06
IVF-RaBitQ-nl316-np15-rf10 (query)                       650.32       138.62       788.94       0.9997          1.0001            1.0000         3.06
IVF-RaBitQ-nl316-np15-rf20 (query)                       650.32       214.60       864.92       0.9997          1.0001            1.0000         3.06
IVF-RaBitQ-nl316-np15-rfB (query)                        650.32        76.54       726.86       0.9996          1.0001            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf10 (query)                       650.32       137.78       788.10       0.9998          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf20 (query)                       650.32       217.47       867.78       0.9998          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rfB (query)                        650.32        79.41       729.72       0.9998          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf10 (query)                       650.32       144.72       795.04       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf20 (query)                       650.32       225.81       876.13       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rfB (query)                        650.32        84.81       735.12       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316 (self)                                  650.32       253.85       904.17       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl223-ex2-np14 (query)                        548.64        84.71       633.35       0.9657          1.0014            1.0000         6.26
IVF-RaBitQ-nl223-ex2-np21 (query)                        548.64        92.34       640.98       0.9658          1.0014            1.0000         6.26
IVF-RaBitQ-nl223-ex4-np14 (query)                        551.82        89.35       641.17       0.9887          1.0001            1.0000         9.31
IVF-RaBitQ-nl223-ex4-np21 (query)                        551.82        97.90       649.72       0.9888          1.0001            1.0000         9.31
IVF-RaBitQ-nl223-ex8-np14 (query)                        580.19        85.53       665.72       0.9978          1.0000            1.0000        15.42
IVF-RaBitQ-nl223-ex8-np21 (query)                        580.19        94.83       675.02       0.9979          1.0000            1.0000        15.42
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
Exhaustive (query)                                        68.26     1_385.66     1_453.92       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.26     4_626.43     4_694.70       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                             600.10       158.78       758.88       0.9088          1.0131            1.0099         4.36
ExhaustiveRaBitQ-rf5 (query)                             600.10       236.85       836.95       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ-rf10 (query)                            600.10       311.84       911.94       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ-rf20 (query)                            600.10       456.19     1_056.29       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ-rfB (query)                             600.10       239.50       839.60       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ (self)                                  600.10       975.25     1_575.35       1.0000          1.0000            1.0000         4.36
IVF-RaBitQ-nl158-np7-rf0 (query)                         757.65        88.22       845.87       0.9142          1.0114            1.0084         4.59
IVF-RaBitQ-nl158-np12-rf0 (query)                        757.65       106.80       864.44       0.9148          1.0112            1.0082         4.59
IVF-RaBitQ-nl158-np17-rf0 (query)                        757.65       114.91       872.55       0.9148          1.0112            1.0082         4.59
IVF-RaBitQ-nl158-np7-rf10 (query)                        757.65       205.37       963.01       0.9986          1.0003            1.0000         4.59
IVF-RaBitQ-nl158-np7-rf20 (query)                        757.65       318.32     1_075.97       0.9986          1.0003            1.0000         4.59
IVF-RaBitQ-nl158-np7-rfB (query)                         757.65       126.32       883.96       0.9986          1.0003            1.0000         4.59
IVF-RaBitQ-nl158-np12-rf10 (query)                       757.65       220.24       977.88       0.9999          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np12-rf20 (query)                       757.65       335.97     1_093.62       0.9999          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np12-rfB (query)                        757.65       139.02       896.67       0.9999          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np17-rf10 (query)                       757.65       228.94       986.59       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np17-rf20 (query)                       757.65       348.58     1_106.23       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np17-rfB (query)                        757.65       151.97       909.61       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158 (self)                                  757.65       445.50     1_203.15       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl223-np11-rf0 (query)                        691.33        92.52       783.85       0.9214          1.0093            1.0068         4.90
IVF-RaBitQ-nl223-np14-rf0 (query)                        691.33        97.34       788.67       0.9215          1.0093            1.0068         4.90
IVF-RaBitQ-nl223-np21-rf0 (query)                        691.33       111.61       802.94       0.9215          1.0093            1.0068         4.90
IVF-RaBitQ-nl223-np11-rf10 (query)                       691.33       211.62       902.95       0.9997          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np11-rf20 (query)                       691.33       317.52     1_008.85       0.9997          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np11-rfB (query)                        691.33       125.45       816.78       0.9997          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np14-rf10 (query)                       691.33       211.62       902.95       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np14-rf20 (query)                       691.33       326.33     1_017.66       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np14-rfB (query)                        691.33       130.49       821.82       0.9999          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np21-rf10 (query)                       691.33       224.18       915.51       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np21-rf20 (query)                       691.33       340.26     1_031.59       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np21-rfB (query)                        691.33       153.93       845.26       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223 (self)                                  691.33       437.84     1_129.17       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl316-np15-rf0 (query)                        827.34        98.23       925.57       0.9266          1.0081            1.0057         5.36
IVF-RaBitQ-nl316-np17-rf0 (query)                        827.34       118.88       946.22       0.9266          1.0081            1.0057         5.36
IVF-RaBitQ-nl316-np25-rf0 (query)                        827.34       113.97       941.31       0.9266          1.0081            1.0057         5.36
IVF-RaBitQ-nl316-np15-rf10 (query)                       827.34       214.02     1_041.36       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np15-rf20 (query)                       827.34       331.01     1_158.35       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np15-rfB (query)                        827.34       130.07       957.42       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np17-rf10 (query)                       827.34       213.76     1_041.11       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np17-rf20 (query)                       827.34       329.87     1_157.21       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np17-rfB (query)                        827.34       133.02       960.37       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np25-rf10 (query)                       827.34       225.93     1_053.27       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np25-rf20 (query)                       827.34       344.07     1_171.41       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np25-rfB (query)                        827.34       141.95       969.29       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl316 (self)                                  827.34       430.89     1_258.23       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl223-ex2-np14 (query)                        760.03       160.55       920.58       0.9760          1.0007            1.0000        11.41
IVF-RaBitQ-nl223-ex2-np21 (query)                        760.03       170.48       930.51       0.9761          1.0007            1.0000        11.41
IVF-RaBitQ-nl223-ex4-np14 (query)                        773.52       151.02       924.53       0.9922          1.0001            1.0000        17.51
IVF-RaBitQ-nl223-ex4-np21 (query)                        773.52       163.03       936.55       0.9923          1.0001            1.0000        17.51
IVF-RaBitQ-nl223-ex8-np14 (query)                        831.01       144.24       975.26       0.9989          1.0000            1.0000        29.72
IVF-RaBitQ-nl223-ex8-np21 (query)                        831.01       156.69       987.70       0.9989          1.0000            1.0000        29.72
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
Exhaustive (query)                                       100.06     1_973.74     2_073.80       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.06     6_962.00     7_062.06       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                             874.23       205.09     1_079.31       0.9143          1.0117            1.0085         6.16
ExhaustiveRaBitQ-rf5 (query)                             874.23       292.90     1_167.12       1.0000          1.0000            1.0000         6.16
ExhaustiveRaBitQ-rf10 (query)                            874.23       400.38     1_274.61       1.0000          1.0000            1.0000         6.16
ExhaustiveRaBitQ-rf20 (query)                            874.23       557.06     1_431.29       1.0000          1.0000            1.0000         6.16
ExhaustiveRaBitQ-rfB (query)                             874.23       347.38     1_221.61       0.9999          1.0000            1.0000         6.16
ExhaustiveRaBitQ (self)                                  874.23     1_190.53     2_064.76       1.0000          1.0000            1.0000         6.16
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_134.21       120.00     1_254.20       0.9165          1.0109            1.0078         6.51
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_134.21       131.96     1_266.17       0.9166          1.0108            1.0077         6.51
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_134.21       149.02     1_283.22       0.9166          1.0108            1.0077         6.51
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_134.21       255.74     1_389.95       0.9995          1.0001            1.0000         6.51
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_134.21       389.69     1_523.90       0.9995          1.0001            1.0000         6.51
IVF-RaBitQ-nl158-np7-rfB (query)                       1_134.21       161.30     1_295.51       0.9995          1.0001            1.0000         6.51
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_134.21       271.26     1_405.46       1.0000          1.0000            1.0000         6.51
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_134.21       413.23     1_547.44       1.0000          1.0000            1.0000         6.51
IVF-RaBitQ-nl158-np12-rfB (query)                      1_134.21       178.66     1_312.86       0.9999          1.0000            1.0000         6.51
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_134.21       288.61     1_422.82       1.0000          1.0000            1.0000         6.51
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_134.21       439.39     1_573.59       1.0000          1.0000            1.0000         6.51
IVF-RaBitQ-nl158-np17-rfB (query)                      1_134.21       194.62     1_328.83       0.9999          1.0000            1.0000         6.51
IVF-RaBitQ-nl158 (self)                                1_134.21       569.26     1_703.46       1.0000          1.0000            1.0000         6.51
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_144.37       124.19     1_268.57       0.9214          1.0096            1.0069         6.97
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_144.37       129.13     1_273.50       0.9214          1.0096            1.0069         6.97
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_144.37       143.41     1_287.78       0.9214          1.0096            1.0069         6.97
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_144.37       261.58     1_405.96       0.9999          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_144.37       393.89     1_538.26       0.9999          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np11-rfB (query)                      1_144.37       165.19     1_309.57       0.9998          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_144.37       267.16     1_411.54       1.0000          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_144.37       417.98     1_562.35       1.0000          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np14-rfB (query)                      1_144.37       172.83     1_317.21       0.9999          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_144.37       284.11     1_428.48       1.0000          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_144.37       427.33     1_571.70       1.0000          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np21-rfB (query)                      1_144.37       188.22     1_332.59       0.9999          1.0000            1.0000         6.97
IVF-RaBitQ-nl223 (self)                                1_144.37       556.97     1_701.34       0.9999          1.0000            1.0000         6.97
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_321.52       128.94     1_450.46       0.9262          1.0084            1.0059         7.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_321.52       132.93     1_454.45       0.9262          1.0084            1.0059         7.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_321.52       147.09     1_468.61       0.9262          1.0084            1.0059         7.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_321.52       273.25     1_594.77       1.0000          1.0000            1.0000         7.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_321.52       411.29     1_732.81       1.0000          1.0000            1.0000         7.63
IVF-RaBitQ-nl316-np15-rfB (query)                      1_321.52       171.20     1_492.72       0.9999          1.0000            1.0000         7.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_321.52       274.43     1_595.95       1.0000          1.0000            1.0000         7.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_321.52       417.30     1_738.82       1.0000          1.0000            1.0000         7.63
IVF-RaBitQ-nl316-np17-rfB (query)                      1_321.52       175.51     1_497.03       0.9999          1.0000            1.0000         7.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_321.52       291.59     1_613.11       1.0000          1.0000            1.0000         7.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_321.52       450.39     1_771.91       1.0000          1.0000            1.0000         7.63
IVF-RaBitQ-nl316-np25-rfB (query)                      1_321.52       189.86     1_511.38       0.9999          1.0000            1.0000         7.63
IVF-RaBitQ-nl316 (self)                                1_321.52       562.07     1_883.59       0.9999          1.0000            1.0000         7.63
IVF-RaBitQ-nl223-ex2-np14 (query)                      1_194.52       187.01     1_381.53       0.9649          1.0018            1.0000        16.50
IVF-RaBitQ-nl223-ex2-np21 (query)                      1_194.52       203.83     1_398.34       0.9649          1.0018            1.0000        16.50
IVF-RaBitQ-nl223-ex4-np14 (query)                      1_220.73       202.18     1_422.91       0.9780          1.0007            1.0000        25.66
IVF-RaBitQ-nl223-ex4-np21 (query)                      1_220.73       216.60     1_437.33       0.9780          1.0007            1.0000        25.66
IVF-RaBitQ-nl223-ex8-np14 (query)                      1_310.61       194.27     1_504.87       0.9839          1.0004            1.0000        43.97
IVF-RaBitQ-nl223-ex8-np21 (query)                      1_310.61       206.85     1_517.46       0.9840          1.0004            1.0000        43.97
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
Exhaustive (query)                                        32.82       708.64       741.46       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.82     2_269.00     2_301.82       1.0000          1.0000            1.0000        48.83
QG-d32-l64-ef15 (query)                                2_566.28       299.64     2_865.92       0.9875          1.3883            1.0000       116.16
QG-d32-l64-ef30 (query)                                2_566.28       425.21     2_991.49       0.9935          1.1300            1.0000       116.16
QG-d32-l64-ef60 (query)                                2_566.28       613.94     3_180.22       0.9957          1.0193            1.0000       116.16
QG-d32-l64-ef120 (query)                               2_566.28       900.67     3_466.96       0.9970          1.0070            1.0000       116.16
QG-d32-l64 (self)                                      2_566.28     2_074.75     4_641.03       0.9960          1.0191            1.0000       116.16
QG-d32-l128-ef15 (query)                               3_844.05       308.61     4_152.66       0.9902          1.0028            1.0000       116.16
QG-d32-l128-ef30 (query)                               3_844.05       445.14     4_289.19       0.9947          1.0021            1.0000       116.16
QG-d32-l128-ef60 (query)                               3_844.05       664.10     4_508.15       0.9963          1.0019            1.0000       116.16
QG-d32-l128-ef120 (query)                              3_844.05       981.26     4_825.31       0.9969          1.0018            1.0000       116.16
QG-d32-l128 (self)                                     3_844.05     2_459.32     6_303.37       0.9964          1.0019            1.0000       116.16
QG-d64-l64-ef15 (query)                                9_136.87       664.57     9_801.44       0.9994          1.0003            1.0000       183.30
QG-d64-l64-ef30 (query)                                9_136.87       918.80    10_055.67       0.9996          1.0002            1.0000       183.30
QG-d64-l64-ef60 (query)                                9_136.87     1_282.87    10_419.74       0.9998          1.0002            1.0000       183.30
QG-d64-l64-ef120 (query)                               9_136.87     1_826.98    10_963.86       0.9998          1.0002            1.0000       183.30
QG-d64-l64 (self)                                      9_136.87     4_135.61    13_272.48       0.9998          1.0002            1.0000       183.30
QG-d64-l128-ef15 (query)                               9_649.64       662.30    10_311.94       0.9995          1.0002            1.0000       183.30
QG-d64-l128-ef30 (query)                               9_649.64       914.64    10_564.28       0.9997          1.0002            1.0000       183.30
QG-d64-l128-ef60 (query)                               9_649.64     1_255.80    10_905.44       0.9998          1.0002            1.0000       183.30
QG-d64-l128-ef120 (query)                              9_649.64     1_843.40    11_493.04       0.9998          1.0001            1.0000       183.30
QG-d64-l128 (self)                                     9_649.64     4_153.61    13_803.25       0.9998          1.0002            1.0000       183.30
HnswRaBitQ-m16-ex1-ef15 (query)                        1_303.70        91.10     1_394.80       0.7665          1.9558            1.0083        10.85
HnswRaBitQ-m16-ex1-ef30 (query)                        1_303.70       112.63     1_416.33       0.7751          1.3324            1.0081        10.85
HnswRaBitQ-m16-ex1-ef60 (query)                        1_303.70       147.26     1_450.96       0.7771          1.0094            1.0080        10.85
HnswRaBitQ-m16-ex1-ef120 (query)                       1_303.70       181.47     1_485.17       0.7773          1.0090            1.0080        10.85
HnswRaBitQ-m16-ex1 (self)                              1_303.70     4_617.76     5_921.46       0.6964          1.0185            1.0161        10.85
HnswRaBitQ-m16-ex3-ef15 (query)                        1_371.94        96.30     1_468.25       0.8986          1.4474            1.0010        13.90
HnswRaBitQ-m16-ex3-ef30 (query)                        1_371.94       119.65     1_491.60       0.9197          1.0157            1.0007        13.90
HnswRaBitQ-m16-ex3-ef60 (query)                        1_371.94       150.05     1_522.00       0.9264          1.0013            1.0006        13.90
HnswRaBitQ-m16-ex3-ef120 (query)                       1_371.94       189.55     1_561.50       0.9278          1.0009            1.0005        13.90
HnswRaBitQ-m16-ex3 (self)                              1_371.94     4_576.55     5_948.50       0.9030          1.0019            1.0012        13.90
HnswRaBitQ-m16-ex5-ef15 (query)                        1_474.93       111.69     1_586.62       0.9292          1.7691            1.0001        16.95
HnswRaBitQ-m16-ex5-ef30 (query)                        1_474.93       134.06     1_608.99       0.9588          1.5277            1.0000        16.95
HnswRaBitQ-m16-ex5-ef60 (query)                        1_474.93       170.17     1_645.11       0.9730          1.0441            1.0000        16.95
HnswRaBitQ-m16-ex5-ef120 (query)                       1_474.93       209.78     1_684.72       0.9758          1.0007            1.0000        16.95
HnswRaBitQ-m16-ex5 (self)                              1_474.93     4_868.70     6_343.63       0.9687          1.0021            1.0000        16.95
HnswRaBitQ-m16-ex8-ef15 (query)                        2_228.78       103.80     2_332.58       0.9440          1.6888            1.0000        21.53
HnswRaBitQ-m16-ex8-ef30 (query)                        2_228.78       129.07     2_357.84       0.9777          1.0044            1.0000        21.53
HnswRaBitQ-m16-ex8-ef60 (query)                        2_228.78       162.06     2_390.83       0.9904          1.0011            1.0000        21.53
HnswRaBitQ-m16-ex8-ef120 (query)                       2_228.78       210.13     2_438.90       0.9936          1.0005            1.0000        21.53
HnswRaBitQ-m16-ex8 (self)                              2_228.78     4_896.38     7_125.15       0.9927          1.0008            1.0000        21.53
HnswRaBitQ-m32-ex1-ef15 (query)                        1_364.71       101.22     1_465.93       0.7729          1.2054            1.0080        16.95
HnswRaBitQ-m32-ex1-ef30 (query)                        1_364.71       126.62     1_491.33       0.7769          1.0119            1.0080        16.95
HnswRaBitQ-m32-ex1-ef60 (query)                        1_364.71       155.16     1_519.88       0.7776          1.0085            1.0080        16.95
HnswRaBitQ-m32-ex1-ef120 (query)                       1_364.71       196.14     1_560.85       0.7776          1.0085            1.0080        16.95
HnswRaBitQ-m32-ex1 (self)                              1_364.71     5_777.91     7_142.62       0.6963          1.0172            1.0162        16.95
HnswRaBitQ-m32-ex3-ef15 (query)                        1_417.26       109.57     1_526.82       0.9157          1.0122            1.0007        20.00
HnswRaBitQ-m32-ex3-ef30 (query)                        1_417.26       138.52     1_555.78       0.9254          1.0020            1.0006        20.00
HnswRaBitQ-m32-ex3-ef60 (query)                        1_417.26       178.81     1_596.07       0.9276          1.0011            1.0005        20.00
HnswRaBitQ-m32-ex3-ef120 (query)                       1_417.26       222.24     1_639.50       0.9282          1.0008            1.0005        20.00
HnswRaBitQ-m32-ex3 (self)                              1_417.26     5_416.08     6_833.34       0.9041          1.0016            1.0012        20.00
HnswRaBitQ-m32-ex5-ef15 (query)                        1_553.18       119.40     1_672.58       0.9582          1.0114            1.0000        23.06
HnswRaBitQ-m32-ex5-ef30 (query)                        1_553.18       147.04     1_700.22       0.9715          1.0030            1.0000        23.06
HnswRaBitQ-m32-ex5-ef60 (query)                        1_553.18       180.63     1_733.81       0.9759          1.0004            1.0000        23.06
HnswRaBitQ-m32-ex5-ef120 (query)                       1_553.18       240.74     1_793.91       0.9767          1.0002            1.0000        23.06
HnswRaBitQ-m32-ex5 (self)                              1_553.18     5_423.70     6_976.88       0.9709          1.0003            1.0000        23.06
HnswRaBitQ-m32-ex8-ef15 (query)                        2_319.31       122.56     2_441.87       0.9724          1.0047            1.0000        27.63
HnswRaBitQ-m32-ex8-ef30 (query)                        2_319.31       146.23     2_465.54       0.9884          1.0013            1.0000        27.63
HnswRaBitQ-m32-ex8-ef60 (query)                        2_319.31       181.82     2_501.13       0.9931          1.0005            1.0000        27.63
HnswRaBitQ-m32-ex8-ef120 (query)                       2_319.31       234.91     2_554.23       0.9944          1.0001            1.0000        27.63
HnswRaBitQ-m32-ex8 (self)                              2_319.31     5_371.99     7_691.31       0.9950          1.0002            1.0000        27.63
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
Exhaustive (query)                                        68.42     1_382.24     1_450.66       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.42     4_625.85     4_694.27       1.0000          1.0000            1.0000        97.66
QG-d32-l64-ef15 (query)                                5_019.93       527.59     5_547.53       0.9775          1.5630            1.0000       213.81
QG-d32-l64-ef30 (query)                                5_019.93       687.54     5_707.47       0.9834          1.4629            1.0000       213.81
QG-d32-l64-ef60 (query)                                5_019.93       970.26     5_990.20       0.9876          1.1460            1.0000       213.81
QG-d32-l64-ef120 (query)                               5_019.93     1_274.67     6_294.61       0.9900          1.0411            1.0000       213.81
QG-d32-l64 (self)                                      5_019.93     3_049.49     8_069.42       0.9873          1.1457            1.0000       213.81
QG-d32-l128-ef15 (query)                               5_667.36       506.68     6_174.04       0.9885          1.0032            1.0000       213.81
QG-d32-l128-ef30 (query)                               5_667.36       681.37     6_348.73       0.9935          1.0029            1.0000       213.81
QG-d32-l128-ef60 (query)                               5_667.36       928.03     6_595.40       0.9956          1.0028            1.0000       213.81
QG-d32-l128-ef120 (query)                              5_667.36     1_269.57     6_936.93       0.9965          1.0027            1.0000       213.81
QG-d32-l128 (self)                                     5_667.36     3_224.96     8_892.32       0.9955          1.0029            1.0000       213.81
QG-d64-l64-ef15 (query)                               18_368.86     1_029.20    19_398.06       0.9984          1.0009            1.0000       329.78
QG-d64-l64-ef30 (query)                               18_368.86     1_328.13    19_696.99       0.9989          1.0009            1.0000       329.78
QG-d64-l64-ef60 (query)                               18_368.86     1_759.17    20_128.02       0.9991          1.0008            1.0000       329.78
QG-d64-l64-ef120 (query)                              18_368.86     2_392.38    20_761.24       0.9992          1.0008            1.0000       329.78
QG-d64-l64 (self)                                     18_368.86     5_764.44    24_133.29       0.9990          1.0008            1.0000       329.78
QG-d64-l128-ef15 (query)                              19_103.34     1_027.94    20_131.28       0.9984          1.0009            1.0000       329.78
QG-d64-l128-ef30 (query)                              19_103.34     1_341.32    20_444.66       0.9988          1.0009            1.0000       329.78
QG-d64-l128-ef60 (query)                              19_103.34     1_752.05    20_855.39       0.9991          1.0008            1.0000       329.78
QG-d64-l128-ef120 (query)                             19_103.34     2_363.06    21_466.40       0.9992          1.0008            1.0000       329.78
QG-d64-l128 (self)                                    19_103.34     5_760.76    24_864.10       0.9990          1.0008            1.0000       329.78
HnswRaBitQ-m16-ex1-ef15 (query)                        2_440.03       169.56     2_609.59       0.7475         60.5939            1.0058        14.15
HnswRaBitQ-m16-ex1-ef30 (query)                        2_440.03       229.51     2_669.55       0.7724          1.1764            1.0055        14.15
HnswRaBitQ-m16-ex1-ef60 (query)                        2_440.03       254.69     2_694.72       0.7757          1.0063            1.0054        14.15
HnswRaBitQ-m16-ex1-ef120 (query)                       2_440.03       313.08     2_753.11       0.7763          1.0060            1.0054        14.15
HnswRaBitQ-m16-ex1 (self)                              2_440.03     9_449.64    11_889.67       0.6929          1.0129            1.0110        14.15
HnswRaBitQ-m16-ex3-ef15 (query)                        2_531.83       189.50     2_721.33       0.8898          1.0793            1.0008        20.25
HnswRaBitQ-m16-ex3-ef30 (query)                        2_531.83       226.66     2_758.49       0.9144          1.0124            1.0005        20.25
HnswRaBitQ-m16-ex3-ef60 (query)                        2_531.83       276.67     2_808.50       0.9241          1.0012            1.0004        20.25
HnswRaBitQ-m16-ex3-ef120 (query)                       2_531.83       343.32     2_875.15       0.9266          1.0007            1.0004        20.25
HnswRaBitQ-m16-ex3 (self)                              2_531.83     9_742.91    12_274.74       0.8998          1.0016            1.0008        20.25
HnswRaBitQ-m16-ex5-ef15 (query)                        2_732.69       204.13     2_936.83       0.9229          1.0684            1.0002        26.36
HnswRaBitQ-m16-ex5-ef30 (query)                        2_732.69       246.74     2_979.43       0.9568          1.0181            1.0000        26.36
HnswRaBitQ-m16-ex5-ef60 (query)                        2_732.69       304.94     3_037.63       0.9708          1.0009            1.0000        26.36
HnswRaBitQ-m16-ex5-ef120 (query)                       2_732.69       364.84     3_097.54       0.9746          1.0004            1.0000        26.36
HnswRaBitQ-m16-ex5 (self)                              2_732.69     9_972.48    12_705.17       0.9656          1.0011            1.0000        26.36
HnswRaBitQ-m16-ex8-ef15 (query)                        4_256.74       196.34     4_453.09       0.9337          1.0203            1.0000        35.51
HnswRaBitQ-m16-ex8-ef30 (query)                        4_256.74       236.14     4_492.88       0.9720          1.0033            1.0000        35.51
HnswRaBitQ-m16-ex8-ef60 (query)                        4_256.74       291.26     4_548.00       0.9880          1.0007            1.0000        35.51
HnswRaBitQ-m16-ex8-ef120 (query)                       4_256.74       360.45     4_617.20       0.9926          1.0003            1.0000        35.51
HnswRaBitQ-m16-ex8 (self)                              4_256.74     9_759.73    14_016.47       0.9898          1.0008            1.0000        35.51
HnswRaBitQ-m32-ex1-ef15 (query)                        2_464.62       196.98     2_661.60       0.7717          1.0720            1.0054        20.25
HnswRaBitQ-m32-ex1-ef30 (query)                        2_464.62       240.49     2_705.12       0.7755          1.0312            1.0054        20.25
HnswRaBitQ-m32-ex1-ef60 (query)                        2_464.62       287.94     2_752.57       0.7767          1.0057            1.0054        20.25
HnswRaBitQ-m32-ex1-ef120 (query)                       2_464.62       345.40     2_810.03       0.7768          1.0057            1.0054        20.25
HnswRaBitQ-m32-ex1 (self)                              2_464.62    11_800.03    14_264.65       0.6929          1.0122            1.0110        20.25
HnswRaBitQ-m32-ex3-ef15 (query)                        2_641.42       220.39     2_861.80       0.9054          2.6492            1.0005        26.36
HnswRaBitQ-m32-ex3-ef30 (query)                        2_641.42       266.15     2_907.57       0.9225          1.0072            1.0004        26.36
HnswRaBitQ-m32-ex3-ef60 (query)                        2_641.42       316.18     2_957.59       0.9262          1.0009            1.0004        26.36
HnswRaBitQ-m32-ex3-ef120 (query)                       2_641.42       377.38     3_018.80       0.9270          1.0006            1.0004        26.36
HnswRaBitQ-m32-ex3 (self)                              2_641.42    12_262.51    14_903.93       0.9017          1.0013            1.0008        26.36
HnswRaBitQ-m32-ex5-ef15 (query)                        2_938.41       237.52     3_175.93       0.9522          1.0141            1.0000        32.46
HnswRaBitQ-m32-ex5-ef30 (query)                        2_938.41       286.90     3_225.31       0.9687          1.0046            1.0000        32.46
HnswRaBitQ-m32-ex5-ef60 (query)                        2_938.41       338.96     3_277.37       0.9740          1.0011            1.0000        32.46
HnswRaBitQ-m32-ex5-ef120 (query)                       2_938.41       402.13     3_340.53       0.9754          1.0002            1.0000        32.46
HnswRaBitQ-m32-ex5 (self)                              2_938.41    12_527.05    15_465.45       0.9685          1.0004            1.0000        32.46
HnswRaBitQ-m32-ex8-ef15 (query)                        4_426.39       230.74     4_657.13       0.9651          1.0146            1.0000        41.62
HnswRaBitQ-m32-ex8-ef30 (query)                        4_426.39       277.91     4_704.30       0.9853          1.0030            1.0000        41.62
HnswRaBitQ-m32-ex8-ef60 (query)                        4_426.39       332.58     4_758.97       0.9920          1.0002            1.0000        41.62
HnswRaBitQ-m32-ex8-ef120 (query)                       4_426.39       394.56     4_820.95       0.9935          1.0001            1.0000        41.62
HnswRaBitQ-m32-ex8 (self)                              4_426.39    12_189.51    16_615.90       0.9937          1.0003            1.0000        41.62
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
Exhaustive (query)                                       101.94     1_964.29     2_066.23       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.94     6_693.38     6_795.33       1.0000          1.0000            1.0000       146.48
QG-d32-l64-ef15 (query)                                7_343.94       652.31     7_996.25       0.9854          1.0044            1.0000       311.47
QG-d32-l64-ef30 (query)                                7_343.94       863.85     8_207.78       0.9909          1.0041            1.0000       311.47
QG-d32-l64-ef60 (query)                                7_343.94     1_166.91     8_510.85       0.9935          1.0040            1.0000       311.47
QG-d32-l64-ef120 (query)                               7_343.94     1_554.69     8_898.62       0.9946          1.0039            1.0000       311.47
QG-d32-l64 (self)                                      7_343.94     3_708.12    11_052.06       0.9936          1.0039            1.0000       311.47
QG-d32-l128-ef15 (query)                               8_109.81       658.94     8_768.75       0.9861          1.0042            1.0000       311.47
QG-d32-l128-ef30 (query)                               8_109.81       865.32     8_975.14       0.9914          1.0039            1.0000       311.47
QG-d32-l128-ef60 (query)                               8_109.81     1_147.29     9_257.10       0.9940          1.0038            1.0000       311.47
QG-d32-l128-ef120 (query)                              8_109.81     1_592.91     9_702.73       0.9950          1.0037            1.0000       311.47
QG-d32-l128 (self)                                     8_109.81     4_361.82    12_471.63       0.9941          1.0037            1.0000       311.47
QG-d64-l64-ef15 (query)                               28_548.13     1_357.39    29_905.52       0.9976          1.0013            1.0000       476.27
QG-d64-l64-ef30 (query)                               28_548.13     1_717.95    30_266.08       0.9982          1.0013            1.0000       476.27
QG-d64-l64-ef60 (query)                               28_548.13     2_193.61    30_741.74       0.9985          1.0013            1.0000       476.27
QG-d64-l64-ef120 (query)                              28_548.13     2_911.35    31_459.49       0.9986          1.0012            1.0000       476.27
QG-d64-l64 (self)                                     28_548.13     7_210.36    35_758.49       0.9986          1.0011            1.0000       476.27
QG-d64-l128-ef15 (query)                              28_112.75     1_359.15    29_471.90       0.9963          1.0045            1.0000       476.27
QG-d64-l128-ef30 (query)                              28_112.75     1_716.87    29_829.62       0.9969          1.0045            1.0000       476.27
QG-d64-l128-ef60 (query)                              28_112.75     2_194.91    30_307.66       0.9972          1.0044            1.0000       476.27
QG-d64-l128-ef120 (query)                             28_112.75     2_883.73    30_996.47       0.9973          1.0044            1.0000       476.27
QG-d64-l128 (self)                                    28_112.75     7_882.41    35_995.16       0.9969          1.0056            1.0000       476.27
HnswRaBitQ-m16-ex1-ef15 (query)                        3_313.47       241.88     3_555.35       0.7628          1.0566            1.0046        17.45
HnswRaBitQ-m16-ex1-ef30 (query)                        3_313.47       291.52     3_604.99       0.7734          1.0114            1.0043        17.45
HnswRaBitQ-m16-ex1-ef60 (query)                        3_313.47       354.09     3_667.56       0.7770          1.0057            1.0042        17.45
HnswRaBitQ-m16-ex1-ef120 (query)                       3_313.47       430.51     3_743.98       0.7779          1.0046            1.0042        17.45
HnswRaBitQ-m16-ex1 (self)                              3_313.47    14_093.45    17_406.91       0.6922          1.0113            1.0088        17.45
HnswRaBitQ-m16-ex3-ef15 (query)                        3_570.76       262.37     3_833.13       0.8850          1.1088            1.0007        26.61
HnswRaBitQ-m16-ex3-ef30 (query)                        3_570.76       319.59     3_890.35       0.9113          1.0503            1.0004        26.61
HnswRaBitQ-m16-ex3-ef60 (query)                        3_570.76       400.09     3_970.85       0.9235          1.0063            1.0003        26.61
HnswRaBitQ-m16-ex3-ef120 (query)                       3_570.76       474.04     4_044.80       0.9266          1.0008            1.0003        26.61
HnswRaBitQ-m16-ex3 (self)                              3_570.76    14_411.55    17_982.31       0.8981          1.0054            1.0007        26.61
HnswRaBitQ-m16-ex5-ef15 (query)                        3_905.79       286.66     4_192.44       0.9149          1.8398            1.0002        35.76
HnswRaBitQ-m16-ex5-ef30 (query)                        3_905.79       354.07     4_259.85       0.9511          1.7320            1.0000        35.76
HnswRaBitQ-m16-ex5-ef60 (query)                        3_905.79       428.24     4_334.03       0.9704          1.0015            1.0000        35.76
HnswRaBitQ-m16-ex5-ef120 (query)                       3_905.79       515.49     4_421.28       0.9747          1.0003            1.0000        35.76
HnswRaBitQ-m16-ex5 (self)                              3_905.79    14_796.41    18_702.19       0.9645          1.0011            1.0000        35.76
HnswRaBitQ-m16-ex8-ef15 (query)                        6_228.43       270.38     6_498.82       0.9264          1.1449            1.0001        49.50
HnswRaBitQ-m16-ex8-ef30 (query)                        6_228.43       333.99     6_562.43       0.9677          1.1067            1.0000        49.50
HnswRaBitQ-m16-ex8-ef60 (query)                        6_228.43       398.24     6_626.67       0.9873          1.0006            1.0000        49.50
HnswRaBitQ-m16-ex8-ef120 (query)                       6_228.43       483.65     6_712.09       0.9923          1.0004            1.0000        49.50
HnswRaBitQ-m16-ex8 (self)                              6_228.43    14_373.10    20_601.53       0.9891          1.0007            1.0000        49.50
HnswRaBitQ-m32-ex1-ef15 (query)                        3_369.93       284.74     3_654.67       0.7736          1.0678            1.0043        23.56
HnswRaBitQ-m32-ex1-ef30 (query)                        3_369.93       339.44     3_709.37       0.7769          1.0338            1.0042        23.56
HnswRaBitQ-m32-ex1-ef60 (query)                        3_369.93       407.57     3_777.50       0.7778          1.0047            1.0042        23.56
HnswRaBitQ-m32-ex1-ef120 (query)                       3_369.93       480.79     3_850.72       0.7780          1.0045            1.0042        23.56
HnswRaBitQ-m32-ex1 (self)                              3_369.93    17_917.82    21_287.75       0.6923          1.0095            1.0088        23.56
HnswRaBitQ-m32-ex3-ef15 (query)                        3_693.31       318.58     4_011.89       0.9076          1.2160            1.0004        32.71
HnswRaBitQ-m32-ex3-ef30 (query)                        3_693.31       381.76     4_075.07       0.9221          1.0060            1.0003        32.71
HnswRaBitQ-m32-ex3-ef60 (query)                        3_693.31       451.13     4_144.44       0.9267          1.0007            1.0003        32.71
HnswRaBitQ-m32-ex3-ef120 (query)                       3_693.31       535.29     4_228.61       0.9275          1.0005            1.0003        32.71
HnswRaBitQ-m32-ex3 (self)                              3_693.31    18_707.38    22_400.69       0.9009          1.0011            1.0006        32.71
HnswRaBitQ-m32-ex5-ef15 (query)                        4_238.14       345.70     4_583.84       0.9507          1.0057            1.0000        41.87
HnswRaBitQ-m32-ex5-ef30 (query)                        4_238.14       414.03     4_652.16       0.9675          1.0015            1.0000        41.87
HnswRaBitQ-m32-ex5-ef60 (query)                        4_238.14       487.75     4_725.89       0.9743          1.0003            1.0000        41.87
HnswRaBitQ-m32-ex5-ef120 (query)                       4_238.14       565.78     4_803.91       0.9756          1.0001            1.0000        41.87
HnswRaBitQ-m32-ex5 (self)                              4_238.14    18_725.20    22_963.34       0.9679          1.0003            1.0000        41.87
HnswRaBitQ-m32-ex8-ef15 (query)                        6_263.10       317.78     6_580.88       0.9614          1.3118            1.0000        55.60
HnswRaBitQ-m32-ex8-ef30 (query)                        6_263.10       382.59     6_645.69       0.9830          1.3016            1.0000        55.60
HnswRaBitQ-m32-ex8-ef60 (query)                        6_263.10       457.85     6_720.95       0.9909          1.2071            1.0000        55.60
HnswRaBitQ-m32-ex8-ef120 (query)                       6_263.10       531.08     6_794.19       0.9934          1.0001            1.0000        55.60
HnswRaBitQ-m32-ex8 (self)                              6_263.10    18_023.89    24_287.00       0.9931          1.0022            1.0000        55.60
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
Exhaustive (query)                                        33.66       724.13       757.79       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.66     2_456.43     2_490.09       1.0000          1.0000            1.0000        48.83
QG-d32-l64-ef15 (query)                                3_121.07        84.17     3_205.24       0.9731          1.0008            1.0000       116.16
QG-d32-l64-ef30 (query)                                3_121.07       143.88     3_264.95       0.9986          1.0000            1.0000       116.16
QG-d32-l64-ef60 (query)                                3_121.07       285.15     3_406.22       0.9999          1.0000            1.0000       116.16
QG-d32-l64-ef120 (query)                               3_121.07       506.88     3_627.95       1.0000          1.0000            1.0000       116.16
QG-d32-l64 (self)                                      3_121.07       803.75     3_924.82       1.0000          1.0000            1.0000       116.16
QG-d32-l128-ef15 (query)                               3_569.97        83.06     3_653.03       0.9732          1.0008            1.0000       116.16
QG-d32-l128-ef30 (query)                               3_569.97       143.47     3_713.44       0.9986          1.0000            1.0000       116.16
QG-d32-l128-ef60 (query)                               3_569.97       265.91     3_835.89       1.0000          1.0000            1.0000       116.16
QG-d32-l128-ef120 (query)                              3_569.97       507.37     4_077.34       1.0000          1.0000            1.0000       116.16
QG-d32-l128 (self)                                     3_569.97       805.33     4_375.31       1.0000          1.0000            1.0000       116.16
QG-d64-l64-ef15 (query)                                6_288.27       129.13     6_417.40       0.9941          1.0001            1.0000       183.30
QG-d64-l64-ef30 (query)                                6_288.27       236.05     6_524.32       1.0000          1.0000            1.0000       183.30
QG-d64-l64-ef60 (query)                                6_288.27       446.44     6_734.70       1.0000          1.0000            1.0000       183.30
QG-d64-l64-ef120 (query)                               6_288.27       858.19     7_146.45       1.0000          1.0000            1.0000       183.30
QG-d64-l64 (self)                                      6_288.27     1_376.59     7_664.86       1.0000          1.0000            1.0000       183.30
QG-d64-l128-ef15 (query)                               6_938.35       126.99     7_065.34       0.9940          1.0001            1.0000       183.30
QG-d64-l128-ef30 (query)                               6_938.35       231.82     7_170.17       1.0000          1.0000            1.0000       183.30
QG-d64-l128-ef60 (query)                               6_938.35       446.04     7_384.38       1.0000          1.0000            1.0000       183.30
QG-d64-l128-ef120 (query)                              6_938.35       865.08     7_803.43       1.0000          1.0000            1.0000       183.30
QG-d64-l128 (self)                                     6_938.35     1_366.86     8_305.21       1.0000          1.0000            1.0000       183.30
HnswRaBitQ-m16-ex1-ef15 (query)                        1_362.10        82.98     1_445.08       0.8537          1.0073            1.0059        10.85
HnswRaBitQ-m16-ex1-ef30 (query)                        1_362.10        99.77     1_461.87       0.8667          1.0057            1.0049        10.85
HnswRaBitQ-m16-ex1-ef60 (query)                        1_362.10       126.77     1_488.87       0.8695          1.0053            1.0046        10.85
HnswRaBitQ-m16-ex1-ef120 (query)                       1_362.10       166.85     1_528.95       0.8698          1.0052            1.0046        10.85
HnswRaBitQ-m16-ex1 (self)                              1_362.10     5_131.90     6_494.00       0.8305          1.0109            1.0098        10.85
HnswRaBitQ-m16-ex3-ef15 (query)                        1_445.45        95.34     1_540.79       0.9275          1.0029            1.0009        13.90
HnswRaBitQ-m16-ex3-ef30 (query)                        1_445.45       132.72     1_578.17       0.9505          1.0010            1.0003        13.90
HnswRaBitQ-m16-ex3-ef60 (query)                        1_445.45       134.93     1_580.38       0.9569          1.0006            1.0001        13.90
HnswRaBitQ-m16-ex3-ef120 (query)                       1_445.45       179.38     1_624.82       0.9578          1.0005            1.0001        13.90
HnswRaBitQ-m16-ex3 (self)                              1_445.45     4_926.69     6_372.14       0.9470          1.0009            1.0005        13.90
HnswRaBitQ-m16-ex5-ef15 (query)                        1_562.65        91.26     1_653.91       0.9471          1.0025            1.0001        16.95
HnswRaBitQ-m16-ex5-ef30 (query)                        1_562.65       118.39     1_681.03       0.9771          1.0006            1.0000        16.95
HnswRaBitQ-m16-ex5-ef60 (query)                        1_562.65       141.29     1_703.93       0.9855          1.0002            1.0000        16.95
HnswRaBitQ-m16-ex5-ef120 (query)                       1_562.65       183.88     1_746.53       0.9868          1.0001            1.0000        16.95
HnswRaBitQ-m16-ex5 (self)                              1_562.65     5_512.39     7_075.04       0.9834          1.0002            1.0000        16.95
HnswRaBitQ-m16-ex8-ef15 (query)                        2_279.80        88.10     2_367.90       0.9524          1.0024            1.0000        21.53
HnswRaBitQ-m16-ex8-ef30 (query)                        2_279.80       110.01     2_389.81       0.9854          1.0006            1.0000        21.53
HnswRaBitQ-m16-ex8-ef60 (query)                        2_279.80       139.99     2_419.79       0.9954          1.0001            1.0000        21.53
HnswRaBitQ-m16-ex8-ef120 (query)                       2_279.80       182.23     2_462.03       0.9971          1.0001            1.0000        21.53
HnswRaBitQ-m16-ex8 (self)                              2_279.80     5_773.14     8_052.94       0.9965          1.0001            1.0000        21.53
HnswRaBitQ-m32-ex1-ef15 (query)                        1_392.60        89.80     1_482.40       0.8624          1.0062            1.0052        16.95
HnswRaBitQ-m32-ex1-ef30 (query)                        1_392.60       110.74     1_503.34       0.8685          1.0054            1.0047        16.95
HnswRaBitQ-m32-ex1-ef60 (query)                        1_392.60       140.83     1_533.43       0.8697          1.0052            1.0046        16.95
HnswRaBitQ-m32-ex1-ef120 (query)                       1_392.60       187.88     1_580.48       0.8698          1.0052            1.0046        16.95
HnswRaBitQ-m32-ex1 (self)                              1_392.60     6_239.03     7_631.63       0.8306          1.0108            1.0098        16.95
HnswRaBitQ-m32-ex3-ef15 (query)                        1_467.59        99.99     1_567.58       0.9430          1.0015            1.0004        20.00
HnswRaBitQ-m32-ex3-ef30 (query)                        1_467.59       118.35     1_585.94       0.9554          1.0006            1.0002        20.00
HnswRaBitQ-m32-ex3-ef60 (query)                        1_467.59       151.81     1_619.40       0.9576          1.0005            1.0001        20.00
HnswRaBitQ-m32-ex3-ef120 (query)                       1_467.59       194.84     1_662.44       0.9579          1.0005            1.0001        20.00
HnswRaBitQ-m32-ex3 (self)                              1_467.59     6_541.48     8_009.08       0.9475          1.0009            1.0005        20.00
HnswRaBitQ-m32-ex5-ef15 (query)                        1_572.39       100.95     1_673.35       0.9665          1.0012            1.0000        23.06
HnswRaBitQ-m32-ex5-ef30 (query)                        1_572.39       130.05     1_702.45       0.9831          1.0003            1.0000        23.06
HnswRaBitQ-m32-ex5-ef60 (query)                        1_572.39       155.68     1_728.08       0.9867          1.0001            1.0000        23.06
HnswRaBitQ-m32-ex5-ef120 (query)                       1_572.39       199.27     1_771.66       0.9871          1.0001            1.0000        23.06
HnswRaBitQ-m32-ex5 (self)                              1_572.39     6_711.21     8_283.61       0.9843          1.0001            1.0000        23.06
HnswRaBitQ-m32-ex8-ef15 (query)                        2_311.47       100.54     2_412.01       0.9744          1.0012            1.0000        27.63
HnswRaBitQ-m32-ex8-ef30 (query)                        2_311.47       125.20     2_436.67       0.9928          1.0002            1.0000        27.63
HnswRaBitQ-m32-ex8-ef60 (query)                        2_311.47       158.55     2_470.02       0.9969          1.0001            1.0000        27.63
HnswRaBitQ-m32-ex8-ef120 (query)                       2_311.47       202.49     2_513.95       0.9974          1.0000            1.0000        27.63
HnswRaBitQ-m32-ex8 (self)                              2_311.47     6_453.47     8_764.94       0.9975          1.0000            1.0000        27.63
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
Exhaustive (query)                                        68.36     1_293.57     1_361.93       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.36     4_468.49     4_536.85       1.0000          1.0000            1.0000        97.66
QG-d32-l64-ef15 (query)                                5_966.07       183.98     6_150.06       0.9599          1.0035            1.0000       213.81
QG-d32-l64-ef30 (query)                                5_966.07       247.13     6_213.21       0.9919          1.0014            1.0000       213.81
QG-d32-l64-ef60 (query)                                5_966.07       424.15     6_390.22       0.9979          1.0007            1.0000       213.81
QG-d32-l64-ef120 (query)                               5_966.07       767.25     6_733.33       0.9992          1.0004            1.0000       213.81
QG-d32-l64 (self)                                      5_966.07     1_271.86     7_237.94       0.9981          1.0007            1.0000       213.81
QG-d32-l128-ef15 (query)                               6_707.36       146.19     6_853.55       0.9619          1.0030            1.0000       213.81
QG-d32-l128-ef30 (query)                               6_707.36       241.30     6_948.66       0.9935          1.0012            1.0000       213.81
QG-d32-l128-ef60 (query)                               6_707.36       419.45     7_126.81       0.9987          1.0006            1.0000       213.81
QG-d32-l128-ef120 (query)                              6_707.36       749.74     7_457.09       0.9995          1.0003            1.0000       213.81
QG-d32-l128 (self)                                     6_707.36     1_294.45     8_001.81       0.9987          1.0006            1.0000       213.81
QG-d64-l64-ef15 (query)                               17_701.05       226.03    17_927.08       0.9930          1.0002            1.0000       329.78
QG-d64-l64-ef30 (query)                               17_701.05       398.65    18_099.71       0.9997          1.0000            1.0000       329.78
QG-d64-l64-ef60 (query)                               17_701.05       712.35    18_413.41       1.0000          1.0000            1.0000       329.78
QG-d64-l64-ef120 (query)                              17_701.05     1_295.22    18_996.27       1.0000          1.0000            1.0000       329.78
QG-d64-l64 (self)                                     17_701.05     2_216.62    19_917.67       1.0000          1.0000            1.0000       329.78
QG-d64-l128-ef15 (query)                              18_313.76       225.81    18_539.57       0.9932          1.0001            1.0000       329.78
QG-d64-l128-ef30 (query)                              18_313.76       399.52    18_713.28       0.9998          1.0000            1.0000       329.78
QG-d64-l128-ef60 (query)                              18_313.76       715.21    19_028.97       1.0000          1.0000            1.0000       329.78
QG-d64-l128-ef120 (query)                             18_313.76     1_281.84    19_595.60       1.0000          1.0000            1.0000       329.78
QG-d64-l128 (self)                                    18_313.76     2_235.35    20_549.12       1.0000          1.0000            1.0000       329.78
HnswRaBitQ-m16-ex1-ef15 (query)                        2_526.58       150.38     2_676.95       0.8450          1.0067            1.0044        14.15
HnswRaBitQ-m16-ex1-ef30 (query)                        2_526.58       179.14     2_705.72       0.8663          1.0043            1.0033        14.15
HnswRaBitQ-m16-ex1-ef60 (query)                        2_526.58       221.02     2_747.59       0.8734          1.0035            1.0029        14.15
HnswRaBitQ-m16-ex1-ef120 (query)                       2_526.58       281.02     2_807.59       0.8747          1.0034            1.0029        14.15
HnswRaBitQ-m16-ex1 (self)                              2_526.58    11_053.27    13_579.85       0.8357          1.0069            1.0060        14.15
HnswRaBitQ-m16-ex3-ef15 (query)                        2_676.59       163.14     2_839.73       0.9085          1.0041            1.0011        20.25
HnswRaBitQ-m16-ex3-ef30 (query)                        2_676.59       193.24     2_869.83       0.9443          1.0013            1.0003        20.25
HnswRaBitQ-m16-ex3-ef60 (query)                        2_676.59       246.75     2_923.34       0.9561          1.0006            1.0001        20.25
HnswRaBitQ-m16-ex3-ef120 (query)                       2_676.59       300.33     2_976.92       0.9585          1.0005            1.0001        20.25
HnswRaBitQ-m16-ex3 (self)                              2_676.59    12_182.85    14_859.44       0.9467          1.0009            1.0003        20.25
HnswRaBitQ-m16-ex5-ef15 (query)                        2_878.73       173.97     3_052.70       0.9244          1.0033            1.0006        26.36
HnswRaBitQ-m16-ex5-ef30 (query)                        2_878.73       205.69     3_084.42       0.9665          1.0011            1.0000        26.36
HnswRaBitQ-m16-ex5-ef60 (query)                        2_878.73       253.69     3_132.42       0.9827          1.0003            1.0000        26.36
HnswRaBitQ-m16-ex5-ef120 (query)                       2_878.73       313.93     3_192.66       0.9860          1.0002            1.0000        26.36
HnswRaBitQ-m16-ex5 (self)                              2_878.73    12_788.91    15_667.64       0.9813          1.0004            1.0000        26.36
HnswRaBitQ-m16-ex8-ef15 (query)                        4_395.26       172.78     4_568.04       0.9289          1.0032            1.0006        35.51
HnswRaBitQ-m16-ex8-ef30 (query)                        4_395.26       206.91     4_602.17       0.9740          1.0010            1.0000        35.51
HnswRaBitQ-m16-ex8-ef60 (query)                        4_395.26       255.86     4_651.12       0.9923          1.0003            1.0000        35.51
HnswRaBitQ-m16-ex8-ef120 (query)                       4_395.26       312.90     4_708.16       0.9962          1.0002            1.0000        35.51
HnswRaBitQ-m16-ex8 (self)                              4_395.26    12_373.07    16_768.33       0.9936          1.0004            1.0000        35.51
HnswRaBitQ-m32-ex1-ef15 (query)                        2_562.39       179.01     2_741.41       0.8621          1.0045            1.0034        20.25
HnswRaBitQ-m32-ex1-ef30 (query)                        2_562.39       210.82     2_773.21       0.8725          1.0035            1.0030        20.25
HnswRaBitQ-m32-ex1-ef60 (query)                        2_562.39       308.97     2_871.37       0.8748          1.0033            1.0028        20.25
HnswRaBitQ-m32-ex1-ef120 (query)                       2_562.39       334.10     2_896.50       0.8750          1.0032            1.0028        20.25
HnswRaBitQ-m32-ex1 (self)                              2_562.39    13_997.47    16_559.87       0.8365          1.0066            1.0060        20.25
HnswRaBitQ-m32-ex3-ef15 (query)                        2_718.41       188.40     2_906.81       0.9367          1.0016            1.0004        26.36
HnswRaBitQ-m32-ex3-ef30 (query)                        2_718.41       224.27     2_942.68       0.9543          1.0006            1.0001        26.36
HnswRaBitQ-m32-ex3-ef60 (query)                        2_718.41       278.92     2_997.32       0.9583          1.0004            1.0001        26.36
HnswRaBitQ-m32-ex3-ef120 (query)                       2_718.41       331.91     3_050.32       0.9590          1.0003            1.0001        26.36
HnswRaBitQ-m32-ex3 (self)                              2_718.41    14_392.90    17_111.30       0.9484          1.0006            1.0003        26.36
HnswRaBitQ-m32-ex5-ef15 (query)                        2_937.25       200.72     3_137.97       0.9573          1.0014            1.0000        32.46
HnswRaBitQ-m32-ex5-ef30 (query)                        2_937.25       238.83     3_176.08       0.9799          1.0003            1.0000        32.46
HnswRaBitQ-m32-ex5-ef60 (query)                        2_937.25       290.27     3_227.53       0.9857          1.0001            1.0000        32.46
HnswRaBitQ-m32-ex5-ef120 (query)                       2_937.25       346.66     3_283.92       0.9866          1.0001            1.0000        32.46
HnswRaBitQ-m32-ex5 (self)                              2_937.25    14_963.48    17_900.73       0.9838          1.0001            1.0000        32.46
HnswRaBitQ-m32-ex8-ef15 (query)                        4_581.97       204.52     4_786.49       0.9641          1.0014            1.0000        41.62
HnswRaBitQ-m32-ex8-ef30 (query)                        4_581.97       250.65     4_832.62       0.9896          1.0003            1.0000        41.62
HnswRaBitQ-m32-ex8-ef60 (query)                        4_581.97       283.22     4_865.19       0.9961          1.0001            1.0000        41.62
HnswRaBitQ-m32-ex8-ef120 (query)                       4_581.97       344.87     4_926.84       0.9970          1.0000            1.0000        41.62
HnswRaBitQ-m32-ex8 (self)                              4_581.97    14_901.97    19_483.93       0.9969          1.0001            1.0000        41.62
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
Exhaustive (query)                                       101.76     1_937.05     2_038.81       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.76     6_500.97     6_602.73       1.0000          1.0000            1.0000       146.48
QG-d32-l64-ef15 (query)                                8_095.40       255.85     8_351.25       0.9517          1.0067            1.0000       311.47
QG-d32-l64-ef30 (query)                                8_095.40       413.54     8_508.94       0.9812          1.0049            1.0000       311.47
QG-d32-l64-ef60 (query)                                8_095.40       672.29     8_767.69       0.9906          1.0042            1.0000       311.47
QG-d32-l64-ef120 (query)                               8_095.40     1_094.93     9_190.33       0.9937          1.0037            1.0000       311.47
QG-d32-l64 (self)                                      8_095.40     2_143.14    10_238.55       0.9909          1.0044            1.0000       311.47
QG-d32-l128-ef15 (query)                               9_015.88       273.12     9_289.00       0.9553          1.0063            1.0000       311.47
QG-d32-l128-ef30 (query)                               9_015.88       413.06     9_428.94       0.9843          1.0045            1.0000       311.47
QG-d32-l128-ef60 (query)                               9_015.88       691.11     9_706.99       0.9927          1.0036            1.0000       311.47
QG-d32-l128-ef120 (query)                              9_015.88     1_096.64    10_112.51       0.9953          1.0032            1.0000       311.47
QG-d32-l128 (self)                                     9_015.88     2_142.44    11_158.32       0.9930          1.0038            1.0000       311.47
QG-d64-l64-ef15 (query)                               30_927.38       505.98    31_433.36       0.9930          1.0009            1.0000       476.27
QG-d64-l64-ef30 (query)                               30_927.38       823.96    31_751.34       0.9985          1.0004            1.0000       476.27
QG-d64-l64-ef60 (query)                               30_927.38     1_318.80    32_246.18       0.9996          1.0002            1.0000       476.27
QG-d64-l64-ef120 (query)                              30_927.38     2_123.43    33_050.81       0.9998          1.0001            1.0000       476.27
QG-d64-l64 (self)                                     30_927.38     4_254.11    35_181.49       0.9996          1.0002            1.0000       476.27
QG-d64-l128-ef15 (query)                              30_987.87       531.03    31_518.89       0.9930          1.0009            1.0000       476.27
QG-d64-l128-ef30 (query)                              30_987.87       840.28    31_828.15       0.9986          1.0004            1.0000       476.27
QG-d64-l128-ef60 (query)                              30_987.87     1_353.57    32_341.44       0.9997          1.0002            1.0000       476.27
QG-d64-l128-ef120 (query)                             30_987.87     2_089.99    33_077.86       0.9999          1.0001            1.0000       476.27
QG-d64-l128 (self)                                    30_987.87     4_292.89    35_280.76       0.9997          1.0002            1.0000       476.27
HnswRaBitQ-m16-ex1-ef15 (query)                        3_490.71       219.06     3_709.76       0.7232        491.5698            1.0046        17.45
HnswRaBitQ-m16-ex1-ef30 (query)                        3_490.71       264.78     3_755.49       0.7512        487.4353            1.0033        17.45
HnswRaBitQ-m16-ex1-ef60 (query)                        3_490.71       361.34     3_852.05       0.8253        182.5190            1.0026        17.45
HnswRaBitQ-m16-ex1-ef120 (query)                       3_490.71       454.35     3_945.06       0.8642          1.0030            1.0024        17.45
HnswRaBitQ-m16-ex1 (self)                              3_490.71    17_015.49    20_506.20       0.7883        180.6363            1.0053        17.45
HnswRaBitQ-m16-ex3-ef15 (query)                        3_739.49       239.87     3_979.36       0.8869          1.0063            1.0013        26.61
HnswRaBitQ-m16-ex3-ef30 (query)                        3_739.49       286.99     4_026.48       0.9312          1.0023            1.0004        26.61
HnswRaBitQ-m16-ex3-ef60 (query)                        3_739.49       348.97     4_088.46       0.9493          1.0008            1.0001        26.61
HnswRaBitQ-m16-ex3-ef120 (query)                       3_739.49       426.05     4_165.54       0.9539          1.0005            1.0001        26.61
HnswRaBitQ-m16-ex3 (self)                              3_739.49    16_053.86    19_793.35       0.9382          1.0011            1.0003        26.61
HnswRaBitQ-m16-ex5-ef15 (query)                        4_085.24       258.52     4_343.76       0.9029          1.0055            1.0010        35.76
HnswRaBitQ-m16-ex5-ef30 (query)                        4_085.24       305.87     4_391.11       0.9534          1.0018            1.0000        35.76
HnswRaBitQ-m16-ex5-ef60 (query)                        4_085.24       370.80     4_456.04       0.9773          1.0006            1.0000        35.76
HnswRaBitQ-m16-ex5-ef120 (query)                       4_085.24       448.24     4_533.48       0.9836          1.0003            1.0000        35.76
HnswRaBitQ-m16-ex5 (self)                              4_085.24    16_045.70    20_130.93       0.9756          1.0006            1.0000        35.76
HnswRaBitQ-m16-ex8-ef15 (query)                        6_373.45       241.67     6_615.12       0.9054          1.0057            1.0010        49.50
HnswRaBitQ-m16-ex8-ef30 (query)                        6_373.45       286.54     6_659.99       0.9602          1.0020            1.0000        49.50
HnswRaBitQ-m16-ex8-ef60 (query)                        6_373.45       351.28     6_724.73       0.9871          1.0007            1.0000        49.50
HnswRaBitQ-m16-ex8-ef120 (query)                       6_373.45       431.81     6_805.26       0.9946          1.0004            1.0000        49.50
HnswRaBitQ-m16-ex8 (self)                              6_373.45    15_605.31    21_978.76       0.9893          1.0006            1.0000        49.50
HnswRaBitQ-m32-ex1-ef15 (query)                        3_533.76       255.06     3_788.82       0.8477          1.0056            1.0030        23.56
HnswRaBitQ-m32-ex1-ef30 (query)                        3_533.76       306.84     3_840.60       0.8609          1.0033            1.0026        23.56
HnswRaBitQ-m32-ex1-ef60 (query)                        3_533.76       370.86     3_904.62       0.8642          1.0028            1.0024        23.56
HnswRaBitQ-m32-ex1-ef120 (query)                       3_533.76       445.77     3_979.53       0.8648          1.0028            1.0024        23.56
HnswRaBitQ-m32-ex1 (self)                              3_533.76    19_123.31    22_657.06       0.8191          1.0056            1.0051        23.56
HnswRaBitQ-m32-ex3-ef15 (query)                        3_802.33       293.68     4_096.01       0.8707        203.0884            1.0005        32.71
HnswRaBitQ-m32-ex3-ef30 (query)                        3_802.33       348.62     4_150.95       0.9087        175.1671            1.0002        32.71
HnswRaBitQ-m32-ex3-ef60 (query)                        3_802.33       413.98     4_216.31       0.9195        174.2556            1.0001        32.71
HnswRaBitQ-m32-ex3-ef120 (query)                       3_802.33       505.63     4_307.95       0.9549          1.0003            1.0001        32.71
HnswRaBitQ-m32-ex3 (self)                              3_802.33    19_837.69    23_640.02       0.9417          1.0006            1.0003        32.71
HnswRaBitQ-m32-ex5-ef15 (query)                        4_074.97       306.07     4_381.04       0.9438          1.2528            1.0000        41.87
HnswRaBitQ-m32-ex5-ef30 (query)                        4_074.97       365.47     4_440.45       0.9744          1.0065            1.0000        41.87
HnswRaBitQ-m32-ex5-ef60 (query)                        4_074.97       426.60     4_501.57       0.9833          1.0002            1.0000        41.87
HnswRaBitQ-m32-ex5-ef120 (query)                       4_074.97       499.46     4_574.43       0.9848          1.0001            1.0000        41.87
HnswRaBitQ-m32-ex5 (self)                              4_074.97    21_861.25    25_936.22       0.9806          1.0002            1.0000        41.87
HnswRaBitQ-m32-ex8-ef15 (query)                        6_372.84       284.63     6_657.47       0.9517          1.0044            1.0000        55.60
HnswRaBitQ-m32-ex8-ef30 (query)                        6_372.84       343.80     6_716.65       0.9838          1.0008            1.0000        55.60
HnswRaBitQ-m32-ex8-ef60 (query)                        6_372.84       411.84     6_784.68       0.9942          1.0002            1.0000        55.60
HnswRaBitQ-m32-ex8-ef120 (query)                       6_372.84       487.08     6_859.92       0.9962          1.0001            1.0000        55.60
HnswRaBitQ-m32-ex8 (self)                              6_372.84    19_405.00    25_777.85       0.9952          1.0002            1.0000        55.60
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
Exhaustive (query)                                        33.24       756.51       789.75       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.24     2_507.99     2_541.23       1.0000          1.0000            1.0000        48.83
QG-d32-l64-ef15 (query)                                2_902.55        62.06     2_964.61       0.9964          1.0001            1.0000       116.16
QG-d32-l64-ef30 (query)                                2_902.55        95.35     2_997.90       1.0000          1.0000            1.0000       116.16
QG-d32-l64-ef60 (query)                                2_902.55       167.59     3_070.14       1.0000          1.0000            1.0000       116.16
QG-d32-l64-ef120 (query)                               2_902.55       319.51     3_222.06       1.0000          1.0000            1.0000       116.16
QG-d32-l64 (self)                                      2_902.55       527.54     3_430.10       1.0000          1.0000            1.0000       116.16
QG-d32-l128-ef15 (query)                               4_160.33        59.49     4_219.82       0.9958          1.0001            1.0000       116.16
QG-d32-l128-ef30 (query)                               4_160.33        93.77     4_254.10       1.0000          1.0000            1.0000       116.16
QG-d32-l128-ef60 (query)                               4_160.33       166.76     4_327.09       1.0000          1.0000            1.0000       116.16
QG-d32-l128-ef120 (query)                              4_160.33       317.01     4_477.34       1.0000          1.0000            1.0000       116.16
QG-d32-l128 (self)                                     4_160.33       517.42     4_677.75       1.0000          1.0000            1.0000       116.16
QG-d64-l64-ef15 (query)                                3_666.32        66.96     3_733.28       0.9980          1.0000            1.0000       183.30
QG-d64-l64-ef30 (query)                                3_666.32       110.65     3_776.98       1.0000          1.0000            1.0000       183.30
QG-d64-l64-ef60 (query)                                3_666.32       197.10     3_863.42       1.0000          1.0000            1.0000       183.30
QG-d64-l64-ef120 (query)                               3_666.32       390.10     4_056.43       1.0000          1.0000            1.0000       183.30
QG-d64-l64 (self)                                      3_666.32       634.46     4_300.79       1.0000          1.0000            1.0000       183.30
QG-d64-l128-ef15 (query)                               5_312.74        69.32     5_382.06       0.9981          1.0000            1.0000       183.30
QG-d64-l128-ef30 (query)                               5_312.74       111.12     5_423.86       1.0000          1.0000            1.0000       183.30
QG-d64-l128-ef60 (query)                               5_312.74       202.26     5_515.00       1.0000          1.0000            1.0000       183.30
QG-d64-l128-ef120 (query)                              5_312.74       399.03     5_711.77       1.0000          1.0000            1.0000       183.30
QG-d64-l128 (self)                                     5_312.74       639.39     5_952.12       1.0000          1.0000            1.0000       183.30
HnswRaBitQ-m16-ex1-ef15 (query)                        1_601.02        62.88     1_663.90       0.9357          1.0360            1.0033        10.85
HnswRaBitQ-m16-ex1-ef30 (query)                        1_601.02        76.61     1_677.62       0.9374          1.0270            1.0031        10.85
HnswRaBitQ-m16-ex1-ef60 (query)                        1_601.02        94.74     1_695.76       0.9384          1.0186            1.0031        10.85
HnswRaBitQ-m16-ex1-ef120 (query)                       1_601.02       125.18     1_726.19       0.9394          1.0111            1.0031        10.85
HnswRaBitQ-m16-ex1 (self)                              1_601.02     3_361.64     4_962.66       0.9112          1.0240            1.0088        10.85
HnswRaBitQ-m16-ex3-ef15 (query)                        1_666.83        66.78     1_733.61       0.9770          1.0105            1.0000        13.90
HnswRaBitQ-m16-ex3-ef30 (query)                        1_666.83        78.41     1_745.25       0.9794          1.0076            1.0000        13.90
HnswRaBitQ-m16-ex3-ef60 (query)                        1_666.83        97.78     1_764.62       0.9799          1.0047            1.0000        13.90
HnswRaBitQ-m16-ex3-ef120 (query)                       1_666.83       134.12     1_800.95       0.9804          1.0025            1.0000        13.90
HnswRaBitQ-m16-ex3 (self)                              1_666.83     3_094.42     4_761.25       0.9729          1.0092            1.0000        13.90
HnswRaBitQ-m16-ex5-ef15 (query)                        1_781.39        72.60     1_853.99       0.9877          1.0325            1.0000        16.95
HnswRaBitQ-m16-ex5-ef30 (query)                        1_781.39        86.74     1_868.13       0.9913          1.0224            1.0000        16.95
HnswRaBitQ-m16-ex5-ef60 (query)                        1_781.39       105.74     1_887.12       0.9927          1.0142            1.0000        16.95
HnswRaBitQ-m16-ex5-ef120 (query)                       1_781.39       141.40     1_922.79       0.9936          1.0068            1.0000        16.95
HnswRaBitQ-m16-ex5 (self)                              1_781.39     3_262.61     5_044.00       0.9911          1.0099            1.0000        16.95
HnswRaBitQ-m16-ex8-ef15 (query)                        2_547.79        71.98     2_619.77       0.9934          1.0271            1.0000        21.53
HnswRaBitQ-m16-ex8-ef30 (query)                        2_547.79        84.56     2_632.35       0.9967          1.0186            1.0000        21.53
HnswRaBitQ-m16-ex8-ef60 (query)                        2_547.79       103.85     2_651.63       0.9978          1.0108            1.0000        21.53
HnswRaBitQ-m16-ex8-ef120 (query)                       2_547.79       140.09     2_687.87       0.9988          1.0020            1.0000        21.53
HnswRaBitQ-m16-ex8 (self)                              2_547.79     3_279.87     5_827.65       0.9975          1.0112            1.0000        21.53
HnswRaBitQ-m32-ex1-ef15 (query)                        1_658.18        71.56     1_729.73       0.9395          1.0099            1.0032        16.95
HnswRaBitQ-m32-ex1-ef30 (query)                        1_658.18        84.68     1_742.85       0.9402          1.0082            1.0031        16.95
HnswRaBitQ-m32-ex1-ef60 (query)                        1_658.18       105.70     1_763.88       0.9404          1.0075            1.0031        16.95
HnswRaBitQ-m32-ex1-ef120 (query)                       1_658.18       145.28     1_803.46       0.9406          1.0061            1.0031        16.95
HnswRaBitQ-m32-ex1 (self)                              1_658.18     3_611.49     5_269.66       0.9128          1.0142            1.0088        16.95
HnswRaBitQ-m32-ex3-ef15 (query)                        1_744.50        76.41     1_820.91       0.9782          1.0096            1.0000        20.00
HnswRaBitQ-m32-ex3-ef30 (query)                        1_744.50        90.22     1_834.72       0.9802          1.0041            1.0000        20.00
HnswRaBitQ-m32-ex3-ef60 (query)                        1_744.50       117.39     1_861.90       0.9806          1.0018            1.0000        20.00
HnswRaBitQ-m32-ex3-ef120 (query)                       1_744.50       153.10     1_897.61       0.9808          1.0008            1.0000        20.00
HnswRaBitQ-m32-ex3 (self)                              1_744.50     3_528.34     5_272.85       0.9741          1.0024            1.0000        20.00
HnswRaBitQ-m32-ex5-ef15 (query)                        1_848.50        80.87     1_929.36       0.9928          1.0012            1.0000        23.06
HnswRaBitQ-m32-ex5-ef30 (query)                        1_848.50        95.57     1_944.07       0.9945          1.0005            1.0000        23.06
HnswRaBitQ-m32-ex5-ef60 (query)                        1_848.50       118.69     1_967.19       0.9948          1.0000            1.0000        23.06
HnswRaBitQ-m32-ex5-ef120 (query)                       1_848.50       159.91     2_008.41       0.9948          1.0000            1.0000        23.06
HnswRaBitQ-m32-ex5 (self)                              1_848.50     3_618.97     5_467.47       0.9927          1.0008            1.0000        23.06
HnswRaBitQ-m32-ex8-ef15 (query)                        2_603.33        81.25     2_684.58       0.9967          1.0032            1.0000        27.63
HnswRaBitQ-m32-ex8-ef30 (query)                        2_603.33        98.02     2_701.35       0.9986          1.0022            1.0000        27.63
HnswRaBitQ-m32-ex8-ef60 (query)                        2_603.33       121.60     2_724.93       0.9991          1.0010            1.0000        27.63
HnswRaBitQ-m32-ex8-ef120 (query)                       2_603.33       161.75     2_765.08       0.9992          1.0005            1.0000        27.63
HnswRaBitQ-m32-ex8 (self)                              2_603.33     3_665.49     6_268.82       0.9989          1.0007            1.0000        27.63
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
Exhaustive (query)                                        68.34     1_375.53     1_443.87       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.34     4_614.14     4_682.48       1.0000          1.0000            1.0000        97.66
QG-d32-l64-ef15 (query)                                5_396.78        91.56     5_488.34       0.9975          1.0001            1.0000       213.81
QG-d32-l64-ef30 (query)                                5_396.78       137.58     5_534.36       1.0000          1.0000            1.0000       213.81
QG-d32-l64-ef60 (query)                                5_396.78       215.46     5_612.24       1.0000          1.0000            1.0000       213.81
QG-d32-l64-ef120 (query)                               5_396.78       392.15     5_788.93       1.0000          1.0000            1.0000       213.81
QG-d32-l64 (self)                                      5_396.78       652.55     6_049.33       1.0000          1.0000            1.0000       213.81
QG-d32-l128-ef15 (query)                               7_963.37        88.95     8_052.32       0.9969          1.0001            1.0000       213.81
QG-d32-l128-ef30 (query)                               7_963.37       128.91     8_092.28       1.0000          1.0000            1.0000       213.81
QG-d32-l128-ef60 (query)                               7_963.37       214.78     8_178.15       1.0000          1.0000            1.0000       213.81
QG-d32-l128-ef120 (query)                              7_963.37       388.87     8_352.24       1.0000          1.0000            1.0000       213.81
QG-d32-l128 (self)                                     7_963.37       639.43     8_602.81       1.0000          1.0000            1.0000       213.81
QG-d64-l64-ef15 (query)                                6_842.36        98.21     6_940.58       0.9986          1.0000            1.0000       329.78
QG-d64-l64-ef30 (query)                                6_842.36       142.68     6_985.04       1.0000          1.0000            1.0000       329.78
QG-d64-l64-ef60 (query)                                6_842.36       237.82     7_080.19       1.0000          1.0000            1.0000       329.78
QG-d64-l64-ef120 (query)                               6_842.36       449.95     7_292.31       1.0000          1.0000            1.0000       329.78
QG-d64-l64 (self)                                      6_842.36       726.47     7_568.83       1.0000          1.0000            1.0000       329.78
QG-d64-l128-ef15 (query)                              10_302.29        97.60    10_399.89       0.9983          1.0000            1.0000       329.78
QG-d64-l128-ef30 (query)                              10_302.29       143.95    10_446.23       1.0000          1.0000            1.0000       329.78
QG-d64-l128-ef60 (query)                              10_302.29       242.19    10_544.48       1.0000          1.0000            1.0000       329.78
QG-d64-l128-ef120 (query)                             10_302.29       449.35    10_751.63       1.0000          1.0000            1.0000       329.78
QG-d64-l128 (self)                                    10_302.29       735.46    11_037.74       1.0000          1.0000            1.0000       329.78
HnswRaBitQ-m16-ex1-ef15 (query)                        2_703.93       112.83     2_816.76       0.9514          1.0615            1.0007        14.15
HnswRaBitQ-m16-ex1-ef30 (query)                        2_703.93       127.60     2_831.53       0.9535          1.0503            1.0006        14.15
HnswRaBitQ-m16-ex1-ef60 (query)                        2_703.93       152.22     2_856.15       0.9550          1.0303            1.0006        14.15
HnswRaBitQ-m16-ex1-ef120 (query)                       2_703.93       195.87     2_899.81       0.9563          1.0174            1.0006        14.15
HnswRaBitQ-m16-ex1 (self)                              2_703.93     5_698.71     8_402.64       0.9317          1.0372            1.0040        14.15
HnswRaBitQ-m16-ex3-ef15 (query)                        2_867.34       130.76     2_998.10       0.9809          1.0378            1.0000        20.25
HnswRaBitQ-m16-ex3-ef30 (query)                        2_867.34       138.66     3_006.00       0.9830          1.0324            1.0000        20.25
HnswRaBitQ-m16-ex3-ef60 (query)                        2_867.34       162.77     3_030.11       0.9837          1.0236            1.0000        20.25
HnswRaBitQ-m16-ex3-ef120 (query)                       2_867.34       208.22     3_075.56       0.9841          1.0181            1.0000        20.25
HnswRaBitQ-m16-ex3 (self)                              2_867.34     5_910.58     8_777.92       0.9796          1.0163            1.0000        20.25
HnswRaBitQ-m16-ex5-ef15 (query)                        3_076.01       129.87     3_205.88       0.9926          1.0115            1.0000        26.36
HnswRaBitQ-m16-ex5-ef30 (query)                        3_076.01       150.58     3_226.59       0.9950          1.0087            1.0000        26.36
HnswRaBitQ-m16-ex5-ef60 (query)                        3_076.01       173.94     3_249.95       0.9952          1.0075            1.0000        26.36
HnswRaBitQ-m16-ex5-ef120 (query)                       3_076.01       222.29     3_298.31       0.9954          1.0055            1.0000        26.36
HnswRaBitQ-m16-ex5 (self)                              3_076.01     6_095.25     9_171.26       0.9933          1.0080            1.0000        26.36
HnswRaBitQ-m16-ex8-ef15 (query)                        4_612.00       126.15     4_738.15       0.9874          1.1382            1.0000        35.51
HnswRaBitQ-m16-ex8-ef30 (query)                        4_612.00       142.46     4_754.46       0.9925          1.0810            1.0000        35.51
HnswRaBitQ-m16-ex8-ef60 (query)                        4_612.00       169.25     4_781.25       0.9947          1.0447            1.0000        35.51
HnswRaBitQ-m16-ex8-ef120 (query)                       4_612.00       216.94     4_828.94       0.9964          1.0194            1.0000        35.51
HnswRaBitQ-m16-ex8 (self)                              4_612.00     5_954.31    10_566.31       0.9949          1.0476            1.0000        35.51
HnswRaBitQ-m32-ex1-ef15 (query)                        2_773.28       126.16     2_899.44       0.9570          1.0027            1.0006        20.25
HnswRaBitQ-m32-ex1-ef30 (query)                        2_773.28       147.13     2_920.41       0.9577          1.0026            1.0006        20.25
HnswRaBitQ-m32-ex1-ef60 (query)                        2_773.28       171.35     2_944.62       0.9578          1.0024            1.0006        20.25
HnswRaBitQ-m32-ex1-ef120 (query)                       2_773.28       221.27     2_994.54       0.9579          1.0023            1.0006        20.25
HnswRaBitQ-m32-ex1 (self)                              2_773.28     6_585.36     9_358.64       0.9344          1.0068            1.0039        20.25
HnswRaBitQ-m32-ex3-ef15 (query)                        2_990.58       136.13     3_126.70       0.9847          1.0005            1.0000        26.36
HnswRaBitQ-m32-ex3-ef30 (query)                        2_990.58       154.29     3_144.87       0.9860          1.0002            1.0000        26.36
HnswRaBitQ-m32-ex3-ef60 (query)                        2_990.58       184.58     3_175.15       0.9860          1.0002            1.0000        26.36
HnswRaBitQ-m32-ex3-ef120 (query)                       2_990.58       232.06     3_222.63       0.9860          1.0002            1.0000        26.36
HnswRaBitQ-m32-ex3 (self)                              2_990.58     6_817.12     9_807.69       0.9815          1.0005            1.0000        26.36
HnswRaBitQ-m32-ex5-ef15 (query)                        3_202.31       144.20     3_346.51       0.9947          1.0002            1.0000        32.46
HnswRaBitQ-m32-ex5-ef30 (query)                        3_202.31       163.25     3_365.56       0.9962          1.0000            1.0000        32.46
HnswRaBitQ-m32-ex5-ef60 (query)                        3_202.31       203.43     3_405.75       0.9963          1.0000            1.0000        32.46
HnswRaBitQ-m32-ex5-ef120 (query)                       3_202.31       242.43     3_444.75       0.9963          1.0000            1.0000        32.46
HnswRaBitQ-m32-ex5 (self)                              3_202.31     6_992.85    10_195.16       0.9945          1.0004            1.0000        32.46
HnswRaBitQ-m32-ex8-ef15 (query)                        4_747.97       140.73     4_888.70       0.9978          1.0002            1.0000        41.62
HnswRaBitQ-m32-ex8-ef30 (query)                        4_747.97       157.92     4_905.89       0.9994          1.0000            1.0000        41.62
HnswRaBitQ-m32-ex8-ef60 (query)                        4_747.97       189.42     4_937.39       0.9995          1.0000            1.0000        41.62
HnswRaBitQ-m32-ex8-ef120 (query)                       4_747.97       236.97     4_984.93       0.9995          1.0000            1.0000        41.62
HnswRaBitQ-m32-ex8 (self)                              4_747.97     6_823.10    11_571.07       0.9993          1.0000            1.0000        41.62
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
Exhaustive (query)                                       101.06     1_992.07     2_093.13       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.06     6_605.51     6_706.56       1.0000          1.0000            1.0000       146.48
QG-d32-l64-ef15 (query)                                7_565.12       116.31     7_681.43       0.9973          1.0001            1.0000       311.47
QG-d32-l64-ef30 (query)                                7_565.12       159.92     7_725.04       1.0000          1.0000            1.0000       311.47
QG-d32-l64-ef60 (query)                                7_565.12       254.10     7_819.22       1.0000          1.0000            1.0000       311.47
QG-d32-l64-ef120 (query)                               7_565.12       453.34     8_018.46       1.0000          1.0000            1.0000       311.47
QG-d32-l64 (self)                                      7_565.12       762.03     8_327.15       1.0000          1.0000            1.0000       311.47
QG-d32-l128-ef15 (query)                              11_062.42       115.80    11_178.23       0.9968          1.0001            1.0000       311.47
QG-d32-l128-ef30 (query)                              11_062.42       158.06    11_220.49       1.0000          1.0000            1.0000       311.47
QG-d32-l128-ef60 (query)                              11_062.42       253.16    11_315.58       1.0000          1.0000            1.0000       311.47
QG-d32-l128-ef120 (query)                             11_062.42       449.47    11_511.89       1.0000          1.0000            1.0000       311.47
QG-d32-l128 (self)                                    11_062.42       754.73    11_817.15       1.0000          1.0000            1.0000       311.47
QG-d64-l64-ef15 (query)                                9_632.12       119.71     9_751.83       0.9981          1.0000            1.0000       476.27
QG-d64-l64-ef30 (query)                                9_632.12       168.89     9_801.01       1.0000          1.0000            1.0000       476.27
QG-d64-l64-ef60 (query)                                9_632.12       275.12     9_907.24       1.0000          1.0000            1.0000       476.27
QG-d64-l64-ef120 (query)                               9_632.12       500.11    10_132.23       1.0000          1.0000            1.0000       476.27
QG-d64-l64 (self)                                      9_632.12       832.50    10_464.62       1.0000          1.0000            1.0000       476.27
QG-d64-l128-ef15 (query)                              13_963.57       118.61    14_082.18       0.9981          1.0000            1.0000       476.27
QG-d64-l128-ef30 (query)                              13_963.57       170.24    14_133.81       1.0000          1.0000            1.0000       476.27
QG-d64-l128-ef60 (query)                              13_963.57       276.25    14_239.81       1.0000          1.0000            1.0000       476.27
QG-d64-l128-ef120 (query)                             13_963.57       504.72    14_468.29       1.0000          1.0000            1.0000       476.27
QG-d64-l128 (self)                                    13_963.57       834.26    14_797.83       1.0000          1.0000            1.0000       476.27
HnswRaBitQ-m16-ex1-ef15 (query)                        3_693.73       154.06     3_847.79       0.9499          1.0170            1.0014        17.45
HnswRaBitQ-m16-ex1-ef30 (query)                        3_693.73       171.39     3_865.12       0.9508          1.0163            1.0013        17.45
HnswRaBitQ-m16-ex1-ef60 (query)                        3_693.73       203.10     3_896.83       0.9511          1.0124            1.0013        17.45
HnswRaBitQ-m16-ex1-ef120 (query)                       3_693.73       253.37     3_947.09       0.9512          1.0117            1.0013        17.45
HnswRaBitQ-m16-ex1 (self)                              3_693.73     7_335.89    11_029.62       0.9209          1.0166            1.0064        17.45
HnswRaBitQ-m16-ex3-ef15 (query)                        3_918.18       171.28     4_089.46       0.9767          1.0059            1.0000        26.61
HnswRaBitQ-m16-ex3-ef30 (query)                        3_918.18       184.08     4_102.25       0.9783          1.0026            1.0000        26.61
HnswRaBitQ-m16-ex3-ef60 (query)                        3_918.18       217.17     4_135.34       0.9785          1.0017            1.0000        26.61
HnswRaBitQ-m16-ex3-ef120 (query)                       3_918.18       268.05     4_186.23       0.9785          1.0009            1.0000        26.61
HnswRaBitQ-m16-ex3 (self)                              3_918.18     7_305.70    11_223.88       0.9710          1.0022            1.0000        26.61
HnswRaBitQ-m16-ex5-ef15 (query)                        4_195.24       176.99     4_372.22       0.9910          1.0258            1.0000        35.76
HnswRaBitQ-m16-ex5-ef30 (query)                        4_195.24       193.53     4_388.77       0.9936          1.0181            1.0000        35.76
HnswRaBitQ-m16-ex5-ef60 (query)                        4_195.24       225.07     4_420.31       0.9944          1.0081            1.0000        35.76
HnswRaBitQ-m16-ex5-ef120 (query)                       4_195.24       277.57     4_472.81       0.9948          1.0034            1.0000        35.76
HnswRaBitQ-m16-ex5 (self)                              4_195.24     7_686.92    11_882.15       0.9927          1.0087            1.0000        35.76
HnswRaBitQ-m16-ex8-ef15 (query)                        6_036.78       166.36     6_203.14       0.9943          1.0401            1.0000        49.50
HnswRaBitQ-m16-ex8-ef30 (query)                        6_036.78       186.09     6_222.86       0.9970          1.0334            1.0000        49.50
HnswRaBitQ-m16-ex8-ef60 (query)                        6_036.78       219.12     6_255.90       0.9980          1.0191            1.0000        49.50
HnswRaBitQ-m16-ex8-ef120 (query)                       6_036.78       272.35     6_309.13       0.9986          1.0103            1.0000        49.50
HnswRaBitQ-m16-ex8 (self)                              6_036.78     7_279.99    13_316.77       0.9981          1.0169            1.0000        49.50
HnswRaBitQ-m32-ex1-ef15 (query)                        3_808.20       174.68     3_982.87       0.9514          1.0039            1.0014        23.56
HnswRaBitQ-m32-ex1-ef30 (query)                        3_808.20       193.73     4_001.93       0.9521          1.0038            1.0013        23.56
HnswRaBitQ-m32-ex1-ef60 (query)                        3_808.20       228.62     4_036.82       0.9522          1.0038            1.0013        23.56
HnswRaBitQ-m32-ex1-ef120 (query)                       3_808.20       283.22     4_091.41       0.9522          1.0038            1.0013        23.56
HnswRaBitQ-m32-ex1 (self)                              3_808.20     8_174.96    11_983.15       0.9216          1.0108            1.0063        23.56
HnswRaBitQ-m32-ex3-ef15 (query)                        4_048.91       188.41     4_237.32       0.9772          1.0014            1.0000        32.71
HnswRaBitQ-m32-ex3-ef30 (query)                        4_048.91       208.11     4_257.02       0.9784          1.0012            1.0000        32.71
HnswRaBitQ-m32-ex3-ef60 (query)                        4_048.91       244.47     4_293.37       0.9785          1.0012            1.0000        32.71
HnswRaBitQ-m32-ex3-ef120 (query)                       4_048.91       299.68     4_348.58       0.9785          1.0012            1.0000        32.71
HnswRaBitQ-m32-ex3 (self)                              4_048.91     8_455.59    12_504.50       0.9710          1.0015            1.0000        32.71
HnswRaBitQ-m32-ex5-ef15 (query)                        4_280.49       201.07     4_481.55       0.9934          1.0009            1.0000        41.87
HnswRaBitQ-m32-ex5-ef30 (query)                        4_280.49       219.77     4_500.26       0.9950          1.0007            1.0000        41.87
HnswRaBitQ-m32-ex5-ef60 (query)                        4_280.49       253.68     4_534.17       0.9950          1.0007            1.0000        41.87
HnswRaBitQ-m32-ex5-ef120 (query)                       4_280.49       311.86     4_592.35       0.9951          1.0007            1.0000        41.87
HnswRaBitQ-m32-ex5 (self)                              4_280.49     8_716.83    12_997.32       0.9932          1.0005            1.0000        41.87
HnswRaBitQ-m32-ex8-ef15 (query)                        6_243.63       189.50     6_433.13       0.9973          1.0008            1.0000        55.60
HnswRaBitQ-m32-ex8-ef30 (query)                        6_243.63       209.83     6_453.47       0.9991          1.0007            1.0000        55.60
HnswRaBitQ-m32-ex8-ef60 (query)                        6_243.63       244.33     6_487.97       0.9992          1.0006            1.0000        55.60
HnswRaBitQ-m32-ex8-ef120 (query)                       6_243.63       301.65     6_545.28       0.9992          1.0006            1.0000        55.60
HnswRaBitQ-m32-ex8 (self)                              6_243.63     8_423.40    14_667.03       0.9991          1.0004            1.0000        55.60
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
Exhaustive (query)                                        33.20       700.68       733.87       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.20     2_340.70     2_373.90       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              148.57       363.49       512.06       0.0971          1.7176            1.5960         7.12
ExhaustiveTQ-b2-rf5 (query)                              148.57       443.04       591.61       0.2336          1.2025            1.2205         7.12
ExhaustiveTQ-b2-rf10 (query)                             148.57       582.14       730.72       0.2853          1.1453            1.1620         7.12
ExhaustiveTQ-b2-rf20 (query)                             148.57       961.84     1_110.41       0.3809          1.0970            1.0941         7.12
ExhaustiveTQ-b2 (self)                                   148.57     3_165.46     3_314.04       0.3815          1.0980            1.0956         7.12
ExhaustiveTQ-b4-rf0 (query)                              230.84       582.31       813.15       0.1094          1.5329            1.4998        13.22
ExhaustiveTQ-b4-rf5 (query)                              230.84       673.77       904.61       0.2368          1.1884            1.2090        13.22
ExhaustiveTQ-b4-rf10 (query)                             230.84       807.42     1_038.26       0.2884          1.1372            1.1544        13.22
ExhaustiveTQ-b4-rf20 (query)                             230.84     1_204.94     1_435.77       0.3823          1.0940            1.0971        13.22
ExhaustiveTQ-b4 (self)                                   230.84     3_963.59     4_194.43       0.3841          1.0937            1.0948        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                          355.63       105.08       460.72       0.0971          1.7176            1.5959         7.61
IVF-TQ-b2-nl158-np12-rf0 (query)                         355.63       115.52       471.16       0.0971          1.7176            1.5959         7.61
IVF-TQ-b2-nl158-np17-rf0 (query)                         355.63       125.03       480.66       0.0971          1.7176            1.5959         7.61
IVF-TQ-b2-nl158-np7-rf10 (query)                         355.63       297.53       653.16       0.2853          1.1453            1.1620         7.61
IVF-TQ-b2-nl158-np7-rf20 (query)                         355.63       612.35       967.98       0.3809          1.0970            1.0940         7.61
IVF-TQ-b2-nl158-np12-rf10 (query)                        355.63       310.28       665.91       0.2853          1.1453            1.1620         7.61
IVF-TQ-b2-nl158-np12-rf20 (query)                        355.63       646.38     1_002.01       0.3809          1.0970            1.0941         7.61
IVF-TQ-b2-nl158-np17-rf10 (query)                        355.63       323.85       679.48       0.2853          1.1453            1.1620         7.61
IVF-TQ-b2-nl158-np17-rf20 (query)                        355.63       673.77     1_029.40       0.3809          1.0970            1.0940         7.61
IVF-TQ-b2-nl158 (self)                                   355.63     1_067.67     1_423.31       0.3815          1.0980            1.0957         7.61
IVF-TQ-b2-nl223-np11-rf0 (query)                         377.37       113.96       491.34       0.0971          1.7165            1.5942         7.74
IVF-TQ-b2-nl223-np14-rf0 (query)                         377.37       124.55       501.93       0.0971          1.7176            1.5959         7.74
IVF-TQ-b2-nl223-np21-rf0 (query)                         377.37       134.86       512.24       0.0971          1.7176            1.5959         7.74
IVF-TQ-b2-nl223-np11-rf10 (query)                        377.37       292.93       670.30       0.2855          1.1450            1.1619         7.74
IVF-TQ-b2-nl223-np11-rf20 (query)                        377.37       572.73       950.11       0.3813          1.0967            1.0933         7.74
IVF-TQ-b2-nl223-np14-rf10 (query)                        377.37       298.81       676.18       0.2853          1.1453            1.1620         7.74
IVF-TQ-b2-nl223-np14-rf20 (query)                        377.37       586.49       963.87       0.3809          1.0970            1.0941         7.74
IVF-TQ-b2-nl223-np21-rf10 (query)                        377.37       315.18       692.55       0.2853          1.1453            1.1620         7.74
IVF-TQ-b2-nl223-np21-rf20 (query)                        377.37       645.22     1_022.59       0.3809          1.0970            1.0940         7.74
IVF-TQ-b2-nl223 (self)                                   377.37     1_062.88     1_440.25       0.3815          1.0980            1.0957         7.74
IVF-TQ-b2-nl316-np15-rf0 (query)                         429.13       117.87       547.00       0.0972          1.7112            1.5945         7.91
IVF-TQ-b2-nl316-np17-rf0 (query)                         429.13       123.78       552.91       0.0971          1.7175            1.5959         7.91
IVF-TQ-b2-nl316-np25-rf0 (query)                         429.13       134.95       564.09       0.0971          1.7176            1.5959         7.91
IVF-TQ-b2-nl316-np15-rf10 (query)                        429.13       287.69       716.83       0.2855          1.1451            1.1619         7.91
IVF-TQ-b2-nl316-np15-rf20 (query)                        429.13       550.02       979.16       0.3813          1.0967            1.0936         7.91
IVF-TQ-b2-nl316-np17-rf10 (query)                        429.13       292.22       721.35       0.2853          1.1453            1.1620         7.91
IVF-TQ-b2-nl316-np17-rf20 (query)                        429.13       559.11       988.25       0.3809          1.0970            1.0941         7.91
IVF-TQ-b2-nl316-np25-rf10 (query)                        429.13       308.71       737.84       0.2853          1.1453            1.1620         7.91
IVF-TQ-b2-nl316-np25-rf20 (query)                        429.13       588.84     1_017.98       0.3809          1.0970            1.0941         7.91
IVF-TQ-b2-nl316 (self)                                   429.13     1_067.73     1_496.86       0.3815          1.0980            1.0956         7.91
IVF-TQ-b4-nl158-np7-rf0 (query)                          430.37       146.39       576.75       0.1094          1.5329            1.4998        13.85
IVF-TQ-b4-nl158-np12-rf0 (query)                         430.37       162.64       593.00       0.1094          1.5329            1.4998        13.85
IVF-TQ-b4-nl158-np17-rf0 (query)                         430.37       179.45       609.82       0.1094          1.5329            1.4998        13.85
IVF-TQ-b4-nl158-np7-rf10 (query)                         430.37       349.79       780.16       0.2884          1.1372            1.1544        13.85
IVF-TQ-b4-nl158-np7-rf20 (query)                         430.37       669.42     1_099.78       0.3823          1.0940            1.0971        13.85
IVF-TQ-b4-nl158-np12-rf10 (query)                        430.37       366.55       796.91       0.2884          1.1372            1.1544        13.85
IVF-TQ-b4-nl158-np12-rf20 (query)                        430.37       704.04     1_134.40       0.3823          1.0940            1.0971        13.85
IVF-TQ-b4-nl158-np17-rf10 (query)                        430.37       386.48       816.84       0.2884          1.1372            1.1544        13.85
IVF-TQ-b4-nl158-np17-rf20 (query)                        430.37       738.09     1_168.46       0.3823          1.0940            1.0971        13.85
IVF-TQ-b4-nl158 (self)                                   430.37     1_173.65     1_604.02       0.3841          1.0937            1.0948        13.85
IVF-TQ-b4-nl223-np11-rf0 (query)                         464.84       155.87       620.71       0.1095          1.5315            1.4988        14.05
IVF-TQ-b4-nl223-np14-rf0 (query)                         464.84       169.29       634.12       0.1094          1.5329            1.4998        14.05
IVF-TQ-b4-nl223-np21-rf0 (query)                         464.84       189.60       654.44       0.1094          1.5329            1.4998        14.05
IVF-TQ-b4-nl223-np11-rf10 (query)                        464.84       341.08       805.92       0.2886          1.1370            1.1542        14.05
IVF-TQ-b4-nl223-np11-rf20 (query)                        464.84       618.47     1_083.30       0.3826          1.0939            1.0967        14.05
IVF-TQ-b4-nl223-np14-rf10 (query)                        464.84       355.01       819.85       0.2884          1.1372            1.1544        14.05
IVF-TQ-b4-nl223-np14-rf20 (query)                        464.84       643.43     1_108.26       0.3823          1.0940            1.0971        14.05
IVF-TQ-b4-nl223-np21-rf10 (query)                        464.84       379.09       843.93       0.2884          1.1372            1.1544        14.05
IVF-TQ-b4-nl223-np21-rf20 (query)                        464.84       676.99     1_141.83       0.3823          1.0940            1.0971        14.05
IVF-TQ-b4-nl223 (self)                                   464.84     1_171.01     1_635.84       0.3841          1.0937            1.0948        14.05
IVF-TQ-b4-nl316-np15-rf0 (query)                         508.68       161.15       669.83       0.1094          1.5320            1.4991        14.30
IVF-TQ-b4-nl316-np17-rf0 (query)                         508.68       172.37       681.05       0.1094          1.5329            1.4998        14.30
IVF-TQ-b4-nl316-np25-rf0 (query)                         508.68       191.03       699.71       0.1094          1.5329            1.4998        14.30
IVF-TQ-b4-nl316-np15-rf10 (query)                        508.68       343.02       851.70       0.2885          1.1371            1.1543        14.30
IVF-TQ-b4-nl316-np15-rf20 (query)                        508.68       610.03     1_118.71       0.3825          1.0939            1.0969        14.30
IVF-TQ-b4-nl316-np17-rf10 (query)                        508.68       351.50       860.18       0.2884          1.1372            1.1544        14.30
IVF-TQ-b4-nl316-np17-rf20 (query)                        508.68       636.66     1_145.34       0.3823          1.0940            1.0971        14.30
IVF-TQ-b4-nl316-np25-rf10 (query)                        508.68       374.41       883.09       0.2884          1.1372            1.1544        14.30
IVF-TQ-b4-nl316-np25-rf20 (query)                        508.68       667.05     1_175.73       0.3823          1.0940            1.0971        14.30
IVF-TQ-b4-nl316 (self)                                   508.68     1_181.55     1_690.23       0.3841          1.0937            1.0948        14.30
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
Exhaustive (query)                                        67.93     1_380.26     1_448.19       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         67.93     4_637.45     4_705.38       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              345.59       662.02     1_007.60       0.1208          1.3711            1.3321        13.97
ExhaustiveTQ-b2-rf5 (query)                              345.59       747.30     1_092.88       0.2421          1.1334            1.1574        13.97
ExhaustiveTQ-b2-rf10 (query)                             345.59       886.29     1_231.87       0.2935          1.0981            1.1178        13.97
ExhaustiveTQ-b2-rf20 (query)                             345.59     1_287.64     1_633.22       0.3880          1.0664            1.0469        13.97
ExhaustiveTQ-b2 (self)                                   345.59     4_185.27     4_530.86       0.3879          1.0667            1.0471        13.97
ExhaustiveTQ-b4-rf0 (query)                              461.94     1_152.18     1_614.12       0.1315          1.3171            1.3127        26.18
ExhaustiveTQ-b4-rf5 (query)                              461.94     1_254.64     1_716.58       0.2471          1.1254            1.1483        26.18
ExhaustiveTQ-b4-rf10 (query)                             461.94     1_394.85     1_856.78       0.2970          1.0929            1.0979        26.18
ExhaustiveTQ-b4-rf20 (query)                             461.94     1_777.82     2_239.76       0.3883          1.0643            1.0493        26.18
ExhaustiveTQ-b4 (self)                                   461.94     6_136.14     6_598.08       0.3882          1.0646            1.0495        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                          631.21       192.89       824.10       0.1208          1.3711            1.3321        14.77
IVF-TQ-b2-nl158-np12-rf0 (query)                         631.21       214.25       845.46       0.1208          1.3711            1.3321        14.77
IVF-TQ-b2-nl158-np17-rf0 (query)                         631.21       226.06       857.27       0.1208          1.3711            1.3321        14.77
IVF-TQ-b2-nl158-np7-rf10 (query)                         631.21       419.53     1_050.74       0.2934          1.0981            1.1178        14.77
IVF-TQ-b2-nl158-np7-rf20 (query)                         631.21       773.43     1_404.64       0.3880          1.0664            1.0469        14.77
IVF-TQ-b2-nl158-np12-rf10 (query)                        631.21       431.91     1_063.13       0.2934          1.0981            1.1178        14.77
IVF-TQ-b2-nl158-np12-rf20 (query)                        631.21       789.74     1_420.96       0.3880          1.0664            1.0469        14.77
IVF-TQ-b2-nl158-np17-rf10 (query)                        631.21       452.26     1_083.48       0.2934          1.0981            1.1178        14.77
IVF-TQ-b2-nl158-np17-rf20 (query)                        631.21       820.67     1_451.88       0.3880          1.0664            1.0469        14.77
IVF-TQ-b2-nl158 (self)                                   631.21     1_464.88     2_096.09       0.3879          1.0667            1.0471        14.77
IVF-TQ-b2-nl223-np11-rf0 (query)                         628.40       205.20       833.60       0.1208          1.3696            1.3298        14.98
IVF-TQ-b2-nl223-np14-rf0 (query)                         628.40       219.31       847.71       0.1208          1.3711            1.3321        14.98
IVF-TQ-b2-nl223-np21-rf0 (query)                         628.40       238.37       866.76       0.1208          1.3711            1.3321        14.98
IVF-TQ-b2-nl223-np11-rf10 (query)                        628.40       426.34     1_054.73       0.2938          1.0979            1.1177        14.98
IVF-TQ-b2-nl223-np11-rf20 (query)                        628.40       716.79     1_345.19       0.3887          1.0662            1.0466        14.98
IVF-TQ-b2-nl223-np14-rf10 (query)                        628.40       422.67     1_051.06       0.2934          1.0981            1.1178        14.98
IVF-TQ-b2-nl223-np14-rf20 (query)                        628.40       733.82     1_362.22       0.3880          1.0664            1.0469        14.98
IVF-TQ-b2-nl223-np21-rf10 (query)                        628.40       442.69     1_071.09       0.2934          1.0981            1.1178        14.98
IVF-TQ-b2-nl223-np21-rf20 (query)                        628.40       779.13     1_407.53       0.3880          1.0664            1.0469        14.98
IVF-TQ-b2-nl223 (self)                                   628.40     1_449.01     2_077.40       0.3879          1.0667            1.0471        14.98
IVF-TQ-b2-nl316-np15-rf0 (query)                         711.19       212.58       923.77       0.1208          1.3690            1.3288        15.37
IVF-TQ-b2-nl316-np17-rf0 (query)                         711.19       219.43       930.62       0.1208          1.3707            1.3313        15.37
IVF-TQ-b2-nl316-np25-rf0 (query)                         711.19       240.85       952.04       0.1208          1.3711            1.3321        15.37
IVF-TQ-b2-nl316-np15-rf10 (query)                        711.19       413.04     1_124.23       0.2939          1.0977            1.1176        15.37
IVF-TQ-b2-nl316-np15-rf20 (query)                        711.19       690.56     1_401.74       0.3891          1.0660            1.0465        15.37
IVF-TQ-b2-nl316-np17-rf10 (query)                        711.19       424.31     1_135.50       0.2935          1.0980            1.1177        15.37
IVF-TQ-b2-nl316-np17-rf20 (query)                        711.19       712.91     1_424.09       0.3882          1.0663            1.0468        15.37
IVF-TQ-b2-nl316-np25-rf10 (query)                        711.19       442.47     1_153.66       0.2934          1.0981            1.1178        15.37
IVF-TQ-b2-nl316-np25-rf20 (query)                        711.19       755.12     1_466.31       0.3880          1.0664            1.0469        15.37
IVF-TQ-b2-nl316 (self)                                   711.19     1_472.57     2_183.76       0.3879          1.0667            1.0471        15.37
IVF-TQ-b4-nl158-np7-rf0 (query)                          743.76       270.32     1_014.08       0.1315          1.3171            1.3127        27.27
IVF-TQ-b4-nl158-np12-rf0 (query)                         743.76       303.09     1_046.85       0.1315          1.3171            1.3127        27.27
IVF-TQ-b4-nl158-np17-rf0 (query)                         743.76       329.93     1_073.69       0.1315          1.3171            1.3127        27.27
IVF-TQ-b4-nl158-np7-rf10 (query)                         743.76       517.95     1_261.72       0.2970          1.0929            1.0979        27.27
IVF-TQ-b4-nl158-np7-rf20 (query)                         743.76       870.32     1_614.08       0.3883          1.0643            1.0493        27.27
IVF-TQ-b4-nl158-np12-rf10 (query)                        743.76       539.66     1_283.42       0.2970          1.0929            1.0979        27.27
IVF-TQ-b4-nl158-np12-rf20 (query)                        743.76       900.96     1_644.72       0.3883          1.0643            1.0493        27.27
IVF-TQ-b4-nl158-np17-rf10 (query)                        743.76       568.64     1_312.40       0.2970          1.0929            1.0979        27.27
IVF-TQ-b4-nl158-np17-rf20 (query)                        743.76       941.07     1_684.83       0.3883          1.0643            1.0493        27.27
IVF-TQ-b4-nl158 (self)                                   743.76     1_631.00     2_374.76       0.3882          1.0646            1.0495        27.27
IVF-TQ-b4-nl223-np11-rf0 (query)                         757.80       289.61     1_047.41       0.1316          1.3158            1.3117        27.58
IVF-TQ-b4-nl223-np14-rf0 (query)                         757.80       309.82     1_067.62       0.1315          1.3171            1.3127        27.58
IVF-TQ-b4-nl223-np21-rf0 (query)                         757.80       349.93     1_107.73       0.1315          1.3171            1.3127        27.58
IVF-TQ-b4-nl223-np11-rf10 (query)                        757.80       529.04     1_286.84       0.2974          1.0926            1.0972        27.58
IVF-TQ-b4-nl223-np11-rf20 (query)                        757.80       812.62     1_570.43       0.3889          1.0641            1.0489        27.58
IVF-TQ-b4-nl223-np14-rf10 (query)                        757.80       526.06     1_283.86       0.2970          1.0929            1.0979        27.58
IVF-TQ-b4-nl223-np14-rf20 (query)                        757.80       840.82     1_598.62       0.3883          1.0643            1.0493        27.58
IVF-TQ-b4-nl223-np21-rf10 (query)                        757.80       566.74     1_324.54       0.2970          1.0929            1.0979        27.58
IVF-TQ-b4-nl223-np21-rf20 (query)                        757.80       905.19     1_662.99       0.3883          1.0643            1.0493        27.58
IVF-TQ-b4-nl223 (self)                                   757.80     1_664.12     2_421.92       0.3882          1.0646            1.0495        27.58
IVF-TQ-b4-nl316-np15-rf0 (query)                         838.23       311.49     1_149.72       0.1316          1.3151            1.3108        28.16
IVF-TQ-b4-nl316-np17-rf0 (query)                         838.23       314.58     1_152.81       0.1315          1.3164            1.3121        28.16
IVF-TQ-b4-nl316-np25-rf0 (query)                         838.23       355.91     1_194.14       0.1315          1.3171            1.3127        28.16
IVF-TQ-b4-nl316-np15-rf10 (query)                        838.23       509.27     1_347.50       0.2976          1.0925            1.0968        28.16
IVF-TQ-b4-nl316-np15-rf20 (query)                        838.23       802.76     1_640.99       0.3894          1.0639            1.0484        28.16
IVF-TQ-b4-nl316-np17-rf10 (query)                        838.23       523.42     1_361.64       0.2971          1.0928            1.0977        28.16
IVF-TQ-b4-nl316-np17-rf20 (query)                        838.23       826.45     1_664.68       0.3886          1.0642            1.0491        28.16
IVF-TQ-b4-nl316-np25-rf10 (query)                        838.23       566.58     1_404.80       0.2970          1.0929            1.0979        28.16
IVF-TQ-b4-nl316-np25-rf20 (query)                        838.23       896.53     1_734.75       0.3883          1.0643            1.0493        28.16
IVF-TQ-b4-nl316 (self)                                   838.23     1_694.37     2_532.60       0.3882          1.0646            1.0495        28.16
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
Exhaustive (query)                                       100.00     2_032.53     2_132.53       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.00     6_724.71     6_824.71       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              616.02       987.35     1_603.37       0.1292          1.2710            1.2627        21.33
ExhaustiveTQ-b2-rf5 (query)                              616.02     1_080.83     1_696.85       0.2468          1.1062            1.1333        21.33
ExhaustiveTQ-b2-rf10 (query)                             616.02     1_229.94     1_845.96       0.3000          1.0773            1.0631        21.33
ExhaustiveTQ-b2-rf20 (query)                             616.02     1_660.67     2_276.69       0.3956          1.0509            1.0334        21.33
ExhaustiveTQ-b2 (self)                                   616.02     5_386.97     6_002.99       0.3973          1.0507            1.0331        21.33
ExhaustiveTQ-b4-rf0 (query)                              762.46     1_799.87     2_562.33       0.1340          1.2532            1.2592        39.64
ExhaustiveTQ-b4-rf5 (query)                              762.46     1_906.39     2_668.85       0.2400          1.1136            1.1402        39.64
ExhaustiveTQ-b4-rf10 (query)                             762.46     2_025.40     2_787.86       0.2870          1.0888            1.1143        39.64
ExhaustiveTQ-b4-rf20 (query)                             762.46     2_420.61     3_183.08       0.3752          1.0657            1.0806        39.64
ExhaustiveTQ-b4 (self)                                   762.46     7_991.91     8_754.38       0.3767          1.0654            1.0640        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_027.45       295.88     1_323.33       0.1292          1.2710            1.2627        22.44
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_027.45       318.98     1_346.43       0.1292          1.2710            1.2627        22.44
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_027.45       336.11     1_363.56       0.1292          1.2710            1.2627        22.44
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_027.45       531.55     1_559.00       0.3000          1.0773            1.0631        22.44
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_027.45       899.64     1_927.09       0.3956          1.0509            1.0334        22.44
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_027.45       560.08     1_587.53       0.3000          1.0773            1.0631        22.44
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_027.45       963.69     1_991.14       0.3956          1.0509            1.0334        22.44
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_027.45       581.46     1_608.91       0.3000          1.0773            1.0631        22.44
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_027.45       986.67     2_014.12       0.3956          1.0509            1.0334        22.44
IVF-TQ-b2-nl158 (self)                                 1_027.45     1_835.66     2_863.11       0.3973          1.0507            1.0331        22.44
IVF-TQ-b2-nl223-np11-rf0 (query)                         974.64       310.38     1_285.02       0.1292          1.2710            1.2628        22.85
IVF-TQ-b2-nl223-np14-rf0 (query)                         974.64       325.51     1_300.15       0.1292          1.2710            1.2628        22.85
IVF-TQ-b2-nl223-np21-rf0 (query)                         974.64       356.40     1_331.04       0.1292          1.2710            1.2627        22.85
IVF-TQ-b2-nl223-np11-rf10 (query)                        974.64       539.60     1_514.24       0.3000          1.0773            1.0631        22.85
IVF-TQ-b2-nl223-np11-rf20 (query)                        974.64       850.51     1_825.15       0.3956          1.0510            1.0334        22.85
IVF-TQ-b2-nl223-np14-rf10 (query)                        974.64       554.09     1_528.73       0.3000          1.0773            1.0631        22.85
IVF-TQ-b2-nl223-np14-rf20 (query)                        974.64       876.55     1_851.19       0.3956          1.0509            1.0334        22.85
IVF-TQ-b2-nl223-np21-rf10 (query)                        974.64       591.71     1_566.35       0.3000          1.0773            1.0631        22.85
IVF-TQ-b2-nl223-np21-rf20 (query)                        974.64       929.47     1_904.11       0.3956          1.0509            1.0334        22.85
IVF-TQ-b2-nl223 (self)                                   974.64     1_867.96     2_842.60       0.3973          1.0507            1.0331        22.85
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_099.32       324.31     1_423.62       0.1292          1.2709            1.2627        23.40
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_099.32       333.67     1_432.99       0.1292          1.2710            1.2627        23.40
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_099.32       373.32     1_472.64       0.1292          1.2710            1.2627        23.40
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_099.32       548.55     1_647.87       0.3000          1.0773            1.0631        23.40
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_099.32       864.27     1_963.59       0.3956          1.0509            1.0334        23.40
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_099.32       556.51     1_655.83       0.3000          1.0773            1.0631        23.40
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_099.32       939.24     2_038.56       0.3956          1.0509            1.0334        23.40
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_099.32       592.68     1_692.00       0.3000          1.0773            1.0631        23.40
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_099.32       919.26     2_018.58       0.3956          1.0509            1.0334        23.40
IVF-TQ-b2-nl316 (self)                                 1_099.32     1_911.13     3_010.45       0.3973          1.0507            1.0331        23.40
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_176.93       424.33     1_601.26       0.1340          1.2532            1.2592        41.21
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_176.93       464.30     1_641.23       0.1340          1.2532            1.2592        41.21
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_176.93       492.15     1_669.08       0.1340          1.2532            1.2592        41.21
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_176.93       669.31     1_846.25       0.2870          1.0888            1.1143        41.21
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_176.93     1_051.03     2_227.96       0.3752          1.0657            1.0807        41.21
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_176.93       717.42     1_894.36       0.2870          1.0888            1.1143        41.21
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_176.93     1_105.64     2_282.57       0.3752          1.0657            1.0807        41.21
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_176.93       755.59     1_932.52       0.2870          1.0888            1.1143        41.21
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_176.93     1_146.01     2_322.95       0.3752          1.0657            1.0807        41.21
IVF-TQ-b4-nl158 (self)                                 1_176.93     2_108.34     3_285.27       0.3767          1.0654            1.0641        41.21
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_123.41       447.12     1_570.53       0.1340          1.2532            1.2592        41.85
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_123.41       477.39     1_600.79       0.1340          1.2532            1.2592        41.85
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_123.41       533.38     1_656.78       0.1340          1.2532            1.2592        41.85
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_123.41       697.77     1_821.17       0.2870          1.0888            1.1143        41.85
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_123.41     1_007.36     2_130.76       0.3752          1.0657            1.0807        41.85
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_123.41       716.55     1_839.95       0.2870          1.0888            1.1143        41.85
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_123.41     1_045.89     2_169.30       0.3752          1.0657            1.0806        41.85
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_123.41       779.14     1_902.54       0.2870          1.0888            1.1143        41.85
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_123.41     1_118.55     2_241.96       0.3752          1.0657            1.0806        41.85
IVF-TQ-b4-nl223 (self)                                 1_123.41     2_203.25     3_326.65       0.3767          1.0654            1.0641        41.85
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_258.02       473.68     1_731.70       0.1340          1.2531            1.2592        42.66
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_258.02       490.47     1_748.50       0.1340          1.2532            1.2592        42.66
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_258.02       548.51     1_806.54       0.1340          1.2532            1.2592        42.66
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_258.02       699.69     1_957.71       0.2870          1.0888            1.1143        42.66
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_258.02     1_009.47     2_267.49       0.3753          1.0657            1.0806        42.66
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_258.02       712.34     1_970.37       0.2870          1.0888            1.1143        42.66
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_258.02     1_032.27     2_290.30       0.3752          1.0657            1.0807        42.66
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_258.02       786.72     2_044.74       0.2870          1.0888            1.1143        42.66
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_258.02     1_106.86     2_364.88       0.3752          1.0657            1.0807        42.66
IVF-TQ-b4-nl316 (self)                                 1_258.02     2_260.41     3_518.43       0.3767          1.0654            1.0641        42.66
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
Exhaustive (query)                                        32.99       748.23       781.22       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.99     2_525.61     2_558.60       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              147.32       387.16       534.48       0.0755          2.3282            1.9295         7.12
ExhaustiveTQ-b2-rf5 (query)                              147.32       454.85       602.17       0.2072          1.3307            1.3578         7.12
ExhaustiveTQ-b2-rf10 (query)                             147.32       584.54       731.85       0.2886          1.2207            1.2321         7.12
ExhaustiveTQ-b2-rf20 (query)                             147.32       956.03     1_103.34       0.4151          1.1328            1.1147         7.12
ExhaustiveTQ-b2 (self)                                   147.32     3_144.09     3_291.40       0.4136          1.1619            1.1367         7.12
ExhaustiveTQ-b4-rf0 (query)                              234.13       614.54       848.67       0.1023          1.7129            1.7532        13.22
ExhaustiveTQ-b4-rf5 (query)                              234.13       695.41       929.54       0.2385          1.2770            1.2999        13.22
ExhaustiveTQ-b4-rf10 (query)                             234.13       839.85     1_073.98       0.3201          1.1873            1.1952        13.22
ExhaustiveTQ-b4-rf20 (query)                             234.13     1_189.66     1_423.78       0.4480          1.1142            1.1029        13.22
ExhaustiveTQ-b4 (self)                                   234.13     3_919.39     4_153.51       0.4462          1.1397            1.1285        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                          320.84       103.30       424.15       0.0756          2.3281            1.9295         7.63
IVF-TQ-b2-nl158-np12-rf0 (query)                         320.84       117.45       438.30       0.0755          2.3282            1.9295         7.63
IVF-TQ-b2-nl158-np17-rf0 (query)                         320.84       132.49       453.34       0.0755          2.3282            1.9295         7.63
IVF-TQ-b2-nl158-np7-rf10 (query)                         320.84       300.04       620.89       0.2887          1.2207            1.2321         7.63
IVF-TQ-b2-nl158-np7-rf20 (query)                         320.84       632.75       953.59       0.4151          1.1328            1.1147         7.63
IVF-TQ-b2-nl158-np12-rf10 (query)                        320.84       318.34       639.18       0.2886          1.2207            1.2321         7.63
IVF-TQ-b2-nl158-np12-rf20 (query)                        320.84       663.42       984.27       0.4151          1.1328            1.1147         7.63
IVF-TQ-b2-nl158-np17-rf10 (query)                        320.84       350.69       671.54       0.2886          1.2207            1.2321         7.63
IVF-TQ-b2-nl158-np17-rf20 (query)                        320.84       735.55     1_056.40       0.4151          1.1328            1.1147         7.63
IVF-TQ-b2-nl158 (self)                                   320.84     1_090.68     1_411.53       0.4136          1.1619            1.1367         7.63
IVF-TQ-b2-nl223-np11-rf0 (query)                         353.63       109.79       463.42       0.0756          2.3254            1.9253         7.73
IVF-TQ-b2-nl223-np14-rf0 (query)                         353.63       116.73       470.36       0.0756          2.3281            1.9295         7.73
IVF-TQ-b2-nl223-np21-rf0 (query)                         353.63       142.66       496.29       0.0755          2.3282            1.9295         7.73
IVF-TQ-b2-nl223-np11-rf10 (query)                        353.63       282.79       636.41       0.2891          1.2202            1.2316         7.73
IVF-TQ-b2-nl223-np11-rf20 (query)                        353.63       559.17       912.79       0.4159          1.1325            1.1142         7.73
IVF-TQ-b2-nl223-np14-rf10 (query)                        353.63       295.35       648.98       0.2886          1.2207            1.2321         7.73
IVF-TQ-b2-nl223-np14-rf20 (query)                        353.63       596.15       949.78       0.4151          1.1328            1.1147         7.73
IVF-TQ-b2-nl223-np21-rf10 (query)                        353.63       336.96       690.58       0.2886          1.2207            1.2322         7.73
IVF-TQ-b2-nl223-np21-rf20 (query)                        353.63       651.06     1_004.69       0.4151          1.1328            1.1147         7.73
IVF-TQ-b2-nl223 (self)                                   353.63     1_085.46     1_439.09       0.4136          1.1619            1.1367         7.73
IVF-TQ-b2-nl316-np15-rf0 (query)                         425.86       118.31       544.17       0.0756          2.3273            1.9286         7.92
IVF-TQ-b2-nl316-np17-rf0 (query)                         425.86       119.59       545.45       0.0756          2.3281            1.9295         7.92
IVF-TQ-b2-nl316-np25-rf0 (query)                         425.86       140.22       566.08       0.0755          2.3282            1.9295         7.92
IVF-TQ-b2-nl316-np15-rf10 (query)                        425.86       289.42       715.28       0.2892          1.2202            1.2316         7.92
IVF-TQ-b2-nl316-np15-rf20 (query)                        425.86       545.96       971.82       0.4159          1.1325            1.1141         7.92
IVF-TQ-b2-nl316-np17-rf10 (query)                        425.86       289.16       715.02       0.2886          1.2206            1.2321         7.92
IVF-TQ-b2-nl316-np17-rf20 (query)                        425.86       546.30       972.16       0.4151          1.1328            1.1147         7.92
IVF-TQ-b2-nl316-np25-rf10 (query)                        425.86       315.78       741.64       0.2886          1.2207            1.2321         7.92
IVF-TQ-b2-nl316-np25-rf20 (query)                        425.86       587.91     1_013.77       0.4151          1.1328            1.1147         7.92
IVF-TQ-b2-nl316 (self)                                   425.86     1_096.91     1_522.77       0.4136          1.1619            1.1367         7.92
IVF-TQ-b4-nl158-np7-rf0 (query)                          397.89       140.86       538.75       0.1023          1.7130            1.7533        13.90
IVF-TQ-b4-nl158-np12-rf0 (query)                         397.89       160.81       558.71       0.1023          1.7129            1.7533        13.90
IVF-TQ-b4-nl158-np17-rf0 (query)                         397.89       190.03       587.92       0.1023          1.7129            1.7533        13.90
IVF-TQ-b4-nl158-np7-rf10 (query)                         397.89       347.36       745.25       0.3201          1.1873            1.1952        13.90
IVF-TQ-b4-nl158-np7-rf20 (query)                         397.89       683.87     1_081.76       0.4480          1.1142            1.1029        13.90
IVF-TQ-b4-nl158-np12-rf10 (query)                        397.89       372.64       770.53       0.3201          1.1873            1.1952        13.90
IVF-TQ-b4-nl158-np12-rf20 (query)                        397.89       722.75     1_120.64       0.4480          1.1142            1.1029        13.90
IVF-TQ-b4-nl158-np17-rf10 (query)                        397.89       416.25       814.14       0.3201          1.1873            1.1952        13.90
IVF-TQ-b4-nl158-np17-rf20 (query)                        397.89       803.39     1_201.29       0.4480          1.1142            1.1029        13.90
IVF-TQ-b4-nl158 (self)                                   397.89     1_122.78     1_520.67       0.4462          1.1397            1.1286        13.90
IVF-TQ-b4-nl223-np11-rf0 (query)                         442.88       151.17       594.05       0.1024          1.7109            1.7520        14.04
IVF-TQ-b4-nl223-np14-rf0 (query)                         442.88       161.69       604.57       0.1023          1.7129            1.7532        14.04
IVF-TQ-b4-nl223-np21-rf0 (query)                         442.88       204.48       647.36       0.1023          1.7129            1.7533        14.04
IVF-TQ-b4-nl223-np11-rf10 (query)                        442.88       335.87       778.74       0.3206          1.1870            1.1948        14.04
IVF-TQ-b4-nl223-np11-rf20 (query)                        442.88       618.43     1_061.31       0.4489          1.1139            1.1025        14.04
IVF-TQ-b4-nl223-np14-rf10 (query)                        442.88       350.41       793.28       0.3202          1.1873            1.1952        14.04
IVF-TQ-b4-nl223-np14-rf20 (query)                        442.88       647.95     1_090.82       0.4481          1.1142            1.1029        14.04
IVF-TQ-b4-nl223-np21-rf10 (query)                        442.88       409.37       852.25       0.3201          1.1873            1.1952        14.04
IVF-TQ-b4-nl223-np21-rf20 (query)                        442.88       730.49     1_173.36       0.4480          1.1142            1.1029        14.04
IVF-TQ-b4-nl223 (self)                                   442.88     1_134.34     1_577.22       0.4462          1.1397            1.1286        14.04
IVF-TQ-b4-nl316-np15-rf0 (query)                         512.52       164.31       676.83       0.1024          1.7113            1.7520        14.32
IVF-TQ-b4-nl316-np17-rf0 (query)                         512.52       164.60       677.12       0.1023          1.7121            1.7529        14.32
IVF-TQ-b4-nl316-np25-rf0 (query)                         512.52       202.17       714.68       0.1023          1.7129            1.7533        14.32
IVF-TQ-b4-nl316-np15-rf10 (query)                        512.52       332.75       845.27       0.3207          1.1869            1.1948        14.32
IVF-TQ-b4-nl316-np15-rf20 (query)                        512.52       648.04     1_160.56       0.4491          1.1138            1.1025        14.32
IVF-TQ-b4-nl316-np17-rf10 (query)                        512.52       426.54       939.05       0.3202          1.1873            1.1952        14.32
IVF-TQ-b4-nl316-np17-rf20 (query)                        512.52       731.19     1_243.71       0.4482          1.1142            1.1029        14.32
IVF-TQ-b4-nl316-np25-rf10 (query)                        512.52       459.73       972.24       0.3201          1.1873            1.1952        14.32
IVF-TQ-b4-nl316-np25-rf20 (query)                        512.52       727.19     1_239.71       0.4480          1.1142            1.1029        14.32
IVF-TQ-b4-nl316 (self)                                   512.52     1_173.93     1_686.44       0.4462          1.1397            1.1286        14.32
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
Exhaustive (query)                                        68.27     1_391.74     1_460.01       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.27     4_637.07     4_705.34       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              340.42       656.12       996.53       0.0844          1.6537            1.5906        13.97
ExhaustiveTQ-b2-rf5 (query)                              340.42       748.79     1_089.21       0.2173          1.2230            1.2550        13.97
ExhaustiveTQ-b2-rf10 (query)                             340.42       888.26     1_228.67       0.2887          1.1550            1.1707        13.97
ExhaustiveTQ-b2-rf20 (query)                             340.42     1_300.29     1_640.71       0.4020          1.0974            1.0847        13.97
ExhaustiveTQ-b2 (self)                                   340.42     4_299.28     4_639.70       0.4025          1.1135            1.0971        13.97
ExhaustiveTQ-b4-rf0 (query)                              464.40     1_153.55     1_617.95       0.1045          1.5026            1.5346        26.18
ExhaustiveTQ-b4-rf5 (query)                              464.40     1_256.57     1_720.97       0.2294          1.2109            1.2410        26.18
ExhaustiveTQ-b4-rf10 (query)                             464.40     1_399.08     1_863.48       0.2943          1.1499            1.1675        26.18
ExhaustiveTQ-b4-rf20 (query)                             464.40     1_784.46     2_248.86       0.4029          1.0975            1.0930        26.18
ExhaustiveTQ-b4 (self)                                   464.40     5_880.98     6_345.39       0.4037          1.1130            1.1087        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                          641.92       187.25       829.16       0.0844          1.6537            1.5906        14.78
IVF-TQ-b2-nl158-np12-rf0 (query)                         641.92       203.49       845.41       0.0844          1.6537            1.5905        14.78
IVF-TQ-b2-nl158-np17-rf0 (query)                         641.92       221.55       863.47       0.0844          1.6537            1.5905        14.78
IVF-TQ-b2-nl158-np7-rf10 (query)                         641.92       415.12     1_057.04       0.2887          1.1550            1.1707        14.78
IVF-TQ-b2-nl158-np7-rf20 (query)                         641.92       768.86     1_410.78       0.4020          1.0974            1.0847        14.78
IVF-TQ-b2-nl158-np12-rf10 (query)                        641.92       425.92     1_067.84       0.2887          1.1550            1.1707        14.78
IVF-TQ-b2-nl158-np12-rf20 (query)                        641.92       790.35     1_432.27       0.4020          1.0974            1.0847        14.78
IVF-TQ-b2-nl158-np17-rf10 (query)                        641.92       446.29     1_088.21       0.2887          1.1550            1.1707        14.78
IVF-TQ-b2-nl158-np17-rf20 (query)                        641.92       824.07     1_465.98       0.4020          1.0974            1.0847        14.78
IVF-TQ-b2-nl158 (self)                                   641.92     1_449.86     2_091.78       0.4025          1.1135            1.0972        14.78
IVF-TQ-b2-nl223-np11-rf0 (query)                         650.47       201.72       852.19       0.0844          1.6536            1.5905        15.00
IVF-TQ-b2-nl223-np14-rf0 (query)                         650.47       219.92       870.39       0.0844          1.6537            1.5906        15.00
IVF-TQ-b2-nl223-np21-rf0 (query)                         650.47       241.11       891.58       0.0844          1.6537            1.5906        15.00
IVF-TQ-b2-nl223-np11-rf10 (query)                        650.47       412.94     1_063.41       0.2887          1.1550            1.1707        15.00
IVF-TQ-b2-nl223-np11-rf20 (query)                        650.47       714.73     1_365.20       0.4020          1.0974            1.0847        15.00
IVF-TQ-b2-nl223-np14-rf10 (query)                        650.47       425.77     1_076.24       0.2887          1.1550            1.1707        15.00
IVF-TQ-b2-nl223-np14-rf20 (query)                        650.47       743.51     1_393.98       0.4020          1.0974            1.0847        15.00
IVF-TQ-b2-nl223-np21-rf10 (query)                        650.47       461.21     1_111.68       0.2887          1.1550            1.1707        15.00
IVF-TQ-b2-nl223-np21-rf20 (query)                        650.47       796.87     1_447.34       0.4020          1.0974            1.0847        15.00
IVF-TQ-b2-nl223 (self)                                   650.47     1_486.72     2_137.19       0.4025          1.1135            1.0972        15.00
IVF-TQ-b2-nl316-np15-rf0 (query)                         747.71       209.20       956.91       0.0844          1.6537            1.5906        15.38
IVF-TQ-b2-nl316-np17-rf0 (query)                         747.71       216.60       964.30       0.0844          1.6537            1.5906        15.38
IVF-TQ-b2-nl316-np25-rf0 (query)                         747.71       241.59       989.30       0.0844          1.6537            1.5906        15.38
IVF-TQ-b2-nl316-np15-rf10 (query)                        747.71       406.39     1_154.10       0.2887          1.1550            1.1707        15.38
IVF-TQ-b2-nl316-np15-rf20 (query)                        747.71       692.46     1_440.17       0.4020          1.0974            1.0847        15.38
IVF-TQ-b2-nl316-np17-rf10 (query)                        747.71       414.31     1_162.02       0.2887          1.1550            1.1707        15.38
IVF-TQ-b2-nl316-np17-rf20 (query)                        747.71       709.32     1_457.03       0.4020          1.0974            1.0847        15.38
IVF-TQ-b2-nl316-np25-rf10 (query)                        747.71       447.66     1_195.37       0.2887          1.1550            1.1707        15.38
IVF-TQ-b2-nl316-np25-rf20 (query)                        747.71       754.42     1_502.13       0.4020          1.0974            1.0847        15.38
IVF-TQ-b2-nl316 (self)                                   747.71     1_482.37     2_230.08       0.4025          1.1135            1.0971        15.38
IVF-TQ-b4-nl158-np7-rf0 (query)                          743.09       261.36     1_004.45       0.1045          1.5026            1.5346        27.30
IVF-TQ-b4-nl158-np12-rf0 (query)                         743.09       291.18     1_034.27       0.1045          1.5026            1.5346        27.30
IVF-TQ-b4-nl158-np17-rf0 (query)                         743.09       321.90     1_064.99       0.1045          1.5026            1.5346        27.30
IVF-TQ-b4-nl158-np7-rf10 (query)                         743.09       508.64     1_251.73       0.2943          1.1499            1.1675        27.30
IVF-TQ-b4-nl158-np7-rf20 (query)                         743.09       869.77     1_612.86       0.4029          1.0975            1.0930        27.30
IVF-TQ-b4-nl158-np12-rf10 (query)                        743.09       524.82     1_267.91       0.2943          1.1499            1.1675        27.30
IVF-TQ-b4-nl158-np12-rf20 (query)                        743.09       897.11     1_640.20       0.4029          1.0975            1.0930        27.30
IVF-TQ-b4-nl158-np17-rf10 (query)                        743.09       560.20     1_303.29       0.2943          1.1499            1.1675        27.30
IVF-TQ-b4-nl158-np17-rf20 (query)                        743.09       937.23     1_680.32       0.4029          1.0975            1.0930        27.30
IVF-TQ-b4-nl158 (self)                                   743.09     1_635.74     2_378.83       0.4037          1.1130            1.1087        27.30
IVF-TQ-b4-nl223-np11-rf0 (query)                         769.73       284.06     1_053.79       0.1045          1.5026            1.5347        27.61
IVF-TQ-b4-nl223-np14-rf0 (query)                         769.73       303.86     1_073.58       0.1045          1.5026            1.5346        27.61
IVF-TQ-b4-nl223-np21-rf0 (query)                         769.73       355.42     1_125.14       0.1045          1.5026            1.5346        27.61
IVF-TQ-b4-nl223-np11-rf10 (query)                        769.73       513.64     1_283.37       0.2943          1.1499            1.1675        27.61
IVF-TQ-b4-nl223-np11-rf20 (query)                        769.73       815.18     1_584.91       0.4029          1.0975            1.0930        27.61
IVF-TQ-b4-nl223-np14-rf10 (query)                        769.73       529.87     1_299.60       0.2943          1.1499            1.1675        27.61
IVF-TQ-b4-nl223-np14-rf20 (query)                        769.73       860.92     1_630.65       0.4029          1.0975            1.0930        27.61
IVF-TQ-b4-nl223-np21-rf10 (query)                        769.73       589.55     1_359.28       0.2943          1.1499            1.1675        27.61
IVF-TQ-b4-nl223-np21-rf20 (query)                        769.73       937.08     1_706.81       0.4029          1.0975            1.0930        27.61
IVF-TQ-b4-nl223 (self)                                   769.73     1_725.27     2_495.00       0.4037          1.1130            1.1087        27.61
IVF-TQ-b4-nl316-np15-rf0 (query)                         865.01       297.24     1_162.25       0.1045          1.5026            1.5346        28.20
IVF-TQ-b4-nl316-np17-rf0 (query)                         865.01       321.98     1_186.99       0.1045          1.5026            1.5346        28.20
IVF-TQ-b4-nl316-np25-rf0 (query)                         865.01       355.62     1_220.64       0.1045          1.5026            1.5346        28.20
IVF-TQ-b4-nl316-np15-rf10 (query)                        865.01       509.71     1_374.72       0.2943          1.1499            1.1675        28.20
IVF-TQ-b4-nl316-np15-rf20 (query)                        865.01       798.08     1_663.09       0.4029          1.0975            1.0930        28.20
IVF-TQ-b4-nl316-np17-rf10 (query)                        865.01       517.23     1_382.25       0.2943          1.1499            1.1675        28.20
IVF-TQ-b4-nl316-np17-rf20 (query)                        865.01       817.90     1_682.91       0.4029          1.0975            1.0930        28.20
IVF-TQ-b4-nl316-np25-rf10 (query)                        865.01       570.92     1_435.94       0.2943          1.1499            1.1675        28.20
IVF-TQ-b4-nl316-np25-rf20 (query)                        865.01       887.42     1_752.43       0.4029          1.0975            1.0930        28.20
IVF-TQ-b4-nl316 (self)                                   865.01     1_705.31     2_570.33       0.4037          1.1130            1.1087        28.20
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
Exhaustive (query)                                       100.09     1_935.83     2_035.92       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.09     6_634.15     6_734.24       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              620.87       972.26     1_593.12       0.0841          1.5107            1.4226        21.33
ExhaustiveTQ-b2-rf5 (query)                              620.87     1_069.05     1_689.92       0.2144          1.1739            1.2057        21.33
ExhaustiveTQ-b2-rf10 (query)                             620.87     1_216.68     1_837.55       0.2771          1.1266            1.1512        21.33
ExhaustiveTQ-b2-rf20 (query)                             620.87     1_631.58     2_252.45       0.3769          1.0843            1.0725        21.33
ExhaustiveTQ-b2 (self)                                   620.87     5_418.41     6_039.28       0.3767          1.0935            1.0803        21.33
ExhaustiveTQ-b4-rf0 (query)                              769.90     1_783.20     2_553.10       0.0985          1.4232            1.4110        39.64
ExhaustiveTQ-b4-rf5 (query)                              769.90     1_896.36     2_666.27       0.2167          1.1746            1.2047        39.64
ExhaustiveTQ-b4-rf10 (query)                             769.90     2_021.89     2_791.79       0.2692          1.1311            1.1558        39.64
ExhaustiveTQ-b4-rf20 (query)                             769.90     2_423.99     3_193.90       0.3604          1.0923            1.1071        39.64
ExhaustiveTQ-b4 (self)                                   769.90     7_993.05     8_762.95       0.3608          1.1024            1.1182        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                          987.72       289.05     1_276.77       0.0841          1.5107            1.4226        22.46
IVF-TQ-b2-nl158-np12-rf0 (query)                         987.72       314.37     1_302.09       0.0841          1.5107            1.4226        22.46
IVF-TQ-b2-nl158-np17-rf0 (query)                         987.72       337.83     1_325.55       0.0841          1.5107            1.4226        22.46
IVF-TQ-b2-nl158-np7-rf10 (query)                         987.72       520.04     1_507.77       0.2771          1.1267            1.1512        22.46
IVF-TQ-b2-nl158-np7-rf20 (query)                         987.72       891.01     1_878.73       0.3769          1.0843            1.0725        22.46
IVF-TQ-b2-nl158-np12-rf10 (query)                        987.72       541.88     1_529.60       0.2771          1.1267            1.1512        22.46
IVF-TQ-b2-nl158-np12-rf20 (query)                        987.72       935.75     1_923.47       0.3770          1.0843            1.0725        22.46
IVF-TQ-b2-nl158-np17-rf10 (query)                        987.72       572.33     1_560.05       0.2771          1.1267            1.1512        22.46
IVF-TQ-b2-nl158-np17-rf20 (query)                        987.72       950.05     1_937.77       0.3769          1.0843            1.0725        22.46
IVF-TQ-b2-nl158 (self)                                   987.72     1_871.38     2_859.11       0.3767          1.0935            1.0803        22.46
IVF-TQ-b2-nl223-np11-rf0 (query)                         993.70       305.91     1_299.61       0.0841          1.5107            1.4226        22.78
IVF-TQ-b2-nl223-np14-rf0 (query)                         993.70       319.91     1_313.61       0.0841          1.5107            1.4226        22.78
IVF-TQ-b2-nl223-np21-rf0 (query)                         993.70       349.32     1_343.02       0.0841          1.5107            1.4226        22.78
IVF-TQ-b2-nl223-np11-rf10 (query)                        993.70       551.84     1_545.54       0.2771          1.1266            1.1512        22.78
IVF-TQ-b2-nl223-np11-rf20 (query)                        993.70       873.34     1_867.04       0.3770          1.0843            1.0724        22.78
IVF-TQ-b2-nl223-np14-rf10 (query)                        993.70       561.40     1_555.10       0.2771          1.1267            1.1512        22.78
IVF-TQ-b2-nl223-np14-rf20 (query)                        993.70       900.15     1_893.85       0.3770          1.0843            1.0724        22.78
IVF-TQ-b2-nl223-np21-rf10 (query)                        993.70       610.25     1_603.95       0.2771          1.1267            1.1512        22.78
IVF-TQ-b2-nl223-np21-rf20 (query)                        993.70       959.32     1_953.02       0.3770          1.0843            1.0725        22.78
IVF-TQ-b2-nl223 (self)                                   993.70     1_914.74     2_908.44       0.3767          1.0935            1.0803        22.78
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_127.05       318.86     1_445.91       0.0841          1.5107            1.4226        23.34
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_127.05       326.63     1_453.68       0.0841          1.5107            1.4226        23.34
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_127.05       357.25     1_484.30       0.0841          1.5107            1.4226        23.34
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_127.05       552.68     1_679.74       0.2771          1.1267            1.1512        23.34
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_127.05       858.53     1_985.58       0.3770          1.0843            1.0724        23.34
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_127.05       556.94     1_683.99       0.2771          1.1267            1.1512        23.34
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_127.05       875.30     2_002.36       0.3770          1.0843            1.0725        23.34
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_127.05       593.10     1_720.16       0.2771          1.1267            1.1512        23.34
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_127.05       922.35     2_049.41       0.3769          1.0843            1.0725        23.34
IVF-TQ-b2-nl316 (self)                                 1_127.05     1_946.31     3_073.36       0.3767          1.0935            1.0803        23.34
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_107.78       402.50     1_510.27       0.0985          1.4232            1.4110        41.24
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_107.78       458.58     1_566.36       0.0985          1.4232            1.4110        41.24
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_107.78       502.55     1_610.32       0.0985          1.4232            1.4110        41.24
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_107.78       666.22     1_774.00       0.2692          1.1311            1.1558        41.24
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_107.78     1_030.64     2_138.41       0.3604          1.0923            1.1071        41.24
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_107.78       693.53     1_801.30       0.2692          1.1311            1.1558        41.24
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_107.78     1_089.69     2_197.47       0.3604          1.0923            1.1071        41.24
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_107.78       741.85     1_849.63       0.2692          1.1311            1.1558        41.24
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_107.78     1_160.30     2_268.08       0.3604          1.0923            1.1071        41.24
IVF-TQ-b4-nl158 (self)                                 1_107.78     2_268.64     3_376.42       0.3608          1.1024            1.1182        41.24
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_140.09       446.35     1_586.44       0.0985          1.4232            1.4110        41.71
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_140.09       466.50     1_606.59       0.0985          1.4232            1.4110        41.71
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_140.09       523.01     1_663.09       0.0985          1.4232            1.4110        41.71
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_140.09       686.87     1_826.95       0.2692          1.1311            1.1558        41.71
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_140.09     1_019.41     2_159.50       0.3604          1.0923            1.1071        41.71
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_140.09       721.73     1_861.81       0.2692          1.1311            1.1558        41.71
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_140.09     1_063.09     2_203.18       0.3604          1.0923            1.1071        41.71
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_140.09       778.69     1_918.78       0.2692          1.1311            1.1558        41.71
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_140.09     1_144.70     2_284.78       0.3604          1.0923            1.1071        41.71
IVF-TQ-b4-nl223 (self)                                 1_140.09     2_265.35     3_405.44       0.3608          1.1024            1.1182        41.71
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_287.13       470.81     1_757.94       0.0985          1.4232            1.4110        42.55
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_287.13       484.28     1_771.41       0.0985          1.4232            1.4110        42.55
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_287.13       546.13     1_833.25       0.0985          1.4232            1.4110        42.55
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_287.13       702.22     1_989.35       0.2692          1.1311            1.1558        42.55
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_287.13     1_012.27     2_299.40       0.3604          1.0923            1.1071        42.55
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_287.13       716.87     2_004.00       0.2692          1.1311            1.1558        42.55
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_287.13     1_032.75     2_319.87       0.3604          1.0923            1.1071        42.55
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_287.13       785.03     2_072.15       0.2692          1.1311            1.1558        42.55
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_287.13     1_113.10     2_400.23       0.3604          1.0923            1.1071        42.55
IVF-TQ-b4-nl316 (self)                                 1_287.13     2_346.65     3_633.77       0.3608          1.1024            1.1182        42.55
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
Exhaustive (query)                                        33.29       869.15       902.44       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.29     2_700.11     2_733.40       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              154.17       418.05       572.22       0.7918          1.0898            1.0632         7.12
ExhaustiveTQ-b2-rf5 (query)                              154.17       463.26       617.43       0.9995          1.0000            1.0000         7.12
ExhaustiveTQ-b2-rf10 (query)                             154.17       600.79       754.95       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b2-rf20 (query)                             154.17       990.51     1_144.67       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b2 (self)                                   154.17     3_244.56     3_398.73       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b4-rf0 (query)                              229.68       613.94       843.62       0.8727          1.0322            1.0183        13.22
ExhaustiveTQ-b4-rf5 (query)                              229.68       699.32       929.01       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4-rf10 (query)                             229.68       825.26     1_054.94       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4-rf20 (query)                             229.68     1_203.84     1_433.52       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4 (self)                                   229.68     3_975.27     4_204.95       1.0000          1.0000            1.0000        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                          510.54       128.51       639.05       0.7916          1.0897            1.0634         7.59
IVF-TQ-b2-nl158-np12-rf0 (query)                         510.54       172.86       683.39       0.7918          1.0898            1.0632         7.59
IVF-TQ-b2-nl158-np17-rf0 (query)                         510.54       211.44       721.97       0.7918          1.0898            1.0632         7.59
IVF-TQ-b2-nl158-np7-rf10 (query)                         510.54       340.97       851.51       0.9982          1.0004            1.0000         7.59
IVF-TQ-b2-nl158-np7-rf20 (query)                         510.54       624.12     1_134.66       0.9982          1.0004            1.0000         7.59
IVF-TQ-b2-nl158-np12-rf10 (query)                        510.54       402.76       913.29       1.0000          1.0000            1.0000         7.59
IVF-TQ-b2-nl158-np12-rf20 (query)                        510.54       724.65     1_235.19       1.0000          1.0000            1.0000         7.59
IVF-TQ-b2-nl158-np17-rf10 (query)                        510.54       451.23       961.77       1.0000          1.0000            1.0000         7.59
IVF-TQ-b2-nl158-np17-rf20 (query)                        510.54       803.59     1_314.13       1.0000          1.0000            1.0000         7.59
IVF-TQ-b2-nl158 (self)                                   510.54     1_207.56     1_718.10       1.0000          1.0000            1.0000         7.59
IVF-TQ-b2-nl223-np11-rf0 (query)                         586.68       127.87       714.56       0.7919          1.0897            1.0632         7.73
IVF-TQ-b2-nl223-np14-rf0 (query)                         586.68       144.42       731.11       0.7918          1.0897            1.0632         7.73
IVF-TQ-b2-nl223-np21-rf0 (query)                         586.68       181.77       768.45       0.7918          1.0898            1.0632         7.73
IVF-TQ-b2-nl223-np11-rf10 (query)                        586.68       321.06       907.74       0.9995          1.0001            1.0000         7.73
IVF-TQ-b2-nl223-np11-rf20 (query)                        586.68       594.66     1_181.34       0.9995          1.0001            1.0000         7.73
IVF-TQ-b2-nl223-np14-rf10 (query)                        586.68       349.18       935.87       0.9999          1.0000            1.0000         7.73
IVF-TQ-b2-nl223-np14-rf20 (query)                        586.68       638.68     1_225.36       0.9999          1.0000            1.0000         7.73
IVF-TQ-b2-nl223-np21-rf10 (query)                        586.68       405.84       992.52       1.0000          1.0000            1.0000         7.73
IVF-TQ-b2-nl223-np21-rf20 (query)                        586.68       728.89     1_315.57       1.0000          1.0000            1.0000         7.73
IVF-TQ-b2-nl223 (self)                                   586.68     1_078.16     1_664.84       1.0000          1.0000            1.0000         7.73
IVF-TQ-b2-nl316-np15-rf0 (query)                         713.10       128.87       841.97       0.7918          1.0897            1.0632         7.92
IVF-TQ-b2-nl316-np17-rf0 (query)                         713.10       140.43       853.53       0.7918          1.0898            1.0632         7.92
IVF-TQ-b2-nl316-np25-rf0 (query)                         713.10       170.07       883.17       0.7918          1.0898            1.0632         7.92
IVF-TQ-b2-nl316-np15-rf10 (query)                        713.10       321.06     1_034.15       0.9998          1.0000            1.0000         7.92
IVF-TQ-b2-nl316-np15-rf20 (query)                        713.10       580.41     1_293.51       0.9998          1.0000            1.0000         7.92
IVF-TQ-b2-nl316-np17-rf10 (query)                        713.10       324.26     1_037.35       0.9999          1.0000            1.0000         7.92
IVF-TQ-b2-nl316-np17-rf20 (query)                        713.10       600.47     1_313.57       0.9999          1.0000            1.0000         7.92
IVF-TQ-b2-nl316-np25-rf10 (query)                        713.10       371.09     1_084.18       1.0000          1.0000            1.0000         7.92
IVF-TQ-b2-nl316-np25-rf20 (query)                        713.10       685.63     1_398.72       1.0000          1.0000            1.0000         7.92
IVF-TQ-b2-nl316 (self)                                   713.10     1_036.45     1_749.54       1.0000          1.0000            1.0000         7.92
IVF-TQ-b4-nl158-np7-rf0 (query)                          597.05       182.81       779.86       0.8721          1.0325            1.0187        13.82
IVF-TQ-b4-nl158-np12-rf0 (query)                         597.05       254.95       852.01       0.8727          1.0322            1.0183        13.82
IVF-TQ-b4-nl158-np17-rf0 (query)                         597.05       318.02       915.07       0.8727          1.0322            1.0183        13.82
IVF-TQ-b4-nl158-np7-rf10 (query)                         597.05       402.80       999.85       0.9982          1.0004            1.0000        13.82
IVF-TQ-b4-nl158-np7-rf20 (query)                         597.05       700.87     1_297.92       0.9982          1.0004            1.0000        13.82
IVF-TQ-b4-nl158-np12-rf10 (query)                        597.05       487.49     1_084.54       1.0000          1.0000            1.0000        13.82
IVF-TQ-b4-nl158-np12-rf20 (query)                        597.05       814.18     1_411.24       1.0000          1.0000            1.0000        13.82
IVF-TQ-b4-nl158-np17-rf10 (query)                        597.05       561.98     1_159.03       1.0000          1.0000            1.0000        13.82
IVF-TQ-b4-nl158-np17-rf20 (query)                        597.05       910.34     1_507.39       1.0000          1.0000            1.0000        13.82
IVF-TQ-b4-nl158 (self)                                   597.05     1_354.00     1_951.05       1.0000          1.0000            1.0000        13.82
IVF-TQ-b4-nl223-np11-rf0 (query)                         669.11       178.52       847.63       0.8726          1.0323            1.0184        14.04
IVF-TQ-b4-nl223-np14-rf0 (query)                         669.11       207.32       876.43       0.8727          1.0322            1.0183        14.04
IVF-TQ-b4-nl223-np21-rf0 (query)                         669.11       270.29       939.40       0.8727          1.0322            1.0183        14.04
IVF-TQ-b4-nl223-np11-rf10 (query)                        669.11       376.49     1_045.59       0.9995          1.0001            1.0000        14.04
IVF-TQ-b4-nl223-np11-rf20 (query)                        669.11       658.81     1_327.92       0.9995          1.0001            1.0000        14.04
IVF-TQ-b4-nl223-np14-rf10 (query)                        669.11       414.92     1_084.02       0.9999          1.0000            1.0000        14.04
IVF-TQ-b4-nl223-np14-rf20 (query)                        669.11       708.01     1_377.11       0.9999          1.0000            1.0000        14.04
IVF-TQ-b4-nl223-np21-rf10 (query)                        669.11       493.39     1_162.50       1.0000          1.0000            1.0000        14.04
IVF-TQ-b4-nl223-np21-rf20 (query)                        669.11       824.16     1_493.27       1.0000          1.0000            1.0000        14.04
IVF-TQ-b4-nl223 (self)                                   669.11     1_194.12     1_863.22       1.0000          1.0000            1.0000        14.04
IVF-TQ-b4-nl316-np15-rf0 (query)                         802.38       179.48       981.86       0.8726          1.0322            1.0184        14.33
IVF-TQ-b4-nl316-np17-rf0 (query)                         802.38       194.53       996.92       0.8727          1.0322            1.0183        14.33
IVF-TQ-b4-nl316-np25-rf0 (query)                         802.38       249.06     1_051.44       0.8727          1.0322            1.0183        14.33
IVF-TQ-b4-nl316-np15-rf10 (query)                        802.38       366.08     1_168.47       0.9998          1.0000            1.0000        14.33
IVF-TQ-b4-nl316-np15-rf20 (query)                        802.38       637.32     1_439.71       0.9998          1.0000            1.0000        14.33
IVF-TQ-b4-nl316-np17-rf10 (query)                        802.38       394.32     1_196.70       0.9999          1.0000            1.0000        14.33
IVF-TQ-b4-nl316-np17-rf20 (query)                        802.38       666.18     1_468.56       0.9999          1.0000            1.0000        14.33
IVF-TQ-b4-nl316-np25-rf10 (query)                        802.38       450.00     1_252.39       1.0000          1.0000            1.0000        14.33
IVF-TQ-b4-nl316-np25-rf20 (query)                        802.38       753.34     1_555.73       1.0000          1.0000            1.0000        14.33
IVF-TQ-b4-nl316 (self)                                   802.38     1_126.09     1_928.48       1.0000          1.0000            1.0000        14.33
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
Exhaustive (query)                                        68.25     1_387.93     1_456.19       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.25     4_671.33     4_739.59       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              340.88       661.48     1_002.36       0.8424          1.0447            1.0331        13.97
ExhaustiveTQ-b2-rf5 (query)                              340.88       751.11     1_091.99       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2-rf10 (query)                             340.88       891.21     1_232.09       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2-rf20 (query)                             340.88     1_293.01     1_633.89       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2 (self)                                   340.88     4_248.54     4_589.42       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b4-rf0 (query)                              459.99     1_155.99     1_615.98       0.8985          1.0191            1.0110        26.18
ExhaustiveTQ-b4-rf5 (query)                              459.99     1_253.46     1_713.45       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4-rf10 (query)                             459.99     1_388.67     1_848.66       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4-rf20 (query)                             459.99     1_791.27     2_251.26       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4 (self)                                   459.99     5_865.49     6_325.48       1.0000          1.0000            1.0000        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                          864.63       233.30     1_097.93       0.8422          1.0449            1.0333        14.78
IVF-TQ-b2-nl158-np12-rf0 (query)                         864.63       300.48     1_165.11       0.8424          1.0447            1.0331        14.78
IVF-TQ-b2-nl158-np17-rf0 (query)                         864.63       365.30     1_229.93       0.8424          1.0447            1.0331        14.78
IVF-TQ-b2-nl158-np7-rf10 (query)                         864.63       464.77     1_329.41       0.9987          1.0003            1.0000        14.78
IVF-TQ-b2-nl158-np7-rf20 (query)                         864.63       770.97     1_635.60       0.9987          1.0003            1.0000        14.78
IVF-TQ-b2-nl158-np12-rf10 (query)                        864.63       546.99     1_411.62       0.9999          1.0000            1.0000        14.78
IVF-TQ-b2-nl158-np12-rf20 (query)                        864.63       883.65     1_748.28       0.9999          1.0000            1.0000        14.78
IVF-TQ-b2-nl158-np17-rf10 (query)                        864.63       616.48     1_481.11       1.0000          1.0000            1.0000        14.78
IVF-TQ-b2-nl158-np17-rf20 (query)                        864.63       976.27     1_840.91       1.0000          1.0000            1.0000        14.78
IVF-TQ-b2-nl158 (self)                                   864.63     1_595.12     2_459.75       1.0000          1.0000            1.0000        14.78
IVF-TQ-b2-nl223-np11-rf0 (query)                         881.32       232.21     1_113.53       0.8424          1.0447            1.0331        15.07
IVF-TQ-b2-nl223-np14-rf0 (query)                         881.32       260.75     1_142.06       0.8424          1.0447            1.0331        15.07
IVF-TQ-b2-nl223-np21-rf0 (query)                         881.32       321.48     1_202.79       0.8424          1.0447            1.0331        15.07
IVF-TQ-b2-nl223-np11-rf10 (query)                        881.32       449.62     1_330.94       0.9997          1.0000            1.0000        15.07
IVF-TQ-b2-nl223-np11-rf20 (query)                        881.32       745.91     1_627.22       0.9997          1.0000            1.0000        15.07
IVF-TQ-b2-nl223-np14-rf10 (query)                        881.32       485.61     1_366.93       0.9999          1.0000            1.0000        15.07
IVF-TQ-b2-nl223-np14-rf20 (query)                        881.32       797.78     1_679.10       0.9999          1.0000            1.0000        15.07
IVF-TQ-b2-nl223-np21-rf10 (query)                        881.32       554.42     1_435.74       1.0000          1.0000            1.0000        15.07
IVF-TQ-b2-nl223-np21-rf20 (query)                        881.32       883.37     1_764.69       1.0000          1.0000            1.0000        15.07
IVF-TQ-b2-nl223 (self)                                   881.32     1_550.43     2_431.75       1.0000          1.0000            1.0000        15.07
IVF-TQ-b2-nl316-np15-rf0 (query)                         984.61       235.32     1_219.93       0.8424          1.0447            1.0331        15.38
IVF-TQ-b2-nl316-np17-rf0 (query)                         984.61       250.78     1_235.39       0.8424          1.0447            1.0331        15.38
IVF-TQ-b2-nl316-np25-rf0 (query)                         984.61       307.84     1_292.45       0.8424          1.0447            1.0331        15.38
IVF-TQ-b2-nl316-np15-rf10 (query)                        984.61       448.90     1_433.51       0.9999          1.0000            1.0000        15.38
IVF-TQ-b2-nl316-np15-rf20 (query)                        984.61       758.53     1_743.14       0.9999          1.0000            1.0000        15.38
IVF-TQ-b2-nl316-np17-rf10 (query)                        984.61       464.14     1_448.76       1.0000          1.0000            1.0000        15.38
IVF-TQ-b2-nl316-np17-rf20 (query)                        984.61       776.28     1_760.90       1.0000          1.0000            1.0000        15.38
IVF-TQ-b2-nl316-np25-rf10 (query)                        984.61       526.98     1_511.59       1.0000          1.0000            1.0000        15.38
IVF-TQ-b2-nl316-np25-rf20 (query)                        984.61       857.59     1_842.20       1.0000          1.0000            1.0000        15.38
IVF-TQ-b2-nl316 (self)                                   984.61     1_445.57     2_430.18       1.0000          1.0000            1.0000        15.38
IVF-TQ-b4-nl158-np7-rf0 (query)                          970.59       334.54     1_305.13       0.8979          1.0193            1.0113        27.29
IVF-TQ-b4-nl158-np12-rf0 (query)                         970.59       462.75     1_433.34       0.8985          1.0191            1.0110        27.29
IVF-TQ-b4-nl158-np17-rf0 (query)                         970.59       573.64     1_544.23       0.8985          1.0191            1.0110        27.29
IVF-TQ-b4-nl158-np7-rf10 (query)                         970.59       570.38     1_540.97       0.9987          1.0003            1.0000        27.29
IVF-TQ-b4-nl158-np7-rf20 (query)                         970.59       881.03     1_851.62       0.9987          1.0003            1.0000        27.29
IVF-TQ-b4-nl158-np12-rf10 (query)                        970.59       727.54     1_698.13       0.9999          1.0000            1.0000        27.29
IVF-TQ-b4-nl158-np12-rf20 (query)                        970.59     1_113.97     2_084.57       0.9999          1.0000            1.0000        27.29
IVF-TQ-b4-nl158-np17-rf10 (query)                        970.59       856.85     1_827.44       1.0000          1.0000            1.0000        27.29
IVF-TQ-b4-nl158-np17-rf20 (query)                        970.59     1_208.76     2_179.35       1.0000          1.0000            1.0000        27.29
IVF-TQ-b4-nl158 (self)                                   970.59     1_927.96     2_898.56       1.0000          1.0000            1.0000        27.29
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_003.54       344.93     1_348.46       0.8984          1.0191            1.0111        27.74
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_003.54       397.51     1_401.05       0.8985          1.0191            1.0110        27.74
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_003.54       540.50     1_544.03       0.8985          1.0191            1.0110        27.74
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_003.54       560.99     1_564.52       0.9997          1.0000            1.0000        27.74
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_003.54       860.98     1_864.52       0.9997          1.0000            1.0000        27.74
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_003.54       633.64     1_637.17       0.9999          1.0000            1.0000        27.74
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_003.54       966.21     1_969.75       0.9999          1.0000            1.0000        27.74
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_003.54       738.85     1_742.39       1.0000          1.0000            1.0000        27.74
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_003.54     1_073.36     2_076.90       1.0000          1.0000            1.0000        27.74
IVF-TQ-b4-nl223 (self)                                 1_003.54     1_763.49     2_767.03       1.0000          1.0000            1.0000        27.74
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_109.29       352.54     1_461.82       0.8985          1.0191            1.0110        28.18
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_109.29       379.94     1_489.22       0.8985          1.0191            1.0110        28.18
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_109.29       471.60     1_580.89       0.8985          1.0191            1.0110        28.18
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_109.29       561.01     1_670.29       0.9999          1.0000            1.0000        28.18
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_109.29       863.19     1_972.47       0.9999          1.0000            1.0000        28.18
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_109.29       627.13     1_736.41       1.0000          1.0000            1.0000        28.18
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_109.29       899.19     2_008.48       1.0000          1.0000            1.0000        28.18
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_109.29       692.57     1_801.85       1.0000          1.0000            1.0000        28.18
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_109.29     1_022.69     2_131.97       1.0000          1.0000            1.0000        28.18
IVF-TQ-b4-nl316 (self)                                 1_109.29     1_690.57     2_799.85       1.0000          1.0000            1.0000        28.18
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
Exhaustive (query)                                       100.17     2_007.09     2_107.26       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.17     6_766.27     6_866.44       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              613.46       993.79     1_607.25       0.8736          1.0271            1.0200        21.33
ExhaustiveTQ-b2-rf5 (query)                              613.46     1_088.70     1_702.15       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2-rf10 (query)                             613.46     1_236.46     1_849.91       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2-rf20 (query)                             613.46     1_645.12     2_258.57       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2 (self)                                   613.46     5_423.90     6_037.36       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b4-rf0 (query)                              769.46     1_803.03     2_572.48       0.9097          1.0146            1.0083        39.64
ExhaustiveTQ-b4-rf5 (query)                              769.46     1_909.83     2_679.28       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4-rf10 (query)                             769.46     2_018.22     2_787.67       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4-rf20 (query)                             769.46     2_407.23     3_176.69       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4 (self)                                   769.46     7_949.44     8_718.89       1.0000          1.0000            1.0000        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_420.81       353.17     1_773.98       0.8735          1.0272            1.0201        22.42
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_420.81       454.99     1_875.81       0.8736          1.0271            1.0200        22.42
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_420.81       529.86     1_950.67       0.8736          1.0271            1.0200        22.42
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_420.81       607.12     2_027.93       0.9995          1.0001            1.0000        22.42
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_420.81       937.06     2_357.88       0.9995          1.0001            1.0000        22.42
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_420.81       720.96     2_141.78       1.0000          1.0000            1.0000        22.42
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_420.81     1_084.94     2_505.75       1.0000          1.0000            1.0000        22.42
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_420.81       807.55     2_228.36       1.0000          1.0000            1.0000        22.42
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_420.81     1_186.54     2_607.36       1.0000          1.0000            1.0000        22.42
IVF-TQ-b2-nl158 (self)                                 1_420.81     2_084.87     3_505.68       1.0000          1.0000            1.0000        22.42
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_514.81       347.33     1_862.15       0.8736          1.0271            1.0200        22.81
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_514.81       386.81     1_901.62       0.8736          1.0271            1.0200        22.81
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_514.81       471.15     1_985.97       0.8736          1.0271            1.0200        22.81
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_514.81       589.27     2_104.08       0.9998          1.0000            1.0000        22.81
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_514.81       917.82     2_432.64       0.9998          1.0000            1.0000        22.81
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_514.81       634.08     2_148.90       1.0000          1.0000            1.0000        22.81
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_514.81       978.31     2_493.13       1.0000          1.0000            1.0000        22.81
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_514.81       729.20     2_244.01       1.0000          1.0000            1.0000        22.81
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_514.81     1_095.26     2_610.08       1.0000          1.0000            1.0000        22.81
IVF-TQ-b2-nl223 (self)                                 1_514.81     1_943.36     3_458.17       1.0000          1.0000            1.0000        22.81
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_798.65       357.40     2_156.05       0.8736          1.0271            1.0200        23.34
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_798.65       380.85     2_179.50       0.8736          1.0271            1.0200        23.34
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_798.65       453.19     2_251.84       0.8736          1.0271            1.0200        23.34
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_798.65       595.65     2_394.30       1.0000          1.0000            1.0000        23.34
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_798.65       921.87     2_720.52       1.0000          1.0000            1.0000        23.34
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_798.65       614.67     2_413.32       1.0000          1.0000            1.0000        23.34
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_798.65       956.12     2_754.77       1.0000          1.0000            1.0000        23.34
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_798.65       695.63     2_494.28       1.0000          1.0000            1.0000        23.34
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_798.65     1_068.73     2_867.38       1.0000          1.0000            1.0000        23.34
IVF-TQ-b2-nl316 (self)                                 1_798.65     1_974.15     3_772.80       1.0000          1.0000            1.0000        23.34
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_532.21       519.32     2_051.53       0.9094          1.0147            1.0084        41.17
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_532.21       713.00     2_245.21       0.9097          1.0146            1.0083        41.17
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_532.21       866.61     2_398.82       0.9097          1.0146            1.0083        41.17
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_532.21       783.42     2_315.63       0.9995          1.0001            1.0000        41.17
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_532.21     1_111.15     2_643.36       0.9995          1.0001            1.0000        41.17
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_532.21       978.34     2_510.56       1.0000          1.0000            1.0000        41.17
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_532.21     1_338.88     2_871.09       1.0000          1.0000            1.0000        41.17
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_532.21     1_131.35     2_663.56       1.0000          1.0000            1.0000        41.17
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_532.21     1_512.32     3_044.54       1.0000          1.0000            1.0000        41.17
IVF-TQ-b4-nl158 (self)                                 1_532.21     2_679.79     4_212.00       1.0000          1.0000            1.0000        41.17
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_676.24       533.04     2_209.28       0.9096          1.0146            1.0084        41.75
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_676.24       606.18     2_282.42       0.9097          1.0146            1.0083        41.75
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_676.24       772.59     2_448.84       0.9097          1.0146            1.0083        41.75
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_676.24       763.38     2_439.62       0.9998          1.0000            1.0000        41.75
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_676.24     1_085.76     2_762.00       0.9998          1.0000            1.0000        41.75
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_676.24       859.73     2_535.97       1.0000          1.0000            1.0000        41.75
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_676.24     1_181.50     2_857.75       1.0000          1.0000            1.0000        41.75
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_676.24     1_024.53     2_700.77       1.0000          1.0000            1.0000        41.75
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_676.24     1_374.10     3_050.34       1.0000          1.0000            1.0000        41.75
IVF-TQ-b4-nl223 (self)                                 1_676.24     2_486.55     4_162.79       1.0000          1.0000            1.0000        41.75
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_968.22       551.78     2_520.00       0.9097          1.0146            1.0083        42.54
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_968.22       587.64     2_555.85       0.9097          1.0146            1.0083        42.54
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_968.22       738.59     2_706.81       0.9097          1.0146            1.0083        42.54
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_968.22       779.10     2_747.32       1.0000          1.0000            1.0000        42.54
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_968.22     1_088.58     3_056.80       1.0000          1.0000            1.0000        42.54
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_968.22       811.78     2_780.00       1.0000          1.0000            1.0000        42.54
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_968.22     1_149.41     3_117.63       1.0000          1.0000            1.0000        42.54
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_968.22       970.16     2_938.38       1.0000          1.0000            1.0000        42.54
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_968.22     1_314.22     3_282.44       1.0000          1.0000            1.0000        42.54
IVF-TQ-b4-nl316 (self)                                 1_968.22     2_454.45     4_422.67       1.0000          1.0000            1.0000        42.54
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*ann-search-rs 0.12.0 (commit v0.11.2-37-ge44af67), run on 2026-10-31.*
*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
