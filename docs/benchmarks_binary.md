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
Exhaustive (query)                                        32.44       671.44       703.89       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.44     2_208.34     2_240.78       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                104.46       236.90       341.36       0.1199          1.4617            1.4199         1.78
ExhaustiveBinary-256-random-rf10 (query)                 104.46       336.43       440.88       0.3411          1.0941            1.0814         1.78
ExhaustiveBinary-256-random-rf20 (query)                 104.46       434.60       539.06       0.4467          1.0571            1.0475         1.78
ExhaustiveBinary-256-random (self)                       104.46     1_089.21     1_193.67       0.3454          1.0895            1.0798         1.78
ExhaustiveBinary-256-pca_no_rr (query)                   140.46       235.69       376.15       0.1153          1.4748            1.4212         1.78
ExhaustiveBinary-256-pca-rf10 (query)                    140.46       337.36       477.82       0.3323          1.1029            1.0834         1.78
ExhaustiveBinary-256-pca-rf20 (query)                    140.46       433.18       573.63       0.4387          1.0631            1.0485         1.78
ExhaustiveBinary-256-pca (self)                          140.46     1_153.18     1_293.63       0.3391          1.0957            1.0813         1.78
ExhaustiveBinary-512-random_no_rr (query)                121.04       355.28       476.32       0.1588          1.3547            1.3300         3.55
ExhaustiveBinary-512-random-rf10 (query)                 121.04       458.44       579.48       0.3786          1.0692            1.0677         3.55
ExhaustiveBinary-512-random-rf20 (query)                 121.04       562.72       683.76       0.4874          1.0424            1.0395         3.55
ExhaustiveBinary-512-random (self)                       121.04     1_495.98     1_617.02       0.3805          1.0675            1.0675         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   149.61       351.07       500.68       0.1564          1.3535            1.3265         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    149.61       457.23       606.83       0.3789          1.0710            1.0663         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    149.61       562.94       712.54       0.4903          1.0433            1.0387         3.55
ExhaustiveBinary-512-pca (self)                          149.61     1_493.29     1_642.89       0.3823          1.0678            1.0665         3.55
ExhaustiveBinary-1024-random_no_rr (query)               152.37       489.66       642.03       0.1929          1.2764            1.2696         7.10
ExhaustiveBinary-1024-random-rf10 (query)                152.37       612.74       765.11       0.4214          1.0550            1.0552         7.10
ExhaustiveBinary-1024-random-rf20 (query)                152.37       716.56       868.93       0.5434          1.0327            1.0308         7.10
ExhaustiveBinary-1024-random (self)                      152.37     1_991.30     2_143.67       0.4232          1.0547            1.0552         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  149.01       496.12       645.13       0.1921          1.2733            1.2652         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   149.01       611.53       760.54       0.4226          1.0546            1.0544         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   149.01       718.68       867.69       0.5443          1.0326            1.0305         7.10
ExhaustiveBinary-1024-pca (self)                         149.01     2_001.02     2_150.04       0.4236          1.0546            1.0548         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   41.79       431.71       473.50       0.1211          1.4987            1.4523         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    41.79       463.87       505.66       0.3284          1.1039            1.0884         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    41.79       720.38       762.18       0.4385          1.0624            1.0494         1.53
ExhaustiveBinary-256-sign (self)                          41.79     1_553.75     1_595.54       0.3334          1.0988            1.0859         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)              249.74        50.71       300.45       0.1245          1.4365            1.4004         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)             249.74        52.30       302.04       0.1245          1.4365            1.4004         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)             249.74        55.77       305.51       0.1245          1.4365            1.4004         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)             249.74        98.95       348.69       0.3492          1.0889            1.0776         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)             249.74       149.40       399.15       0.4555          1.0543            1.0456         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)            249.74        96.46       346.20       0.3492          1.0889            1.0776         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)            249.74       147.28       397.02       0.4555          1.0543            1.0456         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)            249.74        98.24       347.98       0.3492          1.0889            1.0776         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)            249.74       148.46       398.20       0.4555          1.0543            1.0456         1.93
IVF-Binary-256-nl158-random (self)                       249.74       212.69       462.43       0.3542          1.0841            1.0761         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             259.73        44.40       304.13       0.1414          1.3598            1.3149         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             259.73        46.76       306.49       0.1414          1.3603            1.3152         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             259.73        49.12       308.84       0.1414          1.3604            1.3152         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            259.73        96.01       355.74       0.3893          1.0691            1.0625         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            259.73       147.90       407.63       0.4974          1.0428            1.0371         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            259.73        95.30       355.03       0.3890          1.0692            1.0625         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            259.73       146.12       405.85       0.4969          1.0429            1.0372         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            259.73       101.59       361.32       0.3890          1.0692            1.0625         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            259.73       149.42       409.15       0.4969          1.0429            1.0372         2.00
IVF-Binary-256-nl223-random (self)                       259.73       212.23       471.96       0.3942          1.0643            1.0614         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             327.92        47.77       375.70       0.1498          1.3359            1.2910         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             327.92        49.04       376.97       0.1498          1.3363            1.2912         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             327.92        51.56       379.49       0.1498          1.3364            1.2912         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            327.92        97.72       425.65       0.4016          1.0646            1.0586         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            327.92       149.08       477.01       0.5055          1.0413            1.0359         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            327.92        96.97       424.90       0.4015          1.0647            1.0586         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            327.92       144.99       472.92       0.5053          1.0413            1.0360         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            327.92       102.96       430.88       0.4015          1.0647            1.0586         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            327.92       148.19       476.12       0.5053          1.0413            1.0360         2.09
IVF-Binary-256-nl316-random (self)                       327.92       221.97       549.89       0.4062          1.0604            1.0577         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)                 280.13        39.82       319.95       0.1201          1.4453            1.4029         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)                280.13        42.98       323.11       0.1201          1.4453            1.4029         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)                280.13        43.41       323.54       0.1201          1.4453            1.4029         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)                280.13        90.09       370.22       0.3431          1.0959            1.0788         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)                280.13       140.64       420.77       0.4521          1.0584            1.0457         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)               280.13        90.57       370.71       0.3431          1.0959            1.0788         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)               280.13       140.96       421.09       0.4521          1.0584            1.0457         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)               280.13        92.72       372.85       0.3431          1.0959            1.0788         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)               280.13       141.66       421.79       0.4521          1.0584            1.0457         1.93
IVF-Binary-256-nl158-pca (self)                          280.13       201.36       481.49       0.3495          1.0891            1.0771         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                332.30        47.04       379.34       0.1378          1.3708            1.3186         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                332.30        50.42       382.72       0.1377          1.3712            1.3187         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                332.30        62.72       395.02       0.1377          1.3712            1.3187         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               332.30        96.56       428.86       0.3827          1.0753            1.0638         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               332.30       140.27       472.57       0.4957          1.0457            1.0375         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               332.30        96.07       428.36       0.3826          1.0754            1.0638         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               332.30       145.44       477.73       0.4955          1.0457            1.0375         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               332.30        98.19       430.49       0.3826          1.0754            1.0638         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               332.30       147.78       480.08       0.4955          1.0457            1.0375         2.00
IVF-Binary-256-nl223-pca (self)                          332.30       204.19       536.48       0.3896          1.0691            1.0621         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                395.15        47.17       442.32       0.1466          1.3422            1.2923         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                395.15        47.79       442.94       0.1466          1.3426            1.2924         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                395.15        50.78       445.93       0.1466          1.3427            1.2924         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               395.15        98.19       493.33       0.3964          1.0705            1.0596         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               395.15       150.67       545.81       0.5067          1.0437            1.0356         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               395.15        97.31       492.46       0.3963          1.0705            1.0596         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               395.15       144.30       539.45       0.5066          1.0437            1.0356         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               395.15       100.95       496.10       0.3963          1.0705            1.0596         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               395.15       150.57       545.71       0.5066          1.0437            1.0356         2.09
IVF-Binary-256-nl316-pca (self)                          395.15       217.44       612.59       0.4023          1.0647            1.0583         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)              305.23        65.43       370.66       0.1614          1.3415            1.3204         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)             305.23        67.23       372.46       0.1614          1.3415            1.3204         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)             305.23        71.07       376.30       0.1614          1.3415            1.3204         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)             305.23       118.00       423.23       0.3837          1.0670            1.0659         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)             305.23       166.54       471.77       0.4941          1.0409            1.0384         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)            305.23       117.99       423.22       0.3837          1.0670            1.0659         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)            305.23       167.24       472.47       0.4941          1.0409            1.0384         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)            305.23       117.54       422.77       0.3837          1.0670            1.0659         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)            305.23       169.86       475.09       0.4941          1.0409            1.0384         3.71
IVF-Binary-512-nl158-random (self)                       305.23       289.08       594.31       0.3859          1.0652            1.0657         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             314.32        61.45       375.77       0.1711          1.2997            1.2781         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             314.32        63.03       377.35       0.1711          1.3000            1.2783         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             314.32        68.76       383.08       0.1711          1.3000            1.2783         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            314.32       115.19       429.51       0.4018          1.0605            1.0593         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            314.32       162.92       477.24       0.5143          1.0372            1.0348         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            314.32       116.48       430.80       0.4016          1.0605            1.0593         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            314.32       164.61       478.93       0.5140          1.0373            1.0348         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            314.32       119.19       433.51       0.4016          1.0605            1.0593         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            314.32       175.23       489.55       0.5140          1.0373            1.0348         3.77
IVF-Binary-512-nl223-random (self)                       314.32       288.27       602.59       0.4038          1.0592            1.0592         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             378.29        67.50       445.79       0.1755          1.2882            1.2668         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             378.29        68.65       446.94       0.1755          1.2884            1.2671         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             378.29        72.29       450.57       0.1755          1.2884            1.2671         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            378.29       117.91       496.20       0.4061          1.0594            1.0581         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            378.29       170.96       549.25       0.5177          1.0368            1.0341         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            378.29       122.46       500.75       0.4060          1.0595            1.0581         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            378.29       170.74       549.03       0.5174          1.0368            1.0341         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            378.29       122.74       501.03       0.4060          1.0595            1.0581         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            378.29       173.03       551.31       0.5174          1.0368            1.0341         3.86
IVF-Binary-512-nl316-random (self)                       378.29       296.40       674.69       0.4089          1.0580            1.0579         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)                 343.91        57.67       401.58       0.1596          1.3401            1.3152         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)                343.91        60.81       404.72       0.1596          1.3401            1.3152         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)                343.91        64.35       408.26       0.1596          1.3401            1.3152         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)                343.91       112.94       456.86       0.3846          1.0687            1.0645         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)                343.91       162.77       506.68       0.4979          1.0417            1.0374         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)               343.91       115.16       459.07       0.3846          1.0687            1.0645         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)               343.91       166.53       510.44       0.4979          1.0417            1.0374         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)               343.91       116.86       460.77       0.3846          1.0687            1.0645         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)               343.91       170.01       513.92       0.4979          1.0417            1.0374         3.71
IVF-Binary-512-nl158-pca (self)                          343.91       287.21       631.12       0.3875          1.0659            1.0647         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                347.36        62.76       410.11       0.1695          1.3031            1.2743         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                347.36        63.24       410.59       0.1695          1.3032            1.2744         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                347.36        67.96       415.32       0.1695          1.3032            1.2744         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               347.36       113.99       461.35       0.4038          1.0616            1.0582         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               347.36       165.87       513.23       0.5168          1.0382            1.0342         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               347.36       114.61       461.97       0.4037          1.0617            1.0583         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               347.36       164.58       511.94       0.5167          1.0382            1.0342         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               347.36       120.31       467.66       0.4037          1.0617            1.0583         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               347.36       168.79       516.14       0.5167          1.0382            1.0342         3.77
IVF-Binary-512-nl223-pca (self)                          347.36       285.62       632.98       0.4060          1.0597            1.0583         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                412.72        65.81       478.53       0.1734          1.2924            1.2651         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                412.72        66.74       479.46       0.1734          1.2925            1.2652         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                412.72        70.22       482.94       0.1734          1.2925            1.2652         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               412.72       120.24       532.97       0.4092          1.0604            1.0564         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               412.72       167.69       580.41       0.5223          1.0373            1.0334         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               412.72       117.20       529.93       0.4091          1.0604            1.0564         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               412.72       167.59       580.31       0.5223          1.0373            1.0334         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               412.72       121.32       534.04       0.4091          1.0604            1.0564         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               412.72       172.00       584.72       0.5223          1.0373            1.0334         3.86
IVF-Binary-512-nl316-pca (self)                          412.72       293.97       706.69       0.4117          1.0584            1.0567         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)             332.17        89.73       421.91       0.1942          1.2715            1.2650         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)            332.17        95.16       427.33       0.1942          1.2715            1.2650         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)            332.17       100.01       432.19       0.1942          1.2715            1.2650         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)            332.17       156.91       489.09       0.4245          1.0541            1.0543         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)            332.17       207.69       539.87       0.5468          1.0322            1.0302         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)           332.17       156.07       488.24       0.4245          1.0541            1.0543         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)           332.17       212.19       544.36       0.5468          1.0322            1.0302         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)           332.17       166.54       498.72       0.4245          1.0541            1.0543         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)           332.17       216.05       548.22       0.5468          1.0322            1.0302         7.26
IVF-Binary-1024-nl158-random (self)                      332.17       426.59       758.76       0.4263          1.0539            1.0544         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            348.69        95.02       443.72       0.1973          1.2556            1.2486         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            348.69       100.00       448.69       0.1972          1.2558            1.2487         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            348.69       104.96       453.65       0.1972          1.2558            1.2487         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           348.69       159.65       508.34       0.4343          1.0516            1.0515         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           348.69       208.42       557.11       0.5563          1.0309            1.0289         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           348.69       192.04       540.73       0.4341          1.0517            1.0516         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           348.69       213.97       562.66       0.5561          1.0309            1.0289         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           348.69       167.57       516.26       0.4341          1.0517            1.0516         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           348.69       225.57       574.26       0.5561          1.0309            1.0289         7.32
IVF-Binary-1024-nl223-random (self)                      348.69       421.77       770.47       0.4353          1.0515            1.0518         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            403.86        96.74       500.59       0.1988          1.2510            1.2444         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)            403.86        99.93       503.79       0.1988          1.2512            1.2445         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)            403.86       104.20       508.06       0.1988          1.2512            1.2445         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)           403.86       154.30       558.16       0.4364          1.0511            1.0510         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)           403.86       206.25       610.10       0.5577          1.0307            1.0286         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)           403.86       153.34       557.20       0.4363          1.0511            1.0510         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)           403.86       207.56       611.41       0.5576          1.0307            1.0287         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)           403.86       160.46       564.31       0.4363          1.0511            1.0510         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)           403.86       214.00       617.86       0.5576          1.0307            1.0287         7.42
IVF-Binary-1024-nl316-random (self)                      403.86       417.33       821.19       0.4380          1.0509            1.0513         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)                365.77        89.61       455.37       0.1934          1.2687            1.2608         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)               365.77        93.27       459.03       0.1934          1.2687            1.2608         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)               365.77        97.04       462.81       0.1934          1.2687            1.2608         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)               365.77       147.92       513.68       0.4258          1.0537            1.0535         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)               365.77       201.58       567.35       0.5482          1.0320            1.0299         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)              365.77       154.60       520.37       0.4258          1.0537            1.0535         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)              365.77       207.83       573.60       0.5482          1.0320            1.0299         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)              365.77       154.33       520.09       0.4258          1.0537            1.0535         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)              365.77       209.85       575.62       0.5482          1.0320            1.0299         7.26
IVF-Binary-1024-nl158-pca (self)                         365.77       408.83       774.60       0.4266          1.0537            1.0540         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               371.87        93.43       465.30       0.1974          1.2517            1.2440         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               371.87        96.31       468.18       0.1974          1.2518            1.2440         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               371.87       108.55       480.42       0.1974          1.2518            1.2440         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              371.87       150.33       522.20       0.4357          1.0510            1.0507         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              371.87       201.01       572.88       0.5589          1.0305            1.0281         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              371.87       151.18       523.05       0.4356          1.0510            1.0507         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              371.87       205.12       576.99       0.5587          1.0305            1.0282         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              371.87       158.63       530.50       0.4356          1.0510            1.0507         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              371.87       209.58       581.45       0.5587          1.0305            1.0282         7.32
IVF-Binary-1024-nl223-pca (self)                         371.87       409.73       781.60       0.4366          1.0510            1.0512         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               406.25        96.64       502.89       0.1988          1.2476            1.2400         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)               406.25        98.63       504.88       0.1988          1.2477            1.2401         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)               406.25       104.71       510.96       0.1988          1.2477            1.2401         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)              406.25       154.61       560.86       0.4389          1.0502            1.0501         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)              406.25       204.29       610.55       0.5614          1.0302            1.0279         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)              406.25       152.88       559.13       0.4389          1.0503            1.0501         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)              406.25       207.07       613.32       0.5613          1.0302            1.0279         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)              406.25       158.76       565.01       0.4389          1.0503            1.0501         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)              406.25       211.23       617.49       0.5613          1.0302            1.0279         7.42
IVF-Binary-1024-nl316-pca (self)                         406.25       415.28       821.53       0.4398          1.0503            1.0504         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)                225.16       144.60       369.76       0.1213          1.4946            1.4409         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               225.16       146.14       371.30       0.1213          1.4946            1.4409         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               225.16       151.70       376.86       0.1213          1.4946            1.4409         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               225.16       181.81       406.97       0.3330          1.1010            1.0848         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               225.16       326.63       551.79       0.4412          1.0614            1.0485         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              225.16       181.54       406.70       0.3330          1.1010            1.0848         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              225.16       327.85       553.01       0.4412          1.0614            1.0485         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              225.16       185.46       410.62       0.3330          1.1010            1.0848         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              225.16       334.53       559.69       0.4412          1.0614            1.0485         1.68
IVF-Binary-256-nl158-sign (self)                         225.16       490.84       716.00       0.3381          1.0958            1.0832         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               232.70       144.30       377.00       0.1226          1.4837            1.4298         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               232.70       154.06       386.76       0.1226          1.4850            1.4303         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               232.70       150.59       383.29       0.1225          1.4851            1.4303         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              232.70       182.40       415.10       0.3562          1.0888            1.0753         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              232.70       326.34       559.04       0.4587          1.0559            1.0444         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              232.70       186.57       419.27       0.3560          1.0890            1.0754         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              232.70       329.29       561.99       0.4584          1.0560            1.0444         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              232.70       187.07       419.77       0.3560          1.0890            1.0754         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              232.70       330.30       563.00       0.4584          1.0560            1.0444         1.75
IVF-Binary-256-nl223-sign (self)                         232.70       493.71       726.41       0.3612          1.0839            1.0738         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               298.72       147.86       446.58       0.1234          1.4702            1.4163         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               298.72       148.07       446.79       0.1234          1.4713            1.4163         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               298.72       151.06       449.78       0.1234          1.4715            1.4163         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              298.72       188.62       487.34       0.3598          1.0873            1.0740         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              298.72       337.73       636.45       0.4585          1.0560            1.0445         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              298.72       186.16       484.88       0.3597          1.0875            1.0741         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              298.72       339.52       638.24       0.4582          1.0561            1.0445         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              298.72       190.54       489.26       0.3597          1.0875            1.0741         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              298.72       336.06       634.78       0.4582          1.0561            1.0445         1.84
IVF-Binary-256-nl316-sign (self)                         298.72       506.63       805.34       0.3649          1.0824            1.0722         1.84
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
Exhaustive (query)                                        72.65     1_379.85     1_452.50       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         72.65     4_564.33     4_636.98       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                147.60       270.56       418.16       0.1109          1.3512            1.3092         2.03
ExhaustiveBinary-256-random-rf10 (query)                 147.60       389.88       537.48       0.3143          1.0825            1.0613         2.03
ExhaustiveBinary-256-random-rf20 (query)                 147.60       515.92       663.52       0.4100          1.0523            1.0368         2.03
ExhaustiveBinary-256-random (self)                       147.60     1_287.21     1_434.82       0.3161          1.0784            1.0600         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   247.90       269.10       517.00       0.1167          1.3480            1.2981         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    247.90       400.77       648.67       0.3159          1.0791            1.0596         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    247.90       517.08       764.98       0.4121          1.0505            1.0362         2.03
ExhaustiveBinary-256-pca (self)                          247.90     1_225.36     1_473.26       0.3171          1.0782            1.0588         2.03
ExhaustiveBinary-512-random_no_rr (query)                216.96       388.31       605.27       0.1528          1.2601            1.2299         4.05
ExhaustiveBinary-512-random-rf10 (query)                 216.96       521.46       738.42       0.3465          1.0565            1.0514         4.05
ExhaustiveBinary-512-random-rf20 (query)                 216.96       652.80       869.76       0.4452          1.0358            1.0312         4.05
ExhaustiveBinary-512-random (self)                       216.96     1_727.92     1_944.89       0.3476          1.0547            1.0512         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   300.16       386.04       686.20       0.1558          1.2535            1.2254         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    300.16       532.26       832.41       0.3512          1.0523            1.0507         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    300.16       662.35       962.50       0.4484          1.0329            1.0309         4.05
ExhaustiveBinary-512-pca (self)                          300.16     1_679.84     1_980.00       0.3515          1.0522            1.0505         4.05
ExhaustiveBinary-1024-random_no_rr (query)               262.13       590.29       852.41       0.1816          1.2043            1.1936         8.11
ExhaustiveBinary-1024-random-rf10 (query)                262.13       739.55     1_001.67       0.3747          1.0447            1.0452         8.11
ExhaustiveBinary-1024-random-rf20 (query)                262.13       896.32     1_158.45       0.4789          1.0282            1.0270         8.11
ExhaustiveBinary-1024-random (self)                      262.13     2_438.38     2_700.51       0.3754          1.0447            1.0451         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  356.34       589.03       945.36       0.1832          1.2013            1.1905         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   356.34       760.66     1_117.00       0.3798          1.0434            1.0443         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   356.34       920.97     1_277.31       0.4867          1.0272            1.0261         8.11
ExhaustiveBinary-1024-pca (self)                         356.34     2_435.05     2_791.39       0.3787          1.0436            1.0444         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   87.58       697.61       785.18       0.1518          1.2701            1.2528         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    87.58       962.28     1_049.85       0.3399          1.0607            1.0535         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    87.58     1_244.10     1_331.68       0.4406          1.0369            1.0319         3.05
ExhaustiveBinary-512-sign (self)                          87.58     2_496.79     2_584.37       0.3409          1.0595            1.0531         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)              436.73        87.58       524.31       0.1137          1.3389            1.3036         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)             436.73       107.91       544.64       0.1137          1.3389            1.3036         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)             436.73        93.06       529.79       0.1137          1.3389            1.3036         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)             436.73       181.70       618.43       0.3176          1.0801            1.0600         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)             436.73       251.99       688.72       0.4134          1.0507            1.0360         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)            436.73       164.43       601.16       0.3176          1.0801            1.0600         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)            436.73       273.20       709.93       0.4134          1.0507            1.0360         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)            436.73       157.41       594.14       0.3176          1.0801            1.0600         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)            436.73       261.44       698.17       0.4134          1.0507            1.0360         2.34
IVF-Binary-256-nl158-random (self)                       436.73       300.94       737.67       0.3195          1.0756            1.0588         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)             433.19        74.33       507.52       0.1323          1.2781            1.2367         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)             433.19        77.01       510.19       0.1322          1.2784            1.2369         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)             433.19        79.26       512.44       0.1322          1.2785            1.2369         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)            433.19       164.31       597.49       0.3685          1.0557            1.0457         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)            433.19       253.25       686.43       0.4682          1.0358            1.0279         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)            433.19       165.64       598.82       0.3683          1.0558            1.0457         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)            433.19       255.68       688.86       0.4678          1.0359            1.0280         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)            433.19       164.50       597.68       0.3683          1.0558            1.0457         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)            433.19       256.43       689.62       0.4678          1.0359            1.0280         2.47
IVF-Binary-256-nl223-random (self)                       433.19       321.37       754.55       0.3700          1.0518            1.0451         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)             524.67        81.16       605.83       0.1436          1.2537            1.2119         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)             524.67        82.70       607.36       0.1436          1.2539            1.2120         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)             524.67        84.14       608.80       0.1435          1.2545            1.2124         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)            524.67       167.81       692.47       0.3831          1.0510            1.0421         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)            524.67       258.20       782.87       0.4824          1.0341            1.0261         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)            524.67       169.03       693.70       0.3828          1.0511            1.0421         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)            524.67       263.02       787.69       0.4817          1.0343            1.0262         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)            524.67       170.26       694.92       0.3827          1.0511            1.0422         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)            524.67       267.80       792.47       0.4816          1.0343            1.0262         2.65
IVF-Binary-256-nl316-random (self)                       524.67       346.48       871.14       0.3847          1.0473            1.0415         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)                 516.54        69.38       585.92       0.1193          1.3358            1.2928         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)                516.54        69.72       586.25       0.1193          1.3358            1.2928         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)                516.54        71.29       587.83       0.1193          1.3358            1.2928         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)                516.54       154.05       670.59       0.3193          1.0758            1.0585         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)                516.54       245.03       761.57       0.4156          1.0484            1.0355         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)               516.54       153.40       669.93       0.3193          1.0758            1.0585         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)               516.54       253.57       770.11       0.4156          1.0484            1.0355         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)               516.54       154.61       671.15       0.3193          1.0758            1.0585         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)               516.54       280.17       796.70       0.4156          1.0484            1.0355         2.34
IVF-Binary-256-nl158-pca (self)                          516.54       300.55       817.09       0.3209          1.0742            1.0576         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)                543.10        72.55       615.65       0.1362          1.2797            1.2334         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)                543.10        79.63       622.73       0.1361          1.2801            1.2335         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)                543.10        77.36       620.46       0.1361          1.2801            1.2335         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)               543.10       167.30       710.40       0.3632          1.0554            1.0461         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)               543.10       262.26       805.36       0.4637          1.0358            1.0283         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)               543.10       165.95       709.05       0.3631          1.0555            1.0461         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)               543.10       253.22       796.32       0.4635          1.0358            1.0283         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)               543.10       169.64       712.74       0.3630          1.0555            1.0461         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)               543.10       257.15       800.25       0.4635          1.0358            1.0283         2.47
IVF-Binary-256-nl223-pca (self)                          543.10       327.96       871.06       0.3655          1.0537            1.0454         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)                599.87        78.82       678.68       0.1455          1.2599            1.2105         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)                599.87        79.71       679.57       0.1454          1.2600            1.2106         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)                599.87        85.44       685.30       0.1454          1.2604            1.2107         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)               599.87       174.20       774.06       0.3774          1.0508            1.0427         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)               599.87       259.79       859.66       0.4767          1.0333            1.0267         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)               599.87       166.74       766.60       0.3772          1.0509            1.0427         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)               599.87       262.72       862.58       0.4762          1.0334            1.0268         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)               599.87       181.13       781.00       0.3771          1.0509            1.0427         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)               599.87       264.48       864.34       0.4761          1.0334            1.0268         2.65
IVF-Binary-256-nl316-pca (self)                          599.87       363.88       963.75       0.3791          1.0493            1.0422         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)              501.13        93.50       594.64       0.1542          1.2560            1.2276         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)             501.13        98.09       599.22       0.1542          1.2560            1.2276         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)             501.13       102.39       603.52       0.1542          1.2560            1.2276         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)             501.13       208.04       709.18       0.3476          1.0561            1.0510         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)             501.13       277.42       778.55       0.4464          1.0356            1.0310         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)            501.13       189.60       690.73       0.3476          1.0561            1.0510         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)            501.13       290.18       791.31       0.4464          1.0356            1.0310         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)            501.13       192.61       693.75       0.3476          1.0561            1.0510         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)            501.13       318.89       820.02       0.4464          1.0356            1.0310         4.36
IVF-Binary-512-nl158-random (self)                       501.13       437.07       938.20       0.3486          1.0543            1.0508         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)             514.41       106.39       620.80       0.1643          1.2246            1.1980         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)             514.41       104.91       619.33       0.1642          1.2249            1.1983         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)             514.41       107.85       622.26       0.1642          1.2249            1.1983         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)            514.41       203.60       718.01       0.3692          1.0480            1.0450         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)            514.41       287.69       802.10       0.4686          1.0313            1.0278         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)            514.41       194.98       709.39       0.3689          1.0481            1.0451         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)            514.41       295.59       810.00       0.4680          1.0314            1.0279         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)            514.41       202.23       716.65       0.3689          1.0481            1.0451         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)            514.41       295.13       809.54       0.4680          1.0314            1.0279         4.49
IVF-Binary-512-nl223-random (self)                       514.41       454.29       968.70       0.3698          1.0470            1.0451         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)             607.93       113.17       721.10       0.1680          1.2145            1.1880         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)             607.93       107.25       715.19       0.1679          1.2148            1.1884         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)             607.93       112.08       720.01       0.1679          1.2152            1.1886         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)            607.93       200.19       808.12       0.3751          1.0467            1.0439         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)            607.93       291.48       899.41       0.4752          1.0304            1.0271         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)            607.93       202.71       810.64       0.3748          1.0468            1.0440         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)            607.93       290.55       898.48       0.4744          1.0305            1.0272         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)            607.93       201.98       809.91       0.3748          1.0468            1.0440         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)            607.93       304.63       912.56       0.4742          1.0305            1.0273         4.67
IVF-Binary-512-nl316-random (self)                       607.93       479.46     1_087.40       0.3755          1.0457            1.0439         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)                 601.85        95.04       696.89       0.1572          1.2489            1.2229         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)                601.85        98.57       700.42       0.1572          1.2489            1.2229         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)                601.85       101.01       702.86       0.1572          1.2489            1.2229         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)                601.85       190.12       791.98       0.3528          1.0515            1.0501         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)                601.85       287.61       889.47       0.4499          1.0326            1.0306         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)               601.85       185.69       787.55       0.3528          1.0515            1.0501         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)               601.85       285.19       887.05       0.4499          1.0326            1.0306         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)               601.85       188.44       790.29       0.3528          1.0515            1.0501         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)               601.85       285.44       887.29       0.4499          1.0326            1.0306         4.36
IVF-Binary-512-nl158-pca (self)                          601.85       438.88     1_040.73       0.3532          1.0513            1.0499         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)                591.89       100.43       692.32       0.1671          1.2191            1.1940         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)                591.89       103.54       695.43       0.1670          1.2194            1.1943         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)                591.89       115.83       707.72       0.1670          1.2194            1.1943         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)               591.89       191.04       782.93       0.3707          1.0459            1.0453         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)               591.89       293.52       885.41       0.4720          1.0291            1.0275         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)               591.89       198.05       789.94       0.3704          1.0460            1.0453         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)               591.89       284.86       876.75       0.4714          1.0292            1.0276         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)               591.89       207.18       799.07       0.3704          1.0460            1.0453         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)               591.89       296.06       887.95       0.4714          1.0292            1.0276         4.49
IVF-Binary-512-nl223-pca (self)                          591.89       454.67     1_046.56       0.3709          1.0458            1.0452         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)                692.65       108.64       801.30       0.1707          1.2099            1.1841         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)                692.65       106.85       799.50       0.1706          1.2101            1.1844         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)                692.65       121.45       814.10       0.1706          1.2102            1.1846         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)               692.65       200.58       893.23       0.3771          1.0444            1.0436         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)               692.65       295.47       988.12       0.4785          1.0284            1.0267         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)               692.65       200.59       893.24       0.3768          1.0445            1.0437         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)               692.65       292.89       985.54       0.4777          1.0285            1.0267         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)               692.65       208.77       901.42       0.3767          1.0445            1.0437         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)               692.65       300.06       992.71       0.4775          1.0286            1.0268         4.67
IVF-Binary-512-nl316-pca (self)                          692.65       477.32     1_169.97       0.3771          1.0445            1.0438         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)             565.27       149.17       714.43       0.1822          1.2030            1.1925         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)            565.27       150.93       716.20       0.1822          1.2030            1.1925         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)            565.27       153.21       718.48       0.1822          1.2030            1.1925         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)            565.27       240.46       805.73       0.3753          1.0446            1.0450         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)            565.27       349.09       914.35       0.4798          1.0281            1.0269         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)           565.27       248.86       814.12       0.3753          1.0446            1.0450         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)           565.27       358.07       923.34       0.4798          1.0281            1.0269         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)           565.27       276.91       842.18       0.3753          1.0446            1.0450         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)           565.27       364.83       930.10       0.4798          1.0281            1.0269         8.42
IVF-Binary-1024-nl158-random (self)                      565.27       670.62     1_235.89       0.3761          1.0445            1.0449         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)            561.86       161.69       723.55       0.1854          1.1897            1.1801         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)            561.86       161.53       723.39       0.1854          1.1899            1.1803         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)            561.86       168.50       730.36       0.1854          1.1899            1.1803         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)           561.86       259.73       821.59       0.3860          1.0419            1.0423         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)           561.86       346.33       908.20       0.4927          1.0265            1.0252         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)           561.86       256.81       818.67       0.3858          1.0419            1.0423         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)           561.86       364.66       926.52       0.4922          1.0265            1.0252         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)           561.86       259.24       821.10       0.3858          1.0419            1.0423         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)           561.86       369.60       931.46       0.4922          1.0265            1.0252         8.54
IVF-Binary-1024-nl223-random (self)                      561.86       680.63     1_242.49       0.3870          1.0419            1.0423         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)            671.55       164.40       835.95       0.1868          1.1852            1.1758         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)            671.55       171.24       842.79       0.1868          1.1855            1.1761         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)            671.55       171.63       843.18       0.1868          1.1856            1.1762         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)           671.55       263.45       935.00       0.3893          1.0414            1.0415         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)           671.55       372.30     1_043.85       0.4955          1.0262            1.0249         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)           671.55       264.98       936.53       0.3889          1.0415            1.0416         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)           671.55       367.09     1_038.64       0.4949          1.0263            1.0249         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)           671.55       265.17       936.72       0.3888          1.0415            1.0416         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)           671.55       366.20     1_037.75       0.4948          1.0263            1.0250         8.73
IVF-Binary-1024-nl316-random (self)                      671.55       694.61     1_366.16       0.3897          1.0414            1.0417         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)                672.98       169.78       842.76       0.1838          1.2000            1.1896         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)               672.98       152.44       825.43       0.1838          1.2000            1.1896         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)               672.98       157.09       830.07       0.1838          1.2000            1.1896         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)               672.98       249.13       922.11       0.3805          1.0433            1.0441         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)               672.98       340.56     1_013.54       0.4876          1.0270            1.0260         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)              672.98       253.16       926.15       0.3805          1.0433            1.0441         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)              672.98       351.96     1_024.94       0.4876          1.0270            1.0260         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)              672.98       254.69       927.67       0.3805          1.0433            1.0441         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)              672.98       362.93     1_035.92       0.4876          1.0270            1.0260         8.42
IVF-Binary-1024-nl158-pca (self)                         672.98       662.77     1_335.75       0.3794          1.0434            1.0442         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)               676.71       160.51       837.22       0.1870          1.1877            1.1785         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)               676.71       159.51       836.22       0.1870          1.1880            1.1787         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)               676.71       164.29       841.00       0.1870          1.1880            1.1787         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)              676.71       250.52       927.23       0.3905          1.0408            1.0416         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)              676.71       353.82     1_030.53       0.4989          1.0255            1.0246         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)              676.71       250.25       926.97       0.3902          1.0408            1.0417         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)              676.71       353.14     1_029.85       0.4983          1.0256            1.0247         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)              676.71       259.04       935.75       0.3902          1.0408            1.0417         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)              676.71       368.44     1_045.15       0.4983          1.0256            1.0247         8.54
IVF-Binary-1024-nl223-pca (self)                         676.71       692.53     1_369.24       0.3897          1.0410            1.0419         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)               829.14       170.88     1_000.02       0.1882          1.1834            1.1741         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)               829.14       175.20     1_004.33       0.1882          1.1836            1.1744         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)               829.14       168.56       997.70       0.1881          1.1837            1.1745         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)              829.14       256.75     1_085.89       0.3940          1.0401            1.0408         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)              829.14       357.16     1_186.30       0.5030          1.0252            1.0242         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)              829.14       262.31     1_091.45       0.3937          1.0402            1.0410         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)              829.14       358.34     1_187.48       0.5024          1.0252            1.0243         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)              829.14       266.27     1_095.41       0.3936          1.0403            1.0410         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)              829.14       368.31     1_197.45       0.5022          1.0253            1.0243         8.73
IVF-Binary-1024-nl316-pca (self)                         829.14       705.64     1_534.78       0.3929          1.0404            1.0411         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)                381.97       296.83       678.80       0.1519          1.2699            1.2515         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)               381.97       293.70       675.67       0.1519          1.2699            1.2515         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)               381.97       299.11       681.08       0.1519          1.2699            1.2515         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)               381.97       364.84       746.81       0.3414          1.0596            1.0531         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)               381.97       660.87     1_042.84       0.4416          1.0366            1.0317         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)              381.97       362.00       743.97       0.3414          1.0596            1.0531         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)              381.97       682.97     1_064.94       0.4416          1.0366            1.0317         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)              381.97       367.50       749.47       0.3414          1.0596            1.0531         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)              381.97       656.30     1_038.26       0.4416          1.0366            1.0317         3.36
IVF-Binary-512-nl158-sign (self)                         381.97       991.58     1_373.55       0.3425          1.0582            1.0527         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               388.36       291.67       680.03       0.1529          1.2635            1.2471         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               388.36       295.61       683.97       0.1528          1.2643            1.2484         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               388.36       304.27       692.63       0.1528          1.2643            1.2484         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              388.36       377.74       766.10       0.3504          1.0559            1.0502         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              388.36       651.87     1_040.23       0.4492          1.0348            1.0305         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              388.36       370.68       759.04       0.3502          1.0560            1.0502         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              388.36       658.87     1_047.23       0.4486          1.0350            1.0306         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              388.36       374.40       762.76       0.3502          1.0560            1.0502         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              388.36       668.32     1_056.68       0.4486          1.0350            1.0306         3.49
IVF-Binary-512-nl223-sign (self)                         388.36     1_034.71     1_423.07       0.3506          1.0551            1.0500         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)               621.71       307.95       929.66       0.1539          1.2662            1.2437         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)               621.71       300.10       921.81       0.1538          1.2675            1.2453         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)               621.71       299.52       921.23       0.1537          1.2681            1.2458         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)              621.71       388.95     1_010.65       0.3536          1.0549            1.0493         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)              621.71       655.20     1_276.91       0.4499          1.0345            1.0304         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)              621.71       392.99     1_014.69       0.3531          1.0551            1.0494         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)              621.71       656.81     1_278.52       0.4490          1.0347            1.0306         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)              621.71       374.95       996.66       0.3530          1.0551            1.0495         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)              621.71       668.00     1_289.71       0.4487          1.0348            1.0306         3.67
IVF-Binary-512-nl316-sign (self)                         621.71     1_029.25     1_650.96       0.3538          1.0540            1.0492         3.67
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
Exhaustive (query)                                       102.42     1_966.54     2_068.96       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        102.42     6_567.61     6_670.03       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                203.64       284.44       488.07       0.1140          1.2809            1.2433         2.28
ExhaustiveBinary-256-random-rf10 (query)                 203.64       451.31       654.94       0.3148          1.0656            1.0476         2.28
ExhaustiveBinary-256-random-rf20 (query)                 203.64       578.46       782.09       0.4075          1.0420            1.0293         2.28
ExhaustiveBinary-256-random (self)                       203.64     1_321.36     1_524.99       0.3168          1.0618            1.0471         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   403.05       283.68       686.73       0.1054          1.3026            1.2617         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    403.05       426.72       829.76       0.3012          1.0735            1.0517         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    403.05       564.76       967.80       0.3931          1.0471            1.0315         2.28
ExhaustiveBinary-256-pca (self)                          403.05     1_314.32     1_717.36       0.3048          1.0710            1.0504         2.28
ExhaustiveBinary-512-random_no_rr (query)                306.65       417.20       723.86       0.1506          1.2094            1.1809         4.55
ExhaustiveBinary-512-random-rf10 (query)                 306.65       575.92       882.57       0.3395          1.0453            1.0426         4.55
ExhaustiveBinary-512-random-rf20 (query)                 306.65       730.59     1_037.25       0.4326          1.0293            1.0264         4.55
ExhaustiveBinary-512-random (self)                       306.65     1_838.78     2_145.43       0.3401          1.0435            1.0423         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   498.69       426.06       924.75       0.1459          1.2162            1.1914         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    498.69       578.25     1_076.93       0.3341          1.0468            1.0435         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    498.69       723.84     1_222.53       0.4278          1.0295            1.0269         4.55
ExhaustiveBinary-512-pca (self)                          498.69     1_831.24     2_329.93       0.3355          1.0454            1.0433         4.55
ExhaustiveBinary-1024-random_no_rr (query)               505.46       641.41     1_146.87       0.1761          1.1673            1.1571         9.11
ExhaustiveBinary-1024-random-rf10 (query)                505.46       810.59     1_316.05       0.3603          1.0383            1.0383         9.11
ExhaustiveBinary-1024-random-rf20 (query)                505.46       998.17     1_503.63       0.4618          1.0244            1.0230         9.11
ExhaustiveBinary-1024-random (self)                      505.46     2_655.85     3_161.31       0.3602          1.0377            1.0383         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  713.87       629.66     1_343.53       0.1756          1.1686            1.1586         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   713.87       804.32     1_518.19       0.3594          1.0382            1.0385         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   713.87       978.39     1_692.26       0.4576          1.0246            1.0237         9.11
ExhaustiveBinary-1024-pca (self)                         713.87     2_637.09     3_350.96       0.3584          1.0383            1.0389         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  132.39       851.60       984.00       0.1691          1.1871            1.1718         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   132.39       986.76     1_119.15       0.3431          1.0433            1.0415         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   132.39     1_450.22     1_582.62       0.4437          1.0266            1.0250         4.58
ExhaustiveBinary-768-sign (self)                         132.39     3_044.45     3_176.84       0.3438          1.0424            1.0413         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)              583.40        96.67       680.07       0.1164          1.2725            1.2403         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)             583.40        98.12       681.52       0.1164          1.2725            1.2403         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)             583.40       101.78       685.18       0.1164          1.2725            1.2403         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)             583.40       200.04       783.44       0.3172          1.0636            1.0468         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)             583.40       321.67       905.07       0.4095          1.0414            1.0290         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)            583.40       197.69       781.09       0.3172          1.0636            1.0468         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)            583.40       315.30       898.70       0.4095          1.0414            1.0290         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)            583.40       199.74       783.14       0.3172          1.0636            1.0468         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)            583.40       315.88       899.28       0.4095          1.0414            1.0290         2.74
IVF-Binary-256-nl158-random (self)                       583.40       414.12       997.52       0.3191          1.0600            1.0465         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)             603.21        94.58       697.79       0.1331          1.2303            1.1909         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)             603.21        95.28       698.49       0.1331          1.2303            1.1909         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)             603.21        97.57       700.78       0.1331          1.2303            1.1909         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)            603.21       209.66       812.88       0.3574          1.0474            1.0376         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)            603.21       329.47       932.68       0.4568          1.0309            1.0232         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)            603.21       205.68       808.89       0.3573          1.0474            1.0376         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)            603.21       326.42       929.63       0.4568          1.0309            1.0232         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)            603.21       209.41       812.62       0.3573          1.0474            1.0376         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)            603.21       342.53       945.74       0.4568          1.0309            1.0232         2.93
IVF-Binary-256-nl223-random (self)                       603.21       443.81     1_047.02       0.3590          1.0439            1.0372         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)             710.10       105.26       815.36       0.1402          1.2168            1.1747         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)             710.10       104.33       814.43       0.1402          1.2169            1.1747         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)             710.10       109.63       819.73       0.1402          1.2169            1.1747         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)            710.10       228.83       938.93       0.3660          1.0441            1.0360         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)            710.10       344.73     1_054.83       0.4635          1.0293            1.0224         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)            710.10       226.00       936.10       0.3660          1.0441            1.0360         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)            710.10       330.15     1_040.25       0.4634          1.0293            1.0224         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)            710.10       218.81       928.91       0.3660          1.0441            1.0360         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)            710.10       336.57     1_046.67       0.4634          1.0293            1.0224         3.21
IVF-Binary-256-nl316-random (self)                       710.10       502.55     1_212.65       0.3675          1.0408            1.0357         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)                 788.10        88.15       876.25       0.1076          1.2918            1.2570         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)                788.10        87.46       875.56       0.1076          1.2918            1.2570         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)                788.10        89.78       877.88       0.1076          1.2918            1.2570         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)                788.10       195.41       983.51       0.3045          1.0705            1.0511         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)                788.10       305.79     1_093.89       0.3973          1.0449            1.0309         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)               788.10       196.63       984.73       0.3045          1.0705            1.0511         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)               788.10       308.17     1_096.28       0.3973          1.0449            1.0309         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)               788.10       194.19       982.29       0.3045          1.0705            1.0511         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)               788.10       311.46     1_099.56       0.3973          1.0449            1.0309         2.74
IVF-Binary-256-nl158-pca (self)                          788.10       394.02     1_182.12       0.3083          1.0672            1.0497         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)                765.76        93.15       858.92       0.1268          1.2395            1.2037         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)                765.76        94.28       860.05       0.1268          1.2395            1.2037         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)                765.76       101.80       867.56       0.1268          1.2395            1.2037         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)               765.76       206.41       972.17       0.3613          1.0476            1.0372         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)               765.76       319.32     1_085.09       0.4626          1.0301            1.0228         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)               765.76       207.02       972.79       0.3613          1.0476            1.0372         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)               765.76       320.77     1_086.53       0.4626          1.0301            1.0228         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)               765.76       207.72       973.49       0.3613          1.0476            1.0372         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)               765.76       322.01     1_087.78       0.4626          1.0301            1.0228         2.93
IVF-Binary-256-nl223-pca (self)                          765.76       424.15     1_189.91       0.3645          1.0451            1.0365         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)                879.72       101.49       981.21       0.1365          1.2209            1.1850         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)                879.72       102.00       981.72       0.1365          1.2210            1.1850         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)                879.72       108.86       988.58       0.1365          1.2210            1.1850         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)               879.72       216.37     1_096.09       0.3737          1.0433            1.0349         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)               879.72       326.05     1_205.76       0.4727          1.0280            1.0216         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)               879.72       210.41     1_090.13       0.3737          1.0433            1.0349         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)               879.72       328.10     1_207.81       0.4727          1.0280            1.0216         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)               879.72       222.95     1_102.67       0.3737          1.0433            1.0349         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)               879.72       329.35     1_209.06       0.4727          1.0280            1.0216         3.21
IVF-Binary-256-nl316-pca (self)                          879.72       482.15     1_361.86       0.3769          1.0410            1.0341         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)              693.83       129.80       823.63       0.1520          1.2064            1.1790         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)             693.83       133.23       827.07       0.1520          1.2064            1.1790         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)             693.83       134.24       828.07       0.1520          1.2064            1.1790         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)             693.83       245.77       939.60       0.3401          1.0451            1.0423         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)             693.83       355.64     1_049.47       0.4336          1.0292            1.0262         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)            693.83       242.13       935.96       0.3401          1.0451            1.0423         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)            693.83       360.46     1_054.29       0.4336          1.0292            1.0262         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)            693.83       244.87       938.70       0.3401          1.0451            1.0423         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)            693.83       368.94     1_062.77       0.4336          1.0292            1.0262         5.02
IVF-Binary-512-nl158-random (self)                       693.83       592.43     1_286.26       0.3407          1.0433            1.0421         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)             669.77       134.38       804.15       0.1603          1.1838            1.1572         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)             669.77       146.14       815.91       0.1603          1.1838            1.1573         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)             669.77       140.92       810.69       0.1603          1.1838            1.1573         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)            669.77       252.28       922.05       0.3579          1.0405            1.0379         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)            669.77       364.26     1_034.04       0.4565          1.0263            1.0235         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)            669.77       249.09       918.86       0.3579          1.0405            1.0379         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)            669.77       380.81     1_050.58       0.4565          1.0263            1.0235         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)            669.77       253.09       922.86       0.3579          1.0405            1.0379         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)            669.77       379.49     1_049.26       0.4565          1.0263            1.0235         5.21
IVF-Binary-512-nl223-random (self)                       669.77       619.69     1_289.47       0.3585          1.0390            1.0378         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)             787.50       144.02       931.53       0.1626          1.1785            1.1525         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)             787.50       145.15       932.65       0.1625          1.1787            1.1525         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)             787.50       154.14       941.64       0.1625          1.1787            1.1525         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)            787.50       256.16     1_043.66       0.3621          1.0391            1.0372         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)            787.50       389.61     1_177.12       0.4589          1.0256            1.0234         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)            787.50       253.64     1_041.14       0.3621          1.0391            1.0372         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)            787.50       381.90     1_169.40       0.4588          1.0256            1.0234         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)            787.50       264.75     1_052.25       0.3621          1.0391            1.0372         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)            787.50       393.08     1_180.58       0.4588          1.0256            1.0234         5.48
IVF-Binary-512-nl316-random (self)                       787.50       645.36     1_432.86       0.3624          1.0378            1.0372         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)                 908.19       124.61     1_032.80       0.1473          1.2122            1.1891         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)                908.19       126.91     1_035.10       0.1473          1.2122            1.1891         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)                908.19       129.74     1_037.94       0.1473          1.2122            1.1891         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)                908.19       243.06     1_151.25       0.3358          1.0460            1.0431         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)                908.19       373.13     1_281.32       0.4296          1.0292            1.0267         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)               908.19       250.61     1_158.80       0.3358          1.0460            1.0431         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)               908.19       355.10     1_263.29       0.4296          1.0292            1.0267         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)               908.19       244.81     1_153.00       0.3358          1.0460            1.0431         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)               908.19       360.22     1_268.41       0.4296          1.0292            1.0267         5.02
IVF-Binary-512-nl158-pca (self)                          908.19       580.43     1_488.62       0.3372          1.0445            1.0428         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)                865.70       133.72       999.41       0.1587          1.1837            1.1593         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)                865.70       131.55       997.25       0.1587          1.1837            1.1593         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)                865.70       136.80     1_002.50       0.1587          1.1837            1.1593         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)               865.70       247.73     1_113.42       0.3586          1.0397            1.0379         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)               865.70       368.33     1_234.02       0.4554          1.0259            1.0235         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)               865.70       249.49     1_115.19       0.3586          1.0397            1.0379         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)               865.70       408.57     1_274.27       0.4554          1.0259            1.0235         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)               865.70       248.98     1_114.68       0.3586          1.0397            1.0379         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)               865.70       379.35     1_245.05       0.4554          1.0259            1.0235         5.21
IVF-Binary-512-nl223-pca (self)                          865.70       610.81     1_476.50       0.3592          1.0390            1.0377         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)                986.32       141.47     1_127.80       0.1625          1.1766            1.1521         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)                986.32       141.31     1_127.64       0.1624          1.1767            1.1522         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)                986.32       146.53     1_132.85       0.1624          1.1767            1.1522         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)               986.32       253.57     1_239.89       0.3632          1.0386            1.0370         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)               986.32       394.06     1_380.38       0.4592          1.0254            1.0230         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)               986.32       250.58     1_236.90       0.3632          1.0386            1.0370         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)               986.32       379.05     1_365.38       0.4591          1.0254            1.0230         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)               986.32       256.24     1_242.56       0.3632          1.0386            1.0370         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)               986.32       388.39     1_374.71       0.4591          1.0254            1.0230         5.48
IVF-Binary-512-nl316-pca (self)                          986.32       635.91     1_622.24       0.3640          1.0380            1.0369         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)             916.70       199.42     1_116.12       0.1767          1.1663            1.1563         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)            916.70       204.64     1_121.34       0.1767          1.1663            1.1563         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)            916.70       204.40     1_121.10       0.1767          1.1663            1.1563         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)            916.70       315.66     1_232.36       0.3608          1.0382            1.0382         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)            916.70       444.67     1_361.37       0.4625          1.0243            1.0230         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)           916.70       321.10     1_237.81       0.3608          1.0382            1.0382         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)           916.70       458.37     1_375.07       0.4625          1.0243            1.0230         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)           916.70       323.75     1_240.45       0.3608          1.0382            1.0382         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)           916.70       459.00     1_375.70       0.4625          1.0243            1.0230         9.57
IVF-Binary-1024-nl158-random (self)                      916.70       868.54     1_785.24       0.3607          1.0376            1.0382         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)            864.70       219.25     1_083.95       0.1797          1.1568            1.1474         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)            864.70       211.72     1_076.41       0.1797          1.1568            1.1474         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)            864.70       213.59     1_078.29       0.1797          1.1568            1.1474         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)           864.70       331.70     1_196.40       0.3720          1.0360            1.0359         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)           864.70       487.21     1_351.91       0.4751          1.0229            1.0217         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)           864.70       332.03     1_196.73       0.3720          1.0360            1.0359         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)           864.70       467.48     1_332.18       0.4751          1.0229            1.0217         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)           864.70       346.63     1_211.32       0.3720          1.0360            1.0359         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)           864.70       476.13     1_340.82       0.4751          1.0229            1.0217         9.76
IVF-Binary-1024-nl223-random (self)                      864.70       898.82     1_763.52       0.3716          1.0354            1.0359         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)          1_003.12       245.17     1_248.28       0.1805          1.1543            1.1451        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)          1_003.12       218.18     1_221.30       0.1805          1.1544            1.1451        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)          1_003.12       223.74     1_226.86       0.1805          1.1544            1.1451        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)         1_003.12       349.34     1_352.45       0.3733          1.0356            1.0358        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)         1_003.12       483.38     1_486.50       0.4764          1.0227            1.0216        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)         1_003.12       345.12     1_348.24       0.3733          1.0356            1.0358        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)         1_003.12       483.57     1_486.69       0.4763          1.0227            1.0216        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)         1_003.12       356.27     1_359.39       0.3733          1.0356            1.0358        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)         1_003.12       488.16     1_491.28       0.4763          1.0227            1.0216        10.04
IVF-Binary-1024-nl316-random (self)                    1_003.12       947.58     1_950.70       0.3736          1.0350            1.0356        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)              1_101.44       200.43     1_301.87       0.1762          1.1673            1.1573         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)             1_101.44       202.35     1_303.80       0.1762          1.1673            1.1573         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)             1_101.44       204.30     1_305.74       0.1762          1.1673            1.1573         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)             1_101.44       323.38     1_424.82       0.3600          1.0380            1.0384         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)             1_101.44       448.57     1_550.01       0.4586          1.0245            1.0236         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)            1_101.44       325.72     1_427.16       0.3600          1.0380            1.0384         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)            1_101.44       458.95     1_560.39       0.4586          1.0245            1.0236         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)            1_101.44       333.76     1_435.21       0.3600          1.0380            1.0384         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)            1_101.44       464.57     1_566.02       0.4586          1.0245            1.0236         9.57
IVF-Binary-1024-nl158-pca (self)                       1_101.44       886.17     1_987.62       0.3591          1.0381            1.0388         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_075.63       213.87     1_289.50       0.1795          1.1565            1.1467         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_075.63       210.12     1_285.75       0.1795          1.1565            1.1467         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_075.63       220.77     1_296.39       0.1795          1.1565            1.1467         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_075.63       338.30     1_413.93       0.3711          1.0358            1.0360         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_075.63       462.13     1_537.76       0.4724          1.0231            1.0221         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_075.63       336.39     1_412.01       0.3711          1.0358            1.0360         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_075.63       469.11     1_544.74       0.4724          1.0231            1.0221         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_075.63       347.30     1_422.93       0.3711          1.0358            1.0360         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_075.63       492.74     1_568.37       0.4724          1.0231            1.0221         9.76
IVF-Binary-1024-nl223-pca (self)                       1_075.63       900.88     1_976.51       0.3704          1.0359            1.0363         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_203.58       224.71     1_428.29       0.1805          1.1540            1.1448        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_203.58       223.84     1_427.42       0.1805          1.1540            1.1448        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_203.58       228.83     1_432.41       0.1805          1.1540            1.1448        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_203.58       352.87     1_556.45       0.3737          1.0353            1.0357        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_203.58       477.94     1_681.52       0.4739          1.0228            1.0220        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_203.58       345.23     1_548.81       0.3737          1.0353            1.0357        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_203.58       476.99     1_680.57       0.4738          1.0228            1.0220        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_203.58       356.88     1_560.46       0.3737          1.0353            1.0357        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_203.58       487.95     1_691.53       0.4738          1.0228            1.0220        10.04
IVF-Binary-1024-nl316-pca (self)                       1_203.58       936.44     2_140.02       0.3726          1.0354            1.0359        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)                524.60       420.06       944.66       0.1693          1.1870            1.1720         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)               524.60       412.74       937.33       0.1693          1.1870            1.1720         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)               524.60       415.47       940.06       0.1693          1.1870            1.1720         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)               524.60       502.58     1_027.18       0.3434          1.0432            1.0414         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)               524.60       909.71     1_434.31       0.4438          1.0266            1.0249         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)              524.60       501.76     1_026.36       0.3434          1.0432            1.0414         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)              524.60       918.79     1_443.39       0.4438          1.0266            1.0249         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)              524.60       508.90     1_033.49       0.3434          1.0432            1.0414         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)              524.60       920.05     1_444.65       0.4438          1.0266            1.0249         5.04
IVF-Binary-768-nl158-sign (self)                         524.60     1_442.23     1_966.83       0.3440          1.0423            1.0413         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)               618.04       430.80     1_048.85       0.1692          1.1872            1.1717         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)               618.04       420.85     1_038.90       0.1692          1.1872            1.1717         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)               618.04       422.34     1_040.38       0.1692          1.1872            1.1717         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)              618.04       509.38     1_127.43       0.3492          1.0414            1.0401         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)              618.04       921.47     1_539.51       0.4483          1.0259            1.0245         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)              618.04       509.56     1_127.61       0.3492          1.0414            1.0401         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)              618.04       923.51     1_541.55       0.4483          1.0259            1.0245         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)              618.04       516.97     1_135.01       0.3492          1.0414            1.0401         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)              618.04       926.45     1_544.50       0.4483          1.0259            1.0245         5.23
IVF-Binary-768-nl223-sign (self)                         618.04     1_459.85     2_077.90       0.3496          1.0408            1.0400         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)               635.66       425.75     1_061.41       0.1694          1.1867            1.1714         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)               635.66       445.02     1_080.68       0.1694          1.1867            1.1714         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)               635.66       428.33     1_063.98       0.1694          1.1867            1.1714         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)              635.66       516.20     1_151.85       0.3501          1.0411            1.0399         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)              635.66       944.96     1_580.61       0.4490          1.0257            1.0244         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)              635.66       515.48     1_151.14       0.3501          1.0411            1.0399         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)              635.66       926.03     1_561.69       0.4489          1.0257            1.0244         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)              635.66       525.12     1_160.78       0.3501          1.0411            1.0399         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)              635.66       933.47     1_569.13       0.4489          1.0257            1.0244         5.51
IVF-Binary-768-nl316-sign (self)                         635.66     1_479.24     2_114.90       0.3504          1.0405            1.0398         5.51
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
Exhaustive (query)                                        32.36       706.42       738.78       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.36     2_326.41     2_358.78       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                 69.63       240.91       310.55       0.0970          1.6334            1.6378         1.78
ExhaustiveBinary-256-random-rf10 (query)                  69.63       340.66       410.30       0.3643          1.1391            1.1302         1.78
ExhaustiveBinary-256-random-rf20 (query)                  69.63       440.97       510.60       0.5087          1.0798            1.0701         1.78
ExhaustiveBinary-256-random (self)                        69.63     1_089.51     1_159.14       0.3862          1.1443            1.1409         1.78
ExhaustiveBinary-256-pca_no_rr (query)                    96.82       235.26       332.08       0.0922          1.6524            1.6606         1.78
ExhaustiveBinary-256-pca-rf10 (query)                     96.82       337.25       434.07       0.3517          1.1465            1.1384         1.78
ExhaustiveBinary-256-pca-rf20 (query)                     96.82       441.06       537.88       0.4943          1.0846            1.0744         1.78
ExhaustiveBinary-256-pca (self)                           96.82     1_088.84     1_185.66       0.3764          1.1502            1.1477         1.78
ExhaustiveBinary-512-random_no_rr (query)                 83.94       344.57       428.51       0.1464          1.5035            1.5085         3.55
ExhaustiveBinary-512-random-rf10 (query)                  83.94       453.92       537.86       0.4596          1.0936            1.0901         3.55
ExhaustiveBinary-512-random-rf20 (query)                  83.94       560.52       644.46       0.6085          1.0504            1.0459         3.55
ExhaustiveBinary-512-random (self)                        83.94     1_483.94     1_567.88       0.4800          1.0996            1.0995         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   109.62       345.65       455.28       0.1458          1.5041            1.5097         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    109.62       500.21       609.83       0.4543          1.0952            1.0911         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    109.62       560.10       669.72       0.6037          1.0513            1.0464         3.55
ExhaustiveBinary-512-pca (self)                          109.62     1_484.14     1_593.77       0.4777          1.1003            1.0997         3.55
ExhaustiveBinary-1024-random_no_rr (query)               114.69       508.76       623.45       0.2155          1.3655            1.3721         7.10
ExhaustiveBinary-1024-random-rf10 (query)                114.69       619.31       734.01       0.5869          1.0540            1.0515         7.10
ExhaustiveBinary-1024-random-rf20 (query)                114.69       738.06       852.75       0.7380          1.0260            1.0224         7.10
ExhaustiveBinary-1024-random (self)                      114.69     2_049.97     2_164.66       0.6118          1.0576            1.0546         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  141.55       508.10       649.64       0.2122          1.3735            1.3798         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   141.55       622.00       763.55       0.5776          1.0560            1.0532         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   141.55       731.89       873.44       0.7291          1.0273            1.0232         7.10
ExhaustiveBinary-1024-pca (self)                         141.55     2_065.13     2_206.68       0.6017          1.0602            1.0571         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   41.82       440.64       482.46       0.1044          1.6421            1.6499         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    41.82       476.34       518.15       0.3737          1.1368            1.1275         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    41.82       728.49       770.31       0.5265          1.0745            1.0646         1.53
ExhaustiveBinary-256-sign (self)                          41.82     1_546.03     1_587.85       0.3940          1.1439            1.1394         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)              227.32        47.94       275.26       0.1006          1.6235            1.6329         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)             227.32        50.11       277.43       0.1005          1.6238            1.6332         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)             227.32        54.28       281.60       0.1005          1.6238            1.6332         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)             227.32        96.84       324.16       0.3692          1.1376            1.1295         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)             227.32       146.08       373.39       0.5128          1.0789            1.0697         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)            227.32        95.41       322.72       0.3678          1.1379            1.1296         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)            227.32       149.81       377.13       0.5110          1.0792            1.0698         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)            227.32       105.43       332.75       0.3677          1.1379            1.1296         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)            227.32       154.61       381.93       0.5108          1.0792            1.0698         1.93
IVF-Binary-256-nl158-random (self)                       227.32       222.91       450.22       0.3897          1.1428            1.1402         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             265.38        45.34       310.72       0.1122          1.5873            1.5872         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             265.38        47.51       312.89       0.1121          1.5880            1.5877         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             265.38        51.79       317.17       0.1121          1.5881            1.5878         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            265.38       100.07       365.45       0.3938          1.1238            1.1155         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            265.38       149.11       414.49       0.5354          1.0713            1.0627         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            265.38        97.81       363.19       0.3933          1.1239            1.1157         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            265.38       150.24       415.63       0.5347          1.0714            1.0629         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            265.38       103.15       368.53       0.3931          1.1240            1.1157         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            265.38       154.45       419.83       0.5346          1.0714            1.0629         2.00
IVF-Binary-256-nl223-random (self)                       265.38       236.12       501.50       0.4141          1.1288            1.1266         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             340.98        48.39       389.38       0.1174          1.5678            1.5682         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             340.98        47.86       388.84       0.1173          1.5683            1.5687         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             340.98        52.43       393.42       0.1173          1.5685            1.5688         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            340.98       105.24       446.22       0.4041          1.1177            1.1109         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            340.98       150.98       491.96       0.5466          1.0675            1.0600         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            340.98       100.59       441.57       0.4034          1.1179            1.1111         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            340.98       157.48       498.46       0.5459          1.0677            1.0602         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            340.98       105.62       446.60       0.4033          1.1180            1.1112         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            340.98       157.19       498.18       0.5457          1.0677            1.0602         2.09
IVF-Binary-256-nl316-random (self)                       340.98       234.07       575.05       0.4252          1.1217            1.1220         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)                 251.27        40.90       292.17       0.0960          1.6427            1.6567         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)                251.27        42.70       293.97       0.0959          1.6429            1.6568         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)                251.27        45.38       296.65       0.0959          1.6429            1.6568         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)                251.27        95.52       346.80       0.3562          1.1449            1.1376         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)                251.27       145.20       396.47       0.4976          1.0836            1.0741         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)               251.27        93.52       344.79       0.3551          1.1451            1.1377         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)               251.27       148.15       399.42       0.4963          1.0838            1.0742         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)               251.27        96.58       347.85       0.3551          1.1451            1.1377         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)               251.27       151.00       402.28       0.4963          1.0838            1.0742         1.93
IVF-Binary-256-nl158-pca (self)                          251.27       217.37       468.64       0.3800          1.1487            1.1470         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                289.52        45.36       334.88       0.1095          1.5965            1.6007         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                289.52        46.72       336.24       0.1094          1.5975            1.6019         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                289.52        50.87       340.39       0.1094          1.5976            1.6020         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               289.52        97.85       387.37       0.3870          1.1269            1.1202         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               289.52       147.97       437.49       0.5263          1.0737            1.0657         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               289.52        96.29       385.81       0.3864          1.1272            1.1205         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               289.52       149.24       438.76       0.5254          1.0739            1.0660         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               289.52       101.93       391.45       0.3863          1.1272            1.1205         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               289.52       154.77       444.29       0.5253          1.0739            1.0660         2.00
IVF-Binary-256-nl223-pca (self)                          289.52       226.21       515.73       0.4099          1.1303            1.1302         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                366.59        47.93       414.52       0.1160          1.5759            1.5792         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                366.59        47.92       414.50       0.1159          1.5765            1.5798         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                366.59        52.52       419.11       0.1158          1.5771            1.5801         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               366.59       100.68       467.27       0.3965          1.1216            1.1148         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               366.59       149.62       516.20       0.5374          1.0701            1.0627         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               366.59        98.48       465.07       0.3959          1.1218            1.1151         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               366.59       151.61       518.20       0.5366          1.0703            1.0630         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               366.59       102.05       468.63       0.3957          1.1219            1.1152         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               366.59       156.00       522.58       0.5363          1.0703            1.0631         2.09
IVF-Binary-256-nl316-pca (self)                          366.59       235.08       601.67       0.4197          1.1247            1.1254         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)              243.55        58.23       301.78       0.1479          1.5006            1.5075         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)             243.55        61.88       305.43       0.1479          1.5007            1.5075         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)             243.55        65.80       309.35       0.1479          1.5007            1.5075         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)             243.55       117.23       360.78       0.4609          1.0933            1.0899         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)             243.55       168.23       411.79       0.6093          1.0503            1.0458         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)            243.55       116.25       359.81       0.4606          1.0933            1.0899         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)            243.55       171.75       415.30       0.6092          1.0503            1.0458         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)            243.55       121.35       364.90       0.4606          1.0933            1.0899         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)            243.55       179.31       422.86       0.6091          1.0503            1.0458         3.71
IVF-Binary-512-nl158-random (self)                       243.55       308.01       551.57       0.4809          1.0993            1.0994         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             282.20        63.22       345.41       0.1561          1.4799            1.4845         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             282.20        64.06       346.26       0.1561          1.4802            1.4846         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             282.20        71.50       353.70       0.1561          1.4802            1.4846         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            282.20       119.28       401.48       0.4754          1.0877            1.0848         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            282.20       174.40       456.60       0.6216          1.0474            1.0431         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            282.20       118.44       400.64       0.4750          1.0878            1.0849         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            282.20       173.37       455.57       0.6211          1.0475            1.0432         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            282.20       126.27       408.47       0.4750          1.0878            1.0849         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            282.20       179.46       461.66       0.6211          1.0475            1.0432         3.77
IVF-Binary-512-nl223-random (self)                       282.20       306.31       588.51       0.4946          1.0941            1.0938         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             348.64        64.93       413.56       0.1598          1.4705            1.4754         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             348.64        66.30       414.94       0.1597          1.4708            1.4759         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             348.64        72.02       420.66       0.1597          1.4708            1.4759         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            348.64       121.72       470.35       0.4803          1.0859            1.0830         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            348.64       174.40       523.04       0.6273          1.0463            1.0422         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            348.64       120.01       468.65       0.4799          1.0861            1.0831         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            348.64       172.76       521.40       0.6267          1.0464            1.0424         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            348.64       125.88       474.52       0.4798          1.0861            1.0831         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            348.64       192.49       541.13       0.6267          1.0464            1.0424         3.86
IVF-Binary-512-nl316-random (self)                       348.64       305.36       654.00       0.4995          1.0923            1.0917         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)                 268.05        58.63       326.68       0.1474          1.5015            1.5083         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)                268.05        61.23       329.27       0.1473          1.5015            1.5083         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)                268.05        65.75       333.79       0.1473          1.5015            1.5083         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)                268.05       115.17       383.22       0.4558          1.0948            1.0910         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)                268.05       168.02       436.06       0.6048          1.0510            1.0463         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)               268.05       116.25       384.30       0.4554          1.0948            1.0910         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)               268.05       172.17       440.22       0.6044          1.0511            1.0463         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)               268.05       123.67       391.72       0.4554          1.0948            1.0910         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)               268.05       181.06       449.11       0.6044          1.0511            1.0463         3.71
IVF-Binary-512-nl158-pca (self)                          268.05       299.68       567.72       0.4788          1.1000            1.0995         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                304.73        62.93       367.66       0.1557          1.4808            1.4847         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                304.73        66.66       371.39       0.1556          1.4812            1.4854         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                304.73        70.24       374.97       0.1556          1.4812            1.4854         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               304.73       117.45       422.18       0.4715          1.0888            1.0850         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               304.73       168.79       473.52       0.6191          1.0478            1.0432         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               304.73       117.07       421.80       0.4710          1.0889            1.0851         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               304.73       173.32       478.05       0.6185          1.0479            1.0434         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               304.73       135.79       440.52       0.4709          1.0889            1.0851         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               304.73       187.76       492.49       0.6184          1.0479            1.0434         3.77
IVF-Binary-512-nl223-pca (self)                          304.73       298.00       602.73       0.4931          1.0945            1.0940         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                376.61        66.11       442.72       0.1590          1.4717            1.4754         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                376.61        66.18       442.79       0.1589          1.4720            1.4760         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                376.61        72.15       448.75       0.1589          1.4721            1.4761         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               376.61       121.02       497.63       0.4766          1.0869            1.0833         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               376.61       179.53       556.14       0.6238          1.0467            1.0422         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               376.61       120.79       497.40       0.4761          1.0870            1.0835         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               376.61       178.47       555.08       0.6230          1.0469            1.0423         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               376.61       126.33       502.94       0.4761          1.0870            1.0835         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               376.61       180.11       556.72       0.6229          1.0469            1.0424         3.86
IVF-Binary-512-nl316-pca (self)                          376.61       308.29       684.90       0.4984          1.0925            1.0920         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)             275.27        90.20       365.47       0.2162          1.3647            1.3717         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)            275.27        94.88       370.15       0.2162          1.3647            1.3717         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)            275.27       108.29       383.55       0.2162          1.3647            1.3717         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)            275.27       154.00       429.27       0.5873          1.0539            1.0515         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)            275.27       206.07       481.34       0.7382          1.0260            1.0223         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)           275.27       154.89       430.16       0.5872          1.0539            1.0515         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)           275.27       210.99       486.26       0.7382          1.0260            1.0223         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)           275.27       160.00       435.27       0.5872          1.0539            1.0515         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)           275.27       217.95       493.21       0.7382          1.0260            1.0223         7.26
IVF-Binary-1024-nl158-random (self)                      275.27       423.27       698.54       0.6121          1.0575            1.0545         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            317.37        93.22       410.59       0.2204          1.3573            1.3648         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            317.37        95.61       412.98       0.2204          1.3574            1.3648         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            317.37       106.48       423.86       0.2204          1.3574            1.3648         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           317.37       151.20       468.58       0.5941          1.0523            1.0500         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           317.37       207.21       524.59       0.7440          1.0252            1.0214         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           317.37       152.32       469.69       0.5938          1.0524            1.0500         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           317.37       220.93       538.31       0.7437          1.0252            1.0215         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           317.37       163.13       480.50       0.5938          1.0524            1.0500         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           317.37       220.41       537.78       0.7436          1.0252            1.0215         7.32
IVF-Binary-1024-nl223-random (self)                      317.37       418.56       735.93       0.6183          1.0559            1.0530         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            386.19        98.37       484.56       0.2223          1.3533            1.3607         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)            386.19       101.44       487.63       0.2222          1.3534            1.3610         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)            386.19       109.34       495.53       0.2222          1.3534            1.3610         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)           386.19       156.51       542.70       0.5963          1.0517            1.0493         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)           386.19       209.71       595.90       0.7464          1.0248            1.0211         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)           386.19       157.79       543.98       0.5959          1.0518            1.0494         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)           386.19       212.42       598.61       0.7461          1.0248            1.0211         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)           386.19       162.74       548.93       0.5959          1.0518            1.0494         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)           386.19       220.88       607.07       0.7460          1.0248            1.0211         7.42
IVF-Binary-1024-nl316-random (self)                      386.19       429.95       816.14       0.6211          1.0552            1.0522         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)                302.06        90.93       392.99       0.2128          1.3727            1.3795         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)               302.06        93.85       395.91       0.2128          1.3727            1.3795         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)               302.06       101.33       403.39       0.2128          1.3727            1.3795         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)               302.06       150.58       452.65       0.5780          1.0559            1.0531         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)               302.06       205.58       507.64       0.7294          1.0272            1.0232         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)              302.06       152.95       455.01       0.5779          1.0559            1.0531         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)              302.06       210.88       512.94       0.7293          1.0272            1.0232         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)              302.06       167.47       469.54       0.5779          1.0559            1.0531         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)              302.06       220.90       522.96       0.7293          1.0272            1.0232         7.26
IVF-Binary-1024-nl158-pca (self)                         302.06       420.63       722.69       0.6020          1.0601            1.0571         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               335.64        95.32       430.96       0.2176          1.3640            1.3702         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               335.64        97.59       433.23       0.2175          1.3642            1.3705         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               335.64       106.72       442.36       0.2175          1.3642            1.3705         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              335.64       154.49       490.13       0.5853          1.0541            1.0511         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              335.64       206.88       542.52       0.7358          1.0263            1.0223         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              335.64       152.63       488.27       0.5850          1.0542            1.0512         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              335.64       209.55       545.19       0.7354          1.0264            1.0223         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              335.64       165.27       500.91       0.5849          1.0542            1.0512         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              335.64       221.99       557.63       0.7353          1.0264            1.0223         7.32
IVF-Binary-1024-nl223-pca (self)                         335.64       422.12       757.77       0.6094          1.0581            1.0550         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               407.72        98.41       506.13       0.2191          1.3601            1.3666         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)               407.72        99.18       506.90       0.2190          1.3603            1.3667         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)               407.72       108.80       516.52       0.2190          1.3603            1.3667         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)              407.72       155.25       562.96       0.5876          1.0535            1.0508         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)              407.72       209.83       617.55       0.7378          1.0260            1.0222         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)              407.72       155.03       562.75       0.5872          1.0536            1.0509         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)              407.72       213.76       621.48       0.7374          1.0261            1.0222         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)              407.72       163.76       571.48       0.5872          1.0536            1.0509         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)              407.72       223.71       631.43       0.7374          1.0261            1.0222         7.42
IVF-Binary-1024-nl316-pca (self)                         407.72       430.79       838.51       0.6114          1.0575            1.0543         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)                199.50       151.70       351.20       0.1042          1.6406            1.6481         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               199.50       154.06       353.56       0.1041          1.6411            1.6484         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               199.50       157.87       357.38       0.1041          1.6412            1.6485         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               199.50       194.41       393.91       0.3771          1.1355            1.1269         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               199.50       336.09       535.60       0.5286          1.0742            1.0643         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              199.50       193.44       392.94       0.3754          1.1359            1.1272         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              199.50       338.06       537.56       0.5283          1.0743            1.0643         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              199.50       194.07       393.57       0.3752          1.1359            1.1272         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              199.50       341.68       541.18       0.5281          1.0743            1.0644         1.68
IVF-Binary-256-nl158-sign (self)                         199.50       527.60       727.10       0.3957          1.1432            1.1389         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               239.49       151.84       391.33       0.1045          1.6391            1.6451         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               239.49       154.04       393.53       0.1044          1.6403            1.6460         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               239.49       162.70       402.19       0.1043          1.6406            1.6464         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              239.49       190.46       429.95       0.3865          1.1297            1.1209         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              239.49       339.94       579.44       0.5363          1.0715            1.0626         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              239.49       193.49       432.99       0.3859          1.1299            1.1214         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              239.49       337.10       576.59       0.5353          1.0718            1.0629         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              239.49       200.03       439.53       0.3858          1.1300            1.1214         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              239.49       342.66       582.16       0.5350          1.0719            1.0629         1.75
IVF-Binary-256-nl223-sign (self)                         239.49       530.89       770.38       0.4055          1.1379            1.1334         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               309.32       155.12       464.43       0.1050          1.6357            1.6455         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               309.32       155.89       465.21       0.1049          1.6364            1.6457         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               309.32       160.53       469.84       0.1048          1.6371            1.6464         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              309.32       195.13       504.44       0.3920          1.1266            1.1191         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              309.32       338.87       648.19       0.5390          1.0706            1.0621         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              309.32       200.90       510.21       0.3913          1.1269            1.1193         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              309.32       338.84       648.16       0.5381          1.0709            1.0622         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              309.32       196.88       506.20       0.3911          1.1270            1.1193         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              309.32       349.33       658.65       0.5378          1.0709            1.0623         1.84
IVF-Binary-256-nl316-sign (self)                         309.32       536.27       845.59       0.4107          1.1342            1.1317         1.84
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
Exhaustive (query)                                        69.30     1_388.61     1_457.91       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.30     4_694.91     4_764.21       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                147.74       275.89       423.63       0.0733          1.4884            1.4968         2.03
ExhaustiveBinary-256-random-rf10 (query)                 147.74       400.89       548.63       0.2947          1.1327            1.1260         2.03
ExhaustiveBinary-256-random-rf20 (query)                 147.74       515.74       663.48       0.4174          1.0830            1.0716         2.03
ExhaustiveBinary-256-random (self)                       147.74     1_223.39     1_371.13       0.3150          1.1321            1.1268         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   230.52       268.00       498.52       0.0722          1.4941            1.5001         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    230.52       387.83       618.36       0.2934          1.1324            1.1267         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    230.52       519.35       749.88       0.4181          1.0813            1.0715         2.03
ExhaustiveBinary-256-pca (self)                          230.52     1_224.01     1_454.53       0.3112          1.1338            1.1298         2.03
ExhaustiveBinary-512-random_no_rr (query)                207.50       400.96       608.46       0.1110          1.4064            1.4153         4.05
ExhaustiveBinary-512-random-rf10 (query)                 207.50       536.11       743.61       0.3695          1.0935            1.0919         4.05
ExhaustiveBinary-512-random-rf20 (query)                 207.50       665.67       873.18       0.4981          1.0544            1.0517         4.05
ExhaustiveBinary-512-random (self)                       207.50     1_720.57     1_928.07       0.3856          1.0953            1.0998         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   298.29       418.36       716.65       0.1063          1.4156            1.4272         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    298.29       552.06       850.35       0.3574          1.0990            1.0958         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    298.29       663.75       962.04       0.4853          1.0582            1.0544         4.05
ExhaustiveBinary-512-pca (self)                          298.29     1_715.65     2_013.94       0.3753          1.0995            1.1040         4.05
ExhaustiveBinary-1024-random_no_rr (query)               268.66       605.97       874.64       0.1593          1.3242            1.3318         8.11
ExhaustiveBinary-1024-random-rf10 (query)                268.66       777.23     1_045.89       0.4456          1.0660            1.0678         8.11
ExhaustiveBinary-1024-random-rf20 (query)                268.66       904.31     1_172.98       0.5824          1.0370            1.0362         8.11
ExhaustiveBinary-1024-random (self)                      268.66     2_480.10     2_748.77       0.4595          1.0714            1.0740         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  348.78       596.69       945.47       0.1599          1.3236            1.3333         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   348.78       759.18     1_107.96       0.4446          1.0658            1.0680         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   348.78       885.93     1_234.71       0.5812          1.0370            1.0362         8.11
ExhaustiveBinary-1024-pca (self)                         348.78     2_503.12     2_851.90       0.4582          1.0716            1.0746         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   92.93       678.14       771.07       0.1292          1.3815            1.3877         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    92.93       748.78       841.71       0.3927          1.0844            1.0833         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    92.93     1_145.87     1_238.80       0.5336          1.0464            1.0444         3.05
ExhaustiveBinary-512-sign (self)                          92.93     2_398.54     2_491.47       0.4063          1.0885            1.0916         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)              431.56        74.48       506.04       0.0763          1.4794            1.4941         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)             431.56        77.55       509.11       0.0762          1.4798            1.4942         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)             431.56        82.12       513.68       0.0762          1.4798            1.4942         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)             431.56       159.24       590.80       0.3000          1.1310            1.1254         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)             431.56       257.56       689.12       0.4213          1.0820            1.0713         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)            431.56       164.69       596.25       0.2985          1.1313            1.1255         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)            431.56       249.85       681.41       0.4202          1.0821            1.0714         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)            431.56       168.37       599.93       0.2985          1.1313            1.1255         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)            431.56       252.48       684.04       0.4201          1.0821            1.0714         2.34
IVF-Binary-256-nl158-random (self)                       431.56       314.53       746.09       0.3184          1.1308            1.1264         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)             458.65        80.75       539.40       0.0885          1.4508            1.4557         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)             458.65        74.45       533.10       0.0885          1.4511            1.4559         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)             458.65        78.21       536.85       0.0885          1.4511            1.4559         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)            458.65       176.90       635.55       0.3263          1.1151            1.1083         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)            458.65       254.45       713.10       0.4512          1.0704            1.0628         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)            458.65       165.39       624.04       0.3262          1.1151            1.1083         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)            458.65       256.62       715.26       0.4511          1.0704            1.0628         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)            458.65       164.65       623.30       0.3262          1.1151            1.1083         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)            458.65       262.18       720.83       0.4511          1.0704            1.0628         2.47
IVF-Binary-256-nl223-random (self)                       458.65       340.14       798.79       0.3439          1.1148            1.1139         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)             612.06        79.07       691.13       0.0950          1.4366            1.4392         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)             612.06        85.30       697.36       0.0950          1.4366            1.4392         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)             612.06        84.44       696.50       0.0950          1.4367            1.4392         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)            612.06       171.73       783.79       0.3368          1.1084            1.1032         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)            612.06       259.05       871.11       0.4647          1.0652            1.0593         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)            612.06       169.17       781.23       0.3367          1.1084            1.1032         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)            612.06       267.75       879.81       0.4646          1.0652            1.0593         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)            612.06       182.23       794.29       0.3366          1.1084            1.1032         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)            612.06       262.62       874.68       0.4646          1.0652            1.0593         2.65
IVF-Binary-256-nl316-random (self)                       612.06       354.82       966.88       0.3558          1.1069            1.1093         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)                 608.73        69.35       678.08       0.0755          1.4847            1.4975         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)                608.73        68.38       677.11       0.0754          1.4851            1.4976         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)                608.73        69.30       678.03       0.0754          1.4851            1.4976         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)                608.73       155.94       764.67       0.2991          1.1304            1.1259         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)                608.73       246.31       855.04       0.4222          1.0801            1.0710         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)               608.73       168.41       777.14       0.2978          1.1306            1.1260         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)               608.73       247.86       856.59       0.4210          1.0803            1.0711         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)               608.73       155.54       764.27       0.2978          1.1306            1.1260         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)               608.73       253.55       862.28       0.4209          1.0803            1.0711         2.34
IVF-Binary-256-nl158-pca (self)                          608.73       299.97       908.70       0.3154          1.1320            1.1293         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)                535.79        72.15       607.94       0.0881          1.4529            1.4576         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)                535.79        80.21       616.00       0.0880          1.4531            1.4578         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)                535.79        79.42       615.21       0.0880          1.4531            1.4578         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)               535.79       171.65       707.45       0.3292          1.1121            1.1063         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)               535.79       265.86       801.66       0.4516          1.0692            1.0621         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)               535.79       161.01       696.81       0.3291          1.1121            1.1063         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)               535.79       254.44       790.23       0.4515          1.0693            1.0621         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)               535.79       165.59       701.39       0.3291          1.1121            1.1063         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)               535.79       259.98       795.78       0.4515          1.0693            1.0621         2.47
IVF-Binary-256-nl223-pca (self)                          535.79       326.97       862.77       0.3476          1.1104            1.1128         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)                685.76        79.06       764.82       0.0950          1.4351            1.4365         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)                685.76        84.08       769.84       0.0950          1.4352            1.4366         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)                685.76        83.12       768.87       0.0950          1.4352            1.4366         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)               685.76       173.22       858.98       0.3402          1.1056            1.1005         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)               685.76       263.76       949.52       0.4617          1.0661            1.0595         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)               685.76       181.07       866.83       0.3401          1.1056            1.1005         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)               685.76       274.31       960.06       0.4616          1.0661            1.0595         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)               685.76       169.85       855.60       0.3401          1.1056            1.1005         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)               685.76       261.83       947.59       0.4616          1.0661            1.0595         2.65
IVF-Binary-256-nl316-pca (self)                          685.76       364.90     1_050.66       0.3590          1.1032            1.1073         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)              600.35        97.79       698.15       0.1124          1.4040            1.4144         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)             600.35       105.25       705.60       0.1124          1.4041            1.4144         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)             600.35       106.15       706.50       0.1124          1.4041            1.4144         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)             600.35       210.56       810.92       0.3712          1.0931            1.0918         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)             600.35       282.04       882.39       0.4994          1.0541            1.0516         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)            600.35       189.08       789.43       0.3709          1.0932            1.0918         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)            600.35       286.21       886.56       0.4992          1.0541            1.0516         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)            600.35       191.68       792.03       0.3709          1.0932            1.0918         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)            600.35       289.46       889.81       0.4992          1.0541            1.0516         4.36
IVF-Binary-512-nl158-random (self)                       600.35       440.12     1_040.47       0.3868          1.0949            1.0997         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)             535.27        99.19       634.46       0.1208          1.3872            1.3942         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)             535.27       107.14       642.41       0.1208          1.3873            1.3943         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)             535.27       110.50       645.77       0.1208          1.3873            1.3943         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)            535.27       203.88       739.15       0.3843          1.0875            1.0869         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)            535.27       289.94       825.20       0.5124          1.0510            1.0489         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)            535.27       193.76       729.03       0.3842          1.0875            1.0869         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)            535.27       292.50       827.77       0.5123          1.0510            1.0489         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)            535.27       208.81       744.08       0.3842          1.0875            1.0869         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)            535.27       303.46       838.72       0.5123          1.0510            1.0489         4.49
IVF-Binary-512-nl223-random (self)                       535.27       461.90       997.17       0.3990          1.0902            1.0949         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)             631.63       112.07       743.70       0.1243          1.3793            1.3847         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)             631.63       109.57       741.20       0.1243          1.3794            1.3847         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)             631.63       112.11       743.74       0.1243          1.3794            1.3847         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)            631.63       202.54       834.16       0.3889          1.0856            1.0850         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)            631.63       293.21       924.84       0.5162          1.0501            1.0479         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)            631.63       195.79       827.42       0.3888          1.0856            1.0850         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)            631.63       294.83       926.45       0.5162          1.0501            1.0479         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)            631.63       202.89       834.52       0.3888          1.0856            1.0850         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)            631.63       299.15       930.77       0.5162          1.0501            1.0479         4.67
IVF-Binary-512-nl316-random (self)                       631.63       500.51     1_132.14       0.4032          1.0885            1.0932         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)                 596.51        91.81       688.33       0.1079          1.4130            1.4261         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)                596.51        94.88       691.40       0.1079          1.4130            1.4261         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)                596.51        96.78       693.29       0.1079          1.4130            1.4261         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)                596.51       182.60       779.12       0.3595          1.0984            1.0956         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)                596.51       275.87       872.39       0.4868          1.0578            1.0543         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)               596.51       185.62       782.13       0.3591          1.0984            1.0956         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)               596.51       279.79       876.30       0.4865          1.0579            1.0543         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)               596.51       186.30       782.81       0.3591          1.0984            1.0956         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)               596.51       282.83       879.35       0.4865          1.0579            1.0543         4.36
IVF-Binary-512-nl158-pca (self)                          596.51       429.16     1_025.67       0.3769          1.0989            1.1039         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)                590.30        98.67       688.97       0.1169          1.3942            1.4010         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)                590.30       101.87       692.17       0.1169          1.3942            1.4010         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)                590.30       105.32       695.62       0.1169          1.3942            1.4010         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)               590.30       193.84       784.14       0.3738          1.0919            1.0898         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)               590.30       284.53       874.83       0.5015          1.0541            1.0513         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)               590.30       189.61       779.91       0.3738          1.0919            1.0898         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)               590.30       284.72       875.02       0.5015          1.0541            1.0513         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)               590.30       194.52       784.82       0.3738          1.0919            1.0898         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)               590.30       289.49       879.79       0.5015          1.0541            1.0513         4.49
IVF-Binary-512-nl223-pca (self)                          590.30       447.99     1_038.30       0.3906          1.0936            1.0982         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)                688.28       106.35       794.63       0.1208          1.3851            1.3903         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)                688.28       106.60       794.88       0.1208          1.3851            1.3903         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)                688.28       109.98       798.26       0.1208          1.3851            1.3903         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)               688.28       197.12       885.40       0.3799          1.0892            1.0879         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)               688.28       292.58       980.86       0.5061          1.0528            1.0504         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)               688.28       197.92       886.20       0.3799          1.0892            1.0879         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)               688.28       300.80       989.08       0.5060          1.0528            1.0505         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)               688.28       201.53       889.81       0.3799          1.0892            1.0879         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)               688.28       299.62       987.90       0.5060          1.0528            1.0505         4.67
IVF-Binary-512-nl316-pca (self)                          688.28       472.96     1_161.24       0.3958          1.0911            1.0960         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)             539.52       145.92       685.44       0.1598          1.3236            1.3316         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)            539.52       146.02       685.54       0.1598          1.3236            1.3316         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)            539.52       153.62       693.13       0.1598          1.3236            1.3316         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)            539.52       240.44       779.96       0.4461          1.0659            1.0678         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)            539.52       334.20       873.72       0.5827          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)           539.52       238.89       778.41       0.4461          1.0659            1.0678         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)           539.52       342.21       881.72       0.5827          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)           539.52       257.61       797.13       0.4461          1.0659            1.0678         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)           539.52       348.33       887.85       0.5827          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-random (self)                      539.52       653.27     1_192.79       0.4600          1.0713            1.0740         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)            552.62       153.97       706.59       0.1639          1.3163            1.3250         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)            552.62       153.47       706.09       0.1639          1.3163            1.3250         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)            552.62       161.32       713.94       0.1639          1.3163            1.3250         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)           552.62       247.27       799.89       0.4533          1.0639            1.0658         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)           552.62       352.35       904.97       0.5891          1.0358            1.0351         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)           552.62       251.43       804.05       0.4533          1.0639            1.0658         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)           552.62       361.44       914.05       0.5891          1.0358            1.0351         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)           552.62       255.67       808.29       0.4533          1.0639            1.0658         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)           552.62       366.62       919.24       0.5891          1.0358            1.0351         8.54
IVF-Binary-1024-nl223-random (self)                      552.62       662.10     1_214.72       0.4673          1.0692            1.0718         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)            651.30       162.19       813.49       0.1655          1.3134            1.3215         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)            651.30       160.07       811.37       0.1655          1.3134            1.3215         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)            651.30       165.49       816.79       0.1655          1.3134            1.3215         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)           651.30       253.04       904.34       0.4550          1.0634            1.0651         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)           651.30       360.44     1_011.74       0.5912          1.0356            1.0348         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)           651.30       257.25       908.55       0.4550          1.0634            1.0651         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)           651.30       363.82     1_015.12       0.5912          1.0356            1.0348         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)           651.30       264.97       916.27       0.4550          1.0634            1.0651         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)           651.30       380.38     1_031.68       0.5912          1.0356            1.0348         8.73
IVF-Binary-1024-nl316-random (self)                      651.30       688.25     1_339.55       0.4694          1.0686            1.0714         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)                632.84       147.37       780.21       0.1605          1.3229            1.3332         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)               632.84       152.20       785.04       0.1605          1.3229            1.3332         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)               632.84       150.71       783.56       0.1605          1.3229            1.3332         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)               632.84       239.60       872.44       0.4452          1.0657            1.0680         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)               632.84       337.88       970.72       0.5817          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)              632.84       242.16       875.00       0.4452          1.0657            1.0680         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)              632.84       342.52       975.36       0.5817          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)              632.84       244.97       877.81       0.4452          1.0657            1.0680         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)              632.84       349.69       982.53       0.5817          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-pca (self)                         632.84       652.33     1_285.17       0.4588          1.0715            1.0746         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)               645.20       155.13       800.33       0.1641          1.3158            1.3264         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)               645.20       153.69       798.89       0.1641          1.3158            1.3264         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)               645.20       161.61       806.81       0.1641          1.3158            1.3264         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)              645.20       256.55       901.75       0.4515          1.0638            1.0664         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)              645.20       351.58       996.78       0.5882          1.0359            1.0353         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)              645.20       263.28       908.48       0.4515          1.0638            1.0664         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)              645.20       353.31       998.51       0.5882          1.0359            1.0353         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)              645.20       256.91       902.11       0.4515          1.0638            1.0664         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)              645.20       386.54     1_031.74       0.5882          1.0359            1.0353         8.54
IVF-Binary-1024-nl223-pca (self)                         645.20       704.32     1_349.52       0.4657          1.0695            1.0725         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)               769.03       165.47       934.50       0.1656          1.3134            1.3240         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)               769.03       179.00       948.03       0.1656          1.3134            1.3240         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)               769.03       169.83       938.86       0.1656          1.3134            1.3240         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)              769.03       268.07     1_037.11       0.4539          1.0631            1.0657         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)              769.03       358.23     1_127.26       0.5899          1.0356            1.0350         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)              769.03       256.72     1_025.76       0.4539          1.0631            1.0657         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)              769.03       372.65     1_141.68       0.5899          1.0356            1.0350         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)              769.03       264.76     1_033.79       0.4539          1.0631            1.0657         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)              769.03       368.43     1_137.46       0.5899          1.0356            1.0350         8.73
IVF-Binary-1024-nl316-pca (self)                         769.03       690.37     1_459.40       0.4677          1.0689            1.0720         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)                372.72       291.07       663.80       0.1292          1.3813            1.3870         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)               372.72       299.41       672.13       0.1292          1.3813            1.3870         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)               372.72       294.97       667.70       0.1292          1.3813            1.3870         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)               372.72       359.28       732.00       0.3940          1.0840            1.0833         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)               372.72       643.43     1_016.15       0.5343          1.0463            1.0444         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)              372.72       360.15       732.87       0.3939          1.0840            1.0833         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)              372.72       670.37     1_043.10       0.5343          1.0463            1.0444         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)              372.72       365.68       738.40       0.3939          1.0840            1.0833         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)              372.72       710.14     1_082.86       0.5343          1.0463            1.0444         3.36
IVF-Binary-512-nl158-sign (self)                         372.72       993.70     1_366.42       0.4072          1.0883            1.0915         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               377.04       291.13       668.18       0.1295          1.3814            1.3865         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               377.04       294.89       671.93       0.1295          1.3814            1.3865         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               377.04       299.12       676.17       0.1295          1.3814            1.3865         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              377.04       377.84       754.88       0.3991          1.0819            1.0817         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              377.04       656.09     1_033.14       0.5376          1.0456            1.0440         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              377.04       368.09       745.13       0.3991          1.0819            1.0817         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              377.04       658.79     1_035.83       0.5375          1.0456            1.0440         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              377.04       390.20       767.24       0.3991          1.0819            1.0817         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              377.04       662.27     1_039.31       0.5375          1.0456            1.0440         3.49
IVF-Binary-512-nl223-sign (self)                         377.04     1_106.64     1_483.68       0.4125          1.0864            1.0898         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)               484.77       296.00       780.77       0.1294          1.3810            1.3867         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)               484.77       300.19       784.97       0.1294          1.3810            1.3867         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)               484.77       301.71       786.48       0.1294          1.3810            1.3867         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)              484.77       373.99       858.76       0.4000          1.0814            1.0813         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)              484.77       659.12     1_143.89       0.5386          1.0452            1.0437         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)              484.77       371.95       856.72       0.4000          1.0814            1.0813         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)              484.77       695.50     1_180.27       0.5385          1.0452            1.0437         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)              484.77       382.61       867.38       0.4000          1.0814            1.0813         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)              484.77       674.39     1_159.16       0.5385          1.0452            1.0437         3.67
IVF-Binary-512-nl316-sign (self)                         484.77     1_033.67     1_518.44       0.4132          1.0861            1.0894         3.67
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
Exhaustive (query)                                       101.55     1_890.45     1_992.00       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.55     6_496.81     6_598.36       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                200.12       293.05       493.17       0.0662          1.3769            1.3786         2.28
ExhaustiveBinary-256-random-rf10 (query)                 200.12       427.53       627.65       0.2741          1.1112            1.1017         2.28
ExhaustiveBinary-256-random-rf20 (query)                 200.12       571.34       771.46       0.3877          1.0706            1.0590         2.28
ExhaustiveBinary-256-random (self)                       200.12     1_306.91     1_507.04       0.2874          1.1077            1.0994         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   397.61       285.17       682.79       0.0653          1.3793            1.3770         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    397.61       427.38       824.99       0.2701          1.1138            1.1018         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    397.61       570.62       968.23       0.3852          1.0726            1.0585         2.28
ExhaustiveBinary-256-pca (self)                          397.61     1_315.42     1_713.03       0.2820          1.1100            1.0990         2.28
ExhaustiveBinary-512-random_no_rr (query)                301.15       409.03       710.17       0.0934          1.3249            1.3316         4.55
ExhaustiveBinary-512-random-rf10 (query)                 301.15       567.48       868.63       0.3220          1.0860            1.0806         4.55
ExhaustiveBinary-512-random-rf20 (query)                 301.15       723.77     1_024.92       0.4345          1.0527            1.0485         4.55
ExhaustiveBinary-512-random (self)                       301.15     1_870.34     2_171.49       0.3344          1.0826            1.0842         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   498.35       407.53       905.88       0.0951          1.3226            1.3263         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    498.35       563.71     1_062.06       0.3246          1.0847            1.0788         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    498.35       721.68     1_220.03       0.4400          1.0517            1.0468         4.55
ExhaustiveBinary-512-pca (self)                          498.35     1_851.38     2_349.73       0.3361          1.0819            1.0825         4.55
ExhaustiveBinary-1024-random_no_rr (query)               500.69       635.72     1_136.40       0.1319          1.2685            1.2723         9.11
ExhaustiveBinary-1024-random-rf10 (query)                500.69       809.08     1_309.76       0.3742          1.0644            1.0666         9.11
ExhaustiveBinary-1024-random-rf20 (query)                500.69       977.71     1_478.40       0.4906          1.0386            1.0390         9.11
ExhaustiveBinary-1024-random (self)                      500.69     2_653.82     3_154.51       0.3823          1.0666            1.0715         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  690.79       646.00     1_336.79       0.1355          1.2623            1.2651         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   690.79       813.88     1_504.66       0.3804          1.0622            1.0643         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   690.79       995.13     1_685.92       0.4993          1.0369            1.0374         9.11
ExhaustiveBinary-1024-pca (self)                         690.79     2_670.75     3_361.53       0.3870          1.0651            1.0695         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  132.19       854.05       986.25       0.1284          1.2822            1.2821         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   132.19       936.01     1_068.20       0.3618          1.0706            1.0699         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   132.19     1_453.83     1_586.02       0.4847          1.0407            1.0395         4.58
ExhaustiveBinary-768-sign (self)                         132.19     3_024.32     3_156.51       0.3694          1.0716            1.0747         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)              545.44        95.60       641.03       0.0686          1.3708            1.3769         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)             545.44        92.31       637.75       0.0685          1.3711            1.3770         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)             545.44        98.76       644.20       0.0685          1.3711            1.3770         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)             545.44       194.14       739.57       0.2782          1.1103            1.1014         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)             545.44       303.77       849.20       0.3901          1.0700            1.0589         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)            545.44       189.55       734.99       0.2770          1.1104            1.1014         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)            545.44       307.06       852.50       0.3894          1.0701            1.0589         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)            545.44       195.27       740.70       0.2769          1.1104            1.1014         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)            545.44       307.92       853.36       0.3893          1.0701            1.0589         2.74
IVF-Binary-256-nl158-random (self)                       545.44       391.14       936.58       0.2904          1.1068            1.0992         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)             575.97        93.09       669.07       0.0783          1.3507            1.3543         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)             575.97        94.24       670.22       0.0783          1.3508            1.3543         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)             575.97        97.26       673.24       0.0783          1.3508            1.3543         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)            575.97       205.09       781.07       0.3028          1.0946            1.0866         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)            575.97       325.78       901.75       0.4175          1.0588            1.0517         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)            575.97       199.87       775.85       0.3028          1.0946            1.0866         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)            575.97       318.28       894.25       0.4174          1.0588            1.0517         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)            575.97       203.43       779.40       0.3028          1.0946            1.0866         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)            575.97       320.19       896.16       0.4174          1.0588            1.0517         2.93
IVF-Binary-256-nl223-random (self)                       575.97       427.46     1_003.43       0.3172          1.0895            1.0882         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)             702.69       103.51       806.20       0.0851          1.3378            1.3389         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)             702.69       103.58       806.27       0.0851          1.3379            1.3390         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)             702.69       105.80       808.49       0.0851          1.3380            1.3390         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)            702.69       215.59       918.28       0.3176          1.0865            1.0809         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)            702.69       331.52     1_034.21       0.4309          1.0543            1.0490         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)            702.69       215.32       918.01       0.3175          1.0865            1.0809         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)            702.69       330.79     1_033.48       0.4308          1.0543            1.0490         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)            702.69       215.93       918.62       0.3174          1.0865            1.0809         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)            702.69       335.21     1_037.89       0.4307          1.0543            1.0490         3.21
IVF-Binary-256-nl316-random (self)                       702.69       466.71     1_169.39       0.3311          1.0807            1.0831         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)                 748.40        87.05       835.46       0.0678          1.3725            1.3752         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)                748.40        85.89       834.29       0.0677          1.3726            1.3752         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)                748.40        88.07       836.47       0.0677          1.3726            1.3752         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)                748.40       192.42       940.82       0.2747          1.1123            1.1014         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)                748.40       305.56     1_053.96       0.3883          1.0717            1.0584         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)               748.40       188.72       937.12       0.2735          1.1124            1.1014         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)               748.40       305.18     1_053.58       0.3874          1.0717            1.0585         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)               748.40       193.02       941.42       0.2735          1.1124            1.1014         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)               748.40       307.34     1_055.74       0.3873          1.0717            1.0585         2.74
IVF-Binary-256-nl158-pca (self)                          748.40       386.36     1_134.76       0.2853          1.1087            1.0988         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)                798.49        92.15       890.64       0.0776          1.3536            1.3551         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)                798.49        98.48       896.97       0.0776          1.3536            1.3551         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)                798.49        97.29       895.78       0.0776          1.3536            1.3551         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)               798.49       203.37     1_001.86       0.2987          1.0962            1.0873         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)               798.49       317.34     1_115.83       0.4151          1.0600            1.0520         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)               798.49       201.16       999.65       0.2986          1.0962            1.0873         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)               798.49       318.86     1_117.35       0.4150          1.0600            1.0520         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)               798.49       209.07     1_007.55       0.2986          1.0962            1.0873         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)               798.49       328.77     1_127.26       0.4149          1.0600            1.0520         2.93
IVF-Binary-256-nl223-pca (self)                          798.49       433.73     1_232.21       0.3109          1.0907            1.0885         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)                894.58       102.49       997.07       0.0848          1.3398            1.3382         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)                894.58       105.61     1_000.19       0.0848          1.3399            1.3382         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)                894.58       106.38     1_000.96       0.0848          1.3399            1.3382         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)               894.58       212.69     1_107.26       0.3102          1.0896            1.0821         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)               894.58       347.90     1_242.47       0.4249          1.0567            1.0497         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)               894.58       208.83     1_103.41       0.3101          1.0897            1.0821         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)               894.58       329.40     1_223.97       0.4248          1.0567            1.0497         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)               894.58       215.02     1_109.60       0.3101          1.0897            1.0821         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)               894.58       339.01     1_233.58       0.4247          1.0567            1.0497         3.21
IVF-Binary-256-nl316-pca (self)                          894.58       467.65     1_362.23       0.3223          1.0839            1.0844         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)              657.96       122.03       779.99       0.0948          1.3227            1.3308         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)             657.96       129.62       787.58       0.0947          1.3228            1.3308         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)             657.96       124.99       782.96       0.0947          1.3228            1.3308         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)             657.96       232.84       890.80       0.3234          1.0857            1.0806         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)             657.96       349.18     1_007.14       0.4355          1.0524            1.0484         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)            657.96       234.03       892.00       0.3230          1.0857            1.0806         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)            657.96       352.11     1_010.07       0.4354          1.0524            1.0484         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)            657.96       238.18       896.15       0.3230          1.0857            1.0806         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)            657.96       357.24     1_015.20       0.4354          1.0524            1.0484         5.02
IVF-Binary-512-nl158-random (self)                       657.96       556.63     1_214.59       0.3354          1.0823            1.0842         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)             684.85       130.61       815.46       0.1036          1.3077            1.3089         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)             684.85       141.52       826.37       0.1036          1.3077            1.3089         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)             684.85       138.89       823.74       0.1036          1.3077            1.3089         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)            684.85       242.76       927.61       0.3368          1.0788            1.0763         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)            684.85       364.66     1_049.51       0.4484          1.0488            1.0463         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)            684.85       241.73       926.58       0.3368          1.0788            1.0763         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)            684.85       368.74     1_053.59       0.4484          1.0488            1.0463         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)            684.85       247.07       931.91       0.3368          1.0788            1.0763         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)            684.85       371.35     1_056.20       0.4484          1.0488            1.0463         5.21
IVF-Binary-512-nl223-random (self)                       684.85       616.37     1_301.22       0.3474          1.0766            1.0803         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)             809.91       145.25       955.16       0.1079          1.3002            1.2990         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)             809.91       140.66       950.57       0.1079          1.3002            1.2990         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)             809.91       143.73       953.64       0.1079          1.3002            1.2990         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)            809.91       253.64     1_063.55       0.3426          1.0764            1.0745         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)            809.91       377.90     1_187.81       0.4546          1.0469            1.0449         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)            809.91       249.68     1_059.59       0.3426          1.0764            1.0745         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)            809.91       384.75     1_194.66       0.4545          1.0469            1.0449         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)            809.91       266.65     1_076.56       0.3426          1.0764            1.0745         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)            809.91       392.65     1_202.56       0.4545          1.0469            1.0449         5.48
IVF-Binary-512-nl316-random (self)                       809.91       639.99     1_449.90       0.3530          1.0743            1.0786         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)                 927.69       142.51     1_070.20       0.0962          1.3205            1.3253         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)                927.69       123.87     1_051.56       0.0961          1.3206            1.3253         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)                927.69       129.22     1_056.91       0.0961          1.3206            1.3253         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)                927.69       243.68     1_171.37       0.3262          1.0845            1.0787         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)                927.69       364.22     1_291.91       0.4407          1.0516            1.0467         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)               927.69       233.76     1_161.45       0.3256          1.0845            1.0787         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)               927.69       381.12     1_308.81       0.4406          1.0516            1.0467         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)               927.69       244.54     1_172.23       0.3256          1.0845            1.0787         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)               927.69       358.45     1_286.14       0.4406          1.0516            1.0467         5.02
IVF-Binary-512-nl158-pca (self)                          927.69       564.19     1_491.88       0.3369          1.0817            1.0825         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)                881.83       130.97     1_012.80       0.1043          1.3066            1.3057         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)                881.83       130.81     1_012.63       0.1043          1.3066            1.3057         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)                881.83       134.22     1_016.05       0.1043          1.3066            1.3057         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)               881.83       243.19     1_125.02       0.3375          1.0787            1.0753         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)               881.83       365.03     1_246.85       0.4530          1.0478            1.0448         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)               881.83       241.77     1_123.60       0.3375          1.0787            1.0753         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)               881.83       365.91     1_247.73       0.4530          1.0478            1.0448         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)               881.83       246.18     1_128.01       0.3375          1.0787            1.0753         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)               881.83       379.97     1_261.80       0.4530          1.0478            1.0448         5.21
IVF-Binary-512-nl223-pca (self)                          881.83       627.24     1_509.06       0.3473          1.0770            1.0796         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)              1_003.79       141.70     1_145.49       0.1081          1.2992            1.2963         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)              1_003.79       147.61     1_151.40       0.1081          1.2993            1.2963         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)              1_003.79       143.65     1_147.44       0.1081          1.2993            1.2963         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)             1_003.79       252.71     1_256.50       0.3443          1.0757            1.0732         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)             1_003.79       376.95     1_380.74       0.4586          1.0463            1.0439         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)             1_003.79       254.65     1_258.44       0.3442          1.0757            1.0732         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)             1_003.79       379.72     1_383.51       0.4585          1.0463            1.0439         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)             1_003.79       256.28     1_260.07       0.3442          1.0757            1.0732         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)             1_003.79       395.39     1_399.18       0.4585          1.0463            1.0439         5.48
IVF-Binary-512-nl316-pca (self)                        1_003.79       630.67     1_634.46       0.3530          1.0746            1.0780         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)             856.80       196.89     1_053.69       0.1327          1.2679            1.2722         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)            856.80       197.04     1_053.84       0.1327          1.2680            1.2722         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)            856.80       201.17     1_057.97       0.1327          1.2680            1.2722         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)            856.80       336.39     1_193.19       0.3746          1.0644            1.0666         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)            856.80       446.31     1_303.11       0.4908          1.0386            1.0390         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)           856.80       324.72     1_181.52       0.3745          1.0644            1.0666         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)           856.80       457.63     1_314.43       0.4907          1.0386            1.0390         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)           856.80       330.53     1_187.33       0.3745          1.0644            1.0666         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)           856.80       464.22     1_321.02       0.4907          1.0386            1.0390         9.57
IVF-Binary-1024-nl158-random (self)                      856.80       855.24     1_712.04       0.3825          1.0665            1.0715         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)            878.35       217.04     1_095.39       0.1370          1.2612            1.2658         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)            878.35       206.22     1_084.57       0.1370          1.2612            1.2658         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)            878.35       212.13     1_090.48       0.1370          1.2612            1.2658         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)           878.35       335.54     1_213.89       0.3801          1.0625            1.0654         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)           878.35       467.69     1_346.04       0.4966          1.0375            1.0380         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)           878.35       337.33     1_215.68       0.3801          1.0625            1.0654         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)           878.35       477.38     1_355.73       0.4966          1.0375            1.0380         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)           878.35       355.16     1_233.52       0.3801          1.0625            1.0654         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)           878.35       493.70     1_372.06       0.4966          1.0375            1.0380         9.76
IVF-Binary-1024-nl223-random (self)                      878.35       905.48     1_783.83       0.3880          1.0649            1.0699         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)          1_008.50       220.45     1_228.96       0.1388          1.2580            1.2631        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)          1_008.50       219.29     1_227.79       0.1388          1.2580            1.2631        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)          1_008.50       223.26     1_231.76       0.1388          1.2580            1.2631        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)         1_008.50       355.21     1_363.71       0.3834          1.0615            1.0643        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)         1_008.50       481.24     1_489.75       0.5004          1.0368            1.0374        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)         1_008.50       356.07     1_364.58       0.3834          1.0615            1.0643        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)         1_008.50       485.67     1_494.18       0.5004          1.0368            1.0374        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)         1_008.50       361.44     1_369.95       0.3834          1.0615            1.0643        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)         1_008.50       493.45     1_501.95       0.5004          1.0368            1.0374        10.04
IVF-Binary-1024-nl316-random (self)                    1_008.50       953.51     1_962.01       0.3911          1.0639            1.0689        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)              1_058.50       200.21     1_258.71       0.1360          1.2618            1.2650         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)             1_058.50       199.64     1_258.14       0.1360          1.2618            1.2650         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)             1_058.50       203.09     1_261.59       0.1360          1.2618            1.2650         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)             1_058.50       324.07     1_382.58       0.3808          1.0621            1.0643         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)             1_058.50       448.45     1_506.96       0.4995          1.0369            1.0374         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)            1_058.50       342.65     1_401.15       0.3807          1.0621            1.0643         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)            1_058.50       457.49     1_515.99       0.4995          1.0369            1.0374         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)            1_058.50       334.90     1_393.41       0.3807          1.0621            1.0643         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)            1_058.50       503.64     1_562.14       0.4995          1.0369            1.0374         9.57
IVF-Binary-1024-nl158-pca (self)                       1_058.50       881.69     1_940.19       0.3871          1.0651            1.0695         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_080.29       209.29     1_289.58       0.1396          1.2560            1.2601         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_080.29       207.94     1_288.23       0.1396          1.2560            1.2601         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_080.29       214.38     1_294.67       0.1396          1.2560            1.2601         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_080.29       334.89     1_415.18       0.3857          1.0604            1.0631         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_080.29       463.43     1_543.71       0.5050          1.0360            1.0365         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_080.29       336.82     1_417.11       0.3857          1.0604            1.0631         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_080.29       467.43     1_547.71       0.5050          1.0360            1.0365         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_080.29       346.31     1_426.60       0.3857          1.0604            1.0631         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_080.29       479.06     1_559.35       0.5050          1.0360            1.0365         9.76
IVF-Binary-1024-nl223-pca (self)                       1_080.29       922.58     2_002.87       0.3923          1.0636            1.0682         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_219.40       224.74     1_444.14       0.1414          1.2531            1.2577        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_219.40       221.16     1_440.56       0.1414          1.2531            1.2577        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_219.40       224.06     1_443.46       0.1414          1.2531            1.2577        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_219.40       363.97     1_583.37       0.3890          1.0595            1.0623        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_219.40       478.00     1_697.40       0.5081          1.0354            1.0361        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_219.40       354.99     1_574.38       0.3890          1.0595            1.0623        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_219.40       487.87     1_707.26       0.5081          1.0354            1.0361        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_219.40       380.26     1_599.66       0.3890          1.0595            1.0623        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_219.40       492.56     1_711.95       0.5081          1.0354            1.0361        10.04
IVF-Binary-1024-nl316-pca (self)                       1_219.40       960.91     2_180.31       0.3955          1.0627            1.0672        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)                482.11       406.45       888.56       0.1284          1.2834            1.2814         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)               482.11       412.93       895.03       0.1284          1.2834            1.2814         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)               482.11       417.79       899.90       0.1284          1.2834            1.2814         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)               482.11       501.13       983.24       0.3625          1.0704            1.0698         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)               482.11       905.61     1_387.72       0.4852          1.0406            1.0396         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)              482.11       500.11       982.22       0.3625          1.0704            1.0698         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)              482.11       914.50     1_396.60       0.4852          1.0406            1.0396         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)              482.11       505.14       987.25       0.3625          1.0704            1.0698         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)              482.11       914.90     1_397.00       0.4852          1.0406            1.0396         5.04
IVF-Binary-768-nl158-sign (self)                         482.11     1_438.10     1_920.20       0.3700          1.0715            1.0746         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)               514.81       412.71       927.53       0.1285          1.2800            1.2812         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)               514.81       415.16       929.97       0.1285          1.2800            1.2812         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)               514.81       420.23       935.05       0.1285          1.2800            1.2812         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)              514.81       507.63     1_022.45       0.3665          1.0688            1.0688         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)              514.81       911.86     1_426.68       0.4875          1.0400            1.0392         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)              514.81       508.11     1_022.93       0.3665          1.0688            1.0688         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)              514.81       916.90     1_431.72       0.4874          1.0400            1.0392         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)              514.81       528.60     1_043.42       0.3665          1.0688            1.0688         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)              514.81       927.35     1_442.16       0.4874          1.0400            1.0392         5.23
IVF-Binary-768-nl223-sign (self)                         514.81     1_446.68     1_961.50       0.3733          1.0702            1.0737         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)               643.70       423.48     1_067.17       0.1282          1.2804            1.2812         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)               643.70       423.01     1_066.71       0.1282          1.2804            1.2812         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)               643.70       440.83     1_084.53       0.1282          1.2804            1.2812         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)              643.70       520.44     1_164.14       0.3668          1.0684            1.0687         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)              643.70       946.43     1_590.12       0.4884          1.0398            1.0392         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)              643.70       525.54     1_169.24       0.3668          1.0684            1.0687         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)              643.70       936.81     1_580.50       0.4884          1.0398            1.0392         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)              643.70       520.93     1_164.63       0.3668          1.0684            1.0687         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)              643.70       935.71     1_579.41       0.4884          1.0398            1.0392         5.51
IVF-Binary-768-nl316-sign (self)                         643.70     1_465.60     2_109.29       0.3742          1.0700            1.0735         5.51
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
Exhaustive (query)                                        32.45       717.88       750.33       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.45     2_393.98     2_426.43       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                 70.30       239.14       309.45       0.5519          1.8826            1.5884         1.78
ExhaustiveBinary-256-random-rf10 (query)                  70.30       357.25       427.55       0.9881          1.0022            1.0000         1.78
ExhaustiveBinary-256-random-rf20 (query)                  70.30       464.72       535.03       0.9980          1.0003            1.0000         1.78
ExhaustiveBinary-256-random (self)                        70.30     1_156.67     1_226.98       0.9881          1.0022            1.0000         1.78
ExhaustiveBinary-256-pca_no_rr (query)                    96.64       241.02       337.66       0.5930          1.6081            1.4152         1.78
ExhaustiveBinary-256-pca-rf10 (query)                     96.64       371.76       468.40       0.9919          1.0013            1.0000         1.78
ExhaustiveBinary-256-pca-rf20 (query)                     96.64       466.19       562.83       0.9988          1.0001            1.0000         1.78
ExhaustiveBinary-256-pca (self)                           96.64     1_163.53     1_260.17       0.9915          1.0014            1.0000         1.78
ExhaustiveBinary-512-random_no_rr (query)                 83.72       346.98       430.70       0.6306          1.5767            1.3633         3.55
ExhaustiveBinary-512-random-rf10 (query)                  83.72       469.97       553.69       0.9975          1.0004            1.0000         3.55
ExhaustiveBinary-512-random-rf20 (query)                  83.72       582.55       666.27       0.9998          1.0000            1.0000         3.55
ExhaustiveBinary-512-random (self)                        83.72     1_538.80     1_622.52       0.9973          1.0004            1.0000         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   109.56       350.50       460.07       0.6479          1.4884            1.3147         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    109.56       467.56       577.13       0.9983          1.0002            1.0000         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    109.56       580.73       690.29       0.9998          1.0000            1.0000         3.55
ExhaustiveBinary-512-pca (self)                          109.56     1_541.93     1_651.49       0.9981          1.0002            1.0000         3.55
ExhaustiveBinary-1024-random_no_rr (query)               119.81       577.70       697.51       0.6758          1.4452            1.2804         7.10
ExhaustiveBinary-1024-random-rf10 (query)                119.81       687.59       807.40       0.9995          1.0001            1.0000         7.10
ExhaustiveBinary-1024-random-rf20 (query)                119.81       806.89       926.70       0.9999          1.0000            1.0000         7.10
ExhaustiveBinary-1024-random (self)                      119.81     2_248.41     2_368.22       0.9993          1.0001            1.0000         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  144.77       542.70       687.47       0.6838          1.4142            1.2651         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   144.77       672.60       817.37       0.9996          1.0000            1.0000         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   144.77       807.16       951.92       1.0000          1.0000            1.0000         7.10
ExhaustiveBinary-1024-pca (self)                         144.77     2_240.33     2_385.09       0.9995          1.0001            1.0000         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   42.16       440.45       482.62       0.0376         19.4734           14.8778         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    42.16       479.36       521.52       0.1617          2.7567            2.6548         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    42.16       730.75       772.91       0.2739          1.9837            1.9249         1.53
ExhaustiveBinary-256-sign (self)                          42.16     1_554.87     1_597.03       0.1691          2.7353            2.6299         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)              430.83        59.68       490.51       0.5656          1.6695            1.5131         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)             430.83        69.58       500.41       0.5589          1.7299            1.5498         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)             430.83        77.64       508.46       0.5568          1.7640            1.5630         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)             430.83       127.45       558.28       0.9903          1.0018            1.0000         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)             430.83       181.96       612.79       0.9968          1.0006            1.0000         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)            430.83       134.14       564.97       0.9907          1.0016            1.0000         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)            430.83       185.76       616.59       0.9986          1.0002            1.0000         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)            430.83       133.33       564.16       0.9898          1.0018            1.0000         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)            430.83       200.25       631.07       0.9984          1.0002            1.0000         1.93
IVF-Binary-256-nl158-random (self)                       430.83       330.32       761.15       0.9904          1.0017            1.0000         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             495.57        50.39       545.96       0.5629          1.6756            1.5245         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             495.57        54.05       549.62       0.5606          1.7006            1.5423         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             495.57        62.21       557.78       0.5578          1.7444            1.5592         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            495.57       111.50       607.07       0.9912          1.0015            1.0000         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            495.57       173.27       668.85       0.9984          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            495.57       113.98       609.56       0.9909          1.0015            1.0000         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            495.57       172.82       668.39       0.9987          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            495.57       123.90       619.48       0.9900          1.0017            1.0000         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            495.57       185.97       681.55       0.9985          1.0002            1.0000         2.00
IVF-Binary-256-nl223-random (self)                       495.57       297.36       792.93       0.9908          1.0016            1.0000         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             627.19        52.23       679.42       0.5619          1.6821            1.5289         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             627.19        54.32       681.51       0.5608          1.6947            1.5368         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             627.19        61.83       689.02       0.5581          1.7362            1.5552         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            627.19       111.35       738.54       0.9917          1.0014            1.0000         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            627.19       162.18       789.36       0.9987          1.0002            1.0000         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            627.19       112.33       739.51       0.9914          1.0014            1.0000         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            627.19       165.07       792.25       0.9988          1.0002            1.0000         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            627.19       119.03       746.22       0.9904          1.0017            1.0000         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            627.19       172.52       799.70       0.9986          1.0002            1.0000         2.09
IVF-Binary-256-nl316-random (self)                       627.19       286.59       913.78       0.9912          1.0015            1.0000         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)                 448.18        49.41       497.59       0.6039          1.4880            1.3749         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)                448.18        57.73       505.91       0.5989          1.5218            1.3915         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)                448.18        77.53       525.71       0.5975          1.5419            1.3962         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)                448.18       120.95       569.13       0.9926          1.0013            1.0000         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)                448.18       164.92       613.10       0.9972          1.0006            1.0000         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)               448.18       120.24       568.42       0.9933          1.0010            1.0000         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)               448.18       175.62       623.80       0.9991          1.0001            1.0000         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)               448.18       129.88       578.06       0.9927          1.0011            1.0000         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)               448.18       188.93       637.11       0.9990          1.0001            1.0000         1.93
IVF-Binary-256-nl158-pca (self)                          448.18       326.54       774.72       0.9929          1.0011            1.0000         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                520.47        49.13       569.60       0.6019          1.4944            1.3798         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                520.47        55.32       575.80       0.6000          1.5083            1.3871         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                520.47        61.05       581.52       0.5980          1.5316            1.3961         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               520.47       111.48       631.95       0.9937          1.0010            1.0000         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               520.47       165.66       686.13       0.9987          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               520.47       115.98       636.45       0.9935          1.0010            1.0000         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               520.47       167.89       688.36       0.9991          1.0001            1.0000         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               520.47       124.92       645.39       0.9929          1.0011            1.0000         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               520.47       181.59       702.06       0.9990          1.0001            1.0000         2.00
IVF-Binary-256-nl223-pca (self)                          520.47       293.78       814.25       0.9931          1.0011            1.0000         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                649.05        51.50       700.55       0.6011          1.4974            1.3831         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                649.05        52.71       701.76       0.6002          1.5051            1.3864         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                649.05        59.88       708.93       0.5984          1.5259            1.3938         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               649.05       113.47       762.52       0.9939          1.0009            1.0000         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               649.05       165.01       814.06       0.9991          1.0001            1.0000         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               649.05       114.19       763.24       0.9938          1.0009            1.0000         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               649.05       165.15       814.20       0.9992          1.0001            1.0000         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               649.05       119.08       768.13       0.9931          1.0011            1.0000         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               649.05       173.97       823.02       0.9991          1.0001            1.0000         2.09
IVF-Binary-256-nl316-pca (self)                          649.05       280.02       929.07       0.9934          1.0011            1.0000         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)              433.30        67.99       501.29       0.6411          1.4472            1.3312         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)             433.30        82.53       515.83       0.6352          1.4895            1.3489         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)             433.30        95.79       529.08       0.6333          1.5129            1.3546         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)             433.30       134.34       567.64       0.9965          1.0007            1.0000         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)             433.30       189.25       622.55       0.9978          1.0005            1.0000         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)            433.30       148.53       581.83       0.9982          1.0002            1.0000         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)            433.30       202.71       636.00       0.9999          1.0000            1.0000         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)            433.30       164.43       597.73       0.9979          1.0003            1.0000         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)            433.30       218.79       652.09       0.9998          1.0000            1.0000         3.71
IVF-Binary-512-nl158-random (self)                       433.30       413.10       846.40       0.9979          1.0003            1.0000         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             507.71        68.35       576.06       0.6386          1.4527            1.3378         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             507.71        74.81       582.52       0.6366          1.4693            1.3444         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             507.71        85.94       593.65       0.6340          1.5000            1.3529         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            507.71       131.21       638.92       0.9978          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            507.71       182.01       689.72       0.9993          1.0001            1.0000         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            507.71       135.03       642.75       0.9980          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            507.71       201.58       709.30       0.9998          1.0000            1.0000         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            507.71       147.91       655.62       0.9979          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            507.71       203.55       711.26       0.9998          1.0000            1.0000         3.77
IVF-Binary-512-nl223-random (self)                       507.71       371.51       879.22       0.9978          1.0003            1.0000         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             642.87        83.62       726.49       0.6377          1.4609            1.3400         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             642.87        75.50       718.37       0.6368          1.4691            1.3434         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             642.87        82.64       725.51       0.6346          1.4956            1.3511         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            642.87       130.81       773.68       0.9981          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            642.87       188.94       831.80       0.9996          1.0001            1.0000         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            642.87       133.10       775.96       0.9982          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            642.87       186.43       829.30       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            642.87       141.96       784.83       0.9980          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            642.87       198.67       841.54       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-random (self)                       642.87       351.81       994.67       0.9980          1.0003            1.0000         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)                 459.04        68.17       527.21       0.6577          1.3876            1.2898         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)                459.04        82.81       541.85       0.6524          1.4211            1.3034         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)                459.04        99.84       558.88       0.6509          1.4401            1.3080         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)                459.04       134.20       593.24       0.9969          1.0006            1.0000         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)                459.04       187.78       646.82       0.9978          1.0005            1.0000         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)               459.04       147.10       606.15       0.9987          1.0001            1.0000         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)               459.04       204.48       663.52       0.9999          1.0000            1.0000         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)               459.04       169.28       628.32       0.9985          1.0002            1.0000         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)               459.04       220.18       679.22       0.9999          1.0000            1.0000         3.71
IVF-Binary-512-nl158-pca (self)                          459.04       412.92       871.96       0.9986          1.0002            1.0000         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                533.05        69.51       602.56       0.6552          1.3961            1.2931         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                533.05        73.90       606.95       0.6533          1.4086            1.2988         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                533.05        85.13       618.18       0.6513          1.4315            1.3057         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               533.05       133.98       667.03       0.9983          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               533.05       182.04       715.10       0.9993          1.0001            1.0000         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               533.05       135.11       668.16       0.9986          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               533.05       191.29       724.34       0.9998          1.0000            1.0000         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               533.05       150.88       683.93       0.9985          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               533.05       200.68       733.74       0.9999          1.0000            1.0000         3.77
IVF-Binary-512-nl223-pca (self)                          533.05       366.75       899.80       0.9985          1.0002            1.0000         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                674.31        71.15       745.47       0.6544          1.4017            1.2942         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                674.31        72.65       746.96       0.6536          1.4083            1.2967         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                674.31        81.95       756.27       0.6516          1.4283            1.3033         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               674.31       132.15       806.46       0.9986          1.0002            1.0000         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               674.31       181.61       855.92       0.9996          1.0001            1.0000         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               674.31       134.38       808.69       0.9987          1.0001            1.0000         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               674.31       186.22       860.53       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               674.31       142.83       817.15       0.9986          1.0002            1.0000         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               674.31       204.22       878.53       0.9999          1.0000            1.0000         3.86
IVF-Binary-512-nl316-pca (self)                          674.31       352.28     1_026.59       0.9986          1.0002            1.0000         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)             465.80       102.12       567.93       0.6845          1.3518            1.2576         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)            465.80       121.67       587.47       0.6792          1.3861            1.2707         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)            465.80       141.63       607.43       0.6776          1.4035            1.2751         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)            465.80       173.97       639.77       0.9977          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)            465.80       225.46       691.26       0.9978          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)           465.80       187.89       653.69       0.9997          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)           465.80       248.95       714.75       0.9999          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)           465.80       212.18       677.98       0.9996          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)           465.80       271.55       737.35       1.0000          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-random (self)                      465.80       566.98     1_032.78       0.9995          1.0000            1.0000         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            539.87       103.61       643.48       0.6825          1.3585            1.2601         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            539.87       109.01       648.88       0.6806          1.3719            1.2667         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            539.87       125.00       664.87       0.6784          1.3948            1.2738         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           539.87       168.65       708.52       0.9991          1.0002            1.0000         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           539.87       229.03       768.90       0.9994          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           539.87       174.43       714.29       0.9995          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           539.87       230.61       770.47       0.9998          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           539.87       194.78       734.64       0.9996          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           539.87       248.76       788.63       0.9999          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-random (self)                      539.87       503.25     1_043.12       0.9994          1.0001            1.0000         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            674.23       105.63       779.86       0.6815          1.3670            1.2637         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)            674.23       110.35       784.58       0.6806          1.3736            1.2657         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)            674.23       135.61       809.84       0.6786          1.3933            1.2725         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)           674.23       202.93       877.16       0.9994          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)           674.23       234.60       908.83       0.9997          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)           674.23       177.06       851.29       0.9995          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)           674.23       233.55       907.78       0.9998          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)           674.23       209.52       883.75       0.9996          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)           674.23       247.80       922.03       1.0000          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-random (self)                      674.23       488.26     1_162.49       0.9994          1.0001            1.0000         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)                523.68       105.73       629.41       0.6928          1.3299            1.2427         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)               523.68       129.06       652.74       0.6877          1.3593            1.2559         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)               523.68       145.93       669.61       0.6860          1.3758            1.2595         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)               523.68       198.58       722.26       0.9977          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)               523.68       231.80       755.48       0.9978          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)              523.68       195.15       718.83       0.9998          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)              523.68       257.33       781.01       1.0000          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)              523.68       221.43       745.11       0.9997          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)              523.68       277.95       801.63       1.0000          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-pca (self)                         523.68       580.68     1_104.36       0.9997          1.0000            1.0000         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               572.04       104.62       676.66       0.6902          1.3389            1.2452         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               572.04       113.09       685.13       0.6884          1.3495            1.2507         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               572.04       126.90       698.94       0.6863          1.3695            1.2573         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              572.04       167.08       739.11       0.9992          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              572.04       234.87       806.91       0.9994          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              572.04       173.78       745.81       0.9996          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              572.04       236.63       808.66       0.9999          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              572.04       198.82       770.85       0.9996          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              572.04       262.17       834.20       1.0000          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-pca (self)                         572.04       502.62     1_074.66       0.9995          1.0001            1.0000         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               707.43       108.69       816.12       0.6892          1.3446            1.2492         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)               707.43       110.06       817.49       0.6884          1.3504            1.2513         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)               707.43       124.30       831.72       0.6867          1.3679            1.2566         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)              707.43       175.36       882.78       0.9995          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)              707.43       238.78       946.21       0.9997          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)              707.43       173.72       881.14       0.9996          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)              707.43       239.36       946.78       0.9998          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)              707.43       185.03       892.45       0.9997          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)              707.43       251.76       959.18       1.0000          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-pca (self)                         707.43       494.63     1_202.05       0.9995          1.0001            1.0000         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)                408.95       179.45       588.40       0.0687          6.6445            6.1205         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               408.95       193.15       602.09       0.0554          7.8737            7.1934         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               408.95       210.96       619.90       0.0506          8.7363            7.8879         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               408.95       218.42       627.37       0.3993          1.6143            1.5273         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               408.95       396.67       805.62       0.6370          1.2496            1.1927         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              408.95       235.75       644.69       0.3090          1.8465            1.7466         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              408.95       414.54       823.49       0.4807          1.4429            1.3742         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              408.95       246.02       654.97       0.2738          1.9854            1.8794         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              408.95       432.93       841.87       0.4129          1.5612            1.4833         1.68
IVF-Binary-256-nl158-sign (self)                         408.95       701.70     1_110.64       0.3147          1.8346            1.7365         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               472.05       171.80       643.85       0.0659          6.5535            6.0912         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               472.05       176.09       648.13       0.0606          6.9940            6.4957         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               472.05       187.76       659.81       0.0540          7.8607            7.2719         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              472.05       218.72       690.76       0.3659          1.6641            1.5816         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              472.05       373.75       845.80       0.5968          1.2835            1.2268         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              472.05       216.42       688.47       0.3334          1.7511            1.6659         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              472.05       391.83       863.88       0.5313          1.3595            1.2996         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              472.05       227.68       699.73       0.2921          1.9003            1.8127         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              472.05       420.21       892.26       0.4434          1.4929            1.4296         1.75
IVF-Binary-256-nl223-sign (self)                         472.05       676.47     1_148.51       0.3386          1.7410            1.6559         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               627.01       168.28       795.29       0.0661          6.4956            6.0418         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               627.01       172.26       799.27       0.0633          6.7181            6.2205         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               627.01       179.88       806.89       0.0564          7.4857            6.8850         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              627.01       217.25       844.26       0.3650          1.6701            1.5867         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              627.01       377.95     1_004.97       0.5869          1.2940            1.2404         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              627.01       211.98       838.99       0.3481          1.7146            1.6271         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              627.01       384.00     1_011.01       0.5539          1.3321            1.2772         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              627.01       234.34       861.35       0.3059          1.8520            1.7609         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              627.01       396.59     1_023.60       0.4670          1.4544            1.3942         1.84
IVF-Binary-256-nl316-sign (self)                         627.01       633.60     1_260.61       0.3541          1.7014            1.6169         1.84
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
Exhaustive (query)                                        68.00     1_378.09     1_446.09       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.00     4_568.66     4_636.66       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                134.96       266.12       401.09       0.5547          1.7646            1.5366         2.03
ExhaustiveBinary-256-random-rf10 (query)                 134.96       417.49       552.46       0.9898          1.0017            1.0000         2.03
ExhaustiveBinary-256-random-rf20 (query)                 134.96       547.51       682.47       0.9985          1.0002            1.0000         2.03
ExhaustiveBinary-256-random (self)                       134.96     1_298.56     1_433.53       0.9899          1.0016            1.0000         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   227.10       269.28       496.39       0.5767          1.6243            1.4311         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    227.10       418.37       645.47       0.9904          1.0016            1.0000         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    227.10       552.50       779.61       0.9984          1.0002            1.0000         2.03
ExhaustiveBinary-256-pca (self)                          227.10     1_305.66     1_532.76       0.9905          1.0016            1.0000         2.03
ExhaustiveBinary-512-random_no_rr (query)                211.09       394.15       605.23       0.6013          1.6760            1.4608         4.05
ExhaustiveBinary-512-random-rf10 (query)                 211.09       541.18       752.27       0.9977          1.0003            1.0000         4.05
ExhaustiveBinary-512-random-rf20 (query)                 211.09       679.31       890.40       0.9998          1.0000            1.0000         4.05
ExhaustiveBinary-512-random (self)                       211.09     1_760.66     1_971.74       0.9975          1.0003            1.0000         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   293.00       390.88       683.87       0.6443          1.4426            1.3064         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    293.00       547.29       840.29       0.9985          1.0002            1.0000         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    293.00       691.10       984.10       0.9999          1.0000            1.0000         4.05
ExhaustiveBinary-512-pca (self)                          293.00     1_752.70     2_045.70       0.9984          1.0002            1.0000         4.05
ExhaustiveBinary-1024-random_no_rr (query)               257.17       584.79       841.96       0.6624          1.4553            1.3048         8.11
ExhaustiveBinary-1024-random-rf10 (query)                257.17       745.90     1_003.08       0.9995          1.0001            1.0000         8.11
ExhaustiveBinary-1024-random-rf20 (query)                257.17       900.25     1_157.42       1.0000          1.0000            1.0000         8.11
ExhaustiveBinary-1024-random (self)                      257.17     2_467.90     2_725.08       0.9994          1.0001            1.0000         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  349.52       607.71       957.23       0.6865          1.3603            1.2383         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   349.52       762.26     1_111.78       0.9996          1.0001            1.0000         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   349.52       924.21     1_273.73       1.0000          1.0000            1.0000         8.11
ExhaustiveBinary-1024-pca (self)                         349.52     2_463.44     2_812.96       0.9996          1.0001            1.0000         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   85.85       655.85       741.70       0.0400         18.1511           13.6734         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    85.85       722.14       807.99       0.1821          2.5573            2.4620         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    85.85     1_105.48     1_191.33       0.3140          1.8429            1.7786         3.05
ExhaustiveBinary-512-sign (self)                          85.85     2_334.82     2_420.67       0.1897          2.5286            2.4283         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)              633.37        93.61       726.97       0.5627          1.6434            1.4885         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)             633.37        93.88       727.25       0.5594          1.6750            1.5064         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)             633.37       103.72       737.09       0.5580          1.7034            1.5141         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)             633.37       175.49       808.85       0.9915          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)             633.37       266.75       900.12       0.9978          1.0004            1.0000         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)            633.37       181.95       815.32       0.9913          1.0013            1.0000         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)            633.37       287.32       920.69       0.9988          1.0001            1.0000         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)            633.37       191.92       825.29       0.9907          1.0015            1.0000         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)            633.37       287.61       920.98       0.9987          1.0002            1.0000         2.34
IVF-Binary-256-nl158-random (self)                       633.37       411.28     1_044.65       0.9914          1.0014            1.0000         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)             666.00        79.95       745.96       0.5616          1.6456            1.4978         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)             666.00        80.92       746.92       0.5604          1.6579            1.5060         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)             666.00        88.06       754.06       0.5590          1.6819            1.5132         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)            666.00       176.03       842.03       0.9924          1.0011            1.0000         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)            666.00       264.68       930.68       0.9989          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)            666.00       180.77       846.77       0.9919          1.0012            1.0000         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)            666.00       274.35       940.36       0.9989          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)            666.00       182.97       848.97       0.9911          1.0014            1.0000         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)            666.00       276.42       942.42       0.9988          1.0001            1.0000         2.47
IVF-Binary-256-nl223-random (self)                       666.00       377.90     1_043.90       0.9920          1.0012            1.0000         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)             776.07        82.64       858.71       0.5614          1.6442            1.4960         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)             776.07        86.13       862.20       0.5608          1.6514            1.4997         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)             776.07        90.34       866.42       0.5594          1.6719            1.5089         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)            776.07       175.47       951.54       0.9924          1.0011            1.0000         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)            776.07       269.05     1_045.13       0.9990          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)            776.07       174.79       950.86       0.9921          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)            776.07       270.20     1_046.28       0.9990          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)            776.07       213.20       989.27       0.9913          1.0013            1.0000         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)            776.07       281.08     1_057.15       0.9988          1.0001            1.0000         2.65
IVF-Binary-256-nl316-random (self)                       776.07       380.98     1_157.06       0.9922          1.0012            1.0000         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)                 721.84        74.35       796.19       0.5835          1.5315            1.4056         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)                721.84        82.73       804.57       0.5809          1.5555            1.4149         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)                721.84        92.54       814.38       0.5799          1.5748            1.4194         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)                721.84       173.69       895.53       0.9916          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)                721.84       259.93       981.77       0.9978          1.0004            1.0000         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)               721.84       175.15       896.99       0.9916          1.0013            1.0000         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)               721.84       272.73       994.57       0.9988          1.0002            1.0000         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)               721.84       183.46       905.30       0.9910          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)               721.84       285.02     1_006.86       0.9987          1.0002            1.0000         2.34
IVF-Binary-256-nl158-pca (self)                          721.84       409.90     1_131.74       0.9916          1.0014            1.0000         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)                760.34        77.57       837.91       0.5830          1.5327            1.4106         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)                760.34        82.49       842.83       0.5818          1.5424            1.4155         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)                760.34        88.81       849.16       0.5807          1.5585            1.4200         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)               760.34       173.99       934.34       0.9924          1.0012            1.0000         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)               760.34       264.29     1_024.63       0.9988          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)               760.34       172.58       932.92       0.9921          1.0013            1.0000         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)               760.34       273.03     1_033.37       0.9989          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)               760.34       180.37       940.71       0.9914          1.0014            1.0000         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)               760.34       274.19     1_034.53       0.9987          1.0002            1.0000         2.47
IVF-Binary-256-nl223-pca (self)                          760.34       378.30     1_138.64       0.9921          1.0013            1.0000         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)                872.08        81.88       953.96       0.5827          1.5322            1.4087         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)                872.08        85.35       957.43       0.5823          1.5369            1.4103         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)                872.08        90.88       962.96       0.5810          1.5507            1.4178         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)               872.08       179.13     1_051.22       0.9923          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)               872.08       267.06     1_139.15       0.9990          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)               872.08       175.23     1_047.32       0.9920          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)               872.08       271.77     1_143.85       0.9989          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)               872.08       180.25     1_052.34       0.9914          1.0014            1.0000         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)               872.08       277.64     1_149.72       0.9988          1.0002            1.0000         2.65
IVF-Binary-256-nl316-pca (self)                          872.08       381.55     1_253.63       0.9922          1.0012            1.0000         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)              711.57       106.53       818.11       0.6084          1.5746            1.4241         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)             711.57       123.51       835.09       0.6049          1.6062            1.4408         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)             711.57       134.41       845.98       0.6033          1.6340            1.4480         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)             711.57       211.86       923.44       0.9972          1.0004            1.0000         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)             711.57       294.80     1_006.38       0.9985          1.0003            1.0000         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)            711.57       213.39       924.97       0.9981          1.0002            1.0000         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)            711.57       310.56     1_022.13       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)            711.57       225.46       937.03       0.9979          1.0003            1.0000         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)            711.57       323.32     1_034.89       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-random (self)                       711.57       545.96     1_257.53       0.9979          1.0003            1.0000         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)             738.10       111.62       849.72       0.6072          1.5782            1.4305         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)             738.10       116.26       854.36       0.6057          1.5924            1.4386         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)             738.10       126.86       864.96       0.6043          1.6132            1.4454         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)            738.10       205.13       943.23       0.9982          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)            738.10       297.02     1_035.12       0.9996          1.0000            1.0000         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)            738.10       203.94       942.04       0.9983          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)            738.10       302.54     1_040.64       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)            738.10       215.33       953.44       0.9981          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)            738.10       312.98     1_051.09       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-random (self)                       738.10       492.84     1_230.94       0.9982          1.0002            1.0000         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)             841.31       111.73       953.05       0.6067          1.5793            1.4339         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)             841.31       114.11       955.42       0.6061          1.5861            1.4377         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)             841.31       125.31       966.63       0.6044          1.6057            1.4454         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)            841.31       207.76     1_049.07       0.9985          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)            841.31       297.83     1_139.15       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)            841.31       201.88     1_043.19       0.9984          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)            841.31       299.17     1_140.48       0.9999          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)            841.31       213.35     1_054.66       0.9982          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)            841.31       314.98     1_156.30       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-random (self)                       841.31       486.19     1_327.50       0.9983          1.0002            1.0000         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)                 797.37       100.56       897.93       0.6496          1.3874            1.2888         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)                797.37       115.30       912.67       0.6470          1.4043            1.2978         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)                797.37       126.58       923.95       0.6459          1.4173            1.3008         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)                797.37       204.08     1_001.45       0.9975          1.0004            1.0000         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)                797.37       299.71     1_097.08       0.9985          1.0003            1.0000         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)               797.37       208.16     1_005.54       0.9986          1.0001            1.0000         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)               797.37       309.59     1_106.97       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)               797.37       220.21     1_017.58       0.9986          1.0002            1.0000         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)               797.37       321.83     1_119.20       0.9999          1.0000            1.0000         4.36
IVF-Binary-512-nl158-pca (self)                          797.37       543.80     1_341.17       0.9986          1.0002            1.0000         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)                836.63       107.10       943.72       0.6483          1.3915            1.2914         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)                836.63       109.62       946.25       0.6473          1.3993            1.2956         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)                836.63       124.55       961.18       0.6464          1.4090            1.2987         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)               836.63       198.97     1_035.60       0.9986          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)               836.63       291.29     1_127.91       0.9996          1.0001            1.0000         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)               836.63       203.31     1_039.93       0.9987          1.0001            1.0000         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)               836.63       297.95     1_134.57       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)               836.63       213.08     1_049.70       0.9986          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)               836.63       311.34     1_147.97       0.9999          1.0000            1.0000         4.49
IVF-Binary-512-nl223-pca (self)                          836.63       492.15     1_328.77       0.9987          1.0002            1.0000         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)                935.45       109.97     1_045.43       0.6480          1.3927            1.2940         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)                935.45       111.62     1_047.07       0.6475          1.3963            1.2954         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)                935.45       121.27     1_056.72       0.6465          1.4067            1.2990         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)               935.45       208.84     1_144.29       0.9988          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)               935.45       296.81     1_232.27       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)               935.45       202.84     1_138.29       0.9987          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)               935.45       300.16     1_235.61       0.9999          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)               935.45       217.94     1_153.40       0.9986          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)               935.45       312.88     1_248.33       0.9999          1.0000            1.0000         4.67
IVF-Binary-512-nl316-pca (self)                          935.45       496.72     1_432.18       0.9987          1.0001            1.0000         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)             779.35       155.43       934.78       0.6681          1.3933            1.2856         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)            779.35       176.84       956.19       0.6650          1.4133            1.2952         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)            779.35       195.12       974.47       0.6636          1.4310            1.3003         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)            779.35       266.97     1_046.32       0.9983          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)            779.35       350.65     1_130.00       0.9986          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)           779.35       269.95     1_049.31       0.9995          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)           779.35       375.91     1_155.26       0.9999          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)           779.35       290.37     1_069.72       0.9995          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)           779.35       404.56     1_183.91       1.0000          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-random (self)                      779.35       763.90     1_543.25       0.9995          1.0001            1.0000         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)            793.15       161.59       954.75       0.6666          1.3986            1.2895         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)            793.15       172.39       965.55       0.6654          1.4076            1.2936         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)            793.15       183.87       977.03       0.6643          1.4203            1.2986         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)           793.15       254.62     1_047.77       0.9994          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)           793.15       359.49     1_152.64       0.9997          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)           793.15       261.11     1_054.26       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)           793.15       370.16     1_163.31       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)           793.15       279.48     1_072.63       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)           793.15       390.41     1_183.57       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-random (self)                      793.15       717.21     1_510.36       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)            905.29       167.66     1_072.94       0.6664          1.3999            1.2907         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)            905.29       169.31     1_074.60       0.6659          1.4040            1.2926         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)            905.29       182.39     1_087.68       0.6646          1.4168            1.2972         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)           905.29       260.28     1_165.56       0.9996          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)           905.29       365.27     1_270.56       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)           905.29       261.79     1_167.08       0.9996          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)           905.29       390.40     1_295.69       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)           905.29       276.96     1_182.25       0.9996          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)           905.29       389.79     1_295.07       1.0000          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-random (self)                      905.29       706.66     1_611.95       0.9996          1.0001            1.0000         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)                848.65       156.90     1_005.55       0.6909          1.3175            1.2259         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)               848.65       173.65     1_022.30       0.6885          1.3324            1.2316         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)               848.65       193.48     1_042.13       0.6876          1.3436            1.2346         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)               848.65       261.51     1_110.16       0.9984          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)               848.65       354.42     1_203.08       0.9986          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)              848.65       272.14     1_120.79       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)              848.65       378.87     1_227.53       0.9999          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)              848.65       297.38     1_146.04       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)              848.65       405.91     1_254.56       1.0000          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-pca (self)                         848.65       769.13     1_617.78       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)               907.69       162.89     1_070.59       0.6897          1.3214            1.2293         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)               907.69       166.74     1_074.43       0.6887          1.3273            1.2309         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)               907.69       183.79     1_091.48       0.6878          1.3368            1.2338         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)              907.69       256.38     1_164.07       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)              907.69       360.16     1_267.86       0.9997          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)              907.69       262.65     1_170.34       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)              907.69       368.06     1_275.75       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)              907.69       281.29     1_188.98       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)              907.69       390.26     1_297.95       1.0000          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-pca (self)                         907.69       719.53     1_627.22       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)               997.82       167.99     1_165.80       0.6897          1.3225            1.2300         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)               997.82       169.16     1_166.97       0.6892          1.3257            1.2309         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)               997.82       181.70     1_179.51       0.6881          1.3346            1.2341         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)              997.82       263.13     1_260.95       0.9997          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)              997.82       376.50     1_374.32       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)              997.82       265.16     1_262.98       0.9997          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)              997.82       376.69     1_374.50       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)              997.82       286.77     1_284.59       0.9997          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)              997.82       393.57     1_391.39       1.0000          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-pca (self)                         997.82       731.01     1_728.82       0.9996          1.0000            1.0000         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)                589.96       292.06       882.02       0.0587          7.9788            7.1475         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)               589.96       306.86       896.82       0.0517          9.1464            7.9827         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)               589.96       329.16       919.12       0.0486         10.0153            8.6481         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)               589.96       362.29       952.25       0.3222          1.8338            1.7364         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)               589.96       654.94     1_244.90       0.5149          1.4031            1.3349         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)              589.96       380.35       970.31       0.2801          1.9786            1.8635         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)              589.96       674.35     1_264.31       0.4400          1.5255            1.4474         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)              589.96       414.20     1_004.16       0.2590          2.0739            1.9494         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)              589.96       701.33     1_291.29       0.3996          1.6072            1.5243         3.36
IVF-Binary-512-nl158-sign (self)                         589.96     1_090.57     1_680.53       0.2870          1.9578            1.8406         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               617.72       291.21       908.93       0.0570          7.8806            7.1787         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               617.72       299.36       917.07       0.0540          8.2916            7.4994         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               617.72       308.83       926.54       0.0499          9.1818            8.1707         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              617.72       364.16       981.88       0.3139          1.8522            1.7521         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              617.72       649.45     1_267.17       0.5026          1.4175            1.3460         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              617.72       375.93       993.65       0.2945          1.9145            1.8086         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              617.72       679.68     1_297.40       0.4691          1.4689            1.3964         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              617.72       386.93     1_004.64       0.2686          2.0220            1.8987         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              617.72       694.40     1_312.12       0.4170          1.5688            1.4883         3.49
IVF-Binary-512-nl223-sign (self)                         617.72     1_028.36     1_646.07       0.3016          1.8962            1.7845         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)               741.45       298.43     1_039.88       0.0576          7.7316            7.1191         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)               741.45       294.02     1_035.47       0.0558          7.9302            7.2945         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)               741.45       307.19     1_048.64       0.0519          8.6360            7.8466         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)              741.45       366.80     1_108.26       0.3181          1.8372            1.7369         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)              741.45       661.34     1_402.80       0.5020          1.4131            1.3434         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)              741.45       369.17     1_110.63       0.3081          1.8711            1.7676         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)              741.45       658.33     1_399.78       0.4843          1.4392            1.3711         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)              741.45       384.47     1_125.93       0.2819          1.9709            1.8521         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)              741.45       685.45     1_426.90       0.4341          1.5294            1.4542         3.67
IVF-Binary-512-nl316-sign (self)                         741.45     1_030.52     1_771.97       0.3138          1.8534            1.7425         3.67
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
Exhaustive (query)                                       102.52     1_926.85     2_029.36       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        102.52     6_533.29     6_635.81       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                196.40       288.63       485.04       0.5361          1.8068            1.5908         2.28
ExhaustiveBinary-256-random-rf10 (query)                 196.40       451.10       647.50       0.9868          1.0022            1.0000         2.28
ExhaustiveBinary-256-random-rf20 (query)                 196.40       603.76       800.17       0.9980          1.0003            1.0000         2.28
ExhaustiveBinary-256-random (self)                       196.40     1_397.45     1_593.85       0.9876          1.0021            1.0000         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   405.97       287.15       693.13       0.5754          1.5495            1.4128         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    405.97       451.43       857.41       0.9895          1.0018            1.0000         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    405.97       602.59     1_008.57       0.9983          1.0002            1.0000         2.28
ExhaustiveBinary-256-pca (self)                          405.97     1_392.45     1_798.43       0.9897          1.0017            1.0000         2.28
ExhaustiveBinary-512-random_no_rr (query)                302.98       407.05       710.03       0.5866          1.6778            1.4946         4.55
ExhaustiveBinary-512-random-rf10 (query)                 302.98       574.51       877.49       0.9966          1.0005            1.0000         4.55
ExhaustiveBinary-512-random-rf20 (query)                 302.98       749.88     1_052.85       0.9997          1.0001            1.0000         4.55
ExhaustiveBinary-512-random (self)                       302.98     1_857.69     2_160.67       0.9969          1.0004            1.0000         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   503.21       410.20       913.41       0.6388          1.4217            1.3032         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    503.21       575.90     1_079.11       0.9979          1.0003            1.0000         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    503.21       739.93     1_243.14       0.9998          1.0000            1.0000         4.55
ExhaustiveBinary-512-pca (self)                          503.21     1_846.14     2_349.35       0.9981          1.0002            1.0000         4.55
ExhaustiveBinary-1024-random_no_rr (query)               502.72       638.88     1_141.60       0.6446          1.4909            1.3512         9.11
ExhaustiveBinary-1024-random-rf10 (query)                502.72       844.84     1_347.57       0.9993          1.0001            1.0000         9.11
ExhaustiveBinary-1024-random-rf20 (query)                502.72       996.96     1_499.68       0.9999          1.0000            1.0000         9.11
ExhaustiveBinary-1024-random (self)                      502.72     2_691.26     3_193.98       0.9994          1.0001            1.0000         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  701.53       645.96     1_347.49       0.6795          1.3452            1.2483         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   701.53       822.15     1_523.68       0.9996          1.0001            1.0000         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   701.53     1_005.61     1_707.14       1.0000          1.0000            1.0000         9.11
ExhaustiveBinary-1024-pca (self)                         701.53     2_696.78     3_398.31       0.9997          1.0000            1.0000         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  132.26       831.92       964.18       0.0420         17.7082           13.0970         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   132.26       912.71     1_044.96       0.1896          2.5240            2.4052         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   132.26     1_580.20     1_712.46       0.3229          1.8300            1.7348         4.58
ExhaustiveBinary-768-sign (self)                         132.26     3_219.04     3_351.30       0.1997          2.4832            2.3546         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)            1_147.46       109.07     1_256.52       0.5429          1.7099            1.5461         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)           1_147.46       120.62     1_268.07       0.5408          1.7331            1.5595         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)           1_147.46       126.72     1_274.18       0.5397          1.7545            1.5688         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)           1_147.46       238.58     1_386.04       0.9891          1.0018            1.0000         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)           1_147.46       335.29     1_482.74       0.9984          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)          1_147.46       219.27     1_366.72       0.9884          1.0019            1.0000         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)          1_147.46       344.89     1_492.34       0.9986          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)          1_147.46       219.62     1_367.08       0.9877          1.0021            1.0000         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)          1_147.46       340.90     1_488.36       0.9983          1.0002            1.0000         2.74
IVF-Binary-256-nl158-random (self)                     1_147.46       489.57     1_637.03       0.9891          1.0018            1.0000         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)           1_095.25        97.47     1_192.73       0.5419          1.7184            1.5524         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)           1_095.25       101.73     1_196.98       0.5412          1.7280            1.5571         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)           1_095.25       116.63     1_211.88       0.5401          1.7511            1.5659         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)          1_095.25       210.60     1_305.85       0.9888          1.0018            1.0000         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)          1_095.25       336.64     1_431.89       0.9985          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)          1_095.25       212.66     1_307.92       0.9885          1.0019            1.0000         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)          1_095.25       342.33     1_437.59       0.9985          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)          1_095.25       218.13     1_313.39       0.9877          1.0020            1.0000         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)          1_095.25       340.38     1_435.64       0.9983          1.0002            1.0000         2.93
IVF-Binary-256-nl223-random (self)                     1_095.25       477.23     1_572.48       0.9890          1.0018            1.0000         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)           1_375.03       106.37     1_481.39       0.5422          1.7108            1.5502         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)           1_375.03       107.82     1_482.84       0.5418          1.7157            1.5534         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)           1_375.03       113.26     1_488.29       0.5409          1.7314            1.5597         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)          1_375.03       216.28     1_591.31       0.9891          1.0018            1.0000         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)          1_375.03       338.83     1_713.85       0.9987          1.0002            1.0000         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)          1_375.03       216.94     1_591.97       0.9888          1.0018            1.0000         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)          1_375.03       342.55     1_717.58       0.9986          1.0002            1.0000         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)          1_375.03       225.13     1_600.15       0.9882          1.0020            1.0000         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)          1_375.03       348.96     1_723.99       0.9984          1.0002            1.0000         3.21
IVF-Binary-256-nl316-random (self)                     1_375.03       497.56     1_872.58       0.9894          1.0017            1.0000         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)               1_174.97        90.91     1_265.88       0.5812          1.4959            1.3935         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)              1_174.97       122.00     1_296.97       0.5794          1.5088            1.3993         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)              1_174.97       106.99     1_281.96       0.5787          1.5186            1.4018         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)              1_174.97       211.77     1_386.73       0.9913          1.0014            1.0000         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)              1_174.97       322.96     1_497.93       0.9984          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)             1_174.97       211.82     1_386.79       0.9907          1.0015            1.0000         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)             1_174.97       336.45     1_511.42       0.9987          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)             1_174.97       218.58     1_393.55       0.9902          1.0016            1.0000         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)             1_174.97       339.78     1_514.75       0.9985          1.0002            1.0000         2.74
IVF-Binary-256-nl158-pca (self)                        1_174.97       507.37     1_682.34       0.9910          1.0014            1.0000         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)              1_300.91       100.43     1_401.34       0.5800          1.5020            1.3965         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)              1_300.91       100.73     1_401.64       0.5795          1.5067            1.3989         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)              1_300.91       107.73     1_408.64       0.5787          1.5168            1.4019         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)             1_300.91       213.11     1_514.02       0.9909          1.0015            1.0000         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)             1_300.91       331.18     1_632.09       0.9987          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)             1_300.91       212.32     1_513.24       0.9906          1.0015            1.0000         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)             1_300.91       333.04     1_633.96       0.9987          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)             1_300.91       220.42     1_521.33       0.9901          1.0016            1.0000         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)             1_300.91       341.14     1_642.05       0.9985          1.0002            1.0000         2.93
IVF-Binary-256-nl223-pca (self)                        1_300.91       470.02     1_770.93       0.9909          1.0014            1.0000         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)              1_568.13       108.18     1_676.31       0.5807          1.4991            1.3929         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)              1_568.13       108.58     1_676.70       0.5804          1.5019            1.3943         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)              1_568.13       114.03     1_682.16       0.5799          1.5084            1.3972         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)             1_568.13       225.24     1_793.37       0.9912          1.0014            1.0000         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)             1_568.13       336.92     1_905.05       0.9989          1.0001            1.0000         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)             1_568.13       217.87     1_786.00       0.9909          1.0015            1.0000         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)             1_568.13       338.74     1_906.87       0.9988          1.0001            1.0000         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)             1_568.13       223.20     1_791.32       0.9903          1.0016            1.0000         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)             1_568.13       351.16     1_919.28       0.9986          1.0002            1.0000         3.21
IVF-Binary-256-nl316-pca (self)                        1_568.13       489.29     2_057.41       0.9913          1.0014            1.0000         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)            1_098.53       130.98     1_229.51       0.5925          1.6007            1.4618         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)           1_098.53       149.18     1_247.71       0.5899          1.6229            1.4748         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)           1_098.53       164.36     1_262.89       0.5889          1.6417            1.4805         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)           1_098.53       245.63     1_344.16       0.9972          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)           1_098.53       364.63     1_463.16       0.9993          1.0001            1.0000         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)          1_098.53       256.99     1_355.52       0.9972          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)          1_098.53       385.05     1_483.58       0.9998          1.0000            1.0000         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)          1_098.53       268.34     1_366.87       0.9970          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)          1_098.53       421.85     1_520.38       0.9997          1.0000            1.0000         5.02
IVF-Binary-512-nl158-random (self)                     1_098.53       652.31     1_750.84       0.9974          1.0003            1.0000         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)           1_206.56       137.15     1_343.71       0.5912          1.6104            1.4673         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)           1_206.56       140.20     1_346.77       0.5903          1.6195            1.4721         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)           1_206.56       154.64     1_361.20       0.5890          1.6407            1.4796         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)          1_206.56       250.29     1_456.85       0.9972          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)          1_206.56       378.47     1_585.03       0.9996          1.0001            1.0000         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)          1_206.56       253.12     1_459.68       0.9972          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)          1_206.56       386.18     1_592.74       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)          1_206.56       265.27     1_471.83       0.9969          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)          1_206.56       396.89     1_603.45       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-random (self)                     1_206.56       627.17     1_833.74       0.9974          1.0003            1.0000         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)           1_497.84       146.01     1_643.85       0.5911          1.6073            1.4674         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)           1_497.84       149.13     1_646.97       0.5906          1.6115            1.4698         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)           1_497.84       156.57     1_654.41       0.5897          1.6270            1.4771         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)          1_497.84       260.38     1_758.22       0.9974          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)          1_497.84       388.86     1_886.70       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)          1_497.84       262.45     1_760.29       0.9973          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)          1_497.84       389.19     1_887.03       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)          1_497.84       270.64     1_768.48       0.9971          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)          1_497.84       411.73     1_909.57       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-random (self)                     1_497.84       653.95     2_151.79       0.9976          1.0003            1.0000         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)               1_295.69       130.83     1_426.52       0.6432          1.3824            1.2890         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)              1_295.69       143.18     1_438.87       0.6414          1.3937            1.2943         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)              1_295.69       157.99     1_453.68       0.6407          1.4029            1.2975         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)              1_295.69       258.20     1_553.89       0.9980          1.0002            1.0000         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)              1_295.69       392.20     1_687.89       0.9994          1.0001            1.0000         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)             1_295.69       262.29     1_557.98       0.9982          1.0002            1.0000         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)             1_295.69       384.39     1_680.07       0.9999          1.0000            1.0000         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)             1_295.69       271.75     1_567.44       0.9981          1.0003            1.0000         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)             1_295.69       398.22     1_693.91       0.9998          1.0000            1.0000         5.02
IVF-Binary-512-nl158-pca (self)                        1_295.69       653.72     1_949.41       0.9983          1.0002            1.0000         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_409.03       141.30     1_550.33       0.6420          1.3886            1.2933         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_409.03       141.30     1_550.33       0.6414          1.3936            1.2957         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_409.03       153.35     1_562.38       0.6407          1.4030            1.2985         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_409.03       253.18     1_662.21       0.9982          1.0002            1.0000         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_409.03       375.55     1_784.58       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_409.03       252.75     1_661.78       0.9982          1.0002            1.0000         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_409.03       380.42     1_789.45       0.9998          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_409.03       263.16     1_672.19       0.9981          1.0003            1.0000         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_409.03       393.92     1_802.95       0.9998          1.0000            1.0000         5.21
IVF-Binary-512-nl223-pca (self)                        1_409.03       653.25     2_062.28       0.9983          1.0002            1.0000         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)              1_691.88       145.68     1_837.56       0.6422          1.3878            1.2924         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)              1_691.88       148.59     1_840.48       0.6419          1.3897            1.2933         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)              1_691.88       160.19     1_852.07       0.6413          1.3958            1.2957         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)             1_691.88       262.62     1_954.50       0.9984          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)             1_691.88       407.17     2_099.06       0.9999          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)             1_691.88       263.19     1_955.08       0.9983          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)             1_691.88       389.05     2_080.93       0.9999          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)             1_691.88       269.56     1_961.44       0.9981          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)             1_691.88       403.81     2_095.70       0.9999          1.0000            1.0000         5.48
IVF-Binary-512-nl316-pca (self)                        1_691.88       645.05     2_336.94       0.9984          1.0002            1.0000         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)           1_287.14       209.71     1_496.85       0.6492          1.4403            1.3298         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)          1_287.14       234.26     1_521.39       0.6468          1.4562            1.3410         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)          1_287.14       243.19     1_530.33       0.6457          1.4688            1.3455         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)          1_287.14       327.45     1_614.59       0.9990          1.0002            1.0000         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)          1_287.14       458.95     1_746.09       0.9994          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)         1_287.14       349.29     1_636.43       0.9995          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)         1_287.14       489.20     1_776.34       0.9999          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)         1_287.14       367.24     1_654.38       0.9994          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)         1_287.14       516.35     1_803.48       0.9999          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-random (self)                    1_287.14       981.46     2_268.60       0.9995          1.0001            1.0000         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_407.88       224.69     1_632.57       0.6479          1.4467            1.3358         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_407.88       228.80     1_636.68       0.6472          1.4539            1.3395         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_407.88       237.61     1_645.48       0.6461          1.4684            1.3443         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_407.88       341.03     1_748.91       0.9993          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_407.88       471.23     1_879.11       0.9997          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_407.88       343.00     1_750.88       0.9994          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_407.88       483.20     1_891.08       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_407.88       363.25     1_771.13       0.9994          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_407.88       505.13     1_913.01       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-random (self)                    1_407.88       942.38     2_350.26       0.9995          1.0001            1.0000         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)          1_687.22       226.12     1_913.34       0.6478          1.4461            1.3341        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)          1_687.22       227.92     1_915.13       0.6475          1.4494            1.3360        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)          1_687.22       241.21     1_928.43       0.6465          1.4599            1.3405        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)         1_687.22       358.94     2_046.16       0.9995          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)         1_687.22       490.93     2_178.15       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)         1_687.22       356.29     2_043.51       0.9995          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)         1_687.22       500.96     2_188.18       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)         1_687.22       381.15     2_068.36       0.9994          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)         1_687.22       517.78     2_204.99       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-random (self)                    1_687.22       957.27     2_644.49       0.9995          1.0001            1.0000        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)              1_484.15       210.43     1_694.58       0.6828          1.3187            1.2384         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)             1_484.15       227.72     1_711.87       0.6812          1.3271            1.2437         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)             1_484.15       251.36     1_735.51       0.6805          1.3344            1.2453         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)             1_484.15       327.98     1_812.13       0.9992          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)             1_484.15       457.98     1_942.13       0.9994          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)            1_484.15       352.43     1_836.58       0.9997          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)            1_484.15       487.90     1_972.06       1.0000          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)            1_484.15       367.95     1_852.10       0.9996          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)            1_484.15       515.66     1_999.81       1.0000          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-pca (self)                       1_484.15       997.67     2_481.82       0.9997          1.0000            1.0000         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_627.91       222.97     1_850.88       0.6817          1.3239            1.2402         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_627.91       224.56     1_852.48       0.6813          1.3271            1.2427         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_627.91       244.49     1_872.40       0.6805          1.3344            1.2451         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_627.91       344.62     1_972.53       0.9995          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_627.91       491.02     2_118.93       0.9998          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_627.91       353.75     1_981.66       0.9996          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_627.91       488.09     2_116.00       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_627.91       372.17     2_000.08       0.9996          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_627.91       509.65     2_137.56       1.0000          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-pca (self)                       1_627.91       980.80     2_608.71       0.9997          1.0000            1.0000         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             2_033.38       226.50     2_259.88       0.6817          1.3241            1.2409        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)             2_033.38       230.73     2_264.11       0.6815          1.3255            1.2417        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)             2_033.38       247.25     2_280.63       0.6809          1.3307            1.2441        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)            2_033.38       370.67     2_404.05       0.9996          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)            2_033.38       505.56     2_538.94       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)            2_033.38       362.25     2_395.63       0.9996          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)            2_033.38       498.56     2_531.94       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)            2_033.38       383.89     2_417.28       0.9996          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)            2_033.38       518.85     2_552.23       1.0000          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-pca (self)                       2_033.38       974.03     3_007.42       0.9997          1.0000            1.0000        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)              1_043.57       414.20     1_457.77       0.0572          8.2471            7.4925         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)             1_043.57       428.48     1_472.06       0.0520          9.4902            8.2157         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)             1_043.57       471.32     1_514.89       0.0494         10.3593            8.7490         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)             1_043.57       508.86     1_552.43       0.3101          1.8950            1.7877         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)             1_043.57       915.58     1_959.15       0.4773          1.4831            1.3817         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)            1_043.57       530.58     1_574.16       0.2784          2.0012            1.8790         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)            1_043.57       936.23     1_979.81       0.4277          1.5656            1.4630         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)            1_043.57       542.70     1_586.27       0.2619          2.0687            1.9351         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)            1_043.57       951.08     1_994.65       0.3996          1.6191            1.5099         5.04
IVF-Binary-768-nl158-sign (self)                       1_043.57     1_487.21     2_530.79       0.2911          1.9618            1.8389         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)             1_157.25       416.60     1_573.85       0.0575          8.1957            7.3503         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)             1_157.25       423.42     1_580.67       0.0550          8.6555            7.6759         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)             1_157.25       438.69     1_595.94       0.0517          9.6218            8.3391         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)            1_157.25       532.27     1_689.52       0.3112          1.8758            1.7623         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)            1_157.25       917.22     2_074.47       0.4738          1.4729            1.3874         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)            1_157.25       523.04     1_680.29       0.2974          1.9245            1.8040         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)            1_157.25       925.66     2_082.91       0.4500          1.5112            1.4210         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)            1_157.25       532.29     1_689.54       0.2763          2.0107            1.8820         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)            1_157.25       953.43     2_110.68       0.4154          1.5791            1.4831         5.23
IVF-Binary-768-nl223-sign (self)                       1_157.25     1_462.06     2_619.31       0.3097          1.8870            1.7670         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)             1_423.86       435.07     1_858.92       0.0579          7.9049            7.2012         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)             1_423.86       432.28     1_856.14       0.0566          8.1385            7.3655         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)             1_423.86       439.94     1_863.80       0.0534          8.9452            7.8698         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)            1_423.86       520.82     1_944.67       0.3161          1.8521            1.7485         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)            1_423.86       926.78     2_350.64       0.4802          1.4604            1.3754         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)            1_423.86       524.95     1_948.80       0.3080          1.8779            1.7716         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)            1_423.86       936.60     2_360.46       0.4678          1.4796            1.3939         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)            1_423.86       537.37     1_961.23       0.2866          1.9580            1.8379         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)            1_423.86       989.74     2_413.60       0.4331          1.5380            1.4459         5.51
IVF-Binary-768-nl316-sign (self)                       1_423.86     1_511.90     2_935.75       0.3207          1.8447            1.7287         5.51
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
Exhaustive (query)                                        33.27       706.02       739.29       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.27     2_318.00     2_351.28       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             537.41       181.66       719.07       0.5703          1.0361            1.0358         2.56
ExhaustiveRaBitQ-rf5 (query)                             537.41       236.34       773.75       0.9274          1.0016            1.0005         2.56
ExhaustiveRaBitQ-rf10 (query)                            537.41       278.86       816.27       0.9847          1.0003            1.0000         2.56
ExhaustiveRaBitQ-rf20 (query)                            537.41       351.43       888.84       0.9986          1.0000            1.0000         2.56
ExhaustiveRaBitQ (self)                                  537.41       871.54     1_408.95       0.9849          1.0003            1.0000         2.56
IVF-RaBitQ-nl158-np7-rf0 (query)                         596.19        84.97       681.16       0.5827          1.0331            1.0334         2.67
IVF-RaBitQ-nl158-np12-rf0 (query)                        596.19       114.26       710.45       0.5827          1.0331            1.0334         2.67
IVF-RaBitQ-nl158-np17-rf0 (query)                        596.19       150.00       746.19       0.5827          1.0331            1.0334         2.67
IVF-RaBitQ-nl158-np7-rf10 (query)                        596.19       162.65       758.84       0.9864          1.0002            1.0000         2.67
IVF-RaBitQ-nl158-np7-rf20 (query)                        596.19       223.74       819.93       0.9989          1.0000            1.0000         2.67
IVF-RaBitQ-nl158-np12-rf10 (query)                       596.19       190.64       786.83       0.9864          1.0002            1.0000         2.67
IVF-RaBitQ-nl158-np12-rf20 (query)                       596.19       258.64       854.84       0.9989          1.0000            1.0000         2.67
IVF-RaBitQ-nl158-np17-rf10 (query)                       596.19       224.55       820.74       0.9864          1.0002            1.0000         2.67
IVF-RaBitQ-nl158-np17-rf20 (query)                       596.19       283.60       879.79       0.9989          1.0000            1.0000         2.67
IVF-RaBitQ-nl158 (self)                                  596.19       918.54     1_514.73       0.9989          1.0000            1.0000         2.67
IVF-RaBitQ-nl223-np11-rf0 (query)                        582.25       115.43       697.68       0.5929          1.0314            1.0315         2.82
IVF-RaBitQ-nl223-np14-rf0 (query)                        582.25       132.59       714.84       0.5929          1.0314            1.0315         2.82
IVF-RaBitQ-nl223-np21-rf0 (query)                        582.25       179.23       761.48       0.5929          1.0314            1.0315         2.82
IVF-RaBitQ-nl223-np11-rf10 (query)                       582.25       188.02       770.26       0.9889          1.0002            1.0000         2.82
IVF-RaBitQ-nl223-np11-rf20 (query)                       582.25       252.97       835.22       0.9990          1.0000            1.0000         2.82
IVF-RaBitQ-nl223-np14-rf10 (query)                       582.25       199.49       781.73       0.9889          1.0002            1.0000         2.82
IVF-RaBitQ-nl223-np14-rf20 (query)                       582.25       259.59       841.84       0.9991          1.0000            1.0000         2.82
IVF-RaBitQ-nl223-np21-rf10 (query)                       582.25       245.49       827.73       0.9889          1.0002            1.0000         2.82
IVF-RaBitQ-nl223-np21-rf20 (query)                       582.25       294.86       877.11       0.9991          1.0000            1.0000         2.82
IVF-RaBitQ-nl223 (self)                                  582.25       957.59     1_539.84       0.9991          1.0000            1.0000         2.82
IVF-RaBitQ-nl316-np15-rf0 (query)                        641.21       138.38       779.59       0.6009          1.0300            1.0299         3.06
IVF-RaBitQ-nl316-np17-rf0 (query)                        641.21       145.74       786.95       0.6009          1.0299            1.0299         3.06
IVF-RaBitQ-nl316-np25-rf0 (query)                        641.21       192.65       833.86       0.6009          1.0299            1.0299         3.06
IVF-RaBitQ-nl316-np15-rf10 (query)                       641.21       207.36       848.57       0.9900          1.0002            1.0000         3.06
IVF-RaBitQ-nl316-np15-rf20 (query)                       641.21       261.71       902.92       0.9993          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf10 (query)                       641.21       212.55       853.76       0.9900          1.0002            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf20 (query)                       641.21       279.24       920.44       0.9994          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf10 (query)                       641.21       257.55       898.76       0.9900          1.0002            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf20 (query)                       641.21       317.74       958.95       0.9994          1.0000            1.0000         3.06
IVF-RaBitQ-nl316 (self)                                  641.21     1_066.13     1_707.34       0.9993          1.0000            1.0000         3.06
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
Exhaustive (query)                                        69.53     1_351.57     1_421.10       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.53     4_587.20     4_656.74       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                             928.03       322.86     1_250.89       0.5776          1.0229            1.0227         4.37
ExhaustiveRaBitQ-rf5 (query)                             928.03       382.60     1_310.62       0.9262          1.0011            1.0004         4.37
ExhaustiveRaBitQ-rf10 (query)                            928.03       434.52     1_362.55       0.9837          1.0002            1.0000         4.37
ExhaustiveRaBitQ-rf20 (query)                            928.03       536.63     1_464.66       0.9984          1.0000            1.0000         4.37
ExhaustiveRaBitQ (self)                                  928.03     1_374.65     2_302.68       0.9840          1.0002            1.0000         4.37
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_038.67       147.86     1_186.53       0.5897          1.0210            1.0216         4.59
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_038.67       204.06     1_242.73       0.5897          1.0210            1.0216         4.59
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_038.67       258.68     1_297.35       0.5897          1.0210            1.0216         4.59
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_038.67       258.51     1_297.18       0.9855          1.0002            1.0000         4.59
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_038.67       346.95     1_385.62       0.9986          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_038.67       300.69     1_339.36       0.9855          1.0002            1.0000         4.59
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_038.67       405.08     1_443.75       0.9986          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_038.67       357.36     1_396.03       0.9855          1.0002            1.0000         4.59
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_038.67       454.50     1_493.17       0.9986          1.0000            1.0000         4.59
IVF-RaBitQ-nl158 (self)                                1_038.67     1_435.59     2_474.26       0.9988          1.0000            1.0000         4.59
IVF-RaBitQ-nl223-np11-rf0 (query)                        988.48       199.06     1_187.54       0.5984          1.0202            1.0205         4.89
IVF-RaBitQ-nl223-np14-rf0 (query)                        988.48       226.48     1_214.96       0.5984          1.0202            1.0205         4.89
IVF-RaBitQ-nl223-np21-rf0 (query)                        988.48       303.53     1_292.01       0.5984          1.0202            1.0205         4.89
IVF-RaBitQ-nl223-np11-rf10 (query)                       988.48       298.20     1_286.68       0.9877          1.0001            1.0000         4.89
IVF-RaBitQ-nl223-np11-rf20 (query)                       988.48       382.89     1_371.37       0.9988          1.0000            1.0000         4.89
IVF-RaBitQ-nl223-np14-rf10 (query)                       988.48       338.84     1_327.33       0.9878          1.0001            1.0000         4.89
IVF-RaBitQ-nl223-np14-rf20 (query)                       988.48       413.90     1_402.38       0.9989          1.0000            1.0000         4.89
IVF-RaBitQ-nl223-np21-rf10 (query)                       988.48       399.38     1_387.86       0.9878          1.0001            1.0000         4.89
IVF-RaBitQ-nl223-np21-rf20 (query)                       988.48       495.39     1_483.87       0.9989          1.0000            1.0000         4.89
IVF-RaBitQ-nl223 (self)                                  988.48     1_547.38     2_535.86       0.9990          1.0000            1.0000         4.89
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_061.87       240.54     1_302.41       0.6051          1.0193            1.0197         5.35
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_061.87       271.55     1_333.42       0.6051          1.0193            1.0197         5.35
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_061.87       370.14     1_432.00       0.6051          1.0193            1.0197         5.35
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_061.87       338.48     1_400.35       0.9882          1.0001            1.0000         5.35
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_061.87       426.43     1_488.29       0.9990          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_061.87       356.97     1_418.84       0.9882          1.0001            1.0000         5.35
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_061.87       450.07     1_511.94       0.9990          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_061.87       446.88     1_508.74       0.9882          1.0001            1.0000         5.35
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_061.87       533.56     1_595.43       0.9990          1.0000            1.0000         5.35
IVF-RaBitQ-nl316 (self)                                1_061.87     1_712.26     2_774.12       0.9991          1.0000            1.0000         5.35
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
Exhaustive (query)                                       100.63     2_006.19     2_106.83       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.63     6_628.33     6_728.96       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           1_112.77       461.30     1_574.06       0.5778          1.0181            1.0180         6.16
ExhaustiveRaBitQ-rf5 (query)                           1_112.77       519.79     1_632.56       0.9239          1.0009            1.0003         6.16
ExhaustiveRaBitQ-rf10 (query)                          1_112.77       583.41     1_696.17       0.9829          1.0002            1.0000         6.16
ExhaustiveRaBitQ-rf20 (query)                          1_112.77       708.84     1_821.61       0.9983          1.0000            1.0000         6.16
ExhaustiveRaBitQ (self)                                1_112.77     1_857.45     2_970.22       0.9829          1.0002            1.0000         6.16
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_234.88       195.64     1_430.52       0.5908          1.0165            1.0169         6.50
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_234.88       281.84     1_516.72       0.5908          1.0165            1.0169         6.50
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_234.88       359.13     1_594.01       0.5908          1.0165            1.0169         6.50
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_234.88       325.43     1_560.31       0.9851          1.0001            1.0000         6.50
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_234.88       440.68     1_675.56       0.9986          1.0000            1.0000         6.50
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_234.88       398.66     1_633.54       0.9851          1.0001            1.0000         6.50
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_234.88       516.81     1_751.69       0.9986          1.0000            1.0000         6.50
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_234.88       482.07     1_716.95       0.9851          1.0001            1.0000         6.50
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_234.88       595.92     1_830.80       0.9986          1.0000            1.0000         6.50
IVF-RaBitQ-nl158 (self)                                1_234.88     1_897.10     3_131.98       0.9987          1.0000            1.0000         6.50
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_242.02       267.28     1_509.29       0.5898          1.0169            1.0169         6.96
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_242.02       314.41     1_556.42       0.5898          1.0169            1.0169         6.96
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_242.02       435.50     1_677.52       0.5898          1.0169            1.0169         6.96
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_242.02       390.24     1_632.26       0.9848          1.0001            1.0000         6.96
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_242.02       511.08     1_753.10       0.9985          1.0000            1.0000         6.96
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_242.02       432.67     1_674.68       0.9849          1.0001            1.0000         6.96
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_242.02       543.34     1_785.35       0.9985          1.0000            1.0000         6.96
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_242.02       541.16     1_783.17       0.9849          1.0001            1.0000         6.96
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_242.02       655.53     1_897.55       0.9985          1.0000            1.0000         6.96
IVF-RaBitQ-nl223 (self)                                1_242.02     2_100.62     3_342.64       0.9986          1.0000            1.0000         6.96
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_281.45       333.71     1_615.16       0.6025          1.0154            1.0159         7.66
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_281.45       367.15     1_648.59       0.6025          1.0154            1.0159         7.66
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_281.45       506.76     1_788.21       0.6025          1.0154            1.0159         7.66
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_281.45       455.48     1_736.93       0.9873          1.0001            1.0000         7.66
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_281.45       565.72     1_847.17       0.9988          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_281.45       485.11     1_766.56       0.9873          1.0001            1.0000         7.66
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_281.45       594.25     1_875.69       0.9989          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_281.45       611.13     1_892.58       0.9873          1.0001            1.0000         7.66
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_281.45       721.43     2_002.87       0.9989          1.0000            1.0000         7.66
IVF-RaBitQ-nl316 (self)                                1_281.45     2_314.90     3_596.35       0.9989          1.0000            1.0000         7.66
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
Exhaustive (query)                                        33.54       748.69       782.22       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.54     2_393.44     2_426.98       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             550.81       223.25       774.06       0.7383          1.0233            1.0223         2.57
ExhaustiveRaBitQ-rf5 (query)                             550.81       263.07       813.88       0.9978          1.0000            1.0000         2.57
ExhaustiveRaBitQ-rf10 (query)                            550.81       320.00       870.81       1.0000          1.0000            1.0000         2.57
ExhaustiveRaBitQ-rf20 (query)                            550.81       396.81       947.62       1.0000          1.0000            1.0000         2.57
ExhaustiveRaBitQ (self)                                  550.81     1_029.94     1_580.75       1.0000          1.0000            1.0000         2.57
IVF-RaBitQ-nl158-np7-rf0 (query)                         571.42        85.91       657.34       0.7411          1.0229            1.0219         2.68
IVF-RaBitQ-nl158-np12-rf0 (query)                        571.42       116.78       688.21       0.7411          1.0229            1.0219         2.68
IVF-RaBitQ-nl158-np17-rf0 (query)                        571.42       151.23       722.66       0.7411          1.0229            1.0219         2.68
IVF-RaBitQ-nl158-np7-rf10 (query)                        571.42       162.12       733.54       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np7-rf20 (query)                        571.42       230.83       802.25       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np12-rf10 (query)                       571.42       206.24       777.66       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np12-rf20 (query)                       571.42       254.43       825.85       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rf10 (query)                       571.42       223.72       795.15       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rf20 (query)                       571.42       288.14       859.56       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158 (self)                                  571.42       925.60     1_497.02       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl223-np11-rf0 (query)                        580.24       114.25       694.49       0.7444          1.0221            1.0210         2.84
IVF-RaBitQ-nl223-np14-rf0 (query)                        580.24       133.68       713.92       0.7444          1.0221            1.0210         2.84
IVF-RaBitQ-nl223-np21-rf0 (query)                        580.24       183.96       764.20       0.7444          1.0221            1.0210         2.84
IVF-RaBitQ-nl223-np11-rf10 (query)                       580.24       185.96       766.20       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np11-rf20 (query)                       580.24       248.05       828.28       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np14-rf10 (query)                       580.24       212.68       792.91       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np14-rf20 (query)                       580.24       273.30       853.54       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np21-rf10 (query)                       580.24       258.21       838.44       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np21-rf20 (query)                       580.24       314.59       894.82       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223 (self)                                  580.24     1_017.71     1_597.95       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl316-np15-rf0 (query)                        610.31       138.25       748.56       0.7482          1.0214            1.0202         3.06
IVF-RaBitQ-nl316-np17-rf0 (query)                        610.31       151.28       761.59       0.7482          1.0214            1.0202         3.06
IVF-RaBitQ-nl316-np25-rf0 (query)                        610.31       205.17       815.48       0.7482          1.0214            1.0202         3.06
IVF-RaBitQ-nl316-np15-rf10 (query)                       610.31       212.55       822.86       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np15-rf20 (query)                       610.31       272.63       882.94       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf10 (query)                       610.31       223.63       833.94       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf20 (query)                       610.31       284.09       894.40       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf10 (query)                       610.31       273.73       884.04       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf20 (query)                       610.31       339.50       949.80       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316 (self)                                  610.31     1_088.02     1_698.33       1.0000          1.0000            1.0000         3.06
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
Exhaustive (query)                                        69.43     1_375.17     1_444.60       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.43     4_633.50     4_702.93       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                             882.18       374.02     1_256.20       0.7526          1.0138            1.0132         4.36
ExhaustiveRaBitQ-rf5 (query)                             882.18       426.53     1_308.71       0.9982          1.0000            1.0000         4.36
ExhaustiveRaBitQ-rf10 (query)                            882.18       486.02     1_368.20       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ-rf20 (query)                            882.18       592.44     1_474.62       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ (self)                                  882.18     1_519.60     2_401.78       1.0000          1.0000            1.0000         4.36
IVF-RaBitQ-nl158-np7-rf0 (query)                         969.02       144.63     1_113.65       0.7545          1.0137            1.0130         4.59
IVF-RaBitQ-nl158-np12-rf0 (query)                        969.02       202.00     1_171.02       0.7545          1.0137            1.0130         4.59
IVF-RaBitQ-nl158-np17-rf0 (query)                        969.02       260.67     1_229.69       0.7545          1.0137            1.0130         4.59
IVF-RaBitQ-nl158-np7-rf10 (query)                        969.02       253.03     1_222.05       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np7-rf20 (query)                        969.02       347.84     1_316.86       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np12-rf10 (query)                       969.02       300.13     1_269.15       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np12-rf20 (query)                       969.02       403.03     1_372.05       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np17-rf10 (query)                       969.02       361.80     1_330.82       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np17-rf20 (query)                       969.02       461.39     1_430.41       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158 (self)                                  969.02     1_444.97     2_413.98       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl223-np11-rf0 (query)                        952.95       196.47     1_149.42       0.7568          1.0134            1.0128         4.89
IVF-RaBitQ-nl223-np14-rf0 (query)                        952.95       231.14     1_184.10       0.7570          1.0134            1.0128         4.89
IVF-RaBitQ-nl223-np21-rf0 (query)                        952.95       308.12     1_261.07       0.7570          1.0134            1.0128         4.89
IVF-RaBitQ-nl223-np11-rf10 (query)                       952.95       302.44     1_255.39       0.9994          1.0000            1.0000         4.89
IVF-RaBitQ-nl223-np11-rf20 (query)                       952.95       390.76     1_343.71       0.9995          1.0000            1.0000         4.89
IVF-RaBitQ-nl223-np14-rf10 (query)                       952.95       328.69     1_281.65       1.0000          1.0000            1.0000         4.89
IVF-RaBitQ-nl223-np14-rf20 (query)                       952.95       424.91     1_377.86       1.0000          1.0000            1.0000         4.89
IVF-RaBitQ-nl223-np21-rf10 (query)                       952.95       409.04     1_361.99       1.0000          1.0000            1.0000         4.89
IVF-RaBitQ-nl223-np21-rf20 (query)                       952.95       498.15     1_451.10       1.0000          1.0000            1.0000         4.89
IVF-RaBitQ-nl223 (self)                                  952.95     1_630.75     2_583.70       1.0000          1.0000            1.0000         4.89
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_037.15       250.34     1_287.49       0.7583          1.0132            1.0125         5.35
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_037.15       272.79     1_309.94       0.7583          1.0132            1.0125         5.35
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_037.15       351.20     1_388.35       0.7583          1.0132            1.0125         5.35
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_037.15       356.97     1_394.12       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_037.15       442.18     1_479.33       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_037.15       373.35     1_410.50       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_037.15       470.74     1_507.89       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_037.15       463.06     1_500.20       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_037.15       559.97     1_597.12       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316 (self)                                1_037.15     1_752.82     2_789.97       1.0000          1.0000            1.0000         5.35
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
Exhaustive (query)                                       103.05     1_966.83     2_069.87       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        103.05     6_542.58     6_645.63       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           1_120.69       519.00     1_639.70       0.7322          1.0116            1.0111         6.17
ExhaustiveRaBitQ-rf5 (query)                           1_120.69       568.25     1_688.94       0.9963          1.0000            1.0000         6.17
ExhaustiveRaBitQ-rf10 (query)                          1_120.69       639.87     1_760.56       0.9999          1.0000            1.0000         6.17
ExhaustiveRaBitQ-rf20 (query)                          1_120.69       757.85     1_878.54       1.0000          1.0000            1.0000         6.17
ExhaustiveRaBitQ (self)                                1_120.69     2_004.97     3_125.66       1.0000          1.0000            1.0000         6.17
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_225.13       195.66     1_420.79       0.7357          1.0112            1.0107         6.51
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_225.13       278.18     1_503.31       0.7357          1.0112            1.0107         6.51
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_225.13       359.05     1_584.18       0.7357          1.0112            1.0107         6.51
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_225.13       332.72     1_557.85       0.9999          1.0000            1.0000         6.51
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_225.13       438.55     1_663.68       1.0000          1.0000            1.0000         6.51
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_225.13       394.43     1_619.56       0.9999          1.0000            1.0000         6.51
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_225.13       513.05     1_738.18       1.0000          1.0000            1.0000         6.51
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_225.13       483.81     1_708.94       0.9999          1.0000            1.0000         6.51
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_225.13       589.42     1_814.55       1.0000          1.0000            1.0000         6.51
IVF-RaBitQ-nl158 (self)                                1_225.13     1_875.41     3_100.54       1.0000          1.0000            1.0000         6.51
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_194.27       265.13     1_459.40       0.7385          1.0110            1.0105         6.97
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_194.27       309.95     1_504.22       0.7385          1.0110            1.0105         6.97
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_194.27       427.67     1_621.94       0.7385          1.0110            1.0105         6.97
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_194.27       399.98     1_594.25       0.9999          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_194.27       505.82     1_700.09       1.0000          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_194.27       434.70     1_628.97       0.9999          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_194.27       550.43     1_744.70       1.0000          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_194.27       555.44     1_749.71       0.9999          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_194.27       664.10     1_858.37       1.0000          1.0000            1.0000         6.97
IVF-RaBitQ-nl223 (self)                                1_194.27     2_114.85     3_309.12       1.0000          1.0000            1.0000         6.97
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_306.45       348.20     1_654.65       0.7398          1.0109            1.0105         7.67
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_306.45       363.87     1_670.32       0.7398          1.0109            1.0105         7.67
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_306.45       493.33     1_799.78       0.7398          1.0109            1.0105         7.67
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_306.45       460.59     1_767.05       0.9999          1.0000            1.0000         7.67
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_306.45       573.05     1_879.50       1.0000          1.0000            1.0000         7.67
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_306.45       485.82     1_792.27       0.9999          1.0000            1.0000         7.67
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_306.45       606.66     1_913.11       1.0000          1.0000            1.0000         7.67
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_306.45       616.18     1_922.63       0.9999          1.0000            1.0000         7.67
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_306.45       730.00     2_036.45       1.0000          1.0000            1.0000         7.67
IVF-RaBitQ-nl316 (self)                                1_306.45     2_355.56     3_662.01       1.0000          1.0000            1.0000         7.67
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
Exhaustive (query)                                        34.04       734.11       768.15       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         34.04     2_448.48     2_482.52       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             661.45       251.11       912.56       0.8706          1.0280            1.0231         2.57
ExhaustiveRaBitQ-rf5 (query)                             661.45       304.32       965.77       1.0000          1.0000            1.0000         2.57
ExhaustiveRaBitQ-rf10 (query)                            661.45       357.77     1_019.22       1.0000          1.0000            1.0000         2.57
ExhaustiveRaBitQ-rf20 (query)                            661.45       473.17     1_134.62       1.0000          1.0000            1.0000         2.57
ExhaustiveRaBitQ (self)                                  661.45     1_187.30     1_848.75       1.0000          1.0000            1.0000         2.57
IVF-RaBitQ-nl158-np7-rf0 (query)                         816.11        93.08       909.18       0.8738          1.0271            1.0220         2.68
IVF-RaBitQ-nl158-np12-rf0 (query)                        816.11       139.37       955.48       0.8745          1.0267            1.0217         2.68
IVF-RaBitQ-nl158-np17-rf0 (query)                        816.11       181.95       998.06       0.8745          1.0267            1.0217         2.68
IVF-RaBitQ-nl158-np7-rf10 (query)                        816.11       190.05     1_006.15       0.9976          1.0005            1.0000         2.68
IVF-RaBitQ-nl158-np7-rf20 (query)                        816.11       239.56     1_055.66       0.9976          1.0005            1.0000         2.68
IVF-RaBitQ-nl158-np12-rf10 (query)                       816.11       214.70     1_030.80       0.9999          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np12-rf20 (query)                       816.11       285.78     1_101.88       0.9999          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rf10 (query)                       816.11       253.03     1_069.14       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rf20 (query)                       816.11       322.43     1_138.54       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158 (self)                                  816.11     1_048.77     1_864.88       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl223-np11-rf0 (query)                        824.11       116.04       940.14       0.8845          1.0225            1.0183         2.83
IVF-RaBitQ-nl223-np14-rf0 (query)                        824.11       141.00       965.11       0.8846          1.0224            1.0182         2.83
IVF-RaBitQ-nl223-np21-rf0 (query)                        824.11       195.54     1_019.65       0.8846          1.0224            1.0182         2.83
IVF-RaBitQ-nl223-np11-rf10 (query)                       824.11       197.45     1_021.56       0.9993          1.0001            1.0000         2.83
IVF-RaBitQ-nl223-np11-rf20 (query)                       824.11       257.14     1_081.24       0.9993          1.0001            1.0000         2.83
IVF-RaBitQ-nl223-np14-rf10 (query)                       824.11       214.26     1_038.37       0.9998          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np14-rf20 (query)                       824.11       290.40     1_114.51       0.9998          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np21-rf10 (query)                       824.11       268.66     1_092.77       1.0000          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np21-rf20 (query)                       824.11       334.77     1_158.88       1.0000          1.0000            1.0000         2.83
IVF-RaBitQ-nl223 (self)                                  824.11     1_075.42     1_899.53       1.0000          1.0000            1.0000         2.83
IVF-RaBitQ-nl316-np15-rf0 (query)                        922.00       162.26     1_084.26       0.8907          1.0195            1.0158         3.06
IVF-RaBitQ-nl316-np17-rf0 (query)                        922.00       161.56     1_083.56       0.8907          1.0195            1.0158         3.06
IVF-RaBitQ-nl316-np25-rf0 (query)                        922.00       212.35     1_134.35       0.8908          1.0195            1.0157         3.06
IVF-RaBitQ-nl316-np15-rf10 (query)                       922.00       219.93     1_141.93       0.9997          1.0001            1.0000         3.06
IVF-RaBitQ-nl316-np15-rf20 (query)                       922.00       281.38     1_203.38       0.9997          1.0001            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf10 (query)                       922.00       225.29     1_147.29       0.9998          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf20 (query)                       922.00       295.45     1_217.45       0.9998          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf10 (query)                       922.00       279.72     1_201.72       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf20 (query)                       922.00       356.44     1_278.44       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316 (self)                                  922.00     1_135.33     2_057.33       1.0000          1.0000            1.0000         3.06
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
Exhaustive (query)                                        69.02     1_361.13     1_430.15       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.02     4_593.90     4_662.91       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           1_061.55       397.46     1_459.00       0.9095          1.0128            1.0096         4.36
ExhaustiveRaBitQ-rf5 (query)                           1_061.55       471.50     1_533.05       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ-rf10 (query)                          1_061.55       535.23     1_596.77       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ-rf20 (query)                          1_061.55       656.22     1_717.76       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ (self)                                1_061.55     1_707.88     2_769.42       1.0000          1.0000            1.0000         4.36
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_217.39       157.62     1_375.01       0.9150          1.0112            1.0081         4.59
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_217.39       225.25     1_442.65       0.9156          1.0109            1.0080         4.59
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_217.39       288.26     1_505.65       0.9157          1.0109            1.0080         4.59
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_217.39       266.19     1_483.58       0.9986          1.0003            1.0000         4.59
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_217.39       357.44     1_574.83       0.9986          1.0003            1.0000         4.59
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_217.39       322.71     1_540.11       0.9999          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_217.39       425.64     1_643.04       0.9999          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_217.39       395.97     1_613.36       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_217.39       491.50     1_708.90       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158 (self)                                1_217.39     1_570.22     2_787.62       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_160.52       197.99     1_358.51       0.9224          1.0091            1.0066         4.90
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_160.52       236.80     1_397.31       0.9224          1.0091            1.0066         4.90
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_160.52       324.50     1_485.01       0.9224          1.0091            1.0066         4.90
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_160.52       302.52     1_463.04       0.9997          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_160.52       393.31     1_553.82       0.9997          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_160.52       330.81     1_491.32       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_160.52       429.64     1_590.16       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_160.52       424.20     1_584.71       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_160.52       513.82     1_674.34       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223 (self)                                1_160.52     1_655.54     2_816.06       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_278.34       243.41     1_521.75       0.9274          1.0079            1.0055         5.36
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_278.34       271.12     1_549.47       0.9274          1.0079            1.0055         5.36
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_278.34       367.75     1_646.10       0.9274          1.0079            1.0055         5.36
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_278.34       349.63     1_627.98       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_278.34       443.46     1_721.80       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_278.34       368.36     1_646.70       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_278.34       478.81     1_757.16       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_278.34       466.14     1_744.49       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_278.34       559.82     1_838.16       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl316 (self)                                1_278.34     1_806.93     3_085.28       1.0000          1.0000            1.0000         5.36
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
Exhaustive (query)                                       101.87     1_972.02     2_073.88       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.87     6_581.23     6_683.09       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           1_430.58       563.33     1_993.91       0.9148          1.0115            1.0083         6.16
ExhaustiveRaBitQ-rf5 (query)                           1_430.58       641.58     2_072.17       1.0000          1.0000            1.0000         6.16
ExhaustiveRaBitQ-rf10 (query)                          1_430.58       713.44     2_144.02       1.0000          1.0000            1.0000         6.16
ExhaustiveRaBitQ-rf20 (query)                          1_430.58       853.36     2_283.95       1.0000          1.0000            1.0000         6.16
ExhaustiveRaBitQ (self)                                1_430.58     2_277.82     3_708.41       1.0000          1.0000            1.0000         6.16
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_688.75       210.79     1_899.53       0.9172          1.0107            1.0077         6.51
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_688.75       309.90     1_998.65       0.9174          1.0107            1.0076         6.51
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_688.75       407.20     2_095.95       0.9174          1.0107            1.0076         6.51
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_688.75       331.97     2_020.72       0.9995          1.0001            1.0000         6.51
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_688.75       448.37     2_137.11       0.9995          1.0001            1.0000         6.51
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_688.75       421.56     2_110.31       1.0000          1.0000            1.0000         6.51
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_688.75       539.81     2_228.56       1.0000          1.0000            1.0000         6.51
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_688.75       524.58     2_213.33       1.0000          1.0000            1.0000         6.51
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_688.75       633.70     2_322.45       1.0000          1.0000            1.0000         6.51
IVF-RaBitQ-nl158 (self)                                1_688.75     2_037.27     3_726.02       1.0000          1.0000            1.0000         6.51
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_704.15       276.13     1_980.28       0.9221          1.0094            1.0067         6.97
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_704.15       325.32     2_029.47       0.9221          1.0094            1.0067         6.97
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_704.15       453.40     2_157.55       0.9221          1.0094            1.0067         6.97
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_704.15       396.62     2_100.77       0.9999          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_704.15       510.78     2_214.93       0.9999          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_704.15       445.26     2_149.41       1.0000          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_704.15       558.80     2_262.95       1.0000          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_704.15       568.99     2_273.14       1.0000          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_704.15       684.59     2_388.74       1.0000          1.0000            1.0000         6.97
IVF-RaBitQ-nl223 (self)                                1_704.15     2_215.99     3_920.14       1.0000          1.0000            1.0000         6.97
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_876.17       341.09     2_217.26       0.9267          1.0082            1.0057         7.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_876.17       384.46     2_260.63       0.9267          1.0082            1.0057         7.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_876.17       513.98     2_390.15       0.9267          1.0082            1.0057         7.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_876.17       461.89     2_338.07       1.0000          1.0000            1.0000         7.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_876.17       576.97     2_453.14       1.0000          1.0000            1.0000         7.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_876.17       489.39     2_365.56       1.0000          1.0000            1.0000         7.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_876.17       618.69     2_494.87       1.0000          1.0000            1.0000         7.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_876.17       628.99     2_505.16       1.0000          1.0000            1.0000         7.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_876.17       753.91     2_630.08       1.0000          1.0000            1.0000         7.63
IVF-RaBitQ-nl316 (self)                                1_876.17     2_415.18     4_291.35       1.0000          1.0000            1.0000         7.63
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
Exhaustive (query)                                        32.97       693.15       726.12       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.97     2_317.04     2_350.01       1.0000          1.0000            1.0000        48.83
QG-d32-l64-ef15 (query)                                2_528.14       294.54     2_822.68       0.9906          1.0027            1.0000       116.35
QG-d32-l64-ef30 (query)                                2_528.14       424.19     2_952.33       0.9953          1.0020            1.0000       116.35
QG-d32-l64-ef60 (query)                                2_528.14       602.18     3_130.32       0.9971          1.0017            1.0000       116.35
QG-d32-l64-ef120 (query)                               2_528.14       892.02     3_420.15       0.9979          1.0015            1.0000       116.35
QG-d32-l64 (self)                                      2_528.14     1_970.22     4_498.36       0.9973          1.0017            1.0000       116.35
QG-d32-l128-ef15 (query)                               2_878.31       296.19     3_174.49       0.9876          1.1390            1.0000       116.35
QG-d32-l128-ef30 (query)                               2_878.31       425.05     3_303.36       0.9931          1.0769            1.0000       116.35
QG-d32-l128-ef60 (query)                               2_878.31       606.82     3_485.12       0.9956          1.0172            1.0000       116.35
QG-d32-l128-ef120 (query)                              2_878.31       891.91     3_770.22       0.9970          1.0063            1.0000       116.35
QG-d32-l128 (self)                                     2_878.31     1_980.65     4_858.96       0.9959          1.0216            1.0000       116.35
QG-d64-l64-ef15 (query)                                8_694.79       651.34     9_346.13       0.9992          1.0002            1.0000       183.49
QG-d64-l64-ef30 (query)                                8_694.79       883.55     9_578.34       0.9997          1.0002            1.0000       183.49
QG-d64-l64-ef60 (query)                                8_694.79     1_223.62     9_918.40       0.9998          1.0002            1.0000       183.49
QG-d64-l64-ef120 (query)                               8_694.79     1_767.47    10_462.25       0.9998          1.0001            1.0000       183.49
QG-d64-l64 (self)                                      8_694.79     4_073.26    12_768.04       0.9998          1.0002            1.0000       183.49
QG-d64-l128-ef15 (query)                               9_602.76       658.70    10_261.46       0.9992          1.0002            1.0000       183.49
QG-d64-l128-ef30 (query)                               9_602.76       899.51    10_502.28       0.9998          1.0002            1.0000       183.49
QG-d64-l128-ef60 (query)                               9_602.76     1_224.69    10_827.46       0.9998          1.0002            1.0000       183.49
QG-d64-l128-ef120 (query)                              9_602.76     1_769.60    11_372.36       0.9998          1.0001            1.0000       183.49
QG-d64-l128 (self)                                     9_602.76     4_426.23    14_029.00       0.9998          1.0002            1.0000       183.49
HnswRaBitQ-m16-ex1-ef15 (query)                        1_350.30        89.84     1_440.14       0.7488         69.4097            1.0085        10.85
HnswRaBitQ-m16-ex1-ef30 (query)                        1_350.30       109.71     1_460.01       0.7552         69.3514            1.0082        10.85
HnswRaBitQ-m16-ex1-ef60 (query)                        1_350.30       142.00     1_492.30       0.7766          1.0114            1.0080        10.85
HnswRaBitQ-m16-ex1-ef120 (query)                       1_350.30       175.56     1_525.86       0.7772          1.0091            1.0080        10.85
HnswRaBitQ-m16-ex1 (self)                              1_350.30     4_891.28     6_241.58       0.6963          1.0189            1.0161        10.85
HnswRaBitQ-m16-ex3-ef15 (query)                        1_359.20        96.44     1_455.64       0.8997          1.3444            1.0010        13.90
HnswRaBitQ-m16-ex3-ef30 (query)                        1_359.20       121.24     1_480.43       0.9198          1.0075            1.0007        13.90
HnswRaBitQ-m16-ex3-ef60 (query)                        1_359.20       149.99     1_509.19       0.9264          1.0015            1.0006        13.90
HnswRaBitQ-m16-ex3-ef120 (query)                       1_359.20       190.88     1_550.08       0.9280          1.0010            1.0005        13.90
HnswRaBitQ-m16-ex3 (self)                              1_359.20     4_940.60     6_299.80       0.9031          1.0023            1.0012        13.90
HnswRaBitQ-m16-ex5-ef15 (query)                        1_463.25       103.61     1_566.85       0.9382          1.0195            1.0001        16.95
HnswRaBitQ-m16-ex5-ef30 (query)                        1_463.25       137.91     1_601.16       0.9644          1.0035            1.0000        16.95
HnswRaBitQ-m16-ex5-ef60 (query)                        1_463.25       161.78     1_625.03       0.9740          1.0012            1.0000        16.95
HnswRaBitQ-m16-ex5-ef120 (query)                       1_463.25       204.11     1_667.36       0.9763          1.0003            1.0000        16.95
HnswRaBitQ-m16-ex5 (self)                              1_463.25     4_776.47     6_239.71       0.9691          1.0532            1.0000        16.95
HnswRaBitQ-m16-ex8-ef15 (query)                        2_184.26       100.68     2_284.94       0.9477          1.0141            1.0000        21.53
HnswRaBitQ-m16-ex8-ef30 (query)                        2_184.26       124.22     2_308.48       0.9779          1.0055            1.0000        21.53
HnswRaBitQ-m16-ex8-ef60 (query)                        2_184.26       156.64     2_340.90       0.9904          1.0017            1.0000        21.53
HnswRaBitQ-m16-ex8-ef120 (query)                       2_184.26       198.18     2_382.44       0.9935          1.0005            1.0000        21.53
HnswRaBitQ-m16-ex8 (self)                              2_184.26     4_724.27     6_908.52       0.9927          1.0012            1.0000        21.53
HnswRaBitQ-m32-ex1-ef15 (query)                        1_316.68       100.85     1_417.52       0.7746          1.0201            1.0080        16.95
HnswRaBitQ-m32-ex1-ef30 (query)                        1_316.68       122.93     1_439.60       0.7767          1.0127            1.0080        16.95
HnswRaBitQ-m32-ex1-ef60 (query)                        1_316.68       150.77     1_467.44       0.7774          1.0092            1.0080        16.95
HnswRaBitQ-m32-ex1-ef120 (query)                       1_316.68       192.21     1_508.88       0.7775          1.0085            1.0080        16.95
HnswRaBitQ-m32-ex1 (self)                              1_316.68     5_969.33     7_286.01       0.6960          1.0178            1.0162        16.95
HnswRaBitQ-m32-ex3-ef15 (query)                        1_397.76       110.69     1_508.45       0.9142          1.0833            1.0007        20.00
HnswRaBitQ-m32-ex3-ef30 (query)                        1_397.76       138.43     1_536.19       0.9248          1.0116            1.0006        20.00
HnswRaBitQ-m32-ex3-ef60 (query)                        1_397.76       168.37     1_566.13       0.9275          1.0011            1.0005        20.00
HnswRaBitQ-m32-ex3-ef120 (query)                       1_397.76       212.30     1_610.06       0.9281          1.0010            1.0005        20.00
HnswRaBitQ-m32-ex3 (self)                              1_397.76     5_548.48     6_946.24       0.9040          1.0017            1.0012        20.00
HnswRaBitQ-m32-ex5-ef15 (query)                        1_511.83       120.57     1_632.39       0.9584          1.0151            1.0000        23.06
HnswRaBitQ-m32-ex5-ef30 (query)                        1_511.83       148.29     1_660.12       0.9718          1.0061            1.0000        23.06
HnswRaBitQ-m32-ex5-ef60 (query)                        1_511.83       182.91     1_694.74       0.9758          1.0002            1.0000        23.06
HnswRaBitQ-m32-ex5-ef120 (query)                       1_511.83       226.23     1_738.06       0.9767          1.0001            1.0000        23.06
HnswRaBitQ-m32-ex5 (self)                              1_511.83     5_746.67     7_258.50       0.9710          1.0002            1.0000        23.06
HnswRaBitQ-m32-ex8-ef15 (query)                        2_216.21       114.10     2_330.31       0.9721          1.0073            1.0000        27.63
HnswRaBitQ-m32-ex8-ef30 (query)                        2_216.21       142.90     2_359.11       0.9883          1.0012            1.0000        27.63
HnswRaBitQ-m32-ex8-ef60 (query)                        2_216.21       174.66     2_390.87       0.9932          1.0005            1.0000        27.63
HnswRaBitQ-m32-ex8-ef120 (query)                       2_216.21       219.76     2_435.97       0.9942          1.0003            1.0000        27.63
HnswRaBitQ-m32-ex8 (self)                              2_216.21     6_020.89     8_237.10       0.9948          1.0007            1.0000        27.63
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
Exhaustive (query)                                        69.95     1_331.94     1_401.89       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.95     4_533.62     4_603.57       1.0000          1.0000            1.0000        97.66
QG-d32-l64-ef15 (query)                                4_990.35       509.77     5_500.12       0.9862          1.0046            1.0000       214.01
QG-d32-l64-ef30 (query)                                4_990.35       682.52     5_672.87       0.9915          1.0044            1.0000       214.01
QG-d32-l64-ef60 (query)                                4_990.35       915.05     5_905.40       0.9937          1.0042            1.0000       214.01
QG-d32-l64-ef120 (query)                               4_990.35     1_260.10     6_250.44       0.9948          1.0039            1.0000       214.01
QG-d32-l64 (self)                                      4_990.35     3_019.92     8_010.27       0.9937          1.0043            1.0000       214.01
QG-d32-l128-ef15 (query)                               5_534.36       504.72     6_039.08       0.9883          1.0032            1.0000       214.01
QG-d32-l128-ef30 (query)                               5_534.36       683.83     6_218.19       0.9935          1.0030            1.0000       214.01
QG-d32-l128-ef60 (query)                               5_534.36       919.10     6_453.46       0.9956          1.0028            1.0000       214.01
QG-d32-l128-ef120 (query)                              5_534.36     1_257.96     6_792.32       0.9965          1.0027            1.0000       214.01
QG-d32-l128 (self)                                     5_534.36     3_154.49     8_688.85       0.9952          1.0030            1.0000       214.01
QG-d64-l64-ef15 (query)                               19_042.38     1_205.15    20_247.54       0.9982          1.0010            1.0000       329.97
QG-d64-l64-ef30 (query)                               19_042.38     1_405.90    20_448.28       0.9989          1.0009            1.0000       329.97
QG-d64-l64-ef60 (query)                               19_042.38     1_771.24    20_813.62       0.9991          1.0008            1.0000       329.97
QG-d64-l64-ef120 (query)                              19_042.38     2_492.65    21_535.04       0.9992          1.0008            1.0000       329.97
QG-d64-l64 (self)                                     19_042.38     5_686.44    24_728.83       0.9990          1.0008            1.0000       329.97
QG-d64-l128-ef15 (query)                              18_704.18     1_015.23    19_719.41       0.9983          1.0009            1.0000       329.97
QG-d64-l128-ef30 (query)                              18_704.18     1_311.01    20_015.19       0.9989          1.0009            1.0000       329.97
QG-d64-l128-ef60 (query)                              18_704.18     1_725.53    20_429.71       0.9991          1.0008            1.0000       329.97
QG-d64-l128-ef120 (query)                             18_704.18     2_309.67    21_013.85       0.9991          1.0008            1.0000       329.97
QG-d64-l128 (self)                                    18_704.18     5_728.72    24_432.90       0.9990          1.0008            1.0000       329.97
HnswRaBitQ-m16-ex1-ef15 (query)                        2_352.17       166.31     2_518.48       0.7643          1.0230            1.0057        14.15
HnswRaBitQ-m16-ex1-ef30 (query)                        2_352.17       202.11     2_554.27       0.7736          1.0097            1.0055        14.15
HnswRaBitQ-m16-ex1-ef60 (query)                        2_352.17       248.24     2_600.41       0.7762          1.0061            1.0054        14.15
HnswRaBitQ-m16-ex1-ef120 (query)                       2_352.17       304.59     2_656.75       0.7767          1.0058            1.0054        14.15
HnswRaBitQ-m16-ex1 (self)                              2_352.17    10_342.69    12_694.86       0.6931          1.0129            1.0110        14.15
HnswRaBitQ-m16-ex3-ef15 (query)                        2_513.28       192.37     2_705.65       0.8889          1.1954            1.0008        20.25
HnswRaBitQ-m16-ex3-ef30 (query)                        2_513.28       234.31     2_747.59       0.9137          1.0531            1.0005        20.25
HnswRaBitQ-m16-ex3-ef60 (query)                        2_513.28       282.96     2_796.24       0.9240          1.0018            1.0004        20.25
HnswRaBitQ-m16-ex3-ef120 (query)                       2_513.28       347.86     2_861.14       0.9266          1.0008            1.0004        20.25
HnswRaBitQ-m16-ex3 (self)                              2_513.28    10_609.54    13_122.82       0.8996          1.0035            1.0008        20.25
HnswRaBitQ-m16-ex5-ef15 (query)                        2_754.57       203.05     2_957.62       0.9224          2.0032            1.0002        26.36
HnswRaBitQ-m16-ex5-ef30 (query)                        2_754.57       253.26     3_007.84       0.9557          1.5631            1.0000        26.36
HnswRaBitQ-m16-ex5-ef60 (query)                        2_754.57       304.83     3_059.40       0.9710          1.0007            1.0000        26.36
HnswRaBitQ-m16-ex5-ef120 (query)                       2_754.57       373.98     3_128.56       0.9746          1.0004            1.0000        26.36
HnswRaBitQ-m16-ex5 (self)                              2_754.57    11_036.49    13_791.06       0.9654          1.1366            1.0000        26.36
HnswRaBitQ-m16-ex8-ef15 (query)                        4_229.97       189.98     4_419.94       0.9311          2.8398            1.0000        35.51
HnswRaBitQ-m16-ex8-ef30 (query)                        4_229.97       235.55     4_465.51       0.9688          2.7528            1.0000        35.51
HnswRaBitQ-m16-ex8-ef60 (query)                        4_229.97       287.57     4_517.53       0.9859          2.5371            1.0000        35.51
HnswRaBitQ-m16-ex8-ef120 (query)                       4_229.97       353.27     4_583.23       0.9915          1.9298            1.0000        35.51
HnswRaBitQ-m16-ex8 (self)                              4_229.97    10_792.71    15_022.67       0.9880          2.0279            1.0000        35.51
HnswRaBitQ-m32-ex1-ef15 (query)                        2_439.52       195.04     2_634.56       0.7697          1.1137            1.0055        20.25
HnswRaBitQ-m32-ex1-ef30 (query)                        2_439.52       233.64     2_673.16       0.7753          1.0199            1.0054        20.25
HnswRaBitQ-m32-ex1-ef60 (query)                        2_439.52       283.80     2_723.32       0.7766          1.0066            1.0054        20.25
HnswRaBitQ-m32-ex1-ef120 (query)                       2_439.52       343.92     2_783.44       0.7767          1.0057            1.0054        20.25
HnswRaBitQ-m32-ex1 (self)                              2_439.52    13_357.35    15_796.87       0.6926          1.0164            1.0110        20.25
HnswRaBitQ-m32-ex3-ef15 (query)                        2_597.38       224.21     2_821.59       0.9115          1.0293            1.0005        26.36
HnswRaBitQ-m32-ex3-ef30 (query)                        2_597.38       270.11     2_867.49       0.9226          1.0060            1.0004        26.36
HnswRaBitQ-m32-ex3-ef60 (query)                        2_597.38       321.17     2_918.55       0.9262          1.0009            1.0004        26.36
HnswRaBitQ-m32-ex3-ef120 (query)                       2_597.38       382.74     2_980.12       0.9270          1.0006            1.0004        26.36
HnswRaBitQ-m32-ex3 (self)                              2_597.38    13_519.86    16_117.24       0.9016          1.0012            1.0008        26.36
HnswRaBitQ-m32-ex5-ef15 (query)                        2_856.91       242.65     3_099.56       0.9511          1.0650            1.0000        32.46
HnswRaBitQ-m32-ex5-ef30 (query)                        2_856.91       292.29     3_149.20       0.9683          1.0069            1.0000        32.46
HnswRaBitQ-m32-ex5-ef60 (query)                        2_856.91       347.76     3_204.67       0.9739          1.0022            1.0000        32.46
HnswRaBitQ-m32-ex5-ef120 (query)                       2_856.91       406.92     3_263.83       0.9754          1.0004            1.0000        32.46
HnswRaBitQ-m32-ex5 (self)                              2_856.91    13_546.98    16_403.89       0.9684          1.0018            1.0000        32.46
HnswRaBitQ-m32-ex8-ef15 (query)                        4_310.49       229.96     4_540.45       0.9658          1.0415            1.0000        41.62
HnswRaBitQ-m32-ex8-ef30 (query)                        4_310.49       276.45     4_586.94       0.9853          1.0361            1.0000        41.62
HnswRaBitQ-m32-ex8-ef60 (query)                        4_310.49       330.89     4_641.38       0.9922          1.0019            1.0000        41.62
HnswRaBitQ-m32-ex8-ef120 (query)                       4_310.49       400.92     4_711.41       0.9938          1.0001            1.0000        41.62
HnswRaBitQ-m32-ex8 (self)                              4_310.49    13_423.58    17_734.07       0.9938          1.0007            1.0000        41.62
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
Exhaustive (query)                                       102.09     1_957.82     2_059.91       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        102.09     6_647.37     6_749.47       1.0000          1.0000            1.0000       146.48
QG-d32-l64-ef15 (query)                                7_294.54       645.01     7_939.55       0.9842          1.0100            1.0000       311.66
QG-d32-l64-ef30 (query)                                7_294.54       845.12     8_139.66       0.9899          1.0098            1.0000       311.66
QG-d32-l64-ef60 (query)                                7_294.54     1_121.88     8_416.42       0.9926          1.0097            1.0000       311.66
QG-d32-l64-ef120 (query)                               7_294.54     1_495.90     8_790.44       0.9937          1.0095            1.0000       311.66
QG-d32-l64 (self)                                      7_294.54     3_636.07    10_930.61       0.9925          1.0100            1.0000       311.66
QG-d32-l128-ef15 (query)                               7_997.51       644.64     8_642.15       0.9856          1.0052            1.0000       311.66
QG-d32-l128-ef30 (query)                               7_997.51       850.30     8_847.81       0.9911          1.0050            1.0000       311.66
QG-d32-l128-ef60 (query)                               7_997.51     1_129.58     9_127.09       0.9937          1.0048            1.0000       311.66
QG-d32-l128-ef120 (query)                              7_997.51     1_517.53     9_515.04       0.9947          1.0047            1.0000       311.66
QG-d32-l128 (self)                                     7_997.51     3_725.82    11_723.33       0.9936          1.0047            1.0000       311.66
QG-d64-l64-ef15 (query)                               27_060.49     1_349.75    28_410.24       0.9975          1.0014            1.0000       476.46
QG-d64-l64-ef30 (query)                               27_060.49     1_730.49    28_790.99       0.9981          1.0013            1.0000       476.46
QG-d64-l64-ef60 (query)                               27_060.49     2_168.00    29_228.50       0.9984          1.0013            1.0000       476.46
QG-d64-l64-ef120 (query)                              27_060.49     2_820.11    29_880.60       0.9985          1.0013            1.0000       476.46
QG-d64-l64 (self)                                     27_060.49     7_114.71    34_175.21       0.9983          1.0015            1.0000       476.46
QG-d64-l128-ef15 (query)                              27_525.41     1_347.82    28_873.23       0.9972          1.0029            1.0000       476.46
QG-d64-l128-ef30 (query)                              27_525.41     1_710.69    29_236.09       0.9979          1.0029            1.0000       476.46
QG-d64-l128-ef60 (query)                              27_525.41     2_166.44    29_691.84       0.9982          1.0029            1.0000       476.46
QG-d64-l128-ef120 (query)                             27_525.41     2_836.52    30_361.92       0.9983          1.0028            1.0000       476.46
QG-d64-l128 (self)                                    27_525.41     7_127.27    34_652.68       0.9981          1.0028            1.0000       476.46
HnswRaBitQ-m16-ex1-ef15 (query)                        3_286.09       235.80     3_521.89       0.7638          1.0693            1.0045        17.45
HnswRaBitQ-m16-ex1-ef30 (query)                        3_286.09       285.61     3_571.70       0.7737          1.0091            1.0043        17.45
HnswRaBitQ-m16-ex1-ef60 (query)                        3_286.09       354.75     3_640.84       0.7771          1.0057            1.0042        17.45
HnswRaBitQ-m16-ex1-ef120 (query)                       3_286.09       421.07     3_707.16       0.7777          1.0048            1.0042        17.45
HnswRaBitQ-m16-ex1 (self)                              3_286.09    14_912.83    18_198.92       0.6924          1.0103            1.0087        17.45
HnswRaBitQ-m16-ex3-ef15 (query)                        3_528.95       269.19     3_798.15       0.8837          1.7282            1.0007        26.61
HnswRaBitQ-m16-ex3-ef30 (query)                        3_528.95       329.65     3_858.61       0.9106          1.3787            1.0004        26.61
HnswRaBitQ-m16-ex3-ef60 (query)                        3_528.95       403.49     3_932.45       0.9233          1.0022            1.0003        26.61
HnswRaBitQ-m16-ex3-ef120 (query)                       3_528.95       490.64     4_019.60       0.9268          1.0010            1.0003        26.61
HnswRaBitQ-m16-ex3 (self)                              3_528.95    14_669.84    18_198.79       0.8978          1.0986            1.0007        26.61
HnswRaBitQ-m16-ex5-ef15 (query)                        3_837.82       293.56     4_131.37       0.9111          2.2214            1.0002        35.76
HnswRaBitQ-m16-ex5-ef30 (query)                        3_837.82       356.39     4_194.21       0.9504          1.8128            1.0000        35.76
HnswRaBitQ-m16-ex5-ef60 (query)                        3_837.82       449.06     4_286.88       0.9704          1.0209            1.0000        35.76
HnswRaBitQ-m16-ex5-ef120 (query)                       3_837.82       525.10     4_362.91       0.9748          1.0004            1.0000        35.76
HnswRaBitQ-m16-ex5 (self)                              3_837.82    14_922.28    18_760.10       0.9645          1.0009            1.0000        35.76
HnswRaBitQ-m16-ex8-ef15 (query)                        6_253.79       271.95     6_525.74       0.9205          2.2004            1.0001        49.50
HnswRaBitQ-m16-ex8-ef30 (query)                        6_253.79       330.16     6_583.94       0.9618          1.9608            1.0000        49.50
HnswRaBitQ-m16-ex8-ef60 (query)                        6_253.79       421.11     6_674.90       0.9869          1.0016            1.0000        49.50
HnswRaBitQ-m16-ex8-ef120 (query)                       6_253.79       481.96     6_735.75       0.9927          1.0003            1.0000        49.50
HnswRaBitQ-m16-ex8 (self)                              6_253.79    14_339.48    20_593.27       0.9889          1.0016            1.0000        49.50
HnswRaBitQ-m32-ex1-ef15 (query)                        3_333.56       276.68     3_610.24       0.7738          1.0157            1.0043        23.56
HnswRaBitQ-m32-ex1-ef30 (query)                        3_333.56       332.02     3_665.58       0.7765          1.0082            1.0043        23.56
HnswRaBitQ-m32-ex1-ef60 (query)                        3_333.56       394.34     3_727.90       0.7778          1.0047            1.0042        23.56
HnswRaBitQ-m32-ex1-ef120 (query)                       3_333.56       472.63     3_806.19       0.7780          1.0045            1.0042        23.56
HnswRaBitQ-m32-ex1 (self)                              3_333.56    17_532.41    20_865.97       0.6923          1.0094            1.0088        23.56
HnswRaBitQ-m32-ex3-ef15 (query)                        3_639.44       322.94     3_962.37       0.9103          1.0137            1.0004        32.71
HnswRaBitQ-m32-ex3-ef30 (query)                        3_639.44       389.11     4_028.54       0.9226          1.0023            1.0003        32.71
HnswRaBitQ-m32-ex3-ef60 (query)                        3_639.44       457.56     4_096.99       0.9266          1.0006            1.0003        32.71
HnswRaBitQ-m32-ex3-ef120 (query)                       3_639.44       533.85     4_173.29       0.9275          1.0005            1.0003        32.71
HnswRaBitQ-m32-ex3 (self)                              3_639.44    18_152.92    21_792.35       0.9008          1.0011            1.0006        32.71
HnswRaBitQ-m32-ex5-ef15 (query)                        3_915.67       351.88     4_267.55       0.9497          1.0254            1.0000        41.87
HnswRaBitQ-m32-ex5-ef30 (query)                        3_915.67       420.03     4_335.70       0.9680          1.0057            1.0000        41.87
HnswRaBitQ-m32-ex5-ef60 (query)                        3_915.67       491.61     4_407.28       0.9742          1.0003            1.0000        41.87
HnswRaBitQ-m32-ex5-ef120 (query)                       3_915.67       575.54     4_491.20       0.9756          1.0001            1.0000        41.87
HnswRaBitQ-m32-ex5 (self)                              3_915.67    18_814.98    22_730.65       0.9678          1.0006            1.0000        41.87
HnswRaBitQ-m32-ex8-ef15 (query)                        6_328.76       315.83     6_644.59       0.9609          1.0358            1.0000        55.60
HnswRaBitQ-m32-ex8-ef30 (query)                        6_328.76       389.51     6_718.27       0.9832          1.0121            1.0000        55.60
HnswRaBitQ-m32-ex8-ef60 (query)                        6_328.76       448.34     6_777.10       0.9917          1.0027            1.0000        55.60
HnswRaBitQ-m32-ex8-ef120 (query)                       6_328.76       529.31     6_858.07       0.9936          1.0001            1.0000        55.60
HnswRaBitQ-m32-ex8 (self)                              6_328.76    18_013.27    24_342.03       0.9932          1.0038            1.0000        55.60
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
Exhaustive (query)                                        32.89       738.11       771.00       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.89     2_493.93     2_526.83       1.0000          1.0000            1.0000        48.83
QG-d32-l64-ef15 (query)                                3_058.30        84.90     3_143.20       0.9707          1.0009            1.0000       116.35
QG-d32-l64-ef30 (query)                                3_058.30       140.37     3_198.66       0.9983          1.0001            1.0000       116.35
QG-d32-l64-ef60 (query)                                3_058.30       258.98     3_317.27       0.9999          1.0000            1.0000       116.35
QG-d32-l64-ef120 (query)                               3_058.30       487.03     3_545.33       1.0000          1.0000            1.0000       116.35
QG-d32-l64 (self)                                      3_058.30       800.28     3_858.58       0.9999          1.0000            1.0000       116.35
QG-d32-l128-ef15 (query)                               3_416.93        81.93     3_498.86       0.9713          1.0009            1.0000       116.35
QG-d32-l128-ef30 (query)                               3_416.93       141.79     3_558.72       0.9986          1.0000            1.0000       116.35
QG-d32-l128-ef60 (query)                               3_416.93       259.51     3_676.43       1.0000          1.0000            1.0000       116.35
QG-d32-l128-ef120 (query)                              3_416.93       491.95     3_908.88       1.0000          1.0000            1.0000       116.35
QG-d32-l128 (self)                                     3_416.93       785.74     4_202.67       1.0000          1.0000            1.0000       116.35
QG-d64-l64-ef15 (query)                                5_428.08       118.96     5_547.04       0.9901          1.0002            1.0000       183.49
QG-d64-l64-ef30 (query)                                5_428.08       215.85     5_643.92       0.9999          1.0000            1.0000       183.49
QG-d64-l64-ef60 (query)                                5_428.08       408.61     5_836.69       1.0000          1.0000            1.0000       183.49
QG-d64-l64-ef120 (query)                               5_428.08       786.99     6_215.07       1.0000          1.0000            1.0000       183.49
QG-d64-l64 (self)                                      5_428.08     1_255.15     6_683.23       1.0000          1.0000            1.0000       183.49
QG-d64-l128-ef15 (query)                               6_236.21       117.52     6_353.74       0.9903          1.0002            1.0000       183.49
QG-d64-l128-ef30 (query)                               6_236.21       218.26     6_454.47       0.9999          1.0000            1.0000       183.49
QG-d64-l128-ef60 (query)                               6_236.21       410.83     6_647.04       1.0000          1.0000            1.0000       183.49
QG-d64-l128-ef120 (query)                              6_236.21       798.90     7_035.11       1.0000          1.0000            1.0000       183.49
QG-d64-l128 (self)                                     6_236.21     1_276.10     7_512.31       1.0000          1.0000            1.0000       183.49
HnswRaBitQ-m16-ex1-ef15 (query)                        1_335.91        78.01     1_413.92       0.8534          1.0073            1.0059        10.85
HnswRaBitQ-m16-ex1-ef30 (query)                        1_335.91        96.28     1_432.20       0.8663          1.0057            1.0049        10.85
HnswRaBitQ-m16-ex1-ef60 (query)                        1_335.91       123.66     1_459.57       0.8693          1.0053            1.0046        10.85
HnswRaBitQ-m16-ex1-ef120 (query)                       1_335.91       164.80     1_500.72       0.8697          1.0052            1.0046        10.85
HnswRaBitQ-m16-ex1 (self)                              1_335.91     5_839.44     7_175.35       0.8304          1.0109            1.0099        10.85
HnswRaBitQ-m16-ex3-ef15 (query)                        1_470.02        92.47     1_562.49       0.9281          1.0038            1.0008        13.90
HnswRaBitQ-m16-ex3-ef30 (query)                        1_470.02       107.28     1_577.30       0.9508          1.0010            1.0003        13.90
HnswRaBitQ-m16-ex3-ef60 (query)                        1_470.02       135.80     1_605.82       0.9569          1.0006            1.0001        13.90
HnswRaBitQ-m16-ex3-ef120 (query)                       1_470.02       179.49     1_649.51       0.9578          1.0005            1.0001        13.90
HnswRaBitQ-m16-ex3 (self)                              1_470.02     5_727.28     7_197.30       0.9469          1.0009            1.0005        13.90
HnswRaBitQ-m16-ex5-ef15 (query)                        1_534.97        90.19     1_625.16       0.9480          1.0024            1.0001        16.95
HnswRaBitQ-m16-ex5-ef30 (query)                        1_534.97       114.43     1_649.40       0.9772          1.0007            1.0000        16.95
HnswRaBitQ-m16-ex5-ef60 (query)                        1_534.97       143.05     1_678.02       0.9856          1.0002            1.0000        16.95
HnswRaBitQ-m16-ex5-ef120 (query)                       1_534.97       191.68     1_726.65       0.9869          1.0001            1.0000        16.95
HnswRaBitQ-m16-ex5 (self)                              1_534.97     5_917.46     7_452.43       0.9834          1.0002            1.0000        16.95
HnswRaBitQ-m16-ex8-ef15 (query)                        2_247.21        87.03     2_334.24       0.9527          1.0025            1.0000        21.53
HnswRaBitQ-m16-ex8-ef30 (query)                        2_247.21       108.30     2_355.51       0.9858          1.0006            1.0000        21.53
HnswRaBitQ-m16-ex8-ef60 (query)                        2_247.21       137.99     2_385.19       0.9956          1.0002            1.0000        21.53
HnswRaBitQ-m16-ex8-ef120 (query)                       2_247.21       180.19     2_427.39       0.9971          1.0001            1.0000        21.53
HnswRaBitQ-m16-ex8 (self)                              2_247.21     5_217.36     7_464.56       0.9965          1.0001            1.0000        21.53
HnswRaBitQ-m32-ex1-ef15 (query)                        1_402.99        88.34     1_491.33       0.8624          1.0061            1.0051        16.95
HnswRaBitQ-m32-ex1-ef30 (query)                        1_402.99       109.94     1_512.93       0.8686          1.0053            1.0047        16.95
HnswRaBitQ-m32-ex1-ef60 (query)                        1_402.99       140.34     1_543.33       0.8696          1.0052            1.0046        16.95
HnswRaBitQ-m32-ex1-ef120 (query)                       1_402.99       182.68     1_585.67       0.8698          1.0052            1.0046        16.95
HnswRaBitQ-m32-ex1 (self)                              1_402.99     5_883.96     7_286.95       0.8306          1.0108            1.0098        16.95
HnswRaBitQ-m32-ex3-ef15 (query)                        1_449.14       101.84     1_550.98       0.9426          1.0016            1.0004        20.00
HnswRaBitQ-m32-ex3-ef30 (query)                        1_449.14       118.80     1_567.95       0.9551          1.0006            1.0002        20.00
HnswRaBitQ-m32-ex3-ef60 (query)                        1_449.14       151.81     1_600.95       0.9576          1.0005            1.0001        20.00
HnswRaBitQ-m32-ex3-ef120 (query)                       1_449.14       196.56     1_645.70       0.9580          1.0004            1.0001        20.00
HnswRaBitQ-m32-ex3 (self)                              1_449.14     6_585.00     8_034.15       0.9475          1.0009            1.0005        20.00
HnswRaBitQ-m32-ex5-ef15 (query)                        1_558.47       103.03     1_661.51       0.9664          1.0012            1.0000        23.06
HnswRaBitQ-m32-ex5-ef30 (query)                        1_558.47       127.51     1_685.98       0.9831          1.0003            1.0000        23.06
HnswRaBitQ-m32-ex5-ef60 (query)                        1_558.47       160.07     1_718.55       0.9865          1.0001            1.0000        23.06
HnswRaBitQ-m32-ex5-ef120 (query)                       1_558.47       205.47     1_763.95       0.9870          1.0001            1.0000        23.06
HnswRaBitQ-m32-ex5 (self)                              1_558.47     6_644.45     8_202.92       0.9842          1.0001            1.0000        23.06
HnswRaBitQ-m32-ex8-ef15 (query)                        2_297.92        99.61     2_397.53       0.9737          1.0012            1.0000        27.63
HnswRaBitQ-m32-ex8-ef30 (query)                        2_297.92       123.62     2_421.54       0.9927          1.0002            1.0000        27.63
HnswRaBitQ-m32-ex8-ef60 (query)                        2_297.92       158.32     2_456.24       0.9968          1.0000            1.0000        27.63
HnswRaBitQ-m32-ex8-ef120 (query)                       2_297.92       211.37     2_509.29       0.9974          1.0000            1.0000        27.63
HnswRaBitQ-m32-ex8 (self)                              2_297.92     6_706.64     9_004.57       0.9975          1.0000            1.0000        27.63
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
Exhaustive (query)                                        70.13     1_361.03     1_431.16       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         70.13     4_559.51     4_629.63       1.0000          1.0000            1.0000        97.66
QG-d32-l64-ef15 (query)                                5_875.88       143.38     6_019.27       0.9595          1.0032            1.0000       214.01
QG-d32-l64-ef30 (query)                                5_875.88       235.26     6_111.15       0.9919          1.0013            1.0000       214.01
QG-d32-l64-ef60 (query)                                5_875.88       410.70     6_286.59       0.9981          1.0006            1.0000       214.01
QG-d32-l64-ef120 (query)                               5_875.88       728.30     6_604.18       0.9992          1.0004            1.0000       214.01
QG-d32-l64 (self)                                      5_875.88     1_256.68     7_132.57       0.9980          1.0007            1.0000       214.01
QG-d32-l128-ef15 (query)                               6_628.09       143.46     6_771.55       0.9605          1.0033            1.0000       214.01
QG-d32-l128-ef30 (query)                               6_628.09       238.96     6_867.06       0.9932          1.0012            1.0000       214.01
QG-d32-l128-ef60 (query)                               6_628.09       439.13     7_067.22       0.9987          1.0005            1.0000       214.01
QG-d32-l128-ef120 (query)                              6_628.09       790.52     7_418.62       0.9995          1.0003            1.0000       214.01
QG-d32-l128 (self)                                     6_628.09     1_312.46     7_940.55       0.9987          1.0006            1.0000       214.01
QG-d64-l64-ef15 (query)                               16_695.02       226.43    16_921.46       0.9920          1.0002            1.0000       329.97
QG-d64-l64-ef30 (query)                               16_695.02       389.98    17_085.01       0.9997          1.0000            1.0000       329.97
QG-d64-l64-ef60 (query)                               16_695.02       701.00    17_396.02       1.0000          1.0000            1.0000       329.97
QG-d64-l64-ef120 (query)                              16_695.02     1_239.88    17_934.91       1.0000          1.0000            1.0000       329.97
QG-d64-l64 (self)                                     16_695.02     2_161.91    18_856.94       1.0000          1.0000            1.0000       329.97
QG-d64-l128-ef15 (query)                              17_249.93       222.02    17_471.95       0.9921          1.0002            1.0000       329.97
QG-d64-l128-ef30 (query)                              17_249.93       388.61    17_638.54       0.9998          1.0000            1.0000       329.97
QG-d64-l128-ef60 (query)                              17_249.93       696.22    17_946.15       1.0000          1.0000            1.0000       329.97
QG-d64-l128-ef120 (query)                             17_249.93     1_254.70    18_504.63       1.0000          1.0000            1.0000       329.97
QG-d64-l128 (self)                                    17_249.93     2_171.39    19_421.32       1.0000          1.0000            1.0000       329.97
HnswRaBitQ-m16-ex1-ef15 (query)                        2_474.96       151.07     2_626.03       0.8453          1.0066            1.0043        14.15
HnswRaBitQ-m16-ex1-ef30 (query)                        2_474.96       183.60     2_658.56       0.8667          1.0043            1.0032        14.15
HnswRaBitQ-m16-ex1-ef60 (query)                        2_474.96       227.04     2_702.00       0.8734          1.0036            1.0029        14.15
HnswRaBitQ-m16-ex1-ef120 (query)                       2_474.96       281.86     2_756.82       0.8748          1.0034            1.0029        14.15
HnswRaBitQ-m16-ex1 (self)                              2_474.96    11_201.47    13_676.43       0.8359          1.0068            1.0060        14.15
HnswRaBitQ-m16-ex3-ef15 (query)                        2_672.84       170.38     2_843.21       0.9098          1.0036            1.0010        20.25
HnswRaBitQ-m16-ex3-ef30 (query)                        2_672.84       203.80     2_876.64       0.9446          1.0013            1.0003        20.25
HnswRaBitQ-m16-ex3-ef60 (query)                        2_672.84       248.87     2_921.70       0.9561          1.0006            1.0001        20.25
HnswRaBitQ-m16-ex3-ef120 (query)                       2_672.84       305.96     2_978.80       0.9585          1.0004            1.0001        20.25
HnswRaBitQ-m16-ex3 (self)                              2_672.84    11_477.99    14_150.82       0.9466          1.0008            1.0003        20.25
HnswRaBitQ-m16-ex5-ef15 (query)                        2_852.01       188.50     3_040.50       0.9240          1.0036            1.0007        26.36
HnswRaBitQ-m16-ex5-ef30 (query)                        2_852.01       211.91     3_063.92       0.9666          1.0011            1.0000        26.36
HnswRaBitQ-m16-ex5-ef60 (query)                        2_852.01       259.08     3_111.09       0.9826          1.0003            1.0000        26.36
HnswRaBitQ-m16-ex5-ef120 (query)                       2_852.01       318.71     3_170.72       0.9859          1.0002            1.0000        26.36
HnswRaBitQ-m16-ex5 (self)                              2_852.01    11_833.37    14_685.37       0.9811          1.0004            1.0000        26.36
HnswRaBitQ-m16-ex8-ef15 (query)                        4_488.81       173.64     4_662.45       0.9286          1.0033            1.0006        35.51
HnswRaBitQ-m16-ex8-ef30 (query)                        4_488.81       205.50     4_694.30       0.9742          1.0010            1.0000        35.51
HnswRaBitQ-m16-ex8-ef60 (query)                        4_488.81       256.97     4_745.77       0.9922          1.0003            1.0000        35.51
HnswRaBitQ-m16-ex8-ef120 (query)                       4_488.81       315.62     4_804.42       0.9963          1.0002            1.0000        35.51
HnswRaBitQ-m16-ex8 (self)                              4_488.81    11_515.96    16_004.77       0.9934          1.0003            1.0000        35.51
HnswRaBitQ-m32-ex1-ef15 (query)                        2_569.96       186.88     2_756.84       0.8626          1.0044            1.0034        20.25
HnswRaBitQ-m32-ex1-ef30 (query)                        2_569.96       224.95     2_794.91       0.8725          1.0035            1.0030        20.25
HnswRaBitQ-m32-ex1-ef60 (query)                        2_569.96       259.01     2_828.97       0.8747          1.0033            1.0029        20.25
HnswRaBitQ-m32-ex1-ef120 (query)                       2_569.96       316.63     2_886.59       0.8750          1.0032            1.0028        20.25
HnswRaBitQ-m32-ex1 (self)                              2_569.96    13_718.62    16_288.58       0.8366          1.0066            1.0060        20.25
HnswRaBitQ-m32-ex3-ef15 (query)                        2_690.40       196.57     2_886.97       0.9366          1.0015            1.0004        26.36
HnswRaBitQ-m32-ex3-ef30 (query)                        2_690.40       231.84     2_922.24       0.9541          1.0006            1.0001        26.36
HnswRaBitQ-m32-ex3-ef60 (query)                        2_690.40       299.55     2_989.95       0.9585          1.0003            1.0001        26.36
HnswRaBitQ-m32-ex3-ef120 (query)                       2_690.40       340.30     3_030.71       0.9590          1.0003            1.0001        26.36
HnswRaBitQ-m32-ex3 (self)                              2_690.40    14_241.78    16_932.18       0.9485          1.0006            1.0003        26.36
HnswRaBitQ-m32-ex5-ef15 (query)                        2_955.89       208.29     3_164.18       0.9573          1.0014            1.0000        32.46
HnswRaBitQ-m32-ex5-ef30 (query)                        2_955.89       246.56     3_202.44       0.9801          1.0003            1.0000        32.46
HnswRaBitQ-m32-ex5-ef60 (query)                        2_955.89       302.39     3_258.28       0.9859          1.0001            1.0000        32.46
HnswRaBitQ-m32-ex5-ef120 (query)                       2_955.89       356.41     3_312.30       0.9867          1.0001            1.0000        32.46
HnswRaBitQ-m32-ex5 (self)                              2_955.89    14_673.79    17_629.68       0.9838          1.0001            1.0000        32.46
HnswRaBitQ-m32-ex8-ef15 (query)                        4_430.89       200.64     4_631.53       0.9636          1.0017            1.0000        41.62
HnswRaBitQ-m32-ex8-ef30 (query)                        4_430.89       240.60     4_671.48       0.9888          1.0004            1.0000        41.62
HnswRaBitQ-m32-ex8-ef60 (query)                        4_430.89       290.06     4_720.95       0.9958          1.0001            1.0000        41.62
HnswRaBitQ-m32-ex8-ef120 (query)                       4_430.89       347.53     4_778.42       0.9969          1.0000            1.0000        41.62
HnswRaBitQ-m32-ex8 (self)                              4_430.89    14_267.70    18_698.59       0.9968          1.0001            1.0000        41.62
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
Exhaustive (query)                                       101.86     1_894.54     1_996.40       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.86     6_435.21     6_537.06       1.0000          1.0000            1.0000       146.48
QG-d32-l64-ef15 (query)                                7_913.84       251.41     8_165.26       0.9514          1.0069            1.0000       311.66
QG-d32-l64-ef30 (query)                                7_913.84       405.86     8_319.70       0.9809          1.0052            1.0000       311.66
QG-d32-l64-ef60 (query)                                7_913.84       662.11     8_575.95       0.9907          1.0043            1.0000       311.66
QG-d32-l64-ef120 (query)                               7_913.84     1_074.96     8_988.81       0.9938          1.0037            1.0000       311.66
QG-d32-l64 (self)                                      7_913.84     2_098.02    10_011.87       0.9907          1.0045            1.0000       311.66
QG-d32-l128-ef15 (query)                               8_772.64       255.06     9_027.69       0.9546          1.0064            1.0000       311.66
QG-d32-l128-ef30 (query)                               8_772.64       407.66     9_180.30       0.9835          1.0046            1.0000       311.66
QG-d32-l128-ef60 (query)                               8_772.64       668.27     9_440.91       0.9919          1.0038            1.0000       311.66
QG-d32-l128-ef120 (query)                              8_772.64     1_082.04     9_854.68       0.9944          1.0034            1.0000       311.66
QG-d32-l128 (self)                                     8_772.64     2_122.26    10_894.90       0.9922          1.0040            1.0000       311.66
QG-d64-l64-ef15 (query)                               31_742.74       519.17    32_261.92       0.9922          1.0010            1.0000       476.46
QG-d64-l64-ef30 (query)                               31_742.74       829.71    32_572.45       0.9982          1.0005            1.0000       476.46
QG-d64-l64-ef60 (query)                               31_742.74     1_369.55    33_112.29       0.9995          1.0002            1.0000       476.46
QG-d64-l64-ef120 (query)                              31_742.74     2_051.11    33_793.85       0.9998          1.0001            1.0000       476.46
QG-d64-l64 (self)                                     31_742.74     4_218.77    35_961.52       0.9995          1.0002            1.0000       476.46
QG-d64-l128-ef15 (query)                              30_656.27       498.73    31_155.00       0.9930          1.0008            1.0000       476.46
QG-d64-l128-ef30 (query)                              30_656.27       811.41    31_467.68       0.9987          1.0004            1.0000       476.46
QG-d64-l128-ef60 (query)                              30_656.27     1_300.39    31_956.66       0.9997          1.0002            1.0000       476.46
QG-d64-l128-ef120 (query)                             30_656.27     2_101.19    32_757.46       0.9999          1.0001            1.0000       476.46
QG-d64-l128 (self)                                    30_656.27     4_190.01    34_846.28       0.9997          1.0002            1.0000       476.46
HnswRaBitQ-m16-ex1-ef15 (query)                        3_496.70       217.99     3_714.69       0.7248        484.3941            1.0046        17.45
HnswRaBitQ-m16-ex1-ef30 (query)                        3_496.70       264.77     3_761.47       0.7819        308.8213            1.0032        17.45
HnswRaBitQ-m16-ex1-ef60 (query)                        3_496.70       353.42     3_850.12       0.8618          1.0100            1.0025        17.45
HnswRaBitQ-m16-ex1-ef120 (query)                       3_496.70       426.49     3_923.19       0.8643          1.0030            1.0024        17.45
HnswRaBitQ-m16-ex1 (self)                              3_496.70    17_264.39    20_761.09       0.8174          1.0300            1.0051        17.45
HnswRaBitQ-m16-ex3-ef15 (query)                        3_698.49       238.22     3_936.71       0.8863          1.0078            1.0013        26.61
HnswRaBitQ-m16-ex3-ef30 (query)                        3_698.49       287.19     3_985.68       0.9305          1.0030            1.0004        26.61
HnswRaBitQ-m16-ex3-ef60 (query)                        3_698.49       354.78     4_053.27       0.9490          1.0011            1.0002        26.61
HnswRaBitQ-m16-ex3-ef120 (query)                       3_698.49       435.90     4_134.39       0.9538          1.0006            1.0001        26.61
HnswRaBitQ-m16-ex3 (self)                              3_698.49    17_210.39    20_908.88       0.9380          1.0012            1.0003        26.61
HnswRaBitQ-m16-ex5-ef15 (query)                        4_081.36       264.35     4_345.71       0.9021          1.0069            1.0010        35.76
HnswRaBitQ-m16-ex5-ef30 (query)                        4_081.36       306.76     4_388.12       0.9537          1.0023            1.0000        35.76
HnswRaBitQ-m16-ex5-ef60 (query)                        4_081.36       420.06     4_501.42       0.9771          1.0007            1.0000        35.76
HnswRaBitQ-m16-ex5-ef120 (query)                       4_081.36       453.63     4_534.99       0.9835          1.0003            1.0000        35.76
HnswRaBitQ-m16-ex5 (self)                              4_081.36    17_607.13    21_688.49       0.9755          1.0008            1.0000        35.76
HnswRaBitQ-m16-ex8-ef15 (query)                        6_265.64       252.07     6_517.71       0.8617        177.1225            1.0012        49.50
HnswRaBitQ-m16-ex8-ef30 (query)                        6_265.64       316.39     6_582.04       0.9573          3.9205            1.0000        49.50
HnswRaBitQ-m16-ex8-ef60 (query)                        6_265.64       390.94     6_656.58       0.9872          1.0007            1.0000        49.50
HnswRaBitQ-m16-ex8-ef120 (query)                       6_265.64       484.73     6_750.37       0.9948          1.0003            1.0000        49.50
HnswRaBitQ-m16-ex8 (self)                              6_265.64    17_034.35    23_299.99       0.9892          1.0007            1.0000        49.50
HnswRaBitQ-m32-ex1-ef15 (query)                        3_459.03       272.13     3_731.15       0.8059        205.3126            1.0032        23.56
HnswRaBitQ-m32-ex1-ef30 (query)                        3_459.03       333.13     3_792.16       0.8520         38.3574            1.0026        23.56
HnswRaBitQ-m32-ex1-ef60 (query)                        3_459.03       404.99     3_864.01       0.8625          1.4037            1.0024        23.56
HnswRaBitQ-m32-ex1-ef120 (query)                       3_459.03       510.06     3_969.08       0.8632          1.3838            1.0024        23.56
HnswRaBitQ-m32-ex1 (self)                              3_459.03    21_735.96    25_194.99       0.8178          1.6718            1.0051        23.56
HnswRaBitQ-m32-ex3-ef15 (query)                        3_712.89       299.77     4_012.66       0.9244          1.0034            1.0004        32.71
HnswRaBitQ-m32-ex3-ef30 (query)                        3_712.89       334.05     4_046.94       0.9470          1.0010            1.0002        32.71
HnswRaBitQ-m32-ex3-ef60 (query)                        3_712.89       404.71     4_117.60       0.9537          1.0004            1.0001        32.71
HnswRaBitQ-m32-ex3-ef120 (query)                       3_712.89       484.37     4_197.26       0.9547          1.0003            1.0001        32.71
HnswRaBitQ-m32-ex3 (self)                              3_712.89    21_654.97    25_367.85       0.9414          1.0007            1.0003        32.71
HnswRaBitQ-m32-ex5-ef15 (query)                        4_125.15       327.24     4_452.39       0.8816        234.7577            1.0001        41.87
HnswRaBitQ-m32-ex5-ef30 (query)                        4_125.15       397.49     4_522.64       0.9564         60.1448            1.0000        41.87
HnswRaBitQ-m32-ex5-ef60 (query)                        4_125.15       473.28     4_598.43       0.9833          1.0002            1.0000        41.87
HnswRaBitQ-m32-ex5-ef120 (query)                       4_125.15       555.08     4_680.23       0.9848          1.0001            1.0000        41.87
HnswRaBitQ-m32-ex5 (self)                              4_125.15    22_464.75    26_589.90       0.9805          1.0040            1.0000        41.87
HnswRaBitQ-m32-ex8-ef15 (query)                        6_401.83       301.00     6_702.84       0.8533        393.3943            1.0000        55.60
HnswRaBitQ-m32-ex8-ef30 (query)                        6_401.83       374.50     6_776.33       0.9548         86.6394            1.0000        55.60
HnswRaBitQ-m32-ex8-ef60 (query)                        6_401.83       443.56     6_845.39       0.9942          1.0002            1.0000        55.60
HnswRaBitQ-m32-ex8-ef120 (query)                       6_401.83       524.20     6_926.03       0.9962          1.0001            1.0000        55.60
HnswRaBitQ-m32-ex8 (self)                              6_401.83    21_579.82    27_981.66       0.9952          1.0019            1.0000        55.60
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
Exhaustive (query)                                        32.79       757.60       790.39       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.79     2_433.54     2_466.34       1.0000          1.0000            1.0000        48.83
QG-d32-l64-ef15 (query)                                2_384.15        56.58     2_440.73       0.9888          1.0005            1.0000       116.35
QG-d32-l64-ef30 (query)                                2_384.15        87.73     2_471.87       0.9999          1.0000            1.0000       116.35
QG-d32-l64-ef60 (query)                                2_384.15       149.59     2_533.74       1.0000          1.0000            1.0000       116.35
QG-d32-l64-ef120 (query)                               2_384.15       280.77     2_664.91       1.0000          1.0000            1.0000       116.35
QG-d32-l64 (self)                                      2_384.15       465.85     2_850.00       1.0000          1.0000            1.0000       116.35
QG-d32-l128-ef15 (query)                               3_594.54        56.24     3_650.78       0.9888          1.0006            1.0000       116.35
QG-d32-l128-ef30 (query)                               3_594.54        86.15     3_680.69       0.9999          1.0000            1.0000       116.35
QG-d32-l128-ef60 (query)                               3_594.54       148.90     3_743.44       1.0000          1.0000            1.0000       116.35
QG-d32-l128-ef120 (query)                              3_594.54       280.83     3_875.37       1.0000          1.0000            1.0000       116.35
QG-d32-l128 (self)                                     3_594.54       456.53     4_051.07       1.0000          1.0000            1.0000       116.35
QG-d64-l64-ef15 (query)                                2_961.64        65.61     3_027.25       0.9906          1.0004            1.0000       183.49
QG-d64-l64-ef30 (query)                                2_961.64        96.85     3_058.49       1.0000          1.0000            1.0000       183.49
QG-d64-l64-ef60 (query)                                2_961.64       168.62     3_130.26       1.0000          1.0000            1.0000       183.49
QG-d64-l64-ef120 (query)                               2_961.64       339.07     3_300.71       1.0000          1.0000            1.0000       183.49
QG-d64-l64 (self)                                      2_961.64       532.11     3_493.75       1.0000          1.0000            1.0000       183.49
QG-d64-l128-ef15 (query)                               4_559.09        61.53     4_620.62       0.9906          1.0004            1.0000       183.49
QG-d64-l128-ef30 (query)                               4_559.09        96.85     4_655.94       1.0000          1.0000            1.0000       183.49
QG-d64-l128-ef60 (query)                               4_559.09       170.31     4_729.40       1.0000          1.0000            1.0000       183.49
QG-d64-l128-ef120 (query)                              4_559.09       330.06     4_889.15       1.0000          1.0000            1.0000       183.49
QG-d64-l128 (self)                                     4_559.09       584.60     5_143.69       1.0000          1.0000            1.0000       183.49
HnswRaBitQ-m16-ex1-ef15 (query)                        1_580.27        64.31     1_644.58       0.9363          1.0282            1.0033        10.85
HnswRaBitQ-m16-ex1-ef30 (query)                        1_580.27        74.73     1_655.00       0.9376          1.0259            1.0031        10.85
HnswRaBitQ-m16-ex1-ef60 (query)                        1_580.27        94.65     1_674.92       0.9388          1.0170            1.0031        10.85
HnswRaBitQ-m16-ex1-ef120 (query)                       1_580.27       125.91     1_706.19       0.9393          1.0145            1.0031        10.85
HnswRaBitQ-m16-ex1 (self)                              1_580.27     3_474.58     5_054.85       0.9117          1.0217            1.0088        10.85
HnswRaBitQ-m16-ex3-ef15 (query)                        1_669.21        66.48     1_735.68       0.9764          1.0123            1.0000        13.90
HnswRaBitQ-m16-ex3-ef30 (query)                        1_669.21        78.26     1_747.46       0.9787          1.0104            1.0000        13.90
HnswRaBitQ-m16-ex3-ef60 (query)                        1_669.21        97.46     1_766.66       0.9791          1.0089            1.0000        13.90
HnswRaBitQ-m16-ex3-ef120 (query)                       1_669.21       132.51     1_801.71       0.9795          1.0069            1.0000        13.90
HnswRaBitQ-m16-ex3 (self)                              1_669.21     3_507.00     5_176.21       0.9733          1.0052            1.0000        13.90
HnswRaBitQ-m16-ex5-ef15 (query)                        1_769.96        72.49     1_842.45       0.9891          1.0194            1.0000        16.95
HnswRaBitQ-m16-ex5-ef30 (query)                        1_769.96        83.69     1_853.65       0.9929          1.0090            1.0000        16.95
HnswRaBitQ-m16-ex5-ef60 (query)                        1_769.96       104.04     1_874.01       0.9938          1.0039            1.0000        16.95
HnswRaBitQ-m16-ex5-ef120 (query)                       1_769.96       141.72     1_911.68       0.9941          1.0020            1.0000        16.95
HnswRaBitQ-m16-ex5 (self)                              1_769.96     3_599.79     5_369.75       0.9913          1.0083            1.0000        16.95
HnswRaBitQ-m16-ex8-ef15 (query)                        2_492.63        70.04     2_562.68       0.9924          1.0268            1.0000        21.53
HnswRaBitQ-m16-ex8-ef30 (query)                        2_492.63        83.78     2_576.41       0.9959          1.0206            1.0000        21.53
HnswRaBitQ-m16-ex8-ef60 (query)                        2_492.63       103.36     2_595.99       0.9969          1.0135            1.0000        21.53
HnswRaBitQ-m16-ex8-ef120 (query)                       2_492.63       142.02     2_634.66       0.9977          1.0076            1.0000        21.53
HnswRaBitQ-m16-ex8 (self)                              2_492.63     3_534.15     6_026.78       0.9973          1.0107            1.0000        21.53
HnswRaBitQ-m32-ex1-ef15 (query)                        1_647.13        70.60     1_717.73       0.9380          1.0192            1.0032        16.95
HnswRaBitQ-m32-ex1-ef30 (query)                        1_647.13        83.87     1_731.00       0.9393          1.0142            1.0031        16.95
HnswRaBitQ-m32-ex1-ef60 (query)                        1_647.13       104.58     1_751.72       0.9398          1.0112            1.0031        16.95
HnswRaBitQ-m32-ex1-ef120 (query)                       1_647.13       148.49     1_795.62       0.9404          1.0074            1.0031        16.95
HnswRaBitQ-m32-ex1 (self)                              1_647.13     3_923.95     5_571.09       0.9124          1.0176            1.0088        16.95
HnswRaBitQ-m32-ex3-ef15 (query)                        1_728.50        78.09     1_806.59       0.9791          1.0026            1.0000        20.00
HnswRaBitQ-m32-ex3-ef30 (query)                        1_728.50        91.67     1_820.17       0.9806          1.0016            1.0000        20.00
HnswRaBitQ-m32-ex3-ef60 (query)                        1_728.50       114.17     1_842.67       0.9808          1.0011            1.0000        20.00
HnswRaBitQ-m32-ex3-ef120 (query)                       1_728.50       155.95     1_884.46       0.9809          1.0008            1.0000        20.00
HnswRaBitQ-m32-ex3 (self)                              1_728.50     3_990.19     5_718.69       0.9742          1.0012            1.0000        20.00
HnswRaBitQ-m32-ex5-ef15 (query)                        1_838.66        83.14     1_921.80       0.9916          1.0100            1.0000        23.06
HnswRaBitQ-m32-ex5-ef30 (query)                        1_838.66        95.87     1_934.53       0.9934          1.0070            1.0000        23.06
HnswRaBitQ-m32-ex5-ef60 (query)                        1_838.66       120.62     1_959.29       0.9941          1.0034            1.0000        23.06
HnswRaBitQ-m32-ex5-ef120 (query)                       1_838.66       159.02     1_997.68       0.9944          1.0014            1.0000        23.06
HnswRaBitQ-m32-ex5 (self)                              1_838.66     3_797.43     5_636.10       0.9921          1.0040            1.0000        23.06
HnswRaBitQ-m32-ex8-ef15 (query)                        2_556.52        79.90     2_636.41       0.9957          1.0089            1.0000        27.63
HnswRaBitQ-m32-ex8-ef30 (query)                        2_556.52        94.97     2_651.48       0.9977          1.0068            1.0000        27.63
HnswRaBitQ-m32-ex8-ef60 (query)                        2_556.52       118.06     2_674.58       0.9982          1.0056            1.0000        27.63
HnswRaBitQ-m32-ex8-ef120 (query)                       2_556.52       158.64     2_715.16       0.9990          1.0012            1.0000        27.63
HnswRaBitQ-m32-ex8 (self)                              2_556.52     3_936.20     6_492.71       0.9983          1.0044            1.0000        27.63
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
Exhaustive (query)                                        68.69     1_356.47     1_425.17       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.69     4_502.82     4_571.52       1.0000          1.0000            1.0000        97.66
QG-d32-l64-ef15 (query)                                4_411.36        86.06     4_497.42       0.9907          1.0005            1.0000       214.01
QG-d32-l64-ef30 (query)                                4_411.36       122.67     4_534.03       0.9999          1.0000            1.0000       214.01
QG-d32-l64-ef60 (query)                                4_411.36       192.57     4_603.93       1.0000          1.0000            1.0000       214.01
QG-d32-l64-ef120 (query)                               4_411.36       347.50     4_758.86       1.0000          1.0000            1.0000       214.01
QG-d32-l64 (self)                                      4_411.36       595.54     5_006.90       1.0000          1.0000            1.0000       214.01
QG-d32-l128-ef15 (query)                               6_872.18       107.86     6_980.04       0.9904          1.0005            1.0000       214.01
QG-d32-l128-ef30 (query)                               6_872.18       123.22     6_995.40       0.9999          1.0000            1.0000       214.01
QG-d32-l128-ef60 (query)                               6_872.18       196.34     7_068.52       1.0000          1.0000            1.0000       214.01
QG-d32-l128-ef120 (query)                              6_872.18       349.24     7_221.42       1.0000          1.0000            1.0000       214.01
QG-d32-l128 (self)                                     6_872.18       579.23     7_451.41       1.0000          1.0000            1.0000       214.01
QG-d64-l64-ef15 (query)                                5_468.47        92.95     5_561.43       0.9917          1.0004            1.0000       329.97
QG-d64-l64-ef30 (query)                                5_468.47       134.25     5_602.72       0.9999          1.0000            1.0000       329.97
QG-d64-l64-ef60 (query)                                5_468.47       214.83     5_683.30       1.0000          1.0000            1.0000       329.97
QG-d64-l64-ef120 (query)                               5_468.47       391.32     5_859.79       1.0000          1.0000            1.0000       329.97
QG-d64-l64 (self)                                      5_468.47       652.71     6_121.18       1.0000          1.0000            1.0000       329.97
QG-d64-l128-ef15 (query)                               8_725.29        90.49     8_815.78       0.9917          1.0004            1.0000       329.97
QG-d64-l128-ef30 (query)                               8_725.29       135.99     8_861.29       0.9999          1.0000            1.0000       329.97
QG-d64-l128-ef60 (query)                               8_725.29       217.70     8_942.99       1.0000          1.0000            1.0000       329.97
QG-d64-l128-ef120 (query)                              8_725.29       395.27     9_120.56       1.0000          1.0000            1.0000       329.97
QG-d64-l128 (self)                                     8_725.29       654.88     9_380.17       1.0000          1.0000            1.0000       329.97
HnswRaBitQ-m16-ex1-ef15 (query)                        2_646.06       114.81     2_760.88       0.9530          1.0347            1.0008        14.15
HnswRaBitQ-m16-ex1-ef30 (query)                        2_646.06       129.01     2_775.07       0.9544          1.0318            1.0006        14.15
HnswRaBitQ-m16-ex1-ef60 (query)                        2_646.06       154.76     2_800.82       0.9550          1.0268            1.0006        14.15
HnswRaBitQ-m16-ex1-ef120 (query)                       2_646.06       198.73     2_844.80       0.9553          1.0223            1.0006        14.15
HnswRaBitQ-m16-ex1 (self)                              2_646.06     5_637.90     8_283.96       0.9323          1.0230            1.0040        14.15
HnswRaBitQ-m16-ex3-ef15 (query)                        2_835.26       127.57     2_962.83       0.9741          1.1325            1.0000        20.25
HnswRaBitQ-m16-ex3-ef30 (query)                        2_835.26       141.26     2_976.52       0.9788          1.0920            1.0000        20.25
HnswRaBitQ-m16-ex3-ef60 (query)                        2_835.26       166.79     3_002.05       0.9812          1.0600            1.0000        20.25
HnswRaBitQ-m16-ex3-ef120 (query)                       2_835.26       215.15     3_050.41       0.9833          1.0340            1.0000        20.25
HnswRaBitQ-m16-ex3 (self)                              2_835.26     5_921.83     8_757.08       0.9771          1.0554            1.0000        20.25
HnswRaBitQ-m16-ex5-ef15 (query)                        3_025.79       136.44     3_162.23       0.9925          1.0152            1.0000        26.36
HnswRaBitQ-m16-ex5-ef30 (query)                        3_025.79       147.32     3_173.11       0.9949          1.0108            1.0000        26.36
HnswRaBitQ-m16-ex5-ef60 (query)                        3_025.79       177.97     3_203.76       0.9954          1.0078            1.0000        26.36
HnswRaBitQ-m16-ex5-ef120 (query)                       3_025.79       219.49     3_245.28       0.9956          1.0054            1.0000        26.36
HnswRaBitQ-m16-ex5 (self)                              3_025.79     6_063.89     9_089.68       0.9937          1.0091            1.0000        26.36
HnswRaBitQ-m16-ex8-ef15 (query)                        4_521.81       128.28     4_650.09       0.9925          1.0580            1.0000        35.51
HnswRaBitQ-m16-ex8-ef30 (query)                        4_521.81       144.04     4_665.84       0.9959          1.0382            1.0000        35.51
HnswRaBitQ-m16-ex8-ef60 (query)                        4_521.81       170.78     4_692.58       0.9969          1.0273            1.0000        35.51
HnswRaBitQ-m16-ex8-ef120 (query)                       4_521.81       216.46     4_738.27       0.9974          1.0200            1.0000        35.51
HnswRaBitQ-m16-ex8 (self)                              4_521.81     5_929.20    10_451.01       0.9972          1.0198            1.0000        35.51
HnswRaBitQ-m32-ex1-ef15 (query)                        2_742.50       132.13     2_874.63       0.9571          1.0025            1.0006        20.25
HnswRaBitQ-m32-ex1-ef30 (query)                        2_742.50       148.04     2_890.53       0.9578          1.0023            1.0006        20.25
HnswRaBitQ-m32-ex1-ef60 (query)                        2_742.50       174.75     2_917.24       0.9579          1.0023            1.0006        20.25
HnswRaBitQ-m32-ex1-ef120 (query)                       2_742.50       221.61     2_964.11       0.9579          1.0023            1.0006        20.25
HnswRaBitQ-m32-ex1 (self)                              2_742.50     6_551.81     9_294.31       0.9344          1.0070            1.0039        20.25
HnswRaBitQ-m32-ex3-ef15 (query)                        2_897.74       140.66     3_038.41       0.9847          1.0004            1.0000        26.36
HnswRaBitQ-m32-ex3-ef30 (query)                        2_897.74       161.15     3_058.89       0.9859          1.0002            1.0000        26.36
HnswRaBitQ-m32-ex3-ef60 (query)                        2_897.74       187.97     3_085.71       0.9860          1.0002            1.0000        26.36
HnswRaBitQ-m32-ex3-ef120 (query)                       2_897.74       237.36     3_135.10       0.9860          1.0002            1.0000        26.36
HnswRaBitQ-m32-ex3 (self)                              2_897.74     6_803.93     9_701.67       0.9815          1.0005            1.0000        26.36
HnswRaBitQ-m32-ex5-ef15 (query)                        3_114.62       149.15     3_263.77       0.9945          1.0002            1.0000        32.46
HnswRaBitQ-m32-ex5-ef30 (query)                        3_114.62       166.18     3_280.80       0.9962          1.0000            1.0000        32.46
HnswRaBitQ-m32-ex5-ef60 (query)                        3_114.62       195.87     3_310.49       0.9963          1.0000            1.0000        32.46
HnswRaBitQ-m32-ex5-ef120 (query)                       3_114.62       252.25     3_366.86       0.9963          1.0000            1.0000        32.46
HnswRaBitQ-m32-ex5 (self)                              3_114.62     6_979.07    10_093.69       0.9945          1.0003            1.0000        32.46
HnswRaBitQ-m32-ex8-ef15 (query)                        4_671.13       142.88     4_814.01       0.9975          1.0002            1.0000        41.62
HnswRaBitQ-m32-ex8-ef30 (query)                        4_671.13       160.20     4_831.33       0.9994          1.0000            1.0000        41.62
HnswRaBitQ-m32-ex8-ef60 (query)                        4_671.13       190.03     4_861.16       0.9994          1.0000            1.0000        41.62
HnswRaBitQ-m32-ex8-ef120 (query)                       4_671.13       239.91     4_911.04       0.9995          1.0000            1.0000        41.62
HnswRaBitQ-m32-ex8 (self)                              4_671.13     6_772.30    11_443.43       0.9993          1.0001            1.0000        41.62
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
Exhaustive (query)                                       113.21     1_940.63     2_053.84       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        113.21     6_722.52     6_835.73       1.0000          1.0000            1.0000       146.48
QG-d32-l64-ef15 (query)                                6_305.45       121.83     6_427.28       0.9897          1.0005            1.0000       311.66
QG-d32-l64-ef30 (query)                                6_305.45       147.90     6_453.35       0.9998          1.0000            1.0000       311.66
QG-d32-l64-ef60 (query)                                6_305.45       233.25     6_538.70       1.0000          1.0000            1.0000       311.66
QG-d32-l64-ef120 (query)                               6_305.45       416.88     6_722.33       1.0000          1.0000            1.0000       311.66
QG-d32-l64 (self)                                      6_305.45       680.08     6_985.53       1.0000          1.0000            1.0000       311.66
QG-d32-l128-ef15 (query)                               9_762.54       106.15     9_868.69       0.9895          1.0005            1.0000       311.66
QG-d32-l128-ef30 (query)                               9_762.54       147.87     9_910.41       0.9998          1.0000            1.0000       311.66
QG-d32-l128-ef60 (query)                               9_762.54       233.10     9_995.64       1.0000          1.0000            1.0000       311.66
QG-d32-l128-ef120 (query)                              9_762.54       408.88    10_171.42       1.0000          1.0000            1.0000       311.66
QG-d32-l128 (self)                                     9_762.54       681.71    10_444.25       1.0000          1.0000            1.0000       311.66
QG-d64-l64-ef15 (query)                                7_774.63       120.73     7_895.36       0.9907          1.0004            1.0000       476.46
QG-d64-l64-ef30 (query)                                7_774.63       165.55     7_940.18       0.9999          1.0000            1.0000       476.46
QG-d64-l64-ef60 (query)                                7_774.63       269.32     8_043.95       1.0000          1.0000            1.0000       476.46
QG-d64-l64-ef120 (query)                               7_774.63       481.23     8_255.86       1.0000          1.0000            1.0000       476.46
QG-d64-l64 (self)                                      7_774.63       800.87     8_575.50       1.0000          1.0000            1.0000       476.46
QG-d64-l128-ef15 (query)                              12_134.65       117.59    12_252.24       0.9909          1.0004            1.0000       476.46
QG-d64-l128-ef30 (query)                              12_134.65       165.92    12_300.57       0.9999          1.0000            1.0000       476.46
QG-d64-l128-ef60 (query)                              12_134.65       290.54    12_425.18       1.0000          1.0000            1.0000       476.46
QG-d64-l128-ef120 (query)                             12_134.65       488.78    12_623.43       1.0000          1.0000            1.0000       476.46
QG-d64-l128 (self)                                    12_134.65       813.89    12_948.54       1.0000          1.0000            1.0000       476.46
HnswRaBitQ-m16-ex1-ef15 (query)                        3_719.31       160.15     3_879.46       0.9491          1.0302            1.0014        17.45
HnswRaBitQ-m16-ex1-ef30 (query)                        3_719.31       177.88     3_897.18       0.9504          1.0242            1.0013        17.45
HnswRaBitQ-m16-ex1-ef60 (query)                        3_719.31       203.85     3_923.16       0.9510          1.0181            1.0013        17.45
HnswRaBitQ-m16-ex1-ef120 (query)                       3_719.31       266.05     3_985.36       0.9515          1.0117            1.0013        17.45
HnswRaBitQ-m16-ex1 (self)                              3_719.31     7_851.08    11_570.39       0.9210          1.0189            1.0064        17.45
HnswRaBitQ-m16-ex3-ef15 (query)                        3_976.35       167.75     4_144.10       0.9755          1.0198            1.0000        26.61
HnswRaBitQ-m16-ex3-ef30 (query)                        3_976.35       189.07     4_165.41       0.9772          1.0167            1.0000        26.61
HnswRaBitQ-m16-ex3-ef60 (query)                        3_976.35       221.51     4_197.86       0.9777          1.0105            1.0000        26.61
HnswRaBitQ-m16-ex3-ef120 (query)                       3_976.35       306.75     4_283.09       0.9781          1.0064            1.0000        26.61
HnswRaBitQ-m16-ex3 (self)                              3_976.35     8_091.00    12_067.34       0.9702          1.0133            1.0000        26.61
HnswRaBitQ-m16-ex5-ef15 (query)                        4_250.77       179.90     4_430.66       0.9907          1.0314            1.0000        35.76
HnswRaBitQ-m16-ex5-ef30 (query)                        4_250.77       198.93     4_449.70       0.9931          1.0285            1.0000        35.76
HnswRaBitQ-m16-ex5-ef60 (query)                        4_250.77       228.30     4_479.07       0.9938          1.0210            1.0000        35.76
HnswRaBitQ-m16-ex5-ef120 (query)                       4_250.77       303.21     4_553.98       0.9945          1.0085            1.0000        35.76
HnswRaBitQ-m16-ex5 (self)                              4_250.77     8_271.39    12_522.16       0.9925          1.0127            1.0000        35.76
HnswRaBitQ-m16-ex8-ef15 (query)                        5_952.67       190.33     6_143.00       0.9961          1.0074            1.0000        49.50
HnswRaBitQ-m16-ex8-ef30 (query)                        5_952.67       186.09     6_138.76       0.9986          1.0064            1.0000        49.50
HnswRaBitQ-m16-ex8-ef60 (query)                        5_952.67       218.11     6_170.78       0.9988          1.0051            1.0000        49.50
HnswRaBitQ-m16-ex8-ef120 (query)                       5_952.67       277.73     6_230.40       0.9989          1.0032            1.0000        49.50
HnswRaBitQ-m16-ex8 (self)                              5_952.67     7_980.54    13_933.20       0.9987          1.0063            1.0000        49.50
HnswRaBitQ-m32-ex1-ef15 (query)                        3_826.44       175.65     4_002.09       0.9514          1.0041            1.0013        23.56
HnswRaBitQ-m32-ex1-ef30 (query)                        3_826.44       197.17     4_023.61       0.9521          1.0039            1.0013        23.56
HnswRaBitQ-m32-ex1-ef60 (query)                        3_826.44       228.74     4_055.18       0.9521          1.0039            1.0013        23.56
HnswRaBitQ-m32-ex1-ef120 (query)                       3_826.44       286.12     4_112.56       0.9521          1.0039            1.0013        23.56
HnswRaBitQ-m32-ex1 (self)                              3_826.44     8_976.54    12_802.98       0.9216          1.0109            1.0063        23.56
HnswRaBitQ-m32-ex3-ef15 (query)                        4_044.96       196.79     4_241.75       0.9773          1.0015            1.0000        32.71
HnswRaBitQ-m32-ex3-ef30 (query)                        4_044.96       217.16     4_262.12       0.9784          1.0013            1.0000        32.71
HnswRaBitQ-m32-ex3-ef60 (query)                        4_044.96       248.91     4_293.87       0.9785          1.0013            1.0000        32.71
HnswRaBitQ-m32-ex3-ef120 (query)                       4_044.96       303.86     4_348.82       0.9786          1.0008            1.0000        32.71
HnswRaBitQ-m32-ex3 (self)                              4_044.96     9_397.59    13_442.55       0.9710          1.0016            1.0000        32.71
HnswRaBitQ-m32-ex5-ef15 (query)                        4_336.75       215.35     4_552.10       0.9935          1.0002            1.0000        41.87
HnswRaBitQ-m32-ex5-ef30 (query)                        4_336.75       225.49     4_562.24       0.9951          1.0000            1.0000        41.87
HnswRaBitQ-m32-ex5-ef60 (query)                        4_336.75       264.29     4_601.04       0.9952          1.0000            1.0000        41.87
HnswRaBitQ-m32-ex5-ef120 (query)                       4_336.75       319.09     4_655.84       0.9952          1.0000            1.0000        41.87
HnswRaBitQ-m32-ex5 (self)                              4_336.75     9_587.20    13_923.95       0.9933          1.0001            1.0000        41.87
HnswRaBitQ-m32-ex8-ef15 (query)                        6_104.09       185.68     6_289.77       0.9973          1.0008            1.0000        55.60
HnswRaBitQ-m32-ex8-ef30 (query)                        6_104.09       207.10     6_311.19       0.9991          1.0007            1.0000        55.60
HnswRaBitQ-m32-ex8-ef60 (query)                        6_104.09       242.27     6_346.36       0.9992          1.0007            1.0000        55.60
HnswRaBitQ-m32-ex8-ef120 (query)                       6_104.09       303.75     6_407.85       0.9992          1.0007            1.0000        55.60
HnswRaBitQ-m32-ex8 (self)                              6_104.09     9_323.51    15_427.61       0.9991          1.0004            1.0000        55.60
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
Exhaustive (query)                                        34.48       703.18       737.66       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         34.48     2_322.21     2_356.70       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              188.45       361.31       549.76       0.0971          1.7176            1.5958         7.12
ExhaustiveTQ-b2-rf5 (query)                              188.45       456.44       644.88       0.2336          1.2025            1.2204         7.12
ExhaustiveTQ-b2-rf10 (query)                             188.45       594.30       782.75       0.2853          1.1453            1.1620         7.12
ExhaustiveTQ-b2-rf20 (query)                             188.45       993.75     1_182.20       0.3809          1.0970            1.0941         7.12
ExhaustiveTQ-b2 (self)                                   188.45     3_202.35     3_390.79       0.3814          1.0980            1.0957         7.12
ExhaustiveTQ-b4-rf0 (query)                              272.59       581.42       854.01       0.1094          1.5328            1.4997        13.22
ExhaustiveTQ-b4-rf5 (query)                              272.59       671.56       944.15       0.2368          1.1884            1.2090        13.22
ExhaustiveTQ-b4-rf10 (query)                             272.59       806.24     1_078.83       0.2884          1.1372            1.1543        13.22
ExhaustiveTQ-b4-rf20 (query)                             272.59     1_205.63     1_478.22       0.3823          1.0940            1.0970        13.22
ExhaustiveTQ-b4 (self)                                   272.59     3_951.34     4_223.93       0.3841          1.0938            1.0948        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                          397.90       105.15       503.05       0.0971          1.7176            1.5958         7.80
IVF-TQ-b2-nl158-np12-rf0 (query)                         397.90       116.65       514.55       0.0971          1.7176            1.5958         7.80
IVF-TQ-b2-nl158-np17-rf0 (query)                         397.90       126.85       524.75       0.0971          1.7176            1.5958         7.80
IVF-TQ-b2-nl158-np7-rf10 (query)                         397.90       309.56       707.46       0.2853          1.1453            1.1620         7.80
IVF-TQ-b2-nl158-np7-rf20 (query)                         397.90       623.42     1_021.32       0.3809          1.0970            1.0941         7.80
IVF-TQ-b2-nl158-np12-rf10 (query)                        397.90       313.47       711.37       0.2853          1.1453            1.1620         7.80
IVF-TQ-b2-nl158-np12-rf20 (query)                        397.90       655.47     1_053.37       0.3809          1.0970            1.0941         7.80
IVF-TQ-b2-nl158-np17-rf10 (query)                        397.90       321.32       719.22       0.2853          1.1453            1.1620         7.80
IVF-TQ-b2-nl158-np17-rf20 (query)                        397.90       668.89     1_066.79       0.3809          1.0970            1.0941         7.80
IVF-TQ-b2-nl158 (self)                                   397.90     1_079.03     1_476.93       0.3815          1.0980            1.0957         7.80
IVF-TQ-b2-nl223-np11-rf0 (query)                         384.36       112.41       496.77       0.0971          1.7165            1.5941         7.93
IVF-TQ-b2-nl223-np14-rf0 (query)                         384.36       122.33       506.69       0.0971          1.7176            1.5958         7.93
IVF-TQ-b2-nl223-np21-rf0 (query)                         384.36       133.01       517.36       0.0971          1.7176            1.5958         7.93
IVF-TQ-b2-nl223-np11-rf10 (query)                        384.36       304.01       688.37       0.2855          1.1450            1.1618         7.93
IVF-TQ-b2-nl223-np11-rf20 (query)                        384.36       580.27       964.62       0.3813          1.0967            1.0934         7.93
IVF-TQ-b2-nl223-np14-rf10 (query)                        384.36       297.85       682.21       0.2853          1.1453            1.1620         7.93
IVF-TQ-b2-nl223-np14-rf20 (query)                        384.36       592.87       977.23       0.3809          1.0970            1.0941         7.93
IVF-TQ-b2-nl223-np21-rf10 (query)                        384.36       314.48       698.83       0.2853          1.1453            1.1620         7.93
IVF-TQ-b2-nl223-np21-rf20 (query)                        384.36       615.95     1_000.31       0.3809          1.0970            1.0941         7.93
IVF-TQ-b2-nl223 (self)                                   384.36     1_078.17     1_462.52       0.3815          1.0980            1.0957         7.93
IVF-TQ-b2-nl316-np15-rf0 (query)                         475.71       118.73       594.44       0.0973          1.7112            1.5945         8.10
IVF-TQ-b2-nl316-np17-rf0 (query)                         475.71       120.58       596.28       0.0971          1.7175            1.5958         8.10
IVF-TQ-b2-nl316-np25-rf0 (query)                         475.71       137.76       613.47       0.0971          1.7176            1.5958         8.10
IVF-TQ-b2-nl316-np15-rf10 (query)                        475.71       292.78       768.48       0.2855          1.1451            1.1619         8.10
IVF-TQ-b2-nl316-np15-rf20 (query)                        475.71       558.98     1_034.68       0.3812          1.0968            1.0936         8.10
IVF-TQ-b2-nl316-np17-rf10 (query)                        475.71       296.76       772.47       0.2853          1.1453            1.1620         8.10
IVF-TQ-b2-nl316-np17-rf20 (query)                        475.71       561.21     1_036.92       0.3809          1.0970            1.0941         8.10
IVF-TQ-b2-nl316-np25-rf10 (query)                        475.71       309.53       785.23       0.2853          1.1453            1.1620         8.10
IVF-TQ-b2-nl316-np25-rf20 (query)                        475.71       592.84     1_068.55       0.3809          1.0970            1.0941         8.10
IVF-TQ-b2-nl316 (self)                                   475.71     1_038.36     1_514.07       0.3815          1.0980            1.0957         8.10
IVF-TQ-b4-nl158-np7-rf0 (query)                          456.13       143.44       599.57       0.1094          1.5328            1.4997        14.05
IVF-TQ-b4-nl158-np12-rf0 (query)                         456.13       169.84       625.96       0.1094          1.5328            1.4997        14.05
IVF-TQ-b4-nl158-np17-rf0 (query)                         456.13       176.96       633.08       0.1094          1.5328            1.4997        14.05
IVF-TQ-b4-nl158-np7-rf10 (query)                         456.13       361.74       817.87       0.2884          1.1372            1.1543        14.05
IVF-TQ-b4-nl158-np7-rf20 (query)                         456.13       681.80     1_137.93       0.3823          1.0940            1.0970        14.05
IVF-TQ-b4-nl158-np12-rf10 (query)                        456.13       368.30       824.43       0.2884          1.1372            1.1543        14.05
IVF-TQ-b4-nl158-np12-rf20 (query)                        456.13       716.03     1_172.15       0.3823          1.0940            1.0970        14.05
IVF-TQ-b4-nl158-np17-rf10 (query)                        456.13       396.87       852.99       0.2884          1.1372            1.1543        14.05
IVF-TQ-b4-nl158-np17-rf20 (query)                        456.13       743.05     1_199.17       0.3823          1.0940            1.0970        14.05
IVF-TQ-b4-nl158 (self)                                   456.13     1_121.81     1_577.93       0.3841          1.0938            1.0948        14.05
IVF-TQ-b4-nl223-np11-rf0 (query)                         441.82       154.35       596.17       0.1094          1.5315            1.4987        14.24
IVF-TQ-b4-nl223-np14-rf0 (query)                         441.82       183.43       625.24       0.1094          1.5328            1.4996        14.24
IVF-TQ-b4-nl223-np21-rf0 (query)                         441.82       201.11       642.92       0.1094          1.5328            1.4996        14.24
IVF-TQ-b4-nl223-np11-rf10 (query)                        441.82       345.21       787.03       0.2886          1.1370            1.1542        14.24
IVF-TQ-b4-nl223-np11-rf20 (query)                        441.82       631.82     1_073.64       0.3826          1.0939            1.0966        14.24
IVF-TQ-b4-nl223-np14-rf10 (query)                        441.82       354.79       796.61       0.2884          1.1372            1.1543        14.24
IVF-TQ-b4-nl223-np14-rf20 (query)                        441.82       657.37     1_099.19       0.3823          1.0940            1.0970        14.24
IVF-TQ-b4-nl223-np21-rf10 (query)                        441.82       392.86       834.67       0.2884          1.1372            1.1543        14.24
IVF-TQ-b4-nl223-np21-rf20 (query)                        441.82       677.08     1_118.90       0.3823          1.0940            1.0970        14.24
IVF-TQ-b4-nl223 (self)                                   441.82     1_115.70     1_557.51       0.3841          1.0938            1.0948        14.24
IVF-TQ-b4-nl316-np15-rf0 (query)                         526.20       159.72       685.92       0.1094          1.5320            1.4991        14.49
IVF-TQ-b4-nl316-np17-rf0 (query)                         526.20       167.85       694.05       0.1094          1.5328            1.4996        14.49
IVF-TQ-b4-nl316-np25-rf0 (query)                         526.20       189.54       715.74       0.1094          1.5328            1.4997        14.49
IVF-TQ-b4-nl316-np15-rf10 (query)                        526.20       344.39       870.60       0.2885          1.1371            1.1542        14.49
IVF-TQ-b4-nl316-np15-rf20 (query)                        526.20       650.92     1_177.12       0.3826          1.0939            1.0969        14.49
IVF-TQ-b4-nl316-np17-rf10 (query)                        526.20       366.46       892.66       0.2884          1.1372            1.1543        14.49
IVF-TQ-b4-nl316-np17-rf20 (query)                        526.20       629.66     1_155.86       0.3823          1.0940            1.0970        14.49
IVF-TQ-b4-nl316-np25-rf10 (query)                        526.20       375.95       902.15       0.2884          1.1372            1.1543        14.49
IVF-TQ-b4-nl316-np25-rf20 (query)                        526.20       655.87     1_182.07       0.3823          1.0940            1.0970        14.49
IVF-TQ-b4-nl316 (self)                                   526.20     1_107.68     1_633.88       0.3841          1.0938            1.0948        14.49
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
Exhaustive (query)                                        73.75     1_383.44     1_457.18       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         73.75     4_633.33     4_707.08       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              352.68       667.14     1_019.82       0.1207          1.3711            1.3320        13.97
ExhaustiveTQ-b2-rf5 (query)                              352.68       760.33     1_113.01       0.2421          1.1334            1.1574        13.97
ExhaustiveTQ-b2-rf10 (query)                             352.68       898.31     1_250.99       0.2934          1.0981            1.1177        13.97
ExhaustiveTQ-b2-rf20 (query)                             352.68     1_309.02     1_661.70       0.3880          1.0664            1.0469        13.97
ExhaustiveTQ-b2 (self)                                   352.68     4_232.97     4_585.65       0.3879          1.0667            1.0471        13.97
ExhaustiveTQ-b4-rf0 (query)                              473.22     1_169.33     1_642.55       0.1315          1.3172            1.3127        26.18
ExhaustiveTQ-b4-rf5 (query)                              473.22     1_259.57     1_732.79       0.2471          1.1254            1.1483        26.18
ExhaustiveTQ-b4-rf10 (query)                             473.22     1_405.72     1_878.94       0.2970          1.0929            1.0980        26.18
ExhaustiveTQ-b4-rf20 (query)                             473.22     1_795.36     2_268.58       0.3883          1.0643            1.0492        26.18
ExhaustiveTQ-b4 (self)                                   473.22     5_892.81     6_366.03       0.3881          1.0646            1.0495        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                          631.34       193.43       824.78       0.1207          1.3711            1.3320        14.96
IVF-TQ-b2-nl158-np12-rf0 (query)                         631.34       216.35       847.70       0.1207          1.3711            1.3320        14.96
IVF-TQ-b2-nl158-np17-rf0 (query)                         631.34       225.13       856.47       0.1207          1.3711            1.3320        14.96
IVF-TQ-b2-nl158-np7-rf10 (query)                         631.34       430.66     1_062.01       0.2934          1.0981            1.1177        14.96
IVF-TQ-b2-nl158-np7-rf20 (query)                         631.34       786.42     1_417.76       0.3880          1.0664            1.0469        14.96
IVF-TQ-b2-nl158-np12-rf10 (query)                        631.34       432.88     1_064.22       0.2934          1.0981            1.1177        14.96
IVF-TQ-b2-nl158-np12-rf20 (query)                        631.34       794.66     1_426.01       0.3880          1.0664            1.0469        14.96
IVF-TQ-b2-nl158-np17-rf10 (query)                        631.34       454.71     1_086.06       0.2934          1.0981            1.1177        14.96
IVF-TQ-b2-nl158-np17-rf20 (query)                        631.34       823.83     1_455.17       0.3880          1.0664            1.0469        14.96
IVF-TQ-b2-nl158 (self)                                   631.34     1_435.10     2_066.44       0.3879          1.0667            1.0471        14.96
IVF-TQ-b2-nl223-np11-rf0 (query)                         710.54       206.33       916.88       0.1208          1.3696            1.3298        15.18
IVF-TQ-b2-nl223-np14-rf0 (query)                         710.54       220.26       930.81       0.1207          1.3711            1.3320        15.18
IVF-TQ-b2-nl223-np21-rf0 (query)                         710.54       247.02       957.56       0.1207          1.3711            1.3320        15.18
IVF-TQ-b2-nl223-np11-rf10 (query)                        710.54       410.73     1_121.27       0.2937          1.0979            1.1176        15.18
IVF-TQ-b2-nl223-np11-rf20 (query)                        710.54       726.57     1_437.11       0.3887          1.0662            1.0467        15.18
IVF-TQ-b2-nl223-np14-rf10 (query)                        710.54       420.19     1_130.74       0.2934          1.0981            1.1178        15.18
IVF-TQ-b2-nl223-np14-rf20 (query)                        710.54       732.27     1_442.81       0.3880          1.0664            1.0469        15.18
IVF-TQ-b2-nl223-np21-rf10 (query)                        710.54       443.28     1_153.82       0.2934          1.0981            1.1177        15.18
IVF-TQ-b2-nl223-np21-rf20 (query)                        710.54       784.51     1_495.05       0.3880          1.0664            1.0469        15.18
IVF-TQ-b2-nl223 (self)                                   710.54     1_481.63     2_192.17       0.3879          1.0667            1.0471        15.18
IVF-TQ-b2-nl316-np15-rf0 (query)                         729.34       221.42       950.76       0.1208          1.3690            1.3288        15.56
IVF-TQ-b2-nl316-np17-rf0 (query)                         729.34       228.93       958.27       0.1208          1.3707            1.3313        15.56
IVF-TQ-b2-nl316-np25-rf0 (query)                         729.34       244.65       973.99       0.1207          1.3711            1.3320        15.56
IVF-TQ-b2-nl316-np15-rf10 (query)                        729.34       415.31     1_144.65       0.2939          1.0977            1.1176        15.56
IVF-TQ-b2-nl316-np15-rf20 (query)                        729.34       690.46     1_419.80       0.3891          1.0660            1.0466        15.56
IVF-TQ-b2-nl316-np17-rf10 (query)                        729.34       414.47     1_143.81       0.2935          1.0980            1.1177        15.56
IVF-TQ-b2-nl316-np17-rf20 (query)                        729.34       708.87     1_438.21       0.3882          1.0664            1.0469        15.56
IVF-TQ-b2-nl316-np25-rf10 (query)                        729.34       441.05     1_170.39       0.2934          1.0981            1.1178        15.56
IVF-TQ-b2-nl316-np25-rf20 (query)                        729.34       761.43     1_490.77       0.3880          1.0664            1.0469        15.56
IVF-TQ-b2-nl316 (self)                                   729.34     1_456.32     2_185.66       0.3879          1.0667            1.0471        15.56
IVF-TQ-b4-nl158-np7-rf0 (query)                          743.96       271.08     1_015.04       0.1315          1.3172            1.3127        27.46
IVF-TQ-b4-nl158-np12-rf0 (query)                         743.96       309.48     1_053.44       0.1315          1.3172            1.3127        27.46
IVF-TQ-b4-nl158-np17-rf0 (query)                         743.96       330.78     1_074.74       0.1315          1.3172            1.3127        27.46
IVF-TQ-b4-nl158-np7-rf10 (query)                         743.96       512.44     1_256.40       0.2970          1.0929            1.0979        27.46
IVF-TQ-b4-nl158-np7-rf20 (query)                         743.96       871.86     1_615.82       0.3883          1.0643            1.0492        27.46
IVF-TQ-b4-nl158-np12-rf10 (query)                        743.96       542.35     1_286.31       0.2970          1.0929            1.0980        27.46
IVF-TQ-b4-nl158-np12-rf20 (query)                        743.96       911.76     1_655.72       0.3883          1.0643            1.0492        27.46
IVF-TQ-b4-nl158-np17-rf10 (query)                        743.96       582.88     1_326.84       0.2970          1.0929            1.0980        27.46
IVF-TQ-b4-nl158-np17-rf20 (query)                        743.96       938.13     1_682.09       0.3883          1.0643            1.0492        27.46
IVF-TQ-b4-nl158 (self)                                   743.96     1_649.10     2_393.06       0.3881          1.0646            1.0495        27.46
IVF-TQ-b4-nl223-np11-rf0 (query)                         791.43       291.41     1_082.84       0.1316          1.3158            1.3117        27.77
IVF-TQ-b4-nl223-np14-rf0 (query)                         791.43       317.42     1_108.86       0.1315          1.3172            1.3127        27.77
IVF-TQ-b4-nl223-np21-rf0 (query)                         791.43       350.27     1_141.70       0.1315          1.3172            1.3127        27.77
IVF-TQ-b4-nl223-np11-rf10 (query)                        791.43       511.13     1_302.56       0.2974          1.0926            1.0973        27.77
IVF-TQ-b4-nl223-np11-rf20 (query)                        791.43       817.62     1_609.05       0.3889          1.0641            1.0488        27.77
IVF-TQ-b4-nl223-np14-rf10 (query)                        791.43       522.31     1_313.75       0.2970          1.0929            1.0980        27.77
IVF-TQ-b4-nl223-np14-rf20 (query)                        791.43       844.62     1_636.05       0.3883          1.0643            1.0492        27.77
IVF-TQ-b4-nl223-np21-rf10 (query)                        791.43       560.40     1_351.84       0.2970          1.0929            1.0980        27.77
IVF-TQ-b4-nl223-np21-rf20 (query)                        791.43       903.28     1_694.71       0.3882          1.0643            1.0492        27.77
IVF-TQ-b4-nl223 (self)                                   791.43     1_670.51     2_461.94       0.3881          1.0646            1.0495        27.77
IVF-TQ-b4-nl316-np15-rf0 (query)                         886.95       319.03     1_205.97       0.1316          1.3151            1.3108        28.36
IVF-TQ-b4-nl316-np17-rf0 (query)                         886.95       326.20     1_213.15       0.1315          1.3164            1.3121        28.36
IVF-TQ-b4-nl316-np25-rf0 (query)                         886.95       360.78     1_247.73       0.1315          1.3172            1.3127        28.36
IVF-TQ-b4-nl316-np15-rf10 (query)                        886.95       531.66     1_418.61       0.2976          1.0925            1.0968        28.36
IVF-TQ-b4-nl316-np15-rf20 (query)                        886.95       803.47     1_690.42       0.3893          1.0639            1.0484        28.36
IVF-TQ-b4-nl316-np17-rf10 (query)                        886.95       527.12     1_414.07       0.2971          1.0928            1.0978        28.36
IVF-TQ-b4-nl316-np17-rf20 (query)                        886.95       854.87     1_741.82       0.3885          1.0642            1.0491        28.36
IVF-TQ-b4-nl316-np25-rf10 (query)                        886.95       569.39     1_456.33       0.2970          1.0929            1.0980        28.36
IVF-TQ-b4-nl316-np25-rf20 (query)                        886.95       904.64     1_791.58       0.3883          1.0643            1.0492        28.36
IVF-TQ-b4-nl316 (self)                                   886.95     1_698.98     2_585.93       0.3881          1.0646            1.0495        28.36
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
Exhaustive (query)                                       101.47     1_983.56     2_085.03       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.47     6_650.15     6_751.62       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              620.92       985.36     1_606.28       0.1292          1.2710            1.2627        21.33
ExhaustiveTQ-b2-rf5 (query)                              620.92     1_080.64     1_701.56       0.2468          1.1062            1.1332        21.33
ExhaustiveTQ-b2-rf10 (query)                             620.92     1_240.05     1_860.98       0.3000          1.0773            1.0631        21.33
ExhaustiveTQ-b2-rf20 (query)                             620.92     1_639.22     2_260.15       0.3957          1.0509            1.0334        21.33
ExhaustiveTQ-b2 (self)                                   620.92     5_392.86     6_013.78       0.3973          1.0507            1.0331        21.33
ExhaustiveTQ-b4-rf0 (query)                              800.37     1_789.01     2_589.38       0.1340          1.2532            1.2592        39.64
ExhaustiveTQ-b4-rf5 (query)                              800.37     1_916.73     2_717.09       0.2401          1.1136            1.1402        39.64
ExhaustiveTQ-b4-rf10 (query)                             800.37     2_079.99     2_880.36       0.2870          1.0888            1.1143        39.64
ExhaustiveTQ-b4-rf20 (query)                             800.37     2_454.83     3_255.20       0.3752          1.0657            1.0812        39.64
ExhaustiveTQ-b4 (self)                                   800.37     8_170.33     8_970.70       0.3767          1.0654            1.0638        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_138.02       301.55     1_439.57       0.1292          1.2710            1.2627        22.63
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_138.02       316.59     1_454.62       0.1292          1.2710            1.2627        22.63
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_138.02       334.41     1_472.44       0.1292          1.2710            1.2627        22.63
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_138.02       538.02     1_676.05       0.3000          1.0774            1.0631        22.63
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_138.02       933.77     2_071.80       0.3957          1.0509            1.0334        22.63
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_138.02       561.81     1_699.84       0.3000          1.0774            1.0631        22.63
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_138.02       950.53     2_088.55       0.3957          1.0509            1.0334        22.63
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_138.02       585.75     1_723.78       0.3000          1.0773            1.0631        22.63
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_138.02       985.99     2_124.01       0.3957          1.0509            1.0334        22.63
IVF-TQ-b2-nl158 (self)                                 1_138.02     1_846.45     2_984.47       0.3973          1.0507            1.0331        22.63
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_053.08       313.74     1_366.81       0.1292          1.2710            1.2627        23.04
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_053.08       325.87     1_378.94       0.1292          1.2710            1.2627        23.04
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_053.08       353.88     1_406.96       0.1292          1.2710            1.2628        23.04
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_053.08       554.50     1_607.58       0.3000          1.0774            1.0632        23.04
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_053.08       871.49     1_924.57       0.3957          1.0509            1.0334        23.04
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_053.08       558.54     1_611.61       0.3000          1.0774            1.0632        23.04
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_053.08       886.29     1_939.37       0.3957          1.0509            1.0334        23.04
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_053.08       597.82     1_650.90       0.3000          1.0774            1.0631        23.04
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_053.08       946.33     1_999.41       0.3957          1.0509            1.0334        23.04
IVF-TQ-b2-nl223 (self)                                 1_053.08     1_856.03     2_909.11       0.3973          1.0507            1.0331        23.04
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_104.05       328.88     1_432.93       0.1292          1.2709            1.2627        23.59
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_104.05       332.96     1_437.01       0.1292          1.2710            1.2627        23.59
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_104.05       363.01     1_467.06       0.1292          1.2710            1.2627        23.59
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_104.05       560.11     1_664.16       0.3000          1.0773            1.0631        23.59
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_104.05       863.72     1_967.76       0.3957          1.0509            1.0334        23.59
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_104.05       558.12     1_662.17       0.3000          1.0773            1.0631        23.59
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_104.05       879.55     1_983.60       0.3957          1.0509            1.0334        23.59
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_104.05       606.05     1_710.09       0.3000          1.0774            1.0631        23.59
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_104.05       925.04     2_029.09       0.3956          1.0509            1.0334        23.59
IVF-TQ-b2-nl316 (self)                                 1_104.05     1_906.71     3_010.76       0.3973          1.0507            1.0331        23.59
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_155.45       414.16     1_569.61       0.1340          1.2532            1.2592        41.40
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_155.45       462.43     1_617.88       0.1340          1.2532            1.2592        41.40
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_155.45       495.08     1_650.53       0.1340          1.2532            1.2592        41.40
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_155.45       673.38     1_828.83       0.2870          1.0888            1.1143        41.40
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_155.45     1_068.31     2_223.76       0.3752          1.0657            1.0812        41.40
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_155.45       723.98     1_879.43       0.2870          1.0888            1.1143        41.40
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_155.45     1_118.97     2_274.42       0.3752          1.0657            1.0812        41.40
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_155.45       754.02     1_909.47       0.2870          1.0888            1.1143        41.40
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_155.45     1_161.25     2_316.70       0.3752          1.0657            1.0812        41.40
IVF-TQ-b4-nl158 (self)                                 1_155.45     2_112.32     3_267.77       0.3767          1.0654            1.0637        41.40
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_116.64       447.44     1_564.08       0.1340          1.2532            1.2592        42.04
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_116.64       475.43     1_592.07       0.1340          1.2532            1.2592        42.04
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_116.64       531.17     1_647.81       0.1340          1.2532            1.2592        42.04
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_116.64       684.90     1_801.54       0.2870          1.0888            1.1143        42.04
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_116.64     1_003.56     2_120.20       0.3752          1.0657            1.0812        42.04
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_116.64       715.55     1_832.19       0.2870          1.0888            1.1143        42.04
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_116.64     1_042.53     2_159.17       0.3752          1.0657            1.0812        42.04
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_116.64       773.28     1_889.92       0.2870          1.0888            1.1143        42.04
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_116.64     1_119.70     2_236.34       0.3752          1.0657            1.0812        42.04
IVF-TQ-b4-nl223 (self)                                 1_116.64     2_196.61     3_313.25       0.3766          1.0654            1.0638        42.04
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_245.54       470.84     1_716.38       0.1340          1.2531            1.2592        42.85
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_245.54       485.66     1_731.20       0.1340          1.2532            1.2592        42.85
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_245.54       543.25     1_788.78       0.1340          1.2532            1.2592        42.85
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_245.54       693.03     1_938.57       0.2870          1.0888            1.1143        42.85
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_245.54     1_005.93     2_251.46       0.3753          1.0657            1.0812        42.85
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_245.54       718.09     1_963.63       0.2870          1.0888            1.1143        42.85
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_245.54     1_033.58     2_279.12       0.3752          1.0657            1.0812        42.85
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_245.54       779.59     2_025.12       0.2870          1.0888            1.1143        42.85
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_245.54     1_130.06     2_375.60       0.3752          1.0657            1.0812        42.85
IVF-TQ-b4-nl316 (self)                                 1_245.54     2_269.98     3_515.52       0.3767          1.0654            1.0638        42.85
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
Exhaustive (query)                                        33.89       745.44       779.33       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.89     2_480.46     2_514.35       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              154.47       376.24       530.71       0.0756          2.3283            1.9295         7.12
ExhaustiveTQ-b2-rf5 (query)                              154.47       450.83       605.30       0.2072          1.3307            1.3578         7.12
ExhaustiveTQ-b2-rf10 (query)                             154.47       588.90       743.37       0.2886          1.2206            1.2322         7.12
ExhaustiveTQ-b2-rf20 (query)                             154.47       968.72     1_123.19       0.4151          1.1328            1.1147         7.12
ExhaustiveTQ-b2 (self)                                   154.47     3_201.06     3_355.53       0.4136          1.1619            1.1367         7.12
ExhaustiveTQ-b4-rf0 (query)                              270.54       609.12       879.66       0.1023          1.7129            1.7532        13.22
ExhaustiveTQ-b4-rf5 (query)                              270.54       690.61       961.15       0.2385          1.2770            1.3000        13.22
ExhaustiveTQ-b4-rf10 (query)                             270.54       820.10     1_090.64       0.3202          1.1874            1.1953        13.22
ExhaustiveTQ-b4-rf20 (query)                             270.54     1_212.93     1_483.47       0.4481          1.1142            1.1029        13.22
ExhaustiveTQ-b4 (self)                                   270.54     3_931.41     4_201.96       0.4463          1.1397            1.1286        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                          343.13       103.59       446.72       0.0756          2.3282            1.9295         7.82
IVF-TQ-b2-nl158-np12-rf0 (query)                         343.13       115.37       458.50       0.0756          2.3282            1.9295         7.82
IVF-TQ-b2-nl158-np17-rf0 (query)                         343.13       136.25       479.38       0.0756          2.3282            1.9295         7.82
IVF-TQ-b2-nl158-np7-rf10 (query)                         343.13       304.43       647.56       0.2887          1.2206            1.2322         7.82
IVF-TQ-b2-nl158-np7-rf20 (query)                         343.13       654.23       997.35       0.4151          1.1328            1.1147         7.82
IVF-TQ-b2-nl158-np12-rf10 (query)                        343.13       319.56       662.69       0.2886          1.2206            1.2322         7.82
IVF-TQ-b2-nl158-np12-rf20 (query)                        343.13       680.23     1_023.36       0.4151          1.1328            1.1147         7.82
IVF-TQ-b2-nl158-np17-rf10 (query)                        343.13       353.71       696.84       0.2886          1.2206            1.2322         7.82
IVF-TQ-b2-nl158-np17-rf20 (query)                        343.13       765.89     1_109.02       0.4151          1.1328            1.1147         7.82
IVF-TQ-b2-nl158 (self)                                   343.13     1_087.55     1_430.68       0.4136          1.1619            1.1367         7.82
IVF-TQ-b2-nl223-np11-rf0 (query)                         388.76       112.16       500.92       0.0756          2.3254            1.9253         7.93
IVF-TQ-b2-nl223-np14-rf0 (query)                         388.76       119.56       508.31       0.0756          2.3281            1.9295         7.93
IVF-TQ-b2-nl223-np21-rf0 (query)                         388.76       142.52       531.27       0.0756          2.3282            1.9295         7.93
IVF-TQ-b2-nl223-np11-rf10 (query)                        388.76       285.81       674.56       0.2891          1.2202            1.2316         7.93
IVF-TQ-b2-nl223-np11-rf20 (query)                        388.76       573.72       962.47       0.4159          1.1325            1.1141         7.93
IVF-TQ-b2-nl223-np14-rf10 (query)                        388.76       296.97       685.73       0.2886          1.2206            1.2322         7.93
IVF-TQ-b2-nl223-np14-rf20 (query)                        388.76       590.60       979.35       0.4151          1.1328            1.1147         7.93
IVF-TQ-b2-nl223-np21-rf10 (query)                        388.76       337.30       726.06       0.2886          1.2206            1.2322         7.93
IVF-TQ-b2-nl223-np21-rf20 (query)                        388.76       681.97     1_070.73       0.4151          1.1328            1.1147         7.93
IVF-TQ-b2-nl223 (self)                                   388.76     1_077.98     1_466.73       0.4136          1.1619            1.1367         7.93
IVF-TQ-b2-nl316-np15-rf0 (query)                         483.05       116.66       599.72       0.0757          2.3274            1.9285         8.11
IVF-TQ-b2-nl316-np17-rf0 (query)                         483.05       118.91       601.96       0.0756          2.3282            1.9294         8.11
IVF-TQ-b2-nl316-np25-rf0 (query)                         483.05       138.40       621.46       0.0756          2.3282            1.9295         8.11
IVF-TQ-b2-nl316-np15-rf10 (query)                        483.05       283.92       766.97       0.2892          1.2202            1.2316         8.11
IVF-TQ-b2-nl316-np15-rf20 (query)                        483.05       541.40     1_024.46       0.4159          1.1325            1.1141         8.11
IVF-TQ-b2-nl316-np17-rf10 (query)                        483.05       289.50       772.55       0.2887          1.2206            1.2322         8.11
IVF-TQ-b2-nl316-np17-rf20 (query)                        483.05       546.77     1_029.83       0.4151          1.1328            1.1147         8.11
IVF-TQ-b2-nl316-np25-rf10 (query)                        483.05       322.05       805.10       0.2886          1.2206            1.2322         8.11
IVF-TQ-b2-nl316-np25-rf20 (query)                        483.05       603.02     1_086.07       0.4151          1.1328            1.1147         8.11
IVF-TQ-b2-nl316 (self)                                   483.05     1_072.94     1_555.99       0.4136          1.1619            1.1367         8.11
IVF-TQ-b4-nl158-np7-rf0 (query)                          443.50       144.76       588.27       0.1023          1.7129            1.7532        14.09
IVF-TQ-b4-nl158-np12-rf0 (query)                         443.50       175.42       618.93       0.1023          1.7129            1.7532        14.09
IVF-TQ-b4-nl158-np17-rf0 (query)                         443.50       190.06       633.57       0.1023          1.7129            1.7532        14.09
IVF-TQ-b4-nl158-np7-rf10 (query)                         443.50       354.21       797.71       0.3202          1.1873            1.1953        14.09
IVF-TQ-b4-nl158-np7-rf20 (query)                         443.50       697.56     1_141.07       0.4481          1.1142            1.1029        14.09
IVF-TQ-b4-nl158-np12-rf10 (query)                        443.50       380.15       823.65       0.3202          1.1873            1.1953        14.09
IVF-TQ-b4-nl158-np12-rf20 (query)                        443.50       757.05     1_200.56       0.4481          1.1142            1.1029        14.09
IVF-TQ-b4-nl158-np17-rf10 (query)                        443.50       416.66       860.16       0.3202          1.1873            1.1953        14.09
IVF-TQ-b4-nl158-np17-rf20 (query)                        443.50       829.97     1_273.48       0.4481          1.1142            1.1029        14.09
IVF-TQ-b4-nl158 (self)                                   443.50     1_131.07     1_574.57       0.4463          1.1397            1.1286        14.09
IVF-TQ-b4-nl223-np11-rf0 (query)                         468.49       154.30       622.79       0.1024          1.7109            1.7518        14.23
IVF-TQ-b4-nl223-np14-rf0 (query)                         468.49       165.66       634.15       0.1023          1.7129            1.7532        14.23
IVF-TQ-b4-nl223-np21-rf0 (query)                         468.49       203.88       672.37       0.1023          1.7129            1.7532        14.23
IVF-TQ-b4-nl223-np11-rf10 (query)                        468.49       336.90       805.39       0.3206          1.1870            1.1948        14.23
IVF-TQ-b4-nl223-np11-rf20 (query)                        468.49       627.63     1_096.13       0.4489          1.1139            1.1025        14.23
IVF-TQ-b4-nl223-np14-rf10 (query)                        468.49       365.92       834.41       0.3202          1.1873            1.1953        14.23
IVF-TQ-b4-nl223-np14-rf20 (query)                        468.49       652.25     1_120.75       0.4481          1.1142            1.1029        14.23
IVF-TQ-b4-nl223-np21-rf10 (query)                        468.49       416.21       884.70       0.3201          1.1874            1.1954        14.23
IVF-TQ-b4-nl223-np21-rf20 (query)                        468.49       732.58     1_201.07       0.4481          1.1142            1.1029        14.23
IVF-TQ-b4-nl223 (self)                                   468.49     1_132.74     1_601.24       0.4463          1.1397            1.1286        14.23
IVF-TQ-b4-nl316-np15-rf0 (query)                         515.12       154.59       669.71       0.1024          1.7112            1.7520        14.51
IVF-TQ-b4-nl316-np17-rf0 (query)                         515.12       162.50       677.62       0.1023          1.7121            1.7528        14.51
IVF-TQ-b4-nl316-np25-rf0 (query)                         515.12       197.14       712.26       0.1023          1.7129            1.7532        14.51
IVF-TQ-b4-nl316-np15-rf10 (query)                        515.12       332.53       847.65       0.3207          1.1869            1.1949        14.51
IVF-TQ-b4-nl316-np15-rf20 (query)                        515.12       599.51     1_114.63       0.4491          1.1138            1.1024        14.51
IVF-TQ-b4-nl316-np17-rf10 (query)                        515.12       344.84       859.96       0.3202          1.1873            1.1953        14.51
IVF-TQ-b4-nl316-np17-rf20 (query)                        515.12       620.38     1_135.50       0.4482          1.1142            1.1029        14.51
IVF-TQ-b4-nl316-np25-rf10 (query)                        515.12       381.95       897.07       0.3201          1.1874            1.1953        14.51
IVF-TQ-b4-nl316-np25-rf20 (query)                        515.12       663.29     1_178.41       0.4481          1.1142            1.1029        14.51
IVF-TQ-b4-nl316 (self)                                   515.12     1_107.53     1_622.65       0.4463          1.1397            1.1286        14.51
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
Exhaustive (query)                                        75.53     1_357.50     1_433.03       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         75.53     4_558.68     4_634.21       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              354.97       657.31     1_012.28       0.0844          1.6539            1.5906        13.97
ExhaustiveTQ-b2-rf5 (query)                              354.97       752.11     1_107.08       0.2173          1.2230            1.2549        13.97
ExhaustiveTQ-b2-rf10 (query)                             354.97       884.58     1_239.55       0.2887          1.1550            1.1707        13.97
ExhaustiveTQ-b2-rf20 (query)                             354.97     1_292.82     1_647.79       0.4020          1.0974            1.0847        13.97
ExhaustiveTQ-b2 (self)                                   354.97     4_202.08     4_557.04       0.4025          1.1135            1.0971        13.97
ExhaustiveTQ-b4-rf0 (query)                              462.81     1_164.40     1_627.21       0.1044          1.5026            1.5346        26.18
ExhaustiveTQ-b4-rf5 (query)                              462.81     1_270.12     1_732.93       0.2294          1.2110            1.2410        26.18
ExhaustiveTQ-b4-rf10 (query)                             462.81     1_403.92     1_866.73       0.2943          1.1499            1.1675        26.18
ExhaustiveTQ-b4-rf20 (query)                             462.81     1_795.01     2_257.82       0.4029          1.0975            1.0929        26.18
ExhaustiveTQ-b4 (self)                                   462.81     5_949.60     6_412.41       0.4038          1.1130            1.1087        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                          668.68       188.20       856.88       0.0844          1.6539            1.5906        14.97
IVF-TQ-b2-nl158-np12-rf0 (query)                         668.68       206.02       874.70       0.0844          1.6539            1.5906        14.97
IVF-TQ-b2-nl158-np17-rf0 (query)                         668.68       220.66       889.34       0.0844          1.6539            1.5906        14.97
IVF-TQ-b2-nl158-np7-rf10 (query)                         668.68       415.26     1_083.94       0.2887          1.1550            1.1707        14.97
IVF-TQ-b2-nl158-np7-rf20 (query)                         668.68       789.39     1_458.07       0.4020          1.0974            1.0847        14.97
IVF-TQ-b2-nl158-np12-rf10 (query)                        668.68       424.35     1_093.03       0.2887          1.1550            1.1707        14.97
IVF-TQ-b2-nl158-np12-rf20 (query)                        668.68       792.56     1_461.24       0.4020          1.0974            1.0847        14.97
IVF-TQ-b2-nl158-np17-rf10 (query)                        668.68       465.69     1_134.37       0.2887          1.1550            1.1707        14.97
IVF-TQ-b2-nl158-np17-rf20 (query)                        668.68       857.40     1_526.08       0.4020          1.0974            1.0847        14.97
IVF-TQ-b2-nl158 (self)                                   668.68     1_431.63     2_100.31       0.4025          1.1135            1.0971        14.97
IVF-TQ-b2-nl223-np11-rf0 (query)                         645.80       200.26       846.06       0.0845          1.6537            1.5905        15.19
IVF-TQ-b2-nl223-np14-rf0 (query)                         645.80       215.46       861.25       0.0844          1.6539            1.5906        15.19
IVF-TQ-b2-nl223-np21-rf0 (query)                         645.80       241.74       887.54       0.0844          1.6539            1.5906        15.19
IVF-TQ-b2-nl223-np11-rf10 (query)                        645.80       428.32     1_074.11       0.2887          1.1550            1.1707        15.19
IVF-TQ-b2-nl223-np11-rf20 (query)                        645.80       726.41     1_372.20       0.4020          1.0974            1.0847        15.19
IVF-TQ-b2-nl223-np14-rf10 (query)                        645.80       432.40     1_078.20       0.2887          1.1550            1.1707        15.19
IVF-TQ-b2-nl223-np14-rf20 (query)                        645.80       745.65     1_391.45       0.4020          1.0974            1.0847        15.19
IVF-TQ-b2-nl223-np21-rf10 (query)                        645.80       459.50     1_105.30       0.2887          1.1550            1.1707        15.19
IVF-TQ-b2-nl223-np21-rf20 (query)                        645.80       800.50     1_446.30       0.4020          1.0974            1.0847        15.19
IVF-TQ-b2-nl223 (self)                                   645.80     1_458.21     2_104.01       0.4025          1.1135            1.0971        15.19
IVF-TQ-b2-nl316-np15-rf0 (query)                         748.59       210.72       959.31       0.0844          1.6539            1.5906        15.58
IVF-TQ-b2-nl316-np17-rf0 (query)                         748.59       217.14       965.73       0.0844          1.6539            1.5906        15.58
IVF-TQ-b2-nl316-np25-rf0 (query)                         748.59       242.17       990.76       0.0844          1.6539            1.5906        15.58
IVF-TQ-b2-nl316-np15-rf10 (query)                        748.59       410.31     1_158.90       0.2887          1.1550            1.1707        15.58
IVF-TQ-b2-nl316-np15-rf20 (query)                        748.59       716.44     1_465.03       0.4020          1.0974            1.0847        15.58
IVF-TQ-b2-nl316-np17-rf10 (query)                        748.59       429.75     1_178.35       0.2887          1.1550            1.1707        15.58
IVF-TQ-b2-nl316-np17-rf20 (query)                        748.59       732.13     1_480.72       0.4020          1.0974            1.0847        15.58
IVF-TQ-b2-nl316-np25-rf10 (query)                        748.59       447.86     1_196.45       0.2887          1.1550            1.1707        15.58
IVF-TQ-b2-nl316-np25-rf20 (query)                        748.59       778.70     1_527.29       0.4020          1.0974            1.0847        15.58
IVF-TQ-b2-nl316 (self)                                   748.59     1_439.60     2_188.19       0.4025          1.1135            1.0971        15.58
IVF-TQ-b4-nl158-np7-rf0 (query)                          789.07       259.89     1_048.95       0.1044          1.5026            1.5346        27.49
IVF-TQ-b4-nl158-np12-rf0 (query)                         789.07       310.63     1_099.70       0.1044          1.5026            1.5346        27.49
IVF-TQ-b4-nl158-np17-rf0 (query)                         789.07       327.50     1_116.56       0.1044          1.5026            1.5346        27.49
IVF-TQ-b4-nl158-np7-rf10 (query)                         789.07       510.17     1_299.24       0.2943          1.1499            1.1675        27.49
IVF-TQ-b4-nl158-np7-rf20 (query)                         789.07       885.69     1_674.76       0.4029          1.0975            1.0929        27.49
IVF-TQ-b4-nl158-np12-rf10 (query)                        789.07       522.99     1_312.06       0.2943          1.1499            1.1675        27.49
IVF-TQ-b4-nl158-np12-rf20 (query)                        789.07       900.35     1_689.42       0.4029          1.0975            1.0929        27.49
IVF-TQ-b4-nl158-np17-rf10 (query)                        789.07       559.91     1_348.98       0.2943          1.1499            1.1675        27.49
IVF-TQ-b4-nl158-np17-rf20 (query)                        789.07       945.60     1_734.67       0.4029          1.0975            1.0929        27.49
IVF-TQ-b4-nl158 (self)                                   789.07     1_634.26     2_423.33       0.4038          1.1130            1.1088        27.49
IVF-TQ-b4-nl223-np11-rf0 (query)                         934.04       286.95     1_220.99       0.1044          1.5026            1.5346        27.81
IVF-TQ-b4-nl223-np14-rf0 (query)                         934.04       324.70     1_258.74       0.1044          1.5026            1.5346        27.81
IVF-TQ-b4-nl223-np21-rf0 (query)                         934.04       361.77     1_295.81       0.1044          1.5026            1.5346        27.81
IVF-TQ-b4-nl223-np11-rf10 (query)                        934.04       526.09     1_460.13       0.2943          1.1499            1.1675        27.81
IVF-TQ-b4-nl223-np11-rf20 (query)                        934.04       815.58     1_749.62       0.4029          1.0975            1.0929        27.81
IVF-TQ-b4-nl223-np14-rf10 (query)                        934.04       533.71     1_467.75       0.2943          1.1499            1.1675        27.81
IVF-TQ-b4-nl223-np14-rf20 (query)                        934.04       856.69     1_790.73       0.4029          1.0975            1.0929        27.81
IVF-TQ-b4-nl223-np21-rf10 (query)                        934.04       584.28     1_518.32       0.2943          1.1499            1.1675        27.81
IVF-TQ-b4-nl223-np21-rf20 (query)                        934.04       927.02     1_861.06       0.4029          1.0975            1.0929        27.81
IVF-TQ-b4-nl223 (self)                                   934.04     1_656.87     2_590.91       0.4038          1.1130            1.1088        27.81
IVF-TQ-b4-nl316-np15-rf0 (query)                         911.13       299.96     1_211.09       0.1045          1.5026            1.5346        28.39
IVF-TQ-b4-nl316-np17-rf0 (query)                         911.13       313.99     1_225.12       0.1044          1.5026            1.5346        28.39
IVF-TQ-b4-nl316-np25-rf0 (query)                         911.13       365.26     1_276.39       0.1044          1.5026            1.5346        28.39
IVF-TQ-b4-nl316-np15-rf10 (query)                        911.13       510.19     1_421.32       0.2943          1.1499            1.1675        28.39
IVF-TQ-b4-nl316-np15-rf20 (query)                        911.13       803.28     1_714.41       0.4029          1.0975            1.0929        28.39
IVF-TQ-b4-nl316-np17-rf10 (query)                        911.13       523.19     1_434.32       0.2943          1.1499            1.1675        28.39
IVF-TQ-b4-nl316-np17-rf20 (query)                        911.13       821.88     1_733.01       0.4029          1.0975            1.0929        28.39
IVF-TQ-b4-nl316-np25-rf10 (query)                        911.13       568.75     1_479.88       0.2943          1.1499            1.1675        28.39
IVF-TQ-b4-nl316-np25-rf20 (query)                        911.13       886.48     1_797.61       0.4029          1.0975            1.0929        28.39
IVF-TQ-b4-nl316 (self)                                   911.13     1_693.32     2_604.45       0.4038          1.1130            1.1087        28.39
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
Exhaustive (query)                                       100.15     1_928.24     2_028.39       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.15     6_517.49     6_617.65       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              609.47       956.70     1_566.17       0.0841          1.5107            1.4226        21.33
ExhaustiveTQ-b2-rf5 (query)                              609.47     1_057.19     1_666.66       0.2144          1.1739            1.2056        21.33
ExhaustiveTQ-b2-rf10 (query)                             609.47     1_197.55     1_807.03       0.2770          1.1267            1.1512        21.33
ExhaustiveTQ-b2-rf20 (query)                             609.47     1_617.91     2_227.39       0.3770          1.0843            1.0724        21.33
ExhaustiveTQ-b2 (self)                                   609.47     5_337.32     5_946.79       0.3767          1.0935            1.0803        21.33
ExhaustiveTQ-b4-rf0 (query)                              758.95     1_796.82     2_555.77       0.0986          1.4231            1.4109        39.64
ExhaustiveTQ-b4-rf5 (query)                              758.95     1_874.25     2_633.20       0.2167          1.1746            1.2047        39.64
ExhaustiveTQ-b4-rf10 (query)                             758.95     2_012.06     2_771.02       0.2692          1.1311            1.1557        39.64
ExhaustiveTQ-b4-rf20 (query)                             758.95     2_442.90     3_201.85       0.3605          1.0923            1.1071        39.64
ExhaustiveTQ-b4 (self)                                   758.95     7_949.86     8_708.81       0.3609          1.1024            1.1182        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                          976.09       283.62     1_259.71       0.0841          1.5107            1.4226        22.65
IVF-TQ-b2-nl158-np12-rf0 (query)                         976.09       305.78     1_281.86       0.0841          1.5107            1.4226        22.65
IVF-TQ-b2-nl158-np17-rf0 (query)                         976.09       332.42     1_308.50       0.0841          1.5107            1.4226        22.65
IVF-TQ-b2-nl158-np7-rf10 (query)                         976.09       520.64     1_496.73       0.2771          1.1267            1.1512        22.65
IVF-TQ-b2-nl158-np7-rf20 (query)                         976.09       894.51     1_870.59       0.3770          1.0843            1.0724        22.65
IVF-TQ-b2-nl158-np12-rf10 (query)                        976.09       534.40     1_510.49       0.2771          1.1267            1.1512        22.65
IVF-TQ-b2-nl158-np12-rf20 (query)                        976.09       928.65     1_904.73       0.3770          1.0843            1.0724        22.65
IVF-TQ-b2-nl158-np17-rf10 (query)                        976.09       560.71     1_536.79       0.2771          1.1267            1.1512        22.65
IVF-TQ-b2-nl158-np17-rf20 (query)                        976.09       942.56     1_918.65       0.3770          1.0843            1.0724        22.65
IVF-TQ-b2-nl158 (self)                                   976.09     1_853.79     2_829.87       0.3767          1.0935            1.0803        22.65
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_012.42       305.71     1_318.13       0.0841          1.5107            1.4226        22.97
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_012.42       318.20     1_330.62       0.0841          1.5107            1.4226        22.97
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_012.42       346.90     1_359.32       0.0841          1.5107            1.4226        22.97
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_012.42       556.00     1_568.42       0.2771          1.1267            1.1512        22.97
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_012.42       868.06     1_880.49       0.3770          1.0843            1.0724        22.97
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_012.42       557.30     1_569.73       0.2771          1.1267            1.1512        22.97
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_012.42       901.06     1_913.49       0.3770          1.0843            1.0724        22.97
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_012.42       592.34     1_604.77       0.2771          1.1267            1.1512        22.97
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_012.42       966.27     1_978.69       0.3770          1.0843            1.0724        22.97
IVF-TQ-b2-nl223 (self)                                 1_012.42     1_893.61     2_906.03       0.3767          1.0935            1.0803        22.97
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_104.98       316.51     1_421.50       0.0841          1.5107            1.4226        23.53
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_104.98       324.79     1_429.78       0.0841          1.5107            1.4226        23.53
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_104.98       355.25     1_460.23       0.0841          1.5107            1.4226        23.53
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_104.98       545.14     1_650.12       0.2771          1.1267            1.1512        23.53
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_104.98       859.51     1_964.49       0.3770          1.0843            1.0724        23.53
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_104.98       554.72     1_659.71       0.2770          1.1267            1.1512        23.53
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_104.98       880.59     1_985.58       0.3770          1.0843            1.0724        23.53
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_104.98       590.00     1_694.98       0.2771          1.1267            1.1512        23.53
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_104.98       936.59     2_041.57       0.3770          1.0843            1.0724        23.53
IVF-TQ-b2-nl316 (self)                                 1_104.98     1_951.09     3_056.08       0.3767          1.0935            1.0803        23.53
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_118.13       399.13     1_517.26       0.0986          1.4231            1.4109        41.44
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_118.13       441.66     1_559.79       0.0986          1.4231            1.4109        41.44
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_118.13       495.31     1_613.44       0.0986          1.4231            1.4109        41.44
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_118.13       648.72     1_766.85       0.2692          1.1311            1.1557        41.44
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_118.13     1_057.45     2_175.58       0.3605          1.0923            1.1071        41.44
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_118.13       707.51     1_825.64       0.2692          1.1311            1.1557        41.44
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_118.13     1_079.03     2_197.16       0.3605          1.0923            1.1071        41.44
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_118.13       748.71     1_866.84       0.2692          1.1311            1.1557        41.44
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_118.13     1_120.21     2_238.34       0.3605          1.0923            1.1071        41.44
IVF-TQ-b4-nl158 (self)                                 1_118.13     2_184.24     3_302.36       0.3609          1.1024            1.1182        41.44
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_129.79       441.52     1_571.31       0.0986          1.4231            1.4109        41.90
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_129.79       473.70     1_603.49       0.0986          1.4231            1.4109        41.90
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_129.79       519.91     1_649.70       0.0986          1.4231            1.4109        41.90
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_129.79       682.78     1_812.57       0.2692          1.1311            1.1557        41.90
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_129.79     1_024.07     2_153.86       0.3605          1.0923            1.1071        41.90
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_129.79       710.86     1_840.65       0.2692          1.1311            1.1557        41.90
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_129.79     1_055.41     2_185.19       0.3605          1.0923            1.1071        41.90
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_129.79       774.60     1_904.38       0.2692          1.1311            1.1557        41.90
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_129.79     1_140.15     2_269.94       0.3605          1.0923            1.1071        41.90
IVF-TQ-b4-nl223 (self)                                 1_129.79     2_250.10     3_379.89       0.3609          1.1024            1.1182        41.90
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_278.43       463.00     1_741.43       0.0986          1.4231            1.4109        42.74
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_278.43       482.91     1_761.33       0.0986          1.4231            1.4109        42.74
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_278.43       538.21     1_816.64       0.0986          1.4231            1.4109        42.74
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_278.43       694.59     1_973.02       0.2692          1.1311            1.1557        42.74
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_278.43     1_003.75     2_282.18       0.3605          1.0923            1.1071        42.74
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_278.43       709.72     1_988.15       0.2692          1.1311            1.1557        42.74
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_278.43     1_028.78     2_307.21       0.3605          1.0923            1.1071        42.74
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_278.43       783.72     2_062.15       0.2692          1.1311            1.1558        42.74
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_278.43     1_107.52     2_385.94       0.3605          1.0923            1.1071        42.74
IVF-TQ-b4-nl316 (self)                                 1_278.43     2_352.86     3_631.29       0.3609          1.1024            1.1182        42.74
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
Exhaustive (query)                                        33.22       753.80       787.02       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.22     2_543.28     2_576.51       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              148.73       379.09       527.83       0.7918          1.0898            1.0632         7.12
ExhaustiveTQ-b2-rf5 (query)                              148.73       456.65       605.38       0.9995          1.0000            1.0000         7.12
ExhaustiveTQ-b2-rf10 (query)                             148.73       594.44       743.17       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b2-rf20 (query)                             148.73     1_062.94     1_211.67       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b2 (self)                                   148.73     3_287.51     3_436.24       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b4-rf0 (query)                              234.29       620.09       854.38       0.8728          1.0322            1.0183        13.22
ExhaustiveTQ-b4-rf5 (query)                              234.29       712.47       946.76       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4-rf10 (query)                             234.29       829.44     1_063.74       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4-rf20 (query)                             234.29     1_199.76     1_434.06       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4 (self)                                   234.29     4_014.60     4_248.89       1.0000          1.0000            1.0000        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                          549.36       127.62       676.97       0.7917          1.0897            1.0634         7.78
IVF-TQ-b2-nl158-np12-rf0 (query)                         549.36       175.73       725.09       0.7918          1.0898            1.0632         7.78
IVF-TQ-b2-nl158-np17-rf0 (query)                         549.36       214.78       764.14       0.7918          1.0898            1.0632         7.78
IVF-TQ-b2-nl158-np7-rf10 (query)                         549.36       341.47       890.82       0.9982          1.0004            1.0000         7.78
IVF-TQ-b2-nl158-np7-rf20 (query)                         549.36       634.72     1_184.08       0.9982          1.0004            1.0000         7.78
IVF-TQ-b2-nl158-np12-rf10 (query)                        549.36       407.67       957.03       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158-np12-rf20 (query)                        549.36       738.36     1_287.72       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158-np17-rf10 (query)                        549.36       452.73     1_002.09       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158-np17-rf20 (query)                        549.36       839.13     1_388.49       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158 (self)                                   549.36     1_231.18     1_780.54       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl223-np11-rf0 (query)                         622.47       129.38       751.84       0.7919          1.0897            1.0632         7.92
IVF-TQ-b2-nl223-np14-rf0 (query)                         622.47       146.30       768.76       0.7919          1.0897            1.0632         7.92
IVF-TQ-b2-nl223-np21-rf0 (query)                         622.47       186.01       808.47       0.7918          1.0898            1.0632         7.92
IVF-TQ-b2-nl223-np11-rf10 (query)                        622.47       328.28       950.75       0.9995          1.0001            1.0000         7.92
IVF-TQ-b2-nl223-np11-rf20 (query)                        622.47       598.44     1_220.91       0.9995          1.0001            1.0000         7.92
IVF-TQ-b2-nl223-np14-rf10 (query)                        622.47       353.97       976.44       0.9999          1.0000            1.0000         7.92
IVF-TQ-b2-nl223-np14-rf20 (query)                        622.47       645.15     1_267.62       0.9999          1.0000            1.0000         7.92
IVF-TQ-b2-nl223-np21-rf10 (query)                        622.47       421.60     1_044.07       1.0000          1.0000            1.0000         7.92
IVF-TQ-b2-nl223-np21-rf20 (query)                        622.47       738.01     1_360.48       1.0000          1.0000            1.0000         7.92
IVF-TQ-b2-nl223 (self)                                   622.47     1_054.73     1_677.20       1.0000          1.0000            1.0000         7.92
IVF-TQ-b2-nl316-np15-rf0 (query)                         724.15       132.84       856.99       0.7918          1.0897            1.0632         8.11
IVF-TQ-b2-nl316-np17-rf0 (query)                         724.15       150.65       874.80       0.7918          1.0898            1.0632         8.11
IVF-TQ-b2-nl316-np25-rf0 (query)                         724.15       173.35       897.51       0.7918          1.0898            1.0632         8.11
IVF-TQ-b2-nl316-np15-rf10 (query)                        724.15       316.38     1_040.53       0.9998          1.0000            1.0000         8.11
IVF-TQ-b2-nl316-np15-rf20 (query)                        724.15       600.12     1_324.28       0.9998          1.0000            1.0000         8.11
IVF-TQ-b2-nl316-np17-rf10 (query)                        724.15       335.19     1_059.34       0.9999          1.0000            1.0000         8.11
IVF-TQ-b2-nl316-np17-rf20 (query)                        724.15       605.97     1_330.12       0.9999          1.0000            1.0000         8.11
IVF-TQ-b2-nl316-np25-rf10 (query)                        724.15       387.53     1_111.68       1.0000          1.0000            1.0000         8.11
IVF-TQ-b2-nl316-np25-rf20 (query)                        724.15       682.73     1_406.89       1.0000          1.0000            1.0000         8.11
IVF-TQ-b2-nl316 (self)                                   724.15       984.09     1_708.24       1.0000          1.0000            1.0000         8.11
IVF-TQ-b4-nl158-np7-rf0 (query)                          588.68       181.04       769.72       0.8721          1.0325            1.0187        14.01
IVF-TQ-b4-nl158-np12-rf0 (query)                         588.68       255.66       844.35       0.8728          1.0322            1.0183        14.01
IVF-TQ-b4-nl158-np17-rf0 (query)                         588.68       322.58       911.26       0.8728          1.0322            1.0183        14.01
IVF-TQ-b4-nl158-np7-rf10 (query)                         588.68       399.41       988.09       0.9982          1.0004            1.0000        14.01
IVF-TQ-b4-nl158-np7-rf20 (query)                         588.68       689.58     1_278.27       0.9982          1.0004            1.0000        14.01
IVF-TQ-b4-nl158-np12-rf10 (query)                        588.68       495.98     1_084.66       1.0000          1.0000            1.0000        14.01
IVF-TQ-b4-nl158-np12-rf20 (query)                        588.68       818.98     1_407.66       1.0000          1.0000            1.0000        14.01
IVF-TQ-b4-nl158-np17-rf10 (query)                        588.68       557.70     1_146.38       1.0000          1.0000            1.0000        14.01
IVF-TQ-b4-nl158-np17-rf20 (query)                        588.68       917.28     1_505.97       1.0000          1.0000            1.0000        14.01
IVF-TQ-b4-nl158 (self)                                   588.68     1_265.71     1_854.40       1.0000          1.0000            1.0000        14.01
IVF-TQ-b4-nl223-np11-rf0 (query)                         678.86       183.79       862.65       0.8726          1.0323            1.0184        14.23
IVF-TQ-b4-nl223-np14-rf0 (query)                         678.86       207.89       886.75       0.8727          1.0322            1.0183        14.23
IVF-TQ-b4-nl223-np21-rf0 (query)                         678.86       270.60       949.47       0.8728          1.0322            1.0183        14.23
IVF-TQ-b4-nl223-np11-rf10 (query)                        678.86       385.01     1_063.87       0.9995          1.0001            1.0000        14.23
IVF-TQ-b4-nl223-np11-rf20 (query)                        678.86       665.81     1_344.67       0.9995          1.0001            1.0000        14.23
IVF-TQ-b4-nl223-np14-rf10 (query)                        678.86       419.89     1_098.75       0.9999          1.0000            1.0000        14.23
IVF-TQ-b4-nl223-np14-rf20 (query)                        678.86       750.92     1_429.78       0.9999          1.0000            1.0000        14.23
IVF-TQ-b4-nl223-np21-rf10 (query)                        678.86       499.84     1_178.70       1.0000          1.0000            1.0000        14.23
IVF-TQ-b4-nl223-np21-rf20 (query)                        678.86       833.28     1_512.14       1.0000          1.0000            1.0000        14.23
IVF-TQ-b4-nl223 (self)                                   678.86     1_118.67     1_797.54       1.0000          1.0000            1.0000        14.23
IVF-TQ-b4-nl316-np15-rf0 (query)                         831.46       182.85     1_014.31       0.8727          1.0322            1.0184        14.52
IVF-TQ-b4-nl316-np17-rf0 (query)                         831.46       199.64     1_031.10       0.8727          1.0322            1.0183        14.52
IVF-TQ-b4-nl316-np25-rf0 (query)                         831.46       250.84     1_082.30       0.8727          1.0322            1.0183        14.52
IVF-TQ-b4-nl316-np15-rf10 (query)                        831.46       387.53     1_218.99       0.9998          1.0000            1.0000        14.52
IVF-TQ-b4-nl316-np15-rf20 (query)                        831.46       638.74     1_470.20       0.9998          1.0000            1.0000        14.52
IVF-TQ-b4-nl316-np17-rf10 (query)                        831.46       401.21     1_232.67       0.9999          1.0000            1.0000        14.52
IVF-TQ-b4-nl316-np17-rf20 (query)                        831.46       672.08     1_503.54       0.9999          1.0000            1.0000        14.52
IVF-TQ-b4-nl316-np25-rf10 (query)                        831.46       465.26     1_296.73       1.0000          1.0000            1.0000        14.52
IVF-TQ-b4-nl316-np25-rf20 (query)                        831.46       770.44     1_601.90       1.0000          1.0000            1.0000        14.52
IVF-TQ-b4-nl316 (self)                                   831.46     1_094.74     1_926.20       1.0000          1.0000            1.0000        14.52
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
Exhaustive (query)                                        69.84     1_391.37     1_461.21       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.84     4_592.18     4_662.02       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              354.91       659.68     1_014.59       0.8424          1.0447            1.0331        13.97
ExhaustiveTQ-b2-rf5 (query)                              354.91       749.32     1_104.23       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2-rf10 (query)                             354.91       891.51     1_246.42       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2-rf20 (query)                             354.91     1_310.40     1_665.31       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2 (self)                                   354.91     4_272.02     4_626.93       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b4-rf0 (query)                              466.93     1_160.59     1_627.53       0.8985          1.0191            1.0110        26.18
ExhaustiveTQ-b4-rf5 (query)                              466.93     1_262.93     1_729.86       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4-rf10 (query)                             466.93     1_399.63     1_866.56       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4-rf20 (query)                             466.93     1_789.68     2_256.61       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4 (self)                                   466.93     5_948.47     6_415.40       1.0000          1.0000            1.0000        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                          861.98       232.52     1_094.49       0.8421          1.0449            1.0333        14.97
IVF-TQ-b2-nl158-np12-rf0 (query)                         861.98       302.37     1_164.35       0.8424          1.0447            1.0331        14.97
IVF-TQ-b2-nl158-np17-rf0 (query)                         861.98       366.41     1_228.39       0.8424          1.0447            1.0331        14.97
IVF-TQ-b2-nl158-np7-rf10 (query)                         861.98       464.32     1_326.30       0.9987          1.0003            1.0000        14.97
IVF-TQ-b2-nl158-np7-rf20 (query)                         861.98       776.28     1_638.26       0.9987          1.0003            1.0000        14.97
IVF-TQ-b2-nl158-np12-rf10 (query)                        861.98       552.11     1_414.09       0.9999          1.0000            1.0000        14.97
IVF-TQ-b2-nl158-np12-rf20 (query)                        861.98       891.19     1_753.17       0.9999          1.0000            1.0000        14.97
IVF-TQ-b2-nl158-np17-rf10 (query)                        861.98       624.95     1_486.93       1.0000          1.0000            1.0000        14.97
IVF-TQ-b2-nl158-np17-rf20 (query)                        861.98       978.05     1_840.03       1.0000          1.0000            1.0000        14.97
IVF-TQ-b2-nl158 (self)                                   861.98     1_617.95     2_479.93       1.0000          1.0000            1.0000        14.97
IVF-TQ-b2-nl223-np11-rf0 (query)                         907.11       233.42     1_140.53       0.8423          1.0447            1.0331        15.26
IVF-TQ-b2-nl223-np14-rf0 (query)                         907.11       270.99     1_178.10       0.8424          1.0447            1.0330        15.26
IVF-TQ-b2-nl223-np21-rf0 (query)                         907.11       322.81     1_229.92       0.8424          1.0447            1.0331        15.26
IVF-TQ-b2-nl223-np11-rf10 (query)                        907.11       453.88     1_361.00       0.9997          1.0000            1.0000        15.26
IVF-TQ-b2-nl223-np11-rf20 (query)                        907.11       745.64     1_652.75       0.9997          1.0000            1.0000        15.26
IVF-TQ-b2-nl223-np14-rf10 (query)                        907.11       486.24     1_393.35       0.9999          1.0000            1.0000        15.26
IVF-TQ-b2-nl223-np14-rf20 (query)                        907.11       799.95     1_707.07       0.9999          1.0000            1.0000        15.26
IVF-TQ-b2-nl223-np21-rf10 (query)                        907.11       564.19     1_471.30       1.0000          1.0000            1.0000        15.26
IVF-TQ-b2-nl223-np21-rf20 (query)                        907.11       893.60     1_800.71       1.0000          1.0000            1.0000        15.26
IVF-TQ-b2-nl223 (self)                                   907.11     1_482.85     2_389.97       1.0000          1.0000            1.0000        15.26
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_001.59       239.79     1_241.38       0.8424          1.0447            1.0331        15.57
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_001.59       254.24     1_255.83       0.8424          1.0447            1.0331        15.57
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_001.59       312.66     1_314.25       0.8424          1.0447            1.0331        15.57
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_001.59       452.02     1_453.61       0.9999          1.0000            1.0000        15.57
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_001.59       750.10     1_751.69       0.9999          1.0000            1.0000        15.57
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_001.59       476.16     1_477.75       1.0000          1.0000            1.0000        15.57
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_001.59       786.10     1_787.69       1.0000          1.0000            1.0000        15.57
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_001.59       537.90     1_539.49       1.0000          1.0000            1.0000        15.57
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_001.59       858.39     1_859.98       1.0000          1.0000            1.0000        15.57
IVF-TQ-b2-nl316 (self)                                 1_001.59     1_406.49     2_408.08       1.0000          1.0000            1.0000        15.57
IVF-TQ-b4-nl158-np7-rf0 (query)                          975.43       341.51     1_316.95       0.8979          1.0193            1.0113        27.48
IVF-TQ-b4-nl158-np12-rf0 (query)                         975.43       468.37     1_443.80       0.8985          1.0191            1.0110        27.48
IVF-TQ-b4-nl158-np17-rf0 (query)                         975.43       576.65     1_552.09       0.8985          1.0191            1.0110        27.48
IVF-TQ-b4-nl158-np7-rf10 (query)                         975.43       572.43     1_547.87       0.9987          1.0003            1.0000        27.48
IVF-TQ-b4-nl158-np7-rf20 (query)                         975.43       879.68     1_855.11       0.9987          1.0003            1.0000        27.48
IVF-TQ-b4-nl158-np12-rf10 (query)                        975.43       727.49     1_702.93       0.9999          1.0000            1.0000        27.48
IVF-TQ-b4-nl158-np12-rf20 (query)                        975.43     1_055.53     2_030.96       0.9999          1.0000            1.0000        27.48
IVF-TQ-b4-nl158-np17-rf10 (query)                        975.43       831.11     1_806.55       1.0000          1.0000            1.0000        27.48
IVF-TQ-b4-nl158-np17-rf20 (query)                        975.43     1_190.14     2_165.57       1.0000          1.0000            1.0000        27.48
IVF-TQ-b4-nl158 (self)                                   975.43     1_865.94     2_841.38       1.0000          1.0000            1.0000        27.48
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_049.71       358.31     1_408.02       0.8984          1.0191            1.0111        27.93
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_049.71       409.25     1_458.96       0.8985          1.0191            1.0110        27.93
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_049.71       518.86     1_568.57       0.8985          1.0191            1.0110        27.93
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_049.71       563.45     1_613.16       0.9997          1.0000            1.0000        27.93
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_049.71       911.41     1_961.12       0.9997          1.0000            1.0000        27.93
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_049.71       649.51     1_699.22       0.9999          1.0000            1.0000        27.93
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_049.71       939.20     1_988.91       0.9999          1.0000            1.0000        27.93
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_049.71       764.12     1_813.82       1.0000          1.0000            1.0000        27.93
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_049.71     1_085.92     2_135.63       1.0000          1.0000            1.0000        27.93
IVF-TQ-b4-nl223 (self)                                 1_049.71     1_994.57     3_044.28       1.0000          1.0000            1.0000        27.93
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_289.38       371.88     1_661.27       0.8985          1.0191            1.0110        28.37
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_289.38       409.14     1_698.53       0.8985          1.0191            1.0110        28.37
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_289.38       493.52     1_782.91       0.8985          1.0191            1.0110        28.37
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_289.38       654.54     1_943.92       0.9999          1.0000            1.0000        28.37
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_289.38       909.49     2_198.87       0.9999          1.0000            1.0000        28.37
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_289.38       590.21     1_879.60       1.0000          1.0000            1.0000        28.37
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_289.38       914.94     2_204.33       1.0000          1.0000            1.0000        28.37
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_289.38       693.48     1_982.87       1.0000          1.0000            1.0000        28.37
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_289.38     1_023.16     2_312.55       1.0000          1.0000            1.0000        28.37
IVF-TQ-b4-nl316 (self)                                 1_289.38     1_679.86     2_969.24       1.0000          1.0000            1.0000        28.37
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
Exhaustive (query)                                        99.43     2_002.86     2_102.29       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.43     6_656.62     6_756.04       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              611.23       977.64     1_588.87       0.8736          1.0271            1.0199        21.33
ExhaustiveTQ-b2-rf5 (query)                              611.23     1_072.41     1_683.64       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2-rf10 (query)                             611.23     1_211.98     1_823.21       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2-rf20 (query)                             611.23     1_629.09     2_240.32       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2 (self)                                   611.23     5_504.91     6_116.14       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b4-rf0 (query)                              758.77     1_772.80     2_531.57       0.9097          1.0146            1.0083        39.64
ExhaustiveTQ-b4-rf5 (query)                              758.77     1_867.73     2_626.49       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4-rf10 (query)                             758.77     2_000.05     2_758.82       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4-rf20 (query)                             758.77     2_419.60     3_178.37       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4 (self)                                   758.77     7_918.72     8_677.49       1.0000          1.0000            1.0000        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_413.90       348.28     1_762.18       0.8735          1.0272            1.0201        22.61
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_413.90       448.22     1_862.11       0.8736          1.0271            1.0199        22.61
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_413.90       526.87     1_940.76       0.8736          1.0271            1.0199        22.61
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_413.90       605.67     2_019.56       0.9995          1.0001            1.0000        22.61
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_413.90       934.26     2_348.16       0.9995          1.0001            1.0000        22.61
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_413.90       716.01     2_129.90       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_413.90     1_099.92     2_513.82       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_413.90       805.09     2_218.98       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_413.90     1_184.50     2_598.39       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158 (self)                                 1_413.90     2_069.07     3_482.96       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_516.13       344.65     1_860.78       0.8736          1.0271            1.0200        23.00
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_516.13       384.05     1_900.18       0.8736          1.0271            1.0199        23.00
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_516.13       469.31     1_985.44       0.8736          1.0271            1.0199        23.00
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_516.13       602.45     2_118.58       0.9998          1.0000            1.0000        23.00
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_516.13       935.60     2_451.73       0.9998          1.0000            1.0000        23.00
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_516.13       636.51     2_152.64       1.0000          1.0000            1.0000        23.00
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_516.13       986.73     2_502.86       1.0000          1.0000            1.0000        23.00
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_516.13       732.49     2_248.62       1.0000          1.0000            1.0000        23.00
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_516.13     1_094.03     2_610.16       1.0000          1.0000            1.0000        23.00
IVF-TQ-b2-nl223 (self)                                 1_516.13     1_955.04     3_471.17       1.0000          1.0000            1.0000        23.00
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_779.46       356.35     2_135.80       0.8736          1.0271            1.0200        23.53
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_779.46       373.01     2_152.46       0.8736          1.0271            1.0200        23.53
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_779.46       446.81     2_226.26       0.8736          1.0271            1.0199        23.53
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_779.46       597.05     2_376.51       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_779.46       918.78     2_698.24       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_779.46       611.35     2_390.80       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_779.46       956.06     2_735.51       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_779.46       692.03     2_471.49       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_779.46     1_059.19     2_838.64       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316 (self)                                 1_779.46     1_961.92     3_741.37       1.0000          1.0000            1.0000        23.53
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_529.75       520.66     2_050.41       0.9094          1.0147            1.0084        41.37
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_529.75       736.34     2_266.09       0.9097          1.0146            1.0083        41.37
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_529.75       854.49     2_384.25       0.9097          1.0146            1.0083        41.37
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_529.75       794.75     2_324.51       0.9995          1.0001            1.0000        41.37
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_529.75     1_111.68     2_641.43       0.9995          1.0001            1.0000        41.37
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_529.75     1_002.13     2_531.88       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_529.75     1_344.36     2_874.11       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_529.75     1_135.24     2_664.99       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_529.75     1_510.38     3_040.13       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158 (self)                                 1_529.75     2_648.88     4_178.63       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_687.15       526.56     2_213.71       0.9096          1.0146            1.0084        41.94
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_687.15       604.32     2_291.47       0.9097          1.0146            1.0083        41.94
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_687.15       783.10     2_470.24       0.9097          1.0146            1.0083        41.94
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_687.15       762.70     2_449.85       0.9998          1.0000            1.0000        41.94
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_687.15     1_086.88     2_774.03       0.9998          1.0000            1.0000        41.94
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_687.15       854.14     2_541.29       1.0000          1.0000            1.0000        41.94
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_687.15     1_185.42     2_872.57       1.0000          1.0000            1.0000        41.94
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_687.15     1_015.11     2_702.26       1.0000          1.0000            1.0000        41.94
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_687.15     1_373.18     3_060.33       1.0000          1.0000            1.0000        41.94
IVF-TQ-b4-nl223 (self)                                 1_687.15     2_481.04     4_168.19       1.0000          1.0000            1.0000        41.94
IVF-TQ-b4-nl316-np15-rf0 (query)                       2_059.75       540.12     2_599.87       0.9097          1.0146            1.0083        42.73
IVF-TQ-b4-nl316-np17-rf0 (query)                       2_059.75       580.92     2_640.67       0.9097          1.0146            1.0083        42.73
IVF-TQ-b4-nl316-np25-rf0 (query)                       2_059.75       727.44     2_787.19       0.9097          1.0146            1.0083        42.73
IVF-TQ-b4-nl316-np15-rf10 (query)                      2_059.75       767.29     2_827.04       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np15-rf20 (query)                      2_059.75     1_088.91     3_148.66       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np17-rf10 (query)                      2_059.75       831.65     2_891.40       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np17-rf20 (query)                      2_059.75     1_149.88     3_209.63       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np25-rf10 (query)                      2_059.75       958.77     3_018.53       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np25-rf20 (query)                      2_059.75     1_319.97     3_379.72       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316 (self)                                 2_059.75     2_450.95     4_510.70       1.0000          1.0000            1.0000        42.73
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*ann-search-rs 0.10.1 (commit v0.10.1-18-g25372f1), run on 2026-10-09.*
*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
