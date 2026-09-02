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
  spent, since the retained count is capped at `n_bits / 16`.
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
Exhaustive (query)                                        33.10       686.33       719.43       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.10     2_266.54     2_299.64       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                 70.29       240.24       310.53       0.1199          1.4617            1.4199         1.78
ExhaustiveBinary-256-random-rf10 (query)                  70.29       344.50       414.80       0.3411          1.0941            1.0814         1.78
ExhaustiveBinary-256-random-rf20 (query)                  70.29       448.26       518.55       0.4467          1.0571            1.0475         1.78
ExhaustiveBinary-256-random (self)                        70.29     1_138.40     1_208.70       0.3454          1.0895            1.0798         1.78
ExhaustiveBinary-256-pca_no_rr (query)                   101.84       240.56       342.40       0.1153          1.4748            1.4212         1.78
ExhaustiveBinary-256-pca-rf10 (query)                    101.84       362.50       464.34       0.3323          1.1029            1.0834         1.78
ExhaustiveBinary-256-pca-rf20 (query)                    101.84       446.47       548.32       0.4387          1.0631            1.0485         1.78
ExhaustiveBinary-256-pca (self)                          101.84     1_103.59     1_205.43       0.3391          1.0957            1.0813         1.78
ExhaustiveBinary-512-random_no_rr (query)                 91.89       351.54       443.43       0.1588          1.3547            1.3300         3.55
ExhaustiveBinary-512-random-rf10 (query)                  91.89       465.38       557.28       0.3786          1.0692            1.0677         3.55
ExhaustiveBinary-512-random-rf20 (query)                  91.89       574.56       666.45       0.4874          1.0424            1.0395         3.55
ExhaustiveBinary-512-random (self)                        91.89     1_530.81     1_622.70       0.3805          1.0675            1.0675         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   121.92       348.11       470.03       0.1564          1.3535            1.3265         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    121.92       469.93       591.85       0.3789          1.0710            1.0663         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    121.92       582.45       704.37       0.4903          1.0433            1.0387         3.55
ExhaustiveBinary-512-pca (self)                          121.92     1_512.52     1_634.44       0.3823          1.0678            1.0665         3.55
ExhaustiveBinary-1024-random_no_rr (query)               124.00       504.21       628.20       0.1929          1.2764            1.2696         7.10
ExhaustiveBinary-1024-random-rf10 (query)                124.00       627.04       751.04       0.4214          1.0550            1.0552         7.10
ExhaustiveBinary-1024-random-rf20 (query)                124.00       747.66       871.65       0.5434          1.0327            1.0308         7.10
ExhaustiveBinary-1024-random (self)                      124.00     2_080.58     2_204.58       0.4232          1.0547            1.0552         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  149.66       510.86       660.52       0.1921          1.2733            1.2652         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   149.66       636.46       786.12       0.4226          1.0546            1.0544         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   149.66       743.41       893.07       0.5443          1.0326            1.0305         7.10
ExhaustiveBinary-1024-pca (self)                         149.66     2_081.26     2_230.93       0.4236          1.0546            1.0548         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   41.16       443.10       484.26       0.1211          1.4987            1.4523         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    41.16       474.63       515.79       0.3284          1.1039            1.0884         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    41.16       730.56       771.72       0.4385          1.0624            1.0494         1.53
ExhaustiveBinary-256-sign (self)                          41.16     1_561.14     1_602.30       0.3334          1.0988            1.0859         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)              832.41        50.83       883.24       0.1231          1.4432            1.4051         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)             832.41        53.24       885.65       0.1231          1.4432            1.4051         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)             832.41        56.42       888.83       0.1231          1.4432            1.4051         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)             832.41       102.58       934.99       0.3463          1.0912            1.0794         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)             832.41       151.08       983.49       0.4529          1.0552            1.0461         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)            832.41       101.81       934.22       0.3463          1.0912            1.0794         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)            832.41       149.49       981.90       0.4529          1.0552            1.0461         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)            832.41        99.04       931.45       0.3463          1.0912            1.0794         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)            832.41       153.68       986.09       0.4529          1.0552            1.0461         1.93
IVF-Binary-256-nl158-random (self)                       832.41       209.62     1_042.03       0.3507          1.0862            1.0777         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             600.01        46.23       646.24       0.1413          1.3601            1.3150         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             600.01        47.10       647.11       0.1412          1.3605            1.3154         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             600.01        50.18       650.20       0.1412          1.3605            1.3154         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            600.01       100.78       700.79       0.3893          1.0689            1.0625         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            600.01       148.69       748.70       0.4976          1.0427            1.0371         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            600.01       100.75       700.76       0.3891          1.0690            1.0625         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            600.01       180.52       780.53       0.4973          1.0428            1.0372         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            600.01        99.39       699.40       0.3891          1.0690            1.0625         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            600.01       153.31       753.32       0.4973          1.0428            1.0372         2.00
IVF-Binary-256-nl223-random (self)                       600.01       218.16       818.17       0.3943          1.0643            1.0615         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             802.46        48.81       851.27       0.1496          1.3359            1.2904         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             802.46        49.15       851.62       0.1495          1.3365            1.2906         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             802.46        52.59       855.05       0.1495          1.3366            1.2906         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            802.46       100.39       902.85       0.4018          1.0649            1.0585         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            802.46       151.73       954.19       0.5055          1.0413            1.0358         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            802.46        99.17       901.63       0.4016          1.0650            1.0587         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            802.46       152.58       955.04       0.5051          1.0414            1.0359         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            802.46       102.70       905.17       0.4016          1.0650            1.0587         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            802.46       154.43       956.89       0.5051          1.0414            1.0359         2.09
IVF-Binary-256-nl316-random (self)                       802.46       226.09     1_028.55       0.4063          1.0606            1.0577         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)                 878.73        43.08       921.81       0.1190          1.4540            1.4124         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)                878.73        43.32       922.05       0.1190          1.4540            1.4124         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)                878.73        44.87       923.60       0.1190          1.4540            1.4124         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)                878.73        95.69       974.42       0.3368          1.0992            1.0816         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)                878.73       146.90     1_025.63       0.4434          1.0613            1.0475         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)               878.73        95.34       974.08       0.3368          1.0992            1.0816         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)               878.73       152.49     1_031.22       0.4434          1.0613            1.0475         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)               878.73        95.21       973.94       0.3368          1.0992            1.0816         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)               878.73       151.79     1_030.52       0.4434          1.0613            1.0475         1.93
IVF-Binary-256-nl158-pca (self)                          878.73       208.75     1_087.48       0.3434          1.0923            1.0796         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                645.62        45.30       690.92       0.1377          1.3704            1.3177         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                645.62        46.31       691.93       0.1377          1.3708            1.3180         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                645.62        49.70       695.32       0.1377          1.3708            1.3180         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               645.62        98.71       744.33       0.3828          1.0754            1.0637         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               645.62       148.85       794.47       0.4958          1.0458            1.0375         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               645.62        96.52       742.13       0.3827          1.0755            1.0638         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               645.62       149.95       795.57       0.4957          1.0458            1.0375         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               645.62       101.23       746.84       0.3827          1.0755            1.0638         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               645.62       159.50       805.12       0.4957          1.0458            1.0375         2.00
IVF-Binary-256-nl223-pca (self)                          645.62       213.93       859.55       0.3896          1.0692            1.0620         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                825.03        47.74       872.78       0.1471          1.3419            1.2914         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                825.03        51.43       876.46       0.1471          1.3424            1.2916         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                825.03        54.64       879.67       0.1471          1.3425            1.2916         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               825.03       100.17       925.21       0.3970          1.0703            1.0594         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               825.03       152.86       977.89       0.5069          1.0437            1.0356         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               825.03        97.87       922.90       0.3968          1.0704            1.0594         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               825.03       150.51       975.54       0.5067          1.0438            1.0356         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               825.03       106.40       931.43       0.3968          1.0704            1.0594         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               825.03       153.96       979.00       0.5067          1.0438            1.0356         2.09
IVF-Binary-256-nl316-pca (self)                          825.03       221.30     1_046.33       0.4025          1.0646            1.0582         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)              854.68        58.73       913.41       0.1607          1.3465            1.3235         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)             854.68        61.79       916.47       0.1607          1.3465            1.3235         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)             854.68        66.55       921.23       0.1607          1.3465            1.3235         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)             854.68       115.52       970.19       0.3812          1.0681            1.0667         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)             854.68       171.75     1_026.43       0.4908          1.0417            1.0390         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)            854.68       118.95       973.62       0.3812          1.0681            1.0667         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)            854.68       175.11     1_029.79       0.4908          1.0417            1.0390         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)            854.68       120.52       975.20       0.3812          1.0681            1.0667         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)            854.68       176.44     1_031.12       0.4908          1.0417            1.0390         3.71
IVF-Binary-512-nl158-random (self)                       854.68       293.54     1_148.22       0.3832          1.0664            1.0666         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             629.00        63.27       692.28       0.1711          1.2998            1.2780         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             629.00        65.56       694.56       0.1711          1.3001            1.2782         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             629.00        70.44       699.45       0.1711          1.3001            1.2782         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            629.00       119.23       748.23       0.4019          1.0605            1.0593         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            629.00       172.47       801.47       0.5140          1.0373            1.0348         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            629.00       118.94       747.94       0.4017          1.0605            1.0593         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            629.00       176.43       805.43       0.5137          1.0374            1.0349         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            629.00       123.58       752.59       0.4017          1.0605            1.0593         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            629.00       177.71       806.72       0.5137          1.0374            1.0349         3.77
IVF-Binary-512-nl223-random (self)                       629.00       315.33       944.34       0.4039          1.0592            1.0592         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             798.64        66.39       865.03       0.1755          1.2880            1.2669         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             798.64        67.86       866.50       0.1754          1.2885            1.2672         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             798.64        72.27       870.91       0.1754          1.2885            1.2672         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            798.64       121.50       920.14       0.4058          1.0594            1.0581         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            798.64       185.87       984.51       0.5175          1.0368            1.0342         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            798.64       121.46       920.09       0.4056          1.0595            1.0582         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            798.64       215.29     1_013.93       0.5171          1.0369            1.0342         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            798.64       130.44       929.08       0.4056          1.0595            1.0582         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            798.64       196.01       994.65       0.5171          1.0369            1.0342         3.86
IVF-Binary-512-nl316-random (self)                       798.64       308.43     1_107.06       0.4086          1.0581            1.0580         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)                 931.15        62.42       993.57       0.1581          1.3474            1.3214         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)                931.15        63.51       994.66       0.1581          1.3474            1.3214         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)                931.15        64.41       995.55       0.1581          1.3474            1.3214         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)                931.15       115.03     1_046.18       0.3810          1.0702            1.0658         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)                931.15       171.52     1_102.67       0.4928          1.0428            1.0383         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)               931.15       116.74     1_047.88       0.3810          1.0702            1.0658         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)               931.15       174.71     1_105.85       0.4928          1.0428            1.0383         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)               931.15       118.91     1_050.05       0.3810          1.0702            1.0658         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)               931.15       177.97     1_109.11       0.4928          1.0428            1.0383         3.71
IVF-Binary-512-nl158-pca (self)                          931.15       290.28     1_221.42       0.3843          1.0672            1.0657         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                657.29        64.57       721.86       0.1694          1.3033            1.2746         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                657.29        64.94       722.23       0.1694          1.3034            1.2747         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                657.29        69.14       726.43       0.1694          1.3034            1.2747         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               657.29       120.03       777.32       0.4036          1.0619            1.0583         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               657.29       170.90       828.19       0.5164          1.0383            1.0342         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               657.29       118.72       776.01       0.4035          1.0619            1.0583         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               657.29       180.12       837.41       0.5162          1.0384            1.0342         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               657.29       123.23       780.51       0.4035          1.0619            1.0583         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               657.29       180.24       837.52       0.5162          1.0384            1.0342         3.77
IVF-Binary-512-nl223-pca (self)                          657.29       294.04       951.33       0.4059          1.0599            1.0584         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                845.46        67.69       913.16       0.1733          1.2925            1.2652         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                845.46        67.00       912.46       0.1732          1.2927            1.2653         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                845.46        71.25       916.72       0.1732          1.2927            1.2653         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               845.46       120.59       966.05       0.4089          1.0604            1.0565         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               845.46       174.57     1_020.03       0.5220          1.0374            1.0334         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               845.46       120.52       965.98       0.4088          1.0605            1.0565         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               845.46       174.44     1_019.91       0.5217          1.0374            1.0334         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               845.46       127.19       972.66       0.4088          1.0605            1.0565         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               845.46       179.96     1_025.42       0.5217          1.0374            1.0334         3.86
IVF-Binary-512-nl316-pca (self)                          845.46       301.84     1_147.30       0.4117          1.0583            1.0566         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)             913.61        92.15     1_005.76       0.1937          1.2738            1.2669         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)            913.61        96.45     1_010.07       0.1937          1.2738            1.2669         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)            913.61        99.99     1_013.60       0.1937          1.2738            1.2669         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)            913.61       165.19     1_078.80       0.4227          1.0546            1.0549         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)            913.61       209.18     1_122.79       0.5450          1.0325            1.0305         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)           913.61       163.21     1_076.83       0.4227          1.0546            1.0549         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)           913.61       216.62     1_130.23       0.5450          1.0325            1.0305         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)           913.61       157.56     1_071.17       0.4227          1.0546            1.0549         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)           913.61       218.10     1_131.71       0.5450          1.0325            1.0305         7.26
IVF-Binary-1024-nl158-random (self)                      913.61       430.26     1_343.87       0.4246          1.0544            1.0549         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            655.03        96.05       751.08       0.1973          1.2556            1.2488         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            655.03        98.08       753.11       0.1973          1.2558            1.2489         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            655.03       104.05       759.09       0.1973          1.2558            1.2489         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           655.03       156.80       811.83       0.4342          1.0516            1.0516         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           655.03       211.27       866.31       0.5563          1.0309            1.0289         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           655.03       162.90       817.93       0.4341          1.0517            1.0516         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           655.03       216.77       871.81       0.5562          1.0309            1.0289         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           655.03       162.13       817.16       0.4341          1.0517            1.0516         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           655.03       220.91       875.94       0.5562          1.0309            1.0289         7.32
IVF-Binary-1024-nl223-random (self)                      655.03       430.93     1_085.96       0.4353          1.0515            1.0518         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            866.38       102.46       968.83       0.1989          1.2508            1.2444         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)            866.38       103.55       969.93       0.1988          1.2511            1.2446         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)            866.38       109.13       975.51       0.1988          1.2511            1.2446         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)           866.38       158.33     1_024.70       0.4365          1.0510            1.0509         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)           866.38       220.97     1_087.35       0.5576          1.0306            1.0287         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)           866.38       157.55     1_023.93       0.4364          1.0511            1.0509         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)           866.38       218.98     1_085.35       0.5573          1.0307            1.0288         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)           866.38       164.85     1_031.22       0.4364          1.0511            1.0509         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)           866.38       228.99     1_095.37       0.5573          1.0307            1.0288         7.42
IVF-Binary-1024-nl316-random (self)                      866.38       460.10     1_326.47       0.4378          1.0509            1.0513         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)                944.99        98.38     1_043.36       0.1929          1.2710            1.2632         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)               944.99        96.96     1_041.94       0.1929          1.2710            1.2632         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)               944.99        99.98     1_044.96       0.1929          1.2710            1.2632         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)               944.99       153.80     1_098.78       0.4240          1.0542            1.0540         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)               944.99       214.37     1_159.35       0.5461          1.0324            1.0302         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)              944.99       156.00     1_100.99       0.4240          1.0542            1.0540         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)              944.99       214.72     1_159.71       0.5461          1.0324            1.0302         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)              944.99       159.07     1_104.05       0.4240          1.0542            1.0540         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)              944.99       223.70     1_168.69       0.5461          1.0324            1.0302         7.26
IVF-Binary-1024-nl158-pca (self)                         944.99       432.46     1_377.45       0.4249          1.0542            1.0544         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               692.90        97.01       789.91       0.1974          1.2518            1.2441         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               692.90        98.70       791.60       0.1974          1.2519            1.2442         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               692.90       104.65       797.55       0.1974          1.2519            1.2442         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              692.90       156.11       849.01       0.4355          1.0511            1.0508         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              692.90       219.61       912.51       0.5584          1.0305            1.0281         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              692.90       159.28       852.18       0.4354          1.0511            1.0508         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              692.90       217.79       910.69       0.5581          1.0306            1.0282         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              692.90       161.98       854.88       0.4354          1.0511            1.0508         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              692.90       222.97       915.87       0.5581          1.0306            1.0282         7.32
IVF-Binary-1024-nl223-pca (self)                         692.90       433.26     1_126.16       0.4363          1.0511            1.0512         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               875.77       100.54       976.31       0.1987          1.2475            1.2401         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)               875.77       100.20       975.97       0.1987          1.2476            1.2401         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)               875.77       108.54       984.31       0.1987          1.2476            1.2401         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)              875.77       156.55     1_032.33       0.4386          1.0503            1.0501         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)              875.77       214.37     1_090.14       0.5611          1.0302            1.0279         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)              875.77       156.42     1_032.19       0.4384          1.0504            1.0501         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)              875.77       216.47     1_092.24       0.5610          1.0302            1.0279         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)              875.77       164.58     1_040.35       0.4384          1.0504            1.0501         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)              875.77       220.36     1_096.13       0.5610          1.0302            1.0279         7.42
IVF-Binary-1024-nl316-pca (self)                         875.77       456.32     1_332.09       0.4396          1.0504            1.0505         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)                827.42       152.29       979.71       0.1216          1.4959            1.4465         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               827.42       152.40       979.82       0.1216          1.4959            1.4465         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               827.42       154.50       981.92       0.1216          1.4959            1.4465         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               827.42       188.64     1_016.06       0.3305          1.1023            1.0867         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               827.42       342.84     1_170.26       0.4405          1.0617            1.0486         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              827.42       189.68     1_017.10       0.3305          1.1023            1.0867         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              827.42       335.99     1_163.41       0.4405          1.0617            1.0486         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              827.42       193.98     1_021.40       0.3305          1.1023            1.0867         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              827.42       340.54     1_167.96       0.4405          1.0617            1.0486         1.68
IVF-Binary-256-nl158-sign (self)                         827.42       536.32     1_363.74       0.3358          1.0972            1.0848         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               569.22       147.60       716.82       0.1228          1.4832            1.4276         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               569.22       150.16       719.38       0.1228          1.4843            1.4282         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               569.22       152.12       721.34       0.1228          1.4844            1.4282         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              569.22       189.05       758.27       0.3559          1.0888            1.0752         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              569.22       349.34       918.56       0.4583          1.0561            1.0442         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              569.22       210.77       779.99       0.3557          1.0890            1.0752         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              569.22       339.41       908.63       0.4580          1.0562            1.0443         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              569.22       194.69       763.91       0.3557          1.0890            1.0752         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              569.22       340.77       909.99       0.4580          1.0562            1.0443         1.75
IVF-Binary-256-nl223-sign (self)                         569.22       507.77     1_076.99       0.3611          1.0841            1.0736         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               797.26       155.63       952.90       0.1239          1.4676            1.4148         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               797.26       152.72       949.98       0.1237          1.4698            1.4160         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               797.26       165.97       963.23       0.1237          1.4700            1.4161         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              797.26       198.86       996.12       0.3598          1.0872            1.0741         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              797.26       341.32     1_138.58       0.4582          1.0559            1.0445         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              797.26       194.71       991.97       0.3596          1.0874            1.0742         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              797.26       356.61     1_153.87       0.4577          1.0561            1.0446         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              797.26       213.90     1_011.16       0.3596          1.0874            1.0742         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              797.26       378.23     1_175.50       0.4577          1.0561            1.0446         1.84
IVF-Binary-256-nl316-sign (self)                         797.26       532.00     1_329.26       0.3647          1.0824            1.0724         1.84
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
Exhaustive (query)                                        70.24     1_270.73     1_340.96       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         70.24     4_231.02     4_301.26       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                153.53       265.17       418.70       0.1109          1.3512            1.3092         2.03
ExhaustiveBinary-256-random-rf10 (query)                 153.53       393.96       547.49       0.3143          1.0825            1.0613         2.03
ExhaustiveBinary-256-random-rf20 (query)                 153.53       520.98       674.51       0.4100          1.0523            1.0368         2.03
ExhaustiveBinary-256-random (self)                       153.53     1_227.34     1_380.87       0.3161          1.0784            1.0600         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   239.92       266.96       506.88       0.1167          1.3480            1.2981         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    239.92       397.73       637.66       0.3159          1.0791            1.0596         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    239.92       533.28       773.20       0.4121          1.0505            1.0362         2.03
ExhaustiveBinary-256-pca (self)                          239.92     1_236.34     1_476.26       0.3171          1.0782            1.0588         2.03
ExhaustiveBinary-512-random_no_rr (query)                205.17       378.10       583.27       0.1528          1.2601            1.2299         4.05
ExhaustiveBinary-512-random-rf10 (query)                 205.17       533.37       738.54       0.3465          1.0565            1.0514         4.05
ExhaustiveBinary-512-random-rf20 (query)                 205.17       660.74       865.91       0.4452          1.0358            1.0312         4.05
ExhaustiveBinary-512-random (self)                       205.17     1_662.16     1_867.33       0.3476          1.0547            1.0512         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   313.05       386.43       699.48       0.1558          1.2535            1.2254         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    313.05       529.20       842.25       0.3512          1.0523            1.0507         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    313.05       665.90       978.95       0.4484          1.0329            1.0309         4.05
ExhaustiveBinary-512-pca (self)                          313.05     1_749.37     2_062.42       0.3515          1.0522            1.0505         4.05
ExhaustiveBinary-1024-random_no_rr (query)               256.80       551.15       807.95       0.1816          1.2043            1.1936         8.11
ExhaustiveBinary-1024-random-rf10 (query)                256.80       707.39       964.19       0.3747          1.0447            1.0452         8.11
ExhaustiveBinary-1024-random-rf20 (query)                256.80       861.42     1_118.22       0.4789          1.0282            1.0270         8.11
ExhaustiveBinary-1024-random (self)                      256.80     2_323.17     2_579.97       0.3754          1.0447            1.0451         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  365.99       554.90       920.89       0.1832          1.2013            1.1905         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   365.99       718.76     1_084.75       0.3798          1.0434            1.0443         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   365.99       872.42     1_238.40       0.4867          1.0272            1.0261         8.11
ExhaustiveBinary-1024-pca (self)                         365.99     2_326.86     2_692.85       0.3787          1.0436            1.0444         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   91.85       694.03       785.89       0.1518          1.2701            1.2528         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    91.85       752.01       843.86       0.3399          1.0607            1.0535         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    91.85     1_133.53     1_225.38       0.4406          1.0369            1.0319         3.05
ExhaustiveBinary-512-sign (self)                          91.85     2_392.24     2_484.09       0.3409          1.0595            1.0531         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)            1_688.10        78.24     1_766.34       0.1157          1.3331            1.2963         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           1_688.10        80.62     1_768.72       0.1157          1.3331            1.2963         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           1_688.10        86.08     1_774.19       0.1157          1.3331            1.2963         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           1_688.10       166.98     1_855.09       0.3225          1.0778            1.0583         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           1_688.10       245.46     1_933.56       0.4188          1.0495            1.0350         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          1_688.10       155.28     1_843.39       0.3225          1.0778            1.0583         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          1_688.10       256.81     1_944.91       0.4188          1.0495            1.0350         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          1_688.10       150.65     1_838.76       0.3225          1.0778            1.0583         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          1_688.10       245.81     1_933.92       0.4188          1.0495            1.0350         2.34
IVF-Binary-256-nl158-random (self)                     1_688.10       285.01     1_973.12       0.3241          1.0737            1.0571         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)             973.90        71.46     1_045.36       0.1325          1.2776            1.2358         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)             973.90        73.28     1_047.18       0.1325          1.2779            1.2360         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)             973.90        76.49     1_050.39       0.1325          1.2780            1.2360         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)            973.90       159.93     1_133.83       0.3688          1.0554            1.0455         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)            973.90       250.86     1_224.77       0.4683          1.0358            1.0278         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)            973.90       158.93     1_132.83       0.3685          1.0555            1.0456         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)            973.90       252.88     1_226.78       0.4678          1.0359            1.0279         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)            973.90       160.66     1_134.56       0.3685          1.0555            1.0456         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)            973.90       264.41     1_238.31       0.4678          1.0359            1.0279         2.47
IVF-Binary-256-nl223-random (self)                       973.90       308.92     1_282.82       0.3705          1.0515            1.0449         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)           1_329.59        80.66     1_410.25       0.1435          1.2534            1.2115         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           1_329.59        81.01     1_410.60       0.1434          1.2536            1.2119         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           1_329.59        82.51     1_412.10       0.1434          1.2542            1.2121         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          1_329.59       167.74     1_497.33       0.3836          1.0508            1.0420         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          1_329.59       258.57     1_588.16       0.4826          1.0337            1.0261         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          1_329.59       165.13     1_494.72       0.3833          1.0510            1.0420         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          1_329.59       259.84     1_589.43       0.4820          1.0338            1.0262         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          1_329.59       167.48     1_497.07       0.3832          1.0510            1.0420         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          1_329.59       266.68     1_596.28       0.4818          1.0339            1.0262         2.65
IVF-Binary-256-nl316-random (self)                     1_329.59       332.20     1_661.79       0.3852          1.0471            1.0414         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               1_753.13        68.23     1_821.36       0.1212          1.3300            1.2889         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              1_753.13        68.66     1_821.79       0.1212          1.3300            1.2889         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              1_753.13        70.53     1_823.66       0.1212          1.3300            1.2889         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              1_753.13       149.10     1_902.23       0.3222          1.0753            1.0574         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              1_753.13       242.66     1_995.79       0.4196          1.0483            1.0348         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             1_753.13       155.11     1_908.24       0.3222          1.0753            1.0574         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             1_753.13       241.75     1_994.88       0.4196          1.0483            1.0348         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             1_753.13       151.15     1_904.28       0.3222          1.0753            1.0574         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             1_753.13       243.70     1_996.83       0.4196          1.0483            1.0348         2.34
IVF-Binary-256-nl158-pca (self)                        1_753.13       288.57     2_041.70       0.3232          1.0740            1.0566         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              1_099.61        71.75     1_171.36       0.1366          1.2800            1.2327         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              1_099.61        72.98     1_172.58       0.1366          1.2802            1.2328         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              1_099.61        79.43     1_179.03       0.1366          1.2802            1.2328         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             1_099.61       162.54     1_262.15       0.3637          1.0553            1.0460         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             1_099.61       260.76     1_360.36       0.4640          1.0357            1.0283         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             1_099.61       173.28     1_272.89       0.3636          1.0553            1.0460         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             1_099.61       253.94     1_353.54       0.4638          1.0357            1.0283         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             1_099.61       161.77     1_261.38       0.3636          1.0553            1.0460         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             1_099.61       252.80     1_352.41       0.4638          1.0357            1.0283         2.47
IVF-Binary-256-nl223-pca (self)                        1_099.61       313.62     1_413.23       0.3658          1.0535            1.0453         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              1_440.41        78.20     1_518.61       0.1456          1.2588            1.2105         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              1_440.41        80.66     1_521.07       0.1456          1.2588            1.2106         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              1_440.41        81.38     1_521.78       0.1455          1.2592            1.2107         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             1_440.41       164.62     1_605.03       0.3774          1.0509            1.0426         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             1_440.41       257.08     1_697.48       0.4766          1.0332            1.0267         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             1_440.41       164.42     1_604.83       0.3773          1.0509            1.0427         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             1_440.41       256.78     1_697.19       0.4760          1.0333            1.0268         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             1_440.41       168.67     1_609.07       0.3772          1.0509            1.0427         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             1_440.41       259.52     1_699.92       0.4759          1.0333            1.0268         2.65
IVF-Binary-256-nl316-pca (self)                        1_440.41       343.29     1_783.70       0.3792          1.0492            1.0422         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)            1_742.62        92.50     1_835.12       0.1554          1.2515            1.2240         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)           1_742.62        96.24     1_838.86       0.1554          1.2515            1.2240         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)           1_742.62        97.70     1_840.32       0.1554          1.2515            1.2240         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)           1_742.62       182.18     1_924.80       0.3517          1.0543            1.0500         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)           1_742.62       272.43     2_015.05       0.4505          1.0347            1.0305         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)          1_742.62       183.08     1_925.70       0.3517          1.0543            1.0500         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)          1_742.62       277.53     2_020.15       0.4505          1.0347            1.0305         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)          1_742.62       185.93     1_928.55       0.3517          1.0543            1.0500         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)          1_742.62       282.11     2_024.73       0.4505          1.0347            1.0305         4.36
IVF-Binary-512-nl158-random (self)                     1_742.62       414.56     2_157.18       0.3527          1.0528            1.0499         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)           1_068.57       100.00     1_168.57       0.1644          1.2242            1.1976         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)           1_068.57       101.49     1_170.06       0.1643          1.2245            1.1979         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)           1_068.57       108.93     1_177.50       0.1643          1.2246            1.1979         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)          1_068.57       196.43     1_265.00       0.3693          1.0479            1.0450         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)          1_068.57       295.21     1_363.79       0.4691          1.0312            1.0277         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)          1_068.57       194.45     1_263.02       0.3691          1.0480            1.0450         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)          1_068.57       293.17     1_361.74       0.4685          1.0312            1.0278         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)          1_068.57       195.48     1_264.05       0.3691          1.0480            1.0450         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)          1_068.57       306.73     1_375.30       0.4685          1.0312            1.0278         4.49
IVF-Binary-512-nl223-random (self)                     1_068.57       434.24     1_502.81       0.3701          1.0469            1.0450         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)           1_438.52       106.03     1_544.55       0.1680          1.2145            1.1880         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)           1_438.52       108.92     1_547.44       0.1680          1.2148            1.1883         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)           1_438.52       110.30     1_548.82       0.1679          1.2151            1.1885         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)          1_438.52       203.83     1_642.35       0.3756          1.0469            1.0438         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)          1_438.52       296.26     1_734.78       0.4755          1.0304            1.0270         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)          1_438.52       195.54     1_634.06       0.3754          1.0470            1.0438         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)          1_438.52       288.41     1_726.93       0.4747          1.0305            1.0272         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)          1_438.52       197.32     1_635.84       0.3753          1.0470            1.0439         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)          1_438.52       293.99     1_732.51       0.4744          1.0306            1.0272         4.67
IVF-Binary-512-nl316-random (self)                     1_438.52       449.66     1_888.18       0.3761          1.0456            1.0437         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)               1_853.40        93.00     1_946.41       0.1578          1.2477            1.2198         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)              1_853.40        95.23     1_948.63       0.1578          1.2477            1.2198         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)              1_853.40        98.05     1_951.45       0.1578          1.2477            1.2198         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)              1_853.40       191.74     2_045.14       0.3546          1.0514            1.0497         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)              1_853.40       274.80     2_128.20       0.4526          1.0324            1.0303         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)             1_853.40       184.99     2_038.39       0.3546          1.0514            1.0497         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)             1_853.40       281.80     2_135.21       0.4526          1.0324            1.0303         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)             1_853.40       184.79     2_038.19       0.3546          1.0514            1.0497         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)             1_853.40       279.55     2_132.95       0.4526          1.0324            1.0303         4.36
IVF-Binary-512-nl158-pca (self)                        1_853.40       418.44     2_271.85       0.3548          1.0513            1.0495         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_169.47        98.99     1_268.46       0.1671          1.2190            1.1937         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_169.47       102.34     1_271.81       0.1671          1.2192            1.1940         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_169.47       104.90     1_274.37       0.1671          1.2192            1.1940         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_169.47       189.48     1_358.95       0.3708          1.0459            1.0453         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_169.47       283.91     1_453.38       0.4723          1.0291            1.0275         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_169.47       188.16     1_357.64       0.3705          1.0459            1.0453         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_169.47       288.15     1_457.62       0.4716          1.0291            1.0276         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_169.47       192.84     1_362.31       0.3705          1.0459            1.0453         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_169.47       291.79     1_461.26       0.4716          1.0291            1.0276         4.49
IVF-Binary-512-nl223-pca (self)                        1_169.47       433.91     1_603.38       0.3711          1.0457            1.0452         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)              1_503.98       107.78     1_611.76       0.1706          1.2099            1.1842         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)              1_503.98       108.29     1_612.27       0.1706          1.2102            1.1845         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)              1_503.98       112.18     1_616.16       0.1705          1.2103            1.1846         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)             1_503.98       195.77     1_699.75       0.3771          1.0444            1.0437         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)             1_503.98       290.06     1_794.03       0.4786          1.0283            1.0267         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)             1_503.98       193.31     1_697.28       0.3768          1.0445            1.0438         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)             1_503.98       287.60     1_791.57       0.4778          1.0284            1.0267         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)             1_503.98       198.87     1_702.84       0.3767          1.0445            1.0438         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)             1_503.98       292.34     1_796.31       0.4776          1.0285            1.0268         4.67
IVF-Binary-512-nl316-pca (self)                        1_503.98       452.34     1_956.32       0.3772          1.0444            1.0438         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)           1_797.52       149.90     1_947.42       0.1827          1.2009            1.1907         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)          1_797.52       146.72     1_944.24       0.1827          1.2009            1.1907         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)          1_797.52       149.52     1_947.04       0.1827          1.2009            1.1907         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)          1_797.52       235.46     2_032.98       0.3773          1.0441            1.0445         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)          1_797.52       336.52     2_134.04       0.4819          1.0279            1.0266         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)         1_797.52       236.34     2_033.86       0.3773          1.0441            1.0445         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)         1_797.52       342.19     2_139.71       0.4819          1.0279            1.0266         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)         1_797.52       256.96     2_054.48       0.3773          1.0441            1.0445         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)         1_797.52       346.12     2_143.64       0.4819          1.0279            1.0266         8.42
IVF-Binary-1024-nl158-random (self)                    1_797.52       613.58     2_411.10       0.3779          1.0441            1.0445         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_126.65       153.41     1_280.05       0.1855          1.1895            1.1800         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_126.65       152.06     1_278.70       0.1855          1.1897            1.1802         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_126.65       157.97     1_284.62       0.1855          1.1897            1.1802         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_126.65       245.25     1_371.89       0.3862          1.0418            1.0422         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_126.65       346.49     1_473.14       0.4929          1.0264            1.0252         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_126.65       245.89     1_372.54       0.3859          1.0419            1.0423         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_126.65       355.12     1_481.77       0.4924          1.0265            1.0253         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_126.65       251.89     1_378.54       0.3859          1.0419            1.0423         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_126.65       358.16     1_484.81       0.4924          1.0265            1.0253         8.54
IVF-Binary-1024-nl223-random (self)                    1_126.65       628.85     1_755.50       0.3871          1.0418            1.0422         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)          1_442.41       158.60     1_601.01       0.1870          1.1851            1.1757         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)          1_442.41       158.67     1_601.08       0.1869          1.1854            1.1760         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)          1_442.41       163.23     1_605.64       0.1869          1.1855            1.1761         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)         1_442.41       250.72     1_693.13       0.3891          1.0413            1.0415         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)         1_442.41       352.27     1_794.68       0.4957          1.0261            1.0248         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)         1_442.41       253.29     1_695.70       0.3887          1.0414            1.0416         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)         1_442.41       353.91     1_796.32       0.4951          1.0262            1.0248         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)         1_442.41       256.03     1_698.44       0.3886          1.0414            1.0416         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)         1_442.41       364.31     1_806.72       0.4950          1.0262            1.0249         8.73
IVF-Binary-1024-nl316-random (self)                    1_442.41       659.73     2_102.14       0.3899          1.0413            1.0416         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)              1_919.55       145.92     2_065.48       0.1841          1.1989            1.1881         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)             1_919.55       147.41     2_066.96       0.1841          1.1989            1.1881         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)             1_919.55       152.21     2_071.76       0.1841          1.1989            1.1881         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)             1_919.55       236.04     2_155.59       0.3820          1.0431            1.0438         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)             1_919.55       339.57     2_259.12       0.4889          1.0270            1.0258         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)            1_919.55       239.45     2_159.00       0.3820          1.0431            1.0438         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)            1_919.55       345.37     2_264.92       0.4889          1.0270            1.0258         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)            1_919.55       243.20     2_162.75       0.3820          1.0431            1.0438         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)            1_919.55       351.94     2_271.49       0.4889          1.0270            1.0258         8.42
IVF-Binary-1024-nl158-pca (self)                       1_919.55       633.25     2_552.80       0.3808          1.0432            1.0439         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_222.04       150.82     1_372.86       0.1869          1.1877            1.1786         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_222.04       153.65     1_375.69       0.1869          1.1879            1.1788         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_222.04       158.51     1_380.56       0.1869          1.1879            1.1788         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_222.04       245.49     1_467.53       0.3905          1.0407            1.0416         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_222.04       348.13     1_570.17       0.4989          1.0255            1.0247         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_222.04       244.36     1_466.40       0.3902          1.0408            1.0417         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_222.04       352.55     1_574.59       0.4984          1.0256            1.0247         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_222.04       251.43     1_473.47       0.3902          1.0408            1.0417         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_222.04       362.24     1_584.28       0.4984          1.0256            1.0247         8.54
IVF-Binary-1024-nl223-pca (self)                       1_222.04       643.32     1_865.36       0.3897          1.0409            1.0418         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_559.17       166.35     1_725.53       0.1883          1.1833            1.1740         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_559.17       167.05     1_726.23       0.1882          1.1836            1.1743         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_559.17       168.32     1_727.49       0.1882          1.1837            1.1744         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_559.17       261.41     1_820.58       0.3938          1.0401            1.0409         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_559.17       358.74     1_917.91       0.5030          1.0252            1.0243         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_559.17       250.47     1_809.65       0.3935          1.0402            1.0410         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_559.17       375.86     1_935.04       0.5023          1.0252            1.0243         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_559.17       258.32     1_817.50       0.3934          1.0402            1.0410         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_559.17       364.23     1_923.41       0.5022          1.0253            1.0243         8.73
IVF-Binary-1024-nl316-pca (self)                       1_559.17       663.74     2_222.92       0.3929          1.0403            1.0411         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)              1_683.69       284.56     1_968.25       0.1520          1.2698            1.2528         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)             1_683.69       289.82     1_973.50       0.1520          1.2698            1.2528         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)             1_683.69       291.48     1_975.17       0.1520          1.2698            1.2528         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)             1_683.69       352.69     2_036.38       0.3410          1.0600            1.0529         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)             1_683.69       657.42     2_341.11       0.4418          1.0366            1.0317         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)            1_683.69       354.79     2_038.47       0.3410          1.0600            1.0529         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)            1_683.69       643.96     2_327.65       0.4418          1.0366            1.0317         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)            1_683.69       362.31     2_046.00       0.3410          1.0600            1.0529         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)            1_683.69       647.06     2_330.75       0.4418          1.0366            1.0317         3.36
IVF-Binary-512-nl158-sign (self)                       1_683.69       977.51     2_661.19       0.3419          1.0590            1.0526         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               943.51       288.10     1_231.61       0.1530          1.2631            1.2472         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               943.51       290.97     1_234.49       0.1529          1.2639            1.2483         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               943.51       294.00     1_237.52       0.1529          1.2639            1.2483         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              943.51       363.22     1_306.73       0.3502          1.0559            1.0502         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              943.51       663.36     1_606.87       0.4489          1.0348            1.0306         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              943.51       362.00     1_305.52       0.3500          1.0560            1.0502         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              943.51       652.00     1_595.51       0.4482          1.0349            1.0307         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              943.51       367.85     1_311.36       0.3500          1.0560            1.0502         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              943.51       659.34     1_602.86       0.4482          1.0349            1.0307         3.49
IVF-Binary-512-nl223-sign (self)                         943.51       992.82     1_936.34       0.3506          1.0550            1.0500         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)             1_316.17       294.59     1_610.76       0.1530          1.2630            1.2450         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)             1_316.17       296.34     1_612.51       0.1529          1.2642            1.2464         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)             1_316.17       300.33     1_616.50       0.1528          1.2648            1.2472         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)            1_316.17       370.96     1_687.14       0.3525          1.0552            1.0494         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)            1_316.17       653.09     1_969.26       0.4493          1.0348            1.0305         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)            1_316.17       368.11     1_684.28       0.3521          1.0554            1.0496         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)            1_316.17       656.23     1_972.40       0.4482          1.0350            1.0306         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)            1_316.17       372.86     1_689.03       0.3519          1.0555            1.0496         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)            1_316.17       662.14     1_978.31       0.4478          1.0351            1.0307         3.67
IVF-Binary-512-nl316-sign (self)                       1_316.17     1_015.39     2_331.56       0.3528          1.0544            1.0494         3.67
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
Exhaustive (query)                                       100.29     1_928.19     2_028.48       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.29     6_499.74     6_600.03       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                196.24       277.90       474.14       0.1140          1.2809            1.2433         2.28
ExhaustiveBinary-256-random-rf10 (query)                 196.24       423.04       619.28       0.3148          1.0656            1.0476         2.28
ExhaustiveBinary-256-random-rf20 (query)                 196.24       569.81       766.05       0.4075          1.0420            1.0293         2.28
ExhaustiveBinary-256-random (self)                       196.24     1_312.44     1_508.68       0.3168          1.0618            1.0471         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   410.10       277.26       687.36       0.1054          1.3026            1.2617         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    410.10       447.22       857.32       0.3012          1.0735            1.0517         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    410.10       566.14       976.24       0.3931          1.0471            1.0315         2.28
ExhaustiveBinary-256-pca (self)                          410.10     1_298.15     1_708.25       0.3048          1.0710            1.0504         2.28
ExhaustiveBinary-512-random_no_rr (query)                298.50       403.96       702.46       0.1506          1.2094            1.1809         4.55
ExhaustiveBinary-512-random-rf10 (query)                 298.50       567.31       865.81       0.3395          1.0453            1.0426         4.55
ExhaustiveBinary-512-random-rf20 (query)                 298.50       727.49     1_025.99       0.4326          1.0293            1.0264         4.55
ExhaustiveBinary-512-random (self)                       298.50     1_803.57     2_102.07       0.3401          1.0435            1.0423         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   504.64       406.01       910.65       0.1459          1.2162            1.1914         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    504.64       563.90     1_068.54       0.3341          1.0468            1.0435         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    504.64       721.72     1_226.35       0.4278          1.0295            1.0269         4.55
ExhaustiveBinary-512-pca (self)                          504.64     1_796.14     2_300.78       0.3355          1.0454            1.0433         4.55
ExhaustiveBinary-1024-random_no_rr (query)               497.22       637.18     1_134.39       0.1761          1.1673            1.1571         9.11
ExhaustiveBinary-1024-random-rf10 (query)                497.22       798.44     1_295.65       0.3603          1.0383            1.0383         9.11
ExhaustiveBinary-1024-random-rf20 (query)                497.22       985.96     1_483.17       0.4618          1.0244            1.0230         9.11
ExhaustiveBinary-1024-random (self)                      497.22     2_627.01     3_124.22       0.3602          1.0377            1.0383         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  712.57       626.87     1_339.44       0.1756          1.1686            1.1586         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   712.57       802.59     1_515.16       0.3594          1.0382            1.0385         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   712.57       980.17     1_692.75       0.4576          1.0246            1.0237         9.11
ExhaustiveBinary-1024-pca (self)                         712.57     2_656.61     3_369.19       0.3584          1.0383            1.0389         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  131.07       857.83       988.90       0.1691          1.1871            1.1718         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   131.07       940.79     1_071.86       0.3431          1.0433            1.0415         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   131.07     1_465.15     1_596.22       0.4437          1.0266            1.0250         4.58
ExhaustiveBinary-768-sign (self)                         131.07     3_038.87     3_169.94       0.3438          1.0424            1.0413         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)            2_387.89        94.43     2_482.32       0.1164          1.2717            1.2403         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)           2_387.89        94.18     2_482.07       0.1164          1.2717            1.2403         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)           2_387.89        95.26     2_483.15       0.1164          1.2717            1.2403         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)           2_387.89       200.06     2_587.95       0.3175          1.0626            1.0468         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)           2_387.89       310.83     2_698.72       0.4096          1.0407            1.0290         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)          2_387.89       196.92     2_584.81       0.3175          1.0626            1.0468         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)          2_387.89       311.72     2_699.61       0.4096          1.0407            1.0290         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)          2_387.89       199.43     2_587.32       0.3175          1.0626            1.0468         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)          2_387.89       315.76     2_703.65       0.4096          1.0407            1.0290         2.74
IVF-Binary-256-nl158-random (self)                     2_387.89       403.72     2_791.61       0.3193          1.0593            1.0464         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)           1_440.83        93.24     1_534.07       0.1332          1.2302            1.1908         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)           1_440.83        95.59     1_536.42       0.1332          1.2302            1.1908         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)           1_440.83        98.01     1_538.84       0.1332          1.2302            1.1908         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)          1_440.83       206.28     1_647.11       0.3572          1.0474            1.0377         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)          1_440.83       318.31     1_759.14       0.4565          1.0309            1.0232         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)          1_440.83       205.05     1_645.88       0.3572          1.0474            1.0377         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)          1_440.83       320.15     1_760.98       0.4565          1.0309            1.0232         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)          1_440.83       207.51     1_648.34       0.3572          1.0474            1.0377         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)          1_440.83       322.33     1_763.16       0.4565          1.0309            1.0232         2.93
IVF-Binary-256-nl223-random (self)                     1_440.83       431.29     1_872.12       0.3589          1.0438            1.0373         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)           1_810.99       102.61     1_913.60       0.1402          1.2168            1.1748         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)           1_810.99       104.73     1_915.73       0.1402          1.2168            1.1748         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)           1_810.99       105.49     1_916.48       0.1402          1.2168            1.1748         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)          1_810.99       214.57     2_025.56       0.3656          1.0444            1.0361         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)          1_810.99       327.75     2_138.75       0.4629          1.0293            1.0225         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)          1_810.99       211.85     2_022.84       0.3656          1.0444            1.0361         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)          1_810.99       327.21     2_138.21       0.4628          1.0293            1.0225         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)          1_810.99       212.58     2_023.58       0.3656          1.0444            1.0361         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)          1_810.99       343.27     2_154.26       0.4628          1.0293            1.0225         3.21
IVF-Binary-256-nl316-random (self)                     1_810.99       462.81     2_273.80       0.3670          1.0410            1.0358         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)               2_558.88        84.58     2_643.47       0.1080          1.2909            1.2569         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)              2_558.88        88.04     2_646.92       0.1080          1.2909            1.2569         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)              2_558.88        88.11     2_647.00       0.1080          1.2909            1.2569         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)              2_558.88       188.55     2_747.43       0.3044          1.0707            1.0512         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)              2_558.88       318.68     2_877.57       0.3972          1.0447            1.0309         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)             2_558.88       194.02     2_752.90       0.3044          1.0707            1.0512         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)             2_558.88       305.70     2_864.59       0.3972          1.0447            1.0309         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)             2_558.88       190.72     2_749.61       0.3044          1.0707            1.0512         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)             2_558.88       307.51     2_866.39       0.3972          1.0447            1.0309         2.74
IVF-Binary-256-nl158-pca (self)                        2_558.88       383.94     2_942.82       0.3084          1.0672            1.0497         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)              1_638.22        92.76     1_730.99       0.1269          1.2393            1.2037         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)              1_638.22        95.50     1_733.72       0.1269          1.2393            1.2037         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)              1_638.22        98.98     1_737.20       0.1269          1.2393            1.2037         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)             1_638.22       204.38     1_842.60       0.3608          1.0475            1.0373         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)             1_638.22       319.30     1_957.52       0.4623          1.0301            1.0228         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)             1_638.22       201.88     1_840.11       0.3608          1.0475            1.0373         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)             1_638.22       318.93     1_957.15       0.4623          1.0301            1.0228         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)             1_638.22       208.28     1_846.51       0.3608          1.0475            1.0373         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)             1_638.22       327.06     1_965.29       0.4623          1.0301            1.0228         2.93
IVF-Binary-256-nl223-pca (self)                        1_638.22       411.12     2_049.34       0.3641          1.0449            1.0366         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)              2_052.47       104.17     2_156.65       0.1367          1.2212            1.1848         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)              2_052.47       102.72     2_155.20       0.1367          1.2212            1.1848         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)              2_052.47       105.03     2_157.50       0.1367          1.2212            1.1848         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)             2_052.47       210.98     2_263.45       0.3742          1.0434            1.0348         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)             2_052.47       333.04     2_385.51       0.4724          1.0282            1.0217         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)             2_052.47       207.47     2_259.95       0.3742          1.0434            1.0348         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)             2_052.47       324.91     2_377.38       0.4723          1.0283            1.0217         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)             2_052.47       211.32     2_263.80       0.3742          1.0434            1.0348         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)             2_052.47       328.76     2_381.24       0.4723          1.0283            1.0217         3.21
IVF-Binary-256-nl316-pca (self)                        2_052.47       448.71     2_501.18       0.3770          1.0409            1.0340         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)            2_455.22       123.19     2_578.40       0.1520          1.2060            1.1791         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)           2_455.22       124.03     2_579.25       0.1520          1.2060            1.1791         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)           2_455.22       127.63     2_582.84       0.1520          1.2060            1.1791         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)           2_455.22       236.52     2_691.73       0.3405          1.0447            1.0423         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)           2_455.22       360.07     2_815.28       0.4339          1.0290            1.0262         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)          2_455.22       240.15     2_695.37       0.3405          1.0447            1.0423         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)          2_455.22       365.84     2_821.06       0.4339          1.0290            1.0262         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)          2_455.22       242.05     2_697.27       0.3405          1.0447            1.0423         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)          2_455.22       357.07     2_812.29       0.4339          1.0290            1.0262         5.02
IVF-Binary-512-nl158-random (self)                     2_455.22       565.97     3_021.19       0.3409          1.0431            1.0421         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)           1_582.52       133.96     1_716.47       0.1603          1.1840            1.1576         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)           1_582.52       131.59     1_714.10       0.1603          1.1840            1.1576         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)           1_582.52       137.06     1_719.58       0.1603          1.1840            1.1576         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)          1_582.52       244.33     1_826.85       0.3578          1.0405            1.0380         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)          1_582.52       368.51     1_951.03       0.4564          1.0262            1.0235         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)          1_582.52       244.78     1_827.30       0.3578          1.0405            1.0380         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)          1_582.52       368.12     1_950.64       0.4564          1.0262            1.0235         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)          1_582.52       249.46     1_831.98       0.3578          1.0405            1.0380         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)          1_582.52       377.22     1_959.74       0.4564          1.0262            1.0235         5.21
IVF-Binary-512-nl223-random (self)                     1_582.52       587.33     2_169.84       0.3585          1.0389            1.0379         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)           1_924.45       140.52     2_064.97       0.1624          1.1788            1.1525         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)           1_924.45       140.65     2_065.11       0.1624          1.1789            1.1525         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)           1_924.45       164.39     2_088.84       0.1624          1.1789            1.1525         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)          1_924.45       255.26     2_179.71       0.3613          1.0391            1.0372         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)          1_924.45       381.94     2_306.39       0.4582          1.0253            1.0234         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)          1_924.45       252.91     2_177.36       0.3613          1.0392            1.0372         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)          1_924.45       394.10     2_318.56       0.4582          1.0254            1.0234         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)          1_924.45       260.23     2_184.68       0.3613          1.0392            1.0372         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)          1_924.45       387.30     2_311.75       0.4582          1.0254            1.0234         5.48
IVF-Binary-512-nl316-random (self)                     1_924.45       621.42     2_545.88       0.3616          1.0379            1.0373         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)               2_653.50       121.60     2_775.09       0.1476          1.2119            1.1892         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)              2_653.50       124.50     2_777.99       0.1476          1.2119            1.1892         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)              2_653.50       127.11     2_780.61       0.1476          1.2119            1.1892         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)              2_653.50       234.55     2_888.05       0.3359          1.0458            1.0431         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)              2_653.50       351.57     3_005.07       0.4299          1.0290            1.0267         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)             2_653.50       238.32     2_891.81       0.3359          1.0458            1.0431         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)             2_653.50       370.10     3_023.59       0.4299          1.0290            1.0267         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)             2_653.50       236.96     2_890.46       0.3359          1.0458            1.0431         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)             2_653.50       358.62     3_012.11       0.4299          1.0290            1.0267         5.02
IVF-Binary-512-nl158-pca (self)                        2_653.50       556.34     3_209.84       0.3374          1.0443            1.0428         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_736.26       131.84     1_868.11       0.1584          1.1837            1.1596         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_736.26       132.14     1_868.41       0.1584          1.1837            1.1596         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_736.26       135.41     1_871.67       0.1584          1.1837            1.1596         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_736.26       244.21     1_980.48       0.3583          1.0398            1.0379         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_736.26       362.15     2_098.41       0.4550          1.0259            1.0235         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_736.26       245.49     1_981.75       0.3583          1.0398            1.0379         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_736.26       366.72     2_102.98       0.4550          1.0259            1.0235         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_736.26       247.90     1_984.16       0.3583          1.0398            1.0379         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_736.26       369.81     2_106.07       0.4550          1.0259            1.0235         5.21
IVF-Binary-512-nl223-pca (self)                        1_736.26       607.45     2_343.72       0.3590          1.0390            1.0378         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)              2_146.09       139.83     2_285.92       0.1627          1.1763            1.1519         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)              2_146.09       139.68     2_285.77       0.1627          1.1763            1.1519         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)              2_146.09       143.61     2_289.70       0.1627          1.1763            1.1519         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)             2_146.09       253.65     2_399.73       0.3626          1.0388            1.0370         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)             2_146.09       376.23     2_522.31       0.4585          1.0255            1.0231         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)             2_146.09       252.22     2_398.30       0.3626          1.0389            1.0370         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)             2_146.09       375.85     2_521.94       0.4584          1.0255            1.0231         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)             2_146.09       258.40     2_404.49       0.3626          1.0389            1.0370         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)             2_146.09       384.27     2_530.36       0.4584          1.0255            1.0231         5.48
IVF-Binary-512-nl316-pca (self)                        2_146.09       618.78     2_764.87       0.3635          1.0382            1.0370         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)           2_700.19       196.50     2_896.69       0.1768          1.1661            1.1563         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)          2_700.19       198.14     2_898.34       0.1768          1.1661            1.1563         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)          2_700.19       200.10     2_900.29       0.1768          1.1661            1.1563         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)          2_700.19       345.64     3_045.84       0.3609          1.0381            1.0382         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)          2_700.19       443.47     3_143.66       0.4626          1.0243            1.0230         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)         2_700.19       320.53     3_020.72       0.3609          1.0381            1.0382         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)         2_700.19       457.01     3_157.21       0.4626          1.0243            1.0230         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)         2_700.19       331.51     3_031.70       0.3609          1.0381            1.0382         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)         2_700.19       463.08     3_163.27       0.4626          1.0243            1.0230         9.57
IVF-Binary-1024-nl158-random (self)                    2_700.19       856.26     3_556.45       0.3608          1.0375            1.0382         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_759.67       207.64     1_967.31       0.1797          1.1568            1.1474         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_759.67       215.10     1_974.77       0.1797          1.1568            1.1474         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_759.67       212.09     1_971.76       0.1797          1.1568            1.1474         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_759.67       344.82     2_104.49       0.3717          1.0360            1.0360         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_759.67       462.66     2_222.33       0.4749          1.0229            1.0217         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_759.67       339.80     2_099.47       0.3717          1.0360            1.0360         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_759.67       466.19     2_225.86       0.4749          1.0229            1.0217         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_759.67       348.15     2_107.82       0.3717          1.0360            1.0360         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_759.67       478.40     2_238.06       0.4749          1.0229            1.0217         9.76
IVF-Binary-1024-nl223-random (self)                    1_759.67       891.15     2_650.82       0.3714          1.0355            1.0360         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)          2_139.31       221.06     2_360.37       0.1804          1.1544            1.1451        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)          2_139.31       220.57     2_359.88       0.1804          1.1545            1.1451        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)          2_139.31       228.44     2_367.75       0.1804          1.1545            1.1451        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)         2_139.31       358.63     2_497.94       0.3732          1.0356            1.0359        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)         2_139.31       480.64     2_619.95       0.4755          1.0227            1.0217        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)         2_139.31       349.73     2_489.04       0.3732          1.0356            1.0359        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)         2_139.31       486.91     2_626.22       0.4755          1.0228            1.0217        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)         2_139.31       361.00     2_500.30       0.3732          1.0356            1.0359        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)         2_139.31       489.87     2_629.18       0.4755          1.0228            1.0217        10.04
IVF-Binary-1024-nl316-random (self)                    2_139.31       936.24     3_075.55       0.3730          1.0351            1.0357        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)              2_878.16       199.90     3_078.06       0.1762          1.1673            1.1573         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)             2_878.16       202.02     3_080.18       0.1762          1.1673            1.1573         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)             2_878.16       202.17     3_080.33       0.1762          1.1673            1.1573         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)             2_878.16       313.01     3_191.17       0.3600          1.0380            1.0384         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)             2_878.16       442.74     3_320.90       0.4585          1.0245            1.0237         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)            2_878.16       317.18     3_195.34       0.3600          1.0380            1.0384         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)            2_878.16       455.08     3_333.24       0.4585          1.0245            1.0237         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)            2_878.16       332.73     3_210.89       0.3600          1.0380            1.0384         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)            2_878.16       457.91     3_336.07       0.4585          1.0245            1.0237         9.57
IVF-Binary-1024-nl158-pca (self)                       2_878.16       855.86     3_734.02       0.3591          1.0381            1.0388         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_984.76       206.47     2_191.23       0.1794          1.1567            1.1470         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_984.76       207.08     2_191.84       0.1794          1.1567            1.1470         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_984.76       211.39     2_196.15       0.1794          1.1567            1.1470         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_984.76       330.33     2_315.09       0.3709          1.0358            1.0361         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_984.76       460.84     2_445.60       0.4723          1.0231            1.0221         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_984.76       332.72     2_317.47       0.3709          1.0358            1.0361         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_984.76       466.25     2_451.01       0.4723          1.0231            1.0221         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_984.76       341.91     2_326.67       0.3709          1.0358            1.0361         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_984.76       475.26     2_460.01       0.4723          1.0231            1.0221         9.76
IVF-Binary-1024-nl223-pca (self)                       1_984.76       881.26     2_866.02       0.3702          1.0359            1.0363         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             2_342.32       225.56     2_567.88       0.1805          1.1540            1.1447        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)             2_342.32       218.14     2_560.46       0.1805          1.1540            1.1447        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)             2_342.32       220.51     2_562.83       0.1805          1.1540            1.1447        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)            2_342.32       344.09     2_686.41       0.3730          1.0355            1.0357        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)            2_342.32       472.45     2_814.77       0.4735          1.0228            1.0220        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)            2_342.32       344.06     2_686.38       0.3730          1.0355            1.0357        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)            2_342.32       478.08     2_820.40       0.4735          1.0228            1.0220        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)            2_342.32       354.82     2_697.14       0.3730          1.0355            1.0357        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)            2_342.32       489.80     2_832.12       0.4735          1.0228            1.0220        10.04
IVF-Binary-1024-nl316-pca (self)                       2_342.32       919.90     3_262.22       0.3720          1.0356            1.0360        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)              2_419.49       488.23     2_907.72       0.1693          1.1869            1.1720         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)             2_419.49       414.14     2_833.63       0.1693          1.1869            1.1720         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)             2_419.49       415.92     2_835.41       0.1693          1.1869            1.1720         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)             2_419.49       501.08     2_920.58       0.3435          1.0431            1.0414         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)             2_419.49       912.97     3_332.46       0.4441          1.0265            1.0249         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)            2_419.49       504.81     2_924.30       0.3435          1.0431            1.0414         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)            2_419.49       923.31     3_342.81       0.4441          1.0265            1.0249         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)            2_419.49       512.80     2_932.29       0.3435          1.0431            1.0414         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)            2_419.49       921.99     3_341.48       0.4441          1.0265            1.0249         5.04
IVF-Binary-768-nl158-sign (self)                       2_419.49     1_435.62     3_855.12       0.3441          1.0422            1.0413         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)             1_388.78       412.75     1_801.53       0.1694          1.1869            1.1715         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)             1_388.78       424.52     1_813.30       0.1694          1.1869            1.1715         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)             1_388.78       420.28     1_809.06       0.1694          1.1869            1.1715         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)            1_388.78       508.10     1_896.88       0.3491          1.0415            1.0400         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)            1_388.78       928.31     2_317.09       0.4485          1.0259            1.0244         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)            1_388.78       516.10     1_904.88       0.3491          1.0415            1.0400         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)            1_388.78       924.04     2_312.82       0.4485          1.0259            1.0244         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)            1_388.78       513.36     1_902.14       0.3491          1.0415            1.0400         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)            1_388.78       939.68     2_328.46       0.4485          1.0259            1.0244         5.23
IVF-Binary-768-nl223-sign (self)                       1_388.78     1_447.29     2_836.08       0.3494          1.0408            1.0400         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)             1_763.13       433.24     2_196.38       0.1695          1.1865            1.1713         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)             1_763.13       442.97     2_206.11       0.1695          1.1866            1.1713         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)             1_763.13       433.67     2_196.81       0.1695          1.1866            1.1713         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)            1_763.13       519.61     2_282.74       0.3496          1.0411            1.0400         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)            1_763.13       928.82     2_691.96       0.4487          1.0257            1.0245         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)            1_763.13       519.90     2_283.03       0.3495          1.0411            1.0400         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)            1_763.13       942.62     2_705.76       0.4486          1.0257            1.0245         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)            1_763.13       526.98     2_290.12       0.3495          1.0411            1.0400         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)            1_763.13       945.47     2_708.61       0.4486          1.0257            1.0245         5.51
IVF-Binary-768-nl316-sign (self)                       1_763.13     1_478.87     3_242.00       0.3501          1.0405            1.0398         5.51
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
Exhaustive (query)                                        34.51       757.73       792.24       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         34.51     2_635.34     2_669.85       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                 81.16       244.90       326.06       0.0970          1.6334            1.6378         1.78
ExhaustiveBinary-256-random-rf10 (query)                  81.16       353.15       434.31       0.3643          1.1391            1.1302         1.78
ExhaustiveBinary-256-random-rf20 (query)                  81.16       481.45       562.61       0.5087          1.0798            1.0701         1.78
ExhaustiveBinary-256-random (self)                        81.16     1_144.79     1_225.95       0.3862          1.1443            1.1409         1.78
ExhaustiveBinary-256-pca_no_rr (query)                   103.70       249.84       353.55       0.0922          1.6524            1.6606         1.78
ExhaustiveBinary-256-pca-rf10 (query)                    103.70       374.52       478.22       0.3517          1.1465            1.1384         1.78
ExhaustiveBinary-256-pca-rf20 (query)                    103.70       466.54       570.24       0.4943          1.0846            1.0744         1.78
ExhaustiveBinary-256-pca (self)                          103.70     1_154.23     1_257.93       0.3764          1.1502            1.1477         1.78
ExhaustiveBinary-512-random_no_rr (query)                 87.10       362.37       449.47       0.1464          1.5035            1.5085         3.55
ExhaustiveBinary-512-random-rf10 (query)                  87.10       476.28       563.38       0.4596          1.0936            1.0901         3.55
ExhaustiveBinary-512-random-rf20 (query)                  87.10       591.99       679.09       0.6085          1.0504            1.0459         3.55
ExhaustiveBinary-512-random (self)                        87.10     1_588.06     1_675.16       0.4800          1.0996            1.0995         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   121.56       370.44       492.01       0.1458          1.5041            1.5097         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    121.56       501.83       623.39       0.4543          1.0952            1.0911         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    121.56       585.68       707.24       0.6037          1.0513            1.0464         3.55
ExhaustiveBinary-512-pca (self)                          121.56     1_583.16     1_704.72       0.4777          1.1003            1.0997         3.55
ExhaustiveBinary-1024-random_no_rr (query)               118.03       543.94       661.98       0.2155          1.3655            1.3721         7.10
ExhaustiveBinary-1024-random-rf10 (query)                118.03       670.28       788.32       0.5869          1.0540            1.0515         7.10
ExhaustiveBinary-1024-random-rf20 (query)                118.03       772.32       890.35       0.7380          1.0260            1.0224         7.10
ExhaustiveBinary-1024-random (self)                      118.03     2_203.10     2_321.13       0.6118          1.0576            1.0546         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  151.50       545.23       696.74       0.2122          1.3735            1.3798         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   151.50       673.09       824.59       0.5776          1.0560            1.0532         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   151.50       781.10       932.61       0.7291          1.0273            1.0232         7.10
ExhaustiveBinary-1024-pca (self)                         151.50     2_194.89     2_346.40       0.6017          1.0602            1.0571         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   42.16       462.53       504.69       0.1044          1.6421            1.6499         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    42.16       497.91       540.08       0.3737          1.1368            1.1275         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    42.16       757.58       799.74       0.5265          1.0745            1.0646         1.53
ExhaustiveBinary-256-sign (self)                          42.16     1_600.90     1_643.06       0.3940          1.1439            1.1394         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)              950.55        51.94     1_002.49       0.1016          1.6177            1.6285         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)             950.55        59.98     1_010.53       0.1016          1.6179            1.6286         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)             950.55        57.30     1_007.85       0.1016          1.6179            1.6286         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)             950.55       106.78     1_057.32       0.3750          1.1333            1.1268         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)             950.55       161.28     1_111.83       0.5191          1.0759            1.0681         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)            950.55       102.72     1_053.27       0.3742          1.1334            1.1269         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)            950.55       188.19     1_138.74       0.5182          1.0761            1.0681         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)            950.55       106.96     1_057.51       0.3742          1.1335            1.1269         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)            950.55       162.86     1_113.41       0.5181          1.0761            1.0681         1.93
IVF-Binary-256-nl158-random (self)                       950.55       240.38     1_190.93       0.3960          1.1379            1.1379         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             663.93        47.07       711.01       0.1120          1.5877            1.5867         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             663.93        58.67       722.61       0.1119          1.5883            1.5872         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             663.93        53.68       717.62       0.1119          1.5884            1.5874         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            663.93       105.78       769.72       0.3934          1.1239            1.1158         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            663.93       159.90       823.84       0.5355          1.0713            1.0628         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            663.93       104.15       768.08       0.3930          1.1240            1.1160         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            663.93       157.08       821.01       0.5350          1.0714            1.0628         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            663.93       113.12       777.06       0.3929          1.1240            1.1160         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            663.93       161.66       825.59       0.5349          1.0714            1.0628         2.00
IVF-Binary-256-nl223-random (self)                       663.93       243.07       907.00       0.4141          1.1287            1.1268         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             871.14        53.05       924.19       0.1175          1.5678            1.5685         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             871.14        49.35       920.49       0.1174          1.5683            1.5691         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             871.14        54.89       926.03       0.1174          1.5686            1.5693         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            871.14       116.14       987.28       0.4048          1.1173            1.1103         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            871.14       160.32     1_031.46       0.5480          1.0673            1.0599         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            871.14       106.76       977.90       0.4042          1.1175            1.1108         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            871.14       163.21     1_034.35       0.5471          1.0675            1.0601         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            871.14       110.94       982.07       0.4041          1.1175            1.1108         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            871.14       162.43     1_033.57       0.5470          1.0675            1.0601         2.09
IVF-Binary-256-nl316-random (self)                       871.14       248.92     1_120.06       0.4254          1.1215            1.1216         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)                 955.83        43.61       999.43       0.0972          1.6371            1.6509         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)                955.83        45.05     1_000.88       0.0972          1.6373            1.6510         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)                955.83        48.08     1_003.91       0.0972          1.6373            1.6510         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)                955.83       101.18     1_057.01       0.3614          1.1414            1.1351         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)                955.83       160.73     1_116.56       0.5035          1.0811            1.0727         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)               955.83       102.21     1_058.03       0.3609          1.1415            1.1351         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)               955.83       153.40     1_109.22       0.5028          1.0812            1.0727         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)               955.83       105.14     1_060.97       0.3609          1.1415            1.1351         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)               955.83       157.29     1_113.11       0.5028          1.0812            1.0727         1.93
IVF-Binary-256-nl158-pca (self)                          955.83       236.17     1_192.00       0.3857          1.1449            1.1445         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                699.36        45.77       745.12       0.1097          1.5958            1.5999         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                699.36        47.87       747.23       0.1096          1.5968            1.6008         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                699.36        51.95       751.31       0.1096          1.5969            1.6009         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               699.36       104.58       803.94       0.3870          1.1269            1.1204         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               699.36       160.81       860.16       0.5267          1.0735            1.0656         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               699.36       108.65       808.01       0.3865          1.1271            1.1205         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               699.36       156.27       855.62       0.5258          1.0737            1.0659         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               699.36       118.90       818.26       0.3864          1.1271            1.1205         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               699.36       162.90       862.26       0.5258          1.0737            1.0659         2.00
IVF-Binary-256-nl223-pca (self)                          699.36       240.40       939.76       0.4102          1.1301            1.1303         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                890.65        49.16       939.81       0.1159          1.5754            1.5785         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                890.65        51.91       942.56       0.1158          1.5761            1.5790         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                890.65        53.33       943.99       0.1157          1.5767            1.5793         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               890.65       107.17       997.82       0.3971          1.1214            1.1151         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               890.65       163.61     1_054.26       0.5370          1.0701            1.0626         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               890.65       112.31     1_002.96       0.3965          1.1217            1.1154         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               890.65       156.39     1_047.04       0.5362          1.0703            1.0628         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               890.65       111.85     1_002.50       0.3963          1.1218            1.1155         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               890.65       169.26     1_059.91       0.5360          1.0704            1.0629         2.09
IVF-Binary-256-nl316-pca (self)                          890.65       252.73     1_143.38       0.4200          1.1246            1.1253         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)              948.52        62.09     1_010.62       0.1496          1.4966            1.5039         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)             948.52        73.59     1_022.11       0.1496          1.4966            1.5039         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)             948.52        69.35     1_017.87       0.1496          1.4966            1.5039         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)             948.52       124.69     1_073.21       0.4642          1.0917            1.0890         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)             948.52       177.19     1_125.71       0.6126          1.0494            1.0452         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)            948.52       125.66     1_074.18       0.4641          1.0917            1.0890         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)            948.52       199.31     1_147.83       0.6125          1.0494            1.0452         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)            948.52       135.59     1_084.12       0.4641          1.0917            1.0890         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)            948.52       185.29     1_133.81       0.6125          1.0494            1.0452         3.71
IVF-Binary-512-nl158-random (self)                       948.52       311.02     1_259.54       0.4841          1.0980            1.0982         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             689.10        64.78       753.88       0.1563          1.4801            1.4855         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             689.10        67.36       756.46       0.1562          1.4803            1.4857         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             689.10        75.21       764.31       0.1562          1.4803            1.4857         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            689.10       126.03       815.13       0.4750          1.0879            1.0850         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            689.10       175.99       865.09       0.6213          1.0475            1.0433         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            689.10       128.24       817.35       0.4748          1.0879            1.0851         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            689.10       179.52       868.62       0.6210          1.0476            1.0433         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            689.10       133.40       822.50       0.4748          1.0880            1.0851         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            689.10       191.22       880.32       0.6210          1.0476            1.0433         3.77
IVF-Binary-512-nl223-random (self)                       689.10       320.48     1_009.59       0.4945          1.0941            1.0938         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             908.13        67.23       975.37       0.1594          1.4708            1.4746         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             908.13        69.79       977.93       0.1594          1.4711            1.4747         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             908.13        73.52       981.65       0.1594          1.4711            1.4747         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            908.13       132.44     1_040.57       0.4807          1.0857            1.0829         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            908.13       193.70     1_101.84       0.6271          1.0463            1.0423         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            908.13       130.78     1_038.92       0.4803          1.0858            1.0830         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            908.13       185.28     1_093.41       0.6266          1.0464            1.0425         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            908.13       136.82     1_044.95       0.4803          1.0859            1.0831         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            908.13       194.01     1_102.14       0.6265          1.0464            1.0425         3.86
IVF-Binary-512-nl316-random (self)                       908.13       326.49     1_234.63       0.4996          1.0921            1.0915         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               1_049.21        58.57     1_107.78       0.1487          1.4983            1.5052         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              1_049.21        62.20     1_111.41       0.1487          1.4983            1.5052         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              1_049.21        67.80     1_117.02       0.1487          1.4983            1.5052         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              1_049.21       121.33     1_170.55       0.4587          1.0935            1.0902         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              1_049.21       173.61     1_222.83       0.6076          1.0502            1.0457         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             1_049.21       126.58     1_175.80       0.4586          1.0935            1.0902         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             1_049.21       191.28     1_240.49       0.6074          1.0503            1.0457         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             1_049.21       132.32     1_181.54       0.4586          1.0935            1.0902         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             1_049.21       190.34     1_239.56       0.6074          1.0503            1.0457         3.71
IVF-Binary-512-nl158-pca (self)                        1_049.21       310.30     1_359.52       0.4816          1.0989            1.0987         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                722.71        64.24       786.95       0.1556          1.4810            1.4845         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                722.71        66.21       788.92       0.1555          1.4814            1.4852         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                722.71        73.53       796.23       0.1555          1.4814            1.4853         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               722.71       127.81       850.51       0.4716          1.0888            1.0853         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               722.71       194.25       916.95       0.6190          1.0478            1.0432         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               722.71       127.80       850.50       0.4712          1.0890            1.0855         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               722.71       180.62       903.33       0.6185          1.0479            1.0432         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               722.71       135.61       858.32       0.4712          1.0890            1.0855         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               722.71       186.66       909.37       0.6185          1.0479            1.0432         3.77
IVF-Binary-512-nl223-pca (self)                          722.71       321.32     1_044.03       0.4934          1.0944            1.0939         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                925.79        68.66       994.45       0.1591          1.4714            1.4751         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                925.79        67.71       993.51       0.1590          1.4717            1.4756         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                925.79        72.47       998.27       0.1590          1.4719            1.4757         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               925.79       128.61     1_054.41       0.4769          1.0868            1.0833         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               925.79       179.74     1_105.53       0.6238          1.0467            1.0422         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               925.79       129.41     1_055.20       0.4764          1.0870            1.0835         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               925.79       185.28     1_111.08       0.6231          1.0468            1.0424         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               925.79       140.72     1_066.51       0.4764          1.0870            1.0835         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               925.79       196.27     1_122.06       0.6230          1.0469            1.0424         3.86
IVF-Binary-512-nl316-pca (self)                          925.79       335.52     1_261.32       0.4985          1.0924            1.0920         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)           1_001.65        93.96     1_095.61       0.2170          1.3632            1.3702         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)          1_001.65        98.75     1_100.41       0.2170          1.3632            1.3702         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)          1_001.65       105.38     1_107.03       0.2170          1.3632            1.3702         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)          1_001.65       162.46     1_164.11       0.5886          1.0535            1.0510         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)          1_001.65       215.63     1_217.28       0.7396          1.0257            1.0221         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)         1_001.65       164.48     1_166.13       0.5886          1.0535            1.0510         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)         1_001.65       218.75     1_220.40       0.7395          1.0257            1.0221         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)         1_001.65       171.13     1_172.78       0.5886          1.0535            1.0510         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)         1_001.65       236.25     1_237.90       0.7395          1.0257            1.0221         7.26
IVF-Binary-1024-nl158-random (self)                    1_001.65       444.64     1_446.29       0.6136          1.0570            1.0542         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            710.33        96.65       806.99       0.2204          1.3573            1.3647         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            710.33       104.43       814.77       0.2204          1.3574            1.3649         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            710.33       109.66       820.00       0.2204          1.3574            1.3649         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           710.33       161.11       871.44       0.5938          1.0524            1.0499         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           710.33       216.06       926.39       0.7439          1.0252            1.0215         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           710.33       163.28       873.62       0.5936          1.0525            1.0499         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           710.33       219.07       929.41       0.7437          1.0252            1.0215         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           710.33       179.83       890.17       0.5936          1.0525            1.0499         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           710.33       233.20       943.54       0.7437          1.0252            1.0215         7.32
IVF-Binary-1024-nl223-random (self)                      710.33       491.43     1_201.76       0.6185          1.0559            1.0530         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            914.94       102.20     1_017.14       0.2220          1.3537            1.3607         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)            914.94       103.83     1_018.77       0.2219          1.3538            1.3609         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)            914.94       110.47     1_025.40       0.2219          1.3538            1.3609         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)           914.94       179.48     1_094.42       0.5965          1.0517            1.0493         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)           914.94       218.19     1_133.12       0.7465          1.0248            1.0210         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)           914.94       166.48     1_081.42       0.5961          1.0518            1.0493         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)           914.94       228.22     1_143.16       0.7462          1.0248            1.0211         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)           914.94       170.84     1_085.77       0.5961          1.0518            1.0493         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)           914.94       235.38     1_150.32       0.7461          1.0248            1.0211         7.42
IVF-Binary-1024-nl316-random (self)                      914.94       456.74     1_371.68       0.6212          1.0551            1.0523         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)              1_030.73        91.90     1_122.63       0.2134          1.3713            1.3785         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)             1_030.73        97.46     1_128.19       0.2134          1.3713            1.3785         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)             1_030.73       134.69     1_165.42       0.2134          1.3713            1.3785         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)             1_030.73       161.11     1_191.84       0.5796          1.0555            1.0526         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)             1_030.73       211.66     1_242.39       0.7307          1.0270            1.0231         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)            1_030.73       160.59     1_191.32       0.5796          1.0555            1.0526         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)            1_030.73       218.37     1_249.10       0.7306          1.0270            1.0231         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)            1_030.73       175.35     1_206.08       0.5796          1.0555            1.0526         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)            1_030.73       240.82     1_271.55       0.7306          1.0270            1.0231         7.26
IVF-Binary-1024-nl158-pca (self)                       1_030.73       476.84     1_507.57       0.6034          1.0597            1.0567         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               724.49        97.19       821.68       0.2175          1.3639            1.3706         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               724.49        99.53       824.01       0.2175          1.3640            1.3708         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               724.49       115.38       839.87       0.2175          1.3640            1.3708         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              724.49       185.36       909.85       0.5850          1.0542            1.0513         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              724.49       216.11       940.60       0.7356          1.0263            1.0224         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              724.49       179.06       903.55       0.5848          1.0543            1.0513         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              724.49       219.73       944.21       0.7353          1.0264            1.0224         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              724.49       176.92       901.41       0.5847          1.0543            1.0513         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              724.49       228.78       953.27       0.7352          1.0264            1.0224         7.32
IVF-Binary-1024-nl223-pca (self)                         724.49       461.68     1_186.17       0.6093          1.0581            1.0551         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               938.95       101.99     1_040.94       0.2188          1.3604            1.3660         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)               938.95       101.30     1_040.25       0.2187          1.3606            1.3663         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)               938.95       114.26     1_053.21       0.2187          1.3606            1.3663         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)              938.95       162.13     1_101.08       0.5881          1.0534            1.0504         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)              938.95       225.61     1_164.56       0.7382          1.0260            1.0220         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)              938.95       169.29     1_108.24       0.5877          1.0535            1.0505         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)              938.95       234.21     1_173.16       0.7378          1.0260            1.0221         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)              938.95       176.04     1_114.99       0.5877          1.0535            1.0505         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)              938.95       227.67     1_166.62       0.7377          1.0260            1.0221         7.42
IVF-Binary-1024-nl316-pca (self)                         938.95       460.87     1_399.82       0.6116          1.0575            1.0543         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)                926.75       156.83     1_083.57       0.1043          1.6446            1.6491         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               926.75       158.44     1_085.19       0.1043          1.6447            1.6493         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               926.75       163.33     1_090.07       0.1043          1.6447            1.6493         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               926.75       205.07     1_131.81       0.3792          1.1342            1.1256         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               926.75       344.21     1_270.95       0.5308          1.0735            1.0642         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              926.75       196.45     1_123.20       0.3782          1.1344            1.1257         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              926.75       358.11     1_284.85       0.5303          1.0736            1.0642         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              926.75       201.43     1_128.17       0.3782          1.1344            1.1257         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              926.75       352.77     1_279.51       0.5302          1.0736            1.0642         1.68
IVF-Binary-256-nl158-sign (self)                         926.75       545.40     1_472.15       0.3978          1.1422            1.1381         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               633.36       159.78       793.14       0.1047          1.6393            1.6468         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               633.36       156.39       789.75       0.1045          1.6406            1.6480         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               633.36       167.00       800.36       0.1045          1.6410            1.6482         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              633.36       211.17       844.54       0.3864          1.1295            1.1209         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              633.36       352.82       986.18       0.5362          1.0716            1.0627         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              633.36       200.05       833.41       0.3860          1.1297            1.1212         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              633.36       357.04       990.40       0.5354          1.0718            1.0628         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              633.36       207.06       840.42       0.3858          1.1298            1.1212         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              633.36       358.54       991.90       0.5352          1.0718            1.0628         1.75
IVF-Binary-256-nl223-sign (self)                         633.36       550.31     1_183.67       0.4052          1.1380            1.1337         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               868.60       165.37     1_033.97       0.1054          1.6361            1.6424         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               868.60       164.41     1_033.01       0.1053          1.6368            1.6429         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               868.60       165.90     1_034.50       0.1053          1.6375            1.6439         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              868.60       214.43     1_083.03       0.3916          1.1267            1.1189         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              868.60       372.59     1_241.19       0.5390          1.0704            1.0622         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              868.60       215.85     1_084.45       0.3911          1.1270            1.1192         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              868.60       353.79     1_222.39       0.5379          1.0707            1.0623         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              868.60       213.42     1_082.02       0.3909          1.1270            1.1192         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              868.60       383.91     1_252.51       0.5377          1.0708            1.0624         1.84
IVF-Binary-256-nl316-sign (self)                         868.60       562.42     1_431.02       0.4104          1.1342            1.1317         1.84
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
Exhaustive (query)                                        68.41     1_344.07     1_412.49       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.41     4_526.92     4_595.34       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                132.09       266.87       398.96       0.0733          1.4884            1.4968         2.03
ExhaustiveBinary-256-random-rf10 (query)                 132.09       388.85       520.94       0.2947          1.1327            1.1260         2.03
ExhaustiveBinary-256-random-rf20 (query)                 132.09       518.85       650.93       0.4174          1.0830            1.0716         2.03
ExhaustiveBinary-256-random (self)                       132.09     1_221.70     1_353.79       0.3150          1.1321            1.1268         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   216.94       259.96       476.91       0.0722          1.4941            1.5001         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    216.94       391.78       608.72       0.2934          1.1324            1.1267         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    216.94       523.28       740.23       0.4181          1.0813            1.0715         2.03
ExhaustiveBinary-256-pca (self)                          216.94     1_238.66     1_455.60       0.3112          1.1338            1.1298         2.03
ExhaustiveBinary-512-random_no_rr (query)                204.79       381.50       586.29       0.1110          1.4064            1.4153         4.05
ExhaustiveBinary-512-random-rf10 (query)                 204.79       530.57       735.36       0.3695          1.0935            1.0919         4.05
ExhaustiveBinary-512-random-rf20 (query)                 204.79       673.58       878.37       0.4981          1.0544            1.0517         4.05
ExhaustiveBinary-512-random (self)                       204.79     1_694.06     1_898.85       0.3856          1.0953            1.0998         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   299.57       388.39       687.96       0.1063          1.4156            1.4272         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    299.57       522.87       822.44       0.3574          1.0990            1.0958         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    299.57       656.70       956.27       0.4853          1.0582            1.0544         4.05
ExhaustiveBinary-512-pca (self)                          299.57     1_686.04     1_985.61       0.3753          1.0995            1.1040         4.05
ExhaustiveBinary-1024-random_no_rr (query)               256.74       600.12       856.85       0.1593          1.3242            1.3318         8.11
ExhaustiveBinary-1024-random-rf10 (query)                256.74       739.10       995.83       0.4456          1.0660            1.0678         8.11
ExhaustiveBinary-1024-random-rf20 (query)                256.74       894.77     1_151.50       0.5824          1.0370            1.0362         8.11
ExhaustiveBinary-1024-random (self)                      256.74     2_431.04     2_687.78       0.4595          1.0714            1.0740         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  359.77       601.48       961.25       0.1599          1.3236            1.3333         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   359.77       751.89     1_111.66       0.4446          1.0658            1.0680         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   359.77       893.38     1_253.15       0.5812          1.0370            1.0362         8.11
ExhaustiveBinary-1024-pca (self)                         359.77     2_476.01     2_835.77       0.4582          1.0716            1.0746         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   84.58       684.55       769.13       0.1292          1.3815            1.3877         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    84.58       754.14       838.72       0.3927          1.0844            1.0833         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    84.58     1_134.60     1_219.18       0.5336          1.0464            1.0444         3.05
ExhaustiveBinary-512-sign (self)                          84.58     2_404.67     2_489.25       0.4063          1.0885            1.0916         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)            1_681.23        82.14     1_763.37       0.0762          1.4798            1.4941         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           1_681.23        76.99     1_758.22       0.0762          1.4800            1.4942         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           1_681.23        81.66     1_762.89       0.0762          1.4800            1.4942         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           1_681.23       153.75     1_834.97       0.2982          1.1315            1.1256         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           1_681.23       239.02     1_920.24       0.4198          1.0821            1.0715         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          1_681.23       150.44     1_831.66       0.2974          1.1316            1.1256         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          1_681.23       241.46     1_922.69       0.4191          1.0822            1.0715         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          1_681.23       150.16     1_831.39       0.2974          1.1316            1.1256         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          1_681.23       242.67     1_923.90       0.4191          1.0822            1.0715         2.34
IVF-Binary-256-nl158-random (self)                     1_681.23       298.16     1_979.38       0.3173          1.1313            1.1266         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           1_113.72        73.35     1_187.07       0.0883          1.4512            1.4575         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)           1_113.72        74.36     1_188.08       0.0883          1.4514            1.4576         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)           1_113.72        77.49     1_191.21       0.0883          1.4514            1.4576         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)          1_113.72       175.85     1_289.57       0.3261          1.1150            1.1088         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)          1_113.72       250.58     1_364.30       0.4506          1.0704            1.0628         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)          1_113.72       157.00     1_270.72       0.3260          1.1151            1.1088         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)          1_113.72       255.26     1_368.98       0.4505          1.0704            1.0629         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)          1_113.72       162.42     1_276.14       0.3260          1.1151            1.1088         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)          1_113.72       253.43     1_367.15       0.4505          1.0704            1.0629         2.47
IVF-Binary-256-nl223-random (self)                     1_113.72       330.65     1_444.37       0.3442          1.1147            1.1139         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)           1_529.38        78.18     1_607.56       0.0951          1.4360            1.4383         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           1_529.38        79.23     1_608.61       0.0951          1.4361            1.4384         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           1_529.38        82.92     1_612.30       0.0951          1.4361            1.4384         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          1_529.38       175.19     1_704.58       0.3374          1.1080            1.1028         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          1_529.38       257.34     1_786.73       0.4655          1.0650            1.0593         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          1_529.38       164.50     1_693.89       0.3373          1.1080            1.1028         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          1_529.38       256.54     1_785.92       0.4654          1.0650            1.0593         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          1_529.38       170.00     1_699.39       0.3373          1.1080            1.1028         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          1_529.38       258.22     1_787.61       0.4654          1.0650            1.0593         2.65
IVF-Binary-256-nl316-random (self)                     1_529.38       355.32     1_884.70       0.3557          1.1069            1.1093         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               1_795.35        66.98     1_862.32       0.0751          1.4851            1.4978         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              1_795.35        67.76     1_863.10       0.0750          1.4853            1.4978         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              1_795.35        69.03     1_864.38       0.0750          1.4853            1.4979         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              1_795.35       158.79     1_954.13       0.2973          1.1312            1.1262         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              1_795.35       243.20     2_038.55       0.4206          1.0808            1.0713         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             1_795.35       149.29     1_944.64       0.2966          1.1313            1.1262         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             1_795.35       244.93     2_040.27       0.4199          1.0809            1.0713         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             1_795.35       152.45     1_947.79       0.2965          1.1313            1.1262         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             1_795.35       245.72     2_041.07       0.4199          1.0809            1.0713         2.34
IVF-Binary-256-nl158-pca (self)                        1_795.35       321.10     2_116.44       0.3141          1.1326            1.1295         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              1_173.09        72.41     1_245.50       0.0880          1.4534            1.4573         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              1_173.09        73.29     1_246.38       0.0879          1.4535            1.4574         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              1_173.09        78.94     1_252.03       0.0879          1.4535            1.4574         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             1_173.09       161.83     1_334.92       0.3296          1.1118            1.1063         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             1_173.09       250.36     1_423.45       0.4517          1.0691            1.0623         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             1_173.09       158.74     1_331.83       0.3295          1.1118            1.1063         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             1_173.09       252.67     1_425.75       0.4517          1.0691            1.0623         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             1_173.09       163.30     1_336.38       0.3295          1.1118            1.1063         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             1_173.09       255.43     1_428.51       0.4516          1.0691            1.0623         2.47
IVF-Binary-256-nl223-pca (self)                        1_173.09       323.22     1_496.30       0.3479          1.1103            1.1126         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              1_581.92        77.78     1_659.69       0.0950          1.4352            1.4379         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              1_581.92        78.45     1_660.37       0.0950          1.4353            1.4380         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              1_581.92        87.01     1_668.93       0.0950          1.4353            1.4380         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             1_581.92       167.94     1_749.86       0.3407          1.1054            1.0999         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             1_581.92       254.86     1_836.78       0.4613          1.0660            1.0596         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             1_581.92       164.51     1_746.43       0.3406          1.1054            1.0999         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             1_581.92       254.77     1_836.69       0.4612          1.0660            1.0596         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             1_581.92       168.35     1_750.27       0.3406          1.1054            1.0999         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             1_581.92       258.44     1_840.36       0.4612          1.0660            1.0596         2.65
IVF-Binary-256-nl316-pca (self)                        1_581.92       348.64     1_930.55       0.3588          1.1032            1.1073         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)            1_739.45        96.58     1_836.03       0.1123          1.4042            1.4145         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)           1_739.45        94.58     1_834.03       0.1123          1.4043            1.4145         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)           1_739.45        96.41     1_835.86       0.1123          1.4043            1.4145         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)           1_739.45       188.35     1_927.80       0.3703          1.0933            1.0918         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)           1_739.45       272.08     2_011.53       0.4988          1.0542            1.0516         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)          1_739.45       182.49     1_921.93       0.3701          1.0933            1.0918         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)          1_739.45       277.60     2_017.05       0.4986          1.0543            1.0516         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)          1_739.45       185.14     1_924.59       0.3701          1.0933            1.0918         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)          1_739.45       281.08     2_020.53       0.4986          1.0543            1.0516         4.36
IVF-Binary-512-nl158-random (self)                     1_739.45       422.14     2_161.59       0.3862          1.0951            1.0998         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)           1_172.74       100.89     1_273.63       0.1204          1.3874            1.3946         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)           1_172.74       100.69     1_273.43       0.1204          1.3875            1.3946         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)           1_172.74       106.50     1_279.24       0.1204          1.3875            1.3946         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)          1_172.74       190.01     1_362.75       0.3836          1.0877            1.0871         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)          1_172.74       283.18     1_455.92       0.5122          1.0510            1.0488         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)          1_172.74       191.49     1_364.23       0.3836          1.0877            1.0871         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)          1_172.74       287.32     1_460.06       0.5121          1.0510            1.0488         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)          1_172.74       195.67     1_368.41       0.3836          1.0877            1.0871         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)          1_172.74       286.73     1_459.47       0.5121          1.0510            1.0488         4.49
IVF-Binary-512-nl223-random (self)                     1_172.74       457.45     1_630.19       0.3990          1.0902            1.0949         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)           1_568.13       107.01     1_675.14       0.1243          1.3791            1.3837         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)           1_568.13       106.07     1_674.20       0.1243          1.3791            1.3837         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)           1_568.13       111.17     1_679.30       0.1243          1.3791            1.3837         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)          1_568.13       195.97     1_764.10       0.3890          1.0855            1.0847         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)          1_568.13       292.17     1_860.31       0.5158          1.0501            1.0480         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)          1_568.13       195.20     1_763.33       0.3890          1.0855            1.0847         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)          1_568.13       289.68     1_857.81       0.5158          1.0501            1.0480         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)          1_568.13       210.54     1_778.67       0.3890          1.0855            1.0847         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)          1_568.13       297.25     1_865.38       0.5158          1.0501            1.0480         4.67
IVF-Binary-512-nl316-random (self)                     1_568.13       478.81     2_046.94       0.4033          1.0884            1.0931         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)               1_837.32        91.04     1_928.37       0.1076          1.4133            1.4261         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)              1_837.32        94.18     1_931.51       0.1076          1.4134            1.4261         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)              1_837.32        96.27     1_933.60       0.1076          1.4134            1.4261         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)              1_837.32       182.07     2_019.40       0.3583          1.0988            1.0957         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)              1_837.32       275.03     2_112.35       0.4858          1.0581            1.0544         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)             1_837.32       184.01     2_021.34       0.3581          1.0988            1.0957         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)             1_837.32       278.09     2_115.41       0.4857          1.0581            1.0544         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)             1_837.32       188.23     2_025.55       0.3581          1.0988            1.0957         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)             1_837.32       279.94     2_117.26       0.4857          1.0581            1.0544         4.36
IVF-Binary-512-nl158-pca (self)                        1_837.32       420.70     2_258.03       0.3759          1.0993            1.1040         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_249.36        98.44     1_347.80       0.1168          1.3940            1.3999         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_249.36       103.59     1_352.95       0.1168          1.3940            1.4000         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_249.36       109.95     1_359.30       0.1168          1.3940            1.4000         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_249.36       190.75     1_440.11       0.3738          1.0918            1.0901         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_249.36       289.20     1_538.56       0.5017          1.0541            1.0515         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_249.36       189.67     1_439.03       0.3738          1.0918            1.0901         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_249.36       285.46     1_534.81       0.5017          1.0541            1.0515         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_249.36       201.53     1_450.88       0.3738          1.0918            1.0901         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_249.36       287.54     1_536.90       0.5017          1.0541            1.0515         4.49
IVF-Binary-512-nl223-pca (self)                        1_249.36       442.61     1_691.97       0.3906          1.0936            1.0983         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)              1_664.05       104.96     1_769.02       0.1210          1.3851            1.3902         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)              1_664.05       105.46     1_769.51       0.1210          1.3851            1.3902         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)              1_664.05       110.57     1_774.62       0.1210          1.3851            1.3902         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)             1_664.05       196.78     1_860.84       0.3801          1.0890            1.0877         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)             1_664.05       288.71     1_952.77       0.5059          1.0528            1.0503         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)             1_664.05       194.65     1_858.71       0.3800          1.0890            1.0877         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)             1_664.05       289.44     1_953.49       0.5059          1.0528            1.0503         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)             1_664.05       197.62     1_861.67       0.3800          1.0890            1.0877         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)             1_664.05       293.62     1_957.68       0.5059          1.0528            1.0503         4.67
IVF-Binary-512-nl316-pca (self)                        1_664.05       466.35     2_130.40       0.3956          1.0912            1.0960         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)           1_798.04       145.49     1_943.53       0.1598          1.3235            1.3316         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)          1_798.04       148.03     1_946.07       0.1598          1.3235            1.3316         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)          1_798.04       151.96     1_950.00       0.1598          1.3235            1.3316         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)          1_798.04       240.37     2_038.41       0.4458          1.0660            1.0678         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)          1_798.04       338.78     2_136.82       0.5826          1.0370            1.0362         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)         1_798.04       244.22     2_042.26       0.4458          1.0660            1.0678         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)         1_798.04       349.01     2_147.05       0.5826          1.0370            1.0362         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)         1_798.04       249.33     2_047.37       0.4458          1.0660            1.0678         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)         1_798.04       354.01     2_152.05       0.5826          1.0370            1.0362         8.42
IVF-Binary-1024-nl158-random (self)                    1_798.04       637.51     2_435.54       0.4597          1.0713            1.0740         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_214.59       155.00     1_369.59       0.1637          1.3164            1.3251         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_214.59       153.40     1_367.99       0.1637          1.3164            1.3251         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_214.59       161.11     1_375.70       0.1637          1.3164            1.3251         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_214.59       248.01     1_462.60       0.4532          1.0639            1.0658         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_214.59       349.31     1_563.90       0.5895          1.0358            1.0351         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_214.59       250.49     1_465.08       0.4532          1.0639            1.0658         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_214.59       370.75     1_585.35       0.5895          1.0358            1.0351         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_214.59       256.17     1_470.76       0.4532          1.0639            1.0658         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_214.59       365.36     1_579.95       0.5895          1.0358            1.0351         8.54
IVF-Binary-1024-nl223-random (self)                    1_214.59       659.30     1_873.89       0.4672          1.0692            1.0721         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)          1_636.10       160.94     1_797.04       0.1655          1.3134            1.3213         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)          1_636.10       168.95     1_805.05       0.1655          1.3134            1.3213         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)          1_636.10       165.44     1_801.54       0.1655          1.3134            1.3213         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)         1_636.10       255.29     1_891.39       0.4550          1.0633            1.0652         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)         1_636.10       359.97     1_996.07       0.5909          1.0355            1.0347         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)         1_636.10       257.32     1_893.42       0.4550          1.0633            1.0652         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)         1_636.10       372.54     2_008.64       0.5908          1.0355            1.0347         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)         1_636.10       266.88     1_902.98       0.4550          1.0633            1.0652         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)         1_636.10       372.57     2_008.67       0.5908          1.0355            1.0347         8.73
IVF-Binary-1024-nl316-random (self)                    1_636.10       682.03     2_318.12       0.4696          1.0685            1.0715         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)              1_890.86       146.40     2_037.26       0.1603          1.3231            1.3332         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)             1_890.86       148.71     2_039.57       0.1603          1.3231            1.3332         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)             1_890.86       151.58     2_042.44       0.1603          1.3231            1.3332         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)             1_890.86       236.35     2_127.21       0.4447          1.0658            1.0680         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)             1_890.86       336.98     2_227.84       0.5813          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)            1_890.86       239.61     2_130.47       0.4447          1.0658            1.0680         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)            1_890.86       350.72     2_241.58       0.5813          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)            1_890.86       247.01     2_137.87       0.4447          1.0658            1.0680         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)            1_890.86       350.74     2_241.60       0.5813          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-pca (self)                       1_890.86       642.46     2_533.32       0.4583          1.0716            1.0746         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_304.03       153.24     1_457.26       0.1643          1.3158            1.3263         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_304.03       156.20     1_460.23       0.1643          1.3159            1.3263         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_304.03       160.97     1_464.99       0.1643          1.3159            1.3263         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_304.03       249.50     1_553.53       0.4518          1.0637            1.0663         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_304.03       348.69     1_652.71       0.5877          1.0359            1.0354         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_304.03       247.01     1_551.04       0.4518          1.0637            1.0663         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_304.03       352.90     1_656.93       0.5877          1.0359            1.0354         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_304.03       258.10     1_562.12       0.4518          1.0637            1.0663         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_304.03       362.92     1_666.95       0.5877          1.0359            1.0354         8.54
IVF-Binary-1024-nl223-pca (self)                       1_304.03       661.51     1_965.54       0.4656          1.0695            1.0726         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_718.48       162.82     1_881.30       0.1657          1.3130            1.3236         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_718.48       161.43     1_879.91       0.1657          1.3131            1.3236         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_718.48       174.60     1_893.08       0.1657          1.3131            1.3236         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_718.48       252.08     1_970.56       0.4540          1.0631            1.0656         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_718.48       356.52     2_075.01       0.5897          1.0356            1.0351         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_718.48       255.03     1_973.51       0.4540          1.0631            1.0656         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_718.48       357.39     2_075.88       0.5897          1.0356            1.0351         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_718.48       265.92     1_984.41       0.4539          1.0631            1.0656         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_718.48       369.83     2_088.31       0.5897          1.0356            1.0351         8.73
IVF-Binary-1024-nl316-pca (self)                       1_718.48       687.71     2_406.19       0.4678          1.0688            1.0718         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)              1_631.68       287.28     1_918.96       0.1291          1.3814            1.3871         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)             1_631.68       293.38     1_925.06       0.1291          1.3814            1.3871         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)             1_631.68       297.40     1_929.09       0.1291          1.3814            1.3871         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)             1_631.68       366.33     1_998.01       0.3933          1.0843            1.0833         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)             1_631.68       661.83     2_293.51       0.5337          1.0464            1.0445         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)            1_631.68       362.51     1_994.19       0.3931          1.0843            1.0833         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)            1_631.68       647.31     2_278.99       0.5337          1.0464            1.0445         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)            1_631.68       369.15     2_000.83       0.3931          1.0843            1.0833         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)            1_631.68       657.30     2_288.98       0.5337          1.0464            1.0445         3.36
IVF-Binary-512-nl158-sign (self)                       1_631.68       995.38     2_627.06       0.4066          1.0885            1.0915         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)             1_043.95       291.26     1_335.21       0.1295          1.3811            1.3869         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)             1_043.95       290.82     1_334.77       0.1295          1.3812            1.3869         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)             1_043.95       298.94     1_342.90       0.1295          1.3812            1.3869         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)            1_043.95       363.19     1_407.14       0.3989          1.0819            1.0819         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)            1_043.95       655.09     1_699.04       0.5380          1.0455            1.0438         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)            1_043.95       366.55     1_410.50       0.3988          1.0819            1.0819         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)            1_043.95       651.88     1_695.83       0.5379          1.0455            1.0438         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)            1_043.95       369.07     1_413.02       0.3988          1.0819            1.0819         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)            1_043.95       679.20     1_723.15       0.5379          1.0455            1.0438         3.49
IVF-Binary-512-nl223-sign (self)                       1_043.95     1_002.86     2_046.82       0.4123          1.0865            1.0899         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)             1_463.66       299.11     1_762.77       0.1294          1.3809            1.3865         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)             1_463.66       311.20     1_774.86       0.1294          1.3809            1.3865         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)             1_463.66       319.61     1_783.27       0.1294          1.3809            1.3865         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)            1_463.66       372.48     1_836.14       0.4003          1.0814            1.0814         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)            1_463.66       657.56     2_121.22       0.5383          1.0453            1.0439         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)            1_463.66       370.13     1_833.80       0.4003          1.0814            1.0814         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)            1_463.66       666.23     2_129.89       0.5382          1.0453            1.0439         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)            1_463.66       374.98     1_838.64       0.4003          1.0814            1.0814         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)            1_463.66       664.75     2_128.41       0.5382          1.0453            1.0439         3.67
IVF-Binary-512-nl316-sign (self)                       1_463.66     1_020.11     2_483.77       0.4134          1.0860            1.0893         3.67
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
Exhaustive (query)                                       100.33     1_890.20     1_990.53       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.33     6_373.88     6_474.21       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                196.98       298.33       495.31       0.0662          1.3769            1.3786         2.28
ExhaustiveBinary-256-random-rf10 (query)                 196.98       421.84       618.83       0.2741          1.1112            1.1017         2.28
ExhaustiveBinary-256-random-rf20 (query)                 196.98       575.80       772.78       0.3877          1.0706            1.0590         2.28
ExhaustiveBinary-256-random (self)                       196.98     1_435.12     1_632.10       0.2874          1.1077            1.0994         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   385.12       287.65       672.77       0.0653          1.3793            1.3770         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    385.12       423.95       809.07       0.2701          1.1138            1.1018         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    385.12       571.74       956.86       0.3852          1.0726            1.0585         2.28
ExhaustiveBinary-256-pca (self)                          385.12     1_321.06     1_706.18       0.2820          1.1100            1.0990         2.28
ExhaustiveBinary-512-random_no_rr (query)                303.63       403.54       707.17       0.0934          1.3249            1.3316         4.55
ExhaustiveBinary-512-random-rf10 (query)                 303.63       563.82       867.45       0.3220          1.0860            1.0806         4.55
ExhaustiveBinary-512-random-rf20 (query)                 303.63       728.32     1_031.95       0.4345          1.0527            1.0485         4.55
ExhaustiveBinary-512-random (self)                       303.63     1_799.33     2_102.96       0.3344          1.0826            1.0842         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   493.04       404.26       897.31       0.0951          1.3226            1.3263         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    493.04       561.61     1_054.66       0.3246          1.0847            1.0788         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    493.04       726.43     1_219.47       0.4400          1.0517            1.0468         4.55
ExhaustiveBinary-512-pca (self)                          493.04     1_788.38     2_281.42       0.3361          1.0819            1.0825         4.55
ExhaustiveBinary-1024-random_no_rr (query)               497.35       632.66     1_130.01       0.1319          1.2685            1.2723         9.11
ExhaustiveBinary-1024-random-rf10 (query)                497.35       803.44     1_300.79       0.3742          1.0644            1.0666         9.11
ExhaustiveBinary-1024-random-rf20 (query)                497.35       994.68     1_492.03       0.4906          1.0386            1.0390         9.11
ExhaustiveBinary-1024-random (self)                      497.35     2_620.46     3_117.81       0.3823          1.0666            1.0715         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  701.62       644.34     1_345.96       0.1355          1.2623            1.2651         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   701.62       810.78     1_512.40       0.3804          1.0622            1.0643         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   701.62       982.71     1_684.33       0.4993          1.0369            1.0374         9.11
ExhaustiveBinary-1024-pca (self)                         701.62     2_647.94     3_349.56       0.3870          1.0651            1.0695         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  135.33       854.89       990.22       0.1284          1.2822            1.2821         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   135.33       952.01     1_087.34       0.3618          1.0706            1.0699         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   135.33     1_476.17     1_611.50       0.4847          1.0407            1.0395         4.58
ExhaustiveBinary-768-sign (self)                         135.33     3_063.78     3_199.12       0.3694          1.0716            1.0747         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)            2_383.84        91.69     2_475.53       0.0691          1.3703            1.3767         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)           2_383.84        98.64     2_482.48       0.0690          1.3706            1.3768         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)           2_383.84       100.98     2_484.82       0.0690          1.3706            1.3768         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)           2_383.84       193.20     2_577.04       0.2788          1.1100            1.1013         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)           2_383.84       307.34     2_691.18       0.3911          1.0698            1.0589         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)          2_383.84       193.44     2_577.28       0.2780          1.1101            1.1013         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)          2_383.84       303.87     2_687.71       0.3904          1.0699            1.0589         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)          2_383.84       191.41     2_575.25       0.2780          1.1101            1.1013         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)          2_383.84       308.61     2_692.45       0.3903          1.0699            1.0589         2.74
IVF-Binary-256-nl158-random (self)                     2_383.84       379.12     2_762.96       0.2914          1.1066            1.0992         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)           1_473.72        96.17     1_569.89       0.0782          1.3506            1.3552         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)           1_473.72        93.93     1_567.65       0.0782          1.3507            1.3552         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)           1_473.72        95.77     1_569.49       0.0782          1.3507            1.3552         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)          1_473.72       201.27     1_674.98       0.3036          1.0944            1.0863         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)          1_473.72       339.78     1_813.50       0.4176          1.0588            1.0517         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)          1_473.72       199.07     1_672.79       0.3036          1.0944            1.0863         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)          1_473.72       326.63     1_800.35       0.4175          1.0588            1.0517         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)          1_473.72       200.96     1_674.68       0.3036          1.0944            1.0863         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)          1_473.72       335.94     1_809.66       0.4175          1.0588            1.0517         2.93
IVF-Binary-256-nl223-random (self)                     1_473.72       418.71     1_892.43       0.3175          1.0893            1.0880         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)           2_088.91       101.33     2_190.25       0.0857          1.3373            1.3374         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)           2_088.91       104.67     2_193.59       0.0856          1.3375            1.3374         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)           2_088.91       104.74     2_193.65       0.0856          1.3375            1.3374         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)          2_088.91       213.35     2_302.26       0.3192          1.0859            1.0802         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)          2_088.91       330.45     2_419.37       0.4327          1.0540            1.0488         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)          2_088.91       210.08     2_298.99       0.3191          1.0859            1.0802         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)          2_088.91       325.66     2_414.57       0.4326          1.0540            1.0488         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)          2_088.91       210.83     2_299.75       0.3191          1.0859            1.0802         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)          2_088.91       328.31     2_417.22       0.4326          1.0540            1.0488         3.21
IVF-Binary-256-nl316-random (self)                     2_088.91       470.97     2_559.88       0.3327          1.0803            1.0828         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)               2_560.18        86.29     2_646.47       0.0681          1.3721            1.3752         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)              2_560.18        86.28     2_646.46       0.0681          1.3723            1.3752         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)              2_560.18        86.69     2_646.87       0.0681          1.3723            1.3752         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)              2_560.18       187.44     2_747.61       0.2754          1.1116            1.1012         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)              2_560.18       305.26     2_865.44       0.3894          1.0711            1.0583         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)             2_560.18       188.43     2_748.61       0.2745          1.1117            1.1012         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)             2_560.18       307.14     2_867.32       0.3886          1.0712            1.0583         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)             2_560.18       194.10     2_754.27       0.2745          1.1117            1.1012         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)             2_560.18       307.40     2_867.58       0.3885          1.0712            1.0583         2.74
IVF-Binary-256-nl158-pca (self)                        2_560.18       385.75     2_945.93       0.2864          1.1079            1.0986         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)              1_667.50        91.29     1_758.78       0.0775          1.3528            1.3548         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)              1_667.50        92.48     1_759.98       0.0775          1.3529            1.3548         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)              1_667.50        99.00     1_766.50       0.0775          1.3529            1.3548         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)             1_667.50       202.08     1_869.58       0.2987          1.0962            1.0873         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)             1_667.50       317.12     1_984.62       0.4131          1.0605            1.0523         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)             1_667.50       199.09     1_866.58       0.2986          1.0962            1.0873         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)             1_667.50       317.58     1_985.08       0.4130          1.0605            1.0523         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)             1_667.50       202.95     1_870.45       0.2986          1.0962            1.0873         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)             1_667.50       320.85     1_988.35       0.4129          1.0605            1.0523         2.93
IVF-Binary-256-nl223-pca (self)                        1_667.50       422.19     2_089.68       0.3109          1.0909            1.0885         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)              2_290.23       103.26     2_393.50       0.0850          1.3392            1.3373         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)              2_290.23       102.95     2_393.18       0.0850          1.3393            1.3374         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)              2_290.23       105.28     2_395.52       0.0850          1.3393            1.3374         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)             2_290.23       211.87     2_502.10       0.3123          1.0891            1.0816         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)             2_290.23       337.39     2_627.62       0.4271          1.0565            1.0494         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)             2_290.23       212.86     2_503.10       0.3122          1.0892            1.0816         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)             2_290.23       329.42     2_619.65       0.4270          1.0565            1.0494         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)             2_290.23       214.39     2_504.63       0.3122          1.0892            1.0816         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)             2_290.23       330.57     2_620.80       0.4270          1.0565            1.0494         3.21
IVF-Binary-256-nl316-pca (self)                        2_290.23       477.60     2_767.84       0.3237          1.0835            1.0840         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)            2_476.53       119.59     2_596.11       0.0948          1.3226            1.3308         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)           2_476.53       122.07     2_598.60       0.0948          1.3227            1.3308         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)           2_476.53       124.76     2_601.29       0.0948          1.3227            1.3308         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)           2_476.53       230.47     2_706.99       0.3238          1.0856            1.0806         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)           2_476.53       347.03     2_823.56       0.4355          1.0525            1.0484         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)          2_476.53       231.95     2_708.48       0.3235          1.0857            1.0806         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)          2_476.53       351.35     2_827.87       0.4355          1.0525            1.0484         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)          2_476.53       236.75     2_713.28       0.3235          1.0857            1.0806         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)          2_476.53       353.43     2_829.96       0.4355          1.0525            1.0484         5.02
IVF-Binary-512-nl158-random (self)                     2_476.53       554.40     3_030.93       0.3360          1.0822            1.0842         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)           1_561.27       130.00     1_691.28       0.1037          1.3076            1.3082         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)           1_561.27       130.68     1_691.96       0.1037          1.3076            1.3082         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)           1_561.27       134.22     1_695.49       0.1037          1.3076            1.3082         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)          1_561.27       240.94     1_802.21       0.3364          1.0790            1.0764         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)          1_561.27       364.01     1_925.28       0.4477          1.0488            1.0461         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)          1_561.27       241.43     1_802.70       0.3364          1.0790            1.0764         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)          1_561.27       369.90     1_931.17       0.4477          1.0488            1.0461         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)          1_561.27       248.79     1_810.06       0.3364          1.0790            1.0764         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)          1_561.27       378.98     1_940.25       0.4477          1.0488            1.0461         5.21
IVF-Binary-512-nl223-random (self)                     1_561.27       588.09     2_149.36       0.3478          1.0765            1.0802         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)           2_205.66       139.04     2_344.70       0.1080          1.2996            1.2983         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)           2_205.66       141.39     2_347.05       0.1080          1.2996            1.2983         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)           2_205.66       142.26     2_347.92       0.1080          1.2996            1.2983         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)          2_205.66       252.70     2_458.36       0.3430          1.0762            1.0744         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)          2_205.66       382.68     2_588.34       0.4551          1.0469            1.0448         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)          2_205.66       251.61     2_457.27       0.3430          1.0762            1.0744         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)          2_205.66       382.37     2_588.03       0.4550          1.0469            1.0448         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)          2_205.66       274.15     2_479.81       0.3430          1.0762            1.0744         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)          2_205.66       393.25     2_598.91       0.4550          1.0469            1.0448         5.48
IVF-Binary-512-nl316-random (self)                     2_205.66       625.27     2_830.93       0.3535          1.0741            1.0784         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)               2_658.47       120.38     2_778.85       0.0963          1.3201            1.3252         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)              2_658.47       121.55     2_780.02       0.0962          1.3202            1.3252         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)              2_658.47       124.41     2_782.88       0.0962          1.3202            1.3252         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)              2_658.47       231.84     2_890.31       0.3267          1.0842            1.0787         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)              2_658.47       351.22     3_009.69       0.4414          1.0514            1.0467         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)             2_658.47       232.03     2_890.50       0.3264          1.0842            1.0787         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)             2_658.47       355.92     3_014.39       0.4413          1.0514            1.0467         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)             2_658.47       239.76     2_898.22       0.3264          1.0842            1.0787         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)             2_658.47       355.35     3_013.82       0.4413          1.0514            1.0467         5.02
IVF-Binary-512-nl158-pca (self)                        2_658.47       555.77     3_214.24       0.3378          1.0812            1.0824         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_764.68       128.96     1_893.64       0.1044          1.3060            1.3056         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_764.68       129.25     1_893.93       0.1044          1.3060            1.3056         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_764.68       135.56     1_900.24       0.1044          1.3060            1.3056         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_764.68       243.84     2_008.52       0.3374          1.0788            1.0750         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_764.68       364.56     2_129.24       0.4529          1.0478            1.0448         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_764.68       239.69     2_004.37       0.3374          1.0789            1.0750         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_764.68       369.84     2_134.51       0.4529          1.0478            1.0448         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_764.68       247.95     2_012.63       0.3374          1.0789            1.0750         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_764.68       370.58     2_135.25       0.4529          1.0478            1.0448         5.21
IVF-Binary-512-nl223-pca (self)                        1_764.68       589.67     2_354.35       0.3475          1.0770            1.0795         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)              2_379.22       140.86     2_520.08       0.1079          1.2993            1.2969         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)              2_379.22       139.09     2_518.31       0.1079          1.2993            1.2969         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)              2_379.22       141.83     2_521.06       0.1079          1.2993            1.2969         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)             2_379.22       250.62     2_629.85       0.3441          1.0758            1.0730         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)             2_379.22       376.46     2_755.68       0.4591          1.0464            1.0438         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)             2_379.22       251.08     2_630.30       0.3441          1.0758            1.0730         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)             2_379.22       381.45     2_760.68       0.4591          1.0464            1.0438         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)             2_379.22       254.33     2_633.56       0.3441          1.0758            1.0730         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)             2_379.22       384.52     2_763.74       0.4591          1.0464            1.0438         5.48
IVF-Binary-512-nl316-pca (self)                        2_379.22       625.90     3_005.12       0.3536          1.0746            1.0779         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)           2_668.51       194.74     2_863.25       0.1327          1.2680            1.2723         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)          2_668.51       195.87     2_864.38       0.1327          1.2680            1.2723         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)          2_668.51       198.53     2_867.04       0.1327          1.2680            1.2723         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)          2_668.51       313.03     2_981.54       0.3746          1.0643            1.0666         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)          2_668.51       446.34     3_114.85       0.4907          1.0386            1.0390         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)         2_668.51       324.85     2_993.36       0.3745          1.0643            1.0666         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)         2_668.51       460.72     3_129.23       0.4907          1.0386            1.0390         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)         2_668.51       336.53     3_005.04       0.3745          1.0643            1.0666         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)         2_668.51       462.17     3_130.68       0.4907          1.0386            1.0390         9.57
IVF-Binary-1024-nl158-random (self)                    2_668.51       846.67     3_515.18       0.3826          1.0665            1.0715         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_772.41       205.02     1_977.44       0.1368          1.2612            1.2660         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_772.41       204.00     1_976.42       0.1368          1.2612            1.2660         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_772.41       213.72     1_986.14       0.1368          1.2612            1.2660         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_772.41       332.93     2_105.35       0.3802          1.0625            1.0652         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_772.41       461.79     2_234.20       0.4970          1.0375            1.0378         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_772.41       335.97     2_108.38       0.3802          1.0625            1.0652         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_772.41       476.07     2_248.49       0.4970          1.0375            1.0378         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_772.41       343.72     2_116.13       0.3802          1.0625            1.0652         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_772.41       475.84     2_248.26       0.4970          1.0375            1.0378         9.76
IVF-Binary-1024-nl223-random (self)                    1_772.41       895.33     2_667.75       0.3881          1.0649            1.0698         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)          2_394.31       215.75     2_610.06       0.1389          1.2580            1.2624        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)          2_394.31       215.56     2_609.87       0.1389          1.2580            1.2624        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)          2_394.31       219.93     2_614.24       0.1389          1.2580            1.2624        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)         2_394.31       347.34     2_741.65       0.3846          1.0612            1.0639        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)         2_394.31       477.19     2_871.50       0.5016          1.0366            1.0373        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)         2_394.31       348.37     2_742.68       0.3846          1.0612            1.0639        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)         2_394.31       479.22     2_873.53       0.5016          1.0366            1.0373        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)         2_394.31       358.53     2_752.84       0.3846          1.0612            1.0639        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)         2_394.31       503.25     2_897.57       0.5016          1.0366            1.0373        10.04
IVF-Binary-1024-nl316-random (self)                    2_394.31       934.53     3_328.85       0.3917          1.0638            1.0688        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)              2_877.61       195.90     3_073.50       0.1360          1.2617            1.2650         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)             2_877.61       199.31     3_076.92       0.1360          1.2617            1.2650         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)             2_877.61       201.22     3_078.83       0.1360          1.2617            1.2650         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)             2_877.61       315.00     3_192.61       0.3810          1.0621            1.0643         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)             2_877.61       444.17     3_321.78       0.4996          1.0369            1.0374         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)            2_877.61       322.26     3_199.87       0.3809          1.0621            1.0643         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)            2_877.61       462.85     3_340.46       0.4996          1.0369            1.0374         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)            2_877.61       329.41     3_207.02       0.3809          1.0621            1.0643         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)            2_877.61       466.81     3_344.42       0.4996          1.0369            1.0374         9.57
IVF-Binary-1024-nl158-pca (self)                       2_877.61       861.61     3_739.22       0.3875          1.0650            1.0695         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_977.91       209.09     2_187.01       0.1395          1.2561            1.2604         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_977.91       204.80     2_182.71       0.1395          1.2561            1.2604         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_977.91       211.65     2_189.56       0.1395          1.2561            1.2604         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_977.91       336.47     2_314.38       0.3861          1.0602            1.0631         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_977.91       465.51     2_443.42       0.5049          1.0360            1.0365         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_977.91       334.52     2_312.43       0.3861          1.0602            1.0631         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_977.91       467.51     2_445.43       0.5049          1.0360            1.0365         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_977.91       346.48     2_324.39       0.3861          1.0602            1.0631         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_977.91       478.48     2_456.40       0.5049          1.0360            1.0365         9.76
IVF-Binary-1024-nl223-pca (self)                       1_977.91       892.99     2_870.90       0.3927          1.0635            1.0680         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             2_612.47       218.62     2_831.09       0.1415          1.2527            1.2573        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)             2_612.47       218.36     2_830.83       0.1415          1.2527            1.2573        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)             2_612.47       220.53     2_833.00       0.1415          1.2527            1.2573        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)            2_612.47       350.46     2_962.92       0.3894          1.0595            1.0624        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)            2_612.47       476.25     3_088.72       0.5085          1.0354            1.0362        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)            2_612.47       365.03     2_977.50       0.3894          1.0595            1.0624        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)            2_612.47       481.43     3_093.90       0.5085          1.0354            1.0362        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)            2_612.47       368.30     2_980.77       0.3894          1.0595            1.0624        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)            2_612.47       490.00     3_102.46       0.5085          1.0354            1.0362        10.04
IVF-Binary-1024-nl316-pca (self)                       2_612.47       931.81     3_544.28       0.3958          1.0626            1.0672        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)              2_329.94       406.11     2_736.05       0.1284          1.2834            1.2814         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)             2_329.94       409.27     2_739.21       0.1284          1.2834            1.2814         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)             2_329.94       444.38     2_774.31       0.1284          1.2834            1.2814         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)             2_329.94       500.30     2_830.24       0.3627          1.0704            1.0698         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)             2_329.94       916.24     3_246.18       0.4854          1.0406            1.0396         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)            2_329.94       501.73     2_831.66       0.3627          1.0704            1.0698         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)            2_329.94       918.10     3_248.04       0.4854          1.0406            1.0396         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)            2_329.94       506.27     2_836.20       0.3627          1.0704            1.0698         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)            2_329.94       921.75     3_251.68       0.4854          1.0406            1.0396         5.04
IVF-Binary-768-nl158-sign (self)                       2_329.94     1_412.47     3_742.40       0.3701          1.0715            1.0746         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)             1_437.18       414.59     1_851.77       0.1282          1.2804            1.2811         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)             1_437.18       414.88     1_852.06       0.1282          1.2804            1.2811         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)             1_437.18       419.31     1_856.49       0.1282          1.2804            1.2811         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)            1_437.18       529.12     1_966.31       0.3659          1.0690            1.0690         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)            1_437.18       918.36     2_355.54       0.4872          1.0400            1.0394         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)            1_437.18       509.62     1_946.80       0.3659          1.0690            1.0690         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)            1_437.18       927.85     2_365.03       0.4872          1.0400            1.0394         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)            1_437.18       515.06     1_952.24       0.3659          1.0690            1.0690         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)            1_437.18       929.94     2_367.12       0.4872          1.0400            1.0394         5.23
IVF-Binary-768-nl223-sign (self)                       1_437.18     1_449.82     2_887.00       0.3730          1.0703            1.0736         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)             2_046.70       422.08     2_468.78       0.1287          1.2802            1.2810         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)             2_046.70       423.22     2_469.92       0.1287          1.2802            1.2810         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)             2_046.70       426.47     2_473.16       0.1287          1.2802            1.2810         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)            2_046.70       523.32     2_570.02       0.3674          1.0684            1.0685         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)            2_046.70       932.34     2_979.04       0.4890          1.0397            1.0392         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)            2_046.70       539.33     2_586.03       0.3674          1.0684            1.0685         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)            2_046.70       995.15     3_041.85       0.4890          1.0397            1.0392         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)            2_046.70       542.32     2_589.02       0.3674          1.0684            1.0685         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)            2_046.70       951.43     2_998.13       0.4890          1.0397            1.0392         5.51
IVF-Binary-768-nl316-sign (self)                       2_046.70     1_480.18     3_526.87       0.3747          1.0698            1.0734         5.51
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
Exhaustive (query)                                        34.85       758.89       793.75       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         34.85     2_577.48     2_612.34       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                100.76       247.72       348.48       0.5519          1.8826            1.5884         1.78
ExhaustiveBinary-256-random-rf10 (query)                 100.76       379.88       480.64       0.9881          1.0022            1.0000         1.78
ExhaustiveBinary-256-random-rf20 (query)                 100.76       480.03       580.79       0.9980          1.0003            1.0000         1.78
ExhaustiveBinary-256-random (self)                       100.76     1_208.29     1_309.05       0.9881          1.0022            1.0000         1.78
ExhaustiveBinary-256-pca_no_rr (query)                    97.87       256.18       354.04       0.5930          1.6081            1.4152         1.78
ExhaustiveBinary-256-pca-rf10 (query)                     97.87       369.61       467.47       0.9919          1.0013            1.0000         1.78
ExhaustiveBinary-256-pca-rf20 (query)                     97.87       493.07       590.93       0.9988          1.0001            1.0000         1.78
ExhaustiveBinary-256-pca (self)                           97.87     1_227.08     1_324.94       0.9915          1.0014            1.0000         1.78
ExhaustiveBinary-512-random_no_rr (query)                 83.97       361.56       445.53       0.6306          1.5767            1.3633         3.55
ExhaustiveBinary-512-random-rf10 (query)                  83.97       494.52       578.49       0.9975          1.0004            1.0000         3.55
ExhaustiveBinary-512-random-rf20 (query)                  83.97       618.53       702.50       0.9998          1.0000            1.0000         3.55
ExhaustiveBinary-512-random (self)                        83.97     1_622.14     1_706.11       0.9973          1.0004            1.0000         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   108.30       368.15       476.46       0.6479          1.4884            1.3147         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    108.30       490.57       598.87       0.9983          1.0002            1.0000         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    108.30       613.79       722.09       0.9998          1.0000            1.0000         3.55
ExhaustiveBinary-512-pca (self)                          108.30     1_621.45     1_729.76       0.9981          1.0002            1.0000         3.55
ExhaustiveBinary-1024-random_no_rr (query)               120.36       541.73       662.09       0.6758          1.4452            1.2804         7.10
ExhaustiveBinary-1024-random-rf10 (query)                120.36       668.23       788.60       0.9995          1.0001            1.0000         7.10
ExhaustiveBinary-1024-random-rf20 (query)                120.36       819.87       940.23       0.9999          1.0000            1.0000         7.10
ExhaustiveBinary-1024-random (self)                      120.36     2_245.36     2_365.72       0.9993          1.0001            1.0000         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  145.63       540.97       686.60       0.6838          1.4142            1.2651         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   145.63       679.44       825.07       0.9996          1.0000            1.0000         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   145.63       801.10       946.73       1.0000          1.0000            1.0000         7.10
ExhaustiveBinary-1024-pca (self)                         145.63     2_239.13     2_384.76       0.9995          1.0001            1.0000         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   41.42       436.60       478.03       0.0376         19.4734           14.8778         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    41.42       472.71       514.13       0.1617          2.7567            2.6548         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    41.42       736.56       777.99       0.2739          1.9837            1.9249         1.53
ExhaustiveBinary-256-sign (self)                          41.42     1_598.23     1_639.65       0.1691          2.7353            2.6299         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)            1_003.86        57.31     1_061.17       0.5655          1.6704            1.5137         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           1_003.86        66.68     1_070.55       0.5588          1.7297            1.5496         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           1_003.86        78.24     1_082.10       0.5568          1.7636            1.5627         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           1_003.86       126.24     1_130.11       0.9903          1.0018            1.0000         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           1_003.86       177.42     1_181.28       0.9968          1.0006            1.0000         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          1_003.86       133.19     1_137.06       0.9907          1.0016            1.0000         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          1_003.86       186.68     1_190.55       0.9986          1.0002            1.0000         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          1_003.86       135.49     1_139.35       0.9898          1.0018            1.0000         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          1_003.86       197.97     1_201.83       0.9984          1.0002            1.0000         1.93
IVF-Binary-256-nl158-random (self)                     1_003.86       338.95     1_342.81       0.9904          1.0017            1.0000         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             572.09        51.07       623.16       0.5629          1.6755            1.5256         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             572.09        54.50       626.59       0.5605          1.7012            1.5424         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             572.09        61.37       633.46       0.5578          1.7449            1.5594         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            572.09       116.53       688.62       0.9912          1.0015            1.0000         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            572.09       165.17       737.26       0.9984          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            572.09       118.21       690.30       0.9909          1.0015            1.0000         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            572.09       190.16       762.25       0.9987          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            572.09       125.11       697.21       0.9900          1.0017            1.0000         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            572.09       183.36       755.45       0.9985          1.0002            1.0000         2.00
IVF-Binary-256-nl223-random (self)                       572.09       327.42       899.51       0.9908          1.0016            1.0000         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             759.70        54.84       814.54       0.5622          1.6812            1.5290         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             759.70        53.86       813.56       0.5610          1.6936            1.5367         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             759.70        65.43       825.13       0.5584          1.7353            1.5549         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            759.70       120.60       880.30       0.9917          1.0014            1.0000         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            759.70       180.37       940.07       0.9987          1.0002            1.0000         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            759.70       114.48       874.18       0.9914          1.0014            1.0000         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            759.70       178.65       938.35       0.9988          1.0002            1.0000         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            759.70       123.41       883.11       0.9903          1.0017            1.0000         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            759.70       179.99       939.69       0.9986          1.0002            1.0000         2.09
IVF-Binary-256-nl316-random (self)                       759.70       296.15     1_055.85       0.9912          1.0015            1.0000         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               1_056.04        49.11     1_105.15       0.6038          1.4891            1.3755         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              1_056.04        57.62     1_113.66       0.5989          1.5218            1.3913         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              1_056.04        66.96     1_123.00       0.5975          1.5420            1.3965         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              1_056.04       114.71     1_170.75       0.9926          1.0013            1.0000         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              1_056.04       181.59     1_237.64       0.9972          1.0006            1.0000         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             1_056.04       124.88     1_180.92       0.9934          1.0010            1.0000         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             1_056.04       196.19     1_252.23       0.9991          1.0001            1.0000         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             1_056.04       143.50     1_199.54       0.9927          1.0012            1.0000         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             1_056.04       195.99     1_252.03       0.9990          1.0001            1.0000         1.93
IVF-Binary-256-nl158-pca (self)                        1_056.04       336.57     1_392.61       0.9929          1.0011            1.0000         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                582.67        51.81       634.48       0.6017          1.4948            1.3802         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                582.67        54.71       637.39       0.5998          1.5091            1.3875         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                582.67        65.05       647.72       0.5979          1.5320            1.3962         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               582.67       119.86       702.54       0.9937          1.0010            1.0000         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               582.67       169.03       751.70       0.9987          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               582.67       117.78       700.46       0.9935          1.0010            1.0000         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               582.67       173.44       756.11       0.9991          1.0001            1.0000         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               582.67       125.20       707.88       0.9929          1.0011            1.0000         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               582.67       179.91       762.58       0.9990          1.0001            1.0000         2.00
IVF-Binary-256-nl223-pca (self)                          582.67       295.78       878.45       0.9931          1.0011            1.0000         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                777.81        54.07       831.88       0.6012          1.4964            1.3827         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                777.81        53.96       831.77       0.6002          1.5046            1.3866         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                777.81        60.59       838.40       0.5985          1.5256            1.3943         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               777.81       113.40       891.21       0.9940          1.0009            1.0000         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               777.81       169.81       947.62       0.9990          1.0001            1.0000         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               777.81       113.82       891.63       0.9938          1.0009            1.0000         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               777.81       185.33       963.14       0.9992          1.0001            1.0000         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               777.81       123.09       900.90       0.9931          1.0011            1.0000         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               777.81       178.51       956.32       0.9991          1.0001            1.0000         2.09
IVF-Binary-256-nl316-pca (self)                          777.81       286.33     1_064.14       0.9934          1.0011            1.0000         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)              993.40        69.06     1_062.46       0.6409          1.4486            1.3318         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)             993.40        89.65     1_083.05       0.6350          1.4901            1.3495         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)             993.40        96.84     1_090.24       0.6333          1.5130            1.3550         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)             993.40       135.82     1_129.22       0.9965          1.0007            1.0000         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)             993.40       195.61     1_189.01       0.9978          1.0005            1.0000         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)            993.40       151.10     1_144.50       0.9982          1.0002            1.0000         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)            993.40       219.99     1_213.39       0.9999          1.0000            1.0000         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)            993.40       170.04     1_163.44       0.9979          1.0003            1.0000         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)            993.40       229.26     1_222.66       0.9998          1.0000            1.0000         3.71
IVF-Binary-512-nl158-random (self)                       993.40       422.45     1_415.85       0.9979          1.0003            1.0000         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             575.80        70.78       646.57       0.6385          1.4531            1.3378         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             575.80        77.98       653.78       0.6365          1.4696            1.3445         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             575.80        86.85       662.65       0.6339          1.5002            1.3523         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            575.80       142.83       718.62       0.9978          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            575.80       209.31       785.11       0.9993          1.0001            1.0000         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            575.80       139.38       715.18       0.9980          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            575.80       200.80       776.59       0.9998          1.0000            1.0000         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            575.80       154.28       730.07       0.9979          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            575.80       208.63       784.43       0.9998          1.0000            1.0000         3.77
IVF-Binary-512-nl223-random (self)                       575.80       373.10       948.90       0.9978          1.0003            1.0000         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             786.48        73.34       859.82       0.6379          1.4598            1.3395         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             786.48        72.87       859.34       0.6370          1.4674            1.3427         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             786.48        82.90       869.37       0.6347          1.4949            1.3507         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            786.48       135.43       921.91       0.9981          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            786.48       192.70       979.17       0.9996          1.0001            1.0000         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            786.48       136.11       922.59       0.9982          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            786.48       200.57       987.05       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            786.48       147.94       934.41       0.9980          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            786.48       207.96       994.44       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-random (self)                       786.48       365.59     1_152.07       0.9980          1.0003            1.0000         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               1_005.09        67.71     1_072.80       0.6576          1.3883            1.2906         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              1_005.09        82.94     1_088.03       0.6524          1.4212            1.3033         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              1_005.09        99.09     1_104.19       0.6508          1.4401            1.3081         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              1_005.09       138.13     1_143.22       0.9969          1.0006            1.0000         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              1_005.09       205.09     1_210.18       0.9978          1.0005            1.0000         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             1_005.09       154.72     1_159.81       0.9987          1.0001            1.0000         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             1_005.09       216.22     1_221.31       0.9999          1.0000            1.0000         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             1_005.09       166.99     1_172.08       0.9985          1.0002            1.0000         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             1_005.09       233.91     1_239.00       0.9999          1.0000            1.0000         3.71
IVF-Binary-512-nl158-pca (self)                        1_005.09       422.25     1_427.34       0.9986          1.0002            1.0000         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                609.37        74.68       684.05       0.6553          1.3963            1.2930         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                609.37        73.99       683.35       0.6534          1.4090            1.2987         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                609.37        85.71       695.08       0.6514          1.4319            1.3057         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               609.37       133.16       742.52       0.9983          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               609.37       199.38       808.74       0.9993          1.0001            1.0000         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               609.37       140.23       749.59       0.9986          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               609.37       199.92       809.28       0.9998          1.0000            1.0000         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               609.37       152.75       762.11       0.9985          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               609.37       215.01       824.38       0.9999          1.0000            1.0000         3.77
IVF-Binary-512-nl223-pca (self)                          609.37       376.12       985.49       0.9985          1.0002            1.0000         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                810.34        74.25       884.60       0.6545          1.4014            1.2937         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                810.34        72.49       882.83       0.6536          1.4079            1.2965         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                810.34        84.01       894.35       0.6516          1.4281            1.3030         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               810.34       146.77       957.11       0.9986          1.0002            1.0000         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               810.34       189.40       999.74       0.9996          1.0001            1.0000         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               810.34       135.38       945.72       0.9987          1.0001            1.0000         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               810.34       197.85     1_008.20       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               810.34       162.40       972.74       0.9986          1.0002            1.0000         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               810.34       208.99     1_019.34       0.9999          1.0000            1.0000         3.86
IVF-Binary-512-nl316-pca (self)                          810.34       360.42     1_170.76       0.9986          1.0002            1.0000         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)           1_038.09       104.07     1_142.16       0.6845          1.3532            1.2576         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)          1_038.09       122.20     1_160.30       0.6792          1.3863            1.2711         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)          1_038.09       149.14     1_187.23       0.6776          1.4037            1.2752         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)          1_038.09       175.55     1_213.64       0.9976          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)          1_038.09       235.45     1_273.55       0.9978          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)         1_038.09       202.06     1_240.16       0.9997          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)         1_038.09       260.80     1_298.90       0.9999          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)         1_038.09       214.27     1_252.36       0.9996          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)         1_038.09       292.06     1_330.16       1.0000          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-random (self)                    1_038.09       567.89     1_605.99       0.9995          1.0000            1.0000         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            651.19       102.35       753.54       0.6825          1.3587            1.2602         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            651.19       116.76       767.96       0.6805          1.3722            1.2665         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            651.19       129.53       780.72       0.6783          1.3950            1.2736         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           651.19       174.01       825.21       0.9991          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           651.19       237.26       888.45       0.9994          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           651.19       177.61       828.81       0.9995          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           651.19       250.26       901.46       0.9998          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           651.19       195.46       846.66       0.9996          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           651.19       260.21       911.40       0.9999          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-random (self)                      651.19       505.44     1_156.63       0.9994          1.0001            1.0000         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            838.16       112.38       950.54       0.6814          1.3665            1.2637         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)            838.16       107.88       946.05       0.6806          1.3728            1.2656         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)            838.16       121.46       959.62       0.6785          1.3928            1.2727         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)           838.16       176.36     1_014.53       0.9994          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)           838.16       229.39     1_067.56       0.9997          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)           838.16       175.30     1_013.47       0.9995          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)           838.16       238.29     1_076.45       0.9998          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)           838.16       190.05     1_028.21       0.9996          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)           838.16       259.58     1_097.74       0.9999          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-random (self)                      838.16       476.91     1_315.08       0.9994          1.0001            1.0000         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)              1_047.92       104.61     1_152.53       0.6927          1.3310            1.2428         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)             1_047.92       124.22     1_172.13       0.6876          1.3596            1.2559         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)             1_047.92       146.39     1_194.30       0.6860          1.3757            1.2595         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)             1_047.92       174.50     1_222.42       0.9977          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)             1_047.92       235.11     1_283.03       0.9978          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)            1_047.92       200.83     1_248.74       0.9998          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)            1_047.92       256.92     1_304.83       1.0000          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)            1_047.92       224.18     1_272.10       0.9997          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)            1_047.92       281.22     1_329.13       1.0000          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-pca (self)                       1_047.92       566.62     1_614.54       0.9997          1.0000            1.0000         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               648.48       101.90       750.38       0.6901          1.3388            1.2451         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               648.48       112.38       760.86       0.6884          1.3497            1.2507         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               648.48       128.46       776.95       0.6863          1.3698            1.2573         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              648.48       175.13       823.61       0.9992          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              648.48       237.51       885.99       0.9994          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              648.48       178.26       826.75       0.9996          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              648.48       247.71       896.19       0.9999          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              648.48       194.36       842.85       0.9996          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              648.48       260.22       908.70       1.0000          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-pca (self)                         648.48       496.04     1_144.52       0.9995          1.0001            1.0000         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               813.80       106.89       920.69       0.6893          1.3439            1.2492         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)               813.80       109.98       923.78       0.6884          1.3499            1.2513         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)               813.80       121.47       935.26       0.6867          1.3674            1.2565         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)              813.80       171.08       984.88       0.9995          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)              813.80       253.27     1_067.06       0.9997          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)              813.80       189.26     1_003.05       0.9996          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)              813.80       243.85     1_057.65       0.9999          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)              813.80       199.21     1_013.00       0.9997          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)              813.80       254.93     1_068.73       1.0000          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-pca (self)                         813.80       490.04     1_303.84       0.9995          1.0001            1.0000         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)                983.43       183.95     1_167.38       0.0686          6.6373            6.1345         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               983.43       203.26     1_186.69       0.0552          7.8718            7.1703         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               983.43       214.27     1_197.70       0.0506          8.7173            7.9030         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               983.43       230.57     1_214.00       0.3995          1.6136            1.5282         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               983.43       400.44     1_383.87       0.6372          1.2495            1.1925         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              983.43       244.38     1_227.81       0.3092          1.8460            1.7473         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              983.43       421.15     1_404.58       0.4802          1.4437            1.3765         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              983.43       255.48     1_238.91       0.2742          1.9837            1.8755         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              983.43       467.32     1_450.75       0.4127          1.5616            1.4852         1.68
IVF-Binary-256-nl158-sign (self)                         983.43       732.49     1_715.92       0.3153          1.8338            1.7387         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               533.50       174.51       708.01       0.0663          6.5482            6.0899         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               533.50       182.85       716.35       0.0608          6.9856            6.4759         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               533.50       198.81       732.31       0.0542          7.8630            7.2917         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              533.50       222.73       756.23       0.3662          1.6629            1.5796         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              533.50       379.36       912.86       0.5970          1.2833            1.2269         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              533.50       230.90       764.40       0.3332          1.7517            1.6663         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              533.50       391.18       924.68       0.5316          1.3585            1.2997         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              533.50       233.06       766.56       0.2923          1.8995            1.8113         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              533.50       408.32       941.82       0.4448          1.4906            1.4280         1.75
IVF-Binary-256-nl223-sign (self)                         533.50       654.05     1_187.55       0.3388          1.7404            1.6554         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               759.88       172.61       932.49       0.0659          6.5012            6.0523         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               759.88       175.43       935.32       0.0631          6.7220            6.2235         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               759.88       185.41       945.30       0.0561          7.4986            6.9108         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              759.88       213.48       973.36       0.3648          1.6702            1.5892         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              759.88       380.74     1_140.62       0.5865          1.2945            1.2405         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              759.88       221.56       981.44       0.3479          1.7151            1.6275         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              759.88       384.24     1_144.13       0.5531          1.3338            1.2771         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              759.88       228.05       987.93       0.3051          1.8539            1.7618         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              759.88       408.34     1_168.22       0.4659          1.4564            1.3961         1.84
IVF-Binary-256-nl316-sign (self)                         759.88       640.85     1_400.74       0.3540          1.7015            1.6180         1.84
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
Exhaustive (query)                                        70.57     1_370.10     1_440.67       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         70.57     4_488.34     4_558.91       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                133.62       270.75       404.38       0.5547          1.7646            1.5366         2.03
ExhaustiveBinary-256-random-rf10 (query)                 133.62       414.03       547.65       0.9898          1.0017            1.0000         2.03
ExhaustiveBinary-256-random-rf20 (query)                 133.62       557.57       691.20       0.9985          1.0002            1.0000         2.03
ExhaustiveBinary-256-random (self)                       133.62     1_324.96     1_458.58       0.9899          1.0016            1.0000         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   223.43       271.30       494.74       0.5767          1.6243            1.4311         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    223.43       414.24       637.67       0.9904          1.0016            1.0000         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    223.43       554.27       777.70       0.9984          1.0002            1.0000         2.03
ExhaustiveBinary-256-pca (self)                          223.43     1_318.51     1_541.94       0.9905          1.0016            1.0000         2.03
ExhaustiveBinary-512-random_no_rr (query)                204.55       383.87       588.42       0.6013          1.6760            1.4608         4.05
ExhaustiveBinary-512-random-rf10 (query)                 204.55       538.76       743.31       0.9977          1.0003            1.0000         4.05
ExhaustiveBinary-512-random-rf20 (query)                 204.55       681.83       886.38       0.9998          1.0000            1.0000         4.05
ExhaustiveBinary-512-random (self)                       204.55     1_730.80     1_935.35       0.9975          1.0003            1.0000         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   293.19       387.12       680.31       0.6443          1.4426            1.3064         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    293.19       534.38       827.57       0.9985          1.0002            1.0000         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    293.19       719.65     1_012.84       0.9999          1.0000            1.0000         4.05
ExhaustiveBinary-512-pca (self)                          293.19     1_733.55     2_026.74       0.9984          1.0002            1.0000         4.05
ExhaustiveBinary-1024-random_no_rr (query)               255.90       591.92       847.82       0.6624          1.4553            1.3048         8.11
ExhaustiveBinary-1024-random-rf10 (query)                255.90       745.87     1_001.77       0.9995          1.0001            1.0000         8.11
ExhaustiveBinary-1024-random-rf20 (query)                255.90       917.99     1_173.89       1.0000          1.0000            1.0000         8.11
ExhaustiveBinary-1024-random (self)                      255.90     2_459.31     2_715.21       0.9994          1.0001            1.0000         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  346.97       587.87       934.84       0.6865          1.3603            1.2383         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   346.97       747.28     1_094.25       0.9996          1.0001            1.0000         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   346.97       903.52     1_250.49       1.0000          1.0000            1.0000         8.11
ExhaustiveBinary-1024-pca (self)                         346.97     2_487.40     2_834.37       0.9996          1.0001            1.0000         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   85.32       660.68       745.99       0.0400         18.1511           13.6734         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    85.32       736.20       821.51       0.1821          2.5573            2.4620         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    85.32     1_132.74     1_218.05       0.3140          1.8429            1.7786         3.05
ExhaustiveBinary-512-sign (self)                          85.32     2_322.84     2_408.15       0.1897          2.5286            2.4283         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)            1_903.67        81.12     1_984.79       0.5633          1.6328            1.4874         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           1_903.67        95.77     1_999.45       0.5600          1.6630            1.5060         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           1_903.67       105.94     2_009.61       0.5583          1.6921            1.5135         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           1_903.67       173.40     2_077.08       0.9917          1.0013            1.0000         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           1_903.67       265.37     2_169.05       0.9978          1.0004            1.0000         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          1_903.67       173.28     2_076.96       0.9917          1.0013            1.0000         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          1_903.67       277.28     2_180.95       0.9989          1.0001            1.0000         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          1_903.67       183.71     2_087.38       0.9910          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          1_903.67       287.59     2_191.27       0.9988          1.0001            1.0000         2.34
IVF-Binary-256-nl158-random (self)                     1_903.67       396.74     2_300.42       0.9919          1.0013            1.0000         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)             901.37        77.92       979.29       0.5617          1.6446            1.4977         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)             901.37        82.49       983.86       0.5603          1.6574            1.5054         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)             901.37        91.51       992.88       0.5590          1.6801            1.5121         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)            901.37       174.68     1_076.05       0.9924          1.0011            1.0000         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)            901.37       268.13     1_169.50       0.9988          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)            901.37       172.02     1_073.39       0.9918          1.0012            1.0000         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)            901.37       270.28     1_171.65       0.9989          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)            901.37       179.62     1_080.99       0.9911          1.0014            1.0000         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)            901.37       279.37     1_180.74       0.9988          1.0001            1.0000         2.47
IVF-Binary-256-nl223-random (self)                       901.37       379.31     1_280.68       0.9920          1.0012            1.0000         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)           1_157.14        83.55     1_240.69       0.5616          1.6430            1.4953         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           1_157.14        84.93     1_242.07       0.5609          1.6507            1.4993         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           1_157.14        90.39     1_247.53       0.5595          1.6715            1.5088         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          1_157.14       174.82     1_331.96       0.9924          1.0011            1.0000         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          1_157.14       270.17     1_427.31       0.9990          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          1_157.14       175.79     1_332.93       0.9920          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          1_157.14       274.84     1_431.98       0.9990          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          1_157.14       184.83     1_341.97       0.9913          1.0013            1.0000         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          1_157.14       286.33     1_443.47       0.9988          1.0001            1.0000         2.65
IVF-Binary-256-nl316-random (self)                     1_157.14       375.62     1_532.76       0.9922          1.0012            1.0000         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               1_992.34        73.27     2_065.61       0.5839          1.5249            1.4040         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              1_992.34        82.43     2_074.77       0.5812          1.5475            1.4163         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              1_992.34        88.81     2_081.15       0.5800          1.5661            1.4216         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              1_992.34       168.16     2_160.50       0.9918          1.0013            1.0000         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              1_992.34       261.70     2_254.04       0.9977          1.0004            1.0000         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             1_992.34       173.78     2_166.12       0.9918          1.0013            1.0000         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             1_992.34       271.62     2_263.96       0.9988          1.0002            1.0000         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             1_992.34       180.41     2_172.75       0.9912          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             1_992.34       279.15     2_271.49       0.9987          1.0002            1.0000         2.34
IVF-Binary-256-nl158-pca (self)                        1_992.34       403.25     2_395.59       0.9920          1.0013            1.0000         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)                985.46        77.98     1_063.44       0.5829          1.5323            1.4104         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)                985.46        81.31     1_066.77       0.5817          1.5427            1.4161         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)                985.46        91.42     1_076.88       0.5806          1.5586            1.4197         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)               985.46       170.90     1_156.36       0.9924          1.0012            1.0000         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)               985.46       269.31     1_254.76       0.9988          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)               985.46       171.87     1_157.33       0.9921          1.0013            1.0000         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)               985.46       274.24     1_259.69       0.9988          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)               985.46       182.13     1_167.59       0.9914          1.0014            1.0000         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)               985.46       277.22     1_262.67       0.9987          1.0002            1.0000         2.47
IVF-Binary-256-nl223-pca (self)                          985.46       375.59     1_361.05       0.9921          1.0013            1.0000         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              1_241.79        83.13     1_324.92       0.5828          1.5321            1.4094         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              1_241.79        83.87     1_325.67       0.5823          1.5371            1.4112         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              1_241.79        91.02     1_332.82       0.5810          1.5509            1.4187         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             1_241.79       176.22     1_418.01       0.9923          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             1_241.79       272.40     1_514.20       0.9990          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             1_241.79       173.99     1_415.78       0.9920          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             1_241.79       281.78     1_523.57       0.9989          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             1_241.79       181.39     1_423.18       0.9914          1.0014            1.0000         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             1_241.79       286.79     1_528.59       0.9988          1.0002            1.0000         2.65
IVF-Binary-256-nl316-pca (self)                        1_241.79       377.04     1_618.83       0.9923          1.0012            1.0000         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)            1_939.79        99.88     2_039.66       0.6093          1.5616            1.4208         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)           1_939.79       114.91     2_054.70       0.6053          1.5925            1.4401         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)           1_939.79       134.79     2_074.58       0.6034          1.6196            1.4475         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)           1_939.79       196.43     2_136.21       0.9973          1.0004            1.0000         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)           1_939.79       292.94     2_232.73       0.9984          1.0003            1.0000         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)          1_939.79       208.71     2_148.50       0.9983          1.0002            1.0000         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)          1_939.79       310.96     2_250.75       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)          1_939.79       223.24     2_163.03       0.9980          1.0002            1.0000         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)          1_939.79       320.47     2_260.25       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-random (self)                     1_939.79       513.54     2_453.33       0.9982          1.0002            1.0000         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)           1_001.94       104.92     1_106.85       0.6074          1.5768            1.4307         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)           1_001.94       110.51     1_112.45       0.6059          1.5910            1.4390         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)           1_001.94       121.35     1_123.29       0.6046          1.6113            1.4455         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)          1_001.94       202.04     1_203.98       0.9982          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)          1_001.94       298.82     1_300.75       0.9996          1.0001            1.0000         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)          1_001.94       204.38     1_206.31       0.9982          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)          1_001.94       310.63     1_312.57       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)          1_001.94       213.73     1_215.67       0.9980          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)          1_001.94       319.34     1_321.28       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-random (self)                     1_001.94       496.68     1_498.61       0.9982          1.0002            1.0000         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)           1_198.64       110.95     1_309.59       0.6068          1.5788            1.4333         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)           1_198.64       114.50     1_313.15       0.6061          1.5861            1.4378         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)           1_198.64       123.08     1_321.72       0.6045          1.6061            1.4455         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)          1_198.64       205.05     1_403.69       0.9985          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)          1_198.64       301.02     1_499.66       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)          1_198.64       203.98     1_402.62       0.9984          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)          1_198.64       302.94     1_501.58       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)          1_198.64       214.90     1_413.54       0.9982          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)          1_198.64       320.36     1_519.00       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-random (self)                     1_198.64       481.16     1_679.80       0.9983          1.0002            1.0000         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)               2_057.91        99.89     2_157.81       0.6498          1.3846            1.2884         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)              2_057.91       115.33     2_173.24       0.6473          1.4005            1.2964         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)              2_057.91       125.43     2_183.34       0.6462          1.4122            1.3004         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)              2_057.91       195.68     2_253.59       0.9975          1.0004            1.0000         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)              2_057.91       291.34     2_349.25       0.9984          1.0003            1.0000         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)             2_057.91       204.76     2_262.67       0.9987          1.0001            1.0000         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)             2_057.91       308.18     2_366.09       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)             2_057.91       218.30     2_276.22       0.9986          1.0001            1.0000         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)             2_057.91       329.69     2_387.60       0.9999          1.0000            1.0000         4.36
IVF-Binary-512-nl158-pca (self)                        2_057.91       529.21     2_587.12       0.9987          1.0001            1.0000         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_066.84       104.99     1_171.83       0.6484          1.3915            1.2913         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_066.84       110.19     1_177.03       0.6474          1.3991            1.2957         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_066.84       121.81     1_188.65       0.6464          1.4094            1.2989         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_066.84       199.50     1_266.35       0.9986          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_066.84       296.56     1_363.40       0.9996          1.0001            1.0000         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_066.84       203.62     1_270.46       0.9987          1.0001            1.0000         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_066.84       302.73     1_369.57       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_066.84       215.09     1_281.93       0.9986          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_066.84       319.45     1_386.30       0.9999          1.0000            1.0000         4.49
IVF-Binary-512-nl223-pca (self)                        1_066.84       487.20     1_554.04       0.9987          1.0002            1.0000         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)              1_302.33       110.75     1_413.08       0.6480          1.3924            1.2943         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)              1_302.33       112.83     1_415.16       0.6476          1.3963            1.2956         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)              1_302.33       121.76     1_424.09       0.6465          1.4068            1.2991         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)             1_302.33       203.50     1_505.83       0.9988          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)             1_302.33       307.19     1_609.52       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)             1_302.33       202.43     1_504.77       0.9987          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)             1_302.33       304.93     1_607.26       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)             1_302.33       213.16     1_515.50       0.9986          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)             1_302.33       318.36     1_620.69       0.9999          1.0000            1.0000         4.67
IVF-Binary-512-nl316-pca (self)                        1_302.33       485.05     1_787.38       0.9987          1.0001            1.0000         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)           1_998.96       153.81     2_152.78       0.6686          1.3871            1.2852         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)          1_998.96       171.47     2_170.43       0.6654          1.4070            1.2940         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)          1_998.96       190.67     2_189.64       0.6639          1.4242            1.2993         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)          1_998.96       250.50     2_249.47       0.9983          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)          1_998.96       355.04     2_354.00       0.9985          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)         1_998.96       269.04     2_268.01       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)         1_998.96       380.11     2_379.08       0.9999          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)         1_998.96       289.94     2_288.90       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)         1_998.96       403.57     2_402.53       1.0000          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-random (self)                    1_998.96       745.20     2_744.17       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_051.99       161.74     1_213.73       0.6666          1.3978            1.2895         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_051.99       165.74     1_217.73       0.6654          1.4067            1.2934         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_051.99       187.31     1_239.30       0.6643          1.4195            1.2982         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_051.99       257.01     1_309.00       0.9994          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_051.99       360.09     1_412.08       0.9997          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_051.99       262.85     1_314.84       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_051.99       369.88     1_421.87       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_051.99       281.50     1_333.49       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_051.99       393.53     1_445.52       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-random (self)                    1_051.99       714.09     1_766.07       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)          1_276.20       165.35     1_441.55       0.6664          1.3994            1.2908         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)          1_276.20       168.59     1_444.79       0.6658          1.4041            1.2928         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)          1_276.20       180.85     1_457.05       0.6645          1.4169            1.2974         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)         1_276.20       263.08     1_539.28       0.9996          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)         1_276.20       370.40     1_646.60       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)         1_276.20       264.08     1_540.28       0.9996          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)         1_276.20       382.07     1_658.27       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)         1_276.20       285.06     1_561.26       0.9996          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)         1_276.20       399.43     1_675.63       1.0000          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-random (self)                    1_276.20       692.90     1_969.10       0.9996          1.0001            1.0000         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)              2_079.67       156.57     2_236.24       0.6914          1.3138            1.2265         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)             2_079.67       174.15     2_253.83       0.6887          1.3290            1.2329         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)             2_079.67       194.81     2_274.49       0.6877          1.3391            1.2353         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)             2_079.67       249.51     2_329.19       0.9983          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)             2_079.67       351.80     2_431.48       0.9985          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)            2_079.67       267.52     2_347.19       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)            2_079.67       378.35     2_458.02       0.9999          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)            2_079.67       292.53     2_372.20       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)            2_079.67       396.58     2_476.25       1.0000          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-pca (self)                       2_079.67       740.00     2_819.67       0.9996          1.0000            1.0000         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_117.25       160.52     1_277.77       0.6897          1.3215            1.2293         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_117.25       166.30     1_283.56       0.6887          1.3276            1.2310         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_117.25       182.87     1_300.13       0.6878          1.3371            1.2341         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_117.25       256.84     1_374.09       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_117.25       361.02     1_478.27       0.9997          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_117.25       260.40     1_377.65       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_117.25       368.94     1_486.19       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_117.25       281.70     1_398.95       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_117.25       390.98     1_508.23       1.0000          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-pca (self)                       1_117.25       703.30     1_820.55       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_388.51       166.67     1_555.18       0.6897          1.3223            1.2301         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_388.51       167.36     1_555.87       0.6893          1.3259            1.2310         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_388.51       181.18     1_569.69       0.6882          1.3349            1.2341         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_388.51       255.87     1_644.38       0.9997          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_388.51       370.89     1_759.40       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_388.51       259.02     1_647.53       0.9997          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_388.51       372.34     1_760.85       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_388.51       273.49     1_662.00       0.9997          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_388.51       391.02     1_779.53       1.0000          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-pca (self)                       1_388.51       697.08     2_085.59       0.9996          1.0000            1.0000         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)              1_838.23       284.98     2_123.21       0.0594          7.9493            7.2573         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)             1_838.23       302.74     2_140.97       0.0523          9.1717            8.0433         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)             1_838.23       319.78     2_158.02       0.0490         10.2313            8.7259         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)             1_838.23       365.69     2_203.92       0.3200          1.8552            1.7422         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)             1_838.23       643.78     2_482.01       0.5141          1.4089            1.3383         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)            1_838.23       372.54     2_210.77       0.2782          1.9998            1.8714         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)            1_838.23       673.83     2_512.06       0.4424          1.5272            1.4451         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)            1_838.23       388.08     2_226.31       0.2558          2.1020            1.9577         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)            1_838.23       699.28     2_537.51       0.4010          1.6091            1.5201         3.36
IVF-Binary-512-nl158-sign (self)                       1_838.23     1_087.35     2_925.59       0.2859          1.9751            1.8430         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               874.33       288.81     1_163.14       0.0573          7.8797            7.1620         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               874.33       298.64     1_172.97       0.0543          8.2745            7.4598         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               874.33       311.16     1_185.49       0.0504          9.1593            8.1667         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              874.33       366.84     1_241.16       0.3141          1.8536            1.7519         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              874.33       656.29     1_530.62       0.5018          1.4180            1.3471         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              874.33       367.58     1_241.91       0.2951          1.9148            1.8057         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              874.33       664.26     1_538.59       0.4685          1.4696            1.3955         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              874.33       382.35     1_256.68       0.2690          2.0219            1.8959         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              874.33       678.81     1_553.14       0.4174          1.5671            1.4860         3.49
IVF-Binary-512-nl223-sign (self)                         874.33     1_041.37     1_915.70       0.3019          1.8966            1.7840         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)             1_091.32       294.78     1_386.10       0.0576          7.7417            7.1217         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)             1_091.32       296.87     1_388.18       0.0558          7.9505            7.2924         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)             1_091.32       305.67     1_396.99       0.0519          8.6482            7.8795         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)            1_091.32       370.16     1_461.47       0.3184          1.8361            1.7366         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)            1_091.32       659.14     1_750.46       0.5013          1.4137            1.3449         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)            1_091.32       367.80     1_459.11       0.3085          1.8697            1.7662         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)            1_091.32       682.56     1_773.87       0.4838          1.4395            1.3714         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)            1_091.32       385.42     1_476.73       0.2821          1.9696            1.8495         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)            1_091.32       677.43     1_768.74       0.4343          1.5288            1.4528         3.67
IVF-Binary-512-nl316-sign (self)                       1_091.32     1_020.32     2_111.64       0.3137          1.8531            1.7443         3.67
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
Exhaustive (query)                                       104.30     1_899.97     2_004.27       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        104.30     6_455.58     6_559.88       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                190.55       288.51       479.06       0.5361          1.8068            1.5908         2.28
ExhaustiveBinary-256-random-rf10 (query)                 190.55       448.29       638.84       0.9868          1.0022            1.0000         2.28
ExhaustiveBinary-256-random-rf20 (query)                 190.55       613.50       804.05       0.9980          1.0003            1.0000         2.28
ExhaustiveBinary-256-random (self)                       190.55     1_405.10     1_595.64       0.9876          1.0021            1.0000         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   399.73       291.28       691.01       0.5754          1.5495            1.4128         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    399.73       447.75       847.48       0.9895          1.0018            1.0000         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    399.73       608.69     1_008.42       0.9983          1.0002            1.0000         2.28
ExhaustiveBinary-256-pca (self)                          399.73     1_391.29     1_791.02       0.9897          1.0017            1.0000         2.28
ExhaustiveBinary-512-random_no_rr (query)                298.16       410.04       708.20       0.5866          1.6778            1.4946         4.55
ExhaustiveBinary-512-random-rf10 (query)                 298.16       602.62       900.78       0.9966          1.0005            1.0000         4.55
ExhaustiveBinary-512-random-rf20 (query)                 298.16       751.70     1_049.86       0.9997          1.0001            1.0000         4.55
ExhaustiveBinary-512-random (self)                       298.16     1_845.04     2_143.20       0.9969          1.0004            1.0000         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   491.16       411.62       902.77       0.6388          1.4217            1.3032         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    491.16       613.75     1_104.90       0.9979          1.0003            1.0000         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    491.16       756.00     1_247.15       0.9998          1.0000            1.0000         4.55
ExhaustiveBinary-512-pca (self)                          491.16     1_857.71     2_348.86       0.9981          1.0002            1.0000         4.55
ExhaustiveBinary-1024-random_no_rr (query)               500.77       633.60     1_134.37       0.6446          1.4909            1.3512         9.11
ExhaustiveBinary-1024-random-rf10 (query)                500.77       817.45     1_318.22       0.9993          1.0001            1.0000         9.11
ExhaustiveBinary-1024-random-rf20 (query)                500.77     1_010.13     1_510.91       0.9999          1.0000            1.0000         9.11
ExhaustiveBinary-1024-random (self)                      500.77     2_676.14     3_176.91       0.9994          1.0001            1.0000         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  699.27       632.98     1_332.25       0.6795          1.3452            1.2483         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   699.27       822.02     1_521.29       0.9996          1.0001            1.0000         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   699.27     1_001.84     1_701.10       1.0000          1.0000            1.0000         9.11
ExhaustiveBinary-1024-pca (self)                         699.27     2_705.90     3_405.16       0.9997          1.0000            1.0000         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  127.29       849.32       976.61       0.0420         17.7082           13.0970         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   127.29       918.37     1_045.67       0.1896          2.5240            2.4052         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   127.29     1_433.23     1_560.53       0.3229          1.8300            1.7348         4.58
ExhaustiveBinary-768-sign (self)                         127.29     2_974.40     3_101.69       0.1997          2.4832            2.3546         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)            2_766.65       101.16     2_867.81       0.5429          1.7099            1.5460         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)           2_766.65       111.93     2_878.58       0.5407          1.7331            1.5595         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)           2_766.65       121.25     2_887.90       0.5397          1.7545            1.5687         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)           2_766.65       216.16     2_982.82       0.9891          1.0018            1.0000         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)           2_766.65       327.96     3_094.61       0.9984          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)          2_766.65       218.27     2_984.92       0.9884          1.0019            1.0000         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)          2_766.65       339.02     3_105.68       0.9986          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)          2_766.65       227.09     2_993.74       0.9877          1.0021            1.0000         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)          2_766.65       351.52     3_118.17       0.9983          1.0002            1.0000         2.74
IVF-Binary-256-nl158-random (self)                     2_766.65       507.96     3_274.62       0.9891          1.0018            1.0000         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)           1_296.52       106.50     1_403.02       0.5420          1.7182            1.5522         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)           1_296.52       111.04     1_407.56       0.5412          1.7279            1.5568         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)           1_296.52       117.17     1_413.69       0.5401          1.7508            1.5657         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)          1_296.52       218.21     1_514.73       0.9888          1.0018            1.0000         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)          1_296.52       333.38     1_629.90       0.9986          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)          1_296.52       218.00     1_514.53       0.9885          1.0019            1.0000         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)          1_296.52       335.50     1_632.02       0.9985          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)          1_296.52       219.59     1_516.11       0.9877          1.0020            1.0000         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)          1_296.52       341.47     1_637.99       0.9983          1.0002            1.0000         2.93
IVF-Binary-256-nl223-random (self)                     1_296.52       466.94     1_763.46       0.9890          1.0018            1.0000         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)           1_602.53       107.97     1_710.50       0.5422          1.7108            1.5503         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)           1_602.53       109.46     1_711.98       0.5417          1.7157            1.5534         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)           1_602.53       117.20     1_719.73       0.5408          1.7316            1.5599         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)          1_602.53       221.21     1_823.74       0.9891          1.0018            1.0000         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)          1_602.53       337.22     1_939.75       0.9987          1.0002            1.0000         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)          1_602.53       221.16     1_823.69       0.9888          1.0018            1.0000         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)          1_602.53       340.70     1_943.23       0.9986          1.0002            1.0000         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)          1_602.53       225.02     1_827.55       0.9882          1.0020            1.0000         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)          1_602.53       349.23     1_951.76       0.9984          1.0002            1.0000         3.21
IVF-Binary-256-nl316-random (self)                     1_602.53       482.93     2_085.46       0.9895          1.0017            1.0000         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)               2_900.50        95.67     2_996.17       0.5812          1.4959            1.3933         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)              2_900.50        99.81     3_000.31       0.5795          1.5088            1.3992         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)              2_900.50       107.81     3_008.32       0.5787          1.5186            1.4017         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)              2_900.50       206.94     3_107.44       0.9913          1.0014            1.0000         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)              2_900.50       320.07     3_220.57       0.9984          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)             2_900.50       211.75     3_112.25       0.9907          1.0015            1.0000         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)             2_900.50       333.58     3_234.09       0.9987          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)             2_900.50       218.26     3_118.76       0.9902          1.0016            1.0000         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)             2_900.50       342.19     3_242.69       0.9985          1.0002            1.0000         2.74
IVF-Binary-256-nl158-pca (self)                        2_900.50       475.85     3_376.35       0.9910          1.0014            1.0000         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)              1_477.10        99.02     1_576.11       0.5799          1.5020            1.3963         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)              1_477.10       102.25     1_579.35       0.5794          1.5067            1.3987         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)              1_477.10       108.58     1_585.68       0.5786          1.5169            1.4018         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)             1_477.10       212.53     1_689.63       0.9909          1.0015            1.0000         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)             1_477.10       326.41     1_803.51       0.9987          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)             1_477.10       211.79     1_688.88       0.9906          1.0015            1.0000         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)             1_477.10       331.53     1_808.63       0.9987          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)             1_477.10       218.43     1_695.53       0.9901          1.0016            1.0000         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)             1_477.10       342.28     1_819.38       0.9985          1.0002            1.0000         2.93
IVF-Binary-256-nl223-pca (self)                        1_477.10       461.69     1_938.78       0.9909          1.0014            1.0000         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)              1_803.89       109.83     1_913.72       0.5807          1.4988            1.3925         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)              1_803.89       109.09     1_912.97       0.5804          1.5016            1.3940         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)              1_803.89       115.20     1_919.09       0.5798          1.5081            1.3970         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)             1_803.89       219.65     2_023.54       0.9912          1.0014            1.0000         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)             1_803.89       336.94     2_140.83       0.9989          1.0001            1.0000         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)             1_803.89       217.80     2_021.68       0.9909          1.0015            1.0000         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)             1_803.89       348.68     2_152.56       0.9988          1.0001            1.0000         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)             1_803.89       224.35     2_028.24       0.9903          1.0016            1.0000         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)             1_803.89       351.21     2_155.09       0.9986          1.0002            1.0000         3.21
IVF-Binary-256-nl316-pca (self)                        1_803.89       482.99     2_286.88       0.9913          1.0014            1.0000         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)            2_819.99       132.54     2_952.53       0.5925          1.6007            1.4618         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)           2_819.99       145.75     2_965.74       0.5898          1.6229            1.4748         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)           2_819.99       154.10     2_974.09       0.5888          1.6417            1.4806         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)           2_819.99       244.76     3_064.75       0.9972          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)           2_819.99       369.52     3_189.51       0.9993          1.0001            1.0000         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)          2_819.99       255.76     3_075.75       0.9972          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)          2_819.99       383.44     3_203.43       0.9998          1.0000            1.0000         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)          2_819.99       268.08     3_088.07       0.9970          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)          2_819.99       396.04     3_216.03       0.9997          1.0000            1.0000         5.02
IVF-Binary-512-nl158-random (self)                     2_819.99       644.25     3_464.24       0.9974          1.0003            1.0000         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)           1_409.62       137.14     1_546.76       0.5912          1.6104            1.4673         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)           1_409.62       140.20     1_549.82       0.5903          1.6195            1.4718         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)           1_409.62       158.72     1_568.34       0.5890          1.6407            1.4794         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)          1_409.62       251.95     1_661.58       0.9972          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)          1_409.62       376.38     1_786.01       0.9996          1.0001            1.0000         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)          1_409.62       254.81     1_664.44       0.9972          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)          1_409.62       382.96     1_792.58       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)          1_409.62       267.51     1_677.13       0.9969          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)          1_409.62       397.54     1_807.16       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-random (self)                     1_409.62       617.25     2_026.87       0.9974          1.0003            1.0000         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)           1_704.54       143.71     1_848.25       0.5911          1.6075            1.4667         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)           1_704.54       147.31     1_851.85       0.5907          1.6116            1.4696         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)           1_704.54       158.98     1_863.52       0.5897          1.6270            1.4769         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)          1_704.54       279.66     1_984.20       0.9974          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)          1_704.54       392.59     2_097.13       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)          1_704.54       262.01     1_966.55       0.9973          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)          1_704.54       393.50     2_098.04       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)          1_704.54       266.89     1_971.43       0.9971          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)          1_704.54       402.12     2_106.66       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-random (self)                     1_704.54       630.48     2_335.02       0.9976          1.0003            1.0000         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)               3_022.51       128.76     3_151.27       0.6432          1.3824            1.2890         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)              3_022.51       140.99     3_163.50       0.6414          1.3938            1.2945         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)              3_022.51       153.26     3_175.77       0.6407          1.4030            1.2975         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)              3_022.51       246.43     3_268.94       0.9980          1.0003            1.0000         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)              3_022.51       363.04     3_385.55       0.9994          1.0001            1.0000         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)             3_022.51       254.09     3_276.60       0.9982          1.0002            1.0000         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)             3_022.51       381.72     3_404.23       0.9999          1.0000            1.0000         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)             3_022.51       264.96     3_287.47       0.9981          1.0003            1.0000         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)             3_022.51       392.12     3_414.64       0.9998          1.0000            1.0000         5.02
IVF-Binary-512-nl158-pca (self)                        3_022.51       641.60     3_664.11       0.9983          1.0002            1.0000         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_611.36       136.06     1_747.42       0.6420          1.3885            1.2932         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_611.36       140.93     1_752.29       0.6414          1.3935            1.2956         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_611.36       150.29     1_761.65       0.6407          1.4030            1.2985         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_611.36       247.98     1_859.33       0.9982          1.0002            1.0000         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_611.36       375.32     1_986.67       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_611.36       252.07     1_863.43       0.9982          1.0002            1.0000         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_611.36       381.17     1_992.53       0.9998          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_611.36       264.07     1_875.42       0.9981          1.0003            1.0000         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_611.36       407.32     2_018.67       0.9998          1.0000            1.0000         5.21
IVF-Binary-512-nl223-pca (self)                        1_611.36       612.70     2_224.06       0.9983          1.0002            1.0000         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)              1_959.21       143.32     2_102.53       0.6422          1.3876            1.2923         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)              1_959.21       145.63     2_104.84       0.6419          1.3896            1.2933         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)              1_959.21       166.73     2_125.94       0.6413          1.3956            1.2957         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)             1_959.21       255.86     2_215.07       0.9984          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)             1_959.21       385.00     2_344.21       0.9999          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)             1_959.21       256.40     2_215.61       0.9983          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)             1_959.21       392.40     2_351.61       0.9999          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)             1_959.21       270.66     2_229.87       0.9981          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)             1_959.21       405.19     2_364.40       0.9999          1.0000            1.0000         5.48
IVF-Binary-512-nl316-pca (self)                        1_959.21       631.95     2_591.16       0.9984          1.0002            1.0000         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)           3_009.20       208.78     3_217.98       0.6492          1.4402            1.3299         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)          3_009.20       225.80     3_235.00       0.6468          1.4562            1.3410         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)          3_009.20       240.68     3_249.88       0.6457          1.4688            1.3455         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)          3_009.20       326.79     3_336.00       0.9990          1.0002            1.0000         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)          3_009.20       459.46     3_468.66       0.9994          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)         3_009.20       344.71     3_353.91       0.9995          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)         3_009.20       509.31     3_518.51       0.9999          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)         3_009.20       363.09     3_372.29       0.9994          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)         3_009.20       509.20     3_518.40       0.9999          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-random (self)                    3_009.20       967.93     3_977.13       0.9995          1.0001            1.0000         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_616.52       213.99     1_830.52       0.6479          1.4466            1.3357         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_616.52       217.68     1_834.20       0.6472          1.4538            1.3394         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_616.52       235.83     1_852.35       0.6460          1.4684            1.3442         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_616.52       342.04     1_958.57       0.9993          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_616.52       476.50     2_093.02       0.9997          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_616.52       354.56     1_971.08       0.9994          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_616.52       487.37     2_103.89       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_616.52       373.55     1_990.08       0.9994          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_616.52       516.16     2_132.68       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-random (self)                    1_616.52       926.46     2_542.99       0.9995          1.0001            1.0000         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)          1_922.44       224.28     2_146.73       0.6478          1.4461            1.3344        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)          1_922.44       224.49     2_146.93       0.6475          1.4494            1.3361        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)          1_922.44       239.37     2_161.81       0.6465          1.4599            1.3407        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)         1_922.44       376.86     2_299.30       0.9995          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)         1_922.44       489.41     2_411.85       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)         1_922.44       362.04     2_284.48       0.9995          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)         1_922.44       495.03     2_417.47       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)         1_922.44       378.08     2_300.52       0.9994          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)         1_922.44       517.73     2_440.17       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-random (self)                    1_922.44       949.62     2_872.06       0.9995          1.0001            1.0000        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)              3_222.83       215.77     3_438.60       0.6828          1.3187            1.2385         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)             3_222.83       229.36     3_452.19       0.6812          1.3273            1.2437         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)             3_222.83       242.57     3_465.40       0.6805          1.3345            1.2454         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)             3_222.83       327.80     3_550.62       0.9992          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)             3_222.83       457.55     3_680.38       0.9994          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)            3_222.83       348.07     3_570.90       0.9997          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)            3_222.83       491.25     3_714.08       1.0000          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)            3_222.83       374.93     3_597.76       0.9996          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)            3_222.83       511.09     3_733.92       1.0000          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-pca (self)                       3_222.83       968.26     4_191.09       0.9997          1.0000            1.0000         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_841.95       217.21     2_059.16       0.6818          1.3238            1.2403         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_841.95       222.94     2_064.89       0.6813          1.3270            1.2428         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_841.95       240.66     2_082.62       0.6806          1.3344            1.2451         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_841.95       349.80     2_191.76       0.9995          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_841.95       478.11     2_320.07       0.9998          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_841.95       352.41     2_194.36       0.9996          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_841.95       495.43     2_337.38       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_841.95       372.77     2_214.72       0.9996          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_841.95       511.52     2_353.48       1.0000          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-pca (self)                       1_841.95       931.64     2_773.60       0.9997          1.0000            1.0000         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             2_176.28       224.50     2_400.78       0.6817          1.3241            1.2409        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)             2_176.28       226.66     2_402.94       0.6815          1.3255            1.2420        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)             2_176.28       241.50     2_417.78       0.6809          1.3307            1.2441        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)            2_176.28       364.75     2_541.03       0.9997          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)            2_176.28       499.96     2_676.24       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)            2_176.28       384.30     2_560.58       0.9996          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)            2_176.28       502.84     2_679.12       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)            2_176.28       382.55     2_558.83       0.9996          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)            2_176.28       527.76     2_704.04       1.0000          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-pca (self)                       2_176.28       949.11     3_125.39       0.9997          1.0000            1.0000        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)              2_736.75       409.51     3_146.26       0.0573          8.2519            7.4750         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)             2_736.75       425.50     3_162.25       0.0520          9.4871            8.2097         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)             2_736.75       446.98     3_183.73       0.0494         10.3662            8.7466         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)             2_736.75       517.97     3_254.72       0.3103          1.8949            1.7846         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)             2_736.75       987.75     3_724.50       0.4776          1.4825            1.3824         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)            2_736.75       537.34     3_274.09       0.2786          2.0014            1.8766         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)            2_736.75       965.99     3_702.74       0.4285          1.5638            1.4621         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)            2_736.75       548.22     3_284.97       0.2621          2.0692            1.9339         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)            2_736.75       958.56     3_695.32       0.4002          1.6175            1.5078         5.04
IVF-Binary-768-nl158-sign (self)                       2_736.75     1_488.69     4_225.44       0.2910          1.9632            1.8395         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)             1_313.70       442.45     1_756.15       0.0570          8.2117            7.3530         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)             1_313.70       418.61     1_732.32       0.0545          8.6681            7.6901         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)             1_313.70       434.74     1_748.44       0.0514          9.6380            8.3214         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)            1_313.70       516.94     1_830.64       0.3111          1.8746            1.7621         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)            1_313.70       912.11     2_225.82       0.4734          1.4732            1.3870         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)            1_313.70       513.61     1_827.31       0.2972          1.9238            1.8036         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)            1_313.70       929.54     2_243.24       0.4499          1.5110            1.4213         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)            1_313.70       533.20     1_846.90       0.2759          2.0111            1.8835         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)            1_313.70       950.97     2_264.67       0.4143          1.5807            1.4851         5.23
IVF-Binary-768-nl223-sign (self)                       1_313.70     1_459.12     2_772.82       0.3092          1.8880            1.7686         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)             1_599.35       416.89     2_016.23       0.0581          7.9072            7.1781         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)             1_599.35       422.52     2_021.87       0.0568          8.1437            7.3474         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)             1_599.35       440.11     2_039.46       0.0534          8.9543            7.8708         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)            1_599.35       520.83     2_120.18       0.3162          1.8517            1.7465         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)            1_599.35       934.86     2_534.20       0.4808          1.4599            1.3740         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)            1_599.35       517.78     2_117.13       0.3081          1.8776            1.7701         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)            1_599.35       942.71     2_542.06       0.4683          1.4792            1.3936         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)            1_599.35       536.72     2_136.07       0.2865          1.9584            1.8357         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)            1_599.35       980.46     2_579.81       0.4338          1.5372            1.4442         5.51
IVF-Binary-768-nl316-sign (self)                       1_599.35     1_497.10     3_096.45       0.3204          1.8449            1.7293         5.51
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
Exhaustive (query)                                        33.44       693.82       727.27       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.44     2_282.74     2_316.18       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             841.88       196.70     1_038.58       0.5642          1.0369            1.0367         2.95
ExhaustiveRaBitQ-rf5 (query)                             841.88       247.51     1_089.39       0.9246          1.0017            1.0006         2.95
ExhaustiveRaBitQ-rf10 (query)                            841.88       293.15     1_135.03       0.9836          1.0003            1.0000         2.95
ExhaustiveRaBitQ-rf20 (query)                            841.88       366.98     1_208.86       0.9985          1.0000            1.0000         2.95
ExhaustiveRaBitQ (self)                                  841.88       939.66     1_781.54       0.9843          1.0003            1.0000         2.95
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_395.64        91.90     1_487.53       0.5751          1.0343            1.0347         3.04
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_395.64       120.99     1_516.63       0.5751          1.0343            1.0347         3.04
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_395.64       148.45     1_544.09       0.5751          1.0343            1.0347         3.04
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_395.64       172.24     1_567.88       0.9847          1.0003            1.0000         3.04
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_395.64       245.71     1_641.35       0.9986          1.0000            1.0000         3.04
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_395.64       195.71     1_591.35       0.9847          1.0003            1.0000         3.04
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_395.64       263.47     1_659.11       0.9986          1.0000            1.0000         3.04
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_395.64       226.79     1_622.43       0.9847          1.0003            1.0000         3.04
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_395.64       293.64     1_689.28       0.9986          1.0000            1.0000         3.04
IVF-RaBitQ-nl158 (self)                                1_395.64       935.53     2_331.16       0.9987          1.0000            1.0000         3.04
IVF-RaBitQ-nl223-np11-rf0 (query)                        916.95       116.53     1_033.48       0.5864          1.0326            1.0325         3.17
IVF-RaBitQ-nl223-np14-rf0 (query)                        916.95       133.99     1_050.94       0.5864          1.0326            1.0324         3.17
IVF-RaBitQ-nl223-np21-rf0 (query)                        916.95       172.61     1_089.56       0.5864          1.0326            1.0324         3.17
IVF-RaBitQ-nl223-np11-rf10 (query)                       916.95       187.64     1_104.59       0.9878          1.0002            1.0000         3.17
IVF-RaBitQ-nl223-np11-rf20 (query)                       916.95       248.80     1_165.75       0.9989          1.0000            1.0000         3.17
IVF-RaBitQ-nl223-np14-rf10 (query)                       916.95       206.06     1_123.01       0.9879          1.0002            1.0000         3.17
IVF-RaBitQ-nl223-np14-rf20 (query)                       916.95       264.35     1_181.31       0.9990          1.0000            1.0000         3.17
IVF-RaBitQ-nl223-np21-rf10 (query)                       916.95       272.75     1_189.71       0.9879          1.0002            1.0000         3.17
IVF-RaBitQ-nl223-np21-rf20 (query)                       916.95       303.05     1_220.01       0.9990          1.0000            1.0000         3.17
IVF-RaBitQ-nl223 (self)                                  916.95     1_028.27     1_945.23       0.9990          1.0000            1.0000         3.17
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_123.75       142.71     1_266.46       0.5949          1.0309            1.0310         3.35
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_123.75       151.75     1_275.50       0.5949          1.0309            1.0310         3.35
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_123.75       193.22     1_316.97       0.5950          1.0308            1.0310         3.35
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_123.75       210.53     1_334.29       0.9888          1.0002            1.0000         3.35
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_123.75       273.46     1_397.21       0.9990          1.0001            1.0000         3.35
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_123.75       219.77     1_343.53       0.9889          1.0002            1.0000         3.35
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_123.75       279.99     1_403.74       0.9991          1.0000            1.0000         3.35
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_123.75       265.69     1_389.44       0.9890          1.0002            1.0000         3.35
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_123.75       322.76     1_446.51       0.9992          1.0000            1.0000         3.35
IVF-RaBitQ-nl316 (self)                                1_123.75     1_039.81     2_163.56       0.9992          1.0000            1.0000         3.35
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
Exhaustive (query)                                        68.87     1_315.58     1_384.45       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.87     4_441.47     4_510.35       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           2_058.33       354.20     2_412.53       0.5733          1.0234            1.0234         5.44
ExhaustiveRaBitQ-rf5 (query)                           2_058.33       420.47     2_478.79       0.9223          1.0012            1.0005         5.44
ExhaustiveRaBitQ-rf10 (query)                          2_058.33       475.17     2_533.50       0.9828          1.0002            1.0000         5.44
ExhaustiveRaBitQ-rf20 (query)                          2_058.33       579.71     2_638.04       0.9984          1.0000            1.0000         5.44
ExhaustiveRaBitQ (self)                                2_058.33     1_510.10     3_568.43       0.9831          1.0002            1.0000         5.44
IVF-RaBitQ-nl158-np7-rf0 (query)                       3_125.57       171.38     3_296.95       0.5827          1.0219            1.0223         5.63
IVF-RaBitQ-nl158-np12-rf0 (query)                      3_125.57       221.07     3_346.63       0.5827          1.0219            1.0223         5.63
IVF-RaBitQ-nl158-np17-rf0 (query)                      3_125.57       274.95     3_400.51       0.5827          1.0219            1.0223         5.63
IVF-RaBitQ-nl158-np7-rf10 (query)                      3_125.57       282.87     3_408.44       0.9836          1.0002            1.0000         5.63
IVF-RaBitQ-nl158-np7-rf20 (query)                      3_125.57       377.65     3_503.21       0.9985          1.0000            1.0000         5.63
IVF-RaBitQ-nl158-np12-rf10 (query)                     3_125.57       330.45     3_456.02       0.9836          1.0002            1.0000         5.63
IVF-RaBitQ-nl158-np12-rf20 (query)                     3_125.57       427.67     3_553.23       0.9985          1.0000            1.0000         5.63
IVF-RaBitQ-nl158-np17-rf10 (query)                     3_125.57       383.82     3_509.39       0.9836          1.0002            1.0000         5.63
IVF-RaBitQ-nl158-np17-rf20 (query)                     3_125.57       478.36     3_603.92       0.9985          1.0000            1.0000         5.63
IVF-RaBitQ-nl158 (self)                                3_125.57     1_519.89     4_645.46       0.9986          1.0000            1.0000         5.63
IVF-RaBitQ-nl223-np11-rf0 (query)                      2_221.51       225.03     2_446.54       0.5910          1.0210            1.0212         5.88
IVF-RaBitQ-nl223-np14-rf0 (query)                      2_221.51       251.60     2_473.11       0.5910          1.0210            1.0212         5.88
IVF-RaBitQ-nl223-np21-rf0 (query)                      2_221.51       323.16     2_544.67       0.5910          1.0210            1.0212         5.88
IVF-RaBitQ-nl223-np11-rf10 (query)                     2_221.51       326.53     2_548.03       0.9862          1.0002            1.0000         5.88
IVF-RaBitQ-nl223-np11-rf20 (query)                     2_221.51       413.13     2_634.63       0.9988          1.0000            1.0000         5.88
IVF-RaBitQ-nl223-np14-rf10 (query)                     2_221.51       351.99     2_573.50       0.9862          1.0001            1.0000         5.88
IVF-RaBitQ-nl223-np14-rf20 (query)                     2_221.51       441.77     2_663.28       0.9989          1.0000            1.0000         5.88
IVF-RaBitQ-nl223-np21-rf10 (query)                     2_221.51       444.36     2_665.87       0.9862          1.0001            1.0000         5.88
IVF-RaBitQ-nl223-np21-rf20 (query)                     2_221.51       512.21     2_733.71       0.9989          1.0000            1.0000         5.88
IVF-RaBitQ-nl223 (self)                                2_221.51     1_640.82     3_862.33       0.9988          1.0000            1.0000         5.88
IVF-RaBitQ-nl316-np15-rf0 (query)                      2_414.10       266.82     2_680.92       0.5965          1.0201            1.0206         6.24
IVF-RaBitQ-nl316-np17-rf0 (query)                      2_414.10       291.25     2_705.35       0.5965          1.0201            1.0206         6.24
IVF-RaBitQ-nl316-np25-rf0 (query)                      2_414.10       359.96     2_774.06       0.5965          1.0201            1.0206         6.24
IVF-RaBitQ-nl316-np15-rf10 (query)                     2_414.10       364.04     2_778.14       0.9868          1.0001            1.0000         6.24
IVF-RaBitQ-nl316-np15-rf20 (query)                     2_414.10       455.86     2_869.96       0.9989          1.0000            1.0000         6.24
IVF-RaBitQ-nl316-np17-rf10 (query)                     2_414.10       381.54     2_795.64       0.9868          1.0001            1.0000         6.24
IVF-RaBitQ-nl316-np17-rf20 (query)                     2_414.10       478.26     2_892.36       0.9989          1.0000            1.0000         6.24
IVF-RaBitQ-nl316-np25-rf10 (query)                     2_414.10       463.95     2_878.05       0.9868          1.0001            1.0000         6.24
IVF-RaBitQ-nl316-np25-rf20 (query)                     2_414.10       547.51     2_961.61       0.9989          1.0000            1.0000         6.24
IVF-RaBitQ-nl316 (self)                                2_414.10     1_761.37     4_175.47       0.9989          1.0000            1.0000         6.24
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
Exhaustive (query)                                       100.72     1_945.47     2_046.19       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.72     6_556.31     6_657.03       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           3_883.52       529.25     4_412.77       0.5749          1.0183            1.0184         8.44
ExhaustiveRaBitQ-rf5 (query)                           3_883.52       603.22     4_486.74       0.9209          1.0009            1.0004         8.44
ExhaustiveRaBitQ-rf10 (query)                          3_883.52       674.60     4_558.12       0.9818          1.0002            1.0000         8.44
ExhaustiveRaBitQ-rf20 (query)                          3_883.52       795.59     4_679.11       0.9981          1.0000            1.0000         8.44
ExhaustiveRaBitQ (self)                                3_883.52     2_134.89     6_018.41       0.9823          1.0002            1.0000         8.44
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_396.03       266.81     5_662.84       0.5835          1.0171            1.0175         8.71
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_396.03       345.28     5_741.30       0.5835          1.0171            1.0175         8.71
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_396.03       416.43     5_812.45       0.5835          1.0171            1.0175         8.71
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_396.03       396.67     5_792.69       0.9835          1.0001            1.0000         8.71
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_396.03       525.13     5_921.16       0.9984          1.0000            1.0000         8.71
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_396.03       479.52     5_875.54       0.9835          1.0001            1.0000         8.71
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_396.03       599.33     5_995.35       0.9984          1.0000            1.0000         8.71
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_396.03       539.35     5_935.38       0.9835          1.0001            1.0000         8.71
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_396.03       661.11     6_057.14       0.9984          1.0000            1.0000         8.71
IVF-RaBitQ-nl158 (self)                                5_396.03     2_112.22     7_508.24       0.9984          1.0000            1.0000         8.71
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_830.60       341.20     4_171.81       0.5840          1.0174            1.0175         9.09
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_830.60       386.71     4_217.31       0.5840          1.0174            1.0175         9.09
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_830.60       492.00     4_322.60       0.5841          1.0174            1.0175         9.09
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_830.60       466.89     4_297.49       0.9830          1.0002            1.0000         9.09
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_830.60       581.32     4_411.92       0.9982          1.0000            1.0000         9.09
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_830.60       508.54     4_339.14       0.9831          1.0002            1.0000         9.09
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_830.60       630.63     4_461.23       0.9983          1.0000            1.0000         9.09
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_830.60       610.73     4_441.33       0.9831          1.0002            1.0000         9.09
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_830.60       734.81     4_565.41       0.9983          1.0000            1.0000         9.09
IVF-RaBitQ-nl223 (self)                                3_830.60     2_358.68     6_189.28       0.9984          1.0000            1.0000         9.09
IVF-RaBitQ-nl316-np15-rf0 (query)                      4_324.64       407.08     4_731.72       0.5950          1.0161            1.0165         9.64
IVF-RaBitQ-nl316-np17-rf0 (query)                      4_324.64       435.84     4_760.48       0.5950          1.0161            1.0165         9.64
IVF-RaBitQ-nl316-np25-rf0 (query)                      4_324.64       548.65     4_873.29       0.5950          1.0161            1.0165         9.64
IVF-RaBitQ-nl316-np15-rf10 (query)                     4_324.64       528.43     4_853.07       0.9857          1.0001            1.0000         9.64
IVF-RaBitQ-nl316-np15-rf20 (query)                     4_324.64       644.71     4_969.35       0.9987          1.0000            1.0000         9.64
IVF-RaBitQ-nl316-np17-rf10 (query)                     4_324.64       554.35     4_878.99       0.9857          1.0001            1.0000         9.64
IVF-RaBitQ-nl316-np17-rf20 (query)                     4_324.64       674.84     4_999.49       0.9987          1.0000            1.0000         9.64
IVF-RaBitQ-nl316-np25-rf10 (query)                     4_324.64       684.58     5_009.22       0.9857          1.0001            1.0000         9.64
IVF-RaBitQ-nl316-np25-rf20 (query)                     4_324.64       787.00     5_111.64       0.9987          1.0000            1.0000         9.64
IVF-RaBitQ-nl316 (self)                                4_324.64     2_552.49     6_877.14       0.9987          1.0000            1.0000         9.64
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
Exhaustive (query)                                        33.43       726.84       760.27       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.43     2_362.28     2_395.70       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             947.67       249.07     1_196.75       0.7343          1.0241            1.0229         2.95
ExhaustiveRaBitQ-rf5 (query)                             947.67       302.32     1_249.99       0.9975          1.0001            1.0000         2.95
ExhaustiveRaBitQ-rf10 (query)                            947.67       350.23     1_297.90       0.9999          1.0000            1.0000         2.95
ExhaustiveRaBitQ-rf20 (query)                            947.67       438.59     1_386.26       1.0000          1.0000            1.0000         2.95
ExhaustiveRaBitQ (self)                                  947.67     1_132.84     2_080.52       1.0000          1.0000            1.0000         2.95
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_362.79        95.88     1_458.67       0.7358          1.0239            1.0227         3.04
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_362.79       123.60     1_486.39       0.7358          1.0239            1.0227         3.04
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_362.79       164.29     1_527.08       0.7358          1.0239            1.0227         3.04
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_362.79       179.26     1_542.05       1.0000          1.0000            1.0000         3.04
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_362.79       240.19     1_602.97       1.0000          1.0000            1.0000         3.04
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_362.79       202.65     1_565.44       1.0000          1.0000            1.0000         3.04
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_362.79       272.48     1_635.27       1.0000          1.0000            1.0000         3.04
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_362.79       235.53     1_598.31       1.0000          1.0000            1.0000         3.04
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_362.79       301.72     1_664.50       1.0000          1.0000            1.0000         3.04
IVF-RaBitQ-nl158 (self)                                1_362.79       984.51     2_347.30       1.0000          1.0000            1.0000         3.04
IVF-RaBitQ-nl223-np11-rf0 (query)                        980.35       124.25     1_104.61       0.7404          1.0229            1.0220         3.17
IVF-RaBitQ-nl223-np14-rf0 (query)                        980.35       140.14     1_120.49       0.7404          1.0229            1.0220         3.17
IVF-RaBitQ-nl223-np21-rf0 (query)                        980.35       183.87     1_164.22       0.7404          1.0229            1.0220         3.17
IVF-RaBitQ-nl223-np11-rf10 (query)                       980.35       217.71     1_198.06       1.0000          1.0000            1.0000         3.17
IVF-RaBitQ-nl223-np11-rf20 (query)                       980.35       264.73     1_245.08       1.0000          1.0000            1.0000         3.17
IVF-RaBitQ-nl223-np14-rf10 (query)                       980.35       214.55     1_194.90       1.0000          1.0000            1.0000         3.17
IVF-RaBitQ-nl223-np14-rf20 (query)                       980.35       279.38     1_259.73       1.0000          1.0000            1.0000         3.17
IVF-RaBitQ-nl223-np21-rf10 (query)                       980.35       256.48     1_236.83       1.0000          1.0000            1.0000         3.17
IVF-RaBitQ-nl223-np21-rf20 (query)                       980.35       323.05     1_303.40       1.0000          1.0000            1.0000         3.17
IVF-RaBitQ-nl223 (self)                                  980.35     1_031.81     2_012.16       1.0000          1.0000            1.0000         3.17
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_186.09       142.12     1_328.21       0.7444          1.0222            1.0211         3.35
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_186.09       157.76     1_343.85       0.7444          1.0222            1.0211         3.35
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_186.09       199.11     1_385.20       0.7444          1.0222            1.0211         3.35
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_186.09       220.43     1_406.52       1.0000          1.0000            1.0000         3.35
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_186.09       283.45     1_469.54       1.0000          1.0000            1.0000         3.35
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_186.09       228.58     1_414.67       1.0000          1.0000            1.0000         3.35
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_186.09       294.44     1_480.53       1.0000          1.0000            1.0000         3.35
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_186.09       274.92     1_461.01       1.0000          1.0000            1.0000         3.35
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_186.09       341.49     1_527.58       1.0000          1.0000            1.0000         3.35
IVF-RaBitQ-nl316 (self)                                1_186.09     1_093.98     2_280.07       1.0000          1.0000            1.0000         3.35
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
Exhaustive (query)                                        68.43     1_330.41     1_398.84       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.43     4_491.83     4_560.26       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           2_257.08       427.83     2_684.91       0.7491          1.0144            1.0138         5.44
ExhaustiveRaBitQ-rf5 (query)                           2_257.08       490.42     2_747.50       0.9979          1.0000            1.0000         5.44
ExhaustiveRaBitQ-rf10 (query)                          2_257.08       539.40     2_796.48       0.9999          1.0000            1.0000         5.44
ExhaustiveRaBitQ-rf20 (query)                          2_257.08       648.49     2_905.57       1.0000          1.0000            1.0000         5.44
ExhaustiveRaBitQ (self)                                2_257.08     1_720.73     3_977.81       1.0000          1.0000            1.0000         5.44
IVF-RaBitQ-nl158-np7-rf0 (query)                       3_119.08       177.94     3_297.03       0.7509          1.0141            1.0135         5.63
IVF-RaBitQ-nl158-np12-rf0 (query)                      3_119.08       231.25     3_350.33       0.7509          1.0141            1.0135         5.63
IVF-RaBitQ-nl158-np17-rf0 (query)                      3_119.08       287.19     3_406.27       0.7509          1.0141            1.0135         5.63
IVF-RaBitQ-nl158-np7-rf10 (query)                      3_119.08       285.91     3_404.99       1.0000          1.0000            1.0000         5.63
IVF-RaBitQ-nl158-np7-rf20 (query)                      3_119.08       382.57     3_501.65       1.0000          1.0000            1.0000         5.63
IVF-RaBitQ-nl158-np12-rf10 (query)                     3_119.08       338.29     3_457.37       1.0000          1.0000            1.0000         5.63
IVF-RaBitQ-nl158-np12-rf20 (query)                     3_119.08       435.49     3_554.57       1.0000          1.0000            1.0000         5.63
IVF-RaBitQ-nl158-np17-rf10 (query)                     3_119.08       390.58     3_509.66       1.0000          1.0000            1.0000         5.63
IVF-RaBitQ-nl158-np17-rf20 (query)                     3_119.08       490.16     3_609.24       1.0000          1.0000            1.0000         5.63
IVF-RaBitQ-nl158 (self)                                3_119.08     1_551.05     4_670.13       1.0000          1.0000            1.0000         5.63
IVF-RaBitQ-nl223-np11-rf0 (query)                      2_201.85       229.52     2_431.38       0.7535          1.0138            1.0132         5.88
IVF-RaBitQ-nl223-np14-rf0 (query)                      2_201.85       254.44     2_456.30       0.7537          1.0138            1.0132         5.88
IVF-RaBitQ-nl223-np21-rf0 (query)                      2_201.85       337.01     2_538.86       0.7537          1.0138            1.0132         5.88
IVF-RaBitQ-nl223-np11-rf10 (query)                     2_201.85       329.72     2_531.57       0.9994          1.0000            1.0000         5.88
IVF-RaBitQ-nl223-np11-rf20 (query)                     2_201.85       425.24     2_627.10       0.9994          1.0000            1.0000         5.88
IVF-RaBitQ-nl223-np14-rf10 (query)                     2_201.85       361.71     2_563.57       1.0000          1.0000            1.0000         5.88
IVF-RaBitQ-nl223-np14-rf20 (query)                     2_201.85       459.13     2_660.98       1.0000          1.0000            1.0000         5.88
IVF-RaBitQ-nl223-np21-rf10 (query)                     2_201.85       437.55     2_639.41       1.0000          1.0000            1.0000         5.88
IVF-RaBitQ-nl223-np21-rf20 (query)                     2_201.85       536.76     2_738.61       1.0000          1.0000            1.0000         5.88
IVF-RaBitQ-nl223 (self)                                2_201.85     1_706.73     3_908.58       1.0000          1.0000            1.0000         5.88
IVF-RaBitQ-nl316-np15-rf0 (query)                      2_636.77       279.25     2_916.03       0.7549          1.0136            1.0130         6.24
IVF-RaBitQ-nl316-np17-rf0 (query)                      2_636.77       290.41     2_927.18       0.7549          1.0136            1.0130         6.24
IVF-RaBitQ-nl316-np25-rf0 (query)                      2_636.77       373.85     3_010.63       0.7549          1.0136            1.0130         6.24
IVF-RaBitQ-nl316-np15-rf10 (query)                     2_636.77       374.29     3_011.07       1.0000          1.0000            1.0000         6.24
IVF-RaBitQ-nl316-np15-rf20 (query)                     2_636.77       472.38     3_109.16       1.0000          1.0000            1.0000         6.24
IVF-RaBitQ-nl316-np17-rf10 (query)                     2_636.77       393.81     3_030.58       1.0000          1.0000            1.0000         6.24
IVF-RaBitQ-nl316-np17-rf20 (query)                     2_636.77       494.74     3_131.51       1.0000          1.0000            1.0000         6.24
IVF-RaBitQ-nl316-np25-rf10 (query)                     2_636.77       480.77     3_117.54       1.0000          1.0000            1.0000         6.24
IVF-RaBitQ-nl316-np25-rf20 (query)                     2_636.77       574.42     3_211.20       1.0000          1.0000            1.0000         6.24
IVF-RaBitQ-nl316 (self)                                2_636.77     1_902.77     4_539.54       1.0000          1.0000            1.0000         6.24
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
Exhaustive (query)                                       100.22     1_886.55     1_986.77       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.22     6_296.71     6_396.93       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           3_978.68       617.69     4_596.37       0.7295          1.0119            1.0114         8.44
ExhaustiveRaBitQ-rf5 (query)                           3_978.68       702.31     4_680.99       0.9957          1.0000            1.0000         8.44
ExhaustiveRaBitQ-rf10 (query)                          3_978.68       774.49     4_753.17       0.9999          1.0000            1.0000         8.44
ExhaustiveRaBitQ-rf20 (query)                          3_978.68       917.16     4_895.84       1.0000          1.0000            1.0000         8.44
ExhaustiveRaBitQ (self)                                3_978.68     2_488.01     6_466.69       1.0000          1.0000            1.0000         8.44
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_256.83       277.02     5_533.85       0.7328          1.0116            1.0112         8.71
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_256.83       350.82     5_607.64       0.7328          1.0116            1.0112         8.71
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_256.83       445.62     5_702.45       0.7328          1.0116            1.0112         8.71
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_256.83       410.76     5_667.58       0.9999          1.0000            1.0000         8.71
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_256.83       531.02     5_787.84       1.0000          1.0000            1.0000         8.71
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_256.83       478.02     5_734.85       0.9999          1.0000            1.0000         8.71
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_256.83       611.55     5_868.38       1.0000          1.0000            1.0000         8.71
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_256.83       558.08     5_814.91       0.9999          1.0000            1.0000         8.71
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_256.83       683.37     5_940.19       1.0000          1.0000            1.0000         8.71
IVF-RaBitQ-nl158 (self)                                5_256.83     2_187.84     7_444.66       1.0000          1.0000            1.0000         8.71
IVF-RaBitQ-nl223-np11-rf0 (query)                      4_010.65       355.75     4_366.40       0.7340          1.0114            1.0109         9.09
IVF-RaBitQ-nl223-np14-rf0 (query)                      4_010.65       392.74     4_403.39       0.7340          1.0114            1.0109         9.09
IVF-RaBitQ-nl223-np21-rf0 (query)                      4_010.65       529.35     4_540.01       0.7340          1.0114            1.0109         9.09
IVF-RaBitQ-nl223-np11-rf10 (query)                     4_010.65       470.71     4_481.37       0.9999          1.0000            1.0000         9.09
IVF-RaBitQ-nl223-np11-rf20 (query)                     4_010.65       599.88     4_610.54       1.0000          1.0000            1.0000         9.09
IVF-RaBitQ-nl223-np14-rf10 (query)                     4_010.65       515.23     4_525.88       0.9999          1.0000            1.0000         9.09
IVF-RaBitQ-nl223-np14-rf20 (query)                     4_010.65       644.61     4_655.26       1.0000          1.0000            1.0000         9.09
IVF-RaBitQ-nl223-np21-rf10 (query)                     4_010.65       626.99     4_637.65       0.9999          1.0000            1.0000         9.09
IVF-RaBitQ-nl223-np21-rf20 (query)                     4_010.65       749.08     4_759.74       1.0000          1.0000            1.0000         9.09
IVF-RaBitQ-nl223 (self)                                4_010.65     2_448.66     6_459.32       1.0000          1.0000            1.0000         9.09
IVF-RaBitQ-nl316-np15-rf0 (query)                      4_522.43       419.70     4_942.13       0.7352          1.0113            1.0109         9.64
IVF-RaBitQ-nl316-np17-rf0 (query)                      4_522.43       444.73     4_967.16       0.7352          1.0113            1.0109         9.64
IVF-RaBitQ-nl316-np25-rf0 (query)                      4_522.43       563.25     5_085.68       0.7352          1.0113            1.0109         9.64
IVF-RaBitQ-nl316-np15-rf10 (query)                     4_522.43       539.00     5_061.43       0.9999          1.0000            1.0000         9.64
IVF-RaBitQ-nl316-np15-rf20 (query)                     4_522.43       662.98     5_185.41       1.0000          1.0000            1.0000         9.64
IVF-RaBitQ-nl316-np17-rf10 (query)                     4_522.43       568.13     5_090.56       0.9999          1.0000            1.0000         9.64
IVF-RaBitQ-nl316-np17-rf20 (query)                     4_522.43       699.09     5_221.52       1.0000          1.0000            1.0000         9.64
IVF-RaBitQ-nl316-np25-rf10 (query)                     4_522.43       691.46     5_213.89       0.9999          1.0000            1.0000         9.64
IVF-RaBitQ-nl316-np25-rf20 (query)                     4_522.43       823.25     5_345.68       1.0000          1.0000            1.0000         9.64
IVF-RaBitQ-nl316 (self)                                4_522.43     2_695.89     7_218.32       1.0000          1.0000            1.0000         9.64
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
Exhaustive (query)                                        32.81       712.30       745.11       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.81     2_366.35     2_399.16       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                           1_017.84       305.56     1_323.39       0.8680          1.0296            1.0242         2.95
ExhaustiveRaBitQ-rf5 (query)                           1_017.84       371.70     1_389.54       1.0000          1.0000            1.0000         2.95
ExhaustiveRaBitQ-rf10 (query)                          1_017.84       423.26     1_441.10       1.0000          1.0000            1.0000         2.95
ExhaustiveRaBitQ-rf20 (query)                          1_017.84       536.39     1_554.22       1.0000          1.0000            1.0000         2.95
ExhaustiveRaBitQ (self)                                1_017.84     1_387.48     2_405.32       1.0000          1.0000            1.0000         2.95
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_467.61       108.11     1_575.71       0.8728          1.0278            1.0225         3.04
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_467.61       157.56     1_625.17       0.8733          1.0275            1.0223         3.04
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_467.61       209.57     1_677.17       0.8733          1.0275            1.0223         3.04
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_467.61       194.69     1_662.30       0.9976          1.0005            1.0000         3.04
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_467.61       261.31     1_728.91       0.9976          1.0005            1.0000         3.04
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_467.61       242.52     1_710.13       0.9999          1.0000            1.0000         3.04
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_467.61       316.34     1_783.94       0.9999          1.0000            1.0000         3.04
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_467.61       290.69     1_758.30       1.0000          1.0000            1.0000         3.04
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_467.61       398.82     1_866.42       1.0000          1.0000            1.0000         3.04
IVF-RaBitQ-nl158 (self)                                1_467.61     1_202.04     2_669.64       1.0000          1.0000            1.0000         3.04
IVF-RaBitQ-nl223-np11-rf0 (query)                        840.43       133.78       974.21       0.8833          1.0228            1.0186         3.17
IVF-RaBitQ-nl223-np14-rf0 (query)                        840.43       154.14       994.57       0.8834          1.0227            1.0186         3.17
IVF-RaBitQ-nl223-np21-rf0 (query)                        840.43       212.00     1_052.43       0.8833          1.0228            1.0186         3.17
IVF-RaBitQ-nl223-np11-rf10 (query)                       840.43       206.09     1_046.52       0.9994          1.0001            1.0000         3.17
IVF-RaBitQ-nl223-np11-rf20 (query)                       840.43       278.78     1_119.21       0.9994          1.0001            1.0000         3.17
IVF-RaBitQ-nl223-np14-rf10 (query)                       840.43       254.80     1_095.24       0.9999          1.0000            1.0000         3.17
IVF-RaBitQ-nl223-np14-rf20 (query)                       840.43       301.04     1_141.47       0.9999          1.0000            1.0000         3.17
IVF-RaBitQ-nl223-np21-rf10 (query)                       840.43       291.98     1_132.41       1.0000          1.0000            1.0000         3.17
IVF-RaBitQ-nl223-np21-rf20 (query)                       840.43       364.96     1_205.39       1.0000          1.0000            1.0000         3.17
IVF-RaBitQ-nl223 (self)                                  840.43     1_175.28     2_015.72       1.0000          1.0000            1.0000         3.17
IVF-RaBitQ-nl316-np15-rf0 (query)                        993.14       154.19     1_147.33       0.8893          1.0202            1.0165         3.35
IVF-RaBitQ-nl316-np17-rf0 (query)                        993.14       167.84     1_160.98       0.8893          1.0202            1.0165         3.35
IVF-RaBitQ-nl316-np25-rf0 (query)                        993.14       228.41     1_221.54       0.8893          1.0202            1.0165         3.35
IVF-RaBitQ-nl316-np15-rf10 (query)                       993.14       228.01     1_221.15       0.9997          1.0000            1.0000         3.35
IVF-RaBitQ-nl316-np15-rf20 (query)                       993.14       299.99     1_293.13       0.9997          1.0000            1.0000         3.35
IVF-RaBitQ-nl316-np17-rf10 (query)                       993.14       242.64     1_235.78       0.9998          1.0000            1.0000         3.35
IVF-RaBitQ-nl316-np17-rf20 (query)                       993.14       314.98     1_308.12       0.9998          1.0000            1.0000         3.35
IVF-RaBitQ-nl316-np25-rf10 (query)                       993.14       301.51     1_294.65       1.0000          1.0000            1.0000         3.35
IVF-RaBitQ-nl316-np25-rf20 (query)                       993.14       373.95     1_367.09       1.0000          1.0000            1.0000         3.35
IVF-RaBitQ-nl316 (self)                                  993.14     1_212.43     2_205.56       1.0000          1.0000            1.0000         3.35
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
Exhaustive (query)                                        68.80     1_353.24     1_422.04       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.80     4_595.05     4_663.85       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           2_481.73       518.36     3_000.09       0.9024          1.0153            1.0116         5.44
ExhaustiveRaBitQ-rf5 (query)                           2_481.73       600.86     3_082.59       1.0000          1.0000            1.0000         5.44
ExhaustiveRaBitQ-rf10 (query)                          2_481.73       644.86     3_126.59       1.0000          1.0000            1.0000         5.44
ExhaustiveRaBitQ-rf20 (query)                          2_481.73       775.30     3_257.03       1.0000          1.0000            1.0000         5.44
ExhaustiveRaBitQ (self)                                2_481.73     2_082.80     4_564.53       1.0000          1.0000            1.0000         5.44
IVF-RaBitQ-nl158-np7-rf0 (query)                       3_378.36       195.15     3_573.52       0.9068          1.0138            1.0103         5.63
IVF-RaBitQ-nl158-np12-rf0 (query)                      3_378.36       279.65     3_658.01       0.9073          1.0135            1.0101         5.63
IVF-RaBitQ-nl158-np17-rf0 (query)                      3_378.36       358.59     3_736.95       0.9073          1.0135            1.0101         5.63
IVF-RaBitQ-nl158-np7-rf10 (query)                      3_378.36       304.09     3_682.45       0.9986          1.0003            1.0000         5.63
IVF-RaBitQ-nl158-np7-rf20 (query)                      3_378.36       406.38     3_784.74       0.9986          1.0003            1.0000         5.63
IVF-RaBitQ-nl158-np12-rf10 (query)                     3_378.36       382.96     3_761.33       0.9999          1.0000            1.0000         5.63
IVF-RaBitQ-nl158-np12-rf20 (query)                     3_378.36       490.02     3_868.39       0.9999          1.0000            1.0000         5.63
IVF-RaBitQ-nl158-np17-rf10 (query)                     3_378.36       468.88     3_847.24       1.0000          1.0000            1.0000         5.63
IVF-RaBitQ-nl158-np17-rf20 (query)                     3_378.36       600.79     3_979.16       1.0000          1.0000            1.0000         5.63
IVF-RaBitQ-nl158 (self)                                3_378.36     1_881.24     5_259.60       1.0000          1.0000            1.0000         5.63
IVF-RaBitQ-nl223-np11-rf0 (query)                      2_040.66       244.47     2_285.13       0.9151          1.0111            1.0083         5.88
IVF-RaBitQ-nl223-np14-rf0 (query)                      2_040.66       286.60     2_327.26       0.9152          1.0111            1.0083         5.88
IVF-RaBitQ-nl223-np21-rf0 (query)                      2_040.66       383.05     2_423.71       0.9152          1.0111            1.0083         5.88
IVF-RaBitQ-nl223-np11-rf10 (query)                     2_040.66       341.63     2_382.29       0.9997          1.0000            1.0000         5.88
IVF-RaBitQ-nl223-np11-rf20 (query)                     2_040.66       439.55     2_480.20       0.9997          1.0000            1.0000         5.88
IVF-RaBitQ-nl223-np14-rf10 (query)                     2_040.66       381.92     2_422.58       1.0000          1.0000            1.0000         5.88
IVF-RaBitQ-nl223-np14-rf20 (query)                     2_040.66       480.38     2_521.03       1.0000          1.0000            1.0000         5.88
IVF-RaBitQ-nl223-np21-rf10 (query)                     2_040.66       479.43     2_520.08       1.0000          1.0000            1.0000         5.88
IVF-RaBitQ-nl223-np21-rf20 (query)                     2_040.66       582.99     2_623.65       1.0000          1.0000            1.0000         5.88
IVF-RaBitQ-nl223 (self)                                2_040.66     1_861.55     3_902.21       1.0000          1.0000            1.0000         5.88
IVF-RaBitQ-nl316-np15-rf0 (query)                      2_237.54       288.64     2_526.18       0.9190          1.0100            1.0073         6.24
IVF-RaBitQ-nl316-np17-rf0 (query)                      2_237.54       311.96     2_549.50       0.9190          1.0100            1.0073         6.24
IVF-RaBitQ-nl316-np25-rf0 (query)                      2_237.54       417.56     2_655.10       0.9190          1.0100            1.0073         6.24
IVF-RaBitQ-nl316-np15-rf10 (query)                     2_237.54       389.04     2_626.58       0.9999          1.0000            1.0000         6.24
IVF-RaBitQ-nl316-np15-rf20 (query)                     2_237.54       485.25     2_722.79       0.9999          1.0000            1.0000         6.24
IVF-RaBitQ-nl316-np17-rf10 (query)                     2_237.54       410.26     2_647.80       0.9999          1.0000            1.0000         6.24
IVF-RaBitQ-nl316-np17-rf20 (query)                     2_237.54       512.17     2_749.71       0.9999          1.0000            1.0000         6.24
IVF-RaBitQ-nl316-np25-rf10 (query)                     2_237.54       513.05     2_750.59       1.0000          1.0000            1.0000         6.24
IVF-RaBitQ-nl316-np25-rf20 (query)                     2_237.54       613.85     2_851.40       1.0000          1.0000            1.0000         6.24
IVF-RaBitQ-nl316 (self)                                2_237.54     1_987.28     4_224.82       1.0000          1.0000            1.0000         6.24
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
Exhaustive (query)                                       100.47     1_888.85     1_989.33       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.47     6_342.23     6_442.70       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           4_391.60       707.57     5_099.17       0.9249          1.0085            1.0061         8.44
ExhaustiveRaBitQ-rf5 (query)                           4_391.60       798.37     5_189.98       1.0000          1.0000            1.0000         8.44
ExhaustiveRaBitQ-rf10 (query)                          4_391.60       908.98     5_300.59       1.0000          1.0000            1.0000         8.44
ExhaustiveRaBitQ-rf20 (query)                          4_391.60     1_024.73     5_416.34       1.0000          1.0000            1.0000         8.44
ExhaustiveRaBitQ (self)                                4_391.60     2_913.97     7_305.57       1.0000          1.0000            1.0000         8.44
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_728.75       301.06     6_029.81       0.9274          1.0078            1.0055         8.71
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_728.75       417.22     6_145.96       0.9276          1.0078            1.0055         8.71
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_728.75       529.61     6_258.36       0.9276          1.0078            1.0055         8.71
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_728.75       425.93     6_154.68       0.9995          1.0001            1.0000         8.71
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_728.75       547.26     6_276.01       0.9995          1.0001            1.0000         8.71
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_728.75       538.07     6_266.82       1.0000          1.0000            1.0000         8.71
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_728.75       666.23     6_394.97       1.0000          1.0000            1.0000         8.71
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_728.75       651.36     6_380.11       1.0000          1.0000            1.0000         8.71
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_728.75       770.75     6_499.49       1.0000          1.0000            1.0000         8.71
IVF-RaBitQ-nl158 (self)                                5_728.75     2_518.08     8_246.82       1.0000          1.0000            1.0000         8.71
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_792.42       369.61     4_162.03       0.9323          1.0067            1.0046         9.09
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_792.42       428.75     4_221.18       0.9323          1.0067            1.0046         9.09
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_792.42       569.83     4_362.26       0.9323          1.0067            1.0046         9.09
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_792.42       492.95     4_285.38       0.9999          1.0000            1.0000         9.09
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_792.42       619.35     4_411.77       0.9999          1.0000            1.0000         9.09
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_792.42       549.71     4_342.14       1.0000          1.0000            1.0000         9.09
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_792.42       678.74     4_471.17       1.0000          1.0000            1.0000         9.09
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_792.42       693.21     4_485.64       1.0000          1.0000            1.0000         9.09
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_792.42       814.25     4_606.68       1.0000          1.0000            1.0000         9.09
IVF-RaBitQ-nl223 (self)                                3_792.42     2_640.91     6_433.33       1.0000          1.0000            1.0000         9.09
IVF-RaBitQ-nl316-np15-rf0 (query)                      4_133.05       432.81     4_565.86       0.9360          1.0060            1.0038         9.64
IVF-RaBitQ-nl316-np17-rf0 (query)                      4_133.05       480.63     4_613.68       0.9360          1.0060            1.0038         9.64
IVF-RaBitQ-nl316-np25-rf0 (query)                      4_133.05       628.47     4_761.52       0.9360          1.0060            1.0038         9.64
IVF-RaBitQ-nl316-np15-rf10 (query)                     4_133.05       558.70     4_691.75       1.0000          1.0000            1.0000         9.64
IVF-RaBitQ-nl316-np15-rf20 (query)                     4_133.05       692.90     4_825.95       1.0000          1.0000            1.0000         9.64
IVF-RaBitQ-nl316-np17-rf10 (query)                     4_133.05       611.34     4_744.40       1.0000          1.0000            1.0000         9.64
IVF-RaBitQ-nl316-np17-rf20 (query)                     4_133.05       725.30     4_858.35       1.0000          1.0000            1.0000         9.64
IVF-RaBitQ-nl316-np25-rf10 (query)                     4_133.05       758.55     4_891.61       1.0000          1.0000            1.0000         9.64
IVF-RaBitQ-nl316-np25-rf20 (query)                     4_133.05       879.53     5_012.58       1.0000          1.0000            1.0000         9.64
IVF-RaBitQ-nl316 (self)                                4_133.05     2_827.44     6_960.49       1.0000          1.0000            1.0000         9.64
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
Exhaustive (query)                                        33.06       693.25       726.31       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.06     2_312.98     2_346.04       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              144.97       362.24       507.21       0.0971          1.7176            1.5958         7.12
ExhaustiveTQ-b2-rf5 (query)                              144.97       441.54       586.51       0.2336          1.2025            1.2204         7.12
ExhaustiveTQ-b2-rf10 (query)                             144.97       581.69       726.66       0.2853          1.1453            1.1620         7.12
ExhaustiveTQ-b2-rf20 (query)                             144.97       968.69     1_113.65       0.3809          1.0970            1.0941         7.12
ExhaustiveTQ-b2 (self)                                   144.97     3_196.63     3_341.60       0.3814          1.0980            1.0957         7.12
ExhaustiveTQ-b4-rf0 (query)                              225.30       584.12       809.42       0.1094          1.5328            1.4997        13.22
ExhaustiveTQ-b4-rf5 (query)                              225.30       662.55       887.85       0.2368          1.1884            1.2090        13.22
ExhaustiveTQ-b4-rf10 (query)                             225.30       804.17     1_029.47       0.2884          1.1372            1.1543        13.22
ExhaustiveTQ-b4-rf20 (query)                             225.30     1_205.85     1_431.15       0.3823          1.0940            1.0970        13.22
ExhaustiveTQ-b4 (self)                                   225.30     3_954.91     4_180.21       0.3841          1.0938            1.0948        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                          915.03       111.26     1_026.29       0.0971          1.7176            1.5958         7.80
IVF-TQ-b2-nl158-np12-rf0 (query)                         915.03       120.06     1_035.09       0.0971          1.7176            1.5958         7.80
IVF-TQ-b2-nl158-np17-rf0 (query)                         915.03       128.82     1_043.85       0.0971          1.7176            1.5958         7.80
IVF-TQ-b2-nl158-np7-rf10 (query)                         915.03       307.87     1_222.90       0.2853          1.1453            1.1620         7.80
IVF-TQ-b2-nl158-np7-rf20 (query)                         915.03       636.63     1_551.65       0.3809          1.0970            1.0941         7.80
IVF-TQ-b2-nl158-np12-rf10 (query)                        915.03       321.99     1_237.02       0.2853          1.1453            1.1620         7.80
IVF-TQ-b2-nl158-np12-rf20 (query)                        915.03       668.19     1_583.22       0.3809          1.0970            1.0941         7.80
IVF-TQ-b2-nl158-np17-rf10 (query)                        915.03       329.75     1_244.78       0.2853          1.1453            1.1620         7.80
IVF-TQ-b2-nl158-np17-rf20 (query)                        915.03       678.03     1_593.06       0.3809          1.0970            1.0941         7.80
IVF-TQ-b2-nl158 (self)                                   915.03     1_062.12     1_977.14       0.3815          1.0980            1.0957         7.80
IVF-TQ-b2-nl223-np11-rf0 (query)                         692.91       112.79       805.70       0.0971          1.7164            1.5942         7.93
IVF-TQ-b2-nl223-np14-rf0 (query)                         692.91       123.63       816.53       0.0971          1.7176            1.5958         7.93
IVF-TQ-b2-nl223-np21-rf0 (query)                         692.91       133.27       826.18       0.0971          1.7176            1.5958         7.93
IVF-TQ-b2-nl223-np11-rf10 (query)                        692.91       300.35       993.25       0.2855          1.1450            1.1618         7.93
IVF-TQ-b2-nl223-np11-rf20 (query)                        692.91       565.03     1_257.94       0.3813          1.0967            1.0934         7.93
IVF-TQ-b2-nl223-np14-rf10 (query)                        692.91       295.41       988.32       0.2853          1.1453            1.1620         7.93
IVF-TQ-b2-nl223-np14-rf20 (query)                        692.91       586.84     1_279.74       0.3809          1.0970            1.0941         7.93
IVF-TQ-b2-nl223-np21-rf10 (query)                        692.91       313.35     1_006.26       0.2853          1.1453            1.1620         7.93
IVF-TQ-b2-nl223-np21-rf20 (query)                        692.91       616.86     1_309.76       0.3809          1.0970            1.0941         7.93
IVF-TQ-b2-nl223 (self)                                   692.91     1_065.36     1_758.27       0.3815          1.0980            1.0957         7.93
IVF-TQ-b2-nl316-np15-rf0 (query)                         872.33       116.99       989.32       0.0973          1.6435            1.5781         8.10
IVF-TQ-b2-nl316-np17-rf0 (query)                         872.33       121.13       993.46       0.0972          1.7163            1.5957         8.10
IVF-TQ-b2-nl316-np25-rf0 (query)                         872.33       135.15     1_007.48       0.0971          1.7176            1.5958         8.10
IVF-TQ-b2-nl316-np15-rf10 (query)                        872.33       286.53     1_158.86       0.2858          1.1447            1.1615         8.10
IVF-TQ-b2-nl316-np15-rf20 (query)                        872.33       557.33     1_429.66       0.3817          1.0965            1.0931         8.10
IVF-TQ-b2-nl316-np17-rf10 (query)                        872.33       292.67     1_165.00       0.2853          1.1453            1.1620         8.10
IVF-TQ-b2-nl316-np17-rf20 (query)                        872.33       568.07     1_440.40       0.3809          1.0970            1.0941         8.10
IVF-TQ-b2-nl316-np25-rf10 (query)                        872.33       307.11     1_179.44       0.2853          1.1453            1.1620         8.10
IVF-TQ-b2-nl316-np25-rf20 (query)                        872.33       589.83     1_462.16       0.3809          1.0970            1.0941         8.10
IVF-TQ-b2-nl316 (self)                                   872.33     1_046.62     1_918.95       0.3815          1.0980            1.0957         8.10
IVF-TQ-b4-nl158-np7-rf0 (query)                          998.09       146.21     1_144.30       0.1094          1.5328            1.4997        14.05
IVF-TQ-b4-nl158-np12-rf0 (query)                         998.09       172.09     1_170.18       0.1094          1.5328            1.4996        14.05
IVF-TQ-b4-nl158-np17-rf0 (query)                         998.09       179.91     1_178.01       0.1094          1.5328            1.4996        14.05
IVF-TQ-b4-nl158-np7-rf10 (query)                         998.09       359.59     1_357.68       0.2884          1.1372            1.1543        14.05
IVF-TQ-b4-nl158-np7-rf20 (query)                         998.09       722.35     1_720.44       0.3823          1.0940            1.0970        14.05
IVF-TQ-b4-nl158-np12-rf10 (query)                        998.09       384.66     1_382.75       0.2884          1.1372            1.1543        14.05
IVF-TQ-b4-nl158-np12-rf20 (query)                        998.09       731.46     1_729.56       0.3823          1.0940            1.0970        14.05
IVF-TQ-b4-nl158-np17-rf10 (query)                        998.09       395.74     1_393.83       0.2884          1.1372            1.1543        14.05
IVF-TQ-b4-nl158-np17-rf20 (query)                        998.09       753.67     1_751.77       0.3823          1.0940            1.0970        14.05
IVF-TQ-b4-nl158 (self)                                   998.09     1_103.80     2_101.90       0.3841          1.0938            1.0948        14.05
IVF-TQ-b4-nl223-np11-rf0 (query)                         749.91       155.09       905.01       0.1094          1.5317            1.4988        14.25
IVF-TQ-b4-nl223-np14-rf0 (query)                         749.91       166.75       916.66       0.1094          1.5329            1.4997        14.25
IVF-TQ-b4-nl223-np21-rf0 (query)                         749.91       189.77       939.68       0.1094          1.5328            1.4996        14.25
IVF-TQ-b4-nl223-np11-rf10 (query)                        749.91       336.93     1_086.85       0.2886          1.1370            1.1541        14.25
IVF-TQ-b4-nl223-np11-rf20 (query)                        749.91       617.01     1_366.92       0.3826          1.0939            1.0966        14.25
IVF-TQ-b4-nl223-np14-rf10 (query)                        749.91       351.19     1_101.11       0.2884          1.1372            1.1543        14.25
IVF-TQ-b4-nl223-np14-rf20 (query)                        749.91       641.08     1_390.99       0.3823          1.0940            1.0970        14.25
IVF-TQ-b4-nl223-np21-rf10 (query)                        749.91       374.05     1_123.96       0.2884          1.1372            1.1543        14.25
IVF-TQ-b4-nl223-np21-rf20 (query)                        749.91       674.18     1_424.09       0.3823          1.0940            1.0970        14.25
IVF-TQ-b4-nl223 (self)                                   749.91     1_100.69     1_850.60       0.3841          1.0938            1.0948        14.25
IVF-TQ-b4-nl316-np15-rf0 (query)                         945.61       160.80     1_106.41       0.1094          1.5304            1.4978        14.49
IVF-TQ-b4-nl316-np17-rf0 (query)                         945.61       168.55     1_114.16       0.1094          1.5328            1.4996        14.49
IVF-TQ-b4-nl316-np25-rf0 (query)                         945.61       190.31     1_135.92       0.1094          1.5328            1.4997        14.49
IVF-TQ-b4-nl316-np15-rf10 (query)                        945.61       341.95     1_287.56       0.2886          1.1369            1.1542        14.49
IVF-TQ-b4-nl316-np15-rf20 (query)                        945.61       609.23     1_554.84       0.3828          1.0938            1.0967        14.49
IVF-TQ-b4-nl316-np17-rf10 (query)                        945.61       353.44     1_299.05       0.2884          1.1372            1.1543        14.49
IVF-TQ-b4-nl316-np17-rf20 (query)                        945.61       623.16     1_568.77       0.3823          1.0940            1.0970        14.49
IVF-TQ-b4-nl316-np25-rf10 (query)                        945.61       374.79     1_320.40       0.2884          1.1372            1.1543        14.49
IVF-TQ-b4-nl316-np25-rf20 (query)                        945.61       665.49     1_611.10       0.3823          1.0940            1.0970        14.49
IVF-TQ-b4-nl316 (self)                                   945.61     1_114.00     2_059.61       0.3841          1.0938            1.0948        14.49
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
Exhaustive (query)                                        68.35     1_290.26     1_358.61       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.35     4_339.27     4_407.62       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              342.22       629.73       971.95       0.1207          1.3711            1.3320        13.97
ExhaustiveTQ-b2-rf5 (query)                              342.22       721.86     1_064.08       0.2421          1.1334            1.1574        13.97
ExhaustiveTQ-b2-rf10 (query)                             342.22       872.68     1_214.90       0.2934          1.0981            1.1177        13.97
ExhaustiveTQ-b2-rf20 (query)                             342.22     1_286.30     1_628.51       0.3880          1.0664            1.0469        13.97
ExhaustiveTQ-b2 (self)                                   342.22     4_191.20     4_533.42       0.3879          1.0667            1.0471        13.97
ExhaustiveTQ-b4-rf0 (query)                              464.88     1_106.67     1_571.55       0.1315          1.3172            1.3127        26.18
ExhaustiveTQ-b4-rf5 (query)                              464.88     1_223.23     1_688.11       0.2471          1.1254            1.1483        26.18
ExhaustiveTQ-b4-rf10 (query)                             464.88     1_361.80     1_826.68       0.2970          1.0929            1.0980        26.18
ExhaustiveTQ-b4-rf20 (query)                             464.88     1_762.54     2_227.42       0.3883          1.0643            1.0492        26.18
ExhaustiveTQ-b4 (self)                                   464.88     5_844.61     6_309.49       0.3881          1.0646            1.0495        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_918.11       194.18     2_112.29       0.1207          1.3711            1.3320        14.95
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_918.11       209.49     2_127.60       0.1207          1.3711            1.3320        14.95
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_918.11       224.45     2_142.56       0.1207          1.3711            1.3320        14.95
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_918.11       413.68     2_331.79       0.2934          1.0981            1.1177        14.95
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_918.11       782.16     2_700.27       0.3880          1.0664            1.0469        14.95
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_918.11       442.90     2_361.01       0.2934          1.0981            1.1178        14.95
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_918.11       813.33     2_731.44       0.3880          1.0664            1.0469        14.95
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_918.11       457.16     2_375.27       0.2934          1.0981            1.1177        14.95
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_918.11       832.40     2_750.52       0.3880          1.0664            1.0469        14.95
IVF-TQ-b2-nl158 (self)                                 1_918.11     1_472.77     3_390.88       0.3879          1.0667            1.0471        14.95
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_187.00       206.34     1_393.34       0.1208          1.3699            1.3300        15.19
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_187.00       214.90     1_401.90       0.1207          1.3711            1.3320        15.19
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_187.00       235.11     1_422.11       0.1207          1.3711            1.3320        15.19
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_187.00       412.99     1_600.00       0.2937          1.0979            1.1176        15.19
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_187.00       712.91     1_899.91       0.3887          1.0662            1.0467        15.19
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_187.00       426.20     1_613.20       0.2934          1.0981            1.1177        15.19
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_187.00       729.53     1_916.54       0.3880          1.0664            1.0469        15.19
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_187.00       452.72     1_639.72       0.2934          1.0981            1.1178        15.19
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_187.00       781.09     1_968.09       0.3880          1.0664            1.0469        15.19
IVF-TQ-b2-nl223 (self)                                 1_187.00     1_457.10     2_644.10       0.3879          1.0667            1.0471        15.19
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_516.54       213.78     1_730.33       0.1208          1.3689            1.3287        15.56
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_516.54       220.25     1_736.79       0.1208          1.3707            1.3312        15.56
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_516.54       242.45     1_758.99       0.1207          1.3711            1.3320        15.56
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_516.54       404.78     1_921.32       0.2939          1.0977            1.1175        15.56
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_516.54       688.84     2_205.38       0.3892          1.0660            1.0465        15.56
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_516.54       414.02     1_930.57       0.2935          1.0980            1.1177        15.56
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_516.54       705.34     2_221.88       0.3882          1.0664            1.0469        15.56
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_516.54       438.55     1_955.09       0.2934          1.0981            1.1178        15.56
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_516.54       758.92     2_275.46       0.3880          1.0664            1.0469        15.56
IVF-TQ-b2-nl316 (self)                                 1_516.54     1_473.80     2_990.34       0.3879          1.0667            1.0471        15.56
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_986.81       267.58     2_254.39       0.1315          1.3172            1.3127        27.44
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_986.81       297.52     2_284.33       0.1315          1.3172            1.3127        27.44
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_986.81       321.71     2_308.51       0.1315          1.3172            1.3127        27.44
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_986.81       506.43     2_493.24       0.2970          1.0929            1.0979        27.44
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_986.81       859.78     2_846.58       0.3883          1.0643            1.0492        27.44
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_986.81       544.11     2_530.92       0.2970          1.0929            1.0979        27.44
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_986.81       905.65     2_892.46       0.3882          1.0643            1.0492        27.44
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_986.81       566.94     2_553.75       0.2970          1.0929            1.0979        27.44
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_986.81       940.29     2_927.09       0.3883          1.0643            1.0492        27.44
IVF-TQ-b4-nl158 (self)                                 1_986.81     1_612.13     3_598.94       0.3881          1.0646            1.0495        27.44
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_327.83       287.38     1_615.22       0.1315          1.3158            1.3116        27.79
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_327.83       311.81     1_639.65       0.1315          1.3172            1.3127        27.79
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_327.83       342.51     1_670.35       0.1315          1.3172            1.3127        27.79
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_327.83       521.63     1_849.47       0.2973          1.0926            1.0973        27.79
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_327.83       811.47     2_139.30       0.3889          1.0641            1.0489        27.79
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_327.83       521.35     1_849.19       0.2970          1.0929            1.0980        27.79
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_327.83       841.03     2_168.86       0.3883          1.0643            1.0492        27.79
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_327.83       560.08     1_887.91       0.2970          1.0929            1.0980        27.79
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_327.83       927.64     2_255.47       0.3883          1.0643            1.0492        27.79
IVF-TQ-b4-nl223 (self)                                 1_327.83     1_679.33     3_007.16       0.3881          1.0646            1.0495        27.79
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_682.31       299.40     1_981.71       0.1316          1.3151            1.3108        28.35
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_682.31       311.34     1_993.65       0.1315          1.3164            1.3121        28.35
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_682.31       347.29     2_029.59       0.1315          1.3172            1.3127        28.35
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_682.31       506.90     2_189.20       0.2976          1.0925            1.0966        28.35
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_682.31       801.74     2_484.04       0.3893          1.0639            1.0484        28.35
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_682.31       519.24     2_201.55       0.2971          1.0928            1.0977        28.35
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_682.31       821.33     2_503.64       0.3885          1.0642            1.0491        28.35
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_682.31       562.53     2_244.84       0.2970          1.0929            1.0980        28.35
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_682.31       893.45     2_575.75       0.3883          1.0643            1.0492        28.35
IVF-TQ-b4-nl316 (self)                                 1_682.31     1_712.15     3_394.46       0.3881          1.0646            1.0495        28.35
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
Exhaustive (query)                                       102.12     1_985.68     2_087.79       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        102.12     6_543.57     6_645.68       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              607.68       975.89     1_583.56       0.1292          1.2710            1.2627        21.33
ExhaustiveTQ-b2-rf5 (query)                              607.68     1_066.41     1_674.08       0.2468          1.1062            1.1332        21.33
ExhaustiveTQ-b2-rf10 (query)                             607.68     1_208.05     1_815.73       0.3000          1.0773            1.0631        21.33
ExhaustiveTQ-b2-rf20 (query)                             607.68     1_628.78     2_236.46       0.3957          1.0509            1.0334        21.33
ExhaustiveTQ-b2 (self)                                   607.68     5_340.36     5_948.04       0.3973          1.0507            1.0331        21.33
ExhaustiveTQ-b4-rf0 (query)                              758.38     1_787.07     2_545.44       0.1340          1.2532            1.2592        39.64
ExhaustiveTQ-b4-rf5 (query)                              758.38     1_869.89     2_628.27       0.2401          1.1136            1.1402        39.64
ExhaustiveTQ-b4-rf10 (query)                             758.38     2_028.83     2_787.20       0.2870          1.0888            1.1143        39.64
ExhaustiveTQ-b4-rf20 (query)                             758.38     2_425.61     3_183.99       0.3752          1.0657            1.0812        39.64
ExhaustiveTQ-b4 (self)                                   758.38     8_119.33     8_877.71       0.3767          1.0654            1.0638        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        2_817.38       296.99     3_114.38       0.1292          1.2710            1.2627        22.66
IVF-TQ-b2-nl158-np12-rf0 (query)                       2_817.38       319.64     3_137.03       0.1292          1.2710            1.2627        22.66
IVF-TQ-b2-nl158-np17-rf0 (query)                       2_817.38       343.30     3_160.69       0.1292          1.2710            1.2627        22.66
IVF-TQ-b2-nl158-np7-rf10 (query)                       2_817.38       539.01     3_356.39       0.3000          1.0773            1.0631        22.66
IVF-TQ-b2-nl158-np7-rf20 (query)                       2_817.38       900.65     3_718.04       0.3957          1.0509            1.0334        22.66
IVF-TQ-b2-nl158-np12-rf10 (query)                      2_817.38       564.80     3_382.18       0.3000          1.0773            1.0631        22.66
IVF-TQ-b2-nl158-np12-rf20 (query)                      2_817.38       947.13     3_764.51       0.3957          1.0509            1.0334        22.66
IVF-TQ-b2-nl158-np17-rf10 (query)                      2_817.38       584.80     3_402.18       0.3000          1.0774            1.0631        22.66
IVF-TQ-b2-nl158-np17-rf20 (query)                      2_817.38       973.60     3_790.98       0.3957          1.0509            1.0334        22.66
IVF-TQ-b2-nl158 (self)                                 2_817.38     1_833.27     4_650.65       0.3973          1.0507            1.0331        22.66
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_851.85       312.71     2_164.55       0.1292          1.2710            1.2627        23.04
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_851.85       323.14     2_174.98       0.1292          1.2710            1.2627        23.04
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_851.85       356.44     2_208.29       0.1292          1.2710            1.2627        23.04
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_851.85       568.91     2_420.75       0.3000          1.0774            1.0631        23.04
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_851.85       858.94     2_710.79       0.3957          1.0509            1.0334        23.04
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_851.85       577.74     2_429.59       0.3000          1.0774            1.0631        23.04
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_851.85       891.95     2_743.79       0.3957          1.0509            1.0334        23.04
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_851.85       604.12     2_455.96       0.3000          1.0774            1.0631        23.04
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_851.85       939.30     2_791.14       0.3957          1.0509            1.0334        23.04
IVF-TQ-b2-nl223 (self)                                 1_851.85     1_862.77     3_714.62       0.3973          1.0507            1.0331        23.04
IVF-TQ-b2-nl316-np15-rf0 (query)                       2_209.21       326.29     2_535.50       0.1292          1.2709            1.2627        23.57
IVF-TQ-b2-nl316-np17-rf0 (query)                       2_209.21       331.91     2_541.12       0.1292          1.2710            1.2628        23.57
IVF-TQ-b2-nl316-np25-rf0 (query)                       2_209.21       360.59     2_569.80       0.1292          1.2710            1.2628        23.57
IVF-TQ-b2-nl316-np15-rf10 (query)                      2_209.21       549.26     2_758.47       0.3000          1.0773            1.0631        23.57
IVF-TQ-b2-nl316-np15-rf20 (query)                      2_209.21       880.45     3_089.66       0.3957          1.0509            1.0334        23.57
IVF-TQ-b2-nl316-np17-rf10 (query)                      2_209.21       556.76     2_765.97       0.3000          1.0774            1.0632        23.57
IVF-TQ-b2-nl316-np17-rf20 (query)                      2_209.21       876.15     3_085.36       0.3957          1.0509            1.0334        23.57
IVF-TQ-b2-nl316-np25-rf10 (query)                      2_209.21       599.01     2_808.22       0.3000          1.0774            1.0631        23.57
IVF-TQ-b2-nl316-np25-rf20 (query)                      2_209.21       923.36     3_132.57       0.3957          1.0509            1.0334        23.57
IVF-TQ-b2-nl316 (self)                                 2_209.21     1_917.56     4_126.77       0.3973          1.0507            1.0331        23.57
IVF-TQ-b4-nl158-np7-rf0 (query)                        2_811.34       414.73     3_226.07       0.1340          1.2532            1.2592        41.46
IVF-TQ-b4-nl158-np12-rf0 (query)                       2_811.34       461.47     3_272.81       0.1340          1.2532            1.2592        41.46
IVF-TQ-b4-nl158-np17-rf0 (query)                       2_811.34       500.57     3_311.91       0.1340          1.2532            1.2592        41.46
IVF-TQ-b4-nl158-np7-rf10 (query)                       2_811.34       678.25     3_489.59       0.2870          1.0888            1.1143        41.46
IVF-TQ-b4-nl158-np7-rf20 (query)                       2_811.34     1_044.71     3_856.04       0.3752          1.0657            1.0812        41.46
IVF-TQ-b4-nl158-np12-rf10 (query)                      2_811.34       722.08     3_533.41       0.2870          1.0888            1.1143        41.46
IVF-TQ-b4-nl158-np12-rf20 (query)                      2_811.34     1_113.88     3_925.22       0.3752          1.0657            1.0812        41.46
IVF-TQ-b4-nl158-np17-rf10 (query)                      2_811.34       762.87     3_574.20       0.2870          1.0888            1.1143        41.46
IVF-TQ-b4-nl158-np17-rf20 (query)                      2_811.34     1_159.38     3_970.72       0.3752          1.0657            1.0812        41.46
IVF-TQ-b4-nl158 (self)                                 2_811.34     2_161.71     4_973.04       0.3767          1.0654            1.0637        41.46
IVF-TQ-b4-nl223-np11-rf0 (query)                       2_034.86       453.58     2_488.44       0.1340          1.2532            1.2592        42.04
IVF-TQ-b4-nl223-np14-rf0 (query)                       2_034.86       478.44     2_513.30       0.1340          1.2532            1.2592        42.04
IVF-TQ-b4-nl223-np21-rf0 (query)                       2_034.86       532.89     2_567.75       0.1340          1.2532            1.2592        42.04
IVF-TQ-b4-nl223-np11-rf10 (query)                      2_034.86       686.09     2_720.95       0.2870          1.0888            1.1143        42.04
IVF-TQ-b4-nl223-np11-rf20 (query)                      2_034.86     1_008.03     3_042.89       0.3752          1.0657            1.0812        42.04
IVF-TQ-b4-nl223-np14-rf10 (query)                      2_034.86       716.09     2_750.95       0.2870          1.0888            1.1143        42.04
IVF-TQ-b4-nl223-np14-rf20 (query)                      2_034.86     1_073.27     3_108.13       0.3752          1.0657            1.0812        42.04
IVF-TQ-b4-nl223-np21-rf10 (query)                      2_034.86       779.85     2_814.71       0.2870          1.0888            1.1143        42.04
IVF-TQ-b4-nl223-np21-rf20 (query)                      2_034.86     1_125.39     3_160.25       0.3752          1.0657            1.0812        42.04
IVF-TQ-b4-nl223 (self)                                 2_034.86     2_212.82     4_247.68       0.3766          1.0654            1.0638        42.04
IVF-TQ-b4-nl316-np15-rf0 (query)                       2_380.49       469.19     2_849.68       0.1340          1.2531            1.2592        42.81
IVF-TQ-b4-nl316-np17-rf0 (query)                       2_380.49       487.02     2_867.51       0.1340          1.2531            1.2592        42.81
IVF-TQ-b4-nl316-np25-rf0 (query)                       2_380.49       556.06     2_936.54       0.1340          1.2532            1.2592        42.81
IVF-TQ-b4-nl316-np15-rf10 (query)                      2_380.49       698.88     3_079.37       0.2870          1.0888            1.1143        42.81
IVF-TQ-b4-nl316-np15-rf20 (query)                      2_380.49     1_015.07     3_395.56       0.3753          1.0657            1.0812        42.81
IVF-TQ-b4-nl316-np17-rf10 (query)                      2_380.49       713.71     3_094.20       0.2870          1.0888            1.1143        42.81
IVF-TQ-b4-nl316-np17-rf20 (query)                      2_380.49     1_041.97     3_422.46       0.3752          1.0657            1.0812        42.81
IVF-TQ-b4-nl316-np25-rf10 (query)                      2_380.49       775.98     3_156.47       0.2870          1.0888            1.1143        42.81
IVF-TQ-b4-nl316-np25-rf20 (query)                      2_380.49     1_111.20     3_491.69       0.3752          1.0657            1.0812        42.81
IVF-TQ-b4-nl316 (self)                                 2_380.49     2_269.47     4_649.96       0.3767          1.0654            1.0637        42.81
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
Exhaustive (query)                                        33.49       767.79       801.28       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.49     2_497.81     2_531.30       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              143.89       372.88       516.78       0.0756          2.3283            1.9295         7.12
ExhaustiveTQ-b2-rf5 (query)                              143.89       445.74       589.64       0.2072          1.3307            1.3578         7.12
ExhaustiveTQ-b2-rf10 (query)                             143.89       578.78       722.68       0.2886          1.2206            1.2322         7.12
ExhaustiveTQ-b2-rf20 (query)                             143.89       960.23     1_104.13       0.4151          1.1328            1.1147         7.12
ExhaustiveTQ-b2 (self)                                   143.89     3_320.86     3_464.76       0.4136          1.1619            1.1367         7.12
ExhaustiveTQ-b4-rf0 (query)                              230.90       593.84       824.74       0.1023          1.7129            1.7532        13.22
ExhaustiveTQ-b4-rf5 (query)                              230.90       680.72       911.62       0.2385          1.2770            1.3000        13.22
ExhaustiveTQ-b4-rf10 (query)                             230.90       840.23     1_071.13       0.3202          1.1874            1.1953        13.22
ExhaustiveTQ-b4-rf20 (query)                             230.90     1_187.92     1_418.82       0.4481          1.1142            1.1029        13.22
ExhaustiveTQ-b4 (self)                                   230.90     3_933.42     4_164.32       0.4463          1.1397            1.1286        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_029.32       104.16     1_133.48       0.0756          2.3282            1.9295         7.81
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_029.32       115.72     1_145.04       0.0756          2.3283            1.9295         7.81
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_029.32       136.21     1_165.53       0.0756          2.3283            1.9295         7.81
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_029.32       303.03     1_332.35       0.2886          1.2206            1.2322         7.81
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_029.32       631.88     1_661.20       0.4151          1.1328            1.1147         7.81
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_029.32       330.16     1_359.47       0.2886          1.2206            1.2322         7.81
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_029.32       658.96     1_688.28       0.4151          1.1328            1.1147         7.81
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_029.32       353.05     1_382.37       0.2886          1.2206            1.2322         7.81
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_029.32       743.26     1_772.58       0.4151          1.1328            1.1147         7.81
IVF-TQ-b2-nl158 (self)                                 1_029.32     1_091.07     2_120.39       0.4136          1.1619            1.1367         7.81
IVF-TQ-b2-nl223-np11-rf0 (query)                         714.17       110.46       824.63       0.0756          2.3256            1.9254         7.94
IVF-TQ-b2-nl223-np14-rf0 (query)                         714.17       115.50       829.68       0.0756          2.3281            1.9295         7.94
IVF-TQ-b2-nl223-np21-rf0 (query)                         714.17       142.06       856.24       0.0756          2.3282            1.9295         7.94
IVF-TQ-b2-nl223-np11-rf10 (query)                        714.17       283.80       997.97       0.2890          1.2203            1.2319         7.94
IVF-TQ-b2-nl223-np11-rf20 (query)                        714.17       566.19     1_280.36       0.4156          1.1326            1.1144         7.94
IVF-TQ-b2-nl223-np14-rf10 (query)                        714.17       296.61     1_010.78       0.2886          1.2206            1.2322         7.94
IVF-TQ-b2-nl223-np14-rf20 (query)                        714.17       587.06     1_301.23       0.4151          1.1328            1.1147         7.94
IVF-TQ-b2-nl223-np21-rf10 (query)                        714.17       335.68     1_049.85       0.2886          1.2206            1.2322         7.94
IVF-TQ-b2-nl223-np21-rf20 (query)                        714.17       651.77     1_365.94       0.4151          1.1328            1.1147         7.94
IVF-TQ-b2-nl223 (self)                                   714.17     1_083.67     1_797.84       0.4136          1.1619            1.1367         7.94
IVF-TQ-b2-nl316-np15-rf0 (query)                         934.71       117.98     1_052.68       0.0757          2.3274            1.9289         8.11
IVF-TQ-b2-nl316-np17-rf0 (query)                         934.71       119.68     1_054.38       0.0756          2.3282            1.9294         8.11
IVF-TQ-b2-nl316-np25-rf0 (query)                         934.71       140.15     1_074.85       0.0756          2.3282            1.9295         8.11
IVF-TQ-b2-nl316-np15-rf10 (query)                        934.71       285.16     1_219.87       0.2892          1.2202            1.2317         8.11
IVF-TQ-b2-nl316-np15-rf20 (query)                        934.71       543.32     1_478.03       0.4159          1.1325            1.1142         8.11
IVF-TQ-b2-nl316-np17-rf10 (query)                        934.71       292.35     1_227.06       0.2887          1.2206            1.2322         8.11
IVF-TQ-b2-nl316-np17-rf20 (query)                        934.71       552.29     1_487.00       0.4151          1.1328            1.1147         8.11
IVF-TQ-b2-nl316-np25-rf10 (query)                        934.71       327.70     1_262.41       0.2886          1.2206            1.2322         8.11
IVF-TQ-b2-nl316-np25-rf20 (query)                        934.71       588.17     1_522.88       0.4151          1.1328            1.1147         8.11
IVF-TQ-b2-nl316 (self)                                   934.71     1_080.18     2_014.88       0.4136          1.1619            1.1367         8.11
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_096.39       140.53     1_236.92       0.1023          1.7129            1.7532        14.06
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_096.39       165.43     1_261.82       0.1023          1.7129            1.7532        14.06
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_096.39       196.08     1_292.47       0.1023          1.7129            1.7532        14.06
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_096.39       354.33     1_450.73       0.3202          1.1874            1.1953        14.06
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_096.39       687.16     1_783.56       0.4481          1.1142            1.1029        14.06
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_096.39       371.91     1_468.30       0.3202          1.1874            1.1953        14.06
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_096.39       722.73     1_819.12       0.4481          1.1142            1.1029        14.06
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_096.39       423.30     1_519.69       0.3202          1.1873            1.1953        14.06
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_096.39       816.61     1_913.00       0.4481          1.1142            1.1029        14.06
IVF-TQ-b4-nl158 (self)                                 1_096.39     1_154.59     2_250.99       0.4463          1.1397            1.1286        14.06
IVF-TQ-b4-nl223-np11-rf0 (query)                         801.89       150.89       952.77       0.1023          1.7109            1.7520        14.27
IVF-TQ-b4-nl223-np14-rf0 (query)                         801.89       160.96       962.85       0.1023          1.7129            1.7532        14.27
IVF-TQ-b4-nl223-np21-rf0 (query)                         801.89       203.65     1_005.54       0.1023          1.7129            1.7532        14.27
IVF-TQ-b4-nl223-np11-rf10 (query)                        801.89       335.35     1_137.23       0.3205          1.1871            1.1949        14.27
IVF-TQ-b4-nl223-np11-rf20 (query)                        801.89       621.34     1_423.22       0.4486          1.1140            1.1028        14.27
IVF-TQ-b4-nl223-np14-rf10 (query)                        801.89       348.46     1_150.35       0.3202          1.1873            1.1953        14.27
IVF-TQ-b4-nl223-np14-rf20 (query)                        801.89       649.14     1_451.03       0.4481          1.1142            1.1029        14.27
IVF-TQ-b4-nl223-np21-rf10 (query)                        801.89       406.43     1_208.31       0.3201          1.1874            1.1953        14.27
IVF-TQ-b4-nl223-np21-rf20 (query)                        801.89       732.93     1_534.81       0.4481          1.1142            1.1029        14.27
IVF-TQ-b4-nl223 (self)                                   801.89     1_118.04     1_919.92       0.4463          1.1397            1.1286        14.27
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_016.19       156.58     1_172.77       0.1023          1.7112            1.7520        14.52
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_016.19       162.18     1_178.37       0.1023          1.7121            1.7528        14.52
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_016.19       196.83     1_213.02       0.1023          1.7129            1.7532        14.52
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_016.19       335.09     1_351.28       0.3207          1.1869            1.1949        14.52
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_016.19       590.88     1_607.08       0.4491          1.1138            1.1025        14.52
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_016.19       342.49     1_358.68       0.3203          1.1873            1.1953        14.52
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_016.19       607.85     1_624.04       0.4482          1.1142            1.1029        14.52
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_016.19       387.40     1_403.59       0.3202          1.1873            1.1953        14.52
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_016.19       661.98     1_678.17       0.4481          1.1142            1.1029        14.52
IVF-TQ-b4-nl316 (self)                                 1_016.19     1_122.96     2_139.15       0.4463          1.1397            1.1286        14.52
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
Exhaustive (query)                                        67.99     1_284.59     1_352.59       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         67.99     4_307.39     4_375.38       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              336.22       626.20       962.42       0.0844          1.6539            1.5906        13.97
ExhaustiveTQ-b2-rf5 (query)                              336.22       716.16     1_052.37       0.2173          1.2230            1.2549        13.97
ExhaustiveTQ-b2-rf10 (query)                             336.22       865.14     1_201.36       0.2887          1.1550            1.1707        13.97
ExhaustiveTQ-b2-rf20 (query)                             336.22     1_292.71     1_628.93       0.4020          1.0974            1.0847        13.97
ExhaustiveTQ-b2 (self)                                   336.22     4_206.69     4_542.91       0.4025          1.1135            1.0971        13.97
ExhaustiveTQ-b4-rf0 (query)                              462.66     1_112.74     1_575.41       0.1044          1.5026            1.5346        26.18
ExhaustiveTQ-b4-rf5 (query)                              462.66     1_211.62     1_674.28       0.2294          1.2110            1.2410        26.18
ExhaustiveTQ-b4-rf10 (query)                             462.66     1_357.01     1_819.67       0.2943          1.1499            1.1675        26.18
ExhaustiveTQ-b4-rf20 (query)                             462.66     1_772.89     2_235.55       0.4029          1.0975            1.0929        26.18
ExhaustiveTQ-b4 (self)                                   462.66     5_825.90     6_288.57       0.4038          1.1130            1.1087        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_856.20       187.30     2_043.49       0.0844          1.6539            1.5906        14.95
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_856.20       206.07     2_062.26       0.0844          1.6539            1.5906        14.95
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_856.20       227.53     2_083.73       0.0844          1.6539            1.5906        14.95
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_856.20       413.27     2_269.47       0.2887          1.1550            1.1707        14.95
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_856.20       758.34     2_614.54       0.4020          1.0974            1.0847        14.95
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_856.20       432.38     2_288.58       0.2887          1.1550            1.1707        14.95
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_856.20       787.75     2_643.95       0.4020          1.0974            1.0847        14.95
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_856.20       450.28     2_306.48       0.2887          1.1550            1.1707        14.95
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_856.20       819.20     2_675.39       0.4020          1.0974            1.0847        14.95
IVF-TQ-b2-nl158 (self)                                 1_856.20     1_547.31     3_403.50       0.4025          1.1135            1.0971        14.95
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_284.98       202.54     1_487.52       0.0845          1.6537            1.5905        15.23
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_284.98       214.40     1_499.38       0.0844          1.6539            1.5906        15.23
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_284.98       239.54     1_524.52       0.0844          1.6539            1.5906        15.23
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_284.98       429.70     1_714.67       0.2887          1.1550            1.1707        15.23
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_284.98       717.76     2_002.73       0.4020          1.0974            1.0847        15.23
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_284.98       424.42     1_709.39       0.2887          1.1550            1.1707        15.23
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_284.98       742.00     2_026.98       0.4020          1.0974            1.0847        15.23
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_284.98       457.05     1_742.03       0.2887          1.1550            1.1707        15.23
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_284.98       798.85     2_083.83       0.4020          1.0974            1.0847        15.23
IVF-TQ-b2-nl223 (self)                                 1_284.98     1_460.60     2_745.57       0.4025          1.1135            1.0971        15.23
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_700.33       210.56     1_910.90       0.0844          1.6539            1.5906        15.57
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_700.33       217.30     1_917.63       0.0844          1.6539            1.5906        15.57
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_700.33       240.17     1_940.50       0.0844          1.6539            1.5906        15.57
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_700.33       403.46     2_103.79       0.2887          1.1550            1.1707        15.57
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_700.33       714.77     2_415.10       0.4020          1.0974            1.0847        15.57
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_700.33       414.74     2_115.07       0.2887          1.1550            1.1707        15.57
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_700.33       711.49     2_411.82       0.4020          1.0974            1.0847        15.57
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_700.33       442.81     2_143.14       0.2887          1.1550            1.1707        15.57
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_700.33       762.65     2_462.98       0.4020          1.0974            1.0847        15.57
IVF-TQ-b2-nl316 (self)                                 1_700.33     1_483.91     3_184.24       0.4025          1.1135            1.0971        15.57
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_921.78       259.39     2_181.16       0.1044          1.5026            1.5346        27.44
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_921.78       292.05     2_213.83       0.1044          1.5026            1.5346        27.44
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_921.78       317.37     2_239.15       0.1044          1.5026            1.5346        27.44
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_921.78       501.14     2_422.92       0.2943          1.1499            1.1675        27.44
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_921.78       849.05     2_770.82       0.4029          1.0975            1.0929        27.44
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_921.78       532.49     2_454.27       0.2943          1.1499            1.1675        27.44
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_921.78       895.74     2_817.51       0.4029          1.0975            1.0929        27.44
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_921.78       560.39     2_482.16       0.2943          1.1499            1.1675        27.44
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_921.78       958.47     2_880.25       0.4029          1.0975            1.0929        27.44
IVF-TQ-b4-nl158 (self)                                 1_921.78     1_651.19     3_572.97       0.4038          1.1130            1.1088        27.44
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_405.96       283.07     1_689.03       0.1044          1.5026            1.5346        27.87
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_405.96       301.71     1_707.67       0.1044          1.5026            1.5346        27.87
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_405.96       360.08     1_766.04       0.1044          1.5026            1.5346        27.87
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_405.96       505.54     1_911.50       0.2943          1.1499            1.1676        27.87
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_405.96       815.34     2_221.30       0.4029          1.0975            1.0929        27.87
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_405.96       531.04     1_937.00       0.2943          1.1499            1.1676        27.87
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_405.96       855.14     2_261.10       0.4029          1.0975            1.0929        27.87
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_405.96       583.80     1_989.76       0.2943          1.1499            1.1676        27.87
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_405.96       932.81     2_338.77       0.4029          1.0975            1.0929        27.87
IVF-TQ-b4-nl223 (self)                                 1_405.96     1_665.71     3_071.67       0.4038          1.1130            1.1088        27.87
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_824.06       292.02     2_116.08       0.1045          1.5026            1.5346        28.38
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_824.06       305.20     2_129.26       0.1044          1.5026            1.5346        28.38
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_824.06       347.70     2_171.76       0.1044          1.5026            1.5346        28.38
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_824.06       507.08     2_331.14       0.2943          1.1499            1.1675        28.38
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_824.06       809.88     2_633.94       0.4029          1.0975            1.0929        28.38
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_824.06       518.27     2_342.33       0.2943          1.1499            1.1675        28.38
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_824.06       812.14     2_636.20       0.4029          1.0975            1.0929        28.38
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_824.06       564.34     2_388.39       0.2943          1.1499            1.1675        28.38
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_824.06       880.49     2_704.55       0.4029          1.0975            1.0929        28.38
IVF-TQ-b4-nl316 (self)                                 1_824.06     1_676.06     3_500.11       0.4038          1.1130            1.1087        28.38
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
Exhaustive (query)                                       100.23     1_936.61     2_036.84       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.23     6_544.19     6_644.42       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              611.83       966.25     1_578.08       0.0841          1.5107            1.4226        21.33
ExhaustiveTQ-b2-rf5 (query)                              611.83     1_057.80     1_669.62       0.2144          1.1739            1.2056        21.33
ExhaustiveTQ-b2-rf10 (query)                             611.83     1_207.21     1_819.04       0.2770          1.1267            1.1512        21.33
ExhaustiveTQ-b2-rf20 (query)                             611.83     1_656.12     2_267.95       0.3770          1.0843            1.0724        21.33
ExhaustiveTQ-b2 (self)                                   611.83     5_385.10     5_996.93       0.3767          1.0935            1.0803        21.33
ExhaustiveTQ-b4-rf0 (query)                              760.35     1_800.69     2_561.04       0.0986          1.4231            1.4109        39.64
ExhaustiveTQ-b4-rf5 (query)                              760.35     1_893.85     2_654.20       0.2167          1.1746            1.2047        39.64
ExhaustiveTQ-b4-rf10 (query)                             760.35     2_033.70     2_794.05       0.2692          1.1311            1.1557        39.64
ExhaustiveTQ-b4-rf20 (query)                             760.35     2_445.05     3_205.41       0.3605          1.0923            1.1071        39.64
ExhaustiveTQ-b4 (self)                                   760.35     8_027.34     8_787.69       0.3609          1.1024            1.1182        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        2_785.64       287.07     3_072.71       0.0841          1.5107            1.4226        22.62
IVF-TQ-b2-nl158-np12-rf0 (query)                       2_785.64       311.14     3_096.77       0.0841          1.5107            1.4226        22.62
IVF-TQ-b2-nl158-np17-rf0 (query)                       2_785.64       337.01     3_122.64       0.0841          1.5107            1.4226        22.62
IVF-TQ-b2-nl158-np7-rf10 (query)                       2_785.64       524.55     3_310.18       0.2770          1.1267            1.1512        22.62
IVF-TQ-b2-nl158-np7-rf20 (query)                       2_785.64       919.42     3_705.05       0.3770          1.0843            1.0724        22.62
IVF-TQ-b2-nl158-np12-rf10 (query)                      2_785.64       542.75     3_328.39       0.2770          1.1267            1.1512        22.62
IVF-TQ-b2-nl158-np12-rf20 (query)                      2_785.64       931.57     3_717.20       0.3770          1.0843            1.0724        22.62
IVF-TQ-b2-nl158-np17-rf10 (query)                      2_785.64       572.02     3_357.66       0.2771          1.1267            1.1512        22.62
IVF-TQ-b2-nl158-np17-rf20 (query)                      2_785.64       952.89     3_738.53       0.3770          1.0843            1.0724        22.62
IVF-TQ-b2-nl158 (self)                                 2_785.64     1_870.11     4_655.75       0.3767          1.0935            1.0803        22.62
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_872.38       309.33     2_181.72       0.0841          1.5107            1.4226        22.97
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_872.38       322.81     2_195.19       0.0841          1.5107            1.4226        22.97
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_872.38       353.69     2_226.08       0.0841          1.5107            1.4226        22.97
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_872.38       546.97     2_419.35       0.2771          1.1267            1.1512        22.97
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_872.38       914.43     2_786.81       0.3770          1.0843            1.0724        22.97
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_872.38       557.80     2_430.19       0.2771          1.1267            1.1512        22.97
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_872.38       909.10     2_781.48       0.3770          1.0843            1.0724        22.97
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_872.38       595.52     2_467.90       0.2771          1.1267            1.1512        22.97
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_872.38       965.66     2_838.05       0.3770          1.0843            1.0724        22.97
IVF-TQ-b2-nl223 (self)                                 1_872.38     1_890.96     3_763.34       0.3767          1.0935            1.0803        22.97
IVF-TQ-b2-nl316-np15-rf0 (query)                       2_498.00       317.20     2_815.20       0.0841          1.5107            1.4226        23.53
IVF-TQ-b2-nl316-np17-rf0 (query)                       2_498.00       326.46     2_824.46       0.0841          1.5107            1.4226        23.53
IVF-TQ-b2-nl316-np25-rf0 (query)                       2_498.00       355.90     2_853.90       0.0841          1.5107            1.4226        23.53
IVF-TQ-b2-nl316-np15-rf10 (query)                      2_498.00       548.52     3_046.52       0.2771          1.1267            1.1512        23.53
IVF-TQ-b2-nl316-np15-rf20 (query)                      2_498.00       865.51     3_363.50       0.3770          1.0843            1.0724        23.53
IVF-TQ-b2-nl316-np17-rf10 (query)                      2_498.00       559.30     3_057.30       0.2771          1.1267            1.1512        23.53
IVF-TQ-b2-nl316-np17-rf20 (query)                      2_498.00       883.09     3_381.09       0.3770          1.0843            1.0724        23.53
IVF-TQ-b2-nl316-np25-rf10 (query)                      2_498.00       605.11     3_103.11       0.2771          1.1267            1.1512        23.53
IVF-TQ-b2-nl316-np25-rf20 (query)                      2_498.00       939.14     3_437.14       0.3770          1.0843            1.0724        23.53
IVF-TQ-b2-nl316 (self)                                 2_498.00     1_967.06     4_465.06       0.3767          1.0935            1.0803        23.53
IVF-TQ-b4-nl158-np7-rf0 (query)                        2_857.40       408.79     3_266.19       0.0986          1.4231            1.4109        41.39
IVF-TQ-b4-nl158-np12-rf0 (query)                       2_857.40       447.76     3_305.16       0.0986          1.4231            1.4109        41.39
IVF-TQ-b4-nl158-np17-rf0 (query)                       2_857.40       504.33     3_361.73       0.0986          1.4231            1.4109        41.39
IVF-TQ-b4-nl158-np7-rf10 (query)                       2_857.40       657.39     3_514.79       0.2692          1.1311            1.1557        41.39
IVF-TQ-b4-nl158-np7-rf20 (query)                       2_857.40     1_045.99     3_903.39       0.3605          1.0923            1.1071        41.39
IVF-TQ-b4-nl158-np12-rf10 (query)                      2_857.40       697.66     3_555.06       0.2692          1.1311            1.1557        41.39
IVF-TQ-b4-nl158-np12-rf20 (query)                      2_857.40     1_089.73     3_947.12       0.3605          1.0923            1.1071        41.39
IVF-TQ-b4-nl158-np17-rf10 (query)                      2_857.40       750.99     3_608.39       0.2692          1.1311            1.1557        41.39
IVF-TQ-b4-nl158-np17-rf20 (query)                      2_857.40     1_153.30     4_010.69       0.3605          1.0923            1.1071        41.39
IVF-TQ-b4-nl158 (self)                                 2_857.40     2_207.40     5_064.80       0.3609          1.1024            1.1182        41.39
IVF-TQ-b4-nl223-np11-rf0 (query)                       2_035.52       442.18     2_477.69       0.0986          1.4231            1.4109        41.89
IVF-TQ-b4-nl223-np14-rf0 (query)                       2_035.52       468.06     2_503.57       0.0986          1.4231            1.4109        41.89
IVF-TQ-b4-nl223-np21-rf0 (query)                       2_035.52       525.63     2_561.15       0.0986          1.4231            1.4109        41.89
IVF-TQ-b4-nl223-np11-rf10 (query)                      2_035.52       683.19     2_718.71       0.2692          1.1311            1.1557        41.89
IVF-TQ-b4-nl223-np11-rf20 (query)                      2_035.52     1_031.83     3_067.34       0.3605          1.0923            1.1071        41.89
IVF-TQ-b4-nl223-np14-rf10 (query)                      2_035.52       713.51     2_749.03       0.2692          1.1311            1.1557        41.89
IVF-TQ-b4-nl223-np14-rf20 (query)                      2_035.52     1_061.56     3_097.07       0.3605          1.0923            1.1071        41.89
IVF-TQ-b4-nl223-np21-rf10 (query)                      2_035.52       777.48     2_813.00       0.2692          1.1311            1.1557        41.89
IVF-TQ-b4-nl223-np21-rf20 (query)                      2_035.52     1_162.44     3_197.96       0.3605          1.0923            1.1071        41.89
IVF-TQ-b4-nl223 (self)                                 2_035.52     2_284.21     4_319.73       0.3609          1.1024            1.1182        41.89
IVF-TQ-b4-nl316-np15-rf0 (query)                       2_662.47       479.63     3_142.10       0.0986          1.4231            1.4109        42.73
IVF-TQ-b4-nl316-np17-rf0 (query)                       2_662.47       481.07     3_143.54       0.0986          1.4231            1.4109        42.73
IVF-TQ-b4-nl316-np25-rf0 (query)                       2_662.47       541.67     3_204.14       0.0986          1.4231            1.4109        42.73
IVF-TQ-b4-nl316-np15-rf10 (query)                      2_662.47       696.92     3_359.39       0.2692          1.1311            1.1557        42.73
IVF-TQ-b4-nl316-np15-rf20 (query)                      2_662.47     1_023.37     3_685.83       0.3605          1.0923            1.1071        42.73
IVF-TQ-b4-nl316-np17-rf10 (query)                      2_662.47       720.31     3_382.78       0.2692          1.1311            1.1557        42.73
IVF-TQ-b4-nl316-np17-rf20 (query)                      2_662.47     1_068.70     3_731.17       0.3605          1.0923            1.1071        42.73
IVF-TQ-b4-nl316-np25-rf10 (query)                      2_662.47       785.47     3_447.94       0.2692          1.1311            1.1557        42.73
IVF-TQ-b4-nl316-np25-rf20 (query)                      2_662.47     1_119.75     3_782.22       0.3605          1.0923            1.1071        42.73
IVF-TQ-b4-nl316 (self)                                 2_662.47     2_322.14     4_984.61       0.3609          1.1024            1.1182        42.73
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
Exhaustive (query)                                        32.74       745.60       778.34       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.74     2_523.73     2_556.47       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              145.19       384.53       529.72       0.7918          1.0898            1.0632         7.12
ExhaustiveTQ-b2-rf5 (query)                              145.19       454.63       599.82       0.9995          1.0000            1.0000         7.12
ExhaustiveTQ-b2-rf10 (query)                             145.19       589.32       734.51       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b2-rf20 (query)                             145.19       984.42     1_129.61       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b2 (self)                                   145.19     3_252.24     3_397.43       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b4-rf0 (query)                              229.59       600.13       829.71       0.8728          1.0322            1.0183        13.22
ExhaustiveTQ-b4-rf5 (query)                              229.59       684.19       913.77       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4-rf10 (query)                             229.59       818.21     1_047.79       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4-rf20 (query)                             229.59     1_217.26     1_446.85       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4 (self)                                   229.59     3_980.42     4_210.01       1.0000          1.0000            1.0000        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_076.98       131.96     1_208.94       0.7916          1.0897            1.0635         7.78
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_076.98       173.51     1_250.50       0.7918          1.0898            1.0632         7.78
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_076.98       211.09     1_288.07       0.7918          1.0898            1.0632         7.78
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_076.98       335.09     1_412.07       0.9982          1.0004            1.0000         7.78
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_076.98       628.04     1_705.02       0.9982          1.0004            1.0000         7.78
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_076.98       399.35     1_476.33       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_076.98       736.36     1_813.34       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_076.98       450.34     1_527.32       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_076.98       803.22     1_880.20       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158 (self)                                 1_076.98     1_203.42     2_280.41       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl223-np11-rf0 (query)                         629.00       127.99       756.99       0.7919          1.0897            1.0632         7.93
IVF-TQ-b2-nl223-np14-rf0 (query)                         629.00       145.17       774.17       0.7918          1.0897            1.0632         7.93
IVF-TQ-b2-nl223-np21-rf0 (query)                         629.00       184.61       813.61       0.7918          1.0898            1.0632         7.93
IVF-TQ-b2-nl223-np11-rf10 (query)                        629.00       320.33       949.33       0.9995          1.0001            1.0000         7.93
IVF-TQ-b2-nl223-np11-rf20 (query)                        629.00       596.19     1_225.18       0.9995          1.0001            1.0000         7.93
IVF-TQ-b2-nl223-np14-rf10 (query)                        629.00       350.37       979.37       0.9999          1.0000            1.0000         7.93
IVF-TQ-b2-nl223-np14-rf20 (query)                        629.00       643.87     1_272.87       0.9999          1.0000            1.0000         7.93
IVF-TQ-b2-nl223-np21-rf10 (query)                        629.00       404.12     1_033.12       1.0000          1.0000            1.0000         7.93
IVF-TQ-b2-nl223-np21-rf20 (query)                        629.00       731.46     1_360.46       1.0000          1.0000            1.0000         7.93
IVF-TQ-b2-nl223 (self)                                   629.00     1_071.49     1_700.49       1.0000          1.0000            1.0000         7.93
IVF-TQ-b2-nl316-np15-rf0 (query)                         830.83       129.53       960.37       0.7918          1.0897            1.0632         8.12
IVF-TQ-b2-nl316-np17-rf0 (query)                         830.83       137.48       968.32       0.7918          1.0898            1.0632         8.12
IVF-TQ-b2-nl316-np25-rf0 (query)                         830.83       170.75     1_001.59       0.7918          1.0898            1.0632         8.12
IVF-TQ-b2-nl316-np15-rf10 (query)                        830.83       312.31     1_143.14       0.9997          1.0000            1.0000         8.12
IVF-TQ-b2-nl316-np15-rf20 (query)                        830.83       583.32     1_414.15       0.9997          1.0000            1.0000         8.12
IVF-TQ-b2-nl316-np17-rf10 (query)                        830.83       328.62     1_159.45       0.9999          1.0000            1.0000         8.12
IVF-TQ-b2-nl316-np17-rf20 (query)                        830.83       612.97     1_443.81       0.9999          1.0000            1.0000         8.12
IVF-TQ-b2-nl316-np25-rf10 (query)                        830.83       377.33     1_208.16       1.0000          1.0000            1.0000         8.12
IVF-TQ-b2-nl316-np25-rf20 (query)                        830.83       722.11     1_552.95       1.0000          1.0000            1.0000         8.12
IVF-TQ-b2-nl316 (self)                                   830.83     1_004.18     1_835.01       1.0000          1.0000            1.0000         8.12
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_135.00       185.08     1_320.08       0.8721          1.0325            1.0187        14.02
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_135.00       253.47     1_388.46       0.8728          1.0322            1.0183        14.02
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_135.00       321.31     1_456.31       0.8728          1.0322            1.0183        14.02
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_135.00       394.04     1_529.04       0.9982          1.0004            1.0000        14.02
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_135.00       691.72     1_826.72       0.9982          1.0004            1.0000        14.02
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_135.00       486.92     1_621.91       1.0000          1.0000            1.0000        14.02
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_135.00       819.76     1_954.76       1.0000          1.0000            1.0000        14.02
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_135.00       566.28     1_701.28       1.0000          1.0000            1.0000        14.02
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_135.00       923.30     2_058.30       1.0000          1.0000            1.0000        14.02
IVF-TQ-b4-nl158 (self)                                 1_135.00     1_266.82     2_401.81       1.0000          1.0000            1.0000        14.02
IVF-TQ-b4-nl223-np11-rf0 (query)                         710.59       178.67       889.26       0.8726          1.0323            1.0184        14.23
IVF-TQ-b4-nl223-np14-rf0 (query)                         710.59       208.72       919.31       0.8727          1.0322            1.0183        14.23
IVF-TQ-b4-nl223-np21-rf0 (query)                         710.59       269.31       979.90       0.8728          1.0322            1.0183        14.23
IVF-TQ-b4-nl223-np11-rf10 (query)                        710.59       374.02     1_084.62       0.9995          1.0001            1.0000        14.23
IVF-TQ-b4-nl223-np11-rf20 (query)                        710.59       699.18     1_409.77       0.9995          1.0001            1.0000        14.23
IVF-TQ-b4-nl223-np14-rf10 (query)                        710.59       417.87     1_128.46       0.9999          1.0000            1.0000        14.23
IVF-TQ-b4-nl223-np14-rf20 (query)                        710.59       712.68     1_423.27       0.9999          1.0000            1.0000        14.23
IVF-TQ-b4-nl223-np21-rf10 (query)                        710.59       493.34     1_203.93       1.0000          1.0000            1.0000        14.23
IVF-TQ-b4-nl223-np21-rf20 (query)                        710.59       825.33     1_535.92       1.0000          1.0000            1.0000        14.23
IVF-TQ-b4-nl223 (self)                                   710.59     1_117.21     1_827.80       1.0000          1.0000            1.0000        14.23
IVF-TQ-b4-nl316-np15-rf0 (query)                         892.85       179.36     1_072.21       0.8727          1.0322            1.0184        14.54
IVF-TQ-b4-nl316-np17-rf0 (query)                         892.85       195.12     1_087.98       0.8727          1.0322            1.0183        14.54
IVF-TQ-b4-nl316-np25-rf0 (query)                         892.85       246.44     1_139.30       0.8727          1.0322            1.0183        14.54
IVF-TQ-b4-nl316-np15-rf10 (query)                        892.85       366.51     1_259.36       0.9997          1.0000            1.0000        14.54
IVF-TQ-b4-nl316-np15-rf20 (query)                        892.85       638.08     1_530.93       0.9997          1.0000            1.0000        14.54
IVF-TQ-b4-nl316-np17-rf10 (query)                        892.85       384.91     1_277.77       0.9999          1.0000            1.0000        14.54
IVF-TQ-b4-nl316-np17-rf20 (query)                        892.85       666.04     1_558.89       0.9999          1.0000            1.0000        14.54
IVF-TQ-b4-nl316-np25-rf10 (query)                        892.85       453.39     1_346.24       1.0000          1.0000            1.0000        14.54
IVF-TQ-b4-nl316-np25-rf20 (query)                        892.85       760.51     1_653.36       1.0000          1.0000            1.0000        14.54
IVF-TQ-b4-nl316 (self)                                   892.85     1_063.07     1_955.92       1.0000          1.0000            1.0000        14.54
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
Exhaustive (query)                                        68.89     1_291.80     1_360.69       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.89     4_548.45     4_617.34       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              336.17       627.86       964.03       0.8424          1.0447            1.0331        13.97
ExhaustiveTQ-b2-rf5 (query)                              336.17       737.72     1_073.88       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2-rf10 (query)                             336.17       870.20     1_206.37       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2-rf20 (query)                             336.17     1_305.17     1_641.34       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2 (self)                                   336.17     4_243.05     4_579.21       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b4-rf0 (query)                              459.90     1_114.36     1_574.26       0.8985          1.0191            1.0110        26.18
ExhaustiveTQ-b4-rf5 (query)                              459.90     1_218.98     1_678.88       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4-rf10 (query)                             459.90     1_365.61     1_825.51       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4-rf20 (query)                             459.90     1_834.91     2_294.81       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4 (self)                                   459.90     5_846.96     6_306.86       1.0000          1.0000            1.0000        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        2_079.77       229.64     2_309.40       0.8420          1.0449            1.0333        14.96
IVF-TQ-b2-nl158-np12-rf0 (query)                       2_079.77       297.86     2_377.63       0.8424          1.0447            1.0331        14.96
IVF-TQ-b2-nl158-np17-rf0 (query)                       2_079.77       357.00     2_436.76       0.8424          1.0447            1.0331        14.96
IVF-TQ-b2-nl158-np7-rf10 (query)                       2_079.77       459.34     2_539.11       0.9986          1.0003            1.0000        14.96
IVF-TQ-b2-nl158-np7-rf20 (query)                       2_079.77       781.12     2_860.89       0.9986          1.0003            1.0000        14.96
IVF-TQ-b2-nl158-np12-rf10 (query)                      2_079.77       545.43     2_625.20       0.9999          1.0000            1.0000        14.96
IVF-TQ-b2-nl158-np12-rf20 (query)                      2_079.77       904.81     2_984.57       0.9999          1.0000            1.0000        14.96
IVF-TQ-b2-nl158-np17-rf10 (query)                      2_079.77       626.45     2_706.21       1.0000          1.0000            1.0000        14.96
IVF-TQ-b2-nl158-np17-rf20 (query)                      2_079.77       996.03     3_075.80       1.0000          1.0000            1.0000        14.96
IVF-TQ-b2-nl158 (self)                                 2_079.77     1_558.84     3_638.61       1.0000          1.0000            1.0000        14.96
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_131.55       236.26     1_367.81       0.8423          1.0447            1.0331        15.25
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_131.55       261.32     1_392.87       0.8424          1.0447            1.0331        15.25
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_131.55       319.70     1_451.25       0.8424          1.0447            1.0331        15.25
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_131.55       453.09     1_584.63       0.9997          1.0000            1.0000        15.25
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_131.55       744.45     1_876.00       0.9997          1.0000            1.0000        15.25
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_131.55       481.09     1_612.64       0.9999          1.0000            1.0000        15.25
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_131.55       796.16     1_927.70       0.9999          1.0000            1.0000        15.25
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_131.55       554.11     1_685.65       1.0000          1.0000            1.0000        15.25
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_131.55       883.31     2_014.85       1.0000          1.0000            1.0000        15.25
IVF-TQ-b2-nl223 (self)                                 1_131.55     1_470.66     2_602.21       1.0000          1.0000            1.0000        15.25
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_339.38       242.07     1_581.45       0.8424          1.0447            1.0331        15.56
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_339.38       250.50     1_589.87       0.8424          1.0447            1.0331        15.56
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_339.38       301.13     1_640.51       0.8424          1.0447            1.0331        15.56
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_339.38       473.15     1_812.52       0.9999          1.0000            1.0000        15.56
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_339.38       749.31     2_088.69       0.9999          1.0000            1.0000        15.56
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_339.38       464.28     1_803.65       1.0000          1.0000            1.0000        15.56
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_339.38       776.40     2_115.78       1.0000          1.0000            1.0000        15.56
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_339.38       525.06     1_864.44       1.0000          1.0000            1.0000        15.56
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_339.38       858.45     2_197.83       1.0000          1.0000            1.0000        15.56
IVF-TQ-b2-nl316 (self)                                 1_339.38     1_425.95     2_765.33       1.0000          1.0000            1.0000        15.56
IVF-TQ-b4-nl158-np7-rf0 (query)                        2_167.75       334.22     2_501.97       0.8977          1.0194            1.0113        27.46
IVF-TQ-b4-nl158-np12-rf0 (query)                       2_167.75       462.77     2_630.52       0.8985          1.0191            1.0110        27.46
IVF-TQ-b4-nl158-np17-rf0 (query)                       2_167.75       571.70     2_739.45       0.8985          1.0191            1.0110        27.46
IVF-TQ-b4-nl158-np7-rf10 (query)                       2_167.75       567.73     2_735.48       0.9986          1.0003            1.0000        27.46
IVF-TQ-b4-nl158-np7-rf20 (query)                       2_167.75       889.36     3_057.11       0.9986          1.0003            1.0000        27.46
IVF-TQ-b4-nl158-np12-rf10 (query)                      2_167.75       706.11     2_873.86       0.9999          1.0000            1.0000        27.46
IVF-TQ-b4-nl158-np12-rf20 (query)                      2_167.75     1_066.54     3_234.29       0.9999          1.0000            1.0000        27.46
IVF-TQ-b4-nl158-np17-rf10 (query)                      2_167.75       823.25     2_991.00       1.0000          1.0000            1.0000        27.46
IVF-TQ-b4-nl158-np17-rf20 (query)                      2_167.75     1_201.61     3_369.36       1.0000          1.0000            1.0000        27.46
IVF-TQ-b4-nl158 (self)                                 2_167.75     1_832.31     4_000.06       1.0000          1.0000            1.0000        27.46
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_231.45       346.85     1_578.31       0.8984          1.0191            1.0111        27.91
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_231.45       399.09     1_630.55       0.8985          1.0191            1.0110        27.91
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_231.45       506.95     1_738.40       0.8985          1.0191            1.0110        27.91
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_231.45       566.28     1_797.74       0.9997          1.0000            1.0000        27.91
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_231.45       857.62     2_089.07       0.9997          1.0000            1.0000        27.91
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_231.45       619.60     1_851.06       0.9999          1.0000            1.0000        27.91
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_231.45       930.53     2_161.98       0.9999          1.0000            1.0000        27.91
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_231.45       738.44     1_969.89       1.0000          1.0000            1.0000        27.91
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_231.45     1_072.69     2_304.15       1.0000          1.0000            1.0000        27.91
IVF-TQ-b4-nl223 (self)                                 1_231.45     1_722.02     2_953.47       1.0000          1.0000            1.0000        27.91
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_459.36       346.48     1_805.84       0.8985          1.0191            1.0110        28.36
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_459.36       372.52     1_831.88       0.8985          1.0191            1.0110        28.36
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_459.36       467.89     1_927.25       0.8985          1.0191            1.0110        28.36
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_459.36       555.67     2_015.03       0.9999          1.0000            1.0000        28.36
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_459.36       860.01     2_319.37       0.9999          1.0000            1.0000        28.36
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_459.36       583.26     2_042.62       1.0000          1.0000            1.0000        28.36
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_459.36       906.78     2_366.14       1.0000          1.0000            1.0000        28.36
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_459.36       686.49     2_145.85       1.0000          1.0000            1.0000        28.36
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_459.36     1_037.08     2_496.44       1.0000          1.0000            1.0000        28.36
IVF-TQ-b4-nl316 (self)                                 1_459.36     1_665.23     3_124.59       1.0000          1.0000            1.0000        28.36
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
Exhaustive (query)                                       100.60     2_041.41     2_142.00       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.60     6_700.97     6_801.56       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              614.03       986.23     1_600.26       0.8736          1.0271            1.0199        21.33
ExhaustiveTQ-b2-rf5 (query)                              614.03     1_074.66     1_688.69       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2-rf10 (query)                             614.03     1_224.86     1_838.89       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2-rf20 (query)                             614.03     1_646.36     2_260.39       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2 (self)                                   614.03     5_404.63     6_018.66       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b4-rf0 (query)                              763.69     1_797.63     2_561.32       0.9097          1.0146            1.0083        39.64
ExhaustiveTQ-b4-rf5 (query)                              763.69     1_894.23     2_657.92       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4-rf10 (query)                             763.69     2_035.16     2_798.85       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4-rf20 (query)                             763.69     2_431.83     3_195.51       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4 (self)                                   763.69     8_012.72     8_776.40       1.0000          1.0000            1.0000        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        3_125.72       354.80     3_480.53       0.8735          1.0272            1.0201        22.61
IVF-TQ-b2-nl158-np12-rf0 (query)                       3_125.72       457.28     3_583.01       0.8736          1.0271            1.0199        22.61
IVF-TQ-b2-nl158-np17-rf0 (query)                       3_125.72       530.39     3_656.11       0.8736          1.0271            1.0199        22.61
IVF-TQ-b2-nl158-np7-rf10 (query)                       3_125.72       610.03     3_735.75       0.9995          1.0001            1.0000        22.61
IVF-TQ-b2-nl158-np7-rf20 (query)                       3_125.72       958.34     4_084.07       0.9995          1.0001            1.0000        22.61
IVF-TQ-b2-nl158-np12-rf10 (query)                      3_125.72       720.38     3_846.10       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158-np12-rf20 (query)                      3_125.72     1_090.61     4_216.33       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158-np17-rf10 (query)                      3_125.72       808.97     3_934.69       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158-np17-rf20 (query)                      3_125.72     1_212.02     4_337.74       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158 (self)                                 3_125.72     2_094.00     5_219.73       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_694.02       353.90     2_047.92       0.8736          1.0271            1.0200        23.01
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_694.02       388.62     2_082.65       0.8736          1.0271            1.0199        23.01
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_694.02       472.10     2_166.12       0.8736          1.0271            1.0199        23.01
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_694.02       590.57     2_284.60       0.9998          1.0000            1.0000        23.01
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_694.02       925.34     2_619.36       0.9998          1.0000            1.0000        23.01
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_694.02       635.65     2_329.67       1.0000          1.0000            1.0000        23.01
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_694.02       984.30     2_678.32       1.0000          1.0000            1.0000        23.01
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_694.02       729.89     2_423.92       1.0000          1.0000            1.0000        23.01
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_694.02     1_100.93     2_794.96       1.0000          1.0000            1.0000        23.01
IVF-TQ-b2-nl223 (self)                                 1_694.02     1_954.81     3_648.83       1.0000          1.0000            1.0000        23.01
IVF-TQ-b2-nl316-np15-rf0 (query)                       2_008.90       354.07     2_362.97       0.8736          1.0271            1.0200        23.53
IVF-TQ-b2-nl316-np17-rf0 (query)                       2_008.90       375.90     2_384.80       0.8736          1.0271            1.0200        23.53
IVF-TQ-b2-nl316-np25-rf0 (query)                       2_008.90       453.80     2_462.69       0.8736          1.0271            1.0199        23.53
IVF-TQ-b2-nl316-np15-rf10 (query)                      2_008.90       598.99     2_607.88       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np15-rf20 (query)                      2_008.90       927.62     2_936.51       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np17-rf10 (query)                      2_008.90       617.28     2_626.17       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np17-rf20 (query)                      2_008.90       956.23     2_965.13       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np25-rf10 (query)                      2_008.90       699.85     2_708.75       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np25-rf20 (query)                      2_008.90     1_069.23     3_078.12       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316 (self)                                 2_008.90     2_005.45     4_014.35       1.0000          1.0000            1.0000        23.53
IVF-TQ-b4-nl158-np7-rf0 (query)                        3_196.33       524.24     3_720.57       0.9095          1.0147            1.0084        41.37
IVF-TQ-b4-nl158-np12-rf0 (query)                       3_196.33       720.95     3_917.28       0.9097          1.0146            1.0083        41.37
IVF-TQ-b4-nl158-np17-rf0 (query)                       3_196.33       857.75     4_054.08       0.9097          1.0146            1.0083        41.37
IVF-TQ-b4-nl158-np7-rf10 (query)                       3_196.33       787.99     3_984.33       0.9995          1.0001            1.0000        41.37
IVF-TQ-b4-nl158-np7-rf20 (query)                       3_196.33     1_121.30     4_317.63       0.9995          1.0001            1.0000        41.37
IVF-TQ-b4-nl158-np12-rf10 (query)                      3_196.33       977.60     4_173.93       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158-np12-rf20 (query)                      3_196.33     1_346.93     4_543.26       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158-np17-rf10 (query)                      3_196.33     1_138.12     4_334.45       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158-np17-rf20 (query)                      3_196.33     1_618.60     4_814.93       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158 (self)                                 3_196.33     2_634.70     5_831.03       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_867.99       528.96     2_396.95       0.9096          1.0146            1.0084        41.97
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_867.99       610.25     2_478.24       0.9097          1.0146            1.0083        41.97
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_867.99       776.46     2_644.45       0.9097          1.0146            1.0083        41.97
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_867.99       765.76     2_633.75       0.9998          1.0000            1.0000        41.97
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_867.99     1_091.32     2_959.31       0.9998          1.0000            1.0000        41.97
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_867.99       845.11     2_713.10       1.0000          1.0000            1.0000        41.97
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_867.99     1_194.06     3_062.05       1.0000          1.0000            1.0000        41.97
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_867.99     1_020.50     2_888.49       1.0000          1.0000            1.0000        41.97
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_867.99     1_379.79     3_247.78       1.0000          1.0000            1.0000        41.97
IVF-TQ-b4-nl223 (self)                                 1_867.99     2_520.49     4_388.48       1.0000          1.0000            1.0000        41.97
IVF-TQ-b4-nl316-np15-rf0 (query)                       2_198.81       549.64     2_748.45       0.9097          1.0146            1.0083        42.73
IVF-TQ-b4-nl316-np17-rf0 (query)                       2_198.81       588.05     2_786.87       0.9097          1.0146            1.0083        42.73
IVF-TQ-b4-nl316-np25-rf0 (query)                       2_198.81       732.16     2_930.97       0.9097          1.0146            1.0083        42.73
IVF-TQ-b4-nl316-np15-rf10 (query)                      2_198.81       773.11     2_971.93       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np15-rf20 (query)                      2_198.81     1_094.31     3_293.12       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np17-rf10 (query)                      2_198.81       814.04     3_012.85       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np17-rf20 (query)                      2_198.81     1_151.10     3_349.91       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np25-rf10 (query)                      2_198.81       964.31     3_163.12       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np25-rf20 (query)                      2_198.81     1_322.99     3_521.80       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316 (self)                                 2_198.81     2_449.47     4_648.29       1.0000          1.0000            1.0000        42.73
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
