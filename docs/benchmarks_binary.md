## Binarised indices benchmarks and parameter

Binarised indices push the compression to (roughly) bits. Three consequences:

1. The index footprint collapses.
2. Queries usually get faster, because bitwise operations are cheap on modern
   CPUs.
3. Without re-ranking the top candidates, recall drops hard. Less so for RaBitQ,
   and for TurboQuant it depends on the data.

The quantised graph index (QG) is the exception to the first point. It spends
the bits on locality rather than on compression and comes out *larger* than the
raw vectors, so read it as a speed structure that happens to use RaBitQ codes,
not as a small index. See the section below.

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
  - [Quantised graph](#quantised-graph-qg)
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
Exhaustive (query)                                        32.71       681.93       714.64       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.71     2_193.11     2_225.82       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                 68.92       237.16       306.09       0.1199          1.4617            1.4199         1.78
ExhaustiveBinary-256-random-rf10 (query)                  68.92       336.14       405.07       0.3411          1.0941            1.0814         1.78
ExhaustiveBinary-256-random-rf20 (query)                  68.92       430.64       499.57       0.4467          1.0571            1.0475         1.78
ExhaustiveBinary-256-random (self)                        68.92     1_073.48     1_142.40       0.3454          1.0895            1.0798         1.78
ExhaustiveBinary-256-pca_no_rr (query)                   103.39       237.14       340.53       0.1153          1.4748            1.4212         1.78
ExhaustiveBinary-256-pca-rf10 (query)                    103.39       335.35       438.75       0.3323          1.1029            1.0834         1.78
ExhaustiveBinary-256-pca-rf20 (query)                    103.39       431.51       534.91       0.4387          1.0631            1.0485         1.78
ExhaustiveBinary-256-pca (self)                          103.39     1_075.56     1_178.95       0.3391          1.0957            1.0813         1.78
ExhaustiveBinary-512-random_no_rr (query)                 93.08       352.90       445.98       0.1588          1.3547            1.3300         3.55
ExhaustiveBinary-512-random-rf10 (query)                  93.08       470.16       563.24       0.3786          1.0692            1.0677         3.55
ExhaustiveBinary-512-random-rf20 (query)                  93.08       565.47       658.55       0.4874          1.0424            1.0395         3.55
ExhaustiveBinary-512-random (self)                        93.08     1_497.10     1_590.18       0.3805          1.0675            1.0675         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   128.75       354.84       483.60       0.1564          1.3535            1.3265         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    128.75       461.99       590.75       0.3789          1.0710            1.0663         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    128.75       561.15       689.91       0.4903          1.0433            1.0387         3.55
ExhaustiveBinary-512-pca (self)                          128.75     1_492.94     1_621.70       0.3823          1.0678            1.0665         3.55
ExhaustiveBinary-1024-random_no_rr (query)               130.92       488.68       619.61       0.1929          1.2764            1.2696         7.10
ExhaustiveBinary-1024-random-rf10 (query)                130.92       607.72       738.65       0.4214          1.0550            1.0552         7.10
ExhaustiveBinary-1024-random-rf20 (query)                130.92       715.26       846.18       0.5434          1.0327            1.0308         7.10
ExhaustiveBinary-1024-random (self)                      130.92     1_976.19     2_107.11       0.4232          1.0547            1.0552         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  160.71       487.88       648.59       0.1921          1.2733            1.2652         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   160.71       605.28       765.99       0.4226          1.0546            1.0544         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   160.71       746.53       907.24       0.5443          1.0326            1.0305         7.10
ExhaustiveBinary-1024-pca (self)                         160.71     1_990.20     2_150.91       0.4236          1.0546            1.0548         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   58.61       428.55       487.16       0.1211          1.4987            1.4523         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    58.61       458.98       517.60       0.3284          1.1039            1.0884         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    58.61       700.98       759.60       0.4385          1.0624            1.0494         1.53
ExhaustiveBinary-256-sign (self)                          58.61     1_502.99     1_561.60       0.3334          1.0988            1.0859         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)              801.07        48.91       849.98       0.1231          1.4432            1.4051         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)             801.07        52.06       853.12       0.1231          1.4432            1.4051         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)             801.07        51.71       852.78       0.1231          1.4432            1.4051         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)             801.07        97.84       898.91       0.3463          1.0912            1.0794         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)             801.07       146.51       947.58       0.4529          1.0552            1.0461         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)            801.07        98.06       899.13       0.3463          1.0912            1.0794         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)            801.07       147.52       948.59       0.4529          1.0552            1.0461         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)            801.07        96.29       897.36       0.3463          1.0912            1.0794         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)            801.07       147.47       948.54       0.4529          1.0552            1.0461         1.93
IVF-Binary-256-nl158-random (self)                       801.07       213.60     1_014.67       0.3507          1.0862            1.0777         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             588.80        44.31       633.10       0.1413          1.3601            1.3150         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             588.80        45.15       633.95       0.1412          1.3605            1.3154         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             588.80        48.63       637.43       0.1412          1.3605            1.3154         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            588.80        97.43       686.23       0.3893          1.0689            1.0625         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            588.80       148.04       736.84       0.4976          1.0427            1.0371         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            588.80        98.03       686.83       0.3891          1.0690            1.0625         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            588.80       146.71       735.51       0.4973          1.0428            1.0372         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            588.80       101.51       690.31       0.3891          1.0690            1.0625         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            588.80       150.69       739.49       0.4973          1.0428            1.0372         2.00
IVF-Binary-256-nl223-random (self)                       588.80       213.53       802.33       0.3943          1.0643            1.0615         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             758.68        47.09       805.77       0.1496          1.3359            1.2904         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             758.68        49.05       807.72       0.1495          1.3365            1.2906         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             758.68        50.73       809.40       0.1495          1.3366            1.2906         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            758.68        98.10       856.77       0.4018          1.0649            1.0585         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            758.68       148.99       907.67       0.5055          1.0413            1.0358         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            758.68        95.62       854.30       0.4016          1.0650            1.0587         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            758.68       148.24       906.92       0.5051          1.0414            1.0359         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            758.68        99.48       858.16       0.4016          1.0650            1.0587         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            758.68       153.89       912.56       0.5051          1.0414            1.0359         2.09
IVF-Binary-256-nl316-random (self)                       758.68       218.71       977.39       0.4063          1.0606            1.0577         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)                 823.42        41.20       864.62       0.1190          1.4540            1.4124         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)                823.42        42.20       865.62       0.1190          1.4540            1.4124         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)                823.42        43.21       866.63       0.1190          1.4540            1.4124         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)                823.42        89.37       912.79       0.3368          1.0992            1.0816         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)                823.42       143.98       967.40       0.4434          1.0613            1.0475         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)               823.42        93.45       916.86       0.3368          1.0992            1.0816         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)               823.42       145.64       969.06       0.4434          1.0613            1.0475         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)               823.42        91.12       914.54       0.3368          1.0992            1.0816         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)               823.42       145.41       968.83       0.4434          1.0613            1.0475         1.93
IVF-Binary-256-nl158-pca (self)                          823.42       198.91     1_022.33       0.3434          1.0923            1.0796         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                598.36        43.86       642.22       0.1377          1.3704            1.3177         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                598.36        44.93       643.29       0.1377          1.3708            1.3180         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                598.36        49.52       647.88       0.1377          1.3708            1.3180         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               598.36        93.87       692.23       0.3828          1.0754            1.0637         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               598.36       144.74       743.10       0.4958          1.0458            1.0375         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               598.36        94.29       692.65       0.3827          1.0755            1.0638         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               598.36       146.66       745.02       0.4957          1.0458            1.0375         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               598.36        95.43       693.79       0.3827          1.0755            1.0638         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               598.36       148.80       747.16       0.4957          1.0458            1.0375         2.00
IVF-Binary-256-nl223-pca (self)                          598.36       206.82       805.18       0.3896          1.0692            1.0620         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                800.13        46.77       846.90       0.1471          1.3419            1.2914         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                800.13        48.06       848.19       0.1471          1.3424            1.2916         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                800.13        50.43       850.56       0.1471          1.3425            1.2916         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               800.13        95.81       895.93       0.3970          1.0703            1.0594         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               800.13       147.55       947.68       0.5069          1.0437            1.0356         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               800.13        95.34       895.46       0.3968          1.0704            1.0594         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               800.13       147.20       947.33       0.5067          1.0438            1.0356         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               800.13        98.34       898.47       0.3968          1.0704            1.0594         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               800.13       149.72       949.85       0.5067          1.0438            1.0356         2.09
IVF-Binary-256-nl316-pca (self)                          800.13       216.23     1_016.35       0.4025          1.0646            1.0582         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)              804.03        59.12       863.15       0.1607          1.3465            1.3235         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)             804.03        61.07       865.10       0.1607          1.3465            1.3235         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)             804.03        63.51       867.54       0.1607          1.3465            1.3235         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)             804.03       112.97       917.00       0.3812          1.0681            1.0667         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)             804.03       167.48       971.50       0.4908          1.0417            1.0390         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)            804.03       114.90       918.93       0.3812          1.0681            1.0667         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)            804.03       171.62       975.65       0.4908          1.0417            1.0390         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)            804.03       119.44       923.47       0.3812          1.0681            1.0667         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)            804.03       173.87       977.90       0.4908          1.0417            1.0390         3.71
IVF-Binary-512-nl158-random (self)                       804.03       286.92     1_090.95       0.3832          1.0664            1.0666         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             588.20        62.15       650.35       0.1711          1.2998            1.2780         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             588.20        64.42       652.62       0.1711          1.3001            1.2782         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             588.20        68.48       656.67       0.1711          1.3001            1.2782         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            588.20       115.27       703.47       0.4019          1.0605            1.0593         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            588.20       169.42       757.62       0.5140          1.0373            1.0348         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            588.20       116.33       704.52       0.4017          1.0605            1.0593         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            588.20       170.90       759.10       0.5137          1.0374            1.0349         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            588.20       119.96       708.15       0.4017          1.0605            1.0593         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            588.20       175.51       763.71       0.5137          1.0374            1.0349         3.77
IVF-Binary-512-nl223-random (self)                       588.20       289.40       877.59       0.4039          1.0592            1.0592         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             760.97        64.98       825.94       0.1755          1.2880            1.2669         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             760.97        66.62       827.59       0.1754          1.2885            1.2672         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             760.97        70.37       831.34       0.1754          1.2885            1.2672         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            760.97       118.14       879.11       0.4058          1.0594            1.0581         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            760.97       171.47       932.43       0.5175          1.0368            1.0342         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            760.97       117.43       878.40       0.4056          1.0595            1.0582         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            760.97       171.99       932.96       0.5171          1.0369            1.0342         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            760.97       122.82       883.79       0.4056          1.0595            1.0582         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            760.97       176.65       937.62       0.5171          1.0369            1.0342         3.86
IVF-Binary-512-nl316-random (self)                       760.97       296.95     1_057.91       0.4086          1.0581            1.0580         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)                 840.97        57.94       898.90       0.1581          1.3474            1.3214         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)                840.97        60.76       901.73       0.1581          1.3474            1.3214         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)                840.97        63.43       904.40       0.1581          1.3474            1.3214         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)                840.97       112.16       953.13       0.3810          1.0702            1.0658         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)                840.97       172.65     1_013.62       0.4928          1.0428            1.0383         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)               840.97       113.65       954.61       0.3810          1.0702            1.0658         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)               840.97       170.79     1_011.76       0.4928          1.0428            1.0383         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)               840.97       116.29       957.26       0.3810          1.0702            1.0658         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)               840.97       173.58     1_014.55       0.4928          1.0428            1.0383         3.71
IVF-Binary-512-nl158-pca (self)                          840.97       286.72     1_127.69       0.3843          1.0672            1.0657         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                623.43        62.86       686.29       0.1694          1.3033            1.2746         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                623.43        64.38       687.80       0.1694          1.3034            1.2747         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                623.43        69.50       692.93       0.1694          1.3034            1.2747         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               623.43       117.62       741.05       0.4036          1.0619            1.0583         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               623.43       168.04       791.47       0.5164          1.0383            1.0342         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               623.43       114.03       737.46       0.4035          1.0619            1.0583         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               623.43       170.19       793.62       0.5162          1.0384            1.0342         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               623.43       118.01       741.44       0.4035          1.0619            1.0583         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               623.43       174.38       797.81       0.5162          1.0384            1.0342         3.77
IVF-Binary-512-nl223-pca (self)                          623.43       284.36       907.79       0.4059          1.0599            1.0584         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                802.69        65.01       867.70       0.1733          1.2925            1.2652         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                802.69        65.92       868.62       0.1732          1.2927            1.2653         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                802.69        69.94       872.63       0.1732          1.2927            1.2653         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               802.69       122.18       924.88       0.4089          1.0604            1.0565         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               802.69       169.31       972.01       0.5220          1.0374            1.0334         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               802.69       117.11       919.80       0.4088          1.0605            1.0565         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               802.69       171.09       973.79       0.5217          1.0374            1.0334         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               802.69       120.94       923.63       0.4088          1.0605            1.0565         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               802.69       175.68       978.37       0.5217          1.0374            1.0334         3.86
IVF-Binary-512-nl316-pca (self)                          802.69       291.76     1_094.45       0.4117          1.0583            1.0566         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)             856.27        90.14       946.41       0.1937          1.2738            1.2669         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)            856.27        94.44       950.71       0.1937          1.2738            1.2669         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)            856.27        97.76       954.03       0.1937          1.2738            1.2669         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)            856.27       147.95     1_004.22       0.4227          1.0546            1.0549         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)            856.27       213.56     1_069.83       0.5450          1.0325            1.0305         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)           856.27       151.27     1_007.54       0.4227          1.0546            1.0549         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)           856.27       210.13     1_066.40       0.5450          1.0325            1.0305         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)           856.27       156.21     1_012.48       0.4227          1.0546            1.0549         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)           856.27       215.78     1_072.05       0.5450          1.0325            1.0305         7.26
IVF-Binary-1024-nl158-random (self)                      856.27       407.60     1_263.87       0.4246          1.0544            1.0549         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            627.81        93.08       720.89       0.1973          1.2556            1.2488         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            627.81        95.88       723.69       0.1973          1.2558            1.2489         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            627.81       102.99       730.81       0.1973          1.2558            1.2489         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           627.81       152.53       780.34       0.4342          1.0516            1.0516         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           627.81       205.65       833.46       0.5563          1.0309            1.0289         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           627.81       150.89       778.71       0.4341          1.0517            1.0516         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           627.81       208.79       836.60       0.5562          1.0309            1.0289         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           627.81       156.23       784.04       0.4341          1.0517            1.0516         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           627.81       215.26       843.08       0.5562          1.0309            1.0289         7.32
IVF-Binary-1024-nl223-random (self)                      627.81       404.89     1_032.71       0.4353          1.0515            1.0518         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            809.52        96.45       905.97       0.1989          1.2508            1.2444         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)            809.52        97.76       907.28       0.1988          1.2511            1.2446         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)            809.52       103.91       913.43       0.1988          1.2511            1.2446         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)           809.52       152.69       962.21       0.4365          1.0510            1.0509         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)           809.52       207.46     1_016.98       0.5576          1.0306            1.0287         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)           809.52       153.26       962.78       0.4364          1.0511            1.0509         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)           809.52       210.08     1_019.60       0.5573          1.0307            1.0288         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)           809.52       160.21       969.73       0.4364          1.0511            1.0509         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)           809.52       216.13     1_025.65       0.5573          1.0307            1.0288         7.42
IVF-Binary-1024-nl316-random (self)                      809.52       414.29     1_223.81       0.4378          1.0509            1.0513         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)                897.89        89.71       987.59       0.1929          1.2710            1.2632         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)               897.89        92.88       990.77       0.1929          1.2710            1.2632         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)               897.89        96.82       994.70       0.1929          1.2710            1.2632         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)               897.89       152.56     1_050.45       0.4240          1.0542            1.0540         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)               897.89       203.92     1_101.81       0.5461          1.0324            1.0302         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)              897.89       151.43     1_049.32       0.4240          1.0542            1.0540         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)              897.89       209.75     1_107.63       0.5461          1.0324            1.0302         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)              897.89       153.95     1_051.83       0.4240          1.0542            1.0540         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)              897.89       214.91     1_112.80       0.5461          1.0324            1.0302         7.26
IVF-Binary-1024-nl158-pca (self)                         897.89       406.22     1_304.10       0.4249          1.0542            1.0544         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               655.72        92.62       748.34       0.1974          1.2518            1.2441         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               655.72        95.00       750.72       0.1974          1.2519            1.2442         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               655.72       102.11       757.84       0.1974          1.2519            1.2442         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              655.72       151.15       806.88       0.4355          1.0511            1.0508         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              655.72       206.39       862.11       0.5584          1.0305            1.0281         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              655.72       150.39       806.12       0.4354          1.0511            1.0508         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              655.72       208.30       864.03       0.5581          1.0306            1.0282         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              655.72       155.47       811.19       0.4354          1.0511            1.0508         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              655.72       213.71       869.43       0.5581          1.0306            1.0282         7.32
IVF-Binary-1024-nl223-pca (self)                         655.72       406.39     1_062.11       0.4363          1.0511            1.0512         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               848.21        96.38       944.59       0.1987          1.2475            1.2401         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)               848.21        97.86       946.06       0.1987          1.2476            1.2401         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)               848.21       103.32       951.53       0.1987          1.2476            1.2401         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)              848.21       152.21     1_000.41       0.4386          1.0503            1.0501         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)              848.21       207.36     1_055.57       0.5611          1.0302            1.0279         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)              848.21       152.12     1_000.33       0.4384          1.0504            1.0501         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)              848.21       211.27     1_059.47       0.5610          1.0302            1.0279         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)              848.21       158.46     1_006.67       0.4384          1.0504            1.0501         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)              848.21       215.32     1_063.53       0.5610          1.0302            1.0279         7.42
IVF-Binary-1024-nl316-pca (self)                         848.21       413.96     1_262.16       0.4396          1.0504            1.0505         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)                800.02       147.61       947.63       0.1216          1.4959            1.4465         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               800.02       150.39       950.41       0.1216          1.4959            1.4465         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               800.02       151.39       951.41       0.1216          1.4959            1.4465         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               800.02       183.11       983.13       0.3305          1.1023            1.0867         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               800.02       325.23     1_125.24       0.4405          1.0617            1.0486         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              800.02       191.99       992.00       0.3305          1.1023            1.0867         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              800.02       330.32     1_130.34       0.4405          1.0617            1.0486         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              800.02       185.94       985.96       0.3305          1.1023            1.0867         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              800.02       330.78     1_130.79       0.4405          1.0617            1.0486         1.68
IVF-Binary-256-nl158-sign (self)                         800.02       502.61     1_302.63       0.3358          1.0972            1.0848         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               562.05       147.12       709.17       0.1228          1.4832            1.4276         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               562.05       149.89       711.94       0.1228          1.4843            1.4282         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               562.05       151.36       713.41       0.1228          1.4844            1.4282         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              562.05       184.77       746.82       0.3559          1.0888            1.0752         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              562.05       328.45       890.50       0.4583          1.0561            1.0442         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              562.05       185.30       747.35       0.3557          1.0890            1.0752         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              562.05       329.01       891.06       0.4580          1.0562            1.0443         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              562.05       196.55       758.61       0.3557          1.0890            1.0752         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              562.05       332.00       894.05       0.4580          1.0562            1.0443         1.75
IVF-Binary-256-nl223-sign (self)                         562.05       500.75     1_062.80       0.3611          1.0841            1.0736         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               738.20       150.18       888.38       0.1239          1.4676            1.4148         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               738.20       150.57       888.77       0.1237          1.4698            1.4160         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               738.20       155.20       893.39       0.1237          1.4700            1.4161         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              738.20       188.90       927.09       0.3598          1.0872            1.0741         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              738.20       331.63     1_069.82       0.4582          1.0559            1.0445         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              738.20       189.66       927.86       0.3596          1.0874            1.0742         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              738.20       329.59     1_067.79       0.4577          1.0561            1.0446         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              738.20       201.41       939.61       0.3596          1.0874            1.0742         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              738.20       336.81     1_075.00       0.4577          1.0561            1.0446         1.84
IVF-Binary-256-nl316-sign (self)                         738.20       513.90     1_252.10       0.3647          1.0824            1.0724         1.84
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
Exhaustive (query)                                        69.04     1_302.14     1_371.18       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.04     4_347.22     4_416.25       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                131.56       268.12       399.68       0.1109          1.3512            1.3092         2.03
ExhaustiveBinary-256-random-rf10 (query)                 131.56       383.58       515.14       0.3143          1.0825            1.0613         2.03
ExhaustiveBinary-256-random-rf20 (query)                 131.56       506.81       638.37       0.4100          1.0523            1.0368         2.03
ExhaustiveBinary-256-random (self)                       131.56     1_194.54     1_326.09       0.3161          1.0784            1.0600         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   229.29       263.67       492.96       0.1167          1.3480            1.2981         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    229.29       385.43       614.72       0.3159          1.0791            1.0596         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    229.29       512.45       741.74       0.4121          1.0505            1.0362         2.03
ExhaustiveBinary-256-pca (self)                          229.29     1_205.50     1_434.80       0.3171          1.0782            1.0588         2.03
ExhaustiveBinary-512-random_no_rr (query)                204.35       391.27       595.62       0.1528          1.2601            1.2299         4.05
ExhaustiveBinary-512-random-rf10 (query)                 204.35       530.08       734.43       0.3465          1.0565            1.0514         4.05
ExhaustiveBinary-512-random-rf20 (query)                 204.35       654.52       858.87       0.4452          1.0358            1.0312         4.05
ExhaustiveBinary-512-random (self)                       204.35     1_664.17     1_868.52       0.3476          1.0547            1.0512         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   298.76       400.90       699.66       0.1558          1.2535            1.2254         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    298.76       534.13       832.89       0.3512          1.0523            1.0507         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    298.76       669.01       967.77       0.4484          1.0329            1.0309         4.05
ExhaustiveBinary-512-pca (self)                          298.76     1_673.98     1_972.73       0.3515          1.0522            1.0505         4.05
ExhaustiveBinary-1024-random_no_rr (query)               256.61       562.11       818.72       0.1816          1.2043            1.1936         8.11
ExhaustiveBinary-1024-random-rf10 (query)                256.61       710.63       967.24       0.3747          1.0447            1.0452         8.11
ExhaustiveBinary-1024-random-rf20 (query)                256.61       852.33     1_108.95       0.4789          1.0282            1.0270         8.11
ExhaustiveBinary-1024-random (self)                      256.61     2_315.08     2_571.69       0.3754          1.0447            1.0451         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  354.22       560.62       914.84       0.1832          1.2013            1.1905         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   354.22       713.57     1_067.79       0.3798          1.0434            1.0443         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   354.22       852.84     1_207.07       0.4867          1.0272            1.0261         8.11
ExhaustiveBinary-1024-pca (self)                         354.22     2_319.25     2_673.48       0.3787          1.0436            1.0444         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   85.39       667.00       752.39       0.1518          1.2701            1.2528         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    85.39       732.03       817.42       0.3399          1.0607            1.0535         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    85.39     1_127.43     1_212.82       0.4406          1.0369            1.0319         3.05
ExhaustiveBinary-512-sign (self)                          85.39     2_489.25     2_574.65       0.3409          1.0595            1.0531         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)            1_692.66        76.62     1_769.27       0.1157          1.3331            1.2963         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           1_692.66        78.31     1_770.97       0.1157          1.3331            1.2963         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           1_692.66        80.65     1_773.31       0.1157          1.3331            1.2963         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           1_692.66       157.25     1_849.90       0.3225          1.0778            1.0583         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           1_692.66       245.38     1_938.04       0.4188          1.0495            1.0350         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          1_692.66       155.90     1_848.56       0.3225          1.0778            1.0583         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          1_692.66       247.54     1_940.20       0.4188          1.0495            1.0350         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          1_692.66       153.08     1_845.74       0.3225          1.0778            1.0583         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          1_692.66       245.66     1_938.32       0.4188          1.0495            1.0350         2.34
IVF-Binary-256-nl158-random (self)                     1_692.66       289.12     1_981.78       0.3241          1.0737            1.0571         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           1_000.72        71.76     1_072.48       0.1325          1.2776            1.2358         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)           1_000.72        73.47     1_074.19       0.1325          1.2779            1.2360         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)           1_000.72        75.84     1_076.56       0.1325          1.2780            1.2360         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)          1_000.72       160.82     1_161.54       0.3688          1.0554            1.0455         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)          1_000.72       251.02     1_251.74       0.4683          1.0358            1.0278         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)          1_000.72       173.96     1_174.68       0.3685          1.0555            1.0456         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)          1_000.72       252.24     1_252.96       0.4678          1.0359            1.0279         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)          1_000.72       163.38     1_164.10       0.3685          1.0555            1.0456         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)          1_000.72       258.28     1_259.00       0.4678          1.0359            1.0279         2.47
IVF-Binary-256-nl223-random (self)                     1_000.72       314.61     1_315.33       0.3705          1.0515            1.0449         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)           1_303.29        78.62     1_381.91       0.1435          1.2534            1.2115         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           1_303.29        79.40     1_382.69       0.1434          1.2536            1.2119         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           1_303.29        82.26     1_385.55       0.1434          1.2542            1.2121         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          1_303.29       176.81     1_480.10       0.3836          1.0508            1.0420         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          1_303.29       257.29     1_560.58       0.4826          1.0337            1.0261         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          1_303.29       164.71     1_467.99       0.3833          1.0510            1.0420         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          1_303.29       259.94     1_563.22       0.4820          1.0338            1.0262         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          1_303.29       168.41     1_471.70       0.3832          1.0510            1.0420         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          1_303.29       260.77     1_564.06       0.4818          1.0339            1.0262         2.65
IVF-Binary-256-nl316-random (self)                     1_303.29       339.11     1_642.40       0.3852          1.0471            1.0414         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               1_771.27        67.08     1_838.35       0.1212          1.3300            1.2889         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              1_771.27        66.56     1_837.83       0.1212          1.3300            1.2889         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              1_771.27        68.63     1_839.90       0.1212          1.3300            1.2889         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              1_771.27       152.29     1_923.55       0.3222          1.0753            1.0574         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              1_771.27       241.54     2_012.81       0.4196          1.0483            1.0348         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             1_771.27       150.27     1_921.53       0.3222          1.0753            1.0574         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             1_771.27       247.58     2_018.85       0.4196          1.0483            1.0348         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             1_771.27       152.50     1_923.77       0.3222          1.0753            1.0574         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             1_771.27       245.76     2_017.03       0.4196          1.0483            1.0348         2.34
IVF-Binary-256-nl158-pca (self)                        1_771.27       299.15     2_070.42       0.3232          1.0740            1.0566         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              1_072.00        72.51     1_144.51       0.1366          1.2800            1.2327         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              1_072.00        74.58     1_146.58       0.1366          1.2802            1.2328         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              1_072.00        75.93     1_147.93       0.1366          1.2802            1.2328         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             1_072.00       161.07     1_233.08       0.3637          1.0553            1.0460         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             1_072.00       253.69     1_325.69       0.4640          1.0357            1.0283         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             1_072.00       161.92     1_233.92       0.3636          1.0553            1.0460         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             1_072.00       252.68     1_324.68       0.4638          1.0357            1.0283         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             1_072.00       162.21     1_234.22       0.3636          1.0553            1.0460         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             1_072.00       254.81     1_326.82       0.4638          1.0357            1.0283         2.47
IVF-Binary-256-nl223-pca (self)                        1_072.00       317.18     1_389.18       0.3658          1.0535            1.0453         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              1_416.88        79.86     1_496.74       0.1456          1.2588            1.2105         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              1_416.88        78.31     1_495.19       0.1456          1.2588            1.2106         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              1_416.88        81.95     1_498.84       0.1455          1.2592            1.2107         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             1_416.88       166.66     1_583.54       0.3774          1.0509            1.0426         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             1_416.88       256.56     1_673.44       0.4766          1.0332            1.0267         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             1_416.88       164.94     1_581.82       0.3773          1.0509            1.0427         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             1_416.88       257.20     1_674.08       0.4760          1.0333            1.0268         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             1_416.88       167.27     1_584.16       0.3772          1.0509            1.0427         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             1_416.88       261.36     1_678.25       0.4759          1.0333            1.0268         2.65
IVF-Binary-256-nl316-pca (self)                        1_416.88       337.93     1_754.82       0.3792          1.0492            1.0422         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)            1_729.81        93.52     1_823.33       0.1554          1.2515            1.2240         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)           1_729.81        95.55     1_825.35       0.1554          1.2515            1.2240         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)           1_729.81        98.00     1_827.80       0.1554          1.2515            1.2240         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)           1_729.81       183.28     1_913.09       0.3517          1.0543            1.0500         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)           1_729.81       276.66     2_006.47       0.4505          1.0347            1.0305         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)          1_729.81       184.02     1_913.82       0.3517          1.0543            1.0500         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)          1_729.81       280.85     2_010.65       0.4505          1.0347            1.0305         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)          1_729.81       186.56     1_916.36       0.3517          1.0543            1.0500         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)          1_729.81       282.80     2_012.61       0.4505          1.0347            1.0305         4.36
IVF-Binary-512-nl158-random (self)                     1_729.81       416.63     2_146.43       0.3527          1.0528            1.0499         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)           1_066.76        98.98     1_165.74       0.1644          1.2242            1.1976         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)           1_066.76       101.91     1_168.66       0.1643          1.2245            1.1979         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)           1_066.76       108.61     1_175.36       0.1643          1.2246            1.1979         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)          1_066.76       196.60     1_263.36       0.3693          1.0479            1.0450         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)          1_066.76       289.35     1_356.11       0.4691          1.0312            1.0277         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)          1_066.76       194.06     1_260.81       0.3691          1.0480            1.0450         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)          1_066.76       283.97     1_350.72       0.4685          1.0312            1.0278         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)          1_066.76       193.14     1_259.90       0.3691          1.0480            1.0450         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)          1_066.76       289.32     1_356.08       0.4685          1.0312            1.0278         4.49
IVF-Binary-512-nl223-random (self)                     1_066.76       438.07     1_504.82       0.3701          1.0469            1.0450         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)           1_480.57       106.91     1_587.48       0.1680          1.2145            1.1880         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)           1_480.57       109.66     1_590.23       0.1680          1.2148            1.1883         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)           1_480.57       110.83     1_591.40       0.1679          1.2151            1.1885         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)          1_480.57       197.90     1_678.47       0.3756          1.0469            1.0438         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)          1_480.57       287.95     1_768.52       0.4755          1.0304            1.0270         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)          1_480.57       198.14     1_678.70       0.3754          1.0470            1.0438         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)          1_480.57       289.39     1_769.95       0.4747          1.0305            1.0272         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)          1_480.57       201.01     1_681.58       0.3753          1.0470            1.0439         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)          1_480.57       293.35     1_773.92       0.4744          1.0306            1.0272         4.67
IVF-Binary-512-nl316-random (self)                     1_480.57       461.93     1_942.49       0.3761          1.0456            1.0437         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)               1_828.55        92.90     1_921.45       0.1578          1.2477            1.2198         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)              1_828.55        98.49     1_927.04       0.1578          1.2477            1.2198         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)              1_828.55        98.18     1_926.73       0.1578          1.2477            1.2198         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)              1_828.55       183.28     2_011.83       0.3546          1.0514            1.0497         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)              1_828.55       281.93     2_110.48       0.4526          1.0324            1.0303         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)             1_828.55       183.89     2_012.44       0.3546          1.0514            1.0497         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)             1_828.55       276.16     2_104.71       0.4526          1.0324            1.0303         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)             1_828.55       186.90     2_015.45       0.3546          1.0514            1.0497         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)             1_828.55       278.43     2_106.98       0.4526          1.0324            1.0303         4.36
IVF-Binary-512-nl158-pca (self)                        1_828.55       418.29     2_246.85       0.3548          1.0513            1.0495         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_146.62        99.01     1_245.63       0.1671          1.2190            1.1937         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_146.62       100.81     1_247.44       0.1671          1.2192            1.1940         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_146.62       106.23     1_252.85       0.1671          1.2192            1.1940         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_146.62       192.59     1_339.21       0.3708          1.0459            1.0453         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_146.62       282.71     1_429.33       0.4723          1.0291            1.0275         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_146.62       190.22     1_336.85       0.3705          1.0459            1.0453         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_146.62       283.13     1_429.76       0.4716          1.0291            1.0276         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_146.62       196.10     1_342.72       0.3705          1.0459            1.0453         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_146.62       290.13     1_436.75       0.4716          1.0291            1.0276         4.49
IVF-Binary-512-nl223-pca (self)                        1_146.62       436.39     1_583.01       0.3711          1.0457            1.0452         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)              1_508.41       107.28     1_615.69       0.1706          1.2099            1.1842         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)              1_508.41       108.19     1_616.59       0.1706          1.2102            1.1845         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)              1_508.41       110.79     1_619.20       0.1705          1.2103            1.1846         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)             1_508.41       197.59     1_705.99       0.3771          1.0444            1.0437         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)             1_508.41       289.53     1_797.94       0.4786          1.0283            1.0267         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)             1_508.41       197.23     1_705.64       0.3768          1.0445            1.0438         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)             1_508.41       295.09     1_803.49       0.4778          1.0284            1.0267         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)             1_508.41       201.50     1_709.91       0.3767          1.0445            1.0438         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)             1_508.41       297.52     1_805.93       0.4776          1.0285            1.0268         4.67
IVF-Binary-512-nl316-pca (self)                        1_508.41       459.41     1_967.82       0.3772          1.0444            1.0438         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)           1_782.43       146.23     1_928.66       0.1827          1.2009            1.1907         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)          1_782.43       146.95     1_929.38       0.1827          1.2009            1.1907         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)          1_782.43       148.38     1_930.81       0.1827          1.2009            1.1907         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)          1_782.43       238.15     2_020.58       0.3773          1.0441            1.0445         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)          1_782.43       333.88     2_116.31       0.4819          1.0279            1.0266         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)         1_782.43       241.48     2_023.91       0.3773          1.0441            1.0445         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)         1_782.43       340.47     2_122.90       0.4819          1.0279            1.0266         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)         1_782.43       246.03     2_028.45       0.3773          1.0441            1.0445         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)         1_782.43       348.65     2_131.08       0.4819          1.0279            1.0266         8.42
IVF-Binary-1024-nl158-random (self)                    1_782.43       613.39     2_395.81       0.3779          1.0441            1.0445         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_109.60       149.35     1_258.95       0.1855          1.1895            1.1800         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_109.60       152.08     1_261.67       0.1855          1.1897            1.1802         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_109.60       158.52     1_268.12       0.1855          1.1897            1.1802         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_109.60       247.34     1_356.94       0.3862          1.0418            1.0422         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_109.60       345.12     1_454.72       0.4929          1.0264            1.0252         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_109.60       247.13     1_356.73       0.3859          1.0419            1.0423         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_109.60       351.29     1_460.89       0.4924          1.0265            1.0253         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_109.60       256.07     1_365.66       0.3859          1.0419            1.0423         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_109.60       359.94     1_469.53       0.4924          1.0265            1.0253         8.54
IVF-Binary-1024-nl223-random (self)                    1_109.60       641.33     1_750.92       0.3871          1.0418            1.0422         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)          1_559.58       158.77     1_718.35       0.1870          1.1851            1.1757         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)          1_559.58       162.75     1_722.33       0.1869          1.1854            1.1760         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)          1_559.58       165.23     1_724.81       0.1869          1.1855            1.1761         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)         1_559.58       255.33     1_814.91       0.3891          1.0413            1.0415         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)         1_559.58       359.89     1_919.48       0.4957          1.0261            1.0248         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)         1_559.58       267.49     1_827.08       0.3887          1.0414            1.0416         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)         1_559.58       366.76     1_926.34       0.4951          1.0262            1.0248         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)         1_559.58       261.75     1_821.34       0.3886          1.0414            1.0416         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)         1_559.58       364.95     1_924.53       0.4950          1.0262            1.0249         8.73
IVF-Binary-1024-nl316-random (self)                    1_559.58       666.99     2_226.58       0.3899          1.0413            1.0416         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)              1_938.94       149.47     2_088.41       0.1841          1.1989            1.1881         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)             1_938.94       151.98     2_090.92       0.1841          1.1989            1.1881         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)             1_938.94       155.09     2_094.03       0.1841          1.1989            1.1881         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)             1_938.94       242.73     2_181.67       0.3820          1.0431            1.0438         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)             1_938.94       343.13     2_282.08       0.4889          1.0270            1.0258         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)            1_938.94       243.67     2_182.62       0.3820          1.0431            1.0438         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)            1_938.94       347.58     2_286.52       0.4889          1.0270            1.0258         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)            1_938.94       253.19     2_192.13       0.3820          1.0431            1.0438         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)            1_938.94       352.54     2_291.49       0.4889          1.0270            1.0258         8.42
IVF-Binary-1024-nl158-pca (self)                       1_938.94       626.95     2_565.89       0.3808          1.0432            1.0439         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_243.72       155.82     1_399.55       0.1869          1.1877            1.1786         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_243.72       154.56     1_398.28       0.1869          1.1879            1.1788         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_243.72       160.00     1_403.72       0.1869          1.1879            1.1788         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_243.72       250.30     1_494.03       0.3905          1.0407            1.0416         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_243.72       349.25     1_592.97       0.4989          1.0255            1.0247         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_243.72       248.42     1_492.15       0.3902          1.0408            1.0417         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_243.72       353.36     1_597.08       0.4984          1.0256            1.0247         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_243.72       257.31     1_501.03       0.3902          1.0408            1.0417         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_243.72       356.17     1_599.89       0.4984          1.0256            1.0247         8.54
IVF-Binary-1024-nl223-pca (self)                       1_243.72       646.20     1_889.93       0.3897          1.0409            1.0418         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_601.82       161.50     1_763.32       0.1883          1.1833            1.1740         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_601.82       161.79     1_763.61       0.1882          1.1836            1.1743         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_601.82       164.36     1_766.18       0.1882          1.1837            1.1744         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_601.82       267.27     1_869.09       0.3938          1.0401            1.0409         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_601.82       354.67     1_956.49       0.5030          1.0252            1.0243         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_601.82       255.61     1_857.43       0.3935          1.0402            1.0410         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_601.82       368.94     1_970.76       0.5023          1.0252            1.0243         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_601.82       263.07     1_864.89       0.3934          1.0402            1.0410         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_601.82       364.36     1_966.18       0.5022          1.0253            1.0243         8.73
IVF-Binary-1024-nl316-pca (self)                       1_601.82       656.71     2_258.53       0.3929          1.0403            1.0411         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)              1_696.87       289.49     1_986.36       0.1520          1.2698            1.2528         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)             1_696.87       294.10     1_990.97       0.1520          1.2698            1.2528         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)             1_696.87       295.64     1_992.51       0.1520          1.2698            1.2528         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)             1_696.87       359.94     2_056.81       0.3410          1.0600            1.0529         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)             1_696.87       646.00     2_342.86       0.4418          1.0366            1.0317         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)            1_696.87       358.77     2_055.64       0.3410          1.0600            1.0529         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)            1_696.87       647.72     2_344.59       0.4418          1.0366            1.0317         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)            1_696.87       365.77     2_062.64       0.3410          1.0600            1.0529         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)            1_696.87       650.36     2_347.23       0.4418          1.0366            1.0317         3.36
IVF-Binary-512-nl158-sign (self)                       1_696.87       985.52     2_682.39       0.3419          1.0590            1.0526         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               985.45       293.18     1_278.62       0.1530          1.2631            1.2472         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               985.45       292.49     1_277.94       0.1529          1.2639            1.2483         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               985.45       300.72     1_286.17       0.1529          1.2639            1.2483         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              985.45       369.28     1_354.73       0.3502          1.0559            1.0502         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              985.45       659.61     1_645.06       0.4489          1.0348            1.0306         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              985.45       369.20     1_354.64       0.3500          1.0560            1.0502         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              985.45       676.89     1_662.34       0.4482          1.0349            1.0307         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              985.45       369.53     1_354.97       0.3500          1.0560            1.0502         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              985.45       662.54     1_647.99       0.4482          1.0349            1.0307         3.49
IVF-Binary-512-nl223-sign (self)                         985.45     1_087.24     2_072.68       0.3506          1.0550            1.0500         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)             1_347.32       302.78     1_650.09       0.1530          1.2630            1.2450         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)             1_347.32       300.88     1_648.20       0.1529          1.2642            1.2464         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)             1_347.32       304.40     1_651.71       0.1528          1.2648            1.2472         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)            1_347.32       371.21     1_718.52       0.3525          1.0552            1.0494         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)            1_347.32       656.17     2_003.49       0.4493          1.0348            1.0305         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)            1_347.32       380.31     1_727.63       0.3521          1.0554            1.0496         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)            1_347.32       663.39     2_010.70       0.4482          1.0350            1.0306         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)            1_347.32       375.94     1_723.26       0.3519          1.0555            1.0496         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)            1_347.32       667.23     2_014.55       0.4478          1.0351            1.0307         3.67
IVF-Binary-512-nl316-sign (self)                       1_347.32     1_010.21     2_357.52       0.3528          1.0544            1.0494         3.67
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
Exhaustive (query)                                       100.27     1_845.30     1_945.57       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.27     6_198.75     6_299.03       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                191.37       282.19       473.55       0.1140          1.2809            1.2433         2.28
ExhaustiveBinary-256-random-rf10 (query)                 191.37       422.24       613.61       0.3148          1.0656            1.0476         2.28
ExhaustiveBinary-256-random-rf20 (query)                 191.37       569.27       760.64       0.4075          1.0420            1.0293         2.28
ExhaustiveBinary-256-random (self)                       191.37     1_317.88     1_509.25       0.3168          1.0618            1.0471         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   398.07       285.93       684.00       0.1054          1.3026            1.2617         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    398.07       418.25       816.33       0.3012          1.0735            1.0517         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    398.07       561.81       959.89       0.3931          1.0471            1.0315         2.28
ExhaustiveBinary-256-pca (self)                          398.07     1_311.49     1_709.57       0.3048          1.0710            1.0504         2.28
ExhaustiveBinary-512-random_no_rr (query)                298.16       416.13       714.29       0.1506          1.2094            1.1809         4.55
ExhaustiveBinary-512-random-rf10 (query)                 298.16       570.31       868.47       0.3395          1.0453            1.0426         4.55
ExhaustiveBinary-512-random-rf20 (query)                 298.16       726.17     1_024.33       0.4326          1.0293            1.0264         4.55
ExhaustiveBinary-512-random (self)                       298.16     1_796.94     2_095.10       0.3401          1.0435            1.0423         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   501.57       411.97       913.54       0.1459          1.2162            1.1914         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    501.57       568.48     1_070.05       0.3341          1.0468            1.0435         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    501.57       724.02     1_225.59       0.4278          1.0295            1.0269         4.55
ExhaustiveBinary-512-pca (self)                          501.57     1_793.05     2_294.62       0.3355          1.0454            1.0433         4.55
ExhaustiveBinary-1024-random_no_rr (query)               497.42       614.43     1_111.85       0.1761          1.1673            1.1571         9.11
ExhaustiveBinary-1024-random-rf10 (query)                497.42       791.58     1_289.00       0.3603          1.0383            1.0383         9.11
ExhaustiveBinary-1024-random-rf20 (query)                497.42       966.51     1_463.93       0.4618          1.0244            1.0230         9.11
ExhaustiveBinary-1024-random (self)                      497.42     2_564.22     3_061.64       0.3602          1.0377            1.0383         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  701.26       629.20     1_330.46       0.1756          1.1686            1.1586         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   701.26       791.74     1_492.99       0.3594          1.0382            1.0385         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   701.26       989.60     1_690.86       0.4576          1.0246            1.0237         9.11
ExhaustiveBinary-1024-pca (self)                         701.26     2_586.62     3_287.88       0.3584          1.0383            1.0389         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  129.15       851.16       980.31       0.1691          1.1871            1.1718         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   129.15       934.51     1_063.66       0.3431          1.0433            1.0415         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   129.15     1_443.94     1_573.10       0.4437          1.0266            1.0250         4.58
ExhaustiveBinary-768-sign (self)                         129.15     3_001.94     3_131.10       0.3438          1.0424            1.0413         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)            2_323.36        95.28     2_418.64       0.1164          1.2717            1.2403         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)           2_323.36        96.75     2_420.12       0.1164          1.2717            1.2403         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)           2_323.36       104.12     2_427.48       0.1164          1.2717            1.2403         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)           2_323.36       199.52     2_522.89       0.3175          1.0626            1.0468         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)           2_323.36       321.36     2_644.73       0.4096          1.0407            1.0290         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)          2_323.36       198.82     2_522.19       0.3175          1.0626            1.0468         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)          2_323.36       316.22     2_639.59       0.4096          1.0407            1.0290         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)          2_323.36       201.55     2_524.92       0.3175          1.0626            1.0468         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)          2_323.36       315.44     2_638.80       0.4096          1.0407            1.0290         2.74
IVF-Binary-256-nl158-random (self)                     2_323.36       397.41     2_720.78       0.3193          1.0593            1.0464         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)           1_541.76        97.06     1_638.83       0.1332          1.2302            1.1908         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)           1_541.76       100.88     1_642.64       0.1332          1.2302            1.1908         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)           1_541.76        99.92     1_641.68       0.1332          1.2302            1.1908         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)          1_541.76       204.95     1_746.72       0.3572          1.0474            1.0377         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)          1_541.76       323.04     1_864.80       0.4565          1.0309            1.0232         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)          1_541.76       202.81     1_744.57       0.3572          1.0474            1.0377         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)          1_541.76       324.78     1_866.55       0.4565          1.0309            1.0232         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)          1_541.76       206.75     1_748.51       0.3572          1.0474            1.0377         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)          1_541.76       328.82     1_870.59       0.4565          1.0309            1.0232         2.93
IVF-Binary-256-nl223-random (self)                     1_541.76       428.33     1_970.09       0.3589          1.0438            1.0373         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)           1_847.69       104.16     1_951.84       0.1402          1.2168            1.1748         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)           1_847.69       102.46     1_950.14       0.1402          1.2168            1.1748         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)           1_847.69       106.27     1_953.96       0.1402          1.2168            1.1748         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)          1_847.69       212.54     2_060.22       0.3656          1.0444            1.0361         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)          1_847.69       330.43     2_178.12       0.4629          1.0293            1.0225         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)          1_847.69       211.00     2_058.69       0.3656          1.0444            1.0361         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)          1_847.69       329.94     2_177.62       0.4628          1.0293            1.0225         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)          1_847.69       213.10     2_060.79       0.3656          1.0444            1.0361         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)          1_847.69       336.52     2_184.20       0.4628          1.0293            1.0225         3.21
IVF-Binary-256-nl316-random (self)                     1_847.69       462.70     2_310.38       0.3670          1.0410            1.0358         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)               2_510.53        84.26     2_594.79       0.1080          1.2909            1.2569         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)              2_510.53        87.52     2_598.05       0.1080          1.2909            1.2569         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)              2_510.53        88.82     2_599.34       0.1080          1.2909            1.2569         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)              2_510.53       189.30     2_699.83       0.3044          1.0707            1.0512         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)              2_510.53       308.27     2_818.80       0.3972          1.0447            1.0309         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)             2_510.53       189.28     2_699.81       0.3044          1.0707            1.0512         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)             2_510.53       305.95     2_816.48       0.3972          1.0447            1.0309         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)             2_510.53       189.82     2_700.35       0.3044          1.0707            1.0512         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)             2_510.53       325.83     2_836.36       0.3972          1.0447            1.0309         2.74
IVF-Binary-256-nl158-pca (self)                        2_510.53       375.79     2_886.32       0.3084          1.0672            1.0497         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)              1_648.02        91.95     1_739.97       0.1269          1.2393            1.2037         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)              1_648.02        93.64     1_741.66       0.1269          1.2393            1.2037         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)              1_648.02        97.04     1_745.06       0.1269          1.2393            1.2037         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)             1_648.02       203.62     1_851.64       0.3608          1.0475            1.0373         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)             1_648.02       321.36     1_969.38       0.4623          1.0301            1.0228         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)             1_648.02       203.58     1_851.60       0.3608          1.0475            1.0373         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)             1_648.02       336.59     1_984.61       0.4623          1.0301            1.0228         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)             1_648.02       204.01     1_852.03       0.3608          1.0475            1.0373         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)             1_648.02       320.47     1_968.49       0.4623          1.0301            1.0228         2.93
IVF-Binary-256-nl223-pca (self)                        1_648.02       408.40     2_056.41       0.3641          1.0449            1.0366         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)              2_019.86       102.14     2_122.00       0.1367          1.2212            1.1848         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)              2_019.86       101.36     2_121.22       0.1367          1.2212            1.1848         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)              2_019.86       104.75     2_124.61       0.1367          1.2212            1.1848         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)             2_019.86       211.73     2_231.59       0.3742          1.0434            1.0348         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)             2_019.86       333.85     2_353.71       0.4724          1.0282            1.0217         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)             2_019.86       208.85     2_228.71       0.3742          1.0434            1.0348         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)             2_019.86       331.55     2_351.41       0.4723          1.0283            1.0217         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)             2_019.86       212.63     2_232.49       0.3742          1.0434            1.0348         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)             2_019.86       336.41     2_356.27       0.4723          1.0283            1.0217         3.21
IVF-Binary-256-nl316-pca (self)                        2_019.86       442.18     2_462.04       0.3770          1.0409            1.0340         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)            2_412.13       123.83     2_535.96       0.1520          1.2060            1.1791         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)           2_412.13       126.46     2_538.59       0.1520          1.2060            1.1791         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)           2_412.13       128.79     2_540.92       0.1520          1.2060            1.1791         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)           2_412.13       233.52     2_645.65       0.3405          1.0447            1.0423         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)           2_412.13       352.24     2_764.37       0.4339          1.0290            1.0262         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)          2_412.13       233.40     2_645.53       0.3405          1.0447            1.0423         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)          2_412.13       374.78     2_786.91       0.4339          1.0290            1.0262         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)          2_412.13       241.13     2_653.26       0.3405          1.0447            1.0423         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)          2_412.13       369.28     2_781.40       0.4339          1.0290            1.0262         5.02
IVF-Binary-512-nl158-random (self)                     2_412.13       557.45     2_969.58       0.3409          1.0431            1.0421         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)           1_759.26       130.53     1_889.79       0.1603          1.1840            1.1576         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)           1_759.26       152.76     1_912.02       0.1603          1.1840            1.1576         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)           1_759.26       140.52     1_899.78       0.1603          1.1840            1.1576         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)          1_759.26       246.71     2_005.97       0.3578          1.0405            1.0380         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)          1_759.26       365.75     2_125.01       0.4564          1.0262            1.0235         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)          1_759.26       242.79     2_002.05       0.3578          1.0405            1.0380         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)          1_759.26       385.67     2_144.94       0.4564          1.0262            1.0235         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)          1_759.26       249.14     2_008.40       0.3578          1.0405            1.0380         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)          1_759.26       376.06     2_135.33       0.4564          1.0262            1.0235         5.21
IVF-Binary-512-nl223-random (self)                     1_759.26       583.20     2_342.46       0.3585          1.0389            1.0379         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)           2_079.39       141.62     2_221.01       0.1624          1.1788            1.1525         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)           2_079.39       140.14     2_219.52       0.1624          1.1789            1.1525         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)           2_079.39       143.55     2_222.94       0.1624          1.1789            1.1525         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)          2_079.39       254.38     2_333.77       0.3613          1.0391            1.0372         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)          2_079.39       382.56     2_461.95       0.4582          1.0253            1.0234         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)          2_079.39       250.95     2_330.34       0.3613          1.0392            1.0372         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)          2_079.39       377.83     2_457.22       0.4582          1.0254            1.0234         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)          2_079.39       257.19     2_336.58       0.3613          1.0392            1.0372         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)          2_079.39       385.07     2_464.46       0.4582          1.0254            1.0234         5.48
IVF-Binary-512-nl316-random (self)                     2_079.39       611.79     2_691.18       0.3616          1.0379            1.0373         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)               2_611.74       122.96     2_734.71       0.1476          1.2119            1.1892         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)              2_611.74       123.81     2_735.55       0.1476          1.2119            1.1892         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)              2_611.74       125.91     2_737.66       0.1476          1.2119            1.1892         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)              2_611.74       234.69     2_846.43       0.3359          1.0458            1.0431         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)              2_611.74       352.11     2_963.85       0.4299          1.0290            1.0267         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)             2_611.74       234.29     2_846.04       0.3359          1.0458            1.0431         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)             2_611.74       356.44     2_968.19       0.4299          1.0290            1.0267         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)             2_611.74       246.97     2_858.72       0.3359          1.0458            1.0431         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)             2_611.74       359.00     2_970.75       0.4299          1.0290            1.0267         5.02
IVF-Binary-512-nl158-pca (self)                        2_611.74       556.25     3_167.99       0.3374          1.0443            1.0428         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_745.79       128.79     1_874.58       0.1584          1.1837            1.1596         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_745.79       131.22     1_877.01       0.1584          1.1837            1.1596         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_745.79       135.56     1_881.35       0.1584          1.1837            1.1596         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_745.79       243.07     1_988.86       0.3583          1.0398            1.0379         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_745.79       362.79     2_108.58       0.4550          1.0259            1.0235         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_745.79       242.60     1_988.39       0.3583          1.0398            1.0379         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_745.79       363.15     2_108.94       0.4550          1.0259            1.0235         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_745.79       246.00     1_991.79       0.3583          1.0398            1.0379         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_745.79       371.02     2_116.81       0.4550          1.0259            1.0235         5.21
IVF-Binary-512-nl223-pca (self)                        1_745.79       577.40     2_323.19       0.3590          1.0390            1.0378         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)              2_145.95       138.81     2_284.76       0.1627          1.1763            1.1519         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)              2_145.95       140.88     2_286.83       0.1627          1.1763            1.1519         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)              2_145.95       142.92     2_288.87       0.1627          1.1763            1.1519         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)             2_145.95       251.63     2_397.58       0.3626          1.0388            1.0370         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)             2_145.95       372.27     2_518.22       0.4585          1.0255            1.0231         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)             2_145.95       254.97     2_400.92       0.3626          1.0389            1.0370         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)             2_145.95       381.73     2_527.68       0.4584          1.0255            1.0231         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)             2_145.95       256.48     2_402.42       0.3626          1.0389            1.0370         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)             2_145.95       378.69     2_524.64       0.4584          1.0255            1.0231         5.48
IVF-Binary-512-nl316-pca (self)                        2_145.95       613.51     2_759.46       0.3635          1.0382            1.0370         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)           2_653.78       191.85     2_845.63       0.1768          1.1661            1.1563         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)          2_653.78       206.26     2_860.04       0.1768          1.1661            1.1563         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)          2_653.78       198.40     2_852.18       0.1768          1.1661            1.1563         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)          2_653.78       316.84     2_970.61       0.3609          1.0381            1.0382         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)          2_653.78       452.80     3_106.58       0.4626          1.0243            1.0230         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)         2_653.78       323.35     2_977.12       0.3609          1.0381            1.0382         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)         2_653.78       454.71     3_108.49       0.4626          1.0243            1.0230         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)         2_653.78       328.77     2_982.55       0.3609          1.0381            1.0382         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)         2_653.78       462.07     3_115.85       0.4626          1.0243            1.0230         9.57
IVF-Binary-1024-nl158-random (self)                    2_653.78       841.39     3_495.17       0.3608          1.0375            1.0382         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_764.43       203.93     1_968.36       0.1797          1.1568            1.1474         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_764.43       203.38     1_967.81       0.1797          1.1568            1.1474         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_764.43       208.56     1_973.00       0.1797          1.1568            1.1474         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_764.43       335.89     2_100.32       0.3717          1.0360            1.0360         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_764.43       459.36     2_223.79       0.4749          1.0229            1.0217         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_764.43       336.36     2_100.79       0.3717          1.0360            1.0360         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_764.43       468.78     2_233.21       0.4749          1.0229            1.0217         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_764.43       344.41     2_108.84       0.3717          1.0360            1.0360         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_764.43       483.92     2_248.35       0.4749          1.0229            1.0217         9.76
IVF-Binary-1024-nl223-random (self)                    1_764.43       861.04     2_625.47       0.3714          1.0355            1.0360         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)          2_213.44       212.59     2_426.03       0.1804          1.1544            1.1451        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)          2_213.44       215.72     2_429.16       0.1804          1.1545            1.1451        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)          2_213.44       217.52     2_430.95       0.1804          1.1545            1.1451        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)         2_213.44       348.36     2_561.79       0.3732          1.0356            1.0359        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)         2_213.44       480.50     2_693.93       0.4755          1.0227            1.0217        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)         2_213.44       349.09     2_562.52       0.3732          1.0356            1.0359        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)         2_213.44       480.13     2_693.57       0.4755          1.0228            1.0217        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)         2_213.44       359.30     2_572.74       0.3732          1.0356            1.0359        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)         2_213.44       487.08     2_700.52       0.4755          1.0228            1.0217        10.04
IVF-Binary-1024-nl316-random (self)                    2_213.44       899.16     3_112.60       0.3730          1.0351            1.0357        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)              2_874.72       195.88     3_070.60       0.1762          1.1673            1.1573         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)             2_874.72       198.96     3_073.68       0.1762          1.1673            1.1573         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)             2_874.72       203.80     3_078.51       0.1762          1.1673            1.1573         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)             2_874.72       322.75     3_197.47       0.3600          1.0380            1.0384         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)             2_874.72       452.59     3_327.31       0.4585          1.0245            1.0237         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)            2_874.72       332.32     3_207.03       0.3600          1.0380            1.0384         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)            2_874.72       457.32     3_332.03       0.4585          1.0245            1.0237         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)            2_874.72       332.51     3_207.23       0.3600          1.0380            1.0384         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)            2_874.72       461.41     3_336.12       0.4585          1.0245            1.0237         9.57
IVF-Binary-1024-nl158-pca (self)                       2_874.72       841.55     3_716.26       0.3591          1.0381            1.0388         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_966.45       202.24     2_168.70       0.1794          1.1567            1.1470         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_966.45       203.56     2_170.01       0.1794          1.1567            1.1470         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_966.45       209.37     2_175.83       0.1794          1.1567            1.1470         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_966.45       331.99     2_298.44       0.3709          1.0358            1.0361         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_966.45       460.23     2_426.69       0.4723          1.0231            1.0221         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_966.45       335.76     2_302.21       0.3709          1.0358            1.0361         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_966.45       466.20     2_432.66       0.4723          1.0231            1.0221         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_966.45       361.74     2_328.19       0.3709          1.0358            1.0361         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_966.45       474.49     2_440.94       0.4723          1.0231            1.0221         9.76
IVF-Binary-1024-nl223-pca (self)                       1_966.45       867.31     2_833.77       0.3702          1.0359            1.0363         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             2_346.28       216.31     2_562.59       0.1805          1.1540            1.1447        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)             2_346.28       213.64     2_559.92       0.1805          1.1540            1.1447        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)             2_346.28       219.33     2_565.61       0.1805          1.1540            1.1447        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)            2_346.28       350.01     2_696.28       0.3730          1.0355            1.0357        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)            2_346.28       476.67     2_822.95       0.4735          1.0228            1.0220        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)            2_346.28       349.79     2_696.07       0.3730          1.0355            1.0357        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)            2_346.28       477.89     2_824.17       0.4735          1.0228            1.0220        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)            2_346.28       355.23     2_701.51       0.3730          1.0355            1.0357        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)            2_346.28       513.91     2_860.19       0.4735          1.0228            1.0220        10.04
IVF-Binary-1024-nl316-pca (self)                       2_346.28       904.30     3_250.58       0.3720          1.0356            1.0360        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)              2_267.67       406.07     2_673.74       0.1693          1.1869            1.1720         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)             2_267.67       411.16     2_678.83       0.1693          1.1869            1.1720         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)             2_267.67       416.38     2_684.05       0.1693          1.1869            1.1720         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)             2_267.67       505.79     2_773.46       0.3435          1.0431            1.0414         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)             2_267.67       907.80     3_175.47       0.4441          1.0265            1.0249         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)            2_267.67       501.31     2_768.98       0.3435          1.0431            1.0414         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)            2_267.67       926.13     3_193.80       0.4441          1.0265            1.0249         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)            2_267.67       506.94     2_774.60       0.3435          1.0431            1.0414         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)            2_267.67       921.36     3_189.03       0.4441          1.0265            1.0249         5.04
IVF-Binary-768-nl158-sign (self)                       2_267.67     1_426.48     3_694.15       0.3441          1.0422            1.0413         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)             1_389.28       412.93     1_802.22       0.1694          1.1869            1.1715         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)             1_389.28       416.09     1_805.37       0.1694          1.1869            1.1715         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)             1_389.28       420.44     1_809.72       0.1694          1.1869            1.1715         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)            1_389.28       519.22     1_908.50       0.3491          1.0415            1.0400         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)            1_389.28       926.47     2_315.75       0.4485          1.0259            1.0244         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)            1_389.28       508.79     1_898.07       0.3491          1.0415            1.0400         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)            1_389.28       919.85     2_309.13       0.4485          1.0259            1.0244         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)            1_389.28       513.47     1_902.75       0.3491          1.0415            1.0400         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)            1_389.28       927.51     2_316.79       0.4485          1.0259            1.0244         5.23
IVF-Binary-768-nl223-sign (self)                       1_389.28     1_444.29     2_833.57       0.3494          1.0408            1.0400         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)             1_838.43       421.22     2_259.65       0.1695          1.1865            1.1713         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)             1_838.43       425.21     2_263.64       0.1695          1.1866            1.1713         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)             1_838.43       428.09     2_266.53       0.1695          1.1866            1.1713         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)            1_838.43       517.50     2_355.93       0.3496          1.0411            1.0400         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)            1_838.43       927.03     2_765.46       0.4487          1.0257            1.0245         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)            1_838.43       516.64     2_355.07       0.3495          1.0411            1.0400         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)            1_838.43       929.54     2_767.97       0.4486          1.0257            1.0245         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)            1_838.43       521.59     2_360.02       0.3495          1.0411            1.0400         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)            1_838.43       936.70     2_775.13       0.4486          1.0257            1.0245         5.51
IVF-Binary-768-nl316-sign (self)                       1_838.43     1_466.64     3_305.07       0.3501          1.0405            1.0398         5.51
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
Exhaustive (query)                                        32.44       698.17       730.61       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.44     2_382.14     2_414.58       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                 69.11       239.09       308.20       0.0970          1.6334            1.6378         1.78
ExhaustiveBinary-256-random-rf10 (query)                  69.11       338.03       407.14       0.3643          1.1391            1.1302         1.78
ExhaustiveBinary-256-random-rf20 (query)                  69.11       440.34       509.45       0.5087          1.0798            1.0701         1.78
ExhaustiveBinary-256-random (self)                        69.11     1_093.07     1_162.17       0.3862          1.1443            1.1409         1.78
ExhaustiveBinary-256-pca_no_rr (query)                    92.38       237.42       329.79       0.0922          1.6524            1.6606         1.78
ExhaustiveBinary-256-pca-rf10 (query)                     92.38       336.87       429.25       0.3517          1.1465            1.1384         1.78
ExhaustiveBinary-256-pca-rf20 (query)                     92.38       436.70       529.07       0.4943          1.0846            1.0744         1.78
ExhaustiveBinary-256-pca (self)                           92.38     1_092.46     1_184.84       0.3764          1.1502            1.1477         1.78
ExhaustiveBinary-512-random_no_rr (query)                 82.65       354.06       436.71       0.1464          1.5035            1.5085         3.55
ExhaustiveBinary-512-random-rf10 (query)                  82.65       460.43       543.08       0.4596          1.0936            1.0901         3.55
ExhaustiveBinary-512-random-rf20 (query)                  82.65       568.09       650.74       0.6085          1.0504            1.0459         3.55
ExhaustiveBinary-512-random (self)                        82.65     1_501.42     1_584.07       0.4800          1.0996            1.0995         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   107.86       351.09       458.94       0.1458          1.5041            1.5097         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    107.86       461.40       569.26       0.4543          1.0952            1.0911         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    107.86       564.09       671.94       0.6037          1.0513            1.0464         3.55
ExhaustiveBinary-512-pca (self)                          107.86     1_607.88     1_715.74       0.4777          1.1003            1.0997         3.55
ExhaustiveBinary-1024-random_no_rr (query)               112.90       503.24       616.14       0.2155          1.3655            1.3721         7.10
ExhaustiveBinary-1024-random-rf10 (query)                112.90       617.94       730.84       0.5869          1.0540            1.0515         7.10
ExhaustiveBinary-1024-random-rf20 (query)                112.90       724.54       837.44       0.7380          1.0260            1.0224         7.10
ExhaustiveBinary-1024-random (self)                      112.90     2_054.52     2_167.43       0.6118          1.0576            1.0546         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  139.39       515.29       654.68       0.2122          1.3735            1.3798         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   139.39       621.41       760.79       0.5776          1.0560            1.0532         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   139.39       732.61       871.99       0.7291          1.0273            1.0232         7.10
ExhaustiveBinary-1024-pca (self)                         139.39     2_104.00     2_243.39       0.6017          1.0602            1.0571         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   40.94       439.65       480.59       0.1044          1.6421            1.6499         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    40.94       472.69       513.63       0.3737          1.1368            1.1275         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    40.94       720.49       761.43       0.5265          1.0745            1.0646         1.53
ExhaustiveBinary-256-sign (self)                          40.94     1_535.82     1_576.76       0.3940          1.1439            1.1394         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)              912.49        48.67       961.15       0.1016          1.6177            1.6285         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)             912.49        51.50       963.99       0.1016          1.6179            1.6286         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)             912.49        57.49       969.98       0.1016          1.6179            1.6286         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)             912.49       101.48     1_013.96       0.3750          1.1333            1.1268         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)             912.49       151.93     1_064.42       0.5191          1.0759            1.0681         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)            912.49       105.67     1_018.16       0.3742          1.1334            1.1269         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)            912.49       150.07     1_062.55       0.5182          1.0761            1.0681         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)            912.49       102.47     1_014.95       0.3742          1.1335            1.1269         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)            912.49       153.63     1_066.12       0.5181          1.0761            1.0681         1.93
IVF-Binary-256-nl158-random (self)                       912.49       230.29     1_142.77       0.3960          1.1379            1.1379         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             628.70        45.62       674.32       0.1120          1.5877            1.5867         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             628.70        46.18       674.87       0.1119          1.5883            1.5872         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             628.70        52.09       680.78       0.1119          1.5884            1.5874         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            628.70        98.81       727.50       0.3934          1.1239            1.1158         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            628.70       149.21       777.91       0.5355          1.0713            1.0628         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            628.70        99.47       728.17       0.3930          1.1240            1.1160         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            628.70       149.10       777.79       0.5350          1.0714            1.0628         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            628.70       105.04       733.73       0.3929          1.1240            1.1160         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            628.70       149.76       778.45       0.5349          1.0714            1.0628         2.00
IVF-Binary-256-nl223-random (self)                       628.70       222.08       850.78       0.4141          1.1287            1.1268         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             832.51        47.49       880.00       0.1175          1.5678            1.5685         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             832.51        47.69       880.20       0.1174          1.5683            1.5691         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             832.51        52.07       884.59       0.1174          1.5686            1.5693         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            832.51        99.77       932.29       0.4048          1.1173            1.1103         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            832.51       147.39       979.90       0.5480          1.0673            1.0599         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            832.51        97.93       930.45       0.4042          1.1175            1.1108         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            832.51       148.68       981.19       0.5471          1.0675            1.0601         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            832.51       102.50       935.01       0.4041          1.1175            1.1108         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            832.51       152.15       984.67       0.5470          1.0675            1.0601         2.09
IVF-Binary-256-nl316-random (self)                       832.51       233.69     1_066.21       0.4254          1.1215            1.1216         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)                 925.19        39.98       965.16       0.0972          1.6371            1.6509         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)                925.19        42.07       967.26       0.0972          1.6373            1.6510         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)                925.19        45.94       971.12       0.0972          1.6373            1.6510         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)                925.19        92.99     1_018.18       0.3614          1.1414            1.1351         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)                925.19       142.40     1_067.59       0.5035          1.0811            1.0727         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)               925.19        93.39     1_018.57       0.3609          1.1415            1.1351         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)               925.19       144.67     1_069.86       0.5028          1.0812            1.0727         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)               925.19        97.47     1_022.66       0.3609          1.1415            1.1351         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)               925.19       147.37     1_072.56       0.5028          1.0812            1.0727         1.93
IVF-Binary-256-nl158-pca (self)                          925.19       216.83     1_142.02       0.3857          1.1449            1.1445         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                638.80        44.53       683.33       0.1097          1.5958            1.5999         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                638.80        44.46       683.26       0.1096          1.5968            1.6008         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                638.80        49.65       688.45       0.1096          1.5969            1.6009         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               638.80        97.55       736.35       0.3870          1.1269            1.1204         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               638.80       144.38       783.18       0.5267          1.0735            1.0656         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               638.80        96.47       735.27       0.3865          1.1271            1.1205         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               638.80       145.43       784.24       0.5258          1.0737            1.0659         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               638.80        99.57       738.37       0.3864          1.1271            1.1205         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               638.80       148.50       787.30       0.5258          1.0737            1.0659         2.00
IVF-Binary-256-nl223-pca (self)                          638.80       219.77       858.58       0.4102          1.1301            1.1303         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                863.54        47.10       910.64       0.1159          1.5754            1.5785         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                863.54        47.50       911.04       0.1158          1.5761            1.5790         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                863.54        52.00       915.54       0.1157          1.5767            1.5793         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               863.54        98.32       961.86       0.3971          1.1214            1.1151         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               863.54       146.16     1_009.70       0.5370          1.0701            1.0626         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               863.54        97.49       961.03       0.3965          1.1217            1.1154         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               863.54       146.86     1_010.40       0.5362          1.0703            1.0628         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               863.54       101.51       965.05       0.3963          1.1218            1.1155         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               863.54       150.80     1_014.34       0.5360          1.0704            1.0629         2.09
IVF-Binary-256-nl316-pca (self)                          863.54       231.97     1_095.51       0.4200          1.1246            1.1253         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)              915.85        58.20       974.05       0.1496          1.4966            1.5039         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)             915.85        61.72       977.57       0.1496          1.4966            1.5039         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)             915.85        66.70       982.56       0.1496          1.4966            1.5039         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)             915.85       113.62     1_029.47       0.4642          1.0917            1.0890         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)             915.85       164.58     1_080.43       0.6126          1.0494            1.0452         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)            915.85       115.51     1_031.36       0.4641          1.0917            1.0890         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)            915.85       168.15     1_084.00       0.6125          1.0494            1.0452         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)            915.85       121.09     1_036.95       0.4641          1.0917            1.0890         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)            915.85       173.09     1_088.94       0.6125          1.0494            1.0452         3.71
IVF-Binary-512-nl158-random (self)                       915.85       294.54     1_210.39       0.4841          1.0980            1.0982         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             634.11        60.96       695.07       0.1563          1.4801            1.4855         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             634.11        62.57       696.69       0.1562          1.4803            1.4857         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             634.11        70.26       704.37       0.1562          1.4803            1.4857         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            634.11       115.95       750.06       0.4750          1.0879            1.0850         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            634.11       164.78       798.89       0.6213          1.0475            1.0433         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            634.11       116.96       751.08       0.4748          1.0879            1.0851         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            634.11       167.20       801.31       0.6210          1.0476            1.0433         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            634.11       123.67       757.78       0.4748          1.0880            1.0851         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            634.11       174.94       809.05       0.6210          1.0476            1.0433         3.77
IVF-Binary-512-nl223-random (self)                       634.11       298.55       932.67       0.4945          1.0941            1.0938         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             852.13        64.62       916.75       0.1594          1.4708            1.4746         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             852.13        66.05       918.18       0.1594          1.4711            1.4747         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             852.13        71.52       923.66       0.1594          1.4711            1.4747         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            852.13       119.31       971.44       0.4807          1.0857            1.0829         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            852.13       169.17     1_021.31       0.6271          1.0463            1.0423         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            852.13       118.37       970.51       0.4803          1.0858            1.0830         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            852.13       170.38     1_022.52       0.6266          1.0464            1.0425         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            852.13       124.05       976.19       0.4803          1.0859            1.0831         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            852.13       175.29     1_027.42       0.6265          1.0464            1.0425         3.86
IVF-Binary-512-nl316-random (self)                       852.13       305.96     1_158.09       0.4996          1.0921            1.0915         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)                 948.32        57.55     1_005.87       0.1487          1.4983            1.5052         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)                948.32        60.63     1_008.95       0.1487          1.4983            1.5052         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)                948.32        66.53     1_014.86       0.1487          1.4983            1.5052         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)                948.32       113.96     1_062.28       0.4587          1.0935            1.0902         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)                948.32       164.21     1_112.53       0.6076          1.0502            1.0457         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)               948.32       117.36     1_065.68       0.4586          1.0935            1.0902         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)               948.32       169.90     1_118.22       0.6074          1.0503            1.0457         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)               948.32       123.40     1_071.72       0.4586          1.0935            1.0902         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)               948.32       173.71     1_122.04       0.6074          1.0503            1.0457         3.71
IVF-Binary-512-nl158-pca (self)                          948.32       307.03     1_255.36       0.4816          1.0989            1.0987         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                670.71        60.77       731.47       0.1556          1.4810            1.4845         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                670.71        63.51       734.22       0.1555          1.4814            1.4852         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                670.71        70.97       741.68       0.1555          1.4814            1.4853         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               670.71       115.58       786.29       0.4716          1.0888            1.0853         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               670.71       165.63       836.33       0.6190          1.0478            1.0432         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               670.71       121.55       792.26       0.4712          1.0890            1.0855         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               670.71       167.16       837.86       0.6185          1.0479            1.0432         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               670.71       123.90       794.60       0.4712          1.0890            1.0855         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               670.71       174.19       844.90       0.6185          1.0479            1.0432         3.77
IVF-Binary-512-nl223-pca (self)                          670.71       298.54       969.24       0.4934          1.0944            1.0939         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                884.56        65.13       949.69       0.1591          1.4714            1.4751         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                884.56        65.77       950.33       0.1590          1.4717            1.4756         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                884.56        71.16       955.72       0.1590          1.4719            1.4757         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               884.56       120.01     1_004.57       0.4769          1.0868            1.0833         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               884.56       167.85     1_052.41       0.6238          1.0467            1.0422         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               884.56       118.33     1_002.89       0.4764          1.0870            1.0835         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               884.56       170.17     1_054.73       0.6231          1.0468            1.0424         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               884.56       123.95     1_008.51       0.4764          1.0870            1.0835         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               884.56       176.33     1_060.89       0.6230          1.0469            1.0424         3.86
IVF-Binary-512-nl316-pca (self)                          884.56       305.16     1_189.72       0.4985          1.0924            1.0920         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)             950.55        90.42     1_040.97       0.2170          1.3632            1.3702         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)            950.55        98.46     1_049.00       0.2170          1.3632            1.3702         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)            950.55       100.96     1_051.50       0.2170          1.3632            1.3702         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)            950.55       148.10     1_098.64       0.5886          1.0535            1.0510         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)            950.55       206.14     1_156.69       0.7396          1.0257            1.0221         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)           950.55       152.67     1_103.22       0.5886          1.0535            1.0510         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)           950.55       205.45     1_155.99       0.7395          1.0257            1.0221         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)           950.55       159.40     1_109.95       0.5886          1.0535            1.0510         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)           950.55       214.76     1_165.31       0.7395          1.0257            1.0221         7.26
IVF-Binary-1024-nl158-random (self)                      950.55       420.20     1_370.75       0.6136          1.0570            1.0542         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            664.31        93.02       757.33       0.2204          1.3573            1.3647         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            664.31        96.28       760.59       0.2204          1.3574            1.3649         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            664.31       104.93       769.24       0.2204          1.3574            1.3649         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           664.31       150.56       814.87       0.5938          1.0524            1.0499         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           664.31       202.27       866.58       0.7439          1.0252            1.0215         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           664.31       151.36       815.67       0.5936          1.0525            1.0499         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           664.31       206.49       870.80       0.7437          1.0252            1.0215         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           664.31       161.74       826.05       0.5936          1.0525            1.0499         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           664.31       219.16       883.47       0.7437          1.0252            1.0215         7.32
IVF-Binary-1024-nl223-random (self)                      664.31       423.84     1_088.15       0.6185          1.0559            1.0530         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            884.74        99.90       984.64       0.2220          1.3537            1.3607         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)            884.74        97.29       982.03       0.2219          1.3538            1.3609         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)            884.74       105.82       990.56       0.2219          1.3538            1.3609         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)           884.74       153.34     1_038.08       0.5965          1.0517            1.0493         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)           884.74       207.85     1_092.59       0.7465          1.0248            1.0210         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)           884.74       153.48     1_038.22       0.5961          1.0518            1.0493         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)           884.74       209.01     1_093.75       0.7462          1.0248            1.0211         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)           884.74       161.96     1_046.70       0.5961          1.0518            1.0493         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)           884.74       216.61     1_101.35       0.7461          1.0248            1.0211         7.42
IVF-Binary-1024-nl316-random (self)                      884.74       433.77     1_318.51       0.6212          1.0551            1.0523         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)                988.58        89.47     1_078.04       0.2134          1.3713            1.3785         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)               988.58        95.83     1_084.41       0.2134          1.3713            1.3785         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)               988.58       101.29     1_089.87       0.2134          1.3713            1.3785         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)               988.58       149.34     1_137.92       0.5796          1.0555            1.0526         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)               988.58       199.76     1_188.33       0.7307          1.0270            1.0231         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)              988.58       150.88     1_139.46       0.5796          1.0555            1.0526         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)              988.58       206.16     1_194.74       0.7306          1.0270            1.0231         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)              988.58       159.24     1_147.81       0.5796          1.0555            1.0526         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)              988.58       218.06     1_206.64       0.7306          1.0270            1.0231         7.26
IVF-Binary-1024-nl158-pca (self)                         988.58       425.35     1_413.93       0.6034          1.0597            1.0567         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               690.46        93.48       783.94       0.2175          1.3639            1.3706         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               690.46        95.33       785.79       0.2175          1.3640            1.3708         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               690.46       105.38       795.84       0.2175          1.3640            1.3708         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              690.46       151.00       841.46       0.5850          1.0542            1.0513         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              690.46       202.66       893.12       0.7356          1.0263            1.0224         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              690.46       151.93       842.39       0.5848          1.0543            1.0513         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              690.46       206.82       897.28       0.7353          1.0264            1.0224         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              690.46       161.67       852.13       0.5847          1.0543            1.0513         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              690.46       216.49       906.96       0.7352          1.0264            1.0224         7.32
IVF-Binary-1024-nl223-pca (self)                         690.46       433.27     1_123.73       0.6093          1.0581            1.0551         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               923.88        97.91     1_021.79       0.2188          1.3604            1.3660         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)               923.88        97.85     1_021.74       0.2187          1.3606            1.3663         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)               923.88       106.05     1_029.93       0.2187          1.3606            1.3663         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)              923.88       156.27     1_080.15       0.5881          1.0534            1.0504         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)              923.88       209.01     1_132.90       0.7382          1.0260            1.0220         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)              923.88       153.63     1_077.51       0.5877          1.0535            1.0505         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)              923.88       208.16     1_132.04       0.7378          1.0260            1.0221         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)              923.88       162.04     1_085.92       0.5877          1.0535            1.0505         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)              923.88       222.21     1_146.09       0.7377          1.0260            1.0221         7.42
IVF-Binary-1024-nl316-pca (self)                         923.88       437.28     1_361.16       0.6116          1.0575            1.0543         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)                892.61       148.90     1_041.51       0.1043          1.6446            1.6491         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               892.61       151.67     1_044.28       0.1043          1.6447            1.6493         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               892.61       155.07     1_047.68       0.1043          1.6447            1.6493         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               892.61       188.55     1_081.17       0.3792          1.1342            1.1256         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               892.61       331.52     1_224.13       0.5308          1.0735            1.0642         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              892.61       188.97     1_081.59       0.3782          1.1344            1.1257         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              892.61       334.89     1_227.51       0.5303          1.0736            1.0642         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              892.61       192.60     1_085.21       0.3782          1.1344            1.1257         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              892.61       347.83     1_240.45       0.5302          1.0736            1.0642         1.68
IVF-Binary-256-nl158-sign (self)                         892.61       517.17     1_409.78       0.3978          1.1422            1.1381         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               586.89       150.14       737.03       0.1047          1.6393            1.6468         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               586.89       152.28       739.16       0.1045          1.6406            1.6480         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               586.89       156.14       743.03       0.1045          1.6410            1.6482         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              586.89       188.34       775.22       0.3864          1.1295            1.1209         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              586.89       332.56       919.44       0.5362          1.0716            1.0627         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              586.89       188.76       775.65       0.3860          1.1297            1.1212         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              586.89       333.56       920.45       0.5354          1.0718            1.0628         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              586.89       194.04       780.92       0.3858          1.1298            1.1212         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              586.89       339.76       926.64       0.5352          1.0718            1.0628         1.75
IVF-Binary-256-nl223-sign (self)                         586.89       518.88     1_105.76       0.4052          1.1380            1.1337         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               812.90       152.29       965.19       0.1054          1.6361            1.6424         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               812.90       152.68       965.58       0.1053          1.6368            1.6429         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               812.90       156.86       969.76       0.1053          1.6375            1.6439         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              812.90       191.15     1_004.05       0.3916          1.1267            1.1189         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              812.90       334.92     1_147.82       0.5390          1.0704            1.0622         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              812.90       190.57     1_003.47       0.3911          1.1270            1.1192         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              812.90       336.24     1_149.14       0.5379          1.0707            1.0623         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              812.90       195.33     1_008.23       0.3909          1.1270            1.1192         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              812.90       341.55     1_154.46       0.5377          1.0708            1.0624         1.84
IVF-Binary-256-nl316-sign (self)                         812.90       526.64     1_339.54       0.4104          1.1342            1.1317         1.84
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
Exhaustive (query)                                        69.26     1_297.00     1_366.26       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.26     4_319.76     4_389.02       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                134.33       265.41       399.74       0.0733          1.4884            1.4968         2.03
ExhaustiveBinary-256-random-rf10 (query)                 134.33       389.04       523.37       0.2947          1.1327            1.1260         2.03
ExhaustiveBinary-256-random-rf20 (query)                 134.33       519.95       654.28       0.4174          1.0830            1.0716         2.03
ExhaustiveBinary-256-random (self)                       134.33     1_289.15     1_423.48       0.3150          1.1321            1.1268         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   217.39       280.48       497.87       0.0722          1.4941            1.5001         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    217.39       393.93       611.32       0.2934          1.1324            1.1267         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    217.39       529.79       747.18       0.4181          1.0813            1.0715         2.03
ExhaustiveBinary-256-pca (self)                          217.39     1_207.55     1_424.94       0.3112          1.1338            1.1298         2.03
ExhaustiveBinary-512-random_no_rr (query)                206.49       390.26       596.75       0.1110          1.4064            1.4153         4.05
ExhaustiveBinary-512-random-rf10 (query)                 206.49       530.93       737.42       0.3695          1.0935            1.0919         4.05
ExhaustiveBinary-512-random-rf20 (query)                 206.49       664.87       871.36       0.4981          1.0544            1.0517         4.05
ExhaustiveBinary-512-random (self)                       206.49     1_691.16     1_897.65       0.3856          1.0953            1.0998         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   293.45       392.21       685.67       0.1063          1.4156            1.4272         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    293.45       537.76       831.22       0.3574          1.0990            1.0958         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    293.45       658.58       952.03       0.4853          1.0582            1.0544         4.05
ExhaustiveBinary-512-pca (self)                          293.45     1_681.29     1_974.75       0.3753          1.0995            1.1040         4.05
ExhaustiveBinary-1024-random_no_rr (query)               259.52       555.21       814.73       0.1593          1.3242            1.3318         8.11
ExhaustiveBinary-1024-random-rf10 (query)                259.52       716.24       975.76       0.4456          1.0660            1.0678         8.11
ExhaustiveBinary-1024-random-rf20 (query)                259.52       872.49     1_132.01       0.5824          1.0370            1.0362         8.11
ExhaustiveBinary-1024-random (self)                      259.52     2_352.81     2_612.33       0.4595          1.0714            1.0740         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  351.74       566.29       918.04       0.1599          1.3236            1.3333         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   351.74       721.38     1_073.13       0.4446          1.0658            1.0680         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   351.74       863.54     1_215.29       0.5812          1.0370            1.0362         8.11
ExhaustiveBinary-1024-pca (self)                         351.74     2_364.84     2_716.59       0.4582          1.0716            1.0746         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   98.34       678.84       777.18       0.1292          1.3815            1.3877         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    98.34       735.70       834.03       0.3927          1.0844            1.0833         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    98.34     1_122.41     1_220.75       0.5336          1.0464            1.0444         3.05
ExhaustiveBinary-512-sign (self)                          98.34     2_384.55     2_482.88       0.4063          1.0885            1.0916         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)            1_725.90        77.34     1_803.24       0.0762          1.4798            1.4941         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           1_725.90        85.93     1_811.83       0.0762          1.4800            1.4942         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           1_725.90        80.28     1_806.18       0.0762          1.4800            1.4942         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           1_725.90       160.23     1_886.14       0.2982          1.1315            1.1256         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           1_725.90       247.42     1_973.32       0.4198          1.0821            1.0715         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          1_725.90       160.18     1_886.08       0.2974          1.1316            1.1256         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          1_725.90       244.82     1_970.72       0.4191          1.0822            1.0715         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          1_725.90       152.58     1_878.48       0.2974          1.1316            1.1256         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          1_725.90       247.36     1_973.26       0.4191          1.0822            1.0715         2.34
IVF-Binary-256-nl158-random (self)                     1_725.90       312.92     2_038.82       0.3173          1.1313            1.1266         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           1_095.40        76.18     1_171.58       0.0883          1.4512            1.4575         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)           1_095.40        74.67     1_170.07       0.0883          1.4514            1.4576         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)           1_095.40        81.06     1_176.47       0.0883          1.4514            1.4576         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)          1_095.40       161.03     1_256.44       0.3261          1.1150            1.1088         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)          1_095.40       254.62     1_350.02       0.4506          1.0704            1.0628         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)          1_095.40       164.95     1_260.35       0.3260          1.1151            1.1088         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)          1_095.40       254.84     1_350.24       0.4505          1.0704            1.0629         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)          1_095.40       164.32     1_259.72       0.3260          1.1151            1.1088         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)          1_095.40       256.49     1_351.89       0.4505          1.0704            1.0629         2.47
IVF-Binary-256-nl223-random (self)                     1_095.40       344.04     1_439.44       0.3442          1.1147            1.1139         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)           1_525.37        81.85     1_607.22       0.0951          1.4360            1.4383         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           1_525.37        81.70     1_607.07       0.0951          1.4361            1.4384         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           1_525.37        82.31     1_607.69       0.0951          1.4361            1.4384         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          1_525.37       167.97     1_693.34       0.3374          1.1080            1.1028         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          1_525.37       263.35     1_788.72       0.4655          1.0650            1.0593         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          1_525.37       170.00     1_695.37       0.3373          1.1080            1.1028         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          1_525.37       264.02     1_789.40       0.4654          1.0650            1.0593         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          1_525.37       169.45     1_694.82       0.3373          1.1080            1.1028         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          1_525.37       264.02     1_789.39       0.4654          1.0650            1.0593         2.65
IVF-Binary-256-nl316-random (self)                     1_525.37       342.19     1_867.56       0.3557          1.1069            1.1093         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               1_746.80        66.52     1_813.31       0.0751          1.4851            1.4978         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              1_746.80        67.86     1_814.66       0.0750          1.4853            1.4978         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              1_746.80        69.18     1_815.98       0.0750          1.4853            1.4979         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              1_746.80       153.84     1_900.64       0.2973          1.1312            1.1262         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              1_746.80       247.80     1_994.60       0.4206          1.0808            1.0713         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             1_746.80       152.94     1_899.74       0.2966          1.1313            1.1262         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             1_746.80       252.10     1_998.90       0.4199          1.0809            1.0713         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             1_746.80       156.36     1_903.16       0.2965          1.1313            1.1262         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             1_746.80       249.61     1_996.41       0.4199          1.0809            1.0713         2.34
IVF-Binary-256-nl158-pca (self)                        1_746.80       291.93     2_038.73       0.3141          1.1326            1.1295         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              1_282.84        73.15     1_356.00       0.0880          1.4534            1.4573         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              1_282.84        77.41     1_360.25       0.0879          1.4535            1.4574         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              1_282.84        77.35     1_360.19       0.0879          1.4535            1.4574         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             1_282.84       167.11     1_449.96       0.3296          1.1118            1.1063         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             1_282.84       261.97     1_544.81       0.4517          1.0691            1.0623         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             1_282.84       167.96     1_450.80       0.3295          1.1118            1.1063         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             1_282.84       258.46     1_541.30       0.4517          1.0691            1.0623         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             1_282.84       166.69     1_449.53       0.3295          1.1118            1.1063         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             1_282.84       260.77     1_543.61       0.4516          1.0691            1.0623         2.47
IVF-Binary-256-nl223-pca (self)                        1_282.84       320.59     1_603.44       0.3479          1.1103            1.1126         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              1_648.14        79.00     1_727.13       0.0950          1.4352            1.4379         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              1_648.14        80.23     1_728.36       0.0950          1.4353            1.4380         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              1_648.14        82.21     1_730.35       0.0950          1.4353            1.4380         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             1_648.14       168.20     1_816.34       0.3407          1.1054            1.0999         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             1_648.14       262.20     1_910.34       0.4613          1.0660            1.0596         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             1_648.14       166.82     1_814.95       0.3406          1.1054            1.0999         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             1_648.14       261.57     1_909.71       0.4612          1.0660            1.0596         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             1_648.14       168.62     1_816.75       0.3406          1.1054            1.0999         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             1_648.14       266.11     1_914.25       0.4612          1.0660            1.0596         2.65
IVF-Binary-256-nl316-pca (self)                        1_648.14       343.42     1_991.56       0.3588          1.1032            1.1073         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)            1_729.86        91.25     1_821.12       0.1123          1.4042            1.4145         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)           1_729.86        97.07     1_826.93       0.1123          1.4043            1.4145         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)           1_729.86        97.92     1_827.78       0.1123          1.4043            1.4145         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)           1_729.86       194.60     1_924.47       0.3703          1.0933            1.0918         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)           1_729.86       283.10     2_012.96       0.4988          1.0542            1.0516         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)          1_729.86       187.76     1_917.62       0.3701          1.0933            1.0918         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)          1_729.86       294.33     2_024.19       0.4986          1.0543            1.0516         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)          1_729.86       186.92     1_916.78       0.3701          1.0933            1.0918         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)          1_729.86       286.66     2_016.52       0.4986          1.0543            1.0516         4.36
IVF-Binary-512-nl158-random (self)                     1_729.86       419.17     2_149.03       0.3862          1.0951            1.0998         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)           1_157.91       102.43     1_260.34       0.1204          1.3874            1.3946         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)           1_157.91       105.77     1_263.68       0.1204          1.3875            1.3946         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)           1_157.91       107.36     1_265.28       0.1204          1.3875            1.3946         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)          1_157.91       193.00     1_350.92       0.3836          1.0877            1.0871         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)          1_157.91       287.65     1_445.57       0.5122          1.0510            1.0488         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)          1_157.91       193.02     1_350.94       0.3836          1.0877            1.0871         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)          1_157.91       296.33     1_454.24       0.5121          1.0510            1.0488         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)          1_157.91       195.33     1_353.24       0.3836          1.0877            1.0871         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)          1_157.91       295.63     1_453.54       0.5121          1.0510            1.0488         4.49
IVF-Binary-512-nl223-random (self)                     1_157.91       447.96     1_605.87       0.3990          1.0902            1.0949         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)           1_656.96       107.56     1_764.52       0.1243          1.3791            1.3837         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)           1_656.96       106.22     1_763.18       0.1243          1.3791            1.3837         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)           1_656.96       116.27     1_773.23       0.1243          1.3791            1.3837         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)          1_656.96       209.26     1_866.22       0.3890          1.0855            1.0847         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)          1_656.96       300.00     1_956.96       0.5158          1.0501            1.0480         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)          1_656.96       202.07     1_859.02       0.3890          1.0855            1.0847         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)          1_656.96       298.87     1_955.83       0.5158          1.0501            1.0480         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)          1_656.96       210.72     1_867.67       0.3890          1.0855            1.0847         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)          1_656.96       308.01     1_964.96       0.5158          1.0501            1.0480         4.67
IVF-Binary-512-nl316-random (self)                     1_656.96       459.50     2_116.46       0.4033          1.0884            1.0931         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)               1_835.32        91.40     1_926.72       0.1076          1.4133            1.4261         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)              1_835.32        95.31     1_930.63       0.1076          1.4134            1.4261         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)              1_835.32        97.09     1_932.41       0.1076          1.4134            1.4261         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)              1_835.32       184.03     2_019.35       0.3583          1.0988            1.0957         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)              1_835.32       278.49     2_113.81       0.4858          1.0581            1.0544         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)             1_835.32       196.23     2_031.55       0.3581          1.0988            1.0957         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)             1_835.32       285.58     2_120.90       0.4857          1.0581            1.0544         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)             1_835.32       189.80     2_025.12       0.3581          1.0988            1.0957         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)             1_835.32       282.97     2_118.29       0.4857          1.0581            1.0544         4.36
IVF-Binary-512-nl158-pca (self)                        1_835.32       418.34     2_253.66       0.3759          1.0993            1.1040         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_264.22       105.43     1_369.64       0.1168          1.3940            1.3999         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_264.22       115.89     1_380.10       0.1168          1.3940            1.4000         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_264.22       105.69     1_369.91       0.1168          1.3940            1.4000         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_264.22       195.79     1_460.01       0.3738          1.0918            1.0901         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_264.22       290.68     1_554.89       0.5017          1.0541            1.0515         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_264.22       192.97     1_457.18       0.3738          1.0918            1.0901         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_264.22       295.62     1_559.84       0.5017          1.0541            1.0515         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_264.22       196.56     1_460.77       0.3738          1.0918            1.0901         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_264.22       292.45     1_556.67       0.5017          1.0541            1.0515         4.49
IVF-Binary-512-nl223-pca (self)                        1_264.22       435.12     1_699.33       0.3906          1.0936            1.0983         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)              1_683.75       105.52     1_789.27       0.1210          1.3851            1.3902         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)              1_683.75       108.75     1_792.51       0.1210          1.3851            1.3902         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)              1_683.75       111.01     1_794.76       0.1210          1.3851            1.3902         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)             1_683.75       199.73     1_883.48       0.3801          1.0890            1.0877         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)             1_683.75       293.38     1_977.14       0.5059          1.0528            1.0503         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)             1_683.75       273.87     1_957.62       0.3800          1.0890            1.0877         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)             1_683.75       293.63     1_977.38       0.5059          1.0528            1.0503         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)             1_683.75       199.36     1_883.12       0.3800          1.0890            1.0877         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)             1_683.75       295.59     1_979.34       0.5059          1.0528            1.0503         4.67
IVF-Binary-512-nl316-pca (self)                        1_683.75       458.64     2_142.39       0.3956          1.0912            1.0960         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)           1_798.82       143.05     1_941.87       0.1598          1.3235            1.3316         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)          1_798.82       157.41     1_956.23       0.1598          1.3235            1.3316         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)          1_798.82       156.17     1_954.99       0.1598          1.3235            1.3316         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)          1_798.82       248.48     2_047.30       0.4458          1.0660            1.0678         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)          1_798.82       343.58     2_142.40       0.5826          1.0370            1.0362         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)         1_798.82       260.08     2_058.90       0.4458          1.0660            1.0678         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)         1_798.82       347.78     2_146.60       0.5826          1.0370            1.0362         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)         1_798.82       250.27     2_049.09       0.4458          1.0660            1.0678         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)         1_798.82       358.62     2_157.44       0.5826          1.0370            1.0362         8.42
IVF-Binary-1024-nl158-random (self)                    1_798.82       617.89     2_416.71       0.4597          1.0713            1.0740         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_231.08       153.48     1_384.56       0.1637          1.3164            1.3251         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_231.08       158.65     1_389.73       0.1637          1.3164            1.3251         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_231.08       162.60     1_393.68       0.1637          1.3164            1.3251         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_231.08       257.23     1_488.31       0.4532          1.0639            1.0658         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_231.08       352.29     1_583.37       0.5895          1.0358            1.0351         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_231.08       259.08     1_490.16       0.4532          1.0639            1.0658         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_231.08       356.81     1_587.89       0.5895          1.0358            1.0351         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_231.08       258.78     1_489.86       0.4532          1.0639            1.0658         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_231.08       365.04     1_596.12       0.5895          1.0358            1.0351         8.54
IVF-Binary-1024-nl223-random (self)                    1_231.08       639.46     1_870.54       0.4672          1.0692            1.0721         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)          1_660.23       168.26     1_828.49       0.1655          1.3134            1.3213         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)          1_660.23       158.78     1_819.01       0.1655          1.3134            1.3213         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)          1_660.23       163.35     1_823.58       0.1655          1.3134            1.3213         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)         1_660.23       272.93     1_933.16       0.4550          1.0633            1.0652         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)         1_660.23       362.02     2_022.25       0.5909          1.0355            1.0347         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)         1_660.23       251.87     1_912.10       0.4550          1.0633            1.0652         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)         1_660.23       362.95     2_023.18       0.5908          1.0355            1.0347         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)         1_660.23       270.02     1_930.25       0.4550          1.0633            1.0652         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)         1_660.23       373.65     2_033.88       0.5908          1.0355            1.0347         8.73
IVF-Binary-1024-nl316-random (self)                    1_660.23       663.77     2_324.00       0.4696          1.0685            1.0715         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)              2_021.37       146.40     2_167.77       0.1603          1.3231            1.3332         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)             2_021.37       146.66     2_168.03       0.1603          1.3231            1.3332         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)             2_021.37       152.93     2_174.31       0.1603          1.3231            1.3332         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)             2_021.37       235.65     2_257.03       0.4447          1.0658            1.0680         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)             2_021.37       355.72     2_377.09       0.5813          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)            2_021.37       241.51     2_262.88       0.4447          1.0658            1.0680         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)            2_021.37       346.86     2_368.23       0.5813          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)            2_021.37       245.48     2_266.85       0.4447          1.0658            1.0680         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)            2_021.37       354.16     2_375.54       0.5813          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-pca (self)                       2_021.37       621.84     2_643.21       0.4583          1.0716            1.0746         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_573.10       154.40     1_727.49       0.1643          1.3158            1.3263         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_573.10       153.63     1_726.72       0.1643          1.3159            1.3263         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_573.10       161.95     1_735.04       0.1643          1.3159            1.3263         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_573.10       251.66     1_824.75       0.4518          1.0637            1.0663         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_573.10       365.30     1_938.40       0.5877          1.0359            1.0354         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_573.10       256.52     1_829.62       0.4518          1.0637            1.0663         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_573.10       358.56     1_931.65       0.5877          1.0359            1.0354         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_573.10       259.09     1_832.18       0.4518          1.0637            1.0663         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_573.10       368.16     1_941.26       0.5877          1.0359            1.0354         8.54
IVF-Binary-1024-nl223-pca (self)                       1_573.10       641.20     2_214.29       0.4656          1.0695            1.0726         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_739.70       159.95     1_899.65       0.1657          1.3130            1.3236         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_739.70       158.95     1_898.65       0.1657          1.3131            1.3236         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_739.70       166.13     1_905.83       0.1657          1.3131            1.3236         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_739.70       254.64     1_994.34       0.4540          1.0631            1.0656         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_739.70       366.30     2_106.00       0.5897          1.0356            1.0351         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_739.70       253.05     1_992.75       0.4540          1.0631            1.0656         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_739.70       362.21     2_101.90       0.5897          1.0356            1.0351         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_739.70       274.89     2_014.59       0.4539          1.0631            1.0656         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_739.70       381.93     2_121.62       0.5897          1.0356            1.0351         8.73
IVF-Binary-1024-nl316-pca (self)                       1_739.70       667.68     2_407.38       0.4678          1.0688            1.0718         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)              1_614.87       293.15     1_908.03       0.1291          1.3814            1.3871         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)             1_614.87       297.62     1_912.49       0.1291          1.3814            1.3871         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)             1_614.87       296.70     1_911.57       0.1291          1.3814            1.3871         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)             1_614.87       372.55     1_987.42       0.3933          1.0843            1.0833         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)             1_614.87       659.60     2_274.47       0.5337          1.0464            1.0445         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)            1_614.87       361.79     1_976.66       0.3931          1.0843            1.0833         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)            1_614.87       654.98     2_269.85       0.5337          1.0464            1.0445         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)            1_614.87       365.68     1_980.55       0.3931          1.0843            1.0833         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)            1_614.87       671.78     2_286.65       0.5337          1.0464            1.0445         3.36
IVF-Binary-512-nl158-sign (self)                       1_614.87     1_001.73     2_616.60       0.4066          1.0885            1.0915         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)             1_065.63       305.51     1_371.13       0.1295          1.3811            1.3869         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)             1_065.63       298.91     1_364.53       0.1295          1.3812            1.3869         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)             1_065.63       299.58     1_365.21       0.1295          1.3812            1.3869         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)            1_065.63       369.44     1_435.06       0.3989          1.0819            1.0819         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)            1_065.63       674.36     1_739.99       0.5380          1.0455            1.0438         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)            1_065.63       379.38     1_445.01       0.3988          1.0819            1.0819         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)            1_065.63       662.08     1_727.70       0.5379          1.0455            1.0438         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)            1_065.63       376.20     1_441.83       0.3988          1.0819            1.0819         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)            1_065.63       679.48     1_745.10       0.5379          1.0455            1.0438         3.49
IVF-Binary-512-nl223-sign (self)                       1_065.63     1_009.16     2_074.79       0.4123          1.0865            1.0899         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)             1_512.30       300.05     1_812.36       0.1294          1.3809            1.3865         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)             1_512.30       305.85     1_818.15       0.1294          1.3809            1.3865         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)             1_512.30       308.82     1_821.13       0.1294          1.3809            1.3865         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)            1_512.30       372.22     1_884.52       0.4003          1.0814            1.0814         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)            1_512.30       673.32     2_185.63       0.5383          1.0453            1.0439         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)            1_512.30       373.56     1_885.86       0.4003          1.0814            1.0814         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)            1_512.30       676.05     2_188.36       0.5382          1.0453            1.0439         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)            1_512.30       381.90     1_894.20       0.4003          1.0814            1.0814         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)            1_512.30       674.64     2_186.95       0.5382          1.0453            1.0439         3.67
IVF-Binary-512-nl316-sign (self)                       1_512.30     1_042.13     2_554.43       0.4134          1.0860            1.0893         3.67
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
Exhaustive (query)                                       100.30     1_855.77     1_956.07       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.30     6_183.59     6_283.89       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                193.23       280.11       473.34       0.0662          1.3769            1.3786         2.28
ExhaustiveBinary-256-random-rf10 (query)                 193.23       419.99       613.22       0.2741          1.1112            1.1017         2.28
ExhaustiveBinary-256-random-rf20 (query)                 193.23       566.11       759.34       0.3877          1.0706            1.0590         2.28
ExhaustiveBinary-256-random (self)                       193.23     1_297.14     1_490.38       0.2874          1.1077            1.0994         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   389.76       285.41       675.18       0.0653          1.3793            1.3770         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    389.76       418.39       808.15       0.2701          1.1138            1.1018         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    389.76       569.04       958.80       0.3852          1.0726            1.0585         2.28
ExhaustiveBinary-256-pca (self)                          389.76     1_299.47     1_689.23       0.2820          1.1100            1.0990         2.28
ExhaustiveBinary-512-random_no_rr (query)                297.55       441.74       739.29       0.0934          1.3249            1.3316         4.55
ExhaustiveBinary-512-random-rf10 (query)                 297.55       569.72       867.27       0.3220          1.0860            1.0806         4.55
ExhaustiveBinary-512-random-rf20 (query)                 297.55       726.00     1_023.56       0.4345          1.0527            1.0485         4.55
ExhaustiveBinary-512-random (self)                       297.55     1_801.34     2_098.90       0.3344          1.0826            1.0842         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   489.90       418.16       908.06       0.0951          1.3226            1.3263         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    489.90       569.95     1_059.85       0.3246          1.0847            1.0788         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    489.90       727.19     1_217.09       0.4400          1.0517            1.0468         4.55
ExhaustiveBinary-512-pca (self)                          489.90     1_802.14     2_292.04       0.3361          1.0819            1.0825         4.55
ExhaustiveBinary-1024-random_no_rr (query)               502.08       610.95     1_113.03       0.1319          1.2685            1.2723         9.11
ExhaustiveBinary-1024-random-rf10 (query)                502.08       789.23     1_291.31       0.3742          1.0644            1.0666         9.11
ExhaustiveBinary-1024-random-rf20 (query)                502.08       972.49     1_474.57       0.4906          1.0386            1.0390         9.11
ExhaustiveBinary-1024-random (self)                      502.08     2_552.85     3_054.93       0.3823          1.0666            1.0715         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  687.10       621.80     1_308.89       0.1355          1.2623            1.2651         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   687.10       792.49     1_479.58       0.3804          1.0622            1.0643         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   687.10       973.93     1_661.02       0.4993          1.0369            1.0374         9.11
ExhaustiveBinary-1024-pca (self)                         687.10     2_569.94     3_257.03       0.3870          1.0651            1.0695         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  128.43       851.80       980.23       0.1284          1.2822            1.2821         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   128.43       949.85     1_078.28       0.3618          1.0706            1.0699         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   128.43     1_469.82     1_598.25       0.4847          1.0407            1.0395         4.58
ExhaustiveBinary-768-sign (self)                         128.43     3_019.13     3_147.56       0.3694          1.0716            1.0747         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)            2_343.74        89.10     2_432.84       0.0691          1.3703            1.3767         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)           2_343.74        98.00     2_441.75       0.0690          1.3706            1.3768         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)           2_343.74        99.51     2_443.26       0.0690          1.3706            1.3768         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)           2_343.74       196.71     2_540.46       0.2788          1.1100            1.1013         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)           2_343.74       306.98     2_650.72       0.3911          1.0698            1.0589         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)          2_343.74       190.24     2_533.99       0.2780          1.1101            1.1013         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)          2_343.74       308.06     2_651.81       0.3904          1.0699            1.0589         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)          2_343.74       192.00     2_535.75       0.2780          1.1101            1.1013         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)          2_343.74       309.10     2_652.85       0.3903          1.0699            1.0589         2.74
IVF-Binary-256-nl158-random (self)                     2_343.74       386.80     2_730.54       0.2914          1.1066            1.0992         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)           1_460.49        92.15     1_552.63       0.0782          1.3506            1.3552         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)           1_460.49        94.16     1_554.65       0.0782          1.3507            1.3552         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)           1_460.49        97.88     1_558.37       0.0782          1.3507            1.3552         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)          1_460.49       204.40     1_664.89       0.3036          1.0944            1.0863         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)          1_460.49       317.10     1_777.59       0.4176          1.0588            1.0517         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)          1_460.49       202.25     1_662.73       0.3036          1.0944            1.0863         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)          1_460.49       320.06     1_780.55       0.4175          1.0588            1.0517         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)          1_460.49       205.88     1_666.37       0.3036          1.0944            1.0863         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)          1_460.49       322.66     1_783.15       0.4175          1.0588            1.0517         2.93
IVF-Binary-256-nl223-random (self)                     1_460.49       419.23     1_879.71       0.3175          1.0893            1.0880         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)           2_101.49       104.75     2_206.24       0.0857          1.3373            1.3374         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)           2_101.49       103.94     2_205.43       0.0856          1.3375            1.3374         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)           2_101.49       107.92     2_209.40       0.0856          1.3375            1.3374         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)          2_101.49       214.94     2_316.43       0.3192          1.0859            1.0802         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)          2_101.49       330.34     2_431.83       0.4327          1.0540            1.0488         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)          2_101.49       214.05     2_315.54       0.3191          1.0859            1.0802         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)          2_101.49       328.29     2_429.77       0.4326          1.0540            1.0488         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)          2_101.49       216.94     2_318.42       0.3191          1.0859            1.0802         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)          2_101.49       330.88     2_432.37       0.4326          1.0540            1.0488         3.21
IVF-Binary-256-nl316-random (self)                     2_101.49       460.20     2_561.69       0.3327          1.0803            1.0828         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)               2_536.24        84.56     2_620.79       0.0681          1.3721            1.3752         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)              2_536.24        90.56     2_626.80       0.0681          1.3723            1.3752         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)              2_536.24        88.96     2_625.19       0.0681          1.3723            1.3752         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)              2_536.24       194.94     2_731.18       0.2754          1.1116            1.1012         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)              2_536.24       304.20     2_840.43       0.3894          1.0711            1.0583         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)             2_536.24       192.63     2_728.87       0.2745          1.1117            1.1012         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)             2_536.24       307.88     2_844.11       0.3886          1.0712            1.0583         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)             2_536.24       194.56     2_730.79       0.2745          1.1117            1.1012         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)             2_536.24       308.35     2_844.58       0.3885          1.0712            1.0583         2.74
IVF-Binary-256-nl158-pca (self)                        2_536.24       384.21     2_920.45       0.2864          1.1079            1.0986         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)              1_633.61        94.25     1_727.86       0.0775          1.3528            1.3548         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)              1_633.61        96.60     1_730.21       0.0775          1.3529            1.3548         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)              1_633.61        97.33     1_730.94       0.0775          1.3529            1.3548         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)             1_633.61       204.88     1_838.49       0.2987          1.0962            1.0873         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)             1_633.61       324.89     1_958.51       0.4131          1.0605            1.0523         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)             1_633.61       203.79     1_837.40       0.2986          1.0962            1.0873         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)             1_633.61       320.19     1_953.80       0.4130          1.0605            1.0523         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)             1_633.61       205.65     1_839.26       0.2986          1.0962            1.0873         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)             1_633.61       347.92     1_981.53       0.4129          1.0605            1.0523         2.93
IVF-Binary-256-nl223-pca (self)                        1_633.61       430.50     2_064.11       0.3109          1.0909            1.0885         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)              2_297.71       107.03     2_404.74       0.0850          1.3392            1.3373         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)              2_297.71       102.56     2_400.27       0.0850          1.3393            1.3374         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)              2_297.71       106.04     2_403.75       0.0850          1.3393            1.3374         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)             2_297.71       220.29     2_518.00       0.3123          1.0891            1.0816         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)             2_297.71       330.50     2_628.21       0.4271          1.0565            1.0494         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)             2_297.71       212.22     2_509.93       0.3122          1.0892            1.0816         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)             2_297.71       332.91     2_630.61       0.4270          1.0565            1.0494         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)             2_297.71       231.66     2_529.37       0.3122          1.0892            1.0816         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)             2_297.71       333.76     2_631.47       0.4270          1.0565            1.0494         3.21
IVF-Binary-256-nl316-pca (self)                        2_297.71       460.24     2_757.94       0.3237          1.0835            1.0840         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)            2_414.76       122.78     2_537.54       0.0948          1.3226            1.3308         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)           2_414.76       121.62     2_536.38       0.0948          1.3227            1.3308         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)           2_414.76       124.36     2_539.12       0.0948          1.3227            1.3308         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)           2_414.76       235.79     2_650.54       0.3238          1.0856            1.0806         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)           2_414.76       350.17     2_764.93       0.4355          1.0525            1.0484         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)          2_414.76       233.71     2_648.47       0.3235          1.0857            1.0806         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)          2_414.76       352.92     2_767.68       0.4355          1.0525            1.0484         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)          2_414.76       236.18     2_650.94       0.3235          1.0857            1.0806         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)          2_414.76       359.77     2_774.53       0.4355          1.0525            1.0484         5.02
IVF-Binary-512-nl158-random (self)                     2_414.76       553.28     2_968.04       0.3360          1.0822            1.0842         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)           1_578.33       127.89     1_706.22       0.1037          1.3076            1.3082         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)           1_578.33       129.63     1_707.96       0.1037          1.3076            1.3082         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)           1_578.33       133.06     1_711.39       0.1037          1.3076            1.3082         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)          1_578.33       251.15     1_829.48       0.3364          1.0790            1.0764         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)          1_578.33       369.65     1_947.98       0.4477          1.0488            1.0461         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)          1_578.33       256.39     1_834.72       0.3364          1.0790            1.0764         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)          1_578.33       371.27     1_949.60       0.4477          1.0488            1.0461         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)          1_578.33       253.43     1_831.76       0.3364          1.0790            1.0764         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)          1_578.33       372.12     1_950.45       0.4477          1.0488            1.0461         5.21
IVF-Binary-512-nl223-random (self)                     1_578.33       586.32     2_164.65       0.3478          1.0765            1.0802         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)           2_202.37       138.89     2_341.26       0.1080          1.2996            1.2983         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)           2_202.37       139.42     2_341.79       0.1080          1.2996            1.2983         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)           2_202.37       143.04     2_345.40       0.1080          1.2996            1.2983         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)          2_202.37       258.45     2_460.82       0.3430          1.0762            1.0744         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)          2_202.37       374.74     2_577.11       0.4551          1.0469            1.0448         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)          2_202.37       252.73     2_455.10       0.3430          1.0762            1.0744         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)          2_202.37       374.83     2_577.20       0.4550          1.0469            1.0448         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)          2_202.37       257.52     2_459.89       0.3430          1.0762            1.0744         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)          2_202.37       380.37     2_582.74       0.4550          1.0469            1.0448         5.48
IVF-Binary-512-nl316-random (self)                     2_202.37       618.51     2_820.88       0.3535          1.0741            1.0784         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)               2_616.26       120.84     2_737.10       0.0963          1.3201            1.3252         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)              2_616.26       121.99     2_738.25       0.0962          1.3202            1.3252         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)              2_616.26       128.04     2_744.30       0.0962          1.3202            1.3252         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)              2_616.26       233.59     2_849.86       0.3267          1.0842            1.0787         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)              2_616.26       347.69     2_963.95       0.4414          1.0514            1.0467         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)             2_616.26       234.25     2_850.52       0.3264          1.0842            1.0787         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)             2_616.26       351.48     2_967.74       0.4413          1.0514            1.0467         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)             2_616.26       237.06     2_853.32       0.3264          1.0842            1.0787         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)             2_616.26       357.28     2_973.54       0.4413          1.0514            1.0467         5.02
IVF-Binary-512-nl158-pca (self)                        2_616.26       554.38     3_170.65       0.3378          1.0812            1.0824         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_729.04       136.24     1_865.28       0.1044          1.3060            1.3056         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_729.04       130.21     1_859.25       0.1044          1.3060            1.3056         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_729.04       133.51     1_862.55       0.1044          1.3060            1.3056         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_729.04       247.70     1_976.74       0.3374          1.0788            1.0750         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_729.04       367.61     2_096.65       0.4529          1.0478            1.0448         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_729.04       246.09     1_975.13       0.3374          1.0789            1.0750         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_729.04       365.04     2_094.08       0.4529          1.0478            1.0448         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_729.04       249.33     1_978.37       0.3374          1.0789            1.0750         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_729.04       369.29     2_098.33       0.4529          1.0478            1.0448         5.21
IVF-Binary-512-nl223-pca (self)                        1_729.04       584.79     2_313.83       0.3475          1.0770            1.0795         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)              2_421.76       140.70     2_562.46       0.1079          1.2993            1.2969         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)              2_421.76       139.00     2_560.76       0.1079          1.2993            1.2969         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)              2_421.76       142.84     2_564.60       0.1079          1.2993            1.2969         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)             2_421.76       256.41     2_678.17       0.3441          1.0758            1.0730         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)             2_421.76       376.85     2_798.62       0.4591          1.0464            1.0438         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)             2_421.76       257.24     2_679.00       0.3441          1.0758            1.0730         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)             2_421.76       380.99     2_802.76       0.4591          1.0464            1.0438         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)             2_421.76       263.80     2_685.56       0.3441          1.0758            1.0730         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)             2_421.76       383.90     2_805.66       0.4591          1.0464            1.0438         5.48
IVF-Binary-512-nl316-pca (self)                        2_421.76       617.86     3_039.62       0.3536          1.0746            1.0779         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)           2_643.09       192.47     2_835.56       0.1327          1.2680            1.2723         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)          2_643.09       194.02     2_837.11       0.1327          1.2680            1.2723         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)          2_643.09       199.87     2_842.96       0.1327          1.2680            1.2723         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)          2_643.09       322.53     2_965.62       0.3746          1.0643            1.0666         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)          2_643.09       459.40     3_102.49       0.4907          1.0386            1.0390         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)         2_643.09       327.05     2_970.14       0.3745          1.0643            1.0666         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)         2_643.09       463.45     3_106.54       0.4907          1.0386            1.0390         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)         2_643.09       334.81     2_977.90       0.3745          1.0643            1.0666         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)         2_643.09       468.95     3_112.04       0.4907          1.0386            1.0390         9.57
IVF-Binary-1024-nl158-random (self)                    2_643.09       849.56     3_492.65       0.3826          1.0665            1.0715         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_779.24       201.68     1_980.92       0.1368          1.2612            1.2660         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_779.24       204.61     1_983.85       0.1368          1.2612            1.2660         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_779.24       207.31     1_986.56       0.1368          1.2612            1.2660         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_779.24       333.10     2_112.34       0.3802          1.0625            1.0652         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_779.24       463.58     2_242.83       0.4970          1.0375            1.0378         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_779.24       335.69     2_114.93       0.3802          1.0625            1.0652         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_779.24       471.16     2_250.40       0.4970          1.0375            1.0378         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_779.24       343.49     2_122.74       0.3802          1.0625            1.0652         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_779.24       476.43     2_255.67       0.4970          1.0375            1.0378         9.76
IVF-Binary-1024-nl223-random (self)                    1_779.24       868.39     2_647.63       0.3881          1.0649            1.0698         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)          2_406.52       215.34     2_621.86       0.1389          1.2580            1.2624        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)          2_406.52       221.46     2_627.99       0.1389          1.2580            1.2624        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)          2_406.52       215.89     2_622.41       0.1389          1.2580            1.2624        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)         2_406.52       349.78     2_756.31       0.3846          1.0612            1.0639        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)         2_406.52       476.32     2_882.85       0.5016          1.0366            1.0373        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)         2_406.52       351.61     2_758.13       0.3846          1.0612            1.0639        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)         2_406.52       491.58     2_898.11       0.5016          1.0366            1.0373        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)         2_406.52       364.24     2_770.76       0.3846          1.0612            1.0639        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)         2_406.52       494.63     2_901.15       0.5016          1.0366            1.0373        10.04
IVF-Binary-1024-nl316-random (self)                    2_406.52       904.59     3_311.11       0.3917          1.0638            1.0688        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)              2_832.46       193.67     3_026.13       0.1360          1.2617            1.2650         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)             2_832.46       196.34     3_028.80       0.1360          1.2617            1.2650         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)             2_832.46       200.07     3_032.53       0.1360          1.2617            1.2650         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)             2_832.46       324.97     3_157.43       0.3810          1.0621            1.0643         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)             2_832.46       452.26     3_284.72       0.4996          1.0369            1.0374         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)            2_832.46       327.37     3_159.83       0.3809          1.0621            1.0643         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)            2_832.46       461.48     3_293.93       0.4996          1.0369            1.0374         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)            2_832.46       341.26     3_173.71       0.3809          1.0621            1.0643         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)            2_832.46       467.38     3_299.84       0.4996          1.0369            1.0374         9.57
IVF-Binary-1024-nl158-pca (self)                       2_832.46       846.49     3_678.95       0.3875          1.0650            1.0695         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_934.87       201.53     2_136.40       0.1395          1.2561            1.2604         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_934.87       202.70     2_137.57       0.1395          1.2561            1.2604         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_934.87       207.40     2_142.26       0.1395          1.2561            1.2604         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_934.87       337.97     2_272.84       0.3861          1.0602            1.0631         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_934.87       463.37     2_398.24       0.5049          1.0360            1.0365         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_934.87       337.41     2_272.27       0.3861          1.0602            1.0631         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_934.87       472.42     2_407.29       0.5049          1.0360            1.0365         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_934.87       345.98     2_280.85       0.3861          1.0602            1.0631         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_934.87       478.22     2_413.08       0.5049          1.0360            1.0365         9.76
IVF-Binary-1024-nl223-pca (self)                       1_934.87       876.36     2_811.22       0.3927          1.0635            1.0680         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             2_603.20       215.56     2_818.76       0.1415          1.2527            1.2573        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)             2_603.20       212.45     2_815.65       0.1415          1.2527            1.2573        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)             2_603.20       218.18     2_821.38       0.1415          1.2527            1.2573        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)            2_603.20       353.02     2_956.22       0.3894          1.0595            1.0624        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)            2_603.20       481.73     3_084.93       0.5085          1.0354            1.0362        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)            2_603.20       350.31     2_953.51       0.3894          1.0595            1.0624        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)            2_603.20       485.65     3_088.85       0.5085          1.0354            1.0362        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)            2_603.20       362.15     2_965.35       0.3894          1.0595            1.0624        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)            2_603.20       498.61     3_101.81       0.5085          1.0354            1.0362        10.04
IVF-Binary-1024-nl316-pca (self)                       2_603.20       918.09     3_521.29       0.3958          1.0626            1.0672        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)              2_269.41       405.71     2_675.12       0.1284          1.2834            1.2814         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)             2_269.41       409.96     2_679.37       0.1284          1.2834            1.2814         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)             2_269.41       413.69     2_683.10       0.1284          1.2834            1.2814         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)             2_269.41       500.83     2_770.24       0.3627          1.0704            1.0698         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)             2_269.41       908.35     3_177.76       0.4854          1.0406            1.0396         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)            2_269.41       501.58     2_770.99       0.3627          1.0704            1.0698         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)            2_269.41       914.37     3_183.78       0.4854          1.0406            1.0396         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)            2_269.41       524.29     2_793.70       0.3627          1.0704            1.0698         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)            2_269.41       923.81     3_193.22       0.4854          1.0406            1.0396         5.04
IVF-Binary-768-nl158-sign (self)                       2_269.41     1_424.35     3_693.76       0.3701          1.0715            1.0746         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)             1_395.54       412.04     1_807.58       0.1282          1.2804            1.2811         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)             1_395.54       414.45     1_809.99       0.1282          1.2804            1.2811         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)             1_395.54       419.90     1_815.44       0.1282          1.2804            1.2811         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)            1_395.54       507.73     1_903.27       0.3659          1.0690            1.0690         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)            1_395.54       926.94     2_322.48       0.4872          1.0400            1.0394         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)            1_395.54       508.27     1_903.81       0.3659          1.0690            1.0690         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)            1_395.54       918.77     2_314.31       0.4872          1.0400            1.0394         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)            1_395.54       512.81     1_908.35       0.3659          1.0690            1.0690         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)            1_395.54       924.13     2_319.67       0.4872          1.0400            1.0394         5.23
IVF-Binary-768-nl223-sign (self)                       1_395.54     1_456.18     2_851.72       0.3730          1.0703            1.0736         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)             2_037.26       421.12     2_458.38       0.1287          1.2802            1.2810         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)             2_037.26       427.64     2_464.90       0.1287          1.2802            1.2810         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)             2_037.26       426.10     2_463.36       0.1287          1.2802            1.2810         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)            2_037.26       516.58     2_553.84       0.3674          1.0684            1.0685         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)            2_037.26       946.82     2_984.08       0.4890          1.0397            1.0392         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)            2_037.26       526.56     2_563.82       0.3674          1.0684            1.0685         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)            2_037.26       929.79     2_967.05       0.4890          1.0397            1.0392         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)            2_037.26       533.79     2_571.05       0.3674          1.0684            1.0685         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)            2_037.26       942.13     2_979.38       0.4890          1.0397            1.0392         5.51
IVF-Binary-768-nl316-sign (self)                       2_037.26     1_470.30     3_507.56       0.3747          1.0698            1.0734         5.51
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
Exhaustive (query)                                        32.62       710.30       742.92       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.62     2_461.05     2_493.68       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                 69.83       242.39       312.22       0.5519          1.8826            1.5884         1.78
ExhaustiveBinary-256-random-rf10 (query)                  69.83       356.78       426.61       0.9881          1.0022            1.0000         1.78
ExhaustiveBinary-256-random-rf20 (query)                  69.83       459.14       528.97       0.9980          1.0003            1.0000         1.78
ExhaustiveBinary-256-random (self)                        69.83     1_232.24     1_302.07       0.9881          1.0022            1.0000         1.78
ExhaustiveBinary-256-pca_no_rr (query)                    95.85       240.71       336.55       0.5930          1.6081            1.4152         1.78
ExhaustiveBinary-256-pca-rf10 (query)                     95.85       353.50       449.35       0.9919          1.0013            1.0000         1.78
ExhaustiveBinary-256-pca-rf20 (query)                     95.85       477.31       573.16       0.9988          1.0001            1.0000         1.78
ExhaustiveBinary-256-pca (self)                           95.85     1_148.70     1_244.55       0.9915          1.0014            1.0000         1.78
ExhaustiveBinary-512-random_no_rr (query)                 82.92       356.34       439.26       0.6306          1.5767            1.3633         3.55
ExhaustiveBinary-512-random-rf10 (query)                  82.92       473.83       556.75       0.9975          1.0004            1.0000         3.55
ExhaustiveBinary-512-random-rf20 (query)                  82.92       584.22       667.14       0.9998          1.0000            1.0000         3.55
ExhaustiveBinary-512-random (self)                        82.92     1_561.33     1_644.25       0.9973          1.0004            1.0000         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   107.68       363.20       470.88       0.6479          1.4884            1.3147         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    107.68       474.70       582.38       0.9983          1.0002            1.0000         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    107.68       582.47       690.16       0.9998          1.0000            1.0000         3.55
ExhaustiveBinary-512-pca (self)                          107.68     1_565.47     1_673.16       0.9981          1.0002            1.0000         3.55
ExhaustiveBinary-1024-random_no_rr (query)               114.97       526.42       641.40       0.6758          1.4452            1.2804         7.10
ExhaustiveBinary-1024-random-rf10 (query)                114.97       641.82       756.79       0.9995          1.0001            1.0000         7.10
ExhaustiveBinary-1024-random-rf20 (query)                114.97       755.25       870.23       0.9999          1.0000            1.0000         7.10
ExhaustiveBinary-1024-random (self)                      114.97     2_146.93     2_261.90       0.9993          1.0001            1.0000         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  138.62       525.29       663.91       0.6838          1.4142            1.2651         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   138.62       649.67       788.29       0.9996          1.0000            1.0000         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   138.62       755.71       894.33       1.0000          1.0000            1.0000         7.10
ExhaustiveBinary-1024-pca (self)                         138.62     2_138.57     2_277.19       0.9995          1.0001            1.0000         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   40.85       422.86       463.71       0.0376         19.4734           14.8778         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    40.85       456.49       497.33       0.1617          2.7567            2.6548         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    40.85       690.13       730.97       0.2739          1.9837            1.9249         1.53
ExhaustiveBinary-256-sign (self)                          40.85     1_474.76     1_515.60       0.1691          2.7353            2.6299         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)              965.47        57.28     1_022.75       0.5655          1.6704            1.5137         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)             965.47        65.82     1_031.29       0.5588          1.7297            1.5496         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)             965.47        75.14     1_040.61       0.5568          1.7636            1.5627         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)             965.47       116.86     1_082.33       0.9903          1.0018            1.0000         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)             965.47       167.94     1_133.40       0.9968          1.0006            1.0000         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)            965.47       126.21     1_091.68       0.9907          1.0016            1.0000         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)            965.47       180.32     1_145.79       0.9986          1.0002            1.0000         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)            965.47       129.82     1_095.29       0.9898          1.0018            1.0000         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)            965.47       187.19     1_152.66       0.9984          1.0002            1.0000         1.93
IVF-Binary-256-nl158-random (self)                       965.47       323.69     1_289.16       0.9904          1.0017            1.0000         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             536.57        49.24       585.81       0.5629          1.6755            1.5256         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             536.57        53.78       590.35       0.5605          1.7012            1.5424         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             536.57        60.99       597.56       0.5578          1.7449            1.5594         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            536.57       109.69       646.26       0.9912          1.0015            1.0000         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            536.57       162.42       698.99       0.9984          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            536.57       112.47       649.04       0.9909          1.0015            1.0000         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            536.57       166.54       703.11       0.9987          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            536.57       119.96       656.53       0.9900          1.0017            1.0000         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            536.57       175.78       712.35       0.9985          1.0002            1.0000         2.00
IVF-Binary-256-nl223-random (self)                       536.57       286.38       822.95       0.9908          1.0016            1.0000         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             717.39        51.52       768.91       0.5622          1.6812            1.5290         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             717.39        52.68       770.07       0.5610          1.6936            1.5367         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             717.39        59.24       776.63       0.5584          1.7353            1.5549         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            717.39       109.53       826.92       0.9917          1.0014            1.0000         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            717.39       162.41       879.80       0.9987          1.0002            1.0000         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            717.39       112.40       829.79       0.9914          1.0014            1.0000         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            717.39       165.48       882.87       0.9988          1.0002            1.0000         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            717.39       116.79       834.18       0.9903          1.0017            1.0000         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            717.39       172.96       890.35       0.9986          1.0002            1.0000         2.09
IVF-Binary-256-nl316-random (self)                       717.39       272.95       990.34       0.9912          1.0015            1.0000         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)                 981.23        46.68     1_027.91       0.6038          1.4891            1.3755         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)                981.23        55.68     1_036.91       0.5989          1.5218            1.3913         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)                981.23        64.51     1_045.74       0.5975          1.5420            1.3965         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)                981.23       109.96     1_091.19       0.9926          1.0013            1.0000         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)                981.23       170.12     1_151.35       0.9972          1.0006            1.0000         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)               981.23       118.17     1_099.40       0.9934          1.0010            1.0000         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)               981.23       175.33     1_156.56       0.9991          1.0001            1.0000         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)               981.23       127.61     1_108.84       0.9927          1.0012            1.0000         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)               981.23       184.80     1_166.03       0.9990          1.0001            1.0000         1.93
IVF-Binary-256-nl158-pca (self)                          981.23       315.98     1_297.21       0.9929          1.0011            1.0000         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                556.52        48.90       605.42       0.6017          1.4948            1.3802         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                556.52        52.30       608.82       0.5998          1.5091            1.3875         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                556.52        60.04       616.57       0.5979          1.5320            1.3962         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               556.52       108.55       665.08       0.9937          1.0010            1.0000         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               556.52       161.99       718.52       0.9987          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               556.52       116.22       672.74       0.9935          1.0010            1.0000         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               556.52       165.92       722.44       0.9991          1.0001            1.0000         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               556.52       118.95       675.48       0.9929          1.0011            1.0000         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               556.52       175.62       732.14       0.9990          1.0001            1.0000         2.00
IVF-Binary-256-nl223-pca (self)                          556.52       282.18       838.71       0.9931          1.0011            1.0000         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                745.02        50.68       795.71       0.6012          1.4964            1.3827         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                745.02        52.63       797.65       0.6002          1.5046            1.3866         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                745.02        63.34       808.36       0.5985          1.5256            1.3943         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               745.02       108.21       853.23       0.9940          1.0009            1.0000         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               745.02       162.83       907.86       0.9990          1.0001            1.0000         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               745.02       108.49       853.51       0.9938          1.0009            1.0000         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               745.02       166.29       911.31       0.9992          1.0001            1.0000         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               745.02       115.40       860.42       0.9931          1.0011            1.0000         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               745.02       172.34       917.36       0.9991          1.0001            1.0000         2.09
IVF-Binary-256-nl316-pca (self)                          745.02       274.03     1_019.06       0.9934          1.0011            1.0000         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)              958.13        66.76     1_024.88       0.6409          1.4486            1.3318         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)             958.13        79.93     1_038.05       0.6350          1.4901            1.3495         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)             958.13        93.70     1_051.82       0.6333          1.5130            1.3550         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)             958.13       130.41     1_088.54       0.9965          1.0007            1.0000         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)             958.13       187.37     1_145.49       0.9978          1.0005            1.0000         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)            958.13       143.98     1_102.11       0.9982          1.0002            1.0000         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)            958.13       203.81     1_161.94       0.9999          1.0000            1.0000         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)            958.13       160.56     1_118.69       0.9979          1.0003            1.0000         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)            958.13       219.57     1_177.70       0.9998          1.0000            1.0000         3.71
IVF-Binary-512-nl158-random (self)                       958.13       407.18     1_365.30       0.9979          1.0003            1.0000         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             549.57        67.68       617.25       0.6385          1.4531            1.3378         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             549.57        76.10       625.67       0.6365          1.4696            1.3445         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             549.57        83.89       633.47       0.6339          1.5002            1.3523         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            549.57       127.86       677.43       0.9978          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            549.57       183.27       732.84       0.9993          1.0001            1.0000         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            549.57       132.02       681.59       0.9980          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            549.57       189.85       739.43       0.9998          1.0000            1.0000         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            549.57       143.87       693.44       0.9979          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            549.57       202.41       751.99       0.9998          1.0000            1.0000         3.77
IVF-Binary-512-nl223-random (self)                       549.57       357.68       907.26       0.9978          1.0003            1.0000         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             734.61        69.35       803.96       0.6379          1.4598            1.3395         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             734.61        71.62       806.23       0.6370          1.4674            1.3427         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             734.61        80.65       815.26       0.6347          1.4949            1.3507         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            734.61       128.71       863.32       0.9981          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            734.61       182.84       917.45       0.9996          1.0001            1.0000         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            734.61       129.18       863.79       0.9982          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            734.61       187.02       921.63       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            734.61       139.50       874.11       0.9980          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            734.61       197.84       932.45       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-random (self)                       734.61       342.14     1_076.75       0.9980          1.0003            1.0000         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)                 982.41        66.20     1_048.61       0.6576          1.3883            1.2906         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)                982.41        85.25     1_067.66       0.6524          1.4212            1.3033         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)                982.41        94.10     1_076.51       0.6508          1.4401            1.3081         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)                982.41       132.66     1_115.07       0.9969          1.0006            1.0000         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)                982.41       184.85     1_167.25       0.9978          1.0005            1.0000         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)               982.41       144.62     1_127.03       0.9987          1.0001            1.0000         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)               982.41       203.95     1_186.36       0.9999          1.0000            1.0000         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)               982.41       158.39     1_140.80       0.9985          1.0002            1.0000         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)               982.41       218.74     1_201.15       0.9999          1.0000            1.0000         3.71
IVF-Binary-512-nl158-pca (self)                          982.41       405.10     1_387.51       0.9986          1.0002            1.0000         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                573.66        67.92       641.58       0.6553          1.3963            1.2930         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                573.66        73.35       647.02       0.6534          1.4090            1.2987         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                573.66        84.55       658.21       0.6514          1.4319            1.3057         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               573.66       127.81       701.47       0.9983          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               573.66       183.97       757.63       0.9993          1.0001            1.0000         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               573.66       132.04       705.70       0.9986          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               573.66       191.61       765.27       0.9998          1.0000            1.0000         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               573.66       143.92       717.59       0.9985          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               573.66       201.34       775.00       0.9999          1.0000            1.0000         3.77
IVF-Binary-512-nl223-pca (self)                          573.66       359.46       933.12       0.9985          1.0002            1.0000         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                762.57        69.17       831.74       0.6545          1.4014            1.2937         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                762.57        71.58       834.15       0.6536          1.4079            1.2965         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                762.57        81.38       843.95       0.6516          1.4281            1.3030         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               762.57       128.69       891.25       0.9986          1.0002            1.0000         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               762.57       184.72       947.28       0.9996          1.0001            1.0000         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               762.57       129.84       892.41       0.9987          1.0001            1.0000         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               762.57       186.33       948.90       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               762.57       139.23       901.80       0.9986          1.0002            1.0000         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               762.57       197.48       960.04       0.9999          1.0000            1.0000         3.86
IVF-Binary-512-nl316-pca (self)                          762.57       342.67     1_105.23       0.9986          1.0002            1.0000         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)             988.01        99.85     1_087.86       0.6845          1.3532            1.2576         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)            988.01       119.15     1_107.16       0.6792          1.3863            1.2711         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)            988.01       139.08     1_127.09       0.6776          1.4037            1.2752         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)            988.01       166.31     1_154.31       0.9976          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)            988.01       224.84     1_212.85       0.9978          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)           988.01       185.55     1_173.56       0.9997          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)           988.01       248.07     1_236.08       0.9999          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)           988.01       207.48     1_195.49       0.9996          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)           988.01       269.98     1_257.99       1.0000          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-random (self)                      988.01       560.90     1_548.90       0.9995          1.0000            1.0000         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            582.45       102.36       684.81       0.6825          1.3587            1.2602         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            582.45       108.60       691.05       0.6805          1.3722            1.2665         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            582.45       123.16       705.61       0.6783          1.3950            1.2736         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           582.45       165.17       747.62       0.9991          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           582.45       224.32       806.77       0.9994          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           582.45       172.12       754.57       0.9995          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           582.45       313.30       895.75       0.9998          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           582.45       200.49       782.94       0.9996          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           582.45       260.47       842.92       0.9999          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-random (self)                      582.45       499.86     1_082.31       0.9994          1.0001            1.0000         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            855.71       106.25       961.96       0.6814          1.3665            1.2637         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)            855.71       109.76       965.46       0.6806          1.3728            1.2656         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)            855.71       122.31       978.02       0.6785          1.3928            1.2727         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)           855.71       171.77     1_027.48       0.9994          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)           855.71       229.55     1_085.26       0.9997          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)           855.71       172.86     1_028.57       0.9995          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)           855.71       235.64     1_091.35       0.9998          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)           855.71       185.62     1_041.33       0.9996          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)           855.71       251.52     1_107.22       0.9999          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-random (self)                      855.71       477.07     1_332.77       0.9994          1.0001            1.0000         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)              1_076.80       103.43     1_180.23       0.6927          1.3310            1.2428         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)             1_076.80       121.77     1_198.57       0.6876          1.3596            1.2559         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)             1_076.80       150.17     1_226.97       0.6860          1.3757            1.2595         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)             1_076.80       173.21     1_250.01       0.9977          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)             1_076.80       267.25     1_344.05       0.9978          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)            1_076.80       192.58     1_269.37       0.9998          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)            1_076.80       258.45     1_335.25       1.0000          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)            1_076.80       211.28     1_288.08       0.9997          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)            1_076.80       279.18     1_355.98       1.0000          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-pca (self)                       1_076.80       565.67     1_642.46       0.9997          1.0000            1.0000         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               628.59       102.46       731.05       0.6901          1.3388            1.2451         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               628.59       110.35       738.94       0.6884          1.3497            1.2507         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               628.59       126.46       755.06       0.6863          1.3698            1.2573         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              628.59       169.51       798.11       0.9992          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              628.59       229.18       857.77       0.9994          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              628.59       175.49       804.08       0.9996          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              628.59       245.58       874.17       0.9999          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              628.59       193.21       821.80       0.9996          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              628.59       259.96       888.55       1.0000          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-pca (self)                         628.59       496.76     1_125.35       0.9995          1.0001            1.0000         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               865.41       105.28       970.68       0.6893          1.3439            1.2492         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)               865.41       110.30       975.70       0.6884          1.3499            1.2513         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)               865.41       121.64       987.05       0.6867          1.3674            1.2565         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)              865.41       174.13     1_039.54       0.9995          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)              865.41       235.62     1_101.02       0.9997          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)              865.41       172.69     1_038.10       0.9996          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)              865.41       234.73     1_100.14       0.9999          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)              865.41       185.13     1_050.54       0.9997          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)              865.41       271.92     1_137.32       1.0000          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-pca (self)                         865.41       479.90     1_345.31       0.9995          1.0001            1.0000         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)                963.28       185.83     1_149.11       0.0686          6.6373            6.1345         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               963.28       206.04     1_169.32       0.0552          7.8718            7.1703         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               963.28       208.96     1_172.23       0.0506          8.7173            7.9030         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               963.28       221.77     1_185.05       0.3995          1.6136            1.5282         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               963.28       388.99     1_352.26       0.6372          1.2495            1.1925         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              963.28       238.68     1_201.96       0.3092          1.8460            1.7473         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              963.28       420.88     1_384.15       0.4802          1.4437            1.3765         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              963.28       250.08     1_213.36       0.2742          1.9837            1.8755         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              963.28       436.99     1_400.27       0.4127          1.5616            1.4852         1.68
IVF-Binary-256-nl158-sign (self)                         963.28       713.85     1_677.13       0.3153          1.8338            1.7387         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               519.45       174.17       693.62       0.0663          6.5482            6.0899         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               519.45       178.79       698.24       0.0608          6.9856            6.4759         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               519.45       189.63       709.08       0.0542          7.8630            7.2917         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              519.45       213.01       732.46       0.3662          1.6629            1.5796         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              519.45       377.39       896.84       0.5970          1.2833            1.2269         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              519.45       218.23       737.68       0.3332          1.7517            1.6663         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              519.45       385.66       905.11       0.5316          1.3585            1.2997         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              519.45       230.88       750.33       0.2923          1.8995            1.8113         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              519.45       404.85       924.30       0.4448          1.4906            1.4280         1.75
IVF-Binary-256-nl223-sign (self)                         519.45       632.74     1_152.19       0.3388          1.7404            1.6554         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               714.92       169.53       884.44       0.0659          6.5012            6.0523         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               714.92       173.91       888.83       0.0631          6.7220            6.2235         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               714.92       182.36       897.27       0.0561          7.4986            6.9108         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              714.92       211.14       926.06       0.3648          1.6702            1.5892         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              714.92       372.52     1_087.44       0.5865          1.2945            1.2405         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              714.92       224.64       939.56       0.3479          1.7151            1.6275         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              714.92       378.54     1_093.46       0.5531          1.3338            1.2771         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              714.92       225.70       940.61       0.3051          1.8539            1.7618         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              714.92       393.70     1_108.61       0.4659          1.4564            1.3961         1.84
IVF-Binary-256-nl316-sign (self)                         714.92       611.98     1_326.90       0.3540          1.7015            1.6180         1.84
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
Exhaustive (query)                                        70.07     1_327.85     1_397.92       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         70.07     4_321.26     4_391.34       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                134.59       264.87       399.47       0.5547          1.7646            1.5366         2.03
ExhaustiveBinary-256-random-rf10 (query)                 134.59       411.39       545.98       0.9898          1.0017            1.0000         2.03
ExhaustiveBinary-256-random-rf20 (query)                 134.59       550.21       684.80       0.9985          1.0002            1.0000         2.03
ExhaustiveBinary-256-random (self)                       134.59     1_297.39     1_431.99       0.9899          1.0016            1.0000         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   218.73       266.62       485.35       0.5767          1.6243            1.4311         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    218.73       426.58       645.31       0.9904          1.0016            1.0000         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    218.73       557.13       775.86       0.9984          1.0002            1.0000         2.03
ExhaustiveBinary-256-pca (self)                          218.73     1_308.42     1_527.15       0.9905          1.0016            1.0000         2.03
ExhaustiveBinary-512-random_no_rr (query)                215.75       394.16       609.91       0.6013          1.6760            1.4608         4.05
ExhaustiveBinary-512-random-rf10 (query)                 215.75       545.45       761.20       0.9977          1.0003            1.0000         4.05
ExhaustiveBinary-512-random-rf20 (query)                 215.75       686.01       901.76       0.9998          1.0000            1.0000         4.05
ExhaustiveBinary-512-random (self)                       215.75     1_727.42     1_943.16       0.9975          1.0003            1.0000         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   292.26       390.05       682.30       0.6443          1.4426            1.3064         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    292.26       536.39       828.65       0.9985          1.0002            1.0000         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    292.26       679.01       971.26       0.9999          1.0000            1.0000         4.05
ExhaustiveBinary-512-pca (self)                          292.26     1_721.09     2_013.35       0.9984          1.0002            1.0000         4.05
ExhaustiveBinary-1024-random_no_rr (query)               259.31       569.41       828.72       0.6624          1.4553            1.3048         8.11
ExhaustiveBinary-1024-random-rf10 (query)                259.31       731.63       990.94       0.9995          1.0001            1.0000         8.11
ExhaustiveBinary-1024-random-rf20 (query)                259.31       887.80     1_147.11       1.0000          1.0000            1.0000         8.11
ExhaustiveBinary-1024-random (self)                      259.31     2_389.15     2_648.46       0.9994          1.0001            1.0000         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  346.58       566.44       913.02       0.6865          1.3603            1.2383         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   346.58       728.56     1_075.15       0.9996          1.0001            1.0000         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   346.58       881.94     1_228.52       1.0000          1.0000            1.0000         8.11
ExhaustiveBinary-1024-pca (self)                         346.58     2_432.86     2_779.45       0.9996          1.0001            1.0000         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   83.91       658.11       742.02       0.0400         18.1511           13.6734         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    83.91       720.02       803.92       0.1821          2.5573            2.4620         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    83.91     1_099.96     1_183.87       0.3140          1.8429            1.7786         3.05
ExhaustiveBinary-512-sign (self)                          83.91     2_314.73     2_398.63       0.1897          2.5286            2.4283         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)            1_874.64        82.77     1_957.42       0.5633          1.6328            1.4874         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           1_874.64        94.79     1_969.43       0.5600          1.6630            1.5060         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           1_874.64       104.25     1_978.90       0.5583          1.6921            1.5135         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           1_874.64       174.25     2_048.89       0.9917          1.0013            1.0000         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           1_874.64       271.57     2_146.22       0.9978          1.0004            1.0000         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          1_874.64       180.44     2_055.09       0.9917          1.0013            1.0000         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          1_874.64       273.04     2_147.68       0.9989          1.0001            1.0000         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          1_874.64       182.59     2_057.23       0.9910          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          1_874.64       280.06     2_154.70       0.9988          1.0001            1.0000         2.34
IVF-Binary-256-nl158-random (self)                     1_874.64       397.51     2_272.15       0.9919          1.0013            1.0000         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)             920.94        77.60       998.54       0.5617          1.6446            1.4977         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)             920.94        80.73     1_001.66       0.5603          1.6574            1.5054         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)             920.94        88.67     1_009.61       0.5590          1.6801            1.5121         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)            920.94       174.14     1_095.08       0.9924          1.0011            1.0000         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)            920.94       265.92     1_186.86       0.9988          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)            920.94       171.29     1_092.23       0.9918          1.0012            1.0000         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)            920.94       272.99     1_193.93       0.9989          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)            920.94       180.54     1_101.48       0.9911          1.0014            1.0000         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)            920.94       277.73     1_198.67       0.9988          1.0001            1.0000         2.47
IVF-Binary-256-nl223-random (self)                       920.94       375.75     1_296.69       0.9920          1.0012            1.0000         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)           1_130.89        82.59     1_213.48       0.5616          1.6430            1.4953         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           1_130.89        83.49     1_214.38       0.5609          1.6507            1.4993         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           1_130.89        90.82     1_221.71       0.5595          1.6715            1.5088         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          1_130.89       177.14     1_308.03       0.9924          1.0011            1.0000         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          1_130.89       275.12     1_406.02       0.9990          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          1_130.89       177.32     1_308.21       0.9920          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          1_130.89       272.15     1_403.04       0.9990          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          1_130.89       184.24     1_315.13       0.9913          1.0013            1.0000         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          1_130.89       283.16     1_414.05       0.9988          1.0001            1.0000         2.65
IVF-Binary-256-nl316-random (self)                     1_130.89       373.79     1_504.69       0.9922          1.0012            1.0000         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               1_935.24        71.09     2_006.33       0.5839          1.5249            1.4040         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              1_935.24        83.61     2_018.85       0.5812          1.5475            1.4163         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              1_935.24        89.19     2_024.43       0.5800          1.5661            1.4216         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              1_935.24       168.75     2_103.99       0.9918          1.0013            1.0000         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              1_935.24       262.51     2_197.75       0.9977          1.0004            1.0000         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             1_935.24       174.25     2_109.49       0.9918          1.0013            1.0000         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             1_935.24       272.26     2_207.50       0.9988          1.0002            1.0000         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             1_935.24       182.37     2_117.61       0.9912          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             1_935.24       280.26     2_215.50       0.9987          1.0002            1.0000         2.34
IVF-Binary-256-nl158-pca (self)                        1_935.24       393.55     2_328.79       0.9920          1.0013            1.0000         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)                983.83        77.07     1_060.90       0.5829          1.5323            1.4104         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)                983.83        80.19     1_064.02       0.5817          1.5427            1.4161         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)                983.83        88.82     1_072.65       0.5806          1.5586            1.4197         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)               983.83       172.48     1_156.31       0.9924          1.0012            1.0000         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)               983.83       265.80     1_249.63       0.9988          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)               983.83       172.48     1_156.31       0.9921          1.0013            1.0000         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)               983.83       277.06     1_260.89       0.9988          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)               983.83       179.59     1_163.42       0.9914          1.0014            1.0000         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)               983.83       282.73     1_266.57       0.9987          1.0002            1.0000         2.47
IVF-Binary-256-nl223-pca (self)                          983.83       375.87     1_359.70       0.9921          1.0013            1.0000         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              1_221.06        81.95     1_303.01       0.5828          1.5321            1.4094         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              1_221.06        85.10     1_306.16       0.5823          1.5371            1.4112         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              1_221.06        90.68     1_311.74       0.5810          1.5509            1.4187         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             1_221.06       176.97     1_398.03       0.9923          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             1_221.06       282.21     1_503.26       0.9990          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             1_221.06       176.25     1_397.31       0.9920          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             1_221.06       273.89     1_494.95       0.9989          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             1_221.06       206.87     1_427.93       0.9914          1.0014            1.0000         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             1_221.06       280.00     1_501.05       0.9988          1.0002            1.0000         2.65
IVF-Binary-256-nl316-pca (self)                        1_221.06       372.65     1_593.71       0.9923          1.0012            1.0000         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)            1_935.70       107.45     2_043.15       0.6093          1.5616            1.4208         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)           1_935.70       121.21     2_056.91       0.6053          1.5925            1.4401         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)           1_935.70       133.67     2_069.37       0.6034          1.6196            1.4475         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)           1_935.70       201.42     2_137.13       0.9973          1.0004            1.0000         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)           1_935.70       296.14     2_231.84       0.9984          1.0003            1.0000         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)          1_935.70       212.79     2_148.49       0.9983          1.0002            1.0000         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)          1_935.70       314.21     2_249.91       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)          1_935.70       224.89     2_160.59       0.9980          1.0002            1.0000         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)          1_935.70       323.64     2_259.34       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-random (self)                     1_935.70       524.27     2_459.97       0.9982          1.0002            1.0000         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)             969.09       110.67     1_079.76       0.6074          1.5768            1.4307         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)             969.09       111.19     1_080.28       0.6059          1.5910            1.4390         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)             969.09       123.14     1_092.23       0.6046          1.6113            1.4455         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)            969.09       207.19     1_176.28       0.9982          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)            969.09       307.63     1_276.72       0.9996          1.0001            1.0000         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)            969.09       206.11     1_175.20       0.9982          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)            969.09       315.32     1_284.41       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)            969.09       219.00     1_188.09       0.9980          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)            969.09       324.28     1_293.37       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-random (self)                       969.09       487.65     1_456.74       0.9982          1.0002            1.0000         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)           1_230.66       109.20     1_339.86       0.6068          1.5788            1.4333         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)           1_230.66       114.11     1_344.77       0.6061          1.5861            1.4378         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)           1_230.66       121.25     1_351.90       0.6045          1.6061            1.4455         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)          1_230.66       204.45     1_435.11       0.9985          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)          1_230.66       305.58     1_536.24       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)          1_230.66       203.81     1_434.47       0.9984          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)          1_230.66       305.88     1_536.54       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)          1_230.66       213.00     1_443.65       0.9982          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)          1_230.66       317.86     1_548.52       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-random (self)                     1_230.66       478.98     1_709.64       0.9983          1.0002            1.0000         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)               2_058.39       101.28     2_159.67       0.6498          1.3846            1.2884         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)              2_058.39       116.00     2_174.39       0.6473          1.4005            1.2964         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)              2_058.39       128.57     2_186.96       0.6462          1.4122            1.3004         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)              2_058.39       199.60     2_257.99       0.9975          1.0004            1.0000         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)              2_058.39       295.91     2_354.31       0.9984          1.0003            1.0000         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)             2_058.39       207.19     2_265.58       0.9987          1.0001            1.0000         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)             2_058.39       307.14     2_365.53       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)             2_058.39       218.78     2_277.17       0.9986          1.0001            1.0000         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)             2_058.39       324.23     2_382.62       0.9999          1.0000            1.0000         4.36
IVF-Binary-512-nl158-pca (self)                        2_058.39       509.13     2_567.52       0.9987          1.0001            1.0000         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_087.52       104.53     1_192.05       0.6484          1.3915            1.2913         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_087.52       111.57     1_199.09       0.6474          1.3991            1.2957         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_087.52       121.50     1_209.02       0.6464          1.4094            1.2989         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_087.52       199.65     1_287.17       0.9986          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_087.52       297.23     1_384.75       0.9996          1.0001            1.0000         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_087.52       205.85     1_293.37       0.9987          1.0001            1.0000         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_087.52       303.02     1_390.54       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_087.52       213.86     1_301.38       0.9986          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_087.52       314.97     1_402.49       0.9999          1.0000            1.0000         4.49
IVF-Binary-512-nl223-pca (self)                        1_087.52       483.98     1_571.50       0.9987          1.0002            1.0000         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)              1_288.60       110.21     1_398.81       0.6480          1.3924            1.2943         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)              1_288.60       112.93     1_401.54       0.6476          1.3963            1.2956         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)              1_288.60       124.27     1_412.87       0.6465          1.4068            1.2991         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)             1_288.60       203.57     1_492.18       0.9988          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)             1_288.60       300.06     1_588.66       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)             1_288.60       202.95     1_491.55       0.9987          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)             1_288.60       305.37     1_593.97       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)             1_288.60       214.75     1_503.35       0.9986          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)             1_288.60       323.24     1_611.84       0.9999          1.0000            1.0000         4.67
IVF-Binary-512-nl316-pca (self)                        1_288.60       474.04     1_762.65       0.9987          1.0001            1.0000         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)           1_989.88       151.62     2_141.50       0.6686          1.3871            1.2852         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)          1_989.88       169.19     2_159.06       0.6654          1.4070            1.2940         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)          1_989.88       188.54     2_178.42       0.6639          1.4242            1.2993         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)          1_989.88       252.11     2_241.99       0.9983          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)          1_989.88       352.51     2_342.39       0.9985          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)         1_989.88       269.58     2_259.46       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)         1_989.88       380.55     2_370.42       0.9999          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)         1_989.88       306.75     2_296.63       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)         1_989.88       397.06     2_386.94       1.0000          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-random (self)                    1_989.88       730.98     2_720.86       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_045.84       156.83     1_202.68       0.6666          1.3978            1.2895         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_045.84       165.09     1_210.93       0.6654          1.4067            1.2934         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_045.84       179.50     1_225.34       0.6643          1.4195            1.2982         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_045.84       270.67     1_316.52       0.9994          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_045.84       361.26     1_407.10       0.9997          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_045.84       260.90     1_306.75       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_045.84       371.45     1_417.29       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_045.84       283.45     1_329.29       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_045.84       392.67     1_438.51       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-random (self)                    1_045.84       689.55     1_735.39       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)          1_260.30       161.69     1_422.00       0.6664          1.3994            1.2908         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)          1_260.30       164.08     1_424.39       0.6658          1.4041            1.2928         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)          1_260.30       177.48     1_437.78       0.6645          1.4169            1.2974         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)         1_260.30       260.84     1_521.15       0.9996          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)         1_260.30       371.90     1_632.20       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)         1_260.30       261.57     1_521.87       0.9996          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)         1_260.30       375.67     1_635.97       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)         1_260.30       282.01     1_542.31       0.9996          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)         1_260.30       396.22     1_656.52       1.0000          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-random (self)                    1_260.30       684.05     1_944.35       0.9996          1.0001            1.0000         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)              2_070.78       152.36     2_223.14       0.6914          1.3138            1.2265         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)             2_070.78       169.92     2_240.70       0.6887          1.3290            1.2329         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)             2_070.78       187.32     2_258.10       0.6877          1.3391            1.2353         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)             2_070.78       249.54     2_320.32       0.9983          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)             2_070.78       350.11     2_420.89       0.9985          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)            2_070.78       267.92     2_338.70       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)            2_070.78       370.92     2_441.70       0.9999          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)            2_070.78       285.28     2_356.06       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)            2_070.78       394.99     2_465.77       1.0000          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-pca (self)                       2_070.78       728.74     2_799.52       0.9996          1.0000            1.0000         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_129.24       157.02     1_286.26       0.6897          1.3215            1.2293         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_129.24       168.48     1_297.72       0.6887          1.3276            1.2310         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_129.24       180.21     1_309.45       0.6878          1.3371            1.2341         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_129.24       254.51     1_383.75       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_129.24       359.30     1_488.54       0.9997          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_129.24       266.04     1_395.28       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_129.24       370.68     1_499.92       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_129.24       280.94     1_410.18       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_129.24       391.23     1_520.47       1.0000          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-pca (self)                       1_129.24       693.99     1_823.23       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_350.61       161.82     1_512.42       0.6897          1.3223            1.2301         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_350.61       164.07     1_514.68       0.6893          1.3259            1.2310         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_350.61       177.92     1_528.52       0.6882          1.3349            1.2341         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_350.61       260.91     1_611.51       0.9997          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_350.61       366.93     1_717.54       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_350.61       260.99     1_611.60       0.9997          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_350.61       372.04     1_722.64       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_350.61       276.16     1_626.76       0.9997          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_350.61       391.85     1_742.46       1.0000          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-pca (self)                       1_350.61       679.27     2_029.88       0.9996          1.0000            1.0000         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)              1_807.89       284.87     2_092.75       0.0594          7.9493            7.2573         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)             1_807.89       302.25     2_110.13       0.0523          9.1717            8.0433         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)             1_807.89       317.40     2_125.29       0.0490         10.2313            8.7259         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)             1_807.89       360.69     2_168.57       0.3200          1.8552            1.7422         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)             1_807.89       660.11     2_468.00       0.5141          1.4089            1.3383         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)            1_807.89       383.59     2_191.47       0.2782          1.9998            1.8714         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)            1_807.89       674.32     2_482.20       0.4424          1.5272            1.4451         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)            1_807.89       389.02     2_196.90       0.2558          2.1020            1.9577         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)            1_807.89       684.91     2_492.79       0.4010          1.6091            1.5201         3.36
IVF-Binary-512-nl158-sign (self)                       1_807.89     1_053.85     2_861.74       0.2859          1.9751            1.8430         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               860.96       290.01     1_150.97       0.0573          7.8797            7.1620         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               860.96       296.61     1_157.57       0.0543          8.2745            7.4598         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               860.96       312.20     1_173.16       0.0504          9.1593            8.1667         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              860.96       362.78     1_223.74       0.3141          1.8536            1.7519         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              860.96       653.10     1_514.06       0.5018          1.4180            1.3471         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              860.96       368.27     1_229.23       0.2951          1.9148            1.8057         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              860.96       658.23     1_519.19       0.4685          1.4696            1.3955         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              860.96       385.73     1_246.69       0.2690          2.0219            1.8959         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              860.96       686.27     1_547.24       0.4174          1.5671            1.4860         3.49
IVF-Binary-512-nl223-sign (self)                         860.96     1_032.04     1_893.00       0.3019          1.8966            1.7840         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)             1_077.96       291.61     1_369.57       0.0576          7.7417            7.1217         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)             1_077.96       298.58     1_376.54       0.0558          7.9505            7.2924         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)             1_077.96       305.81     1_383.76       0.0519          8.6482            7.8795         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)            1_077.96       370.90     1_448.85       0.3184          1.8361            1.7366         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)            1_077.96       653.24     1_731.20       0.5013          1.4137            1.3449         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)            1_077.96       368.65     1_446.60       0.3085          1.8697            1.7662         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)            1_077.96       704.72     1_782.68       0.4838          1.4395            1.3714         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)            1_077.96       387.94     1_465.90       0.2821          1.9696            1.8495         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)            1_077.96       675.56     1_753.52       0.4343          1.5288            1.4528         3.67
IVF-Binary-512-nl316-sign (self)                       1_077.96     1_024.18     2_102.13       0.3137          1.8531            1.7443         3.67
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
Exhaustive (query)                                       100.73     1_852.96     1_953.69       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.73     6_245.07     6_345.80       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                192.95       286.09       479.04       0.5361          1.8068            1.5908         2.28
ExhaustiveBinary-256-random-rf10 (query)                 192.95       445.07       638.02       0.9868          1.0022            1.0000         2.28
ExhaustiveBinary-256-random-rf20 (query)                 192.95       604.54       797.50       0.9980          1.0003            1.0000         2.28
ExhaustiveBinary-256-random (self)                       192.95     1_407.24     1_600.19       0.9876          1.0021            1.0000         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   395.98       292.30       688.28       0.5754          1.5495            1.4128         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    395.98       446.60       842.58       0.9895          1.0018            1.0000         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    395.98       622.99     1_018.97       0.9983          1.0002            1.0000         2.28
ExhaustiveBinary-256-pca (self)                          395.98     1_399.01     1_794.99       0.9897          1.0017            1.0000         2.28
ExhaustiveBinary-512-random_no_rr (query)                297.69       421.59       719.28       0.5866          1.6778            1.4946         4.55
ExhaustiveBinary-512-random-rf10 (query)                 297.69       588.45       886.13       0.9966          1.0005            1.0000         4.55
ExhaustiveBinary-512-random-rf20 (query)                 297.69       760.73     1_058.42       0.9997          1.0001            1.0000         4.55
ExhaustiveBinary-512-random (self)                       297.69     1_862.23     2_159.92       0.9969          1.0004            1.0000         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   495.39       421.25       916.65       0.6388          1.4217            1.3032         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    495.39       587.82     1_083.22       0.9979          1.0003            1.0000         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    495.39       755.57     1_250.96       0.9998          1.0000            1.0000         4.55
ExhaustiveBinary-512-pca (self)                          495.39     1_869.99     2_365.39       0.9981          1.0002            1.0000         4.55
ExhaustiveBinary-1024-random_no_rr (query)               498.53       627.42     1_125.96       0.6446          1.4909            1.3512         9.11
ExhaustiveBinary-1024-random-rf10 (query)                498.53       812.49     1_311.03       0.9993          1.0001            1.0000         9.11
ExhaustiveBinary-1024-random-rf20 (query)                498.53       999.97     1_498.50       0.9999          1.0000            1.0000         9.11
ExhaustiveBinary-1024-random (self)                      498.53     2_652.03     3_150.56       0.9994          1.0001            1.0000         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  695.71       629.34     1_325.04       0.6795          1.3452            1.2483         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   695.71       805.65     1_501.36       0.9996          1.0001            1.0000         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   695.71       986.44     1_682.15       1.0000          1.0000            1.0000         9.11
ExhaustiveBinary-1024-pca (self)                         695.71     2_622.46     3_318.17       0.9997          1.0000            1.0000         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  127.39       843.10       970.49       0.0420         17.7082           13.0970         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   127.39       921.74     1_049.13       0.1896          2.5240            2.4052         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   127.39     1_420.75     1_548.14       0.3229          1.8300            1.7348         4.58
ExhaustiveBinary-768-sign (self)                         127.39     2_941.67     3_069.06       0.1997          2.4832            2.3546         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)            2_706.51       105.98     2_812.49       0.5429          1.7099            1.5460         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)           2_706.51       119.50     2_826.01       0.5407          1.7331            1.5595         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)           2_706.51       124.95     2_831.46       0.5397          1.7545            1.5687         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)           2_706.51       217.27     2_923.78       0.9891          1.0018            1.0000         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)           2_706.51       333.30     3_039.81       0.9984          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)          2_706.51       224.87     2_931.38       0.9884          1.0019            1.0000         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)          2_706.51       348.85     3_055.36       0.9986          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)          2_706.51       231.99     2_938.50       0.9877          1.0021            1.0000         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)          2_706.51       358.11     3_064.62       0.9983          1.0002            1.0000         2.74
IVF-Binary-256-nl158-random (self)                     2_706.51       518.78     3_225.29       0.9891          1.0018            1.0000         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)           1_351.01       108.88     1_459.90       0.5420          1.7182            1.5522         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)           1_351.01       112.34     1_463.36       0.5412          1.7279            1.5568         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)           1_351.01       121.36     1_472.37       0.5401          1.7508            1.5657         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)          1_351.01       221.44     1_572.45       0.9888          1.0018            1.0000         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)          1_351.01       337.71     1_688.72       0.9986          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)          1_351.01       215.10     1_566.11       0.9885          1.0019            1.0000         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)          1_351.01       337.01     1_688.02       0.9985          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)          1_351.01       220.85     1_571.86       0.9877          1.0020            1.0000         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)          1_351.01       349.15     1_700.16       0.9983          1.0002            1.0000         2.93
IVF-Binary-256-nl223-random (self)                     1_351.01       472.49     1_823.50       0.9890          1.0018            1.0000         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)           1_641.57       108.48     1_750.04       0.5422          1.7108            1.5503         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)           1_641.57       107.93     1_749.49       0.5417          1.7157            1.5534         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)           1_641.57       117.50     1_759.06       0.5408          1.7316            1.5599         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)          1_641.57       222.34     1_863.91       0.9891          1.0018            1.0000         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)          1_641.57       344.61     1_986.18       0.9987          1.0002            1.0000         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)          1_641.57       224.68     1_866.24       0.9888          1.0018            1.0000         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)          1_641.57       349.23     1_990.80       0.9986          1.0002            1.0000         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)          1_641.57       229.32     1_870.89       0.9882          1.0020            1.0000         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)          1_641.57       354.15     1_995.71       0.9984          1.0002            1.0000         3.21
IVF-Binary-256-nl316-random (self)                     1_641.57       487.05     2_128.61       0.9895          1.0017            1.0000         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)               2_910.74       101.71     3_012.45       0.5812          1.4959            1.3933         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)              2_910.74       130.47     3_041.21       0.5795          1.5088            1.3992         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)              2_910.74       142.89     3_053.63       0.5787          1.5186            1.4017         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)              2_910.74       239.32     3_150.06       0.9913          1.0014            1.0000         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)              2_910.74       320.75     3_231.49       0.9984          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)             2_910.74       215.58     3_126.32       0.9907          1.0015            1.0000         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)             2_910.74       339.05     3_249.79       0.9987          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)             2_910.74       220.20     3_130.94       0.9902          1.0016            1.0000         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)             2_910.74       339.74     3_250.48       0.9985          1.0002            1.0000         2.74
IVF-Binary-256-nl158-pca (self)                        2_910.74       482.55     3_393.29       0.9910          1.0014            1.0000         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)              1_569.11        98.19     1_667.29       0.5799          1.5020            1.3963         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)              1_569.11       101.81     1_670.91       0.5794          1.5067            1.3987         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)              1_569.11       110.36     1_679.47       0.5786          1.5169            1.4018         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)             1_569.11       211.50     1_780.60       0.9909          1.0015            1.0000         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)             1_569.11       326.41     1_895.51       0.9987          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)             1_569.11       212.11     1_781.22       0.9906          1.0015            1.0000         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)             1_569.11       330.00     1_899.11       0.9987          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)             1_569.11       222.36     1_791.47       0.9901          1.0016            1.0000         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)             1_569.11       341.82     1_910.92       0.9985          1.0002            1.0000         2.93
IVF-Binary-256-nl223-pca (self)                        1_569.11       478.99     2_048.10       0.9909          1.0014            1.0000         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)              1_835.26       107.80     1_943.07       0.5807          1.4988            1.3925         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)              1_835.26       108.19     1_943.45       0.5804          1.5016            1.3940         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)              1_835.26       114.49     1_949.76       0.5798          1.5081            1.3970         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)             1_835.26       219.30     2_054.56       0.9912          1.0014            1.0000         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)             1_835.26       386.40     2_221.67       0.9989          1.0001            1.0000         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)             1_835.26       251.68     2_086.94       0.9909          1.0015            1.0000         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)             1_835.26       398.10     2_233.36       0.9988          1.0001            1.0000         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)             1_835.26       234.74     2_070.00       0.9903          1.0016            1.0000         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)             1_835.26       345.47     2_180.74       0.9986          1.0002            1.0000         3.21
IVF-Binary-256-nl316-pca (self)                        1_835.26       482.67     2_317.93       0.9913          1.0014            1.0000         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)            2_757.12       136.29     2_893.40       0.5925          1.6007            1.4618         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)           2_757.12       156.45     2_913.57       0.5898          1.6229            1.4748         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)           2_757.12       167.24     2_924.36       0.5888          1.6417            1.4806         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)           2_757.12       254.41     3_011.53       0.9972          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)           2_757.12       375.46     3_132.58       0.9993          1.0001            1.0000         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)          2_757.12       266.40     3_023.52       0.9972          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)          2_757.12       393.70     3_150.82       0.9998          1.0000            1.0000         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)          2_757.12       279.62     3_036.74       0.9970          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)          2_757.12       416.97     3_174.08       0.9997          1.0000            1.0000         5.02
IVF-Binary-512-nl158-random (self)                     2_757.12       672.65     3_429.77       0.9974          1.0003            1.0000         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)           1_501.97       143.15     1_645.12       0.5912          1.6104            1.4673         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)           1_501.97       149.38     1_651.36       0.5903          1.6195            1.4718         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)           1_501.97       173.93     1_675.91       0.5890          1.6407            1.4794         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)          1_501.97       256.64     1_758.61       0.9972          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)          1_501.97       376.17     1_878.15       0.9996          1.0001            1.0000         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)          1_501.97       259.51     1_761.49       0.9972          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)          1_501.97       383.10     1_885.07       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)          1_501.97       271.71     1_773.69       0.9969          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)          1_501.97       399.69     1_901.66       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-random (self)                     1_501.97       632.53     2_134.50       0.9974          1.0003            1.0000         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)           1_810.83       158.11     1_968.94       0.5911          1.6075            1.4667         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)           1_810.83       155.48     1_966.31       0.5907          1.6116            1.4696         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)           1_810.83       169.07     1_979.89       0.5897          1.6270            1.4769         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)          1_810.83       272.75     2_083.58       0.9974          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)          1_810.83       389.67     2_200.49       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)          1_810.83       263.43     2_074.25       0.9973          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)          1_810.83       393.55     2_204.38       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)          1_810.83       272.47     2_083.30       0.9971          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)          1_810.83       401.65     2_212.47       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-random (self)                     1_810.83       624.79     2_435.62       0.9976          1.0003            1.0000         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)               3_018.91       129.78     3_148.69       0.6432          1.3824            1.2890         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)              3_018.91       143.93     3_162.85       0.6414          1.3938            1.2945         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)              3_018.91       154.41     3_173.33       0.6407          1.4030            1.2975         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)              3_018.91       245.78     3_264.70       0.9980          1.0003            1.0000         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)              3_018.91       366.64     3_385.56       0.9994          1.0001            1.0000         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)             3_018.91       256.06     3_274.98       0.9982          1.0002            1.0000         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)             3_018.91       380.72     3_399.63       0.9999          1.0000            1.0000         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)             3_018.91       268.69     3_287.60       0.9981          1.0003            1.0000         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)             3_018.91       399.63     3_418.55       0.9998          1.0000            1.0000         5.02
IVF-Binary-512-nl158-pca (self)                        3_018.91       659.86     3_678.77       0.9983          1.0002            1.0000         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_658.48       141.54     1_800.02       0.6420          1.3885            1.2932         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_658.48       142.06     1_800.53       0.6414          1.3935            1.2956         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_658.48       153.24     1_811.71       0.6407          1.4030            1.2985         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_658.48       256.71     1_915.18       0.9982          1.0002            1.0000         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_658.48       385.16     2_043.64       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_658.48       255.49     1_913.97       0.9982          1.0002            1.0000         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_658.48       382.30     2_040.78       0.9998          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_658.48       265.16     1_923.64       0.9981          1.0003            1.0000         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_658.48       396.16     2_054.63       0.9998          1.0000            1.0000         5.21
IVF-Binary-512-nl223-pca (self)                        1_658.48       610.88     2_269.35       0.9983          1.0002            1.0000         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)              2_006.17       164.20     2_170.37       0.6422          1.3876            1.2923         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)              2_006.17       154.06     2_160.23       0.6419          1.3896            1.2933         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)              2_006.17       166.84     2_173.01       0.6413          1.3956            1.2957         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)             2_006.17       269.44     2_275.61       0.9984          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)             2_006.17       391.05     2_397.22       0.9999          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)             2_006.17       265.77     2_271.94       0.9983          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)             2_006.17       393.78     2_399.95       0.9999          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)             2_006.17       278.53     2_284.71       0.9981          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)             2_006.17       417.06     2_423.23       0.9999          1.0000            1.0000         5.48
IVF-Binary-512-nl316-pca (self)                        2_006.17       632.85     2_639.02       0.9984          1.0002            1.0000         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)           3_010.28       201.47     3_211.74       0.6492          1.4402            1.3299         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)          3_010.28       216.49     3_226.76       0.6468          1.4562            1.3410         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)          3_010.28       233.15     3_243.42       0.6457          1.4688            1.3455         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)          3_010.28       324.30     3_334.58       0.9990          1.0002            1.0000         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)          3_010.28       453.10     3_463.37       0.9994          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)         3_010.28       345.26     3_355.54       0.9995          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)         3_010.28       484.55     3_494.83       0.9999          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)         3_010.28       364.54     3_374.82       0.9994          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)         3_010.28       504.94     3_515.22       0.9999          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-random (self)                    3_010.28       945.56     3_955.83       0.9995          1.0001            1.0000         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_659.85       203.71     1_863.56       0.6479          1.4466            1.3357         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_659.85       209.91     1_869.75       0.6472          1.4538            1.3394         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_659.85       226.78     1_886.63       0.6460          1.4684            1.3442         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_659.85       331.61     1_991.46       0.9993          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_659.85       462.29     2_122.14       0.9997          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_659.85       336.70     1_996.55       0.9994          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_659.85       471.59     2_131.43       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_659.85       357.24     2_017.09       0.9994          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_659.85       496.73     2_156.58       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-random (self)                    1_659.85       877.34     2_537.18       0.9995          1.0001            1.0000         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)          1_983.57       212.52     2_196.09       0.6478          1.4461            1.3344        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)          1_983.57       215.48     2_199.05       0.6475          1.4494            1.3361        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)          1_983.57       230.03     2_213.60       0.6465          1.4599            1.3407        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)         1_983.57       348.92     2_332.49       0.9995          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)         1_983.57       484.91     2_468.48       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)         1_983.57       351.17     2_334.74       0.9995          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)         1_983.57       483.70     2_467.27       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)         1_983.57       370.64     2_354.21       0.9994          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)         1_983.57       505.34     2_488.91       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-random (self)                    1_983.57       894.79     2_878.36       0.9995          1.0001            1.0000        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)              3_187.76       203.08     3_390.84       0.6828          1.3187            1.2385         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)             3_187.76       217.14     3_404.90       0.6812          1.3273            1.2437         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)             3_187.76       234.12     3_421.88       0.6805          1.3345            1.2454         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)             3_187.76       316.43     3_504.18       0.9992          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)             3_187.76       445.57     3_633.33       0.9994          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)            3_187.76       337.10     3_524.85       0.9997          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)            3_187.76       475.58     3_663.34       1.0000          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)            3_187.76       359.45     3_547.20       0.9996          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)            3_187.76       501.93     3_689.69       1.0000          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-pca (self)                       3_187.76       925.13     4_112.89       0.9997          1.0000            1.0000         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_759.34       205.03     1_964.37       0.6818          1.3238            1.2403         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_759.34       212.51     1_971.84       0.6813          1.3270            1.2428         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_759.34       227.85     1_987.18       0.6806          1.3344            1.2451         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_759.34       328.96     2_088.30       0.9995          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_759.34       462.95     2_222.28       0.9998          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_759.34       341.16     2_100.50       0.9996          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_759.34       476.24     2_235.57       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_759.34       361.72     2_121.05       0.9996          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_759.34       498.51     2_257.85       1.0000          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-pca (self)                       1_759.34       880.67     2_640.01       0.9997          1.0000            1.0000         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             2_089.81       216.84     2_306.65       0.6817          1.3241            1.2409        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)             2_089.81       225.41     2_315.22       0.6815          1.3255            1.2420        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)             2_089.81       232.41     2_322.22       0.6809          1.3307            1.2441        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)            2_089.81       356.18     2_445.99       0.9997          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)            2_089.81       493.34     2_583.15       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)            2_089.81       379.63     2_469.45       0.9996          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)            2_089.81       499.86     2_589.67       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)            2_089.81       369.89     2_459.70       0.9996          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)            2_089.81       508.01     2_597.82       1.0000          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-pca (self)                       2_089.81       897.59     2_987.40       0.9997          1.0000            1.0000        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)              2_544.83       407.83     2_952.66       0.0573          8.2519            7.4750         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)             2_544.83       428.80     2_973.63       0.0520          9.4871            8.2097         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)             2_544.83       442.40     2_987.23       0.0494         10.3662            8.7466         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)             2_544.83       499.96     3_044.79       0.3103          1.8949            1.7846         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)             2_544.83       912.77     3_457.61       0.4776          1.4825            1.3824         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)            2_544.83       516.98     3_061.81       0.2786          2.0014            1.8766         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)            2_544.83       911.88     3_456.72       0.4285          1.5638            1.4621         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)            2_544.83       525.62     3_070.45       0.2621          2.0692            1.9339         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)            2_544.83       988.60     3_533.44       0.4002          1.6175            1.5078         5.04
IVF-Binary-768-nl158-sign (self)                       2_544.83     1_446.48     3_991.31       0.2910          1.9632            1.8395         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)             1_232.93       410.89     1_643.82       0.0570          8.2117            7.3530         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)             1_232.93       414.48     1_647.41       0.0545          8.6681            7.6901         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)             1_232.93       435.64     1_668.56       0.0514          9.6380            8.3214         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)            1_232.93       510.70     1_743.62       0.3111          1.8746            1.7621         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)            1_232.93       918.49     2_151.41       0.4734          1.4732            1.3870         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)            1_232.93       497.78     1_730.71       0.2972          1.9238            1.8036         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)            1_232.93       895.47     2_128.39       0.4499          1.5110            1.4213         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)            1_232.93       515.04     1_747.96       0.2759          2.0111            1.8835         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)            1_232.93       917.02     2_149.95       0.4143          1.5807            1.4851         5.23
IVF-Binary-768-nl223-sign (self)                       1_232.93     1_415.76     2_648.68       0.3092          1.8880            1.7686         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)             1_527.33       410.63     1_937.96       0.0581          7.9072            7.1781         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)             1_527.33       413.41     1_940.74       0.0568          8.1437            7.3474         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)             1_527.33       427.12     1_954.45       0.0534          8.9543            7.8708         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)            1_527.33       503.03     2_030.36       0.3162          1.8517            1.7465         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)            1_527.33       898.37     2_425.70       0.4808          1.4599            1.3740         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)            1_527.33       504.04     2_031.37       0.3081          1.8776            1.7701         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)            1_527.33       905.47     2_432.80       0.4683          1.4792            1.3936         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)            1_527.33       518.27     2_045.60       0.2865          1.9584            1.8357         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)            1_527.33       924.11     2_451.44       0.4338          1.5372            1.4442         5.51
IVF-Binary-768-nl316-sign (self)                       1_527.33     1_450.52     2_977.85       0.3204          1.8449            1.7293         5.51
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

#### Quantised graph (QG)

The `QgIndex` rows in the tables below come from the same run. It is a
SymphonyQG-style index (Gou et al., SIGMOD 2025): a Vamana graph where every
vertex additionally stores its own neighbours' one-bit RaBitQ codes, quantised
against that vertex and pre-transposed into the fast-scan block layout. One hop
is therefore a contiguous read plus one byte-shuffle sweep that estimates all 32
neighbour distances at once, instead of one random memory access and one
distance kernel per neighbour. Exact distances are computed only for the
vertices the walk actually pops, which the estimator needs as its anchor anyway,
so there is no separate re-ranking stage and no `VecStore`.

**This one is not about memory.** Each vector's code is duplicated once per
in-edge, and the raw vectors have to stay resident for the exact distances, so
the index lands at roughly two to three times the size of the data. The point is
query speed at a given recall. If memory is the binding constraint, `IVF-RaBitQ`
above is the index you want.

The graph is not from the paper. SymphonyQG builds its own topology with
random init, repeated search-prune-reverse rounds and a cosine-threshold refill
to force exact-degree regularity; `VamanaIndex` already yields a fixed-degree
graph, so it builds the topology here and what is kept from the paper is the
storage layout and the estimator. Euclidean and cosine only, no Manhattan.

**Tunable parameters *(QG)*:**

- *degree (d)*: Neighbour slots per vertex, a non-zero multiple of 32. Default
  `32`, which is exactly one fast-scan sweep per hop. Doubling it doubles both
  the code footprint and the per-hop work, so it wants a reason; the grid runs
  `32` and `64` to show whether the extra edges pay.
- *l_build (l)*: Beam width during construction, second Vamana pass. This is
  where the build time goes: the encoding is a flat few hundred milliseconds and
  everything else is Vamana. The first pass runs at the crate default, which is
  a small constant, because a wide first pass is both slower and worse.
- *ef_search (ef)*: Beam width at query time, the usual recall/latency dial. The
  grid runs `k`, `2k`, `4k` and `8k`.

Self queries run at `ef_search = 4k`.

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
Exhaustive (query)                                        32.68       707.87       740.55       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.68     2_266.77     2_299.45       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             774.56       182.64       957.20       0.5723          1.0356            1.0352         2.56
ExhaustiveRaBitQ-rf5 (query)                             774.56       228.54     1_003.10       0.9285          1.0016            1.0005         2.56
ExhaustiveRaBitQ-rf10 (query)                            774.56       272.73     1_047.28       0.9851          1.0003            1.0000         2.56
ExhaustiveRaBitQ-rf20 (query)                            774.56       339.25     1_113.80       0.9986          1.0000            1.0000         2.56
ExhaustiveRaBitQ (self)                                  774.56       876.90     1_651.46       0.9853          1.0003            1.0000         2.56
QG-d32-l32-ef15 (query)                                1_613.00       295.06     1_908.06       0.9860          1.0109            1.0000       116.35
QG-d32-l32-ef30 (query)                                1_613.00       419.78     2_032.78       0.9926          1.0043            1.0000       116.35
QG-d32-l32-ef60 (query)                                1_613.00       601.86     2_214.86       0.9950          1.0024            1.0000       116.35
QG-d32-l32-ef120 (query)                               1_613.00       884.03     2_497.03       0.9961          1.0019            1.0000       116.35
QG-d32-l32 (self)                                      1_613.00     2_065.77     3_678.77       0.9954          1.0023            1.0000       116.35
QG-d32-l128-ef15 (query)                               2_291.42       311.21     2_602.64       0.9167          5.7570            1.0000       116.35
QG-d32-l128-ef30 (query)                               2_291.42       443.65     2_735.07       0.9362          4.5665            1.0000       116.35
QG-d32-l128-ef60 (query)                               2_291.42       636.59     2_928.01       0.9526          2.9450            1.0000       116.35
QG-d32-l128-ef120 (query)                              2_291.42       914.46     3_205.88       0.9642          2.3026            1.0000       116.35
QG-d32-l128 (self)                                     2_291.42     2_092.08     4_383.50       0.9530          3.0517            1.0000       116.35
QG-d64-l32-ef15 (query)                                2_487.39       564.34     3_051.73       0.9989          1.0002            1.0000       183.49
QG-d64-l32-ef30 (query)                                2_487.39       787.11     3_274.50       0.9995          1.0001            1.0000       183.49
QG-d64-l32-ef60 (query)                                2_487.39     1_060.63     3_548.02       0.9996          1.0001            1.0000       183.49
QG-d64-l32-ef120 (query)                               2_487.39     1_489.24     3_976.63       0.9997          1.0001            1.0000       183.49
QG-d64-l32 (self)                                      2_487.39     3_497.34     5_984.73       0.9997          1.0001            1.0000       183.49
QG-d64-l128-ef15 (query)                               4_103.65       649.32     4_752.97       0.9952          1.0068            1.0000       183.49
QG-d64-l128-ef30 (query)                               4_103.65       879.42     4_983.06       0.9971          1.0026            1.0000       183.49
QG-d64-l128-ef60 (query)                               4_103.65     1_211.25     5_314.90       0.9982          1.0011            1.0000       183.49
QG-d64-l128-ef120 (query)                              4_103.65     1_739.84     5_843.49       0.9988          1.0007            1.0000       183.49
QG-d64-l128 (self)                                     4_103.65     4_022.17     8_125.82       0.9984          1.0010            1.0000       183.49
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_297.39        86.73     1_384.12       0.5810          1.0333            1.0335         2.67
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_297.39       115.82     1_413.21       0.5810          1.0333            1.0335         2.67
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_297.39       147.57     1_444.96       0.5810          1.0333            1.0335         2.67
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_297.39       156.38     1_453.78       0.9861          1.0002            1.0000         2.67
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_297.39       220.77     1_518.16       0.9988          1.0000            1.0000         2.67
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_297.39       185.28     1_482.67       0.9861          1.0002            1.0000         2.67
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_297.39       244.74     1_542.14       0.9988          1.0000            1.0000         2.67
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_297.39       211.10     1_508.49       0.9861          1.0002            1.0000         2.67
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_297.39       277.33     1_574.72       0.9988          1.0000            1.0000         2.67
IVF-RaBitQ-nl158 (self)                                1_297.39       875.33     2_172.72       0.9988          1.0000            1.0000         2.67
IVF-RaBitQ-nl223-np11-rf0 (query)                        798.19       109.39       907.58       0.5930          1.0314            1.0314         2.83
IVF-RaBitQ-nl223-np14-rf0 (query)                        798.19       123.89       922.08       0.5930          1.0314            1.0313         2.83
IVF-RaBitQ-nl223-np21-rf0 (query)                        798.19       166.28       964.47       0.5930          1.0314            1.0313         2.83
IVF-RaBitQ-nl223-np11-rf10 (query)                       798.19       175.46       973.65       0.9889          1.0002            1.0000         2.83
IVF-RaBitQ-nl223-np11-rf20 (query)                       798.19       228.77     1_026.96       0.9989          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np14-rf10 (query)                       798.19       189.31       987.50       0.9890          1.0002            1.0000         2.83
IVF-RaBitQ-nl223-np14-rf20 (query)                       798.19       246.66     1_044.85       0.9990          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np21-rf10 (query)                       798.19       239.80     1_037.99       0.9890          1.0002            1.0000         2.83
IVF-RaBitQ-nl223-np21-rf20 (query)                       798.19       290.00     1_088.19       0.9990          1.0000            1.0000         2.83
IVF-RaBitQ-nl223 (self)                                  798.19       934.06     1_732.25       0.9992          1.0000            1.0000         2.83
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_031.47       133.14     1_164.61       0.6007          1.0301            1.0300         3.06
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_031.47       143.09     1_174.56       0.6007          1.0301            1.0300         3.06
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_031.47       203.16     1_234.63       0.6008          1.0301            1.0300         3.06
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_031.47       202.24     1_233.70       0.9897          1.0002            1.0000         3.06
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_031.47       254.71     1_286.18       0.9991          1.0001            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_031.47       211.58     1_243.05       0.9898          1.0002            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_031.47       266.37     1_297.83       0.9992          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_031.47       263.78     1_295.25       0.9899          1.0002            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_031.47       312.95     1_344.42       0.9993          1.0000            1.0000         3.06
IVF-RaBitQ-nl316 (self)                                1_031.47     1_032.25     2_063.72       0.9993          1.0000            1.0000         3.06
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
Exhaustive (query)                                        67.68     1_266.96     1_334.65       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         67.68     4_258.17     4_325.85       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           1_405.22       317.98     1_723.19       0.5810          1.0225            1.0225         4.36
ExhaustiveRaBitQ-rf5 (query)                           1_405.22       377.66     1_782.87       0.9276          1.0010            1.0004         4.36
ExhaustiveRaBitQ-rf10 (query)                          1_405.22       424.46     1_829.67       0.9842          1.0002            1.0000         4.36
ExhaustiveRaBitQ-rf20 (query)                          1_405.22       524.56     1_929.78       0.9986          1.0000            1.0000         4.36
ExhaustiveRaBitQ (self)                                1_405.22     1_460.56     2_865.77       0.9846          1.0002            1.0000         4.36
QG-d32-l32-ef15 (query)                                3_428.33       515.42     3_943.76       0.9540          1.9691            1.0000       214.01
QG-d32-l32-ef30 (query)                                3_428.33       704.11     4_132.44       0.9712          1.4250            1.0000       214.01
QG-d32-l32-ef60 (query)                                3_428.33       962.52     4_390.86       0.9816          1.0550            1.0000       214.01
QG-d32-l32-ef120 (query)                               3_428.33     1_295.97     4_724.31       0.9875          1.0197            1.0000       214.01
QG-d32-l32 (self)                                      3_428.33     3_254.24     6_682.58       0.9812          1.0548            1.0000       214.01
QG-d32-l128-ef15 (query)                               4_752.71       553.19     5_305.90       0.8895         11.9727            1.0000       214.01
QG-d32-l128-ef30 (query)                               4_752.71       732.89     5_485.59       0.8973         11.7318            1.0000       214.01
QG-d32-l128-ef60 (query)                               4_752.71       999.61     5_752.32       0.9028         10.9929            1.0000       214.01
QG-d32-l128-ef120 (query)                              4_752.71     1_374.17     6_126.88       0.9102          9.4678            1.0000       214.01
QG-d32-l128 (self)                                     4_752.71     3_260.76     8_013.47       0.9028         10.8008            1.0000       214.01
QG-d64-l32-ef15 (query)                                6_274.09       989.35     7_263.44       0.9989          1.0003            1.0000       329.97
QG-d64-l32-ef30 (query)                                6_274.09     1_286.49     7_560.58       0.9994          1.0002            1.0000       329.97
QG-d64-l32-ef60 (query)                                6_274.09     1_667.74     7_941.83       0.9996          1.0002            1.0000       329.97
QG-d64-l32-ef120 (query)                               6_274.09     2_422.66     8_696.75       0.9997          1.0001            1.0000       329.97
QG-d64-l32 (self)                                      6_274.09     6_343.98    12_618.07       0.9996          1.0002            1.0000       329.97
QG-d64-l128-ef15 (query)                              10_464.78     1_134.38    11_599.16       0.9652          3.3747            1.0000       329.97
QG-d64-l128-ef30 (query)                              10_464.78     1_381.03    11_845.81       0.9770          1.5004            1.0000       329.97
QG-d64-l128-ef60 (query)                              10_464.78     1_819.72    12_284.50       0.9833          1.1644            1.0000       329.97
QG-d64-l128-ef120 (query)                             10_464.78     2_393.36    12_858.14       0.9869          1.0735            1.0000       329.97
QG-d64-l128 (self)                                    10_464.78     6_033.00    16_497.78       0.9821          1.1853            1.0000       329.97
IVF-RaBitQ-nl158-np7-rf0 (query)                       2_531.25       167.76     2_699.01       0.5890          1.0211            1.0216         4.58
IVF-RaBitQ-nl158-np12-rf0 (query)                      2_531.25       207.53     2_738.78       0.5890          1.0211            1.0216         4.58
IVF-RaBitQ-nl158-np17-rf0 (query)                      2_531.25       277.02     2_808.27       0.5890          1.0211            1.0216         4.58
IVF-RaBitQ-nl158-np7-rf10 (query)                      2_531.25       269.79     2_801.05       0.9852          1.0002            1.0000         4.58
IVF-RaBitQ-nl158-np7-rf20 (query)                      2_531.25       362.50     2_893.75       0.9985          1.0000            1.0000         4.58
IVF-RaBitQ-nl158-np12-rf10 (query)                     2_531.25       332.21     2_863.46       0.9852          1.0002            1.0000         4.58
IVF-RaBitQ-nl158-np12-rf20 (query)                     2_531.25       430.77     2_962.03       0.9985          1.0000            1.0000         4.58
IVF-RaBitQ-nl158-np17-rf10 (query)                     2_531.25       386.90     2_918.16       0.9852          1.0002            1.0000         4.58
IVF-RaBitQ-nl158-np17-rf20 (query)                     2_531.25       510.54     3_041.79       0.9985          1.0000            1.0000         4.58
IVF-RaBitQ-nl158 (self)                                2_531.25     1_516.02     4_047.27       0.9988          1.0000            1.0000         4.58
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_544.35       217.58     1_761.93       0.5984          1.0202            1.0206         4.90
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_544.35       248.47     1_792.82       0.5984          1.0202            1.0206         4.90
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_544.35       330.31     1_874.66       0.5984          1.0202            1.0206         4.90
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_544.35       313.99     1_858.33       0.9877          1.0001            1.0000         4.90
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_544.35       418.19     1_962.53       0.9989          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_544.35       327.38     1_871.73       0.9877          1.0001            1.0000         4.90
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_544.35       404.14     1_948.49       0.9990          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_544.35       388.71     1_933.05       0.9877          1.0001            1.0000         4.90
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_544.35       474.38     2_018.73       0.9990          1.0000            1.0000         4.90
IVF-RaBitQ-nl223 (self)                                1_544.35     1_506.78     3_051.12       0.9990          1.0000            1.0000         4.90
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_768.73       251.57     2_020.30       0.6047          1.0193            1.0198         5.35
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_768.73       258.19     2_026.92       0.6047          1.0193            1.0198         5.35
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_768.73       344.53     2_113.27       0.6047          1.0193            1.0198         5.35
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_768.73       331.50     2_100.23       0.9885          1.0001            1.0000         5.35
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_768.73       417.12     2_185.85       0.9991          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_768.73       350.36     2_119.09       0.9885          1.0001            1.0000         5.35
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_768.73       441.38     2_210.11       0.9991          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_768.73       434.10     2_202.83       0.9885          1.0001            1.0000         5.35
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_768.73       521.90     2_290.64       0.9991          1.0000            1.0000         5.35
IVF-RaBitQ-nl316 (self)                                1_768.73     1_643.56     3_412.29       0.9991          1.0000            1.0000         5.35
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
Exhaustive (query)                                        99.52     1_807.19     1_906.71       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.52     6_108.60     6_208.12       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           1_901.02       455.83     2_356.85       0.5817          1.0176            1.0178         6.16
ExhaustiveRaBitQ-rf5 (query)                           1_901.02       506.26     2_407.27       0.9265          1.0009            1.0003         6.16
ExhaustiveRaBitQ-rf10 (query)                          1_901.02       566.79     2_467.81       0.9840          1.0001            1.0000         6.16
ExhaustiveRaBitQ-rf20 (query)                          1_901.02       682.43     2_583.45       0.9985          1.0000            1.0000         6.16
ExhaustiveRaBitQ (self)                                1_901.02     1_793.37     3_694.38       0.9839          1.0001            1.0000         6.16
QG-d32-l32-ef15 (query)                                4_741.68       639.23     5_380.91       0.9246          3.3595            1.0000       311.66
QG-d32-l32-ef30 (query)                                4_741.68       867.17     5_608.85       0.9428          2.8438            1.0000       311.66
QG-d32-l32-ef60 (query)                                4_741.68     1_149.15     5_890.83       0.9586          1.6823            1.0000       311.66
QG-d32-l32-ef120 (query)                               4_741.68     1_536.85     6_278.53       0.9724          1.1755            1.0000       311.66
QG-d32-l32 (self)                                      4_741.68     3_765.38     8_507.06       0.9586          1.6771            1.0000       311.66
QG-d32-l128-ef15 (query)                               6_758.39       651.93     7_410.33       0.8816          7.7079            1.0000       311.66
QG-d32-l128-ef30 (query)                               6_758.39       887.81     7_646.20       0.8870          7.7068            1.0000       311.66
QG-d32-l128-ef60 (query)                               6_758.39     1_180.78     7_939.17       0.8896          7.3037            1.0000       311.66
QG-d32-l128-ef120 (query)                              6_758.39     1_581.10     8_339.49       0.8906          7.2914            1.0000       311.66
QG-d32-l128 (self)                                     6_758.39     3_901.76    10_660.16       0.8881          7.5360            1.0000       311.66
QG-d64-l32-ef15 (query)                                9_382.36     1_342.34    10_724.70       0.9984          1.0027            1.0000       476.46
QG-d64-l32-ef30 (query)                                9_382.36     1_681.79    11_064.15       0.9993          1.0003            1.0000       476.46
QG-d64-l32-ef60 (query)                                9_382.36     2_153.53    11_535.89       0.9995          1.0002            1.0000       476.46
QG-d64-l32-ef120 (query)                               9_382.36     2_795.44    12_177.79       0.9996          1.0002            1.0000       476.46
QG-d64-l32 (self)                                      9_382.36     7_065.79    16_448.15       0.9995          1.0002            1.0000       476.46
QG-d64-l128-ef15 (query)                              13_952.64     1_385.29    15_337.92       0.9021          6.1161            1.0000       476.46
QG-d64-l128-ef30 (query)                              13_952.64     1_774.86    15_727.49       0.9058          5.9594            1.0000       476.46
QG-d64-l128-ef60 (query)                              13_952.64     2_307.17    16_259.81       0.9115          5.5662            1.0000       476.46
QG-d64-l128-ef120 (query)                             13_952.64     2_970.72    16_923.36       0.9274          3.6844            1.0000       476.46
QG-d64-l128 (self)                                    13_952.64     7_521.04    21_473.68       0.9094          5.8040            1.0000       476.46
IVF-RaBitQ-nl158-np7-rf0 (query)                       3_436.02       193.57     3_629.59       0.5926          1.0163            1.0168         6.49
IVF-RaBitQ-nl158-np12-rf0 (query)                      3_436.02       273.06     3_709.08       0.5926          1.0163            1.0168         6.49
IVF-RaBitQ-nl158-np17-rf0 (query)                      3_436.02       351.99     3_788.01       0.5926          1.0163            1.0168         6.49
IVF-RaBitQ-nl158-np7-rf10 (query)                      3_436.02       314.08     3_750.10       0.9851          1.0001            1.0000         6.49
IVF-RaBitQ-nl158-np7-rf20 (query)                      3_436.02       429.49     3_865.51       0.9986          1.0000            1.0000         6.49
IVF-RaBitQ-nl158-np12-rf10 (query)                     3_436.02       387.55     3_823.57       0.9851          1.0001            1.0000         6.49
IVF-RaBitQ-nl158-np12-rf20 (query)                     3_436.02       506.24     3_942.26       0.9986          1.0000            1.0000         6.49
IVF-RaBitQ-nl158-np17-rf10 (query)                     3_436.02       466.46     3_902.48       0.9851          1.0001            1.0000         6.49
IVF-RaBitQ-nl158-np17-rf20 (query)                     3_436.02       579.64     4_015.67       0.9986          1.0000            1.0000         6.49
IVF-RaBitQ-nl158 (self)                                3_436.02     1_832.47     5_268.49       0.9987          1.0000            1.0000         6.49
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_963.45       265.69     2_229.14       0.5902          1.0168            1.0168         6.97
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_963.45       310.66     2_274.11       0.5902          1.0168            1.0168         6.97
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_963.45       423.59     2_387.04       0.5902          1.0168            1.0168         6.97
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_963.45       378.21     2_341.66       0.9845          1.0002            1.0000         6.97
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_963.45       518.79     2_482.24       0.9984          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_963.45       424.08     2_387.53       0.9845          1.0002            1.0000         6.97
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_963.45       535.26     2_498.71       0.9985          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_963.45       533.24     2_496.69       0.9846          1.0001            1.0000         6.97
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_963.45       643.41     2_606.86       0.9985          1.0000            1.0000         6.97
IVF-RaBitQ-nl223 (self)                                1_963.45     2_046.59     4_010.04       0.9986          1.0000            1.0000         6.97
IVF-RaBitQ-nl316-np15-rf0 (query)                      2_339.50       331.19     2_670.69       0.6029          1.0154            1.0159         7.64
IVF-RaBitQ-nl316-np17-rf0 (query)                      2_339.50       360.09     2_699.59       0.6029          1.0154            1.0159         7.64
IVF-RaBitQ-nl316-np25-rf0 (query)                      2_339.50       488.40     2_827.90       0.6029          1.0154            1.0159         7.64
IVF-RaBitQ-nl316-np15-rf10 (query)                     2_339.50       445.99     2_785.49       0.9875          1.0001            1.0000         7.64
IVF-RaBitQ-nl316-np15-rf20 (query)                     2_339.50       553.83     2_893.33       0.9988          1.0000            1.0000         7.64
IVF-RaBitQ-nl316-np17-rf10 (query)                     2_339.50       473.45     2_812.95       0.9875          1.0001            1.0000         7.64
IVF-RaBitQ-nl316-np17-rf20 (query)                     2_339.50       584.93     2_924.43       0.9988          1.0000            1.0000         7.64
IVF-RaBitQ-nl316-np25-rf10 (query)                     2_339.50       600.59     2_940.09       0.9875          1.0001            1.0000         7.64
IVF-RaBitQ-nl316-np25-rf20 (query)                     2_339.50       710.73     3_050.23       0.9988          1.0000            1.0000         7.64
IVF-RaBitQ-nl316 (self)                                2_339.50     2_264.61     4_604.11       0.9989          1.0000            1.0000         7.64
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
Exhaustive (query)                                        32.52       706.11       738.62       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.52     2_383.76     2_416.28       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             854.71       204.59     1_059.30       0.7390          1.0232            1.0221         2.56
ExhaustiveRaBitQ-rf5 (query)                             854.71       250.16     1_104.88       0.9977          1.0000            1.0000         2.56
ExhaustiveRaBitQ-rf10 (query)                            854.71       310.26     1_164.97       1.0000          1.0000            1.0000         2.56
ExhaustiveRaBitQ-rf20 (query)                            854.71       384.84     1_239.56       1.0000          1.0000            1.0000         2.56
ExhaustiveRaBitQ (self)                                  854.71       967.10     1_821.81       1.0000          1.0000            1.0000         2.56
QG-d32-l32-ef15 (query)                                1_313.38        72.52     1_385.90       0.9580          1.0015            1.0000       116.35
QG-d32-l32-ef30 (query)                                1_313.38       124.01     1_437.39       0.9950          1.0002            1.0000       116.35
QG-d32-l32-ef60 (query)                                1_313.38       228.60     1_541.98       0.9992          1.0001            1.0000       116.35
QG-d32-l32-ef120 (query)                               1_313.38       424.44     1_737.82       0.9997          1.0000            1.0000       116.35
QG-d32-l32 (self)                                      1_313.38       690.87     2_004.25       0.9992          1.0001            1.0000       116.35
QG-d32-l128-ef15 (query)                               2_224.85        79.67     2_304.52       0.9701          1.0012            1.0000       116.35
QG-d32-l128-ef30 (query)                               2_224.85       135.39     2_360.24       0.9982          1.0001            1.0000       116.35
QG-d32-l128-ef60 (query)                               2_224.85       249.94     2_474.79       0.9999          1.0000            1.0000       116.35
QG-d32-l128-ef120 (query)                              2_224.85       485.18     2_710.03       1.0000          1.0000            1.0000       116.35
QG-d32-l128 (self)                                     2_224.85       765.71     2_990.56       0.9999          1.0000            1.0000       116.35
QG-d64-l32-ef15 (query)                                1_498.74        92.39     1_591.12       0.9729          1.0008            1.0000       183.49
QG-d64-l32-ef30 (query)                                1_498.74       165.12     1_663.86       0.9982          1.0001            1.0000       183.49
QG-d64-l32-ef60 (query)                                1_498.74       305.40     1_804.14       0.9998          1.0000            1.0000       183.49
QG-d64-l32-ef120 (query)                               1_498.74       565.08     2_063.82       0.9999          1.0000            1.0000       183.49
QG-d64-l32 (self)                                      1_498.74       930.54     2_429.28       0.9998          1.0000            1.0000       183.49
QG-d64-l128-ef15 (query)                               2_836.27       109.36     2_945.63       0.9882          1.0003            1.0000       183.49
QG-d64-l128-ef30 (query)                               2_836.27       201.25     3_037.52       0.9997          1.0000            1.0000       183.49
QG-d64-l128-ef60 (query)                               2_836.27       378.79     3_215.06       1.0000          1.0000            1.0000       183.49
QG-d64-l128-ef120 (query)                              2_836.27       714.40     3_550.67       1.0000          1.0000            1.0000       183.49
QG-d64-l128 (self)                                     2_836.27     1_169.86     4_006.13       1.0000          1.0000            1.0000       183.49
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_312.79        81.22     1_394.01       0.7402          1.0230            1.0219         2.68
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_312.79       111.06     1_423.85       0.7402          1.0230            1.0219         2.68
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_312.79       147.34     1_460.13       0.7402          1.0230            1.0219         2.68
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_312.79       153.78     1_466.57       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_312.79       217.41     1_530.19       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_312.79       180.21     1_493.00       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_312.79       246.05     1_558.84       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_312.79       213.07     1_525.86       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_312.79       279.12     1_591.90       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158 (self)                                1_312.79       898.63     2_211.42       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl223-np11-rf0 (query)                        901.61       108.94     1_010.55       0.7451          1.0220            1.0210         2.84
IVF-RaBitQ-nl223-np14-rf0 (query)                        901.61       126.96     1_028.58       0.7451          1.0220            1.0210         2.84
IVF-RaBitQ-nl223-np21-rf0 (query)                        901.61       173.51     1_075.12       0.7451          1.0220            1.0210         2.84
IVF-RaBitQ-nl223-np11-rf10 (query)                       901.61       177.25     1_078.87       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np11-rf20 (query)                       901.61       237.62     1_139.24       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np14-rf10 (query)                       901.61       193.99     1_095.61       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np14-rf20 (query)                       901.61       255.74     1_157.36       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np21-rf10 (query)                       901.61       239.23     1_140.85       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np21-rf20 (query)                       901.61       299.92     1_201.53       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223 (self)                                  901.61       963.26     1_864.87       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_090.13       132.25     1_222.38       0.7480          1.0215            1.0205         3.07
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_090.13       146.43     1_236.55       0.7480          1.0215            1.0205         3.07
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_090.13       194.90     1_285.03       0.7480          1.0215            1.0205         3.07
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_090.13       203.93     1_294.06       1.0000          1.0000            1.0000         3.07
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_090.13       262.31     1_352.44       1.0000          1.0000            1.0000         3.07
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_090.13       214.57     1_304.70       1.0000          1.0000            1.0000         3.07
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_090.13       275.28     1_365.41       1.0000          1.0000            1.0000         3.07
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_090.13       263.02     1_353.15       1.0000          1.0000            1.0000         3.07
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_090.13       322.36     1_412.49       1.0000          1.0000            1.0000         3.07
IVF-RaBitQ-nl316 (self)                                1_090.13     1_050.27     2_140.40       1.0000          1.0000            1.0000         3.07
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
Exhaustive (query)                                        67.87     1_287.08     1_354.95       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         67.87     4_324.49     4_392.36       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           1_570.48       350.93     1_921.41       0.7526          1.0139            1.0132         4.36
ExhaustiveRaBitQ-rf5 (query)                           1_570.48       405.32     1_975.80       0.9982          1.0000            1.0000         4.36
ExhaustiveRaBitQ-rf10 (query)                          1_570.48       455.56     2_026.04       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ-rf20 (query)                          1_570.48       561.22     2_131.70       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ (self)                                1_570.48     1_444.61     3_015.09       1.0000          1.0000            1.0000         4.36
QG-d32-l32-ef15 (query)                                2_928.87       130.97     3_059.83       0.9488          1.0024            1.0000       214.01
QG-d32-l32-ef30 (query)                                2_928.87       217.70     3_146.57       0.9885          1.0007            1.0000       214.01
QG-d32-l32-ef60 (query)                                2_928.87       381.85     3_310.72       0.9971          1.0003            1.0000       214.01
QG-d32-l32-ef120 (query)                               2_928.87       673.49     3_602.36       0.9988          1.0002            1.0000       214.01
QG-d32-l32 (self)                                      2_928.87     1_170.73     4_099.59       0.9971          1.0003            1.0000       214.01
QG-d32-l128-ef15 (query)                               4_621.51       143.23     4_764.74       0.9383          6.8721            1.0000       214.01
QG-d32-l128-ef30 (query)                               4_621.51       243.20     4_864.71       0.9898          1.0037            1.0000       214.01
QG-d32-l128-ef60 (query)                               4_621.51       419.34     5_040.85       0.9977          1.0009            1.0000       214.01
QG-d32-l128-ef120 (query)                              4_621.51       726.38     5_347.89       0.9989          1.0004            1.0000       214.01
QG-d32-l128 (self)                                     4_621.51     1_252.43     5_873.94       0.9975          1.0010            1.0000       214.01
QG-d64-l32-ef15 (query)                                3_964.49       175.81     4_140.30       0.9799          1.0005            1.0000       329.97
QG-d64-l32-ef30 (query)                                3_964.49       307.46     4_271.95       0.9983          1.0001            1.0000       329.97
QG-d64-l32-ef60 (query)                                3_964.49       543.92     4_508.41       0.9997          1.0000            1.0000       329.97
QG-d64-l32-ef120 (query)                               3_964.49       957.90     4_922.38       0.9998          1.0000            1.0000       329.97
QG-d64-l32 (self)                                      3_964.49     1_684.40     5_648.89       0.9997          1.0000            1.0000       329.97
QG-d64-l128-ef15 (query)                               7_266.14       225.64     7_491.78       0.9909          1.0003            1.0000       329.97
QG-d64-l128-ef30 (query)                               7_266.14       370.83     7_636.97       0.9995          1.0000            1.0000       329.97
QG-d64-l128-ef60 (query)                               7_266.14       655.28     7_921.41       0.9999          1.0000            1.0000       329.97
QG-d64-l128-ef120 (query)                              7_266.14     1_169.73     8_435.87       0.9999          1.0000            1.0000       329.97
QG-d64-l128 (self)                                     7_266.14     2_045.32     9_311.46       0.9999          1.0000            1.0000       329.97
IVF-RaBitQ-nl158-np7-rf0 (query)                       2_398.17       143.07     2_541.23       0.7552          1.0136            1.0129         4.58
IVF-RaBitQ-nl158-np12-rf0 (query)                      2_398.17       199.72     2_597.89       0.7552          1.0136            1.0129         4.58
IVF-RaBitQ-nl158-np17-rf0 (query)                      2_398.17       257.80     2_655.96       0.7552          1.0136            1.0129         4.58
IVF-RaBitQ-nl158-np7-rf10 (query)                      2_398.17       245.94     2_644.10       1.0000          1.0000            1.0000         4.58
IVF-RaBitQ-nl158-np7-rf20 (query)                      2_398.17       349.12     2_747.29       1.0000          1.0000            1.0000         4.58
IVF-RaBitQ-nl158-np12-rf10 (query)                     2_398.17       294.88     2_693.05       1.0000          1.0000            1.0000         4.58
IVF-RaBitQ-nl158-np12-rf20 (query)                     2_398.17       387.78     2_785.95       1.0000          1.0000            1.0000         4.58
IVF-RaBitQ-nl158-np17-rf10 (query)                     2_398.17       351.35     2_749.52       1.0000          1.0000            1.0000         4.58
IVF-RaBitQ-nl158-np17-rf20 (query)                     2_398.17       443.78     2_841.95       1.0000          1.0000            1.0000         4.58
IVF-RaBitQ-nl158 (self)                                2_398.17     1_404.75     3_802.92       1.0000          1.0000            1.0000         4.58
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_538.32       195.28     1_733.60       0.7564          1.0134            1.0127         4.91
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_538.32       225.09     1_763.42       0.7565          1.0134            1.0127         4.91
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_538.32       308.35     1_846.67       0.7565          1.0134            1.0127         4.91
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_538.32       292.44     1_830.76       0.9994          1.0000            1.0000         4.91
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_538.32       383.60     1_921.92       0.9994          1.0000            1.0000         4.91
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_538.32       321.90     1_860.22       1.0000          1.0000            1.0000         4.91
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_538.32       415.10     1_953.42       1.0000          1.0000            1.0000         4.91
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_538.32       402.81     1_941.13       1.0000          1.0000            1.0000         4.91
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_538.32       494.54     2_032.87       1.0000          1.0000            1.0000         4.91
IVF-RaBitQ-nl223 (self)                                1_538.32     1_547.55     3_085.87       1.0000          1.0000            1.0000         4.91
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_957.94       237.24     2_195.19       0.7585          1.0132            1.0125         5.35
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_957.94       260.92     2_218.86       0.7585          1.0132            1.0125         5.35
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_957.94       350.09     2_308.04       0.7585          1.0132            1.0125         5.35
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_957.94       339.68     2_297.62       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_957.94       429.63     2_387.57       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_957.94       356.79     2_314.73       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_957.94       459.75     2_417.70       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_957.94       440.95     2_398.90       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_957.94       531.63     2_489.57       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316 (self)                                1_957.94     1_686.57     3_644.52       1.0000          1.0000            1.0000         5.35
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
Exhaustive (query)                                       101.46     1_776.52     1_877.98       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.46     5_944.88     6_046.34       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           2_066.34       488.54     2_554.88       0.7336          1.0115            1.0110         6.16
ExhaustiveRaBitQ-rf5 (query)                           2_066.34       565.48     2_631.82       0.9966          1.0000            1.0000         6.16
ExhaustiveRaBitQ-rf10 (query)                          2_066.34       634.36     2_700.70       1.0000          1.0000            1.0000         6.16
ExhaustiveRaBitQ-rf20 (query)                          2_066.34       778.95     2_845.29       1.0000          1.0000            1.0000         6.16
ExhaustiveRaBitQ (self)                                2_066.34     2_029.42     4_095.76       1.0000          1.0000            1.0000         6.16
QG-d32-l32-ef15 (query)                                4_554.65       230.11     4_784.77       0.9370          1.0133            1.0000       311.66
QG-d32-l32-ef30 (query)                                4_554.65       391.23     4_945.88       0.9780          1.0035            1.0000       311.66
QG-d32-l32-ef60 (query)                                4_554.65       636.91     5_191.56       0.9912          1.0016            1.0000       311.66
QG-d32-l32-ef120 (query)                               4_554.65     1_031.96     5_586.61       0.9953          1.0011            1.0000       311.66
QG-d32-l32 (self)                                      4_554.65     2_060.40     6_615.05       0.9913          1.0017            1.0000       311.66
QG-d32-l128-ef15 (query)                               6_837.55       187.17     7_024.72       0.5571        288.0829            1.0035       311.66
QG-d32-l128-ef30 (query)                               6_837.55       319.13     7_156.68       0.6256        251.7763            1.0000       311.66
QG-d32-l128-ef60 (query)                               6_837.55       586.02     7_423.57       0.7206        187.5431            1.0000       311.66
QG-d32-l128-ef120 (query)                              6_837.55     1_131.61     7_969.16       0.8861         56.9982            1.0000       311.66
QG-d32-l128 (self)                                     6_837.55     1_856.39     8_693.94       0.7184        211.0852            1.0000       311.66
QG-d64-l32-ef15 (query)                                7_912.73       398.01     8_310.74       0.9891          1.0004            1.0000       476.46
QG-d64-l32-ef30 (query)                                7_912.73       671.77     8_584.50       0.9986          1.0001            1.0000       476.46
QG-d64-l32-ef60 (query)                                7_912.73     1_091.99     9_004.72       0.9996          1.0000            1.0000       476.46
QG-d64-l32-ef120 (query)                               7_912.73     1_725.28     9_638.01       0.9998          1.0000            1.0000       476.46
QG-d64-l32 (self)                                      7_912.73     3_698.70    11_611.42       0.9996          1.0000            1.0000       476.46
QG-d64-l128-ef15 (query)                              13_523.50       472.62    13_996.12       0.9876          1.0042            1.0000       476.46
QG-d64-l128-ef30 (query)                              13_523.50       797.64    14_321.14       0.9969          1.0010            1.0000       476.46
QG-d64-l128-ef60 (query)                              13_523.50     1_251.32    14_774.81       0.9993          1.0002            1.0000       476.46
QG-d64-l128-ef120 (query)                             13_523.50     1_984.78    15_508.27       0.9997          1.0001            1.0000       476.46
QG-d64-l128 (self)                                    13_523.50     4_019.10    17_542.59       0.9993          1.0002            1.0000       476.46
IVF-RaBitQ-nl158-np7-rf0 (query)                       3_346.64       192.27     3_538.90       0.7361          1.0112            1.0107         6.50
IVF-RaBitQ-nl158-np12-rf0 (query)                      3_346.64       271.09     3_617.72       0.7361          1.0112            1.0107         6.50
IVF-RaBitQ-nl158-np17-rf0 (query)                      3_346.64       353.23     3_699.87       0.7361          1.0112            1.0107         6.50
IVF-RaBitQ-nl158-np7-rf10 (query)                      3_346.64       313.10     3_659.73       0.9999          1.0000            1.0000         6.50
IVF-RaBitQ-nl158-np7-rf20 (query)                      3_346.64       427.15     3_773.79       1.0000          1.0000            1.0000         6.50
IVF-RaBitQ-nl158-np12-rf10 (query)                     3_346.64       399.25     3_745.88       0.9999          1.0000            1.0000         6.50
IVF-RaBitQ-nl158-np12-rf20 (query)                     3_346.64       502.29     3_848.92       1.0000          1.0000            1.0000         6.50
IVF-RaBitQ-nl158-np17-rf10 (query)                     3_346.64       465.21     3_811.85       0.9999          1.0000            1.0000         6.50
IVF-RaBitQ-nl158-np17-rf20 (query)                     3_346.64       581.13     3_927.77       1.0000          1.0000            1.0000         6.50
IVF-RaBitQ-nl158 (self)                                3_346.64     1_842.74     5_189.38       1.0000          1.0000            1.0000         6.50
IVF-RaBitQ-nl223-np11-rf0 (query)                      2_058.21       280.73     2_338.94       0.7385          1.0110            1.0107         6.98
IVF-RaBitQ-nl223-np14-rf0 (query)                      2_058.21       311.41     2_369.62       0.7385          1.0110            1.0107         6.98
IVF-RaBitQ-nl223-np21-rf0 (query)                      2_058.21       423.26     2_481.47       0.7385          1.0110            1.0107         6.98
IVF-RaBitQ-nl223-np11-rf10 (query)                     2_058.21       382.72     2_440.93       1.0000          1.0000            1.0000         6.98
IVF-RaBitQ-nl223-np11-rf20 (query)                     2_058.21       496.17     2_554.37       1.0000          1.0000            1.0000         6.98
IVF-RaBitQ-nl223-np14-rf10 (query)                     2_058.21       425.05     2_483.26       1.0000          1.0000            1.0000         6.98
IVF-RaBitQ-nl223-np14-rf20 (query)                     2_058.21       570.66     2_628.86       1.0000          1.0000            1.0000         6.98
IVF-RaBitQ-nl223-np21-rf10 (query)                     2_058.21       549.99     2_608.20       1.0000          1.0000            1.0000         6.98
IVF-RaBitQ-nl223-np21-rf20 (query)                     2_058.21       660.41     2_718.62       1.0000          1.0000            1.0000         6.98
IVF-RaBitQ-nl223 (self)                                2_058.21     2_074.44     4_132.65       1.0000          1.0000            1.0000         6.98
IVF-RaBitQ-nl316-np15-rf0 (query)                      2_595.92       331.96     2_927.88       0.7401          1.0109            1.0104         7.66
IVF-RaBitQ-nl316-np17-rf0 (query)                      2_595.92       364.84     2_960.76       0.7401          1.0109            1.0104         7.66
IVF-RaBitQ-nl316-np25-rf0 (query)                      2_595.92       494.98     3_090.90       0.7401          1.0109            1.0104         7.66
IVF-RaBitQ-nl316-np15-rf10 (query)                     2_595.92       451.55     3_047.48       0.9999          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np15-rf20 (query)                     2_595.92       576.17     3_172.10       1.0000          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np17-rf10 (query)                     2_595.92       481.76     3_077.68       0.9999          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np17-rf20 (query)                     2_595.92       596.59     3_192.51       1.0000          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np25-rf10 (query)                     2_595.92       608.17     3_204.09       0.9999          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np25-rf20 (query)                     2_595.92       720.30     3_316.23       1.0000          1.0000            1.0000         7.66
IVF-RaBitQ-nl316 (self)                                2_595.92     2_305.24     4_901.16       1.0000          1.0000            1.0000         7.66
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
Exhaustive (query)                                        32.54       726.61       759.15       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.54     2_352.64     2_385.18       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             946.43       240.50     1_186.93       0.8711          1.0279            1.0229         2.56
ExhaustiveRaBitQ-rf5 (query)                             946.43       291.01     1_237.44       1.0000          1.0000            1.0000         2.56
ExhaustiveRaBitQ-rf10 (query)                            946.43       343.99     1_290.42       1.0000          1.0000            1.0000         2.56
ExhaustiveRaBitQ-rf20 (query)                            946.43       438.76     1_385.19       1.0000          1.0000            1.0000         2.56
ExhaustiveRaBitQ (self)                                  946.43     1_114.53     2_060.95       1.0000          1.0000            1.0000         2.56
QG-d32-l32-ef15 (query)                                  730.52        56.38       786.90       0.9859          1.0040            1.0000       116.35
QG-d32-l32-ef30 (query)                                  730.52        76.65       807.17       0.9994          1.0001            1.0000       116.35
QG-d32-l32-ef60 (query)                                  730.52       124.84       855.36       1.0000          1.0000            1.0000       116.35
QG-d32-l32-ef120 (query)                                 730.52       229.91       960.43       1.0000          1.0000            1.0000       116.35
QG-d32-l32 (self)                                        730.52       383.88     1_114.40       0.9999          1.0000            1.0000       116.35
QG-d32-l128-ef15 (query)                               1_463.14        52.14     1_515.28       0.9900          1.0005            1.0000       116.35
QG-d32-l128-ef30 (query)                               1_463.14        79.61     1_542.75       0.9999          1.0000            1.0000       116.35
QG-d32-l128-ef60 (query)                               1_463.14       138.76     1_601.90       1.0000          1.0000            1.0000       116.35
QG-d32-l128-ef120 (query)                              1_463.14       255.80     1_718.94       1.0000          1.0000            1.0000       116.35
QG-d32-l128 (self)                                     1_463.14       416.47     1_879.61       1.0000          1.0000            1.0000       116.35
QG-d64-l32-ef15 (query)                                  809.79        55.64       865.42       0.9865          1.0021            1.0000       183.49
QG-d64-l32-ef30 (query)                                  809.79        83.48       893.27       0.9993          1.0001            1.0000       183.49
QG-d64-l32-ef60 (query)                                  809.79       139.14       948.93       0.9999          1.0000            1.0000       183.49
QG-d64-l32-ef120 (query)                                 809.79       252.90     1_062.68       1.0000          1.0000            1.0000       183.49
QG-d64-l32 (self)                                        809.79       430.12     1_239.90       0.9999          1.0000            1.0000       183.49
QG-d64-l128-ef15 (query)                               1_611.13        57.06     1_668.19       0.9901          1.0005            1.0000       183.49
QG-d64-l128-ef30 (query)                               1_611.13        88.18     1_699.31       0.9999          1.0000            1.0000       183.49
QG-d64-l128-ef60 (query)                               1_611.13       150.18     1_761.31       1.0000          1.0000            1.0000       183.49
QG-d64-l128-ef120 (query)                              1_611.13       281.49     1_892.62       1.0000          1.0000            1.0000       183.49
QG-d64-l128 (self)                                     1_611.13       471.41     2_082.53       1.0000          1.0000            1.0000       183.49
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_404.00        95.36     1_499.36       0.8744          1.0268            1.0218         2.68
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_404.00       129.16     1_533.16       0.8750          1.0265            1.0215         2.68
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_404.00       169.04     1_573.04       0.8750          1.0265            1.0215         2.68
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_404.00       163.37     1_567.37       0.9976          1.0005            1.0000         2.68
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_404.00       230.14     1_634.14       0.9976          1.0005            1.0000         2.68
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_404.00       204.39     1_608.39       0.9999          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_404.00       271.38     1_675.38       0.9999          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_404.00       242.97     1_646.97       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_404.00       313.56     1_717.56       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158 (self)                                1_404.00     1_011.19     2_415.19       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl223-np11-rf0 (query)                        762.55       116.21       878.76       0.8844          1.0224            1.0182         2.83
IVF-RaBitQ-nl223-np14-rf0 (query)                        762.55       134.55       897.10       0.8845          1.0224            1.0181         2.83
IVF-RaBitQ-nl223-np21-rf0 (query)                        762.55       179.40       941.95       0.8845          1.0224            1.0181         2.83
IVF-RaBitQ-nl223-np11-rf10 (query)                       762.55       184.30       946.85       0.9994          1.0001            1.0000         2.83
IVF-RaBitQ-nl223-np11-rf20 (query)                       762.55       247.38     1_009.93       0.9994          1.0001            1.0000         2.83
IVF-RaBitQ-nl223-np14-rf10 (query)                       762.55       204.74       967.29       0.9999          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np14-rf20 (query)                       762.55       268.50     1_031.05       0.9999          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np21-rf10 (query)                       762.55       254.93     1_017.48       1.0000          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np21-rf20 (query)                       762.55       317.97     1_080.52       1.0000          1.0000            1.0000         2.83
IVF-RaBitQ-nl223 (self)                                  762.55     1_024.78     1_787.33       1.0000          1.0000            1.0000         2.83
IVF-RaBitQ-nl316-np15-rf0 (query)                        911.98       141.36     1_053.34       0.8902          1.0196            1.0162         3.06
IVF-RaBitQ-nl316-np17-rf0 (query)                        911.98       154.40     1_066.38       0.8902          1.0196            1.0162         3.06
IVF-RaBitQ-nl316-np25-rf0 (query)                        911.98       210.85     1_122.83       0.8902          1.0196            1.0162         3.06
IVF-RaBitQ-nl316-np15-rf10 (query)                       911.98       207.11     1_119.09       0.9997          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np15-rf20 (query)                       911.98       265.87     1_177.85       0.9997          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf10 (query)                       911.98       230.06     1_142.04       0.9998          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf20 (query)                       911.98       277.68     1_189.66       0.9998          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf10 (query)                       911.98       269.22     1_181.20       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf20 (query)                       911.98       333.09     1_245.07       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316 (self)                                  911.98     1_093.07     2_005.05       1.0000          1.0000            1.0000         3.06
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
Exhaustive (query)                                        70.71     1_362.30     1_433.01       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         70.71     4_636.38     4_707.10       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           1_894.50       431.87     2_326.36       0.9105          1.0126            1.0093         4.37
ExhaustiveRaBitQ-rf5 (query)                           1_894.50       485.98     2_380.48       1.0000          1.0000            1.0000         4.37
ExhaustiveRaBitQ-rf10 (query)                          1_894.50       575.00     2_469.49       1.0000          1.0000            1.0000         4.37
ExhaustiveRaBitQ-rf20 (query)                          1_894.50       680.27     2_574.76       1.0000          1.0000            1.0000         4.37
ExhaustiveRaBitQ (self)                                1_894.50     1_707.15     3_601.65       1.0000          1.0000            1.0000         4.37
QG-d32-l32-ef15 (query)                                1_550.40        86.16     1_636.56       0.9862          1.0088            1.0000       214.01
QG-d32-l32-ef30 (query)                                1_550.40       130.92     1_681.32       0.9991          1.0002            1.0000       214.01
QG-d32-l32-ef60 (query)                                1_550.40       212.73     1_763.13       0.9998          1.0000            1.0000       214.01
QG-d32-l32-ef120 (query)                               1_550.40       359.99     1_910.39       1.0000          1.0000            1.0000       214.01
QG-d32-l32 (self)                                      1_550.40       558.05     2_108.45       0.9999          1.0000            1.0000       214.01
QG-d32-l128-ef15 (query)                               2_938.71        84.04     3_022.75       0.9914          1.0006            1.0000       214.01
QG-d32-l128-ef30 (query)                               2_938.71       117.71     3_056.42       0.9999          1.0000            1.0000       214.01
QG-d32-l128-ef60 (query)                               2_938.71       214.36     3_153.07       1.0000          1.0000            1.0000       214.01
QG-d32-l128-ef120 (query)                              2_938.71       348.68     3_287.39       1.0000          1.0000            1.0000       214.01
QG-d32-l128 (self)                                     2_938.71       630.45     3_569.16       1.0000          1.0000            1.0000       214.01
QG-d64-l32-ef15 (query)                                1_802.03        93.39     1_895.42       0.9879          1.0030            1.0000       329.97
QG-d64-l32-ef30 (query)                                1_802.03       130.67     1_932.70       0.9993          1.0001            1.0000       329.97
QG-d64-l32-ef60 (query)                                1_802.03       223.89     2_025.92       0.9998          1.0000            1.0000       329.97
QG-d64-l32-ef120 (query)                               1_802.03       393.76     2_195.79       1.0000          1.0000            1.0000       329.97
QG-d64-l32 (self)                                      1_802.03       622.52     2_424.55       0.9998          1.0000            1.0000       329.97
QG-d64-l128-ef15 (query)                               3_614.80       113.92     3_728.72       0.9916          1.0004            1.0000       329.97
QG-d64-l128-ef30 (query)                               3_614.80       145.14     3_759.94       0.9999          1.0000            1.0000       329.97
QG-d64-l128-ef60 (query)                               3_614.80       219.49     3_834.29       1.0000          1.0000            1.0000       329.97
QG-d64-l128-ef120 (query)                              3_614.80       392.55     4_007.35       1.0000          1.0000            1.0000       329.97
QG-d64-l128 (self)                                     3_614.80       626.81     4_241.60       1.0000          1.0000            1.0000       329.97
IVF-RaBitQ-nl158-np7-rf0 (query)                       2_722.11       153.38     2_875.49       0.9150          1.0112            1.0083         4.59
IVF-RaBitQ-nl158-np12-rf0 (query)                      2_722.11       229.52     2_951.63       0.9157          1.0109            1.0082         4.59
IVF-RaBitQ-nl158-np17-rf0 (query)                      2_722.11       290.21     3_012.33       0.9157          1.0109            1.0082         4.59
IVF-RaBitQ-nl158-np7-rf10 (query)                      2_722.11       273.12     2_995.23       0.9986          1.0003            1.0000         4.59
IVF-RaBitQ-nl158-np7-rf20 (query)                      2_722.11       351.05     3_073.16       0.9986          1.0003            1.0000         4.59
IVF-RaBitQ-nl158-np12-rf10 (query)                     2_722.11       319.20     3_041.31       0.9999          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np12-rf20 (query)                     2_722.11       419.30     3_141.41       0.9999          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np17-rf10 (query)                     2_722.11       412.25     3_134.36       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np17-rf20 (query)                     2_722.11       481.93     3_204.04       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158 (self)                                2_722.11     1_543.34     4_265.46       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_435.38       198.73     1_634.11       0.9224          1.0091            1.0066         4.90
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_435.38       235.44     1_670.82       0.9224          1.0090            1.0066         4.90
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_435.38       322.66     1_758.04       0.9225          1.0090            1.0066         4.90
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_435.38       294.13     1_729.51       0.9997          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_435.38       390.94     1_826.32       0.9997          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_435.38       329.32     1_764.70       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_435.38       424.82     1_860.20       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_435.38       413.67     1_849.05       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_435.38       509.38     1_944.76       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223 (self)                                1_435.38     1_627.06     3_062.44       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_668.04       244.93     1_912.97       0.9276          1.0078            1.0055         5.36
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_668.04       269.57     1_937.61       0.9276          1.0078            1.0055         5.36
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_668.04       363.32     2_031.36       0.9276          1.0078            1.0055         5.36
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_668.04       339.36     2_007.40       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_668.04       432.42     2_100.46       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_668.04       360.67     2_028.72       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_668.04       452.74     2_120.78       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_668.04       455.55     2_123.60       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_668.04       550.39     2_218.43       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl316 (self)                                1_668.04     1_762.89     3_430.93       1.0000          1.0000            1.0000         5.36
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
Exhaustive (query)                                        99.64     1_828.61     1_928.25       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.64     6_092.55     6_192.19       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           2_531.02       545.52     3_076.54       0.9146          1.0115            1.0083         6.15
ExhaustiveRaBitQ-rf5 (query)                           2_531.02       624.65     3_155.67       1.0000          1.0000            1.0000         6.15
ExhaustiveRaBitQ-rf10 (query)                          2_531.02       690.61     3_221.63       1.0000          1.0000            1.0000         6.15
ExhaustiveRaBitQ-rf20 (query)                          2_531.02       826.75     3_357.77       1.0000          1.0000            1.0000         6.15
ExhaustiveRaBitQ (self)                                2_531.02     2_241.20     4_772.22       1.0000          1.0000            1.0000         6.15
QG-d32-l32-ef15 (query)                                2_143.56       103.13     2_246.68       0.9865          1.0044            1.0000       311.66
QG-d32-l32-ef30 (query)                                2_143.56       140.58     2_284.14       0.9992          1.0002            1.0000       311.66
QG-d32-l32-ef60 (query)                                2_143.56       217.02     2_360.57       0.9999          1.0000            1.0000       311.66
QG-d32-l32-ef120 (query)                               2_143.56       366.05     2_509.61       1.0000          1.0000            1.0000       311.66
QG-d32-l32 (self)                                      2_143.56       672.85     2_816.41       0.9999          1.0000            1.0000       311.66
QG-d32-l128-ef15 (query)                               4_231.43       123.31     4_354.74       0.9904          1.0009            1.0000       311.66
QG-d32-l128-ef30 (query)                               4_231.43       154.85     4_386.28       0.9999          1.0000            1.0000       311.66
QG-d32-l128-ef60 (query)                               4_231.43       225.34     4_456.77       1.0000          1.0000            1.0000       311.66
QG-d32-l128-ef120 (query)                              4_231.43       393.30     4_624.73       1.0000          1.0000            1.0000       311.66
QG-d32-l128 (self)                                     4_231.43       658.32     4_889.75       1.0000          1.0000            1.0000       311.66
QG-d64-l32-ef15 (query)                                2_412.39       113.16     2_525.55       0.9873          1.0012            1.0000       476.46
QG-d64-l32-ef30 (query)                                2_412.39       158.38     2_570.78       0.9990          1.0002            1.0000       476.46
QG-d64-l32-ef60 (query)                                2_412.39       247.52     2_659.91       0.9998          1.0000            1.0000       476.46
QG-d64-l32-ef120 (query)                               2_412.39       441.27     2_853.67       1.0000          1.0000            1.0000       476.46
QG-d64-l32 (self)                                      2_412.39       751.61     3_164.01       0.9998          1.0000            1.0000       476.46
QG-d64-l128-ef15 (query)                               4_806.83       115.63     4_922.46       0.9905          1.0006            1.0000       476.46
QG-d64-l128-ef30 (query)                               4_806.83       160.76     4_967.59       0.9999          1.0000            1.0000       476.46
QG-d64-l128-ef60 (query)                               4_806.83       258.24     5_065.07       1.0000          1.0000            1.0000       476.46
QG-d64-l128-ef120 (query)                              4_806.83       462.25     5_269.08       1.0000          1.0000            1.0000       476.46
QG-d64-l128 (self)                                     4_806.83       773.13     5_579.96       1.0000          1.0000            1.0000       476.46
IVF-RaBitQ-nl158-np7-rf0 (query)                       3_887.20       211.32     4_098.52       0.9171          1.0109            1.0079         6.49
IVF-RaBitQ-nl158-np12-rf0 (query)                      3_887.20       312.63     4_199.83       0.9173          1.0109            1.0079         6.49
IVF-RaBitQ-nl158-np17-rf0 (query)                      3_887.20       409.39     4_296.59       0.9173          1.0109            1.0079         6.49
IVF-RaBitQ-nl158-np7-rf10 (query)                      3_887.20       330.40     4_217.60       0.9995          1.0001            1.0000         6.49
IVF-RaBitQ-nl158-np7-rf20 (query)                      3_887.20       446.69     4_333.89       0.9995          1.0001            1.0000         6.49
IVF-RaBitQ-nl158-np12-rf10 (query)                     3_887.20       425.55     4_312.75       1.0000          1.0000            1.0000         6.49
IVF-RaBitQ-nl158-np12-rf20 (query)                     3_887.20       543.56     4_430.76       1.0000          1.0000            1.0000         6.49
IVF-RaBitQ-nl158-np17-rf10 (query)                     3_887.20       525.42     4_412.62       1.0000          1.0000            1.0000         6.49
IVF-RaBitQ-nl158-np17-rf20 (query)                     3_887.20       642.90     4_530.10       1.0000          1.0000            1.0000         6.49
IVF-RaBitQ-nl158 (self)                                3_887.20     2_043.43     5_930.63       1.0000          1.0000            1.0000         6.49
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_860.24       277.93     2_138.16       0.9220          1.0094            1.0067         6.96
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_860.24       334.22     2_194.45       0.9220          1.0094            1.0067         6.96
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_860.24       464.87     2_325.10       0.9220          1.0094            1.0067         6.96
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_860.24       393.66     2_253.90       0.9999          1.0000            1.0000         6.96
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_860.24       513.23     2_373.47       0.9999          1.0000            1.0000         6.96
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_860.24       444.32     2_304.56       1.0000          1.0000            1.0000         6.96
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_860.24       565.21     2_425.44       1.0000          1.0000            1.0000         6.96
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_860.24       569.22     2_429.45       1.0000          1.0000            1.0000         6.96
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_860.24       684.36     2_544.59       1.0000          1.0000            1.0000         6.96
IVF-RaBitQ-nl223 (self)                                1_860.24     2_200.05     4_060.29       1.0000          1.0000            1.0000         6.96
IVF-RaBitQ-nl316-np15-rf0 (query)                      2_273.80       341.41     2_615.21       0.9267          1.0082            1.0057         7.64
IVF-RaBitQ-nl316-np17-rf0 (query)                      2_273.80       375.76     2_649.56       0.9267          1.0082            1.0057         7.64
IVF-RaBitQ-nl316-np25-rf0 (query)                      2_273.80       513.39     2_787.20       0.9267          1.0082            1.0057         7.64
IVF-RaBitQ-nl316-np15-rf10 (query)                     2_273.80       467.61     2_741.42       1.0000          1.0000            1.0000         7.64
IVF-RaBitQ-nl316-np15-rf20 (query)                     2_273.80       575.43     2_849.23       1.0000          1.0000            1.0000         7.64
IVF-RaBitQ-nl316-np17-rf10 (query)                     2_273.80       490.90     2_764.71       1.0000          1.0000            1.0000         7.64
IVF-RaBitQ-nl316-np17-rf20 (query)                     2_273.80       609.94     2_883.74       1.0000          1.0000            1.0000         7.64
IVF-RaBitQ-nl316-np25-rf10 (query)                     2_273.80       632.40     2_906.20       1.0000          1.0000            1.0000         7.64
IVF-RaBitQ-nl316-np25-rf20 (query)                     2_273.80       748.84     3_022.64       1.0000          1.0000            1.0000         7.64
IVF-RaBitQ-nl316 (self)                                2_273.80     2_426.77     4_700.57       1.0000          1.0000            1.0000         7.64
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
Exhaustive (query)                                        32.79       692.58       725.37       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.79     2_302.47     2_335.26       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              144.43       358.45       502.88       0.0971          1.7176            1.5958         7.12
ExhaustiveTQ-b2-rf5 (query)                              144.43       438.86       583.30       0.2336          1.2025            1.2204         7.12
ExhaustiveTQ-b2-rf10 (query)                             144.43       576.06       720.49       0.2853          1.1453            1.1620         7.12
ExhaustiveTQ-b2-rf20 (query)                             144.43       958.72     1_103.15       0.3809          1.0970            1.0941         7.12
ExhaustiveTQ-b2 (self)                                   144.43     3_171.73     3_316.16       0.3814          1.0980            1.0957         7.12
ExhaustiveTQ-b4-rf0 (query)                              224.51       567.37       791.88       0.1094          1.5328            1.4997        13.22
ExhaustiveTQ-b4-rf5 (query)                              224.51       656.70       881.21       0.2368          1.1884            1.2090        13.22
ExhaustiveTQ-b4-rf10 (query)                             224.51       800.73     1_025.23       0.2884          1.1372            1.1543        13.22
ExhaustiveTQ-b4-rf20 (query)                             224.51     1_200.15     1_424.65       0.3823          1.0940            1.0970        13.22
ExhaustiveTQ-b4 (self)                                   224.51     3_917.22     4_141.73       0.3841          1.0938            1.0948        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                          886.50       106.25       992.75       0.0971          1.7176            1.5958         7.80
IVF-TQ-b2-nl158-np12-rf0 (query)                         886.50       116.16     1_002.66       0.0971          1.7176            1.5958         7.80
IVF-TQ-b2-nl158-np17-rf0 (query)                         886.50       125.02     1_011.52       0.0971          1.7176            1.5958         7.80
IVF-TQ-b2-nl158-np7-rf10 (query)                         886.50       305.70     1_192.20       0.2853          1.1453            1.1620         7.80
IVF-TQ-b2-nl158-np7-rf20 (query)                         886.50       632.45     1_518.95       0.3809          1.0970            1.0941         7.80
IVF-TQ-b2-nl158-np12-rf10 (query)                        886.50       324.80     1_211.30       0.2853          1.1453            1.1620         7.80
IVF-TQ-b2-nl158-np12-rf20 (query)                        886.50       671.88     1_558.38       0.3809          1.0970            1.0941         7.80
IVF-TQ-b2-nl158-np17-rf10 (query)                        886.50       332.12     1_218.62       0.2853          1.1453            1.1620         7.80
IVF-TQ-b2-nl158-np17-rf20 (query)                        886.50       675.68     1_562.18       0.3809          1.0970            1.0941         7.80
IVF-TQ-b2-nl158 (self)                                   886.50     1_049.61     1_936.11       0.3815          1.0980            1.0957         7.80
IVF-TQ-b2-nl223-np11-rf0 (query)                         675.40       111.47       786.87       0.0971          1.7164            1.5942         7.93
IVF-TQ-b2-nl223-np14-rf0 (query)                         675.40       118.05       793.45       0.0971          1.7176            1.5958         7.93
IVF-TQ-b2-nl223-np21-rf0 (query)                         675.40       131.88       807.28       0.0971          1.7176            1.5958         7.93
IVF-TQ-b2-nl223-np11-rf10 (query)                        675.40       283.61       959.01       0.2855          1.1450            1.1618         7.93
IVF-TQ-b2-nl223-np11-rf20 (query)                        675.40       557.20     1_232.60       0.3813          1.0967            1.0934         7.93
IVF-TQ-b2-nl223-np14-rf10 (query)                        675.40       291.49       966.89       0.2853          1.1453            1.1620         7.93
IVF-TQ-b2-nl223-np14-rf20 (query)                        675.40       576.76     1_252.16       0.3809          1.0970            1.0941         7.93
IVF-TQ-b2-nl223-np21-rf10 (query)                        675.40       308.54       983.94       0.2853          1.1453            1.1620         7.93
IVF-TQ-b2-nl223-np21-rf20 (query)                        675.40       614.40     1_289.80       0.3809          1.0970            1.0941         7.93
IVF-TQ-b2-nl223 (self)                                   675.40     1_038.00     1_713.40       0.3815          1.0980            1.0957         7.93
IVF-TQ-b2-nl316-np15-rf0 (query)                         857.93       117.89       975.83       0.0973          1.6435            1.5781         8.10
IVF-TQ-b2-nl316-np17-rf0 (query)                         857.93       120.27       978.20       0.0972          1.7163            1.5957         8.10
IVF-TQ-b2-nl316-np25-rf0 (query)                         857.93       133.58       991.52       0.0971          1.7176            1.5958         8.10
IVF-TQ-b2-nl316-np15-rf10 (query)                        857.93       285.38     1_143.32       0.2858          1.1447            1.1615         8.10
IVF-TQ-b2-nl316-np15-rf20 (query)                        857.93       547.62     1_405.56       0.3817          1.0965            1.0931         8.10
IVF-TQ-b2-nl316-np17-rf10 (query)                        857.93       292.03     1_149.96       0.2853          1.1453            1.1620         8.10
IVF-TQ-b2-nl316-np17-rf20 (query)                        857.93       562.72     1_420.65       0.3809          1.0970            1.0941         8.10
IVF-TQ-b2-nl316-np25-rf10 (query)                        857.93       307.48     1_165.42       0.2853          1.1453            1.1620         8.10
IVF-TQ-b2-nl316-np25-rf20 (query)                        857.93       586.72     1_444.65       0.3809          1.0970            1.0941         8.10
IVF-TQ-b2-nl316 (self)                                   857.93     1_052.64     1_910.58       0.3815          1.0980            1.0957         8.10
IVF-TQ-b4-nl158-np7-rf0 (query)                          968.09       144.53     1_112.61       0.1094          1.5328            1.4997        14.05
IVF-TQ-b4-nl158-np12-rf0 (query)                         968.09       171.00     1_139.09       0.1094          1.5328            1.4996        14.05
IVF-TQ-b4-nl158-np17-rf0 (query)                         968.09       194.71     1_162.79       0.1094          1.5328            1.4996        14.05
IVF-TQ-b4-nl158-np7-rf10 (query)                         968.09       358.90     1_326.99       0.2884          1.1372            1.1543        14.05
IVF-TQ-b4-nl158-np7-rf20 (query)                         968.09       691.12     1_659.20       0.3823          1.0940            1.0970        14.05
IVF-TQ-b4-nl158-np12-rf10 (query)                        968.09       420.66     1_388.75       0.2884          1.1372            1.1543        14.05
IVF-TQ-b4-nl158-np12-rf20 (query)                        968.09       773.84     1_741.93       0.3823          1.0940            1.0970        14.05
IVF-TQ-b4-nl158-np17-rf10 (query)                        968.09       391.19     1_359.28       0.2884          1.1372            1.1543        14.05
IVF-TQ-b4-nl158-np17-rf20 (query)                        968.09       747.89     1_715.98       0.3823          1.0940            1.0970        14.05
IVF-TQ-b4-nl158 (self)                                   968.09     1_185.59     2_153.67       0.3841          1.0938            1.0948        14.05
IVF-TQ-b4-nl223-np11-rf0 (query)                         745.07       152.94       898.01       0.1094          1.5317            1.4988        14.25
IVF-TQ-b4-nl223-np14-rf0 (query)                         745.07       166.39       911.45       0.1094          1.5329            1.4997        14.25
IVF-TQ-b4-nl223-np21-rf0 (query)                         745.07       187.43       932.50       0.1094          1.5328            1.4996        14.25
IVF-TQ-b4-nl223-np11-rf10 (query)                        745.07       334.64     1_079.71       0.2886          1.1370            1.1541        14.25
IVF-TQ-b4-nl223-np11-rf20 (query)                        745.07       618.42     1_363.49       0.3826          1.0939            1.0966        14.25
IVF-TQ-b4-nl223-np14-rf10 (query)                        745.07       347.95     1_093.02       0.2884          1.1372            1.1543        14.25
IVF-TQ-b4-nl223-np14-rf20 (query)                        745.07       636.23     1_381.29       0.3823          1.0940            1.0970        14.25
IVF-TQ-b4-nl223-np21-rf10 (query)                        745.07       372.96     1_118.03       0.2884          1.1372            1.1543        14.25
IVF-TQ-b4-nl223-np21-rf20 (query)                        745.07       670.69     1_415.76       0.3823          1.0940            1.0970        14.25
IVF-TQ-b4-nl223 (self)                                   745.07     1_151.79     1_896.86       0.3841          1.0938            1.0948        14.25
IVF-TQ-b4-nl316-np15-rf0 (query)                         937.90       158.57     1_096.47       0.1094          1.5304            1.4978        14.49
IVF-TQ-b4-nl316-np17-rf0 (query)                         937.90       165.86     1_103.76       0.1094          1.5328            1.4996        14.49
IVF-TQ-b4-nl316-np25-rf0 (query)                         937.90       200.62     1_138.52       0.1094          1.5328            1.4997        14.49
IVF-TQ-b4-nl316-np15-rf10 (query)                        937.90       340.08     1_277.98       0.2886          1.1369            1.1542        14.49
IVF-TQ-b4-nl316-np15-rf20 (query)                        937.90       606.37     1_544.27       0.3828          1.0938            1.0967        14.49
IVF-TQ-b4-nl316-np17-rf10 (query)                        937.90       346.83     1_284.73       0.2884          1.1372            1.1543        14.49
IVF-TQ-b4-nl316-np17-rf20 (query)                        937.90       618.26     1_556.16       0.3823          1.0940            1.0970        14.49
IVF-TQ-b4-nl316-np25-rf10 (query)                        937.90       371.17     1_309.07       0.2884          1.1372            1.1543        14.49
IVF-TQ-b4-nl316-np25-rf20 (query)                        937.90       658.44     1_596.34       0.3823          1.0940            1.0970        14.49
IVF-TQ-b4-nl316 (self)                                   937.90     1_176.77     2_114.67       0.3841          1.0938            1.0948        14.49
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
Exhaustive (query)                                        68.26     1_353.01     1_421.27       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.26     4_538.68     4_606.94       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              338.26       657.88       996.14       0.1207          1.3711            1.3320        13.97
ExhaustiveTQ-b2-rf5 (query)                              338.26       740.78     1_079.04       0.2421          1.1334            1.1574        13.97
ExhaustiveTQ-b2-rf10 (query)                             338.26       880.02     1_218.28       0.2934          1.0981            1.1177        13.97
ExhaustiveTQ-b2-rf20 (query)                             338.26     1_275.53     1_613.79       0.3880          1.0664            1.0469        13.97
ExhaustiveTQ-b2 (self)                                   338.26     4_168.17     4_506.43       0.3879          1.0667            1.0471        13.97
ExhaustiveTQ-b4-rf0 (query)                              465.76     1_139.30     1_605.06       0.1315          1.3172            1.3127        26.18
ExhaustiveTQ-b4-rf5 (query)                              465.76     1_249.68     1_715.44       0.2471          1.1254            1.1483        26.18
ExhaustiveTQ-b4-rf10 (query)                             465.76     1_408.71     1_874.47       0.2970          1.0929            1.0980        26.18
ExhaustiveTQ-b4-rf20 (query)                             465.76     1_773.26     2_239.02       0.3883          1.0643            1.0492        26.18
ExhaustiveTQ-b4 (self)                                   465.76     5_838.87     6_304.63       0.3881          1.0646            1.0495        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_955.03       192.19     2_147.22       0.1207          1.3711            1.3320        14.95
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_955.03       209.66     2_164.70       0.1207          1.3711            1.3320        14.95
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_955.03       222.12     2_177.16       0.1207          1.3711            1.3320        14.95
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_955.03       413.72     2_368.75       0.2934          1.0981            1.1177        14.95
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_955.03       761.03     2_716.06       0.3880          1.0664            1.0469        14.95
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_955.03       434.85     2_389.88       0.2934          1.0981            1.1178        14.95
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_955.03       810.67     2_765.70       0.3880          1.0664            1.0469        14.95
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_955.03       450.46     2_405.49       0.2934          1.0981            1.1177        14.95
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_955.03       817.21     2_772.24       0.3880          1.0664            1.0469        14.95
IVF-TQ-b2-nl158 (self)                                 1_955.03     1_414.97     3_370.00       0.3879          1.0667            1.0471        14.95
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_215.06       205.38     1_420.43       0.1208          1.3699            1.3300        15.19
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_215.06       216.30     1_431.36       0.1207          1.3711            1.3320        15.19
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_215.06       237.43     1_452.49       0.1207          1.3711            1.3320        15.19
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_215.06       405.37     1_620.43       0.2937          1.0979            1.1176        15.19
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_215.06       710.71     1_925.76       0.3887          1.0662            1.0467        15.19
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_215.06       417.23     1_632.28       0.2934          1.0981            1.1177        15.19
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_215.06       733.09     1_948.14       0.3880          1.0664            1.0469        15.19
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_215.06       465.43     1_680.49       0.2934          1.0981            1.1178        15.19
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_215.06       775.79     1_990.85       0.3880          1.0664            1.0469        15.19
IVF-TQ-b2-nl223 (self)                                 1_215.06     1_434.30     2_649.36       0.3879          1.0667            1.0471        15.19
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_526.13       215.50     1_741.63       0.1208          1.3689            1.3287        15.56
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_526.13       219.30     1_745.43       0.1208          1.3707            1.3312        15.56
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_526.13       240.44     1_766.57       0.1207          1.3711            1.3320        15.56
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_526.13       403.51     1_929.64       0.2939          1.0977            1.1175        15.56
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_526.13       687.09     2_213.22       0.3892          1.0660            1.0465        15.56
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_526.13       415.14     1_941.28       0.2935          1.0980            1.1177        15.56
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_526.13       705.86     2_231.99       0.3882          1.0664            1.0469        15.56
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_526.13       442.33     1_968.46       0.2934          1.0981            1.1178        15.56
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_526.13       761.72     2_287.86       0.3880          1.0664            1.0469        15.56
IVF-TQ-b2-nl316 (self)                                 1_526.13     1_459.77     2_985.90       0.3879          1.0667            1.0471        15.56
IVF-TQ-b4-nl158-np7-rf0 (query)                        2_012.66       271.08     2_283.74       0.1315          1.3172            1.3127        27.44
IVF-TQ-b4-nl158-np12-rf0 (query)                       2_012.66       297.82     2_310.48       0.1315          1.3172            1.3127        27.44
IVF-TQ-b4-nl158-np17-rf0 (query)                       2_012.66       324.50     2_337.16       0.1315          1.3172            1.3127        27.44
IVF-TQ-b4-nl158-np7-rf10 (query)                       2_012.66       505.52     2_518.18       0.2970          1.0929            1.0979        27.44
IVF-TQ-b4-nl158-np7-rf20 (query)                       2_012.66       861.82     2_874.47       0.3883          1.0643            1.0492        27.44
IVF-TQ-b4-nl158-np12-rf10 (query)                      2_012.66       548.35     2_561.01       0.2970          1.0929            1.0979        27.44
IVF-TQ-b4-nl158-np12-rf20 (query)                      2_012.66       919.23     2_931.89       0.3882          1.0643            1.0492        27.44
IVF-TQ-b4-nl158-np17-rf10 (query)                      2_012.66       563.47     2_576.13       0.2970          1.0929            1.0979        27.44
IVF-TQ-b4-nl158-np17-rf20 (query)                      2_012.66       934.56     2_947.22       0.3883          1.0643            1.0492        27.44
IVF-TQ-b4-nl158 (self)                                 2_012.66     1_604.62     3_617.27       0.3881          1.0646            1.0495        27.44
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_326.93       292.52     1_619.44       0.1315          1.3158            1.3116        27.79
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_326.93       311.15     1_638.07       0.1315          1.3172            1.3127        27.79
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_326.93       344.52     1_671.44       0.1315          1.3172            1.3127        27.79
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_326.93       509.87     1_836.79       0.2973          1.0926            1.0973        27.79
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_326.93       812.42     2_139.35       0.3889          1.0641            1.0489        27.79
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_326.93       522.09     1_849.02       0.2970          1.0929            1.0980        27.79
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_326.93       837.41     2_164.34       0.3883          1.0643            1.0492        27.79
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_326.93       562.11     1_889.04       0.2970          1.0929            1.0980        27.79
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_326.93       901.20     2_228.12       0.3883          1.0643            1.0492        27.79
IVF-TQ-b4-nl223 (self)                                 1_326.93     1_649.95     2_976.88       0.3881          1.0646            1.0495        27.79
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_657.82       301.45     1_959.27       0.1316          1.3151            1.3108        28.35
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_657.82       313.64     1_971.46       0.1315          1.3164            1.3121        28.35
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_657.82       352.76     2_010.58       0.1315          1.3172            1.3127        28.35
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_657.82       503.41     2_161.23       0.2976          1.0925            1.0966        28.35
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_657.82       797.52     2_455.34       0.3893          1.0639            1.0484        28.35
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_657.82       519.47     2_177.29       0.2971          1.0928            1.0977        28.35
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_657.82       820.42     2_478.24       0.3885          1.0642            1.0491        28.35
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_657.82       575.99     2_233.81       0.2970          1.0929            1.0980        28.35
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_657.82       883.89     2_541.71       0.3883          1.0643            1.0492        28.35
IVF-TQ-b4-nl316 (self)                                 1_657.82     1_689.41     3_347.23       0.3881          1.0646            1.0495        28.35
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
Exhaustive (query)                                       100.66     1_904.66     2_005.32       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.66     6_377.31     6_477.97       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              605.17       947.23     1_552.41       0.1292          1.2710            1.2627        21.33
ExhaustiveTQ-b2-rf5 (query)                              605.17     1_059.93     1_665.11       0.2468          1.1062            1.1332        21.33
ExhaustiveTQ-b2-rf10 (query)                             605.17     1_200.07     1_805.24       0.3000          1.0773            1.0631        21.33
ExhaustiveTQ-b2-rf20 (query)                             605.17     1_614.90     2_220.07       0.3957          1.0509            1.0334        21.33
ExhaustiveTQ-b2 (self)                                   605.17     5_283.25     5_888.42       0.3973          1.0507            1.0331        21.33
ExhaustiveTQ-b4-rf0 (query)                              751.81     1_751.84     2_503.65       0.1340          1.2532            1.2592        39.64
ExhaustiveTQ-b4-rf5 (query)                              751.81     1_848.33     2_600.14       0.2401          1.1136            1.1402        39.64
ExhaustiveTQ-b4-rf10 (query)                             751.81     1_999.42     2_751.23       0.2870          1.0888            1.1143        39.64
ExhaustiveTQ-b4-rf20 (query)                             751.81     2_485.80     3_237.61       0.3752          1.0657            1.0812        39.64
ExhaustiveTQ-b4 (self)                                   751.81     7_958.42     8_710.22       0.3767          1.0654            1.0638        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        2_779.72       291.45     3_071.18       0.1292          1.2710            1.2627        22.66
IVF-TQ-b2-nl158-np12-rf0 (query)                       2_779.72       323.04     3_102.76       0.1292          1.2710            1.2627        22.66
IVF-TQ-b2-nl158-np17-rf0 (query)                       2_779.72       334.38     3_114.10       0.1292          1.2710            1.2627        22.66
IVF-TQ-b2-nl158-np7-rf10 (query)                       2_779.72       529.26     3_308.98       0.3000          1.0773            1.0631        22.66
IVF-TQ-b2-nl158-np7-rf20 (query)                       2_779.72       893.10     3_672.82       0.3957          1.0509            1.0334        22.66
IVF-TQ-b2-nl158-np12-rf10 (query)                      2_779.72       560.72     3_340.45       0.3000          1.0773            1.0631        22.66
IVF-TQ-b2-nl158-np12-rf20 (query)                      2_779.72       941.41     3_721.13       0.3957          1.0509            1.0334        22.66
IVF-TQ-b2-nl158-np17-rf10 (query)                      2_779.72       580.07     3_359.79       0.3000          1.0774            1.0631        22.66
IVF-TQ-b2-nl158-np17-rf20 (query)                      2_779.72       967.76     3_747.49       0.3957          1.0509            1.0334        22.66
IVF-TQ-b2-nl158 (self)                                 2_779.72     1_842.89     4_622.62       0.3973          1.0507            1.0331        22.66
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_858.81       307.11     2_165.93       0.1292          1.2710            1.2627        23.04
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_858.81       323.25     2_182.07       0.1292          1.2710            1.2627        23.04
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_858.81       370.70     2_229.51       0.1292          1.2710            1.2627        23.04
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_858.81       536.35     2_395.16       0.3000          1.0774            1.0631        23.04
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_858.81       853.67     2_712.49       0.3957          1.0509            1.0334        23.04
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_858.81       550.87     2_409.68       0.3000          1.0774            1.0631        23.04
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_858.81       878.84     2_737.66       0.3957          1.0509            1.0334        23.04
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_858.81       587.99     2_446.80       0.3000          1.0774            1.0631        23.04
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_858.81       930.68     2_789.49       0.3957          1.0509            1.0334        23.04
IVF-TQ-b2-nl223 (self)                                 1_858.81     1_877.02     3_735.84       0.3973          1.0507            1.0331        23.04
IVF-TQ-b2-nl316-np15-rf0 (query)                       2_244.96       320.16     2_565.11       0.1292          1.2709            1.2627        23.57
IVF-TQ-b2-nl316-np17-rf0 (query)                       2_244.96       329.47     2_574.43       0.1292          1.2710            1.2628        23.57
IVF-TQ-b2-nl316-np25-rf0 (query)                       2_244.96       357.89     2_602.85       0.1292          1.2710            1.2628        23.57
IVF-TQ-b2-nl316-np15-rf10 (query)                      2_244.96       541.70     2_786.66       0.3000          1.0773            1.0631        23.57
IVF-TQ-b2-nl316-np15-rf20 (query)                      2_244.96       856.66     3_101.62       0.3957          1.0509            1.0334        23.57
IVF-TQ-b2-nl316-np17-rf10 (query)                      2_244.96       551.67     2_796.62       0.3000          1.0774            1.0632        23.57
IVF-TQ-b2-nl316-np17-rf20 (query)                      2_244.96       926.19     3_171.15       0.3957          1.0509            1.0334        23.57
IVF-TQ-b2-nl316-np25-rf10 (query)                      2_244.96       589.00     2_833.96       0.3000          1.0774            1.0631        23.57
IVF-TQ-b2-nl316-np25-rf20 (query)                      2_244.96       920.34     3_165.30       0.3957          1.0509            1.0334        23.57
IVF-TQ-b2-nl316 (self)                                 2_244.96     1_880.32     4_125.28       0.3973          1.0507            1.0331        23.57
IVF-TQ-b4-nl158-np7-rf0 (query)                        2_821.83       412.74     3_234.57       0.1340          1.2532            1.2592        41.46
IVF-TQ-b4-nl158-np12-rf0 (query)                       2_821.83       474.21     3_296.04       0.1340          1.2532            1.2592        41.46
IVF-TQ-b4-nl158-np17-rf0 (query)                       2_821.83       495.60     3_317.43       0.1340          1.2532            1.2592        41.46
IVF-TQ-b4-nl158-np7-rf10 (query)                       2_821.83       666.84     3_488.67       0.2870          1.0888            1.1143        41.46
IVF-TQ-b4-nl158-np7-rf20 (query)                       2_821.83     1_037.55     3_859.38       0.3752          1.0657            1.0812        41.46
IVF-TQ-b4-nl158-np12-rf10 (query)                      2_821.83       715.79     3_537.61       0.2870          1.0888            1.1143        41.46
IVF-TQ-b4-nl158-np12-rf20 (query)                      2_821.83     1_104.80     3_926.63       0.3752          1.0657            1.0812        41.46
IVF-TQ-b4-nl158-np17-rf10 (query)                      2_821.83       754.99     3_576.81       0.2870          1.0888            1.1143        41.46
IVF-TQ-b4-nl158-np17-rf20 (query)                      2_821.83     1_157.69     3_979.52       0.3752          1.0657            1.0812        41.46
IVF-TQ-b4-nl158 (self)                                 2_821.83     2_146.45     4_968.28       0.3767          1.0654            1.0637        41.46
IVF-TQ-b4-nl223-np11-rf0 (query)                       2_011.37       447.59     2_458.96       0.1340          1.2532            1.2592        42.04
IVF-TQ-b4-nl223-np14-rf0 (query)                       2_011.37       477.27     2_488.64       0.1340          1.2532            1.2592        42.04
IVF-TQ-b4-nl223-np21-rf0 (query)                       2_011.37       529.18     2_540.55       0.1340          1.2532            1.2592        42.04
IVF-TQ-b4-nl223-np11-rf10 (query)                      2_011.37       685.10     2_696.47       0.2870          1.0888            1.1143        42.04
IVF-TQ-b4-nl223-np11-rf20 (query)                      2_011.37     1_005.93     3_017.30       0.3752          1.0657            1.0812        42.04
IVF-TQ-b4-nl223-np14-rf10 (query)                      2_011.37       712.76     2_724.13       0.2870          1.0888            1.1143        42.04
IVF-TQ-b4-nl223-np14-rf20 (query)                      2_011.37     1_048.28     3_059.64       0.3752          1.0657            1.0812        42.04
IVF-TQ-b4-nl223-np21-rf10 (query)                      2_011.37       773.77     2_785.13       0.2870          1.0888            1.1143        42.04
IVF-TQ-b4-nl223-np21-rf20 (query)                      2_011.37     1_133.88     3_145.24       0.3752          1.0657            1.0812        42.04
IVF-TQ-b4-nl223 (self)                                 2_011.37     2_193.28     4_204.64       0.3766          1.0654            1.0638        42.04
IVF-TQ-b4-nl316-np15-rf0 (query)                       2_387.14       492.02     2_879.16       0.1340          1.2531            1.2592        42.81
IVF-TQ-b4-nl316-np17-rf0 (query)                       2_387.14       485.07     2_872.21       0.1340          1.2531            1.2592        42.81
IVF-TQ-b4-nl316-np25-rf0 (query)                       2_387.14       537.48     2_924.62       0.1340          1.2532            1.2592        42.81
IVF-TQ-b4-nl316-np15-rf10 (query)                      2_387.14       693.01     3_080.15       0.2870          1.0888            1.1143        42.81
IVF-TQ-b4-nl316-np15-rf20 (query)                      2_387.14     1_013.09     3_400.23       0.3753          1.0657            1.0812        42.81
IVF-TQ-b4-nl316-np17-rf10 (query)                      2_387.14       706.94     3_094.08       0.2870          1.0888            1.1143        42.81
IVF-TQ-b4-nl316-np17-rf20 (query)                      2_387.14     1_034.63     3_421.77       0.3752          1.0657            1.0812        42.81
IVF-TQ-b4-nl316-np25-rf10 (query)                      2_387.14       772.04     3_159.18       0.2870          1.0888            1.1143        42.81
IVF-TQ-b4-nl316-np25-rf20 (query)                      2_387.14     1_108.83     3_495.97       0.3752          1.0657            1.0812        42.81
IVF-TQ-b4-nl316 (self)                                 2_387.14     2_253.70     4_640.84       0.3767          1.0654            1.0637        42.81
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
Exhaustive (query)                                        33.06       709.65       742.71       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.06     2_368.93     2_401.99       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              167.56       366.52       534.08       0.0756          2.3283            1.9295         7.12
ExhaustiveTQ-b2-rf5 (query)                              167.56       438.16       605.72       0.2072          1.3307            1.3578         7.12
ExhaustiveTQ-b2-rf10 (query)                             167.56       575.26       742.82       0.2886          1.2206            1.2322         7.12
ExhaustiveTQ-b2-rf20 (query)                             167.56       956.18     1_123.73       0.4151          1.1328            1.1147         7.12
ExhaustiveTQ-b2 (self)                                   167.56     3_158.89     3_326.44       0.4136          1.1619            1.1367         7.12
ExhaustiveTQ-b4-rf0 (query)                              230.45       579.66       810.11       0.1023          1.7129            1.7532        13.22
ExhaustiveTQ-b4-rf5 (query)                              230.45       665.50       895.95       0.2385          1.2770            1.3000        13.22
ExhaustiveTQ-b4-rf10 (query)                             230.45       798.66     1_029.11       0.3202          1.1874            1.1953        13.22
ExhaustiveTQ-b4-rf20 (query)                             230.45     1_192.66     1_423.11       0.4481          1.1142            1.1029        13.22
ExhaustiveTQ-b4 (self)                                   230.45     3_900.25     4_130.70       0.4463          1.1397            1.1286        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                          997.22       102.43     1_099.65       0.0756          2.3282            1.9295         7.81
IVF-TQ-b2-nl158-np12-rf0 (query)                         997.22       113.51     1_110.73       0.0756          2.3283            1.9295         7.81
IVF-TQ-b2-nl158-np17-rf0 (query)                         997.22       135.10     1_132.32       0.0756          2.3283            1.9295         7.81
IVF-TQ-b2-nl158-np7-rf10 (query)                         997.22       302.37     1_299.59       0.2886          1.2206            1.2322         7.81
IVF-TQ-b2-nl158-np7-rf20 (query)                         997.22       629.33     1_626.55       0.4151          1.1328            1.1147         7.81
IVF-TQ-b2-nl158-np12-rf10 (query)                        997.22       333.15     1_330.37       0.2886          1.2206            1.2322         7.81
IVF-TQ-b2-nl158-np12-rf20 (query)                        997.22       656.68     1_653.90       0.4151          1.1328            1.1147         7.81
IVF-TQ-b2-nl158-np17-rf10 (query)                        997.22       349.13     1_346.35       0.2886          1.2206            1.2322         7.81
IVF-TQ-b2-nl158-np17-rf20 (query)                        997.22       738.24     1_735.46       0.4151          1.1328            1.1147         7.81
IVF-TQ-b2-nl158 (self)                                   997.22     1_080.93     2_078.15       0.4136          1.1619            1.1367         7.81
IVF-TQ-b2-nl223-np11-rf0 (query)                         711.77       109.03       820.80       0.0756          2.3256            1.9254         7.94
IVF-TQ-b2-nl223-np14-rf0 (query)                         711.77       116.01       827.78       0.0756          2.3281            1.9295         7.94
IVF-TQ-b2-nl223-np21-rf0 (query)                         711.77       145.72       857.50       0.0756          2.3282            1.9295         7.94
IVF-TQ-b2-nl223-np11-rf10 (query)                        711.77       281.03       992.81       0.2890          1.2203            1.2319         7.94
IVF-TQ-b2-nl223-np11-rf20 (query)                        711.77       563.94     1_275.71       0.4156          1.1326            1.1144         7.94
IVF-TQ-b2-nl223-np14-rf10 (query)                        711.77       292.66     1_004.43       0.2886          1.2206            1.2322         7.94
IVF-TQ-b2-nl223-np14-rf20 (query)                        711.77       582.81     1_294.58       0.4151          1.1328            1.1147         7.94
IVF-TQ-b2-nl223-np21-rf10 (query)                        711.77       335.55     1_047.32       0.2886          1.2206            1.2322         7.94
IVF-TQ-b2-nl223-np21-rf20 (query)                        711.77       647.58     1_359.35       0.4151          1.1328            1.1147         7.94
IVF-TQ-b2-nl223 (self)                                   711.77     1_071.97     1_783.74       0.4136          1.1619            1.1367         7.94
IVF-TQ-b2-nl316-np15-rf0 (query)                         928.64       113.77     1_042.41       0.0757          2.3274            1.9289         8.11
IVF-TQ-b2-nl316-np17-rf0 (query)                         928.64       119.62     1_048.26       0.0756          2.3282            1.9294         8.11
IVF-TQ-b2-nl316-np25-rf0 (query)                         928.64       138.43     1_067.07       0.0756          2.3282            1.9295         8.11
IVF-TQ-b2-nl316-np15-rf10 (query)                        928.64       279.36     1_208.00       0.2892          1.2202            1.2317         8.11
IVF-TQ-b2-nl316-np15-rf20 (query)                        928.64       537.00     1_465.65       0.4159          1.1325            1.1142         8.11
IVF-TQ-b2-nl316-np17-rf10 (query)                        928.64       284.33     1_212.98       0.2887          1.2206            1.2322         8.11
IVF-TQ-b2-nl316-np17-rf20 (query)                        928.64       545.66     1_474.30       0.4151          1.1328            1.1147         8.11
IVF-TQ-b2-nl316-np25-rf10 (query)                        928.64       314.23     1_242.88       0.2886          1.2206            1.2322         8.11
IVF-TQ-b2-nl316-np25-rf20 (query)                        928.64       590.58     1_519.23       0.4151          1.1328            1.1147         8.11
IVF-TQ-b2-nl316 (self)                                   928.64     1_063.27     1_991.92       0.4136          1.1619            1.1367         8.11
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_076.62       140.26     1_216.87       0.1023          1.7129            1.7532        14.06
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_076.62       159.42     1_236.04       0.1023          1.7129            1.7532        14.06
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_076.62       193.62     1_270.23       0.1023          1.7129            1.7532        14.06
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_076.62       352.19     1_428.81       0.3202          1.1874            1.1953        14.06
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_076.62       683.01     1_759.62       0.4481          1.1142            1.1029        14.06
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_076.62       372.16     1_448.77       0.3202          1.1874            1.1953        14.06
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_076.62       717.68     1_794.29       0.4481          1.1142            1.1029        14.06
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_076.62       423.22     1_499.83       0.3202          1.1873            1.1953        14.06
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_076.62       859.84     1_936.45       0.4481          1.1142            1.1029        14.06
IVF-TQ-b4-nl158 (self)                                 1_076.62     1_116.25     2_192.87       0.4463          1.1397            1.1286        14.06
IVF-TQ-b4-nl223-np11-rf0 (query)                         805.81       149.63       955.43       0.1023          1.7109            1.7520        14.27
IVF-TQ-b4-nl223-np14-rf0 (query)                         805.81       161.02       966.83       0.1023          1.7129            1.7532        14.27
IVF-TQ-b4-nl223-np21-rf0 (query)                         805.81       202.43     1_008.24       0.1023          1.7129            1.7532        14.27
IVF-TQ-b4-nl223-np11-rf10 (query)                        805.81       334.17     1_139.97       0.3205          1.1871            1.1949        14.27
IVF-TQ-b4-nl223-np11-rf20 (query)                        805.81       623.78     1_429.59       0.4486          1.1140            1.1028        14.27
IVF-TQ-b4-nl223-np14-rf10 (query)                        805.81       358.63     1_164.44       0.3202          1.1873            1.1953        14.27
IVF-TQ-b4-nl223-np14-rf20 (query)                        805.81       643.50     1_449.30       0.4481          1.1142            1.1029        14.27
IVF-TQ-b4-nl223-np21-rf10 (query)                        805.81       404.11     1_209.92       0.3201          1.1874            1.1953        14.27
IVF-TQ-b4-nl223-np21-rf20 (query)                        805.81       727.89     1_533.70       0.4481          1.1142            1.1029        14.27
IVF-TQ-b4-nl223 (self)                                   805.81     1_112.77     1_918.58       0.4463          1.1397            1.1286        14.27
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_004.90       154.61     1_159.50       0.1023          1.7112            1.7520        14.52
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_004.90       161.77     1_166.67       0.1023          1.7121            1.7528        14.52
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_004.90       207.40     1_212.29       0.1023          1.7129            1.7532        14.52
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_004.90       328.92     1_333.81       0.3207          1.1869            1.1949        14.52
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_004.90       588.09     1_592.99       0.4491          1.1138            1.1025        14.52
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_004.90       337.36     1_342.25       0.3203          1.1873            1.1953        14.52
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_004.90       600.30     1_605.19       0.4482          1.1142            1.1029        14.52
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_004.90       379.35     1_384.25       0.3202          1.1873            1.1953        14.52
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_004.90       656.42     1_661.32       0.4481          1.1142            1.1029        14.52
IVF-TQ-b4-nl316 (self)                                 1_004.90     1_106.76     2_111.66       0.4463          1.1397            1.1286        14.52
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
Exhaustive (query)                                        68.01     1_324.20     1_392.20       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.01     4_445.60     4_513.60       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              341.43       641.92       983.35       0.0844          1.6539            1.5906        13.97
ExhaustiveTQ-b2-rf5 (query)                              341.43       727.84     1_069.27       0.2173          1.2230            1.2549        13.97
ExhaustiveTQ-b2-rf10 (query)                             341.43       872.03     1_213.46       0.2887          1.1550            1.1707        13.97
ExhaustiveTQ-b2-rf20 (query)                             341.43     1_274.86     1_616.29       0.4020          1.0974            1.0847        13.97
ExhaustiveTQ-b2 (self)                                   341.43     4_187.57     4_529.00       0.4025          1.1135            1.0971        13.97
ExhaustiveTQ-b4-rf0 (query)                              448.38     1_131.68     1_580.06       0.1044          1.5026            1.5346        26.18
ExhaustiveTQ-b4-rf5 (query)                              448.38     1_237.80     1_686.18       0.2294          1.2110            1.2410        26.18
ExhaustiveTQ-b4-rf10 (query)                             448.38     1_401.80     1_850.18       0.2943          1.1499            1.1675        26.18
ExhaustiveTQ-b4-rf20 (query)                             448.38     1_757.20     2_205.58       0.4029          1.0975            1.0929        26.18
ExhaustiveTQ-b4 (self)                                   448.38     5_812.32     6_260.70       0.4038          1.1130            1.1087        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_857.35       187.71     2_045.06       0.0844          1.6539            1.5906        14.95
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_857.35       202.96     2_060.30       0.0844          1.6539            1.5906        14.95
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_857.35       219.04     2_076.38       0.0844          1.6539            1.5906        14.95
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_857.35       414.91     2_272.25       0.2887          1.1550            1.1707        14.95
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_857.35       788.08     2_645.43       0.4020          1.0974            1.0847        14.95
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_857.35       435.00     2_292.35       0.2887          1.1550            1.1707        14.95
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_857.35       787.21     2_644.55       0.4020          1.0974            1.0847        14.95
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_857.35       449.80     2_307.14       0.2887          1.1550            1.1707        14.95
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_857.35       817.93     2_675.28       0.4020          1.0974            1.0847        14.95
IVF-TQ-b2-nl158 (self)                                 1_857.35     1_429.43     3_286.78       0.4025          1.1135            1.0971        14.95
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_303.65       201.62     1_505.26       0.0845          1.6537            1.5905        15.23
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_303.65       212.86     1_516.50       0.0844          1.6539            1.5906        15.23
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_303.65       241.24     1_544.89       0.0844          1.6539            1.5906        15.23
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_303.65       411.57     1_715.22       0.2887          1.1550            1.1707        15.23
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_303.65       712.02     2_015.67       0.4020          1.0974            1.0847        15.23
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_303.65       425.60     1_729.25       0.2887          1.1550            1.1707        15.23
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_303.65       755.23     2_058.88       0.4020          1.0974            1.0847        15.23
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_303.65       459.89     1_763.54       0.2887          1.1550            1.1707        15.23
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_303.65       796.22     2_099.86       0.4020          1.0974            1.0847        15.23
IVF-TQ-b2-nl223 (self)                                 1_303.65     1_450.06     2_753.71       0.4025          1.1135            1.0971        15.23
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_710.29       208.36     1_918.65       0.0844          1.6539            1.5906        15.57
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_710.29       214.79     1_925.08       0.0844          1.6539            1.5906        15.57
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_710.29       241.56     1_951.85       0.0844          1.6539            1.5906        15.57
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_710.29       406.36     2_116.65       0.2887          1.1550            1.1707        15.57
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_710.29       692.84     2_403.13       0.4020          1.0974            1.0847        15.57
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_710.29       411.01     2_121.30       0.2887          1.1550            1.1707        15.57
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_710.29       704.90     2_415.19       0.4020          1.0974            1.0847        15.57
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_710.29       443.88     2_154.17       0.2887          1.1550            1.1707        15.57
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_710.29       750.88     2_461.17       0.4020          1.0974            1.0847        15.57
IVF-TQ-b2-nl316 (self)                                 1_710.29     1_421.77     3_132.06       0.4025          1.1135            1.0971        15.57
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_940.08       258.69     2_198.76       0.1044          1.5026            1.5346        27.44
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_940.08       291.79     2_231.87       0.1044          1.5026            1.5346        27.44
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_940.08       316.91     2_256.99       0.1044          1.5026            1.5346        27.44
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_940.08       498.15     2_438.22       0.2943          1.1499            1.1675        27.44
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_940.08       845.49     2_785.57       0.4029          1.0975            1.0929        27.44
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_940.08       527.67     2_467.75       0.2943          1.1499            1.1675        27.44
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_940.08       891.93     2_832.01       0.4029          1.0975            1.0929        27.44
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_940.08       557.70     2_497.78       0.2943          1.1499            1.1675        27.44
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_940.08       932.99     2_873.07       0.4029          1.0975            1.0929        27.44
IVF-TQ-b4-nl158 (self)                                 1_940.08     1_585.64     3_525.71       0.4038          1.1130            1.1088        27.44
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_426.01       282.66     1_708.66       0.1044          1.5026            1.5346        27.87
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_426.01       302.06     1_728.06       0.1044          1.5026            1.5346        27.87
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_426.01       352.59     1_778.60       0.1044          1.5026            1.5346        27.87
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_426.01       506.32     1_932.33       0.2943          1.1499            1.1676        27.87
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_426.01       813.42     2_239.43       0.4029          1.0975            1.0929        27.87
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_426.01       529.95     1_955.96       0.2943          1.1499            1.1676        27.87
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_426.01       859.88     2_285.88       0.4029          1.0975            1.0929        27.87
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_426.01       586.23     2_012.23       0.2943          1.1499            1.1676        27.87
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_426.01       942.96     2_368.97       0.4029          1.0975            1.0929        27.87
IVF-TQ-b4-nl223 (self)                                 1_426.01     1_632.38     3_058.39       0.4038          1.1130            1.1088        27.87
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_823.98       292.87     2_116.84       0.1045          1.5026            1.5346        28.38
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_823.98       304.60     2_128.57       0.1044          1.5026            1.5346        28.38
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_823.98       349.88     2_173.86       0.1044          1.5026            1.5346        28.38
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_823.98       503.63     2_327.60       0.2943          1.1499            1.1675        28.38
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_823.98       793.01     2_616.99       0.4029          1.0975            1.0929        28.38
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_823.98       516.67     2_340.65       0.2943          1.1499            1.1675        28.38
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_823.98       812.15     2_636.12       0.4029          1.0975            1.0929        28.38
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_823.98       565.72     2_389.70       0.2943          1.1499            1.1675        28.38
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_823.98       875.77     2_699.74       0.4029          1.0975            1.0929        28.38
IVF-TQ-b4-nl316 (self)                                 1_823.98     1_657.78     3_481.75       0.4038          1.1130            1.1087        28.38
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
Exhaustive (query)                                       100.16     1_883.09     1_983.26       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.16     6_282.47     6_382.64       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              602.15       966.49     1_568.65       0.0841          1.5107            1.4226        21.33
ExhaustiveTQ-b2-rf5 (query)                              602.15     1_022.34     1_624.49       0.2144          1.1739            1.2056        21.33
ExhaustiveTQ-b2-rf10 (query)                             602.15     1_183.52     1_785.67       0.2770          1.1267            1.1512        21.33
ExhaustiveTQ-b2-rf20 (query)                             602.15     1_613.97     2_216.12       0.3770          1.0843            1.0724        21.33
ExhaustiveTQ-b2 (self)                                   602.15     5_278.64     5_880.79       0.3767          1.0935            1.0803        21.33
ExhaustiveTQ-b4-rf0 (query)                              751.99     1_752.46     2_504.45       0.0986          1.4231            1.4109        39.64
ExhaustiveTQ-b4-rf5 (query)                              751.99     1_869.80     2_621.79       0.2167          1.1746            1.2047        39.64
ExhaustiveTQ-b4-rf10 (query)                             751.99     2_005.92     2_757.92       0.2692          1.1311            1.1557        39.64
ExhaustiveTQ-b4-rf20 (query)                             751.99     2_401.98     3_153.97       0.3605          1.0923            1.1071        39.64
ExhaustiveTQ-b4 (self)                                   751.99     8_092.79     8_844.78       0.3609          1.1024            1.1182        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        2_796.80       286.51     3_083.31       0.0841          1.5107            1.4226        22.62
IVF-TQ-b2-nl158-np12-rf0 (query)                       2_796.80       307.81     3_104.61       0.0841          1.5107            1.4226        22.62
IVF-TQ-b2-nl158-np17-rf0 (query)                       2_796.80       333.96     3_130.76       0.0841          1.5107            1.4226        22.62
IVF-TQ-b2-nl158-np7-rf10 (query)                       2_796.80       522.99     3_319.79       0.2770          1.1267            1.1512        22.62
IVF-TQ-b2-nl158-np7-rf20 (query)                       2_796.80       899.72     3_696.52       0.3770          1.0843            1.0724        22.62
IVF-TQ-b2-nl158-np12-rf10 (query)                      2_796.80       541.32     3_338.12       0.2770          1.1267            1.1512        22.62
IVF-TQ-b2-nl158-np12-rf20 (query)                      2_796.80       960.93     3_757.73       0.3770          1.0843            1.0724        22.62
IVF-TQ-b2-nl158-np17-rf10 (query)                      2_796.80       569.24     3_366.04       0.2771          1.1267            1.1512        22.62
IVF-TQ-b2-nl158-np17-rf20 (query)                      2_796.80       952.15     3_748.95       0.3770          1.0843            1.0724        22.62
IVF-TQ-b2-nl158 (self)                                 2_796.80     1_867.30     4_664.10       0.3767          1.0935            1.0803        22.62
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_875.62       301.75     2_177.37       0.0841          1.5107            1.4226        22.97
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_875.62       316.24     2_191.86       0.0841          1.5107            1.4226        22.97
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_875.62       344.97     2_220.59       0.0841          1.5107            1.4226        22.97
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_875.62       540.55     2_416.17       0.2771          1.1267            1.1512        22.97
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_875.62       878.21     2_753.83       0.3770          1.0843            1.0724        22.97
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_875.62       557.19     2_432.81       0.2771          1.1267            1.1512        22.97
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_875.62       898.64     2_774.26       0.3770          1.0843            1.0724        22.97
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_875.62       594.49     2_470.11       0.2771          1.1267            1.1512        22.97
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_875.62       954.82     2_830.44       0.3770          1.0843            1.0724        22.97
IVF-TQ-b2-nl223 (self)                                 1_875.62     1_905.71     3_781.33       0.3767          1.0935            1.0803        22.97
IVF-TQ-b2-nl316-np15-rf0 (query)                       2_511.98       314.88     2_826.87       0.0841          1.5107            1.4226        23.53
IVF-TQ-b2-nl316-np17-rf0 (query)                       2_511.98       337.99     2_849.98       0.0841          1.5107            1.4226        23.53
IVF-TQ-b2-nl316-np25-rf0 (query)                       2_511.98       361.84     2_873.82       0.0841          1.5107            1.4226        23.53
IVF-TQ-b2-nl316-np15-rf10 (query)                      2_511.98       553.82     3_065.80       0.2771          1.1267            1.1512        23.53
IVF-TQ-b2-nl316-np15-rf20 (query)                      2_511.98       859.01     3_371.00       0.3770          1.0843            1.0724        23.53
IVF-TQ-b2-nl316-np17-rf10 (query)                      2_511.98       556.75     3_068.74       0.2771          1.1267            1.1512        23.53
IVF-TQ-b2-nl316-np17-rf20 (query)                      2_511.98       879.18     3_391.16       0.3770          1.0843            1.0724        23.53
IVF-TQ-b2-nl316-np25-rf10 (query)                      2_511.98       640.14     3_152.13       0.2771          1.1267            1.1512        23.53
IVF-TQ-b2-nl316-np25-rf20 (query)                      2_511.98       987.39     3_499.37       0.3770          1.0843            1.0724        23.53
IVF-TQ-b2-nl316 (self)                                 2_511.98     1_955.36     4_467.34       0.3767          1.0935            1.0803        23.53
IVF-TQ-b4-nl158-np7-rf0 (query)                        2_834.02       407.01     3_241.03       0.0986          1.4231            1.4109        41.39
IVF-TQ-b4-nl158-np12-rf0 (query)                       2_834.02       444.14     3_278.16       0.0986          1.4231            1.4109        41.39
IVF-TQ-b4-nl158-np17-rf0 (query)                       2_834.02       496.16     3_330.17       0.0986          1.4231            1.4109        41.39
IVF-TQ-b4-nl158-np7-rf10 (query)                       2_834.02       652.79     3_486.81       0.2692          1.1311            1.1557        41.39
IVF-TQ-b4-nl158-np7-rf20 (query)                       2_834.02     1_039.94     3_873.96       0.3605          1.0923            1.1071        41.39
IVF-TQ-b4-nl158-np12-rf10 (query)                      2_834.02       702.85     3_536.87       0.2692          1.1311            1.1557        41.39
IVF-TQ-b4-nl158-np12-rf20 (query)                      2_834.02     1_090.13     3_924.14       0.3605          1.0923            1.1071        41.39
IVF-TQ-b4-nl158-np17-rf10 (query)                      2_834.02       740.72     3_574.74       0.2692          1.1311            1.1557        41.39
IVF-TQ-b4-nl158-np17-rf20 (query)                      2_834.02     1_128.14     3_962.16       0.3605          1.0923            1.1071        41.39
IVF-TQ-b4-nl158 (self)                                 2_834.02     2_203.67     5_037.69       0.3609          1.1024            1.1182        41.39
IVF-TQ-b4-nl223-np11-rf0 (query)                       2_039.19       435.96     2_475.15       0.0986          1.4231            1.4109        41.89
IVF-TQ-b4-nl223-np14-rf0 (query)                       2_039.19       462.95     2_502.14       0.0986          1.4231            1.4109        41.89
IVF-TQ-b4-nl223-np21-rf0 (query)                       2_039.19       518.85     2_558.04       0.0986          1.4231            1.4109        41.89
IVF-TQ-b4-nl223-np11-rf10 (query)                      2_039.19       679.61     2_718.80       0.2692          1.1311            1.1557        41.89
IVF-TQ-b4-nl223-np11-rf20 (query)                      2_039.19     1_016.35     3_055.54       0.3605          1.0923            1.1071        41.89
IVF-TQ-b4-nl223-np14-rf10 (query)                      2_039.19       719.92     2_759.11       0.2692          1.1311            1.1557        41.89
IVF-TQ-b4-nl223-np14-rf20 (query)                      2_039.19     1_080.16     3_119.35       0.3605          1.0923            1.1071        41.89
IVF-TQ-b4-nl223-np21-rf10 (query)                      2_039.19       783.75     2_822.94       0.2692          1.1311            1.1557        41.89
IVF-TQ-b4-nl223-np21-rf20 (query)                      2_039.19     1_156.08     3_195.27       0.3605          1.0923            1.1071        41.89
IVF-TQ-b4-nl223 (self)                                 2_039.19     2_282.66     4_321.85       0.3609          1.1024            1.1182        41.89
IVF-TQ-b4-nl316-np15-rf0 (query)                       2_656.88       456.20     3_113.08       0.0986          1.4231            1.4109        42.73
IVF-TQ-b4-nl316-np17-rf0 (query)                       2_656.88       473.63     3_130.51       0.0986          1.4231            1.4109        42.73
IVF-TQ-b4-nl316-np25-rf0 (query)                       2_656.88       531.26     3_188.14       0.0986          1.4231            1.4109        42.73
IVF-TQ-b4-nl316-np15-rf10 (query)                      2_656.88       707.30     3_364.18       0.2692          1.1311            1.1557        42.73
IVF-TQ-b4-nl316-np15-rf20 (query)                      2_656.88     1_029.25     3_686.13       0.3605          1.0923            1.1071        42.73
IVF-TQ-b4-nl316-np17-rf10 (query)                      2_656.88       722.58     3_379.47       0.2692          1.1311            1.1557        42.73
IVF-TQ-b4-nl316-np17-rf20 (query)                      2_656.88     1_047.24     3_704.13       0.3605          1.0923            1.1071        42.73
IVF-TQ-b4-nl316-np25-rf10 (query)                      2_656.88       804.21     3_461.10       0.2692          1.1311            1.1557        42.73
IVF-TQ-b4-nl316-np25-rf20 (query)                      2_656.88     1_118.02     3_774.91       0.3605          1.0923            1.1071        42.73
IVF-TQ-b4-nl316 (self)                                 2_656.88     2_353.05     5_009.94       0.3609          1.1024            1.1182        42.73
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
Exhaustive (query)                                        32.62       725.33       757.95       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.62     2_473.80     2_506.42       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              145.07       368.88       513.95       0.7918          1.0898            1.0632         7.12
ExhaustiveTQ-b2-rf5 (query)                              145.07       448.56       593.63       0.9995          1.0000            1.0000         7.12
ExhaustiveTQ-b2-rf10 (query)                             145.07       583.30       728.37       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b2-rf20 (query)                             145.07       977.14     1_122.20       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b2 (self)                                   145.07     3_227.01     3_372.08       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b4-rf0 (query)                              230.31       588.84       819.15       0.8728          1.0322            1.0183        13.22
ExhaustiveTQ-b4-rf5 (query)                              230.31       715.87       946.17       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4-rf10 (query)                             230.31       808.77     1_039.07       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4-rf20 (query)                             230.31     1_195.28     1_425.59       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4 (self)                                   230.31     3_959.26     4_189.56       1.0000          1.0000            1.0000        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_061.61       127.58     1_189.19       0.7916          1.0897            1.0635         7.78
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_061.61       172.60     1_234.21       0.7918          1.0898            1.0632         7.78
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_061.61       208.95     1_270.56       0.7918          1.0898            1.0632         7.78
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_061.61       334.29     1_395.90       0.9982          1.0004            1.0000         7.78
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_061.61       625.66     1_687.27       0.9982          1.0004            1.0000         7.78
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_061.61       416.82     1_478.43       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_061.61       731.39     1_793.00       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_061.61       447.11     1_508.71       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_061.61       796.59     1_858.19       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158 (self)                                 1_061.61     1_200.17     2_261.77       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl223-np11-rf0 (query)                         622.58       127.47       750.05       0.7919          1.0897            1.0632         7.93
IVF-TQ-b2-nl223-np14-rf0 (query)                         622.58       145.81       768.39       0.7918          1.0897            1.0632         7.93
IVF-TQ-b2-nl223-np21-rf0 (query)                         622.58       181.81       804.39       0.7918          1.0898            1.0632         7.93
IVF-TQ-b2-nl223-np11-rf10 (query)                        622.58       321.44       944.03       0.9995          1.0001            1.0000         7.93
IVF-TQ-b2-nl223-np11-rf20 (query)                        622.58       594.76     1_217.34       0.9995          1.0001            1.0000         7.93
IVF-TQ-b2-nl223-np14-rf10 (query)                        622.58       347.85       970.43       0.9999          1.0000            1.0000         7.93
IVF-TQ-b2-nl223-np14-rf20 (query)                        622.58       639.66     1_262.24       0.9999          1.0000            1.0000         7.93
IVF-TQ-b2-nl223-np21-rf10 (query)                        622.58       399.86     1_022.44       1.0000          1.0000            1.0000         7.93
IVF-TQ-b2-nl223-np21-rf20 (query)                        622.58       729.72     1_352.31       1.0000          1.0000            1.0000         7.93
IVF-TQ-b2-nl223 (self)                                   622.58     1_054.85     1_677.43       1.0000          1.0000            1.0000         7.93
IVF-TQ-b2-nl316-np15-rf0 (query)                         817.37       129.68       947.05       0.7918          1.0897            1.0632         8.12
IVF-TQ-b2-nl316-np17-rf0 (query)                         817.37       144.43       961.79       0.7918          1.0898            1.0632         8.12
IVF-TQ-b2-nl316-np25-rf0 (query)                         817.37       168.87       986.24       0.7918          1.0898            1.0632         8.12
IVF-TQ-b2-nl316-np15-rf10 (query)                        817.37       310.85     1_128.22       0.9997          1.0000            1.0000         8.12
IVF-TQ-b2-nl316-np15-rf20 (query)                        817.37       579.05     1_396.42       0.9997          1.0000            1.0000         8.12
IVF-TQ-b2-nl316-np17-rf10 (query)                        817.37       320.69     1_138.06       0.9999          1.0000            1.0000         8.12
IVF-TQ-b2-nl316-np17-rf20 (query)                        817.37       598.50     1_415.87       0.9999          1.0000            1.0000         8.12
IVF-TQ-b2-nl316-np25-rf10 (query)                        817.37       370.59     1_187.95       1.0000          1.0000            1.0000         8.12
IVF-TQ-b2-nl316-np25-rf20 (query)                        817.37       670.80     1_488.17       1.0000          1.0000            1.0000         8.12
IVF-TQ-b2-nl316 (self)                                   817.37       999.78     1_817.15       1.0000          1.0000            1.0000         8.12
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_147.47       181.84     1_329.31       0.8721          1.0325            1.0187        14.02
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_147.47       255.31     1_402.78       0.8728          1.0322            1.0183        14.02
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_147.47       320.50     1_467.97       0.8728          1.0322            1.0183        14.02
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_147.47       392.41     1_539.88       0.9982          1.0004            1.0000        14.02
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_147.47       683.16     1_830.62       0.9982          1.0004            1.0000        14.02
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_147.47       486.80     1_634.27       1.0000          1.0000            1.0000        14.02
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_147.47       814.74     1_962.21       1.0000          1.0000            1.0000        14.02
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_147.47       567.83     1_715.30       1.0000          1.0000            1.0000        14.02
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_147.47       928.20     2_075.67       1.0000          1.0000            1.0000        14.02
IVF-TQ-b4-nl158 (self)                                 1_147.47     1_348.57     2_496.04       1.0000          1.0000            1.0000        14.02
IVF-TQ-b4-nl223-np11-rf0 (query)                         701.46       178.18       879.64       0.8726          1.0323            1.0184        14.23
IVF-TQ-b4-nl223-np14-rf0 (query)                         701.46       209.18       910.64       0.8727          1.0322            1.0183        14.23
IVF-TQ-b4-nl223-np21-rf0 (query)                         701.46       269.46       970.92       0.8728          1.0322            1.0183        14.23
IVF-TQ-b4-nl223-np11-rf10 (query)                        701.46       374.54     1_076.00       0.9995          1.0001            1.0000        14.23
IVF-TQ-b4-nl223-np11-rf20 (query)                        701.46       649.76     1_351.22       0.9995          1.0001            1.0000        14.23
IVF-TQ-b4-nl223-np14-rf10 (query)                        701.46       412.72     1_114.18       0.9999          1.0000            1.0000        14.23
IVF-TQ-b4-nl223-np14-rf20 (query)                        701.46       706.92     1_408.38       0.9999          1.0000            1.0000        14.23
IVF-TQ-b4-nl223-np21-rf10 (query)                        701.46       490.34     1_191.80       1.0000          1.0000            1.0000        14.23
IVF-TQ-b4-nl223-np21-rf20 (query)                        701.46       818.97     1_520.43       1.0000          1.0000            1.0000        14.23
IVF-TQ-b4-nl223 (self)                                   701.46     1_168.37     1_869.83       1.0000          1.0000            1.0000        14.23
IVF-TQ-b4-nl316-np15-rf0 (query)                         894.36       179.27     1_073.63       0.8727          1.0322            1.0184        14.54
IVF-TQ-b4-nl316-np17-rf0 (query)                         894.36       195.38     1_089.74       0.8727          1.0322            1.0183        14.54
IVF-TQ-b4-nl316-np25-rf0 (query)                         894.36       248.76     1_143.12       0.8727          1.0322            1.0183        14.54
IVF-TQ-b4-nl316-np15-rf10 (query)                        894.36       364.01     1_258.37       0.9997          1.0000            1.0000        14.54
IVF-TQ-b4-nl316-np15-rf20 (query)                        894.36       634.93     1_529.29       0.9997          1.0000            1.0000        14.54
IVF-TQ-b4-nl316-np17-rf10 (query)                        894.36       381.97     1_276.33       0.9999          1.0000            1.0000        14.54
IVF-TQ-b4-nl316-np17-rf20 (query)                        894.36       662.54     1_556.90       0.9999          1.0000            1.0000        14.54
IVF-TQ-b4-nl316-np25-rf10 (query)                        894.36       450.15     1_344.51       1.0000          1.0000            1.0000        14.54
IVF-TQ-b4-nl316-np25-rf20 (query)                        894.36       758.41     1_652.76       1.0000          1.0000            1.0000        14.54
IVF-TQ-b4-nl316 (self)                                   894.36     1_114.13     2_008.49       1.0000          1.0000            1.0000        14.54
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
Exhaustive (query)                                        68.63     1_325.92     1_394.55       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.63     4_433.30     4_501.93       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              337.16       631.08       968.24       0.8424          1.0447            1.0331        13.97
ExhaustiveTQ-b2-rf5 (query)                              337.16       742.20     1_079.37       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2-rf10 (query)                             337.16       866.39     1_203.55       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2-rf20 (query)                             337.16     1_296.06     1_633.22       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2 (self)                                   337.16     4_221.47     4_558.63       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b4-rf0 (query)                              448.18     1_121.48     1_569.66       0.8985          1.0191            1.0110        26.18
ExhaustiveTQ-b4-rf5 (query)                              448.18     1_231.02     1_679.20       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4-rf10 (query)                             448.18     1_368.38     1_816.55       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4-rf20 (query)                             448.18     1_767.86     2_216.04       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4 (self)                                   448.18     5_826.94     6_275.12       1.0000          1.0000            1.0000        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        2_097.84       227.48     2_325.32       0.8420          1.0449            1.0333        14.96
IVF-TQ-b2-nl158-np12-rf0 (query)                       2_097.84       296.77     2_394.61       0.8424          1.0447            1.0331        14.96
IVF-TQ-b2-nl158-np17-rf0 (query)                       2_097.84       356.51     2_454.35       0.8424          1.0447            1.0331        14.96
IVF-TQ-b2-nl158-np7-rf10 (query)                       2_097.84       458.52     2_556.37       0.9986          1.0003            1.0000        14.96
IVF-TQ-b2-nl158-np7-rf20 (query)                       2_097.84       776.96     2_874.80       0.9986          1.0003            1.0000        14.96
IVF-TQ-b2-nl158-np12-rf10 (query)                      2_097.84       542.57     2_640.41       0.9999          1.0000            1.0000        14.96
IVF-TQ-b2-nl158-np12-rf20 (query)                      2_097.84       897.55     2_995.39       0.9999          1.0000            1.0000        14.96
IVF-TQ-b2-nl158-np17-rf10 (query)                      2_097.84       611.26     2_709.11       1.0000          1.0000            1.0000        14.96
IVF-TQ-b2-nl158-np17-rf20 (query)                      2_097.84       977.20     3_075.04       1.0000          1.0000            1.0000        14.96
IVF-TQ-b2-nl158 (self)                                 2_097.84     1_515.95     3_613.80       1.0000          1.0000            1.0000        14.96
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_128.71       231.13     1_359.84       0.8423          1.0447            1.0331        15.25
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_128.71       260.36     1_389.06       0.8424          1.0447            1.0331        15.25
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_128.71       323.56     1_452.27       0.8424          1.0447            1.0331        15.25
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_128.71       446.88     1_575.59       0.9997          1.0000            1.0000        15.25
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_128.71       750.63     1_879.33       0.9997          1.0000            1.0000        15.25
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_128.71       478.93     1_607.64       0.9999          1.0000            1.0000        15.25
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_128.71       787.41     1_916.11       0.9999          1.0000            1.0000        15.25
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_128.71       550.84     1_679.54       1.0000          1.0000            1.0000        15.25
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_128.71       892.37     2_021.08       1.0000          1.0000            1.0000        15.25
IVF-TQ-b2-nl223 (self)                                 1_128.71     1_434.51     2_563.22       1.0000          1.0000            1.0000        15.25
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_329.49       235.50     1_565.00       0.8424          1.0447            1.0331        15.56
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_329.49       249.55     1_579.04       0.8424          1.0447            1.0331        15.56
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_329.49       300.05     1_629.54       0.8424          1.0447            1.0331        15.56
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_329.49       446.29     1_775.78       0.9999          1.0000            1.0000        15.56
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_329.49       742.13     2_071.62       0.9999          1.0000            1.0000        15.56
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_329.49       460.73     1_790.22       1.0000          1.0000            1.0000        15.56
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_329.49       809.35     2_138.85       1.0000          1.0000            1.0000        15.56
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_329.49       524.44     1_853.93       1.0000          1.0000            1.0000        15.56
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_329.49       853.02     2_182.52       1.0000          1.0000            1.0000        15.56
IVF-TQ-b2-nl316 (self)                                 1_329.49     1_411.83     2_741.32       1.0000          1.0000            1.0000        15.56
IVF-TQ-b4-nl158-np7-rf0 (query)                        2_157.96       333.34     2_491.30       0.8977          1.0194            1.0113        27.46
IVF-TQ-b4-nl158-np12-rf0 (query)                       2_157.96       460.89     2_618.85       0.8985          1.0191            1.0110        27.46
IVF-TQ-b4-nl158-np17-rf0 (query)                       2_157.96       567.89     2_725.85       0.8985          1.0191            1.0110        27.46
IVF-TQ-b4-nl158-np7-rf10 (query)                       2_157.96       567.13     2_725.09       0.9986          1.0003            1.0000        27.46
IVF-TQ-b4-nl158-np7-rf20 (query)                       2_157.96       904.67     3_062.63       0.9986          1.0003            1.0000        27.46
IVF-TQ-b4-nl158-np12-rf10 (query)                      2_157.96       716.26     2_874.22       0.9999          1.0000            1.0000        27.46
IVF-TQ-b4-nl158-np12-rf20 (query)                      2_157.96     1_060.25     3_218.21       0.9999          1.0000            1.0000        27.46
IVF-TQ-b4-nl158-np17-rf10 (query)                      2_157.96       820.92     2_978.88       1.0000          1.0000            1.0000        27.46
IVF-TQ-b4-nl158-np17-rf20 (query)                      2_157.96     1_192.60     3_350.56       1.0000          1.0000            1.0000        27.46
IVF-TQ-b4-nl158 (self)                                 2_157.96     1_854.96     4_012.92       1.0000          1.0000            1.0000        27.46
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_244.15       341.09     1_585.24       0.8984          1.0191            1.0111        27.91
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_244.15       395.05     1_639.19       0.8985          1.0191            1.0110        27.91
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_244.15       504.05     1_748.20       0.8985          1.0191            1.0110        27.91
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_244.15       558.51     1_802.66       0.9997          1.0000            1.0000        27.91
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_244.15       854.34     2_098.48       0.9997          1.0000            1.0000        27.91
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_244.15       614.44     1_858.59       0.9999          1.0000            1.0000        27.91
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_244.15       927.47     2_171.61       0.9999          1.0000            1.0000        27.91
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_244.15       740.41     1_984.55       1.0000          1.0000            1.0000        27.91
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_244.15     1_064.77     2_308.91       1.0000          1.0000            1.0000        27.91
IVF-TQ-b4-nl223 (self)                                 1_244.15     1_718.91     2_963.06       1.0000          1.0000            1.0000        27.91
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_475.14       341.31     1_816.45       0.8985          1.0191            1.0110        28.36
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_475.14       366.17     1_841.31       0.8985          1.0191            1.0110        28.36
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_475.14       462.02     1_937.17       0.8985          1.0191            1.0110        28.36
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_475.14       554.28     2_029.43       0.9999          1.0000            1.0000        28.36
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_475.14       857.25     2_332.39       0.9999          1.0000            1.0000        28.36
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_475.14       590.69     2_065.83       1.0000          1.0000            1.0000        28.36
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_475.14       896.19     2_371.34       1.0000          1.0000            1.0000        28.36
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_475.14       685.21     2_160.35       1.0000          1.0000            1.0000        28.36
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_475.14     1_018.58     2_493.73       1.0000          1.0000            1.0000        28.36
IVF-TQ-b4-nl316 (self)                                 1_475.14     1_663.09     3_138.24       1.0000          1.0000            1.0000        28.36
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
Exhaustive (query)                                       100.64     1_911.18     2_011.82       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.64     6_528.84     6_629.48       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              604.85       953.13     1_557.98       0.8736          1.0271            1.0199        21.33
ExhaustiveTQ-b2-rf5 (query)                              604.85     1_047.96     1_652.81       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2-rf10 (query)                             604.85     1_199.76     1_804.61       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2-rf20 (query)                             604.85     1_623.47     2_228.31       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2 (self)                                   604.85     5_336.60     5_941.45       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b4-rf0 (query)                              746.47     1_766.42     2_512.90       0.9097          1.0146            1.0083        39.64
ExhaustiveTQ-b4-rf5 (query)                              746.47     1_858.92     2_605.39       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4-rf10 (query)                             746.47     1_988.20     2_734.67       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4-rf20 (query)                             746.47     2_396.61     3_143.09       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4 (self)                                   746.47     7_906.59     8_653.07       1.0000          1.0000            1.0000        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        3_097.14       349.53     3_446.67       0.8735          1.0272            1.0201        22.61
IVF-TQ-b2-nl158-np12-rf0 (query)                       3_097.14       447.66     3_544.80       0.8736          1.0271            1.0199        22.61
IVF-TQ-b2-nl158-np17-rf0 (query)                       3_097.14       525.65     3_622.78       0.8736          1.0271            1.0199        22.61
IVF-TQ-b2-nl158-np7-rf10 (query)                       3_097.14       610.84     3_707.98       0.9995          1.0001            1.0000        22.61
IVF-TQ-b2-nl158-np7-rf20 (query)                       3_097.14       938.29     4_035.43       0.9995          1.0001            1.0000        22.61
IVF-TQ-b2-nl158-np12-rf10 (query)                      3_097.14       717.86     3_815.00       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158-np12-rf20 (query)                      3_097.14     1_158.93     4_256.07       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158-np17-rf10 (query)                      3_097.14       804.35     3_901.49       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158-np17-rf20 (query)                      3_097.14     1_188.90     4_286.04       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158 (self)                                 3_097.14     2_018.88     5_116.02       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_744.61       345.74     2_090.35       0.8736          1.0271            1.0200        23.01
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_744.61       385.29     2_129.90       0.8736          1.0271            1.0199        23.01
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_744.61       471.09     2_215.70       0.8736          1.0271            1.0199        23.01
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_744.61       591.31     2_335.92       0.9998          1.0000            1.0000        23.01
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_744.61       917.40     2_662.00       0.9998          1.0000            1.0000        23.01
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_744.61       633.33     2_377.94       1.0000          1.0000            1.0000        23.01
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_744.61       981.92     2_726.53       1.0000          1.0000            1.0000        23.01
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_744.61       728.38     2_472.99       1.0000          1.0000            1.0000        23.01
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_744.61     1_095.57     2_840.18       1.0000          1.0000            1.0000        23.01
IVF-TQ-b2-nl223 (self)                                 1_744.61     1_987.73     3_732.33       1.0000          1.0000            1.0000        23.01
IVF-TQ-b2-nl316-np15-rf0 (query)                       2_044.63       356.26     2_400.89       0.8736          1.0271            1.0200        23.53
IVF-TQ-b2-nl316-np17-rf0 (query)                       2_044.63       376.34     2_420.97       0.8736          1.0271            1.0200        23.53
IVF-TQ-b2-nl316-np25-rf0 (query)                       2_044.63       457.73     2_502.36       0.8736          1.0271            1.0199        23.53
IVF-TQ-b2-nl316-np15-rf10 (query)                      2_044.63       606.84     2_651.47       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np15-rf20 (query)                      2_044.63       938.45     2_983.09       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np17-rf10 (query)                      2_044.63       628.73     2_673.36       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np17-rf20 (query)                      2_044.63       954.57     2_999.20       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np25-rf10 (query)                      2_044.63       695.05     2_739.68       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np25-rf20 (query)                      2_044.63     1_059.28     3_103.91       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316 (self)                                 2_044.63     1_973.51     4_018.15       1.0000          1.0000            1.0000        23.53
IVF-TQ-b4-nl158-np7-rf0 (query)                        3_171.81       525.76     3_697.56       0.9095          1.0147            1.0084        41.37
IVF-TQ-b4-nl158-np12-rf0 (query)                       3_171.81       702.22     3_874.02       0.9097          1.0146            1.0083        41.37
IVF-TQ-b4-nl158-np17-rf0 (query)                       3_171.81       852.22     4_024.03       0.9097          1.0146            1.0083        41.37
IVF-TQ-b4-nl158-np7-rf10 (query)                       3_171.81       795.85     3_967.66       0.9995          1.0001            1.0000        41.37
IVF-TQ-b4-nl158-np7-rf20 (query)                       3_171.81     1_138.81     4_310.62       0.9995          1.0001            1.0000        41.37
IVF-TQ-b4-nl158-np12-rf10 (query)                      3_171.81       994.05     4_165.86       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158-np12-rf20 (query)                      3_171.81     1_359.90     4_531.71       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158-np17-rf10 (query)                      3_171.81     1_137.94     4_309.75       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158-np17-rf20 (query)                      3_171.81     1_512.24     4_684.05       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158 (self)                                 3_171.81     2_609.96     5_781.77       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_883.37       522.98     2_406.35       0.9096          1.0146            1.0084        41.97
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_883.37       600.40     2_483.77       0.9097          1.0146            1.0083        41.97
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_883.37       765.60     2_648.97       0.9097          1.0146            1.0083        41.97
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_883.37       761.10     2_644.47       0.9998          1.0000            1.0000        41.97
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_883.37     1_086.90     2_970.27       0.9998          1.0000            1.0000        41.97
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_883.37       897.55     2_780.93       1.0000          1.0000            1.0000        41.97
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_883.37     1_183.63     3_067.00       1.0000          1.0000            1.0000        41.97
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_883.37     1_014.66     2_898.03       1.0000          1.0000            1.0000        41.97
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_883.37     1_379.06     3_262.43       1.0000          1.0000            1.0000        41.97
IVF-TQ-b4-nl223 (self)                                 1_883.37     2_468.92     4_352.30       1.0000          1.0000            1.0000        41.97
IVF-TQ-b4-nl316-np15-rf0 (query)                       2_187.59       533.68     2_721.27       0.9097          1.0146            1.0083        42.73
IVF-TQ-b4-nl316-np17-rf0 (query)                       2_187.59       573.17     2_760.77       0.9097          1.0146            1.0083        42.73
IVF-TQ-b4-nl316-np25-rf0 (query)                       2_187.59       718.73     2_906.32       0.9097          1.0146            1.0083        42.73
IVF-TQ-b4-nl316-np15-rf10 (query)                      2_187.59       758.22     2_945.81       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np15-rf20 (query)                      2_187.59     1_160.65     3_348.24       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np17-rf10 (query)                      2_187.59       799.68     2_987.27       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np17-rf20 (query)                      2_187.59     1_136.72     3_324.31       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np25-rf10 (query)                      2_187.59       954.67     3_142.26       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np25-rf20 (query)                      2_187.59     1_332.92     3_520.51       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316 (self)                                 2_187.59     2_438.11     4_625.71       1.0000          1.0000            1.0000        42.73
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
