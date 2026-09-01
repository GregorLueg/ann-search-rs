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
  dimensionality the recall degrades dramatically. In the IVF version it encodes
  the sign of the residual against each vector's own cell centroid instead,
  because a global sign bit tells you which cluster a point is in rather than
  where it sits inside that cluster.

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
  embedding dimensions (or the residual, for the IVF). In that last case
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
Exhaustive (query)                                        32.79       688.87       721.66       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.79     2_244.82     2_277.61       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                 77.28       239.93       317.22       0.1199          1.4617            1.4199         1.78
ExhaustiveBinary-256-random-rf10 (query)                  77.28       342.84       420.13       0.3412          1.0941            1.0814         1.78
ExhaustiveBinary-256-random-rf20 (query)                  77.28       449.46       526.74       0.4468          1.0571            1.0475         1.78
ExhaustiveBinary-256-random (self)                        77.28     1_106.74     1_184.02       0.3455          1.0895            1.0798         1.78
ExhaustiveBinary-256-pca_no_rr (query)                   110.45       238.99       349.44       0.1153          1.4748            1.4212         1.78
ExhaustiveBinary-256-pca-rf10 (query)                    110.45       343.47       453.92       0.3325          1.1029            1.0834         1.78
ExhaustiveBinary-256-pca-rf20 (query)                    110.45       449.92       560.37       0.4389          1.0631            1.0485         1.78
ExhaustiveBinary-256-pca (self)                          110.45     1_098.36     1_208.81       0.3391          1.0957            1.0813         1.78
ExhaustiveBinary-512-random_no_rr (query)                 84.35       348.94       433.29       0.1588          1.3547            1.3300         3.55
ExhaustiveBinary-512-random-rf10 (query)                  84.35       461.98       546.33       0.3786          1.0692            1.0677         3.55
ExhaustiveBinary-512-random-rf20 (query)                  84.35       580.15       664.50       0.4875          1.0424            1.0395         3.55
ExhaustiveBinary-512-random (self)                        84.35     1_515.39     1_599.75       0.3805          1.0675            1.0675         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   119.44       358.68       478.12       0.1564          1.3535            1.3265         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    119.44       465.15       584.59       0.3789          1.0710            1.0663         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    119.44       574.68       694.11       0.4905          1.0432            1.0387         3.55
ExhaustiveBinary-512-pca (self)                          119.44     1_506.94     1_626.38       0.3822          1.0678            1.0664         3.55
ExhaustiveBinary-1024-random_no_rr (query)               117.42       506.30       623.72       0.1928          1.2763            1.2695         7.10
ExhaustiveBinary-1024-random-rf10 (query)                117.42       628.49       745.91       0.4214          1.0550            1.0552         7.10
ExhaustiveBinary-1024-random-rf20 (query)                117.42       744.10       861.52       0.5434          1.0327            1.0308         7.10
ExhaustiveBinary-1024-random (self)                      117.42     2_079.75     2_197.17       0.4233          1.0547            1.0552         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  155.37       513.35       668.72       0.1923          1.2733            1.2652         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   155.37       646.50       801.87       0.4227          1.0546            1.0544         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   155.37       754.61       909.98       0.5444          1.0326            1.0304         7.10
ExhaustiveBinary-1024-pca (self)                         155.37     2_094.63     2_249.99       0.4235          1.0546            1.0548         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   40.66       442.81       483.47       0.1211          1.4987            1.4523         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    40.66       478.14       518.80       0.3286          1.1039            1.0884         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    40.66       723.61       764.27       0.4385          1.0623            1.0494         1.53
ExhaustiveBinary-256-sign (self)                          40.66     1_595.34     1_636.01       0.3333          1.0988            1.0859         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)            1_061.51        51.93     1_113.44       0.1239          1.4352            1.3995         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           1_061.51        54.25     1_115.76       0.1239          1.4352            1.3995         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           1_061.51        56.77     1_118.28       0.1239          1.4352            1.3995         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           1_061.51       105.58     1_167.09       0.3497          1.0878            1.0780         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           1_061.51       160.30     1_221.81       0.4589          1.0524            1.0449         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          1_061.51       101.62     1_163.13       0.3497          1.0878            1.0780         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          1_061.51       153.44     1_214.95       0.4589          1.0524            1.0449         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          1_061.51       101.86     1_163.37       0.3497          1.0878            1.0780         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          1_061.51       156.56     1_218.07       0.4589          1.0524            1.0449         1.93
IVF-Binary-256-nl158-random (self)                     1_061.51       228.98     1_290.49       0.3544          1.0828            1.0764         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             558.37        48.67       607.04       0.1409          1.3640            1.3206         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             558.37        51.22       609.59       0.1409          1.3646            1.3210         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             558.37        53.71       612.08       0.1409          1.3646            1.3210         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            558.37        99.58       657.95       0.3881          1.0691            1.0628         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            558.37       154.50       712.87       0.4965          1.0424            1.0371         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            558.37       105.35       663.72       0.3880          1.0691            1.0628         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            558.37       155.55       713.92       0.4956          1.0427            1.0372         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            558.37       109.07       667.44       0.3880          1.0691            1.0628         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            558.37       159.40       717.77       0.4956          1.0428            1.0372         2.00
IVF-Binary-256-nl223-random (self)                       558.37       227.71       786.08       0.3928          1.0648            1.0620         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             770.12        53.02       823.14       0.1513          1.3330            1.2878         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             770.12        54.70       824.82       0.1513          1.3333            1.2880         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             770.12        57.22       827.34       0.1513          1.3333            1.2880         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            770.12       104.19       874.31       0.4040          1.0639            1.0580         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            770.12       157.64       927.76       0.5072          1.0411            1.0355         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            770.12       104.15       874.27       0.4038          1.0640            1.0581         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            770.12       158.60       928.72       0.5061          1.0414            1.0357         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            770.12       107.24       877.37       0.4038          1.0639            1.0581         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            770.12       161.64       931.76       0.5062          1.0413            1.0357         2.09
IVF-Binary-256-nl316-random (self)                       770.12       242.04     1_012.16       0.4078          1.0601            1.0574         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               1_092.12        43.37     1_135.49       0.1195          1.4503            1.4064         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              1_092.12        44.58     1_136.70       0.1195          1.4503            1.4064         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              1_092.12        47.82     1_139.94       0.1195          1.4503            1.4064         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              1_092.12        96.29     1_188.41       0.3391          1.0969            1.0805         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              1_092.12       153.32     1_245.44       0.4472          1.0595            1.0465         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             1_092.12        97.65     1_189.76       0.3391          1.0969            1.0805         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             1_092.12       157.34     1_249.45       0.4472          1.0595            1.0465         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             1_092.12        99.39     1_191.50       0.3391          1.0969            1.0805         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             1_092.12       166.07     1_258.18       0.4472          1.0595            1.0465         1.93
IVF-Binary-256-nl158-pca (self)                        1_092.12       216.16     1_308.28       0.3458          1.0903            1.0782         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                606.73        53.38       660.11       0.1372          1.3751            1.3233         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                606.73        49.89       656.62       0.1371          1.3759            1.3238         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                606.73        61.63       668.36       0.1371          1.3759            1.3238         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               606.73        98.99       705.72       0.3797          1.0779            1.0646         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               606.73       153.43       760.16       0.4924          1.0475            1.0374         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               606.73       111.16       717.89       0.3793          1.0781            1.0647         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               606.73       152.08       758.81       0.4917          1.0476            1.0376         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               606.73       101.38       708.11       0.3793          1.0781            1.0647         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               606.73       159.76       766.49       0.4917          1.0476            1.0376         2.00
IVF-Binary-256-nl223-pca (self)                          606.73       239.40       846.13       0.3858          1.0717            1.0630         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                818.63        55.43       874.06       0.1481          1.3393            1.2928         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                818.63        52.71       871.34       0.1481          1.3398            1.2930         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                818.63        59.16       877.78       0.1480          1.3411            1.2932         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               818.63       105.90       924.53       0.3965          1.0698            1.0597         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               818.63       153.49       972.11       0.5089          1.0434            1.0354         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               818.63       102.42       921.05       0.3961          1.0700            1.0598         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               818.63       154.76       973.39       0.5081          1.0436            1.0356         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               818.63       105.58       924.20       0.3961          1.0699            1.0598         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               818.63       155.53       974.16       0.5081          1.0435            1.0356         2.09
IVF-Binary-256-nl316-pca (self)                          818.63       240.80     1_059.43       0.4019          1.0644            1.0586         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            1_085.02        62.47     1_147.50       0.1612          1.3437            1.3212         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           1_085.02        63.89     1_148.91       0.1612          1.3437            1.3212         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           1_085.02        66.50     1_151.53       0.1612          1.3437            1.3212         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           1_085.02       118.67     1_203.69       0.3829          1.0672            1.0659         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           1_085.02       169.36     1_254.38       0.4931          1.0411            1.0384         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          1_085.02       125.77     1_210.79       0.3829          1.0672            1.0659         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          1_085.02       174.21     1_259.23       0.4931          1.0411            1.0384         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          1_085.02       122.21     1_207.23       0.3829          1.0672            1.0659         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          1_085.02       175.76     1_260.78       0.4931          1.0411            1.0384         3.71
IVF-Binary-512-nl158-random (self)                     1_085.02       303.79     1_388.81       0.3848          1.0655            1.0659         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             586.05        68.53       654.58       0.1711          1.3022            1.2803         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             586.05        69.03       655.08       0.1711          1.3025            1.2807         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             586.05        73.57       659.63       0.1711          1.3025            1.2807         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            586.05       126.48       712.54       0.4021          1.0606            1.0594         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            586.05       171.81       757.87       0.5154          1.0371            1.0346         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            586.05       126.07       712.13       0.4017          1.0608            1.0595         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            586.05       174.10       760.15       0.5145          1.0372            1.0347         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            586.05       134.66       720.71       0.4017          1.0608            1.0595         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            586.05       177.68       763.74       0.5145          1.0372            1.0347         3.77
IVF-Binary-512-nl223-random (self)                       586.05       312.16       898.21       0.4033          1.0594            1.0594         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             781.81        70.49       852.29       0.1758          1.2873            1.2661         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             781.81        72.17       853.98       0.1758          1.2879            1.2667         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             781.81        76.18       857.99       0.1758          1.2879            1.2667         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            781.81       127.18       908.99       0.4078          1.0591            1.0574         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            781.81       174.62       956.43       0.5205          1.0365            1.0337         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            781.81       125.22       907.02       0.4072          1.0593            1.0576         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            781.81       177.82       959.63       0.5194          1.0367            1.0338         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            781.81       130.92       912.73       0.4073          1.0592            1.0576         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            781.81       180.90       962.70       0.5195          1.0367            1.0338         3.86
IVF-Binary-512-nl316-random (self)                       781.81       325.23     1_107.04       0.4096          1.0580            1.0577         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               1_117.10        61.53     1_178.63       0.1587          1.3439            1.3181         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              1_117.10        65.33     1_182.44       0.1587          1.3439            1.3181         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              1_117.10        67.00     1_184.10       0.1587          1.3439            1.3181         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              1_117.10       119.16     1_236.26       0.3829          1.0693            1.0650         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              1_117.10       169.34     1_286.45       0.4956          1.0422            1.0378         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             1_117.10       118.91     1_236.02       0.3829          1.0693            1.0650         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             1_117.10       171.38     1_288.48       0.4956          1.0422            1.0378         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             1_117.10       120.49     1_237.59       0.3829          1.0693            1.0650         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             1_117.10       178.07     1_295.17       0.4956          1.0422            1.0378         3.71
IVF-Binary-512-nl158-pca (self)                        1_117.10       302.14     1_419.24       0.3861          1.0663            1.0650         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                609.69        66.54       676.23       0.1683          1.3069            1.2786         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                609.69        72.53       682.21       0.1682          1.3077            1.2791         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                609.69        75.31       685.00       0.1682          1.3077            1.2791         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               609.69       122.16       731.84       0.4030          1.0620            1.0582         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               609.69       171.50       781.19       0.5181          1.0379            1.0340         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               609.69       121.88       731.57       0.4026          1.0622            1.0583         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               609.69       174.97       784.65       0.5173          1.0381            1.0341         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               609.69       125.76       735.44       0.4026          1.0622            1.0583         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               609.69       179.20       788.89       0.5173          1.0381            1.0341         3.77
IVF-Binary-512-nl223-pca (self)                          609.69       308.77       918.46       0.4053          1.0599            1.0584         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                813.49        70.89       884.38       0.1730          1.2922            1.2661         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                813.49        71.44       884.93       0.1730          1.2929            1.2665         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                813.49        76.29       889.79       0.1729          1.2929            1.2666         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               813.49       125.33       938.82       0.4106          1.0599            1.0564         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               813.49       173.98       987.47       0.5234          1.0369            1.0331         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               813.49       124.69       938.19       0.4102          1.0601            1.0564         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               813.49       176.04       989.53       0.5226          1.0371            1.0332         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               813.49       129.28       942.77       0.4102          1.0600            1.0564         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               813.49       189.72     1_003.21       0.5226          1.0370            1.0332         3.86
IVF-Binary-512-nl316-pca (self)                          813.49       321.69     1_135.18       0.4119          1.0581            1.0566         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)           1_138.16        95.03     1_233.19       0.1940          1.2725            1.2657         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)          1_138.16        99.59     1_237.75       0.1940          1.2725            1.2657         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)          1_138.16       101.69     1_239.85       0.1940          1.2725            1.2657         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)          1_138.16       154.71     1_292.87       0.4237          1.0543            1.0547         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)          1_138.16       209.70     1_347.86       0.5462          1.0323            1.0303         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)         1_138.16       157.18     1_295.34       0.4237          1.0543            1.0547         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)         1_138.16       215.08     1_353.24       0.5462          1.0323            1.0303         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)         1_138.16       160.07     1_298.23       0.4237          1.0543            1.0547         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)         1_138.16       218.43     1_356.59       0.5462          1.0323            1.0303         7.26
IVF-Binary-1024-nl158-random (self)                    1_138.16       438.64     1_576.79       0.4255          1.0541            1.0546         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            629.19       101.19       730.37       0.1972          1.2567            1.2493         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            629.19       102.90       732.08       0.1971          1.2572            1.2498         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            629.19       110.30       739.48       0.1971          1.2572            1.2498         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           629.19       157.94       787.13       0.4345          1.0516            1.0515         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           629.19       213.41       842.59       0.5571          1.0307            1.0286         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           629.19       160.14       789.33       0.4341          1.0517            1.0517         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           629.19       213.76       842.94       0.5566          1.0308            1.0287         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           629.19       169.30       798.49       0.4341          1.0517            1.0517         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           629.19       220.03       849.22       0.5566          1.0308            1.0287         7.32
IVF-Binary-1024-nl223-random (self)                      629.19       446.92     1_076.10       0.4359          1.0514            1.0517         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            819.99       106.11       926.09       0.1988          1.2508            1.2435         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)            819.99       106.44       926.43       0.1988          1.2515            1.2441         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)            819.99       112.01       932.00       0.1988          1.2514            1.2441         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)           819.99       161.61       981.59       0.4379          1.0509            1.0505         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)           819.99       214.98     1_034.97       0.5598          1.0304            1.0283         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)           819.99       161.65       981.64       0.4374          1.0510            1.0506         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)           819.99       216.84     1_036.82       0.5589          1.0306            1.0284         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)           819.99       168.68       988.67       0.4374          1.0510            1.0506         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)           819.99       241.09     1_061.07       0.5589          1.0305            1.0284         7.42
IVF-Binary-1024-nl316-random (self)                      819.99       460.07     1_280.06       0.4390          1.0508            1.0508         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)              1_187.37        93.43     1_280.80       0.1932          1.2695            1.2617         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)             1_187.37        99.10     1_286.48       0.1932          1.2695            1.2617         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)             1_187.37       100.85     1_288.22       0.1932          1.2695            1.2617         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)             1_187.37       154.37     1_341.74       0.4252          1.0539            1.0536         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)             1_187.37       207.92     1_395.30       0.5477          1.0321            1.0299         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)            1_187.37       159.44     1_346.82       0.4252          1.0539            1.0536         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)            1_187.37       212.29     1_399.66       0.5477          1.0321            1.0299         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)            1_187.37       162.26     1_349.63       0.4252          1.0539            1.0536         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)            1_187.37       223.66     1_411.03       0.5477          1.0321            1.0299         7.26
IVF-Binary-1024-nl158-pca (self)                       1_187.37       441.25     1_628.63       0.4260          1.0538            1.0540         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               667.04       101.68       768.72       0.1970          1.2529            1.2456         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               667.04       109.34       776.38       0.1970          1.2534            1.2461         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               667.04       108.76       775.81       0.1970          1.2534            1.2461         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              667.04       157.71       824.76       0.4369          1.0508            1.0506         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              667.04       209.99       877.04       0.5609          1.0302            1.0279         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              667.04       159.78       826.82       0.4366          1.0509            1.0506         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              667.04       237.67       904.71       0.5604          1.0303            1.0280         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              667.04       166.48       833.52       0.4366          1.0509            1.0506         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              667.04       219.87       886.91       0.5604          1.0303            1.0280         7.32
IVF-Binary-1024-nl223-pca (self)                         667.04       450.41     1_117.45       0.4367          1.0509            1.0511         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               861.01       105.41       966.42       0.1989          1.2474            1.2401         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)               861.01       109.16       970.17       0.1989          1.2478            1.2404         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)               861.01       112.64       973.66       0.1989          1.2478            1.2404         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)              861.01       162.60     1_023.61       0.4398          1.0504            1.0498         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)              861.01       213.41     1_074.43       0.5621          1.0301            1.0277         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)              861.01       161.67     1_022.68       0.4394          1.0505            1.0498         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)              861.01       216.04     1_077.05       0.5616          1.0302            1.0278         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)              861.01       168.59     1_029.61       0.4395          1.0504            1.0498         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)              861.01       223.88     1_084.89       0.5616          1.0302            1.0278         7.42
IVF-Binary-1024-nl316-pca (self)                         861.01       463.29     1_324.30       0.4397          1.0505            1.0503         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)              1_072.84       202.86     1_275.69       0.0604        339.0710          301.2567         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)             1_072.84       226.65     1_299.49       0.0584        646.7530          625.6251         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)             1_072.84       238.38     1_311.22       0.0581        710.8123          692.7255         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)             1_072.84       223.25     1_296.09       0.2402         12.7621           10.6530         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)             1_072.84       385.19     1_458.02       0.7125          1.3805            1.0092         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)            1_072.84       240.02     1_312.86       0.1305         18.9834           18.2392         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)            1_072.84       401.27     1_474.11       0.5024          4.8334            1.0288         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)            1_072.84       258.95     1_331.79       0.1206         19.2033           18.3189         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)            1_072.84       422.03     1_494.87       0.4474          5.9673            1.0402         1.68
IVF-Binary-256-nl158-sign (self)                       1_072.84       728.87     1_801.70       0.1311         19.1169           18.3513         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               580.85       223.59       804.44       0.0745        330.3456          242.0893         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               580.85       236.15       817.00       0.0575        523.6585          462.5988         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               580.85       264.69       845.54       0.0543        646.9479          644.7309         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              580.85       247.06       827.91       0.2680         13.4476            5.8520         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              580.85       412.57       993.42       0.5451          5.9049            1.0202         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              580.85       255.48       836.33       0.1819         19.0778           17.8634         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              580.85       418.50       999.35       0.4633          6.4645            1.0375         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              580.85       276.02       856.87       0.1175         22.8228           21.9953         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              580.85       445.46     1_026.31       0.3815          7.4572            1.0641         1.75
IVF-Binary-256-nl223-sign (self)                         580.85       770.10     1_350.95       0.1853         19.1727           17.9324         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               754.94       243.28       998.22       0.0835        323.0485          202.1954         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               754.94       260.93     1_015.87       0.0756        395.7054          299.1095         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               754.94       275.99     1_030.93       0.0590        673.9989          711.5677         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              754.94       266.39     1_021.33       0.2813         15.5690            5.6256         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              754.94       438.25     1_193.19       0.5377          2.7530            1.0229         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              754.94       270.04     1_024.98       0.2461         19.0979           17.6045         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              754.94       433.61     1_188.56       0.4848          4.3947            1.0327         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              754.94       289.22     1_044.16       0.1196         25.9642           25.1980         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              754.94       460.45     1_215.39       0.3660          6.9475            1.0688         1.84
IVF-Binary-256-nl316-sign (self)                         754.94       807.23     1_562.17       0.2474         19.2637           17.1280         1.84
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
Exhaustive (query)                                        69.96     1_355.96     1_425.92       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.96     4_573.05     4_643.02       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                141.69       270.95       412.65       0.1109          1.3512            1.3090         2.03
ExhaustiveBinary-256-random-rf10 (query)                 141.69       395.34       537.03       0.3145          1.0825            1.0612         2.03
ExhaustiveBinary-256-random-rf20 (query)                 141.69       526.11       667.80       0.4104          1.0522            1.0368         2.03
ExhaustiveBinary-256-random (self)                       141.69     1_255.09     1_396.78       0.3162          1.0784            1.0600         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   234.41       275.72       510.13       0.1167          1.3480            1.2981         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    234.41       391.69       626.10       0.3160          1.0791            1.0596         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    234.41       527.78       762.19       0.4121          1.0505            1.0362         2.03
ExhaustiveBinary-256-pca (self)                          234.41     1_231.85     1_466.26       0.3172          1.0782            1.0588         2.03
ExhaustiveBinary-512-random_no_rr (query)                205.40       385.30       590.70       0.1529          1.2600            1.2299         4.05
ExhaustiveBinary-512-random-rf10 (query)                 205.40       523.26       728.66       0.3465          1.0564            1.0513         4.05
ExhaustiveBinary-512-random-rf20 (query)                 205.40       655.18       860.58       0.4455          1.0358            1.0312         4.05
ExhaustiveBinary-512-random (self)                       205.40     1_716.12     1_921.51       0.3476          1.0547            1.0512         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   313.40       386.31       699.71       0.1559          1.2535            1.2254         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    313.40       527.79       841.19       0.3514          1.0523            1.0507         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    313.40       671.98       985.38       0.4487          1.0329            1.0309         4.05
ExhaustiveBinary-512-pca (self)                          313.40     1_677.57     1_990.97       0.3515          1.0522            1.0504         4.05
ExhaustiveBinary-1024-random_no_rr (query)               258.28       591.78       850.05       0.1815          1.2043            1.1936         8.11
ExhaustiveBinary-1024-random-rf10 (query)                258.28       788.94     1_047.21       0.3748          1.0447            1.0451         8.11
ExhaustiveBinary-1024-random-rf20 (query)                258.28       881.86     1_140.14       0.4789          1.0282            1.0270         8.11
ExhaustiveBinary-1024-random (self)                      258.28     2_411.85     2_670.13       0.3753          1.0447            1.0451         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  368.85       604.82       973.67       0.1832          1.2013            1.1905         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   368.85       747.70     1_116.55       0.3797          1.0434            1.0443         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   368.85       883.40     1_252.25       0.4865          1.0272            1.0261         8.11
ExhaustiveBinary-1024-pca (self)                         368.85     2_414.20     2_783.05       0.3786          1.0436            1.0444         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   95.30       683.43       778.73       0.1518          1.2700            1.2528         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    95.30       750.81       846.11       0.3399          1.0607            1.0535         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    95.30     1_146.18     1_241.48       0.4406          1.0369            1.0318         3.05
ExhaustiveBinary-512-sign (self)                          95.30     2_394.43     2_489.73       0.3407          1.0594            1.0531         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)            2_077.09        79.51     2_156.59       0.1150          1.3343            1.2989         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           2_077.09        81.22     2_158.31       0.1150          1.3343            1.2989         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           2_077.09        82.61     2_159.69       0.1150          1.3343            1.2989         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           2_077.09       164.13     2_241.22       0.3211          1.0782            1.0587         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           2_077.09       253.73     2_330.82       0.4184          1.0495            1.0354         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          2_077.09       161.14     2_238.23       0.3211          1.0782            1.0587         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          2_077.09       254.93     2_332.02       0.4184          1.0495            1.0354         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          2_077.09       161.75     2_238.83       0.3211          1.0782            1.0587         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          2_077.09       249.76     2_326.85       0.4184          1.0495            1.0354         2.34
IVF-Binary-256-nl158-random (self)                     2_077.09       312.06     2_389.14       0.3229          1.0742            1.0575         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           1_110.92        76.91     1_187.83       0.1336          1.2722            1.2321         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)           1_110.92        78.87     1_189.79       0.1336          1.2723            1.2321         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)           1_110.92        80.84     1_191.76       0.1336          1.2723            1.2321         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)          1_110.92       168.85     1_279.77       0.3734          1.0541            1.0444         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)          1_110.92       268.60     1_379.52       0.4736          1.0348            1.0271         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)          1_110.92       169.16     1_280.08       0.3734          1.0541            1.0444         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)          1_110.92       268.79     1_379.71       0.4736          1.0348            1.0271         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)          1_110.92       171.09     1_282.01       0.3734          1.0541            1.0444         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)          1_110.92       263.58     1_374.50       0.4736          1.0348            1.0271         2.47
IVF-Binary-256-nl223-random (self)                     1_110.92       336.62     1_447.53       0.3753          1.0508            1.0437         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)           1_339.22        85.78     1_425.00       0.1440          1.2536            1.2122         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           1_339.22        94.59     1_433.81       0.1440          1.2536            1.2122         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           1_339.22        89.03     1_428.26       0.1440          1.2536            1.2122         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          1_339.22       176.15     1_515.37       0.3845          1.0503            1.0417         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          1_339.22       267.06     1_606.28       0.4834          1.0332            1.0259         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          1_339.22       185.16     1_524.38       0.3845          1.0503            1.0417         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          1_339.22       268.05     1_607.27       0.4834          1.0332            1.0259         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          1_339.22       185.82     1_525.04       0.3845          1.0503            1.0417         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          1_339.22       268.19     1_607.42       0.4834          1.0332            1.0259         2.65
IVF-Binary-256-nl316-random (self)                     1_339.22       373.06     1_712.28       0.3860          1.0466            1.0412         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               2_149.15        72.84     2_221.99       0.1207          1.3294            1.2860         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              2_149.15        72.10     2_221.25       0.1207          1.3294            1.2860         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              2_149.15        72.64     2_221.79       0.1207          1.3294            1.2860         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              2_149.15       156.72     2_305.87       0.3226          1.0747            1.0574         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              2_149.15       254.54     2_403.69       0.4206          1.0469            1.0347         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             2_149.15       158.39     2_307.54       0.3226          1.0747            1.0574         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             2_149.15       256.35     2_405.50       0.4206          1.0469            1.0347         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             2_149.15       158.02     2_307.17       0.3226          1.0747            1.0574         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             2_149.15       252.77     2_401.92       0.4206          1.0469            1.0347         2.34
IVF-Binary-256-nl158-pca (self)                        2_149.15       312.65     2_461.80       0.3236          1.0739            1.0567         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              1_140.78        77.78     1_218.56       0.1385          1.2725            1.2270         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              1_140.78        88.91     1_229.69       0.1385          1.2726            1.2270         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              1_140.78        81.14     1_221.92       0.1385          1.2726            1.2270         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             1_140.78       171.77     1_312.55       0.3708          1.0524            1.0443         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             1_140.78       267.18     1_407.96       0.4712          1.0340            1.0275         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             1_140.78       174.12     1_314.90       0.3708          1.0524            1.0443         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             1_140.78       265.09     1_405.88       0.4712          1.0340            1.0275         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             1_140.78       172.55     1_313.33       0.3708          1.0524            1.0443         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             1_140.78       262.27     1_403.05       0.4712          1.0340            1.0275         2.47
IVF-Binary-256-nl223-pca (self)                        1_140.78       345.53     1_486.31       0.3731          1.0506            1.0437         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              1_403.28        87.06     1_490.35       0.1467          1.2561            1.2091         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              1_403.28        88.88     1_492.16       0.1467          1.2562            1.2091         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              1_403.28        88.95     1_492.24       0.1467          1.2562            1.2091         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             1_403.28       174.30     1_577.58       0.3799          1.0501            1.0423         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             1_403.28       268.12     1_671.40       0.4786          1.0329            1.0264         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             1_403.28       175.89     1_579.18       0.3799          1.0501            1.0423         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             1_403.28       265.04     1_668.32       0.4786          1.0329            1.0264         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             1_403.28       178.08     1_581.36       0.3799          1.0501            1.0423         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             1_403.28       278.53     1_681.81       0.4786          1.0329            1.0264         2.65
IVF-Binary-256-nl316-pca (self)                        1_403.28       368.94     1_772.22       0.3811          1.0488            1.0419         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)            2_114.27        97.32     2_211.59       0.1550          1.2528            1.2249         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)           2_114.27       102.77     2_217.04       0.1550          1.2528            1.2249         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)           2_114.27       106.25     2_220.52       0.1550          1.2528            1.2249         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)           2_114.27       194.24     2_308.51       0.3498          1.0552            1.0504         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)           2_114.27       283.52     2_397.79       0.4498          1.0349            1.0306         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)          2_114.27       187.31     2_301.58       0.3498          1.0552            1.0504         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)          2_114.27       291.94     2_406.21       0.4498          1.0349            1.0306         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)          2_114.27       196.25     2_310.52       0.3498          1.0552            1.0504         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)          2_114.27       291.81     2_406.07       0.4498          1.0349            1.0306         4.36
IVF-Binary-512-nl158-random (self)                     2_114.27       448.96     2_563.23       0.3507          1.0536            1.0502         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)           1_141.86       106.43     1_248.29       0.1644          1.2224            1.1966         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)           1_141.86       106.13     1_247.99       0.1644          1.2224            1.1966         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)           1_141.86       134.35     1_276.21       0.1644          1.2224            1.1966         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)          1_141.86       198.84     1_340.69       0.3714          1.0473            1.0446         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)          1_141.86       288.97     1_430.83       0.4711          1.0308            1.0275         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)          1_141.86       201.58     1_343.44       0.3714          1.0473            1.0446         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)          1_141.86       294.58     1_436.44       0.4711          1.0308            1.0275         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)          1_141.86       200.31     1_342.16       0.3714          1.0473            1.0446         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)          1_141.86       296.19     1_438.05       0.4711          1.0308            1.0275         4.49
IVF-Binary-512-nl223-random (self)                     1_141.86       469.72     1_611.58       0.3719          1.0462            1.0446         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)           1_421.49       110.35     1_531.84       0.1675          1.2154            1.1889         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)           1_421.49       111.95     1_533.44       0.1675          1.2154            1.1889         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)           1_421.49       114.83     1_536.32       0.1675          1.2154            1.1889         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)          1_421.49       203.36     1_624.85       0.3763          1.0461            1.0435         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)          1_421.49       297.29     1_718.78       0.4754          1.0302            1.0270         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)          1_421.49       203.53     1_625.02       0.3763          1.0461            1.0435         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)          1_421.49       298.71     1_720.20       0.4754          1.0302            1.0270         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)          1_421.49       211.57     1_633.06       0.3763          1.0461            1.0435         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)          1_421.49       299.46     1_720.95       0.4754          1.0302            1.0270         4.67
IVF-Binary-512-nl316-random (self)                     1_421.49       510.77     1_932.26       0.3763          1.0452            1.0436         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)               2_132.96        94.39     2_227.35       0.1582          1.2458            1.2197         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)              2_132.96        99.60     2_232.57       0.1582          1.2458            1.2197         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)              2_132.96       100.66     2_233.62       0.1582          1.2458            1.2197         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)              2_132.96       190.18     2_323.15       0.3554          1.0508            1.0495         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)              2_132.96       284.25     2_417.22       0.4537          1.0321            1.0300         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)             2_132.96       188.40     2_321.37       0.3554          1.0508            1.0495         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)             2_132.96       286.92     2_419.89       0.4537          1.0321            1.0300         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)             2_132.96       192.90     2_325.87       0.3554          1.0508            1.0495         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)             2_132.96       287.85     2_420.81       0.4537          1.0321            1.0300         4.36
IVF-Binary-512-nl158-pca (self)                        2_132.96       438.09     2_571.05       0.3554          1.0508            1.0493         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_229.55       103.64     1_333.19       0.1678          1.2159            1.1908         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_229.55       105.46     1_335.01       0.1678          1.2159            1.1908         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_229.55       110.04     1_339.59       0.1678          1.2159            1.1908         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_229.55       196.71     1_426.26       0.3745          1.0450            1.0445         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_229.55       293.11     1_522.66       0.4743          1.0289            1.0271         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_229.55       201.07     1_430.62       0.3745          1.0450            1.0445         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_229.55       293.25     1_522.80       0.4743          1.0289            1.0271         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_229.55       205.37     1_434.92       0.3745          1.0450            1.0445         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_229.55       294.85     1_524.40       0.4743          1.0289            1.0271         4.49
IVF-Binary-512-nl223-pca (self)                        1_229.55       452.01     1_681.56       0.3745          1.0449            1.0443         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)              1_454.44       111.92     1_566.36       0.1707          1.2098            1.1839         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)              1_454.44       113.15     1_567.58       0.1707          1.2098            1.1839         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)              1_454.44       116.22     1_570.66       0.1707          1.2098            1.1839         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)             1_454.44       203.74     1_658.17       0.3777          1.0443            1.0437         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)             1_454.44       301.87     1_756.31       0.4785          1.0283            1.0268         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)             1_454.44       201.17     1_655.61       0.3777          1.0443            1.0437         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)             1_454.44       307.37     1_761.81       0.4785          1.0283            1.0268         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)             1_454.44       211.18     1_665.62       0.3777          1.0443            1.0437         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)             1_454.44       301.61     1_756.05       0.4785          1.0283            1.0268         4.67
IVF-Binary-512-nl316-pca (self)                        1_454.44       505.09     1_959.53       0.3778          1.0443            1.0436         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)           2_106.26       150.63     2_256.89       0.1825          1.2017            1.1917         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)          2_106.26       149.13     2_255.38       0.1825          1.2017            1.1917         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)          2_106.26       152.37     2_258.62       0.1825          1.2017            1.1917         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)          2_106.26       240.62     2_346.87       0.3769          1.0441            1.0446         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)          2_106.26       340.66     2_446.92       0.4817          1.0278            1.0266         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)         2_106.26       246.68     2_352.93       0.3769          1.0441            1.0446         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)         2_106.26       345.49     2_451.74       0.4817          1.0278            1.0266         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)         2_106.26       247.30     2_353.56       0.3769          1.0441            1.0446         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)         2_106.26       352.65     2_458.91       0.4817          1.0278            1.0266         8.42
IVF-Binary-1024-nl158-random (self)                    2_106.26       634.27     2_740.52       0.3775          1.0441            1.0445         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_190.82       154.88     1_345.71       0.1859          1.1883            1.1790         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_190.82       155.92     1_346.75       0.1859          1.1883            1.1790         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_190.82       162.62     1_353.44       0.1859          1.1883            1.1790         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_190.82       248.94     1_439.77       0.3873          1.0417            1.0419         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_190.82       350.43     1_541.26       0.4938          1.0263            1.0249         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_190.82       259.98     1_450.80       0.3873          1.0417            1.0419         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_190.82       354.57     1_545.39       0.4938          1.0263            1.0249         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_190.82       259.80     1_450.62       0.3873          1.0417            1.0419         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_190.82       359.95     1_550.77       0.4938          1.0263            1.0249         8.54
IVF-Binary-1024-nl223-random (self)                    1_190.82       658.04     1_848.86       0.3882          1.0416            1.0419         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)          1_451.74       164.76     1_616.50       0.1870          1.1851            1.1756         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)          1_451.74       164.16     1_615.90       0.1870          1.1851            1.1756         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)          1_451.74       167.94     1_619.68       0.1870          1.1851            1.1756         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)         1_451.74       263.61     1_715.35       0.3898          1.0411            1.0414         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)         1_451.74       373.59     1_825.33       0.4959          1.0260            1.0247         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)         1_451.74       267.27     1_719.01       0.3898          1.0411            1.0414         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)         1_451.74       381.39     1_833.13       0.4959          1.0260            1.0247         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)         1_451.74       268.88     1_720.62       0.3898          1.0411            1.0414         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)         1_451.74       367.63     1_819.37       0.4959          1.0260            1.0247         8.73
IVF-Binary-1024-nl316-random (self)                    1_451.74       681.33     2_133.07       0.3905          1.0410            1.0415         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)              2_208.13       146.34     2_354.46       0.1841          1.1983            1.1886         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)             2_208.13       151.25     2_359.37       0.1841          1.1983            1.1886         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)             2_208.13       152.60     2_360.72       0.1841          1.1983            1.1886         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)             2_208.13       241.88     2_450.01       0.3818          1.0429            1.0438         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)             2_208.13       333.52     2_541.65       0.4890          1.0268            1.0257         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)            2_208.13       244.00     2_452.12       0.3818          1.0429            1.0438         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)            2_208.13       346.01     2_554.14       0.4890          1.0268            1.0257         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)            2_208.13       250.31     2_458.44       0.3818          1.0429            1.0438         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)            2_208.13       346.55     2_554.68       0.4890          1.0268            1.0257         8.42
IVF-Binary-1024-nl158-pca (self)                       2_208.13       638.46     2_846.58       0.3807          1.0431            1.0439         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_263.02       156.12     1_419.14       0.1874          1.1858            1.1764         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_263.02       156.07     1_419.09       0.1874          1.1858            1.1764         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_263.02       161.27     1_424.29       0.1874          1.1858            1.1764         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_263.02       252.95     1_515.97       0.3923          1.0404            1.0414         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_263.02       350.91     1_613.93       0.5010          1.0253            1.0243         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_263.02       251.49     1_514.51       0.3923          1.0404            1.0414         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_263.02       350.03     1_613.05       0.5010          1.0253            1.0243         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_263.02       257.16     1_520.18       0.3923          1.0404            1.0414         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_263.02       370.37     1_633.39       0.5010          1.0253            1.0243         8.54
IVF-Binary-1024-nl223-pca (self)                       1_263.02       652.85     1_915.87       0.3917          1.0405            1.0412         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_529.77       163.94     1_693.71       0.1885          1.1831            1.1738         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_529.77       163.32     1_693.09       0.1885          1.1831            1.1738         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_529.77       168.20     1_697.97       0.1885          1.1831            1.1738         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_529.77       261.30     1_791.07       0.3944          1.0399            1.0409         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_529.77       359.29     1_889.06       0.5032          1.0251            1.0241         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_529.77       260.17     1_789.94       0.3944          1.0399            1.0409         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_529.77       360.45     1_890.22       0.5032          1.0251            1.0241         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_529.77       270.12     1_799.89       0.3944          1.0399            1.0409         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_529.77       386.12     1_915.89       0.5032          1.0251            1.0241         8.73
IVF-Binary-1024-nl316-pca (self)                       1_529.77       687.76     2_217.53       0.3937          1.0401            1.0409         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)              1_966.13       350.05     2_316.18       0.0656        327.6909          208.3967         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)             1_966.13       385.55     2_351.68       0.0642        547.4464          360.0161         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)             1_966.13       408.78     2_374.91       0.0638        615.5449          479.5490         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)             1_966.13       388.48     2_354.61       0.2285         15.8942           16.1011         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)             1_966.13       691.08     2_657.22       0.7342          1.0280            1.0061         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)            1_966.13       416.44     2_382.58       0.1416         20.2673           21.4708         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)            1_966.13       751.70     2_717.84       0.5249          4.3538            1.0162         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)            1_966.13       444.95     2_411.09       0.1257         21.2309           22.1920         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)            1_966.13       750.18     2_716.32       0.4614          5.7308            1.0237         3.36
IVF-Binary-512-nl158-sign (self)                       1_966.13     1_203.99     3_170.13       0.1411         20.4652           21.6805         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)             1_015.88       384.23     1_400.10       0.0732        401.4540          189.1446         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)             1_015.88       402.31     1_418.19       0.0646        583.1073          476.3585         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)             1_015.88       446.53     1_462.40       0.0642        732.8963          852.3300         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)            1_015.88       430.79     1_446.66       0.2369         14.7391            9.3703         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)            1_015.88       729.92     1_745.80       0.5809          2.3790            1.0130         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)            1_015.88       446.31     1_462.18       0.1610         21.3471           22.9807         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)            1_015.88       740.19     1_756.06       0.4856          4.2385            1.0213         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)            1_015.88       491.47     1_507.34       0.1234         25.3001           25.3670         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)            1_015.88       812.08     1_827.96       0.3760          8.6040            1.0429         3.49
IVF-Binary-512-nl223-sign (self)                       1_015.88     1_297.32     2_313.19       0.1578         21.4598           23.1244         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)             1_245.71       414.55     1_660.26       0.0683        381.9140          167.4888         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)             1_245.71       423.75     1_669.46       0.0670        488.5484          403.1886         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)             1_245.71       481.11     1_726.82       0.0663        746.8099          831.8734         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)            1_245.71       460.61     1_706.32       0.2539         11.2563            4.2341         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)            1_245.71       761.09     2_006.80       0.5646          2.6870            1.0142         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)            1_245.71       479.73     1_725.44       0.1958         14.5463            9.4992         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)            1_245.71       767.44     2_013.15       0.5034          3.2888            1.0197         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)            1_245.71       506.15     1_751.86       0.1199         25.9296           19.2330         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)            1_245.71       810.52     2_056.23       0.3402          5.3031            1.0540         3.67
IVF-Binary-512-nl316-sign (self)                       1_245.71     1_368.79     2_614.50       0.1934         14.6881            9.6092         3.67
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
Exhaustive (query)                                       101.18     1_869.40     1_970.58       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.18     6_357.43     6_458.61       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                192.71       281.43       474.14       0.1141          1.2808            1.2432         2.28
ExhaustiveBinary-256-random-rf10 (query)                 192.71       425.10       617.82       0.3149          1.0655            1.0475         2.28
ExhaustiveBinary-256-random-rf20 (query)                 192.71       585.48       778.19       0.4080          1.0420            1.0293         2.28
ExhaustiveBinary-256-random (self)                       192.71     1_313.40     1_506.12       0.3170          1.0618            1.0471         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   405.05       283.22       688.28       0.1053          1.3025            1.2616         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    405.05       419.65       824.70       0.3013          1.0735            1.0517         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    405.05       572.44       977.50       0.3931          1.0471            1.0314         2.28
ExhaustiveBinary-256-pca (self)                          405.05     1_300.87     1_705.93       0.3049          1.0709            1.0504         2.28
ExhaustiveBinary-512-random_no_rr (query)                298.57       402.97       701.54       0.1506          1.2093            1.1808         4.55
ExhaustiveBinary-512-random-rf10 (query)                 298.57       561.77       860.34       0.3395          1.0453            1.0425         4.55
ExhaustiveBinary-512-random-rf20 (query)                 298.57       718.56     1_017.13       0.4325          1.0293            1.0264         4.55
ExhaustiveBinary-512-random (self)                       298.57     1_776.36     2_074.93       0.3401          1.0435            1.0422         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   505.04       416.91       921.95       0.1460          1.2161            1.1913         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    505.04       589.74     1_094.78       0.3341          1.0467            1.0434         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    505.04       723.89     1_228.93       0.4278          1.0295            1.0268         4.55
ExhaustiveBinary-512-pca (self)                          505.04     1_786.43     2_291.47       0.3356          1.0453            1.0432         4.55
ExhaustiveBinary-1024-random_no_rr (query)               499.45       627.00     1_126.45       0.1762          1.1673            1.1570         9.11
ExhaustiveBinary-1024-random-rf10 (query)                499.45       802.28     1_301.72       0.3602          1.0383            1.0383         9.11
ExhaustiveBinary-1024-random-rf20 (query)                499.45       997.77     1_497.22       0.4619          1.0244            1.0230         9.11
ExhaustiveBinary-1024-random (self)                      499.45     2_587.05     3_086.50       0.3601          1.0377            1.0382         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  705.60       619.97     1_325.57       0.1757          1.1686            1.1585         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   705.60       801.82     1_507.42       0.3596          1.0382            1.0385         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   705.60       976.43     1_682.03       0.4577          1.0246            1.0237         9.11
ExhaustiveBinary-1024-pca (self)                         705.60     2_589.89     3_295.49       0.3583          1.0383            1.0389         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  142.54       857.35       999.89       0.1691          1.1871            1.1717         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   142.54       942.29     1_084.82       0.3431          1.0433            1.0415         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   142.54     1_455.30     1_597.83       0.4437          1.0266            1.0249         4.58
ExhaustiveBinary-768-sign (self)                         142.54     3_026.07     3_168.60       0.3437          1.0423            1.0413         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)            2_897.08        97.62     2_994.70       0.1169          1.2711            1.2394         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)           2_897.08       100.73     2_997.81       0.1169          1.2711            1.2394         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)           2_897.08       105.09     3_002.17       0.1169          1.2711            1.2394         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)           2_897.08       204.03     3_101.11       0.3177          1.0634            1.0467         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)           2_897.08       317.34     3_214.42       0.4105          1.0411            1.0289         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)          2_897.08       203.54     3_100.62       0.3177          1.0634            1.0467         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)          2_897.08       319.82     3_216.89       0.4105          1.0411            1.0289         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)          2_897.08       204.62     3_101.70       0.3177          1.0634            1.0467         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)          2_897.08       318.17     3_215.24       0.4105          1.0411            1.0289         2.74
IVF-Binary-256-nl158-random (self)                     2_897.08       417.98     3_315.06       0.3197          1.0600            1.0463         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)           1_404.79       103.72     1_508.51       0.1316          1.2343            1.1950         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)           1_404.79       102.12     1_506.91       0.1316          1.2346            1.1951         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)           1_404.79       119.54     1_524.34       0.1316          1.2346            1.1951         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)          1_404.79       245.94     1_650.73       0.3458          1.0521            1.0395         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)          1_404.79       344.01     1_748.81       0.4475          1.0325            1.0241         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)          1_404.79       241.48     1_646.27       0.3456          1.0522            1.0396         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)          1_404.79       362.57     1_767.36       0.4470          1.0326            1.0242         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)          1_404.79       220.46     1_625.25       0.3456          1.0522            1.0396         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)          1_404.79       329.22     1_734.01       0.4470          1.0326            1.0242         2.93
IVF-Binary-256-nl223-random (self)                     1_404.79       454.49     1_859.29       0.3478          1.0487            1.0393         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)           1_880.76       113.04     1_993.81       0.1392          1.2180            1.1762         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)           1_880.76       112.21     1_992.98       0.1392          1.2187            1.1764         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)           1_880.76       114.67     1_995.44       0.1392          1.2187            1.1764         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)          1_880.76       220.97     2_101.74       0.3596          1.0465            1.0373         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)          1_880.76       338.93     2_219.69       0.4602          1.0294            1.0229         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)          1_880.76       219.03     2_099.79       0.3595          1.0465            1.0373         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)          1_880.76       339.31     2_220.07       0.4600          1.0295            1.0230         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)          1_880.76       223.19     2_103.96       0.3594          1.0465            1.0373         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)          1_880.76       342.70     2_223.46       0.4600          1.0295            1.0230         3.21
IVF-Binary-256-nl316-random (self)                     1_880.76       501.06     2_381.83       0.3615          1.0433            1.0370         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)               3_102.89        92.33     3_195.22       0.1083          1.2892            1.2557         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)              3_102.89        91.53     3_194.42       0.1083          1.2892            1.2557         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)              3_102.89        92.20     3_195.10       0.1083          1.2892            1.2557         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)              3_102.89       196.36     3_299.25       0.3064          1.0701            1.0506         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)              3_102.89       309.57     3_412.46       0.3987          1.0444            1.0306         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)             3_102.89       195.78     3_298.67       0.3064          1.0701            1.0506         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)             3_102.89       314.79     3_417.69       0.3987          1.0444            1.0306         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)             3_102.89       196.39     3_299.29       0.3064          1.0701            1.0506         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)             3_102.89       313.78     3_416.68       0.3987          1.0444            1.0306         2.74
IVF-Binary-256-nl158-pca (self)                        3_102.89       399.20     3_502.09       0.3102          1.0665            1.0493         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)              1_536.75        99.95     1_636.70       0.1260          1.2452            1.2083         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)              1_536.75       100.31     1_637.06       0.1260          1.2454            1.2084         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)              1_536.75       103.06     1_639.81       0.1260          1.2454            1.2084         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)             1_536.75       212.31     1_749.06       0.3428          1.0552            1.0403         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)             1_536.75       323.89     1_860.64       0.4451          1.0344            1.0243         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)             1_536.75       227.26     1_764.01       0.3425          1.0553            1.0404         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)             1_536.75       343.00     1_879.75       0.4442          1.0346            1.0245         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)             1_536.75       222.56     1_759.31       0.3425          1.0553            1.0404         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)             1_536.75       394.01     1_930.76       0.4442          1.0346            1.0245         2.93
IVF-Binary-256-nl223-pca (self)                        1_536.75       475.54     2_012.29       0.3452          1.0530            1.0398         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)              1_984.22       110.16     2_094.38       0.1355          1.2243            1.1879         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)              1_984.22       110.56     2_094.78       0.1355          1.2245            1.1881         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)              1_984.22       113.65     2_097.87       0.1355          1.2245            1.1881         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)             1_984.22       220.69     2_204.91       0.3607          1.0481            1.0370         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)             1_984.22       335.48     2_319.70       0.4597          1.0308            1.0229         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)             1_984.22       219.71     2_203.93       0.3604          1.0482            1.0370         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)             1_984.22       337.21     2_321.43       0.4593          1.0309            1.0230         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)             1_984.22       226.04     2_210.26       0.3603          1.0482            1.0371         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)             1_984.22       342.04     2_326.26       0.4592          1.0309            1.0230         3.21
IVF-Binary-256-nl316-pca (self)                        1_984.22       483.66     2_467.88       0.3631          1.0457            1.0364         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)            3_005.07       128.36     3_133.43       0.1522          1.2056            1.1785         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)           3_005.07       129.22     3_134.28       0.1522          1.2056            1.1785         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)           3_005.07       131.70     3_136.77       0.1522          1.2056            1.1785         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)           3_005.07       241.01     3_246.08       0.3406          1.0448            1.0422         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)           3_005.07       358.74     3_363.80       0.4343          1.0290            1.0261         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)          3_005.07       240.86     3_245.93       0.3406          1.0448            1.0422         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)          3_005.07       371.40     3_376.47       0.4343          1.0290            1.0261         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)          3_005.07       247.06     3_252.13       0.3406          1.0448            1.0422         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)          3_005.07       366.81     3_371.88       0.4343          1.0290            1.0261         5.02
IVF-Binary-512-nl158-random (self)                     3_005.07       580.99     3_586.06       0.3414          1.0431            1.0419         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)           1_423.54       136.96     1_560.50       0.1589          1.1868            1.1603         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)           1_423.54       137.44     1_560.98       0.1588          1.1870            1.1604         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)           1_423.54       141.31     1_564.85       0.1588          1.1870            1.1604         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)          1_423.54       251.81     1_675.35       0.3555          1.0408            1.0385         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)          1_423.54       390.36     1_813.90       0.4528          1.0262            1.0238         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)          1_423.54       251.86     1_675.40       0.3552          1.0408            1.0386         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)          1_423.54       377.19     1_800.73       0.4522          1.0262            1.0239         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)          1_423.54       257.01     1_680.55       0.3552          1.0408            1.0386         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)          1_423.54       381.01     1_804.55       0.4522          1.0262            1.0239         5.21
IVF-Binary-512-nl223-random (self)                     1_423.54       612.48     2_036.02       0.3555          1.0394            1.0385         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)           1_886.84       148.56     2_035.40       0.1619          1.1799            1.1536         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)           1_886.84       147.56     2_034.40       0.1618          1.1800            1.1538         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)           1_886.84       156.47     2_043.31       0.1618          1.1801            1.1538         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)          1_886.84       264.28     2_151.11       0.3616          1.0393            1.0373         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)          1_886.84       382.30     2_269.14       0.4576          1.0255            1.0234         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)          1_886.84       261.96     2_148.80       0.3614          1.0394            1.0374         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)          1_886.84       387.52     2_274.36       0.4573          1.0255            1.0235         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)          1_886.84       273.38     2_160.22       0.3614          1.0394            1.0374         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)          1_886.84       394.40     2_281.24       0.4572          1.0255            1.0235         5.48
IVF-Binary-512-nl316-random (self)                     1_886.84       652.45     2_539.28       0.3614          1.0379            1.0374         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)               3_191.54       125.53     3_317.07       0.1478          1.2109            1.1885         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)              3_191.54       127.86     3_319.40       0.1478          1.2109            1.1885         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)              3_191.54       130.25     3_321.79       0.1478          1.2109            1.1885         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)              3_191.54       239.54     3_431.08       0.3369          1.0454            1.0427         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)              3_191.54       353.54     3_545.08       0.4306          1.0290            1.0265         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)             3_191.54       242.58     3_434.12       0.3369          1.0454            1.0427         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)             3_191.54       358.58     3_550.12       0.4306          1.0290            1.0265         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)             3_191.54       241.39     3_432.94       0.3369          1.0454            1.0427         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)             3_191.54       367.39     3_558.93       0.4306          1.0290            1.0265         5.02
IVF-Binary-512-nl158-pca (self)                        3_191.54       591.64     3_783.19       0.3384          1.0441            1.0425         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_663.29       136.48     1_799.77       0.1561          1.1888            1.1636         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_663.29       137.48     1_800.77       0.1560          1.1891            1.1640         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_663.29       141.27     1_804.57       0.1560          1.1891            1.1640         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_663.29       254.72     1_918.02       0.3536          1.0409            1.0386         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_663.29       369.00     2_032.29       0.4509          1.0265            1.0240         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_663.29       252.07     1_915.37       0.3532          1.0410            1.0387         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_663.29       369.49     2_032.78       0.4501          1.0266            1.0241         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_663.29       255.70     1_918.99       0.3533          1.0410            1.0387         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_663.29       384.85     2_048.14       0.4501          1.0266            1.0241         5.21
IVF-Binary-512-nl223-pca (self)                        1_663.29       617.65     2_280.94       0.3541          1.0403            1.0387         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)              2_139.30       146.53     2_285.83       0.1600          1.1800            1.1553         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)              2_139.30       152.66     2_291.96       0.1600          1.1803            1.1554         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)              2_139.30       152.49     2_291.79       0.1600          1.1804            1.1555         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)             2_139.30       261.78     2_401.08       0.3604          1.0394            1.0374         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)             2_139.30       378.86     2_518.17       0.4568          1.0255            1.0233         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)             2_139.30       266.63     2_405.93       0.3602          1.0395            1.0375         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)             2_139.30       383.72     2_523.02       0.4563          1.0256            1.0234         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)             2_139.30       267.97     2_407.27       0.3602          1.0395            1.0375         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)             2_139.30       388.25     2_527.55       0.4563          1.0256            1.0234         5.48
IVF-Binary-512-nl316-pca (self)                        2_139.30       656.20     2_795.50       0.3610          1.0384            1.0375         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)           3_221.14       207.71     3_428.85       0.1768          1.1660            1.1560         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)          3_221.14       209.65     3_430.79       0.1768          1.1660            1.1560         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)          3_221.14       216.57     3_437.71       0.1768          1.1660            1.1560         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)          3_221.14       325.66     3_546.80       0.3610          1.0381            1.0381         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)          3_221.14       446.28     3_667.42       0.4630          1.0242            1.0229         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)         3_221.14       323.54     3_544.68       0.3610          1.0381            1.0381         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)         3_221.14       452.02     3_673.16       0.4630          1.0242            1.0229         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)         3_221.14       328.59     3_549.73       0.3610          1.0381            1.0381         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)         3_221.14       473.84     3_694.98       0.4630          1.0242            1.0229         9.57
IVF-Binary-1024-nl158-random (self)                    3_221.14       860.35     4_081.49       0.3610          1.0375            1.0381         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_637.72       210.84     1_848.56       0.1790          1.1581            1.1484         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_637.72       217.91     1_855.63       0.1789          1.1584            1.1487         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_637.72       216.35     1_854.07       0.1789          1.1584            1.1487         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_637.72       338.88     1_976.61       0.3691          1.0364            1.0364         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_637.72       462.80     2_100.52       0.4723          1.0232            1.0220         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_637.72       339.69     1_977.42       0.3687          1.0364            1.0365         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_637.72       463.72     2_101.44       0.4716          1.0233            1.0220         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_637.72       346.26     1_983.98       0.3687          1.0364            1.0365         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_637.72       484.46     2_122.18       0.4716          1.0233            1.0220         9.76
IVF-Binary-1024-nl223-random (self)                    1_637.72       892.57     2_530.29       0.3685          1.0359            1.0364         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)          2_101.68       223.95     2_325.64       0.1803          1.1548            1.1453        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)          2_101.68       224.76     2_326.44       0.1802          1.1550            1.1455        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)          2_101.68       227.62     2_329.31       0.1802          1.1551            1.1455        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)         2_101.68       352.27     2_453.95       0.3724          1.0358            1.0359        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)         2_101.68       479.84     2_581.52       0.4757          1.0228            1.0216        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)         2_101.68       352.22     2_453.90       0.3723          1.0358            1.0359        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)         2_101.68       484.16     2_585.84       0.4754          1.0228            1.0216        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)         2_101.68       363.06     2_464.74       0.3723          1.0358            1.0359        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)         2_101.68       491.61     2_593.29       0.4754          1.0228            1.0216        10.04
IVF-Binary-1024-nl316-random (self)                    2_101.68       973.63     3_075.32       0.3721          1.0352            1.0358        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)              3_429.79       201.63     3_631.42       0.1765          1.1668            1.1570         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)             3_429.79       201.38     3_631.16       0.1765          1.1668            1.1570         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)             3_429.79       208.03     3_637.82       0.1765          1.1668            1.1570         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)             3_429.79       316.89     3_746.68       0.3606          1.0379            1.0382         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)             3_429.79       440.43     3_870.22       0.4592          1.0244            1.0235         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)            3_429.79       320.01     3_749.79       0.3606          1.0379            1.0382         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)            3_429.79       452.29     3_882.08       0.4592          1.0244            1.0235         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)            3_429.79       326.46     3_756.25       0.3606          1.0379            1.0382         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)            3_429.79       473.14     3_902.93       0.4592          1.0244            1.0235         9.57
IVF-Binary-1024-nl158-pca (self)                       3_429.79       877.89     4_307.68       0.3594          1.0380            1.0386         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_843.33       211.70     2_055.04       0.1787          1.1580            1.1479         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_843.33       212.15     2_055.48       0.1786          1.1582            1.1482         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_843.33       218.45     2_061.78       0.1786          1.1582            1.1482         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_843.33       331.92     2_175.26       0.3690          1.0362            1.0364         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_843.33       463.24     2_306.57       0.4697          1.0232            1.0224         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_843.33       366.47     2_209.80       0.3686          1.0362            1.0365         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_843.33       480.06     2_323.39       0.4690          1.0233            1.0225         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_843.33       353.59     2_196.92       0.3687          1.0362            1.0365         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_843.33       477.69     2_321.02       0.4690          1.0233            1.0225         9.76
IVF-Binary-1024-nl223-pca (self)                       1_843.33       909.53     2_752.86       0.3679          1.0363            1.0367         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             2_306.21       227.16     2_533.37       0.1801          1.1545            1.1450        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)             2_306.21       222.60     2_528.81       0.1801          1.1547            1.1451        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)             2_306.21       229.35     2_535.56       0.1801          1.1547            1.1451        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)            2_306.21       349.58     2_655.79       0.3722          1.0355            1.0358        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)            2_306.21       480.94     2_787.15       0.4731          1.0229            1.0221        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)            2_306.21       351.69     2_657.90       0.3721          1.0356            1.0358        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)            2_306.21       477.83     2_784.04       0.4727          1.0229            1.0221        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)            2_306.21       361.43     2_667.64       0.3720          1.0356            1.0359        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)            2_306.21       494.40     2_800.61       0.4726          1.0229            1.0221        10.04
IVF-Binary-1024-nl316-pca (self)                       2_306.21       951.32     3_257.53       0.3712          1.0356            1.0362        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)              2_881.15       492.80     3_373.95       0.0683        392.3382          248.0775         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)             2_881.15       542.87     3_424.02       0.0671        599.1377          464.9128         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)             2_881.15       577.38     3_458.52       0.0669        664.1896          501.3283         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)             2_881.15       542.04     3_423.19       0.2247         16.2957           17.1916         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)             2_881.15       972.05     3_853.20       0.7264          1.0235            1.0049         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)            2_881.15       599.32     3_480.47       0.1381         20.2411           21.5074         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)            2_881.15     1_059.98     3_941.13       0.5311          4.8959            1.0116         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)            2_881.15       628.57     3_509.72       0.1232         22.6937           22.6334         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)            2_881.15     1_052.88     3_934.03       0.4592          5.6620            1.0180         5.04
IVF-Binary-768-nl158-sign (self)                       2_881.15     1_686.58     4_567.73       0.1381         20.2340           21.5947         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)             1_272.47       538.18     1_810.65       0.0838        322.6009          138.5984         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)             1_272.47       566.80     1_839.26       0.0705        422.9169          307.0583         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)             1_272.47       628.73     1_901.19       0.0626        721.8519          842.0128         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)            1_272.47       623.85     1_896.31       0.3018         17.9210            4.4404         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)            1_272.47     1_026.31     2_298.78       0.5599          6.1700            1.0085         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)            1_272.47       631.55     1_904.02       0.2273         20.4026           10.6272         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)            1_272.47     1_042.21     2_314.68       0.4968          7.9296            1.0133         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)            1_272.47       684.18     1_956.65       0.1226         30.9641           21.5813         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)            1_272.47     1_097.60     2_370.07       0.3579          9.4895            1.0431         5.23
IVF-Binary-768-nl223-sign (self)                       1_272.47     1_804.59     3_077.05       0.2258         20.7209           12.1499         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)             1_733.46       589.20     2_322.66       0.0902        339.0098          154.5000         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)             1_733.46       593.42     2_326.88       0.0758        428.5042          211.1469         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)             1_733.46       658.56     2_392.02       0.0675        757.1592          810.8178         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)            1_733.46       647.29     2_380.75       0.2936         12.5064            3.7891         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)            1_733.46     1_064.52     2_797.98       0.5371          4.4503            1.0117         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)            1_733.46       660.68     2_394.14       0.2574         13.8389            4.0363         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)            1_733.46     1_080.01     2_813.47       0.4890          5.8066            1.0154         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)            1_733.46       711.37     2_444.83       0.1284         23.6495           19.1251         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)            1_733.46     1_155.16     2_888.62       0.3559          7.2503            1.0411         5.51
IVF-Binary-768-nl316-sign (self)                       1_733.46     1_932.99     3_666.45       0.2564         14.0152            4.2503         5.51
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
Exhaustive (query)                                        33.26       730.32       763.57       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.26     2_520.22     2_553.47       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                 70.15       240.43       310.58       0.0970          1.6334            1.6378         1.78
ExhaustiveBinary-256-random-rf10 (query)                  70.15       349.53       419.68       0.3643          1.1391            1.1302         1.78
ExhaustiveBinary-256-random-rf20 (query)                  70.15       455.95       526.10       0.5086          1.0798            1.0701         1.78
ExhaustiveBinary-256-random (self)                        70.15     1_132.96     1_203.11       0.3862          1.1443            1.1409         1.78
ExhaustiveBinary-256-pca_no_rr (query)                    99.45       241.55       341.00       0.0923          1.6524            1.6606         1.78
ExhaustiveBinary-256-pca-rf10 (query)                     99.45       346.23       445.68       0.3518          1.1465            1.1384         1.78
ExhaustiveBinary-256-pca-rf20 (query)                     99.45       457.20       556.65       0.4944          1.0846            1.0744         1.78
ExhaustiveBinary-256-pca (self)                           99.45     1_171.55     1_271.00       0.3765          1.1502            1.1476         1.78
ExhaustiveBinary-512-random_no_rr (query)                 83.68       357.99       441.67       0.1464          1.5035            1.5085         3.55
ExhaustiveBinary-512-random-rf10 (query)                  83.68       475.18       558.86       0.4596          1.0936            1.0901         3.55
ExhaustiveBinary-512-random-rf20 (query)                  83.68       588.82       672.50       0.6085          1.0504            1.0459         3.55
ExhaustiveBinary-512-random (self)                        83.68     1_561.33     1_645.01       0.4800          1.0996            1.0995         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   112.53       358.72       471.25       0.1458          1.5041            1.5097         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    112.53       482.81       595.34       0.4545          1.0952            1.0911         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    112.53       587.14       699.67       0.6038          1.0513            1.0464         3.55
ExhaustiveBinary-512-pca (self)                          112.53     1_572.11     1_684.64       0.4777          1.1003            1.0997         3.55
ExhaustiveBinary-1024-random_no_rr (query)               119.67       537.07       656.74       0.2154          1.3655            1.3721         7.10
ExhaustiveBinary-1024-random-rf10 (query)                119.67       673.91       793.58       0.5868          1.0540            1.0515         7.10
ExhaustiveBinary-1024-random-rf20 (query)                119.67       800.94       920.61       0.7379          1.0260            1.0224         7.10
ExhaustiveBinary-1024-random (self)                      119.67     2_184.81     2_304.47       0.6117          1.0575            1.0546         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  144.62       541.33       685.96       0.2122          1.3735            1.3798         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   144.62       664.78       809.41       0.5776          1.0560            1.0532         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   144.62       784.99       929.61       0.7291          1.0273            1.0232         7.10
ExhaustiveBinary-1024-pca (self)                         144.62     2_164.93     2_309.55       0.6017          1.0602            1.0571         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   41.33       455.42       496.75       0.1044          1.6421            1.6499         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    41.33       490.18       531.51       0.3737          1.1368            1.1275         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    41.33       754.76       796.09       0.5266          1.0745            1.0646         1.53
ExhaustiveBinary-256-sign (self)                          41.33     1_593.06     1_634.40       0.3940          1.1439            1.1394         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)            1_118.00        52.32     1_170.32       0.1011          1.6221            1.6308         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           1_118.00        53.52     1_171.52       0.1011          1.6223            1.6309         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           1_118.00        57.46     1_175.46       0.1011          1.6223            1.6309         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           1_118.00       107.30     1_225.30       0.3712          1.1366            1.1285         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           1_118.00       154.65     1_272.64       0.5144          1.0783            1.0692         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          1_118.00       101.71     1_219.71       0.3700          1.1368            1.1285         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          1_118.00       152.84     1_270.84       0.5130          1.0785            1.0693         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          1_118.00       108.86     1_226.86       0.3699          1.1368            1.1285         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          1_118.00       164.08     1_282.08       0.5129          1.0785            1.0693         1.93
IVF-Binary-256-nl158-random (self)                     1_118.00       231.38     1_349.38       0.3920          1.1417            1.1393         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             686.04        49.53       735.58       0.1142          1.5772            1.5811         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             686.04        53.45       739.50       0.1141          1.5774            1.5812         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             686.04        55.15       741.19       0.1141          1.5774            1.5812         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            686.04       103.84       789.89       0.4010          1.1186            1.1129         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            686.04       154.94       840.99       0.5423          1.0684            1.0612         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            686.04       103.59       789.64       0.4009          1.1186            1.1129         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            686.04       152.40       838.45       0.5421          1.0685            1.0613         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            686.04       108.10       794.14       0.4009          1.1187            1.1129         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            686.04       159.31       845.35       0.5420          1.0685            1.0613         2.00
IVF-Binary-256-nl223-random (self)                       686.04       242.37       928.41       0.4218          1.1228            1.1240         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             862.03        53.20       915.23       0.1196          1.5615            1.5630         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             862.03        54.10       916.13       0.1196          1.5616            1.5630         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             862.03        57.89       919.93       0.1196          1.5616            1.5630         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            862.03       106.55       968.59       0.4079          1.1158            1.1096         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            862.03       155.40     1_017.44       0.5489          1.0666            1.0596         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            862.03       108.66       970.69       0.4077          1.1159            1.1096         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            862.03       155.25     1_017.28       0.5488          1.0666            1.0596         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            862.03       111.32       973.36       0.4077          1.1159            1.1096         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            862.03       159.62     1_021.65       0.5487          1.0666            1.0596         2.09
IVF-Binary-256-nl316-random (self)                       862.03       253.57     1_115.60       0.4286          1.1195            1.1203         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               1_081.67        43.49     1_125.16       0.0965          1.6410            1.6539         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              1_081.67        45.00     1_126.67       0.0965          1.6412            1.6541         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              1_081.67        47.86     1_129.54       0.0965          1.6412            1.6541         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              1_081.67        96.47     1_178.14       0.3580          1.1444            1.1373         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              1_081.67       147.21     1_228.88       0.4992          1.0834            1.0737         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             1_081.67       102.17     1_183.84       0.3572          1.1445            1.1373         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             1_081.67       147.65     1_229.33       0.4983          1.0835            1.0737         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             1_081.67        99.81     1_181.48       0.3572          1.1445            1.1373         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             1_081.67       151.46     1_233.13       0.4983          1.0835            1.0737         1.93
IVF-Binary-256-nl158-pca (self)                        1_081.67       227.61     1_309.29       0.3820          1.1481            1.1462         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                681.46        49.27       730.72       0.1119          1.5907            1.5950         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                681.46        50.07       731.52       0.1118          1.5908            1.5951         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                681.46        55.10       736.55       0.1118          1.5908            1.5951         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               681.46       103.09       784.55       0.3907          1.1241            1.1187         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               681.46       149.03       830.49       0.5321          1.0716            1.0644         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               681.46       101.52       782.98       0.3906          1.1241            1.1187         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               681.46       152.76       834.21       0.5319          1.0716            1.0644         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               681.46       106.77       788.23       0.3906          1.1241            1.1187         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               681.46       154.62       836.08       0.5319          1.0716            1.0644         2.00
IVF-Binary-256-nl223-pca (self)                          681.46       238.14       919.60       0.4147          1.1273            1.1286         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                872.13        53.03       925.16       0.1171          1.5729            1.5766         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                872.13        53.63       925.76       0.1171          1.5730            1.5768         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                872.13        57.83       929.96       0.1171          1.5731            1.5768         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               872.13       106.00       978.14       0.3990          1.1199            1.1145         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               872.13       154.84     1_026.97       0.5393          1.0695            1.0623         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               872.13       104.96       977.09       0.3989          1.1199            1.1145         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               872.13       161.82     1_033.95       0.5391          1.0696            1.0623         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               872.13       108.82       980.95       0.3988          1.1199            1.1145         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               872.13       157.59     1_029.72       0.5390          1.0696            1.0623         2.09
IVF-Binary-256-nl316-pca (self)                          872.13       254.41     1_126.54       0.4225          1.1227            1.1240         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            1_076.81        60.81     1_137.62       0.1484          1.4999            1.5068         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           1_076.81        63.35     1_140.15       0.1484          1.4999            1.5068         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           1_076.81        68.64     1_145.45       0.1484          1.4999            1.5068         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           1_076.81       121.81     1_198.62       0.4619          1.0930            1.0897         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           1_076.81       169.77     1_246.57       0.6102          1.0501            1.0456         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          1_076.81       121.49     1_198.30       0.4615          1.0931            1.0898         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          1_076.81       172.69     1_249.50       0.6100          1.0501            1.0456         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          1_076.81       126.06     1_202.87       0.4615          1.0931            1.0898         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          1_076.81       177.19     1_253.99       0.6099          1.0501            1.0456         3.71
IVF-Binary-512-nl158-random (self)                     1_076.81       303.30     1_380.10       0.4821          1.0990            1.0989         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             665.29        66.76       732.05       0.1576          1.4760            1.4813         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             665.29        68.12       733.41       0.1576          1.4760            1.4813         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             665.29        78.70       743.99       0.1576          1.4760            1.4813         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            665.29       124.31       789.59       0.4774          1.0867            1.0842         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            665.29       173.99       839.28       0.6240          1.0468            1.0430         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            665.29       125.87       791.16       0.4774          1.0867            1.0842         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            665.29       177.45       842.74       0.6239          1.0468            1.0430         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            665.29       135.08       800.37       0.4774          1.0867            1.0842         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            665.29       189.81       855.10       0.6239          1.0468            1.0430         3.77
IVF-Binary-512-nl223-random (self)                       665.29       324.47       989.76       0.4974          1.0929            1.0926         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             932.05        71.66     1_003.71       0.1603          1.4696            1.4738         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             932.05        73.25     1_005.30       0.1603          1.4696            1.4738         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             932.05        78.24     1_010.29       0.1603          1.4696            1.4738         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            932.05       131.20     1_063.25       0.4804          1.0859            1.0830         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            932.05       182.47     1_114.52       0.6264          1.0465            1.0423         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            932.05       130.46     1_062.51       0.4804          1.0859            1.0830         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            932.05       182.17     1_114.22       0.6264          1.0465            1.0423         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            932.05       145.30     1_077.35       0.4804          1.0859            1.0830         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            932.05       189.20     1_121.24       0.6264          1.0465            1.0423         3.86
IVF-Binary-512-nl316-random (self)                       932.05       332.85     1_264.89       0.5001          1.0919            1.0914         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               1_151.65        60.62     1_212.27       0.1477          1.5004            1.5074         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              1_151.65        65.93     1_217.58       0.1477          1.5004            1.5074         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              1_151.65        70.94     1_222.59       0.1477          1.5004            1.5074         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              1_151.65       123.77     1_275.42       0.4569          1.0945            1.0906         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              1_151.65       180.97     1_332.62       0.6056          1.0509            1.0460         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             1_151.65       125.26     1_276.91       0.4567          1.0945            1.0906         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             1_151.65       175.58     1_327.23       0.6055          1.0509            1.0460         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             1_151.65       126.55     1_278.20       0.4567          1.0945            1.0906         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             1_151.65       184.13     1_335.78       0.6055          1.0509            1.0460         3.71
IVF-Binary-512-nl158-pca (self)                        1_151.65       305.77     1_457.42       0.4800          1.0997            1.0991         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                742.41        69.87       812.28       0.1572          1.4778            1.4831         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                742.41        69.23       811.64       0.1572          1.4779            1.4831         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                742.41        75.98       818.39       0.1572          1.4779            1.4831         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               742.41       125.66       868.07       0.4732          1.0880            1.0849         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               742.41       173.05       915.46       0.6198          1.0474            1.0432         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               742.41       126.60       869.01       0.4732          1.0880            1.0849         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               742.41       179.00       921.41       0.6198          1.0474            1.0432         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               742.41       133.63       876.04       0.4732          1.0880            1.0849         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               742.41       183.20       925.61       0.6198          1.0474            1.0432         3.77
IVF-Binary-512-nl223-pca (self)                          742.41       317.25     1_059.66       0.4957          1.0934            1.0931         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                898.15        74.01       972.16       0.1595          1.4706            1.4745         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                898.15        73.16       971.32       0.1594          1.4706            1.4745         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                898.15        80.85       979.00       0.1594          1.4707            1.4745         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               898.15       130.05     1_028.20       0.4766          1.0869            1.0835         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               898.15       189.16     1_087.31       0.6234          1.0468            1.0424         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               898.15       128.29     1_026.45       0.4766          1.0869            1.0835         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               898.15       184.28     1_082.43       0.6234          1.0468            1.0424         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               898.15       135.57     1_033.72       0.4766          1.0869            1.0835         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               898.15       187.75     1_085.90       0.6234          1.0468            1.0424         3.86
IVF-Binary-512-nl316-pca (self)                          898.15       341.63     1_239.78       0.4988          1.0922            1.0916         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)           1_163.07        92.93     1_256.00       0.2165          1.3642            1.3710         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)          1_163.07        97.62     1_260.69       0.2165          1.3642            1.3710         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)          1_163.07       104.25     1_267.32       0.2165          1.3642            1.3710         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)          1_163.07       162.03     1_325.10       0.5876          1.0538            1.0514         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)          1_163.07       209.76     1_372.84       0.7384          1.0259            1.0223         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)         1_163.07       159.83     1_322.91       0.5876          1.0538            1.0514         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)         1_163.07       217.97     1_381.04       0.7384          1.0259            1.0223         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)         1_163.07       170.11     1_333.19       0.5876          1.0538            1.0514         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)         1_163.07       220.42     1_383.49       0.7384          1.0259            1.0223         7.26
IVF-Binary-1024-nl158-random (self)                    1_163.07       437.30     1_600.37       0.6126          1.0574            1.0544         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            748.44       100.56       849.00       0.2206          1.3566            1.3640         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            748.44       102.16       850.60       0.2206          1.3566            1.3640         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            748.44       112.09       860.53       0.2206          1.3566            1.3640         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           748.44       163.09       911.53       0.5944          1.0521            1.0498         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           748.44       242.33       990.77       0.7439          1.0251            1.0214         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           748.44       220.77       969.22       0.5943          1.0521            1.0498         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           748.44       255.26     1_003.70       0.7439          1.0251            1.0214         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           748.44       183.53       931.97       0.5943          1.0521            1.0498         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           748.44       239.07       987.51       0.7439          1.0251            1.0214         7.32
IVF-Binary-1024-nl223-random (self)                      748.44       456.18     1_204.62       0.6194          1.0556            1.0528         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)          1_028.17       114.76     1_142.93       0.2220          1.3538            1.3607         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)          1_028.17       107.05     1_135.22       0.2220          1.3538            1.3607         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)          1_028.17       116.98     1_145.14       0.2220          1.3538            1.3607         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)         1_028.17       166.09     1_194.26       0.5963          1.0517            1.0494         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)         1_028.17       224.53     1_252.70       0.7450          1.0249            1.0212         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)         1_028.17       179.83     1_208.00       0.5963          1.0517            1.0494         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)         1_028.17       229.66     1_257.82       0.7450          1.0249            1.0212         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)         1_028.17       176.86     1_205.03       0.5963          1.0517            1.0494         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)         1_028.17       234.12     1_262.29       0.7450          1.0249            1.0212         7.42
IVF-Binary-1024-nl316-random (self)                    1_028.17       473.44     1_501.61       0.6209          1.0552            1.0523         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)              1_226.13        98.00     1_324.13       0.2131          1.3724            1.3791         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)             1_226.13       107.85     1_333.98       0.2131          1.3724            1.3791         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)             1_226.13       106.81     1_332.94       0.2131          1.3724            1.3791         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)             1_226.13       155.29     1_381.41       0.5785          1.0558            1.0530         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)             1_226.13       212.38     1_438.50       0.7299          1.0272            1.0232         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)            1_226.13       163.98     1_390.10       0.5784          1.0558            1.0530         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)            1_226.13       218.64     1_444.76       0.7298          1.0272            1.0232         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)            1_226.13       171.02     1_397.14       0.5784          1.0558            1.0530         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)            1_226.13       223.33     1_449.46       0.7298          1.0272            1.0232         7.26
IVF-Binary-1024-nl158-pca (self)                       1_226.13       452.93     1_679.06       0.6025          1.0600            1.0570         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               901.98       102.39     1_004.37       0.2179          1.3632            1.3707         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               901.98       102.35     1_004.33       0.2179          1.3632            1.3707         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               901.98       116.10     1_018.09       0.2179          1.3632            1.3707         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              901.98       162.85     1_064.84       0.5857          1.0540            1.0511         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              901.98       217.12     1_119.11       0.7360          1.0262            1.0223         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              901.98       175.13     1_077.11       0.5857          1.0540            1.0511         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              901.98       221.10     1_123.09       0.7360          1.0262            1.0223         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              901.98       177.53     1_079.51       0.5857          1.0540            1.0511         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              901.98       231.00     1_132.99       0.7360          1.0262            1.0223         7.32
IVF-Binary-1024-nl223-pca (self)                         901.98       458.87     1_360.86       0.6104          1.0579            1.0548         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               958.43       105.84     1_064.26       0.2192          1.3600            1.3671         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)               958.43       106.82     1_065.25       0.2192          1.3600            1.3671         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)               958.43       116.16     1_074.58       0.2192          1.3600            1.3671         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)              958.43       167.83     1_126.26       0.5878          1.0535            1.0508         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)              958.43       240.66     1_199.09       0.7376          1.0260            1.0220         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)              958.43       227.91     1_186.34       0.5878          1.0535            1.0508         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)              958.43       221.37     1_179.80       0.7376          1.0260            1.0220         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)              958.43       172.67     1_131.10       0.5878          1.0535            1.0508         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)              958.43       230.99     1_189.42       0.7376          1.0260            1.0220         7.42
IVF-Binary-1024-nl316-pca (self)                         958.43       492.33     1_450.76       0.6119          1.0574            1.0543         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)              1_129.83       203.33     1_333.16       0.1210          7.7702            6.5339         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)             1_129.83       229.09     1_358.92       0.1064         16.7605            7.3611         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)             1_129.83       248.16     1_377.99       0.0918         42.2224           22.4290         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)             1_129.83       234.83     1_364.66       0.8422          1.0245            1.0041         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)             1_129.83       399.28     1_529.11       0.9387          1.0095            1.0000         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)            1_129.83       255.78     1_385.61       0.7567          1.0562            1.0099         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)            1_129.83       431.08     1_560.91       0.9105          1.0161            1.0000         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)            1_129.83       275.21     1_405.04       0.6666          1.1393            1.0219         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)            1_129.83       458.63     1_588.46       0.8638          1.0265            1.0015         1.68
IVF-Binary-256-nl158-sign (self)                       1_129.83       734.18     1_864.01       0.7880          1.0523            1.0081         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               669.66       235.17       904.83       0.1202         20.0567           10.8032         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               669.66       241.80       911.46       0.1086         28.9079           16.9131         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               669.66       276.49       946.14       0.0918         61.8330           52.4644         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              669.66       258.23       927.89       0.7661          1.0366            1.0108         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              669.66       423.69     1_093.34       0.9179          1.0106            1.0000         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              669.66       272.14       941.80       0.7067          1.0529            1.0180         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              669.66       435.39     1_105.04       0.8909          1.0149            1.0011         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              669.66       300.57       970.23       0.5888          1.1377            1.0428         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              669.66       474.10     1_143.76       0.8131          1.0282            1.0061         1.75
IVF-Binary-256-nl223-sign (self)                         669.66       806.46     1_476.12       0.7487          1.0480            1.0145         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               849.81       244.41     1_094.23       0.1591         18.0822            8.6937         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               849.81       259.05     1_108.86       0.1472         22.4513           12.9078         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               849.81       294.94     1_144.75       0.1172         59.0188           51.6847         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              849.81       278.77     1_128.58       0.7580          1.0406            1.0117         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              849.81       450.81     1_300.62       0.9036          1.0126            1.0004         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              849.81       289.83     1_139.64       0.7269          1.0488            1.0150         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              849.81       464.67     1_314.48       0.8881          1.0152            1.0013         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              849.81       317.08     1_166.89       0.6247          1.0902            1.0323         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              849.81       487.67     1_337.48       0.8261          1.0268            1.0053         1.84
IVF-Binary-256-nl316-sign (self)                         849.81       851.04     1_700.85       0.7637          1.0462            1.0125         1.84
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
Exhaustive (query)                                        67.82     1_299.80     1_367.62       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         67.82     4_346.00     4_413.82       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                134.43       270.04       404.48       0.0733          1.4884            1.4968         2.03
ExhaustiveBinary-256-random-rf10 (query)                 134.43       393.76       528.20       0.2947          1.1326            1.1260         2.03
ExhaustiveBinary-256-random-rf20 (query)                 134.43       524.16       658.60       0.4175          1.0830            1.0716         2.03
ExhaustiveBinary-256-random (self)                       134.43     1_272.86     1_407.29       0.3150          1.1321            1.1268         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   228.24       269.53       497.76       0.0722          1.4941            1.5000         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    228.24       388.01       616.25       0.2934          1.1324            1.1267         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    228.24       522.36       750.60       0.4180          1.0813            1.0715         2.03
ExhaustiveBinary-256-pca (self)                          228.24     1_217.22     1_445.46       0.3112          1.1337            1.1298         2.03
ExhaustiveBinary-512-random_no_rr (query)                209.41       377.11       586.52       0.1110          1.4064            1.4153         4.05
ExhaustiveBinary-512-random-rf10 (query)                 209.41       527.88       737.29       0.3694          1.0935            1.0919         4.05
ExhaustiveBinary-512-random-rf20 (query)                 209.41       661.76       871.17       0.4980          1.0544            1.0516         4.05
ExhaustiveBinary-512-random (self)                       209.41     1_653.81     1_863.22       0.3856          1.0953            1.0998         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   296.97       391.49       688.46       0.1064          1.4156            1.4272         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    296.97       517.14       814.11       0.3575          1.0990            1.0958         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    296.97       674.31       971.29       0.4854          1.0582            1.0544         4.05
ExhaustiveBinary-512-pca (self)                          296.97     1_645.28     1_942.25       0.3753          1.0995            1.1040         4.05
ExhaustiveBinary-1024-random_no_rr (query)               255.30       568.90       824.20       0.1593          1.3242            1.3318         8.11
ExhaustiveBinary-1024-random-rf10 (query)                255.30       729.56       984.86       0.4456          1.0660            1.0678         8.11
ExhaustiveBinary-1024-random-rf20 (query)                255.30       872.28     1_127.59       0.5824          1.0370            1.0362         8.11
ExhaustiveBinary-1024-random (self)                      255.30     2_366.39     2_621.70       0.4597          1.0714            1.0740         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  346.75       583.39       930.14       0.1600          1.3236            1.3333         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   346.75       722.60     1_069.35       0.4447          1.0658            1.0680         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   346.75       879.32     1_226.07       0.5813          1.0369            1.0362         8.11
ExhaustiveBinary-1024-pca (self)                         346.75     2_379.30     2_726.05       0.4581          1.0716            1.0746         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   84.00       682.54       766.54       0.1291          1.3815            1.3877         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    84.00       740.78       824.77       0.3926          1.0844            1.0833         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    84.00     1_172.00     1_255.99       0.5334          1.0464            1.0444         3.05
ExhaustiveBinary-512-sign (self)                          84.00     2_398.99     2_482.98       0.4063          1.0885            1.0916         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)            1_925.95        79.19     2_005.14       0.0767          1.4793            1.4940         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           1_925.95        79.01     2_004.96       0.0766          1.4796            1.4942         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           1_925.95        81.02     2_006.97       0.0766          1.4796            1.4942         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           1_925.95       160.16     2_086.11       0.2995          1.1307            1.1253         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           1_925.95       244.43     2_170.38       0.4214          1.0815            1.0713         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          1_925.95       154.80     2_080.75       0.2986          1.1309            1.1253         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          1_925.95       247.30     2_173.25       0.4204          1.0816            1.0713         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          1_925.95       155.64     2_081.59       0.2986          1.1309            1.1253         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          1_925.95       249.53     2_175.48       0.4204          1.0816            1.0713         2.34
IVF-Binary-256-nl158-random (self)                     1_925.95       307.37     2_233.32       0.3188          1.1304            1.1264         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           1_055.54        76.55     1_132.10       0.0874          1.4526            1.4605         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)           1_055.54        77.92     1_133.46       0.0874          1.4528            1.4607         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)           1_055.54        80.47     1_136.02       0.0874          1.4528            1.4607         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)          1_055.54       166.78     1_222.32       0.3261          1.1145            1.1088         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)          1_055.54       256.15     1_311.70       0.4507          1.0697            1.0627         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)          1_055.54       160.95     1_216.50       0.3257          1.1146            1.1089         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)          1_055.54       254.53     1_310.07       0.4503          1.0698            1.0628         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)          1_055.54       165.02     1_220.57       0.3257          1.1146            1.1089         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)          1_055.54       257.11     1_312.66       0.4502          1.0698            1.0628         2.47
IVF-Binary-256-nl223-random (self)                     1_055.54       332.96     1_388.50       0.3448          1.1137            1.1144         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)           1_473.26        83.42     1_556.68       0.0934          1.4376            1.4416         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           1_473.26        87.98     1_561.25       0.0934          1.4378            1.4417         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           1_473.26        87.15     1_560.41       0.0934          1.4379            1.4417         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          1_473.26       169.76     1_643.03       0.3380          1.1077            1.1029         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          1_473.26       260.91     1_734.18       0.4655          1.0645            1.0592         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          1_473.26       174.92     1_648.19       0.3378          1.1078            1.1029         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          1_473.26       269.34     1_742.61       0.4654          1.0645            1.0592         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          1_473.26       172.34     1_645.61       0.3378          1.1078            1.1029         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          1_473.26       264.57     1_737.84       0.4654          1.0645            1.0592         2.65
IVF-Binary-256-nl316-random (self)                     1_473.26       364.25     1_837.51       0.3570          1.1061            1.1088         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               2_008.06        68.25     2_076.31       0.0755          1.4848            1.4976         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              2_008.06        70.80     2_078.86       0.0755          1.4850            1.4977         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              2_008.06        72.45     2_080.51       0.0755          1.4850            1.4977         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              2_008.06       151.82     2_159.88       0.2987          1.1305            1.1259         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              2_008.06       247.30     2_255.36       0.4220          1.0803            1.0711         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             2_008.06       152.30     2_160.35       0.2980          1.1306            1.1259         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             2_008.06       246.37     2_254.43       0.4210          1.0804            1.0711         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             2_008.06       153.00     2_161.06       0.2980          1.1306            1.1259         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             2_008.06       257.42     2_265.47       0.4209          1.0804            1.0711         2.34
IVF-Binary-256-nl158-pca (self)                        2_008.06       299.34     2_307.39       0.3157          1.1319            1.1292         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              1_136.74        75.30     1_212.04       0.0869          1.4554            1.4608         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              1_136.74        76.44     1_213.18       0.0868          1.4557            1.4610         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              1_136.74        81.05     1_217.80       0.0868          1.4557            1.4610         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             1_136.74       163.57     1_300.31       0.3262          1.1141            1.1083         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             1_136.74       255.02     1_391.76       0.4481          1.0705            1.0629         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             1_136.74       162.25     1_298.99       0.3259          1.1142            1.1084         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             1_136.74       257.10     1_393.85       0.4475          1.0706            1.0630         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             1_136.74       164.64     1_301.39       0.3259          1.1142            1.1084         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             1_136.74       261.78     1_398.52       0.4475          1.0706            1.0630         2.47
IVF-Binary-256-nl223-pca (self)                        1_136.74       330.95     1_467.69       0.3437          1.1130            1.1145         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              1_544.62        83.85     1_628.47       0.0938          1.4381            1.4394         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              1_544.62        83.93     1_628.55       0.0938          1.4385            1.4396         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              1_544.62        87.12     1_631.74       0.0938          1.4385            1.4396         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             1_544.62       170.70     1_715.32       0.3379          1.1070            1.1018         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             1_544.62       262.03     1_806.65       0.4595          1.0665            1.0599         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             1_544.62       168.43     1_713.05       0.3378          1.1070            1.1018         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             1_544.62       262.22     1_806.84       0.4594          1.0666            1.0599         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             1_544.62       172.43     1_717.05       0.3378          1.1070            1.1018         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             1_544.62       265.64     1_810.26       0.4593          1.0666            1.0599         2.65
IVF-Binary-256-nl316-pca (self)                        1_544.62       360.79     1_905.41       0.3565          1.1048            1.1088         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)            1_989.87        93.94     2_083.82       0.1124          1.4039            1.4144         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)           1_989.87        97.33     2_087.21       0.1124          1.4040            1.4144         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)           1_989.87       100.15     2_090.03       0.1124          1.4040            1.4144         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)           1_989.87       185.39     2_175.26       0.3709          1.0930            1.0917         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)           1_989.87       290.98     2_280.85       0.4993          1.0540            1.0515         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)          1_989.87       185.48     2_175.35       0.3707          1.0930            1.0917         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)          1_989.87       284.67     2_274.54       0.4991          1.0540            1.0515         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)          1_989.87       191.94     2_181.82       0.3707          1.0930            1.0917         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)          1_989.87       291.69     2_281.56       0.4991          1.0540            1.0515         4.36
IVF-Binary-512-nl158-random (self)                     1_989.87       442.31     2_432.18       0.3870          1.0948            1.0997         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)           1_133.40       102.67     1_236.07       0.1207          1.3867            1.3939         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)           1_133.40       104.22     1_237.63       0.1207          1.3869            1.3940         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)           1_133.40       108.39     1_241.79       0.1207          1.3869            1.3940         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)          1_133.40       194.79     1_328.20       0.3839          1.0873            1.0869         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)          1_133.40       297.63     1_431.04       0.5117          1.0511            1.0492         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)          1_133.40       194.45     1_327.85       0.3837          1.0874            1.0870         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)          1_133.40       290.71     1_424.11       0.5114          1.0512            1.0493         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)          1_133.40       199.65     1_333.05       0.3837          1.0874            1.0870         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)          1_133.40       294.93     1_428.33       0.5114          1.0512            1.0493         4.49
IVF-Binary-512-nl223-random (self)                     1_133.40       450.26     1_583.67       0.3984          1.0905            1.0953         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)           1_541.70       111.12     1_652.81       0.1239          1.3798            1.3852         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)           1_541.70       111.62     1_653.32       0.1239          1.3798            1.3853         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)           1_541.70       115.16     1_656.86       0.1239          1.3798            1.3853         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)          1_541.70       200.19     1_741.89       0.3891          1.0851            1.0848         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)          1_541.70       299.35     1_841.05       0.5172          1.0499            1.0480         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)          1_541.70       199.43     1_741.13       0.3891          1.0851            1.0849         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)          1_541.70       296.78     1_838.48       0.5172          1.0499            1.0480         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)          1_541.70       203.10     1_744.80       0.3891          1.0851            1.0849         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)          1_541.70       303.06     1_844.76       0.5172          1.0499            1.0480         4.67
IVF-Binary-512-nl316-random (self)                     1_541.70       479.42     2_021.11       0.4039          1.0882            1.0930         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)               2_100.11        97.49     2_197.60       0.1080          1.4128            1.4259         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)              2_100.11        96.99     2_197.10       0.1080          1.4128            1.4259         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)              2_100.11        99.34     2_199.45       0.1080          1.4128            1.4259         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)              2_100.11       184.18     2_284.29       0.3593          1.0985            1.0956         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)              2_100.11       278.15     2_378.27       0.4867          1.0579            1.0543         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)             2_100.11       183.26     2_283.38       0.3591          1.0985            1.0956         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)             2_100.11       282.07     2_382.18       0.4864          1.0579            1.0543         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)             2_100.11       188.72     2_288.84       0.3591          1.0985            1.0956         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)             2_100.11       286.24     2_386.36       0.4864          1.0579            1.0543         4.36
IVF-Binary-512-nl158-pca (self)                        2_100.11       425.47     2_525.59       0.3768          1.0990            1.1039         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_231.75       101.97     1_333.73       0.1163          1.3949            1.4021         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_231.75       105.02     1_336.78       0.1163          1.3950            1.4021         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_231.75       109.12     1_340.87       0.1163          1.3950            1.4021         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_231.75       191.75     1_423.50       0.3725          1.0921            1.0901         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_231.75       295.47     1_527.22       0.5013          1.0541            1.0513         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_231.75       190.55     1_422.30       0.3723          1.0922            1.0902         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_231.75       289.71     1_521.46       0.5008          1.0541            1.0513         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_231.75       198.82     1_430.57       0.3723          1.0922            1.0902         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_231.75       294.24     1_525.99       0.5008          1.0541            1.0513         4.49
IVF-Binary-512-nl223-pca (self)                        1_231.75       453.39     1_685.15       0.3892          1.0940            1.0986         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)              1_622.30       110.10     1_732.40       0.1206          1.3855            1.3907         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)              1_622.30       110.66     1_732.96       0.1206          1.3855            1.3907         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)              1_622.30       115.99     1_738.29       0.1206          1.3856            1.3907         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)             1_622.30       199.58     1_821.87       0.3791          1.0892            1.0879         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)             1_622.30       295.20     1_917.50       0.5064          1.0526            1.0502         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)             1_622.30       198.31     1_820.61       0.3791          1.0892            1.0879         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)             1_622.30       298.52     1_920.82       0.5064          1.0526            1.0503         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)             1_622.30       202.45     1_824.74       0.3791          1.0892            1.0879         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)             1_622.30       305.66     1_927.96       0.5064          1.0526            1.0503         4.67
IVF-Binary-512-nl316-pca (self)                        1_622.30       480.13     2_102.43       0.3952          1.0911            1.0962         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)           2_059.72       146.14     2_205.86       0.1600          1.3234            1.3316         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)          2_059.72       149.85     2_209.57       0.1600          1.3234            1.3316         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)          2_059.72       151.88     2_211.60       0.1600          1.3234            1.3316         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)          2_059.72       238.67     2_298.39       0.4461          1.0658            1.0677         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)          2_059.72       344.17     2_403.89       0.5829          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)         2_059.72       240.96     2_300.68       0.4461          1.0658            1.0677         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)         2_059.72       354.08     2_413.80       0.5829          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)         2_059.72       245.51     2_305.22       0.4461          1.0658            1.0677         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)         2_059.72       357.39     2_417.11       0.5829          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-random (self)                    2_059.72       644.33     2_704.05       0.4602          1.0712            1.0740         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_210.74       156.13     1_366.86       0.1639          1.3163            1.3238         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_210.74       156.68     1_367.41       0.1638          1.3163            1.3240         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_210.74       162.74     1_373.48       0.1638          1.3163            1.3240         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_210.74       248.79     1_459.52       0.4524          1.0641            1.0657         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_210.74       351.39     1_562.13       0.5891          1.0359            1.0351         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_210.74       253.15     1_463.88       0.4522          1.0642            1.0657         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_210.74       360.65     1_571.38       0.5888          1.0359            1.0351         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_210.74       261.31     1_472.04       0.4522          1.0642            1.0657         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_210.74       361.05     1_571.79       0.5888          1.0359            1.0351         8.54
IVF-Binary-1024-nl223-random (self)                    1_210.74       670.80     1_881.53       0.4669          1.0693            1.0720         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)          1_589.73       167.48     1_757.22       0.1652          1.3134            1.3216         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)          1_589.73       170.62     1_760.35       0.1652          1.3134            1.3216         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)          1_589.73       170.24     1_759.97       0.1652          1.3134            1.3216         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)         1_589.73       265.24     1_854.98       0.4554          1.0631            1.0651         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)         1_589.73       362.95     1_952.68       0.5919          1.0353            1.0347         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)         1_589.73       255.67     1_845.41       0.4554          1.0631            1.0651         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)         1_589.73       364.48     1_954.21       0.5918          1.0353            1.0347         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)         1_589.73       263.85     1_853.59       0.4554          1.0631            1.0651         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)         1_589.73       374.72     1_964.45       0.5918          1.0353            1.0347         8.73
IVF-Binary-1024-nl316-random (self)                    1_589.73       712.13     2_301.87       0.4698          1.0684            1.0714         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)              2_162.34       147.77     2_310.11       0.1606          1.3228            1.3332         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)             2_162.34       149.16     2_311.50       0.1606          1.3228            1.3332         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)             2_162.34       153.85     2_316.19       0.1606          1.3228            1.3332         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)             2_162.34       240.03     2_402.37       0.4452          1.0657            1.0680         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)             2_162.34       337.38     2_499.72       0.5817          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)            2_162.34       239.52     2_401.86       0.4452          1.0657            1.0680         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)            2_162.34       345.28     2_507.62       0.5817          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)            2_162.34       246.34     2_408.68       0.4452          1.0657            1.0680         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)            2_162.34       351.34     2_513.68       0.5817          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-pca (self)                       2_162.34       652.46     2_814.80       0.4586          1.0715            1.0746         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_278.05       156.17     1_434.23       0.1643          1.3158            1.3260         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_278.05       161.27     1_439.32       0.1643          1.3159            1.3261         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_278.05       162.98     1_441.03       0.1643          1.3159            1.3261         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_278.05       249.87     1_527.92       0.4518          1.0638            1.0665         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_278.05       349.29     1_627.35       0.5876          1.0359            1.0354         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_278.05       252.78     1_530.83       0.4516          1.0638            1.0665         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_278.05       354.36     1_632.41       0.5873          1.0360            1.0354         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_278.05       256.61     1_534.66       0.4516          1.0638            1.0665         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_278.05       362.49     1_640.54       0.5873          1.0360            1.0354         8.54
IVF-Binary-1024-nl223-pca (self)                       1_278.05       665.36     1_943.42       0.4650          1.0696            1.0727         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_659.84       168.52     1_828.37       0.1659          1.3132            1.3237         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_659.84       165.78     1_825.62       0.1658          1.3132            1.3237         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_659.84       170.57     1_830.42       0.1658          1.3132            1.3237         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_659.84       258.24     1_918.09       0.4542          1.0631            1.0656         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_659.84       361.91     2_021.75       0.5902          1.0355            1.0350         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_659.84       259.91     1_919.75       0.4542          1.0631            1.0656         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_659.84       363.76     2_023.61       0.5902          1.0355            1.0350         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_659.84       264.58     1_924.42       0.4542          1.0631            1.0656         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_659.84       376.59     2_036.43       0.5902          1.0355            1.0350         8.73
IVF-Binary-1024-nl316-pca (self)                       1_659.84       698.58     2_358.42       0.4676          1.0689            1.0721         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)              1_916.25       356.60     2_272.85       0.0832         14.7176            9.4099         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)             1_916.25       384.39     2_300.63       0.0789         21.5952            9.5065         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)             1_916.25       424.11     2_340.36       0.0780         25.6575           10.1784         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)             1_916.25       402.71     2_318.96       0.7193          1.0353            1.0120         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)             1_916.25       706.63     2_622.88       0.9183          1.0087            1.0000         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)            1_916.25       428.96     2_345.21       0.5640          1.8556            1.0321         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)            1_916.25       732.84     2_649.09       0.8684          1.0153            1.0014         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)            1_916.25       460.31     2_376.55       0.4582          2.4667            1.2314         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)            1_916.25       763.13     2_679.37       0.8109          1.0246            1.0040         3.36
IVF-Binary-512-nl158-sign (self)                       1_916.25     1_227.92     3_144.17       0.6094          1.5816            1.0265         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)             1_028.55       383.88     1_412.43       0.0919         13.2324           10.5118         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)             1_028.55       406.65     1_435.20       0.0763         21.7864           12.6025         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)             1_028.55       451.76     1_480.31       0.0742         46.0116           20.1871         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)            1_028.55       438.01     1_466.56       0.5906          1.6001            1.0254         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)            1_028.55       735.35     1_763.90       0.8753          1.0121            1.0014         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)            1_028.55       460.61     1_489.16       0.4954          1.9272            1.0715         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)            1_028.55       753.89     1_782.44       0.8341          1.0168            1.0032         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)            1_028.55       496.24     1_524.79       0.3813          2.3737            1.3856         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)            1_028.55       802.70     1_831.25       0.7400          1.0313            1.0091         3.49
IVF-Binary-512-nl223-sign (self)                       1_028.55     1_319.13     2_347.68       0.5443          1.7695            1.0496         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)             1_414.73       418.78     1_833.51       0.0962         15.1116           11.1304         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)             1_414.73       426.60     1_841.33       0.0820         17.8246           12.2406         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)             1_414.73       476.85     1_891.58       0.0755         45.1389           20.0100         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)            1_414.73       470.11     1_884.84       0.6068          1.2492            1.0246         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)            1_414.73       770.58     2_185.31       0.8691          1.0132            1.0016         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)            1_414.73       478.40     1_893.12       0.5634          1.3163            1.0368         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)            1_414.73       781.27     2_196.00       0.8488          1.0156            1.0025         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)            1_414.73       527.02     1_941.75       0.4351          1.6693            1.1970         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)            1_414.73       835.86     2_250.59       0.7655          1.0273            1.0072         3.67
IVF-Binary-512-nl316-sign (self)                       1_414.73     1_398.82     2_813.55       0.6093          1.1744            1.0301         3.67
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
Exhaustive (query)                                        99.98     1_934.65     2_034.64       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.98     6_987.83     7_087.81       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                205.99       319.18       525.17       0.0663          1.3769            1.3785         2.28
ExhaustiveBinary-256-random-rf10 (query)                 205.99       461.72       667.72       0.2744          1.1112            1.1016         2.28
ExhaustiveBinary-256-random-rf20 (query)                 205.99       642.85       848.85       0.3881          1.0705            1.0589         2.28
ExhaustiveBinary-256-random (self)                       205.99     1_301.39     1_507.38       0.2875          1.1076            1.0993         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   406.07       283.02       689.09       0.0653          1.3792            1.3768         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    406.07       424.86       830.93       0.2700          1.1137            1.1016         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    406.07       586.37       992.44       0.3853          1.0725            1.0585         2.28
ExhaustiveBinary-256-pca (self)                          406.07     1_299.72     1_705.79       0.2821          1.1099            1.0989         2.28
ExhaustiveBinary-512-random_no_rr (query)                297.44       414.16       711.60       0.0935          1.3249            1.3315         4.55
ExhaustiveBinary-512-random-rf10 (query)                 297.44       561.97       859.41       0.3218          1.0860            1.0805         4.55
ExhaustiveBinary-512-random-rf20 (query)                 297.44       731.12     1_028.56       0.4346          1.0526            1.0484         4.55
ExhaustiveBinary-512-random (self)                       297.44     1_779.51     2_076.95       0.3345          1.0826            1.0842         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   493.72       425.96       919.68       0.0950          1.3226            1.3263         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    493.72       589.02     1_082.74       0.3247          1.0847            1.0787         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    493.72       736.45     1_230.17       0.4402          1.0516            1.0467         4.55
ExhaustiveBinary-512-pca (self)                          493.72     1_858.91     2_352.63       0.3363          1.0818            1.0825         4.55
ExhaustiveBinary-1024-random_no_rr (query)               497.77       631.44     1_129.21       0.1319          1.2684            1.2722         9.11
ExhaustiveBinary-1024-random-rf10 (query)                497.77       806.73     1_304.50       0.3744          1.0643            1.0666         9.11
ExhaustiveBinary-1024-random-rf20 (query)                497.77       978.46     1_476.23       0.4905          1.0385            1.0390         9.11
ExhaustiveBinary-1024-random (self)                      497.77     2_623.74     3_121.51       0.3822          1.0665            1.0714         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  702.42       635.98     1_338.39       0.1354          1.2622            1.2650         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   702.42       840.15     1_542.57       0.3805          1.0621            1.0642         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   702.42       987.37     1_689.79       0.4993          1.0369            1.0373         9.11
ExhaustiveBinary-1024-pca (self)                         702.42     2_656.45     3_358.87       0.3869          1.0651            1.0694         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  128.11       854.20       982.31       0.1284          1.2821            1.2820         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   128.11       941.37     1_069.48       0.3615          1.0706            1.0698         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   128.11     1_457.11     1_585.22       0.4846          1.0406            1.0394         4.58
ExhaustiveBinary-768-sign (self)                         128.11     3_027.61     3_155.72       0.3693          1.0715            1.0747         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)            2_867.29        96.16     2_963.45       0.0686          1.3707            1.3770         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)           2_867.29        97.31     2_964.60       0.0686          1.3709            1.3770         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)           2_867.29       101.60     2_968.89       0.0686          1.3709            1.3770         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)           2_867.29       196.00     3_063.29       0.2782          1.1101            1.1013         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)           2_867.29       310.23     3_177.52       0.3911          1.0699            1.0588         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)          2_867.29       194.49     3_061.78       0.2776          1.1102            1.1013         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)          2_867.29       314.54     3_181.83       0.3904          1.0699            1.0588         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)          2_867.29       201.18     3_068.47       0.2776          1.1102            1.1013         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)          2_867.29       314.98     3_182.27       0.3903          1.0699            1.0588         2.74
IVF-Binary-256-nl158-random (self)                     2_867.29       416.06     3_283.35       0.2908          1.1066            1.0991         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)           1_506.06       103.80     1_609.86       0.0796          1.3505            1.3543         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)           1_506.06        99.61     1_605.67       0.0796          1.3505            1.3543         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)           1_506.06       101.91     1_607.97       0.0796          1.3505            1.3543         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)          1_506.06       205.80     1_711.86       0.3011          1.0957            1.0874         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)          1_506.06       319.21     1_825.27       0.4149          1.0602            1.0529         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)          1_506.06       207.88     1_713.94       0.3010          1.0957            1.0874         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)          1_506.06       327.50     1_833.56       0.4148          1.0602            1.0529         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)          1_506.06       210.48     1_716.54       0.3010          1.0957            1.0874         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)          1_506.06       330.64     1_836.70       0.4148          1.0602            1.0529         2.93
IVF-Binary-256-nl223-random (self)                     1_506.06       445.93     1_951.99       0.3144          1.0916            1.0890         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)           1_992.66       109.93     2_102.59       0.0861          1.3377            1.3378         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)           1_992.66       109.22     2_101.88       0.0861          1.3379            1.3378         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)           1_992.66       115.20     2_107.86       0.0861          1.3380            1.3378         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)          1_992.66       219.60     2_212.26       0.3162          1.0878            1.0817         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)          1_992.66       335.90     2_328.56       0.4292          1.0555            1.0496         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)          1_992.66       215.70     2_208.36       0.3161          1.0878            1.0817         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)          1_992.66       342.49     2_335.15       0.4291          1.0555            1.0496         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)          1_992.66       221.69     2_214.36       0.3161          1.0878            1.0817         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)          1_992.66       342.16     2_334.83       0.4291          1.0555            1.0496         3.21
IVF-Binary-256-nl316-random (self)                     1_992.66       485.28     2_477.94       0.3289          1.0828            1.0840         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)               3_023.08        87.06     3_110.14       0.0679          1.3724            1.3752         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)              3_023.08        89.24     3_112.32       0.0678          1.3724            1.3752         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)              3_023.08        90.22     3_113.30       0.0678          1.3724            1.3752         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)              3_023.08       196.91     3_219.99       0.2742          1.1122            1.1011         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)              3_023.08       308.39     3_331.47       0.3887          1.0715            1.0583         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)             3_023.08       190.25     3_213.33       0.2734          1.1122            1.1011         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)             3_023.08       311.25     3_334.33       0.3876          1.0715            1.0583         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)             3_023.08       193.55     3_216.63       0.2734          1.1122            1.1011         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)             3_023.08       313.24     3_336.32       0.3875          1.0715            1.0583         2.74
IVF-Binary-256-nl158-pca (self)                        3_023.08       412.60     3_435.68       0.2854          1.1085            1.0987         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)              1_712.80        97.49     1_810.29       0.0779          1.3524            1.3537         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)              1_712.80        98.41     1_811.21       0.0779          1.3525            1.3538         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)              1_712.80       101.69     1_814.48       0.0779          1.3526            1.3538         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)             1_712.80       208.24     1_921.03       0.2999          1.0953            1.0866         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)             1_712.80       323.03     2_035.83       0.4159          1.0595            1.0517         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)             1_712.80       203.90     1_916.70       0.2999          1.0953            1.0866         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)             1_712.80       324.63     2_037.42       0.4158          1.0595            1.0517         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)             1_712.80       208.02     1_920.82       0.2998          1.0953            1.0866         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)             1_712.80       327.55     2_040.35       0.4158          1.0595            1.0517         2.93
IVF-Binary-256-nl223-pca (self)                        1_712.80       437.97     2_150.77       0.3125          1.0897            1.0883         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)              2_209.56       112.16     2_321.72       0.0853          1.3390            1.3375         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)              2_209.56       110.38     2_319.94       0.0853          1.3392            1.3375         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)              2_209.56       112.23     2_321.79       0.0853          1.3393            1.3375         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)             2_209.56       220.14     2_429.69       0.3114          1.0893            1.0816         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)             2_209.56       334.58     2_544.14       0.4247          1.0567            1.0498         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)             2_209.56       215.65     2_425.21       0.3113          1.0893            1.0816         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)             2_209.56       336.74     2_546.30       0.4246          1.0567            1.0498         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)             2_209.56       218.27     2_427.83       0.3112          1.0893            1.0816         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)             2_209.56       338.14     2_547.69       0.4246          1.0567            1.0498         3.21
IVF-Binary-256-nl316-pca (self)                        2_209.56       495.84     2_705.40       0.3229          1.0838            1.0843         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)            2_929.44       123.42     3_052.86       0.0948          1.3226            1.3308         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)           2_929.44       125.45     3_054.89       0.0948          1.3226            1.3308         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)           2_929.44       127.63     3_057.07       0.0948          1.3226            1.3308         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)           2_929.44       250.29     3_179.73       0.3234          1.0856            1.0804         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)           2_929.44       354.74     3_284.18       0.4358          1.0524            1.0484         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)          2_929.44       233.59     3_163.02       0.3232          1.0857            1.0804         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)          2_929.44       361.21     3_290.65       0.4357          1.0524            1.0484         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)          2_929.44       236.32     3_165.76       0.3232          1.0857            1.0804         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)          2_929.44       364.64     3_294.08       0.4357          1.0524            1.0484         5.02
IVF-Binary-512-nl158-random (self)                     2_929.44       564.01     3_493.44       0.3358          1.0822            1.0841         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)           1_599.63       137.16     1_736.79       0.1031          1.3086            1.3103         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)           1_599.63       135.24     1_734.87       0.1031          1.3087            1.3103         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)           1_599.63       145.60     1_745.23       0.1031          1.3087            1.3103         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)          1_599.63       250.76     1_850.39       0.3366          1.0789            1.0765         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)          1_599.63       370.07     1_969.70       0.4487          1.0486            1.0459         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)          1_599.63       245.47     1_845.10       0.3366          1.0789            1.0765         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)          1_599.63       377.80     1_977.43       0.4486          1.0486            1.0459         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)          1_599.63       251.57     1_851.20       0.3366          1.0789            1.0765         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)          1_599.63       385.91     1_985.54       0.4486          1.0486            1.0459         5.21
IVF-Binary-512-nl223-random (self)                     1_599.63       602.97     2_202.60       0.3478          1.0763            1.0802         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)           2_103.13       148.41     2_251.54       0.1078          1.3004            1.2995         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)           2_103.13       147.36     2_250.49       0.1078          1.3004            1.2995         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)           2_103.13       151.19     2_254.31       0.1078          1.3004            1.2995         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)          2_103.13       257.42     2_360.55       0.3423          1.0763            1.0744         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)          2_103.13       383.17     2_486.30       0.4552          1.0469            1.0448         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)          2_103.13       256.88     2_360.01       0.3423          1.0763            1.0744         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)          2_103.13       395.54     2_498.67       0.4551          1.0469            1.0448         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)          2_103.13       259.79     2_362.92       0.3423          1.0763            1.0744         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)          2_103.13       389.55     2_492.68       0.4551          1.0469            1.0448         5.48
IVF-Binary-512-nl316-random (self)                     2_103.13       661.14     2_764.27       0.3533          1.0740            1.0786         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)               3_115.62       130.20     3_245.82       0.0961          1.3203            1.3252         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)              3_115.62       133.61     3_249.23       0.0960          1.3204            1.3252         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)              3_115.62       129.65     3_245.27       0.0960          1.3204            1.3252         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)              3_115.62       241.77     3_357.38       0.3261          1.0844            1.0786         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)              3_115.62       357.29     3_472.91       0.4410          1.0514            1.0466         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)             3_115.62       241.92     3_357.54       0.3259          1.0844            1.0786         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)             3_115.62       359.71     3_475.33       0.4409          1.0515            1.0466         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)             3_115.62       240.74     3_356.36       0.3259          1.0844            1.0786         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)             3_115.62       364.90     3_480.52       0.4409          1.0515            1.0466         5.02
IVF-Binary-512-nl158-pca (self)                        3_115.62       567.96     3_683.57       0.3373          1.0815            1.0824         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_807.78       155.62     1_963.40       0.1049          1.3059            1.3063         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_807.78       136.40     1_944.17       0.1049          1.3059            1.3063         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_807.78       140.27     1_948.04       0.1049          1.3059            1.3063         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_807.78       249.32     2_057.09       0.3401          1.0771            1.0744         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_807.78       368.70     2_176.48       0.4552          1.0470            1.0443         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_807.78       245.81     2_053.59       0.3401          1.0771            1.0744         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_807.78       373.71     2_181.48       0.4551          1.0470            1.0443         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_807.78       249.98     2_057.76       0.3401          1.0771            1.0744         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_807.78       377.64     2_185.41       0.4551          1.0470            1.0443         5.21
IVF-Binary-512-nl223-pca (self)                        1_807.78       607.92     2_415.70       0.3502          1.0752            1.0788         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)              2_321.79       148.75     2_470.54       0.1090          1.2980            1.2945         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)              2_321.79       149.25     2_471.04       0.1090          1.2980            1.2945         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)              2_321.79       152.21     2_474.00       0.1090          1.2980            1.2945         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)             2_321.79       258.43     2_580.22       0.3442          1.0753            1.0733         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)             2_321.79       393.77     2_715.56       0.4593          1.0460            1.0437         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)             2_321.79       256.30     2_578.08       0.3442          1.0753            1.0733         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)             2_321.79       386.88     2_708.67       0.4593          1.0460            1.0437         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)             2_321.79       260.93     2_582.72       0.3441          1.0753            1.0733         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)             2_321.79       391.61     2_713.40       0.4593          1.0460            1.0437         5.48
IVF-Binary-512-nl316-pca (self)                        2_321.79       649.92     2_971.71       0.3540          1.0738            1.0776         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)           3_139.87       203.42     3_343.28       0.1326          1.2679            1.2721         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)          3_139.87       198.77     3_338.64       0.1326          1.2679            1.2721         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)          3_139.87       203.82     3_343.69       0.1326          1.2679            1.2721         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)          3_139.87       326.99     3_466.85       0.3747          1.0643            1.0666         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)          3_139.87       452.26     3_592.13       0.4907          1.0385            1.0390         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)         3_139.87       329.29     3_469.15       0.3747          1.0643            1.0666         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)         3_139.87       461.51     3_601.38       0.4907          1.0385            1.0390         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)         3_139.87       347.75     3_487.61       0.3747          1.0643            1.0666         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)         3_139.87       465.19     3_605.06       0.4907          1.0385            1.0390         9.57
IVF-Binary-1024-nl158-random (self)                    3_139.87       863.03     4_002.89       0.3824          1.0665            1.0714         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_826.09       211.92     2_038.01       0.1371          1.2606            1.2655         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_826.09       211.67     2_037.77       0.1371          1.2606            1.2655         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_826.09       216.11     2_042.21       0.1371          1.2606            1.2655         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_826.09       351.85     2_177.94       0.3811          1.0620            1.0649         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_826.09       488.29     2_314.38       0.4978          1.0372            1.0379         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_826.09       339.76     2_165.85       0.3811          1.0620            1.0649         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_826.09       471.80     2_297.90       0.4978          1.0372            1.0379         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_826.09       351.05     2_177.14       0.3811          1.0620            1.0649         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_826.09       503.72     2_329.81       0.4978          1.0372            1.0379         9.76
IVF-Binary-1024-nl223-random (self)                    1_826.09       918.08     2_744.17       0.3894          1.0644            1.0694         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)          2_307.70       226.83     2_534.54       0.1391          1.2573            1.2620        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)          2_307.70       224.88     2_532.58       0.1391          1.2573            1.2620        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)          2_307.70       232.63     2_540.33       0.1391          1.2573            1.2620        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)         2_307.70       358.07     2_665.78       0.3841          1.0612            1.0643        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)         2_307.70       483.88     2_791.59       0.5002          1.0368            1.0373        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)         2_307.70       357.28     2_664.98       0.3841          1.0612            1.0643        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)         2_307.70       484.65     2_792.36       0.5002          1.0368            1.0373        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)         2_307.70       363.65     2_671.36       0.3841          1.0612            1.0643        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)         2_307.70       496.47     2_804.18       0.5002          1.0368            1.0373        10.04
IVF-Binary-1024-nl316-random (self)                    2_307.70       963.94     3_271.64       0.3917          1.0637            1.0687        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)              3_356.92       203.19     3_560.11       0.1360          1.2617            1.2649         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)             3_356.92       202.49     3_559.41       0.1360          1.2617            1.2649         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)             3_356.92       204.60     3_561.52       0.1360          1.2617            1.2649         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)             3_356.92       323.67     3_680.59       0.3808          1.0621            1.0642         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)             3_356.92       451.60     3_808.52       0.4995          1.0368            1.0373         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)            3_356.92       329.36     3_686.28       0.3807          1.0621            1.0642         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)            3_356.92       458.35     3_815.27       0.4995          1.0368            1.0373         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)            3_356.92       334.77     3_691.69       0.3807          1.0621            1.0642         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)            3_356.92       466.31     3_823.23       0.4995          1.0368            1.0373         9.57
IVF-Binary-1024-nl158-pca (self)                       3_356.92       881.23     4_238.15       0.3872          1.0650            1.0694         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)             2_044.51       222.05     2_266.56       0.1397          1.2559            1.2601         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)             2_044.51       215.30     2_259.81       0.1397          1.2559            1.2601         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)             2_044.51       215.58     2_260.09       0.1397          1.2559            1.2601         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)            2_044.51       343.64     2_388.16       0.3868          1.0600            1.0627         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)            2_044.51       479.91     2_524.42       0.5062          1.0357            1.0364         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)            2_044.51       344.12     2_388.63       0.3868          1.0600            1.0627         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)            2_044.51       471.87     2_516.39       0.5062          1.0357            1.0364         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)            2_044.51       370.00     2_414.51       0.3868          1.0600            1.0627         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)            2_044.51       481.04     2_525.55       0.5062          1.0357            1.0364         9.76
IVF-Binary-1024-nl223-pca (self)                       2_044.51       912.15     2_956.67       0.3931          1.0633            1.0678         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             2_525.43       225.96     2_751.38       0.1412          1.2532            1.2577        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)             2_525.43       224.34     2_749.77       0.1412          1.2532            1.2577        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)             2_525.43       228.58     2_754.01       0.1412          1.2532            1.2577        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)            2_525.43       359.81     2_885.23       0.3890          1.0594            1.0625        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)            2_525.43       489.13     3_014.55       0.5082          1.0353            1.0360        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)            2_525.43       358.44     2_883.87       0.3890          1.0594            1.0625        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)            2_525.43       487.85     3_013.28       0.5082          1.0353            1.0360        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)            2_525.43       368.25     2_893.68       0.3890          1.0594            1.0625        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)            2_525.43       501.24     3_026.66       0.5082          1.0353            1.0360        10.04
IVF-Binary-1024-nl316-pca (self)                       2_525.43       987.89     3_513.31       0.3952          1.0627            1.0672        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)              2_802.70       497.41     3_300.11       0.0807         14.0287           14.3094         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)             2_802.70       546.87     3_349.57       0.0791         14.3242           14.3173         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)             2_802.70       582.59     3_385.28       0.0790         14.5580           14.3622         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)             2_802.70       556.28     3_358.97       0.6167          1.5913            1.0166         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)             2_802.70       983.80     3_786.50       0.8955          1.0066            1.0006         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)            2_802.70       591.94     3_394.64       0.4039          4.5022            1.1738         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)            2_802.70     1_077.84     3_880.53       0.8247          1.0134            1.0027         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)            2_802.70       659.64     3_462.34       0.3009          6.8037            3.5768         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)            2_802.70     1_064.65     3_867.34       0.7511          1.0223            1.0059         5.04
IVF-Binary-768-nl158-sign (self)                       2_802.70     1_709.80     4_512.50       0.4381          4.1636            1.0685         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)             1_457.63       534.86     1_992.49       0.0741         46.2758           15.5306         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)             1_457.63       562.69     2_020.32       0.0735         56.4000           15.9839         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)             1_457.63       644.48     2_102.11       0.0726         95.4891           21.2890         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)            1_457.63       602.07     2_059.70       0.4705          2.8757            1.0332         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)            1_457.63     1_019.30     2_476.93       0.8165          1.0159            1.0024         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)            1_457.63       624.77     2_082.40       0.3425          3.6055            1.3437         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)            1_457.63     1_055.01     2_512.63       0.7570          1.0272            1.0050         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)            1_457.63       723.68     2_181.31       0.2384          4.1964            3.0932         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)            1_457.63     1_120.37     2_577.99       0.6331          1.2819            1.0140         5.23
IVF-Binary-768-nl223-sign (self)                       1_457.63     1_825.04     3_282.67       0.3683          3.7108            1.2403         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)             1_962.75       586.44     2_549.19       0.0745         38.1145           14.6740         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)             1_962.75       596.33     2_559.08       0.0739         40.1604           17.7505         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)             1_962.75       671.59     2_634.34       0.0728         80.6407           24.6782         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)            1_962.75       651.51     2_614.25       0.4920          2.3394            1.0321         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)            1_962.75     1_084.04     3_046.79       0.8126          1.0144            1.0027         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)            1_962.75       662.13     2_624.88       0.4334          2.5222            1.0631         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)            1_962.75     1_111.40     3_074.15       0.7853          1.0171            1.0038         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)            1_962.75       727.52     2_690.27       0.2899          3.1950            1.6078         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)            1_962.75     1_160.22     3_122.97       0.6708          1.3531            1.0104         5.51
IVF-Binary-768-nl316-sign (self)                       1_962.75     1_958.53     3_921.28       0.4575          2.5731            1.0511         5.51
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
Exhaustive (query)                                        33.31       738.44       771.75       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.31     2_607.95     2_641.25       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                 79.37       284.63       364.00       0.5519          1.8826            1.5884         1.78
ExhaustiveBinary-256-random-rf10 (query)                  79.37       432.64       512.01       0.9881          1.0022            1.0000         1.78
ExhaustiveBinary-256-random-rf20 (query)                  79.37       546.21       625.58       0.9980          1.0003            1.0000         1.78
ExhaustiveBinary-256-random (self)                        79.37     1_347.22     1_426.59       0.9881          1.0022            1.0000         1.78
ExhaustiveBinary-256-pca_no_rr (query)                   108.38       249.16       357.54       0.5930          1.6081            1.4152         1.78
ExhaustiveBinary-256-pca-rf10 (query)                    108.38       391.36       499.74       0.9919          1.0013            1.0000         1.78
ExhaustiveBinary-256-pca-rf20 (query)                    108.38       515.79       624.17       0.9988          1.0001            1.0000         1.78
ExhaustiveBinary-256-pca (self)                          108.38     1_364.99     1_473.37       0.9915          1.0014            1.0000         1.78
ExhaustiveBinary-512-random_no_rr (query)                 92.79       366.92       459.71       0.6306          1.5767            1.3633         3.55
ExhaustiveBinary-512-random-rf10 (query)                  92.79       512.51       605.30       0.9975          1.0004            1.0000         3.55
ExhaustiveBinary-512-random-rf20 (query)                  92.79       652.39       745.18       0.9998          1.0000            1.0000         3.55
ExhaustiveBinary-512-random (self)                        92.79     1_613.15     1_705.94       0.9972          1.0004            1.0000         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   112.01       365.12       477.13       0.6479          1.4884            1.3147         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    112.01       492.48       604.49       0.9983          1.0002            1.0000         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    112.01       621.05       733.06       0.9998          1.0000            1.0000         3.55
ExhaustiveBinary-512-pca (self)                          112.01     1_613.66     1_725.67       0.9981          1.0002            1.0000         3.55
ExhaustiveBinary-1024-random_no_rr (query)               118.05       558.28       676.34       0.6758          1.4452            1.2804         7.10
ExhaustiveBinary-1024-random-rf10 (query)                118.05       671.72       789.77       0.9995          1.0001            1.0000         7.10
ExhaustiveBinary-1024-random-rf20 (query)                118.05       790.83       908.89       0.9999          1.0000            1.0000         7.10
ExhaustiveBinary-1024-random (self)                      118.05     2_235.04     2_353.09       0.9993          1.0001            1.0000         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  141.94       537.70       679.64       0.6838          1.4142            1.2651         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   141.94       670.28       812.23       0.9996          1.0000            1.0000         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   141.94       787.81       929.76       1.0000          1.0000            1.0000         7.10
ExhaustiveBinary-1024-pca (self)                         141.94     2_200.53     2_342.47       0.9995          1.0001            1.0000         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   44.33       439.21       483.54       0.0376         19.4734           14.8778         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    44.33       480.07       524.41       0.1617          2.7567            2.6548         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    44.33       743.48       787.81       0.2739          1.9837            1.9249         1.53
ExhaustiveBinary-256-sign (self)                          44.33     1_548.42     1_592.75       0.1691          2.7353            2.6299         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)            1_247.95        66.39     1_314.34       0.5644          1.6745            1.5171         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           1_247.95        77.92     1_325.87       0.5580          1.7410            1.5527         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           1_247.95        80.49     1_328.44       0.5562          1.7768            1.5630         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           1_247.95       124.32     1_372.27       0.9901          1.0018            1.0000         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           1_247.95       179.11     1_427.06       0.9967          1.0006            1.0000         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          1_247.95       136.02     1_383.97       0.9903          1.0017            1.0000         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          1_247.95       190.56     1_438.51       0.9985          1.0002            1.0000         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          1_247.95       138.96     1_386.91       0.9894          1.0019            1.0000         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          1_247.95       200.87     1_448.82       0.9984          1.0002            1.0000         1.93
IVF-Binary-256-nl158-random (self)                     1_247.95       347.25     1_595.20       0.9902          1.0017            1.0000         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             534.76        54.96       589.72       0.5623          1.6900            1.5271         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             534.76        58.80       593.56       0.5598          1.7203            1.5426         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             534.76        65.57       600.33       0.5573          1.7614            1.5583         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            534.76       118.06       652.82       0.9910          1.0015            1.0000         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            534.76       176.40       711.16       0.9983          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            534.76       123.47       658.23       0.9906          1.0016            1.0000         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            534.76       184.19       718.95       0.9986          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            534.76       130.22       664.98       0.9896          1.0018            1.0000         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            534.76       189.90       724.66       0.9984          1.0002            1.0000         2.00
IVF-Binary-256-nl223-random (self)                       534.76       313.63       848.39       0.9904          1.0017            1.0000         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             690.27        58.79       749.07       0.5622          1.6843            1.5282         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             690.27        59.65       749.92       0.5610          1.6983            1.5368         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             690.27        69.38       759.65       0.5582          1.7399            1.5536         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            690.27       120.47       810.74       0.9911          1.0016            1.0000         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            690.27       178.08       868.36       0.9986          1.0002            1.0000         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            690.27       121.00       811.27       0.9908          1.0016            1.0000         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            690.27       177.74       868.01       0.9986          1.0002            1.0000         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            690.27       138.17       828.44       0.9899          1.0018            1.0000         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            690.27       187.22       877.49       0.9985          1.0002            1.0000         2.09
IVF-Binary-256-nl316-random (self)                       690.27       308.55       998.83       0.9908          1.0016            1.0000         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               1_243.71        52.06     1_295.77       0.6035          1.4885            1.3774         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              1_243.71        60.91     1_304.62       0.5986          1.5276            1.3951         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              1_243.71        68.43     1_312.14       0.5972          1.5500            1.4004         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              1_243.71       121.74     1_365.45       0.9925          1.0013            1.0000         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              1_243.71       182.25     1_425.96       0.9970          1.0006            1.0000         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             1_243.71       132.34     1_376.05       0.9933          1.0010            1.0000         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             1_243.71       194.61     1_438.32       0.9991          1.0001            1.0000         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             1_243.71       147.81     1_391.52       0.9927          1.0012            1.0000         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             1_243.71       203.38     1_447.09       0.9990          1.0001            1.0000         1.93
IVF-Binary-256-nl158-pca (self)                        1_243.71       344.03     1_587.74       0.9927          1.0012            1.0000         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                585.19        55.93       641.12       0.6013          1.5006            1.3805         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                585.19        65.88       651.07       0.5995          1.5166            1.3884         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                585.19        65.28       650.48       0.5977          1.5397            1.3961         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               585.19       118.62       703.81       0.9933          1.0010            1.0000         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               585.19       181.68       766.88       0.9986          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               585.19       121.22       706.42       0.9933          1.0010            1.0000         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               585.19       177.22       762.41       0.9990          1.0001            1.0000         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               585.19       133.30       718.49       0.9928          1.0012            1.0000         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               585.19       200.03       785.22       0.9989          1.0001            1.0000         2.00
IVF-Binary-256-nl223-pca (self)                          585.19       315.55       900.74       0.9928          1.0012            1.0000         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                738.40        57.79       796.19       0.6014          1.4982            1.3803         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                738.40        59.42       797.82       0.6004          1.5067            1.3842         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                738.40        66.70       805.10       0.5985          1.5283            1.3915         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               738.40       123.71       862.11       0.9936          1.0010            1.0000         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               738.40       177.31       915.71       0.9989          1.0001            1.0000         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               738.40       123.94       862.34       0.9934          1.0010            1.0000         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               738.40       184.20       922.60       0.9990          1.0001            1.0000         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               738.40       130.50       868.90       0.9929          1.0011            1.0000         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               738.40       185.70       924.10       0.9990          1.0001            1.0000         2.09
IVF-Binary-256-nl316-pca (self)                          738.40       306.96     1_045.36       0.9931          1.0011            1.0000         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            1_202.48        72.18     1_274.66       0.6403          1.4492            1.3343         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           1_202.48        88.44     1_290.92       0.6347          1.4946            1.3509         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           1_202.48       100.03     1_302.51       0.6329          1.5211            1.3568         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           1_202.48       141.85     1_344.33       0.9963          1.0007            1.0000         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           1_202.48       196.40     1_398.87       0.9976          1.0005            1.0000         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          1_202.48       158.23     1_360.71       0.9980          1.0003            1.0000         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          1_202.48       216.50     1_418.97       0.9998          1.0000            1.0000         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          1_202.48       169.15     1_371.62       0.9978          1.0003            1.0000         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          1_202.48       232.85     1_435.32       0.9998          1.0000            1.0000         3.71
IVF-Binary-512-nl158-random (self)                     1_202.48       431.60     1_634.07       0.9978          1.0003            1.0000         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             549.58        73.51       623.09       0.6379          1.4648            1.3384         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             549.58        78.57       628.15       0.6357          1.4848            1.3474         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             549.58        89.62       639.20       0.6336          1.5131            1.3536         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            549.58       140.60       690.18       0.9976          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            549.58       208.71       758.29       0.9993          1.0001            1.0000         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            549.58       143.29       692.87       0.9979          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            549.58       217.46       767.04       0.9997          1.0000            1.0000         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            549.58       164.07       713.65       0.9978          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            549.58       217.91       767.48       0.9998          1.0000            1.0000         3.77
IVF-Binary-512-nl223-random (self)                       549.58       413.24       962.82       0.9977          1.0003            1.0000         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             703.66        75.95       779.61       0.6377          1.4619            1.3395         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             703.66        79.74       783.40       0.6368          1.4709            1.3425         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             703.66        91.17       794.83       0.6344          1.4972            1.3509         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            703.66       142.97       846.63       0.9979          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            703.66       202.52       906.18       0.9996          1.0001            1.0000         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            703.66       143.20       846.86       0.9980          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            703.66       201.59       905.25       0.9997          1.0000            1.0000         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            703.66       154.46       858.12       0.9978          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            703.66       222.27       925.93       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-random (self)                       703.66       387.24     1_090.90       0.9978          1.0003            1.0000         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               1_335.64        73.40     1_409.04       0.6570          1.3904            1.2913         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              1_335.64        91.92     1_427.56       0.6520          1.4262            1.3049         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              1_335.64       103.09     1_438.73       0.6506          1.4461            1.3093         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              1_335.64       141.62     1_477.27       0.9967          1.0006            1.0000         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              1_335.64       212.88     1_548.52       0.9976          1.0005            1.0000         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             1_335.64       158.13     1_493.77       0.9986          1.0002            1.0000         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             1_335.64       227.74     1_563.39       0.9998          1.0000            1.0000         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             1_335.64       174.70     1_510.35       0.9984          1.0002            1.0000         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             1_335.64       233.10     1_568.74       0.9999          1.0000            1.0000         3.71
IVF-Binary-512-nl158-pca (self)                        1_335.64       433.35     1_769.00       0.9985          1.0002            1.0000         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                617.23        75.03       692.26       0.6548          1.4025            1.2929         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                617.23        80.98       698.21       0.6527          1.4187            1.3006         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                617.23        99.68       716.91       0.6509          1.4388            1.3062         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               617.23       149.15       766.38       0.9982          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               617.23       201.10       818.33       0.9993          1.0001            1.0000         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               617.23       153.38       770.61       0.9985          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               617.23       209.72       826.95       0.9997          1.0000            1.0000         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               617.23       155.97       773.20       0.9984          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               617.23       225.31       842.54       0.9999          1.0000            1.0000         3.77
IVF-Binary-512-nl223-pca (self)                          617.23       402.57     1_019.80       0.9984          1.0002            1.0000         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                748.60        75.97       824.57       0.6540          1.4029            1.2949         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                748.60        78.36       826.96       0.6531          1.4103            1.2970         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                748.60        91.44       840.04       0.6511          1.4312            1.3034         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               748.60       142.87       891.47       0.9985          1.0002            1.0000         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               748.60       198.41       947.01       0.9996          1.0001            1.0000         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               748.60       142.73       891.33       0.9985          1.0002            1.0000         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               748.60       201.76       950.36       0.9997          1.0000            1.0000         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               748.60       154.86       903.46       0.9985          1.0002            1.0000         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               748.60       227.49       976.08       0.9999          1.0000            1.0000         3.86
IVF-Binary-512-nl316-pca (self)                          748.60       400.49     1_149.09       0.9985          1.0002            1.0000         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)           1_270.45       116.45     1_386.90       0.6843          1.3554            1.2564         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)          1_270.45       129.70     1_400.15       0.6791          1.3904            1.2728         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)          1_270.45       143.94     1_414.39       0.6775          1.4122            1.2763         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)          1_270.45       183.81     1_454.25       0.9974          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)          1_270.45       238.05     1_508.50       0.9976          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)         1_270.45       198.57     1_469.01       0.9996          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)         1_270.45       261.32     1_531.77       0.9999          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)         1_270.45       220.99     1_491.43       0.9996          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)         1_270.45       288.64     1_559.09       0.9999          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-random (self)                    1_270.45       625.46     1_895.91       0.9994          1.0001            1.0000         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            607.27       119.10       726.36       0.6818          1.3682            1.2621         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            607.27       117.32       724.58       0.6799          1.3836            1.2680         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            607.27       131.85       739.12       0.6781          1.4037            1.2740         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           607.27       183.54       790.81       0.9990          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           607.27       234.65       841.92       0.9993          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           607.27       194.41       801.67       0.9995          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           607.27       253.56       860.83       0.9998          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           607.27       200.72       807.99       0.9995          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           607.27       264.65       871.92       0.9999          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-random (self)                      607.27       538.03     1_145.30       0.9993          1.0001            1.0000         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            742.91       116.18       859.09       0.6814          1.3673            1.2633         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)            742.91       119.37       862.28       0.6806          1.3746            1.2655         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)            742.91       129.94       872.85       0.6785          1.3952            1.2721         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)           742.91       176.68       919.59       0.9993          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)           742.91       257.02       999.94       0.9997          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)           742.91       179.53       922.44       0.9994          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)           742.91       263.89     1_006.80       0.9998          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)           742.91       208.15       951.06       0.9995          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)           742.91       259.45     1_002.36       0.9999          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-random (self)                      742.91       531.41     1_274.32       0.9993          1.0001            1.0000         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)              1_344.10       117.76     1_461.87       0.6920          1.3324            1.2445         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)             1_344.10       127.17     1_471.27       0.6870          1.3652            1.2566         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)             1_344.10       153.96     1_498.07       0.6853          1.3837            1.2597         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)             1_344.10       179.35     1_523.45       0.9975          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)             1_344.10       247.83     1_591.94       0.9976          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)            1_344.10       202.21     1_546.32       0.9997          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)            1_344.10       282.51     1_626.61       0.9999          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)            1_344.10       251.16     1_595.26       0.9996          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)            1_344.10       284.29     1_628.40       1.0000          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-pca (self)                       1_344.10       604.18     1_948.29       0.9996          1.0001            1.0000         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               623.78       111.36       735.14       0.6896          1.3463            1.2460         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               623.78       117.86       741.64       0.6879          1.3599            1.2522         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               623.78       132.63       756.41       0.6861          1.3769            1.2574         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              623.78       178.29       802.07       0.9991          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              623.78       236.64       860.43       0.9993          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              623.78       185.90       809.68       0.9995          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              623.78       248.59       872.37       0.9998          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              623.78       203.17       826.96       0.9996          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              623.78       274.49       898.27       1.0000          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-pca (self)                         623.78       539.17     1_162.95       0.9995          1.0001            1.0000         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               747.46       114.84       862.30       0.6893          1.3467            1.2489         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)               747.46       118.92       866.37       0.6885          1.3532            1.2507         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)               747.46       141.70       889.16       0.6865          1.3709            1.2561         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)              747.46       180.55       928.00       0.9994          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)              747.46       239.62       987.08       0.9997          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)              747.46       182.44       929.90       0.9995          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)              747.46       243.73       991.19       0.9998          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)              747.46       201.92       949.38       0.9996          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)              747.46       266.12     1_013.57       1.0000          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-pca (self)                         747.46       526.26     1_273.72       0.9995          1.0001            1.0000         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)              1_201.90       222.63     1_424.52       0.3698          2.2923            2.0232         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)             1_201.90       256.95     1_458.85       0.3462          2.4709            2.1626         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)             1_201.90       289.08     1_490.98       0.3312          2.6495            2.2473         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)             1_201.90       265.55     1_467.44       0.7370          1.1542            1.0836         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)             1_201.90       441.06     1_642.96       0.9127          1.0437            1.0000         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)            1_201.90       300.88     1_502.77       0.6111          1.2797            1.1979         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)            1_201.90       489.80     1_691.70       0.8375          1.0872            1.0259         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)            1_201.90       336.69     1_538.58       0.5503          1.3843            1.2645         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)            1_201.90       527.20     1_729.10       0.7858          1.1267            1.0514         1.68
IVF-Binary-256-nl158-sign (self)                       1_201.90       895.88     2_097.77       0.6110          1.2817            1.1969         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               527.18       234.06       761.24       0.3263          2.5881            2.1617         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               527.18       254.14       781.32       0.3166          2.6847            2.2373         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               527.18       291.78       818.96       0.2971          2.9458            2.3783         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              527.18       278.22       805.40       0.6563          1.2510            1.1328         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              527.18       462.96       990.14       0.8243          1.1194            1.0197         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              527.18       292.30       819.47       0.6037          1.3094            1.1855         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              527.18       485.28     1_012.46       0.7917          1.1453            1.0344         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              527.18       328.56       855.74       0.5241          1.4384            1.2862         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              527.18       520.58     1_047.76       0.7306          1.2030            1.0694         1.75
IVF-Binary-256-nl223-sign (self)                         527.18       862.96     1_390.14       0.6051          1.3104            1.1855         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               680.63       251.29       931.92       0.2930          2.8391            2.3380         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               680.63       263.84       944.47       0.2880          2.9065            2.3795         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               680.63       304.06       984.69       0.2711          3.1733            2.5248         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              680.63       292.60       973.23       0.6133          1.3217            1.1663         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              680.63       472.10     1_152.73       0.7590          1.1822            1.0464         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              680.63       305.88       986.51       0.5879          1.3579            1.1935         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              680.63       504.51     1_185.14       0.7428          1.1998            1.0547         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              680.63       336.57     1_017.20       0.5132          1.4900            1.2890         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              680.63       532.00     1_212.63       0.6920          1.2620            1.0877         1.84
IVF-Binary-256-nl316-sign (self)                         680.63       918.68     1_599.30       0.5883          1.3553            1.1938         1.84
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
Exhaustive (query)                                        68.81     1_347.12     1_415.93       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.81     4_503.78     4_572.58       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                131.71       267.76       399.48       0.5546          1.7646            1.5366         2.03
ExhaustiveBinary-256-random-rf10 (query)                 131.71       410.24       541.95       0.9898          1.0017            1.0000         2.03
ExhaustiveBinary-256-random-rf20 (query)                 131.71       555.21       686.92       0.9984          1.0002            1.0000         2.03
ExhaustiveBinary-256-random (self)                       131.71     1_316.12     1_447.83       0.9899          1.0016            1.0000         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   228.91       271.88       500.80       0.5767          1.6243            1.4311         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    228.91       411.47       640.38       0.9903          1.0016            1.0000         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    228.91       564.27       793.18       0.9984          1.0002            1.0000         2.03
ExhaustiveBinary-256-pca (self)                          228.91     1_299.49     1_528.41       0.9904          1.0016            1.0000         2.03
ExhaustiveBinary-512-random_no_rr (query)                204.20       388.14       592.34       0.6013          1.6760            1.4608         4.05
ExhaustiveBinary-512-random-rf10 (query)                 204.20       537.92       742.12       0.9977          1.0003            1.0000         4.05
ExhaustiveBinary-512-random-rf20 (query)                 204.20       690.37       894.57       0.9997          1.0000            1.0000         4.05
ExhaustiveBinary-512-random (self)                       204.20     1_742.99     1_947.19       0.9975          1.0003            1.0000         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   291.86       397.16       689.02       0.6443          1.4426            1.3064         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    291.86       545.20       837.06       0.9984          1.0002            1.0000         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    291.86       719.71     1_011.57       0.9998          1.0000            1.0000         4.05
ExhaustiveBinary-512-pca (self)                          291.86     1_739.63     2_031.50       0.9984          1.0002            1.0000         4.05
ExhaustiveBinary-1024-random_no_rr (query)               254.98       633.28       888.26       0.6623          1.4553            1.3048         8.11
ExhaustiveBinary-1024-random-rf10 (query)                254.98       779.72     1_034.70       0.9994          1.0001            1.0000         8.11
ExhaustiveBinary-1024-random-rf20 (query)                254.98       972.20     1_227.18       0.9999          1.0000            1.0000         8.11
ExhaustiveBinary-1024-random (self)                      254.98     2_530.36     2_785.34       0.9994          1.0001            1.0000         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  347.72       596.94       944.66       0.6865          1.3603            1.2383         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   347.72       761.12     1_108.84       0.9996          1.0001            1.0000         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   347.72       915.60     1_263.32       0.9999          1.0000            1.0000         8.11
ExhaustiveBinary-1024-pca (self)                         347.72     2_483.06     2_830.79       0.9995          1.0001            1.0000         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   83.86       663.66       747.52       0.0400         18.1511           13.6734         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    83.86       721.11       804.97       0.1821          2.5573            2.4620         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    83.86     1_150.52     1_234.39       0.3140          1.8429            1.7786         3.05
ExhaustiveBinary-512-sign (self)                          83.86     2_349.88     2_433.74       0.1897          2.5286            2.4283         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)            2_274.91        84.34     2_359.26       0.5635          1.6351            1.4891         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           2_274.91        98.67     2_373.59       0.5601          1.6665            1.5058         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           2_274.91       108.81     2_383.72       0.5586          1.6897            1.5137         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           2_274.91       178.12     2_453.03       0.9918          1.0013            1.0000         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           2_274.91       269.03     2_543.94       0.9978          1.0003            1.0000         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          2_274.91       190.74     2_465.66       0.9916          1.0013            1.0000         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          2_274.91       279.69     2_554.61       0.9988          1.0001            1.0000         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          2_274.91       193.52     2_468.43       0.9909          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          2_274.91       286.31     2_561.22       0.9987          1.0001            1.0000         2.34
IVF-Binary-256-nl158-random (self)                     2_274.91       413.14     2_688.05       0.9918          1.0012            1.0000         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)             893.83        84.25       978.09       0.5616          1.6429            1.4936         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)             893.83        84.78       978.61       0.5604          1.6551            1.5018         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)             893.83        93.01       986.84       0.5588          1.6784            1.5117         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)            893.83       174.94     1_068.77       0.9922          1.0012            1.0000         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)            893.83       269.13     1_162.96       0.9989          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)            893.83       176.37     1_070.20       0.9919          1.0012            1.0000         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)            893.83       279.53     1_173.36       0.9989          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)            893.83       185.48     1_079.31       0.9911          1.0014            1.0000         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)            893.83       279.59     1_173.42       0.9988          1.0001            1.0000         2.47
IVF-Binary-256-nl223-random (self)                       893.83       391.10     1_284.94       0.9921          1.0012            1.0000         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)           1_132.37        87.18     1_219.55       0.5613          1.6459            1.4961         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           1_132.37        89.71     1_222.08       0.5607          1.6528            1.5009         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           1_132.37        95.38     1_227.75       0.5595          1.6692            1.5101         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          1_132.37       181.09     1_313.45       0.9922          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          1_132.37       278.45     1_410.81       0.9990          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          1_132.37       183.05     1_315.42       0.9919          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          1_132.37       283.83     1_416.20       0.9990          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          1_132.37       186.45     1_318.82       0.9913          1.0014            1.0000         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          1_132.37       287.74     1_420.11       0.9989          1.0001            1.0000         2.65
IVF-Binary-256-nl316-random (self)                     1_132.37       401.41     1_533.77       0.9922          1.0012            1.0000         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               2_332.43        74.61     2_407.04       0.5842          1.5221            1.4049         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              2_332.43        83.49     2_415.92       0.5814          1.5460            1.4159         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              2_332.43        91.90     2_424.33       0.5803          1.5591            1.4200         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              2_332.43       172.13     2_504.56       0.9919          1.0013            1.0000         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              2_332.43       264.57     2_597.00       0.9977          1.0004            1.0000         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             2_332.43       177.29     2_509.72       0.9919          1.0012            1.0000         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             2_332.43       275.67     2_608.10       0.9988          1.0001            1.0000         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             2_332.43       186.49     2_518.93       0.9913          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             2_332.43       284.65     2_617.08       0.9987          1.0002            1.0000         2.34
IVF-Binary-256-nl158-pca (self)                        2_332.43       409.75     2_742.19       0.9920          1.0013            1.0000         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)                978.87        80.64     1_059.52       0.5826          1.5293            1.4089         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)                978.87        84.03     1_062.90       0.5817          1.5377            1.4128         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)                978.87        92.43     1_071.30       0.5805          1.5564            1.4188         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)               978.87       176.20     1_155.07       0.9921          1.0012            1.0000         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)               978.87       270.61     1_249.49       0.9988          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)               978.87       176.09     1_154.96       0.9920          1.0012            1.0000         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)               978.87       272.08     1_250.95       0.9989          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)               978.87       182.35     1_161.22       0.9913          1.0014            1.0000         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)               978.87       282.03     1_260.90       0.9987          1.0002            1.0000         2.47
IVF-Binary-256-nl223-pca (self)                          978.87       395.72     1_374.60       0.9921          1.0012            1.0000         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              1_213.08        88.21     1_301.29       0.5831          1.5311            1.4097         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              1_213.08        90.57     1_303.65       0.5827          1.5355            1.4112         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              1_213.08        95.57     1_308.65       0.5816          1.5479            1.4167         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             1_213.08       180.98     1_394.07       0.9923          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             1_213.08       274.88     1_487.96       0.9989          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             1_213.08       182.92     1_396.01       0.9922          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             1_213.08       274.93     1_488.02       0.9989          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             1_213.08       186.84     1_399.92       0.9915          1.0013            1.0000         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             1_213.08       283.95     1_497.03       0.9987          1.0002            1.0000         2.65
IVF-Binary-256-nl316-pca (self)                        1_213.08       399.10     1_612.18       0.9923          1.0012            1.0000         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)            2_295.83       102.13     2_397.96       0.6099          1.5613            1.4188         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)           2_295.83       115.62     2_411.45       0.6060          1.5937            1.4384         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)           2_295.83       130.38     2_426.22       0.6043          1.6153            1.4468         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)           2_295.83       199.89     2_495.73       0.9973          1.0004            1.0000         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)           2_295.83       294.62     2_590.45       0.9985          1.0003            1.0000         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)          2_295.83       209.62     2_505.46       0.9982          1.0002            1.0000         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)          2_295.83       311.93     2_607.76       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)          2_295.83       221.27     2_517.10       0.9981          1.0002            1.0000         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)          2_295.83       323.67     2_619.51       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-random (self)                     2_295.83       531.28     2_827.12       0.9982          1.0002            1.0000         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)             967.72       108.20     1_075.92       0.6073          1.5771            1.4298         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)             967.72       116.92     1_084.64       0.6060          1.5876            1.4373         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)             967.72       124.74     1_092.47       0.6042          1.6107            1.4479         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)            967.72       203.53     1_171.25       0.9982          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)            967.72       300.01     1_267.73       0.9996          1.0001            1.0000         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)            967.72       220.45     1_188.17       0.9983          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)            967.72       303.48     1_271.21       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)            967.72       218.85     1_186.57       0.9980          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)            967.72       318.99     1_286.71       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-random (self)                       967.72       502.92     1_470.64       0.9981          1.0002            1.0000         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)           1_206.71       115.25     1_321.96       0.6071          1.5794            1.4300         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)           1_206.71       117.42     1_324.13       0.6063          1.5867            1.4335         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)           1_206.71       127.25     1_333.96       0.6048          1.6036            1.4413         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)          1_206.71       208.91     1_415.62       0.9983          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)          1_206.71       306.07     1_512.78       0.9997          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)          1_206.71       210.43     1_417.15       0.9983          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)          1_206.71       310.77     1_517.48       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)          1_206.71       219.77     1_426.48       0.9981          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)          1_206.71       324.22     1_530.93       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-random (self)                     1_206.71       504.84     1_711.56       0.9982          1.0002            1.0000         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)               2_386.03       102.38     2_488.42       0.6502          1.3846            1.2877         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)              2_386.03       116.58     2_502.61       0.6474          1.4003            1.2963         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)              2_386.03       128.02     2_514.05       0.6463          1.4103            1.3000         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)              2_386.03       199.98     2_586.01       0.9975          1.0004            1.0000         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)              2_386.03       294.63     2_680.66       0.9985          1.0003            1.0000         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)             2_386.03       210.16     2_596.19       0.9986          1.0001            1.0000         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)             2_386.03       308.98     2_695.01       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)             2_386.03       224.54     2_610.57       0.9986          1.0001            1.0000         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)             2_386.03       325.94     2_711.97       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-pca (self)                        2_386.03       529.48     2_915.51       0.9987          1.0001            1.0000         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_046.93       108.32     1_155.25       0.6484          1.3912            1.2918         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_046.93       113.06     1_160.00       0.6475          1.3973            1.2953         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_046.93       130.24     1_177.17       0.6464          1.4096            1.2996         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_046.93       203.40     1_250.33       0.9985          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_046.93       300.77     1_347.70       0.9996          1.0001            1.0000         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_046.93       206.76     1_253.69       0.9987          1.0001            1.0000         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_046.93       310.74     1_357.67       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_046.93       217.93     1_264.87       0.9986          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_046.93       324.81     1_371.75       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-pca (self)                        1_046.93       499.01     1_545.94       0.9987          1.0002            1.0000         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)              1_350.20       118.09     1_468.29       0.6485          1.3918            1.2914         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)              1_350.20       117.19     1_467.39       0.6481          1.3947            1.2931         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)              1_350.20       126.66     1_476.85       0.6471          1.4035            1.2972         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)             1_350.20       209.59     1_559.79       0.9987          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)             1_350.20       304.39     1_654.59       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)             1_350.20       208.23     1_558.43       0.9987          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)             1_350.20       308.85     1_659.05       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)             1_350.20       217.99     1_568.19       0.9986          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)             1_350.20       318.73     1_668.93       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-pca (self)                        1_350.20       504.26     1_854.46       0.9987          1.0001            1.0000         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)           2_367.21       157.37     2_524.58       0.6689          1.3866            1.2841         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)          2_367.21       173.64     2_540.85       0.6656          1.4072            1.2938         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)          2_367.21       192.27     2_559.48       0.6641          1.4215            1.2990         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)          2_367.21       254.75     2_621.95       0.9983          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)          2_367.21       371.02     2_738.23       0.9985          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)         2_367.21       276.16     2_643.37       0.9995          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)         2_367.21       377.92     2_745.12       0.9999          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)         2_367.21       291.55     2_658.76       0.9995          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)         2_367.21       408.28     2_775.49       0.9999          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-random (self)                    2_367.21       756.05     3_123.25       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_010.78       161.49     1_172.26       0.6668          1.3969            1.2901         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_010.78       167.80     1_178.58       0.6657          1.4047            1.2936         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_010.78       184.11     1_194.88       0.6643          1.4195            1.2992         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_010.78       262.37     1_273.14       0.9994          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_010.78       370.97     1_381.74       0.9997          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_010.78       263.25     1_274.03       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_010.78       377.11     1_387.88       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_010.78       282.33     1_293.11       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_010.78       408.56     1_419.33       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-random (self)                    1_010.78       704.69     1_715.46       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)          1_258.95       172.42     1_431.37       0.6663          1.4016            1.2908         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)          1_258.95       176.58     1_435.53       0.6658          1.4059            1.2929         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)          1_258.95       186.77     1_445.72       0.6646          1.4173            1.2982         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)         1_258.95       264.37     1_523.31       0.9995          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)         1_258.95       373.40     1_632.35       0.9998          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)         1_258.95       271.57     1_530.51       0.9995          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)         1_258.95       377.99     1_636.94       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)         1_258.95       289.69     1_548.63       0.9996          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)         1_258.95       408.48     1_667.42       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-random (self)                    1_258.95       724.22     1_983.16       0.9995          1.0001            1.0000         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)              2_500.12       159.79     2_659.92       0.6919          1.3132            1.2255         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)             2_500.12       174.00     2_674.12       0.6891          1.3276            1.2320         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)             2_500.12       192.48     2_692.60       0.6879          1.3368            1.2349         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)             2_500.12       257.45     2_757.57       0.9983          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)             2_500.12       378.49     2_878.61       0.9985          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)            2_500.12       278.38     2_778.50       0.9996          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)            2_500.12       382.99     2_883.11       0.9999          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)            2_500.12       291.21     2_791.33       0.9996          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)            2_500.12       402.39     2_902.51       0.9999          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-pca (self)                       2_500.12       755.59     3_255.71       0.9996          1.0000            1.0000         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_185.81       162.91     1_348.73       0.6901          1.3205            1.2294         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_185.81       171.62     1_357.43       0.6893          1.3259            1.2315         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_185.81       192.60     1_378.41       0.6881          1.3368            1.2355         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_185.81       257.01     1_442.83       0.9994          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_185.81       367.03     1_552.84       0.9997          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_185.81       266.98     1_452.79       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_185.81       373.60     1_559.41       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_185.81       284.45     1_470.26       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_185.81       395.92     1_581.73       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-pca (self)                       1_185.81       716.23     1_902.05       0.9996          1.0000            1.0000         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_377.06       170.82     1_547.88       0.6894          1.3233            1.2299         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_377.06       173.86     1_550.92       0.6890          1.3264            1.2308         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_377.06       185.07     1_562.13       0.6881          1.3339            1.2343         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_377.06       268.35     1_645.41       0.9996          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_377.06       373.55     1_750.61       0.9998          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_377.06       268.87     1_645.92       0.9996          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_377.06       388.49     1_765.55       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_377.06       283.51     1_660.57       0.9996          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_377.06       396.44     1_773.50       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-pca (self)                       1_377.06       718.67     2_095.73       0.9996          1.0000            1.0000         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)              2_207.84       357.02     2_564.86       0.1510          4.5058            3.8746         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)             2_207.84       406.46     2_614.30       0.1361          5.0356            4.2653         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)             2_207.84       455.50     2_663.34       0.1277          5.5367            4.5857         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)             2_207.84       429.17     2_637.01       0.4704          1.6381            1.3330         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)             2_207.84       734.98     2_942.82       0.5942          1.4484            1.1576         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)            2_207.84       465.81     2_673.65       0.3942          1.8935            1.4818         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)            2_207.84       805.40     3_013.23       0.5140          1.6480            1.2489         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)            2_207.84       506.39     2_714.23       0.3569          2.0547            1.5908         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)            2_207.84       848.38     3_056.22       0.4697          1.7895            1.3254         3.36
IVF-Binary-512-nl158-sign (self)                       2_207.84     1_333.00     3_540.84       0.3955          1.8825            1.4821         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               868.90       382.92     1_251.82       0.1213          4.6896            4.1250         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               868.90       407.58     1_276.48       0.1174          4.9249            4.2951         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               868.90       461.74     1_330.63       0.1106          5.4835            4.6867         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              868.90       459.56     1_328.45       0.4197          1.7170            1.4180         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              868.90       765.95     1_634.85       0.5241          1.5150            1.2411         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              868.90       500.08     1_368.98       0.3911          1.7989            1.4901         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              868.90       795.06     1_663.96       0.4903          1.5872            1.2903         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              868.90       517.34     1_386.23       0.3459          1.9626            1.6202         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              868.90       857.87     1_726.76       0.4403          1.7076            1.3831         3.49
IVF-Binary-512-nl223-sign (self)                         868.90     1_321.23     2_190.13       0.3897          1.7981            1.4918         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)             1_084.21       409.96     1_494.17       0.1151          4.6038            4.0973         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)             1_084.21       427.82     1_512.03       0.1136          4.7084            4.1602         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)             1_084.21       476.72     1_560.93       0.1106          5.1122            4.4320         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)            1_084.21       481.24     1_565.45       0.4126          1.7028            1.4384         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)            1_084.21       807.03     1_891.25       0.5130          1.5028            1.2665         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)            1_084.21       491.77     1_575.99       0.3986          1.7407            1.4735         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)            1_084.21       813.13     1_897.34       0.4955          1.5385            1.2897         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)            1_084.21       540.18     1_624.40       0.3574          1.8706            1.5859         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)            1_084.21       878.76     1_962.97       0.4496          1.6513            1.3709         3.67
IVF-Binary-512-nl316-sign (self)                       1_084.21     1_366.46     2_450.67       0.3990          1.7352            1.4743         3.67
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
Exhaustive (query)                                       101.16     1_907.33     2_008.49       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.16     6_546.81     6_647.97       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                192.05       289.96       482.01       0.5361          1.8068            1.5908         2.28
ExhaustiveBinary-256-random-rf10 (query)                 192.05       447.92       639.97       0.9868          1.0022            1.0000         2.28
ExhaustiveBinary-256-random-rf20 (query)                 192.05       612.37       804.42       0.9980          1.0003            1.0000         2.28
ExhaustiveBinary-256-random (self)                       192.05     1_393.47     1_585.52       0.9875          1.0021            1.0000         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   395.80       286.39       682.19       0.5754          1.5495            1.4128         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    395.80       453.66       849.46       0.9895          1.0018            1.0000         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    395.80       610.47     1_006.28       0.9982          1.0002            1.0000         2.28
ExhaustiveBinary-256-pca (self)                          395.80     1_412.36     1_808.16       0.9897          1.0017            1.0000         2.28
ExhaustiveBinary-512-random_no_rr (query)                298.41       403.75       702.16       0.5866          1.6778            1.4946         4.55
ExhaustiveBinary-512-random-rf10 (query)                 298.41       578.48       876.89       0.9966          1.0005            1.0000         4.55
ExhaustiveBinary-512-random-rf20 (query)                 298.41       749.22     1_047.63       0.9996          1.0001            1.0000         4.55
ExhaustiveBinary-512-random (self)                       298.41     1_834.93     2_133.34       0.9969          1.0004            1.0000         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   498.00       413.01       911.01       0.6388          1.4217            1.3032         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    498.00       573.78     1_071.78       0.9979          1.0003            1.0000         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    498.00       769.02     1_267.02       0.9998          1.0000            1.0000         4.55
ExhaustiveBinary-512-pca (self)                          498.00     1_834.12     2_332.13       0.9980          1.0002            1.0000         4.55
ExhaustiveBinary-1024-random_no_rr (query)               499.28       622.07     1_121.36       0.6446          1.4909            1.3512         9.11
ExhaustiveBinary-1024-random-rf10 (query)                499.28       842.58     1_341.86       0.9993          1.0001            1.0000         9.11
ExhaustiveBinary-1024-random-rf20 (query)                499.28       990.81     1_490.09       0.9999          1.0000            1.0000         9.11
ExhaustiveBinary-1024-random (self)                      499.28     2_634.98     3_134.26       0.9993          1.0001            1.0000         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  701.88       615.15     1_317.03       0.6795          1.3452            1.2483         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   701.88       807.60     1_509.47       0.9996          1.0001            1.0000         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   701.88       994.88     1_696.76       0.9999          1.0000            1.0000         9.11
ExhaustiveBinary-1024-pca (self)                         701.88     2_631.15     3_333.03       0.9996          1.0000            1.0000         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  127.62       837.24       964.86       0.0420         17.7082           13.0970         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   127.62       916.31     1_043.94       0.1896          2.5240            2.4052         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   127.62     1_427.42     1_555.05       0.3229          1.8300            1.7348         4.58
ExhaustiveBinary-768-sign (self)                         127.62     2_954.12     3_081.74       0.1997          2.4832            2.3546         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)            3_278.54       107.89     3_386.43       0.5435          1.7067            1.5434         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)           3_278.54       120.49     3_399.03       0.5415          1.7256            1.5559         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)           3_278.54       129.74     3_408.28       0.5404          1.7418            1.5622         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)           3_278.54       214.05     3_492.59       0.9891          1.0017            1.0000         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)           3_278.54       327.93     3_606.47       0.9984          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)          3_278.54       215.61     3_494.15       0.9884          1.0019            1.0000         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)          3_278.54       337.96     3_616.49       0.9985          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)          3_278.54       228.63     3_507.17       0.9878          1.0020            1.0000         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)          3_278.54       344.61     3_623.15       0.9983          1.0002            1.0000         2.74
IVF-Binary-256-nl158-random (self)                     3_278.54       499.27     3_777.81       0.9891          1.0018            1.0000         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)           1_296.44       105.10     1_401.55       0.5424          1.7134            1.5511         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)           1_296.44       108.21     1_404.65       0.5416          1.7222            1.5568         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)           1_296.44       115.92     1_412.36       0.5406          1.7398            1.5637         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)          1_296.44       218.83     1_515.28       0.9890          1.0018            1.0000         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)          1_296.44       348.53     1_644.97       0.9986          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)          1_296.44       218.63     1_515.07       0.9885          1.0019            1.0000         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)          1_296.44       338.40     1_634.84       0.9986          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)          1_296.44       226.05     1_522.50       0.9879          1.0020            1.0000         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)          1_296.44       349.99     1_646.43       0.9983          1.0002            1.0000         2.93
IVF-Binary-256-nl223-random (self)                     1_296.44       494.70     1_791.14       0.9893          1.0017            1.0000         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)           1_653.56       113.40     1_766.96       0.5427          1.7074            1.5483         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)           1_653.56       115.66     1_769.22       0.5424          1.7118            1.5506         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)           1_653.56       121.29     1_774.86       0.5414          1.7262            1.5577         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)          1_653.56       235.98     1_889.55       0.9891          1.0018            1.0000         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)          1_653.56       347.26     2_000.82       0.9987          1.0002            1.0000         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)          1_653.56       225.59     1_879.16       0.9888          1.0018            1.0000         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)          1_653.56       349.84     2_003.41       0.9986          1.0002            1.0000         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)          1_653.56       232.81     1_886.38       0.9882          1.0020            1.0000         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)          1_653.56       356.72     2_010.28       0.9984          1.0002            1.0000         3.21
IVF-Binary-256-nl316-random (self)                     1_653.56       513.24     2_166.81       0.9896          1.0017            1.0000         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)               3_435.07        94.09     3_529.16       0.5809          1.4959            1.3911         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)              3_435.07       104.80     3_539.87       0.5795          1.5073            1.3981         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)              3_435.07       112.09     3_547.16       0.5787          1.5149            1.4012         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)              3_435.07       225.06     3_660.13       0.9914          1.0013            1.0000         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)              3_435.07       326.60     3_761.67       0.9986          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)             3_435.07       223.39     3_658.46       0.9908          1.0015            1.0000         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)             3_435.07       348.78     3_783.85       0.9987          1.0001            1.0000         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)             3_435.07       227.99     3_663.06       0.9902          1.0016            1.0000         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)             3_435.07       344.14     3_779.21       0.9985          1.0002            1.0000         2.74
IVF-Binary-256-nl158-pca (self)                        3_435.07       496.24     3_931.31       0.9911          1.0014            1.0000         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)              1_479.68       102.99     1_582.68       0.5803          1.5006            1.3943         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)              1_479.68       106.36     1_586.04       0.5797          1.5059            1.3976         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)              1_479.68       116.73     1_596.41       0.5790          1.5140            1.4008         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)             1_479.68       218.06     1_697.75       0.9912          1.0014            1.0000         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)             1_479.68       339.12     1_818.80       0.9988          1.0001            1.0000         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)             1_479.68       222.70     1_702.38       0.9908          1.0015            1.0000         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)             1_479.68       341.34     1_821.02       0.9988          1.0001            1.0000         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)             1_479.68       229.18     1_708.87       0.9902          1.0016            1.0000         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)             1_479.68       347.02     1_826.70       0.9986          1.0002            1.0000         2.93
IVF-Binary-256-nl223-pca (self)                        1_479.68       494.72     1_974.40       0.9912          1.0014            1.0000         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)              1_850.54       113.71     1_964.25       0.5807          1.4990            1.3926         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)              1_850.54       114.74     1_965.27       0.5804          1.5013            1.3941         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)              1_850.54       143.97     1_994.51       0.5798          1.5079            1.3972         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)             1_850.54       229.87     2_080.41       0.9913          1.0014            1.0000         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)             1_850.54       344.76     2_195.30       0.9988          1.0001            1.0000         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)             1_850.54       225.35     2_075.89       0.9911          1.0014            1.0000         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)             1_850.54       346.44     2_196.98       0.9988          1.0001            1.0000         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)             1_850.54       232.44     2_082.98       0.9904          1.0016            1.0000         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)             1_850.54       370.64     2_221.18       0.9986          1.0002            1.0000         3.21
IVF-Binary-256-nl316-pca (self)                        1_850.54       522.28     2_372.82       0.9913          1.0013            1.0000         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)            3_348.88       130.90     3_479.79       0.5928          1.5995            1.4605         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)           3_348.88       143.80     3_492.68       0.5905          1.6171            1.4731         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)           3_348.88       157.38     3_506.26       0.5893          1.6328            1.4788         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)           3_348.88       251.29     3_600.17       0.9972          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)           3_348.88       380.19     3_729.08       0.9994          1.0001            1.0000         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)          3_348.88       258.91     3_607.80       0.9972          1.0003            1.0000         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)          3_348.88       386.51     3_735.39       0.9998          1.0000            1.0000         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)          3_348.88       272.76     3_621.64       0.9969          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)          3_348.88       401.49     3_750.38       0.9997          1.0000            1.0000         5.02
IVF-Binary-512-nl158-random (self)                     3_348.88       657.07     4_005.95       0.9975          1.0003            1.0000         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)           1_399.88       145.33     1_545.21       0.5912          1.6080            1.4661         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)           1_399.88       147.37     1_547.25       0.5904          1.6164            1.4718         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)           1_399.88       158.16     1_558.04       0.5893          1.6325            1.4790         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)          1_399.88       259.18     1_659.06       0.9973          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)          1_399.88       381.39     1_781.28       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)          1_399.88       260.19     1_660.07       0.9971          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)          1_399.88       387.19     1_787.08       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)          1_399.88       270.91     1_670.79       0.9970          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)          1_399.88       402.47     1_802.35       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-random (self)                     1_399.88       644.08     2_043.97       0.9975          1.0003            1.0000         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)           1_758.28       151.36     1_909.64       0.5914          1.6047            1.4667         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)           1_758.28       152.47     1_910.74       0.5910          1.6093            1.4695         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)           1_758.28       164.43     1_922.71       0.5899          1.6237            1.4754         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)          1_758.28       268.64     2_026.92       0.9974          1.0003            1.0000         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)          1_758.28       394.58     2_152.86       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)          1_758.28       269.19     2_027.47       0.9974          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)          1_758.28       397.68     2_155.95       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)          1_758.28       277.09     2_035.36       0.9971          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)          1_758.28       411.50     2_169.78       0.9997          1.0000            1.0000         5.48
IVF-Binary-512-nl316-random (self)                     1_758.28       670.97     2_429.25       0.9976          1.0003            1.0000         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)               3_537.86       132.07     3_669.92       0.6434          1.3816            1.2885         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)              3_537.86       144.81     3_682.66       0.6417          1.3915            1.2942         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)              3_537.86       156.24     3_694.10       0.6410          1.3994            1.2971         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)              3_537.86       264.81     3_802.67       0.9982          1.0002            1.0000         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)              3_537.86       367.14     3_905.00       0.9994          1.0001            1.0000         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)             3_537.86       259.59     3_797.45       0.9983          1.0002            1.0000         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)             3_537.86       394.67     3_932.53       0.9999          1.0000            1.0000         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)             3_537.86       271.66     3_809.52       0.9981          1.0002            1.0000         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)             3_537.86       410.07     3_947.93       0.9998          1.0000            1.0000         5.02
IVF-Binary-512-nl158-pca (self)                        3_537.86       660.81     4_198.67       0.9984          1.0002            1.0000         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_576.79       141.20     1_717.99       0.6425          1.3869            1.2915         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_576.79       146.24     1_723.03       0.6417          1.3920            1.2940         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_576.79       156.22     1_733.01       0.6410          1.4002            1.2970         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_576.79       261.11     1_837.90       0.9983          1.0002            1.0000         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_576.79       389.12     1_965.91       0.9998          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_576.79       261.48     1_838.27       0.9983          1.0002            1.0000         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_576.79       387.65     1_964.44       0.9998          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_576.79       271.42     1_848.21       0.9981          1.0002            1.0000         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_576.79       397.43     1_974.22       0.9998          1.0000            1.0000         5.21
IVF-Binary-512-nl223-pca (self)                        1_576.79       639.86     2_216.65       0.9984          1.0002            1.0000         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)              1_932.14       150.63     2_082.76       0.6422          1.3880            1.2923         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)              1_932.14       152.25     2_084.39       0.6420          1.3899            1.2934         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)              1_932.14       161.19     2_093.33       0.6414          1.3958            1.2959         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)             1_932.14       267.89     2_200.02       0.9984          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)             1_932.14       394.14     2_326.28       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)             1_932.14       268.79     2_200.93       0.9983          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)             1_932.14       397.19     2_329.33       0.9999          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)             1_932.14       277.11     2_209.24       0.9982          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)             1_932.14       406.65     2_338.79       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-pca (self)                        1_932.14       670.66     2_602.80       0.9984          1.0002            1.0000         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)           3_528.45       216.05     3_744.50       0.6493          1.4395            1.3309         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)          3_528.45       231.49     3_759.95       0.6471          1.4527            1.3401         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)          3_528.45       251.22     3_779.67       0.6459          1.4637            1.3437         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)          3_528.45       347.67     3_876.12       0.9991          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)          3_528.45       482.44     4_010.89       0.9995          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)         3_528.45       348.16     3_876.61       0.9995          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)         3_528.45       482.12     4_010.57       0.9999          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)         3_528.45       378.54     3_906.99       0.9994          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)         3_528.45       510.27     4_038.73       0.9999          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-random (self)                    3_528.45       979.51     4_507.96       0.9995          1.0001            1.0000         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_571.62       228.37     1_799.99       0.6478          1.4482            1.3356         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_571.62       225.21     1_796.83       0.6470          1.4545            1.3394         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_571.62       241.47     1_813.09       0.6459          1.4661            1.3437         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_571.62       345.14     1_916.77       0.9994          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_571.62       471.37     2_042.99       0.9998          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_571.62       352.28     1_923.90       0.9994          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_571.62       488.80     2_060.42       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_571.62       374.41     1_946.03       0.9994          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_571.62       509.52     2_081.14       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-random (self)                    1_571.62       943.62     2_515.24       0.9995          1.0001            1.0000         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)          1_961.07       230.36     2_191.43       0.6478          1.4451            1.3360        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)          1_961.07       231.60     2_192.67       0.6474          1.4486            1.3381        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)          1_961.07       244.67     2_205.74       0.6465          1.4581            1.3419        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)         1_961.07       364.09     2_325.16       0.9995          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)         1_961.07       490.88     2_451.94       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)         1_961.07       387.63     2_348.70       0.9995          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)         1_961.07       499.00     2_460.06       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)         1_961.07       386.77     2_347.83       0.9994          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)         1_961.07       519.15     2_480.22       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-random (self)                    1_961.07       969.84     2_930.91       0.9995          1.0001            1.0000        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)              3_797.34       211.55     4_008.89       0.6828          1.3190            1.2380         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)             3_797.34       227.07     4_024.41       0.6813          1.3268            1.2437         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)             3_797.34       244.67     4_042.01       0.6805          1.3325            1.2450         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)             3_797.34       329.07     4_126.41       0.9993          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)             3_797.34       464.85     4_262.20       0.9995          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)            3_797.34       350.28     4_147.62       0.9996          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)            3_797.34       489.76     4_287.10       0.9999          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)            3_797.34       364.31     4_161.65       0.9996          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)            3_797.34       509.53     4_306.87       0.9999          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-pca (self)                       3_797.34       966.60     4_763.94       0.9997          1.0000            1.0000         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_794.78       218.94     2_013.72       0.6820          1.3229            1.2405         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_794.78       227.84     2_022.62       0.6813          1.3267            1.2428         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_794.78       239.55     2_034.33       0.6806          1.3327            1.2451         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_794.78       336.03     2_130.81       0.9995          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_794.78       472.03     2_266.82       0.9998          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_794.78       342.74     2_137.52       0.9996          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_794.78       480.71     2_275.49       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_794.78       379.67     2_174.46       0.9996          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_794.78       503.74     2_298.52       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-pca (self)                       1_794.78       938.08     2_732.86       0.9997          1.0000            1.0000         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             2_133.22       232.20     2_365.42       0.6817          1.3231            1.2411        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)             2_133.22       231.66     2_364.88       0.6814          1.3247            1.2421        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)             2_133.22       245.73     2_378.95       0.6809          1.3298            1.2441        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)            2_133.22       359.83     2_493.04       0.9996          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)            2_133.22       491.55     2_624.77       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)            2_133.22       362.28     2_495.50       0.9997          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)            2_133.22       502.64     2_635.86       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)            2_133.22       383.02     2_516.24       0.9996          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)            2_133.22       520.30     2_653.52       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-pca (self)                       2_133.22       963.54     3_096.75       0.9997          1.0000            1.0000        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)              3_200.58       503.62     3_704.20       0.1087          4.9638            4.2820         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)             3_200.58       561.45     3_762.02       0.0910          5.6532            4.8194         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)             3_200.58       612.79     3_813.37       0.0835          6.4783            5.2784         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)             3_200.58       593.74     3_794.32       0.3875          1.9344            1.4961         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)             3_200.58     1_041.75     4_242.33       0.4922          1.6906            1.2787         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)            3_200.58       674.16     3_874.74       0.3241          2.2103            1.7117         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)            3_200.58     1_117.67     4_318.25       0.4045          1.9478            1.4779         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)            3_200.58       691.22     3_891.80       0.2968          2.3464            1.8295         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)            3_200.58     1_155.12     4_355.70       0.3625          2.1099            1.6150         5.04
IVF-Binary-768-nl158-sign (self)                       3_200.58     1_848.11     5_048.69       0.3246          2.2070            1.7069         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)             1_217.51       542.18     1_759.69       0.1024          4.9875            4.3487         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)             1_217.51       578.87     1_796.39       0.0957          5.3044            4.5915         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)             1_217.51       640.44     1_857.95       0.0884          6.0740            5.0397         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)            1_217.51       627.51     1_845.02       0.3639          1.9436            1.5676         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)            1_217.51     1_070.12     2_287.63       0.4695          1.6779            1.3224         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)            1_217.51       651.46     1_868.97       0.3385          2.0460            1.6505         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)            1_217.51     1_099.71     2_317.22       0.4321          1.7706            1.4013         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)            1_217.51       716.59     1_934.10       0.3021          2.2125            1.7971         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)            1_217.51     1_192.34     2_409.86       0.3777          1.9467            1.5524         5.23
IVF-Binary-768-nl223-sign (self)                       1_217.51     1_855.19     3_072.71       0.3378          2.0482            1.6525         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)             1_616.82       608.22     2_225.03       0.0933          5.0942            4.4866         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)             1_616.82       609.85     2_226.66       0.0905          5.2553            4.5966         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)             1_616.82       673.44     2_290.25       0.0847          5.8642            4.9952         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)            1_616.82       670.34     2_287.16       0.3578          1.8929            1.5824         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)            1_616.82     1_131.39     2_748.21       0.4584          1.6471            1.3457         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)            1_616.82       684.36     2_301.18       0.3461          1.9342            1.6236         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)            1_616.82     1_203.72     2_820.54       0.4395          1.6901            1.3849         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)            1_616.82       769.14     2_385.96       0.3116          2.0759            1.7505         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)            1_616.82     1_236.06     2_852.87       0.3881          1.8299            1.5147         5.51
IVF-Binary-768-nl316-sign (self)                       1_616.82     1_947.30     3_564.12       0.3468          1.9292            1.6208         5.51
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
Exhaustive (query)                                        32.94       692.83       725.77       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.94     2_300.44     2_333.38       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             826.27       195.55     1_021.82       0.5641          1.0369            1.0367         2.95
ExhaustiveRaBitQ-rf5 (query)                             826.27       247.66     1_073.93       0.9225          1.0017            1.0006         2.95
ExhaustiveRaBitQ-rf10 (query)                            826.27       293.41     1_119.68       0.9776          1.0003            1.0000         2.95
ExhaustiveRaBitQ-rf20 (query)                            826.27       368.08     1_194.35       0.9903          1.0000            1.0000         2.95
ExhaustiveRaBitQ (self)                                  826.27       934.20     1_760.47       0.9780          1.0003            1.0000         2.95
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_345.57        93.66     1_439.24       0.5751          1.0343            1.0347         3.04
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_345.57       118.44     1_464.01       0.5751          1.0343            1.0347         3.04
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_345.57       146.45     1_492.02       0.5751          1.0343            1.0347         3.04
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_345.57       172.81     1_518.38       0.9785          1.0003            1.0000         3.04
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_345.57       238.93     1_584.50       0.9905          1.0000            1.0000         3.04
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_345.57       197.14     1_542.71       0.9785          1.0003            1.0000         3.04
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_345.57       264.76     1_610.33       0.9905          1.0000            1.0000         3.04
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_345.57       227.52     1_573.09       0.9785          1.0003            1.0000         3.04
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_345.57       288.67     1_634.24       0.9905          1.0000            1.0000         3.04
IVF-RaBitQ-nl158 (self)                                1_345.57       924.62     2_270.20       0.9904          1.0000            1.0000         3.04
IVF-RaBitQ-nl223-np11-rf0 (query)                        879.93       119.35       999.28       0.5864          1.0326            1.0324         3.17
IVF-RaBitQ-nl223-np14-rf0 (query)                        879.93       139.09     1_019.02       0.5865          1.0326            1.0324         3.17
IVF-RaBitQ-nl223-np21-rf0 (query)                        879.93       176.41     1_056.35       0.5865          1.0326            1.0324         3.17
IVF-RaBitQ-nl223-np11-rf10 (query)                       879.93       190.27     1_070.21       0.9814          1.0002            1.0000         3.17
IVF-RaBitQ-nl223-np11-rf20 (query)                       879.93       250.15     1_130.09       0.9907          1.0000            1.0000         3.17
IVF-RaBitQ-nl223-np14-rf10 (query)                       879.93       211.95     1_091.88       0.9815          1.0002            1.0000         3.17
IVF-RaBitQ-nl223-np14-rf20 (query)                       879.93       264.51     1_144.45       0.9908          1.0000            1.0000         3.17
IVF-RaBitQ-nl223-np21-rf10 (query)                       879.93       262.49     1_142.42       0.9815          1.0002            1.0000         3.17
IVF-RaBitQ-nl223-np21-rf20 (query)                       879.93       312.24     1_192.18       0.9908          1.0000            1.0000         3.17
IVF-RaBitQ-nl223 (self)                                  879.93       973.11     1_853.04       0.9907          1.0000            1.0000         3.17
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_085.83       148.01     1_233.84       0.5947          1.0309            1.0310         3.35
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_085.83       153.59     1_239.41       0.5947          1.0309            1.0310         3.35
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_085.83       195.85     1_281.67       0.5947          1.0308            1.0309         3.35
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_085.83       213.07     1_298.90       0.9822          1.0002            1.0000         3.35
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_085.83       271.34     1_357.17       0.9908          1.0000            1.0000         3.35
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_085.83       220.53     1_306.36       0.9823          1.0002            1.0000         3.35
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_085.83       289.24     1_375.07       0.9909          1.0000            1.0000         3.35
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_085.83       266.44     1_352.27       0.9824          1.0002            1.0000         3.35
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_085.83       321.81     1_407.64       0.9910          1.0000            1.0000         3.35
IVF-RaBitQ-nl316 (self)                                1_085.83     1_037.51     2_123.34       0.9908          1.0000            1.0000         3.35
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
Exhaustive (query)                                        68.15     1_319.47     1_387.62       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.15     4_426.49     4_494.64       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           2_065.59       354.81     2_420.40       0.5730          1.0233            1.0233         5.44
ExhaustiveRaBitQ-rf5 (query)                           2_065.59       430.70     2_496.29       0.9181          1.0011            1.0004         5.44
ExhaustiveRaBitQ-rf10 (query)                          2_065.59       475.04     2_540.63       0.9701          1.0002            1.0000         5.44
ExhaustiveRaBitQ-rf20 (query)                          2_065.59       578.65     2_644.24       0.9820          1.0000            1.0000         5.44
ExhaustiveRaBitQ (self)                                2_065.59     1_511.58     3_577.17       0.9701          1.0002            1.0000         5.44
IVF-RaBitQ-nl158-np7-rf0 (query)                       3_131.96       177.94     3_309.90       0.5821          1.0218            1.0223         5.63
IVF-RaBitQ-nl158-np12-rf0 (query)                      3_131.96       230.55     3_362.50       0.5821          1.0218            1.0223         5.63
IVF-RaBitQ-nl158-np17-rf0 (query)                      3_131.96       282.38     3_414.34       0.5821          1.0218            1.0223         5.63
IVF-RaBitQ-nl158-np7-rf10 (query)                      3_131.96       283.93     3_415.88       0.9707          1.0002            1.0000         5.63
IVF-RaBitQ-nl158-np7-rf20 (query)                      3_131.96       399.83     3_531.78       0.9820          1.0000            1.0000         5.63
IVF-RaBitQ-nl158-np12-rf10 (query)                     3_131.96       329.00     3_460.96       0.9707          1.0002            1.0000         5.63
IVF-RaBitQ-nl158-np12-rf20 (query)                     3_131.96       429.89     3_561.85       0.9820          1.0000            1.0000         5.63
IVF-RaBitQ-nl158-np17-rf10 (query)                     3_131.96       383.21     3_515.17       0.9707          1.0002            1.0000         5.63
IVF-RaBitQ-nl158-np17-rf20 (query)                     3_131.96       481.30     3_613.26       0.9820          1.0000            1.0000         5.63
IVF-RaBitQ-nl158 (self)                                3_131.96     1_524.42     4_656.37       0.9817          1.0000            1.0000         5.63
IVF-RaBitQ-nl223-np11-rf0 (query)                      2_257.18       230.82     2_488.00       0.5907          1.0210            1.0212         5.88
IVF-RaBitQ-nl223-np14-rf0 (query)                      2_257.18       257.24     2_514.42       0.5907          1.0210            1.0212         5.88
IVF-RaBitQ-nl223-np21-rf0 (query)                      2_257.18       328.76     2_585.93       0.5907          1.0210            1.0212         5.88
IVF-RaBitQ-nl223-np11-rf10 (query)                     2_257.18       324.86     2_582.03       0.9728          1.0001            1.0000         5.88
IVF-RaBitQ-nl223-np11-rf20 (query)                     2_257.18       414.54     2_671.71       0.9823          1.0000            1.0000         5.88
IVF-RaBitQ-nl223-np14-rf10 (query)                     2_257.18       352.70     2_609.87       0.9729          1.0001            1.0000         5.88
IVF-RaBitQ-nl223-np14-rf20 (query)                     2_257.18       444.99     2_702.17       0.9823          1.0000            1.0000         5.88
IVF-RaBitQ-nl223-np21-rf10 (query)                     2_257.18       425.22     2_682.40       0.9729          1.0001            1.0000         5.88
IVF-RaBitQ-nl223-np21-rf20 (query)                     2_257.18       510.82     2_768.00       0.9823          1.0000            1.0000         5.88
IVF-RaBitQ-nl223 (self)                                2_257.18     1_636.09     3_893.27       0.9819          1.0000            1.0000         5.88
IVF-RaBitQ-nl316-np15-rf0 (query)                      2_475.37       273.63     2_749.00       0.5965          1.0201            1.0205         6.24
IVF-RaBitQ-nl316-np17-rf0 (query)                      2_475.37       301.88     2_777.24       0.5965          1.0201            1.0205         6.24
IVF-RaBitQ-nl316-np25-rf0 (query)                      2_475.37       376.95     2_852.32       0.5965          1.0201            1.0205         6.24
IVF-RaBitQ-nl316-np15-rf10 (query)                     2_475.37       371.54     2_846.91       0.9732          1.0001            1.0000         6.24
IVF-RaBitQ-nl316-np15-rf20 (query)                     2_475.37       464.05     2_939.42       0.9824          1.0000            1.0000         6.24
IVF-RaBitQ-nl316-np17-rf10 (query)                     2_475.37       388.02     2_863.39       0.9732          1.0001            1.0000         6.24
IVF-RaBitQ-nl316-np17-rf20 (query)                     2_475.37       478.47     2_953.84       0.9824          1.0000            1.0000         6.24
IVF-RaBitQ-nl316-np25-rf10 (query)                     2_475.37       467.03     2_942.40       0.9732          1.0001            1.0000         6.24
IVF-RaBitQ-nl316-np25-rf20 (query)                     2_475.37       555.48     3_030.85       0.9824          1.0000            1.0000         6.24
IVF-RaBitQ-nl316 (self)                                2_475.37     1_781.66     4_257.03       0.9819          1.0000            1.0000         6.24
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
Exhaustive (query)                                       100.99     1_951.21     2_052.20       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.99     6_622.56     6_723.55       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           3_927.17       540.75     4_467.92       0.5744          1.0183            1.0183         8.44
ExhaustiveRaBitQ-rf5 (query)                           3_927.17       609.11     4_536.29       0.9135          1.0009            1.0003         8.44
ExhaustiveRaBitQ-rf10 (query)                          3_927.17       674.72     4_601.90       0.9630          1.0001            1.0000         8.44
ExhaustiveRaBitQ-rf20 (query)                          3_927.17       797.83     4_725.01       0.9740          1.0000            1.0000         8.44
ExhaustiveRaBitQ (self)                                3_927.17     2_187.94     6_115.11       0.9631          1.0001            1.0000         8.44
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_457.41       271.47     5_728.88       0.5831          1.0170            1.0175         8.71
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_457.41       353.67     5_811.08       0.5831          1.0170            1.0175         8.71
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_457.41       423.61     5_881.03       0.5831          1.0170            1.0175         8.71
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_457.41       401.01     5_858.42       0.9643          1.0001            1.0000         8.71
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_457.41       542.27     5_999.68       0.9741          1.0000            1.0000         8.71
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_457.41       468.39     5_925.80       0.9643          1.0001            1.0000         8.71
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_457.41       612.84     6_070.26       0.9741          1.0000            1.0000         8.71
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_457.41       549.68     6_007.09       0.9643          1.0001            1.0000         8.71
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_457.41       666.22     6_123.63       0.9741          1.0000            1.0000         8.71
IVF-RaBitQ-nl158 (self)                                5_457.41     2_108.91     7_566.32       0.9738          1.0000            1.0000         8.71
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_850.74       351.55     4_202.29       0.5837          1.0174            1.0174         9.09
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_850.74       394.60     4_245.34       0.5837          1.0174            1.0174         9.09
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_850.74       505.74     4_356.48       0.5837          1.0174            1.0174         9.09
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_850.74       473.84     4_324.58       0.9635          1.0001            1.0000         9.09
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_850.74       586.17     4_436.91       0.9740          1.0000            1.0000         9.09
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_850.74       512.63     4_363.37       0.9636          1.0001            1.0000         9.09
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_850.74       641.42     4_492.16       0.9740          1.0000            1.0000         9.09
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_850.74       619.15     4_469.89       0.9636          1.0001            1.0000         9.09
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_850.74       780.21     4_630.95       0.9740          1.0000            1.0000         9.09
IVF-RaBitQ-nl223 (self)                                3_850.74     2_399.82     6_250.56       0.9738          1.0000            1.0000         9.09
IVF-RaBitQ-nl316-np15-rf0 (query)                      4_349.18       413.96     4_763.14       0.5946          1.0161            1.0165         9.64
IVF-RaBitQ-nl316-np17-rf0 (query)                      4_349.18       458.69     4_807.87       0.5946          1.0161            1.0165         9.64
IVF-RaBitQ-nl316-np25-rf0 (query)                      4_349.18       558.52     4_907.71       0.5946          1.0161            1.0165         9.64
IVF-RaBitQ-nl316-np15-rf10 (query)                     4_349.18       537.75     4_886.93       0.9656          1.0001            1.0000         9.64
IVF-RaBitQ-nl316-np15-rf20 (query)                     4_349.18       659.22     5_008.41       0.9744          1.0000            1.0000         9.64
IVF-RaBitQ-nl316-np17-rf10 (query)                     4_349.18       563.82     4_913.01       0.9657          1.0001            1.0000         9.64
IVF-RaBitQ-nl316-np17-rf20 (query)                     4_349.18       686.68     5_035.86       0.9744          1.0000            1.0000         9.64
IVF-RaBitQ-nl316-np25-rf10 (query)                     4_349.18       678.76     5_027.94       0.9657          1.0001            1.0000         9.64
IVF-RaBitQ-nl316-np25-rf20 (query)                     4_349.18       797.97     5_147.15       0.9744          1.0000            1.0000         9.64
IVF-RaBitQ-nl316 (self)                                4_349.18     2_565.59     6_914.78       0.9740          1.0000            1.0000         9.64
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
Exhaustive (query)                                        32.95       692.34       725.28       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.95     2_299.04     2_331.99       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             939.21       246.72     1_185.93       0.7344          1.0241            1.0229         2.95
ExhaustiveRaBitQ-rf5 (query)                             939.21       301.18     1_240.38       0.9953          1.0001            1.0000         2.95
ExhaustiveRaBitQ-rf10 (query)                            939.21       347.14     1_286.34       0.9976          1.0000            1.0000         2.95
ExhaustiveRaBitQ-rf20 (query)                            939.21       437.48     1_376.68       0.9976          1.0000            1.0000         2.95
ExhaustiveRaBitQ (self)                                  939.21     1_116.53     2_055.74       0.9976          1.0000            1.0000         2.95
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_347.02        96.66     1_443.68       0.7357          1.0239            1.0227         3.04
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_347.02       126.46     1_473.48       0.7357          1.0239            1.0227         3.04
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_347.02       159.00     1_506.02       0.7357          1.0239            1.0227         3.04
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_347.02       175.95     1_522.97       0.9976          1.0000            1.0000         3.04
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_347.02       240.28     1_587.30       0.9976          1.0000            1.0000         3.04
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_347.02       204.26     1_551.29       0.9976          1.0000            1.0000         3.04
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_347.02       273.04     1_620.06       0.9976          1.0000            1.0000         3.04
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_347.02       239.35     1_586.37       0.9976          1.0000            1.0000         3.04
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_347.02       303.69     1_650.72       0.9976          1.0000            1.0000         3.04
IVF-RaBitQ-nl158 (self)                                1_347.02       987.79     2_334.81       0.9976          1.0000            1.0000         3.04
IVF-RaBitQ-nl223-np11-rf0 (query)                        976.98       122.85     1_099.83       0.7404          1.0229            1.0220         3.17
IVF-RaBitQ-nl223-np14-rf0 (query)                        976.98       139.23     1_116.21       0.7404          1.0229            1.0220         3.17
IVF-RaBitQ-nl223-np21-rf0 (query)                        976.98       184.86     1_161.84       0.7404          1.0229            1.0220         3.17
IVF-RaBitQ-nl223-np11-rf10 (query)                       976.98       202.93     1_179.91       0.9976          1.0000            1.0000         3.17
IVF-RaBitQ-nl223-np11-rf20 (query)                       976.98       263.26     1_240.24       0.9976          1.0000            1.0000         3.17
IVF-RaBitQ-nl223-np14-rf10 (query)                       976.98       215.39     1_192.37       0.9976          1.0000            1.0000         3.17
IVF-RaBitQ-nl223-np14-rf20 (query)                       976.98       289.28     1_266.26       0.9976          1.0000            1.0000         3.17
IVF-RaBitQ-nl223-np21-rf10 (query)                       976.98       258.10     1_235.08       0.9976          1.0000            1.0000         3.17
IVF-RaBitQ-nl223-np21-rf20 (query)                       976.98       326.40     1_303.38       0.9976          1.0000            1.0000         3.17
IVF-RaBitQ-nl223 (self)                                  976.98     1_045.73     2_022.71       0.9976          1.0000            1.0000         3.17
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_160.33       147.62     1_307.96       0.7444          1.0222            1.0211         3.35
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_160.33       159.45     1_319.78       0.7444          1.0222            1.0211         3.35
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_160.33       205.57     1_365.90       0.7444          1.0222            1.0211         3.35
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_160.33       225.35     1_385.68       0.9976          1.0000            1.0000         3.35
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_160.33       286.10     1_446.43       0.9976          1.0000            1.0000         3.35
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_160.33       235.30     1_395.63       0.9976          1.0000            1.0000         3.35
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_160.33       297.57     1_457.90       0.9976          1.0000            1.0000         3.35
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_160.33       281.01     1_441.34       0.9976          1.0000            1.0000         3.35
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_160.33       352.35     1_512.68       0.9976          1.0000            1.0000         3.35
IVF-RaBitQ-nl316 (self)                                1_160.33     1_107.29     2_267.62       0.9976          1.0000            1.0000         3.35
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
Exhaustive (query)                                        67.83     1_343.78     1_411.61       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         67.83     4_493.71     4_561.54       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           2_256.08       428.54     2_684.61       0.7491          1.0144            1.0138         5.44
ExhaustiveRaBitQ-rf5 (query)                           2_256.08       484.74     2_740.81       0.9908          1.0000            1.0000         5.44
ExhaustiveRaBitQ-rf10 (query)                          2_256.08       549.74     2_805.82       0.9925          1.0000            1.0000         5.44
ExhaustiveRaBitQ-rf20 (query)                          2_256.08       652.94     2_909.02       0.9925          1.0000            1.0000         5.44
ExhaustiveRaBitQ (self)                                2_256.08     1_719.05     3_975.13       0.9925          1.0000            1.0000         5.44
IVF-RaBitQ-nl158-np7-rf0 (query)                       3_162.20       183.03     3_345.23       0.7509          1.0141            1.0135         5.63
IVF-RaBitQ-nl158-np12-rf0 (query)                      3_162.20       259.45     3_421.65       0.7509          1.0141            1.0135         5.63
IVF-RaBitQ-nl158-np17-rf0 (query)                      3_162.20       288.65     3_450.85       0.7509          1.0141            1.0135         5.63
IVF-RaBitQ-nl158-np7-rf10 (query)                      3_162.20       291.56     3_453.76       0.9925          1.0000            1.0000         5.63
IVF-RaBitQ-nl158-np7-rf20 (query)                      3_162.20       396.01     3_558.21       0.9925          1.0000            1.0000         5.63
IVF-RaBitQ-nl158-np12-rf10 (query)                     3_162.20       341.47     3_503.67       0.9925          1.0000            1.0000         5.63
IVF-RaBitQ-nl158-np12-rf20 (query)                     3_162.20       441.09     3_603.29       0.9925          1.0000            1.0000         5.63
IVF-RaBitQ-nl158-np17-rf10 (query)                     3_162.20       395.40     3_557.61       0.9925          1.0000            1.0000         5.63
IVF-RaBitQ-nl158-np17-rf20 (query)                     3_162.20       496.62     3_658.82       0.9925          1.0000            1.0000         5.63
IVF-RaBitQ-nl158 (self)                                3_162.20     1_557.79     4_719.99       0.9926          1.0000            1.0000         5.63
IVF-RaBitQ-nl223-np11-rf0 (query)                      2_288.06       231.42     2_519.48       0.7534          1.0138            1.0132         5.88
IVF-RaBitQ-nl223-np14-rf0 (query)                      2_288.06       274.12     2_562.18       0.7536          1.0138            1.0132         5.88
IVF-RaBitQ-nl223-np21-rf0 (query)                      2_288.06       345.67     2_633.74       0.7536          1.0138            1.0132         5.88
IVF-RaBitQ-nl223-np11-rf10 (query)                     2_288.06       335.15     2_623.22       0.9920          1.0000            1.0000         5.88
IVF-RaBitQ-nl223-np11-rf20 (query)                     2_288.06       433.15     2_721.21       0.9920          1.0000            1.0000         5.88
IVF-RaBitQ-nl223-np14-rf10 (query)                     2_288.06       370.35     2_658.41       0.9925          1.0000            1.0000         5.88
IVF-RaBitQ-nl223-np14-rf20 (query)                     2_288.06       465.57     2_753.63       0.9925          1.0000            1.0000         5.88
IVF-RaBitQ-nl223-np21-rf10 (query)                     2_288.06       440.62     2_728.68       0.9925          1.0000            1.0000         5.88
IVF-RaBitQ-nl223-np21-rf20 (query)                     2_288.06       540.99     2_829.05       0.9925          1.0000            1.0000         5.88
IVF-RaBitQ-nl223 (self)                                2_288.06     1_720.17     4_008.24       0.9926          1.0000            1.0000         5.88
IVF-RaBitQ-nl316-np15-rf0 (query)                      2_686.86       278.05     2_964.91       0.7548          1.0136            1.0130         6.24
IVF-RaBitQ-nl316-np17-rf0 (query)                      2_686.86       299.16     2_986.02       0.7548          1.0136            1.0130         6.24
IVF-RaBitQ-nl316-np25-rf0 (query)                      2_686.86       393.26     3_080.12       0.7548          1.0136            1.0130         6.24
IVF-RaBitQ-nl316-np15-rf10 (query)                     2_686.86       389.92     3_076.78       0.9925          1.0000            1.0000         6.24
IVF-RaBitQ-nl316-np15-rf20 (query)                     2_686.86       476.58     3_163.44       0.9925          1.0000            1.0000         6.24
IVF-RaBitQ-nl316-np17-rf10 (query)                     2_686.86       406.38     3_093.24       0.9925          1.0000            1.0000         6.24
IVF-RaBitQ-nl316-np17-rf20 (query)                     2_686.86       513.87     3_200.73       0.9925          1.0000            1.0000         6.24
IVF-RaBitQ-nl316-np25-rf10 (query)                     2_686.86       492.25     3_179.11       0.9925          1.0000            1.0000         6.24
IVF-RaBitQ-nl316-np25-rf20 (query)                     2_686.86       588.47     3_275.33       0.9925          1.0000            1.0000         6.24
IVF-RaBitQ-nl316 (self)                                2_686.86     1_851.81     4_538.66       0.9926          1.0000            1.0000         6.24
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
Exhaustive (query)                                        99.90     1_841.58     1_941.49       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.90     6_245.65     6_345.56       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           4_061.59       615.70     4_677.30       0.7291          1.0119            1.0114         8.44
ExhaustiveRaBitQ-rf5 (query)                           4_061.59       712.41     4_774.00       0.9764          1.0000            1.0000         8.44
ExhaustiveRaBitQ-rf10 (query)                          4_061.59       778.16     4_839.75       0.9790          0.9999            1.0000         8.44
ExhaustiveRaBitQ-rf20 (query)                          4_061.59       924.78     4_986.38       0.9791          0.9999            1.0000         8.44
ExhaustiveRaBitQ (self)                                4_061.59     2_509.28     6_570.87       0.9783          0.9999            1.0000         8.44
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_264.07       288.68     5_552.75       0.7326          1.0115            1.0111         8.71
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_264.07       357.31     5_621.39       0.7326          1.0115            1.0111         8.71
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_264.07       441.59     5_705.66       0.7326          1.0115            1.0111         8.71
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_264.07       412.18     5_676.26       0.9790          0.9999            1.0000         8.71
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_264.07       532.39     5_796.47       0.9791          0.9999            1.0000         8.71
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_264.07       490.86     5_754.93       0.9790          0.9999            1.0000         8.71
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_264.07       607.74     5_871.82       0.9791          0.9999            1.0000         8.71
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_264.07       562.08     5_826.15       0.9790          0.9999            1.0000         8.71
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_264.07       686.19     5_950.26       0.9791          0.9999            1.0000         8.71
IVF-RaBitQ-nl158 (self)                                5_264.07     2_183.75     7_447.83       0.9783          0.9999            1.0000         8.71
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_998.96       360.06     4_359.02       0.7338          1.0114            1.0109         9.09
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_998.96       396.61     4_395.58       0.7338          1.0114            1.0109         9.09
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_998.96       506.99     4_505.96       0.7338          1.0114            1.0109         9.09
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_998.96       476.26     4_475.22       0.9790          0.9999            1.0000         9.09
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_998.96       599.42     4_598.38       0.9791          0.9999            1.0000         9.09
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_998.96       520.39     4_519.35       0.9790          0.9999            1.0000         9.09
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_998.96       643.07     4_642.03       0.9791          0.9999            1.0000         9.09
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_998.96       632.21     4_631.17       0.9790          0.9999            1.0000         9.09
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_998.96       753.38     4_752.34       0.9791          0.9999            1.0000         9.09
IVF-RaBitQ-nl223 (self)                                3_998.96     2_437.00     6_435.96       0.9783          0.9999            1.0000         9.09
IVF-RaBitQ-nl316-np15-rf0 (query)                      4_542.43       418.87     4_961.29       0.7349          1.0113            1.0108         9.64
IVF-RaBitQ-nl316-np17-rf0 (query)                      4_542.43       449.46     4_991.89       0.7349          1.0113            1.0108         9.64
IVF-RaBitQ-nl316-np25-rf0 (query)                      4_542.43       588.00     5_130.42       0.7349          1.0113            1.0108         9.64
IVF-RaBitQ-nl316-np15-rf10 (query)                     4_542.43       550.49     5_092.92       0.9790          0.9999            1.0000         9.64
IVF-RaBitQ-nl316-np15-rf20 (query)                     4_542.43       692.77     5_235.20       0.9791          0.9999            1.0000         9.64
IVF-RaBitQ-nl316-np17-rf10 (query)                     4_542.43       586.87     5_129.29       0.9790          0.9999            1.0000         9.64
IVF-RaBitQ-nl316-np17-rf20 (query)                     4_542.43       707.65     5_250.07       0.9791          0.9999            1.0000         9.64
IVF-RaBitQ-nl316-np25-rf10 (query)                     4_542.43       700.98     5_243.41       0.9790          0.9999            1.0000         9.64
IVF-RaBitQ-nl316-np25-rf20 (query)                     4_542.43       821.88     5_364.30       0.9791          0.9999            1.0000         9.64
IVF-RaBitQ-nl316 (self)                                4_542.43     2_673.61     7_216.04       0.9783          0.9999            1.0000         9.64
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
Exhaustive (query)                                        32.84       707.96       740.81       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.84     2_351.51     2_384.35       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                           1_026.25       303.37     1_329.63       0.8680          1.0296            1.0242         2.95
ExhaustiveRaBitQ-rf5 (query)                           1_026.25       372.79     1_399.04       1.0000          1.0000            1.0000         2.95
ExhaustiveRaBitQ-rf10 (query)                          1_026.25       424.44     1_450.69       1.0000          1.0000            1.0000         2.95
ExhaustiveRaBitQ-rf20 (query)                          1_026.25       525.40     1_551.66       1.0000          1.0000            1.0000         2.95
ExhaustiveRaBitQ (self)                                1_026.25     1_387.12     2_413.37       1.0000          1.0000            1.0000         2.95
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_487.20       109.60     1_596.80       0.8728          1.0278            1.0225         3.04
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_487.20       161.56     1_648.76       0.8733          1.0275            1.0223         3.04
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_487.20       209.75     1_696.96       0.8733          1.0275            1.0223         3.04
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_487.20       194.38     1_681.58       0.9976          1.0005            1.0000         3.04
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_487.20       259.60     1_746.81       0.9976          1.0005            1.0000         3.04
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_487.20       242.95     1_730.15       0.9999          1.0000            1.0000         3.04
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_487.20       314.88     1_802.08       0.9999          1.0000            1.0000         3.04
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_487.20       293.54     1_780.74       1.0000          1.0000            1.0000         3.04
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_487.20       380.22     1_867.42       1.0000          1.0000            1.0000         3.04
IVF-RaBitQ-nl158 (self)                                1_487.20     1_192.73     2_679.94       1.0000          1.0000            1.0000         3.04
IVF-RaBitQ-nl223-np11-rf0 (query)                        835.41       132.46       967.87       0.8833          1.0228            1.0186         3.17
IVF-RaBitQ-nl223-np14-rf0 (query)                        835.41       157.06       992.47       0.8834          1.0227            1.0186         3.17
IVF-RaBitQ-nl223-np21-rf0 (query)                        835.41       215.28     1_050.69       0.8833          1.0228            1.0186         3.17
IVF-RaBitQ-nl223-np11-rf10 (query)                       835.41       211.68     1_047.09       0.9994          1.0001            1.0000         3.17
IVF-RaBitQ-nl223-np11-rf20 (query)                       835.41       278.79     1_114.20       0.9994          1.0001            1.0000         3.17
IVF-RaBitQ-nl223-np14-rf10 (query)                       835.41       234.26     1_069.67       0.9999          1.0000            1.0000         3.17
IVF-RaBitQ-nl223-np14-rf20 (query)                       835.41       302.78     1_138.19       0.9999          1.0000            1.0000         3.17
IVF-RaBitQ-nl223-np21-rf10 (query)                       835.41       293.52     1_128.93       1.0000          1.0000            1.0000         3.17
IVF-RaBitQ-nl223-np21-rf20 (query)                       835.41       362.56     1_197.97       1.0000          1.0000            1.0000         3.17
IVF-RaBitQ-nl223 (self)                                  835.41     1_182.07     2_017.48       1.0000          1.0000            1.0000         3.17
IVF-RaBitQ-nl316-np15-rf0 (query)                        982.74       157.94     1_140.68       0.8893          1.0202            1.0165         3.35
IVF-RaBitQ-nl316-np17-rf0 (query)                        982.74       173.80     1_156.54       0.8893          1.0202            1.0165         3.35
IVF-RaBitQ-nl316-np25-rf0 (query)                        982.74       233.34     1_216.08       0.8893          1.0202            1.0165         3.35
IVF-RaBitQ-nl316-np15-rf10 (query)                       982.74       235.77     1_218.51       0.9997          1.0000            1.0000         3.35
IVF-RaBitQ-nl316-np15-rf20 (query)                       982.74       304.80     1_287.54       0.9997          1.0000            1.0000         3.35
IVF-RaBitQ-nl316-np17-rf10 (query)                       982.74       250.02     1_232.76       0.9998          1.0000            1.0000         3.35
IVF-RaBitQ-nl316-np17-rf20 (query)                       982.74       317.37     1_300.11       0.9998          1.0000            1.0000         3.35
IVF-RaBitQ-nl316-np25-rf10 (query)                       982.74       313.91     1_296.65       1.0000          1.0000            1.0000         3.35
IVF-RaBitQ-nl316-np25-rf20 (query)                       982.74       377.44     1_360.18       1.0000          1.0000            1.0000         3.35
IVF-RaBitQ-nl316 (self)                                  982.74     1_225.46     2_208.20       1.0000          1.0000            1.0000         3.35
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
Exhaustive (query)                                        69.63     1_344.02     1_413.64       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.63     4_479.85     4_549.48       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           2_566.93       509.39     3_076.32       0.9024          1.0153            1.0116         5.44
ExhaustiveRaBitQ-rf5 (query)                           2_566.93       586.87     3_153.80       1.0000          1.0000            1.0000         5.44
ExhaustiveRaBitQ-rf10 (query)                          2_566.93       659.61     3_226.54       1.0000          1.0000            1.0000         5.44
ExhaustiveRaBitQ-rf20 (query)                          2_566.93       787.93     3_354.86       1.0000          1.0000            1.0000         5.44
ExhaustiveRaBitQ (self)                                2_566.93     2_086.82     4_653.75       1.0000          1.0000            1.0000         5.44
IVF-RaBitQ-nl158-np7-rf0 (query)                       3_460.22       201.59     3_661.82       0.9068          1.0138            1.0103         5.63
IVF-RaBitQ-nl158-np12-rf0 (query)                      3_460.22       284.64     3_744.86       0.9073          1.0135            1.0101         5.63
IVF-RaBitQ-nl158-np17-rf0 (query)                      3_460.22       380.63     3_840.85       0.9073          1.0135            1.0101         5.63
IVF-RaBitQ-nl158-np7-rf10 (query)                      3_460.22       310.83     3_771.06       0.9985          1.0003            1.0000         5.63
IVF-RaBitQ-nl158-np7-rf20 (query)                      3_460.22       412.50     3_872.72       0.9985          1.0003            1.0000         5.63
IVF-RaBitQ-nl158-np12-rf10 (query)                     3_460.22       387.88     3_848.11       0.9999          1.0000            1.0000         5.63
IVF-RaBitQ-nl158-np12-rf20 (query)                     3_460.22       488.29     3_948.52       0.9999          1.0000            1.0000         5.63
IVF-RaBitQ-nl158-np17-rf10 (query)                     3_460.22       469.45     3_929.67       1.0000          1.0000            1.0000         5.63
IVF-RaBitQ-nl158-np17-rf20 (query)                     3_460.22       574.13     4_034.36       1.0000          1.0000            1.0000         5.63
IVF-RaBitQ-nl158 (self)                                3_460.22     1_828.54     5_288.76       1.0000          1.0000            1.0000         5.63
IVF-RaBitQ-nl223-np11-rf0 (query)                      2_048.05       250.79     2_298.84       0.9151          1.0111            1.0083         5.88
IVF-RaBitQ-nl223-np14-rf0 (query)                      2_048.05       292.98     2_341.03       0.9152          1.0111            1.0083         5.88
IVF-RaBitQ-nl223-np21-rf0 (query)                      2_048.05       387.84     2_435.89       0.9152          1.0111            1.0083         5.88
IVF-RaBitQ-nl223-np11-rf10 (query)                     2_048.05       348.97     2_397.02       0.9997          1.0000            1.0000         5.88
IVF-RaBitQ-nl223-np11-rf20 (query)                     2_048.05       448.56     2_496.61       0.9997          1.0000            1.0000         5.88
IVF-RaBitQ-nl223-np14-rf10 (query)                     2_048.05       391.37     2_439.42       0.9999          1.0000            1.0000         5.88
IVF-RaBitQ-nl223-np14-rf20 (query)                     2_048.05       484.94     2_533.00       0.9999          1.0000            1.0000         5.88
IVF-RaBitQ-nl223-np21-rf10 (query)                     2_048.05       490.39     2_538.44       1.0000          1.0000            1.0000         5.88
IVF-RaBitQ-nl223-np21-rf20 (query)                     2_048.05       585.04     2_633.09       1.0000          1.0000            1.0000         5.88
IVF-RaBitQ-nl223 (self)                                2_048.05     1_878.78     3_926.83       1.0000          1.0000            1.0000         5.88
IVF-RaBitQ-nl316-np15-rf0 (query)                      2_307.05       295.97     2_603.02       0.9189          1.0100            1.0073         6.24
IVF-RaBitQ-nl316-np17-rf0 (query)                      2_307.05       321.92     2_628.97       0.9189          1.0100            1.0073         6.24
IVF-RaBitQ-nl316-np25-rf0 (query)                      2_307.05       432.59     2_739.65       0.9190          1.0100            1.0073         6.24
IVF-RaBitQ-nl316-np15-rf10 (query)                     2_307.05       402.00     2_709.06       0.9998          1.0000            1.0000         6.24
IVF-RaBitQ-nl316-np15-rf20 (query)                     2_307.05       492.24     2_799.29       0.9998          1.0000            1.0000         6.24
IVF-RaBitQ-nl316-np17-rf10 (query)                     2_307.05       420.53     2_727.58       0.9999          1.0000            1.0000         6.24
IVF-RaBitQ-nl316-np17-rf20 (query)                     2_307.05       530.77     2_837.82       0.9999          1.0000            1.0000         6.24
IVF-RaBitQ-nl316-np25-rf10 (query)                     2_307.05       545.97     2_853.02       1.0000          1.0000            1.0000         6.24
IVF-RaBitQ-nl316-np25-rf20 (query)                     2_307.05       626.82     2_933.87       1.0000          1.0000            1.0000         6.24
IVF-RaBitQ-nl316 (self)                                2_307.05     2_013.93     4_320.98       0.9999          1.0000            1.0000         6.24
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
Exhaustive (query)                                       100.61     1_879.91     1_980.53       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.61     6_305.99     6_406.61       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           4_399.87       715.51     5_115.39       0.9249          1.0085            1.0061         8.44
ExhaustiveRaBitQ-rf5 (query)                           4_399.87       810.99     5_210.87       1.0000          1.0000            1.0000         8.44
ExhaustiveRaBitQ-rf10 (query)                          4_399.87       877.16     5_277.03       1.0000          1.0000            1.0000         8.44
ExhaustiveRaBitQ-rf20 (query)                          4_399.87     1_026.48     5_426.36       1.0000          1.0000            1.0000         8.44
ExhaustiveRaBitQ (self)                                4_399.87     2_847.41     7_247.28       0.9999          1.0000            1.0000         8.44
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_715.81       313.03     6_028.84       0.9274          1.0078            1.0055         8.71
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_715.81       428.14     6_143.95       0.9276          1.0078            1.0055         8.71
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_715.81       540.39     6_256.20       0.9276          1.0078            1.0055         8.71
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_715.81       432.45     6_148.25       0.9995          1.0001            1.0000         8.71
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_715.81       550.29     6_266.10       0.9995          1.0001            1.0000         8.71
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_715.81       542.32     6_258.13       1.0000          1.0000            1.0000         8.71
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_715.81       667.24     6_383.05       1.0000          1.0000            1.0000         8.71
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_715.81       655.49     6_371.30       1.0000          1.0000            1.0000         8.71
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_715.81       773.93     6_489.73       1.0000          1.0000            1.0000         8.71
IVF-RaBitQ-nl158 (self)                                5_715.81     2_500.86     8_216.66       0.9999          1.0000            1.0000         8.71
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_762.51       404.43     4_166.95       0.9323          1.0067            1.0046         9.09
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_762.51       433.78     4_196.29       0.9323          1.0067            1.0046         9.09
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_762.51       574.01     4_336.52       0.9323          1.0067            1.0046         9.09
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_762.51       494.94     4_257.46       0.9998          1.0000            1.0000         9.09
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_762.51       614.94     4_377.46       0.9998          1.0000            1.0000         9.09
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_762.51       562.07     4_324.58       0.9999          1.0000            1.0000         9.09
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_762.51       672.71     4_435.22       0.9999          1.0000            1.0000         9.09
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_762.51       710.06     4_472.57       1.0000          1.0000            1.0000         9.09
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_762.51       816.99     4_579.50       1.0000          1.0000            1.0000         9.09
IVF-RaBitQ-nl223 (self)                                3_762.51     2_718.04     6_480.56       0.9999          1.0000            1.0000         9.09
IVF-RaBitQ-nl316-np15-rf0 (query)                      4_320.83       445.04     4_765.87       0.9360          1.0060            1.0038         9.64
IVF-RaBitQ-nl316-np17-rf0 (query)                      4_320.83       481.43     4_802.26       0.9360          1.0060            1.0038         9.64
IVF-RaBitQ-nl316-np25-rf0 (query)                      4_320.83       671.04     4_991.87       0.9360          1.0060            1.0038         9.64
IVF-RaBitQ-nl316-np15-rf10 (query)                     4_320.83       569.65     4_890.48       0.9999          1.0000            1.0000         9.64
IVF-RaBitQ-nl316-np15-rf20 (query)                     4_320.83       688.62     5_009.45       0.9999          1.0000            1.0000         9.64
IVF-RaBitQ-nl316-np17-rf10 (query)                     4_320.83       598.95     4_919.78       1.0000          1.0000            1.0000         9.64
IVF-RaBitQ-nl316-np17-rf20 (query)                     4_320.83       725.86     5_046.69       1.0000          1.0000            1.0000         9.64
IVF-RaBitQ-nl316-np25-rf10 (query)                     4_320.83       750.71     5_071.54       1.0000          1.0000            1.0000         9.64
IVF-RaBitQ-nl316-np25-rf20 (query)                     4_320.83       877.08     5_197.91       1.0000          1.0000            1.0000         9.64
IVF-RaBitQ-nl316 (self)                                4_320.83     2_835.00     7_155.83       0.9999          1.0000            1.0000         9.64
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
Exhaustive (query)                                        32.96       680.24       713.19       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.96     2_239.48     2_272.44       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              153.21       358.21       511.42       0.0971          1.7176            1.5958         7.12
ExhaustiveTQ-b2-rf5 (query)                              153.21       442.37       595.58       0.2336          1.2025            1.2204         7.12
ExhaustiveTQ-b2-rf10 (query)                             153.21       582.79       736.00       0.2854          1.1453            1.1620         7.12
ExhaustiveTQ-b2-rf20 (query)                             153.21     1_007.68     1_160.89       0.3808          1.0970            1.0941         7.12
ExhaustiveTQ-b2 (self)                                   153.21     3_220.63     3_373.84       0.3816          1.0980            1.0957         7.12
ExhaustiveTQ-b4-rf0 (query)                              226.83       593.62       820.46       0.1094          1.5328            1.4996        13.22
ExhaustiveTQ-b4-rf5 (query)                              226.83       662.72       889.56       0.2368          1.1884            1.2090        13.22
ExhaustiveTQ-b4-rf10 (query)                             226.83       800.08     1_026.91       0.2885          1.1372            1.1543        13.22
ExhaustiveTQ-b4-rf20 (query)                             226.83     1_196.02     1_422.85       0.3822          1.0940            1.0970        13.22
ExhaustiveTQ-b4 (self)                                   226.83     3_969.14     4_195.97       0.3840          1.0938            1.0948        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                          906.88       108.21     1_015.09       0.0971          1.7176            1.5958         7.80
IVF-TQ-b2-nl158-np12-rf0 (query)                         906.88       121.89     1_028.77       0.0971          1.7176            1.5958         7.80
IVF-TQ-b2-nl158-np17-rf0 (query)                         906.88       129.63     1_036.52       0.0971          1.7176            1.5958         7.80
IVF-TQ-b2-nl158-np7-rf10 (query)                         906.88       314.02     1_220.91       0.2854          1.1453            1.1619         7.80
IVF-TQ-b2-nl158-np7-rf20 (query)                         906.88       638.00     1_544.89       0.3808          1.0970            1.0941         7.80
IVF-TQ-b2-nl158-np12-rf10 (query)                        906.88       327.09     1_233.97       0.2854          1.1453            1.1619         7.80
IVF-TQ-b2-nl158-np12-rf20 (query)                        906.88       672.06     1_578.95       0.3808          1.0970            1.0941         7.80
IVF-TQ-b2-nl158-np17-rf10 (query)                        906.88       336.04     1_242.93       0.2854          1.1453            1.1619         7.80
IVF-TQ-b2-nl158-np17-rf20 (query)                        906.88       688.17     1_595.06       0.3808          1.0970            1.0941         7.80
IVF-TQ-b2-nl158 (self)                                   906.88     1_069.80     1_976.68       0.3816          1.0980            1.0957         7.80
IVF-TQ-b2-nl223-np11-rf0 (query)                         690.02       117.28       807.31       0.0971          1.7164            1.5942         7.93
IVF-TQ-b2-nl223-np14-rf0 (query)                         690.02       123.11       813.13       0.0971          1.7176            1.5958         7.93
IVF-TQ-b2-nl223-np21-rf0 (query)                         690.02       135.87       825.89       0.0971          1.7176            1.5958         7.93
IVF-TQ-b2-nl223-np11-rf10 (query)                        690.02       294.24       984.26       0.2856          1.1450            1.1618         7.93
IVF-TQ-b2-nl223-np11-rf20 (query)                        690.02       568.34     1_258.36       0.3813          1.0967            1.0934         7.93
IVF-TQ-b2-nl223-np14-rf10 (query)                        690.02       299.42       989.45       0.2854          1.1453            1.1620         7.93
IVF-TQ-b2-nl223-np14-rf20 (query)                        690.02       582.49     1_272.51       0.3808          1.0970            1.0941         7.93
IVF-TQ-b2-nl223-np21-rf10 (query)                        690.02       319.58     1_009.60       0.2854          1.1453            1.1620         7.93
IVF-TQ-b2-nl223-np21-rf20 (query)                        690.02       606.25     1_296.27       0.3808          1.0970            1.0941         7.93
IVF-TQ-b2-nl223 (self)                                   690.02     1_070.44     1_760.46       0.3816          1.0980            1.0957         7.93
IVF-TQ-b2-nl316-np15-rf0 (query)                         879.75       123.56     1_003.32       0.0973          1.6435            1.5781         8.10
IVF-TQ-b2-nl316-np17-rf0 (query)                         879.75       125.54     1_005.30       0.0972          1.7163            1.5957         8.10
IVF-TQ-b2-nl316-np25-rf0 (query)                         879.75       138.98     1_018.74       0.0971          1.7176            1.5958         8.10
IVF-TQ-b2-nl316-np15-rf10 (query)                        879.75       291.08     1_170.83       0.2859          1.1447            1.1615         8.10
IVF-TQ-b2-nl316-np15-rf20 (query)                        879.75       553.58     1_433.34       0.3816          1.0965            1.0931         8.10
IVF-TQ-b2-nl316-np17-rf10 (query)                        879.75       299.94     1_179.69       0.2854          1.1453            1.1619         8.10
IVF-TQ-b2-nl316-np17-rf20 (query)                        879.75       561.72     1_441.47       0.3808          1.0970            1.0941         8.10
IVF-TQ-b2-nl316-np25-rf10 (query)                        879.75       312.63     1_192.39       0.2854          1.1453            1.1620         8.10
IVF-TQ-b2-nl316-np25-rf20 (query)                        879.75       597.56     1_477.31       0.3808          1.0970            1.0941         8.10
IVF-TQ-b2-nl316 (self)                                   879.75     1_076.25     1_956.00       0.3816          1.0980            1.0957         8.10
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_000.01       149.39     1_149.41       0.1094          1.5328            1.4996        14.05
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_000.01       166.46     1_166.48       0.1094          1.5328            1.4996        14.05
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_000.01       186.33     1_186.34       0.1094          1.5328            1.4996        14.05
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_000.01       363.35     1_363.36       0.2885          1.1372            1.1543        14.05
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_000.01       698.75     1_698.76       0.3823          1.0940            1.0970        14.05
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_000.01       385.05     1_385.07       0.2885          1.1372            1.1543        14.05
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_000.01       735.11     1_735.12       0.3822          1.0940            1.0970        14.05
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_000.01       395.69     1_395.71       0.2885          1.1372            1.1543        14.05
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_000.01       761.18     1_761.19       0.3823          1.0940            1.0970        14.05
IVF-TQ-b4-nl158 (self)                                 1_000.01     1_098.34     2_098.35       0.3840          1.0938            1.0948        14.05
IVF-TQ-b4-nl223-np11-rf0 (query)                         746.80       158.52       905.32       0.1095          1.5317            1.4988        14.25
IVF-TQ-b4-nl223-np14-rf0 (query)                         746.80       171.94       918.74       0.1094          1.5328            1.4996        14.25
IVF-TQ-b4-nl223-np21-rf0 (query)                         746.80       192.02       938.82       0.1094          1.5328            1.4996        14.25
IVF-TQ-b4-nl223-np11-rf10 (query)                        746.80       342.75     1_089.55       0.2887          1.1370            1.1541        14.25
IVF-TQ-b4-nl223-np11-rf20 (query)                        746.80       621.61     1_368.42       0.3825          1.0938            1.0966        14.25
IVF-TQ-b4-nl223-np14-rf10 (query)                        746.80       355.78     1_102.58       0.2885          1.1372            1.1543        14.25
IVF-TQ-b4-nl223-np14-rf20 (query)                        746.80       698.91     1_445.71       0.3822          1.0940            1.0970        14.25
IVF-TQ-b4-nl223-np21-rf10 (query)                        746.80       379.11     1_125.92       0.2885          1.1372            1.1543        14.25
IVF-TQ-b4-nl223-np21-rf20 (query)                        746.80       677.89     1_424.70       0.3822          1.0940            1.0970        14.25
IVF-TQ-b4-nl223 (self)                                   746.80     1_110.14     1_856.94       0.3840          1.0938            1.0948        14.25
IVF-TQ-b4-nl316-np15-rf0 (query)                         971.50       165.11     1_136.61       0.1094          1.5304            1.4978        14.49
IVF-TQ-b4-nl316-np17-rf0 (query)                         971.50       171.62     1_143.12       0.1094          1.5328            1.4996        14.49
IVF-TQ-b4-nl316-np25-rf0 (query)                         971.50       192.69     1_164.19       0.1094          1.5328            1.4996        14.49
IVF-TQ-b4-nl316-np15-rf10 (query)                        971.50       374.87     1_346.37       0.2887          1.1369            1.1541        14.49
IVF-TQ-b4-nl316-np15-rf20 (query)                        971.50       612.75     1_584.25       0.3827          1.0937            1.0967        14.49
IVF-TQ-b4-nl316-np17-rf10 (query)                        971.50       351.66     1_323.16       0.2885          1.1372            1.1543        14.49
IVF-TQ-b4-nl316-np17-rf20 (query)                        971.50       624.73     1_596.23       0.3822          1.0940            1.0970        14.49
IVF-TQ-b4-nl316-np25-rf10 (query)                        971.50       378.32     1_349.82       0.2885          1.1372            1.1543        14.49
IVF-TQ-b4-nl316-np25-rf20 (query)                        971.50       669.80     1_641.30       0.3822          1.0940            1.0970        14.49
IVF-TQ-b4-nl316 (self)                                   971.50     1_121.36     2_092.86       0.3840          1.0938            1.0948        14.49
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
Exhaustive (query)                                        68.15     1_344.28     1_412.42       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.15     4_559.94     4_628.08       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              344.11       652.48       996.59       0.1208          1.3711            1.3320        13.97
ExhaustiveTQ-b2-rf5 (query)                              344.11       737.37     1_081.47       0.2421          1.1333            1.1573        13.97
ExhaustiveTQ-b2-rf10 (query)                             344.11       885.45     1_229.56       0.2934          1.0981            1.1177        13.97
ExhaustiveTQ-b2-rf20 (query)                             344.11     1_281.02     1_625.12       0.3881          1.0664            1.0469        13.97
ExhaustiveTQ-b2 (self)                                   344.11     4_198.84     4_542.95       0.3879          1.0667            1.0471        13.97
ExhaustiveTQ-b4-rf0 (query)                              466.64     1_150.47     1_617.11       0.1314          1.3172            1.3126        26.18
ExhaustiveTQ-b4-rf5 (query)                              466.64     1_251.47     1_718.12       0.2469          1.1254            1.1483        26.18
ExhaustiveTQ-b4-rf10 (query)                             466.64     1_458.02     1_924.67       0.2968          1.0928            1.0979        26.18
ExhaustiveTQ-b4-rf20 (query)                             466.64     1_792.69     2_259.33       0.3881          1.0643            1.0492        26.18
ExhaustiveTQ-b4 (self)                                   466.64     6_129.32     6_595.97       0.3881          1.0646            1.0495        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_937.06       204.00     2_141.05       0.1208          1.3711            1.3320        14.95
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_937.06       213.10     2_150.16       0.1208          1.3711            1.3320        14.95
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_937.06       225.06     2_162.11       0.1208          1.3711            1.3320        14.95
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_937.06       427.04     2_364.09       0.2934          1.0981            1.1177        14.95
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_937.06       770.00     2_707.05       0.3881          1.0664            1.0469        14.95
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_937.06       436.03     2_373.09       0.2934          1.0981            1.1177        14.95
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_937.06       813.19     2_750.24       0.3881          1.0664            1.0469        14.95
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_937.06       456.19     2_393.25       0.2934          1.0981            1.1177        14.95
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_937.06       819.82     2_756.87       0.3881          1.0664            1.0469        14.95
IVF-TQ-b2-nl158 (self)                                 1_937.06     1_426.79     3_363.85       0.3879          1.0667            1.0471        14.95
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_216.08       212.05     1_428.13       0.1208          1.3698            1.3299        15.19
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_216.08       222.67     1_438.75       0.1208          1.3711            1.3320        15.19
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_216.08       248.20     1_464.28       0.1208          1.3711            1.3320        15.19
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_216.08       410.84     1_626.93       0.2937          1.0979            1.1176        15.19
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_216.08       727.91     1_943.99       0.3887          1.0662            1.0466        15.19
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_216.08       426.27     1_642.36       0.2934          1.0981            1.1177        15.19
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_216.08       745.02     1_961.11       0.3881          1.0664            1.0469        15.19
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_216.08       447.69     1_663.78       0.2934          1.0981            1.1177        15.19
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_216.08       786.13     2_002.21       0.3881          1.0664            1.0469        15.19
IVF-TQ-b2-nl223 (self)                                 1_216.08     1_403.13     2_619.21       0.3879          1.0667            1.0471        15.19
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_529.09       221.22     1_750.31       0.1208          1.3689            1.3287        15.56
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_529.09       226.08     1_755.17       0.1208          1.3707            1.3311        15.56
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_529.09       246.28     1_775.36       0.1208          1.3711            1.3320        15.56
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_529.09       408.55     1_937.64       0.2938          1.0976            1.1175        15.56
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_529.09       692.43     2_221.52       0.3892          1.0660            1.0465        15.56
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_529.09       423.37     1_952.46       0.2934          1.0980            1.1176        15.56
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_529.09       715.61     2_244.69       0.3883          1.0663            1.0469        15.56
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_529.09       449.70     1_978.79       0.2934          1.0981            1.1177        15.56
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_529.09       766.28     2_295.37       0.3881          1.0664            1.0469        15.56
IVF-TQ-b2-nl316 (self)                                 1_529.09     1_436.79     2_965.88       0.3879          1.0667            1.0471        15.56
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_993.01       274.16     2_267.17       0.1314          1.3172            1.3127        27.44
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_993.01       300.64     2_293.65       0.1314          1.3172            1.3126        27.44
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_993.01       325.46     2_318.47       0.1314          1.3172            1.3127        27.44
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_993.01       509.77     2_502.78       0.2968          1.0928            1.0979        27.44
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_993.01       862.17     2_855.18       0.3880          1.0643            1.0492        27.44
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_993.01       539.06     2_532.07       0.2969          1.0928            1.0979        27.44
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_993.01       907.97     2_900.98       0.3880          1.0643            1.0492        27.44
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_993.01       567.31     2_560.33       0.2969          1.0928            1.0979        27.44
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_993.01       959.12     2_952.13       0.3880          1.0643            1.0492        27.44
IVF-TQ-b4-nl158 (self)                                 1_993.01     1_565.73     3_558.74       0.3881          1.0646            1.0495        27.44
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_329.37       293.54     1_622.91       0.1315          1.3158            1.3116        27.79
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_329.37       310.00     1_639.37       0.1314          1.3172            1.3126        27.79
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_329.37       345.93     1_675.31       0.1314          1.3172            1.3126        27.79
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_329.37       504.69     1_834.06       0.2972          1.0926            1.0973        27.79
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_329.37       823.87     2_153.24       0.3887          1.0641            1.0489        27.79
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_329.37       524.96     1_854.33       0.2968          1.0928            1.0979        27.79
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_329.37       846.49     2_175.86       0.3881          1.0643            1.0492        27.79
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_329.37       563.52     1_892.89       0.2968          1.0928            1.0979        27.79
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_329.37       904.55     2_233.92       0.3880          1.0643            1.0492        27.79
IVF-TQ-b4-nl223 (self)                                 1_329.37     1_608.79     2_938.16       0.3881          1.0646            1.0495        27.79
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_653.78       307.29     1_961.07       0.1315          1.3151            1.3108        28.35
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_653.78       325.12     1_978.90       0.1315          1.3164            1.3120        28.35
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_653.78       369.40     2_023.18       0.1314          1.3172            1.3126        28.35
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_653.78       536.89     2_190.67       0.2974          1.0925            1.0966        28.35
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_653.78       834.42     2_488.19       0.3891          1.0639            1.0484        28.35
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_653.78       526.32     2_180.10       0.2970          1.0928            1.0977        28.35
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_653.78       827.64     2_481.42       0.3883          1.0642            1.0491        28.35
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_653.78       569.87     2_223.65       0.2968          1.0928            1.0979        28.35
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_653.78       907.45     2_561.23       0.3881          1.0643            1.0492        28.35
IVF-TQ-b4-nl316 (self)                                 1_653.78     1_667.60     3_321.38       0.3881          1.0646            1.0495        28.35
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
Exhaustive (query)                                        99.77     1_914.91     2_014.68       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.77     6_697.49     6_797.25       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              610.70       957.11     1_567.81       0.1292          1.2709            1.2627        21.33
ExhaustiveTQ-b2-rf5 (query)                              610.70     1_050.23     1_660.93       0.2468          1.1062            1.1332        21.33
ExhaustiveTQ-b2-rf10 (query)                             610.70     1_199.95     1_810.65       0.3001          1.0773            1.0630        21.33
ExhaustiveTQ-b2-rf20 (query)                             610.70     1_626.63     2_237.34       0.3961          1.0509            1.0334        21.33
ExhaustiveTQ-b2 (self)                                   610.70     5_311.02     5_921.72       0.3973          1.0507            1.0331        21.33
ExhaustiveTQ-b4-rf0 (query)                              738.59     1_772.65     2_511.24       0.1340          1.2531            1.2591        39.64
ExhaustiveTQ-b4-rf5 (query)                              738.59     1_872.71     2_611.30       0.2401          1.1135            1.1401        39.64
ExhaustiveTQ-b4-rf10 (query)                             738.59     2_007.78     2_746.37       0.2871          1.0888            1.1142        39.64
ExhaustiveTQ-b4-rf20 (query)                             738.59     2_413.92     3_152.51       0.3752          1.0657            1.0812        39.64
ExhaustiveTQ-b4 (self)                                   738.59     8_074.02     8_812.61       0.3767          1.0653            1.0638        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        2_770.07       298.30     3_068.37       0.1292          1.2709            1.2627        22.66
IVF-TQ-b2-nl158-np12-rf0 (query)                       2_770.07       320.97     3_091.04       0.1292          1.2709            1.2627        22.66
IVF-TQ-b2-nl158-np17-rf0 (query)                       2_770.07       342.07     3_112.14       0.1292          1.2709            1.2627        22.66
IVF-TQ-b2-nl158-np7-rf10 (query)                       2_770.07       537.10     3_307.17       0.3001          1.0773            1.0630        22.66
IVF-TQ-b2-nl158-np7-rf20 (query)                       2_770.07       899.99     3_670.06       0.3961          1.0509            1.0334        22.66
IVF-TQ-b2-nl158-np12-rf10 (query)                      2_770.07       562.38     3_332.45       0.3001          1.0773            1.0630        22.66
IVF-TQ-b2-nl158-np12-rf20 (query)                      2_770.07       948.75     3_718.82       0.3961          1.0509            1.0334        22.66
IVF-TQ-b2-nl158-np17-rf10 (query)                      2_770.07       587.22     3_357.29       0.3001          1.0773            1.0630        22.66
IVF-TQ-b2-nl158-np17-rf20 (query)                      2_770.07       974.37     3_744.44       0.3961          1.0509            1.0334        22.66
IVF-TQ-b2-nl158 (self)                                 2_770.07     1_892.22     4_662.29       0.3973          1.0507            1.0331        22.66
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_885.54       316.54     2_202.09       0.1292          1.2709            1.2627        23.04
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_885.54       359.75     2_245.29       0.1292          1.2709            1.2627        23.04
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_885.54       372.96     2_258.51       0.1292          1.2709            1.2627        23.04
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_885.54       580.75     2_466.29       0.3001          1.0773            1.0630        23.04
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_885.54       859.39     2_744.93       0.3961          1.0509            1.0334        23.04
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_885.54       562.66     2_448.20       0.3001          1.0773            1.0630        23.04
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_885.54       891.52     2_777.07       0.3961          1.0509            1.0334        23.04
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_885.54       595.78     2_481.32       0.3001          1.0773            1.0630        23.04
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_885.54       935.07     2_820.61       0.3961          1.0509            1.0333        23.04
IVF-TQ-b2-nl223 (self)                                 1_885.54     1_889.84     3_775.38       0.3973          1.0507            1.0331        23.04
IVF-TQ-b2-nl316-np15-rf0 (query)                       2_228.93       329.84     2_558.77       0.1292          1.2709            1.2627        23.57
IVF-TQ-b2-nl316-np17-rf0 (query)                       2_228.93       338.76     2_567.69       0.1292          1.2709            1.2627        23.57
IVF-TQ-b2-nl316-np25-rf0 (query)                       2_228.93       382.55     2_611.48       0.1292          1.2709            1.2627        23.57
IVF-TQ-b2-nl316-np15-rf10 (query)                      2_228.93       552.52     2_781.45       0.3001          1.0773            1.0630        23.57
IVF-TQ-b2-nl316-np15-rf20 (query)                      2_228.93       866.24     3_095.16       0.3962          1.0508            1.0333        23.57
IVF-TQ-b2-nl316-np17-rf10 (query)                      2_228.93       558.44     2_787.37       0.3001          1.0773            1.0631        23.57
IVF-TQ-b2-nl316-np17-rf20 (query)                      2_228.93       878.80     3_107.73       0.3961          1.0509            1.0334        23.57
IVF-TQ-b2-nl316-np25-rf10 (query)                      2_228.93       601.93     2_830.86       0.3001          1.0773            1.0630        23.57
IVF-TQ-b2-nl316-np25-rf20 (query)                      2_228.93       936.57     3_165.50       0.3961          1.0509            1.0334        23.57
IVF-TQ-b2-nl316 (self)                                 2_228.93     1_912.16     4_141.09       0.3973          1.0507            1.0331        23.57
IVF-TQ-b4-nl158-np7-rf0 (query)                        2_839.50       420.03     3_259.52       0.1340          1.2531            1.2591        41.46
IVF-TQ-b4-nl158-np12-rf0 (query)                       2_839.50       466.79     3_306.29       0.1340          1.2531            1.2591        41.46
IVF-TQ-b4-nl158-np17-rf0 (query)                       2_839.50       507.23     3_346.73       0.1340          1.2531            1.2592        41.46
IVF-TQ-b4-nl158-np7-rf10 (query)                       2_839.50       679.91     3_519.41       0.2871          1.0888            1.1142        41.46
IVF-TQ-b4-nl158-np7-rf20 (query)                       2_839.50     1_045.87     3_885.37       0.3752          1.0657            1.0812        41.46
IVF-TQ-b4-nl158-np12-rf10 (query)                      2_839.50       751.33     3_590.83       0.2871          1.0888            1.1142        41.46
IVF-TQ-b4-nl158-np12-rf20 (query)                      2_839.50     1_113.61     3_953.11       0.3752          1.0657            1.0812        41.46
IVF-TQ-b4-nl158-np17-rf10 (query)                      2_839.50       767.14     3_606.64       0.2871          1.0888            1.1142        41.46
IVF-TQ-b4-nl158-np17-rf20 (query)                      2_839.50     1_159.48     3_998.98       0.3752          1.0657            1.0812        41.46
IVF-TQ-b4-nl158 (self)                                 2_839.50     2_138.26     4_977.76       0.3767          1.0653            1.0637        41.46
IVF-TQ-b4-nl223-np11-rf0 (query)                       2_007.66       488.16     2_495.81       0.1340          1.2531            1.2591        42.04
IVF-TQ-b4-nl223-np14-rf0 (query)                       2_007.66       484.06     2_491.71       0.1340          1.2531            1.2591        42.04
IVF-TQ-b4-nl223-np21-rf0 (query)                       2_007.66       541.09     2_548.75       0.1340          1.2531            1.2591        42.04
IVF-TQ-b4-nl223-np11-rf10 (query)                      2_007.66       692.03     2_699.68       0.2871          1.0888            1.1142        42.04
IVF-TQ-b4-nl223-np11-rf20 (query)                      2_007.66     1_012.26     3_019.91       0.3752          1.0657            1.0812        42.04
IVF-TQ-b4-nl223-np14-rf10 (query)                      2_007.66       720.33     2_727.98       0.2871          1.0888            1.1142        42.04
IVF-TQ-b4-nl223-np14-rf20 (query)                      2_007.66     1_054.87     3_062.52       0.3752          1.0657            1.0812        42.04
IVF-TQ-b4-nl223-np21-rf10 (query)                      2_007.66       780.22     2_787.87       0.2871          1.0888            1.1142        42.04
IVF-TQ-b4-nl223-np21-rf20 (query)                      2_007.66     1_134.31     3_141.96       0.3752          1.0657            1.0812        42.04
IVF-TQ-b4-nl223 (self)                                 2_007.66     2_239.64     4_247.30       0.3767          1.0653            1.0638        42.04
IVF-TQ-b4-nl316-np15-rf0 (query)                       2_438.85       485.29     2_924.14       0.1340          1.2530            1.2591        42.81
IVF-TQ-b4-nl316-np17-rf0 (query)                       2_438.85       522.02     2_960.87       0.1340          1.2531            1.2591        42.81
IVF-TQ-b4-nl316-np25-rf0 (query)                       2_438.85       561.90     3_000.75       0.1340          1.2531            1.2591        42.81
IVF-TQ-b4-nl316-np15-rf10 (query)                      2_438.85       714.34     3_153.19       0.2871          1.0887            1.1142        42.81
IVF-TQ-b4-nl316-np15-rf20 (query)                      2_438.85     1_021.69     3_460.54       0.3753          1.0657            1.0812        42.81
IVF-TQ-b4-nl316-np17-rf10 (query)                      2_438.85       721.38     3_160.23       0.2871          1.0888            1.1142        42.81
IVF-TQ-b4-nl316-np17-rf20 (query)                      2_438.85     1_048.20     3_487.06       0.3752          1.0657            1.0812        42.81
IVF-TQ-b4-nl316-np25-rf10 (query)                      2_438.85       787.64     3_226.49       0.2871          1.0888            1.1142        42.81
IVF-TQ-b4-nl316-np25-rf20 (query)                      2_438.85     1_169.94     3_608.80       0.3752          1.0657            1.0812        42.81
IVF-TQ-b4-nl316 (self)                                 2_438.85     2_280.37     4_719.22       0.3767          1.0653            1.0637        42.81
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
Exhaustive (query)                                        33.89       736.36       770.25       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.89     2_444.14     2_478.04       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              152.26       373.10       525.36       0.0755          2.3282            1.9295         7.12
ExhaustiveTQ-b2-rf5 (query)                              152.26       447.21       599.47       0.2072          1.3307            1.3578         7.12
ExhaustiveTQ-b2-rf10 (query)                             152.26       585.71       737.97       0.2887          1.2206            1.2322         7.12
ExhaustiveTQ-b2-rf20 (query)                             152.26       978.11     1_130.37       0.4152          1.1328            1.1147         7.12
ExhaustiveTQ-b2 (self)                                   152.26     3_244.16     3_396.43       0.4135          1.1619            1.1367         7.12
ExhaustiveTQ-b4-rf0 (query)                              231.19       630.78       861.98       0.1023          1.7129            1.7532        13.22
ExhaustiveTQ-b4-rf5 (query)                              231.19       715.21       946.40       0.2385          1.2770            1.3000        13.22
ExhaustiveTQ-b4-rf10 (query)                             231.19       851.77     1_082.96       0.3201          1.1873            1.1953        13.22
ExhaustiveTQ-b4-rf20 (query)                             231.19     1_204.76     1_435.96       0.4481          1.1142            1.1029        13.22
ExhaustiveTQ-b4 (self)                                   231.19     3_978.47     4_209.67       0.4462          1.1397            1.1286        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_041.48       105.13     1_146.61       0.0756          2.3282            1.9295         7.81
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_041.48       117.57     1_159.05       0.0755          2.3282            1.9295         7.81
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_041.48       139.42     1_180.90       0.0755          2.3283            1.9295         7.81
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_041.48       311.24     1_352.72       0.2887          1.2206            1.2322         7.81
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_041.48       640.41     1_681.89       0.4152          1.1328            1.1147         7.81
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_041.48       323.09     1_364.57       0.2887          1.2206            1.2322         7.81
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_041.48       664.81     1_706.29       0.4152          1.1328            1.1147         7.81
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_041.48       353.75     1_395.23       0.2887          1.2206            1.2322         7.81
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_041.48       738.67     1_780.15       0.4152          1.1328            1.1147         7.81
IVF-TQ-b2-nl158 (self)                                 1_041.48     1_085.09     2_126.57       0.4135          1.1619            1.1367         7.81
IVF-TQ-b2-nl223-np11-rf0 (query)                         716.96       114.36       831.32       0.0756          2.3256            1.9254         7.94
IVF-TQ-b2-nl223-np14-rf0 (query)                         716.96       120.53       837.50       0.0756          2.3281            1.9295         7.94
IVF-TQ-b2-nl223-np21-rf0 (query)                         716.96       146.94       863.91       0.0756          2.3282            1.9295         7.94
IVF-TQ-b2-nl223-np11-rf10 (query)                        716.96       292.02     1_008.98       0.2891          1.2203            1.2319         7.94
IVF-TQ-b2-nl223-np11-rf20 (query)                        716.96       570.07     1_287.04       0.4157          1.1326            1.1144         7.94
IVF-TQ-b2-nl223-np14-rf10 (query)                        716.96       298.96     1_015.93       0.2887          1.2206            1.2322         7.94
IVF-TQ-b2-nl223-np14-rf20 (query)                        716.96       588.23     1_305.19       0.4152          1.1328            1.1147         7.94
IVF-TQ-b2-nl223-np21-rf10 (query)                        716.96       337.98     1_054.94       0.2887          1.2206            1.2322         7.94
IVF-TQ-b2-nl223-np21-rf20 (query)                        716.96       657.67     1_374.63       0.4152          1.1328            1.1147         7.94
IVF-TQ-b2-nl223 (self)                                   716.96     1_086.54     1_803.50       0.4135          1.1619            1.1367         7.94
IVF-TQ-b2-nl316-np15-rf0 (query)                         921.39       118.60     1_039.99       0.0756          2.3274            1.9289         8.11
IVF-TQ-b2-nl316-np17-rf0 (query)                         921.39       121.93     1_043.31       0.0756          2.3282            1.9294         8.11
IVF-TQ-b2-nl316-np25-rf0 (query)                         921.39       143.36     1_064.75       0.0756          2.3282            1.9295         8.11
IVF-TQ-b2-nl316-np15-rf10 (query)                        921.39       285.67     1_207.06       0.2893          1.2202            1.2317         8.11
IVF-TQ-b2-nl316-np15-rf20 (query)                        921.39       543.53     1_464.92       0.4160          1.1325            1.1142         8.11
IVF-TQ-b2-nl316-np17-rf10 (query)                        921.39       291.49     1_212.88       0.2887          1.2206            1.2322         8.11
IVF-TQ-b2-nl316-np17-rf20 (query)                        921.39       552.39     1_473.78       0.4152          1.1328            1.1147         8.11
IVF-TQ-b2-nl316-np25-rf10 (query)                        921.39       319.46     1_240.85       0.2887          1.2206            1.2322         8.11
IVF-TQ-b2-nl316-np25-rf20 (query)                        921.39       601.51     1_522.89       0.4152          1.1328            1.1147         8.11
IVF-TQ-b2-nl316 (self)                                   921.39     1_088.67     2_010.06       0.4135          1.1619            1.1367         8.11
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_093.87       141.74     1_235.61       0.1023          1.7129            1.7532        14.06
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_093.87       162.42     1_256.29       0.1023          1.7129            1.7532        14.06
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_093.87       197.39     1_291.26       0.1023          1.7129            1.7532        14.06
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_093.87       361.89     1_455.76       0.3201          1.1873            1.1953        14.06
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_093.87       685.71     1_779.58       0.4481          1.1142            1.1029        14.06
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_093.87       375.39     1_469.27       0.3201          1.1874            1.1953        14.06
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_093.87       728.18     1_822.05       0.4481          1.1142            1.1029        14.06
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_093.87       445.56     1_539.44       0.3201          1.1873            1.1953        14.06
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_093.87       855.29     1_949.16       0.4481          1.1142            1.1029        14.06
IVF-TQ-b4-nl158 (self)                                 1_093.87     1_192.33     2_286.20       0.4462          1.1397            1.1286        14.06
IVF-TQ-b4-nl223-np11-rf0 (query)                         793.22       153.04       946.27       0.1023          1.7109            1.7520        14.27
IVF-TQ-b4-nl223-np14-rf0 (query)                         793.22       164.91       958.14       0.1023          1.7129            1.7532        14.27
IVF-TQ-b4-nl223-np21-rf0 (query)                         793.22       207.99     1_001.21       0.1023          1.7129            1.7532        14.27
IVF-TQ-b4-nl223-np11-rf10 (query)                        793.22       336.48     1_129.70       0.3205          1.1871            1.1949        14.27
IVF-TQ-b4-nl223-np11-rf20 (query)                        793.22       627.57     1_420.80       0.4486          1.1140            1.1028        14.27
IVF-TQ-b4-nl223-np14-rf10 (query)                        793.22       353.13     1_146.36       0.3202          1.1873            1.1953        14.27
IVF-TQ-b4-nl223-np14-rf20 (query)                        793.22       650.95     1_444.17       0.4481          1.1142            1.1029        14.27
IVF-TQ-b4-nl223-np21-rf10 (query)                        793.22       408.71     1_201.94       0.3201          1.1874            1.1953        14.27
IVF-TQ-b4-nl223-np21-rf20 (query)                        793.22       738.95     1_532.17       0.4481          1.1142            1.1029        14.27
IVF-TQ-b4-nl223 (self)                                   793.22     1_153.92     1_947.14       0.4462          1.1397            1.1286        14.27
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_022.77       159.88     1_182.65       0.1023          1.7112            1.7520        14.52
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_022.77       166.04     1_188.82       0.1023          1.7121            1.7528        14.52
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_022.77       200.03     1_222.81       0.1023          1.7129            1.7532        14.52
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_022.77       345.90     1_368.67       0.3207          1.1869            1.1949        14.52
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_022.77       601.89     1_624.67       0.4491          1.1138            1.1025        14.52
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_022.77       344.93     1_367.70       0.3202          1.1873            1.1953        14.52
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_022.77       608.35     1_631.12       0.4482          1.1142            1.1029        14.52
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_022.77       385.45     1_408.23       0.3201          1.1873            1.1953        14.52
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_022.77       664.86     1_687.63       0.4481          1.1142            1.1029        14.52
IVF-TQ-b4-nl316 (self)                                 1_022.77     1_130.54     2_153.31       0.4462          1.1397            1.1286        14.52
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
Exhaustive (query)                                        67.93     1_274.81     1_342.75       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         67.93     4_255.16     4_323.09       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              334.77       627.01       961.79       0.0844          1.6538            1.5906        13.97
ExhaustiveTQ-b2-rf5 (query)                              334.77       719.92     1_054.69       0.2172          1.2230            1.2549        13.97
ExhaustiveTQ-b2-rf10 (query)                             334.77       861.52     1_196.29       0.2885          1.1549            1.1706        13.97
ExhaustiveTQ-b2-rf20 (query)                             334.77     1_302.05     1_636.82       0.4019          1.0974            1.0847        13.97
ExhaustiveTQ-b2 (self)                                   334.77     4_215.25     4_550.02       0.4025          1.1135            1.0971        13.97
ExhaustiveTQ-b4-rf0 (query)                              463.76     1_095.99     1_559.76       0.1044          1.5026            1.5346        26.18
ExhaustiveTQ-b4-rf5 (query)                              463.76     1_210.31     1_674.07       0.2293          1.2110            1.2410        26.18
ExhaustiveTQ-b4-rf10 (query)                             463.76     1_365.21     1_828.98       0.2942          1.1499            1.1675        26.18
ExhaustiveTQ-b4-rf20 (query)                             463.76     1_764.12     2_227.89       0.4027          1.0975            1.0929        26.18
ExhaustiveTQ-b4 (self)                                   463.76     5_842.60     6_306.36       0.4036          1.1130            1.1087        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_851.27       189.38     2_040.65       0.0844          1.6538            1.5906        14.95
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_851.27       211.99     2_063.25       0.0844          1.6538            1.5906        14.95
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_851.27       224.73     2_075.99       0.0844          1.6538            1.5906        14.95
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_851.27       417.49     2_268.76       0.2885          1.1550            1.1706        14.95
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_851.27       765.17     2_616.43       0.4019          1.0974            1.0847        14.95
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_851.27       437.55     2_288.81       0.2885          1.1550            1.1706        14.95
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_851.27       791.79     2_643.06       0.4019          1.0974            1.0847        14.95
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_851.27       453.56     2_304.83       0.2885          1.1550            1.1706        14.95
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_851.27       824.33     2_675.60       0.4019          1.0974            1.0847        14.95
IVF-TQ-b2-nl158 (self)                                 1_851.27     1_436.79     3_288.05       0.4025          1.1135            1.0971        14.95
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_316.89       209.87     1_526.76       0.0844          1.6537            1.5905        15.23
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_316.89       218.38     1_535.27       0.0844          1.6539            1.5906        15.23
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_316.89       247.86     1_564.75       0.0844          1.6538            1.5906        15.23
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_316.89       418.11     1_735.00       0.2885          1.1550            1.1706        15.23
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_316.89       721.10     2_037.99       0.4019          1.0974            1.0847        15.23
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_316.89       432.98     1_749.87       0.2885          1.1550            1.1706        15.23
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_316.89       750.18     2_067.07       0.4019          1.0974            1.0847        15.23
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_316.89       463.29     1_780.18       0.2885          1.1550            1.1706        15.23
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_316.89       801.99     2_118.88       0.4019          1.0974            1.0847        15.23
IVF-TQ-b2-nl223 (self)                                 1_316.89     1_414.96     2_731.85       0.4025          1.1135            1.0971        15.23
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_705.38       222.62     1_928.00       0.0844          1.6538            1.5906        15.57
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_705.38       220.85     1_926.23       0.0844          1.6538            1.5906        15.57
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_705.38       249.06     1_954.44       0.0844          1.6538            1.5906        15.57
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_705.38       413.38     2_118.76       0.2885          1.1550            1.1706        15.57
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_705.38       699.66     2_405.04       0.4019          1.0974            1.0847        15.57
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_705.38       419.22     2_124.59       0.2885          1.1550            1.1706        15.57
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_705.38       743.95     2_449.33       0.4019          1.0974            1.0847        15.57
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_705.38       479.13     2_184.50       0.2885          1.1550            1.1706        15.57
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_705.38       778.09     2_483.47       0.4019          1.0974            1.0847        15.57
IVF-TQ-b2-nl316 (self)                                 1_705.38     1_470.57     3_175.95       0.4025          1.1135            1.0971        15.57
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_936.18       262.12     2_198.30       0.1044          1.5026            1.5346        27.44
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_936.18       291.23     2_227.41       0.1044          1.5026            1.5346        27.44
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_936.18       323.38     2_259.57       0.1044          1.5026            1.5346        27.44
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_936.18       502.99     2_439.18       0.2942          1.1499            1.1675        27.44
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_936.18       857.65     2_793.83       0.4027          1.0975            1.0929        27.44
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_936.18       534.18     2_470.37       0.2942          1.1499            1.1675        27.44
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_936.18       901.63     2_837.81       0.4027          1.0975            1.0929        27.44
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_936.18       563.65     2_499.83       0.2942          1.1499            1.1675        27.44
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_936.18       944.04     2_880.22       0.4027          1.0975            1.0929        27.44
IVF-TQ-b4-nl158 (self)                                 1_936.18     1_603.87     3_540.05       0.4036          1.1130            1.1087        27.44
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_428.38       303.37     1_731.75       0.1044          1.5026            1.5346        27.87
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_428.38       304.52     1_732.89       0.1044          1.5026            1.5346        27.87
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_428.38       356.24     1_784.62       0.1044          1.5026            1.5346        27.87
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_428.38       514.21     1_942.59       0.2942          1.1499            1.1676        27.87
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_428.38       828.46     2_256.84       0.4027          1.0975            1.0929        27.87
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_428.38       531.33     1_959.71       0.2942          1.1499            1.1676        27.87
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_428.38       857.95     2_286.33       0.4027          1.0975            1.0929        27.87
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_428.38       587.10     2_015.48       0.2942          1.1499            1.1676        27.87
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_428.38       933.47     2_361.84       0.4027          1.0975            1.0929        27.87
IVF-TQ-b4-nl223 (self)                                 1_428.38     1_643.93     3_072.31       0.4036          1.1130            1.1087        27.87
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_810.24       304.95     2_115.19       0.1044          1.5026            1.5346        28.38
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_810.24       308.64     2_118.88       0.1044          1.5026            1.5346        28.38
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_810.24       354.08     2_164.31       0.1044          1.5026            1.5346        28.38
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_810.24       509.07     2_319.31       0.2942          1.1499            1.1675        28.38
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_810.24       806.99     2_617.23       0.4027          1.0975            1.0929        28.38
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_810.24       523.65     2_333.89       0.2942          1.1499            1.1675        28.38
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_810.24       819.81     2_630.05       0.4027          1.0975            1.0929        28.38
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_810.24       573.16     2_383.40       0.2942          1.1499            1.1675        28.38
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_810.24       886.53     2_696.77       0.4027          1.0975            1.0929        28.38
IVF-TQ-b4-nl316 (self)                                 1_810.24     1_684.37     3_494.61       0.4036          1.1130            1.1087        28.38
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
Exhaustive (query)                                       101.04     1_930.08     2_031.12       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.04     6_596.08     6_697.12       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              615.82       973.67     1_589.50       0.0841          1.5106            1.4225        21.33
ExhaustiveTQ-b2-rf5 (query)                              615.82     1_065.40     1_681.22       0.2143          1.1739            1.2055        21.33
ExhaustiveTQ-b2-rf10 (query)                             615.82     1_208.87     1_824.69       0.2768          1.1266            1.1511        21.33
ExhaustiveTQ-b2-rf20 (query)                             615.82     1_632.48     2_248.30       0.3768          1.0842            1.0723        21.33
ExhaustiveTQ-b2 (self)                                   615.82     5_402.40     6_018.22       0.3768          1.0935            1.0802        21.33
ExhaustiveTQ-b4-rf0 (query)                              769.58     1_805.52     2_575.10       0.0986          1.4230            1.4108        39.64
ExhaustiveTQ-b4-rf5 (query)                              769.58     1_928.31     2_697.89       0.2167          1.1745            1.2046        39.64
ExhaustiveTQ-b4-rf10 (query)                             769.58     2_030.60     2_800.17       0.2692          1.1310            1.1556        39.64
ExhaustiveTQ-b4-rf20 (query)                             769.58     2_438.05     3_207.63       0.3605          1.0922            1.1070        39.64
ExhaustiveTQ-b4 (self)                                   769.58     8_010.94     8_780.51       0.3608          1.1023            1.1181        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        2_811.50       290.62     3_102.12       0.0841          1.5106            1.4225        22.62
IVF-TQ-b2-nl158-np12-rf0 (query)                       2_811.50       314.08     3_125.58       0.0841          1.5106            1.4225        22.62
IVF-TQ-b2-nl158-np17-rf0 (query)                       2_811.50       339.39     3_150.89       0.0841          1.5106            1.4225        22.62
IVF-TQ-b2-nl158-np7-rf10 (query)                       2_811.50       539.33     3_350.83       0.2768          1.1266            1.1511        22.62
IVF-TQ-b2-nl158-np7-rf20 (query)                       2_811.50       931.39     3_742.89       0.3768          1.0842            1.0723        22.62
IVF-TQ-b2-nl158-np12-rf10 (query)                      2_811.50       550.07     3_361.58       0.2768          1.1266            1.1511        22.62
IVF-TQ-b2-nl158-np12-rf20 (query)                      2_811.50       931.39     3_742.89       0.3768          1.0842            1.0723        22.62
IVF-TQ-b2-nl158-np17-rf10 (query)                      2_811.50       576.57     3_388.07       0.2768          1.1266            1.1511        22.62
IVF-TQ-b2-nl158-np17-rf20 (query)                      2_811.50       969.31     3_780.82       0.3768          1.0842            1.0723        22.62
IVF-TQ-b2-nl158 (self)                                 2_811.50     1_882.97     4_694.47       0.3768          1.0935            1.0802        22.62
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_892.21       326.95     2_219.16       0.0841          1.5106            1.4225        22.97
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_892.21       323.61     2_215.82       0.0841          1.5106            1.4225        22.97
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_892.21       352.93     2_245.14       0.0841          1.5106            1.4225        22.97
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_892.21       549.50     2_441.70       0.2768          1.1266            1.1511        22.97
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_892.21       882.58     2_774.79       0.3768          1.0842            1.0723        22.97
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_892.21       561.99     2_454.20       0.2768          1.1266            1.1511        22.97
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_892.21       904.43     2_796.63       0.3768          1.0842            1.0723        22.97
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_892.21       605.16     2_497.37       0.2768          1.1266            1.1511        22.97
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_892.21     1_039.16     2_931.37       0.3768          1.0842            1.0723        22.97
IVF-TQ-b2-nl223 (self)                                 1_892.21     1_899.39     3_791.60       0.3768          1.0935            1.0802        22.97
IVF-TQ-b2-nl316-np15-rf0 (query)                       2_474.98       334.21     2_809.19       0.0841          1.5106            1.4225        23.53
IVF-TQ-b2-nl316-np17-rf0 (query)                       2_474.98       335.07     2_810.05       0.0841          1.5106            1.4225        23.53
IVF-TQ-b2-nl316-np25-rf0 (query)                       2_474.98       378.05     2_853.03       0.0841          1.5106            1.4225        23.53
IVF-TQ-b2-nl316-np15-rf10 (query)                      2_474.98       554.78     3_029.76       0.2768          1.1266            1.1511        23.53
IVF-TQ-b2-nl316-np15-rf20 (query)                      2_474.98       876.49     3_351.47       0.3768          1.0842            1.0723        23.53
IVF-TQ-b2-nl316-np17-rf10 (query)                      2_474.98       563.68     3_038.66       0.2768          1.1266            1.1511        23.53
IVF-TQ-b2-nl316-np17-rf20 (query)                      2_474.98       885.04     3_360.02       0.3768          1.0842            1.0723        23.53
IVF-TQ-b2-nl316-np25-rf10 (query)                      2_474.98       600.59     3_075.57       0.2768          1.1266            1.1511        23.53
IVF-TQ-b2-nl316-np25-rf20 (query)                      2_474.98       947.77     3_422.75       0.3768          1.0842            1.0723        23.53
IVF-TQ-b2-nl316 (self)                                 2_474.98     1_972.28     4_447.26       0.3768          1.0935            1.0802        23.53
IVF-TQ-b4-nl158-np7-rf0 (query)                        2_852.12       409.70     3_261.82       0.0986          1.4230            1.4108        41.39
IVF-TQ-b4-nl158-np12-rf0 (query)                       2_852.12       450.48     3_302.60       0.0986          1.4230            1.4108        41.39
IVF-TQ-b4-nl158-np17-rf0 (query)                       2_852.12       517.13     3_369.25       0.0986          1.4230            1.4108        41.39
IVF-TQ-b4-nl158-np7-rf10 (query)                       2_852.12       661.23     3_513.35       0.2692          1.1310            1.1556        41.39
IVF-TQ-b4-nl158-np7-rf20 (query)                       2_852.12     1_042.96     3_895.08       0.3605          1.0922            1.1070        41.39
IVF-TQ-b4-nl158-np12-rf10 (query)                      2_852.12       700.87     3_553.00       0.2692          1.1310            1.1556        41.39
IVF-TQ-b4-nl158-np12-rf20 (query)                      2_852.12     1_108.08     3_960.21       0.3605          1.0922            1.1070        41.39
IVF-TQ-b4-nl158-np17-rf10 (query)                      2_852.12       745.67     3_597.80       0.2692          1.1310            1.1556        41.39
IVF-TQ-b4-nl158-np17-rf20 (query)                      2_852.12     1_132.72     3_984.84       0.3605          1.0922            1.1070        41.39
IVF-TQ-b4-nl158 (self)                                 2_852.12     2_208.06     5_060.18       0.3608          1.1023            1.1181        41.39
IVF-TQ-b4-nl223-np11-rf0 (query)                       2_028.80       463.04     2_491.84       0.0986          1.4230            1.4108        41.89
IVF-TQ-b4-nl223-np14-rf0 (query)                       2_028.80       472.22     2_501.02       0.0986          1.4230            1.4108        41.89
IVF-TQ-b4-nl223-np21-rf0 (query)                       2_028.80       528.30     2_557.10       0.0986          1.4230            1.4108        41.89
IVF-TQ-b4-nl223-np11-rf10 (query)                      2_028.80       688.24     2_717.04       0.2692          1.1310            1.1556        41.89
IVF-TQ-b4-nl223-np11-rf20 (query)                      2_028.80     1_027.17     3_055.97       0.3605          1.0922            1.1070        41.89
IVF-TQ-b4-nl223-np14-rf10 (query)                      2_028.80       731.18     2_759.99       0.2692          1.1310            1.1556        41.89
IVF-TQ-b4-nl223-np14-rf20 (query)                      2_028.80     1_063.80     3_092.60       0.3605          1.0922            1.1070        41.89
IVF-TQ-b4-nl223-np21-rf10 (query)                      2_028.80       779.79     2_808.60       0.2692          1.1310            1.1556        41.89
IVF-TQ-b4-nl223-np21-rf20 (query)                      2_028.80     1_158.08     3_186.88       0.3605          1.0922            1.1070        41.89
IVF-TQ-b4-nl223 (self)                                 2_028.80     2_394.92     4_423.72       0.3608          1.1023            1.1181        41.89
IVF-TQ-b4-nl316-np15-rf0 (query)                       2_687.51       485.91     3_173.42       0.0986          1.4230            1.4108        42.73
IVF-TQ-b4-nl316-np17-rf0 (query)                       2_687.51       490.49     3_178.00       0.0986          1.4230            1.4108        42.73
IVF-TQ-b4-nl316-np25-rf0 (query)                       2_687.51       549.95     3_237.46       0.0986          1.4230            1.4108        42.73
IVF-TQ-b4-nl316-np15-rf10 (query)                      2_687.51       706.20     3_393.71       0.2692          1.1310            1.1556        42.73
IVF-TQ-b4-nl316-np15-rf20 (query)                      2_687.51     1_016.57     3_704.08       0.3605          1.0922            1.1070        42.73
IVF-TQ-b4-nl316-np17-rf10 (query)                      2_687.51       729.66     3_417.16       0.2692          1.1310            1.1556        42.73
IVF-TQ-b4-nl316-np17-rf20 (query)                      2_687.51     1_048.91     3_736.42       0.3605          1.0922            1.1070        42.73
IVF-TQ-b4-nl316-np25-rf10 (query)                      2_687.51       791.93     3_479.43       0.2692          1.1310            1.1556        42.73
IVF-TQ-b4-nl316-np25-rf20 (query)                      2_687.51     1_143.11     3_830.62       0.3605          1.0922            1.1070        42.73
IVF-TQ-b4-nl316 (self)                                 2_687.51     2_326.31     5_013.82       0.3608          1.1023            1.1181        42.73
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
Exhaustive (query)                                        33.02       755.21       788.23       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.02     2_486.35     2_519.37       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              144.76       373.92       518.68       0.7918          1.0898            1.0632         7.12
ExhaustiveTQ-b2-rf5 (query)                              144.76       455.63       600.40       0.9995          1.0000            1.0000         7.12
ExhaustiveTQ-b2-rf10 (query)                             144.76       588.90       733.67       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b2-rf20 (query)                             144.76     1_028.56     1_173.33       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b2 (self)                                   144.76     3_258.50     3_403.26       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b4-rf0 (query)                              233.65       604.84       838.50       0.8727          1.0322            1.0183        13.22
ExhaustiveTQ-b4-rf5 (query)                              233.65       686.78       920.44       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4-rf10 (query)                             233.65       817.93     1_051.58       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4-rf20 (query)                             233.65     1_201.83     1_435.48       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4 (self)                                   233.65     3_986.01     4_219.66       1.0000          1.0000            1.0000        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_107.26       135.38     1_242.64       0.7916          1.0897            1.0635         7.78
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_107.26       186.66     1_293.93       0.7918          1.0898            1.0632         7.78
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_107.26       218.65     1_325.92       0.7918          1.0898            1.0632         7.78
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_107.26       342.24     1_449.51       0.9981          1.0004            1.0000         7.78
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_107.26       668.59     1_775.85       0.9982          1.0004            1.0000         7.78
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_107.26       402.22     1_509.48       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_107.26       732.62     1_839.88       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_107.26       468.73     1_575.99       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_107.26       807.82     1_915.08       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158 (self)                                 1_107.26     1_213.26     2_320.52       0.9999          1.0000            1.0000         7.78
IVF-TQ-b2-nl223-np11-rf0 (query)                         630.06       130.58       760.64       0.7919          1.0897            1.0632         7.93
IVF-TQ-b2-nl223-np14-rf0 (query)                         630.06       150.59       780.65       0.7918          1.0897            1.0632         7.93
IVF-TQ-b2-nl223-np21-rf0 (query)                         630.06       191.32       821.38       0.7918          1.0898            1.0632         7.93
IVF-TQ-b2-nl223-np11-rf10 (query)                        630.06       329.57       959.62       0.9995          1.0001            1.0000         7.93
IVF-TQ-b2-nl223-np11-rf20 (query)                        630.06       611.45     1_241.50       0.9995          1.0001            1.0000         7.93
IVF-TQ-b2-nl223-np14-rf10 (query)                        630.06       357.55       987.61       0.9999          1.0000            1.0000         7.93
IVF-TQ-b2-nl223-np14-rf20 (query)                        630.06       649.94     1_279.99       0.9999          1.0000            1.0000         7.93
IVF-TQ-b2-nl223-np21-rf10 (query)                        630.06       404.86     1_034.91       1.0000          1.0000            1.0000         7.93
IVF-TQ-b2-nl223-np21-rf20 (query)                        630.06       742.19     1_372.25       1.0000          1.0000            1.0000         7.93
IVF-TQ-b2-nl223 (self)                                   630.06     1_082.24     1_712.30       1.0000          1.0000            1.0000         7.93
IVF-TQ-b2-nl316-np15-rf0 (query)                         805.41       134.20       939.61       0.7918          1.0897            1.0632         8.12
IVF-TQ-b2-nl316-np17-rf0 (query)                         805.41       144.37       949.78       0.7918          1.0898            1.0632         8.12
IVF-TQ-b2-nl316-np25-rf0 (query)                         805.41       177.04       982.44       0.7918          1.0898            1.0632         8.12
IVF-TQ-b2-nl316-np15-rf10 (query)                        805.41       317.33     1_122.73       0.9997          1.0000            1.0000         8.12
IVF-TQ-b2-nl316-np15-rf20 (query)                        805.41       586.52     1_391.92       0.9997          1.0000            1.0000         8.12
IVF-TQ-b2-nl316-np17-rf10 (query)                        805.41       328.21     1_133.62       0.9999          1.0000            1.0000         8.12
IVF-TQ-b2-nl316-np17-rf20 (query)                        805.41       607.49     1_412.90       0.9999          1.0000            1.0000         8.12
IVF-TQ-b2-nl316-np25-rf10 (query)                        805.41       374.60     1_180.00       1.0000          1.0000            1.0000         8.12
IVF-TQ-b2-nl316-np25-rf20 (query)                        805.41       689.36     1_494.77       1.0000          1.0000            1.0000         8.12
IVF-TQ-b2-nl316 (self)                                   805.41     1_039.58     1_844.99       1.0000          1.0000            1.0000         8.12
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_151.19       184.53     1_335.73       0.8721          1.0325            1.0187        14.02
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_151.19       258.89     1_410.08       0.8727          1.0322            1.0183        14.02
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_151.19       321.65     1_472.84       0.8727          1.0322            1.0183        14.02
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_151.19       397.81     1_549.00       0.9981          1.0004            1.0000        14.02
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_151.19       693.27     1_844.46       0.9982          1.0004            1.0000        14.02
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_151.19       490.20     1_641.39       1.0000          1.0000            1.0000        14.02
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_151.19       882.51     2_033.70       1.0000          1.0000            1.0000        14.02
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_151.19       564.39     1_715.58       1.0000          1.0000            1.0000        14.02
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_151.19       916.70     2_067.90       1.0000          1.0000            1.0000        14.02
IVF-TQ-b4-nl158 (self)                                 1_151.19     1_281.01     2_432.20       0.9999          1.0000            1.0000        14.02
IVF-TQ-b4-nl223-np11-rf0 (query)                         713.98       182.37       896.34       0.8726          1.0323            1.0184        14.23
IVF-TQ-b4-nl223-np14-rf0 (query)                         713.98       213.85       927.83       0.8727          1.0322            1.0183        14.23
IVF-TQ-b4-nl223-np21-rf0 (query)                         713.98       276.54       990.51       0.8727          1.0322            1.0183        14.23
IVF-TQ-b4-nl223-np11-rf10 (query)                        713.98       377.65     1_091.63       0.9995          1.0001            1.0000        14.23
IVF-TQ-b4-nl223-np11-rf20 (query)                        713.98       658.74     1_372.72       0.9995          1.0001            1.0000        14.23
IVF-TQ-b4-nl223-np14-rf10 (query)                        713.98       418.40     1_132.37       0.9999          1.0000            1.0000        14.23
IVF-TQ-b4-nl223-np14-rf20 (query)                        713.98       714.95     1_428.93       0.9999          1.0000            1.0000        14.23
IVF-TQ-b4-nl223-np21-rf10 (query)                        713.98       499.54     1_213.52       1.0000          1.0000            1.0000        14.23
IVF-TQ-b4-nl223-np21-rf20 (query)                        713.98       841.46     1_555.43       1.0000          1.0000            1.0000        14.23
IVF-TQ-b4-nl223 (self)                                   713.98     1_130.46     1_844.44       1.0000          1.0000            1.0000        14.23
IVF-TQ-b4-nl316-np15-rf0 (query)                         893.15       185.34     1_078.50       0.8727          1.0322            1.0184        14.54
IVF-TQ-b4-nl316-np17-rf0 (query)                         893.15       200.62     1_093.78       0.8727          1.0322            1.0183        14.54
IVF-TQ-b4-nl316-np25-rf0 (query)                         893.15       254.37     1_147.52       0.8727          1.0322            1.0183        14.54
IVF-TQ-b4-nl316-np15-rf10 (query)                        893.15       370.17     1_263.32       0.9997          1.0000            1.0000        14.54
IVF-TQ-b4-nl316-np15-rf20 (query)                        893.15       640.68     1_533.83       0.9997          1.0000            1.0000        14.54
IVF-TQ-b4-nl316-np17-rf10 (query)                        893.15       390.49     1_283.64       0.9999          1.0000            1.0000        14.54
IVF-TQ-b4-nl316-np17-rf20 (query)                        893.15       670.64     1_563.79       0.9999          1.0000            1.0000        14.54
IVF-TQ-b4-nl316-np25-rf10 (query)                        893.15       455.56     1_348.72       1.0000          1.0000            1.0000        14.54
IVF-TQ-b4-nl316-np25-rf20 (query)                        893.15       767.94     1_661.10       1.0000          1.0000            1.0000        14.54
IVF-TQ-b4-nl316 (self)                                   893.15     1_088.95     1_982.10       1.0000          1.0000            1.0000        14.54
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
Exhaustive (query)                                        69.59     1_281.25     1_350.84       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.59     4_406.84     4_476.43       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              340.86       626.73       967.59       0.8424          1.0447            1.0331        13.97
ExhaustiveTQ-b2-rf5 (query)                              340.86       722.40     1_063.26       0.9999          1.0000            1.0000        13.97
ExhaustiveTQ-b2-rf10 (query)                             340.86       870.04     1_210.90       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2-rf20 (query)                             340.86     1_295.84     1_636.70       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2 (self)                                   340.86     4_265.50     4_606.36       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b4-rf0 (query)                              453.32     1_105.07     1_558.39       0.8985          1.0191            1.0110        26.18
ExhaustiveTQ-b4-rf5 (query)                              453.32     1_210.28     1_663.60       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4-rf10 (query)                             453.32     1_362.78     1_816.10       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4-rf20 (query)                             453.32     1_782.98     2_236.29       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4 (self)                                   453.32     5_867.82     6_321.14       1.0000          1.0000            1.0000        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        2_087.85       231.06     2_318.90       0.8420          1.0449            1.0333        14.96
IVF-TQ-b2-nl158-np12-rf0 (query)                       2_087.85       300.56     2_388.41       0.8424          1.0447            1.0331        14.96
IVF-TQ-b2-nl158-np17-rf0 (query)                       2_087.85       358.95     2_446.80       0.8424          1.0447            1.0331        14.96
IVF-TQ-b2-nl158-np7-rf10 (query)                       2_087.85       464.88     2_552.73       0.9986          1.0003            1.0000        14.96
IVF-TQ-b2-nl158-np7-rf20 (query)                       2_087.85       782.74     2_870.59       0.9986          1.0003            1.0000        14.96
IVF-TQ-b2-nl158-np12-rf10 (query)                      2_087.85       551.27     2_639.12       0.9999          1.0000            1.0000        14.96
IVF-TQ-b2-nl158-np12-rf20 (query)                      2_087.85       909.91     2_997.76       0.9999          1.0000            1.0000        14.96
IVF-TQ-b2-nl158-np17-rf10 (query)                      2_087.85       616.55     2_704.40       0.9999          1.0000            1.0000        14.96
IVF-TQ-b2-nl158-np17-rf20 (query)                      2_087.85       985.13     3_072.98       0.9999          1.0000            1.0000        14.96
IVF-TQ-b2-nl158 (self)                                 2_087.85     1_539.69     3_627.54       1.0000          1.0000            1.0000        14.96
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_126.86       239.72     1_366.58       0.8423          1.0447            1.0331        15.25
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_126.86       268.66     1_395.53       0.8424          1.0447            1.0331        15.25
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_126.86       325.50     1_452.36       0.8424          1.0447            1.0331        15.25
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_126.86       461.55     1_588.42       0.9997          1.0000            1.0000        15.25
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_126.86       748.66     1_875.52       0.9997          1.0000            1.0000        15.25
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_126.86       486.00     1_612.86       0.9999          1.0000            1.0000        15.25
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_126.86       795.92     1_922.78       0.9999          1.0000            1.0000        15.25
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_126.86       566.97     1_693.83       1.0000          1.0000            1.0000        15.25
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_126.86       890.20     2_017.06       1.0000          1.0000            1.0000        15.25
IVF-TQ-b2-nl223 (self)                                 1_126.86     1_443.77     2_570.64       1.0000          1.0000            1.0000        15.25
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_363.96       248.03     1_611.99       0.8424          1.0447            1.0331        15.56
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_363.96       255.43     1_619.39       0.8424          1.0447            1.0331        15.56
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_363.96       307.68     1_671.64       0.8424          1.0447            1.0331        15.56
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_363.96       453.73     1_817.68       0.9999          1.0000            1.0000        15.56
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_363.96       754.25     2_118.21       0.9999          1.0000            1.0000        15.56
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_363.96       476.68     1_840.64       0.9999          1.0000            1.0000        15.56
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_363.96       778.91     2_142.87       0.9999          1.0000            1.0000        15.56
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_363.96       540.25     1_904.20       1.0000          1.0000            1.0000        15.56
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_363.96       864.04     2_227.99       1.0000          1.0000            1.0000        15.56
IVF-TQ-b2-nl316 (self)                                 1_363.96     1_411.32     2_775.27       1.0000          1.0000            1.0000        15.56
IVF-TQ-b4-nl158-np7-rf0 (query)                        2_173.13       336.51     2_509.63       0.8977          1.0194            1.0113        27.46
IVF-TQ-b4-nl158-np12-rf0 (query)                       2_173.13       466.97     2_640.10       0.8985          1.0191            1.0110        27.46
IVF-TQ-b4-nl158-np17-rf0 (query)                       2_173.13       572.06     2_745.19       0.8985          1.0191            1.0110        27.46
IVF-TQ-b4-nl158-np7-rf10 (query)                       2_173.13       572.24     2_745.37       0.9986          1.0003            1.0000        27.46
IVF-TQ-b4-nl158-np7-rf20 (query)                       2_173.13       906.13     3_079.26       0.9986          1.0003            1.0000        27.46
IVF-TQ-b4-nl158-np12-rf10 (query)                      2_173.13       712.63     2_885.76       0.9999          1.0000            1.0000        27.46
IVF-TQ-b4-nl158-np12-rf20 (query)                      2_173.13     1_068.67     3_241.80       0.9999          1.0000            1.0000        27.46
IVF-TQ-b4-nl158-np17-rf10 (query)                      2_173.13       825.46     2_998.59       0.9999          1.0000            1.0000        27.46
IVF-TQ-b4-nl158-np17-rf20 (query)                      2_173.13     1_207.77     3_380.90       0.9999          1.0000            1.0000        27.46
IVF-TQ-b4-nl158 (self)                                 2_173.13     1_832.41     4_005.53       1.0000          1.0000            1.0000        27.46
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_244.58       356.57     1_601.16       0.8984          1.0191            1.0111        27.91
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_244.58       398.55     1_643.13       0.8985          1.0191            1.0110        27.91
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_244.58       510.18     1_754.76       0.8985          1.0191            1.0110        27.91
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_244.58       563.43     1_808.02       0.9997          1.0000            1.0000        27.91
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_244.58       860.23     2_104.81       0.9997          1.0000            1.0000        27.91
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_244.58       621.00     1_865.58       0.9999          1.0000            1.0000        27.91
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_244.58       929.70     2_174.28       0.9999          1.0000            1.0000        27.91
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_244.58       738.20     1_982.79       1.0000          1.0000            1.0000        27.91
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_244.58     1_093.17     2_337.75       1.0000          1.0000            1.0000        27.91
IVF-TQ-b4-nl223 (self)                                 1_244.58     1_730.07     2_974.65       1.0000          1.0000            1.0000        27.91
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_490.71       348.44     1_839.15       0.8985          1.0191            1.0110        28.36
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_490.71       374.66     1_865.37       0.8985          1.0191            1.0110        28.36
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_490.71       471.36     1_962.07       0.8985          1.0191            1.0110        28.36
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_490.71       560.00     2_050.71       0.9999          1.0000            1.0000        28.36
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_490.71       883.83     2_374.54       0.9999          1.0000            1.0000        28.36
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_490.71       594.53     2_085.25       0.9999          1.0000            1.0000        28.36
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_490.71       901.14     2_391.85       0.9999          1.0000            1.0000        28.36
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_490.71       691.32     2_182.03       1.0000          1.0000            1.0000        28.36
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_490.71     1_079.39     2_570.11       1.0000          1.0000            1.0000        28.36
IVF-TQ-b4-nl316 (self)                                 1_490.71     1_703.27     3_193.98       1.0000          1.0000            1.0000        28.36
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
Exhaustive (query)                                       100.56     1_960.79     2_061.36       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.56     6_645.50     6_746.06       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              606.97       979.24     1_586.21       0.8736          1.0271            1.0199        21.33
ExhaustiveTQ-b2-rf5 (query)                              606.97     1_074.18     1_681.15       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2-rf10 (query)                             606.97     1_222.04     1_829.01       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2-rf20 (query)                             606.97     1_661.29     2_268.26       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2 (self)                                   606.97     5_407.13     6_014.10       0.9999          1.0000            1.0000        21.33
ExhaustiveTQ-b4-rf0 (query)                              747.19     1_780.40     2_527.58       0.9097          1.0146            1.0083        39.64
ExhaustiveTQ-b4-rf5 (query)                              747.19     1_879.64     2_626.83       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4-rf10 (query)                             747.19     2_012.69     2_759.88       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4-rf20 (query)                             747.19     2_427.87     3_175.06       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4 (self)                                   747.19     8_025.86     8_773.05       0.9999          1.0000            1.0000        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        3_125.07       353.56     3_478.63       0.8735          1.0272            1.0201        22.61
IVF-TQ-b2-nl158-np12-rf0 (query)                       3_125.07       449.75     3_574.82       0.8736          1.0271            1.0199        22.61
IVF-TQ-b2-nl158-np17-rf0 (query)                       3_125.07       529.53     3_654.60       0.8736          1.0271            1.0199        22.61
IVF-TQ-b2-nl158-np7-rf10 (query)                       3_125.07       612.80     3_737.88       0.9995          1.0001            1.0000        22.61
IVF-TQ-b2-nl158-np7-rf20 (query)                       3_125.07       947.48     4_072.55       0.9995          1.0001            1.0000        22.61
IVF-TQ-b2-nl158-np12-rf10 (query)                      3_125.07       725.62     3_850.69       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158-np12-rf20 (query)                      3_125.07     1_086.77     4_211.84       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158-np17-rf10 (query)                      3_125.07       812.97     3_938.04       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158-np17-rf20 (query)                      3_125.07     1_191.19     4_316.26       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158 (self)                                 3_125.07     2_076.31     5_201.38       0.9999          1.0000            1.0000        22.61
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_703.44       352.05     2_055.48       0.8736          1.0271            1.0200        23.01
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_703.44       395.57     2_099.00       0.8736          1.0271            1.0199        23.01
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_703.44       480.17     2_183.61       0.8736          1.0271            1.0199        23.01
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_703.44       595.54     2_298.98       0.9998          1.0000            1.0000        23.01
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_703.44       924.88     2_628.31       0.9998          1.0000            1.0000        23.01
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_703.44       640.57     2_344.01       0.9999          1.0000            1.0000        23.01
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_703.44       988.45     2_691.89       0.9999          1.0000            1.0000        23.01
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_703.44       738.60     2_442.04       1.0000          1.0000            1.0000        23.01
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_703.44     1_101.40     2_804.83       1.0000          1.0000            1.0000        23.01
IVF-TQ-b2-nl223 (self)                                 1_703.44     2_034.28     3_737.72       0.9999          1.0000            1.0000        23.01
IVF-TQ-b2-nl316-np15-rf0 (query)                       2_031.98       361.60     2_393.58       0.8736          1.0271            1.0200        23.53
IVF-TQ-b2-nl316-np17-rf0 (query)                       2_031.98       383.08     2_415.06       0.8736          1.0271            1.0200        23.53
IVF-TQ-b2-nl316-np25-rf0 (query)                       2_031.98       453.40     2_485.38       0.8736          1.0271            1.0199        23.53
IVF-TQ-b2-nl316-np15-rf10 (query)                      2_031.98       598.59     2_630.57       0.9999          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np15-rf20 (query)                      2_031.98       929.36     2_961.34       0.9999          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np17-rf10 (query)                      2_031.98       619.41     2_651.40       0.9999          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np17-rf20 (query)                      2_031.98       960.43     2_992.42       0.9999          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np25-rf10 (query)                      2_031.98       702.83     2_734.81       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np25-rf20 (query)                      2_031.98     1_065.93     3_097.91       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316 (self)                                 2_031.98     1_942.56     3_974.54       0.9999          1.0000            1.0000        23.53
IVF-TQ-b4-nl158-np7-rf0 (query)                        3_191.39       535.43     3_726.82       0.9094          1.0147            1.0084        41.37
IVF-TQ-b4-nl158-np12-rf0 (query)                       3_191.39       717.00     3_908.40       0.9097          1.0146            1.0083        41.37
IVF-TQ-b4-nl158-np17-rf0 (query)                       3_191.39       863.96     4_055.35       0.9097          1.0146            1.0083        41.37
IVF-TQ-b4-nl158-np7-rf10 (query)                       3_191.39       785.35     3_976.75       0.9995          1.0001            1.0000        41.37
IVF-TQ-b4-nl158-np7-rf20 (query)                       3_191.39     1_119.90     4_311.30       0.9995          1.0001            1.0000        41.37
IVF-TQ-b4-nl158-np12-rf10 (query)                      3_191.39       978.53     4_169.93       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158-np12-rf20 (query)                      3_191.39     1_348.11     4_539.50       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158-np17-rf10 (query)                      3_191.39     1_137.10     4_328.50       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158-np17-rf20 (query)                      3_191.39     1_526.46     4_717.86       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158 (self)                                 3_191.39     2_647.71     5_839.10       0.9999          1.0000            1.0000        41.37
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_880.74       544.41     2_425.15       0.9096          1.0146            1.0084        41.97
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_880.74       630.10     2_510.84       0.9097          1.0146            1.0083        41.97
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_880.74       793.26     2_673.99       0.9097          1.0146            1.0083        41.97
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_880.74       768.79     2_649.53       0.9998          1.0000            1.0000        41.97
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_880.74     1_102.79     2_983.53       0.9998          1.0000            1.0000        41.97
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_880.74       850.82     2_731.56       0.9999          1.0000            1.0000        41.97
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_880.74     1_190.18     3_070.92       0.9999          1.0000            1.0000        41.97
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_880.74     1_025.85     2_906.59       1.0000          1.0000            1.0000        41.97
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_880.74     1_379.79     3_260.53       1.0000          1.0000            1.0000        41.97
IVF-TQ-b4-nl223 (self)                                 1_880.74     2_491.73     4_372.46       0.9999          1.0000            1.0000        41.97
IVF-TQ-b4-nl316-np15-rf0 (query)                       2_185.87       580.05     2_765.91       0.9097          1.0146            1.0083        42.73
IVF-TQ-b4-nl316-np17-rf0 (query)                       2_185.87       590.73     2_776.60       0.9097          1.0146            1.0083        42.73
IVF-TQ-b4-nl316-np25-rf0 (query)                       2_185.87       736.83     2_922.70       0.9097          1.0146            1.0083        42.73
IVF-TQ-b4-nl316-np15-rf10 (query)                      2_185.87       772.49     2_958.35       0.9999          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np15-rf20 (query)                      2_185.87     1_111.75     3_297.62       0.9999          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np17-rf10 (query)                      2_185.87       812.15     2_998.01       0.9999          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np17-rf20 (query)                      2_185.87     1_149.58     3_335.44       0.9999          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np25-rf10 (query)                      2_185.87       967.84     3_153.70       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np25-rf20 (query)                      2_185.87     1_384.46     3_570.32       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316 (self)                                 2_185.87     2_419.01     4_604.88       0.9999          1.0000            1.0000        42.73
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
