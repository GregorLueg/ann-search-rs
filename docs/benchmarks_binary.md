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
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 256D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.31       663.62       695.93       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.31     2_186.66     2_218.97       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                 96.16       236.55       332.70       0.1199          1.4617            1.4199         1.78
ExhaustiveBinary-256-random-rf10 (query)                  96.16       338.25       434.41       0.3411          1.0941            1.0814         1.78
ExhaustiveBinary-256-random-rf20 (query)                  96.16       447.38       543.53       0.4467          1.0571            1.0475         1.78
ExhaustiveBinary-256-random (self)                        96.16     1_094.97     1_191.13       0.3454          1.0895            1.0798         1.78
ExhaustiveBinary-256-pca_no_rr (query)                   133.36       238.74       372.10       0.1153          1.4748            1.4212         1.78
ExhaustiveBinary-256-pca-rf10 (query)                    133.36       336.45       469.82       0.3323          1.1029            1.0834         1.78
ExhaustiveBinary-256-pca-rf20 (query)                    133.36       440.22       573.58       0.4387          1.0631            1.0485         1.78
ExhaustiveBinary-256-pca (self)                          133.36     1_087.29     1_220.65       0.3391          1.0957            1.0813         1.78
ExhaustiveBinary-512-random_no_rr (query)                110.94       351.16       462.10       0.1588          1.3547            1.3300         3.55
ExhaustiveBinary-512-random-rf10 (query)                 110.94       460.61       571.55       0.3786          1.0692            1.0677         3.55
ExhaustiveBinary-512-random-rf20 (query)                 110.94       568.29       679.24       0.4874          1.0424            1.0395         3.55
ExhaustiveBinary-512-random (self)                       110.94     1_495.49     1_606.43       0.3805          1.0675            1.0675         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   142.26       350.53       492.79       0.1564          1.3535            1.3265         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    142.26       458.85       601.11       0.3789          1.0710            1.0663         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    142.26       567.63       709.89       0.4903          1.0433            1.0387         3.55
ExhaustiveBinary-512-pca (self)                          142.26     1_491.65     1_633.91       0.3823          1.0678            1.0665         3.55
ExhaustiveBinary-1024-random_no_rr (query)               141.34       487.24       628.57       0.1929          1.2764            1.2696         7.10
ExhaustiveBinary-1024-random-rf10 (query)                141.34       605.96       747.29       0.4214          1.0550            1.0552         7.10
ExhaustiveBinary-1024-random-rf20 (query)                141.34       719.91       861.25       0.5434          1.0327            1.0308         7.10
ExhaustiveBinary-1024-random (self)                      141.34     1_972.39     2_113.73       0.4232          1.0547            1.0552         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  182.79       490.08       672.87       0.1921          1.2733            1.2652         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   182.79       602.47       785.26       0.4226          1.0546            1.0544         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   182.79       719.77       902.57       0.5443          1.0326            1.0305         7.10
ExhaustiveBinary-1024-pca (self)                         182.79     1_966.14     2_148.93       0.4236          1.0546            1.0548         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   66.20       417.21       483.41       0.1211          1.4987            1.4523         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    66.20       448.13       514.34       0.3284          1.1039            1.0884         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    66.20       672.30       738.50       0.4385          1.0624            1.0494         1.53
ExhaustiveBinary-256-sign (self)                          66.20     1_457.97     1_524.17       0.3334          1.0988            1.0859         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)              266.61        50.89       317.50       0.1295          1.4040            1.3634         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)             266.61        53.72       320.33       0.1295          1.4040            1.3634         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)             266.61        56.56       323.16       0.1295          1.4040            1.3634         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)             266.61        98.30       364.91       0.3669          1.0780            1.0703         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)             266.61       146.28       412.89       0.4763          1.0477            1.0412         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)            266.61        94.90       361.51       0.3669          1.0780            1.0703         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)            266.61       147.48       414.09       0.4763          1.0477            1.0412         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)            266.61        97.47       364.08       0.3669          1.0780            1.0703         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)            266.61       147.56       414.17       0.4763          1.0477            1.0412         1.93
IVF-Binary-256-nl158-random (self)                       266.61       212.15       478.76       0.3716          1.0736            1.0691         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             208.68        45.00       253.68       0.1408          1.3609            1.3173         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             208.68        46.13       254.81       0.1408          1.3609            1.3173         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             208.68        48.90       257.58       0.1408          1.3609            1.3173         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            208.68        96.68       305.35       0.3889          1.0686            1.0624         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            208.68       146.95       355.63       0.4942          1.0435            1.0378         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            208.68        97.00       305.68       0.3889          1.0686            1.0624         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            208.68       146.99       355.67       0.4942          1.0435            1.0378         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            208.68        99.51       308.19       0.3889          1.0686            1.0624         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            208.68       148.79       357.47       0.4942          1.0435            1.0378         2.00
IVF-Binary-256-nl223-random (self)                       208.68       219.13       427.81       0.3927          1.0646            1.0619         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             233.05        50.51       283.56       0.1497          1.3331            1.2916         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             233.05        50.62       283.68       0.1497          1.3331            1.2916         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             233.05        55.70       288.76       0.1497          1.3331            1.2916         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            233.05       111.12       344.17       0.4042          1.0638            1.0580         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            233.05       164.18       397.23       0.5080          1.0410            1.0353         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            233.05       111.06       344.12       0.4042          1.0638            1.0580         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            233.05       169.94       403.00       0.5080          1.0410            1.0353         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            233.05       112.47       345.52       0.4042          1.0638            1.0580         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            233.05       169.14       402.20       0.5080          1.0410            1.0353         2.09
IVF-Binary-256-nl316-random (self)                       233.05       248.38       481.43       0.4082          1.0595            1.0572         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)                 315.58        42.72       358.30       0.1257          1.4100            1.3677         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)                315.58        45.27       360.85       0.1257          1.4100            1.3677         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)                315.58        47.20       362.78       0.1257          1.4100            1.3677         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)                315.58       102.24       417.82       0.3611          1.0855            1.0711         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)                315.58       160.08       475.66       0.4750          1.0512            1.0412         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)               315.58       102.93       418.51       0.3611          1.0855            1.0711         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)               315.58       157.95       473.53       0.4750          1.0512            1.0412         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)               315.58       107.06       422.64       0.3611          1.0855            1.0711         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)               315.58       160.26       475.84       0.4750          1.0512            1.0412         1.93
IVF-Binary-256-nl158-pca (self)                          315.58       220.81       536.39       0.3675          1.0789            1.0698         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                260.87        44.45       305.32       0.1368          1.3713            1.3234         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                260.87        45.24       306.11       0.1368          1.3713            1.3234         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                260.87        48.02       308.89       0.1368          1.3713            1.3234         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               260.87        93.36       354.22       0.3818          1.0759            1.0639         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               260.87       145.26       406.13       0.4945          1.0462            1.0376         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               260.87        93.60       354.47       0.3818          1.0759            1.0639         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               260.87       145.52       406.39       0.4945          1.0462            1.0376         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               260.87        97.60       358.47       0.3818          1.0759            1.0639         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               260.87       154.51       415.38       0.4945          1.0462            1.0376         2.00
IVF-Binary-256-nl223-pca (self)                          260.87       208.28       469.15       0.3882          1.0696            1.0628         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                266.24        47.48       313.72       0.1484          1.3354            1.2914         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                266.24        47.43       313.67       0.1484          1.3354            1.2914         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                266.24        50.45       316.69       0.1484          1.3354            1.2914         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               266.24        97.18       363.42       0.4017          1.0681            1.0585         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               266.24       145.59       411.83       0.5108          1.0426            1.0349         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               266.24        99.21       365.45       0.4017          1.0681            1.0585         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               266.24       145.89       412.13       0.5108          1.0426            1.0349         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               266.24        98.82       365.06       0.4017          1.0681            1.0585         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               266.24       148.51       414.75       0.5108          1.0426            1.0349         2.09
IVF-Binary-256-nl316-pca (self)                          266.24       217.17       483.41       0.4073          1.0624            1.0573         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)              277.15        64.71       341.85       0.1648          1.3234            1.3029         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)             277.15        69.42       346.57       0.1648          1.3234            1.3029         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)             277.15        71.85       349.00       0.1648          1.3234            1.3029         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)             277.15       114.79       391.94       0.3925          1.0635            1.0625         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)             277.15       166.53       443.68       0.5048          1.0390            1.0365         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)            277.15       117.10       394.25       0.3925          1.0635            1.0625         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)            277.15       169.54       446.68       0.5048          1.0390            1.0365         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)            277.15       117.29       394.44       0.3925          1.0635            1.0625         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)            277.15       170.78       447.93       0.5048          1.0390            1.0365         3.71
IVF-Binary-512-nl158-random (self)                       277.15       283.80       560.94       0.3950          1.0620            1.0624         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             222.20        62.23       284.43       0.1715          1.2995            1.2791         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             222.20        66.33       288.52       0.1715          1.2995            1.2791         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             222.20        69.55       291.75       0.1715          1.2995            1.2791         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            222.20       123.43       345.62       0.4022          1.0605            1.0593         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            222.20       167.61       389.81       0.5135          1.0375            1.0348         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            222.20       118.97       341.16       0.4022          1.0605            1.0593         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            222.20       170.64       392.84       0.5135          1.0375            1.0348         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            222.20       119.55       341.75       0.4022          1.0605            1.0593         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            222.20       171.98       394.17       0.5135          1.0375            1.0348         3.77
IVF-Binary-512-nl223-random (self)                       222.20       289.30       511.49       0.4038          1.0593            1.0592         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             209.11        65.27       274.38       0.1751          1.2887            1.2673         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             209.11        66.50       275.61       0.1751          1.2887            1.2673         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             209.11        70.04       279.15       0.1751          1.2887            1.2673         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            209.11       117.78       326.89       0.4084          1.0592            1.0577         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            209.11       174.00       383.11       0.5194          1.0367            1.0337         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            209.11       118.23       327.34       0.4084          1.0592            1.0577         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            209.11       169.47       378.58       0.5194          1.0367            1.0337         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            209.11       121.70       330.81       0.4084          1.0592            1.0577         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            209.11       173.02       382.13       0.5194          1.0367            1.0337         3.86
IVF-Binary-512-nl316-random (self)                       209.11       295.99       505.10       0.4103          1.0578            1.0576         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)                 288.93        57.96       346.89       0.1636          1.3225            1.2970         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)                288.93        61.22       350.15       0.1636          1.3225            1.2970         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)                288.93        64.17       353.10       0.1636          1.3225            1.2970         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)                288.93       111.16       400.08       0.3947          1.0651            1.0609         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)                288.93       163.41       452.33       0.5095          1.0396            1.0354         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)               288.93       115.66       404.59       0.3947          1.0651            1.0609         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)               288.93       166.77       455.70       0.5095          1.0396            1.0354         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)               288.93       116.54       405.47       0.3947          1.0651            1.0609         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)               288.93       172.25       461.18       0.5095          1.0396            1.0354         3.71
IVF-Binary-512-nl158-pca (self)                          288.93       282.76       571.69       0.3976          1.0625            1.0612         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                233.26        62.13       295.39       0.1692          1.3036            1.2757         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                233.26        63.54       296.80       0.1692          1.3036            1.2757         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                233.26        67.54       300.80       0.1692          1.3036            1.2757         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               233.26       115.24       348.50       0.4045          1.0617            1.0580         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               233.26       167.59       400.85       0.5170          1.0381            1.0341         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               233.26       114.98       348.24       0.4045          1.0617            1.0580         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               233.26       167.56       400.82       0.5170          1.0381            1.0341         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               233.26       122.79       356.05       0.4045          1.0617            1.0580         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               233.26       171.85       405.11       0.5170          1.0381            1.0341         3.77
IVF-Binary-512-nl223-pca (self)                          233.26       287.61       520.87       0.4068          1.0596            1.0581         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                283.53        65.41       348.94       0.1738          1.2897            1.2644         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                283.53        66.06       349.59       0.1738          1.2897            1.2644         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                283.53        69.59       353.12       0.1738          1.2897            1.2644         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               283.53       116.66       400.19       0.4116          1.0598            1.0561         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               283.53       168.29       451.82       0.5247          1.0370            1.0330         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               283.53       116.82       400.35       0.4116          1.0598            1.0561         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               283.53       168.77       452.30       0.5247          1.0370            1.0330         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               283.53       120.53       404.06       0.4116          1.0598            1.0561         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               283.53       172.20       455.73       0.5247          1.0370            1.0330         3.86
IVF-Binary-512-nl316-pca (self)                          283.53       292.40       575.93       0.4141          1.0578            1.0561         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)             314.07        89.75       403.82       0.1957          1.2639            1.2577         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)            314.07        93.83       407.90       0.1957          1.2639            1.2577         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)            314.07        98.11       412.18       0.1957          1.2639            1.2577         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)            314.07       146.04       460.11       0.4295          1.0528            1.0529         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)            314.07       200.38       514.45       0.5527          1.0314            1.0294         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)           314.07       149.89       463.96       0.4295          1.0528            1.0529         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)           314.07       205.91       519.98       0.5527          1.0314            1.0294         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)           314.07       153.98       468.05       0.4295          1.0528            1.0529         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)           314.07       213.52       527.59       0.5527          1.0314            1.0294         7.26
IVF-Binary-1024-nl158-random (self)                      314.07       402.33       716.40       0.4313          1.0525            1.0529         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            225.38        94.41       319.79       0.1975          1.2557            1.2492         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            225.38        95.79       321.17       0.1975          1.2557            1.2492         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            225.38       100.86       326.24       0.1975          1.2557            1.2492         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           225.38       151.24       376.62       0.4340          1.0517            1.0515         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           225.38       204.19       429.58       0.5561          1.0309            1.0289         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           225.38       150.67       376.06       0.4340          1.0517            1.0515         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           225.38       206.66       432.05       0.5561          1.0309            1.0289         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           225.38       157.03       382.42       0.4340          1.0517            1.0515         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           225.38       211.63       437.01       0.5561          1.0309            1.0289         7.32
IVF-Binary-1024-nl223-random (self)                      225.38       404.12       629.50       0.4356          1.0515            1.0517         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            251.56        96.69       348.26       0.1992          1.2506            1.2436         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)            251.56        97.92       349.49       0.1992          1.2506            1.2436         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)            251.56       103.17       354.74       0.1992          1.2506            1.2436         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)           251.56       152.38       403.95       0.4377          1.0511            1.0508         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)           251.56       208.05       459.61       0.5586          1.0306            1.0286         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)           251.56       152.08       403.64       0.4377          1.0511            1.0508         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)           251.56       208.11       459.67       0.5586          1.0306            1.0286         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)           251.56       157.69       409.25       0.4377          1.0511            1.0508         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)           251.56       213.97       465.53       0.5586          1.0306            1.0286         7.41
IVF-Binary-1024-nl316-random (self)                      251.56       409.51       661.08       0.4392          1.0507            1.0509         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)                318.91        95.46       414.37       0.1950          1.2612            1.2537         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)               318.91        95.80       414.72       0.1950          1.2612            1.2537         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)               318.91        97.88       416.79       0.1950          1.2612            1.2537         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)               318.91       146.85       465.76       0.4314          1.0521            1.0517         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)               318.91       201.60       520.52       0.5548          1.0311            1.0288         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)              318.91       148.85       467.76       0.4314          1.0521            1.0517         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)              318.91       205.95       524.86       0.5548          1.0311            1.0288         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)              318.91       153.25       472.16       0.4314          1.0521            1.0517         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)              318.91       209.53       528.45       0.5548          1.0311            1.0288         7.26
IVF-Binary-1024-nl158-pca (self)                         318.91       400.91       719.82       0.4322          1.0522            1.0524         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               261.34        93.48       354.82       0.1972          1.2521            1.2451         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               261.34        95.94       357.28       0.1972          1.2521            1.2451         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               261.34       100.85       362.19       0.1972          1.2521            1.2451         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              261.34       150.30       411.64       0.4364          1.0509            1.0507         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              261.34       202.93       464.27       0.5588          1.0306            1.0281         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              261.34       151.15       412.49       0.4364          1.0509            1.0507         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              261.34       207.32       468.66       0.5588          1.0306            1.0281         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              261.34       156.10       417.45       0.4364          1.0509            1.0507         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              261.34       223.30       484.64       0.5588          1.0306            1.0281         7.32
IVF-Binary-1024-nl223-pca (self)                         261.34       403.39       664.73       0.4366          1.0511            1.0511         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               286.48        96.16       382.64       0.1991          1.2470            1.2400         7.41
IVF-Binary-1024-nl316-np17-rf0-pca (query)               286.48        97.52       384.00       0.1991          1.2470            1.2400         7.41
IVF-Binary-1024-nl316-np25-rf0-pca (query)               286.48       103.64       390.12       0.1991          1.2470            1.2400         7.41
IVF-Binary-1024-nl316-np15-rf10-pca (query)              286.48       152.08       438.56       0.4397          1.0502            1.0497         7.41
IVF-Binary-1024-nl316-np15-rf20-pca (query)              286.48       208.81       495.29       0.5626          1.0302            1.0277         7.41
IVF-Binary-1024-nl316-np17-rf10-pca (query)              286.48       152.19       438.67       0.4397          1.0502            1.0497         7.41
IVF-Binary-1024-nl316-np17-rf20-pca (query)              286.48       206.61       493.09       0.5626          1.0302            1.0277         7.41
IVF-Binary-1024-nl316-np25-rf10-pca (query)              286.48       157.34       443.82       0.4397          1.0502            1.0497         7.41
IVF-Binary-1024-nl316-np25-rf20-pca (query)              286.48       217.14       503.61       0.5626          1.0302            1.0277         7.41
IVF-Binary-1024-nl316-pca (self)                         286.48       407.96       694.44       0.4409          1.0502            1.0503         7.41
IVF-Binary-256-nl158-np7-rf0-sign (query)                212.68       127.03       339.71       0.1215          1.4891            1.4318         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               212.68       128.81       341.49       0.1215          1.4891            1.4318         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               212.68       130.70       343.38       0.1215          1.4891            1.4318         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               212.68       163.34       376.02       0.3438          1.0943            1.0804         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               212.68       278.97       491.65       0.4524          1.0579            1.0456         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              212.68       162.77       375.45       0.3438          1.0943            1.0804         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              212.68       284.34       497.02       0.4524          1.0579            1.0456         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              212.68       165.04       377.72       0.3438          1.0943            1.0804         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              212.68       286.12       498.80       0.4524          1.0579            1.0456         1.68
IVF-Binary-256-nl158-sign (self)                         212.68       425.96       638.64       0.3486          1.0893            1.0789         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               152.59       128.58       281.17       0.1229          1.4837            1.4301         1.74
IVF-Binary-256-nl223-np14-rf0-sign (query)               152.59       129.42       282.01       0.1229          1.4837            1.4301         1.74
IVF-Binary-256-nl223-np21-rf0-sign (query)               152.59       134.20       286.79       0.1229          1.4837            1.4301         1.74
IVF-Binary-256-nl223-np11-rf10-sign (query)              152.59       164.04       316.63       0.3580          1.0888            1.0743         1.74
IVF-Binary-256-nl223-np11-rf20-sign (query)              152.59       284.78       437.37       0.4581          1.0560            1.0446         1.74
IVF-Binary-256-nl223-np14-rf10-sign (query)              152.59       164.78       317.38       0.3580          1.0888            1.0743         1.74
IVF-Binary-256-nl223-np14-rf20-sign (query)              152.59       284.19       436.78       0.4581          1.0560            1.0446         1.74
IVF-Binary-256-nl223-np21-rf10-sign (query)              152.59       167.03       319.62       0.3580          1.0888            1.0743         1.74
IVF-Binary-256-nl223-np21-rf20-sign (query)              152.59       288.15       440.74       0.4581          1.0560            1.0446         1.74
IVF-Binary-256-nl223-sign (self)                         152.59       433.83       586.42       0.3631          1.0837            1.0727         1.74
IVF-Binary-256-nl316-np15-rf0-sign (query)               177.98       129.80       307.78       0.1238          1.4746            1.4172         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               177.98       130.69       308.67       0.1238          1.4746            1.4172         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               177.98       133.26       311.24       0.1238          1.4746            1.4172         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              177.98       166.74       344.73       0.3594          1.0870            1.0734         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              177.98       296.88       474.86       0.4564          1.0564            1.0447         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              177.98       170.34       348.33       0.3594          1.0870            1.0734         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              177.98       290.28       468.26       0.4564          1.0564            1.0447         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              177.98       169.91       347.89       0.3594          1.0870            1.0734         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              177.98       301.54       479.52       0.4564          1.0564            1.0447         1.84
IVF-Binary-256-nl316-sign (self)                         177.98       442.68       620.66       0.3649          1.0824            1.0718         1.84
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 512D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.45     1_365.89     1_434.34       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.45     4_575.02     4_643.47       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                132.51       264.24       396.75       0.1109          1.3512            1.3092         2.03
ExhaustiveBinary-256-random-rf10 (query)                 132.51       397.63       530.14       0.3143          1.0825            1.0613         2.03
ExhaustiveBinary-256-random-rf20 (query)                 132.51       542.50       675.01       0.4100          1.0523            1.0368         2.03
ExhaustiveBinary-256-random (self)                       132.51     1_242.94     1_375.45       0.3161          1.0784            1.0600         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   247.82       292.59       540.41       0.1167          1.3480            1.2981         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    247.82       450.40       698.22       0.3159          1.0791            1.0596         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    247.82       577.03       824.85       0.4121          1.0505            1.0362         2.03
ExhaustiveBinary-256-pca (self)                          247.82     1_239.55     1_487.36       0.3171          1.0782            1.0588         2.03
ExhaustiveBinary-512-random_no_rr (query)                215.41       399.49       614.90       0.1528          1.2601            1.2299         4.05
ExhaustiveBinary-512-random-rf10 (query)                 215.41       547.13       762.54       0.3465          1.0565            1.0514         4.05
ExhaustiveBinary-512-random-rf20 (query)                 215.41       684.27       899.69       0.4452          1.0358            1.0312         4.05
ExhaustiveBinary-512-random (self)                       215.41     1_760.89     1_976.30       0.3476          1.0547            1.0512         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   297.76       396.94       694.70       0.1558          1.2535            1.2254         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    297.76       541.32       839.08       0.3512          1.0523            1.0507         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    297.76       674.55       972.31       0.4484          1.0329            1.0309         4.05
ExhaustiveBinary-512-pca (self)                          297.76     1_734.74     2_032.51       0.3515          1.0522            1.0505         4.05
ExhaustiveBinary-1024-random_no_rr (query)               259.06       594.08       853.14       0.1816          1.2043            1.1936         8.11
ExhaustiveBinary-1024-random-rf10 (query)                259.06       749.71     1_008.78       0.3747          1.0447            1.0452         8.11
ExhaustiveBinary-1024-random-rf20 (query)                259.06       871.12     1_130.18       0.4789          1.0282            1.0270         8.11
ExhaustiveBinary-1024-random (self)                      259.06     2_406.48     2_665.55       0.3754          1.0447            1.0451         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  361.92       593.31       955.23       0.1832          1.2013            1.1905         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   361.92       743.65     1_105.58       0.3798          1.0434            1.0443         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   361.92       900.07     1_261.99       0.4867          1.0272            1.0261         8.11
ExhaustiveBinary-1024-pca (self)                         361.92     2_449.58     2_811.50       0.3787          1.0436            1.0444         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   86.28       655.11       741.39       0.1518          1.2701            1.2528         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    86.28       721.21       807.49       0.3399          1.0607            1.0535         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    86.28     1_066.51     1_152.80       0.4406          1.0369            1.0319         3.05
ExhaustiveBinary-512-sign (self)                          86.28     2_318.98     2_405.26       0.3409          1.0595            1.0531         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)              468.86        79.62       548.48       0.1197          1.3113            1.2724         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)             468.86        90.20       559.06       0.1197          1.3113            1.2724         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)             468.86        83.43       552.29       0.1197          1.3113            1.2724         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)             468.86       160.50       629.36       0.3417          1.0668            1.0524         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)             468.86       257.58       726.44       0.4430          1.0412            1.0314         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)            468.86       163.04       631.91       0.3417          1.0668            1.0524         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)            468.86       265.29       734.15       0.4430          1.0412            1.0314         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)            468.86       166.81       635.68       0.3417          1.0668            1.0524         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)            468.86       249.88       718.75       0.4430          1.0412            1.0314         2.34
IVF-Binary-256-nl158-random (self)                       468.86       299.45       768.31       0.3436          1.0628            1.0515         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)             364.07        72.20       436.27       0.1327          1.2745            1.2362         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)             364.07        73.76       437.83       0.1327          1.2745            1.2362         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)             364.07        76.36       440.43       0.1327          1.2745            1.2362         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)            364.07       170.28       534.34       0.3705          1.0551            1.0451         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)            364.07       260.78       624.85       0.4727          1.0355            1.0273         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)            364.07       160.70       524.76       0.3705          1.0551            1.0451         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)            364.07       256.01       620.08       0.4727          1.0355            1.0273         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)            364.07       162.56       526.63       0.3705          1.0551            1.0451         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)            364.07       257.33       621.40       0.4727          1.0355            1.0273         2.46
IVF-Binary-256-nl223-random (self)                       364.07       316.75       680.82       0.3719          1.0512            1.0446         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)             902.02        78.81       980.83       0.1423          1.2543            1.2154         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)             902.02        80.46       982.48       0.1423          1.2543            1.2154         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)             902.02        91.21       993.23       0.1423          1.2543            1.2154         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)            902.02       169.58     1_071.59       0.3810          1.0511            1.0426         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)            902.02       266.21     1_168.23       0.4794          1.0335            1.0265         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)            902.02       168.80     1_070.81       0.3810          1.0511            1.0426         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)            902.02       274.03     1_176.04       0.4794          1.0335            1.0265         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)            902.02       171.76     1_073.78       0.3810          1.0511            1.0426         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)            902.02       261.08     1_163.09       0.4794          1.0335            1.0265         2.65
IVF-Binary-256-nl316-random (self)                       902.02       336.42     1_238.43       0.3830          1.0473            1.0420         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)                 546.12        69.59       615.71       0.1256          1.3082            1.2639         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)                546.12        70.29       616.41       0.1256          1.3083            1.2639         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)                546.12        73.25       619.37       0.1256          1.3083            1.2639         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)                546.12       155.24       701.36       0.3406          1.0646            1.0521         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)                546.12       254.64       800.76       0.4427          1.0398            1.0314         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)               546.12       158.29       704.41       0.3406          1.0646            1.0521         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)               546.12       257.48       803.59       0.4427          1.0398            1.0314         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)               546.12       157.99       704.11       0.3406          1.0646            1.0521         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)               546.12       265.93       812.04       0.4427          1.0398            1.0314         2.34
IVF-Binary-256-nl158-pca (self)                          546.12       314.97       861.09       0.3421          1.0637            1.0516         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)                685.04        74.80       759.83       0.1369          1.2750            1.2317         2.46
IVF-Binary-256-nl223-np14-rf0-pca (query)                685.04        82.11       767.14       0.1369          1.2750            1.2317         2.46
IVF-Binary-256-nl223-np21-rf0-pca (query)                685.04        76.50       761.54       0.1369          1.2750            1.2317         2.46
IVF-Binary-256-nl223-np11-rf10-pca (query)               685.04       164.47       849.51       0.3675          1.0532            1.0453         2.46
IVF-Binary-256-nl223-np11-rf20-pca (query)               685.04       266.44       951.48       0.4689          1.0345            1.0279         2.46
IVF-Binary-256-nl223-np14-rf10-pca (query)               685.04       174.27       859.31       0.3675          1.0532            1.0453         2.46
IVF-Binary-256-nl223-np14-rf20-pca (query)               685.04       262.31       947.34       0.4689          1.0345            1.0279         2.46
IVF-Binary-256-nl223-np21-rf10-pca (query)               685.04       165.39       850.43       0.3675          1.0532            1.0453         2.46
IVF-Binary-256-nl223-np21-rf20-pca (query)               685.04       269.07       954.10       0.4689          1.0345            1.0279         2.46
IVF-Binary-256-nl223-pca (self)                          685.04       325.78     1_010.81       0.3695          1.0523            1.0446         2.46
IVF-Binary-256-nl316-np15-rf0-pca (query)                488.36        80.75       569.11       0.1441          1.2600            1.2139         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)                488.36        82.62       570.98       0.1441          1.2600            1.2139         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)                488.36        83.01       571.37       0.1441          1.2600            1.2139         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)               488.36       171.31       659.67       0.3777          1.0505            1.0427         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)               488.36       273.45       761.81       0.4765          1.0335            1.0268         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)               488.36       166.16       654.52       0.3777          1.0505            1.0427         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)               488.36       267.43       755.79       0.4765          1.0335            1.0268         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)               488.36       178.03       666.39       0.3777          1.0505            1.0427         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)               488.36       273.59       761.95       0.4765          1.0335            1.0268         2.65
IVF-Binary-256-nl316-pca (self)                          488.36       348.57       836.93       0.3797          1.0486            1.0422         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)              528.41       105.31       633.72       0.1585          1.2385            1.2123         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)             528.41       108.28       636.70       0.1585          1.2385            1.2123         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)             528.41       116.50       644.91       0.1585          1.2385            1.2123         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)             528.41       190.28       718.70       0.3600          1.0509            1.0475         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)             528.41       292.08       820.49       0.4602          1.0328            1.0290         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)            528.41       194.07       722.48       0.3600          1.0509            1.0475         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)            528.41       324.52       852.93       0.4602          1.0328            1.0290         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)            528.41       245.62       774.03       0.3600          1.0509            1.0475         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)            528.41       347.39       875.81       0.4602          1.0328            1.0290         4.36
IVF-Binary-512-nl158-random (self)                       528.41       568.70     1_097.11       0.3609          1.0495            1.0474         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)             494.69       113.29       607.99       0.1639          1.2226            1.1963         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)             494.69       111.69       606.38       0.1639          1.2226            1.1963         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)             494.69       110.43       605.12       0.1639          1.2226            1.1963         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)            494.69       196.40       691.10       0.3712          1.0477            1.0447         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)            494.69       292.30       787.00       0.4708          1.0311            1.0276         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)            494.69       225.58       720.27       0.3712          1.0477            1.0447         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)            494.69       356.86       851.55       0.4708          1.0311            1.0276         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)            494.69       217.55       712.25       0.3712          1.0477            1.0447         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)            494.69       296.19       790.88       0.4708          1.0311            1.0276         4.49
IVF-Binary-512-nl223-random (self)                       494.69       467.51       962.20       0.3718          1.0466            1.0447         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)             458.24       108.40       566.65       0.1670          1.2153            1.1895         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)             458.24       108.11       566.36       0.1670          1.2153            1.1895         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)             458.24       111.15       569.39       0.1670          1.2153            1.1895         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)            458.24       195.50       653.74       0.3745          1.0468            1.0440         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)            458.24       290.58       748.83       0.4729          1.0307            1.0274         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)            458.24       195.18       653.42       0.3745          1.0468            1.0440         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)            458.24       297.10       755.34       0.4729          1.0307            1.0274         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)            458.24       202.90       661.14       0.3745          1.0468            1.0440         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)            458.24       299.65       757.89       0.4729          1.0307            1.0274         4.67
IVF-Binary-512-nl316-random (self)                       458.24       481.44       939.69       0.3752          1.0456            1.0439         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)                 611.30        93.49       704.79       0.1616          1.2328            1.2085         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)                611.30        97.42       708.72       0.1616          1.2328            1.2085         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)                611.30       101.83       713.13       0.1616          1.2328            1.2085         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)                611.30       185.07       796.37       0.3633          1.0479            1.0472         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)                611.30       280.08       891.38       0.4636          1.0303            1.0286         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)               611.30       186.17       797.46       0.3633          1.0479            1.0472         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)               611.30       283.56       894.85       0.4636          1.0303            1.0286         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)               611.30       187.79       799.09       0.3633          1.0479            1.0472         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)               611.30       288.63       899.93       0.4636          1.0303            1.0286         4.36
IVF-Binary-512-nl158-pca (self)                          611.30       431.56     1_042.86       0.3635          1.0480            1.0470         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)                492.27       100.55       592.82       0.1675          1.2167            1.1920         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)                492.27       100.80       593.06       0.1675          1.2167            1.1920         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)                492.27       105.11       597.37       0.1675          1.2167            1.1920         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)               492.27       191.44       683.71       0.3732          1.0455            1.0447         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)               492.27       283.54       775.81       0.4741          1.0292            1.0272         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)               492.27       193.09       685.36       0.3732          1.0455            1.0447         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)               492.27       284.00       776.26       0.4741          1.0292            1.0272         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)               492.27       199.65       691.91       0.3732          1.0455            1.0447         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)               492.27       300.99       793.25       0.4741          1.0292            1.0272         4.49
IVF-Binary-512-nl223-pca (self)                          492.27       450.70       942.96       0.3737          1.0457            1.0445         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)                658.39       108.79       767.18       0.1704          1.2102            1.1853         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)                658.39       112.76       771.15       0.1704          1.2102            1.1853         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)                658.39       120.63       779.02       0.1704          1.2102            1.1853         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)               658.39       198.92       857.31       0.3768          1.0447            1.0438         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)               658.39       296.37       954.76       0.4766          1.0288            1.0269         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)               658.39       202.93       861.31       0.3768          1.0447            1.0438         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)               658.39       306.89       965.28       0.4766          1.0288            1.0269         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)               658.39       200.24       858.62       0.3768          1.0447            1.0438         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)               658.39       305.15       963.53       0.4766          1.0288            1.0269         4.67
IVF-Binary-512-nl316-pca (self)                          658.39       467.66     1_126.05       0.3771          1.0448            1.0438         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)             575.19       147.77       722.96       0.1839          1.1956            1.1862         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)            575.19       149.90       725.09       0.1839          1.1956            1.1862         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)            575.19       155.61       730.79       0.1839          1.1956            1.1862         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)            575.19       238.31       813.49       0.3813          1.0430            1.0434         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)            575.19       346.81       922.00       0.4874          1.0271            1.0258         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)           575.19       243.86       819.04       0.3813          1.0430            1.0434         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)           575.19       370.45       945.63       0.4874          1.0271            1.0258         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)           575.19       249.09       824.28       0.3813          1.0430            1.0434         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)           575.19       352.42       927.61       0.4874          1.0271            1.0258         8.42
IVF-Binary-1024-nl158-random (self)                      575.19       669.89     1_245.08       0.3822          1.0430            1.0433         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)            529.02       162.51       691.53       0.1863          1.1878            1.1788         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)            529.02       167.25       696.27       0.1863          1.1878            1.1788         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)            529.02       164.72       693.74       0.1863          1.1878            1.1788         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)           529.02       259.11       788.13       0.3872          1.0418            1.0421         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)           529.02       348.34       877.36       0.4937          1.0264            1.0251         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)           529.02       258.05       787.08       0.3872          1.0418            1.0421         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)           529.02       352.72       881.74       0.4937          1.0264            1.0251         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)           529.02       256.55       785.57       0.3872          1.0418            1.0421         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)           529.02       362.30       891.32       0.4937          1.0264            1.0251         8.54
IVF-Binary-1024-nl223-random (self)                      529.02       704.78     1_233.80       0.3883          1.0417            1.0420         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)            630.09       167.64       797.72       0.1869          1.1852            1.1759         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)            630.09       165.48       795.56       0.1869          1.1852            1.1759         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)            630.09       170.11       800.20       0.1869          1.1852            1.1759         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)           630.09       257.29       887.38       0.3887          1.0414            1.0418         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)           630.09       363.43       993.51       0.4942          1.0263            1.0248         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)           630.09       262.95       893.04       0.3887          1.0414            1.0418         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)           630.09       370.58     1_000.66       0.4942          1.0263            1.0248         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)           630.09       270.07       900.15       0.3887          1.0414            1.0418         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)           630.09       362.97       993.06       0.4942          1.0263            1.0248         8.72
IVF-Binary-1024-nl316-random (self)                      630.09       696.58     1_326.66       0.3898          1.0413            1.0417         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)                679.77       150.48       830.25       0.1854          1.1929            1.1838         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)               679.77       151.51       831.27       0.1854          1.1929            1.1838         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)               679.77       159.07       838.84       0.1854          1.1929            1.1838         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)               679.77       244.34       924.11       0.3869          1.0417            1.0427         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)               679.77       338.94     1_018.71       0.4952          1.0260            1.0250         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)              679.77       242.63       922.39       0.3869          1.0417            1.0427         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)              679.77       346.56     1_026.33       0.4952          1.0260            1.0250         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)              679.77       245.60       925.37       0.3869          1.0417            1.0427         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)              679.77       351.81     1_031.58       0.4952          1.0260            1.0250         8.42
IVF-Binary-1024-nl158-pca (self)                         679.77       667.86     1_347.63       0.3858          1.0418            1.0427         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)               674.10       167.59       841.68       0.1873          1.1862            1.1771         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)               674.10       165.89       839.99       0.1873          1.1862            1.1771         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)               674.10       165.71       839.80       0.1873          1.1862            1.1771         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)              674.10       269.36       943.45       0.3923          1.0406            1.0413         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)              674.10       356.55     1_030.64       0.5008          1.0254            1.0245         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)              674.10       252.53       926.63       0.3923          1.0406            1.0413         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)              674.10       375.47     1_049.57       0.5008          1.0254            1.0245         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)              674.10       259.39       933.49       0.3923          1.0406            1.0413         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)              674.10       368.95     1_043.04       0.5008          1.0254            1.0245         8.54
IVF-Binary-1024-nl223-pca (self)                         674.10       698.07     1_372.17       0.3913          1.0407            1.0414         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)               696.52       177.73       874.24       0.1884          1.1834            1.1743         8.72
IVF-Binary-1024-nl316-np17-rf0-pca (query)               696.52       165.12       861.64       0.1884          1.1834            1.1743         8.72
IVF-Binary-1024-nl316-np25-rf0-pca (query)               696.52       170.24       866.76       0.1884          1.1834            1.1743         8.72
IVF-Binary-1024-nl316-np15-rf10-pca (query)              696.52       259.09       955.61       0.3933          1.0404            1.0410         8.72
IVF-Binary-1024-nl316-np15-rf20-pca (query)              696.52       359.45     1_055.97       0.5013          1.0254            1.0244         8.72
IVF-Binary-1024-nl316-np17-rf10-pca (query)              696.52       258.52       955.04       0.3933          1.0404            1.0410         8.72
IVF-Binary-1024-nl316-np17-rf20-pca (query)              696.52       361.23     1_057.75       0.5013          1.0254            1.0244         8.72
IVF-Binary-1024-nl316-np25-rf10-pca (query)              696.52       268.38       964.90       0.3933          1.0404            1.0410         8.72
IVF-Binary-1024-nl316-np25-rf20-pca (query)              696.52       378.28     1_074.80       0.5013          1.0254            1.0244         8.72
IVF-Binary-1024-nl316-pca (self)                         696.52       728.58     1_425.10       0.3927          1.0405            1.0411         8.72
IVF-Binary-512-nl158-np7-rf0-sign (query)                411.42       261.19       672.60       0.1520          1.2666            1.2502         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)               411.42       261.78       673.20       0.1520          1.2666            1.2502         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)               411.42       264.15       675.57       0.1520          1.2666            1.2502         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)               411.42       332.18       743.60       0.3458          1.0574            1.0515         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)               411.42       579.78       991.20       0.4446          1.0360            1.0312         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)              411.42       337.59       749.01       0.3458          1.0574            1.0515         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)              411.42       615.11     1_026.53       0.4446          1.0360            1.0312         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)              411.42       330.08       741.50       0.3458          1.0574            1.0515         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)              411.42       580.96       992.38       0.4446          1.0360            1.0312         3.36
IVF-Binary-512-nl158-sign (self)                         411.42       878.42     1_289.83       0.3470          1.0564            1.0511         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               340.44       262.93       603.38       0.1523          1.2671            1.2491         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               340.44       261.99       602.44       0.1523          1.2671            1.2491         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               340.44       266.66       607.11       0.1523          1.2671            1.2491         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              340.44       334.58       675.02       0.3518          1.0555            1.0499         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              340.44       588.58       929.02       0.4484          1.0349            1.0307         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              340.44       331.47       671.91       0.3518          1.0555            1.0499         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              340.44       584.27       924.71       0.4484          1.0349            1.0307         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              340.44       336.33       676.77       0.3518          1.0555            1.0499         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              340.44       591.16       931.60       0.4484          1.0349            1.0307         3.49
IVF-Binary-512-nl223-sign (self)                         340.44       898.97     1_239.42       0.3526          1.0544            1.0494         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)               386.85       275.17       662.03       0.1527          1.2740            1.2488         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)               386.85       268.79       655.64       0.1527          1.2740            1.2488         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)               386.85       269.91       656.76       0.1527          1.2740            1.2488         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)              386.85       334.85       721.70       0.3528          1.0553            1.0495         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)              386.85       589.61       976.46       0.4490          1.0348            1.0306         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)              386.85       335.69       722.54       0.3528          1.0553            1.0495         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)              386.85       591.04       977.89       0.4490          1.0348            1.0306         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)              386.85       341.29       728.14       0.3528          1.0553            1.0495         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)              386.85       596.87       983.72       0.4490          1.0348            1.0306         3.67
IVF-Binary-512-nl316-sign (self)                         386.85       909.74     1_296.60       0.3536          1.0540            1.0493         3.67
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 768D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       100.39     1_958.06     2_058.46       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.39     6_782.08     6_882.47       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                202.85       284.43       487.28       0.1140          1.2809            1.2433         2.28
ExhaustiveBinary-256-random-rf10 (query)                 202.85       430.06       632.91       0.3148          1.0656            1.0476         2.28
ExhaustiveBinary-256-random-rf20 (query)                 202.85       616.44       819.29       0.4075          1.0420            1.0293         2.28
ExhaustiveBinary-256-random (self)                       202.85     1_376.09     1_578.94       0.3168          1.0618            1.0471         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   408.17       286.40       694.57       0.1054          1.3026            1.2617         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    408.17       421.18       829.34       0.3012          1.0735            1.0517         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    408.17       583.00       991.17       0.3931          1.0471            1.0315         2.28
ExhaustiveBinary-256-pca (self)                          408.17     1_311.05     1_719.22       0.3048          1.0710            1.0504         2.28
ExhaustiveBinary-512-random_no_rr (query)                302.53       433.04       735.57       0.1506          1.2094            1.1809         4.55
ExhaustiveBinary-512-random-rf10 (query)                 302.53       573.75       876.28       0.3395          1.0453            1.0426         4.55
ExhaustiveBinary-512-random-rf20 (query)                 302.53       749.05     1_051.58       0.4326          1.0293            1.0264         4.55
ExhaustiveBinary-512-random (self)                       302.53     1_816.88     2_119.40       0.3401          1.0435            1.0423         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   498.09       444.58       942.67       0.1459          1.2162            1.1914         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    498.09       571.19     1_069.28       0.3341          1.0468            1.0435         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    498.09       728.81     1_226.90       0.4278          1.0295            1.0269         4.55
ExhaustiveBinary-512-pca (self)                          498.09     1_842.84     2_340.93       0.3355          1.0454            1.0433         4.55
ExhaustiveBinary-1024-random_no_rr (query)               499.07       633.28     1_132.35       0.1761          1.1673            1.1571         9.11
ExhaustiveBinary-1024-random-rf10 (query)                499.07       806.32     1_305.39       0.3603          1.0383            1.0383         9.11
ExhaustiveBinary-1024-random-rf20 (query)                499.07       983.94     1_483.01       0.4618          1.0244            1.0230         9.11
ExhaustiveBinary-1024-random (self)                      499.07     2_643.48     3_142.55       0.3602          1.0377            1.0383         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  706.87       643.63     1_350.50       0.1756          1.1686            1.1586         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   706.87       810.06     1_516.93       0.3594          1.0382            1.0385         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   706.87       986.27     1_693.13       0.4576          1.0246            1.0237         9.11
ExhaustiveBinary-1024-pca (self)                         706.87     2_658.43     3_365.30       0.3584          1.0383            1.0389         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  130.93       827.51       958.43       0.1691          1.1871            1.1718         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   130.93       908.39     1_039.31       0.3431          1.0433            1.0415         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   130.93     1_419.83     1_550.75       0.4437          1.0266            1.0250         4.58
ExhaustiveBinary-768-sign (self)                         130.93     2_944.05     3_074.98       0.3438          1.0424            1.0413         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)              616.81        95.47       712.28       0.1208          1.2561            1.2248         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)             616.81        97.67       714.48       0.1208          1.2561            1.2248         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)             616.81        99.62       716.43       0.1208          1.2561            1.2248         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)             616.81       200.47       817.28       0.3292          1.0572            1.0437         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)             616.81       318.31       935.12       0.4271          1.0368            1.0267         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)            616.81       202.21       819.02       0.3292          1.0572            1.0437         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)            616.81       316.15       932.96       0.4271          1.0368            1.0267         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)            616.81       203.17       819.98       0.3292          1.0572            1.0437         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)            616.81       318.30       935.11       0.4271          1.0368            1.0267         2.74
IVF-Binary-256-nl158-random (self)                       616.81       419.88     1_036.69       0.3307          1.0538            1.0434         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)             480.43        93.43       573.85       0.1302          1.2329            1.1991         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)             480.43        94.64       575.06       0.1302          1.2329            1.1991         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)             480.43        98.31       578.74       0.1302          1.2329            1.1991         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)            480.43       206.05       686.48       0.3508          1.0493            1.0391         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)            480.43       317.76       798.19       0.4513          1.0316            1.0237         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)            480.43       203.27       683.70       0.3508          1.0493            1.0391         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)            480.43       320.30       800.73       0.4513          1.0316            1.0237         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)            480.43       205.77       686.20       0.3508          1.0493            1.0391         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)            480.43       321.10       801.53       0.4513          1.0316            1.0237         2.93
IVF-Binary-256-nl223-random (self)                       480.43       441.94       922.36       0.3526          1.0462            1.0387         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)             528.97       104.42       633.39       0.1390          1.2157            1.1768         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)             528.97       103.48       632.45       0.1390          1.2157            1.1768         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)             528.97       105.05       634.02       0.1390          1.2157            1.1768         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)            528.97       234.58       763.55       0.3634          1.0448            1.0364         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)            528.97       334.34       863.31       0.4635          1.0295            1.0225         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)            528.97       211.86       740.82       0.3634          1.0448            1.0364         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)            528.97       331.36       860.32       0.4635          1.0295            1.0225         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)            528.97       223.59       752.56       0.3634          1.0448            1.0364         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)            528.97       327.93       856.90       0.4635          1.0295            1.0225         3.21
IVF-Binary-256-nl316-random (self)                       528.97       469.82       998.78       0.3652          1.0415            1.0361         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)                 819.36        85.28       904.64       0.1127          1.2726            1.2404         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)                819.36        87.40       906.75       0.1127          1.2726            1.2404         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)                819.36        90.46       909.82       0.1127          1.2726            1.2404         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)                819.36       192.01     1_011.37       0.3191          1.0634            1.0465         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)                819.36       304.01     1_123.37       0.4160          1.0396            1.0281         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)               819.36       198.26     1_017.62       0.3191          1.0634            1.0465         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)               819.36       308.61     1_127.97       0.4160          1.0396            1.0281         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)               819.36       196.32     1_015.68       0.3191          1.0634            1.0465         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)               819.36       315.75     1_135.11       0.4160          1.0396            1.0281         2.74
IVF-Binary-256-nl158-pca (self)                          819.36       392.84     1_212.20       0.3223          1.0605            1.0455         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)                687.02        99.80       786.82       0.1236          1.2441            1.2112         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)                687.02        94.49       781.51       0.1236          1.2441            1.2112         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)                687.02        97.04       784.05       0.1236          1.2441            1.2112         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)               687.02       201.44       888.46       0.3467          1.0533            1.0400         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)               687.02       315.40     1_002.41       0.4492          1.0332            1.0241         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)               687.02       201.15       888.17       0.3467          1.0533            1.0400         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)               687.02       316.41     1_003.43       0.4492          1.0332            1.0241         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)               687.02       203.51       890.53       0.3467          1.0533            1.0400         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)               687.02       317.99     1_005.01       0.4492          1.0332            1.0241         2.93
IVF-Binary-256-nl223-pca (self)                          687.02       414.04     1_101.06       0.3497          1.0507            1.0393         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)                819.91       102.83       922.74       0.1340          1.2222            1.1895         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)                819.91       106.13       926.04       0.1340          1.2222            1.1895         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)                819.91       104.76       924.66       0.1340          1.2222            1.1895         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)               819.91       214.56     1_034.46       0.3691          1.0452            1.0357         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)               819.91       325.21     1_145.12       0.4697          1.0284            1.0220         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)               819.91       210.06     1_029.96       0.3691          1.0452            1.0357         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)               819.91       336.16     1_156.07       0.4697          1.0284            1.0220         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)               819.91       211.75     1_031.66       0.3691          1.0452            1.0357         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)               819.91       339.96     1_159.86       0.4697          1.0284            1.0220         3.21
IVF-Binary-256-nl316-pca (self)                          819.91       453.72     1_273.63       0.3727          1.0425            1.0352         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)              739.70       130.75       870.45       0.1546          1.1976            1.1715         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)             739.70       130.08       869.78       0.1546          1.1976            1.1715         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)             739.70       138.39       878.09       0.1546          1.1976            1.1715         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)             739.70       238.51       978.21       0.3469          1.0430            1.0407         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)             739.70       351.03     1_090.73       0.4426          1.0279            1.0250         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)            739.70       238.77       978.46       0.3469          1.0430            1.0407         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)            739.70       360.81     1_100.51       0.4426          1.0279            1.0250         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)            739.70       242.13       981.83       0.3469          1.0430            1.0407         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)            739.70       359.30     1_099.00       0.4426          1.0279            1.0250         5.02
IVF-Binary-512-nl158-random (self)                       739.70       579.20     1_318.90       0.3471          1.0415            1.0405         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)             592.20       134.33       726.53       0.1594          1.1855            1.1597         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)             592.20       134.39       726.60       0.1594          1.1855            1.1597         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)             592.20       141.36       733.56       0.1594          1.1855            1.1597         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)            592.20       243.15       835.35       0.3564          1.0404            1.0382         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)            592.20       368.06       960.26       0.4533          1.0263            1.0239         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)            592.20       243.28       835.48       0.3564          1.0404            1.0382         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)            592.20       363.18       955.38       0.4533          1.0263            1.0239         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)            592.20       248.04       840.24       0.3564          1.0404            1.0382         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)            592.20       373.89       966.09       0.4533          1.0263            1.0239         5.21
IVF-Binary-512-nl223-random (self)                       592.20       603.27     1_195.48       0.3569          1.0390            1.0383         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)             647.11       140.85       787.96       0.1628          1.1775            1.1525         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)             647.11       144.44       791.55       0.1628          1.1775            1.1525         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)             647.11       150.27       797.38       0.1628          1.1775            1.1525         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)            647.11       251.16       898.27       0.3616          1.0395            1.0373         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)            647.11       371.30     1_018.41       0.4582          1.0259            1.0234         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)            647.11       249.90       897.01       0.3616          1.0395            1.0373         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)            647.11       372.94     1_020.05       0.4582          1.0259            1.0234         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)            647.11       254.50       901.61       0.3616          1.0395            1.0373         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)            647.11       380.50     1_027.61       0.4582          1.0259            1.0234         5.48
IVF-Binary-512-nl316-random (self)                       647.11       634.49     1_281.60       0.3618          1.0380            1.0373         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)                 927.63       121.09     1_048.73       0.1500          1.2030            1.1794         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)                927.63       124.72     1_052.36       0.1500          1.2030            1.1794         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)                927.63       127.00     1_054.64       0.1500          1.2030            1.1794         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)                927.63       239.02     1_166.65       0.3425          1.0437            1.0412         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)                927.63       355.48     1_283.11       0.4393          1.0279            1.0255         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)               927.63       232.18     1_159.82       0.3425          1.0437            1.0412         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)               927.63       352.75     1_280.39       0.4393          1.0279            1.0255         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)               927.63       237.11     1_164.74       0.3425          1.0437            1.0412         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)               927.63       357.41     1_285.05       0.4393          1.0279            1.0255         5.02
IVF-Binary-512-nl158-pca (self)                          927.63       563.06     1_490.69       0.3438          1.0428            1.0411         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)                778.70       130.27       908.97       0.1561          1.1880            1.1643         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)                778.70       130.66       909.37       0.1561          1.1880            1.1643         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)                778.70       134.13       912.83       0.1561          1.1880            1.1643         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)               778.70       250.14     1_028.85       0.3559          1.0404            1.0385         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)               778.70       371.77     1_150.47       0.4523          1.0263            1.0239         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)               778.70       240.00     1_018.71       0.3559          1.0404            1.0385         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)               778.70       367.33     1_146.03       0.4523          1.0263            1.0239         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)               778.70       245.17     1_023.88       0.3559          1.0404            1.0385         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)               778.70       366.97     1_145.67       0.4523          1.0263            1.0239         5.21
IVF-Binary-512-nl223-pca (self)                          778.70       586.62     1_365.33       0.3565          1.0396            1.0384         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)                835.02       139.98       975.00       0.1609          1.1777            1.1539         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)                835.02       139.41       974.43       0.1609          1.1777            1.1539         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)                835.02       142.65       977.67       0.1609          1.1777            1.1539         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)               835.02       248.77     1_083.79       0.3620          1.0392            1.0373         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)               835.02       368.38     1_203.40       0.4573          1.0258            1.0233         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)               835.02       247.88     1_082.90       0.3620          1.0392            1.0373         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)               835.02       369.99     1_205.00       0.4573          1.0258            1.0233         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)               835.02       253.47     1_088.49       0.3620          1.0392            1.0373         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)               835.02       376.40     1_211.42       0.4573          1.0258            1.0233         5.48
IVF-Binary-512-nl316-pca (self)                          835.02       630.76     1_465.78       0.3629          1.0383            1.0371         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)             927.91       201.22     1_129.14       0.1776          1.1627            1.1532         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)            927.91       201.91     1_129.82       0.1776          1.1627            1.1532         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)            927.91       211.55     1_139.47       0.1776          1.1627            1.1532         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)            927.91       311.74     1_239.66       0.3646          1.0374            1.0373         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)            927.91       439.84     1_367.75       0.4669          1.0238            1.0224         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)           927.91       317.53     1_245.44       0.3646          1.0374            1.0373         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)           927.91       448.16     1_376.07       0.4669          1.0238            1.0224         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)           927.91       326.43     1_254.35       0.3646          1.0374            1.0373         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)           927.91       458.82     1_386.74       0.4669          1.0238            1.0224         9.57
IVF-Binary-1024-nl158-random (self)                      927.91       860.08     1_787.99       0.3645          1.0367            1.0373         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)            803.42       211.00     1_014.42       0.1792          1.1574            1.1483         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)            803.42       208.42     1_011.84       0.1792          1.1574            1.1483         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)            803.42       213.21     1_016.62       0.1792          1.1574            1.1483         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)           803.42       328.82     1_132.24       0.3701          1.0362            1.0363         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)           803.42       458.41     1_261.83       0.4729          1.0231            1.0219         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)           803.42       329.86     1_133.28       0.3701          1.0362            1.0363         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)           803.42       464.45     1_267.87       0.4729          1.0231            1.0219         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)           803.42       352.99     1_156.40       0.3701          1.0362            1.0363         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)           803.42       468.12     1_271.53       0.4729          1.0231            1.0219         9.76
IVF-Binary-1024-nl223-random (self)                      803.42       883.91     1_687.33       0.3699          1.0357            1.0362         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)            837.23       219.22     1_056.45       0.1805          1.1543            1.1451        10.03
IVF-Binary-1024-nl316-np17-rf0-random (query)            837.23       217.33     1_054.56       0.1805          1.1543            1.1451        10.03
IVF-Binary-1024-nl316-np25-rf0-random (query)            837.23       221.14     1_058.37       0.1805          1.1543            1.1451        10.03
IVF-Binary-1024-nl316-np15-rf10-random (query)           837.23       346.91     1_184.14       0.3732          1.0357            1.0359        10.03
IVF-Binary-1024-nl316-np15-rf20-random (query)           837.23       471.86     1_309.09       0.4751          1.0229            1.0217        10.03
IVF-Binary-1024-nl316-np17-rf10-random (query)           837.23       350.27     1_187.50       0.3732          1.0357            1.0359        10.03
IVF-Binary-1024-nl316-np17-rf20-random (query)           837.23       478.52     1_315.75       0.4751          1.0229            1.0217        10.03
IVF-Binary-1024-nl316-np25-rf10-random (query)           837.23       358.77     1_196.00       0.3732          1.0357            1.0359        10.03
IVF-Binary-1024-nl316-np25-rf20-random (query)           837.23       483.33     1_320.55       0.4751          1.0229            1.0217        10.03
IVF-Binary-1024-nl316-random (self)                      837.23       930.47     1_767.70       0.3729          1.0352            1.0358        10.03
IVF-Binary-1024-nl158-np7-rf0-pca (query)              1_142.67       197.82     1_340.49       0.1773          1.1632            1.1536         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)             1_142.67       199.70     1_342.37       0.1773          1.1632            1.1536         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)             1_142.67       204.92     1_347.58       0.1773          1.1632            1.1536         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)             1_142.67       313.48     1_456.14       0.3643          1.0372            1.0375         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)             1_142.67       439.92     1_582.59       0.4640          1.0239            1.0230         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)            1_142.67       316.57     1_459.23       0.3643          1.0372            1.0375         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)            1_142.67       451.40     1_594.06       0.4640          1.0239            1.0230         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)            1_142.67       325.31     1_467.98       0.3643          1.0372            1.0375         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)            1_142.67       456.71     1_599.37       0.4640          1.0239            1.0230         9.57
IVF-Binary-1024-nl158-pca (self)                       1_142.67       873.73     2_016.40       0.3633          1.0373            1.0378         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_001.05       213.01     1_214.06       0.1789          1.1577            1.1482         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_001.05       207.27     1_208.31       0.1789          1.1577            1.1482         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_001.05       212.60     1_213.64       0.1789          1.1577            1.1482         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_001.05       328.72     1_329.76       0.3696          1.0361            1.0363         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_001.05       457.08     1_458.13       0.4707          1.0233            1.0223         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_001.05       330.12     1_331.16       0.3696          1.0361            1.0363         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_001.05       461.20     1_462.25       0.4707          1.0233            1.0223         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_001.05       339.91     1_340.96       0.3696          1.0361            1.0363         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_001.05       466.51     1_467.55       0.4707          1.0233            1.0223         9.76
IVF-Binary-1024-nl223-pca (self)                       1_001.05       896.20     1_897.25       0.3689          1.0362            1.0366         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_041.86       225.25     1_267.11       0.1807          1.1540            1.1446        10.03
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_041.86       219.99     1_261.85       0.1807          1.1540            1.1446        10.03
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_041.86       225.50     1_267.36       0.1807          1.1540            1.1446        10.03
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_041.86       347.77     1_389.63       0.3726          1.0357            1.0359        10.03
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_041.86       475.32     1_517.18       0.4729          1.0230            1.0220        10.03
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_041.86       346.83     1_388.69       0.3726          1.0357            1.0359        10.03
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_041.86       475.66     1_517.52       0.4729          1.0230            1.0220        10.03
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_041.86       356.19     1_398.05       0.3726          1.0357            1.0359        10.03
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_041.86       483.87     1_525.73       0.4729          1.0230            1.0220        10.03
IVF-Binary-1024-nl316-pca (self)                       1_041.86       938.12     1_979.98       0.3717          1.0357            1.0362        10.03
IVF-Binary-768-nl158-np7-rf0-sign (query)                559.30       380.05       939.35       0.1693          1.1870            1.1718         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)               559.30       384.45       943.75       0.1693          1.1870            1.1718         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)               559.30       388.91       948.21       0.1693          1.1870            1.1718         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)               559.30       470.33     1_029.64       0.3461          1.0423            1.0408         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)               559.30       837.67     1_396.98       0.4461          1.0262            1.0247         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)              559.30       471.38     1_030.68       0.3461          1.0423            1.0408         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)              559.30       846.60     1_405.90       0.4461          1.0262            1.0247         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)              559.30       476.15     1_035.45       0.3461          1.0423            1.0408         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)              559.30       855.98     1_415.28       0.4461          1.0262            1.0247         5.04
IVF-Binary-768-nl158-sign (self)                         559.30     1_311.77     1_871.07       0.3467          1.0415            1.0406         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)               410.03       402.34       812.37       0.1694          1.1862            1.1716         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)               410.03       386.65       796.68       0.1694          1.1862            1.1716         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)               410.03       393.58       803.62       0.1694          1.1862            1.1716         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)              410.03       474.83       884.86       0.3484          1.0415            1.0401         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)              410.03       844.04     1_254.07       0.4478          1.0258            1.0245         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)              410.03       472.75       882.78       0.3484          1.0415            1.0401         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)              410.03       859.17     1_269.21       0.4478          1.0258            1.0245         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)              410.03       477.27       887.30       0.3484          1.0415            1.0401         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)              410.03       863.19     1_273.22       0.4478          1.0258            1.0245         5.23
IVF-Binary-768-nl223-sign (self)                         410.03     1_329.05     1_739.08       0.3488          1.0409            1.0401         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)               461.20       398.40       859.60       0.1693          1.1868            1.1718         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)               461.20       400.43       861.63       0.1693          1.1868            1.1718         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)               461.20       403.82       865.02       0.1693          1.1868            1.1718         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)              461.20       485.69       946.88       0.3497          1.0413            1.0399         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)              461.20       854.30     1_315.50       0.4484          1.0258            1.0245         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)              461.20       479.91       941.11       0.3497          1.0413            1.0399         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)              461.20       856.30     1_317.49       0.4484          1.0258            1.0245         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)              461.20       484.61       945.80       0.3497          1.0413            1.0399         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)              461.20       861.73     1_322.93       0.4484          1.0258            1.0245         5.51
IVF-Binary-768-nl316-sign (self)                         461.20     1_358.54     1_819.74       0.3502          1.0406            1.0398         5.51
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 256D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.53       683.48       716.01       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.53     2_279.42     2_311.95       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                 70.40       238.22       308.63       0.0970          1.6334            1.6378         1.78
ExhaustiveBinary-256-random-rf10 (query)                  70.40       344.33       414.74       0.3643          1.1391            1.1302         1.78
ExhaustiveBinary-256-random-rf20 (query)                  70.40       450.58       520.99       0.5087          1.0798            1.0701         1.78
ExhaustiveBinary-256-random (self)                        70.40     1_108.78     1_179.19       0.3862          1.1443            1.1409         1.78
ExhaustiveBinary-256-pca_no_rr (query)                    95.70       237.36       333.07       0.0922          1.6524            1.6606         1.78
ExhaustiveBinary-256-pca-rf10 (query)                     95.70       342.42       438.12       0.3517          1.1465            1.1384         1.78
ExhaustiveBinary-256-pca-rf20 (query)                     95.70       451.13       546.83       0.4943          1.0846            1.0744         1.78
ExhaustiveBinary-256-pca (self)                           95.70     1_107.84     1_203.54       0.3764          1.1502            1.1477         1.78
ExhaustiveBinary-512-random_no_rr (query)                 83.35       351.96       435.30       0.1464          1.5035            1.5085         3.55
ExhaustiveBinary-512-random-rf10 (query)                  83.35       466.44       549.79       0.4596          1.0936            1.0901         3.55
ExhaustiveBinary-512-random-rf20 (query)                  83.35       571.60       654.95       0.6085          1.0504            1.0459         3.55
ExhaustiveBinary-512-random (self)                        83.35     1_517.99     1_601.34       0.4800          1.0996            1.0995         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   107.63       351.25       458.87       0.1458          1.5041            1.5097         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    107.63       464.97       572.59       0.4543          1.0952            1.0911         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    107.63       573.88       681.51       0.6037          1.0513            1.0464         3.55
ExhaustiveBinary-512-pca (self)                          107.63     1_549.27     1_656.89       0.4777          1.1003            1.0997         3.55
ExhaustiveBinary-1024-random_no_rr (query)               112.49       494.16       606.65       0.2155          1.3655            1.3721         7.10
ExhaustiveBinary-1024-random-rf10 (query)                112.49       616.96       729.45       0.5869          1.0540            1.0515         7.10
ExhaustiveBinary-1024-random-rf20 (query)                112.49       722.43       834.93       0.7380          1.0260            1.0224         7.10
ExhaustiveBinary-1024-random (self)                      112.49     2_039.39     2_151.89       0.6118          1.0576            1.0546         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  137.94       498.70       636.64       0.2122          1.3735            1.3798         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   137.94       625.63       763.57       0.5776          1.0560            1.0532         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   137.94       727.41       865.35       0.7291          1.0273            1.0232         7.10
ExhaustiveBinary-1024-pca (self)                         137.94     2_183.43     2_321.37       0.6017          1.0602            1.0571         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   40.89       427.79       468.68       0.1044          1.6421            1.6499         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    40.89       461.72       502.61       0.3737          1.1368            1.1275         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    40.89       690.82       731.71       0.5265          1.0745            1.0646         1.53
ExhaustiveBinary-256-sign (self)                          40.89     1_497.79     1_538.68       0.3940          1.1439            1.1394         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)              235.92        50.96       286.89       0.1070          1.6028            1.6092         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)             235.92        53.91       289.83       0.1070          1.6028            1.6092         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)             235.92        56.62       292.54       0.1070          1.6028            1.6092         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)             235.92       100.68       336.60       0.3847          1.1274            1.1207         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)             235.92       150.28       386.20       0.5298          1.0724            1.0649         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)            235.92       102.15       338.07       0.3844          1.1275            1.1207         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)            235.92       153.81       389.73       0.5294          1.0725            1.0649         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)            235.92       102.06       337.98       0.3844          1.1275            1.1207         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)            235.92       153.91       389.83       0.5294          1.0725            1.0649         1.93
IVF-Binary-256-nl158-random (self)                       235.92       222.03       457.95       0.4061          1.1324            1.1316         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             166.90        44.24       211.14       0.1131          1.5832            1.5861         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             166.90        45.65       212.55       0.1131          1.5833            1.5862         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             166.90        50.56       217.46       0.1131          1.5833            1.5862         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            166.90        96.63       263.53       0.3976          1.1211            1.1139         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            166.90       149.34       316.24       0.5393          1.0695            1.0619         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            166.90        96.70       263.61       0.3975          1.1212            1.1139         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            166.90       149.76       316.66       0.5392          1.0695            1.0619         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            166.90       101.93       268.83       0.3975          1.1212            1.1139         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            166.90       155.73       322.63       0.5391          1.0695            1.0619         2.00
IVF-Binary-256-nl223-random (self)                       166.90       224.39       391.29       0.4179          1.1260            1.1253         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             199.91        47.51       247.42       0.1179          1.5669            1.5683         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             199.91        47.58       247.49       0.1179          1.5669            1.5683         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             199.91        51.71       251.62       0.1179          1.5669            1.5683         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            199.91        99.68       299.59       0.4042          1.1173            1.1107         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            199.91       151.97       351.88       0.5466          1.0673            1.0605         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            199.91        98.27       298.18       0.4041          1.1173            1.1107         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            199.91       151.79       351.70       0.5465          1.0674            1.0605         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            199.91       102.47       302.38       0.4040          1.1173            1.1107         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            199.91       155.96       355.87       0.5464          1.0674            1.0605         2.09
IVF-Binary-256-nl316-random (self)                       199.91       233.92       433.83       0.4258          1.1212            1.1215         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)                 262.68        39.95       302.63       0.1021          1.6200            1.6292         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)                262.68        42.37       305.05       0.1021          1.6201            1.6293         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)                262.68        46.04       308.71       0.1021          1.6201            1.6293         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)                262.68        92.48       355.15       0.3744          1.1333            1.1269         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)                262.68       145.12       407.79       0.5167          1.0765            1.0687         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)               262.68        93.91       356.58       0.3742          1.1334            1.1269         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)               262.68       150.00       412.67       0.5164          1.0766            1.0687         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)               262.68        97.32       359.99       0.3742          1.1334            1.1269         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)               262.68       153.65       416.32       0.5163          1.0766            1.0687         1.93
IVF-Binary-256-nl158-pca (self)                          262.68       216.38       479.06       0.3986          1.1369            1.1369         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                194.35        44.44       238.79       0.1108          1.5929            1.5980         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                194.35        45.55       239.90       0.1108          1.5930            1.5981         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                194.35        51.01       245.35       0.1108          1.5930            1.5981         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               194.35        95.69       290.04       0.3884          1.1259            1.1194         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               194.35       148.74       343.09       0.5286          1.0726            1.0651         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               194.35        96.04       290.39       0.3883          1.1259            1.1194         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               194.35       149.28       343.63       0.5285          1.0726            1.0651         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               194.35       100.75       295.09       0.3883          1.1259            1.1194         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               194.35       154.38       348.73       0.5284          1.0726            1.0651         2.00
IVF-Binary-256-nl223-pca (self)                          194.35       222.44       416.79       0.4110          1.1294            1.1295         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                229.85        46.94       276.79       0.1162          1.5739            1.5756         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                229.85        47.66       277.50       0.1161          1.5740            1.5756         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                229.85        51.24       281.09       0.1161          1.5740            1.5756         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               229.85        98.82       328.66       0.3964          1.1214            1.1153         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               229.85       151.82       381.67       0.5365          1.0702            1.0629         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               229.85        97.94       327.79       0.3963          1.1214            1.1153         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               229.85       150.64       380.49       0.5363          1.0703            1.0629         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               229.85       103.47       333.32       0.3963          1.1214            1.1153         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               229.85       154.80       384.65       0.5363          1.0703            1.0629         2.09
IVF-Binary-256-nl316-pca (self)                          229.85       232.84       462.69       0.4201          1.1244            1.1253         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)              250.90        64.22       315.12       0.1531          1.4876            1.4927         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)             250.90        71.68       322.57       0.1531          1.4876            1.4927         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)             250.90        74.27       325.17       0.1531          1.4876            1.4927         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)             250.90       117.05       367.95       0.4708          1.0894            1.0869         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)             250.90       171.74       422.64       0.6178          1.0482            1.0440         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)            250.90       120.29       371.19       0.4707          1.0894            1.0869         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)            250.90       172.10       423.00       0.6178          1.0482            1.0440         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)            250.90       121.83       372.72       0.4707          1.0894            1.0869         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)            250.90       178.94       429.84       0.6178          1.0482            1.0440         3.71
IVF-Binary-512-nl158-random (self)                       250.90       296.67       547.57       0.4902          1.0957            1.0954         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             194.40        62.13       256.52       0.1575          1.4765            1.4813         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             194.40        63.39       257.79       0.1575          1.4766            1.4813         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             194.40        70.36       264.75       0.1575          1.4766            1.4813         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            194.40       118.26       312.65       0.4765          1.0872            1.0844         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            194.40       170.69       365.09       0.6238          1.0469            1.0430         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            194.40       117.68       312.08       0.4765          1.0872            1.0844         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            194.40       172.10       366.50       0.6238          1.0469            1.0430         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            194.40       124.19       318.58       0.4765          1.0872            1.0844         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            194.40       178.84       373.24       0.6238          1.0469            1.0430         3.77
IVF-Binary-512-nl223-random (self)                       194.40       297.77       492.16       0.4964          1.0933            1.0930         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             210.53        64.86       275.39       0.1599          1.4702            1.4747         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             210.53        66.21       276.74       0.1599          1.4702            1.4747         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             210.53        72.33       282.86       0.1599          1.4702            1.4747         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            210.53       118.69       329.22       0.4803          1.0858            1.0829         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            210.53       171.89       382.43       0.6266          1.0464            1.0422         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            210.53       118.77       329.30       0.4803          1.0858            1.0829         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            210.53       177.33       387.87       0.6266          1.0464            1.0422         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            210.53       123.69       334.22       0.4803          1.0858            1.0829         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            210.53       179.16       389.69       0.6266          1.0464            1.0422         3.86
IVF-Binary-512-nl316-random (self)                       210.53       308.92       519.45       0.4998          1.0920            1.0916         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)                 279.14        59.63       338.76       0.1524          1.4888            1.4941         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)                279.14        61.48       340.62       0.1524          1.4888            1.4941         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)                279.14        69.21       348.35       0.1524          1.4888            1.4941         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)                279.14       113.66       392.79       0.4662          1.0907            1.0872         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)                279.14       168.79       447.92       0.6141          1.0487            1.0443         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)               279.14       116.04       395.18       0.4662          1.0908            1.0872         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)               279.14       174.76       453.90       0.6140          1.0487            1.0443         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)               279.14       121.92       401.05       0.4662          1.0908            1.0872         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)               279.14       178.96       458.09       0.6140          1.0487            1.0443         3.71
IVF-Binary-512-nl158-pca (self)                          279.14       308.02       587.15       0.4883          1.0963            1.0960         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                219.44        61.52       280.97       0.1566          1.4793            1.4840         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                219.44        63.07       282.52       0.1566          1.4793            1.4840         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                219.44        70.41       289.85       0.1566          1.4793            1.4840         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               219.44       116.73       336.17       0.4720          1.0885            1.0852         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               219.44       171.71       391.15       0.6195          1.0476            1.0433         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               219.44       120.12       339.57       0.4720          1.0885            1.0852         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               219.44       173.89       393.33       0.6194          1.0476            1.0433         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               219.44       130.28       349.72       0.4720          1.0885            1.0852         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               219.44       190.76       410.20       0.6194          1.0476            1.0433         3.77
IVF-Binary-512-nl223-pca (self)                          219.44       298.50       517.94       0.4939          1.0941            1.0935         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                232.35        64.93       297.27       0.1590          1.4722            1.4760         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                232.35        65.76       298.11       0.1590          1.4722            1.4760         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                232.35        71.47       303.82       0.1590          1.4722            1.4760         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               232.35       118.96       351.31       0.4756          1.0872            1.0838         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               232.35       172.29       404.64       0.6233          1.0468            1.0424         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               232.35       118.90       351.25       0.4756          1.0872            1.0838         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               232.35       173.58       405.93       0.6233          1.0468            1.0424         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               232.35       123.75       356.10       0.4756          1.0872            1.0838         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               232.35       178.81       411.16       0.6233          1.0468            1.0424         3.86
IVF-Binary-512-nl316-pca (self)                          232.35       303.81       536.16       0.4983          1.0925            1.0920         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)             282.00        90.95       372.95       0.2185          1.3602            1.3674         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)            282.00        94.40       376.40       0.2185          1.3602            1.3674         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)            282.00       102.95       384.95       0.2185          1.3602            1.3674         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)            282.00       148.12       430.12       0.5915          1.0528            1.0505         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)            282.00       202.60       484.59       0.7424          1.0253            1.0217         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)           282.00       160.09       442.09       0.5915          1.0528            1.0505         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)           282.00       245.45       527.45       0.7424          1.0253            1.0217         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)           282.00       160.72       442.72       0.5915          1.0528            1.0505         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)           282.00       220.01       502.01       0.7424          1.0253            1.0217         7.26
IVF-Binary-1024-nl158-random (self)                      282.00       425.80       707.80       0.6168          1.0562            1.0534         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            225.08        92.66       317.74       0.2206          1.3565            1.3634         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            225.08        95.08       320.16       0.2206          1.3565            1.3634         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            225.08       104.12       329.20       0.2206          1.3565            1.3634         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           225.08       150.72       375.80       0.5949          1.0521            1.0498         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           225.08       207.28       432.36       0.7440          1.0252            1.0215         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           225.08       152.22       377.30       0.5949          1.0521            1.0498         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           225.08       210.31       435.39       0.7440          1.0252            1.0215         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           225.08       161.75       386.83       0.5949          1.0521            1.0498         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           225.08       225.78       450.86       0.7440          1.0252            1.0215         7.32
IVF-Binary-1024-nl223-random (self)                      225.08       432.78       657.86       0.6190          1.0557            1.0528         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            251.12        95.74       346.86       0.2221          1.3535            1.3614         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)            251.12        97.33       348.45       0.2221          1.3535            1.3614         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)            251.12       104.25       355.37       0.2221          1.3535            1.3614         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)           251.12       152.76       403.88       0.5967          1.0516            1.0491         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)           251.12       209.12       460.24       0.7457          1.0249            1.0212         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)           251.12       154.34       405.46       0.5967          1.0516            1.0491         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)           251.12       211.91       463.03       0.7457          1.0249            1.0212         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)           251.12       161.63       412.75       0.5967          1.0516            1.0491         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)           251.12       219.43       470.55       0.7457          1.0249            1.0212         7.41
IVF-Binary-1024-nl316-random (self)                      251.12       421.52       672.64       0.6207          1.0553            1.0525         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)                310.75        91.25       402.00       0.2157          1.3673            1.3747         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)               310.75        97.14       407.90       0.2157          1.3673            1.3747         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)               310.75       110.04       420.79       0.2157          1.3673            1.3747         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)               310.75       145.84       456.59       0.5828          1.0547            1.0519         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)               310.75       204.74       515.49       0.7339          1.0265            1.0226         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)              310.75       152.20       462.96       0.5828          1.0547            1.0519         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)              310.75       210.30       521.06       0.7339          1.0265            1.0226         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)              310.75       158.90       469.65       0.5828          1.0547            1.0519         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)              310.75       218.39       529.14       0.7339          1.0265            1.0226         7.26
IVF-Binary-1024-nl158-pca (self)                         310.75       418.51       729.26       0.6069          1.0587            1.0557         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               249.30        93.36       342.66       0.2181          1.3633            1.3703         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               249.30        94.53       343.83       0.2181          1.3633            1.3703         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               249.30       104.16       353.46       0.2181          1.3633            1.3703         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              249.30       151.19       400.49       0.5854          1.0540            1.0509         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              249.30       209.28       458.59       0.7356          1.0263            1.0225         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              249.30       151.78       401.08       0.5854          1.0540            1.0510         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              249.30       216.07       465.37       0.7356          1.0263            1.0225         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              249.30       161.99       411.29       0.5854          1.0540            1.0510         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              249.30       221.27       470.57       0.7356          1.0263            1.0225         7.32
IVF-Binary-1024-nl223-pca (self)                         249.30       417.34       666.65       0.6099          1.0580            1.0548         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               275.51        96.18       371.69       0.2192          1.3606            1.3677         7.41
IVF-Binary-1024-nl316-np17-rf0-pca (query)               275.51        96.89       372.41       0.2192          1.3606            1.3677         7.41
IVF-Binary-1024-nl316-np25-rf0-pca (query)               275.51       104.10       379.61       0.2192          1.3606            1.3677         7.41
IVF-Binary-1024-nl316-np15-rf10-pca (query)              275.51       153.45       428.96       0.5874          1.0535            1.0507         7.41
IVF-Binary-1024-nl316-np15-rf20-pca (query)              275.51       210.08       485.60       0.7376          1.0260            1.0222         7.41
IVF-Binary-1024-nl316-np17-rf10-pca (query)              275.51       154.04       429.55       0.5874          1.0535            1.0507         7.41
IVF-Binary-1024-nl316-np17-rf20-pca (query)              275.51       212.32       487.83       0.7376          1.0260            1.0222         7.41
IVF-Binary-1024-nl316-np25-rf10-pca (query)              275.51       161.53       437.04       0.5874          1.0535            1.0507         7.41
IVF-Binary-1024-nl316-np25-rf20-pca (query)              275.51       220.42       495.93       0.7376          1.0260            1.0222         7.41
IVF-Binary-1024-nl316-pca (self)                         275.51       423.26       698.77       0.6119          1.0574            1.0542         7.41
IVF-Binary-256-nl158-np7-rf0-sign (query)                209.75       132.84       342.59       0.1042          1.6412            1.6488         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               209.75       135.81       345.56       0.1042          1.6413            1.6488         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               209.75       139.87       349.61       0.1042          1.6413            1.6488         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               209.75       169.63       379.37       0.3829          1.1317            1.1232         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               209.75       289.20       498.95       0.5345          1.0721            1.0635         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              209.75       169.38       379.13       0.3823          1.1318            1.1233         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              209.75       293.24       502.99       0.5338          1.0722            1.0635         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              209.75       187.58       397.33       0.3823          1.1318            1.1233         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              209.75       297.01       506.76       0.5338          1.0722            1.0635         1.68
IVF-Binary-256-nl158-sign (self)                         209.75       461.84       671.59       0.4016          1.1396            1.1359         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               157.85       134.31       292.16       0.1045          1.6388            1.6464         1.74
IVF-Binary-256-nl223-np14-rf0-sign (query)               157.85       132.86       290.71       0.1045          1.6389            1.6465         1.74
IVF-Binary-256-nl223-np21-rf0-sign (query)               157.85       137.62       295.47       0.1045          1.6389            1.6465         1.74
IVF-Binary-256-nl223-np11-rf10-sign (query)              157.85       192.52       350.37       0.3881          1.1285            1.1206         1.74
IVF-Binary-256-nl223-np11-rf20-sign (query)              157.85       306.02       463.87       0.5362          1.0716            1.0629         1.74
IVF-Binary-256-nl223-np14-rf10-sign (query)              157.85       168.46       326.31       0.3879          1.1285            1.1206         1.74
IVF-Binary-256-nl223-np14-rf20-sign (query)              157.85       292.97       450.82       0.5360          1.0716            1.0629         1.74
IVF-Binary-256-nl223-np21-rf10-sign (query)              157.85       172.55       330.40       0.3878          1.1285            1.1206         1.74
IVF-Binary-256-nl223-np21-rf20-sign (query)              157.85       298.90       456.75       0.5358          1.0717            1.0630         1.74
IVF-Binary-256-nl223-sign (self)                         157.85       454.18       612.03       0.4072          1.1361            1.1333         1.74
IVF-Binary-256-nl316-np15-rf0-sign (query)               182.64       137.80       320.44       0.1053          1.6356            1.6439         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               182.64       134.01       316.65       0.1052          1.6357            1.6441         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               182.64       138.07       320.71       0.1052          1.6358            1.6441         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              182.64       176.90       359.54       0.3920          1.1263            1.1192         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              182.64       293.34       475.99       0.5384          1.0708            1.0621         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              182.64       171.69       354.34       0.3919          1.1264            1.1192         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              182.64       298.51       481.15       0.5382          1.0708            1.0621         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              182.64       173.95       356.59       0.3918          1.1264            1.1192         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              182.64       349.01       531.65       0.5380          1.0708            1.0622         1.84
IVF-Binary-256-nl316-sign (self)                         182.64       462.32       644.97       0.4115          1.1336            1.1313         1.84
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 512D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        70.36     1_389.31     1_459.67       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         70.36     4_642.75     4_713.11       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                133.55       273.30       406.85       0.0733          1.4884            1.4968         2.03
ExhaustiveBinary-256-random-rf10 (query)                 133.55       393.73       527.28       0.2947          1.1327            1.1260         2.03
ExhaustiveBinary-256-random-rf20 (query)                 133.55       526.35       659.90       0.4174          1.0830            1.0716         2.03
ExhaustiveBinary-256-random (self)                       133.55     1_238.39     1_371.94       0.3150          1.1321            1.1268         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   224.19       265.08       489.27       0.0722          1.4941            1.5001         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    224.19       396.82       621.01       0.2934          1.1324            1.1267         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    224.19       525.84       750.02       0.4181          1.0813            1.0715         2.03
ExhaustiveBinary-256-pca (self)                          224.19     1_252.61     1_476.80       0.3112          1.1338            1.1298         2.03
ExhaustiveBinary-512-random_no_rr (query)                206.00       395.65       601.66       0.1110          1.4064            1.4153         4.05
ExhaustiveBinary-512-random-rf10 (query)                 206.00       549.07       755.07       0.3695          1.0935            1.0919         4.05
ExhaustiveBinary-512-random-rf20 (query)                 206.00       681.49       887.49       0.4981          1.0544            1.0517         4.05
ExhaustiveBinary-512-random (self)                       206.00     1_760.48     1_966.48       0.3856          1.0953            1.0998         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   293.48       393.28       686.77       0.1063          1.4156            1.4272         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    293.48       533.25       826.74       0.3574          1.0990            1.0958         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    293.48       667.57       961.06       0.4853          1.0582            1.0544         4.05
ExhaustiveBinary-512-pca (self)                          293.48     1_717.36     2_010.84       0.3753          1.0995            1.1040         4.05
ExhaustiveBinary-1024-random_no_rr (query)               258.36       596.94       855.29       0.1593          1.3242            1.3318         8.11
ExhaustiveBinary-1024-random-rf10 (query)                258.36       747.25     1_005.61       0.4456          1.0660            1.0678         8.11
ExhaustiveBinary-1024-random-rf20 (query)                258.36       892.44     1_150.79       0.5824          1.0370            1.0362         8.11
ExhaustiveBinary-1024-random (self)                      258.36     2_454.09     2_712.45       0.4595          1.0714            1.0740         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  347.01       592.46       939.47       0.1599          1.3236            1.3333         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   347.01       741.79     1_088.79       0.4446          1.0658            1.0680         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   347.01       884.02     1_231.03       0.5812          1.0370            1.0362         8.11
ExhaustiveBinary-1024-pca (self)                         347.01     2_477.71     2_824.71       0.4582          1.0716            1.0746         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   85.49       657.04       742.53       0.1292          1.3815            1.3877         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    85.49       719.85       805.34       0.3927          1.0844            1.0833         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    85.49     1_086.97     1_172.45       0.5336          1.0464            1.0444         3.05
ExhaustiveBinary-512-sign (self)                          85.49     2_310.76     2_396.25       0.4063          1.0885            1.0916         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)              439.57        77.08       516.65       0.0805          1.4699            1.4811         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)             439.57        77.89       517.46       0.0805          1.4701            1.4812         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)             439.57        84.35       523.92       0.0805          1.4701            1.4812         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)             439.57       159.17       598.74       0.3103          1.1242            1.1175         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)             439.57       245.60       685.17       0.4322          1.0773            1.0678         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)            439.57       156.13       595.70       0.3099          1.1242            1.1175         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)            439.57       244.85       684.42       0.4317          1.0774            1.0678         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)            439.57       158.40       597.97       0.3098          1.1242            1.1175         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)            439.57       247.64       687.22       0.4317          1.0774            1.0678         2.34
IVF-Binary-256-nl158-random (self)                       439.57       308.80       748.37       0.3292          1.1235            1.1211         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)             333.36        71.79       405.15       0.0864          1.4571            1.4649         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)             333.36        73.64       407.00       0.0864          1.4571            1.4649         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)             333.36        76.40       409.76       0.0864          1.4571            1.4649         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)            333.36       158.09       491.45       0.3210          1.1178            1.1110         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)            333.36       251.21       584.57       0.4453          1.0725            1.0641         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)            333.36       158.02       491.38       0.3209          1.1178            1.1110         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)            333.36       252.84       586.20       0.4452          1.0725            1.0641         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)            333.36       161.03       494.39       0.3209          1.1178            1.1110         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)            333.36       253.66       587.02       0.4451          1.0725            1.0641         2.46
IVF-Binary-256-nl223-random (self)                       333.36       334.87       668.23       0.3394          1.1177            1.1162         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)             379.77        84.49       464.26       0.0939          1.4392            1.4433         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)             379.77        78.75       458.52       0.0939          1.4392            1.4433         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)             379.77        82.28       462.06       0.0939          1.4392            1.4433         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)            379.77       169.20       548.97       0.3359          1.1090            1.1039         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)            379.77       256.96       636.73       0.4642          1.0651            1.0594         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)            379.77       173.92       553.70       0.3358          1.1090            1.1039         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)            379.77       257.84       637.61       0.4641          1.0651            1.0594         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)            379.77       172.35       552.12       0.3358          1.1090            1.1039         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)            379.77       263.04       642.81       0.4640          1.0651            1.0594         2.65
IVF-Binary-256-nl316-random (self)                       379.77       348.42       728.19       0.3546          1.1076            1.1101         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)                 523.24        66.38       589.62       0.0796          1.4739            1.4846         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)                523.24        69.20       592.44       0.0796          1.4740            1.4847         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)                523.24        71.12       594.36       0.0796          1.4740            1.4847         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)                523.24       151.91       675.15       0.3104          1.1224            1.1176         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)                523.24       243.48       766.72       0.4353          1.0745            1.0667         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)               523.24       151.30       674.54       0.3100          1.1225            1.1176         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)               523.24       249.41       772.65       0.4345          1.0746            1.0667         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)               523.24       155.04       678.28       0.3100          1.1225            1.1176         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)               523.24       252.08       775.33       0.4345          1.0746            1.0667         2.34
IVF-Binary-256-nl158-pca (self)                          523.24       296.84       820.08       0.3278          1.1228            1.1230         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)                414.03        71.74       485.77       0.0856          1.4582            1.4654         2.46
IVF-Binary-256-nl223-np14-rf0-pca (query)                414.03        73.72       487.75       0.0856          1.4583            1.4654         2.46
IVF-Binary-256-nl223-np21-rf0-pca (query)                414.03        75.68       489.71       0.0856          1.4583            1.4654         2.46
IVF-Binary-256-nl223-np11-rf10-pca (query)               414.03       159.60       573.63       0.3255          1.1136            1.1084         2.46
IVF-Binary-256-nl223-np11-rf20-pca (query)               414.03       252.03       666.06       0.4490          1.0696            1.0626         2.46
IVF-Binary-256-nl223-np14-rf10-pca (query)               414.03       158.44       572.47       0.3253          1.1137            1.1084         2.46
IVF-Binary-256-nl223-np14-rf20-pca (query)               414.03       252.57       666.60       0.4488          1.0696            1.0626         2.46
IVF-Binary-256-nl223-np21-rf10-pca (query)               414.03       161.30       575.34       0.3253          1.1137            1.1084         2.46
IVF-Binary-256-nl223-np21-rf20-pca (query)               414.03       257.68       671.72       0.4488          1.0696            1.0626         2.46
IVF-Binary-256-nl223-pca (self)                          414.03       323.00       737.03       0.3435          1.1121            1.1151         2.46
IVF-Binary-256-nl316-np15-rf0-pca (query)                461.17        78.23       539.40       0.0943          1.4379            1.4397         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)                461.17        81.92       543.09       0.0943          1.4380            1.4397         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)                461.17        82.29       543.46       0.0943          1.4380            1.4397         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)               461.17       166.01       627.17       0.3383          1.1069            1.1014         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)               461.17       257.24       718.41       0.4601          1.0666            1.0601         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)               461.17       163.33       624.50       0.3383          1.1069            1.1014         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)               461.17       257.69       718.86       0.4601          1.0666            1.0601         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)               461.17       168.99       630.15       0.3383          1.1069            1.1014         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)               461.17       262.19       723.35       0.4600          1.0666            1.0601         2.65
IVF-Binary-256-nl316-pca (self)                          461.17       360.52       821.69       0.3564          1.1050            1.1086         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)              506.52       100.34       606.86       0.1161          1.3972            1.4064         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)             506.52       101.17       607.69       0.1161          1.3972            1.4064         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)             506.52       103.47       609.99       0.1161          1.3972            1.4064         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)             506.52       185.81       692.33       0.3762          1.0908            1.0897         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)             506.52       277.38       783.90       0.5045          1.0529            1.0505         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)            506.52       192.21       698.73       0.3761          1.0908            1.0897         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)            506.52       283.79       790.31       0.5043          1.0529            1.0505         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)            506.52       190.81       697.33       0.3761          1.0908            1.0897         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)            506.52       289.67       796.19       0.5043          1.0529            1.0505         4.36
IVF-Binary-512-nl158-random (self)                       506.52       444.08       950.60       0.3916          1.0931            1.0979         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)             452.79       100.49       553.28       0.1199          1.3890            1.3964         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)             452.79       106.22       559.02       0.1199          1.3890            1.3964         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)             452.79       104.76       557.55       0.1199          1.3890            1.3964         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)            452.79       196.10       648.89       0.3824          1.0882            1.0877         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)            452.79       286.66       739.46       0.5105          1.0513            1.0495         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)            452.79       194.00       646.79       0.3824          1.0882            1.0877         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)            452.79       300.05       752.84       0.5104          1.0513            1.0495         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)            452.79       195.00       647.79       0.3824          1.0882            1.0877         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)            452.79       295.40       748.19       0.5104          1.0513            1.0495         4.49
IVF-Binary-512-nl223-random (self)                       452.79       448.83       901.62       0.3975          1.0909            1.0956         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)             469.95       105.43       575.37       0.1243          1.3792            1.3845         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)             469.95       105.94       575.88       0.1243          1.3792            1.3845         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)             469.95       113.66       583.61       0.1243          1.3792            1.3845         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)            469.95       201.65       671.59       0.3890          1.0856            1.0851         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)            469.95       304.05       774.00       0.5169          1.0501            1.0480         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)            469.95       198.60       668.54       0.3890          1.0856            1.0851         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)            469.95       296.04       765.99       0.5168          1.0501            1.0480         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)            469.95       199.28       669.22       0.3890          1.0856            1.0851         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)            469.95       293.68       763.62       0.5168          1.0501            1.0480         4.67
IVF-Binary-512-nl316-random (self)                       469.95       467.64       937.59       0.4038          1.0884            1.0931         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)                 597.70        93.49       691.19       0.1116          1.4053            1.4161         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)                597.70        95.69       693.39       0.1116          1.4053            1.4161         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)                597.70        98.83       696.54       0.1116          1.4053            1.4161         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)                597.70       183.66       781.36       0.3652          1.0954            1.0932         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)                597.70       274.97       872.67       0.4926          1.0560            1.0530         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)               597.70       182.62       780.32       0.3651          1.0955            1.0932         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)               597.70       279.03       876.74       0.4924          1.0561            1.0531         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)               597.70       188.14       785.84       0.3651          1.0955            1.0932         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)               597.70       287.37       885.07       0.4924          1.0561            1.0531         4.36
IVF-Binary-512-nl158-pca (self)                          597.70       426.18     1_023.88       0.3821          1.0965            1.1016         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)                486.75        98.05       584.80       0.1165          1.3954            1.4036         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)                486.75       100.45       587.20       0.1165          1.3954            1.4036         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)                486.75       104.11       590.86       0.1165          1.3954            1.4036         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)               486.75       189.74       676.49       0.3727          1.0920            1.0903         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)               486.75       284.11       770.86       0.5003          1.0542            1.0515         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)               486.75       188.90       675.65       0.3727          1.0920            1.0903         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)               486.75       285.38       772.13       0.5002          1.0542            1.0515         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)               486.75       192.90       679.65       0.3727          1.0920            1.0903         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)               486.75       293.71       780.46       0.5002          1.0542            1.0515         4.49
IVF-Binary-512-nl223-pca (self)                          486.75       443.74       930.49       0.3895          1.0936            1.0987         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)                536.49       105.61       642.10       0.1211          1.3849            1.3913         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)                536.49       110.58       647.07       0.1211          1.3849            1.3913         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)                536.49       109.97       646.46       0.1211          1.3849            1.3913         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)               536.49       196.61       733.10       0.3797          1.0890            1.0878         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)               536.49       293.30       829.79       0.5070          1.0525            1.0502         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)               536.49       194.93       731.42       0.3797          1.0890            1.0878         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)               536.49       291.39       827.88       0.5070          1.0525            1.0502         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)               536.49       197.60       734.09       0.3797          1.0890            1.0878         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)               536.49       291.57       828.06       0.5070          1.0525            1.0502         4.67
IVF-Binary-512-nl316-pca (self)                          536.49       492.80     1_029.29       0.3954          1.0912            1.0960         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)             625.80       142.82       768.62       0.1617          1.3203            1.3290         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)            625.80       148.01       773.81       0.1617          1.3203            1.3290         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)            625.80       154.16       779.95       0.1617          1.3203            1.3290         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)            625.80       236.46       862.26       0.4487          1.0651            1.0669         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)            625.80       333.54       959.34       0.5846          1.0366            1.0358         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)           625.80       241.86       867.66       0.4487          1.0651            1.0669         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)           625.80       342.15       967.95       0.5846          1.0366            1.0358         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)           625.80       250.84       876.64       0.4487          1.0651            1.0669         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)           625.80       357.67       983.47       0.5846          1.0366            1.0358         8.42
IVF-Binary-1024-nl158-random (self)                      625.80       643.66     1_269.46       0.4626          1.0705            1.0733         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)            562.80       150.42       713.22       0.1637          1.3169            1.3260         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)            562.80       159.52       722.32       0.1637          1.3169            1.3260         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)            562.80       159.17       721.96       0.1637          1.3169            1.3260         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)           562.80       244.00       806.80       0.4516          1.0643            1.0661         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)           562.80       345.57       908.37       0.5885          1.0359            1.0353         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)           562.80       246.52       809.32       0.4516          1.0643            1.0661         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)           562.80       347.95       910.75       0.5885          1.0359            1.0353         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)           562.80       251.74       814.54       0.4516          1.0643            1.0661         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)           562.80       361.35       924.15       0.5885          1.0359            1.0353         8.54
IVF-Binary-1024-nl223-random (self)                      562.80       653.46     1_216.26       0.4664          1.0695            1.0722         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)            511.89       159.27       671.16       0.1655          1.3135            1.3227         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)            511.89       157.99       669.88       0.1655          1.3135            1.3227         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)            511.89       171.81       683.70       0.1655          1.3135            1.3227         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)           511.89       250.88       762.77       0.4555          1.0633            1.0652         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)           511.89       353.14       865.04       0.5912          1.0355            1.0347         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)           511.89       250.64       762.54       0.4555          1.0633            1.0652         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)           511.89       359.31       871.20       0.5912          1.0355            1.0347         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)           511.89       259.39       771.28       0.4555          1.0633            1.0652         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)           511.89       363.23       875.12       0.5912          1.0355            1.0347         8.72
IVF-Binary-1024-nl316-random (self)                      511.89       694.82     1_206.71       0.4694          1.0686            1.0714         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)                663.23       149.31       812.54       0.1623          1.3197            1.3298         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)               663.23       154.57       817.80       0.1623          1.3197            1.3298         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)               663.23       153.64       816.87       0.1623          1.3197            1.3298         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)               663.23       239.14       902.37       0.4480          1.0648            1.0672         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)               663.23       339.97     1_003.20       0.5842          1.0364            1.0358         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)              663.23       243.39       906.62       0.4480          1.0648            1.0672         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)              663.23       356.90     1_020.13       0.5842          1.0364            1.0358         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)              663.23       250.13       913.36       0.4480          1.0648            1.0672         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)              663.23       366.86     1_030.09       0.5842          1.0364            1.0358         8.42
IVF-Binary-1024-nl158-pca (self)                         663.23       639.99     1_303.22       0.4613          1.0707            1.0738         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)               545.26       155.15       700.40       0.1642          1.3165            1.3271         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)               545.26       152.20       697.46       0.1642          1.3165            1.3271         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)               545.26       157.12       702.38       0.1642          1.3165            1.3271         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)              545.26       243.78       789.04       0.4509          1.0640            1.0665         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)              545.26       341.58       886.84       0.5863          1.0361            1.0356         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)              545.26       251.44       796.70       0.4509          1.0640            1.0665         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)              545.26       359.47       904.72       0.5863          1.0361            1.0356         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)              545.26       254.83       800.08       0.4509          1.0640            1.0665         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)              545.26       362.29       907.54       0.5863          1.0361            1.0356         8.54
IVF-Binary-1024-nl223-pca (self)                         545.26       657.51     1_202.77       0.4642          1.0699            1.0728         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)               594.37       161.59       755.96       0.1662          1.3128            1.3236         8.72
IVF-Binary-1024-nl316-np17-rf0-pca (query)               594.37       158.32       752.69       0.1662          1.3128            1.3236         8.72
IVF-Binary-1024-nl316-np25-rf0-pca (query)               594.37       164.08       758.45       0.1662          1.3128            1.3236         8.72
IVF-Binary-1024-nl316-np15-rf10-pca (query)              594.37       255.31       849.68       0.4540          1.0631            1.0656         8.72
IVF-Binary-1024-nl316-np15-rf20-pca (query)              594.37       355.72       950.09       0.5899          1.0355            1.0352         8.72
IVF-Binary-1024-nl316-np17-rf10-pca (query)              594.37       256.77       851.15       0.4540          1.0631            1.0656         8.72
IVF-Binary-1024-nl316-np17-rf20-pca (query)              594.37       375.22       969.59       0.5899          1.0355            1.0352         8.72
IVF-Binary-1024-nl316-np25-rf10-pca (query)              594.37       263.19       857.56       0.4540          1.0631            1.0656         8.72
IVF-Binary-1024-nl316-np25-rf20-pca (query)              594.37       365.88       960.25       0.5899          1.0355            1.0352         8.72
IVF-Binary-1024-nl316-pca (self)                         594.37       687.10     1_281.47       0.4672          1.0690            1.0721         8.72
IVF-Binary-512-nl158-np7-rf0-sign (query)                397.05       256.08       653.13       0.1292          1.3817            1.3866         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)               397.05       263.74       660.78       0.1292          1.3817            1.3866         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)               397.05       263.50       660.55       0.1292          1.3817            1.3866         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)               397.05       329.20       726.25       0.3969          1.0827            1.0824         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)               397.05       578.01       975.06       0.5361          1.0458            1.0441         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)              397.05       332.34       729.39       0.3967          1.0828            1.0824         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)              397.05       583.21       980.26       0.5357          1.0459            1.0442         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)              397.05       330.78       727.83       0.3967          1.0828            1.0824         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)              397.05       589.19       986.24       0.5357          1.0459            1.0442         3.36
IVF-Binary-512-nl158-sign (self)                         397.05       886.49     1_283.54       0.4096          1.0874            1.0907         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               290.06       262.55       552.61       0.1294          1.3807            1.3868         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               290.06       275.36       565.42       0.1294          1.3807            1.3868         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               290.06       269.65       559.71       0.1294          1.3807            1.3868         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              290.06       336.63       626.69       0.3980          1.0823            1.0820         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              290.06       599.99       890.05       0.5369          1.0456            1.0440         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              290.06       337.98       628.04       0.3980          1.0823            1.0820         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              290.06       587.94       878.00       0.5369          1.0456            1.0440         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              290.06       335.84       625.90       0.3980          1.0823            1.0820         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              290.06       591.43       881.49       0.5369          1.0456            1.0440         3.49
IVF-Binary-512-nl223-sign (self)                         290.06       904.41     1_194.46       0.4111          1.0869            1.0902         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)               341.98       270.16       612.14       0.1294          1.3806            1.3862         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)               341.98       268.10       610.08       0.1294          1.3806            1.3862         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)               341.98       271.32       613.29       0.1294          1.3806            1.3862         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)              341.98       336.53       678.51       0.4004          1.0812            1.0811         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)              341.98       593.52       935.49       0.5387          1.0452            1.0436         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)              341.98       336.11       678.08       0.4004          1.0812            1.0811         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)              341.98       596.56       938.54       0.5387          1.0452            1.0436         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)              341.98       339.50       681.48       0.4004          1.0812            1.0811         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)              341.98       595.12       937.09       0.5387          1.0452            1.0436         3.67
IVF-Binary-512-nl316-sign (self)                         341.98       935.18     1_277.16       0.4138          1.0859            1.0894         3.67
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 768D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       100.25     1_923.63     2_023.89       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.25     6_465.73     6_565.99       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                192.14       287.39       479.53       0.0662          1.3769            1.3786         2.28
ExhaustiveBinary-256-random-rf10 (query)                 192.14       422.03       614.17       0.2741          1.1112            1.1017         2.28
ExhaustiveBinary-256-random-rf20 (query)                 192.14       580.83       772.97       0.3877          1.0706            1.0590         2.28
ExhaustiveBinary-256-random (self)                       192.14     1_304.82     1_496.96       0.2874          1.1077            1.0994         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   386.41       284.07       670.48       0.0653          1.3793            1.3770         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    386.41       427.10       813.51       0.2701          1.1138            1.1018         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    386.41       573.93       960.34       0.3852          1.0726            1.0585         2.28
ExhaustiveBinary-256-pca (self)                          386.41     1_312.82     1_699.23       0.2820          1.1100            1.0990         2.28
ExhaustiveBinary-512-random_no_rr (query)                300.20       423.24       723.44       0.0934          1.3249            1.3316         4.55
ExhaustiveBinary-512-random-rf10 (query)                 300.20       571.07       871.27       0.3220          1.0860            1.0806         4.55
ExhaustiveBinary-512-random-rf20 (query)                 300.20       731.07     1_031.27       0.4345          1.0527            1.0485         4.55
ExhaustiveBinary-512-random (self)                       300.20     1_825.78     2_125.98       0.3344          1.0826            1.0842         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   490.54       414.29       904.83       0.0951          1.3226            1.3263         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    490.54       566.38     1_056.92       0.3246          1.0847            1.0788         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    490.54       728.64     1_219.18       0.4400          1.0517            1.0468         4.55
ExhaustiveBinary-512-pca (self)                          490.54     1_811.67     2_302.21       0.3361          1.0819            1.0825         4.55
ExhaustiveBinary-1024-random_no_rr (query)               497.22       626.05     1_123.27       0.1319          1.2685            1.2723         9.11
ExhaustiveBinary-1024-random-rf10 (query)                497.22       799.77     1_296.99       0.3742          1.0644            1.0666         9.11
ExhaustiveBinary-1024-random-rf20 (query)                497.22       974.12     1_471.34       0.4906          1.0386            1.0390         9.11
ExhaustiveBinary-1024-random (self)                      497.22     2_634.50     3_131.72       0.3823          1.0666            1.0715         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  689.56       650.63     1_340.18       0.1355          1.2623            1.2651         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   689.56       811.97     1_501.53       0.3804          1.0622            1.0643         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   689.56       980.23     1_669.78       0.4993          1.0369            1.0374         9.11
ExhaustiveBinary-1024-pca (self)                         689.56     2_661.83     3_351.39       0.3870          1.0651            1.0695         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  130.09       858.24       988.33       0.1284          1.2822            1.2821         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   130.09       907.18     1_037.27       0.3618          1.0706            1.0699         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   130.09     1_409.85     1_539.94       0.4847          1.0407            1.0395         4.58
ExhaustiveBinary-768-sign (self)                         130.09     2_964.83     3_094.92       0.3694          1.0716            1.0747         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)              618.27        92.90       711.17       0.0717          1.3654            1.3713         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)             618.27        99.74       718.01       0.0717          1.3654            1.3713         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)             618.27        98.69       716.96       0.0717          1.3654            1.3713         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)             618.27       194.91       813.18       0.2882          1.1032            1.0952         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)             618.27       305.70       923.97       0.4024          1.0646            1.0558         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)            618.27       195.96       814.23       0.2880          1.1033            1.0952         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)            618.27       310.51       928.79       0.4019          1.0646            1.0558         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)            618.27       199.89       818.16       0.2880          1.1033            1.0952         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)            618.27       307.21       925.48       0.4019          1.0646            1.0558         2.74
IVF-Binary-256-nl158-random (self)                       618.27       391.34     1_009.61       0.3017          1.0989            1.0947         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)             470.76        93.10       563.85       0.0767          1.3556            1.3613         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)             470.76        94.15       564.90       0.0766          1.3556            1.3613         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)             470.76        95.38       566.13       0.0766          1.3556            1.3613         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)            470.76       200.03       670.79       0.2997          1.0966            1.0891         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)            470.76       315.64       786.40       0.4133          1.0600            1.0531         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)            470.76       198.43       669.18       0.2997          1.0966            1.0891         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)            470.76       316.86       787.61       0.4132          1.0600            1.0531         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)            470.76       201.07       671.83       0.2997          1.0966            1.0891         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)            470.76       319.34       790.10       0.4132          1.0600            1.0531         2.93
IVF-Binary-256-nl223-random (self)                       470.76       419.30       890.05       0.3130          1.0920            1.0906         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)             535.64       101.45       637.09       0.0842          1.3401            1.3418         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)             535.64       101.40       637.04       0.0842          1.3401            1.3418         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)             535.64       103.38       639.02       0.0842          1.3401            1.3418         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)            535.64       224.71       760.35       0.3158          1.0872            1.0819         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)            535.64       336.10       871.74       0.4294          1.0546            1.0494         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)            535.64       207.49       743.13       0.3157          1.0872            1.0819         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)            535.64       326.70       862.34       0.4293          1.0546            1.0494         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)            535.64       210.23       745.87       0.3157          1.0872            1.0819         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)            535.64       328.70       864.34       0.4293          1.0547            1.0494         3.21
IVF-Binary-256-nl316-random (self)                       535.64       462.34       997.98       0.3296          1.0820            1.0840         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)                 886.28        83.55       969.83       0.0712          1.3661            1.3696         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)                886.28        84.98       971.26       0.0712          1.3661            1.3696         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)                886.28        87.63       973.91       0.0712          1.3661            1.3696         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)                886.28       199.19     1_085.47       0.2836          1.1060            1.0947         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)                886.28       302.22     1_188.50       0.3991          1.0670            1.0556         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)               886.28       189.68     1_075.97       0.2834          1.1060            1.0947         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)               886.28       304.81     1_191.09       0.3987          1.0670            1.0556         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)               886.28       191.41     1_077.70       0.2834          1.1060            1.0947         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)               886.28       308.93     1_195.21       0.3987          1.0670            1.0556         2.74
IVF-Binary-256-nl158-pca (self)                          886.28       388.53     1_274.81       0.2954          1.1018            1.0949         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)                679.05        91.28       770.33       0.0759          1.3561            1.3594         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)                679.05        92.18       771.23       0.0759          1.3561            1.3594         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)                679.05        95.65       774.70       0.0759          1.3561            1.3594         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)               679.05       199.85       878.90       0.2958          1.0975            1.0889         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)               679.05       324.62     1_003.67       0.4111          1.0605            1.0529         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)               679.05       199.26       878.30       0.2958          1.0975            1.0889         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)               679.05       314.60       993.64       0.4109          1.0605            1.0529         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)               679.05       202.37       881.42       0.2958          1.0975            1.0889         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)               679.05       317.60       996.65       0.4109          1.0605            1.0529         2.93
IVF-Binary-256-nl223-pca (self)                          679.05       419.57     1_098.62       0.3086          1.0922            1.0903         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)                723.38       103.46       826.83       0.0841          1.3410            1.3409         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)                723.38       101.44       824.82       0.0841          1.3410            1.3409         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)                723.38       103.63       827.00       0.0841          1.3410            1.3409         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)               723.38       213.99       937.36       0.3099          1.0899            1.0826         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)               723.38       330.20     1_053.58       0.4237          1.0569            1.0501         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)               723.38       211.09       934.47       0.3098          1.0899            1.0826         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)               723.38       324.72     1_048.10       0.4237          1.0569            1.0501         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)               723.38       209.81       933.19       0.3098          1.0899            1.0826         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)               723.38       327.30     1_050.68       0.4236          1.0569            1.0501         3.21
IVF-Binary-256-nl316-pca (self)                          723.38       463.13     1_186.51       0.3216          1.0842            1.0848         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)              737.32       130.09       867.41       0.0979          1.3172            1.3238         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)             737.32       136.61       873.93       0.0979          1.3172            1.3238         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)             737.32       133.14       870.46       0.0979          1.3172            1.3238         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)             737.32       234.93       972.26       0.3289          1.0826            1.0789         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)             737.32       351.47     1_088.79       0.4420          1.0501            1.0472         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)            737.32       235.35       972.67       0.3288          1.0826            1.0789         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)            737.32       357.99     1_095.31       0.4420          1.0501            1.0472         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)            737.32       241.73       979.06       0.3288          1.0826            1.0789         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)            737.32       360.84     1_098.17       0.4420          1.0501            1.0472         5.02
IVF-Binary-512-nl158-random (self)                       737.32       575.08     1_312.40       0.3408          1.0795            1.0827         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)             588.92       132.23       721.15       0.1021          1.3103            1.3128         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)             588.92       138.08       727.00       0.1021          1.3103            1.3128         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)             588.92       139.54       728.45       0.1021          1.3103            1.3128         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)            588.92       244.64       833.56       0.3351          1.0796            1.0772         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)            588.92       363.81       952.72       0.4471          1.0486            1.0462         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)            588.92       243.61       832.52       0.3351          1.0796            1.0772         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)            588.92       368.63       957.54       0.4471          1.0486            1.0462         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)            588.92       246.08       835.00       0.3351          1.0796            1.0772         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)            588.92       367.34       956.26       0.4471          1.0486            1.0462         5.21
IVF-Binary-512-nl223-random (self)                       588.92       591.07     1_179.99       0.3467          1.0771            1.0811         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)             634.53       141.55       776.08       0.1072          1.3004            1.2998         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)             634.53       139.08       773.61       0.1072          1.3004            1.2998         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)             634.53       144.00       778.53       0.1072          1.3004            1.2998         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)            634.53       249.48       884.01       0.3428          1.0761            1.0745         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)            634.53       373.96     1_008.50       0.4550          1.0469            1.0448         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)            634.53       249.62       884.15       0.3428          1.0761            1.0745         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)            634.53       384.45     1_018.98       0.4549          1.0469            1.0448         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)            634.53       256.64       891.18       0.3428          1.0761            1.0745         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)            634.53       378.30     1_012.83       0.4549          1.0469            1.0448         5.48
IVF-Binary-512-nl316-random (self)                       634.53       636.16     1_270.69       0.3533          1.0738            1.0787         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)                 927.78       121.98     1_049.76       0.0991          1.3158            1.3193         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)                927.78       124.22     1_052.01       0.0991          1.3158            1.3193         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)                927.78       128.06     1_055.84       0.0991          1.3158            1.3193         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)                927.78       230.94     1_158.72       0.3315          1.0820            1.0771         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)                927.78       353.21     1_280.99       0.4459          1.0503            1.0460         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)               927.78       242.94     1_170.72       0.3314          1.0820            1.0771         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)               927.78       358.58     1_286.36       0.4458          1.0503            1.0460         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)               927.78       239.18     1_166.96       0.3314          1.0820            1.0771         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)               927.78       359.60     1_287.38       0.4458          1.0503            1.0460         5.02
IVF-Binary-512-nl158-pca (self)                          927.78       557.64     1_485.42       0.3418          1.0797            1.0813         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)                782.41       129.93       912.33       0.1034          1.3085            1.3094         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)                782.41       130.42       912.83       0.1034          1.3085            1.3094         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)                782.41       133.99       916.40       0.1034          1.3085            1.3094         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)               782.41       242.25     1_024.66       0.3378          1.0782            1.0753         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)               782.41       369.20     1_151.60       0.4528          1.0475            1.0447         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)               782.41       241.40     1_023.81       0.3378          1.0782            1.0753         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)               782.41       362.33     1_144.74       0.4528          1.0475            1.0447         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)               782.41       245.17     1_027.58       0.3378          1.0782            1.0753         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)               782.41       365.00     1_147.41       0.4528          1.0475            1.0447         5.21
IVF-Binary-512-nl223-pca (self)                          782.41       588.54     1_370.94       0.3477          1.0767            1.0799         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)                885.59       150.19     1_035.79       0.1085          1.2990            1.2960         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)                885.59       147.60     1_033.19       0.1085          1.2990            1.2960         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)                885.59       142.28     1_027.88       0.1085          1.2990            1.2960         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)               885.59       260.08     1_145.67       0.3440          1.0755            1.0732         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)               885.59       377.01     1_262.60       0.4586          1.0462            1.0438         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)               885.59       249.42     1_135.01       0.3440          1.0755            1.0732         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)               885.59       377.76     1_263.35       0.4586          1.0462            1.0438         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)               885.59       262.33     1_147.93       0.3440          1.0755            1.0732         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)               885.59       388.39     1_273.98       0.4586          1.0462            1.0438         5.48
IVF-Binary-512-nl316-pca (self)                          885.59       639.59     1_525.19       0.3534          1.0743            1.0778         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)             937.06       199.24     1_136.30       0.1345          1.2652            1.2700         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)            937.06       198.11     1_135.16       0.1345          1.2652            1.2700         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)            937.06       203.42     1_140.48       0.1345          1.2652            1.2700         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)            937.06       319.31     1_256.37       0.3775          1.0632            1.0658         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)            937.06       444.45     1_381.50       0.4935          1.0379            1.0386         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)           937.06       324.86     1_261.92       0.3775          1.0632            1.0658         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)           937.06       459.71     1_396.76       0.4935          1.0379            1.0386         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)           937.06       333.04     1_270.10       0.3775          1.0632            1.0658         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)           937.06       467.46     1_404.52       0.4935          1.0379            1.0386         9.57
IVF-Binary-1024-nl158-random (self)                      937.06       861.26     1_798.32       0.3852          1.0657            1.0708         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)            785.05       208.59       993.64       0.1365          1.2617            1.2659         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)            785.05       205.35       990.40       0.1365          1.2617            1.2659         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)            785.05       211.83       996.88       0.1365          1.2617            1.2659         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)           785.05       329.00     1_114.05       0.3808          1.0623            1.0653         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)           785.05       458.55     1_243.60       0.4963          1.0374            1.0381         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)           785.05       331.28     1_116.34       0.3808          1.0623            1.0653         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)           785.05       462.79     1_247.84       0.4963          1.0374            1.0381         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)           785.05       340.56     1_125.62       0.3808          1.0623            1.0653         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)           785.05       472.20     1_257.26       0.4963          1.0374            1.0381         9.76
IVF-Binary-1024-nl223-random (self)                      785.05       891.33     1_676.38       0.3882          1.0648            1.0699         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)            838.63       218.64     1_057.27       0.1394          1.2572            1.2623        10.03
IVF-Binary-1024-nl316-np17-rf0-random (query)            838.63       217.38     1_056.01       0.1394          1.2572            1.2623        10.03
IVF-Binary-1024-nl316-np25-rf0-random (query)            838.63       221.25     1_059.88       0.1394          1.2572            1.2623        10.03
IVF-Binary-1024-nl316-np15-rf10-random (query)           838.63       351.01     1_189.64       0.3841          1.0611            1.0642        10.03
IVF-Binary-1024-nl316-np15-rf20-random (query)           838.63       477.44     1_316.07       0.5009          1.0366            1.0374        10.03
IVF-Binary-1024-nl316-np17-rf10-random (query)           838.63       351.27     1_189.90       0.3841          1.0611            1.0642        10.03
IVF-Binary-1024-nl316-np17-rf20-random (query)           838.63       482.11     1_320.74       0.5009          1.0366            1.0374        10.03
IVF-Binary-1024-nl316-np25-rf10-random (query)           838.63       358.09     1_196.72       0.3841          1.0611            1.0642        10.03
IVF-Binary-1024-nl316-np25-rf20-random (query)           838.63       489.38     1_328.01       0.5009          1.0366            1.0374        10.03
IVF-Binary-1024-nl316-random (self)                      838.63       945.88     1_784.51       0.3916          1.0637            1.0688        10.03
IVF-Binary-1024-nl158-np7-rf0-pca (query)              1_136.32       199.68     1_336.00       0.1376          1.2595            1.2634         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)             1_136.32       201.04     1_337.36       0.1376          1.2595            1.2634         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)             1_136.32       205.63     1_341.95       0.1376          1.2595            1.2634         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)             1_136.32       317.87     1_454.19       0.3834          1.0614            1.0637         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)             1_136.32       453.38     1_589.70       0.5014          1.0366            1.0371         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)            1_136.32       326.47     1_462.79       0.3834          1.0614            1.0637         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)            1_136.32       471.72     1_608.05       0.5014          1.0366            1.0371         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)            1_136.32       339.39     1_475.72       0.3834          1.0614            1.0637         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)            1_136.32       465.09     1_601.42       0.5014          1.0366            1.0371         9.57
IVF-Binary-1024-nl158-pca (self)                       1_136.32       890.62     2_026.95       0.3894          1.0645            1.0690         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)               980.32       208.29     1_188.62       0.1397          1.2562            1.2608         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)               980.32       206.18     1_186.50       0.1397          1.2562            1.2608         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)               980.32       211.31     1_191.64       0.1397          1.2562            1.2608         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)              980.32       334.68     1_315.00       0.3865          1.0602            1.0631         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)              980.32       460.65     1_440.97       0.5043          1.0359            1.0367         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)              980.32       336.74     1_317.07       0.3865          1.0602            1.0631         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)              980.32       465.71     1_446.03       0.5043          1.0359            1.0367         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)              980.32       343.64     1_323.97       0.3865          1.0602            1.0631         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)              980.32       474.68     1_455.01       0.5043          1.0359            1.0367         9.76
IVF-Binary-1024-nl223-pca (self)                         980.32       915.70     1_896.02       0.3926          1.0635            1.0681         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_045.68       220.74     1_266.41       0.1416          1.2529            1.2573        10.03
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_045.68       216.89     1_262.57       0.1416          1.2529            1.2573        10.03
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_045.68       224.11     1_269.79       0.1416          1.2529            1.2573        10.03
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_045.68       352.73     1_398.40       0.3898          1.0592            1.0622        10.03
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_045.68       474.60     1_520.28       0.5084          1.0353            1.0362        10.03
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_045.68       357.75     1_403.42       0.3898          1.0592            1.0622        10.03
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_045.68       489.98     1_535.66       0.5084          1.0353            1.0362        10.03
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_045.68       375.66     1_421.33       0.3898          1.0592            1.0622        10.03
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_045.68       487.70     1_533.37       0.5084          1.0353            1.0362        10.03
IVF-Binary-1024-nl316-pca (self)                       1_045.68       955.97     2_001.64       0.3956          1.0626            1.0672        10.03
IVF-Binary-768-nl158-np7-rf0-sign (query)                560.91       381.06       941.98       0.1284          1.2835            1.2815         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)               560.91       380.74       941.66       0.1284          1.2835            1.2815         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)               560.91       385.63       946.54       0.1284          1.2835            1.2815         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)               560.91       464.12     1_025.03       0.3643          1.0698            1.0695         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)               560.91       838.96     1_399.87       0.4864          1.0404            1.0394         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)              560.91       466.68     1_027.60       0.3642          1.0698            1.0695         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)              560.91       852.52     1_413.44       0.4864          1.0404            1.0394         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)              560.91       471.81     1_032.72       0.3642          1.0698            1.0695         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)              560.91       854.54     1_415.46       0.4864          1.0404            1.0394         5.04
IVF-Binary-768-nl158-sign (self)                         560.91     1_311.88     1_872.79       0.3714          1.0711            1.0744         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)               438.73       384.08       822.80       0.1285          1.2803            1.2810         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)               438.73       387.12       825.85       0.1285          1.2803            1.2810         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)               438.73       391.21       829.93       0.1285          1.2803            1.2810         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)              438.73       472.53       911.26       0.3655          1.0688            1.0690         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)              438.73       845.74     1_284.47       0.4876          1.0399            1.0393         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)              438.73       472.15       910.88       0.3655          1.0688            1.0690         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)              438.73       905.86     1_344.58       0.4876          1.0399            1.0393         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)              438.73       483.53       922.26       0.3655          1.0688            1.0690         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)              438.73       855.23     1_293.96       0.4876          1.0399            1.0393         5.23
IVF-Binary-768-nl223-sign (self)                         438.73     1_332.63     1_771.35       0.3732          1.0703            1.0739         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)               465.39       391.64       857.03       0.1286          1.2804            1.2812         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)               465.39       393.82       859.20       0.1286          1.2804            1.2812         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)               465.39       397.91       863.30       0.1286          1.2804            1.2812         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)              465.39       481.97       947.36       0.3675          1.0684            1.0684         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)              465.39       893.61     1_359.00       0.4883          1.0398            1.0390         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)              465.39       481.59       946.98       0.3675          1.0684            1.0684         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)              465.39       856.52     1_321.91       0.4883          1.0398            1.0390         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)              465.39       486.43       951.82       0.3675          1.0684            1.0684         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)              465.39       863.68     1_329.07       0.4883          1.0398            1.0390         5.51
IVF-Binary-768-nl316-sign (self)                         465.39     1_360.76     1_826.15       0.3745          1.0699            1.0735         5.51
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Cell embeddings

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 256D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.54       691.43       723.97       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.54     2_360.43     2_392.97       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                 69.45       242.10       311.55       0.5519          1.8826            1.5884         1.78
ExhaustiveBinary-256-random-rf10 (query)                  69.45       360.21       429.66       0.9881          1.0022            1.0000         1.78
ExhaustiveBinary-256-random-rf20 (query)                  69.45       470.87       540.32       0.9980          1.0003            1.0000         1.78
ExhaustiveBinary-256-random (self)                        69.45     1_165.68     1_235.13       0.9881          1.0022            1.0000         1.78
ExhaustiveBinary-256-pca_no_rr (query)                    96.05       242.36       338.41       0.5930          1.6081            1.4152         1.78
ExhaustiveBinary-256-pca-rf10 (query)                     96.05       358.37       454.42       0.9919          1.0013            1.0000         1.78
ExhaustiveBinary-256-pca-rf20 (query)                     96.05       470.45       566.50       0.9988          1.0001            1.0000         1.78
ExhaustiveBinary-256-pca (self)                           96.05     1_218.83     1_314.88       0.9915          1.0014            1.0000         1.78
ExhaustiveBinary-512-random_no_rr (query)                 89.59       463.85       553.44       0.6306          1.5767            1.3633         3.55
ExhaustiveBinary-512-random-rf10 (query)                  89.59       497.51       587.10       0.9975          1.0004            1.0000         3.55
ExhaustiveBinary-512-random-rf20 (query)                  89.59       609.35       698.94       0.9998          1.0000            1.0000         3.55
ExhaustiveBinary-512-random (self)                        89.59     1_573.20     1_662.79       0.9973          1.0004            1.0000         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   107.84       357.24       465.07       0.6479          1.4884            1.3147         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    107.84       479.39       587.23       0.9983          1.0002            1.0000         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    107.84       629.11       736.95       0.9998          1.0000            1.0000         3.55
ExhaustiveBinary-512-pca (self)                          107.84     1_556.76     1_664.59       0.9981          1.0002            1.0000         3.55
ExhaustiveBinary-1024-random_no_rr (query)               113.53       509.01       622.53       0.6758          1.4452            1.2804         7.10
ExhaustiveBinary-1024-random-rf10 (query)                113.53       635.13       748.65       0.9995          1.0001            1.0000         7.10
ExhaustiveBinary-1024-random-rf20 (query)                113.53       757.27       870.80       0.9999          1.0000            1.0000         7.10
ExhaustiveBinary-1024-random (self)                      113.53     2_106.61     2_220.13       0.9993          1.0001            1.0000         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  138.20       519.93       658.12       0.6838          1.4142            1.2651         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   138.20       641.34       779.54       0.9996          1.0000            1.0000         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   138.20       753.96       892.16       1.0000          1.0000            1.0000         7.10
ExhaustiveBinary-1024-pca (self)                         138.20     2_110.10     2_248.29       0.9995          1.0001            1.0000         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   54.63       410.35       464.98       0.0376         19.4734           14.8778         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    54.63       440.84       495.47       0.1617          2.7567            2.6548         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    54.63       662.12       716.76       0.2739          1.9837            1.9249         1.53
ExhaustiveBinary-256-sign (self)                          54.63     1_472.38     1_527.01       0.1691          2.7353            2.6299         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)              238.94        56.66       295.60       0.5659          1.6650            1.5101         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)             238.94        63.96       302.90       0.5593          1.7235            1.5473         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)             238.94        75.45       314.39       0.5573          1.7568            1.5619         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)             238.94       114.17       353.11       0.9900          1.0019            1.0000         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)             238.94       165.38       404.32       0.9963          1.0008            1.0000         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)            238.94       124.46       363.40       0.9906          1.0017            1.0000         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)            238.94       173.65       412.59       0.9986          1.0002            1.0000         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)            238.94       126.28       365.22       0.9897          1.0018            1.0000         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)            238.94       183.32       422.26       0.9984          1.0002            1.0000         1.93
IVF-Binary-256-nl158-random (self)                       238.94       313.20       552.13       0.9904          1.0017            1.0000         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             168.41        49.69       218.10       0.5628          1.6781            1.5260         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             168.41        54.57       222.98       0.5603          1.7037            1.5421         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             168.41        61.04       229.45       0.5576          1.7448            1.5600         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            168.41       109.63       278.04       0.9913          1.0015            1.0000         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            168.41       163.02       331.43       0.9984          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            168.41       113.53       281.94       0.9910          1.0016            1.0000         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            168.41       168.28       336.69       0.9986          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            168.41       120.84       289.25       0.9900          1.0017            1.0000         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            168.41       176.90       345.30       0.9985          1.0002            1.0000         2.00
IVF-Binary-256-nl223-random (self)                       168.41       289.89       458.30       0.9907          1.0016            1.0000         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             193.87        54.36       248.23       0.5631          1.6756            1.5249         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             193.87        53.56       247.43       0.5617          1.6898            1.5331         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             193.87        59.91       253.79       0.5587          1.7346            1.5511         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            193.87       110.50       304.37       0.9916          1.0014            1.0000         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            193.87       162.92       356.79       0.9987          1.0002            1.0000         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            193.87       111.50       305.37       0.9914          1.0014            1.0000         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            193.87       165.66       359.53       0.9988          1.0002            1.0000         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            193.87       118.75       312.62       0.9904          1.0017            1.0000         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            193.87       173.16       367.04       0.9986          1.0002            1.0000         2.09
IVF-Binary-256-nl316-random (self)                       193.87       281.51       475.38       0.9912          1.0015            1.0000         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)                 261.10        45.07       306.16       0.6041          1.4864            1.3741         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)                261.10        53.51       314.61       0.5993          1.5196            1.3915         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)                261.10        61.50       322.60       0.5977          1.5391            1.3979         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)                261.10       107.25       368.35       0.9923          1.0014            1.0000         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)                261.10       159.85       420.95       0.9966          1.0007            1.0000         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)               261.10       115.69       376.79       0.9933          1.0010            1.0000         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)               261.10       174.61       435.71       0.9990          1.0001            1.0000         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)               261.10       123.16       384.25       0.9928          1.0011            1.0000         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)               261.10       180.79       441.88       0.9990          1.0001            1.0000         1.93
IVF-Binary-256-nl158-pca (self)                          261.10       305.67       566.77       0.9929          1.0011            1.0000         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                204.91        50.76       255.67       0.6022          1.4952            1.3803         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                204.91        52.86       257.76       0.6001          1.5103            1.3880         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                204.91        61.53       266.43       0.5981          1.5331            1.3963         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               204.91       108.43       313.34       0.9935          1.0010            1.0000         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               204.91       161.99       366.90       0.9987          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               204.91       111.73       316.63       0.9933          1.0010            1.0000         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               204.91       165.68       370.58       0.9990          1.0001            1.0000         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               204.91       119.46       324.37       0.9928          1.0011            1.0000         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               204.91       187.59       392.50       0.9990          1.0001            1.0000         2.00
IVF-Binary-256-nl223-pca (self)                          204.91       288.01       492.92       0.9930          1.0011            1.0000         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                227.53        51.64       279.18       0.6016          1.4963            1.3821         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                227.53        61.93       289.46       0.6007          1.5047            1.3864         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                227.53        59.36       286.89       0.5987          1.5267            1.3946         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               227.53       109.33       336.86       0.9938          1.0009            1.0000         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               227.53       162.42       389.96       0.9989          1.0001            1.0000         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               227.53       116.14       343.67       0.9937          1.0010            1.0000         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               227.53       164.14       391.67       0.9991          1.0001            1.0000         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               227.53       121.32       348.85       0.9931          1.0011            1.0000         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               227.53       172.04       399.57       0.9991          1.0001            1.0000         2.09
IVF-Binary-256-nl316-pca (self)                          227.53       279.08       506.61       0.9933          1.0011            1.0000         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)              253.54        69.86       323.40       0.6413          1.4450            1.3286         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)             253.54        82.40       335.94       0.6358          1.4838            1.3485         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)             253.54        94.84       348.39       0.6339          1.5064            1.3542         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)             253.54       128.77       382.31       0.9959          1.0008            1.0000         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)             253.54       184.08       437.62       0.9972          1.0007            1.0000         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)            253.54       142.02       395.56       0.9980          1.0003            1.0000         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)            253.54       197.79       451.33       0.9998          1.0000            1.0000         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)            253.54       157.50       411.04       0.9978          1.0003            1.0000         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)            253.54       209.63       463.17       0.9998          1.0000            1.0000         3.71
IVF-Binary-512-nl158-random (self)                       253.54       387.17       640.71       0.9978          1.0003            1.0000         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             199.11        68.44       267.55       0.6386          1.4543            1.3368         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             199.11        72.76       271.87       0.6363          1.4730            1.3448         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             199.11        84.38       283.48       0.6339          1.5008            1.3535         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            199.11       128.46       327.57       0.9978          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            199.11       185.24       384.35       0.9993          1.0001            1.0000         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            199.11       132.92       332.03       0.9980          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            199.11       189.03       388.14       0.9997          1.0000            1.0000         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            199.11       144.88       343.99       0.9979          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            199.11       201.75       400.86       0.9998          1.0000            1.0000         3.77
IVF-Binary-512-nl223-random (self)                       199.11       358.44       557.54       0.9978          1.0003            1.0000         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             211.74        70.36       282.10       0.6384          1.4561            1.3391         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             211.74        72.06       283.80       0.6372          1.4667            1.3435         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             211.74        81.82       293.56       0.6344          1.4971            1.3518         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            211.74       129.21       340.95       0.9980          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            211.74       187.58       399.33       0.9995          1.0001            1.0000         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            211.74       129.79       341.53       0.9981          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            211.74       189.26       401.01       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            211.74       144.30       356.04       0.9980          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            211.74       196.79       408.54       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-random (self)                       211.74       376.45       588.19       0.9980          1.0003            1.0000         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)                 278.86        63.89       342.75       0.6577          1.3876            1.2868         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)                278.86        75.93       354.79       0.6525          1.4199            1.3023         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)                278.86        88.81       367.67       0.6509          1.4373            1.3074         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)                278.86       126.57       405.43       0.9963          1.0008            1.0000         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)                278.86       180.16       459.02       0.9971          1.0007            1.0000         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)               278.86       139.56       418.42       0.9986          1.0002            1.0000         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)               278.86       197.02       475.88       0.9998          1.0000            1.0000         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)               278.86       151.27       430.13       0.9984          1.0002            1.0000         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)               278.86       213.90       492.76       0.9999          1.0000            1.0000         3.71
IVF-Binary-512-nl158-pca (self)                          278.86       386.38       665.24       0.9985          1.0002            1.0000         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                220.13        68.56       288.70       0.6553          1.3966            1.2913         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                220.13        72.87       293.00       0.6532          1.4106            1.2977         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                220.13        84.61       304.74       0.6513          1.4322            1.3056         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               220.13       130.15       350.28       0.9983          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               220.13       182.02       402.15       0.9994          1.0001            1.0000         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               220.13       132.67       352.80       0.9985          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               220.13       189.46       409.60       0.9998          1.0000            1.0000         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               220.13       145.90       366.04       0.9984          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               220.13       203.42       423.55       0.9999          1.0000            1.0000         3.77
IVF-Binary-512-nl223-pca (self)                          220.13       368.91       589.04       0.9985          1.0002            1.0000         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                242.55        69.84       312.40       0.6548          1.3991            1.2940         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                242.55        72.80       315.35       0.6538          1.4070            1.2973         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                242.55        84.95       327.51       0.6515          1.4289            1.3052         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               242.55       129.03       371.59       0.9985          1.0002            1.0000         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               242.55       183.73       426.28       0.9995          1.0001            1.0000         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               242.55       130.17       372.73       0.9986          1.0002            1.0000         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               242.55       187.25       429.81       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               242.55       148.10       390.65       0.9985          1.0002            1.0000         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               242.55       197.55       440.10       0.9999          1.0000            1.0000         3.86
IVF-Binary-512-nl316-pca (self)                          242.55       347.35       589.91       0.9986          1.0002            1.0000         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)             285.55        96.12       381.67       0.6850          1.3503            1.2541         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)            285.55       130.37       415.92       0.6800          1.3813            1.2699         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)            285.55       139.88       425.43       0.6782          1.3988            1.2746         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)            285.55       178.51       464.07       0.9970          1.0007            1.0000         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)            285.55       242.74       528.30       0.9972          1.0007            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)           285.55       201.60       487.15       0.9996          1.0001            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)           285.55       276.59       562.14       0.9999          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)           285.55       223.49       509.04       0.9996          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)           285.55       292.44       577.99       0.9999          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-random (self)                      285.55       585.27       870.82       0.9994          1.0001            1.0000         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            245.78       110.39       356.17       0.6827          1.3604            1.2598         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            245.78       121.08       366.86       0.6806          1.3751            1.2657         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            245.78       141.27       387.04       0.6785          1.3958            1.2733         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           245.78       181.59       427.37       0.9991          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           245.78       247.12       492.90       0.9994          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           245.78       192.11       437.89       0.9995          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           245.78       252.84       498.61       0.9998          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           245.78       192.85       438.63       0.9996          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           245.78       250.20       495.98       0.9999          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-random (self)                      245.78       494.58       740.36       0.9994          1.0001            1.0000         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            251.75       103.13       354.88       0.6820          1.3626            1.2613         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)            251.75       106.42       358.17       0.6810          1.3706            1.2646         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)            251.75       120.58       372.33       0.6786          1.3927            1.2726         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)           251.75       168.48       420.23       0.9993          1.0001            1.0000         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)           251.75       223.77       475.52       0.9996          1.0001            1.0000         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)           251.75       169.70       421.45       0.9995          1.0001            1.0000         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)           251.75       249.69       501.44       0.9998          1.0000            1.0000         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)           251.75       200.73       452.48       0.9996          1.0000            1.0000         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)           251.75       284.77       536.52       0.9999          1.0000            1.0000         7.41
IVF-Binary-1024-nl316-random (self)                      251.75       528.36       780.11       0.9994          1.0001            1.0000         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)                479.09       121.57       600.65       0.6928          1.3307            1.2401         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)               479.09       142.72       621.81       0.6877          1.3588            1.2547         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)               479.09       162.26       641.35       0.6861          1.3745            1.2595         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)               479.09       180.57       659.66       0.9971          1.0007            1.0000         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)               479.09       244.40       723.49       0.9972          1.0007            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)              479.09       201.12       680.21       0.9996          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)              479.09       268.59       747.68       0.9999          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)              479.09       221.22       700.30       0.9996          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)              479.09       295.37       774.46       1.0000          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-pca (self)                         479.09       591.12     1_070.21       0.9996          1.0001            1.0000         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               275.90       114.73       390.63       0.6901          1.3419            1.2466         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               275.90       126.23       402.12       0.6881          1.3546            1.2524         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               275.90       145.68       421.57       0.6861          1.3743            1.2579         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              275.90       168.17       444.07       0.9992          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              275.90       222.98       498.88       0.9994          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              275.90       175.42       451.32       0.9996          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              275.90       268.11       544.01       0.9998          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              275.90       197.01       472.90       0.9996          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              275.90       248.73       524.63       1.0000          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-pca (self)                         275.90       502.94       778.84       0.9995          1.0001            1.0000         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               309.84       112.96       422.80       0.6896          1.3424            1.2461         7.41
IVF-Binary-1024-nl316-np17-rf0-pca (query)               309.84       115.61       425.45       0.6887          1.3491            1.2492         7.41
IVF-Binary-1024-nl316-np25-rf0-pca (query)               309.84       129.61       439.45       0.6866          1.3683            1.2569         7.41
IVF-Binary-1024-nl316-np15-rf10-pca (query)              309.84       183.36       493.20       0.9994          1.0001            1.0000         7.41
IVF-Binary-1024-nl316-np15-rf20-pca (query)              309.84       246.19       556.03       0.9996          1.0001            1.0000         7.41
IVF-Binary-1024-nl316-np17-rf10-pca (query)              309.84       185.83       495.67       0.9996          1.0001            1.0000         7.41
IVF-Binary-1024-nl316-np17-rf20-pca (query)              309.84       255.95       565.79       0.9998          1.0000            1.0000         7.41
IVF-Binary-1024-nl316-np25-rf10-pca (query)              309.84       209.08       518.92       0.9997          1.0000            1.0000         7.41
IVF-Binary-1024-nl316-np25-rf20-pca (query)              309.84       277.36       587.20       1.0000          1.0000            1.0000         7.41
IVF-Binary-1024-nl316-pca (self)                         309.84       532.81       842.65       0.9995          1.0001            1.0000         7.41
IVF-Binary-256-nl158-np7-rf0-sign (query)                225.97       162.91       388.88       0.0702          6.5012            5.9593         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               225.97       179.08       405.05       0.0569          7.6990            7.0407         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               225.97       193.55       419.52       0.0516          8.5681            7.7948         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               225.97       202.32       428.29       0.4086          1.5860            1.5019         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               225.97       327.14       553.11       0.6588          1.2254            1.1711         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              225.97       202.40       428.38       0.3190          1.8128            1.7163         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              225.97       351.61       577.58       0.4987          1.4165            1.3496         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              225.97       231.83       457.81       0.2839          1.9440            1.8465         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              225.97       368.98       594.96       0.4258          1.5383            1.4599         1.68
IVF-Binary-256-nl158-sign (self)                         225.97       609.21       835.19       0.3247          1.8006            1.7051         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               154.92       151.25       306.17       0.0657          6.5478            6.0793         1.74
IVF-Binary-256-nl223-np14-rf0-sign (query)               154.92       157.03       311.95       0.0605          7.0103            6.4875         1.74
IVF-Binary-256-nl223-np21-rf0-sign (query)               154.92       167.49       322.41       0.0536          7.9113            7.3016         1.74
IVF-Binary-256-nl223-np11-rf10-sign (query)              154.92       187.00       341.92       0.3689          1.6592            1.5777         1.74
IVF-Binary-256-nl223-np11-rf20-sign (query)              154.92       322.91       477.83       0.5986          1.2821            1.2287         1.74
IVF-Binary-256-nl223-np14-rf10-sign (query)              154.92       192.44       347.35       0.3351          1.7485            1.6627         1.74
IVF-Binary-256-nl223-np14-rf20-sign (query)              154.92       333.49       488.41       0.5338          1.3563            1.2976         1.74
IVF-Binary-256-nl223-np21-rf10-sign (query)              154.92       203.17       358.09       0.2928          1.8998            1.8082         1.74
IVF-Binary-256-nl223-np21-rf20-sign (query)              154.92       351.65       506.56       0.4453          1.4917            1.4259         1.74
IVF-Binary-256-nl223-sign (self)                         154.92       557.58       712.49       0.3404          1.7398            1.6540         1.74
IVF-Binary-256-nl316-np15-rf0-sign (query)               344.01       170.25       514.26       0.0654          6.5521            6.1037         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               344.01       155.19       499.20       0.0628          6.7792            6.3025         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               344.01       163.56       507.56       0.0562          7.5430            6.9699         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              344.01       192.99       536.99       0.3629          1.6757            1.5919         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              344.01       328.46       672.47       0.5856          1.2996            1.2410         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              344.01       193.14       537.15       0.3461          1.7205            1.6321         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              344.01       339.91       683.92       0.5530          1.3369            1.2786         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              344.01       205.77       549.78       0.3042          1.8584            1.7663         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              344.01       351.49       695.49       0.4668          1.4566            1.3918         1.84
IVF-Binary-256-nl316-sign (self)                         344.01       549.70       893.70       0.3516          1.7088            1.6228         1.84
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 512D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.45     1_392.16     1_460.61       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.45     4_649.36     4_717.81       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                133.86       268.52       402.38       0.5547          1.7646            1.5366         2.03
ExhaustiveBinary-256-random-rf10 (query)                 133.86       416.02       549.88       0.9898          1.0017            1.0000         2.03
ExhaustiveBinary-256-random-rf20 (query)                 133.86       557.36       691.22       0.9985          1.0002            1.0000         2.03
ExhaustiveBinary-256-random (self)                       133.86     1_311.81     1_445.67       0.9899          1.0016            1.0000         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   222.03       268.89       490.92       0.5767          1.6243            1.4311         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    222.03       422.43       644.46       0.9904          1.0016            1.0000         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    222.03       571.59       793.62       0.9984          1.0002            1.0000         2.03
ExhaustiveBinary-256-pca (self)                          222.03     1_317.05     1_539.08       0.9905          1.0016            1.0000         2.03
ExhaustiveBinary-512-random_no_rr (query)                214.88       403.08       617.96       0.6013          1.6760            1.4608         4.05
ExhaustiveBinary-512-random-rf10 (query)                 214.88       544.06       758.94       0.9977          1.0003            1.0000         4.05
ExhaustiveBinary-512-random-rf20 (query)                 214.88       690.36       905.24       0.9998          1.0000            1.0000         4.05
ExhaustiveBinary-512-random (self)                       214.88     1_766.48     1_981.36       0.9975          1.0003            1.0000         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   297.56       396.57       694.13       0.6443          1.4426            1.3064         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    297.56       563.18       860.73       0.9985          1.0002            1.0000         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    297.56       690.80       988.36       0.9999          1.0000            1.0000         4.05
ExhaustiveBinary-512-pca (self)                          297.56     1_774.81     2_072.36       0.9984          1.0002            1.0000         4.05
ExhaustiveBinary-1024-random_no_rr (query)               256.35       587.48       843.83       0.6624          1.4553            1.3048         8.11
ExhaustiveBinary-1024-random-rf10 (query)                256.35       745.77     1_002.11       0.9995          1.0001            1.0000         8.11
ExhaustiveBinary-1024-random-rf20 (query)                256.35       897.42     1_153.77       1.0000          1.0000            1.0000         8.11
ExhaustiveBinary-1024-random (self)                      256.35     2_465.61     2_721.96       0.9994          1.0001            1.0000         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  346.96       592.97       939.93       0.6865          1.3603            1.2383         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   346.96       769.86     1_116.82       0.9996          1.0001            1.0000         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   346.96       915.54     1_262.50       1.0000          1.0000            1.0000         8.11
ExhaustiveBinary-1024-pca (self)                         346.96     2_504.50     2_851.46       0.9996          1.0001            1.0000         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   91.06       661.11       752.17       0.0400         18.1511           13.6734         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    91.06       706.19       797.25       0.1821          2.5573            2.4620         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    91.06     1_066.48     1_157.55       0.3140          1.8429            1.7786         3.05
ExhaustiveBinary-512-sign (self)                          91.06     2_247.91     2_338.97       0.1897          2.5286            2.4283         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)              439.54        81.83       521.37       0.5634          1.6373            1.4869         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)             439.54        91.73       531.27       0.5600          1.6733            1.5065         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)             439.54       102.59       542.12       0.5587          1.6948            1.5123         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)             439.54       178.52       618.05       0.9915          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)             439.54       262.46       702.00       0.9978          1.0004            1.0000         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)            439.54       176.74       616.28       0.9913          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)            439.54       273.90       713.44       0.9988          1.0001            1.0000         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)            439.54       184.12       623.65       0.9906          1.0015            1.0000         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)            439.54       280.98       720.52       0.9987          1.0002            1.0000         2.34
IVF-Binary-256-nl158-random (self)                       439.54       398.41       837.95       0.9915          1.0013            1.0000         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)             355.18        78.64       433.82       0.5614          1.6502            1.4940         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)             355.18        83.27       438.45       0.5601          1.6630            1.5026         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)             355.18        92.06       447.24       0.5587          1.6842            1.5117         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)            355.18       171.89       527.07       0.9922          1.0012            1.0000         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)            355.18       278.19       633.37       0.9988          1.0002            1.0000         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)            355.18       177.00       532.19       0.9918          1.0013            1.0000         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)            355.18       271.56       626.74       0.9989          1.0001            1.0000         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)            355.18       184.44       539.62       0.9910          1.0014            1.0000         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)            355.18       279.95       635.13       0.9988          1.0002            1.0000         2.46
IVF-Binary-256-nl223-random (self)                       355.18       376.51       731.69       0.9920          1.0012            1.0000         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)             379.41        83.61       463.02       0.5614          1.6495            1.4999         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)             379.41        85.22       464.63       0.5610          1.6549            1.5018         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)             379.41        91.11       470.52       0.5597          1.6741            1.5100         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)            379.41       172.09       551.51       0.9924          1.0011            1.0000         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)            379.41       267.59       647.00       0.9991          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)            379.41       175.43       554.84       0.9922          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)            379.41       272.22       651.63       0.9990          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)            379.41       180.98       560.39       0.9914          1.0013            1.0000         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)            379.41       282.23       661.64       0.9988          1.0001            1.0000         2.65
IVF-Binary-256-nl316-random (self)                       379.41       376.88       756.29       0.9922          1.0012            1.0000         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)                 518.50        71.73       590.23       0.5836          1.5300            1.4064         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)                518.50        80.51       599.01       0.5809          1.5561            1.4172         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)                518.50        88.57       607.07       0.5800          1.5725            1.4204         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)                518.50       173.04       691.53       0.9915          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)                518.50       258.03       776.52       0.9977          1.0004            1.0000         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)               518.50       172.20       690.70       0.9915          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)               518.50       276.99       795.48       0.9987          1.0002            1.0000         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)               518.50       180.73       699.23       0.9911          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)               518.50       275.16       793.66       0.9986          1.0002            1.0000         2.34
IVF-Binary-256-nl158-pca (self)                          518.50       393.72       912.22       0.9916          1.0014            1.0000         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)                416.76        77.51       494.28       0.5829          1.5345            1.4098         2.46
IVF-Binary-256-nl223-np14-rf0-pca (query)                416.76        81.37       498.13       0.5819          1.5433            1.4140         2.46
IVF-Binary-256-nl223-np21-rf0-pca (query)                416.76        88.95       505.71       0.5807          1.5577            1.4197         2.46
IVF-Binary-256-nl223-np11-rf10-pca (query)               416.76       168.78       585.54       0.9924          1.0012            1.0000         2.46
IVF-Binary-256-nl223-np11-rf20-pca (query)               416.76       261.40       678.16       0.9988          1.0002            1.0000         2.46
IVF-Binary-256-nl223-np14-rf10-pca (query)               416.76       171.51       588.27       0.9919          1.0013            1.0000         2.46
IVF-Binary-256-nl223-np14-rf20-pca (query)               416.76       278.55       695.31       0.9989          1.0001            1.0000         2.46
IVF-Binary-256-nl223-np21-rf10-pca (query)               416.76       186.73       603.49       0.9913          1.0014            1.0000         2.46
IVF-Binary-256-nl223-np21-rf20-pca (query)               416.76       284.61       701.37       0.9987          1.0002            1.0000         2.46
IVF-Binary-256-nl223-pca (self)                          416.76       382.20       798.97       0.9921          1.0013            1.0000         2.46
IVF-Binary-256-nl316-np15-rf0-pca (query)                512.30        84.70       596.99       0.5828          1.5372            1.4101         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)                512.30        85.97       598.27       0.5823          1.5411            1.4120         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)                512.30        92.44       604.74       0.5813          1.5534            1.4174         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)               512.30       176.46       688.76       0.9923          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)               512.30       291.07       803.37       0.9989          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)               512.30       175.02       687.32       0.9922          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)               512.30       274.99       787.28       0.9989          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)               512.30       187.94       700.24       0.9915          1.0013            1.0000         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)               512.30       277.67       789.96       0.9988          1.0002            1.0000         2.65
IVF-Binary-256-nl316-pca (self)                          512.30       386.73       899.03       0.9923          1.0012            1.0000         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)              537.86       105.93       643.79       0.6086          1.5700            1.4226         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)             537.86       119.64       657.50       0.6048          1.6057            1.4407         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)             537.86       128.46       666.32       0.6035          1.6257            1.4476         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)             537.86       205.44       743.29       0.9971          1.0004            1.0000         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)             537.86       288.75       826.61       0.9985          1.0003            1.0000         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)            537.86       208.15       746.00       0.9981          1.0002            1.0000         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)            537.86       304.81       842.67       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)            537.86       219.98       757.84       0.9979          1.0003            1.0000         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)            537.86       318.72       856.57       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-random (self)                       537.86       530.63     1_068.49       0.9980          1.0003            1.0000         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)             399.70       106.95       506.65       0.6070          1.5829            1.4302         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)             399.70       111.03       510.73       0.6056          1.5947            1.4377         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)             399.70       122.13       521.83       0.6042          1.6130            1.4461         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)            399.70       198.14       597.83       0.9982          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)            399.70       291.98       691.67       0.9996          1.0001            1.0000         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)            399.70       201.62       601.32       0.9983          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)            399.70       299.35       699.05       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)            399.70       213.52       613.21       0.9980          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)            399.70       325.14       724.84       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-random (self)                       399.70       522.03       921.72       0.9982          1.0002            1.0000         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)             501.69       112.08       613.77       0.6066          1.5816            1.4306         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)             501.69       118.16       619.85       0.6060          1.5872            1.4332         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)             501.69       127.94       629.63       0.6046          1.6067            1.4417         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)            501.69       211.65       713.34       0.9984          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)            501.69       304.31       806.00       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)            501.69       205.49       707.18       0.9984          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)            501.69       306.32       808.00       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)            501.69       217.59       719.28       0.9982          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)            501.69       319.46       821.15       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-random (self)                       501.69       502.33     1_004.02       0.9983          1.0002            1.0000         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)                 595.29        98.43       693.73       0.6495          1.3870            1.2897         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)                595.29       111.93       707.22       0.6469          1.4063            1.2984         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)                595.29       123.34       718.63       0.6459          1.4169            1.3015         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)                595.29       193.03       788.32       0.9975          1.0004            1.0000         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)                595.29       287.93       883.23       0.9985          1.0003            1.0000         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)               595.29       203.57       798.87       0.9986          1.0002            1.0000         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)               595.29       302.03       897.32       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)               595.29       216.76       812.05       0.9986          1.0002            1.0000         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)               595.29       314.27       909.56       0.9999          1.0000            1.0000         4.36
IVF-Binary-512-nl158-pca (self)                          595.29       512.88     1_108.17       0.9986          1.0002            1.0000         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)                499.66       105.04       604.70       0.6485          1.3923            1.2910         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)                499.66       113.10       612.76       0.6475          1.3989            1.2951         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)                499.66       124.54       624.20       0.6463          1.4102            1.2992         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)               499.66       199.17       698.83       0.9986          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)               499.66       295.10       794.76       0.9996          1.0001            1.0000         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)               499.66       201.32       700.98       0.9987          1.0001            1.0000         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)               499.66       301.81       801.47       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)               499.66       217.31       716.97       0.9986          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)               499.66       312.44       812.10       0.9999          1.0000            1.0000         4.49
IVF-Binary-512-nl223-pca (self)                          499.66       490.06       989.72       0.9987          1.0002            1.0000         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)                549.44       110.41       659.85       0.6479          1.3962            1.2944         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)                549.44       114.52       663.97       0.6475          1.3988            1.2956         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)                549.44       122.37       671.82       0.6465          1.4082            1.2988         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)               549.44       206.54       755.98       0.9987          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)               549.44       299.80       849.24       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)               549.44       212.20       761.64       0.9987          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)               549.44       301.76       851.20       0.9999          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)               549.44       212.76       762.21       0.9986          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)               549.44       313.77       863.21       0.9999          1.0000            1.0000         4.67
IVF-Binary-512-nl316-pca (self)                          549.44       489.25     1_038.70       0.9987          1.0001            1.0000         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)             569.52       152.74       722.26       0.6684          1.3902            1.2852         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)            569.52       168.43       737.95       0.6650          1.4127            1.2951         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)            569.52       188.55       758.07       0.6638          1.4277            1.2989         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)            569.52       248.46       817.97       0.9982          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)            569.52       348.47       917.99       0.9985          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)           569.52       264.32       833.84       0.9995          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)           569.52       369.88       939.40       0.9999          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)           569.52       282.22       851.73       0.9995          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)           569.52       401.99       971.51       0.9999          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-random (self)                      569.52       750.64     1_320.15       0.9995          1.0001            1.0000         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)            462.37       162.01       624.38       0.6665          1.4019            1.2893         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)            462.37       166.25       628.62       0.6653          1.4101            1.2935         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)            462.37       185.05       647.42       0.6639          1.4229            1.2995         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)           462.37       253.47       715.84       0.9994          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)           462.37       361.22       823.59       0.9997          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)           462.37       261.69       724.06       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)           462.37       371.12       833.50       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)           462.37       283.81       746.18       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)           462.37       424.97       887.34       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-random (self)                      462.37       725.82     1_188.20       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)            510.86       172.10       682.95       0.6661          1.4029            1.2911         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)            510.86       169.67       680.52       0.6658          1.4056            1.2923         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)            510.86       182.90       693.75       0.6646          1.4176            1.2976         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)           510.86       260.33       771.19       0.9996          1.0001            1.0000         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)           510.86       371.15       882.01       0.9999          1.0000            1.0000         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)           510.86       263.43       774.29       0.9996          1.0001            1.0000         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)           510.86       381.35       892.20       0.9999          1.0000            1.0000         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)           510.86       303.98       814.84       0.9996          1.0001            1.0000         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)           510.86       391.37       902.22       1.0000          1.0000            1.0000         8.72
IVF-Binary-1024-nl316-random (self)                      510.86       716.42     1_227.28       0.9996          1.0001            1.0000         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)                660.04       154.16       814.20       0.6911          1.3166            1.2268         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)               660.04       170.05       830.09       0.6886          1.3332            1.2328         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)               660.04       187.07       847.11       0.6877          1.3423            1.2348         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)               660.04       248.95       908.99       0.9983          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)               660.04       349.71     1_009.75       0.9985          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)              660.04       280.96       941.00       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)              660.04       379.10     1_039.14       0.9999          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)              660.04       287.12       947.15       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)              660.04       394.93     1_054.97       1.0000          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-pca (self)                         660.04       752.28     1_412.32       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)               542.23       163.16       705.39       0.6899          1.3227            1.2295         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)               542.23       167.23       709.46       0.6889          1.3281            1.2312         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)               542.23       184.64       726.87       0.6878          1.3378            1.2349         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)              542.23       261.13       803.37       0.9994          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)              542.23       363.93       906.16       0.9997          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)              542.23       262.50       804.73       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)              542.23       371.29       913.52       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)              542.23       282.30       824.54       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)              542.23       395.40       937.64       1.0000          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-pca (self)                         542.23       721.47     1_263.71       0.9996          1.0000            1.0000         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)               615.01       168.49       783.50       0.6894          1.3254            1.2310         8.72
IVF-Binary-1024-nl316-np17-rf0-pca (query)               615.01       171.17       786.18       0.6891          1.3273            1.2321         8.72
IVF-Binary-1024-nl316-np25-rf0-pca (query)               615.01       183.72       798.73       0.6881          1.3357            1.2348         8.72
IVF-Binary-1024-nl316-np15-rf10-pca (query)              615.01       278.42       893.43       0.9996          1.0001            1.0000         8.72
IVF-Binary-1024-nl316-np15-rf20-pca (query)              615.01       368.80       983.81       0.9999          1.0000            1.0000         8.72
IVF-Binary-1024-nl316-np17-rf10-pca (query)              615.01       265.91       880.92       0.9997          1.0000            1.0000         8.72
IVF-Binary-1024-nl316-np17-rf20-pca (query)              615.01       375.21       990.22       0.9999          1.0000            1.0000         8.72
IVF-Binary-1024-nl316-np25-rf10-pca (query)              615.01       282.15       897.16       0.9997          1.0001            1.0000         8.72
IVF-Binary-1024-nl316-np25-rf20-pca (query)              615.01       396.67     1_011.68       1.0000          1.0000            1.0000         8.72
IVF-Binary-1024-nl316-pca (self)                         615.01       728.09     1_343.10       0.9996          1.0000            1.0000         8.72
IVF-Binary-512-nl158-np7-rf0-sign (query)                392.66       258.56       651.21       0.0599          7.8187            7.0837         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)               392.66       272.47       665.13       0.0531          8.8976            7.9526         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)               392.66       286.20       678.86       0.0490          9.8779            8.5677         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)               392.66       378.55       771.20       0.3280          1.8159            1.7167         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)               392.66       606.97       999.63       0.5253          1.3907            1.3172         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)              392.66       343.59       736.25       0.2847          1.9641            1.8465         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)              392.66       604.73       997.38       0.4484          1.5123            1.4339         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)              392.66       355.50       748.16       0.2613          2.0647            1.9344         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)              392.66       621.03     1_013.69       0.4068          1.5960            1.5104         3.36
IVF-Binary-512-nl158-sign (self)                         392.66       951.38     1_344.04       0.2912          1.9452            1.8236         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               290.39       260.49       550.88       0.0577          7.8281            7.1585         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               290.39       270.52       560.91       0.0546          8.3001            7.5057         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               290.39       280.50       570.89       0.0505          9.1439            8.1402         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              290.39       345.61       636.00       0.3147          1.8520            1.7431         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              290.39       584.24       874.63       0.5027          1.4125            1.3453         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              290.39       338.59       628.98       0.2947          1.9188            1.8045         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              290.39       597.58       887.97       0.4678          1.4692            1.3975         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              290.39       350.62       641.01       0.2692          2.0240            1.8933         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              290.39       616.28       906.67       0.4177          1.5636            1.4849         3.49
IVF-Binary-512-nl223-sign (self)                         290.39       921.87     1_212.26       0.3024          1.8948            1.7792         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)               334.53       263.65       598.18       0.0577          7.7120            7.0592         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)               334.53       267.57       602.10       0.0559          7.9326            7.2580         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)               334.53       278.98       613.51       0.0521          8.6287            7.8189         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)              334.53       340.42       674.95       0.3158          1.8398            1.7379         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)              334.53       589.17       923.71       0.5020          1.4135            1.3477         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)              334.53       337.14       671.67       0.3056          1.8727            1.7707         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)              334.53       596.98       931.51       0.4830          1.4414            1.3763         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)              334.53       350.32       684.85       0.2803          1.9712            1.8511         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)              334.53       644.57       979.10       0.4348          1.5263            1.4540         3.67
IVF-Binary-512-nl316-sign (self)                         334.53       926.60     1_261.13       0.3123          1.8552            1.7464         3.67
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 768D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       100.24     1_940.80     2_041.04       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.24     6_565.00     6_665.24       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                194.32       292.90       487.23       0.5361          1.8068            1.5908         2.28
ExhaustiveBinary-256-random-rf10 (query)                 194.32       450.09       644.41       0.9868          1.0022            1.0000         2.28
ExhaustiveBinary-256-random-rf20 (query)                 194.32       610.35       804.67       0.9980          1.0003            1.0000         2.28
ExhaustiveBinary-256-random (self)                       194.32     1_410.49     1_604.81       0.9876          1.0021            1.0000         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   404.40       293.63       698.03       0.5754          1.5495            1.4128         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    404.40       451.30       855.70       0.9895          1.0018            1.0000         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    404.40       611.60     1_016.00       0.9983          1.0002            1.0000         2.28
ExhaustiveBinary-256-pca (self)                          404.40     1_416.43     1_820.83       0.9897          1.0017            1.0000         2.28
ExhaustiveBinary-512-random_no_rr (query)                299.68       427.54       727.21       0.5866          1.6778            1.4946         4.55
ExhaustiveBinary-512-random-rf10 (query)                 299.68       589.38       889.06       0.9966          1.0005            1.0000         4.55
ExhaustiveBinary-512-random-rf20 (query)                 299.68       756.62     1_056.29       0.9997          1.0001            1.0000         4.55
ExhaustiveBinary-512-random (self)                       299.68     1_889.22     2_188.90       0.9969          1.0004            1.0000         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   494.01       421.60       915.60       0.6388          1.4217            1.3032         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    494.01       596.15     1_090.15       0.9979          1.0003            1.0000         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    494.01       771.01     1_265.02       0.9998          1.0000            1.0000         4.55
ExhaustiveBinary-512-pca (self)                          494.01     1_886.93     2_380.94       0.9981          1.0002            1.0000         4.55
ExhaustiveBinary-1024-random_no_rr (query)               498.29       634.54     1_132.82       0.6446          1.4909            1.3512         9.11
ExhaustiveBinary-1024-random-rf10 (query)                498.29       820.27     1_318.56       0.9993          1.0001            1.0000         9.11
ExhaustiveBinary-1024-random-rf20 (query)                498.29     1_002.44     1_500.72       0.9999          1.0000            1.0000         9.11
ExhaustiveBinary-1024-random (self)                      498.29     2_707.24     3_205.53       0.9994          1.0001            1.0000         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  698.91       639.60     1_338.52       0.6795          1.3452            1.2483         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   698.91       825.32     1_524.24       0.9996          1.0001            1.0000         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   698.91     1_007.65     1_706.57       1.0000          1.0000            1.0000         9.11
ExhaustiveBinary-1024-pca (self)                         698.91     2_708.10     3_407.02       0.9997          1.0000            1.0000         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  127.20       810.55       937.74       0.0420         17.7082           13.0970         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   127.20       885.08     1_012.27       0.1896          2.5240            2.4052         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   127.20     1_364.24     1_491.44       0.3229          1.8300            1.7348         4.58
ExhaustiveBinary-768-sign (self)                         127.20     2_857.27     2_984.47       0.1997          2.4832            2.3546         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)              620.02        99.49       719.51       0.5428          1.7090            1.5496         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)             620.02       114.89       734.90       0.5406          1.7334            1.5625         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)             620.02       121.61       741.63       0.5396          1.7531            1.5705         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)             620.02       211.18       831.19       0.9891          1.0018            1.0000         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)             620.02       341.41       961.43       0.9982          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)            620.02       210.28       830.30       0.9884          1.0020            1.0000         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)            620.02       336.56       956.58       0.9984          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)            620.02       219.44       839.46       0.9878          1.0021            1.0000         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)            620.02       340.31       960.32       0.9982          1.0002            1.0000         2.74
IVF-Binary-256-nl158-random (self)                       620.02       474.54     1_094.56       0.9891          1.0018            1.0000         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)             474.23        99.42       573.65       0.5422          1.7157            1.5515         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)             474.23       104.39       578.62       0.5414          1.7255            1.5576         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)             474.23       111.43       585.66       0.5404          1.7477            1.5654         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)            474.23       216.65       690.88       0.9890          1.0018            1.0000         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)            474.23       328.93       803.16       0.9986          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)            474.23       212.88       687.11       0.9885          1.0019            1.0000         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)            474.23       333.31       807.54       0.9985          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)            474.23       218.50       692.73       0.9878          1.0021            1.0000         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)            474.23       342.81       817.04       0.9983          1.0002            1.0000         2.93
IVF-Binary-256-nl223-random (self)                       474.23       473.38       947.61       0.9892          1.0018            1.0000         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)             514.72       109.70       624.42       0.5421          1.7133            1.5522         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)             514.72       109.20       623.92       0.5417          1.7180            1.5546         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)             514.72       115.73       630.46       0.5407          1.7339            1.5610         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)            514.72       219.67       734.39       0.9892          1.0018            1.0000         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)            514.72       343.06       857.78       0.9987          1.0002            1.0000         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)            514.72       219.53       734.25       0.9889          1.0018            1.0000         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)            514.72       346.20       860.92       0.9987          1.0002            1.0000         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)            514.72       225.73       740.45       0.9882          1.0020            1.0000         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)            514.72       347.44       862.16       0.9985          1.0002            1.0000         3.21
IVF-Binary-256-nl316-random (self)                       514.72       493.02     1_007.75       0.9895          1.0017            1.0000         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)                 812.83        96.13       908.96       0.5809          1.4971            1.3930         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)                812.83       103.09       915.92       0.5792          1.5101            1.3989         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)                812.83       106.32       919.15       0.5786          1.5188            1.4015         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)                812.83       202.61     1_015.43       0.9910          1.0014            1.0000         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)                812.83       319.18     1_132.00       0.9984          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)               812.83       210.24     1_023.07       0.9906          1.0015            1.0000         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)               812.83       328.18     1_141.01       0.9987          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)               812.83       217.86     1_030.69       0.9902          1.0016            1.0000         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)               812.83       337.40     1_150.22       0.9985          1.0002            1.0000         2.74
IVF-Binary-256-nl158-pca (self)                          812.83       467.45     1_280.28       0.9909          1.0014            1.0000         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)                659.79        98.15       757.94       0.5803          1.5034            1.3951         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)                659.79       101.12       760.91       0.5797          1.5083            1.3974         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)                659.79       109.57       769.36       0.5789          1.5174            1.3997         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)               659.79       210.40       870.19       0.9910          1.0015            1.0000         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)               659.79       326.80       986.58       0.9987          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)               659.79       212.86       872.64       0.9906          1.0015            1.0000         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)               659.79       330.56       990.35       0.9987          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)               659.79       223.34       883.13       0.9902          1.0016            1.0000         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)               659.79       340.64     1_000.42       0.9985          1.0002            1.0000         2.93
IVF-Binary-256-nl223-pca (self)                          659.79       470.85     1_130.63       0.9910          1.0014            1.0000         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)                723.12       108.21       831.33       0.5801          1.5018            1.3952         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)                723.12       110.63       833.75       0.5799          1.5047            1.3966         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)                723.12       117.55       840.67       0.5792          1.5123            1.3987         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)               723.12       219.74       942.86       0.9912          1.0014            1.0000         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)               723.12       336.52     1_059.64       0.9988          1.0001            1.0000         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)               723.12       217.47       940.59       0.9910          1.0014            1.0000         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)               723.12       337.61     1_060.73       0.9988          1.0001            1.0000         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)               723.12       225.34       948.47       0.9903          1.0016            1.0000         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)               723.12       360.14     1_083.26       0.9986          1.0002            1.0000         3.21
IVF-Binary-256-nl316-pca (self)                          723.12       498.13     1_221.25       0.9914          1.0013            1.0000         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)              745.69       137.16       882.84       0.5923          1.6023            1.4620         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)             745.69       159.14       904.82       0.5899          1.6243            1.4744         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)             745.69       168.75       914.43       0.5888          1.6417            1.4804         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)             745.69       246.40       992.08       0.9971          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)             745.69       367.14     1_112.82       0.9992          1.0001            1.0000         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)            745.69       257.92     1_003.60       0.9971          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)            745.69       383.67     1_129.35       0.9998          1.0000            1.0000         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)            745.69       270.63     1_016.31       0.9969          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)            745.69       391.52     1_137.20       0.9997          1.0000            1.0000         5.02
IVF-Binary-512-nl158-random (self)                       745.69       640.29     1_385.98       0.9974          1.0003            1.0000         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)             578.43       139.33       717.76       0.5911          1.6103            1.4676         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)             578.43       144.21       722.64       0.5902          1.6197            1.4730         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)             578.43       158.00       736.42       0.5889          1.6399            1.4803         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)            578.43       256.42       834.85       0.9973          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)            578.43       375.58       954.01       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)            578.43       258.79       837.22       0.9972          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)            578.43       392.80       971.23       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)            578.43       272.55       850.98       0.9970          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)            578.43       398.73       977.16       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-random (self)                       578.43       636.05     1_214.48       0.9974          1.0003            1.0000         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)             634.68       150.67       785.34       0.5912          1.6086            1.4684         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)             634.68       159.21       793.89       0.5909          1.6125            1.4703         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)             634.68       161.88       796.56       0.5898          1.6268            1.4764         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)            634.68       259.54       894.22       0.9974          1.0003            1.0000         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)            634.68       387.03     1_021.71       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)            634.68       261.47       896.15       0.9974          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)            634.68       392.72     1_027.40       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)            634.68       272.02       906.70       0.9971          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)            634.68       449.75     1_084.42       0.9997          1.0000            1.0000         5.48
IVF-Binary-512-nl316-random (self)                       634.68       776.73     1_411.41       0.9976          1.0003            1.0000         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)                 988.54       140.89     1_129.43       0.6430          1.3832            1.2899         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)                988.54       156.82     1_145.35       0.6414          1.3947            1.2951         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)                988.54       173.49     1_162.03       0.6405          1.4045            1.2983         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)                988.54       250.23     1_238.77       0.9980          1.0003            1.0000         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)                988.54       457.32     1_445.85       0.9993          1.0001            1.0000         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)               988.54       275.25     1_263.79       0.9982          1.0002            1.0000         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)               988.54       410.75     1_399.29       0.9998          1.0000            1.0000         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)               988.54       268.16     1_256.70       0.9980          1.0003            1.0000         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)               988.54       399.18     1_387.72       0.9998          1.0000            1.0000         5.02
IVF-Binary-512-nl158-pca (self)                          988.54       643.14     1_631.67       0.9983          1.0002            1.0000         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)                778.30       139.30       917.60       0.6422          1.3884            1.2919         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)                778.30       145.19       923.49       0.6415          1.3935            1.2938         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)                778.30       158.16       936.46       0.6407          1.4024            1.2975         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)               778.30       261.89     1_040.19       0.9983          1.0002            1.0000         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)               778.30       380.35     1_158.65       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)               778.30       254.83     1_033.13       0.9983          1.0002            1.0000         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)               778.30       383.26     1_161.56       0.9998          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)               778.30       277.40     1_055.70       0.9981          1.0003            1.0000         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)               778.30       406.99     1_185.29       0.9998          1.0000            1.0000         5.21
IVF-Binary-512-nl223-pca (self)                          778.30       638.37     1_416.67       0.9984          1.0002            1.0000         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)                822.56       150.58       973.14       0.6423          1.3903            1.2927         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)                822.56       151.59       974.15       0.6420          1.3925            1.2937         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)                822.56       167.35       989.91       0.6414          1.3991            1.2961         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)               822.56       264.12     1_086.68       0.9984          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)               822.56       388.46     1_211.02       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)               822.56       266.75     1_089.32       0.9983          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)               822.56       393.21     1_215.77       0.9999          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)               822.56       271.76     1_094.32       0.9982          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)               822.56       405.84     1_228.40       0.9999          1.0000            1.0000         5.48
IVF-Binary-512-nl316-pca (self)                          822.56       653.96     1_476.52       0.9985          1.0002            1.0000         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)             928.11       219.41     1_147.51       0.6491          1.4436            1.3314         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)            928.11       240.73     1_168.84       0.6468          1.4602            1.3409         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)            928.11       238.14     1_166.25       0.6457          1.4722            1.3447         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)            928.11       329.80     1_257.90       0.9990          1.0002            1.0000         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)            928.11       457.27     1_385.38       0.9993          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)           928.11       343.03     1_271.14       0.9994          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)           928.11       484.66     1_412.76       0.9999          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)           928.11       362.23     1_290.34       0.9994          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)           928.11       510.82     1_438.92       0.9999          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-random (self)                      928.11       960.63     1_888.73       0.9995          1.0001            1.0000         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)            784.51       219.24     1_003.75       0.6479          1.4492            1.3354         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)            784.51       220.60     1_005.11       0.6470          1.4558            1.3394         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)            784.51       238.46     1_022.98       0.6459          1.4693            1.3443         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)           784.51       339.07     1_123.59       0.9993          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)           784.51       476.87     1_261.39       0.9998          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)           784.51       354.24     1_138.76       0.9994          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)           784.51       490.24     1_274.76       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)           784.51       371.03     1_155.54       0.9994          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)           784.51       515.93     1_300.44       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-random (self)                      784.51       958.97     1_743.49       0.9995          1.0001            1.0000         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)            842.85       230.31     1_073.16       0.6477          1.4477            1.3364        10.03
IVF-Binary-1024-nl316-np17-rf0-random (query)            842.85       230.16     1_073.01       0.6473          1.4509            1.3386        10.03
IVF-Binary-1024-nl316-np25-rf0-random (query)            842.85       247.00     1_089.85       0.6464          1.4612            1.3420        10.03
IVF-Binary-1024-nl316-np15-rf10-random (query)           842.85       365.03     1_207.88       0.9995          1.0001            1.0000        10.03
IVF-Binary-1024-nl316-np15-rf20-random (query)           842.85       504.02     1_346.87       0.9999          1.0000            1.0000        10.03
IVF-Binary-1024-nl316-np17-rf10-random (query)           842.85       362.91     1_205.76       0.9995          1.0001            1.0000        10.03
IVF-Binary-1024-nl316-np17-rf20-random (query)           842.85       497.89     1_340.75       0.9999          1.0000            1.0000        10.03
IVF-Binary-1024-nl316-np25-rf10-random (query)           842.85       397.16     1_240.01       0.9994          1.0001            1.0000        10.03
IVF-Binary-1024-nl316-np25-rf20-random (query)           842.85       525.40     1_368.26       0.9999          1.0000            1.0000        10.03
IVF-Binary-1024-nl316-random (self)                      842.85       977.21     1_820.06       0.9995          1.0001            1.0000        10.03
IVF-Binary-1024-nl158-np7-rf0-pca (query)              1_128.42       208.37     1_336.79       0.6828          1.3188            1.2377         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)             1_128.42       235.48     1_363.90       0.6811          1.3289            1.2434         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)             1_128.42       242.14     1_370.56       0.6804          1.3356            1.2451         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)             1_128.42       335.35     1_463.77       0.9991          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)             1_128.42       461.57     1_589.99       0.9993          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)            1_128.42       364.66     1_493.08       0.9996          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)            1_128.42       486.77     1_615.19       0.9999          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)            1_128.42       373.77     1_502.19       0.9996          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)            1_128.42       520.94     1_649.35       0.9999          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-pca (self)                       1_128.42       982.76     2_111.18       0.9997          1.0000            1.0000         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)               974.97       218.31     1_193.28       0.6818          1.3245            1.2403         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)               974.97       222.44     1_197.42       0.6812          1.3284            1.2421         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)               974.97       240.00     1_214.97       0.6805          1.3347            1.2444         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)              974.97       350.84     1_325.82       0.9996          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)              974.97       482.47     1_457.44       0.9998          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)              974.97       357.55     1_332.52       0.9996          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)              974.97       495.21     1_470.18       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)              974.97       370.04     1_345.01       0.9996          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)              974.97       517.57     1_492.55       1.0000          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-pca (self)                         974.97       996.54     1_971.52       0.9997          1.0000            1.0000         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_048.80       229.39     1_278.19       0.6816          1.3253            1.2414        10.03
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_048.80       244.56     1_293.37       0.6814          1.3274            1.2423        10.03
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_048.80       247.67     1_296.47       0.6807          1.3324            1.2441        10.03
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_048.80       360.96     1_409.76       0.9996          1.0000            1.0000        10.03
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_048.80       492.48     1_541.29       0.9999          1.0000            1.0000        10.03
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_048.80       366.24     1_415.04       0.9997          1.0000            1.0000        10.03
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_048.80       499.86     1_548.66       1.0000          1.0000            1.0000        10.03
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_048.80       383.76     1_432.56       0.9996          1.0000            1.0000        10.03
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_048.80       523.80     1_572.60       1.0000          1.0000            1.0000        10.03
IVF-Binary-1024-nl316-pca (self)                       1_048.80       976.58     2_025.39       0.9997          1.0000            1.0000        10.03
IVF-Binary-768-nl158-np7-rf0-sign (query)                561.95       373.65       935.60       0.0583          8.1268            7.2961         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)               561.95       396.96       958.91       0.0529          9.3091            7.9708         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)               561.95       404.55       966.50       0.0505         10.1556            8.5015         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)               561.95       464.42     1_026.37       0.3170          1.8630            1.7551         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)               561.95       830.53     1_392.47       0.4855          1.4615            1.3677         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)              561.95       476.90     1_038.85       0.2845          1.9753            1.8503         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)              561.95       852.65     1_414.60       0.4332          1.5441            1.4471         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)              561.95       492.22     1_054.17       0.2675          2.0445            1.9133         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)              561.95       892.62     1_454.57       0.4073          1.5928            1.4944         5.04
IVF-Binary-768-nl158-sign (self)                         561.95     1_344.40     1_906.35       0.2972          1.9369            1.8085         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)               411.87       378.82       790.69       0.0576          8.2518            7.3733         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)               411.87       385.84       797.71       0.0551          8.6834            7.6674         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)               411.87       403.86       815.73       0.0516          9.5662            8.2933         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)              411.87       473.33       885.21       0.3089          1.8833            1.7710         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)              411.87       897.66     1_309.54       0.4712          1.4806            1.3939         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)              411.87       474.75       886.62       0.2945          1.9305            1.8092         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)              411.87       852.96     1_264.83       0.4486          1.5152            1.4266         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)              411.87       491.34       903.21       0.2746          2.0094            1.8730         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)              411.87       877.53     1_289.41       0.4160          1.5734            1.4836         5.23
IVF-Binary-768-nl223-sign (self)                         411.87     1_338.48     1_750.35       0.3066          1.8971            1.7753         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)               461.17       388.36       849.53       0.0578          8.0396            7.2472         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)               461.17       391.44       852.61       0.0567          8.2545            7.4105         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)               461.17       429.54       890.70       0.0535          8.9914            7.9034         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)              461.17       480.87       942.03       0.3154          1.8548            1.7481         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)              461.17       865.38     1_326.54       0.4766          1.4696            1.3796         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)              461.17       482.22       943.38       0.3078          1.8784            1.7688         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)              461.17       859.03     1_320.19       0.4650          1.4877            1.3979         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)              461.17       495.59       956.75       0.2871          1.9530            1.8316         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)              461.17       878.41     1_339.58       0.4311          1.5454            1.4535         5.51
IVF-Binary-768-nl316-sign (self)                         461.17     1_375.85     1_837.01       0.3198          1.8447            1.7339         5.51
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
- *reranking_factor (rf)*: The RaBitQ estimate picks the candidates, then the
  on-disk vectors are loaded and re-scored exactly. `10` means `10 * k` vectors
  get re-scored. `rfB` is the default (`None`): every estimate carries an error
  bound, and only candidates whose lower bound still beats the current k-th
  distance are re-scored, so the shortlist sizes itself per query. The
  exhaustive grid runs `rf0` (the estimate alone, no re-ranking), `5`, `10`,
  `20` and `B`; the IVF grid runs `rf0`, `10`, `20` and `B`.
- *ex_bits (ex)*: Magnitude bits per coordinate on top of the sign bit, `0` by
  default. With `ex_bits > 0` the candidates the one-bit bound cannot rule out
  are re-scored on the wider code instead of from disk, so there is no store at
  all. The `ex` rows run `2`, `4` and `8` at `nl = sqrt(n)` with `np` at
  `sqrt(nlist)` and 5% of `nlist`.

**Tunable parameters *(IVF-specific)*:**

- *Number of lists (nl)*: Number of k-means clusters, `sqrt(n)` as a default.
  The grid runs `sqrt(n/2)`, `sqrt(n)` and `sqrt(2n)`.
- *Number of probes (np)*: `sqrt(nlist)`, `sqrt(2 * nlist)` and 5% of `nlist`,
  deduplicated.

Self queries run at `reranking_factor = 10` for the exhaustive index and at the
bound-gated default for IVF, the latter at `np = sqrt(2 * nlist)`.

#### Correlated data

<details>
<summary><b>Correlated data - 256 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 256D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.93       706.38       739.32       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.93     2_317.56     2_350.49       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             253.84        61.18       315.02       0.4587          1.0613            1.0633         2.56
ExhaustiveRaBitQ-rf5 (query)                             253.84       115.91       369.75       0.8093          1.0067            1.0054         2.56
ExhaustiveRaBitQ-rf10 (query)                            253.84       166.54       420.38       0.9208          1.0021            1.0006         2.56
ExhaustiveRaBitQ-rf20 (query)                            253.84       249.61       503.45       0.9802          1.0004            1.0000         2.56
ExhaustiveRaBitQ-rfB (query)                             253.84       159.20       413.04       0.9787          1.0005            1.0000         2.56
ExhaustiveRaBitQ (self)                                  253.84       514.58       768.42       0.9199          1.0022            1.0007         2.56
IVF-RaBitQ-nl158-np7-rf0 (query)                         253.62        47.13       300.75       0.4737          1.0570            1.0594         2.68
IVF-RaBitQ-nl158-np12-rf0 (query)                        253.62        50.21       303.82       0.4737          1.0570            1.0594         2.68
IVF-RaBitQ-nl158-np17-rf0 (query)                        253.62        53.09       306.71       0.4737          1.0570            1.0594         2.68
IVF-RaBitQ-nl158-np7-rf10 (query)                        253.62       126.68       380.30       0.9287          1.0018            1.0003         2.68
IVF-RaBitQ-nl158-np7-rf20 (query)                        253.62       196.41       450.02       0.9829          1.0004            1.0000         2.68
IVF-RaBitQ-nl158-np7-rfB (query)                         253.62       110.19       363.81       0.9781          1.0005            1.0000         2.68
IVF-RaBitQ-nl158-np12-rf10 (query)                       253.62       126.71       380.33       0.9287          1.0018            1.0003         2.68
IVF-RaBitQ-nl158-np12-rf20 (query)                       253.62       200.01       453.63       0.9829          1.0004            1.0000         2.68
IVF-RaBitQ-nl158-np12-rfB (query)                        253.62       112.47       366.09       0.9781          1.0005            1.0000         2.68
IVF-RaBitQ-nl158-np17-rf10 (query)                       253.62       131.77       385.38       0.9287          1.0018            1.0003         2.68
IVF-RaBitQ-nl158-np17-rf20 (query)                       253.62       202.73       456.35       0.9829          1.0004            1.0000         2.68
IVF-RaBitQ-nl158-np17-rfB (query)                        253.62       115.71       369.33       0.9781          1.0005            1.0000         2.68
IVF-RaBitQ-nl158 (self)                                  253.62       348.85       602.47       0.9781          1.0005            1.0000         2.68
IVF-RaBitQ-nl223-np11-rf0 (query)                        213.89        50.49       264.37       0.4803          1.0550            1.0574         2.83
IVF-RaBitQ-nl223-np14-rf0 (query)                        213.89        51.21       265.09       0.4803          1.0550            1.0574         2.83
IVF-RaBitQ-nl223-np21-rf0 (query)                        213.89        54.82       268.70       0.4803          1.0550            1.0574         2.83
IVF-RaBitQ-nl223-np11-rf10 (query)                       213.89       125.24       339.12       0.9340          1.0017            1.0001         2.83
IVF-RaBitQ-nl223-np11-rf20 (query)                       213.89       197.62       411.51       0.9847          1.0003            1.0000         2.83
IVF-RaBitQ-nl223-np11-rfB (query)                        213.89       108.34       322.22       0.9783          1.0005            1.0000         2.83
IVF-RaBitQ-nl223-np14-rf10 (query)                       213.89       125.64       339.52       0.9340          1.0017            1.0001         2.83
IVF-RaBitQ-nl223-np14-rf20 (query)                       213.89       198.50       412.38       0.9847          1.0003            1.0000         2.83
IVF-RaBitQ-nl223-np14-rfB (query)                        213.89       111.61       325.50       0.9783          1.0005            1.0000         2.83
IVF-RaBitQ-nl223-np21-rf10 (query)                       213.89       128.39       342.27       0.9340          1.0017            1.0001         2.83
IVF-RaBitQ-nl223-np21-rf20 (query)                       213.89       207.15       421.04       0.9847          1.0003            1.0000         2.83
IVF-RaBitQ-nl223-np21-rfB (query)                        213.89       112.98       326.87       0.9783          1.0005            1.0000         2.83
IVF-RaBitQ-nl223 (self)                                  213.89       347.91       561.79       0.9781          1.0005            1.0000         2.83
IVF-RaBitQ-nl316-np15-rf0 (query)                        250.67        52.62       303.30       0.4874          1.0531            1.0555         3.05
IVF-RaBitQ-nl316-np17-rf0 (query)                        250.67        56.54       307.21       0.4874          1.0531            1.0555         3.05
IVF-RaBitQ-nl316-np25-rf0 (query)                        250.67        56.93       307.60       0.4874          1.0531            1.0555         3.05
IVF-RaBitQ-nl316-np15-rf10 (query)                       250.67       129.25       379.92       0.9356          1.0016            1.0001         3.05
IVF-RaBitQ-nl316-np15-rf20 (query)                       250.67       192.09       442.76       0.9853          1.0003            1.0000         3.05
IVF-RaBitQ-nl316-np15-rfB (query)                        250.67       106.78       357.46       0.9779          1.0005            1.0000         3.05
IVF-RaBitQ-nl316-np17-rf10 (query)                       250.67       126.35       377.03       0.9356          1.0016            1.0001         3.05
IVF-RaBitQ-nl316-np17-rf20 (query)                       250.67       192.86       443.53       0.9853          1.0003            1.0000         3.05
IVF-RaBitQ-nl316-np17-rfB (query)                        250.67       108.05       358.72       0.9779          1.0005            1.0000         3.05
IVF-RaBitQ-nl316-np25-rf10 (query)                       250.67       128.45       379.13       0.9356          1.0016            1.0001         3.05
IVF-RaBitQ-nl316-np25-rf20 (query)                       250.67       198.10       448.77       0.9853          1.0003            1.0000         3.05
IVF-RaBitQ-nl316-np25-rfB (query)                        250.67       111.23       361.90       0.9779          1.0005            1.0000         3.05
IVF-RaBitQ-nl316 (self)                                  250.67       341.60       592.27       0.9781          1.0005            1.0000         3.05
IVF-RaBitQ-nl223-ex2-np14 (query)                        215.97       118.57       334.54       0.8667          1.0029            1.0025         6.26
IVF-RaBitQ-nl223-ex2-np21 (query)                        215.97       122.44       338.41       0.8667          1.0029            1.0025         6.26
IVF-RaBitQ-nl223-ex4-np14 (query)                        226.51       134.21       360.72       0.9451          1.0007            1.0002         9.32
IVF-RaBitQ-nl223-ex4-np21 (query)                        226.51       138.39       364.89       0.9451          1.0007            1.0002         9.32
IVF-RaBitQ-nl223-ex8-np14 (query)                        255.43       121.07       376.50       0.9713          1.0005            1.0000        15.42
IVF-RaBitQ-nl223-ex8-np21 (query)                        255.43       123.83       379.26       0.9713          1.0005            1.0000        15.42
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 512D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.27     1_344.70     1_412.97       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.27     4_494.88     4_563.15       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                             387.54       100.33       487.87       0.4490          1.0427            1.0449         4.37
ExhaustiveRaBitQ-rf5 (query)                             387.54       167.75       555.29       0.7775          1.0056            1.0048         4.37
ExhaustiveRaBitQ-rf10 (query)                            387.54       232.53       620.07       0.8975          1.0019            1.0010         4.37
ExhaustiveRaBitQ-rf20 (query)                            387.54       356.66       744.20       0.9713          1.0005            1.0000         4.37
ExhaustiveRaBitQ-rfB (query)                             387.54       322.62       710.16       0.9714          1.0005            1.0000         4.37
ExhaustiveRaBitQ (self)                                  387.54       706.23     1_093.77       0.8982          1.0020            1.0010         4.37
IVF-RaBitQ-nl158-np7-rf0 (query)                         510.69        82.06       592.75       0.4604          1.0403            1.0428         4.59
IVF-RaBitQ-nl158-np12-rf0 (query)                        510.69        85.48       596.17       0.4604          1.0403            1.0428         4.59
IVF-RaBitQ-nl158-np17-rf0 (query)                        510.69        89.65       600.34       0.4604          1.0403            1.0428         4.59
IVF-RaBitQ-nl158-np7-rf10 (query)                        510.69       197.31       708.00       0.9052          1.0018            1.0008         4.59
IVF-RaBitQ-nl158-np7-rf20 (query)                        510.69       304.96       815.65       0.9742          1.0004            1.0000         4.59
IVF-RaBitQ-nl158-np7-rfB (query)                         510.69       227.95       738.64       0.9713          1.0005            1.0000         4.59
IVF-RaBitQ-nl158-np12-rf10 (query)                       510.69       199.69       710.38       0.9052          1.0018            1.0008         4.59
IVF-RaBitQ-nl158-np12-rf20 (query)                       510.69       310.18       820.86       0.9742          1.0004            1.0000         4.59
IVF-RaBitQ-nl158-np12-rfB (query)                        510.69       232.66       743.35       0.9713          1.0005            1.0000         4.59
IVF-RaBitQ-nl158-np17-rf10 (query)                       510.69       210.40       721.09       0.9052          1.0018            1.0008         4.59
IVF-RaBitQ-nl158-np17-rf20 (query)                       510.69       314.75       825.43       0.9742          1.0004            1.0000         4.59
IVF-RaBitQ-nl158-np17-rfB (query)                        510.69       237.04       747.72       0.9713          1.0005            1.0000         4.59
IVF-RaBitQ-nl158 (self)                                  510.69       720.07     1_230.75       0.9712          1.0005            1.0000         4.59
IVF-RaBitQ-nl223-np11-rf0 (query)                        376.53        89.20       465.73       0.4649          1.0394            1.0418         4.89
IVF-RaBitQ-nl223-np14-rf0 (query)                        376.53        90.34       466.87       0.4649          1.0394            1.0418         4.89
IVF-RaBitQ-nl223-np21-rf0 (query)                        376.53        96.86       473.39       0.4649          1.0394            1.0418         4.89
IVF-RaBitQ-nl223-np11-rf10 (query)                       376.53       201.87       578.39       0.9108          1.0016            1.0007         4.89
IVF-RaBitQ-nl223-np11-rf20 (query)                       376.53       305.97       682.50       0.9760          1.0004            1.0000         4.89
IVF-RaBitQ-nl223-np11-rfB (query)                        376.53       226.45       602.97       0.9720          1.0004            1.0000         4.89
IVF-RaBitQ-nl223-np14-rf10 (query)                       376.53       206.42       582.95       0.9108          1.0016            1.0007         4.89
IVF-RaBitQ-nl223-np14-rf20 (query)                       376.53       322.38       698.91       0.9760          1.0004            1.0000         4.89
IVF-RaBitQ-nl223-np14-rfB (query)                        376.53       225.02       601.55       0.9720          1.0004            1.0000         4.89
IVF-RaBitQ-nl223-np21-rf10 (query)                       376.53       206.73       583.26       0.9108          1.0016            1.0007         4.89
IVF-RaBitQ-nl223-np21-rf20 (query)                       376.53       315.01       691.53       0.9760          1.0004            1.0000         4.89
IVF-RaBitQ-nl223-np21-rfB (query)                        376.53       229.91       606.44       0.9720          1.0004            1.0000         4.89
IVF-RaBitQ-nl223 (self)                                  376.53       692.75     1_069.28       0.9717          1.0004            1.0000         4.89
IVF-RaBitQ-nl316-np15-rf0 (query)                        404.44        97.95       502.38       0.4716          1.0383            1.0409         5.34
IVF-RaBitQ-nl316-np17-rf0 (query)                        404.44        98.18       502.62       0.4716          1.0383            1.0409         5.34
IVF-RaBitQ-nl316-np25-rf0 (query)                        404.44       101.07       505.50       0.4716          1.0383            1.0409         5.34
IVF-RaBitQ-nl316-np15-rf10 (query)                       404.44       210.53       614.96       0.9131          1.0016            1.0006         5.34
IVF-RaBitQ-nl316-np15-rf20 (query)                       404.44       333.76       738.20       0.9766          1.0004            1.0000         5.34
IVF-RaBitQ-nl316-np15-rfB (query)                        404.44       222.49       626.93       0.9718          1.0004            1.0000         5.34
IVF-RaBitQ-nl316-np17-rf10 (query)                       404.44       211.16       615.59       0.9131          1.0016            1.0006         5.34
IVF-RaBitQ-nl316-np17-rf20 (query)                       404.44       321.29       725.73       0.9766          1.0004            1.0000         5.34
IVF-RaBitQ-nl316-np17-rfB (query)                        404.44       225.41       629.84       0.9718          1.0004            1.0000         5.34
IVF-RaBitQ-nl316-np25-rf10 (query)                       404.44       216.48       620.91       0.9131          1.0016            1.0006         5.34
IVF-RaBitQ-nl316-np25-rf20 (query)                       404.44       327.98       732.41       0.9766          1.0004            1.0000         5.34
IVF-RaBitQ-nl316-np25-rfB (query)                        404.44       229.70       634.13       0.9718          1.0004            1.0000         5.34
IVF-RaBitQ-nl316 (self)                                  404.44       693.68     1_098.11       0.9718          1.0004            1.0000         5.34
IVF-RaBitQ-nl223-ex2-np14 (query)                        414.56       231.73       646.28       0.8652          1.0020            1.0017        11.37
IVF-RaBitQ-nl223-ex2-np21 (query)                        414.56       235.43       649.98       0.8652          1.0020            1.0017        11.37
IVF-RaBitQ-nl223-ex4-np14 (query)                        422.30       262.00       684.30       0.9419          1.0006            1.0002        17.48
IVF-RaBitQ-nl223-ex4-np21 (query)                        422.30       266.80       689.10       0.9419          1.0006            1.0002        17.48
IVF-RaBitQ-nl223-ex8-np14 (query)                        473.49       239.12       712.61       0.9676          1.0004            1.0000        29.68
IVF-RaBitQ-nl223-ex8-np21 (query)                        473.49       241.37       714.87       0.9676          1.0004            1.0000        29.68
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 768D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        99.99     1_975.76     2_075.75       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.99     6_691.07     6_791.06       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                             541.25       126.04       667.29       0.4437          1.0346            1.0365         6.16
ExhaustiveRaBitQ-rf5 (query)                             541.25       205.51       746.76       0.7684          1.0047            1.0042         6.16
ExhaustiveRaBitQ-rf10 (query)                            541.25       282.23       823.47       0.8898          1.0017            1.0010         6.16
ExhaustiveRaBitQ-rf20 (query)                            541.25       425.90       967.14       0.9668          1.0004            1.0000         6.16
ExhaustiveRaBitQ-rfB (query)                             541.25       418.08       959.33       0.9693          1.0004            1.0000         6.16
ExhaustiveRaBitQ (self)                                  541.25       853.23     1_394.48       0.8897          1.0017            1.0010         6.16
IVF-RaBitQ-nl158-np7-rf0 (query)                         931.81       109.13     1_040.94       0.4547          1.0327            1.0351         6.49
IVF-RaBitQ-nl158-np12-rf0 (query)                        931.81       113.31     1_045.12       0.4547          1.0327            1.0351         6.49
IVF-RaBitQ-nl158-np17-rf0 (query)                        931.81       119.57     1_051.39       0.4547          1.0327            1.0351         6.49
IVF-RaBitQ-nl158-np7-rf10 (query)                        931.81       248.93     1_180.74       0.8974          1.0016            1.0008         6.49
IVF-RaBitQ-nl158-np7-rf20 (query)                        931.81       390.95     1_322.76       0.9698          1.0004            1.0000         6.49
IVF-RaBitQ-nl158-np7-rfB (query)                         931.81       332.32     1_264.13       0.9691          1.0004            1.0000         6.49
IVF-RaBitQ-nl158-np12-rf10 (query)                       931.81       252.23     1_184.04       0.8974          1.0016            1.0008         6.49
IVF-RaBitQ-nl158-np12-rf20 (query)                       931.81       385.89     1_317.70       0.9698          1.0004            1.0000         6.49
IVF-RaBitQ-nl158-np12-rfB (query)                        931.81       323.70     1_255.51       0.9691          1.0004            1.0000         6.49
IVF-RaBitQ-nl158-np17-rf10 (query)                       931.81       259.20     1_191.01       0.8974          1.0016            1.0008         6.49
IVF-RaBitQ-nl158-np17-rf20 (query)                       931.81       395.84     1_327.66       0.9698          1.0004            1.0000         6.49
IVF-RaBitQ-nl158-np17-rfB (query)                        931.81       323.17     1_254.98       0.9691          1.0004            1.0000         6.49
IVF-RaBitQ-nl158 (self)                                  931.81       987.36     1_919.17       0.9688          1.0004            1.0000         6.49
IVF-RaBitQ-nl223-np11-rf0 (query)                        662.07       115.31       777.38       0.4625          1.0317            1.0342         6.99
IVF-RaBitQ-nl223-np14-rf0 (query)                        662.07       141.56       803.62       0.4625          1.0317            1.0342         6.99
IVF-RaBitQ-nl223-np21-rf0 (query)                        662.07       128.56       790.62       0.4625          1.0317            1.0342         6.99
IVF-RaBitQ-nl223-np11-rf10 (query)                       662.07       254.54       916.60       0.9024          1.0015            1.0007         6.99
IVF-RaBitQ-nl223-np11-rf20 (query)                       662.07       382.09     1_044.15       0.9720          1.0003            1.0000         6.99
IVF-RaBitQ-nl223-np11-rfB (query)                        662.07       301.61       963.68       0.9698          1.0004            1.0000         6.99
IVF-RaBitQ-nl223-np14-rf10 (query)                       662.07       259.69       921.76       0.9024          1.0015            1.0007         6.99
IVF-RaBitQ-nl223-np14-rf20 (query)                       662.07       390.77     1_052.84       0.9720          1.0003            1.0000         6.99
IVF-RaBitQ-nl223-np14-rfB (query)                        662.07       318.30       980.37       0.9698          1.0004            1.0000         6.99
IVF-RaBitQ-nl223-np21-rf10 (query)                       662.07       265.63       927.70       0.9024          1.0015            1.0007         6.99
IVF-RaBitQ-nl223-np21-rf20 (query)                       662.07       414.25     1_076.31       0.9720          1.0003            1.0000         6.99
IVF-RaBitQ-nl223-np21-rfB (query)                        662.07       318.26       980.32       0.9698          1.0004            1.0000         6.99
IVF-RaBitQ-nl223 (self)                                  662.07       943.17     1_605.23       0.9692          1.0004            1.0000         6.99
IVF-RaBitQ-nl316-np15-rf0 (query)                        643.57       127.80       771.38       0.4653          1.0312            1.0337         7.64
IVF-RaBitQ-nl316-np17-rf0 (query)                        643.57       128.38       771.96       0.4653          1.0312            1.0337         7.64
IVF-RaBitQ-nl316-np25-rf0 (query)                        643.57       132.28       775.85       0.4653          1.0312            1.0337         7.64
IVF-RaBitQ-nl316-np15-rf10 (query)                       643.57       266.28       909.86       0.9042          1.0014            1.0007         7.64
IVF-RaBitQ-nl316-np15-rf20 (query)                       643.57       397.37     1_040.95       0.9727          1.0003            1.0000         7.64
IVF-RaBitQ-nl316-np15-rfB (query)                        643.57       299.89       943.47       0.9696          1.0004            1.0000         7.64
IVF-RaBitQ-nl316-np17-rf10 (query)                       643.57       270.29       913.86       0.9042          1.0014            1.0007         7.64
IVF-RaBitQ-nl316-np17-rf20 (query)                       643.57       395.44     1_039.01       0.9727          1.0003            1.0000         7.64
IVF-RaBitQ-nl316-np17-rfB (query)                        643.57       302.26       945.84       0.9696          1.0004            1.0000         7.64
IVF-RaBitQ-nl316-np25-rf10 (query)                       643.57       287.40       930.97       0.9042          1.0014            1.0007         7.64
IVF-RaBitQ-nl316-np25-rf20 (query)                       643.57       406.43     1_050.00       0.9727          1.0003            1.0000         7.64
IVF-RaBitQ-nl316-np25-rfB (query)                        643.57       317.47       961.04       0.9696          1.0004            1.0000         7.64
IVF-RaBitQ-nl316 (self)                                  643.57       945.64     1_589.21       0.9693          1.0004            1.0000         7.64
IVF-RaBitQ-nl223-ex2-np14 (query)                        563.91       344.50       908.40       0.8637          1.0016            1.0014        16.49
IVF-RaBitQ-nl223-ex2-np21 (query)                        563.91       349.88       913.78       0.8637          1.0016            1.0014        16.49
IVF-RaBitQ-nl223-ex4-np14 (query)                        597.50       388.08       985.58       0.9396          1.0005            1.0001        25.65
IVF-RaBitQ-nl223-ex4-np21 (query)                        597.50       396.47       993.97       0.9396          1.0005            1.0001        25.65
IVF-RaBitQ-nl223-ex8-np14 (query)                        680.70       348.25     1_028.95       0.9652          1.0004            1.0000        43.96
IVF-RaBitQ-nl223-ex8-np21 (query)                        680.70       350.62     1_031.32       0.9652          1.0004            1.0000        43.96
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 256D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.83       715.67       748.49       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.83     2_366.25     2_399.08       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             245.17        80.35       325.53       0.6872          1.0347            1.0333         2.57
ExhaustiveRaBitQ-rf5 (query)                             245.17       133.42       378.60       0.9895          1.0003            1.0000         2.57
ExhaustiveRaBitQ-rf10 (query)                            245.17       190.77       435.94       0.9995          1.0000            1.0000         2.57
ExhaustiveRaBitQ-rf20 (query)                            245.17       289.91       535.08       1.0000          1.0000            1.0000         2.57
ExhaustiveRaBitQ-rfB (query)                             245.17       131.87       377.05       0.9959          1.0001            1.0000         2.57
ExhaustiveRaBitQ (self)                                  245.17       599.07       844.24       0.9997          1.0000            1.0000         2.57
IVF-RaBitQ-nl158-np7-rf0 (query)                         254.13        46.20       300.33       0.6910          1.0338            1.0325         2.67
IVF-RaBitQ-nl158-np12-rf0 (query)                        254.13        49.67       303.79       0.6910          1.0338            1.0325         2.67
IVF-RaBitQ-nl158-np17-rf0 (query)                        254.13        53.32       307.45       0.6910          1.0338            1.0325         2.67
IVF-RaBitQ-nl158-np7-rf10 (query)                        254.13       126.02       380.15       0.9996          1.0000            1.0000         2.67
IVF-RaBitQ-nl158-np7-rf20 (query)                        254.13       200.80       454.92       1.0000          1.0000            1.0000         2.67
IVF-RaBitQ-nl158-np7-rfB (query)                         254.13        82.43       336.55       0.9961          1.0001            1.0000         2.67
IVF-RaBitQ-nl158-np12-rf10 (query)                       254.13       128.98       383.10       0.9996          1.0000            1.0000         2.67
IVF-RaBitQ-nl158-np12-rf20 (query)                       254.13       205.50       459.62       1.0000          1.0000            1.0000         2.67
IVF-RaBitQ-nl158-np12-rfB (query)                        254.13        85.78       339.91       0.9961          1.0001            1.0000         2.67
IVF-RaBitQ-nl158-np17-rf10 (query)                       254.13       131.87       386.00       0.9996          1.0000            1.0000         2.67
IVF-RaBitQ-nl158-np17-rf20 (query)                       254.13       211.28       465.41       1.0000          1.0000            1.0000         2.67
IVF-RaBitQ-nl158-np17-rfB (query)                        254.13        88.00       342.13       0.9961          1.0001            1.0000         2.67
IVF-RaBitQ-nl158 (self)                                  254.13       251.62       505.75       0.9956          1.0001            1.0000         2.67
IVF-RaBitQ-nl223-np11-rf0 (query)                        204.86        51.02       255.88       0.6962          1.0327            1.0313         2.84
IVF-RaBitQ-nl223-np14-rf0 (query)                        204.86        51.59       256.45       0.6962          1.0327            1.0313         2.84
IVF-RaBitQ-nl223-np21-rf0 (query)                        204.86        55.42       260.28       0.6962          1.0327            1.0313         2.84
IVF-RaBitQ-nl223-np11-rf10 (query)                       204.86       127.03       331.89       0.9996          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np11-rf20 (query)                       204.86       203.45       408.31       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np11-rfB (query)                        204.86        84.21       289.07       0.9960          1.0001            1.0000         2.84
IVF-RaBitQ-nl223-np14-rf10 (query)                       204.86       127.89       332.75       0.9996          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np14-rf20 (query)                       204.86       205.42       410.28       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np14-rfB (query)                        204.86        85.39       290.25       0.9960          1.0001            1.0000         2.84
IVF-RaBitQ-nl223-np21-rf10 (query)                       204.86       131.95       336.81       0.9996          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np21-rf20 (query)                       204.86       224.24       429.10       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np21-rfB (query)                        204.86        90.05       294.91       0.9960          1.0001            1.0000         2.84
IVF-RaBitQ-nl223 (self)                                  204.86       257.17       462.03       0.9959          1.0001            1.0000         2.84
IVF-RaBitQ-nl316-np15-rf0 (query)                        233.29        53.52       286.81       0.6995          1.0319            1.0306         3.06
IVF-RaBitQ-nl316-np17-rf0 (query)                        233.29        52.60       285.89       0.6995          1.0319            1.0306         3.06
IVF-RaBitQ-nl316-np25-rf0 (query)                        233.29        57.58       290.87       0.6995          1.0319            1.0306         3.06
IVF-RaBitQ-nl316-np15-rf10 (query)                       233.29       129.66       362.95       0.9996          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np15-rf20 (query)                       233.29       200.34       433.63       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np15-rfB (query)                        233.29        86.84       320.13       0.9963          1.0001            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf10 (query)                       233.29       128.02       361.31       0.9996          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf20 (query)                       233.29       200.79       434.08       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rfB (query)                        233.29        87.40       320.69       0.9963          1.0001            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf10 (query)                       233.29       133.00       366.29       0.9996          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf20 (query)                       233.29       205.62       438.91       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rfB (query)                        233.29        91.52       324.81       0.9963          1.0001            1.0000         3.06
IVF-RaBitQ-nl316 (self)                                  233.29       266.96       500.25       0.9960          1.0001            1.0000         3.06
IVF-RaBitQ-nl223-ex2-np14 (query)                        215.78        96.83       312.61       0.9249          1.0016            1.0012         6.27
IVF-RaBitQ-nl223-ex2-np21 (query)                        215.78        97.75       313.54       0.9249          1.0016            1.0012         6.27
IVF-RaBitQ-nl223-ex4-np14 (query)                        226.58       103.72       330.30       0.9733          1.0002            1.0000         9.32
IVF-RaBitQ-nl223-ex4-np21 (query)                        226.58       106.90       333.48       0.9733          1.0002            1.0000         9.32
IVF-RaBitQ-nl223-ex8-np14 (query)                        252.92        93.88       346.80       0.9915          1.0001            1.0000        15.42
IVF-RaBitQ-nl223-ex8-np21 (query)                        252.92        98.55       351.47       0.9915          1.0001            1.0000        15.42
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 512D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.07     1_345.95     1_414.02       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.07     4_600.76     4_668.83       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                             413.51       131.76       545.27       0.6581          1.0282            1.0278         4.35
ExhaustiveRaBitQ-rf5 (query)                             413.51       202.67       616.17       0.9782          1.0005            1.0000         4.35
ExhaustiveRaBitQ-rf10 (query)                            413.51       263.47       676.97       0.9981          1.0000            1.0000         4.35
ExhaustiveRaBitQ-rf20 (query)                            413.51       390.64       804.15       0.9999          1.0000            1.0000         4.35
ExhaustiveRaBitQ-rfB (query)                             413.51       234.25       647.76       0.9902          1.0002            1.0000         4.35
ExhaustiveRaBitQ (self)                                  413.51       811.61     1_225.12       0.9985          1.0000            1.0000         4.35
IVF-RaBitQ-nl158-np7-rf0 (query)                         523.77        83.28       607.06       0.6616          1.0278            1.0276         4.59
IVF-RaBitQ-nl158-np12-rf0 (query)                        523.77        89.31       613.08       0.6616          1.0278            1.0276         4.59
IVF-RaBitQ-nl158-np17-rf0 (query)                        523.77        96.93       620.71       0.6616          1.0278            1.0276         4.59
IVF-RaBitQ-nl158-np7-rf10 (query)                        523.77       204.66       728.43       0.9981          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np7-rf20 (query)                        523.77       316.57       840.34       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np7-rfB (query)                         523.77       153.57       677.35       0.9901          1.0002            1.0000         4.59
IVF-RaBitQ-nl158-np12-rf10 (query)                       523.77       204.77       728.54       0.9981          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np12-rf20 (query)                       523.77       315.93       839.71       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np12-rfB (query)                        523.77       158.38       682.16       0.9901          1.0002            1.0000         4.59
IVF-RaBitQ-nl158-np17-rf10 (query)                       523.77       206.17       729.95       0.9981          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np17-rf20 (query)                       523.77       322.26       846.04       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np17-rfB (query)                        523.77       164.19       687.96       0.9901          1.0002            1.0000         4.59
IVF-RaBitQ-nl158 (self)                                  523.77       451.39       975.17       0.9890          1.0003            1.0000         4.59
IVF-RaBitQ-nl223-np11-rf0 (query)                        410.63        88.06       498.68       0.6631          1.0275            1.0275         4.88
IVF-RaBitQ-nl223-np14-rf0 (query)                        410.63        90.66       501.29       0.6631          1.0275            1.0275         4.88
IVF-RaBitQ-nl223-np21-rf0 (query)                        410.63        94.40       505.03       0.6631          1.0275            1.0275         4.88
IVF-RaBitQ-nl223-np11-rf10 (query)                       410.63       204.85       615.48       0.9982          1.0000            1.0000         4.88
IVF-RaBitQ-nl223-np11-rf20 (query)                       410.63       313.71       724.34       1.0000          1.0000            1.0000         4.88
IVF-RaBitQ-nl223-np11-rfB (query)                        410.63       157.38       568.01       0.9901          1.0002            1.0000         4.88
IVF-RaBitQ-nl223-np14-rf10 (query)                       410.63       209.55       620.17       0.9982          1.0000            1.0000         4.88
IVF-RaBitQ-nl223-np14-rf20 (query)                       410.63       315.88       726.51       1.0000          1.0000            1.0000         4.88
IVF-RaBitQ-nl223-np14-rfB (query)                        410.63       160.85       571.48       0.9901          1.0002            1.0000         4.88
IVF-RaBitQ-nl223-np21-rf10 (query)                       410.63       209.74       620.37       0.9982          1.0000            1.0000         4.88
IVF-RaBitQ-nl223-np21-rf20 (query)                       410.63       324.73       735.36       1.0000          1.0000            1.0000         4.88
IVF-RaBitQ-nl223-np21-rfB (query)                        410.63       164.84       575.47       0.9901          1.0002            1.0000         4.88
IVF-RaBitQ-nl223 (self)                                  410.63       466.21       876.84       0.9890          1.0003            1.0000         4.88
IVF-RaBitQ-nl316-np15-rf0 (query)                        414.65        95.12       509.78       0.6662          1.0269            1.0267         5.36
IVF-RaBitQ-nl316-np17-rf0 (query)                        414.65        97.10       511.75       0.6662          1.0269            1.0267         5.36
IVF-RaBitQ-nl316-np25-rf0 (query)                        414.65       100.65       515.30       0.6662          1.0269            1.0267         5.36
IVF-RaBitQ-nl316-np15-rf10 (query)                       414.65       211.22       625.87       0.9982          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np15-rf20 (query)                       414.65       317.13       731.78       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np15-rfB (query)                        414.65       163.10       577.75       0.9906          1.0002            1.0000         5.36
IVF-RaBitQ-nl316-np17-rf10 (query)                       414.65       212.48       627.13       0.9982          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np17-rf20 (query)                       414.65       325.03       739.68       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np17-rfB (query)                        414.65       164.95       579.60       0.9906          1.0002            1.0000         5.36
IVF-RaBitQ-nl316-np25-rf10 (query)                       414.65       218.00       632.65       0.9982          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np25-rf20 (query)                       414.65       326.45       741.10       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np25-rfB (query)                        414.65       169.52       584.17       0.9906          1.0002            1.0000         5.36
IVF-RaBitQ-nl316 (self)                                  414.65       480.83       895.49       0.9893          1.0003            1.0000         5.36
IVF-RaBitQ-nl223-ex2-np14 (query)                        407.84       173.45       581.29       0.9257          1.0011            1.0008        11.37
IVF-RaBitQ-nl223-ex2-np21 (query)                        407.84       179.45       587.29       0.9257          1.0011            1.0008        11.37
IVF-RaBitQ-nl223-ex4-np14 (query)                        422.98       193.35       616.33       0.9699          1.0003            1.0000        17.47
IVF-RaBitQ-nl223-ex4-np21 (query)                        422.98       200.01       622.99       0.9699          1.0003            1.0000        17.47
IVF-RaBitQ-nl223-ex8-np14 (query)                        476.56       179.57       656.12       0.9867          1.0002            1.0000        29.68
IVF-RaBitQ-nl223-ex8-np21 (query)                        476.56       183.34       659.89       0.9867          1.0002            1.0000        29.68
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 768D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       100.26     1_949.88     2_050.15       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.26     6_545.14     6_645.40       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                             582.27       167.20       749.48       0.5777          1.0323            1.0331         6.17
ExhaustiveRaBitQ-rf5 (query)                             582.27       243.33       825.61       0.9286          1.0016            1.0006         6.17
ExhaustiveRaBitQ-rf10 (query)                            582.27       340.68       922.96       0.9841          1.0003            1.0000         6.17
ExhaustiveRaBitQ-rf20 (query)                            582.27       473.01     1_055.29       0.9987          1.0000            1.0000         6.17
ExhaustiveRaBitQ-rfB (query)                             582.27       351.03       933.30       0.9772          1.0004            1.0000         6.17
ExhaustiveRaBitQ (self)                                  582.27       983.35     1_565.62       0.9852          1.0003            1.0000         6.17
IVF-RaBitQ-nl158-np7-rf0 (query)                         680.22       104.67       784.89       0.5836          1.0314            1.0324         6.50
IVF-RaBitQ-nl158-np12-rf0 (query)                        680.22       110.22       790.44       0.5836          1.0314            1.0324         6.50
IVF-RaBitQ-nl158-np17-rf0 (query)                        680.22       115.52       795.74       0.5836          1.0314            1.0324         6.50
IVF-RaBitQ-nl158-np7-rf10 (query)                        680.22       246.96       927.18       0.9852          1.0003            1.0000         6.50
IVF-RaBitQ-nl158-np7-rf20 (query)                        680.22       379.25     1_059.47       0.9987          1.0000            1.0000         6.50
IVF-RaBitQ-nl158-np7-rfB (query)                         680.22       217.79       898.01       0.9775          1.0004            1.0000         6.50
IVF-RaBitQ-nl158-np12-rf10 (query)                       680.22       245.95       926.17       0.9852          1.0003            1.0000         6.50
IVF-RaBitQ-nl158-np12-rf20 (query)                       680.22       385.12     1_065.34       0.9987          1.0000            1.0000         6.50
IVF-RaBitQ-nl158-np12-rfB (query)                        680.22       226.98       907.20       0.9775          1.0004            1.0000         6.50
IVF-RaBitQ-nl158-np17-rf10 (query)                       680.22       253.40       933.63       0.9852          1.0003            1.0000         6.50
IVF-RaBitQ-nl158-np17-rf20 (query)                       680.22       394.56     1_074.78       0.9987          1.0000            1.0000         6.50
IVF-RaBitQ-nl158-np17-rfB (query)                        680.22       230.72       910.95       0.9775          1.0004            1.0000         6.50
IVF-RaBitQ-nl158 (self)                                  680.22       658.45     1_338.68       0.9755          1.0005            1.0000         6.50
IVF-RaBitQ-nl223-np11-rf0 (query)                        518.81       114.18       632.99       0.5855          1.0312            1.0322         6.94
IVF-RaBitQ-nl223-np14-rf0 (query)                        518.81       114.60       633.41       0.5855          1.0312            1.0322         6.94
IVF-RaBitQ-nl223-np21-rf0 (query)                        518.81       120.75       639.56       0.5855          1.0312            1.0322         6.94
IVF-RaBitQ-nl223-np11-rf10 (query)                       518.81       255.65       774.46       0.9855          1.0002            1.0000         6.94
IVF-RaBitQ-nl223-np11-rf20 (query)                       518.81       384.78       903.59       0.9988          1.0000            1.0000         6.94
IVF-RaBitQ-nl223-np11-rfB (query)                        518.81       224.54       743.35       0.9771          1.0004            1.0000         6.94
IVF-RaBitQ-nl223-np14-rf10 (query)                       518.81       253.93       772.74       0.9855          1.0002            1.0000         6.94
IVF-RaBitQ-nl223-np14-rf20 (query)                       518.81       390.53       909.34       0.9988          1.0000            1.0000         6.94
IVF-RaBitQ-nl223-np14-rfB (query)                        518.81       230.78       749.59       0.9771          1.0004            1.0000         6.94
IVF-RaBitQ-nl223-np21-rf10 (query)                       518.81       261.76       780.57       0.9855          1.0002            1.0000         6.94
IVF-RaBitQ-nl223-np21-rf20 (query)                       518.81       398.32       917.13       0.9988          1.0000            1.0000         6.94
IVF-RaBitQ-nl223-np21-rfB (query)                        518.81       234.45       753.26       0.9771          1.0004            1.0000         6.94
IVF-RaBitQ-nl223 (self)                                  518.81       702.01     1_220.82       0.9758          1.0005            1.0000         6.94
IVF-RaBitQ-nl316-np15-rf0 (query)                        571.86       124.36       696.22       0.5876          1.0308            1.0321         7.65
IVF-RaBitQ-nl316-np17-rf0 (query)                        571.86       124.17       696.03       0.5876          1.0308            1.0321         7.65
IVF-RaBitQ-nl316-np25-rf0 (query)                        571.86       130.96       702.83       0.5876          1.0308            1.0321         7.65
IVF-RaBitQ-nl316-np15-rf10 (query)                       571.86       267.54       839.40       0.9861          1.0002            1.0000         7.65
IVF-RaBitQ-nl316-np15-rf20 (query)                       571.86       399.72       971.58       0.9989          1.0000            1.0000         7.65
IVF-RaBitQ-nl316-np15-rfB (query)                        571.86       236.60       808.46       0.9774          1.0004            1.0000         7.65
IVF-RaBitQ-nl316-np17-rf10 (query)                       571.86       267.56       839.43       0.9861          1.0002            1.0000         7.65
IVF-RaBitQ-nl316-np17-rf20 (query)                       571.86       401.60       973.47       0.9989          1.0000            1.0000         7.65
IVF-RaBitQ-nl316-np17-rfB (query)                        571.86       246.34       818.20       0.9774          1.0004            1.0000         7.65
IVF-RaBitQ-nl316-np25-rf10 (query)                       571.86       275.32       847.19       0.9861          1.0002            1.0000         7.65
IVF-RaBitQ-nl316-np25-rf20 (query)                       571.86       416.12       987.98       0.9989          1.0000            1.0000         7.65
IVF-RaBitQ-nl316-np25-rfB (query)                        571.86       243.28       815.14       0.9774          1.0004            1.0000         7.65
IVF-RaBitQ-nl316 (self)                                  571.86       709.92     1_281.79       0.9760          1.0005            1.0000         7.65
IVF-RaBitQ-nl223-ex2-np14 (query)                        560.45       263.81       824.26       0.9138          1.0012            1.0008        16.49
IVF-RaBitQ-nl223-ex2-np21 (query)                        560.45       273.84       834.29       0.9138          1.0012            1.0008        16.49
IVF-RaBitQ-nl223-ex4-np14 (query)                        611.61       297.76       909.38       0.9590          1.0005            1.0000        25.65
IVF-RaBitQ-nl223-ex4-np21 (query)                        611.61       302.41       914.03       0.9590          1.0005            1.0000        25.65
IVF-RaBitQ-nl223-ex8-np14 (query)                        670.64       269.86       940.51       0.9748          1.0004            1.0000        43.96
IVF-RaBitQ-nl223-ex8-np21 (query)                        670.64       281.82       952.46       0.9748          1.0004            1.0000        43.96
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Cell embeddings

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 256D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.86       725.87       758.73       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.86     2_414.89     2_447.75       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             395.84       100.98       496.82       0.8703          1.0284            1.0233         2.57
ExhaustiveRaBitQ-rf5 (query)                             395.84       169.06       564.90       1.0000          1.0000            1.0000         2.57
ExhaustiveRaBitQ-rf10 (query)                            395.84       230.96       626.80       1.0000          1.0000            1.0000         2.57
ExhaustiveRaBitQ-rf20 (query)                            395.84       340.84       736.68       1.0000          1.0000            1.0000         2.57
ExhaustiveRaBitQ-rfB (query)                             395.84       142.15       537.99       1.0000          1.0000            1.0000         2.57
ExhaustiveRaBitQ (self)                                  395.84       723.25     1_119.09       1.0000          1.0000            1.0000         2.57
IVF-RaBitQ-nl158-np7-rf0 (query)                         254.60        50.06       304.65       0.8746          1.0263            1.0216         2.68
IVF-RaBitQ-nl158-np12-rf0 (query)                        254.60        59.66       314.25       0.8752          1.0259            1.0214         2.68
IVF-RaBitQ-nl158-np17-rf0 (query)                        254.60        65.53       320.13       0.8753          1.0259            1.0213         2.68
IVF-RaBitQ-nl158-np7-rf10 (query)                        254.60       137.72       392.32       0.9972          1.0006            1.0000         2.68
IVF-RaBitQ-nl158-np7-rf20 (query)                        254.60       216.23       470.83       0.9972          1.0006            1.0000         2.68
IVF-RaBitQ-nl158-np7-rfB (query)                         254.60        79.17       333.77       0.9972          1.0006            1.0000         2.68
IVF-RaBitQ-nl158-np12-rf10 (query)                       254.60       144.50       399.09       0.9998          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np12-rf20 (query)                       254.60       222.26       476.86       0.9998          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np12-rfB (query)                        254.60        82.61       337.21       0.9998          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rf10 (query)                       254.60       154.34       408.93       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rf20 (query)                       254.60       231.23       485.83       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rfB (query)                        254.60        89.05       343.65       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158 (self)                                  254.60       267.60       522.20       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl223-np11-rf0 (query)                        200.34        53.07       253.40       0.8826          1.0229            1.0188         2.83
IVF-RaBitQ-nl223-np14-rf0 (query)                        200.34        57.20       257.53       0.8827          1.0228            1.0187         2.83
IVF-RaBitQ-nl223-np21-rf0 (query)                        200.34        69.35       269.69       0.8828          1.0228            1.0187         2.83
IVF-RaBitQ-nl223-np11-rf10 (query)                       200.34       134.71       335.04       0.9994          1.0001            1.0000         2.83
IVF-RaBitQ-nl223-np11-rf20 (query)                       200.34       208.61       408.94       0.9994          1.0001            1.0000         2.83
IVF-RaBitQ-nl223-np11-rfB (query)                        200.34        74.69       275.03       0.9993          1.0001            1.0000         2.83
IVF-RaBitQ-nl223-np14-rf10 (query)                       200.34       137.98       338.31       0.9998          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np14-rf20 (query)                       200.34       214.97       415.31       0.9998          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np14-rfB (query)                        200.34        80.16       280.49       0.9998          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np21-rf10 (query)                       200.34       145.97       346.31       1.0000          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np21-rf20 (query)                       200.34       228.09       428.43       1.0000          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np21-rfB (query)                        200.34        86.58       286.92       1.0000          1.0000            1.0000         2.83
IVF-RaBitQ-nl223 (self)                                  200.34       256.34       456.68       1.0000          1.0000            1.0000         2.83
IVF-RaBitQ-nl316-np15-rf0 (query)                        228.67        55.61       284.28       0.8900          1.0198            1.0162         3.06
IVF-RaBitQ-nl316-np17-rf0 (query)                        228.67        59.37       288.04       0.8900          1.0198            1.0162         3.06
IVF-RaBitQ-nl316-np25-rf0 (query)                        228.67        64.04       292.71       0.8900          1.0198            1.0162         3.06
IVF-RaBitQ-nl316-np15-rf10 (query)                       228.67       136.83       365.50       0.9997          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np15-rf20 (query)                       228.67       212.44       441.11       0.9997          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np15-rfB (query)                        228.67        75.71       304.38       0.9997          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf10 (query)                       228.67       135.89       364.56       0.9998          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf20 (query)                       228.67       217.06       445.73       0.9998          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rfB (query)                        228.67        77.24       305.91       0.9998          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf10 (query)                       228.67       143.47       372.14       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf20 (query)                       228.67       223.22       451.89       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rfB (query)                        228.67        84.40       313.07       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316 (self)                                  228.67       248.94       477.61       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl223-ex2-np14 (query)                        216.31        86.19       302.50       0.9661          1.0014            1.0000         6.27
IVF-RaBitQ-nl223-ex2-np21 (query)                        216.31        92.05       308.35       0.9661          1.0014            1.0000         6.27
IVF-RaBitQ-nl223-ex4-np14 (query)                        221.38        88.99       310.37       0.9886          1.0002            1.0000         9.32
IVF-RaBitQ-nl223-ex4-np21 (query)                        221.38        96.93       318.32       0.9887          1.0001            1.0000         9.32
IVF-RaBitQ-nl223-ex8-np14 (query)                        252.01        85.47       337.48       0.9977          1.0000            1.0000        15.42
IVF-RaBitQ-nl223-ex8-np21 (query)                        252.01        93.09       345.10       0.9978          1.0000            1.0000        15.42
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 512D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.81     1_362.58     1_430.40       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         67.81     4_642.55     4_710.36       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                             601.05       157.94       758.99       0.9088          1.0131            1.0099         4.36
ExhaustiveRaBitQ-rf5 (query)                             601.05       233.71       834.77       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ-rf10 (query)                            601.05       307.88       908.93       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ-rf20 (query)                            601.05       449.24     1_050.29       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ-rfB (query)                             601.05       228.54       829.59       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ (self)                                  601.05       967.18     1_568.24       1.0000          1.0000            1.0000         4.36
IVF-RaBitQ-nl158-np7-rf0 (query)                         503.60        85.60       589.19       0.9165          1.0109            1.0081         4.57
IVF-RaBitQ-nl158-np12-rf0 (query)                        503.60        99.84       603.44       0.9170          1.0107            1.0079         4.57
IVF-RaBitQ-nl158-np17-rf0 (query)                        503.60       111.60       615.19       0.9170          1.0107            1.0079         4.57
IVF-RaBitQ-nl158-np7-rf10 (query)                        503.60       201.71       705.31       0.9986          1.0003            1.0000         4.57
IVF-RaBitQ-nl158-np7-rf20 (query)                        503.60       313.52       817.12       0.9986          1.0003            1.0000         4.57
IVF-RaBitQ-nl158-np7-rfB (query)                         503.60       121.35       624.95       0.9986          1.0003            1.0000         4.57
IVF-RaBitQ-nl158-np12-rf10 (query)                       503.60       214.62       718.22       0.9999          1.0000            1.0000         4.57
IVF-RaBitQ-nl158-np12-rf20 (query)                       503.60       327.46       831.05       0.9999          1.0000            1.0000         4.57
IVF-RaBitQ-nl158-np12-rfB (query)                        503.60       132.49       636.08       0.9999          1.0000            1.0000         4.57
IVF-RaBitQ-nl158-np17-rf10 (query)                       503.60       233.18       736.78       1.0000          1.0000            1.0000         4.57
IVF-RaBitQ-nl158-np17-rf20 (query)                       503.60       342.87       846.47       1.0000          1.0000            1.0000         4.57
IVF-RaBitQ-nl158-np17-rfB (query)                        503.60       153.55       657.14       1.0000          1.0000            1.0000         4.57
IVF-RaBitQ-nl158 (self)                                  503.60       414.96       918.56       1.0000          1.0000            1.0000         4.57
IVF-RaBitQ-nl223-np11-rf0 (query)                        773.02        94.17       867.19       0.9215          1.0093            1.0069         4.90
IVF-RaBitQ-nl223-np14-rf0 (query)                        773.02       101.13       874.15       0.9216          1.0093            1.0068         4.90
IVF-RaBitQ-nl223-np21-rf0 (query)                        773.02       112.30       885.32       0.9216          1.0093            1.0068         4.90
IVF-RaBitQ-nl223-np11-rf10 (query)                       773.02       207.62       980.64       0.9997          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np11-rf20 (query)                       773.02       317.08     1_090.10       0.9997          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np11-rfB (query)                        773.02       126.40       899.43       0.9997          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np14-rf10 (query)                       773.02       214.59       987.61       0.9999          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np14-rf20 (query)                       773.02       328.80     1_101.83       0.9999          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np14-rfB (query)                        773.02       133.26       906.29       0.9999          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np21-rf10 (query)                       773.02       226.17       999.19       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np21-rf20 (query)                       773.02       337.90     1_110.92       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np21-rfB (query)                        773.02       141.08       914.10       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223 (self)                                  773.02       406.50     1_179.52       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl316-np15-rf0 (query)                        440.93        97.93       538.86       0.9263          1.0081            1.0058         5.36
IVF-RaBitQ-nl316-np17-rf0 (query)                        440.93       100.44       541.37       0.9263          1.0081            1.0058         5.36
IVF-RaBitQ-nl316-np25-rf0 (query)                        440.93       111.28       552.21       0.9263          1.0081            1.0058         5.36
IVF-RaBitQ-nl316-np15-rf10 (query)                       440.93       212.14       653.07       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np15-rf20 (query)                       440.93       323.51       764.44       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np15-rfB (query)                        440.93       130.42       571.35       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np17-rf10 (query)                       440.93       214.95       655.87       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np17-rf20 (query)                       440.93       327.57       768.50       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np17-rfB (query)                        440.93       133.81       574.74       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np25-rf10 (query)                       440.93       233.26       674.19       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np25-rf20 (query)                       440.93       359.08       800.01       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np25-rfB (query)                        440.93       146.34       587.27       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl316 (self)                                  440.93       422.28       863.21       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl223-ex2-np14 (query)                        408.96       139.98       548.95       0.9761          1.0007            1.0000        11.39
IVF-RaBitQ-nl223-ex2-np21 (query)                        408.96       151.56       560.52       0.9761          1.0007            1.0000        11.39
IVF-RaBitQ-nl223-ex4-np14 (query)                        423.57       150.31       573.88       0.9921          1.0001            1.0000        17.49
IVF-RaBitQ-nl223-ex4-np21 (query)                        423.57       160.97       584.54       0.9921          1.0001            1.0000        17.49
IVF-RaBitQ-nl223-ex8-np14 (query)                        482.59       144.16       626.75       0.9988          1.0000            1.0000        29.70
IVF-RaBitQ-nl223-ex8-np21 (query)                        482.59       153.82       636.41       0.9989          1.0000            1.0000        29.70
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 768D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        99.96     1_938.03     2_037.99       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.96     6_823.31     6_923.27       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                             868.33       201.70     1_070.03       0.9143          1.0117            1.0085         6.16
ExhaustiveRaBitQ-rf5 (query)                             868.33       294.47     1_162.80       1.0000          1.0000            1.0000         6.16
ExhaustiveRaBitQ-rf10 (query)                            868.33       374.73     1_243.06       1.0000          1.0000            1.0000         6.16
ExhaustiveRaBitQ-rf20 (query)                            868.33       538.32     1_406.65       1.0000          1.0000            1.0000         6.16
ExhaustiveRaBitQ-rfB (query)                             868.33       313.54     1_181.87       0.9999          1.0000            1.0000         6.16
ExhaustiveRaBitQ (self)                                  868.33     1_187.89     2_056.22       1.0000          1.0000            1.0000         6.16
IVF-RaBitQ-nl158-np7-rf0 (query)                         686.30       109.43       795.73       0.9182          1.0105            1.0075         6.48
IVF-RaBitQ-nl158-np12-rf0 (query)                        686.30       125.38       811.68       0.9184          1.0104            1.0075         6.48
IVF-RaBitQ-nl158-np17-rf0 (query)                        686.30       140.40       826.71       0.9184          1.0104            1.0075         6.48
IVF-RaBitQ-nl158-np7-rf10 (query)                        686.30       248.60       934.90       0.9994          1.0001            1.0000         6.48
IVF-RaBitQ-nl158-np7-rf20 (query)                        686.30       375.10     1_061.41       0.9994          1.0001            1.0000         6.48
IVF-RaBitQ-nl158-np7-rfB (query)                         686.30       156.84       843.14       0.9994          1.0001            1.0000         6.48
IVF-RaBitQ-nl158-np12-rf10 (query)                       686.30       265.68       951.99       1.0000          1.0000            1.0000         6.48
IVF-RaBitQ-nl158-np12-rf20 (query)                       686.30       398.89     1_085.19       1.0000          1.0000            1.0000         6.48
IVF-RaBitQ-nl158-np12-rfB (query)                        686.30       170.79       857.10       0.9999          1.0000            1.0000         6.48
IVF-RaBitQ-nl158-np17-rf10 (query)                       686.30       277.36       963.66       1.0000          1.0000            1.0000         6.48
IVF-RaBitQ-nl158-np17-rf20 (query)                       686.30       417.66     1_103.97       1.0000          1.0000            1.0000         6.48
IVF-RaBitQ-nl158-np17-rfB (query)                        686.30       186.39       872.69       0.9999          1.0000            1.0000         6.48
IVF-RaBitQ-nl158 (self)                                  686.30       533.20     1_219.50       0.9999          1.0000            1.0000         6.48
IVF-RaBitQ-nl223-np11-rf0 (query)                        514.80       122.18       636.98       0.9222          1.0095            1.0066         6.95
IVF-RaBitQ-nl223-np14-rf0 (query)                        514.80       130.23       645.03       0.9223          1.0094            1.0066         6.95
IVF-RaBitQ-nl223-np21-rf0 (query)                        514.80       142.04       656.83       0.9223          1.0094            1.0066         6.95
IVF-RaBitQ-nl223-np11-rf10 (query)                       514.80       256.49       771.29       0.9998          1.0000            1.0000         6.95
IVF-RaBitQ-nl223-np11-rf20 (query)                       514.80       389.33       904.13       0.9998          1.0000            1.0000         6.95
IVF-RaBitQ-nl223-np11-rfB (query)                        514.80       161.48       676.27       0.9997          1.0000            1.0000         6.95
IVF-RaBitQ-nl223-np14-rf10 (query)                       514.80       263.47       778.27       1.0000          1.0000            1.0000         6.95
IVF-RaBitQ-nl223-np14-rf20 (query)                       514.80       400.00       914.80       1.0000          1.0000            1.0000         6.95
IVF-RaBitQ-nl223-np14-rfB (query)                        514.80       169.58       684.38       0.9999          1.0000            1.0000         6.95
IVF-RaBitQ-nl223-np21-rf10 (query)                       514.80       280.95       795.75       1.0000          1.0000            1.0000         6.95
IVF-RaBitQ-nl223-np21-rf20 (query)                       514.80       423.82       938.62       1.0000          1.0000            1.0000         6.95
IVF-RaBitQ-nl223-np21-rfB (query)                        514.80       185.19       699.98       0.9999          1.0000            1.0000         6.95
IVF-RaBitQ-nl223 (self)                                  514.80       533.62     1_048.42       1.0000          1.0000            1.0000         6.95
IVF-RaBitQ-nl316-np15-rf0 (query)                        561.31       129.57       690.88       0.9259          1.0083            1.0058         7.66
IVF-RaBitQ-nl316-np17-rf0 (query)                        561.31       132.94       694.25       0.9259          1.0083            1.0058         7.66
IVF-RaBitQ-nl316-np25-rf0 (query)                        561.31       147.04       708.35       0.9259          1.0083            1.0058         7.66
IVF-RaBitQ-nl316-np15-rf10 (query)                       561.31       280.26       841.57       0.9999          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np15-rf20 (query)                       561.31       401.67       962.98       0.9999          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np15-rfB (query)                        561.31       168.84       730.15       0.9999          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np17-rf10 (query)                       561.31       270.34       831.65       1.0000          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np17-rf20 (query)                       561.31       408.45       969.76       1.0000          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np17-rfB (query)                        561.31       171.97       733.28       0.9999          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np25-rf10 (query)                       561.31       287.88       849.19       1.0000          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np25-rf20 (query)                       561.31       438.53       999.84       1.0000          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np25-rfB (query)                        561.31       189.05       750.36       0.9999          1.0000            1.0000         7.66
IVF-RaBitQ-nl316 (self)                                  561.31       542.59     1_103.90       0.9999          1.0000            1.0000         7.66
IVF-RaBitQ-nl223-ex2-np14 (query)                        559.73       185.43       745.16       0.9645          1.0018            1.0000        16.49
IVF-RaBitQ-nl223-ex2-np21 (query)                        559.73       204.09       763.82       0.9645          1.0018            1.0000        16.49
IVF-RaBitQ-nl223-ex4-np14 (query)                        586.01       200.39       786.40       0.9777          1.0007            1.0000        25.64
IVF-RaBitQ-nl223-ex4-np21 (query)                        586.01       218.39       804.40       0.9777          1.0007            1.0000        25.64
IVF-RaBitQ-nl223-ex8-np14 (query)                        746.25       189.78       936.04       0.9835          1.0004            1.0000        43.95
IVF-RaBitQ-nl223-ex8-np21 (query)                        746.25       206.03       952.28       0.9835          1.0004            1.0000        43.95
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
to force exact degree; here `VamanaIndex` builds the topology and what is kept
from the paper is the storage layout and the estimator. Vamana's pruning leaves
slots empty, and an empty fast-scan lane costs the same as a full one, so the
build fills the empty slots of every partly filled batch with the nearest 2-hop
neighbours, and the query stops at the first batch that is empty throughout.

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
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 256D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.94       701.52       734.46       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.94     2_236.82     2_269.75       1.0000          1.0000            1.0000        48.83
QG-d32-l64-ef15 (query)                                2_568.87       293.59     2_862.46       0.9916          1.0022            1.0000       116.16
QG-d32-l64-ef30 (query)                                2_568.87       436.52     3_005.39       0.9959          1.0016            1.0000       116.16
QG-d32-l64-ef60 (query)                                2_568.87       629.96     3_198.83       0.9975          1.0015            1.0000       116.16
QG-d32-l64-ef120 (query)                               2_568.87       909.35     3_478.22       0.9981          1.0013            1.0000       116.16
QG-d32-l64 (self)                                      2_568.87     2_012.12     4_580.99       0.9975          1.0015            1.0000       116.16
QG-d32-l128-ef15 (query)                               2_868.04       291.43     3_159.47       0.9916          1.0022            1.0000       116.16
QG-d32-l128-ef30 (query)                               2_868.04       417.07     3_285.11       0.9960          1.0017            1.0000       116.16
QG-d32-l128-ef60 (query)                               2_868.04       603.06     3_471.10       0.9977          1.0015            1.0000       116.16
QG-d32-l128-ef120 (query)                              2_868.04       889.76     3_757.80       0.9984          1.0012            1.0000       116.16
QG-d32-l128 (self)                                     2_868.04     1_974.32     4_842.35       0.9977          1.0015            1.0000       116.16
QG-d64-l64-ef15 (query)                                8_782.53       683.02     9_465.55       0.9995          1.0002            1.0000       183.30
QG-d64-l64-ef30 (query)                                8_782.53       880.11     9_662.64       0.9998          1.0002            1.0000       183.30
QG-d64-l64-ef60 (query)                                8_782.53     1_249.99    10_032.52       0.9998          1.0002            1.0000       183.30
QG-d64-l64-ef120 (query)                               8_782.53     1_763.50    10_546.03       0.9998          1.0001            1.0000       183.30
QG-d64-l64 (self)                                      8_782.53     4_090.42    12_872.95       0.9998          1.0002            1.0000       183.30
QG-d64-l128-ef15 (query)                               9_566.12       651.35    10_217.47       0.9996          1.0002            1.0000       183.30
QG-d64-l128-ef30 (query)                               9_566.12       888.85    10_454.97       0.9998          1.0002            1.0000       183.30
QG-d64-l128-ef60 (query)                               9_566.12     1_235.31    10_801.43       0.9998          1.0001            1.0000       183.30
QG-d64-l128-ef120 (query)                              9_566.12     1_778.13    11_344.25       0.9999          1.0001            1.0000       183.30
QG-d64-l128 (self)                                     9_566.12     4_081.64    13_647.76       0.9998          1.0002            1.0000       183.30
HnswRaBitQ-m16-ex1-ef15 (query)                        1_189.67        89.04     1_278.72       0.7517         58.5476            1.0084        10.85
HnswRaBitQ-m16-ex1-ef30 (query)                        1_189.67       113.83     1_303.50       0.7759          1.6750            1.0079        10.85
HnswRaBitQ-m16-ex1-ef60 (query)                        1_189.67       140.00     1_329.67       0.7784          1.0096            1.0079        10.85
HnswRaBitQ-m16-ex1-ef120 (query)                       1_189.67       182.63     1_372.30       0.7786          1.0092            1.0080        10.85
HnswRaBitQ-m16-ex1 (self)                              1_189.67     4_775.23     5_964.90       0.6971          1.0177            1.0160        10.85
HnswRaBitQ-m16-ex3-ef15 (query)                        1_267.55        96.93     1_364.48       0.8949          2.9114            1.0010        13.90
HnswRaBitQ-m16-ex3-ef30 (query)                        1_267.55       121.08     1_388.63       0.9200          1.0567            1.0007        13.90
HnswRaBitQ-m16-ex3-ef60 (query)                        1_267.55       150.20     1_417.75       0.9263          1.0024            1.0006        13.90
HnswRaBitQ-m16-ex3-ef120 (query)                       1_267.55       191.74     1_459.30       0.9278          1.0022            1.0005        13.90
HnswRaBitQ-m16-ex3 (self)                              1_267.55     4_809.00     6_076.56       0.9029          1.0029            1.0012        13.90
HnswRaBitQ-m16-ex5-ef15 (query)                        1_379.83       113.20     1_493.03       0.9371          1.1071            1.0001        16.95
HnswRaBitQ-m16-ex5-ef30 (query)                        1_379.83       134.90     1_514.73       0.9634          1.0098            1.0000        16.95
HnswRaBitQ-m16-ex5-ef60 (query)                        1_379.83       168.25     1_548.09       0.9735          1.0021            1.0000        16.95
HnswRaBitQ-m16-ex5-ef120 (query)                       1_379.83       212.06     1_591.89       0.9761          1.0006            1.0000        16.95
HnswRaBitQ-m16-ex5 (self)                              1_379.83     4_892.29     6_272.12       0.9692          1.0025            1.0000        16.95
HnswRaBitQ-m16-ex8-ef15 (query)                        2_077.65       101.72     2_179.37       0.9076        101.3743            1.0000        21.53
HnswRaBitQ-m16-ex8-ef30 (query)                        2_077.65       128.58     2_206.24       0.9543         58.5896            1.0000        21.53
HnswRaBitQ-m16-ex8-ef60 (query)                        2_077.65       166.66     2_244.31       0.9873          7.4090            1.0000        21.53
HnswRaBitQ-m16-ex8-ef120 (query)                       2_077.65       213.04     2_290.69       0.9937          1.0006            1.0000        21.53
HnswRaBitQ-m16-ex8 (self)                              2_077.65     4_819.78     6_897.43       0.9928          1.0010            1.0000        21.53
HnswRaBitQ-m32-ex1-ef15 (query)                        1_221.66       100.73     1_322.39       0.7774          1.0190            1.0079        16.95
HnswRaBitQ-m32-ex1-ef30 (query)                        1_221.66       125.23     1_346.89       0.7792          1.0088            1.0079        16.95
HnswRaBitQ-m32-ex1-ef60 (query)                        1_221.66       152.81     1_374.47       0.7793          1.0083            1.0079        16.95
HnswRaBitQ-m32-ex1-ef120 (query)                       1_221.66       195.55     1_417.20       0.7794          1.0082            1.0079        16.95
HnswRaBitQ-m32-ex1 (self)                              1_221.66     6_186.20     7_407.85       0.6972          1.0169            1.0160        16.95
HnswRaBitQ-m32-ex3-ef15 (query)                        1_317.00       111.48     1_428.48       0.9164          1.0088            1.0007        20.00
HnswRaBitQ-m32-ex3-ef30 (query)                        1_317.00       135.42     1_452.42       0.9256          1.0022            1.0006        20.00
HnswRaBitQ-m32-ex3-ef60 (query)                        1_317.00       167.22     1_484.22       0.9281          1.0009            1.0005        20.00
HnswRaBitQ-m32-ex3-ef120 (query)                       1_317.00       211.50     1_528.50       0.9287          1.0008            1.0005        20.00
HnswRaBitQ-m32-ex3 (self)                              1_317.00     6_250.80     7_567.80       0.9043          1.0016            1.0011        20.00
HnswRaBitQ-m32-ex5-ef15 (query)                        1_414.22       121.95     1_536.17       0.9585          1.0044            1.0000        23.06
HnswRaBitQ-m32-ex5-ef30 (query)                        1_414.22       145.59     1_559.81       0.9718          1.0017            1.0000        23.06
HnswRaBitQ-m32-ex5-ef60 (query)                        1_414.22       180.23     1_594.45       0.9759          1.0008            1.0000        23.06
HnswRaBitQ-m32-ex5-ef120 (query)                       1_414.22       223.46     1_637.68       0.9767          1.0002            1.0000        23.06
HnswRaBitQ-m32-ex5 (self)                              1_414.22     6_375.22     7_789.43       0.9711          1.0005            1.0000        23.06
HnswRaBitQ-m32-ex8-ef15 (query)                        2_135.81       114.77     2_250.58       0.9723          1.0087            1.0000        27.63
HnswRaBitQ-m32-ex8-ef30 (query)                        2_135.81       145.06     2_280.87       0.9889          1.0009            1.0000        27.63
HnswRaBitQ-m32-ex8-ef60 (query)                        2_135.81       181.00     2_316.81       0.9935          1.0003            1.0000        27.63
HnswRaBitQ-m32-ex8-ef120 (query)                       2_135.81       231.15     2_366.96       0.9945          1.0001            1.0000        27.63
HnswRaBitQ-m32-ex8 (self)                              2_135.81     5_953.77     8_089.58       0.9952          1.0001            1.0000        27.63
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 512D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.06     1_362.87     1_430.94       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.06     4_560.76     4_628.82       1.0000          1.0000            1.0000        97.66
QG-d32-l64-ef15 (query)                                5_003.10       508.82     5_511.92       0.9879          1.0034            1.0000       213.81
QG-d32-l64-ef30 (query)                                5_003.10       680.29     5_683.39       0.9931          1.0031            1.0000       213.81
QG-d32-l64-ef60 (query)                                5_003.10       914.80     5_917.89       0.9953          1.0030            1.0000       213.81
QG-d32-l64-ef120 (query)                               5_003.10     1_257.18     6_260.27       0.9963          1.0029            1.0000       213.81
QG-d32-l64 (self)                                      5_003.10     2_984.81     7_987.91       0.9952          1.0030            1.0000       213.81
QG-d32-l128-ef15 (query)                               5_625.75       506.98     6_132.73       0.9884          1.0031            1.0000       213.81
QG-d32-l128-ef30 (query)                               5_625.75       682.45     6_308.21       0.9936          1.0028            1.0000       213.81
QG-d32-l128-ef60 (query)                               5_625.75       915.71     6_541.46       0.9957          1.0027            1.0000       213.81
QG-d32-l128-ef120 (query)                              5_625.75     1_245.62     6_871.37       0.9966          1.0026            1.0000       213.81
QG-d32-l128 (self)                                     5_625.75     2_958.36     8_584.12       0.9955          1.0029            1.0000       213.81
QG-d64-l64-ef15 (query)                               18_190.50     1_017.60    19_208.10       0.9980          1.0009            1.0000       329.78
QG-d64-l64-ef30 (query)                               18_190.50     1_305.23    19_495.73       0.9985          1.0009            1.0000       329.78
QG-d64-l64-ef60 (query)                               18_190.50     1_812.55    20_003.05       0.9988          1.0008            1.0000       329.78
QG-d64-l64-ef120 (query)                              18_190.50     2_312.29    20_502.79       0.9989          1.0008            1.0000       329.78
QG-d64-l64 (self)                                     18_190.50     5_650.99    23_841.48       0.9987          1.0009            1.0000       329.78
QG-d64-l128-ef15 (query)                              18_677.00     1_019.44    19_696.44       0.9982          1.0012            1.0000       329.78
QG-d64-l128-ef30 (query)                              18_677.00     1_363.88    20_040.88       0.9986          1.0012            1.0000       329.78
QG-d64-l128-ef60 (query)                              18_677.00     1_715.91    20_392.91       0.9989          1.0010            1.0000       329.78
QG-d64-l128-ef120 (query)                             18_677.00     2_314.70    20_991.70       0.9991          1.0008            1.0000       329.78
QG-d64-l128 (self)                                    18_677.00     5_713.36    24_390.36       0.9989          1.0009            1.0000       329.78
HnswRaBitQ-m16-ex1-ef15 (query)                        2_303.93       170.50     2_474.43       0.7373        104.3188            1.0058        14.15
HnswRaBitQ-m16-ex1-ef30 (query)                        2_303.93       204.66     2_508.60       0.7748          1.0565            1.0053        14.15
HnswRaBitQ-m16-ex1-ef60 (query)                        2_303.93       248.51     2_552.44       0.7792          1.0066            1.0052        14.15
HnswRaBitQ-m16-ex1-ef120 (query)                       2_303.93       302.91     2_606.84       0.7797          1.0057            1.0052        14.15
HnswRaBitQ-m16-ex1 (self)                              2_303.93     9_973.24    12_277.17       0.6968          1.0130            1.0107        14.15
HnswRaBitQ-m16-ex3-ef15 (query)                        2_417.65       183.57     2_601.22       0.8888          1.1601            1.0008        20.25
HnswRaBitQ-m16-ex3-ef30 (query)                        2_417.65       225.84     2_643.50       0.9145          1.0587            1.0005        20.25
HnswRaBitQ-m16-ex3-ef60 (query)                        2_417.65       274.42     2_692.08       0.9248          1.0019            1.0004        20.25
HnswRaBitQ-m16-ex3-ef120 (query)                       2_417.65       336.10     2_753.75       0.9277          1.0008            1.0004        20.25
HnswRaBitQ-m16-ex3 (self)                              2_417.65     9_602.73    12_020.38       0.9010          1.0019            1.0008        20.25
HnswRaBitQ-m16-ex5-ef15 (query)                        2_630.72       191.94     2_822.66       0.8823        132.2438            1.0002        26.36
HnswRaBitQ-m16-ex5-ef30 (query)                        2_630.72       240.74     2_871.46       0.9140        132.2053            1.0000        26.36
HnswRaBitQ-m16-ex5-ef60 (query)                        2_630.72       289.37     2_920.09       0.9264        132.2019            1.0000        26.36
HnswRaBitQ-m16-ex5-ef120 (query)                       2_630.72       379.19     3_009.91       0.9749          1.0004            1.0000        26.36
HnswRaBitQ-m16-ex5 (self)                              2_630.72     9_387.67    12_018.39       0.9212        133.1413            1.0000        26.36
HnswRaBitQ-m16-ex8-ef15 (query)                        4_097.91       190.46     4_288.37       0.9346          1.0139            1.0000        35.51
HnswRaBitQ-m16-ex8-ef30 (query)                        4_097.91       232.10     4_330.01       0.9712          1.0057            1.0000        35.51
HnswRaBitQ-m16-ex8-ef60 (query)                        4_097.91       283.94     4_381.84       0.9885          1.0022            1.0000        35.51
HnswRaBitQ-m16-ex8-ef120 (query)                       4_097.91       347.30     4_445.21       0.9930          1.0015            1.0000        35.51
HnswRaBitQ-m16-ex8 (self)                              4_097.91     9_890.06    13_987.96       0.9891          1.0033            1.0000        35.51
HnswRaBitQ-m32-ex1-ef15 (query)                        2_739.44       212.20     2_951.64       0.7715          1.1860            1.0053        20.25
HnswRaBitQ-m32-ex1-ef30 (query)                        2_739.44       267.31     3_006.75       0.7772          1.0436            1.0052        20.25
HnswRaBitQ-m32-ex1-ef60 (query)                        2_739.44       311.59     3_051.03       0.7785          1.0295            1.0052        20.25
HnswRaBitQ-m32-ex1-ef120 (query)                       2_739.44       342.07     3_081.51       0.7798          1.0055            1.0052        20.25
HnswRaBitQ-m32-ex1 (self)                              2_739.44    13_244.75    15_984.19       0.6956          1.0464            1.0107        20.25
HnswRaBitQ-m32-ex3-ef15 (query)                        2_528.13       215.37     2_743.50       0.9097          1.3915            1.0005        26.36
HnswRaBitQ-m32-ex3-ef30 (query)                        2_528.13       257.79     2_785.92       0.9220          1.2620            1.0004        26.36
HnswRaBitQ-m32-ex3-ef60 (query)                        2_528.13       321.37     2_849.51       0.9271          1.0009            1.0004        26.36
HnswRaBitQ-m32-ex3-ef120 (query)                       2_528.13       369.44     2_897.57       0.9280          1.0006            1.0004        26.36
HnswRaBitQ-m32-ex3 (self)                              2_528.13    12_076.80    14_604.93       0.9025          1.0013            1.0008        26.36
HnswRaBitQ-m32-ex5-ef15 (query)                        2_726.50       231.93     2_958.43       0.9531          1.0132            1.0000        32.46
HnswRaBitQ-m32-ex5-ef30 (query)                        2_726.50       280.56     3_007.05       0.9692          1.0015            1.0000        32.46
HnswRaBitQ-m32-ex5-ef60 (query)                        2_726.50       332.92     3_059.41       0.9745          1.0004            1.0000        32.46
HnswRaBitQ-m32-ex5-ef120 (query)                       2_726.50       393.81     3_120.31       0.9756          1.0002            1.0000        32.46
HnswRaBitQ-m32-ex5 (self)                              2_726.50    12_462.61    15_189.11       0.9691          1.0004            1.0000        32.46
HnswRaBitQ-m32-ex8-ef15 (query)                        4_183.78       226.24     4_410.02       0.9641          1.1004            1.0000        41.62
HnswRaBitQ-m32-ex8-ef30 (query)                        4_183.78       275.68     4_459.46       0.9858          1.0039            1.0000        41.62
HnswRaBitQ-m32-ex8-ef60 (query)                        4_183.78       324.04     4_507.82       0.9925          1.0010            1.0000        41.62
HnswRaBitQ-m32-ex8-ef120 (query)                       4_183.78       386.22     4_570.00       0.9942          1.0001            1.0000        41.62
HnswRaBitQ-m32-ex8 (self)                              4_183.78    12_183.81    16_367.60       0.9937          1.0007            1.0000        41.62
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 768D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       100.43     1_952.74     2_053.17       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.43     6_605.46     6_705.89       1.0000          1.0000            1.0000       146.48
QG-d32-l64-ef15 (query)                                7_255.41       633.50     7_888.91       0.9857          1.0043            1.0000       311.47
QG-d32-l64-ef30 (query)                                7_255.41       842.08     8_097.48       0.9913          1.0040            1.0000       311.47
QG-d32-l64-ef60 (query)                                7_255.41     1_111.83     8_367.23       0.9940          1.0039            1.0000       311.47
QG-d32-l64-ef120 (query)                               7_255.41     1_493.89     8_749.29       0.9951          1.0038            1.0000       311.47
QG-d32-l64 (self)                                      7_255.41     3_631.49    10_886.90       0.9940          1.0038            1.0000       311.47
QG-d32-l128-ef15 (query)                               8_024.85       644.82     8_669.67       0.9864          1.0041            1.0000       311.47
QG-d32-l128-ef30 (query)                               8_024.85       851.82     8_876.67       0.9918          1.0038            1.0000       311.47
QG-d32-l128-ef60 (query)                               8_024.85     1_140.41     9_165.26       0.9944          1.0037            1.0000       311.47
QG-d32-l128-ef120 (query)                              8_024.85     1_515.18     9_540.03       0.9954          1.0036            1.0000       311.47
QG-d32-l128 (self)                                     8_024.85     3_675.42    11_700.27       0.9945          1.0036            1.0000       311.47
QG-d64-l64-ef15 (query)                               27_193.95     1_336.03    28_529.98       0.9973          1.0029            1.0000       476.27
QG-d64-l64-ef30 (query)                               27_193.95     1_680.42    28_874.37       0.9979          1.0029            1.0000       476.27
QG-d64-l64-ef60 (query)                               27_193.95     2_146.77    29_340.71       0.9981          1.0029            1.0000       476.27
QG-d64-l64-ef120 (query)                              27_193.95     2_810.13    30_004.07       0.9982          1.0029            1.0000       476.27
QG-d64-l64 (self)                                     27_193.95     7_051.64    34_245.58       0.9981          1.0028            1.0000       476.27
QG-d64-l128-ef15 (query)                              27_557.65     1_331.97    28_889.62       0.9970          1.0023            1.0000       476.27
QG-d64-l128-ef30 (query)                              27_557.65     1_683.83    29_241.48       0.9976          1.0023            1.0000       476.27
QG-d64-l128-ef60 (query)                              27_557.65     2_150.11    29_707.76       0.9979          1.0022            1.0000       476.27
QG-d64-l128-ef120 (query)                             27_557.65     2_819.09    30_376.74       0.9980          1.0022            1.0000       476.27
QG-d64-l128 (self)                                    27_557.65     7_076.84    34_634.49       0.9978          1.0037            1.0000       476.27
HnswRaBitQ-m16-ex1-ef15 (query)                        3_144.95       254.44     3_399.39       0.7629          1.1655            1.0045        17.45
HnswRaBitQ-m16-ex1-ef30 (query)                        3_144.95       295.29     3_440.25       0.7745          1.0084            1.0042        17.45
HnswRaBitQ-m16-ex1-ef60 (query)                        3_144.95       349.14     3_494.10       0.7783          1.0052            1.0041        17.45
HnswRaBitQ-m16-ex1-ef120 (query)                       3_144.95       423.95     3_568.90       0.7794          1.0046            1.0041        17.45
HnswRaBitQ-m16-ex1 (self)                              3_144.95    13_941.12    17_086.07       0.6952          1.0102            1.0085        17.45
HnswRaBitQ-m16-ex3-ef15 (query)                        3_396.99       259.98     3_656.97       0.8871          1.0346            1.0007        26.61
HnswRaBitQ-m16-ex3-ef30 (query)                        3_396.99       319.17     3_716.16       0.9127          1.0082            1.0004        26.61
HnswRaBitQ-m16-ex3-ef60 (query)                        3_396.99       390.63     3_787.62       0.9240          1.0021            1.0003        26.61
HnswRaBitQ-m16-ex3-ef120 (query)                       3_396.99       472.28     3_869.27       0.9273          1.0008            1.0003        26.61
HnswRaBitQ-m16-ex3 (self)                              3_396.99    14_441.18    17_838.17       0.8991          1.0025            1.0007        26.61
HnswRaBitQ-m16-ex5-ef15 (query)                        3_702.82       292.84     3_995.66       0.9155          1.5988            1.0002        35.76
HnswRaBitQ-m16-ex5-ef30 (query)                        3_702.82       346.65     4_049.47       0.9537          1.1172            1.0000        35.76
HnswRaBitQ-m16-ex5-ef60 (query)                        3_702.82       423.19     4_126.01       0.9706          1.0023            1.0000        35.76
HnswRaBitQ-m16-ex5-ef120 (query)                       3_702.82       511.83     4_214.65       0.9745          1.0008            1.0000        35.76
HnswRaBitQ-m16-ex5 (self)                              3_702.82    14_765.12    18_467.94       0.9645          1.0103            1.0000        35.76
HnswRaBitQ-m16-ex8-ef15 (query)                        5_988.37       264.94     6_253.31       0.9195         19.7230            1.0001        49.50
HnswRaBitQ-m16-ex8-ef30 (query)                        5_988.37       323.12     6_311.48       0.9672          1.0177            1.0000        49.50
HnswRaBitQ-m16-ex8-ef60 (query)                        5_988.37       395.20     6_383.56       0.9864          1.0016            1.0000        49.50
HnswRaBitQ-m16-ex8-ef120 (query)                       5_988.37       478.96     6_467.32       0.9917          1.0007            1.0000        49.50
HnswRaBitQ-m16-ex8 (self)                              5_988.37    14_525.35    20_513.71       0.9885          1.0026            1.0000        49.50
HnswRaBitQ-m32-ex1-ef15 (query)                        3_223.59       290.54     3_514.12       0.7735          1.0346            1.0042        23.56
HnswRaBitQ-m32-ex1-ef30 (query)                        3_223.59       338.86     3_562.45       0.7778          1.0076            1.0041        23.56
HnswRaBitQ-m32-ex1-ef60 (query)                        3_223.59       397.96     3_621.55       0.7792          1.0045            1.0041        23.56
HnswRaBitQ-m32-ex1-ef120 (query)                       3_223.59       479.50     3_703.08       0.7796          1.0044            1.0041        23.56
HnswRaBitQ-m32-ex1 (self)                              3_223.59    17_663.25    20_886.84       0.6952          1.0090            1.0086        23.56
HnswRaBitQ-m32-ex3-ef15 (query)                        3_497.13       317.07     3_814.19       0.9100          1.3340            1.0004        32.71
HnswRaBitQ-m32-ex3-ef30 (query)                        3_497.13       377.18     3_874.31       0.9228          1.0024            1.0003        32.71
HnswRaBitQ-m32-ex3-ef60 (query)                        3_497.13       448.01     3_945.14       0.9270          1.0007            1.0003        32.71
HnswRaBitQ-m32-ex3-ef120 (query)                       3_497.13       522.48     4_019.60       0.9280          1.0005            1.0003        32.71
HnswRaBitQ-m32-ex3 (self)                              3_497.13    18_166.69    21_663.81       0.9018          1.0010            1.0006        32.71
HnswRaBitQ-m32-ex5-ef15 (query)                        3_832.86       343.83     4_176.70       0.9496          1.0879            1.0000        41.87
HnswRaBitQ-m32-ex5-ef30 (query)                        3_832.86       411.11     4_243.98       0.9677          1.0441            1.0000        41.87
HnswRaBitQ-m32-ex5-ef60 (query)                        3_832.86       497.62     4_330.48       0.9737          1.0007            1.0000        41.87
HnswRaBitQ-m32-ex5-ef120 (query)                       3_832.86       558.69     4_391.55       0.9753          1.0001            1.0000        41.87
HnswRaBitQ-m32-ex5 (self)                              3_832.86    18_613.09    22_445.96       0.9681          1.0003            1.0000        41.87
HnswRaBitQ-m32-ex8-ef15 (query)                        6_093.77       316.24     6_410.01       0.9602          1.5101            1.0000        55.60
HnswRaBitQ-m32-ex8-ef30 (query)                        6_093.77       385.63     6_479.40       0.9831          1.2723            1.0000        55.60
HnswRaBitQ-m32-ex8-ef60 (query)                        6_093.77       449.65     6_543.42       0.9913          1.1007            1.0000        55.60
HnswRaBitQ-m32-ex8-ef120 (query)                       6_093.77       528.70     6_622.47       0.9934          1.0001            1.0000        55.60
HnswRaBitQ-m32-ex8 (self)                              6_093.77    18_142.17    24_235.95       0.9933          1.0003            1.0000        55.60
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 256D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.95       729.92       762.87       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.95     2_513.91     2_546.86       1.0000          1.0000            1.0000        48.83
QG-d32-l64-ef15 (query)                                3_130.58        83.93     3_214.51       0.9733          1.0008            1.0000       116.16
QG-d32-l64-ef30 (query)                                3_130.58       141.87     3_272.45       0.9986          1.0001            1.0000       116.16
QG-d32-l64-ef60 (query)                                3_130.58       265.30     3_395.88       1.0000          1.0000            1.0000       116.16
QG-d32-l64-ef120 (query)                               3_130.58       512.76     3_643.34       1.0000          1.0000            1.0000       116.16
QG-d32-l64 (self)                                      3_130.58       804.59     3_935.17       1.0000          1.0000            1.0000       116.16
QG-d32-l128-ef15 (query)                               3_642.86        85.79     3_728.66       0.9732          1.0008            1.0000       116.16
QG-d32-l128-ef30 (query)                               3_642.86       143.64     3_786.51       0.9986          1.0001            1.0000       116.16
QG-d32-l128-ef60 (query)                               3_642.86       259.22     3_902.08       1.0000          1.0000            1.0000       116.16
QG-d32-l128-ef120 (query)                              3_642.86       497.98     4_140.84       1.0000          1.0000            1.0000       116.16
QG-d32-l128 (self)                                     3_642.86       789.53     4_432.39       1.0000          1.0000            1.0000       116.16
QG-d64-l64-ef15 (query)                                6_313.82       135.52     6_449.34       0.9942          1.0001            1.0000       183.30
QG-d64-l64-ef30 (query)                                6_313.82       233.20     6_547.03       1.0000          1.0000            1.0000       183.30
QG-d64-l64-ef60 (query)                                6_313.82       439.88     6_753.70       1.0000          1.0000            1.0000       183.30
QG-d64-l64-ef120 (query)                               6_313.82       850.27     7_164.10       1.0000          1.0000            1.0000       183.30
QG-d64-l64 (self)                                      6_313.82     1_370.02     7_683.85       1.0000          1.0000            1.0000       183.30
QG-d64-l128-ef15 (query)                               7_078.90       128.64     7_207.54       0.9940          1.0001            1.0000       183.30
QG-d64-l128-ef30 (query)                               7_078.90       234.24     7_313.14       1.0000          1.0000            1.0000       183.30
QG-d64-l128-ef60 (query)                               7_078.90       443.25     7_522.15       1.0000          1.0000            1.0000       183.30
QG-d64-l128-ef120 (query)                              7_078.90       868.71     7_947.61       1.0000          1.0000            1.0000       183.30
QG-d64-l128 (self)                                     7_078.90     1_371.56     8_450.46       1.0000          1.0000            1.0000       183.30
HnswRaBitQ-m16-ex1-ef15 (query)                        1_261.29        80.15     1_341.44       0.8537          1.0073            1.0059        10.85
HnswRaBitQ-m16-ex1-ef30 (query)                        1_261.29        97.26     1_358.55       0.8659          1.0057            1.0049        10.85
HnswRaBitQ-m16-ex1-ef60 (query)                        1_261.29       125.21     1_386.50       0.8693          1.0053            1.0046        10.85
HnswRaBitQ-m16-ex1-ef120 (query)                       1_261.29       164.73     1_426.01       0.8696          1.0052            1.0046        10.85
HnswRaBitQ-m16-ex1 (self)                              1_261.29     5_838.65     7_099.94       0.8312          1.0108            1.0097        10.85
HnswRaBitQ-m16-ex3-ef15 (query)                        1_336.46        85.24     1_421.69       0.9281          1.0028            1.0008        13.90
HnswRaBitQ-m16-ex3-ef30 (query)                        1_336.46       103.64     1_440.10       0.9503          1.0010            1.0003        13.90
HnswRaBitQ-m16-ex3-ef60 (query)                        1_336.46       133.17     1_469.62       0.9565          1.0006            1.0001        13.90
HnswRaBitQ-m16-ex3-ef120 (query)                       1_336.46       174.22     1_510.68       0.9575          1.0005            1.0001        13.90
HnswRaBitQ-m16-ex3 (self)                              1_336.46     5_641.51     6_977.97       0.9469          1.0010            1.0005        13.90
HnswRaBitQ-m16-ex5-ef15 (query)                        1_408.79        87.44     1_496.23       0.9473          1.0025            1.0001        16.95
HnswRaBitQ-m16-ex5-ef30 (query)                        1_408.79       114.41     1_523.20       0.9770          1.0006            1.0000        16.95
HnswRaBitQ-m16-ex5-ef60 (query)                        1_408.79       137.95     1_546.74       0.9852          1.0002            1.0000        16.95
HnswRaBitQ-m16-ex5-ef120 (query)                       1_408.79       183.87     1_592.66       0.9865          1.0001            1.0000        16.95
HnswRaBitQ-m16-ex5 (self)                              1_408.79     5_705.11     7_113.90       0.9836          1.0002            1.0000        16.95
HnswRaBitQ-m16-ex8-ef15 (query)                        2_139.30        87.70     2_227.00       0.9529          1.0024            1.0000        21.53
HnswRaBitQ-m16-ex8-ef30 (query)                        2_139.30       108.89     2_248.19       0.9855          1.0006            1.0000        21.53
HnswRaBitQ-m16-ex8-ef60 (query)                        2_139.30       138.74     2_278.04       0.9953          1.0002            1.0000        21.53
HnswRaBitQ-m16-ex8-ef120 (query)                       2_139.30       180.29     2_319.59       0.9970          1.0001            1.0000        21.53
HnswRaBitQ-m16-ex8 (self)                              2_139.30     5_694.84     7_834.14       0.9965          1.0001            1.0000        21.53
HnswRaBitQ-m32-ex1-ef15 (query)                        1_275.86        89.54     1_365.40       0.8623          1.0062            1.0052        16.95
HnswRaBitQ-m32-ex1-ef30 (query)                        1_275.86       110.94     1_386.80       0.8684          1.0054            1.0047        16.95
HnswRaBitQ-m32-ex1-ef60 (query)                        1_275.86       139.85     1_415.71       0.8695          1.0052            1.0046        16.95
HnswRaBitQ-m32-ex1-ef120 (query)                       1_275.86       181.25     1_457.12       0.8697          1.0052            1.0046        16.95
HnswRaBitQ-m32-ex1 (self)                              1_275.86     6_678.78     7_954.64       0.8314          1.0108            1.0097        16.95
HnswRaBitQ-m32-ex3-ef15 (query)                        1_355.07        96.83     1_451.90       0.9428          1.0016            1.0004        20.00
HnswRaBitQ-m32-ex3-ef30 (query)                        1_355.07       116.80     1_471.87       0.9548          1.0007            1.0002        20.00
HnswRaBitQ-m32-ex3-ef60 (query)                        1_355.07       148.44     1_503.51       0.9574          1.0005            1.0001        20.00
HnswRaBitQ-m32-ex3-ef120 (query)                       1_355.07       191.69     1_546.75       0.9577          1.0004            1.0001        20.00
HnswRaBitQ-m32-ex3 (self)                              1_355.07     7_335.11     8_690.18       0.9475          1.0009            1.0005        20.00
HnswRaBitQ-m32-ex5-ef15 (query)                        1_466.30       102.10     1_568.41       0.9672          1.0011            1.0000        23.06
HnswRaBitQ-m32-ex5-ef30 (query)                        1_466.30       133.84     1_600.14       0.9830          1.0003            1.0000        23.06
HnswRaBitQ-m32-ex5-ef60 (query)                        1_466.30       157.21     1_623.52       0.9864          1.0001            1.0000        23.06
HnswRaBitQ-m32-ex5-ef120 (query)                       1_466.30       199.82     1_666.13       0.9868          1.0001            1.0000        23.06
HnswRaBitQ-m32-ex5 (self)                              1_466.30     6_833.64     8_299.95       0.9845          1.0001            1.0000        23.06
HnswRaBitQ-m32-ex8-ef15 (query)                        2_276.73       100.54     2_377.27       0.9742          1.0012            1.0000        27.63
HnswRaBitQ-m32-ex8-ef30 (query)                        2_276.73       127.44     2_404.18       0.9927          1.0002            1.0000        27.63
HnswRaBitQ-m32-ex8-ef60 (query)                        2_276.73       157.07     2_433.81       0.9967          1.0001            1.0000        27.63
HnswRaBitQ-m32-ex8-ef120 (query)                       2_276.73       226.86     2_503.59       0.9972          1.0000            1.0000        27.63
HnswRaBitQ-m32-ex8 (self)                              2_276.73     7_060.11     9_336.84       0.9975          1.0000            1.0000        27.63
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 512D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.10     1_340.82     1_408.93       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.10     4_496.99     4_565.09       1.0000          1.0000            1.0000        97.66
QG-d32-l64-ef15 (query)                                5_953.23       145.68     6_098.91       0.9607          1.0031            1.0000       213.81
QG-d32-l64-ef30 (query)                                5_953.23       239.72     6_192.95       0.9922          1.0012            1.0000       213.81
QG-d32-l64-ef60 (query)                                5_953.23       411.73     6_364.96       0.9982          1.0006            1.0000       213.81
QG-d32-l64-ef120 (query)                               5_953.23       734.43     6_687.66       0.9993          1.0004            1.0000       213.81
QG-d32-l64 (self)                                      5_953.23     1_263.15     7_216.38       0.9982          1.0007            1.0000       213.81
QG-d32-l128-ef15 (query)                               6_711.46       144.83     6_856.28       0.9616          1.0031            1.0000       213.81
QG-d32-l128-ef30 (query)                               6_711.46       241.58     6_953.04       0.9932          1.0012            1.0000       213.81
QG-d32-l128-ef60 (query)                               6_711.46       430.98     7_142.44       0.9987          1.0006            1.0000       213.81
QG-d32-l128-ef120 (query)                              6_711.46       736.77     7_448.23       0.9996          1.0003            1.0000       213.81
QG-d32-l128 (self)                                     6_711.46     1_275.33     7_986.79       0.9987          1.0006            1.0000       213.81
QG-d64-l64-ef15 (query)                               17_738.48       226.70    17_965.18       0.9931          1.0002            1.0000       329.78
QG-d64-l64-ef30 (query)                               17_738.48       397.19    18_135.67       0.9997          1.0000            1.0000       329.78
QG-d64-l64-ef60 (query)                               17_738.48       710.20    18_448.68       1.0000          1.0000            1.0000       329.78
QG-d64-l64-ef120 (query)                              17_738.48     1_268.44    19_006.91       1.0000          1.0000            1.0000       329.78
QG-d64-l64 (self)                                     17_738.48     2_259.89    19_998.37       1.0000          1.0000            1.0000       329.78
QG-d64-l128-ef15 (query)                              18_073.52       227.68    18_301.20       0.9932          1.0002            1.0000       329.78
QG-d64-l128-ef30 (query)                              18_073.52       423.30    18_496.82       0.9998          1.0000            1.0000       329.78
QG-d64-l128-ef60 (query)                              18_073.52       716.16    18_789.68       1.0000          1.0000            1.0000       329.78
QG-d64-l128-ef120 (query)                             18_073.52     1_273.29    19_346.81       1.0000          1.0000            1.0000       329.78
QG-d64-l128 (self)                                    18_073.52     2_212.78    20_286.30       1.0000          1.0000            1.0000       329.78
HnswRaBitQ-m16-ex1-ef15 (query)                        2_392.77       149.93     2_542.70       0.8468          1.0062            1.0043        14.15
HnswRaBitQ-m16-ex1-ef30 (query)                        2_392.77       177.40     2_570.17       0.8672          1.0042            1.0033        14.15
HnswRaBitQ-m16-ex1-ef60 (query)                        2_392.77       219.27     2_612.03       0.8738          1.0035            1.0029        14.15
HnswRaBitQ-m16-ex1-ef120 (query)                       2_392.77       274.57     2_667.34       0.8749          1.0034            1.0029        14.15
HnswRaBitQ-m16-ex1 (self)                              2_392.77    11_319.54    13_712.31       0.8357          1.0068            1.0060        14.15
HnswRaBitQ-m16-ex3-ef15 (query)                        2_558.39       162.02     2_720.41       0.9100          1.0035            1.0010        20.25
HnswRaBitQ-m16-ex3-ef30 (query)                        2_558.39       194.50     2_752.89       0.9446          1.0013            1.0003        20.25
HnswRaBitQ-m16-ex3-ef60 (query)                        2_558.39       238.43     2_796.82       0.9559          1.0006            1.0001        20.25
HnswRaBitQ-m16-ex3-ef120 (query)                       2_558.39       295.58     2_853.97       0.9583          1.0004            1.0001        20.25
HnswRaBitQ-m16-ex3 (self)                              2_558.39    11_707.23    14_265.62       0.9463          1.0009            1.0003        20.25
HnswRaBitQ-m16-ex5-ef15 (query)                        2_743.68       185.40     2_929.08       0.9245          1.0036            1.0007        26.36
HnswRaBitQ-m16-ex5-ef30 (query)                        2_743.68       204.09     2_947.77       0.9659          1.0011            1.0000        26.36
HnswRaBitQ-m16-ex5-ef60 (query)                        2_743.68       257.35     3_001.03       0.9822          1.0004            1.0000        26.36
HnswRaBitQ-m16-ex5-ef120 (query)                       2_743.68       313.82     3_057.50       0.9857          1.0002            1.0000        26.36
HnswRaBitQ-m16-ex5 (self)                              2_743.68    12_072.45    14_816.13       0.9811          1.0004            1.0000        26.36
HnswRaBitQ-m16-ex8-ef15 (query)                        4_216.56       164.88     4_381.43       0.9287          1.0034            1.0006        35.51
HnswRaBitQ-m16-ex8-ef30 (query)                        4_216.56       199.48     4_416.04       0.9739          1.0010            1.0000        35.51
HnswRaBitQ-m16-ex8-ef60 (query)                        4_216.56       245.84     4_462.40       0.9923          1.0004            1.0000        35.51
HnswRaBitQ-m16-ex8-ef120 (query)                       4_216.56       305.93     4_522.48       0.9962          1.0002            1.0000        35.51
HnswRaBitQ-m16-ex8 (self)                              4_216.56    11_225.84    15_442.40       0.9938          1.0004            1.0000        35.51
HnswRaBitQ-m32-ex1-ef15 (query)                        2_432.62       170.04     2_602.66       0.8635          1.0044            1.0034        20.25
HnswRaBitQ-m32-ex1-ef30 (query)                        2_432.62       204.98     2_637.59       0.8728          1.0035            1.0030        20.25
HnswRaBitQ-m32-ex1-ef60 (query)                        2_432.62       250.87     2_683.49       0.8750          1.0033            1.0029        20.25
HnswRaBitQ-m32-ex1-ef120 (query)                       2_432.62       308.08     2_740.70       0.8752          1.0032            1.0028        20.25
HnswRaBitQ-m32-ex1 (self)                              2_432.62    13_412.61    15_845.22       0.8364          1.0066            1.0059        20.25
HnswRaBitQ-m32-ex3-ef15 (query)                        2_621.45       187.66     2_809.12       0.9124          9.9456            1.0004        26.36
HnswRaBitQ-m32-ex3-ef30 (query)                        2_621.45       229.60     2_851.05       0.9542          1.0006            1.0001        26.36
HnswRaBitQ-m32-ex3-ef60 (query)                        2_621.45       283.32     2_904.77       0.9584          1.0003            1.0001        26.36
HnswRaBitQ-m32-ex3-ef120 (query)                       2_621.45       353.46     2_974.91       0.9589          1.0003            1.0001        26.36
HnswRaBitQ-m32-ex3 (self)                              2_621.45    14_112.10    16_733.55       0.9482          1.0006            1.0003        26.36
HnswRaBitQ-m32-ex5-ef15 (query)                        2_802.73       198.51     3_001.24       0.9576          1.0014            1.0000        32.46
HnswRaBitQ-m32-ex5-ef30 (query)                        2_802.73       238.35     3_041.08       0.9800          1.0003            1.0000        32.46
HnswRaBitQ-m32-ex5-ef60 (query)                        2_802.73       317.89     3_120.62       0.9855          1.0001            1.0000        32.46
HnswRaBitQ-m32-ex5-ef120 (query)                       2_802.73       346.02     3_148.75       0.9863          1.0001            1.0000        32.46
HnswRaBitQ-m32-ex5 (self)                              2_802.73    14_398.07    17_200.81       0.9837          1.0001            1.0000        32.46
HnswRaBitQ-m32-ex8-ef15 (query)                        4_273.62       201.04     4_474.66       0.9645          1.0025            1.0000        41.62
HnswRaBitQ-m32-ex8-ef30 (query)                        4_273.62       237.20     4_510.82       0.9897          1.0003            1.0000        41.62
HnswRaBitQ-m32-ex8-ef60 (query)                        4_273.62       284.74     4_558.36       0.9960          1.0001            1.0000        41.62
HnswRaBitQ-m32-ex8-ef120 (query)                       4_273.62       343.56     4_617.17       0.9969          1.0001            1.0000        41.62
HnswRaBitQ-m32-ex8 (self)                              4_273.62    14_769.73    19_043.35       0.9970          1.0001            1.0000        41.62
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 768D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       100.08     1_875.68     1_975.76       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.08     6_344.49     6_444.57       1.0000          1.0000            1.0000       146.48
QG-d32-l64-ef15 (query)                                7_944.79       248.50     8_193.29       0.9481          1.0140            1.0000       311.47
QG-d32-l64-ef30 (query)                                7_944.79       404.01     8_348.80       0.9776          1.0118            1.0000       311.47
QG-d32-l64-ef60 (query)                                7_944.79       666.00     8_610.78       0.9892          1.0069            1.0000       311.47
QG-d32-l64-ef120 (query)                               7_944.79     1_071.36     9_016.15       0.9938          1.0037            1.0000       311.47
QG-d32-l64 (self)                                      7_944.79     2_140.01    10_084.79       0.9892          1.0071            1.0000       311.47
QG-d32-l128-ef15 (query)                               8_847.08       255.20     9_102.28       0.9547          1.0064            1.0000       311.47
QG-d32-l128-ef30 (query)                               8_847.08       406.01     9_253.09       0.9838          1.0045            1.0000       311.47
QG-d32-l128-ef60 (query)                               8_847.08       665.93     9_513.02       0.9924          1.0038            1.0000       311.47
QG-d32-l128-ef120 (query)                              8_847.08     1_093.08     9_940.17       0.9950          1.0032            1.0000       311.47
QG-d32-l128 (self)                                     8_847.08     2_105.44    10_952.52       0.9927          1.0040            1.0000       311.47
QG-d64-l64-ef15 (query)                               30_926.73       491.48    31_418.21       0.9930          1.0008            1.0000       476.27
QG-d64-l64-ef30 (query)                               30_926.73       804.63    31_731.36       0.9985          1.0004            1.0000       476.27
QG-d64-l64-ef60 (query)                               30_926.73     1_296.74    32_223.46       0.9995          1.0002            1.0000       476.27
QG-d64-l64-ef120 (query)                              30_926.73     2_187.41    33_114.13       0.9998          1.0001            1.0000       476.27
QG-d64-l64 (self)                                     30_926.73     4_194.42    35_121.15       0.9996          1.0002            1.0000       476.27
QG-d64-l128-ef15 (query)                              30_626.54       492.11    31_118.65       0.9934          1.0009            1.0000       476.27
QG-d64-l128-ef30 (query)                              30_626.54       809.71    31_436.25       0.9988          1.0003            1.0000       476.27
QG-d64-l128-ef60 (query)                              30_626.54     1_367.34    31_993.88       0.9997          1.0001            1.0000       476.27
QG-d64-l128-ef120 (query)                             30_626.54     2_062.78    32_689.32       0.9999          1.0001            1.0000       476.27
QG-d64-l128 (self)                                    30_626.54     4_284.30    34_910.84       0.9997          1.0002            1.0000       476.27
HnswRaBitQ-m16-ex1-ef15 (query)                        3_377.18       234.11     3_611.29       0.7931        184.1552            1.0041        17.45
HnswRaBitQ-m16-ex1-ef30 (query)                        3_377.18       279.42     3_656.60       0.8278        140.8663            1.0030        17.45
HnswRaBitQ-m16-ex1-ef60 (query)                        3_377.18       348.14     3_725.32       0.8626          1.0033            1.0025        17.45
HnswRaBitQ-m16-ex1-ef120 (query)                       3_377.18       426.51     3_803.69       0.8654          1.0029            1.0024        17.45
HnswRaBitQ-m16-ex1 (self)                              3_377.18    17_527.13    20_904.31       0.8183          1.0061            1.0051        17.45
HnswRaBitQ-m16-ex3-ef15 (query)                        3_587.16       241.37     3_828.53       0.8881          1.0062            1.0012        26.61
HnswRaBitQ-m16-ex3-ef30 (query)                        3_587.16       289.05     3_876.22       0.9311          1.0026            1.0003        26.61
HnswRaBitQ-m16-ex3-ef60 (query)                        3_587.16       345.35     3_932.52       0.9499          1.0009            1.0001        26.61
HnswRaBitQ-m16-ex3-ef120 (query)                       3_587.16       424.70     4_011.86       0.9545          1.0006            1.0001        26.61
HnswRaBitQ-m16-ex3 (self)                              3_587.16    17_128.34    20_715.50       0.9388          1.0011            1.0003        26.61
HnswRaBitQ-m16-ex5-ef15 (query)                        3_862.88       255.30     4_118.17       0.9022          1.0057            1.0010        35.76
HnswRaBitQ-m16-ex5-ef30 (query)                        3_862.88       301.90     4_164.78       0.9530          1.0021            1.0000        35.76
HnswRaBitQ-m16-ex5-ef60 (query)                        3_862.88       367.86     4_230.74       0.9770          1.0008            1.0000        35.76
HnswRaBitQ-m16-ex5-ef120 (query)                       3_862.88       447.69     4_310.56       0.9833          1.0004            1.0000        35.76
HnswRaBitQ-m16-ex5 (self)                              3_862.88    18_912.43    22_775.31       0.9755          1.0008            1.0000        35.76
HnswRaBitQ-m16-ex8-ef15 (query)                        6_370.60       252.50     6_623.10       0.8990          7.5303            1.0010        49.50
HnswRaBitQ-m16-ex8-ef30 (query)                        6_370.60       300.18     6_670.78       0.9601          1.0022            1.0000        49.50
HnswRaBitQ-m16-ex8-ef60 (query)                        6_370.60       354.59     6_725.19       0.9872          1.0009            1.0000        49.50
HnswRaBitQ-m16-ex8-ef120 (query)                       6_370.60       434.76     6_805.36       0.9946          1.0003            1.0000        49.50
HnswRaBitQ-m16-ex8 (self)                              6_370.60    18_030.24    24_400.84       0.9893          1.0007            1.0000        49.50
HnswRaBitQ-m32-ex1-ef15 (query)                        3_352.39       260.24     3_612.63       0.7915        299.9318            1.0032        23.56
HnswRaBitQ-m32-ex1-ef30 (query)                        3_352.39       329.85     3_682.24       0.8328        155.1727            1.0026        23.56
HnswRaBitQ-m32-ex1-ef60 (query)                        3_352.39       397.48     3_749.87       0.8650          1.0029            1.0024        23.56
HnswRaBitQ-m32-ex1-ef120 (query)                       3_352.39       470.10     3_822.49       0.8658          1.0027            1.0024        23.56
HnswRaBitQ-m32-ex1 (self)                              3_352.39    21_661.93    25_014.32       0.8200          1.0056            1.0050        23.56
HnswRaBitQ-m32-ex3-ef15 (query)                        3_649.47       283.81     3_933.28       0.9240          1.0046            1.0004        32.71
HnswRaBitQ-m32-ex3-ef30 (query)                        3_649.47       335.73     3_985.20       0.9473          1.0010            1.0002        32.71
HnswRaBitQ-m32-ex3-ef60 (query)                        3_649.47       396.83     4_046.31       0.9542          1.0005            1.0001        32.71
HnswRaBitQ-m32-ex3-ef120 (query)                       3_649.47       471.50     4_120.98       0.9556          1.0003            1.0001        32.71
HnswRaBitQ-m32-ex3 (self)                              3_649.47    22_257.30    25_906.77       0.9421          1.0006            1.0003        32.71
HnswRaBitQ-m32-ex5-ef15 (query)                        4_212.09       324.56     4_536.65       0.9462          1.0032            1.0000        41.87
HnswRaBitQ-m32-ex5-ef30 (query)                        4_212.09       379.13     4_591.22       0.9743          1.0009            1.0000        41.87
HnswRaBitQ-m32-ex5-ef60 (query)                        4_212.09       445.96     4_658.05       0.9832          1.0002            1.0000        41.87
HnswRaBitQ-m32-ex5-ef120 (query)                       4_212.09       527.80     4_739.89       0.9848          1.0001            1.0000        41.87
HnswRaBitQ-m32-ex5 (self)                              4_212.09    22_250.51    26_462.60       0.9805          1.0002            1.0000        41.87
HnswRaBitQ-m32-ex8-ef15 (query)                        6_009.93       279.58     6_289.52       0.9202        186.4235            1.0000        55.60
HnswRaBitQ-m32-ex8-ef30 (query)                        6_009.93       361.07     6_371.00       0.9826          1.4203            1.0000        55.60
HnswRaBitQ-m32-ex8-ef60 (query)                        6_009.93       417.22     6_427.16       0.9934          1.1969            1.0000        55.60
HnswRaBitQ-m32-ex8-ef120 (query)                       6_009.93       489.59     6_499.52       0.9961          1.0001            1.0000        55.60
HnswRaBitQ-m32-ex8 (self)                              6_009.93    21_424.67    27_434.60       0.9953          1.0002            1.0000        55.60
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Cell embeddings data

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 256D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.89       733.49       766.39       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.89     2_456.09     2_488.99       1.0000          1.0000            1.0000        48.83
QG-d32-l64-ef15 (query)                                2_868.86        60.24     2_929.10       0.9966          1.0001            1.0000       116.16
QG-d32-l64-ef30 (query)                                2_868.86        96.69     2_965.56       1.0000          1.0000            1.0000       116.16
QG-d32-l64-ef60 (query)                                2_868.86       166.78     3_035.64       1.0000          1.0000            1.0000       116.16
QG-d32-l64-ef120 (query)                               2_868.86       319.47     3_188.33       1.0000          1.0000            1.0000       116.16
QG-d32-l64 (self)                                      2_868.86       517.69     3_386.56       1.0000          1.0000            1.0000       116.16
QG-d32-l128-ef15 (query)                               4_052.25        58.38     4_110.63       0.9960          1.0001            1.0000       116.16
QG-d32-l128-ef30 (query)                               4_052.25        92.23     4_144.48       1.0000          1.0000            1.0000       116.16
QG-d32-l128-ef60 (query)                               4_052.25       162.71     4_214.95       1.0000          1.0000            1.0000       116.16
QG-d32-l128-ef120 (query)                              4_052.25       325.09     4_377.34       1.0000          1.0000            1.0000       116.16
QG-d32-l128 (self)                                     4_052.25       510.32     4_562.57       1.0000          1.0000            1.0000       116.16
QG-d64-l64-ef15 (query)                                3_604.25        67.88     3_672.13       0.9980          1.0000            1.0000       183.30
QG-d64-l64-ef30 (query)                                3_604.25       107.17     3_711.42       1.0000          1.0000            1.0000       183.30
QG-d64-l64-ef60 (query)                                3_604.25       193.52     3_797.77       1.0000          1.0000            1.0000       183.30
QG-d64-l64-ef120 (query)                               3_604.25       382.20     3_986.45       1.0000          1.0000            1.0000       183.30
QG-d64-l64 (self)                                      3_604.25       625.53     4_229.78       1.0000          1.0000            1.0000       183.30
QG-d64-l128-ef15 (query)                               5_323.55        68.33     5_391.87       0.9982          1.0000            1.0000       183.30
QG-d64-l128-ef30 (query)                               5_323.55       114.97     5_438.52       1.0000          1.0000            1.0000       183.30
QG-d64-l128-ef60 (query)                               5_323.55       202.29     5_525.84       1.0000          1.0000            1.0000       183.30
QG-d64-l128-ef120 (query)                              5_323.55       393.15     5_716.70       1.0000          1.0000            1.0000       183.30
QG-d64-l128 (self)                                     5_323.55       669.10     5_992.65       1.0000          1.0000            1.0000       183.30
HnswRaBitQ-m16-ex1-ef15 (query)                        1_333.79        66.28     1_400.07       0.9385          1.0134            1.0034        10.85
HnswRaBitQ-m16-ex1-ef30 (query)                        1_333.79        76.16     1_409.95       0.9398          1.0107            1.0032        10.85
HnswRaBitQ-m16-ex1-ef60 (query)                        1_333.79        95.13     1_428.93       0.9406          1.0075            1.0032        10.85
HnswRaBitQ-m16-ex1-ef120 (query)                       1_333.79       128.69     1_462.49       0.9408          1.0061            1.0032        10.85
HnswRaBitQ-m16-ex1 (self)                              1_333.79     3_470.41     4_804.20       0.9132          1.0127            1.0086        10.85
HnswRaBitQ-m16-ex3-ef15 (query)                        1_347.03        67.12     1_414.16       0.9741          1.0326            1.0000        13.90
HnswRaBitQ-m16-ex3-ef30 (query)                        1_347.03        76.49     1_423.52       0.9769          1.0241            1.0000        13.90
HnswRaBitQ-m16-ex3-ef60 (query)                        1_347.03        95.70     1_442.73       0.9782          1.0161            1.0000        13.90
HnswRaBitQ-m16-ex3-ef120 (query)                       1_347.03       129.07     1_476.10       0.9790          1.0097            1.0000        13.90
HnswRaBitQ-m16-ex3 (self)                              1_347.03     3_616.24     4_963.27       0.9725          1.0103            1.0000        13.90
HnswRaBitQ-m16-ex5-ef15 (query)                        1_453.71        70.14     1_523.85       0.9893          1.0161            1.0000        16.95
HnswRaBitQ-m16-ex5-ef30 (query)                        1_453.71        82.48     1_536.20       0.9920          1.0133            1.0000        16.95
HnswRaBitQ-m16-ex5-ef60 (query)                        1_453.71       102.95     1_556.66       0.9927          1.0098            1.0000        16.95
HnswRaBitQ-m16-ex5-ef120 (query)                       1_453.71       141.70     1_595.41       0.9936          1.0053            1.0000        16.95
HnswRaBitQ-m16-ex5 (self)                              1_453.71     3_644.82     5_098.53       0.9909          1.0097            1.0000        16.95
HnswRaBitQ-m16-ex8-ef15 (query)                        2_166.52        69.12     2_235.64       0.9931          1.0193            1.0000        21.53
HnswRaBitQ-m16-ex8-ef30 (query)                        2_166.52        82.48     2_249.00       0.9966          1.0147            1.0000        21.53
HnswRaBitQ-m16-ex8-ef60 (query)                        2_166.52       103.57     2_270.09       0.9975          1.0114            1.0000        21.53
HnswRaBitQ-m16-ex8-ef120 (query)                       2_166.52       138.55     2_305.07       0.9984          1.0061            1.0000        21.53
HnswRaBitQ-m16-ex8 (self)                              2_166.52     3_650.84     5_817.36       0.9973          1.0115            1.0000        21.53
HnswRaBitQ-m32-ex1-ef15 (query)                        1_331.77        71.14     1_402.91       0.9397          1.0083            1.0033        16.95
HnswRaBitQ-m32-ex1-ef30 (query)                        1_331.77        84.98     1_416.75       0.9404          1.0074            1.0032        16.95
HnswRaBitQ-m32-ex1-ef60 (query)                        1_331.77       108.21     1_439.99       0.9406          1.0065            1.0032        16.95
HnswRaBitQ-m32-ex1-ef120 (query)                       1_331.77       142.70     1_474.48       0.9408          1.0056            1.0031        16.95
HnswRaBitQ-m32-ex1 (self)                              1_331.77     4_006.44     5_338.21       0.9131          1.0136            1.0086        16.95
HnswRaBitQ-m32-ex3-ef15 (query)                        1_418.87        75.97     1_494.84       0.9776          1.0102            1.0000        20.00
HnswRaBitQ-m32-ex3-ef30 (query)                        1_418.87        89.88     1_508.75       0.9791          1.0078            1.0000        20.00
HnswRaBitQ-m32-ex3-ef60 (query)                        1_418.87       113.73     1_532.60       0.9797          1.0046            1.0000        20.00
HnswRaBitQ-m32-ex3-ef120 (query)                       1_418.87       155.24     1_574.11       0.9804          1.0012            1.0000        20.00
HnswRaBitQ-m32-ex3 (self)                              1_418.87     4_166.78     5_585.64       0.9734          1.0049            1.0000        20.00
HnswRaBitQ-m32-ex5-ef15 (query)                        1_517.66        80.97     1_598.63       0.9912          1.0103            1.0000        23.06
HnswRaBitQ-m32-ex5-ef30 (query)                        1_517.66        95.38     1_613.04       0.9931          1.0082            1.0000        23.06
HnswRaBitQ-m32-ex5-ef60 (query)                        1_517.66       117.63     1_635.29       0.9935          1.0064            1.0000        23.06
HnswRaBitQ-m32-ex5-ef120 (query)                       1_517.66       158.14     1_675.79       0.9940          1.0034            1.0000        23.06
HnswRaBitQ-m32-ex5 (self)                              1_517.66     4_140.61     5_658.27       0.9920          1.0040            1.0000        23.06
HnswRaBitQ-m32-ex8-ef15 (query)                        2_294.27        80.62     2_374.90       0.9966          1.0045            1.0000        27.63
HnswRaBitQ-m32-ex8-ef30 (query)                        2_294.27        95.06     2_389.33       0.9985          1.0034            1.0000        27.63
HnswRaBitQ-m32-ex8-ef60 (query)                        2_294.27       116.55     2_410.83       0.9989          1.0019            1.0000        27.63
HnswRaBitQ-m32-ex8-ef120 (query)                       2_294.27       157.61     2_451.89       0.9992          1.0006            1.0000        27.63
HnswRaBitQ-m32-ex8 (self)                              2_294.27     4_195.92     6_490.19       0.9986          1.0023            1.0000        27.63
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 512D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.69     1_327.44     1_396.13       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.69     4_505.18     4_573.87       1.0000          1.0000            1.0000        97.66
QG-d32-l64-ef15 (query)                                5_299.71        92.04     5_391.75       0.9976          1.0000            1.0000       213.81
QG-d32-l64-ef30 (query)                                5_299.71       130.91     5_430.62       1.0000          1.0000            1.0000       213.81
QG-d32-l64-ef60 (query)                                5_299.71       213.51     5_513.22       1.0000          1.0000            1.0000       213.81
QG-d32-l64-ef120 (query)                               5_299.71       385.75     5_685.46       1.0000          1.0000            1.0000       213.81
QG-d32-l64 (self)                                      5_299.71       667.43     5_967.13       1.0000          1.0000            1.0000       213.81
QG-d32-l128-ef15 (query)                               7_885.02        89.63     7_974.65       0.9969          1.0001            1.0000       213.81
QG-d32-l128-ef30 (query)                               7_885.02       130.50     8_015.52       1.0000          1.0000            1.0000       213.81
QG-d32-l128-ef60 (query)                               7_885.02       211.27     8_096.29       1.0000          1.0000            1.0000       213.81
QG-d32-l128-ef120 (query)                              7_885.02       407.11     8_292.13       1.0000          1.0000            1.0000       213.81
QG-d32-l128 (self)                                     7_885.02       648.01     8_533.03       1.0000          1.0000            1.0000       213.81
QG-d64-l64-ef15 (query)                                6_870.52       121.46     6_991.99       0.9984          1.0000            1.0000       329.78
QG-d64-l64-ef30 (query)                                6_870.52       160.73     7_031.26       1.0000          1.0000            1.0000       329.78
QG-d64-l64-ef60 (query)                                6_870.52       245.73     7_116.25       1.0000          1.0000            1.0000       329.78
QG-d64-l64-ef120 (query)                               6_870.52       439.94     7_310.47       1.0000          1.0000            1.0000       329.78
QG-d64-l64 (self)                                      6_870.52       732.09     7_602.62       1.0000          1.0000            1.0000       329.78
QG-d64-l128-ef15 (query)                              10_228.21       107.80    10_336.02       0.9983          1.0000            1.0000       329.78
QG-d64-l128-ef30 (query)                              10_228.21       146.64    10_374.86       1.0000          1.0000            1.0000       329.78
QG-d64-l128-ef60 (query)                              10_228.21       238.62    10_466.83       1.0000          1.0000            1.0000       329.78
QG-d64-l128-ef120 (query)                             10_228.21       447.07    10_675.28       1.0000          1.0000            1.0000       329.78
QG-d64-l128 (self)                                    10_228.21       733.55    10_961.77       1.0000          1.0000            1.0000       329.78
HnswRaBitQ-m16-ex1-ef15 (query)                        2_317.53       113.10     2_430.63       0.9562          1.0122            1.0006        14.15
HnswRaBitQ-m16-ex1-ef30 (query)                        2_317.53       127.51     2_445.04       0.9574          1.0116            1.0005        14.15
HnswRaBitQ-m16-ex1-ef60 (query)                        2_317.53       153.40     2_470.93       0.9576          1.0086            1.0005        14.15
HnswRaBitQ-m16-ex1-ef120 (query)                       2_317.53       193.65     2_511.18       0.9580          1.0035            1.0005        14.15
HnswRaBitQ-m16-ex1 (self)                              2_317.53     5_605.01     7_922.54       0.9342          1.0159            1.0039        14.15
HnswRaBitQ-m16-ex3-ef15 (query)                        2_494.44       124.54     2_618.98       0.9736          1.1754            1.0000        20.25
HnswRaBitQ-m16-ex3-ef30 (query)                        2_494.44       140.82     2_635.26       0.9796          1.0956            1.0000        20.25
HnswRaBitQ-m16-ex3-ef60 (query)                        2_494.44       169.62     2_664.06       0.9833          1.0377            1.0000        20.25
HnswRaBitQ-m16-ex3-ef120 (query)                       2_494.44       214.81     2_709.25       0.9851          1.0128            1.0000        20.25
HnswRaBitQ-m16-ex3 (self)                              2_494.44     5_897.90     8_392.34       0.9790          1.0343            1.0000        20.25
HnswRaBitQ-m16-ex5-ef15 (query)                        2_793.08       131.67     2_924.75       0.9917          1.0129            1.0000        26.36
HnswRaBitQ-m16-ex5-ef30 (query)                        2_793.08       145.09     2_938.17       0.9943          1.0095            1.0000        26.36
HnswRaBitQ-m16-ex5-ef60 (query)                        2_793.08       176.18     2_969.26       0.9947          1.0080            1.0000        26.36
HnswRaBitQ-m16-ex5-ef120 (query)                       2_793.08       218.19     3_011.26       0.9949          1.0061            1.0000        26.36
HnswRaBitQ-m16-ex5 (self)                              2_793.08     5_982.72     8_775.80       0.9935          1.0070            1.0000        26.36
HnswRaBitQ-m16-ex8-ef15 (query)                        4_175.89       125.60     4_301.49       0.9905          1.0760            1.0000        35.51
HnswRaBitQ-m16-ex8-ef30 (query)                        4_175.89       142.27     4_318.16       0.9941          1.0606            1.0000        35.51
HnswRaBitQ-m16-ex8-ef60 (query)                        4_175.89       169.05     4_344.94       0.9964          1.0350            1.0000        35.51
HnswRaBitQ-m16-ex8-ef120 (query)                       4_175.89       218.79     4_394.68       0.9975          1.0205            1.0000        35.51
HnswRaBitQ-m16-ex8 (self)                              4_175.89     5_872.17    10_048.06       0.9957          1.0418            1.0000        35.51
HnswRaBitQ-m32-ex1-ef15 (query)                        2_420.28       126.54     2_546.81       0.9571          1.0053            1.0006        20.25
HnswRaBitQ-m32-ex1-ef30 (query)                        2_420.28       143.64     2_563.91       0.9580          1.0041            1.0005        20.25
HnswRaBitQ-m32-ex1-ef60 (query)                        2_420.28       171.74     2_592.02       0.9583          1.0022            1.0005        20.25
HnswRaBitQ-m32-ex1-ef120 (query)                       2_420.28       220.33     2_640.60       0.9583          1.0022            1.0005        20.25
HnswRaBitQ-m32-ex1 (self)                              2_420.28     6_584.50     9_004.77       0.9351          1.0068            1.0039        20.25
HnswRaBitQ-m32-ex3-ef15 (query)                        2_568.72       136.71     2_705.42       0.9851          1.0004            1.0000        26.36
HnswRaBitQ-m32-ex3-ef30 (query)                        2_568.72       155.40     2_724.12       0.9865          1.0002            1.0000        26.36
HnswRaBitQ-m32-ex3-ef60 (query)                        2_568.72       183.98     2_752.69       0.9865          1.0002            1.0000        26.36
HnswRaBitQ-m32-ex3-ef120 (query)                       2_568.72       233.24     2_801.96       0.9865          1.0002            1.0000        26.36
HnswRaBitQ-m32-ex3 (self)                              2_568.72     6_683.67     9_252.39       0.9817          1.0006            1.0000        26.36
HnswRaBitQ-m32-ex5-ef15 (query)                        2_778.42       144.43     2_922.85       0.9943          1.0011            1.0000        32.46
HnswRaBitQ-m32-ex5-ef30 (query)                        2_778.42       162.57     2_940.99       0.9958          1.0009            1.0000        32.46
HnswRaBitQ-m32-ex5-ef60 (query)                        2_778.42       191.54     2_969.96       0.9960          1.0002            1.0000        32.46
HnswRaBitQ-m32-ex5-ef120 (query)                       2_778.42       251.60     3_030.03       0.9961          1.0000            1.0000        32.46
HnswRaBitQ-m32-ex5 (self)                              2_778.42     6_902.03     9_680.45       0.9945          1.0007            1.0000        32.46
HnswRaBitQ-m32-ex8-ef15 (query)                        4_308.00       141.25     4_449.25       0.9978          1.0002            1.0000        41.62
HnswRaBitQ-m32-ex8-ef30 (query)                        4_308.00       169.07     4_477.07       0.9994          1.0000            1.0000        41.62
HnswRaBitQ-m32-ex8-ef60 (query)                        4_308.00       187.34     4_495.34       0.9995          1.0000            1.0000        41.62
HnswRaBitQ-m32-ex8-ef120 (query)                       4_308.00       240.83     4_548.83       0.9995          1.0000            1.0000        41.62
HnswRaBitQ-m32-ex8 (self)                              4_308.00     6_694.43    11_002.43       0.9992          1.0003            1.0000        41.62
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 768D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        99.91     1_883.15     1_983.06       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.91     6_795.30     6_895.21       1.0000          1.0000            1.0000       146.48
QG-d32-l64-ef15 (query)                                7_982.99       117.63     8_100.62       0.9973          1.0001            1.0000       311.47
QG-d32-l64-ef30 (query)                                7_982.99       167.51     8_150.50       1.0000          1.0000            1.0000       311.47
QG-d32-l64-ef60 (query)                                7_982.99       266.16     8_249.15       1.0000          1.0000            1.0000       311.47
QG-d32-l64-ef120 (query)                               7_982.99       443.81     8_426.80       1.0000          1.0000            1.0000       311.47
QG-d32-l64 (self)                                      7_982.99       743.84     8_726.82       1.0000          1.0000            1.0000       311.47
QG-d32-l128-ef15 (query)                              11_244.23       119.79    11_364.02       0.9968          1.0001            1.0000       311.47
QG-d32-l128-ef30 (query)                              11_244.23       164.78    11_409.01       1.0000          1.0000            1.0000       311.47
QG-d32-l128-ef60 (query)                              11_244.23       250.18    11_494.41       1.0000          1.0000            1.0000       311.47
QG-d32-l128-ef120 (query)                             11_244.23       451.90    11_696.13       1.0000          1.0000            1.0000       311.47
QG-d32-l128 (self)                                    11_244.23       749.68    11_993.91       1.0000          1.0000            1.0000       311.47
QG-d64-l64-ef15 (query)                                9_442.68       118.18     9_560.86       0.9983          1.0000            1.0000       476.27
QG-d64-l64-ef30 (query)                                9_442.68       165.00     9_607.67       1.0000          1.0000            1.0000       476.27
QG-d64-l64-ef60 (query)                                9_442.68       269.90     9_712.57       1.0000          1.0000            1.0000       476.27
QG-d64-l64-ef120 (query)                               9_442.68       484.86     9_927.53       1.0000          1.0000            1.0000       476.27
QG-d64-l64 (self)                                      9_442.68       810.15    10_252.83       1.0000          1.0000            1.0000       476.27
QG-d64-l128-ef15 (query)                              13_679.10       114.21    13_793.30       0.9981          1.0000            1.0000       476.27
QG-d64-l128-ef30 (query)                              13_679.10       164.20    13_843.29       1.0000          1.0000            1.0000       476.27
QG-d64-l128-ef60 (query)                              13_679.10       267.57    13_946.66       1.0000          1.0000            1.0000       476.27
QG-d64-l128-ef120 (query)                             13_679.10       525.71    14_204.80       1.0000          1.0000            1.0000       476.27
QG-d64-l128 (self)                                    13_679.10       808.63    14_487.73       1.0000          1.0000            1.0000       476.27
HnswRaBitQ-m16-ex1-ef15 (query)                        2_948.11       152.99     3_101.10       0.9489          1.0259            1.0015        17.45
HnswRaBitQ-m16-ex1-ef30 (query)                        2_948.11       169.13     3_117.24       0.9501          1.0222            1.0014        17.45
HnswRaBitQ-m16-ex1-ef60 (query)                        2_948.11       199.50     3_147.61       0.9504          1.0178            1.0014        17.45
HnswRaBitQ-m16-ex1-ef120 (query)                       2_948.11       251.29     3_199.40       0.9506          1.0154            1.0014        17.45
HnswRaBitQ-m16-ex1 (self)                              2_948.11     7_647.55    10_595.66       0.9208          1.0185            1.0063        17.45
HnswRaBitQ-m16-ex3-ef15 (query)                        3_166.09       163.05     3_329.14       0.9745          1.0178            1.0000        26.61
HnswRaBitQ-m16-ex3-ef30 (query)                        3_166.09       180.33     3_346.42       0.9762          1.0155            1.0000        26.61
HnswRaBitQ-m16-ex3-ef60 (query)                        3_166.09       211.51     3_377.60       0.9765          1.0110            1.0000        26.61
HnswRaBitQ-m16-ex3-ef120 (query)                       3_166.09       264.98     3_431.07       0.9769          1.0056            1.0000        26.61
HnswRaBitQ-m16-ex3 (self)                              3_166.09     7_872.84    11_038.93       0.9693          1.0098            1.0000        26.61
HnswRaBitQ-m16-ex5-ef15 (query)                        3_385.86       170.43     3_556.28       0.9912          1.0184            1.0000        35.76
HnswRaBitQ-m16-ex5-ef30 (query)                        3_385.86       188.44     3_574.30       0.9936          1.0149            1.0000        35.76
HnswRaBitQ-m16-ex5-ef60 (query)                        3_385.86       220.61     3_606.46       0.9938          1.0137            1.0000        35.76
HnswRaBitQ-m16-ex5-ef120 (query)                       3_385.86       272.24     3_658.10       0.9940          1.0109            1.0000        35.76
HnswRaBitQ-m16-ex5 (self)                              3_385.86     8_099.47    11_485.33       0.9923          1.0107            1.0000        35.76
HnswRaBitQ-m16-ex8-ef15 (query)                        5_084.66       163.00     5_247.66       0.9955          1.0171            1.0000        49.50
HnswRaBitQ-m16-ex8-ef30 (query)                        5_084.66       180.74     5_265.39       0.9981          1.0127            1.0000        49.50
HnswRaBitQ-m16-ex8-ef60 (query)                        5_084.66       211.73     5_296.38       0.9985          1.0111            1.0000        49.50
HnswRaBitQ-m16-ex8-ef120 (query)                       5_084.66       264.94     5_349.60       0.9989          1.0050            1.0000        49.50
HnswRaBitQ-m16-ex8 (self)                              5_084.66     7_757.11    12_841.77       0.9988          1.0067            1.0000        49.50
HnswRaBitQ-m32-ex1-ef15 (query)                        3_034.79       170.62     3_205.41       0.9507          1.0038            1.0015        23.56
HnswRaBitQ-m32-ex1-ef30 (query)                        3_034.79       189.52     3_224.31       0.9514          1.0036            1.0014        23.56
HnswRaBitQ-m32-ex1-ef60 (query)                        3_034.79       224.20     3_258.98       0.9514          1.0036            1.0014        23.56
HnswRaBitQ-m32-ex1-ef120 (query)                       3_034.79       280.23     3_315.01       0.9514          1.0036            1.0014        23.56
HnswRaBitQ-m32-ex1 (self)                              3_034.79     8_762.71    11_797.49       0.9214          1.0108            1.0063        23.56
HnswRaBitQ-m32-ex3-ef15 (query)                        3_299.95       183.51     3_483.46       0.9768          1.0010            1.0000        32.71
HnswRaBitQ-m32-ex3-ef30 (query)                        3_299.95       200.63     3_500.58       0.9779          1.0008            1.0000        32.71
HnswRaBitQ-m32-ex3-ef60 (query)                        3_299.95       233.10     3_533.05       0.9780          1.0008            1.0000        32.71
HnswRaBitQ-m32-ex3-ef120 (query)                       3_299.95       290.78     3_590.73       0.9780          1.0008            1.0000        32.71
HnswRaBitQ-m32-ex3 (self)                              3_299.95     9_118.33    12_418.28       0.9706          1.0015            1.0000        32.71
HnswRaBitQ-m32-ex5-ef15 (query)                        3_492.63       202.75     3_695.38       0.9934          1.0002            1.0000        41.87
HnswRaBitQ-m32-ex5-ef30 (query)                        3_492.63       222.37     3_715.00       0.9950          1.0000            1.0000        41.87
HnswRaBitQ-m32-ex5-ef60 (query)                        3_492.63       271.76     3_764.39       0.9950          1.0000            1.0000        41.87
HnswRaBitQ-m32-ex5-ef120 (query)                       3_492.63       313.31     3_805.94       0.9951          1.0000            1.0000        41.87
HnswRaBitQ-m32-ex5 (self)                              3_492.63     9_942.85    13_435.48       0.9932          1.0002            1.0000        41.87
HnswRaBitQ-m32-ex8-ef15 (query)                        5_466.19       182.08     5_648.27       0.9975          1.0002            1.0000        55.60
HnswRaBitQ-m32-ex8-ef30 (query)                        5_466.19       204.76     5_670.95       0.9992          1.0000            1.0000        55.60
HnswRaBitQ-m32-ex8-ef60 (query)                        5_466.19       234.36     5_700.54       0.9993          1.0000            1.0000        55.60
HnswRaBitQ-m32-ex8-ef120 (query)                       5_466.19       302.99     5_769.18       0.9993          1.0000            1.0000        55.60
HnswRaBitQ-m32-ex8 (self)                              5_466.19     9_024.19    14_490.37       0.9992          1.0000            1.0000        55.60
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
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 256D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.74       681.23       713.98       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.74     2_296.82     2_329.57       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              144.67       358.76       503.42       0.0971          1.7176            1.5960         7.12
ExhaustiveTQ-b2-rf5 (query)                              144.67       451.42       596.09       0.2336          1.2025            1.2205         7.12
ExhaustiveTQ-b2-rf10 (query)                             144.67       605.06       749.73       0.2853          1.1453            1.1620         7.12
ExhaustiveTQ-b2-rf20 (query)                             144.67     1_011.69     1_156.36       0.3809          1.0970            1.0941         7.12
ExhaustiveTQ-b2 (self)                                   144.67     3_280.55     3_425.22       0.3815          1.0980            1.0956         7.12
ExhaustiveTQ-b4-rf0 (query)                              230.56       579.94       810.50       0.1094          1.5329            1.4998        13.22
ExhaustiveTQ-b4-rf5 (query)                              230.56       651.89       882.45       0.2368          1.1884            1.2090        13.22
ExhaustiveTQ-b4-rf10 (query)                             230.56       787.74     1_018.30       0.2884          1.1372            1.1544        13.22
ExhaustiveTQ-b4-rf20 (query)                             230.56     1_151.30     1_381.86       0.3823          1.0940            1.0971        13.22
ExhaustiveTQ-b4 (self)                                   230.56     3_803.62     4_034.18       0.3841          1.0937            1.0948        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                          334.67       103.85       438.52       0.0971          1.7176            1.5960         7.61
IVF-TQ-b2-nl158-np12-rf0 (query)                         334.67       115.40       450.08       0.0971          1.7176            1.5960         7.61
IVF-TQ-b2-nl158-np17-rf0 (query)                         334.67       126.38       461.05       0.0971          1.7176            1.5960         7.61
IVF-TQ-b2-nl158-np7-rf10 (query)                         334.67       282.79       617.46       0.2853          1.1453            1.1620         7.61
IVF-TQ-b2-nl158-np7-rf20 (query)                         334.67       575.27       909.94       0.3809          1.0970            1.0941         7.61
IVF-TQ-b2-nl158-np12-rf10 (query)                        334.67       291.20       625.87       0.2853          1.1453            1.1620         7.61
IVF-TQ-b2-nl158-np12-rf20 (query)                        334.67       591.76       926.43       0.3809          1.0970            1.0941         7.61
IVF-TQ-b2-nl158-np17-rf10 (query)                        334.67       300.29       634.96       0.2853          1.1453            1.1620         7.61
IVF-TQ-b2-nl158-np17-rf20 (query)                        334.67       599.34       934.01       0.3809          1.0970            1.0941         7.61
IVF-TQ-b2-nl158 (self)                                   334.67     1_027.41     1_362.09       0.3815          1.0980            1.0956         7.61
IVF-TQ-b2-nl223-np11-rf0 (query)                         292.20       110.93       403.13       0.0971          1.7176            1.5959         7.74
IVF-TQ-b2-nl223-np14-rf0 (query)                         292.20       116.77       408.97       0.0971          1.7176            1.5959         7.74
IVF-TQ-b2-nl223-np21-rf0 (query)                         292.20       129.12       421.33       0.0971          1.7176            1.5959         7.74
IVF-TQ-b2-nl223-np11-rf10 (query)                        292.20       291.02       583.22       0.2853          1.1453            1.1620         7.74
IVF-TQ-b2-nl223-np11-rf20 (query)                        292.20       547.76       839.96       0.3809          1.0970            1.0941         7.74
IVF-TQ-b2-nl223-np14-rf10 (query)                        292.20       295.31       587.51       0.2853          1.1453            1.1620         7.74
IVF-TQ-b2-nl223-np14-rf20 (query)                        292.20       563.95       856.15       0.3809          1.0970            1.0941         7.74
IVF-TQ-b2-nl223-np21-rf10 (query)                        292.20       322.74       614.94       0.2853          1.1453            1.1620         7.74
IVF-TQ-b2-nl223-np21-rf20 (query)                        292.20       596.19       888.39       0.3809          1.0970            1.0940         7.74
IVF-TQ-b2-nl223 (self)                                   292.20     1_023.54     1_315.74       0.3815          1.0980            1.0956         7.74
IVF-TQ-b2-nl316-np15-rf0 (query)                         278.78       115.38       394.16       0.0971          1.7176            1.5960         7.93
IVF-TQ-b2-nl316-np17-rf0 (query)                         278.78       119.70       398.48       0.0971          1.7176            1.5960         7.93
IVF-TQ-b2-nl316-np25-rf0 (query)                         278.78       132.42       411.20       0.0971          1.7176            1.5960         7.93
IVF-TQ-b2-nl316-np15-rf10 (query)                        278.78       333.22       612.00       0.2853          1.1453            1.1620         7.93
IVF-TQ-b2-nl316-np15-rf20 (query)                        278.78       520.06       798.84       0.3809          1.0970            1.0941         7.93
IVF-TQ-b2-nl316-np17-rf10 (query)                        278.78       288.11       566.89       0.2853          1.1453            1.1620         7.93
IVF-TQ-b2-nl316-np17-rf20 (query)                        278.78       525.43       804.21       0.3809          1.0970            1.0941         7.93
IVF-TQ-b2-nl316-np25-rf10 (query)                        278.78       302.40       581.19       0.2853          1.1453            1.1620         7.93
IVF-TQ-b2-nl316-np25-rf20 (query)                        278.78       550.93       829.71       0.3809          1.0970            1.0941         7.93
IVF-TQ-b2-nl316 (self)                                   278.78     1_040.38     1_319.17       0.3815          1.0980            1.0957         7.93
IVF-TQ-b4-nl158-np7-rf0 (query)                          421.02       141.67       562.69       0.1094          1.5329            1.4998        13.85
IVF-TQ-b4-nl158-np12-rf0 (query)                         421.02       161.09       582.11       0.1094          1.5329            1.4998        13.85
IVF-TQ-b4-nl158-np17-rf0 (query)                         421.02       178.13       599.15       0.1094          1.5329            1.4998        13.85
IVF-TQ-b4-nl158-np7-rf10 (query)                         421.02       331.95       752.97       0.2884          1.1372            1.1544        13.85
IVF-TQ-b4-nl158-np7-rf20 (query)                         421.02       623.84     1_044.86       0.3823          1.0940            1.0971        13.85
IVF-TQ-b4-nl158-np12-rf10 (query)                        421.02       343.54       764.57       0.2884          1.1372            1.1543        13.85
IVF-TQ-b4-nl158-np12-rf20 (query)                        421.02       645.31     1_066.33       0.3823          1.0940            1.0971        13.85
IVF-TQ-b4-nl158-np17-rf10 (query)                        421.02       359.92       780.94       0.2884          1.1372            1.1544        13.85
IVF-TQ-b4-nl158-np17-rf20 (query)                        421.02       658.87     1_079.89       0.3823          1.0940            1.0971        13.85
IVF-TQ-b4-nl158 (self)                                   421.02     1_069.07     1_490.10       0.3841          1.0937            1.0948        13.85
IVF-TQ-b4-nl223-np11-rf0 (query)                         369.43       151.74       521.17       0.1094          1.5329            1.4998        14.05
IVF-TQ-b4-nl223-np14-rf0 (query)                         369.43       166.87       536.30       0.1094          1.5329            1.4998        14.05
IVF-TQ-b4-nl223-np21-rf0 (query)                         369.43       185.44       554.88       0.1094          1.5329            1.4998        14.05
IVF-TQ-b4-nl223-np11-rf10 (query)                        369.43       338.76       708.19       0.2884          1.1372            1.1544        14.05
IVF-TQ-b4-nl223-np11-rf20 (query)                        369.43       601.74       971.18       0.3823          1.0940            1.0971        14.05
IVF-TQ-b4-nl223-np14-rf10 (query)                        369.43       355.76       725.19       0.2884          1.1372            1.1544        14.05
IVF-TQ-b4-nl223-np14-rf20 (query)                        369.43       624.22       993.65       0.3823          1.0940            1.0971        14.05
IVF-TQ-b4-nl223-np21-rf10 (query)                        369.43       369.69       739.13       0.2884          1.1372            1.1544        14.05
IVF-TQ-b4-nl223-np21-rf20 (query)                        369.43       664.34     1_033.77       0.3823          1.0940            1.0971        14.05
IVF-TQ-b4-nl223 (self)                                   369.43     1_050.31     1_419.75       0.3841          1.0937            1.0948        14.05
IVF-TQ-b4-nl316-np15-rf0 (query)                         366.26       158.29       524.54       0.1094          1.5329            1.4998        14.34
IVF-TQ-b4-nl316-np17-rf0 (query)                         366.26       180.45       546.71       0.1094          1.5329            1.4998        14.34
IVF-TQ-b4-nl316-np25-rf0 (query)                         366.26       185.70       551.96       0.1094          1.5329            1.4998        14.34
IVF-TQ-b4-nl316-np15-rf10 (query)                        366.26       332.79       699.04       0.2884          1.1372            1.1544        14.34
IVF-TQ-b4-nl316-np15-rf20 (query)                        366.26       570.78       937.04       0.3823          1.0940            1.0971        14.34
IVF-TQ-b4-nl316-np17-rf10 (query)                        366.26       348.87       715.13       0.2884          1.1372            1.1544        14.34
IVF-TQ-b4-nl316-np17-rf20 (query)                        366.26       585.23       951.49       0.3823          1.0940            1.0971        14.34
IVF-TQ-b4-nl316-np25-rf10 (query)                        366.26       368.99       735.25       0.2884          1.1372            1.1544        14.34
IVF-TQ-b4-nl316-np25-rf20 (query)                        366.26       623.13       989.39       0.3823          1.0940            1.0971        14.34
IVF-TQ-b4-nl316 (self)                                   366.26     1_065.20     1_431.46       0.3841          1.0937            1.0948        14.34
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 512D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.24     1_353.23     1_421.48       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.24     4_501.58     4_569.82       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              334.77       642.57       977.34       0.1208          1.3711            1.3321        13.97
ExhaustiveTQ-b2-rf5 (query)                              334.77       724.81     1_059.58       0.2421          1.1334            1.1574        13.97
ExhaustiveTQ-b2-rf10 (query)                             334.77       861.22     1_195.99       0.2935          1.0981            1.1178        13.97
ExhaustiveTQ-b2-rf20 (query)                             334.77     1_238.22     1_572.99       0.3880          1.0664            1.0469        13.97
ExhaustiveTQ-b2 (self)                                   334.77     4_057.06     4_391.83       0.3879          1.0667            1.0471        13.97
ExhaustiveTQ-b4-rf0 (query)                              444.21     1_123.81     1_568.02       0.1315          1.3171            1.3127        26.18
ExhaustiveTQ-b4-rf5 (query)                              444.21     1_225.29     1_669.50       0.2471          1.1254            1.1483        26.18
ExhaustiveTQ-b4-rf10 (query)                             444.21     1_360.30     1_804.51       0.2970          1.0929            1.0979        26.18
ExhaustiveTQ-b4-rf20 (query)                             444.21     1_727.75     2_171.96       0.3883          1.0643            1.0493        26.18
ExhaustiveTQ-b4 (self)                                   444.21     5_699.39     6_143.61       0.3882          1.0646            1.0495        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                          654.73       189.95       844.68       0.1208          1.3711            1.3321        14.78
IVF-TQ-b2-nl158-np12-rf0 (query)                         654.73       209.35       864.08       0.1208          1.3711            1.3321        14.78
IVF-TQ-b2-nl158-np17-rf0 (query)                         654.73       225.18       879.91       0.1208          1.3711            1.3321        14.78
IVF-TQ-b2-nl158-np7-rf10 (query)                         654.73       395.47     1_050.20       0.2934          1.0981            1.1178        14.78
IVF-TQ-b2-nl158-np7-rf20 (query)                         654.73       744.00     1_398.73       0.3880          1.0664            1.0469        14.78
IVF-TQ-b2-nl158-np12-rf10 (query)                        654.73       414.14     1_068.87       0.2935          1.0981            1.1178        14.78
IVF-TQ-b2-nl158-np12-rf20 (query)                        654.73       734.03     1_388.76       0.3880          1.0664            1.0469        14.78
IVF-TQ-b2-nl158-np17-rf10 (query)                        654.73       435.12     1_089.86       0.2934          1.0981            1.1178        14.78
IVF-TQ-b2-nl158-np17-rf20 (query)                        654.73       758.95     1_413.68       0.3880          1.0664            1.0469        14.78
IVF-TQ-b2-nl158 (self)                                   654.73     1_393.57     2_048.30       0.3879          1.0667            1.0471        14.78
IVF-TQ-b2-nl223-np11-rf0 (query)                         537.18       203.74       740.93       0.1208          1.3711            1.3321        15.00
IVF-TQ-b2-nl223-np14-rf0 (query)                         537.18       212.86       750.04       0.1208          1.3711            1.3321        15.00
IVF-TQ-b2-nl223-np21-rf0 (query)                         537.18       231.30       768.48       0.1208          1.3711            1.3321        15.00
IVF-TQ-b2-nl223-np11-rf10 (query)                        537.18       398.88       936.07       0.2935          1.0981            1.1178        15.00
IVF-TQ-b2-nl223-np11-rf20 (query)                        537.18       681.95     1_219.14       0.3880          1.0664            1.0469        15.00
IVF-TQ-b2-nl223-np14-rf10 (query)                        537.18       404.87       942.05       0.2935          1.0981            1.1178        15.00
IVF-TQ-b2-nl223-np14-rf20 (query)                        537.18       702.60     1_239.78       0.3880          1.0664            1.0469        15.00
IVF-TQ-b2-nl223-np21-rf10 (query)                        537.18       430.26       967.44       0.2934          1.0981            1.1178        15.00
IVF-TQ-b2-nl223-np21-rf20 (query)                        537.18       741.97     1_279.16       0.3880          1.0664            1.0469        15.00
IVF-TQ-b2-nl223 (self)                                   537.18     1_380.54     1_917.73       0.3879          1.0667            1.0471        15.00
IVF-TQ-b2-nl316-np15-rf0 (query)                         561.33       209.17       770.50       0.1208          1.3711            1.3321        15.38
IVF-TQ-b2-nl316-np17-rf0 (query)                         561.33       216.74       778.08       0.1208          1.3711            1.3321        15.38
IVF-TQ-b2-nl316-np25-rf0 (query)                         561.33       232.41       793.74       0.1208          1.3711            1.3321        15.38
IVF-TQ-b2-nl316-np15-rf10 (query)                        561.33       401.73       963.07       0.2934          1.0981            1.1178        15.38
IVF-TQ-b2-nl316-np15-rf20 (query)                        561.33       674.76     1_236.09       0.3880          1.0664            1.0469        15.38
IVF-TQ-b2-nl316-np17-rf10 (query)                        561.33       408.55       969.88       0.2934          1.0981            1.1178        15.38
IVF-TQ-b2-nl316-np17-rf20 (query)                        561.33       689.55     1_250.89       0.3880          1.0664            1.0469        15.38
IVF-TQ-b2-nl316-np25-rf10 (query)                        561.33       433.71       995.04       0.2934          1.0981            1.1178        15.38
IVF-TQ-b2-nl316-np25-rf20 (query)                        561.33       728.27     1_289.61       0.3880          1.0664            1.0469        15.38
IVF-TQ-b2-nl316 (self)                                   561.33     1_411.01     1_972.34       0.3879          1.0667            1.0471        15.38
IVF-TQ-b4-nl158-np7-rf0 (query)                          748.25       264.79     1_013.04       0.1315          1.3171            1.3127        27.30
IVF-TQ-b4-nl158-np12-rf0 (query)                         748.25       298.74     1_046.99       0.1315          1.3171            1.3127        27.30
IVF-TQ-b4-nl158-np17-rf0 (query)                         748.25       325.45     1_073.70       0.1315          1.3171            1.3127        27.30
IVF-TQ-b4-nl158-np7-rf10 (query)                         748.25       485.10     1_233.35       0.2970          1.0929            1.0979        27.30
IVF-TQ-b4-nl158-np7-rf20 (query)                         748.25       798.91     1_547.16       0.3883          1.0643            1.0493        27.30
IVF-TQ-b4-nl158-np12-rf10 (query)                        748.25       515.10     1_263.35       0.2970          1.0929            1.0979        27.30
IVF-TQ-b4-nl158-np12-rf20 (query)                        748.25       843.26     1_591.51       0.3883          1.0643            1.0493        27.30
IVF-TQ-b4-nl158-np17-rf10 (query)                        748.25       550.46     1_298.71       0.2970          1.0929            1.0979        27.30
IVF-TQ-b4-nl158-np17-rf20 (query)                        748.25       886.94     1_635.19       0.3883          1.0643            1.0493        27.30
IVF-TQ-b4-nl158 (self)                                   748.25     1_574.87     2_323.12       0.3882          1.0646            1.0495        27.30
IVF-TQ-b4-nl223-np11-rf0 (query)                         651.85       283.10       934.95       0.1315          1.3171            1.3127        27.61
IVF-TQ-b4-nl223-np14-rf0 (query)                         651.85       325.21       977.06       0.1315          1.3171            1.3127        27.61
IVF-TQ-b4-nl223-np21-rf0 (query)                         651.85       334.31       986.16       0.1315          1.3171            1.3127        27.61
IVF-TQ-b4-nl223-np11-rf10 (query)                        651.85       492.53     1_144.38       0.2970          1.0929            1.0979        27.61
IVF-TQ-b4-nl223-np11-rf20 (query)                        651.85       779.99     1_431.84       0.3883          1.0643            1.0493        27.61
IVF-TQ-b4-nl223-np14-rf10 (query)                        651.85       509.04     1_160.90       0.2970          1.0929            1.0979        27.61
IVF-TQ-b4-nl223-np14-rf20 (query)                        651.85       811.96     1_463.81       0.3883          1.0643            1.0493        27.61
IVF-TQ-b4-nl223-np21-rf10 (query)                        651.85       545.43     1_197.29       0.2970          1.0929            1.0979        27.61
IVF-TQ-b4-nl223-np21-rf20 (query)                        651.85       866.31     1_518.16       0.3883          1.0643            1.0493        27.61
IVF-TQ-b4-nl223 (self)                                   651.85     1_583.21     2_235.07       0.3882          1.0646            1.0495        27.61
IVF-TQ-b4-nl316-np15-rf0 (query)                         744.22       294.17     1_038.39       0.1315          1.3171            1.3127        28.19
IVF-TQ-b4-nl316-np17-rf0 (query)                         744.22       313.29     1_057.51       0.1315          1.3171            1.3127        28.19
IVF-TQ-b4-nl316-np25-rf0 (query)                         744.22       336.32     1_080.54       0.1315          1.3171            1.3127        28.19
IVF-TQ-b4-nl316-np15-rf10 (query)                        744.22       499.50     1_243.72       0.2970          1.0929            1.0979        28.19
IVF-TQ-b4-nl316-np15-rf20 (query)                        744.22       778.78     1_523.00       0.3883          1.0643            1.0493        28.19
IVF-TQ-b4-nl316-np17-rf10 (query)                        744.22       510.13     1_254.35       0.2970          1.0929            1.0979        28.19
IVF-TQ-b4-nl316-np17-rf20 (query)                        744.22       801.41     1_545.63       0.3883          1.0643            1.0493        28.19
IVF-TQ-b4-nl316-np25-rf10 (query)                        744.22       554.12     1_298.35       0.2970          1.0929            1.0979        28.19
IVF-TQ-b4-nl316-np25-rf20 (query)                        744.22       853.73     1_597.95       0.3883          1.0643            1.0493        28.19
IVF-TQ-b4-nl316 (self)                                   744.22     1_612.61     2_356.83       0.3882          1.0646            1.0495        28.19
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 768D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        99.75     1_931.94     2_031.69       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.75     6_454.04     6_553.78       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              606.09       949.59     1_555.67       0.1292          1.2710            1.2627        21.33
ExhaustiveTQ-b2-rf5 (query)                              606.09     1_040.95     1_647.04       0.2468          1.1062            1.1333        21.33
ExhaustiveTQ-b2-rf10 (query)                             606.09     1_184.14     1_790.22       0.3000          1.0773            1.0631        21.33
ExhaustiveTQ-b2-rf20 (query)                             606.09     1_578.63     2_184.72       0.3956          1.0509            1.0334        21.33
ExhaustiveTQ-b2 (self)                                   606.09     5_156.29     5_762.38       0.3973          1.0507            1.0331        21.33
ExhaustiveTQ-b4-rf0 (query)                              737.20     1_744.70     2_481.90       0.1340          1.2532            1.2592        39.64
ExhaustiveTQ-b4-rf5 (query)                              737.20     1_838.65     2_575.84       0.2400          1.1136            1.1402        39.64
ExhaustiveTQ-b4-rf10 (query)                             737.20     1_975.40     2_712.60       0.2870          1.0888            1.1143        39.64
ExhaustiveTQ-b4-rf20 (query)                             737.20     2_375.20     3_112.40       0.3752          1.0657            1.0806        39.64
ExhaustiveTQ-b4 (self)                                   737.20     7_766.60     8_503.79       0.3767          1.0654            1.0640        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_033.00       286.33     1_319.34       0.1292          1.2710            1.2627        22.44
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_033.00       310.67     1_343.67       0.1292          1.2710            1.2627        22.44
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_033.00       331.10     1_364.10       0.1292          1.2710            1.2628        22.44
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_033.00       519.52     1_552.52       0.3000          1.0773            1.0631        22.44
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_033.00       847.95     1_880.95       0.3956          1.0509            1.0334        22.44
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_033.00       544.42     1_577.42       0.3000          1.0773            1.0631        22.44
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_033.00       897.43     1_930.43       0.3956          1.0509            1.0334        22.44
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_033.00       568.59     1_601.59       0.3000          1.0773            1.0631        22.44
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_033.00       981.96     2_014.96       0.3956          1.0509            1.0334        22.44
IVF-TQ-b2-nl158 (self)                                 1_033.00     1_772.48     2_805.48       0.3973          1.0507            1.0331        22.44
IVF-TQ-b2-nl223-np11-rf0 (query)                         862.29       300.21     1_162.50       0.1292          1.2710            1.2627        22.78
IVF-TQ-b2-nl223-np14-rf0 (query)                         862.29       325.02     1_187.31       0.1292          1.2710            1.2627        22.78
IVF-TQ-b2-nl223-np21-rf0 (query)                         862.29       340.63     1_202.92       0.1292          1.2710            1.2627        22.78
IVF-TQ-b2-nl223-np11-rf10 (query)                        862.29       526.62     1_388.91       0.3000          1.0773            1.0631        22.78
IVF-TQ-b2-nl223-np11-rf20 (query)                        862.29       834.12     1_696.41       0.3956          1.0509            1.0334        22.78
IVF-TQ-b2-nl223-np14-rf10 (query)                        862.29       541.99     1_404.29       0.3000          1.0773            1.0631        22.78
IVF-TQ-b2-nl223-np14-rf20 (query)                        862.29       860.23     1_722.52       0.3956          1.0509            1.0334        22.78
IVF-TQ-b2-nl223-np21-rf10 (query)                        862.29       615.02     1_477.31       0.3000          1.0773            1.0631        22.78
IVF-TQ-b2-nl223-np21-rf20 (query)                        862.29       908.81     1_771.10       0.3956          1.0509            1.0334        22.78
IVF-TQ-b2-nl223 (self)                                   862.29     1_788.89     2_651.18       0.3973          1.0507            1.0331        22.78
IVF-TQ-b2-nl316-np15-rf0 (query)                         988.23       313.35     1_301.58       0.1292          1.2710            1.2627        23.39
IVF-TQ-b2-nl316-np17-rf0 (query)                         988.23       321.07     1_309.30       0.1292          1.2710            1.2627        23.39
IVF-TQ-b2-nl316-np25-rf0 (query)                         988.23       346.68     1_334.91       0.1292          1.2710            1.2627        23.39
IVF-TQ-b2-nl316-np15-rf10 (query)                        988.23       532.62     1_520.85       0.3000          1.0773            1.0631        23.39
IVF-TQ-b2-nl316-np15-rf20 (query)                        988.23       830.80     1_819.03       0.3956          1.0509            1.0334        23.39
IVF-TQ-b2-nl316-np17-rf10 (query)                        988.23       540.48     1_528.71       0.3000          1.0773            1.0631        23.39
IVF-TQ-b2-nl316-np17-rf20 (query)                        988.23       872.38     1_860.61       0.3956          1.0509            1.0334        23.39
IVF-TQ-b2-nl316-np25-rf10 (query)                        988.23       570.61     1_558.84       0.3000          1.0773            1.0631        23.39
IVF-TQ-b2-nl316-np25-rf20 (query)                        988.23       874.69     1_862.92       0.3956          1.0509            1.0334        23.39
IVF-TQ-b2-nl316 (self)                                   988.23     1_853.61     2_841.84       0.3973          1.0507            1.0331        23.39
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_232.41       404.23     1_636.64       0.1340          1.2532            1.2592        41.21
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_232.41       450.57     1_682.98       0.1340          1.2532            1.2592        41.21
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_232.41       489.38     1_721.80       0.1340          1.2532            1.2592        41.21
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_232.41       644.44     1_876.85       0.2870          1.0888            1.1143        41.21
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_232.41       986.00     2_218.42       0.3752          1.0657            1.0806        41.21
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_232.41       700.90     1_933.31       0.2870          1.0888            1.1143        41.21
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_232.41     1_058.22     2_290.63       0.3752          1.0657            1.0806        41.21
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_232.41       743.82     1_976.24       0.2870          1.0888            1.1143        41.21
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_232.41     1_116.03     2_348.44       0.3752          1.0657            1.0807        41.21
IVF-TQ-b4-nl158 (self)                                 1_232.41     2_046.68     3_279.09       0.3767          1.0654            1.0641        41.21
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_000.74       431.90     1_432.65       0.1340          1.2532            1.2592        41.69
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_000.74       457.89     1_458.64       0.1340          1.2532            1.2592        41.69
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_000.74       504.26     1_505.00       0.1340          1.2532            1.2592        41.69
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_000.74       667.90     1_668.65       0.2870          1.0888            1.1143        41.69
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_000.74       979.32     1_980.06       0.3752          1.0657            1.0807        41.69
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_000.74       694.57     1_695.32       0.2870          1.0888            1.1143        41.69
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_000.74     1_020.63     2_021.37       0.3752          1.0657            1.0807        41.69
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_000.74       747.78     1_748.53       0.2870          1.0888            1.1143        41.69
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_000.74     1_084.55     2_085.29       0.3752          1.0657            1.0807        41.69
IVF-TQ-b4-nl223 (self)                                 1_000.74     2_094.82     3_095.56       0.3767          1.0654            1.0640        41.69
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_117.77       453.40     1_571.17       0.1340          1.2532            1.2592        42.65
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_117.77       470.24     1_588.01       0.1340          1.2532            1.2592        42.65
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_117.77       518.56     1_636.34       0.1340          1.2532            1.2592        42.65
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_117.77       679.96     1_797.73       0.2870          1.0888            1.1143        42.65
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_117.77       973.84     2_091.62       0.3752          1.0657            1.0806        42.65
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_117.77       695.85     1_813.62       0.2870          1.0888            1.1143        42.65
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_117.77       992.96     2_110.73       0.3752          1.0657            1.0806        42.65
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_117.77       748.83     1_866.60       0.2870          1.0888            1.1143        42.65
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_117.77     1_053.11     2_170.88       0.3752          1.0657            1.0807        42.65
IVF-TQ-b4-nl316 (self)                                 1_117.77     2_192.80     3_310.57       0.3767          1.0654            1.0641        42.65
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 256D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.82       729.58       762.40       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.82     2_438.02     2_470.84       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              143.83       365.34       509.17       0.0755          2.3282            1.9295         7.12
ExhaustiveTQ-b2-rf5 (query)                              143.83       436.99       580.83       0.2072          1.3307            1.3578         7.12
ExhaustiveTQ-b2-rf10 (query)                             143.83       566.62       710.45       0.2886          1.2207            1.2321         7.12
ExhaustiveTQ-b2-rf20 (query)                             143.83       932.18     1_076.01       0.4151          1.1328            1.1147         7.12
ExhaustiveTQ-b2 (self)                                   143.83     3_088.07     3_231.91       0.4136          1.1619            1.1367         7.12
ExhaustiveTQ-b4-rf0 (query)                              224.29       581.12       805.40       0.1023          1.7129            1.7532        13.22
ExhaustiveTQ-b4-rf5 (query)                              224.29       662.90       887.19       0.2385          1.2770            1.2999        13.22
ExhaustiveTQ-b4-rf10 (query)                             224.29       791.15     1_015.44       0.3201          1.1873            1.1952        13.22
ExhaustiveTQ-b4-rf20 (query)                             224.29     1_147.82     1_372.11       0.4480          1.1142            1.1029        13.22
ExhaustiveTQ-b4 (self)                                   224.29     3_788.92     4_013.21       0.4462          1.1397            1.1285        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                          321.90       100.40       422.30       0.0755          2.3282            1.9295         7.60
IVF-TQ-b2-nl158-np12-rf0 (query)                         321.90       117.09       438.99       0.0755          2.3282            1.9295         7.60
IVF-TQ-b2-nl158-np17-rf0 (query)                         321.90       143.64       465.54       0.0755          2.3282            1.9295         7.60
IVF-TQ-b2-nl158-np7-rf10 (query)                         321.90       274.39       596.29       0.2886          1.2207            1.2321         7.60
IVF-TQ-b2-nl158-np7-rf20 (query)                         321.90       562.44       884.34       0.4151          1.1328            1.1147         7.60
IVF-TQ-b2-nl158-np12-rf10 (query)                        321.90       293.63       615.52       0.2886          1.2207            1.2321         7.60
IVF-TQ-b2-nl158-np12-rf20 (query)                        321.90       593.32       915.22       0.4151          1.1328            1.1147         7.60
IVF-TQ-b2-nl158-np17-rf10 (query)                        321.90       315.96       637.86       0.2886          1.2207            1.2322         7.60
IVF-TQ-b2-nl158-np17-rf20 (query)                        321.90       633.21       955.11       0.4151          1.1328            1.1147         7.60
IVF-TQ-b2-nl158 (self)                                   321.90     1_049.54     1_371.43       0.4136          1.1619            1.1367         7.60
IVF-TQ-b2-nl223-np11-rf0 (query)                         255.57       106.39       361.95       0.0756          2.3281            1.9295         7.74
IVF-TQ-b2-nl223-np14-rf0 (query)                         255.57       112.86       368.43       0.0755          2.3282            1.9295         7.74
IVF-TQ-b2-nl223-np21-rf0 (query)                         255.57       136.88       392.45       0.0755          2.3282            1.9295         7.74
IVF-TQ-b2-nl223-np11-rf10 (query)                        255.57       280.82       536.39       0.2886          1.2207            1.2321         7.74
IVF-TQ-b2-nl223-np11-rf20 (query)                        255.57       525.45       781.02       0.4151          1.1328            1.1147         7.74
IVF-TQ-b2-nl223-np14-rf10 (query)                        255.57       274.59       530.16       0.2886          1.2207            1.2321         7.74
IVF-TQ-b2-nl223-np14-rf20 (query)                        255.57       527.32       782.88       0.4151          1.1328            1.1147         7.74
IVF-TQ-b2-nl223-np21-rf10 (query)                        255.57       313.37       568.94       0.2886          1.2207            1.2321         7.74
IVF-TQ-b2-nl223-np21-rf20 (query)                        255.57       584.14       839.71       0.4151          1.1328            1.1147         7.74
IVF-TQ-b2-nl223 (self)                                   255.57     1_051.94     1_307.50       0.4136          1.1619            1.1367         7.74
IVF-TQ-b2-nl316-np15-rf0 (query)                         305.72       110.94       416.66       0.0756          2.3281            1.9295         7.92
IVF-TQ-b2-nl316-np17-rf0 (query)                         305.72       114.79       420.51       0.0755          2.3282            1.9295         7.92
IVF-TQ-b2-nl316-np25-rf0 (query)                         305.72       134.69       440.41       0.0755          2.3282            1.9295         7.92
IVF-TQ-b2-nl316-np15-rf10 (query)                        305.72       267.79       573.51       0.2886          1.2207            1.2321         7.92
IVF-TQ-b2-nl316-np15-rf20 (query)                        305.72       502.97       808.69       0.4151          1.1328            1.1147         7.92
IVF-TQ-b2-nl316-np17-rf10 (query)                        305.72       273.16       578.88       0.2886          1.2207            1.2321         7.92
IVF-TQ-b2-nl316-np17-rf20 (query)                        305.72       512.22       817.94       0.4151          1.1328            1.1147         7.92
IVF-TQ-b2-nl316-np25-rf10 (query)                        305.72       299.65       605.37       0.2886          1.2207            1.2322         7.92
IVF-TQ-b2-nl316-np25-rf20 (query)                        305.72       549.10       854.82       0.4151          1.1328            1.1147         7.92
IVF-TQ-b2-nl316 (self)                                   305.72     1_043.52     1_349.24       0.4136          1.1619            1.1367         7.92
IVF-TQ-b4-nl158-np7-rf0 (query)                          427.06       137.24       564.29       0.1023          1.7129            1.7533        13.85
IVF-TQ-b4-nl158-np12-rf0 (query)                         427.06       162.37       589.43       0.1023          1.7129            1.7533        13.85
IVF-TQ-b4-nl158-np17-rf0 (query)                         427.06       193.22       620.28       0.1023          1.7130            1.7533        13.85
IVF-TQ-b4-nl158-np7-rf10 (query)                         427.06       313.20       740.26       0.3201          1.1873            1.1952        13.85
IVF-TQ-b4-nl158-np7-rf20 (query)                         427.06       604.02     1_031.07       0.4480          1.1142            1.1029        13.85
IVF-TQ-b4-nl158-np12-rf10 (query)                        427.06       349.00       776.05       0.3201          1.1873            1.1952        13.85
IVF-TQ-b4-nl158-np12-rf20 (query)                        427.06       650.68     1_077.74       0.4480          1.1142            1.1029        13.85
IVF-TQ-b4-nl158-np17-rf10 (query)                        427.06       383.90       810.96       0.3201          1.1873            1.1952        13.85
IVF-TQ-b4-nl158-np17-rf20 (query)                        427.06       698.99     1_126.05       0.4480          1.1142            1.1029        13.85
IVF-TQ-b4-nl158 (self)                                   427.06     1_075.55     1_502.61       0.4462          1.1397            1.1285        13.85
IVF-TQ-b4-nl223-np11-rf0 (query)                         392.22       148.52       540.74       0.1023          1.7129            1.7532        14.05
IVF-TQ-b4-nl223-np14-rf0 (query)                         392.22       157.18       549.40       0.1023          1.7129            1.7532        14.05
IVF-TQ-b4-nl223-np21-rf0 (query)                         392.22       197.18       589.40       0.1023          1.7129            1.7532        14.05
IVF-TQ-b4-nl223-np11-rf10 (query)                        392.22       319.07       711.29       0.3201          1.1873            1.1952        14.05
IVF-TQ-b4-nl223-np11-rf20 (query)                        392.22       586.31       978.53       0.4480          1.1142            1.1029        14.05
IVF-TQ-b4-nl223-np14-rf10 (query)                        392.22       333.61       725.83       0.3201          1.1873            1.1952        14.05
IVF-TQ-b4-nl223-np14-rf20 (query)                        392.22       597.57       989.79       0.4480          1.1142            1.1029        14.05
IVF-TQ-b4-nl223-np21-rf10 (query)                        392.22       382.48       774.70       0.3201          1.1873            1.1952        14.05
IVF-TQ-b4-nl223-np21-rf20 (query)                        392.22       658.50     1_050.72       0.4480          1.1142            1.1029        14.05
IVF-TQ-b4-nl223 (self)                                   392.22     1_071.83     1_464.05       0.4462          1.1397            1.1285        14.05
IVF-TQ-b4-nl316-np15-rf0 (query)                         370.53       152.46       522.99       0.1023          1.7129            1.7532        14.32
IVF-TQ-b4-nl316-np17-rf0 (query)                         370.53       159.18       529.71       0.1023          1.7129            1.7532        14.32
IVF-TQ-b4-nl316-np25-rf0 (query)                         370.53       191.71       562.24       0.1023          1.7129            1.7532        14.32
IVF-TQ-b4-nl316-np15-rf10 (query)                        370.53       321.90       692.43       0.3201          1.1873            1.1952        14.32
IVF-TQ-b4-nl316-np15-rf20 (query)                        370.53       562.53       933.06       0.4480          1.1142            1.1029        14.32
IVF-TQ-b4-nl316-np17-rf10 (query)                        370.53       327.73       698.26       0.3201          1.1873            1.1952        14.32
IVF-TQ-b4-nl316-np17-rf20 (query)                        370.53       572.89       943.42       0.4480          1.1142            1.1029        14.32
IVF-TQ-b4-nl316-np25-rf10 (query)                        370.53       366.92       737.45       0.3201          1.1873            1.1952        14.32
IVF-TQ-b4-nl316-np25-rf20 (query)                        370.53       628.90       999.43       0.4480          1.1142            1.1029        14.32
IVF-TQ-b4-nl316 (self)                                   370.53     1_079.23     1_449.76       0.4462          1.1397            1.1286        14.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 512D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.28     1_313.55     1_381.83       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.28     4_598.30     4_666.58       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              341.17       632.12       973.29       0.0844          1.6537            1.5906        13.97
ExhaustiveTQ-b2-rf5 (query)                              341.17       715.25     1_056.42       0.2173          1.2230            1.2550        13.97
ExhaustiveTQ-b2-rf10 (query)                             341.17       908.40     1_249.57       0.2887          1.1550            1.1707        13.97
ExhaustiveTQ-b2-rf20 (query)                             341.17     1_352.45     1_693.62       0.4020          1.0974            1.0847        13.97
ExhaustiveTQ-b2 (self)                                   341.17     4_062.00     4_403.17       0.4025          1.1135            1.0971        13.97
ExhaustiveTQ-b4-rf0 (query)                              444.96     1_118.96     1_563.92       0.1045          1.5026            1.5346        26.18
ExhaustiveTQ-b4-rf5 (query)                              444.96     1_216.75     1_661.71       0.2294          1.2109            1.2410        26.18
ExhaustiveTQ-b4-rf10 (query)                             444.96     1_355.69     1_800.64       0.2943          1.1499            1.1675        26.18
ExhaustiveTQ-b4-rf20 (query)                             444.96     1_733.14     2_178.09       0.4029          1.0975            1.0930        26.18
ExhaustiveTQ-b4 (self)                                   444.96     5_705.12     6_150.08       0.4037          1.1130            1.1087        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                          643.34       185.48       828.82       0.0844          1.6537            1.5906        14.76
IVF-TQ-b2-nl158-np12-rf0 (query)                         643.34       205.31       848.65       0.0844          1.6537            1.5906        14.76
IVF-TQ-b2-nl158-np17-rf0 (query)                         643.34       225.73       869.07       0.0844          1.6537            1.5905        14.76
IVF-TQ-b2-nl158-np7-rf10 (query)                         643.34       398.96     1_042.30       0.2887          1.1550            1.1707        14.76
IVF-TQ-b2-nl158-np7-rf20 (query)                         643.34       719.45     1_362.79       0.4020          1.0974            1.0847        14.76
IVF-TQ-b2-nl158-np12-rf10 (query)                        643.34       417.58     1_060.92       0.2887          1.1550            1.1707        14.76
IVF-TQ-b2-nl158-np12-rf20 (query)                        643.34       763.16     1_406.50       0.4020          1.0974            1.0847        14.76
IVF-TQ-b2-nl158-np17-rf10 (query)                        643.34       448.92     1_092.26       0.2887          1.1550            1.1707        14.76
IVF-TQ-b2-nl158-np17-rf20 (query)                        643.34       803.72     1_447.06       0.4020          1.0974            1.0847        14.76
IVF-TQ-b2-nl158 (self)                                   643.34     1_384.24     2_027.58       0.4025          1.1135            1.0971        14.76
IVF-TQ-b2-nl223-np11-rf0 (query)                         529.48       196.20       725.68       0.0844          1.6537            1.5906        14.99
IVF-TQ-b2-nl223-np14-rf0 (query)                         529.48       204.72       734.20       0.0844          1.6537            1.5906        14.99
IVF-TQ-b2-nl223-np21-rf0 (query)                         529.48       232.41       761.89       0.0844          1.6537            1.5906        14.99
IVF-TQ-b2-nl223-np11-rf10 (query)                        529.48       398.43       927.91       0.2887          1.1550            1.1707        14.99
IVF-TQ-b2-nl223-np11-rf20 (query)                        529.48       693.12     1_222.60       0.4020          1.0974            1.0847        14.99
IVF-TQ-b2-nl223-np14-rf10 (query)                        529.48       407.56       937.04       0.2887          1.1550            1.1707        14.99
IVF-TQ-b2-nl223-np14-rf20 (query)                        529.48       717.07     1_246.54       0.4020          1.0974            1.0847        14.99
IVF-TQ-b2-nl223-np21-rf10 (query)                        529.48       434.25       963.73       0.2887          1.1550            1.1707        14.99
IVF-TQ-b2-nl223-np21-rf20 (query)                        529.48       754.47     1_283.95       0.4020          1.0974            1.0847        14.99
IVF-TQ-b2-nl223 (self)                                   529.48     1_354.95     1_884.43       0.4025          1.1135            1.0971        14.99
IVF-TQ-b2-nl316-np15-rf0 (query)                         562.72       203.63       766.35       0.0844          1.6537            1.5906        15.33
IVF-TQ-b2-nl316-np17-rf0 (query)                         562.72       209.66       772.38       0.0844          1.6537            1.5906        15.33
IVF-TQ-b2-nl316-np25-rf0 (query)                         562.72       231.90       794.62       0.0844          1.6537            1.5905        15.33
IVF-TQ-b2-nl316-np15-rf10 (query)                        562.72       396.63       959.35       0.2887          1.1550            1.1707        15.33
IVF-TQ-b2-nl316-np15-rf20 (query)                        562.72       654.26     1_216.98       0.4020          1.0974            1.0847        15.33
IVF-TQ-b2-nl316-np17-rf10 (query)                        562.72       401.83       964.55       0.2887          1.1550            1.1707        15.33
IVF-TQ-b2-nl316-np17-rf20 (query)                        562.72       665.14     1_227.86       0.4020          1.0974            1.0847        15.33
IVF-TQ-b2-nl316-np25-rf10 (query)                        562.72       433.18       995.90       0.2887          1.1550            1.1707        15.33
IVF-TQ-b2-nl316-np25-rf20 (query)                        562.72       710.89     1_273.61       0.4020          1.0974            1.0847        15.33
IVF-TQ-b2-nl316 (self)                                   562.72     1_377.77     1_940.49       0.4025          1.1135            1.0971        15.33
IVF-TQ-b4-nl158-np7-rf0 (query)                          745.39       255.78     1_001.17       0.1045          1.5026            1.5346        27.26
IVF-TQ-b4-nl158-np12-rf0 (query)                         745.39       290.35     1_035.73       0.1045          1.5026            1.5346        27.26
IVF-TQ-b4-nl158-np17-rf0 (query)                         745.39       328.50     1_073.89       0.1045          1.5026            1.5346        27.26
IVF-TQ-b4-nl158-np7-rf10 (query)                         745.39       482.17     1_227.56       0.2943          1.1499            1.1675        27.26
IVF-TQ-b4-nl158-np7-rf20 (query)                         745.39       811.73     1_557.11       0.4029          1.0975            1.0930        27.26
IVF-TQ-b4-nl158-np12-rf10 (query)                        745.39       518.33     1_263.72       0.2943          1.1499            1.1675        27.26
IVF-TQ-b4-nl158-np12-rf20 (query)                        745.39       868.13     1_613.52       0.4029          1.0975            1.0930        27.26
IVF-TQ-b4-nl158-np17-rf10 (query)                        745.39       558.69     1_304.07       0.2943          1.1499            1.1675        27.26
IVF-TQ-b4-nl158-np17-rf20 (query)                        745.39       924.66     1_670.05       0.4029          1.0975            1.0930        27.26
IVF-TQ-b4-nl158 (self)                                   745.39     1_559.25     2_304.64       0.4037          1.1130            1.1087        27.26
IVF-TQ-b4-nl223-np11-rf0 (query)                         634.09       271.90       905.99       0.1045          1.5027            1.5346        27.58
IVF-TQ-b4-nl223-np14-rf0 (query)                         634.09       303.10       937.19       0.1045          1.5027            1.5346        27.58
IVF-TQ-b4-nl223-np21-rf0 (query)                         634.09       350.22       984.31       0.1045          1.5026            1.5346        27.58
IVF-TQ-b4-nl223-np11-rf10 (query)                        634.09       557.15     1_191.24       0.2943          1.1499            1.1675        27.58
IVF-TQ-b4-nl223-np11-rf20 (query)                        634.09       849.06     1_483.15       0.4029          1.0975            1.0930        27.58
IVF-TQ-b4-nl223-np14-rf10 (query)                        634.09       547.00     1_181.09       0.2943          1.1499            1.1675        27.58
IVF-TQ-b4-nl223-np14-rf20 (query)                        634.09       886.22     1_520.31       0.4029          1.0975            1.0930        27.58
IVF-TQ-b4-nl223-np21-rf10 (query)                        634.09       587.02     1_221.11       0.2943          1.1499            1.1675        27.58
IVF-TQ-b4-nl223-np21-rf20 (query)                        634.09       882.64     1_516.73       0.4029          1.0975            1.0930        27.58
IVF-TQ-b4-nl223 (self)                                   634.09     1_569.71     2_203.80       0.4037          1.1130            1.1087        27.58
IVF-TQ-b4-nl316-np15-rf0 (query)                         685.68       287.09       972.77       0.1045          1.5026            1.5346        28.09
IVF-TQ-b4-nl316-np17-rf0 (query)                         685.68       296.10       981.78       0.1045          1.5026            1.5346        28.09
IVF-TQ-b4-nl316-np25-rf0 (query)                         685.68       336.75     1_022.43       0.1045          1.5026            1.5346        28.09
IVF-TQ-b4-nl316-np15-rf10 (query)                        685.68       491.56     1_177.24       0.2943          1.1499            1.1675        28.09
IVF-TQ-b4-nl316-np15-rf20 (query)                        685.68       756.55     1_442.23       0.4029          1.0975            1.0930        28.09
IVF-TQ-b4-nl316-np17-rf10 (query)                        685.68       502.89     1_188.57       0.2943          1.1499            1.1675        28.09
IVF-TQ-b4-nl316-np17-rf20 (query)                        685.68       777.44     1_463.12       0.4029          1.0975            1.0930        28.09
IVF-TQ-b4-nl316-np25-rf10 (query)                        685.68       549.69     1_235.37       0.2943          1.1499            1.1675        28.09
IVF-TQ-b4-nl316-np25-rf20 (query)                        685.68       828.19     1_513.87       0.4029          1.0975            1.0930        28.09
IVF-TQ-b4-nl316 (self)                                   685.68     1_591.77     2_277.45       0.4037          1.1130            1.1087        28.09
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 768D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        99.68     1_851.74     1_951.41       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.68     6_279.42     6_379.10       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              596.01       927.35     1_523.36       0.0841          1.5107            1.4226        21.33
ExhaustiveTQ-b2-rf5 (query)                              596.01     1_017.65     1_613.66       0.2144          1.1739            1.2057        21.33
ExhaustiveTQ-b2-rf10 (query)                             596.01     1_168.80     1_764.81       0.2771          1.1266            1.1512        21.33
ExhaustiveTQ-b2-rf20 (query)                             596.01     1_576.07     2_172.08       0.3769          1.0843            1.0725        21.33
ExhaustiveTQ-b2 (self)                                   596.01     5_160.17     5_756.18       0.3767          1.0935            1.0803        21.33
ExhaustiveTQ-b4-rf0 (query)                              732.30     1_991.67     2_723.97       0.0985          1.4232            1.4110        39.64
ExhaustiveTQ-b4-rf5 (query)                              732.30     2_068.02     2_800.31       0.2167          1.1746            1.2047        39.64
ExhaustiveTQ-b4-rf10 (query)                             732.30     2_182.11     2_914.41       0.2692          1.1311            1.1558        39.64
ExhaustiveTQ-b4-rf20 (query)                             732.30     2_627.09     3_359.39       0.3604          1.0923            1.1071        39.64
ExhaustiveTQ-b4 (self)                                   732.30     8_055.11     8_787.40       0.3608          1.1024            1.1182        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_107.49       296.71     1_404.20       0.0841          1.5107            1.4226        22.43
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_107.49       322.01     1_429.50       0.0841          1.5107            1.4226        22.43
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_107.49       348.74     1_456.23       0.0841          1.5107            1.4226        22.43
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_107.49       526.68     1_634.17       0.2771          1.1266            1.1512        22.43
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_107.49       883.24     1_990.74       0.3770          1.0843            1.0725        22.43
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_107.49       565.94     1_673.43       0.2771          1.1266            1.1512        22.43
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_107.49     1_080.85     2_188.34       0.3770          1.0843            1.0725        22.43
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_107.49       653.74     1_761.23       0.2771          1.1266            1.1512        22.43
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_107.49       995.54     2_103.04       0.3770          1.0843            1.0725        22.43
IVF-TQ-b2-nl158 (self)                                 1_107.49     1_859.11     2_966.60       0.3767          1.0935            1.0803        22.43
IVF-TQ-b2-nl223-np11-rf0 (query)                         890.32       298.44     1_188.76       0.0841          1.5107            1.4226        22.78
IVF-TQ-b2-nl223-np14-rf0 (query)                         890.32       310.44     1_200.75       0.0841          1.5107            1.4226        22.78
IVF-TQ-b2-nl223-np21-rf0 (query)                         890.32       353.45     1_243.77       0.0841          1.5107            1.4226        22.78
IVF-TQ-b2-nl223-np11-rf10 (query)                        890.32       537.15     1_427.46       0.2771          1.1266            1.1512        22.78
IVF-TQ-b2-nl223-np11-rf20 (query)                        890.32       864.27     1_754.59       0.3770          1.0843            1.0725        22.78
IVF-TQ-b2-nl223-np14-rf10 (query)                        890.32       553.46     1_443.78       0.2771          1.1266            1.1512        22.78
IVF-TQ-b2-nl223-np14-rf20 (query)                        890.32       897.16     1_787.48       0.3770          1.0843            1.0725        22.78
IVF-TQ-b2-nl223-np21-rf10 (query)                        890.32       592.69     1_483.01       0.2771          1.1266            1.1512        22.78
IVF-TQ-b2-nl223-np21-rf20 (query)                        890.32       946.71     1_837.03       0.3769          1.0843            1.0725        22.78
IVF-TQ-b2-nl223 (self)                                   890.32     1_871.57     2_761.89       0.3767          1.0935            1.0803        22.78
IVF-TQ-b2-nl316-np15-rf0 (query)                         961.87       321.46     1_283.34       0.0841          1.5107            1.4226        23.35
IVF-TQ-b2-nl316-np17-rf0 (query)                         961.87       317.91     1_279.78       0.0841          1.5107            1.4226        23.35
IVF-TQ-b2-nl316-np25-rf0 (query)                         961.87       347.12     1_309.00       0.0841          1.5107            1.4226        23.35
IVF-TQ-b2-nl316-np15-rf10 (query)                        961.87       538.92     1_500.79       0.2771          1.1266            1.1512        23.35
IVF-TQ-b2-nl316-np15-rf20 (query)                        961.87       837.87     1_799.75       0.3770          1.0843            1.0724        23.35
IVF-TQ-b2-nl316-np17-rf10 (query)                        961.87       556.44     1_518.31       0.2771          1.1266            1.1512        23.35
IVF-TQ-b2-nl316-np17-rf20 (query)                        961.87       879.60     1_841.48       0.3770          1.0843            1.0724        23.35
IVF-TQ-b2-nl316-np25-rf10 (query)                        961.87       578.52     1_540.39       0.2771          1.1266            1.1512        23.35
IVF-TQ-b2-nl316-np25-rf20 (query)                        961.87       902.95     1_864.83       0.3770          1.0843            1.0725        23.35
IVF-TQ-b2-nl316 (self)                                   961.87     1_931.01     2_892.88       0.3767          1.0935            1.0803        23.35
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_509.00       409.00     1_918.00       0.0985          1.4232            1.4110        41.20
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_509.00       453.62     1_962.62       0.0985          1.4232            1.4110        41.20
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_509.00       514.73     2_023.73       0.0985          1.4232            1.4110        41.20
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_509.00       658.79     2_167.79       0.2692          1.1311            1.1558        41.20
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_509.00     1_019.80     2_528.80       0.3604          1.0923            1.1071        41.20
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_509.00       717.64     2_226.64       0.2692          1.1311            1.1558        41.20
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_509.00     1_113.60     2_622.60       0.3604          1.0923            1.1071        41.20
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_509.00       783.42     2_292.42       0.2692          1.1311            1.1558        41.20
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_509.00     1_203.99     2_713.00       0.3604          1.0923            1.1071        41.20
IVF-TQ-b4-nl158 (self)                                 1_509.00     2_194.85     3_703.86       0.3608          1.1024            1.1182        41.20
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_205.75       435.14     1_640.90       0.0985          1.4232            1.4110        41.69
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_205.75       458.40     1_664.16       0.0985          1.4232            1.4110        41.69
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_205.75       523.68     1_729.43       0.0985          1.4232            1.4110        41.69
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_205.75       686.59     1_892.34       0.2692          1.1311            1.1558        41.69
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_205.75     1_020.12     2_225.88       0.3604          1.0923            1.1071        41.69
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_205.75       713.53     1_919.29       0.2692          1.1311            1.1558        41.69
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_205.75     1_047.15     2_252.90       0.3604          1.0923            1.1071        41.69
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_205.75       768.05     1_973.80       0.2692          1.1311            1.1558        41.69
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_205.75     1_139.26     2_345.01       0.3604          1.0923            1.1071        41.69
IVF-TQ-b4-nl223 (self)                                 1_205.75     2_249.30     3_455.06       0.3608          1.1024            1.1182        41.69
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_132.69       458.20     1_590.89       0.0985          1.4232            1.4110        42.56
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_132.69       470.60     1_603.29       0.0985          1.4232            1.4110        42.56
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_132.69       525.83     1_658.52       0.0985          1.4232            1.4110        42.56
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_132.69       683.19     1_815.88       0.2692          1.1311            1.1558        42.56
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_132.69       982.89     2_115.58       0.3604          1.0923            1.1071        42.56
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_132.69       698.56     1_831.26       0.2692          1.1311            1.1558        42.56
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_132.69     1_005.02     2_137.71       0.3604          1.0923            1.1071        42.56
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_132.69       757.18     1_889.87       0.2692          1.1311            1.1558        42.56
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_132.69     1_075.15     2_207.84       0.3604          1.0923            1.1071        42.56
IVF-TQ-b4-nl316 (self)                                 1_132.69     2_289.84     3_422.53       0.3608          1.1024            1.1182        42.56
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Cell embeddings data

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 256D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.51       716.18       748.69       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.51     2_425.87     2_458.39       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              143.72       366.91       510.63       0.7918          1.0898            1.0632         7.12
ExhaustiveTQ-b2-rf5 (query)                              143.72       442.55       586.27       0.9995          1.0000            1.0000         7.12
ExhaustiveTQ-b2-rf10 (query)                             143.72       581.36       725.08       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b2-rf20 (query)                             143.72       948.58     1_092.30       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b2 (self)                                   143.72     3_135.64     3_279.36       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b4-rf0 (query)                              228.34       589.51       817.85       0.8727          1.0322            1.0183        13.22
ExhaustiveTQ-b4-rf5 (query)                              228.34       667.16       895.51       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4-rf10 (query)                             228.34       795.24     1_023.59       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4-rf20 (query)                             228.34     1_159.25     1_387.59       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4 (self)                                   228.34     4_044.54     4_272.88       1.0000          1.0000            1.0000        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                          323.78       119.48       443.26       0.7916          1.0896            1.0636         7.61
IVF-TQ-b2-nl158-np12-rf0 (query)                         323.78       159.50       483.28       0.7919          1.0897            1.0632         7.61
IVF-TQ-b2-nl158-np17-rf0 (query)                         323.78       195.88       519.67       0.7918          1.0898            1.0632         7.61
IVF-TQ-b2-nl158-np7-rf10 (query)                         323.78       323.87       647.65       0.9975          1.0006            1.0000         7.61
IVF-TQ-b2-nl158-np7-rf20 (query)                         323.78       603.84       927.62       0.9975          1.0006            1.0000         7.61
IVF-TQ-b2-nl158-np12-rf10 (query)                        323.78       381.55       705.33       0.9999          1.0000            1.0000         7.61
IVF-TQ-b2-nl158-np12-rf20 (query)                        323.78       703.55     1_027.33       0.9999          1.0000            1.0000         7.61
IVF-TQ-b2-nl158-np17-rf10 (query)                        323.78       426.20       749.98       1.0000          1.0000            1.0000         7.61
IVF-TQ-b2-nl158-np17-rf20 (query)                        323.78       764.02     1_087.80       1.0000          1.0000            1.0000         7.61
IVF-TQ-b2-nl158 (self)                                   323.78     1_127.88     1_451.66       1.0000          1.0000            1.0000         7.61
IVF-TQ-b2-nl223-np11-rf0 (query)                         248.82       126.17       374.99       0.7919          1.0895            1.0632         7.75
IVF-TQ-b2-nl223-np14-rf0 (query)                         248.82       144.01       392.83       0.7919          1.0897            1.0632         7.75
IVF-TQ-b2-nl223-np21-rf0 (query)                         248.82       182.49       431.31       0.7918          1.0898            1.0632         7.75
IVF-TQ-b2-nl223-np11-rf10 (query)                        248.82       311.95       560.78       0.9995          1.0001            1.0000         7.75
IVF-TQ-b2-nl223-np11-rf20 (query)                        248.82       573.34       822.16       0.9995          1.0001            1.0000         7.75
IVF-TQ-b2-nl223-np14-rf10 (query)                        248.82       336.25       585.08       0.9999          1.0000            1.0000         7.75
IVF-TQ-b2-nl223-np14-rf20 (query)                        248.82       613.31       862.13       0.9999          1.0000            1.0000         7.75
IVF-TQ-b2-nl223-np21-rf10 (query)                        248.82       390.06       638.88       1.0000          1.0000            1.0000         7.75
IVF-TQ-b2-nl223-np21-rf20 (query)                        248.82       704.20       953.02       1.0000          1.0000            1.0000         7.75
IVF-TQ-b2-nl223 (self)                                   248.82     1_043.75     1_292.58       1.0000          1.0000            1.0000         7.75
IVF-TQ-b2-nl316-np15-rf0 (query)                         283.21       139.91       423.11       0.7918          1.0898            1.0632         7.93
IVF-TQ-b2-nl316-np17-rf0 (query)                         283.21       137.71       420.91       0.7918          1.0898            1.0632         7.93
IVF-TQ-b2-nl316-np25-rf0 (query)                         283.21       168.98       452.19       0.7918          1.0898            1.0632         7.93
IVF-TQ-b2-nl316-np15-rf10 (query)                        283.21       304.69       587.90       0.9996          1.0001            1.0000         7.93
IVF-TQ-b2-nl316-np15-rf20 (query)                        283.21       562.33       845.53       0.9996          1.0001            1.0000         7.93
IVF-TQ-b2-nl316-np17-rf10 (query)                        283.21       319.70       602.90       0.9999          1.0000            1.0000         7.93
IVF-TQ-b2-nl316-np17-rf20 (query)                        283.21       582.23       865.44       0.9999          1.0000            1.0000         7.93
IVF-TQ-b2-nl316-np25-rf10 (query)                        283.21       363.82       647.03       1.0000          1.0000            1.0000         7.93
IVF-TQ-b2-nl316-np25-rf20 (query)                        283.21       655.33       938.54       1.0000          1.0000            1.0000         7.93
IVF-TQ-b2-nl316 (self)                                   283.21       995.86     1_279.07       1.0000          1.0000            1.0000         7.93
IVF-TQ-b4-nl158-np7-rf0 (query)                          398.94       167.29       566.23       0.8719          1.0326            1.0188        13.87
IVF-TQ-b4-nl158-np12-rf0 (query)                         398.94       234.44       633.38       0.8727          1.0322            1.0183        13.87
IVF-TQ-b4-nl158-np17-rf0 (query)                         398.94       292.65       691.60       0.8727          1.0322            1.0183        13.87
IVF-TQ-b4-nl158-np7-rf10 (query)                         398.94       369.33       768.27       0.9975          1.0006            1.0000        13.87
IVF-TQ-b4-nl158-np7-rf20 (query)                         398.94       645.50     1_044.44       0.9975          1.0006            1.0000        13.87
IVF-TQ-b4-nl158-np12-rf10 (query)                        398.94       456.72       855.66       0.9999          1.0000            1.0000        13.87
IVF-TQ-b4-nl158-np12-rf20 (query)                        398.94       773.76     1_172.70       0.9999          1.0000            1.0000        13.87
IVF-TQ-b4-nl158-np17-rf10 (query)                        398.94       524.81       923.75       1.0000          1.0000            1.0000        13.87
IVF-TQ-b4-nl158-np17-rf20 (query)                        398.94       869.31     1_268.25       1.0000          1.0000            1.0000        13.87
IVF-TQ-b4-nl158 (self)                                   398.94     1_250.04     1_648.98       1.0000          1.0000            1.0000        13.87
IVF-TQ-b4-nl223-np11-rf0 (query)                         327.88       177.97       505.85       0.8727          1.0322            1.0184        14.07
IVF-TQ-b4-nl223-np14-rf0 (query)                         327.88       206.02       533.91       0.8727          1.0322            1.0183        14.07
IVF-TQ-b4-nl223-np21-rf0 (query)                         327.88       270.32       598.21       0.8727          1.0322            1.0183        14.07
IVF-TQ-b4-nl223-np11-rf10 (query)                        327.88       364.56       692.44       0.9995          1.0001            1.0000        14.07
IVF-TQ-b4-nl223-np11-rf20 (query)                        327.88       625.79       953.68       0.9995          1.0001            1.0000        14.07
IVF-TQ-b4-nl223-np14-rf10 (query)                        327.88       400.49       728.37       0.9999          1.0000            1.0000        14.07
IVF-TQ-b4-nl223-np14-rf20 (query)                        327.88       678.92     1_006.81       0.9999          1.0000            1.0000        14.07
IVF-TQ-b4-nl223-np21-rf10 (query)                        327.88       479.92       807.80       1.0000          1.0000            1.0000        14.07
IVF-TQ-b4-nl223-np21-rf20 (query)                        327.88       798.73     1_126.61       1.0000          1.0000            1.0000        14.07
IVF-TQ-b4-nl223 (self)                                   327.88     1_165.69     1_493.58       1.0000          1.0000            1.0000        14.07
IVF-TQ-b4-nl316-np15-rf0 (query)                         356.62       181.24       537.86       0.8726          1.0322            1.0184        14.35
IVF-TQ-b4-nl316-np17-rf0 (query)                         356.62       194.04       550.66       0.8727          1.0322            1.0184        14.35
IVF-TQ-b4-nl316-np25-rf0 (query)                         356.62       248.52       605.14       0.8727          1.0322            1.0183        14.35
IVF-TQ-b4-nl316-np15-rf10 (query)                        356.62       359.65       716.27       0.9996          1.0001            1.0000        14.35
IVF-TQ-b4-nl316-np15-rf20 (query)                        356.62       618.85       975.46       0.9996          1.0001            1.0000        14.35
IVF-TQ-b4-nl316-np17-rf10 (query)                        356.62       375.98       732.59       0.9999          1.0000            1.0000        14.35
IVF-TQ-b4-nl316-np17-rf20 (query)                        356.62       641.15       997.76       0.9999          1.0000            1.0000        14.35
IVF-TQ-b4-nl316-np25-rf10 (query)                        356.62       444.28       800.90       1.0000          1.0000            1.0000        14.35
IVF-TQ-b4-nl316-np25-rf20 (query)                        356.62       735.49     1_092.11       1.0000          1.0000            1.0000        14.35
IVF-TQ-b4-nl316 (self)                                   356.62     1_092.94     1_449.56       1.0000          1.0000            1.0000        14.35
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 512D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.39     1_320.62     1_389.01       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.39     4_444.18     4_512.57       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              334.88       631.96       966.83       0.8424          1.0447            1.0331        13.97
ExhaustiveTQ-b2-rf5 (query)                              334.88       717.88     1_052.76       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2-rf10 (query)                             334.88       862.64     1_197.52       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2-rf20 (query)                             334.88     1_250.39     1_585.27       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2 (self)                                   334.88     4_106.61     4_441.49       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b4-rf0 (query)                              456.72     1_118.36     1_575.08       0.8985          1.0191            1.0110        26.18
ExhaustiveTQ-b4-rf5 (query)                              456.72     1_228.37     1_685.09       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4-rf10 (query)                             456.72     1_361.36     1_818.08       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4-rf20 (query)                             456.72     1_733.20     2_189.92       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4 (self)                                   456.72     5_714.09     6_170.81       1.0000          1.0000            1.0000        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                          645.74       215.45       861.19       0.8421          1.0449            1.0333        14.80
IVF-TQ-b2-nl158-np12-rf0 (query)                         645.74       279.96       925.70       0.8424          1.0447            1.0331        14.80
IVF-TQ-b2-nl158-np17-rf0 (query)                         645.74       337.29       983.04       0.8424          1.0447            1.0331        14.80
IVF-TQ-b2-nl158-np7-rf10 (query)                         645.74       436.90     1_082.65       0.9987          1.0003            1.0000        14.80
IVF-TQ-b2-nl158-np7-rf20 (query)                         645.74       734.29     1_380.03       0.9987          1.0003            1.0000        14.80
IVF-TQ-b2-nl158-np12-rf10 (query)                        645.74       520.03     1_165.77       0.9999          1.0000            1.0000        14.80
IVF-TQ-b2-nl158-np12-rf20 (query)                        645.74       854.78     1_500.52       0.9999          1.0000            1.0000        14.80
IVF-TQ-b2-nl158-np17-rf10 (query)                        645.74       590.74     1_236.48       1.0000          1.0000            1.0000        14.80
IVF-TQ-b2-nl158-np17-rf20 (query)                        645.74       948.27     1_594.01       1.0000          1.0000            1.0000        14.80
IVF-TQ-b2-nl158 (self)                                   645.74     1_501.16     2_146.91       1.0000          1.0000            1.0000        14.80
IVF-TQ-b2-nl223-np11-rf0 (query)                         522.12       229.25       751.37       0.8424          1.0447            1.0331        15.03
IVF-TQ-b2-nl223-np14-rf0 (query)                         522.12       256.96       779.08       0.8424          1.0447            1.0331        15.03
IVF-TQ-b2-nl223-np21-rf0 (query)                         522.12       317.59       839.72       0.8424          1.0447            1.0331        15.03
IVF-TQ-b2-nl223-np11-rf10 (query)                        522.12       439.57       961.69       0.9997          1.0001            1.0000        15.03
IVF-TQ-b2-nl223-np11-rf20 (query)                        522.12       726.52     1_248.65       0.9997          1.0001            1.0000        15.03
IVF-TQ-b2-nl223-np14-rf10 (query)                        522.12       472.09       994.22       0.9999          1.0000            1.0000        15.03
IVF-TQ-b2-nl223-np14-rf20 (query)                        522.12       777.72     1_299.84       0.9999          1.0000            1.0000        15.03
IVF-TQ-b2-nl223-np21-rf10 (query)                        522.12       543.20     1_065.32       1.0000          1.0000            1.0000        15.03
IVF-TQ-b2-nl223-np21-rf20 (query)                        522.12       877.71     1_399.83       1.0000          1.0000            1.0000        15.03
IVF-TQ-b2-nl223 (self)                                   522.12     1_447.43     1_969.56       1.0000          1.0000            1.0000        15.03
IVF-TQ-b2-nl316-np15-rf0 (query)                         563.76       233.61       797.37       0.8424          1.0447            1.0331        15.39
IVF-TQ-b2-nl316-np17-rf0 (query)                         563.76       248.23       811.98       0.8424          1.0447            1.0331        15.39
IVF-TQ-b2-nl316-np25-rf0 (query)                         563.76       300.50       864.25       0.8424          1.0447            1.0331        15.39
IVF-TQ-b2-nl316-np15-rf10 (query)                        563.76       439.89     1_003.65       0.9999          1.0000            1.0000        15.39
IVF-TQ-b2-nl316-np15-rf20 (query)                        563.76       730.28     1_294.03       0.9999          1.0000            1.0000        15.39
IVF-TQ-b2-nl316-np17-rf10 (query)                        563.76       462.58     1_026.34       1.0000          1.0000            1.0000        15.39
IVF-TQ-b2-nl316-np17-rf20 (query)                        563.76       757.82     1_321.57       1.0000          1.0000            1.0000        15.39
IVF-TQ-b2-nl316-np25-rf10 (query)                        563.76       520.02     1_083.78       1.0000          1.0000            1.0000        15.39
IVF-TQ-b2-nl316-np25-rf20 (query)                        563.76       838.19     1_401.95       1.0000          1.0000            1.0000        15.39
IVF-TQ-b2-nl316 (self)                                   563.76     1_390.23     1_953.99       1.0000          1.0000            1.0000        15.39
IVF-TQ-b4-nl158-np7-rf0 (query)                          745.28       312.80     1_058.08       0.8979          1.0193            1.0112        27.33
IVF-TQ-b4-nl158-np12-rf0 (query)                         745.28       435.53     1_180.82       0.8985          1.0191            1.0110        27.33
IVF-TQ-b4-nl158-np17-rf0 (query)                         745.28       535.10     1_280.38       0.8985          1.0191            1.0110        27.33
IVF-TQ-b4-nl158-np7-rf10 (query)                         745.28       568.57     1_313.85       0.9987          1.0003            1.0000        27.33
IVF-TQ-b4-nl158-np7-rf20 (query)                         745.28       836.90     1_582.18       0.9987          1.0003            1.0000        27.33
IVF-TQ-b4-nl158-np12-rf10 (query)                        745.28       670.46     1_415.75       0.9999          1.0000            1.0000        27.33
IVF-TQ-b4-nl158-np12-rf20 (query)                        745.28     1_009.46     1_754.75       0.9999          1.0000            1.0000        27.33
IVF-TQ-b4-nl158-np17-rf10 (query)                        745.28       785.06     1_530.35       1.0000          1.0000            1.0000        27.33
IVF-TQ-b4-nl158-np17-rf20 (query)                        745.28     1_145.19     1_890.47       1.0000          1.0000            1.0000        27.33
IVF-TQ-b4-nl158 (self)                                   745.28     1_752.36     2_497.65       1.0000          1.0000            1.0000        27.33
IVF-TQ-b4-nl223-np11-rf0 (query)                         652.67       341.03       993.70       0.8984          1.0192            1.0111        27.66
IVF-TQ-b4-nl223-np14-rf0 (query)                         652.67       388.70     1_041.36       0.8985          1.0191            1.0110        27.66
IVF-TQ-b4-nl223-np21-rf0 (query)                         652.67       499.63     1_152.30       0.8985          1.0191            1.0110        27.66
IVF-TQ-b4-nl223-np11-rf10 (query)                        652.67       547.40     1_200.06       0.9997          1.0001            1.0000        27.66
IVF-TQ-b4-nl223-np11-rf20 (query)                        652.67       842.02     1_494.68       0.9997          1.0001            1.0000        27.66
IVF-TQ-b4-nl223-np14-rf10 (query)                        652.67       602.99     1_255.66       0.9999          1.0000            1.0000        27.66
IVF-TQ-b4-nl223-np14-rf20 (query)                        652.67       910.47     1_563.13       0.9999          1.0000            1.0000        27.66
IVF-TQ-b4-nl223-np21-rf10 (query)                        652.67       721.83     1_374.49       1.0000          1.0000            1.0000        27.66
IVF-TQ-b4-nl223-np21-rf20 (query)                        652.67     1_053.81     1_706.48       1.0000          1.0000            1.0000        27.66
IVF-TQ-b4-nl223 (self)                                   652.67     1_676.89     2_329.55       1.0000          1.0000            1.0000        27.66
IVF-TQ-b4-nl316-np15-rf0 (query)                         692.74       345.73     1_038.47       0.8985          1.0191            1.0110        28.20
IVF-TQ-b4-nl316-np17-rf0 (query)                         692.74       369.33     1_062.07       0.8985          1.0191            1.0110        28.20
IVF-TQ-b4-nl316-np25-rf0 (query)                         692.74       466.65     1_159.39       0.8985          1.0191            1.0110        28.20
IVF-TQ-b4-nl316-np15-rf10 (query)                        692.74       546.36     1_239.10       0.9999          1.0000            1.0000        28.20
IVF-TQ-b4-nl316-np15-rf20 (query)                        692.74       845.54     1_538.28       0.9999          1.0000            1.0000        28.20
IVF-TQ-b4-nl316-np17-rf10 (query)                        692.74       573.48     1_266.22       1.0000          1.0000            1.0000        28.20
IVF-TQ-b4-nl316-np17-rf20 (query)                        692.74       878.92     1_571.66       1.0000          1.0000            1.0000        28.20
IVF-TQ-b4-nl316-np25-rf10 (query)                        692.74       679.39     1_372.13       1.0000          1.0000            1.0000        28.20
IVF-TQ-b4-nl316-np25-rf20 (query)                        692.74     1_000.01     1_692.75       1.0000          1.0000            1.0000        28.20
IVF-TQ-b4-nl316 (self)                                   692.74     1_642.80     2_335.54       1.0000          1.0000            1.0000        28.20
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.13.0 (commit v0.12.2-51-g4075c56), run on 2026-11-16
=====================================================================================================================================================
Benchmark: 50k samples, 768D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       100.13     1_964.38     2_064.51       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.13     6_663.04     6_763.16       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              617.39       986.80     1_604.18       0.8736          1.0271            1.0200        21.33
ExhaustiveTQ-b2-rf5 (query)                              617.39     1_068.02     1_685.40       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2-rf10 (query)                             617.39     1_211.95     1_829.34       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2-rf20 (query)                             617.39     1_624.55     2_241.94       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2 (self)                                   617.39     5_328.27     5_945.65       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b4-rf0 (query)                              742.06     1_774.30     2_516.36       0.9097          1.0146            1.0083        39.64
ExhaustiveTQ-b4-rf5 (query)                              742.06     1_875.96     2_618.02       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4-rf10 (query)                             742.06     1_997.76     2_739.82       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4-rf20 (query)                             742.06     2_386.36     3_128.42       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4 (self)                                   742.06     7_880.28     8_622.34       1.0000          1.0000            1.0000        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_057.06       333.41     1_390.46       0.8735          1.0272            1.0201        22.43
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_057.06       425.58     1_482.64       0.8736          1.0271            1.0200        22.43
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_057.06       505.99     1_563.05       0.8736          1.0271            1.0200        22.43
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_057.06       595.35     1_652.41       0.9994          1.0001            1.0000        22.43
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_057.06       919.43     1_976.49       0.9994          1.0001            1.0000        22.43
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_057.06       695.80     1_752.86       1.0000          1.0000            1.0000        22.43
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_057.06     1_059.53     2_116.59       1.0000          1.0000            1.0000        22.43
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_057.06       781.89     1_838.95       1.0000          1.0000            1.0000        22.43
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_057.06     1_157.93     2_214.98       1.0000          1.0000            1.0000        22.43
IVF-TQ-b2-nl158 (self)                                 1_057.06     1_986.38     3_043.44       1.0000          1.0000            1.0000        22.43
IVF-TQ-b2-nl223-np11-rf0 (query)                         915.09       348.05     1_263.13       0.8736          1.0271            1.0200        22.77
IVF-TQ-b2-nl223-np14-rf0 (query)                         915.09       388.81     1_303.90       0.8736          1.0271            1.0200        22.77
IVF-TQ-b2-nl223-np21-rf0 (query)                         915.09       473.74     1_388.83       0.8736          1.0271            1.0200        22.77
IVF-TQ-b2-nl223-np11-rf10 (query)                        915.09       591.85     1_506.93       0.9999          1.0000            1.0000        22.77
IVF-TQ-b2-nl223-np11-rf20 (query)                        915.09       914.86     1_829.95       0.9999          1.0000            1.0000        22.77
IVF-TQ-b2-nl223-np14-rf10 (query)                        915.09       635.69     1_550.78       1.0000          1.0000            1.0000        22.77
IVF-TQ-b2-nl223-np14-rf20 (query)                        915.09     1_052.58     1_967.67       1.0000          1.0000            1.0000        22.77
IVF-TQ-b2-nl223-np21-rf10 (query)                        915.09       731.81     1_646.90       1.0000          1.0000            1.0000        22.77
IVF-TQ-b2-nl223-np21-rf20 (query)                        915.09     1_095.33     2_010.42       1.0000          1.0000            1.0000        22.77
IVF-TQ-b2-nl223 (self)                                   915.09     1_960.45     2_875.54       1.0000          1.0000            1.0000        22.77
IVF-TQ-b2-nl316-np15-rf0 (query)                         950.84       363.90     1_314.74       0.8736          1.0271            1.0200        23.34
IVF-TQ-b2-nl316-np17-rf0 (query)                         950.84       377.08     1_327.92       0.8736          1.0271            1.0200        23.34
IVF-TQ-b2-nl316-np25-rf0 (query)                         950.84       450.63     1_401.46       0.8736          1.0271            1.0200        23.34
IVF-TQ-b2-nl316-np15-rf10 (query)                        950.84       608.37     1_559.20       0.9999          1.0000            1.0000        23.34
IVF-TQ-b2-nl316-np15-rf20 (query)                        950.84       941.02     1_891.85       0.9999          1.0000            1.0000        23.34
IVF-TQ-b2-nl316-np17-rf10 (query)                        950.84       617.41     1_568.25       1.0000          1.0000            1.0000        23.34
IVF-TQ-b2-nl316-np17-rf20 (query)                        950.84       953.30     1_904.13       1.0000          1.0000            1.0000        23.34
IVF-TQ-b2-nl316-np25-rf10 (query)                        950.84       712.27     1_663.11       1.0000          1.0000            1.0000        23.34
IVF-TQ-b2-nl316-np25-rf20 (query)                        950.84     1_083.74     2_034.58       1.0000          1.0000            1.0000        23.34
IVF-TQ-b2-nl316 (self)                                   950.84     1_952.31     2_903.15       1.0000          1.0000            1.0000        23.34
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_181.49       489.97     1_671.46       0.9094          1.0147            1.0084        41.20
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_181.49       663.44     1_844.92       0.9097          1.0146            1.0083        41.20
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_181.49       811.92     1_993.41       0.9097          1.0146            1.0083        41.20
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_181.49       755.67     1_937.15       0.9994          1.0001            1.0000        41.20
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_181.49     1_078.78     2_260.27       0.9994          1.0001            1.0000        41.20
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_181.49       938.39     2_119.88       1.0000          1.0000            1.0000        41.20
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_181.49     1_311.89     2_493.38       1.0000          1.0000            1.0000        41.20
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_181.49     1_091.92     2_273.41       1.0000          1.0000            1.0000        41.20
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_181.49     1_474.75     2_656.24       1.0000          1.0000            1.0000        41.20
IVF-TQ-b4-nl158 (self)                                 1_181.49     2_552.16     3_733.65       1.0000          1.0000            1.0000        41.20
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_068.44       526.04     1_594.48       0.9096          1.0146            1.0084        41.68
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_068.44       604.97     1_673.41       0.9097          1.0146            1.0083        41.68
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_068.44       767.20     1_835.64       0.9097          1.0146            1.0083        41.68
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_068.44       765.62     1_834.06       0.9999          1.0000            1.0000        41.68
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_068.44     1_088.37     2_156.82       0.9999          1.0000            1.0000        41.68
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_068.44       843.00     1_911.44       1.0000          1.0000            1.0000        41.68
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_068.44     1_182.22     2_250.66       1.0000          1.0000            1.0000        41.68
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_068.44     1_022.01     2_090.46       1.0000          1.0000            1.0000        41.68
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_068.44     1_372.92     2_441.36       1.0000          1.0000            1.0000        41.68
IVF-TQ-b4-nl223 (self)                                 1_068.44     2_467.66     3_536.10       1.0000          1.0000            1.0000        41.68
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_155.81       547.35     1_703.16       0.9097          1.0146            1.0084        42.54
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_155.81       586.02     1_741.82       0.9097          1.0146            1.0083        42.54
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_155.81       732.25     1_888.06       0.9097          1.0146            1.0083        42.54
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_155.81       772.76     1_928.57       0.9999          1.0000            1.0000        42.54
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_155.81     1_110.52     2_266.32       0.9999          1.0000            1.0000        42.54
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_155.81       919.71     2_075.52       1.0000          1.0000            1.0000        42.54
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_155.81     1_411.91     2_567.71       1.0000          1.0000            1.0000        42.54
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_155.81     1_000.54     2_156.34       1.0000          1.0000            1.0000        42.54
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_155.81     1_336.93     2_492.73       1.0000          1.0000            1.0000        42.54
IVF-TQ-b4-nl316 (self)                                 1_155.81     2_384.59     3_540.39       1.0000          1.0000            1.0000        42.54
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*Each table starts with the ann-search-rs version and commit it was run on.*
*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
