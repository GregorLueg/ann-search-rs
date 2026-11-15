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
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 256D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.87       689.00       721.88       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.87     2_183.05     2_215.92       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                 69.34       237.40       306.73       0.1199          1.4617            1.4199         1.78
ExhaustiveBinary-256-random-rf10 (query)                  69.34       335.24       404.57       0.3411          1.0941            1.0814         1.78
ExhaustiveBinary-256-random-rf20 (query)                  69.34       436.11       505.45       0.4467          1.0571            1.0475         1.78
ExhaustiveBinary-256-random (self)                        69.34     1_099.56     1_168.89       0.3454          1.0895            1.0798         1.78
ExhaustiveBinary-256-pca_no_rr (query)                   103.84       244.70       348.53       0.1153          1.4748            1.4212         1.78
ExhaustiveBinary-256-pca-rf10 (query)                    103.84       347.37       451.20       0.3323          1.1029            1.0834         1.78
ExhaustiveBinary-256-pca-rf20 (query)                    103.84       435.47       539.31       0.4387          1.0631            1.0485         1.78
ExhaustiveBinary-256-pca (self)                          103.84     1_088.46     1_192.30       0.3391          1.0957            1.0813         1.78
ExhaustiveBinary-512-random_no_rr (query)                 82.65       345.70       428.34       0.1588          1.3547            1.3300         3.55
ExhaustiveBinary-512-random-rf10 (query)                  82.65       488.60       571.25       0.3786          1.0692            1.0677         3.55
ExhaustiveBinary-512-random-rf20 (query)                  82.65       554.74       637.38       0.4874          1.0424            1.0395         3.55
ExhaustiveBinary-512-random (self)                        82.65     1_465.89     1_548.54       0.3805          1.0675            1.0675         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   116.44       342.99       459.43       0.1564          1.3535            1.3265         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    116.44       454.15       570.59       0.3789          1.0710            1.0663         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    116.44       557.13       673.58       0.4903          1.0433            1.0387         3.55
ExhaustiveBinary-512-pca (self)                          116.44     1_464.88     1_581.32       0.3823          1.0678            1.0665         3.55
ExhaustiveBinary-1024-random_no_rr (query)               117.05       486.07       603.12       0.1929          1.2764            1.2696         7.10
ExhaustiveBinary-1024-random-rf10 (query)                117.05       605.83       722.89       0.4214          1.0550            1.0552         7.10
ExhaustiveBinary-1024-random-rf20 (query)                117.05       713.69       830.74       0.5434          1.0327            1.0308         7.10
ExhaustiveBinary-1024-random (self)                      117.05     1_969.25     2_086.30       0.4232          1.0547            1.0552         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  153.96       485.75       639.71       0.1921          1.2733            1.2652         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   153.96       601.23       755.19       0.4226          1.0546            1.0544         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   153.96       712.82       866.79       0.5443          1.0326            1.0305         7.10
ExhaustiveBinary-1024-pca (self)                         153.96     1_976.78     2_130.74       0.4236          1.0546            1.0548         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   40.93       414.90       455.83       0.1211          1.4987            1.4523         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    40.93       444.19       485.13       0.3284          1.1039            1.0884         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    40.93       666.76       707.70       0.4385          1.0624            1.0494         1.53
ExhaustiveBinary-256-sign (self)                          40.93     1_445.24     1_486.18       0.3334          1.0988            1.0859         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)              178.12        51.79       229.91       0.1331          1.3842            1.3443         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)             178.12        53.66       231.78       0.1331          1.3842            1.3443         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)             178.12        54.78       232.91       0.1331          1.3842            1.3443         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)             178.12       100.47       278.59       0.3773          1.0740            1.0663         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)             178.12       149.02       327.14       0.4863          1.0458            1.0392         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)            178.12       100.96       279.09       0.3773          1.0740            1.0663         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)            178.12       149.25       327.37       0.4863          1.0458            1.0392         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)            178.12        99.40       277.52       0.3773          1.0740            1.0663         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)            178.12       149.50       327.62       0.4863          1.0458            1.0392         1.93
IVF-Binary-256-nl158-random (self)                       178.12       216.03       394.15       0.3818          1.0691            1.0654         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             172.54        45.04       217.58       0.1408          1.3609            1.3173         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             172.54        45.78       218.32       0.1408          1.3609            1.3173         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             172.54        47.43       219.97       0.1408          1.3609            1.3173         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            172.54        93.54       266.07       0.3889          1.0686            1.0624         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            172.54       144.40       316.94       0.4942          1.0435            1.0378         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            172.54        93.24       265.77       0.3889          1.0686            1.0624         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            172.54       145.21       317.75       0.4942          1.0435            1.0378         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            172.54        97.81       270.34       0.3889          1.0686            1.0624         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            172.54       148.25       320.79       0.4942          1.0435            1.0378         2.00
IVF-Binary-256-nl223-random (self)                       172.54       209.97       382.51       0.3927          1.0646            1.0619         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             205.45        49.60       255.05       0.1497          1.3331            1.2916         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             205.45        48.02       253.48       0.1497          1.3331            1.2916         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             205.45        51.31       256.76       0.1497          1.3331            1.2916         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            205.45        96.96       302.42       0.4042          1.0638            1.0580         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            205.45       147.09       352.54       0.5080          1.0410            1.0353         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            205.45        96.95       302.41       0.4042          1.0638            1.0580         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            205.45       147.95       353.40       0.5080          1.0410            1.0353         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            205.45       102.45       307.90       0.4042          1.0638            1.0580         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            205.45       150.41       355.86       0.5080          1.0410            1.0353         2.09
IVF-Binary-256-nl316-random (self)                       205.45       220.43       425.89       0.4082          1.0595            1.0572         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)                 212.39        40.19       252.58       0.1287          1.4015            1.3547         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)                212.39        43.85       256.25       0.1287          1.4015            1.3547         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)                212.39        44.78       257.17       0.1287          1.4015            1.3547         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)                212.39        89.24       301.63       0.3647          1.0845            1.0700         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)                212.39       140.88       353.27       0.4812          1.0492            1.0399         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)               212.39        90.88       303.27       0.3647          1.0845            1.0700         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)               212.39       143.35       355.75       0.4812          1.0492            1.0399         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)               212.39        93.19       305.58       0.3647          1.0845            1.0700         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)               212.39       145.52       357.91       0.4812          1.0492            1.0399         1.93
IVF-Binary-256-nl158-pca (self)                          212.39       199.93       412.32       0.3712          1.0780            1.0685         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                226.43        43.35       269.78       0.1368          1.3713            1.3234         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                226.43        44.37       270.80       0.1368          1.3713            1.3234         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                226.43        47.07       273.50       0.1368          1.3713            1.3234         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               226.43        92.37       318.80       0.3818          1.0759            1.0639         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               226.43       143.21       369.64       0.4945          1.0462            1.0376         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               226.43        93.27       319.70       0.3818          1.0759            1.0639         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               226.43       144.62       371.05       0.4945          1.0462            1.0376         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               226.43        95.56       321.99       0.3818          1.0759            1.0639         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               226.43       148.04       374.47       0.4945          1.0462            1.0376         2.00
IVF-Binary-256-nl223-pca (self)                          226.43       205.38       431.81       0.3882          1.0696            1.0628         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                253.15        47.25       300.40       0.1484          1.3354            1.2914         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                253.15        48.38       301.53       0.1484          1.3354            1.2914         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                253.15        50.68       303.83       0.1484          1.3354            1.2914         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               253.15        96.18       349.33       0.4017          1.0681            1.0585         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               253.15       146.09       399.24       0.5108          1.0426            1.0349         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               253.15        95.75       348.90       0.4017          1.0681            1.0585         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               253.15       149.22       402.36       0.5108          1.0426            1.0349         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               253.15        99.09       352.24       0.4017          1.0681            1.0585         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               253.15       151.28       404.42       0.5108          1.0426            1.0349         2.09
IVF-Binary-256-nl316-pca (self)                          253.15       218.46       471.60       0.4073          1.0624            1.0573         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)              219.68        58.49       278.17       0.1678          1.3128            1.2918         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)             219.68        61.60       281.28       0.1678          1.3128            1.2918         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)             219.68        65.58       285.26       0.1678          1.3128            1.2918         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)             219.68       112.41       332.10       0.3973          1.0620            1.0608         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)             219.68       165.62       385.31       0.5107          1.0381            1.0353         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)            219.68       115.18       334.87       0.3973          1.0620            1.0608         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)            219.68       167.88       387.57       0.5107          1.0381            1.0353         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)            219.68       117.54       337.22       0.3973          1.0620            1.0608         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)            219.68       172.28       391.96       0.5107          1.0381            1.0353         3.71
IVF-Binary-512-nl158-random (self)                       219.68       286.27       505.96       0.3992          1.0607            1.0608         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             219.41        61.57       280.98       0.1715          1.2995            1.2791         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             219.41        64.14       283.55       0.1715          1.2995            1.2791         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             219.41        67.83       287.24       0.1715          1.2995            1.2791         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            219.41       115.06       334.47       0.4022          1.0605            1.0593         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            219.41       167.64       387.05       0.5135          1.0375            1.0348         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            219.41       116.08       335.49       0.4022          1.0605            1.0593         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            219.41       171.82       391.23       0.5135          1.0375            1.0348         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            219.41       120.38       339.79       0.4022          1.0605            1.0593         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            219.41       178.72       398.13       0.5135          1.0375            1.0348         3.77
IVF-Binary-512-nl223-random (self)                       219.41       288.69       508.10       0.4038          1.0593            1.0592         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             247.17        65.50       312.67       0.1751          1.2887            1.2673         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             247.17        66.45       313.62       0.1751          1.2887            1.2673         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             247.17        69.77       316.94       0.1751          1.2887            1.2673         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            247.17       117.40       364.57       0.4084          1.0592            1.0577         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            247.17       170.04       417.21       0.5194          1.0367            1.0337         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            247.17       117.76       364.92       0.4084          1.0592            1.0577         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            247.17       170.70       417.87       0.5194          1.0367            1.0337         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            247.17       121.25       368.42       0.4084          1.0592            1.0577         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            247.17       174.22       421.38       0.5194          1.0367            1.0337         3.86
IVF-Binary-512-nl316-random (self)                       247.17       295.73       542.89       0.4103          1.0578            1.0576         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)                 255.57        58.43       314.00       0.1651          1.3153            1.2894         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)                255.57        61.29       316.86       0.1651          1.3153            1.2894         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)                255.57        65.31       320.88       0.1651          1.3153            1.2894         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)                255.57       111.60       367.17       0.3980          1.0636            1.0600         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)                255.57       165.26       420.83       0.5124          1.0389            1.0350         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)               255.57       113.27       368.84       0.3980          1.0636            1.0600         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)               255.57       168.05       423.62       0.5124          1.0389            1.0350         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)               255.57       116.74       372.31       0.3980          1.0636            1.0600         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)               255.57       173.28       428.85       0.5124          1.0389            1.0350         3.71
IVF-Binary-512-nl158-pca (self)                          255.57       281.84       537.41       0.4006          1.0614            1.0598         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                245.24        61.22       306.46       0.1692          1.3036            1.2757         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                245.24        65.12       310.36       0.1692          1.3036            1.2757         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                245.24        66.79       312.03       0.1692          1.3036            1.2757         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               245.24       113.27       358.50       0.4045          1.0617            1.0580         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               245.24       168.80       414.03       0.5170          1.0381            1.0341         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               245.24       114.36       359.60       0.4045          1.0617            1.0580         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               245.24       167.78       413.01       0.5170          1.0381            1.0341         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               245.24       118.09       363.33       0.4045          1.0617            1.0580         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               245.24       171.66       416.89       0.5170          1.0381            1.0341         3.77
IVF-Binary-512-nl223-pca (self)                          245.24       286.18       531.42       0.4068          1.0596            1.0581         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                242.93        64.95       307.89       0.1738          1.2897            1.2644         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                242.93        66.74       309.67       0.1738          1.2897            1.2644         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                242.93        69.67       312.60       0.1738          1.2897            1.2644         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               242.93       117.42       360.35       0.4116          1.0598            1.0561         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               242.93       168.82       411.75       0.5247          1.0370            1.0330         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               242.93       116.96       359.89       0.4116          1.0598            1.0561         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               242.93       170.19       413.13       0.5247          1.0370            1.0330         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               242.93       122.63       365.56       0.4116          1.0598            1.0561         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               242.93       174.91       417.84       0.5247          1.0370            1.0330         3.86
IVF-Binary-512-nl316-pca (self)                          242.93       294.93       537.86       0.4141          1.0578            1.0561         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)             220.44        89.25       309.69       0.1963          1.2612            1.2546         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)            220.44        94.12       314.56       0.1963          1.2612            1.2546         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)            220.44       100.27       320.71       0.1963          1.2612            1.2546         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)            220.44       147.18       367.62       0.4314          1.0524            1.0524         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)            220.44       201.66       422.10       0.5544          1.0312            1.0292         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)           220.44       149.66       370.10       0.4314          1.0524            1.0524         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)           220.44       205.51       425.95       0.5544          1.0312            1.0292         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)           220.44       154.66       375.10       0.4314          1.0524            1.0524         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)           220.44       210.93       431.37       0.5544          1.0312            1.0292         7.26
IVF-Binary-1024-nl158-random (self)                      220.44       402.51       622.95       0.4332          1.0521            1.0524         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            221.43        96.14       317.57       0.1975          1.2557            1.2492         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            221.43        95.23       316.66       0.1975          1.2557            1.2492         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            221.43       102.58       324.01       0.1975          1.2557            1.2492         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           221.43       148.78       370.22       0.4340          1.0517            1.0515         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           221.43       204.02       425.45       0.5561          1.0309            1.0289         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           221.43       150.44       371.88       0.4340          1.0517            1.0515         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           221.43       206.84       428.27       0.5561          1.0309            1.0289         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           221.43       157.07       378.50       0.4340          1.0517            1.0515         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           221.43       213.18       434.62       0.5561          1.0309            1.0289         7.32
IVF-Binary-1024-nl223-random (self)                      221.43       404.78       626.21       0.4356          1.0515            1.0517         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            240.12        97.45       337.56       0.1992          1.2506            1.2436         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)            240.12       100.97       341.09       0.1992          1.2506            1.2436         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)            240.12       104.17       344.29       0.1992          1.2506            1.2436         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)           240.12       152.17       392.29       0.4377          1.0511            1.0508         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)           240.12       207.56       447.68       0.5586          1.0306            1.0286         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)           240.12       152.75       392.87       0.4377          1.0511            1.0508         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)           240.12       214.36       454.48       0.5586          1.0306            1.0286         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)           240.12       159.63       399.75       0.4377          1.0511            1.0508         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)           240.12       214.21       454.33       0.5586          1.0306            1.0286         7.41
IVF-Binary-1024-nl316-random (self)                      240.12       428.83       668.94       0.4392          1.0507            1.0509         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)                256.61        89.01       345.62       0.1957          1.2570            1.2500         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)               256.61        93.63       350.24       0.1957          1.2570            1.2500         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)               256.61        98.62       355.24       0.1957          1.2570            1.2500         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)               256.61       144.95       401.57       0.4340          1.0515            1.0511         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)               256.61       202.05       458.67       0.5574          1.0308            1.0283         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)              256.61       148.91       405.53       0.4340          1.0515            1.0511         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)              256.61       206.00       462.61       0.5574          1.0308            1.0283         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)              256.61       154.15       410.76       0.4340          1.0515            1.0511         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)              256.61       210.44       467.05       0.5574          1.0308            1.0283         7.26
IVF-Binary-1024-nl158-pca (self)                         256.61       402.19       658.81       0.4348          1.0515            1.0517         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               251.50        95.44       346.94       0.1972          1.2521            1.2451         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               251.50        94.53       346.03       0.1972          1.2521            1.2451         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               251.50       102.27       353.77       0.1972          1.2521            1.2451         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              251.50       148.74       400.24       0.4364          1.0509            1.0507         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              251.50       202.45       453.95       0.5588          1.0306            1.0281         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              251.50       149.69       401.18       0.4364          1.0509            1.0507         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              251.50       205.02       456.52       0.5588          1.0306            1.0281         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              251.50       154.74       406.24       0.4364          1.0509            1.0507         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              251.50       213.75       465.25       0.5588          1.0306            1.0281         7.32
IVF-Binary-1024-nl223-pca (self)                         251.50       400.24       651.74       0.4366          1.0511            1.0511         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               275.79        96.33       372.12       0.1991          1.2470            1.2400         7.41
IVF-Binary-1024-nl316-np17-rf0-pca (query)               275.79        97.83       373.62       0.1991          1.2470            1.2400         7.41
IVF-Binary-1024-nl316-np25-rf0-pca (query)               275.79       103.42       379.21       0.1991          1.2470            1.2400         7.41
IVF-Binary-1024-nl316-np15-rf10-pca (query)              275.79       150.87       426.66       0.4397          1.0502            1.0497         7.41
IVF-Binary-1024-nl316-np15-rf20-pca (query)              275.79       205.99       481.78       0.5626          1.0302            1.0277         7.41
IVF-Binary-1024-nl316-np17-rf10-pca (query)              275.79       152.51       428.30       0.4397          1.0502            1.0497         7.41
IVF-Binary-1024-nl316-np17-rf20-pca (query)              275.79       206.93       482.72       0.5626          1.0302            1.0277         7.41
IVF-Binary-1024-nl316-np25-rf10-pca (query)              275.79       156.57       432.36       0.4397          1.0502            1.0497         7.41
IVF-Binary-1024-nl316-np25-rf20-pca (query)              275.79       212.23       488.01       0.5626          1.0302            1.0277         7.41
IVF-Binary-1024-nl316-pca (self)                         275.79       406.52       682.30       0.4409          1.0502            1.0503         7.41
IVF-Binary-256-nl158-np7-rf0-sign (query)                152.42       127.81       280.23       0.1217          1.4973            1.4388         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               152.42       130.35       282.76       0.1217          1.4973            1.4388         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               152.42       138.09       290.51       0.1217          1.4973            1.4388         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               152.42       158.96       311.38       0.3499          1.0923            1.0776         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               152.42       283.40       435.82       0.4544          1.0572            1.0453         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              152.42       160.58       313.00       0.3499          1.0923            1.0776         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              152.42       289.07       441.49       0.4544          1.0572            1.0453         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              152.42       165.01       317.43       0.3499          1.0923            1.0776         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              152.42       286.31       438.73       0.4544          1.0572            1.0453         1.68
IVF-Binary-256-nl158-sign (self)                         152.42       419.56       571.98       0.3553          1.0869            1.0760         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               146.58       126.89       273.47       0.1229          1.4837            1.4301         1.74
IVF-Binary-256-nl223-np14-rf0-sign (query)               146.58       127.11       273.69       0.1229          1.4837            1.4301         1.74
IVF-Binary-256-nl223-np21-rf0-sign (query)               146.58       129.74       276.32       0.1229          1.4837            1.4301         1.74
IVF-Binary-256-nl223-np11-rf10-sign (query)              146.58       161.40       307.97       0.3580          1.0888            1.0743         1.74
IVF-Binary-256-nl223-np11-rf20-sign (query)              146.58       284.50       431.08       0.4581          1.0560            1.0446         1.74
IVF-Binary-256-nl223-np14-rf10-sign (query)              146.58       161.66       308.24       0.3580          1.0888            1.0743         1.74
IVF-Binary-256-nl223-np14-rf20-sign (query)              146.58       288.35       434.92       0.4581          1.0560            1.0446         1.74
IVF-Binary-256-nl223-np21-rf10-sign (query)              146.58       165.40       311.97       0.3580          1.0888            1.0743         1.74
IVF-Binary-256-nl223-np21-rf20-sign (query)              146.58       305.70       452.28       0.4581          1.0560            1.0446         1.74
IVF-Binary-256-nl223-sign (self)                         146.58       423.75       570.33       0.3631          1.0837            1.0727         1.74
IVF-Binary-256-nl316-np15-rf0-sign (query)               172.46       128.67       301.13       0.1238          1.4746            1.4172         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               172.46       128.86       301.32       0.1238          1.4746            1.4172         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               172.46       131.96       304.42       0.1238          1.4746            1.4172         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              172.46       169.87       342.33       0.3594          1.0870            1.0734         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              172.46       287.19       459.65       0.4564          1.0564            1.0447         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              172.46       164.31       336.76       0.3594          1.0870            1.0734         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              172.46       288.46       460.91       0.4564          1.0564            1.0447         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              172.46       166.50       338.96       0.3594          1.0870            1.0734         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              172.46       291.50       463.96       0.4564          1.0564            1.0447         1.84
IVF-Binary-256-nl316-sign (self)                         172.46       435.14       607.60       0.3649          1.0824            1.0718         1.84
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 512D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        69.23     1_286.44     1_355.68       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.23     4_321.61     4_390.84       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                135.00       262.96       397.96       0.1109          1.3512            1.3092         2.03
ExhaustiveBinary-256-random-rf10 (query)                 135.00       385.40       520.40       0.3143          1.0825            1.0613         2.03
ExhaustiveBinary-256-random-rf20 (query)                 135.00       520.46       655.46       0.4100          1.0523            1.0368         2.03
ExhaustiveBinary-256-random (self)                       135.00     1_188.56     1_323.56       0.3161          1.0784            1.0600         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   226.80       263.70       490.50       0.1167          1.3480            1.2981         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    226.80       381.74       608.54       0.3159          1.0791            1.0596         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    226.80       514.81       741.61       0.4121          1.0505            1.0362         2.03
ExhaustiveBinary-256-pca (self)                          226.80     1_208.57     1_435.37       0.3171          1.0782            1.0588         2.03
ExhaustiveBinary-512-random_no_rr (query)                212.35       392.82       605.17       0.1528          1.2601            1.2299         4.05
ExhaustiveBinary-512-random-rf10 (query)                 212.35       524.80       737.15       0.3465          1.0565            1.0514         4.05
ExhaustiveBinary-512-random-rf20 (query)                 212.35       648.62       860.97       0.4452          1.0358            1.0312         4.05
ExhaustiveBinary-512-random (self)                       212.35     1_648.83     1_861.18       0.3476          1.0547            1.0512         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   296.85       384.28       681.13       0.1558          1.2535            1.2254         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    296.85       507.14       803.99       0.3512          1.0523            1.0507         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    296.85       634.72       931.57       0.4484          1.0329            1.0309         4.05
ExhaustiveBinary-512-pca (self)                          296.85     1_632.95     1_929.80       0.3515          1.0522            1.0505         4.05
ExhaustiveBinary-1024-random_no_rr (query)               260.14       547.74       807.88       0.1816          1.2043            1.1936         8.11
ExhaustiveBinary-1024-random-rf10 (query)                260.14       689.39       949.53       0.3747          1.0447            1.0452         8.11
ExhaustiveBinary-1024-random-rf20 (query)                260.14       826.50     1_086.65       0.4789          1.0282            1.0270         8.11
ExhaustiveBinary-1024-random (self)                      260.14     2_272.49     2_532.63       0.3754          1.0447            1.0451         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  351.05       562.42       913.47       0.1832          1.2013            1.1905         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   351.05       701.33     1_052.38       0.3798          1.0434            1.0443         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   351.05       842.76     1_193.81       0.4867          1.0272            1.0261         8.11
ExhaustiveBinary-1024-pca (self)                         351.05     2_282.36     2_633.41       0.3787          1.0436            1.0444         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   86.08       635.85       721.93       0.1518          1.2701            1.2528         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    86.08       685.69       771.77       0.3399          1.0607            1.0535         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    86.08     1_036.27     1_122.35       0.4406          1.0369            1.0319         3.05
ExhaustiveBinary-512-sign (self)                          86.08     2_209.76     2_295.84       0.3409          1.0595            1.0531         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)              342.99        74.30       417.29       0.1216          1.3043            1.2662         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)             342.99        78.82       421.80       0.1216          1.3043            1.2662         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)             342.99        80.05       423.04       0.1216          1.3043            1.2662         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)             342.99       153.95       496.93       0.3473          1.0644            1.0513         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)             342.99       253.91       596.90       0.4501          1.0393            1.0305         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)            342.99       151.58       494.57       0.3473          1.0644            1.0513         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)            342.99       243.03       586.01       0.4501          1.0393            1.0305         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)            342.99       151.78       494.77       0.3473          1.0644            1.0513         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)            342.99       241.70       584.69       0.4501          1.0393            1.0305         2.34
IVF-Binary-256-nl158-random (self)                       342.99       288.81       631.80       0.3488          1.0607            1.0504         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)             309.66        71.14       380.80       0.1327          1.2745            1.2362         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)             309.66        74.14       383.80       0.1327          1.2745            1.2362         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)             309.66        75.84       385.50       0.1327          1.2745            1.2362         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)            309.66       155.48       465.14       0.3705          1.0551            1.0451         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)            309.66       243.13       552.79       0.4727          1.0355            1.0273         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)            309.66       154.95       464.61       0.3705          1.0551            1.0451         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)            309.66       244.81       554.47       0.4727          1.0355            1.0273         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)            309.66       158.75       468.41       0.3705          1.0551            1.0451         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)            309.66       248.07       557.73       0.4727          1.0355            1.0273         2.46
IVF-Binary-256-nl223-random (self)                       309.66       309.10       618.76       0.3719          1.0512            1.0446         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)             363.73        76.83       440.55       0.1423          1.2543            1.2154         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)             363.73        78.62       442.35       0.1423          1.2543            1.2154         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)             363.73        80.59       444.31       0.1423          1.2543            1.2154         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)            363.73       161.65       525.38       0.3810          1.0511            1.0426         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)            363.73       248.97       612.70       0.4794          1.0335            1.0265         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)            363.73       159.00       522.72       0.3810          1.0511            1.0426         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)            363.73       250.21       613.93       0.4794          1.0335            1.0265         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)            363.73       161.72       525.44       0.3810          1.0511            1.0426         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)            363.73       250.33       614.05       0.4794          1.0335            1.0265         2.65
IVF-Binary-256-nl316-random (self)                       363.73       328.20       691.93       0.3830          1.0473            1.0420         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)                 404.17        64.73       468.91       0.1273          1.3033            1.2601         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)                404.17        67.29       471.46       0.1273          1.3033            1.2601         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)                404.17        70.20       474.38       0.1273          1.3033            1.2601         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)                404.17       149.34       553.51       0.3444          1.0627            1.0512         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)                404.17       238.13       642.31       0.4465          1.0386            1.0308         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)               404.17       148.85       553.03       0.3444          1.0627            1.0512         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)               404.17       239.57       643.74       0.4465          1.0386            1.0308         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)               404.17       151.03       555.20       0.3444          1.0627            1.0512         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)               404.17       241.06       645.24       0.4465          1.0386            1.0308         2.34
IVF-Binary-256-nl158-pca (self)                          404.17       290.97       695.14       0.3454          1.0616            1.0504         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)                474.96        71.70       546.66       0.1369          1.2750            1.2317         2.46
IVF-Binary-256-nl223-np14-rf0-pca (query)                474.96        72.18       547.15       0.1369          1.2750            1.2317         2.46
IVF-Binary-256-nl223-np21-rf0-pca (query)                474.96        75.09       550.05       0.1369          1.2750            1.2317         2.46
IVF-Binary-256-nl223-np11-rf10-pca (query)               474.96       156.23       631.19       0.3675          1.0532            1.0453         2.46
IVF-Binary-256-nl223-np11-rf20-pca (query)               474.96       251.69       726.65       0.4689          1.0345            1.0279         2.46
IVF-Binary-256-nl223-np14-rf10-pca (query)               474.96       154.58       629.55       0.3675          1.0532            1.0453         2.46
IVF-Binary-256-nl223-np14-rf20-pca (query)               474.96       244.43       719.39       0.4689          1.0345            1.0279         2.46
IVF-Binary-256-nl223-np21-rf10-pca (query)               474.96       156.98       631.94       0.3675          1.0532            1.0453         2.46
IVF-Binary-256-nl223-np21-rf20-pca (query)               474.96       247.49       722.45       0.4689          1.0345            1.0279         2.46
IVF-Binary-256-nl223-pca (self)                          474.96       309.50       784.46       0.3695          1.0523            1.0446         2.46
IVF-Binary-256-nl316-np15-rf0-pca (query)                454.10        80.35       534.45       0.1441          1.2600            1.2139         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)                454.10        80.58       534.68       0.1441          1.2600            1.2139         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)                454.10        84.41       538.51       0.1441          1.2600            1.2139         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)               454.10       161.95       616.04       0.3777          1.0505            1.0427         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)               454.10       252.27       706.37       0.4765          1.0335            1.0268         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)               454.10       160.43       614.53       0.3777          1.0505            1.0427         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)               454.10       254.78       708.88       0.4765          1.0335            1.0268         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)               454.10       161.40       615.50       0.3777          1.0505            1.0427         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)               454.10       259.38       713.48       0.4765          1.0335            1.0268         2.65
IVF-Binary-256-nl316-pca (self)                          454.10       329.83       783.93       0.3797          1.0486            1.0422         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)              377.26        99.01       476.27       0.1587          1.2371            1.2109         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)             377.26       103.43       480.70       0.1587          1.2371            1.2109         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)             377.26       101.95       479.22       0.1587          1.2371            1.2109         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)             377.26       181.09       558.35       0.3612          1.0504            1.0471         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)             377.26       269.12       646.38       0.4601          1.0326            1.0289         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)            377.26       182.13       559.39       0.3612          1.0504            1.0471         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)            377.26       274.49       651.75       0.4601          1.0326            1.0289         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)            377.26       185.59       562.85       0.3612          1.0504            1.0471         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)            377.26       275.90       653.17       0.4601          1.0326            1.0289         4.36
IVF-Binary-512-nl158-random (self)                       377.26       410.93       788.20       0.3617          1.0492            1.0470         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)             385.12        97.60       482.71       0.1639          1.2226            1.1963         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)             385.12        98.65       483.76       0.1639          1.2226            1.1963         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)             385.12       104.82       489.94       0.1639          1.2226            1.1963         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)            385.12       184.21       569.33       0.3712          1.0477            1.0447         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)            385.12       278.02       663.13       0.4708          1.0311            1.0276         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)            385.12       184.37       569.49       0.3712          1.0477            1.0447         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)            385.12       275.60       660.72       0.4708          1.0311            1.0276         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)            385.12       187.95       573.07       0.3712          1.0477            1.0447         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)            385.12       278.54       663.66       0.4708          1.0311            1.0276         4.49
IVF-Binary-512-nl223-random (self)                       385.12       424.17       809.29       0.3718          1.0466            1.0447         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)             437.14       103.89       541.02       0.1670          1.2153            1.1895         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)             437.14       104.99       542.13       0.1670          1.2153            1.1895         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)             437.14       107.55       544.69       0.1670          1.2153            1.1895         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)            437.14       196.97       634.10       0.3745          1.0468            1.0440         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)            437.14       279.39       716.53       0.4729          1.0307            1.0274         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)            437.14       196.83       633.96       0.3745          1.0468            1.0440         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)            437.14       287.86       724.99       0.4729          1.0307            1.0274         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)            437.14       192.77       629.90       0.3745          1.0468            1.0440         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)            437.14       287.91       725.04       0.4729          1.0307            1.0274         4.67
IVF-Binary-512-nl316-random (self)                       437.14       448.49       885.62       0.3752          1.0456            1.0439         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)                 541.40        91.01       632.40       0.1624          1.2302            1.2051         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)                541.40        94.93       636.32       0.1624          1.2302            1.2051         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)                541.40        97.45       638.84       0.1624          1.2302            1.2051         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)                541.40       177.83       719.23       0.3643          1.0476            1.0470         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)                541.40       267.95       809.35       0.4641          1.0304            1.0285         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)               541.40       179.92       721.32       0.3643          1.0476            1.0470         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)               541.40       276.58       817.97       0.4641          1.0304            1.0285         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)               541.40       185.84       727.24       0.3643          1.0476            1.0470         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)               541.40       283.78       825.18       0.4641          1.0304            1.0285         4.36
IVF-Binary-512-nl158-pca (self)                          541.40       410.89       952.28       0.3644          1.0476            1.0468         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)                545.50        97.84       643.34       0.1675          1.2167            1.1920         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)                545.50        99.49       644.99       0.1675          1.2167            1.1920         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)                545.50       104.79       650.28       0.1675          1.2167            1.1920         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)               545.50       183.81       729.31       0.3732          1.0455            1.0447         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)               545.50       274.72       820.22       0.4741          1.0292            1.0272         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)               545.50       184.37       729.86       0.3732          1.0455            1.0447         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)               545.50       275.29       820.78       0.4741          1.0292            1.0272         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)               545.50       187.79       733.29       0.3732          1.0455            1.0447         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)               545.50       280.77       826.26       0.4741          1.0292            1.0272         4.49
IVF-Binary-512-nl223-pca (self)                          545.50       424.02       969.52       0.3737          1.0457            1.0445         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)                536.27       103.28       639.55       0.1704          1.2102            1.1853         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)                536.27       104.98       641.25       0.1704          1.2102            1.1853         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)                536.27       107.68       643.95       0.1704          1.2102            1.1853         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)               536.27       190.12       726.39       0.3768          1.0447            1.0438         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)               536.27       278.12       814.39       0.4766          1.0288            1.0269         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)               536.27       189.09       725.36       0.3768          1.0447            1.0438         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)               536.27       282.53       818.80       0.4766          1.0288            1.0269         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)               536.27       204.10       740.37       0.3768          1.0447            1.0438         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)               536.27       284.88       821.15       0.4766          1.0288            1.0269         4.67
IVF-Binary-512-nl316-pca (self)                          536.27       449.65       985.92       0.3771          1.0448            1.0438         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)             426.88       141.87       568.75       0.1841          1.1946            1.1854         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)            426.88       146.64       573.51       0.1841          1.1946            1.1854         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)            426.88       148.49       575.37       0.1841          1.1946            1.1854         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)            426.88       232.80       659.67       0.3818          1.0429            1.0433         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)            426.88       325.46       752.33       0.4875          1.0271            1.0259         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)           426.88       235.54       662.42       0.3818          1.0429            1.0433         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)           426.88       333.93       760.81       0.4875          1.0271            1.0259         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)           426.88       244.44       671.31       0.3818          1.0429            1.0433         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)           426.88       338.93       765.81       0.4875          1.0271            1.0259         8.42
IVF-Binary-1024-nl158-random (self)                      426.88       601.35     1_028.23       0.3826          1.0428            1.0432         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)            437.93       146.56       584.49       0.1863          1.1878            1.1788         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)            437.93       148.25       586.17       0.1863          1.1878            1.1788         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)            437.93       156.46       594.39       0.1863          1.1878            1.1788         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)           437.93       239.16       677.08       0.3872          1.0418            1.0421         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)           437.93       336.07       774.00       0.4937          1.0264            1.0251         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)           437.93       241.25       679.17       0.3872          1.0418            1.0421         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)           437.93       341.63       779.55       0.4937          1.0264            1.0251         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)           437.93       246.11       684.04       0.3872          1.0418            1.0421         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)           437.93       371.94       809.87       0.4937          1.0264            1.0251         8.54
IVF-Binary-1024-nl223-random (self)                      437.93       638.65     1_076.58       0.3883          1.0417            1.0420         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)            485.11       153.30       638.41       0.1869          1.1852            1.1759         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)            485.11       154.16       639.27       0.1869          1.1852            1.1759         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)            485.11       158.24       643.35       0.1869          1.1852            1.1759         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)           485.11       245.39       730.50       0.3887          1.0414            1.0418         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)           485.11       343.66       828.77       0.4942          1.0263            1.0248         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)           485.11       245.24       730.35       0.3887          1.0414            1.0418         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)           485.11       349.45       834.55       0.4942          1.0263            1.0248         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)           485.11       261.06       746.17       0.3887          1.0414            1.0418         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)           485.11       355.44       840.54       0.4942          1.0263            1.0248         8.72
IVF-Binary-1024-nl316-random (self)                      485.11       639.62     1_124.73       0.3898          1.0413            1.0417         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)                526.18       141.29       667.47       0.1857          1.1919            1.1825         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)               526.18       145.14       671.32       0.1857          1.1919            1.1825         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)               526.18       147.91       674.09       0.1857          1.1919            1.1825         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)               526.18       230.80       756.98       0.3866          1.0417            1.0426         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)               526.18       328.27       854.45       0.4948          1.0261            1.0251         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)              526.18       233.46       759.65       0.3866          1.0417            1.0426         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)              526.18       336.47       862.65       0.4948          1.0261            1.0251         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)              526.18       238.56       764.74       0.3866          1.0417            1.0426         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)              526.18       338.98       865.17       0.4948          1.0261            1.0251         8.42
IVF-Binary-1024-nl158-pca (self)                         526.18       602.86     1_129.04       0.3860          1.0418            1.0426         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)               534.25       146.89       681.14       0.1873          1.1862            1.1771         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)               534.25       149.03       683.28       0.1873          1.1862            1.1771         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)               534.25       153.01       687.26       0.1873          1.1862            1.1771         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)              534.25       236.95       771.20       0.3923          1.0406            1.0413         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)              534.25       335.16       869.41       0.5008          1.0254            1.0245         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)              534.25       245.86       780.11       0.3923          1.0406            1.0413         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)              534.25       337.32       871.57       0.5008          1.0254            1.0245         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)              534.25       257.64       791.89       0.3923          1.0406            1.0413         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)              534.25       350.54       884.79       0.5008          1.0254            1.0245         8.54
IVF-Binary-1024-nl223-pca (self)                         534.25       616.30     1_150.55       0.3913          1.0407            1.0414         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)               587.93       153.45       741.39       0.1884          1.1834            1.1743         8.72
IVF-Binary-1024-nl316-np17-rf0-pca (query)               587.93       153.48       741.41       0.1884          1.1834            1.1743         8.72
IVF-Binary-1024-nl316-np25-rf0-pca (query)               587.93       158.17       746.10       0.1884          1.1834            1.1743         8.72
IVF-Binary-1024-nl316-np15-rf10-pca (query)              587.93       244.49       832.42       0.3933          1.0404            1.0410         8.72
IVF-Binary-1024-nl316-np15-rf20-pca (query)              587.93       343.37       931.30       0.5013          1.0254            1.0244         8.72
IVF-Binary-1024-nl316-np17-rf10-pca (query)              587.93       245.06       832.99       0.3933          1.0404            1.0410         8.72
IVF-Binary-1024-nl316-np17-rf20-pca (query)              587.93       346.28       934.21       0.5013          1.0254            1.0244         8.72
IVF-Binary-1024-nl316-np25-rf10-pca (query)              587.93       249.23       837.17       0.3933          1.0404            1.0410         8.72
IVF-Binary-1024-nl316-np25-rf20-pca (query)              587.93       350.53       938.47       0.5013          1.0254            1.0244         8.72
IVF-Binary-1024-nl316-pca (self)                         587.93       637.55     1_225.48       0.3927          1.0405            1.0411         8.72
IVF-Binary-512-nl158-np7-rf0-sign (query)                258.84       250.67       509.51       0.1523          1.2668            1.2496         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)               258.84       254.48       513.32       0.1523          1.2668            1.2496         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)               258.84       257.14       515.99       0.1523          1.2668            1.2496         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)               258.84       313.71       572.55       0.3469          1.0572            1.0513         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)               258.84       556.20       815.04       0.4448          1.0360            1.0311         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)              258.84       316.58       575.43       0.3469          1.0572            1.0513         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)              258.84       563.36       822.20       0.4448          1.0360            1.0311         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)              258.84       320.80       579.64       0.3469          1.0572            1.0513         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)              258.84       565.45       824.29       0.4448          1.0360            1.0311         3.36
IVF-Binary-512-nl158-sign (self)                         258.84       864.77     1_123.61       0.3477          1.0561            1.0509         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               267.11       255.06       522.18       0.1523          1.2671            1.2491         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               267.11       256.95       524.07       0.1523          1.2671            1.2491         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               267.11       262.64       529.76       0.1523          1.2671            1.2491         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              267.11       321.28       588.39       0.3518          1.0555            1.0499         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              267.11       562.26       829.38       0.4484          1.0349            1.0307         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              267.11       321.80       588.91       0.3518          1.0555            1.0499         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              267.11       565.40       832.51       0.4484          1.0349            1.0307         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              267.11       326.69       593.80       0.3518          1.0555            1.0499         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              267.11       575.89       843.00       0.4484          1.0349            1.0307         3.49
IVF-Binary-512-nl223-sign (self)                         267.11       877.27     1_144.38       0.3526          1.0544            1.0494         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)               328.24       260.53       588.77       0.1527          1.2740            1.2488         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)               328.24       262.15       590.38       0.1527          1.2740            1.2488         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)               328.24       266.65       594.89       0.1527          1.2740            1.2488         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)              328.24       330.56       658.80       0.3528          1.0553            1.0495         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)              328.24       568.88       897.12       0.4490          1.0348            1.0306         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)              328.24       326.94       655.18       0.3528          1.0553            1.0495         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)              328.24       572.16       900.40       0.4490          1.0348            1.0306         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)              328.24       331.88       660.11       0.3528          1.0553            1.0495         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)              328.24       575.91       904.15       0.4490          1.0348            1.0306         3.67
IVF-Binary-512-nl316-sign (self)                         328.24       889.78     1_218.02       0.3536          1.0540            1.0493         3.67
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 768D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       103.91     1_840.25     1_944.16       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        103.91     6_276.62     6_380.54       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                192.25       283.89       476.14       0.1140          1.2809            1.2433         2.28
ExhaustiveBinary-256-random-rf10 (query)                 192.25       415.73       607.98       0.3148          1.0656            1.0476         2.28
ExhaustiveBinary-256-random-rf20 (query)                 192.25       560.85       753.10       0.4075          1.0420            1.0293         2.28
ExhaustiveBinary-256-random (self)                       192.25     1_288.23     1_480.48       0.3168          1.0618            1.0471         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   394.79       278.49       673.28       0.1054          1.3026            1.2617         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    394.79       410.97       805.77       0.3012          1.0735            1.0517         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    394.79       552.16       946.95       0.3931          1.0471            1.0315         2.28
ExhaustiveBinary-256-pca (self)                          394.79     1_267.38     1_662.17       0.3048          1.0710            1.0504         2.28
ExhaustiveBinary-512-random_no_rr (query)                297.21       395.90       693.11       0.1506          1.2094            1.1809         4.55
ExhaustiveBinary-512-random-rf10 (query)                 297.21       543.81       841.03       0.3395          1.0453            1.0426         4.55
ExhaustiveBinary-512-random-rf20 (query)                 297.21       694.58       991.79       0.4326          1.0293            1.0264         4.55
ExhaustiveBinary-512-random (self)                       297.21     1_747.23     2_044.44       0.3401          1.0435            1.0423         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   502.19       395.17       897.36       0.1459          1.2162            1.1914         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    502.19       545.18     1_047.36       0.3341          1.0468            1.0435         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    502.19       700.23     1_202.42       0.4278          1.0295            1.0269         4.55
ExhaustiveBinary-512-pca (self)                          502.19     1_730.98     2_233.17       0.3355          1.0454            1.0433         4.55
ExhaustiveBinary-1024-random_no_rr (query)               500.82       599.74     1_100.55       0.1761          1.1673            1.1571         9.11
ExhaustiveBinary-1024-random-rf10 (query)                500.82       769.85     1_270.67       0.3603          1.0383            1.0383         9.11
ExhaustiveBinary-1024-random-rf20 (query)                500.82       937.05     1_437.87       0.4618          1.0244            1.0230         9.11
ExhaustiveBinary-1024-random (self)                      500.82     2_508.70     3_009.51       0.3602          1.0377            1.0383         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  708.55       608.60     1_317.15       0.1756          1.1686            1.1586         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   708.55       774.54     1_483.08       0.3594          1.0382            1.0385         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   708.55       945.54     1_654.08       0.4576          1.0246            1.0237         9.11
ExhaustiveBinary-1024-pca (self)                         708.55     2_512.22     3_220.77       0.3584          1.0383            1.0389         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  127.75       807.09       934.84       0.1691          1.1871            1.1718         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   127.75       877.69     1_005.44       0.3431          1.0433            1.0415         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   127.75     1_367.20     1_494.95       0.4437          1.0266            1.0250         4.58
ExhaustiveBinary-768-sign (self)                         127.75     2_835.33     2_963.07       0.3438          1.0424            1.0413         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)              433.37        92.87       526.23       0.1235          1.2513            1.2181         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)             433.37        99.37       532.74       0.1235          1.2513            1.2181         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)             433.37        98.03       531.40       0.1235          1.2513            1.2181         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)             433.37       193.86       627.22       0.3340          1.0555            1.0426         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)             433.37       300.50       733.86       0.4312          1.0357            1.0261         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)            433.37       188.87       622.24       0.3340          1.0555            1.0426         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)            433.37       304.63       738.00       0.4312          1.0357            1.0261         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)            433.37       191.00       624.36       0.3340          1.0555            1.0426         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)            433.37       305.42       738.78       0.4312          1.0357            1.0261         2.74
IVF-Binary-256-nl158-random (self)                       433.37       398.50       831.87       0.3355          1.0521            1.0424         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)             452.95        91.73       544.67       0.1302          1.2329            1.1991         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)             452.95        93.49       546.43       0.1302          1.2329            1.1991         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)             452.95        97.64       550.58       0.1302          1.2329            1.1991         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)            452.95       200.27       653.22       0.3508          1.0493            1.0391         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)            452.95       312.27       765.22       0.4513          1.0316            1.0237         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)            452.95       199.81       652.76       0.3508          1.0493            1.0391         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)            452.95       312.89       765.84       0.4513          1.0316            1.0237         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)            452.95       198.63       651.58       0.3508          1.0493            1.0391         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)            452.95       314.26       767.20       0.4513          1.0316            1.0237         2.93
IVF-Binary-256-nl223-random (self)                       452.95       425.68       878.63       0.3526          1.0462            1.0387         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)             515.44       100.37       615.82       0.1390          1.2157            1.1768         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)             515.44       102.33       617.78       0.1390          1.2157            1.1768         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)             515.44       103.37       618.82       0.1390          1.2157            1.1768         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)            515.44       205.80       721.24       0.3634          1.0448            1.0364         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)            515.44       319.21       834.65       0.4635          1.0295            1.0225         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)            515.44       205.36       720.80       0.3634          1.0448            1.0364         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)            515.44       321.17       836.62       0.4635          1.0295            1.0225         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)            515.44       209.63       725.08       0.3634          1.0448            1.0364         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)            515.44       325.16       840.61       0.4635          1.0295            1.0225         3.21
IVF-Binary-256-nl316-random (self)                       515.44       451.61       967.05       0.3652          1.0415            1.0361         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)                 638.33        83.91       722.25       0.1159          1.2675            1.2324         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)                638.33        86.25       724.58       0.1159          1.2675            1.2324         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)                638.33        87.64       725.97       0.1159          1.2675            1.2324         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)                638.33       186.61       824.94       0.3257          1.0609            1.0447         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)                638.33       297.72       936.05       0.4226          1.0385            1.0273         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)               638.33       185.82       824.15       0.3257          1.0609            1.0447         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)               638.33       300.86       939.19       0.4226          1.0385            1.0273         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)               638.33       187.62       825.95       0.3257          1.0609            1.0447         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)               638.33       302.30       940.64       0.4226          1.0385            1.0273         2.74
IVF-Binary-256-nl158-pca (self)                          638.33       378.24     1_016.57       0.3286          1.0584            1.0439         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)                650.19        91.06       741.25       0.1236          1.2441            1.2112         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)                650.19        92.71       742.89       0.1236          1.2441            1.2112         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)                650.19        94.36       744.55       0.1236          1.2441            1.2112         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)               650.19       195.60       845.79       0.3467          1.0533            1.0400         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)               650.19       307.93       958.12       0.4492          1.0332            1.0241         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)               650.19       195.80       845.99       0.3467          1.0533            1.0400         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)               650.19       308.99       959.18       0.4492          1.0332            1.0241         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)               650.19       197.14       847.33       0.3467          1.0533            1.0400         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)               650.19       314.02       964.21       0.4492          1.0332            1.0241         2.93
IVF-Binary-256-nl223-pca (self)                          650.19       399.79     1_049.98       0.3497          1.0507            1.0393         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)                717.46       100.16       817.61       0.1340          1.2222            1.1895         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)                717.46       101.03       818.49       0.1340          1.2222            1.1895         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)                717.46       103.37       820.83       0.1340          1.2222            1.1895         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)               717.46       205.33       922.78       0.3691          1.0452            1.0357         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)               717.46       319.18     1_036.64       0.4697          1.0284            1.0220         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)               717.46       203.24       920.70       0.3691          1.0452            1.0357         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)               717.46       318.40     1_035.86       0.4697          1.0284            1.0220         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)               717.46       206.69       924.15       0.3691          1.0452            1.0357         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)               717.46       323.16     1_040.62       0.4697          1.0284            1.0220         3.21
IVF-Binary-256-nl316-pca (self)                          717.46       459.44     1_176.90       0.3727          1.0425            1.0352         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)              545.35       118.69       664.04       0.1558          1.1944            1.1685         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)             545.35       122.85       668.20       0.1558          1.1944            1.1685         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)             545.35       132.56       677.91       0.1558          1.1944            1.1685         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)             545.35       229.00       774.35       0.3501          1.0418            1.0399         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)             545.35       341.98       887.34       0.4461          1.0269            1.0247         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)            545.35       229.21       774.56       0.3501          1.0418            1.0399         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)            545.35       345.57       890.92       0.4461          1.0269            1.0247         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)            545.35       232.19       777.54       0.3501          1.0418            1.0399         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)            545.35       352.83       898.19       0.4461          1.0269            1.0247         5.02
IVF-Binary-512-nl158-random (self)                       545.35       554.18     1_099.54       0.3507          1.0403            1.0398         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)             562.11       129.10       691.20       0.1594          1.1855            1.1597         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)             562.11       128.82       690.92       0.1594          1.1855            1.1597         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)             562.11       132.14       694.24       0.1594          1.1855            1.1597         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)            562.11       237.19       799.29       0.3564          1.0404            1.0382         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)            562.11       351.48       913.58       0.4533          1.0263            1.0239         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)            562.11       236.15       798.26       0.3564          1.0404            1.0382         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)            562.11       354.10       916.21       0.4533          1.0263            1.0239         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)            562.11       241.22       803.32       0.3564          1.0404            1.0382         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)            562.11       358.78       920.89       0.4533          1.0263            1.0239         5.21
IVF-Binary-512-nl223-random (self)                       562.11       570.96     1_133.07       0.3569          1.0390            1.0383         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)             630.01       136.79       766.80       0.1628          1.1775            1.1525         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)             630.01       137.25       767.26       0.1628          1.1775            1.1525         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)             630.01       140.08       770.09       0.1628          1.1775            1.1525         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)            630.01       246.60       876.61       0.3616          1.0395            1.0373         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)            630.01       365.41       995.42       0.4582          1.0259            1.0234         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)            630.01       249.83       879.84       0.3616          1.0395            1.0373         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)            630.01       365.19       995.20       0.4582          1.0259            1.0234         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)            630.01       252.85       882.86       0.3616          1.0395            1.0373         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)            630.01       368.81       998.82       0.4582          1.0259            1.0234         5.48
IVF-Binary-512-nl316-random (self)                       630.01       599.84     1_229.85       0.3618          1.0380            1.0373         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)                 733.93       118.04       851.97       0.1515          1.1995            1.1746         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)                733.93       121.96       855.89       0.1515          1.1995            1.1746         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)                733.93       125.17       859.10       0.1515          1.1995            1.1746         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)                733.93       227.28       961.20       0.3460          1.0430            1.0406         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)                733.93       339.20     1_073.13       0.4423          1.0274            1.0251         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)               733.93       228.36       962.29       0.3460          1.0430            1.0406         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)               733.93       343.57     1_077.49       0.4423          1.0274            1.0251         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)               733.93       230.65       964.57       0.3460          1.0430            1.0406         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)               733.93       349.77     1_083.70       0.4423          1.0274            1.0251         5.02
IVF-Binary-512-nl158-pca (self)                          733.93       544.12     1_278.05       0.3471          1.0420            1.0405         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)                749.85       126.50       876.36       0.1561          1.1880            1.1643         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)                749.85       128.24       878.09       0.1561          1.1880            1.1643         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)                749.85       132.56       882.41       0.1561          1.1880            1.1643         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)               749.85       235.56       985.42       0.3559          1.0404            1.0385         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)               749.85       352.45     1_102.30       0.4523          1.0263            1.0239         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)               749.85       236.33       986.19       0.3559          1.0404            1.0385         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)               749.85       353.63     1_103.48       0.4523          1.0263            1.0239         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)               749.85       238.68       988.53       0.3559          1.0404            1.0385         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)               749.85       358.23     1_108.09       0.4523          1.0263            1.0239         5.21
IVF-Binary-512-nl223-pca (self)                          749.85       563.65     1_313.50       0.3565          1.0396            1.0384         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)                820.17       136.22       956.39       0.1609          1.1777            1.1539         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)                820.17       136.73       956.90       0.1609          1.1777            1.1539         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)                820.17       140.81       960.98       0.1609          1.1777            1.1539         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)               820.17       247.23     1_067.40       0.3620          1.0392            1.0373         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)               820.17       364.98     1_185.15       0.4573          1.0258            1.0233         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)               820.17       245.84     1_066.01       0.3620          1.0392            1.0373         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)               820.17       360.89     1_181.06       0.4573          1.0258            1.0233         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)               820.17       247.46     1_067.63       0.3620          1.0392            1.0373         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)               820.17       366.42     1_186.59       0.4573          1.0258            1.0233         5.48
IVF-Binary-512-nl316-pca (self)                          820.17       601.59     1_421.76       0.3629          1.0383            1.0371         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)             742.28       189.03       931.31       0.1780          1.1612            1.1521         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)            742.28       192.11       934.39       0.1780          1.1612            1.1521         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)            742.28       201.60       943.88       0.1780          1.1612            1.1521         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)            742.28       321.14     1_063.43       0.3663          1.0369            1.0371         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)            742.28       433.71     1_176.00       0.4685          1.0236            1.0224         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)           742.28       314.02     1_056.31       0.3663          1.0369            1.0371         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)           742.28       441.67     1_183.95       0.4685          1.0236            1.0224         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)           742.28       325.81     1_068.10       0.3663          1.0369            1.0371         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)           742.28       452.82     1_195.11       0.4685          1.0236            1.0224         9.57
IVF-Binary-1024-nl158-random (self)                      742.28       826.34     1_568.63       0.3662          1.0364            1.0370         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)            759.09       199.66       958.75       0.1792          1.1574            1.1483         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)            759.09       205.32       964.42       0.1792          1.1574            1.1483         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)            759.09       204.97       964.06       0.1792          1.1574            1.1483         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)           759.09       317.05     1_076.14       0.3701          1.0362            1.0363         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)           759.09       445.17     1_204.27       0.4729          1.0231            1.0219         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)           759.09       323.44     1_082.54       0.3701          1.0362            1.0363         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)           759.09       451.03     1_210.13       0.4729          1.0231            1.0219         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)           759.09       331.98     1_091.07       0.3701          1.0362            1.0363         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)           759.09       461.36     1_220.46       0.4729          1.0231            1.0219         9.76
IVF-Binary-1024-nl223-random (self)                      759.09       851.04     1_610.14       0.3699          1.0357            1.0362         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)            832.76       211.48     1_044.24       0.1805          1.1543            1.1451        10.03
IVF-Binary-1024-nl316-np17-rf0-random (query)            832.76       209.79     1_042.55       0.1805          1.1543            1.1451        10.03
IVF-Binary-1024-nl316-np25-rf0-random (query)            832.76       230.03     1_062.79       0.1805          1.1543            1.1451        10.03
IVF-Binary-1024-nl316-np15-rf10-random (query)           832.76       326.00     1_158.76       0.3732          1.0357            1.0359        10.03
IVF-Binary-1024-nl316-np15-rf20-random (query)           832.76       460.93     1_293.69       0.4751          1.0229            1.0217        10.03
IVF-Binary-1024-nl316-np17-rf10-random (query)           832.76       326.70     1_159.46       0.3732          1.0357            1.0359        10.03
IVF-Binary-1024-nl316-np17-rf20-random (query)           832.76       465.43     1_298.19       0.4751          1.0229            1.0217        10.03
IVF-Binary-1024-nl316-np25-rf10-random (query)           832.76       340.32     1_173.07       0.3732          1.0357            1.0359        10.03
IVF-Binary-1024-nl316-np25-rf20-random (query)           832.76       473.07     1_305.83       0.4751          1.0229            1.0217        10.03
IVF-Binary-1024-nl316-random (self)                      832.76       907.06     1_739.82       0.3729          1.0352            1.0358        10.03
IVF-Binary-1024-nl158-np7-rf0-pca (query)                956.92       196.64     1_153.57       0.1777          1.1619            1.1522         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)               956.92       197.24     1_154.17       0.1777          1.1619            1.1522         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)               956.92       201.01     1_157.93       0.1777          1.1619            1.1522         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)               956.92       308.29     1_265.21       0.3660          1.0369            1.0372         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)               956.92       429.94     1_386.86       0.4656          1.0237            1.0228         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)              956.92       311.30     1_268.22       0.3660          1.0369            1.0372         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)              956.92       439.94     1_396.86       0.4656          1.0237            1.0228         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)              956.92       318.28     1_275.20       0.3660          1.0369            1.0372         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)              956.92       446.72     1_403.64       0.4656          1.0237            1.0228         9.57
IVF-Binary-1024-nl158-pca (self)                         956.92       825.33     1_782.25       0.3649          1.0369            1.0376         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)               963.99       199.58     1_163.57       0.1789          1.1577            1.1482         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)               963.99       199.88     1_163.88       0.1789          1.1577            1.1482         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)               963.99       205.53     1_169.52       0.1789          1.1577            1.1482         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)              963.99       315.25     1_279.25       0.3696          1.0361            1.0363         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)              963.99       447.46     1_411.46       0.4707          1.0233            1.0223         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)              963.99       324.35     1_288.35       0.3696          1.0361            1.0363         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)              963.99       449.92     1_413.91       0.4707          1.0233            1.0223         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)              963.99       328.81     1_292.81       0.3696          1.0361            1.0363         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)              963.99       456.00     1_420.00       0.4707          1.0233            1.0223         9.76
IVF-Binary-1024-nl223-pca (self)                         963.99       851.85     1_815.84       0.3689          1.0362            1.0366         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_029.79       211.91     1_241.71       0.1807          1.1540            1.1446        10.03
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_029.79       210.63     1_240.43       0.1807          1.1540            1.1446        10.03
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_029.79       213.25     1_243.05       0.1807          1.1540            1.1446        10.03
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_029.79       326.91     1_356.71       0.3726          1.0357            1.0359        10.03
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_029.79       457.69     1_487.49       0.4729          1.0230            1.0220        10.03
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_029.79       328.37     1_358.16       0.3726          1.0357            1.0359        10.03
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_029.79       462.27     1_492.06       0.4729          1.0230            1.0220        10.03
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_029.79       348.40     1_378.20       0.3726          1.0357            1.0359        10.03
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_029.79       469.09     1_498.88       0.4729          1.0230            1.0220        10.03
IVF-Binary-1024-nl316-pca (self)                       1_029.79       880.06     1_909.85       0.3717          1.0357            1.0362        10.03
IVF-Binary-768-nl158-np7-rf0-sign (query)                376.19       375.83       752.03       0.1693          1.1871            1.1718         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)               376.19       380.43       756.62       0.1693          1.1871            1.1718         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)               376.19       381.25       757.44       0.1693          1.1871            1.1718         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)               376.19       452.39       828.58       0.3467          1.0421            1.0406         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)               376.19       816.80     1_192.99       0.4466          1.0261            1.0247         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)              376.19       455.19       831.39       0.3467          1.0421            1.0406         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)              376.19       819.68     1_195.88       0.4466          1.0261            1.0247         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)              376.19       485.73       861.92       0.3467          1.0421            1.0406         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)              376.19       826.18     1_202.37       0.4466          1.0261            1.0247         5.04
IVF-Binary-768-nl158-sign (self)                         376.19     1_285.97     1_662.16       0.3472          1.0414            1.0406         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)               391.60       379.30       770.90       0.1694          1.1862            1.1716         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)               391.60       384.99       776.59       0.1694          1.1862            1.1716         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)               391.60       395.42       787.02       0.1694          1.1862            1.1716         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)              391.60       459.24       850.84       0.3484          1.0415            1.0401         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)              391.60       824.61     1_216.20       0.4478          1.0258            1.0245         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)              391.60       460.27       851.87       0.3484          1.0415            1.0401         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)              391.60       827.04     1_218.63       0.4478          1.0258            1.0245         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)              391.60       464.42       856.02       0.3484          1.0415            1.0401         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)              391.60       834.54     1_226.14       0.4478          1.0258            1.0245         5.23
IVF-Binary-768-nl223-sign (self)                         391.60     1_297.84     1_689.43       0.3488          1.0409            1.0401         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)               461.40       385.99       847.39       0.1693          1.1868            1.1718         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)               461.40       388.25       849.65       0.1693          1.1868            1.1718         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)               461.40       392.19       853.59       0.1693          1.1868            1.1718         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)              461.40       483.54       944.94       0.3497          1.0413            1.0399         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)              461.40       832.96     1_294.36       0.4484          1.0258            1.0245         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)              461.40       471.50       932.90       0.3497          1.0413            1.0399         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)              461.40       839.54     1_300.94       0.4484          1.0258            1.0245         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)              461.40       475.11       936.51       0.3497          1.0413            1.0399         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)              461.40       843.18     1_304.58       0.4484          1.0258            1.0245         5.51
IVF-Binary-768-nl316-sign (self)                         461.40     1_330.39     1_791.80       0.3502          1.0406            1.0398         5.51
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 256D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.85       677.43       710.29       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.85     2_305.86     2_338.71       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                 72.43       240.31       312.74       0.0970          1.6334            1.6378         1.78
ExhaustiveBinary-256-random-rf10 (query)                  72.43       342.69       415.12       0.3643          1.1391            1.1302         1.78
ExhaustiveBinary-256-random-rf20 (query)                  72.43       443.04       515.47       0.5087          1.0798            1.0701         1.78
ExhaustiveBinary-256-random (self)                        72.43     1_098.11     1_170.54       0.3862          1.1443            1.1409         1.78
ExhaustiveBinary-256-pca_no_rr (query)                    99.07       237.55       336.62       0.0922          1.6524            1.6606         1.78
ExhaustiveBinary-256-pca-rf10 (query)                     99.07       339.93       439.00       0.3517          1.1465            1.1384         1.78
ExhaustiveBinary-256-pca-rf20 (query)                     99.07       440.52       539.59       0.4943          1.0846            1.0744         1.78
ExhaustiveBinary-256-pca (self)                           99.07     1_098.35     1_197.42       0.3764          1.1502            1.1477         1.78
ExhaustiveBinary-512-random_no_rr (query)                 82.79       344.94       427.73       0.1464          1.5035            1.5085         3.55
ExhaustiveBinary-512-random-rf10 (query)                  82.79       453.56       536.35       0.4596          1.0936            1.0901         3.55
ExhaustiveBinary-512-random-rf20 (query)                  82.79       558.70       641.49       0.6085          1.0504            1.0459         3.55
ExhaustiveBinary-512-random (self)                        82.79     1_484.66     1_567.46       0.4800          1.0996            1.0995         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   108.35       342.66       451.00       0.1458          1.5041            1.5097         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    108.35       454.74       563.09       0.4543          1.0952            1.0911         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    108.35       562.31       670.65       0.6037          1.0513            1.0464         3.55
ExhaustiveBinary-512-pca (self)                          108.35     1_482.24     1_590.59       0.4777          1.1003            1.0997         3.55
ExhaustiveBinary-1024-random_no_rr (query)               123.64       493.26       616.90       0.2155          1.3655            1.3721         7.10
ExhaustiveBinary-1024-random-rf10 (query)                123.64       610.63       734.27       0.5869          1.0540            1.0515         7.10
ExhaustiveBinary-1024-random-rf20 (query)                123.64       717.30       840.93       0.7380          1.0260            1.0224         7.10
ExhaustiveBinary-1024-random (self)                      123.64     2_043.85     2_167.48       0.6118          1.0576            1.0546         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  170.07       495.42       665.48       0.2122          1.3735            1.3798         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   170.07       609.20       779.26       0.5776          1.0560            1.0532         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   170.07       711.84       881.90       0.7291          1.0273            1.0232         7.10
ExhaustiveBinary-1024-pca (self)                         170.07     2_007.33     2_177.39       0.6017          1.0602            1.0571         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   40.99       419.27       460.26       0.1044          1.6421            1.6499         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    40.99       483.78       524.77       0.3737          1.1368            1.1275         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    40.99       681.95       722.95       0.5265          1.0745            1.0646         1.53
ExhaustiveBinary-256-sign (self)                          40.99     1_464.09     1_505.08       0.3940          1.1439            1.1394         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)              177.15        50.30       227.45       0.1065          1.6047            1.6089         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)             177.15        53.82       230.96       0.1064          1.6050            1.6091         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)             177.15        60.10       237.25       0.1064          1.6050            1.6091         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)             177.15        98.44       275.59       0.3857          1.1277            1.1208         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)             177.15       147.85       324.99       0.5288          1.0727            1.0650         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)            177.15        98.83       275.98       0.3852          1.1278            1.1209         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)            177.15       149.91       327.06       0.5282          1.0727            1.0650         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)            177.15       103.61       280.76       0.3852          1.1278            1.1209         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)            177.15       154.79       331.93       0.5282          1.0727            1.0650         1.93
IVF-Binary-256-nl158-random (self)                       177.15       232.38       409.53       0.4062          1.1325            1.1316         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             181.16        46.13       227.29       0.1131          1.5832            1.5861         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             181.16        46.82       227.98       0.1131          1.5833            1.5862         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             181.16        50.32       231.48       0.1131          1.5833            1.5862         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            181.16        99.84       281.00       0.3976          1.1211            1.1139         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            181.16       149.10       330.26       0.5393          1.0695            1.0619         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            181.16        99.81       280.97       0.3975          1.1212            1.1139         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            181.16       149.03       330.19       0.5392          1.0695            1.0619         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            181.16       103.79       284.95       0.3975          1.1212            1.1139         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            181.16       153.66       334.82       0.5391          1.0695            1.0619         2.00
IVF-Binary-256-nl223-random (self)                       181.16       239.10       420.26       0.4179          1.1260            1.1253         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             206.97        47.08       254.05       0.1179          1.5669            1.5683         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             206.97        47.64       254.61       0.1179          1.5669            1.5683         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             206.97        53.85       260.82       0.1179          1.5669            1.5683         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            206.97       102.35       309.32       0.4042          1.1173            1.1107         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            206.97       150.17       357.14       0.5466          1.0673            1.0605         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            206.97       101.22       308.19       0.4041          1.1173            1.1107         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            206.97       162.89       369.86       0.5465          1.0674            1.0605         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            206.97       109.01       315.98       0.4040          1.1173            1.1107         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            206.97       154.50       361.47       0.5464          1.0674            1.0605         2.09
IVF-Binary-256-nl316-random (self)                       206.97       245.47       452.44       0.4258          1.1212            1.1215         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)                 195.06        40.69       235.74       0.1032          1.6184            1.6249         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)                195.06        43.92       238.98       0.1031          1.6186            1.6250         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)                195.06        48.01       243.07       0.1031          1.6186            1.6250         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)                195.06        95.30       290.35       0.3744          1.1342            1.1273         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)                195.06       150.14       345.19       0.5163          1.0767            1.0691         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)               195.06        97.18       292.23       0.3741          1.1342            1.1273         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)               195.06       147.99       343.05       0.5158          1.0768            1.0691         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)               195.06       101.48       296.53       0.3741          1.1342            1.1273         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)               195.06       153.65       348.71       0.5158          1.0768            1.0691         1.93
IVF-Binary-256-nl158-pca (self)                          195.06       226.02       421.07       0.3981          1.1378            1.1374         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                203.05        45.35       248.40       0.1108          1.5929            1.5980         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                203.05        45.61       248.66       0.1108          1.5930            1.5981         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                203.05        50.55       253.60       0.1108          1.5930            1.5981         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               203.05        98.88       301.93       0.3884          1.1259            1.1194         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               203.05       150.26       353.32       0.5286          1.0726            1.0651         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               203.05        99.95       303.00       0.3883          1.1259            1.1194         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               203.05       150.05       353.10       0.5285          1.0726            1.0651         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               203.05       105.97       309.03       0.3883          1.1259            1.1194         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               203.05       153.73       356.78       0.5284          1.0726            1.0651         2.00
IVF-Binary-256-nl223-pca (self)                          203.05       232.19       435.24       0.4110          1.1294            1.1295         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                241.63        47.07       288.71       0.1162          1.5739            1.5756         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                241.63        47.49       289.13       0.1161          1.5740            1.5756         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                241.63        51.14       292.77       0.1161          1.5740            1.5756         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               241.63       118.13       359.76       0.3964          1.1214            1.1153         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               241.63       149.61       391.24       0.5365          1.0702            1.0629         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               241.63       100.08       341.72       0.3963          1.1214            1.1153         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               241.63       150.53       392.16       0.5363          1.0703            1.0629         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               241.63       104.40       346.03       0.3963          1.1214            1.1153         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               241.63       153.93       395.56       0.5363          1.0703            1.0629         2.09
IVF-Binary-256-nl316-pca (self)                          241.63       241.60       483.23       0.4201          1.1244            1.1253         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)              195.59        57.42       253.01       0.1531          1.4883            1.4927         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)             195.59        61.47       257.06       0.1530          1.4884            1.4927         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)             195.59        68.16       263.75       0.1530          1.4884            1.4927         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)             195.59       116.31       311.90       0.4702          1.0897            1.0869         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)             195.59       168.43       364.02       0.6180          1.0481            1.0441         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)            195.59       120.75       316.34       0.4701          1.0897            1.0869         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)            195.59       174.53       370.12       0.6179          1.0481            1.0441         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)            195.59       125.52       321.11       0.4701          1.0897            1.0869         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)            195.59       177.84       373.43       0.6179          1.0481            1.0441         3.71
IVF-Binary-512-nl158-random (self)                       195.59       301.57       497.16       0.4899          1.0959            1.0957         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             192.12        62.10       254.22       0.1575          1.4765            1.4813         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             192.12        63.03       255.15       0.1575          1.4766            1.4813         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             192.12        70.00       262.12       0.1575          1.4766            1.4813         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            192.12       119.68       311.80       0.4765          1.0872            1.0844         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            192.12       170.96       363.08       0.6238          1.0469            1.0430         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            192.12       119.37       311.49       0.4765          1.0872            1.0844         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            192.12       172.02       364.14       0.6238          1.0469            1.0430         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            192.12       126.93       319.06       0.4765          1.0872            1.0844         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            192.12       178.92       371.05       0.6238          1.0469            1.0430         3.77
IVF-Binary-512-nl223-random (self)                       192.12       311.97       504.09       0.4964          1.0933            1.0930         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             234.79        69.01       303.80       0.1599          1.4702            1.4747         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             234.79        75.71       310.50       0.1599          1.4702            1.4747         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             234.79        70.72       305.50       0.1599          1.4702            1.4747         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            234.79       121.62       356.41       0.4803          1.0858            1.0829         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            234.79       171.44       406.22       0.6266          1.0464            1.0422         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            234.79       120.95       355.74       0.4803          1.0858            1.0829         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            234.79       180.92       415.71       0.6266          1.0464            1.0422         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            234.79       126.62       361.41       0.4803          1.0858            1.0829         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            234.79       177.15       411.94       0.6266          1.0464            1.0422         3.86
IVF-Binary-512-nl316-random (self)                       234.79       310.78       545.57       0.4998          1.0920            1.0916         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)                 219.23        57.48       276.70       0.1520          1.4901            1.4951         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)                219.23        61.74       280.96       0.1520          1.4901            1.4951         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)                219.23        67.89       287.12       0.1520          1.4901            1.4951         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)                219.23       116.21       335.43       0.4653          1.0910            1.0875         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)                219.23       167.97       387.19       0.6131          1.0489            1.0445         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)               219.23       124.48       343.71       0.4652          1.0910            1.0875         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)               219.23       171.09       390.32       0.6129          1.0490            1.0445         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)               219.23       130.84       350.07       0.4652          1.0910            1.0875         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)               219.23       178.09       397.32       0.6129          1.0490            1.0445         3.71
IVF-Binary-512-nl158-pca (self)                          219.23       297.54       516.77       0.4881          1.0964            1.0962         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                214.67        63.81       278.48       0.1566          1.4793            1.4840         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                214.67        63.55       278.22       0.1566          1.4793            1.4840         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                214.67        70.19       284.86       0.1566          1.4793            1.4840         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               214.67       116.45       331.12       0.4720          1.0885            1.0852         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               214.67       168.60       383.27       0.6195          1.0476            1.0433         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               214.67       117.02       331.68       0.4720          1.0885            1.0852         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               214.67       170.32       384.99       0.6194          1.0476            1.0433         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               214.67       124.34       339.01       0.4720          1.0885            1.0852         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               214.67       177.55       392.21       0.6194          1.0476            1.0433         3.77
IVF-Binary-512-nl223-pca (self)                          214.67       298.06       512.73       0.4939          1.0941            1.0935         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                233.67        64.12       297.79       0.1590          1.4722            1.4760         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                233.67        65.76       299.43       0.1590          1.4722            1.4760         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                233.67        70.35       304.02       0.1590          1.4722            1.4760         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               233.67       118.45       352.12       0.4756          1.0872            1.0838         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               233.67       171.29       404.96       0.6233          1.0468            1.0424         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               233.67       120.24       353.91       0.4756          1.0872            1.0838         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               233.67       172.43       406.10       0.6233          1.0468            1.0424         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               233.67       124.84       358.51       0.4756          1.0872            1.0838         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               233.67       178.38       412.05       0.6233          1.0468            1.0424         3.86
IVF-Binary-512-nl316-pca (self)                          233.67       317.33       551.01       0.4983          1.0925            1.0920         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)             223.02        88.81       311.83       0.2190          1.3603            1.3675         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)            223.02        93.95       316.96       0.2190          1.3603            1.3675         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)            223.02       102.81       325.83       0.2190          1.3603            1.3675         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)            223.02       147.18       370.20       0.5917          1.0528            1.0505         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)            223.02       202.15       425.17       0.7418          1.0254            1.0218         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)           223.02       153.49       376.51       0.5916          1.0528            1.0505         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)           223.02       208.83       431.85       0.7418          1.0254            1.0218         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)           223.02       161.37       384.38       0.5916          1.0528            1.0505         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)           223.02       219.02       442.04       0.7418          1.0254            1.0218         7.26
IVF-Binary-1024-nl158-random (self)                      223.02       412.36       635.38       0.6161          1.0564            1.0536         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            222.26        92.21       314.47       0.2206          1.3565            1.3634         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            222.26       100.04       322.30       0.2206          1.3565            1.3634         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            222.26       104.17       326.43       0.2206          1.3565            1.3634         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           222.26       153.05       375.31       0.5949          1.0521            1.0498         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           222.26       206.05       428.31       0.7440          1.0252            1.0215         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           222.26       152.45       374.71       0.5949          1.0521            1.0498         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           222.26       217.63       439.89       0.7440          1.0252            1.0215         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           222.26       163.13       385.39       0.5949          1.0521            1.0498         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           222.26       218.11       440.37       0.7440          1.0252            1.0215         7.32
IVF-Binary-1024-nl223-random (self)                      222.26       410.04       632.30       0.6190          1.0557            1.0528         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            247.18        95.61       342.79       0.2221          1.3535            1.3614         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)            247.18        96.58       343.76       0.2221          1.3535            1.3614         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)            247.18       106.16       353.34       0.2221          1.3535            1.3614         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)           247.18       152.78       399.96       0.5967          1.0516            1.0491         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)           247.18       208.19       455.37       0.7457          1.0249            1.0212         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)           247.18       153.04       400.22       0.5967          1.0516            1.0491         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)           247.18       209.15       456.33       0.7457          1.0249            1.0212         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)           247.18       160.09       407.27       0.5967          1.0516            1.0491         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)           247.18       216.88       464.06       0.7457          1.0249            1.0212         7.41
IVF-Binary-1024-nl316-random (self)                      247.18       418.46       665.64       0.6207          1.0553            1.0525         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)                254.10        89.17       343.28       0.2157          1.3674            1.3744         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)               254.10        93.99       348.09       0.2157          1.3674            1.3744         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)               254.10       103.05       357.15       0.2157          1.3674            1.3744         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)               254.10       147.42       401.53       0.5822          1.0549            1.0520         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)               254.10       203.00       457.10       0.7325          1.0268            1.0228         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)              254.10       151.20       405.31       0.5822          1.0549            1.0520         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)              254.10       211.70       465.80       0.7324          1.0268            1.0228         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)              254.10       160.11       414.21       0.5822          1.0549            1.0520         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)              254.10       226.44       480.54       0.7324          1.0268            1.0228         7.26
IVF-Binary-1024-nl158-pca (self)                         254.10       411.01       665.11       0.6061          1.0590            1.0559         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               246.71        93.52       340.23       0.2181          1.3633            1.3703         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               246.71        94.43       341.14       0.2181          1.3633            1.3703         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               246.71       104.22       350.93       0.2181          1.3633            1.3703         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              246.71       153.02       399.73       0.5854          1.0540            1.0509         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              246.71       205.82       452.53       0.7356          1.0263            1.0225         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              246.71       153.40       400.11       0.5854          1.0540            1.0510         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              246.71       207.98       454.69       0.7356          1.0263            1.0225         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              246.71       166.95       413.66       0.5854          1.0540            1.0510         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              246.71       218.67       465.38       0.7356          1.0263            1.0225         7.32
IVF-Binary-1024-nl223-pca (self)                         246.71       412.73       659.44       0.6099          1.0580            1.0548         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               265.54       101.55       367.09       0.2192          1.3606            1.3677         7.41
IVF-Binary-1024-nl316-np17-rf0-pca (query)               265.54        96.97       362.51       0.2192          1.3606            1.3677         7.41
IVF-Binary-1024-nl316-np25-rf0-pca (query)               265.54       104.91       370.45       0.2192          1.3606            1.3677         7.41
IVF-Binary-1024-nl316-np15-rf10-pca (query)              265.54       152.30       417.84       0.5874          1.0535            1.0507         7.41
IVF-Binary-1024-nl316-np15-rf20-pca (query)              265.54       207.72       473.25       0.7376          1.0260            1.0222         7.41
IVF-Binary-1024-nl316-np17-rf10-pca (query)              265.54       152.60       418.14       0.5874          1.0535            1.0507         7.41
IVF-Binary-1024-nl316-np17-rf20-pca (query)              265.54       211.59       477.13       0.7376          1.0260            1.0222         7.41
IVF-Binary-1024-nl316-np25-rf10-pca (query)              265.54       160.85       426.39       0.5874          1.0535            1.0507         7.41
IVF-Binary-1024-nl316-np25-rf20-pca (query)              265.54       223.40       488.94       0.7376          1.0260            1.0222         7.41
IVF-Binary-1024-nl316-pca (self)                         265.54       429.75       695.29       0.6119          1.0574            1.0542         7.41
IVF-Binary-256-nl158-np7-rf0-sign (query)                150.28       133.51       283.80       0.1044          1.6450            1.6501         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               150.28       133.21       283.49       0.1044          1.6453            1.6504         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               150.28       138.86       289.14       0.1044          1.6453            1.6504         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               150.28       178.14       328.42       0.3835          1.1315            1.1231         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               150.28       290.29       440.57       0.5344          1.0723            1.0633         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              150.28       166.75       317.04       0.3829          1.1316            1.1231         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              150.28       293.61       443.89       0.5338          1.0724            1.0633         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              150.28       172.76       323.05       0.3829          1.1317            1.1231         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              150.28       299.63       449.91       0.5337          1.0725            1.0633         1.68
IVF-Binary-256-nl158-sign (self)                         150.28       460.56       610.84       0.4024          1.1396            1.1355         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               152.67       132.99       285.66       0.1045          1.6388            1.6464         1.74
IVF-Binary-256-nl223-np14-rf0-sign (query)               152.67       134.10       286.76       0.1045          1.6389            1.6465         1.74
IVF-Binary-256-nl223-np21-rf0-sign (query)               152.67       137.51       290.18       0.1045          1.6389            1.6465         1.74
IVF-Binary-256-nl223-np11-rf10-sign (query)              152.67       169.01       321.67       0.3881          1.1285            1.1206         1.74
IVF-Binary-256-nl223-np11-rf20-sign (query)              152.67       291.16       443.83       0.5362          1.0716            1.0629         1.74
IVF-Binary-256-nl223-np14-rf10-sign (query)              152.67       168.56       321.23       0.3879          1.1285            1.1206         1.74
IVF-Binary-256-nl223-np14-rf20-sign (query)              152.67       293.99       446.66       0.5360          1.0716            1.0629         1.74
IVF-Binary-256-nl223-np21-rf10-sign (query)              152.67       174.04       326.70       0.3878          1.1285            1.1206         1.74
IVF-Binary-256-nl223-np21-rf20-sign (query)              152.67       299.92       452.59       0.5358          1.0717            1.0630         1.74
IVF-Binary-256-nl223-sign (self)                         152.67       452.67       605.33       0.4072          1.1361            1.1333         1.74
IVF-Binary-256-nl316-np15-rf0-sign (query)               166.93       132.87       299.80       0.1053          1.6356            1.6439         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               166.93       133.67       300.60       0.1052          1.6357            1.6441         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               166.93       137.59       304.52       0.1052          1.6358            1.6441         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              166.93       169.18       336.11       0.3920          1.1263            1.1192         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              166.93       297.39       464.32       0.5384          1.0708            1.0621         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              166.93       169.36       336.29       0.3919          1.1264            1.1192         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              166.93       300.32       467.25       0.5382          1.0708            1.0621         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              166.93       187.59       354.52       0.3918          1.1264            1.1192         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              166.93       301.03       467.96       0.5380          1.0708            1.0622         1.84
IVF-Binary-256-nl316-sign (self)                         166.93       462.28       629.21       0.4115          1.1336            1.1313         1.84
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 512D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.53     1_263.18     1_331.71       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.53     4_234.02     4_302.55       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                132.45       262.96       395.41       0.0733          1.4884            1.4968         2.03
ExhaustiveBinary-256-random-rf10 (query)                 132.45       379.16       511.61       0.2947          1.1327            1.1260         2.03
ExhaustiveBinary-256-random-rf20 (query)                 132.45       538.53       670.98       0.4174          1.0830            1.0716         2.03
ExhaustiveBinary-256-random (self)                       132.45     1_187.18     1_319.63       0.3150          1.1321            1.1268         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   216.89       259.11       476.00       0.0722          1.4941            1.5001         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    216.89       379.91       596.79       0.2934          1.1324            1.1267         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    216.89       506.63       723.51       0.4181          1.0813            1.0715         2.03
ExhaustiveBinary-256-pca (self)                          216.89     1_190.81     1_407.69       0.3112          1.1338            1.1298         2.03
ExhaustiveBinary-512-random_no_rr (query)                202.36       383.00       585.36       0.1110          1.4064            1.4153         4.05
ExhaustiveBinary-512-random-rf10 (query)                 202.36       515.81       718.18       0.3695          1.0935            1.0919         4.05
ExhaustiveBinary-512-random-rf20 (query)                 202.36       640.52       842.88       0.4981          1.0544            1.0517         4.05
ExhaustiveBinary-512-random (self)                       202.36     1_665.61     1_867.98       0.3856          1.0953            1.0998         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   296.18       384.52       680.70       0.1063          1.4156            1.4272         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    296.18       510.05       806.23       0.3574          1.0990            1.0958         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    296.18       641.17       937.36       0.4853          1.0582            1.0544         4.05
ExhaustiveBinary-512-pca (self)                          296.18     1_616.26     1_912.44       0.3753          1.0995            1.1040         4.05
ExhaustiveBinary-1024-random_no_rr (query)               253.59       551.07       804.66       0.1593          1.3242            1.3318         8.11
ExhaustiveBinary-1024-random-rf10 (query)                253.59       690.68       944.26       0.4456          1.0660            1.0678         8.11
ExhaustiveBinary-1024-random-rf20 (query)                253.59       834.04     1_087.63       0.5824          1.0370            1.0362         8.11
ExhaustiveBinary-1024-random (self)                      253.59     2_260.58     2_514.17       0.4595          1.0714            1.0740         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  338.09       557.29       895.38       0.1599          1.3236            1.3333         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   338.09       701.95     1_040.03       0.4446          1.0658            1.0680         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   338.09       835.71     1_173.80       0.5812          1.0370            1.0362         8.11
ExhaustiveBinary-1024-pca (self)                         338.09     2_279.12     2_617.21       0.4582          1.0716            1.0746         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   84.21       637.12       721.33       0.1292          1.3815            1.3877         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    84.21       693.13       777.34       0.3927          1.0844            1.0833         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    84.21     1_041.57     1_125.78       0.5336          1.0464            1.0444         3.05
ExhaustiveBinary-512-sign (self)                          84.21     2_237.00     2_321.21       0.4063          1.0885            1.0916         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)              301.90        71.70       373.60       0.0805          1.4700            1.4823         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)             301.90        77.87       379.77       0.0805          1.4701            1.4824         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)             301.90        81.57       383.47       0.0805          1.4701            1.4824         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)             301.90       151.99       453.89       0.3107          1.1239            1.1175         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)             301.90       237.50       539.40       0.4324          1.0774            1.0680         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)            301.90       154.28       456.18       0.3103          1.1239            1.1175         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)            301.90       236.01       537.91       0.4318          1.0775            1.0680         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)            301.90       150.13       452.03       0.3103          1.1239            1.1175         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)            301.90       239.12       541.02       0.4318          1.0775            1.0680         2.34
IVF-Binary-256-nl158-random (self)                       301.90       297.51       599.41       0.3297          1.1233            1.1210         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)             311.69        70.50       382.19       0.0864          1.4571            1.4649         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)             311.69        70.89       382.58       0.0864          1.4571            1.4649         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)             311.69        73.71       385.40       0.0864          1.4571            1.4649         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)            311.69       153.79       465.47       0.3210          1.1178            1.1110         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)            311.69       241.07       552.76       0.4453          1.0725            1.0641         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)            311.69       153.23       464.91       0.3209          1.1178            1.1110         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)            311.69       242.33       554.02       0.4452          1.0725            1.0641         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)            311.69       160.00       471.69       0.3209          1.1178            1.1110         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)            311.69       256.62       568.30       0.4451          1.0725            1.0641         2.46
IVF-Binary-256-nl223-random (self)                       311.69       312.66       624.35       0.3394          1.1177            1.1162         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)             362.55        76.29       438.85       0.0939          1.4392            1.4433         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)             362.55        79.19       441.74       0.0939          1.4392            1.4433         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)             362.55        80.09       442.64       0.0939          1.4392            1.4433         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)            362.55       161.20       523.76       0.3359          1.1090            1.1039         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)            362.55       247.96       610.51       0.4642          1.0651            1.0594         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)            362.55       161.31       523.87       0.3358          1.1090            1.1039         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)            362.55       259.31       621.86       0.4641          1.0651            1.0594         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)            362.55       162.22       524.77       0.3358          1.1090            1.1039         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)            362.55       253.08       615.63       0.4640          1.0651            1.0594         2.65
IVF-Binary-256-nl316-random (self)                       362.55       338.67       701.23       0.3546          1.1076            1.1101         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)                 393.14        64.06       457.20       0.0799          1.4729            1.4834         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)                393.14        67.98       461.12       0.0799          1.4731            1.4834         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)                393.14        68.31       461.46       0.0799          1.4731            1.4834         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)                393.14       149.53       542.67       0.3116          1.1220            1.1169         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)                393.14       238.01       631.16       0.4355          1.0743            1.0666         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)               393.14       149.17       542.31       0.3112          1.1221            1.1169         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)               393.14       240.09       633.23       0.4349          1.0743            1.0666         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)               393.14       151.29       544.44       0.3112          1.1221            1.1169         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)               393.14       242.15       635.29       0.4349          1.0743            1.0666         2.34
IVF-Binary-256-nl158-pca (self)                          393.14       289.67       682.81       0.3297          1.1214            1.1221         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)                397.59        74.00       471.58       0.0856          1.4582            1.4654         2.46
IVF-Binary-256-nl223-np14-rf0-pca (query)                397.59        70.36       467.95       0.0856          1.4583            1.4654         2.46
IVF-Binary-256-nl223-np21-rf0-pca (query)                397.59        74.31       471.90       0.0856          1.4583            1.4654         2.46
IVF-Binary-256-nl223-np11-rf10-pca (query)               397.59       154.65       552.24       0.3255          1.1136            1.1084         2.46
IVF-Binary-256-nl223-np11-rf20-pca (query)               397.59       245.40       642.99       0.4490          1.0696            1.0626         2.46
IVF-Binary-256-nl223-np14-rf10-pca (query)               397.59       154.65       552.24       0.3253          1.1137            1.1084         2.46
IVF-Binary-256-nl223-np14-rf20-pca (query)               397.59       245.43       643.01       0.4488          1.0696            1.0626         2.46
IVF-Binary-256-nl223-np21-rf10-pca (query)               397.59       157.72       555.31       0.3253          1.1137            1.1084         2.46
IVF-Binary-256-nl223-np21-rf20-pca (query)               397.59       247.78       645.37       0.4488          1.0696            1.0626         2.46
IVF-Binary-256-nl223-pca (self)                          397.59       310.00       707.58       0.3435          1.1121            1.1151         2.46
IVF-Binary-256-nl316-np15-rf0-pca (query)                446.69        76.54       523.23       0.0943          1.4379            1.4397         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)                446.69        77.29       523.98       0.0943          1.4380            1.4397         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)                446.69        80.62       527.31       0.0943          1.4380            1.4397         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)               446.69       161.39       608.08       0.3383          1.1069            1.1014         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)               446.69       252.08       698.77       0.4601          1.0666            1.0601         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)               446.69       159.69       606.38       0.3383          1.1069            1.1014         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)               446.69       252.38       699.07       0.4601          1.0666            1.0601         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)               446.69       168.74       615.43       0.3383          1.1069            1.1014         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)               446.69       256.72       703.41       0.4600          1.0666            1.0601         2.65
IVF-Binary-256-nl316-pca (self)                          446.69       338.50       785.19       0.3564          1.1050            1.1086         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)              382.01        93.57       475.58       0.1159          1.3974            1.4067         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)             382.01       100.29       482.30       0.1159          1.3974            1.4067         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)             382.01       105.98       488.00       0.1159          1.3974            1.4067         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)             382.01       179.44       561.45       0.3765          1.0907            1.0897         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)             382.01       271.10       653.11       0.5051          1.0527            1.0506         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)            382.01       182.22       564.23       0.3765          1.0907            1.0897         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)            382.01       283.75       665.76       0.5051          1.0527            1.0506         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)            382.01       188.03       570.05       0.3765          1.0907            1.0897         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)            382.01       275.90       657.91       0.5051          1.0527            1.0506         4.36
IVF-Binary-512-nl158-random (self)                       382.01       410.29       792.30       0.3919          1.0930            1.0978         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)             384.59        95.79       480.38       0.1199          1.3890            1.3964         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)             384.59        96.90       481.48       0.1199          1.3890            1.3964         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)             384.59       103.17       487.76       0.1199          1.3890            1.3964         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)            384.59       183.26       567.85       0.3824          1.0882            1.0877         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)            384.59       273.14       657.73       0.5105          1.0513            1.0495         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)            384.59       182.49       567.07       0.3824          1.0882            1.0877         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)            384.59       279.17       663.75       0.5104          1.0513            1.0495         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)            384.59       186.64       571.23       0.3824          1.0882            1.0877         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)            384.59       280.05       664.64       0.5104          1.0513            1.0495         4.49
IVF-Binary-512-nl223-random (self)                       384.59       425.69       810.27       0.3975          1.0909            1.0956         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)             433.07       103.12       536.18       0.1243          1.3792            1.3845         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)             433.07       103.75       536.81       0.1243          1.3792            1.3845         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)             433.07       107.04       540.11       0.1243          1.3792            1.3845         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)            433.07       190.58       623.65       0.3890          1.0856            1.0851         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)            433.07       285.00       718.07       0.5169          1.0501            1.0480         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)            433.07       189.59       622.66       0.3890          1.0856            1.0851         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)            433.07       284.20       717.27       0.5168          1.0501            1.0480         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)            433.07       193.59       626.66       0.3890          1.0856            1.0851         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)            433.07       296.53       729.59       0.5168          1.0501            1.0480         4.67
IVF-Binary-512-nl316-random (self)                       433.07       449.79       882.85       0.4038          1.0884            1.0931         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)                 459.96        92.49       552.45       0.1115          1.4050            1.4150         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)                459.96        93.87       553.83       0.1115          1.4050            1.4150         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)                459.96        97.79       557.75       0.1115          1.4050            1.4150         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)                459.96       177.48       637.44       0.3654          1.0955            1.0927         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)                459.96       267.31       727.27       0.4932          1.0559            1.0530         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)               459.96       179.00       638.95       0.3654          1.0955            1.0927         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)               459.96       271.51       731.47       0.4931          1.0559            1.0530         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)               459.96       182.07       642.03       0.3654          1.0955            1.0927         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)               459.96       273.94       733.90       0.4931          1.0559            1.0530         4.36
IVF-Binary-512-nl158-pca (self)                          459.96       411.28       871.24       0.3830          1.0963            1.1015         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)                476.59        95.54       572.13       0.1165          1.3954            1.4036         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)                476.59       100.91       577.49       0.1165          1.3954            1.4036         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)                476.59       100.81       577.40       0.1165          1.3954            1.4036         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)               476.59       182.45       659.03       0.3727          1.0920            1.0903         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)               476.59       271.53       748.12       0.5003          1.0542            1.0515         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)               476.59       181.78       658.37       0.3727          1.0920            1.0903         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)               476.59       276.80       753.38       0.5002          1.0542            1.0515         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)               476.59       185.80       662.39       0.3727          1.0920            1.0903         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)               476.59       281.05       757.64       0.5002          1.0542            1.0515         4.49
IVF-Binary-512-nl223-pca (self)                          476.59       429.09       905.68       0.3895          1.0936            1.0987         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)                522.11       102.68       624.79       0.1211          1.3849            1.3913         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)                522.11       105.50       627.61       0.1211          1.3849            1.3913         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)                522.11       107.50       629.61       0.1211          1.3849            1.3913         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)               522.11       192.52       714.63       0.3797          1.0890            1.0878         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)               522.11       284.64       806.75       0.5070          1.0525            1.0502         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)               522.11       189.49       711.60       0.3797          1.0890            1.0878         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)               522.11       282.05       804.17       0.5070          1.0525            1.0502         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)               522.11       192.13       714.25       0.3797          1.0890            1.0878         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)               522.11       289.75       811.86       0.5070          1.0525            1.0502         4.67
IVF-Binary-512-nl316-pca (self)                          522.11       448.40       970.51       0.3954          1.0912            1.0960         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)             435.49       138.82       574.31       0.1619          1.3203            1.3287         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)            435.49       145.02       580.51       0.1619          1.3203            1.3287         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)            435.49       148.77       584.26       0.1619          1.3203            1.3287         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)            435.49       234.15       669.64       0.4489          1.0650            1.0669         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)            435.49       331.86       767.35       0.5855          1.0364            1.0357         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)           435.49       235.33       670.82       0.4489          1.0650            1.0669         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)           435.49       337.83       773.32       0.5855          1.0364            1.0357         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)           435.49       240.84       676.33       0.4489          1.0650            1.0669         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)           435.49       345.85       781.34       0.5855          1.0364            1.0357         8.42
IVF-Binary-1024-nl158-random (self)                      435.49       604.48     1_039.97       0.4629          1.0705            1.0733         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)            437.79       144.93       582.71       0.1637          1.3169            1.3260         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)            437.79       146.43       584.22       0.1637          1.3169            1.3260         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)            437.79       151.13       588.92       0.1637          1.3169            1.3260         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)           437.79       238.50       676.29       0.4516          1.0643            1.0661         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)           437.79       340.42       778.21       0.5885          1.0359            1.0353         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)           437.79       237.65       675.43       0.4516          1.0643            1.0661         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)           437.79       342.46       780.25       0.5885          1.0359            1.0353         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)           437.79       245.22       683.00       0.4516          1.0643            1.0661         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)           437.79       352.28       790.07       0.5885          1.0359            1.0353         8.54
IVF-Binary-1024-nl223-random (self)                      437.79       621.26     1_059.05       0.4664          1.0695            1.0722         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)            488.18       153.13       641.32       0.1655          1.3135            1.3227         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)            488.18       154.27       642.45       0.1655          1.3135            1.3227         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)            488.18       160.29       648.47       0.1655          1.3135            1.3227         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)           488.18       250.54       738.73       0.4555          1.0633            1.0652         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)           488.18       353.24       841.42       0.5912          1.0355            1.0347         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)           488.18       248.06       736.24       0.4555          1.0633            1.0652         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)           488.18       350.71       838.89       0.5912          1.0355            1.0347         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)           488.18       255.26       743.44       0.4555          1.0633            1.0652         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)           488.18       358.96       847.14       0.5912          1.0355            1.0347         8.72
IVF-Binary-1024-nl316-random (self)                      488.18       650.08     1_138.26       0.4694          1.0686            1.0714         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)                520.33       140.25       660.57       0.1623          1.3200            1.3305         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)               520.33       144.90       665.23       0.1623          1.3200            1.3305         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)               520.33       147.79       668.11       0.1623          1.3200            1.3305         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)               520.33       231.26       751.59       0.4481          1.0648            1.0671         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)               520.33       335.26       855.58       0.5841          1.0365            1.0359         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)              520.33       234.30       754.62       0.4481          1.0648            1.0671         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)              520.33       347.60       867.93       0.5841          1.0365            1.0359         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)              520.33       242.21       762.54       0.4481          1.0648            1.0671         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)              520.33       346.68       867.01       0.5841          1.0365            1.0359         8.42
IVF-Binary-1024-nl158-pca (self)                         520.33       614.04     1_134.36       0.4613          1.0708            1.0737         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)               523.31       145.66       668.96       0.1642          1.3165            1.3271         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)               523.31       146.56       669.86       0.1642          1.3165            1.3271         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)               523.31       152.39       675.69       0.1642          1.3165            1.3271         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)              523.31       236.28       759.58       0.4509          1.0640            1.0665         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)              523.31       345.72       869.02       0.5863          1.0361            1.0356         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)              523.31       247.48       770.79       0.4509          1.0640            1.0665         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)              523.31       342.86       866.17       0.5863          1.0361            1.0356         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)              523.31       244.02       767.32       0.4509          1.0640            1.0665         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)              523.31       350.30       873.60       0.5863          1.0361            1.0356         8.54
IVF-Binary-1024-nl223-pca (self)                         523.31       619.82     1_143.13       0.4642          1.0699            1.0728         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)               647.41       154.07       801.48       0.1662          1.3128            1.3236         8.72
IVF-Binary-1024-nl316-np17-rf0-pca (query)               647.41       154.28       801.68       0.1662          1.3128            1.3236         8.72
IVF-Binary-1024-nl316-np25-rf0-pca (query)               647.41       161.81       809.22       0.1662          1.3128            1.3236         8.72
IVF-Binary-1024-nl316-np15-rf10-pca (query)              647.41       251.41       898.82       0.4540          1.0631            1.0656         8.72
IVF-Binary-1024-nl316-np15-rf20-pca (query)              647.41       349.82       997.23       0.5899          1.0355            1.0352         8.72
IVF-Binary-1024-nl316-np17-rf10-pca (query)              647.41       248.90       896.31       0.4540          1.0631            1.0656         8.72
IVF-Binary-1024-nl316-np17-rf20-pca (query)              647.41       350.01       997.42       0.5899          1.0355            1.0352         8.72
IVF-Binary-1024-nl316-np25-rf10-pca (query)              647.41       255.05       902.45       0.4540          1.0631            1.0656         8.72
IVF-Binary-1024-nl316-np25-rf20-pca (query)              647.41       358.49     1_005.89       0.5899          1.0355            1.0352         8.72
IVF-Binary-1024-nl316-pca (self)                         647.41       647.19     1_294.59       0.4672          1.0690            1.0721         8.72
IVF-Binary-512-nl158-np7-rf0-sign (query)                257.10       254.03       511.12       0.1294          1.3819            1.3873         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)               257.10       258.60       515.70       0.1294          1.3819            1.3873         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)               257.10       269.69       526.79       0.1294          1.3819            1.3873         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)               257.10       318.00       575.10       0.3962          1.0829            1.0824         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)               257.10       565.46       822.56       0.5357          1.0459            1.0442         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)              257.10       324.74       581.84       0.3960          1.0829            1.0824         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)              257.10       599.83       856.93       0.5355          1.0459            1.0442         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)              257.10       325.50       582.60       0.3960          1.0829            1.0824         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)              257.10       572.70       829.80       0.5355          1.0459            1.0442         3.36
IVF-Binary-512-nl158-sign (self)                         257.10       885.05     1_142.15       0.4094          1.0875            1.0908         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               268.28       257.46       525.74       0.1294          1.3807            1.3868         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               268.28       260.44       528.72       0.1294          1.3807            1.3868         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               268.28       279.28       547.56       0.1294          1.3807            1.3868         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              268.28       324.15       592.43       0.3980          1.0823            1.0820         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              268.28       567.87       836.15       0.5369          1.0456            1.0440         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              268.28       323.93       592.21       0.3980          1.0823            1.0820         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              268.28       576.04       844.32       0.5369          1.0456            1.0440         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              268.28       338.92       607.20       0.3980          1.0823            1.0820         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              268.28       579.76       848.03       0.5369          1.0456            1.0440         3.49
IVF-Binary-512-nl223-sign (self)                         268.28       898.88     1_167.16       0.4111          1.0869            1.0902         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)               316.89       263.38       580.27       0.1294          1.3806            1.3862         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)               316.89       265.87       582.77       0.1294          1.3806            1.3862         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)               316.89       268.47       585.36       0.1294          1.3806            1.3862         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)              316.89       329.29       646.19       0.4004          1.0812            1.0811         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)              316.89       577.58       894.48       0.5387          1.0452            1.0436         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)              316.89       331.70       648.60       0.4004          1.0812            1.0811         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)              316.89       582.08       898.97       0.5387          1.0452            1.0436         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)              316.89       337.50       654.40       0.4004          1.0812            1.0811         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)              316.89       590.05       906.95       0.5387          1.0452            1.0436         3.67
IVF-Binary-512-nl316-sign (self)                         316.89       917.82     1_234.72       0.4138          1.0859            1.0894         3.67
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 768D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       100.77     1_793.93     1_894.70       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.77     6_136.41     6_237.18       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                194.03       282.06       476.09       0.0662          1.3769            1.3786         2.28
ExhaustiveBinary-256-random-rf10 (query)                 194.03       413.51       607.55       0.2741          1.1112            1.1017         2.28
ExhaustiveBinary-256-random-rf20 (query)                 194.03       558.63       752.66       0.3877          1.0706            1.0590         2.28
ExhaustiveBinary-256-random (self)                       194.03     1_274.38     1_468.41       0.2874          1.1077            1.0994         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   391.69       279.64       671.33       0.0653          1.3793            1.3770         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    391.69       412.48       804.17       0.2701          1.1138            1.1018         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    391.69       571.75       963.45       0.3852          1.0726            1.0585         2.28
ExhaustiveBinary-256-pca (self)                          391.69     1_272.29     1_663.98       0.2820          1.1100            1.0990         2.28
ExhaustiveBinary-512-random_no_rr (query)                298.00       393.21       691.21       0.0934          1.3249            1.3316         4.55
ExhaustiveBinary-512-random-rf10 (query)                 298.00       544.66       842.66       0.3220          1.0860            1.0806         4.55
ExhaustiveBinary-512-random-rf20 (query)                 298.00       694.29       992.29       0.4345          1.0527            1.0485         4.55
ExhaustiveBinary-512-random (self)                       298.00     1_711.98     2_009.98       0.3344          1.0826            1.0842         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   486.88       393.18       880.06       0.0951          1.3226            1.3263         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    486.88       543.24     1_030.11       0.3246          1.0847            1.0788         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    486.88       696.47     1_183.34       0.4400          1.0517            1.0468         4.55
ExhaustiveBinary-512-pca (self)                          486.88     1_715.77     2_202.65       0.3361          1.0819            1.0825         4.55
ExhaustiveBinary-1024-random_no_rr (query)               495.87       594.22     1_090.09       0.1319          1.2685            1.2723         9.11
ExhaustiveBinary-1024-random-rf10 (query)                495.87       764.10     1_259.98       0.3742          1.0644            1.0666         9.11
ExhaustiveBinary-1024-random-rf20 (query)                495.87       928.96     1_424.83       0.4906          1.0386            1.0390         9.11
ExhaustiveBinary-1024-random (self)                      495.87     2_477.68     2_973.55       0.3823          1.0666            1.0715         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  691.36       599.58     1_290.94       0.1355          1.2623            1.2651         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   691.36       767.49     1_458.84       0.3804          1.0622            1.0643         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   691.36       950.33     1_641.68       0.4993          1.0369            1.0374         9.11
ExhaustiveBinary-1024-pca (self)                         691.36     2_505.47     3_196.83       0.3870          1.0651            1.0695         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  128.34       808.32       936.66       0.1284          1.2822            1.2821         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   128.34       882.87     1_011.21       0.3618          1.0706            1.0699         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   128.34     1_353.28     1_481.62       0.4847          1.0407            1.0395         4.58
ExhaustiveBinary-768-sign (self)                         128.34     3_076.34     3_204.68       0.3694          1.0716            1.0747         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)              436.93        96.28       533.20       0.0717          1.3656            1.3726         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)             436.93        93.14       530.07       0.0717          1.3656            1.3726         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)             436.93        98.88       535.80       0.0717          1.3656            1.3726         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)             436.93       191.06       627.99       0.2823          1.1072            1.0974         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)             436.93       299.79       736.72       0.3951          1.0682            1.0573         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)            436.93       189.86       626.79       0.2822          1.1072            1.0974         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)            436.93       297.16       734.08       0.3947          1.0683            1.0573         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)            436.93       186.90       623.83       0.2822          1.1072            1.0974         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)            436.93       299.33       736.25       0.3947          1.0683            1.0573         2.74
IVF-Binary-256-nl158-random (self)                       436.93       381.28       818.21       0.2953          1.1042            1.0971         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)             443.66        92.64       536.30       0.0767          1.3556            1.3613         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)             443.66        94.05       537.71       0.0766          1.3556            1.3613         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)             443.66        94.38       538.04       0.0766          1.3556            1.3613         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)            443.66       196.96       640.62       0.2997          1.0966            1.0891         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)            443.66       310.22       753.88       0.4133          1.0600            1.0531         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)            443.66       194.31       637.98       0.2997          1.0966            1.0891         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)            443.66       308.62       752.28       0.4132          1.0600            1.0531         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)            443.66       197.87       641.53       0.2997          1.0966            1.0891         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)            443.66       311.51       755.18       0.4132          1.0600            1.0531         2.93
IVF-Binary-256-nl223-random (self)                       443.66       408.65       852.31       0.3130          1.0920            1.0906         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)             522.10        99.50       621.61       0.0842          1.3401            1.3418         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)             522.10       100.84       622.94       0.0842          1.3401            1.3418         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)             522.10       103.32       625.42       0.0842          1.3401            1.3418         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)            522.10       206.67       728.77       0.3158          1.0872            1.0819         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)            522.10       319.73       841.83       0.4294          1.0546            1.0494         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)            522.10       204.59       726.70       0.3157          1.0872            1.0819         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)            522.10       317.64       839.74       0.4293          1.0546            1.0494         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)            522.10       206.64       728.74       0.3157          1.0872            1.0819         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)            522.10       323.33       845.43       0.4293          1.0547            1.0494         3.21
IVF-Binary-256-nl316-random (self)                       522.10       442.95       965.05       0.3296          1.0820            1.0840         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)                 626.76        82.82       709.59       0.0707          1.3664            1.3706         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)                626.76        85.05       711.81       0.0706          1.3664            1.3706         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)                626.76        86.73       713.49       0.0706          1.3664            1.3706         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)                626.76       185.17       811.94       0.2806          1.1073            1.0971         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)                626.76       297.30       924.07       0.3953          1.0679            1.0569         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)               626.76       185.35       812.12       0.2805          1.1073            1.0972         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)               626.76       306.29       933.05       0.3948          1.0679            1.0569         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)               626.76       186.39       813.15       0.2805          1.1073            1.0972         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)               626.76       302.37       929.13       0.3948          1.0679            1.0569         2.74
IVF-Binary-256-nl158-pca (self)                          626.76       375.52     1_002.29       0.2928          1.1031            1.0962         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)                639.74        90.87       730.62       0.0759          1.3561            1.3594         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)                639.74        91.97       731.71       0.0759          1.3561            1.3594         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)                639.74        94.52       734.26       0.0759          1.3561            1.3594         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)               639.74       195.09       834.84       0.2958          1.0975            1.0889         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)               639.74       308.28       948.03       0.4111          1.0605            1.0529         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)               639.74       199.61       839.36       0.2958          1.0975            1.0889         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)               639.74       317.39       957.14       0.4109          1.0605            1.0529         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)               639.74       195.85       835.59       0.2958          1.0975            1.0889         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)               639.74       312.41       952.15       0.4109          1.0605            1.0529         2.93
IVF-Binary-256-nl223-pca (self)                          639.74       407.70     1_047.45       0.3086          1.0922            1.0903         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)                760.97       107.41       868.38       0.0841          1.3410            1.3409         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)                760.97       103.51       864.48       0.0841          1.3410            1.3409         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)                760.97       117.30       878.27       0.0841          1.3410            1.3409         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)               760.97       221.67       982.64       0.3099          1.0899            1.0826         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)               760.97       317.90     1_078.87       0.4237          1.0569            1.0501         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)               760.97       202.56       963.53       0.3098          1.0899            1.0826         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)               760.97       319.36     1_080.33       0.4237          1.0569            1.0501         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)               760.97       205.52       966.49       0.3098          1.0899            1.0826         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)               760.97       325.41     1_086.38       0.4236          1.0569            1.0501         3.21
IVF-Binary-256-nl316-pca (self)                          760.97       445.05     1_206.02       0.3216          1.0842            1.0848         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)              542.04       117.64       659.68       0.0970          1.3194            1.3257         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)             542.04       121.75       663.79       0.0970          1.3194            1.3257         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)             542.04       124.54       666.58       0.0970          1.3194            1.3257         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)             542.04       226.62       768.66       0.3263          1.0842            1.0796         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)             542.04       342.16       884.20       0.4387          1.0516            1.0478         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)            542.04       224.72       766.76       0.3263          1.0842            1.0796         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)            542.04       347.80       889.84       0.4387          1.0516            1.0478         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)            542.04       231.88       773.91       0.3263          1.0842            1.0796         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)            542.04       352.36       894.40       0.4387          1.0516            1.0478         5.02
IVF-Binary-512-nl158-random (self)                       542.04       532.59     1_074.63       0.3383          1.0813            1.0834         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)             550.93       127.45       678.38       0.1021          1.3103            1.3128         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)             550.93       127.42       678.35       0.1021          1.3103            1.3128         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)             550.93       131.68       682.61       0.1021          1.3103            1.3128         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)            550.93       234.04       784.97       0.3351          1.0796            1.0772         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)            550.93       351.83       902.76       0.4471          1.0486            1.0462         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)            550.93       233.83       784.76       0.3351          1.0796            1.0772         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)            550.93       354.06       904.99       0.4471          1.0486            1.0462         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)            550.93       238.41       789.34       0.3351          1.0796            1.0772         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)            550.93       359.12       910.05       0.4471          1.0486            1.0462         5.21
IVF-Binary-512-nl223-random (self)                       550.93       566.31     1_117.24       0.3467          1.0771            1.0811         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)             627.78       135.81       763.60       0.1072          1.3004            1.2998         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)             627.78       138.79       766.57       0.1072          1.3004            1.2998         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)             627.78       140.14       767.92       0.1072          1.3004            1.2998         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)            627.78       254.52       882.30       0.3428          1.0761            1.0745         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)            627.78       371.42       999.21       0.4550          1.0469            1.0448         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)            627.78       247.61       875.39       0.3428          1.0761            1.0745         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)            627.78       397.66     1_025.44       0.4549          1.0469            1.0448         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)            627.78       254.58       882.36       0.3428          1.0761            1.0745         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)            627.78       376.21     1_003.99       0.4549          1.0469            1.0448         5.48
IVF-Binary-512-nl316-random (self)                       627.78       599.22     1_227.00       0.3533          1.0738            1.0787         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)                 728.83       117.97       846.80       0.0987          1.3160            1.3203         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)                728.83       120.97       849.80       0.0987          1.3160            1.3203         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)                728.83       124.54       853.37       0.0987          1.3160            1.3203         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)                728.83       225.19       954.02       0.3300          1.0820            1.0779         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)                728.83       340.26     1_069.09       0.4457          1.0498            1.0461         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)               728.83       226.71       955.54       0.3300          1.0820            1.0779         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)               728.83       344.71     1_073.54       0.4457          1.0498            1.0461         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)               728.83       235.08       963.91       0.3300          1.0820            1.0779         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)               728.83       348.82     1_077.65       0.4457          1.0498            1.0461         5.02
IVF-Binary-512-nl158-pca (self)                          728.83       538.47     1_267.30       0.3408          1.0793            1.0817         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)                746.80       147.56       894.36       0.1034          1.3085            1.3094         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)                746.80       134.34       881.15       0.1034          1.3085            1.3094         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)                746.80       131.30       878.10       0.1034          1.3085            1.3094         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)               746.80       234.52       981.32       0.3378          1.0782            1.0753         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)               746.80       350.55     1_097.35       0.4528          1.0475            1.0447         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)               746.80       235.53       982.33       0.3378          1.0782            1.0753         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)               746.80       355.61     1_102.42       0.4528          1.0475            1.0447         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)               746.80       238.04       984.84       0.3378          1.0782            1.0753         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)               746.80       357.75     1_104.55       0.4528          1.0475            1.0447         5.21
IVF-Binary-512-nl223-pca (self)                          746.80       563.92     1_310.73       0.3477          1.0767            1.0799         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)                815.56       139.29       954.85       0.1085          1.2990            1.2960         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)                815.56       137.13       952.69       0.1085          1.2990            1.2960         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)                815.56       146.27       961.83       0.1085          1.2990            1.2960         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)               815.56       246.37     1_061.93       0.3440          1.0755            1.0732         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)               815.56       367.64     1_183.20       0.4586          1.0462            1.0438         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)               815.56       244.82     1_060.38       0.3440          1.0755            1.0732         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)               815.56       372.27     1_187.83       0.4586          1.0462            1.0438         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)               815.56       251.94     1_067.50       0.3440          1.0755            1.0732         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)               815.56       377.70     1_193.26       0.4586          1.0462            1.0438         5.48
IVF-Binary-512-nl316-pca (self)                          815.56       621.13     1_436.69       0.3534          1.0743            1.0778         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)             739.91       189.56       929.47       0.1338          1.2662            1.2701         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)            739.91       192.40       932.31       0.1338          1.2662            1.2701         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)            739.91       195.34       935.25       0.1338          1.2662            1.2701         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)            739.91       306.11     1_046.02       0.3762          1.0638            1.0662         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)            739.91       436.04     1_175.94       0.4927          1.0382            1.0387         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)           739.91       317.95     1_057.85       0.3762          1.0638            1.0662         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)           739.91       447.45     1_187.36       0.4927          1.0382            1.0387         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)           739.91       326.50     1_066.41       0.3762          1.0638            1.0662         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)           739.91       455.44     1_195.35       0.4927          1.0382            1.0387         9.57
IVF-Binary-1024-nl158-random (self)                      739.91       828.12     1_568.03       0.3844          1.0660            1.0710         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)            753.82       198.65       952.47       0.1365          1.2617            1.2659         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)            753.82       199.30       953.12       0.1365          1.2617            1.2659         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)            753.82       203.43       957.25       0.1365          1.2617            1.2659         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)           753.82       322.20     1_076.03       0.3808          1.0623            1.0653         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)           753.82       447.55     1_201.37       0.4963          1.0374            1.0381         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)           753.82       326.23     1_080.06       0.3808          1.0623            1.0653         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)           753.82       454.97     1_208.80       0.4963          1.0374            1.0381         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)           753.82       335.77     1_089.60       0.3808          1.0623            1.0653         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)           753.82       464.33     1_218.15       0.4963          1.0374            1.0381         9.76
IVF-Binary-1024-nl223-random (self)                      753.82       844.76     1_598.59       0.3882          1.0648            1.0699         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)            833.40       211.35     1_044.76       0.1394          1.2572            1.2623        10.03
IVF-Binary-1024-nl316-np17-rf0-random (query)            833.40       208.17     1_041.57       0.1394          1.2572            1.2623        10.03
IVF-Binary-1024-nl316-np25-rf0-random (query)            833.40       212.65     1_046.05       0.1394          1.2572            1.2623        10.03
IVF-Binary-1024-nl316-np15-rf10-random (query)           833.40       335.48     1_168.88       0.3841          1.0611            1.0642        10.03
IVF-Binary-1024-nl316-np15-rf20-random (query)           833.40       469.08     1_302.48       0.5009          1.0366            1.0374        10.03
IVF-Binary-1024-nl316-np17-rf10-random (query)           833.40       339.05     1_172.46       0.3841          1.0611            1.0642        10.03
IVF-Binary-1024-nl316-np17-rf20-random (query)           833.40       467.08     1_300.49       0.5009          1.0366            1.0374        10.03
IVF-Binary-1024-nl316-np25-rf10-random (query)           833.40       351.65     1_185.05       0.3841          1.0611            1.0642        10.03
IVF-Binary-1024-nl316-np25-rf20-random (query)           833.40       477.54     1_310.94       0.5009          1.0366            1.0374        10.03
IVF-Binary-1024-nl316-random (self)                      833.40       885.89     1_719.30       0.3916          1.0637            1.0688        10.03
IVF-Binary-1024-nl158-np7-rf0-pca (query)                933.02       189.90     1_122.92       0.1374          1.2597            1.2636         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)               933.02       196.47     1_129.49       0.1374          1.2597            1.2636         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)               933.02       197.69     1_130.71       0.1374          1.2597            1.2636         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)               933.02       307.67     1_240.69       0.3829          1.0614            1.0640         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)               933.02       434.30     1_367.32       0.5019          1.0365            1.0370         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)              933.02       316.67     1_249.69       0.3829          1.0614            1.0640         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)              933.02       445.15     1_378.17       0.5019          1.0365            1.0370         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)              933.02       328.38     1_261.40       0.3829          1.0614            1.0640         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)              933.02       454.58     1_387.60       0.5019          1.0365            1.0370         9.57
IVF-Binary-1024-nl158-pca (self)                         933.02       820.70     1_753.71       0.3892          1.0644            1.0690         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)               955.81       199.26     1_155.07       0.1397          1.2562            1.2608         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)               955.81       198.65     1_154.47       0.1397          1.2562            1.2608         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)               955.81       203.22     1_159.04       0.1397          1.2562            1.2608         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)              955.81       317.06     1_272.87       0.3865          1.0602            1.0631         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)              955.81       457.87     1_413.68       0.5043          1.0359            1.0367         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)              955.81       328.30     1_284.11       0.3865          1.0602            1.0631         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)              955.81       451.74     1_407.55       0.5043          1.0359            1.0367         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)              955.81       329.82     1_285.63       0.3865          1.0602            1.0631         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)              955.81       460.81     1_416.62       0.5043          1.0359            1.0367         9.76
IVF-Binary-1024-nl223-pca (self)                         955.81       853.98     1_809.79       0.3926          1.0635            1.0681         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_028.90       211.78     1_240.67       0.1416          1.2529            1.2573        10.03
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_028.90       210.26     1_239.15       0.1416          1.2529            1.2573        10.03
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_028.90       214.25     1_243.14       0.1416          1.2529            1.2573        10.03
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_028.90       341.26     1_370.16       0.3898          1.0592            1.0622        10.03
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_028.90       468.45     1_497.35       0.5084          1.0353            1.0362        10.03
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_028.90       341.11     1_370.00       0.3898          1.0592            1.0622        10.03
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_028.90       468.07     1_496.97       0.5084          1.0353            1.0362        10.03
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_028.90       348.86     1_377.76       0.3898          1.0592            1.0622        10.03
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_028.90       477.34     1_506.23       0.5084          1.0353            1.0362        10.03
IVF-Binary-1024-nl316-pca (self)                       1_028.90       895.11     1_924.01       0.3956          1.0626            1.0672        10.03
IVF-Binary-768-nl158-np7-rf0-sign (query)                378.30       370.68       748.97       0.1284          1.2827            1.2816         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)               378.30       375.26       753.56       0.1284          1.2827            1.2816         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)               378.30       378.67       756.97       0.1284          1.2827            1.2816         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)               378.30       453.57       831.86       0.3632          1.0701            1.0696         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)               378.30       815.60     1_193.90       0.4861          1.0405            1.0394         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)              378.30       457.63       835.92       0.3632          1.0701            1.0696         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)              378.30       821.32     1_199.62       0.4861          1.0405            1.0394         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)              378.30       468.57       846.86       0.3632          1.0701            1.0696         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)              378.30       826.42     1_204.71       0.4861          1.0405            1.0394         5.04
IVF-Binary-768-nl158-sign (self)                         378.30     1_285.80     1_664.10       0.3708          1.0712            1.0744         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)               397.46       382.66       780.12       0.1285          1.2803            1.2810         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)               397.46       379.93       777.39       0.1285          1.2803            1.2810         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)               397.46       396.90       794.36       0.1285          1.2803            1.2810         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)              397.46       462.01       859.47       0.3655          1.0688            1.0690         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)              397.46       825.44     1_222.90       0.4876          1.0399            1.0393         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)              397.46       465.44       862.90       0.3655          1.0688            1.0690         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)              397.46       828.20     1_225.66       0.4876          1.0399            1.0393         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)              397.46       466.88       864.34       0.3655          1.0688            1.0690         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)              397.46       831.74     1_229.20       0.4876          1.0399            1.0393         5.23
IVF-Binary-768-nl223-sign (self)                         397.46     1_325.53     1_722.99       0.3732          1.0703            1.0739         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)               461.82       387.78       849.60       0.1286          1.2804            1.2812         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)               461.82       386.81       848.64       0.1286          1.2804            1.2812         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)               461.82       391.59       853.41       0.1286          1.2804            1.2812         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)              461.82       472.54       934.36       0.3675          1.0684            1.0684         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)              461.82       837.28     1_299.10       0.4883          1.0398            1.0390         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)              461.82       470.44       932.26       0.3675          1.0684            1.0684         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)              461.82       836.61     1_298.44       0.4883          1.0398            1.0390         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)              461.82       476.45       938.27       0.3675          1.0684            1.0684         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)              461.82       839.39     1_301.22       0.4883          1.0398            1.0390         5.51
IVF-Binary-768-nl316-sign (self)                         461.82     1_593.18     2_055.00       0.3745          1.0699            1.0735         5.51
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Cell embeddings

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 256D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        33.02       686.04       719.06       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.02     2_350.14     2_383.17       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                 70.20       238.51       308.71       0.5519          1.8826            1.5884         1.78
ExhaustiveBinary-256-random-rf10 (query)                  70.20       357.51       427.72       0.9881          1.0022            1.0000         1.78
ExhaustiveBinary-256-random-rf20 (query)                  70.20       460.59       530.79       0.9980          1.0003            1.0000         1.78
ExhaustiveBinary-256-random (self)                        70.20     1_162.20     1_232.40       0.9881          1.0022            1.0000         1.78
ExhaustiveBinary-256-pca_no_rr (query)                    95.77       239.42       335.19       0.5930          1.6081            1.4152         1.78
ExhaustiveBinary-256-pca-rf10 (query)                     95.77       377.96       473.72       0.9919          1.0013            1.0000         1.78
ExhaustiveBinary-256-pca-rf20 (query)                     95.77       485.67       581.44       0.9988          1.0001            1.0000         1.78
ExhaustiveBinary-256-pca (self)                           95.77     1_155.47     1_251.23       0.9915          1.0014            1.0000         1.78
ExhaustiveBinary-512-random_no_rr (query)                 82.75       345.03       427.78       0.6306          1.5767            1.3633         3.55
ExhaustiveBinary-512-random-rf10 (query)                  82.75       465.55       548.30       0.9975          1.0004            1.0000         3.55
ExhaustiveBinary-512-random-rf20 (query)                  82.75       617.64       700.39       0.9998          1.0000            1.0000         3.55
ExhaustiveBinary-512-random (self)                        82.75     1_513.09     1_595.84       0.9973          1.0004            1.0000         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   109.62       345.16       454.78       0.6479          1.4884            1.3147         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    109.62       462.84       572.46       0.9983          1.0002            1.0000         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    109.62       595.71       705.33       0.9998          1.0000            1.0000         3.55
ExhaustiveBinary-512-pca (self)                          109.62     1_515.52     1_625.14       0.9981          1.0002            1.0000         3.55
ExhaustiveBinary-1024-random_no_rr (query)               116.33       531.97       648.30       0.6758          1.4452            1.2804         7.10
ExhaustiveBinary-1024-random-rf10 (query)                116.33       622.51       738.84       0.9995          1.0001            1.0000         7.10
ExhaustiveBinary-1024-random-rf20 (query)                116.33       733.25       849.59       0.9999          1.0000            1.0000         7.10
ExhaustiveBinary-1024-random (self)                      116.33     2_051.55     2_167.89       0.9993          1.0001            1.0000         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  139.32       501.11       640.43       0.6838          1.4142            1.2651         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   139.32       618.65       757.97       0.9996          1.0000            1.0000         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   139.32       731.89       871.21       1.0000          1.0000            1.0000         7.10
ExhaustiveBinary-1024-pca (self)                         139.32     2_049.45     2_188.78       0.9995          1.0001            1.0000         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   41.21       460.68       501.89       0.0376         19.4734           14.8778         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    41.21       534.59       575.80       0.1617          2.7567            2.6548         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    41.21       769.60       810.81       0.2739          1.9837            1.9249         1.53
ExhaustiveBinary-256-sign (self)                          41.21     1_535.64     1_576.85       0.1691          2.7353            2.6299         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)              196.92        56.88       253.80       0.5664          1.6566            1.5068         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)             196.92        66.49       263.41       0.5601          1.7154            1.5418         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)             196.92        78.60       275.52       0.5579          1.7512            1.5592         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)             196.92       125.44       322.36       0.9898          1.0019            1.0000         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)             196.92       167.09       364.01       0.9961          1.0008            1.0000         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)            196.92       138.84       335.76       0.9907          1.0016            1.0000         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)            196.92       226.28       423.20       0.9986          1.0002            1.0000         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)            196.92       156.29       353.21       0.9899          1.0018            1.0000         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)            196.92       205.98       402.89       0.9985          1.0002            1.0000         1.93
IVF-Binary-256-nl158-random (self)                       196.92       327.33       524.25       0.9906          1.0016            1.0000         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             217.38        51.76       269.14       0.5628          1.6781            1.5260         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             217.38        55.83       273.20       0.5603          1.7037            1.5421         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             217.38        61.87       279.24       0.5576          1.7448            1.5600         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            217.38       112.03       329.41       0.9913          1.0015            1.0000         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            217.38       162.82       380.19       0.9984          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            217.38       114.49       331.86       0.9910          1.0016            1.0000         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            217.38       168.55       385.92       0.9986          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            217.38       121.97       339.35       0.9900          1.0017            1.0000         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            217.38       177.06       394.44       0.9985          1.0002            1.0000         2.00
IVF-Binary-256-nl223-random (self)                       217.38       297.59       514.97       0.9907          1.0016            1.0000         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             193.35        51.88       245.23       0.5631          1.6756            1.5249         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             193.35        54.75       248.10       0.5617          1.6898            1.5331         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             193.35        60.06       253.41       0.5587          1.7346            1.5511         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            193.35       111.65       305.00       0.9916          1.0014            1.0000         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            193.35       164.22       357.57       0.9987          1.0002            1.0000         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            193.35       113.11       306.46       0.9914          1.0014            1.0000         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            193.35       167.28       360.63       0.9988          1.0002            1.0000         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            193.35       118.58       311.93       0.9904          1.0017            1.0000         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            193.35       175.54       368.89       0.9986          1.0002            1.0000         2.09
IVF-Binary-256-nl316-random (self)                       193.35       292.64       485.99       0.9912          1.0015            1.0000         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)                 218.63        45.10       263.73       0.6041          1.4823            1.3733         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)                218.63        52.94       271.57       0.5993          1.5149            1.3905         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)                218.63        60.91       279.55       0.5978          1.5352            1.3976         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)                218.63       107.59       326.22       0.9920          1.0014            1.0000         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)                218.63       160.84       379.47       0.9964          1.0007            1.0000         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)               218.63       114.99       333.62       0.9933          1.0010            1.0000         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)               218.63       171.17       389.80       0.9991          1.0001            1.0000         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)               218.63       123.88       342.51       0.9927          1.0011            1.0000         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)               218.63       179.65       398.29       0.9990          1.0001            1.0000         1.93
IVF-Binary-256-nl158-pca (self)                          218.63       305.57       524.20       0.9929          1.0011            1.0000         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                201.01        49.19       250.20       0.6022          1.4952            1.3803         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                201.01        52.89       253.90       0.6001          1.5103            1.3880         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                201.01        62.50       263.51       0.5981          1.5331            1.3963         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               201.01       110.93       311.94       0.9935          1.0010            1.0000         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               201.01       164.93       365.94       0.9987          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               201.01       112.33       313.34       0.9933          1.0010            1.0000         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               201.01       167.39       368.40       0.9990          1.0001            1.0000         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               201.01       128.00       329.01       0.9928          1.0011            1.0000         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               201.01       175.81       376.82       0.9990          1.0001            1.0000         2.00
IVF-Binary-256-nl223-pca (self)                          201.01       288.88       489.89       0.9930          1.0011            1.0000         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                216.90        51.57       268.47       0.6016          1.4963            1.3821         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                216.90        53.38       270.28       0.6007          1.5047            1.3864         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                216.90        59.38       276.28       0.5987          1.5267            1.3946         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               216.90       111.12       328.02       0.9938          1.0009            1.0000         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               216.90       165.26       382.16       0.9989          1.0001            1.0000         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               216.90       112.67       329.57       0.9937          1.0010            1.0000         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               216.90       165.62       382.52       0.9991          1.0001            1.0000         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               216.90       117.98       334.88       0.9931          1.0011            1.0000         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               216.90       173.92       390.82       0.9991          1.0001            1.0000         2.09
IVF-Binary-256-nl316-pca (self)                          216.90       282.01       498.91       0.9933          1.0011            1.0000         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)              187.34        63.36       250.70       0.6415          1.4419            1.3288         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)             187.34        75.21       262.55       0.6360          1.4813            1.3474         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)             187.34        89.38       276.72       0.6340          1.5061            1.3538         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)             187.34       133.28       320.62       0.9957          1.0008            1.0000         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)             187.34       180.76       368.09       0.9969          1.0007            1.0000         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)            187.34       138.49       325.82       0.9981          1.0003            1.0000         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)            187.34       196.11       383.45       0.9998          1.0000            1.0000         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)            187.34       151.37       338.71       0.9978          1.0003            1.0000         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)            187.34       209.43       396.77       0.9998          1.0000            1.0000         3.71
IVF-Binary-512-nl158-random (self)                       187.34       384.58       571.92       0.9979          1.0003            1.0000         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             190.72        68.06       258.78       0.6386          1.4543            1.3368         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             190.72        75.32       266.04       0.6363          1.4730            1.3448         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             190.72        85.54       276.26       0.6339          1.5008            1.3535         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            190.72       129.51       320.24       0.9978          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            190.72       185.11       375.84       0.9993          1.0001            1.0000         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            190.72       133.68       324.40       0.9980          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            190.72       193.24       383.96       0.9997          1.0000            1.0000         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            190.72       147.30       338.02       0.9979          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            190.72       204.64       395.36       0.9998          1.0000            1.0000         3.77
IVF-Binary-512-nl223-random (self)                       190.72       366.23       556.96       0.9978          1.0003            1.0000         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             221.31        70.94       292.25       0.6384          1.4561            1.3391         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             221.31        72.81       294.12       0.6372          1.4667            1.3435         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             221.31        82.20       303.51       0.6344          1.4971            1.3518         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            221.31       129.53       350.84       0.9980          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            221.31       185.68       407.00       0.9995          1.0001            1.0000         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            221.31       131.51       352.82       0.9981          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            221.31       186.68       407.99       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            221.31       142.75       364.06       0.9980          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            221.31       198.15       419.46       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-random (self)                       221.31       350.64       571.95       0.9980          1.0003            1.0000         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)                 211.02        63.54       274.57       0.6577          1.3846            1.2857         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)                211.02        75.48       286.50       0.6527          1.4165            1.3000         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)                211.02        88.81       299.84       0.6510          1.4356            1.3057         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)                211.02       126.40       337.42       0.9961          1.0007            1.0000         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)                211.02       181.58       392.60       0.9969          1.0007            1.0000         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)               211.02       139.93       350.95       0.9986          1.0002            1.0000         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)               211.02       195.57       406.60       0.9998          1.0000            1.0000         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)               211.02       153.16       364.18       0.9985          1.0002            1.0000         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)               211.02       208.96       419.99       0.9999          1.0000            1.0000         3.71
IVF-Binary-512-nl158-pca (self)                          211.02       386.57       597.59       0.9985          1.0002            1.0000         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                213.78        68.29       282.08       0.6553          1.3966            1.2913         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                213.78        73.11       286.90       0.6532          1.4106            1.2977         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                213.78        85.03       298.81       0.6513          1.4322            1.3056         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               213.78       130.34       344.12       0.9983          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               213.78       183.24       397.02       0.9994          1.0001            1.0000         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               213.78       133.52       347.30       0.9985          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               213.78       191.53       405.32       0.9998          1.0000            1.0000         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               213.78       145.97       359.75       0.9984          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               213.78       203.21       416.99       0.9999          1.0000            1.0000         3.77
IVF-Binary-512-nl223-pca (self)                          213.78       363.95       577.74       0.9985          1.0002            1.0000         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                239.18        70.20       309.37       0.6548          1.3991            1.2940         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                239.18        72.40       311.58       0.6538          1.4070            1.2973         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                239.18        81.77       320.95       0.6515          1.4289            1.3052         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               239.18       129.37       368.55       0.9985          1.0002            1.0000         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               239.18       186.21       425.39       0.9995          1.0001            1.0000         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               239.18       134.78       373.96       0.9986          1.0002            1.0000         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               239.18       187.99       427.17       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               239.18       142.41       381.59       0.9985          1.0002            1.0000         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               239.18       200.00       439.17       0.9999          1.0000            1.0000         3.86
IVF-Binary-512-nl316-pca (self)                          239.18       354.91       594.08       0.9986          1.0002            1.0000         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)             223.85        96.87       320.72       0.6851          1.3494            1.2539         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)            223.85       113.00       336.85       0.6801          1.3803            1.2688         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)            223.85       128.65       352.50       0.6783          1.3989            1.2738         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)            223.85       161.18       385.03       0.9967          1.0007            1.0000         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)            223.85       216.46       440.31       0.9969          1.0007            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)           223.85       178.97       402.83       0.9996          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)           223.85       238.06       461.91       0.9999          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)           223.85       195.37       419.22       0.9996          1.0001            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)           223.85       257.57       481.42       1.0000          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-random (self)                      223.85       520.59       744.45       0.9994          1.0001            1.0000         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            221.95       100.95       322.89       0.6827          1.3604            1.2598         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            221.95       107.73       329.67       0.6806          1.3751            1.2657         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            221.95       124.10       346.05       0.6785          1.3958            1.2733         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           221.95       164.33       386.28       0.9991          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           221.95       221.49       443.44       0.9994          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           221.95       171.53       393.48       0.9995          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           221.95       231.10       453.05       0.9998          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           221.95       188.68       410.62       0.9996          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           221.95       250.55       472.50       0.9999          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-random (self)                      221.95       489.48       711.43       0.9994          1.0001            1.0000         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            242.65       102.61       345.26       0.6820          1.3626            1.2613         7.41
IVF-Binary-1024-nl316-np17-rf0-random (query)            242.65       106.43       349.08       0.6810          1.3706            1.2646         7.41
IVF-Binary-1024-nl316-np25-rf0-random (query)            242.65       120.93       363.57       0.6786          1.3927            1.2726         7.41
IVF-Binary-1024-nl316-np15-rf10-random (query)           242.65       170.65       413.29       0.9993          1.0001            1.0000         7.41
IVF-Binary-1024-nl316-np15-rf20-random (query)           242.65       229.29       471.94       0.9996          1.0001            1.0000         7.41
IVF-Binary-1024-nl316-np17-rf10-random (query)           242.65       167.26       409.90       0.9995          1.0001            1.0000         7.41
IVF-Binary-1024-nl316-np17-rf20-random (query)           242.65       227.11       469.75       0.9998          1.0000            1.0000         7.41
IVF-Binary-1024-nl316-np25-rf10-random (query)           242.65       181.00       423.64       0.9996          1.0000            1.0000         7.41
IVF-Binary-1024-nl316-np25-rf20-random (query)           242.65       244.37       487.02       0.9999          1.0000            1.0000         7.41
IVF-Binary-1024-nl316-random (self)                      242.65       471.82       714.47       0.9994          1.0001            1.0000         7.41
IVF-Binary-1024-nl158-np7-rf0-pca (query)                248.90        95.35       344.25       0.6924          1.3298            1.2421         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)               248.90       112.31       361.22       0.6878          1.3576            1.2545         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)               248.90       132.59       381.50       0.6861          1.3744            1.2584         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)               248.90       163.30       412.20       0.9968          1.0007            1.0000         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)               248.90       223.92       472.83       0.9969          1.0007            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)              248.90       179.11       428.01       0.9996          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)              248.90       238.74       487.64       0.9999          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)              248.90       201.90       450.80       0.9997          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)              248.90       258.01       506.91       1.0000          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-pca (self)                         248.90       541.02       789.92       0.9996          1.0001            1.0000         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               244.51       100.84       345.35       0.6901          1.3419            1.2466         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               244.51       107.87       352.38       0.6881          1.3546            1.2524         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               244.51       126.56       371.06       0.6861          1.3743            1.2579         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              244.51       164.25       408.76       0.9992          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              244.51       225.50       470.00       0.9994          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              244.51       170.98       415.48       0.9996          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              244.51       232.50       477.00       0.9998          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              244.51       191.97       436.48       0.9996          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              244.51       249.72       494.23       1.0000          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-pca (self)                         244.51       492.08       736.58       0.9995          1.0001            1.0000         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               271.62       103.53       375.15       0.6896          1.3424            1.2461         7.41
IVF-Binary-1024-nl316-np17-rf0-pca (query)               271.62       105.77       377.39       0.6887          1.3491            1.2492         7.41
IVF-Binary-1024-nl316-np25-rf0-pca (query)               271.62       119.82       391.44       0.6866          1.3683            1.2569         7.41
IVF-Binary-1024-nl316-np15-rf10-pca (query)              271.62       168.32       439.94       0.9994          1.0001            1.0000         7.41
IVF-Binary-1024-nl316-np15-rf20-pca (query)              271.62       224.36       495.98       0.9996          1.0001            1.0000         7.41
IVF-Binary-1024-nl316-np17-rf10-pca (query)              271.62       176.91       448.54       0.9996          1.0001            1.0000         7.41
IVF-Binary-1024-nl316-np17-rf20-pca (query)              271.62       228.30       499.92       0.9998          1.0000            1.0000         7.41
IVF-Binary-1024-nl316-np25-rf10-pca (query)              271.62       188.07       459.69       0.9997          1.0000            1.0000         7.41
IVF-Binary-1024-nl316-np25-rf20-pca (query)              271.62       243.95       515.57       1.0000          1.0000            1.0000         7.41
IVF-Binary-1024-nl316-pca (self)                         271.62       474.68       746.30       0.9995          1.0001            1.0000         7.41
IVF-Binary-256-nl158-np7-rf0-sign (query)                150.55       153.56       304.11       0.0714          6.2830            5.8232         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               150.55       169.76       320.31       0.0574          7.4019            6.8441         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               150.55       180.14       330.69       0.0522          8.2359            7.5704         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               150.55       189.13       339.68       0.4097          1.5739            1.4950         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               150.55       324.92       475.47       0.6655          1.2205            1.1598         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              150.55       205.32       355.87       0.3164          1.8077            1.7176         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              150.55       349.28       499.83       0.5033          1.4047            1.3351         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              150.55       215.11       365.66       0.2812          1.9430            1.8454         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              150.55       367.00       517.55       0.4271          1.5292            1.4556         1.68
IVF-Binary-256-nl158-sign (self)                         150.55       601.12       751.67       0.3242          1.7900            1.7014         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               171.08       155.85       326.93       0.0657          6.5478            6.0793         1.74
IVF-Binary-256-nl223-np14-rf0-sign (query)               171.08       159.04       330.12       0.0605          7.0103            6.4875         1.74
IVF-Binary-256-nl223-np21-rf0-sign (query)               171.08       172.37       343.45       0.0536          7.9113            7.3016         1.74
IVF-Binary-256-nl223-np11-rf10-sign (query)              171.08       190.01       361.09       0.3689          1.6592            1.5777         1.74
IVF-Binary-256-nl223-np11-rf20-sign (query)              171.08       327.53       498.61       0.5986          1.2821            1.2287         1.74
IVF-Binary-256-nl223-np14-rf10-sign (query)              171.08       196.91       368.00       0.3351          1.7485            1.6627         1.74
IVF-Binary-256-nl223-np14-rf20-sign (query)              171.08       337.35       508.44       0.5338          1.3563            1.2976         1.74
IVF-Binary-256-nl223-np21-rf10-sign (query)              171.08       207.47       378.56       0.2928          1.8998            1.8082         1.74
IVF-Binary-256-nl223-np21-rf20-sign (query)              171.08       360.62       531.70       0.4453          1.4917            1.4259         1.74
IVF-Binary-256-nl223-sign (self)                         171.08       566.65       737.73       0.3404          1.7398            1.6540         1.74
IVF-Binary-256-nl316-np15-rf0-sign (query)               181.86       152.57       334.42       0.0654          6.5521            6.1037         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               181.86       156.66       338.52       0.0628          6.7792            6.3025         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               181.86       164.64       346.49       0.0562          7.5430            6.9699         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              181.86       196.06       377.92       0.3629          1.6757            1.5919         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              181.86       330.65       512.51       0.5856          1.2996            1.2410         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              181.86       193.91       375.76       0.3461          1.7205            1.6321         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              181.86       333.37       515.23       0.5530          1.3369            1.2786         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              181.86       200.53       382.39       0.3042          1.8584            1.7663         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              181.86       349.30       531.16       0.4668          1.4566            1.3918         1.84
IVF-Binary-256-nl316-sign (self)                         181.86       550.73       732.58       0.3516          1.7088            1.6228         1.84
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 512D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        69.06     1_298.13     1_367.18       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.06     4_281.90     4_350.96       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                132.41       261.80       394.20       0.5547          1.7646            1.5366         2.03
ExhaustiveBinary-256-random-rf10 (query)                 132.41       401.63       534.04       0.9898          1.0017            1.0000         2.03
ExhaustiveBinary-256-random-rf20 (query)                 132.41       537.31       669.72       0.9985          1.0002            1.0000         2.03
ExhaustiveBinary-256-random (self)                       132.41     1_270.00     1_402.40       0.9899          1.0016            1.0000         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   251.04       280.14       531.17       0.5767          1.6243            1.4311         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    251.04       413.52       664.55       0.9904          1.0016            1.0000         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    251.04       555.90       806.94       0.9984          1.0002            1.0000         2.03
ExhaustiveBinary-256-pca (self)                          251.04     1_279.61     1_530.64       0.9905          1.0016            1.0000         2.03
ExhaustiveBinary-512-random_no_rr (query)                203.74       373.29       577.03       0.6013          1.6760            1.4608         4.05
ExhaustiveBinary-512-random-rf10 (query)                 203.74       524.59       728.33       0.9977          1.0003            1.0000         4.05
ExhaustiveBinary-512-random-rf20 (query)                 203.74       664.32       868.06       0.9998          1.0000            1.0000         4.05
ExhaustiveBinary-512-random (self)                       203.74     1_665.79     1_869.52       0.9975          1.0003            1.0000         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   293.43       378.22       671.65       0.6443          1.4426            1.3064         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    293.43       537.89       831.32       0.9985          1.0002            1.0000         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    293.43       667.93       961.36       0.9999          1.0000            1.0000         4.05
ExhaustiveBinary-512-pca (self)                          293.43     1_664.72     1_958.15       0.9984          1.0002            1.0000         4.05
ExhaustiveBinary-1024-random_no_rr (query)               255.81       559.62       815.43       0.6624          1.4553            1.3048         8.11
ExhaustiveBinary-1024-random-rf10 (query)                255.81       714.70       970.51       0.9995          1.0001            1.0000         8.11
ExhaustiveBinary-1024-random-rf20 (query)                255.81       863.26     1_119.07       1.0000          1.0000            1.0000         8.11
ExhaustiveBinary-1024-random (self)                      255.81     2_343.90     2_599.71       0.9994          1.0001            1.0000         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  343.25       563.70       906.96       0.6865          1.3603            1.2383         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   343.25       719.75     1_063.01       0.9996          1.0001            1.0000         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   343.25       867.26     1_210.51       1.0000          1.0000            1.0000         8.11
ExhaustiveBinary-1024-pca (self)                         343.25     2_345.43     2_688.68       0.9996          1.0001            1.0000         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   92.03       627.65       719.68       0.0400         18.1511           13.6734         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    92.03       671.70       763.73       0.1821          2.5573            2.4620         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    92.03     1_021.53     1_113.56       0.3140          1.8429            1.7786         3.05
ExhaustiveBinary-512-sign (self)                          92.03     2_176.30     2_268.33       0.1897          2.5286            2.4283         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)              311.53        82.64       394.16       0.5627          1.6389            1.4875         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)             311.53        90.85       402.38       0.5594          1.6737            1.5069         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)             311.53       101.39       412.92       0.5582          1.6954            1.5138         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)             311.53       167.12       478.65       0.9916          1.0013            1.0000         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)             311.53       259.15       570.68       0.9978          1.0004            1.0000         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)            311.53       174.56       486.09       0.9914          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)            311.53       270.63       582.16       0.9988          1.0001            1.0000         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)            311.53       182.39       493.92       0.9907          1.0015            1.0000         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)            311.53       271.38       582.90       0.9987          1.0002            1.0000         2.34
IVF-Binary-256-nl158-random (self)                       311.53       388.45       699.98       0.9915          1.0013            1.0000         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)             360.72        76.32       437.04       0.5614          1.6502            1.4940         2.46
IVF-Binary-256-nl223-np14-rf0-random (query)             360.72        80.84       441.55       0.5601          1.6630            1.5026         2.46
IVF-Binary-256-nl223-np21-rf0-random (query)             360.72        87.73       448.45       0.5587          1.6842            1.5117         2.46
IVF-Binary-256-nl223-np11-rf10-random (query)            360.72       168.38       529.10       0.9922          1.0012            1.0000         2.46
IVF-Binary-256-nl223-np11-rf20-random (query)            360.72       260.23       620.95       0.9988          1.0002            1.0000         2.46
IVF-Binary-256-nl223-np14-rf10-random (query)            360.72       168.81       529.53       0.9918          1.0013            1.0000         2.46
IVF-Binary-256-nl223-np14-rf20-random (query)            360.72       262.78       623.50       0.9989          1.0001            1.0000         2.46
IVF-Binary-256-nl223-np21-rf10-random (query)            360.72       175.62       536.34       0.9910          1.0014            1.0000         2.46
IVF-Binary-256-nl223-np21-rf20-random (query)            360.72       270.80       631.52       0.9988          1.0002            1.0000         2.46
IVF-Binary-256-nl223-random (self)                       360.72       379.30       740.02       0.9920          1.0012            1.0000         2.46
IVF-Binary-256-nl316-np15-rf0-random (query)             409.82        82.27       492.09       0.5614          1.6495            1.4999         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)             409.82        83.59       493.42       0.5610          1.6549            1.5018         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)             409.82        89.76       499.58       0.5597          1.6741            1.5100         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)            409.82       170.60       580.42       0.9924          1.0011            1.0000         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)            409.82       270.19       680.01       0.9991          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)            409.82       170.12       579.94       0.9922          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)            409.82       264.43       674.25       0.9990          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)            409.82       177.12       586.94       0.9914          1.0013            1.0000         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)            409.82       271.96       681.79       0.9988          1.0001            1.0000         2.65
IVF-Binary-256-nl316-random (self)                       409.82       376.70       786.52       0.9922          1.0012            1.0000         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)                 495.61        72.11       567.72       0.5835          1.5288            1.4040         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)                495.61        79.87       575.48       0.5807          1.5545            1.4163         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)                495.61        87.79       583.41       0.5798          1.5702            1.4204         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)                495.61       170.03       665.64       0.9915          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)                495.61       260.49       756.10       0.9976          1.0004            1.0000         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)               495.61       169.29       664.91       0.9915          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)               495.61       269.57       765.18       0.9987          1.0002            1.0000         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)               495.61       179.35       674.97       0.9910          1.0015            1.0000         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)               495.61       276.85       772.46       0.9986          1.0002            1.0000         2.34
IVF-Binary-256-nl158-pca (self)                          495.61       390.76       886.37       0.9915          1.0014            1.0000         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)                513.88        77.50       591.38       0.5829          1.5345            1.4098         2.46
IVF-Binary-256-nl223-np14-rf0-pca (query)                513.88        85.62       599.50       0.5819          1.5433            1.4140         2.46
IVF-Binary-256-nl223-np21-rf0-pca (query)                513.88        89.14       603.03       0.5807          1.5577            1.4197         2.46
IVF-Binary-256-nl223-np11-rf10-pca (query)               513.88       170.18       684.06       0.9924          1.0012            1.0000         2.46
IVF-Binary-256-nl223-np11-rf20-pca (query)               513.88       261.12       775.00       0.9988          1.0002            1.0000         2.46
IVF-Binary-256-nl223-np14-rf10-pca (query)               513.88       170.95       684.83       0.9919          1.0013            1.0000         2.46
IVF-Binary-256-nl223-np14-rf20-pca (query)               513.88       266.77       780.65       0.9989          1.0001            1.0000         2.46
IVF-Binary-256-nl223-np21-rf10-pca (query)               513.88       178.89       692.78       0.9913          1.0014            1.0000         2.46
IVF-Binary-256-nl223-np21-rf20-pca (query)               513.88       276.13       790.02       0.9987          1.0002            1.0000         2.46
IVF-Binary-256-nl223-pca (self)                          513.88       371.21       885.09       0.9921          1.0013            1.0000         2.46
IVF-Binary-256-nl316-np15-rf0-pca (query)                514.38        85.02       599.39       0.5828          1.5372            1.4101         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)                514.38        84.60       598.98       0.5823          1.5411            1.4120         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)                514.38        93.48       607.85       0.5813          1.5534            1.4174         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)               514.38       172.66       687.03       0.9923          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)               514.38       268.19       782.57       0.9989          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)               514.38       172.01       686.38       0.9922          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)               514.38       267.11       781.49       0.9989          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)               514.38       178.83       693.21       0.9915          1.0013            1.0000         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)               514.38       274.27       788.64       0.9988          1.0002            1.0000         2.65
IVF-Binary-256-nl316-pca (self)                          514.38       377.58       891.95       0.9923          1.0012            1.0000         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)              393.99        98.98       492.97       0.6087          1.5703            1.4231         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)             393.99       110.80       504.79       0.6050          1.6027            1.4384         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)             393.99       122.39       516.38       0.6037          1.6236            1.4455         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)             393.99       195.92       589.91       0.9971          1.0004            1.0000         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)             393.99       287.79       681.78       0.9984          1.0003            1.0000         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)            393.99       202.03       596.02       0.9981          1.0002            1.0000         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)            393.99       298.52       692.51       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)            393.99       211.01       605.00       0.9979          1.0003            1.0000         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)            393.99       309.32       703.31       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-random (self)                       393.99       489.78       883.77       0.9980          1.0003            1.0000         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)             393.89       105.27       499.16       0.6070          1.5829            1.4302         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)             393.89       111.69       505.58       0.6056          1.5947            1.4377         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)             393.89       120.78       514.68       0.6042          1.6130            1.4461         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)            393.89       195.64       589.53       0.9982          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)            393.89       290.84       684.73       0.9996          1.0001            1.0000         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)            393.89       203.08       596.98       0.9983          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)            393.89       294.13       688.03       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)            393.89       208.93       602.82       0.9980          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)            393.89       317.29       711.18       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-random (self)                       393.89       480.26       874.15       0.9982          1.0002            1.0000         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)             447.35       109.42       556.77       0.6066          1.5816            1.4306         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)             447.35       112.03       559.38       0.6060          1.5872            1.4332         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)             447.35       121.61       568.96       0.6046          1.6067            1.4417         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)            447.35       199.45       646.80       0.9984          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)            447.35       313.17       760.52       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)            447.35       200.31       647.66       0.9984          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)            447.35       304.80       752.15       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)            447.35       208.95       656.30       0.9982          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)            447.35       309.00       756.35       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-random (self)                       447.35       473.90       921.25       0.9983          1.0002            1.0000         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)                 476.46        96.64       573.10       0.6493          1.3882            1.2897         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)                476.46       110.51       586.97       0.6468          1.4055            1.2978         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)                476.46       120.12       596.58       0.6460          1.4168            1.3001         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)                476.46       189.31       665.77       0.9974          1.0004            1.0000         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)                476.46       280.53       756.99       0.9984          1.0003            1.0000         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)               476.46       201.40       677.86       0.9986          1.0002            1.0000         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)               476.46       303.02       779.48       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)               476.46       210.50       686.96       0.9985          1.0002            1.0000         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)               476.46       307.03       783.49       0.9999          1.0000            1.0000         4.36
IVF-Binary-512-nl158-pca (self)                          476.46       487.19       963.65       0.9986          1.0002            1.0000         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)                504.02       104.48       608.50       0.6485          1.3923            1.2910         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)                504.02       119.71       623.73       0.6475          1.3989            1.2951         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)                504.02       121.36       625.38       0.6463          1.4102            1.2992         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)               504.02       202.67       706.69       0.9986          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)               504.02       296.19       800.21       0.9996          1.0001            1.0000         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)               504.02       204.14       708.16       0.9987          1.0001            1.0000         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)               504.02       296.93       800.95       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)               504.02       208.59       712.61       0.9986          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)               504.02       308.36       812.38       0.9999          1.0000            1.0000         4.49
IVF-Binary-512-nl223-pca (self)                          504.02       504.13     1_008.15       0.9987          1.0002            1.0000         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)                520.52       109.75       630.27       0.6479          1.3962            1.2944         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)                520.52       112.51       633.04       0.6475          1.3988            1.2956         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)                520.52       120.09       640.61       0.6465          1.4082            1.2988         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)               520.52       197.48       718.00       0.9987          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)               520.52       292.88       813.40       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)               520.52       199.52       720.04       0.9987          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)               520.52       298.51       819.03       0.9999          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)               520.52       218.56       739.08       0.9986          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)               520.52       315.44       835.96       0.9999          1.0000            1.0000         4.67
IVF-Binary-512-nl316-pca (self)                          520.52       471.55       992.07       0.9987          1.0001            1.0000         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)             427.80       146.54       574.34       0.6684          1.3892            1.2847         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)            427.80       166.09       593.89       0.6651          1.4111            1.2943         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)            427.80       178.09       605.89       0.6640          1.4256            1.2986         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)            427.80       242.15       669.95       0.9982          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)            427.80       340.42       768.22       0.9985          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)           427.80       260.54       688.34       0.9995          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)           427.80       365.53       793.33       0.9999          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)           427.80       275.97       703.76       0.9995          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)           427.80       381.43       809.23       0.9999          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-random (self)                      427.80       705.15     1_132.95       0.9995          1.0001            1.0000         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)            436.80       158.07       594.87       0.6665          1.4019            1.2893         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)            436.80       161.32       598.12       0.6653          1.4101            1.2935         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)            436.80       178.46       615.26       0.6639          1.4229            1.2995         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)           436.80       254.80       691.60       0.9994          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)           436.80       354.85       791.64       0.9997          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)           436.80       262.98       699.78       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)           436.80       367.19       803.99       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)           436.80       278.78       715.58       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)           436.80       386.83       823.63       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-random (self)                      436.80       683.10     1_119.90       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)            486.09       162.75       648.84       0.6661          1.4029            1.2911         8.72
IVF-Binary-1024-nl316-np17-rf0-random (query)            486.09       166.12       652.21       0.6658          1.4056            1.2923         8.72
IVF-Binary-1024-nl316-np25-rf0-random (query)            486.09       177.27       663.37       0.6646          1.4176            1.2976         8.72
IVF-Binary-1024-nl316-np15-rf10-random (query)           486.09       256.98       743.07       0.9996          1.0001            1.0000         8.72
IVF-Binary-1024-nl316-np15-rf20-random (query)           486.09       362.79       848.89       0.9999          1.0000            1.0000         8.72
IVF-Binary-1024-nl316-np17-rf10-random (query)           486.09       259.33       745.43       0.9996          1.0001            1.0000         8.72
IVF-Binary-1024-nl316-np17-rf20-random (query)           486.09       370.99       857.09       0.9999          1.0000            1.0000         8.72
IVF-Binary-1024-nl316-np25-rf10-random (query)           486.09       276.71       762.80       0.9996          1.0001            1.0000         8.72
IVF-Binary-1024-nl316-np25-rf20-random (query)           486.09       385.65       871.74       1.0000          1.0000            1.0000         8.72
IVF-Binary-1024-nl316-random (self)                      486.09       726.69     1_212.78       0.9996          1.0001            1.0000         8.72
IVF-Binary-1024-nl158-np7-rf0-pca (query)                520.48       147.89       668.37       0.6912          1.3164            1.2248         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)               520.48       164.20       684.68       0.6885          1.3324            1.2317         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)               520.48       178.93       699.41       0.6877          1.3427            1.2341         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)               520.48       238.82       759.30       0.9983          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)               520.48       336.54       857.02       0.9985          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)              520.48       254.89       775.37       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)              520.48       360.14       880.63       0.9999          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)              520.48       275.02       795.50       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)              520.48       377.11       897.60       1.0000          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-pca (self)                         520.48       699.76     1_220.24       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)               524.02       155.32       679.34       0.6899          1.3227            1.2295         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)               524.02       162.18       686.20       0.6889          1.3281            1.2312         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)               524.02       177.71       701.73       0.6878          1.3378            1.2349         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)              524.02       246.36       770.38       0.9994          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)              524.02       347.94       871.96       0.9997          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)              524.02       252.74       776.76       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)              524.02       359.67       883.69       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)              524.02       274.60       798.62       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)              524.02       379.58       903.60       1.0000          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-pca (self)                         524.02       678.14     1_202.16       0.9996          1.0000            1.0000         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)               581.16       161.99       743.15       0.6894          1.3254            1.2310         8.72
IVF-Binary-1024-nl316-np17-rf0-pca (query)               581.16       165.60       746.76       0.6891          1.3273            1.2321         8.72
IVF-Binary-1024-nl316-np25-rf0-pca (query)               581.16       176.48       757.64       0.6881          1.3357            1.2348         8.72
IVF-Binary-1024-nl316-np15-rf10-pca (query)              581.16       251.24       832.40       0.9996          1.0001            1.0000         8.72
IVF-Binary-1024-nl316-np15-rf20-pca (query)              581.16       356.00       937.17       0.9999          1.0000            1.0000         8.72
IVF-Binary-1024-nl316-np17-rf10-pca (query)              581.16       253.86       835.02       0.9997          1.0000            1.0000         8.72
IVF-Binary-1024-nl316-np17-rf20-pca (query)              581.16       361.72       942.88       0.9999          1.0000            1.0000         8.72
IVF-Binary-1024-nl316-np25-rf10-pca (query)              581.16       268.28       849.45       0.9997          1.0001            1.0000         8.72
IVF-Binary-1024-nl316-np25-rf20-pca (query)              581.16       382.50       963.67       1.0000          1.0000            1.0000         8.72
IVF-Binary-1024-nl316-pca (self)                         581.16       672.95     1_254.11       0.9996          1.0000            1.0000         8.72
IVF-Binary-512-nl158-np7-rf0-sign (query)                257.54       253.03       510.56       0.0599          7.6431            7.0595         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)               257.54       268.77       526.30       0.0527          8.6921            7.8268         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)               257.54       279.79       537.33       0.0494          9.5135            8.3808         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)               257.54       314.13       571.66       0.3259          1.8213            1.7221         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)               257.54       560.54       818.07       0.5207          1.3937            1.3301         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)              257.54       329.47       587.00       0.2832          1.9691            1.8563         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)              257.54       581.41       838.95       0.4455          1.5151            1.4405         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)              257.54       342.68       600.22       0.2616          2.0675            1.9420         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)              257.54       604.34       861.88       0.4061          1.5911            1.5152         3.36
IVF-Binary-512-nl158-sign (self)                         257.54       925.64     1_183.17       0.2912          1.9456            1.8280         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               269.37       258.06       527.42       0.0577          7.8281            7.1585         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               269.37       262.69       532.05       0.0546          8.3001            7.5057         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               269.37       276.94       546.31       0.0505          9.1439            8.1402         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              269.37       322.37       591.73       0.3147          1.8520            1.7431         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              269.37       566.48       835.85       0.5027          1.4125            1.3453         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              269.37       329.80       599.16       0.2947          1.9188            1.8045         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              269.37       576.95       846.31       0.4678          1.4692            1.3975         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              269.37       342.02       611.38       0.2692          2.0240            1.8933         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              269.37       597.88       867.24       0.4177          1.5636            1.4849         3.49
IVF-Binary-512-nl223-sign (self)                         269.37       907.59     1_176.96       0.3024          1.8948            1.7792         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)               317.84       260.18       578.02       0.0577          7.7120            7.0592         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)               317.84       264.09       581.93       0.0559          7.9326            7.2580         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)               317.84       294.84       612.68       0.0521          8.6287            7.8189         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)              317.84       325.04       642.88       0.3158          1.8398            1.7379         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)              317.84       574.64       892.48       0.5020          1.4135            1.3477         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)              317.84       333.05       650.89       0.3056          1.8727            1.7707         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)              317.84       583.19       901.03       0.4830          1.4414            1.3763         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)              317.84       343.48       661.31       0.2803          1.9712            1.8511         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)              317.84       598.69       916.53       0.4348          1.5263            1.4540         3.67
IVF-Binary-512-nl316-sign (self)                         317.84       915.66     1_233.50       0.3123          1.8552            1.7464         3.67
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 768D - Binary Quantisation
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       101.61     2_171.54     2_273.14       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.61     6_297.33     6_398.93       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                207.50       287.85       495.35       0.5361          1.8068            1.5908         2.28
ExhaustiveBinary-256-random-rf10 (query)                 207.50       439.24       646.74       0.9868          1.0022            1.0000         2.28
ExhaustiveBinary-256-random-rf20 (query)                 207.50       588.52       796.02       0.9980          1.0003            1.0000         2.28
ExhaustiveBinary-256-random (self)                       207.50     1_365.81     1_573.31       0.9876          1.0021            1.0000         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   394.28       287.04       681.33       0.5754          1.5495            1.4128         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    394.28       441.04       835.32       0.9895          1.0018            1.0000         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    394.28       587.57       981.85       0.9983          1.0002            1.0000         2.28
ExhaustiveBinary-256-pca (self)                          394.28     1_368.82     1_763.11       0.9897          1.0017            1.0000         2.28
ExhaustiveBinary-512-random_no_rr (query)                297.59       399.22       696.81       0.5866          1.6778            1.4946         4.55
ExhaustiveBinary-512-random-rf10 (query)                 297.59       564.08       861.66       0.9966          1.0005            1.0000         4.55
ExhaustiveBinary-512-random-rf20 (query)                 297.59       731.23     1_028.82       0.9997          1.0001            1.0000         4.55
ExhaustiveBinary-512-random (self)                       297.59     1_810.65     2_108.24       0.9969          1.0004            1.0000         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   491.66       399.75       891.41       0.6388          1.4217            1.3032         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    491.66       556.81     1_048.48       0.9979          1.0003            1.0000         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    491.66       723.55     1_215.21       0.9998          1.0000            1.0000         4.55
ExhaustiveBinary-512-pca (self)                          491.66     1_784.39     2_276.06       0.9981          1.0002            1.0000         4.55
ExhaustiveBinary-1024-random_no_rr (query)               496.84       607.88     1_104.72       0.6446          1.4909            1.3512         9.11
ExhaustiveBinary-1024-random-rf10 (query)                496.84       787.86     1_284.70       0.9993          1.0001            1.0000         9.11
ExhaustiveBinary-1024-random-rf20 (query)                496.84       963.70     1_460.54       0.9999          1.0000            1.0000         9.11
ExhaustiveBinary-1024-random (self)                      496.84     2_568.98     3_065.82       0.9994          1.0001            1.0000         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  700.06       620.54     1_320.60       0.6795          1.3452            1.2483         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   700.06       790.11     1_490.17       0.9996          1.0001            1.0000         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   700.06       968.05     1_668.11       1.0000          1.0000            1.0000         9.11
ExhaustiveBinary-1024-pca (self)                         700.06     2_582.57     3_282.63       0.9997          1.0000            1.0000         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  127.82       792.92       920.74       0.0420         17.7082           13.0970         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   127.82       860.60       988.42       0.1896          2.5240            2.4052         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   127.82     1_327.05     1_454.87       0.3229          1.8300            1.7348         4.58
ExhaustiveBinary-768-sign (self)                         127.82     2_788.79     2_916.61       0.1997          2.4832            2.3546         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)              431.77        99.77       531.53       0.5425          1.7134            1.5513         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)             431.77       108.45       540.22       0.5405          1.7376            1.5631         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)             431.77       116.55       548.31       0.5397          1.7555            1.5696         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)             431.77       204.85       636.62       0.9889          1.0019            1.0000         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)             431.77       311.63       743.40       0.9984          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)            431.77       205.71       637.48       0.9881          1.0020            1.0000         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)            431.77       322.32       754.08       0.9985          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)            431.77       209.73       641.50       0.9877          1.0021            1.0000         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)            431.77       326.98       758.75       0.9983          1.0002            1.0000         2.74
IVF-Binary-256-nl158-random (self)                       431.77       457.72       889.49       0.9889          1.0019            1.0000         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)             451.43        97.14       548.58       0.5422          1.7157            1.5515         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)             451.43       100.92       552.35       0.5414          1.7255            1.5576         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)             451.43       107.00       558.43       0.5404          1.7477            1.5654         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)            451.43       204.93       656.37       0.9890          1.0018            1.0000         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)            451.43       317.30       768.73       0.9986          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)            451.43       210.91       662.34       0.9885          1.0019            1.0000         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)            451.43       323.72       775.16       0.9985          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)            451.43       216.00       667.44       0.9878          1.0021            1.0000         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)            451.43       333.40       784.83       0.9983          1.0002            1.0000         2.93
IVF-Binary-256-nl223-random (self)                       451.43       465.60       917.03       0.9892          1.0018            1.0000         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)             522.21       104.74       626.95       0.5421          1.7133            1.5522         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)             522.21       106.33       628.55       0.5417          1.7180            1.5546         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)             522.21       118.75       640.96       0.5407          1.7339            1.5610         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)            522.21       226.98       749.20       0.9892          1.0018            1.0000         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)            522.21       330.55       852.77       0.9987          1.0002            1.0000         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)            522.21       211.90       734.11       0.9889          1.0018            1.0000         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)            522.21       329.16       851.38       0.9987          1.0002            1.0000         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)            522.21       217.59       739.80       0.9882          1.0020            1.0000         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)            522.21       341.51       863.72       0.9985          1.0002            1.0000         3.21
IVF-Binary-256-nl316-random (self)                       522.21       473.39       995.61       0.9895          1.0017            1.0000         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)                 630.55        86.91       717.46       0.5806          1.5007            1.3950         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)                630.55        95.48       726.03       0.5789          1.5133            1.4012         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)                630.55       103.31       733.86       0.5783          1.5210            1.4039         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)                630.55       197.17       827.72       0.9909          1.0015            1.0000         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)                630.55       311.10       941.66       0.9984          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)               630.55       201.64       832.20       0.9904          1.0016            1.0000         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)               630.55       318.97       949.52       0.9986          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)               630.55       209.20       839.75       0.9901          1.0017            1.0000         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)               630.55       325.84       956.40       0.9985          1.0002            1.0000         2.74
IVF-Binary-256-nl158-pca (self)                          630.55       457.73     1_088.29       0.9908          1.0014            1.0000         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)                641.86        96.83       738.68       0.5803          1.5034            1.3951         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)                641.86       100.55       742.40       0.5797          1.5083            1.3974         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)                641.86       107.82       749.68       0.5789          1.5174            1.3997         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)               641.86       206.53       848.39       0.9910          1.0015            1.0000         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)               641.86       320.98       962.84       0.9987          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)               641.86       207.35       849.21       0.9906          1.0015            1.0000         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)               641.86       322.16       964.02       0.9987          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)               641.86       213.63       855.48       0.9902          1.0016            1.0000         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)               641.86       332.81       974.66       0.9985          1.0002            1.0000         2.93
IVF-Binary-256-nl223-pca (self)                          641.86       483.47     1_125.33       0.9910          1.0014            1.0000         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)                733.15       106.66       839.81       0.5801          1.5018            1.3952         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)                733.15       106.20       839.34       0.5799          1.5047            1.3966         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)                733.15       113.77       846.92       0.5792          1.5123            1.3987         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)               733.15       213.40       946.55       0.9912          1.0014            1.0000         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)               733.15       336.11     1_069.25       0.9988          1.0001            1.0000         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)               733.15       212.90       946.04       0.9910          1.0014            1.0000         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)               733.15       336.99     1_070.14       0.9988          1.0001            1.0000         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)               733.15       222.38       955.53       0.9903          1.0016            1.0000         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)               733.15       338.63     1_071.77       0.9986          1.0002            1.0000         3.21
IVF-Binary-256-nl316-pca (self)                          733.15       473.97     1_207.11       0.9914          1.0013            1.0000         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)              552.91       126.48       679.39       0.5921          1.6039            1.4642         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)             552.91       135.91       688.81       0.5899          1.6250            1.4749         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)             552.91       146.37       699.27       0.5890          1.6409            1.4812         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)             552.91       233.56       786.47       0.9971          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)             552.91       350.04       902.95       0.9993          1.0001            1.0000         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)            552.91       243.46       796.37       0.9971          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)            552.91       367.23       920.14       0.9997          1.0001            1.0000         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)            552.91       256.31       809.21       0.9969          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)            552.91       378.27       931.18       0.9997          1.0000            1.0000         5.02
IVF-Binary-512-nl158-random (self)                       552.91       607.86     1_160.76       0.9973          1.0004            1.0000         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)             562.55       133.84       696.39       0.5911          1.6103            1.4676         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)             562.55       138.30       700.86       0.5902          1.6197            1.4730         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)             562.55       151.81       714.36       0.5889          1.6399            1.4803         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)            562.55       251.14       813.69       0.9973          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)            562.55       373.47       936.02       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)            562.55       245.49       808.04       0.9972          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)            562.55       372.38       934.94       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)            562.55       260.56       823.11       0.9970          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)            562.55       383.10       945.65       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-random (self)                       562.55       614.24     1_176.80       0.9974          1.0003            1.0000         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)             630.85       142.92       773.77       0.5912          1.6086            1.4684         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)             630.85       143.88       774.73       0.5909          1.6125            1.4703         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)             630.85       152.63       783.48       0.5898          1.6268            1.4764         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)            630.85       251.31       882.16       0.9974          1.0003            1.0000         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)            630.85       378.80     1_009.65       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)            630.85       256.46       887.31       0.9974          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)            630.85       384.13     1_014.98       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)            630.85       264.83       895.68       0.9971          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)            630.85       395.35     1_026.20       0.9997          1.0000            1.0000         5.48
IVF-Binary-512-nl316-random (self)                       630.85       619.87     1_250.72       0.9976          1.0003            1.0000         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)                 732.00       123.58       855.58       0.6426          1.3861            1.2902         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)                732.00       135.81       867.81       0.6412          1.3979            1.2952         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)                732.00       150.78       882.78       0.6405          1.4059            1.2980         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)                732.00       234.81       966.81       0.9980          1.0003            1.0000         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)                732.00       350.95     1_082.95       0.9994          1.0001            1.0000         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)               732.00       243.07       975.08       0.9981          1.0002            1.0000         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)               732.00       366.37     1_098.37       0.9998          1.0000            1.0000         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)               732.00       258.83       990.83       0.9981          1.0003            1.0000         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)               732.00       385.99     1_117.99       0.9998          1.0000            1.0000         5.02
IVF-Binary-512-nl158-pca (self)                          732.00       610.49     1_342.49       0.9983          1.0002            1.0000         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)                750.29       134.47       884.76       0.6422          1.3884            1.2919         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)                750.29       138.48       888.77       0.6415          1.3935            1.2938         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)                750.29       149.42       899.71       0.6407          1.4024            1.2975         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)               750.29       243.75       994.05       0.9983          1.0002            1.0000         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)               750.29       364.82     1_115.11       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)               750.29       252.05     1_002.34       0.9983          1.0002            1.0000         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)               750.29       372.44     1_122.73       0.9998          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)               750.29       258.61     1_008.91       0.9981          1.0003            1.0000         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)               750.29       386.56     1_136.86       0.9998          1.0000            1.0000         5.21
IVF-Binary-512-nl223-pca (self)                          750.29       607.97     1_358.26       0.9984          1.0002            1.0000         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)                823.01       142.11       965.12       0.6423          1.3903            1.2927         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)                823.01       147.59       970.60       0.6420          1.3925            1.2937         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)                823.01       155.25       978.26       0.6414          1.3991            1.2961         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)               823.01       251.09     1_074.10       0.9984          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)               823.01       375.22     1_198.23       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)               823.01       250.93     1_073.94       0.9983          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)               823.01       375.14     1_198.15       0.9999          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)               823.01       263.43     1_086.43       0.9982          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)               823.01       389.44     1_212.45       0.9999          1.0000            1.0000         5.48
IVF-Binary-512-nl316-pca (self)                          823.01       622.74     1_445.75       0.9985          1.0002            1.0000         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)             747.47       196.08       943.55       0.6487          1.4430            1.3327         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)            747.47       212.90       960.37       0.6467          1.4592            1.3411         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)            747.47       228.24       975.71       0.6458          1.4715            1.3457         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)            747.47       320.02     1_067.48       0.9991          1.0002            1.0000         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)            747.47       449.82     1_197.28       0.9994          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)           747.47       344.95     1_092.42       0.9994          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)           747.47       476.69     1_224.15       0.9999          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)           747.47       362.97     1_110.44       0.9994          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)           747.47       497.84     1_245.31       0.9999          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-random (self)                      747.47       921.15     1_668.62       0.9995          1.0001            1.0000         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)            753.19       208.95       962.14       0.6479          1.4492            1.3354         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)            753.19       213.50       966.69       0.6470          1.4558            1.3394         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)            753.19       230.29       983.48       0.6459          1.4693            1.3443         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)           753.19       336.30     1_089.49       0.9993          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)           753.19       472.36     1_225.55       0.9998          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)           753.19       342.29     1_095.48       0.9994          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)           753.19       476.73     1_229.92       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)           753.19       362.62     1_115.81       0.9994          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)           753.19       502.03     1_255.22       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-random (self)                      753.19       913.49     1_666.68       0.9995          1.0001            1.0000         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)            827.74       221.06     1_048.80       0.6477          1.4477            1.3364        10.03
IVF-Binary-1024-nl316-np17-rf0-random (query)            827.74       219.46     1_047.21       0.6473          1.4509            1.3386        10.03
IVF-Binary-1024-nl316-np25-rf0-random (query)            827.74       239.12     1_066.87       0.6464          1.4612            1.3420        10.03
IVF-Binary-1024-nl316-np15-rf10-random (query)           827.74       358.67     1_186.41       0.9995          1.0001            1.0000        10.03
IVF-Binary-1024-nl316-np15-rf20-random (query)           827.74       484.23     1_311.97       0.9999          1.0000            1.0000        10.03
IVF-Binary-1024-nl316-np17-rf10-random (query)           827.74       360.43     1_188.17       0.9995          1.0001            1.0000        10.03
IVF-Binary-1024-nl316-np17-rf20-random (query)           827.74       485.07     1_312.81       0.9999          1.0000            1.0000        10.03
IVF-Binary-1024-nl316-np25-rf10-random (query)           827.74       371.98     1_199.72       0.9994          1.0001            1.0000        10.03
IVF-Binary-1024-nl316-np25-rf20-random (query)           827.74       504.47     1_332.22       0.9999          1.0000            1.0000        10.03
IVF-Binary-1024-nl316-random (self)                      827.74       933.10     1_760.85       0.9995          1.0001            1.0000        10.03
IVF-Binary-1024-nl158-np7-rf0-pca (query)                947.59       199.51     1_147.10       0.6824          1.3209            1.2391         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)               947.59       214.10     1_161.69       0.6809          1.3308            1.2440         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)               947.59       229.26     1_176.85       0.6803          1.3366            1.2458         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)               947.59       319.47     1_267.06       0.9992          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)               947.59       446.57     1_394.16       0.9995          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)              947.59       340.29     1_287.88       0.9996          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)              947.59       477.71     1_425.30       0.9999          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)              947.59       357.71     1_305.30       0.9996          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)              947.59       495.08     1_442.66       1.0000          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-pca (self)                         947.59       926.18     1_873.77       0.9997          1.0000            1.0000         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)               961.92       209.93     1_171.86       0.6818          1.3245            1.2403         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)               961.92       217.66     1_179.59       0.6812          1.3284            1.2421         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)               961.92       238.77     1_200.69       0.6805          1.3347            1.2444         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)              961.92       327.91     1_289.84       0.9996          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)              961.92       464.68     1_426.60       0.9998          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)              961.92       336.49     1_298.41       0.9996          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)              961.92       468.23     1_430.16       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)              961.92       353.94     1_315.86       0.9996          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)              961.92       493.06     1_454.98       1.0000          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-pca (self)                         961.92       906.41     1_868.33       0.9997          1.0000            1.0000         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_026.73       219.99     1_246.73       0.6816          1.3253            1.2414        10.03
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_026.73       220.17     1_246.90       0.6814          1.3274            1.2423        10.03
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_026.73       233.60     1_260.34       0.6807          1.3324            1.2441        10.03
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_026.73       342.53     1_369.26       0.9996          1.0000            1.0000        10.03
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_026.73       502.17     1_528.90       0.9999          1.0000            1.0000        10.03
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_026.73       345.03     1_371.76       0.9997          1.0000            1.0000        10.03
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_026.73       483.30     1_510.03       1.0000          1.0000            1.0000        10.03
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_026.73       365.33     1_392.07       0.9996          1.0000            1.0000        10.03
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_026.73       522.55     1_549.29       1.0000          1.0000            1.0000        10.03
IVF-Binary-1024-nl316-pca (self)                       1_026.73       926.25     1_952.98       0.9997          1.0000            1.0000        10.03
IVF-Binary-768-nl158-np7-rf0-sign (query)                377.67       364.61       742.28       0.0593          7.9762            7.2841         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)               377.67       382.23       759.89       0.0529          9.1876            8.0626         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)               377.67       399.58       777.25       0.0503          9.9604            8.5963         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)               377.67       448.94       826.60       0.3168          1.8677            1.7539         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)               377.67       806.19     1_183.86       0.4883          1.4612            1.3629         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)              377.67       463.28       840.95       0.2816          1.9851            1.8597         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)              377.67       832.51     1_210.18       0.4329          1.5525            1.4516         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)              377.67       479.12       856.79       0.2658          2.0515            1.9190         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)              377.67       850.80     1_228.47       0.4072          1.6034            1.4969         5.04
IVF-Binary-768-nl158-sign (self)                         377.67     1_319.75     1_697.42       0.2937          1.9481            1.8252         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)               387.93       374.41       762.34       0.0576          8.2518            7.3733         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)               387.93       384.53       772.46       0.0551          8.6834            7.6674         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)               387.93       395.69       783.62       0.0516          9.5662            8.2933         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)              387.93       459.70       847.63       0.3089          1.8833            1.7710         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)              387.93       822.98     1_210.91       0.4712          1.4806            1.3939         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)              387.93       464.94       852.87       0.2945          1.9305            1.8092         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)              387.93       832.93     1_220.86       0.4486          1.5152            1.4266         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)              387.93       482.35       870.28       0.2746          2.0094            1.8730         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)              387.93       852.34     1_240.28       0.4160          1.5734            1.4836         5.23
IVF-Binary-768-nl223-sign (self)                         387.93     1_316.22     1_704.15       0.3066          1.8971            1.7753         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)               456.26       381.86       838.11       0.0578          8.0396            7.2472         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)               456.26       386.63       842.88       0.0567          8.2545            7.4105         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)               456.26       403.03       859.29       0.0535          8.9914            7.9034         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)              456.26       469.36       925.61       0.3154          1.8548            1.7481         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)              456.26       835.25     1_291.50       0.4766          1.4696            1.3796         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)              456.26       472.53       928.79       0.3078          1.8784            1.7688         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)              456.26       840.47     1_296.72       0.4650          1.4877            1.3979         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)              456.26       486.33       942.58       0.2871          1.9530            1.8316         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)              456.26       860.68     1_316.93       0.4311          1.5454            1.4535         5.51
IVF-Binary-768-nl316-sign (self)                         456.26     1_339.78     1_796.04       0.3198          1.8447            1.7339         5.51
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
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 256D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        33.16       675.76       708.93       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.16     2_236.75     2_269.91       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             235.04        61.66       296.70       0.4587          1.0613            1.0633         2.56
ExhaustiveRaBitQ-rf5 (query)                             235.04       110.60       345.63       0.8093          1.0067            1.0054         2.56
ExhaustiveRaBitQ-rf10 (query)                            235.04       159.57       394.61       0.9208          1.0021            1.0006         2.56
ExhaustiveRaBitQ-rf20 (query)                            235.04       240.67       475.71       0.9802          1.0004            1.0000         2.56
ExhaustiveRaBitQ-rfB (query)                             235.04       153.31       388.35       0.9787          1.0005            1.0000         2.56
ExhaustiveRaBitQ (self)                                  235.04       500.89       735.93       0.9199          1.0022            1.0007         2.56
IVF-RaBitQ-nl158-np7-rf0 (query)                         170.85        47.44       218.29       0.4710          1.0577            1.0592         2.67
IVF-RaBitQ-nl158-np12-rf0 (query)                        170.85        48.58       219.43       0.4709          1.0577            1.0592         2.67
IVF-RaBitQ-nl158-np17-rf0 (query)                        170.85        50.82       221.67       0.4709          1.0577            1.0592         2.67
IVF-RaBitQ-nl158-np7-rf10 (query)                        170.85       121.04       291.89       0.9288          1.0018            1.0003         2.67
IVF-RaBitQ-nl158-np7-rf20 (query)                        170.85       195.84       366.69       0.9834          1.0004            1.0000         2.67
IVF-RaBitQ-nl158-np7-rfB (query)                         170.85       108.88       279.73       0.9783          1.0005            1.0000         2.67
IVF-RaBitQ-nl158-np12-rf10 (query)                       170.85       126.06       296.91       0.9288          1.0018            1.0003         2.67
IVF-RaBitQ-nl158-np12-rf20 (query)                       170.85       190.21       361.05       0.9834          1.0004            1.0000         2.67
IVF-RaBitQ-nl158-np12-rfB (query)                        170.85       110.81       281.66       0.9783          1.0005            1.0000         2.67
IVF-RaBitQ-nl158-np17-rf10 (query)                       170.85       125.17       296.02       0.9288          1.0018            1.0003         2.67
IVF-RaBitQ-nl158-np17-rf20 (query)                       170.85       193.07       363.92       0.9834          1.0004            1.0000         2.67
IVF-RaBitQ-nl158-np17-rfB (query)                        170.85       114.34       285.19       0.9783          1.0005            1.0000         2.67
IVF-RaBitQ-nl158 (self)                                  170.85       341.16       512.01       0.9782          1.0005            1.0000         2.67
IVF-RaBitQ-nl223-np11-rf0 (query)                        194.26        49.79       244.05       0.4803          1.0550            1.0574         2.83
IVF-RaBitQ-nl223-np14-rf0 (query)                        194.26        57.78       252.04       0.4803          1.0550            1.0574         2.83
IVF-RaBitQ-nl223-np21-rf0 (query)                        194.26        54.97       249.23       0.4803          1.0550            1.0574         2.83
IVF-RaBitQ-nl223-np11-rf10 (query)                       194.26       124.10       318.36       0.9340          1.0017            1.0001         2.83
IVF-RaBitQ-nl223-np11-rf20 (query)                       194.26       188.03       382.29       0.9847          1.0003            1.0000         2.83
IVF-RaBitQ-nl223-np11-rfB (query)                        194.26       107.00       301.26       0.9783          1.0005            1.0000         2.83
IVF-RaBitQ-nl223-np14-rf10 (query)                       194.26       125.36       319.62       0.9340          1.0017            1.0001         2.83
IVF-RaBitQ-nl223-np14-rf20 (query)                       194.26       191.18       385.44       0.9847          1.0003            1.0000         2.83
IVF-RaBitQ-nl223-np14-rfB (query)                        194.26       108.86       303.12       0.9783          1.0005            1.0000         2.83
IVF-RaBitQ-nl223-np21-rf10 (query)                       194.26       128.86       323.12       0.9340          1.0017            1.0001         2.83
IVF-RaBitQ-nl223-np21-rf20 (query)                       194.26       193.70       387.96       0.9847          1.0003            1.0000         2.83
IVF-RaBitQ-nl223-np21-rfB (query)                        194.26       112.42       306.68       0.9783          1.0005            1.0000         2.83
IVF-RaBitQ-nl223 (self)                                  194.26       337.00       531.26       0.9781          1.0005            1.0000         2.83
IVF-RaBitQ-nl316-np15-rf0 (query)                        210.61        54.42       265.03       0.4874          1.0531            1.0555         3.05
IVF-RaBitQ-nl316-np17-rf0 (query)                        210.61        52.33       262.95       0.4874          1.0531            1.0555         3.05
IVF-RaBitQ-nl316-np25-rf0 (query)                        210.61        56.55       267.16       0.4874          1.0531            1.0555         3.05
IVF-RaBitQ-nl316-np15-rf10 (query)                       210.61       125.14       335.75       0.9356          1.0016            1.0001         3.05
IVF-RaBitQ-nl316-np15-rf20 (query)                       210.61       190.72       401.34       0.9853          1.0003            1.0000         3.05
IVF-RaBitQ-nl316-np15-rfB (query)                        210.61       105.65       316.27       0.9779          1.0005            1.0000         3.05
IVF-RaBitQ-nl316-np17-rf10 (query)                       210.61       123.58       334.19       0.9356          1.0016            1.0001         3.05
IVF-RaBitQ-nl316-np17-rf20 (query)                       210.61       187.76       398.38       0.9853          1.0003            1.0000         3.05
IVF-RaBitQ-nl316-np17-rfB (query)                        210.61       106.64       317.25       0.9779          1.0005            1.0000         3.05
IVF-RaBitQ-nl316-np25-rf10 (query)                       210.61       127.08       337.70       0.9356          1.0016            1.0001         3.05
IVF-RaBitQ-nl316-np25-rf20 (query)                       210.61       194.43       405.04       0.9853          1.0003            1.0000         3.05
IVF-RaBitQ-nl316-np25-rfB (query)                        210.61       110.20       320.82       0.9779          1.0005            1.0000         3.05
IVF-RaBitQ-nl316 (self)                                  210.61       331.04       541.65       0.9781          1.0005            1.0000         3.05
IVF-RaBitQ-nl223-ex2-np14 (query)                        215.37       117.11       332.48       0.8667          1.0029            1.0025         6.26
IVF-RaBitQ-nl223-ex2-np21 (query)                        215.37       119.53       334.90       0.8667          1.0029            1.0025         6.26
IVF-RaBitQ-nl223-ex4-np14 (query)                        215.20       131.32       346.52       0.9451          1.0007            1.0002         9.32
IVF-RaBitQ-nl223-ex4-np21 (query)                        215.20       135.39       350.60       0.9451          1.0007            1.0002         9.32
IVF-RaBitQ-nl223-ex8-np14 (query)                        245.26       119.36       364.62       0.9713          1.0005            1.0000        15.42
IVF-RaBitQ-nl223-ex8-np21 (query)                        245.26       121.46       366.72       0.9713          1.0005            1.0000        15.42
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 512D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        69.48     1_265.01     1_334.49       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.48     4_274.12     4_343.60       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                             372.63       100.42       473.05       0.4490          1.0427            1.0449         4.37
ExhaustiveRaBitQ-rf5 (query)                             372.63       165.21       537.84       0.7775          1.0056            1.0048         4.37
ExhaustiveRaBitQ-rf10 (query)                            372.63       226.20       598.84       0.8975          1.0019            1.0010         4.37
ExhaustiveRaBitQ-rf20 (query)                            372.63       342.82       715.45       0.9713          1.0005            1.0000         4.37
ExhaustiveRaBitQ-rfB (query)                             372.63       310.18       682.82       0.9714          1.0005            1.0000         4.37
ExhaustiveRaBitQ (self)                                  372.63       684.45     1_057.08       0.8982          1.0020            1.0010         4.37
IVF-RaBitQ-nl158-np7-rf0 (query)                         318.84        81.06       399.91       0.4606          1.0403            1.0426         4.58
IVF-RaBitQ-nl158-np12-rf0 (query)                        318.84        84.96       403.81       0.4606          1.0403            1.0426         4.58
IVF-RaBitQ-nl158-np17-rf0 (query)                        318.84        89.71       408.56       0.4606          1.0403            1.0426         4.58
IVF-RaBitQ-nl158-np7-rf10 (query)                        318.84       190.45       509.30       0.9070          1.0017            1.0008         4.58
IVF-RaBitQ-nl158-np7-rf20 (query)                        318.84       294.09       612.93       0.9748          1.0004            1.0000         4.58
IVF-RaBitQ-nl158-np7-rfB (query)                         318.84       218.06       536.90       0.9714          1.0004            1.0000         4.58
IVF-RaBitQ-nl158-np12-rf10 (query)                       318.84       192.93       511.77       0.9070          1.0017            1.0008         4.58
IVF-RaBitQ-nl158-np12-rf20 (query)                       318.84       315.41       634.25       0.9748          1.0004            1.0000         4.58
IVF-RaBitQ-nl158-np12-rfB (query)                        318.84       223.10       541.94       0.9714          1.0004            1.0000         4.58
IVF-RaBitQ-nl158-np17-rf10 (query)                       318.84       196.51       515.35       0.9070          1.0017            1.0008         4.58
IVF-RaBitQ-nl158-np17-rf20 (query)                       318.84       302.03       620.88       0.9748          1.0004            1.0000         4.58
IVF-RaBitQ-nl158-np17-rfB (query)                        318.84       228.89       547.74       0.9714          1.0004            1.0000         4.58
IVF-RaBitQ-nl158 (self)                                  318.84       685.37     1_004.21       0.9712          1.0005            1.0000         4.58
IVF-RaBitQ-nl223-np11-rf0 (query)                        369.71        86.80       456.52       0.4649          1.0394            1.0418         4.89
IVF-RaBitQ-nl223-np14-rf0 (query)                        369.71        89.65       459.36       0.4649          1.0394            1.0418         4.89
IVF-RaBitQ-nl223-np21-rf0 (query)                        369.71        92.13       461.85       0.4649          1.0394            1.0418         4.89
IVF-RaBitQ-nl223-np11-rf10 (query)                       369.71       206.91       576.62       0.9108          1.0016            1.0007         4.89
IVF-RaBitQ-nl223-np11-rf20 (query)                       369.71       296.96       666.67       0.9760          1.0004            1.0000         4.89
IVF-RaBitQ-nl223-np11-rfB (query)                        369.71       216.66       586.37       0.9720          1.0004            1.0000         4.89
IVF-RaBitQ-nl223-np14-rf10 (query)                       369.71       204.83       574.54       0.9108          1.0016            1.0007         4.89
IVF-RaBitQ-nl223-np14-rf20 (query)                       369.71       327.44       697.16       0.9760          1.0004            1.0000         4.89
IVF-RaBitQ-nl223-np14-rfB (query)                        369.71       232.14       601.85       0.9720          1.0004            1.0000         4.89
IVF-RaBitQ-nl223-np21-rf10 (query)                       369.71       205.83       575.54       0.9108          1.0016            1.0007         4.89
IVF-RaBitQ-nl223-np21-rf20 (query)                       369.71       316.46       686.18       0.9760          1.0004            1.0000         4.89
IVF-RaBitQ-nl223-np21-rfB (query)                        369.71       225.78       595.50       0.9720          1.0004            1.0000         4.89
IVF-RaBitQ-nl223 (self)                                  369.71       693.82     1_063.54       0.9717          1.0004            1.0000         4.89
IVF-RaBitQ-nl316-np15-rf0 (query)                        423.57        92.67       516.24       0.4716          1.0383            1.0409         5.34
IVF-RaBitQ-nl316-np17-rf0 (query)                        423.57        94.61       518.18       0.4716          1.0383            1.0409         5.34
IVF-RaBitQ-nl316-np25-rf0 (query)                        423.57       102.84       526.41       0.4716          1.0383            1.0409         5.34
IVF-RaBitQ-nl316-np15-rf10 (query)                       423.57       205.24       628.81       0.9131          1.0016            1.0006         5.34
IVF-RaBitQ-nl316-np15-rf20 (query)                       423.57       310.65       734.22       0.9766          1.0004            1.0000         5.34
IVF-RaBitQ-nl316-np15-rfB (query)                        423.57       215.96       639.54       0.9718          1.0004            1.0000         5.34
IVF-RaBitQ-nl316-np17-rf10 (query)                       423.57       202.58       626.15       0.9131          1.0016            1.0006         5.34
IVF-RaBitQ-nl316-np17-rf20 (query)                       423.57       310.19       733.76       0.9766          1.0004            1.0000         5.34
IVF-RaBitQ-nl316-np17-rfB (query)                        423.57       218.74       642.31       0.9718          1.0004            1.0000         5.34
IVF-RaBitQ-nl316-np25-rf10 (query)                       423.57       213.46       637.03       0.9131          1.0016            1.0006         5.34
IVF-RaBitQ-nl316-np25-rf20 (query)                       423.57       311.50       735.07       0.9766          1.0004            1.0000         5.34
IVF-RaBitQ-nl316-np25-rfB (query)                        423.57       221.93       645.50       0.9718          1.0004            1.0000         5.34
IVF-RaBitQ-nl316 (self)                                  423.57       672.36     1_095.93       0.9718          1.0004            1.0000         5.34
IVF-RaBitQ-nl223-ex2-np14 (query)                        409.90       225.04       634.94       0.8652          1.0020            1.0017        11.37
IVF-RaBitQ-nl223-ex2-np21 (query)                        409.90       231.13       641.03       0.8652          1.0020            1.0017        11.37
IVF-RaBitQ-nl223-ex4-np14 (query)                        440.23       256.69       696.92       0.9419          1.0006            1.0002        17.48
IVF-RaBitQ-nl223-ex4-np21 (query)                        440.23       260.95       701.18       0.9419          1.0006            1.0002        17.48
IVF-RaBitQ-nl223-ex8-np14 (query)                        476.70       231.68       708.38       0.9676          1.0004            1.0000        29.68
IVF-RaBitQ-nl223-ex8-np21 (query)                        476.70       241.61       718.31       0.9676          1.0004            1.0000        29.68
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 768D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       101.23     1_839.11     1_940.35       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.23     6_324.81     6_426.05       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                             539.11       126.84       665.95       0.4437          1.0346            1.0365         6.16
ExhaustiveRaBitQ-rf5 (query)                             539.11       201.23       740.35       0.7684          1.0047            1.0042         6.16
ExhaustiveRaBitQ-rf10 (query)                            539.11       273.96       813.07       0.8898          1.0017            1.0010         6.16
ExhaustiveRaBitQ-rf20 (query)                            539.11       415.71       954.82       0.9668          1.0004            1.0000         6.16
ExhaustiveRaBitQ-rfB (query)                             539.11       405.68       944.80       0.9693          1.0004            1.0000         6.16
ExhaustiveRaBitQ (self)                                  539.11       825.26     1_364.37       0.8897          1.0017            1.0010         6.16
IVF-RaBitQ-nl158-np7-rf0 (query)                         456.46       104.29       560.76       0.4552          1.0327            1.0351         6.50
IVF-RaBitQ-nl158-np12-rf0 (query)                        456.46       111.02       567.49       0.4552          1.0327            1.0351         6.50
IVF-RaBitQ-nl158-np17-rf0 (query)                        456.46       115.07       571.53       0.4552          1.0327            1.0351         6.50
IVF-RaBitQ-nl158-np7-rf10 (query)                        456.46       240.08       696.54       0.8982          1.0015            1.0008         6.50
IVF-RaBitQ-nl158-np7-rf20 (query)                        456.46       363.09       819.56       0.9706          1.0004            1.0000         6.50
IVF-RaBitQ-nl158-np7-rfB (query)                         456.46       292.17       748.63       0.9695          1.0004            1.0000         6.50
IVF-RaBitQ-nl158-np12-rf10 (query)                       456.46       242.26       698.72       0.8982          1.0015            1.0008         6.50
IVF-RaBitQ-nl158-np12-rf20 (query)                       456.46       370.64       827.10       0.9706          1.0004            1.0000         6.50
IVF-RaBitQ-nl158-np12-rfB (query)                        456.46       299.54       756.00       0.9695          1.0004            1.0000         6.50
IVF-RaBitQ-nl158-np17-rf10 (query)                       456.46       248.44       704.90       0.8982          1.0015            1.0008         6.50
IVF-RaBitQ-nl158-np17-rf20 (query)                       456.46       376.16       832.62       0.9706          1.0004            1.0000         6.50
IVF-RaBitQ-nl158-np17-rfB (query)                        456.46       305.30       761.76       0.9695          1.0004            1.0000         6.50
IVF-RaBitQ-nl158 (self)                                  456.46       926.97     1_383.43       0.9689          1.0004            1.0000         6.50
IVF-RaBitQ-nl223-np11-rf0 (query)                        508.40       114.09       622.49       0.4625          1.0317            1.0342         6.99
IVF-RaBitQ-nl223-np14-rf0 (query)                        508.40       117.61       626.02       0.4625          1.0317            1.0342         6.99
IVF-RaBitQ-nl223-np21-rf0 (query)                        508.40       120.80       629.20       0.4625          1.0317            1.0342         6.99
IVF-RaBitQ-nl223-np11-rf10 (query)                       508.40       245.62       754.02       0.9024          1.0015            1.0007         6.99
IVF-RaBitQ-nl223-np11-rf20 (query)                       508.40       370.88       879.28       0.9720          1.0003            1.0000         6.99
IVF-RaBitQ-nl223-np11-rfB (query)                        508.40       289.04       797.44       0.9698          1.0004            1.0000         6.99
IVF-RaBitQ-nl223-np14-rf10 (query)                       508.40       246.68       755.09       0.9024          1.0015            1.0007         6.99
IVF-RaBitQ-nl223-np14-rf20 (query)                       508.40       378.54       886.94       0.9720          1.0003            1.0000         6.99
IVF-RaBitQ-nl223-np14-rfB (query)                        508.40       293.68       802.08       0.9698          1.0004            1.0000         6.99
IVF-RaBitQ-nl223-np21-rf10 (query)                       508.40       257.34       765.75       0.9024          1.0015            1.0007         6.99
IVF-RaBitQ-nl223-np21-rf20 (query)                       508.40       383.92       892.32       0.9720          1.0003            1.0000         6.99
IVF-RaBitQ-nl223-np21-rfB (query)                        508.40       299.51       807.91       0.9698          1.0004            1.0000         6.99
IVF-RaBitQ-nl223 (self)                                  508.40       928.19     1_436.59       0.9692          1.0004            1.0000         6.99
IVF-RaBitQ-nl316-np15-rf0 (query)                        560.38       127.04       687.42       0.4653          1.0312            1.0337         7.64
IVF-RaBitQ-nl316-np17-rf0 (query)                        560.38       125.07       685.45       0.4653          1.0312            1.0337         7.64
IVF-RaBitQ-nl316-np25-rf0 (query)                        560.38       129.77       690.15       0.4653          1.0312            1.0337         7.64
IVF-RaBitQ-nl316-np15-rf10 (query)                       560.38       260.02       820.40       0.9042          1.0014            1.0007         7.64
IVF-RaBitQ-nl316-np15-rf20 (query)                       560.38       383.89       944.27       0.9727          1.0003            1.0000         7.64
IVF-RaBitQ-nl316-np15-rfB (query)                        560.38       291.97       852.35       0.9696          1.0004            1.0000         7.64
IVF-RaBitQ-nl316-np17-rf10 (query)                       560.38       256.58       816.96       0.9042          1.0014            1.0007         7.64
IVF-RaBitQ-nl316-np17-rf20 (query)                       560.38       385.10       945.48       0.9727          1.0003            1.0000         7.64
IVF-RaBitQ-nl316-np17-rfB (query)                        560.38       294.95       855.33       0.9696          1.0004            1.0000         7.64
IVF-RaBitQ-nl316-np25-rf10 (query)                       560.38       265.35       825.73       0.9042          1.0014            1.0007         7.64
IVF-RaBitQ-nl316-np25-rf20 (query)                       560.38       394.15       954.52       0.9727          1.0003            1.0000         7.64
IVF-RaBitQ-nl316-np25-rfB (query)                        560.38       300.75       861.13       0.9696          1.0004            1.0000         7.64
IVF-RaBitQ-nl316 (self)                                  560.38       916.77     1_477.15       0.9693          1.0004            1.0000         7.64
IVF-RaBitQ-nl223-ex2-np14 (query)                        546.42       333.12       879.54       0.8637          1.0016            1.0014        16.49
IVF-RaBitQ-nl223-ex2-np21 (query)                        546.42       339.64       886.06       0.8637          1.0016            1.0014        16.49
IVF-RaBitQ-nl223-ex4-np14 (query)                        589.03       376.49       965.51       0.9396          1.0005            1.0001        25.65
IVF-RaBitQ-nl223-ex4-np21 (query)                        589.03       384.13       973.15       0.9396          1.0005            1.0001        25.65
IVF-RaBitQ-nl223-ex8-np14 (query)                        672.70       345.17     1_017.87       0.9652          1.0004            1.0000        43.96
IVF-RaBitQ-nl223-ex8-np21 (query)                        672.70       341.51     1_014.21       0.9652          1.0004            1.0000        43.96
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 256D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        33.07       685.18       718.25       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.07     2_318.97     2_352.04       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             235.28        78.59       313.87       0.6872          1.0347            1.0333         2.57
ExhaustiveRaBitQ-rf5 (query)                             235.28       130.79       366.08       0.9895          1.0003            1.0000         2.57
ExhaustiveRaBitQ-rf10 (query)                            235.28       187.23       422.51       0.9995          1.0000            1.0000         2.57
ExhaustiveRaBitQ-rf20 (query)                            235.28       280.47       515.75       1.0000          1.0000            1.0000         2.57
ExhaustiveRaBitQ-rfB (query)                             235.28       125.03       360.31       0.9959          1.0001            1.0000         2.57
ExhaustiveRaBitQ (self)                                  235.28       593.19       828.48       0.9997          1.0000            1.0000         2.57
IVF-RaBitQ-nl158-np7-rf0 (query)                         173.01        45.77       218.78       0.6915          1.0336            1.0322         2.67
IVF-RaBitQ-nl158-np12-rf0 (query)                        173.01        50.54       223.55       0.6915          1.0336            1.0322         2.67
IVF-RaBitQ-nl158-np17-rf0 (query)                        173.01        53.19       226.21       0.6915          1.0336            1.0322         2.67
IVF-RaBitQ-nl158-np7-rf10 (query)                        173.01       124.43       297.45       0.9996          1.0000            1.0000         2.67
IVF-RaBitQ-nl158-np7-rf20 (query)                        173.01       195.07       368.08       1.0000          1.0000            1.0000         2.67
IVF-RaBitQ-nl158-np7-rfB (query)                         173.01        80.99       254.01       0.9961          1.0001            1.0000         2.67
IVF-RaBitQ-nl158-np12-rf10 (query)                       173.01       126.74       299.75       0.9996          1.0000            1.0000         2.67
IVF-RaBitQ-nl158-np12-rf20 (query)                       173.01       198.24       371.25       1.0000          1.0000            1.0000         2.67
IVF-RaBitQ-nl158-np12-rfB (query)                        173.01        83.22       256.23       0.9961          1.0001            1.0000         2.67
IVF-RaBitQ-nl158-np17-rf10 (query)                       173.01       131.21       304.22       0.9996          1.0000            1.0000         2.67
IVF-RaBitQ-nl158-np17-rf20 (query)                       173.01       203.12       376.14       1.0000          1.0000            1.0000         2.67
IVF-RaBitQ-nl158-np17-rfB (query)                        173.01        87.76       260.77       0.9961          1.0001            1.0000         2.67
IVF-RaBitQ-nl158 (self)                                  173.01       245.96       418.98       0.9957          1.0001            1.0000         2.67
IVF-RaBitQ-nl223-np11-rf0 (query)                        190.04        49.67       239.71       0.6962          1.0327            1.0313         2.84
IVF-RaBitQ-nl223-np14-rf0 (query)                        190.04        50.72       240.76       0.6962          1.0327            1.0313         2.84
IVF-RaBitQ-nl223-np21-rf0 (query)                        190.04        55.33       245.37       0.6962          1.0327            1.0313         2.84
IVF-RaBitQ-nl223-np11-rf10 (query)                       190.04       126.70       316.74       0.9996          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np11-rf20 (query)                       190.04       195.09       385.13       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np11-rfB (query)                        190.04        82.24       272.28       0.9960          1.0001            1.0000         2.84
IVF-RaBitQ-nl223-np14-rf10 (query)                       190.04       128.36       318.40       0.9996          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np14-rf20 (query)                       190.04       198.66       388.70       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np14-rfB (query)                        190.04        83.69       273.73       0.9960          1.0001            1.0000         2.84
IVF-RaBitQ-nl223-np21-rf10 (query)                       190.04       132.21       322.25       0.9996          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np21-rf20 (query)                       190.04       203.63       393.67       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np21-rfB (query)                        190.04        87.20       277.24       0.9960          1.0001            1.0000         2.84
IVF-RaBitQ-nl223 (self)                                  190.04       251.73       441.77       0.9959          1.0001            1.0000         2.84
IVF-RaBitQ-nl316-np15-rf0 (query)                        208.35        53.82       262.18       0.6995          1.0319            1.0306         3.06
IVF-RaBitQ-nl316-np17-rf0 (query)                        208.35        53.54       261.89       0.6995          1.0319            1.0306         3.06
IVF-RaBitQ-nl316-np25-rf0 (query)                        208.35        57.17       265.52       0.6995          1.0319            1.0306         3.06
IVF-RaBitQ-nl316-np15-rf10 (query)                       208.35       130.28       338.63       0.9996          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np15-rf20 (query)                       208.35       197.94       406.29       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np15-rfB (query)                        208.35        86.10       294.45       0.9963          1.0001            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf10 (query)                       208.35       128.60       336.96       0.9996          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf20 (query)                       208.35       201.38       409.73       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rfB (query)                        208.35        85.31       293.66       0.9963          1.0001            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf10 (query)                       208.35       133.87       342.22       0.9996          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf20 (query)                       208.35       204.05       412.40       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rfB (query)                        208.35        92.12       300.47       0.9963          1.0001            1.0000         3.06
IVF-RaBitQ-nl316 (self)                                  208.35       255.99       464.34       0.9960          1.0001            1.0000         3.06
IVF-RaBitQ-nl223-ex2-np14 (query)                        213.13        90.74       303.87       0.9249          1.0016            1.0012         6.27
IVF-RaBitQ-nl223-ex2-np21 (query)                        213.13        96.97       310.09       0.9249          1.0016            1.0012         6.27
IVF-RaBitQ-nl223-ex4-np14 (query)                        215.24        98.85       314.09       0.9733          1.0002            1.0000         9.32
IVF-RaBitQ-nl223-ex4-np21 (query)                        215.24       104.83       320.07       0.9733          1.0002            1.0000         9.32
IVF-RaBitQ-nl223-ex8-np14 (query)                        242.76        93.00       335.76       0.9915          1.0001            1.0000        15.42
IVF-RaBitQ-nl223-ex8-np21 (query)                        242.76        97.28       340.04       0.9915          1.0001            1.0000        15.42
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 512D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        69.00     1_266.41     1_335.41       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.00     4_295.26     4_364.26       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                             389.52       123.57       513.09       0.6581          1.0282            1.0278         4.35
ExhaustiveRaBitQ-rf5 (query)                             389.52       187.49       577.01       0.9782          1.0005            1.0000         4.35
ExhaustiveRaBitQ-rf10 (query)                            389.52       256.94       646.46       0.9981          1.0000            1.0000         4.35
ExhaustiveRaBitQ-rf20 (query)                            389.52       375.22       764.74       0.9999          1.0000            1.0000         4.35
ExhaustiveRaBitQ-rfB (query)                             389.52       225.27       614.79       0.9902          1.0002            1.0000         4.35
ExhaustiveRaBitQ (self)                                  389.52       777.00     1_166.52       0.9985          1.0000            1.0000         4.35
IVF-RaBitQ-nl158-np7-rf0 (query)                         318.62        80.19       398.82       0.6603          1.0279            1.0275         4.58
IVF-RaBitQ-nl158-np12-rf0 (query)                        318.62        84.08       402.70       0.6603          1.0279            1.0275         4.58
IVF-RaBitQ-nl158-np17-rf0 (query)                        318.62        88.24       406.86       0.6603          1.0279            1.0275         4.58
IVF-RaBitQ-nl158-np7-rf10 (query)                        318.62       192.59       511.21       0.9981          1.0000            1.0000         4.58
IVF-RaBitQ-nl158-np7-rf20 (query)                        318.62       300.87       619.49       0.9999          1.0000            1.0000         4.58
IVF-RaBitQ-nl158-np7-rfB (query)                         318.62       147.05       465.67       0.9903          1.0002            1.0000         4.58
IVF-RaBitQ-nl158-np12-rf10 (query)                       318.62       195.70       514.33       0.9981          1.0000            1.0000         4.58
IVF-RaBitQ-nl158-np12-rf20 (query)                       318.62       301.65       620.27       0.9999          1.0000            1.0000         4.58
IVF-RaBitQ-nl158-np12-rfB (query)                        318.62       151.22       469.85       0.9903          1.0002            1.0000         4.58
IVF-RaBitQ-nl158-np17-rf10 (query)                       318.62       199.65       518.27       0.9981          1.0000            1.0000         4.58
IVF-RaBitQ-nl158-np17-rf20 (query)                       318.62       307.92       626.54       0.9999          1.0000            1.0000         4.58
IVF-RaBitQ-nl158-np17-rfB (query)                        318.62       154.36       472.98       0.9903          1.0002            1.0000         4.58
IVF-RaBitQ-nl158 (self)                                  318.62       432.24       750.86       0.9890          1.0003            1.0000         4.58
IVF-RaBitQ-nl223-np11-rf0 (query)                        366.04        86.61       452.65       0.6631          1.0275            1.0275         4.88
IVF-RaBitQ-nl223-np14-rf0 (query)                        366.04        92.90       458.94       0.6631          1.0275            1.0275         4.88
IVF-RaBitQ-nl223-np21-rf0 (query)                        366.04        94.00       460.04       0.6631          1.0275            1.0275         4.88
IVF-RaBitQ-nl223-np11-rf10 (query)                       366.04       199.45       565.48       0.9982          1.0000            1.0000         4.88
IVF-RaBitQ-nl223-np11-rf20 (query)                       366.04       306.01       672.05       1.0000          1.0000            1.0000         4.88
IVF-RaBitQ-nl223-np11-rfB (query)                        366.04       159.74       525.78       0.9901          1.0002            1.0000         4.88
IVF-RaBitQ-nl223-np14-rf10 (query)                       366.04       212.60       578.64       0.9982          1.0000            1.0000         4.88
IVF-RaBitQ-nl223-np14-rf20 (query)                       366.04       307.29       673.33       1.0000          1.0000            1.0000         4.88
IVF-RaBitQ-nl223-np14-rfB (query)                        366.04       155.09       521.13       0.9901          1.0002            1.0000         4.88
IVF-RaBitQ-nl223-np21-rf10 (query)                       366.04       203.78       569.81       0.9982          1.0000            1.0000         4.88
IVF-RaBitQ-nl223-np21-rf20 (query)                       366.04       311.90       677.94       1.0000          1.0000            1.0000         4.88
IVF-RaBitQ-nl223-np21-rfB (query)                        366.04       159.12       525.16       0.9901          1.0002            1.0000         4.88
IVF-RaBitQ-nl223 (self)                                  366.04       446.42       812.46       0.9890          1.0003            1.0000         4.88
IVF-RaBitQ-nl316-np15-rf0 (query)                        405.83        96.28       502.12       0.6662          1.0269            1.0267         5.36
IVF-RaBitQ-nl316-np17-rf0 (query)                        405.83        94.26       500.10       0.6662          1.0269            1.0267         5.36
IVF-RaBitQ-nl316-np25-rf0 (query)                        405.83        99.32       505.15       0.6662          1.0269            1.0267         5.36
IVF-RaBitQ-nl316-np15-rf10 (query)                       405.83       207.75       613.59       0.9982          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np15-rf20 (query)                       405.83       312.22       718.05       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np15-rfB (query)                        405.83       160.12       565.95       0.9906          1.0002            1.0000         5.36
IVF-RaBitQ-nl316-np17-rf10 (query)                       405.83       206.33       612.16       0.9982          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np17-rf20 (query)                       405.83       313.82       719.65       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np17-rfB (query)                        405.83       159.80       565.63       0.9906          1.0002            1.0000         5.36
IVF-RaBitQ-nl316-np25-rf10 (query)                       405.83       211.41       617.24       0.9982          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np25-rf20 (query)                       405.83       320.57       726.40       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np25-rfB (query)                        405.83       165.42       571.26       0.9906          1.0002            1.0000         5.36
IVF-RaBitQ-nl316 (self)                                  405.83       466.02       871.85       0.9893          1.0003            1.0000         5.36
IVF-RaBitQ-nl223-ex2-np14 (query)                        399.07       169.81       568.89       0.9257          1.0011            1.0008        11.37
IVF-RaBitQ-nl223-ex2-np21 (query)                        399.07       178.96       578.03       0.9257          1.0011            1.0008        11.37
IVF-RaBitQ-nl223-ex4-np14 (query)                        439.21       191.28       630.48       0.9699          1.0003            1.0000        17.47
IVF-RaBitQ-nl223-ex4-np21 (query)                        439.21       193.22       632.43       0.9699          1.0003            1.0000        17.47
IVF-RaBitQ-nl223-ex8-np14 (query)                        473.72       176.91       650.63       0.9867          1.0002            1.0000        29.68
IVF-RaBitQ-nl223-ex8-np21 (query)                        473.72       180.53       654.26       0.9867          1.0002            1.0000        29.68
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 768D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       100.83     1_816.74     1_917.56       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.83     6_184.44     6_285.27       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                             551.50       161.87       713.37       0.5777          1.0323            1.0331         6.17
ExhaustiveRaBitQ-rf5 (query)                             551.50       238.33       789.83       0.9286          1.0016            1.0006         6.17
ExhaustiveRaBitQ-rf10 (query)                            551.50       313.44       864.94       0.9841          1.0003            1.0000         6.17
ExhaustiveRaBitQ-rf20 (query)                            551.50       458.49     1_009.99       0.9987          1.0000            1.0000         6.17
ExhaustiveRaBitQ-rfB (query)                             551.50       328.77       880.27       0.9772          1.0004            1.0000         6.17
ExhaustiveRaBitQ (self)                                  551.50       957.99     1_509.49       0.9852          1.0003            1.0000         6.17
IVF-RaBitQ-nl158-np7-rf0 (query)                         449.06       102.39       551.45       0.5842          1.0314            1.0325         6.49
IVF-RaBitQ-nl158-np12-rf0 (query)                        449.06       108.19       557.25       0.5842          1.0314            1.0325         6.49
IVF-RaBitQ-nl158-np17-rf0 (query)                        449.06       113.43       562.49       0.5842          1.0314            1.0325         6.49
IVF-RaBitQ-nl158-np7-rf10 (query)                        449.06       237.18       686.24       0.9853          1.0003            1.0000         6.49
IVF-RaBitQ-nl158-np7-rf20 (query)                        449.06       362.04       811.09       0.9987          1.0000            1.0000         6.49
IVF-RaBitQ-nl158-np7-rfB (query)                         449.06       212.36       661.42       0.9776          1.0004            1.0000         6.49
IVF-RaBitQ-nl158-np12-rf10 (query)                       449.06       242.73       691.79       0.9853          1.0003            1.0000         6.49
IVF-RaBitQ-nl158-np12-rf20 (query)                       449.06       369.78       818.83       0.9987          1.0000            1.0000         6.49
IVF-RaBitQ-nl158-np12-rfB (query)                        449.06       218.45       667.51       0.9776          1.0004            1.0000         6.49
IVF-RaBitQ-nl158-np17-rf10 (query)                       449.06       247.86       696.92       0.9853          1.0003            1.0000         6.49
IVF-RaBitQ-nl158-np17-rf20 (query)                       449.06       378.64       827.70       0.9987          1.0000            1.0000         6.49
IVF-RaBitQ-nl158-np17-rfB (query)                        449.06       223.42       672.48       0.9776          1.0004            1.0000         6.49
IVF-RaBitQ-nl158 (self)                                  449.06       654.67     1_103.73       0.9757          1.0005            1.0000         6.49
IVF-RaBitQ-nl223-np11-rf0 (query)                        518.77       116.60       635.38       0.5855          1.0312            1.0322         6.94
IVF-RaBitQ-nl223-np14-rf0 (query)                        518.77       116.78       635.55       0.5855          1.0312            1.0322         6.94
IVF-RaBitQ-nl223-np21-rf0 (query)                        518.77       119.94       638.71       0.5855          1.0312            1.0322         6.94
IVF-RaBitQ-nl223-np11-rf10 (query)                       518.77       248.42       767.19       0.9855          1.0002            1.0000         6.94
IVF-RaBitQ-nl223-np11-rf20 (query)                       518.77       375.44       894.21       0.9988          1.0000            1.0000         6.94
IVF-RaBitQ-nl223-np11-rfB (query)                        518.77       221.28       740.05       0.9771          1.0004            1.0000         6.94
IVF-RaBitQ-nl223-np14-rf10 (query)                       518.77       247.67       766.45       0.9855          1.0002            1.0000         6.94
IVF-RaBitQ-nl223-np14-rf20 (query)                       518.77       379.83       898.60       0.9988          1.0000            1.0000         6.94
IVF-RaBitQ-nl223-np14-rfB (query)                        518.77       222.99       741.77       0.9771          1.0004            1.0000         6.94
IVF-RaBitQ-nl223-np21-rf10 (query)                       518.77       257.95       776.72       0.9855          1.0002            1.0000         6.94
IVF-RaBitQ-nl223-np21-rf20 (query)                       518.77       392.37       911.14       0.9988          1.0000            1.0000         6.94
IVF-RaBitQ-nl223-np21-rfB (query)                        518.77       235.58       754.36       0.9771          1.0004            1.0000         6.94
IVF-RaBitQ-nl223 (self)                                  518.77       660.50     1_179.28       0.9758          1.0005            1.0000         6.94
IVF-RaBitQ-nl316-np15-rf0 (query)                        563.03       125.28       688.31       0.5876          1.0308            1.0321         7.65
IVF-RaBitQ-nl316-np17-rf0 (query)                        563.03       123.91       686.94       0.5876          1.0308            1.0321         7.65
IVF-RaBitQ-nl316-np25-rf0 (query)                        563.03       130.51       693.54       0.5876          1.0308            1.0321         7.65
IVF-RaBitQ-nl316-np15-rf10 (query)                       563.03       262.44       825.47       0.9861          1.0002            1.0000         7.65
IVF-RaBitQ-nl316-np15-rf20 (query)                       563.03       389.16       952.19       0.9989          1.0000            1.0000         7.65
IVF-RaBitQ-nl316-np15-rfB (query)                        563.03       230.09       793.13       0.9774          1.0004            1.0000         7.65
IVF-RaBitQ-nl316-np17-rf10 (query)                       563.03       259.78       822.82       0.9861          1.0002            1.0000         7.65
IVF-RaBitQ-nl316-np17-rf20 (query)                       563.03       392.51       955.54       0.9989          1.0000            1.0000         7.65
IVF-RaBitQ-nl316-np17-rfB (query)                        563.03       230.52       793.56       0.9774          1.0004            1.0000         7.65
IVF-RaBitQ-nl316-np25-rf10 (query)                       563.03       269.75       832.79       0.9861          1.0002            1.0000         7.65
IVF-RaBitQ-nl316-np25-rf20 (query)                       563.03       402.55       965.58       0.9989          1.0000            1.0000         7.65
IVF-RaBitQ-nl316-np25-rfB (query)                        563.03       238.28       801.31       0.9774          1.0004            1.0000         7.65
IVF-RaBitQ-nl316 (self)                                  563.03       691.09     1_254.13       0.9760          1.0005            1.0000         7.65
IVF-RaBitQ-nl223-ex2-np14 (query)                        557.00       263.64       820.64       0.9138          1.0012            1.0008        16.49
IVF-RaBitQ-nl223-ex2-np21 (query)                        557.00       264.14       821.14       0.9138          1.0012            1.0008        16.49
IVF-RaBitQ-nl223-ex4-np14 (query)                        592.16       288.82       880.98       0.9590          1.0005            1.0000        25.65
IVF-RaBitQ-nl223-ex4-np21 (query)                        592.16       294.74       886.90       0.9590          1.0005            1.0000        25.65
IVF-RaBitQ-nl223-ex8-np14 (query)                        658.88       264.23       923.11       0.9748          1.0004            1.0000        43.96
IVF-RaBitQ-nl223-ex8-np21 (query)                        658.88       268.82       927.70       0.9748          1.0004            1.0000        43.96
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Cell embeddings

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 256D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        33.14       680.61       713.76       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         33.14     2_299.32     2_332.47       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             363.15        99.08       462.23       0.8703          1.0284            1.0233         2.57
ExhaustiveRaBitQ-rf5 (query)                             363.15       163.57       526.72       1.0000          1.0000            1.0000         2.57
ExhaustiveRaBitQ-rf10 (query)                            363.15       218.98       582.14       1.0000          1.0000            1.0000         2.57
ExhaustiveRaBitQ-rf20 (query)                            363.15       322.91       686.06       1.0000          1.0000            1.0000         2.57
ExhaustiveRaBitQ-rfB (query)                             363.15       135.77       498.92       1.0000          1.0000            1.0000         2.57
ExhaustiveRaBitQ (self)                                  363.15       705.54     1_068.69       1.0000          1.0000            1.0000         2.57
IVF-RaBitQ-nl158-np7-rf0 (query)                         171.16        48.49       219.66       0.8765          1.0258            1.0210         2.68
IVF-RaBitQ-nl158-np12-rf0 (query)                        171.16        56.43       227.59       0.8773          1.0253            1.0207         2.68
IVF-RaBitQ-nl158-np17-rf0 (query)                        171.16        64.57       235.74       0.8773          1.0253            1.0207         2.68
IVF-RaBitQ-nl158-np7-rf10 (query)                        171.16       129.53       300.69       0.9968          1.0007            1.0000         2.68
IVF-RaBitQ-nl158-np7-rf20 (query)                        171.16       203.06       374.23       0.9968          1.0007            1.0000         2.68
IVF-RaBitQ-nl158-np7-rfB (query)                         171.16        70.91       242.07       0.9967          1.0007            1.0000         2.68
IVF-RaBitQ-nl158-np12-rf10 (query)                       171.16       140.75       311.92       0.9999          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np12-rf20 (query)                       171.16       213.89       385.05       0.9999          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np12-rfB (query)                        171.16        80.74       251.91       0.9998          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rf10 (query)                       171.16       149.36       320.52       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rf20 (query)                       171.16       225.32       396.48       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rfB (query)                        171.16        87.50       258.66       0.9999          1.0000            1.0000         2.68
IVF-RaBitQ-nl158 (self)                                  171.16       256.38       427.55       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl223-np11-rf0 (query)                        201.68        51.85       253.53       0.8826          1.0229            1.0188         2.83
IVF-RaBitQ-nl223-np14-rf0 (query)                        201.68        54.96       256.64       0.8827          1.0228            1.0187         2.83
IVF-RaBitQ-nl223-np21-rf0 (query)                        201.68        62.71       264.39       0.8828          1.0228            1.0187         2.83
IVF-RaBitQ-nl223-np11-rf10 (query)                       201.68       130.02       331.70       0.9994          1.0001            1.0000         2.83
IVF-RaBitQ-nl223-np11-rf20 (query)                       201.68       202.19       403.87       0.9994          1.0001            1.0000         2.83
IVF-RaBitQ-nl223-np11-rfB (query)                        201.68        74.77       276.45       0.9993          1.0001            1.0000         2.83
IVF-RaBitQ-nl223-np14-rf10 (query)                       201.68       133.55       335.23       0.9998          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np14-rf20 (query)                       201.68       206.57       408.25       0.9998          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np14-rfB (query)                        201.68        75.38       277.06       0.9998          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np21-rf10 (query)                       201.68       141.91       343.59       1.0000          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np21-rf20 (query)                       201.68       215.64       417.32       1.0000          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np21-rfB (query)                        201.68        83.43       285.11       1.0000          1.0000            1.0000         2.83
IVF-RaBitQ-nl223 (self)                                  201.68       247.07       448.75       1.0000          1.0000            1.0000         2.83
IVF-RaBitQ-nl316-np15-rf0 (query)                        215.95        55.26       271.21       0.8900          1.0198            1.0162         3.06
IVF-RaBitQ-nl316-np17-rf0 (query)                        215.95        55.77       271.72       0.8900          1.0198            1.0162         3.06
IVF-RaBitQ-nl316-np25-rf0 (query)                        215.95        62.49       278.44       0.8900          1.0198            1.0162         3.06
IVF-RaBitQ-nl316-np15-rf10 (query)                       215.95       130.87       346.82       0.9997          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np15-rf20 (query)                       215.95       205.15       421.11       0.9997          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np15-rfB (query)                        215.95        73.85       289.80       0.9997          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf10 (query)                       215.95       131.18       347.13       0.9998          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf20 (query)                       215.95       203.46       419.41       0.9998          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rfB (query)                        215.95        74.79       290.74       0.9998          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf10 (query)                       215.95       147.63       363.58       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf20 (query)                       215.95       223.20       439.15       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rfB (query)                        215.95        81.54       297.49       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316 (self)                                  215.95       240.85       456.80       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl223-ex2-np14 (query)                        210.55        80.87       291.41       0.9661          1.0014            1.0000         6.27
IVF-RaBitQ-nl223-ex2-np21 (query)                        210.55        88.65       299.19       0.9661          1.0014            1.0000         6.27
IVF-RaBitQ-nl223-ex4-np14 (query)                        215.44        86.56       302.00       0.9886          1.0002            1.0000         9.32
IVF-RaBitQ-nl223-ex4-np21 (query)                        215.44        94.76       310.20       0.9887          1.0001            1.0000         9.32
IVF-RaBitQ-nl223-ex8-np14 (query)                        252.44        82.45       334.89       0.9977          1.0000            1.0000        15.42
IVF-RaBitQ-nl223-ex8-np21 (query)                        252.44        89.85       342.29       0.9978          1.0000            1.0000        15.42
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 512D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.68     1_277.69     1_346.38       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.68     4_312.00     4_380.69       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                             579.02       155.17       734.19       0.9088          1.0131            1.0099         4.36
ExhaustiveRaBitQ-rf5 (query)                             579.02       243.69       822.71       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ-rf10 (query)                            579.02       299.10       878.12       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ-rf20 (query)                            579.02       436.75     1_015.77       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ-rfB (query)                             579.02       224.11       803.13       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ (self)                                  579.02       937.72     1_516.75       1.0000          1.0000            1.0000         4.36
IVF-RaBitQ-nl158-np7-rf0 (query)                         319.56        84.10       403.66       0.9158          1.0108            1.0079         4.58
IVF-RaBitQ-nl158-np12-rf0 (query)                        319.56        96.30       415.87       0.9163          1.0106            1.0078         4.58
IVF-RaBitQ-nl158-np17-rf0 (query)                        319.56       107.36       426.92       0.9163          1.0106            1.0078         4.58
IVF-RaBitQ-nl158-np7-rf10 (query)                        319.56       193.38       512.94       0.9985          1.0003            1.0000         4.58
IVF-RaBitQ-nl158-np7-rf20 (query)                        319.56       300.17       619.73       0.9985          1.0003            1.0000         4.58
IVF-RaBitQ-nl158-np7-rfB (query)                         319.56       116.74       436.30       0.9985          1.0003            1.0000         4.58
IVF-RaBitQ-nl158-np12-rf10 (query)                       319.56       206.52       526.08       0.9999          1.0000            1.0000         4.58
IVF-RaBitQ-nl158-np12-rf20 (query)                       319.56       318.54       638.11       0.9999          1.0000            1.0000         4.58
IVF-RaBitQ-nl158-np12-rfB (query)                        319.56       146.03       465.60       0.9999          1.0000            1.0000         4.58
IVF-RaBitQ-nl158-np17-rf10 (query)                       319.56       216.61       536.17       1.0000          1.0000            1.0000         4.58
IVF-RaBitQ-nl158-np17-rf20 (query)                       319.56       332.21       651.78       1.0000          1.0000            1.0000         4.58
IVF-RaBitQ-nl158-np17-rfB (query)                        319.56       139.39       458.95       1.0000          1.0000            1.0000         4.58
IVF-RaBitQ-nl158 (self)                                  319.56       401.73       721.29       1.0000          1.0000            1.0000         4.58
IVF-RaBitQ-nl223-np11-rf0 (query)                        366.24        91.00       457.24       0.9215          1.0093            1.0069         4.90
IVF-RaBitQ-nl223-np14-rf0 (query)                        366.24        95.97       462.21       0.9216          1.0093            1.0068         4.90
IVF-RaBitQ-nl223-np21-rf0 (query)                        366.24       107.71       473.95       0.9216          1.0093            1.0068         4.90
IVF-RaBitQ-nl223-np11-rf10 (query)                       366.24       199.37       565.61       0.9997          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np11-rf20 (query)                       366.24       308.71       674.95       0.9997          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np11-rfB (query)                        366.24       121.39       487.62       0.9997          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np14-rf10 (query)                       366.24       205.44       571.67       0.9999          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np14-rf20 (query)                       366.24       313.20       679.44       0.9999          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np14-rfB (query)                        366.24       126.37       492.61       0.9999          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np21-rf10 (query)                       366.24       215.98       582.22       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np21-rf20 (query)                       366.24       333.21       699.44       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np21-rfB (query)                        366.24       138.62       504.86       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223 (self)                                  366.24       416.74       782.98       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl316-np15-rf0 (query)                        400.35        97.37       497.72       0.9263          1.0081            1.0058         5.36
IVF-RaBitQ-nl316-np17-rf0 (query)                        400.35        99.28       499.64       0.9263          1.0081            1.0058         5.36
IVF-RaBitQ-nl316-np25-rf0 (query)                        400.35       109.58       509.93       0.9263          1.0081            1.0058         5.36
IVF-RaBitQ-nl316-np15-rf10 (query)                       400.35       206.72       607.07       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np15-rf20 (query)                       400.35       314.16       714.51       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np15-rfB (query)                        400.35       126.52       526.87       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np17-rf10 (query)                       400.35       211.06       611.41       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np17-rf20 (query)                       400.35       327.88       728.23       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np17-rfB (query)                        400.35       128.02       528.37       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np25-rf10 (query)                       400.35       218.35       618.70       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np25-rf20 (query)                       400.35       332.07       732.42       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np25-rfB (query)                        400.35       141.17       541.52       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl316 (self)                                  400.35       396.55       796.91       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl223-ex2-np14 (query)                        396.23       141.14       537.36       0.9761          1.0007            1.0000        11.39
IVF-RaBitQ-nl223-ex2-np21 (query)                        396.23       150.35       546.58       0.9761          1.0007            1.0000        11.39
IVF-RaBitQ-nl223-ex4-np14 (query)                        450.97       150.50       601.47       0.9921          1.0001            1.0000        17.49
IVF-RaBitQ-nl223-ex4-np21 (query)                        450.97       162.45       613.42       0.9921          1.0001            1.0000        17.49
IVF-RaBitQ-nl223-ex8-np14 (query)                        469.68       141.08       610.77       0.9988          1.0000            1.0000        29.70
IVF-RaBitQ-nl223-ex8-np21 (query)                        469.68       153.43       623.11       0.9989          1.0000            1.0000        29.70
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 768D - RaBitQ (IVF and exhaustive)
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       100.33     1_875.07     1_975.40       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.33     6_242.97     6_343.30       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                             856.58       197.81     1_054.39       0.9143          1.0117            1.0085         6.16
ExhaustiveRaBitQ-rf5 (query)                             856.58       283.00     1_139.59       1.0000          1.0000            1.0000         6.16
ExhaustiveRaBitQ-rf10 (query)                            856.58       365.59     1_222.17       1.0000          1.0000            1.0000         6.16
ExhaustiveRaBitQ-rf20 (query)                            856.58       522.78     1_379.36       1.0000          1.0000            1.0000         6.16
ExhaustiveRaBitQ-rfB (query)                             856.58       319.50     1_176.09       0.9999          1.0000            1.0000         6.16
ExhaustiveRaBitQ (self)                                  856.58     1_138.82     1_995.40       1.0000          1.0000            1.0000         6.16
IVF-RaBitQ-nl158-np7-rf0 (query)                         449.05       109.04       558.10       0.9175          1.0108            1.0078         6.50
IVF-RaBitQ-nl158-np12-rf0 (query)                        449.05       126.79       575.84       0.9178          1.0107            1.0077         6.50
IVF-RaBitQ-nl158-np17-rf0 (query)                        449.05       140.37       589.42       0.9178          1.0107            1.0077         6.50
IVF-RaBitQ-nl158-np7-rf10 (query)                        449.05       241.13       690.19       0.9993          1.0001            1.0000         6.50
IVF-RaBitQ-nl158-np7-rf20 (query)                        449.05       368.22       817.27       0.9993          1.0001            1.0000         6.50
IVF-RaBitQ-nl158-np7-rfB (query)                         449.05       152.23       601.28       0.9993          1.0001            1.0000         6.50
IVF-RaBitQ-nl158-np12-rf10 (query)                       449.05       256.33       705.38       0.9999          1.0000            1.0000         6.50
IVF-RaBitQ-nl158-np12-rf20 (query)                       449.05       389.04       838.10       0.9999          1.0000            1.0000         6.50
IVF-RaBitQ-nl158-np12-rfB (query)                        449.05       169.15       618.20       0.9999          1.0000            1.0000         6.50
IVF-RaBitQ-nl158-np17-rf10 (query)                       449.05       272.09       721.14       1.0000          1.0000            1.0000         6.50
IVF-RaBitQ-nl158-np17-rf20 (query)                       449.05       408.28       857.34       1.0000          1.0000            1.0000         6.50
IVF-RaBitQ-nl158-np17-rfB (query)                        449.05       186.59       635.65       1.0000          1.0000            1.0000         6.50
IVF-RaBitQ-nl158 (self)                                  449.05       529.58       978.63       1.0000          1.0000            1.0000         6.50
IVF-RaBitQ-nl223-np11-rf0 (query)                        508.88       119.56       628.44       0.9222          1.0095            1.0066         6.95
IVF-RaBitQ-nl223-np14-rf0 (query)                        508.88       125.87       634.76       0.9223          1.0094            1.0066         6.95
IVF-RaBitQ-nl223-np21-rf0 (query)                        508.88       142.49       651.37       0.9223          1.0094            1.0066         6.95
IVF-RaBitQ-nl223-np11-rf10 (query)                       508.88       249.45       758.34       0.9998          1.0000            1.0000         6.95
IVF-RaBitQ-nl223-np11-rf20 (query)                       508.88       379.37       888.25       0.9998          1.0000            1.0000         6.95
IVF-RaBitQ-nl223-np11-rfB (query)                        508.88       159.57       668.45       0.9997          1.0000            1.0000         6.95
IVF-RaBitQ-nl223-np14-rf10 (query)                       508.88       257.47       766.35       1.0000          1.0000            1.0000         6.95
IVF-RaBitQ-nl223-np14-rf20 (query)                       508.88       392.90       901.78       1.0000          1.0000            1.0000         6.95
IVF-RaBitQ-nl223-np14-rfB (query)                        508.88       166.34       675.22       0.9999          1.0000            1.0000         6.95
IVF-RaBitQ-nl223-np21-rf10 (query)                       508.88       273.78       782.67       1.0000          1.0000            1.0000         6.95
IVF-RaBitQ-nl223-np21-rf20 (query)                       508.88       410.20       919.08       1.0000          1.0000            1.0000         6.95
IVF-RaBitQ-nl223-np21-rfB (query)                        508.88       182.05       690.93       0.9999          1.0000            1.0000         6.95
IVF-RaBitQ-nl223 (self)                                  508.88       527.05     1_035.94       1.0000          1.0000            1.0000         6.95
IVF-RaBitQ-nl316-np15-rf0 (query)                        565.49       127.85       693.34       0.9259          1.0083            1.0058         7.66
IVF-RaBitQ-nl316-np17-rf0 (query)                        565.49       129.58       695.07       0.9259          1.0083            1.0058         7.66
IVF-RaBitQ-nl316-np25-rf0 (query)                        565.49       143.77       709.26       0.9259          1.0083            1.0058         7.66
IVF-RaBitQ-nl316-np15-rf10 (query)                       565.49       262.84       828.32       0.9999          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np15-rf20 (query)                       565.49       396.05       961.54       0.9999          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np15-rfB (query)                        565.49       166.69       732.17       0.9999          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np17-rf10 (query)                       565.49       264.44       829.93       1.0000          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np17-rf20 (query)                       565.49       397.19       962.68       1.0000          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np17-rfB (query)                        565.49       168.90       734.38       0.9999          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np25-rf10 (query)                       565.49       281.09       846.58       1.0000          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np25-rf20 (query)                       565.49       422.26       987.75       1.0000          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np25-rfB (query)                        565.49       184.74       750.22       0.9999          1.0000            1.0000         7.66
IVF-RaBitQ-nl316 (self)                                  565.49       531.60     1_097.09       0.9999          1.0000            1.0000         7.66
IVF-RaBitQ-nl223-ex2-np14 (query)                        547.11       182.62       729.73       0.9645          1.0018            1.0000        16.49
IVF-RaBitQ-nl223-ex2-np21 (query)                        547.11       200.10       747.21       0.9645          1.0018            1.0000        16.49
IVF-RaBitQ-nl223-ex4-np14 (query)                        588.38       195.56       783.94       0.9777          1.0007            1.0000        25.64
IVF-RaBitQ-nl223-ex4-np21 (query)                        588.38       215.38       803.76       0.9777          1.0007            1.0000        25.64
IVF-RaBitQ-nl223-ex8-np14 (query)                        659.00       187.88       846.88       0.9835          1.0004            1.0000        43.95
IVF-RaBitQ-nl223-ex8-np21 (query)                        659.00       200.62       859.62       0.9835          1.0004            1.0000        43.95
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
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 256D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.64       665.15       697.79       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.64     2_187.86     2_220.50       1.0000          1.0000            1.0000        48.83
QG-d32-l64-ef15 (query)                                2_438.57       281.15     2_719.73       0.9879          1.3882            1.0000       116.16
QG-d32-l64-ef30 (query)                                2_438.57       399.97     2_838.55       0.9943          1.1239            1.0000       116.16
QG-d32-l64-ef60 (query)                                2_438.57       575.98     3_014.56       0.9969          1.0094            1.0000       116.16
QG-d32-l64-ef120 (query)                               2_438.57       852.21     3_290.78       0.9979          1.0018            1.0000       116.16
QG-d32-l64 (self)                                      2_438.57     2_030.68     4_469.25       0.9970          1.0074            1.0000       116.16
QG-d32-l128-ef15 (query)                               2_767.68       281.07     3_048.75       0.9921          1.0018            1.0000       116.16
QG-d32-l128-ef30 (query)                               2_767.68       400.31     3_167.99       0.9964          1.0015            1.0000       116.16
QG-d32-l128-ef60 (query)                               2_767.68       586.99     3_354.68       0.9981          1.0013            1.0000       116.16
QG-d32-l128-ef120 (query)                              2_767.68       845.62     3_613.30       0.9986          1.0012            1.0000       116.16
QG-d32-l128 (self)                                     2_767.68     1_875.55     4_643.23       0.9980          1.0013            1.0000       116.16
QG-d64-l64-ef15 (query)                                8_320.10       632.19     8_952.28       0.9995          1.0002            1.0000       183.30
QG-d64-l64-ef30 (query)                                8_320.10       861.95     9_182.05       0.9997          1.0002            1.0000       183.30
QG-d64-l64-ef60 (query)                                8_320.10     1_208.02     9_528.12       0.9998          1.0001            1.0000       183.30
QG-d64-l64-ef120 (query)                               8_320.10     1_700.93    10_021.03       0.9999          1.0001            1.0000       183.30
QG-d64-l64 (self)                                      8_320.10     3_942.35    12_262.45       0.9998          1.0002            1.0000       183.30
QG-d64-l128-ef15 (query)                               9_064.26       636.86     9_701.11       0.9996          1.0002            1.0000       183.30
QG-d64-l128-ef30 (query)                               9_064.26       863.58     9_927.84       0.9998          1.0002            1.0000       183.30
QG-d64-l128-ef60 (query)                               9_064.26     1_195.14    10_259.40       0.9998          1.0001            1.0000       183.30
QG-d64-l128-ef120 (query)                              9_064.26     1_712.94    10_777.20       0.9999          1.0001            1.0000       183.30
QG-d64-l128 (self)                                     9_064.26     4_271.64    13_335.90       0.9998          1.0002            1.0000       183.30
HnswRaBitQ-m16-ex1-ef15 (query)                        1_150.98        89.05     1_240.02       0.7508         59.0415            1.0084        10.85
HnswRaBitQ-m16-ex1-ef30 (query)                        1_150.98       105.82     1_256.79       0.7582         58.7451            1.0081        10.85
HnswRaBitQ-m16-ex1-ef60 (query)                        1_150.98       131.75     1_282.72       0.7599         58.4871            1.0081        10.85
HnswRaBitQ-m16-ex1-ef120 (query)                       1_150.98       176.96     1_327.94       0.7791          1.0088            1.0079        10.85
HnswRaBitQ-m16-ex1 (self)                              1_150.98     4_712.09     5_863.07       0.6949          9.1160            1.0160        10.85
HnswRaBitQ-m16-ex3-ef15 (query)                        1_228.85        92.76     1_321.61       0.9003          1.5269            1.0010        13.90
HnswRaBitQ-m16-ex3-ef30 (query)                        1_228.85       115.51     1_344.36       0.9195          1.1496            1.0007        13.90
HnswRaBitQ-m16-ex3-ef60 (query)                        1_228.85       146.43     1_375.28       0.9267          1.0019            1.0005        13.90
HnswRaBitQ-m16-ex3-ef120 (query)                       1_228.85       182.26     1_411.11       0.9283          1.0014            1.0005        13.90
HnswRaBitQ-m16-ex3 (self)                              1_228.85     4_769.07     5_997.92       0.9033          1.0021            1.0012        13.90
HnswRaBitQ-m16-ex5-ef15 (query)                        1_320.40       101.66     1_422.06       0.9375          1.1652            1.0001        16.95
HnswRaBitQ-m16-ex5-ef30 (query)                        1_320.40       127.38     1_447.78       0.9646          1.0037            1.0000        16.95
HnswRaBitQ-m16-ex5-ef60 (query)                        1_320.40       157.62     1_478.02       0.9738          1.0012            1.0000        16.95
HnswRaBitQ-m16-ex5-ef120 (query)                       1_320.40       203.45     1_523.85       0.9761          1.0005            1.0000        16.95
HnswRaBitQ-m16-ex5 (self)                              1_320.40     4_894.33     6_214.72       0.9695          1.0011            1.0000        16.95
HnswRaBitQ-m16-ex8-ef15 (query)                        2_021.38        97.56     2_118.94       0.9474          1.0292            1.0000        21.53
HnswRaBitQ-m16-ex8-ef30 (query)                        2_021.38       122.30     2_143.69       0.9783          1.0094            1.0000        21.53
HnswRaBitQ-m16-ex8-ef60 (query)                        2_021.38       155.83     2_177.22       0.9906          1.0013            1.0000        21.53
HnswRaBitQ-m16-ex8-ef120 (query)                       2_021.38       196.48     2_217.86       0.9938          1.0005            1.0000        21.53
HnswRaBitQ-m16-ex8 (self)                              2_021.38     4_868.28     6_889.66       0.9926          1.0011            1.0000        21.53
HnswRaBitQ-m32-ex1-ef15 (query)                        1_173.91        97.52     1_271.42       0.7776          1.0141            1.0079        16.95
HnswRaBitQ-m32-ex1-ef30 (query)                        1_173.91       122.24     1_296.15       0.7789          1.0097            1.0079        16.95
HnswRaBitQ-m32-ex1-ef60 (query)                        1_173.91       148.62     1_322.52       0.7789          1.0085            1.0079        16.95
HnswRaBitQ-m32-ex1-ef120 (query)                       1_173.91       189.16     1_363.06       0.7793          1.0083            1.0079        16.95
HnswRaBitQ-m32-ex1 (self)                              1_173.91     5_787.28     6_961.19       0.6970          1.0194            1.0160        16.95
HnswRaBitQ-m32-ex3-ef15 (query)                        1_251.10       107.00     1_358.11       0.9157          1.0095            1.0007        20.00
HnswRaBitQ-m32-ex3-ef30 (query)                        1_251.10       131.56     1_382.66       0.9253          1.0026            1.0006        20.00
HnswRaBitQ-m32-ex3-ef60 (query)                        1_251.10       173.79     1_424.90       0.9281          1.0011            1.0005        20.00
HnswRaBitQ-m32-ex3-ef120 (query)                       1_251.10       211.37     1_462.48       0.9289          1.0008            1.0005        20.00
HnswRaBitQ-m32-ex3 (self)                              1_251.10     5_835.49     7_086.59       0.9042          1.0016            1.0011        20.00
HnswRaBitQ-m32-ex5-ef15 (query)                        1_356.61       115.43     1_472.04       0.9575          1.0054            1.0000        23.06
HnswRaBitQ-m32-ex5-ef30 (query)                        1_356.61       142.36     1_498.97       0.9717          1.0018            1.0000        23.06
HnswRaBitQ-m32-ex5-ef60 (query)                        1_356.61       174.74     1_531.35       0.9756          1.0006            1.0000        23.06
HnswRaBitQ-m32-ex5-ef120 (query)                       1_356.61       218.55     1_575.16       0.9765          1.0003            1.0000        23.06
HnswRaBitQ-m32-ex5 (self)                              1_356.61     6_034.37     7_390.99       0.9710          1.0005            1.0000        23.06
HnswRaBitQ-m32-ex8-ef15 (query)                        2_056.19       112.60     2_168.79       0.9729          1.0077            1.0000        27.63
HnswRaBitQ-m32-ex8-ef30 (query)                        2_056.19       148.10     2_204.29       0.9891          1.0023            1.0000        27.63
HnswRaBitQ-m32-ex8-ef60 (query)                        2_056.19       171.44     2_227.63       0.9937          1.0001            1.0000        27.63
HnswRaBitQ-m32-ex8-ef120 (query)                       2_056.19       216.24     2_272.43       0.9944          1.0001            1.0000        27.63
HnswRaBitQ-m32-ex8 (self)                              2_056.19     6_019.68     8_075.87       0.9951          1.0002            1.0000        27.63
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 512D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.79     1_295.47     1_363.26       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         67.79     4_417.92     4_485.71       1.0000          1.0000            1.0000        97.66
QG-d32-l64-ef15 (query)                                4_759.34       492.45     5_251.79       0.9865          1.0038            1.0000       213.81
QG-d32-l64-ef30 (query)                                4_759.34       652.00     5_411.34       0.9917          1.0035            1.0000       213.81
QG-d32-l64-ef60 (query)                                4_759.34       884.66     5_644.00       0.9940          1.0034            1.0000       213.81
QG-d32-l64-ef120 (query)                               4_759.34     1_205.98     5_965.33       0.9951          1.0033            1.0000       213.81
QG-d32-l64 (self)                                      4_759.34     2_883.16     7_642.50       0.9939          1.0035            1.0000       213.81
QG-d32-l128-ef15 (query)                               5_408.54       482.95     5_891.49       0.9882          1.0034            1.0000       213.81
QG-d32-l128-ef30 (query)                               5_408.54       647.21     6_055.75       0.9932          1.0032            1.0000       213.81
QG-d32-l128-ef60 (query)                               5_408.54       875.30     6_283.84       0.9954          1.0030            1.0000       213.81
QG-d32-l128-ef120 (query)                              5_408.54     1_199.58     6_608.12       0.9963          1.0029            1.0000       213.81
QG-d32-l128 (self)                                     5_408.54     2_846.94     8_255.48       0.9953          1.0031            1.0000       213.81
QG-d64-l64-ef15 (query)                               17_676.47       975.64    18_652.11       0.9982          1.0009            1.0000       329.78
QG-d64-l64-ef30 (query)                               17_676.47     1_265.86    18_942.33       0.9987          1.0009            1.0000       329.78
QG-d64-l64-ef60 (query)                               17_676.47     1_662.71    19_339.19       0.9989          1.0008            1.0000       329.78
QG-d64-l64-ef120 (query)                              17_676.47     2_235.58    19_912.05       0.9990          1.0008            1.0000       329.78
QG-d64-l64 (self)                                     17_676.47     5_460.45    23_136.92       0.9989          1.0009            1.0000       329.78
QG-d64-l128-ef15 (query)                              18_770.79     1_093.85    19_864.64       0.9982          1.0013            1.0000       329.78
QG-d64-l128-ef30 (query)                              18_770.79     1_324.11    20_094.89       0.9987          1.0013            1.0000       329.78
QG-d64-l128-ef60 (query)                              18_770.79     1_995.73    20_766.51       0.9989          1.0012            1.0000       329.78
QG-d64-l128-ef120 (query)                             18_770.79     2_320.32    21_091.10       0.9989          1.0012            1.0000       329.78
QG-d64-l128 (self)                                    18_770.79     5_671.45    24_442.24       0.9987          1.0013            1.0000       329.78
HnswRaBitQ-m16-ex1-ef15 (query)                        2_203.08       161.05     2_364.13       0.7320        132.7705            1.0057        14.15
HnswRaBitQ-m16-ex1-ef30 (query)                        2_203.08       211.55     2_414.63       0.7403        132.6705            1.0055        14.15
HnswRaBitQ-m16-ex1-ef60 (query)                        2_203.08       235.18     2_438.26       0.7445        132.2136            1.0054        14.15
HnswRaBitQ-m16-ex1-ef120 (query)                       2_203.08       289.69     2_492.77       0.7456        132.2105            1.0054        14.15
HnswRaBitQ-m16-ex1 (self)                              2_203.08     8_166.39    10_369.47       0.6670        133.1145            1.0110        14.15
HnswRaBitQ-m16-ex3-ef15 (query)                        2_346.34       179.09     2_525.43       0.8889          1.6459            1.0008        20.25
HnswRaBitQ-m16-ex3-ef30 (query)                        2_346.34       219.20     2_565.54       0.9147          1.6134            1.0005        20.25
HnswRaBitQ-m16-ex3-ef60 (query)                        2_346.34       269.23     2_615.57       0.9244          1.6036            1.0004        20.25
HnswRaBitQ-m16-ex3-ef120 (query)                       2_346.34       328.36     2_674.70       0.9270          1.3972            1.0004        20.25
HnswRaBitQ-m16-ex3 (self)                              2_346.34     8_581.49    10_927.83       0.9003          1.3026            1.0008        20.25
HnswRaBitQ-m16-ex5-ef15 (query)                        2_535.63       194.53     2_730.16       0.9074         45.3449            1.0002        26.36
HnswRaBitQ-m16-ex5-ef30 (query)                        2_535.63       242.11     2_777.74       0.9569          1.0099            1.0000        26.36
HnswRaBitQ-m16-ex5-ef60 (query)                        2_535.63       294.66     2_830.29       0.9711          1.0022            1.0000        26.36
HnswRaBitQ-m16-ex5-ef120 (query)                       2_535.63       355.95     2_891.58       0.9748          1.0004            1.0000        26.36
HnswRaBitQ-m16-ex5 (self)                              2_535.63     8_981.22    11_516.85       0.9658          1.0016            1.0000        26.36
HnswRaBitQ-m16-ex8-ef15 (query)                        3_976.33       187.13     4_163.46       0.9341          1.0318            1.0000        35.51
HnswRaBitQ-m16-ex8-ef30 (query)                        3_976.33       228.67     4_205.00       0.9713          1.0033            1.0000        35.51
HnswRaBitQ-m16-ex8-ef60 (query)                        3_976.33       302.59     4_278.92       0.9886          1.0007            1.0000        35.51
HnswRaBitQ-m16-ex8-ef120 (query)                       3_976.33       342.03     4_318.36       0.9930          1.0004            1.0000        35.51
HnswRaBitQ-m16-ex8 (self)                              3_976.33     8_653.82    12_630.15       0.9898          1.0008            1.0000        35.51
HnswRaBitQ-m32-ex1-ef15 (query)                        2_258.52       185.09     2_443.61       0.7430        118.4893            1.0054        20.25
HnswRaBitQ-m32-ex1-ef30 (query)                        2_258.52       222.94     2_481.46       0.7462        118.4319            1.0053        20.25
HnswRaBitQ-m32-ex1-ef60 (query)                        2_258.52       280.13     2_538.65       0.7796          1.0058            1.0052        20.25
HnswRaBitQ-m32-ex1-ef120 (query)                       2_258.52       335.98     2_594.49       0.7798          1.0055            1.0052        20.25
HnswRaBitQ-m32-ex1 (self)                              2_258.52    10_553.36    12_811.88       0.6966          1.0115            1.0107        20.25
HnswRaBitQ-m32-ex3-ef15 (query)                        2_434.19       210.15     2_644.33       0.9121          1.0155            1.0005        26.36
HnswRaBitQ-m32-ex3-ef30 (query)                        2_434.19       256.19     2_690.38       0.9239          1.0021            1.0004        26.36
HnswRaBitQ-m32-ex3-ef60 (query)                        2_434.19       304.63     2_738.82       0.9272          1.0006            1.0004        26.36
HnswRaBitQ-m32-ex3-ef120 (query)                       2_434.19       365.02     2_799.21       0.9279          1.0006            1.0004        26.36
HnswRaBitQ-m32-ex3 (self)                              2_434.19    10_775.18    13_209.37       0.9026          1.0012            1.0008        26.36
HnswRaBitQ-m32-ex5-ef15 (query)                        2_631.42       231.51     2_862.93       0.9525          1.0820            1.0000        32.46
HnswRaBitQ-m32-ex5-ef30 (query)                        2_631.42       279.36     2_910.77       0.9693          1.0237            1.0000        32.46
HnswRaBitQ-m32-ex5-ef60 (query)                        2_631.42       344.86     2_976.28       0.9746          1.0003            1.0000        32.46
HnswRaBitQ-m32-ex5-ef120 (query)                       2_631.42       393.31     3_024.72       0.9756          1.0001            1.0000        32.46
HnswRaBitQ-m32-ex5 (self)                              2_631.42    11_093.56    13_724.98       0.9679          3.0744            1.0000        32.46
HnswRaBitQ-m32-ex8-ef15 (query)                        4_068.14       219.97     4_288.11       0.9661          1.0112            1.0000        41.62
HnswRaBitQ-m32-ex8-ef30 (query)                        4_068.14       266.84     4_334.98       0.9863          1.0007            1.0000        41.62
HnswRaBitQ-m32-ex8-ef60 (query)                        4_068.14       316.09     4_384.23       0.9926          1.0003            1.0000        41.62
HnswRaBitQ-m32-ex8-ef120 (query)                       4_068.14       377.90     4_446.04       0.9941          1.0001            1.0000        41.62
HnswRaBitQ-m32-ex8 (self)                              4_068.14    11_015.37    15_083.51       0.9939          1.0003            1.0000        41.62
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 768D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       100.41     1_895.37     1_995.78       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.41     6_370.07     6_470.48       1.0000          1.0000            1.0000       146.48
QG-d32-l64-ef15 (query)                                7_150.91       619.40     7_770.31       0.9841          1.0083            1.0000       311.47
QG-d32-l64-ef30 (query)                                7_150.91       815.51     7_966.42       0.9896          1.0081            1.0000       311.47
QG-d32-l64-ef60 (query)                                7_150.91     1_084.50     8_235.40       0.9923          1.0080            1.0000       311.47
QG-d32-l64-ef120 (query)                               7_150.91     1_445.72     8_596.63       0.9934          1.0079            1.0000       311.47
QG-d32-l64 (self)                                      7_150.91     3_512.56    10_663.47       0.9923          1.0075            1.0000       311.47
QG-d32-l128-ef15 (query)                               7_804.53       633.50     8_438.03       0.9853          1.0053            1.0000       311.47
QG-d32-l128-ef30 (query)                               7_804.53       828.99     8_633.52       0.9908          1.0051            1.0000       311.47
QG-d32-l128-ef60 (query)                               7_804.53     1_091.52     8_896.05       0.9933          1.0049            1.0000       311.47
QG-d32-l128-ef120 (query)                              7_804.53     1_461.46     9_265.99       0.9944          1.0048            1.0000       311.47
QG-d32-l128 (self)                                     7_804.53     3_542.60    11_347.13       0.9935          1.0047            1.0000       311.47
QG-d64-l64-ef15 (query)                               26_568.82     1_289.52    27_858.34       0.9971          1.0023            1.0000       476.27
QG-d64-l64-ef30 (query)                               26_568.82     1_624.50    28_193.33       0.9977          1.0023            1.0000       476.27
QG-d64-l64-ef60 (query)                               26_568.82     2_077.90    28_646.72       0.9980          1.0022            1.0000       476.27
QG-d64-l64-ef120 (query)                              26_568.82     2_721.65    29_290.47       0.9981          1.0022            1.0000       476.27
QG-d64-l64 (self)                                     26_568.82     6_848.81    33_417.63       0.9979          1.0035            1.0000       476.27
QG-d64-l128-ef15 (query)                              26_941.77     1_286.03    28_227.80       0.9972          1.0026            1.0000       476.27
QG-d64-l128-ef30 (query)                              26_941.77     1_630.32    28_572.10       0.9978          1.0026            1.0000       476.27
QG-d64-l128-ef60 (query)                              26_941.77     2_077.99    29_019.76       0.9981          1.0026            1.0000       476.27
QG-d64-l128-ef120 (query)                             26_941.77     2_729.63    29_671.40       0.9982          1.0025            1.0000       476.27
QG-d64-l128 (self)                                    26_941.77     6_875.91    33_817.68       0.9980          1.0029            1.0000       476.27
HnswRaBitQ-m16-ex1-ef15 (query)                        3_052.97       232.26     3_285.22       0.7636          1.0467            1.0045        17.45
HnswRaBitQ-m16-ex1-ef30 (query)                        3_052.97       280.45     3_333.42       0.7739          1.0211            1.0042        17.45
HnswRaBitQ-m16-ex1-ef60 (query)                        3_052.97       342.87     3_395.83       0.7785          1.0057            1.0041        17.45
HnswRaBitQ-m16-ex1-ef120 (query)                       3_052.97       420.69     3_473.65       0.7793          1.0051            1.0041        17.45
HnswRaBitQ-m16-ex1 (self)                              3_052.97    13_788.60    16_841.56       0.6952          1.0114            1.0085        17.45
HnswRaBitQ-m16-ex3-ef15 (query)                        3_325.93       259.23     3_585.16       0.8873          1.1647            1.0007        26.61
HnswRaBitQ-m16-ex3-ef30 (query)                        3_325.93       313.01     3_638.94       0.9138          1.0687            1.0004        26.61
HnswRaBitQ-m16-ex3-ef60 (query)                        3_325.93       383.79     3_709.72       0.9247          1.0013            1.0003        26.61
HnswRaBitQ-m16-ex3-ef120 (query)                       3_325.93       464.88     3_790.81       0.9276          1.0007            1.0003        26.61
HnswRaBitQ-m16-ex3 (self)                              3_325.93    14_078.72    17_404.64       0.8998          1.0014            1.0007        26.61
HnswRaBitQ-m16-ex5-ef15 (query)                        3_623.99       279.97     3_903.96       0.9162          1.0807            1.0002        35.76
HnswRaBitQ-m16-ex5-ef30 (query)                        3_623.99       340.88     3_964.86       0.9536          1.0127            1.0000        35.76
HnswRaBitQ-m16-ex5-ef60 (query)                        3_623.99       414.32     4_038.30       0.9703          1.0007            1.0000        35.76
HnswRaBitQ-m16-ex5-ef120 (query)                       3_623.99       497.22     4_121.21       0.9743          1.0004            1.0000        35.76
HnswRaBitQ-m16-ex5 (self)                              3_623.99    14_404.05    18_028.03       0.9646          1.0015            1.0000        35.76
HnswRaBitQ-m16-ex8-ef15 (query)                        5_775.38       273.61     6_048.99       0.9245          1.4514            1.0001        49.50
HnswRaBitQ-m16-ex8-ef30 (query)                        5_775.38       317.96     6_093.34       0.9679          1.0038            1.0000        49.50
HnswRaBitQ-m16-ex8-ef60 (query)                        5_775.38       390.27     6_165.65       0.9866          1.0009            1.0000        49.50
HnswRaBitQ-m16-ex8-ef120 (query)                       5_775.38       469.51     6_244.89       0.9922          1.0004            1.0000        49.50
HnswRaBitQ-m16-ex8 (self)                              5_775.38    13_952.76    19_728.14       0.9888          1.0009            1.0000        49.50
HnswRaBitQ-m32-ex1-ef15 (query)                        3_119.11       269.31     3_388.42       0.7745          1.0933            1.0042        23.56
HnswRaBitQ-m32-ex1-ef30 (query)                        3_119.11       330.48     3_449.60       0.7785          1.0177            1.0041        23.56
HnswRaBitQ-m32-ex1-ef60 (query)                        3_119.11       388.09     3_507.20       0.7794          1.0047            1.0041        23.56
HnswRaBitQ-m32-ex1-ef120 (query)                       3_119.11       462.77     3_581.89       0.7798          1.0043            1.0041        23.56
HnswRaBitQ-m32-ex1 (self)                              3_119.11    17_149.07    20_268.18       0.6953          1.0090            1.0086        23.56
HnswRaBitQ-m32-ex3-ef15 (query)                        3_382.97       303.84     3_686.81       0.9103          1.3084            1.0004        32.71
HnswRaBitQ-m32-ex3-ef30 (query)                        3_382.97       365.82     3_748.79       0.9230          1.0019            1.0003        32.71
HnswRaBitQ-m32-ex3-ef60 (query)                        3_382.97       430.91     3_813.88       0.9271          1.0006            1.0003        32.71
HnswRaBitQ-m32-ex3-ef120 (query)                       3_382.97       510.74     3_893.72       0.9280          1.0005            1.0003        32.71
HnswRaBitQ-m32-ex3 (self)                              3_382.97    17_621.20    21_004.17       0.9017          1.0010            1.0006        32.71
HnswRaBitQ-m32-ex5-ef15 (query)                        3_679.50       333.61     4_013.11       0.9498          1.3401            1.0000        41.87
HnswRaBitQ-m32-ex5-ef30 (query)                        3_679.50       396.29     4_075.79       0.9677          1.2712            1.0000        41.87
HnswRaBitQ-m32-ex5-ef60 (query)                        3_679.50       465.51     4_145.02       0.9736          1.1802            1.0000        41.87
HnswRaBitQ-m32-ex5-ef120 (query)                       3_679.50       543.34     4_222.84       0.9754          1.0001            1.0000        41.87
HnswRaBitQ-m32-ex5 (self)                              3_679.50    18_255.60    21_935.11       0.9680          1.0003            1.0000        41.87
HnswRaBitQ-m32-ex8-ef15 (query)                        5_837.19       306.54     6_143.73       0.9605          1.4464            1.0000        55.60
HnswRaBitQ-m32-ex8-ef30 (query)                        5_837.19       368.75     6_205.94       0.9832          1.2867            1.0000        55.60
HnswRaBitQ-m32-ex8-ef60 (query)                        5_837.19       465.77     6_302.96       0.9915          1.1273            1.0000        55.60
HnswRaBitQ-m32-ex8-ef120 (query)                       5_837.19       513.88     6_351.07       0.9932          1.0001            1.0000        55.60
HnswRaBitQ-m32-ex8 (self)                              5_837.19    16_153.55    21_990.73       0.9936          1.0002            1.0000        55.60
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 256D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.42       723.00       755.42       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.42     2_417.65     2_450.07       1.0000          1.0000            1.0000        48.83
QG-d32-l64-ef15 (query)                                3_016.97        81.67     3_098.65       0.9732          1.0008            1.0000       116.16
QG-d32-l64-ef30 (query)                                3_016.97       141.77     3_158.75       0.9986          1.0000            1.0000       116.16
QG-d32-l64-ef60 (query)                                3_016.97       249.93     3_266.91       0.9999          1.0000            1.0000       116.16
QG-d32-l64-ef120 (query)                               3_016.97       473.87     3_490.85       1.0000          1.0000            1.0000       116.16
QG-d32-l64 (self)                                      3_016.97       797.95     3_814.93       1.0000          1.0000            1.0000       116.16
QG-d32-l128-ef15 (query)                               3_446.88        78.40     3_525.28       0.9729          1.0008            1.0000       116.16
QG-d32-l128-ef30 (query)                               3_446.88       134.44     3_581.32       0.9986          1.0000            1.0000       116.16
QG-d32-l128-ef60 (query)                               3_446.88       248.36     3_695.24       1.0000          1.0000            1.0000       116.16
QG-d32-l128-ef120 (query)                              3_446.88       478.42     3_925.30       1.0000          1.0000            1.0000       116.16
QG-d32-l128 (self)                                     3_446.88       762.79     4_209.67       1.0000          1.0000            1.0000       116.16
QG-d64-l64-ef15 (query)                                6_154.33       120.79     6_275.12       0.9941          1.0001            1.0000       183.30
QG-d64-l64-ef30 (query)                                6_154.33       222.45     6_376.78       1.0000          1.0000            1.0000       183.30
QG-d64-l64-ef60 (query)                                6_154.33       423.93     6_578.27       1.0000          1.0000            1.0000       183.30
QG-d64-l64-ef120 (query)                               6_154.33       813.48     6_967.82       1.0000          1.0000            1.0000       183.30
QG-d64-l64 (self)                                      6_154.33     1_331.25     7_485.58       1.0000          1.0000            1.0000       183.30
QG-d64-l128-ef15 (query)                               6_738.27       121.96     6_860.22       0.9941          1.0001            1.0000       183.30
QG-d64-l128-ef30 (query)                               6_738.27       223.10     6_961.36       1.0000          1.0000            1.0000       183.30
QG-d64-l128-ef60 (query)                               6_738.27       425.96     7_164.22       1.0000          1.0000            1.0000       183.30
QG-d64-l128-ef120 (query)                              6_738.27       826.46     7_564.73       1.0000          1.0000            1.0000       183.30
QG-d64-l128 (self)                                     6_738.27     1_326.74     8_065.00       1.0000          1.0000            1.0000       183.30
HnswRaBitQ-m16-ex1-ef15 (query)                        1_207.93        82.20     1_290.13       0.8537          1.0073            1.0059        10.85
HnswRaBitQ-m16-ex1-ef30 (query)                        1_207.93       100.70     1_308.62       0.8661          1.0057            1.0049        10.85
HnswRaBitQ-m16-ex1-ef60 (query)                        1_207.93       128.17     1_336.10       0.8692          1.0053            1.0046        10.85
HnswRaBitQ-m16-ex1-ef120 (query)                       1_207.93       164.66     1_372.58       0.8695          1.0053            1.0046        10.85
HnswRaBitQ-m16-ex1 (self)                              1_207.93     5_622.60     6_830.53       0.8312          1.0108            1.0097        10.85
HnswRaBitQ-m16-ex3-ef15 (query)                        1_280.30        78.92     1_359.22       0.9278          1.0028            1.0009        13.90
HnswRaBitQ-m16-ex3-ef30 (query)                        1_280.30        97.73     1_378.02       0.9504          1.0010            1.0003        13.90
HnswRaBitQ-m16-ex3-ef60 (query)                        1_280.30       125.30     1_405.60       0.9565          1.0006            1.0001        13.90
HnswRaBitQ-m16-ex3-ef120 (query)                       1_280.30       164.81     1_445.11       0.9575          1.0005            1.0001        13.90
HnswRaBitQ-m16-ex3 (self)                              1_280.30     5_597.21     6_877.51       0.9469          1.0010            1.0005        13.90
HnswRaBitQ-m16-ex5-ef15 (query)                        1_378.44        85.29     1_463.73       0.9474          1.0025            1.0001        16.95
HnswRaBitQ-m16-ex5-ef30 (query)                        1_378.44       107.37     1_485.81       0.9767          1.0006            1.0000        16.95
HnswRaBitQ-m16-ex5-ef60 (query)                        1_378.44       134.53     1_512.97       0.9853          1.0001            1.0000        16.95
HnswRaBitQ-m16-ex5-ef120 (query)                       1_378.44       175.21     1_553.65       0.9867          1.0001            1.0000        16.95
HnswRaBitQ-m16-ex5 (self)                              1_378.44     5_760.34     7_138.78       0.9837          1.0002            1.0000        16.95
HnswRaBitQ-m16-ex8-ef15 (query)                        2_071.78        83.94     2_155.73       0.9531          1.0023            1.0000        21.53
HnswRaBitQ-m16-ex8-ef30 (query)                        2_071.78       105.05     2_176.83       0.9856          1.0006            1.0000        21.53
HnswRaBitQ-m16-ex8-ef60 (query)                        2_071.78       132.61     2_204.39       0.9956          1.0001            1.0000        21.53
HnswRaBitQ-m16-ex8-ef120 (query)                       2_071.78       174.87     2_246.66       0.9971          1.0001            1.0000        21.53
HnswRaBitQ-m16-ex8 (self)                              2_071.78     5_802.25     7_874.03       0.9965          1.0001            1.0000        21.53
HnswRaBitQ-m32-ex1-ef15 (query)                        1_229.68        86.80     1_316.48       0.8623          1.0062            1.0052        16.95
HnswRaBitQ-m32-ex1-ef30 (query)                        1_229.68       107.83     1_337.52       0.8683          1.0054            1.0047        16.95
HnswRaBitQ-m32-ex1-ef60 (query)                        1_229.68       136.10     1_365.79       0.8696          1.0052            1.0046        16.95
HnswRaBitQ-m32-ex1-ef120 (query)                       1_229.68       177.40     1_407.09       0.8697          1.0052            1.0046        16.95
HnswRaBitQ-m32-ex1 (self)                              1_229.68     6_618.07     7_847.75       0.8314          1.0108            1.0097        16.95
HnswRaBitQ-m32-ex3-ef15 (query)                        1_304.52        92.39     1_396.91       0.9417          1.0017            1.0004        20.00
HnswRaBitQ-m32-ex3-ef30 (query)                        1_304.52       118.98     1_423.50       0.9547          1.0007            1.0002        20.00
HnswRaBitQ-m32-ex3-ef60 (query)                        1_304.52       146.15     1_450.67       0.9573          1.0005            1.0001        20.00
HnswRaBitQ-m32-ex3-ef120 (query)                       1_304.52       188.14     1_492.66       0.9576          1.0005            1.0001        20.00
HnswRaBitQ-m32-ex3 (self)                              1_304.52     6_693.51     7_998.03       0.9475          1.0009            1.0005        20.00
HnswRaBitQ-m32-ex5-ef15 (query)                        1_407.07        99.50     1_506.57       0.9664          1.0013            1.0000        23.06
HnswRaBitQ-m32-ex5-ef30 (query)                        1_407.07       121.19     1_528.25       0.9827          1.0003            1.0000        23.06
HnswRaBitQ-m32-ex5-ef60 (query)                        1_407.07       151.38     1_558.45       0.9864          1.0001            1.0000        23.06
HnswRaBitQ-m32-ex5-ef120 (query)                       1_407.07       194.73     1_601.80       0.9868          1.0001            1.0000        23.06
HnswRaBitQ-m32-ex5 (self)                              1_407.07     6_858.49     8_265.56       0.9845          1.0001            1.0000        23.06
HnswRaBitQ-m32-ex8-ef15 (query)                        2_119.64        96.66     2_216.30       0.9742          1.0012            1.0000        27.63
HnswRaBitQ-m32-ex8-ef30 (query)                        2_119.64       120.28     2_239.92       0.9930          1.0002            1.0000        27.63
HnswRaBitQ-m32-ex8-ef60 (query)                        2_119.64       159.68     2_279.32       0.9967          1.0001            1.0000        27.63
HnswRaBitQ-m32-ex8-ef120 (query)                       2_119.64       194.95     2_314.59       0.9974          1.0000            1.0000        27.63
HnswRaBitQ-m32-ex8 (self)                              2_119.64     7_097.05     9_216.69       0.9975          1.0000            1.0000        27.63
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 512D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.80     1_295.63     1_364.43       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.80     4_397.89     4_466.69       1.0000          1.0000            1.0000        97.66
QG-d32-l64-ef15 (query)                                5_796.25       139.59     5_935.84       0.9604          1.0033            1.0000       213.81
QG-d32-l64-ef30 (query)                                5_796.25       228.25     6_024.49       0.9922          1.0013            1.0000       213.81
QG-d32-l64-ef60 (query)                                5_796.25       398.78     6_195.03       0.9981          1.0006            1.0000       213.81
QG-d32-l64-ef120 (query)                               5_796.25       715.35     6_511.60       0.9993          1.0003            1.0000       213.81
QG-d32-l64 (self)                                      5_796.25     1_271.32     7_067.57       0.9982          1.0007            1.0000       213.81
QG-d32-l128-ef15 (query)                               6_664.52       142.44     6_806.96       0.9623          1.0029            1.0000       213.81
QG-d32-l128-ef30 (query)                               6_664.52       231.37     6_895.89       0.9935          1.0012            1.0000       213.81
QG-d32-l128-ef60 (query)                               6_664.52       399.69     7_064.21       0.9988          1.0006            1.0000       213.81
QG-d32-l128-ef120 (query)                              6_664.52       710.44     7_374.96       0.9996          1.0003            1.0000       213.81
QG-d32-l128 (self)                                     6_664.52     1_216.24     7_880.76       0.9988          1.0006            1.0000       213.81
QG-d64-l64-ef15 (query)                               16_994.61       222.44    17_217.05       0.9931          1.0002            1.0000       329.78
QG-d64-l64-ef30 (query)                               16_994.61       388.04    17_382.65       0.9997          1.0000            1.0000       329.78
QG-d64-l64-ef60 (query)                               16_994.61       688.33    17_682.94       1.0000          1.0000            1.0000       329.78
QG-d64-l64-ef120 (query)                              16_994.61     1_224.51    18_219.12       1.0000          1.0000            1.0000       329.78
QG-d64-l64 (self)                                     16_994.61     2_130.52    19_125.14       1.0000          1.0000            1.0000       329.78
QG-d64-l128-ef15 (query)                              17_659.78       218.32    17_878.09       0.9930          1.0002            1.0000       329.78
QG-d64-l128-ef30 (query)                              17_659.78       381.54    18_041.31       0.9998          1.0000            1.0000       329.78
QG-d64-l128-ef60 (query)                              17_659.78       687.44    18_347.21       1.0000          1.0000            1.0000       329.78
QG-d64-l128-ef120 (query)                             17_659.78     1_227.07    18_886.85       1.0000          1.0000            1.0000       329.78
QG-d64-l128 (self)                                    17_659.78     2_126.66    19_786.43       1.0000          1.0000            1.0000       329.78
HnswRaBitQ-m16-ex1-ef15 (query)                        2_304.71       146.42     2_451.13       0.8467          1.0065            1.0043        14.15
HnswRaBitQ-m16-ex1-ef30 (query)                        2_304.71       182.47     2_487.18       0.8672          1.0042            1.0033        14.15
HnswRaBitQ-m16-ex1-ef60 (query)                        2_304.71       217.72     2_522.43       0.8737          1.0035            1.0029        14.15
HnswRaBitQ-m16-ex1-ef120 (query)                       2_304.71       272.88     2_577.59       0.8750          1.0034            1.0029        14.15
HnswRaBitQ-m16-ex1 (self)                              2_304.71    10_823.12    13_127.82       0.8357          1.0068            1.0060        14.15
HnswRaBitQ-m16-ex3-ef15 (query)                        2_471.50       160.91     2_632.41       0.9092          1.0039            1.0011        20.25
HnswRaBitQ-m16-ex3-ef30 (query)                        2_471.50       190.17     2_661.67       0.9442          1.0014            1.0003        20.25
HnswRaBitQ-m16-ex3-ef60 (query)                        2_471.50       235.52     2_707.02       0.9561          1.0006            1.0001        20.25
HnswRaBitQ-m16-ex3-ef120 (query)                       2_471.50       293.43     2_764.93       0.9584          1.0005            1.0001        20.25
HnswRaBitQ-m16-ex3 (self)                              2_471.50    11_254.90    13_726.40       0.9463          1.0008            1.0003        20.25
HnswRaBitQ-m16-ex5-ef15 (query)                        2_665.43       171.42     2_836.85       0.9244          1.0034            1.0007        26.36
HnswRaBitQ-m16-ex5-ef30 (query)                        2_665.43       200.44     2_865.88       0.9665          1.0010            1.0000        26.36
HnswRaBitQ-m16-ex5-ef60 (query)                        2_665.43       246.07     2_911.50       0.9823          1.0003            1.0000        26.36
HnswRaBitQ-m16-ex5-ef120 (query)                       2_665.43       305.08     2_970.51       0.9858          1.0002            1.0000        26.36
HnswRaBitQ-m16-ex5 (self)                              2_665.43    11_753.17    14_418.60       0.9811          1.0004            1.0000        26.36
HnswRaBitQ-m16-ex8-ef15 (query)                        4_107.33       166.96     4_274.29       0.9283          1.0035            1.0006        35.51
HnswRaBitQ-m16-ex8-ef30 (query)                        4_107.33       196.11     4_303.44       0.9741          1.0010            1.0000        35.51
HnswRaBitQ-m16-ex8-ef60 (query)                        4_107.33       241.25     4_348.58       0.9922          1.0003            1.0000        35.51
HnswRaBitQ-m16-ex8-ef120 (query)                       4_107.33       302.29     4_409.62       0.9962          1.0002            1.0000        35.51
HnswRaBitQ-m16-ex8 (self)                              4_107.33    11_526.08    15_633.41       0.9937          1.0003            1.0000        35.51
HnswRaBitQ-m32-ex1-ef15 (query)                        2_341.88       169.04     2_510.93       0.8629          1.0046            1.0034        20.25
HnswRaBitQ-m32-ex1-ef30 (query)                        2_341.88       203.50     2_545.39       0.8727          1.0035            1.0030        20.25
HnswRaBitQ-m32-ex1-ef60 (query)                        2_341.88       249.55     2_591.43       0.8750          1.0033            1.0029        20.25
HnswRaBitQ-m32-ex1-ef120 (query)                       2_341.88       311.22     2_653.10       0.8752          1.0032            1.0028        20.25
HnswRaBitQ-m32-ex1 (self)                              2_341.88    13_507.48    15_849.36       0.8364          1.0066            1.0059        20.25
HnswRaBitQ-m32-ex3-ef15 (query)                        2_503.14       184.51     2_687.65       0.9369          1.0017            1.0004        26.36
HnswRaBitQ-m32-ex3-ef30 (query)                        2_503.14       219.31     2_722.45       0.9540          1.0006            1.0001        26.36
HnswRaBitQ-m32-ex3-ef60 (query)                        2_503.14       266.96     2_770.10       0.9583          1.0004            1.0001        26.36
HnswRaBitQ-m32-ex3-ef120 (query)                       2_503.14       323.89     2_827.03       0.9589          1.0003            1.0001        26.36
HnswRaBitQ-m32-ex3 (self)                              2_503.14    14_611.64    17_114.78       0.9482          1.0006            1.0003        26.36
HnswRaBitQ-m32-ex5-ef15 (query)                        2_707.99       200.44     2_908.43       0.9573          1.0019            1.0000        32.46
HnswRaBitQ-m32-ex5-ef30 (query)                        2_707.99       232.97     2_940.97       0.9798          1.0004            1.0000        32.46
HnswRaBitQ-m32-ex5-ef60 (query)                        2_707.99       281.10     2_989.09       0.9856          1.0001            1.0000        32.46
HnswRaBitQ-m32-ex5-ef120 (query)                       2_707.99       339.51     3_047.51       0.9865          1.0001            1.0000        32.46
HnswRaBitQ-m32-ex5 (self)                              2_707.99    14_541.28    17_249.28       0.9837          1.0001            1.0000        32.46
HnswRaBitQ-m32-ex8-ef15 (query)                        4_191.14       190.53     4_381.67       0.9648          1.0012            1.0000        41.62
HnswRaBitQ-m32-ex8-ef30 (query)                        4_191.14       226.43     4_417.56       0.9898          1.0003            1.0000        41.62
HnswRaBitQ-m32-ex8-ef60 (query)                        4_191.14       279.83     4_470.97       0.9963          1.0001            1.0000        41.62
HnswRaBitQ-m32-ex8-ef120 (query)                       4_191.14       332.01     4_523.15       0.9971          1.0001            1.0000        41.62
HnswRaBitQ-m32-ex8 (self)                              4_191.14    14_209.23    18_400.37       0.9970          1.0001            1.0000        41.62
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 768D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       100.45     1_857.04     1_957.49       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.45     6_233.52     6_333.97       1.0000          1.0000            1.0000       146.48
QG-d32-l64-ef15 (query)                                7_755.28       243.49     7_998.77       0.9517          1.0067            1.0000       311.47
QG-d32-l64-ef30 (query)                                7_755.28       393.04     8_148.33       0.9812          1.0050            1.0000       311.47
QG-d32-l64-ef60 (query)                                7_755.28       639.02     8_394.31       0.9907          1.0042            1.0000       311.47
QG-d32-l64-ef120 (query)                               7_755.28     1_036.04     8_791.33       0.9939          1.0037            1.0000       311.47
QG-d32-l64 (self)                                      7_755.28     2_025.14     9_780.42       0.9910          1.0044            1.0000       311.47
QG-d32-l128-ef15 (query)                               8_650.23       245.67     8_895.89       0.9556          1.0061            1.0000       311.47
QG-d32-l128-ef30 (query)                               8_650.23       399.26     9_049.49       0.9844          1.0044            1.0000       311.47
QG-d32-l128-ef60 (query)                               8_650.23       653.99     9_304.22       0.9928          1.0036            1.0000       311.47
QG-d32-l128-ef120 (query)                              8_650.23     1_056.49     9_706.72       0.9953          1.0030            1.0000       311.47
QG-d32-l128 (self)                                     8_650.23     2_114.43    10_764.66       0.9930          1.0038            1.0000       311.47
QG-d64-l64-ef15 (query)                               30_060.54       479.09    30_539.63       0.9930          1.0008            1.0000       476.27
QG-d64-l64-ef30 (query)                               30_060.54       780.01    30_840.55       0.9986          1.0004            1.0000       476.27
QG-d64-l64-ef60 (query)                               30_060.54     1_247.95    31_308.49       0.9996          1.0002            1.0000       476.27
QG-d64-l64-ef120 (query)                              30_060.54     1_997.62    32_058.16       0.9999          1.0001            1.0000       476.27
QG-d64-l64 (self)                                     30_060.54     4_006.19    34_066.73       0.9997          1.0002            1.0000       476.27
QG-d64-l128-ef15 (query)                              29_764.28       478.57    30_242.85       0.9932          1.0008            1.0000       476.27
QG-d64-l128-ef30 (query)                              29_764.28       783.50    30_547.78       0.9986          1.0004            1.0000       476.27
QG-d64-l128-ef60 (query)                              29_764.28     1_245.71    31_009.99       0.9996          1.0002            1.0000       476.27
QG-d64-l128-ef120 (query)                             29_764.28     1_979.82    31_744.09       0.9999          1.0001            1.0000       476.27
QG-d64-l128 (self)                                    29_764.28     4_008.59    33_772.86       0.9997          1.0001            1.0000       476.27
HnswRaBitQ-m16-ex1-ef15 (query)                        3_198.07       210.62     3_408.69       0.7563        346.6193            1.0043        17.45
HnswRaBitQ-m16-ex1-ef30 (query)                        3_198.07       265.13     3_463.20       0.8140        164.4257            1.0030        17.45
HnswRaBitQ-m16-ex1-ef60 (query)                        3_198.07       351.88     3_549.95       0.8578         22.2250            1.0025        17.45
HnswRaBitQ-m16-ex1-ef120 (query)                       3_198.07       423.39     3_621.46       0.8652          1.0030            1.0024        17.45
HnswRaBitQ-m16-ex1 (self)                              3_198.07    18_060.14    21_258.21       0.8159          7.7987            1.0051        17.45
HnswRaBitQ-m16-ex3-ef15 (query)                        3_652.50       244.81     3_897.31       0.8144        335.9227            1.0016        26.61
HnswRaBitQ-m16-ex3-ef30 (query)                        3_652.50       292.78     3_945.28       0.8920        142.2620            1.0004        26.61
HnswRaBitQ-m16-ex3-ef60 (query)                        3_652.50       366.99     4_019.49       0.9476          1.6393            1.0001        26.61
HnswRaBitQ-m16-ex3-ef120 (query)                       3_652.50       445.37     4_097.87       0.9545          1.0005            1.0001        26.61
HnswRaBitQ-m16-ex3 (self)                              3_652.50    16_550.35    20_202.85       0.9385          1.0011            1.0003        26.61
HnswRaBitQ-m16-ex5-ef15 (query)                        3_751.72       255.16     4_006.88       0.8643        161.1442            1.0012        35.76
HnswRaBitQ-m16-ex5-ef30 (query)                        3_751.72       313.99     4_065.71       0.9447         35.3877            1.0000        35.76
HnswRaBitQ-m16-ex5-ef60 (query)                        3_751.72       383.19     4_134.92       0.9775          1.0008            1.0000        35.76
HnswRaBitQ-m16-ex5-ef120 (query)                       3_751.72       460.51     4_212.23       0.9836          1.0004            1.0000        35.76
HnswRaBitQ-m16-ex5 (self)                              3_751.72    16_980.65    20_732.37       0.9757          1.0006            1.0000        35.76
HnswRaBitQ-m16-ex8-ef15 (query)                        5_935.25       231.62     6_166.88       0.9057          1.0059            1.0010        49.50
HnswRaBitQ-m16-ex8-ef30 (query)                        5_935.25       282.51     6_217.77       0.9601          1.0019            1.0000        49.50
HnswRaBitQ-m16-ex8-ef60 (query)                        5_935.25       340.53     6_275.79       0.9871          1.0007            1.0000        49.50
HnswRaBitQ-m16-ex8-ef120 (query)                       5_935.25       415.86     6_351.11       0.9945          1.0004            1.0000        49.50
HnswRaBitQ-m16-ex8 (self)                              5_935.25    16_335.60    22_270.85       0.9892          1.0007            1.0000        49.50
HnswRaBitQ-m32-ex1-ef15 (query)                        3_238.53       256.37     3_494.90       0.7807        362.2610            1.0032        23.56
HnswRaBitQ-m32-ex1-ef30 (query)                        3_238.53       322.03     3_560.56       0.8261        195.0033            1.0026        23.56
HnswRaBitQ-m32-ex1-ef60 (query)                        3_238.53       407.92     3_646.45       0.8653          1.0028            1.0024        23.56
HnswRaBitQ-m32-ex1-ef120 (query)                       3_238.53       484.55     3_723.08       0.8659          1.0027            1.0024        23.56
HnswRaBitQ-m32-ex1 (self)                              3_238.53    20_850.06    24_088.59       0.8200          1.0056            1.0050        23.56
HnswRaBitQ-m32-ex3-ef15 (query)                        3_504.28       283.20     3_787.48       0.8908        186.7745            1.0004        32.71
HnswRaBitQ-m32-ex3-ef30 (query)                        3_504.28       352.83     3_857.11       0.9348         76.7468            1.0002        32.71
HnswRaBitQ-m32-ex3-ef60 (query)                        3_504.28       430.02     3_934.30       0.9544          1.0004            1.0001        32.71
HnswRaBitQ-m32-ex3-ef120 (query)                       3_504.28       492.97     3_997.25       0.9556          1.0003            1.0001        32.71
HnswRaBitQ-m32-ex3 (self)                              3_504.28    19_094.27    22_598.55       0.9420          1.0006            1.0003        32.71
HnswRaBitQ-m32-ex5-ef15 (query)                        3_847.56       286.39     4_133.94       0.8816        353.2316            1.0001        41.87
HnswRaBitQ-m32-ex5-ef30 (query)                        3_847.56       371.67     4_219.22       0.9734          1.2254            1.0000        41.87
HnswRaBitQ-m32-ex5-ef60 (query)                        3_847.56       432.71     4_280.27       0.9832          1.0002            1.0000        41.87
HnswRaBitQ-m32-ex5-ef120 (query)                       3_847.56       504.27     4_351.82       0.9848          1.0001            1.0000        41.87
HnswRaBitQ-m32-ex5 (self)                              3_847.56    19_615.17    23_462.73       0.9805          1.0002            1.0000        41.87
HnswRaBitQ-m32-ex8-ef15 (query)                        5_966.53       286.90     6_253.43       0.8850        292.3750            1.0000        55.60
HnswRaBitQ-m32-ex8-ef30 (query)                        5_966.53       346.99     6_313.52       0.9460        172.2250            1.0000        55.60
HnswRaBitQ-m32-ex8-ef60 (query)                        5_966.53       425.97     6_392.50       0.9942          1.0002            1.0000        55.60
HnswRaBitQ-m32-ex8-ef120 (query)                       5_966.53       499.36     6_465.89       0.9962          1.0001            1.0000        55.60
HnswRaBitQ-m32-ex8 (self)                              5_966.53    19_043.17    25_009.70       0.9954          1.0018            1.0000        55.60
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Cell embeddings data

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 256D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.52       702.90       735.42       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.52     2_378.91     2_411.43       1.0000          1.0000            1.0000        48.83
QG-d32-l64-ef15 (query)                                2_736.73        58.95     2_795.68       0.9964          1.0001            1.0000       116.16
QG-d32-l64-ef30 (query)                                2_736.73        88.88     2_825.60       1.0000          1.0000            1.0000       116.16
QG-d32-l64-ef60 (query)                                2_736.73       157.90     2_894.63       1.0000          1.0000            1.0000       116.16
QG-d32-l64-ef120 (query)                               2_736.73       300.79     3_037.52       1.0000          1.0000            1.0000       116.16
QG-d32-l64 (self)                                      2_736.73       492.26     3_228.99       1.0000          1.0000            1.0000       116.16
QG-d32-l128-ef15 (query)                               3_924.64        56.80     3_981.43       0.9958          1.0001            1.0000       116.16
QG-d32-l128-ef30 (query)                               3_924.64        90.05     4_014.69       1.0000          1.0000            1.0000       116.16
QG-d32-l128-ef60 (query)                               3_924.64       157.36     4_082.00       1.0000          1.0000            1.0000       116.16
QG-d32-l128-ef120 (query)                              3_924.64       299.71     4_224.34       1.0000          1.0000            1.0000       116.16
QG-d32-l128 (self)                                     3_924.64       489.12     4_413.76       1.0000          1.0000            1.0000       116.16
QG-d64-l64-ef15 (query)                                3_487.89        64.69     3_552.58       0.9981          1.0000            1.0000       183.30
QG-d64-l64-ef30 (query)                                3_487.89       105.48     3_593.36       1.0000          1.0000            1.0000       183.30
QG-d64-l64-ef60 (query)                                3_487.89       190.07     3_677.96       1.0000          1.0000            1.0000       183.30
QG-d64-l64-ef120 (query)                               3_487.89       367.26     3_855.15       1.0000          1.0000            1.0000       183.30
QG-d64-l64 (self)                                      3_487.89       606.27     4_094.15       1.0000          1.0000            1.0000       183.30
QG-d64-l128-ef15 (query)                               5_141.88        66.15     5_208.03       0.9981          1.0000            1.0000       183.30
QG-d64-l128-ef30 (query)                               5_141.88       109.10     5_250.99       1.0000          1.0000            1.0000       183.30
QG-d64-l128-ef60 (query)                               5_141.88       195.70     5_337.59       1.0000          1.0000            1.0000       183.30
QG-d64-l128-ef120 (query)                              5_141.88       382.57     5_524.45       1.0000          1.0000            1.0000       183.30
QG-d64-l128 (self)                                     5_141.88       616.62     5_758.51       1.0000          1.0000            1.0000       183.30
HnswRaBitQ-m16-ex1-ef15 (query)                        1_221.91        69.47     1_291.39       0.9373          1.0208            1.0034        10.85
HnswRaBitQ-m16-ex1-ef30 (query)                        1_221.91        73.99     1_295.90       0.9385          1.0190            1.0032        10.85
HnswRaBitQ-m16-ex1-ef60 (query)                        1_221.91        92.54     1_314.45       0.9394          1.0134            1.0032        10.85
HnswRaBitQ-m16-ex1-ef120 (query)                       1_221.91       129.58     1_351.49       0.9402          1.0088            1.0032        10.85
HnswRaBitQ-m16-ex1 (self)                              1_221.91     3_200.97     4_422.88       0.9120          1.0191            1.0087        10.85
HnswRaBitQ-m16-ex3-ef15 (query)                        1_295.12        64.49     1_359.61       0.9746          1.0265            1.0000        13.90
HnswRaBitQ-m16-ex3-ef30 (query)                        1_295.12        75.74     1_370.86       0.9773          1.0192            1.0000        13.90
HnswRaBitQ-m16-ex3-ef60 (query)                        1_295.12        99.73     1_394.85       0.9788          1.0096            1.0000        13.90
HnswRaBitQ-m16-ex3-ef120 (query)                       1_295.12       128.02     1_423.15       0.9796          1.0051            1.0000        13.90
HnswRaBitQ-m16-ex3 (self)                              1_295.12     3_382.89     4_678.02       0.9726          1.0091            1.0000        13.90
HnswRaBitQ-m16-ex5-ef15 (query)                        1_394.43        69.82     1_464.25       0.9881          1.0277            1.0000        16.95
HnswRaBitQ-m16-ex5-ef30 (query)                        1_394.43        84.26     1_478.69       0.9912          1.0217            1.0000        16.95
HnswRaBitQ-m16-ex5-ef60 (query)                        1_394.43       100.83     1_495.26       0.9924          1.0120            1.0000        16.95
HnswRaBitQ-m16-ex5-ef120 (query)                       1_394.43       136.92     1_531.36       0.9935          1.0047            1.0000        16.95
HnswRaBitQ-m16-ex5 (self)                              1_394.43     3_311.75     4_706.18       0.9912          1.0082            1.0000        16.95
HnswRaBitQ-m16-ex8-ef15 (query)                        2_086.15        69.40     2_155.56       0.9943          1.0140            1.0000        21.53
HnswRaBitQ-m16-ex8-ef30 (query)                        2_086.15        81.47     2_167.62       0.9972          1.0109            1.0000        21.53
HnswRaBitQ-m16-ex8-ef60 (query)                        2_086.15       101.16     2_187.32       0.9981          1.0070            1.0000        21.53
HnswRaBitQ-m16-ex8-ef120 (query)                       2_086.15       141.70     2_227.85       0.9984          1.0048            1.0000        21.53
HnswRaBitQ-m16-ex8 (self)                              2_086.15     3_498.00     5_584.16       0.9982          1.0038            1.0000        21.53
HnswRaBitQ-m32-ex1-ef15 (query)                        1_278.47        69.64     1_348.10       0.9393          1.0105            1.0033        16.95
HnswRaBitQ-m32-ex1-ef30 (query)                        1_278.47        81.90     1_360.37       0.9400          1.0093            1.0032        16.95
HnswRaBitQ-m32-ex1-ef60 (query)                        1_278.47       102.14     1_380.60       0.9403          1.0080            1.0032        16.95
HnswRaBitQ-m32-ex1-ef120 (query)                       1_278.47       139.48     1_417.95       0.9406          1.0068            1.0032        16.95
HnswRaBitQ-m32-ex1 (self)                              1_278.47     3_815.59     5_094.06       0.9130          1.0144            1.0086        16.95
HnswRaBitQ-m32-ex3-ef15 (query)                        1_355.29        75.01     1_430.30       0.9774          1.0111            1.0000        20.00
HnswRaBitQ-m32-ex3-ef30 (query)                        1_355.29        87.78     1_443.07       0.9789          1.0084            1.0000        20.00
HnswRaBitQ-m32-ex3-ef60 (query)                        1_355.29       109.38     1_464.67       0.9797          1.0043            1.0000        20.00
HnswRaBitQ-m32-ex3-ef120 (query)                       1_355.29       147.13     1_502.42       0.9803          1.0021            1.0000        20.00
HnswRaBitQ-m32-ex3 (self)                              1_355.29     4_029.26     5_384.55       0.9734          1.0042            1.0000        20.00
HnswRaBitQ-m32-ex5-ef15 (query)                        1_465.03        79.56     1_544.59       0.9916          1.0091            1.0000        23.06
HnswRaBitQ-m32-ex5-ef30 (query)                        1_465.03        93.54     1_558.57       0.9932          1.0079            1.0000        23.06
HnswRaBitQ-m32-ex5-ef60 (query)                        1_465.03       115.90     1_580.93       0.9940          1.0038            1.0000        23.06
HnswRaBitQ-m32-ex5-ef120 (query)                       1_465.03       156.09     1_621.12       0.9944          1.0008            1.0000        23.06
HnswRaBitQ-m32-ex5 (self)                              1_465.03     3_713.79     5_178.82       0.9921          1.0034            1.0000        23.06
HnswRaBitQ-m32-ex8-ef15 (query)                        2_151.13        78.99     2_230.12       0.9963          1.0068            1.0000        27.63
HnswRaBitQ-m32-ex8-ef30 (query)                        2_151.13        93.42     2_244.55       0.9985          1.0040            1.0000        27.63
HnswRaBitQ-m32-ex8-ef60 (query)                        2_151.13       118.28     2_269.41       0.9988          1.0029            1.0000        27.63
HnswRaBitQ-m32-ex8-ef120 (query)                       2_151.13       162.83     2_313.96       0.9990          1.0013            1.0000        27.63
HnswRaBitQ-m32-ex8 (self)                              2_151.13     3_860.66     6_011.79       0.9986          1.0017            1.0000        27.63
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 512D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.93     1_313.22     1_382.15       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.93     4_414.87     4_483.81       1.0000          1.0000            1.0000        97.66
QG-d32-l64-ef15 (query)                                5_223.44        89.79     5_313.23       0.9976          1.0001            1.0000       213.81
QG-d32-l64-ef30 (query)                                5_223.44       129.66     5_353.10       1.0000          1.0000            1.0000       213.81
QG-d32-l64-ef60 (query)                                5_223.44       207.96     5_431.40       1.0000          1.0000            1.0000       213.81
QG-d32-l64-ef120 (query)                               5_223.44       375.98     5_599.42       1.0000          1.0000            1.0000       213.81
QG-d32-l64 (self)                                      5_223.44       626.81     5_850.25       1.0000          1.0000            1.0000       213.81
QG-d32-l128-ef15 (query)                               7_640.81        88.30     7_729.11       0.9970          1.0001            1.0000       213.81
QG-d32-l128-ef30 (query)                               7_640.81       126.49     7_767.30       1.0000          1.0000            1.0000       213.81
QG-d32-l128-ef60 (query)                               7_640.81       203.49     7_844.30       1.0000          1.0000            1.0000       213.81
QG-d32-l128-ef120 (query)                              7_640.81       369.04     8_009.85       1.0000          1.0000            1.0000       213.81
QG-d32-l128 (self)                                     7_640.81       616.89     8_257.70       1.0000          1.0000            1.0000       213.81
QG-d64-l64-ef15 (query)                                6_613.64        97.01     6_710.65       0.9984          1.0000            1.0000       329.78
QG-d64-l64-ef30 (query)                                6_613.64       138.19     6_751.83       1.0000          1.0000            1.0000       329.78
QG-d64-l64-ef60 (query)                                6_613.64       228.39     6_842.03       1.0000          1.0000            1.0000       329.78
QG-d64-l64-ef120 (query)                               6_613.64       422.91     7_036.55       1.0000          1.0000            1.0000       329.78
QG-d64-l64 (self)                                      6_613.64       695.36     7_309.00       1.0000          1.0000            1.0000       329.78
QG-d64-l128-ef15 (query)                              10_055.99        95.93    10_151.92       0.9983          1.0000            1.0000       329.78
QG-d64-l128-ef30 (query)                              10_055.99       139.97    10_195.96       1.0000          1.0000            1.0000       329.78
QG-d64-l128-ef60 (query)                              10_055.99       231.54    10_287.53       1.0000          1.0000            1.0000       329.78
QG-d64-l128-ef120 (query)                             10_055.99       430.12    10_486.11       1.0000          1.0000            1.0000       329.78
QG-d64-l128 (self)                                    10_055.99       702.26    10_758.25       1.0000          1.0000            1.0000       329.78
HnswRaBitQ-m16-ex1-ef15 (query)                        2_231.55       112.42     2_343.97       0.9419          1.2536            1.0007        14.15
HnswRaBitQ-m16-ex1-ef30 (query)                        2_231.55       126.63     2_358.18       0.9462          1.1795            1.0006        14.15
HnswRaBitQ-m16-ex1-ef60 (query)                        2_231.55       151.08     2_382.64       0.9514          1.0888            1.0006        14.15
HnswRaBitQ-m16-ex1-ef120 (query)                       2_231.55       192.74     2_424.29       0.9552          1.0310            1.0005        14.15
HnswRaBitQ-m16-ex1 (self)                              2_231.55     5_728.58     7_960.13       0.9294          1.0746            1.0040        14.15
HnswRaBitQ-m16-ex3-ef15 (query)                        2_410.47       120.49     2_530.95       0.9730          1.1781            1.0000        20.25
HnswRaBitQ-m16-ex3-ef30 (query)                        2_410.47       133.91     2_544.38       0.9775          1.1320            1.0000        20.25
HnswRaBitQ-m16-ex3-ef60 (query)                        2_410.47       158.96     2_569.42       0.9806          1.0805            1.0000        20.25
HnswRaBitQ-m16-ex3-ef120 (query)                       2_410.47       202.01     2_612.48       0.9831          1.0362            1.0000        20.25
HnswRaBitQ-m16-ex3 (self)                              2_410.47     5_760.59     8_171.06       0.9764          1.0678            1.0000        20.25
HnswRaBitQ-m16-ex5-ef15 (query)                        2_690.80       125.55     2_816.35       0.9885          1.0534            1.0000        26.36
HnswRaBitQ-m16-ex5-ef30 (query)                        2_690.80       145.35     2_836.15       0.9916          1.0418            1.0000        26.36
HnswRaBitQ-m16-ex5-ef60 (query)                        2_690.80       169.17     2_859.97       0.9929          1.0272            1.0000        26.36
HnswRaBitQ-m16-ex5-ef120 (query)                       2_690.80       212.16     2_902.96       0.9945          1.0107            1.0000        26.36
HnswRaBitQ-m16-ex5 (self)                              2_690.80     5_956.98     8_647.78       0.9920          1.0225            1.0000        26.36
HnswRaBitQ-m16-ex8-ef15 (query)                        4_054.30       123.52     4_177.81       0.9814          1.2256            1.0000        35.51
HnswRaBitQ-m16-ex8-ef30 (query)                        4_054.30       139.02     4_193.31       0.9874          1.1711            1.0000        35.51
HnswRaBitQ-m16-ex8-ef60 (query)                        4_054.30       167.82     4_222.12       0.9910          1.1149            1.0000        35.51
HnswRaBitQ-m16-ex8-ef120 (query)                       4_054.30       211.51     4_265.80       0.9948          1.0569            1.0000        35.51
HnswRaBitQ-m16-ex8 (self)                              4_054.30     5_836.71     9_891.01       0.9935          1.0690            1.0000        35.51
HnswRaBitQ-m32-ex1-ef15 (query)                        2_319.73       123.58     2_443.31       0.9572          1.0033            1.0006        20.25
HnswRaBitQ-m32-ex1-ef30 (query)                        2_319.73       139.34     2_459.08       0.9580          1.0031            1.0005        20.25
HnswRaBitQ-m32-ex1-ef60 (query)                        2_319.73       168.08     2_487.82       0.9582          1.0024            1.0005        20.25
HnswRaBitQ-m32-ex1-ef120 (query)                       2_319.73       214.85     2_534.58       0.9583          1.0022            1.0005        20.25
HnswRaBitQ-m32-ex1 (self)                              2_319.73     6_399.11     8_718.84       0.9351          1.0068            1.0039        20.25
HnswRaBitQ-m32-ex3-ef15 (query)                        2_505.43       132.72     2_638.15       0.9850          1.0015            1.0000        26.36
HnswRaBitQ-m32-ex3-ef30 (query)                        2_505.43       149.79     2_655.22       0.9862          1.0013            1.0000        26.36
HnswRaBitQ-m32-ex3-ef60 (query)                        2_505.43       177.52     2_682.95       0.9865          1.0002            1.0000        26.36
HnswRaBitQ-m32-ex3-ef120 (query)                       2_505.43       227.34     2_732.77       0.9865          1.0002            1.0000        26.36
HnswRaBitQ-m32-ex3 (self)                              2_505.43     6_670.53     9_175.96       0.9817          1.0007            1.0000        26.36
HnswRaBitQ-m32-ex5-ef15 (query)                        2_693.04       142.25     2_835.29       0.9944          1.0002            1.0000        32.46
HnswRaBitQ-m32-ex5-ef30 (query)                        2_693.04       156.81     2_849.86       0.9959          1.0000            1.0000        32.46
HnswRaBitQ-m32-ex5-ef60 (query)                        2_693.04       186.14     2_879.18       0.9960          1.0000            1.0000        32.46
HnswRaBitQ-m32-ex5-ef120 (query)                       2_693.04       233.73     2_926.77       0.9960          1.0000            1.0000        32.46
HnswRaBitQ-m32-ex5 (self)                              2_693.04     6_863.19     9_556.23       0.9947          1.0002            1.0000        32.46
HnswRaBitQ-m32-ex8-ef15 (query)                        4_134.59       136.13     4_270.72       0.9976          1.0004            1.0000        41.62
HnswRaBitQ-m32-ex8-ef30 (query)                        4_134.59       154.78     4_289.37       0.9994          1.0002            1.0000        41.62
HnswRaBitQ-m32-ex8-ef60 (query)                        4_134.59       181.37     4_315.96       0.9995          1.0002            1.0000        41.62
HnswRaBitQ-m32-ex8-ef120 (query)                       4_134.59       231.71     4_366.30       0.9995          1.0000            1.0000        41.62
HnswRaBitQ-m32-ex8 (self)                              4_134.59     6_608.47    10_743.06       0.9993          1.0000            1.0000        41.62
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 768D - RaBitQ graph indices
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       100.18     1_870.78     1_970.96       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.18     6_524.05     6_624.23       1.0000          1.0000            1.0000       146.48
QG-d32-l64-ef15 (query)                                7_281.54       108.96     7_390.50       0.9974          1.0001            1.0000       311.47
QG-d32-l64-ef30 (query)                                7_281.54       152.79     7_434.33       1.0000          1.0000            1.0000       311.47
QG-d32-l64-ef60 (query)                                7_281.54       242.59     7_524.14       1.0000          1.0000            1.0000       311.47
QG-d32-l64-ef120 (query)                               7_281.54       432.16     7_713.70       1.0000          1.0000            1.0000       311.47
QG-d32-l64 (self)                                      7_281.54       723.97     8_005.51       1.0000          1.0000            1.0000       311.47
QG-d32-l128-ef15 (query)                              10_672.39       109.80    10_782.19       0.9968          1.0001            1.0000       311.47
QG-d32-l128-ef30 (query)                              10_672.39       150.30    10_822.69       1.0000          1.0000            1.0000       311.47
QG-d32-l128-ef60 (query)                              10_672.39       239.74    10_912.13       1.0000          1.0000            1.0000       311.47
QG-d32-l128-ef120 (query)                             10_672.39       429.13    11_101.52       1.0000          1.0000            1.0000       311.47
QG-d32-l128 (self)                                    10_672.39       720.21    11_392.60       1.0000          1.0000            1.0000       311.47
QG-d64-l64-ef15 (query)                                9_265.39       116.17     9_381.56       0.9981          1.0000            1.0000       476.27
QG-d64-l64-ef30 (query)                                9_265.39       169.16     9_434.55       1.0000          1.0000            1.0000       476.27
QG-d64-l64-ef60 (query)                                9_265.39       265.57     9_530.96       1.0000          1.0000            1.0000       476.27
QG-d64-l64-ef120 (query)                               9_265.39       476.52     9_741.91       1.0000          1.0000            1.0000       476.27
QG-d64-l64 (self)                                      9_265.39       807.42    10_072.80       1.0000          1.0000            1.0000       476.27
QG-d64-l128-ef15 (query)                              13_568.66       115.07    13_683.74       0.9979          1.0000            1.0000       476.27
QG-d64-l128-ef30 (query)                              13_568.66       163.66    13_732.32       1.0000          1.0000            1.0000       476.27
QG-d64-l128-ef60 (query)                              13_568.66       265.73    13_834.39       1.0000          1.0000            1.0000       476.27
QG-d64-l128-ef120 (query)                             13_568.66       484.22    14_052.88       1.0000          1.0000            1.0000       476.27
QG-d64-l128 (self)                                    13_568.66       798.00    14_366.66       1.0000          1.0000            1.0000       476.27
HnswRaBitQ-m16-ex1-ef15 (query)                        2_918.29       152.60     3_070.88       0.9497          1.0102            1.0015        17.45
HnswRaBitQ-m16-ex1-ef30 (query)                        2_918.29       168.28     3_086.57       0.9509          1.0093            1.0014        17.45
HnswRaBitQ-m16-ex1-ef60 (query)                        2_918.29       198.28     3_116.57       0.9512          1.0065            1.0014        17.45
HnswRaBitQ-m16-ex1-ef120 (query)                       2_918.29       247.86     3_166.15       0.9514          1.0037            1.0014        17.45
HnswRaBitQ-m16-ex1 (self)                              2_918.29     7_644.07    10_562.35       0.9212          1.0132            1.0063        17.45
HnswRaBitQ-m16-ex3-ef15 (query)                        3_171.44       169.10     3_340.54       0.9754          1.0093            1.0000        26.61
HnswRaBitQ-m16-ex3-ef30 (query)                        3_171.44       182.72     3_354.16       0.9772          1.0058            1.0000        26.61
HnswRaBitQ-m16-ex3-ef60 (query)                        3_171.44       210.43     3_381.86       0.9774          1.0046            1.0000        26.61
HnswRaBitQ-m16-ex3-ef120 (query)                       3_171.44       259.20     3_430.63       0.9775          1.0041            1.0000        26.61
HnswRaBitQ-m16-ex3 (self)                              3_171.44     7_783.65    10_955.08       0.9699          1.0099            1.0000        26.61
HnswRaBitQ-m16-ex5-ef15 (query)                        3_366.04       179.20     3_545.24       0.9924          1.0018            1.0000        35.76
HnswRaBitQ-m16-ex5-ef30 (query)                        3_366.04       192.03     3_558.07       0.9946          1.0015            1.0000        35.76
HnswRaBitQ-m16-ex5-ef60 (query)                        3_366.04       222.32     3_588.36       0.9948          1.0011            1.0000        35.76
HnswRaBitQ-m16-ex5-ef120 (query)                       3_366.04       279.61     3_645.65       0.9949          1.0011            1.0000        35.76
HnswRaBitQ-m16-ex5 (self)                              3_366.04     8_049.71    11_415.75       0.9931          1.0012            1.0000        35.76
HnswRaBitQ-m16-ex8-ef15 (query)                        5_103.72       161.21     5_264.94       0.9962          1.0063            1.0000        49.50
HnswRaBitQ-m16-ex8-ef30 (query)                        5_103.72       179.18     5_282.90       0.9986          1.0061            1.0000        49.50
HnswRaBitQ-m16-ex8-ef60 (query)                        5_103.72       211.25     5_314.98       0.9988          1.0054            1.0000        49.50
HnswRaBitQ-m16-ex8-ef120 (query)                       5_103.72       262.96     5_366.68       0.9988          1.0054            1.0000        49.50
HnswRaBitQ-m16-ex8 (self)                              5_103.72     7_962.51    13_066.24       0.9988          1.0055            1.0000        49.50
HnswRaBitQ-m32-ex1-ef15 (query)                        3_042.12       167.79     3_209.91       0.9507          1.0037            1.0015        23.56
HnswRaBitQ-m32-ex1-ef30 (query)                        3_042.12       186.86     3_228.99       0.9514          1.0036            1.0014        23.56
HnswRaBitQ-m32-ex1-ef60 (query)                        3_042.12       222.21     3_264.34       0.9514          1.0036            1.0014        23.56
HnswRaBitQ-m32-ex1-ef120 (query)                       3_042.12       277.32     3_319.44       0.9514          1.0036            1.0014        23.56
HnswRaBitQ-m32-ex1 (self)                              3_042.12     8_711.39    11_753.51       0.9215          1.0107            1.0063        23.56
HnswRaBitQ-m32-ex3-ef15 (query)                        3_255.73       180.18     3_435.91       0.9767          1.0010            1.0000        32.71
HnswRaBitQ-m32-ex3-ef30 (query)                        3_255.73       200.02     3_455.75       0.9779          1.0008            1.0000        32.71
HnswRaBitQ-m32-ex3-ef60 (query)                        3_255.73       232.54     3_488.27       0.9779          1.0008            1.0000        32.71
HnswRaBitQ-m32-ex3-ef120 (query)                       3_255.73       290.31     3_546.04       0.9779          1.0008            1.0000        32.71
HnswRaBitQ-m32-ex3 (self)                              3_255.73     9_020.93    12_276.66       0.9706          1.0015            1.0000        32.71
HnswRaBitQ-m32-ex5-ef15 (query)                        3_470.91       193.44     3_664.35       0.9934          1.0002            1.0000        41.87
HnswRaBitQ-m32-ex5-ef30 (query)                        3_470.91       210.88     3_681.79       0.9950          1.0000            1.0000        41.87
HnswRaBitQ-m32-ex5-ef60 (query)                        3_470.91       244.43     3_715.33       0.9951          1.0000            1.0000        41.87
HnswRaBitQ-m32-ex5-ef120 (query)                       3_470.91       301.91     3_772.81       0.9951          1.0000            1.0000        41.87
HnswRaBitQ-m32-ex5 (self)                              3_470.91     9_162.81    12_633.71       0.9932          1.0001            1.0000        41.87
HnswRaBitQ-m32-ex8-ef15 (query)                        5_210.58       179.79     5_390.37       0.9975          1.0003            1.0000        55.60
HnswRaBitQ-m32-ex8-ef30 (query)                        5_210.58       200.12     5_410.70       0.9992          1.0002            1.0000        55.60
HnswRaBitQ-m32-ex8-ef60 (query)                        5_210.58       233.83     5_444.41       0.9993          1.0002            1.0000        55.60
HnswRaBitQ-m32-ex8-ef120 (query)                       5_210.58       295.70     5_506.28       0.9993          1.0002            1.0000        55.60
HnswRaBitQ-m32-ex8 (self)                              5_210.58     8_922.55    14_133.13       0.9992          1.0004            1.0000        55.60
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
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 256D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.41       678.93       711.34       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.41     2_244.15     2_276.56       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              147.10       360.01       507.11       0.0971          1.7176            1.5960         7.12
ExhaustiveTQ-b2-rf5 (query)                              147.10       436.48       583.58       0.2336          1.2025            1.2205         7.12
ExhaustiveTQ-b2-rf10 (query)                             147.10       568.15       715.25       0.2853          1.1453            1.1620         7.12
ExhaustiveTQ-b2-rf20 (query)                             147.10       937.54     1_084.64       0.3809          1.0970            1.0941         7.12
ExhaustiveTQ-b2 (self)                                   147.10     3_066.51     3_213.61       0.3815          1.0980            1.0956         7.12
ExhaustiveTQ-b4-rf0 (query)                              225.08       563.90       788.98       0.1094          1.5329            1.4998        13.22
ExhaustiveTQ-b4-rf5 (query)                              225.08       645.44       870.52       0.2368          1.1884            1.2090        13.22
ExhaustiveTQ-b4-rf10 (query)                             225.08       802.74     1_027.82       0.2884          1.1372            1.1544        13.22
ExhaustiveTQ-b4-rf20 (query)                             225.08     1_149.36     1_374.43       0.3823          1.0940            1.0971        13.22
ExhaustiveTQ-b4 (self)                                   225.08     3_799.36     4_024.44       0.3841          1.0937            1.0948        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                          264.49       104.55       369.03       0.0971          1.7176            1.5960         7.62
IVF-TQ-b2-nl158-np12-rf0 (query)                         264.49       117.15       381.63       0.0971          1.7176            1.5960         7.62
IVF-TQ-b2-nl158-np17-rf0 (query)                         264.49       128.08       392.56       0.0971          1.7176            1.5960         7.62
IVF-TQ-b2-nl158-np7-rf10 (query)                         264.49       282.27       546.75       0.2853          1.1453            1.1620         7.62
IVF-TQ-b2-nl158-np7-rf20 (query)                         264.49       547.69       812.18       0.3809          1.0970            1.0941         7.62
IVF-TQ-b2-nl158-np12-rf10 (query)                        264.49       305.89       570.38       0.2853          1.1453            1.1620         7.62
IVF-TQ-b2-nl158-np12-rf20 (query)                        264.49       584.69       849.18       0.3809          1.0970            1.0941         7.62
IVF-TQ-b2-nl158-np17-rf10 (query)                        264.49       321.88       586.37       0.2853          1.1453            1.1620         7.62
IVF-TQ-b2-nl158-np17-rf20 (query)                        264.49       618.92       883.41       0.3809          1.0970            1.0941         7.62
IVF-TQ-b2-nl158 (self)                                   264.49     1_020.26     1_284.75       0.3815          1.0980            1.0956         7.62
IVF-TQ-b2-nl223-np11-rf0 (query)                         261.48       111.05       372.53       0.0971          1.7176            1.5959         7.74
IVF-TQ-b2-nl223-np14-rf0 (query)                         261.48       117.40       378.88       0.0971          1.7176            1.5959         7.74
IVF-TQ-b2-nl223-np21-rf0 (query)                         261.48       129.38       390.86       0.0971          1.7176            1.5959         7.74
IVF-TQ-b2-nl223-np11-rf10 (query)                        261.48       307.90       569.38       0.2853          1.1453            1.1620         7.74
IVF-TQ-b2-nl223-np11-rf20 (query)                        261.48       546.06       807.55       0.3809          1.0970            1.0941         7.74
IVF-TQ-b2-nl223-np14-rf10 (query)                        261.48       294.55       556.03       0.2853          1.1453            1.1620         7.74
IVF-TQ-b2-nl223-np14-rf20 (query)                        261.48       562.96       824.45       0.3809          1.0970            1.0941         7.74
IVF-TQ-b2-nl223-np21-rf10 (query)                        261.48       307.23       568.71       0.2853          1.1453            1.1620         7.74
IVF-TQ-b2-nl223-np21-rf20 (query)                        261.48       594.76       856.24       0.3809          1.0970            1.0940         7.74
IVF-TQ-b2-nl223 (self)                                   261.48     1_018.71     1_280.19       0.3815          1.0980            1.0956         7.74
IVF-TQ-b2-nl316-np15-rf0 (query)                         269.89       114.52       384.41       0.0971          1.7176            1.5960         7.93
IVF-TQ-b2-nl316-np17-rf0 (query)                         269.89       119.00       388.89       0.0971          1.7176            1.5960         7.93
IVF-TQ-b2-nl316-np25-rf0 (query)                         269.89       132.15       402.04       0.0971          1.7176            1.5960         7.93
IVF-TQ-b2-nl316-np15-rf10 (query)                        269.89       277.92       547.81       0.2853          1.1453            1.1620         7.93
IVF-TQ-b2-nl316-np15-rf20 (query)                        269.89       505.78       775.67       0.3809          1.0970            1.0941         7.93
IVF-TQ-b2-nl316-np17-rf10 (query)                        269.89       281.51       551.41       0.2853          1.1453            1.1620         7.93
IVF-TQ-b2-nl316-np17-rf20 (query)                        269.89       524.42       794.31       0.3809          1.0970            1.0941         7.93
IVF-TQ-b2-nl316-np25-rf10 (query)                        269.89       298.56       568.45       0.2853          1.1453            1.1620         7.93
IVF-TQ-b2-nl316-np25-rf20 (query)                        269.89       548.60       818.49       0.3809          1.0970            1.0941         7.93
IVF-TQ-b2-nl316 (self)                                   269.89     1_027.58     1_297.47       0.3815          1.0980            1.0957         7.93
IVF-TQ-b4-nl158-np7-rf0 (query)                          340.82       142.41       483.22       0.1094          1.5329            1.4998        13.87
IVF-TQ-b4-nl158-np12-rf0 (query)                         340.82       163.33       504.15       0.1094          1.5329            1.4998        13.87
IVF-TQ-b4-nl158-np17-rf0 (query)                         340.82       180.96       521.78       0.1094          1.5329            1.4998        13.87
IVF-TQ-b4-nl158-np7-rf10 (query)                         340.82       327.76       668.57       0.2884          1.1372            1.1543        13.87
IVF-TQ-b4-nl158-np7-rf20 (query)                         340.82       592.46       933.27       0.3823          1.0940            1.0971        13.87
IVF-TQ-b4-nl158-np12-rf10 (query)                        340.82       354.46       695.27       0.2884          1.1372            1.1543        13.87
IVF-TQ-b4-nl158-np12-rf20 (query)                        340.82       659.14       999.95       0.3823          1.0940            1.0971        13.87
IVF-TQ-b4-nl158-np17-rf10 (query)                        340.82       385.78       726.59       0.2884          1.1372            1.1543        13.87
IVF-TQ-b4-nl158-np17-rf20 (query)                        340.82       674.87     1_015.68       0.3823          1.0940            1.0971        13.87
IVF-TQ-b4-nl158 (self)                                   340.82     1_118.80     1_459.61       0.3841          1.0937            1.0948        13.87
IVF-TQ-b4-nl223-np11-rf0 (query)                         332.99       151.76       484.76       0.1094          1.5329            1.4998        14.05
IVF-TQ-b4-nl223-np14-rf0 (query)                         332.99       162.61       495.60       0.1094          1.5329            1.4998        14.05
IVF-TQ-b4-nl223-np21-rf0 (query)                         332.99       182.81       515.81       0.1094          1.5329            1.4998        14.05
IVF-TQ-b4-nl223-np11-rf10 (query)                        332.99       336.50       669.50       0.2884          1.1372            1.1544        14.05
IVF-TQ-b4-nl223-np11-rf20 (query)                        332.99       600.31       933.30       0.3823          1.0940            1.0971        14.05
IVF-TQ-b4-nl223-np14-rf10 (query)                        332.99       349.86       682.85       0.2884          1.1372            1.1544        14.05
IVF-TQ-b4-nl223-np14-rf20 (query)                        332.99       625.79       958.78       0.3823          1.0940            1.0971        14.05
IVF-TQ-b4-nl223-np21-rf10 (query)                        332.99       369.16       702.16       0.2884          1.1372            1.1544        14.05
IVF-TQ-b4-nl223-np21-rf20 (query)                        332.99       661.51       994.50       0.3823          1.0940            1.0971        14.05
IVF-TQ-b4-nl223 (self)                                   332.99     1_118.90     1_451.90       0.3841          1.0937            1.0948        14.05
IVF-TQ-b4-nl316-np15-rf0 (query)                         358.32       163.57       521.90       0.1094          1.5329            1.4998        14.34
IVF-TQ-b4-nl316-np17-rf0 (query)                         358.32       165.33       523.66       0.1094          1.5329            1.4998        14.34
IVF-TQ-b4-nl316-np25-rf0 (query)                         358.32       186.48       544.80       0.1094          1.5329            1.4998        14.34
IVF-TQ-b4-nl316-np15-rf10 (query)                        358.32       330.89       689.21       0.2884          1.1372            1.1544        14.34
IVF-TQ-b4-nl316-np15-rf20 (query)                        358.32       572.56       930.88       0.3823          1.0940            1.0971        14.34
IVF-TQ-b4-nl316-np17-rf10 (query)                        358.32       336.62       694.94       0.2884          1.1372            1.1544        14.34
IVF-TQ-b4-nl316-np17-rf20 (query)                        358.32       581.66       939.98       0.3823          1.0940            1.0971        14.34
IVF-TQ-b4-nl316-np25-rf10 (query)                        358.32       363.20       721.52       0.2884          1.1372            1.1544        14.34
IVF-TQ-b4-nl316-np25-rf20 (query)                        358.32       618.36       976.69       0.3823          1.0940            1.0971        14.34
IVF-TQ-b4-nl316 (self)                                   358.32     1_120.60     1_478.93       0.3841          1.0937            1.0948        14.34
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 512D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.68     1_302.57     1_370.26       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         67.68     4_416.91     4_484.60       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              335.61       631.85       967.46       0.1208          1.3711            1.3321        13.97
ExhaustiveTQ-b2-rf5 (query)                              335.61       716.02     1_051.63       0.2421          1.1334            1.1574        13.97
ExhaustiveTQ-b2-rf10 (query)                             335.61       851.16     1_186.77       0.2935          1.0981            1.1178        13.97
ExhaustiveTQ-b2-rf20 (query)                             335.61     1_238.61     1_574.23       0.3880          1.0664            1.0469        13.97
ExhaustiveTQ-b2 (self)                                   335.61     4_033.41     4_369.02       0.3879          1.0667            1.0471        13.97
ExhaustiveTQ-b4-rf0 (query)                              458.80     1_103.94     1_562.74       0.1315          1.3171            1.3127        26.18
ExhaustiveTQ-b4-rf5 (query)                              458.80     1_204.85     1_663.65       0.2471          1.1254            1.1483        26.18
ExhaustiveTQ-b4-rf10 (query)                             458.80     1_339.46     1_798.26       0.2970          1.0929            1.0979        26.18
ExhaustiveTQ-b4-rf20 (query)                             458.80     1_717.05     2_175.84       0.3883          1.0643            1.0493        26.18
ExhaustiveTQ-b4 (self)                                   458.80     5_644.49     6_103.29       0.3882          1.0646            1.0495        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                          534.58       193.37       727.95       0.1208          1.3711            1.3321        14.77
IVF-TQ-b2-nl158-np12-rf0 (query)                         534.58       212.04       746.62       0.1208          1.3711            1.3321        14.77
IVF-TQ-b2-nl158-np17-rf0 (query)                         534.58       227.90       762.48       0.1208          1.3711            1.3321        14.77
IVF-TQ-b2-nl158-np7-rf10 (query)                         534.58       411.38       945.96       0.2934          1.0981            1.1178        14.77
IVF-TQ-b2-nl158-np7-rf20 (query)                         534.58       723.03     1_257.61       0.3880          1.0664            1.0469        14.77
IVF-TQ-b2-nl158-np12-rf10 (query)                        534.58       432.82       967.40       0.2934          1.0981            1.1178        14.77
IVF-TQ-b2-nl158-np12-rf20 (query)                        534.58       770.92     1_305.50       0.3880          1.0664            1.0469        14.77
IVF-TQ-b2-nl158-np17-rf10 (query)                        534.58       450.61       985.19       0.2934          1.0981            1.1178        14.77
IVF-TQ-b2-nl158-np17-rf20 (query)                        534.58       804.95     1_339.53       0.3880          1.0664            1.0469        14.77
IVF-TQ-b2-nl158 (self)                                   534.58     1_378.27     1_912.85       0.3879          1.0667            1.0471        14.77
IVF-TQ-b2-nl223-np11-rf0 (query)                         518.57       200.34       718.91       0.1208          1.3711            1.3321        15.00
IVF-TQ-b2-nl223-np14-rf0 (query)                         518.57       216.01       734.59       0.1208          1.3711            1.3321        15.00
IVF-TQ-b2-nl223-np21-rf0 (query)                         518.57       229.73       748.30       0.1208          1.3711            1.3321        15.00
IVF-TQ-b2-nl223-np11-rf10 (query)                        518.57       396.46       915.03       0.2935          1.0981            1.1178        15.00
IVF-TQ-b2-nl223-np11-rf20 (query)                        518.57       677.04     1_195.62       0.3880          1.0664            1.0469        15.00
IVF-TQ-b2-nl223-np14-rf10 (query)                        518.57       406.22       924.79       0.2935          1.0981            1.1178        15.00
IVF-TQ-b2-nl223-np14-rf20 (query)                        518.57       711.64     1_230.22       0.3880          1.0664            1.0469        15.00
IVF-TQ-b2-nl223-np21-rf10 (query)                        518.57       431.17       949.74       0.2934          1.0981            1.1178        15.00
IVF-TQ-b2-nl223-np21-rf20 (query)                        518.57       742.12     1_260.69       0.3880          1.0664            1.0469        15.00
IVF-TQ-b2-nl223 (self)                                   518.57     1_388.65     1_907.22       0.3879          1.0667            1.0471        15.00
IVF-TQ-b2-nl316-np15-rf0 (query)                         565.95       209.36       775.31       0.1208          1.3711            1.3321        15.38
IVF-TQ-b2-nl316-np17-rf0 (query)                         565.95       216.12       782.08       0.1208          1.3711            1.3321        15.38
IVF-TQ-b2-nl316-np25-rf0 (query)                         565.95       234.75       800.70       0.1208          1.3711            1.3321        15.38
IVF-TQ-b2-nl316-np15-rf10 (query)                        565.95       400.36       966.32       0.2934          1.0981            1.1178        15.38
IVF-TQ-b2-nl316-np15-rf20 (query)                        565.95       719.89     1_285.84       0.3880          1.0664            1.0469        15.38
IVF-TQ-b2-nl316-np17-rf10 (query)                        565.95       406.22       972.17       0.2934          1.0981            1.1178        15.38
IVF-TQ-b2-nl316-np17-rf20 (query)                        565.95       684.00     1_249.96       0.3880          1.0664            1.0469        15.38
IVF-TQ-b2-nl316-np25-rf10 (query)                        565.95       432.15       998.10       0.2934          1.0981            1.1178        15.38
IVF-TQ-b2-nl316-np25-rf20 (query)                        565.95       722.06     1_288.01       0.3880          1.0664            1.0469        15.38
IVF-TQ-b2-nl316 (self)                                   565.95     1_412.93     1_978.89       0.3879          1.0667            1.0471        15.38
IVF-TQ-b4-nl158-np7-rf0 (query)                          634.57       266.96       901.53       0.1315          1.3171            1.3127        27.27
IVF-TQ-b4-nl158-np12-rf0 (query)                         634.57       302.16       936.73       0.1315          1.3171            1.3127        27.27
IVF-TQ-b4-nl158-np17-rf0 (query)                         634.57       328.26       962.83       0.1315          1.3171            1.3127        27.27
IVF-TQ-b4-nl158-np7-rf10 (query)                         634.57       497.41     1_131.98       0.2970          1.0929            1.0979        27.27
IVF-TQ-b4-nl158-np7-rf20 (query)                         634.57       816.01     1_450.57       0.3883          1.0643            1.0493        27.27
IVF-TQ-b4-nl158-np12-rf10 (query)                        634.57       537.60     1_172.17       0.2970          1.0929            1.0979        27.27
IVF-TQ-b4-nl158-np12-rf20 (query)                        634.57       882.29     1_516.86       0.3883          1.0643            1.0493        27.27
IVF-TQ-b4-nl158-np17-rf10 (query)                        634.57       568.32     1_202.89       0.2970          1.0929            1.0979        27.27
IVF-TQ-b4-nl158-np17-rf20 (query)                        634.57       927.86     1_562.43       0.3883          1.0643            1.0493        27.27
IVF-TQ-b4-nl158 (self)                                   634.57     1_595.30     2_229.87       0.3882          1.0646            1.0495        27.27
IVF-TQ-b4-nl223-np11-rf0 (query)                         641.46       281.44       922.89       0.1315          1.3171            1.3127        27.61
IVF-TQ-b4-nl223-np14-rf0 (query)                         641.46       301.26       942.71       0.1315          1.3171            1.3127        27.61
IVF-TQ-b4-nl223-np21-rf0 (query)                         641.46       332.67       974.13       0.1315          1.3171            1.3127        27.61
IVF-TQ-b4-nl223-np11-rf10 (query)                        641.46       490.05     1_131.51       0.2970          1.0929            1.0979        27.61
IVF-TQ-b4-nl223-np11-rf20 (query)                        641.46       782.90     1_424.35       0.3883          1.0643            1.0493        27.61
IVF-TQ-b4-nl223-np14-rf10 (query)                        641.46       537.31     1_178.77       0.2970          1.0929            1.0979        27.61
IVF-TQ-b4-nl223-np14-rf20 (query)                        641.46       810.35     1_451.80       0.3883          1.0643            1.0493        27.61
IVF-TQ-b4-nl223-np21-rf10 (query)                        641.46       541.75     1_183.21       0.2970          1.0929            1.0979        27.61
IVF-TQ-b4-nl223-np21-rf20 (query)                        641.46       858.36     1_499.81       0.3883          1.0643            1.0493        27.61
IVF-TQ-b4-nl223 (self)                                   641.46     1_568.50     2_209.95       0.3882          1.0646            1.0495        27.61
IVF-TQ-b4-nl316-np15-rf0 (query)                         690.51       292.10       982.61       0.1315          1.3171            1.3127        28.19
IVF-TQ-b4-nl316-np17-rf0 (query)                         690.51       305.82       996.33       0.1315          1.3171            1.3127        28.19
IVF-TQ-b4-nl316-np25-rf0 (query)                         690.51       335.80     1_026.31       0.1315          1.3171            1.3127        28.19
IVF-TQ-b4-nl316-np15-rf10 (query)                        690.51       496.20     1_186.71       0.2970          1.0929            1.0979        28.19
IVF-TQ-b4-nl316-np15-rf20 (query)                        690.51       774.86     1_465.38       0.3883          1.0643            1.0493        28.19
IVF-TQ-b4-nl316-np17-rf10 (query)                        690.51       512.88     1_203.40       0.2970          1.0929            1.0979        28.19
IVF-TQ-b4-nl316-np17-rf20 (query)                        690.51       795.75     1_486.27       0.3883          1.0643            1.0493        28.19
IVF-TQ-b4-nl316-np25-rf10 (query)                        690.51       557.28     1_247.80       0.2970          1.0929            1.0979        28.19
IVF-TQ-b4-nl316-np25-rf20 (query)                        690.51       848.30     1_538.82       0.3883          1.0643            1.0493        28.19
IVF-TQ-b4-nl316 (self)                                   690.51     1_559.31     2_249.83       0.3882          1.0646            1.0495        28.19
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 768D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       100.81     1_883.00     1_983.81       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.81     6_408.29     6_509.10       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              599.90       941.54     1_541.45       0.1292          1.2710            1.2627        21.33
ExhaustiveTQ-b2-rf5 (query)                              599.90     1_026.55     1_626.46       0.2468          1.1062            1.1333        21.33
ExhaustiveTQ-b2-rf10 (query)                             599.90     1_173.82     1_773.73       0.3000          1.0773            1.0631        21.33
ExhaustiveTQ-b2-rf20 (query)                             599.90     1_568.96     2_168.87       0.3956          1.0509            1.0334        21.33
ExhaustiveTQ-b2 (self)                                   599.90     5_139.35     5_739.25       0.3973          1.0507            1.0331        21.33
ExhaustiveTQ-b4-rf0 (query)                              740.74     1_793.47     2_534.22       0.1340          1.2532            1.2592        39.64
ExhaustiveTQ-b4-rf5 (query)                              740.74     1_833.35     2_574.09       0.2400          1.1136            1.1402        39.64
ExhaustiveTQ-b4-rf10 (query)                             740.74     1_957.49     2_698.24       0.2870          1.0888            1.1143        39.64
ExhaustiveTQ-b4-rf20 (query)                             740.74     2_353.43     3_094.18       0.3752          1.0657            1.0806        39.64
ExhaustiveTQ-b4 (self)                                   740.74     7_735.27     8_476.01       0.3767          1.0654            1.0640        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                          874.56       304.24     1_178.79       0.1292          1.2710            1.2627        22.42
IVF-TQ-b2-nl158-np12-rf0 (query)                         874.56       314.09     1_188.65       0.1292          1.2710            1.2628        22.42
IVF-TQ-b2-nl158-np17-rf0 (query)                         874.56       334.44     1_209.00       0.1292          1.2710            1.2628        22.42
IVF-TQ-b2-nl158-np7-rf10 (query)                         874.56       495.34     1_369.90       0.3000          1.0773            1.0631        22.42
IVF-TQ-b2-nl158-np7-rf20 (query)                         874.56       806.78     1_681.34       0.3956          1.0509            1.0334        22.42
IVF-TQ-b2-nl158-np12-rf10 (query)                        874.56       530.97     1_405.53       0.3000          1.0773            1.0631        22.42
IVF-TQ-b2-nl158-np12-rf20 (query)                        874.56       856.39     1_730.95       0.3956          1.0509            1.0334        22.42
IVF-TQ-b2-nl158-np17-rf10 (query)                        874.56       562.45     1_437.01       0.3000          1.0773            1.0631        22.42
IVF-TQ-b2-nl158-np17-rf20 (query)                        874.56       887.77     1_762.33       0.3956          1.0509            1.0334        22.42
IVF-TQ-b2-nl158 (self)                                   874.56     1_774.03     2_648.59       0.3973          1.0507            1.0331        22.42
IVF-TQ-b2-nl223-np11-rf0 (query)                         861.06       301.49     1_162.55       0.1292          1.2710            1.2627        22.78
IVF-TQ-b2-nl223-np14-rf0 (query)                         861.06       315.78     1_176.84       0.1292          1.2710            1.2627        22.78
IVF-TQ-b2-nl223-np21-rf0 (query)                         861.06       340.57     1_201.63       0.1292          1.2710            1.2627        22.78
IVF-TQ-b2-nl223-np11-rf10 (query)                        861.06       545.40     1_406.46       0.3000          1.0773            1.0631        22.78
IVF-TQ-b2-nl223-np11-rf20 (query)                        861.06       842.71     1_703.77       0.3956          1.0509            1.0334        22.78
IVF-TQ-b2-nl223-np14-rf10 (query)                        861.06       540.83     1_401.89       0.3000          1.0773            1.0631        22.78
IVF-TQ-b2-nl223-np14-rf20 (query)                        861.06       855.33     1_716.39       0.3956          1.0509            1.0334        22.78
IVF-TQ-b2-nl223-np21-rf10 (query)                        861.06       593.54     1_454.60       0.3000          1.0773            1.0631        22.78
IVF-TQ-b2-nl223-np21-rf20 (query)                        861.06       907.42     1_768.48       0.3956          1.0509            1.0334        22.78
IVF-TQ-b2-nl223 (self)                                   861.06     1_799.32     2_660.38       0.3973          1.0507            1.0331        22.78
IVF-TQ-b2-nl316-np15-rf0 (query)                         917.89       314.60     1_232.49       0.1292          1.2710            1.2627        23.39
IVF-TQ-b2-nl316-np17-rf0 (query)                         917.89       324.07     1_241.96       0.1292          1.2710            1.2627        23.39
IVF-TQ-b2-nl316-np25-rf0 (query)                         917.89       351.69     1_269.58       0.1292          1.2710            1.2627        23.39
IVF-TQ-b2-nl316-np15-rf10 (query)                        917.89       530.55     1_448.44       0.3000          1.0773            1.0631        23.39
IVF-TQ-b2-nl316-np15-rf20 (query)                        917.89       825.12     1_743.01       0.3956          1.0509            1.0334        23.39
IVF-TQ-b2-nl316-np17-rf10 (query)                        917.89       537.70     1_455.59       0.3000          1.0773            1.0631        23.39
IVF-TQ-b2-nl316-np17-rf20 (query)                        917.89       826.13     1_744.02       0.3956          1.0509            1.0334        23.39
IVF-TQ-b2-nl316-np25-rf10 (query)                        917.89       569.71     1_487.60       0.3000          1.0773            1.0631        23.39
IVF-TQ-b2-nl316-np25-rf20 (query)                        917.89       875.51     1_793.39       0.3956          1.0509            1.0334        23.39
IVF-TQ-b2-nl316 (self)                                   917.89     1_803.32     2_721.20       0.3973          1.0507            1.0331        23.39
IVF-TQ-b4-nl158-np7-rf0 (query)                          985.46       403.90     1_389.36       0.1340          1.2532            1.2592        41.16
IVF-TQ-b4-nl158-np12-rf0 (query)                         985.46       456.76     1_442.21       0.1340          1.2532            1.2592        41.16
IVF-TQ-b4-nl158-np17-rf0 (query)                         985.46       519.23     1_504.69       0.1340          1.2532            1.2592        41.16
IVF-TQ-b4-nl158-np7-rf10 (query)                         985.46       632.01     1_617.47       0.2870          1.0888            1.1143        41.16
IVF-TQ-b4-nl158-np7-rf20 (query)                         985.46       948.79     1_934.24       0.3752          1.0657            1.0807        41.16
IVF-TQ-b4-nl158-np12-rf10 (query)                        985.46       690.29     1_675.75       0.2870          1.0888            1.1143        41.16
IVF-TQ-b4-nl158-np12-rf20 (query)                        985.46     1_012.47     1_997.92       0.3752          1.0657            1.0806        41.16
IVF-TQ-b4-nl158-np17-rf10 (query)                        985.46       735.29     1_720.74       0.2870          1.0888            1.1143        41.16
IVF-TQ-b4-nl158-np17-rf20 (query)                        985.46     1_077.62     2_063.07       0.3752          1.0657            1.0807        41.16
IVF-TQ-b4-nl158 (self)                                   985.46     2_069.81     3_055.26       0.3767          1.0654            1.0641        41.16
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_001.79       432.18     1_433.98       0.1340          1.2532            1.2592        41.69
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_001.79       460.21     1_462.01       0.1340          1.2532            1.2592        41.69
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_001.79       505.27     1_507.06       0.1340          1.2532            1.2592        41.69
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_001.79       669.52     1_671.31       0.2870          1.0888            1.1143        41.69
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_001.79       989.17     1_990.96       0.3752          1.0657            1.0807        41.69
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_001.79       699.93     1_701.73       0.2870          1.0888            1.1143        41.69
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_001.79     1_013.47     2_015.26       0.3752          1.0657            1.0807        41.69
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_001.79       746.08     1_747.87       0.2870          1.0888            1.1143        41.69
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_001.79     1_085.33     2_087.13       0.3752          1.0657            1.0807        41.69
IVF-TQ-b4-nl223 (self)                                 1_001.79     2_123.83     3_125.63       0.3767          1.0654            1.0640        41.69
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_093.94       454.71     1_548.64       0.1340          1.2532            1.2592        42.65
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_093.94       470.58     1_564.51       0.1340          1.2532            1.2592        42.65
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_093.94       515.95     1_609.89       0.1340          1.2532            1.2592        42.65
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_093.94       680.38     1_774.31       0.2870          1.0888            1.1143        42.65
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_093.94       973.24     2_067.18       0.3752          1.0657            1.0806        42.65
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_093.94       695.10     1_789.04       0.2870          1.0888            1.1143        42.65
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_093.94       995.73     2_089.66       0.3752          1.0657            1.0806        42.65
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_093.94       760.19     1_854.12       0.2870          1.0888            1.1143        42.65
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_093.94     1_053.83     2_147.76       0.3752          1.0657            1.0807        42.65
IVF-TQ-b4-nl316 (self)                                 1_093.94     2_135.52     3_229.45       0.3767          1.0654            1.0641        42.65
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 256D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.49       693.01       725.50       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.49     2_347.93     2_380.42       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              143.57       358.52       502.09       0.0755          2.3282            1.9295         7.12
ExhaustiveTQ-b2-rf5 (query)                              143.57       433.53       577.10       0.2072          1.3307            1.3578         7.12
ExhaustiveTQ-b2-rf10 (query)                             143.57       563.64       707.21       0.2886          1.2207            1.2321         7.12
ExhaustiveTQ-b2-rf20 (query)                             143.57       926.13     1_069.70       0.4151          1.1328            1.1147         7.12
ExhaustiveTQ-b2 (self)                                   143.57     3_054.59     3_198.17       0.4136          1.1619            1.1367         7.12
ExhaustiveTQ-b4-rf0 (query)                              225.10       574.05       799.15       0.1023          1.7129            1.7532        13.22
ExhaustiveTQ-b4-rf5 (query)                              225.10       654.92       880.02       0.2385          1.2770            1.2999        13.22
ExhaustiveTQ-b4-rf10 (query)                             225.10       782.85     1_007.95       0.3201          1.1873            1.1952        13.22
ExhaustiveTQ-b4-rf20 (query)                             225.10     1_143.70     1_368.80       0.4480          1.1142            1.1029        13.22
ExhaustiveTQ-b4 (self)                                   225.10     3_828.40     4_053.49       0.4462          1.1397            1.1285        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                          266.30       100.55       366.85       0.0756          2.3281            1.9295         7.60
IVF-TQ-b2-nl158-np12-rf0 (query)                         266.30       116.09       382.39       0.0755          2.3282            1.9295         7.60
IVF-TQ-b2-nl158-np17-rf0 (query)                         266.30       144.15       410.45       0.0755          2.3282            1.9295         7.60
IVF-TQ-b2-nl158-np7-rf10 (query)                         266.30       282.40       548.70       0.2886          1.2207            1.2321         7.60
IVF-TQ-b2-nl158-np7-rf20 (query)                         266.30       554.15       820.45       0.4151          1.1328            1.1147         7.60
IVF-TQ-b2-nl158-np12-rf10 (query)                        266.30       309.83       576.13       0.2886          1.2207            1.2321         7.60
IVF-TQ-b2-nl158-np12-rf20 (query)                        266.30       616.82       883.13       0.4151          1.1328            1.1147         7.60
IVF-TQ-b2-nl158-np17-rf10 (query)                        266.30       341.27       607.58       0.2886          1.2207            1.2321         7.60
IVF-TQ-b2-nl158-np17-rf20 (query)                        266.30       677.55       943.85       0.4151          1.1328            1.1147         7.60
IVF-TQ-b2-nl158 (self)                                   266.30     1_048.22     1_314.52       0.4136          1.1619            1.1367         7.60
IVF-TQ-b2-nl223-np11-rf0 (query)                         258.41       105.93       364.35       0.0756          2.3281            1.9295         7.74
IVF-TQ-b2-nl223-np14-rf0 (query)                         258.41       111.14       369.55       0.0755          2.3282            1.9295         7.74
IVF-TQ-b2-nl223-np21-rf0 (query)                         258.41       135.66       394.07       0.0755          2.3282            1.9295         7.74
IVF-TQ-b2-nl223-np11-rf10 (query)                        258.41       267.00       525.41       0.2886          1.2207            1.2321         7.74
IVF-TQ-b2-nl223-np11-rf20 (query)                        258.41       509.97       768.39       0.4151          1.1328            1.1147         7.74
IVF-TQ-b2-nl223-np14-rf10 (query)                        258.41       272.41       530.83       0.2886          1.2207            1.2321         7.74
IVF-TQ-b2-nl223-np14-rf20 (query)                        258.41       570.77       829.18       0.4151          1.1328            1.1147         7.74
IVF-TQ-b2-nl223-np21-rf10 (query)                        258.41       309.48       567.89       0.2886          1.2207            1.2321         7.74
IVF-TQ-b2-nl223-np21-rf20 (query)                        258.41       583.46       841.87       0.4151          1.1328            1.1147         7.74
IVF-TQ-b2-nl223 (self)                                   258.41     1_037.85     1_296.26       0.4136          1.1619            1.1367         7.74
IVF-TQ-b2-nl316-np15-rf0 (query)                         274.58       110.58       385.16       0.0756          2.3281            1.9295         7.92
IVF-TQ-b2-nl316-np17-rf0 (query)                         274.58       114.33       388.91       0.0755          2.3282            1.9295         7.92
IVF-TQ-b2-nl316-np25-rf0 (query)                         274.58       133.75       408.33       0.0755          2.3282            1.9295         7.92
IVF-TQ-b2-nl316-np15-rf10 (query)                        274.58       267.61       542.20       0.2886          1.2207            1.2321         7.92
IVF-TQ-b2-nl316-np15-rf20 (query)                        274.58       499.34       773.93       0.4151          1.1328            1.1147         7.92
IVF-TQ-b2-nl316-np17-rf10 (query)                        274.58       272.69       547.28       0.2886          1.2207            1.2321         7.92
IVF-TQ-b2-nl316-np17-rf20 (query)                        274.58       511.61       786.20       0.4151          1.1328            1.1147         7.92
IVF-TQ-b2-nl316-np25-rf10 (query)                        274.58       298.41       572.99       0.2886          1.2207            1.2322         7.92
IVF-TQ-b2-nl316-np25-rf20 (query)                        274.58       559.22       833.80       0.4151          1.1328            1.1147         7.92
IVF-TQ-b2-nl316 (self)                                   274.58     1_043.93     1_318.52       0.4136          1.1619            1.1367         7.92
IVF-TQ-b4-nl158-np7-rf0 (query)                          338.25       137.07       475.32       0.1023          1.7129            1.7533        13.85
IVF-TQ-b4-nl158-np12-rf0 (query)                         338.25       161.79       500.03       0.1023          1.7129            1.7533        13.85
IVF-TQ-b4-nl158-np17-rf0 (query)                         338.25       198.12       536.37       0.1023          1.7130            1.7533        13.85
IVF-TQ-b4-nl158-np7-rf10 (query)                         338.25       325.44       663.68       0.3201          1.1873            1.1952        13.85
IVF-TQ-b4-nl158-np7-rf20 (query)                         338.25       604.85       943.10       0.4481          1.1142            1.1029        13.85
IVF-TQ-b4-nl158-np12-rf10 (query)                        338.25       376.48       714.72       0.3201          1.1873            1.1952        13.85
IVF-TQ-b4-nl158-np12-rf20 (query)                        338.25       677.87     1_016.12       0.4480          1.1142            1.1029        13.85
IVF-TQ-b4-nl158-np17-rf10 (query)                        338.25       410.15       748.40       0.3201          1.1873            1.1952        13.85
IVF-TQ-b4-nl158-np17-rf20 (query)                        338.25       756.80     1_095.05       0.4480          1.1142            1.1029        13.85
IVF-TQ-b4-nl158 (self)                                   338.25     1_140.49     1_478.73       0.4462          1.1397            1.1285        13.85
IVF-TQ-b4-nl223-np11-rf0 (query)                         338.36       144.78       483.14       0.1023          1.7129            1.7532        14.05
IVF-TQ-b4-nl223-np14-rf0 (query)                         338.36       154.45       492.81       0.1023          1.7129            1.7532        14.05
IVF-TQ-b4-nl223-np21-rf0 (query)                         338.36       193.88       532.24       0.1023          1.7129            1.7532        14.05
IVF-TQ-b4-nl223-np11-rf10 (query)                        338.36       320.29       658.65       0.3201          1.1873            1.1952        14.05
IVF-TQ-b4-nl223-np11-rf20 (query)                        338.36       565.57       903.93       0.4480          1.1142            1.1029        14.05
IVF-TQ-b4-nl223-np14-rf10 (query)                        338.36       329.50       667.86       0.3201          1.1873            1.1952        14.05
IVF-TQ-b4-nl223-np14-rf20 (query)                        338.36       584.02       922.39       0.4480          1.1142            1.1029        14.05
IVF-TQ-b4-nl223-np21-rf10 (query)                        338.36       384.65       723.01       0.3201          1.1873            1.1952        14.05
IVF-TQ-b4-nl223-np21-rf20 (query)                        338.36       656.05       994.41       0.4480          1.1142            1.1029        14.05
IVF-TQ-b4-nl223 (self)                                   338.36     1_062.74     1_401.10       0.4462          1.1397            1.1285        14.05
IVF-TQ-b4-nl316-np15-rf0 (query)                         359.99       151.03       511.01       0.1023          1.7129            1.7532        14.32
IVF-TQ-b4-nl316-np17-rf0 (query)                         359.99       165.06       525.04       0.1023          1.7129            1.7532        14.32
IVF-TQ-b4-nl316-np25-rf0 (query)                         359.99       189.67       549.66       0.1023          1.7129            1.7532        14.32
IVF-TQ-b4-nl316-np15-rf10 (query)                        359.99       327.89       687.88       0.3201          1.1873            1.1952        14.32
IVF-TQ-b4-nl316-np15-rf20 (query)                        359.99       555.01       915.00       0.4480          1.1142            1.1029        14.32
IVF-TQ-b4-nl316-np17-rf10 (query)                        359.99       324.91       684.89       0.3201          1.1873            1.1952        14.32
IVF-TQ-b4-nl316-np17-rf20 (query)                        359.99       564.52       924.51       0.4480          1.1142            1.1029        14.32
IVF-TQ-b4-nl316-np25-rf10 (query)                        359.99       365.80       725.79       0.3201          1.1873            1.1952        14.32
IVF-TQ-b4-nl316-np25-rf20 (query)                        359.99       616.58       976.57       0.4480          1.1142            1.1029        14.32
IVF-TQ-b4-nl316 (self)                                   359.99     1_067.00     1_426.99       0.4462          1.1397            1.1286        14.32
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 512D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.17     1_286.06     1_354.23       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.17     4_387.57     4_455.74       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              335.09       631.10       966.19       0.0844          1.6537            1.5906        13.97
ExhaustiveTQ-b2-rf5 (query)                              335.09       713.80     1_048.89       0.2173          1.2230            1.2550        13.97
ExhaustiveTQ-b2-rf10 (query)                             335.09       851.54     1_186.63       0.2887          1.1550            1.1707        13.97
ExhaustiveTQ-b2-rf20 (query)                             335.09     1_239.55     1_574.65       0.4020          1.0974            1.0847        13.97
ExhaustiveTQ-b2 (self)                                   335.09     4_041.41     4_376.51       0.4025          1.1135            1.0971        13.97
ExhaustiveTQ-b4-rf0 (query)                              454.21     1_100.70     1_554.91       0.1045          1.5026            1.5346        26.18
ExhaustiveTQ-b4-rf5 (query)                              454.21     1_204.46     1_658.68       0.2294          1.2109            1.2410        26.18
ExhaustiveTQ-b4-rf10 (query)                             454.21     1_345.04     1_799.26       0.2943          1.1499            1.1675        26.18
ExhaustiveTQ-b4-rf20 (query)                             454.21     1_718.01     2_172.23       0.4029          1.0975            1.0930        26.18
ExhaustiveTQ-b4 (self)                                   454.21     5_679.24     6_133.46       0.4037          1.1130            1.1087        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                          524.00       185.37       709.37       0.0844          1.6537            1.5905        14.76
IVF-TQ-b2-nl158-np12-rf0 (query)                         524.00       207.01       731.01       0.0844          1.6537            1.5905        14.76
IVF-TQ-b2-nl158-np17-rf0 (query)                         524.00       232.34       756.34       0.0844          1.6537            1.5905        14.76
IVF-TQ-b2-nl158-np7-rf10 (query)                         524.00       394.65       918.65       0.2887          1.1550            1.1707        14.76
IVF-TQ-b2-nl158-np7-rf20 (query)                         524.00       701.45     1_225.45       0.4020          1.0974            1.0847        14.76
IVF-TQ-b2-nl158-np12-rf10 (query)                        524.00       427.02       951.02       0.2887          1.1550            1.1707        14.76
IVF-TQ-b2-nl158-np12-rf20 (query)                        524.00       774.97     1_298.97       0.4020          1.0974            1.0847        14.76
IVF-TQ-b2-nl158-np17-rf10 (query)                        524.00       450.36       974.36       0.2887          1.1550            1.1707        14.76
IVF-TQ-b2-nl158-np17-rf20 (query)                        524.00       784.94     1_308.94       0.4020          1.0974            1.0847        14.76
IVF-TQ-b2-nl158 (self)                                   524.00     1_391.49     1_915.49       0.4025          1.1135            1.0971        14.76
IVF-TQ-b2-nl223-np11-rf0 (query)                         521.64       194.01       715.65       0.0844          1.6537            1.5906        14.99
IVF-TQ-b2-nl223-np14-rf0 (query)                         521.64       204.12       725.76       0.0844          1.6537            1.5906        14.99
IVF-TQ-b2-nl223-np21-rf0 (query)                         521.64       226.92       748.56       0.0844          1.6537            1.5906        14.99
IVF-TQ-b2-nl223-np11-rf10 (query)                        521.64       396.21       917.85       0.2887          1.1550            1.1707        14.99
IVF-TQ-b2-nl223-np11-rf20 (query)                        521.64       685.69     1_207.33       0.4020          1.0974            1.0847        14.99
IVF-TQ-b2-nl223-np14-rf10 (query)                        521.64       429.33       950.97       0.2887          1.1550            1.1707        14.99
IVF-TQ-b2-nl223-np14-rf20 (query)                        521.64       704.47     1_226.11       0.4020          1.0974            1.0847        14.99
IVF-TQ-b2-nl223-np21-rf10 (query)                        521.64       432.14       953.78       0.2887          1.1550            1.1707        14.99
IVF-TQ-b2-nl223-np21-rf20 (query)                        521.64       748.44     1_270.08       0.4020          1.0974            1.0847        14.99
IVF-TQ-b2-nl223 (self)                                   521.64     1_400.27     1_921.91       0.4025          1.1135            1.0971        14.99
IVF-TQ-b2-nl316-np15-rf0 (query)                         567.40       204.66       772.06       0.0844          1.6537            1.5906        15.33
IVF-TQ-b2-nl316-np17-rf0 (query)                         567.40       208.66       776.05       0.0844          1.6537            1.5906        15.33
IVF-TQ-b2-nl316-np25-rf0 (query)                         567.40       230.99       798.39       0.0844          1.6537            1.5905        15.33
IVF-TQ-b2-nl316-np15-rf10 (query)                        567.40       395.79       963.19       0.2887          1.1550            1.1707        15.33
IVF-TQ-b2-nl316-np15-rf20 (query)                        567.40       649.72     1_217.11       0.4020          1.0974            1.0847        15.33
IVF-TQ-b2-nl316-np17-rf10 (query)                        567.40       399.28       966.67       0.2887          1.1550            1.1707        15.33
IVF-TQ-b2-nl316-np17-rf20 (query)                        567.40       660.75     1_228.15       0.4020          1.0974            1.0847        15.33
IVF-TQ-b2-nl316-np25-rf10 (query)                        567.40       428.30       995.70       0.2887          1.1550            1.1707        15.33
IVF-TQ-b2-nl316-np25-rf20 (query)                        567.40       710.97     1_278.36       0.4020          1.0974            1.0847        15.33
IVF-TQ-b2-nl316 (self)                                   567.40     1_415.84     1_983.23       0.4025          1.1135            1.0971        15.33
IVF-TQ-b4-nl158-np7-rf0 (query)                          623.42       255.29       878.71       0.1045          1.5026            1.5346        27.25
IVF-TQ-b4-nl158-np12-rf0 (query)                         623.42       292.33       915.75       0.1045          1.5026            1.5346        27.25
IVF-TQ-b4-nl158-np17-rf0 (query)                         623.42       324.41       947.83       0.1045          1.5026            1.5346        27.25
IVF-TQ-b4-nl158-np7-rf10 (query)                         623.42       478.87     1_102.29       0.2943          1.1499            1.1675        27.25
IVF-TQ-b4-nl158-np7-rf20 (query)                         623.42       795.79     1_419.20       0.4029          1.0975            1.0930        27.25
IVF-TQ-b4-nl158-np12-rf10 (query)                        623.42       523.31     1_146.72       0.2943          1.1499            1.1675        27.25
IVF-TQ-b4-nl158-np12-rf20 (query)                        623.42       856.24     1_479.66       0.4029          1.0975            1.0930        27.25
IVF-TQ-b4-nl158-np17-rf10 (query)                        623.42       564.68     1_188.09       0.2943          1.1499            1.1675        27.25
IVF-TQ-b4-nl158-np17-rf20 (query)                        623.42       906.10     1_529.52       0.4029          1.0975            1.0930        27.25
IVF-TQ-b4-nl158 (self)                                   623.42     1_571.70     2_195.11       0.4037          1.1130            1.1087        27.25
IVF-TQ-b4-nl223-np11-rf0 (query)                         643.47       273.11       916.58       0.1045          1.5027            1.5346        27.58
IVF-TQ-b4-nl223-np14-rf0 (query)                         643.47       289.96       933.43       0.1045          1.5027            1.5346        27.58
IVF-TQ-b4-nl223-np21-rf0 (query)                         643.47       329.01       972.48       0.1045          1.5026            1.5346        27.58
IVF-TQ-b4-nl223-np11-rf10 (query)                        643.47       488.45     1_131.92       0.2943          1.1499            1.1675        27.58
IVF-TQ-b4-nl223-np11-rf20 (query)                        643.47       785.47     1_428.95       0.4029          1.0975            1.0930        27.58
IVF-TQ-b4-nl223-np14-rf10 (query)                        643.47       541.73     1_185.20       0.2943          1.1499            1.1675        27.58
IVF-TQ-b4-nl223-np14-rf20 (query)                        643.47       814.64     1_458.11       0.4029          1.0975            1.0930        27.58
IVF-TQ-b4-nl223-np21-rf10 (query)                        643.47       551.58     1_195.05       0.2943          1.1499            1.1675        27.58
IVF-TQ-b4-nl223-np21-rf20 (query)                        643.47       881.02     1_524.49       0.4029          1.0975            1.0930        27.58
IVF-TQ-b4-nl223 (self)                                   643.47     1_600.40     2_243.87       0.4037          1.1130            1.1087        27.58
IVF-TQ-b4-nl316-np15-rf0 (query)                         684.22       285.77       969.99       0.1045          1.5026            1.5346        28.09
IVF-TQ-b4-nl316-np17-rf0 (query)                         684.22       296.22       980.44       0.1045          1.5026            1.5346        28.09
IVF-TQ-b4-nl316-np25-rf0 (query)                         684.22       334.74     1_018.96       0.1045          1.5026            1.5346        28.09
IVF-TQ-b4-nl316-np15-rf10 (query)                        684.22       487.15     1_171.38       0.2943          1.1499            1.1675        28.09
IVF-TQ-b4-nl316-np15-rf20 (query)                        684.22       751.26     1_435.48       0.4029          1.0975            1.0930        28.09
IVF-TQ-b4-nl316-np17-rf10 (query)                        684.22       500.29     1_184.51       0.2943          1.1499            1.1675        28.09
IVF-TQ-b4-nl316-np17-rf20 (query)                        684.22       769.13     1_453.35       0.4029          1.0975            1.0930        28.09
IVF-TQ-b4-nl316-np25-rf10 (query)                        684.22       544.25     1_228.47       0.2943          1.1499            1.1675        28.09
IVF-TQ-b4-nl316-np25-rf20 (query)                        684.22       825.46     1_509.68       0.4029          1.0975            1.0930        28.09
IVF-TQ-b4-nl316 (self)                                   684.22     1_614.17     2_298.39       0.4037          1.1130            1.1087        28.09
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 768D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       101.32     1_819.24     1_920.57       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.32     7_342.06     7_443.39       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              620.20       990.00     1_610.20       0.0841          1.5107            1.4226        21.33
ExhaustiveTQ-b2-rf5 (query)                              620.20     1_119.89     1_740.09       0.2144          1.1739            1.2057        21.33
ExhaustiveTQ-b2-rf10 (query)                             620.20     1_237.64     1_857.84       0.2771          1.1266            1.1512        21.33
ExhaustiveTQ-b2-rf20 (query)                             620.20     1_604.86     2_225.06       0.3769          1.0843            1.0725        21.33
ExhaustiveTQ-b2 (self)                                   620.20     5_145.89     5_766.09       0.3767          1.0935            1.0803        21.33
ExhaustiveTQ-b4-rf0 (query)                              728.94     1_730.68     2_459.62       0.0985          1.4232            1.4110        39.64
ExhaustiveTQ-b4-rf5 (query)                              728.94     1_828.86     2_557.80       0.2167          1.1746            1.2047        39.64
ExhaustiveTQ-b4-rf10 (query)                             728.94     1_976.66     2_705.60       0.2692          1.1311            1.1558        39.64
ExhaustiveTQ-b4-rf20 (query)                             728.94     2_349.64     3_078.58       0.3604          1.0923            1.1071        39.64
ExhaustiveTQ-b4 (self)                                   728.94     7_743.40     8_472.34       0.3608          1.1024            1.1182        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                          868.84       280.15     1_148.99       0.0841          1.5107            1.4226        22.46
IVF-TQ-b2-nl158-np12-rf0 (query)                         868.84       310.73     1_179.57       0.0841          1.5107            1.4226        22.46
IVF-TQ-b2-nl158-np17-rf0 (query)                         868.84       336.96     1_205.80       0.0841          1.5107            1.4226        22.46
IVF-TQ-b2-nl158-np7-rf10 (query)                         868.84       508.34     1_377.19       0.2771          1.1266            1.1512        22.46
IVF-TQ-b2-nl158-np7-rf20 (query)                         868.84       852.94     1_721.79       0.3769          1.0843            1.0725        22.46
IVF-TQ-b2-nl158-np12-rf10 (query)                        868.84       543.00     1_411.84       0.2771          1.1267            1.1512        22.46
IVF-TQ-b2-nl158-np12-rf20 (query)                        868.84       904.96     1_773.80       0.3769          1.0843            1.0725        22.46
IVF-TQ-b2-nl158-np17-rf10 (query)                        868.84       573.44     1_442.28       0.2771          1.1266            1.1512        22.46
IVF-TQ-b2-nl158-np17-rf20 (query)                        868.84       946.05     1_814.89       0.3769          1.0843            1.0725        22.46
IVF-TQ-b2-nl158 (self)                                   868.84     1_785.04     2_653.88       0.3767          1.0935            1.0803        22.46
IVF-TQ-b2-nl223-np11-rf0 (query)                         883.20       296.78     1_179.97       0.0841          1.5107            1.4226        22.78
IVF-TQ-b2-nl223-np14-rf0 (query)                         883.20       307.69     1_190.89       0.0841          1.5107            1.4226        22.78
IVF-TQ-b2-nl223-np21-rf0 (query)                         883.20       340.01     1_223.21       0.0841          1.5107            1.4226        22.78
IVF-TQ-b2-nl223-np11-rf10 (query)                        883.20       523.74     1_406.94       0.2771          1.1266            1.1512        22.78
IVF-TQ-b2-nl223-np11-rf20 (query)                        883.20       840.95     1_724.15       0.3770          1.0843            1.0725        22.78
IVF-TQ-b2-nl223-np14-rf10 (query)                        883.20       535.80     1_418.99       0.2771          1.1266            1.1512        22.78
IVF-TQ-b2-nl223-np14-rf20 (query)                        883.20       863.26     1_746.46       0.3770          1.0843            1.0725        22.78
IVF-TQ-b2-nl223-np21-rf10 (query)                        883.20       571.67     1_454.87       0.2771          1.1266            1.1512        22.78
IVF-TQ-b2-nl223-np21-rf20 (query)                        883.20       918.51     1_801.71       0.3769          1.0843            1.0725        22.78
IVF-TQ-b2-nl223 (self)                                   883.20     1_839.58     2_722.78       0.3767          1.0935            1.0803        22.78
IVF-TQ-b2-nl316-np15-rf0 (query)                         916.12       307.30     1_223.42       0.0841          1.5107            1.4226        23.35
IVF-TQ-b2-nl316-np17-rf0 (query)                         916.12       316.64     1_232.76       0.0841          1.5107            1.4226        23.35
IVF-TQ-b2-nl316-np25-rf0 (query)                         916.12       342.33     1_258.45       0.0841          1.5107            1.4226        23.35
IVF-TQ-b2-nl316-np15-rf10 (query)                        916.12       551.93     1_468.05       0.2771          1.1266            1.1512        23.35
IVF-TQ-b2-nl316-np15-rf20 (query)                        916.12       819.51     1_735.62       0.3770          1.0843            1.0724        23.35
IVF-TQ-b2-nl316-np17-rf10 (query)                        916.12       528.23     1_444.35       0.2771          1.1266            1.1512        23.35
IVF-TQ-b2-nl316-np17-rf20 (query)                        916.12       829.25     1_745.37       0.3770          1.0843            1.0724        23.35
IVF-TQ-b2-nl316-np25-rf10 (query)                        916.12       561.29     1_477.41       0.2771          1.1266            1.1512        23.35
IVF-TQ-b2-nl316-np25-rf20 (query)                        916.12       872.74     1_788.86       0.3770          1.0843            1.0725        23.35
IVF-TQ-b2-nl316 (self)                                   916.12     1_882.08     2_798.20       0.3767          1.0935            1.0803        23.35
IVF-TQ-b4-nl158-np7-rf0 (query)                          989.44       393.68     1_383.12       0.0985          1.4232            1.4110        41.26
IVF-TQ-b4-nl158-np12-rf0 (query)                         989.44       445.93     1_435.37       0.0985          1.4232            1.4110        41.26
IVF-TQ-b4-nl158-np17-rf0 (query)                         989.44       499.24     1_488.68       0.0985          1.4232            1.4110        41.26
IVF-TQ-b4-nl158-np7-rf10 (query)                         989.44       640.20     1_629.64       0.2692          1.1311            1.1558        41.26
IVF-TQ-b4-nl158-np7-rf20 (query)                         989.44     1_002.62     1_992.06       0.3604          1.0923            1.1071        41.26
IVF-TQ-b4-nl158-np12-rf10 (query)                        989.44       698.79     1_688.23       0.2692          1.1311            1.1558        41.26
IVF-TQ-b4-nl158-np12-rf20 (query)                        989.44     1_066.37     2_055.81       0.3604          1.0923            1.1071        41.26
IVF-TQ-b4-nl158-np17-rf10 (query)                        989.44       751.21     1_740.65       0.2692          1.1311            1.1558        41.26
IVF-TQ-b4-nl158-np17-rf20 (query)                        989.44     1_136.87     2_126.31       0.3604          1.0923            1.1071        41.26
IVF-TQ-b4-nl158 (self)                                   989.44     2_143.37     3_132.81       0.3608          1.1024            1.1182        41.26
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_008.51       443.19     1_451.70       0.0985          1.4232            1.4110        41.69
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_008.51       446.62     1_455.13       0.0985          1.4232            1.4110        41.69
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_008.51       500.45     1_508.96       0.0985          1.4232            1.4110        41.69
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_008.51       668.52     1_677.03       0.2692          1.1311            1.1558        41.69
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_008.51       982.94     1_991.45       0.3604          1.0923            1.1071        41.69
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_008.51       684.80     1_693.31       0.2692          1.1311            1.1558        41.69
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_008.51     1_021.75     2_030.26       0.3604          1.0923            1.1071        41.69
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_008.51       746.27     1_754.78       0.2692          1.1311            1.1558        41.69
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_008.51     1_096.70     2_105.21       0.3604          1.0923            1.1071        41.69
IVF-TQ-b4-nl223 (self)                                 1_008.51     2_158.36     3_166.87       0.3608          1.1024            1.1182        41.69
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_080.84       442.36     1_523.19       0.0985          1.4232            1.4110        42.56
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_080.84       456.35     1_537.18       0.0985          1.4232            1.4110        42.56
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_080.84       507.31     1_588.15       0.0985          1.4232            1.4110        42.56
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_080.84       698.50     1_779.33       0.2692          1.1311            1.1558        42.56
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_080.84       971.42     2_052.26       0.3604          1.0923            1.1071        42.56
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_080.84       677.70     1_758.53       0.2692          1.1311            1.1558        42.56
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_080.84       980.18     2_061.01       0.3604          1.0923            1.1071        42.56
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_080.84       730.84     1_811.67       0.2692          1.1311            1.1558        42.56
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_080.84     1_050.79     2_131.62       0.3604          1.0923            1.1071        42.56
IVF-TQ-b4-nl316 (self)                                 1_080.84     2_219.88     3_300.71       0.3608          1.1024            1.1182        42.56
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Cell embeddings data

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 256D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.79       704.14       736.93       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.79     2_379.76     2_412.55       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              143.94       360.53       504.47       0.7918          1.0898            1.0632         7.12
ExhaustiveTQ-b2-rf5 (query)                              143.94       436.94       580.87       0.9995          1.0000            1.0000         7.12
ExhaustiveTQ-b2-rf10 (query)                             143.94       576.60       720.54       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b2-rf20 (query)                             143.94       945.72     1_089.65       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b2 (self)                                   143.94     3_113.09     3_257.02       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b4-rf0 (query)                              223.92       574.78       798.70       0.8727          1.0322            1.0183        13.22
ExhaustiveTQ-b4-rf5 (query)                              223.92       655.31       879.23       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4-rf10 (query)                             223.92       785.80     1_009.72       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4-rf20 (query)                             223.92     1_152.55     1_376.47       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4 (self)                                   223.92     3_815.06     4_038.98       1.0000          1.0000            1.0000        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                          265.49       115.38       380.87       0.7917          1.0896            1.0634         7.61
IVF-TQ-b2-nl158-np12-rf0 (query)                         265.49       154.97       420.46       0.7918          1.0898            1.0632         7.61
IVF-TQ-b2-nl158-np17-rf0 (query)                         265.49       190.65       456.14       0.7918          1.0898            1.0632         7.61
IVF-TQ-b2-nl158-np7-rf10 (query)                         265.49       304.31       569.81       0.9973          1.0006            1.0000         7.61
IVF-TQ-b2-nl158-np7-rf20 (query)                         265.49       553.55       819.05       0.9973          1.0006            1.0000         7.61
IVF-TQ-b2-nl158-np12-rf10 (query)                        265.49       359.78       625.27       1.0000          1.0000            1.0000         7.61
IVF-TQ-b2-nl158-np12-rf20 (query)                        265.49       645.23       910.72       1.0000          1.0000            1.0000         7.61
IVF-TQ-b2-nl158-np17-rf10 (query)                        265.49       404.17       669.67       1.0000          1.0000            1.0000         7.61
IVF-TQ-b2-nl158-np17-rf20 (query)                        265.49       713.43       978.92       1.0000          1.0000            1.0000         7.61
IVF-TQ-b2-nl158 (self)                                   265.49     1_109.25     1_374.74       1.0000          1.0000            1.0000         7.61
IVF-TQ-b2-nl223-np11-rf0 (query)                         258.47       126.15       384.62       0.7919          1.0895            1.0632         7.75
IVF-TQ-b2-nl223-np14-rf0 (query)                         258.47       143.80       402.27       0.7919          1.0897            1.0632         7.75
IVF-TQ-b2-nl223-np21-rf0 (query)                         258.47       182.39       440.86       0.7918          1.0898            1.0632         7.75
IVF-TQ-b2-nl223-np11-rf10 (query)                        258.47       310.47       568.94       0.9995          1.0001            1.0000         7.75
IVF-TQ-b2-nl223-np11-rf20 (query)                        258.47       568.16       826.63       0.9995          1.0001            1.0000         7.75
IVF-TQ-b2-nl223-np14-rf10 (query)                        258.47       334.38       592.85       0.9999          1.0000            1.0000         7.75
IVF-TQ-b2-nl223-np14-rf20 (query)                        258.47       609.82       868.29       0.9999          1.0000            1.0000         7.75
IVF-TQ-b2-nl223-np21-rf10 (query)                        258.47       388.93       647.40       1.0000          1.0000            1.0000         7.75
IVF-TQ-b2-nl223-np21-rf20 (query)                        258.47       692.21       950.68       1.0000          1.0000            1.0000         7.75
IVF-TQ-b2-nl223 (self)                                   258.47     1_037.13     1_295.59       1.0000          1.0000            1.0000         7.75
IVF-TQ-b2-nl316-np15-rf0 (query)                         273.91       128.32       402.23       0.7918          1.0898            1.0632         7.93
IVF-TQ-b2-nl316-np17-rf0 (query)                         273.91       136.71       410.63       0.7918          1.0898            1.0632         7.93
IVF-TQ-b2-nl316-np25-rf0 (query)                         273.91       168.51       442.43       0.7918          1.0898            1.0632         7.93
IVF-TQ-b2-nl316-np15-rf10 (query)                        273.91       302.98       576.89       0.9996          1.0001            1.0000         7.93
IVF-TQ-b2-nl316-np15-rf20 (query)                        273.91       557.77       831.68       0.9996          1.0001            1.0000         7.93
IVF-TQ-b2-nl316-np17-rf10 (query)                        273.91       317.57       591.49       0.9999          1.0000            1.0000         7.93
IVF-TQ-b2-nl316-np17-rf20 (query)                        273.91       577.77       851.68       0.9999          1.0000            1.0000         7.93
IVF-TQ-b2-nl316-np25-rf10 (query)                        273.91       360.46       634.37       1.0000          1.0000            1.0000         7.93
IVF-TQ-b2-nl316-np25-rf20 (query)                        273.91       650.01       923.92       1.0000          1.0000            1.0000         7.93
IVF-TQ-b2-nl316 (self)                                   273.91       986.27     1_260.18       1.0000          1.0000            1.0000         7.93
IVF-TQ-b4-nl158-np7-rf0 (query)                          340.61       165.84       506.45       0.8719          1.0326            1.0188        13.86
IVF-TQ-b4-nl158-np12-rf0 (query)                         340.61       229.17       569.78       0.8727          1.0322            1.0183        13.86
IVF-TQ-b4-nl158-np17-rf0 (query)                         340.61       286.09       626.70       0.8727          1.0322            1.0183        13.86
IVF-TQ-b4-nl158-np7-rf10 (query)                         340.61       349.82       690.43       0.9973          1.0006            1.0000        13.86
IVF-TQ-b4-nl158-np7-rf20 (query)                         340.61       605.23       945.84       0.9973          1.0006            1.0000        13.86
IVF-TQ-b4-nl158-np12-rf10 (query)                        340.61       436.59       777.20       1.0000          1.0000            1.0000        13.86
IVF-TQ-b4-nl158-np12-rf20 (query)                        340.61       724.52     1_065.13       1.0000          1.0000            1.0000        13.86
IVF-TQ-b4-nl158-np17-rf10 (query)                        340.61       501.17       841.78       1.0000          1.0000            1.0000        13.86
IVF-TQ-b4-nl158-np17-rf20 (query)                        340.61       815.27     1_155.88       1.0000          1.0000            1.0000        13.86
IVF-TQ-b4-nl158 (self)                                   340.61     1_219.98     1_560.59       1.0000          1.0000            1.0000        13.86
IVF-TQ-b4-nl223-np11-rf0 (query)                         339.82       176.11       515.93       0.8727          1.0322            1.0184        14.07
IVF-TQ-b4-nl223-np14-rf0 (query)                         339.82       207.07       546.89       0.8727          1.0322            1.0183        14.07
IVF-TQ-b4-nl223-np21-rf0 (query)                         339.82       265.76       605.58       0.8727          1.0322            1.0183        14.07
IVF-TQ-b4-nl223-np11-rf10 (query)                        339.82       362.12       701.93       0.9995          1.0001            1.0000        14.07
IVF-TQ-b4-nl223-np11-rf20 (query)                        339.82       621.37       961.19       0.9995          1.0001            1.0000        14.07
IVF-TQ-b4-nl223-np14-rf10 (query)                        339.82       397.31       737.13       0.9999          1.0000            1.0000        14.07
IVF-TQ-b4-nl223-np14-rf20 (query)                        339.82       672.95     1_012.76       0.9999          1.0000            1.0000        14.07
IVF-TQ-b4-nl223-np21-rf10 (query)                        339.82       477.57       817.39       1.0000          1.0000            1.0000        14.07
IVF-TQ-b4-nl223-np21-rf20 (query)                        339.82       835.29     1_175.11       1.0000          1.0000            1.0000        14.07
IVF-TQ-b4-nl223 (self)                                   339.82     1_139.13     1_478.95       1.0000          1.0000            1.0000        14.07
IVF-TQ-b4-nl316-np15-rf0 (query)                         355.99       177.79       533.78       0.8726          1.0322            1.0184        14.35
IVF-TQ-b4-nl316-np17-rf0 (query)                         355.99       193.56       549.54       0.8727          1.0322            1.0184        14.35
IVF-TQ-b4-nl316-np25-rf0 (query)                         355.99       246.77       602.76       0.8727          1.0322            1.0183        14.35
IVF-TQ-b4-nl316-np15-rf10 (query)                        355.99       354.11       710.10       0.9996          1.0001            1.0000        14.35
IVF-TQ-b4-nl316-np15-rf20 (query)                        355.99       610.92       966.91       0.9996          1.0001            1.0000        14.35
IVF-TQ-b4-nl316-np17-rf10 (query)                        355.99       373.22       729.21       0.9999          1.0000            1.0000        14.35
IVF-TQ-b4-nl316-np17-rf20 (query)                        355.99       636.94       992.93       0.9999          1.0000            1.0000        14.35
IVF-TQ-b4-nl316-np25-rf10 (query)                        355.99       438.50       794.49       1.0000          1.0000            1.0000        14.35
IVF-TQ-b4-nl316-np25-rf20 (query)                        355.99       728.80     1_084.79       1.0000          1.0000            1.0000        14.35
IVF-TQ-b4-nl316 (self)                                   355.99     1_087.23     1_443.22       1.0000          1.0000            1.0000        14.35
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 512D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.05     1_297.58     1_365.63       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.05     4_405.30     4_473.35       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              338.73       630.20       968.93       0.8424          1.0447            1.0331        13.97
ExhaustiveTQ-b2-rf5 (query)                              338.73       769.92     1_108.65       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2-rf10 (query)                             338.73       854.18     1_192.91       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2-rf20 (query)                             338.73     1_249.61     1_588.34       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2 (self)                                   338.73     4_099.00     4_437.73       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b4-rf0 (query)                              448.40     1_107.59     1_555.99       0.8985          1.0191            1.0110        26.18
ExhaustiveTQ-b4-rf5 (query)                              448.40     1_207.00     1_655.41       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4-rf10 (query)                             448.40     1_367.21     1_815.61       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4-rf20 (query)                             448.40     1_729.70     2_178.10       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4 (self)                                   448.40     5_702.54     6_150.94       1.0000          1.0000            1.0000        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                          532.57       215.30       747.87       0.8421          1.0449            1.0333        14.78
IVF-TQ-b2-nl158-np12-rf0 (query)                         532.57       279.44       812.00       0.8424          1.0447            1.0331        14.78
IVF-TQ-b2-nl158-np17-rf0 (query)                         532.57       350.84       883.41       0.8424          1.0447            1.0331        14.78
IVF-TQ-b2-nl158-np7-rf10 (query)                         532.57       426.55       959.11       0.9987          1.0002            1.0000        14.78
IVF-TQ-b2-nl158-np7-rf20 (query)                         532.57       715.60     1_248.17       0.9987          1.0002            1.0000        14.78
IVF-TQ-b2-nl158-np12-rf10 (query)                        532.57       506.01     1_038.58       1.0000          1.0000            1.0000        14.78
IVF-TQ-b2-nl158-np12-rf20 (query)                        532.57       824.32     1_356.88       1.0000          1.0000            1.0000        14.78
IVF-TQ-b2-nl158-np17-rf10 (query)                        532.57       571.71     1_104.28       1.0000          1.0000            1.0000        14.78
IVF-TQ-b2-nl158-np17-rf20 (query)                        532.57       908.76     1_441.33       1.0000          1.0000            1.0000        14.78
IVF-TQ-b2-nl158 (self)                                   532.57     1_425.71     1_958.28       1.0000          1.0000            1.0000        14.78
IVF-TQ-b2-nl223-np11-rf0 (query)                         544.27       233.23       777.51       0.8424          1.0447            1.0331        15.03
IVF-TQ-b2-nl223-np14-rf0 (query)                         544.27       260.76       805.04       0.8424          1.0447            1.0331        15.03
IVF-TQ-b2-nl223-np21-rf0 (query)                         544.27       319.06       863.33       0.8424          1.0447            1.0331        15.03
IVF-TQ-b2-nl223-np11-rf10 (query)                        544.27       435.88       980.15       0.9997          1.0001            1.0000        15.03
IVF-TQ-b2-nl223-np11-rf20 (query)                        544.27       719.12     1_263.39       0.9997          1.0001            1.0000        15.03
IVF-TQ-b2-nl223-np14-rf10 (query)                        544.27       471.60     1_015.87       0.9999          1.0000            1.0000        15.03
IVF-TQ-b2-nl223-np14-rf20 (query)                        544.27       769.62     1_313.89       0.9999          1.0000            1.0000        15.03
IVF-TQ-b2-nl223-np21-rf10 (query)                        544.27       541.78     1_086.05       1.0000          1.0000            1.0000        15.03
IVF-TQ-b2-nl223-np21-rf20 (query)                        544.27       865.38     1_409.65       1.0000          1.0000            1.0000        15.03
IVF-TQ-b2-nl223 (self)                                   544.27     1_376.64     1_920.91       1.0000          1.0000            1.0000        15.03
IVF-TQ-b2-nl316-np15-rf0 (query)                         567.57       234.75       802.32       0.8424          1.0447            1.0331        15.39
IVF-TQ-b2-nl316-np17-rf0 (query)                         567.57       254.26       821.83       0.8424          1.0447            1.0331        15.39
IVF-TQ-b2-nl316-np25-rf0 (query)                         567.57       300.50       868.07       0.8424          1.0447            1.0331        15.39
IVF-TQ-b2-nl316-np15-rf10 (query)                        567.57       438.46     1_006.03       0.9999          1.0000            1.0000        15.39
IVF-TQ-b2-nl316-np15-rf20 (query)                        567.57       722.96     1_290.53       0.9999          1.0000            1.0000        15.39
IVF-TQ-b2-nl316-np17-rf10 (query)                        567.57       450.72     1_018.29       1.0000          1.0000            1.0000        15.39
IVF-TQ-b2-nl316-np17-rf20 (query)                        567.57       748.26     1_315.83       1.0000          1.0000            1.0000        15.39
IVF-TQ-b2-nl316-np25-rf10 (query)                        567.57       515.59     1_083.16       1.0000          1.0000            1.0000        15.39
IVF-TQ-b2-nl316-np25-rf20 (query)                        567.57       827.77     1_395.34       1.0000          1.0000            1.0000        15.39
IVF-TQ-b2-nl316 (self)                                   567.57     1_345.61     1_913.18       1.0000          1.0000            1.0000        15.39
IVF-TQ-b4-nl158-np7-rf0 (query)                          632.16       310.32       942.48       0.8978          1.0193            1.0113        27.29
IVF-TQ-b4-nl158-np12-rf0 (query)                         632.16       427.98     1_060.15       0.8985          1.0191            1.0110        27.29
IVF-TQ-b4-nl158-np17-rf0 (query)                         632.16       533.87     1_166.04       0.8985          1.0191            1.0110        27.29
IVF-TQ-b4-nl158-np7-rf10 (query)                         632.16       525.31     1_157.47       0.9987          1.0002            1.0000        27.29
IVF-TQ-b4-nl158-np7-rf20 (query)                         632.16       813.33     1_445.49       0.9987          1.0002            1.0000        27.29
IVF-TQ-b4-nl158-np12-rf10 (query)                        632.16       658.37     1_290.53       1.0000          1.0000            1.0000        27.29
IVF-TQ-b4-nl158-np12-rf20 (query)                        632.16       980.91     1_613.07       1.0000          1.0000            1.0000        27.29
IVF-TQ-b4-nl158-np17-rf10 (query)                        632.16       767.57     1_399.73       1.0000          1.0000            1.0000        27.29
IVF-TQ-b4-nl158-np17-rf20 (query)                        632.16     1_106.01     1_738.17       1.0000          1.0000            1.0000        27.29
IVF-TQ-b4-nl158 (self)                                   632.16     1_718.07     2_350.23       1.0000          1.0000            1.0000        27.29
IVF-TQ-b4-nl223-np11-rf0 (query)                         633.48       351.74       985.22       0.8984          1.0192            1.0111        27.66
IVF-TQ-b4-nl223-np14-rf0 (query)                         633.48       389.38     1_022.86       0.8985          1.0191            1.0110        27.66
IVF-TQ-b4-nl223-np21-rf0 (query)                         633.48       504.30     1_137.78       0.8985          1.0191            1.0110        27.66
IVF-TQ-b4-nl223-np11-rf10 (query)                        633.48       547.76     1_181.24       0.9997          1.0001            1.0000        27.66
IVF-TQ-b4-nl223-np11-rf20 (query)                        633.48       831.56     1_465.04       0.9997          1.0001            1.0000        27.66
IVF-TQ-b4-nl223-np14-rf10 (query)                        633.48       600.41     1_233.89       0.9999          1.0000            1.0000        27.66
IVF-TQ-b4-nl223-np14-rf20 (query)                        633.48       903.52     1_537.00       0.9999          1.0000            1.0000        27.66
IVF-TQ-b4-nl223-np21-rf10 (query)                        633.48       724.31     1_357.79       1.0000          1.0000            1.0000        27.66
IVF-TQ-b4-nl223-np21-rf20 (query)                        633.48     1_045.26     1_678.75       1.0000          1.0000            1.0000        27.66
IVF-TQ-b4-nl223 (self)                                   633.48     1_654.35     2_287.83       1.0000          1.0000            1.0000        27.66
IVF-TQ-b4-nl316-np15-rf0 (query)                         684.08       342.98     1_027.06       0.8985          1.0191            1.0110        28.20
IVF-TQ-b4-nl316-np17-rf0 (query)                         684.08       368.32     1_052.39       0.8985          1.0191            1.0110        28.20
IVF-TQ-b4-nl316-np25-rf0 (query)                         684.08       464.69     1_148.77       0.8985          1.0191            1.0110        28.20
IVF-TQ-b4-nl316-np15-rf10 (query)                        684.08       544.18     1_228.26       0.9999          1.0000            1.0000        28.20
IVF-TQ-b4-nl316-np15-rf20 (query)                        684.08       830.18     1_514.25       0.9999          1.0000            1.0000        28.20
IVF-TQ-b4-nl316-np17-rf10 (query)                        684.08       582.99     1_267.07       1.0000          1.0000            1.0000        28.20
IVF-TQ-b4-nl316-np17-rf20 (query)                        684.08       870.38     1_554.46       1.0000          1.0000            1.0000        28.20
IVF-TQ-b4-nl316-np25-rf10 (query)                        684.08       676.62     1_360.70       1.0000          1.0000            1.0000        28.20
IVF-TQ-b4-nl316-np25-rf20 (query)                        684.08       992.12     1_676.19       1.0000          1.0000            1.0000        28.20
IVF-TQ-b4-nl316 (self)                                   684.08     1_603.36     2_287.44       1.0000          1.0000            1.0000        28.20
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
ann-search-rs 0.12.2 (commit v0.12.2-23-g8e59906-dirty), run on 2026-11-13
=====================================================================================================================================================
Benchmark: 50k samples, 768D - TurboQuant + IVF
=====================================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio Median dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       100.16     1_873.85     1_974.01       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.16     6_305.06     6_405.22       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              600.87       933.12     1_533.99       0.8736          1.0271            1.0200        21.33
ExhaustiveTQ-b2-rf5 (query)                              600.87     1_019.60     1_620.47       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2-rf10 (query)                             600.87     1_247.77     1_848.64       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2-rf20 (query)                             600.87     1_567.83     2_168.69       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2 (self)                                   600.87     5_172.92     5_773.78       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b4-rf0 (query)                              736.30     1_713.25     2_449.55       0.9097          1.0146            1.0083        39.64
ExhaustiveTQ-b4-rf5 (query)                              736.30     1_824.80     2_561.10       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4-rf10 (query)                             736.30     1_952.55     2_688.85       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4-rf20 (query)                             736.30     2_338.37     3_074.67       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4 (self)                                   736.30     7_676.62     8_412.92       1.0000          1.0000            1.0000        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                          864.70       322.55     1_187.25       0.8735          1.0272            1.0201        22.45
IVF-TQ-b2-nl158-np12-rf0 (query)                         864.70       416.37     1_281.07       0.8736          1.0271            1.0200        22.45
IVF-TQ-b2-nl158-np17-rf0 (query)                         864.70       490.15     1_354.84       0.8736          1.0271            1.0200        22.45
IVF-TQ-b2-nl158-np7-rf10 (query)                         864.70       560.15     1_424.84       0.9995          1.0001            1.0000        22.45
IVF-TQ-b2-nl158-np7-rf20 (query)                         864.70       866.97     1_731.67       0.9995          1.0001            1.0000        22.45
IVF-TQ-b2-nl158-np12-rf10 (query)                        864.70       671.66     1_536.35       1.0000          1.0000            1.0000        22.45
IVF-TQ-b2-nl158-np12-rf20 (query)                        864.70     1_011.24     1_875.94       1.0000          1.0000            1.0000        22.45
IVF-TQ-b2-nl158-np17-rf10 (query)                        864.70       761.62     1_626.31       1.0000          1.0000            1.0000        22.45
IVF-TQ-b2-nl158-np17-rf20 (query)                        864.70     1_127.99     1_992.69       1.0000          1.0000            1.0000        22.45
IVF-TQ-b2-nl158 (self)                                   864.70     1_951.65     2_816.35       1.0000          1.0000            1.0000        22.45
IVF-TQ-b2-nl223-np11-rf0 (query)                         881.06       342.78     1_223.84       0.8736          1.0271            1.0200        22.77
IVF-TQ-b2-nl223-np14-rf0 (query)                         881.06       382.45     1_263.52       0.8736          1.0271            1.0200        22.77
IVF-TQ-b2-nl223-np21-rf0 (query)                         881.06       474.10     1_355.17       0.8736          1.0271            1.0200        22.77
IVF-TQ-b2-nl223-np11-rf10 (query)                        881.06       575.96     1_457.02       0.9999          1.0000            1.0000        22.77
IVF-TQ-b2-nl223-np11-rf20 (query)                        881.06       887.90     1_768.96       0.9999          1.0000            1.0000        22.77
IVF-TQ-b2-nl223-np14-rf10 (query)                        881.06       630.65     1_511.71       1.0000          1.0000            1.0000        22.77
IVF-TQ-b2-nl223-np14-rf20 (query)                        881.06       954.59     1_835.66       1.0000          1.0000            1.0000        22.77
IVF-TQ-b2-nl223-np21-rf10 (query)                        881.06       715.32     1_596.39       1.0000          1.0000            1.0000        22.77
IVF-TQ-b2-nl223-np21-rf20 (query)                        881.06     1_068.81     1_949.87       1.0000          1.0000            1.0000        22.77
IVF-TQ-b2-nl223 (self)                                   881.06     1_897.31     2_778.37       1.0000          1.0000            1.0000        22.77
IVF-TQ-b2-nl316-np15-rf0 (query)                         921.37       353.16     1_274.53       0.8736          1.0271            1.0200        23.34
IVF-TQ-b2-nl316-np17-rf0 (query)                         921.37       373.85     1_295.22       0.8736          1.0271            1.0200        23.34
IVF-TQ-b2-nl316-np25-rf0 (query)                         921.37       444.68     1_366.04       0.8736          1.0271            1.0200        23.34
IVF-TQ-b2-nl316-np15-rf10 (query)                        921.37       583.81     1_505.18       0.9999          1.0000            1.0000        23.34
IVF-TQ-b2-nl316-np15-rf20 (query)                        921.37       899.22     1_820.59       0.9999          1.0000            1.0000        23.34
IVF-TQ-b2-nl316-np17-rf10 (query)                        921.37       607.82     1_529.19       1.0000          1.0000            1.0000        23.34
IVF-TQ-b2-nl316-np17-rf20 (query)                        921.37       933.70     1_855.07       1.0000          1.0000            1.0000        23.34
IVF-TQ-b2-nl316-np25-rf10 (query)                        921.37       688.24     1_609.60       1.0000          1.0000            1.0000        23.34
IVF-TQ-b2-nl316-np25-rf20 (query)                        921.37     1_039.13     1_960.49       1.0000          1.0000            1.0000        23.34
IVF-TQ-b2-nl316 (self)                                   921.37     1_914.01     2_835.38       1.0000          1.0000            1.0000        23.34
IVF-TQ-b4-nl158-np7-rf0 (query)                          981.79       475.19     1_456.98       0.9095          1.0147            1.0084        41.23
IVF-TQ-b4-nl158-np12-rf0 (query)                         981.79       644.37     1_626.16       0.9097          1.0146            1.0083        41.23
IVF-TQ-b4-nl158-np17-rf0 (query)                         981.79       790.82     1_772.60       0.9097          1.0146            1.0083        41.23
IVF-TQ-b4-nl158-np7-rf10 (query)                         981.79       716.84     1_698.62       0.9995          1.0001            1.0000        41.23
IVF-TQ-b4-nl158-np7-rf20 (query)                         981.79     1_026.28     2_008.06       0.9995          1.0001            1.0000        41.23
IVF-TQ-b4-nl158-np12-rf10 (query)                        981.79       907.74     1_889.53       1.0000          1.0000            1.0000        41.23
IVF-TQ-b4-nl158-np12-rf20 (query)                        981.79     1_248.52     2_230.31       1.0000          1.0000            1.0000        41.23
IVF-TQ-b4-nl158-np17-rf10 (query)                        981.79     1_060.34     2_042.13       1.0000          1.0000            1.0000        41.23
IVF-TQ-b4-nl158-np17-rf20 (query)                        981.79     1_424.59     2_406.38       1.0000          1.0000            1.0000        41.23
IVF-TQ-b4-nl158 (self)                                   981.79     2_459.57     3_441.36       1.0000          1.0000            1.0000        41.23
IVF-TQ-b4-nl223-np11-rf0 (query)                         996.61       514.93     1_511.54       0.9096          1.0146            1.0084        41.68
IVF-TQ-b4-nl223-np14-rf0 (query)                         996.61       592.16     1_588.77       0.9097          1.0146            1.0083        41.68
IVF-TQ-b4-nl223-np21-rf0 (query)                         996.61       750.32     1_746.93       0.9097          1.0146            1.0083        41.68
IVF-TQ-b4-nl223-np11-rf10 (query)                        996.61       796.05     1_792.66       0.9999          1.0000            1.0000        41.68
IVF-TQ-b4-nl223-np11-rf20 (query)                        996.61     1_055.18     2_051.78       0.9999          1.0000            1.0000        41.68
IVF-TQ-b4-nl223-np14-rf10 (query)                        996.61       829.92     1_826.53       1.0000          1.0000            1.0000        41.68
IVF-TQ-b4-nl223-np14-rf20 (query)                        996.61     1_165.19     2_161.80       1.0000          1.0000            1.0000        41.68
IVF-TQ-b4-nl223-np21-rf10 (query)                        996.61       988.32     1_984.92       1.0000          1.0000            1.0000        41.68
IVF-TQ-b4-nl223-np21-rf20 (query)                        996.61     1_340.88     2_337.49       1.0000          1.0000            1.0000        41.68
IVF-TQ-b4-nl223 (self)                                   996.61     2_388.78     3_385.39       1.0000          1.0000            1.0000        41.68
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_098.98       528.40     1_627.38       0.9097          1.0146            1.0084        42.54
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_098.98       593.62     1_692.61       0.9097          1.0146            1.0083        42.54
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_098.98       709.64     1_808.62       0.9097          1.0146            1.0083        42.54
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_098.98       752.48     1_851.46       0.9999          1.0000            1.0000        42.54
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_098.98     1_063.72     2_162.70       0.9999          1.0000            1.0000        42.54
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_098.98       790.16     1_889.14       1.0000          1.0000            1.0000        42.54
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_098.98     1_121.12     2_220.11       1.0000          1.0000            1.0000        42.54
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_098.98       948.09     2_047.08       1.0000          1.0000            1.0000        42.54
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_098.98     1_290.37     2_389.35       1.0000          1.0000            1.0000        42.54
IVF-TQ-b4-nl316 (self)                                 1_098.98     2_356.16     3_455.15       1.0000          1.0000            1.0000        42.54
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*Each table starts with the ann-search-rs version and commit it was run on.*
*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
