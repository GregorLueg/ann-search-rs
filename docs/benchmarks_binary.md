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
Exhaustive (query)                                        32.56       726.72       759.28       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.56     2_414.78     2_447.34       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                107.35       235.58       342.92       0.1199          1.4617            1.4199         1.78
ExhaustiveBinary-256-random-rf10 (query)                 107.35       345.02       452.37       0.3411          1.0941            1.0814         1.78
ExhaustiveBinary-256-random-rf20 (query)                 107.35       431.67       539.02       0.4467          1.0571            1.0475         1.78
ExhaustiveBinary-256-random (self)                       107.35     1_077.11     1_184.45       0.3454          1.0895            1.0798         1.78
ExhaustiveBinary-256-pca_no_rr (query)                   138.71       250.40       389.12       0.1153          1.4748            1.4212         1.78
ExhaustiveBinary-256-pca-rf10 (query)                    138.71       334.29       473.01       0.3323          1.1029            1.0834         1.78
ExhaustiveBinary-256-pca-rf20 (query)                    138.71       432.20       570.91       0.4387          1.0631            1.0485         1.78
ExhaustiveBinary-256-pca (self)                          138.71     1_065.53     1_204.24       0.3391          1.0957            1.0813         1.78
ExhaustiveBinary-512-random_no_rr (query)                116.61       344.63       461.24       0.1588          1.3547            1.3300         3.55
ExhaustiveBinary-512-random-rf10 (query)                 116.61       456.11       572.72       0.3786          1.0692            1.0677         3.55
ExhaustiveBinary-512-random-rf20 (query)                 116.61       577.66       694.27       0.4874          1.0424            1.0395         3.55
ExhaustiveBinary-512-random (self)                       116.61     1_487.72     1_604.33       0.3805          1.0675            1.0675         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   153.53       345.16       498.69       0.1564          1.3535            1.3265         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    153.53       452.49       606.02       0.3789          1.0710            1.0663         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    153.53       581.97       735.50       0.4903          1.0433            1.0387         3.55
ExhaustiveBinary-512-pca (self)                          153.53     1_479.98     1_633.50       0.3823          1.0678            1.0665         3.55
ExhaustiveBinary-1024-random_no_rr (query)               151.57       511.21       662.78       0.1929          1.2764            1.2696         7.10
ExhaustiveBinary-1024-random-rf10 (query)                151.57       627.49       779.05       0.4214          1.0550            1.0552         7.10
ExhaustiveBinary-1024-random-rf20 (query)                151.57       734.95       886.51       0.5434          1.0327            1.0308         7.10
ExhaustiveBinary-1024-random (self)                      151.57     2_079.58     2_231.14       0.4232          1.0547            1.0552         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  188.42       512.05       700.47       0.1921          1.2733            1.2652         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   188.42       621.12       809.55       0.4226          1.0546            1.0544         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   188.42       726.60       915.03       0.5443          1.0326            1.0305         7.10
ExhaustiveBinary-1024-pca (self)                         188.42     2_063.13     2_251.55       0.4236          1.0546            1.0548         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   77.51       429.99       507.49       0.1211          1.4987            1.4523         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    77.51       459.70       537.21       0.3284          1.1039            1.0884         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    77.51       699.69       777.19       0.4385          1.0624            1.0494         1.53
ExhaustiveBinary-256-sign (self)                          77.51     1_514.06     1_591.56       0.3334          1.0988            1.0859         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)              865.34        49.13       914.46       0.1231          1.4432            1.4051         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)             865.34        49.09       914.43       0.1231          1.4432            1.4051         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)             865.34        51.98       917.32       0.1231          1.4432            1.4051         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)             865.34        96.39       961.73       0.3463          1.0912            1.0794         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)             865.34       150.31     1_015.65       0.4529          1.0552            1.0461         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)            865.34        95.01       960.35       0.3463          1.0912            1.0794         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)            865.34       145.82     1_011.16       0.4529          1.0552            1.0461         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)            865.34        94.85       960.18       0.3463          1.0912            1.0794         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)            865.34       154.18     1_019.52       0.4529          1.0552            1.0461         1.93
IVF-Binary-256-nl158-random (self)                       865.34       207.34     1_072.68       0.3507          1.0862            1.0777         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             584.49        44.39       628.87       0.1413          1.3601            1.3150         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             584.49        45.98       630.47       0.1412          1.3605            1.3154         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             584.49        49.55       634.03       0.1412          1.3605            1.3154         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            584.49        95.23       679.72       0.3893          1.0689            1.0625         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            584.49       145.84       730.32       0.4976          1.0427            1.0371         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            584.49       101.85       686.34       0.3891          1.0690            1.0625         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            584.49       146.59       731.08       0.4973          1.0428            1.0372         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            584.49        96.60       681.08       0.3891          1.0690            1.0625         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            584.49       149.63       734.11       0.4973          1.0428            1.0372         2.00
IVF-Binary-256-nl223-random (self)                       584.49       213.56       798.05       0.3943          1.0643            1.0615         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             776.95        50.31       827.27       0.1496          1.3359            1.2904         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             776.95        48.16       825.11       0.1495          1.3365            1.2906         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             776.95        52.75       829.70       0.1495          1.3366            1.2906         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            776.95        97.23       874.18       0.4018          1.0649            1.0585         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            776.95       147.04       924.00       0.5055          1.0413            1.0358         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            776.95        97.10       874.05       0.4016          1.0650            1.0587         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            776.95       152.03       928.98       0.5051          1.0414            1.0359         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            776.95       100.99       877.94       0.4016          1.0650            1.0587         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            776.95       154.70       931.65       0.5051          1.0414            1.0359         2.09
IVF-Binary-256-nl316-random (self)                       776.95       222.12       999.07       0.4063          1.0606            1.0577         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)                 842.44        40.73       883.17       0.1190          1.4540            1.4124         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)                842.44        41.74       884.18       0.1190          1.4540            1.4124         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)                842.44        45.41       887.85       0.1190          1.4540            1.4124         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)                842.44        90.53       932.97       0.3368          1.0992            1.0816         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)                842.44       140.70       983.13       0.4434          1.0613            1.0475         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)               842.44        90.14       932.57       0.3368          1.0992            1.0816         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)               842.44       143.41       985.85       0.4434          1.0613            1.0475         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)               842.44        93.70       936.14       0.3368          1.0992            1.0816         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)               842.44       144.40       986.83       0.4434          1.0613            1.0475         1.93
IVF-Binary-256-nl158-pca (self)                          842.44       201.32     1_043.76       0.3434          1.0923            1.0796         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                604.00        44.83       648.82       0.1377          1.3704            1.3177         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                604.00        46.03       650.03       0.1377          1.3708            1.3180         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                604.00        48.96       652.96       0.1377          1.3708            1.3180         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               604.00        96.91       700.91       0.3828          1.0754            1.0637         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               604.00       145.77       749.76       0.4958          1.0458            1.0375         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               604.00        93.34       697.34       0.3827          1.0755            1.0638         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               604.00       144.27       748.27       0.4957          1.0458            1.0375         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               604.00        97.14       701.14       0.3827          1.0755            1.0638         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               604.00       154.83       758.83       0.4957          1.0458            1.0375         2.00
IVF-Binary-256-nl223-pca (self)                          604.00       212.61       816.61       0.3896          1.0692            1.0620         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                819.01        46.79       865.80       0.1471          1.3419            1.2914         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                819.01        46.70       865.71       0.1471          1.3424            1.2916         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                819.01        50.12       869.13       0.1471          1.3425            1.2916         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               819.01        96.41       915.42       0.3970          1.0703            1.0594         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               819.01       159.36       978.37       0.5069          1.0437            1.0356         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               819.01        96.49       915.50       0.3968          1.0704            1.0594         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               819.01       147.73       966.74       0.5067          1.0438            1.0356         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               819.01       100.84       919.85       0.3968          1.0704            1.0594         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               819.01       159.44       978.45       0.5067          1.0438            1.0356         2.09
IVF-Binary-256-nl316-pca (self)                          819.01       218.65     1_037.66       0.4025          1.0646            1.0582         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)              840.71        58.50       899.21       0.1607          1.3465            1.3235         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)             840.71        62.39       903.11       0.1607          1.3465            1.3235         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)             840.71        63.46       904.17       0.1607          1.3465            1.3235         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)             840.71       113.59       954.31       0.3812          1.0681            1.0667         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)             840.71       167.03     1_007.75       0.4908          1.0417            1.0390         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)            840.71       114.70       955.41       0.3812          1.0681            1.0667         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)            840.71       170.65     1_011.36       0.4908          1.0417            1.0390         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)            840.71       123.33       964.04       0.3812          1.0681            1.0667         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)            840.71       174.74     1_015.45       0.4908          1.0417            1.0390         3.71
IVF-Binary-512-nl158-random (self)                       840.71       285.62     1_126.33       0.3832          1.0664            1.0666         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             596.40        62.56       658.96       0.1711          1.2998            1.2780         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             596.40        65.14       661.54       0.1711          1.3001            1.2782         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             596.40        68.42       664.82       0.1711          1.3001            1.2782         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            596.40       118.06       714.46       0.4019          1.0605            1.0593         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            596.40       174.79       771.19       0.5140          1.0373            1.0348         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            596.40       118.23       714.63       0.4017          1.0605            1.0593         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            596.40       172.27       768.67       0.5137          1.0374            1.0349         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            596.40       120.74       717.14       0.4017          1.0605            1.0593         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            596.40       176.07       772.47       0.5137          1.0374            1.0349         3.77
IVF-Binary-512-nl223-random (self)                       596.40       296.02       892.42       0.4039          1.0592            1.0592         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             797.67        65.72       863.39       0.1755          1.2880            1.2669         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             797.67        67.30       864.97       0.1754          1.2885            1.2672         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             797.67        72.01       869.68       0.1754          1.2885            1.2672         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            797.67       119.19       916.86       0.4058          1.0594            1.0581         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            797.67       170.78       968.46       0.5175          1.0368            1.0342         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            797.67       119.78       917.46       0.4056          1.0595            1.0582         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            797.67       176.29       973.96       0.5171          1.0369            1.0342         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            797.67       125.75       923.42       0.4056          1.0595            1.0582         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            797.67       176.68       974.35       0.5171          1.0369            1.0342         3.86
IVF-Binary-512-nl316-random (self)                       797.67       298.11     1_095.78       0.4086          1.0581            1.0580         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)                 884.80        59.66       944.46       0.1581          1.3474            1.3214         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)                884.80        64.35       949.15       0.1581          1.3474            1.3214         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)                884.80        67.05       951.85       0.1581          1.3474            1.3214         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)                884.80       117.32     1_002.12       0.3810          1.0702            1.0658         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)                884.80       167.20     1_052.00       0.4928          1.0428            1.0383         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)               884.80       113.90       998.70       0.3810          1.0702            1.0658         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)               884.80       170.41     1_055.21       0.4928          1.0428            1.0383         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)               884.80       117.14     1_001.94       0.3810          1.0702            1.0658         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)               884.80       174.75     1_059.55       0.4928          1.0428            1.0383         3.71
IVF-Binary-512-nl158-pca (self)                          884.80       287.91     1_172.71       0.3843          1.0672            1.0657         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                661.81        62.40       724.21       0.1694          1.3033            1.2746         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                661.81        65.57       727.38       0.1694          1.3034            1.2747         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                661.81        68.11       729.92       0.1694          1.3034            1.2747         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               661.81       125.19       787.00       0.4036          1.0619            1.0583         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               661.81       166.92       828.72       0.5164          1.0383            1.0342         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               661.81       119.19       780.99       0.4035          1.0619            1.0583         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               661.81       171.47       833.28       0.5162          1.0384            1.0342         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               661.81       120.57       782.38       0.4035          1.0619            1.0583         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               661.81       175.11       836.92       0.5162          1.0384            1.0342         3.77
IVF-Binary-512-nl223-pca (self)                          661.81       287.89       949.70       0.4059          1.0599            1.0584         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                863.18        67.35       930.53       0.1733          1.2925            1.2652         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                863.18        65.74       928.92       0.1732          1.2927            1.2653         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                863.18        74.17       937.35       0.1732          1.2927            1.2653         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               863.18       118.16       981.34       0.4089          1.0604            1.0565         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               863.18       180.51     1_043.69       0.5220          1.0374            1.0334         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               863.18       116.96       980.14       0.4088          1.0605            1.0565         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               863.18       169.88     1_033.06       0.5217          1.0374            1.0334         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               863.18       121.34       984.52       0.4088          1.0605            1.0565         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               863.18       176.50     1_039.68       0.5217          1.0374            1.0334         3.86
IVF-Binary-512-nl316-pca (self)                          863.18       298.85     1_162.03       0.4117          1.0583            1.0566         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)             892.73        90.73       983.46       0.1937          1.2738            1.2669         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)            892.73        94.54       987.27       0.1937          1.2738            1.2669         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)            892.73        97.14       989.87       0.1937          1.2738            1.2669         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)            892.73       147.65     1_040.38       0.4227          1.0546            1.0549         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)            892.73       209.41     1_102.13       0.5450          1.0325            1.0305         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)           892.73       153.69     1_046.41       0.4227          1.0546            1.0549         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)           892.73       217.24     1_109.97       0.5450          1.0325            1.0305         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)           892.73       160.69     1_053.42       0.4227          1.0546            1.0549         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)           892.73       212.89     1_105.62       0.5450          1.0325            1.0305         7.26
IVF-Binary-1024-nl158-random (self)                      892.73       416.83     1_309.55       0.4246          1.0544            1.0549         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            629.54        96.12       725.67       0.1973          1.2556            1.2488         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            629.54        99.38       728.92       0.1973          1.2558            1.2489         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            629.54       105.10       734.64       0.1973          1.2558            1.2489         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           629.54       151.22       780.76       0.4342          1.0516            1.0516         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           629.54       204.95       834.49       0.5563          1.0309            1.0289         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           629.54       153.64       783.18       0.4341          1.0517            1.0516         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           629.54       210.04       839.58       0.5562          1.0309            1.0289         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           629.54       162.89       792.44       0.4341          1.0517            1.0516         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           629.54       218.43       847.97       0.5562          1.0309            1.0289         7.32
IVF-Binary-1024-nl223-random (self)                      629.54       413.25     1_042.79       0.4353          1.0515            1.0518         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            845.82       100.26       946.07       0.1989          1.2508            1.2444         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)            845.82        98.38       944.19       0.1988          1.2511            1.2446         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)            845.82       104.24       950.06       0.1988          1.2511            1.2446         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)           845.82       159.09     1_004.90       0.4365          1.0510            1.0509         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)           845.82       207.41     1_053.22       0.5576          1.0306            1.0287         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)           845.82       153.60       999.42       0.4364          1.0511            1.0509         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)           845.82       214.61     1_060.43       0.5573          1.0307            1.0288         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)           845.82       160.96     1_006.78       0.4364          1.0511            1.0509         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)           845.82       217.90     1_063.72       0.5573          1.0307            1.0288         7.42
IVF-Binary-1024-nl316-random (self)                      845.82       421.66     1_267.47       0.4378          1.0509            1.0513         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)                937.30        90.03     1_027.33       0.1929          1.2710            1.2632         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)               937.30        94.44     1_031.74       0.1929          1.2710            1.2632         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)               937.30        97.55     1_034.85       0.1929          1.2710            1.2632         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)               937.30       147.52     1_084.82       0.4240          1.0542            1.0540         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)               937.30       206.87     1_144.17       0.5461          1.0324            1.0302         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)              937.30       151.26     1_088.56       0.4240          1.0542            1.0540         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)              937.30       212.30     1_149.60       0.5461          1.0324            1.0302         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)              937.30       156.01     1_093.31       0.4240          1.0542            1.0540         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)              937.30       214.26     1_151.56       0.5461          1.0324            1.0302         7.26
IVF-Binary-1024-nl158-pca (self)                         937.30       410.02     1_347.32       0.4249          1.0542            1.0544         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               689.99        94.48       784.47       0.1974          1.2518            1.2441         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               689.99        96.95       786.94       0.1974          1.2519            1.2442         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               689.99       103.56       793.55       0.1974          1.2519            1.2442         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              689.99       151.96       841.95       0.4355          1.0511            1.0508         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              689.99       210.54       900.53       0.5584          1.0305            1.0281         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              689.99       151.40       841.40       0.4354          1.0511            1.0508         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              689.99       217.72       907.71       0.5581          1.0306            1.0282         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              689.99       160.24       850.23       0.4354          1.0511            1.0508         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              689.99       215.11       905.10       0.5581          1.0306            1.0282         7.32
IVF-Binary-1024-nl223-pca (self)                         689.99       417.40     1_107.39       0.4363          1.0511            1.0512         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               883.57        98.58       982.15       0.1987          1.2475            1.2401         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)               883.57       100.25       983.82       0.1987          1.2476            1.2401         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)               883.57       105.62       989.19       0.1987          1.2476            1.2401         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)              883.57       154.34     1_037.91       0.4386          1.0503            1.0501         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)              883.57       217.28     1_100.86       0.5611          1.0302            1.0279         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)              883.57       154.15     1_037.72       0.4384          1.0504            1.0501         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)              883.57       210.24     1_093.81       0.5610          1.0302            1.0279         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)              883.57       164.13     1_047.70       0.4384          1.0504            1.0501         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)              883.57       217.75     1_101.32       0.5610          1.0302            1.0279         7.42
IVF-Binary-1024-nl316-pca (self)                         883.57       417.79     1_301.36       0.4396          1.0504            1.0505         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)                820.46       155.19       975.64       0.1216          1.4959            1.4465         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               820.46       150.00       970.46       0.1216          1.4959            1.4465         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               820.46       152.08       972.53       0.1216          1.4959            1.4465         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               820.46       192.66     1_013.11       0.3305          1.1023            1.0867         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               820.46       346.75     1_167.21       0.4405          1.0617            1.0486         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              820.46       188.31     1_008.77       0.3305          1.1023            1.0867         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              820.46       344.80     1_165.26       0.4405          1.0617            1.0486         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              820.46       189.89     1_010.34       0.3305          1.1023            1.0867         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              820.46       344.34     1_164.79       0.4405          1.0617            1.0486         1.68
IVF-Binary-256-nl158-sign (self)                         820.46       494.56     1_315.02       0.3358          1.0972            1.0848         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               548.13       147.81       695.94       0.1228          1.4832            1.4276         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               548.13       154.78       702.91       0.1228          1.4843            1.4282         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               548.13       155.25       703.38       0.1228          1.4844            1.4282         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              548.13       185.90       734.03       0.3559          1.0888            1.0752         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              548.13       329.88       878.01       0.4583          1.0561            1.0442         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              548.13       198.28       746.41       0.3557          1.0890            1.0752         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              548.13       340.87       889.00       0.4580          1.0562            1.0443         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              548.13       190.49       738.62       0.3557          1.0890            1.0752         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              548.13       335.05       883.18       0.4580          1.0562            1.0443         1.75
IVF-Binary-256-nl223-sign (self)                         548.13       500.19     1_048.32       0.3611          1.0841            1.0736         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               749.63       150.36       899.99       0.1239          1.4676            1.4148         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               749.63       150.26       899.89       0.1237          1.4698            1.4160         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               749.63       158.19       907.81       0.1237          1.4700            1.4161         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              749.63       192.73       942.36       0.3598          1.0872            1.0741         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              749.63       334.10     1_083.73       0.4582          1.0559            1.0445         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              749.63       189.23       938.86       0.3596          1.0874            1.0742         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              749.63       332.97     1_082.59       0.4577          1.0561            1.0446         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              749.63       194.25       943.88       0.3596          1.0874            1.0742         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              749.63       338.62     1_088.25       0.4577          1.0561            1.0446         1.84
IVF-Binary-256-nl316-sign (self)                         749.63       513.86     1_263.49       0.3647          1.0824            1.0724         1.84
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
Exhaustive (query)                                        68.78     1_318.59     1_387.37       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.78     4_353.13     4_421.91       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                135.45       256.56       392.01       0.1109          1.3512            1.3092         2.03
ExhaustiveBinary-256-random-rf10 (query)                 135.45       385.73       521.18       0.3143          1.0825            1.0613         2.03
ExhaustiveBinary-256-random-rf20 (query)                 135.45       497.68       633.14       0.4100          1.0523            1.0368         2.03
ExhaustiveBinary-256-random (self)                       135.45     1_175.10     1_310.55       0.3161          1.0784            1.0600         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   230.64       266.14       496.78       0.1167          1.3480            1.2981         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    230.64       387.70       618.35       0.3159          1.0791            1.0596         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    230.64       503.14       733.78       0.4121          1.0505            1.0362         2.03
ExhaustiveBinary-256-pca (self)                          230.64     1_173.76     1_404.41       0.3171          1.0782            1.0588         2.03
ExhaustiveBinary-512-random_no_rr (query)                205.24       375.92       581.17       0.1528          1.2601            1.2299         4.05
ExhaustiveBinary-512-random-rf10 (query)                 205.24       507.65       712.90       0.3465          1.0565            1.0514         4.05
ExhaustiveBinary-512-random-rf20 (query)                 205.24       633.37       838.61       0.4452          1.0358            1.0312         4.05
ExhaustiveBinary-512-random (self)                       205.24     1_612.33     1_817.58       0.3476          1.0547            1.0512         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   299.44       373.80       673.24       0.1558          1.2535            1.2254         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    299.44       510.54       809.98       0.3512          1.0523            1.0507         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    299.44       639.39       938.84       0.4484          1.0329            1.0309         4.05
ExhaustiveBinary-512-pca (self)                          299.44     1_632.93     1_932.38       0.3515          1.0522            1.0505         4.05
ExhaustiveBinary-1024-random_no_rr (query)               258.13       596.21       854.34       0.1816          1.2043            1.1936         8.11
ExhaustiveBinary-1024-random-rf10 (query)                258.13       712.33       970.46       0.3747          1.0447            1.0452         8.11
ExhaustiveBinary-1024-random-rf20 (query)                258.13       855.75     1_113.89       0.4789          1.0282            1.0270         8.11
ExhaustiveBinary-1024-random (self)                      258.13     2_314.60     2_572.73       0.3754          1.0447            1.0451         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  354.77       560.55       915.31       0.1832          1.2013            1.1905         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   354.77       705.69     1_060.46       0.3798          1.0434            1.0443         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   354.77       882.56     1_237.33       0.4867          1.0272            1.0261         8.11
ExhaustiveBinary-1024-pca (self)                         354.77     2_315.39     2_670.15       0.3787          1.0436            1.0444         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   86.24       663.76       750.00       0.1518          1.2701            1.2528         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    86.24       724.42       810.66       0.3399          1.0607            1.0535         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    86.24     1_109.21     1_195.46       0.4406          1.0369            1.0319         3.05
ExhaustiveBinary-512-sign (self)                          86.24     2_309.53     2_395.77       0.3409          1.0595            1.0531         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)            1_702.98        76.05     1_779.03       0.1157          1.3331            1.2963         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           1_702.98        77.97     1_780.95       0.1157          1.3331            1.2963         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           1_702.98        78.91     1_781.89       0.1157          1.3331            1.2963         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           1_702.98       155.92     1_858.90       0.3225          1.0778            1.0583         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           1_702.98       238.19     1_941.17       0.4188          1.0495            1.0350         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          1_702.98       152.41     1_855.38       0.3225          1.0778            1.0583         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          1_702.98       238.00     1_940.98       0.4188          1.0495            1.0350         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          1_702.98       147.96     1_850.94       0.3225          1.0778            1.0583         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          1_702.98       242.19     1_945.17       0.4188          1.0495            1.0350         2.34
IVF-Binary-256-nl158-random (self)                     1_702.98       288.35     1_991.32       0.3241          1.0737            1.0571         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)             942.86        72.30     1_015.17       0.1325          1.2776            1.2358         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)             942.86        73.13     1_015.99       0.1325          1.2779            1.2360         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)             942.86        75.31     1_018.18       0.1325          1.2780            1.2360         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)            942.86       159.29     1_102.15       0.3688          1.0554            1.0455         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)            942.86       247.43     1_190.29       0.4683          1.0358            1.0278         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)            942.86       157.81     1_100.67       0.3685          1.0555            1.0456         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)            942.86       259.76     1_202.62       0.4678          1.0359            1.0279         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)            942.86       160.10     1_102.97       0.3685          1.0555            1.0456         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)            942.86       249.05     1_191.91       0.4678          1.0359            1.0279         2.47
IVF-Binary-256-nl223-random (self)                       942.86       309.08     1_251.94       0.3705          1.0515            1.0449         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)           1_282.60        77.55     1_360.14       0.1435          1.2534            1.2115         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           1_282.60        78.60     1_361.19       0.1434          1.2536            1.2119         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           1_282.60        81.97     1_364.57       0.1434          1.2542            1.2121         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          1_282.60       175.73     1_458.33       0.3836          1.0508            1.0420         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          1_282.60       252.76     1_535.36       0.4826          1.0337            1.0261         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          1_282.60       159.96     1_442.56       0.3833          1.0510            1.0420         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          1_282.60       258.27     1_540.86       0.4820          1.0338            1.0262         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          1_282.60       168.11     1_450.71       0.3832          1.0510            1.0420         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          1_282.60       259.59     1_542.19       0.4818          1.0339            1.0262         2.65
IVF-Binary-256-nl316-random (self)                     1_282.60       328.96     1_611.56       0.3852          1.0471            1.0414         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               1_776.56        64.49     1_841.05       0.1212          1.3300            1.2889         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              1_776.56        68.51     1_845.08       0.1212          1.3300            1.2889         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              1_776.56        68.24     1_844.80       0.1212          1.3300            1.2889         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              1_776.56       152.30     1_928.86       0.3222          1.0753            1.0574         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              1_776.56       237.51     2_014.07       0.4196          1.0483            1.0348         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             1_776.56       147.58     1_924.14       0.3222          1.0753            1.0574         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             1_776.56       241.18     2_017.74       0.4196          1.0483            1.0348         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             1_776.56       150.81     1_927.38       0.3222          1.0753            1.0574         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             1_776.56       242.06     2_018.62       0.4196          1.0483            1.0348         2.34
IVF-Binary-256-nl158-pca (self)                        1_776.56       297.35     2_073.91       0.3232          1.0740            1.0566         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              1_060.64        70.91     1_131.56       0.1366          1.2800            1.2327         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              1_060.64        72.18     1_132.82       0.1366          1.2802            1.2328         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              1_060.64        79.77     1_140.42       0.1366          1.2802            1.2328         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             1_060.64       159.47     1_220.12       0.3637          1.0553            1.0460         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             1_060.64       249.46     1_310.10       0.4640          1.0357            1.0283         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             1_060.64       157.38     1_218.03       0.3636          1.0553            1.0460         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             1_060.64       248.18     1_308.82       0.4638          1.0357            1.0283         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             1_060.64       158.41     1_219.05       0.3636          1.0553            1.0460         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             1_060.64       252.82     1_313.46       0.4638          1.0357            1.0283         2.47
IVF-Binary-256-nl223-pca (self)                        1_060.64       309.98     1_370.62       0.3658          1.0535            1.0453         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              1_393.27        78.76     1_472.03       0.1456          1.2588            1.2105         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              1_393.27        77.98     1_471.25       0.1456          1.2588            1.2106         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              1_393.27        81.10     1_474.37       0.1455          1.2592            1.2107         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             1_393.27       165.20     1_558.47       0.3774          1.0509            1.0426         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             1_393.27       256.88     1_650.15       0.4766          1.0332            1.0267         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             1_393.27       161.90     1_555.17       0.3773          1.0509            1.0427         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             1_393.27       253.96     1_647.23       0.4760          1.0333            1.0268         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             1_393.27       165.87     1_559.14       0.3772          1.0509            1.0427         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             1_393.27       257.63     1_650.90       0.4759          1.0333            1.0268         2.65
IVF-Binary-256-nl316-pca (self)                        1_393.27       344.14     1_737.41       0.3792          1.0492            1.0422         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)            1_737.41        92.54     1_829.95       0.1554          1.2515            1.2240         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)           1_737.41        96.13     1_833.54       0.1554          1.2515            1.2240         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)           1_737.41        95.84     1_833.25       0.1554          1.2515            1.2240         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)           1_737.41       182.33     1_919.74       0.3517          1.0543            1.0500         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)           1_737.41       273.93     2_011.34       0.4505          1.0347            1.0305         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)          1_737.41       188.84     1_926.25       0.3517          1.0543            1.0500         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)          1_737.41       277.27     2_014.68       0.4505          1.0347            1.0305         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)          1_737.41       183.82     1_921.23       0.3517          1.0543            1.0500         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)          1_737.41       276.31     2_013.72       0.4505          1.0347            1.0305         4.36
IVF-Binary-512-nl158-random (self)                     1_737.41       411.30     2_148.71       0.3527          1.0528            1.0499         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)           1_019.35        98.08     1_117.43       0.1644          1.2242            1.1976         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)           1_019.35       104.54     1_123.88       0.1643          1.2245            1.1979         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)           1_019.35       108.90     1_128.24       0.1643          1.2246            1.1979         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)          1_019.35       188.99     1_208.34       0.3693          1.0479            1.0450         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)          1_019.35       278.56     1_297.90       0.4691          1.0312            1.0277         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)          1_019.35       185.63     1_204.98       0.3691          1.0480            1.0450         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)          1_019.35       281.29     1_300.64       0.4685          1.0312            1.0278         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)          1_019.35       196.61     1_215.96       0.3691          1.0480            1.0450         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)          1_019.35       282.86     1_302.20       0.4685          1.0312            1.0278         4.49
IVF-Binary-512-nl223-random (self)                     1_019.35       440.04     1_459.39       0.3701          1.0469            1.0450         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)           1_422.18       112.20     1_534.38       0.1680          1.2145            1.1880         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)           1_422.18       108.02     1_530.20       0.1680          1.2148            1.1883         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)           1_422.18       112.21     1_534.39       0.1679          1.2151            1.1885         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)          1_422.18       196.50     1_618.68       0.3756          1.0469            1.0438         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)          1_422.18       283.95     1_706.13       0.4755          1.0304            1.0270         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)          1_422.18       190.77     1_612.95       0.3754          1.0470            1.0438         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)          1_422.18       284.28     1_706.46       0.4747          1.0305            1.0272         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)          1_422.18       195.48     1_617.66       0.3753          1.0470            1.0439         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)          1_422.18       289.55     1_711.73       0.4744          1.0306            1.0272         4.67
IVF-Binary-512-nl316-random (self)                     1_422.18       451.15     1_873.33       0.3761          1.0456            1.0437         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)               1_825.20        96.66     1_921.86       0.1578          1.2477            1.2198         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)              1_825.20        94.83     1_920.02       0.1578          1.2477            1.2198         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)              1_825.20        96.19     1_921.38       0.1578          1.2477            1.2198         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)              1_825.20       189.94     2_015.13       0.3546          1.0514            1.0497         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)              1_825.20       269.53     2_094.72       0.4526          1.0324            1.0303         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)             1_825.20       180.05     2_005.25       0.3546          1.0514            1.0497         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)             1_825.20       276.38     2_101.58       0.4526          1.0324            1.0303         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)             1_825.20       182.41     2_007.61       0.3546          1.0514            1.0497         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)             1_825.20       278.81     2_104.00       0.4526          1.0324            1.0303         4.36
IVF-Binary-512-nl158-pca (self)                        1_825.20       443.92     2_269.12       0.3548          1.0513            1.0495         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_135.83        98.29     1_234.11       0.1671          1.2190            1.1937         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_135.83       101.55     1_237.38       0.1671          1.2192            1.1940         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_135.83       103.46     1_239.29       0.1671          1.2192            1.1940         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_135.83       192.53     1_328.36       0.3708          1.0459            1.0453         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_135.83       282.41     1_418.24       0.4723          1.0291            1.0275         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_135.83       187.56     1_323.39       0.3705          1.0459            1.0453         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_135.83       280.74     1_416.56       0.4716          1.0291            1.0276         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_135.83       191.62     1_327.45       0.3705          1.0459            1.0453         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_135.83       284.79     1_420.61       0.4716          1.0291            1.0276         4.49
IVF-Binary-512-nl223-pca (self)                        1_135.83       436.68     1_572.51       0.3711          1.0457            1.0452         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)              1_499.14       107.40     1_606.54       0.1706          1.2099            1.1842         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)              1_499.14       108.46     1_607.60       0.1706          1.2102            1.1845         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)              1_499.14       108.71     1_607.85       0.1705          1.2103            1.1846         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)             1_499.14       196.29     1_695.43       0.3771          1.0444            1.0437         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)             1_499.14       284.88     1_784.02       0.4786          1.0283            1.0267         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)             1_499.14       190.04     1_689.18       0.3768          1.0445            1.0438         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)             1_499.14       284.95     1_784.09       0.4778          1.0284            1.0267         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)             1_499.14       198.24     1_697.38       0.3767          1.0445            1.0438         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)             1_499.14       297.56     1_796.70       0.4776          1.0285            1.0268         4.67
IVF-Binary-512-nl316-pca (self)                        1_499.14       444.93     1_944.07       0.3772          1.0444            1.0438         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)           1_804.56       142.25     1_946.81       0.1827          1.2009            1.1907         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)          1_804.56       148.24     1_952.80       0.1827          1.2009            1.1907         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)          1_804.56       147.02     1_951.58       0.1827          1.2009            1.1907         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)          1_804.56       236.59     2_041.15       0.3773          1.0441            1.0445         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)          1_804.56       328.46     2_133.02       0.4819          1.0279            1.0266         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)         1_804.56       236.64     2_041.20       0.3773          1.0441            1.0445         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)         1_804.56       340.63     2_145.19       0.4819          1.0279            1.0266         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)         1_804.56       253.93     2_058.49       0.3773          1.0441            1.0445         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)         1_804.56       339.61     2_144.17       0.4819          1.0279            1.0266         8.42
IVF-Binary-1024-nl158-random (self)                    1_804.56       603.55     2_408.11       0.3779          1.0441            1.0445         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_163.12       151.75     1_314.87       0.1855          1.1895            1.1800         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_163.12       150.68     1_313.81       0.1855          1.1897            1.1802         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_163.12       155.08     1_318.21       0.1855          1.1897            1.1802         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_163.12       246.13     1_409.25       0.3862          1.0418            1.0422         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_163.12       340.10     1_503.22       0.4929          1.0264            1.0252         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_163.12       248.42     1_411.55       0.3859          1.0419            1.0423         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_163.12       343.05     1_506.17       0.4924          1.0265            1.0253         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_163.12       252.93     1_416.05       0.3859          1.0419            1.0423         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_163.12       353.15     1_516.27       0.4924          1.0265            1.0253         8.54
IVF-Binary-1024-nl223-random (self)                    1_163.12       624.29     1_787.41       0.3871          1.0418            1.0422         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)          1_441.56       155.61     1_597.17       0.1870          1.1851            1.1757         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)          1_441.56       166.75     1_608.31       0.1869          1.1854            1.1760         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)          1_441.56       169.59     1_611.15       0.1869          1.1855            1.1761         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)         1_441.56       270.81     1_712.37       0.3891          1.0413            1.0415         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)         1_441.56       363.83     1_805.39       0.4957          1.0261            1.0248         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)         1_441.56       252.16     1_693.72       0.3887          1.0414            1.0416         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)         1_441.56       405.72     1_847.28       0.4951          1.0262            1.0248         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)         1_441.56       294.17     1_735.73       0.3886          1.0414            1.0416         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)         1_441.56       359.48     1_801.04       0.4950          1.0262            1.0249         8.73
IVF-Binary-1024-nl316-random (self)                    1_441.56       766.56     2_208.12       0.3899          1.0413            1.0416         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)              1_890.97       140.09     2_031.07       0.1841          1.1989            1.1881         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)             1_890.97       145.40     2_036.37       0.1841          1.1989            1.1881         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)             1_890.97       149.26     2_040.23       0.1841          1.1989            1.1881         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)             1_890.97       232.30     2_123.27       0.3820          1.0431            1.0438         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)             1_890.97       330.26     2_221.23       0.4889          1.0270            1.0258         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)            1_890.97       246.11     2_137.09       0.3820          1.0431            1.0438         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)            1_890.97       338.90     2_229.87       0.4889          1.0270            1.0258         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)            1_890.97       239.36     2_130.33       0.3820          1.0431            1.0438         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)            1_890.97       338.19     2_229.17       0.4889          1.0270            1.0258         8.42
IVF-Binary-1024-nl158-pca (self)                       1_890.97       609.72     2_500.69       0.3808          1.0432            1.0439         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_185.10       148.21     1_333.30       0.1869          1.1877            1.1786         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_185.10       151.18     1_336.27       0.1869          1.1879            1.1788         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_185.10       155.55     1_340.65       0.1869          1.1879            1.1788         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_185.10       243.15     1_428.24       0.3905          1.0407            1.0416         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_185.10       340.41     1_525.50       0.4989          1.0255            1.0247         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_185.10       242.20     1_427.30       0.3902          1.0408            1.0417         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_185.10       342.34     1_527.43       0.4984          1.0256            1.0247         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_185.10       247.09     1_432.18       0.3902          1.0408            1.0417         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_185.10       357.87     1_542.97       0.4984          1.0256            1.0247         8.54
IVF-Binary-1024-nl223-pca (self)                       1_185.10       628.29     1_813.38       0.3897          1.0409            1.0418         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_481.45       157.28     1_638.73       0.1883          1.1833            1.1740         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_481.45       158.02     1_639.47       0.1882          1.1836            1.1743         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_481.45       160.75     1_642.20       0.1882          1.1837            1.1744         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_481.45       258.83     1_740.29       0.3938          1.0401            1.0409         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_481.45       347.83     1_829.29       0.5030          1.0252            1.0243         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_481.45       256.72     1_738.17       0.3935          1.0402            1.0410         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_481.45       363.19     1_844.65       0.5023          1.0252            1.0243         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_481.45       262.38     1_743.83       0.3934          1.0402            1.0410         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_481.45       362.83     1_844.28       0.5022          1.0253            1.0243         8.73
IVF-Binary-1024-nl316-pca (self)                       1_481.45       667.86     2_149.31       0.3929          1.0403            1.0411         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)              1_652.67       280.79     1_933.46       0.1520          1.2698            1.2528         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)             1_652.67       283.37     1_936.03       0.1520          1.2698            1.2528         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)             1_652.67       284.14     1_936.81       0.1520          1.2698            1.2528         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)             1_652.67       359.48     2_012.15       0.3410          1.0600            1.0529         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)             1_652.67       627.07     2_279.73       0.4418          1.0366            1.0317         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)            1_652.67       349.30     2_001.97       0.3410          1.0600            1.0529         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)            1_652.67       630.21     2_282.87       0.4418          1.0366            1.0317         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)            1_652.67       357.09     2_009.76       0.3410          1.0600            1.0529         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)            1_652.67       632.60     2_285.26       0.4418          1.0366            1.0317         3.36
IVF-Binary-512-nl158-sign (self)                       1_652.67       967.31     2_619.98       0.3419          1.0590            1.0526         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               906.83       285.51     1_192.34       0.1530          1.2631            1.2472         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               906.83       288.70     1_195.54       0.1529          1.2639            1.2483         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               906.83       290.34     1_197.17       0.1529          1.2639            1.2483         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              906.83       362.58     1_269.42       0.3502          1.0559            1.0502         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              906.83       629.69     1_536.52       0.4489          1.0348            1.0306         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              906.83       357.23     1_264.07       0.3500          1.0560            1.0502         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              906.83       638.78     1_545.61       0.4482          1.0349            1.0307         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              906.83       363.14     1_269.98       0.3500          1.0560            1.0502         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              906.83       640.94     1_547.78       0.4482          1.0349            1.0307         3.49
IVF-Binary-512-nl223-sign (self)                         906.83       978.68     1_885.51       0.3506          1.0550            1.0500         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)             1_236.36       290.56     1_526.92       0.1530          1.2630            1.2450         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)             1_236.36       291.52     1_527.88       0.1529          1.2642            1.2464         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)             1_236.36       293.37     1_529.73       0.1528          1.2648            1.2472         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)            1_236.36       368.25     1_604.60       0.3525          1.0552            1.0494         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)            1_236.36       640.20     1_876.55       0.4493          1.0348            1.0305         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)            1_236.36       357.80     1_594.16       0.3521          1.0554            1.0496         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)            1_236.36       642.36     1_878.72       0.4482          1.0350            1.0306         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)            1_236.36       363.33     1_599.69       0.3519          1.0555            1.0496         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)            1_236.36       649.98     1_886.34       0.4478          1.0351            1.0307         3.67
IVF-Binary-512-nl316-sign (self)                       1_236.36       991.77     2_228.12       0.3528          1.0544            1.0494         3.67
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
Exhaustive (query)                                       102.04     1_853.63     1_955.67       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        102.04     6_220.64     6_322.68       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                203.60       287.54       491.14       0.1140          1.2809            1.2433         2.28
ExhaustiveBinary-256-random-rf10 (query)                 203.60       430.79       634.39       0.3148          1.0656            1.0476         2.28
ExhaustiveBinary-256-random-rf20 (query)                 203.60       566.07       769.66       0.4075          1.0420            1.0293         2.28
ExhaustiveBinary-256-random (self)                       203.60     1_536.09     1_739.69       0.3168          1.0618            1.0471         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   403.54       302.34       705.88       0.1054          1.3026            1.2617         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    403.54       454.67       858.21       0.3012          1.0735            1.0517         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    403.54       666.35     1_069.89       0.3931          1.0471            1.0315         2.28
ExhaustiveBinary-256-pca (self)                          403.54     1_319.90     1_723.44       0.3048          1.0710            1.0504         2.28
ExhaustiveBinary-512-random_no_rr (query)                306.84       399.52       706.36       0.1506          1.2094            1.1809         4.55
ExhaustiveBinary-512-random-rf10 (query)                 306.84       555.03       861.87       0.3395          1.0453            1.0426         4.55
ExhaustiveBinary-512-random-rf20 (query)                 306.84       715.21     1_022.04       0.4326          1.0293            1.0264         4.55
ExhaustiveBinary-512-random (self)                       306.84     1_738.99     2_045.83       0.3401          1.0435            1.0423         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   512.25       400.84       913.09       0.1459          1.2162            1.1914         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    512.25       553.50     1_065.75       0.3341          1.0468            1.0435         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    512.25       699.92     1_212.17       0.4278          1.0295            1.0269         4.55
ExhaustiveBinary-512-pca (self)                          512.25     1_738.54     2_250.79       0.3355          1.0454            1.0433         4.55
ExhaustiveBinary-1024-random_no_rr (query)               496.48       614.22     1_110.70       0.1761          1.1673            1.1571         9.11
ExhaustiveBinary-1024-random-rf10 (query)                496.48       788.39     1_284.87       0.3603          1.0383            1.0383         9.11
ExhaustiveBinary-1024-random-rf20 (query)                496.48       958.19     1_454.67       0.4618          1.0244            1.0230         9.11
ExhaustiveBinary-1024-random (self)                      496.48     2_527.71     3_024.19       0.3602          1.0377            1.0383         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  709.56       614.90     1_324.46       0.1756          1.1686            1.1586         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   709.56       790.75     1_500.30       0.3594          1.0382            1.0385         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   709.56       954.20     1_663.76       0.4576          1.0246            1.0237         9.11
ExhaustiveBinary-1024-pca (self)                         709.56     2_535.65     3_245.20       0.3584          1.0383            1.0389         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  135.93       840.02       975.95       0.1691          1.1871            1.1718         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   135.93       915.23     1_051.16       0.3431          1.0433            1.0415         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   135.93     1_420.75     1_556.68       0.4437          1.0266            1.0250         4.58
ExhaustiveBinary-768-sign (self)                         135.93     2_957.02     3_092.95       0.3438          1.0424            1.0413         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)            2_314.83        99.23     2_414.06       0.1164          1.2717            1.2403         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)           2_314.83       101.35     2_416.18       0.1164          1.2717            1.2403         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)           2_314.83       101.08     2_415.91       0.1164          1.2717            1.2403         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)           2_314.83       210.16     2_524.99       0.3175          1.0626            1.0468         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)           2_314.83       317.78     2_632.61       0.4096          1.0407            1.0290         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)          2_314.83       210.73     2_525.56       0.3175          1.0626            1.0468         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)          2_314.83       319.92     2_634.75       0.4096          1.0407            1.0290         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)          2_314.83       201.98     2_516.81       0.3175          1.0626            1.0468         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)          2_314.83       316.25     2_631.08       0.4096          1.0407            1.0290         2.74
IVF-Binary-256-nl158-random (self)                     2_314.83       420.35     2_735.17       0.3193          1.0593            1.0464         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)           1_433.55        95.19     1_528.74       0.1332          1.2302            1.1908         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)           1_433.55        98.57     1_532.12       0.1332          1.2302            1.1908         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)           1_433.55        96.51     1_530.06       0.1332          1.2302            1.1908         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)          1_433.55       208.60     1_642.15       0.3572          1.0474            1.0377         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)          1_433.55       313.42     1_746.97       0.4565          1.0309            1.0232         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)          1_433.55       201.49     1_635.04       0.3572          1.0474            1.0377         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)          1_433.55       316.12     1_749.67       0.4565          1.0309            1.0232         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)          1_433.55       210.43     1_643.98       0.3572          1.0474            1.0377         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)          1_433.55       321.72     1_755.27       0.4565          1.0309            1.0232         2.93
IVF-Binary-256-nl223-random (self)                     1_433.55       424.58     1_858.13       0.3589          1.0438            1.0373         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)           1_781.92       102.76     1_884.67       0.1402          1.2168            1.1748         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)           1_781.92       103.41     1_885.33       0.1402          1.2168            1.1748         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)           1_781.92       107.24     1_889.16       0.1402          1.2168            1.1748         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)          1_781.92       212.52     1_994.44       0.3656          1.0444            1.0361         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)          1_781.92       328.00     2_109.92       0.4629          1.0293            1.0225         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)          1_781.92       210.48     1_992.40       0.3656          1.0444            1.0361         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)          1_781.92       320.28     2_102.20       0.4628          1.0293            1.0225         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)          1_781.92       212.60     1_994.52       0.3656          1.0444            1.0361         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)          1_781.92       323.96     2_105.88       0.4628          1.0293            1.0225         3.21
IVF-Binary-256-nl316-random (self)                     1_781.92       458.18     2_240.10       0.3670          1.0410            1.0358         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)               2_472.95        86.75     2_559.69       0.1080          1.2909            1.2569         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)              2_472.95        85.33     2_558.28       0.1080          1.2909            1.2569         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)              2_472.95        90.42     2_563.37       0.1080          1.2909            1.2569         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)              2_472.95       191.57     2_664.52       0.3044          1.0707            1.0512         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)              2_472.95       299.30     2_772.24       0.3972          1.0447            1.0309         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)             2_472.95       188.50     2_661.44       0.3044          1.0707            1.0512         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)             2_472.95       299.34     2_772.28       0.3972          1.0447            1.0309         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)             2_472.95       192.06     2_665.01       0.3044          1.0707            1.0512         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)             2_472.95       308.08     2_781.02       0.3972          1.0447            1.0309         2.74
IVF-Binary-256-nl158-pca (self)                        2_472.95       395.35     2_868.30       0.3084          1.0672            1.0497         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)              1_562.41        91.86     1_654.28       0.1269          1.2393            1.2037         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)              1_562.41        92.66     1_655.07       0.1269          1.2393            1.2037         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)              1_562.41        96.60     1_659.02       0.1269          1.2393            1.2037         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)             1_562.41       206.64     1_769.05       0.3608          1.0475            1.0373         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)             1_562.41       309.97     1_872.38       0.4623          1.0301            1.0228         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)             1_562.41       197.41     1_759.82       0.3608          1.0475            1.0373         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)             1_562.41       309.93     1_872.35       0.4623          1.0301            1.0228         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)             1_562.41       203.37     1_765.79       0.3608          1.0475            1.0373         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)             1_562.41       320.84     1_883.26       0.4623          1.0301            1.0228         2.93
IVF-Binary-256-nl223-pca (self)                        1_562.41       409.83     1_972.25       0.3641          1.0449            1.0366         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)              2_005.51       103.20     2_108.71       0.1367          1.2212            1.1848         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)              2_005.51       108.42     2_113.94       0.1367          1.2212            1.1848         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)              2_005.51       104.06     2_109.57       0.1367          1.2212            1.1848         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)             2_005.51       210.58     2_216.10       0.3742          1.0434            1.0348         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)             2_005.51       319.46     2_324.97       0.4724          1.0282            1.0217         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)             2_005.51       215.99     2_221.50       0.3742          1.0434            1.0348         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)             2_005.51       317.97     2_323.49       0.4723          1.0283            1.0217         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)             2_005.51       217.77     2_223.28       0.3742          1.0434            1.0348         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)             2_005.51       324.83     2_330.34       0.4723          1.0283            1.0217         3.21
IVF-Binary-256-nl316-pca (self)                        2_005.51       442.51     2_448.02       0.3770          1.0409            1.0340         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)            2_381.13       118.97     2_500.10       0.1520          1.2060            1.1791         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)           2_381.13       124.08     2_505.21       0.1520          1.2060            1.1791         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)           2_381.13       128.96     2_510.09       0.1520          1.2060            1.1791         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)           2_381.13       234.17     2_615.30       0.3405          1.0447            1.0423         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)           2_381.13       345.28     2_726.41       0.4339          1.0290            1.0262         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)          2_381.13       232.70     2_613.83       0.3405          1.0447            1.0423         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)          2_381.13       347.09     2_728.22       0.4339          1.0290            1.0262         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)          2_381.13       237.99     2_619.12       0.3405          1.0447            1.0423         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)          2_381.13       351.30     2_732.43       0.4339          1.0290            1.0262         5.02
IVF-Binary-512-nl158-random (self)                     2_381.13       563.61     2_944.74       0.3409          1.0431            1.0421         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)           1_509.82       132.56     1_642.39       0.1603          1.1840            1.1576         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)           1_509.82       133.85     1_643.67       0.1603          1.1840            1.1576         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)           1_509.82       132.98     1_642.81       0.1603          1.1840            1.1576         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)          1_509.82       241.68     1_751.51       0.3578          1.0405            1.0380         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)          1_509.82       354.55     1_864.38       0.4564          1.0262            1.0235         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)          1_509.82       242.21     1_752.03       0.3578          1.0405            1.0380         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)          1_509.82       357.68     1_867.50       0.4564          1.0262            1.0235         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)          1_509.82       245.50     1_755.33       0.3578          1.0405            1.0380         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)          1_509.82       361.31     1_871.13       0.4564          1.0262            1.0235         5.21
IVF-Binary-512-nl223-random (self)                     1_509.82       574.38     2_084.21       0.3585          1.0389            1.0379         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)           1_898.28       139.27     2_037.55       0.1624          1.1788            1.1525         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)           1_898.28       138.11     2_036.39       0.1624          1.1789            1.1525         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)           1_898.28       143.65     2_041.93       0.1624          1.1789            1.1525         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)          1_898.28       251.55     2_149.83       0.3613          1.0391            1.0372         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)          1_898.28       372.02     2_270.31       0.4582          1.0253            1.0234         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)          1_898.28       246.29     2_144.58       0.3613          1.0392            1.0372         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)          1_898.28       370.36     2_268.65       0.4582          1.0254            1.0234         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)          1_898.28       266.34     2_164.62       0.3613          1.0392            1.0372         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)          1_898.28       394.80     2_293.09       0.4582          1.0254            1.0234         5.48
IVF-Binary-512-nl316-random (self)                     1_898.28       625.41     2_523.69       0.3616          1.0379            1.0373         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)               2_603.57       120.53     2_724.10       0.1476          1.2119            1.1892         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)              2_603.57       123.54     2_727.11       0.1476          1.2119            1.1892         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)              2_603.57       126.18     2_729.74       0.1476          1.2119            1.1892         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)              2_603.57       234.10     2_837.67       0.3359          1.0458            1.0431         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)              2_603.57       345.70     2_949.26       0.4299          1.0290            1.0267         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)             2_603.57       234.14     2_837.71       0.3359          1.0458            1.0431         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)             2_603.57       347.10     2_950.66       0.4299          1.0290            1.0267         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)             2_603.57       232.15     2_835.72       0.3359          1.0458            1.0431         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)             2_603.57       350.08     2_953.65       0.4299          1.0290            1.0267         5.02
IVF-Binary-512-nl158-pca (self)                        2_603.57       542.57     3_146.14       0.3374          1.0443            1.0428         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_691.20       127.77     1_818.97       0.1584          1.1837            1.1596         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_691.20       135.85     1_827.05       0.1584          1.1837            1.1596         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_691.20       133.45     1_824.65       0.1584          1.1837            1.1596         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_691.20       239.49     1_930.70       0.3583          1.0398            1.0379         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_691.20       354.72     2_045.93       0.4550          1.0259            1.0235         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_691.20       238.50     1_929.70       0.3583          1.0398            1.0379         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_691.20       365.60     2_056.80       0.4550          1.0259            1.0235         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_691.20       243.00     1_934.20       0.3583          1.0398            1.0379         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_691.20       362.90     2_054.10       0.4550          1.0259            1.0235         5.21
IVF-Binary-512-nl223-pca (self)                        1_691.20       570.34     2_261.54       0.3590          1.0390            1.0378         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)              2_132.67       138.75     2_271.42       0.1627          1.1763            1.1519         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)              2_132.67       143.50     2_276.16       0.1627          1.1763            1.1519         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)              2_132.67       145.27     2_277.93       0.1627          1.1763            1.1519         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)             2_132.67       250.74     2_383.40       0.3626          1.0388            1.0370         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)             2_132.67       366.33     2_499.00       0.4585          1.0255            1.0231         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)             2_132.67       245.57     2_378.24       0.3626          1.0389            1.0370         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)             2_132.67       369.03     2_501.70       0.4584          1.0255            1.0231         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)             2_132.67       257.31     2_389.98       0.3626          1.0389            1.0370         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)             2_132.67       374.30     2_506.97       0.4584          1.0255            1.0231         5.48
IVF-Binary-512-nl316-pca (self)                        2_132.67       597.10     2_729.77       0.3635          1.0382            1.0370         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)           2_586.18       197.16     2_783.34       0.1768          1.1661            1.1563         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)          2_586.18       195.41     2_781.59       0.1768          1.1661            1.1563         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)          2_586.18       198.15     2_784.33       0.1768          1.1661            1.1563         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)          2_586.18       304.26     2_890.45       0.3609          1.0381            1.0382         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)          2_586.18       436.67     3_022.85       0.4626          1.0243            1.0230         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)         2_586.18       307.70     2_893.88       0.3609          1.0381            1.0382         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)         2_586.18       445.58     3_031.77       0.4626          1.0243            1.0230         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)         2_586.18       319.72     2_905.91       0.3609          1.0381            1.0382         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)         2_586.18       445.75     3_031.93       0.4626          1.0243            1.0230         9.57
IVF-Binary-1024-nl158-random (self)                    2_586.18       823.83     3_410.01       0.3608          1.0375            1.0382         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_712.21       208.63     1_920.84       0.1797          1.1568            1.1474         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_712.21       202.72     1_914.93       0.1797          1.1568            1.1474         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_712.21       217.69     1_929.91       0.1797          1.1568            1.1474         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_712.21       335.02     2_047.24       0.3717          1.0360            1.0360         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_712.21       453.08     2_165.29       0.4749          1.0229            1.0217         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_712.21       334.00     2_046.21       0.3717          1.0360            1.0360         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_712.21       459.63     2_171.84       0.4749          1.0229            1.0217         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_712.21       334.69     2_046.91       0.3717          1.0360            1.0360         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_712.21       479.28     2_191.49       0.4749          1.0229            1.0217         9.76
IVF-Binary-1024-nl223-random (self)                    1_712.21       841.69     2_553.90       0.3714          1.0355            1.0360         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)          2_145.10       223.90     2_369.00       0.1804          1.1544            1.1451        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)          2_145.10       208.20     2_353.31       0.1804          1.1545            1.1451        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)          2_145.10       216.65     2_361.75       0.1804          1.1545            1.1451        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)         2_145.10       343.25     2_488.35       0.3732          1.0356            1.0359        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)         2_145.10       471.11     2_616.21       0.4755          1.0227            1.0217        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)         2_145.10       346.93     2_492.03       0.3732          1.0356            1.0359        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)         2_145.10       466.98     2_612.08       0.4755          1.0228            1.0217        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)         2_145.10       350.92     2_496.02       0.3732          1.0356            1.0359        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)         2_145.10       480.68     2_625.78       0.4755          1.0228            1.0217        10.04
IVF-Binary-1024-nl316-random (self)                    2_145.10       876.74     3_021.84       0.3730          1.0351            1.0357        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)              2_799.06       197.56     2_996.62       0.1762          1.1673            1.1573         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)             2_799.06       197.23     2_996.29       0.1762          1.1673            1.1573         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)             2_799.06       196.47     2_995.53       0.1762          1.1673            1.1573         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)             2_799.06       314.83     3_113.89       0.3600          1.0380            1.0384         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)             2_799.06       431.82     3_230.89       0.4585          1.0245            1.0237         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)            2_799.06       307.70     3_106.77       0.3600          1.0380            1.0384         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)            2_799.06       450.21     3_249.28       0.4585          1.0245            1.0237         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)            2_799.06       312.71     3_111.78       0.3600          1.0380            1.0384         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)            2_799.06       451.75     3_250.82       0.4585          1.0245            1.0237         9.57
IVF-Binary-1024-nl158-pca (self)                       2_799.06       838.64     3_637.70       0.3591          1.0381            1.0388         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_927.81       197.81     2_125.62       0.1794          1.1567            1.1470         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_927.81       202.46     2_130.27       0.1794          1.1567            1.1470         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_927.81       217.75     2_145.56       0.1794          1.1567            1.1470         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_927.81       321.55     2_249.35       0.3709          1.0358            1.0361         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_927.81       454.91     2_382.72       0.4723          1.0231            1.0221         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_927.81       332.32     2_260.12       0.3709          1.0358            1.0361         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_927.81       461.32     2_389.13       0.4723          1.0231            1.0221         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_927.81       335.96     2_263.77       0.3709          1.0358            1.0361         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_927.81       466.69     2_394.50       0.4723          1.0231            1.0221         9.76
IVF-Binary-1024-nl223-pca (self)                       1_927.81       849.89     2_777.69       0.3702          1.0359            1.0363         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             2_304.27       212.78     2_517.05       0.1805          1.1540            1.1447        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)             2_304.27       212.28     2_516.54       0.1805          1.1540            1.1447        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)             2_304.27       215.44     2_519.71       0.1805          1.1540            1.1447        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)            2_304.27       342.46     2_646.72       0.3730          1.0355            1.0357        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)            2_304.27       470.90     2_775.16       0.4735          1.0228            1.0220        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)            2_304.27       344.57     2_648.84       0.3730          1.0355            1.0357        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)            2_304.27       472.51     2_776.77       0.4735          1.0228            1.0220        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)            2_304.27       351.35     2_655.62       0.3730          1.0355            1.0357        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)            2_304.27       483.16     2_787.42       0.4735          1.0228            1.0220        10.04
IVF-Binary-1024-nl316-pca (self)                       2_304.27       882.28     3_186.55       0.3720          1.0356            1.0360        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)              2_239.25       402.75     2_642.00       0.1693          1.1869            1.1720         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)             2_239.25       411.37     2_650.62       0.1693          1.1869            1.1720         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)             2_239.25       409.06     2_648.31       0.1693          1.1869            1.1720         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)             2_239.25       491.18     2_730.43       0.3435          1.0431            1.0414         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)             2_239.25       889.98     3_129.23       0.4441          1.0265            1.0249         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)            2_239.25       489.67     2_728.92       0.3435          1.0431            1.0414         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)            2_239.25       896.47     3_135.72       0.4441          1.0265            1.0249         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)            2_239.25       495.90     2_735.15       0.3435          1.0431            1.0414         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)            2_239.25       901.42     3_140.67       0.4441          1.0265            1.0249         5.04
IVF-Binary-768-nl158-sign (self)                       2_239.25     1_404.92     3_644.17       0.3441          1.0422            1.0413         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)             1_323.82       426.68     1_750.50       0.1694          1.1869            1.1715         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)             1_323.82       412.04     1_735.86       0.1694          1.1869            1.1715         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)             1_323.82       417.29     1_741.11       0.1694          1.1869            1.1715         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)            1_323.82       496.12     1_819.94       0.3491          1.0415            1.0400         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)            1_323.82       903.65     2_227.47       0.4485          1.0259            1.0244         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)            1_323.82       497.65     1_821.47       0.3491          1.0415            1.0400         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)            1_323.82       899.66     2_223.48       0.4485          1.0259            1.0244         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)            1_323.82       507.86     1_831.68       0.3491          1.0415            1.0400         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)            1_323.82       959.19     2_283.01       0.4485          1.0259            1.0244         5.23
IVF-Binary-768-nl223-sign (self)                       1_323.82     1_417.75     2_741.57       0.3494          1.0408            1.0400         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)             1_741.71       420.65     2_162.37       0.1695          1.1865            1.1713         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)             1_741.71       419.21     2_160.92       0.1695          1.1866            1.1713         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)             1_741.71       423.91     2_165.62       0.1695          1.1866            1.1713         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)            1_741.71       518.13     2_259.84       0.3496          1.0411            1.0400         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)            1_741.71       909.97     2_651.69       0.4487          1.0257            1.0245         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)            1_741.71       527.21     2_268.92       0.3495          1.0411            1.0400         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)            1_741.71       912.50     2_654.21       0.4486          1.0257            1.0245         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)            1_741.71       526.82     2_268.54       0.3495          1.0411            1.0400         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)            1_741.71       923.54     2_665.25       0.4486          1.0257            1.0245         5.51
IVF-Binary-768-nl316-sign (self)                       1_741.71     1_455.42     3_197.13       0.3501          1.0405            1.0398         5.51
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
Exhaustive (query)                                        32.66       704.90       737.57       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.66     2_349.86     2_382.52       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                 71.43       240.65       312.08       0.0970          1.6334            1.6378         1.78
ExhaustiveBinary-256-random-rf10 (query)                  71.43       342.56       414.00       0.3643          1.1391            1.1302         1.78
ExhaustiveBinary-256-random-rf20 (query)                  71.43       441.29       512.72       0.5087          1.0798            1.0701         1.78
ExhaustiveBinary-256-random (self)                        71.43     1_098.24     1_169.68       0.3862          1.1443            1.1409         1.78
ExhaustiveBinary-256-pca_no_rr (query)                    93.98       236.46       330.44       0.0922          1.6524            1.6606         1.78
ExhaustiveBinary-256-pca-rf10 (query)                     93.98       342.73       436.71       0.3517          1.1465            1.1384         1.78
ExhaustiveBinary-256-pca-rf20 (query)                     93.98       441.63       535.61       0.4943          1.0846            1.0744         1.78
ExhaustiveBinary-256-pca (self)                           93.98     1_102.41     1_196.39       0.3764          1.1502            1.1477         1.78
ExhaustiveBinary-512-random_no_rr (query)                 83.31       345.53       428.84       0.1464          1.5035            1.5085         3.55
ExhaustiveBinary-512-random-rf10 (query)                  83.31       455.17       538.48       0.4596          1.0936            1.0901         3.55
ExhaustiveBinary-512-random-rf20 (query)                  83.31       560.80       644.11       0.6085          1.0504            1.0459         3.55
ExhaustiveBinary-512-random (self)                        83.31     1_484.33     1_567.65       0.4800          1.0996            1.0995         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   112.09       345.46       457.55       0.1458          1.5041            1.5097         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    112.09       460.68       572.78       0.4543          1.0952            1.0911         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    112.09       571.94       684.03       0.6037          1.0513            1.0464         3.55
ExhaustiveBinary-512-pca (self)                          112.09     1_497.73     1_609.82       0.4777          1.1003            1.0997         3.55
ExhaustiveBinary-1024-random_no_rr (query)               114.87       503.26       618.12       0.2155          1.3655            1.3721         7.10
ExhaustiveBinary-1024-random-rf10 (query)                114.87       621.85       736.72       0.5869          1.0540            1.0515         7.10
ExhaustiveBinary-1024-random-rf20 (query)                114.87       735.60       850.46       0.7380          1.0260            1.0224         7.10
ExhaustiveBinary-1024-random (self)                      114.87     2_054.89     2_169.76       0.6118          1.0576            1.0546         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  141.22       507.77       648.99       0.2122          1.3735            1.3798         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   141.22       626.62       767.84       0.5776          1.0560            1.0532         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   141.22       737.95       879.17       0.7291          1.0273            1.0232         7.10
ExhaustiveBinary-1024-pca (self)                         141.22     2_055.69     2_196.91       0.6017          1.0602            1.0571         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   41.85       443.51       485.36       0.1044          1.6421            1.6499         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    41.85       480.60       522.45       0.3737          1.1368            1.1275         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    41.85       730.79       772.64       0.5265          1.0745            1.0646         1.53
ExhaustiveBinary-256-sign (self)                          41.85     1_588.20     1_630.05       0.3940          1.1439            1.1394         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)              910.91        50.03       960.94       0.1016          1.6177            1.6285         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)             910.91        52.73       963.64       0.1016          1.6179            1.6286         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)             910.91        58.06       968.97       0.1016          1.6179            1.6286         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)             910.91       100.73     1_011.64       0.3750          1.1333            1.1268         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)             910.91       150.54     1_061.45       0.5191          1.0759            1.0681         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)            910.91        97.18     1_008.09       0.3742          1.1334            1.1269         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)            910.91       150.47     1_061.38       0.5182          1.0761            1.0681         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)            910.91       100.37     1_011.28       0.3742          1.1335            1.1269         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)            910.91       155.18     1_066.09       0.5181          1.0761            1.0681         1.93
IVF-Binary-256-nl158-random (self)                       910.91       224.23     1_135.14       0.3960          1.1379            1.1379         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             646.00        45.66       691.65       0.1120          1.5877            1.5867         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             646.00        47.12       693.11       0.1119          1.5883            1.5872         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             646.00        51.00       696.99       0.1119          1.5884            1.5874         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            646.00        98.42       744.41       0.3934          1.1239            1.1158         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            646.00       150.05       796.05       0.5355          1.0713            1.0628         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            646.00        98.42       744.42       0.3930          1.1240            1.1160         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            646.00       151.39       797.39       0.5350          1.0714            1.0628         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            646.00       103.16       749.16       0.3929          1.1240            1.1160         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            646.00       166.01       812.00       0.5349          1.0714            1.0628         2.00
IVF-Binary-256-nl223-random (self)                       646.00       226.70       872.70       0.4141          1.1287            1.1268         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             839.39        48.15       887.54       0.1175          1.5678            1.5685         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             839.39        48.78       888.18       0.1174          1.5683            1.5691         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             839.39        52.93       892.32       0.1174          1.5686            1.5693         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            839.39       101.13       940.52       0.4048          1.1173            1.1103         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            839.39       150.52       989.91       0.5480          1.0673            1.0599         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            839.39        99.73       939.13       0.4042          1.1175            1.1108         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            839.39       150.13       989.52       0.5471          1.0675            1.0601         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            839.39       104.48       943.87       0.4041          1.1175            1.1108         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            839.39       162.17     1_001.56       0.5470          1.0675            1.0601         2.09
IVF-Binary-256-nl316-random (self)                       839.39       235.65     1_075.04       0.4254          1.1215            1.1216         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)                 915.45        41.36       956.80       0.0972          1.6371            1.6509         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)                915.45        43.55       959.00       0.0972          1.6373            1.6510         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)                915.45        46.17       961.61       0.0972          1.6373            1.6510         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)                915.45        93.39     1_008.84       0.3614          1.1414            1.1351         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)                915.45       144.37     1_059.82       0.5035          1.0811            1.0727         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)               915.45        94.21     1_009.66       0.3609          1.1415            1.1351         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)               915.45       145.33     1_060.78       0.5028          1.0812            1.0727         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)               915.45        97.85     1_013.29       0.3609          1.1415            1.1351         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)               915.45       155.59     1_071.04       0.5028          1.0812            1.0727         1.93
IVF-Binary-256-nl158-pca (self)                          915.45       215.93     1_131.37       0.3857          1.1449            1.1445         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                667.38        45.69       713.07       0.1097          1.5958            1.5999         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                667.38        46.29       713.68       0.1096          1.5968            1.6008         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                667.38        50.96       718.34       0.1096          1.5969            1.6009         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               667.38       100.01       767.39       0.3870          1.1269            1.1204         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               667.38       147.00       814.38       0.5267          1.0735            1.0656         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               667.38        96.51       763.89       0.3865          1.1271            1.1205         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               667.38       148.27       815.65       0.5258          1.0737            1.0659         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               667.38       101.30       768.69       0.3864          1.1271            1.1205         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               667.38       153.34       820.72       0.5258          1.0737            1.0659         2.00
IVF-Binary-256-nl223-pca (self)                          667.38       225.24       892.62       0.4102          1.1301            1.1303         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                881.52        47.42       928.93       0.1159          1.5754            1.5785         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                881.52        47.92       929.43       0.1158          1.5761            1.5790         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                881.52        51.81       933.33       0.1157          1.5767            1.5793         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               881.52       101.87       983.39       0.3971          1.1214            1.1151         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               881.52       156.39     1_037.91       0.5370          1.0701            1.0626         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               881.52        99.86       981.38       0.3965          1.1217            1.1154         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               881.52       149.17     1_030.69       0.5362          1.0703            1.0628         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               881.52       103.16       984.68       0.3963          1.1218            1.1155         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               881.52       153.23     1_034.75       0.5360          1.0704            1.0629         2.09
IVF-Binary-256-nl316-pca (self)                          881.52       233.62     1_115.14       0.4200          1.1246            1.1253         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)              906.99        58.97       965.95       0.1496          1.4966            1.5039         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)             906.99        62.86       969.85       0.1496          1.4966            1.5039         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)             906.99        68.10       975.09       0.1496          1.4966            1.5039         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)             906.99       115.14     1_022.13       0.4642          1.0917            1.0890         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)             906.99       172.05     1_079.03       0.6126          1.0494            1.0452         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)            906.99       117.33     1_024.32       0.4641          1.0917            1.0890         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)            906.99       169.41     1_076.39       0.6125          1.0494            1.0452         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)            906.99       128.12     1_035.11       0.4641          1.0917            1.0890         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)            906.99       190.02     1_097.01       0.6125          1.0494            1.0452         3.71
IVF-Binary-512-nl158-random (self)                       906.99       300.14     1_207.13       0.4841          1.0980            1.0982         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             706.03        64.71       770.74       0.1563          1.4801            1.4855         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             706.03        64.94       770.96       0.1562          1.4803            1.4857         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             706.03        71.66       777.68       0.1562          1.4803            1.4857         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            706.03       119.27       825.30       0.4750          1.0879            1.0850         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            706.03       174.64       880.67       0.6213          1.0475            1.0433         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            706.03       118.34       824.36       0.4748          1.0879            1.0851         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            706.03       170.83       876.85       0.6210          1.0476            1.0433         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            706.03       130.44       836.47       0.4748          1.0880            1.0851         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            706.03       179.92       885.94       0.6210          1.0476            1.0433         3.77
IVF-Binary-512-nl223-random (self)                       706.03       298.34     1_004.37       0.4945          1.0941            1.0938         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             855.64        65.08       920.72       0.1594          1.4708            1.4746         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             855.64        67.46       923.10       0.1594          1.4711            1.4747         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             855.64        72.30       927.94       0.1594          1.4711            1.4747         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            855.64       120.67       976.32       0.4807          1.0857            1.0829         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            855.64       170.99     1_026.64       0.6271          1.0463            1.0423         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            855.64       119.10       974.74       0.4803          1.0858            1.0830         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            855.64       172.40     1_028.04       0.6266          1.0464            1.0425         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            855.64       124.80       980.44       0.4803          1.0859            1.0831         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            855.64       178.25     1_033.89       0.6265          1.0464            1.0425         3.86
IVF-Binary-512-nl316-random (self)                       855.64       304.18     1_159.83       0.4996          1.0921            1.0915         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)                 935.20        61.55       996.76       0.1487          1.4983            1.5052         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)                935.20        61.28       996.48       0.1487          1.4983            1.5052         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)                935.20        66.92     1_002.13       0.1487          1.4983            1.5052         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)                935.20       114.25     1_049.45       0.4587          1.0935            1.0902         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)                935.20       169.79     1_104.99       0.6076          1.0502            1.0457         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)               935.20       121.21     1_056.41       0.4586          1.0935            1.0902         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)               935.20       170.45     1_105.65       0.6074          1.0503            1.0457         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)               935.20       121.94     1_057.15       0.4586          1.0935            1.0902         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)               935.20       176.85     1_112.05       0.6074          1.0503            1.0457         3.71
IVF-Binary-512-nl158-pca (self)                          935.20       303.61     1_238.81       0.4816          1.0989            1.0987         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                664.26        63.67       727.92       0.1556          1.4810            1.4845         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                664.26        64.05       728.30       0.1555          1.4814            1.4852         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                664.26        71.73       735.99       0.1555          1.4814            1.4853         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               664.26       117.95       782.21       0.4716          1.0888            1.0853         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               664.26       171.34       835.60       0.6190          1.0478            1.0432         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               664.26       119.45       783.70       0.4712          1.0890            1.0855         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               664.26       171.41       835.66       0.6185          1.0479            1.0432         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               664.26       125.77       790.03       0.4712          1.0890            1.0855         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               664.26       177.77       842.02       0.6185          1.0479            1.0432         3.77
IVF-Binary-512-nl223-pca (self)                          664.26       311.19       975.45       0.4934          1.0944            1.0939         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                885.81        66.78       952.59       0.1591          1.4714            1.4751         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                885.81        66.67       952.47       0.1590          1.4717            1.4756         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                885.81        73.94       959.75       0.1590          1.4719            1.4757         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               885.81       121.26     1_007.06       0.4769          1.0868            1.0833         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               885.81       174.97     1_060.77       0.6238          1.0467            1.0422         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               885.81       119.83     1_005.64       0.4764          1.0870            1.0835         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               885.81       172.25     1_058.05       0.6231          1.0468            1.0424         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               885.81       125.16     1_010.97       0.4764          1.0870            1.0835         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               885.81       178.46     1_064.26       0.6230          1.0469            1.0424         3.86
IVF-Binary-512-nl316-pca (self)                          885.81       303.86     1_189.67       0.4985          1.0924            1.0920         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)             942.12        90.01     1_032.13       0.2170          1.3632            1.3702         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)            942.12        94.26     1_036.38       0.2170          1.3632            1.3702         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)            942.12       103.27     1_045.39       0.2170          1.3632            1.3702         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)            942.12       149.25     1_091.37       0.5886          1.0535            1.0510         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)            942.12       205.89     1_148.01       0.7396          1.0257            1.0221         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)           942.12       154.17     1_096.29       0.5886          1.0535            1.0510         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)           942.12       208.16     1_150.28       0.7395          1.0257            1.0221         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)           942.12       160.39     1_102.51       0.5886          1.0535            1.0510         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)           942.12       218.00     1_160.12       0.7395          1.0257            1.0221         7.26
IVF-Binary-1024-nl158-random (self)                      942.12       420.15     1_362.27       0.6136          1.0570            1.0542         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            670.43        93.90       764.33       0.2204          1.3573            1.3647         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            670.43        98.52       768.94       0.2204          1.3574            1.3649         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            670.43       108.09       778.52       0.2204          1.3574            1.3649         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           670.43       161.01       831.43       0.5938          1.0524            1.0499         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           670.43       208.11       878.53       0.7439          1.0252            1.0215         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           670.43       153.67       824.10       0.5936          1.0525            1.0499         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           670.43       209.35       879.78       0.7437          1.0252            1.0215         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           670.43       163.83       834.26       0.5936          1.0525            1.0499         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           670.43       219.37       889.80       0.7437          1.0252            1.0215         7.32
IVF-Binary-1024-nl223-random (self)                      670.43       419.56     1_089.98       0.6185          1.0559            1.0530         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            896.41        97.45       993.86       0.2220          1.3537            1.3607         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)            896.41        99.23       995.64       0.2219          1.3538            1.3609         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)            896.41       106.15     1_002.56       0.2219          1.3538            1.3609         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)           896.41       156.18     1_052.59       0.5965          1.0517            1.0493         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)           896.41       208.43     1_104.84       0.7465          1.0248            1.0210         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)           896.41       156.05     1_052.47       0.5961          1.0518            1.0493         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)           896.41       209.48     1_105.89       0.7462          1.0248            1.0211         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)           896.41       164.98     1_061.39       0.5961          1.0518            1.0493         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)           896.41       230.28     1_126.69       0.7461          1.0248            1.0211         7.42
IVF-Binary-1024-nl316-random (self)                      896.41       426.01     1_322.42       0.6212          1.0551            1.0523         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)                973.49        90.09     1_063.58       0.2134          1.3713            1.3785         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)               973.49        93.14     1_066.63       0.2134          1.3713            1.3785         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)               973.49       101.53     1_075.02       0.2134          1.3713            1.3785         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)               973.49       148.41     1_121.91       0.5796          1.0555            1.0526         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)               973.49       203.33     1_176.82       0.7307          1.0270            1.0231         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)              973.49       157.07     1_130.56       0.5796          1.0555            1.0526         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)              973.49       211.10     1_184.59       0.7306          1.0270            1.0231         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)              973.49       162.42     1_135.91       0.5796          1.0555            1.0526         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)              973.49       215.72     1_189.21       0.7306          1.0270            1.0231         7.26
IVF-Binary-1024-nl158-pca (self)                         973.49       418.51     1_392.00       0.6034          1.0597            1.0567         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               707.11        94.58       801.70       0.2175          1.3639            1.3706         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               707.11        99.05       806.16       0.2175          1.3640            1.3708         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               707.11       108.74       815.85       0.2175          1.3640            1.3708         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              707.11       151.70       858.81       0.5850          1.0542            1.0513         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              707.11       206.28       913.39       0.7356          1.0263            1.0224         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              707.11       154.91       862.03       0.5848          1.0543            1.0513         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              707.11       210.32       917.43       0.7353          1.0264            1.0224         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              707.11       165.49       872.60       0.5847          1.0543            1.0513         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              707.11       219.26       926.37       0.7352          1.0264            1.0224         7.32
IVF-Binary-1024-nl223-pca (self)                         707.11       419.88     1_126.99       0.6093          1.0581            1.0551         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               916.33        97.55     1_013.89       0.2188          1.3604            1.3660         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)               916.33        97.90     1_014.23       0.2187          1.3606            1.3663         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)               916.33       105.61     1_021.94       0.2187          1.3606            1.3663         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)              916.33       156.55     1_072.88       0.5881          1.0534            1.0504         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)              916.33       207.44     1_123.77       0.7382          1.0260            1.0220         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)              916.33       154.81     1_071.15       0.5877          1.0535            1.0505         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)              916.33       212.73     1_129.06       0.7378          1.0260            1.0221         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)              916.33       162.82     1_079.15       0.5877          1.0535            1.0505         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)              916.33       222.59     1_138.92       0.7377          1.0260            1.0221         7.42
IVF-Binary-1024-nl316-pca (self)                         916.33       424.85     1_341.19       0.6116          1.0575            1.0543         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)                881.09       153.14     1_034.23       0.1043          1.6446            1.6491         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               881.09       155.46     1_036.54       0.1043          1.6447            1.6493         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               881.09       161.36     1_042.45       0.1043          1.6447            1.6493         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               881.09       189.98     1_071.07       0.3792          1.1342            1.1256         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               881.09       335.42     1_216.50       0.5308          1.0735            1.0642         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              881.09       191.37     1_072.45       0.3782          1.1344            1.1257         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              881.09       338.94     1_220.02       0.5303          1.0736            1.0642         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              881.09       197.00     1_078.08       0.3782          1.1344            1.1257         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              881.09       349.48     1_230.57       0.5302          1.0736            1.0642         1.68
IVF-Binary-256-nl158-sign (self)                         881.09       526.72     1_407.80       0.3978          1.1422            1.1381         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               657.12       157.60       814.72       0.1047          1.6393            1.6468         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               657.12       155.16       812.27       0.1045          1.6406            1.6480         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               657.12       159.81       816.93       0.1045          1.6410            1.6482         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              657.12       195.36       852.48       0.3864          1.1295            1.1209         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              657.12       340.00       997.12       0.5362          1.0716            1.0627         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              657.12       192.37       849.49       0.3860          1.1297            1.1212         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              657.12       336.40       993.52       0.5354          1.0718            1.0628         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              657.12       202.32       859.44       0.3858          1.1298            1.1212         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              657.12       345.06     1_002.18       0.5352          1.0718            1.0628         1.75
IVF-Binary-256-nl223-sign (self)                         657.12       530.92     1_188.04       0.4052          1.1380            1.1337         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               812.62       153.12       965.74       0.1054          1.6361            1.6424         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               812.62       157.68       970.31       0.1053          1.6368            1.6429         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               812.62       167.81       980.44       0.1053          1.6375            1.6439         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              812.62       193.16     1_005.79       0.3916          1.1267            1.1189         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              812.62       341.22     1_153.85       0.5390          1.0704            1.0622         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              812.62       198.36     1_010.99       0.3911          1.1270            1.1192         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              812.62       340.82     1_153.44       0.5379          1.0707            1.0623         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              812.62       197.50     1_010.13       0.3909          1.1270            1.1192         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              812.62       350.26     1_162.89       0.5377          1.0708            1.0624         1.84
IVF-Binary-256-nl316-sign (self)                         812.62       534.77     1_347.39       0.4104          1.1342            1.1317         1.84
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
Exhaustive (query)                                        68.90     1_284.32     1_353.22       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.90     4_263.45     4_332.35       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                134.69       263.28       397.97       0.0733          1.4884            1.4968         2.03
ExhaustiveBinary-256-random-rf10 (query)                 134.69       384.27       518.96       0.2947          1.1327            1.1260         2.03
ExhaustiveBinary-256-random-rf20 (query)                 134.69       501.17       635.86       0.4174          1.0830            1.0716         2.03
ExhaustiveBinary-256-random (self)                       134.69     1_192.00     1_326.69       0.3150          1.1321            1.1268         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   218.52       261.76       480.27       0.0722          1.4941            1.5001         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    218.52       381.51       600.03       0.2934          1.1324            1.1267         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    218.52       511.64       730.16       0.4181          1.0813            1.0715         2.03
ExhaustiveBinary-256-pca (self)                          218.52     1_201.09     1_419.61       0.3112          1.1338            1.1298         2.03
ExhaustiveBinary-512-random_no_rr (query)                209.03       374.06       583.10       0.1110          1.4064            1.4153         4.05
ExhaustiveBinary-512-random-rf10 (query)                 209.03       510.23       719.26       0.3695          1.0935            1.0919         4.05
ExhaustiveBinary-512-random-rf20 (query)                 209.03       637.41       846.45       0.4981          1.0544            1.0517         4.05
ExhaustiveBinary-512-random (self)                       209.03     1_607.49     1_816.52       0.3856          1.0953            1.0998         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   292.76       382.39       675.15       0.1063          1.4156            1.4272         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    292.76       507.82       800.58       0.3574          1.0990            1.0958         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    292.76       644.92       937.68       0.4853          1.0582            1.0544         4.05
ExhaustiveBinary-512-pca (self)                          292.76     1_620.23     1_912.98       0.3753          1.0995            1.1040         4.05
ExhaustiveBinary-1024-random_no_rr (query)               255.20       571.20       826.40       0.1593          1.3242            1.3318         8.11
ExhaustiveBinary-1024-random-rf10 (query)                255.20       711.32       966.52       0.4456          1.0660            1.0678         8.11
ExhaustiveBinary-1024-random-rf20 (query)                255.20       846.57     1_101.77       0.5824          1.0370            1.0362         8.11
ExhaustiveBinary-1024-random (self)                      255.20     2_311.92     2_567.12       0.4595          1.0714            1.0740         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  344.60       591.02       935.62       0.1599          1.3236            1.3333         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   344.60       706.80     1_051.41       0.4446          1.0658            1.0680         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   344.60       851.38     1_195.98       0.5812          1.0370            1.0362         8.11
ExhaustiveBinary-1024-pca (self)                         344.60     2_301.41     2_646.02       0.4582          1.0716            1.0746         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   85.32       729.74       815.07       0.1292          1.3815            1.3877         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    85.32       759.58       844.90       0.3927          1.0844            1.0833         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    85.32     1_100.91     1_186.23       0.5336          1.0464            1.0444         3.05
ExhaustiveBinary-512-sign (self)                          85.32     2_348.34     2_433.67       0.4063          1.0885            1.0916         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)            1_622.96        74.02     1_696.98       0.0762          1.4798            1.4941         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           1_622.96        78.87     1_701.83       0.0762          1.4800            1.4942         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           1_622.96        79.32     1_702.28       0.0762          1.4800            1.4942         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           1_622.96       150.39     1_773.35       0.2982          1.1315            1.1256         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           1_622.96       237.67     1_860.63       0.4198          1.0821            1.0715         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          1_622.96       148.36     1_771.32       0.2974          1.1316            1.1256         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          1_622.96       236.02     1_858.98       0.4191          1.0822            1.0715         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          1_622.96       148.38     1_771.34       0.2974          1.1316            1.1256         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          1_622.96       243.32     1_866.28       0.4191          1.0822            1.0715         2.34
IVF-Binary-256-nl158-random (self)                     1_622.96       295.25     1_918.21       0.3173          1.1313            1.1266         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           1_048.58        71.63     1_120.21       0.0883          1.4512            1.4575         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)           1_048.58        73.47     1_122.05       0.0883          1.4514            1.4576         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)           1_048.58        75.96     1_124.53       0.0883          1.4514            1.4576         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)          1_048.58       158.60     1_207.17       0.3261          1.1150            1.1088         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)          1_048.58       247.87     1_296.45       0.4506          1.0704            1.0628         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)          1_048.58       153.61     1_202.18       0.3260          1.1151            1.1088         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)          1_048.58       253.46     1_302.04       0.4505          1.0704            1.0629         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)          1_048.58       158.65     1_207.23       0.3260          1.1151            1.1088         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)          1_048.58       250.10     1_298.68       0.4505          1.0704            1.0629         2.47
IVF-Binary-256-nl223-random (self)                     1_048.58       313.88     1_362.45       0.3442          1.1147            1.1139         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)           1_493.65        77.07     1_570.72       0.0951          1.4360            1.4383         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           1_493.65        78.50     1_572.15       0.0951          1.4361            1.4384         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           1_493.65        81.07     1_574.72       0.0951          1.4361            1.4384         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          1_493.65       163.33     1_656.98       0.3374          1.1080            1.1028         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          1_493.65       254.32     1_747.97       0.4655          1.0650            1.0593         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          1_493.65       165.49     1_659.14       0.3373          1.1080            1.1028         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          1_493.65       253.12     1_746.77       0.4654          1.0650            1.0593         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          1_493.65       164.69     1_658.34       0.3373          1.1080            1.1028         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          1_493.65       257.18     1_750.83       0.4654          1.0650            1.0593         2.65
IVF-Binary-256-nl316-random (self)                     1_493.65       336.14     1_829.79       0.3557          1.1069            1.1093         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               1_683.80        64.66     1_748.46       0.0751          1.4851            1.4978         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              1_683.80        68.27     1_752.07       0.0750          1.4853            1.4978         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              1_683.80        67.94     1_751.74       0.0750          1.4853            1.4979         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              1_683.80       149.19     1_832.99       0.2973          1.1312            1.1262         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              1_683.80       243.95     1_927.75       0.4206          1.0808            1.0713         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             1_683.80       146.66     1_830.45       0.2966          1.1313            1.1262         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             1_683.80       239.32     1_923.11       0.4199          1.0809            1.0713         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             1_683.80       147.07     1_830.87       0.2965          1.1313            1.1262         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             1_683.80       241.27     1_925.07       0.4199          1.0809            1.0713         2.34
IVF-Binary-256-nl158-pca (self)                        1_683.80       297.07     1_980.87       0.3141          1.1326            1.1295         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              1_145.34        74.81     1_220.15       0.0880          1.4534            1.4573         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              1_145.34        72.50     1_217.85       0.0879          1.4535            1.4574         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              1_145.34        77.50     1_222.85       0.0879          1.4535            1.4574         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             1_145.34       159.58     1_304.92       0.3296          1.1118            1.1063         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             1_145.34       252.29     1_397.63       0.4517          1.0691            1.0623         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             1_145.34       156.88     1_302.23       0.3295          1.1118            1.1063         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             1_145.34       249.81     1_395.15       0.4517          1.0691            1.0623         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             1_145.34       169.44     1_314.78       0.3295          1.1118            1.1063         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             1_145.34       251.21     1_396.55       0.4516          1.0691            1.0623         2.47
IVF-Binary-256-nl223-pca (self)                        1_145.34       316.73     1_462.08       0.3479          1.1103            1.1126         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              1_566.21        80.43     1_646.63       0.0950          1.4352            1.4379         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              1_566.21        77.35     1_643.56       0.0950          1.4353            1.4380         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              1_566.21        82.09     1_648.30       0.0950          1.4353            1.4380         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             1_566.21       164.49     1_730.70       0.3407          1.1054            1.0999         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             1_566.21       251.99     1_818.20       0.4613          1.0660            1.0596         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             1_566.21       161.46     1_727.67       0.3406          1.1054            1.0999         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             1_566.21       256.44     1_822.65       0.4612          1.0660            1.0596         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             1_566.21       164.29     1_730.49       0.3406          1.1054            1.0999         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             1_566.21       257.34     1_823.55       0.4612          1.0660            1.0596         2.65
IVF-Binary-256-nl316-pca (self)                        1_566.21       336.43     1_902.64       0.3588          1.1032            1.1073         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)            1_662.62        95.95     1_758.58       0.1123          1.4042            1.4145         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)           1_662.62        93.79     1_756.41       0.1123          1.4043            1.4145         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)           1_662.62        95.45     1_758.07       0.1123          1.4043            1.4145         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)           1_662.62       179.15     1_841.78       0.3703          1.0933            1.0918         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)           1_662.62       272.48     1_935.10       0.4988          1.0542            1.0516         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)          1_662.62       180.80     1_843.42       0.3701          1.0933            1.0918         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)          1_662.62       275.67     1_938.30       0.4986          1.0543            1.0516         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)          1_662.62       183.05     1_845.68       0.3701          1.0933            1.0918         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)          1_662.62       278.69     1_941.31       0.4986          1.0543            1.0516         4.36
IVF-Binary-512-nl158-random (self)                     1_662.62       411.32     2_073.95       0.3862          1.0951            1.0998         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)           1_125.94        97.42     1_223.37       0.1204          1.3874            1.3946         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)           1_125.94       100.04     1_225.98       0.1204          1.3875            1.3946         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)           1_125.94       105.57     1_231.51       0.1204          1.3875            1.3946         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)          1_125.94       186.65     1_312.60       0.3836          1.0877            1.0871         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)          1_125.94       281.08     1_407.02       0.5122          1.0510            1.0488         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)          1_125.94       187.11     1_313.05       0.3836          1.0877            1.0871         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)          1_125.94       286.00     1_411.95       0.5121          1.0510            1.0488         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)          1_125.94       190.71     1_316.65       0.3836          1.0877            1.0871         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)          1_125.94       286.32     1_412.27       0.5121          1.0510            1.0488         4.49
IVF-Binary-512-nl223-random (self)                     1_125.94       436.81     1_562.76       0.3990          1.0902            1.0949         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)           1_553.70       103.39     1_657.09       0.1243          1.3791            1.3837         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)           1_553.70       105.92     1_659.62       0.1243          1.3791            1.3837         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)           1_553.70       108.35     1_662.05       0.1243          1.3791            1.3837         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)          1_553.70       195.86     1_749.56       0.3890          1.0855            1.0847         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)          1_553.70       295.10     1_848.80       0.5158          1.0501            1.0480         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)          1_553.70       189.15     1_742.86       0.3890          1.0855            1.0847         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)          1_553.70       287.68     1_841.39       0.5158          1.0501            1.0480         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)          1_553.70       195.32     1_749.03       0.3890          1.0855            1.0847         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)          1_553.70       292.62     1_846.32       0.5158          1.0501            1.0480         4.67
IVF-Binary-512-nl316-random (self)                     1_553.70       448.31     2_002.01       0.4033          1.0884            1.0931         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)               1_759.36        91.41     1_850.77       0.1076          1.4133            1.4261         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)              1_759.36        93.64     1_853.00       0.1076          1.4134            1.4261         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)              1_759.36        95.94     1_855.30       0.1076          1.4134            1.4261         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)              1_759.36       179.86     1_939.22       0.3583          1.0988            1.0957         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)              1_759.36       275.61     2_034.97       0.4858          1.0581            1.0544         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)             1_759.36       177.77     1_937.13       0.3581          1.0988            1.0957         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)             1_759.36       280.71     2_040.07       0.4857          1.0581            1.0544         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)             1_759.36       182.65     1_942.01       0.3581          1.0988            1.0957         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)             1_759.36       277.05     2_036.41       0.4857          1.0581            1.0544         4.36
IVF-Binary-512-nl158-pca (self)                        1_759.36       414.50     2_173.86       0.3759          1.0993            1.1040         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_262.00       100.77     1_362.76       0.1168          1.3940            1.3999         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_262.00       101.46     1_363.46       0.1168          1.3940            1.4000         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_262.00       104.86     1_366.86       0.1168          1.3940            1.4000         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_262.00       190.15     1_452.15       0.3738          1.0918            1.0901         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_262.00       278.07     1_540.07       0.5017          1.0541            1.0515         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_262.00       194.91     1_456.90       0.3738          1.0918            1.0901         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_262.00       283.40     1_545.40       0.5017          1.0541            1.0515         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_262.00       189.27     1_451.27       0.3738          1.0918            1.0901         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_262.00       287.56     1_549.56       0.5017          1.0541            1.0515         4.49
IVF-Binary-512-nl223-pca (self)                        1_262.00       425.97     1_687.97       0.3906          1.0936            1.0983         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)              1_656.12       103.54     1_759.65       0.1210          1.3851            1.3902         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)              1_656.12       104.84     1_760.95       0.1210          1.3851            1.3902         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)              1_656.12       108.65     1_764.76       0.1210          1.3851            1.3902         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)             1_656.12       195.49     1_851.61       0.3801          1.0890            1.0877         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)             1_656.12       283.69     1_939.80       0.5059          1.0528            1.0503         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)             1_656.12       192.36     1_848.47       0.3800          1.0890            1.0877         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)             1_656.12       286.37     1_942.49       0.5059          1.0528            1.0503         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)             1_656.12       194.48     1_850.60       0.3800          1.0890            1.0877         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)             1_656.12       292.20     1_948.32       0.5059          1.0528            1.0503         4.67
IVF-Binary-512-nl316-pca (self)                        1_656.12       444.85     2_100.97       0.3956          1.0912            1.0960         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)           1_719.30       145.76     1_865.06       0.1598          1.3235            1.3316         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)          1_719.30       146.99     1_866.29       0.1598          1.3235            1.3316         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)          1_719.30       147.18     1_866.48       0.1598          1.3235            1.3316         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)          1_719.30       234.30     1_953.60       0.4458          1.0660            1.0678         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)          1_719.30       343.50     2_062.80       0.5826          1.0370            1.0362         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)         1_719.30       234.19     1_953.49       0.4458          1.0660            1.0678         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)         1_719.30       337.57     2_056.87       0.5826          1.0370            1.0362         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)         1_719.30       238.68     1_957.98       0.4458          1.0660            1.0678         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)         1_719.30       349.65     2_068.95       0.5826          1.0370            1.0362         8.42
IVF-Binary-1024-nl158-random (self)                    1_719.30       605.44     2_324.74       0.4597          1.0713            1.0740         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_161.16       147.05     1_308.21       0.1637          1.3164            1.3251         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_161.16       153.49     1_314.65       0.1637          1.3164            1.3251         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_161.16       157.40     1_318.56       0.1637          1.3164            1.3251         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_161.16       244.39     1_405.55       0.4532          1.0639            1.0658         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_161.16       344.25     1_505.41       0.5895          1.0358            1.0351         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_161.16       242.41     1_403.56       0.4532          1.0639            1.0658         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_161.16       348.52     1_509.67       0.5895          1.0358            1.0351         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_161.16       251.37     1_412.53       0.4532          1.0639            1.0658         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_161.16       361.62     1_522.77       0.5895          1.0358            1.0351         8.54
IVF-Binary-1024-nl223-random (self)                    1_161.16       629.53     1_790.68       0.4672          1.0692            1.0721         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)          1_614.67       158.57     1_773.24       0.1655          1.3134            1.3213         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)          1_614.67       156.16     1_770.82       0.1655          1.3134            1.3213         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)          1_614.67       159.51     1_774.18       0.1655          1.3134            1.3213         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)         1_614.67       249.38     1_864.05       0.4550          1.0633            1.0652         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)         1_614.67       353.54     1_968.21       0.5909          1.0355            1.0347         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)         1_614.67       247.61     1_862.28       0.4550          1.0633            1.0652         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)         1_614.67       355.17     1_969.84       0.5908          1.0355            1.0347         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)         1_614.67       262.40     1_877.07       0.4550          1.0633            1.0652         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)         1_614.67       367.37     1_982.04       0.5908          1.0355            1.0347         8.73
IVF-Binary-1024-nl316-random (self)                    1_614.67       645.16     2_259.83       0.4696          1.0685            1.0715         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)              1_804.41       143.20     1_947.61       0.1603          1.3231            1.3332         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)             1_804.41       145.06     1_949.47       0.1603          1.3231            1.3332         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)             1_804.41       146.38     1_950.79       0.1603          1.3231            1.3332         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)             1_804.41       237.15     2_041.57       0.4447          1.0658            1.0680         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)             1_804.41       337.51     2_141.93       0.5813          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)            1_804.41       244.16     2_048.57       0.4447          1.0658            1.0680         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)            1_804.41       340.69     2_145.11       0.5813          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)            1_804.41       253.88     2_058.29       0.4447          1.0658            1.0680         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)            1_804.41       347.28     2_151.70       0.5813          1.0369            1.0362         8.42
IVF-Binary-1024-nl158-pca (self)                       1_804.41       606.27     2_410.69       0.4583          1.0716            1.0746         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_255.29       151.37     1_406.65       0.1643          1.3158            1.3263         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_255.29       148.60     1_403.88       0.1643          1.3159            1.3263         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_255.29       157.96     1_413.25       0.1643          1.3159            1.3263         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_255.29       246.29     1_501.58       0.4518          1.0637            1.0663         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_255.29       360.31     1_615.59       0.5877          1.0359            1.0354         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_255.29       246.82     1_502.10       0.4518          1.0637            1.0663         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_255.29       349.90     1_605.19       0.5877          1.0359            1.0354         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_255.29       251.06     1_506.35       0.4518          1.0637            1.0663         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_255.29       360.33     1_615.62       0.5877          1.0359            1.0354         8.54
IVF-Binary-1024-nl223-pca (self)                       1_255.29       623.40     1_878.69       0.4656          1.0695            1.0726         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_703.48       154.59     1_858.08       0.1657          1.3130            1.3236         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_703.48       155.48     1_858.96       0.1657          1.3131            1.3236         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_703.48       160.28     1_863.77       0.1657          1.3131            1.3236         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_703.48       252.25     1_955.73       0.4540          1.0631            1.0656         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_703.48       351.73     2_055.21       0.5897          1.0356            1.0351         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_703.48       250.50     1_953.98       0.4540          1.0631            1.0656         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_703.48       365.08     2_068.57       0.5897          1.0356            1.0351         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_703.48       256.45     1_959.93       0.4539          1.0631            1.0656         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_703.48       365.13     2_068.61       0.5897          1.0356            1.0351         8.73
IVF-Binary-1024-nl316-pca (self)                       1_703.48       644.50     2_347.98       0.4678          1.0688            1.0718         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)              1_565.18       281.29     1_846.46       0.1291          1.3814            1.3871         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)             1_565.18       286.10     1_851.28       0.1291          1.3814            1.3871         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)             1_565.18       286.61     1_851.79       0.1291          1.3814            1.3871         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)             1_565.18       380.43     1_945.61       0.3933          1.0843            1.0833         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)             1_565.18       630.32     2_195.50       0.5337          1.0464            1.0445         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)            1_565.18       352.13     1_917.31       0.3931          1.0843            1.0833         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)            1_565.18       647.03     2_212.20       0.5337          1.0464            1.0445         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)            1_565.18       358.69     1_923.86       0.3931          1.0843            1.0833         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)            1_565.18       649.95     2_215.12       0.5337          1.0464            1.0445         3.36
IVF-Binary-512-nl158-sign (self)                       1_565.18       978.84     2_544.01       0.4066          1.0885            1.0915         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)             1_005.23       287.29     1_292.51       0.1295          1.3811            1.3869         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)             1_005.23       295.97     1_301.20       0.1295          1.3812            1.3869         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)             1_005.23       303.34     1_308.57       0.1295          1.3812            1.3869         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)            1_005.23       373.88     1_379.11       0.3989          1.0819            1.0819         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)            1_005.23       652.93     1_658.16       0.5380          1.0455            1.0438         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)            1_005.23       357.24     1_362.46       0.3988          1.0819            1.0819         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)            1_005.23       647.12     1_652.35       0.5379          1.0455            1.0438         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)            1_005.23       364.21     1_369.44       0.3988          1.0819            1.0819         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)            1_005.23       654.12     1_659.35       0.5379          1.0455            1.0438         3.49
IVF-Binary-512-nl223-sign (self)                       1_005.23       994.83     2_000.06       0.4123          1.0865            1.0899         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)             1_444.88       297.35     1_742.23       0.1294          1.3809            1.3865         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)             1_444.88       295.56     1_740.44       0.1294          1.3809            1.3865         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)             1_444.88       307.78     1_752.66       0.1294          1.3809            1.3865         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)            1_444.88       364.41     1_809.29       0.4003          1.0814            1.0814         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)            1_444.88       654.12     2_099.00       0.5383          1.0453            1.0439         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)            1_444.88       388.46     1_833.34       0.4003          1.0814            1.0814         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)            1_444.88       651.41     2_096.29       0.5382          1.0453            1.0439         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)            1_444.88       369.22     1_814.10       0.4003          1.0814            1.0814         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)            1_444.88       655.00     2_099.88       0.5382          1.0453            1.0439         3.67
IVF-Binary-512-nl316-sign (self)                       1_444.88     1_013.25     2_458.13       0.4134          1.0860            1.0893         3.67
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
Exhaustive (query)                                       101.45     1_831.46     1_932.91       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.45     6_303.80     6_405.25       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                198.67       284.93       483.61       0.0662          1.3769            1.3786         2.28
ExhaustiveBinary-256-random-rf10 (query)                 198.67       427.03       625.70       0.2741          1.1112            1.1017         2.28
ExhaustiveBinary-256-random-rf20 (query)                 198.67       564.76       763.43       0.3877          1.0706            1.0590         2.28
ExhaustiveBinary-256-random (self)                       198.67     1_273.60     1_472.27       0.2874          1.1077            1.0994         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   397.41       279.55       676.96       0.0653          1.3793            1.3770         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    397.41       419.59       817.00       0.2701          1.1138            1.1018         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    397.41       557.29       954.70       0.3852          1.0726            1.0585         2.28
ExhaustiveBinary-256-pca (self)                          397.41     1_285.70     1_683.11       0.2820          1.1100            1.0990         2.28
ExhaustiveBinary-512-random_no_rr (query)                299.30       398.11       697.42       0.0934          1.3249            1.3316         4.55
ExhaustiveBinary-512-random-rf10 (query)                 299.30       547.65       846.95       0.3220          1.0860            1.0806         4.55
ExhaustiveBinary-512-random-rf20 (query)                 299.30       700.05       999.35       0.4345          1.0527            1.0485         4.55
ExhaustiveBinary-512-random (self)                       299.30     1_728.86     2_028.17       0.3344          1.0826            1.0842         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   495.06       405.47       900.53       0.0951          1.3226            1.3263         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    495.06       553.64     1_048.71       0.3246          1.0847            1.0788         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    495.06       758.09     1_253.15       0.4400          1.0517            1.0468         4.55
ExhaustiveBinary-512-pca (self)                          495.06     1_744.13     2_239.19       0.3361          1.0819            1.0825         4.55
ExhaustiveBinary-1024-random_no_rr (query)               494.38       605.25     1_099.62       0.1319          1.2685            1.2723         9.11
ExhaustiveBinary-1024-random-rf10 (query)                494.38       775.88     1_270.26       0.3742          1.0644            1.0666         9.11
ExhaustiveBinary-1024-random-rf20 (query)                494.38       950.72     1_445.09       0.4906          1.0386            1.0390         9.11
ExhaustiveBinary-1024-random (self)                      494.38     2_514.46     3_008.84       0.3823          1.0666            1.0715         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  691.72       609.58     1_301.30       0.1355          1.2623            1.2651         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   691.72       781.08     1_472.80       0.3804          1.0622            1.0643         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   691.72       947.94     1_639.67       0.4993          1.0369            1.0374         9.11
ExhaustiveBinary-1024-pca (self)                         691.72     2_545.45     3_237.17       0.3870          1.0651            1.0695         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  128.81       843.52       972.34       0.1284          1.2822            1.2821         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   128.81       929.47     1_058.28       0.3618          1.0706            1.0699         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   128.81     1_437.80     1_566.61       0.4847          1.0407            1.0395         4.58
ExhaustiveBinary-768-sign (self)                         128.81     2_973.41     3_102.23       0.3694          1.0716            1.0747         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)            2_305.60        94.98     2_400.58       0.0691          1.3703            1.3767         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)           2_305.60        98.42     2_404.02       0.0690          1.3706            1.3768         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)           2_305.60       100.49     2_406.09       0.0690          1.3706            1.3768         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)           2_305.60       198.30     2_503.90       0.2788          1.1100            1.1013         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)           2_305.60       298.73     2_604.33       0.3911          1.0698            1.0589         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)          2_305.60       191.30     2_496.90       0.2780          1.1101            1.1013         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)          2_305.60       298.05     2_603.65       0.3904          1.0699            1.0589         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)          2_305.60       187.73     2_493.33       0.2780          1.1101            1.1013         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)          2_305.60       302.43     2_608.03       0.3903          1.0699            1.0589         2.74
IVF-Binary-256-nl158-random (self)                     2_305.60       372.55     2_678.15       0.2914          1.1066            1.0992         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)           1_411.05        93.05     1_504.10       0.0782          1.3506            1.3552         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)           1_411.05        92.94     1_503.99       0.0782          1.3507            1.3552         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)           1_411.05        95.63     1_506.68       0.0782          1.3507            1.3552         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)          1_411.05       201.09     1_612.14       0.3036          1.0944            1.0863         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)          1_411.05       316.31     1_727.35       0.4176          1.0588            1.0517         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)          1_411.05       199.72     1_610.77       0.3036          1.0944            1.0863         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)          1_411.05       315.46     1_726.51       0.4175          1.0588            1.0517         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)          1_411.05       205.75     1_616.79       0.3036          1.0944            1.0863         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)          1_411.05       322.76     1_733.81       0.4175          1.0588            1.0517         2.93
IVF-Binary-256-nl223-random (self)                     1_411.05       412.75     1_823.80       0.3175          1.0893            1.0880         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)           2_100.00       102.03     2_202.03       0.0857          1.3373            1.3374         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)           2_100.00       101.34     2_201.34       0.0856          1.3375            1.3374         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)           2_100.00       103.43     2_203.43       0.0856          1.3375            1.3374         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)          2_100.00       210.21     2_310.20       0.3192          1.0859            1.0802         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)          2_100.00       319.85     2_419.85       0.4327          1.0540            1.0488         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)          2_100.00       207.39     2_307.39       0.3191          1.0859            1.0802         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)          2_100.00       323.48     2_423.48       0.4326          1.0540            1.0488         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)          2_100.00       213.49     2_313.49       0.3191          1.0859            1.0802         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)          2_100.00       326.42     2_426.42       0.4326          1.0540            1.0488         3.21
IVF-Binary-256-nl316-random (self)                     2_100.00       456.44     2_556.44       0.3327          1.0803            1.0828         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)               2_489.42        82.53     2_571.94       0.0681          1.3721            1.3752         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)              2_489.42        86.28     2_575.70       0.0681          1.3723            1.3752         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)              2_489.42        86.37     2_575.79       0.0681          1.3723            1.3752         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)              2_489.42       190.50     2_679.92       0.2754          1.1116            1.1012         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)              2_489.42       296.53     2_785.95       0.3894          1.0711            1.0583         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)             2_489.42       186.56     2_675.97       0.2745          1.1117            1.1012         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)             2_489.42       307.63     2_797.05       0.3886          1.0712            1.0583         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)             2_489.42       186.73     2_676.14       0.2745          1.1117            1.1012         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)             2_489.42       313.18     2_802.60       0.3885          1.0712            1.0583         2.74
IVF-Binary-256-nl158-pca (self)                        2_489.42       371.33     2_860.75       0.2864          1.1079            1.0986         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)              1_602.71        90.65     1_693.36       0.0775          1.3528            1.3548         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)              1_602.71        92.36     1_695.07       0.0775          1.3529            1.3548         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)              1_602.71        94.19     1_696.91       0.0775          1.3529            1.3548         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)             1_602.71       202.95     1_805.67       0.2987          1.0962            1.0873         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)             1_602.71       311.63     1_914.35       0.4131          1.0605            1.0523         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)             1_602.71       205.62     1_808.34       0.2986          1.0962            1.0873         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)             1_602.71       321.38     1_924.09       0.4130          1.0605            1.0523         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)             1_602.71       199.18     1_801.89       0.2986          1.0962            1.0873         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)             1_602.71       313.32     1_916.03       0.4129          1.0605            1.0523         2.93
IVF-Binary-256-nl223-pca (self)                        1_602.71       414.23     2_016.95       0.3109          1.0909            1.0885         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)              2_287.40        99.67     2_387.08       0.0850          1.3392            1.3373         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)              2_287.40       100.98     2_388.38       0.0850          1.3393            1.3374         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)              2_287.40       109.33     2_396.73       0.0850          1.3393            1.3374         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)             2_287.40       210.82     2_498.22       0.3123          1.0891            1.0816         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)             2_287.40       325.15     2_612.55       0.4271          1.0565            1.0494         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)             2_287.40       204.62     2_492.02       0.3122          1.0892            1.0816         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)             2_287.40       323.23     2_610.63       0.4270          1.0565            1.0494         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)             2_287.40       211.11     2_498.51       0.3122          1.0892            1.0816         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)             2_287.40       326.66     2_614.06       0.4270          1.0565            1.0494         3.21
IVF-Binary-256-nl316-pca (self)                        2_287.40       458.59     2_746.00       0.3237          1.0835            1.0840         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)            2_392.32       126.46     2_518.78       0.0948          1.3226            1.3308         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)           2_392.32       126.46     2_518.78       0.0948          1.3227            1.3308         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)           2_392.32       123.32     2_515.65       0.0948          1.3227            1.3308         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)           2_392.32       239.06     2_631.39       0.3238          1.0856            1.0806         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)           2_392.32       341.91     2_734.23       0.4355          1.0525            1.0484         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)          2_392.32       229.11     2_621.43       0.3235          1.0857            1.0806         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)          2_392.32       347.02     2_739.34       0.4355          1.0525            1.0484         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)          2_392.32       232.58     2_624.91       0.3235          1.0857            1.0806         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)          2_392.32       360.79     2_753.12       0.4355          1.0525            1.0484         5.02
IVF-Binary-512-nl158-random (self)                     2_392.32       538.68     2_931.00       0.3360          1.0822            1.0842         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)           1_532.89       126.83     1_659.71       0.1037          1.3076            1.3082         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)           1_532.89       129.46     1_662.35       0.1037          1.3076            1.3082         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)           1_532.89       134.42     1_667.31       0.1037          1.3076            1.3082         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)          1_532.89       238.96     1_771.84       0.3364          1.0790            1.0764         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)          1_532.89       359.60     1_892.49       0.4477          1.0488            1.0461         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)          1_532.89       240.25     1_773.14       0.3364          1.0790            1.0764         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)          1_532.89       369.51     1_902.40       0.4477          1.0488            1.0461         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)          1_532.89       242.85     1_775.74       0.3364          1.0790            1.0764         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)          1_532.89       366.90     1_899.79       0.4477          1.0488            1.0461         5.21
IVF-Binary-512-nl223-random (self)                     1_532.89       567.16     2_100.05       0.3478          1.0765            1.0802         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)           2_183.04       141.78     2_324.82       0.1080          1.2996            1.2983         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)           2_183.04       136.47     2_319.51       0.1080          1.2996            1.2983         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)           2_183.04       140.49     2_323.52       0.1080          1.2996            1.2983         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)          2_183.04       251.14     2_434.17       0.3430          1.0762            1.0744         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)          2_183.04       370.82     2_553.86       0.4551          1.0469            1.0448         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)          2_183.04       256.93     2_439.97       0.3430          1.0762            1.0744         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)          2_183.04       372.58     2_555.61       0.4550          1.0469            1.0448         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)          2_183.04       250.10     2_433.14       0.3430          1.0762            1.0744         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)          2_183.04       376.30     2_559.34       0.4550          1.0469            1.0448         5.48
IVF-Binary-512-nl316-random (self)                     2_183.04       602.48     2_785.52       0.3535          1.0741            1.0784         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)               2_588.85       117.94     2_706.79       0.0963          1.3201            1.3252         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)              2_588.85       121.00     2_709.85       0.0962          1.3202            1.3252         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)              2_588.85       125.38     2_714.23       0.0962          1.3202            1.3252         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)              2_588.85       231.66     2_820.51       0.3267          1.0842            1.0787         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)              2_588.85       343.77     2_932.62       0.4414          1.0514            1.0467         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)             2_588.85       227.35     2_816.19       0.3264          1.0842            1.0787         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)             2_588.85       352.78     2_941.62       0.4413          1.0514            1.0467         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)             2_588.85       234.91     2_823.76       0.3264          1.0842            1.0787         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)             2_588.85       354.02     2_942.86       0.4413          1.0514            1.0467         5.02
IVF-Binary-512-nl158-pca (self)                        2_588.85       559.55     3_148.39       0.3378          1.0812            1.0824         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_775.49       129.30     1_904.79       0.1044          1.3060            1.3056         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_775.49       128.22     1_903.71       0.1044          1.3060            1.3056         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_775.49       132.66     1_908.15       0.1044          1.3060            1.3056         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_775.49       247.00     2_022.49       0.3374          1.0788            1.0750         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_775.49       356.32     2_131.81       0.4529          1.0478            1.0448         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_775.49       236.62     2_012.12       0.3374          1.0789            1.0750         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_775.49       357.07     2_132.57       0.4529          1.0478            1.0448         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_775.49       253.80     2_029.29       0.3374          1.0789            1.0750         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_775.49       365.32     2_140.81       0.4529          1.0478            1.0448         5.21
IVF-Binary-512-nl223-pca (self)                        1_775.49       593.69     2_369.18       0.3475          1.0770            1.0795         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)              2_359.46       141.20     2_500.66       0.1079          1.2993            1.2969         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)              2_359.46       137.72     2_497.19       0.1079          1.2993            1.2969         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)              2_359.46       141.22     2_500.68       0.1079          1.2993            1.2969         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)             2_359.46       248.13     2_607.60       0.3441          1.0758            1.0730         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)             2_359.46       368.66     2_728.13       0.4591          1.0464            1.0438         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)             2_359.46       249.66     2_609.12       0.3441          1.0758            1.0730         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)             2_359.46       376.94     2_736.41       0.4591          1.0464            1.0438         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)             2_359.46       252.30     2_611.76       0.3441          1.0758            1.0730         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)             2_359.46       381.35     2_740.81       0.4591          1.0464            1.0438         5.48
IVF-Binary-512-nl316-pca (self)                        2_359.46       602.53     2_961.99       0.3536          1.0746            1.0779         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)           2_578.89       189.87     2_768.76       0.1327          1.2680            1.2723         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)          2_578.89       191.89     2_770.79       0.1327          1.2680            1.2723         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)          2_578.89       201.57     2_780.46       0.1327          1.2680            1.2723         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)          2_578.89       315.37     2_894.26       0.3746          1.0643            1.0666         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)          2_578.89       440.76     3_019.65       0.4907          1.0386            1.0390         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)         2_578.89       319.57     2_898.46       0.3745          1.0643            1.0666         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)         2_578.89       447.54     3_026.43       0.4907          1.0386            1.0390         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)         2_578.89       327.44     2_906.33       0.3745          1.0643            1.0666         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)         2_578.89       459.81     3_038.70       0.4907          1.0386            1.0390         9.57
IVF-Binary-1024-nl158-random (self)                    2_578.89       832.02     3_410.91       0.3826          1.0665            1.0715         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_717.61       199.28     1_916.89       0.1368          1.2612            1.2660         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_717.61       204.72     1_922.33       0.1368          1.2612            1.2660         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_717.61       205.08     1_922.70       0.1368          1.2612            1.2660         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_717.61       328.56     2_046.17       0.3802          1.0625            1.0652         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_717.61       453.37     2_170.98       0.4970          1.0375            1.0378         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_717.61       331.97     2_049.59       0.3802          1.0625            1.0652         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_717.61       487.82     2_205.43       0.4970          1.0375            1.0378         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_717.61       341.77     2_059.38       0.3802          1.0625            1.0652         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_717.61       470.97     2_188.58       0.4970          1.0375            1.0378         9.76
IVF-Binary-1024-nl223-random (self)                    1_717.61       849.51     2_567.12       0.3881          1.0649            1.0698         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)          2_388.09       209.90     2_597.99       0.1389          1.2580            1.2624        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)          2_388.09       211.48     2_599.57       0.1389          1.2580            1.2624        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)          2_388.09       218.78     2_606.86       0.1389          1.2580            1.2624        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)         2_388.09       348.33     2_736.42       0.3846          1.0612            1.0639        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)         2_388.09       480.77     2_868.86       0.5016          1.0366            1.0373        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)         2_388.09       344.37     2_732.46       0.3846          1.0612            1.0639        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)         2_388.09       479.35     2_867.44       0.5016          1.0366            1.0373        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)         2_388.09       355.61     2_743.70       0.3846          1.0612            1.0639        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)         2_388.09       483.11     2_871.19       0.5016          1.0366            1.0373        10.04
IVF-Binary-1024-nl316-random (self)                    2_388.09       903.16     3_291.25       0.3917          1.0638            1.0688        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)              2_807.06       191.82     2_998.88       0.1360          1.2617            1.2650         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)             2_807.06       193.18     3_000.24       0.1360          1.2617            1.2650         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)             2_807.06       195.98     3_003.04       0.1360          1.2617            1.2650         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)             2_807.06       315.86     3_122.92       0.3810          1.0621            1.0643         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)             2_807.06       441.28     3_248.34       0.4996          1.0369            1.0374         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)            2_807.06       323.60     3_130.66       0.3809          1.0621            1.0643         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)            2_807.06       447.49     3_254.56       0.4996          1.0369            1.0374         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)            2_807.06       327.13     3_134.19       0.3809          1.0621            1.0643         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)            2_807.06       455.12     3_262.18       0.4996          1.0369            1.0374         9.57
IVF-Binary-1024-nl158-pca (self)                       2_807.06       839.47     3_646.53       0.3875          1.0650            1.0695         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_952.89       202.33     2_155.22       0.1395          1.2561            1.2604         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_952.89       200.33     2_153.22       0.1395          1.2561            1.2604         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_952.89       205.06     2_157.95       0.1395          1.2561            1.2604         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_952.89       333.52     2_286.41       0.3861          1.0602            1.0631         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_952.89       463.03     2_415.92       0.5049          1.0360            1.0365         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_952.89       330.24     2_283.13       0.3861          1.0602            1.0631         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_952.89       461.49     2_414.38       0.5049          1.0360            1.0365         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_952.89       341.89     2_294.78       0.3861          1.0602            1.0631         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_952.89       472.59     2_425.48       0.5049          1.0360            1.0365         9.76
IVF-Binary-1024-nl223-pca (self)                       1_952.89       885.88     2_838.77       0.3927          1.0635            1.0680         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             2_584.50       213.78     2_798.28       0.1415          1.2527            1.2573        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)             2_584.50       210.85     2_795.35       0.1415          1.2527            1.2573        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)             2_584.50       219.24     2_803.74       0.1415          1.2527            1.2573        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)            2_584.50       349.90     2_934.40       0.3894          1.0595            1.0624        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)            2_584.50       469.49     3_053.99       0.5085          1.0354            1.0362        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)            2_584.50       346.15     2_930.65       0.3894          1.0595            1.0624        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)            2_584.50       473.08     3_057.58       0.5085          1.0354            1.0362        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)            2_584.50       353.98     2_938.49       0.3894          1.0595            1.0624        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)            2_584.50       480.23     3_064.73       0.5085          1.0354            1.0362        10.04
IVF-Binary-1024-nl316-pca (self)                       2_584.50       909.92     3_494.42       0.3958          1.0626            1.0672        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)              2_271.99       397.79     2_669.78       0.1284          1.2834            1.2814         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)             2_271.99       407.76     2_679.75       0.1284          1.2834            1.2814         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)             2_271.99       406.57     2_678.56       0.1284          1.2834            1.2814         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)             2_271.99       488.87     2_760.86       0.3627          1.0704            1.0698         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)             2_271.99       892.93     3_164.92       0.4854          1.0406            1.0396         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)            2_271.99       490.98     2_762.97       0.3627          1.0704            1.0698         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)            2_271.99       896.09     3_168.08       0.4854          1.0406            1.0396         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)            2_271.99       494.59     2_766.58       0.3627          1.0704            1.0698         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)            2_271.99       898.17     3_170.16       0.4854          1.0406            1.0396         5.04
IVF-Binary-768-nl158-sign (self)                       2_271.99     1_395.91     3_667.90       0.3701          1.0715            1.0746         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)             1_401.00       409.17     1_810.18       0.1282          1.2804            1.2811         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)             1_401.00       410.31     1_811.31       0.1282          1.2804            1.2811         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)             1_401.00       416.04     1_817.04       0.1282          1.2804            1.2811         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)            1_401.00       497.01     1_898.01       0.3659          1.0690            1.0690         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)            1_401.00       896.14     2_297.14       0.4872          1.0400            1.0394         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)            1_401.00       498.62     1_899.62       0.3659          1.0690            1.0690         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)            1_401.00       899.78     2_300.78       0.4872          1.0400            1.0394         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)            1_401.00       501.02     1_902.02       0.3659          1.0690            1.0690         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)            1_401.00       907.29     2_308.30       0.4872          1.0400            1.0394         5.23
IVF-Binary-768-nl223-sign (self)                       1_401.00     1_528.57     2_929.57       0.3730          1.0703            1.0736         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)             2_035.81       416.02     2_451.83       0.1287          1.2802            1.2810         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)             2_035.81       421.94     2_457.75       0.1287          1.2802            1.2810         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)             2_035.81       424.68     2_460.50       0.1287          1.2802            1.2810         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)            2_035.81       512.12     2_547.93       0.3674          1.0684            1.0685         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)            2_035.81       908.22     2_944.03       0.4890          1.0397            1.0392         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)            2_035.81       505.06     2_540.88       0.3674          1.0684            1.0685         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)            2_035.81       918.82     2_954.63       0.4890          1.0397            1.0392         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)            2_035.81       510.15     2_545.97       0.3674          1.0684            1.0685         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)            2_035.81       920.14     2_955.96       0.4890          1.0397            1.0392         5.51
IVF-Binary-768-nl316-sign (self)                       2_035.81     1_448.73     3_484.55       0.3747          1.0698            1.0734         5.51
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
Exhaustive (query)                                        32.70       722.52       755.22       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.70     2_379.52     2_412.22       1.0000          1.0000            1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)                 72.05       241.20       313.25       0.5519          1.8826            1.5884         1.78
ExhaustiveBinary-256-random-rf10 (query)                  72.05       359.44       431.49       0.9881          1.0022            1.0000         1.78
ExhaustiveBinary-256-random-rf20 (query)                  72.05       465.81       537.86       0.9980          1.0003            1.0000         1.78
ExhaustiveBinary-256-random (self)                        72.05     1_157.09     1_229.14       0.9881          1.0022            1.0000         1.78
ExhaustiveBinary-256-pca_no_rr (query)                    96.83       244.06       340.89       0.5930          1.6081            1.4152         1.78
ExhaustiveBinary-256-pca-rf10 (query)                     96.83       358.27       455.10       0.9919          1.0013            1.0000         1.78
ExhaustiveBinary-256-pca-rf20 (query)                     96.83       481.89       578.71       0.9988          1.0001            1.0000         1.78
ExhaustiveBinary-256-pca (self)                           96.83     1_182.93     1_279.76       0.9915          1.0014            1.0000         1.78
ExhaustiveBinary-512-random_no_rr (query)                 84.27       355.57       439.84       0.6306          1.5767            1.3633         3.55
ExhaustiveBinary-512-random-rf10 (query)                  84.27       470.55       554.82       0.9975          1.0004            1.0000         3.55
ExhaustiveBinary-512-random-rf20 (query)                  84.27       588.55       672.82       0.9998          1.0000            1.0000         3.55
ExhaustiveBinary-512-random (self)                        84.27     1_536.11     1_620.38       0.9973          1.0004            1.0000         3.55
ExhaustiveBinary-512-pca_no_rr (query)                   115.60       346.51       462.12       0.6479          1.4884            1.3147         3.55
ExhaustiveBinary-512-pca-rf10 (query)                    115.60       468.92       584.53       0.9983          1.0002            1.0000         3.55
ExhaustiveBinary-512-pca-rf20 (query)                    115.60       583.31       698.92       0.9998          1.0000            1.0000         3.55
ExhaustiveBinary-512-pca (self)                          115.60     1_531.46     1_647.07       0.9981          1.0002            1.0000         3.55
ExhaustiveBinary-1024-random_no_rr (query)               117.99       509.77       627.76       0.6758          1.4452            1.2804         7.10
ExhaustiveBinary-1024-random-rf10 (query)                117.99       631.65       749.64       0.9995          1.0001            1.0000         7.10
ExhaustiveBinary-1024-random-rf20 (query)                117.99       748.37       866.36       0.9999          1.0000            1.0000         7.10
ExhaustiveBinary-1024-random (self)                      117.99     2_101.87     2_219.86       0.9993          1.0001            1.0000         7.10
ExhaustiveBinary-1024-pca_no_rr (query)                  142.52       512.96       655.48       0.6838          1.4142            1.2651         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                   142.52       633.29       775.81       0.9996          1.0000            1.0000         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                   142.52       749.94       892.46       1.0000          1.0000            1.0000         7.10
ExhaustiveBinary-1024-pca (self)                         142.52     2_105.38     2_247.90       0.9995          1.0001            1.0000         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   47.58       427.37       474.95       0.0376         19.4734           14.8778         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    47.58       452.80       500.38       0.1617          2.7567            2.6548         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    47.58       697.10       744.68       0.2739          1.9837            1.9249         1.53
ExhaustiveBinary-256-sign (self)                          47.58     1_485.67     1_533.25       0.1691          2.7353            2.6299         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)              965.45        58.10     1_023.54       0.5655          1.6704            1.5137         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)             965.45        68.90     1_034.34       0.5588          1.7297            1.5496         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)             965.45        76.58     1_042.02       0.5568          1.7636            1.5627         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)             965.45       115.60     1_081.04       0.9903          1.0018            1.0000         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)             965.45       175.68     1_141.13       0.9968          1.0006            1.0000         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)            965.45       128.88     1_094.32       0.9907          1.0016            1.0000         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)            965.45       177.26     1_142.71       0.9986          1.0002            1.0000         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)            965.45       128.85     1_094.30       0.9898          1.0018            1.0000         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)            965.45       191.81     1_157.25       0.9984          1.0002            1.0000         1.93
IVF-Binary-256-nl158-random (self)                       965.45       319.36     1_284.80       0.9904          1.0017            1.0000         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)             536.83        48.94       585.77       0.5629          1.6755            1.5256         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)             536.83        52.27       589.10       0.5605          1.7012            1.5424         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)             536.83        61.50       598.33       0.5578          1.7449            1.5594         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)            536.83       107.82       644.65       0.9912          1.0015            1.0000         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)            536.83       166.18       703.01       0.9984          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)            536.83       108.71       645.54       0.9909          1.0015            1.0000         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)            536.83       166.20       703.03       0.9987          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)            536.83       117.65       654.49       0.9900          1.0017            1.0000         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)            536.83       174.61       711.44       0.9985          1.0002            1.0000         2.00
IVF-Binary-256-nl223-random (self)                       536.83       282.40       819.23       0.9908          1.0016            1.0000         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)             735.15        51.17       786.33       0.5622          1.6812            1.5290         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)             735.15        52.81       787.96       0.5610          1.6936            1.5367         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)             735.15        60.03       795.19       0.5584          1.7353            1.5549         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)            735.15       108.91       844.06       0.9917          1.0014            1.0000         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)            735.15       166.04       901.19       0.9987          1.0002            1.0000         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)            735.15       110.29       845.45       0.9914          1.0014            1.0000         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)            735.15       163.92       899.07       0.9988          1.0002            1.0000         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)            735.15       114.91       850.06       0.9903          1.0017            1.0000         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)            735.15       171.47       906.62       0.9986          1.0002            1.0000         2.09
IVF-Binary-256-nl316-random (self)                       735.15       274.79     1_009.94       0.9912          1.0015            1.0000         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)                 983.93        47.33     1_031.27       0.6038          1.4891            1.3755         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)                983.93        56.30     1_040.23       0.5989          1.5218            1.3913         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)                983.93        65.26     1_049.20       0.5975          1.5420            1.3965         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)                983.93       117.79     1_101.73       0.9926          1.0013            1.0000         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)                983.93       163.04     1_146.97       0.9972          1.0006            1.0000         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)               983.93       118.66     1_102.60       0.9934          1.0010            1.0000         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)               983.93       177.08     1_161.01       0.9991          1.0001            1.0000         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)               983.93       127.17     1_111.10       0.9927          1.0012            1.0000         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)               983.93       184.65     1_168.58       0.9990          1.0001            1.0000         1.93
IVF-Binary-256-nl158-pca (self)                          983.93       326.11     1_310.05       0.9929          1.0011            1.0000         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)                583.36        49.36       632.71       0.6017          1.4948            1.3802         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)                583.36        51.96       635.32       0.5998          1.5091            1.3875         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)                583.36        59.63       642.99       0.5979          1.5320            1.3962         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)               583.36       106.52       689.88       0.9937          1.0010            1.0000         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)               583.36       166.33       749.69       0.9987          1.0002            1.0000         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)               583.36       108.94       692.30       0.9935          1.0010            1.0000         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)               583.36       167.11       750.47       0.9991          1.0001            1.0000         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)               583.36       116.97       700.33       0.9929          1.0011            1.0000         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)               583.36       174.01       757.37       0.9990          1.0001            1.0000         2.00
IVF-Binary-256-nl223-pca (self)                          583.36       282.17       865.53       0.9931          1.0011            1.0000         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)                742.66        50.43       793.09       0.6012          1.4964            1.3827         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)                742.66        52.18       794.84       0.6002          1.5046            1.3866         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)                742.66        59.81       802.47       0.5985          1.5256            1.3943         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)               742.66       107.44       850.10       0.9940          1.0009            1.0000         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)               742.66       160.78       903.44       0.9990          1.0001            1.0000         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)               742.66       107.97       850.63       0.9938          1.0009            1.0000         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)               742.66       168.54       911.20       0.9992          1.0001            1.0000         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)               742.66       115.26       857.93       0.9931          1.0011            1.0000         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)               742.66       172.20       914.86       0.9991          1.0001            1.0000         2.09
IVF-Binary-256-nl316-pca (self)                          742.66       269.38     1_012.04       0.9934          1.0011            1.0000         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)              976.78        66.20     1_042.97       0.6409          1.4486            1.3318         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)             976.78        80.29     1_057.07       0.6350          1.4901            1.3495         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)             976.78       102.12     1_078.90       0.6333          1.5130            1.3550         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)             976.78       130.50     1_107.27       0.9965          1.0007            1.0000         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)             976.78       190.50     1_167.28       0.9978          1.0005            1.0000         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)            976.78       143.34     1_120.12       0.9982          1.0002            1.0000         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)            976.78       211.52     1_188.29       0.9999          1.0000            1.0000         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)            976.78       156.76     1_133.54       0.9979          1.0003            1.0000         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)            976.78       217.37     1_194.14       0.9998          1.0000            1.0000         3.71
IVF-Binary-512-nl158-random (self)                       976.78       414.88     1_391.66       0.9979          1.0003            1.0000         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)             556.77        68.54       625.31       0.6385          1.4531            1.3378         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)             556.77        72.73       629.51       0.6365          1.4696            1.3445         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)             556.77        83.72       640.49       0.6339          1.5002            1.3523         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)            556.77       126.77       683.55       0.9978          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)            556.77       182.59       739.36       0.9993          1.0001            1.0000         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)            556.77       131.46       688.23       0.9980          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)            556.77       188.34       745.11       0.9998          1.0000            1.0000         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)            556.77       142.24       699.01       0.9979          1.0003            1.0000         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)            556.77       203.36       760.13       0.9998          1.0000            1.0000         3.77
IVF-Binary-512-nl223-random (self)                       556.77       354.05       910.82       0.9978          1.0003            1.0000         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)             739.59        68.71       808.30       0.6379          1.4598            1.3395         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)             739.59        71.98       811.57       0.6370          1.4674            1.3427         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)             739.59        80.41       820.00       0.6347          1.4949            1.3507         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)            739.59       126.83       866.42       0.9981          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)            739.59       198.87       938.46       0.9996          1.0001            1.0000         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)            739.59       138.30       877.88       0.9982          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)            739.59       187.13       926.72       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)            739.59       143.10       882.69       0.9980          1.0003            1.0000         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)            739.59       198.82       938.41       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-random (self)                       739.59       340.42     1_080.01       0.9980          1.0003            1.0000         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)                 995.44        67.32     1_062.76       0.6576          1.3883            1.2906         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)                995.44        80.38     1_075.83       0.6524          1.4212            1.3033         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)                995.44        93.82     1_089.27       0.6508          1.4401            1.3081         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)                995.44       129.58     1_125.02       0.9969          1.0006            1.0000         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)                995.44       185.98     1_181.42       0.9978          1.0005            1.0000         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)               995.44       142.66     1_138.10       0.9987          1.0001            1.0000         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)               995.44       205.97     1_201.42       0.9999          1.0000            1.0000         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)               995.44       156.71     1_152.16       0.9985          1.0002            1.0000         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)               995.44       216.26     1_211.70       0.9999          1.0000            1.0000         3.71
IVF-Binary-512-nl158-pca (self)                          995.44       414.47     1_409.91       0.9986          1.0002            1.0000         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)                583.68        70.91       654.59       0.6553          1.3963            1.2930         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)                583.68        74.62       658.30       0.6534          1.4090            1.2987         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)                583.68        84.36       668.04       0.6514          1.4319            1.3057         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)               583.68       127.10       710.78       0.9983          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)               583.68       182.14       765.82       0.9993          1.0001            1.0000         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)               583.68       130.97       714.65       0.9986          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)               583.68       191.51       775.19       0.9998          1.0000            1.0000         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)               583.68       142.92       726.60       0.9985          1.0002            1.0000         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)               583.68       201.23       784.91       0.9999          1.0000            1.0000         3.77
IVF-Binary-512-nl223-pca (self)                          583.68       353.44       937.12       0.9985          1.0002            1.0000         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)                765.10        68.65       833.75       0.6545          1.4014            1.2937         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)                765.10        73.09       838.18       0.6536          1.4079            1.2965         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)                765.10        82.41       847.51       0.6516          1.4281            1.3030         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)               765.10       127.05       892.15       0.9986          1.0002            1.0000         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)               765.10       182.21       947.30       0.9996          1.0001            1.0000         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)               765.10       127.20       892.30       0.9987          1.0001            1.0000         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)               765.10       187.61       952.71       0.9998          1.0000            1.0000         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)               765.10       138.66       903.76       0.9986          1.0002            1.0000         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)               765.10       197.48       962.58       0.9999          1.0000            1.0000         3.86
IVF-Binary-512-nl316-pca (self)                          765.10       339.04     1_104.13       0.9986          1.0002            1.0000         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)           1_032.00       101.89     1_133.89       0.6845          1.3532            1.2576         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)          1_032.00       121.12     1_153.12       0.6792          1.3863            1.2711         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)          1_032.00       141.47     1_173.47       0.6776          1.4037            1.2752         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)          1_032.00       169.21     1_201.21       0.9976          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)          1_032.00       223.64     1_255.65       0.9978          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)         1_032.00       184.49     1_216.49       0.9997          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)         1_032.00       250.32     1_282.32       0.9999          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)         1_032.00       205.97     1_237.97       0.9996          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)         1_032.00       271.82     1_303.82       1.0000          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-random (self)                    1_032.00       551.18     1_583.19       0.9995          1.0000            1.0000         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)            583.82       101.01       684.83       0.6825          1.3587            1.2602         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)            583.82       108.58       692.40       0.6805          1.3722            1.2665         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)            583.82       125.60       709.42       0.6783          1.3950            1.2736         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)           583.82       162.30       746.12       0.9991          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)           583.82       222.01       805.83       0.9994          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)           583.82       170.62       754.44       0.9995          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)           583.82       230.21       814.02       0.9998          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)           583.82       187.19       771.00       0.9996          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)           583.82       250.07       833.89       0.9999          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-random (self)                      583.82       485.55     1_069.37       0.9994          1.0001            1.0000         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)            778.48       103.22       881.70       0.6814          1.3665            1.2637         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)            778.48       108.46       886.94       0.6806          1.3728            1.2656         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)            778.48       118.56       897.03       0.6785          1.3928            1.2727         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)           778.48       161.91       940.39       0.9994          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)           778.48       220.68       999.15       0.9997          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)           778.48       168.52       947.00       0.9995          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)           778.48       230.05     1_008.53       0.9998          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)           778.48       179.15       957.62       0.9996          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)           778.48       242.30     1_020.78       0.9999          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-random (self)                      778.48       469.34     1_247.82       0.9994          1.0001            1.0000         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)              1_097.49       101.42     1_198.92       0.6927          1.3310            1.2428         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)             1_097.49       119.53     1_217.02       0.6876          1.3596            1.2559         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)             1_097.49       155.71     1_253.20       0.6860          1.3757            1.2595         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)             1_097.49       179.60     1_277.09       0.9977          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)             1_097.49       226.90     1_324.40       0.9978          1.0005            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)            1_097.49       186.90     1_284.40       0.9998          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)            1_097.49       246.68     1_344.17       1.0000          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)            1_097.49       205.22     1_302.72       0.9997          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)            1_097.49       272.18     1_369.68       1.0000          1.0000            1.0000         7.26
IVF-Binary-1024-nl158-pca (self)                       1_097.49       551.12     1_648.62       0.9997          1.0000            1.0000         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)               606.11       102.51       708.62       0.6901          1.3388            1.2451         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)               606.11       107.26       713.37       0.6884          1.3497            1.2507         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)               606.11       124.06       730.17       0.6863          1.3698            1.2573         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)              606.11       164.47       770.58       0.9992          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)              606.11       220.40       826.51       0.9994          1.0001            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)              606.11       174.46       780.57       0.9996          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)              606.11       234.54       840.65       0.9999          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)              606.11       185.50       791.61       0.9996          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)              606.11       248.37       854.48       1.0000          1.0000            1.0000         7.32
IVF-Binary-1024-nl223-pca (self)                         606.11       489.60     1_095.71       0.9995          1.0001            1.0000         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)               796.15       102.83       898.98       0.6893          1.3439            1.2492         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)               796.15       105.70       901.85       0.6884          1.3499            1.2513         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)               796.15       119.63       915.78       0.6867          1.3674            1.2565         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)              796.15       167.15       963.30       0.9995          1.0001            1.0000         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)              796.15       220.31     1_016.46       0.9997          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)              796.15       163.74       959.90       0.9996          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)              796.15       227.17     1_023.32       0.9999          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)              796.15       184.87       981.02       0.9997          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)              796.15       243.88     1_040.03       1.0000          1.0000            1.0000         7.42
IVF-Binary-1024-nl316-pca (self)                         796.15       482.66     1_278.81       0.9995          1.0001            1.0000         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)                936.11       176.75     1_112.86       0.0686          6.6373            6.1345         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               936.11       191.84     1_127.95       0.0552          7.8718            7.1703         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               936.11       204.21     1_140.32       0.0506          8.7173            7.9030         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               936.11       218.53     1_154.65       0.3995          1.6136            1.5282         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               936.11       375.11     1_311.22       0.6372          1.2495            1.1925         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              936.11       233.19     1_169.31       0.3092          1.8460            1.7473         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              936.11       404.37     1_340.49       0.4802          1.4437            1.3765         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              936.11       243.33     1_179.44       0.2742          1.9837            1.8755         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              936.11       429.99     1_366.10       0.4127          1.5616            1.4852         1.68
IVF-Binary-256-nl158-sign (self)                         936.11       706.44     1_642.56       0.3153          1.8338            1.7387         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               507.59       170.87       678.46       0.0663          6.5482            6.0899         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               507.59       174.92       682.51       0.0608          6.9856            6.4759         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               507.59       188.55       696.14       0.0542          7.8630            7.2917         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              507.59       207.67       715.26       0.3662          1.6629            1.5796         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              507.59       368.99       876.58       0.5970          1.2833            1.2269         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              507.59       229.35       736.94       0.3332          1.7517            1.6663         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              507.59       387.16       894.75       0.5316          1.3585            1.2997         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              507.59       223.36       730.95       0.2923          1.8995            1.8113         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              507.59       396.04       903.63       0.4448          1.4906            1.4280         1.75
IVF-Binary-256-nl223-sign (self)                         507.59       625.18     1_132.77       0.3388          1.7404            1.6554         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               696.62       165.53       862.15       0.0659          6.5012            6.0523         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               696.62       169.77       866.39       0.0631          6.7220            6.2235         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               696.62       178.68       875.30       0.0561          7.4986            6.9108         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              696.62       208.53       905.15       0.3648          1.6702            1.5892         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              696.62       365.61     1_062.23       0.5865          1.2945            1.2405         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              696.62       210.32       906.94       0.3479          1.7151            1.6275         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              696.62       369.61     1_066.23       0.5531          1.3338            1.2771         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              696.62       218.02       914.64       0.3051          1.8539            1.7618         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              696.62       384.00     1_080.62       0.4659          1.4564            1.3961         1.84
IVF-Binary-256-nl316-sign (self)                         696.62       609.54     1_306.16       0.3540          1.7015            1.6180         1.84
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
Exhaustive (query)                                        68.39     1_278.68     1_347.07       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.39     4_255.59     4_323.98       1.0000          1.0000            1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)                135.79       276.35       412.14       0.5547          1.7646            1.5366         2.03
ExhaustiveBinary-256-random-rf10 (query)                 135.79       405.90       541.68       0.9898          1.0017            1.0000         2.03
ExhaustiveBinary-256-random-rf20 (query)                 135.79       540.43       676.21       0.9985          1.0002            1.0000         2.03
ExhaustiveBinary-256-random (self)                       135.79     1_268.97     1_404.75       0.9899          1.0016            1.0000         2.03
ExhaustiveBinary-256-pca_no_rr (query)                   218.24       266.87       485.11       0.5767          1.6243            1.4311         2.03
ExhaustiveBinary-256-pca-rf10 (query)                    218.24       406.88       625.12       0.9904          1.0016            1.0000         2.03
ExhaustiveBinary-256-pca-rf20 (query)                    218.24       541.97       760.22       0.9984          1.0002            1.0000         2.03
ExhaustiveBinary-256-pca (self)                          218.24     1_269.57     1_487.82       0.9905          1.0016            1.0000         2.03
ExhaustiveBinary-512-random_no_rr (query)                209.23       375.36       584.59       0.6013          1.6760            1.4608         4.05
ExhaustiveBinary-512-random-rf10 (query)                 209.23       535.66       744.90       0.9977          1.0003            1.0000         4.05
ExhaustiveBinary-512-random-rf20 (query)                 209.23       660.71       869.94       0.9998          1.0000            1.0000         4.05
ExhaustiveBinary-512-random (self)                       209.23     1_661.62     1_870.85       0.9975          1.0003            1.0000         4.05
ExhaustiveBinary-512-pca_no_rr (query)                   295.39       375.51       670.90       0.6443          1.4426            1.3064         4.05
ExhaustiveBinary-512-pca-rf10 (query)                    295.39       519.62       815.00       0.9985          1.0002            1.0000         4.05
ExhaustiveBinary-512-pca-rf20 (query)                    295.39       687.31       982.70       0.9999          1.0000            1.0000         4.05
ExhaustiveBinary-512-pca (self)                          295.39     1_688.87     1_984.25       0.9984          1.0002            1.0000         4.05
ExhaustiveBinary-1024-random_no_rr (query)               255.86       559.88       815.75       0.6624          1.4553            1.3048         8.11
ExhaustiveBinary-1024-random-rf10 (query)                255.86       719.41       975.27       0.9995          1.0001            1.0000         8.11
ExhaustiveBinary-1024-random-rf20 (query)                255.86       870.95     1_126.81       1.0000          1.0000            1.0000         8.11
ExhaustiveBinary-1024-random (self)                      255.86     2_341.87     2_597.73       0.9994          1.0001            1.0000         8.11
ExhaustiveBinary-1024-pca_no_rr (query)                  344.90       566.74       911.64       0.6865          1.3603            1.2383         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                   344.90       728.60     1_073.50       0.9996          1.0001            1.0000         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                   344.90       885.84     1_230.73       1.0000          1.0000            1.0000         8.11
ExhaustiveBinary-1024-pca (self)                         344.90     2_360.75     2_705.65       0.9996          1.0001            1.0000         8.11
ExhaustiveBinary-512-sign_no_rr (query)                   85.02       650.87       735.89       0.0400         18.1511           13.6734         3.05
ExhaustiveBinary-512-sign-rf10 (query)                    85.02       707.65       792.67       0.1821          2.5573            2.4620         3.05
ExhaustiveBinary-512-sign-rf20 (query)                    85.02     1_082.19     1_167.21       0.3140          1.8429            1.7786         3.05
ExhaustiveBinary-512-sign (self)                          85.02     2_283.74     2_368.76       0.1897          2.5286            2.4283         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)            1_846.15        79.90     1_926.05       0.5633          1.6328            1.4874         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           1_846.15        92.22     1_938.37       0.5600          1.6630            1.5060         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           1_846.15       101.14     1_947.29       0.5583          1.6921            1.5135         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           1_846.15       170.70     2_016.85       0.9917          1.0013            1.0000         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           1_846.15       262.59     2_108.74       0.9978          1.0004            1.0000         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          1_846.15       175.15     2_021.31       0.9917          1.0013            1.0000         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          1_846.15       269.64     2_115.79       0.9989          1.0001            1.0000         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          1_846.15       184.73     2_030.89       0.9910          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          1_846.15       271.16     2_117.31       0.9988          1.0001            1.0000         2.34
IVF-Binary-256-nl158-random (self)                     1_846.15       401.51     2_247.66       0.9919          1.0013            1.0000         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)             907.60        77.88       985.48       0.5617          1.6446            1.4977         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)             907.60        81.58       989.19       0.5603          1.6574            1.5054         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)             907.60        87.47       995.08       0.5590          1.6801            1.5121         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)            907.60       169.56     1_077.16       0.9924          1.0011            1.0000         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)            907.60       263.92     1_171.52       0.9988          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)            907.60       167.09     1_074.70       0.9918          1.0012            1.0000         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)            907.60       263.90     1_171.50       0.9989          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)            907.60       176.07     1_083.67       0.9911          1.0014            1.0000         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)            907.60       269.87     1_177.47       0.9988          1.0001            1.0000         2.47
IVF-Binary-256-nl223-random (self)                       907.60       369.93     1_277.53       0.9920          1.0012            1.0000         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)           1_119.06        81.48     1_200.54       0.5616          1.6430            1.4953         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           1_119.06        87.95     1_207.01       0.5609          1.6507            1.4993         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           1_119.06        90.19     1_209.25       0.5595          1.6715            1.5088         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          1_119.06       170.66     1_289.72       0.9924          1.0011            1.0000         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          1_119.06       267.47     1_386.53       0.9990          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          1_119.06       168.16     1_287.22       0.9920          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          1_119.06       278.49     1_397.55       0.9990          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          1_119.06       176.02     1_295.08       0.9913          1.0013            1.0000         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          1_119.06       270.45     1_389.51       0.9988          1.0001            1.0000         2.65
IVF-Binary-256-nl316-random (self)                     1_119.06       370.82     1_489.88       0.9922          1.0012            1.0000         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               1_899.05        71.35     1_970.40       0.5839          1.5249            1.4040         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              1_899.05        81.62     1_980.67       0.5812          1.5475            1.4163         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              1_899.05        87.99     1_987.04       0.5800          1.5661            1.4216         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              1_899.05       165.25     2_064.30       0.9918          1.0013            1.0000         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              1_899.05       254.20     2_153.25       0.9977          1.0004            1.0000         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             1_899.05       169.28     2_068.34       0.9918          1.0013            1.0000         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             1_899.05       262.66     2_161.71       0.9988          1.0002            1.0000         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             1_899.05       179.90     2_078.95       0.9912          1.0014            1.0000         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             1_899.05       270.51     2_169.56       0.9987          1.0002            1.0000         2.34
IVF-Binary-256-nl158-pca (self)                        1_899.05       384.51     2_283.56       0.9920          1.0013            1.0000         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)                976.73        76.32     1_053.04       0.5829          1.5323            1.4104         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)                976.73        81.32     1_058.05       0.5817          1.5427            1.4161         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)                976.73        88.09     1_064.82       0.5806          1.5586            1.4197         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)               976.73       167.29     1_144.02       0.9924          1.0012            1.0000         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)               976.73       256.95     1_233.67       0.9988          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)               976.73       168.00     1_144.72       0.9921          1.0013            1.0000         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)               976.73       261.03     1_237.76       0.9988          1.0001            1.0000         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)               976.73       179.05     1_155.78       0.9914          1.0014            1.0000         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)               976.73       271.00     1_247.73       0.9987          1.0002            1.0000         2.47
IVF-Binary-256-nl223-pca (self)                          976.73       372.77     1_349.50       0.9921          1.0013            1.0000         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              1_227.02        81.52     1_308.54       0.5828          1.5321            1.4094         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              1_227.02        82.98     1_310.00       0.5823          1.5371            1.4112         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              1_227.02        89.79     1_316.81       0.5810          1.5509            1.4187         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             1_227.02       170.77     1_397.79       0.9923          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             1_227.02       269.63     1_496.65       0.9990          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             1_227.02       169.81     1_396.83       0.9920          1.0012            1.0000         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             1_227.02       262.58     1_489.60       0.9989          1.0001            1.0000         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             1_227.02       181.91     1_408.93       0.9914          1.0014            1.0000         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             1_227.02       271.14     1_498.16       0.9988          1.0002            1.0000         2.65
IVF-Binary-256-nl316-pca (self)                        1_227.02       365.08     1_592.10       0.9923          1.0012            1.0000         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)            1_877.27        98.07     1_975.34       0.6093          1.5616            1.4208         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)           1_877.27       111.95     1_989.22       0.6053          1.5925            1.4401         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)           1_877.27       123.69     2_000.96       0.6034          1.6196            1.4475         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)           1_877.27       190.32     2_067.59       0.9973          1.0004            1.0000         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)           1_877.27       294.22     2_171.50       0.9984          1.0003            1.0000         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)          1_877.27       203.14     2_080.41       0.9983          1.0002            1.0000         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)          1_877.27       307.51     2_184.78       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)          1_877.27       213.32     2_090.60       0.9980          1.0002            1.0000         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)          1_877.27       313.17     2_190.44       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-random (self)                     1_877.27       494.45     2_371.73       0.9982          1.0002            1.0000         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)             965.51       112.46     1_077.97       0.6074          1.5768            1.4307         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)             965.51       109.77     1_075.29       0.6059          1.5910            1.4390         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)             965.51       119.84     1_085.35       0.6046          1.6113            1.4455         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)            965.51       196.96     1_162.47       0.9982          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)            965.51       287.17     1_252.68       0.9996          1.0001            1.0000         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)            965.51       195.68     1_161.19       0.9982          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)            965.51       293.64     1_259.15       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)            965.51       210.64     1_176.15       0.9980          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)            965.51       308.87     1_274.38       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-random (self)                       965.51       473.35     1_438.86       0.9982          1.0002            1.0000         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)           1_200.55       111.37     1_311.93       0.6068          1.5788            1.4333         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)           1_200.55       111.29     1_311.84       0.6061          1.5861            1.4378         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)           1_200.55       120.41     1_320.96       0.6045          1.6061            1.4455         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)          1_200.55       200.20     1_400.75       0.9985          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)          1_200.55       295.70     1_496.26       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)          1_200.55       197.51     1_398.06       0.9984          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)          1_200.55       297.50     1_498.05       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)          1_200.55       208.58     1_409.13       0.9982          1.0002            1.0000         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)          1_200.55       311.19     1_511.74       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-random (self)                     1_200.55       464.31     1_664.87       0.9983          1.0002            1.0000         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)               1_952.58        98.33     2_050.91       0.6498          1.3846            1.2884         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)              1_952.58       112.09     2_064.67       0.6473          1.4005            1.2964         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)              1_952.58       123.33     2_075.91       0.6462          1.4122            1.3004         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)              1_952.58       191.41     2_143.99       0.9975          1.0004            1.0000         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)              1_952.58       284.29     2_236.87       0.9984          1.0003            1.0000         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)             1_952.58       199.08     2_151.67       0.9987          1.0001            1.0000         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)             1_952.58       312.76     2_265.34       0.9998          1.0000            1.0000         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)             1_952.58       212.33     2_164.91       0.9986          1.0001            1.0000         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)             1_952.58       313.08     2_265.66       0.9999          1.0000            1.0000         4.36
IVF-Binary-512-nl158-pca (self)                        1_952.58       494.46     2_447.04       0.9987          1.0001            1.0000         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_041.50       106.24     1_147.74       0.6484          1.3915            1.2913         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_041.50       109.00     1_150.50       0.6474          1.3991            1.2957         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_041.50       119.78     1_161.28       0.6464          1.4094            1.2989         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_041.50       193.96     1_235.47       0.9986          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_041.50       286.56     1_328.07       0.9996          1.0001            1.0000         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_041.50       197.53     1_239.04       0.9987          1.0001            1.0000         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_041.50       290.44     1_331.95       0.9998          1.0000            1.0000         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_041.50       206.40     1_247.91       0.9986          1.0002            1.0000         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_041.50       318.76     1_360.27       0.9999          1.0000            1.0000         4.49
IVF-Binary-512-nl223-pca (self)                        1_041.50       468.95     1_510.45       0.9987          1.0002            1.0000         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)              1_274.92       108.52     1_383.44       0.6480          1.3924            1.2943         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)              1_274.92       112.14     1_387.06       0.6476          1.3963            1.2956         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)              1_274.92       120.58     1_395.50       0.6465          1.4068            1.2991         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)             1_274.92       200.11     1_475.03       0.9988          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)             1_274.92       295.05     1_569.97       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)             1_274.92       209.99     1_484.91       0.9987          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)             1_274.92       312.61     1_587.53       0.9998          1.0000            1.0000         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)             1_274.92       207.40     1_482.31       0.9986          1.0001            1.0000         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)             1_274.92       322.04     1_596.96       0.9999          1.0000            1.0000         4.67
IVF-Binary-512-nl316-pca (self)                        1_274.92       477.76     1_752.68       0.9987          1.0001            1.0000         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)           1_978.78       149.63     2_128.41       0.6686          1.3871            1.2852         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)          1_978.78       167.70     2_146.48       0.6654          1.4070            1.2940         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)          1_978.78       192.83     2_171.61       0.6639          1.4242            1.2993         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)          1_978.78       257.42     2_236.20       0.9983          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)          1_978.78       354.10     2_332.88       0.9985          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)         1_978.78       262.02     2_240.79       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)         1_978.78       371.34     2_350.12       0.9999          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)         1_978.78       280.17     2_258.95       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)         1_978.78       391.94     2_370.72       1.0000          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-random (self)                    1_978.78       723.95     2_702.72       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_024.15       156.08     1_180.23       0.6666          1.3978            1.2895         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_024.15       160.65     1_184.80       0.6654          1.4067            1.2934         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_024.15       177.14     1_201.29       0.6643          1.4195            1.2982         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_024.15       249.93     1_274.08       0.9994          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_024.15       355.68     1_379.83       0.9997          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_024.15       254.80     1_278.95       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_024.15       367.98     1_392.14       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_024.15       273.87     1_298.02       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_024.15       386.21     1_410.36       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-random (self)                    1_024.15       675.06     1_699.22       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)          1_234.98       160.52     1_395.50       0.6664          1.3994            1.2908         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)          1_234.98       162.89     1_397.87       0.6658          1.4041            1.2928         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)          1_234.98       176.22     1_411.20       0.6645          1.4169            1.2974         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)         1_234.98       262.47     1_497.45       0.9996          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)         1_234.98       362.33     1_597.31       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)         1_234.98       257.19     1_492.17       0.9996          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)         1_234.98       367.27     1_602.25       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)         1_234.98       275.50     1_510.48       0.9996          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)         1_234.98       384.36     1_619.34       1.0000          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-random (self)                    1_234.98       671.27     1_906.25       0.9996          1.0001            1.0000         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)              2_019.74       151.35     2_171.09       0.6914          1.3138            1.2265         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)             2_019.74       168.25     2_187.99       0.6887          1.3290            1.2329         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)             2_019.74       185.37     2_205.11       0.6877          1.3391            1.2353         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)             2_019.74       251.43     2_271.17       0.9983          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)             2_019.74       344.15     2_363.89       0.9985          1.0003            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)            2_019.74       263.37     2_283.11       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)            2_019.74       368.17     2_387.92       0.9999          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)            2_019.74       282.35     2_302.10       0.9996          1.0001            1.0000         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)            2_019.74       387.87     2_407.61       1.0000          1.0000            1.0000         8.42
IVF-Binary-1024-nl158-pca (self)                       2_019.74       724.98     2_744.72       0.9996          1.0000            1.0000         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_118.36       156.08     1_274.44       0.6897          1.3215            1.2293         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_118.36       161.24     1_279.61       0.6887          1.3276            1.2310         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_118.36       177.78     1_296.14       0.6878          1.3371            1.2341         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_118.36       253.87     1_372.23       0.9995          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_118.36       354.99     1_473.35       0.9997          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_118.36       257.43     1_375.79       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_118.36       368.28     1_486.64       0.9999          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_118.36       272.23     1_390.59       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_118.36       384.56     1_502.93       1.0000          1.0000            1.0000         8.54
IVF-Binary-1024-nl223-pca (self)                       1_118.36       671.34     1_789.70       0.9996          1.0001            1.0000         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)             1_345.48       160.32     1_505.80       0.6897          1.3223            1.2301         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)             1_345.48       171.73     1_517.21       0.6893          1.3259            1.2310         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)             1_345.48       175.46     1_520.94       0.6882          1.3349            1.2341         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)            1_345.48       255.53     1_601.01       0.9997          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)            1_345.48       363.38     1_708.86       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)            1_345.48       254.86     1_600.34       0.9997          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)            1_345.48       367.65     1_713.13       0.9999          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)            1_345.48       274.37     1_619.85       0.9997          1.0001            1.0000         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)            1_345.48       388.70     1_734.18       1.0000          1.0000            1.0000         8.73
IVF-Binary-1024-nl316-pca (self)                       1_345.48       668.25     2_013.73       0.9996          1.0000            1.0000         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)              1_763.56       280.30     2_043.86       0.0594          7.9493            7.2573         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)             1_763.56       300.18     2_063.74       0.0523          9.1717            8.0433         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)             1_763.56       313.40     2_076.96       0.0490         10.2313            8.7259         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)             1_763.56       362.73     2_126.29       0.3200          1.8552            1.7422         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)             1_763.56       640.08     2_403.63       0.5141          1.4089            1.3383         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)            1_763.56       381.70     2_145.25       0.2782          1.9998            1.8714         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)            1_763.56       659.14     2_422.70       0.4424          1.5272            1.4451         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)            1_763.56       382.50     2_146.06       0.2558          2.1020            1.9577         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)            1_763.56       676.45     2_440.01       0.4010          1.6091            1.5201         3.36
IVF-Binary-512-nl158-sign (self)                       1_763.56     1_047.32     2_810.88       0.2859          1.9751            1.8430         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               862.16       291.76     1_153.92       0.0573          7.8797            7.1620         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               862.16       292.19     1_154.35       0.0543          8.2745            7.4598         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               862.16       305.06     1_167.22       0.0504          9.1593            8.1667         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              862.16       361.51     1_223.67       0.3141          1.8536            1.7519         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              862.16       647.58     1_509.74       0.5018          1.4180            1.3471         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              862.16       360.56     1_222.72       0.2951          1.9148            1.8057         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              862.16       647.80     1_509.96       0.4685          1.4696            1.3955         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              862.16       377.51     1_239.67       0.2690          2.0219            1.8959         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              862.16       665.57     1_527.73       0.4174          1.5671            1.4860         3.49
IVF-Binary-512-nl223-sign (self)                         862.16     1_010.57     1_872.73       0.3019          1.8966            1.7840         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)             1_067.75       287.32     1_355.08       0.0576          7.7417            7.1217         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)             1_067.75       297.00     1_364.75       0.0558          7.9505            7.2924         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)             1_067.75       301.85     1_369.60       0.0519          8.6482            7.8795         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)            1_067.75       371.42     1_439.17       0.3184          1.8361            1.7366         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)            1_067.75       668.07     1_735.83       0.5013          1.4137            1.3449         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)            1_067.75       368.89     1_436.64       0.3085          1.8697            1.7662         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)            1_067.75       648.41     1_716.16       0.4838          1.4395            1.3714         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)            1_067.75       382.58     1_450.34       0.2821          1.9696            1.8495         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)            1_067.75       662.35     1_730.10       0.4343          1.5288            1.4528         3.67
IVF-Binary-512-nl316-sign (self)                       1_067.75     1_023.55     2_091.30       0.3137          1.8531            1.7443         3.67
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
Exhaustive (query)                                       102.34     1_846.02     1_948.36       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        102.34     6_191.86     6_294.20       1.0000          1.0000            1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)                194.40       286.20       480.60       0.5361          1.8068            1.5908         2.28
ExhaustiveBinary-256-random-rf10 (query)                 194.40       455.91       650.32       0.9868          1.0022            1.0000         2.28
ExhaustiveBinary-256-random-rf20 (query)                 194.40       600.17       794.57       0.9980          1.0003            1.0000         2.28
ExhaustiveBinary-256-random (self)                       194.40     1_382.23     1_576.63       0.9876          1.0021            1.0000         2.28
ExhaustiveBinary-256-pca_no_rr (query)                   401.39       299.56       700.95       0.5754          1.5495            1.4128         2.28
ExhaustiveBinary-256-pca-rf10 (query)                    401.39       452.17       853.56       0.9895          1.0018            1.0000         2.28
ExhaustiveBinary-256-pca-rf20 (query)                    401.39       590.05       991.43       0.9983          1.0002            1.0000         2.28
ExhaustiveBinary-256-pca (self)                          401.39     1_369.42     1_770.81       0.9897          1.0017            1.0000         2.28
ExhaustiveBinary-512-random_no_rr (query)                300.13       401.99       702.12       0.5866          1.6778            1.4946         4.55
ExhaustiveBinary-512-random-rf10 (query)                 300.13       574.18       874.31       0.9966          1.0005            1.0000         4.55
ExhaustiveBinary-512-random-rf20 (query)                 300.13       733.11     1_033.24       0.9997          1.0001            1.0000         4.55
ExhaustiveBinary-512-random (self)                       300.13     1_808.78     2_108.92       0.9969          1.0004            1.0000         4.55
ExhaustiveBinary-512-pca_no_rr (query)                   493.00       399.88       892.89       0.6388          1.4217            1.3032         4.55
ExhaustiveBinary-512-pca-rf10 (query)                    493.00       563.77     1_056.77       0.9979          1.0003            1.0000         4.55
ExhaustiveBinary-512-pca-rf20 (query)                    493.00       738.17     1_231.18       0.9998          1.0000            1.0000         4.55
ExhaustiveBinary-512-pca (self)                          493.00     1_798.78     2_291.79       0.9981          1.0002            1.0000         4.55
ExhaustiveBinary-1024-random_no_rr (query)               496.42       616.84     1_113.25       0.6446          1.4909            1.3512         9.11
ExhaustiveBinary-1024-random-rf10 (query)                496.42       794.05     1_290.46       0.9993          1.0001            1.0000         9.11
ExhaustiveBinary-1024-random-rf20 (query)                496.42       969.33     1_465.75       0.9999          1.0000            1.0000         9.11
ExhaustiveBinary-1024-random (self)                      496.42     2_587.24     3_083.66       0.9994          1.0001            1.0000         9.11
ExhaustiveBinary-1024-pca_no_rr (query)                  698.65       611.31     1_309.96       0.6795          1.3452            1.2483         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                   698.65       789.89     1_488.54       0.9996          1.0001            1.0000         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                   698.65       968.56     1_667.21       1.0000          1.0000            1.0000         9.11
ExhaustiveBinary-1024-pca (self)                         698.65     2_591.22     3_289.87       0.9997          1.0000            1.0000         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  129.22       822.93       952.15       0.0420         17.7082           13.0970         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   129.22       899.01     1_028.23       0.1896          2.5240            2.4052         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   129.22     1_390.04     1_519.26       0.3229          1.8300            1.7348         4.58
ExhaustiveBinary-768-sign (self)                         129.22     2_887.65     3_016.87       0.1997          2.4832            2.3546         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)            2_675.77       104.08     2_779.85       0.5429          1.7099            1.5460         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)           2_675.77       116.40     2_792.16       0.5407          1.7331            1.5595         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)           2_675.77       122.29     2_798.06       0.5397          1.7545            1.5687         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)           2_675.77       211.81     2_887.57       0.9891          1.0018            1.0000         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)           2_675.77       323.14     2_998.91       0.9984          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)          2_675.77       212.23     2_888.00       0.9884          1.0019            1.0000         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)          2_675.77       330.37     3_006.14       0.9986          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)          2_675.77       225.16     2_900.93       0.9877          1.0021            1.0000         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)          2_675.77       341.92     3_017.69       0.9983          1.0002            1.0000         2.74
IVF-Binary-256-nl158-random (self)                     2_675.77       492.49     3_168.26       0.9891          1.0018            1.0000         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)           1_287.63       107.69     1_395.32       0.5420          1.7182            1.5522         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)           1_287.63       108.96     1_396.60       0.5412          1.7279            1.5568         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)           1_287.63       119.63     1_407.27       0.5401          1.7508            1.5657         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)          1_287.63       216.55     1_504.18       0.9888          1.0018            1.0000         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)          1_287.63       326.35     1_613.98       0.9986          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)          1_287.63       220.73     1_508.36       0.9885          1.0019            1.0000         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)          1_287.63       328.73     1_616.36       0.9985          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)          1_287.63       225.46     1_513.09       0.9877          1.0020            1.0000         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)          1_287.63       347.87     1_635.51       0.9983          1.0002            1.0000         2.93
IVF-Binary-256-nl223-random (self)                     1_287.63       459.51     1_747.15       0.9890          1.0018            1.0000         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)           1_596.48       104.61     1_701.09       0.5422          1.7108            1.5503         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)           1_596.48       107.96     1_704.44       0.5417          1.7157            1.5534         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)           1_596.48       112.50     1_708.98       0.5408          1.7316            1.5599         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)          1_596.48       217.91     1_814.39       0.9891          1.0018            1.0000         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)          1_596.48       334.01     1_930.48       0.9987          1.0002            1.0000         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)          1_596.48       215.98     1_812.46       0.9888          1.0018            1.0000         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)          1_596.48       336.91     1_933.39       0.9986          1.0002            1.0000         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)          1_596.48       230.17     1_826.65       0.9882          1.0020            1.0000         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)          1_596.48       348.53     1_945.01       0.9984          1.0002            1.0000         3.21
IVF-Binary-256-nl316-random (self)                     1_596.48       474.55     2_071.03       0.9895          1.0017            1.0000         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)               2_876.93        93.88     2_970.82       0.5812          1.4959            1.3933         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)              2_876.93       100.70     2_977.64       0.5795          1.5088            1.3992         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)              2_876.93       106.37     2_983.30       0.5787          1.5186            1.4017         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)              2_876.93       230.30     3_107.23       0.9913          1.0014            1.0000         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)              2_876.93       317.54     3_194.47       0.9984          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)             2_876.93       211.84     3_088.78       0.9907          1.0015            1.0000         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)             2_876.93       334.30     3_211.23       0.9987          1.0002            1.0000         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)             2_876.93       222.66     3_099.60       0.9902          1.0016            1.0000         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)             2_876.93       333.50     3_210.43       0.9985          1.0002            1.0000         2.74
IVF-Binary-256-nl158-pca (self)                        2_876.93       465.51     3_342.44       0.9910          1.0014            1.0000         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)              1_477.36        96.26     1_573.62       0.5799          1.5020            1.3963         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)              1_477.36       101.80     1_579.17       0.5794          1.5067            1.3987         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)              1_477.36       107.79     1_585.15       0.5786          1.5169            1.4018         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)             1_477.36       208.61     1_685.97       0.9909          1.0015            1.0000         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)             1_477.36       323.23     1_800.60       0.9987          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)             1_477.36       207.98     1_685.34       0.9906          1.0015            1.0000         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)             1_477.36       326.21     1_803.57       0.9987          1.0002            1.0000         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)             1_477.36       220.26     1_697.63       0.9901          1.0016            1.0000         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)             1_477.36       337.35     1_814.71       0.9985          1.0002            1.0000         2.93
IVF-Binary-256-nl223-pca (self)                        1_477.36       457.60     1_934.96       0.9909          1.0014            1.0000         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)              1_809.52       105.79     1_915.31       0.5807          1.4988            1.3925         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)              1_809.52       107.77     1_917.30       0.5804          1.5016            1.3940         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)              1_809.52       111.89     1_921.41       0.5798          1.5081            1.3970         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)             1_809.52       215.97     2_025.49       0.9912          1.0014            1.0000         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)             1_809.52       332.04     2_141.56       0.9989          1.0001            1.0000         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)             1_809.52       212.72     2_022.25       0.9909          1.0015            1.0000         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)             1_809.52       335.08     2_144.60       0.9988          1.0001            1.0000         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)             1_809.52       225.99     2_035.51       0.9903          1.0016            1.0000         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)             1_809.52       340.47     2_149.99       0.9986          1.0002            1.0000         3.21
IVF-Binary-256-nl316-pca (self)                        1_809.52       471.68     2_281.21       0.9913          1.0014            1.0000         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)            2_746.60       131.47     2_878.07       0.5925          1.6007            1.4618         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)           2_746.60       140.27     2_886.87       0.5898          1.6229            1.4748         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)           2_746.60       152.25     2_898.86       0.5888          1.6417            1.4806         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)           2_746.60       251.05     2_997.65       0.9972          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)           2_746.60       361.79     3_108.39       0.9993          1.0001            1.0000         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)          2_746.60       253.54     3_000.15       0.9972          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)          2_746.60       375.95     3_122.55       0.9998          1.0000            1.0000         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)          2_746.60       261.18     3_007.78       0.9970          1.0004            1.0000         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)          2_746.60       392.80     3_139.40       0.9997          1.0000            1.0000         5.02
IVF-Binary-512-nl158-random (self)                     2_746.60       626.92     3_373.52       0.9974          1.0003            1.0000         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)           1_415.59       137.52     1_553.11       0.5912          1.6104            1.4673         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)           1_415.59       137.82     1_553.41       0.5903          1.6195            1.4718         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)           1_415.59       152.62     1_568.21       0.5890          1.6407            1.4794         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)          1_415.59       245.85     1_661.44       0.9972          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)          1_415.59       376.36     1_791.95       0.9996          1.0001            1.0000         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)          1_415.59       247.07     1_662.66       0.9972          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)          1_415.59       377.32     1_792.91       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)          1_415.59       262.49     1_678.08       0.9969          1.0004            1.0000         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)          1_415.59       391.34     1_806.93       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-random (self)                     1_415.59       605.88     2_021.47       0.9974          1.0003            1.0000         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)           1_729.86       140.67     1_870.53       0.5911          1.6075            1.4667         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)           1_729.86       143.66     1_873.51       0.5907          1.6116            1.4696         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)           1_729.86       152.24     1_882.09       0.5897          1.6270            1.4769         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)          1_729.86       257.17     1_987.02       0.9974          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)          1_729.86       383.11     2_112.97       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)          1_729.86       260.04     1_989.90       0.9973          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)          1_729.86       381.84     2_111.70       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)          1_729.86       266.70     1_996.55       0.9971          1.0004            1.0000         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)          1_729.86       404.30     2_134.16       0.9998          1.0000            1.0000         5.48
IVF-Binary-512-nl316-random (self)                     1_729.86       622.90     2_352.76       0.9976          1.0003            1.0000         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)               2_941.36       126.84     3_068.19       0.6432          1.3824            1.2890         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)              2_941.36       141.43     3_082.79       0.6414          1.3938            1.2945         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)              2_941.36       151.28     3_092.64       0.6407          1.4030            1.2975         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)              2_941.36       242.85     3_184.21       0.9980          1.0003            1.0000         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)              2_941.36       362.20     3_303.56       0.9994          1.0001            1.0000         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)             2_941.36       257.52     3_198.87       0.9982          1.0002            1.0000         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)             2_941.36       370.87     3_312.23       0.9999          1.0000            1.0000         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)             2_941.36       264.23     3_205.58       0.9981          1.0003            1.0000         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)             2_941.36       386.08     3_327.43       0.9998          1.0000            1.0000         5.02
IVF-Binary-512-nl158-pca (self)                        2_941.36       626.33     3_567.69       0.9983          1.0002            1.0000         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)              1_587.32       133.23     1_720.55       0.6420          1.3885            1.2932         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)              1_587.32       143.01     1_730.33       0.6414          1.3935            1.2956         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)              1_587.32       149.66     1_736.98       0.6407          1.4030            1.2985         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)             1_587.32       250.46     1_837.78       0.9982          1.0002            1.0000         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)             1_587.32       368.74     1_956.06       0.9997          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)             1_587.32       247.84     1_835.16       0.9982          1.0002            1.0000         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)             1_587.32       375.45     1_962.78       0.9998          1.0000            1.0000         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)             1_587.32       273.17     1_860.50       0.9981          1.0003            1.0000         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)             1_587.32       404.41     1_991.74       0.9998          1.0000            1.0000         5.21
IVF-Binary-512-nl223-pca (self)                        1_587.32       610.74     2_198.06       0.9983          1.0002            1.0000         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)              1_935.99       145.33     2_081.33       0.6422          1.3876            1.2923         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)              1_935.99       148.97     2_084.97       0.6419          1.3896            1.2933         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)              1_935.99       154.45     2_090.44       0.6413          1.3956            1.2957         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)             1_935.99       267.34     2_203.34       0.9984          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)             1_935.99       387.13     2_323.13       0.9999          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)             1_935.99       259.76     2_195.75       0.9983          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)             1_935.99       396.57     2_332.56       0.9999          1.0000            1.0000         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)             1_935.99       269.57     2_205.57       0.9981          1.0002            1.0000         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)             1_935.99       414.51     2_350.51       0.9999          1.0000            1.0000         5.48
IVF-Binary-512-nl316-pca (self)                        1_935.99       622.92     2_558.91       0.9984          1.0002            1.0000         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)           2_935.13       202.27     3_137.40       0.6492          1.4402            1.3299         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)          2_935.13       220.85     3_155.98       0.6468          1.4562            1.3410         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)          2_935.13       240.05     3_175.18       0.6457          1.4688            1.3455         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)          2_935.13       326.54     3_261.67       0.9990          1.0002            1.0000         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)          2_935.13       455.03     3_390.16       0.9994          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)         2_935.13       343.49     3_278.62       0.9995          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)         2_935.13       484.69     3_419.82       0.9999          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)         2_935.13       364.00     3_299.13       0.9994          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)         2_935.13       503.60     3_438.73       0.9999          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-random (self)                    2_935.13       940.18     3_875.31       0.9995          1.0001            1.0000         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)          1_595.60       209.99     1_805.59       0.6479          1.4466            1.3357         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)          1_595.60       217.85     1_813.45       0.6472          1.4538            1.3394         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)          1_595.60       284.42     1_880.02       0.6460          1.4684            1.3442         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)         1_595.60       341.40     1_937.00       0.9993          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)         1_595.60       467.27     2_062.87       0.9997          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)         1_595.60       343.29     1_938.89       0.9994          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)         1_595.60       487.99     2_083.59       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)         1_595.60       383.00     1_978.60       0.9994          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)         1_595.60       505.84     2_101.45       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-random (self)                    1_595.60       896.25     2_491.86       0.9995          1.0001            1.0000         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)          1_917.15       218.40     2_135.55       0.6478          1.4461            1.3344        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)          1_917.15       218.13     2_135.29       0.6475          1.4494            1.3361        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)          1_917.15       233.05     2_150.20       0.6465          1.4599            1.3407        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)         1_917.15       355.00     2_272.15       0.9995          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)         1_917.15       481.08     2_398.23       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)         1_917.15       356.81     2_273.96       0.9995          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)         1_917.15       492.19     2_409.34       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)         1_917.15       373.26     2_290.41       0.9994          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)         1_917.15       510.38     2_427.53       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-random (self)                    1_917.15       913.35     2_830.50       0.9995          1.0001            1.0000        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)              3_164.31       202.80     3_367.11       0.6828          1.3187            1.2385         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)             3_164.31       219.58     3_383.89       0.6812          1.3273            1.2437         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)             3_164.31       237.90     3_402.21       0.6805          1.3345            1.2454         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)             3_164.31       321.50     3_485.81       0.9992          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)             3_164.31       446.90     3_611.21       0.9994          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)            3_164.31       339.15     3_503.46       0.9997          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)            3_164.31       478.56     3_642.87       1.0000          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)            3_164.31       365.00     3_529.31       0.9996          1.0001            1.0000         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)            3_164.31       500.16     3_664.47       1.0000          1.0000            1.0000         9.57
IVF-Binary-1024-nl158-pca (self)                       3_164.31       939.79     4_104.10       0.9997          1.0000            1.0000         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)             1_775.04       207.08     1_982.12       0.6818          1.3238            1.2403         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)             1_775.04       213.71     1_988.75       0.6813          1.3270            1.2428         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)             1_775.04       229.36     2_004.40       0.6806          1.3344            1.2451         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)            1_775.04       342.19     2_117.24       0.9995          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)            1_775.04       468.43     2_243.48       0.9998          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)            1_775.04       351.21     2_126.25       0.9996          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)            1_775.04       476.91     2_251.95       0.9999          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)            1_775.04       359.03     2_134.07       0.9996          1.0001            1.0000         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)            1_775.04       510.47     2_285.52       1.0000          1.0000            1.0000         9.76
IVF-Binary-1024-nl223-pca (self)                       1_775.04       895.67     2_670.71       0.9997          1.0000            1.0000         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)             2_107.91       218.07     2_325.97       0.6817          1.3241            1.2409        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)             2_107.91       217.98     2_325.89       0.6815          1.3255            1.2420        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)             2_107.91       231.89     2_339.79       0.6809          1.3307            1.2441        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)            2_107.91       355.05     2_462.96       0.9997          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)            2_107.91       484.30     2_592.20       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)            2_107.91       356.35     2_464.26       0.9996          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)            2_107.91       508.25     2_616.15       0.9999          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)            2_107.91       384.15     2_492.06       0.9996          1.0001            1.0000        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)            2_107.91       523.49     2_631.40       1.0000          1.0000            1.0000        10.04
IVF-Binary-1024-nl316-pca (self)                       2_107.91       914.28     3_022.19       0.9997          1.0000            1.0000        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)              2_565.46       400.94     2_966.41       0.0573          8.2519            7.4750         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)             2_565.46       417.04     2_982.50       0.0520          9.4871            8.2097         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)             2_565.46       435.96     3_001.42       0.0494         10.3662            8.7466         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)             2_565.46       497.14     3_062.60       0.3103          1.8949            1.7846         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)             2_565.46       890.94     3_456.40       0.4776          1.4825            1.3824         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)            2_565.46       508.94     3_074.40       0.2786          2.0014            1.8766         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)            2_565.46       921.41     3_486.87       0.4285          1.5638            1.4621         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)            2_565.46       539.38     3_104.84       0.2621          2.0692            1.9339         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)            2_565.46       937.43     3_502.89       0.4002          1.6175            1.5078         5.04
IVF-Binary-768-nl158-sign (self)                       2_565.46     1_459.03     4_024.49       0.2910          1.9632            1.8395         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)             1_240.18       407.78     1_647.96       0.0570          8.2117            7.3530         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)             1_240.18       418.75     1_658.93       0.0545          8.6681            7.6901         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)             1_240.18       447.29     1_687.47       0.0514          9.6380            8.3214         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)            1_240.18       498.52     1_738.70       0.3111          1.8746            1.7621         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)            1_240.18       904.37     2_144.55       0.4734          1.4732            1.3870         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)            1_240.18       501.31     1_741.49       0.2972          1.9238            1.8036         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)            1_240.18       905.94     2_146.12       0.4499          1.5110            1.4213         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)            1_240.18       520.99     1_761.17       0.2759          2.0111            1.8835         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)            1_240.18       931.87     2_172.05       0.4143          1.5807            1.4851         5.23
IVF-Binary-768-nl223-sign (self)                       1_240.18     1_433.49     2_673.67       0.3092          1.8880            1.7686         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)             1_577.09       418.24     1_995.33       0.0581          7.9072            7.1781         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)             1_577.09       419.84     1_996.93       0.0568          8.1437            7.3474         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)             1_577.09       448.62     2_025.71       0.0534          8.9543            7.8708         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)            1_577.09       512.05     2_089.14       0.3162          1.8517            1.7465         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)            1_577.09       916.76     2_493.85       0.4808          1.4599            1.3740         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)            1_577.09       516.33     2_093.42       0.3081          1.8776            1.7701         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)            1_577.09       919.91     2_497.00       0.4683          1.4792            1.3936         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)            1_577.09       533.03     2_110.12       0.2865          1.9584            1.8357         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)            1_577.09       941.53     2_518.62       0.4338          1.5372            1.4442         5.51
IVF-Binary-768-nl316-sign (self)                       1_577.09     1_449.38     3_026.47       0.3204          1.8449            1.7293         5.51
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
Exhaustive (query)                                        35.82       721.38       757.20       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         35.82     2_259.74     2_295.56       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             780.40       191.33       971.73       0.5723          1.0356            1.0352         2.56
ExhaustiveRaBitQ-rf5 (query)                             780.40       244.09     1_024.49       0.9285          1.0016            1.0005         2.56
ExhaustiveRaBitQ-rf10 (query)                            780.40       282.46     1_062.86       0.9851          1.0003            1.0000         2.56
ExhaustiveRaBitQ-rf20 (query)                            780.40       367.65     1_148.05       0.9986          1.0000            1.0000         2.56
ExhaustiveRaBitQ (self)                                  780.40       924.56     1_704.96       0.9853          1.0003            1.0000         2.56
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_296.48        86.58     1_383.06       0.5810          1.0333            1.0335         2.67
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_296.48       119.71     1_416.20       0.5810          1.0333            1.0335         2.67
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_296.48       154.04     1_450.53       0.5810          1.0333            1.0335         2.67
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_296.48       163.34     1_459.82       0.9861          1.0002            1.0000         2.67
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_296.48       228.45     1_524.93       0.9988          1.0000            1.0000         2.67
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_296.48       195.89     1_492.37       0.9861          1.0002            1.0000         2.67
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_296.48       256.47     1_552.96       0.9988          1.0000            1.0000         2.67
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_296.48       234.04     1_530.52       0.9861          1.0002            1.0000         2.67
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_296.48       293.45     1_589.93       0.9988          1.0000            1.0000         2.67
IVF-RaBitQ-nl158 (self)                                1_296.48       935.24     2_231.73       0.9988          1.0000            1.0000         2.67
IVF-RaBitQ-nl223-np11-rf0 (query)                        849.65       114.42       964.07       0.5930          1.0314            1.0314         2.83
IVF-RaBitQ-nl223-np14-rf0 (query)                        849.65       132.86       982.51       0.5930          1.0314            1.0313         2.83
IVF-RaBitQ-nl223-np21-rf0 (query)                        849.65       178.42     1_028.07       0.5930          1.0314            1.0313         2.83
IVF-RaBitQ-nl223-np11-rf10 (query)                       849.65       184.08     1_033.73       0.9889          1.0002            1.0000         2.83
IVF-RaBitQ-nl223-np11-rf20 (query)                       849.65       239.52     1_089.17       0.9989          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np14-rf10 (query)                       849.65       200.27     1_049.92       0.9890          1.0002            1.0000         2.83
IVF-RaBitQ-nl223-np14-rf20 (query)                       849.65       258.03     1_107.68       0.9990          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np21-rf10 (query)                       849.65       245.22     1_094.87       0.9890          1.0002            1.0000         2.83
IVF-RaBitQ-nl223-np21-rf20 (query)                       849.65       305.78     1_155.43       0.9990          1.0000            1.0000         2.83
IVF-RaBitQ-nl223 (self)                                  849.65       965.13     1_814.78       0.9992          1.0000            1.0000         2.83
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_032.51       137.88     1_170.39       0.6007          1.0301            1.0300         3.06
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_032.51       152.74     1_185.25       0.6007          1.0301            1.0300         3.06
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_032.51       206.10     1_238.61       0.6008          1.0301            1.0300         3.06
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_032.51       226.67     1_259.18       0.9897          1.0002            1.0000         3.06
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_032.51       259.52     1_292.03       0.9991          1.0001            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_032.51       226.30     1_258.81       0.9898          1.0002            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_032.51       281.49     1_313.99       0.9992          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_032.51       268.13     1_300.63       0.9899          1.0002            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_032.51       338.18     1_370.69       0.9993          1.0000            1.0000         3.06
IVF-RaBitQ-nl316 (self)                                1_032.51     1_040.17     2_072.67       0.9993          1.0000            1.0000         3.06
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
Exhaustive (query)                                        68.53     1_297.76     1_366.29       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.53     4_356.72     4_425.24       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           1_456.27       334.57     1_790.84       0.5810          1.0225            1.0225         4.36
ExhaustiveRaBitQ-rf5 (query)                           1_456.27       400.93     1_857.20       0.9276          1.0010            1.0004         4.36
ExhaustiveRaBitQ-rf10 (query)                          1_456.27       440.85     1_897.13       0.9842          1.0002            1.0000         4.36
ExhaustiveRaBitQ-rf20 (query)                          1_456.27       534.56     1_990.83       0.9986          1.0000            1.0000         4.36
ExhaustiveRaBitQ (self)                                1_456.27     1_381.16     2_837.44       0.9846          1.0002            1.0000         4.36
IVF-RaBitQ-nl158-np7-rf0 (query)                       2_534.73       153.02     2_687.75       0.5890          1.0211            1.0216         4.58
IVF-RaBitQ-nl158-np12-rf0 (query)                      2_534.73       207.84     2_742.57       0.5890          1.0211            1.0216         4.58
IVF-RaBitQ-nl158-np17-rf0 (query)                      2_534.73       272.12     2_806.85       0.5890          1.0211            1.0216         4.58
IVF-RaBitQ-nl158-np7-rf10 (query)                      2_534.73       250.28     2_785.02       0.9852          1.0002            1.0000         4.58
IVF-RaBitQ-nl158-np7-rf20 (query)                      2_534.73       341.97     2_876.70       0.9985          1.0000            1.0000         4.58
IVF-RaBitQ-nl158-np12-rf10 (query)                     2_534.73       303.82     2_838.55       0.9852          1.0002            1.0000         4.58
IVF-RaBitQ-nl158-np12-rf20 (query)                     2_534.73       407.39     2_942.12       0.9985          1.0000            1.0000         4.58
IVF-RaBitQ-nl158-np17-rf10 (query)                     2_534.73       353.72     2_888.45       0.9852          1.0002            1.0000         4.58
IVF-RaBitQ-nl158-np17-rf20 (query)                     2_534.73       450.20     2_984.94       0.9985          1.0000            1.0000         4.58
IVF-RaBitQ-nl158 (self)                                2_534.73     1_440.45     3_975.18       0.9988          1.0000            1.0000         4.58
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_600.56       195.36     1_795.92       0.5984          1.0202            1.0206         4.90
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_600.56       228.06     1_828.62       0.5984          1.0202            1.0206         4.90
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_600.56       331.62     1_932.19       0.5984          1.0202            1.0206         4.90
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_600.56       290.07     1_890.64       0.9877          1.0001            1.0000         4.90
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_600.56       392.48     1_993.05       0.9989          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_600.56       324.81     1_925.38       0.9877          1.0001            1.0000         4.90
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_600.56       418.62     2_019.19       0.9990          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_600.56       398.55     1_999.12       0.9877          1.0001            1.0000         4.90
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_600.56       483.31     2_083.88       0.9990          1.0000            1.0000         4.90
IVF-RaBitQ-nl223 (self)                                1_600.56     1_538.24     3_138.81       0.9990          1.0000            1.0000         4.90
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_836.43       245.86     2_082.30       0.6047          1.0193            1.0198         5.35
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_836.43       266.14     2_102.58       0.6047          1.0193            1.0198         5.35
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_836.43       369.98     2_206.42       0.6047          1.0193            1.0198         5.35
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_836.43       338.23     2_174.67       0.9885          1.0001            1.0000         5.35
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_836.43       431.33     2_267.76       0.9991          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_836.43       358.77     2_195.20       0.9885          1.0001            1.0000         5.35
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_836.43       460.64     2_297.07       0.9991          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_836.43       447.34     2_283.78       0.9885          1.0001            1.0000         5.35
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_836.43       535.81     2_372.24       0.9991          1.0000            1.0000         5.35
IVF-RaBitQ-nl316 (self)                                1_836.43     1_708.51     3_544.95       0.9991          1.0000            1.0000         5.35
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
Exhaustive (query)                                       103.65     1_973.14     2_076.79       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        103.65     6_502.58     6_606.23       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           2_004.81       451.82     2_456.63       0.5817          1.0176            1.0178         6.16
ExhaustiveRaBitQ-rf5 (query)                           2_004.81       519.86     2_524.66       0.9265          1.0009            1.0003         6.16
ExhaustiveRaBitQ-rf10 (query)                          2_004.81       583.45     2_588.25       0.9840          1.0001            1.0000         6.16
ExhaustiveRaBitQ-rf20 (query)                          2_004.81       700.55     2_705.35       0.9985          1.0000            1.0000         6.16
ExhaustiveRaBitQ (self)                                2_004.81     1_866.37     3_871.18       0.9839          1.0001            1.0000         6.16
IVF-RaBitQ-nl158-np7-rf0 (query)                       3_558.24       200.32     3_758.56       0.5926          1.0163            1.0168         6.49
IVF-RaBitQ-nl158-np12-rf0 (query)                      3_558.24       283.72     3_841.96       0.5926          1.0163            1.0168         6.49
IVF-RaBitQ-nl158-np17-rf0 (query)                      3_558.24       358.67     3_916.91       0.5926          1.0163            1.0168         6.49
IVF-RaBitQ-nl158-np7-rf10 (query)                      3_558.24       318.20     3_876.44       0.9851          1.0001            1.0000         6.49
IVF-RaBitQ-nl158-np7-rf20 (query)                      3_558.24       437.18     3_995.42       0.9986          1.0000            1.0000         6.49
IVF-RaBitQ-nl158-np12-rf10 (query)                     3_558.24       391.36     3_949.60       0.9851          1.0001            1.0000         6.49
IVF-RaBitQ-nl158-np12-rf20 (query)                     3_558.24       532.15     4_090.39       0.9986          1.0000            1.0000         6.49
IVF-RaBitQ-nl158-np17-rf10 (query)                     3_558.24       472.14     4_030.38       0.9851          1.0001            1.0000         6.49
IVF-RaBitQ-nl158-np17-rf20 (query)                     3_558.24       584.85     4_143.09       0.9986          1.0000            1.0000         6.49
IVF-RaBitQ-nl158 (self)                                3_558.24     1_855.12     5_413.36       0.9987          1.0000            1.0000         6.49
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_912.46       274.67     2_187.14       0.5902          1.0168            1.0168         6.97
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_912.46       314.43     2_226.90       0.5902          1.0168            1.0168         6.97
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_912.46       430.24     2_342.70       0.5902          1.0168            1.0168         6.97
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_912.46       400.11     2_312.57       0.9845          1.0002            1.0000         6.97
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_912.46       495.74     2_408.20       0.9984          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_912.46       426.97     2_339.43       0.9845          1.0002            1.0000         6.97
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_912.46       547.26     2_459.72       0.9985          1.0000            1.0000         6.97
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_912.46       548.19     2_460.65       0.9846          1.0001            1.0000         6.97
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_912.46       646.63     2_559.09       0.9985          1.0000            1.0000         6.97
IVF-RaBitQ-nl223 (self)                                1_912.46     2_064.11     3_976.57       0.9986          1.0000            1.0000         6.97
IVF-RaBitQ-nl316-np15-rf0 (query)                      2_365.14       331.27     2_696.40       0.6029          1.0154            1.0159         7.64
IVF-RaBitQ-nl316-np17-rf0 (query)                      2_365.14       372.66     2_737.79       0.6029          1.0154            1.0159         7.64
IVF-RaBitQ-nl316-np25-rf0 (query)                      2_365.14       491.80     2_856.93       0.6029          1.0154            1.0159         7.64
IVF-RaBitQ-nl316-np15-rf10 (query)                     2_365.14       449.40     2_814.54       0.9875          1.0001            1.0000         7.64
IVF-RaBitQ-nl316-np15-rf20 (query)                     2_365.14       558.88     2_924.01       0.9988          1.0000            1.0000         7.64
IVF-RaBitQ-nl316-np17-rf10 (query)                     2_365.14       479.33     2_844.47       0.9875          1.0001            1.0000         7.64
IVF-RaBitQ-nl316-np17-rf20 (query)                     2_365.14       591.31     2_956.45       0.9988          1.0000            1.0000         7.64
IVF-RaBitQ-nl316-np25-rf10 (query)                     2_365.14       600.48     2_965.61       0.9875          1.0001            1.0000         7.64
IVF-RaBitQ-nl316-np25-rf20 (query)                     2_365.14       707.68     3_072.81       0.9988          1.0000            1.0000         7.64
IVF-RaBitQ-nl316 (self)                                2_365.14     2_317.46     4_682.59       0.9989          1.0000            1.0000         7.64
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
Exhaustive (query)                                        34.30       699.05       733.35       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         34.30     2_293.81     2_328.10       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             857.34       213.50     1_070.84       0.7390          1.0232            1.0221         2.56
ExhaustiveRaBitQ-rf5 (query)                             857.34       258.94     1_116.28       0.9977          1.0000            1.0000         2.56
ExhaustiveRaBitQ-rf10 (query)                            857.34       314.65     1_171.99       1.0000          1.0000            1.0000         2.56
ExhaustiveRaBitQ-rf20 (query)                            857.34       391.80     1_249.14       1.0000          1.0000            1.0000         2.56
ExhaustiveRaBitQ (self)                                  857.34       979.28     1_836.62       1.0000          1.0000            1.0000         2.56
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_294.50        83.37     1_377.88       0.7402          1.0230            1.0219         2.68
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_294.50       112.66     1_407.16       0.7402          1.0230            1.0219         2.68
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_294.50       144.23     1_438.73       0.7402          1.0230            1.0219         2.68
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_294.50       155.53     1_450.03       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_294.50       222.41     1_516.91       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_294.50       186.35     1_480.86       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_294.50       260.34     1_554.84       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_294.50       223.14     1_517.64       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_294.50       291.37     1_585.87       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158 (self)                                1_294.50       930.47     2_224.97       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl223-np11-rf0 (query)                        941.68       110.05     1_051.73       0.7451          1.0220            1.0210         2.84
IVF-RaBitQ-nl223-np14-rf0 (query)                        941.68       134.57     1_076.25       0.7451          1.0220            1.0210         2.84
IVF-RaBitQ-nl223-np21-rf0 (query)                        941.68       176.24     1_117.92       0.7451          1.0220            1.0210         2.84
IVF-RaBitQ-nl223-np11-rf10 (query)                       941.68       182.42     1_124.10       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np11-rf20 (query)                       941.68       252.62     1_194.30       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np14-rf10 (query)                       941.68       203.89     1_145.56       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np14-rf20 (query)                       941.68       262.32     1_204.00       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np21-rf10 (query)                       941.68       254.01     1_195.69       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223-np21-rf20 (query)                       941.68       306.90     1_248.58       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl223 (self)                                  941.68       998.72     1_940.40       1.0000          1.0000            1.0000         2.84
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_110.00       136.47     1_246.46       0.7480          1.0215            1.0205         3.07
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_110.00       147.39     1_257.39       0.7480          1.0215            1.0205         3.07
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_110.00       192.35     1_302.35       0.7480          1.0215            1.0205         3.07
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_110.00       209.68     1_319.68       1.0000          1.0000            1.0000         3.07
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_110.00       276.71     1_386.70       1.0000          1.0000            1.0000         3.07
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_110.00       216.01     1_326.01       1.0000          1.0000            1.0000         3.07
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_110.00       277.60     1_387.60       1.0000          1.0000            1.0000         3.07
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_110.00       266.38     1_376.38       1.0000          1.0000            1.0000         3.07
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_110.00       324.69     1_434.69       1.0000          1.0000            1.0000         3.07
IVF-RaBitQ-nl316 (self)                                1_110.00     1_060.69     2_170.69       1.0000          1.0000            1.0000         3.07
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
Exhaustive (query)                                        69.68     1_325.50     1_395.18       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.68     4_389.16     4_458.84       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           1_600.33       353.52     1_953.85       0.7526          1.0139            1.0132         4.36
ExhaustiveRaBitQ-rf5 (query)                           1_600.33       420.24     2_020.58       0.9982          1.0000            1.0000         4.36
ExhaustiveRaBitQ-rf10 (query)                          1_600.33       479.78     2_080.11       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ-rf20 (query)                          1_600.33       573.68     2_174.02       1.0000          1.0000            1.0000         4.36
ExhaustiveRaBitQ (self)                                1_600.33     1_488.78     3_089.11       1.0000          1.0000            1.0000         4.36
IVF-RaBitQ-nl158-np7-rf0 (query)                       2_486.00       147.16     2_633.16       0.7552          1.0136            1.0129         4.58
IVF-RaBitQ-nl158-np12-rf0 (query)                      2_486.00       203.00     2_689.00       0.7552          1.0136            1.0129         4.58
IVF-RaBitQ-nl158-np17-rf0 (query)                      2_486.00       264.81     2_750.81       0.7552          1.0136            1.0129         4.58
IVF-RaBitQ-nl158-np7-rf10 (query)                      2_486.00       253.02     2_739.03       1.0000          1.0000            1.0000         4.58
IVF-RaBitQ-nl158-np7-rf20 (query)                      2_486.00       350.74     2_836.75       1.0000          1.0000            1.0000         4.58
IVF-RaBitQ-nl158-np12-rf10 (query)                     2_486.00       300.72     2_786.72       1.0000          1.0000            1.0000         4.58
IVF-RaBitQ-nl158-np12-rf20 (query)                     2_486.00       396.66     2_882.67       1.0000          1.0000            1.0000         4.58
IVF-RaBitQ-nl158-np17-rf10 (query)                     2_486.00       351.39     2_837.39       1.0000          1.0000            1.0000         4.58
IVF-RaBitQ-nl158-np17-rf20 (query)                     2_486.00       442.90     2_928.90       1.0000          1.0000            1.0000         4.58
IVF-RaBitQ-nl158 (self)                                2_486.00     1_437.79     3_923.80       1.0000          1.0000            1.0000         4.58
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_582.69       193.16     1_775.85       0.7564          1.0134            1.0127         4.91
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_582.69       226.23     1_808.91       0.7565          1.0134            1.0127         4.91
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_582.69       311.54     1_894.23       0.7565          1.0134            1.0127         4.91
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_582.69       294.77     1_877.46       0.9994          1.0000            1.0000         4.91
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_582.69       388.09     1_970.78       0.9994          1.0000            1.0000         4.91
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_582.69       331.58     1_914.26       1.0000          1.0000            1.0000         4.91
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_582.69       431.63     2_014.31       1.0000          1.0000            1.0000         4.91
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_582.69       401.96     1_984.65       1.0000          1.0000            1.0000         4.91
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_582.69       496.13     2_078.82       1.0000          1.0000            1.0000         4.91
IVF-RaBitQ-nl223 (self)                                1_582.69     1_570.70     3_153.39       1.0000          1.0000            1.0000         4.91
IVF-RaBitQ-nl316-np15-rf0 (query)                      2_013.06       252.00     2_265.06       0.7585          1.0132            1.0125         5.35
IVF-RaBitQ-nl316-np17-rf0 (query)                      2_013.06       261.28     2_274.34       0.7585          1.0132            1.0125         5.35
IVF-RaBitQ-nl316-np25-rf0 (query)                      2_013.06       360.31     2_373.37       0.7585          1.0132            1.0125         5.35
IVF-RaBitQ-nl316-np15-rf10 (query)                     2_013.06       345.56     2_358.62       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np15-rf20 (query)                     2_013.06       441.48     2_454.54       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np17-rf10 (query)                     2_013.06       359.46     2_372.52       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np17-rf20 (query)                     2_013.06       456.48     2_469.54       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np25-rf10 (query)                     2_013.06       456.23     2_469.29       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316-np25-rf20 (query)                     2_013.06       543.67     2_556.73       1.0000          1.0000            1.0000         5.35
IVF-RaBitQ-nl316 (self)                                2_013.06     1_723.97     3_737.03       1.0000          1.0000            1.0000         5.35
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
Exhaustive (query)                                       101.23     1_858.10     1_959.33       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.23     6_268.36     6_369.58       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           2_107.70       485.03     2_592.73       0.7336          1.0115            1.0110         6.16
ExhaustiveRaBitQ-rf5 (query)                           2_107.70       567.94     2_675.64       0.9966          1.0000            1.0000         6.16
ExhaustiveRaBitQ-rf10 (query)                          2_107.70       631.29     2_738.99       1.0000          1.0000            1.0000         6.16
ExhaustiveRaBitQ-rf20 (query)                          2_107.70       770.55     2_878.25       1.0000          1.0000            1.0000         6.16
ExhaustiveRaBitQ (self)                                2_107.70     2_022.66     4_130.36       1.0000          1.0000            1.0000         6.16
IVF-RaBitQ-nl158-np7-rf0 (query)                       3_366.34       194.42     3_560.76       0.7361          1.0112            1.0107         6.50
IVF-RaBitQ-nl158-np12-rf0 (query)                      3_366.34       273.68     3_640.01       0.7361          1.0112            1.0107         6.50
IVF-RaBitQ-nl158-np17-rf0 (query)                      3_366.34       356.29     3_722.63       0.7361          1.0112            1.0107         6.50
IVF-RaBitQ-nl158-np7-rf10 (query)                      3_366.34       318.78     3_685.11       0.9999          1.0000            1.0000         6.50
IVF-RaBitQ-nl158-np7-rf20 (query)                      3_366.34       431.56     3_797.90       1.0000          1.0000            1.0000         6.50
IVF-RaBitQ-nl158-np12-rf10 (query)                     3_366.34       391.26     3_757.60       0.9999          1.0000            1.0000         6.50
IVF-RaBitQ-nl158-np12-rf20 (query)                     3_366.34       507.63     3_873.97       1.0000          1.0000            1.0000         6.50
IVF-RaBitQ-nl158-np17-rf10 (query)                     3_366.34       467.36     3_833.70       0.9999          1.0000            1.0000         6.50
IVF-RaBitQ-nl158-np17-rf20 (query)                     3_366.34       588.06     3_954.40       1.0000          1.0000            1.0000         6.50
IVF-RaBitQ-nl158 (self)                                3_366.34     1_881.29     5_247.63       1.0000          1.0000            1.0000         6.50
IVF-RaBitQ-nl223-np11-rf0 (query)                      2_136.49       273.15     2_409.64       0.7385          1.0110            1.0107         6.98
IVF-RaBitQ-nl223-np14-rf0 (query)                      2_136.49       314.96     2_451.45       0.7385          1.0110            1.0107         6.98
IVF-RaBitQ-nl223-np21-rf0 (query)                      2_136.49       435.63     2_572.12       0.7385          1.0110            1.0107         6.98
IVF-RaBitQ-nl223-np11-rf10 (query)                     2_136.49       400.40     2_536.89       1.0000          1.0000            1.0000         6.98
IVF-RaBitQ-nl223-np11-rf20 (query)                     2_136.49       508.22     2_644.71       1.0000          1.0000            1.0000         6.98
IVF-RaBitQ-nl223-np14-rf10 (query)                     2_136.49       437.25     2_573.74       1.0000          1.0000            1.0000         6.98
IVF-RaBitQ-nl223-np14-rf20 (query)                     2_136.49       545.91     2_682.40       1.0000          1.0000            1.0000         6.98
IVF-RaBitQ-nl223-np21-rf10 (query)                     2_136.49       545.59     2_682.08       1.0000          1.0000            1.0000         6.98
IVF-RaBitQ-nl223-np21-rf20 (query)                     2_136.49       660.21     2_796.71       1.0000          1.0000            1.0000         6.98
IVF-RaBitQ-nl223 (self)                                2_136.49     2_107.87     4_244.36       1.0000          1.0000            1.0000         6.98
IVF-RaBitQ-nl316-np15-rf0 (query)                      2_623.54       339.44     2_962.98       0.7401          1.0109            1.0104         7.66
IVF-RaBitQ-nl316-np17-rf0 (query)                      2_623.54       378.49     3_002.03       0.7401          1.0109            1.0104         7.66
IVF-RaBitQ-nl316-np25-rf0 (query)                      2_623.54       494.20     3_117.74       0.7401          1.0109            1.0104         7.66
IVF-RaBitQ-nl316-np15-rf10 (query)                     2_623.54       455.97     3_079.51       0.9999          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np15-rf20 (query)                     2_623.54       570.44     3_193.98       1.0000          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np17-rf10 (query)                     2_623.54       481.92     3_105.46       0.9999          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np17-rf20 (query)                     2_623.54       595.82     3_219.36       1.0000          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np25-rf10 (query)                     2_623.54       610.84     3_234.38       0.9999          1.0000            1.0000         7.66
IVF-RaBitQ-nl316-np25-rf20 (query)                     2_623.54       721.48     3_345.02       1.0000          1.0000            1.0000         7.66
IVF-RaBitQ-nl316 (self)                                2_623.54     2_305.91     4_929.45       1.0000          1.0000            1.0000         7.66
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
Exhaustive (query)                                        36.62       711.29       747.91       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         36.62     2_343.32     2_379.94       1.0000          1.0000            1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             997.10       241.88     1_238.98       0.8711          1.0279            1.0229         2.56
ExhaustiveRaBitQ-rf5 (query)                             997.10       296.65     1_293.75       1.0000          1.0000            1.0000         2.56
ExhaustiveRaBitQ-rf10 (query)                            997.10       353.81     1_350.90       1.0000          1.0000            1.0000         2.56
ExhaustiveRaBitQ-rf20 (query)                            997.10       449.35     1_446.45       1.0000          1.0000            1.0000         2.56
ExhaustiveRaBitQ (self)                                  997.10     1_145.87     2_142.97       1.0000          1.0000            1.0000         2.56
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_419.29        89.73     1_509.02       0.8744          1.0268            1.0218         2.68
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_419.29       129.42     1_548.72       0.8750          1.0265            1.0215         2.68
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_419.29       174.16     1_593.46       0.8750          1.0265            1.0215         2.68
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_419.29       168.40     1_587.70       0.9976          1.0005            1.0000         2.68
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_419.29       232.52     1_651.81       0.9976          1.0005            1.0000         2.68
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_419.29       211.90     1_631.20       0.9999          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_419.29       276.62     1_695.92       0.9999          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_419.29       246.06     1_665.35       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_419.29       332.83     1_752.13       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl158 (self)                                1_419.29     1_030.74     2_450.04       1.0000          1.0000            1.0000         2.68
IVF-RaBitQ-nl223-np11-rf0 (query)                        803.99       112.01       916.00       0.8844          1.0224            1.0182         2.83
IVF-RaBitQ-nl223-np14-rf0 (query)                        803.99       134.69       938.68       0.8845          1.0224            1.0181         2.83
IVF-RaBitQ-nl223-np21-rf0 (query)                        803.99       183.57       987.56       0.8845          1.0224            1.0181         2.83
IVF-RaBitQ-nl223-np11-rf10 (query)                       803.99       187.74       991.73       0.9994          1.0001            1.0000         2.83
IVF-RaBitQ-nl223-np11-rf20 (query)                       803.99       250.53     1_054.52       0.9994          1.0001            1.0000         2.83
IVF-RaBitQ-nl223-np14-rf10 (query)                       803.99       208.60     1_012.58       0.9999          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np14-rf20 (query)                       803.99       287.70     1_091.69       0.9999          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np21-rf10 (query)                       803.99       268.30     1_072.29       1.0000          1.0000            1.0000         2.83
IVF-RaBitQ-nl223-np21-rf20 (query)                       803.99       333.23     1_137.22       1.0000          1.0000            1.0000         2.83
IVF-RaBitQ-nl223 (self)                                  803.99     1_066.39     1_870.38       1.0000          1.0000            1.0000         2.83
IVF-RaBitQ-nl316-np15-rf0 (query)                        941.48       136.43     1_077.91       0.8902          1.0196            1.0162         3.06
IVF-RaBitQ-nl316-np17-rf0 (query)                        941.48       151.22     1_092.70       0.8902          1.0196            1.0162         3.06
IVF-RaBitQ-nl316-np25-rf0 (query)                        941.48       207.74     1_149.22       0.8902          1.0196            1.0162         3.06
IVF-RaBitQ-nl316-np15-rf10 (query)                       941.48       219.76     1_161.24       0.9997          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np15-rf20 (query)                       941.48       272.92     1_214.40       0.9997          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf10 (query)                       941.48       223.16     1_164.64       0.9998          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np17-rf20 (query)                       941.48       294.73     1_236.21       0.9998          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf10 (query)                       941.48       274.79     1_216.27       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316-np25-rf20 (query)                       941.48       351.35     1_292.83       1.0000          1.0000            1.0000         3.06
IVF-RaBitQ-nl316 (self)                                  941.48     1_106.31     2_047.79       1.0000          1.0000            1.0000         3.06
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
Exhaustive (query)                                        69.85     1_347.60     1_417.45       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.85     4_579.56     4_649.41       1.0000          1.0000            1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           1_871.57       395.52     2_267.09       0.9105          1.0126            1.0093         4.37
ExhaustiveRaBitQ-rf5 (query)                           1_871.57       461.13     2_332.70       1.0000          1.0000            1.0000         4.37
ExhaustiveRaBitQ-rf10 (query)                          1_871.57       526.90     2_398.47       1.0000          1.0000            1.0000         4.37
ExhaustiveRaBitQ-rf20 (query)                          1_871.57       666.94     2_538.50       1.0000          1.0000            1.0000         4.37
ExhaustiveRaBitQ (self)                                1_871.57     1_692.50     3_564.07       1.0000          1.0000            1.0000         4.37
IVF-RaBitQ-nl158-np7-rf0 (query)                       2_750.81       153.09     2_903.90       0.9150          1.0112            1.0083         4.59
IVF-RaBitQ-nl158-np12-rf0 (query)                      2_750.81       222.38     2_973.19       0.9157          1.0109            1.0082         4.59
IVF-RaBitQ-nl158-np17-rf0 (query)                      2_750.81       295.91     3_046.72       0.9157          1.0109            1.0082         4.59
IVF-RaBitQ-nl158-np7-rf10 (query)                      2_750.81       260.24     3_011.06       0.9986          1.0003            1.0000         4.59
IVF-RaBitQ-nl158-np7-rf20 (query)                      2_750.81       364.60     3_115.42       0.9986          1.0003            1.0000         4.59
IVF-RaBitQ-nl158-np12-rf10 (query)                     2_750.81       321.24     3_072.05       0.9999          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np12-rf20 (query)                     2_750.81       428.77     3_179.59       0.9999          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np17-rf10 (query)                     2_750.81       403.88     3_154.69       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158-np17-rf20 (query)                     2_750.81       492.32     3_243.14       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl158 (self)                                2_750.81     1_565.00     4_315.82       1.0000          1.0000            1.0000         4.59
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_437.37       198.25     1_635.62       0.9224          1.0091            1.0066         4.90
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_437.37       236.19     1_673.56       0.9224          1.0090            1.0066         4.90
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_437.37       331.03     1_768.41       0.9225          1.0090            1.0066         4.90
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_437.37       302.18     1_739.55       0.9997          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_437.37       396.81     1_834.19       0.9997          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_437.37       328.75     1_766.12       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_437.37       434.66     1_872.04       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_437.37       440.30     1_877.67       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_437.37       518.82     1_956.19       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl223 (self)                                1_437.37     1_633.75     3_071.12       1.0000          1.0000            1.0000         4.90
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_667.88       258.11     1_925.99       0.9276          1.0078            1.0055         5.36
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_667.88       269.27     1_937.15       0.9276          1.0078            1.0055         5.36
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_667.88       366.57     2_034.45       0.9276          1.0078            1.0055         5.36
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_667.88       345.51     2_013.39       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_667.88       444.07     2_111.96       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_667.88       371.57     2_039.45       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_667.88       459.78     2_127.66       0.9999          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_667.88       468.39     2_136.27       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_667.88       559.14     2_227.02       1.0000          1.0000            1.0000         5.36
IVF-RaBitQ-nl316 (self)                                1_667.88     1_797.30     3_465.18       1.0000          1.0000            1.0000         5.36
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
Exhaustive (query)                                       102.38     1_854.54     1_956.91       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        102.38     6_240.30     6_342.68       1.0000          1.0000            1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           2_499.08       536.11     3_035.19       0.9146          1.0115            1.0083         6.15
ExhaustiveRaBitQ-rf5 (query)                           2_499.08       613.69     3_112.77       1.0000          1.0000            1.0000         6.15
ExhaustiveRaBitQ-rf10 (query)                          2_499.08       686.09     3_185.17       1.0000          1.0000            1.0000         6.15
ExhaustiveRaBitQ-rf20 (query)                          2_499.08       834.35     3_333.42       1.0000          1.0000            1.0000         6.15
ExhaustiveRaBitQ (self)                                2_499.08     2_209.56     4_708.64       1.0000          1.0000            1.0000         6.15
IVF-RaBitQ-nl158-np7-rf0 (query)                       3_760.96       204.23     3_965.19       0.9171          1.0109            1.0079         6.49
IVF-RaBitQ-nl158-np12-rf0 (query)                      3_760.96       300.51     4_061.47       0.9173          1.0109            1.0079         6.49
IVF-RaBitQ-nl158-np17-rf0 (query)                      3_760.96       398.70     4_159.65       0.9173          1.0109            1.0079         6.49
IVF-RaBitQ-nl158-np7-rf10 (query)                      3_760.96       329.87     4_090.83       0.9995          1.0001            1.0000         6.49
IVF-RaBitQ-nl158-np7-rf20 (query)                      3_760.96       438.87     4_199.83       0.9995          1.0001            1.0000         6.49
IVF-RaBitQ-nl158-np12-rf10 (query)                     3_760.96       425.40     4_186.36       1.0000          1.0000            1.0000         6.49
IVF-RaBitQ-nl158-np12-rf20 (query)                     3_760.96       536.33     4_297.29       1.0000          1.0000            1.0000         6.49
IVF-RaBitQ-nl158-np17-rf10 (query)                     3_760.96       515.95     4_276.91       1.0000          1.0000            1.0000         6.49
IVF-RaBitQ-nl158-np17-rf20 (query)                     3_760.96       640.85     4_401.80       1.0000          1.0000            1.0000         6.49
IVF-RaBitQ-nl158 (self)                                3_760.96     2_007.22     5_768.18       1.0000          1.0000            1.0000         6.49
IVF-RaBitQ-nl223-np11-rf0 (query)                      1_879.20       276.23     2_155.44       0.9220          1.0094            1.0067         6.96
IVF-RaBitQ-nl223-np14-rf0 (query)                      1_879.20       326.68     2_205.88       0.9220          1.0094            1.0067         6.96
IVF-RaBitQ-nl223-np21-rf0 (query)                      1_879.20       483.74     2_362.94       0.9220          1.0094            1.0067         6.96
IVF-RaBitQ-nl223-np11-rf10 (query)                     1_879.20       406.92     2_286.12       0.9999          1.0000            1.0000         6.96
IVF-RaBitQ-nl223-np11-rf20 (query)                     1_879.20       502.69     2_381.89       0.9999          1.0000            1.0000         6.96
IVF-RaBitQ-nl223-np14-rf10 (query)                     1_879.20       449.54     2_328.74       1.0000          1.0000            1.0000         6.96
IVF-RaBitQ-nl223-np14-rf20 (query)                     1_879.20       570.13     2_449.34       1.0000          1.0000            1.0000         6.96
IVF-RaBitQ-nl223-np21-rf10 (query)                     1_879.20       561.66     2_440.86       1.0000          1.0000            1.0000         6.96
IVF-RaBitQ-nl223-np21-rf20 (query)                     1_879.20       677.13     2_556.33       1.0000          1.0000            1.0000         6.96
IVF-RaBitQ-nl223 (self)                                1_879.20     2_163.26     4_042.47       1.0000          1.0000            1.0000         6.96
IVF-RaBitQ-nl316-np15-rf0 (query)                      2_228.99       341.64     2_570.63       0.9267          1.0082            1.0057         7.64
IVF-RaBitQ-nl316-np17-rf0 (query)                      2_228.99       373.39     2_602.38       0.9267          1.0082            1.0057         7.64
IVF-RaBitQ-nl316-np25-rf0 (query)                      2_228.99       510.07     2_739.06       0.9267          1.0082            1.0057         7.64
IVF-RaBitQ-nl316-np15-rf10 (query)                     2_228.99       454.54     2_683.53       1.0000          1.0000            1.0000         7.64
IVF-RaBitQ-nl316-np15-rf20 (query)                     2_228.99       581.49     2_810.48       1.0000          1.0000            1.0000         7.64
IVF-RaBitQ-nl316-np17-rf10 (query)                     2_228.99       486.40     2_715.39       1.0000          1.0000            1.0000         7.64
IVF-RaBitQ-nl316-np17-rf20 (query)                     2_228.99       604.37     2_833.35       1.0000          1.0000            1.0000         7.64
IVF-RaBitQ-nl316-np25-rf10 (query)                     2_228.99       621.34     2_850.33       1.0000          1.0000            1.0000         7.64
IVF-RaBitQ-nl316-np25-rf20 (query)                     2_228.99       750.82     2_979.81       1.0000          1.0000            1.0000         7.64
IVF-RaBitQ-nl316 (self)                                2_228.99     2_384.00     4_612.99       1.0000          1.0000            1.0000         7.64
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
Exhaustive (query)                                        32.76       703.95       736.71       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.76     2_309.35     2_342.11       1.0000          1.0000            1.0000        48.83
QG-d32-l64-ef15 (query)                                2_496.00       296.44     2_792.45       0.9907          1.0029            1.0000       116.35
QG-d32-l64-ef30 (query)                                2_496.00       413.26     2_909.27       0.9956          1.0022            1.0000       116.35
QG-d32-l64-ef60 (query)                                2_496.00       614.27     3_110.28       0.9975          1.0015            1.0000       116.35
QG-d32-l64-ef120 (query)                               2_496.00       857.24     3_353.24       0.9982          1.0013            1.0000       116.35
QG-d32-l64 (self)                                      2_496.00     1_899.84     4_395.85       0.9975          1.0015            1.0000       116.35
QG-d32-l128-ef15 (query)                               2_806.73       286.40     3_093.13       0.9873          1.3969            1.0000       116.35
QG-d32-l128-ef30 (query)                               2_806.73       412.57     3_219.30       0.9938          1.1299            1.0000       116.35
QG-d32-l128-ef60 (query)                               2_806.73       603.48     3_410.21       0.9961          1.0192            1.0000       116.35
QG-d32-l128-ef120 (query)                              2_806.73       860.69     3_667.42       0.9975          1.0068            1.0000       116.35
QG-d32-l128 (self)                                     2_806.73     1_911.41     4_718.14       0.9966          1.0205            1.0000       116.35
QG-d64-l64-ef15 (query)                                8_589.34       630.50     9_219.83       0.9991          1.0003            1.0000       183.49
QG-d64-l64-ef30 (query)                                8_589.34       859.18     9_448.52       0.9996          1.0002            1.0000       183.49
QG-d64-l64-ef60 (query)                                8_589.34     1_193.77     9_783.10       0.9997          1.0002            1.0000       183.49
QG-d64-l64-ef120 (query)                               8_589.34     1_711.34    10_300.67       0.9997          1.0002            1.0000       183.49
QG-d64-l64 (self)                                      8_589.34     3_957.26    12_546.59       0.9997          1.0002            1.0000       183.49
QG-d64-l128-ef15 (query)                               9_010.45       649.63     9_660.08       0.9993          1.0002            1.0000       183.49
QG-d64-l128-ef30 (query)                               9_010.45       874.30     9_884.75       0.9998          1.0002            1.0000       183.49
QG-d64-l128-ef60 (query)                               9_010.45     1_214.54    10_224.98       0.9998          1.0001            1.0000       183.49
QG-d64-l128-ef120 (query)                              9_010.45     1_735.96    10_746.41       0.9998          1.0001            1.0000       183.49
QG-d64-l128 (self)                                     9_010.45     3_974.65    12_985.10       0.9998          1.0002            1.0000       183.49
HnswRaBitQ-m16-ex1-ef15 (query)                        1_550.38       609.45     2_159.82       0.7657          1.7592            1.0084        10.85
HnswRaBitQ-m16-ex1-ef30 (query)                        1_550.38       878.70     2_429.08       0.7751          1.0141            1.0080        10.85
HnswRaBitQ-m16-ex1-ef60 (query)                        1_550.38     1_244.29     2_794.67       0.7769          1.0096            1.0080        10.85
HnswRaBitQ-m16-ex1-ef120 (query)                       1_550.38     1_721.60     3_271.98       0.7773          1.0088            1.0080        10.85
HnswRaBitQ-m16-ex1 (self)                              1_550.38     6_795.12     8_345.50       0.6967          1.0182            1.0161        10.85
HnswRaBitQ-m16-ex3-ef15 (query)                        1_667.79       600.60     2_268.39       0.9016          1.0092            1.0010        13.90
HnswRaBitQ-m16-ex3-ef30 (query)                        1_667.79       868.16     2_535.95       0.9202          1.0039            1.0006        13.90
HnswRaBitQ-m16-ex3-ef60 (query)                        1_667.79     1_244.69     2_912.48       0.9273          1.0017            1.0005        13.90
HnswRaBitQ-m16-ex3-ef120 (query)                       1_667.79     1_719.61     3_387.40       0.9288          1.0013            1.0005        13.90
HnswRaBitQ-m16-ex3 (self)                              1_667.79     6_742.98     8_410.77       0.9027          1.0025            1.0012        13.90
HnswRaBitQ-m16-ex5-ef15 (query)                        1_740.87       605.27     2_346.14       0.9389          1.0132            1.0001        16.95
HnswRaBitQ-m16-ex5-ef30 (query)                        1_740.87       878.64     2_619.51       0.9647          1.0065            1.0000        16.95
HnswRaBitQ-m16-ex5-ef60 (query)                        1_740.87     1_253.23     2_994.10       0.9751          1.0012            1.0000        16.95
HnswRaBitQ-m16-ex5-ef120 (query)                       1_740.87     1_728.86     3_469.73       0.9774          1.0007            1.0000        16.95
HnswRaBitQ-m16-ex5 (self)                              1_740.87     6_772.51     8_513.38       0.9690          1.0018            1.0000        16.95
HnswRaBitQ-m16-ex8-ef15 (query)                        2_562.05       617.36     3_179.41       0.9493          1.0791            1.0000        21.53
HnswRaBitQ-m16-ex8-ef30 (query)                        2_562.05       909.56     3_471.60       0.9808          1.0706            1.0000        21.53
HnswRaBitQ-m16-ex8-ef60 (query)                        2_562.05     1_282.69     3_844.74       0.9931          1.0014            1.0000        21.53
HnswRaBitQ-m16-ex8-ef120 (query)                       2_562.05     1_764.15     4_326.20       0.9962          1.0005            1.0000        21.53
HnswRaBitQ-m16-ex8 (self)                              2_562.05     6_892.12     9_454.17       0.9927          1.0532            1.0000        21.53
HnswRaBitQ-m32-ex1-ef15 (query)                        1_595.32       789.85     2_385.17       0.7755          1.0172            1.0079        16.95
HnswRaBitQ-m32-ex1-ef30 (query)                        1_595.32     1_093.57     2_688.89       0.7768          1.0143            1.0080        16.95
HnswRaBitQ-m32-ex1-ef60 (query)                        1_595.32     1_495.39     3_090.71       0.7777          1.0087            1.0080        16.95
HnswRaBitQ-m32-ex1-ef120 (query)                       1_595.32     1_967.37     3_562.69       0.7777          1.0085            1.0080        16.95
HnswRaBitQ-m32-ex1 (self)                              1_595.32     8_261.46     9_856.77       0.6964          1.0174            1.0162        16.95
HnswRaBitQ-m32-ex3-ef15 (query)                        1_666.05       829.16     2_495.21       0.9170          1.0117            1.0007        20.00
HnswRaBitQ-m32-ex3-ef30 (query)                        1_666.05     1_122.66     2_788.71       0.9262          1.0063            1.0005        20.00
HnswRaBitQ-m32-ex3-ef60 (query)                        1_666.05     1_531.68     3_197.73       0.9287          1.0012            1.0005        20.00
HnswRaBitQ-m32-ex3-ef120 (query)                       1_666.05     1_961.15     3_627.20       0.9293          1.0009            1.0005        20.00
HnswRaBitQ-m32-ex3 (self)                              1_666.05     8_256.06     9_922.11       0.9041          1.0016            1.0012        20.00
HnswRaBitQ-m32-ex5-ef15 (query)                        1_762.01       790.88     2_552.89       0.9601          1.0324            1.0000        23.06
HnswRaBitQ-m32-ex5-ef30 (query)                        1_762.01     1_105.27     2_867.28       0.9738          1.0013            1.0000        23.06
HnswRaBitQ-m32-ex5-ef60 (query)                        1_762.01     1_507.10     3_269.11       0.9775          1.0003            1.0000        23.06
HnswRaBitQ-m32-ex5-ef120 (query)                       1_762.01     1_988.10     3_750.11       0.9783          1.0001            1.0000        23.06
HnswRaBitQ-m32-ex5 (self)                              1_762.01     8_254.95    10_016.96       0.9712          1.0003            1.0000        23.06
HnswRaBitQ-m32-ex8-ef15 (query)                        2_599.57       801.75     3_401.32       0.9742          1.0055            1.0000        27.63
HnswRaBitQ-m32-ex8-ef30 (query)                        2_599.57     1_128.57     3_728.14       0.9909          1.0018            1.0000        27.63
HnswRaBitQ-m32-ex8-ef60 (query)                        2_599.57     1_537.43     4_137.00       0.9958          1.0007            1.0000        27.63
HnswRaBitQ-m32-ex8-ef120 (query)                       2_599.57     2_034.61     4_634.18       0.9970          1.0001            1.0000        27.63
HnswRaBitQ-m32-ex8 (self)                              2_599.57     8_425.33    11_024.90       0.9950          1.0004            1.0000        27.63
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
Exhaustive (query)                                        70.19     1_305.10     1_375.29       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         70.19     4_389.65     4_459.83       1.0000          1.0000            1.0000        97.66
QG-d32-l64-ef15 (query)                                4_765.74       499.35     5_265.08       0.9878          1.0033            1.0000       214.01
QG-d32-l64-ef30 (query)                                4_765.74       666.54     5_432.28       0.9931          1.0031            1.0000       214.01
QG-d32-l64-ef60 (query)                                4_765.74       898.55     5_664.29       0.9953          1.0030            1.0000       214.01
QG-d32-l64-ef120 (query)                               4_765.74     1_229.65     5_995.38       0.9963          1.0029            1.0000       214.01
QG-d32-l64 (self)                                      4_765.74     2_931.33     7_697.06       0.9952          1.0030            1.0000       214.01
QG-d32-l128-ef15 (query)                               5_481.37       493.09     5_974.46       0.9884          1.0031            1.0000       214.01
QG-d32-l128-ef30 (query)                               5_481.37       658.70     6_140.06       0.9935          1.0029            1.0000       214.01
QG-d32-l128-ef60 (query)                               5_481.37       888.83     6_370.19       0.9956          1.0027            1.0000       214.01
QG-d32-l128-ef120 (query)                              5_481.37     1_212.72     6_694.09       0.9965          1.0027            1.0000       214.01
QG-d32-l128 (self)                                     5_481.37     2_876.96     8_358.33       0.9955          1.0029            1.0000       214.01
QG-d64-l64-ef15 (query)                               18_033.31       995.52    19_028.83       0.9982          1.0009            1.0000       329.97
QG-d64-l64-ef30 (query)                               18_033.31     1_284.30    19_317.62       0.9988          1.0009            1.0000       329.97
QG-d64-l64-ef60 (query)                               18_033.31     1_684.74    19_718.05       0.9990          1.0009            1.0000       329.97
QG-d64-l64-ef120 (query)                              18_033.31     2_266.27    20_299.58       0.9991          1.0009            1.0000       329.97
QG-d64-l64 (self)                                     18_033.31     5_532.23    23_565.54       0.9990          1.0008            1.0000       329.97
QG-d64-l128-ef15 (query)                              18_287.76       995.02    19_282.77       0.9981          1.0012            1.0000       329.97
QG-d64-l128-ef30 (query)                              18_287.76     1_282.09    19_569.84       0.9986          1.0012            1.0000       329.97
QG-d64-l128-ef60 (query)                              18_287.76     1_682.37    19_970.13       0.9989          1.0010            1.0000       329.97
QG-d64-l128-ef120 (query)                             18_287.76     2_401.68    20_689.44       0.9990          1.0008            1.0000       329.97
QG-d64-l128 (self)                                    18_287.76     5_507.76    23_795.52       0.9989          1.0009            1.0000       329.97
HnswRaBitQ-m16-ex1-ef15 (query)                        2_807.27     1_318.73     4_126.00       0.7598          1.2901            1.0058        14.15
HnswRaBitQ-m16-ex1-ef30 (query)                        2_807.27     1_881.91     4_689.19       0.7719          1.0809            1.0055        14.15
HnswRaBitQ-m16-ex1-ef60 (query)                        2_807.27     2_716.72     5_523.99       0.7762          1.0065            1.0055        14.15
HnswRaBitQ-m16-ex1-ef120 (query)                       2_807.27     3_631.45     6_438.72       0.7770          1.0059            1.0055        14.15
HnswRaBitQ-m16-ex1 (self)                              2_807.27    14_345.09    17_152.36       0.6932          1.0128            1.0110        14.15
HnswRaBitQ-m16-ex3-ef15 (query)                        3_002.34     1_300.83     4_303.17       0.8912          1.0192            1.0008        20.25
HnswRaBitQ-m16-ex3-ef30 (query)                        3_002.34     1_869.14     4_871.48       0.9152          1.0058            1.0005        20.25
HnswRaBitQ-m16-ex3-ef60 (query)                        3_002.34     2_742.08     5_744.42       0.9250          1.0011            1.0004        20.25
HnswRaBitQ-m16-ex3-ef120 (query)                       3_002.34     3_628.58     6_630.92       0.9274          1.0008            1.0004        20.25
HnswRaBitQ-m16-ex3 (self)                              3_002.34    14_352.71    17_355.05       0.8999          1.0016            1.0008        20.25
HnswRaBitQ-m16-ex5-ef15 (query)                        3_178.66     1_290.12     4_468.78       0.9230          2.1200            1.0001        26.36
HnswRaBitQ-m16-ex5-ef30 (query)                        3_178.66     1_876.10     5_054.75       0.9572          1.5743            1.0000        26.36
HnswRaBitQ-m16-ex5-ef60 (query)                        3_178.66     2_680.37     5_859.03       0.9725          1.0014            1.0000        26.36
HnswRaBitQ-m16-ex5-ef120 (query)                       3_178.66     3_668.50     6_847.15       0.9767          1.0005            1.0000        26.36
HnswRaBitQ-m16-ex5 (self)                              3_178.66    15_389.15    18_567.80       0.9652          1.0015            1.0000        26.36
HnswRaBitQ-m16-ex8-ef15 (query)                        4_677.62     1_346.64     6_024.26       0.9316          3.2384            1.0000        35.51
HnswRaBitQ-m16-ex8-ef30 (query)                        4_677.62     1_925.50     6_603.12       0.9703          3.2265            1.0000        35.51
HnswRaBitQ-m16-ex8-ef60 (query)                        4_677.62     2_726.15     7_403.77       0.9885          2.5760            1.0000        35.51
HnswRaBitQ-m16-ex8-ef120 (query)                       4_677.62     3_701.03     8_378.64       0.9952          1.1221            1.0000        35.51
HnswRaBitQ-m16-ex8 (self)                              4_677.62    14_536.13    19_213.75       0.9871          2.4604            1.0000        35.51
HnswRaBitQ-m32-ex1-ef15 (query)                        2_911.37     1_755.81     4_667.18       0.7732          1.0233            1.0055        20.25
HnswRaBitQ-m32-ex1-ef30 (query)                        2_911.37     2_429.21     5_340.58       0.7761          1.0082            1.0054        20.25
HnswRaBitQ-m32-ex1-ef60 (query)                        2_911.37     3_289.50     6_200.87       0.7767          1.0059            1.0054        20.25
HnswRaBitQ-m32-ex1-ef120 (query)                       2_911.37     4_224.83     7_136.20       0.7770          1.0058            1.0054        20.25
HnswRaBitQ-m32-ex1 (self)                              2_911.37    17_677.62    20_588.99       0.6926          1.0127            1.0110        20.25
HnswRaBitQ-m32-ex3-ef15 (query)                        3_081.60     1_754.86     4_836.46       0.9126          1.0075            1.0005        26.36
HnswRaBitQ-m32-ex3-ef30 (query)                        3_081.60     2_455.09     5_536.69       0.9237          1.0014            1.0004        26.36
HnswRaBitQ-m32-ex3-ef60 (query)                        3_081.60     3_345.14     6_426.74       0.9270          1.0008            1.0004        26.36
HnswRaBitQ-m32-ex3-ef120 (query)                       3_081.60     4_264.87     7_346.46       0.9278          1.0006            1.0004        26.36
HnswRaBitQ-m32-ex3 (self)                              3_081.60    17_740.63    20_822.23       0.9017          1.0012            1.0008        26.36
HnswRaBitQ-m32-ex5-ef15 (query)                        3_276.51     1_844.47     5_120.97       0.9543          1.0077            1.0000        32.46
HnswRaBitQ-m32-ex5-ef30 (query)                        3_276.51     2_474.15     5_750.66       0.9711          1.0013            1.0000        32.46
HnswRaBitQ-m32-ex5-ef60 (query)                        3_276.51     3_384.86     6_661.37       0.9761          1.0005            1.0000        32.46
HnswRaBitQ-m32-ex5-ef120 (query)                       3_276.51     4_286.75     7_563.26       0.9773          1.0001            1.0000        32.46
HnswRaBitQ-m32-ex5 (self)                              3_276.51    18_005.98    21_282.48       0.9687          1.0006            1.0000        32.46
HnswRaBitQ-m32-ex8-ef15 (query)                        4_798.03     1_851.13     6_649.17       0.9669          1.1384            1.0000        41.62
HnswRaBitQ-m32-ex8-ef30 (query)                        4_798.03     2_528.62     7_326.65       0.9870          1.1178            1.0000        41.62
HnswRaBitQ-m32-ex8-ef60 (query)                        4_798.03     3_395.09     8_193.12       0.9949          1.0004            1.0000        41.62
HnswRaBitQ-m32-ex8-ef120 (query)                       4_798.03     4_328.36     9_126.40       0.9965          1.0001            1.0000        41.62
HnswRaBitQ-m32-ex8 (self)                              4_798.03    18_241.48    23_039.51       0.9937          1.0003            1.0000        41.62
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
Exhaustive (query)                                       101.12     1_883.08     1_984.20       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        101.12     6_332.48     6_433.60       1.0000          1.0000            1.0000       146.48
QG-d32-l64-ef15 (query)                                7_032.70       627.31     7_660.01       0.9853          1.0054            1.0000       311.66
QG-d32-l64-ef30 (query)                                7_032.70       835.83     7_868.53       0.9909          1.0052            1.0000       311.66
QG-d32-l64-ef60 (query)                                7_032.70     1_095.48     8_128.18       0.9936          1.0051            1.0000       311.66
QG-d32-l64-ef120 (query)                               7_032.70     1_469.51     8_502.20       0.9948          1.0046            1.0000       311.66
QG-d32-l64 (self)                                      7_032.70     3_607.25    10_639.95       0.9934          1.0053            1.0000       311.66
QG-d32-l128-ef15 (query)                               7_871.34       627.55     8_498.89       0.9864          1.0040            1.0000       311.66
QG-d32-l128-ef30 (query)                               7_871.34       830.57     8_701.92       0.9919          1.0038            1.0000       311.66
QG-d32-l128-ef60 (query)                               7_871.34     1_105.03     8_976.37       0.9945          1.0036            1.0000       311.66
QG-d32-l128-ef120 (query)                              7_871.34     1_481.27     9_352.62       0.9956          1.0035            1.0000       311.66
QG-d32-l128 (self)                                     7_871.34     3_578.29    11_449.63       0.9946          1.0035            1.0000       311.66
QG-d64-l64-ef15 (query)                               26_815.94     1_331.49    28_147.42       0.9970          1.0030            1.0000       476.46
QG-d64-l64-ef30 (query)                               26_815.94     1_682.56    28_498.50       0.9977          1.0030            1.0000       476.46
QG-d64-l64-ef60 (query)                               26_815.94     2_115.72    28_931.65       0.9980          1.0029            1.0000       476.46
QG-d64-l64-ef120 (query)                              26_815.94     2_767.80    29_583.73       0.9981          1.0029            1.0000       476.46
QG-d64-l64 (self)                                     26_815.94     6_977.11    33_793.05       0.9978          1.0030            1.0000       476.46
QG-d64-l128-ef15 (query)                              27_253.36     1_326.07    28_579.42       0.9973          1.0014            1.0000       476.46
QG-d64-l128-ef30 (query)                              27_253.36     1_671.12    28_924.48       0.9980          1.0014            1.0000       476.46
QG-d64-l128-ef60 (query)                              27_253.36     2_114.54    29_367.90       0.9982          1.0014            1.0000       476.46
QG-d64-l128-ef120 (query)                             27_253.36     2_777.39    30_030.75       0.9983          1.0013            1.0000       476.46
QG-d64-l128 (self)                                    27_253.36     7_036.63    34_289.99       0.9984          1.0013            1.0000       476.46
HnswRaBitQ-m16-ex1-ef15 (query)                        4_092.85     1_987.51     6_080.36       0.7638          1.0140            1.0046        17.45
HnswRaBitQ-m16-ex1-ef30 (query)                        4_092.85     2_873.22     6_966.07       0.7731          1.0081            1.0044        17.45
HnswRaBitQ-m16-ex1-ef60 (query)                        4_092.85     4_091.12     8_183.97       0.7766          1.0050            1.0043        17.45
HnswRaBitQ-m16-ex1-ef120 (query)                       4_092.85     5_549.63     9_642.49       0.7772          1.0048            1.0043        17.45
HnswRaBitQ-m16-ex1 (self)                              4_092.85    22_055.91    26_148.77       0.6925          1.0104            1.0087        17.45
HnswRaBitQ-m16-ex3-ef15 (query)                        4_292.64     1_991.35     6_283.99       0.8889          1.0816            1.0007        26.61
HnswRaBitQ-m16-ex3-ef30 (query)                        4_292.64     2_878.78     7_171.42       0.9137          1.0343            1.0004        26.61
HnswRaBitQ-m16-ex3-ef60 (query)                        4_292.64     4_196.75     8_489.39       0.9247          1.0010            1.0003        26.61
HnswRaBitQ-m16-ex3-ef120 (query)                       4_292.64     5_566.74     9_859.38       0.9274          1.0006            1.0003        26.61
HnswRaBitQ-m16-ex3 (self)                              4_292.64    21_913.71    26_206.35       0.8988          1.0019            1.0007        26.61
HnswRaBitQ-m16-ex5-ef15 (query)                        4_577.29     1_996.61     6_573.90       0.9198          1.0452            1.0001        35.76
HnswRaBitQ-m16-ex5-ef30 (query)                        4_577.29     2_895.63     7_472.92       0.9552          1.0161            1.0000        35.76
HnswRaBitQ-m16-ex5-ef60 (query)                        4_577.29     4_129.88     8_707.17       0.9717          1.0029            1.0000        35.76
HnswRaBitQ-m16-ex5-ef120 (query)                       4_577.29     5_638.40    10_215.69       0.9761          1.0007            1.0000        35.76
HnswRaBitQ-m16-ex5 (self)                              4_577.29    22_075.96    26_653.26       0.9639          1.0022            1.0000        35.76
HnswRaBitQ-m16-ex8-ef15 (query)                        6_824.73     2_020.55     8_845.28       0.9299          1.0334            1.0000        49.50
HnswRaBitQ-m16-ex8-ef30 (query)                        6_824.73     2_935.68     9_760.41       0.9697          1.0201            1.0000        49.50
HnswRaBitQ-m16-ex8-ef60 (query)                        6_824.73     4_155.62    10_980.35       0.9888          1.0105            1.0000        49.50
HnswRaBitQ-m16-ex8-ef120 (query)                       6_824.73     5_636.77    12_461.50       0.9945          1.0054            1.0000        49.50
HnswRaBitQ-m16-ex8 (self)                              6_824.73    22_331.95    29_156.68       0.9881          1.0095            1.0000        49.50
HnswRaBitQ-m32-ex1-ef15 (query)                        4_103.15     2_723.50     6_826.65       0.7714          1.0944            1.0043        23.56
HnswRaBitQ-m32-ex1-ef30 (query)                        4_103.15     3_799.87     7_903.02       0.7756          1.0260            1.0042        23.56
HnswRaBitQ-m32-ex1-ef60 (query)                        4_103.15     5_111.52     9_214.66       0.7772          1.0048            1.0043        23.56
HnswRaBitQ-m32-ex1-ef120 (query)                       4_103.15     6_487.09    10_590.24       0.7777          1.0045            1.0042        23.56
HnswRaBitQ-m32-ex1 (self)                              4_103.15    27_544.79    31_647.94       0.6922          1.0096            1.0088        23.56
HnswRaBitQ-m32-ex3-ef15 (query)                        4_365.94     2_745.60     7_111.54       0.9109          1.0107            1.0004        32.71
HnswRaBitQ-m32-ex3-ef30 (query)                        4_365.94     3_840.07     8_206.00       0.9232          1.0010            1.0003        32.71
HnswRaBitQ-m32-ex3-ef60 (query)                        4_365.94     5_208.81     9_574.74       0.9268          1.0008            1.0003        32.71
HnswRaBitQ-m32-ex3-ef120 (query)                       4_365.94     6_574.31    10_940.25       0.9278          1.0005            1.0003        32.71
HnswRaBitQ-m32-ex3 (self)                              4_365.94    27_588.29    31_954.23       0.9007          1.0011            1.0006        32.71
HnswRaBitQ-m32-ex5-ef15 (query)                        4_677.64     2_807.89     7_485.53       0.9515          1.0228            1.0000        41.87
HnswRaBitQ-m32-ex5-ef30 (query)                        4_677.64     3_883.88     8_561.52       0.9694          1.0043            1.0000        41.87
HnswRaBitQ-m32-ex5-ef60 (query)                        4_677.64     5_212.30     9_889.94       0.9756          1.0003            1.0000        41.87
HnswRaBitQ-m32-ex5-ef120 (query)                       4_677.64     6_579.03    11_256.68       0.9770          1.0001            1.0000        41.87
HnswRaBitQ-m32-ex5 (self)                              4_677.64    27_802.56    32_480.20       0.9675          1.0004            1.0000        41.87
HnswRaBitQ-m32-ex8-ef15 (query)                        6_901.61     2_796.29     9_697.90       0.9660          1.0409            1.0000        55.60
HnswRaBitQ-m32-ex8-ef30 (query)                        6_901.61     3_896.46    10_798.08       0.9874          1.0017            1.0000        55.60
HnswRaBitQ-m32-ex8-ef60 (query)                        6_901.61     5_260.45    12_162.07       0.9945          1.0001            1.0000        55.60
HnswRaBitQ-m32-ex8-ef120 (query)                       6_901.61     6_658.76    13_560.38       0.9961          1.0001            1.0000        55.60
HnswRaBitQ-m32-ex8 (self)                              6_901.61    28_011.35    34_912.97       0.9933          1.0002            1.0000        55.60
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
Exhaustive (query)                                        32.85       687.76       720.61       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.85     2_294.70     2_327.55       1.0000          1.0000            1.0000        48.83
QG-d32-l64-ef15 (query)                                2_843.24        78.97     2_922.21       0.9709          1.0010            1.0000       116.35
QG-d32-l64-ef30 (query)                                2_843.24       140.86     2_984.11       0.9983          1.0001            1.0000       116.35
QG-d32-l64-ef60 (query)                                2_843.24       248.86     3_092.10       0.9999          1.0000            1.0000       116.35
QG-d32-l64-ef120 (query)                               2_843.24       471.84     3_315.09       1.0000          1.0000            1.0000       116.35
QG-d32-l64 (self)                                      2_843.24       751.78     3_595.03       0.9999          1.0000            1.0000       116.35
QG-d32-l128-ef15 (query)                               3_279.79        78.77     3_358.56       0.9712          1.0009            1.0000       116.35
QG-d32-l128-ef30 (query)                               3_279.79       133.64     3_413.43       0.9985          1.0000            1.0000       116.35
QG-d32-l128-ef60 (query)                               3_279.79       249.18     3_528.98       1.0000          1.0000            1.0000       116.35
QG-d32-l128-ef120 (query)                              3_279.79       476.35     3_756.15       1.0000          1.0000            1.0000       116.35
QG-d32-l128 (self)                                     3_279.79       757.10     4_036.89       1.0000          1.0000            1.0000       116.35
QG-d64-l64-ef15 (query)                                5_056.51       116.28     5_172.80       0.9901          1.0002            1.0000       183.49
QG-d64-l64-ef30 (query)                                5_056.51       209.09     5_265.60       0.9999          1.0000            1.0000       183.49
QG-d64-l64-ef60 (query)                                5_056.51       400.55     5_457.07       1.0000          1.0000            1.0000       183.49
QG-d64-l64-ef120 (query)                               5_056.51       764.28     5_820.80       1.0000          1.0000            1.0000       183.49
QG-d64-l64 (self)                                      5_056.51     1_221.11     6_277.62       1.0000          1.0000            1.0000       183.49
QG-d64-l128-ef15 (query)                               5_771.70       114.76     5_886.46       0.9903          1.0002            1.0000       183.49
QG-d64-l128-ef30 (query)                               5_771.70       214.86     5_986.56       0.9999          1.0000            1.0000       183.49
QG-d64-l128-ef60 (query)                               5_771.70       402.79     6_174.48       1.0000          1.0000            1.0000       183.49
QG-d64-l128-ef120 (query)                              5_771.70       776.66     6_548.36       1.0000          1.0000            1.0000       183.49
QG-d64-l128 (self)                                     5_771.70     1_243.63     7_015.33       1.0000          1.0000            1.0000       183.49
HnswRaBitQ-m16-ex1-ef15 (query)                        1_638.59       641.64     2_280.23       0.8532          1.0073            1.0059        10.85
HnswRaBitQ-m16-ex1-ef30 (query)                        1_638.59       970.08     2_608.68       0.8657          1.0057            1.0050        10.85
HnswRaBitQ-m16-ex1-ef60 (query)                        1_638.59     1_448.25     3_086.85       0.8687          1.0053            1.0047        10.85
HnswRaBitQ-m16-ex1-ef120 (query)                       1_638.59     2_021.26     3_659.85       0.8691          1.0053            1.0047        10.85
HnswRaBitQ-m16-ex1 (self)                              1_638.59     8_219.27     9_857.87       0.8312          1.0109            1.0098        10.85
HnswRaBitQ-m16-ex3-ef15 (query)                        1_741.41       663.91     2_405.32       0.9276          1.5029            1.0008        13.90
HnswRaBitQ-m16-ex3-ef30 (query)                        1_741.41     1_007.87     2_749.29       0.9515          1.0010            1.0002        13.90
HnswRaBitQ-m16-ex3-ef60 (query)                        1_741.41     1_475.53     3_216.95       0.9575          1.0005            1.0001        13.90
HnswRaBitQ-m16-ex3-ef120 (query)                       1_741.41     2_054.38     3_795.79       0.9583          1.0005            1.0001        13.90
HnswRaBitQ-m16-ex3 (self)                              1_741.41     8_045.89     9_787.31       0.9466          1.0010            1.0005        13.90
HnswRaBitQ-m16-ex5-ef15 (query)                        1_829.00       656.81     2_485.82       0.9479          1.0024            1.0001        16.95
HnswRaBitQ-m16-ex5-ef30 (query)                        1_829.00       991.07     2_820.08       0.9779          1.0006            1.0000        16.95
HnswRaBitQ-m16-ex5-ef60 (query)                        1_829.00     1_479.83     3_308.83       0.9865          1.0001            1.0000        16.95
HnswRaBitQ-m16-ex5-ef120 (query)                       1_829.00     2_063.16     3_892.16       0.9876          1.0001            1.0000        16.95
HnswRaBitQ-m16-ex5 (self)                              1_829.00     8_396.34    10_225.34       0.9835          1.0002            1.0000        16.95
HnswRaBitQ-m16-ex8-ef15 (query)                        2_594.57       662.58     3_257.15       0.9461          3.3271            1.0000        21.53
HnswRaBitQ-m16-ex8-ef30 (query)                        2_594.57     1_008.32     3_602.89       0.9846          1.7199            1.0000        21.53
HnswRaBitQ-m16-ex8-ef60 (query)                        2_594.57     1_506.52     4_101.09       0.9968          1.0055            1.0000        21.53
HnswRaBitQ-m16-ex8-ef120 (query)                       2_594.57     2_117.35     4_711.92       0.9982          1.0000            1.0000        21.53
HnswRaBitQ-m16-ex8 (self)                              2_594.57     8_193.00    10_787.56       0.9965          1.0018            1.0000        21.53
HnswRaBitQ-m32-ex1-ef15 (query)                        1_685.71       817.49     2_503.20       0.8618          1.0061            1.0053        16.95
HnswRaBitQ-m32-ex1-ef30 (query)                        1_685.71     1_199.39     2_885.10       0.8680          1.0054            1.0048        16.95
HnswRaBitQ-m32-ex1-ef60 (query)                        1_685.71     1_691.79     3_377.50       0.8690          1.0052            1.0047        16.95
HnswRaBitQ-m32-ex1-ef120 (query)                       1_685.71     2_249.46     3_935.17       0.8691          1.0052            1.0047        16.95
HnswRaBitQ-m32-ex1 (self)                              1_685.71     9_397.97    11_083.68       0.8314          1.0108            1.0098        16.95
HnswRaBitQ-m32-ex3-ef15 (query)                        1_767.86       829.96     2_597.82       0.9433          1.0016            1.0004        20.00
HnswRaBitQ-m32-ex3-ef30 (query)                        1_767.86     1_211.27     2_979.13       0.9557          1.0006            1.0002        20.00
HnswRaBitQ-m32-ex3-ef60 (query)                        1_767.86     1_713.19     3_481.05       0.9581          1.0004            1.0001        20.00
HnswRaBitQ-m32-ex3-ef120 (query)                       1_767.86     2_290.02     4_057.88       0.9584          1.0004            1.0001        20.00
HnswRaBitQ-m32-ex3 (self)                              1_767.86     9_899.73    11_667.59       0.9471          1.0009            1.0005        20.00
HnswRaBitQ-m32-ex5-ef15 (query)                        1_919.14       837.13     2_756.26       0.9673          1.0011            1.0000        23.06
HnswRaBitQ-m32-ex5-ef30 (query)                        1_919.14     1_225.92     3_145.06       0.9838          1.0002            1.0000        23.06
HnswRaBitQ-m32-ex5-ef60 (query)                        1_919.14     1_754.47     3_673.61       0.9873          1.0001            1.0000        23.06
HnswRaBitQ-m32-ex5-ef120 (query)                       1_919.14     2_357.44     4_276.58       0.9878          1.0000            1.0000        23.06
HnswRaBitQ-m32-ex5 (self)                              1_919.14     9_536.88    11_456.02       0.9842          1.0001            1.0000        23.06
HnswRaBitQ-m32-ex8-ef15 (query)                        2_610.80       849.73     3_460.54       0.9751          1.0011            1.0000        27.63
HnswRaBitQ-m32-ex8-ef30 (query)                        2_610.80     1_244.01     3_854.82       0.9939          1.0002            1.0000        27.63
HnswRaBitQ-m32-ex8-ef60 (query)                        2_610.80     1_772.73     4_383.53       0.9978          1.0000            1.0000        27.63
HnswRaBitQ-m32-ex8-ef120 (query)                       2_610.80     2_340.30     4_951.11       0.9984          1.0000            1.0000        27.63
HnswRaBitQ-m32-ex8 (self)                              2_610.80     9_758.46    12_369.26       0.9975          1.0000            1.0000        27.63
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
Exhaustive (query)                                        68.80     1_299.93     1_368.73       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.80     4_289.17     4_357.96       1.0000          1.0000            1.0000        97.66
QG-d32-l64-ef15 (query)                                5_605.68       140.49     5_746.17       0.9584          1.0041            1.0000       214.01
QG-d32-l64-ef30 (query)                                5_605.68       233.62     5_839.30       0.9916          1.0014            1.0000       214.01
QG-d32-l64-ef60 (query)                                5_605.68       400.32     6_006.00       0.9978          1.0007            1.0000       214.01
QG-d32-l64-ef120 (query)                               5_605.68       713.26     6_318.95       0.9992          1.0003            1.0000       214.01
QG-d32-l64 (self)                                      5_605.68     1_217.84     6_823.52       0.9980          1.0007            1.0000       214.01
QG-d32-l128-ef15 (query)                               6_458.35       143.23     6_601.59       0.9611          1.0030            1.0000       214.01
QG-d32-l128-ef30 (query)                               6_458.35       232.46     6_690.81       0.9931          1.0012            1.0000       214.01
QG-d32-l128-ef60 (query)                               6_458.35       399.43     6_857.78       0.9987          1.0005            1.0000       214.01
QG-d32-l128-ef120 (query)                              6_458.35       717.49     7_175.85       0.9995          1.0003            1.0000       214.01
QG-d32-l128 (self)                                     6_458.35     1_223.87     7_682.23       0.9987          1.0006            1.0000       214.01
QG-d64-l64-ef15 (query)                               16_392.44       217.68    16_610.12       0.9922          1.0002            1.0000       329.97
QG-d64-l64-ef30 (query)                               16_392.44       385.41    16_777.85       0.9997          1.0000            1.0000       329.97
QG-d64-l64-ef60 (query)                               16_392.44       676.76    17_069.19       1.0000          1.0000            1.0000       329.97
QG-d64-l64-ef120 (query)                              16_392.44     1_214.36    17_606.80       1.0000          1.0000            1.0000       329.97
QG-d64-l64 (self)                                     16_392.44     2_178.91    18_571.35       1.0000          1.0000            1.0000       329.97
QG-d64-l128-ef15 (query)                              17_141.73       217.18    17_358.91       0.9922          1.0002            1.0000       329.97
QG-d64-l128-ef30 (query)                              17_141.73       382.39    17_524.12       0.9997          1.0000            1.0000       329.97
QG-d64-l128-ef60 (query)                              17_141.73       681.82    17_823.54       1.0000          1.0000            1.0000       329.97
QG-d64-l128-ef120 (query)                             17_141.73     1_218.87    18_360.60       1.0000          1.0000            1.0000       329.97
QG-d64-l128 (self)                                    17_141.73     2_195.50    19_337.22       1.0000          1.0000            1.0000       329.97
HnswRaBitQ-m16-ex1-ef15 (query)                        3_042.61     1_423.74     4_466.35       0.8452          1.0073            1.0044        14.15
HnswRaBitQ-m16-ex1-ef30 (query)                        3_042.61     2_134.12     5_176.74       0.8672          1.0044            1.0032        14.15
HnswRaBitQ-m16-ex1-ef60 (query)                        3_042.61     3_121.22     6_163.83       0.8742          1.0035            1.0029        14.15
HnswRaBitQ-m16-ex1-ef120 (query)                       3_042.61     4_283.16     7_325.77       0.8753          1.0033            1.0028        14.15
HnswRaBitQ-m16-ex1 (self)                              3_042.61    16_734.63    19_777.24       0.8358          1.0069            1.0060        14.15
HnswRaBitQ-m16-ex3-ef15 (query)                        3_223.46     1_422.29     4_645.75       0.9101          1.0035            1.0010        20.25
HnswRaBitQ-m16-ex3-ef30 (query)                        3_223.46     2_122.32     5_345.78       0.9450          1.0012            1.0003        20.25
HnswRaBitQ-m16-ex3-ef60 (query)                        3_223.46     3_208.41     6_431.87       0.9568          1.0006            1.0001        20.25
HnswRaBitQ-m16-ex3-ef120 (query)                       3_223.46     4_325.61     7_549.07       0.9591          1.0004            1.0001        20.25
HnswRaBitQ-m16-ex3 (self)                              3_223.46    16_766.76    19_990.22       0.9465          1.0009            1.0003        20.25
HnswRaBitQ-m16-ex5-ef15 (query)                        3_393.38     1_415.32     4_808.70       0.9241          1.0035            1.0006        26.36
HnswRaBitQ-m16-ex5-ef30 (query)                        3_393.38     2_160.85     5_554.23       0.9671          1.0011            1.0000        26.36
HnswRaBitQ-m16-ex5-ef60 (query)                        3_393.38     3_134.42     6_527.80       0.9836          1.0004            1.0000        26.36
HnswRaBitQ-m16-ex5-ef120 (query)                       3_393.38     4_314.40     7_707.79       0.9870          1.0002            1.0000        26.36
HnswRaBitQ-m16-ex5 (self)                              3_393.38    17_901.36    21_294.74       0.9811          1.0004            1.0000        26.36
HnswRaBitQ-m16-ex8-ef15 (query)                        4_859.12     1_435.35     6_294.47       0.9293          1.0035            1.0006        35.51
HnswRaBitQ-m16-ex8-ef30 (query)                        4_859.12     2_150.06     7_009.18       0.9752          1.0010            1.0000        35.51
HnswRaBitQ-m16-ex8-ef60 (query)                        4_859.12     3_169.81     8_028.93       0.9934          1.0003            1.0000        35.51
HnswRaBitQ-m16-ex8-ef120 (query)                       4_859.12     4_406.84     9_265.96       0.9975          1.0002            1.0000        35.51
HnswRaBitQ-m16-ex8 (self)                              4_859.12    16_932.07    21_791.19       0.9937          1.0003            1.0000        35.51
HnswRaBitQ-m32-ex1-ef15 (query)                        3_075.92     1_924.03     4_999.95       0.8634          1.0044            1.0034        20.25
HnswRaBitQ-m32-ex1-ef30 (query)                        3_075.92     2_774.68     5_850.60       0.8730          1.0035            1.0029        20.25
HnswRaBitQ-m32-ex1-ef60 (query)                        3_075.92     3_798.99     6_874.91       0.8752          1.0032            1.0028        20.25
HnswRaBitQ-m32-ex1-ef120 (query)                       3_075.92     4_836.09     7_912.00       0.8755          1.0032            1.0028        20.25
HnswRaBitQ-m32-ex1 (self)                              3_075.92    21_755.28    24_831.20       0.8367          1.0066            1.0059        20.25
HnswRaBitQ-m32-ex3-ef15 (query)                        3_308.76     1_948.59     5_257.34       0.9373          1.0016            1.0003        26.36
HnswRaBitQ-m32-ex3-ef30 (query)                        3_308.76     2_806.49     6_115.24       0.9548          1.0006            1.0001        26.36
HnswRaBitQ-m32-ex3-ef60 (query)                        3_308.76     3_844.52     7_153.28       0.9589          1.0003            1.0001        26.36
HnswRaBitQ-m32-ex3-ef120 (query)                       3_308.76     4_885.71     8_194.46       0.9595          1.0003            1.0001        26.36
HnswRaBitQ-m32-ex3 (self)                              3_308.76    21_109.96    24_418.71       0.9483          1.0006            1.0003        26.36
HnswRaBitQ-m32-ex5-ef15 (query)                        3_444.81     1_952.94     5_397.76       0.9580          1.0014            1.0000        32.46
HnswRaBitQ-m32-ex5-ef30 (query)                        3_444.81     2_816.37     6_261.18       0.9810          1.0003            1.0000        32.46
HnswRaBitQ-m32-ex5-ef60 (query)                        3_444.81     3_872.38     7_317.20       0.9867          1.0001            1.0000        32.46
HnswRaBitQ-m32-ex5-ef120 (query)                       3_444.81     4_913.31     8_358.13       0.9875          1.0001            1.0000        32.46
HnswRaBitQ-m32-ex5 (self)                              3_444.81    22_160.87    25_605.68       0.9837          1.0001            1.0000        32.46
HnswRaBitQ-m32-ex8-ef15 (query)                        4_925.94     1_964.11     6_890.04       0.9651          1.0013            1.0000        41.62
HnswRaBitQ-m32-ex8-ef30 (query)                        4_925.94     2_836.51     7_762.45       0.9905          1.0003            1.0000        41.62
HnswRaBitQ-m32-ex8-ef60 (query)                        4_925.94     3_904.72     8_830.65       0.9973          1.0001            1.0000        41.62
HnswRaBitQ-m32-ex8-ef120 (query)                       4_925.94     4_944.47     9_870.41       0.9982          1.0000            1.0000        41.62
HnswRaBitQ-m32-ex8 (self)                              4_925.94    20_967.30    25_893.24       0.9968          1.0001            1.0000        41.62
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
Exhaustive (query)                                       100.88     1_863.33     1_964.21       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.88     6_249.95     6_350.82       1.0000          1.0000            1.0000       146.48
QG-d32-l64-ef15 (query)                                7_867.07       251.06     8_118.13       0.9516          1.0067            1.0000       311.66
QG-d32-l64-ef30 (query)                                7_867.07       401.41     8_268.48       0.9812          1.0050            1.0000       311.66
QG-d32-l64-ef60 (query)                                7_867.07       648.37     8_515.44       0.9907          1.0042            1.0000       311.66
QG-d32-l64-ef120 (query)                               7_867.07     1_060.22     8_927.29       0.9938          1.0037            1.0000       311.66
QG-d32-l64 (self)                                      7_867.07     2_061.41     9_928.48       0.9909          1.0044            1.0000       311.66
QG-d32-l128-ef15 (query)                               8_787.66       247.29     9_034.95       0.9546          1.0064            1.0000       311.66
QG-d32-l128-ef30 (query)                               8_787.66       402.57     9_190.23       0.9838          1.0046            1.0000       311.66
QG-d32-l128-ef60 (query)                               8_787.66       649.70     9_437.37       0.9923          1.0038            1.0000       311.66
QG-d32-l128-ef120 (query)                              8_787.66     1_069.78     9_857.45       0.9950          1.0033            1.0000       311.66
QG-d32-l128 (self)                                     8_787.66     2_105.62    10_893.29       0.9926          1.0040            1.0000       311.66
QG-d64-l64-ef15 (query)                               29_949.02       483.05    30_432.07       0.9921          1.0010            1.0000       476.46
QG-d64-l64-ef30 (query)                               29_949.02       789.68    30_738.70       0.9984          1.0004            1.0000       476.46
QG-d64-l64-ef60 (query)                               29_949.02     1_266.25    31_215.26       0.9995          1.0002            1.0000       476.46
QG-d64-l64-ef120 (query)                              29_949.02     2_001.38    31_950.39       0.9997          1.0001            1.0000       476.46
QG-d64-l64 (self)                                     29_949.02     4_066.85    34_015.87       0.9995          1.0002            1.0000       476.46
QG-d64-l128-ef15 (query)                              29_875.88       488.95    30_364.83       0.9931          1.0008            1.0000       476.46
QG-d64-l128-ef30 (query)                              29_875.88       790.25    30_666.13       0.9986          1.0004            1.0000       476.46
QG-d64-l128-ef60 (query)                              29_875.88     1_266.54    31_142.43       0.9996          1.0002            1.0000       476.46
QG-d64-l128-ef120 (query)                             29_875.88     2_003.84    31_879.72       0.9999          1.0001            1.0000       476.46
QG-d64-l128 (self)                                    29_875.88     4_084.70    33_960.58       0.9997          1.0002            1.0000       476.46
HnswRaBitQ-m16-ex1-ef15 (query)                        4_178.88     2_199.75     6_378.63       0.8248          1.0080            1.0039        17.45
HnswRaBitQ-m16-ex1-ef30 (query)                        4_178.88     3_273.34     7_452.22       0.8517          1.0047            1.0029        17.45
HnswRaBitQ-m16-ex1-ef60 (query)                        4_178.88     4_802.73     8_981.61       0.8626          1.0033            1.0025        17.45
HnswRaBitQ-m16-ex1-ef120 (query)                       4_178.88     6_441.78    10_620.66       0.8652          1.0029            1.0024        17.45
HnswRaBitQ-m16-ex1 (self)                              4_178.88    27_375.53    31_554.41       0.8180          1.0059            1.0051        17.45
HnswRaBitQ-m16-ex3-ef15 (query)                        4_417.87     2_212.91     6_630.79       0.8876          1.0065            1.0013        26.61
HnswRaBitQ-m16-ex3-ef30 (query)                        4_417.87     3_298.55     7_716.42       0.9318          1.0023            1.0003        26.61
HnswRaBitQ-m16-ex3-ef60 (query)                        4_417.87     4_815.89     9_233.76       0.9505          1.0009            1.0001        26.61
HnswRaBitQ-m16-ex3-ef120 (query)                       4_417.87     6_611.48    11_029.35       0.9551          1.0005            1.0001        26.61
HnswRaBitQ-m16-ex3 (self)                              4_417.87    25_791.73    30_209.60       0.9381          1.0011            1.0003        26.61
HnswRaBitQ-m16-ex5-ef15 (query)                        4_772.20     2_243.86     7_016.05       0.9017          1.0278            1.0010        35.76
HnswRaBitQ-m16-ex5-ef30 (query)                        4_772.20     3_344.55     8_116.75       0.9538          1.0230            1.0000        35.76
HnswRaBitQ-m16-ex5-ef60 (query)                        4_772.20     4_881.19     9_653.39       0.9783          1.0007            1.0000        35.76
HnswRaBitQ-m16-ex5-ef120 (query)                       4_772.20     6_594.50    11_366.70       0.9846          1.0003            1.0000        35.76
HnswRaBitQ-m16-ex5 (self)                              4_772.20    25_952.77    30_724.97       0.9756          1.0007            1.0000        35.76
HnswRaBitQ-m16-ex8-ef15 (query)                        6_995.73     2_228.79     9_224.52       0.9044          1.0078            1.0010        49.50
HnswRaBitQ-m16-ex8-ef30 (query)                        6_995.73     3_336.14    10_331.87       0.9599          1.0032            1.0000        49.50
HnswRaBitQ-m16-ex8-ef60 (query)                        6_995.73     4_879.79    11_875.52       0.9877          1.0014            1.0000        49.50
HnswRaBitQ-m16-ex8-ef120 (query)                       6_995.73     6_556.45    13_552.19       0.9954          1.0007            1.0000        49.50
HnswRaBitQ-m16-ex8 (self)                              6_995.73    26_159.75    33_155.49       0.9884          1.0011            1.0000        49.50
HnswRaBitQ-m32-ex1-ef15 (query)                        4_221.56     3_221.04     7_442.59       0.8048        171.6451            1.0031        23.56
HnswRaBitQ-m32-ex1-ef30 (query)                        4_221.56     4_689.09     8_910.65       0.8509          7.3915            1.0026        23.56
HnswRaBitQ-m32-ex1-ef60 (query)                        4_221.56     6_311.16    10_532.72       0.8651          1.0028            1.0024        23.56
HnswRaBitQ-m32-ex1-ef120 (query)                       4_221.56     7_697.28    11_918.84       0.8659          1.0027            1.0024        23.56
HnswRaBitQ-m32-ex1 (self)                              4_221.56    33_737.57    37_959.13       0.8194          1.0084            1.0050        23.56
HnswRaBitQ-m32-ex3-ef15 (query)                        4_516.44     3_153.58     7_670.01       0.9253          2.5784            1.0004        32.71
HnswRaBitQ-m32-ex3-ef30 (query)                        4_516.44     4_518.46     9_034.90       0.9478          1.0536            1.0001        32.71
HnswRaBitQ-m32-ex3-ef60 (query)                        4_516.44     6_075.20    10_591.64       0.9550          1.0004            1.0001        32.71
HnswRaBitQ-m32-ex3-ef120 (query)                       4_516.44     7_524.82    12_041.26       0.9562          1.0003            1.0001        32.71
HnswRaBitQ-m32-ex3 (self)                              4_516.44    32_597.82    37_114.26       0.9420          1.0006            1.0003        32.71
HnswRaBitQ-m32-ex5-ef15 (query)                        4_804.73     3_175.57     7_980.29       0.9464          1.0028            1.0000        41.87
HnswRaBitQ-m32-ex5-ef30 (query)                        4_804.73     4_503.99     9_308.72       0.9747          1.0009            1.0000        41.87
HnswRaBitQ-m32-ex5-ef60 (query)                        4_804.73     6_518.64    11_323.37       0.9840          1.0002            1.0000        41.87
HnswRaBitQ-m32-ex5-ef120 (query)                       4_804.73     7_695.61    12_500.33       0.9856          1.0001            1.0000        41.87
HnswRaBitQ-m32-ex5 (self)                              4_804.73    34_894.22    39_698.95       0.9804          1.0002            1.0000        41.87
HnswRaBitQ-m32-ex8-ef15 (query)                        7_090.74     3_155.13    10_245.87       0.9549          1.0021            1.0000        55.60
HnswRaBitQ-m32-ex8-ef30 (query)                        7_090.74     4_527.19    11_617.93       0.9855          1.0005            1.0000        55.60
HnswRaBitQ-m32-ex8-ef60 (query)                        7_090.74     6_117.74    13_208.48       0.9958          1.0002            1.0000        55.60
HnswRaBitQ-m32-ex8-ef120 (query)                       7_090.74     7_570.01    14_660.75       0.9978          1.0001            1.0000        55.60
HnswRaBitQ-m32-ex8 (self)                              7_090.74    32_670.39    39_761.13       0.9953          1.0002            1.0000        55.60
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
Exhaustive (query)                                        32.77       705.95       738.72       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.77     2_346.98     2_379.75       1.0000          1.0000            1.0000        48.83
QG-d32-l64-ef15 (query)                                2_288.77        55.63     2_344.40       0.9891          1.0005            1.0000       116.35
QG-d32-l64-ef30 (query)                                2_288.77        83.55     2_372.31       0.9999          1.0000            1.0000       116.35
QG-d32-l64-ef60 (query)                                2_288.77       146.35     2_435.11       1.0000          1.0000            1.0000       116.35
QG-d32-l64-ef120 (query)                               2_288.77       275.09     2_563.85       1.0000          1.0000            1.0000       116.35
QG-d32-l64 (self)                                      2_288.77       448.20     2_736.96       1.0000          1.0000            1.0000       116.35
QG-d32-l128-ef15 (query)                               3_515.39        53.97     3_569.36       0.9885          1.0006            1.0000       116.35
QG-d32-l128-ef30 (query)                               3_515.39        83.69     3_599.08       0.9999          1.0000            1.0000       116.35
QG-d32-l128-ef60 (query)                               3_515.39       145.26     3_660.65       1.0000          1.0000            1.0000       116.35
QG-d32-l128-ef120 (query)                              3_515.39       279.80     3_795.19       1.0000          1.0000            1.0000       116.35
QG-d32-l128 (self)                                     3_515.39       454.43     3_969.82       1.0000          1.0000            1.0000       116.35
QG-d64-l64-ef15 (query)                                2_844.67        60.49     2_905.16       0.9905          1.0004            1.0000       183.49
QG-d64-l64-ef30 (query)                                2_844.67        96.30     2_940.97       1.0000          1.0000            1.0000       183.49
QG-d64-l64-ef60 (query)                                2_844.67       165.52     3_010.19       1.0000          1.0000            1.0000       183.49
QG-d64-l64-ef120 (query)                               2_844.67       321.97     3_166.64       1.0000          1.0000            1.0000       183.49
QG-d64-l64 (self)                                      2_844.67       528.21     3_372.88       1.0000          1.0000            1.0000       183.49
QG-d64-l128-ef15 (query)                               4_424.21        59.48     4_483.69       0.9907          1.0004            1.0000       183.49
QG-d64-l128-ef30 (query)                               4_424.21        97.80     4_522.01       1.0000          1.0000            1.0000       183.49
QG-d64-l128-ef60 (query)                               4_424.21       168.26     4_592.47       1.0000          1.0000            1.0000       183.49
QG-d64-l128-ef120 (query)                              4_424.21       327.60     4_751.81       1.0000          1.0000            1.0000       183.49
QG-d64-l128 (self)                                     4_424.21       526.19     4_950.40       1.0000          1.0000            1.0000       183.49
HnswRaBitQ-m16-ex1-ef15 (query)                        1_575.93       396.30     1_972.23       0.9381          1.0169            1.0033        10.85
HnswRaBitQ-m16-ex1-ef30 (query)                        1_575.93       569.48     2_145.41       0.9392          1.0155            1.0031        10.85
HnswRaBitQ-m16-ex1-ef60 (query)                        1_575.93       866.73     2_442.67       0.9400          1.0101            1.0031        10.85
HnswRaBitQ-m16-ex1-ef120 (query)                       1_575.93     1_371.35     2_947.29       0.9405          1.0072            1.0031        10.85
HnswRaBitQ-m16-ex1 (self)                              1_575.93     4_810.85     6_386.79       0.9122          1.0175            1.0088        10.85
HnswRaBitQ-m16-ex3-ef15 (query)                        1_719.71       393.99     2_113.70       0.9772          1.0110            1.0000        13.90
HnswRaBitQ-m16-ex3-ef30 (query)                        1_719.71       566.02     2_285.74       0.9795          1.0086            1.0000        13.90
HnswRaBitQ-m16-ex3-ef60 (query)                        1_719.71       880.46     2_600.17       0.9803          1.0053            1.0000        13.90
HnswRaBitQ-m16-ex3-ef120 (query)                       1_719.71     1_396.92     3_116.63       0.9807          1.0029            1.0000        13.90
HnswRaBitQ-m16-ex3 (self)                              1_719.71     4_829.83     6_549.54       0.9733          1.0057            1.0000        13.90
HnswRaBitQ-m16-ex5-ef15 (query)                        1_796.13       404.26     2_200.39       0.9901          1.0145            1.0000        16.95
HnswRaBitQ-m16-ex5-ef30 (query)                        1_796.13       589.34     2_385.47       0.9928          1.0112            1.0000        16.95
HnswRaBitQ-m16-ex5-ef60 (query)                        1_796.13       902.36     2_698.49       0.9933          1.0078            1.0000        16.95
HnswRaBitQ-m16-ex5-ef120 (query)                       1_796.13     1_413.88     3_210.01       0.9942          1.0032            1.0000        16.95
HnswRaBitQ-m16-ex5 (self)                              1_796.13     4_840.43     6_636.56       0.9912          1.0076            1.0000        16.95
HnswRaBitQ-m16-ex8-ef15 (query)                        2_510.44       418.35     2_928.78       0.9950          1.0107            1.0000        21.53
HnswRaBitQ-m16-ex8-ef30 (query)                        2_510.44       584.62     3_095.06       0.9979          1.0079            1.0000        21.53
HnswRaBitQ-m16-ex8-ef60 (query)                        2_510.44       896.17     3_406.60       0.9989          1.0030            1.0000        21.53
HnswRaBitQ-m16-ex8-ef120 (query)                       2_510.44     1_444.06     3_954.49       0.9993          1.0000            1.0000        21.53
HnswRaBitQ-m16-ex8 (self)                              2_510.44     4_900.67     7_411.10       0.9983          1.0047            1.0000        21.53
HnswRaBitQ-m32-ex1-ef15 (query)                        1_669.69       490.90     2_160.59       0.9397          1.0102            1.0031        16.95
HnswRaBitQ-m32-ex1-ef30 (query)                        1_669.69       665.94     2_335.62       0.9404          1.0089            1.0031        16.95
HnswRaBitQ-m32-ex1-ef60 (query)                        1_669.69       999.09     2_668.78       0.9409          1.0062            1.0030        16.95
HnswRaBitQ-m32-ex1-ef120 (query)                       1_669.69     1_535.00     3_204.69       0.9409          1.0056            1.0030        16.95
HnswRaBitQ-m32-ex1 (self)                              1_669.69     5_480.96     7_150.65       0.9131          1.0128            1.0087        16.95
HnswRaBitQ-m32-ex3-ef15 (query)                        1_773.98       482.09     2_256.07       0.9785          1.0086            1.0000        20.00
HnswRaBitQ-m32-ex3-ef30 (query)                        1_773.98       695.52     2_469.50       0.9800          1.0064            1.0000        20.00
HnswRaBitQ-m32-ex3-ef60 (query)                        1_773.98       992.40     2_766.38       0.9806          1.0037            1.0000        20.00
HnswRaBitQ-m32-ex3-ef120 (query)                       1_773.98     1_568.09     3_342.07       0.9811          1.0015            1.0000        20.00
HnswRaBitQ-m32-ex3 (self)                              1_773.98     5_648.13     7_422.11       0.9739          1.0043            1.0000        20.00
HnswRaBitQ-m32-ex5-ef15 (query)                        1_847.23       494.54     2_341.77       0.9921          1.0054            1.0000        23.06
HnswRaBitQ-m32-ex5-ef30 (query)                        1_847.23       674.31     2_521.54       0.9939          1.0047            1.0000        23.06
HnswRaBitQ-m32-ex5-ef60 (query)                        1_847.23     1_041.69     2_888.93       0.9944          1.0029            1.0000        23.06
HnswRaBitQ-m32-ex5-ef120 (query)                       1_847.23     1_602.89     3_450.12       0.9946          1.0012            1.0000        23.06
HnswRaBitQ-m32-ex5 (self)                              1_847.23     5_535.13     7_382.36       0.9922          1.0031            1.0000        23.06
HnswRaBitQ-m32-ex8-ef15 (query)                        2_583.11       494.94     3_078.05       0.9969          1.0032            1.0000        27.63
HnswRaBitQ-m32-ex8-ef30 (query)                        2_583.11       699.99     3_283.10       0.9986          1.0022            1.0000        27.63
HnswRaBitQ-m32-ex8-ef60 (query)                        2_583.11     1_027.16     3_610.27       0.9989          1.0016            1.0000        27.63
HnswRaBitQ-m32-ex8-ef120 (query)                       2_583.11     1_606.92     4_190.03       0.9992          1.0005            1.0000        27.63
HnswRaBitQ-m32-ex8 (self)                              2_583.11     5_580.45     8_163.56       0.9988          1.0014            1.0000        27.63
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
Exhaustive (query)                                        67.99     1_280.60     1_348.59       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         67.99     4_311.43     4_379.42       1.0000          1.0000            1.0000        97.66
QG-d32-l64-ef15 (query)                                4_247.81        84.37     4_332.18       0.9904          1.0005            1.0000       214.01
QG-d32-l64-ef30 (query)                                4_247.81       117.71     4_365.52       0.9999          1.0000            1.0000       214.01
QG-d32-l64-ef60 (query)                                4_247.81       189.36     4_437.17       1.0000          1.0000            1.0000       214.01
QG-d32-l64-ef120 (query)                               4_247.81       335.00     4_582.82       1.0000          1.0000            1.0000       214.01
QG-d32-l64 (self)                                      4_247.81       561.13     4_808.94       1.0000          1.0000            1.0000       214.01
QG-d32-l128-ef15 (query)                               6_767.94        84.28     6_852.22       0.9902          1.0005            1.0000       214.01
QG-d32-l128-ef30 (query)                               6_767.94       118.91     6_886.85       0.9999          1.0000            1.0000       214.01
QG-d32-l128-ef60 (query)                               6_767.94       186.61     6_954.55       1.0000          1.0000            1.0000       214.01
QG-d32-l128-ef120 (query)                              6_767.94       337.60     7_105.54       1.0000          1.0000            1.0000       214.01
QG-d32-l128 (self)                                     6_767.94       561.25     7_329.19       1.0000          1.0000            1.0000       214.01
QG-d64-l64-ef15 (query)                                5_395.24        92.63     5_487.87       0.9916          1.0004            1.0000       329.97
QG-d64-l64-ef30 (query)                                5_395.24       130.08     5_525.32       0.9999          1.0000            1.0000       329.97
QG-d64-l64-ef60 (query)                                5_395.24       211.60     5_606.84       1.0000          1.0000            1.0000       329.97
QG-d64-l64-ef120 (query)                               5_395.24       383.42     5_778.66       1.0000          1.0000            1.0000       329.97
QG-d64-l64 (self)                                      5_395.24       667.54     6_062.78       1.0000          1.0000            1.0000       329.97
QG-d64-l128-ef15 (query)                               8_637.96        90.28     8_728.24       0.9919          1.0003            1.0000       329.97
QG-d64-l128-ef30 (query)                               8_637.96       132.84     8_770.80       0.9999          1.0000            1.0000       329.97
QG-d64-l128-ef60 (query)                               8_637.96       216.43     8_854.39       1.0000          1.0000            1.0000       329.97
QG-d64-l128-ef120 (query)                              8_637.96       391.34     9_029.30       1.0000          1.0000            1.0000       329.97
QG-d64-l128 (self)                                     8_637.96       642.71     9_280.67       1.0000          1.0000            1.0000       329.97
HnswRaBitQ-m16-ex1-ef15 (query)                        2_796.87       738.08     3_534.95       0.9537          1.0389            1.0007        14.15
HnswRaBitQ-m16-ex1-ef30 (query)                        2_796.87     1_031.31     3_828.18       0.9555          1.0284            1.0006        14.15
HnswRaBitQ-m16-ex1-ef60 (query)                        2_796.87     1_541.00     4_337.87       0.9565          1.0182            1.0006        14.15
HnswRaBitQ-m16-ex1-ef120 (query)                       2_796.87     2_406.09     5_202.96       0.9571          1.0114            1.0006        14.15
HnswRaBitQ-m16-ex1 (self)                              2_796.87     8_342.07    11_138.94       0.9335          1.0190            1.0040        14.15
HnswRaBitQ-m16-ex3-ef15 (query)                        2_966.64       757.58     3_724.22       0.9810          1.0372            1.0000        20.25
HnswRaBitQ-m16-ex3-ef30 (query)                        2_966.64     1_092.38     4_059.02       0.9835          1.0266            1.0000        20.25
HnswRaBitQ-m16-ex3-ef60 (query)                        2_966.64     1_710.50     4_677.14       0.9843          1.0194            1.0000        20.25
HnswRaBitQ-m16-ex3-ef120 (query)                       2_966.64     2_633.99     5_600.63       0.9849          1.0119            1.0000        20.25
HnswRaBitQ-m16-ex3 (self)                              2_966.64     8_433.35    11_399.99       0.9796          1.0196            1.0000        20.25
HnswRaBitQ-m16-ex5-ef15 (query)                        3_213.55       764.80     3_978.35       0.9792          1.2099            1.0000        26.36
HnswRaBitQ-m16-ex5-ef30 (query)                        3_213.55     1_073.95     4_287.49       0.9861          1.1230            1.0000        26.36
HnswRaBitQ-m16-ex5-ef60 (query)                        3_213.55     1_602.83     4_816.38       0.9899          1.0682            1.0000        26.36
HnswRaBitQ-m16-ex5-ef120 (query)                       3_213.55     2_498.60     5_712.15       0.9927          1.0346            1.0000        26.36
HnswRaBitQ-m16-ex5 (self)                              3_213.55     8_526.08    11_739.63       0.9885          1.0681            1.0000        26.36
HnswRaBitQ-m16-ex8-ef15 (query)                        4_652.27       757.85     5_410.12       0.9865          1.1649            1.0000        35.51
HnswRaBitQ-m16-ex8-ef30 (query)                        4_652.27     1_088.18     5_740.45       0.9919          1.1039            1.0000        35.51
HnswRaBitQ-m16-ex8-ef60 (query)                        4_652.27     1_604.03     6_256.30       0.9949          1.0492            1.0000        35.51
HnswRaBitQ-m16-ex8-ef120 (query)                       4_652.27     2_481.46     7_133.73       0.9968          1.0195            1.0000        35.51
HnswRaBitQ-m16-ex8 (self)                              4_652.27     8_507.53    13_159.80       0.9950          1.0479            1.0000        35.51
HnswRaBitQ-m32-ex1-ef15 (query)                        2_915.77       925.31     3_841.08       0.9573          1.0024            1.0006        20.25
HnswRaBitQ-m32-ex1-ef30 (query)                        2_915.77     1_237.50     4_153.27       0.9580          1.0023            1.0006        20.25
HnswRaBitQ-m32-ex1-ef60 (query)                        2_915.77     1_773.86     4_689.63       0.9581          1.0023            1.0006        20.25
HnswRaBitQ-m32-ex1-ef120 (query)                       2_915.77     2_721.35     5_637.12       0.9581          1.0023            1.0006        20.25
HnswRaBitQ-m32-ex1 (self)                              2_915.77     9_580.50    12_496.27       0.9347          1.0067            1.0039        20.25
HnswRaBitQ-m32-ex3-ef15 (query)                        3_113.17       932.78     4_045.95       0.9847          1.0020            1.0000        26.36
HnswRaBitQ-m32-ex3-ef30 (query)                        3_113.17     1_254.22     4_367.39       0.9860          1.0018            1.0000        26.36
HnswRaBitQ-m32-ex3-ef60 (query)                        3_113.17     1_785.93     4_899.10       0.9862          1.0002            1.0000        26.36
HnswRaBitQ-m32-ex3-ef120 (query)                       3_113.17     2_714.89     5_828.06       0.9863          1.0002            1.0000        26.36
HnswRaBitQ-m32-ex3 (self)                              3_113.17     9_610.22    12_723.39       0.9815          1.0007            1.0000        26.36
HnswRaBitQ-m32-ex5-ef15 (query)                        3_422.21       943.97     4_366.18       0.9944          1.0004            1.0000        32.46
HnswRaBitQ-m32-ex5-ef30 (query)                        3_422.21     1_255.42     4_677.63       0.9960          1.0002            1.0000        32.46
HnswRaBitQ-m32-ex5-ef60 (query)                        3_422.21     1_811.63     5_233.84       0.9962          1.0000            1.0000        32.46
HnswRaBitQ-m32-ex5-ef120 (query)                       3_422.21     2_739.86     6_162.07       0.9962          1.0000            1.0000        32.46
HnswRaBitQ-m32-ex5 (self)                              3_422.21     9_693.04    13_115.25       0.9946          1.0003            1.0000        32.46
HnswRaBitQ-m32-ex8-ef15 (query)                        4_793.75       936.30     5_730.05       0.9976          1.0003            1.0000        41.62
HnswRaBitQ-m32-ex8-ef30 (query)                        4_793.75     1_251.87     6_045.62       0.9993          1.0001            1.0000        41.62
HnswRaBitQ-m32-ex8-ef60 (query)                        4_793.75     1_842.03     6_635.78       0.9995          1.0000            1.0000        41.62
HnswRaBitQ-m32-ex8-ef120 (query)                       4_793.75     2_738.01     7_531.76       0.9995          1.0000            1.0000        41.62
HnswRaBitQ-m32-ex8 (self)                              4_793.75     9_671.98    14_465.72       0.9993          1.0001            1.0000        41.62
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
Exhaustive (query)                                       100.53     1_841.78     1_942.31       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.53     6_240.17     6_340.70       1.0000          1.0000            1.0000       146.48
QG-d32-l64-ef15 (query)                                6_030.52       105.20     6_135.71       0.9898          1.0005            1.0000       311.66
QG-d32-l64-ef30 (query)                                6_030.52       144.62     6_175.14       0.9998          1.0000            1.0000       311.66
QG-d32-l64-ef60 (query)                                6_030.52       225.07     6_255.59       1.0000          1.0000            1.0000       311.66
QG-d32-l64-ef120 (query)                               6_030.52       394.69     6_425.21       1.0000          1.0000            1.0000       311.66
QG-d32-l64 (self)                                      6_030.52       668.22     6_698.74       1.0000          1.0000            1.0000       311.66
QG-d32-l128-ef15 (query)                               9_483.88       109.52     9_593.40       0.9892          1.0005            1.0000       311.66
QG-d32-l128-ef30 (query)                               9_483.88       143.49     9_627.37       0.9998          1.0000            1.0000       311.66
QG-d32-l128-ef60 (query)                               9_483.88       223.63     9_707.51       1.0000          1.0000            1.0000       311.66
QG-d32-l128-ef120 (query)                              9_483.88       397.47     9_881.35       1.0000          1.0000            1.0000       311.66
QG-d32-l128 (self)                                     9_483.88       665.69    10_149.56       1.0000          1.0000            1.0000       311.66
QG-d64-l64-ef15 (query)                                7_607.99       116.76     7_724.75       0.9910          1.0004            1.0000       476.46
QG-d64-l64-ef30 (query)                                7_607.99       163.85     7_771.84       0.9999          1.0000            1.0000       476.46
QG-d64-l64-ef60 (query)                                7_607.99       258.96     7_866.95       1.0000          1.0000            1.0000       476.46
QG-d64-l64-ef120 (query)                               7_607.99       468.13     8_076.13       1.0000          1.0000            1.0000       476.46
QG-d64-l64 (self)                                      7_607.99       788.01     8_396.00       1.0000          1.0000            1.0000       476.46
QG-d64-l128-ef15 (query)                              11_850.45       116.80    11_967.25       0.9909          1.0004            1.0000       476.46
QG-d64-l128-ef30 (query)                              11_850.45       161.59    12_012.04       0.9999          1.0000            1.0000       476.46
QG-d64-l128-ef60 (query)                              11_850.45       265.32    12_115.77       1.0000          1.0000            1.0000       476.46
QG-d64-l128-ef120 (query)                             11_850.45       467.84    12_318.29       1.0000          1.0000            1.0000       476.46
QG-d64-l128 (self)                                    11_850.45       787.52    12_637.97       1.0000          1.0000            1.0000       476.46
HnswRaBitQ-m16-ex1-ef15 (query)                        3_740.02     1_068.75     4_808.77       0.9500          1.0150            1.0014        17.45
HnswRaBitQ-m16-ex1-ef30 (query)                        3_740.02     1_530.16     5_270.18       0.9512          1.0128            1.0013        17.45
HnswRaBitQ-m16-ex1-ef60 (query)                        3_740.02     2_185.23     5_925.25       0.9516          1.0088            1.0013        17.45
HnswRaBitQ-m16-ex1-ef120 (query)                       3_740.02     3_414.31     7_154.33       0.9517          1.0084            1.0013        17.45
HnswRaBitQ-m16-ex1 (self)                              3_740.02    11_712.87    15_452.88       0.9212          1.0165            1.0063        17.45
HnswRaBitQ-m16-ex3-ef15 (query)                        4_029.61     1_087.24     5_116.85       0.9763          1.0046            1.0000        26.61
HnswRaBitQ-m16-ex3-ef30 (query)                        4_029.61     1_507.79     5_537.40       0.9779          1.0034            1.0000        26.61
HnswRaBitQ-m16-ex3-ef60 (query)                        4_029.61     2_207.48     6_237.09       0.9781          1.0033            1.0000        26.61
HnswRaBitQ-m16-ex3-ef120 (query)                       4_029.61     3_381.41     7_411.03       0.9781          1.0030            1.0000        26.61
HnswRaBitQ-m16-ex3 (self)                              4_029.61    11_769.91    15_799.52       0.9707          1.0037            1.0000        26.61
HnswRaBitQ-m16-ex5-ef15 (query)                        4_198.48     1_104.55     5_303.03       0.9902          1.0399            1.0000        35.76
HnswRaBitQ-m16-ex5-ef30 (query)                        4_198.48     1_529.61     5_728.09       0.9926          1.0360            1.0000        35.76
HnswRaBitQ-m16-ex5-ef60 (query)                        4_198.48     2_219.22     6_417.70       0.9933          1.0270            1.0000        35.76
HnswRaBitQ-m16-ex5-ef120 (query)                       4_198.48     3_387.80     7_586.28       0.9938          1.0196            1.0000        35.76
HnswRaBitQ-m16-ex5 (self)                              4_198.48    11_876.47    16_074.95       0.9919          1.0205            1.0000        35.76
HnswRaBitQ-m16-ex8-ef15 (query)                        5_970.27     1_089.93     7_060.20       0.9955          1.0135            1.0000        49.50
HnswRaBitQ-m16-ex8-ef30 (query)                        5_970.27     1_506.97     7_477.24       0.9982          1.0099            1.0000        49.50
HnswRaBitQ-m16-ex8-ef60 (query)                        5_970.27     2_217.73     8_188.01       0.9990          1.0039            1.0000        49.50
HnswRaBitQ-m16-ex8-ef120 (query)                       5_970.27     3_405.85     9_376.12       0.9991          1.0029            1.0000        49.50
HnswRaBitQ-m16-ex8 (self)                              5_970.27    11_818.38    17_788.65       0.9988          1.0040            1.0000        49.50
HnswRaBitQ-m32-ex1-ef15 (query)                        3_850.40     1_377.43     5_227.83       0.9513          1.0035            1.0014        23.56
HnswRaBitQ-m32-ex1-ef30 (query)                        3_850.40     1_798.50     5_648.90       0.9520          1.0033            1.0013        23.56
HnswRaBitQ-m32-ex1-ef60 (query)                        3_850.40     2_518.90     6_369.30       0.9521          1.0033            1.0013        23.56
HnswRaBitQ-m32-ex1-ef120 (query)                       3_850.40     3_750.37     7_600.77       0.9521          1.0033            1.0013        23.56
HnswRaBitQ-m32-ex1 (self)                              3_850.40    13_638.09    17_488.49       0.9217          1.0104            1.0063        23.56
HnswRaBitQ-m32-ex3-ef15 (query)                        4_068.65     1_387.81     5_456.46       0.9770          1.0017            1.0000        32.71
HnswRaBitQ-m32-ex3-ef30 (query)                        4_068.65     1_806.15     5_874.80       0.9782          1.0015            1.0000        32.71
HnswRaBitQ-m32-ex3-ef60 (query)                        4_068.65     2_541.09     6_609.73       0.9783          1.0015            1.0000        32.71
HnswRaBitQ-m32-ex3-ef120 (query)                       4_068.65     3_768.86     7_837.50       0.9783          1.0011            1.0000        32.71
HnswRaBitQ-m32-ex3 (self)                              4_068.65    13_635.75    17_704.40       0.9710          1.0015            1.0000        32.71
HnswRaBitQ-m32-ex5-ef15 (query)                        4_294.00     1_370.76     5_664.76       0.9935          1.0011            1.0000        41.87
HnswRaBitQ-m32-ex5-ef30 (query)                        4_294.00     1_813.56     6_107.56       0.9950          1.0009            1.0000        41.87
HnswRaBitQ-m32-ex5-ef60 (query)                        4_294.00     2_542.14     6_836.14       0.9951          1.0009            1.0000        41.87
HnswRaBitQ-m32-ex5-ef120 (query)                       4_294.00     3_767.71     8_061.71       0.9951          1.0009            1.0000        41.87
HnswRaBitQ-m32-ex5 (self)                              4_294.00    13_659.95    17_953.95       0.9933          1.0005            1.0000        41.87
HnswRaBitQ-m32-ex8-ef15 (query)                        6_097.43     1_368.55     7_465.97       0.9975          1.0011            1.0000        55.60
HnswRaBitQ-m32-ex8-ef30 (query)                        6_097.43     1_808.09     7_905.52       0.9992          1.0010            1.0000        55.60
HnswRaBitQ-m32-ex8-ef60 (query)                        6_097.43     2_544.75     8_642.18       0.9993          1.0009            1.0000        55.60
HnswRaBitQ-m32-ex8-ef120 (query)                       6_097.43     3_806.76     9_904.19       0.9994          1.0005            1.0000        55.60
HnswRaBitQ-m32-ex8 (self)                              6_097.43    13_620.04    19_717.46       0.9992          1.0004            1.0000        55.60
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
Exhaustive (query)                                        32.54       703.38       735.92       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.54     2_379.22     2_411.76       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              162.51       361.27       523.78       0.0971          1.7176            1.5958         7.12
ExhaustiveTQ-b2-rf5 (query)                              162.51       437.17       599.68       0.2336          1.2025            1.2204         7.12
ExhaustiveTQ-b2-rf10 (query)                             162.51       569.58       732.09       0.2853          1.1453            1.1620         7.12
ExhaustiveTQ-b2-rf20 (query)                             162.51       943.56     1_106.07       0.3809          1.0970            1.0941         7.12
ExhaustiveTQ-b2 (self)                                   162.51     3_095.81     3_258.32       0.3814          1.0980            1.0957         7.12
ExhaustiveTQ-b4-rf0 (query)                              255.49       576.99       832.48       0.1094          1.5328            1.4997        13.22
ExhaustiveTQ-b4-rf5 (query)                              255.49       671.11       926.60       0.2368          1.1884            1.2090        13.22
ExhaustiveTQ-b4-rf10 (query)                             255.49       795.25     1_050.74       0.2884          1.1372            1.1543        13.22
ExhaustiveTQ-b4-rf20 (query)                             255.49     1_164.84     1_420.33       0.3823          1.0940            1.0970        13.22
ExhaustiveTQ-b4 (self)                                   255.49     3_898.65     4_154.14       0.3841          1.0938            1.0948        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                          939.20       105.46     1_044.65       0.0971          1.7176            1.5958         7.80
IVF-TQ-b2-nl158-np12-rf0 (query)                         939.20       116.90     1_056.10       0.0971          1.7176            1.5958         7.80
IVF-TQ-b2-nl158-np17-rf0 (query)                         939.20       125.30     1_064.49       0.0971          1.7176            1.5958         7.80
IVF-TQ-b2-nl158-np7-rf10 (query)                         939.20       302.52     1_241.71       0.2853          1.1453            1.1620         7.80
IVF-TQ-b2-nl158-np7-rf20 (query)                         939.20       632.96     1_572.16       0.3809          1.0970            1.0941         7.80
IVF-TQ-b2-nl158-np12-rf10 (query)                        939.20       317.21     1_256.41       0.2853          1.1453            1.1620         7.80
IVF-TQ-b2-nl158-np12-rf20 (query)                        939.20       642.28     1_581.47       0.3809          1.0970            1.0941         7.80
IVF-TQ-b2-nl158-np17-rf10 (query)                        939.20       320.69     1_259.89       0.2853          1.1453            1.1620         7.80
IVF-TQ-b2-nl158-np17-rf20 (query)                        939.20       662.58     1_601.78       0.3809          1.0970            1.0941         7.80
IVF-TQ-b2-nl158 (self)                                   939.20     1_050.57     1_989.77       0.3815          1.0980            1.0957         7.80
IVF-TQ-b2-nl223-np11-rf0 (query)                         682.84       110.86       793.70       0.0971          1.7164            1.5942         7.93
IVF-TQ-b2-nl223-np14-rf0 (query)                         682.84       120.28       803.12       0.0971          1.7176            1.5958         7.93
IVF-TQ-b2-nl223-np21-rf0 (query)                         682.84       133.13       815.97       0.0971          1.7176            1.5958         7.93
IVF-TQ-b2-nl223-np11-rf10 (query)                        682.84       295.71       978.55       0.2855          1.1450            1.1618         7.93
IVF-TQ-b2-nl223-np11-rf20 (query)                        682.84       541.68     1_224.52       0.3813          1.0967            1.0934         7.93
IVF-TQ-b2-nl223-np14-rf10 (query)                        682.84       288.71       971.55       0.2853          1.1453            1.1620         7.93
IVF-TQ-b2-nl223-np14-rf20 (query)                        682.84       562.24     1_245.08       0.3809          1.0970            1.0941         7.93
IVF-TQ-b2-nl223-np21-rf10 (query)                        682.84       303.34       986.18       0.2853          1.1453            1.1620         7.93
IVF-TQ-b2-nl223-np21-rf20 (query)                        682.84       585.35     1_268.19       0.3809          1.0970            1.0941         7.93
IVF-TQ-b2-nl223 (self)                                   682.84     1_036.73     1_719.57       0.3815          1.0980            1.0957         7.93
IVF-TQ-b2-nl316-np15-rf0 (query)                         902.16       116.39     1_018.55       0.0973          1.6435            1.5781         8.10
IVF-TQ-b2-nl316-np17-rf0 (query)                         902.16       119.87     1_022.03       0.0972          1.7163            1.5957         8.10
IVF-TQ-b2-nl316-np25-rf0 (query)                         902.16       133.34     1_035.50       0.0971          1.7176            1.5958         8.10
IVF-TQ-b2-nl316-np15-rf10 (query)                        902.16       289.51     1_191.68       0.2858          1.1447            1.1615         8.10
IVF-TQ-b2-nl316-np15-rf20 (query)                        902.16       532.87     1_435.03       0.3817          1.0965            1.0931         8.10
IVF-TQ-b2-nl316-np17-rf10 (query)                        902.16       290.62     1_192.78       0.2853          1.1453            1.1620         8.10
IVF-TQ-b2-nl316-np17-rf20 (query)                        902.16       539.20     1_441.36       0.3809          1.0970            1.0941         8.10
IVF-TQ-b2-nl316-np25-rf10 (query)                        902.16       300.48     1_202.64       0.2853          1.1453            1.1620         8.10
IVF-TQ-b2-nl316-np25-rf20 (query)                        902.16       583.59     1_485.76       0.3809          1.0970            1.0941         8.10
IVF-TQ-b2-nl316 (self)                                   902.16     1_040.29     1_942.45       0.3815          1.0980            1.0957         8.10
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_046.27       145.39     1_191.66       0.1094          1.5328            1.4997        14.05
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_046.27       162.57     1_208.83       0.1094          1.5328            1.4996        14.05
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_046.27       176.82     1_223.09       0.1094          1.5328            1.4996        14.05
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_046.27       354.93     1_401.19       0.2884          1.1372            1.1543        14.05
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_046.27       680.85     1_727.12       0.3823          1.0940            1.0970        14.05
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_046.27       369.95     1_416.22       0.2884          1.1372            1.1543        14.05
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_046.27       706.90     1_753.16       0.3823          1.0940            1.0970        14.05
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_046.27       382.93     1_429.19       0.2884          1.1372            1.1543        14.05
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_046.27       723.23     1_769.49       0.3823          1.0940            1.0970        14.05
IVF-TQ-b4-nl158 (self)                                 1_046.27     1_060.72     2_106.98       0.3841          1.0938            1.0948        14.05
IVF-TQ-b4-nl223-np11-rf0 (query)                         731.12       153.96       885.08       0.1094          1.5317            1.4988        14.25
IVF-TQ-b4-nl223-np14-rf0 (query)                         731.12       167.85       898.96       0.1094          1.5329            1.4997        14.25
IVF-TQ-b4-nl223-np21-rf0 (query)                         731.12       187.72       918.83       0.1094          1.5328            1.4996        14.25
IVF-TQ-b4-nl223-np11-rf10 (query)                        731.12       327.51     1_058.62       0.2886          1.1370            1.1541        14.25
IVF-TQ-b4-nl223-np11-rf20 (query)                        731.12       597.21     1_328.33       0.3826          1.0939            1.0966        14.25
IVF-TQ-b4-nl223-np14-rf10 (query)                        731.12       342.04     1_073.15       0.2884          1.1372            1.1543        14.25
IVF-TQ-b4-nl223-np14-rf20 (query)                        731.12       619.82     1_350.94       0.3823          1.0940            1.0970        14.25
IVF-TQ-b4-nl223-np21-rf10 (query)                        731.12       365.31     1_096.42       0.2884          1.1372            1.1543        14.25
IVF-TQ-b4-nl223-np21-rf20 (query)                        731.12       652.31     1_383.42       0.3823          1.0940            1.0970        14.25
IVF-TQ-b4-nl223 (self)                                   731.12     1_061.35     1_792.46       0.3841          1.0938            1.0948        14.25
IVF-TQ-b4-nl316-np15-rf0 (query)                         927.64       158.43     1_086.07       0.1094          1.5304            1.4978        14.49
IVF-TQ-b4-nl316-np17-rf0 (query)                         927.64       165.70     1_093.34       0.1094          1.5328            1.4996        14.49
IVF-TQ-b4-nl316-np25-rf0 (query)                         927.64       188.61     1_116.25       0.1094          1.5328            1.4997        14.49
IVF-TQ-b4-nl316-np15-rf10 (query)                        927.64       338.78     1_266.42       0.2886          1.1369            1.1542        14.49
IVF-TQ-b4-nl316-np15-rf20 (query)                        927.64       597.07     1_524.71       0.3828          1.0938            1.0967        14.49
IVF-TQ-b4-nl316-np17-rf10 (query)                        927.64       340.78     1_268.42       0.2884          1.1372            1.1543        14.49
IVF-TQ-b4-nl316-np17-rf20 (query)                        927.64       598.42     1_526.06       0.3823          1.0940            1.0970        14.49
IVF-TQ-b4-nl316-np25-rf10 (query)                        927.64       362.17     1_289.81       0.2884          1.1372            1.1543        14.49
IVF-TQ-b4-nl316-np25-rf20 (query)                        927.64       666.66     1_594.29       0.3823          1.0940            1.0970        14.49
IVF-TQ-b4-nl316 (self)                                   927.64     1_073.04     2_000.68       0.3841          1.0938            1.0948        14.49
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
Exhaustive (query)                                        68.09     1_293.41     1_361.50       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.09     4_330.71     4_398.80       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              338.66       626.81       965.47       0.1207          1.3711            1.3320        13.97
ExhaustiveTQ-b2-rf5 (query)                              338.66       718.79     1_057.45       0.2421          1.1334            1.1574        13.97
ExhaustiveTQ-b2-rf10 (query)                             338.66       861.30     1_199.96       0.2934          1.0981            1.1177        13.97
ExhaustiveTQ-b2-rf20 (query)                             338.66     1_249.97     1_588.63       0.3880          1.0664            1.0469        13.97
ExhaustiveTQ-b2 (self)                                   338.66     4_072.18     4_410.84       0.3879          1.0667            1.0471        13.97
ExhaustiveTQ-b4-rf0 (query)                              516.17     1_104.02     1_620.19       0.1315          1.3172            1.3127        26.18
ExhaustiveTQ-b4-rf5 (query)                              516.17     1_213.13     1_729.31       0.2471          1.1254            1.1483        26.18
ExhaustiveTQ-b4-rf10 (query)                             516.17     1_350.27     1_866.44       0.2970          1.0929            1.0980        26.18
ExhaustiveTQ-b4-rf20 (query)                             516.17     1_757.28     2_273.45       0.3883          1.0643            1.0492        26.18
ExhaustiveTQ-b4 (self)                                   516.17     5_700.02     6_216.19       0.3881          1.0646            1.0495        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_894.47       189.74     2_084.22       0.1207          1.3711            1.3320        14.95
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_894.47       205.13     2_099.60       0.1207          1.3711            1.3320        14.95
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_894.47       230.38     2_124.86       0.1207          1.3711            1.3320        14.95
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_894.47       409.12     2_303.59       0.2934          1.0981            1.1177        14.95
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_894.47       738.80     2_633.28       0.3880          1.0664            1.0469        14.95
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_894.47       423.99     2_318.46       0.2934          1.0981            1.1178        14.95
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_894.47       776.36     2_670.84       0.3880          1.0664            1.0469        14.95
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_894.47       439.02     2_333.49       0.2934          1.0981            1.1177        14.95
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_894.47       826.57     2_721.04       0.3880          1.0664            1.0469        14.95
IVF-TQ-b2-nl158 (self)                                 1_894.47     1_392.43     3_286.91       0.3879          1.0667            1.0471        14.95
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_177.10       206.63     1_383.73       0.1208          1.3699            1.3300        15.19
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_177.10       224.53     1_401.63       0.1207          1.3711            1.3320        15.19
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_177.10       246.14     1_423.24       0.1207          1.3711            1.3320        15.19
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_177.10       407.74     1_584.84       0.2937          1.0979            1.1176        15.19
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_177.10       700.30     1_877.39       0.3887          1.0662            1.0467        15.19
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_177.10       415.87     1_592.97       0.2934          1.0981            1.1177        15.19
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_177.10       717.49     1_894.58       0.3880          1.0664            1.0469        15.19
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_177.10       433.20     1_610.29       0.2934          1.0981            1.1178        15.19
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_177.10       756.74     1_933.83       0.3880          1.0664            1.0469        15.19
IVF-TQ-b2-nl223 (self)                                 1_177.10     1_405.63     2_582.73       0.3879          1.0667            1.0471        15.19
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_498.05       214.68     1_712.73       0.1208          1.3689            1.3287        15.56
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_498.05       219.99     1_718.04       0.1208          1.3707            1.3312        15.56
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_498.05       237.96     1_736.01       0.1207          1.3711            1.3320        15.56
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_498.05       401.96     1_900.01       0.2939          1.0977            1.1175        15.56
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_498.05       670.42     2_168.46       0.3892          1.0660            1.0465        15.56
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_498.05       405.07     1_903.12       0.2935          1.0980            1.1177        15.56
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_498.05       778.54     2_276.58       0.3882          1.0664            1.0469        15.56
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_498.05       439.18     1_937.23       0.2934          1.0981            1.1178        15.56
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_498.05       733.20     2_231.24       0.3880          1.0664            1.0469        15.56
IVF-TQ-b2-nl316 (self)                                 1_498.05     1_429.19     2_927.24       0.3879          1.0667            1.0471        15.56
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_974.76       267.07     2_241.83       0.1315          1.3172            1.3127        27.44
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_974.76       311.84     2_286.61       0.1315          1.3172            1.3127        27.44
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_974.76       319.13     2_293.89       0.1315          1.3172            1.3127        27.44
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_974.76       506.74     2_481.50       0.2970          1.0929            1.0979        27.44
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_974.76       832.93     2_807.69       0.3883          1.0643            1.0492        27.44
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_974.76       529.77     2_504.53       0.2970          1.0929            1.0979        27.44
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_974.76       880.89     2_855.65       0.3882          1.0643            1.0492        27.44
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_974.76       552.34     2_527.10       0.2970          1.0929            1.0979        27.44
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_974.76       916.40     2_891.16       0.3883          1.0643            1.0492        27.44
IVF-TQ-b4-nl158 (self)                                 1_974.76     1_568.35     3_543.11       0.3881          1.0646            1.0495        27.44
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_288.52       290.86     1_579.38       0.1315          1.3158            1.3116        27.79
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_288.52       308.64     1_597.15       0.1315          1.3172            1.3127        27.79
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_288.52       339.50     1_628.02       0.1315          1.3172            1.3127        27.79
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_288.52       497.36     1_785.88       0.2973          1.0926            1.0973        27.79
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_288.52       791.60     2_080.12       0.3889          1.0641            1.0489        27.79
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_288.52       508.94     1_797.46       0.2970          1.0929            1.0980        27.79
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_288.52       814.36     2_102.88       0.3883          1.0643            1.0492        27.79
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_288.52       553.15     1_841.66       0.2970          1.0929            1.0980        27.79
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_288.52       875.78     2_164.30       0.3883          1.0643            1.0492        27.79
IVF-TQ-b4-nl223 (self)                                 1_288.52     1_602.41     2_890.93       0.3881          1.0646            1.0495        27.79
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_627.42       300.37     1_927.79       0.1316          1.3151            1.3108        28.35
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_627.42       313.47     1_940.89       0.1315          1.3164            1.3121        28.35
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_627.42       357.67     1_985.08       0.1315          1.3172            1.3127        28.35
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_627.42       496.35     2_123.77       0.2976          1.0925            1.0966        28.35
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_627.42       776.71     2_404.12       0.3893          1.0639            1.0484        28.35
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_627.42       519.67     2_147.09       0.2971          1.0928            1.0977        28.35
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_627.42       797.97     2_425.39       0.3885          1.0642            1.0491        28.35
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_627.42       550.66     2_178.08       0.2970          1.0929            1.0980        28.35
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_627.42       865.01     2_492.43       0.3883          1.0643            1.0492        28.35
IVF-TQ-b4-nl316 (self)                                 1_627.42     1_630.19     3_257.61       0.3881          1.0646            1.0495        28.35
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
Exhaustive (query)                                       100.22     1_834.28     1_934.51       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.22     6_213.81     6_314.03       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              608.49       925.49     1_533.98       0.1292          1.2710            1.2627        21.33
ExhaustiveTQ-b2-rf5 (query)                              608.49     1_036.26     1_644.74       0.2468          1.1062            1.1332        21.33
ExhaustiveTQ-b2-rf10 (query)                             608.49     1_177.67     1_786.16       0.3000          1.0773            1.0631        21.33
ExhaustiveTQ-b2-rf20 (query)                             608.49     1_658.64     2_267.13       0.3957          1.0509            1.0334        21.33
ExhaustiveTQ-b2 (self)                                   608.49     5_170.40     5_778.88       0.3973          1.0507            1.0331        21.33
ExhaustiveTQ-b4-rf0 (query)                              746.09     1_751.74     2_497.83       0.1340          1.2532            1.2592        39.64
ExhaustiveTQ-b4-rf5 (query)                              746.09     1_834.16     2_580.25       0.2401          1.1136            1.1402        39.64
ExhaustiveTQ-b4-rf10 (query)                             746.09     1_996.83     2_742.93       0.2870          1.0888            1.1143        39.64
ExhaustiveTQ-b4-rf20 (query)                             746.09     2_362.77     3_108.86       0.3752          1.0657            1.0812        39.64
ExhaustiveTQ-b4 (self)                                   746.09     7_974.84     8_720.93       0.3767          1.0654            1.0638        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        2_721.17       296.00     3_017.17       0.1292          1.2710            1.2627        22.66
IVF-TQ-b2-nl158-np12-rf0 (query)                       2_721.17       313.80     3_034.96       0.1292          1.2710            1.2627        22.66
IVF-TQ-b2-nl158-np17-rf0 (query)                       2_721.17       332.42     3_053.59       0.1292          1.2710            1.2627        22.66
IVF-TQ-b2-nl158-np7-rf10 (query)                       2_721.17       516.19     3_237.36       0.3000          1.0773            1.0631        22.66
IVF-TQ-b2-nl158-np7-rf20 (query)                       2_721.17       868.01     3_589.17       0.3957          1.0509            1.0334        22.66
IVF-TQ-b2-nl158-np12-rf10 (query)                      2_721.17       543.81     3_264.97       0.3000          1.0773            1.0631        22.66
IVF-TQ-b2-nl158-np12-rf20 (query)                      2_721.17       914.04     3_635.21       0.3957          1.0509            1.0334        22.66
IVF-TQ-b2-nl158-np17-rf10 (query)                      2_721.17       582.60     3_303.76       0.3000          1.0774            1.0631        22.66
IVF-TQ-b2-nl158-np17-rf20 (query)                      2_721.17       941.51     3_662.67       0.3957          1.0509            1.0334        22.66
IVF-TQ-b2-nl158 (self)                                 2_721.17     1_795.97     4_517.14       0.3973          1.0507            1.0331        22.66
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_803.75       313.98     2_117.73       0.1292          1.2710            1.2627        23.04
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_803.75       319.76     2_123.51       0.1292          1.2710            1.2627        23.04
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_803.75       347.78     2_151.53       0.1292          1.2710            1.2627        23.04
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_803.75       524.21     2_327.96       0.3000          1.0774            1.0631        23.04
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_803.75       825.35     2_629.10       0.3957          1.0509            1.0334        23.04
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_803.75       571.47     2_375.22       0.3000          1.0774            1.0631        23.04
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_803.75       853.21     2_656.96       0.3957          1.0509            1.0334        23.04
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_803.75       574.36     2_378.11       0.3000          1.0774            1.0631        23.04
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_803.75       905.09     2_708.83       0.3957          1.0509            1.0334        23.04
IVF-TQ-b2-nl223 (self)                                 1_803.75     1_827.38     3_631.13       0.3973          1.0507            1.0331        23.04
IVF-TQ-b2-nl316-np15-rf0 (query)                       2_208.94       322.34     2_531.28       0.1292          1.2709            1.2627        23.57
IVF-TQ-b2-nl316-np17-rf0 (query)                       2_208.94       328.19     2_537.13       0.1292          1.2710            1.2628        23.57
IVF-TQ-b2-nl316-np25-rf0 (query)                       2_208.94       366.48     2_575.42       0.1292          1.2710            1.2628        23.57
IVF-TQ-b2-nl316-np15-rf10 (query)                      2_208.94       535.93     2_744.86       0.3000          1.0773            1.0631        23.57
IVF-TQ-b2-nl316-np15-rf20 (query)                      2_208.94       828.02     3_036.95       0.3957          1.0509            1.0334        23.57
IVF-TQ-b2-nl316-np17-rf10 (query)                      2_208.94       538.88     2_747.81       0.3000          1.0774            1.0632        23.57
IVF-TQ-b2-nl316-np17-rf20 (query)                      2_208.94       842.70     3_051.64       0.3957          1.0509            1.0334        23.57
IVF-TQ-b2-nl316-np25-rf10 (query)                      2_208.94       585.62     2_794.55       0.3000          1.0774            1.0631        23.57
IVF-TQ-b2-nl316-np25-rf20 (query)                      2_208.94       894.82     3_103.75       0.3957          1.0509            1.0334        23.57
IVF-TQ-b2-nl316 (self)                                 2_208.94     1_843.04     4_051.98       0.3973          1.0507            1.0331        23.57
IVF-TQ-b4-nl158-np7-rf0 (query)                        2_799.00       414.89     3_213.89       0.1340          1.2532            1.2592        41.46
IVF-TQ-b4-nl158-np12-rf0 (query)                       2_799.00       455.34     3_254.35       0.1340          1.2532            1.2592        41.46
IVF-TQ-b4-nl158-np17-rf0 (query)                       2_799.00       492.41     3_291.42       0.1340          1.2532            1.2592        41.46
IVF-TQ-b4-nl158-np7-rf10 (query)                       2_799.00       658.59     3_457.59       0.2870          1.0888            1.1143        41.46
IVF-TQ-b4-nl158-np7-rf20 (query)                       2_799.00     1_005.50     3_804.51       0.3752          1.0657            1.0812        41.46
IVF-TQ-b4-nl158-np12-rf10 (query)                      2_799.00       702.15     3_501.15       0.2870          1.0888            1.1143        41.46
IVF-TQ-b4-nl158-np12-rf20 (query)                      2_799.00     1_071.49     3_870.49       0.3752          1.0657            1.0812        41.46
IVF-TQ-b4-nl158-np17-rf10 (query)                      2_799.00       741.10     3_540.10       0.2870          1.0888            1.1143        41.46
IVF-TQ-b4-nl158-np17-rf20 (query)                      2_799.00     1_129.26     3_928.26       0.3752          1.0657            1.0812        41.46
IVF-TQ-b4-nl158 (self)                                 2_799.00     2_048.58     4_847.58       0.3767          1.0654            1.0637        41.46
IVF-TQ-b4-nl223-np11-rf0 (query)                       2_049.54       442.96     2_492.50       0.1340          1.2532            1.2592        42.04
IVF-TQ-b4-nl223-np14-rf0 (query)                       2_049.54       474.71     2_524.25       0.1340          1.2532            1.2592        42.04
IVF-TQ-b4-nl223-np21-rf0 (query)                       2_049.54       522.74     2_572.28       0.1340          1.2532            1.2592        42.04
IVF-TQ-b4-nl223-np11-rf10 (query)                      2_049.54       679.27     2_728.81       0.2870          1.0888            1.1143        42.04
IVF-TQ-b4-nl223-np11-rf20 (query)                      2_049.54       989.76     3_039.30       0.3752          1.0657            1.0812        42.04
IVF-TQ-b4-nl223-np14-rf10 (query)                      2_049.54       698.47     2_748.01       0.2870          1.0888            1.1143        42.04
IVF-TQ-b4-nl223-np14-rf20 (query)                      2_049.54     1_017.65     3_067.19       0.3752          1.0657            1.0812        42.04
IVF-TQ-b4-nl223-np21-rf10 (query)                      2_049.54       760.70     2_810.24       0.2870          1.0888            1.1143        42.04
IVF-TQ-b4-nl223-np21-rf20 (query)                      2_049.54     1_091.21     3_140.75       0.3752          1.0657            1.0812        42.04
IVF-TQ-b4-nl223 (self)                                 2_049.54     2_145.05     4_194.59       0.3766          1.0654            1.0638        42.04
IVF-TQ-b4-nl316-np15-rf0 (query)                       2_349.95       458.10     2_808.05       0.1340          1.2531            1.2592        42.81
IVF-TQ-b4-nl316-np17-rf0 (query)                       2_349.95       496.51     2_846.46       0.1340          1.2531            1.2592        42.81
IVF-TQ-b4-nl316-np25-rf0 (query)                       2_349.95       529.78     2_879.72       0.1340          1.2532            1.2592        42.81
IVF-TQ-b4-nl316-np15-rf10 (query)                      2_349.95       677.93     3_027.88       0.2870          1.0888            1.1143        42.81
IVF-TQ-b4-nl316-np15-rf20 (query)                      2_349.95       990.94     3_340.89       0.3753          1.0657            1.0812        42.81
IVF-TQ-b4-nl316-np17-rf10 (query)                      2_349.95       695.43     3_045.38       0.2870          1.0888            1.1143        42.81
IVF-TQ-b4-nl316-np17-rf20 (query)                      2_349.95     1_006.59     3_356.54       0.3752          1.0657            1.0812        42.81
IVF-TQ-b4-nl316-np25-rf10 (query)                      2_349.95       758.63     3_108.57       0.2870          1.0888            1.1143        42.81
IVF-TQ-b4-nl316-np25-rf20 (query)                      2_349.95     1_079.85     3_429.80       0.3752          1.0657            1.0812        42.81
IVF-TQ-b4-nl316 (self)                                 2_349.95     2_177.91     4_527.85       0.3767          1.0654            1.0637        42.81
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
Exhaustive (query)                                        32.63       734.31       766.93       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.63     2_435.42     2_468.04       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              181.64       367.00       548.65       0.0756          2.3283            1.9295         7.12
ExhaustiveTQ-b2-rf5 (query)                              181.64       439.05       620.70       0.2072          1.3307            1.3578         7.12
ExhaustiveTQ-b2-rf10 (query)                             181.64       567.39       749.03       0.2886          1.2206            1.2322         7.12
ExhaustiveTQ-b2-rf20 (query)                             181.64       941.06     1_122.70       0.4151          1.1328            1.1147         7.12
ExhaustiveTQ-b2 (self)                                   181.64     3_082.47     3_264.11       0.4136          1.1619            1.1367         7.12
ExhaustiveTQ-b4-rf0 (query)                              260.66       581.18       841.84       0.1023          1.7129            1.7532        13.22
ExhaustiveTQ-b4-rf5 (query)                              260.66       667.34       928.00       0.2385          1.2770            1.3000        13.22
ExhaustiveTQ-b4-rf10 (query)                             260.66       799.24     1_059.91       0.3202          1.1874            1.1953        13.22
ExhaustiveTQ-b4-rf20 (query)                             260.66     1_159.28     1_419.95       0.4481          1.1142            1.1029        13.22
ExhaustiveTQ-b4 (self)                                   260.66     3_814.01     4_074.67       0.4463          1.1397            1.1286        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_020.63       101.82     1_122.46       0.0756          2.3282            1.9295         7.81
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_020.63       114.59     1_135.23       0.0756          2.3283            1.9295         7.81
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_020.63       134.76     1_155.39       0.0756          2.3283            1.9295         7.81
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_020.63       299.78     1_320.41       0.2886          1.2206            1.2322         7.81
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_020.63       609.85     1_630.49       0.4151          1.1328            1.1147         7.81
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_020.63       307.25     1_327.89       0.2886          1.2206            1.2322         7.81
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_020.63       641.92     1_662.55       0.4151          1.1328            1.1147         7.81
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_020.63       345.77     1_366.40       0.2886          1.2206            1.2322         7.81
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_020.63       709.21     1_729.84       0.4151          1.1328            1.1147         7.81
IVF-TQ-b2-nl158 (self)                                 1_020.63     1_057.22     2_077.86       0.4136          1.1619            1.1367         7.81
IVF-TQ-b2-nl223-np11-rf0 (query)                         694.80       107.86       802.66       0.0756          2.3256            1.9254         7.94
IVF-TQ-b2-nl223-np14-rf0 (query)                         694.80       115.03       809.83       0.0756          2.3281            1.9295         7.94
IVF-TQ-b2-nl223-np21-rf0 (query)                         694.80       142.73       837.53       0.0756          2.3282            1.9295         7.94
IVF-TQ-b2-nl223-np11-rf10 (query)                        694.80       275.25       970.05       0.2890          1.2203            1.2319         7.94
IVF-TQ-b2-nl223-np11-rf20 (query)                        694.80       547.08     1_241.89       0.4156          1.1326            1.1144         7.94
IVF-TQ-b2-nl223-np14-rf10 (query)                        694.80       286.90       981.70       0.2886          1.2206            1.2322         7.94
IVF-TQ-b2-nl223-np14-rf20 (query)                        694.80       567.90     1_262.70       0.4151          1.1328            1.1147         7.94
IVF-TQ-b2-nl223-np21-rf10 (query)                        694.80       326.56     1_021.36       0.2886          1.2206            1.2322         7.94
IVF-TQ-b2-nl223-np21-rf20 (query)                        694.80       631.71     1_326.51       0.4151          1.1328            1.1147         7.94
IVF-TQ-b2-nl223 (self)                                   694.80     1_051.45     1_746.26       0.4136          1.1619            1.1367         7.94
IVF-TQ-b2-nl316-np15-rf0 (query)                         914.34       112.41     1_026.76       0.0757          2.3274            1.9289         8.11
IVF-TQ-b2-nl316-np17-rf0 (query)                         914.34       117.15     1_031.49       0.0756          2.3282            1.9294         8.11
IVF-TQ-b2-nl316-np25-rf0 (query)                         914.34       143.76     1_058.10       0.0756          2.3282            1.9295         8.11
IVF-TQ-b2-nl316-np15-rf10 (query)                        914.34       272.69     1_187.03       0.2892          1.2202            1.2317         8.11
IVF-TQ-b2-nl316-np15-rf20 (query)                        914.34       519.99     1_434.33       0.4159          1.1325            1.1142         8.11
IVF-TQ-b2-nl316-np17-rf10 (query)                        914.34       282.21     1_196.56       0.2887          1.2206            1.2322         8.11
IVF-TQ-b2-nl316-np17-rf20 (query)                        914.34       533.96     1_448.30       0.4151          1.1328            1.1147         8.11
IVF-TQ-b2-nl316-np25-rf10 (query)                        914.34       307.09     1_221.43       0.2886          1.2206            1.2322         8.11
IVF-TQ-b2-nl316-np25-rf20 (query)                        914.34       589.71     1_504.05       0.4151          1.1328            1.1147         8.11
IVF-TQ-b2-nl316 (self)                                   914.34     1_063.00     1_977.34       0.4136          1.1619            1.1367         8.11
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_062.81       139.56     1_202.37       0.1023          1.7129            1.7532        14.06
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_062.81       158.60     1_221.41       0.1023          1.7129            1.7532        14.06
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_062.81       191.29     1_254.10       0.1023          1.7129            1.7532        14.06
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_062.81       342.30     1_405.11       0.3202          1.1874            1.1953        14.06
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_062.81       664.23     1_727.04       0.4481          1.1142            1.1029        14.06
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_062.81       371.45     1_434.26       0.3202          1.1874            1.1953        14.06
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_062.81       694.65     1_757.46       0.4481          1.1142            1.1029        14.06
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_062.81       413.86     1_476.67       0.3202          1.1873            1.1953        14.06
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_062.81       780.10     1_842.91       0.4481          1.1142            1.1029        14.06
IVF-TQ-b4-nl158 (self)                                 1_062.81     1_081.66     2_144.47       0.4463          1.1397            1.1286        14.06
IVF-TQ-b4-nl223-np11-rf0 (query)                         787.43       150.40       937.83       0.1023          1.7109            1.7520        14.27
IVF-TQ-b4-nl223-np14-rf0 (query)                         787.43       159.33       946.76       0.1023          1.7129            1.7532        14.27
IVF-TQ-b4-nl223-np21-rf0 (query)                         787.43       200.46       987.89       0.1023          1.7129            1.7532        14.27
IVF-TQ-b4-nl223-np11-rf10 (query)                        787.43       324.89     1_112.32       0.3205          1.1871            1.1949        14.27
IVF-TQ-b4-nl223-np11-rf20 (query)                        787.43       598.22     1_385.65       0.4486          1.1140            1.1028        14.27
IVF-TQ-b4-nl223-np14-rf10 (query)                        787.43       339.19     1_126.62       0.3202          1.1873            1.1953        14.27
IVF-TQ-b4-nl223-np14-rf20 (query)                        787.43       621.68     1_409.12       0.4481          1.1142            1.1029        14.27
IVF-TQ-b4-nl223-np21-rf10 (query)                        787.43       392.50     1_179.93       0.3201          1.1874            1.1953        14.27
IVF-TQ-b4-nl223-np21-rf20 (query)                        787.43       701.32     1_488.76       0.4481          1.1142            1.1029        14.27
IVF-TQ-b4-nl223 (self)                                   787.43     1_091.19     1_878.62       0.4463          1.1397            1.1286        14.27
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_008.74       155.24     1_163.98       0.1023          1.7112            1.7520        14.52
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_008.74       159.46     1_168.20       0.1023          1.7121            1.7528        14.52
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_008.74       193.50     1_202.24       0.1023          1.7129            1.7532        14.52
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_008.74       325.71     1_334.45       0.3207          1.1869            1.1949        14.52
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_008.74       572.73     1_581.46       0.4491          1.1138            1.1025        14.52
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_008.74       331.32     1_340.06       0.3203          1.1873            1.1953        14.52
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_008.74       582.73     1_591.46       0.4482          1.1142            1.1029        14.52
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_008.74       384.01     1_392.75       0.3202          1.1873            1.1953        14.52
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_008.74       635.80     1_644.54       0.4481          1.1142            1.1029        14.52
IVF-TQ-b4-nl316 (self)                                 1_008.74     1_082.12     2_090.86       0.4463          1.1397            1.1286        14.52
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
Exhaustive (query)                                        68.92     1_275.38     1_344.30       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         68.92     4_270.34     4_339.26       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              336.32       620.29       956.61       0.0844          1.6539            1.5906        13.97
ExhaustiveTQ-b2-rf5 (query)                              336.32       713.03     1_049.35       0.2173          1.2230            1.2549        13.97
ExhaustiveTQ-b2-rf10 (query)                             336.32       850.43     1_186.75       0.2887          1.1550            1.1707        13.97
ExhaustiveTQ-b2-rf20 (query)                             336.32     1_278.74     1_615.06       0.4020          1.0974            1.0847        13.97
ExhaustiveTQ-b2 (self)                                   336.32     4_116.27     4_452.59       0.4025          1.1135            1.0971        13.97
ExhaustiveTQ-b4-rf0 (query)                              458.26     1_103.63     1_561.90       0.1044          1.5026            1.5346        26.18
ExhaustiveTQ-b4-rf5 (query)                              458.26     1_207.11     1_665.37       0.2294          1.2110            1.2410        26.18
ExhaustiveTQ-b4-rf10 (query)                             458.26     1_357.83     1_816.09       0.2943          1.1499            1.1675        26.18
ExhaustiveTQ-b4-rf20 (query)                             458.26     1_734.46     2_192.72       0.4029          1.0975            1.0929        26.18
ExhaustiveTQ-b4 (self)                                   458.26     5_703.57     6_161.83       0.4038          1.1130            1.1087        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_845.14       185.23     2_030.37       0.0844          1.6539            1.5906        14.95
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_845.14       202.11     2_047.26       0.0844          1.6539            1.5906        14.95
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_845.14       219.63     2_064.77       0.0844          1.6539            1.5906        14.95
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_845.14       404.86     2_250.00       0.2887          1.1550            1.1707        14.95
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_845.14       725.36     2_570.50       0.4020          1.0974            1.0847        14.95
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_845.14       421.70     2_266.84       0.2887          1.1550            1.1707        14.95
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_845.14       761.56     2_606.70       0.4020          1.0974            1.0847        14.95
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_845.14       439.57     2_284.71       0.2887          1.1550            1.1707        14.95
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_845.14       794.77     2_639.91       0.4020          1.0974            1.0847        14.95
IVF-TQ-b2-nl158 (self)                                 1_845.14     1_398.78     3_243.92       0.4025          1.1135            1.0971        14.95
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_270.65       198.67     1_469.32       0.0845          1.6537            1.5905        15.23
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_270.65       211.09     1_481.74       0.0844          1.6539            1.5906        15.23
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_270.65       239.63     1_510.29       0.0844          1.6539            1.5906        15.23
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_270.65       403.05     1_673.70       0.2887          1.1550            1.1707        15.23
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_270.65       684.80     1_955.45       0.4020          1.0974            1.0847        15.23
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_270.65       418.21     1_688.86       0.2887          1.1550            1.1707        15.23
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_270.65       720.18     1_990.84       0.4020          1.0974            1.0847        15.23
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_270.65       449.06     1_719.71       0.2887          1.1550            1.1707        15.23
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_270.65       766.27     2_036.92       0.4020          1.0974            1.0847        15.23
IVF-TQ-b2-nl223 (self)                                 1_270.65     1_420.23     2_690.88       0.4025          1.1135            1.0971        15.23
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_689.90       207.34     1_897.24       0.0844          1.6539            1.5906        15.57
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_689.90       213.47     1_903.37       0.0844          1.6539            1.5906        15.57
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_689.90       238.28     1_928.17       0.0844          1.6539            1.5906        15.57
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_689.90       398.41     2_088.31       0.2887          1.1550            1.1707        15.57
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_689.90       679.12     2_369.01       0.4020          1.0974            1.0847        15.57
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_689.90       405.68     2_095.58       0.2887          1.1550            1.1707        15.57
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_689.90       680.85     2_370.75       0.4020          1.0974            1.0847        15.57
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_689.90       433.78     2_123.68       0.2887          1.1550            1.1707        15.57
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_689.90       726.77     2_416.67       0.4020          1.0974            1.0847        15.57
IVF-TQ-b2-nl316 (self)                                 1_689.90     1_435.01     3_124.91       0.4025          1.1135            1.0971        15.57
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_904.16       256.59     2_160.75       0.1044          1.5026            1.5346        27.44
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_904.16       286.47     2_190.63       0.1044          1.5026            1.5346        27.44
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_904.16       315.25     2_219.41       0.1044          1.5026            1.5346        27.44
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_904.16       489.25     2_393.41       0.2943          1.1499            1.1675        27.44
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_904.16       821.90     2_726.06       0.4029          1.0975            1.0929        27.44
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_904.16       517.04     2_421.19       0.2943          1.1499            1.1675        27.44
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_904.16       863.98     2_768.14       0.4029          1.0975            1.0929        27.44
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_904.16       550.19     2_454.35       0.2943          1.1499            1.1675        27.44
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_904.16       912.50     2_816.66       0.4029          1.0975            1.0929        27.44
IVF-TQ-b4-nl158 (self)                                 1_904.16     1_550.19     3_454.35       0.4038          1.1130            1.1088        27.44
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_380.44       279.18     1_659.61       0.1044          1.5026            1.5346        27.87
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_380.44       298.54     1_678.98       0.1044          1.5026            1.5346        27.87
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_380.44       347.89     1_728.32       0.1044          1.5026            1.5346        27.87
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_380.44       494.43     1_874.87       0.2943          1.1499            1.1676        27.87
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_380.44       805.00     2_185.44       0.4029          1.0975            1.0929        27.87
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_380.44       532.72     1_913.15       0.2943          1.1499            1.1676        27.87
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_380.44       858.33     2_238.77       0.4029          1.0975            1.0929        27.87
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_380.44       591.69     1_972.12       0.2943          1.1499            1.1676        27.87
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_380.44       902.82     2_283.26       0.4029          1.0975            1.0929        27.87
IVF-TQ-b4-nl223 (self)                                 1_380.44     1_617.66     2_998.10       0.4038          1.1130            1.1088        27.87
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_812.83       287.58     2_100.40       0.1045          1.5026            1.5346        28.38
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_812.83       300.71     2_113.54       0.1044          1.5026            1.5346        28.38
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_812.83       342.99     2_155.82       0.1044          1.5026            1.5346        28.38
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_812.83       504.32     2_317.15       0.2943          1.1499            1.1675        28.38
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_812.83       781.54     2_594.37       0.4029          1.0975            1.0929        28.38
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_812.83       503.59     2_316.42       0.2943          1.1499            1.1675        28.38
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_812.83       785.14     2_597.97       0.4029          1.0975            1.0929        28.38
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_812.83       557.53     2_370.35       0.2943          1.1499            1.1675        28.38
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_812.83       856.73     2_669.55       0.4029          1.0975            1.0929        28.38
IVF-TQ-b4-nl316 (self)                                 1_812.83     1_645.58     3_458.41       0.4038          1.1130            1.1087        28.38
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
Exhaustive (query)                                       100.30     1_829.88     1_930.18       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                        100.30     6_182.15     6_282.45       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              603.93       954.76     1_558.69       0.0841          1.5107            1.4226        21.33
ExhaustiveTQ-b2-rf5 (query)                              603.93     1_025.07     1_629.01       0.2144          1.1739            1.2056        21.33
ExhaustiveTQ-b2-rf10 (query)                             603.93     1_174.91     1_778.84       0.2770          1.1267            1.1512        21.33
ExhaustiveTQ-b2-rf20 (query)                             603.93     1_585.36     2_189.29       0.3770          1.0843            1.0724        21.33
ExhaustiveTQ-b2 (self)                                   603.93     5_185.81     5_789.75       0.3767          1.0935            1.0803        21.33
ExhaustiveTQ-b4-rf0 (query)                              736.06     1_747.30     2_483.36       0.0986          1.4231            1.4109        39.64
ExhaustiveTQ-b4-rf5 (query)                              736.06     1_860.23     2_596.29       0.2167          1.1746            1.2047        39.64
ExhaustiveTQ-b4-rf10 (query)                             736.06     1_995.37     2_731.43       0.2692          1.1311            1.1557        39.64
ExhaustiveTQ-b4-rf20 (query)                             736.06     2_380.18     3_116.24       0.3605          1.0923            1.1071        39.64
ExhaustiveTQ-b4 (self)                                   736.06     7_836.41     8_572.47       0.3609          1.1024            1.1182        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        2_723.94       282.72     3_006.66       0.0841          1.5107            1.4226        22.62
IVF-TQ-b2-nl158-np12-rf0 (query)                       2_723.94       322.34     3_046.28       0.0841          1.5107            1.4226        22.62
IVF-TQ-b2-nl158-np17-rf0 (query)                       2_723.94       327.85     3_051.79       0.0841          1.5107            1.4226        22.62
IVF-TQ-b2-nl158-np7-rf10 (query)                       2_723.94       513.21     3_237.15       0.2770          1.1267            1.1512        22.62
IVF-TQ-b2-nl158-np7-rf20 (query)                       2_723.94       873.09     3_597.03       0.3770          1.0843            1.0724        22.62
IVF-TQ-b2-nl158-np12-rf10 (query)                      2_723.94       528.67     3_252.61       0.2770          1.1267            1.1512        22.62
IVF-TQ-b2-nl158-np12-rf20 (query)                      2_723.94       894.27     3_618.21       0.3770          1.0843            1.0724        22.62
IVF-TQ-b2-nl158-np17-rf10 (query)                      2_723.94       563.70     3_287.64       0.2771          1.1267            1.1512        22.62
IVF-TQ-b2-nl158-np17-rf20 (query)                      2_723.94       915.27     3_639.21       0.3770          1.0843            1.0724        22.62
IVF-TQ-b2-nl158 (self)                                 2_723.94     1_820.93     4_544.87       0.3767          1.0935            1.0803        22.62
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_877.61       297.21     2_174.82       0.0841          1.5107            1.4226        22.97
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_877.61       313.14     2_190.75       0.0841          1.5107            1.4226        22.97
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_877.61       347.80     2_225.41       0.0841          1.5107            1.4226        22.97
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_877.61       558.07     2_435.68       0.2771          1.1267            1.1512        22.97
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_877.61       858.73     2_736.34       0.3770          1.0843            1.0724        22.97
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_877.61       545.21     2_422.83       0.2771          1.1267            1.1512        22.97
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_877.61       873.03     2_750.64       0.3770          1.0843            1.0724        22.97
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_877.61       579.20     2_456.81       0.2771          1.1267            1.1512        22.97
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_877.61       928.27     2_805.89       0.3770          1.0843            1.0724        22.97
IVF-TQ-b2-nl223 (self)                                 1_877.61     1_855.59     3_733.20       0.3767          1.0935            1.0803        22.97
IVF-TQ-b2-nl316-np15-rf0 (query)                       2_470.03       311.08     2_781.11       0.0841          1.5107            1.4226        23.53
IVF-TQ-b2-nl316-np17-rf0 (query)                       2_470.03       320.97     2_791.00       0.0841          1.5107            1.4226        23.53
IVF-TQ-b2-nl316-np25-rf0 (query)                       2_470.03       349.22     2_819.25       0.0841          1.5107            1.4226        23.53
IVF-TQ-b2-nl316-np15-rf10 (query)                      2_470.03       533.97     3_004.00       0.2771          1.1267            1.1512        23.53
IVF-TQ-b2-nl316-np15-rf20 (query)                      2_470.03       848.92     3_318.95       0.3770          1.0843            1.0724        23.53
IVF-TQ-b2-nl316-np17-rf10 (query)                      2_470.03       548.44     3_018.47       0.2771          1.1267            1.1512        23.53
IVF-TQ-b2-nl316-np17-rf20 (query)                      2_470.03       851.11     3_321.14       0.3770          1.0843            1.0724        23.53
IVF-TQ-b2-nl316-np25-rf10 (query)                      2_470.03       577.76     3_047.78       0.2771          1.1267            1.1512        23.53
IVF-TQ-b2-nl316-np25-rf20 (query)                      2_470.03       904.39     3_374.42       0.3770          1.0843            1.0724        23.53
IVF-TQ-b2-nl316 (self)                                 2_470.03     1_890.80     4_360.82       0.3767          1.0935            1.0803        23.53
IVF-TQ-b4-nl158-np7-rf0 (query)                        2_813.42       405.92     3_219.34       0.0986          1.4231            1.4109        41.39
IVF-TQ-b4-nl158-np12-rf0 (query)                       2_813.42       437.47     3_250.89       0.0986          1.4231            1.4109        41.39
IVF-TQ-b4-nl158-np17-rf0 (query)                       2_813.42       488.50     3_301.91       0.0986          1.4231            1.4109        41.39
IVF-TQ-b4-nl158-np7-rf10 (query)                       2_813.42       639.04     3_452.45       0.2692          1.1311            1.1557        41.39
IVF-TQ-b4-nl158-np7-rf20 (query)                       2_813.42     1_003.57     3_816.99       0.3605          1.0923            1.1071        41.39
IVF-TQ-b4-nl158-np12-rf10 (query)                      2_813.42       676.10     3_489.52       0.2692          1.1311            1.1557        41.39
IVF-TQ-b4-nl158-np12-rf20 (query)                      2_813.42     1_052.75     3_866.16       0.3605          1.0923            1.1071        41.39
IVF-TQ-b4-nl158-np17-rf10 (query)                      2_813.42       726.14     3_539.56       0.2692          1.1311            1.1557        41.39
IVF-TQ-b4-nl158-np17-rf20 (query)                      2_813.42     1_102.38     3_915.79       0.3605          1.0923            1.1071        41.39
IVF-TQ-b4-nl158 (self)                                 2_813.42     2_126.51     4_939.93       0.3609          1.1024            1.1182        41.39
IVF-TQ-b4-nl223-np11-rf0 (query)                       2_025.97       432.91     2_458.88       0.0986          1.4231            1.4109        41.89
IVF-TQ-b4-nl223-np14-rf0 (query)                       2_025.97       456.83     2_482.80       0.0986          1.4231            1.4109        41.89
IVF-TQ-b4-nl223-np21-rf0 (query)                       2_025.97       513.48     2_539.45       0.0986          1.4231            1.4109        41.89
IVF-TQ-b4-nl223-np11-rf10 (query)                      2_025.97       666.21     2_692.18       0.2692          1.1311            1.1557        41.89
IVF-TQ-b4-nl223-np11-rf20 (query)                      2_025.97       985.91     3_011.88       0.3605          1.0923            1.1071        41.89
IVF-TQ-b4-nl223-np14-rf10 (query)                      2_025.97       694.06     2_720.03       0.2692          1.1311            1.1557        41.89
IVF-TQ-b4-nl223-np14-rf20 (query)                      2_025.97     1_026.19     3_052.15       0.3605          1.0923            1.1071        41.89
IVF-TQ-b4-nl223-np21-rf10 (query)                      2_025.97       755.20     2_781.17       0.2692          1.1311            1.1557        41.89
IVF-TQ-b4-nl223-np21-rf20 (query)                      2_025.97     1_110.56     3_136.53       0.3605          1.0923            1.1071        41.89
IVF-TQ-b4-nl223 (self)                                 2_025.97     2_195.47     4_221.44       0.3609          1.1024            1.1182        41.89
IVF-TQ-b4-nl316-np15-rf0 (query)                       2_760.64       452.42     3_213.06       0.0986          1.4231            1.4109        42.73
IVF-TQ-b4-nl316-np17-rf0 (query)                       2_760.64       476.37     3_237.01       0.0986          1.4231            1.4109        42.73
IVF-TQ-b4-nl316-np25-rf0 (query)                       2_760.64       526.91     3_287.55       0.0986          1.4231            1.4109        42.73
IVF-TQ-b4-nl316-np15-rf10 (query)                      2_760.64       682.53     3_443.17       0.2692          1.1311            1.1557        42.73
IVF-TQ-b4-nl316-np15-rf20 (query)                      2_760.64       989.23     3_749.87       0.3605          1.0923            1.1071        42.73
IVF-TQ-b4-nl316-np17-rf10 (query)                      2_760.64       694.95     3_455.59       0.2692          1.1311            1.1557        42.73
IVF-TQ-b4-nl316-np17-rf20 (query)                      2_760.64     1_011.88     3_772.52       0.3605          1.0923            1.1071        42.73
IVF-TQ-b4-nl316-np25-rf10 (query)                      2_760.64       759.38     3_520.01       0.2692          1.1311            1.1557        42.73
IVF-TQ-b4-nl316-np25-rf20 (query)                      2_760.64     1_090.44     3_851.08       0.3605          1.0923            1.1071        42.73
IVF-TQ-b4-nl316 (self)                                 2_760.64     2_248.07     5_008.71       0.3609          1.1024            1.1182        42.73
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
Exhaustive (query)                                        32.61       713.46       746.07       1.0000          1.0000            1.0000        48.83
Exhaustive (self)                                         32.61     2_475.98     2_508.59       1.0000          1.0000            1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              144.95       371.68       516.63       0.7918          1.0898            1.0632         7.12
ExhaustiveTQ-b2-rf5 (query)                              144.95       446.08       591.03       0.9995          1.0000            1.0000         7.12
ExhaustiveTQ-b2-rf10 (query)                             144.95       582.03       726.98       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b2-rf20 (query)                             144.95       951.04     1_095.98       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b2 (self)                                   144.95     3_140.24     3_285.18       1.0000          1.0000            1.0000         7.12
ExhaustiveTQ-b4-rf0 (query)                              231.22       582.59       813.81       0.8728          1.0322            1.0183        13.22
ExhaustiveTQ-b4-rf5 (query)                              231.22       667.06       898.28       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4-rf10 (query)                             231.22       797.11     1_028.33       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4-rf20 (query)                             231.22     1_222.59     1_453.81       1.0000          1.0000            1.0000        13.22
ExhaustiveTQ-b4 (self)                                   231.22     3_870.64     4_101.86       1.0000          1.0000            1.0000        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_051.02       127.10     1_178.12       0.7916          1.0897            1.0635         7.78
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_051.02       170.90     1_221.92       0.7918          1.0898            1.0632         7.78
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_051.02       209.19     1_260.21       0.7918          1.0898            1.0632         7.78
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_051.02       333.57     1_384.59       0.9982          1.0004            1.0000         7.78
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_051.02       606.93     1_657.95       0.9982          1.0004            1.0000         7.78
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_051.02       390.06     1_441.08       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_051.02       712.98     1_764.00       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_051.02       442.39     1_493.41       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_051.02       774.11     1_825.14       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl158 (self)                                 1_051.02     1_165.93     2_216.96       1.0000          1.0000            1.0000         7.78
IVF-TQ-b2-nl223-np11-rf0 (query)                         611.07       126.72       737.79       0.7919          1.0897            1.0632         7.93
IVF-TQ-b2-nl223-np14-rf0 (query)                         611.07       142.95       754.01       0.7918          1.0897            1.0632         7.93
IVF-TQ-b2-nl223-np21-rf0 (query)                         611.07       181.12       792.19       0.7918          1.0898            1.0632         7.93
IVF-TQ-b2-nl223-np11-rf10 (query)                        611.07       314.45       925.52       0.9995          1.0001            1.0000         7.93
IVF-TQ-b2-nl223-np11-rf20 (query)                        611.07       572.92     1_183.99       0.9995          1.0001            1.0000         7.93
IVF-TQ-b2-nl223-np14-rf10 (query)                        611.07       346.86       957.93       0.9999          1.0000            1.0000         7.93
IVF-TQ-b2-nl223-np14-rf20 (query)                        611.07       618.60     1_229.67       0.9999          1.0000            1.0000         7.93
IVF-TQ-b2-nl223-np21-rf10 (query)                        611.07       393.19     1_004.26       1.0000          1.0000            1.0000         7.93
IVF-TQ-b2-nl223-np21-rf20 (query)                        611.07       709.02     1_320.09       1.0000          1.0000            1.0000         7.93
IVF-TQ-b2-nl223 (self)                                   611.07     1_039.49     1_650.56       1.0000          1.0000            1.0000         7.93
IVF-TQ-b2-nl316-np15-rf0 (query)                         811.40       127.00       938.41       0.7918          1.0897            1.0632         8.12
IVF-TQ-b2-nl316-np17-rf0 (query)                         811.40       135.69       947.09       0.7918          1.0898            1.0632         8.12
IVF-TQ-b2-nl316-np25-rf0 (query)                         811.40       173.01       984.41       0.7918          1.0898            1.0632         8.12
IVF-TQ-b2-nl316-np15-rf10 (query)                        811.40       307.09     1_118.50       0.9997          1.0000            1.0000         8.12
IVF-TQ-b2-nl316-np15-rf20 (query)                        811.40       556.94     1_368.34       0.9997          1.0000            1.0000         8.12
IVF-TQ-b2-nl316-np17-rf10 (query)                        811.40       333.39     1_144.79       0.9999          1.0000            1.0000         8.12
IVF-TQ-b2-nl316-np17-rf20 (query)                        811.40       577.74     1_389.15       0.9999          1.0000            1.0000         8.12
IVF-TQ-b2-nl316-np25-rf10 (query)                        811.40       359.42     1_170.82       1.0000          1.0000            1.0000         8.12
IVF-TQ-b2-nl316-np25-rf20 (query)                        811.40       651.41     1_462.82       1.0000          1.0000            1.0000         8.12
IVF-TQ-b2-nl316 (self)                                   811.40     1_036.08     1_847.48       1.0000          1.0000            1.0000         8.12
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_110.22       181.41     1_291.63       0.8721          1.0325            1.0187        14.02
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_110.22       255.65     1_365.87       0.8728          1.0322            1.0183        14.02
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_110.22       318.10     1_428.32       0.8728          1.0322            1.0183        14.02
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_110.22       402.60     1_512.82       0.9982          1.0004            1.0000        14.02
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_110.22       670.63     1_780.85       0.9982          1.0004            1.0000        14.02
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_110.22       476.89     1_587.11       1.0000          1.0000            1.0000        14.02
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_110.22       813.79     1_924.01       1.0000          1.0000            1.0000        14.02
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_110.22       545.58     1_655.80       1.0000          1.0000            1.0000        14.02
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_110.22       882.54     1_992.76       1.0000          1.0000            1.0000        14.02
IVF-TQ-b4-nl158 (self)                                 1_110.22     1_232.01     2_342.23       1.0000          1.0000            1.0000        14.02
IVF-TQ-b4-nl223-np11-rf0 (query)                         697.00       178.66       875.66       0.8726          1.0323            1.0184        14.23
IVF-TQ-b4-nl223-np14-rf0 (query)                         697.00       204.23       901.23       0.8727          1.0322            1.0183        14.23
IVF-TQ-b4-nl223-np21-rf0 (query)                         697.00       264.42       961.42       0.8728          1.0322            1.0183        14.23
IVF-TQ-b4-nl223-np11-rf10 (query)                        697.00       376.58     1_073.58       0.9995          1.0001            1.0000        14.23
IVF-TQ-b4-nl223-np11-rf20 (query)                        697.00       642.39     1_339.40       0.9995          1.0001            1.0000        14.23
IVF-TQ-b4-nl223-np14-rf10 (query)                        697.00       406.43     1_103.43       0.9999          1.0000            1.0000        14.23
IVF-TQ-b4-nl223-np14-rf20 (query)                        697.00       686.55     1_383.55       0.9999          1.0000            1.0000        14.23
IVF-TQ-b4-nl223-np21-rf10 (query)                        697.00       480.21     1_177.21       1.0000          1.0000            1.0000        14.23
IVF-TQ-b4-nl223-np21-rf20 (query)                        697.00       798.84     1_495.85       1.0000          1.0000            1.0000        14.23
IVF-TQ-b4-nl223 (self)                                   697.00     1_077.56     1_774.56       1.0000          1.0000            1.0000        14.23
IVF-TQ-b4-nl316-np15-rf0 (query)                         894.91       179.29     1_074.20       0.8727          1.0322            1.0184        14.54
IVF-TQ-b4-nl316-np17-rf0 (query)                         894.91       191.26     1_086.17       0.8727          1.0322            1.0183        14.54
IVF-TQ-b4-nl316-np25-rf0 (query)                         894.91       243.18     1_138.08       0.8727          1.0322            1.0183        14.54
IVF-TQ-b4-nl316-np15-rf10 (query)                        894.91       365.04     1_259.95       0.9997          1.0000            1.0000        14.54
IVF-TQ-b4-nl316-np15-rf20 (query)                        894.91       617.96     1_512.87       0.9997          1.0000            1.0000        14.54
IVF-TQ-b4-nl316-np17-rf10 (query)                        894.91       372.15     1_267.06       0.9999          1.0000            1.0000        14.54
IVF-TQ-b4-nl316-np17-rf20 (query)                        894.91       643.66     1_538.57       0.9999          1.0000            1.0000        14.54
IVF-TQ-b4-nl316-np25-rf10 (query)                        894.91       438.35     1_333.26       1.0000          1.0000            1.0000        14.54
IVF-TQ-b4-nl316-np25-rf20 (query)                        894.91       731.31     1_626.22       1.0000          1.0000            1.0000        14.54
IVF-TQ-b4-nl316 (self)                                   894.91     1_033.96     1_928.86       1.0000          1.0000            1.0000        14.54
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
Exhaustive (query)                                        69.45     1_262.30     1_331.75       1.0000          1.0000            1.0000        97.66
Exhaustive (self)                                         69.45     4_261.29     4_330.74       1.0000          1.0000            1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              338.26       620.16       958.42       0.8424          1.0447            1.0331        13.97
ExhaustiveTQ-b2-rf5 (query)                              338.26       711.09     1_049.34       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2-rf10 (query)                             338.26       852.02     1_190.28       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2-rf20 (query)                             338.26     1_257.71     1_595.97       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b2 (self)                                   338.26     4_154.11     4_492.37       1.0000          1.0000            1.0000        13.97
ExhaustiveTQ-b4-rf0 (query)                              509.32     1_085.22     1_594.54       0.8985          1.0191            1.0110        26.18
ExhaustiveTQ-b4-rf5 (query)                              509.32     1_197.63     1_706.96       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4-rf10 (query)                             509.32     1_340.07     1_849.39       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4-rf20 (query)                             509.32     1_731.48     2_240.80       1.0000          1.0000            1.0000        26.18
ExhaustiveTQ-b4 (self)                                   509.32     5_701.84     6_211.16       1.0000          1.0000            1.0000        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        2_048.97       224.49     2_273.45       0.8420          1.0449            1.0333        14.96
IVF-TQ-b2-nl158-np12-rf0 (query)                       2_048.97       292.85     2_341.81       0.8424          1.0447            1.0331        14.96
IVF-TQ-b2-nl158-np17-rf0 (query)                       2_048.97       356.60     2_405.57       0.8424          1.0447            1.0331        14.96
IVF-TQ-b2-nl158-np7-rf10 (query)                       2_048.97       452.97     2_501.93       0.9986          1.0003            1.0000        14.96
IVF-TQ-b2-nl158-np7-rf20 (query)                       2_048.97       751.25     2_800.22       0.9986          1.0003            1.0000        14.96
IVF-TQ-b2-nl158-np12-rf10 (query)                      2_048.97       533.70     2_582.67       0.9999          1.0000            1.0000        14.96
IVF-TQ-b2-nl158-np12-rf20 (query)                      2_048.97       868.90     2_917.87       0.9999          1.0000            1.0000        14.96
IVF-TQ-b2-nl158-np17-rf10 (query)                      2_048.97       598.47     2_647.44       1.0000          1.0000            1.0000        14.96
IVF-TQ-b2-nl158-np17-rf20 (query)                      2_048.97       948.48     2_997.44       1.0000          1.0000            1.0000        14.96
IVF-TQ-b2-nl158 (self)                                 2_048.97     1_515.01     3_563.98       1.0000          1.0000            1.0000        14.96
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_097.78       229.03     1_326.81       0.8423          1.0447            1.0331        15.25
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_097.78       260.68     1_358.46       0.8424          1.0447            1.0331        15.25
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_097.78       316.73     1_414.51       0.8424          1.0447            1.0331        15.25
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_097.78       445.39     1_543.17       0.9997          1.0000            1.0000        15.25
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_097.78       726.01     1_823.79       0.9997          1.0000            1.0000        15.25
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_097.78       471.82     1_569.60       0.9999          1.0000            1.0000        15.25
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_097.78       774.60     1_872.38       0.9999          1.0000            1.0000        15.25
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_097.78       553.85     1_651.63       1.0000          1.0000            1.0000        15.25
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_097.78       869.60     1_967.38       1.0000          1.0000            1.0000        15.25
IVF-TQ-b2-nl223 (self)                                 1_097.78     1_447.36     2_545.14       1.0000          1.0000            1.0000        15.25
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_337.33       234.34     1_571.67       0.8424          1.0447            1.0331        15.56
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_337.33       245.72     1_583.04       0.8424          1.0447            1.0331        15.56
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_337.33       305.55     1_642.87       0.8424          1.0447            1.0331        15.56
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_337.33       435.52     1_772.84       0.9999          1.0000            1.0000        15.56
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_337.33       724.43     2_061.75       0.9999          1.0000            1.0000        15.56
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_337.33       454.14     1_791.46       1.0000          1.0000            1.0000        15.56
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_337.33       795.49     2_132.82       1.0000          1.0000            1.0000        15.56
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_337.33       511.16     1_848.49       1.0000          1.0000            1.0000        15.56
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_337.33       828.09     2_165.41       1.0000          1.0000            1.0000        15.56
IVF-TQ-b2-nl316 (self)                                 1_337.33     1_459.90     2_797.23       1.0000          1.0000            1.0000        15.56
IVF-TQ-b4-nl158-np7-rf0 (query)                        2_120.67       328.62     2_449.29       0.8977          1.0194            1.0113        27.46
IVF-TQ-b4-nl158-np12-rf0 (query)                       2_120.67       456.25     2_576.92       0.8985          1.0191            1.0110        27.46
IVF-TQ-b4-nl158-np17-rf0 (query)                       2_120.67       564.58     2_685.24       0.8985          1.0191            1.0110        27.46
IVF-TQ-b4-nl158-np7-rf10 (query)                       2_120.67       558.61     2_679.28       0.9986          1.0003            1.0000        27.46
IVF-TQ-b4-nl158-np7-rf20 (query)                       2_120.67       866.63     2_987.30       0.9986          1.0003            1.0000        27.46
IVF-TQ-b4-nl158-np12-rf10 (query)                      2_120.67       695.06     2_815.72       0.9999          1.0000            1.0000        27.46
IVF-TQ-b4-nl158-np12-rf20 (query)                      2_120.67     1_033.03     3_153.70       0.9999          1.0000            1.0000        27.46
IVF-TQ-b4-nl158-np17-rf10 (query)                      2_120.67       819.87     2_940.53       1.0000          1.0000            1.0000        27.46
IVF-TQ-b4-nl158-np17-rf20 (query)                      2_120.67     1_168.25     3_288.92       1.0000          1.0000            1.0000        27.46
IVF-TQ-b4-nl158 (self)                                 2_120.67     1_802.23     3_922.90       1.0000          1.0000            1.0000        27.46
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_215.89       335.45     1_551.34       0.8984          1.0191            1.0111        27.91
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_215.89       387.64     1_603.54       0.8985          1.0191            1.0110        27.91
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_215.89       506.89     1_722.78       0.8985          1.0191            1.0110        27.91
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_215.89       550.60     1_766.50       0.9997          1.0000            1.0000        27.91
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_215.89       831.15     2_047.04       0.9997          1.0000            1.0000        27.91
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_215.89       599.15     1_815.04       0.9999          1.0000            1.0000        27.91
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_215.89       899.04     2_114.94       0.9999          1.0000            1.0000        27.91
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_215.89       719.68     1_935.57       1.0000          1.0000            1.0000        27.91
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_215.89     1_039.42     2_255.31       1.0000          1.0000            1.0000        27.91
IVF-TQ-b4-nl223 (self)                                 1_215.89     1_669.21     2_885.10       1.0000          1.0000            1.0000        27.91
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_510.36       337.58     1_847.94       0.8985          1.0191            1.0110        28.36
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_510.36       360.65     1_871.01       0.8985          1.0191            1.0110        28.36
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_510.36       465.26     1_975.62       0.8985          1.0191            1.0110        28.36
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_510.36       543.50     2_053.86       0.9999          1.0000            1.0000        28.36
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_510.36       829.85     2_340.21       0.9999          1.0000            1.0000        28.36
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_510.36       566.65     2_077.01       1.0000          1.0000            1.0000        28.36
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_510.36       867.43     2_377.79       1.0000          1.0000            1.0000        28.36
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_510.36       673.16     2_183.51       1.0000          1.0000            1.0000        28.36
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_510.36       992.47     2_502.83       1.0000          1.0000            1.0000        28.36
IVF-TQ-b4-nl316 (self)                                 1_510.36     1_626.58     3_136.94       1.0000          1.0000            1.0000        28.36
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
Exhaustive (query)                                        99.95     1_906.24     2_006.19       1.0000          1.0000            1.0000       146.48
Exhaustive (self)                                         99.95     6_406.93     6_506.88       1.0000          1.0000            1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              615.15       962.53     1_577.69       0.8736          1.0271            1.0199        21.33
ExhaustiveTQ-b2-rf5 (query)                              615.15     1_045.61     1_660.77       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2-rf10 (query)                             615.15     1_190.86     1_806.02       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2-rf20 (query)                             615.15     1_590.75     2_205.90       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b2 (self)                                   615.15     5_238.59     5_853.74       1.0000          1.0000            1.0000        21.33
ExhaustiveTQ-b4-rf0 (query)                              755.08     1_754.51     2_509.58       0.9097          1.0146            1.0083        39.64
ExhaustiveTQ-b4-rf5 (query)                              755.08     1_874.55     2_629.63       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4-rf10 (query)                             755.08     1_991.30     2_746.38       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4-rf20 (query)                             755.08     2_371.10     3_126.17       1.0000          1.0000            1.0000        39.64
ExhaustiveTQ-b4 (self)                                   755.08     7_816.22     8_571.30       1.0000          1.0000            1.0000        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        3_053.20       348.13     3_401.33       0.8735          1.0272            1.0201        22.61
IVF-TQ-b2-nl158-np12-rf0 (query)                       3_053.20       442.24     3_495.44       0.8736          1.0271            1.0199        22.61
IVF-TQ-b2-nl158-np17-rf0 (query)                       3_053.20       521.14     3_574.33       0.8736          1.0271            1.0199        22.61
IVF-TQ-b2-nl158-np7-rf10 (query)                       3_053.20       594.83     3_648.03       0.9995          1.0001            1.0000        22.61
IVF-TQ-b2-nl158-np7-rf20 (query)                       3_053.20       910.66     3_963.86       0.9995          1.0001            1.0000        22.61
IVF-TQ-b2-nl158-np12-rf10 (query)                      3_053.20       706.69     3_759.89       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158-np12-rf20 (query)                      3_053.20     1_048.66     4_101.86       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158-np17-rf10 (query)                      3_053.20       794.49     3_847.69       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158-np17-rf20 (query)                      3_053.20     1_151.55     4_204.75       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl158 (self)                                 3_053.20     2_004.99     5_058.19       1.0000          1.0000            1.0000        22.61
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_679.79       343.51     2_023.30       0.8736          1.0271            1.0200        23.01
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_679.79       382.91     2_062.70       0.8736          1.0271            1.0199        23.01
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_679.79       467.52     2_147.31       0.8736          1.0271            1.0199        23.01
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_679.79       582.67     2_262.46       0.9998          1.0000            1.0000        23.01
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_679.79       888.74     2_568.53       0.9998          1.0000            1.0000        23.01
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_679.79       620.95     2_300.74       1.0000          1.0000            1.0000        23.01
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_679.79       954.46     2_634.25       1.0000          1.0000            1.0000        23.01
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_679.79       720.46     2_400.25       1.0000          1.0000            1.0000        23.01
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_679.79     1_061.69     2_741.48       1.0000          1.0000            1.0000        23.01
IVF-TQ-b2-nl223 (self)                                 1_679.79     1_920.98     3_600.77       1.0000          1.0000            1.0000        23.01
IVF-TQ-b2-nl316-np15-rf0 (query)                       2_009.83       353.25     2_363.08       0.8736          1.0271            1.0200        23.53
IVF-TQ-b2-nl316-np17-rf0 (query)                       2_009.83       373.03     2_382.85       0.8736          1.0271            1.0200        23.53
IVF-TQ-b2-nl316-np25-rf0 (query)                       2_009.83       443.79     2_453.61       0.8736          1.0271            1.0199        23.53
IVF-TQ-b2-nl316-np15-rf10 (query)                      2_009.83       579.97     2_589.80       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np15-rf20 (query)                      2_009.83       897.10     2_906.92       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np17-rf10 (query)                      2_009.83       598.40     2_608.22       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np17-rf20 (query)                      2_009.83       928.06     2_937.88       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np25-rf10 (query)                      2_009.83       681.77     2_691.59       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316-np25-rf20 (query)                      2_009.83     1_025.65     3_035.48       1.0000          1.0000            1.0000        23.53
IVF-TQ-b2-nl316 (self)                                 2_009.83     1_918.26     3_928.08       1.0000          1.0000            1.0000        23.53
IVF-TQ-b4-nl158-np7-rf0 (query)                        3_139.88       518.74     3_658.62       0.9095          1.0147            1.0084        41.37
IVF-TQ-b4-nl158-np12-rf0 (query)                       3_139.88       708.56     3_848.44       0.9097          1.0146            1.0083        41.37
IVF-TQ-b4-nl158-np17-rf0 (query)                       3_139.88       857.81     3_997.69       0.9097          1.0146            1.0083        41.37
IVF-TQ-b4-nl158-np7-rf10 (query)                       3_139.88       765.05     3_904.93       0.9995          1.0001            1.0000        41.37
IVF-TQ-b4-nl158-np7-rf20 (query)                       3_139.88     1_079.05     4_218.93       0.9995          1.0001            1.0000        41.37
IVF-TQ-b4-nl158-np12-rf10 (query)                      3_139.88       957.81     4_097.69       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158-np12-rf20 (query)                      3_139.88     1_308.79     4_448.67       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158-np17-rf10 (query)                      3_139.88     1_114.49     4_254.37       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158-np17-rf20 (query)                      3_139.88     1_472.13     4_612.02       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl158 (self)                                 3_139.88     2_562.87     5_702.75       1.0000          1.0000            1.0000        41.37
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_858.30       527.25     2_385.55       0.9096          1.0146            1.0084        41.97
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_858.30       599.15     2_457.45       0.9097          1.0146            1.0083        41.97
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_858.30       767.18     2_625.48       0.9097          1.0146            1.0083        41.97
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_858.30       755.74     2_614.03       0.9998          1.0000            1.0000        41.97
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_858.30     1_060.46     2_918.76       0.9998          1.0000            1.0000        41.97
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_858.30       831.98     2_690.28       1.0000          1.0000            1.0000        41.97
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_858.30     1_152.78     3_011.08       1.0000          1.0000            1.0000        41.97
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_858.30     1_002.11     2_860.41       1.0000          1.0000            1.0000        41.97
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_858.30     1_340.61     3_198.91       1.0000          1.0000            1.0000        41.97
IVF-TQ-b4-nl223 (self)                                 1_858.30     2_417.91     4_276.21       1.0000          1.0000            1.0000        41.97
IVF-TQ-b4-nl316-np15-rf0 (query)                       2_203.40       531.37     2_734.77       0.9097          1.0146            1.0083        42.73
IVF-TQ-b4-nl316-np17-rf0 (query)                       2_203.40       571.35     2_774.75       0.9097          1.0146            1.0083        42.73
IVF-TQ-b4-nl316-np25-rf0 (query)                       2_203.40       716.33     2_919.73       0.9097          1.0146            1.0083        42.73
IVF-TQ-b4-nl316-np15-rf10 (query)                      2_203.40       750.55     2_953.95       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np15-rf20 (query)                      2_203.40     1_061.91     3_265.31       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np17-rf10 (query)                      2_203.40       788.59     2_991.99       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np17-rf20 (query)                      2_203.40     1_105.81     3_309.21       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np25-rf10 (query)                      2_203.40       941.48     3_144.87       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316-np25-rf20 (query)                      2_203.40     1_274.30     3_477.70       1.0000          1.0000            1.0000        42.73
IVF-TQ-b4-nl316 (self)                                 2_203.40     2_369.78     4_573.18       1.0000          1.0000            1.0000        42.73
-----------------------------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
