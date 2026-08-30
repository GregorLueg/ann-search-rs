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
===================================================================================================================================
Benchmark: 50k samples, 256D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        33.23       673.29       706.52       1.0000          1.0000        48.83
Exhaustive (self)                                         33.23     2_199.85     2_233.07       1.0000          1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_697.30       288.03     2_985.33       0.0377          1.4542         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_697.30       380.37     3_077.68       0.1695          1.1187         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_697.30       478.70     3_176.00       0.2766          1.0758         1.78
ExhaustiveBinary-256-random (self)                     2_697.30     1_236.46     3_933.77       0.1763          1.1125         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_765.06       292.85     3_057.91       0.1872        292.8713         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_765.06       403.21     3_168.27       0.5340          1.0265         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_765.06       508.47     3_273.53       0.6681          1.0147         1.78
ExhaustiveBinary-256-pca (self)                        2_765.06     1_301.94     4_067.00       0.5319          1.0270         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_327.47       452.93     5_780.40       0.0697          1.3712         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_327.47       568.95     5_896.42       0.2154          1.0884         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_327.47       659.52     5_986.99       0.3320          1.0553         3.55
ExhaustiveBinary-512-random (self)                     5_327.47     1_814.60     7_142.07       0.2187          1.0859         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_378.85       455.29     5_834.14       0.2012          1.1802         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_378.85       574.14     5_952.99       0.6341          1.0175         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_378.85       679.62     6_058.46       0.7996          1.0073         3.55
ExhaustiveBinary-512-pca (self)                        5_378.85     1_851.62     7_230.46       0.6349          1.0176         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_417.34       788.78    11_206.12       0.0965          1.2966         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_417.34       893.77    11_311.11       0.2686          1.0681         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_417.34     1_006.15    11_423.49       0.4004          1.0419         7.10
ExhaustiveBinary-1024-random (self)                   10_417.34     2_941.39    13_358.74       0.2692          1.0684         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_482.73       794.18    11_276.91       0.2079          1.1699         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_482.73       913.67    11_396.40       0.6481          1.0163         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_482.73     1_026.22    11_508.95       0.8098          1.0067         7.10
ExhaustiveBinary-1024-pca (self)                      10_482.73     2_962.45    13_445.17       0.6480          1.0165         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   58.27       502.03       560.30       0.0290          1.4758         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    58.27       529.68       587.95       0.1603          1.1277         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    58.27       820.42       878.69       0.2715          1.0802         1.53
ExhaustiveBinary-256-sign (self)                          58.27     1_690.88     1_749.15       0.1661          1.1226         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)            3_707.39       127.08     3_834.47       0.0688          1.3009         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           3_707.39       129.75     3_837.13       0.0685          1.3033         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           3_707.39       136.04     3_843.43       0.0685          1.3033         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           3_707.39       178.74     3_886.12       0.2596          1.0746         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           3_707.39       237.56     3_944.94       0.3791          1.0478         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          3_707.39       185.40     3_892.79       0.2554          1.0764         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          3_707.39       231.56     3_938.94       0.3697          1.0498         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          3_707.39       189.63     3_897.02       0.2554          1.0764         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          3_707.39       247.61     3_955.00       0.3697          1.0498         1.93
IVF-Binary-256-nl158-random (self)                     3_707.39       514.81     4_222.20       0.2627          1.0706         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           3_142.10       144.57     3_286.67       0.0772          1.2830         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           3_142.10       130.68     3_272.78       0.0771          1.2836         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           3_142.10       136.86     3_278.96       0.0771          1.2842         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          3_142.10       184.10     3_326.20       0.2679          1.0725         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          3_142.10       234.04     3_376.14       0.3839          1.0475         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          3_142.10       184.01     3_326.11       0.2663          1.0731         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          3_142.10       240.51     3_382.60       0.3801          1.0483         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          3_142.10       194.78     3_336.87       0.2648          1.0737         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          3_142.10       246.56     3_388.65       0.3775          1.0488         2.00
IVF-Binary-256-nl223-random (self)                     3_142.10       514.50     3_656.59       0.2732          1.0676         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           3_315.69       135.88     3_451.56       0.0838          1.2701         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           3_315.69       134.36     3_450.05       0.0837          1.2707         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           3_315.69       141.80     3_457.49       0.0837          1.2713         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          3_315.69       187.81     3_503.50       0.2724          1.0714         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          3_315.69       234.79     3_550.47       0.3866          1.0471         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          3_315.69       186.77     3_502.45       0.2712          1.0719         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          3_315.69       239.47     3_555.15       0.3839          1.0477         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          3_315.69       193.24     3_508.92       0.2694          1.0726         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          3_315.69       245.74     3_561.43       0.3801          1.0485         2.09
IVF-Binary-256-nl316-random (self)                     3_315.69       536.08     3_851.77       0.2782          1.0664         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               3_776.59       124.95     3_901.54       0.1990          6.9107         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              3_776.59       130.93     3_907.52       0.1975         18.8218         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              3_776.59       137.11     3_913.70       0.1970         32.3603         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              3_776.59       186.03     3_962.62       0.6300          1.0178         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              3_776.59       242.71     4_019.30       0.7957          1.0074         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             3_776.59       219.55     3_996.14       0.6199          1.0186         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             3_776.59       243.32     4_019.91       0.7833          1.0080         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             3_776.59       199.04     3_975.63       0.6132          1.0191         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             3_776.59       252.37     4_028.95       0.7743          1.0084         1.93
IVF-Binary-256-nl158-pca (self)                        3_776.59       547.97     4_324.56       0.6193          1.0188         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              3_208.85       129.45     3_338.30       0.1990         11.9212         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              3_208.85       132.42     3_341.27       0.1980         17.5978         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              3_208.85       141.81     3_350.66       0.1972         27.7332         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             3_208.85       192.35     3_401.20       0.6279          1.0179         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             3_208.85       253.57     3_462.42       0.7934          1.0075         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             3_208.85       195.42     3_404.28       0.6222          1.0184         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             3_208.85       244.40     3_453.25       0.7864          1.0078         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             3_208.85       197.48     3_406.33       0.6153          1.0189         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             3_208.85       252.23     3_461.08       0.7771          1.0083         2.00
IVF-Binary-256-nl223-pca (self)                        3_208.85       547.30     3_756.15       0.6216          1.0186         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              3_387.02       140.30     3_527.32       0.1996          7.7382         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              3_387.02       141.74     3_528.76       0.1990         13.1219         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              3_387.02       142.06     3_529.08       0.1980         22.5604         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             3_387.02       193.10     3_580.12       0.6293          1.0178         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             3_387.02       244.82     3_631.84       0.7948          1.0074         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             3_387.02       193.61     3_580.63       0.6258          1.0181         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             3_387.02       245.15     3_632.17       0.7906          1.0076         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             3_387.02       202.97     3_589.99       0.6183          1.0187         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             3_387.02       253.55     3_640.57       0.7814          1.0081         2.09
IVF-Binary-256-nl316-pca (self)                        3_387.02       555.90     3_942.92       0.6252          1.0183         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_311.87       219.22     6_531.09       0.0877          1.2850         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_311.87       231.63     6_543.51       0.0869          1.2904         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_311.87       236.77     6_548.64       0.0869          1.2904         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_311.87       276.38     6_588.25       0.2612          1.0705         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_311.87       325.42     6_637.30       0.3836          1.0449         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_311.87       299.05     6_610.92       0.2555          1.0727         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_311.87       342.89     6_654.76       0.3735          1.0468         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_311.87       298.31     6_610.19       0.2555          1.0727         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_311.87       351.83     6_663.70       0.3735          1.0468         3.71
IVF-Binary-512-nl158-random (self)                     6_311.87       866.17     7_178.05       0.2578          1.0711         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_731.19       225.06     5_956.25       0.0908          1.2781         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_731.19       227.63     5_958.82       0.0905          1.2799         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_731.19       236.89     5_968.08       0.0903          1.2822         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_731.19       279.99     6_011.18       0.2648          1.0697         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_731.19       328.74     6_059.93       0.3863          1.0445         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_731.19       306.25     6_037.44       0.2618          1.0708         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_731.19       340.30     6_071.49       0.3811          1.0454         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_731.19       293.78     6_024.97       0.2596          1.0718         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_731.19       345.76     6_076.95       0.3774          1.0463         3.77
IVF-Binary-512-nl223-random (self)                     5_731.19       859.34     6_590.53       0.2638          1.0693         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           5_899.36       235.29     6_134.65       0.0929          1.2729         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           5_899.36       230.49     6_129.85       0.0927          1.2743         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           5_899.36       237.74     6_137.10       0.0924          1.2770         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          5_899.36       284.47     6_183.84       0.2639          1.0698         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          5_899.36       333.37     6_232.73       0.3842          1.0447         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          5_899.36       285.30     6_184.67       0.2621          1.0705         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          5_899.36       337.22     6_236.58       0.3811          1.0453         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          5_899.36       292.27     6_191.63       0.2595          1.0716         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          5_899.36       350.39     6_249.75       0.3765          1.0464         3.86
IVF-Binary-512-nl316-random (self)                     5_899.36       866.11     6_765.48       0.2648          1.0690         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               6_303.64       219.33     6_522.97       0.2033          1.1762         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              6_303.64       227.18     6_530.83       0.2023          1.1787         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              6_303.64       237.03     6_540.67       0.2023          1.1787         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              6_303.64       286.47     6_590.12       0.6394          1.0170         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              6_303.64       340.49     6_644.13       0.8051          1.0070         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             6_303.64       286.13     6_589.77       0.6355          1.0174         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             6_303.64       340.57     6_644.22       0.8006          1.0072         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             6_303.64       294.87     6_598.51       0.6355          1.0174         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             6_303.64       346.20     6_649.85       0.8006          1.0072         3.71
IVF-Binary-512-nl158-pca (self)                        6_303.64       890.24     7_193.88       0.6361          1.0175         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              5_755.50       224.56     5_980.06       0.2037          1.1754         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              5_755.50       228.25     5_983.75       0.2030          1.1772         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              5_755.50       235.44     5_990.94       0.2026          1.1784         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             5_755.50       286.21     6_041.71       0.6397          1.0170         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             5_755.50       332.86     6_088.36       0.8058          1.0069         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             5_755.50       291.27     6_046.77       0.6372          1.0172         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             5_755.50       338.37     6_093.87       0.8026          1.0071         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             5_755.50       293.07     6_048.57       0.6356          1.0174         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             5_755.50       345.16     6_100.66       0.8007          1.0072         3.77
IVF-Binary-512-nl223-pca (self)                        5_755.50       865.10     6_620.60       0.6376          1.0174         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              5_913.94       231.64     6_145.58       0.2039          1.1750         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              5_913.94       251.46     6_165.40       0.2035          1.1759         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              5_913.94       238.91     6_152.85       0.2029          1.1777         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             5_913.94       291.91     6_205.84       0.6398          1.0170         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             5_913.94       341.02     6_254.96       0.8053          1.0069         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             5_913.94       287.20     6_201.14       0.6384          1.0171         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             5_913.94       337.05     6_250.99       0.8036          1.0070         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             5_913.94       295.60     6_209.54       0.6363          1.0173         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             5_913.94       345.94     6_259.88       0.8010          1.0072         3.86
IVF-Binary-512-nl316-pca (self)                        5_913.94       897.90     6_811.84       0.6387          1.0172         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_415.87       416.05    11_831.92       0.1033          1.2620         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_415.87       436.29    11_852.16       0.1026          1.2673         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_415.87       442.85    11_858.72       0.1026          1.2673         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_415.87       476.66    11_892.53       0.2931          1.0615         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_415.87       533.77    11_949.64       0.4304          1.0376         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_415.87       494.29    11_910.16       0.2871          1.0632         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_415.87       544.27    11_960.14       0.4204          1.0391         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_415.87       512.38    11_928.25       0.2871          1.0632         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_415.87       565.38    11_981.25       0.4204          1.0391         7.26
IVF-Binary-1024-nl158-random (self)                   11_415.87     1_530.72    12_946.59       0.2877          1.0635         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_838.88       420.15    11_259.03       0.1042          1.2597         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_838.88       431.55    11_270.43       0.1039          1.2622         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_838.88       445.98    11_284.86       0.1036          1.2646         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_838.88       484.76    11_323.63       0.2943          1.0612         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_838.88       530.14    11_369.02       0.4316          1.0374         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_838.88       493.33    11_332.20       0.2913          1.0621         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_838.88       544.85    11_383.73       0.4263          1.0382         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_838.88       495.63    11_334.51       0.2889          1.0628         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_838.88       551.16    11_390.04       0.4224          1.0389         7.32
IVF-Binary-1024-nl223-random (self)                   10_838.88     1_539.86    12_378.74       0.2914          1.0624         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         11_015.37       422.69    11_438.06       0.1043          1.2579         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)         11_015.37       425.63    11_441.00       0.1041          1.2596         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)         11_015.37       438.59    11_453.96       0.1038          1.2628         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)        11_015.37       478.91    11_494.29       0.2932          1.0613         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)        11_015.37       530.39    11_545.77       0.4297          1.0376         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)        11_015.37       481.48    11_496.85       0.2913          1.0619         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)        11_015.37       534.46    11_549.84       0.4262          1.0381         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)        11_015.37       508.98    11_524.35       0.2887          1.0628         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)        11_015.37       550.11    11_565.48       0.4213          1.0390         7.42
IVF-Binary-1024-nl316-random (self)                   11_015.37     1_531.71    12_547.08       0.2918          1.0622         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)             11_429.44       425.00    11_854.44       0.2100          1.1663         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            11_429.44       430.33    11_859.77       0.2092          1.1683         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            11_429.44       441.36    11_870.80       0.2092          1.1683         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            11_429.44       470.30    11_899.74       0.6530          1.0159         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            11_429.44       529.03    11_958.47       0.8149          1.0065         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           11_429.44       491.97    11_921.41       0.6494          1.0163         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           11_429.44       539.76    11_969.20       0.8108          1.0067         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           11_429.44       531.03    11_960.47       0.6494          1.0163         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           11_429.44       572.46    12_001.90       0.8108          1.0067         7.26
IVF-Binary-1024-nl158-pca (self)                      11_429.44     1_600.93    13_030.37       0.6495          1.0164         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            10_889.41       417.87    11_307.29       0.2103          1.1657         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            10_889.41       424.60    11_314.02       0.2097          1.1672         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            10_889.41       446.88    11_336.29       0.2093          1.1681         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           10_889.41       487.20    11_376.62       0.6528          1.0159         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           10_889.41       537.07    11_426.49       0.8153          1.0064         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           10_889.41       493.20    11_382.62       0.6503          1.0161         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           10_889.41       541.67    11_431.08       0.8125          1.0066         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           10_889.41       504.30    11_393.71       0.6490          1.0163         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           10_889.41       576.33    11_465.74       0.8107          1.0067         7.32
IVF-Binary-1024-nl223-pca (self)                      10_889.41     1_535.79    12_425.21       0.6507          1.0163         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            11_071.52       425.92    11_497.44       0.2104          1.1655         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            11_071.52       435.12    11_506.64       0.2101          1.1662         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            11_071.52       438.93    11_510.45       0.2096          1.1676         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           11_071.52       489.03    11_560.55       0.6527          1.0159         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           11_071.52       574.24    11_645.76       0.8146          1.0065         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           11_071.52       491.49    11_563.01       0.6514          1.0161         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           11_071.52       537.78    11_609.30       0.8132          1.0066         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           11_071.52       511.95    11_583.47       0.6496          1.0162         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           11_071.52       549.81    11_621.33       0.8108          1.0067         7.42
IVF-Binary-1024-nl316-pca (self)                      11_071.52     1_554.64    12_626.16       0.6517          1.0162         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)              1_088.62       264.92     1_353.55       0.1890         37.9119         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)             1_088.62       302.42     1_391.04       0.1685        141.5978         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)             1_088.62       318.60     1_407.23       0.1648        276.7293         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)             1_088.62       330.69     1_419.31       0.5688          1.0245         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)             1_088.62       559.59     1_648.22       0.7516          1.0100         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)            1_088.62       342.87     1_431.50       0.4909          1.0332         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)            1_088.62       565.41     1_654.04       0.6587          1.0159         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)            1_088.62       353.95     1_442.57       0.4526          1.0386         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)            1_088.62       620.85     1_709.47       0.6050          1.0202         1.68
IVF-Binary-256-nl158-sign (self)                       1_088.62     1_094.27     2_182.89       0.4901          1.0336         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               522.75       317.04       839.78       0.1800         85.2386         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               522.75       331.35       854.10       0.1702        178.4554         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               522.75       354.15       876.89       0.1584        324.7099         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              522.75       343.08       865.83       0.5368          1.0275         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              522.75       582.72     1_105.46       0.7157          1.0121         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              522.75       354.39       877.14       0.5010          1.0315         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              522.75       566.45     1_089.20       0.6725          1.0149         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              522.75       357.19       879.94       0.4463          1.0390         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              522.75       619.49     1_142.24       0.6029          1.0202         1.75
IVF-Binary-256-nl223-sign (self)                         522.75     1_063.39     1_586.13       0.4994          1.0320         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               691.48       328.37     1_019.85       0.1750         69.4249         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               691.48       336.34     1_027.83       0.1686        130.7190         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               691.48       363.11     1_054.60       0.1560        233.2169         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              691.48       365.29     1_056.77       0.5255          1.0283         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              691.48       591.61     1_283.09       0.7077          1.0124         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              691.48       359.89     1_051.37       0.5033          1.0308         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              691.48       629.17     1_320.66       0.6813          1.0142         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              691.48       393.09     1_084.58       0.4431          1.0390         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              691.48       649.71     1_341.19       0.6057          1.0197         1.84
IVF-Binary-256-nl316-sign (self)                         691.48     1_077.04     1_768.53       0.5010          1.0314         1.84
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.68     1_200.91     1_268.59       1.0000          1.0000        97.66
Exhaustive (self)                                         67.68     3_920.84     3_988.53       1.0000          1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_775.31       430.99     6_206.30       0.0310          1.3169         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_775.31       556.20     6_331.51       0.1456          1.0932         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_775.31       658.15     6_433.46       0.2426          1.0614         2.03
ExhaustiveBinary-256-random (self)                     5_775.31     1_753.99     7_529.30       0.1495          1.0890         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_190.20       429.28     6_619.47       0.1393        316.8431         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_190.20       572.26     6_762.46       0.3908          1.0290         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_190.20       702.38     6_892.57       0.5139          1.0182         2.03
ExhaustiveBinary-256-pca (self)                        6_190.20     1_801.71     7_991.91       0.3906          1.0291         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_408.19       685.66    12_093.85       0.0645          1.2534         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_408.19       824.45    12_232.64       0.1842          1.0654         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_408.19       924.62    12_332.81       0.2849          1.0420         4.05
ExhaustiveBinary-512-random (self)                    11_408.19     2_642.71    14_050.91       0.1862          1.0632         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_560.96       687.22    12_248.18       0.1676       1026.8993         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_560.96       838.29    12_399.25       0.4489          1.2485         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_560.96       954.26    12_515.22       0.5690          1.0152         4.05
ExhaustiveBinary-512-pca (self)                       11_560.96     2_638.76    14_199.71       0.4499          1.3207         4.05
ExhaustiveBinary-1024-random_no_rr (query)            22_655.68     1_247.58    23_903.26       0.0840          1.2148         8.11
ExhaustiveBinary-1024-random-rf10 (query)             22_655.68     1_356.82    24_012.50       0.2135          1.0553         8.11
ExhaustiveBinary-1024-random-rf20 (query)             22_655.68     1_490.95    24_146.63       0.3233          1.0357         8.11
ExhaustiveBinary-1024-random (self)                   22_655.68     4_541.48    27_197.16       0.2127          1.0553         8.11
ExhaustiveBinary-1024-pca_no_rr (query)               22_867.80     1_232.11    24_099.90       0.2036          1.1084         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                22_867.80     1_368.56    24_236.36       0.6275          1.0116         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                22_867.80     1_513.24    24_381.04       0.7906          1.0049         8.11
ExhaustiveBinary-1024-pca (self)                      22_867.80     4_518.19    27_385.99       0.6278          1.0116         8.11
ExhaustiveBinary-512-sign_no_rr (query)                  125.05       677.98       803.03       0.0497          1.2798         3.05
ExhaustiveBinary-512-sign-rf10 (query)                   125.05       726.69       851.74       0.1718          1.0748         3.05
ExhaustiveBinary-512-sign-rf20 (query)                   125.05     1_157.65     1_282.69       0.2765          1.0469         3.05
ExhaustiveBinary-512-sign (self)                         125.05     2_323.16     2_448.21       0.1739          1.0727         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)            7_682.96       249.43     7_932.38       0.0629          1.2081         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           7_682.96       257.52     7_940.48       0.0628          1.2089         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           7_682.96       263.05     7_946.01       0.0628          1.2089         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           7_682.96       328.00     8_010.95       0.2391          1.0535         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           7_682.96       410.57     8_093.53       0.3513          1.0345         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          7_682.96       334.20     8_017.16       0.2369          1.0541         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          7_682.96       414.96     8_097.92       0.3473          1.0350         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          7_682.96       344.87     8_027.83       0.2369          1.0541         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          7_682.96       420.20     8_103.16       0.3473          1.0350         2.34
IVF-Binary-256-nl158-random (self)                     7_682.96       943.57     8_626.53       0.2402          1.0503         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           6_646.12       260.82     6_906.93       0.0683          1.1998         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)           6_646.12       261.08     6_907.19       0.0682          1.2003         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)           6_646.12       277.66     6_923.77       0.0682          1.2006         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)          6_646.12       335.79     6_981.90       0.2454          1.0513         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)          6_646.12       411.26     7_057.38       0.3545          1.0341         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)          6_646.12       335.35     6_981.46       0.2439          1.0518         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)          6_646.12       418.89     7_065.00       0.3516          1.0345         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)          6_646.12       341.91     6_988.03       0.2433          1.0520         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)          6_646.12       423.79     7_069.90       0.3503          1.0347         2.47
IVF-Binary-256-nl223-random (self)                     6_646.12       970.55     7_616.66       0.2483          1.0477         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)           7_045.64       275.50     7_321.14       0.0749          1.1896         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           7_045.64       266.72     7_312.36       0.0748          1.1899         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           7_045.64       272.75     7_318.39       0.0747          1.1904         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          7_045.64       345.84     7_391.47       0.2507          1.0503         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          7_045.64       419.78     7_465.42       0.3597          1.0334         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          7_045.64       346.68     7_392.32       0.2496          1.0506         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          7_045.64       418.94     7_464.57       0.3579          1.0337         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          7_045.64       351.47     7_397.10       0.2484          1.0509         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          7_045.64       426.88     7_472.52       0.3551          1.0341         2.65
IVF-Binary-256-nl316-random (self)                     7_045.64       980.81     8_026.44       0.2538          1.0465         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               7_879.10       252.34     8_131.43       0.1474          3.4936         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              7_879.10       262.92     8_142.02       0.1463         10.1771         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              7_879.10       267.99     8_147.09       0.1461         20.5126         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              7_879.10       345.55     8_224.64       0.4622          1.0222         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              7_879.10       424.55     8_303.64       0.6302          1.0116         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             7_879.10       346.43     8_225.53       0.4545          1.0229         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             7_879.10       438.27     8_317.36       0.6170          1.0123         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             7_879.10       359.58     8_238.67       0.4509          1.0232         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             7_879.10       440.94     8_320.04       0.6104          1.0126         2.34
IVF-Binary-256-nl158-pca (self)                        7_879.10     1_013.20     8_892.29       0.4545          1.0230         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              6_863.02       260.40     7_123.42       0.1473          5.0915         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              6_863.02       267.66     7_130.68       0.1466          8.1540         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              6_863.02       270.92     7_133.95       0.1461         14.7048         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             6_863.02       351.50     7_214.52       0.4621          1.0222         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             6_863.02       429.32     7_292.34       0.6293          1.0117         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             6_863.02       350.54     7_213.56       0.4572          1.0226         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             6_863.02       436.10     7_299.12       0.6214          1.0121         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             6_863.02       358.95     7_221.97       0.4527          1.0230         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             6_863.02       448.33     7_311.35       0.6134          1.0125         2.47
IVF-Binary-256-nl223-pca (self)                        6_863.02     1_033.41     7_896.44       0.4570          1.0227         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              7_219.68       276.94     7_496.62       0.1473          4.6163         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              7_219.68       267.84     7_487.52       0.1470          6.2857         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              7_219.68       277.52     7_497.20       0.1463         11.9665         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             7_219.68       361.78     7_581.46       0.4627          1.0221         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             7_219.68       442.12     7_661.80       0.6297          1.0116         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             7_219.68       355.86     7_575.54       0.4600          1.0224         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             7_219.68       443.83     7_663.51       0.6256          1.0119         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             7_219.68       365.65     7_585.33       0.4540          1.0229         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             7_219.68       458.79     7_678.47       0.6160          1.0123         2.65
IVF-Binary-256-nl316-pca (self)                        7_219.68     1_106.24     8_325.92       0.4596          1.0225         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           13_302.86       452.20    13_755.07       0.0821          1.1946         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          13_302.86       456.55    13_759.42       0.0817          1.1976         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          13_302.86       468.51    13_771.37       0.0817          1.1976         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          13_302.86       534.53    13_837.39       0.2220          1.0530         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          13_302.86       606.68    13_909.54       0.3323          1.0345         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         13_302.86       541.46    13_844.33       0.2176          1.0544         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         13_302.86       628.73    13_931.59       0.3240          1.0358         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         13_302.86       550.48    13_853.34       0.2176          1.0544         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         13_302.86       627.73    13_930.59       0.3240          1.0358         4.36
IVF-Binary-512-nl158-random (self)                    13_302.86     1_638.59    14_941.46       0.2191          1.0530         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          12_307.26       458.95    12_766.22       0.0837          1.1919         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          12_307.26       460.56    12_767.82       0.0833          1.1939         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          12_307.26       473.69    12_780.95       0.0832          1.1945         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         12_307.26       532.04    12_839.30       0.2219          1.0530         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         12_307.26       613.01    12_920.27       0.3322          1.0345         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         12_307.26       538.73    12_846.00       0.2186          1.0540         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         12_307.26       612.02    12_919.28       0.3260          1.0355         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         12_307.26       548.37    12_855.63       0.2174          1.0544         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         12_307.26       627.61    12_934.87       0.3237          1.0358         4.49
IVF-Binary-512-nl223-random (self)                    12_307.26     1_646.42    13_953.69       0.2207          1.0526         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          12_709.52       498.14    13_207.67       0.0852          1.1891         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          12_709.52       472.41    13_181.94       0.0850          1.1902         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          12_709.52       482.46    13_191.98       0.0848          1.1917         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         12_709.52       547.33    13_256.85       0.2239          1.0525         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         12_709.52       627.38    13_336.91       0.3332          1.0343         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         12_709.52       561.91    13_271.44       0.2220          1.0531         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         12_709.52       627.42    13_336.95       0.3297          1.0348         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         12_709.52       553.56    13_263.08       0.2195          1.0539         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         12_709.52       666.80    13_376.33       0.3248          1.0356         4.67
IVF-Binary-512-nl316-random (self)                    12_709.52     1_670.89    14_380.42       0.2240          1.0518         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              13_464.51       452.78    13_917.29       0.2022         18.4445         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             13_464.51       461.67    13_926.18       0.1994         61.1144         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             13_464.51       483.77    13_948.28       0.1973        117.3131         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             13_464.51       536.15    14_000.67       0.6215          1.0118         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             13_464.51       620.54    14_085.06       0.7850          1.0051         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            13_464.51       547.71    14_012.23       0.6053          1.0127         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            13_464.51       629.69    14_094.21       0.7644          1.0058         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            13_464.51       552.61    14_017.12       0.5922          1.0134         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            13_464.51       644.33    14_108.84       0.7488          1.0063         4.36
IVF-Binary-512-nl158-pca (self)                       13_464.51     1_672.70    15_137.22       0.6058          1.0127         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             12_439.77       468.21    12_907.97       0.2023         31.0785         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             12_439.77       464.88    12_904.65       0.2007         48.7339         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             12_439.77       476.13    12_915.90       0.1985         89.2924         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            12_439.77       543.96    12_983.73       0.6188          1.0119         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            12_439.77       622.36    13_062.13       0.7818          1.0052         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            12_439.77       555.12    12_994.89       0.6110          1.0124         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            12_439.77       634.35    13_074.12       0.7728          1.0055         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            12_439.77       553.96    12_993.73       0.5993          1.0130         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            12_439.77       637.69    13_077.46       0.7574          1.0060         4.49
IVF-Binary-512-nl223-pca (self)                       12_439.77     1_669.10    14_108.87       0.6118          1.0124         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             12_849.78       469.35    13_319.12       0.2023         27.8099         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             12_849.78       472.24    13_322.02       0.2015         38.0402         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             12_849.78       483.30    13_333.07       0.1992         72.3334         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            12_849.78       551.00    13_400.77       0.6198          1.0119         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            12_849.78       635.17    13_484.95       0.7832          1.0051         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            12_849.78       550.35    13_400.13       0.6156          1.0121         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            12_849.78       647.65    13_497.42       0.7781          1.0053         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            12_849.78       558.50    13_408.27       0.6036          1.0127         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            12_849.78       647.14    13_496.92       0.7633          1.0058         4.67
IVF-Binary-512-nl316-pca (self)                       12_849.78     1_686.36    14_536.14       0.6163          1.0121         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          24_549.13       860.52    25_409.65       0.0906          1.1854         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)         24_549.13       881.88    25_431.01       0.0901          1.1896         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)         24_549.13       886.25    25_435.38       0.0901          1.1896         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)         24_549.13       942.25    25_491.38       0.2371          1.0494         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)         24_549.13     1_014.21    25_563.34       0.3545          1.0316         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)        24_549.13       955.19    25_504.32       0.2324          1.0507         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)        24_549.13     1_026.65    25_575.78       0.3453          1.0329         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)        24_549.13       963.22    25_512.35       0.2324          1.0507         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)        24_549.13     1_038.07    25_587.19       0.3453          1.0329         8.42
IVF-Binary-1024-nl158-random (self)                   24_549.13     3_021.46    27_570.59       0.2312          1.0508         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)         23_538.26       867.84    24_406.10       0.0909          1.1852         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         23_538.26       878.94    24_417.20       0.0904          1.1878         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         23_538.26       890.81    24_429.07       0.0903          1.1888         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        23_538.26       941.18    24_479.44       0.2361          1.0495         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        23_538.26     1_017.88    24_556.14       0.3537          1.0318         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        23_538.26       993.22    24_531.48       0.2325          1.0505         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        23_538.26     1_023.44    24_561.70       0.3472          1.0327         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        23_538.26       958.92    24_497.18       0.2312          1.0508         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        23_538.26     1_042.77    24_581.03       0.3449          1.0330         8.54
IVF-Binary-1024-nl223-random (self)                   23_538.26     3_011.44    26_549.70       0.2324          1.0504         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         23_933.12       876.32    24_809.45       0.0912          1.1840         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)         23_933.12       877.69    24_810.81       0.0910          1.1854         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)         23_933.12       942.05    24_875.17       0.0907          1.1874         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)        23_933.12       958.86    24_891.98       0.2374          1.0492         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)        23_933.12     1_042.40    24_975.52       0.3547          1.0316         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)        23_933.12       958.27    24_891.39       0.2355          1.0497         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)        23_933.12     1_036.33    24_969.45       0.3509          1.0321         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)        23_933.12       972.61    24_905.73       0.2328          1.0504         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)        23_933.12     1_045.05    24_978.17       0.3461          1.0328         8.73
IVF-Binary-1024-nl316-random (self)                   23_933.12     3_092.79    27_025.92       0.2351          1.0497         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)             24_816.41       872.45    25_688.86       0.2052          1.1063         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            24_816.41       876.38    25_692.79       0.2041          1.1079         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            24_816.41       890.39    25_706.80       0.2041          1.1079         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            24_816.41       940.57    25_756.98       0.6321          1.0113         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            24_816.41     1_012.87    25_829.28       0.7961          1.0047         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           24_816.41       954.18    25_770.59       0.6282          1.0115         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           24_816.41     1_031.28    25_847.69       0.7912          1.0049         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           24_816.41       971.85    25_788.26       0.6282          1.0115         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           24_816.41     1_046.04    25_862.45       0.7912          1.0049         8.42
IVF-Binary-1024-nl158-pca (self)                      24_816.41     3_024.29    27_840.70       0.6286          1.0116         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            23_778.41       866.03    24_644.44       0.2055          1.1057         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            23_778.41       871.92    24_650.33       0.2046          1.1069         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            23_778.41       917.28    24_695.69       0.2042          1.1077         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           23_778.41       947.52    24_725.93       0.6330          1.0113         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           23_778.41     1_024.36    24_802.77       0.7970          1.0047         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           23_778.41       951.79    24_730.20       0.6297          1.0114         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           23_778.41     1_032.81    24_811.22       0.7933          1.0048         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           23_778.41       967.12    24_745.53       0.6280          1.0115         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           23_778.41     1_052.58    24_830.99       0.7912          1.0049         8.54
IVF-Binary-1024-nl223-pca (self)                      23_778.41     3_037.25    26_815.66       0.6306          1.0115         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            24_129.81       887.22    25_017.02       0.2058          1.1056         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            24_129.81       881.28    25_011.09       0.2052          1.1062         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            24_129.81       893.15    25_022.95       0.2044          1.1075         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           24_129.81     1_031.19    25_160.99       0.6334          1.0112         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           24_129.81     1_031.54    25_161.35       0.7970          1.0047         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           24_129.81       962.75    25_092.56       0.6317          1.0113         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           24_129.81     1_044.39    25_174.20       0.7951          1.0048         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           24_129.81       974.33    25_104.14       0.6286          1.0115         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           24_129.81     1_051.37    25_181.18       0.7915          1.0049         8.73
IVF-Binary-1024-nl316-pca (self)                      24_129.81     3_037.86    27_167.67       0.6323          1.0113         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)              2_031.55       429.58     2_461.14       0.1907         83.4217         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)             2_031.55       490.48     2_522.03       0.1671        328.5012         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)             2_031.55       498.39     2_529.94       0.1612        531.1522         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)             2_031.55       478.44     2_509.99       0.5638          1.0159         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)             2_031.55       810.64     2_842.19       0.7459          1.0067         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)            2_031.55       492.90     2_524.45       0.4735          1.0226         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)            2_031.55       845.46     2_877.01       0.6384          1.0113         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)            2_031.55       536.53     2_568.08       0.4291          1.0271         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)            2_031.55       933.11     2_964.66       0.5779          1.0146         3.36
IVF-Binary-512-nl158-sign (self)                       2_031.55     1_513.21     3_544.76       0.4724          1.0228         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)             1_041.14       442.28     1_483.43       0.1847        127.4580         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)             1_041.14       462.99     1_504.13       0.1728        210.2308         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)             1_041.14       496.49     1_537.64       0.1608        400.4256         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)            1_041.14       504.45     1_545.59       0.5402          1.0176         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)            1_041.14       853.41     1_894.55       0.7190          1.0078         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)            1_041.14       521.14     1_562.29       0.4988          1.0208         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)            1_041.14       866.12     1_907.26       0.6696          1.0099         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)            1_041.14       573.09     1_614.23       0.4444          1.0722         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)            1_041.14       917.43     1_958.57       0.5983          1.0135         3.49
IVF-Binary-512-nl223-sign (self)                       1_041.14     1_550.01     2_591.16       0.4981          1.0209         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)             1_392.67       480.84     1_873.51       0.1802        147.2456         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)             1_392.67       500.65     1_893.32       0.1730        195.4281         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)             1_392.67       523.89     1_916.56       0.1570        419.0854         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)            1_392.67       533.42     1_926.10       0.5316          1.0180         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)            1_392.67       882.68     2_275.35       0.7114          1.0080         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)            1_392.67       533.16     1_925.83       0.5090          1.0197         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)            1_392.67       901.97     2_294.64       0.6853          1.0092         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)            1_392.67       571.43     1_964.10       0.4498          1.0556         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)            1_392.67       920.73     2_313.40       0.6104          1.0128         3.67
IVF-Binary-512-nl316-sign (self)                       1_392.67     1_600.36     2_993.03       0.5076          1.0199         3.67
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       102.22     1_761.35     1_863.57       1.0000          1.0000       146.48
Exhaustive (self)                                        102.22     5_847.76     5_949.98       1.0000          1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)              9_018.16       533.99     9_552.14       0.0332          1.2528         2.28
ExhaustiveBinary-256-random-rf10 (query)               9_018.16       665.89     9_684.05       0.1444          1.0757         2.28
ExhaustiveBinary-256-random-rf20 (query)               9_018.16       809.61     9_827.77       0.2377          1.0487         2.28
ExhaustiveBinary-256-random (self)                     9_018.16     2_148.12    11_166.28       0.1477          1.0720         2.28
ExhaustiveBinary-256-pca_no_rr (query)                 9_324.03       539.81     9_863.83       0.1248        269.6020         2.28
ExhaustiveBinary-256-pca-rf10 (query)                  9_324.03       689.98    10_014.00       0.3427          1.0270         2.28
ExhaustiveBinary-256-pca-rf20 (query)                  9_324.03       849.28    10_173.31       0.4576          1.0176         2.28
ExhaustiveBinary-256-pca (self)                        9_324.03     2_196.74    11_520.77       0.3430          1.0271         2.28
ExhaustiveBinary-512-random_no_rr (query)             17_642.99       900.35    18_543.33       0.0615          1.2068         4.55
ExhaustiveBinary-512-random-rf10 (query)              17_642.99     1_032.64    18_675.62       0.1713          1.0555         4.55
ExhaustiveBinary-512-random-rf20 (query)              17_642.99     1_182.73    18_825.71       0.2651          1.0361         4.55
ExhaustiveBinary-512-random (self)                    17_642.99     3_351.71    20_994.70       0.1731          1.0536         4.55
ExhaustiveBinary-512-pca_no_rr (query)                17_943.30       906.10    18_849.40       0.1449        971.3894         4.55
ExhaustiveBinary-512-pca-rf10 (query)                 17_943.30     1_071.66    19_014.96       0.3789          1.0668         4.55
ExhaustiveBinary-512-pca-rf20 (query)                 17_943.30     1_246.08    19_189.39       0.4883          1.0162         4.55
ExhaustiveBinary-512-pca (self)                       17_943.30     3_784.61    21_727.91       0.3798          1.0958         4.55
ExhaustiveBinary-1024-random_no_rr (query)            35_209.48     1_661.82    36_871.30       0.0803          1.1744         9.11
ExhaustiveBinary-1024-random-rf10 (query)             35_209.48     1_810.36    37_019.84       0.1929          1.0470         9.11
ExhaustiveBinary-1024-random-rf20 (query)             35_209.48     1_971.60    37_181.08       0.2950          1.0307         9.11
ExhaustiveBinary-1024-random (self)                   35_209.48     5_946.40    41_155.88       0.1939          1.0467         9.11
ExhaustiveBinary-1024-pca_no_rr (query)               35_384.50     1_675.02    37_059.51       0.2012          1.0858         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                35_384.50     1_930.33    37_314.82       0.6292          1.0091         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                35_384.50     1_983.05    37_367.55       0.7916          1.0038         9.11
ExhaustiveBinary-1024-pca (self)                      35_384.50     5_991.87    41_376.36       0.6280          1.0092         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  221.61       886.51     1_108.13       0.0656          1.1999         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   221.61       965.25     1_186.87       0.1790          1.0519         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   221.61     1_557.07     1_778.68       0.2788          1.0330         4.58
ExhaustiveBinary-768-sign (self)                         221.61     3_129.86     3_351.47       0.1794          1.0511         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)           11_943.40       372.91    12_316.31       0.0631          1.1715         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)          11_943.40       374.32    12_317.73       0.0629          1.1726         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)          11_943.40       385.86    12_329.27       0.0629          1.1726         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)          11_943.40       460.77    12_404.17       0.2169          1.0478         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)          11_943.40       552.81    12_496.21       0.3251          1.0311         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)         11_943.40       462.32    12_405.72       0.2130          1.0488         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)         11_943.40       559.21    12_502.61       0.3156          1.0322         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)         11_943.40       469.54    12_412.94       0.2130          1.0488         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)         11_943.40       566.55    12_509.95       0.3156          1.0322         2.74
IVF-Binary-256-nl158-random (self)                    11_943.40     1_357.83    13_301.23       0.2159          1.0455         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)          10_139.55       373.13    10_512.68       0.0691          1.1614         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)          10_139.55       381.21    10_520.76       0.0689          1.1622         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)          10_139.55       384.52    10_524.07       0.0688          1.1627         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)         10_139.55       470.54    10_610.08       0.2307          1.0438         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)         10_139.55       564.59    10_704.14       0.3380          1.0285         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)         10_139.55       472.30    10_611.85       0.2281          1.0445         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)         10_139.55       569.22    10_708.77       0.3326          1.0293         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)         10_139.55       479.30    10_618.85       0.2272          1.0448         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)         10_139.55       580.51    10_720.06       0.3306          1.0296         2.93
IVF-Binary-256-nl223-random (self)                    10_139.55     1_385.32    11_524.87       0.2307          1.0413         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)          10_803.75       389.06    11_192.81       0.0747          1.1538         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)          10_803.75       388.40    11_192.15       0.0747          1.1542         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)          10_803.75       392.34    11_196.09       0.0745          1.1551         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)         10_803.75       494.27    11_298.02       0.2362          1.0424         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)         10_803.75       581.78    11_385.53       0.3407          1.0283         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)         10_803.75       491.91    11_295.66       0.2348          1.0427         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)         10_803.75       578.99    11_382.74       0.3379          1.0286         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)         10_803.75       488.44    11_292.19       0.2331          1.0432         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)         10_803.75       586.20    11_389.95       0.3345          1.0291         3.21
IVF-Binary-256-nl316-random (self)                    10_803.75     1_427.53    12_231.28       0.2368          1.0395         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)              12_236.68       366.20    12_602.87       0.1305          4.1075         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)             12_236.68       372.44    12_609.11       0.1296         13.3923         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)             12_236.68       385.85    12_622.53       0.1295         19.6555         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)             12_236.68       476.42    12_713.10       0.4030          1.0216         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)             12_236.68       579.62    12_816.29       0.5645          1.0120         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)            12_236.68       483.85    12_720.53       0.3951          1.0223         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)            12_236.68       585.73    12_822.41       0.5508          1.0127         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)            12_236.68       490.43    12_727.10       0.3928          1.0225         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)            12_236.68       609.41    12_846.09       0.5459          1.0129         2.74
IVF-Binary-256-nl158-pca (self)                       12_236.68     1_448.09    13_684.77       0.3956          1.0224         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)             10_633.22       375.31    11_008.54       0.1303         10.0806         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)             10_633.22       380.73    11_013.95       0.1296         13.7631         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)             10_633.22       386.34    11_019.56       0.1294         21.2464         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)            10_633.22       490.04    11_123.26       0.4010          1.0217         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)            10_633.22       582.95    11_216.18       0.5602          1.0121         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)            10_633.22       489.67    11_122.90       0.3960          1.0222         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)            10_633.22       594.03    11_227.25       0.5524          1.0125         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)            10_633.22       498.97    11_132.19       0.3924          1.0225         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)            10_633.22       599.37    11_232.59       0.5456          1.0129         2.93
IVF-Binary-256-nl223-pca (self)                       10_633.22     1_445.48    12_078.70       0.3963          1.0222         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)             11_072.18       386.00    11_458.18       0.1302          9.6362         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)             11_072.18       408.06    11_480.24       0.1299         11.1167         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)             11_072.18       397.62    11_469.79       0.1295         18.0129         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)            11_072.18       497.97    11_570.14       0.4005          1.0218         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)            11_072.18       603.70    11_675.88       0.5596          1.0122         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)            11_072.18       495.57    11_567.75       0.3980          1.0220         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)            11_072.18       600.99    11_673.17       0.5557          1.0124         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)            11_072.18       502.27    11_574.44       0.3936          1.0224         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)            11_072.18       645.13    11_717.31       0.5478          1.0128         3.21
IVF-Binary-256-nl316-pca (self)                       11_072.18     1_483.91    12_556.09       0.3982          1.0221         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)           20_471.43       678.48    21_149.91       0.0781          1.1601         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)          20_471.43       724.93    21_196.36       0.0776          1.1627         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)          20_471.43       723.52    21_194.95       0.0776          1.1627         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)          20_471.43       817.75    21_289.18       0.2085          1.0447         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)          20_471.43       867.12    21_338.55       0.3146          1.0290         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)         20_471.43       810.17    21_281.60       0.2030          1.0459         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)         20_471.43       866.18    21_337.62       0.3038          1.0301         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)         20_471.43       798.13    21_269.57       0.2030          1.0459         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)         20_471.43       878.18    21_349.61       0.3038          1.0301         5.02
IVF-Binary-512-nl158-random (self)                    20_471.43     2_407.62    22_879.05       0.2041          1.0446         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)          18_867.54       685.23    19_552.77       0.0817          1.1530         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)          18_867.54       710.21    19_577.75       0.0813          1.1549         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)          18_867.54       701.79    19_569.32       0.0813          1.1559         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)         18_867.54       777.48    19_645.02       0.2120          1.0435         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)         18_867.54       866.44    19_733.98       0.3161          1.0286         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)         18_867.54       780.56    19_648.10       0.2088          1.0444         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)         18_867.54       872.36    19_739.89       0.3096          1.0295         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)         18_867.54       789.20    19_656.74       0.2076          1.0447         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)         18_867.54       885.48    19_753.01       0.3076          1.0298         5.21
IVF-Binary-512-nl223-random (self)                    18_867.54     2_436.94    21_304.48       0.2098          1.0431         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)          19_396.91       743.65    20_140.56       0.0833          1.1507         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)          19_396.91       718.54    20_115.44       0.0831          1.1517         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)          19_396.91       715.38    20_112.28       0.0829          1.1534         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)         19_396.91       789.74    20_186.64       0.2133          1.0432         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)         19_396.91       882.16    20_279.06       0.3163          1.0286         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)         19_396.91       792.78    20_189.69       0.2112          1.0437         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)         19_396.91       888.87    20_285.78       0.3129          1.0290         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)         19_396.91       798.79    20_195.69       0.2091          1.0444         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)         19_396.91       904.12    20_301.03       0.3090          1.0296         5.48
IVF-Binary-512-nl316-random (self)                    19_396.91     2_461.72    21_858.63       0.2126          1.0426         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)              20_730.14       712.93    21_443.06       0.1700         16.3134         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)             20_730.14       680.89    21_411.03       0.1678         59.6375         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)             20_730.14       689.26    21_419.39       0.1669         95.9194         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)             20_730.14       772.85    21_502.99       0.5300          1.0137         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)             20_730.14       880.47    21_610.60       0.6991          1.0067         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)            20_730.14       779.97    21_510.10       0.5155          1.0144         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)            20_730.14       881.19    21_611.33       0.6789          1.0073         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)            20_730.14       789.46    21_519.59       0.5065          1.0149         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)            20_730.14       887.71    21_617.84       0.6657          1.0078         5.02
IVF-Binary-512-nl158-pca (self)                       20_730.14     2_428.45    23_158.59       0.5164          1.0145         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)             19_136.79       690.70    19_827.49       0.1693         40.6613         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)             19_136.79       692.57    19_829.36       0.1678         61.4213         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)             19_136.79       707.20    19_843.99       0.1664        101.1288         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)            19_136.79       786.14    19_922.93       0.5242          1.0139         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)            19_136.79       879.38    20_016.17       0.6912          1.0069         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)            19_136.79       787.41    19_924.20       0.5164          1.0144         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)            19_136.79       888.22    20_025.01       0.6805          1.0072         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)            19_136.79       797.41    19_934.19       0.5062          1.0149         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)            19_136.79       896.89    20_033.68       0.6652          1.0078         5.21
IVF-Binary-512-nl223-pca (self)                       19_136.79     2_460.29    21_597.08       0.5171          1.0144         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)             19_832.27       717.98    20_550.24       0.1692         36.9672         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)             19_832.27       701.30    20_533.57       0.1686         45.7279         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)             19_832.27       716.21    20_548.47       0.1671         81.1721         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)            19_832.27       800.28    20_632.55       0.5248          1.0139         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)            19_832.27       892.97    20_725.24       0.6920          1.0069         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)            19_832.27       819.63    20_651.90       0.5209          1.0141         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)            19_832.27       900.71    20_732.98       0.6869          1.0070         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)            19_832.27       806.68    20_638.95       0.5110          1.0147         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)            19_832.27       912.27    20_744.53       0.6724          1.0075         5.48
IVF-Binary-512-nl316-pca (self)                       19_832.27     2_497.53    22_329.79       0.5215          1.0142         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)          38_032.52     1_306.67    39_339.20       0.0862          1.1524         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)         38_032.52     1_313.54    39_346.06       0.0856          1.1557         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)         38_032.52     1_314.82    39_347.35       0.0856          1.1557         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)         38_032.52     1_379.69    39_412.22       0.2153          1.0421         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)         38_032.52     1_468.91    39_501.43       0.3259          1.0274         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)        38_032.52     1_394.65    39_427.18       0.2093          1.0434         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)        38_032.52     1_488.96    39_521.49       0.3144          1.0285         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)        38_032.52     1_414.73    39_447.26       0.2093          1.0434         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)        38_032.52     1_499.85    39_532.38       0.3144          1.0285         9.57
IVF-Binary-1024-nl158-random (self)                   38_032.52     4_489.11    42_521.63       0.2103          1.0432         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)         36_349.42     1_313.61    37_663.03       0.0874          1.1495         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)         36_349.42     1_321.32    37_670.74       0.0869          1.1521         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)         36_349.42     1_323.74    37_673.16       0.0867          1.1533         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)        36_349.42     1_400.14    37_749.56       0.2155          1.0418         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)        36_349.42     1_488.67    37_838.09       0.3261          1.0271         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)        36_349.42     1_410.74    37_760.16       0.2116          1.0428         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)        36_349.42     1_508.65    37_858.06       0.3183          1.0280         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)        36_349.42     1_429.44    37_778.86       0.2105          1.0431         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)        36_349.42     1_550.68    37_900.10       0.3163          1.0283         9.76
IVF-Binary-1024-nl223-random (self)                   36_349.42     4_542.43    40_891.85       0.2124          1.0426         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)         37_272.49     1_338.97    38_611.46       0.0873          1.1489        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)         37_272.49     1_322.90    38_595.39       0.0870          1.1503        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)         37_272.49     1_347.87    38_620.36       0.0867          1.1525        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)        37_272.49     1_423.17    38_695.66       0.2157          1.0419        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)        37_272.49     1_524.49    38_796.98       0.3255          1.0272        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)        37_272.49     1_428.51    38_701.00       0.2134          1.0424        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)        37_272.49     1_540.94    38_813.43       0.3215          1.0277        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)        37_272.49     1_433.65    38_706.15       0.2111          1.0430        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)        37_272.49     1_549.53    38_822.02       0.3170          1.0283        10.04
IVF-Binary-1024-nl316-random (self)                   37_272.49     4_588.78    41_861.27       0.2140          1.0422        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)             38_583.11     1_314.26    39_897.37       0.2023          1.0850         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)            38_583.11     1_335.74    39_918.85       0.2014          1.0856         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)            38_583.11     1_335.90    39_919.01       0.2014          1.0856         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)            38_583.11     1_416.37    39_999.48       0.6330          1.0089         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)            38_583.11     1_500.94    40_084.06       0.7964          1.0037         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)           38_583.11     1_420.65    40_003.77       0.6293          1.0091         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)           38_583.11     1_521.86    40_104.98       0.7921          1.0038         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)           38_583.11     1_421.85    40_004.97       0.6293          1.0091         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)           38_583.11     1_517.30    40_100.41       0.7921          1.0038         9.57
IVF-Binary-1024-nl158-pca (self)                      38_583.11     4_541.43    43_124.55       0.6284          1.0092         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)            36_891.67     1_301.01    38_192.68       0.2024          1.0844         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)            36_891.67     1_311.24    38_202.91       0.2017          1.0852         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)            36_891.67     1_328.47    38_220.14       0.2015          1.0855         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)           36_891.67     1_405.38    38_297.06       0.6335          1.0089         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)           36_891.67     1_496.16    38_387.83       0.7966          1.0037         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)           36_891.67     1_401.54    38_293.22       0.6307          1.0090         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)           36_891.67     1_519.49    38_411.16       0.7932          1.0038         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)           36_891.67     1_440.96    38_332.63       0.6300          1.0090         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)           36_891.67     1_517.76    38_409.43       0.7923          1.0038         9.76
IVF-Binary-1024-nl223-pca (self)                      36_891.67     4_553.66    41_445.33       0.6293          1.0091         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)            37_216.98     1_319.77    38_536.75       0.2024          1.0845        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)            37_216.98     1_342.72    38_559.70       0.2019          1.0850        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)            37_216.98     1_334.70    38_551.68       0.2016          1.0855        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)           37_216.98     1_432.40    38_649.37       0.6326          1.0089        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)           37_216.98     1_507.60    38_724.57       0.7959          1.0037        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)           37_216.98     1_417.81    38_634.79       0.6310          1.0090        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)           37_216.98     1_514.34    38_731.32       0.7942          1.0037        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)           37_216.98     1_433.90    38_650.88       0.6295          1.0091        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)           37_216.98     1_557.42    38_774.40       0.7921          1.0038        10.04
IVF-Binary-1024-nl316-pca (self)                      37_216.98     4_562.71    41_779.68       0.6300          1.0091        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)              2_998.42       589.29     3_587.71       0.1931        105.5071         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)             2_998.42       619.50     3_617.93       0.1709        331.1900         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)             2_998.42       663.30     3_661.72       0.1646        536.1351         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)             2_998.42       670.13     3_668.56       0.5703          1.0122         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)             2_998.42     1_080.49     4_078.91       0.7499          1.0051         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)            2_998.42       671.33     3_669.76       0.4903          1.0166         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)            2_998.42     1_179.12     4_177.55       0.6594          1.0080         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)            2_998.42       708.99     3_707.41       0.4435          1.0200         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)            2_998.42     1_209.94     4_208.37       0.5966          1.0106         5.04
IVF-Binary-768-nl158-sign (self)                       2_998.42     2_092.75     5_091.17       0.4905          1.0167         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)             1_413.96       636.98     2_050.94       0.1853        194.6633         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)             1_413.96       653.78     2_067.74       0.1720        268.8091         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)             1_413.96       688.84     2_102.80       0.1587        556.5721         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)            1_413.96       669.90     2_083.86       0.5318          1.0142         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)            1_413.96     1_147.53     2_561.49       0.7082          1.0064         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)            1_413.96       695.15     2_109.11       0.4913          1.0166         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)            1_413.96     1_160.39     2_574.35       0.6613          1.0081         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)            1_413.96       733.62     2_147.58       0.4267          1.0214         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)            1_413.96     1_241.13     2_655.09       0.5770          1.0115         5.23
IVF-Binary-768-nl223-sign (self)                       1_413.96     2_152.79     3_566.75       0.4919          1.0167         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)             1_983.62       647.94     2_631.56       0.1808        170.0529         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)             1_983.62       662.81     2_646.43       0.1738        225.2926         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)             1_983.62       702.29     2_685.91       0.1578        447.2399         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)            1_983.62       712.52     2_696.13       0.5242          1.0145         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)            1_983.62     1_173.60     3_157.22       0.7034          1.0066         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)            1_983.62       727.22     2_710.84       0.5024          1.0158         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)            1_983.62     1_191.54     3_175.16       0.6787          1.0074         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)            1_983.62       774.11     2_757.73       0.4362          1.0204         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)            1_983.62     1_255.50     3_239.12       0.5990          1.0105         5.51
IVF-Binary-768-nl316-sign (self)                       1_983.62     2_260.92     4_244.54       0.5039          1.0159         5.51
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.99       669.88       702.86       1.0000          1.0000        48.83
Exhaustive (self)                                         32.99     2_203.18     2_236.16       1.0000          1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_703.86       290.93     2_994.79       0.0883          1.6472         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_703.86       395.80     3_099.66       0.3318          1.1509         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_703.86       493.95     3_197.82       0.4735          1.0879         1.78
ExhaustiveBinary-256-random (self)                     2_703.86     1_261.79     3_965.65       0.3570          1.1574         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_738.77       294.64     3_033.41       0.1109         80.1365         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_738.77       404.63     3_143.39       0.3158          1.5913         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_738.77       518.38     3_257.15       0.4282          1.3071         1.78
ExhaustiveBinary-256-pca (self)                        2_738.77     1_307.44     4_046.20       0.2950          2.2349         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_219.93       452.96     5_672.89       0.1384          1.5143         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_219.93       560.78     5_780.71       0.4321          1.0993         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_219.93       667.88     5_887.81       0.5844          1.0539         3.55
ExhaustiveBinary-512-random (self)                     5_219.93     1_822.66     7_042.59       0.4549          1.1058         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_441.87       472.61     5_914.48       0.1283          1.7193         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_441.87       574.41     6_016.28       0.4093          1.1233         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_441.87       685.26     6_127.13       0.5766          1.0656         3.55
ExhaustiveBinary-512-pca (self)                        5_441.87     1_864.26     7_306.13       0.4109          1.1443         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_503.31       798.31    11_301.62       0.1995          1.3854         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_503.31       913.22    11_416.53       0.5516          1.0599         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_503.31     1_035.81    11_539.12       0.7089          1.0295         7.10
ExhaustiveBinary-1024-random (self)                   10_503.31     2_981.89    13_485.20       0.5787          1.0643         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_580.48       800.22    11_380.71       0.1536          1.5190         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_580.48       916.26    11_496.74       0.4703          1.0875         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_580.48     1_026.99    11_607.47       0.6409          1.0449         7.10
ExhaustiveBinary-1024-pca (self)                      10_580.48     3_045.40    13_625.88       0.4638          1.1066         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   60.01       483.52       543.53       0.0927          1.6630         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    60.01       511.54       571.55       0.3446          1.1453         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    60.01       807.85       867.86       0.5011          1.0792         1.53
ExhaustiveBinary-256-sign (self)                          60.01     1_672.78     1_732.79       0.3670          1.1539         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)            3_659.32       127.45     3_786.77       0.0925          1.6368         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           3_659.32       127.20     3_786.52       0.0925          1.6370         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           3_659.32       128.00     3_787.32       0.0925          1.6370         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           3_659.32       183.93     3_843.25       0.3386          1.1478         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           3_659.32       234.20     3_893.52       0.4784          1.0862         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          3_659.32       182.28     3_841.60       0.3378          1.1479         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          3_659.32       235.48     3_894.81       0.4781          1.0862         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          3_659.32       184.57     3_843.89       0.3378          1.1479         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          3_659.32       242.45     3_901.77       0.4781          1.0863         1.93
IVF-Binary-256-nl158-random (self)                     3_659.32       507.71     4_167.03       0.3625          1.1542         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           3_284.84       128.78     3_413.62       0.1059          1.5891         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           3_284.84       139.95     3_424.79       0.1059          1.5892         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           3_284.84       138.11     3_422.95       0.1059          1.5892         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          3_284.84       190.59     3_475.42       0.3701          1.1286         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          3_284.84       236.58     3_521.41       0.5088          1.0753         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          3_284.84       191.91     3_476.75       0.3700          1.1286         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          3_284.84       240.57     3_525.41       0.5087          1.0753         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          3_284.84       191.32     3_476.16       0.3700          1.1286         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          3_284.84       244.66     3_529.50       0.5087          1.0753         2.00
IVF-Binary-256-nl223-random (self)                     3_284.84       537.41     3_822.25       0.3938          1.1339         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           3_479.96       132.68     3_612.64       0.1116          1.5721         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           3_479.96       137.59     3_617.55       0.1116          1.5725         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           3_479.96       139.25     3_619.21       0.1116          1.5725         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          3_479.96       192.21     3_672.17       0.3793          1.1238         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          3_479.96       255.62     3_735.59       0.5182          1.0725         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          3_479.96       192.83     3_672.79       0.3791          1.1239         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          3_479.96       239.87     3_719.84       0.5181          1.0725         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          3_479.96       194.46     3_674.42       0.3791          1.1239         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          3_479.96       248.18     3_728.14       0.5180          1.0725         2.09
IVF-Binary-256-nl316-random (self)                     3_479.96       531.95     4_011.91       0.4033          1.1281         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               3_739.21       125.10     3_864.31       0.1196          2.2348         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              3_739.21       133.09     3_872.30       0.1159          2.7209         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              3_739.21       131.90     3_871.11       0.1149          3.1662         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              3_739.21       196.47     3_935.68       0.3989          1.1281         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              3_739.21       254.70     3_993.91       0.5641          1.0697         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             3_739.21       212.41     3_951.62       0.3773          1.1546         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             3_739.21       257.05     3_996.26       0.5416          1.0820         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             3_739.21       209.25     3_948.46       0.3706          1.1749         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             3_739.21       256.96     3_996.17       0.5311          1.0926         1.93
IVF-Binary-256-nl158-pca (self)                        3_739.21       588.83     4_328.03       0.3839          1.1735         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              3_355.73       129.30     3_485.04       0.1170          2.4465         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              3_355.73       133.20     3_488.93       0.1162          2.7282         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              3_355.73       137.74     3_493.47       0.1153          3.4563         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             3_355.73       203.01     3_558.75       0.3793          1.1454         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             3_355.73       266.91     3_622.65       0.5457          1.0767         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             3_355.73       200.70     3_556.44       0.3747          1.1574         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             3_355.73       262.23     3_617.96       0.5384          1.0831         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             3_355.73       215.21     3_570.95       0.3682          1.1785         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             3_355.73       265.39     3_621.12       0.5270          1.0946         2.00
IVF-Binary-256-nl223-pca (self)                        3_355.73       589.38     3_945.12       0.3821          1.1748         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              3_576.87       139.72     3_716.59       0.1168          2.4348         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              3_576.87       137.19     3_714.06       0.1163          2.6126         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              3_576.87       141.40     3_718.27       0.1153          3.2327         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             3_576.87       215.85     3_792.71       0.3788          1.1469         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             3_576.87       265.67     3_842.54       0.5451          1.0773         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             3_576.87       205.38     3_782.25       0.3765          1.1518         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             3_576.87       259.94     3_836.80       0.5413          1.0802         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             3_576.87       208.84     3_785.71       0.3699          1.1709         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             3_576.87       267.21     3_844.07       0.5302          1.0908         2.09
IVF-Binary-256-nl316-pca (self)                        3_576.87       604.29     4_181.16       0.3839          1.1696         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_317.15       218.61     6_535.77       0.1407          1.5101         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_317.15       226.66     6_543.82       0.1406          1.5102         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_317.15       228.35     6_545.51       0.1406          1.5102         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_317.15       284.92     6_602.08       0.4352          1.0982         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_317.15       338.20     6_655.35       0.5866          1.0534         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_317.15       312.76     6_629.91       0.4348          1.0983         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_317.15       338.14     6_655.29       0.5864          1.0534         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_317.15       295.45     6_612.61       0.4348          1.0983         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_317.15       340.57     6_657.72       0.5864          1.0534         3.71
IVF-Binary-512-nl158-random (self)                     6_317.15       854.77     7_171.93       0.4573          1.1049         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_901.25       229.30     6_130.55       0.1491          1.4885         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_901.25       235.52     6_136.78       0.1491          1.4885         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_901.25       234.88     6_136.14       0.1491          1.4885         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_901.25       290.92     6_192.18       0.4493          1.0924         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_901.25       360.82     6_262.07       0.5986          1.0505         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_901.25       288.69     6_189.94       0.4493          1.0924         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_901.25       334.98     6_236.23       0.5986          1.0505         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_901.25       293.36     6_194.61       0.4493          1.0924         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_901.25       349.21     6_250.47       0.5986          1.0505         3.77
IVF-Binary-512-nl223-random (self)                     5_901.25       853.57     6_754.82       0.4712          1.0994         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           6_091.67       228.31     6_319.98       0.1514          1.4814         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           6_091.67       229.39     6_321.07       0.1514          1.4815         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           6_091.67       242.94     6_334.61       0.1514          1.4815         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          6_091.67       286.20     6_377.88       0.4542          1.0908         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          6_091.67       339.56     6_431.24       0.6033          1.0495         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          6_091.67       294.10     6_385.78       0.4541          1.0908         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          6_091.67       343.58     6_435.25       0.6032          1.0495         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          6_091.67       299.05     6_390.73       0.4541          1.0908         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          6_091.67       347.08     6_438.75       0.6032          1.0495         3.86
IVF-Binary-512-nl316-random (self)                     6_091.67       873.14     6_964.82       0.4756          1.0976         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               6_322.10       225.17     6_547.27       0.1322          1.6282         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              6_322.10       223.95     6_546.05       0.1293          1.6816         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              6_322.10       225.52     6_547.62       0.1287          1.7030         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              6_322.10       290.19     6_612.30       0.4307          1.1079         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              6_322.10       344.10     6_666.20       0.5984          1.0571         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             6_322.10       290.94     6_613.04       0.4144          1.1177         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             6_322.10       350.93     6_673.03       0.5839          1.0618         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             6_322.10       303.05     6_625.15       0.4108          1.1209         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             6_322.10       350.16     6_672.26       0.5792          1.0638         3.71
IVF-Binary-512-nl158-pca (self)                        6_322.10       898.57     7_220.68       0.4164          1.1371         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              5_980.83       228.11     6_208.94       0.1302          1.6704         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              5_980.83       231.01     6_211.85       0.1298          1.6865         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              5_980.83       263.72     6_244.56       0.1294          1.7053         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             5_980.83       292.24     6_273.07       0.4158          1.1153         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             5_980.83       355.57     6_336.40       0.5850          1.0604         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             5_980.83       298.12     6_278.95       0.4133          1.1179         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             5_980.83       348.11     6_328.94       0.5819          1.0620         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             5_980.83       319.51     6_300.35       0.4110          1.1205         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             5_980.83       356.54     6_337.37       0.5788          1.0636         3.77
IVF-Binary-512-nl223-pca (self)                        5_980.83       907.56     6_888.40       0.4152          1.1371         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              6_150.93       233.32     6_384.25       0.1303          1.6707         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              6_150.93       229.68     6_380.61       0.1300          1.6804         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              6_150.93       233.53     6_384.46       0.1296          1.7019         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             6_150.93       299.48     6_450.41       0.4150          1.1160         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             6_150.93       355.44     6_506.38       0.5846          1.0607         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             6_150.93       303.99     6_454.93       0.4138          1.1173         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             6_150.93       372.76     6_523.69       0.5829          1.0615         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             6_150.93       307.81     6_458.74       0.4111          1.1200         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             6_150.93       362.72     6_513.65       0.5794          1.0632         3.86
IVF-Binary-512-nl316-pca (self)                        6_150.93       911.29     7_062.22       0.4158          1.1364         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_415.33       417.48    11_832.81       0.2007          1.3838         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_415.33       417.20    11_832.53       0.2007          1.3838         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_415.33       418.81    11_834.14       0.2007          1.3838         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_415.33       481.10    11_896.43       0.5525          1.0596         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_415.33       547.37    11_962.70       0.7098          1.0293         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_415.33       482.40    11_897.73       0.5524          1.0596         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_415.33       542.40    11_957.73       0.7098          1.0293         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_415.33       501.35    11_916.68       0.5524          1.0596         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_415.33       547.19    11_962.52       0.7098          1.0293         7.26
IVF-Binary-1024-nl158-random (self)                   11_415.33     1_561.93    12_977.26       0.5796          1.0641         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_947.97       413.92    11_361.89       0.2055          1.3748         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_947.97       424.80    11_372.78       0.2055          1.3748         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_947.97       424.68    11_372.66       0.2055          1.3748         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_947.97       470.95    11_418.92       0.5594          1.0579         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_947.97       521.26    11_469.23       0.7146          1.0285         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_947.97       474.77    11_422.74       0.5594          1.0579         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_947.97       523.78    11_471.75       0.7146          1.0285         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_947.97       482.69    11_430.67       0.5594          1.0579         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_947.97       543.75    11_491.72       0.7146          1.0285         7.32
IVF-Binary-1024-nl223-random (self)                   10_947.97     1_519.61    12_467.58       0.5861          1.0623         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         11_139.19       419.99    11_559.18       0.2070          1.3717         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)         11_139.19       421.86    11_561.05       0.2070          1.3717         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)         11_139.19       429.66    11_568.85       0.2070          1.3717         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)        11_139.19       489.93    11_629.12       0.5618          1.0573         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)        11_139.19       529.96    11_669.15       0.7169          1.0282         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)        11_139.19       479.08    11_618.27       0.5618          1.0573         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)        11_139.19       530.65    11_669.84       0.7169          1.0282         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)        11_139.19       494.27    11_633.46       0.5618          1.0573         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)        11_139.19       544.91    11_684.09       0.7169          1.0282         7.42
IVF-Binary-1024-nl316-random (self)                   11_139.19     1_496.46    12_635.65       0.5887          1.0616         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)             11_369.94       410.50    11_780.44       0.1551          1.5010         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            11_369.94       418.15    11_788.09       0.1538          1.5156         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            11_369.94       416.46    11_786.40       0.1538          1.5171         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            11_369.94       506.75    11_876.69       0.4829          1.0838         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            11_369.94       529.59    11_899.53       0.6509          1.0428         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           11_369.94       487.20    11_857.14       0.4724          1.0867         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           11_369.94       534.30    11_904.24       0.6434          1.0443         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           11_369.94       481.91    11_851.85       0.4710          1.0871         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           11_369.94       551.88    11_921.82       0.6414          1.0447         7.26
IVF-Binary-1024-nl158-pca (self)                      11_369.94     1_560.54    12_930.48       0.4658          1.1058         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            10_988.45       416.64    11_405.09       0.1546          1.5129         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            10_988.45       418.62    11_407.07       0.1545          1.5147         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            10_988.45       426.58    11_415.03       0.1545          1.5156         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           10_988.45       482.63    11_471.08       0.4737          1.0859         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           10_988.45       531.02    11_519.47       0.6441          1.0439         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           10_988.45       491.20    11_479.65       0.4725          1.0864         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           10_988.45       533.94    11_522.39       0.6427          1.0443         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           10_988.45       503.59    11_492.04       0.4717          1.0866         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           10_988.45       557.17    11_545.62       0.6417          1.0445         7.32
IVF-Binary-1024-nl223-pca (self)                      10_988.45     1_529.38    12_517.83       0.4656          1.1055         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            11_200.15       418.86    11_619.01       0.1549          1.5124         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            11_200.15       419.45    11_619.60       0.1548          1.5138         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            11_200.15       439.26    11_639.42       0.1547          1.5151         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           11_200.15       484.31    11_684.47       0.4732          1.0861         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           11_200.15       535.29    11_735.45       0.6439          1.0440         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           11_200.15       488.20    11_688.35       0.4726          1.0864         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           11_200.15       536.52    11_736.68       0.6432          1.0442         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           11_200.15       508.81    11_708.96       0.4716          1.0867         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           11_200.15       558.07    11_758.23       0.6419          1.0444         7.42
IVF-Binary-1024-nl316-pca (self)                      11_200.15     1_565.92    12_766.07       0.4659          1.1055         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)                990.58       262.02     1_252.60       0.1181          8.9404         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)               990.58       280.78     1_271.36       0.1030         22.8355         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)               990.58       303.23     1_293.82       0.0901         39.0570         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)               990.58       291.59     1_282.17       0.8716          1.0202         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)               990.58       500.91     1_491.49       0.9479          1.0093         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)              990.58       316.59     1_307.17       0.7958          1.0555         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)              990.58       538.77     1_529.35       0.9253          1.0139         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)              990.58       331.17     1_321.75       0.7262          1.1070         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)              990.58       559.00     1_549.58       0.8927          1.0226         1.68
IVF-Binary-256-nl158-sign (self)                         990.58       977.10     1_967.69       0.8248          1.0405         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               634.25       296.65       930.90       0.1164         23.6305         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               634.25       310.35       944.60       0.1087         33.0540         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               634.25       340.70       974.95       0.0924         64.0223         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              634.25       329.80       964.04       0.8017          1.0297         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              634.25       575.88     1_210.13       0.9323          1.0095         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              634.25       354.19       988.44       0.7470          1.0466         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              634.25       590.77     1_225.01       0.9102          1.0133         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              634.25       370.97     1_005.22       0.6397          1.1152         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              634.25       647.64     1_281.89       0.8492          1.0243         1.75
IVF-Binary-256-nl223-sign (self)                         634.25     1_059.36     1_693.61       0.7850          1.0405         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               829.36       324.33     1_153.68       0.1424         25.1331         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               829.36       332.23     1_161.59       0.1293         29.1378         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               829.36       360.21     1_189.56       0.1047         67.1736         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              829.36       352.40     1_181.75       0.7897          1.0337         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              829.36       599.19     1_428.55       0.9218          1.0106         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              829.36       361.29     1_190.64       0.7606          1.0405         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              829.36       605.05     1_434.40       0.9097          1.0126         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              829.36       391.61     1_220.97       0.6515          1.0819         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              829.36       639.27     1_468.63       0.8496          1.0231         1.84
IVF-Binary-256-nl316-sign (self)                         829.36     1_102.57     1_931.92       0.7929          1.0389         1.84
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.05     1_234.67     1_302.72       1.0000          1.0000        97.66
Exhaustive (self)                                         68.05     4_066.10     4_134.15       1.0000          1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_833.19       430.97     6_264.16       0.0656          1.4921         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_833.19       543.07     6_376.26       0.2689          1.1363         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_833.19       676.01     6_509.21       0.3927          1.0839         2.03
ExhaustiveBinary-256-random (self)                     5_833.19     1_744.12     7_577.31       0.2893          1.1351         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_011.06       431.42     6_442.48       0.1753        309.4819         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_011.06       563.75     6_574.81       0.4663          1.2428         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_011.06       699.00     6_710.05       0.5849          1.1744         2.03
ExhaustiveBinary-256-pca (self)                        6_011.06     1_798.60     7_809.66       0.4694          1.2512         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_500.86       688.23    12_189.08       0.1027          1.4104         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_500.86       800.19    12_301.05       0.3403          1.0982         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_500.86       929.40    12_430.25       0.4712          1.0574         4.05
ExhaustiveBinary-512-random (self)                    11_500.86     2_594.21    14_095.06       0.3573          1.1005         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_652.99       681.77    12_334.76       0.1672       1095.7764         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_652.99       818.73    12_471.72       0.4213          1.4308         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_652.99       955.67    12_608.66       0.5294          1.2408         4.05
ExhaustiveBinary-512-pca (self)                       11_652.99     2_647.18    14_300.18       0.4105          1.6976         4.05
ExhaustiveBinary-1024-random_no_rr (query)            22_729.80     1_309.83    24_039.63       0.1484          1.3311         8.11
ExhaustiveBinary-1024-random-rf10 (query)             22_729.80     1_357.96    24_087.76       0.4138          1.0699         8.11
ExhaustiveBinary-1024-random-rf20 (query)             22_729.80     1_498.95    24_228.75       0.5532          1.0397         8.11
ExhaustiveBinary-1024-random (self)                   22_729.80     4_438.08    27_167.88       0.4277          1.0760         8.11
ExhaustiveBinary-1024-pca_no_rr (query)               22_977.37     1_230.01    24_207.38       0.2384          1.4238         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                22_977.37     1_366.39    24_343.77       0.6733          1.0459         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                22_977.37     1_513.21    24_490.59       0.8205          1.0198         8.11
ExhaustiveBinary-1024-pca (self)                      22_977.37     4_500.50    27_477.87       0.6814          1.0491         8.11
ExhaustiveBinary-512-sign_no_rr (query)                  129.93       687.84       817.77       0.1122          1.4085         3.05
ExhaustiveBinary-512-sign-rf10 (query)                   129.93       742.57       872.50       0.3436          1.0978         3.05
ExhaustiveBinary-512-sign-rf20 (query)                   129.93     1_186.38     1_316.31       0.4809          1.0554         3.05
ExhaustiveBinary-512-sign (self)                         129.93     2_338.36     2_468.29       0.3589          1.1017         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)            7_414.41       249.84     7_664.25       0.0682          1.4854         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           7_414.41       256.76     7_671.17       0.0682          1.4856         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           7_414.41       250.35     7_664.76       0.0682          1.4856         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           7_414.41       333.58     7_747.99       0.2719          1.1354         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           7_414.41       406.59     7_821.00       0.3944          1.0835         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          7_414.41       326.41     7_740.82       0.2709          1.1355         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          7_414.41       408.40     7_822.81       0.3938          1.0835         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          7_414.41       330.42     7_744.83       0.2709          1.1355         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          7_414.41       416.64     7_831.05       0.3937          1.0836         2.34
IVF-Binary-256-nl158-random (self)                     7_414.41       922.35     8_336.76       0.2912          1.1342         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           6_699.06       255.63     6_954.68       0.0816          1.4541         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)           6_699.06       255.19     6_954.25       0.0816          1.4542         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)           6_699.06       257.27     6_956.33       0.0816          1.4542         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)          6_699.06       337.50     7_036.55       0.3014          1.1172         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)          6_699.06       411.25     7_110.31       0.4247          1.0722         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)          6_699.06       337.45     7_036.51       0.3012          1.1174         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)          6_699.06       419.40     7_118.46       0.4240          1.0724         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)          6_699.06       339.01     7_038.07       0.3012          1.1174         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)          6_699.06       427.96     7_127.02       0.4239          1.0724         2.47
IVF-Binary-256-nl223-random (self)                     6_699.06       956.04     7_655.10       0.3207          1.1157         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)           7_147.98       260.94     7_408.92       0.0882          1.4389         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           7_147.98       263.27     7_411.26       0.0881          1.4390         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           7_147.98       264.86     7_412.84       0.0881          1.4391         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          7_147.98       344.22     7_492.21       0.3120          1.1115         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          7_147.98       421.67     7_569.65       0.4343          1.0692         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          7_147.98       342.30     7_490.28       0.3118          1.1116         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          7_147.98       420.47     7_568.45       0.4337          1.0693         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          7_147.98       346.46     7_494.44       0.3116          1.1117         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          7_147.98       424.74     7_572.72       0.4333          1.0694         2.65
IVF-Binary-256-nl316-random (self)                     7_147.98       981.42     8_129.41       0.3309          1.1095         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               7_588.89       248.07     7_836.95       0.1937          2.5726         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              7_588.89       252.93     7_841.82       0.1882          3.1866         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              7_588.89       254.42     7_843.31       0.1871          3.7048         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              7_588.89       347.92     7_936.80       0.5814          1.0746         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              7_588.89       432.32     8_021.21       0.7398          1.0348         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             7_588.89       358.51     7_947.39       0.5579          1.0958         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             7_588.89       446.91     8_035.79       0.7182          1.0418         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             7_588.89       355.78     7_944.67       0.5520          1.1115         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             7_588.89       445.05     8_033.93       0.7102          1.0476         2.34
IVF-Binary-256-nl158-pca (self)                        7_588.89     1_025.59     8_614.47       0.5791          1.0992         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              6_914.11       260.44     7_174.55       0.1890          2.8892         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              6_914.11       261.36     7_175.47       0.1881          3.4498         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              6_914.11       261.70     7_175.81       0.1872          4.3649         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             6_914.11       353.52     7_267.63       0.5606          1.0856         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             6_914.11       439.14     7_353.25       0.7221          1.0378         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             6_914.11       352.68     7_266.78       0.5559          1.0946         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             6_914.11       441.59     7_355.70       0.7152          1.0414         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             6_914.11       358.55     7_272.66       0.5504          1.1111         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             6_914.11       446.62     7_360.72       0.7069          1.0481         2.47
IVF-Binary-256-nl223-pca (self)                        6_914.11     1_059.97     7_974.08       0.5768          1.0982         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              7_339.09       266.55     7_605.64       0.1889          2.9462         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              7_339.09       266.98     7_606.07       0.1885          3.2387         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              7_339.09       270.16     7_609.25       0.1875          4.1378         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             7_339.09       361.62     7_700.71       0.5596          1.0869         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             7_339.09       452.20     7_791.30       0.7207          1.0384         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             7_339.09       360.10     7_699.19       0.5574          1.0905         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             7_339.09       448.42     7_787.51       0.7175          1.0399         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             7_339.09       363.28     7_702.37       0.5514          1.1070         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             7_339.09       453.91     7_793.00       0.7091          1.0462         2.65
IVF-Binary-256-nl316-pca (self)                        7_339.09     1_066.91     8_406.01       0.5783          1.0941         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           12_945.89       462.16    13_408.05       0.1037          1.4086         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          12_945.89       457.28    13_403.17       0.1037          1.4086         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          12_945.89       461.83    13_407.72       0.1037          1.4086         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          12_945.89       525.16    13_471.06       0.3407          1.0981         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          12_945.89       603.17    13_549.07       0.4715          1.0573         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         12_945.89       524.93    13_470.83       0.3407          1.0981         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         12_945.89       611.91    13_557.80       0.4715          1.0573         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         12_945.89       530.16    13_476.06       0.3407          1.0981         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         12_945.89       648.52    13_594.42       0.4715          1.0573         4.36
IVF-Binary-512-nl158-random (self)                    12_945.89     1_605.16    14_551.05       0.3577          1.1003         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          12_347.43       452.02    12_799.45       0.1132          1.3888         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          12_347.43       456.58    12_804.01       0.1131          1.3891         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          12_347.43       462.77    12_810.20       0.1131          1.3891         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         12_347.43       533.26    12_880.69       0.3571          1.0910         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         12_347.43       612.00    12_959.43       0.4871          1.0533         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         12_347.43       532.28    12_879.71       0.3566          1.0912         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         12_347.43       610.51    12_957.94       0.4863          1.0535         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         12_347.43       551.09    12_898.52       0.3566          1.0912         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         12_347.43       619.41    12_966.84       0.4863          1.0535         4.49
IVF-Binary-512-nl223-random (self)                    12_347.43     1_621.58    13_969.01       0.3716          1.0945         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          12_768.76       466.36    13_235.12       0.1167          1.3816         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          12_768.76       463.50    13_232.26       0.1167          1.3818         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          12_768.76       468.84    13_237.60       0.1166          1.3819         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         12_768.76       541.97    13_310.73       0.3624          1.0887         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         12_768.76       616.71    13_385.47       0.4925          1.0522         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         12_768.76       544.11    13_312.87       0.3621          1.0888         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         12_768.76       627.43    13_396.19       0.4920          1.0524         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         12_768.76       548.83    13_317.59       0.3620          1.0889         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         12_768.76       624.74    13_393.50       0.4916          1.0524         4.67
IVF-Binary-512-nl316-random (self)                    12_768.76     1_654.51    14_423.27       0.3767          1.0925         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              13_120.17       454.60    13_574.76       0.2260          3.6107         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             13_120.17       455.65    13_575.82       0.2194          5.7358         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             13_120.17       464.10    13_584.26       0.2175          8.8217         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             13_120.17       545.83    13_666.00       0.6545          1.0666         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             13_120.17       619.42    13_739.59       0.8017          1.0301         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            13_120.17       546.84    13_667.01       0.6280          1.0924         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            13_120.17       628.54    13_748.71       0.7777          1.0388         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            13_120.17       545.22    13_665.39       0.6190          1.1147         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            13_120.17       633.74    13_753.90       0.7656          1.0484         4.36
IVF-Binary-512-nl158-pca (self)                       13_120.17     1_673.31    14_793.48       0.6371          1.0961         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             12_501.65       458.48    12_960.13       0.2206          5.2740         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             12_501.65       472.34    12_974.00       0.2189          7.4234         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             12_501.65       467.01    12_968.66       0.2164         13.2704         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            12_501.65       549.40    13_051.05       0.6310          1.0817         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            12_501.65       633.96    13_135.61       0.7810          1.0346         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            12_501.65       552.26    13_053.91       0.6238          1.0950         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            12_501.65       642.74    13_144.39       0.7721          1.0398         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            12_501.65       568.12    13_069.77       0.6123          1.1198         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            12_501.65       639.92    13_141.57       0.7574          1.0518         4.49
IVF-Binary-512-nl223-pca (self)                       12_501.65     1_693.96    14_195.61       0.6323          1.0983         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             12_943.56       470.93    13_414.48       0.2202          4.9794         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             12_943.56       473.53    13_417.09       0.2193          6.4216         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             12_943.56       481.03    13_424.58       0.2167         11.5553         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            12_943.56       556.52    13_500.08       0.6305          1.0828         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            12_943.56       643.07    13_586.63       0.7803          1.0349         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            12_943.56       568.98    13_512.54       0.6268          1.0884         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            12_943.56       640.79    13_584.35       0.7758          1.0371         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            12_943.56       562.21    13_505.76       0.6155          1.1127         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            12_943.56       645.70    13_589.25       0.7615          1.0479         4.67
IVF-Binary-512-nl316-pca (self)                       12_943.56     1_715.10    14_658.66       0.6353          1.0920         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          24_367.28       854.47    25_221.75       0.1488          1.3307         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)         24_367.28       867.68    25_234.97       0.1488          1.3307         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)         24_367.28       864.80    25_232.08       0.1488          1.3307         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)         24_367.28       947.23    25_314.51       0.4138          1.0699         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)         24_367.28     1_022.75    25_390.03       0.5533          1.0396         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)        24_367.28       938.43    25_305.71       0.4138          1.0699         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)        24_367.28     1_015.69    25_382.97       0.5533          1.0396         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)        24_367.28       942.41    25_309.69       0.4138          1.0699         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)        24_367.28     1_017.37    25_384.65       0.5533          1.0396         8.42
IVF-Binary-1024-nl158-random (self)                   24_367.28     2_975.99    27_343.27       0.4278          1.0760         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)         23_607.95       866.83    24_474.78       0.1538          1.3218         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         23_607.95       868.31    24_476.26       0.1537          1.3219         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         23_607.95       877.80    24_485.74       0.1537          1.3219         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        23_607.95       946.74    24_554.69       0.4217          1.0675         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        23_607.95     1_014.50    24_622.45       0.5606          1.0383         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        23_607.95       951.17    24_559.11       0.4213          1.0676         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        23_607.95     1_028.34    24_636.29       0.5600          1.0384         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        23_607.95     1_008.07    24_616.02       0.4213          1.0676         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        23_607.95     1_034.24    24_642.19       0.5600          1.0384         8.54
IVF-Binary-1024-nl223-random (self)                   23_607.95     3_020.06    26_628.01       0.4358          1.0736         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         24_028.58       873.60    24_902.18       0.1558          1.3182         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)         24_028.58       876.87    24_905.45       0.1557          1.3183         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)         24_028.58       889.38    24_917.96       0.1557          1.3184         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)        24_028.58       949.99    24_978.56       0.4242          1.0669         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)        24_028.58     1_037.86    25_066.44       0.5639          1.0378         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)        24_028.58       955.92    24_984.50       0.4240          1.0669         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)        24_028.58     1_031.06    25_059.63       0.5636          1.0379         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)        24_028.58       959.70    24_988.28       0.4238          1.0670         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)        24_028.58     1_036.64    25_065.21       0.5633          1.0380         8.73
IVF-Binary-1024-nl316-random (self)                   24_028.58     3_047.54    27_076.12       0.4386          1.0728         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)             24_387.57       866.85    25_254.42       0.2432          1.3581         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            24_387.57       883.45    25_271.02       0.2392          1.4031         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            24_387.57       875.38    25_262.95       0.2386          1.4170         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            24_387.57       944.28    25_331.85       0.6894          1.0422         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            24_387.57     1_039.63    25_427.20       0.8320          1.0183         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           24_387.57       942.55    25_330.12       0.6766          1.0450         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           24_387.57     1_027.21    25_414.77       0.8238          1.0193         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           24_387.57       945.56    25_333.12       0.6742          1.0455         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           24_387.57     1_033.68    25_421.25       0.8215          1.0195         8.42
IVF-Binary-1024-nl158-pca (self)                      24_387.57     3_014.40    27_401.97       0.6844          1.0483         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            23_880.60       881.59    24_762.19       0.2399          1.3930         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            23_880.60       871.48    24_752.08       0.2392          1.4054         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            23_880.60       883.70    24_764.30       0.2389          1.4186         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           23_880.60       958.16    24_838.76       0.6778          1.0436         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           23_880.60     1_028.92    24_909.52       0.8252          1.0185         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           23_880.60       956.13    24_836.73       0.6758          1.0445         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           23_880.60     1_034.01    24_914.61       0.8234          1.0190         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           23_880.60       975.98    24_856.58       0.6741          1.0450         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           23_880.60     1_044.45    24_925.05       0.8216          1.0193         8.54
IVF-Binary-1024-nl223-pca (self)                      23_880.60     3_044.09    26_924.69       0.6842          1.0477         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            24_315.82       882.94    25_198.76       0.2396          1.3949         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            24_315.82       876.26    25_192.08       0.2394          1.3998         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            24_315.82       884.94    25_200.75       0.2388          1.4160         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           24_315.82       999.93    25_315.75       0.6773          1.0438         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           24_315.82     1_037.97    25_353.79       0.8249          1.0186         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           24_315.82       960.73    25_276.54       0.6766          1.0441         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           24_315.82     1_046.06    25_361.88       0.8243          1.0188         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           24_315.82       979.76    25_295.58       0.6744          1.0449         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           24_315.82     1_056.29    25_372.11       0.8221          1.0192         8.73
IVF-Binary-1024-nl316-pca (self)                      24_315.82     3_069.50    27_385.32       0.6847          1.0474         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)              1_615.76       426.82     2_042.58       0.0796         12.0508         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)             1_615.76       440.97     2_056.73       0.0771         17.7820         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)             1_615.76       469.49     2_085.25       0.0767         20.0653         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)             1_615.76       455.61     2_071.38       0.8031          1.0346         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)             1_615.76       838.11     2_453.87       0.9295          1.0088         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)            1_615.76       491.16     2_106.92       0.6753          1.3610         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)            1_615.76       823.90     2_439.67       0.9067          1.0120         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)            1_615.76       506.25     2_122.02       0.5847          1.4920         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)            1_615.76       855.21     2_470.97       0.8736          1.0176         3.36
IVF-Binary-512-nl158-sign (self)                       1_615.76     1_399.81     3_015.58       0.7091          1.2592         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)             1_068.29       447.44     1_515.74       0.0999         20.0265         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)             1_068.29       464.56     1_532.85       0.0807         33.3106         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)             1_068.29       508.06     1_576.36       0.0726         66.1032         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)            1_068.29       517.90     1_586.19       0.6638          1.1579         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)            1_068.29       819.74     1_888.03       0.9054          1.0097         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)            1_068.29       504.50     1_572.79       0.5689          1.3836         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)            1_068.29       847.69     1_915.98       0.8783          1.0129         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)            1_068.29       538.90     1_607.19       0.4446          2.2376         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)            1_068.29       889.98     1_958.27       0.8040          1.0215         3.49
IVF-Binary-512-nl223-sign (self)                       1_068.29     1_491.37     2_559.67       0.6163          1.1785         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)             1_475.00       472.78     1_947.78       0.0942         20.0074         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)             1_475.00       481.22     1_956.22       0.0897         29.5793         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)             1_475.00       525.71     2_000.71       0.0746         49.6884         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)            1_475.00       511.20     1_986.19       0.6848          1.0538         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)            1_475.00       855.84     2_330.84       0.8963          1.0116         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)            1_475.00       523.58     1_998.58       0.6344          1.1266         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)            1_475.00       872.71     2_347.71       0.8807          1.0134         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)            1_475.00       565.73     2_040.72       0.5206          1.4766         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)            1_475.00       913.10     2_388.09       0.8242          1.0213         3.67
IVF-Binary-512-nl316-sign (self)                       1_475.00     1_564.07     3_039.07       0.6741          1.0530         3.67
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       100.41     1_770.75     1_871.16       1.0000          1.0000       146.48
Exhaustive (self)                                        100.41     5_942.06     6_042.47       1.0000          1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)              9_036.77       531.78     9_568.54       0.0560          1.3883         2.28
ExhaustiveBinary-256-random-rf10 (query)               9_036.77       666.30     9_703.07       0.2348          1.1233         2.28
ExhaustiveBinary-256-random-rf20 (query)               9_036.77       801.98     9_838.75       0.3419          1.0805         2.28
ExhaustiveBinary-256-random (self)                     9_036.77     2_116.62    11_153.39       0.2472          1.1217         2.28
ExhaustiveBinary-256-pca_no_rr (query)                 9_414.35       536.10     9_950.45       0.1693        237.6734         2.28
ExhaustiveBinary-256-pca-rf10 (query)                  9_414.35       688.08    10_102.43       0.4634          1.1896         2.28
ExhaustiveBinary-256-pca-rf20 (query)                  9_414.35       837.50    10_251.85       0.5917          1.1224         2.28
ExhaustiveBinary-256-pca (self)                        9_414.35     2_197.70    11_612.05       0.4804          1.1913         2.28
ExhaustiveBinary-512-random_no_rr (query)             17_657.20       920.61    18_577.81       0.0796          1.3400         4.55
ExhaustiveBinary-512-random-rf10 (query)              17_657.20     1_040.08    18_697.27       0.2824          1.0956         4.55
ExhaustiveBinary-512-random-rf20 (query)              17_657.20     1_178.78    18_835.98       0.3933          1.0588         4.55
ExhaustiveBinary-512-random (self)                    17_657.20     3_341.76    20_998.96       0.2970          1.0893         4.55
ExhaustiveBinary-512-pca_no_rr (query)                17_970.73       909.28    18_880.00       0.1882       1500.0959         4.55
ExhaustiveBinary-512-pca-rf10 (query)                 17_970.73     1_049.78    19_020.50       0.4651          1.2548         4.55
ExhaustiveBinary-512-pca-rf20 (query)                 17_970.73     1_204.64    19_175.36       0.5757          1.1920         4.55
ExhaustiveBinary-512-pca (self)                       17_970.73     3_437.71    21_408.44       0.4710          1.2567         4.55
ExhaustiveBinary-1024-random_no_rr (query)            35_294.73     1_687.89    36_982.61       0.1221          1.2753         9.11
ExhaustiveBinary-1024-random-rf10 (query)             35_294.73     1_809.31    37_104.04       0.3372          1.0685         9.11
ExhaustiveBinary-1024-random-rf20 (query)             35_294.73     1_968.96    37_263.69       0.4564          1.0414         9.11
ExhaustiveBinary-1024-random (self)                   35_294.73     5_941.83    41_236.56       0.3447          1.0714         9.11
ExhaustiveBinary-1024-pca_no_rr (query)               35_330.78     1_689.04    37_019.82       0.2681          1.8439         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                35_330.78     1_833.36    37_164.15       0.7069          1.0551         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                35_330.78     2_013.83    37_344.62       0.8317          1.0248         9.11
ExhaustiveBinary-1024-pca (self)                      35_330.78     6_005.45    41_336.24       0.7148          1.0572         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  219.08       889.09     1_108.17       0.1171          1.2871         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   219.08       954.89     1_173.97       0.3258          1.0737         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   219.08     1_525.56     1_744.64       0.4465          1.0436         4.58
ExhaustiveBinary-768-sign (self)                         219.08     3_096.66     3_315.74       0.3335          1.0755         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)           11_163.40       369.35    11_532.75       0.0586          1.3825         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)          11_163.40       364.75    11_528.15       0.0585          1.3826         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)          11_163.40       364.62    11_528.01       0.0585          1.3826         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)          11_163.40       459.52    11_622.91       0.2381          1.1220         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)          11_163.40       542.73    11_706.13       0.3443          1.0797         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)         11_163.40       447.83    11_611.23       0.2374          1.1221         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)         11_163.40       542.62    11_706.02       0.3438          1.0797         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)         11_163.40       451.09    11_614.49       0.2374          1.1221         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)         11_163.40       547.65    11_711.05       0.3438          1.0797         2.74
IVF-Binary-256-nl158-random (self)                    11_163.40     1_298.04    12_461.44       0.2499          1.1204         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)          10_218.35       372.79    10_591.14       0.0684          1.3633         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)          10_218.35       372.28    10_590.63       0.0684          1.3635         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)          10_218.35       382.91    10_601.26       0.0684          1.3635         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)         10_218.35       462.24    10_680.59       0.2654          1.1042         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)         10_218.35       563.93    10_782.27       0.3732          1.0669         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)         10_218.35       461.40    10_679.75       0.2653          1.1042         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)         10_218.35       555.82    10_774.16       0.3731          1.0669         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)         10_218.35       460.41    10_678.75       0.2653          1.1042         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)         10_218.35       558.94    10_777.28       0.3731          1.0669         2.93
IVF-Binary-256-nl223-random (self)                    10_218.35     1_320.54    11_538.88       0.2793          1.0999         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)          10_848.91       395.17    11_244.08       0.0754          1.3475         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)          10_848.91       390.66    11_239.57       0.0754          1.3476         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)          10_848.91       396.78    11_245.69       0.0754          1.3478         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)         10_848.91       480.79    11_329.70       0.2804          1.0950         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)         10_848.91       575.26    11_424.17       0.3885          1.0613         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)         10_848.91       480.80    11_329.71       0.2802          1.0950         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)         10_848.91       574.08    11_422.99       0.3883          1.0613         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)         10_848.91       478.79    11_327.70       0.2801          1.0951         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)         10_848.91       577.90    11_426.81       0.3882          1.0613         3.21
IVF-Binary-256-nl316-random (self)                    10_848.91     1_380.13    12_229.04       0.2938          1.0897         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)              11_531.07       363.51    11_894.58       0.1804          2.4698         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)             11_531.07       369.45    11_900.52       0.1754          2.9727         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)             11_531.07       366.20    11_897.27       0.1745          3.1815         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)             11_531.07       477.92    12_008.99       0.5384          1.0691         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)             11_531.07       576.74    12_107.81       0.6973          1.0352         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)            11_531.07       474.67    12_005.74       0.5174          1.0846         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)            11_531.07       576.50    12_107.57       0.6786          1.0396         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)            11_531.07       478.13    12_009.19       0.5129          1.0937         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)            11_531.07       587.83    12_118.89       0.6722          1.0428         2.74
IVF-Binary-256-nl158-pca (self)                       11_531.07     1_462.04    12_993.11       0.5382          1.0860         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)             10_657.57       374.85    11_032.42       0.1767          2.6320         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)             10_657.57       382.47    11_040.04       0.1755          2.9612         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)             10_657.57       377.06    11_034.64       0.1748          3.3432         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)            10_657.57       483.15    11_140.72       0.5229          1.0734         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)            10_657.57       585.38    11_242.95       0.6851          1.0350         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)            10_657.57       486.10    11_143.67       0.5180          1.0795         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)            10_657.57       589.98    11_247.56       0.6788          1.0374         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)            10_657.57       486.13    11_143.71       0.5135          1.0881         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)            10_657.57       589.27    11_246.84       0.6722          1.0407         2.93
IVF-Binary-256-nl223-pca (self)                       10_657.57     1_440.12    12_097.70       0.5385          1.0805         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)             11_205.33       393.48    11_598.81       0.1766          2.6847         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)             11_205.33       385.52    11_590.86       0.1761          2.8714         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)             11_205.33       390.16    11_595.49       0.1752          3.2819         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)            11_205.33       498.68    11_704.01       0.5213          1.0746         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)            11_205.33       598.98    11_804.31       0.6831          1.0357         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)            11_205.33       494.41    11_699.74       0.5189          1.0772         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)            11_205.33       609.27    11_814.60       0.6802          1.0367         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)            11_205.33       511.98    11_717.32       0.5141          1.0869         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)            11_205.33       618.61    11_823.94       0.6729          1.0404         3.21
IVF-Binary-256-nl316-pca (self)                       11_205.33     1_477.77    12_683.10       0.5394          1.0784         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)           19_904.34       670.53    20_574.87       0.0808          1.3381         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)          19_904.34       675.82    20_580.16       0.0808          1.3381         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)          19_904.34       680.41    20_584.75       0.0808          1.3381         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)          19_904.34       773.15    20_677.49       0.2831          1.0954         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)          19_904.34       853.39    20_757.73       0.3938          1.0587         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)         19_904.34       783.10    20_687.44       0.2830          1.0954         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)         19_904.34       859.51    20_763.85       0.3937          1.0587         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)         19_904.34       767.40    20_671.74       0.2830          1.0954         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)         19_904.34       859.06    20_763.40       0.3937          1.0587         5.02
IVF-Binary-512-nl158-random (self)                    19_904.34     2_371.01    22_275.35       0.2976          1.0891         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)          18_985.81       677.42    19_663.24       0.0907          1.3213         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)          18_985.81       683.44    19_669.25       0.0907          1.3213         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)          18_985.81       695.72    19_681.54       0.0907          1.3213         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)         18_985.81       783.55    19_769.36       0.2982          1.0874         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)         18_985.81       866.09    19_851.91       0.4081          1.0543         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)         18_985.81       784.60    19_770.42       0.2982          1.0874         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)         18_985.81       865.75    19_851.56       0.4081          1.0543         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)         18_985.81       794.82    19_780.64       0.2982          1.0874         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)         18_985.81       899.30    19_885.11       0.4081          1.0543         5.21
IVF-Binary-512-nl223-random (self)                    18_985.81     2_396.64    21_382.46       0.3107          1.0827         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)          19_585.26       704.65    20_289.92       0.0961          1.3108         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)          19_585.26       703.41    20_288.67       0.0961          1.3108         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)          19_585.26       696.77    20_282.03       0.0961          1.3108         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)         19_585.26       787.56    20_372.83       0.3052          1.0839         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)         19_585.26       880.37    20_465.63       0.4151          1.0523         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)         19_585.26       788.24    20_373.51       0.3051          1.0839         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)         19_585.26       882.10    20_467.36       0.4149          1.0523         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)         19_585.26       800.68    20_385.95       0.3051          1.0839         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)         19_585.26       882.31    20_467.57       0.4148          1.0523         5.48
IVF-Binary-512-nl316-random (self)                    19_585.26     2_454.58    22_039.84       0.3169          1.0798         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)              20_296.33       684.01    20_980.33       0.2344          4.0218         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)             20_296.33       705.63    21_001.95       0.2283          6.5456         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)             20_296.33       682.00    20_978.33       0.2269          8.6596         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)             20_296.33       792.81    21_089.13       0.6553          1.0603         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)             20_296.33       881.24    21_177.56       0.7965          1.0290         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)            20_296.33       806.07    21_102.40       0.6309          1.0809         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)            20_296.33       906.41    21_202.74       0.7752          1.0352         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)            20_296.33       790.96    21_087.29       0.6241          1.0965         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)            20_296.33       889.72    21_186.04       0.7663          1.0412         5.02
IVF-Binary-512-nl158-pca (self)                       20_296.33     2_472.47    22_768.80       0.6458          1.0828         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)             19_340.82       716.57    20_057.39       0.2299          5.1521         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)             19_340.82       724.94    20_065.76       0.2284          6.9168         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)             19_340.82       687.76    20_028.58       0.2268         10.6677         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)            19_340.82       842.27    20_183.09       0.6368          1.0674         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)            19_340.82       887.87    20_228.69       0.7834          1.0302         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)            19_340.82       798.16    20_138.98       0.6299          1.0769         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)            19_340.82       913.64    20_254.46       0.7747          1.0336         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)            19_340.82       805.45    20_146.27       0.6217          1.0942         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)            19_340.82       914.95    20_255.77       0.7635          1.0405         5.21
IVF-Binary-512-nl223-pca (self)                       19_340.82     2_470.96    21_811.78       0.6446          1.0783         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)             19_897.39       694.32    20_591.71       0.2293          5.2151         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)             19_897.39       700.44    20_597.82       0.2286          6.2822         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)             19_897.39       694.65    20_592.04       0.2268          9.8368         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)            19_897.39       806.53    20_703.92       0.6348          1.0690         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)            19_897.39       922.32    20_819.71       0.7808          1.0307         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)            19_897.39       810.87    20_708.26       0.6317          1.0729         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)            19_897.39       901.85    20_799.24       0.7766          1.0321         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)            19_897.39       806.35    20_703.74       0.6231          1.0907         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)            19_897.39       908.06    20_805.45       0.7655          1.0390         5.48
IVF-Binary-512-nl316-pca (self)                       19_897.39     2_530.75    22_428.14       0.6463          1.0743         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)          37_388.46     1_292.93    38_681.39       0.1225          1.2749         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)         37_388.46     1_297.62    38_686.08       0.1225          1.2749         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)         37_388.46     1_312.66    38_701.12       0.1225          1.2749         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)         37_388.46     1_381.93    38_770.39       0.3374          1.0684         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)         37_388.46     1_471.01    38_859.47       0.4566          1.0413         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)        37_388.46     1_382.36    38_770.82       0.3374          1.0685         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)        37_388.46     1_481.02    38_869.48       0.4566          1.0413         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)        37_388.46     1_395.18    38_783.64       0.3374          1.0685         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)        37_388.46     1_479.06    38_867.53       0.4566          1.0413         9.57
IVF-Binary-1024-nl158-random (self)                   37_388.46     4_496.65    41_885.11       0.3449          1.0714         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)         36_475.97     1_309.84    37_785.80       0.1270          1.2675         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)         36_475.97     1_308.46    37_784.42       0.1270          1.2675         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)         36_475.97     1_307.11    37_783.08       0.1270          1.2675         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)        36_475.97     1_386.71    37_862.67       0.3440          1.0663         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)        36_475.97     1_482.17    37_958.13       0.4630          1.0400         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)        36_475.97     1_390.59    37_866.55       0.3440          1.0663         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)        36_475.97     1_486.65    37_962.61       0.4629          1.0400         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)        36_475.97     1_395.91    37_871.87       0.3440          1.0663         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)        36_475.97     1_491.19    37_967.16       0.4629          1.0400         9.76
IVF-Binary-1024-nl223-random (self)                   36_475.97     4_467.51    40_943.47       0.3511          1.0694         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)         37_060.32     1_400.50    38_460.81       0.1292          1.2641        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)         37_060.32     1_332.40    38_392.72       0.1291          1.2641        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)         37_060.32     1_322.04    38_382.36       0.1291          1.2641        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)        37_060.32     1_402.69    38_463.01       0.3474          1.0652        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)        37_060.32     1_496.77    38_557.09       0.4662          1.0394        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)        37_060.32     1_401.19    38_461.50       0.3473          1.0652        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)        37_060.32     1_495.05    38_555.37       0.4660          1.0395        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)        37_060.32     1_406.32    38_466.63       0.3473          1.0652        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)        37_060.32     1_497.29    38_557.60       0.4659          1.0395        10.04
IVF-Binary-1024-nl316-random (self)                   37_060.32     4_499.91    41_560.23       0.3541          1.0686        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)             37_605.39     1_291.07    38_896.46       0.2750          1.6270         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)            37_605.39     1_299.81    38_905.20       0.2697          1.7363         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)            37_605.39     1_304.95    38_910.34       0.2687          1.7745         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)            37_605.39     1_392.68    38_998.07       0.7304          1.0438         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)            37_605.39     1_484.79    39_090.17       0.8523          1.0203         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)           37_605.39     1_396.30    39_001.69       0.7134          1.0492         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)           37_605.39     1_499.79    39_105.18       0.8403          1.0219         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)           37_605.39     1_396.67    39_002.06       0.7099          1.0511         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)           37_605.39     1_507.83    39_113.22       0.8361          1.0227         9.57
IVF-Binary-1024-nl158-pca (self)                      37_605.39     4_500.85    42_106.24       0.7213          1.0509         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)            36_685.73     1_298.20    37_983.93       0.2707          1.6902         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)            36_685.73     1_298.78    37_984.51       0.2697          1.7395         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)            36_685.73     1_303.33    37_989.06       0.2689          1.7755         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)           36_685.73     1_403.25    38_088.98       0.7170          1.0460         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)           36_685.73     1_493.78    38_179.51       0.8449          1.0205         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)           36_685.73     1_418.18    38_103.91       0.7130          1.0482         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)           36_685.73     1_512.40    38_198.13       0.8402          1.0214         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)           36_685.73     1_407.74    38_093.47       0.7099          1.0503         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)           36_685.73     1_506.97    38_192.70       0.8365          1.0223         9.76
IVF-Binary-1024-nl223-pca (self)                      36_685.73     4_517.48    41_203.21       0.7209          1.0497         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)            37_254.86     1_314.16    38_569.02       0.2705          1.6860        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)            37_254.86     1_332.39    38_587.25       0.2700          1.7156        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)            37_254.86     1_341.62    38_596.48       0.2690          1.7717        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)           37_254.86     1_434.65    38_689.51       0.7154          1.0466        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)           37_254.86     1_506.62    38_761.48       0.8434          1.0206        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)           37_254.86     1_410.36    38_665.22       0.7139          1.0476        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)           37_254.86     1_534.82    38_789.68       0.8414          1.0211        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)           37_254.86     1_416.91    38_671.77       0.7103          1.0500        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)           37_254.86     1_521.63    38_776.49       0.8370          1.0221        10.04
IVF-Binary-1024-nl316-pca (self)                      37_254.86     4_557.52    41_812.38       0.7221          1.0490        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)              2_473.74       582.46     3_056.19       0.0781         16.7559         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)             2_473.74       618.77     3_092.51       0.0770         18.7743         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)             2_473.74       654.76     3_128.50       0.0768         21.4159         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)             2_473.74       638.02     3_111.76       0.7106          1.1278         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)             2_473.74     1_140.70     3_614.44       0.9017          1.0072         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)            2_473.74       668.47     3_142.21       0.5239          3.5417         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)            2_473.74     1_178.10     3_651.83       0.8635          1.0107         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)            2_473.74       683.13     3_156.87       0.4113          4.8456         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)            2_473.74     1_176.10     3_649.84       0.8197          1.0153         5.04
IVF-Binary-768-nl158-sign (self)                       2_473.74     1_943.26     4_417.00       0.5474          3.5670         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)             1_558.09       645.12     2_203.21       0.0862         49.5242         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)             1_558.09       660.27     2_218.36       0.0739         58.1859         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)             1_558.09       683.47     2_241.56       0.0726         74.5487         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)            1_558.09       668.60     2_226.69       0.5502          2.1281         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)            1_558.09     1_186.08     2_744.17       0.8658          1.0102         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)            1_558.09       698.84     2_256.93       0.4427          2.8890         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)            1_558.09     1_156.34     2_714.43       0.8333          1.0132         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)            1_558.09       731.33     2_289.42       0.3235          3.9461         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)            1_558.09     1_227.73     2_785.82       0.7485          1.0213         5.23
IVF-Binary-768-nl223-sign (self)                       1_558.09     2_015.83     3_573.92       0.4734          2.7249         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)             2_114.38       649.38     2_763.75       0.0846         42.9395         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)             2_114.38       663.86     2_778.23       0.0839         55.0976         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)             2_114.38       728.13     2_842.51       0.0715         89.4095         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)            2_114.38       707.41     2_821.79       0.5454          1.5844         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)            2_114.38     1_186.94     3_301.32       0.8546          1.0119         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)            2_114.38       710.99     2_825.37       0.4891          1.9253         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)            2_114.38     1_222.94     3_337.32       0.8352          1.0136         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)            2_114.38       790.69     2_905.07       0.3684          2.5474         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)            2_114.38     1_286.49     3_400.87       0.7643          1.0210         5.51
IVF-Binary-768-nl316-sign (self)                       2_114.38     2_205.57     4_319.95       0.5195          1.7909         5.51
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Cell embeddings

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.61       664.10       696.71       1.0000          1.0000        48.83
Exhaustive (self)                                         32.61     2_187.50     2_220.11       1.0000          1.0000        48.83
ExhaustiveBinary-256-random_no_rr (query)              2_675.73       293.51     2_969.24       0.5519          1.8827         1.78
ExhaustiveBinary-256-random-rf10 (query)               2_675.73       404.31     3_080.03       0.9881          1.0022         1.78
ExhaustiveBinary-256-random-rf20 (query)               2_675.73       511.89     3_187.62       0.9980          1.0003         1.78
ExhaustiveBinary-256-random (self)                     2_675.73     1_310.05     3_985.77       0.9881          1.0022         1.78
ExhaustiveBinary-256-pca_no_rr (query)                 2_757.24       294.13     3_051.37       0.1183         14.1069         1.78
ExhaustiveBinary-256-pca-rf10 (query)                  2_757.24       397.19     3_154.43       0.3215          1.9397         1.78
ExhaustiveBinary-256-pca-rf20 (query)                  2_757.24       509.22     3_266.46       0.4181          1.5883         1.78
ExhaustiveBinary-256-pca (self)                        2_757.24     1_298.56     4_055.80       0.3192          1.9523         1.78
ExhaustiveBinary-512-random_no_rr (query)              5_229.31       455.31     5_684.62       0.6305          1.5768         3.55
ExhaustiveBinary-512-random-rf10 (query)               5_229.31       567.39     5_796.70       0.9975          1.0004         3.55
ExhaustiveBinary-512-random-rf20 (query)               5_229.31       690.73     5_920.04       0.9998          1.0000         3.55
ExhaustiveBinary-512-random (self)                     5_229.31     1_863.90     7_093.21       0.9972          1.0004         3.55
ExhaustiveBinary-512-pca_no_rr (query)                 5_321.55       455.08     5_776.63       0.3665          2.7738         3.55
ExhaustiveBinary-512-pca-rf10 (query)                  5_321.55       609.74     5_931.29       0.8453          1.0657         3.55
ExhaustiveBinary-512-pca-rf20 (query)                  5_321.55       674.56     5_996.10       0.9396          1.0206         3.55
ExhaustiveBinary-512-pca (self)                        5_321.55     1_848.04     7_169.58       0.8325          1.0737         3.55
ExhaustiveBinary-1024-random_no_rr (query)            10_364.11       790.92    11_155.03       0.6758          1.4452         7.10
ExhaustiveBinary-1024-random-rf10 (query)             10_364.11       903.90    11_268.01       0.9995          1.0001         7.10
ExhaustiveBinary-1024-random-rf20 (query)             10_364.11     1_021.75    11_385.86       0.9999          1.0000         7.10
ExhaustiveBinary-1024-random (self)                   10_364.11     2_992.98    13_357.09       0.9993          1.0001         7.10
ExhaustiveBinary-1024-pca_no_rr (query)               10_469.36       784.98    11_254.35       0.5577          1.7682         7.10
ExhaustiveBinary-1024-pca-rf10 (query)                10_469.36       918.79    11_388.15       0.9880          1.0028         7.10
ExhaustiveBinary-1024-pca-rf20 (query)                10_469.36     1_015.57    11_484.94       0.9987          1.0003         7.10
ExhaustiveBinary-1024-pca (self)                      10_469.36     2_975.83    13_445.20       0.9861          1.0033         7.10
ExhaustiveBinary-256-sign_no_rr (query)                   58.79       489.29       548.07       0.0376         19.4742         1.53
ExhaustiveBinary-256-sign-rf10 (query)                    58.79       498.68       557.47       0.1617          2.7567         1.53
ExhaustiveBinary-256-sign-rf20 (query)                    58.79       863.56       922.34       0.2739          1.9837         1.53
ExhaustiveBinary-256-sign (self)                          58.79     1_755.14     1_813.93       0.1691          2.7353         1.53
IVF-Binary-256-nl158-np7-rf0-random (query)            3_748.56       128.72     3_877.28       0.5644          1.6744         1.93
IVF-Binary-256-nl158-np12-rf0-random (query)           3_748.56       139.79     3_888.35       0.5580          1.7409         1.93
IVF-Binary-256-nl158-np17-rf0-random (query)           3_748.56       148.39     3_896.95       0.5562          1.7768         1.93
IVF-Binary-256-nl158-np7-rf10-random (query)           3_748.56       194.05     3_942.61       0.9901          1.0018         1.93
IVF-Binary-256-nl158-np7-rf20-random (query)           3_748.56       247.52     3_996.09       0.9967          1.0006         1.93
IVF-Binary-256-nl158-np12-rf10-random (query)          3_748.56       201.86     3_950.43       0.9903          1.0017         1.93
IVF-Binary-256-nl158-np12-rf20-random (query)          3_748.56       264.31     4_012.88       0.9985          1.0002         1.93
IVF-Binary-256-nl158-np17-rf10-random (query)          3_748.56       215.01     3_963.57       0.9894          1.0019         1.93
IVF-Binary-256-nl158-np17-rf20-random (query)          3_748.56       274.48     4_023.04       0.9983          1.0002         1.93
IVF-Binary-256-nl158-random (self)                     3_748.56       602.89     4_351.45       0.9902          1.0017         1.93
IVF-Binary-256-nl223-np11-rf0-random (query)           3_082.21       130.53     3_212.74       0.5623          1.6901         2.00
IVF-Binary-256-nl223-np14-rf0-random (query)           3_082.21       138.54     3_220.74       0.5598          1.7203         2.00
IVF-Binary-256-nl223-np21-rf0-random (query)           3_082.21       146.22     3_228.42       0.5573          1.7614         2.00
IVF-Binary-256-nl223-np11-rf10-random (query)          3_082.21       193.41     3_275.62       0.9910          1.0015         2.00
IVF-Binary-256-nl223-np11-rf20-random (query)          3_082.21       250.91     3_333.12       0.9983          1.0002         2.00
IVF-Binary-256-nl223-np14-rf10-random (query)          3_082.21       196.92     3_279.13       0.9906          1.0016         2.00
IVF-Binary-256-nl223-np14-rf20-random (query)          3_082.21       274.56     3_356.77       0.9986          1.0002         2.00
IVF-Binary-256-nl223-np21-rf10-random (query)          3_082.21       210.73     3_292.93       0.9896          1.0018         2.00
IVF-Binary-256-nl223-np21-rf20-random (query)          3_082.21       261.29     3_343.50       0.9984          1.0002         2.00
IVF-Binary-256-nl223-random (self)                     3_082.21       570.26     3_652.47       0.9904          1.0017         2.00
IVF-Binary-256-nl316-np15-rf0-random (query)           3_229.33       138.07     3_367.40       0.5622          1.6843         2.09
IVF-Binary-256-nl316-np17-rf0-random (query)           3_229.33       139.37     3_368.70       0.5610          1.6983         2.09
IVF-Binary-256-nl316-np25-rf0-random (query)           3_229.33       144.99     3_374.32       0.5582          1.7399         2.09
IVF-Binary-256-nl316-np15-rf10-random (query)          3_229.33       195.24     3_424.57       0.9911          1.0015         2.09
IVF-Binary-256-nl316-np15-rf20-random (query)          3_229.33       247.72     3_477.05       0.9986          1.0002         2.09
IVF-Binary-256-nl316-np17-rf10-random (query)          3_229.33       195.28     3_424.61       0.9908          1.0016         2.09
IVF-Binary-256-nl316-np17-rf20-random (query)          3_229.33       250.63     3_479.96       0.9986          1.0002         2.09
IVF-Binary-256-nl316-np25-rf10-random (query)          3_229.33       205.61     3_434.94       0.9899          1.0018         2.09
IVF-Binary-256-nl316-np25-rf20-random (query)          3_229.33       259.64     3_488.97       0.9985          1.0002         2.09
IVF-Binary-256-nl316-random (self)                     3_229.33       567.51     3_796.84       0.9908          1.0016         2.09
IVF-Binary-256-nl158-np7-rf0-pca (query)               3_776.39       128.21     3_904.60       0.1527          5.1206         1.93
IVF-Binary-256-nl158-np12-rf0-pca (query)              3_776.39       140.62     3_917.01       0.1376          6.0959         1.93
IVF-Binary-256-nl158-np17-rf0-pca (query)              3_776.39       148.15     3_924.55       0.1312          6.8304         1.93
IVF-Binary-256-nl158-np7-rf10-pca (query)              3_776.39       199.60     3_975.99       0.4905          1.4114         1.93
IVF-Binary-256-nl158-np7-rf20-pca (query)              3_776.39       258.11     4_034.50       0.6392          1.2170         1.93
IVF-Binary-256-nl158-np12-rf10-pca (query)             3_776.39       211.22     3_987.62       0.4274          1.5364         1.93
IVF-Binary-256-nl158-np12-rf20-pca (query)             3_776.39       280.74     4_057.13       0.5675          1.2973         1.93
IVF-Binary-256-nl158-np17-rf10-pca (query)             3_776.39       225.85     4_002.24       0.3950          1.6191         1.93
IVF-Binary-256-nl158-np17-rf20-pca (query)             3_776.39       293.72     4_070.12       0.5273          1.3518         1.93
IVF-Binary-256-nl158-pca (self)                        3_776.39       647.96     4_424.35       0.4248          1.5444         1.93
IVF-Binary-256-nl223-np11-rf0-pca (query)              3_190.50       131.42     3_321.92       0.1485          5.1295         2.00
IVF-Binary-256-nl223-np14-rf0-pca (query)              3_190.50       138.08     3_328.57       0.1425          5.5191         2.00
IVF-Binary-256-nl223-np21-rf0-pca (query)              3_190.50       144.03     3_334.53       0.1343          6.2989         2.00
IVF-Binary-256-nl223-np11-rf10-pca (query)             3_190.50       200.29     3_390.79       0.4802          1.4241         2.00
IVF-Binary-256-nl223-np11-rf20-pca (query)             3_190.50       257.10     3_447.60       0.6314          1.2222         2.00
IVF-Binary-256-nl223-np14-rf10-pca (query)             3_190.50       206.32     3_396.82       0.4522          1.4777         2.00
IVF-Binary-256-nl223-np14-rf20-pca (query)             3_190.50       265.39     3_455.89       0.6001          1.2555         2.00
IVF-Binary-256-nl223-np21-rf10-pca (query)             3_190.50       214.10     3_404.60       0.4119          1.5715         2.00
IVF-Binary-256-nl223-np21-rf20-pca (query)             3_190.50       280.13     3_470.63       0.5507          1.3172         2.00
IVF-Binary-256-nl223-pca (self)                        3_190.50       614.50     3_805.00       0.4498          1.4839         2.00
IVF-Binary-256-nl316-np15-rf0-pca (query)              3_335.30       136.65     3_471.96       0.1484          5.0403         2.09
IVF-Binary-256-nl316-np17-rf0-pca (query)              3_335.30       138.74     3_474.04       0.1450          5.2295         2.09
IVF-Binary-256-nl316-np25-rf0-pca (query)              3_335.30       147.39     3_482.70       0.1368          5.9003         2.09
IVF-Binary-256-nl316-np15-rf10-pca (query)             3_335.30       204.39     3_539.69       0.4799          1.4228         2.09
IVF-Binary-256-nl316-np15-rf20-pca (query)             3_335.30       261.49     3_596.79       0.6328          1.2200         2.09
IVF-Binary-256-nl316-np17-rf10-pca (query)             3_335.30       204.38     3_539.68       0.4654          1.4492         2.09
IVF-Binary-256-nl316-np17-rf20-pca (query)             3_335.30       267.45     3_602.75       0.6163          1.2367         2.09
IVF-Binary-256-nl316-np25-rf10-pca (query)             3_335.30       213.82     3_549.12       0.4256          1.5342         2.09
IVF-Binary-256-nl316-np25-rf20-pca (query)             3_335.30       276.86     3_612.16       0.5689          1.2913         2.09
IVF-Binary-256-nl316-pca (self)                        3_335.30       651.15     3_986.45       0.4629          1.4549         2.09
IVF-Binary-512-nl158-np7-rf0-random (query)            6_260.64       244.16     6_504.80       0.6403          1.4493         3.71
IVF-Binary-512-nl158-np12-rf0-random (query)           6_260.64       236.02     6_496.66       0.6346          1.4947         3.71
IVF-Binary-512-nl158-np17-rf0-random (query)           6_260.64       249.45     6_510.09       0.6329          1.5212         3.71
IVF-Binary-512-nl158-np7-rf10-random (query)           6_260.64       290.75     6_551.39       0.9963          1.0007         3.71
IVF-Binary-512-nl158-np7-rf20-random (query)           6_260.64       342.89     6_603.53       0.9976          1.0005         3.71
IVF-Binary-512-nl158-np12-rf10-random (query)          6_260.64       304.57     6_565.21       0.9980          1.0003         3.71
IVF-Binary-512-nl158-np12-rf20-random (query)          6_260.64       358.52     6_619.16       0.9998          1.0000         3.71
IVF-Binary-512-nl158-np17-rf10-random (query)          6_260.64       313.75     6_574.39       0.9978          1.0003         3.71
IVF-Binary-512-nl158-np17-rf20-random (query)          6_260.64       371.88     6_632.52       0.9998          1.0000         3.71
IVF-Binary-512-nl158-random (self)                     6_260.64       934.55     7_195.19       0.9978          1.0003         3.71
IVF-Binary-512-nl223-np11-rf0-random (query)           5_652.80       226.22     5_879.02       0.6379          1.4649         3.77
IVF-Binary-512-nl223-np14-rf0-random (query)           5_652.80       279.77     5_932.58       0.6357          1.4848         3.77
IVF-Binary-512-nl223-np21-rf0-random (query)           5_652.80       288.73     5_941.53       0.6336          1.5131         3.77
IVF-Binary-512-nl223-np11-rf10-random (query)          5_652.80       347.39     6_000.19       0.9976          1.0003         3.77
IVF-Binary-512-nl223-np11-rf20-random (query)          5_652.80       353.06     6_005.86       0.9993          1.0001         3.77
IVF-Binary-512-nl223-np14-rf10-random (query)          5_652.80       304.32     5_957.12       0.9979          1.0003         3.77
IVF-Binary-512-nl223-np14-rf20-random (query)          5_652.80       418.33     6_071.13       0.9997          1.0000         3.77
IVF-Binary-512-nl223-np21-rf10-random (query)          5_652.80       375.65     6_028.46       0.9978          1.0003         3.77
IVF-Binary-512-nl223-np21-rf20-random (query)          5_652.80       378.30     6_031.11       0.9998          1.0000         3.77
IVF-Binary-512-nl223-random (self)                     5_652.80       896.20     6_549.01       0.9977          1.0003         3.77
IVF-Binary-512-nl316-np15-rf0-random (query)           5_819.52       232.96     6_052.47       0.6377          1.4619         3.86
IVF-Binary-512-nl316-np17-rf0-random (query)           5_819.52       237.05     6_056.57       0.6368          1.4710         3.86
IVF-Binary-512-nl316-np25-rf0-random (query)           5_819.52       246.88     6_066.39       0.6344          1.4973         3.86
IVF-Binary-512-nl316-np15-rf10-random (query)          5_819.52       310.35     6_129.87       0.9979          1.0003         3.86
IVF-Binary-512-nl316-np15-rf20-random (query)          5_819.52       344.81     6_164.33       0.9996          1.0001         3.86
IVF-Binary-512-nl316-np17-rf10-random (query)          5_819.52       300.60     6_120.12       0.9980          1.0003         3.86
IVF-Binary-512-nl316-np17-rf20-random (query)          5_819.52       345.36     6_164.87       0.9997          1.0000         3.86
IVF-Binary-512-nl316-np25-rf10-random (query)          5_819.52       306.38     6_125.89       0.9978          1.0003         3.86
IVF-Binary-512-nl316-np25-rf20-random (query)          5_819.52       358.17     6_177.69       0.9998          1.0000         3.86
IVF-Binary-512-nl316-random (self)                     5_819.52       883.38     6_702.90       0.9978          1.0003         3.86
IVF-Binary-512-nl158-np7-rf0-pca (query)               6_334.58       224.38     6_558.96       0.3819          2.2246         3.71
IVF-Binary-512-nl158-np12-rf0-pca (query)              6_334.58       237.05     6_571.63       0.3733          2.3651         3.71
IVF-Binary-512-nl158-np17-rf0-pca (query)              6_334.58       247.67     6_582.25       0.3705          2.4596         3.71
IVF-Binary-512-nl158-np7-rf10-pca (query)              6_334.58       291.49     6_626.07       0.8865          1.0420         3.71
IVF-Binary-512-nl158-np7-rf20-pca (query)              6_334.58       347.79     6_682.37       0.9653          1.0104         3.71
IVF-Binary-512-nl158-np12-rf10-pca (query)             6_334.58       304.48     6_639.06       0.8656          1.0534         3.71
IVF-Binary-512-nl158-np12-rf20-pca (query)             6_334.58       373.23     6_707.81       0.9542          1.0147         3.71
IVF-Binary-512-nl158-np17-rf10-pca (query)             6_334.58       325.05     6_659.63       0.8565          1.0588         3.71
IVF-Binary-512-nl158-np17-rf20-pca (query)             6_334.58       377.56     6_712.14       0.9480          1.0171         3.71
IVF-Binary-512-nl158-pca (self)                        6_334.58       943.69     7_278.26       0.8541          1.0600         3.71
IVF-Binary-512-nl223-np11-rf0-pca (query)              5_748.88       227.20     5_976.08       0.3781          2.2518         3.77
IVF-Binary-512-nl223-np14-rf0-pca (query)              5_748.88       245.84     5_994.72       0.3748          2.3105         3.77
IVF-Binary-512-nl223-np21-rf0-pca (query)              5_748.88       245.91     5_994.79       0.3714          2.4105         3.77
IVF-Binary-512-nl223-np11-rf10-pca (query)             5_748.88       295.07     6_043.95       0.8810          1.0451         3.77
IVF-Binary-512-nl223-np11-rf20-pca (query)             5_748.88       348.94     6_097.82       0.9631          1.0112         3.77
IVF-Binary-512-nl223-np14-rf10-pca (query)             5_748.88       293.46     6_042.35       0.8711          1.0504         3.77
IVF-Binary-512-nl223-np14-rf20-pca (query)             5_748.88       353.84     6_102.73       0.9578          1.0132         3.77
IVF-Binary-512-nl223-np21-rf10-pca (query)             5_748.88       305.46     6_054.34       0.8591          1.0574         3.77
IVF-Binary-512-nl223-np21-rf20-pca (query)             5_748.88       378.19     6_127.08       0.9499          1.0164         3.77
IVF-Binary-512-nl223-pca (self)                        5_748.88       924.14     6_673.03       0.8602          1.0566         3.77
IVF-Binary-512-nl316-np15-rf0-pca (query)              5_896.76       230.81     6_127.56       0.3779          2.2494         3.86
IVF-Binary-512-nl316-np17-rf0-pca (query)              5_896.76       236.43     6_133.19       0.3763          2.2755         3.86
IVF-Binary-512-nl316-np25-rf0-pca (query)              5_896.76       241.89     6_138.64       0.3724          2.3637         3.86
IVF-Binary-512-nl316-np15-rf10-pca (query)             5_896.76       292.29     6_189.05       0.8796          1.0459         3.86
IVF-Binary-512-nl316-np15-rf20-pca (query)             5_896.76       356.12     6_252.88       0.9626          1.0116         3.86
IVF-Binary-512-nl316-np17-rf10-pca (query)             5_896.76       299.19     6_195.95       0.8749          1.0483         3.86
IVF-Binary-512-nl316-np17-rf20-pca (query)             5_896.76       353.92     6_250.68       0.9602          1.0125         3.86
IVF-Binary-512-nl316-np25-rf10-pca (query)             5_896.76       302.82     6_199.58       0.8626          1.0551         3.86
IVF-Binary-512-nl316-np25-rf20-pca (query)             5_896.76       363.29     6_260.04       0.9524          1.0154         3.86
IVF-Binary-512-nl316-pca (self)                        5_896.76       906.32     6_803.08       0.8641          1.0545         3.86
IVF-Binary-1024-nl158-np7-rf0-random (query)          11_393.34       419.50    11_812.84       0.6843          1.3554         7.26
IVF-Binary-1024-nl158-np12-rf0-random (query)         11_393.34       441.79    11_835.13       0.6791          1.3904         7.26
IVF-Binary-1024-nl158-np17-rf0-random (query)         11_393.34       469.12    11_862.45       0.6775          1.4122         7.26
IVF-Binary-1024-nl158-np7-rf10-random (query)         11_393.34       487.39    11_880.73       0.9974          1.0005         7.26
IVF-Binary-1024-nl158-np7-rf20-random (query)         11_393.34       539.33    11_932.67       0.9976          1.0005         7.26
IVF-Binary-1024-nl158-np12-rf10-random (query)        11_393.34       510.97    11_904.31       0.9996          1.0000         7.26
IVF-Binary-1024-nl158-np12-rf20-random (query)        11_393.34       579.35    11_972.68       0.9999          1.0000         7.26
IVF-Binary-1024-nl158-np17-rf10-random (query)        11_393.34       535.77    11_929.11       0.9996          1.0000         7.26
IVF-Binary-1024-nl158-np17-rf20-random (query)        11_393.34       589.64    11_982.98       0.9999          1.0000         7.26
IVF-Binary-1024-nl158-random (self)                   11_393.34     1_616.90    13_010.24       0.9994          1.0001         7.26
IVF-Binary-1024-nl223-np11-rf0-random (query)         10_792.28       431.46    11_223.73       0.6818          1.3682         7.32
IVF-Binary-1024-nl223-np14-rf0-random (query)         10_792.28       437.87    11_230.14       0.6799          1.3836         7.32
IVF-Binary-1024-nl223-np21-rf0-random (query)         10_792.28       445.05    11_237.32       0.6781          1.4037         7.32
IVF-Binary-1024-nl223-np11-rf10-random (query)        10_792.28       485.13    11_277.41       0.9990          1.0001         7.32
IVF-Binary-1024-nl223-np11-rf20-random (query)        10_792.28       547.23    11_339.51       0.9993          1.0001         7.32
IVF-Binary-1024-nl223-np14-rf10-random (query)        10_792.28       488.06    11_280.33       0.9995          1.0001         7.32
IVF-Binary-1024-nl223-np14-rf20-random (query)        10_792.28       543.98    11_336.26       0.9998          1.0000         7.32
IVF-Binary-1024-nl223-np21-rf10-random (query)        10_792.28       506.57    11_298.85       0.9995          1.0001         7.32
IVF-Binary-1024-nl223-np21-rf20-random (query)        10_792.28       564.80    11_357.08       0.9999          1.0000         7.32
IVF-Binary-1024-nl223-random (self)                   10_792.28     1_544.02    12_336.30       0.9993          1.0001         7.32
IVF-Binary-1024-nl316-np15-rf0-random (query)         10_940.48       439.59    11_380.07       0.6814          1.3673         7.42
IVF-Binary-1024-nl316-np17-rf0-random (query)         10_940.48       448.03    11_388.50       0.6805          1.3746         7.42
IVF-Binary-1024-nl316-np25-rf0-random (query)         10_940.48       465.52    11_405.99       0.6785          1.3952         7.42
IVF-Binary-1024-nl316-np15-rf10-random (query)        10_940.48       485.45    11_425.92       0.9993          1.0001         7.42
IVF-Binary-1024-nl316-np15-rf20-random (query)        10_940.48       537.79    11_478.27       0.9997          1.0001         7.42
IVF-Binary-1024-nl316-np17-rf10-random (query)        10_940.48       488.54    11_429.02       0.9994          1.0001         7.42
IVF-Binary-1024-nl316-np17-rf20-random (query)        10_940.48       548.29    11_488.77       0.9998          1.0000         7.42
IVF-Binary-1024-nl316-np25-rf10-random (query)        10_940.48       508.91    11_449.38       0.9995          1.0001         7.42
IVF-Binary-1024-nl316-np25-rf20-random (query)        10_940.48       565.41    11_505.89       0.9999          1.0000         7.42
IVF-Binary-1024-nl316-random (self)                   10_940.48     1_531.40    12_471.87       0.9993          1.0001         7.42
IVF-Binary-1024-nl158-np7-rf0-pca (query)             11_464.14       418.60    11_882.75       0.5686          1.5695         7.26
IVF-Binary-1024-nl158-np12-rf0-pca (query)            11_464.14       439.74    11_903.89       0.5623          1.6313         7.26
IVF-Binary-1024-nl158-np17-rf0-pca (query)            11_464.14       458.23    11_922.38       0.5605          1.6705         7.26
IVF-Binary-1024-nl158-np7-rf10-pca (query)            11_464.14       485.47    11_949.61       0.9912          1.0019         7.26
IVF-Binary-1024-nl158-np7-rf20-pca (query)            11_464.14       536.65    12_000.80       0.9972          1.0006         7.26
IVF-Binary-1024-nl158-np12-rf10-pca (query)           11_464.14       514.62    11_978.77       0.9907          1.0020         7.26
IVF-Binary-1024-nl158-np12-rf20-pca (query)           11_464.14       562.07    12_026.22       0.9992          1.0002         7.26
IVF-Binary-1024-nl158-np17-rf10-pca (query)           11_464.14       521.55    11_985.69       0.9895          1.0023         7.26
IVF-Binary-1024-nl158-np17-rf20-pca (query)           11_464.14       587.11    12_051.25       0.9990          1.0002         7.26
IVF-Binary-1024-nl158-pca (self)                      11_464.14     1_607.23    13_071.37       0.9889          1.0025         7.26
IVF-Binary-1024-nl223-np11-rf0-pca (query)            10_884.48       429.91    11_314.40       0.5654          1.5890         7.32
IVF-Binary-1024-nl223-np14-rf0-pca (query)            10_884.48       435.92    11_320.40       0.5631          1.6138         7.32
IVF-Binary-1024-nl223-np21-rf0-pca (query)            10_884.48       454.33    11_338.81       0.5608          1.6523         7.32
IVF-Binary-1024-nl223-np11-rf10-pca (query)           10_884.48       486.39    11_370.87       0.9916          1.0017         7.32
IVF-Binary-1024-nl223-np11-rf20-pca (query)           10_884.48       537.25    11_421.74       0.9987          1.0002         7.32
IVF-Binary-1024-nl223-np14-rf10-pca (query)           10_884.48       502.92    11_387.41       0.9909          1.0019         7.32
IVF-Binary-1024-nl223-np14-rf20-pca (query)           10_884.48       549.40    11_433.88       0.9990          1.0002         7.32
IVF-Binary-1024-nl223-np21-rf10-pca (query)           10_884.48       507.33    11_391.82       0.9897          1.0023         7.32
IVF-Binary-1024-nl223-np21-rf20-pca (query)           10_884.48       568.48    11_452.96       0.9990          1.0002         7.32
IVF-Binary-1024-nl223-pca (self)                      10_884.48     1_583.78    12_468.27       0.9892          1.0024         7.32
IVF-Binary-1024-nl316-np15-rf0-pca (query)            11_032.59       425.93    11_458.52       0.5650          1.5883         7.42
IVF-Binary-1024-nl316-np17-rf0-pca (query)            11_032.59       428.83    11_461.42       0.5640          1.5992         7.42
IVF-Binary-1024-nl316-np25-rf0-pca (query)            11_032.59       447.57    11_480.16       0.5616          1.6346         7.42
IVF-Binary-1024-nl316-np15-rf10-pca (query)           11_032.59       490.10    11_522.69       0.9917          1.0017         7.42
IVF-Binary-1024-nl316-np15-rf20-pca (query)           11_032.59       540.69    11_573.28       0.9990          1.0002         7.42
IVF-Binary-1024-nl316-np17-rf10-pca (query)           11_032.59       493.29    11_525.88       0.9914          1.0018         7.42
IVF-Binary-1024-nl316-np17-rf20-pca (query)           11_032.59       547.48    11_580.07       0.9991          1.0002         7.42
IVF-Binary-1024-nl316-np25-rf10-pca (query)           11_032.59       509.59    11_542.18       0.9902          1.0022         7.42
IVF-Binary-1024-nl316-np25-rf20-pca (query)           11_032.59       562.30    11_594.89       0.9991          1.0002         7.42
IVF-Binary-1024-nl316-pca (self)                      11_032.59     1_539.57    12_572.16       0.9896          1.0023         7.42
IVF-Binary-256-nl158-np7-rf0-sign (query)              1_104.66       291.54     1_396.21       0.3698          2.2923         1.68
IVF-Binary-256-nl158-np12-rf0-sign (query)             1_104.66       323.37     1_428.03       0.3462          2.4709         1.68
IVF-Binary-256-nl158-np17-rf0-sign (query)             1_104.66       348.12     1_452.79       0.3312          2.6495         1.68
IVF-Binary-256-nl158-np7-rf10-sign (query)             1_104.66       328.90     1_433.56       0.7370          1.1542         1.68
IVF-Binary-256-nl158-np7-rf20-sign (query)             1_104.66       567.05     1_671.71       0.9127          1.0437         1.68
IVF-Binary-256-nl158-np12-rf10-sign (query)            1_104.66       365.95     1_470.62       0.6111          1.2797         1.68
IVF-Binary-256-nl158-np12-rf20-sign (query)            1_104.66       623.80     1_728.47       0.8375          1.0872         1.68
IVF-Binary-256-nl158-np17-rf10-sign (query)            1_104.66       382.30     1_486.96       0.5503          1.3843         1.68
IVF-Binary-256-nl158-np17-rf20-sign (query)            1_104.66       652.46     1_757.12       0.7858          1.1267         1.68
IVF-Binary-256-nl158-sign (self)                       1_104.66     1_126.85     2_231.51       0.6110          1.2817         1.68
IVF-Binary-256-nl223-np11-rf0-sign (query)               511.68       302.72       814.41       0.3263          2.5881         1.75
IVF-Binary-256-nl223-np14-rf0-sign (query)               511.68       318.17       829.85       0.3166          2.6847         1.75
IVF-Binary-256-nl223-np21-rf0-sign (query)               511.68       349.09       860.77       0.2971          2.9458         1.75
IVF-Binary-256-nl223-np11-rf10-sign (query)              511.68       336.77       848.45       0.6563          1.2510         1.75
IVF-Binary-256-nl223-np11-rf20-sign (query)              511.68       585.81     1_097.49       0.8243          1.1194         1.75
IVF-Binary-256-nl223-np14-rf10-sign (query)              511.68       356.59       868.27       0.6037          1.3094         1.75
IVF-Binary-256-nl223-np14-rf20-sign (query)              511.68       598.85     1_110.53       0.7917          1.1453         1.75
IVF-Binary-256-nl223-np21-rf10-sign (query)              511.68       385.08       896.77       0.5241          1.4384         1.75
IVF-Binary-256-nl223-np21-rf20-sign (query)              511.68       656.03     1_167.72       0.7306          1.2030         1.75
IVF-Binary-256-nl223-sign (self)                         511.68     1_077.12     1_588.81       0.6051          1.3104         1.75
IVF-Binary-256-nl316-np15-rf0-sign (query)               648.62       317.12       965.74       0.2930          2.8391         1.84
IVF-Binary-256-nl316-np17-rf0-sign (query)               648.62       325.16       973.78       0.2880          2.9065         1.84
IVF-Binary-256-nl316-np25-rf0-sign (query)               648.62       356.91     1_005.53       0.2711          3.1733         1.84
IVF-Binary-256-nl316-np15-rf10-sign (query)              648.62       358.19     1_006.81       0.6133          1.3217         1.84
IVF-Binary-256-nl316-np15-rf20-sign (query)              648.62       592.98     1_241.60       0.7590          1.1822         1.84
IVF-Binary-256-nl316-np17-rf10-sign (query)              648.62       360.64     1_009.26       0.5879          1.3579         1.84
IVF-Binary-256-nl316-np17-rf20-sign (query)              648.62       603.33     1_251.95       0.7428          1.1998         1.84
IVF-Binary-256-nl316-np25-rf10-sign (query)              648.62       393.51     1_042.13       0.5132          1.4900         1.84
IVF-Binary-256-nl316-np25-rf20-sign (query)              648.62       673.07     1_321.69       0.6920          1.2620         1.84
IVF-Binary-256-nl316-sign (self)                         648.62     1_108.53     1_757.15       0.5883          1.3553         1.84
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        69.72     1_247.19     1_316.91       1.0000          1.0000        97.66
Exhaustive (self)                                         69.72     4_100.34     4_170.06       1.0000          1.0000        97.66
ExhaustiveBinary-256-random_no_rr (query)              5_853.77       444.62     6_298.39       0.5547          1.7646         2.03
ExhaustiveBinary-256-random-rf10 (query)               5_853.77       564.71     6_418.47       0.9898          1.0017         2.03
ExhaustiveBinary-256-random-rf20 (query)               5_853.77       703.12     6_556.89       0.9984          1.0002         2.03
ExhaustiveBinary-256-random (self)                     5_853.77     1_818.25     7_672.01       0.9899          1.0016         2.03
ExhaustiveBinary-256-pca_no_rr (query)                 6_061.64       430.89     6_492.53       0.1212         13.1779         2.03
ExhaustiveBinary-256-pca-rf10 (query)                  6_061.64       555.53     6_617.17       0.3406          1.8751         2.03
ExhaustiveBinary-256-pca-rf20 (query)                  6_061.64       683.01     6_744.65       0.4406          1.5429         2.03
ExhaustiveBinary-256-pca (self)                        6_061.64     1_784.38     7_846.01       0.3366          1.8907         2.03
ExhaustiveBinary-512-random_no_rr (query)             11_454.76       687.72    12_142.48       0.6013          1.6760         4.05
ExhaustiveBinary-512-random-rf10 (query)              11_454.76       820.65    12_275.41       0.9977          1.0003         4.05
ExhaustiveBinary-512-random-rf20 (query)              11_454.76       952.73    12_407.49       0.9997          1.0000         4.05
ExhaustiveBinary-512-random (self)                    11_454.76     2_822.23    14_276.99       0.9975          1.0003         4.05
ExhaustiveBinary-512-pca_no_rr (query)                11_663.99       687.90    12_351.89       0.1147         15.9331         4.05
ExhaustiveBinary-512-pca-rf10 (query)                 11_663.99       822.12    12_486.11       0.2782          2.2254         4.05
ExhaustiveBinary-512-pca-rf20 (query)                 11_663.99       936.84    12_600.83       0.3475          1.8252         4.05
ExhaustiveBinary-512-pca (self)                       11_663.99     2_638.92    14_302.92       0.2742          2.2528         4.05
ExhaustiveBinary-1024-random_no_rr (query)            22_686.82     1_233.31    23_920.13       0.6624          1.4553         8.11
ExhaustiveBinary-1024-random-rf10 (query)             22_686.82     1_376.66    24_063.47       0.9994          1.0001         8.11
ExhaustiveBinary-1024-random-rf20 (query)             22_686.82     1_528.22    24_215.04       0.9999          1.0000         8.11
ExhaustiveBinary-1024-random (self)                   22_686.82     4_533.45    27_220.27       0.9994          1.0001         8.11
ExhaustiveBinary-1024-pca_no_rr (query)               23_048.17     1_244.10    24_292.27       0.3939          2.4382         8.11
ExhaustiveBinary-1024-pca-rf10 (query)                23_048.17     1_371.13    24_419.30       0.8322          1.0743         8.11
ExhaustiveBinary-1024-pca-rf20 (query)                23_048.17     1_510.30    24_558.47       0.9198          1.0285         8.11
ExhaustiveBinary-1024-pca (self)                      23_048.17     4_517.69    27_565.86       0.8160          1.0854         8.11
ExhaustiveBinary-512-sign_no_rr (query)                  130.69       665.32       796.01       0.0400         18.1509         3.05
ExhaustiveBinary-512-sign-rf10 (query)                   130.69       709.09       839.77       0.1821          2.5572         3.05
ExhaustiveBinary-512-sign-rf20 (query)                   130.69     1_150.87     1_281.56       0.3140          1.8428         3.05
ExhaustiveBinary-512-sign (self)                         130.69     2_300.20     2_430.89       0.1897          2.5286         3.05
IVF-Binary-256-nl158-np7-rf0-random (query)            7_823.17       254.85     8_078.02       0.5635          1.6351         2.34
IVF-Binary-256-nl158-np12-rf0-random (query)           7_823.17       263.95     8_087.11       0.5601          1.6665         2.34
IVF-Binary-256-nl158-np17-rf0-random (query)           7_823.17       270.93     8_094.10       0.5586          1.6897         2.34
IVF-Binary-256-nl158-np7-rf10-random (query)           7_823.17       341.12     8_164.29       0.9918          1.0013         2.34
IVF-Binary-256-nl158-np7-rf20-random (query)           7_823.17       416.37     8_239.54       0.9978          1.0003         2.34
IVF-Binary-256-nl158-np12-rf10-random (query)          7_823.17       343.69     8_166.86       0.9916          1.0013         2.34
IVF-Binary-256-nl158-np12-rf20-random (query)          7_823.17       426.21     8_249.38       0.9988          1.0001         2.34
IVF-Binary-256-nl158-np17-rf10-random (query)          7_823.17       355.06     8_178.23       0.9909          1.0014         2.34
IVF-Binary-256-nl158-np17-rf20-random (query)          7_823.17       436.05     8_259.21       0.9987          1.0001         2.34
IVF-Binary-256-nl158-random (self)                     7_823.17     1_001.39     8_824.56       0.9918          1.0012         2.34
IVF-Binary-256-nl223-np11-rf0-random (query)           6_528.80       254.52     6_783.32       0.5616          1.6428         2.47
IVF-Binary-256-nl223-np14-rf0-random (query)           6_528.80       258.48     6_787.28       0.5604          1.6550         2.47
IVF-Binary-256-nl223-np21-rf0-random (query)           6_528.80       274.86     6_803.66       0.5589          1.6783         2.47
IVF-Binary-256-nl223-np11-rf10-random (query)          6_528.80       338.47     6_867.27       0.9922          1.0012         2.47
IVF-Binary-256-nl223-np11-rf20-random (query)          6_528.80       416.36     6_945.16       0.9989          1.0001         2.47
IVF-Binary-256-nl223-np14-rf10-random (query)          6_528.80       342.27     6_871.07       0.9919          1.0012         2.47
IVF-Binary-256-nl223-np14-rf20-random (query)          6_528.80       420.50     6_949.29       0.9989          1.0001         2.47
IVF-Binary-256-nl223-np21-rf10-random (query)          6_528.80       346.92     6_875.72       0.9911          1.0014         2.47
IVF-Binary-256-nl223-np21-rf20-random (query)          6_528.80       432.96     6_961.76       0.9988          1.0001         2.47
IVF-Binary-256-nl223-random (self)                     6_528.80       970.46     7_499.26       0.9920          1.0012         2.47
IVF-Binary-256-nl316-np15-rf0-random (query)           6_822.87       263.69     7_086.56       0.5613          1.6459         2.65
IVF-Binary-256-nl316-np17-rf0-random (query)           6_822.87       266.32     7_089.19       0.5607          1.6528         2.65
IVF-Binary-256-nl316-np25-rf0-random (query)           6_822.87       270.75     7_093.62       0.5595          1.6692         2.65
IVF-Binary-256-nl316-np15-rf10-random (query)          6_822.87       345.20     7_168.07       0.9922          1.0012         2.65
IVF-Binary-256-nl316-np15-rf20-random (query)          6_822.87       423.78     7_246.65       0.9990          1.0001         2.65
IVF-Binary-256-nl316-np17-rf10-random (query)          6_822.87       347.51     7_170.39       0.9919          1.0012         2.65
IVF-Binary-256-nl316-np17-rf20-random (query)          6_822.87       428.88     7_251.75       0.9990          1.0001         2.65
IVF-Binary-256-nl316-np25-rf10-random (query)          6_822.87       352.60     7_175.47       0.9913          1.0014         2.65
IVF-Binary-256-nl316-np25-rf20-random (query)          6_822.87       434.56     7_257.43       0.9989          1.0001         2.65
IVF-Binary-256-nl316-random (self)                     6_822.87       983.57     7_806.44       0.9922          1.0012         2.65
IVF-Binary-256-nl158-np7-rf0-pca (query)               8_059.96       253.56     8_313.52       0.1449          5.7762         2.34
IVF-Binary-256-nl158-np12-rf0-pca (query)              8_059.96       257.90     8_317.87       0.1357          6.6787         2.34
IVF-Binary-256-nl158-np17-rf0-pca (query)              8_059.96       269.60     8_329.56       0.1312          7.3557         2.34
IVF-Binary-256-nl158-np7-rf10-pca (query)              8_059.96       336.23     8_396.19       0.4661          1.4656         2.34
IVF-Binary-256-nl158-np7-rf20-pca (query)              8_059.96       437.39     8_497.35       0.6170          1.2408         2.34
IVF-Binary-256-nl158-np12-rf10-pca (query)             8_059.96       347.72     8_407.68       0.4251          1.5539         2.34
IVF-Binary-256-nl158-np12-rf20-pca (query)             8_059.96       435.54     8_495.50       0.5648          1.3027         2.34
IVF-Binary-256-nl158-np17-rf10-pca (query)             8_059.96       377.78     8_437.74       0.4036          1.6097         2.34
IVF-Binary-256-nl158-np17-rf20-pca (query)             8_059.96       451.11     8_511.07       0.5372          1.3413         2.34
IVF-Binary-256-nl158-pca (self)                        8_059.96     1_026.07     9_086.03       0.4223          1.5629         2.34
IVF-Binary-256-nl223-np11-rf0-pca (query)              6_789.53       254.96     7_044.50       0.1421          5.7900         2.47
IVF-Binary-256-nl223-np14-rf0-pca (query)              6_789.53       261.62     7_051.16       0.1380          6.1428         2.47
IVF-Binary-256-nl223-np21-rf0-pca (query)              6_789.53       267.08     7_056.61       0.1324          6.8902         2.47
IVF-Binary-256-nl223-np11-rf10-pca (query)             6_789.53       340.34     7_129.87       0.4570          1.4773         2.47
IVF-Binary-256-nl223-np11-rf20-pca (query)             6_789.53       423.66     7_213.20       0.6071          1.2480         2.47
IVF-Binary-256-nl223-np14-rf10-pca (query)             6_789.53       342.72     7_132.25       0.4391          1.5146         2.47
IVF-Binary-256-nl223-np14-rf20-pca (query)             6_789.53       433.91     7_223.45       0.5852          1.2733         2.47
IVF-Binary-256-nl223-np21-rf10-pca (query)             6_789.53       355.49     7_145.02       0.4120          1.5813         2.47
IVF-Binary-256-nl223-np21-rf20-pca (query)             6_789.53       442.98     7_232.51       0.5505          1.3188         2.47
IVF-Binary-256-nl223-pca (self)                        6_789.53     1_008.00     7_797.54       0.4362          1.5225         2.47
IVF-Binary-256-nl316-np15-rf0-pca (query)              7_088.41       264.71     7_353.12       0.1410          5.7370         2.65
IVF-Binary-256-nl316-np17-rf0-pca (query)              7_088.41       263.32     7_351.73       0.1390          5.8998         2.65
IVF-Binary-256-nl316-np25-rf0-pca (query)              7_088.41       270.91     7_359.32       0.1341          6.4951         2.65
IVF-Binary-256-nl316-np15-rf10-pca (query)             7_088.41       347.47     7_435.88       0.4555          1.4768         2.65
IVF-Binary-256-nl316-np15-rf20-pca (query)             7_088.41       431.68     7_520.09       0.6070          1.2468         2.65
IVF-Binary-256-nl316-np17-rf10-pca (query)             7_088.41       346.80     7_435.21       0.4462          1.4963         2.65
IVF-Binary-256-nl316-np17-rf20-pca (query)             7_088.41       432.11     7_520.52       0.5954          1.2599         2.65
IVF-Binary-256-nl316-np25-rf10-pca (query)             7_088.41       355.81     7_444.22       0.4208          1.5556         2.65
IVF-Binary-256-nl316-np25-rf20-pca (query)             7_088.41       443.51     7_531.92       0.5625          1.3010         2.65
IVF-Binary-256-nl316-pca (self)                        7_088.41     1_020.69     8_109.10       0.4438          1.5029         2.65
IVF-Binary-512-nl158-np7-rf0-random (query)           13_467.97       455.19    13_923.16       0.6099          1.5613         4.36
IVF-Binary-512-nl158-np12-rf0-random (query)          13_467.97       466.03    13_934.01       0.6060          1.5937         4.36
IVF-Binary-512-nl158-np17-rf0-random (query)          13_467.97       478.57    13_946.54       0.6043          1.6153         4.36
IVF-Binary-512-nl158-np7-rf10-random (query)          13_467.97       539.60    14_007.57       0.9973          1.0004         4.36
IVF-Binary-512-nl158-np7-rf20-random (query)          13_467.97       611.34    14_079.32       0.9985          1.0003         4.36
IVF-Binary-512-nl158-np12-rf10-random (query)         13_467.97       545.55    14_013.53       0.9982          1.0002         4.36
IVF-Binary-512-nl158-np12-rf20-random (query)         13_467.97       628.66    14_096.64       0.9998          1.0000         4.36
IVF-Binary-512-nl158-np17-rf10-random (query)         13_467.97       560.60    14_028.57       0.9981          1.0002         4.36
IVF-Binary-512-nl158-np17-rf20-random (query)         13_467.97       638.28    14_106.25       0.9998          1.0000         4.36
IVF-Binary-512-nl158-random (self)                    13_467.97     1_679.28    15_147.25       0.9982          1.0002         4.36
IVF-Binary-512-nl223-np11-rf0-random (query)          12_173.29       462.11    12_635.40       0.6073          1.5772         4.49
IVF-Binary-512-nl223-np14-rf0-random (query)          12_173.29       462.20    12_635.49       0.6060          1.5877         4.49
IVF-Binary-512-nl223-np21-rf0-random (query)          12_173.29       475.45    12_648.75       0.6042          1.6107         4.49
IVF-Binary-512-nl223-np11-rf10-random (query)         12_173.29       539.25    12_712.54       0.9982          1.0002         4.49
IVF-Binary-512-nl223-np11-rf20-random (query)         12_173.29       631.14    12_804.44       0.9996          1.0001         4.49
IVF-Binary-512-nl223-np14-rf10-random (query)         12_173.29       552.22    12_725.51       0.9983          1.0002         4.49
IVF-Binary-512-nl223-np14-rf20-random (query)         12_173.29       632.01    12_805.30       0.9998          1.0000         4.49
IVF-Binary-512-nl223-np21-rf10-random (query)         12_173.29       551.95    12_725.24       0.9980          1.0002         4.49
IVF-Binary-512-nl223-np21-rf20-random (query)         12_173.29       634.70    12_808.00       0.9998          1.0000         4.49
IVF-Binary-512-nl223-random (self)                    12_173.29     1_645.27    13_818.56       0.9981          1.0002         4.49
IVF-Binary-512-nl316-np15-rf0-random (query)          12_421.38       468.70    12_890.08       0.6071          1.5794         4.67
IVF-Binary-512-nl316-np17-rf0-random (query)          12_421.38       469.19    12_890.57       0.6063          1.5867         4.67
IVF-Binary-512-nl316-np25-rf0-random (query)          12_421.38       481.19    12_902.57       0.6048          1.6036         4.67
IVF-Binary-512-nl316-np15-rf10-random (query)         12_421.38       551.78    12_973.16       0.9983          1.0002         4.67
IVF-Binary-512-nl316-np15-rf20-random (query)         12_421.38       639.90    13_061.28       0.9997          1.0000         4.67
IVF-Binary-512-nl316-np17-rf10-random (query)         12_421.38       554.44    12_975.82       0.9983          1.0002         4.67
IVF-Binary-512-nl316-np17-rf20-random (query)         12_421.38       621.16    13_042.54       0.9998          1.0000         4.67
IVF-Binary-512-nl316-np25-rf10-random (query)         12_421.38       575.17    12_996.56       0.9981          1.0002         4.67
IVF-Binary-512-nl316-np25-rf20-random (query)         12_421.38       633.99    13_055.37       0.9998          1.0000         4.67
IVF-Binary-512-nl316-random (self)                    12_421.38     1_663.80    14_085.18       0.9982          1.0002         4.67
IVF-Binary-512-nl158-np7-rf0-pca (query)              13_555.50       454.30    14_009.80       0.1402          6.0802         4.36
IVF-Binary-512-nl158-np12-rf0-pca (query)             13_555.50       470.09    14_025.58       0.1307          7.1069         4.36
IVF-Binary-512-nl158-np17-rf0-pca (query)             13_555.50       482.06    14_037.56       0.1259          7.9042         4.36
IVF-Binary-512-nl158-np7-rf10-pca (query)             13_555.50       538.13    14_093.62       0.4249          1.5448         4.36
IVF-Binary-512-nl158-np7-rf20-pca (query)             13_555.50       618.42    14_173.92       0.5632          1.3027         4.36
IVF-Binary-512-nl158-np12-rf10-pca (query)            13_555.50       561.38    14_116.88       0.3819          1.6569         4.36
IVF-Binary-512-nl158-np12-rf20-pca (query)            13_555.50       636.56    14_192.06       0.5036          1.3868         4.36
IVF-Binary-512-nl158-np17-rf10-pca (query)            13_555.50       591.26    14_146.76       0.3597          1.7319         4.36
IVF-Binary-512-nl158-np17-rf20-pca (query)            13_555.50       657.19    14_212.69       0.4726          1.4420         4.36
IVF-Binary-512-nl158-pca (self)                       13_555.50     1_723.09    15_278.58       0.3781          1.6708         4.36
IVF-Binary-512-nl223-np11-rf0-pca (query)             12_378.60       464.55    12_843.15       0.1373          6.0883         4.49
IVF-Binary-512-nl223-np14-rf0-pca (query)             12_378.60       475.94    12_854.54       0.1333          6.4764         4.49
IVF-Binary-512-nl223-np21-rf0-pca (query)             12_378.60       520.90    12_899.50       0.1274          7.3349         4.49
IVF-Binary-512-nl223-np11-rf10-pca (query)            12_378.60       555.76    12_934.36       0.4163          1.5550         4.49
IVF-Binary-512-nl223-np11-rf20-pca (query)            12_378.60       633.82    13_012.42       0.5535          1.3089         4.49
IVF-Binary-512-nl223-np14-rf10-pca (query)            12_378.60       548.33    12_926.93       0.3974          1.6030         4.49
IVF-Binary-512-nl223-np14-rf20-pca (query)            12_378.60       634.21    13_012.81       0.5275          1.3449         4.49
IVF-Binary-512-nl223-np21-rf10-pca (query)            12_378.60       560.05    12_938.65       0.3699          1.6875         4.49
IVF-Binary-512-nl223-np21-rf20-pca (query)            12_378.60       651.25    13_029.85       0.4885          1.4075         4.49
IVF-Binary-512-nl223-pca (self)                       12_378.60     1_703.23    14_081.84       0.3937          1.6147         4.49
IVF-Binary-512-nl316-np15-rf0-pca (query)             12_596.10       468.85    13_064.95       0.1362          6.0280         4.67
IVF-Binary-512-nl316-np17-rf0-pca (query)             12_596.10       471.07    13_067.18       0.1342          6.2163         4.67
IVF-Binary-512-nl316-np25-rf0-pca (query)             12_596.10       492.99    13_089.10       0.1289          6.8940         4.67
IVF-Binary-512-nl316-np15-rf10-pca (query)            12_596.10       552.95    13_149.05       0.4152          1.5547         4.67
IVF-Binary-512-nl316-np15-rf20-pca (query)            12_596.10       642.83    13_238.93       0.5522          1.3079         4.67
IVF-Binary-512-nl316-np17-rf10-pca (query)            12_596.10       552.11    13_148.21       0.4053          1.5792         4.67
IVF-Binary-512-nl316-np17-rf20-pca (query)            12_596.10       639.31    13_235.41       0.5384          1.3264         4.67
IVF-Binary-512-nl316-np25-rf10-pca (query)            12_596.10       592.81    13_188.91       0.3791          1.6548         4.67
IVF-Binary-512-nl316-np25-rf20-pca (query)            12_596.10       648.82    13_244.93       0.5016          1.3833         4.67
IVF-Binary-512-nl316-pca (self)                       12_596.10     1_724.39    14_320.49       0.4012          1.5900         4.67
IVF-Binary-1024-nl158-np7-rf0-random (query)          24_580.02       874.40    25_454.43       0.6689          1.3867         8.42
IVF-Binary-1024-nl158-np12-rf0-random (query)         24_580.02       890.13    25_470.15       0.6656          1.4072         8.42
IVF-Binary-1024-nl158-np17-rf0-random (query)         24_580.02       901.98    25_482.00       0.6641          1.4214         8.42
IVF-Binary-1024-nl158-np7-rf10-random (query)         24_580.02       944.44    25_524.46       0.9983          1.0003         8.42
IVF-Binary-1024-nl158-np7-rf20-random (query)         24_580.02     1_025.08    25_605.11       0.9985          1.0003         8.42
IVF-Binary-1024-nl158-np12-rf10-random (query)        24_580.02       961.46    25_541.49       0.9995          1.0001         8.42
IVF-Binary-1024-nl158-np12-rf20-random (query)        24_580.02     1_049.91    25_629.93       0.9999          1.0000         8.42
IVF-Binary-1024-nl158-np17-rf10-random (query)        24_580.02       994.55    25_574.58       0.9995          1.0001         8.42
IVF-Binary-1024-nl158-np17-rf20-random (query)        24_580.02     1_086.52    25_666.55       0.9999          1.0000         8.42
IVF-Binary-1024-nl158-random (self)                   24_580.02     3_067.40    27_647.42       0.9996          1.0001         8.42
IVF-Binary-1024-nl223-np11-rf0-random (query)         23_411.77       872.89    24_284.66       0.6669          1.3969         8.54
IVF-Binary-1024-nl223-np14-rf0-random (query)         23_411.77       876.25    24_288.03       0.6657          1.4046         8.54
IVF-Binary-1024-nl223-np21-rf0-random (query)         23_411.77       895.35    24_307.12       0.6644          1.4194         8.54
IVF-Binary-1024-nl223-np11-rf10-random (query)        23_411.77       945.82    24_357.59       0.9994          1.0001         8.54
IVF-Binary-1024-nl223-np11-rf20-random (query)        23_411.77     1_020.17    24_431.94       0.9997          1.0000         8.54
IVF-Binary-1024-nl223-np14-rf10-random (query)        23_411.77       949.63    24_361.41       0.9995          1.0001         8.54
IVF-Binary-1024-nl223-np14-rf20-random (query)        23_411.77     1_025.17    24_436.95       0.9999          1.0000         8.54
IVF-Binary-1024-nl223-np21-rf10-random (query)        23_411.77       977.99    24_389.77       0.9995          1.0001         8.54
IVF-Binary-1024-nl223-np21-rf20-random (query)        23_411.77     1_048.48    24_460.26       0.9999          1.0000         8.54
IVF-Binary-1024-nl223-random (self)                   23_411.77     3_015.10    26_426.88       0.9995          1.0001         8.54
IVF-Binary-1024-nl316-np15-rf0-random (query)         23_642.14       880.71    24_522.86       0.6663          1.4016         8.73
IVF-Binary-1024-nl316-np17-rf0-random (query)         23_642.14       882.88    24_525.02       0.6658          1.4059         8.73
IVF-Binary-1024-nl316-np25-rf0-random (query)         23_642.14       897.14    24_539.28       0.6646          1.4173         8.73
IVF-Binary-1024-nl316-np15-rf10-random (query)        23_642.14       950.86    24_593.00       0.9995          1.0001         8.73
IVF-Binary-1024-nl316-np15-rf20-random (query)        23_642.14     1_028.80    24_670.94       0.9998          1.0000         8.73
IVF-Binary-1024-nl316-np17-rf10-random (query)        23_642.14       952.74    24_594.88       0.9995          1.0001         8.73
IVF-Binary-1024-nl316-np17-rf20-random (query)        23_642.14     1_029.91    24_672.05       0.9999          1.0000         8.73
IVF-Binary-1024-nl316-np25-rf10-random (query)        23_642.14       968.25    24_610.39       0.9996          1.0001         8.73
IVF-Binary-1024-nl316-np25-rf20-random (query)        23_642.14     1_046.83    24_688.97       0.9999          1.0000         8.73
IVF-Binary-1024-nl316-random (self)                   23_642.14     3_010.08    26_652.22       0.9995          1.0001         8.73
IVF-Binary-1024-nl158-np7-rf0-pca (query)             24_838.50       863.82    25_702.31       0.4023          2.1616         8.42
IVF-Binary-1024-nl158-np12-rf0-pca (query)            24_838.50       887.55    25_726.04       0.3979          2.2360         8.42
IVF-Binary-1024-nl158-np17-rf0-pca (query)            24_838.50       904.73    25_743.23       0.3962          2.2803         8.42
IVF-Binary-1024-nl158-np7-rf10-pca (query)            24_838.50       943.45    25_781.94       0.8540          1.0604         8.42
IVF-Binary-1024-nl158-np7-rf20-pca (query)            24_838.50     1_019.15    25_857.64       0.9426          1.0186         8.42
IVF-Binary-1024-nl158-np12-rf10-pca (query)           24_838.50       966.75    25_805.24       0.8431          1.0671         8.42
IVF-Binary-1024-nl158-np12-rf20-pca (query)           24_838.50     1_041.49    25_879.99       0.9316          1.0231         8.42
IVF-Binary-1024-nl158-np17-rf10-pca (query)           24_838.50       979.24    25_817.74       0.8385          1.0701         8.42
IVF-Binary-1024-nl158-np17-rf20-pca (query)           24_838.50     1_062.80    25_901.30       0.9264          1.0254         8.42
IVF-Binary-1024-nl158-pca (self)                      24_838.50     3_115.13    27_953.62       0.8277          1.0772         8.42
IVF-Binary-1024-nl223-np11-rf0-pca (query)            23_662.71       887.17    24_549.89       0.3995          2.1870         8.54
IVF-Binary-1024-nl223-np14-rf0-pca (query)            23_662.71       876.88    24_539.60       0.3982          2.2111         8.54
IVF-Binary-1024-nl223-np21-rf0-pca (query)            23_662.71       947.80    24_610.51       0.3964          2.2676         8.54
IVF-Binary-1024-nl223-np11-rf10-pca (query)           23_662.71       944.99    24_607.70       0.8498          1.0631         8.54
IVF-Binary-1024-nl223-np11-rf20-pca (query)           23_662.71     1_029.32    24_692.04       0.9391          1.0200         8.54
IVF-Binary-1024-nl223-np14-rf10-pca (query)           23_662.71       968.83    24_631.54       0.8455          1.0657         8.54
IVF-Binary-1024-nl223-np14-rf20-pca (query)           23_662.71     1_028.51    24_691.22       0.9342          1.0220         8.54
IVF-Binary-1024-nl223-np21-rf10-pca (query)           23_662.71     1_040.67    24_703.39       0.8394          1.0696         8.54
IVF-Binary-1024-nl223-np21-rf20-pca (query)           23_662.71     1_052.52    24_715.23       0.9272          1.0251         8.54
IVF-Binary-1024-nl223-pca (self)                      23_662.71     3_037.66    26_700.38       0.8300          1.0757         8.54
IVF-Binary-1024-nl316-np15-rf0-pca (query)            23_932.62       888.75    24_821.37       0.3992          2.1899         8.73
IVF-Binary-1024-nl316-np17-rf0-pca (query)            23_932.62       879.61    24_812.23       0.3984          2.2034         8.73
IVF-Binary-1024-nl316-np25-rf0-pca (query)            23_932.62       898.19    24_830.81       0.3969          2.2462         8.73
IVF-Binary-1024-nl316-np15-rf10-pca (query)           23_932.62       949.88    24_882.50       0.8494          1.0634         8.73
IVF-Binary-1024-nl316-np15-rf20-pca (query)           23_932.62     1_044.50    24_977.11       0.9381          1.0205         8.73
IVF-Binary-1024-nl316-np17-rf10-pca (query)           23_932.62       973.02    24_905.64       0.8471          1.0648         8.73
IVF-Binary-1024-nl316-np17-rf20-pca (query)           23_932.62     1_038.75    24_971.37       0.9355          1.0215         8.73
IVF-Binary-1024-nl316-np25-rf10-pca (query)           23_932.62       974.22    24_906.83       0.8409          1.0685         8.73
IVF-Binary-1024-nl316-np25-rf20-pca (query)           23_932.62     1_053.41    24_986.03       0.9293          1.0242         8.73
IVF-Binary-1024-nl316-pca (self)                      23_932.62     3_035.23    26_967.85       0.8315          1.0747         8.73
IVF-Binary-512-nl158-np7-rf0-sign (query)              2_009.80       424.71     2_434.51       0.1510          4.5058         3.36
IVF-Binary-512-nl158-np12-rf0-sign (query)             2_009.80       463.82     2_473.61       0.1361          5.0356         3.36
IVF-Binary-512-nl158-np17-rf0-sign (query)             2_009.80       523.65     2_533.45       0.1277          5.5367         3.36
IVF-Binary-512-nl158-np7-rf10-sign (query)             2_009.80       472.62     2_482.41       0.4704          1.6381         3.36
IVF-Binary-512-nl158-np7-rf20-sign (query)             2_009.80       837.39     2_847.19       0.5942          1.4484         3.36
IVF-Binary-512-nl158-np12-rf10-sign (query)            2_009.80       513.47     2_523.27       0.3942          1.8935         3.36
IVF-Binary-512-nl158-np12-rf20-sign (query)            2_009.80       904.31     2_914.10       0.5140          1.6480         3.36
IVF-Binary-512-nl158-np17-rf10-sign (query)            2_009.80       551.06     2_560.85       0.3569          2.0547         3.36
IVF-Binary-512-nl158-np17-rf20-sign (query)            2_009.80       942.00     2_951.79       0.4697          1.7895         3.36
IVF-Binary-512-nl158-sign (self)                       2_009.80     1_511.20     3_521.00       0.3955          1.8825         3.36
IVF-Binary-512-nl223-np11-rf0-sign (query)               853.26       441.61     1_294.86       0.1213          4.6896         3.49
IVF-Binary-512-nl223-np14-rf0-sign (query)               853.26       484.72     1_337.98       0.1174          4.9249         3.49
IVF-Binary-512-nl223-np21-rf0-sign (query)               853.26       510.92     1_364.18       0.1106          5.4835         3.49
IVF-Binary-512-nl223-np11-rf10-sign (query)              853.26       496.60     1_349.86       0.4197          1.7170         3.49
IVF-Binary-512-nl223-np11-rf20-sign (query)              853.26       864.80     1_718.05       0.5241          1.5150         3.49
IVF-Binary-512-nl223-np14-rf10-sign (query)              853.26       515.23     1_368.49       0.3911          1.7989         3.49
IVF-Binary-512-nl223-np14-rf20-sign (query)              853.26       902.77     1_756.02       0.4903          1.5872         3.49
IVF-Binary-512-nl223-np21-rf10-sign (query)              853.26       593.15     1_446.41       0.3459          1.9626         3.49
IVF-Binary-512-nl223-np21-rf20-sign (query)              853.26       952.20     1_805.46       0.4403          1.7076         3.49
IVF-Binary-512-nl223-sign (self)                         853.26     1_512.35     2_365.61       0.3897          1.7981         3.49
IVF-Binary-512-nl316-np15-rf0-sign (query)             1_087.60       467.24     1_554.83       0.1151          4.6038         3.67
IVF-Binary-512-nl316-np17-rf0-sign (query)             1_087.60       479.61     1_567.21       0.1136          4.7084         3.67
IVF-Binary-512-nl316-np25-rf0-sign (query)             1_087.60       529.04     1_616.64       0.1106          5.1122         3.67
IVF-Binary-512-nl316-np15-rf10-sign (query)            1_087.60       526.86     1_614.46       0.4126          1.7028         3.67
IVF-Binary-512-nl316-np15-rf20-sign (query)            1_087.60       893.82     1_981.42       0.5130          1.5028         3.67
IVF-Binary-512-nl316-np17-rf10-sign (query)            1_087.60       535.57     1_623.17       0.3986          1.7407         3.67
IVF-Binary-512-nl316-np17-rf20-sign (query)            1_087.60       914.90     2_002.50       0.4955          1.5385         3.67
IVF-Binary-512-nl316-np25-rf10-sign (query)            1_087.60       581.54     1_669.13       0.3574          1.8706         3.67
IVF-Binary-512-nl316-np25-rf20-sign (query)            1_087.60       969.24     2_056.83       0.4496          1.6513         3.67
IVF-Binary-512-nl316-sign (self)                       1_087.60     1_547.18     2_634.77       0.3990          1.7352         3.67
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D - Binary Quantisation
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       100.06     1_782.35     1_882.41       1.0000          1.0000       146.48
Exhaustive (self)                                        100.06     5_994.96     6_095.01       1.0000          1.0000       146.48
ExhaustiveBinary-256-random_no_rr (query)              9_052.05       539.48     9_591.52       0.5361          1.8069         2.28
ExhaustiveBinary-256-random-rf10 (query)               9_052.05       692.55     9_744.60       0.9868          1.0022         2.28
ExhaustiveBinary-256-random-rf20 (query)               9_052.05       849.85     9_901.89       0.9980          1.0003         2.28
ExhaustiveBinary-256-random (self)                     9_052.05     2_378.16    11_430.21       0.9875          1.0021         2.28
ExhaustiveBinary-256-pca_no_rr (query)                 9_435.01       537.71     9_972.72       0.1281         12.2485         2.28
ExhaustiveBinary-256-pca-rf10 (query)                  9_435.01       702.28    10_137.30       0.3750          1.7664         2.28
ExhaustiveBinary-256-pca-rf20 (query)                  9_435.01       834.07    10_269.08       0.5003          1.4421         2.28
ExhaustiveBinary-256-pca (self)                        9_435.01     2_194.09    11_629.11       0.3725          1.7770         2.28
ExhaustiveBinary-512-random_no_rr (query)             17_745.82       907.01    18_652.83       0.5866          1.6777         4.55
ExhaustiveBinary-512-random-rf10 (query)              17_745.82     1_050.13    18_795.95       0.9966          1.0005         4.55
ExhaustiveBinary-512-random-rf20 (query)              17_745.82     1_206.49    18_952.31       0.9996          1.0001         4.55
ExhaustiveBinary-512-random (self)                    17_745.82     3_408.89    21_154.70       0.9969          1.0004         4.55
ExhaustiveBinary-512-pca_no_rr (query)                17_992.93       901.12    18_894.05       0.1131         15.1544         4.55
ExhaustiveBinary-512-pca-rf10 (query)                 17_992.93     1_054.56    19_047.49       0.3166          2.0536         4.55
ExhaustiveBinary-512-pca-rf20 (query)                 17_992.93     1_187.74    19_180.67       0.4179          1.6492         4.55
ExhaustiveBinary-512-pca (self)                       17_992.93     3_387.17    21_380.11       0.3146          2.0599         4.55
ExhaustiveBinary-1024-random_no_rr (query)            35_094.69     1_692.33    36_787.03       0.6446          1.4908         9.11
ExhaustiveBinary-1024-random-rf10 (query)             35_094.69     1_817.41    36_912.10       0.9993          1.0001         9.11
ExhaustiveBinary-1024-random-rf20 (query)             35_094.69     1_975.50    37_070.19       0.9999          1.0000         9.11
ExhaustiveBinary-1024-random (self)                   35_094.69     5_973.84    41_068.54       0.9993          1.0001         9.11
ExhaustiveBinary-1024-pca_no_rr (query)               35_549.64     1_666.65    37_216.29       0.2379          4.2518         9.11
ExhaustiveBinary-1024-pca-rf10 (query)                35_549.64     1_818.64    37_368.28       0.6180          1.2507         9.11
ExhaustiveBinary-1024-pca-rf20 (query)                35_549.64     1_970.17    37_519.81       0.7452          1.1303         9.11
ExhaustiveBinary-1024-pca (self)                      35_549.64     5_966.82    41_516.46       0.6017          1.2739         9.11
ExhaustiveBinary-768-sign_no_rr (query)                  211.12       903.40     1_114.52       0.0420         17.7101         4.58
ExhaustiveBinary-768-sign-rf10 (query)                   211.12       933.92     1_145.04       0.1896          2.5240         4.58
ExhaustiveBinary-768-sign-rf20 (query)                   211.12     1_492.22     1_703.34       0.3228          1.8300         4.58
ExhaustiveBinary-768-sign (self)                         211.12     2_998.99     3_210.11       0.1997          2.4832         4.58
IVF-Binary-256-nl158-np7-rf0-random (query)           11_679.34       368.97    12_048.31       0.5435          1.7067         2.74
IVF-Binary-256-nl158-np12-rf0-random (query)          11_679.34       403.23    12_082.57       0.5415          1.7256         2.74
IVF-Binary-256-nl158-np17-rf0-random (query)          11_679.34       387.96    12_067.30       0.5404          1.7418         2.74
IVF-Binary-256-nl158-np7-rf10-random (query)          11_679.34       473.61    12_152.95       0.9891          1.0017         2.74
IVF-Binary-256-nl158-np7-rf20-random (query)          11_679.34       568.56    12_247.90       0.9984          1.0002         2.74
IVF-Binary-256-nl158-np12-rf10-random (query)         11_679.34       479.24    12_158.58       0.9884          1.0019         2.74
IVF-Binary-256-nl158-np12-rf20-random (query)         11_679.34       582.92    12_262.26       0.9985          1.0002         2.74
IVF-Binary-256-nl158-np17-rf10-random (query)         11_679.34       490.69    12_170.03       0.9878          1.0020         2.74
IVF-Binary-256-nl158-np17-rf20-random (query)         11_679.34       589.33    12_268.67       0.9983          1.0002         2.74
IVF-Binary-256-nl158-random (self)                    11_679.34     1_419.02    13_098.36       0.9891          1.0018         2.74
IVF-Binary-256-nl223-np11-rf0-random (query)           9_913.89       382.55    10_296.45       0.5424          1.7135         2.93
IVF-Binary-256-nl223-np14-rf0-random (query)           9_913.89       414.57    10_328.47       0.5416          1.7223         2.93
IVF-Binary-256-nl223-np21-rf0-random (query)           9_913.89       397.70    10_311.60       0.5406          1.7399         2.93
IVF-Binary-256-nl223-np11-rf10-random (query)          9_913.89       477.75    10_391.64       0.9890          1.0018         2.93
IVF-Binary-256-nl223-np11-rf20-random (query)          9_913.89       587.01    10_500.90       0.9986          1.0002         2.93
IVF-Binary-256-nl223-np14-rf10-random (query)          9_913.89       482.06    10_395.95       0.9885          1.0019         2.93
IVF-Binary-256-nl223-np14-rf20-random (query)          9_913.89       582.62    10_496.51       0.9986          1.0002         2.93
IVF-Binary-256-nl223-np21-rf10-random (query)          9_913.89       490.14    10_404.03       0.9879          1.0021         2.93
IVF-Binary-256-nl223-np21-rf20-random (query)          9_913.89       596.32    10_510.21       0.9983          1.0002         2.93
IVF-Binary-256-nl223-random (self)                     9_913.89     1_402.50    11_316.40       0.9893          1.0017         2.93
IVF-Binary-256-nl316-np15-rf0-random (query)          10_320.67       389.64    10_710.31       0.5427          1.7075         3.21
IVF-Binary-256-nl316-np17-rf0-random (query)          10_320.67       392.46    10_713.13       0.5424          1.7119         3.21
IVF-Binary-256-nl316-np25-rf0-random (query)          10_320.67       397.45    10_718.12       0.5414          1.7263         3.21
IVF-Binary-256-nl316-np15-rf10-random (query)         10_320.67       488.44    10_809.11       0.9891          1.0018         3.21
IVF-Binary-256-nl316-np15-rf20-random (query)         10_320.67       587.94    10_908.61       0.9987          1.0002         3.21
IVF-Binary-256-nl316-np17-rf10-random (query)         10_320.67       489.53    10_810.20       0.9888          1.0019         3.21
IVF-Binary-256-nl316-np17-rf20-random (query)         10_320.67       593.07    10_913.74       0.9986          1.0002         3.21
IVF-Binary-256-nl316-np25-rf10-random (query)         10_320.67       496.91    10_817.58       0.9881          1.0020         3.21
IVF-Binary-256-nl316-np25-rf20-random (query)         10_320.67       606.44    10_927.11       0.9984          1.0002         3.21
IVF-Binary-256-nl316-random (self)                    10_320.67     1_432.22    11_752.89       0.9896          1.0017         3.21
IVF-Binary-256-nl158-np7-rf0-pca (query)              12_089.74       370.34    12_460.08       0.1457          5.8563         2.74
IVF-Binary-256-nl158-np12-rf0-pca (query)             12_089.74       380.24    12_469.98       0.1391          6.6825         2.74
IVF-Binary-256-nl158-np17-rf0-pca (query)             12_089.74       391.91    12_481.65       0.1357          7.3261         2.74
IVF-Binary-256-nl158-np7-rf10-pca (query)             12_089.74       479.32    12_569.06       0.4666          1.4653         2.74
IVF-Binary-256-nl158-np7-rf20-pca (query)             12_089.74       572.53    12_662.28       0.6312          1.2274         2.74
IVF-Binary-256-nl158-np12-rf10-pca (query)            12_089.74       484.36    12_574.10       0.4346          1.5336         2.74
IVF-Binary-256-nl158-np12-rf20-pca (query)            12_089.74       593.85    12_683.59       0.5900          1.2731         2.74
IVF-Binary-256-nl158-np17-rf10-pca (query)            12_089.74       500.10    12_589.85       0.4168          1.5776         2.74
IVF-Binary-256-nl158-np17-rf20-pca (query)            12_089.74       638.29    12_728.03       0.5672          1.3020         2.74
IVF-Binary-256-nl158-pca (self)                       12_089.74     1_461.03    13_550.78       0.4323          1.5394         2.74
IVF-Binary-256-nl223-np11-rf0-pca (query)             10_374.53       381.14    10_755.67       0.1439          5.8917         2.93
IVF-Binary-256-nl223-np14-rf0-pca (query)             10_374.53       380.63    10_755.16       0.1410          6.1872         2.93
IVF-Binary-256-nl223-np21-rf0-pca (query)             10_374.53       389.01    10_763.54       0.1368          6.8758         2.93
IVF-Binary-256-nl223-np11-rf10-pca (query)            10_374.53       484.50    10_859.02       0.4605          1.4712         2.93
IVF-Binary-256-nl223-np11-rf20-pca (query)            10_374.53       588.80    10_963.33       0.6245          1.2311         2.93
IVF-Binary-256-nl223-np14-rf10-pca (query)            10_374.53       487.38    10_861.90       0.4464          1.5013         2.93
IVF-Binary-256-nl223-np14-rf20-pca (query)            10_374.53       618.52    10_993.05       0.6063          1.2512         2.93
IVF-Binary-256-nl223-np21-rf10-pca (query)            10_374.53       509.12    10_883.65       0.4240          1.5556         2.93
IVF-Binary-256-nl223-np21-rf20-pca (query)            10_374.53       615.02    10_989.54       0.5775          1.2871         2.93
IVF-Binary-256-nl223-pca (self)                       10_374.53     1_451.02    11_825.55       0.4444          1.5069         2.93
IVF-Binary-256-nl316-np15-rf0-pca (query)             10_737.04       390.56    11_127.60       0.1440          5.7939         3.21
IVF-Binary-256-nl316-np17-rf0-pca (query)             10_737.04       391.17    11_128.21       0.1425          5.9525         3.21
IVF-Binary-256-nl316-np25-rf0-pca (query)             10_737.04       396.88    11_133.93       0.1385          6.5002         3.21
IVF-Binary-256-nl316-np15-rf10-pca (query)            10_737.04       496.15    11_233.20       0.4618          1.4673         3.21
IVF-Binary-256-nl316-np15-rf20-pca (query)            10_737.04       597.97    11_335.02       0.6271          1.2275         3.21
IVF-Binary-256-nl316-np17-rf10-pca (query)            10_737.04       501.22    11_238.27       0.4542          1.4835         3.21
IVF-Binary-256-nl316-np17-rf20-pca (query)            10_737.04       609.42    11_346.47       0.6168          1.2384         3.21
IVF-Binary-256-nl316-np25-rf10-pca (query)            10_737.04       504.23    11_241.27       0.4328          1.5314         3.21
IVF-Binary-256-nl316-np25-rf20-pca (query)            10_737.04       609.06    11_346.10       0.5893          1.2709         3.21
IVF-Binary-256-nl316-pca (self)                       10_737.04     1_479.49    12_216.53       0.4521          1.4878         3.21
IVF-Binary-512-nl158-np7-rf0-random (query)           20_380.37       685.34    21_065.71       0.5928          1.5995         5.02
IVF-Binary-512-nl158-np12-rf0-random (query)          20_380.37       695.87    21_076.24       0.5905          1.6171         5.02
IVF-Binary-512-nl158-np17-rf0-random (query)          20_380.37       712.71    21_093.08       0.5893          1.6328         5.02
IVF-Binary-512-nl158-np7-rf10-random (query)          20_380.37       779.47    21_159.84       0.9972          1.0004         5.02
IVF-Binary-512-nl158-np7-rf20-random (query)          20_380.37       867.49    21_247.86       0.9994          1.0001         5.02
IVF-Binary-512-nl158-np12-rf10-random (query)         20_380.37       815.15    21_195.52       0.9972          1.0003         5.02
IVF-Binary-512-nl158-np12-rf20-random (query)         20_380.37       880.03    21_260.40       0.9998          1.0000         5.02
IVF-Binary-512-nl158-np17-rf10-random (query)         20_380.37       852.99    21_233.36       0.9969          1.0004         5.02
IVF-Binary-512-nl158-np17-rf20-random (query)         20_380.37       896.06    21_276.43       0.9997          1.0000         5.02
IVF-Binary-512-nl158-random (self)                    20_380.37     2_437.87    22_818.24       0.9975          1.0003         5.02
IVF-Binary-512-nl223-np11-rf0-random (query)          18_688.90       706.99    19_395.89       0.5912          1.6081         5.21
IVF-Binary-512-nl223-np14-rf0-random (query)          18_688.90       691.02    19_379.92       0.5904          1.6164         5.21
IVF-Binary-512-nl223-np21-rf0-random (query)          18_688.90       717.81    19_406.71       0.5893          1.6325         5.21
IVF-Binary-512-nl223-np11-rf10-random (query)         18_688.90       786.17    19_475.07       0.9973          1.0004         5.21
IVF-Binary-512-nl223-np11-rf20-random (query)         18_688.90       871.00    19_559.90       0.9997          1.0000         5.21
IVF-Binary-512-nl223-np14-rf10-random (query)         18_688.90       788.45    19_477.35       0.9971          1.0004         5.21
IVF-Binary-512-nl223-np14-rf20-random (query)         18_688.90       879.73    19_568.63       0.9997          1.0000         5.21
IVF-Binary-512-nl223-np21-rf10-random (query)         18_688.90       796.60    19_485.50       0.9970          1.0004         5.21
IVF-Binary-512-nl223-np21-rf20-random (query)         18_688.90       894.09    19_582.99       0.9997          1.0000         5.21
IVF-Binary-512-nl223-random (self)                    18_688.90     2_424.42    21_113.32       0.9975          1.0003         5.21
IVF-Binary-512-nl316-np15-rf0-random (query)          19_083.26       700.23    19_783.48       0.5914          1.6049         5.48
IVF-Binary-512-nl316-np17-rf0-random (query)          19_083.26       714.64    19_797.89       0.5910          1.6095         5.48
IVF-Binary-512-nl316-np25-rf0-random (query)          19_083.26       712.67    19_795.93       0.5899          1.6239         5.48
IVF-Binary-512-nl316-np15-rf10-random (query)         19_083.26       799.61    19_882.87       0.9974          1.0003         5.48
IVF-Binary-512-nl316-np15-rf20-random (query)         19_083.26       955.08    20_038.34       0.9998          1.0000         5.48
IVF-Binary-512-nl316-np17-rf10-random (query)         19_083.26       798.29    19_881.54       0.9974          1.0004         5.48
IVF-Binary-512-nl316-np17-rf20-random (query)         19_083.26       893.69    19_976.95       0.9998          1.0000         5.48
IVF-Binary-512-nl316-np25-rf10-random (query)         19_083.26       803.62    19_886.88       0.9971          1.0004         5.48
IVF-Binary-512-nl316-np25-rf20-random (query)         19_083.26       900.92    19_984.18       0.9997          1.0000         5.48
IVF-Binary-512-nl316-random (self)                    19_083.26     2_456.40    21_539.66       0.9976          1.0003         5.48
IVF-Binary-512-nl158-np7-rf0-pca (query)              20_867.12       679.94    21_547.05       0.1305          6.4503         5.02
IVF-Binary-512-nl158-np12-rf0-pca (query)             20_867.12       700.55    21_567.66       0.1240          7.4486         5.02
IVF-Binary-512-nl158-np17-rf0-pca (query)             20_867.12       706.90    21_574.02       0.1206          8.2329         5.02
IVF-Binary-512-nl158-np7-rf10-pca (query)             20_867.12       783.89    21_651.01       0.4226          1.5592         5.02
IVF-Binary-512-nl158-np7-rf20-pca (query)             20_867.12       897.31    21_764.43       0.5795          1.2871         5.02
IVF-Binary-512-nl158-np12-rf10-pca (query)            20_867.12       787.07    21_654.18       0.3886          1.6494         5.02
IVF-Binary-512-nl158-np12-rf20-pca (query)            20_867.12       900.69    21_767.81       0.5326          1.3504         5.02
IVF-Binary-512-nl158-np17-rf10-pca (query)            20_867.12       805.06    21_672.18       0.3705          1.7076         5.02
IVF-Binary-512-nl158-np17-rf20-pca (query)            20_867.12       912.52    21_779.64       0.5073          1.3915         5.02
IVF-Binary-512-nl158-pca (self)                       20_867.12     2_487.63    23_354.75       0.3870          1.6546         5.02
IVF-Binary-512-nl223-np11-rf0-pca (query)             19_063.40       694.17    19_757.56       0.1290          6.4930         5.21
IVF-Binary-512-nl223-np14-rf0-pca (query)             19_063.40       699.76    19_763.16       0.1260          6.8508         5.21
IVF-Binary-512-nl223-np21-rf0-pca (query)             19_063.40       708.46    19_771.86       0.1218          7.6778         5.21
IVF-Binary-512-nl223-np11-rf10-pca (query)            19_063.40       786.49    19_849.89       0.4165          1.5663         5.21
IVF-Binary-512-nl223-np11-rf20-pca (query)            19_063.40       887.13    19_950.52       0.5722          1.2915         5.21
IVF-Binary-512-nl223-np14-rf10-pca (query)            19_063.40       792.48    19_855.87       0.4017          1.6050         5.21
IVF-Binary-512-nl223-np14-rf20-pca (query)            19_063.40       892.64    19_956.04       0.5516          1.3190         5.21
IVF-Binary-512-nl223-np21-rf10-pca (query)            19_063.40       804.11    19_867.50       0.3784          1.6758         5.21
IVF-Binary-512-nl223-np21-rf20-pca (query)            19_063.40       919.00    19_982.40       0.5198          1.3682         5.21
IVF-Binary-512-nl223-pca (self)                       19_063.40     2_480.08    21_543.48       0.4000          1.6099         5.21
IVF-Binary-512-nl316-np15-rf0-pca (query)             19_428.20       708.87    20_137.07       0.1288          6.3626         5.48
IVF-Binary-512-nl316-np17-rf0-pca (query)             19_428.20       701.60    20_129.80       0.1273          6.5540         5.48
IVF-Binary-512-nl316-np25-rf0-pca (query)             19_428.20       711.76    20_139.95       0.1233          7.2119         5.48
IVF-Binary-512-nl316-np15-rf10-pca (query)            19_428.20       811.80    20_240.00       0.4179          1.5602         5.48
IVF-Binary-512-nl316-np15-rf20-pca (query)            19_428.20       896.04    20_324.24       0.5746          1.2868         5.48
IVF-Binary-512-nl316-np17-rf10-pca (query)            19_428.20       812.41    20_240.61       0.4100          1.5803         5.48
IVF-Binary-512-nl316-np17-rf20-pca (query)            19_428.20       901.41    20_329.61       0.5630          1.3018         5.48
IVF-Binary-512-nl316-np25-rf10-pca (query)            19_428.20       825.22    20_253.42       0.3876          1.6436         5.48
IVF-Binary-512-nl316-np25-rf20-pca (query)            19_428.20       915.67    20_343.87       0.5324          1.3461         5.48
IVF-Binary-512-nl316-pca (self)                       19_428.20     2_514.66    21_942.86       0.4084          1.5847         5.48
IVF-Binary-1024-nl158-np7-rf0-random (query)          37_901.46     1_308.34    39_209.80       0.6492          1.4395         9.57
IVF-Binary-1024-nl158-np12-rf0-random (query)         37_901.46     1_322.74    39_224.20       0.6471          1.4527         9.57
IVF-Binary-1024-nl158-np17-rf0-random (query)         37_901.46     1_337.30    39_238.76       0.6459          1.4636         9.57
IVF-Binary-1024-nl158-np7-rf10-random (query)         37_901.46     1_392.06    39_293.51       0.9991          1.0001         9.57
IVF-Binary-1024-nl158-np7-rf20-random (query)         37_901.46     1_481.17    39_382.63       0.9995          1.0001         9.57
IVF-Binary-1024-nl158-np12-rf10-random (query)        37_901.46     1_413.52    39_314.98       0.9995          1.0001         9.57
IVF-Binary-1024-nl158-np12-rf20-random (query)        37_901.46     1_513.43    39_414.88       0.9999          1.0000         9.57
IVF-Binary-1024-nl158-np17-rf10-random (query)        37_901.46     1_427.28    39_328.73       0.9994          1.0001         9.57
IVF-Binary-1024-nl158-np17-rf20-random (query)        37_901.46     1_526.53    39_427.99       0.9999          1.0000         9.57
IVF-Binary-1024-nl158-random (self)                   37_901.46     4_525.57    42_427.02       0.9995          1.0001         9.57
IVF-Binary-1024-nl223-np11-rf0-random (query)         36_216.40     1_338.60    37_555.00       0.6478          1.4482         9.76
IVF-Binary-1024-nl223-np14-rf0-random (query)         36_216.40     1_344.73    37_561.13       0.6470          1.4545         9.76
IVF-Binary-1024-nl223-np21-rf0-random (query)         36_216.40     1_332.97    37_549.37       0.6459          1.4660         9.76
IVF-Binary-1024-nl223-np11-rf10-random (query)        36_216.40     1_402.59    37_618.99       0.9994          1.0001         9.76
IVF-Binary-1024-nl223-np11-rf20-random (query)        36_216.40     1_490.14    37_706.54       0.9998          1.0000         9.76
IVF-Binary-1024-nl223-np14-rf10-random (query)        36_216.40     1_404.04    37_620.43       0.9994          1.0001         9.76
IVF-Binary-1024-nl223-np14-rf20-random (query)        36_216.40     1_500.15    37_716.55       0.9999          1.0000         9.76
IVF-Binary-1024-nl223-np21-rf10-random (query)        36_216.40     1_419.35    37_635.75       0.9994          1.0001         9.76
IVF-Binary-1024-nl223-np21-rf20-random (query)        36_216.40     1_522.15    37_738.55       0.9999          1.0000         9.76
IVF-Binary-1024-nl223-random (self)                   36_216.40     4_506.17    40_722.57       0.9995          1.0001         9.76
IVF-Binary-1024-nl316-np15-rf0-random (query)         36_572.46     1_344.41    37_916.88       0.6478          1.4451        10.04
IVF-Binary-1024-nl316-np17-rf0-random (query)         36_572.46     1_332.45    37_904.91       0.6474          1.4486        10.04
IVF-Binary-1024-nl316-np25-rf0-random (query)         36_572.46     1_349.59    37_922.06       0.6465          1.4581        10.04
IVF-Binary-1024-nl316-np15-rf10-random (query)        36_572.46     1_426.75    37_999.21       0.9995          1.0001        10.04
IVF-Binary-1024-nl316-np15-rf20-random (query)        36_572.46     1_539.69    38_112.16       0.9999          1.0000        10.04
IVF-Binary-1024-nl316-np17-rf10-random (query)        36_572.46     1_415.81    37_988.28       0.9995          1.0001        10.04
IVF-Binary-1024-nl316-np17-rf20-random (query)        36_572.46     1_510.84    38_083.30       0.9999          1.0000        10.04
IVF-Binary-1024-nl316-np25-rf10-random (query)        36_572.46     1_435.90    38_008.36       0.9994          1.0001        10.04
IVF-Binary-1024-nl316-np25-rf20-random (query)        36_572.46     1_536.15    38_108.61       0.9999          1.0000        10.04
IVF-Binary-1024-nl316-random (self)                   36_572.46     4_520.85    41_093.31       0.9995          1.0001        10.04
IVF-Binary-1024-nl158-np7-rf0-pca (query)             38_331.26     1_312.08    39_643.34       0.2469          3.4833         9.57
IVF-Binary-1024-nl158-np12-rf0-pca (query)            38_331.26     1_332.63    39_663.90       0.2424          3.6736         9.57
IVF-Binary-1024-nl158-np17-rf0-pca (query)            38_331.26     1_346.71    39_677.98       0.2408          3.7958         9.57
IVF-Binary-1024-nl158-np7-rf10-pca (query)            38_331.26     1_426.54    39_757.81       0.6578          1.2066         9.57
IVF-Binary-1024-nl158-np7-rf20-pca (query)            38_331.26     1_496.20    39_827.47       0.7968          1.0953         9.57
IVF-Binary-1024-nl158-np12-rf10-pca (query)           38_331.26     1_452.22    39_783.48       0.6394          1.2250         9.57
IVF-Binary-1024-nl158-np12-rf20-pca (query)           38_331.26     1_543.35    39_874.62       0.7721          1.1103         9.57
IVF-Binary-1024-nl158-np17-rf10-pca (query)           38_331.26     1_438.23    39_769.49       0.6306          1.2347         9.57
IVF-Binary-1024-nl158-np17-rf20-pca (query)           38_331.26     1_551.02    39_882.29       0.7608          1.1180         9.57
IVF-Binary-1024-nl158-pca (self)                      38_331.26     4_583.94    42_915.20       0.6243          1.2452         9.57
IVF-Binary-1024-nl223-np11-rf0-pca (query)            36_451.40     1_325.11    37_776.51       0.2449          3.5197         9.76
IVF-Binary-1024-nl223-np14-rf0-pca (query)            36_451.40     1_323.02    37_774.42       0.2431          3.5920         9.76
IVF-Binary-1024-nl223-np21-rf0-pca (query)            36_451.40     1_329.88    37_781.28       0.2408          3.7558         9.76
IVF-Binary-1024-nl223-np11-rf10-pca (query)           36_451.40     1_400.10    37_851.50       0.6523          1.2109         9.76
IVF-Binary-1024-nl223-np11-rf20-pca (query)           36_451.40     1_511.15    37_962.55       0.7909          1.0980         9.76
IVF-Binary-1024-nl223-np14-rf10-pca (query)           36_451.40     1_410.23    37_861.62       0.6443          1.2192         9.76
IVF-Binary-1024-nl223-np14-rf20-pca (query)           36_451.40     1_507.62    37_959.02       0.7798          1.1049         9.76
IVF-Binary-1024-nl223-np21-rf10-pca (query)           36_451.40     1_434.10    37_885.50       0.6327          1.2324         9.76
IVF-Binary-1024-nl223-np21-rf20-pca (query)           36_451.40     1_577.85    38_029.25       0.7644          1.1156         9.76
IVF-Binary-1024-nl223-pca (self)                      36_451.40     4_591.79    41_043.19       0.6304          1.2374         9.76
IVF-Binary-1024-nl316-np15-rf0-pca (query)            36_818.97     1_315.95    38_134.92       0.2449          3.5038        10.04
IVF-Binary-1024-nl316-np17-rf0-pca (query)            36_818.97     1_319.43    38_138.40       0.2440          3.5444        10.04
IVF-Binary-1024-nl316-np25-rf0-pca (query)            36_818.97     1_425.69    38_244.66       0.2417          3.6757        10.04
IVF-Binary-1024-nl316-np15-rf10-pca (query)           36_818.97     1_424.25    38_243.22       0.6517          1.2115        10.04
IVF-Binary-1024-nl316-np15-rf20-pca (query)           36_818.97     1_516.18    38_335.15       0.7905          1.0983        10.04
IVF-Binary-1024-nl316-np17-rf10-pca (query)           36_818.97     1_413.46    38_232.43       0.6474          1.2159        10.04
IVF-Binary-1024-nl316-np17-rf20-pca (query)           36_818.97     1_525.37    38_344.34       0.7846          1.1020        10.04
IVF-Binary-1024-nl316-np25-rf10-pca (query)           36_818.97     1_433.34    38_252.31       0.6363          1.2280        10.04
IVF-Binary-1024-nl316-np25-rf20-pca (query)           36_818.97     1_546.66    38_365.63       0.7696          1.1118        10.04
IVF-Binary-1024-nl316-pca (self)                      36_818.97     4_614.22    41_433.19       0.6343          1.2333        10.04
IVF-Binary-768-nl158-np7-rf0-sign (query)              2_939.96       569.08     3_509.03       0.1087          4.9638         5.04
IVF-Binary-768-nl158-np12-rf0-sign (query)             2_939.96       636.71     3_576.66       0.0910          5.6532         5.04
IVF-Binary-768-nl158-np17-rf0-sign (query)             2_939.96       725.79     3_665.75       0.0835          6.4783         5.04
IVF-Binary-768-nl158-np7-rf10-sign (query)             2_939.96       638.94     3_578.90       0.3875          1.9344         5.04
IVF-Binary-768-nl158-np7-rf20-sign (query)             2_939.96     1_133.62     4_073.58       0.4922          1.6906         5.04
IVF-Binary-768-nl158-np12-rf10-sign (query)            2_939.96       694.29     3_634.25       0.3241          2.2103         5.04
IVF-Binary-768-nl158-np12-rf20-sign (query)            2_939.96     1_189.32     4_129.28       0.4045          1.9478         5.04
IVF-Binary-768-nl158-np17-rf10-sign (query)            2_939.96       748.69     3_688.65       0.2968          2.3464         5.04
IVF-Binary-768-nl158-np17-rf20-sign (query)            2_939.96     1_274.68     4_214.63       0.3625          2.1099         5.04
IVF-Binary-768-nl158-sign (self)                       2_939.96     2_040.70     4_980.66       0.3246          2.2070         5.04
IVF-Binary-768-nl223-np11-rf0-sign (query)             1_254.23       606.22     1_860.45       0.1024          4.9876         5.23
IVF-Binary-768-nl223-np14-rf0-sign (query)             1_254.23       639.67     1_893.90       0.0957          5.3044         5.23
IVF-Binary-768-nl223-np21-rf0-sign (query)             1_254.23       707.36     1_961.59       0.0884          6.0740         5.23
IVF-Binary-768-nl223-np11-rf10-sign (query)            1_254.23       676.45     1_930.68       0.3639          1.9436         5.23
IVF-Binary-768-nl223-np11-rf20-sign (query)            1_254.23     1_158.97     2_413.20       0.4695          1.6779         5.23
IVF-Binary-768-nl223-np14-rf10-sign (query)            1_254.23       709.02     1_963.25       0.3385          2.0460         5.23
IVF-Binary-768-nl223-np14-rf20-sign (query)            1_254.23     1_198.06     2_452.29       0.4321          1.7706         5.23
IVF-Binary-768-nl223-np21-rf10-sign (query)            1_254.23       774.20     2_028.43       0.3021          2.2125         5.23
IVF-Binary-768-nl223-np21-rf20-sign (query)            1_254.23     1_283.92     2_538.15       0.3777          1.9467         5.23
IVF-Binary-768-nl223-sign (self)                       1_254.23     2_073.45     3_327.69       0.3378          2.0482         5.23
IVF-Binary-768-nl316-np15-rf0-sign (query)             1_661.68       656.40     2_318.07       0.0933          5.0942         5.51
IVF-Binary-768-nl316-np17-rf0-sign (query)             1_661.68       665.32     2_327.00       0.0905          5.2553         5.51
IVF-Binary-768-nl316-np25-rf0-sign (query)             1_661.68       733.35     2_395.03       0.0847          5.8642         5.51
IVF-Binary-768-nl316-np15-rf10-sign (query)            1_661.68       717.19     2_378.87       0.3578          1.8929         5.51
IVF-Binary-768-nl316-np15-rf20-sign (query)            1_661.68     1_204.12     2_865.80       0.4584          1.6471         5.51
IVF-Binary-768-nl316-np17-rf10-sign (query)            1_661.68       731.50     2_393.17       0.3461          1.9342         5.51
IVF-Binary-768-nl316-np17-rf20-sign (query)            1_661.68     1_234.83     2_896.51       0.4395          1.6901         5.51
IVF-Binary-768-nl316-np25-rf10-sign (query)            1_661.68       801.91     2_463.58       0.3116          2.0759         5.51
IVF-Binary-768-nl316-np25-rf20-sign (query)            1_661.68     1_313.06     2_974.74       0.3881          1.8299         5.51
IVF-Binary-768-nl316-sign (self)                       1_661.68     2_191.10     3_852.77       0.3468          1.9292         5.51
-----------------------------------------------------------------------------------------------------------------------------------
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
===================================================================================================================================
Benchmark: 50k samples, 256D - IVF-RaBitQ
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        33.46       675.89       709.35       1.0000          1.0000        48.83
Exhaustive (self)                                         33.46     2_177.14     2_210.60       1.0000          1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             993.13     1_042.50     2_035.63       0.5172          1.0357         2.84
ExhaustiveRaBitQ-rf5 (query)                             993.13     1_107.67     2_100.80       0.9122          1.0018         2.84
ExhaustiveRaBitQ-rf10 (query)                            993.13     1_175.15     2_168.28       0.9733          1.0003         2.84
ExhaustiveRaBitQ-rf20 (query)                            993.13     1_282.34     2_275.47       0.9876          1.0000         2.84
ExhaustiveRaBitQ (self)                                  993.13     3_864.92     4_858.06       0.9740          1.0003         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_443.71       281.27     1_724.98       0.5211          1.0351         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_443.71       447.08     1_890.79       0.5210          1.0351         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_443.71       618.45     2_062.16       0.5210          1.0351         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_443.71       363.88     1_807.59       0.9729          1.0003         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_443.71       456.24     1_899.95       0.9866          1.0000         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_443.71       534.55     1_978.26       0.9737          1.0003         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_443.71       599.45     2_043.16       0.9876          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_443.71       700.28     2_143.99       0.9737          1.0003         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_443.71       768.76     2_212.47       0.9876          1.0000         2.89
IVF-RaBitQ-nl158 (self)                                1_443.71     2_530.01     3_973.72       0.9881          1.0000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                        896.34       340.99     1_237.33       0.5225          1.0348         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                        896.34       426.58     1_322.92       0.5225          1.0348         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                        896.34       611.09     1_507.43       0.5224          1.0348         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                       896.34       423.02     1_319.36       0.9734          1.0003         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                       896.34       502.53     1_398.87       0.9872          1.0000         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                       896.34       510.57     1_406.91       0.9738          1.0003         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                       896.34       595.08     1_491.42       0.9877          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                       896.34       688.59     1_584.92       0.9738          1.0003         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                       896.34       773.51     1_669.85       0.9878          1.0000         2.95
IVF-RaBitQ-nl223 (self)                                  896.34     2_492.26     3_388.60       0.9881          1.0000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_047.54       377.74     1_425.28       0.5257          1.0343         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_047.54       433.03     1_480.57       0.5257          1.0343         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_047.54       605.91     1_653.45       0.5256          1.0344         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_047.54       486.53     1_534.06       0.9740          1.0003         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_047.54       530.66     1_578.20       0.9875          1.0000         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_047.54       514.48     1_562.01       0.9742          1.0003         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_047.54       581.75     1_629.29       0.9878          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_047.54       684.22     1_731.76       0.9742          1.0003         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_047.54       757.77     1_805.30       0.9878          1.0000         3.04
IVF-RaBitQ-nl316 (self)                                1_047.54     2_471.64     3_519.17       0.9882          1.0000         3.04
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D - IVF-RaBitQ
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.90     1_242.33     1_310.23       1.0000          1.0000        97.66
Exhaustive (self)                                         67.90     4_180.71     4_248.61       1.0000          1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           2_394.69     2_239.06     4_633.75       0.5134          1.0237         5.23
ExhaustiveRaBitQ-rf5 (query)                           2_394.69     2_290.72     4_685.41       0.9034          1.0013         5.23
ExhaustiveRaBitQ-rf10 (query)                          2_394.69     2_363.13     4_757.83       0.9628          1.0002         5.23
ExhaustiveRaBitQ-rf20 (query)                          2_394.69     2_480.20     4_874.90       0.9769          1.0000         5.23
ExhaustiveRaBitQ (self)                                2_394.69     7_812.58    10_207.27       0.9626          1.0002         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       3_286.65       623.75     3_910.41       0.5159          1.0234         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      3_286.65     1_040.40     4_327.06       0.5158          1.0235         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      3_286.65     1_450.05     4_736.71       0.5158          1.0235         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      3_286.65       734.68     4_021.33       0.9631          1.0002         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      3_286.65       828.87     4_115.52       0.9765          1.0000         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     3_286.65     1_141.03     4_427.68       0.9633          1.0002         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     3_286.65     1_230.57     4_517.22       0.9768          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     3_286.65     1_560.17     4_846.82       0.9633          1.0002         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     3_286.65     1_643.15     4_929.80       0.9768          1.0000         5.32
IVF-RaBitQ-nl158 (self)                                3_286.65     5_420.52     8_707.17       0.9765          1.0000         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      2_192.55       826.20     3_018.75       0.5165          1.0234         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      2_192.55     1_035.70     3_228.25       0.5166          1.0234         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      2_192.55     1_503.65     3_696.19       0.5165          1.0234         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     2_192.55       933.86     3_126.41       0.9626          1.0002         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     2_192.55     1_038.59     3_231.13       0.9758          1.0000         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     2_192.55     1_193.34     3_385.89       0.9633          1.0002         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     2_192.55     1_415.16     3_607.71       0.9767          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     2_192.55     1_756.83     3_949.37       0.9634          1.0002         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     2_192.55     1_780.82     3_973.37       0.9768          1.0000         5.44
IVF-RaBitQ-nl223 (self)                                2_192.55     6_051.06     8_243.61       0.9765          1.0000         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      2_596.00       976.74     3_572.74       0.5182          1.0232         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      2_596.00     1_088.66     3_684.65       0.5181          1.0232         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      2_596.00     1_574.00     4_169.99       0.5180          1.0232         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     2_596.00     1_077.12     3_673.11       0.9630          1.0002         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     2_596.00     1_174.18     3_770.18       0.9761          1.0000         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     2_596.00     1_191.92     3_787.91       0.9634          1.0002         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     2_596.00     1_285.45     3_881.45       0.9766          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     2_596.00     1_681.08     4_277.08       0.9636          1.0002         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     2_596.00     1_787.69     4_383.69       0.9769          1.0000         5.63
IVF-RaBitQ-nl316 (self)                                2_596.00     5_806.29     8_402.29       0.9766          1.0000         5.63
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D - IVF-RaBitQ
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        99.69     1_767.97     1_867.66       1.0000          1.0000       146.48
Exhaustive (self)                                         99.69     5_847.45     5_947.14       1.0000          1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           4_284.17     3_864.16     8_148.33       0.5106          1.0189         8.11
ExhaustiveRaBitQ-rf5 (query)                           4_284.17     3_962.13     8_246.30       0.8948          1.0010         8.11
ExhaustiveRaBitQ-rf10 (query)                          4_284.17     4_039.80     8_323.97       0.9531          1.0001         8.11
ExhaustiveRaBitQ-rf20 (query)                          4_284.17     4_274.74     8_558.91       0.9661          1.0000         8.11
ExhaustiveRaBitQ (self)                                4_284.17    13_665.04    17_949.21       0.9530          1.0002         8.11
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_599.89     1_117.96     6_717.85       0.5133          1.0187         8.25
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_599.89     1_768.99     7_368.88       0.5131          1.0187         8.25
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_599.89     2_489.17     8_089.06       0.5131          1.0187         8.25
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_599.89     1_227.87     6_827.75       0.9520          1.0002         8.25
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_599.89     1_342.18     6_942.07       0.9651          1.0000         8.25
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_599.89     1_890.57     7_490.46       0.9529          1.0001         8.25
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_599.89     2_007.62     7_607.51       0.9661          1.0000         8.25
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_599.89     2_610.20     8_210.08       0.9529          1.0001         8.25
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_599.89     2_719.89     8_319.77       0.9661          1.0000         8.25
IVF-RaBitQ-nl158 (self)                                5_599.89     8_994.93    14_594.81       0.9663          1.0000         8.25
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_912.92     1_555.54     5_468.46       0.5142          1.0186         8.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_912.92     1_921.67     5_834.59       0.5141          1.0186         8.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_912.92     2_860.37     6_773.29       0.5141          1.0186         8.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_912.92     1_684.39     5_597.31       0.9533          1.0001         8.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_912.92     1_788.83     5_701.75       0.9657          1.0000         8.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_912.92     2_037.90     5_950.82       0.9536          1.0001         8.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_912.92     2_150.88     6_063.80       0.9662          1.0000         8.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_912.92     3_003.83     6_916.75       0.9536          1.0001         8.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_912.92     3_084.11     6_997.03       0.9662          1.0000         8.44
IVF-RaBitQ-nl223 (self)                                3_912.92    10_215.60    14_128.52       0.9664          1.0000         8.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      4_470.31     1_896.18     6_366.49       0.5150          1.0185         8.71
IVF-RaBitQ-nl316-np17-rf0 (query)                      4_470.31     2_114.26     6_584.57       0.5150          1.0185         8.71
IVF-RaBitQ-nl316-np25-rf0 (query)                      4_470.31     3_069.34     7_539.65       0.5149          1.0185         8.71
IVF-RaBitQ-nl316-np15-rf10 (query)                     4_470.31     2_005.50     6_475.80       0.9536          1.0001         8.71
IVF-RaBitQ-nl316-np15-rf20 (query)                     4_470.31     2_104.93     6_575.24       0.9659          1.0000         8.71
IVF-RaBitQ-nl316-np17-rf10 (query)                     4_470.31     2_236.00     6_706.31       0.9539          1.0001         8.71
IVF-RaBitQ-nl316-np17-rf20 (query)                     4_470.31     2_337.77     6_808.08       0.9661          1.0000         8.71
IVF-RaBitQ-nl316-np25-rf10 (query)                     4_470.31     3_172.44     7_642.75       0.9540          1.0001         8.71
IVF-RaBitQ-nl316-np25-rf20 (query)                     4_470.31     3_308.26     7_778.56       0.9663          1.0000         8.71
IVF-RaBitQ-nl316 (self)                                4_470.31    10_900.64    15_370.95       0.9664          1.0000         8.71
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D - IVF-RaBitQ
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        33.33       685.15       718.49       1.0000          1.0000        48.83
Exhaustive (self)                                         33.33     2_347.14     2_380.47       1.0000          1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                             985.25       902.37     1_887.62       0.7286          1.0245         2.84
ExhaustiveRaBitQ-rf5 (query)                             985.25       974.30     1_959.56       0.9947          1.0001         2.84
ExhaustiveRaBitQ-rf10 (query)                            985.25     1_005.09     1_990.34       0.9976          1.0000         2.84
ExhaustiveRaBitQ-rf20 (query)                            985.25     1_132.34     2_117.59       0.9977          1.0000         2.84
ExhaustiveRaBitQ (self)                                  985.25     3_373.01     4_358.26       0.9977          1.0000         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_403.06       245.72     1_648.78       0.7297          1.0243         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_403.06       338.19     1_741.25       0.7297          1.0243         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_403.06       441.40     1_844.46       0.7297          1.0243         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_403.06       352.07     1_755.13       0.9976          1.0000         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_403.06       429.93     1_832.99       0.9977          1.0000         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_403.06       444.56     1_847.62       0.9976          1.0000         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_403.06       514.46     1_917.52       0.9977          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_403.06       535.11     1_938.17       0.9976          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_403.06       611.03     2_014.09       0.9977          1.0000         2.89
IVF-RaBitQ-nl158 (self)                                1_403.06     2_147.48     3_550.54       0.9977          1.0000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                        984.28       310.08     1_294.37       0.7341          1.0236         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                        984.28       367.55     1_351.83       0.7341          1.0236         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                        984.28       524.07     1_508.36       0.7341          1.0236         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                       984.28       399.67     1_383.95       0.9976          1.0000         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                       984.28       479.88     1_464.17       0.9977          1.0000         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                       984.28       452.92     1_437.21       0.9976          1.0000         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                       984.28       534.31     1_518.59       0.9977          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                       984.28       618.30     1_602.59       0.9976          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                       984.28       687.81     1_672.09       0.9977          1.0000         2.95
IVF-RaBitQ-nl223 (self)                                  984.28     2_264.97     3_249.25       0.9977          1.0000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                      1_180.34       360.50     1_540.84       0.7371          1.0230         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                      1_180.34       401.98     1_582.32       0.7371          1.0230         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                      1_180.34       562.24     1_742.58       0.7371          1.0230         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                     1_180.34       448.08     1_628.42       0.9976          1.0000         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                     1_180.34       517.04     1_697.38       0.9977          1.0000         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                     1_180.34       480.36     1_660.70       0.9976          1.0000         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                     1_180.34       556.08     1_736.42       0.9977          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                     1_180.34       676.85     1_857.19       0.9976          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                     1_180.34       691.91     1_872.25       0.9977          1.0000         3.04
IVF-RaBitQ-nl316 (self)                                1_180.34     2_280.46     3_460.80       0.9977          1.0000         3.04
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D - IVF-RaBitQ
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.77     1_221.03     1_288.81       1.0000          1.0000        97.66
Exhaustive (self)                                         67.77     4_002.55     4_070.32       1.0000          1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           2_107.16     2_062.16     4_169.32       0.7429          1.0147         5.23
ExhaustiveRaBitQ-rf5 (query)                           2_107.16     2_108.94     4_216.10       0.9905          1.0000         5.23
ExhaustiveRaBitQ-rf10 (query)                          2_107.16     2_172.94     4_280.09       0.9923          1.0000         5.23
ExhaustiveRaBitQ-rf20 (query)                          2_107.16     2_279.86     4_387.02       0.9923          1.0000         5.23
ExhaustiveRaBitQ (self)                                2_107.16     7_198.50     9_305.66       0.9923          1.0000         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       2_893.10       598.63     3_491.73       0.7437          1.0145         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      2_893.10       809.39     3_702.49       0.7437          1.0145         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      2_893.10     1_068.95     3_962.05       0.7437          1.0145         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      2_893.10       693.06     3_586.16       0.9923          1.0000         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      2_893.10       792.61     3_685.71       0.9923          1.0000         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     2_893.10       927.52     3_820.62       0.9923          1.0000         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     2_893.10     1_023.22     3_916.31       0.9923          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     2_893.10     1_186.13     4_079.22       0.9923          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     2_893.10     1_279.91     4_173.01       0.9923          1.0000         5.32
IVF-RaBitQ-nl158 (self)                                2_893.10     4_228.39     7_121.49       0.9923          1.0000         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      2_232.20       776.08     3_008.28       0.7464          1.0142         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      2_232.20       935.96     3_168.16       0.7467          1.0142         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      2_232.20     1_339.16     3_571.36       0.7467          1.0142         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     2_232.20       890.88     3_123.08       0.9912          1.0000         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     2_232.20       981.47     3_213.67       0.9912          1.0000         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     2_232.20     1_038.60     3_270.80       0.9923          1.0000         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     2_232.20     1_135.05     3_367.25       0.9923          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     2_232.20     1_446.78     3_678.98       0.9923          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     2_232.20     1_526.10     3_758.30       0.9923          1.0000         5.44
IVF-RaBitQ-nl223 (self)                                2_232.20     5_049.43     7_281.63       0.9923          1.0000         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      2_631.08       955.03     3_586.11       0.7473          1.0141         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      2_631.08     1_054.96     3_686.05       0.7477          1.0141         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      2_631.08     1_476.74     4_107.82       0.7478          1.0140         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     2_631.08     1_060.71     3_691.79       0.9909          1.0001         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     2_631.08     1_154.26     3_785.35       0.9910          1.0001         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     2_631.08     1_175.08     3_806.16       0.9918          1.0000         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     2_631.08     1_251.64     3_882.73       0.9919          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     2_631.08     1_578.34     4_209.43       0.9923          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     2_631.08     1_665.63     4_296.72       0.9923          1.0000         5.63
IVF-RaBitQ-nl316 (self)                                2_631.08     5_539.39     8_170.47       0.9923          1.0000         5.63
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D - IVF-RaBitQ
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        99.58     1_730.89     1_830.46       1.0000          1.0000       146.48
Exhaustive (self)                                         99.58     5_666.12     5_765.70       1.0000          1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           3_892.31     3_518.96     7_411.27       0.7241          1.0120         8.11
ExhaustiveRaBitQ-rf5 (query)                           3_892.31     3_603.89     7_496.19       0.9743          1.0000         8.11
ExhaustiveRaBitQ-rf10 (query)                          3_892.31     3_672.18     7_564.48       0.9771          0.9999         8.11
ExhaustiveRaBitQ-rf20 (query)                          3_892.31     3_799.33     7_691.63       0.9772          0.9999         8.11
ExhaustiveRaBitQ (self)                                3_892.31    12_153.20    16_045.51       0.9772          0.9999         8.11
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_001.55     1_028.64     6_030.19       0.7258          1.0119         8.25
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_001.55     1_532.33     6_533.88       0.7258          1.0119         8.25
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_001.55     2_028.79     7_030.34       0.7258          1.0119         8.25
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_001.55     1_169.23     6_170.77       0.9771          0.9999         8.25
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_001.55     1_308.89     6_310.43       0.9772          0.9999         8.25
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_001.55     1_661.99     6_663.54       0.9771          0.9999         8.25
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_001.55     1_781.05     6_782.60       0.9772          0.9999         8.25
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_001.55     2_161.26     7_162.80       0.9771          0.9999         8.25
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_001.55     2_279.45     7_281.00       0.9772          0.9999         8.25
IVF-RaBitQ-nl158 (self)                                5_001.55     7_568.80    12_570.34       0.9772          0.9999         8.25
IVF-RaBitQ-nl223-np11-rf0 (query)                      4_028.32     1_420.41     5_448.73       0.7273          1.0117         8.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      4_028.32     1_723.47     5_751.79       0.7273          1.0117         8.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      4_028.32     2_463.92     6_492.24       0.7273          1.0117         8.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     4_028.32     1_554.05     5_582.37       0.9770          0.9999         8.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     4_028.32     1_685.10     5_713.41       0.9770          0.9999         8.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     4_028.32     1_860.94     5_889.26       0.9771          0.9999         8.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     4_028.32     1_983.81     6_012.13       0.9772          0.9999         8.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     4_028.32     2_601.13     6_629.45       0.9771          0.9999         8.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     4_028.32     2_708.34     6_736.66       0.9772          0.9999         8.44
IVF-RaBitQ-nl223 (self)                                4_028.32     8_972.85    13_001.17       0.9772          0.9999         8.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      4_653.40     1_809.20     6_462.60       0.7281          1.0116         8.71
IVF-RaBitQ-nl316-np17-rf0 (query)                      4_653.40     2_014.18     6_667.59       0.7283          1.0116         8.71
IVF-RaBitQ-nl316-np25-rf0 (query)                      4_653.40     2_851.09     7_504.49       0.7283          1.0116         8.71
IVF-RaBitQ-nl316-np15-rf10 (query)                     4_653.40     1_939.24     6_592.65       0.9767          1.0000         8.71
IVF-RaBitQ-nl316-np15-rf20 (query)                     4_653.40     2_051.32     6_704.72       0.9767          1.0000         8.71
IVF-RaBitQ-nl316-np17-rf10 (query)                     4_653.40     2_149.04     6_802.45       0.9770          0.9999         8.71
IVF-RaBitQ-nl316-np17-rf20 (query)                     4_653.40     2_257.40     6_910.80       0.9771          0.9999         8.71
IVF-RaBitQ-nl316-np25-rf10 (query)                     4_653.40     3_001.04     7_654.45       0.9771          0.9999         8.71
IVF-RaBitQ-nl316-np25-rf20 (query)                     4_653.40     3_066.37     7_719.77       0.9772          0.9999         8.71
IVF-RaBitQ-nl316 (self)                                4_653.40    10_143.49    14_796.90       0.9772          0.9999         8.71
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Cell embeddings

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D - IVF-RaBitQ
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.46       696.63       729.09       1.0000          1.0000        48.83
Exhaustive (self)                                         32.46     2_308.73     2_341.19       1.0000          1.0000        48.83
ExhaustiveRaBitQ-rf0 (query)                           1_017.36     1_268.56     2_285.93       0.8680          1.0296         2.84
ExhaustiveRaBitQ-rf5 (query)                           1_017.36     1_320.89     2_338.25       1.0000          1.0000         2.84
ExhaustiveRaBitQ-rf10 (query)                          1_017.36     1_386.29     2_403.65       1.0000          1.0000         2.84
ExhaustiveRaBitQ-rf20 (query)                          1_017.36     1_514.31     2_531.67       1.0000          1.0000         2.84
ExhaustiveRaBitQ (self)                                1_017.36     4_598.71     5_616.07       1.0000          1.0000         2.84
IVF-RaBitQ-nl158-np7-rf0 (query)                       1_479.38       333.63     1_813.01       0.8728          1.0278         2.89
IVF-RaBitQ-nl158-np12-rf0 (query)                      1_479.38       553.72     2_033.10       0.8733          1.0275         2.89
IVF-RaBitQ-nl158-np17-rf0 (query)                      1_479.38       797.23     2_276.61       0.8733          1.0275         2.89
IVF-RaBitQ-nl158-np7-rf10 (query)                      1_479.38       433.47     1_912.84       0.9976          1.0005         2.89
IVF-RaBitQ-nl158-np7-rf20 (query)                      1_479.38       518.03     1_997.41       0.9976          1.0005         2.89
IVF-RaBitQ-nl158-np12-rf10 (query)                     1_479.38       641.42     2_120.80       0.9999          1.0000         2.89
IVF-RaBitQ-nl158-np12-rf20 (query)                     1_479.38       758.69     2_238.07       0.9999          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf10 (query)                     1_479.38       854.82     2_334.20       1.0000          1.0000         2.89
IVF-RaBitQ-nl158-np17-rf20 (query)                     1_479.38       941.90     2_421.28       1.0000          1.0000         2.89
IVF-RaBitQ-nl158 (self)                                1_479.38     3_095.86     4_575.24       1.0000          1.0000         2.89
IVF-RaBitQ-nl223-np11-rf0 (query)                        835.14       364.49     1_199.63       0.8833          1.0228         2.95
IVF-RaBitQ-nl223-np14-rf0 (query)                        835.14       453.19     1_288.33       0.8834          1.0227         2.95
IVF-RaBitQ-nl223-np21-rf0 (query)                        835.14       675.24     1_510.38       0.8833          1.0228         2.95
IVF-RaBitQ-nl223-np11-rf10 (query)                       835.14       457.77     1_292.92       0.9994          1.0001         2.95
IVF-RaBitQ-nl223-np11-rf20 (query)                       835.14       531.84     1_366.98       0.9994          1.0001         2.95
IVF-RaBitQ-nl223-np14-rf10 (query)                       835.14       544.08     1_379.23       0.9999          1.0000         2.95
IVF-RaBitQ-nl223-np14-rf20 (query)                       835.14       626.42     1_461.57       0.9999          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf10 (query)                       835.14       752.02     1_587.17       1.0000          1.0000         2.95
IVF-RaBitQ-nl223-np21-rf20 (query)                       835.14       834.37     1_669.51       1.0000          1.0000         2.95
IVF-RaBitQ-nl223 (self)                                  835.14     2_737.86     3_573.00       1.0000          1.0000         2.95
IVF-RaBitQ-nl316-np15-rf0 (query)                        996.10       401.52     1_397.61       0.8893          1.0202         3.04
IVF-RaBitQ-nl316-np17-rf0 (query)                        996.10       452.23     1_448.33       0.8893          1.0202         3.04
IVF-RaBitQ-nl316-np25-rf0 (query)                        996.10       654.97     1_651.07       0.8893          1.0202         3.04
IVF-RaBitQ-nl316-np15-rf10 (query)                       996.10       488.32     1_484.42       0.9997          1.0000         3.04
IVF-RaBitQ-nl316-np15-rf20 (query)                       996.10       561.78     1_557.88       0.9997          1.0000         3.04
IVF-RaBitQ-nl316-np17-rf10 (query)                       996.10       534.43     1_530.53       0.9998          1.0000         3.04
IVF-RaBitQ-nl316-np17-rf20 (query)                       996.10       624.30     1_620.39       0.9998          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf10 (query)                       996.10       733.00     1_729.10       1.0000          1.0000         3.04
IVF-RaBitQ-nl316-np25-rf20 (query)                       996.10       806.40     1_802.50       1.0000          1.0000         3.04
IVF-RaBitQ-nl316 (self)                                  996.10     2_654.06     3_650.16       1.0000          1.0000         3.04
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D - IVF-RaBitQ
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        69.40     1_232.83     1_302.22       1.0000          1.0000        97.66
Exhaustive (self)                                         69.40     4_068.46     4_137.86       1.0000          1.0000        97.66
ExhaustiveRaBitQ-rf0 (query)                           2_394.70     2_636.33     5_031.03       0.9024          1.0153         5.23
ExhaustiveRaBitQ-rf5 (query)                           2_394.70     2_675.62     5_070.32       1.0000          1.0000         5.23
ExhaustiveRaBitQ-rf10 (query)                          2_394.70     2_759.38     5_154.08       1.0000          1.0000         5.23
ExhaustiveRaBitQ-rf20 (query)                          2_394.70     2_899.20     5_293.91       1.0000          1.0000         5.23
ExhaustiveRaBitQ (self)                                2_394.70     9_122.62    11_517.32       1.0000          1.0000         5.23
IVF-RaBitQ-nl158-np7-rf0 (query)                       3_233.60       734.99     3_968.58       0.9068          1.0138         5.32
IVF-RaBitQ-nl158-np12-rf0 (query)                      3_233.60     1_205.68     4_439.27       0.9073          1.0135         5.32
IVF-RaBitQ-nl158-np17-rf0 (query)                      3_233.60     1_672.78     4_906.37       0.9073          1.0135         5.32
IVF-RaBitQ-nl158-np7-rf10 (query)                      3_233.60       841.00     4_074.60       0.9985          1.0003         5.32
IVF-RaBitQ-nl158-np7-rf20 (query)                      3_233.60       928.84     4_162.43       0.9985          1.0003         5.32
IVF-RaBitQ-nl158-np12-rf10 (query)                     3_233.60     1_303.68     4_537.27       0.9999          1.0000         5.32
IVF-RaBitQ-nl158-np12-rf20 (query)                     3_233.60     1_396.32     4_629.91       0.9999          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf10 (query)                     3_233.60     1_756.84     4_990.44       1.0000          1.0000         5.32
IVF-RaBitQ-nl158-np17-rf20 (query)                     3_233.60     1_881.78     5_115.38       1.0000          1.0000         5.32
IVF-RaBitQ-nl158 (self)                                3_233.60     6_124.04     9_357.64       1.0000          1.0000         5.32
IVF-RaBitQ-nl223-np11-rf0 (query)                      2_004.79       876.67     2_881.46       0.9151          1.0111         5.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      2_004.79     1_098.19     3_102.98       0.9152          1.0111         5.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      2_004.79     1_632.40     3_637.19       0.9152          1.0111         5.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     2_004.79       975.20     2_979.99       0.9997          1.0000         5.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     2_004.79     1_077.17     3_081.96       0.9997          1.0000         5.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     2_004.79     1_202.33     3_207.12       0.9999          1.0000         5.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     2_004.79     1_294.94     3_299.73       0.9999          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     2_004.79     1_803.87     3_808.66       1.0000          1.0000         5.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     2_004.79     1_831.82     3_836.61       1.0000          1.0000         5.44
IVF-RaBitQ-nl223 (self)                                2_004.79     6_010.38     8_015.17       1.0000          1.0000         5.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      2_264.91     1_031.02     3_295.92       0.9189          1.0100         5.63
IVF-RaBitQ-nl316-np17-rf0 (query)                      2_264.91     1_169.42     3_434.33       0.9189          1.0100         5.63
IVF-RaBitQ-nl316-np25-rf0 (query)                      2_264.91     1_692.81     3_957.71       0.9190          1.0100         5.63
IVF-RaBitQ-nl316-np15-rf10 (query)                     2_264.91     1_124.78     3_389.69       0.9998          1.0000         5.63
IVF-RaBitQ-nl316-np15-rf20 (query)                     2_264.91     1_211.45     3_476.36       0.9998          1.0000         5.63
IVF-RaBitQ-nl316-np17-rf10 (query)                     2_264.91     1_261.79     3_526.70       0.9999          1.0000         5.63
IVF-RaBitQ-nl316-np17-rf20 (query)                     2_264.91     1_339.55     3_604.45       0.9999          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf10 (query)                     2_264.91     1_789.64     4_054.55       1.0000          1.0000         5.63
IVF-RaBitQ-nl316-np25-rf20 (query)                     2_264.91     1_861.48     4_126.38       1.0000          1.0000         5.63
IVF-RaBitQ-nl316 (self)                                2_264.91     6_138.42     8_403.32       0.9999          1.0000         5.63
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D - IVF-RaBitQ
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        99.34     1_815.48     1_914.82       1.0000          1.0000       146.48
Exhaustive (self)                                         99.34     5_708.23     5_807.57       1.0000          1.0000       146.48
ExhaustiveRaBitQ-rf0 (query)                           4_296.72     4_346.94     8_643.66       0.9249          1.0085         8.11
ExhaustiveRaBitQ-rf5 (query)                           4_296.72     4_429.04     8_725.76       1.0000          1.0000         8.11
ExhaustiveRaBitQ-rf10 (query)                          4_296.72     4_510.94     8_807.66       1.0000          1.0000         8.11
ExhaustiveRaBitQ-rf20 (query)                          4_296.72     4_689.86     8_986.58       1.0000          1.0000         8.11
ExhaustiveRaBitQ (self)                                4_296.72    14_983.98    19_280.71       0.9999          1.0000         8.11
IVF-RaBitQ-nl158-np7-rf0 (query)                       5_454.95     1_266.82     6_721.77       0.9274          1.0078         8.25
IVF-RaBitQ-nl158-np12-rf0 (query)                      5_454.95     2_085.70     7_540.66       0.9276          1.0078         8.25
IVF-RaBitQ-nl158-np17-rf0 (query)                      5_454.95     2_918.06     8_373.02       0.9276          1.0078         8.25
IVF-RaBitQ-nl158-np7-rf10 (query)                      5_454.95     1_409.03     6_863.98       0.9995          1.0001         8.25
IVF-RaBitQ-nl158-np7-rf20 (query)                      5_454.95     1_500.06     6_955.01       0.9995          1.0001         8.25
IVF-RaBitQ-nl158-np12-rf10 (query)                     5_454.95     2_233.74     7_688.70       1.0000          1.0000         8.25
IVF-RaBitQ-nl158-np12-rf20 (query)                     5_454.95     2_325.62     7_780.58       1.0000          1.0000         8.25
IVF-RaBitQ-nl158-np17-rf10 (query)                     5_454.95     3_016.79     8_471.75       1.0000          1.0000         8.25
IVF-RaBitQ-nl158-np17-rf20 (query)                     5_454.95     3_129.47     8_584.42       1.0000          1.0000         8.25
IVF-RaBitQ-nl158 (self)                                5_454.95    10_426.30    15_881.26       0.9999          1.0000         8.25
IVF-RaBitQ-nl223-np11-rf0 (query)                      3_721.51     1_620.40     5_341.91       0.9323          1.0067         8.44
IVF-RaBitQ-nl223-np14-rf0 (query)                      3_721.51     2_048.17     5_769.68       0.9323          1.0067         8.44
IVF-RaBitQ-nl223-np21-rf0 (query)                      3_721.51     3_038.89     6_760.40       0.9323          1.0067         8.44
IVF-RaBitQ-nl223-np11-rf10 (query)                     3_721.51     1_739.23     5_460.74       0.9998          1.0000         8.44
IVF-RaBitQ-nl223-np11-rf20 (query)                     3_721.51     1_841.66     5_563.17       0.9998          1.0000         8.44
IVF-RaBitQ-nl223-np14-rf10 (query)                     3_721.51     2_148.65     5_870.16       0.9999          1.0000         8.44
IVF-RaBitQ-nl223-np14-rf20 (query)                     3_721.51     2_253.41     5_974.92       0.9999          1.0000         8.44
IVF-RaBitQ-nl223-np21-rf10 (query)                     3_721.51     3_125.17     6_846.69       1.0000          1.0000         8.44
IVF-RaBitQ-nl223-np21-rf20 (query)                     3_721.51     3_225.42     6_946.94       1.0000          1.0000         8.44
IVF-RaBitQ-nl223 (self)                                3_721.51    10_689.22    14_410.74       0.9999          1.0000         8.44
IVF-RaBitQ-nl316-np15-rf0 (query)                      4_084.53     1_931.76     6_016.29       0.9360          1.0060         8.71
IVF-RaBitQ-nl316-np17-rf0 (query)                      4_084.53     2_180.25     6_264.78       0.9360          1.0060         8.71
IVF-RaBitQ-nl316-np25-rf0 (query)                      4_084.53     3_215.19     7_299.72       0.9360          1.0060         8.71
IVF-RaBitQ-nl316-np15-rf10 (query)                     4_084.53     2_069.84     6_154.37       0.9999          1.0000         8.71
IVF-RaBitQ-nl316-np15-rf20 (query)                     4_084.53     2_168.68     6_253.21       0.9999          1.0000         8.71
IVF-RaBitQ-nl316-np17-rf10 (query)                     4_084.53     2_295.60     6_380.13       1.0000          1.0000         8.71
IVF-RaBitQ-nl316-np17-rf20 (query)                     4_084.53     2_407.01     6_491.54       1.0000          1.0000         8.71
IVF-RaBitQ-nl316-np25-rf10 (query)                     4_084.53     3_279.80     7_364.33       1.0000          1.0000         8.71
IVF-RaBitQ-nl316-np25-rf20 (query)                     4_084.53     3_406.33     7_490.86       1.0000          1.0000         8.71
IVF-RaBitQ-nl316 (self)                                4_084.53    11_258.08    15_342.61       0.9999          1.0000         8.71
-----------------------------------------------------------------------------------------------------------------------------------
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
===================================================================================================================================
Benchmark: 50k samples, 256D - TurboQuant + IVF
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.15       668.51       700.66       1.0000          1.0000        48.83
Exhaustive (self)                                         32.15     2_172.84     2_205.00       1.0000          1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              143.60       355.63       499.23       0.0109          1.5964         7.12
ExhaustiveTQ-b2-rf5 (query)                              143.60       436.57       580.17       0.0526          1.2562         7.12
ExhaustiveTQ-b2-rf10 (query)                             143.60       558.57       702.17       0.1030          1.1894         7.12
ExhaustiveTQ-b2-rf20 (query)                             143.60       931.07     1_074.66       0.2003          1.1318         7.12
ExhaustiveTQ-b2 (self)                                   143.60     3_125.76     3_269.36       0.1994          1.1335         7.12
ExhaustiveTQ-b4-rf0 (query)                              231.06       560.53       791.59       0.0131          1.5544        13.22
ExhaustiveTQ-b4-rf5 (query)                              231.06       641.81       872.87       0.0575          1.2376        13.22
ExhaustiveTQ-b4-rf10 (query)                             231.06       775.21     1_006.27       0.1078          1.1773        13.22
ExhaustiveTQ-b4-rf20 (query)                             231.06     1_187.20     1_418.26       0.2031          1.1256        13.22
ExhaustiveTQ-b4 (self)                                   231.06     3_968.99     4_200.05       0.2033          1.1266        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                          983.88       108.68     1_092.55       0.0117          1.5499         7.80
IVF-TQ-b2-nl158-np12-rf0 (query)                         983.88       141.90     1_125.77       0.0110          1.5928         7.80
IVF-TQ-b2-nl158-np17-rf0 (query)                         983.88       183.42     1_167.29       0.0109          1.5964         7.80
IVF-TQ-b2-nl158-np7-rf10 (query)                         983.88       269.84     1_253.72       0.1106          1.1789         7.80
IVF-TQ-b2-nl158-np7-rf20 (query)                         983.88       497.89     1_481.77       0.2159          1.1227         7.80
IVF-TQ-b2-nl158-np12-rf10 (query)                        983.88       318.04     1_301.92       0.1035          1.1886         7.80
IVF-TQ-b2-nl158-np12-rf20 (query)                        983.88       589.97     1_573.85       0.2012          1.1311         7.80
IVF-TQ-b2-nl158-np17-rf10 (query)                        983.88       365.40     1_349.27       0.1030          1.1894         7.80
IVF-TQ-b2-nl158-np17-rf20 (query)                        983.88       654.78     1_638.66       0.2003          1.1318         7.80
IVF-TQ-b2-nl158 (self)                                   983.88     1_285.26     2_269.13       0.1994          1.1335         7.80
IVF-TQ-b2-nl223-np11-rf0 (query)                         612.50       122.52       735.02       0.0114          1.5686         7.92
IVF-TQ-b2-nl223-np14-rf0 (query)                         612.50       134.70       747.20       0.0110          1.5909         7.92
IVF-TQ-b2-nl223-np21-rf0 (query)                         612.50       169.78       782.28       0.0109          1.5964         7.92
IVF-TQ-b2-nl223-np11-rf10 (query)                        612.50       296.39       908.89       0.1067          1.1837         7.92
IVF-TQ-b2-nl223-np11-rf20 (query)                        612.50       554.68     1_167.18       0.2081          1.1268         7.92
IVF-TQ-b2-nl223-np14-rf10 (query)                        612.50       318.61       931.11       0.1035          1.1885         7.92
IVF-TQ-b2-nl223-np14-rf20 (query)                        612.50       597.15     1_209.64       0.2014          1.1310         7.92
IVF-TQ-b2-nl223-np21-rf10 (query)                        612.50       371.88       984.38       0.1030          1.1894         7.92
IVF-TQ-b2-nl223-np21-rf20 (query)                        612.50       672.65     1_285.15       0.2003          1.1318         7.92
IVF-TQ-b2-nl223 (self)                                   612.50     1_228.66     1_841.16       0.1994          1.1335         7.92
IVF-TQ-b2-nl316-np15-rf0 (query)                         842.53       124.64       967.17       0.0114          1.5653         8.12
IVF-TQ-b2-nl316-np17-rf0 (query)                         842.53       131.97       974.49       0.0111          1.5820         8.12
IVF-TQ-b2-nl316-np25-rf0 (query)                         842.53       164.07     1_006.59       0.0109          1.5964         8.12
IVF-TQ-b2-nl316-np15-rf10 (query)                        842.53       289.68     1_132.21       0.1074          1.1830         8.12
IVF-TQ-b2-nl316-np15-rf20 (query)                        842.53       546.16     1_388.69       0.2092          1.1262         8.12
IVF-TQ-b2-nl316-np17-rf10 (query)                        842.53       299.39     1_141.91       0.1049          1.1867         8.12
IVF-TQ-b2-nl316-np17-rf20 (query)                        842.53       563.70     1_406.23       0.2040          1.1294         8.12
IVF-TQ-b2-nl316-np25-rf10 (query)                        842.53       347.77     1_190.30       0.1030          1.1894         8.12
IVF-TQ-b2-nl316-np25-rf20 (query)                        842.53       639.93     1_482.46       0.2003          1.1318         8.12
IVF-TQ-b2-nl316 (self)                                   842.53     1_224.62     2_067.15       0.1994          1.1335         8.12
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_070.49       148.10     1_218.60       0.0140          1.5215        14.06
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_070.49       206.86     1_277.36       0.0132          1.5494        14.06
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_070.49       257.64     1_328.13       0.0131          1.5544        14.06
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_070.49       320.02     1_390.51       0.1158          1.1693        14.06
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_070.49       555.77     1_626.27       0.2187          1.1184        14.06
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_070.49       395.41     1_465.90       0.1082          1.1766        14.06
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_070.49       663.70     1_734.19       0.2041          1.1250        14.06
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_070.49       450.34     1_520.83       0.1078          1.1773        14.06
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_070.49       756.97     1_827.47       0.2031          1.1256        14.06
IVF-TQ-b4-nl158 (self)                                 1_070.49     1_372.82     2_443.32       0.2033          1.1266        14.06
IVF-TQ-b4-nl223-np11-rf0 (query)                         712.91       168.06       880.96       0.0137          1.5299        14.23
IVF-TQ-b4-nl223-np14-rf0 (query)                         712.91       189.30       902.21       0.0132          1.5510        14.23
IVF-TQ-b4-nl223-np21-rf0 (query)                         712.91       247.04       959.94       0.0131          1.5544        14.23
IVF-TQ-b4-nl223-np11-rf10 (query)                        712.91       348.95     1_061.86       0.1122          1.1723        14.23
IVF-TQ-b4-nl223-np11-rf20 (query)                        712.91       607.76     1_320.66       0.2117          1.1211        14.23
IVF-TQ-b4-nl223-np14-rf10 (query)                        712.91       380.69     1_093.60       0.1084          1.1766        14.23
IVF-TQ-b4-nl223-np14-rf20 (query)                        712.91       657.06     1_369.96       0.2041          1.1250        14.23
IVF-TQ-b4-nl223-np21-rf10 (query)                        712.91       452.53     1_165.44       0.1078          1.1773        14.23
IVF-TQ-b4-nl223-np21-rf20 (query)                        712.91       753.72     1_466.63       0.2031          1.1256        14.23
IVF-TQ-b4-nl223 (self)                                   712.91     1_377.51     2_090.42       0.2033          1.1266        14.23
IVF-TQ-b4-nl316-np15-rf0 (query)                         934.65       168.25     1_102.91       0.0137          1.5242        14.53
IVF-TQ-b4-nl316-np17-rf0 (query)                         934.65       178.78     1_113.43       0.0134          1.5390        14.53
IVF-TQ-b4-nl316-np25-rf0 (query)                         934.65       231.34     1_165.99       0.0131          1.5544        14.53
IVF-TQ-b4-nl316-np15-rf10 (query)                        934.65       338.71     1_273.37       0.1129          1.1713        14.53
IVF-TQ-b4-nl316-np15-rf20 (query)                        934.65       592.74     1_527.39       0.2125          1.1204        14.53
IVF-TQ-b4-nl316-np17-rf10 (query)                        934.65       372.52     1_307.17       0.1102          1.1743        14.53
IVF-TQ-b4-nl316-np17-rf20 (query)                        934.65       612.97     1_547.62       0.2075          1.1232        14.53
IVF-TQ-b4-nl316-np25-rf10 (query)                        934.65       422.29     1_356.95       0.1078          1.1773        14.53
IVF-TQ-b4-nl316-np25-rf20 (query)                        934.65       717.79     1_652.44       0.2031          1.1256        14.53
IVF-TQ-b4-nl316 (self)                                   934.65     1_348.57     2_283.22       0.2033          1.1266        14.53
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D - TurboQuant + IVF
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        67.99     1_281.15     1_349.14       1.0000          1.0000        97.66
Exhaustive (self)                                         67.99     4_326.66     4_394.66       1.0000          1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              366.28       631.87       998.15       0.0120          1.3610        13.97
ExhaustiveTQ-b2-rf5 (query)                              366.28       728.29     1_094.57       0.0560          1.1729        13.97
ExhaustiveTQ-b2-rf10 (query)                             366.28       884.77     1_251.06       0.1081          1.1302        13.97
ExhaustiveTQ-b2-rf20 (query)                             366.28     1_278.61     1_644.90       0.2057          1.0911        13.97
ExhaustiveTQ-b2 (self)                                   366.28     4_161.17     4_527.45       0.2053          1.0916        13.97
ExhaustiveTQ-b4-rf0 (query)                              505.57     1_112.21     1_617.79       0.0183          1.3439        26.18
ExhaustiveTQ-b4-rf5 (query)                              505.57     1_216.48     1_722.05       0.0633          1.1620        26.18
ExhaustiveTQ-b4-rf10 (query)                             505.57     1_355.46     1_861.04       0.1139          1.1229        26.18
ExhaustiveTQ-b4-rf20 (query)                             505.57     1_741.33     2_246.90       0.2060          1.0883        26.18
ExhaustiveTQ-b4 (self)                                   505.57     5_713.43     6_219.00       0.2070          1.0880        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        2_036.72       214.13     2_250.85       0.0125          1.3450        14.97
IVF-TQ-b2-nl158-np12-rf0 (query)                       2_036.72       262.55     2_299.27       0.0120          1.3610        14.97
IVF-TQ-b2-nl158-np17-rf0 (query)                       2_036.72       297.36     2_334.08       0.0120          1.3610        14.97
IVF-TQ-b2-nl158-np7-rf10 (query)                       2_036.72       395.67     2_432.39       0.1139          1.1257        14.97
IVF-TQ-b2-nl158-np7-rf20 (query)                       2_036.72       653.32     2_690.04       0.2176          1.0868        14.97
IVF-TQ-b2-nl158-np12-rf10 (query)                      2_036.72       455.10     2_491.83       0.1081          1.1302        14.97
IVF-TQ-b2-nl158-np12-rf20 (query)                      2_036.72       735.94     2_772.66       0.2057          1.0911        14.97
IVF-TQ-b2-nl158-np17-rf10 (query)                      2_036.72       515.58     2_552.30       0.1081          1.1302        14.97
IVF-TQ-b2-nl158-np17-rf20 (query)                      2_036.72       817.80     2_854.52       0.2057          1.0911        14.97
IVF-TQ-b2-nl158 (self)                                 2_036.72     1_632.97     3_669.69       0.2053          1.0916        14.97
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_250.86       220.07     1_470.93       0.0123          1.3536        15.19
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_250.86       245.13     1_495.99       0.0120          1.3596        15.19
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_250.86       297.68     1_548.54       0.0120          1.3610        15.19
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_250.86       407.53     1_658.39       0.1112          1.1281        15.19
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_250.86       681.17     1_932.03       0.2118          1.0891        15.19
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_250.86       435.30     1_686.16       0.1085          1.1299        15.19
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_250.86       742.17     1_993.03       0.2066          1.0908        15.19
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_250.86       509.10     1_759.97       0.1081          1.1302        15.19
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_250.86       825.63     2_076.49       0.2057          1.0911        15.19
IVF-TQ-b2-nl223 (self)                                 1_250.86     1_719.44     2_970.30       0.2053          1.0916        15.19
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_555.58       225.72     1_781.30       0.0123          1.3502        15.59
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_555.58       247.79     1_803.37       0.0121          1.3577        15.59
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_555.58       288.22     1_843.81       0.0120          1.3610        15.59
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_555.58       419.71     1_975.29       0.1119          1.1273        15.59
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_555.58       702.61     2_258.19       0.2133          1.0883        15.59
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_555.58       488.67     2_044.25       0.1094          1.1293        15.59
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_555.58       861.69     2_417.27       0.2085          1.0902        15.59
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_555.58       544.57     2_100.15       0.1081          1.1302        15.59
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_555.58       879.25     2_434.83       0.2057          1.0911        15.59
IVF-TQ-b2-nl316 (self)                                 1_555.58     1_693.77     3_249.35       0.2053          1.0916        15.59
IVF-TQ-b4-nl158-np7-rf0 (query)                        2_453.60       285.85     2_739.45       0.0190          1.3289        27.48
IVF-TQ-b4-nl158-np12-rf0 (query)                       2_453.60       394.14     2_847.74       0.0183          1.3439        27.48
IVF-TQ-b4-nl158-np17-rf0 (query)                       2_453.60       466.99     2_920.59       0.0183          1.3439        27.48
IVF-TQ-b4-nl158-np7-rf10 (query)                       2_453.60       493.55     2_947.15       0.1205          1.1185        27.48
IVF-TQ-b4-nl158-np7-rf20 (query)                       2_453.60       760.09     3_213.69       0.2184          1.0843        27.48
IVF-TQ-b4-nl158-np12-rf10 (query)                      2_453.60       642.60     3_096.20       0.1140          1.1229        27.48
IVF-TQ-b4-nl158-np12-rf20 (query)                      2_453.60       889.69     3_343.29       0.2061          1.0883        27.48
IVF-TQ-b4-nl158-np17-rf10 (query)                      2_453.60       690.54     3_144.14       0.1140          1.1229        27.48
IVF-TQ-b4-nl158-np17-rf20 (query)                      2_453.60     1_006.03     3_459.63       0.2060          1.0883        27.48
IVF-TQ-b4-nl158 (self)                                 2_453.60     1_910.19     4_363.79       0.2069          1.0880        27.48
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_387.20       313.78     1_700.98       0.0186          1.3359        27.81
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_387.20       352.88     1_740.08       0.0183          1.3423        27.81
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_387.20       455.84     1_843.04       0.0183          1.3439        27.81
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_387.20       511.59     1_898.79       0.1170          1.1210        27.81
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_387.20       796.43     2_183.63       0.2120          1.0864        27.81
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_387.20       569.42     1_956.62       0.1143          1.1226        27.81
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_387.20       873.71     2_260.91       0.2069          1.0880        27.81
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_387.20       676.93     2_064.13       0.1140          1.1229        27.81
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_387.20     1_002.98     2_390.18       0.2061          1.0883        27.81
IVF-TQ-b4-nl223 (self)                                 1_387.20     1_918.23     3_305.43       0.2069          1.0880        27.81
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_699.27       322.32     2_021.58       0.0187          1.3324        28.41
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_699.27       352.67     2_051.94       0.0184          1.3393        28.41
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_699.27       431.78     2_131.04       0.0183          1.3439        28.41
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_699.27       526.88     2_226.15       0.1178          1.1203        28.41
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_699.27       861.72     2_560.99       0.2141          1.0857        28.41
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_699.27       545.31     2_244.57       0.1153          1.1220        28.41
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_699.27       854.56     2_553.83       0.2090          1.0873        28.41
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_699.27       650.14     2_349.41       0.1140          1.1229        28.41
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_699.27       972.65     2_671.91       0.2060          1.0883        28.41
IVF-TQ-b4-nl316 (self)                                 1_699.27     1_924.00     3_623.26       0.2069          1.0880        28.41
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Correlated data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D - TurboQuant + IVF
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       108.22     1_895.94     2_004.16       1.0000          1.0000       146.48
Exhaustive (self)                                        108.22     6_422.32     6_530.54       1.0000          1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              607.34       950.01     1_557.35       0.0154          1.2813        21.33
ExhaustiveTQ-b2-rf5 (query)                              607.34     1_050.89     1_658.24       0.0627          1.1384        21.33
ExhaustiveTQ-b2-rf10 (query)                             607.34     1_201.61     1_808.95       0.1151          1.1036        21.33
ExhaustiveTQ-b2-rf20 (query)                             607.34     1_596.29     2_203.63       0.2128          1.0709        21.33
ExhaustiveTQ-b2 (self)                                   607.34     5_257.09     5_864.43       0.2130          1.0712        21.33
ExhaustiveTQ-b4-rf0 (query)                              753.96     1_764.71     2_518.67       0.0147          1.2811        39.64
ExhaustiveTQ-b4-rf5 (query)                              753.96     1_875.64     2_629.60       0.0559          1.1453        39.64
ExhaustiveTQ-b4-rf10 (query)                             753.96     1_991.03     2_744.99       0.1025          1.1153        39.64
ExhaustiveTQ-b4-rf20 (query)                             753.96     2_386.36     3_140.32       0.1925          1.0877        39.64
ExhaustiveTQ-b4 (self)                                   753.96     7_893.63     8_647.59       0.1917          1.0881        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        3_152.96       303.58     3_456.53       0.0162          1.2716        22.63
IVF-TQ-b2-nl158-np12-rf0 (query)                       3_152.96       372.31     3_525.27       0.0154          1.2813        22.63
IVF-TQ-b2-nl158-np17-rf0 (query)                       3_152.96       438.19     3_591.15       0.0154          1.2813        22.63
IVF-TQ-b2-nl158-np7-rf10 (query)                       3_152.96       513.28     3_666.24       0.1214          1.1003        22.63
IVF-TQ-b2-nl158-np7-rf20 (query)                       3_152.96       789.54     3_942.50       0.2245          1.0676        22.63
IVF-TQ-b2-nl158-np12-rf10 (query)                      3_152.96       598.57     3_751.52       0.1151          1.1036        22.63
IVF-TQ-b2-nl158-np12-rf20 (query)                      3_152.96       906.71     4_059.66       0.2128          1.0709        22.63
IVF-TQ-b2-nl158-np17-rf10 (query)                      3_152.96       700.46     3_853.42       0.1151          1.1036        22.63
IVF-TQ-b2-nl158-np17-rf20 (query)                      3_152.96     1_028.78     4_181.74       0.2128          1.0709        22.63
IVF-TQ-b2-nl158 (self)                                 3_152.96     2_072.98     5_225.94       0.2130          1.0712        22.63
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_939.07       319.57     2_258.64       0.0160          1.2698        22.99
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_939.07       348.28     2_287.34       0.0155          1.2782        22.99
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_939.07       413.27     2_352.34       0.0154          1.2813        22.99
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_939.07       544.01     2_483.07       0.1208          1.1002        22.99
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_939.07       850.08     2_789.15       0.2234          1.0678        22.99
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_939.07       581.96     2_521.02       0.1161          1.1029        22.99
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_939.07       893.47     2_832.54       0.2148          1.0703        22.99
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_939.07       642.48     2_581.54       0.1151          1.1036        22.99
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_939.07       999.31     2_938.38       0.2128          1.0709        22.99
IVF-TQ-b2-nl223 (self)                                 1_939.07     2_061.85     4_000.91       0.2130          1.0712        22.99
IVF-TQ-b2-nl316-np15-rf0 (query)                       2_501.59       332.55     2_834.13       0.0160          1.2708        23.52
IVF-TQ-b2-nl316-np17-rf0 (query)                       2_501.59       346.86     2_848.45       0.0157          1.2753        23.52
IVF-TQ-b2-nl316-np25-rf0 (query)                       2_501.59       415.02     2_916.61       0.0154          1.2813        23.52
IVF-TQ-b2-nl316-np15-rf10 (query)                      2_501.59       552.02     3_053.61       0.1202          1.1006        23.52
IVF-TQ-b2-nl316-np15-rf20 (query)                      2_501.59       854.52     3_356.11       0.2224          1.0682        23.52
IVF-TQ-b2-nl316-np17-rf10 (query)                      2_501.59       606.86     3_108.45       0.1176          1.1020        23.52
IVF-TQ-b2-nl316-np17-rf20 (query)                      2_501.59       876.78     3_378.37       0.2176          1.0694        23.52
IVF-TQ-b2-nl316-np25-rf10 (query)                      2_501.59       630.02     3_131.61       0.1151          1.1036        23.52
IVF-TQ-b2-nl316-np25-rf20 (query)                      2_501.59       951.16     3_452.75       0.2128          1.0709        23.52
IVF-TQ-b2-nl316 (self)                                 2_501.59     2_048.55     4_550.14       0.2130          1.0712        23.52
IVF-TQ-b4-nl158-np7-rf0 (query)                        3_151.63       435.92     3_587.55       0.0154          1.2718        41.40
IVF-TQ-b4-nl158-np12-rf0 (query)                       3_151.63       564.94     3_716.57       0.0147          1.2811        41.40
IVF-TQ-b4-nl158-np17-rf0 (query)                       3_151.63       684.86     3_836.50       0.0147          1.2811        41.40
IVF-TQ-b4-nl158-np7-rf10 (query)                       3_151.63       660.53     3_812.16       0.1084          1.1124        41.40
IVF-TQ-b4-nl158-np7-rf20 (query)                       3_151.63       945.84     4_097.47       0.2041          1.0850        41.40
IVF-TQ-b4-nl158-np12-rf10 (query)                      3_151.63       805.36     3_957.00       0.1025          1.1153        41.40
IVF-TQ-b4-nl158-np12-rf20 (query)                      3_151.63     1_121.00     4_272.64       0.1925          1.0877        41.40
IVF-TQ-b4-nl158-np17-rf10 (query)                      3_151.63       945.91     4_097.55       0.1025          1.1153        41.40
IVF-TQ-b4-nl158-np17-rf20 (query)                      3_151.63     1_294.25     4_445.88       0.1925          1.0877        41.40
IVF-TQ-b4-nl158 (self)                                 3_151.63     2_560.81     5_712.45       0.1917          1.0881        41.40
IVF-TQ-b4-nl223-np11-rf0 (query)                       2_072.07       467.05     2_539.12       0.0155          1.2699        41.92
IVF-TQ-b4-nl223-np14-rf0 (query)                       2_072.07       529.82     2_601.89       0.0149          1.2783        41.92
IVF-TQ-b4-nl223-np21-rf0 (query)                       2_072.07       653.30     2_725.37       0.0147          1.2811        41.92
IVF-TQ-b4-nl223-np11-rf10 (query)                      2_072.07       703.88     2_775.95       0.1075          1.1123        41.92
IVF-TQ-b4-nl223-np11-rf20 (query)                      2_072.07     1_011.56     3_083.63       0.2027          1.0849        41.92
IVF-TQ-b4-nl223-np14-rf10 (query)                      2_072.07       747.73     2_819.80       0.1035          1.1147        41.92
IVF-TQ-b4-nl223-np14-rf20 (query)                      2_072.07     1_071.68     3_143.74       0.1943          1.0871        41.92
IVF-TQ-b4-nl223-np21-rf10 (query)                      2_072.07       881.91     2_953.98       0.1025          1.1153        41.92
IVF-TQ-b4-nl223-np21-rf20 (query)                      2_072.07     1_228.23     3_300.29       0.1925          1.0877        41.92
IVF-TQ-b4-nl223 (self)                                 2_072.07     2_547.13     4_619.20       0.1917          1.0881        41.92
IVF-TQ-b4-nl316-np15-rf0 (query)                       2_671.86       481.34     3_153.20       0.0154          1.2703        42.72
IVF-TQ-b4-nl316-np17-rf0 (query)                       2_671.86       513.07     3_184.93       0.0151          1.2749        42.72
IVF-TQ-b4-nl316-np25-rf0 (query)                       2_671.86       652.81     3_324.67       0.0147          1.2811        42.72
IVF-TQ-b4-nl316-np15-rf10 (query)                      2_671.86       709.04     3_380.90       0.1073          1.1125        42.72
IVF-TQ-b4-nl316-np15-rf20 (query)                      2_671.86     1_016.77     3_688.62       0.2019          1.0851        42.72
IVF-TQ-b4-nl316-np17-rf10 (query)                      2_671.86       735.83     3_407.69       0.1050          1.1138        42.72
IVF-TQ-b4-nl316-np17-rf20 (query)                      2_671.86     1_050.22     3_722.08       0.1970          1.0863        42.72
IVF-TQ-b4-nl316-np25-rf10 (query)                      2_671.86       857.18     3_529.03       0.1025          1.1153        42.72
IVF-TQ-b4-nl316-np25-rf20 (query)                      2_671.86     1_166.05     3_837.91       0.1925          1.0877        42.72
IVF-TQ-b4-nl316 (self)                                 2_671.86     2_486.67     5_158.52       0.1917          1.0881        42.72
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Lowrank data

<details>
<summary><b>Lowrank data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D - TurboQuant + IVF
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.62       694.76       727.38       1.0000          1.0000        48.83
Exhaustive (self)                                         32.62     2_204.88     2_237.49       1.0000          1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              143.67       355.01       498.68       0.0662          1.9161         7.12
ExhaustiveTQ-b2-rf5 (query)                              143.67       430.12       573.79       0.1862          1.3185         7.12
ExhaustiveTQ-b2-rf10 (query)                             143.67       561.58       705.25       0.2699          1.2136         7.12
ExhaustiveTQ-b2-rf20 (query)                             143.67       926.80     1_070.47       0.4056          1.1279         7.12
ExhaustiveTQ-b2 (self)                                   143.67     3_088.79     3_232.46       0.4070          1.1561         7.12
ExhaustiveTQ-b4-rf0 (query)                              229.26       565.66       794.92       0.0871          1.7209        13.22
ExhaustiveTQ-b4-rf5 (query)                              229.26       646.74       876.00       0.2058          1.2890        13.22
ExhaustiveTQ-b4-rf10 (query)                             229.26       784.08     1_013.35       0.2865          1.1965        13.22
ExhaustiveTQ-b4-rf20 (query)                             229.26     1_154.26     1_383.52       0.4169          1.1210        13.22
ExhaustiveTQ-b4 (self)                                   229.26     3_840.19     4_069.45       0.4165          1.1485        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                          979.84       103.59     1_083.43       0.0662          1.9161         7.80
IVF-TQ-b2-nl158-np12-rf0 (query)                         979.84       116.86     1_096.70       0.0662          1.9161         7.80
IVF-TQ-b2-nl158-np17-rf0 (query)                         979.84       129.17     1_109.01       0.0662          1.9162         7.80
IVF-TQ-b2-nl158-np7-rf10 (query)                         979.84       309.58     1_289.43       0.2699          1.2136         7.80
IVF-TQ-b2-nl158-np7-rf20 (query)                         979.84       596.75     1_576.60       0.4056          1.1279         7.80
IVF-TQ-b2-nl158-np12-rf10 (query)                        979.84       308.23     1_288.07       0.2699          1.2136         7.80
IVF-TQ-b2-nl158-np12-rf20 (query)                        979.84       624.04     1_603.88       0.4056          1.1279         7.80
IVF-TQ-b2-nl158-np17-rf10 (query)                        979.84       338.12     1_317.96       0.2699          1.2136         7.80
IVF-TQ-b2-nl158-np17-rf20 (query)                        979.84       669.79     1_649.63       0.4056          1.1279         7.80
IVF-TQ-b2-nl158 (self)                                   979.84     1_080.44     2_060.29       0.4070          1.1561         7.80
IVF-TQ-b2-nl223-np11-rf0 (query)                         722.93       111.06       833.99       0.0665          1.9134         7.94
IVF-TQ-b2-nl223-np14-rf0 (query)                         722.93       117.73       840.65       0.0662          1.9160         7.94
IVF-TQ-b2-nl223-np21-rf0 (query)                         722.93       143.77       866.70       0.0662          1.9162         7.94
IVF-TQ-b2-nl223-np11-rf10 (query)                        722.93       285.47     1_008.40       0.2711          1.2125         7.94
IVF-TQ-b2-nl223-np11-rf20 (query)                        722.93       552.16     1_275.09       0.4078          1.1269         7.94
IVF-TQ-b2-nl223-np14-rf10 (query)                        722.93       294.72     1_017.65       0.2699          1.2136         7.94
IVF-TQ-b2-nl223-np14-rf20 (query)                        722.93       567.39     1_290.32       0.4056          1.1279         7.94
IVF-TQ-b2-nl223-np21-rf10 (query)                        722.93       331.92     1_054.85       0.2699          1.2136         7.94
IVF-TQ-b2-nl223-np21-rf20 (query)                        722.93       620.68     1_343.60       0.4056          1.1279         7.94
IVF-TQ-b2-nl223 (self)                                   722.93     1_087.19     1_810.11       0.4070          1.1561         7.94
IVF-TQ-b2-nl316-np15-rf0 (query)                         923.71       117.15     1_040.85       0.0664          1.9139         8.12
IVF-TQ-b2-nl316-np17-rf0 (query)                         923.71       123.56     1_047.27       0.0663          1.9153         8.12
IVF-TQ-b2-nl316-np25-rf0 (query)                         923.71       139.79     1_063.49       0.0662          1.9161         8.12
IVF-TQ-b2-nl316-np15-rf10 (query)                        923.71       280.01     1_203.72       0.2708          1.2128         8.12
IVF-TQ-b2-nl316-np15-rf20 (query)                        923.71       521.40     1_445.11       0.4072          1.1271         8.12
IVF-TQ-b2-nl316-np17-rf10 (query)                        923.71       286.72     1_210.43       0.2702          1.2133         8.12
IVF-TQ-b2-nl316-np17-rf20 (query)                        923.71       531.12     1_454.83       0.4061          1.1277         8.12
IVF-TQ-b2-nl316-np25-rf10 (query)                        923.71       308.69     1_232.40       0.2699          1.2136         8.12
IVF-TQ-b2-nl316-np25-rf20 (query)                        923.71       576.95     1_500.66       0.4056          1.1279         8.12
IVF-TQ-b2-nl316 (self)                                   923.71     1_048.94     1_972.65       0.4070          1.1561         8.12
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_030.40       137.54     1_167.93       0.0871          1.7208        14.05
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_030.40       158.45     1_188.85       0.0871          1.7208        14.05
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_030.40       191.64     1_222.03       0.0871          1.7209        14.05
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_030.40       344.88     1_375.28       0.2865          1.1965        14.05
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_030.40       644.28     1_674.67       0.4169          1.1210        14.05
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_030.40       366.91     1_397.30       0.2865          1.1965        14.05
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_030.40       683.18     1_713.57       0.4169          1.1210        14.05
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_030.40       405.56     1_435.96       0.2865          1.1965        14.05
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_030.40       734.13     1_764.52       0.4169          1.1210        14.05
IVF-TQ-b4-nl158 (self)                                 1_030.40     1_096.23     2_126.63       0.4165          1.1485        14.05
IVF-TQ-b4-nl223-np11-rf0 (query)                         819.88       152.38       972.26       0.0873          1.7186        14.26
IVF-TQ-b4-nl223-np14-rf0 (query)                         819.88       163.47       983.35       0.0871          1.7199        14.26
IVF-TQ-b4-nl223-np21-rf0 (query)                         819.88       204.30     1_024.19       0.0871          1.7209        14.26
IVF-TQ-b4-nl223-np11-rf10 (query)                        819.88       332.89     1_152.77       0.2875          1.1957        14.26
IVF-TQ-b4-nl223-np11-rf20 (query)                        819.88       599.20     1_419.08       0.4186          1.1202        14.26
IVF-TQ-b4-nl223-np14-rf10 (query)                        819.88       348.71     1_168.59       0.2866          1.1964        14.26
IVF-TQ-b4-nl223-np14-rf20 (query)                        819.88       624.10     1_443.98       0.4170          1.1210        14.26
IVF-TQ-b4-nl223-np21-rf10 (query)                        819.88       403.11     1_223.00       0.2865          1.1965        14.26
IVF-TQ-b4-nl223-np21-rf20 (query)                        819.88       704.10     1_523.99       0.4169          1.1210        14.26
IVF-TQ-b4-nl223 (self)                                   819.88     1_131.73     1_951.61       0.4165          1.1485        14.26
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_019.15       161.98     1_181.13       0.0872          1.7182        14.53
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_019.15       166.36     1_185.51       0.0872          1.7195        14.53
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_019.15       198.23     1_217.38       0.0871          1.7208        14.53
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_019.15       335.70     1_354.85       0.2872          1.1958        14.53
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_019.15       581.47     1_600.62       0.4182          1.1204        14.53
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_019.15       341.40     1_360.55       0.2868          1.1962        14.53
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_019.15       600.46     1_619.61       0.4173          1.1208        14.53
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_019.15       377.86     1_397.01       0.2865          1.1965        14.53
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_019.15       638.97     1_658.12       0.4169          1.1210        14.53
IVF-TQ-b4-nl316 (self)                                 1_019.15     1_113.96     2_133.11       0.4165          1.1485        14.53
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D - TurboQuant + IVF
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        68.49     1_308.37     1_376.86       1.0000          1.0000        97.66
Exhaustive (self)                                         68.49     4_476.58     4_545.07       1.0000          1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              349.02       651.12     1_000.14       0.0709          1.7142        13.97
ExhaustiveTQ-b2-rf5 (query)                              349.02       729.59     1_078.60       0.1815          1.2341        13.97
ExhaustiveTQ-b2-rf10 (query)                             349.02       859.14     1_208.15       0.2476          1.1648        13.97
ExhaustiveTQ-b2-rf20 (query)                             349.02     1_253.51     1_602.53       0.3618          1.1046        13.97
ExhaustiveTQ-b2 (self)                                   349.02     4_043.86     4_392.87       0.3623          1.1225        13.97
ExhaustiveTQ-b4-rf0 (query)                              470.72     1_147.31     1_618.03       0.0861          1.5231        26.18
ExhaustiveTQ-b4-rf5 (query)                              470.72     1_232.55     1_703.27       0.1891          1.2262        26.18
ExhaustiveTQ-b4-rf10 (query)                             470.72     1_381.15     1_851.87       0.2497          1.1619        26.18
ExhaustiveTQ-b4-rf20 (query)                             470.72     1_764.99     2_235.72       0.3582          1.1058        26.18
ExhaustiveTQ-b4 (self)                                   470.72     5_740.15     6_210.87       0.3580          1.1245        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_790.02       191.04     1_981.06       0.0709          1.7142        14.98
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_790.02       204.07     1_994.09       0.0709          1.7142        14.98
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_790.02       215.39     2_005.41       0.0709          1.7142        14.98
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_790.02       422.63     2_212.65       0.2475          1.1648        14.98
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_790.02       744.28     2_534.30       0.3618          1.1046        14.98
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_790.02       421.80     2_211.83       0.2475          1.1648        14.98
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_790.02       776.36     2_566.38       0.3618          1.1046        14.98
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_790.02       445.01     2_235.03       0.2475          1.1648        14.98
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_790.02       810.54     2_600.56       0.3618          1.1046        14.98
IVF-TQ-b2-nl158 (self)                                 1_790.02     1_492.29     3_282.31       0.3623          1.1225        14.98
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_418.56       207.52     1_626.08       0.0709          1.7142        15.21
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_418.56       215.60     1_634.16       0.0709          1.7142        15.21
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_418.56       241.55     1_660.11       0.0709          1.7142        15.21
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_418.56       412.00     1_830.56       0.2475          1.1648        15.21
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_418.56       689.66     2_108.22       0.3618          1.1046        15.21
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_418.56       426.24     1_844.80       0.2476          1.1648        15.21
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_418.56       723.43     2_141.99       0.3618          1.1046        15.21
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_418.56       446.84     1_865.39       0.2476          1.1648        15.21
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_418.56       754.80     2_173.36       0.3618          1.1046        15.21
IVF-TQ-b2-nl223 (self)                                 1_418.56     1_481.35     2_899.90       0.3622          1.1225        15.21
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_754.12       214.94     1_969.07       0.0709          1.7135        15.55
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_754.12       221.19     1_975.31       0.0709          1.7142        15.55
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_754.12       252.62     2_006.74       0.0709          1.7142        15.55
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_754.12       418.73     2_172.85       0.2476          1.1648        15.55
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_754.12       680.49     2_434.61       0.3618          1.1046        15.55
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_754.12       440.60     2_194.72       0.2476          1.1648        15.55
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_754.12       690.98     2_445.11       0.3618          1.1046        15.55
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_754.12       441.72     2_195.84       0.2476          1.1648        15.55
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_754.12       735.01     2_489.13       0.3618          1.1046        15.55
IVF-TQ-b2-nl316 (self)                                 1_754.12     1_503.07     3_257.19       0.3623          1.1225        15.55
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_845.82       261.26     2_107.08       0.0861          1.5231        27.51
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_845.82       292.89     2_138.72       0.0861          1.5231        27.51
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_845.82       306.79     2_152.61       0.0861          1.5231        27.51
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_845.82       500.86     2_346.68       0.2497          1.1619        27.51
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_845.82       847.15     2_692.98       0.3582          1.1058        27.51
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_845.82       516.25     2_362.08       0.2497          1.1619        27.51
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_845.82       872.46     2_718.29       0.3582          1.1058        27.51
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_845.82       554.35     2_400.17       0.2497          1.1619        27.51
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_845.82       913.53     2_759.35       0.3582          1.1058        27.51
IVF-TQ-b4-nl158 (self)                                 1_845.82     1_661.39     3_507.22       0.3580          1.1245        27.51
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_517.57       289.62     1_807.19       0.0861          1.5231        27.85
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_517.57       305.47     1_823.04       0.0861          1.5231        27.85
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_517.57       344.00     1_861.57       0.0861          1.5231        27.85
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_517.57       499.76     2_017.33       0.2497          1.1619        27.85
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_517.57       808.26     2_325.83       0.3582          1.1058        27.85
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_517.57       517.64     2_035.20       0.2497          1.1619        27.85
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_517.57       813.41     2_330.98       0.3582          1.1058        27.85
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_517.57       570.02     2_087.58       0.2497          1.1619        27.85
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_517.57       890.70     2_408.27       0.3582          1.1058        27.85
IVF-TQ-b4-nl223 (self)                                 1_517.57     1_697.83     3_215.40       0.3580          1.1245        27.85
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_872.79       295.69     2_168.48       0.0861          1.5231        28.33
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_872.79       304.73     2_177.52       0.0861          1.5231        28.33
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_872.79       349.42     2_222.21       0.0861          1.5231        28.33
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_872.79       508.71     2_381.50       0.2497          1.1619        28.33
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_872.79       779.03     2_651.82       0.3582          1.1058        28.33
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_872.79       513.66     2_386.45       0.2497          1.1619        28.33
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_872.79       798.70     2_671.49       0.3582          1.1058        28.33
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_872.79       552.54     2_425.33       0.2497          1.1619        28.33
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_872.79       855.19     2_727.98       0.3582          1.1058        28.33
IVF-TQ-b4-nl316 (self)                                 1_872.79     1_693.07     3_565.86       0.3580          1.1245        28.33
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Lowrank data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D - TurboQuant + IVF
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        99.56     1_857.50     1_957.06       1.0000          1.0000       146.48
Exhaustive (self)                                         99.56     6_297.73     6_397.29       1.0000          1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              603.56       935.75     1_539.31       0.0719          1.4569        21.33
ExhaustiveTQ-b2-rf5 (query)                              603.56     1_033.02     1_636.58       0.1764          1.1854        21.33
ExhaustiveTQ-b2-rf10 (query)                             603.56     1_168.41     1_771.97       0.2313          1.1364        21.33
ExhaustiveTQ-b2-rf20 (query)                             603.56     1_570.23     2_173.79       0.3303          1.0920        21.33
ExhaustiveTQ-b2 (self)                                   603.56     5_152.14     5_755.70       0.3296          1.1026        21.33
ExhaustiveTQ-b4-rf0 (query)                              743.81     1_763.24     2_507.06       0.0844          1.4135        39.64
ExhaustiveTQ-b4-rf5 (query)                              743.81     1_847.42     2_591.23       0.1812          1.1812        39.64
ExhaustiveTQ-b4-rf10 (query)                             743.81     1_973.07     2_716.88       0.2329          1.1351        39.64
ExhaustiveTQ-b4-rf20 (query)                             743.81     2_361.62     3_105.43       0.3263          1.0942        39.64
ExhaustiveTQ-b4 (self)                                   743.81     7_807.82     8_551.64       0.3285          1.1030        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        2_607.86       289.07     2_896.92       0.0719          1.4569        22.63
IVF-TQ-b2-nl158-np12-rf0 (query)                       2_607.86       307.68     2_915.54       0.0719          1.4569        22.63
IVF-TQ-b2-nl158-np17-rf0 (query)                       2_607.86       325.05     2_932.90       0.0719          1.4569        22.63
IVF-TQ-b2-nl158-np7-rf10 (query)                       2_607.86       538.49     3_146.35       0.2313          1.1364        22.63
IVF-TQ-b2-nl158-np7-rf20 (query)                       2_607.86       903.64     3_511.50       0.3303          1.0920        22.63
IVF-TQ-b2-nl158-np12-rf10 (query)                      2_607.86       565.46     3_173.32       0.2313          1.1364        22.63
IVF-TQ-b2-nl158-np12-rf20 (query)                      2_607.86       934.90     3_542.76       0.3303          1.0920        22.63
IVF-TQ-b2-nl158-np17-rf10 (query)                      2_607.86       581.92     3_189.77       0.2313          1.1364        22.63
IVF-TQ-b2-nl158-np17-rf20 (query)                      2_607.86       961.28     3_569.14       0.3303          1.0920        22.63
IVF-TQ-b2-nl158 (self)                                 2_607.86     1_814.24     4_422.09       0.3296          1.1026        22.63
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_879.38       303.89     2_183.27       0.0719          1.4569        22.96
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_879.38       311.92     2_191.30       0.0719          1.4569        22.96
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_879.38       339.92     2_219.30       0.0719          1.4569        22.96
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_879.38       527.80     2_407.18       0.2313          1.1364        22.96
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_879.38       833.55     2_712.93       0.3303          1.0920        22.96
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_879.38       538.03     2_417.41       0.2313          1.1364        22.96
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_879.38       847.45     2_726.83       0.3303          1.0920        22.96
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_879.38       586.49     2_465.87       0.2313          1.1364        22.96
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_879.38       903.66     2_783.04       0.3303          1.0920        22.96
IVF-TQ-b2-nl223 (self)                                 1_879.38     1_865.08     3_744.46       0.3296          1.1026        22.96
IVF-TQ-b2-nl316-np15-rf0 (query)                       2_542.55       317.66     2_860.21       0.0719          1.4568        23.53
IVF-TQ-b2-nl316-np17-rf0 (query)                       2_542.55       330.80     2_873.35       0.0719          1.4569        23.53
IVF-TQ-b2-nl316-np25-rf0 (query)                       2_542.55       350.16     2_892.72       0.0719          1.4569        23.53
IVF-TQ-b2-nl316-np15-rf10 (query)                      2_542.55       532.86     3_075.41       0.2313          1.1364        23.53
IVF-TQ-b2-nl316-np15-rf20 (query)                      2_542.55       812.39     3_354.94       0.3303          1.0920        23.53
IVF-TQ-b2-nl316-np17-rf10 (query)                      2_542.55       536.81     3_079.36       0.2313          1.1364        23.53
IVF-TQ-b2-nl316-np17-rf20 (query)                      2_542.55       827.79     3_370.34       0.3303          1.0920        23.53
IVF-TQ-b2-nl316-np25-rf10 (query)                      2_542.55       576.01     3_118.56       0.2313          1.1364        23.53
IVF-TQ-b2-nl316-np25-rf20 (query)                      2_542.55       869.53     3_412.08       0.3303          1.0920        23.53
IVF-TQ-b2-nl316 (self)                                 2_542.55     1_919.82     4_462.37       0.3296          1.1026        23.53
IVF-TQ-b4-nl158-np7-rf0 (query)                        2_660.94       401.68     3_062.61       0.0844          1.4135        41.40
IVF-TQ-b4-nl158-np12-rf0 (query)                       2_660.94       441.53     3_102.47       0.0844          1.4135        41.40
IVF-TQ-b4-nl158-np17-rf0 (query)                       2_660.94       471.57     3_132.51       0.0844          1.4135        41.40
IVF-TQ-b4-nl158-np7-rf10 (query)                       2_660.94       670.92     3_331.86       0.2329          1.1351        41.40
IVF-TQ-b4-nl158-np7-rf20 (query)                       2_660.94     1_044.76     3_705.70       0.3263          1.0942        41.40
IVF-TQ-b4-nl158-np12-rf10 (query)                      2_660.94       706.24     3_367.18       0.2329          1.1351        41.40
IVF-TQ-b4-nl158-np12-rf20 (query)                      2_660.94     1_090.36     3_751.30       0.3263          1.0942        41.40
IVF-TQ-b4-nl158-np17-rf10 (query)                      2_660.94       747.54     3_408.48       0.2329          1.1351        41.40
IVF-TQ-b4-nl158-np17-rf20 (query)                      2_660.94     1_136.63     3_797.57       0.3263          1.0942        41.40
IVF-TQ-b4-nl158 (self)                                 2_660.94     2_144.25     4_805.18       0.3285          1.1030        41.40
IVF-TQ-b4-nl223-np11-rf0 (query)                       2_102.21       425.64     2_527.85       0.0844          1.4135        41.87
IVF-TQ-b4-nl223-np14-rf0 (query)                       2_102.21       452.63     2_554.84       0.0844          1.4135        41.87
IVF-TQ-b4-nl223-np21-rf0 (query)                       2_102.21       505.27     2_607.48       0.0844          1.4135        41.87
IVF-TQ-b4-nl223-np11-rf10 (query)                      2_102.21       674.51     2_776.72       0.2329          1.1351        41.87
IVF-TQ-b4-nl223-np11-rf20 (query)                      2_102.21       967.47     3_069.68       0.3263          1.0942        41.87
IVF-TQ-b4-nl223-np14-rf10 (query)                      2_102.21       686.73     2_788.94       0.2329          1.1351        41.87
IVF-TQ-b4-nl223-np14-rf20 (query)                      2_102.21     1_001.46     3_103.67       0.3263          1.0942        41.87
IVF-TQ-b4-nl223-np21-rf10 (query)                      2_102.21       765.84     2_868.05       0.2329          1.1351        41.87
IVF-TQ-b4-nl223-np21-rf20 (query)                      2_102.21     1_099.95     3_202.16       0.3263          1.0942        41.87
IVF-TQ-b4-nl223 (self)                                 2_102.21     2_212.37     4_314.58       0.3285          1.1030        41.87
IVF-TQ-b4-nl316-np15-rf0 (query)                       2_715.38       452.25     3_167.63       0.0844          1.4131        42.73
IVF-TQ-b4-nl316-np17-rf0 (query)                       2_715.38       468.16     3_183.55       0.0844          1.4135        42.73
IVF-TQ-b4-nl316-np25-rf0 (query)                       2_715.38       520.14     3_235.52       0.0844          1.4135        42.73
IVF-TQ-b4-nl316-np15-rf10 (query)                      2_715.38       673.91     3_389.29       0.2329          1.1351        42.73
IVF-TQ-b4-nl316-np15-rf20 (query)                      2_715.38       987.82     3_703.20       0.3263          1.0942        42.73
IVF-TQ-b4-nl316-np17-rf10 (query)                      2_715.38       684.77     3_400.15       0.2329          1.1351        42.73
IVF-TQ-b4-nl316-np17-rf20 (query)                      2_715.38       982.93     3_698.32       0.3263          1.0942        42.73
IVF-TQ-b4-nl316-np25-rf10 (query)                      2_715.38       740.58     3_455.96       0.2329          1.1351        42.73
IVF-TQ-b4-nl316-np25-rf20 (query)                      2_715.38     1_048.45     3_763.83       0.3263          1.0942        42.73
IVF-TQ-b4-nl316 (self)                                 2_715.38     2_287.20     5_002.59       0.3285          1.1030        42.73
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

#### Cell embeddings data

<details>
<summary><b>Cell embedding data - 256 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 256D - TurboQuant + IVF
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        32.28       677.66       709.95       1.0000          1.0000        48.83
Exhaustive (self)                                         32.28     2_267.03     2_299.31       1.0000          1.0000        48.83
ExhaustiveTQ-b2-rf0 (query)                              143.58       361.94       505.52       0.7918          1.0898         7.12
ExhaustiveTQ-b2-rf5 (query)                              143.58       441.05       584.63       0.9995          1.0000         7.12
ExhaustiveTQ-b2-rf10 (query)                             143.58       569.46       713.04       1.0000          1.0000         7.12
ExhaustiveTQ-b2-rf20 (query)                             143.58       958.41     1_101.98       1.0000          1.0000         7.12
ExhaustiveTQ-b2 (self)                                   143.58     3_123.76     3_267.33       1.0000          1.0000         7.12
ExhaustiveTQ-b4-rf0 (query)                              234.68       571.40       806.08       0.8727          1.0322        13.22
ExhaustiveTQ-b4-rf5 (query)                              234.68       655.06       889.73       1.0000          1.0000        13.22
ExhaustiveTQ-b4-rf10 (query)                             234.68       783.33     1_018.01       1.0000          1.0000        13.22
ExhaustiveTQ-b4-rf20 (query)                             234.68     1_166.79     1_401.46       1.0000          1.0000        13.22
ExhaustiveTQ-b4 (self)                                   234.68     3_834.64     4_069.31       1.0000          1.0000        13.22
IVF-TQ-b2-nl158-np7-rf0 (query)                        1_035.26       131.11     1_166.37       0.7916          1.0897         7.78
IVF-TQ-b2-nl158-np12-rf0 (query)                       1_035.26       175.39     1_210.65       0.7918          1.0898         7.78
IVF-TQ-b2-nl158-np17-rf0 (query)                       1_035.26       215.76     1_251.02       0.7918          1.0898         7.78
IVF-TQ-b2-nl158-np7-rf10 (query)                       1_035.26       334.80     1_370.06       0.9981          1.0004         7.78
IVF-TQ-b2-nl158-np7-rf20 (query)                       1_035.26       615.06     1_650.32       0.9982          1.0004         7.78
IVF-TQ-b2-nl158-np12-rf10 (query)                      1_035.26       401.68     1_436.95       1.0000          1.0000         7.78
IVF-TQ-b2-nl158-np12-rf20 (query)                      1_035.26       704.54     1_739.80       1.0000          1.0000         7.78
IVF-TQ-b2-nl158-np17-rf10 (query)                      1_035.26       449.88     1_485.15       1.0000          1.0000         7.78
IVF-TQ-b2-nl158-np17-rf20 (query)                      1_035.26       797.78     1_833.04       1.0000          1.0000         7.78
IVF-TQ-b2-nl158 (self)                                 1_035.26     1_209.39     2_244.65       0.9999          1.0000         7.78
IVF-TQ-b2-nl223-np11-rf0 (query)                         644.45       131.99       776.44       0.7919          1.0897         7.93
IVF-TQ-b2-nl223-np14-rf0 (query)                         644.45       152.20       796.65       0.7918          1.0897         7.93
IVF-TQ-b2-nl223-np21-rf0 (query)                         644.45       185.97       830.42       0.7918          1.0898         7.93
IVF-TQ-b2-nl223-np11-rf10 (query)                        644.45       323.57       968.02       0.9995          1.0001         7.93
IVF-TQ-b2-nl223-np11-rf20 (query)                        644.45       600.26     1_244.71       0.9995          1.0001         7.93
IVF-TQ-b2-nl223-np14-rf10 (query)                        644.45       347.78       992.23       0.9999          1.0000         7.93
IVF-TQ-b2-nl223-np14-rf20 (query)                        644.45       631.82     1_276.28       0.9999          1.0000         7.93
IVF-TQ-b2-nl223-np21-rf10 (query)                        644.45       406.03     1_050.48       1.0000          1.0000         7.93
IVF-TQ-b2-nl223-np21-rf20 (query)                        644.45       715.07     1_359.52       1.0000          1.0000         7.93
IVF-TQ-b2-nl223 (self)                                   644.45     1_029.94     1_674.39       1.0000          1.0000         7.93
IVF-TQ-b2-nl316-np15-rf0 (query)                         902.71       137.96     1_040.66       0.7918          1.0897         8.12
IVF-TQ-b2-nl316-np17-rf0 (query)                         902.71       142.03     1_044.74       0.7918          1.0898         8.12
IVF-TQ-b2-nl316-np25-rf0 (query)                         902.71       185.57     1_088.28       0.7918          1.0898         8.12
IVF-TQ-b2-nl316-np15-rf10 (query)                        902.71       311.32     1_214.02       0.9997          1.0000         8.12
IVF-TQ-b2-nl316-np15-rf20 (query)                        902.71       568.52     1_471.23       0.9997          1.0000         8.12
IVF-TQ-b2-nl316-np17-rf10 (query)                        902.71       326.22     1_228.92       0.9999          1.0000         8.12
IVF-TQ-b2-nl316-np17-rf20 (query)                        902.71       585.43     1_488.13       0.9999          1.0000         8.12
IVF-TQ-b2-nl316-np25-rf10 (query)                        902.71       374.65     1_277.36       1.0000          1.0000         8.12
IVF-TQ-b2-nl316-np25-rf20 (query)                        902.71       658.42     1_561.12       1.0000          1.0000         8.12
IVF-TQ-b2-nl316 (self)                                   902.71       954.73     1_857.44       1.0000          1.0000         8.12
IVF-TQ-b4-nl158-np7-rf0 (query)                        1_185.23       184.18     1_369.41       0.8721          1.0325        14.02
IVF-TQ-b4-nl158-np12-rf0 (query)                       1_185.23       264.32     1_449.55       0.8727          1.0322        14.02
IVF-TQ-b4-nl158-np17-rf0 (query)                       1_185.23       327.74     1_512.97       0.8727          1.0322        14.02
IVF-TQ-b4-nl158-np7-rf10 (query)                       1_185.23       394.93     1_580.16       0.9981          1.0004        14.02
IVF-TQ-b4-nl158-np7-rf20 (query)                       1_185.23       684.04     1_869.27       0.9982          1.0004        14.02
IVF-TQ-b4-nl158-np12-rf10 (query)                      1_185.23       476.76     1_661.98       1.0000          1.0000        14.02
IVF-TQ-b4-nl158-np12-rf20 (query)                      1_185.23       803.33     1_988.56       1.0000          1.0000        14.02
IVF-TQ-b4-nl158-np17-rf10 (query)                      1_185.23       565.53     1_750.76       1.0000          1.0000        14.02
IVF-TQ-b4-nl158-np17-rf20 (query)                      1_185.23       895.23     2_080.46       1.0000          1.0000        14.02
IVF-TQ-b4-nl158 (self)                                 1_185.23     1_257.13     2_442.36       0.9999          1.0000        14.02
IVF-TQ-b4-nl223-np11-rf0 (query)                         728.10       180.78       908.88       0.8726          1.0323        14.23
IVF-TQ-b4-nl223-np14-rf0 (query)                         728.10       213.12       941.22       0.8727          1.0322        14.23
IVF-TQ-b4-nl223-np21-rf0 (query)                         728.10       274.69     1_002.79       0.8727          1.0322        14.23
IVF-TQ-b4-nl223-np11-rf10 (query)                        728.10       378.69     1_106.79       0.9995          1.0001        14.23
IVF-TQ-b4-nl223-np11-rf20 (query)                        728.10       651.19     1_379.29       0.9995          1.0001        14.23
IVF-TQ-b4-nl223-np14-rf10 (query)                        728.10       412.57     1_140.67       0.9999          1.0000        14.23
IVF-TQ-b4-nl223-np14-rf20 (query)                        728.10       696.61     1_424.71       0.9999          1.0000        14.23
IVF-TQ-b4-nl223-np21-rf10 (query)                        728.10       495.00     1_223.10       1.0000          1.0000        14.23
IVF-TQ-b4-nl223-np21-rf20 (query)                        728.10       811.91     1_540.01       1.0000          1.0000        14.23
IVF-TQ-b4-nl223 (self)                                   728.10     1_103.90     1_832.00       1.0000          1.0000        14.23
IVF-TQ-b4-nl316-np15-rf0 (query)                         914.42       183.76     1_098.18       0.8727          1.0322        14.54
IVF-TQ-b4-nl316-np17-rf0 (query)                         914.42       197.69     1_112.12       0.8727          1.0322        14.54
IVF-TQ-b4-nl316-np25-rf0 (query)                         914.42       255.96     1_170.39       0.8727          1.0322        14.54
IVF-TQ-b4-nl316-np15-rf10 (query)                        914.42       371.76     1_286.18       0.9997          1.0000        14.54
IVF-TQ-b4-nl316-np15-rf20 (query)                        914.42       630.60     1_545.02       0.9997          1.0000        14.54
IVF-TQ-b4-nl316-np17-rf10 (query)                        914.42       385.76     1_300.19       0.9999          1.0000        14.54
IVF-TQ-b4-nl316-np17-rf20 (query)                        914.42       657.35     1_571.78       0.9999          1.0000        14.54
IVF-TQ-b4-nl316-np25-rf10 (query)                        914.42       456.12     1_370.54       1.0000          1.0000        14.54
IVF-TQ-b4-nl316-np25-rf20 (query)                        914.42       801.25     1_715.67       1.0000          1.0000        14.54
IVF-TQ-b4-nl316 (self)                                   914.42     1_063.08     1_977.51       1.0000          1.0000        14.54
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 512 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 512D - TurboQuant + IVF
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                        69.59     1_307.83     1_377.43       1.0000          1.0000        97.66
Exhaustive (self)                                         69.59     4_421.49     4_491.08       1.0000          1.0000        97.66
ExhaustiveTQ-b2-rf0 (query)                              352.03       663.28     1_015.31       0.8424          1.0447        13.97
ExhaustiveTQ-b2-rf5 (query)                              352.03       725.76     1_077.79       0.9999          1.0000        13.97
ExhaustiveTQ-b2-rf10 (query)                             352.03       874.39     1_226.42       1.0000          1.0000        13.97
ExhaustiveTQ-b2-rf20 (query)                             352.03     1_267.11     1_619.13       1.0000          1.0000        13.97
ExhaustiveTQ-b2 (self)                                   352.03     4_166.40     4_518.42       1.0000          1.0000        13.97
ExhaustiveTQ-b4-rf0 (query)                              460.29     1_141.24     1_601.53       0.8985          1.0191        26.18
ExhaustiveTQ-b4-rf5 (query)                              460.29     1_247.96     1_708.25       1.0000          1.0000        26.18
ExhaustiveTQ-b4-rf10 (query)                             460.29     1_374.06     1_834.35       1.0000          1.0000        26.18
ExhaustiveTQ-b4-rf20 (query)                             460.29     1_746.49     2_206.78       1.0000          1.0000        26.18
ExhaustiveTQ-b4 (self)                                   460.29     5_769.48     6_229.77       1.0000          1.0000        26.18
IVF-TQ-b2-nl158-np7-rf0 (query)                        2_141.37       232.24     2_373.60       0.8420          1.0449        14.96
IVF-TQ-b2-nl158-np12-rf0 (query)                       2_141.37       300.54     2_441.91       0.8424          1.0447        14.96
IVF-TQ-b2-nl158-np17-rf0 (query)                       2_141.37       361.78     2_503.15       0.8424          1.0447        14.96
IVF-TQ-b2-nl158-np7-rf10 (query)                       2_141.37       457.42     2_598.79       0.9986          1.0003        14.96
IVF-TQ-b2-nl158-np7-rf20 (query)                       2_141.37       759.43     2_900.80       0.9986          1.0003        14.96
IVF-TQ-b2-nl158-np12-rf10 (query)                      2_141.37       538.19     2_679.56       0.9999          1.0000        14.96
IVF-TQ-b2-nl158-np12-rf20 (query)                      2_141.37       876.37     3_017.74       0.9999          1.0000        14.96
IVF-TQ-b2-nl158-np17-rf10 (query)                      2_141.37       604.41     2_745.78       0.9999          1.0000        14.96
IVF-TQ-b2-nl158-np17-rf20 (query)                      2_141.37       972.47     3_113.84       0.9999          1.0000        14.96
IVF-TQ-b2-nl158 (self)                                 2_141.37     1_564.48     3_705.85       1.0000          1.0000        14.96
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_109.33       237.13     1_346.46       0.8423          1.0447        15.25
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_109.33       269.06     1_378.39       0.8424          1.0447        15.25
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_109.33       325.92     1_435.25       0.8424          1.0447        15.25
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_109.33       446.44     1_555.77       0.9997          1.0000        15.25
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_109.33       732.05     1_841.38       0.9997          1.0000        15.25
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_109.33       476.56     1_585.89       0.9999          1.0000        15.25
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_109.33       769.98     1_879.31       0.9999          1.0000        15.25
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_109.33       547.36     1_656.69       1.0000          1.0000        15.25
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_109.33       864.59     1_973.92       1.0000          1.0000        15.25
IVF-TQ-b2-nl223 (self)                                 1_109.33     1_461.68     2_571.01       1.0000          1.0000        15.25
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_346.22       264.42     1_610.64       0.8424          1.0447        15.56
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_346.22       255.75     1_601.98       0.8424          1.0447        15.56
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_346.22       304.38     1_650.61       0.8424          1.0447        15.56
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_346.22       447.13     1_793.35       0.9999          1.0000        15.56
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_346.22       729.43     2_075.66       0.9999          1.0000        15.56
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_346.22       474.20     1_820.43       0.9999          1.0000        15.56
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_346.22       763.41     2_109.64       0.9999          1.0000        15.56
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_346.22       525.86     1_872.08       1.0000          1.0000        15.56
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_346.22       845.84     2_192.06       1.0000          1.0000        15.56
IVF-TQ-b2-nl316 (self)                                 1_346.22     1_409.21     2_755.44       1.0000          1.0000        15.56
IVF-TQ-b4-nl158-np7-rf0 (query)                        2_140.25       342.77     2_483.01       0.8977          1.0194        27.46
IVF-TQ-b4-nl158-np12-rf0 (query)                       2_140.25       466.83     2_607.08       0.8985          1.0191        27.46
IVF-TQ-b4-nl158-np17-rf0 (query)                       2_140.25       568.14     2_708.39       0.8985          1.0191        27.46
IVF-TQ-b4-nl158-np7-rf10 (query)                       2_140.25       574.59     2_714.84       0.9986          1.0003        27.46
IVF-TQ-b4-nl158-np7-rf20 (query)                       2_140.25       886.48     3_026.73       0.9986          1.0003        27.46
IVF-TQ-b4-nl158-np12-rf10 (query)                      2_140.25       709.72     2_849.97       0.9999          1.0000        27.46
IVF-TQ-b4-nl158-np12-rf20 (query)                      2_140.25     1_044.71     3_184.96       0.9999          1.0000        27.46
IVF-TQ-b4-nl158-np17-rf10 (query)                      2_140.25       825.73     2_965.98       0.9999          1.0000        27.46
IVF-TQ-b4-nl158-np17-rf20 (query)                      2_140.25     1_185.16     3_325.41       0.9999          1.0000        27.46
IVF-TQ-b4-nl158 (self)                                 2_140.25     1_850.79     3_991.04       1.0000          1.0000        27.46
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_293.56       342.89     1_636.45       0.8984          1.0191        27.91
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_293.56       407.17     1_700.73       0.8985          1.0191        27.91
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_293.56       505.87     1_799.43       0.8985          1.0191        27.91
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_293.56       574.91     1_868.47       0.9997          1.0000        27.91
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_293.56       844.27     2_137.83       0.9997          1.0000        27.91
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_293.56       616.66     1_910.22       0.9999          1.0000        27.91
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_293.56       935.73     2_229.29       0.9999          1.0000        27.91
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_293.56       732.15     2_025.71       1.0000          1.0000        27.91
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_293.56     1_052.30     2_345.86       1.0000          1.0000        27.91
IVF-TQ-b4-nl223 (self)                                 1_293.56     1_721.38     3_014.94       1.0000          1.0000        27.91
IVF-TQ-b4-nl316-np15-rf0 (query)                       1_488.77       345.31     1_834.08       0.8985          1.0191        28.36
IVF-TQ-b4-nl316-np17-rf0 (query)                       1_488.77       377.33     1_866.10       0.8985          1.0191        28.36
IVF-TQ-b4-nl316-np25-rf0 (query)                       1_488.77       478.17     1_966.94       0.8985          1.0191        28.36
IVF-TQ-b4-nl316-np15-rf10 (query)                      1_488.77       558.17     2_046.93       0.9999          1.0000        28.36
IVF-TQ-b4-nl316-np15-rf20 (query)                      1_488.77       845.62     2_334.38       0.9999          1.0000        28.36
IVF-TQ-b4-nl316-np17-rf10 (query)                      1_488.77       576.70     2_065.46       0.9999          1.0000        28.36
IVF-TQ-b4-nl316-np17-rf20 (query)                      1_488.77       882.49     2_371.25       0.9999          1.0000        28.36
IVF-TQ-b4-nl316-np25-rf10 (query)                      1_488.77       678.98     2_167.75       1.0000          1.0000        28.36
IVF-TQ-b4-nl316-np25-rf20 (query)                      1_488.77     1_003.53     2_492.30       1.0000          1.0000        28.36
IVF-TQ-b4-nl316 (self)                                 1_488.77     1_649.55     3_138.32       1.0000          1.0000        28.36
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

---

<details>
<summary><b>Cell embedding data - 768 dimensions</b>:</summary>
</br>
<pre><code>
===================================================================================================================================
Benchmark: 50k samples, 768D - TurboQuant + IVF
===================================================================================================================================
Method                                               Build (ms)   Query (ms)   Total (ms)     Recall@k Mean dist ratio    Size (MB)
-----------------------------------------------------------------------------------------------------------------------------------
Exhaustive (query)                                       100.87     1_869.91     1_970.79       1.0000          1.0000       146.48
Exhaustive (self)                                        100.87     6_301.11     6_401.98       1.0000          1.0000       146.48
ExhaustiveTQ-b2-rf0 (query)                              628.54       938.90     1_567.44       0.8736          1.0271        21.33
ExhaustiveTQ-b2-rf5 (query)                              628.54     1_034.47     1_663.00       1.0000          1.0000        21.33
ExhaustiveTQ-b2-rf10 (query)                             628.54     1_178.00     1_806.54       1.0000          1.0000        21.33
ExhaustiveTQ-b2-rf20 (query)                             628.54     1_579.45     2_207.99       1.0000          1.0000        21.33
ExhaustiveTQ-b2 (self)                                   628.54     5_204.41     5_832.94       0.9999          1.0000        21.33
ExhaustiveTQ-b4-rf0 (query)                              745.25     1_737.99     2_483.24       0.9097          1.0146        39.64
ExhaustiveTQ-b4-rf5 (query)                              745.25     1_847.92     2_593.17       1.0000          1.0000        39.64
ExhaustiveTQ-b4-rf10 (query)                             745.25     1_962.14     2_707.39       1.0000          1.0000        39.64
ExhaustiveTQ-b4-rf20 (query)                             745.25     2_351.35     3_096.60       1.0000          1.0000        39.64
ExhaustiveTQ-b4 (self)                                   745.25     7_747.91     8_493.17       0.9999          1.0000        39.64
IVF-TQ-b2-nl158-np7-rf0 (query)                        3_048.09       349.50     3_397.59       0.8735          1.0272        22.61
IVF-TQ-b2-nl158-np12-rf0 (query)                       3_048.09       449.16     3_497.25       0.8736          1.0271        22.61
IVF-TQ-b2-nl158-np17-rf0 (query)                       3_048.09       528.06     3_576.15       0.8736          1.0271        22.61
IVF-TQ-b2-nl158-np7-rf10 (query)                       3_048.09       606.30     3_654.39       0.9995          1.0001        22.61
IVF-TQ-b2-nl158-np7-rf20 (query)                       3_048.09       922.79     3_970.88       0.9995          1.0001        22.61
IVF-TQ-b2-nl158-np12-rf10 (query)                      3_048.09       703.42     3_751.51       1.0000          1.0000        22.61
IVF-TQ-b2-nl158-np12-rf20 (query)                      3_048.09     1_055.64     4_103.72       1.0000          1.0000        22.61
IVF-TQ-b2-nl158-np17-rf10 (query)                      3_048.09       789.78     3_837.87       1.0000          1.0000        22.61
IVF-TQ-b2-nl158-np17-rf20 (query)                      3_048.09     1_155.95     4_204.04       1.0000          1.0000        22.61
IVF-TQ-b2-nl158 (self)                                 3_048.09     2_007.43     5_055.52       0.9999          1.0000        22.61
IVF-TQ-b2-nl223-np11-rf0 (query)                       1_671.58       346.46     2_018.04       0.8736          1.0271        23.01
IVF-TQ-b2-nl223-np14-rf0 (query)                       1_671.58       385.96     2_057.54       0.8736          1.0271        23.01
IVF-TQ-b2-nl223-np21-rf0 (query)                       1_671.58       468.39     2_139.98       0.8736          1.0271        23.01
IVF-TQ-b2-nl223-np11-rf10 (query)                      1_671.58       583.54     2_255.12       0.9998          1.0000        23.01
IVF-TQ-b2-nl223-np11-rf20 (query)                      1_671.58       894.34     2_565.92       0.9998          1.0000        23.01
IVF-TQ-b2-nl223-np14-rf10 (query)                      1_671.58       622.69     2_294.27       0.9999          1.0000        23.01
IVF-TQ-b2-nl223-np14-rf20 (query)                      1_671.58       950.96     2_622.54       0.9999          1.0000        23.01
IVF-TQ-b2-nl223-np21-rf10 (query)                      1_671.58       716.68     2_388.27       1.0000          1.0000        23.01
IVF-TQ-b2-nl223-np21-rf20 (query)                      1_671.58     1_064.35     2_735.93       1.0000          1.0000        23.01
IVF-TQ-b2-nl223 (self)                                 1_671.58     1_897.78     3_569.36       0.9999          1.0000        23.01
IVF-TQ-b2-nl316-np15-rf0 (query)                       1_983.82       364.62     2_348.44       0.8736          1.0271        23.53
IVF-TQ-b2-nl316-np17-rf0 (query)                       1_983.82       374.77     2_358.59       0.8736          1.0271        23.53
IVF-TQ-b2-nl316-np25-rf0 (query)                       1_983.82       447.83     2_431.65       0.8736          1.0271        23.53
IVF-TQ-b2-nl316-np15-rf10 (query)                      1_983.82       608.74     2_592.56       0.9999          1.0000        23.53
IVF-TQ-b2-nl316-np15-rf20 (query)                      1_983.82       896.63     2_880.46       0.9999          1.0000        23.53
IVF-TQ-b2-nl316-np17-rf10 (query)                      1_983.82       609.69     2_593.51       0.9999          1.0000        23.53
IVF-TQ-b2-nl316-np17-rf20 (query)                      1_983.82       933.00     2_916.83       0.9999          1.0000        23.53
IVF-TQ-b2-nl316-np25-rf10 (query)                      1_983.82       683.97     2_667.79       1.0000          1.0000        23.53
IVF-TQ-b2-nl316-np25-rf20 (query)                      1_983.82     1_031.68     3_015.50       1.0000          1.0000        23.53
IVF-TQ-b2-nl316 (self)                                 1_983.82     1_902.55     3_886.38       0.9999          1.0000        23.53
IVF-TQ-b4-nl158-np7-rf0 (query)                        3_095.34       513.98     3_609.32       0.9094          1.0147        41.37
IVF-TQ-b4-nl158-np12-rf0 (query)                       3_095.34       694.19     3_789.53       0.9097          1.0146        41.37
IVF-TQ-b4-nl158-np17-rf0 (query)                       3_095.34       838.96     3_934.30       0.9097          1.0146        41.37
IVF-TQ-b4-nl158-np7-rf10 (query)                       3_095.34       766.48     3_861.82       0.9995          1.0001        41.37
IVF-TQ-b4-nl158-np7-rf20 (query)                       3_095.34     1_081.84     4_177.18       0.9995          1.0001        41.37
IVF-TQ-b4-nl158-np12-rf10 (query)                      3_095.34       961.17     4_056.51       1.0000          1.0000        41.37
IVF-TQ-b4-nl158-np12-rf20 (query)                      3_095.34     1_306.51     4_401.85       1.0000          1.0000        41.37
IVF-TQ-b4-nl158-np17-rf10 (query)                      3_095.34     1_108.08     4_203.42       1.0000          1.0000        41.37
IVF-TQ-b4-nl158-np17-rf20 (query)                      3_095.34     1_472.97     4_568.31       1.0000          1.0000        41.37
IVF-TQ-b4-nl158 (self)                                 3_095.34     2_565.61     5_660.95       0.9999          1.0000        41.37
IVF-TQ-b4-nl223-np11-rf0 (query)                       1_839.08       520.49     2_359.56       0.9096          1.0146        41.97
IVF-TQ-b4-nl223-np14-rf0 (query)                       1_839.08       593.55     2_432.63       0.9097          1.0146        41.97
IVF-TQ-b4-nl223-np21-rf0 (query)                       1_839.08       754.51     2_593.59       0.9097          1.0146        41.97
IVF-TQ-b4-nl223-np11-rf10 (query)                      1_839.08       747.44     2_586.52       0.9998          1.0000        41.97
IVF-TQ-b4-nl223-np11-rf20 (query)                      1_839.08     1_060.23     2_899.30       0.9998          1.0000        41.97
IVF-TQ-b4-nl223-np14-rf10 (query)                      1_839.08       828.74     2_667.82       0.9999          1.0000        41.97
IVF-TQ-b4-nl223-np14-rf20 (query)                      1_839.08     1_152.58     2_991.66       0.9999          1.0000        41.97
IVF-TQ-b4-nl223-np21-rf10 (query)                      1_839.08       997.24     2_836.32       1.0000          1.0000        41.97
IVF-TQ-b4-nl223-np21-rf20 (query)                      1_839.08     1_341.36     3_180.44       1.0000          1.0000        41.97
IVF-TQ-b4-nl223 (self)                                 1_839.08     2_441.98     4_281.06       0.9999          1.0000        41.97
IVF-TQ-b4-nl316-np15-rf0 (query)                       2_172.04       533.15     2_705.19       0.9097          1.0146        42.73
IVF-TQ-b4-nl316-np17-rf0 (query)                       2_172.04       572.86     2_744.90       0.9097          1.0146        42.73
IVF-TQ-b4-nl316-np25-rf0 (query)                       2_172.04       715.74     2_887.78       0.9097          1.0146        42.73
IVF-TQ-b4-nl316-np15-rf10 (query)                      2_172.04       753.70     2_925.74       0.9999          1.0000        42.73
IVF-TQ-b4-nl316-np15-rf20 (query)                      2_172.04     1_064.79     3_236.84       0.9999          1.0000        42.73
IVF-TQ-b4-nl316-np17-rf10 (query)                      2_172.04       803.88     2_975.92       0.9999          1.0000        42.73
IVF-TQ-b4-nl316-np17-rf20 (query)                      2_172.04     1_110.98     3_283.02       0.9999          1.0000        42.73
IVF-TQ-b4-nl316-np25-rf10 (query)                      2_172.04       943.32     3_115.36       1.0000          1.0000        42.73
IVF-TQ-b4-nl316-np25-rf20 (query)                      2_172.04     1_284.64     3_456.68       1.0000          1.0000        42.73
IVF-TQ-b4-nl316 (self)                                 2_172.04     2_368.88     4_540.92       0.9999          1.0000        42.73
-----------------------------------------------------------------------------------------------------------------------------------
</code></pre>
</details>

### Runtime info

*All benchmarks were run on M1 Max MacBook Pro with 64 GB unified memory.*
